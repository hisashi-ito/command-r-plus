#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【chat_cohere】
#
# 概要:
#      langchain の`ChatModel`を`Cohere`で利用できるようにするためのカスタムモデル
#      推論器側で`stream`が動作しない問題があって解決していないので、langchain からは
#      一旦は非strem 対応のみで実施する
#
# 更新履歴:
#          2024.04.22 新規作成
#      
import os
import enum
import logging
import json

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import (
    Any,
    List,
    Union,
    Dict,
    Mapping,
    Optional,
    Iterable,
    Iterator,
    AsyncIterator,
    Callable,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)

from langchain.schema.output import ChatGenerationChunk
from inference import Inference, InputMaxTokenException


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)

def _default_format_message_as_text(message: BaseMessage) -> str:
    if isinstance(message, ChatMessage):
        message_text = f"\n\n{message.role.capitalize()}: {message.content}"
    elif isinstance(message, HumanMessage):
        message_text = f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{message.content}"
    elif isinstance(message, AIMessage):
        message_text = f"{message.content}<|END_OF_TURN_TOKEN|>"
    elif isinstance(message, SystemMessage):
        message_text = f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{message.content}"
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_text


class ChatCohere(BaseChatModel):
    inference: Inference
    format_message_as_text_func: Callable = Field(
        default_factory=lambda: _default_format_message_as_text
    )

    # 設定ファイルで規定される設定
    max_new_tokens: int = 4096
    min_new_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 40
    num_return_sequences: int = 1
    repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    num_beam_groups: int = 1

    use_cache: bool = False
    do_sample: bool = True
    eos_token_id: int  = 0
    pad_token_id: int = 0

    # Cohere の場合は空文字でいいはず...
    prompt_line_separator: str = ""

    @property
    def _llm_type(self) -> str:
        return "ChatCohere"

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        # messages 配列をpromptに変換
        return self.prompt_line_separator.join(
            [self.format_message_as_text_func(message) for message in messages]
        )
    
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> ChatResult:

        """
        本来は推論器が`stream`になっていることを想定するけど、動作しないので一旦、バッチ推論`__call__`で
        なんとなく実装してごまかして見る、基本的には事前に構築したRAG(langchain) から呼び出して精度を見たいだけ
        """

        prompt = self._format_messages_as_text(messages)
    
        # paramsにparameterを保存
        params = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "temperature": self.temperature,
            "n": self.num_return_sequences,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "num_beam_groups": self.num_beam_groups
        }
        
        # 推論 (batch)
        chunk_gen = self.inference(params, None, prompt=prompt)
        generation_text = ""
        completion_tokens = 0

        for chunk in list(chunk_gen):
            completion_tokens += 1
            d = json.loads(chunk)
            text = d["choices"][0]["delta"]["content"]
            generation_text += text
        
        chat_generation = ChatGeneration(message=AIMessage(content=generation_text))
        return ChatResult(
            generations=[chat_generation],
            llm_output={"completion_tokens": completion_tokens}
        )
