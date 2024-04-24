#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 【inference】
#
#  概要: Cohere のlocal model 推論コード
#        messages 配列と生のpromptの両方で推論できるようにする
#
#  更新履歴:
#            2024.04.21 新規作成
#
import os
import asyncio
import enum
import logging
import random
import string
import time
import warnings
import threading
import json
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from typing import AsyncIterator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)

def cln_params(dic):
    ret = {}
    for k, v in dic.items():
        if isinstance(v, str):
            if v.lower() == 'none':
                v = None
            elif v.lower() == "true":
                v = True
            elif v.lower() == "false":
                v = False
        ret[k] = v

    return ret

def get_params(dic, key, default):
    if key not in dic:
        return default
    else:
        if dic[key] is None:
            return default
        else:
            return dic[key]

class InputMaxTokenException(Exception):
    pass    


class Inference(object):
    def __init__(self, params):
        # params は初期パラメータ(dict)
        self.params = params
        
        # Load tokenizer and model
        try:
            model_id = "CohereForAI/c4ai-command-r-plus-4bit"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.eval()
        except Exception as e:
            raise RuntimeError("Unable to load tokenizer or model.") from e

        # Autocast
        self.autocast_context = nullcontext()

    def gen_prompt(self, messages):
        # message 配列から`prompt`を作成する
        prompt = ""
        last_role = None
        
        for msg in messages:
            last_role = role = msg["role"]
            content = msg["content"]           
            if role == "user" or role == "system":
                prompt += f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            else:
                prompt += f"{content}<|END_OF_TURN_TOKEN|>"
        return prompt, last_role
    
    def __call__(self, params, messages=None, prompt=None):
        temp = get_params(params, "temperature", self.params["temperature"])
        generate_kwargs = {
            'max_new_tokens': get_params(params, "max_new_tokens", self.params["max_new_tokens"]),
            'min_new_tokens': get_params(params, "min_new_tokens", self.params["min_new_tokens"]),
            'temperature': temp,
            'top_p': get_params(params, "top_p", self.params["top_p"]),
            'top_k': get_params(params, "top_k", self.params["top_k"]),
            'repetition_penalty': get_params(params, "repetition_penalty", self.params["repetition_penalty"]),
            'no_repeat_ngram_size': get_params(params, "no_repeat_ngram_size", self.params["no_repeat_ngram_size"]),
            'num_beams': get_params(params, "num_beams", self.params["num_beams"]),
            'num_beam_groups': get_params(params, "num_beam_groups", self.params["num_beam_groups"]),
            'num_return_sequences': get_params(params, "num_return_sequences", self.params["num_return_sequences"]),
            'use_cache': self.params["use_cache"],
            'do_sample': False if temp == 0 else self.params["do_sample"],
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        
        # Generate function with correct context managers
        def _generate(encoded_inp_: Dict[str, torch.Tensor]):
            with torch.no_grad():
                with self.autocast_context:
                    return self.model.generate(
                        input_ids=encoded_inp_['input_ids'],
                        attention_mask=encoded_inp_['attention_mask'],
                        **generate_kwargs,
                    )

        # messages 配列から`prompt`を作成する
        if prompt == None:
            prompt, last_role = self.gen_prompt(messages)
        else:
            # 引数に`prompt`があればそのまま利用する
            # その場合`last_role`は`assistant`で固定
            last_role = 'assistant'

        # Split into prompt batches
        # バッチサイズは実質的には常に1 となる
        batch = [prompt]
        
        encoded_inp = self.tokenizer(batch, return_tensors='pt', padding=True)
        encoded_inp.to("cuda")
        if len(encoded_inp[0]) >= self.params["input_max_tokens"]:
            input_max_tokens = int(self.params["input_max_tokens"])
            raise InputMaxTokenException(
                f"入力トークンが最大値を超えました: {int(len(encoded_inp[0]))}, 最大値: {input_max_tokens}")

        # 入力のトークン数(batch)
        input_tokens = torch.sum(
            encoded_inp['input_ids'] != self.tokenizer.pad_token_id,
            axis=1).numpy(force=True)  # type: ignore

        # Run HF generate
        logger.info('Generating responses... (batch)')
        encoded_gen = _generate(encoded_inp)
        decoded_gen = self.tokenizer.batch_decode(encoded_gen, skip_special_tokens=True)
        
        gen_tokens = torch.sum(encoded_gen != self.tokenizer.pad_token_id, axis=1).numpy(force=True)  # type: ignore
        effective_prompts = self.tokenizer.batch_decode(encoded_inp['input_ids'], skip_special_tokens=True)

        logger.info(gen_tokens)
        
        # 生成結果を複数生成
        choices = []
        for i, (dgen, egen) in enumerate(zip(decoded_gen, encoded_gen)):
            # 返却文字列は入力トークンを含まず出力だけを考慮
            # 終了条件も同時に調査、終了条件は`stop`と`length`のみを判定
            finish_reason = None
            if (gen_tokens[i] - input_tokens[0]) == generate_kwargs["max_new_tokens"]:
                finish_reason = "length"
            else:
                egen = egen.to('cpu').detach().numpy().copy()
                egen = egen[egen != self.tokenizer.pad_token_id]
                if egen[-1] == self.tokenizer.eos_token_id:
                    finish_reason = "stop"

            continuation = dgen[len(effective_prompts[0]):]

            if last_role == "user" or last_role == "system":
                disp_role = "assistant"
            else:
                disp_role = "user"
                
            choices.append(
                {
                    "finish_reason": finish_reason,
                    "index": i,
                    "message": {
                        "content": continuation,
                        "role": disp_role
                    }
                }
            )
            
        # numpy.int64 はjson.dumpsで失敗するのでintに変換しておく
        ret = {
            "choices": choices,
            "created": int(time.time()),
            "object": "chat.completion",
            "usage": {
                "completion_tokens": int(np.sum(gen_tokens) - np.sum(input_tokens)),
                "prompt_tokens": int(np.sum(input_tokens)),
                "total_tokens": int(np.sum(gen_tokens))
            }
        }

        return ret


if __name__ == '__main__':
    import yaml
    with open("cohere_inference.yaml", mode="r", encoding="utf-8") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)["inference"]

    params = cln_params(params)
    
    """        
    params = {
        "input_max_tokens": 4096,
        "max_new_tokens": 512,
        "min_new_tokens": 1,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "diversity_penalty": None,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_beam_groups": 1,
        "seed": 42,
        "do_sample": True,
        "use_cache": True,
        "autocast_dtype": None,
        "device": "cuda:0",
        "attn_impl": "torch",
        "use_fast": False,
        "renormalize_logits": False,
        "num_return_sequences": 1,
        "bad_words": []
    }
    """
    hf = Inference(params)

    print("*** messages ***")
    messages = [{"role": "user", "content": "京都アニメーションの映画でお勧めを３つ教えてください"}] 
    print(hf({}, messages=messages))

    print("*** prompt ***")
    prompt = "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>京都アニメーションの映画でお勧めを３つ教えてください<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    print(hf({}, prompt=prompt))
    
    # stream が動作しないなぜ？
    # messages = [{"role": "user", "content": "京都アニメーションの映画でお勧めを３つ教えてください"}]                 
    # chunk_gen = hf(params, messages=messages)
    # print(chunk_gen)
    # for x in chunk_gen:
    #    print(x)
