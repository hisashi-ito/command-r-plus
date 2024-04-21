from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-plus-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)

""" model info
CohereForCausalLM(
  (model): CohereModel(
    (embed_tokens): Embedding(256000, 12288, padding_idx=0)
    (layers): ModuleList(
      (0-63): 64 x CohereDecoderLayer(
        (self_attn): CohereAttention(
          (q_norm): CohereLayerNorm()
          (k_norm): CohereLayerNorm()
          (q_proj): Linear4bit(in_features=12288, out_features=12288, bias=False)
          (k_proj): Linear4bit(in_features=12288, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=12288, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=12288, out_features=12288, bias=False)
          (rotary_emb): CohereRotaryEmbedding()
        )
        (mlp): CohereMLP(
          (gate_proj): Linear4bit(in_features=12288, out_features=33792, bias=False)
          (up_proj): Linear4bit(in_features=12288, out_features=33792, bias=False)
          (down_proj): Linear4bit(in_features=33792, out_features=12288, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): CohereLayerNorm()
      )
    )
    (norm): CohereLayerNorm()
  )
  (lm_head): Linear(in_features=12288, out_features=256000, bias=False)
)
"""

model = AutoModelForCausalLM.from_pretrained(model_id)

# Format message with the command-r-plus chat template
messages = [{"role": "user", "content": "京都アニメーションの映画でお勧めを３つ教えてください"}] 
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>京都アニメーションの映画でお勧めを３つ教えてください<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=1024, 
    do_sample=True, 
    temperature=0.7,
)

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
