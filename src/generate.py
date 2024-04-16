from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-plus-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Format message with the command-r-plus chat template
# messages = [{"role": "user", "content": "Hello, how are you?"}]
messages = [{"role": "user", "content": "土用の丑の日の提唱者を教えてください"}] 
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>土用の丑の日の提唱者を教えてください<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=1024, 
    do_sample=True, 
    temperature=0.7,
)

gen_text = tokenizer.decode(gen_tokens[0])
# <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>土用の丑の日の提唱者を教えてください<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>平賀源内<|END_OF_TURN_TOKEN|>
print(gen_text)
