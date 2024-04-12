from mlx_lm import load, generate

model, tokenizer = load("mlx-community/c4ai-command-r-plus-4bit")

# Format message with the command-r tool use template
conversation = [
    {"role": "user", "content": "Whats the biggest penguin in the world?"}
]
# Define tools available for the model to use:
tools = [
  {
    "name": "internet_search",
    "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
    "parameter_definitions": {
      "query": {
        "description": "Query to search the internet with",
        "type": 'str',
        "required": True
      }
    }
  },
  {
    'name': "directly_answer",
    "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
    'parameter_definitions': {}
  }
]

formatted_input = tokenizer.apply_tool_use_template(conversation, tools=tools, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=formatted_input, verbose=True)
