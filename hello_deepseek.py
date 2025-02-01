from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Generate text
input_text = "Hello, DeepSeek! Can you complete this sentence:"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs, max_length=50)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
