from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer
model_name = "deepseek-ai/deepseek-coder-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model in CPU mode with float32 precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Adjust precision for CPU
    device_map="cpu"  # Explicitly force CPU mode
)

# Test generation
input_text = "Hello, DeepSeek! Can you complete this sentence:"
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")  # Force CPU

with torch.no_grad():
    output = model.generate(**inputs, max_length=50)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
