import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Select model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Try a smaller model if OOM persists

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model with optimized settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map="auto" if device != "cpu" else None,
    offload_folder="offload" if device != "cpu" else None,
)
model.to(device)

# Run a test inference
prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

# Decode and print output
print(tokenizer.decode(output[0], skip_special_tokens=True))
