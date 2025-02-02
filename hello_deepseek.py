import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from html_formatter import generate_html  # Import the HTML generator module

# Select model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  

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

# Example prompts
prompts = [
    "Tell me a short sci-fi story.",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean."
]

responses = []

# Generate responses
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    responses.append(response_text)

# Send output to HTML formatter
generate_html(prompts, responses)
