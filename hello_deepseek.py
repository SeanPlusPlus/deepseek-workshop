import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import webbrowser
import os

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define example prompts
prompts = [
    "Tell me a short sci-fi story about space exploration.",
    "Explain the concept of quantum entanglement in simple terms.",
    "Write a motivational message for someone learning to code."
]

# Generate responses
responses = []
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    responses.append((prompt, response_text))

# Generate HTML content
html_content = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>DeepSeek AI Responses</title>
    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
</head>
<body class='container mt-5'>
    <h1 class='mb-4'>DeepSeek AI Model Responses</h1>
    {''.join(f"""
        <div class='card mb-3'>
            <div class='card-header'><strong>Prompt:</strong> {prompt}</div>
            <div class='card-body'><p class='card-text'>{response}</p></div>
        </div>
    """ for prompt, response in responses)}
</body>
</html>
"""

# Save and open in browser
output_file = "deepseek_responses.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

webbrowser.open(f"file://{os.path.abspath(output_file)}")

print("✅ Responses saved and opened in browser!")
