import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from html_formatter import generate_html  # Import the module that formats output as an HTML page

# -------------------------------------------------------------------
# 🧠 DeepSeek AI Model: Overview
# -------------------------------------------------------------------
# This script loads and runs the DeepSeek-R1-Distill-Qwen-1.5B model, 
# a distilled version of the larger DeepSeek AI model family.
#
# - The model is a **causal language model** (CLM), meaning it generates 
#   text by predicting the next token in a sequence based on previous tokens.
#
# - It is designed for **open-ended text generation**, meaning it can be 
#   used for tasks like storytelling, answering questions, or summarization.
#
# - We use the Hugging Face `transformers` library to handle the model and
#   its tokenizer efficiently.
# -------------------------------------------------------------------

# 🏆 Select a DeepSeek AI model
# If you run into memory issues (OOM = "Out of Memory"), try a **smaller** model.
# You can find available models at: https://huggingface.co/deepseek-ai
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  

# 🔥 Select a compute device
# This script will automatically detect whether you have an MPS (Mac GPU) 
# or a CPU, and adjust accordingly.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# -------------------------------------------------------------------
# 🔠 Load Tokenizer
# -------------------------------------------------------------------
# The tokenizer is responsible for:
# 1. Converting text into numerical tokens that the model can understand.
# 2. Converting the model's numerical outputs back into readable text.
#
# We enable `trust_remote_code=True` to allow custom model architectures
# to be loaded properly.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# -------------------------------------------------------------------
# 🏗️ Load Model with Optimized Settings
# -------------------------------------------------------------------
# Since this is a large model, we optimize its loading using:
# - `torch_dtype`: Sets the precision of the model's weights (float16 for efficiency).
# - `device_map="auto"`: Automatically distributes the model across available hardware.
# - `offload_folder`: Used for offloading to disk (if necessary) to reduce memory usage.
#
# If you are running into memory issues, try:
# - Reducing `max_new_tokens` in the generation step.
# - Using a smaller model (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.3B").
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,  # Use FP16 for efficiency
    device_map="auto" if device != "cpu" else None,  # Distribute model efficiently
    offload_folder="offload" if device != "cpu" else None,  # Disk offloading for large models
)

# Move model to the selected device
model.to(device)

# -------------------------------------------------------------------
# 🎭 Define Example Prompts
# -------------------------------------------------------------------
# We define three different prompts to test various types of model responses.
# These examples showcase storytelling, technical explanation, and poetry.
prompts = [
    "Tell me a short sci-fi story about a robot discovering emotions.",
    "Explain quantum computing to a 10-year-old.",
    "Write a haiku about the ocean."
]

responses = []  # Store model responses

# -------------------------------------------------------------------
# ✍️ Generate Responses
# -------------------------------------------------------------------
# The model will generate text based on each input prompt.
# We use `model.generate()` to produce responses up to 100 new tokens.
for prompt in prompts:
    print(f"\n📝 Generating response for: {prompt}")

    # Convert prompt to numerical tokens and move to the correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Perform inference (disable gradient calculation to save memory)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    # Convert model output back to human-readable text
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Store the generated response
    responses.append(response_text)

    # Print the response in the terminal
    print(f"💬 Response: {response_text}\n")

# -------------------------------------------------------------------
# 🖥️ Format & Save Output as an HTML Page
# -------------------------------------------------------------------
# We use `html_formatter.py` to create a nicely styled Bootstrap HTML page.
# This allows us to view and share the model's responses easily.
generate_html(prompts, responses)

# 🚀 Script complete! Open `output.html` in your browser to see the results.
