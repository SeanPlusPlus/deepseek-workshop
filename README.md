# 🚀 DeepSeek Workshop

Welcome to the **DeepSeek Workshop**, a collaborative project between **Sean Stephenson** and **ChatGPT**! 🤝 This project is all about exploring **cutting-edge AI models** and fine-tuning them for **practical text generation**.

## 📌 Overview
This project demonstrates how to:
- Load and interact with **DeepSeek AI models** using Hugging Face's `transformers` library.
- Optimize model performance on different hardware (Mac MPS, CPU, etc.).
- Handle potential **out-of-memory (OOM)** issues with best practices.
- **Format model outputs** into a nicely styled **HTML page** for easy review.

---

## 📜 Running the Script


## Running the Project 🚀

Follow these steps to set up and run the project locally:

### 1️⃣ Clone the Repository
```bash
git clone git@github.com:SeanPlusPlus/deepseek-workshop.git
cd deepseek-workshop
```

### 2️⃣ Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip  # Ensure pip is up-to-date
pip install -r requirements.txt
```

### 4️⃣ Run the Script
```bash
python hello_deepseek.py
```

This will:
- Load the **DeepSeek-R1-Distill-Qwen-1.5B** model.
- Run an inference on a sample prompt.
- Generate and save results into a formatted HTML file.

### 5️⃣ View the Output
The output will be saved as an HTML file and automatically opened in your default browser.

If you run into memory issues, try switching to a **smaller model** or using CPU mode by adjusting `device = "cpu"` in the script.

---

## 🔍 **Diving Deeper: Understanding `hello_deepseek.py`**
To **fully grasp how the model works**, we **highly encourage** you to review the **extensive comments** in `hello_deepseek.py`.  

This script **not only runs the model** but also explains:

✅ **What DeepSeek AI is and how it works**  
✅ **Why we use specific optimizations** (e.g., `torch_dtype`, `device_map`, `offload_folder`)  
✅ **How tokenization transforms text into numbers and back**  
✅ **How to adjust model settings to prevent memory issues**  
✅ **Why using `torch.no_grad()` helps save memory during inference**  
✅ **How we structure example prompts to test various AI capabilities**  

Reading through `hello_deepseek.py` will give you **insight into AI model deployment** and help you debug or customize the script for your own use.

---

## 🎨 **Formatting Model Outputs as HTML**
The generated AI responses are automatically saved into an **HTML file** for a clean and shareable output.  

To modify how this works:
- Check out `html_formatter.py`, which takes model outputs and creates a **Bootstrap-styled** web page.
- Feel free to **customize the CSS and layout** as needed!

---

## ⚠️ **Troubleshooting & Performance Tuning**
We hit **a ton of roadblocks** while getting this to work, and we want to make sure you don't have to struggle as much! Here are the major lessons learned:

### **1️⃣ Resolving Model Authentication Issues** 🔑
- We initially hit 401 and 404 errors when trying to load DeepSeek models.
- Solution: We needed to **log in to Hugging Face** and use an **access token**.
  ```bash
  huggingface-cli login
  ```
  Then, visit [Hugging Face Tokens](https://huggingface.co/settings/tokens) to generate an access token.

### **2️⃣ Fixing Out-of-Memory (OOM) Errors** 🚀
- Running larger models caused memory crashes on **Mac MPS (Metal Performance Shaders)**.
- Solutions:
  - **Use a smaller model**: If `"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"` fails, try `"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.3B"`.
  - **Lower `max_new_tokens`** in `model.generate()`.
  - **Offload to CPU**: If necessary, manually set `device = "cpu"`.

### **3️⃣ Bitsandbytes & PyTorch Compatibility** 🏗️
- Installing `bitsandbytes` was tricky due to **NumPy version conflicts**.
- Solution:
  ```bash
  pip install --no-cache-dir --force-reinstall numpy==1.26.4 bitsandbytes torch
  ```
- If using MPS, ensure `torch==2.2.2` is installed!

### **4️⃣ Debugging Torch Device Errors** 🔥
- If `MPS` backend gives errors:
  ```bash
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python hello_deepseek.py
  ```
  This disables memory limits (use cautiously!).

---

## Example Output

<img width="740" alt="image" src="https://github.com/user-attachments/assets/54644eac-660a-4ef7-ac4c-1d39057b644e" />

---

## 🔥 Next Steps
- **Run the script** and see what AI-generated text it produces!  
- **Modify prompts** to test different types of responses.  
- **Experiment with different models** from Hugging Face.  

Let us know if you have any questions or improvements! 🚀💡