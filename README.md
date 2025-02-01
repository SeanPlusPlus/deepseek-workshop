# DeepSeek Workshop

## Overview
This repository is designed to help run the lightweight, laptop-friendly DeepSeek next-token predictor model locally. The goal is to get the latest small DeepSeek model up and running with a simple "Hello World" example.

## Setup
### 1. Install Python 3.12
Ensure you have Python 3.12 installed:
```sh
python --version
```
If not installed, use Homebrew:
```sh
brew install python@3.12
```

### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Model
### 1. Run the "Hello World" Script
Execute the script to load the model and generate text:
```sh
python hello_deepseek.py
```

### 2. Expected Output
Once the model is loaded, it should generate text based on the given prompt.

## Optimizations
- **Use a smaller model:** If memory is an issue, switch to `deepseek-ai/deepseek-coder-1.3b`.
- **Reduce memory usage:** If running on CPU, modify the script to use `torch_dtype=torch.float32`.
- **Use quantization:** Enable `BitsAndBytesConfig` for 8-bit or 4-bit model loading.

## Troubleshooting
- If the model download is slow, check your internet connection or use a wired connection.
- If `bitsandbytes` fails, ensure it is installed correctly:
  ```sh
  python -m pip install -U bitsandbytes accelerate
  ```
- If `torch` is missing, install it explicitly:
  ```sh
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

## Notes
- Model files are cached in `~/.cache/huggingface/`.
- The `.gitignore` includes virtual environments, build artifacts, and Hugging Face cache.

## License
This repository is open-source under the MIT License.

## Acknowledgments
- Built using [Hugging Face Transformers](https://huggingface.co/docs/transformers/).
- DeepSeek models provided by [DeepSeek AI](https://huggingface.co/deepseek-ai).

