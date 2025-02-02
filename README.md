# DeepSeek Workshop

## Introduction
This repository is dedicated to experimenting with DeepSeek models using Hugging Face's `transformers` library. The goal is to explore text generation capabilities, experiment with various prompts, and fine-tune the performance on a local machine.

## Installation
To set up the environment, follow these steps:

```sh
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Model
Ensure that Hugging Face authentication is set up:

```sh
huggingface-cli login
```

Then, run the script:

```sh
python hello_deepseek.py
```

## Challenges Encountered

### 1. Authentication Issues
Initially, we ran into errors when trying to access the DeepSeek models from Hugging Face. These were resolved by generating a token from [Hugging Face Tokens](https://huggingface.co/settings/tokens) and logging in using `huggingface-cli login`.

### 2. Incorrect Model Identifiers
Some model names were incorrect or outdated. We had to verify the available DeepSeek models on [Hugging Face Model Hub](https://huggingface.co/models) and select a valid identifier.

### 3. Quantization Conflicts
Our initial attempts to use quantization resulted in errors due to unsupported `fp8` quantization. To resolve this:
- We explicitly disabled quantization.
- We installed `bitsandbytes` correctly but later realized it wasn’t necessary for our setup.

### 4. Dependency Issues
We faced a conflict between NumPy versions:
- `bitsandbytes` expected an older NumPy version (`<2`), but some dependencies required NumPy 2.x.
- Downgrading to `numpy<2` resolved the issue.

### 5. Memory Errors on macOS (MPS Backend)
Since we were running on a Mac with an Apple Silicon GPU (MPS backend), we encountered `MPS backend out of memory` errors when loading large models. Solutions:
- Reduced model size to `DeepSeek-R1-Distill-Qwen-7B` instead of larger variants.
- Used `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to disable strict memory limits.
- Ensured we were using PyTorch 2.2+ with MPS optimizations.

## Next Steps
- Test performance and inference speed with different batch sizes.
- Explore fine-tuning for domain-specific tasks.
- Compare results with OpenAI models.
