# Qwen3 Model Implementation

This repository contains a PyTorch implementation of the Qwen3 language model from scratch, including all the necessary components for text generation.

## Overview

This implementation provides:
- **Qwen3Model**: A complete transformer model with Grouped Query Attention (GQA), RoPE positional encoding, and RMS normalization
- **Qwen3Tokenizer**: A wrapper for the Hugging Face tokenizer with chat formatting
- **Text Generation**: Sampling-based text generation with temperature and top-k filtering
- **Multiple Model Sizes**: Support for 0.6B, 1.7B, 4B, 8B, 14B, and 32B parameter models

## Features

- **Grouped Query Attention (GQA)**: Efficient attention mechanism with key-value grouping
- **RoPE (Rotary Position Embedding)**: Advanced positional encoding for better long-context performance  
- **RMS Normalization**: Layer normalization variant used in modern LLMs
- **Chat Formatting**: Proper formatting for conversational interactions
- **Multi-device Support**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU

## Requirements

Install the required dependencies:

```bash
pip install torch safetensors tokenizers huggingface_hub
```

## Getting the Model Files

You need to download two files from the Hugging Face model repository:

### Option 1: Manual Download
1. Go to [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/main)
2. Download the following files to the same directory as `main.py`:
   - `model.safetensors` (model weights)
   - `tokenizer.json` (tokenizer configuration)

### Option 2: Using Python (automatic download)
The code will automatically download the tokenizer if you provide a `repo_id`. For the model weights, you can use:

```python
from huggingface_hub import snapshot_download

# Download all files to current directory
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B-Base", 
    local_dir=".",
    allow_patterns=["model.safetensors", "tokenizer.json"]
)
```

## How to Run

1. **Ensure you have the required files:**
   - `main.py` (this implementation)
   - `model.safetensors` (model weights)
   - `tokenizer.json` (tokenizer)

2. **Run the script:**
   ```bash
   python main.py
   ```

3. **Customize the generation:**
   Edit the variables at the bottom of `main.py`:
   ```python
   USE_REASONING_MODEL = False  # Set to True for reasoning mode
   CHOOSE_MODEL = "0.6B"        # Choose model size
   prompt = "Your custom prompt here"
   ```

## Model Sizes

The implementation supports multiple Qwen3 model sizes:

| Size | Parameters | Embedding Dim | Layers | Heads | Context Length |
|------|------------|---------------|--------|-------|----------------|
| 0.6B | ~600M      | 1024          | 28     | 16    | 40,960         |
| 1.7B | ~1.7B      | 2048          | 28     | 16    | 40,960         |
| 4B   | ~4B        | 2560          | 36     | 32    | 40,960         |
| 8B   | ~8B        | 4096          | 36     | 32    | 40,960         |
| 14B  | ~14B       | 5120          | 40     | 40    | 40,960         |
| 32B  | ~32B       | 5120          | 64     | 64    | 40,960         |

## Generation Parameters

The `generate()` function supports several parameters:

- `max_new_tokens`: Maximum number of tokens to generate (default: 2048)
- `temperature`: Controls randomness (0.0 = deterministic, higher = more random)
- `top_k`: Limits sampling to top-k most likely tokens
- `context_size`: Maximum context window to consider

## Example Usage

```python
# Initialize model
config = get_config("0.6B")
model = Qwen3Model(config)

# Load weights
from safetensors.torch import load_file
weights = load_file("model.safetensors")
load_weights_into_qwen(model, config, weights)

# Initialize tokenizer
tokenizer = Qwen3Tokenizer("tokenizer.json")

# Generate text
prompt = "Explain quantum computing"
input_ids = tokenizer.encode(prompt)
output_ids = generate(model, torch.tensor(input_ids).unsqueeze(0), max_new_tokens=512)
output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
print(output_text)
```

## File Structure

```
.
├── main.py           # Main implementation
├── model.safetensors # Model weights (download required)
├── tokenizer.json    # Tokenizer config (download required)
└── README.md         # This file
```

## Notes

- The model automatically detects and uses the best available device (CUDA > MPS > CPU)
- Generation can be interrupted with Ctrl+C
- The implementation uses bfloat16 precision by default for memory efficiency
- Weight tying is supported (sharing embeddings between input and output layers)