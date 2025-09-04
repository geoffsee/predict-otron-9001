# Gemma Runner

Fast Gemma inference with Candle framework in Rust.

## Features

- Support for multiple Gemma model versions (v1, v2, v3)
- GPU acceleration with CUDA and Metal
- Configurable sampling parameters
- Multiple model variants including instruct and code models

## Supported Models

### Gemma v1
- `gemma-2b` - Base 2B model
- `gemma-7b` - Base 7B model  
- `gemma-2b-it` - Instruct 2B model
- `gemma-7b-it` - Instruct 7B model
- `gemma-1.1-2b-it` - Instruct 2B v1.1 model
- `gemma-1.1-7b-it` - Instruct 7B v1.1 model

### CodeGemma
- `codegemma-2b` - Code base 2B model
- `codegemma-7b` - Code base 7B model
- `codegemma-2b-it` - Code instruct 2B model
- `codegemma-7b-it` - Code instruct 7B model

### Gemma v2
- `gemma-2-2b` - Base 2B v2 model (default)
- `gemma-2-2b-it` - Instruct 2B v2 model
- `gemma-2-9b` - Base 9B v2 model
- `gemma-2-9b-it` - Instruct 9B v2 model

### Gemma v3
- `gemma-3-1b` - Base 1B v3 model
- `gemma-3-1b-it` - Instruct 1B v3 model

## Installation

```bash
cd gemma-runner
cargo build --release
```

For GPU support:
```bash
# CUDA
cargo build --release --features cuda

# Metal (macOS)
cargo build --release --features metal
```

## Usage

### Basic Usage

```bash
# Run with default model (gemma-2-2b)
cargo run -- --prompt "The capital of France is"

# Specify a different model
cargo run -- --model gemma-2b-it --prompt "Explain quantum computing"

# Generate more tokens
cargo run -- --model codegemma-2b-it --prompt "Write a Python function to sort a list" --max-tokens 200
```

### Advanced Options

```bash
# Use CPU instead of GPU
cargo run -- --cpu --prompt "Hello world"

# Adjust sampling parameters
cargo run -- --temperature 0.8 --top-p 0.9 --prompt "Write a story about"

# Use custom model from HuggingFace Hub
cargo run -- --model-id "google/gemma-2-2b-it" --prompt "What is AI?"

# Enable tracing for performance analysis
cargo run -- --tracing --prompt "Explain machine learning"
```

### Command Line Arguments

- `--prompt, -p` - The prompt to generate text from (default: "The capital of France is")
- `--model, -m` - The model to use (default: "gemma-2-2b")
- `--cpu` - Run on CPU rather than GPU
- `--temperature, -t` - Sampling temperature (optional)
- `--top-p` - Nucleus sampling probability cutoff (optional)
- `--seed` - Random seed (default: 299792458)
- `--max-tokens, -n` - Maximum tokens to generate (default: 100)
- `--model-id` - Custom model ID from HuggingFace Hub
- `--revision` - Model revision (default: "main")
- `--use-flash-attn` - Use flash attention
- `--repeat-penalty` - Repetition penalty (default: 1.1)
- `--repeat-last-n` - Context size for repeat penalty (default: 64)
- `--dtype` - Data type (f16, bf16, f32)
- `--tracing` - Enable performance tracing

## Examples

### Text Generation
```bash
cargo run -- --model gemma-2b-it --prompt "Explain the theory of relativity" --max-tokens 150
```

### Code Generation
```bash
cargo run -- --model codegemma-7b-it --prompt "Write a Rust function to calculate factorial" --max-tokens 100
```

### Creative Writing
```bash
cargo run -- --model gemma-7b-it --temperature 0.9 --prompt "Once upon a time in a magical forest" --max-tokens 200
```

### Chat with Gemma 3 (Instruct format)
```bash
cargo run -- --model gemma-3-1b-it --prompt "How do I learn Rust programming?"
```

## Performance Notes

- GPU acceleration is automatically detected and used when available
- BF16 precision is used on CUDA for better performance
- F32 precision is used on CPU
- Flash attention can be enabled with `--use-flash-attn` for supported models
- Model files are cached locally after first download

## Requirements

- Rust 1.70+
- CUDA toolkit (for CUDA support)
- Metal (automatically available on macOS)
- Internet connection for first-time model download