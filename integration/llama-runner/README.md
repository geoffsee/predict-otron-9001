# Llama Runner

A fast Rust implementation for running Llama and other language models using the Candle deep learning framework. Built on the official Candle examples with optimizations for speed and usability.

## Features

- 🚀 **High Performance**: Metal GPU acceleration on macOS, CUDA support on Linux/Windows
- 🤖 **Multiple Models**: Supports Llama 3.2, SmolLM2, TinyLlama, and more
- ⚡ **Fast Inference**: Optimized with F16 precision and KV caching
- 🎯 **Advanced Sampling**: Top-k, top-p, temperature, and repeat penalty controls  
- 📊 **Performance Metrics**: Real-time tokens/second reporting
- 🔧 **Easy CLI**: Simple command-line interface with sensible defaults

## Supported Models

| Model | Size | Command | Description |
|-------|------|---------|-------------|
| SmolLM2-135M | 135M | `smollm2-135m` | Tiny, fast model for testing |
| SmolLM2-360M | 360M | `smollm2-360m` | Small, efficient model |
| SmolLM2-1.7B | 1.7B | `smollm2-1.7b` | Balanced performance/speed |
| Llama-3.2-1B | 1B | `llama-3.2-1b` | Meta's compact model |
| Llama-3.2-3B | 3B | `llama-3.2-3b` | Larger Llama model |
| TinyLlama-1.1B | 1.1B | `tinyllama-1.1b-chat` | Chat-optimized small model |

Add `-instruct` suffix for instruction-tuned variants (e.g., `smollm2-135m-instruct`).

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llama-runner

# Build with GPU acceleration (recommended)
cargo build --release --features metal  # macOS
cargo build --release --features cuda   # Linux/Windows with NVIDIA GPU

# CPU-only build
cargo build --release
```

## Quick Start

```bash
# Fast inference with GPU acceleration
cargo run --features metal -- --prompt "What is quantum computing?"

# Specify a model and parameters
cargo run --features metal -- \
  --prompt "Write a short story about space exploration" \
  --model smollm2-360m \
  --max-tokens 100 \
  --temperature 0.8

# Use CPU (slower but works everywhere)
cargo run -- --prompt "Hello, world!" --model smollm2-135m --cpu
```

## Usage Examples

### Basic Text Generation
```bash
# Simple completion
cargo run --features metal -- --prompt "The capital of France is"

# Creative writing with higher temperature
cargo run --features metal -- \
  --prompt "Once upon a time" \
  --temperature 1.0 \
  --max-tokens 200
```

### Advanced Sampling
```bash
# Top-k and top-p sampling
cargo run --features metal -- \
  --prompt "Explain artificial intelligence" \
  --top-k 40 \
  --top-p 0.9 \
  --temperature 0.7

# Reduce repetition
cargo run --features metal -- \
  --prompt "List the benefits of renewable energy" \
  --repeat-penalty 1.2 \
  --repeat-last-n 64
```

### Different Models
```bash
# Ultra-fast with tiny model
cargo run --features metal -- \
  --prompt "Quick test" \
  --model smollm2-135m

# Better quality with larger model
cargo run --features metal -- \
  --prompt "Explain quantum physics" \
  --model llama-3.2-1b \
  --max-tokens 150
```

## Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompt` | `-p` | "The capital of France is" | Input prompt |
| `--model` | `-m` | `smollm2-135m` | Model to use |
| `--max-tokens` | `-n` | 100 | Maximum tokens to generate |
| `--temperature` | `-t` | 0.8 | Sampling temperature (0.0 = deterministic) |
| `--top-k` | | None | Top-k sampling |
| `--top-p` | | None | Top-p (nucleus) sampling |
| `--seed` | | 299792458 | Random seed for reproducibility |
| `--repeat-penalty` | | 1.1 | Repetition penalty (1.0 = no penalty) |
| `--repeat-last-n` | | 128 | Context window for repeat penalty |
| `--cpu` | | false | Force CPU usage |
| `--dtype` | | f16 | Data type: f16, bf16, f32 |
| `--no-kv-cache` | | false | Disable key-value caching |

## Performance

Typical performance on Apple M2 with Metal acceleration:

| Model | Size | Speed | Memory |
|-------|------|-------|--------|
| SmolLM2-135M | 135M | ~100 tok/s | ~500MB |
| SmolLM2-360M | 360M | ~80 tok/s | ~1GB |
| SmolLM2-1.7B | 1.7B | ~50 tok/s | ~3GB |
| Llama-3.2-1B | 1B | ~40 tok/s | ~2GB |

## Requirements

- **Rust**: 1.70+ (latest stable recommended)
- **Memory**: 2-8GB RAM depending on model size
- **Storage**: 1-10GB for model weights
- **Network**: Internet connection for first-time model download
- **GPU** (optional): Metal on macOS, CUDA on Linux/Windows

## GPU Support

### macOS (Metal)
```bash
cargo run --features metal -- [options]
```

### Linux/Windows (CUDA)
```bash
cargo run --features cuda -- [options]  
```

### CPU Only
```bash
cargo run -- --cpu [options]
```

## Model Downloads

Models are automatically downloaded from HuggingFace Hub on first use and cached locally. Download times:

- SmolLM2-135M: ~1 minute
- SmolLM2-360M: ~2 minutes  
- Llama-3.2-1B: ~5 minutes
- Larger models: 10+ minutes

## Troubleshooting

### Slow Performance
- Use `--features metal` on macOS or `--features cuda` on Linux/Windows
- Try smaller models like `smollm2-135m` for faster inference
- Ensure sufficient RAM for your chosen model

### Out of Memory
- Use `--cpu` to use system RAM instead of GPU memory
- Try smaller models or reduce `--max-tokens`
- Use `--dtype f32` if f16 causes issues

### Model Download Issues
- Check internet connection
- Some models may require HuggingFace Hub authentication
- Verify sufficient disk space in `~/.cache/huggingface/`

## Contributing

Contributions welcome! This project is based on the [Candle](https://github.com/huggingface/candle) framework by HuggingFace.

## License

MIT License - see LICENSE file for details.