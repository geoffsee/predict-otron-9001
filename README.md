<h1 align="center">
    predict-otron-9000
</h1>
<p align="center">
AI inference Server with OpenAI-compatible API (Limited Features)
</p>
<p align="center">
  <img src="https://github.com/geoffsee/predict-otron-9001/blob/master/predict-otron-9000.png?raw=true" width="90%" />
</p>

<br/>
> This project is an educational aide for bootstrapping my understanding of language model inferencing at the lowest levels I can, serving as a "rubber-duck" solution for Kubernetes based performance-oriented inference capabilities on air-gapped networks.

> By isolating application behaviors in components at the crate level, development reduces to a short feedback loop for validation and integration, ultimately smoothing the learning curve for scalable AI systems.
Stability is currently best effort. Many models require unique configuration. When stability is achieved, this project will be promoted to the seemueller-io GitHub organization under a different name.

A comprehensive multi-service AI platform built around local LLM inference, embeddings, and web interfaces.


## Project Overview

The predict-otron-9000 is a flexible AI platform that provides:

- **Local LLM Inference**: Run Gemma and Llama models locally with CPU or GPU acceleration
- **Embeddings Generation**: Create text embeddings with FastEmbed
- **Web Interface**: Interact with models through a Leptos WASM chat interface
- **TypeScript CLI**: Command-line client for testing and automation
- **Production Deployment**: Docker and Kubernetes deployment options

The system supports both CPU and GPU acceleration (CUDA/Metal), with intelligent fallbacks and platform-specific optimizations.

## Features

- **OpenAI Compatible**: API endpoints match OpenAI's format for easy integration
- **Text Embeddings**: Generate high-quality text embeddings using FastEmbed
- **Text Generation**: Chat completions with OpenAI-compatible API using Gemma and Llama models (various sizes including instruction-tuned variants)
- **Performance Optimized**: Efficient caching and platform-specific optimizations for improved throughput
- **Web Chat Interface**: Leptos chat interface
- **Flexible Deployment**: Run as monolithic service or microservices architecture

## Architecture Overview

### Workspace Structure

The project uses a 7-crate Rust workspace plus TypeScript components:

```
crates/
├── predict-otron-9000/     # Main orchestration server (Rust 2024)
├── inference-engine/       # Multi-model inference orchestrator (Rust 2021)
├── gemma-runner/          # Gemma model inference via Candle (Rust 2021)
├── llama-runner/          # Llama model inference via Candle (Rust 2021)
├── embeddings-engine/     # FastEmbed embeddings service (Rust 2024)
├── leptos-app/            # WASM web frontend (Rust 2021)
├── helm-chart-tool/       # Kubernetes deployment tooling (Rust 2024)
└── scripts/
    └── cli.ts             # TypeScript/Bun CLI client
```

### Service Architecture

- **Main Server** (port 8080): Orchestrates inference and embeddings services
- **Embeddings Service** (port 8080): Standalone FastEmbed service with OpenAI API compatibility  
- **Web Frontend** (port 8788): cargo leptos SSR app
- **CLI Client**: TypeScript/Bun client for testing and automation

### Deployment Modes

The architecture supports multiple deployment patterns:

1. **Development Mode**: All services run in a single process for simplified development
2. **Docker Monolithic**: Single containerized service handling all functionality
3. **Kubernetes Microservices**: Separate services for horizontal scalability and fault isolation

## Build and Configuration

### Dependencies and Environment Prerequisites

#### Rust Toolchain
- **Editions**: Mixed - main services use Rust 2024, some components use 2021
- **Recommended**: Latest stable Rust toolchain: `rustup default stable && rustup update`
- **Developer tools**:
  - `rustup component add rustfmt` (formatting)
  - `rustup component add clippy` (linting)

#### Node.js/Bun Toolchain  
- **Bun**: Required for TypeScript CLI client: `curl -fsSL https://bun.sh/install | bash`
- **Node.js**: Alternative to Bun, supports OpenAI SDK v5.16.0+

#### ML Framework Dependencies
- **Candle**: Version 0.9.1 with conditional compilation:
  - macOS: Metal support with CPU fallback for stability
  - Linux: CUDA support with CPU fallback
  - CPU-only: Supported on all platforms
- **FastEmbed**: Version 4.x for embeddings functionality

#### Hugging Face Access
- **Required for**: Gemma model downloads (gated models)
- **Authentication**: 
  - CLI: `pip install -U "huggingface_hub[cli]" && huggingface-cli login`
  - Environment: `export HF_TOKEN="<your_token>"`
- **Cache management**: `export HF_HOME="$PWD/.hf-cache"` (optional, keeps cache local)
- **Model access**: Accept Gemma model licenses on Hugging Face before use

#### Platform-Specific Notes
- **macOS**: Metal acceleration available but routed to CPU for Gemma v3 stability
- **Linux**: CUDA support with BF16 precision on GPU, F32 on CPU  
- **Conditional compilation**: Handled automatically per platform in Cargo.toml

### Build Procedures

#### Full Workspace Build
```bash
cargo build --workspace --release
```

#### Individual Services

**Main Server:**
```bash
cargo build --bin predict-otron-9000 --release
```

**Inference Engine CLI:**
```bash  
cargo build --bin cli --package inference-engine --release
```

**Embeddings Service:**
```bash
cargo build --bin embeddings-engine --release
```


### Running Services

#### Main Server (Port 8080)
```bash
./scripts/run_server.sh
```
- Respects `SERVER_PORT` (default: 8080) and `RUST_LOG` (default: info)
- Boots with default model: `gemma-3-1b-it`
- Requires HF authentication for first-time model download

#### Web Frontend (Port 8788)  
```bash
cd crates/leptos-app
./run.sh
```
- Serves Leptos WASM frontend on port 8788
- Sets required RUSTFLAGS for WebAssembly getrandom support
- Auto-reloads during development

#### TypeScript CLI Client
```bash
# List available models
bun run scripts/cli.ts --list-models

# Chat completion
bun run scripts/cli.ts "What is the capital of France?"

# With specific model
bun run scripts/cli.ts --model gemma-3-1b-it --prompt "Hello, world!"

# Show help
bun run scripts/cli.ts --help
```

## API Usage

### Health Checks and Model Inventory
```bash
curl -s http://localhost:8080/v1/models | jq
```

### Chat Completions

**Non-streaming:**
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 64
  }' | jq
```

**Streaming (Server-Sent Events):**
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default", 
    "messages": [{"role": "user", "content": "Tell a short joke"}],
    "stream": true,
    "max_tokens": 64
  }'
```

**Model Specification:**
- Use `"model": "default"` for configured model
- Or specify exact model ID: `"model": "gemma-3-1b-it"`
- Requests with unknown models will be rejected

### Embeddings API

Generate text embeddings compatible with OpenAI's embeddings API.

**Endpoint**: `POST /v1/embeddings`

**Request Body**:
```json
{
  "input": "Your text to embed",
  "model": "nomic-embed-text-v1.5"
}
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3]
    }
  ],
  "model": "nomic-embed-text-v1.5",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

### Web Frontend
- Navigate to `http://localhost:8788` 
- Real-time chat interface with the inference server
- Supports streaming responses and conversation history

## Testing

### Test Categories

1. **Offline/fast tests**: No network or model downloads required
2. **Online tests**: Require HF authentication and model downloads  
3. **Integration tests**: Multi-service end-to-end testing

### Quick Start: Offline Tests

**Prompt formatting tests:**
```bash
cargo test --workspace build_gemma_prompt
```

**Model metadata tests:**
```bash  
cargo test --workspace which_
```

These verify core functionality without requiring HF access.

### Full Test Suite (Requires HF)

**Prerequisites:**
1. Accept Gemma model licenses on Hugging Face
2. Authenticate: `huggingface-cli login` or `export HF_TOKEN=...`
3. Optional: `export HF_HOME="$PWD/.hf-cache"`

**Run all tests:**
```bash
cargo test --workspace
```

### Integration Testing

**End-to-end test script:**
```bash
./smoke_test.sh
```

This script:
- Starts the server in background with proper cleanup
- Waits for server readiness via health checks  
- Runs CLI tests for model listing and chat completion
- Includes 60-second timeout and process management

## Development

### Code Style and Tooling

**Formatting:**
```bash
cargo fmt --all
```

**Linting:**
```bash  
cargo clippy --workspace --all-targets -- -D warnings
```

**Logging:**
- Server uses `tracing` framework
- Control via `RUST_LOG` (e.g., `RUST_LOG=debug ./scripts/run_server.sh`)

### Adding Tests

**For fast, offline tests:**
- Exercise pure logic without tokenizers/models
- Use descriptive names for easy filtering: `cargo test specific_test_name`
- Example patterns: prompt construction, metadata selection, tensor math

**Process:**
1. Add test to existing module
2. Run filtered: `cargo test --workspace new_test_name` 
3. Verify in full suite: `cargo test --workspace`

### OpenAI API Compatibility

**Features:**
- POST `/v1/chat/completions` with streaming and non-streaming
- Single configured model enforcement (use `"model": "default"`)
- Gemma-style prompt formatting with `<start_of_turn>`/`<end_of_turn>` markers
- System prompt injection into first user turn
- Repetition detection and early stopping in streaming mode

**CORS:**
- Fully open by default (`tower-http CorsLayer::Any`)
- Adjust for production deployment

### Architecture Details

**Device Selection:**
- Automatic device/dtype selection
- CPU: Universal fallback (F32 precision)
- CUDA: BF16 precision on compatible GPUs  
- Metal: Available but routed to CPU for Gemma v3 stability

**Model Loading:**
- Single-file `model.safetensors` preferred
- Falls back to index resolution via `utilities_lib::hub_load_safetensors`
- HF cache populated on first access

**Multi-Service Design:**
- Main server orchestrates inference and embeddings
- Services can run independently for horizontal scaling
- Docker/Kubernetes metadata included for deployment

## Deployment

### Docker Support

All services include Docker metadata in `Cargo.toml`:

**Main Server:**
- Image: `ghcr.io/geoffsee/predict-otron-9000:latest`
- Port: 8080

**Inference Service:**
- Image: `ghcr.io/geoffsee/inference-service:latest`  
- Port: 8080

**Embeddings Service:**
- Image: `ghcr.io/geoffsee/embeddings-service:latest`
- Port: 8080

**Web Frontend:**
- Image: `ghcr.io/geoffsee/leptos-app:latest`
- Port: 8788

**Docker Compose:**
```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes Support

All services include Kubernetes manifest metadata:
- Single replica deployments by default
- Service-specific port configurations
- Ready for horizontal pod autoscaling

For Kubernetes deployment details, see the [ARCHITECTURE.md](docs/ARCHITECTURE.md) document.

### Build Artifacts

**Ignored by Git:**
- `target/` (Rust build artifacts)
- `node_modules/` (Node.js dependencies)  
- `dist/` (Frontend build output)
- `.fastembed_cache/` (FastEmbed model cache)
- `.hf-cache/` (Hugging Face cache, if configured)

## Common Issues and Solutions

### Authentication/Licensing
**Symptom:** 404 or permission errors fetching models  
**Solution:** 
1. Accept Gemma model licenses on Hugging Face
2. Authenticate with `huggingface-cli login` or `HF_TOKEN`
3. Verify token with `huggingface-cli whoami`

### GPU Issues  
**Symptom:** OOM errors or GPU panics  
**Solution:**
1. Test on CPU first: ensure `CUDA_VISIBLE_DEVICES=""` if needed
2. Check available VRAM vs model requirements
3. Consider using smaller model variants

### Model Mismatch Errors
**Symptom:** 400 errors with `type=model_mismatch`  
**Solution:**
- Use `"model": "default"` in API requests
- Or match configured model ID exactly: `"model": "gemma-3-1b-it"`

### Frontend Build Issues
**Symptom:** WASM compilation failures  
**Solution:**
1. Install required targets: `rustup target add wasm32-unknown-unknown`
2. Check RUSTFLAGS in leptos-app/run.sh

### Network/Timeout Issues
**Symptom:** First-time model downloads timing out  
**Solution:**
1. Ensure stable internet connection
2. Consider using local HF cache: `export HF_HOME="$PWD/.hf-cache"`
3. Download models manually with `huggingface-cli`

## Minimal End-to-End Verification

**Build verification:**
```bash
cargo build --workspace --release
```

**Fast offline tests:**
```bash
cargo test --workspace build_gemma_prompt
cargo test --workspace which_
```

**Service startup:**
```bash
./scripts/run_server.sh &
sleep 10  # Wait for server startup
curl -s http://localhost:8080/v1/models | jq
```

**CLI client test:**
```bash
bun run scripts/cli.ts "What is 2+2?"
```

**Web frontend:**
```bash
cd crates/leptos-app && ./run.sh &
# Navigate to http://localhost:8788
```

**Integration test:**
```bash
./smoke_test.sh
```

**Cleanup:**
```bash
pkill -f "predict-otron-9000"
```

For networked tests and full functionality, ensure Hugging Face authentication is configured as described above.

## Further Reading

### Documentation

- [Architecture](docs/ARCHITECTURE.md) - Detailed architectural diagrams and deployment patterns
- [Server Configuration Guide](docs/SERVER_CONFIG.md) - Detailed server configuration options
- [Testing Documentation](docs/TESTING.md) - Comprehensive testing guide
- [Performance Benchmarking](docs/BENCHMARKING.md) - Instructions for benchmarking

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `cargo test`
5. Submit a pull request

_Warning: Do NOT use this in production unless you are cool like that._
