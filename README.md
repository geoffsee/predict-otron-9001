# predict-otron-9000

_Warning: Do NOT use this in production unless you are cool like that._

<p align="center">
Aliens, in a native executable.
</p>


## Features
- **OpenAI Compatible**: API endpoints match OpenAI's format for easy integration
- **Text Embeddings**: Generate high-quality text embeddings using the Nomic Embed Text v1.5 model
- **Text Generation**: Chat completions with OpenAI-compatible API using Gemma models (1B, 2B, 7B, 9B variants including base and instruction-tuned models)
- **Performance Optimized**: Implements efficient caching and singleton patterns for improved throughput and reduced latency
- **Performance Benchmarking**: Includes tools for measuring performance and generating HTML reports
- **Web Chat Interface**: A Leptos-based WebAssembly (WASM) chat interface for browser-based interaction with the inference engine

## Architecture

### Core Components

- **`predict-otron-9000`**: Main unified server that combines both engines
- **`embeddings-engine`**: Handles text embeddings using FastEmbed with the Nomic Embed Text v1.5 model
- **`inference-engine`**: Provides text generation capabilities using Gemma models (1B, 2B, 7B, 9B variants) via Candle transformers
- **`leptos-app`**: WebAssembly-based chat interface built with Leptos framework for browser-based interaction with the inference engine

## Further Reading

### Documentation

- [Architecture](docs/ARCHITECTURE.md) - Detailed server configuration options and deployment modes
- [Server Configuration Guide](docs/SERVER_CONFIG.md) - Detailed server configuration options and deployment modes
- [Testing Documentation](docs/TESTING.md) - Comprehensive testing guide including unit, integration and e2e tests
- [Performance Benchmarking](docs/BENCHMARKING.md) - Instructions for running and analyzing performance benchmarks

## Installation

### Prerequisites

- Rust 1.70+ with 2024 edition support
- Cargo package manager

### Build from Source
```shell
# 1. Clone the repository
git clone <repository-url>
cd predict-otron-9000

# 2. Build the project
cargo build --release

# 3. Run the unified server
./run_server.sh

# Alternative: Build and run individual components
# For inference engine only:
cargo run -p inference-engine --release -- --server --port 3777
# For embeddings engine only:
cargo run -p embeddings-engine --release
```

## Usage

### Starting the Server

The server can be started using the provided script or directly with cargo:

```shell
# Using the provided script
./run_server.sh

# Or directly with cargo
cargo run --bin predict-otron-9000
```

### Configuration

Environment variables for server configuration:

- `SERVER_HOST`: Server bind address (default: `0.0.0.0`)
- `SERVER_PORT`: Server port (default: `8080`)
- `SERVER_CONFIG`: JSON configuration for deployment mode (default: Local mode)
- `RUST_LOG`: Logging level configuration

#### Deployment Modes

The server supports two deployment modes controlled by `SERVER_CONFIG`:

**Local Mode (default)**: Runs inference and embeddings services locally
```shell
./run_server.sh
```

**HighAvailability Mode**: Proxies requests to external services
```shell
export SERVER_CONFIG='{"serverMode": "HighAvailability"}'
./run_server.sh
```

See [docs/SERVER_CONFIG.md](docs/SERVER_CONFIG.md) for complete configuration options, Docker Compose, and Kubernetes examples.

#### Basic Configuration Example:
```shell
export SERVER_PORT=3000
export RUST_LOG=debug
./run_server.sh
```

## API Endpoints

### Text Embeddings

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

### Chat Completions

Generate chat completions (simplified implementation).

**Endpoint**: `POST /v1/chat/completions`

**Request Body**:
```json
{
  "model": "gemma-2b-it",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}
```

**Response**:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1699123456,
  "model": "gemma-2b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! This is the unified predict-otron-9000 server..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 35,
    "total_tokens": 45
  }
}
```

### Health Check

**Endpoint**: `GET /`

Returns a simple "Hello, World!" message to verify the server is running.

## Development

### Project Structure

```
predict-otron-9000/
├── Cargo.toml                 # Workspace configuration
├── README.md                  # This file
├── run_server.sh             # Server startup script
└── crates/
    ├── predict-otron-9000/   # Main unified server
    │   ├── Cargo.toml
    │   └── src/
    │       └── main.rs
    ├── embeddings-engine/    # Text embeddings functionality
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs
    │       └── main.rs
    └── inference-engine/     # Text generation functionality
        ├── Cargo.toml
        ├── src/
        │   ├── lib.rs
        │   ├── cli.rs
        │   ├── server.rs
        │   ├── model.rs
        │   ├── text_generation.rs
        │   ├── token_output_stream.rs
        │   ├── utilities_lib.rs
        │   └── openai_types.rs
        └── tests/
```

### Running Tests

```shell
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p embeddings-engine
cargo test -p inference-engine
```

For comprehensive testing documentation, including unit tests, integration tests, end-to-end tests, and performance testing, please refer to the [TESTING.md](docs/TESTING.md) document.

For performance benchmarking with HTML report generation, see the [BENCHMARKING.md](BENCHMARKING.md) guide.

### Adding Features

1. **Embeddings Engine**: Modify `crates/embeddings-engine/src/lib.rs` to add new embedding models or functionality
2. **Inference Engine**: The inference engine has a modular structure - add new models in the `model.rs` module
3. **Unified Server**: Update `crates/predict-otron-9000/src/main.rs` to integrate new capabilities

## Logging and Debugging

The application uses structured logging with tracing. Log levels can be controlled via the `RUST_LOG` environment variable:

```shell
# Debug level logging
export RUST_LOG=debug

# Trace level for detailed embeddings debugging
export RUST_LOG=trace

# Module-specific logging
export RUST_LOG=predict_otron_9000=debug,embeddings_engine=trace
```

### Usage

The chat interface connects to the inference engine API and provides a user-friendly way to interact with the AI models. To use:

1. Start the predict-otron-9000 server
2. Open the chat interface in a web browser
3. Enter messages and receive AI-generated responses

The interface supports:
- Real-time messaging with the AI
- Visual indication of when the AI is generating a response
- Message history display

## Limitations

- **Inference Engine**: Currently provides a simplified implementation for chat completions. Full model loading and text generation capabilities from the inference-engine crate are not yet integrated into the unified server.
- **Model Support**: Embeddings are limited to the Nomic Embed Text v1.5 model.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `cargo test`
5. Submit a pull request


## Quick cURL verification for Chat Endpoints

Start the unified server:

```
./run_server.sh
```

Non-streaming chat completion (expects JSON response):

```
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-it",
    "messages": [
      {"role": "user", "content": "Who was the 16th president of the United States?"}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

Streaming chat completion via Server-Sent Events (SSE):

```
curl -N -X POST http://localhost:8080/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-it",
    "messages": [
      {"role": "user", "content": "Who was the 16th president of the United States?"}
    ],
    "max_tokens": 128,
    "stream": true
  }'
```

Helper scripts are also available:
- scripts/curl_chat.sh
- scripts/curl_chat_stream.sh
