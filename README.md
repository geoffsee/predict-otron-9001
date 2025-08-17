# predict-otron-9000

_Warning: Do NOT use this in production unless you are cool like that._

<p align="center">
  <img src="https://github.com/seemueller-io/predict-otron-9000/blob/main/predict-otron-9000.png?raw=true" width="250" />
</p>

<p align="center">
Aliens, in a native executable.
</p>


## Features
- **OpenAI Compatible**: API endpoints match OpenAI's format for easy integration
- **Text Embeddings**: Generate high-quality text embeddings using the Nomic Embed Text v1.5 model
- **Text Generation**: Chat completions with OpenAI-compatible API (simplified implementation)

## Architecture

### Core Components

- **`predict-otron-9000`**: Main unified server that combines both engines
- **`embeddings-engine`**: Handles text embeddings using FastEmbed and Nomic models
- **`inference-engine`**: Provides text generation capabilities (with modular design for various models)

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

# 3. Run the server
./run_server.sh
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
- `RUST_LOG`: Logging level configuration

Example:
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

## Limitations

- **Inference Engine**: Currently provides a simplified implementation for chat completions. Full model loading and text generation capabilities from the inference-engine crate are not yet integrated into the unified server.
- **Model Support**: Embeddings are limited to the Nomic Embed Text v1.5 model.
- **Scalability**: Single-threaded model loading may impact performance under heavy load.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `cargo test`
5. Submit a pull request