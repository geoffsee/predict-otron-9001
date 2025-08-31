# Embeddings Engine

A high-performance text embeddings service that generates vector representations of text using state-of-the-art models. This crate wraps the FastEmbed library to provide embeddings with OpenAI-compatible API endpoints.

## Overview

The embeddings-engine provides a standalone service for generating text embeddings that can be used for semantic search, similarity comparisons, and other NLP tasks. It's designed to be compatible with OpenAI's embeddings API format.

## Features

- **OpenAI-Compatible API**: `/v1/embeddings` endpoint matching OpenAI's specification
- **FastEmbed Integration**: Powered by the FastEmbed library for high-quality embeddings
- **Multiple Model Support**: Support for various embedding models
- **High Performance**: Optimized for fast embedding generation
- **Standalone Service**: Can run independently or as part of the predict-otron-9000 platform

## Building and Running

### Prerequisites
- Rust toolchain
- Internet connection for initial model downloads

### Standalone Server
```bash
cargo run --bin embeddings-engine --release
```

The service will start on port 8080 by default.

## API Usage

### Generate Embeddings

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
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "model": "nomic-embed-text-v1.5",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

### Example Usage

**Using cURL**:
```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog",
    "model": "nomic-embed-text-v1.5"
  }' | jq
```

**Using Python OpenAI Client**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"  # Not validated but required by client
)

response = client.embeddings.create(
    input="Your text here",
    model="nomic-embed-text-v1.5"
)

print(response.data[0].embedding)
```

## Configuration

The service can be configured through environment variables:
- `SERVER_PORT`: Port to run on (default: 8080)
- `RUST_LOG`: Logging level (default: info)

## Integration

This service is designed to work seamlessly with the predict-otron-9000 main server, but can also be deployed independently for dedicated embeddings workloads.