# inference-engine

A Rust-based inference engine for running large language models locally. This tool supports both CLI mode for direct text generation and server mode with an OpenAI-compatible API.

## Features

- Run Gemma models locally (1B, 2B, 7B, 9B variants)
- CLI mode for direct text generation
- Server mode with OpenAI-compatible API
- Support for various model configurations (base, instruction-tuned)
- Metal acceleration on macOS

## Installation

### Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- Cargo package manager
- For GPU acceleration:
  - macOS: Metal support
  - Linux/Windows: CUDA support (requires appropriate drivers)

## Usage

### CLI Mode

Run the inference engine in CLI mode to generate text directly:

```bash
cargo run -p inference-engine --release -- --prompt "Your prompt text here" --which 3-1b-it
```

#### CLI Options

- `--prompt <TEXT>`: The prompt text to generate from
- `--which <MODEL>`: Model variant to use (default: "3-1b-it")
- `--server`: Run OpenAI compatible server  
- Available options: "2b", "7b", "2b-it", "7b-it", "1.1-2b-it", "1.1-7b-it", "code-2b", "code-7b", "code-2b-it", "code-7b-it", "2-2b", "2-2b-it", "2-9b", "2-9b-it", "3-1b", "3-1b-it"
- `--temperature <FLOAT>`: Temperature for sampling (higher = more random)
- `--top-p <FLOAT>`: Nucleus sampling probability cutoff
- `--sample-len <INT>`: Maximum number of tokens to generate (default: 10000)
- `--repeat-penalty <FLOAT>`: Penalty for repeating tokens (default: 1.1)
- `--repeat-last-n <INT>`: Context size for repeat penalty (default: 64)
- `--cpu`: Run on CPU instead of GPU
- `--tracing`: Enable tracing (generates a trace-timestamp.json file)

### Server Mode with OpenAI-compatible API

Run the inference engine in server mode to expose an OpenAI-compatible API:

```bash
cargo run -p inference-engine --release -- --server --port 3777 --which 3-1b-it
```

This starts a web server on the specified port (default: 3777) with an OpenAI-compatible chat completions endpoint.

#### Server Options

- `--server`: Run in server mode
- `--port <INT>`: Port to use for the server (default: 3777)
- `--which <MODEL>`: Model variant to use (default: "3-1b-it")
- Other model options as described in CLI mode

## API Usage

The server exposes an OpenAI-compatible chat completions endpoint:

### Chat Completions

```
POST /v1/chat/completions
```

#### Request Format

```json
{
  "model": "gemma-3-1b-it",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.9,
  "stream": false
}
```

#### Response Format

```json
{
  "id": "chatcmpl-123abc456def789ghi",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gemma-3-1b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking! How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 15,
    "total_tokens": 40
  }
}
```

### Example: Using cURL

```bash
curl -X POST http://localhost:3777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-it",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Example: Using Python with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3777/v1",
    api_key="dummy"  # API key is not validated but required by the client
)

response = client.chat.completions.create(
    model="gemma-3-1b-it",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Example: Using JavaScript/TypeScript with OpenAI SDK

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:3777/v1',
  apiKey: 'dummy', // API key is not validated but required by the client
});

async function main() {
  const response = await openai.chat.completions.create({
    model: 'gemma-3-1b-it',
    messages: [
      { role: 'user', content: 'What is the capital of France?' }
    ],
    temperature: 0.7,
    max_tokens: 100,
  });

  console.log(response.choices[0].message.content);
}

main();
```

## Troubleshooting

### Common Issues

1. **Model download errors**: Make sure you have a stable internet connection. The models are downloaded from Hugging Face Hub.

2. **Out of memory errors**: Try using a smaller model variant or reducing the batch size.

3. **Slow inference on CPU**: This is expected. For better performance, use GPU acceleration if available.

4. **Metal/CUDA errors**: Ensure you have the latest drivers installed for your GPU.

## License

This project is licensed under the terms specified in the LICENSE file.
