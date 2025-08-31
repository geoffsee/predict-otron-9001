# cli

A Rust/Typescript Hybrid

```console
./cli [options] [prompt]

Simple CLI tool for testing the local OpenAI-compatible API server.

Options:
  --model <model>     Model to use (default: gemma-3-1b-it)
  --prompt <prompt>   The prompt to send (can also be provided as positional argument)
  --list-models       List all available models from the server
  --help              Show this help message

Examples:
  ./cli "What is the capital of France?"
  ./cli --model gemma-3-1b-it --prompt "Hello, world!"
  ./cli --prompt "Who was the 16th president of the United States?"
  ./cli --list-models

The server must be running at http://localhost:8080
```