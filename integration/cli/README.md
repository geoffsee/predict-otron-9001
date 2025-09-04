# cli

A Rust/Typescript Hybrid

```console
bun run cli.ts [options] [prompt]

Simple CLI tool for testing the local OpenAI-compatible API server.

Options:
  --model <model>     Model to use (default: gemma-3-1b-it)
  --prompt <prompt>   The prompt to send (can also be provided as positional argument)
  --list-models       List all available models from the server
  --help              Show this help message

Examples:
  cd integration/cli/package
  bun run cli.ts "What is the capital of France?"
  bun run cli.ts --model gemma-3-1b-it --prompt "Hello, world!"
  bun run cli.ts --prompt "Who was the 16th president of the United States?"
  bun run cli.ts --list-models

The server must be running at http://localhost:8080
```