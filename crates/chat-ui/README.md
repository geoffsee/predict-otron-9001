# chat-ui

A WASM-based web chat interface for the predict-otron-9000 AI platform.

## Overview

The chat-ui provides a real-time web interface for interacting with language models through the predict-otron-9000 server. Built with Leptos and compiled to WebAssembly, it offers a modern chat experience with streaming response support.

## Features

- Real-time chat interface with the inference server
- Streaming response support
- Conversation history
- Responsive web design
- WebAssembly-powered for optimal performance

## Building and Running

### Prerequisites
- Rust toolchain with WASM target: `rustup target add wasm32-unknown-unknown`
- The predict-otron-9000 server must be running on port 8080

### Development Server
```bash
cd crates/chat-ui
./run.sh
```

This starts the development server on port 8788 with auto-reload capabilities.

### Usage
1. Start the predict-otron-9000 server: `./scripts/run.sh`
2. Start the chat-ui: `cd crates/chat-ui && ./run.sh`
3. Navigate to `http://localhost:8788`
4. Start chatting with your AI models!

## Technical Details
- Built with Leptos framework
- Compiled to WebAssembly for browser execution
- Communicates with predict-otron-9000 API via HTTP
- Sets required RUSTFLAGS for WebAssembly getrandom support