#!/usr/bin/env sh

# Set RUSTFLAGS for getrandom's WebAssembly support
export RUSTFLAGS='--cfg getrandom_backend="wasm_js"'

trunk serve --port 8788