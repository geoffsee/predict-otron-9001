#!/usr/bin/env sh

# Resolve the project root (script_dir/..)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Move into the chat-ui crate
cd "$PROJECT_ROOT/crates/chat-ui" || exit 1

# Build with cargo leptos
cargo leptos build --release

# Move the wasm file, keeping paths relative to the project root
mv "$PROJECT_ROOT/target/site/pkg/chat-ui.wasm" \
   "$PROJECT_ROOT/target/site/pkg/chat-ui_bg.wasm"