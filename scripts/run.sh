#!/bin/bash

set -e

# Resolve the project root (script_dir/..)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# todo, conditionally run this only when those files change
"$PROJECT_ROOT/scripts/build_ui.sh"

# build the frontend first
# Start the unified predict-otron-9000 server on port 8080
export SERVER_PORT=${SERVER_PORT:-8080}
export RUST_LOG=${RUST_LOG:-info}

cd "$PROJECT_ROOT" || exit 1
cargo run --bin predict-otron-9000 --release