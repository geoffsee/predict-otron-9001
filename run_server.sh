#!/bin/bash

# Start the unified predict-otron-9000 server on port 8080
export SERVER_PORT=${SERVER_PORT:-8080}
export RUST_LOG=${RUST_LOG:-info}

cargo run --bin predict-otron-9000 --release