#!/bin/bash

# Script to test predict-otron-9000 server with 2 sequential CLI requests
# Ensures proper cleanup of child processes on exit

set -e  # Exit on any error

# Function to cleanup background processes
cleanup() {
    echo "[INFO] Cleaning up background processes..."
    if [[ -n "$SERVER_PID" ]]; then
        echo "[INFO] Killing server process (PID: $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    
    # Kill any remaining cargo processes related to predict-otron-9000
    pkill -f "predict-otron-9000" 2>/dev/null || true
    
    echo "[INFO] Cleanup complete"
}

# Set up trap to ensure cleanup on script exit
trap cleanup EXIT INT TERM

# Set environment variables
export SERVER_PORT=${SERVER_PORT:-8080}
export RUST_LOG=${RUST_LOG:-info}

echo "[INFO] Starting predict-otron-9000 server in background..."

# Start the server in background and capture its PID
cargo run --bin predict-otron-9000 --release > server.log 2>&1 &
SERVER_PID=$!

echo "[INFO] Server started with PID: $SERVER_PID"

# Function to check if server is ready
check_server() {
    curl -s -f http://localhost:8080/v1/models > /dev/null 2>&1
}

# Wait for server to be ready
echo "[INFO] Waiting for server to be ready..."
TIMEOUT=60  # 60 seconds timeout
ELAPSED=0

while ! check_server; do
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "[ERROR] Server did not start within $TIMEOUT seconds"
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo "[INFO] Still waiting for server... (${ELAPSED}s elapsed)"
done

echo "[INFO] Server is ready!"

# Run first CLI request
echo "[INFO] Running first CLI request - listing models..."
./cli.ts --list-models

echo ""
echo "[INFO] Running second CLI request - chat completion..."
./cli.ts "What is 2+2?"

echo ""
echo "[INFO] Both CLI requests completed successfully!"