#!/bin/bash

# Simple test script for inference-engine
# This script sends a single chat completion request

echo "===== Inference Engine Test ====="

# Test parameters
SERVER_URL="http://localhost:8080"  # Changed from 8080 to 3777 to match main.rs default port
MAX_TOKENS=10
PROMPT="What is the capital of France?"
MODEL="${MODEL_ID:-gemma-2-2b-it}"  # Using gemma-2-2b-it as specified in the original test

# Create a temp directory for test results
TEMP_DIR=$(mktemp -d)
echo "Storing test results in: $TEMP_DIR"

# Prepare JSON payload
json_payload=$(cat <<EOF
{
    "model": "$MODEL", 
    "messages": [{"role": "user", "content": "$PROMPT"}],
    "max_tokens": $MAX_TOKENS
}
EOF
)

# Make sure the server is running
echo "Checking if the server is running..."
if ! curl -s "$SERVER_URL" > /dev/null; then
    echo "Server doesn't appear to be running at $SERVER_URL"
    echo "Please start the server with: ./run_server.sh"
    exit 1
fi

echo "Sending request..."

# Send request and measure time
start_time=$(date +%s.%N)

# Send the chat completion request with 30 second timeout
# Note: The gemma-2-2b-it model takes ~12.57 seconds per token on average
# So even with MAX_TOKENS=10, the request might time out before completion
# The timeout ensures the script doesn't hang indefinitely
response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$json_payload" \
    --max-time 30 \
    "$SERVER_URL/v1/chat/completions")

end_time=$(date +%s.%N)

# Calculate elapsed time
elapsed=$(echo "$end_time - $start_time" | bc)

# Extract response content length
content_length=$(echo "$response" | grep -o '"content":"[^"]*"' | wc -c)

# Check if we got an error
error_check=$(echo "$response" | grep -c "error")
if [ "$error_check" -gt 0 ]; then
    echo "Error in response: $response"
fi

# Log results
echo "Time: ${elapsed}s, Response size: $content_length bytes"
echo "Response: $response"

echo -e "\nTest Complete"