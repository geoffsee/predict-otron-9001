#!/bin/bash

# Performance testing script for inference-engine
# This script sends a series of chat completion requests to measure performance

echo "===== Inference Engine Performance Test ====="
echo "Testing with varying prompt sizes to establish baseline performance"

# Test parameters
SERVER_URL="http://localhost:8080"
ITERATIONS=3  # Lower than embeddings test due to longer processing time
TEST_SIZES=("small" "medium" "large")
MAX_TOKENS=50  # Limit token generation to keep tests shorter

# Define test prompts of different sizes
SMALL_PROMPT="What is the capital of France?"
MEDIUM_PROMPT="Explain the basic principles of machine learning. Include a brief overview of supervised and unsupervised learning."
LARGE_PROMPT="Write a comprehensive explanation of large language models. Include details about their architecture, training process, capabilities, limitations, and potential future developments. Also discuss ethical considerations around their use and deployment."

# Create a temp directory for test results
TEMP_DIR=$(mktemp -d)
echo "Storing test results in: $TEMP_DIR"

# Function to run a single test and record the results
run_test() {
    local size=$1
    local prompt=$2
    local output_file="${TEMP_DIR}/${size}_results.txt"
    
    echo -e "\n===== Testing $size prompt =====" | tee -a "$output_file"
    echo "Prompt length: $(echo "$prompt" | wc -w) words" | tee -a "$output_file"
    
    # Prepare JSON payload
    local json_payload=$(cat <<EOF
{
    "model": "gemma-3-1b-it",
    "messages": [{"role": "user", "content": "$prompt"}],
    "max_tokens": $MAX_TOKENS
}
EOF
)
    
    # Run the test multiple times
    for i in $(seq 1 $ITERATIONS); do
        echo "Iteration $i:" | tee -a "$output_file"
        
        # Send request and measure time
        start_time=$(date +%s.%N)
        
        # Send the chat completion request
        response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$json_payload" \
            "$SERVER_URL/v1/chat/completions")
        
        end_time=$(date +%s.%N)
        
        # Calculate elapsed time
        elapsed=$(echo "$end_time - $start_time" | bc)
        
        # Extract response content length
        content_length=$(echo "$response" | grep -o '"content":"[^"]*"' | wc -c)
        
        # Check if we got an error (for troubleshooting)
        error_check=$(echo "$response" | grep -c "error")
        if [ "$error_check" -gt 0 ]; then
            echo "  Error in response: $response" | tee -a "$output_file"
        fi
        
        # Log results
        echo "  Time: ${elapsed}s, Response size: $content_length bytes" | tee -a "$output_file"
        
        # Add a delay between requests to allow server to recover
        sleep 2
    done
    
    # Calculate average time
    avg_time=$(grep "Time:" "$output_file" | grep -v "Error" | awk '{sum+=$2} END {if(NR>0) print sum/NR; else print "N/A"}')
    echo "Average time for $size prompt: ${avg_time}s" | tee -a "$output_file"
}

# Make sure the server is running
echo "Checking if the server is running..."
if ! curl -s "$SERVER_URL" > /dev/null; then
    echo "Server doesn't appear to be running at $SERVER_URL"
    echo "Please start the server with: ./run_server.sh"
    exit 1
fi

# Run tests for each prompt size
echo "Starting performance tests..."
run_test "small" "$SMALL_PROMPT"
run_test "medium" "$MEDIUM_PROMPT"
run_test "large" "$LARGE_PROMPT"

echo -e "\n===== Performance Test Summary ====="
for size in "${TEST_SIZES[@]}"; do
    avg=$(grep "Average time for $size prompt" "${TEMP_DIR}/${size}_results.txt" | awk '{print $6}')
    if [ -z "$avg" ]; then
        avg="N/A (possible errors)"
    else
        avg="${avg}s"
    fi
    echo "$size prompt: $avg"
done

# Provide more detailed analysis if possible
echo -e "\n===== Performance Analysis ====="
echo "Note: The inference-engine response times include:"
echo "  - Input prompt tokenization"
echo "  - Model inference (token generation)"
echo "  - Response post-processing"
echo "Check server logs for more detailed performance breakdown"

echo -e "\nDetailed results are available in: $TEMP_DIR"
echo "===== Test Complete ====="