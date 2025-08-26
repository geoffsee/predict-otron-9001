#!/bin/bash

# Performance testing script for embeddings-engine
# This script sends a series of embedding requests to measure performance

echo "===== Embeddings Engine Performance Test ====="
echo "Testing with varying input sizes to establish baseline performance"

# Test parameters
SERVER_URL="http://localhost:8080"
ITERATIONS=5
TEST_SIZES=("small" "medium" "large")

# Define test inputs of different sizes
SMALL_INPUT="This is a small test input for embeddings."
MEDIUM_INPUT="This is a medium-sized test input for embeddings. It contains multiple sentences with varying structure and vocabulary. The goal is to test how the embedding engine handles moderately sized inputs that might be typical in a production environment."
LARGE_INPUT="This is a large test input for embeddings. It contains multiple paragraphs with varying structure and vocabulary. The purpose of this test is to evaluate how the embedding engine performs with larger texts that might represent documents or long-form content. In a production environment, users might submit anything from short queries to entire documents for embedding, so it's important to understand the performance characteristics across different input sizes. This paragraph continues with additional text to ensure we have a sufficiently large input for testing purposes. The text doesn't need to be particularly meaningful, but it should represent a realistic workload in terms of token count and language patterns. We're particularly interested in how processing time scales with input size, as this information will help us optimize the service for different use cases and load patterns."

# Create a temp directory for test results
TEMP_DIR=$(mktemp -d)
echo "Storing test results in: $TEMP_DIR"

# Function to run a single test and record the results
run_test() {
    local size=$1
    local input=$2
    local output_file="${TEMP_DIR}/${size}_results.txt"
    
    echo -e "\n===== Testing $size input =====" | tee -a "$output_file"
    echo "Input length: $(echo "$input" | wc -w) words" | tee -a "$output_file"
    
    # Prepare JSON payload
    local json_payload=$(cat <<EOF
{
    "input": "$input",
    "model": "text-embedding-3-small"
}
EOF
)
    
    # Run the test multiple times
    for i in $(seq 1 $ITERATIONS); do
        echo "Iteration $i:" | tee -a "$output_file"
        
        # Send request and measure time
        start_time=$(date +%s.%N)
        
        # Send the embedding request
        response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$json_payload" \
            "$SERVER_URL/v1/embeddings")
        
        end_time=$(date +%s.%N)
        
        # Calculate elapsed time
        elapsed=$(echo "$end_time - $start_time" | bc)
        
        # Extract embedding dimensions
        dimensions=$(echo "$response" | grep -o '"embedding":\[[^]]*\]' | wc -c)
        
        # Log results
        echo "  Time: ${elapsed}s, Response size: $dimensions bytes" | tee -a "$output_file"
        
        # Add a small delay between requests
        sleep 1
    done
    
    # Calculate average time
    avg_time=$(grep "Time:" "$output_file" | awk '{sum+=$2} END {print sum/NR}')
    echo "Average time for $size input: ${avg_time}s" | tee -a "$output_file"
}

# Make sure the server is running
echo "Checking if the server is running..."
if ! curl -s "$SERVER_URL" > /dev/null; then
    echo "Server doesn't appear to be running at $SERVER_URL"
    echo "Please start the server with: ./run_server.sh"
    exit 1
fi

# Run tests for each input size
echo "Starting performance tests..."
run_test "small" "$SMALL_INPUT"
run_test "medium" "$MEDIUM_INPUT"
run_test "large" "$LARGE_INPUT"

echo -e "\n===== Performance Test Summary ====="
for size in "${TEST_SIZES[@]}"; do
    avg=$(grep "Average time for $size input" "${TEMP_DIR}/${size}_results.txt" | awk '{print $6}')
    echo "$size input: $avg seconds"
done

echo -e "\nDetailed results are available in: $TEMP_DIR"
echo "===== Test Complete ====="