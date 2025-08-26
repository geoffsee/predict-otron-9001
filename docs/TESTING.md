# Testing Guide for Predict-otron-9000

This document provides comprehensive guidance on testing the Predict-otron-9000 system, including how to run existing tests and how to write new ones. The testing strategy covers different levels of testing from unit tests to performance evaluation.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Performance Testing](#performance-testing)
- [How to Run Existing Tests](#how-to-run-existing-tests)
- [Writing New Tests](#writing-new-tests)
- [Test Coverage](#test-coverage)

## Testing Overview

Predict-otron-9000 follows a multi-layered testing approach to ensure the reliability and performance of its components:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Test the complete system from user input to output
4. **Performance Tests**: Evaluate system performance under various conditions

## Unit Testing

Unit tests focus on testing individual components in isolation. The project uses Rust's built-in testing framework with the `#[test]` attribute.

### Inference Engine

The inference engine has dedicated unit tests in the `tests` directory:

- `text_generation_tests.rs`: Tests for the text generation components
- `token_output_stream_tests.rs`: Tests for token stream handling
- `model_tests.rs`: Tests for model-related functionality

These tests focus on individual components like the `Which` enum, `TokenOutputStream`, and `LogitsProcessor`.

### Embeddings Engine

The embeddings engine has unit tests embedded in the main source file:

- Tests for HTTP endpoints (`test_root` and `test_embeddings_create`)
- Validates response formats and embedding dimensions

### Running Unit Tests

To run unit tests for a specific crate:

```bash
# Run all tests for a specific crate
cd crates/inference-engine
cargo test

# Run a specific test
cargo test test_token_output_stream

# Run tests with output
cargo test -- --nocapture
```

### Writing New Unit Tests

To add new unit tests:

1. For the inference engine, add test functions to the appropriate file in the `tests` directory
2. For the embeddings engine, add test functions to the `tests` module in `main.rs`

Example of a new unit test for the inference engine:

```rust
#[test]
fn test_my_new_feature() {
    // Arrange: Set up the test data
    let input = "Test input";
    
    // Act: Call the function being tested
    let result = my_function(input);
    
    // Assert: Verify the results
    assert_eq!(result, expected_output);
}
```

## Integration Testing

Integration tests verify that different components work correctly together. 

### Current Integration Tests

- The embeddings engine tests in `main.rs` function as integration tests by testing the HTTP API endpoints

### Writing New Integration Tests

To add new integration tests:

1. Create a new test file in the `tests` directory
2. Use the Axum testing utilities to simulate HTTP requests

Example of an integration test for the API:

```rust
#[tokio::test]
async fn test_chat_completions_endpoint() {
    // Arrange: Create a test app
    let app = create_app();
    
    // Create a test request
    let request_body = serde_json::json!({
        "model": "gemma-3-1b-it",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    
    // Act: Send the request
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method(axum::http::Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    
    // Assert: Verify the response
    assert_eq!(response.status(), StatusCode::OK);
    
    // Verify response format
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(response_json.get("choices").is_some());
}
```

## End-to-End Testing

End-to-end tests validate the entire system from client request to server response.

### Manual End-to-End Testing

1. Start the server:
```bash
./run_server.sh
```

2. Use curl or other HTTP clients to test the endpoints:

```bash
# Test embeddings endpoint
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": "Hello, world!"}'

# Test chat completions endpoint
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-1b-it", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Automated End-to-End Testing

You can create automated end-to-end tests using shell scripts:

1. Create a new script in the project root:

```bash
#!/bin/bash
# e2e_test.sh

# Start the server in the background
./run_server.sh &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Run tests
echo "Testing embeddings endpoint..."
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": "Test input"}' \
  -o /tmp/embeddings_response.json

# Validate response
if grep -q "embedding" /tmp/embeddings_response.json; then
  echo "Embeddings test passed"
else
  echo "Embeddings test failed"
  exit 1
fi

# Clean up
kill $SERVER_PID
echo "All tests passed!"
```

2. Make the script executable and run it:

```bash
chmod +x e2e_test.sh
./e2e_test.sh
```

## Performance Testing

Performance testing evaluates the system's response time, throughput, and resource usage.

### Existing Performance Tests

The project includes two performance testing scripts:

1. `performance_test_embeddings.sh`: Tests the embeddings engine with various input sizes
2. `performance_test_inference.sh`: Tests the inference engine with different prompt sizes

### Running Performance Tests

Ensure the server is running, then execute the performance test scripts:

```bash
# Test embeddings performance
./performance_test_embeddings.sh

# Test inference performance
./performance_test_inference.sh
```

### Creating New Performance Tests

To create new performance tests:

1. Use the existing scripts as templates
2. Modify the test parameters (iterations, input sizes, etc.)
3. Add specific metrics you want to measure

Example of a new performance test focusing on concurrent requests:

```bash
#!/bin/bash
# concurrent_performance_test.sh

SERVER_URL="http://localhost:8080"
CONCURRENT_REQUESTS=10
TEST_INPUT="This is a test input for concurrent performance testing."

echo "Testing with $CONCURRENT_REQUESTS concurrent requests..."

# Function to send a single request
send_request() {
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"text-embedding-3-small\", \"input\": \"$TEST_INPUT\"}" \
        "$SERVER_URL/v1/embeddings" > /dev/null
    echo "Request completed"
}

# Start server if not running
# [server startup code here]

# Send concurrent requests
start_time=$(date +%s.%N)

for i in $(seq 1 $CONCURRENT_REQUESTS); do
    send_request &
done

# Wait for all requests to complete
wait

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)

echo "All $CONCURRENT_REQUESTS requests completed in ${elapsed}s"
echo "Average time per request: $(echo "$elapsed / $CONCURRENT_REQUESTS" | bc -l)s"
```

## How to Run Existing Tests

### Running All Tests

To run all tests in the project:

```bash
# From the project root
cargo test --workspace
```

### Running Specific Tests

To run tests for a specific crate:

```bash
cargo test -p inference-engine
cargo test -p embeddings-engine
```

To run a specific test:

```bash
cargo test -p inference-engine test_token_output_stream
```

### Running Tests with Output

To see the output of tests, including `println!` statements:

```bash
cargo test -- --nocapture
```

### Running Performance Tests

```bash
# Make sure server is running
./run_server.sh &

# Run performance tests
./performance_test_embeddings.sh
./performance_test_inference.sh
```

## Writing New Tests

### Test Organization

- **Unit Tests**: Place in the `tests` directory or in a `tests` module within the source file
- **Integration Tests**: Create in the `tests` directory with a focus on component interactions
- **End-to-End Tests**: Implement as shell scripts or separate Rust binaries
- **Performance Tests**: Create shell scripts that measure specific performance metrics

### Test Naming Conventions

- Use descriptive test names that indicate what is being tested
- Prefix test functions with `test_`
- For complex tests, use comments to explain the test purpose

### Test Best Practices

1. **Arrange-Act-Assert**: Structure tests with clear setup, action, and verification phases
2. **Independence**: Tests should not depend on each other
3. **Determinism**: Tests should produce the same result every time
4. **Focused Scope**: Each test should verify a single behavior
5. **Error Messages**: Use descriptive assertions that explain the expected vs. actual results

Example of a well-structured test:

```rust
#[test]
fn test_embedding_dimension_matches_specification() {
    // Arrange: Set up the test environment
    let model = create_test_model();
    let input = "Test input";
    
    // Act: Generate the embedding
    let embedding = model.embed(input);
    
    // Assert: Verify the dimension
    assert_eq!(
        embedding.len(), 
        768, 
        "Embedding dimension should be 768, but got {}", 
        embedding.len()
    );
}
```

## Test Coverage

The project currently has test coverage for:

- **Inference Engine**: Basic unit tests for key components
- **Embeddings Engine**: API endpoint tests
- **Performance**: Scripts for benchmarking both engines

Areas that could benefit from additional testing:

1. **Main Server Component**: The `predict-otron-9000` crate has limited test coverage
2. **Error Handling**: Tests for error conditions and edge cases
3. **Concurrency**: Testing behavior under concurrent load
4. **Long-Running Tests**: Stability tests for extended operation

To improve test coverage:

1. Use `cargo tarpaulin` or similar tools to measure code coverage
2. Identify uncovered code paths
3. Add tests for error conditions and edge cases
4. Implement integration tests for the main server component

---

By following this testing guide, you can ensure that the Predict-otron-9000 system maintains its reliability, performance, and correctness as it evolves.