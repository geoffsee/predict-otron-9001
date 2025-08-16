// Test requests for the OpenAI-compatible endpoint in the inference server
// This file contains IIFE (Immediately Invoked Function Expression) JavaScript requests
// to test the /v1/chat/completions endpoint

// Basic chat completion request
(async function testBasicChatCompletion() {
  console.log("Test 1: Basic chat completion request");
  try {
    const response = await fetch('http://localhost:3777/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: "gemma-2-2b-it",
        messages: [
          {
            role: "user",
            content: "Who was the 16th president of the United States?"
          }
        ],
        max_tokens: 100
      })
    });

    const data = await response.json();
    console.log("Response:", JSON.stringify(data, null, 2));
  } catch (error) {
    console.error("Error:", error);
  }
})();

// Multi-turn conversation
(async function testMultiTurnConversation() {
  console.log("\nTest 2: Multi-turn conversation");
  try {
    const response = await fetch('http://localhost:3777/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: "gemma-2-2b-it",
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant that provides concise answers."
          },
          {
            role: "user",
            content: "What is machine learning?"
          },
          {
            role: "assistant",
            content: "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
          },
          {
            role: "user",
            content: "Give me an example of a machine learning algorithm."
          }
        ],
        max_tokens: 150
      })
    });

    const data = await response.json();
    console.log("Response:", JSON.stringify(data, null, 2));
  } catch (error) {
    console.error("Error:", error);
  }
})();

// Request with temperature and top_p parameters
(async function testTemperatureAndTopP() {
  console.log("\nTest 3: Request with temperature and top_p parameters");
  try {
    const response = await fetch('http://localhost:3777/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: "gemma-2-2b-it",
        messages: [
          {
            role: "user",
            content: "Write a short poem about artificial intelligence."
          }
        ],
        max_tokens: 200,
        temperature: 0.8,
        top_p: 0.9
      })
    });

    const data = await response.json();
    console.log("Response:", JSON.stringify(data, null, 2));
  } catch (error) {
    console.error("Error:", error);
  }
})();

// Request with streaming enabled
(async function testStreaming() {
  console.log("\nTest 4: Request with streaming enabled");
  try {
    const response = await fetch('http://localhost:3777/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: "gemma-2-2b-it",
        messages: [
          {
            role: "user",
            content: "Explain quantum computing in simple terms."
          }
        ],
        max_tokens: 150,
        stream: true
      })
    });

    // Note: Streaming might not be implemented yet, this is to test the API's handling of the parameter
    if (response.headers.get('content-type')?.includes('text/event-stream')) {
      console.log("Streaming response detected. Reading stream...");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        console.log("Chunk:", chunk);
      }
    } else {
      const data = await response.json();
      console.log("Non-streaming response:", JSON.stringify(data, null, 2));
    }
  } catch (error) {
    console.error("Error:", error);
  }
})();

// Request with a different model
(async function testDifferentModel() {
  console.log("\nTest 5: Request with a different model");
  try {
    const response = await fetch('http://localhost:3777/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: "gemma-2-2b-it", // Using a different model if available
        messages: [
          {
            role: "user",
            content: "What are the benefits of renewable energy?"
          }
        ],
        max_tokens: 150
      })
    });

    const data = await response.json();
    console.log("Response:", JSON.stringify(data, null, 2));
  } catch (error) {
    console.error("Error:", error);
  }
})();

console.log("\nAll test requests have been sent. Check the server logs for more details.");
console.log("To run the server, use: cargo run --bin inference-engine -- --server");
