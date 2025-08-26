#!/usr/bin/env bun

import OpenAI from "openai";
import { parseArgs } from "util";

const DEFAULT_MODEL = "gemma-3-1b-it";
const DEFAULT_MAX_TOKENS = 100;

function printHelp() {
    console.log(`
Usage: bun client_cli.ts [options] [prompt]

Simple CLI tool for testing the local OpenAI-compatible API server.

Options:
  --model <model>     Model to use (default: ${DEFAULT_MODEL})
  --prompt <prompt>   The prompt to send (can also be provided as positional argument)
  --help              Show this help message

Examples:
  ./cli.ts "What is the capital of France?"
  ./cli.ts --model gemma-3-1b-it --prompt "Hello, world!"
  ./cli.ts --prompt "Who was the 16th president of the United States?"

The server should be running at http://localhost:8080
Start it with: ./run_server.sh
`);
}

const { values, positionals } = parseArgs({
    args: Bun.argv,
    options: {
        model: {
            type: 'string',
        },
        prompt: {
            type: 'string',
        },
        help: {
            type: 'boolean',
        },
    },
    strict: false,
    allowPositionals: true,
});

async function requestLocalOpenAI(model: string, userPrompt: string) {
    const openai = new OpenAI({
        baseURL: "http://localhost:8080/v1",
        apiKey: "not used",
    });
    try {
        return openai.chat.completions.create({
            model: model,
            max_tokens: DEFAULT_MAX_TOKENS,
            stream: true,
            messages: [
                {name: "assistant_1", role: "system", content: "I am a helpful assistant" },
                {name: "user_1", role: "user", content: userPrompt}
            ]
        });
    } catch (e) {
        console.error("[ERROR] Failed to connect to local OpenAI server:", e.message);
        console.error("[HINT] Make sure the server is running at http://localhost:8080");
        console.error("[HINT] Start it with: ./run_server.sh");
        throw e;
    }
}

async function main() {
    // Show help if requested
    if (values.help) {
        printHelp();
        process.exit(0);
    }

    // Get the prompt from either --prompt flag or positional argument
    const prompt = values.prompt || positionals[2]; // positionals[0] is 'bun', positionals[1] is 'client_cli.ts'
    
    if (!prompt) {
        console.error("[ERROR] No prompt provided!");
        printHelp();
        process.exit(1);
    }

    // Get the model (use default if not provided)
    const model = values.model || DEFAULT_MODEL;

    console.log(`[INFO] Using model: ${model}`);
    console.log(`[INFO] Prompt: ${prompt}`);
    console.log(`[INFO] Connecting to: http://localhost:8080/v1`);
    console.log("---");

    try {
        const response = await requestLocalOpenAI(model, prompt);
        
        // Handle streaming response
        let fullResponse = "";
        for await (const chunk of response) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
                process.stdout.write(content);
                fullResponse += content;
            }
        }
        
        console.log("\n---");
        console.log(`[INFO] Response completed. Total length: ${fullResponse.length} characters`);
        
    } catch (error) {
        console.error("\n[ERROR] Request failed:", error.message);
        process.exit(1);
    }
}

// Run the main function
main().catch(error => {
    console.error("[FATAL ERROR]:", error);
    process.exit(1);
});


