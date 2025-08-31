#!/usr/bin/env bun

import OpenAI from "openai";
import { parseArgs } from "util";

// =====================
// Config
// =====================
const DEFAULT_MODEL = "gemma-3-1b-it";
const DEFAULT_MAX_TOKENS = 256;

// Toggle this to reduce log overhead during timing runs
const PRINT_CHUNK_DEBUG = false;

// How many rows to show in the timing tables
const SHOW_FIRST_N = 3;
const SHOW_SLOWEST_N = 3;

// =====================
// Helpers
// =====================
const now = () => performance.now();

type ChunkStat = {
    index: number;
    tSinceRequestStartMs: number;
    dtSincePrevMs: number;
    contentChars: number;
};

function printHelp() {
    console.log(`
./cli [options] [prompt]

Simple CLI tool for testing the local OpenAI-compatible API server.

Options:
  --model <model>     Model to use (default: gemma-3-1b-it)
  --prompt <prompt>   The prompt to send (can also be provided as positional argument)
  --list-models       List all available models from the server
  --help              Show this help message

Examples:
  ./cli "What is the capital of France?"
  ./cli --model gemma-3-1b-it --prompt "Hello, world!"
  ./cli --prompt "Who was the 16th president of the United States?"
  ./cli --list-models

The server must be running at http://localhost:8080
`);
}

const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    options: {
        model: { type: "string" },
        prompt: { type: "string" },
        help: { type: "boolean" },
        "list-models": { type: "boolean" },
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
        console.log("[DEBUG] Creating chat completion request...");
        return openai.chat.completions.create({
            model,
            max_tokens: DEFAULT_MAX_TOKENS,
            stream: true,
            messages: [
                {
                    role: "system",
                    content: "You are a helpful assistant who responds thoughtfully and concisely.",
                },
                { role: "user", content: userPrompt },
            ],
        });
    } catch (e: any) {
        console.error("[ERROR] Failed to connect to local OpenAI server:", e.message);
        console.error("[HINT] Make sure the server is running at http://localhost:8080");
        console.error("[HINT] Start it with: ./run_server.sh");
        throw e;
    }
}

async function listModels() {
    const openai = new OpenAI({
        baseURL: "http://localhost:8080/v1",
        apiKey: "not used",
    });
    try {
        const models = await openai.models.list();
        console.log(`[INFO] Available models from http://localhost:8080/v1:`);
        console.log("---");

        if (models.data && models.data.length > 0) {
            models.data.forEach((model, index) => {
                console.log(`${index + 1}. ${model.id}`);
                console.log(`   Owner: ${model.owned_by}`);
                console.log(`   Created: ${new Date(model.created * 1000).toISOString()}`);
                console.log("");
            });
            console.log(`Total: ${models.data.length} models available`);
        } else {
            console.log("No models found.");
        }
    } catch (e: any) {
        console.error("[ERROR] Failed to fetch models from local OpenAI server:", e.message);
        console.error("[HINT] Make sure the server is running at http://localhost:8080");
        console.error("[HINT] Start it with: ./run_server.sh");
        throw e;
    }
}

// =====================
// Timing math
// =====================
function median(nums: number[]) {
    if (nums.length === 0) return 0;
    const s = [...nums].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function quantile(nums: number[], q: number) {
    if (nums.length === 0) return 0;
    const s = [...nums].sort((a, b) => a - b);
    const pos = (s.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    return s[base + 1] !== undefined ? s[base] + rest * (s[base + 1] - s[base]) : s[base];
}

function ms(n: number) {
    return `${n.toFixed(1)} ms`;
}

// =====================
// Main
// =====================
async function main() {
    const tProgramStart = now();

    if (values.help) {
        printHelp();
        process.exit(0);
    }

    if (values["list-models"]) {
        try {
            await listModels();
            process.exit(0);
        } catch (error: any) {
            console.error("\n[ERROR] Failed to list models:", error.message);
            process.exit(1);
        }
    }

    const prompt = values.prompt ?? positionals[0];

    if (!prompt) {
        console.error("[ERROR] No prompt provided!");
        printHelp();
        process.exit(1);
    }

    const model = values.model || DEFAULT_MODEL;

    console.log(`[INFO] Using model: ${model}`);
    console.log(`[INFO] Prompt: ${prompt}`);
    console.log(`[INFO] Connecting to: http://localhost:8080/v1`);
    console.log("---");

    const tBeforeRequest = now();

    try {
        console.log("[DEBUG] Initiating request to OpenAI server...");
        const response = await requestLocalOpenAI(model, prompt);
        const tAfterCreate = now();

        // Streaming handling + timing
        let fullResponse = "";
        let chunkCount = 0;

        const chunkStats: ChunkStat[] = [];
        let tFirstChunk: number | null = null;
        let tPrevChunk: number | null = null;

        console.log("[INFO] Waiting for model to generate response...");
        let loadingInterval;
        if (!PRINT_CHUNK_DEBUG) {
            // Show loading animation only if not in debug mode
            const loadingChars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let i = 0;
            process.stdout.write('\r[INFO] Thinking  ');
            loadingInterval = setInterval(() => {
                process.stdout.write(`\r[INFO] Thinking ${loadingChars[i++ % loadingChars.length]} `);
            }, 80);
        } else {
            console.log("[DEBUG] Starting to receive streaming response...");
        }

        for await (const chunk of response) {
            // Clear loading animation on first chunk
            if (loadingInterval) {
                clearInterval(loadingInterval);
                process.stdout.write('\r                      \r');
            }
            const tNow = now();
            chunkCount++;

            // Extract content (delta) if present
            const content = chunk.choices?.[0]?.delta?.content ?? "";
            if (PRINT_CHUNK_DEBUG) {
                console.log(`[DEBUG] Received chunk #${chunkCount}:`, JSON.stringify(chunk));
                if (content) console.log(`[DEBUG] Chunk content: "${content}"`);
            }

            if (content) {
                process.stdout.write(content);
                fullResponse += content;
            }

            if (tFirstChunk === null) tFirstChunk = tNow;

            const dtSincePrev = tPrevChunk === null ? 0 : tNow - tPrevChunk;
            chunkStats.push({
                index: chunkCount,
                tSinceRequestStartMs: tNow - tBeforeRequest,
                dtSincePrevMs: dtSincePrev,
                contentChars: content.length,
            });

            tPrevChunk = tNow;
        }

        // =========
        // Summary
        // =========
        const tStreamEnd = now();
        const totalChars = fullResponse.length;

        console.log("\n---");
        console.log(`[DEBUG] Stream completed after ${chunkCount} chunks`);
        console.log(`[INFO] Response completed. Total length: ${totalChars} characters`);

        // Build timing metrics
        const ttfbMs = (tFirstChunk ?? tStreamEnd) - tAfterCreate; // time from create() resolved → first chunk
        const createOverheadMs = tAfterCreate - tBeforeRequest;    // time spent awaiting create() promise
        const totalSinceRequestMs = tStreamEnd - tBeforeRequest;   // from just before create() to last chunk
        const streamDurationMs =
            tFirstChunk === null ? 0 : tStreamEnd - tFirstChunk;

        const gaps = chunkStats
            .map((c) => c.dtSincePrevMs)
            // ignore the first "gap" which is 0 by construction
            .slice(1);

        const avgGapMs = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
        const medGapMs = median(gaps);
        const p95GapMs = quantile(gaps, 0.95);

        let maxGapMs = 0;
        let maxGapAtChunk = 0;
        for (let i = 0; i < gaps.length; i++) {
            if (gaps[i] > maxGapMs) {
                maxGapMs = gaps[i];
                maxGapAtChunk = i + 2; // +1 to move from 0-based, +1 because we sliced starting at second chunk
            }
        }

        // Pretty print summary
        console.log("\n=== Timing Summary ===");
        console.log(`create() await time:        ${ms(createOverheadMs)}`);
        console.log(`TTFB (to 1st chunk):        ${ms(ttfbMs)}`);
        console.log(`Stream duration:            ${ms(streamDurationMs)}`);
        console.log(`End-to-end (req→last):      ${ms(totalSinceRequestMs)}`);
        console.log(`Chunks:                     ${chunkCount}`);
        console.log(`Total content chars:        ${totalChars}`);
        console.log(`Avg chars/chunk:            ${(chunkCount ? totalChars / chunkCount : 0).toFixed(1)}`);
        console.log(`Inter-chunk gap (avg):      ${ms(avgGapMs)}`);
        console.log(`Inter-chunk gap (median):   ${ms(medGapMs)}`);
        console.log(`Inter-chunk gap (p95):      ${ms(p95GapMs)}`);
        if (gaps.length > 0) {
            console.log(`Largest gap:                ${ms(maxGapMs)} (before chunk #${maxGapAtChunk})`);
        }

        // Small tables: first N and slowest N gaps
        const firstRows = chunkStats.slice(0, SHOW_FIRST_N).map((c) => ({
            chunk: c.index,
            "t since request": `${c.tSinceRequestStartMs.toFixed(1)} ms`,
            "dt since prev": `${c.dtSincePrevMs.toFixed(1)} ms`,
            "chars": c.contentChars,
        }));

        const slowestRows = chunkStats
            .slice(1) // skip first (no meaningful gap)
            .sort((a, b) => b.dtSincePrevMs - a.dtSincePrevMs)
            .slice(0, SHOW_SLOWEST_N)
            .map((c) => ({
                chunk: c.index,
                "t since request": `${c.tSinceRequestStartMs.toFixed(1)} ms`,
                "dt since prev": `${c.dtSincePrevMs.toFixed(1)} ms`,
                "chars": c.contentChars,
            }));

        if (firstRows.length > 0) {
            console.log("\n--- First chunk timings ---");
            // @ts-ignore Bun/Node support console.table
            console.table(firstRows);
        }

        if (slowestRows.length > 0) {
            console.log(`\n--- Slowest ${SHOW_SLOWEST_N} gaps ---`);
            // @ts-ignore
            console.table(slowestRows);
        }

        const tProgramEnd = now();
        console.log("\n=== Program Overhead ===");
        console.log(`Total program runtime:      ${ms(tProgramEnd - tProgramStart)}`);

    } catch (error: any) {
        console.error("\n[ERROR] Request failed:", error.message);
        process.exit(1);
    }
}

// Run the main function
main().catch((error) => {
    console.error("[FATAL ERROR]:", error);
    process.exit(1);
});
