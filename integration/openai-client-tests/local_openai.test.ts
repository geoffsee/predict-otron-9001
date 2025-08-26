import OpenAI from "openai";
import {describe, test, expect} from "bun:test";

const supportedModels = ["gemma-3-1b-it"];


async function requestLocalOpenAI(model: string, userPrompt: string) {
    const openai = new OpenAI({
        baseURL: "http://localhost:8080/v1",
        apiKey: "not used",
    });
    try {
        return openai.chat.completions.create({
            model: model,
            max_tokens: 100,
            stream: true,
            messages: [
                {name: "assistant_1", role: "system", content: "I am a helpful assistant" },
                {name: "user_1", role: "user", content: userPrompt}
            ]
        });
    } catch (e) {
        console.error(e);
        throw e;
    }
}

describe("Local OpenAI Completions", () => {
    test("Should return a valid message", async () => {
        const model = supportedModels.pop();
        const userPrompt = "Who was the 16th president of the United States?";
        const response = await requestLocalOpenAI(model, userPrompt);

        const chunks = [];
        for await (const chunk of response) {
            console.log('Received chunk:', chunk);
            chunks.push(chunk);
        }

        expect(chunks.length).toBeGreaterThan(0);
    })
})

