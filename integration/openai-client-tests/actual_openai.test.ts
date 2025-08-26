// #!/usr/bin/env bun
//
// import OpenAI from "openai";
// import {describe, test, expect} from "bun:test";
//
// async function requestActualOpenAI(userPrompt: string) {
//     const openai = new OpenAI();
//     return await openai.chat.completions.create({
//         model: "gpt-4o",
//         max_tokens: 100,
//         messages: [{name: "user_1", role: "user", content: userPrompt}]
//     }).then(result => result.choices[0].message);
// }
//
// // Exists as a smoke test.
// describe("Actual OpenAI Completions", () => {
//     test("Should return a valid message", async () => {
//         const userPrompt = "Who was the 16th president of the United States?";
//         const result = await requestActualOpenAI(userPrompt);
//
//         console.log({
//             test: "hitting actual openai to ensure basic functionality",
//             modelResponse: result.content,
//             userPrompt
//         });
//
//         expect(result.annotations).toEqual([])
//         expect(result.content).toBeDefined();
//         expect(result.refusal).toEqual(null);
//         expect(result.role).toEqual("assistant");
//     })
// })