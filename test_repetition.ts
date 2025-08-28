#!/usr/bin/env node

// Test script to reproduce token repetition issue with special characters
const { fetch } = require('node-fetch');

async function testTokenRepetition() {
    console.log("Testing token repetition with special characters...");
    
    try {
        const response = await fetch('http://localhost:8080/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: "Write a simple greeting with punctuation marks like: Hello! How are you? I'm fine, thanks."
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('No reader available');
        }

        let fullResponse = '';
        let tokens = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = new TextDecoder().decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        continue;
                    }
                    
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.token) {
                            tokens.push(parsed.token);
                            fullResponse += parsed.token;
                            console.log(`Token: "${parsed.token}"`);
                        }
                    } catch (e) {
                        console.log(`Non-JSON data: ${data}`);
                    }
                }
            }
        }
        
        console.log('\n=== ANALYSIS ===');
        console.log('Full response:', fullResponse);
        console.log('Total tokens:', tokens.length);
        
        // Check for repetition issues
        const tokenCounts = {};
        let hasRepetition = false;
        
        for (const token of tokens) {
            tokenCounts[token] = (tokenCounts[token] || 0) + 1;
            if (tokenCounts[token] > 1 && token.match(/[!?,.;:]/)) {
                console.log(`⚠️  Repetition detected: "${token}" appears ${tokenCounts[token]} times`);
                hasRepetition = true;
            }
        }
        
        if (!hasRepetition) {
            console.log('✅ No token repetition detected');
        }
        
    } catch (error) {
        console.error('Error testing token repetition:', error);
    }
}

testTokenRepetition();