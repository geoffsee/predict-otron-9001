#!/usr/bin/env bash

PROMPT='Who was the 16th president'


# will pull gemma-3-1b-it and run the prompt
cargo run -- --prompt "${PROMPT}"

#avx: false, neon: true, simd128: false, f16c: false
 #temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
 #retrieved the files in 1.388209ms
 #loaded the model in 321.509333ms
 # user
 #Who was the 16th president
 # model
 #The 16th President of the United States was **Abraham Lincoln**. He served from March 4, 1861, to March 4, 1865.
 #40 tokens generated (31.85 token/s)