use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::io::Write;

use crate::model::Model;
use crate::token_output_stream::TokenOutputStream;

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    // Run text generation and print to stdout
    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };

        let eot_token = match self.tokenizer.get_token("<end_of_turn>") {
            Some(token) => token,
            None => {
                println!(
                    "Warning: <end_of_turn> token not found in tokenizer, using <eos> as a backup"
                );
                eos_token
            }
        };

        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);

                // Manual implementation of repeat penalty to avoid type conflicts
                let mut logits_vec = logits.to_vec1::<f32>()?;

                for &token_id in &tokens[start_at..] {
                    let token_id = token_id as usize;
                    if token_id < logits_vec.len() {
                        let score = logits_vec[token_id];
                        let sign = if score < 0.0 { -1.0 } else { 1.0 };
                        logits_vec[token_id] = sign * score / self.repeat_penalty;
                    }
                }

                // Create a new tensor with the modified logits
                let device = logits.device().clone();
                let shape = logits.shape().clone();
                let new_logits = Tensor::new(&logits_vec[..], &device)?;
                new_logits.reshape(shape)?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eot_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }

    // Run text generation and write to a buffer
    pub fn run_with_output(&mut self, prompt: &str, sample_len: usize, output: &mut Vec<u8>) -> Result<()> {
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // Write prompt tokens to output
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                write!(output, "{}", t)?;
            }
        }

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };

        let eot_token = match self.tokenizer.get_token("<end_of_turn>") {
            Some(token) => token,
            None => {
                write!(output, "Warning: <end_of_turn> token not found in tokenizer, using <eos> as a backup")?;
                eos_token
            }
        };

        // Determine if we're using a Model3 (gemma-3) variant
        let is_model3 = match &self.model {
            Model::V3(_) => true,
            _ => false,
        };

        // For Model3, we need to use a different approach
        if is_model3 {
            // For gemma-3 models, we'll generate one token at a time with the full context
            let start_gen = std::time::Instant::now();

            // Initial generation with the full prompt
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let mut logits = self.model.forward(&input, 0)?;
            logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            for _ in 0..sample_len {
                // Apply repeat penalty if needed
                let current_logits = if self.repeat_penalty == 1. {
                    logits.clone()
                } else {
                    let start_at = tokens.len().saturating_sub(self.repeat_last_n);

                    // Manual implementation of repeat penalty to avoid type conflicts
                    let mut logits_vec = logits.to_vec1::<f32>()?;

                    for &token_id in &tokens[start_at..] {
                        let token_id = token_id as usize;
                        if token_id < logits_vec.len() {
                            let score = logits_vec[token_id];
                            let sign = if score < 0.0 { -1.0 } else { 1.0 };
                            logits_vec[token_id] = sign * score / self.repeat_penalty;
                        }
                    }

                    // Create a new tensor with the modified logits
                    let device = logits.device().clone();
                    let shape = logits.shape().clone();
                    let new_logits = Tensor::new(&logits_vec[..], &device)?;
                    new_logits.reshape(shape)?
                };

                let next_token = self.logits_processor.sample(&current_logits)?;
                tokens.push(next_token);
                generated_tokens += 1;

                if next_token == eos_token || next_token == eot_token {
                    break;
                }

                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    write!(output, "{}", t)?;
                }

                // For the next iteration, just use the new token
                let new_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                logits = self.model.forward(&new_input, tokens.len() - 1)?;
                logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            }

            return Ok(());
        }

        // Standard approach for other models
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);

                // Manual implementation of repeat penalty to avoid type conflicts
                let mut logits_vec = logits.to_vec1::<f32>()?;

                for &token_id in &tokens[start_at..] {
                    let token_id = token_id as usize;
                    if token_id < logits_vec.len() {
                        let score = logits_vec[token_id];
                        let sign = if score < 0.0 { -1.0 } else { 1.0 };
                        logits_vec[token_id] = sign * score / self.repeat_penalty;
                    }
                }

                // Create a new tensor with the modified logits
                let device = logits.device().clone();
                let shape = logits.shape().clone();
                let new_logits = Tensor::new(&logits_vec[..], &device)?;
                new_logits.reshape(shape)?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eot_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                write!(output, "{}", t)?;
            }
        }

        // Write any remaining tokens
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            write!(output, "{}", rest)?;
        }

        Ok(())
    }
}