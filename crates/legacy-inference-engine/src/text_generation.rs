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
    // CPU device for fallback when operations are unsupported on primary device
    cpu_device: Option<Device>,
    // Flag to indicate if we should try to use the primary device first
    try_primary_device: bool,
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
        
        // Initialize CPU device only if the primary device is not already CPU
        let (cpu_device, try_primary_device) = if device.is_cpu() {
            // If already on CPU, no need for a fallback device
            (None, false)
        } else {
            // Store CPU device for fallback and set flag to try primary device first
            (Some(Device::Cpu), true)
        };
        
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            cpu_device,
            try_primary_device,
        }
    }

    // Helper method for model execution with fallback to CPU for unsupported operations
    fn execute_with_fallback(&mut self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        // If we're not trying primary device anymore, go straight to CPU if available
        if !self.try_primary_device {
            if let Some(cpu_device) = &self.cpu_device {
                let cpu_input = input.to_device(cpu_device).map_err(E::msg)?;
                let cpu_result = self.model.forward(&cpu_input, start_pos).map_err(E::msg)?;
                return cpu_result.to_device(&self.device).map_err(E::msg);
            } else {
                // No CPU fallback, use primary device
                return self.model.forward(input, start_pos).map_err(E::msg);
            }
        }
        
        // Try running on the primary device first
        match self.model.forward(input, start_pos) {
            Ok(result) => Ok(result),
            Err(err) => {
                // Convert to string to check for unsupported operation
                let err_string = err.to_string();
                
                // Check if the error is about unsupported operations
                if (err_string.contains("no metal implementation for") || 
                    err_string.contains("no cuda implementation for")) &&
                   self.cpu_device.is_some() {
                    
                    // Extract operation name for better logging
                    let op_name = if let Some(idx) = err_string.find("for ") {
                        &err_string[(idx + 4)..]
                    } else {
                        "an operation"
                    };
                    
                    // Log the fallback
                    println!("Warning: The primary device does not support {}. Falling back to CPU.", op_name);
                    
                    // Move input to CPU and try again
                    let cpu_device = self.cpu_device.as_ref().unwrap();
                    let cpu_input = input.to_device(cpu_device).map_err(E::msg)?;
                    let cpu_result = self.model.forward(&cpu_input, start_pos).map_err(E::msg)?;
                    
                    // Don't try primary device for future operations
                    self.try_primary_device = false;
                    println!("Successfully executed on CPU. Will use CPU for subsequent operations.");
                    
                    // Move result back to original device
                    cpu_result.to_device(&self.device).map_err(E::msg)
                } else {
                    // Not an unsupported operation error or no CPU fallback
                    Err(E::msg(err))
                }
            }
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
            // Use execute_with_fallback instead of model.forward
            let logits = self.execute_with_fallback(&input, start_pos)?;
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
            // Use execute_with_fallback instead of model.forward
            let mut logits = self.execute_with_fallback(&input, 0)?;
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
                // Use execute_with_fallback instead of model.forward
                logits = self.execute_with_fallback(&new_input, tokens.len() - 1)?;
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
            // Use execute_with_fallback instead of model.forward
            let logits = self.execute_with_fallback(&input, start_pos)?;
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