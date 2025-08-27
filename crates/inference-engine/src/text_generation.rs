use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::collections::HashMap;

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
    // Cache for repeat penalty computation to avoid redundant calculations
    penalty_cache: HashMap<usize, f32>,
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
            penalty_cache: HashMap::new(),
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
                
                // Check if the error is about unsupported operations or shape mismatches
                if (err_string.contains("no metal implementation for") ||
                     err_string.contains("no cuda implementation for") ||
                     err_string.contains("shape mismatch") ||
                     err_string.contains("broadcast_add")) &&
                   self.cpu_device.is_some() {
                    
                    // Extract operation name for better logging
                    let op_name = if let Some(idx) = err_string.find("for ") {
                        &err_string[(idx + 4)..]
                    } else if err_string.contains("shape mismatch") {
                        "shape mismatch operation"
                    } else {
                        "an operation"
                    };
                    
                    // Log the fallback
                    tracing::warn!("The primary device does not support {}. Falling back to CPU.", op_name);
                    
                    // Move input to CPU and try again
                    let cpu_device = self.cpu_device.as_ref().unwrap();
                    let cpu_input = input.to_device(cpu_device).map_err(E::msg)?;
                    let cpu_result = self.model.forward(&cpu_input, start_pos).map_err(E::msg)?;
                    
                    // Don't try primary device for future operations
                    self.try_primary_device = false;
                    tracing::info!("Successfully executed on CPU. Will use CPU for subsequent operations.");
                    
                    // Move result back to original device
                    cpu_result.to_device(&self.device).map_err(E::msg)
                } else {
                    // Not an unsupported operation error or no CPU fallback
                    Err(E::msg(err))
                }
            }
        }
    }
    
    // Reset method to clear state between requests
    pub fn reset_state(&mut self) {
        // Reset the primary device flag so we try the primary device first for each new request
        if !self.device.is_cpu() {
            self.try_primary_device = true;
        }
        // Clear the penalty cache to avoid stale cached values from previous requests
        self.penalty_cache.clear();
    }

    // Helper method to apply repeat penalty with caching for optimization
    pub fn apply_cached_repeat_penalty(
        &mut self,
        logits: Tensor,
        tokens: &[u32],
    ) -> Result<(Tensor, std::time::Duration)> {
        let repeat_start = std::time::Instant::now();

        // If no penalty, return the original logits
        if self.repeat_penalty == 1.0 {
            return Ok((logits, repeat_start.elapsed()));
        }

        // Get the tokens to penalize (the last n tokens)
        let start_at = tokens.len().saturating_sub(self.repeat_last_n);
        let penalty_tokens = &tokens[start_at..];

        // Extract logits to a vector for modification
        let mut logits_vec = logits.to_vec1::<f32>()?;
        let cache_hits = std::cell::Cell::new(0);

        // Apply penalties with caching
        for &token_id in penalty_tokens {
            let token_id = token_id as usize;
            if token_id < logits_vec.len() {
                // Check if we've already calculated this token's penalty
                if let Some(penalized_score) = self.penalty_cache.get(&token_id) {
                    // Use cached value
                    logits_vec[token_id] = *penalized_score;
                    cache_hits.set(cache_hits.get() + 1);
                } else {
                    // Calculate and cache new value
                    let score = logits_vec[token_id];
                    let sign = if score < 0.0 { -1.0 } else { 1.0 };
                    let penalized_score = sign * score / self.repeat_penalty;
                    logits_vec[token_id] = penalized_score;
                    self.penalty_cache.insert(token_id, penalized_score);
                }
            }
        }

        // Log cache efficiency statistics
        if !penalty_tokens.is_empty() {
            let cache_efficiency = (cache_hits.get() as f32 / penalty_tokens.len() as f32) * 100.0;
            tracing::trace!("Repeat penalty cache hits: {}/{} ({:.1}%)",
                           cache_hits.get(), penalty_tokens.len(), cache_efficiency);
        }

        // Create a new tensor with the modified logits (single tensor creation)
        let device = logits.device().clone();
        let shape = logits.shape().clone();
        let new_logits = Tensor::new(&logits_vec[..], &device)?;
        let result = new_logits.reshape(shape)?;

        let elapsed = repeat_start.elapsed();
        Ok((result, elapsed))
    }

    // Run text generation and print to stdout
    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        
        // Track overall performance
        let start_time = std::time::Instant::now();
        
        // Clear penalty cache for new generation
        self.penalty_cache.clear();
        tracing::debug!("Cleared penalty cache for new generation");
        
        // Phase 1: Tokenize input
        let tokenize_start = std::time::Instant::now();
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        
        let tokenize_time = tokenize_start.elapsed();
        tracing::debug!("Tokenization completed in {:.2?}", tokenize_time);
        tracing::debug!("Input tokens: {}", tokens.len());
        
        // Print tokenized prompt
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

        // Determine if we're using a Model2 (gemma-2) or Model3 (gemma-3) variant
        // Both need special handling for shape compatibility
        let needs_special_handling = match &self.model {
            Model::V2(_) => true,
            Model::V3(_) => true,
            _ => false,
        };

        // Phase 2: Text generation
        let start_gen = std::time::Instant::now();
        
        // Track per-token generation timing for performance analysis
        let mut token_times = Vec::new();
        let mut forward_times = Vec::new();
        let mut repeat_penalty_times = Vec::new();
        let mut sampling_times = Vec::new();
        
        // For Model2 and Model3, we need to use a special approach for shape compatibility
        if needs_special_handling {
            // For gemma-2 and gemma-3 models, we'll generate one token at a time with the full context
            tracing::debug!("Using special generation approach for gemma-2/gemma-3 models");

            // Initial generation with the full prompt
            let forward_start = std::time::Instant::now();
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            
            // Use execute_with_fallback which handles both device compatibility and shape mismatches
            let mut logits = self.execute_with_fallback(&input, 0)?;
            
            logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let forward_time = forward_start.elapsed();
            forward_times.push(forward_time);

            for _ in 0..sample_len {
                let token_start = std::time::Instant::now();
                
                // Apply repeat penalty using optimized cached implementation
                let (current_logits, repeat_time) = self.apply_cached_repeat_penalty(logits.clone(), &tokens)?;
                repeat_penalty_times.push(repeat_time);

                // Track token sampling
                let sampling_start = std::time::Instant::now();
                let next_token = self.logits_processor.sample(&current_logits)?;
                let sampling_time = sampling_start.elapsed();
                sampling_times.push(sampling_time);

                tokens.push(next_token);
                generated_tokens += 1;

                if next_token == eos_token || next_token == eot_token {
                    break;
                }

                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }

                // For the next iteration, just use the new token
                let forward_start = std::time::Instant::now();
                let new_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                
                // Use execute_with_fallback for both Gemma 3 and other models
                logits = self.execute_with_fallback(&new_input, tokens.len() - 1)?;
                
                logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let forward_time = forward_start.elapsed();
                forward_times.push(forward_time);
                
                let token_time = token_start.elapsed();
                token_times.push(token_time);
            }
        } else {
            // Standard approach for other models
            tracing::debug!("Using standard generation approach");
            
            for index in 0..sample_len {
            let token_start = std::time::Instant::now();
            
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            
            // Track tensor operations and model forward pass
            let forward_start = std::time::Instant::now();
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.execute_with_fallback(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let forward_time = forward_start.elapsed();
            forward_times.push(forward_time);
            
            // Apply repeat penalty using optimized cached implementation
            let (logits, repeat_time) = self.apply_cached_repeat_penalty(logits, &tokens)?;
            repeat_penalty_times.push(repeat_time);

            // Track token sampling
            let sampling_start = std::time::Instant::now();
            let next_token = self.logits_processor.sample(&logits)?;
            let sampling_time = sampling_start.elapsed();
            sampling_times.push(sampling_time);
            
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eot_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            
            let token_time = token_start.elapsed();
            token_times.push(token_time);
        }
        }
        
        let dt = start_gen.elapsed();
        
        // Phase 3: Final decoding and output
        let decode_start = std::time::Instant::now();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        let decode_time = decode_start.elapsed();
        
        std::io::stdout().flush()?;
        
        // Calculate generation speed
        let tokens_per_second = generated_tokens as f64 / dt.as_secs_f64();
        
        // Calculate average time per token and component breakdown
        let avg_token_time = if !token_times.is_empty() {
            token_times.iter().sum::<std::time::Duration>() / token_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let avg_forward_time = if !forward_times.is_empty() {
            forward_times.iter().sum::<std::time::Duration>() / forward_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let avg_repeat_time = if !repeat_penalty_times.is_empty() {
            repeat_penalty_times.iter().sum::<std::time::Duration>() / repeat_penalty_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let avg_sampling_time = if !sampling_times.is_empty() {
            sampling_times.iter().sum::<std::time::Duration>() / sampling_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        // Log performance metrics
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            tokens_per_second,
        );
        
        // Record detailed performance metrics
        tracing::info!("Text generation completed in {:.2?}", dt);
        tracing::info!("Tokens generated: {}", generated_tokens);
        tracing::info!("Generation speed: {:.2} tokens/second", tokens_per_second);
        tracing::info!("Average time per token: {:.2?}", avg_token_time);
        tracing::debug!("  - Forward pass: {:.2?} ({:.1}%)", 
            avg_forward_time, 
            avg_forward_time.as_secs_f64() / avg_token_time.as_secs_f64() * 100.0
        );
        tracing::debug!("  - Repeat penalty: {:.2?} ({:.1}%)", 
            avg_repeat_time,
            avg_repeat_time.as_secs_f64() / avg_token_time.as_secs_f64() * 100.0
        );
        tracing::debug!("  - Sampling: {:.2?} ({:.1}%)", 
            avg_sampling_time,
            avg_sampling_time.as_secs_f64() / avg_token_time.as_secs_f64() * 100.0
        );
        
        // Log total request time
        let total_time = start_time.elapsed();
        tracing::info!("Total request time: {:.2?}", total_time);
        tracing::debug!("  - Tokenization: {:.2?} ({:.1}%)", 
            tokenize_time,
            tokenize_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        tracing::debug!("  - Generation: {:.2?} ({:.1}%)", 
            dt,
            dt.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        tracing::debug!("  - Final decoding: {:.2?} ({:.1}%)", 
            decode_time,
            decode_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        
        Ok(())
    }

    // Run text generation and write to a buffer
    pub fn run_with_output(&mut self, prompt: &str, sample_len: usize, output: &mut Vec<u8>) -> Result<()> {
        use std::io::Write;
        
        // Track overall performance
        let start_time = std::time::Instant::now();
        
        // Clear penalty cache for new generation
        self.penalty_cache.clear();
        tracing::debug!("Cleared penalty cache for new generation (API mode)");
        
        // Phase 1: Tokenize input
        let tokenize_start = std::time::Instant::now();
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        
        let tokenize_time = tokenize_start.elapsed();
        tracing::debug!("API Tokenization completed in {:.2?}", tokenize_time);
        tracing::debug!("API Input tokens: {}", tokens.len());

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

        // Determine if we're using a Model2 (gemma-2) or Model3 (gemma-3) variant
        // Both need special handling for shape compatibility
        let needs_special_handling = match &self.model {
            Model::V2(_) => true,
            Model::V3(_) => true,
            _ => false,
        };

        // Check if we're specifically using a Model3 (gemma-3) for additional error handling
        // let is_model_v3 = matches!(&self.model, Model::V3(_));

        // Track generation timing
        let start_gen = std::time::Instant::now();
        
        // Track per-token generation timing for performance analysis
        let mut token_times = Vec::new();
        let mut forward_times = Vec::new();
        let mut repeat_penalty_times = Vec::new();
        let mut sampling_times = Vec::new();

        // For Model2 and Model3, we need to use a special approach for shape compatibility
        if needs_special_handling {
            // For gemma-2 and gemma-3 models, we'll generate one token at a time with the full context
            tracing::debug!("Using special generation approach for gemma-2/gemma-3 models");

            // Initial generation with the full prompt
            let forward_start = std::time::Instant::now();
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            
            // Use execute_with_fallback which handles both device compatibility and shape mismatches
            let mut logits = self.execute_with_fallback(&input, 0)?;
            
            logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let forward_time = forward_start.elapsed();
            forward_times.push(forward_time);

            for _ in 0..sample_len {
                let token_start = std::time::Instant::now();
                
                // Apply repeat penalty using optimized cached implementation
                let (current_logits, repeat_time) = self.apply_cached_repeat_penalty(logits.clone(), &tokens)?;
                repeat_penalty_times.push(repeat_time);

                // Track token sampling
                let sampling_start = std::time::Instant::now();
                let next_token = self.logits_processor.sample(&current_logits)?;
                let sampling_time = sampling_start.elapsed();
                sampling_times.push(sampling_time);

                tokens.push(next_token);
                generated_tokens += 1;

                if next_token == eos_token || next_token == eot_token {
                    break;
                }

                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    write!(output, "{}", t)?;
                }

                // For the next iteration, just use the new token
                let forward_start = std::time::Instant::now();
                let new_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                
                // Use execute_with_fallback for both Gemma 3 and other models
                logits = self.execute_with_fallback(&new_input, tokens.len() - 1)?;
                
                logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let forward_time = forward_start.elapsed();
                forward_times.push(forward_time);
                
                let token_time = token_start.elapsed();
                token_times.push(token_time);
            }
            
            let dt = start_gen.elapsed();
            
            // Calculate and log performance metrics
            Self::log_performance_metrics(
                dt, generated_tokens, &token_times, &forward_times, 
                &repeat_penalty_times, &sampling_times, tokenize_time, 
                std::time::Duration::from_secs(0), start_time, "API"
            );

            return Ok(());
        }

        // Standard approach for other models
        tracing::debug!("Using standard generation approach");
        
        for index in 0..sample_len {
            let token_start = std::time::Instant::now();
            
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            
            // Track tensor operations and model forward pass
            let forward_start = std::time::Instant::now();
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.execute_with_fallback(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let forward_time = forward_start.elapsed();
            forward_times.push(forward_time);
            
            // Apply repeat penalty using optimized cached implementation
            let (logits, repeat_time) = self.apply_cached_repeat_penalty(logits, &tokens)?;
            repeat_penalty_times.push(repeat_time);

            // Track token sampling
            let sampling_start = std::time::Instant::now();
            let next_token = self.logits_processor.sample(&logits)?;
            let sampling_time = sampling_start.elapsed();
            sampling_times.push(sampling_time);
            
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eot_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                write!(output, "{}", t)?;
            }
            
            let token_time = token_start.elapsed();
            token_times.push(token_time);
        }
        
        let dt = start_gen.elapsed();
        
        // Phase 3: Final decoding and output
        let decode_start = std::time::Instant::now();
        
        // Write any remaining tokens
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            write!(output, "{}", rest)?;
        }
        
        let decode_time = decode_start.elapsed();
        
        // Log performance metrics
        Self::log_performance_metrics(
            dt, generated_tokens, &token_times, &forward_times, 
            &repeat_penalty_times, &sampling_times, tokenize_time, 
            decode_time, start_time, "API"
        );
        
        Ok(())
    }
    
    // Helper function for logging performance metrics
    fn log_performance_metrics(
        generation_time: std::time::Duration,
        generated_tokens: usize,
        token_times: &[std::time::Duration],
        forward_times: &[std::time::Duration],
        repeat_penalty_times: &[std::time::Duration],
        sampling_times: &[std::time::Duration],
        tokenize_time: std::time::Duration,
        decode_time: std::time::Duration,
        start_time: std::time::Instant,
        prefix: &str,
    ) {
        // Calculate generation speed
        let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
            generated_tokens as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        };
        
        // Calculate average time per token and component breakdown
        let avg_token_time = if !token_times.is_empty() {
            token_times.iter().sum::<std::time::Duration>() / token_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let avg_forward_time = if !forward_times.is_empty() {
            forward_times.iter().sum::<std::time::Duration>() / forward_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let avg_repeat_time = if !repeat_penalty_times.is_empty() {
            repeat_penalty_times.iter().sum::<std::time::Duration>() / repeat_penalty_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let avg_sampling_time = if !sampling_times.is_empty() {
            sampling_times.iter().sum::<std::time::Duration>() / sampling_times.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        // Record detailed performance metrics
        tracing::info!("{} Text generation completed in {:.2?}", prefix, generation_time);
        tracing::info!("{} Tokens generated: {}", prefix, generated_tokens);
        tracing::info!("{} Generation speed: {:.2} tokens/second", prefix, tokens_per_second);
        tracing::info!("{} Average time per token: {:.2?}", prefix, avg_token_time);
        
        if !avg_token_time.is_zero() {
            tracing::debug!("{}  - Forward pass: {:.2?} ({:.1}%)", 
                prefix,
                avg_forward_time, 
                avg_forward_time.as_secs_f64() / avg_token_time.as_secs_f64() * 100.0
            );
            tracing::debug!("{}  - Repeat penalty: {:.2?} ({:.1}%)", 
                prefix,
                avg_repeat_time,
                avg_repeat_time.as_secs_f64() / avg_token_time.as_secs_f64() * 100.0
            );
            tracing::debug!("{}  - Sampling: {:.2?} ({:.1}%)", 
                prefix,
                avg_sampling_time,
                avg_sampling_time.as_secs_f64() / avg_token_time.as_secs_f64() * 100.0
            );
        }
        
        // Log total request time
        let total_time = start_time.elapsed();
        tracing::info!("{} Total request time: {:.2?}", prefix, total_time);
        
        if !total_time.is_zero() {
            tracing::debug!("{}  - Tokenization: {:.2?} ({:.1}%)", 
                prefix,
                tokenize_time,
                tokenize_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
            );
            tracing::debug!("{}  - Generation: {:.2?} ({:.1}%)", 
                prefix,
                generation_time,
                generation_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
            );
            tracing::debug!("{}  - Final decoding: {:.2?} ({:.1}%)", 
                prefix,
                decode_time,
                decode_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
            );
        }
    }
}