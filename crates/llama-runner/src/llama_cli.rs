use crate::llama_api::{run_llama_inference, LlamaInferenceConfig, WhichModel};
use clap::Parser;
use std::io::Write;

#[derive(Parser, Debug, Default)]
#[command(author, version, about = "Fast Llama inference with Candle", long_about = None)]
struct Args {
    /// The prompt to generate text from
    #[arg(short, long, default_value = "The capital of France is")]
    prompt: String,

    /// The model to use
    #[arg(short, long, default_value = "llama-3.2-1b-instruct")]
    model: WhichModel,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens)
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Disable the key-value cache
    #[arg(long)]
    no_kv_cache: bool,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Custom model ID from HuggingFace Hub
    #[arg(long)]
    model_id: Option<String>,

    /// Model revision
    #[arg(long)]
    revision: Option<String>,

    /// Use flash attention
    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

impl Into<LlamaInferenceConfig> for Args {
    fn into(self) -> LlamaInferenceConfig {
        LlamaInferenceConfig {
            prompt: self.prompt,
            model: self.model,
            cpu: self.cpu,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            max_tokens: self.max_tokens,
            no_kv_cache: self.no_kv_cache,
            dtype: self.dtype,
            model_id: self.model_id,
            revision: self.revision,
            use_flash_attn: self.use_flash_attn,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
        }
    }
}


pub fn run_cli() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = args.into();
    let rx = run_llama_inference(cfg)?;
    for msg in rx {
        match msg {
            Ok(tok) => {
                print!("{tok}");
                let _ = std::io::stdout().flush(); // <- force it out now
            }
            Err(e) => {
                eprintln!("generation error: {e}");
                break;
            }
        }
    }
    Ok(())
}