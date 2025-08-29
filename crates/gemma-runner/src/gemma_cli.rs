use std::io::Write;
use clap::Parser;
use crate::gemma_api::{run_gemma_api, GemmaInferenceConfig, WhichModel};

#[derive(Parser, Debug)]
#[command(author, version, about = "Fast Gemma inference with Candle", long_about = None)]
pub struct Args {
    /// The prompt to generate text from
    #[arg(short, long, default_value = "The capital of France is")]
    pub(crate) prompt: String,

    /// The model to use
    #[arg(short, long, default_value = "gemma-2-2b")]
    pub(crate) model: WhichModel,

    /// Run on CPU rather than GPU
    #[arg(long)]
    pub(crate) cpu: bool,

    /// The temperature used to generate samples
    #[arg(short, long)]
    pub(crate) temperature: Option<f64>,

    /// Nucleus sampling probability cutoff
    #[arg(long)]
    pub(crate) top_p: Option<f64>,

    /// The seed to use when generating random samples
    #[arg(long, default_value_t = 299792458)]
    pub(crate) seed: u64,

    /// The length of the sample to generate (in tokens)
    #[arg(short = 'n', long, default_value_t = 100)]
    pub(crate) max_tokens: usize,

    /// Use different dtype than default
    #[arg(long)]
    pub(crate) dtype: Option<String>,

    /// Custom model ID from HuggingFace Hub
    #[arg(long)]
    pub(crate) model_id: Option<String>,

    /// Model revision
    #[arg(long, default_value = "main")]
    pub(crate) revision: String,

    /// Use flash attention
    #[arg(long)]
    pub(crate) use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty
    #[arg(long, default_value_t = 1.1)]
    pub(crate) repeat_penalty: f32,

    /// The context size to consider for the repeat penalty
    #[arg(long, default_value_t = 64)]
    pub(crate) repeat_last_n: usize,

    /// Enable tracing
    #[arg(long)]
    pub(crate) tracing: bool,
}

pub fn run_cli() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg = GemmaInferenceConfig {
        tracing: args.tracing,
        prompt: args.prompt,
        model: args.model,
        cpu: args.cpu,
        dtype: args.dtype,
        model_id: args.model_id,
        revision: args.revision,
        use_flash_attn: args.use_flash_attn,
        seed: args.seed,
        temperature: args.temperature.unwrap_or(0.8),
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        max_tokens: args.max_tokens,
    };
    let rx = run_gemma_api(cfg)?;
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