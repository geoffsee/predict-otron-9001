use clap::Parser;
use crate::model::Which;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,

    /// Run in server mode with OpenAI compatible API
    #[arg(long)]
    pub server: bool,

    /// Port to use for the server
    #[arg(long, default_value_t = 3777)]
    pub port: u16,

    /// Prompt for text generation (not used in server mode)
    #[arg(long)]
    pub prompt: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    pub sample_len: usize,

    #[arg(long)]
    pub model_id: Option<String>,

    #[arg(long, default_value = "main")]
    pub revision: String,

    #[arg(long)]
    pub tokenizer_file: Option<String>,

    #[arg(long)]
    pub config_file: Option<String>,

    #[arg(long)]
    pub weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// The model to use.
    #[arg(long, default_value = "3-1b-it")]
    pub which: Which,

    #[arg(long)]
    pub use_flash_attn: bool,
}