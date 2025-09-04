pub mod llama_api;

use clap::ValueEnum;
pub use llama_api::{run_llama_inference, LlamaInferenceConfig, WhichModel};

// Re-export constants and types that might be needed
pub const EOS_TOKEN: &str = "</s>";
