#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
mod llama_api;
mod llama_cli;

use anyhow::Result;

use crate::llama_cli::run_cli;

const EOS_TOKEN: &str = "</s>";

fn main() -> Result<()> {
    run_cli()
}
