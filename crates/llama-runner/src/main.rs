#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
mod llama_cli;
mod llama_api;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use std::io::Write;

use crate::llama_cli::run_cli;

const EOS_TOKEN: &str = "</s>";


fn main() -> Result<()> {
    run_cli()
}