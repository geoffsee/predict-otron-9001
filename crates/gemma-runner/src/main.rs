#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
mod gemma_api;
mod gemma_cli;

use anyhow::Error;
use clap::{Parser, ValueEnum};

use crate::gemma_cli::run_cli;
use std::io::Write;

/// just a placeholder, not used for anything
fn main() -> std::result::Result<(), Error> {
    run_cli()
}
