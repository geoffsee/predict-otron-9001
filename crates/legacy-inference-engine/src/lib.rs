// Expose modules for testing and library usage
pub mod token_output_stream;
pub mod model;
pub mod text_generation;
pub mod utilities_lib;
pub mod openai_types;
pub mod cli;
pub mod server;

// Re-export key components for easier access
pub use model::{Model, Which};
pub use text_generation::TextGeneration;
pub use token_output_stream::TokenOutputStream;