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
pub use server::{AppState, create_router};

use axum::{Json, http::StatusCode, routing::post, Router};
use serde_json;
use std::env;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Server configuration constants
pub const DEFAULT_SERVER_HOST: &str = "0.0.0.0";
pub const DEFAULT_SERVER_PORT: &str = "8080";

/// Get server configuration from environment variables with defaults
pub fn get_server_config() -> (String, String, String) {
    let server_host = env::var("SERVER_HOST").unwrap_or_else(|_| DEFAULT_SERVER_HOST.to_string());
    let server_port = env::var("SERVER_PORT").unwrap_or_else(|_| DEFAULT_SERVER_PORT.to_string());
    let server_address = format!("{}:{}", server_host, server_port);
    (server_host, server_port, server_address)
}

/// Initialize tracing with configurable log levels
pub fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!(
                    "{}=debug,tower_http=debug,axum::rejection=trace",
                    env!("CARGO_CRATE_NAME")
                )
                .into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

/// Create a simplified inference router that returns appropriate error messages
/// indicating that full model loading is required for production use
pub fn create_inference_router() -> Router {
    Router::new()
        .route("/v1/chat/completions", post(simplified_chat_completions))
}

async fn simplified_chat_completions(
    axum::Json(request): axum::Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    // Return the same error message as the actual server implementation
    // to indicate that full inference functionality requires proper model initialization
    Err((
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "message": "The OpenAI API is currently not supported due to compatibility issues with the tensor operations. Please use the CLI mode instead with: cargo run --bin inference-engine -- --prompt \"Your prompt here\"",
                "type": "unsupported_api"
            }
        })),
    ))
}