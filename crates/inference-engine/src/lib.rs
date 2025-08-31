// Expose modules for testing and library usage
pub mod model;
pub mod openai_types;
// pub mod cli;
pub mod inference;
pub mod server;

// Re-export key components for easier access
pub use inference::ModelInference;
pub use model::{Model, Which};
pub use server::{create_router, AppState};

use std::env;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Server configuration constants
pub const DEFAULT_SERVER_HOST: &str = "127.0.0.1";
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
