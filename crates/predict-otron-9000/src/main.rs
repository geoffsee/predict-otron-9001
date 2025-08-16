use axum::{Router, serve, http::StatusCode};
use std::env;
use tokio::net::TcpListener;
use tower::Service;
use tower_http::trace::TraceLayer;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const DEFAULT_SERVER_HOST: &str = "0.0.0.0";
const DEFAULT_SERVER_PORT: &str = "8080";

#[tokio::main]
async fn main() {
    // Initialize tracing
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

    // Create unified router by merging embeddings and inference routers
    let embeddings_router = embeddings_engine::create_embeddings_router();
    // Get the inference router directly from the inference engine
    let inference_router = inference_engine::create_inference_router();


    // Create CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);


    // Merge the routers
    let app = Router::new()
        .merge(embeddings_router)
        .merge(inference_router)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    // Server configuration
    let server_host = env::var("SERVER_HOST").unwrap_or_else(|_| DEFAULT_SERVER_HOST.to_string());
    let server_port = env::var("SERVER_PORT").unwrap_or_else(|_| DEFAULT_SERVER_PORT.to_string());
    let server_address = format!("{}:{}", server_host, server_port);

    let listener = TcpListener::bind(&server_address).await.unwrap();
    tracing::info!("Unified predict-otron-9000 server listening on {}", listener.local_addr().unwrap());
    tracing::info!("Available endpoints:");
    tracing::info!("  GET  / - Root endpoint from embeddings-engine");
    tracing::info!("  POST /v1/embeddings - Text embeddings");
    tracing::info!("  POST /v1/chat/completions - Chat completions");

    serve(listener, app).await.unwrap();
}

// Chat completions handler that properly uses the inference server crate's error handling
// This function is no longer needed as we're using the inference_engine router directly
