mod config;
mod ha_mode;
mod middleware;
mod standalone_mode;

use crate::standalone_mode::create_standalone_router;
use axum::handler::Handler;
use axum::http::StatusCode as AxumStatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Router, ServiceExt, http::Uri, response::Html, serve};
use config::ServerConfig;
use ha_mode::create_ha_router;
use inference_engine::AppState;
use log::info;
use middleware::{MetricsLayer, MetricsLoggerFuture, MetricsStore};
use mime_guess::from_path;
use rust_embed::Embed;
use std::env;
use std::path::Component::ParentDir;
use tokio::net::TcpListener;
use tower::MakeService;
use tower_http::classify::ServerErrorsFailureClass::StatusCode;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Embed)]
#[folder = "../../target/site"]
#[include = "*.js"]
#[include = "*.wasm"]
#[include = "*.css"]
#[include = "*.ico"]
struct Asset;

async fn static_handler(uri: Uri) -> axum::response::Response {
    // Strip the leading `/`
    let path = uri.path().trim_start_matches('/');

    tracing::info!("Static file: {}", &path);

    // If root is requested, serve index.html
    let path = if path.is_empty() { "index.html" } else { path };

    match Asset::get(path) {
        Some(content) => {
            let body = content.data.into_owned();
            let mime = from_path(path).first_or_octet_stream();

            ([(header::CONTENT_TYPE, mime.as_ref())], body).into_response()
        }
        None => (AxumStatusCode::NOT_FOUND, "404 Not Found").into_response(),
    }
}

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

    // Initialize metrics store for performance tracking
    let metrics_store = MetricsStore::new();

    // Create a metrics logger that will periodically log metrics (every 60 seconds)
    let metrics_logger = MetricsLoggerFuture::new(metrics_store.clone(), 60);

    // Spawn the metrics logger in a background task
    tokio::spawn(metrics_logger);

    // Load server configuration from environment variable
    let server_config = ServerConfig::from_env();

    // Extract the server_host and server_port before potentially moving server_config
    let default_host = server_config.server_host.clone();
    let default_port = server_config.server_port;

    let service_router = match server_config.clone().is_high_availability() {
        Ok(is_ha) => {
            if is_ha {
                log_config(server_config.clone());
                create_ha_router(server_config.clone())
            } else {
                log_config(server_config.clone());
                create_standalone_router(server_config)
            }
        }
        Err(error) => {
            panic!("{}", error);
        }
    };

    // Create CORS layer
    let cors = CorsLayer::new()
        .allow_headers(Any)
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Create metrics layer
    let metrics_layer = MetricsLayer::new(metrics_store);

    let leptos_config = chat_ui::app::AppConfig::default();

    // Create the leptos router for the web frontend
    let leptos_router = chat_ui::app::create_router(leptos_config.config.leptos_options);

    // Merge the service router with base routes and add middleware layers
    let app = Router::new()
        .route("/pkg/{*path}", get(static_handler))
        .route("/health", get(|| async { "ok" }))
        .merge(service_router)
        .merge(leptos_router)
        .layer(metrics_layer) // Add metrics tracking
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    // Server configuration
    let server_host = env::var("SERVER_HOST").unwrap_or_else(|_| String::from(default_host));

    let server_port = env::var("SERVER_PORT")
        .map(|v| v.parse::<u16>().unwrap_or(default_port))
        .unwrap_or_else(|_| default_port);

    let server_address = format!("{}:{}", server_host, server_port);

    let listener = TcpListener::bind(&server_address).await.unwrap();
    tracing::info!(
        "Unified predict-otron-9000 server listening on {}",
        listener.local_addr().unwrap()
    );
    tracing::info!("Performance metrics tracking enabled - summary logs every 60 seconds");
    tracing::info!("Available endpoints:");
    tracing::info!("  GET  / - Leptos chat web application");
    tracing::info!("  GET  /health - Health check");
    tracing::info!("  POST /v1/embeddings - Text embeddings API");
    tracing::info!("  POST /v1/chat/completions - Chat completions API");

    serve(listener, app.into_make_service()).await.unwrap();
}

fn log_config(config: ServerConfig) {
    match config.is_high_availability() {
        Ok(is_high) => {
            if is_high {
                tracing::info!("Running in HighAvailability mode - proxying to external services");
                tracing::info!("Inference service URL: {}", config.inference_url().unwrap());
                tracing::info!(
                    "Embeddings service URL: {}",
                    config.embeddings_url().unwrap()
                );
            } else {
                tracing::info!("Running in Standalone mode");
            }
        }
        Err(error) => {
            panic!("{}", error);
        }
    }
}

// Chat completions handler that properly uses the inference server crate's error handling
// This function is no longer needed as we're using the inference_engine router directly
