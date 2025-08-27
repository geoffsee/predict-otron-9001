mod middleware;

use axum::{
    Router, 
    serve,
};
use std::env;
use axum::routing::get;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use inference_engine::AppState;
use middleware::{MetricsStore, MetricsLoggerFuture, MetricsLayer};

const DEFAULT_SERVER_HOST: &str = "127.0.0.1";
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


    // Initialize metrics store for performance tracking
    let metrics_store = MetricsStore::new();
    
    // Create a metrics logger that will periodically log metrics (every 60 seconds)
    let metrics_logger = MetricsLoggerFuture::new(metrics_store.clone(), 60);
    
    // Spawn the metrics logger in a background task
    tokio::spawn(metrics_logger);

    // Create unified router by merging embeddings and inference routers
    let embeddings_router = embeddings_engine::create_embeddings_router();
    
    
    // Create AppState with correct model configuration
    use inference_engine::server::{PipelineArgs, build_pipeline};
    use inference_engine::Which;
    let mut pipeline_args = PipelineArgs::default();
    pipeline_args.model_id = "google/gemma-3-1b-it".to_string();
    pipeline_args.which = Which::InstructV3_1B;
    
    let text_generation = build_pipeline(pipeline_args.clone());
    let app_state = AppState {
        text_generation: std::sync::Arc::new(tokio::sync::Mutex::new(text_generation)),
        model_id: "google/gemma-3-1b-it".to_string(),
        build_args: pipeline_args,
    };
    
    // Get the inference router directly from the inference engine
    let inference_router = inference_engine::create_router(app_state);

    // Create CORS layer
    let cors = CorsLayer::new()
        .allow_headers(Any)
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Create metrics layer
    let metrics_layer = MetricsLayer::new(metrics_store);

    // Merge the routers and add middleware layers
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/health", get(|| async { "ok" }))
        .merge(embeddings_router)
        .merge(inference_router)
        .layer(metrics_layer)  // Add metrics tracking
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    // Server configuration
    let server_host = env::var("SERVER_HOST").unwrap_or_else(|_| DEFAULT_SERVER_HOST.to_string());
    let server_port = env::var("SERVER_PORT").unwrap_or_else(|_| DEFAULT_SERVER_PORT.to_string());
    let server_address = format!("{}:{}", server_host, server_port);

    let listener = TcpListener::bind(&server_address).await.unwrap();
    tracing::info!("Unified predict-otron-9000 server listening on {}", listener.local_addr().unwrap());
    tracing::info!("Performance metrics tracking enabled - summary logs every 60 seconds");
    tracing::info!("Available endpoints:");
    tracing::info!("  GET  / - Root endpoint from embeddings-engine");
    tracing::info!("  POST /v1/embeddings - Text embeddings");
    tracing::info!("  POST /v1/chat/completions - Chat completions");

    serve(listener, app).await.unwrap();
}



// Chat completions handler that properly uses the inference server crate's error handling
// This function is no longer needed as we're using the inference_engine router directly
