mod middleware;
mod config;
mod proxy;

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
use config::ServerConfig;
use proxy::create_proxy_router;


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

    // Create router based on server mode
    let service_router = if server_config.clone().is_high_availability() {
        tracing::info!("Running in HighAvailability mode - proxying to external services");
        tracing::info!("  Inference service URL: {}", server_config.inference_url());
        tracing::info!("  Embeddings service URL: {}", server_config.embeddings_url());

        // Use proxy router that forwards requests to external services
        create_proxy_router(server_config.clone())
    } else {
        tracing::info!("Running in Local mode - using embedded services");

        // Create unified router by merging embeddings and inference routers (existing behavior)
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

        // Merge the local routers
        Router::new()
            .merge(embeddings_router)
            .merge(inference_router)
    };

    // Create CORS layer
    let cors = CorsLayer::new()
        .allow_headers(Any)
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Create metrics layer
    let metrics_layer = MetricsLayer::new(metrics_store);

    // Merge the service router with base routes and add middleware layers
    let app = Router::new()
        .route("/", get(|| async { "API ready. This can serve the Leptos web app, but it doesn't." }))
        .route("/health", get(|| async { "ok" }))
        .merge(service_router)
        .layer(metrics_layer)  // Add metrics tracking
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    // Server configuration
    let server_host = env::var("SERVER_HOST").unwrap_or_else(|_| {
        String::from(default_host)
    });

    let server_port = env::var("SERVER_PORT").map(|v| v.parse::<u16>().unwrap_or(default_port)).unwrap_or_else(|_| {
        default_port
    });

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
