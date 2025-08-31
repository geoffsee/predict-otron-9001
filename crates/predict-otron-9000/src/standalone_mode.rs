use crate::config::ServerConfig;
use axum::Router;
use inference_engine::AppState;

pub fn create_standalone_router(server_config: ServerConfig) -> Router {
    // Create unified router by merging embeddings and inference routers (existing behavior)
    let embeddings_router = embeddings_engine::create_embeddings_router();

    // Create AppState with correct model configuration
    let app_state = AppState::default();

    // Get the inference router directly from the inference engine
    let inference_router = inference_engine::create_router(app_state);

    // Merge the local routers
    Router::new()
        .merge(embeddings_router)
        .merge(inference_router)
}
