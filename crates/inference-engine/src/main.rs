use inference_engine::{create_router, init_tracing, get_server_config, AppState};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    
    let app_state = AppState::default();
    let app = create_router(app_state);
    
    let (server_host, server_port, server_address) = get_server_config();
    let listener = TcpListener::bind(&server_address).await?;
    
    info!("Inference Engine server starting on http://{}", server_address);
    info!("Available endpoints:");
    info!("  POST /v1/chat/completions - OpenAI-compatible chat completions");
    info!("  GET  /v1/models         - List available models");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}