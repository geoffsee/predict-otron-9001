use async_openai::types::{CreateEmbeddingRequest, EmbeddingInput};
use axum::{
    Json, Router,
    response::Json as ResponseJson,
    routing::{get, post},
};
use std::env;
use tower_http::trace::TraceLayer;
use tracing;

const DEFAULT_SERVER_HOST: &str = "127.0.0.1";
const DEFAULT_SERVER_PORT: &str = "8080";

use embeddings_engine;

async fn embeddings_create(
    Json(payload): Json<CreateEmbeddingRequest>,
) -> Result<ResponseJson<serde_json::Value>, axum::response::Response> {
    match embeddings_engine::embeddings_create(Json(payload)).await {
        Ok(response) => Ok(response),
        Err((status_code, message)) => {
            Err(axum::response::Response::builder()
                .status(status_code)
                .body(axum::body::Body::from(message))
                .unwrap())
        }
    }
}

async fn models_list() -> ResponseJson<embeddings_engine::ModelsResponse> {
    embeddings_engine::models_list().await
}

fn create_app() -> Router {
    Router::new()
        .route("/v1/embeddings", post(embeddings_create))
        .route("/v1/models", get(models_list))
        .layer(TraceLayer::new_for_http())
}
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // axum logs rejections from built-in extractors with the `axum::rejection`
                // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                format!(
                    "{}=debug,tower_http=debug,axum::rejection=trace",
                    env!("CARGO_CRATE_NAME")
                )
                .into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    let app = create_app();

    let server_host = env::var("SERVER_HOST").unwrap_or_else(|_| DEFAULT_SERVER_HOST.to_string());
    let server_port = env::var("SERVER_PORT").unwrap_or_else(|_| DEFAULT_SERVER_PORT.to_string());
    let server_address = format!("{}:{}", server_host, server_port);
    let listener = tokio::net::TcpListener::bind(server_address).await.unwrap();
    tracing::info!("Listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_embeddings_create() {
        // Start a test server
        let app = create_app();

        // Use the OpenAI client with our test server

        let body = CreateEmbeddingRequest {
            model: "nomic-text-embed".to_string(),
            input: EmbeddingInput::from(vec![
                "The food was delicious and the waiter...".to_string(),
            ]),
            encoding_format: None,
            user: None,
            dimensions: Some(768),
        };

        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method(axum::http::Method::POST)
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();

        let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(response_json["object"], "list");
        assert!(response_json["data"].is_array());
        assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
        assert_eq!(response_json["model"], "nomic-text-embed");

        let embedding_obj = &response_json["data"][0];
        assert_eq!(embedding_obj["object"], "embedding");
        assert_eq!(embedding_obj["index"], 0);
        assert!(embedding_obj["embedding"].is_array());

        let embedding = embedding_obj["embedding"].as_array().unwrap();
        assert_eq!(embedding.len(), 768);
    }
}
