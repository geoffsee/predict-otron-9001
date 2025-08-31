use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

use crate::config::ServerConfig;

/// HTTP client configured for proxying requests
#[derive(Clone)]
pub struct ProxyClient {
    client: Client,
    config: ServerConfig,
}

impl ProxyClient {
    pub fn new(config: ServerConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute timeout for long-running inference
            .build()
            .expect("Failed to create HTTP client for proxy");

        Self { client, config }
    }
}

/// Create a router that proxies requests to external services in HighAvailability mode
pub fn create_proxy_router(config: ServerConfig) -> Router {
    let proxy_client = ProxyClient::new(config.clone());

    Router::new()
        .route("/v1/chat/completions", post(proxy_chat_completions))
        .route("/v1/models", get(proxy_models))
        .route("/v1/embeddings", post(proxy_embeddings))
        .with_state(proxy_client)
}

/// Proxy handler for POST /v1/chat/completions
async fn proxy_chat_completions(
    State(proxy_client): State<ProxyClient>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, StatusCode> {
    let target_url = format!(
        "{}/v1/chat/completions",
        proxy_client
            .config
            .inference_url()
            .expect("Invalid Configuration")
    );

    tracing::info!("Proxying chat completions request to: {}", target_url);

    // Extract body as bytes
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("Failed to read request body: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Check if this is a streaming request
    let is_streaming = if let Ok(body_str) = String::from_utf8(body_bytes.to_vec()) {
        if let Ok(json) = serde_json::from_str::<Value>(&body_str) {
            json.get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        } else {
            false
        }
    } else {
        false
    };

    // Forward the request
    let mut req_builder = proxy_client
        .client
        .post(&target_url)
        .body(body_bytes.to_vec());

    // Forward relevant headers
    for (name, value) in headers.iter() {
        if should_forward_header(name.as_str()) {
            req_builder = req_builder.header(name, value);
        }
    }

    match req_builder.send().await {
        Ok(response) => {
            let mut resp_builder = Response::builder().status(response.status());

            // Forward response headers
            for (name, value) in response.headers().iter() {
                if should_forward_response_header(name.as_str()) {
                    resp_builder = resp_builder.header(name, value);
                }
            }

            // Handle streaming vs non-streaming responses
            if is_streaming {
                // For streaming, we need to forward the response as-is
                match response.bytes().await {
                    Ok(body) => resp_builder
                        .header("content-type", "text/plain; charset=utf-8")
                        .header("cache-control", "no-cache")
                        .header("connection", "keep-alive")
                        .body(Body::from(body))
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR),
                    Err(e) => {
                        tracing::error!("Failed to read streaming response body: {}", e);
                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                }
            } else {
                // For non-streaming, forward the JSON response
                match response.bytes().await {
                    Ok(body) => resp_builder
                        .body(Body::from(body))
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR),
                    Err(e) => {
                        tracing::error!("Failed to read response body: {}", e);
                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                }
            }
        }
        Err(e) => {
            tracing::error!("Failed to proxy chat completions request: {}", e);
            Err(StatusCode::BAD_GATEWAY)
        }
    }
}

/// Proxy handler for GET /v1/models
async fn proxy_models(
    State(proxy_client): State<ProxyClient>,
    headers: HeaderMap,
) -> Result<Response, StatusCode> {
    let target_url = format!(
        "{}/v1/models",
        proxy_client
            .config
            .inference_url()
            .expect("Invalid Configuration Detected")
    );

    tracing::info!("Proxying models request to: {}", target_url);

    let mut req_builder = proxy_client.client.get(&target_url);

    // Forward relevant headers
    for (name, value) in headers.iter() {
        if should_forward_header(name.as_str()) {
            req_builder = req_builder.header(name, value);
        }
    }

    match req_builder.send().await {
        Ok(response) => {
            let mut resp_builder = Response::builder().status(response.status());

            // Forward response headers
            for (name, value) in response.headers().iter() {
                if should_forward_response_header(name.as_str()) {
                    resp_builder = resp_builder.header(name, value);
                }
            }

            match response.bytes().await {
                Ok(body) => resp_builder
                    .body(Body::from(body))
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR),
                Err(e) => {
                    tracing::error!("Failed to read models response body: {}", e);
                    Err(StatusCode::INTERNAL_SERVER_ERROR)
                }
            }
        }
        Err(e) => {
            tracing::error!("Failed to proxy models request: {}", e);
            Err(StatusCode::BAD_GATEWAY)
        }
    }
}

/// Proxy handler for POST /v1/embeddings
async fn proxy_embeddings(
    State(proxy_client): State<ProxyClient>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, StatusCode> {
    let target_url = format!(
        "{}/v1/embeddings",
        proxy_client
            .config
            .embeddings_url()
            .expect("Invalid Configuration Detected")
    );

    tracing::info!("Proxying embeddings request to: {}", target_url);

    // Extract body as bytes
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("Failed to read request body: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // Forward the request
    let mut req_builder = proxy_client
        .client
        .post(&target_url)
        .body(body_bytes.to_vec());

    // Forward relevant headers
    for (name, value) in headers.iter() {
        if should_forward_header(name.as_str()) {
            req_builder = req_builder.header(name, value);
        }
    }

    match req_builder.send().await {
        Ok(response) => {
            let mut resp_builder = Response::builder().status(response.status());

            // Forward response headers
            for (name, value) in response.headers().iter() {
                if should_forward_response_header(name.as_str()) {
                    resp_builder = resp_builder.header(name, value);
                }
            }

            match response.bytes().await {
                Ok(body) => resp_builder
                    .body(Body::from(body))
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR),
                Err(e) => {
                    tracing::error!("Failed to read embeddings response body: {}", e);
                    Err(StatusCode::INTERNAL_SERVER_ERROR)
                }
            }
        }
        Err(e) => {
            tracing::error!("Failed to proxy embeddings request: {}", e);
            Err(StatusCode::BAD_GATEWAY)
        }
    }
}

/// Determine if a request header should be forwarded to the target service
fn should_forward_header(header_name: &str) -> bool {
    match header_name.to_lowercase().as_str() {
        "content-type" | "content-length" | "authorization" | "user-agent" | "accept" => true,
        "host" | "connection" | "upgrade" => false, // Don't forward connection-specific headers
        _ => true,                                  // Forward other headers by default
    }
}

/// Determine if a response header should be forwarded back to the client
fn should_forward_response_header(header_name: &str) -> bool {
    match header_name.to_lowercase().as_str() {
        "content-type" | "content-length" | "cache-control" | "connection" => true,
        "server" | "date" => false, // Don't forward server-specific headers
        _ => true,                  // Forward other headers by default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ServerMode, Services};

    #[test]
    fn test_should_forward_header() {
        assert!(should_forward_header("content-type"));
        assert!(should_forward_header("authorization"));
        assert!(!should_forward_header("host"));
        assert!(!should_forward_header("connection"));
    }

    #[test]
    fn test_should_forward_response_header() {
        assert!(should_forward_response_header("content-type"));
        assert!(should_forward_response_header("cache-control"));
        assert!(!should_forward_response_header("server"));
        assert!(!should_forward_response_header("date"));
    }

    #[test]
    fn test_proxy_client_creation() {
        let config = ServerConfig {
            server_host: "127.0.0.1".to_string(),
            server_port: 8080,
            server_mode: ServerMode::HighAvailability,
            services: Some(Services {
                inference_url: Some("http://test-inference:8080".to_string()),
                embeddings_url: Some("http://test-embeddings:8080".to_string()),
            }),
        };

        let proxy_client = ProxyClient::new(config);
        assert_eq!(
            proxy_client.config.inference_url().unwrap().as_str(),
            "http://test-inference:8080"
        );
        assert_eq!(
            proxy_client.config.embeddings_url().unwrap().as_str(),
            "http://test-embeddings:8080"
        );
    }
}
