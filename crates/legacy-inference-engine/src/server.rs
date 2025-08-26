use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::openai_types::{ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, Message, MessageContent, Usage};
use crate::text_generation::TextGeneration;
use either::Either;

// Application state shared between handlers
#[derive(Clone)]
pub struct AppState {
    pub text_generation: Arc<Mutex<TextGeneration>>,
    pub model_id: String,
}

// Chat completions endpoint handler
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut prompt = String::new();

    // Convert messages to a prompt string
    for message in &request.messages {
        let role = &message.role;
        let content = match &message.content {
            Some(content) => match &content.0 {
                Either::Left(text) => text.clone(),
                Either::Right(_) => "".to_string(), // Handle complex content if needed
            },
            None => "".to_string(),
        };

        // Format based on role
        match role.as_str() {
            "system" => prompt.push_str(&format!("System: {}\n", content)),
            "user" => prompt.push_str(&format!("User: {}\n", content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n", content)),
            _ => prompt.push_str(&format!("{}: {}\n", role, content)),
        }
    }

    // Add the assistant prefix for the response
    prompt.push_str("Assistant: ");

    // Capture the output
    let mut output = Vec::new();
    {
        let mut text_gen = state.text_generation.lock().await;

        // Buffer to capture the output
        let mut buffer = Vec::new();

        // Run text generation
        let max_tokens = request.max_tokens.unwrap_or(1000);
        let result = text_gen.run_with_output(&prompt, max_tokens, &mut buffer);

        if let Err(e) = result {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": "The OpenAI API is currently not supported due to compatibility issues with the tensor operations. Please use the CLI mode instead with: cargo run --bin inference-engine -- --prompt \"Your prompt here\"",
                        "type": "unsupported_api"
                    }
                })),
            ));
        }

        // Convert buffer to string
        if let Ok(text) = String::from_utf8(buffer) {
            output.push(text);
        }
    }

    // Create response
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4().to_string().replace("-", "")),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: request.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: Some(MessageContent(Either::Left(output.join("")))),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt.len() / 4, // Rough estimate
            completion_tokens: output.join("").len() / 4, // Rough estimate
            total_tokens: (prompt.len() + output.join("").len()) / 4, // Rough estimate
        },
    };

    // Return the response as JSON
    Ok(Json(response))
}

// Create the router with the chat completions endpoint
pub fn create_router(app_state: AppState) -> Router {
    // CORS layer to allow requests from any origin
    let cors = CorsLayer::new()
        .allow_headers(Any)
        .allow_credentials(true)
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // OpenAI compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        // Add more endpoints as needed
        .layer(cors)
        .with_state(app_state)
}