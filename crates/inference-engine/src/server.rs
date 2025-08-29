use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, sse::Sse, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::{self, Stream};
use tokio_stream::wrappers::UnboundedReceiverStream;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::openai_types::{ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest, ChatCompletionResponse, Delta, Message, MessageContent, Model, ModelListResponse, Usage};
use crate::Which;
use either::Either;
use serde_json::Value;
use gemma_runner::{run_gemma_api, GemmaInferenceConfig};
use llama_runner::{run_llama_inference, LlamaInferenceConfig};
// -------------------------
// Shared app state
// -------------------------

#[derive(Clone, Debug)]
pub enum ModelType {
    Gemma,
    Llama,
}

#[derive(Clone)]
pub struct AppState {
    pub model_type: ModelType,
    pub model_id: String,
    pub gemma_config: Option<GemmaInferenceConfig>,
    pub llama_config: Option<LlamaInferenceConfig>,
}

impl Default for AppState {
    fn default() -> Self {
        let gemma_config = GemmaInferenceConfig {
            model: gemma_runner::WhichModel::InstructV3_1B,
            ..Default::default()
        };
        Self {
            model_type: ModelType::Gemma,
            model_id: "gemma-3-1b-it".to_string(),
            gemma_config: Some(gemma_config),
            llama_config: None,
        }
    }
}

// -------------------------
// Helper functions
// -------------------------

fn normalize_model_id(model_id: &str) -> String {
    model_id.to_lowercase().replace("_", "-")
}

fn build_gemma_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    
    for message in messages {
        match message.role.as_str() {
            "system" => {
                if let Some(MessageContent(Either::Left(content))) = &message.content {
                    prompt.push_str(&format!("<start_of_turn>system\n{}<end_of_turn>\n", content));
                }
            }
            "user" => {
                if let Some(MessageContent(Either::Left(content))) = &message.content {
                    prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", content));
                }
            }
            "assistant" => {
                if let Some(MessageContent(Either::Left(content))) = &message.content {
                    prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", content));
                }
            }
            _ => {}
        }
    }
    
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

// -------------------------
// OpenAI-compatible handler
// -------------------------

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    if !request.stream.unwrap_or(false) {
        return Ok(chat_completions_non_streaming_proxy(state, request).await.into_response());
    }
    Ok(chat_completions_stream(state, request).await.into_response())
}

pub async fn chat_completions_non_streaming_proxy(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    // Enforce model selection behavior: reject if a different model is requested
    let configured_model = state.model_id.clone();
    let requested_model = request.model.clone();
    if requested_model.to_lowercase() != "default" {
        let normalized_requested = normalize_model_id(&requested_model);
        let normalized_configured = normalize_model_id(&configured_model);
        if normalized_requested != normalized_configured {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "Requested model '{}' is not available. This server is running '{}' only.",
                            requested_model, configured_model
                        ),
                        "type": "model_mismatch"
                    }
                })),
            ));
        }
    }

    let model_id = state.model_id.clone();
    let max_tokens = request.max_tokens.unwrap_or(1000);

    // Build prompt based on model type
    let prompt = match state.model_type {
        ModelType::Gemma => build_gemma_prompt(&request.messages),
        ModelType::Llama => {
            // For Llama, just use the last user message for now
            request.messages.last()
                .and_then(|m| m.content.as_ref())
                .and_then(|c| match c {
                    MessageContent(Either::Left(text)) => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        }
    };

    // Get streaming receiver based on model type
    let rx = match state.model_type {
        ModelType::Gemma => {
            if let Some(mut config) = state.gemma_config {
                config.prompt = prompt.clone();
                config.max_tokens = max_tokens;
                run_gemma_api(config).map_err(|e| (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": format!("Error initializing Gemma model: {}", e) }
                    }))
                ))?
            } else {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": "Gemma configuration not available" }
                    }))
                ));
            }
        }
        ModelType::Llama => {
            if let Some(mut config) = state.llama_config {
                config.prompt = prompt.clone();
                config.max_tokens = max_tokens;
                run_llama_inference(config).map_err(|e| (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": format!("Error initializing Llama model: {}", e) }
                    }))
                ))?
            } else {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": "Llama configuration not available" }
                    }))
                ));
            }
        }
    };

    // Collect all tokens from the stream
    let mut completion = String::new();
    while let Ok(token_result) = rx.recv() {
        match token_result {
            Ok(token) => completion.push_str(&token),
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": { "message": format!("Error generating text: {}", e) }
                    })),
                ));
            }
        }
    }

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4().to_string().replace('-', "")),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model_id,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: Some(MessageContent(Either::Left(completion.clone()))),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt.len() / 4,
            completion_tokens: completion.len() / 4,
            total_tokens: (prompt.len() + completion.len()) / 4,
        },
    };
    Ok(Json(response).into_response())
}

// -------------------------
// Streaming implementation
// -------------------------

pub async fn chat_completions_stream(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<Value>)> {
    handle_streaming_request(state, request).await
}

async fn handle_streaming_request(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<Value>)> {
    // Validate requested model vs configured model
    let configured_model = state.model_id.clone();
    let requested_model = request.model.clone();
    if requested_model.to_lowercase() != "default" {
        let normalized_requested = normalize_model_id(&requested_model);
        let normalized_configured = normalize_model_id(&configured_model);
        if normalized_requested != normalized_configured {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "Requested model '{}' is not available. This server is running '{}' only.",
                            requested_model, configured_model
                        ),
                        "type": "model_mismatch"
                    }
                })),
            ));
        }
    }

    // Generate a unique ID and metadata
    let response_id = format!("chatcmpl-{}", Uuid::new_v4().to_string().replace('-', ""));
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_id = state.model_id.clone();
    let max_tokens = request.max_tokens.unwrap_or(1000);

    // Build prompt based on model type
    let prompt = match state.model_type {
        ModelType::Gemma => build_gemma_prompt(&request.messages),
        ModelType::Llama => {
            // For Llama, just use the last user message for now
            request.messages.last()
                .and_then(|m| m.content.as_ref())
                .and_then(|c| match c {
                    MessageContent(Either::Left(text)) => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        }
    };
    tracing::debug!("Formatted prompt: {}", prompt);

    // Channel for streaming SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

    // Send initial role event
    let initial_chunk = ChatCompletionChunk {
        id: response_id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_id.clone(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: Delta { role: Some("assistant".to_string()), content: None },
            finish_reason: None,
        }],
    };
    if let Ok(json) = serde_json::to_string(&initial_chunk) {
        let _ = tx.send(Ok(Event::default().data(json)));
    }

    // Get streaming receiver based on model type
    let model_rx = match state.model_type {
        ModelType::Gemma => {
            if let Some(mut config) = state.gemma_config {
                config.prompt = prompt.clone();
                config.max_tokens = max_tokens;
                match run_gemma_api(config) {
                    Ok(rx) => rx,
                    Err(e) => {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": { "message": format!("Error initializing Gemma model: {}", e) }
                            }))
                        ));
                    }
                }
            } else {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": "Gemma configuration not available" }
                    }))
                ));
            }
        }
        ModelType::Llama => {
            if let Some(mut config) = state.llama_config {
                config.prompt = prompt.clone();
                config.max_tokens = max_tokens;
                match run_llama_inference(config) {
                    Ok(rx) => rx,
                    Err(e) => {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": { "message": format!("Error initializing Llama model: {}", e) }
                            }))
                        ));
                    }
                }
            } else {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": "Llama configuration not available" }
                    }))
                ));
            }
        }
    };

    // Spawn task to receive tokens from model and forward as SSE events
    let response_id_clone = response_id.clone();
    let model_id_clone = model_id.clone();
    tokio::spawn(async move {
        // Stream tokens with repetition detection
        let mut recent_tokens = Vec::new();
        let mut repetition_count = 0;
        const MAX_REPETITION_COUNT: usize = 5;
        const REPETITION_WINDOW: usize = 8;

        while let Ok(token_result) = model_rx.recv() {
            match token_result {
                Ok(token) => {
                    // Skip sending empty tokens
                    if token.is_empty() {
                        continue;
                    }

                    // Add token to recent history for repetition detection
                    recent_tokens.push(token.clone());
                    if recent_tokens.len() > REPETITION_WINDOW {
                        recent_tokens.remove(0);
                    }
                    
                    // Check for repetitive patterns
                    if recent_tokens.len() >= 4 {
                        let last_token = &recent_tokens[recent_tokens.len() - 1];
                        let second_last = &recent_tokens[recent_tokens.len() - 2];
                        
                        if last_token == second_last {
                            repetition_count += 1;
                            tracing::warn!("Detected repetition pattern: '{}' (count: {})", last_token, repetition_count);
                            
                            if repetition_count >= MAX_REPETITION_COUNT {
                                tracing::info!("Stopping generation due to excessive repetition");
                                break;
                            }
                        } else {
                            repetition_count = 0;
                        }
                    }

                    let chunk = ChatCompletionChunk {
                        id: response_id_clone.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![ChatCompletionChunkChoice {
                            index: 0,
                            delta: Delta { role: None, content: Some(token) },
                            finish_reason: None,
                        }],
                    };
                    
                    if let Ok(json) = serde_json::to_string(&chunk) {
                        let _ = tx.send(Ok(Event::default().data(json)));
                    }
                }
                Err(e) => {
                    tracing::info!("Text generation stopped: {}", e);
                    break;
                }
            }
        }

        // Send final stop chunk and DONE marker
        let final_chunk = ChatCompletionChunk {
            id: response_id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id_clone.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: Delta { role: None, content: None },
                finish_reason: Some("stop".to_string()),
            }],
        };
        if let Ok(json) = serde_json::to_string(&final_chunk) {
            let _ = tx.send(Ok(Event::default().data(json)));
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });

    // Convert receiver into a Stream for SSE
    let stream = UnboundedReceiverStream::new(rx);
    Ok(Sse::new(stream))
}



// -------------------------
// Router
// -------------------------

pub fn create_router(app_state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_headers(Any)
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .layer(cors)
        .with_state(app_state)
}

/// Handler for GET /v1/models - returns list of available models
pub async fn list_models() -> Json<ModelListResponse> {
    // Get all available model variants from the Which enum
    let models = vec![
        // Gemma models
        Model {
            id: "gemma-2b".to_string(),
            object: "model".to_string(),
            created: 1686935002, // Using same timestamp as OpenAI example
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-7b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-2b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-7b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-1.1-2b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-1.1-7b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "codegemma-2b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "codegemma-7b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "codegemma-2b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "codegemma-7b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-2-2b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-2-2b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-2-9b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-2-9b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-3-1b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        Model {
            id: "gemma-3-1b-it".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "google".to_string(),
        },
        // Llama models
        Model {
            id: "llama-3.2-1b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "meta".to_string(),
        },
        Model {
            id: "llama-3.2-1b-instruct".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "meta".to_string(),
        },
        Model {
            id: "llama-3.2-3b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "meta".to_string(),
        },
        Model {
            id: "llama-3.2-3b-instruct".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "meta".to_string(),
        },
        Model {
            id: "smollm2-135m".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "huggingface".to_string(),
        },
        Model {
            id: "smollm2-135m-instruct".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "huggingface".to_string(),
        },
        Model {
            id: "smollm2-360m".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "huggingface".to_string(),
        },
        Model {
            id: "smollm2-360m-instruct".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "huggingface".to_string(),
        },
        Model {
            id: "smollm2-1.7b".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "huggingface".to_string(),
        },
        Model {
            id: "smollm2-1.7b-instruct".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "huggingface".to_string(),
        },
        Model {
            id: "tinyllama-1.1b-chat".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "tinyllama".to_string(),
        },
    ];

    Json(ModelListResponse {
        object: "list".to_string(),
        data: models,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_types::{Message, MessageContent};
    use either::Either;

    #[test]
    fn test_build_gemma_prompt() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Some(MessageContent(Either::Left("System message".to_string()))),
                name: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(MessageContent(Either::Left("Knock knock.".to_string()))),
                name: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Some(MessageContent(Either::Left("Who's there?".to_string()))),
                name: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(MessageContent(Either::Left("Gemma.".to_string()))),
                name: None,
            },
        ];

        let prompt = build_gemma_prompt(&messages);

        let expected = "<start_of_turn>user\nSystem message\n\nKnock knock.<end_of_turn>\n\
                       <start_of_turn>model\nWho's there?<end_of_turn>\n\
                       <start_of_turn>user\nGemma.<end_of_turn>\n\
                       <start_of_turn>model\n";

        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_empty_messages() {
        let messages: Vec<Message> = vec![];
        let prompt = build_gemma_prompt(&messages);
        assert_eq!(prompt, "<start_of_turn>model\n");
    }

    #[test]
    fn test_missing_content() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: None,
                name: None,
            }
        ];

        let prompt = build_gemma_prompt(&messages);
        assert_eq!(prompt, "<start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n");
    }
}
