use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, sse::Sse, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::{self, Stream};
use std::convert::Infallible;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::openai_types::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
    ChatCompletionResponse, Delta, Message, MessageContent, Model, ModelListResponse, Usage,
};
use crate::Which;
use either::Either;
use embeddings_engine::models_list;
use gemma_runner::{run_gemma_api, GemmaInferenceConfig, WhichModel};
use llama_runner::{run_llama_inference, LlamaInferenceConfig};
use serde_json::Value;
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
    pub model_type: Option<ModelType>,
    pub model_id: String,
    pub gemma_config: Option<GemmaInferenceConfig>,
    pub llama_config: Option<LlamaInferenceConfig>,
}


impl Default for AppState {
    fn default() -> Self {
        // Configure a default model to prevent 503 errors from the chat-ui
        // This can be overridden by environment variables if needed
        let default_model_id = std::env::var("DEFAULT_MODEL").unwrap_or_else(|_| "gemma-3-1b-it".to_string());
        
        let gemma_config = GemmaInferenceConfig {
            model: None,
            ..Default::default()
        };

        Self {
            model_type: None,
            model_id: default_model_id,
            gemma_config: Some(gemma_config),
            llama_config: None,
        }
    }
}

// -------------------------
// Helper functions
// -------------------------

fn model_id_to_which(model_id: &str) -> Option<Which> {
    let normalized = normalize_model_id(model_id);
    match normalized.as_str() {
        "gemma-2b" => Some(Which::Base2B),
        "gemma-7b" => Some(Which::Base7B),
        "gemma-2b-it" => Some(Which::Instruct2B),
        "gemma-7b-it" => Some(Which::Instruct7B),
        "gemma-1.1-2b-it" => Some(Which::InstructV1_1_2B),
        "gemma-1.1-7b-it" => Some(Which::InstructV1_1_7B),
        "codegemma-2b" => Some(Which::CodeBase2B),
        "codegemma-7b" => Some(Which::CodeBase7B),
        "codegemma-2b-it" => Some(Which::CodeInstruct2B),
        "codegemma-7b-it" => Some(Which::CodeInstruct7B),
        "gemma-2-2b" => Some(Which::BaseV2_2B),
        "gemma-2-2b-it" => Some(Which::InstructV2_2B),
        "gemma-2-9b" => Some(Which::BaseV2_9B),
        "gemma-2-9b-it" => Some(Which::InstructV2_9B),
        "gemma-3-1b" => Some(Which::BaseV3_1B),
        "gemma-3-1b-it" => Some(Which::InstructV3_1B),
        "llama-3.2-1b" => Some(Which::Llama32_1B),
        "llama-3.2-1b-instruct" => Some(Which::Llama32_1BInstruct),
        "llama-3.2-3b" => Some(Which::Llama32_3B),
        "llama-3.2-3b-instruct" => Some(Which::Llama32_3BInstruct),
        _ => None,
    }
}




fn normalize_model_id(model_id: &str) -> String {
    model_id.to_lowercase().replace("_", "-")
}

fn build_gemma_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();

    for message in messages {
        match message.role.as_str() {
            "system" => {
                if let Some(MessageContent(Either::Left(content))) = &message.content {
                    prompt.push_str(&format!(
                        "<start_of_turn>system\n{}<end_of_turn>\n",
                        content
                    ));
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
        return Ok(chat_completions_non_streaming_proxy(state, request)
            .await
            .into_response());
    }
    Ok(chat_completions_stream(state, request)
        .await
        .into_response())
}

pub async fn chat_completions_non_streaming_proxy(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    // Use the model specified in the request
    let model_id = request.model.clone();
    let which_model = model_id_to_which(&model_id);
    
    // Validate that the requested model is supported
    let which_model = match which_model {
        Some(model) => model,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Unsupported model: {}", model_id),
                        "type": "model_not_supported"
                    }
                })),
            ));
        }
    };
    let max_tokens = request.max_tokens.unwrap_or(1000);

    // Build prompt based on model type
    let prompt = if which_model.is_llama_model() {
        // For Llama, just use the last user message for now
        request
            .messages
            .last()
            .and_then(|m| m.content.as_ref())
            .and_then(|c| match c {
                MessageContent(Either::Left(text)) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    } else {
        build_gemma_prompt(&request.messages)
    };

    // Get streaming receiver based on model type
    let rx = if which_model.is_llama_model() {
        // Create Llama configuration dynamically
        let llama_model = match which_model {
            Which::Llama32_1B => llama_runner::WhichModel::Llama32_1B,
            Which::Llama32_1BInstruct => llama_runner::WhichModel::Llama32_1BInstruct,
            Which::Llama32_3B => llama_runner::WhichModel::Llama32_3B,
            Which::Llama32_3BInstruct => llama_runner::WhichModel::Llama32_3BInstruct,
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": { "message": format!("Model {} is not a Llama model", model_id) }
                    }))
                ));
            }
        };
        let mut config = LlamaInferenceConfig::new(llama_model);
        config.prompt = prompt.clone();
        config.max_tokens = max_tokens;
        run_llama_inference(config).map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": { "message": format!("Error initializing Llama model: {}", e) }
            }))
        ))?
    } else {
        // Create Gemma configuration dynamically
        let gemma_model = match which_model {
            Which::Base2B => gemma_runner::WhichModel::Base2B,
            Which::Base7B => gemma_runner::WhichModel::Base7B,
            Which::Instruct2B => gemma_runner::WhichModel::Instruct2B,
            Which::Instruct7B => gemma_runner::WhichModel::Instruct7B,
            Which::InstructV1_1_2B => gemma_runner::WhichModel::InstructV1_1_2B,
            Which::InstructV1_1_7B => gemma_runner::WhichModel::InstructV1_1_7B,
            Which::CodeBase2B => gemma_runner::WhichModel::CodeBase2B,
            Which::CodeBase7B => gemma_runner::WhichModel::CodeBase7B,
            Which::CodeInstruct2B => gemma_runner::WhichModel::CodeInstruct2B,
            Which::CodeInstruct7B => gemma_runner::WhichModel::CodeInstruct7B,
            Which::BaseV2_2B => gemma_runner::WhichModel::BaseV2_2B,
            Which::InstructV2_2B => gemma_runner::WhichModel::InstructV2_2B,
            Which::BaseV2_9B => gemma_runner::WhichModel::BaseV2_9B,
            Which::InstructV2_9B => gemma_runner::WhichModel::InstructV2_9B,
            Which::BaseV3_1B => gemma_runner::WhichModel::BaseV3_1B,
            Which::InstructV3_1B => gemma_runner::WhichModel::InstructV3_1B,
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": { "message": format!("Model {} is not a Gemma model", model_id) }
                    }))
                ));
            }
        };
        
        let mut config = GemmaInferenceConfig {
            model: Some(gemma_model),
            ..Default::default()
        };
        config.prompt = prompt.clone();
        config.max_tokens = max_tokens;
        run_gemma_api(config).map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": { "message": format!("Error initializing Gemma model: {}", e) }
            }))
        ))?
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
    // Use the model specified in the request
    let model_id = request.model.clone();
    let which_model = model_id_to_which(&model_id);
    
    // Validate that the requested model is supported
    let which_model = match which_model {
        Some(model) => model,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Unsupported model: {}", model_id),
                        "type": "model_not_supported"
                    }
                })),
            ));
        }
    };

    // Generate a unique ID and metadata
    let response_id = format!("chatcmpl-{}", Uuid::new_v4().to_string().replace('-', ""));
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let max_tokens = request.max_tokens.unwrap_or(1000);

    // Build prompt based on model type
    let prompt = if which_model.is_llama_model() {
        // For Llama, just use the last user message for now
        request
            .messages
            .last()
            .and_then(|m| m.content.as_ref())
            .and_then(|c| match c {
                MessageContent(Either::Left(text)) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    } else {
        build_gemma_prompt(&request.messages)
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
            delta: Delta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    if let Ok(json) = serde_json::to_string(&initial_chunk) {
        let _ = tx.send(Ok(Event::default().data(json)));
    }

    // Get streaming receiver based on model type
    let model_rx = if which_model.is_llama_model() {
        // Create Llama configuration dynamically
        let llama_model = match which_model {
            Which::Llama32_1B => llama_runner::WhichModel::Llama32_1B,
            Which::Llama32_1BInstruct => llama_runner::WhichModel::Llama32_1BInstruct,
            Which::Llama32_3B => llama_runner::WhichModel::Llama32_3B,
            Which::Llama32_3BInstruct => llama_runner::WhichModel::Llama32_3BInstruct,
            _ => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": format!("Model {} is not a Llama model", model_id) }
                    }))
                ));
            }
        };
        let mut config = LlamaInferenceConfig::new(llama_model);
        config.prompt = prompt.clone();
        config.max_tokens = max_tokens;
        match run_llama_inference(config) {
            Ok(rx) => rx,
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": format!("Error initializing Llama model: {}", e) }
                    })),
                ));
            }
        }
    } else {
        // Create Gemma configuration dynamically
        let gemma_model = match which_model {
            Which::Base2B => gemma_runner::WhichModel::Base2B,
            Which::Base7B => gemma_runner::WhichModel::Base7B,
            Which::Instruct2B => gemma_runner::WhichModel::Instruct2B,
            Which::Instruct7B => gemma_runner::WhichModel::Instruct7B,
            Which::InstructV1_1_2B => gemma_runner::WhichModel::InstructV1_1_2B,
            Which::InstructV1_1_7B => gemma_runner::WhichModel::InstructV1_1_7B,
            Which::CodeBase2B => gemma_runner::WhichModel::CodeBase2B,
            Which::CodeBase7B => gemma_runner::WhichModel::CodeBase7B,
            Which::CodeInstruct2B => gemma_runner::WhichModel::CodeInstruct2B,
            Which::CodeInstruct7B => gemma_runner::WhichModel::CodeInstruct7B,
            Which::BaseV2_2B => gemma_runner::WhichModel::BaseV2_2B,
            Which::InstructV2_2B => gemma_runner::WhichModel::InstructV2_2B,
            Which::BaseV2_9B => gemma_runner::WhichModel::BaseV2_9B,
            Which::InstructV2_9B => gemma_runner::WhichModel::InstructV2_9B,
            Which::BaseV3_1B => gemma_runner::WhichModel::BaseV3_1B,
            Which::InstructV3_1B => gemma_runner::WhichModel::InstructV3_1B,
            _ => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": format!("Model {} is not a Gemma model", model_id) }
                    }))
                ));
            }
        };
        
        let mut config = GemmaInferenceConfig {
            model: Some(gemma_model),
            ..Default::default()
        };
        config.prompt = prompt.clone();
        config.max_tokens = max_tokens;
        match run_gemma_api(config) {
            Ok(rx) => rx,
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": { "message": format!("Error initializing Gemma model: {}", e) }
                    })),
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
                            tracing::warn!(
                                "Detected repetition pattern: '{}' (count: {})",
                                last_token,
                                repetition_count
                            );

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
                            delta: Delta {
                                role: None,
                                content: Some(token),
                            },
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
                delta: Delta {
                    role: None,
                    content: None,
                },
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
    let which_variants = vec![
        Which::Base2B,
        Which::Base7B,
        Which::Instruct2B,
        Which::Instruct7B,
        Which::InstructV1_1_2B,
        Which::InstructV1_1_7B,
        Which::CodeBase2B,
        Which::CodeBase7B,
        Which::CodeInstruct2B,
        Which::CodeInstruct7B,
        Which::BaseV2_2B,
        Which::InstructV2_2B,
        Which::BaseV2_9B,
        Which::InstructV2_9B,
        Which::BaseV3_1B,
        Which::InstructV3_1B,
        Which::Llama32_1B,
        Which::Llama32_1BInstruct,
        Which::Llama32_3B,
        Which::Llama32_3BInstruct,
    ];



    let mut models: Vec<Model> = which_variants.into_iter().map(|which| {
        let meta = which.meta();
        let model_id = match which {
            Which::Base2B => "gemma-2b",
            Which::Base7B => "gemma-7b",
            Which::Instruct2B => "gemma-2b-it",
            Which::Instruct7B => "gemma-7b-it",
            Which::InstructV1_1_2B => "gemma-1.1-2b-it",
            Which::InstructV1_1_7B => "gemma-1.1-7b-it",
            Which::CodeBase2B => "codegemma-2b",
            Which::CodeBase7B => "codegemma-7b",
            Which::CodeInstruct2B => "codegemma-2b-it",
            Which::CodeInstruct7B => "codegemma-7b-it",
            Which::BaseV2_2B => "gemma-2-2b",
            Which::InstructV2_2B => "gemma-2-2b-it",
            Which::BaseV2_9B => "gemma-2-9b",
            Which::InstructV2_9B => "gemma-2-9b-it",
            Which::BaseV3_1B => "gemma-3-1b",
            Which::InstructV3_1B => "gemma-3-1b-it",
            Which::Llama32_1B => "llama-3.2-1b",
            Which::Llama32_1BInstruct => "llama-3.2-1b-instruct",
            Which::Llama32_3B => "llama-3.2-3b",
            Which::Llama32_3BInstruct => "llama-3.2-3b-instruct",
        };

        let owned_by = if meta.id.starts_with("google/") {
            "google"
        } else if meta.id.starts_with("meta-llama/") {
            "meta"
        } else {
            "unknown"
        };

        Model {
            id: model_id.to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: owned_by.to_string(),
        }
    }).collect();

    // Get embeddings models and convert them to inference Model format
    let embeddings_response = models_list().await;
    let embeddings_models: Vec<Model> = embeddings_response.0.data.into_iter().map(|embedding_model| {
        Model {
            id: embedding_model.id,
            object: embedding_model.object,
            created: 1686935002,
            owned_by: format!("{} - {}", embedding_model.owned_by, embedding_model.description),
        }
    }).collect();

    // Add embeddings models to the main models list
    models.extend(embeddings_models);

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

        let expected = "<start_of_turn>system\nSystem message<end_of_turn>\n<start_of_turn>user\nKnock knock.<end_of_turn>\n<start_of_turn>model\nWho's there?<end_of_turn>\n<start_of_turn>user\nGemma.<end_of_turn>\n<start_of_turn>model\n";

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
        let messages = vec![Message {
            role: "user".to_string(),
            content: None,
            name: None,
        }];

        let prompt = build_gemma_prompt(&messages);
        assert_eq!(prompt, "<start_of_turn>model\n");
    }
}
