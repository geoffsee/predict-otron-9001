use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, sse::Sse, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use candle_core::DType;
use candle_nn::VarBuilder;
use futures_util::stream::{self, Stream};
use tokio_stream::wrappers::UnboundedReceiverStream;
use std::convert::Infallible;
use std::{path::PathBuf, sync::Arc};
use tokio::sync::{Mutex, mpsc};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::openai_types::{ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest, ChatCompletionResponse, Delta, Message, MessageContent, Model, ModelListResponse, Usage};
use crate::text_generation::TextGeneration;
use crate::{utilities_lib, Model as GemmaModel, Which};
use either::Either;
use hf_hub::api::sync::{Api, ApiError};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::gemma::{Config as Config1, Model as Model1};
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};
use serde_json::Value;
// -------------------------
// Shared app state
// -------------------------

#[derive(Clone)]
pub struct AppState {
    pub text_generation: Arc<Mutex<TextGeneration>>,
    pub model_id: String,
    // Store build args to recreate TextGeneration when needed
    pub build_args: PipelineArgs,
}

impl Default for AppState {
    fn default() -> Self {
        let args = PipelineArgs::default();
        let text_generation = build_pipeline(args.clone());
        Self {
            text_generation: Arc::new(Mutex::new(text_generation)),
            model_id: args.model_id.clone(),
            build_args: args,
        }
    }
}

// -------------------------
// Pipeline configuration
// -------------------------

#[derive(Debug, Clone)]
pub struct PipelineArgs {
    pub model_id: String,
    pub which: Which,
    pub revision: Option<String>,
    pub tokenizer_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub weight_paths: Vec<PathBuf>,
    pub use_flash_attn: bool,
    pub force_cpu: bool,
    pub seed: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for PipelineArgs {
    fn default() -> Self {
        Self {
            model_id: Which::InstructV3_1B.to_model_id().to_string(),
            which: Which::InstructV3_1B,
            revision: None,
            tokenizer_path: None,
            config_path: None,
            weight_paths: Vec::new(),
            use_flash_attn: false,
            force_cpu: false,
            seed: 299792458, // Speed of light in vacuum (m/s)
            temperature: Some(0.8), // Good balance between creativity and coherence
            top_p: Some(0.9), // Keep diverse but reasonable options
            repeat_penalty: 1.2, // Stronger penalty for repetition to prevent looping
            repeat_last_n: 64, // Consider last 64 tokens for repetition
        }
    }
}

fn normalize_model_id(model_id: &str) -> String {
    if model_id.contains('/') {
        model_id.to_string()
    } else {
        format!("google/{}", model_id)
    }
}

fn ensure_repo_exists(api: &Api, model_id: &str, revision: &str) -> anyhow::Result<()> {
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    match repo.get("config.json") {
        Ok(_) => Ok(()),
        Err(e) => match e {
            ApiError::RequestError(resp) => {
                let error_str = resp.to_string();
                if error_str.contains("404") {
                    anyhow::bail!(
                        "Hugging Face model repo not found: '{model_id}' at revision '{revision}'."
                    )
                }
                Err(anyhow::Error::new(ApiError::RequestError(resp)))
            }
            other => Err(anyhow::Error::new(other)),
        },
    }
}

// -------------------------
// Pipeline builder
// -------------------------

pub fn build_pipeline(mut args: PipelineArgs) -> TextGeneration {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let api = Api::new().unwrap();
    let revision = args.revision.as_deref().unwrap_or("main");

    if args.model_id.trim().is_empty() {
        panic!("No model ID specified.");
    }
    args.model_id = normalize_model_id(&args.model_id);

    match ensure_repo_exists(&api, &args.model_id, revision) {
        Ok(_) => {}
        Err(e) => panic!("{}", e),
    };

    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        revision.to_string(),
    ));

    let tokenizer_path = args
        .tokenizer_path
        .unwrap_or_else(|| repo.get("tokenizer.json").unwrap());
    let config_path = args
        .config_path
        .unwrap_or_else(|| repo.get("config.json").unwrap());

    if !matches!(
        args.which,
        Which::Base2B
            | Which::Base7B
            | Which::Instruct2B
            | Which::Instruct7B
            | Which::InstructV1_1_2B
            | Which::InstructV1_1_7B
            | Which::CodeBase2B
            | Which::CodeBase7B
            | Which::CodeInstruct2B
            | Which::CodeInstruct7B
            | Which::BaseV2_2B
            | Which::InstructV2_2B
            | Which::BaseV2_9B
            | Which::InstructV2_9B
            | Which::BaseV3_1B
            | Which::InstructV3_1B
    ) {
        if args.model_id.contains("gemma-2-2b-it") {
            args.which = Which::InstructV2_2B;
        } else if args.model_id.contains("gemma-3-1b-it") {
            args.which = Which::InstructV3_1B;
        } else if let Ok(file) = std::fs::File::open(config_path.clone()) {
            if let Ok(cfg_val) = serde_json::from_reader::<_, serde_json::Value>(file) {
                if let Some(model_type) = cfg_val.get("model_type").and_then(|v| v.as_str()) {
                    if model_type.contains("gemma3") {
                        args.which = Which::InstructV3_1B;
                    } else if model_type.contains("gemma2") {
                        args.which = Which::InstructV2_2B;
                    } else {
                        args.which = Which::Instruct2B;
                    }
                }
            }
        }
    }

    let weight_paths = if !args.weight_paths.is_empty() {
        args.weight_paths
    } else {
        match repo.get("model.safetensors") {
            Ok(single) => vec![single],
            Err(_) => match utilities_lib::hub_load_safetensors(&repo, "model.safetensors.index.json") {
                Ok(paths) => paths,
                Err(e) => {
                    panic!("Unable to locate model weights: {}", e);
                }
            },
        }
    };

    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    let initial_device = utilities_lib::device(args.force_cpu).unwrap();
    let is_v3_model = args.which.is_v3_model();
    let is_metal = !initial_device.is_cpu()
        && candle_core::utils::metal_is_available()
        && !args.force_cpu;

    let device = if is_v3_model && is_metal {
        candle_core::Device::Cpu
    } else {
        initial_device
    };

    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device).unwrap() };

    let model = match args.which {
        Which::Base2B
        | Which::Base7B
        | Which::Instruct2B
        | Which::Instruct7B
        | Which::InstructV1_1_2B
        | Which::InstructV1_1_7B
        | Which::CodeBase2B
        | Which::CodeBase7B
        | Which::CodeInstruct2B
        | Which::CodeInstruct7B => {
            let config: Config1 = serde_json::from_reader(std::fs::File::open(config_path.clone()).unwrap()).unwrap();
            GemmaModel::V1(Model1::new(args.use_flash_attn, &config, vb).unwrap())
        }
        Which::BaseV2_2B | Which::InstructV2_2B | Which::BaseV2_9B | Which::InstructV2_9B => {
            let config: Config2 = serde_json::from_reader(std::fs::File::open(config_path.clone()).unwrap()).unwrap();
            GemmaModel::V2(Model2::new(args.use_flash_attn, &config, vb).unwrap())
        }
        Which::BaseV3_1B | Which::InstructV3_1B => {
            let config: Config3 = serde_json::from_reader(std::fs::File::open(config_path).unwrap()).unwrap();
            GemmaModel::V3(Model3::new(args.use_flash_attn, &config, vb).unwrap())
        }
    };

    TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    )
}

fn build_gemma_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    let mut system_prompt: Option<String> = None;

    for message in messages {
        let content = match &message.content {
            Some(content) => match &content.0 {
                Either::Left(text) => text.clone(),
                Either::Right(_) => "".to_string(),
            },
            None => "".to_string(),
        };

        match message.role.as_str() {
            "system" => system_prompt = Some(content),
            "user" => {
                prompt.push_str("<start_of_turn>user\n");
                if let Some(sys_prompt) = system_prompt.take() {
                    prompt.push_str(&sys_prompt);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&content);
                prompt.push_str("<end_of_turn>\n");
            }
            "assistant" => {
                prompt.push_str("<start_of_turn>model\n");
                prompt.push_str(&content);
                prompt.push_str("<end_of_turn>\n");
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
    let prompt = build_gemma_prompt(&request.messages);

    // Enforce model selection behavior: reject if a different model is requested
    let configured_model = state.build_args.model_id.clone();
    let requested_model = request.model.clone();
    if requested_model.to_lowercase() != "default" {
        let normalized_requested = normalize_model_id(&requested_model);
        if normalized_requested != configured_model {
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

    let mut buffer = Vec::new();
    {
        let mut text_gen = state.text_generation.lock().await;
        // Reset per-request state without rebuilding the whole pipeline
        text_gen.reset_state();
        let max_tokens = request.max_tokens.unwrap_or(1000);
        if let Err(e) = text_gen.run_with_output(&prompt, max_tokens, &mut buffer) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": { "message": format!("Error generating text: {}", e) }
                })),
            ));
        }
    }

    let completion = match String::from_utf8(buffer) {
        Ok(s) => s,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": { "message": format!("UTF-8 conversion error: {}", e) }
                })),
            ));
        }
    };

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
    let configured_model = state.build_args.model_id.clone();
    let requested_model = request.model.clone();
    if requested_model.to_lowercase() != "default" {
        let normalized_requested = normalize_model_id(&requested_model);
        if normalized_requested != configured_model {
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

    // Build prompt
    let prompt = build_gemma_prompt(&request.messages);
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

    // Spawn generation task that streams tokens as they are generated
    let state_clone = state.clone();
    let response_id_clone = response_id.clone();
    tokio::spawn(async move {
        let max_tokens = request.max_tokens.unwrap_or(1000);
        let mut text_gen = state_clone.text_generation.lock().await;
        text_gen.reset_state();

        // Stream tokens via callback with repetition detection
        let mut recent_tokens = Vec::new();
        let mut repetition_count = 0;
        const MAX_REPETITION_COUNT: usize = 5; // Stop after 5 consecutive repetitions
        const REPETITION_WINDOW: usize = 8; // Look at last 8 tokens for patterns
        
        let result = text_gen.run_with_streaming(&prompt, max_tokens, |token| {
            // Debug log to verify token content
            tracing::debug!("Streaming token: '{}'", token);
            
            // Skip sending empty tokens
            if token.is_empty() {
                tracing::debug!("Skipping empty token");
                return Ok(());
            }
            
            // Add token to recent history for repetition detection
            recent_tokens.push(token.to_string());
            if recent_tokens.len() > REPETITION_WINDOW {
                recent_tokens.remove(0);
            }
            
            // Check for repetitive patterns
            if recent_tokens.len() >= 4 {
                let last_token = &recent_tokens[recent_tokens.len() - 1];
                let second_last = &recent_tokens[recent_tokens.len() - 2];
                
                // Check if we're repeating the same token or pattern
                if last_token == second_last || 
                   (last_token.trim() == "plus" && second_last.trim() == "plus") ||
                   (recent_tokens.len() >= 6 && 
                    recent_tokens[recent_tokens.len()-3..].iter().all(|t| t.trim() == "plus" || t.trim().is_empty())) {
                    repetition_count += 1;
                    tracing::warn!("Detected repetition pattern: '{}' (count: {})", last_token, repetition_count);
                    
                    if repetition_count >= MAX_REPETITION_COUNT {
                        tracing::info!("Stopping generation due to excessive repetition");
                        return Err(anyhow::Error::msg("Repetition detected - stopping generation"));
                    }
                } else {
                    repetition_count = 0; // Reset counter if pattern breaks
                }
            }
            
            let chunk = ChatCompletionChunk {
                id: response_id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: Delta { role: None, content: Some(token.to_string()) },
                    finish_reason: None,
                }],
            };
            if let Ok(json) = serde_json::to_string(&chunk) {
                tracing::debug!("Sending chunk with content: '{}'", token);
                let _ = tx.send(Ok(Event::default().data(json)));
            }
            Ok(())
        }).await;
        
        // Log result of generation
        match result {
            Ok(_) => tracing::debug!("Text generation completed successfully"),
            Err(e) => tracing::info!("Text generation stopped: {}", e),
        }

        // Send final stop chunk and DONE marker
        let final_chunk = ChatCompletionChunk {
            id: response_id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id.clone(),
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
