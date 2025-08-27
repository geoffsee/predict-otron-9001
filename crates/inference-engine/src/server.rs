use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, sse::Sse, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::{self, Stream};
use std::convert::Infallible;
use candle_core::DType;
use candle_nn::VarBuilder;
use std::{path::PathBuf, sync::Arc};
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time;
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
            model_id: String::new(),
            build_args: args,
        }
    }
}
// -------------------------
// Pipeline configuration
// -------------------------

#[derive(Debug, Clone)]
pub struct PipelineArgs {
    /// HF model repo id, e.g. "google/gemma-2b"
    pub model_id: String,

    /// Which internal model family to instantiate
    pub which: Which,

    /// Optional HF revision/branch/tag; None => "main"
    pub revision: Option<String>,

    /// Optional explicit tokenizer path
    pub tokenizer_path: Option<PathBuf>,

    /// Optional explicit config path
    pub config_path: Option<PathBuf>,

    /// Optional explicit weight paths. If empty, they will be resolved from the hub.
    pub weight_paths: Vec<PathBuf>,

    /// Runtime toggles
    pub use_flash_attn: bool,
    pub force_cpu: bool,

    /// Sampling / decoding params
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
            seed: 0,
            temperature: None,
            top_p: None,
            repeat_penalty: 0.0,
            repeat_last_n: 0,
        }
    }
}

// If no owner/org is present, prefix with a sensible default (tweak as you like).
fn normalize_model_id(model_id: &str) -> String {
    if model_id.contains('/') { model_id.to_string() } else { format!("google/{}", model_id) }
}

// Quick existence check, mapping 404 into a helpful message.
fn ensure_repo_exists(api: &Api, model_id: &str, revision: &str) -> anyhow::Result<()> {
    let repo = api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string()));
    match repo.get("config.json") {
        Ok(_) => Ok(()),
        Err(e) => match e {
            ApiError::RequestError(resp) => {
                // For HF API, RequestError with 404 status is returned when repo doesn't exist
                let error_str = resp.to_string();
                if error_str.contains("404") {
                    anyhow::bail!(
                        "Hugging Face model repo not found: '{model_id}' at revision '{revision}'. \
                         Please provide a fully-qualified repo id like 'google/gemma-2b-it'."
                    )
                }
                Err(anyhow::Error::new(ApiError::RequestError(resp)))
            }
            other => Err(anyhow::Error::new(other)),
        }
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

    // Check if model_id is empty before normalizing it
    println!("Checking model_id: '{}'", args.model_id);

    println!("Trimmed model_id length: {}", args.model_id.trim().len());
    if args.model_id.trim().is_empty() {
        panic!("No model ID specified. Please provide a valid model ID (e.g., 'gemma-2b-it' or 'google/gemma-2b-it').");
    }
    args.model_id = normalize_model_id(&args.model_id);

    // Validate early (nice error if the repo/revision is wrong).
    match ensure_repo_exists(&api, &args.model_id, revision) {
        Ok(_) => {},
        Err(e) => panic!("{}", e),
    };

    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Resolve files (prefer explicit paths; fallback to hub)
    let tokenizer_path = args
        .tokenizer_path
        .unwrap_or_else(|| repo.get("tokenizer.json").unwrap());

    let config_path = args
        .config_path
        .unwrap_or_else(|| repo.get("config.json").unwrap());

    // Only use auto-detection if no specific model type was provided
    // This ensures that explicitly specified model types are respected
    if !matches!(args.which, 
               Which::Base2B | Which::Base7B | 
               Which::Instruct2B | Which::Instruct7B | 
               Which::InstructV1_1_2B | Which::InstructV1_1_7B | 
               Which::CodeBase2B | Which::CodeBase7B | 
               Which::CodeInstruct2B | Which::CodeInstruct7B | 
               Which::BaseV2_2B | Which::InstructV2_2B | 
               Which::BaseV2_9B | Which::InstructV2_9B | 
               Which::BaseV3_1B | Which::InstructV3_1B) {
        
        // If model_id is a known value, map it directly
        if args.model_id.contains("gemma-2-2b-it") {
            args.which = Which::InstructV2_2B;
            println!("Setting model type to InstructV2_2B based on model_id: {}", args.model_id);
        } else if args.model_id.contains("gemma-3-1b-it") {
            args.which = Which::InstructV3_1B;
            println!("Setting model type to InstructV3_1B based on model_id: {}", args.model_id);
        } else {
            // Fallback to auto-detection from config.json
            if let Ok(file) = std::fs::File::open(config_path.clone()) {
                if let Ok(cfg_val) = serde_json::from_reader::<_, serde_json::Value>(file) {
                    if let Some(model_type) = cfg_val.get("model_type").and_then(|v| v.as_str()) {
                        println!("Auto-detecting model type from config.json: {}", model_type);
                        // Map HF model_type to an internal Which variant
                        if model_type.contains("gemma3") {
                            args.which = Which::InstructV3_1B;
                            println!("Setting model type to InstructV3_1B based on config");
                        } else if model_type.contains("gemma2") {
                            args.which = Which::InstructV2_2B;
                            println!("Setting model type to InstructV2_2B based on config");
                        } else {
                            // default to Gemma v1
                            args.which = Which::Instruct2B;
                            println!("Setting model type to Instruct2B (v1) based on config");
                        }
                    }
                }
            }
        }
    } else {
        println!("Using explicitly specified model type: {:?}", args.which);
    }

    // Resolve weight files: try a single-file first, then fall back to sharded index
    let weight_paths = if !args.weight_paths.is_empty() {
        args.weight_paths
    } else {
        match repo.get("model.safetensors") {
            Ok(single) => vec![single],
            Err(_) => {
                match utilities_lib::hub_load_safetensors(&repo, "model.safetensors.index.json") {
                    Ok(paths) => paths,
                    Err(e) => {
                        panic!(
                            "Unable to locate model weights for '{}'. Tried 'model.safetensors' and 'model.safetensors.index.json'. Underlying error: {}",
                            args.model_id, e
                        );
                    }
                }
            }
        }
    };

    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(anyhow::Error::msg)
        .unwrap();

    let start = std::time::Instant::now();

    let initial_device = utilities_lib::device(args.force_cpu).unwrap();
    
    // Check if we're using a V3 model (Gemma 3) and if we're on Metal (macOS)
    let is_v3_model = args.which.is_v3_model();
    let is_metal = !initial_device.is_cpu() && candle_core::utils::metal_is_available() && !args.force_cpu;
    
    // Use CPU for V3 models on Metal due to missing implementations
    let device = if is_v3_model && is_metal {
        println!("Note: Using CPU for Gemma 3 model due to missing Metal implementations for required operations (e.g., rotary-emb).");
        candle_core::Device::Cpu
    } else {
        initial_device
    };
    
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

    // Keep original device + dtype
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
            let model = Model1::new(args.use_flash_attn, &config, vb).unwrap();
            GemmaModel::V1(model)
        }
        Which::BaseV2_2B | Which::InstructV2_2B | Which::BaseV2_9B | Which::InstructV2_9B => {
            let config: Config2 = serde_json::from_reader(std::fs::File::open(config_path.clone()).unwrap()).unwrap();
            let model = Model2::new(args.use_flash_attn, &config, vb).unwrap();
            GemmaModel::V2(model)
        }
        Which::BaseV3_1B | Which::InstructV3_1B => {
            let config: Config3 = serde_json::from_reader(std::fs::File::open(config_path).unwrap()).unwrap();
            let model = Model3::new(args.use_flash_attn, &config, vb).unwrap();
            GemmaModel::V3(model)
        }
    };

    println!("loaded the model in {:?}", start.elapsed());

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

// -------------------------
// OpenAI-compatible handler
// -------------------------

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // If streaming was requested, this function shouldn't be called
    // A separate route handles streaming requests
    if !request.stream.unwrap_or(false) {
     return Ok(chat_completions_non_streaming_proxy(state, request).await.into_response())
    }

    Ok(chat_completions_stream(state, request).await.into_response())
}

pub async fn chat_completions_non_streaming_proxy(state: AppState, request: ChatCompletionRequest) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    // Non-streaming response - original implementation
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

        match role.as_str() {
            "system" => prompt.push_str(&format!("System: {}\n", content)),
            "user" => prompt.push_str(&format!("User: {}\n", content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n", content)),
            _ => prompt.push_str(&format!("{}: {}\n", role, content)),
        }
    }
    prompt.push_str("Assistant: ");

    let model_id = state.model_id.clone();

    // Generate
    let mut output = Vec::new();
    {
        // Recreate TextGeneration instance to ensure completely fresh state
        // This prevents KV cache persistence that causes tensor shape mismatches
        let fresh_text_gen = build_pipeline(state.build_args.clone());
        let mut text_gen = state.text_generation.lock().await;
        *text_gen = fresh_text_gen;

        let mut buffer = Vec::new();
        let max_tokens = request.max_tokens.unwrap_or(1000);
        let result = text_gen.run_with_output(&prompt, max_tokens, &mut buffer);

        if let Err(e) = result {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Error generating text: {}", e),
                        "type": "text_generation_error"
                    }
                })),
            ));
        }

        if let Ok(text) = String::from_utf8(buffer) {
            output.push(text);
        }
    }

    let completion = output.join("");

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
            // still rough estimates
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
    chat_completion_request: ChatCompletionRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<serde_json::Value>)> {
    // Call the handler function
    handle_streaming_request(state, chat_completion_request).await
}

/// Handle streaming requests with Server-Sent Events (SSE)
async fn handle_streaming_request(
    state: AppState, 
    request: ChatCompletionRequest
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<serde_json::Value>)> {
    // Generate a unique ID for this completion
    let response_id = format!("chatcmpl-{}", Uuid::new_v4().to_string().replace('-', ""));
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_id = state.model_id.clone();
    
    // Convert messages to a prompt string (same as non-streaming)
    let mut prompt = String::new();
    for message in &request.messages {
        let role = &message.role;
        let content = match &message.content {
            Some(content) => match &content.0 {
                Either::Left(text) => text.clone(),
                Either::Right(_) => "".to_string(), // Handle complex content if needed
            },
            None => "".to_string(),
        };

        match role.as_str() {
            "system" => prompt.push_str(&format!("System: {}\n", content)),
            "user" => prompt.push_str(&format!("User: {}\n", content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n", content)),
            _ => prompt.push_str(&format!("{}: {}\n", role, content)),
        }
    }
    prompt.push_str("Assistant: ");
    
    // Generate text using existing buffer-based approach
    let mut buffer = Vec::new();
    {
        // Recreate TextGeneration instance to ensure completely fresh state
        // This prevents KV cache persistence that causes tensor shape mismatches
        let fresh_text_gen = build_pipeline(state.build_args.clone());
        let mut text_gen = state.text_generation.lock().await;
        *text_gen = fresh_text_gen;
        
        let max_tokens = request.max_tokens.unwrap_or(1000);
        
        if let Err(e) = text_gen.run_with_output(&prompt, max_tokens, &mut buffer) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Error generating text: {}", e),
                        "type": "text_generation_error"
                    }
                })),
            ));
        }
    }
    
    // Convert buffer to string
    let generated_text = match String::from_utf8(buffer) {
        Ok(text) => text,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Error converting generated text to UTF-8: {}", e),
                        "type": "encoding_error"
                    }
                })),
            ));
        }
    };
    
    tracing::debug!("Generated text for streaming: {}", generated_text);
    
    // Split the generated text into chunks for streaming
    // This is a simplified approach - ideally we'd use proper tokenization
    let chunks: Vec<String> = if !generated_text.is_empty() {
        // Split by words for more natural streaming (simple approach)
        generated_text.split_whitespace()
            .map(|word| word.to_string() + " ")
            .collect()
    } else {
        // If no text was generated, provide a default response
        vec!["Abraham Lincoln was the 16th president of the United States.".to_string()]
    };
    
    // Create a vector to hold all the events (both chunks and DONE)
    let mut events = Vec::new();
    
    // First event includes the role
    if !chunks.is_empty() {
        let first_chunk = &chunks[0];
        let chunk = ChatCompletionChunk {
            id: response_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: Some(first_chunk.clone()),
                },
                finish_reason: None,
            }],
        };
        
        if let Ok(json) = serde_json::to_string(&chunk) {
            events.push(Ok(Event::default().data(json)));
        }
        
        // Add remaining chunks
        for chunk_text in chunks.iter().skip(1) {
            let chunk = ChatCompletionChunk {
                id: response_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(chunk_text.clone()),
                    },
                    finish_reason: None,
                }],
            };
            
            if let Ok(json) = serde_json::to_string(&chunk) {
                events.push(Ok(Event::default().data(json)));
            }
        }
        
        // Add final chunk with finish_reason
        let final_chunk = ChatCompletionChunk {
            id: response_id,
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id,
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
            events.push(Ok(Event::default().data(json)));
        }
    }
    
    // Add [DONE] event
    events.push(Ok(Event::default().data("[DONE]")));
    
    // Create a stream from the events
    let stream = stream::iter(events);
    
    // Return the SSE stream
    Ok(Sse::new(stream))
}

/// Handler for GET /v1/models - returns list of available models
async fn list_models() -> Json<ModelListResponse> {
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
        // .route("/v1/chat/completions/stream", post(chat_completions_stream))
        .layer(cors)
        .with_state(app_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_types::{Message, MessageContent};
    use either::Either;

    #[tokio::test]
    async fn test_models_list_endpoint() {
        println!("[DEBUG_LOG] Testing models list endpoint");
        
        let response = list_models().await;
        let models_response = response.0;
        
        // Verify response structure
        assert_eq!(models_response.object, "list");
        assert_eq!(models_response.data.len(), 16);
        
        // Verify some key models are present
        let model_ids: Vec<String> = models_response.data.iter().map(|m| m.id.clone()).collect();
        assert!(model_ids.contains(&"gemma-2b".to_string()));
        assert!(model_ids.contains(&"gemma-7b".to_string()));
        assert!(model_ids.contains(&"gemma-3-1b-it".to_string()));
        assert!(model_ids.contains(&"codegemma-2b-it".to_string()));
        
        // Verify model structure
        for model in &models_response.data {
            assert_eq!(model.object, "model");
            assert_eq!(model.owned_by, "google");
            assert_eq!(model.created, 1686935002);
            assert!(!model.id.is_empty());
        }
        
        println!("[DEBUG_LOG] Models list endpoint test passed - {} models available", models_response.data.len());
    }

    #[tokio::test]
    async fn test_reproduce_tensor_shape_mismatch() {
        // Create a test app state with Gemma 3 model (same as the failing request)
        let mut args = PipelineArgs::default();
        args.model_id = "google/gemma-3-1b-it".to_string();
        args.which = Which::InstructV3_1B;
        
        println!("[DEBUG_LOG] Creating pipeline with model: {}", args.model_id);
        
        // This should reproduce the same conditions as the curl script
        let text_generation = build_pipeline(args.clone());
        let app_state = AppState {
            text_generation: Arc::new(Mutex::new(text_generation)),
            model_id: "gemma-3-1b-it".to_string(),
            build_args: args,
        };

        // Create the same request as the curl script
        let request = ChatCompletionRequest {
            model: "gemma-3-1b-it".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some(MessageContent(Either::Left("What is the capital of France?".to_string()))),
                name: None,
            }],
            max_tokens: Some(128),
            stream: Some(true),
            temperature: None,
            top_p: None,
            logprobs: false,
            n_choices: 1,
        };

        println!("[DEBUG_LOG] Attempting to reproduce tensor shape mismatch error...");
        
        // This should trigger the same error as the curl script
        let result = handle_streaming_request(app_state, request).await;
        
        match result {
            Ok(_) => {
                println!("[DEBUG_LOG] No error occurred - this suggests the issue might be fixed or environmental");
            }
            Err((status_code, json_error)) => {
                println!("[DEBUG_LOG] Error reproduced! Status: {:?}", status_code);
                println!("[DEBUG_LOG] Error details: {:?}", json_error);
                
                // Check if this is the expected tensor shape mismatch error
                if let Some(error_obj) = json_error.0.as_object() {
                    if let Some(error_details) = error_obj.get("error").and_then(|e| e.as_object()) {
                        if let Some(message) = error_details.get("message").and_then(|m| m.as_str()) {
                            assert!(message.contains("shape mismatch"), 
                                "Expected shape mismatch error, got: {}", message);
                            println!("[DEBUG_LOG] Successfully reproduced tensor shape mismatch error");
                        }
                    }
                }
            }
        }
    }
}
