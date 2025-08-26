mod token_output_stream;
mod utilities_lib;

#[cfg(feature = "intel-mkl-src")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate-src")]
extern crate accelerate_src;

#[cfg(feature = "metal")]
extern crate metal_src;

use anyhow::{Error as E, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use either::Either;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use utoipa::ToSchema;

use candle_transformers::models::gemma::{Config as Config1, Model as Model1};
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};

// OpenAI API compatible structs

/// Inner content structure for messages that can be either a string or key-value pairs
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageInnerContent(
    #[serde(with = "either::serde_untagged")] pub Either<String, HashMap<String, String>>,
);

impl ToSchema<'_> for MessageInnerContent {
    fn schema() -> (&'static str, utoipa::openapi::RefOr<utoipa::openapi::Schema>) {
        (
            "MessageInnerContent",
            utoipa::openapi::RefOr::T(message_inner_content_schema()),
        )
    }
}

/// Function for MessageInnerContent Schema generation to handle `Either`
fn message_inner_content_schema() -> utoipa::openapi::Schema {
    use utoipa::openapi::{ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, SchemaType};

    Schema::OneOf(
        OneOfBuilder::new()
            // Either::Left - simple string
            .item(Schema::Object(
                ObjectBuilder::new().schema_type(SchemaType::String).build(),
            ))
            // Either::Right - object with string values
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Object)
                    .additional_properties(Some(RefOr::T(Schema::Object(
                        ObjectBuilder::new().schema_type(SchemaType::String).build(),
                    ))))
                    .build(),
            ))
            .build(),
    )
}

/// Message content that can be either simple text or complex structured content
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageContent(
    #[serde(with = "either::serde_untagged")]
    Either<String, Vec<HashMap<String, MessageInnerContent>>>,
);

impl ToSchema<'_> for MessageContent {
    fn schema() -> (&'static str, utoipa::openapi::RefOr<utoipa::openapi::Schema>) {
        ("MessageContent", utoipa::openapi::RefOr::T(message_content_schema()))
    }
}

/// Function for MessageContent Schema generation to handle `Either`
fn message_content_schema() -> utoipa::openapi::Schema {
    use utoipa::openapi::{ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, SchemaType};

    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Object(
                ObjectBuilder::new().schema_type(SchemaType::String).build(),
            ))
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::T(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Object)
                            .additional_properties(Some(RefOr::Ref(
                                utoipa::openapi::Ref::from_schema_name("MessageInnerContent"),
                            )))
                            .build(),
                    )))
                    .build(),
            ))
            .build(),
    )
}

/// Represents a single message in a conversation
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct Message {
    /// The message content
    pub content: Option<MessageContent>,
    /// The role of the message sender ("user", "assistant", "system", "tool", etc.)
    pub role: String,
    pub name: Option<String>,
}

/// Stop token configuration for generation
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum StopTokens {
    ///  Multiple possible stop sequences
    Multi(Vec<String>),
    /// Single stop sequence
    Single(String),
}

/// Default value helper
fn default_false() -> bool {
    false
}

/// Default value helper
fn default_1usize() -> usize {
    1
}

/// Default value helper
fn default_model() -> String {
    "default".to_string()
}

/// Chat completion request following OpenAI's specification
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ChatCompletionRequest {
    #[schema(example = json!([{"role": "user", "content": "Why did the crab cross the road?"}]))]
    pub messages: Vec<Message>,
    #[schema(example = "gemma-3-1b-it")]
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub logprobs: bool,
    #[schema(example = 256)]
    pub max_tokens: Option<usize>,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    #[schema(example = 0.9)]
    pub top_p: Option<f64>,
    #[schema(example = false)]
    pub stream: Option<bool>,
}

/// Chat completion choice
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

/// Chat completion response
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

/// Token usage information
#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// Application state shared between handlers
#[derive(Clone)]
struct AppState {
    text_generation: Arc<Mutex<TextGeneration>>,
    model_id: String,
}

// Chat completions endpoint handler
async fn chat_completions(
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
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().to_string().replace("-", "")),
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

use candle_core::{DType, Device, MetalDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde_json::json;
use tokenizers::Tokenizer;
use crate::token_output_stream::TokenOutputStream;
use crate::utilities_lib::device;

// Create the router with the chat completions endpoint
fn create_router(app_state: AppState) -> Router {
    // CORS layer to allow requests from any origin
    let cors = CorsLayer::new()
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

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "2b")]
    Base2B,
    #[value(name = "7b")]
    Base7B,
    #[value(name = "2b-it")]
    Instruct2B,
    #[value(name = "7b-it")]
    Instruct7B,
    #[value(name = "1.1-2b-it")]
    InstructV1_1_2B,
    #[value(name = "1.1-7b-it")]
    InstructV1_1_7B,
    #[value(name = "code-2b")]
    CodeBase2B,
    #[value(name = "code-7b")]
    CodeBase7B,
    #[value(name = "code-2b-it")]
    CodeInstruct2B,
    #[value(name = "code-7b-it")]
    CodeInstruct7B,
    #[value(name = "2-2b")]
    BaseV2_2B,
    #[value(name = "2-2b-it")]
    InstructV2_2B,
    #[value(name = "2-9b")]
    BaseV2_9B,
    #[value(name = "2-9b-it")]
    InstructV2_9B,
    #[value(name = "3-1b")]
    BaseV3_1B,
    #[value(name = "3-1b-it")]
    InstructV3_1B,
}

enum Model {
    V1(Model1),
    V2(Model2),
    V3(Model3),
}

impl Model {
    fn forward(&mut self, input_ids: &candle_core::Tensor, pos: usize) -> candle_core::Result<candle_core::Tensor> {
        match self {
            Self::V1(m) => m.forward(input_ids, pos),
            Self::V2(m) => m.forward(input_ids, pos),
            Self::V3(m) => m.forward(input_ids, pos),
        }
    }
}



struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    // Run text generation and print to stdout
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };

        let eot_token = match self.tokenizer.get_token("<end_of_turn>") {
            Some(token) => token,
            None => {
                println!(
                    "Warning: <end_of_turn> token not found in tokenizer, using <eos> as a backup"
                );
                eos_token
            }
        };

        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);

                // Manual implementation of repeat penalty to avoid type conflicts
                let mut logits_vec = logits.to_vec1::<f32>()?;

                for &token_id in &tokens[start_at..] {
                    let token_id = token_id as usize;
                    if token_id < logits_vec.len() {
                        let score = logits_vec[token_id];
                        let sign = if score < 0.0 { -1.0 } else { 1.0 };
                        logits_vec[token_id] = sign * score / self.repeat_penalty;
                    }
                }

                // Create a new tensor with the modified logits
                let device = logits.device().clone();
                let shape = logits.shape().clone();
                let new_logits = Tensor::new(&logits_vec[..], &device)?;
                new_logits.reshape(shape)?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eot_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }

    // Run text generation and write to a buffer
    fn run_with_output(&mut self, prompt: &str, sample_len: usize, output: &mut Vec<u8>) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // Write prompt tokens to output
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                write!(output, "{}", t)?;
            }
        }

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };

        let eot_token = match self.tokenizer.get_token("<end_of_turn>") {
            Some(token) => token,
            None => {
                write!(output, "Warning: <end_of_turn> token not found in tokenizer, using <eos> as a backup")?;
                eos_token
            }
        };

        // Determine if we're using a Model3 (gemma-3) variant
        let is_model3 = match &self.model {
            Model::V3(_) => true,
            _ => false,
        };

        // For Model3, we need to use a different approach
        if is_model3 {
            // For gemma-3 models, we'll generate one token at a time with the full context
            let start_gen = std::time::Instant::now();

            // Initial generation with the full prompt
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let mut logits = self.model.forward(&input, 0)?;
            logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            for _ in 0..sample_len {
                // Apply repeat penalty if needed
                let current_logits = if self.repeat_penalty == 1. {
                    logits.clone()
                } else {
                    let start_at = tokens.len().saturating_sub(self.repeat_last_n);

                    // Manual implementation of repeat penalty to avoid type conflicts
                    let mut logits_vec = logits.to_vec1::<f32>()?;

                    for &token_id in &tokens[start_at..] {
                        let token_id = token_id as usize;
                        if token_id < logits_vec.len() {
                            let score = logits_vec[token_id];
                            let sign = if score < 0.0 { -1.0 } else { 1.0 };
                            logits_vec[token_id] = sign * score / self.repeat_penalty;
                        }
                    }

                    // Create a new tensor with the modified logits
                    let device = logits.device().clone();
                    let shape = logits.shape().clone();
                    let new_logits = Tensor::new(&logits_vec[..], &device)?;
                    new_logits.reshape(shape)?
                };

                let next_token = self.logits_processor.sample(&current_logits)?;
                tokens.push(next_token);
                generated_tokens += 1;

                if next_token == eos_token || next_token == eot_token {
                    break;
                }

                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    write!(output, "{}", t)?;
                }

                // For the next iteration, just use the new token
                let new_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                logits = self.model.forward(&new_input, tokens.len() - 1)?;
                logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            }

            return Ok(());
        }

        // Standard approach for other models
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
 
                // Manual implementation of repeat penalty to avoid type conflicts
                let mut logits_vec = logits.to_vec1::<f32>()?;

                for &token_id in &tokens[start_at..] {
                    let token_id = token_id as usize;
                    if token_id < logits_vec.len() {
                        let score = logits_vec[token_id];
                        let sign = if score < 0.0 { -1.0 } else { 1.0 };
                        logits_vec[token_id] = sign * score / self.repeat_penalty;
                    }
                }

                // Create a new tensor with the modified logits
                let device = logits.device().clone();
                let shape = logits.shape().clone();
                let new_logits = Tensor::new(&logits_vec[..], &device)?;
                new_logits.reshape(shape)?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eot_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                write!(output, "{}", t)?;
            }
        }

        // Write any remaining tokens
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            write!(output, "{}", rest)?;
        }

        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Run in server mode with OpenAI compatible API
    #[arg(long)]
    server: bool,

    /// Port to use for the server
    #[arg(long, default_value_t = 3777)]
    port: u16,

    /// Prompt for text generation (not used in server mode)
    #[arg(long)]
    prompt: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The model to use.
    #[arg(long, default_value = "3-1b-it")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.which {
            Which::InstructV1_1_2B => "google/gemma-1.1-2b-it".to_string(),
            Which::InstructV1_1_7B => "google/gemma-1.1-7b-it".to_string(),
            Which::Base2B => "google/gemma-2b".to_string(),
            Which::Base7B => "google/gemma-7b".to_string(),
            Which::Instruct2B => "google/gemma-2b-it".to_string(),
            Which::Instruct7B => "google/gemma-7b-it".to_string(),
            Which::CodeBase2B => "google/codegemma-2b".to_string(),
            Which::CodeBase7B => "google/codegemma-7b".to_string(),
            Which::CodeInstruct2B => "google/codegemma-2b-it".to_string(),
            Which::CodeInstruct7B => "google/codegemma-7b-it".to_string(),
            Which::BaseV2_2B => "google/gemma-2-2b".to_string(),
            Which::InstructV2_2B => "google/gemma-2-2b-it".to_string(),
            Which::BaseV2_9B => "google/gemma-2-9b".to_string(),
            Which::InstructV2_9B => "google/gemma-2-9b-it".to_string(),
            Which::BaseV3_1B => "google/gemma-3-1b-pt".to_string(),
            Which::InstructV3_1B => "google/gemma-3-1b-it".to_string(),
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id.clone(),
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => match args.which {
            Which::BaseV3_1B | Which::InstructV3_1B => vec![repo.get("model.safetensors")?],
            _ => utilities_lib::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        },
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let initial_device = utilities_lib::device(args.cpu)?;
    
    // Check if we're using a V3 model (Gemma 3) and if we're on Metal (macOS)
    let is_v3_model = matches!(args.which, Which::BaseV3_1B | Which::InstructV3_1B);
    let is_metal = !initial_device.is_cpu() && candle_core::utils::metal_is_available() && !args.cpu;
    
    // Use CPU for V3 models on Metal due to missing implementations
    let device = if is_v3_model && is_metal {
        println!("Note: Using CPU for Gemma 3 model due to missing Metal implementations for required operations (e.g., rotary-emb).");
        Device::Cpu
    } else {
        initial_device
    };
    
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    
    // Use the selected device and dtype
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
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
            let config: Config1 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model1::new(args.use_flash_attn, &config, vb)?;
            Model::V1(model)
        }
        Which::BaseV2_2B | Which::InstructV2_2B | Which::BaseV2_9B | Which::InstructV2_9B => {
            let config: Config2 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model2::new(args.use_flash_attn, &config, vb)?;
            Model::V2(model)
        }
        Which::BaseV3_1B | Which::InstructV3_1B => {
            let config: Config3 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model3::new(args.use_flash_attn, &config, vb)?;
            Model::V3(model)
        }
    };

    println!("loaded the model in {:?}", start.elapsed());

    let pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );

    if args.server {
        // Start the server
        println!("Starting server on port {}", args.port);

        // Create app state
        let app_state = AppState {
            text_generation: Arc::new(Mutex::new(pipeline)),
            model_id,
        };

        // Create router
        let app = create_router(app_state);

        // Run the server
        let addr = SocketAddr::from(([0, 0, 0, 0], args.port));

        // Use tokio to run the server
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(async {
                axum::serve(tokio::net::TcpListener::bind(&addr).await?, app)
                    .await
                    .map_err(|e| anyhow::anyhow!("Server error: {}", e))
            })?;

        Ok(())
    } else {
        // Run in CLI mode
        if let Some(prompt_text) = &args.prompt {
            let prompt = match args.which {
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
                | Which::BaseV3_1B => prompt_text.clone(),
                Which::InstructV3_1B => {
                    format!(
                        "<start_of_turn> user\n{}<end_of_turn>\n<start_of_turn> model\n",
                        prompt_text
                    )
                }
            };

            let mut pipeline = pipeline;
            pipeline.run(&prompt, args.sample_len)?;
            Ok(())
        } else {
            anyhow::bail!("Prompt is required in CLI mode. Use --prompt to specify a prompt or --server to run in server mode.")
        }
    }
}
