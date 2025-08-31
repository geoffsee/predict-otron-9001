use either::Either;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use utoipa::ToSchema;

/// Inner content structure for messages that can be either a string or key-value pairs
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageInnerContent(
    #[serde(with = "either::serde_untagged")] pub Either<String, HashMap<String, String>>,
);

impl ToSchema<'_> for MessageInnerContent {
    fn schema() -> (
        &'static str,
        utoipa::openapi::RefOr<utoipa::openapi::Schema>,
    ) {
        (
            "MessageInnerContent",
            utoipa::openapi::RefOr::T(message_inner_content_schema()),
        )
    }
}

/// Function for MessageInnerContent Schema generation to handle `Either`
fn message_inner_content_schema() -> utoipa::openapi::Schema {
    use utoipa::openapi::{ObjectBuilder, OneOfBuilder, RefOr, Schema, SchemaType};

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
    pub  Either<String, Vec<HashMap<String, MessageInnerContent>>>,
);

impl ToSchema<'_> for MessageContent {
    fn schema() -> (
        &'static str,
        utoipa::openapi::RefOr<utoipa::openapi::Schema>,
    ) {
        (
            "MessageContent",
            utoipa::openapi::RefOr::T(message_content_schema()),
        )
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
pub fn default_false() -> bool {
    false
}

/// Default value helper
pub fn default_1usize() -> usize {
    1
}

/// Default value helper
pub fn default_model() -> String {
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

/// Delta for streaming responses - contains incremental content updates
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Delta {
    /// The role of the message sender (only in first chunk)
    pub role: Option<String>,
    /// The incremental content
    pub content: Option<String>,
}

/// Chat completion choice for streaming chunks
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// Chat completion chunk for streaming responses
#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

/// Token usage information
#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Model object representing an available model
#[derive(Debug, Serialize, ToSchema)]
pub struct Model {
    /// The model identifier
    pub id: String,
    /// The object type, always "model"
    pub object: String,
    /// Unix timestamp of when the model was created
    pub created: u64,
    /// The organization that owns the model
    pub owned_by: String,
}

/// Response for listing available models
#[derive(Debug, Serialize, ToSchema)]
pub struct ModelListResponse {
    /// The object type, always "list"
    pub object: String,
    /// Array of available models
    pub data: Vec<Model>,
}
