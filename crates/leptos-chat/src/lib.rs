use leptos::*;
use leptos_meta::*;
use leptos_router::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;
use js_sys::Date;
use web_sys::{HtmlInputElement, KeyboardEvent, SubmitEvent};
use futures_util::StreamExt;
use async_openai_wasm::{
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, Model as OpenAIModel,
    },
    Client,
};
use async_openai_wasm::config::OpenAIConfig;
use async_openai_wasm::types::{ChatCompletionResponseStream, Model, Role, FinishReason};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContent(pub either::Either<String, Vec<std::collections::HashMap<String, MessageInnerContent>>>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageInnerContent(pub either::Either<String, std::collections::HashMap<String, String>>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<MessageContent>,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/style/main.css"/>
        <Title text="Chat Interface"/>
        <Router>
            <main>
                <Routes>
                    <Route path="/" view=ChatInterface/>
                </Routes>
            </main>
        </Router>
    }
}

async fn fetch_available_models() -> Result<Vec<OpenAIModel>, String> {
    log::info!("[DEBUG_LOG] fetch_available_models: Starting model fetch from http://localhost:8080/v1");
    
    let config = OpenAIConfig::new().with_api_base("http://localhost:8080/v1".to_string());
    let client = Client::with_config(config);
    
    match client.models().list().await {
        Ok(response) => {
            let model_count = response.data.len();
            log::info!("[DEBUG_LOG] fetch_available_models: Successfully fetched {} models", model_count);
            
            if model_count > 0 {
                let model_names: Vec<String> = response.data.iter().map(|m| m.id.clone()).collect();
                log::debug!("[DEBUG_LOG] fetch_available_models: Available models: {:?}", model_names);
            } else {
                log::warn!("[DEBUG_LOG] fetch_available_models: No models returned by server");
            }
            
            Ok(response.data)
        },
        Err(e) => {
            log::error!("[DEBUG_LOG] fetch_available_models: Failed to fetch models: {:?}", e);
            
            let error_details = format!("{:?}", e);
            if error_details.contains("400") || error_details.contains("Bad Request") {
                log::error!("[DEBUG_LOG] fetch_available_models: HTTP 400 - Server rejected models request");
            } else if error_details.contains("404") || error_details.contains("Not Found") {
                log::error!("[DEBUG_LOG] fetch_available_models: HTTP 404 - Models endpoint not found");
            } else if error_details.contains("Connection") || error_details.contains("connection") {
                log::error!("[DEBUG_LOG] fetch_available_models: Connection error - server may be down");
            }
            
            Err(format!("Failed to fetch models: {}", e))
        }
    }
}

async fn send_chat_request(chat_request: ChatRequest) -> ChatCompletionResponseStream {
    let config = OpenAIConfig::new().with_api_base("http://localhost:8080/v1".to_string());
    let client = Client::with_config(config);

    let mut typed_chat = async_openai_wasm::types::CreateChatCompletionRequest {
        messages: vec![],
        model: "".to_string(),
        store: None,
        reasoning_effort: None,
        metadata: None,
        frequency_penalty: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        max_tokens: None,
        max_completion_tokens: None,
        n: None,
        modalities: None,
        prediction: None,
        audio: None,
        presence_penalty: None,
        response_format: None,
        seed: None,
        service_tier: None,
        stop: None,
        stream: None,
        stream_options: None,
        temperature: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        user: None,
        function_call: None,
        functions: None,
        web_search_options: None,
        extra_params: None,
    };

    typed_chat.messages = chat_request.messages
        .iter()
        .map(|msg| {
            let content = match &msg.content {
                Some(MessageContent(either::Either::Left(text))) => text.clone(),
                _ => "".to_string()
            };
            let role = msg.role.clone();
            match role.as_str() {
                "system" => ChatCompletionRequestSystemMessageArgs::default()
                    .content(content)
                    .build()
                    .expect("failed to build system message")
                    .into(),
                "user" => ChatCompletionRequestUserMessageArgs::default()
                    .content(content)
                    .build()
                    .expect("failed to build user message")
                    .into(),
                "assistant" => ChatCompletionRequestAssistantMessageArgs::default()
                    .content(content)
                    .build()
                    .expect("failed to build assistant message")
                    .into(),
                _ => ChatCompletionRequestUserMessageArgs::default()
                    .content(content)
                    .build()
                    .expect("failed to build default message")
                    .into()
            }
        })
        .collect();
    client.chat().create_stream(typed_chat).await.unwrap()
}

// #[cfg(not(target_arch = "wasm32"))]
// async fn send_chat_request(_chat_request: ChatRequest) -> Result<ChatResponse, String> {
//     Err("leptos-chat chat request only supported on wasm32 target".to_string())
// }

const DEFAULT_MODEL: &str = "default";

#[component]
fn ChatInterface() -> impl IntoView {
    let (messages, set_messages) = create_signal::<VecDeque<Message>>(VecDeque::new());
    let (input_value, set_input_value) = create_signal(String::new());
    let (is_loading, set_is_loading) = create_signal(false);
    let (available_models, set_available_models) = create_signal::<Vec<OpenAIModel>>(Vec::new());
    let (selected_model, set_selected_model) = create_signal(DEFAULT_MODEL.to_string());
    let (models_loading, set_models_loading) = create_signal(false);

    // Fetch models on component initialization
    create_effect(move |_| {
        spawn_local(async move {
            set_models_loading.set(true);
            match fetch_available_models().await {
                Ok(models) => {
                    set_available_models.set(models);
                    set_models_loading.set(false);
                }
                Err(e) => {
                    log::error!("Failed to fetch models: {}", e);
                    // Set a default model if fetching fails
                    set_available_models.set(vec![]);
                    set_models_loading.set(false);
                }
            }
        });
    });

    let send_message = create_action(move |content: &String| {
        let content = content.clone();
        async move {
            if content.trim().is_empty() {
                log::debug!("[DEBUG_LOG] send_message: Empty content, skipping");
                return;
            }

            log::info!("[DEBUG_LOG] send_message: Starting message send process");
            log::debug!("[DEBUG_LOG] send_message: User message content length: {}", content.len());

            set_is_loading.set(true);

            // Add user message to chat
            let user_message = Message {
                id: Uuid::new_v4().to_string(),
                role: "user".to_string(),
                content: content.clone(),
                timestamp: Date::now(),
            };

            set_messages.update(|msgs| msgs.push_back(user_message.clone()));
            set_input_value.set(String::new());

            let mut chat_messages = Vec::new();

            // Add system message
            let system_message = ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()
                .expect("failed to build system message");
            chat_messages.push(system_message.into());

            // Add history messages
            let history_count = messages.with_untracked(|msgs| {
                let count = msgs.len();
                for msg in msgs.iter() {
                        match msg.role.as_str() {
                        "user" => {
                            let message = ChatCompletionRequestUserMessageArgs::default()
                                .content(msg.content.clone())
                                .build()
                                .expect("failed to build user message");
                            chat_messages.push(message.into());
                        }
                        "assistant" => {
                            let message = ChatCompletionRequestAssistantMessageArgs::default()
                                .content(msg.content.clone())
                                .build()
                                .expect("failed to build assistant message");
                            chat_messages.push(message.into());
                        }
                        "system" => {
                            let message = ChatCompletionRequestSystemMessageArgs::default()
                                .content(msg.content.clone())
                                .build()
                                .expect("failed to build system message");
                            chat_messages.push(message.into());
                        }
                        _ => {
                            // Default to user message for unknown roles
                            let message = ChatCompletionRequestUserMessageArgs::default()
                                .content(msg.content.clone())
                                .build()
                                .expect("failed to build default message");
                            chat_messages.push(message.into());
                        }
                    }
                }
                count
            });

            // Add current user message
            let message = ChatCompletionRequestUserMessageArgs::default()
                .content(user_message.content.clone())
                .build()
                .expect("failed to build user message");
            chat_messages.push(message.into());

            let current_model = selected_model.get_untracked();
            let total_messages = chat_messages.len();
            
            log::info!("[DEBUG_LOG] send_message: Preparing request - model: '{}', history_count: {}, total_messages: {}", 
                      current_model, history_count, total_messages);

            let request = CreateChatCompletionRequestArgs::default()
                .model(current_model.as_str())
                .max_tokens(512u32)
                .messages(chat_messages)
                .stream(true) // ensure server streams
                .build()
                .expect("failed to build request");

            // Log request details for debugging server issues
            log::info!("[DEBUG_LOG] send_message: Request configuration - model: '{}', max_tokens: 512, stream: true, messages_count: {}", 
                      current_model, total_messages);
            log::debug!("[DEBUG_LOG] send_message: Request details - history_messages: {}, system_messages: 1, user_messages: {}", 
                       history_count, total_messages - history_count - 1);

            // Send request
            let config = OpenAIConfig::new().with_api_base("http://localhost:8080/v1".to_string());
            let client = Client::with_config(config);
            
            log::info!("[DEBUG_LOG] send_message: Sending request to http://localhost:8080/v1 with model: '{}'", current_model);


            match client.chat().create_stream(request).await {
                Ok(mut stream) => {
                    log::info!("[DEBUG_LOG] send_message: Successfully created stream, starting to receive response");
                    
                    // Defer creating assistant message until we receive role=assistant from the stream
                    let mut assistant_created = false;
                    let mut content_appended = false;
                    let mut chunks_received = 0;
                    // Stream loop: handle deltas and finish events
                    while let Some(next) = stream.next().await {
                        match next {
                            Ok(chunk) => {
                                chunks_received += 1;
                                if let Some(choice) = chunk.choices.get(0) {
                                    // 1) Create assistant message when role arrives
                                    if !assistant_created {
                                        if let Some(role) = &choice.delta.role {
                                            if role == &Role::Assistant {
                                                assistant_created = true;
                                                let assistant_id = Uuid::new_v4().to_string();
                                                set_messages.update(|msgs| {
                                                    msgs.push_back(Message {
                                                        id: assistant_id,
                                                        role: "assistant".to_string(),
                                                        content: String::new(),
                                                        timestamp: Date::now(),
                                                    });
                                                });
                                            }
                                        }
                                    }

                                    // 2) Append content tokens when provided
                                    if let Some(content) = &choice.delta.content {
                                        if !content.is_empty() {
                                            // If content arrives before role, create assistant message now
                                            if !assistant_created {
                                                assistant_created = true;
                                                let assistant_id = Uuid::new_v4().to_string();
                                                set_messages.update(|msgs| {
                                                    msgs.push_back(Message {
                                                        id: assistant_id,
                                                        role: "assistant".to_string(),
                                                        content: String::new(),
                                                        timestamp: Date::now(),
                                                    });
                                                });
                                            }
                                            content_appended = true;
                                            set_messages.update(|msgs| {
                                                if let Some(last) = msgs.back_mut() {
                                                    if last.role == "assistant" {
                                                        last.content.push_str(content);
                                                        last.timestamp = Date::now();
                                                    }
                                                }
                                            });
                                        }
                                    }

                                    // 3) Stop on finish_reason=="stop" (mirrors [DONE])
                                    if let Some(reason) = &choice.finish_reason {
                                        if reason == &FinishReason::Stop {
                                            log::info!("[DEBUG_LOG] send_message: Received finish_reason=stop after {} chunks", chunks_received);
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("[DEBUG_LOG] send_message: Stream error after {} chunks: {:?}", chunks_received, e);
                                log::error!("[DEBUG_LOG] send_message: Stream error details - model: '{}', chunks_received: {}", current_model, chunks_received);
                                set_messages.update(|msgs| {
                                    msgs.push_back(Message {
                                        id: Uuid::new_v4().to_string(),
                                        role: "system".to_string(),
                                        content: format!("Stream error after {} chunks: {}", chunks_received, e),
                                        timestamp: Date::now(),
                                    });
                                });
                                break;
                            }
                        }
                    }

                    // Cleanup: If we created an assistant message but no content ever arrived, remove the empty message
                    if assistant_created && !content_appended {
                        set_messages.update(|msgs| {
                            let should_pop = msgs
                                .back()
                                .map(|m| m.role == "assistant" && m.content.is_empty())
                                .unwrap_or(false);
                            if should_pop {
                                log::info!("[DEBUG_LOG] send_message: Removing empty assistant message (no content received)");
                                msgs.pop_back();
                            }
                        });
                    }

                    log::info!("[DEBUG_LOG] send_message: Stream completed successfully, received {} chunks", chunks_received);
                }
                Err(e) => {
                    // Detailed error logging for different types of errors
                    log::error!("[DEBUG_LOG] send_message: Request failed with error: {:?}", e);
                    log::error!("[DEBUG_LOG] send_message: Request context - model: '{}', total_messages: {}, endpoint: http://localhost:8080/v1", 
                               current_model, total_messages);
                    
                    // Try to extract more specific error information
                    let error_details = format!("{:?}", e);
                    let user_message = if error_details.contains("400") || error_details.contains("Bad Request") {
                        log::error!("[DEBUG_LOG] send_message: HTTP 400 Bad Request detected - possible issues:");
                        log::error!("[DEBUG_LOG] send_message: - Invalid model name: '{}'", current_model);
                        log::error!("[DEBUG_LOG] send_message: - Invalid message format or content");
                        log::error!("[DEBUG_LOG] send_message: - Server configuration issue");
                        format!("Error: HTTP 400 Bad Request - Check model '{}' and message format. See console for details.", current_model)
                    } else if error_details.contains("404") || error_details.contains("Not Found") {
                        log::error!("[DEBUG_LOG] send_message: HTTP 404 Not Found - server endpoint may be incorrect");
                        "Error: HTTP 404 Not Found - Server endpoint not found".to_string()
                    } else if error_details.contains("500") || error_details.contains("Internal Server Error") {
                        log::error!("[DEBUG_LOG] send_message: HTTP 500 Internal Server Error - server-side issue");
                        "Error: HTTP 500 Internal Server Error - Server problem".to_string()
                    } else if error_details.contains("Connection") || error_details.contains("connection") {
                        log::error!("[DEBUG_LOG] send_message: Connection error - server may be down");
                        "Error: Cannot connect to server at http://localhost:8080".to_string()
                    } else {
                        format!("Error: Request failed - {}", e)
                    };
                    
                    let error_message = Message {
                        id: Uuid::new_v4().to_string(),
                        role: "system".to_string(),
                        content: user_message,
                        timestamp: Date::now(),
                    };
                    set_messages.update(|msgs| msgs.push_back(error_message));
                }
            }

            set_is_loading.set(false);
        }
    });

    let on_input = move |ev| {
        let input = event_target::<HtmlInputElement>(&ev);
        set_input_value.set(input.value());
    };

    let on_submit = move |ev: SubmitEvent| {
        ev.prevent_default();
        let content = input_value.get();
        send_message.dispatch(content);
    };

    let on_keypress = move |ev: KeyboardEvent| {
        if ev.key() == "Enter" && !ev.shift_key() {
            ev.prevent_default();
            let content = input_value.get();
            send_message.dispatch(content);
        }
    };

    let on_model_change = move |ev| {
        let select = event_target::<web_sys::HtmlSelectElement>(&ev);
        set_selected_model.set(select.value());
    };

    let messages_list = move || {
        messages.get()
            .into_iter()
            .map(|message| {
                let role_class = match message.role.as_str() {
                    "user" => "user-message",
                    "assistant" => "assistant-message",
                    _ => "system-message",
                };

                view! {
                    <div class=format!("message {}", role_class)>
                        <div class="message-role">{message.role}</div>
                        <div class="message-content">{message.content}</div>
                    </div>
                }
            })
            .collect_view()
    };

    let loading_indicator = move || {
        is_loading.get().then(|| {
            view! {
                <div class="message assistant-message">
                    <div class="message-role">"assistant"</div>
                    <div class="message-content">"Thinking..."</div>
                </div>
            }
        })
    };

    view! {
        <div class="chat-container">
            <h1>"Chat Interface"</h1>
            <div class="model-selector">
                <label for="model-select">"Model: "</label>
                <select 
                    id="model-select"
                    on:change=on_model_change
                    prop:value=selected_model
                    prop:disabled=models_loading
                >
                    {move || {
                        if models_loading.get() {
                            view! {
                                <option value="">"Loading models..."</option>
                            }.into_view()
                        } else {
                            let models = available_models.get();
                            if models.is_empty() {
                                view! {
                                    <option selected=true value="gemma-3b-it">"gemma-3b-it (default)"</option>
                                }.into_view()
                            } else {
                                models.into_iter().map(|model| {
                                    view! {
                                        <option value=model.id.clone() selected={model.id == DEFAULT_MODEL}>{model.id}</option>
                                    }
                                }).collect_view()
                            }
                        }
                    }}
                </select>
            </div>
            <div class="messages-container">
                {messages_list}
                {loading_indicator}
            </div>
            <form class="input-form" on:submit=on_submit>
                <input
                    type="text"
                    class="message-input"
                    placeholder="Type your message here..."
                    prop:value=input_value
                    on:input=on_input
                    on:keypress=on_keypress
                    prop:disabled=is_loading
                />
                <button
                    type="submit"
                    class="send-button"
                    prop:disabled=move || is_loading.get() || input_value.get().trim().is_empty()
                >
                    "Send"
                </button>
            </form>
        </div>
    }
}

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    // Set up error handling and logging for WebAssembly
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("error initializing logger");

    // Mount the App component to the document body

    leptos::mount_to_body(App)
}