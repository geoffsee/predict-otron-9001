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
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use async_openai_wasm::config::OpenAIConfig;
use async_openai_wasm::types::ChatCompletionResponseStream;

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

async fn send_chat_request(chat_request: ChatRequest) -> ChatCompletionResponseStream {
    let config = OpenAIConfig::new().with_api_base("http://localhost:8080".to_string());
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

#[component]
fn ChatInterface() -> impl IntoView {
    let (messages, set_messages) = create_signal::<VecDeque<Message>>(VecDeque::new());
    let (input_value, set_input_value) = create_signal(String::new());
    let (is_loading, set_is_loading) = create_signal(false);

    let send_message = create_action(move |content: &String| {
        let content = content.clone();
        async move {
            if content.trim().is_empty() {
                return;
            }

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
            messages.with(|msgs| {
                for msg in msgs.iter() {
                    let message = ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .expect("failed to build message");
                    chat_messages.push(message.into());
                }
            });

            // Add current user message
            let message = ChatCompletionRequestUserMessageArgs::default()
                .content(user_message.content.clone())
                .build()
                .expect("failed to build user message");
            chat_messages.push(message.into());

            let request = CreateChatCompletionRequestArgs::default()
                .model("gemma-2b-it")
                .max_tokens(512u32)
                .messages(chat_messages)
                .stream(true) // ensure server streams
                .build()
                .expect("failed to build request");

            // Send request
            let config = OpenAIConfig::new().with_api_base("http://localhost:8080/v1".to_string());
            let client = Client::with_config(config);

            match client.chat().create_stream(request).await {
                Ok(mut stream) => {
                    // Insert a placeholder assistant message to append into
                    let assistant_id = Uuid::new_v4().to_string();
                    set_messages.update(|msgs| {
                        msgs.push_back(Message {
                            id: assistant_id.clone(),
                            role: "assistant".to_string(),
                            content: String::new(),
                            timestamp: Date::now(),
                        });
                    });

                    // Stream loop: append deltas to the last message
                    while let Some(next) = stream.next().await {
                        match next {
                            Ok(chunk) => {
                                // Try to pull out the content delta in a tolerant way.
                                // async-openai 0.28.x stream chunk usually looks like:
                                // choices[0].delta.content: Option<String>
                                let mut delta_txt = String::new();

                                if let Some(choice) = chunk.choices.get(0) {
                                    // Newer message API may expose different shapes; try common ones
                                    // 1) Simple string content delta
                                    if let Some(content) = &choice.delta.content {
                                        delta_txt.push_str(content);
                                    }

                                    // 2) Some providers pack text under .delta.role/.delta.<other>
                                    //    If nothing extracted, ignore quietly.

                                    // If a finish_reason arrives, we could stop early,
                                    // but usually the stream naturally ends.
                                }

                                if !delta_txt.is_empty() {
                                    set_messages.update(|msgs| {
                                        if let Some(last) = msgs.back_mut() {
                                            if last.role == "assistant" {
                                                last.content.push_str(&delta_txt);
                                                last.timestamp = Date::now();
                                            }
                                        }
                                    });
                                }
                            }
                            Err(e) => {
                                log::error!("Stream error: {:?}", e);
                                set_messages.update(|msgs| {
                                    msgs.push_back(Message {
                                        id: Uuid::new_v4().to_string(),
                                        role: "system".to_string(),
                                        content: format!("Stream error: {e}"),
                                        timestamp: Date::now(),
                                    });
                                });
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to send request: {:?}", e);
                    let error_message = Message {
                        id: Uuid::new_v4().to_string(),
                        role: "system".to_string(),
                        content: "Error: Failed to connect to server".to_string(),
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

// 
// #[component]
// fn ChatInterface() -> impl IntoView {
//     let (messages, set_messages) = create_signal::<VecDeque<Message>>(VecDeque::new());
//     let (input_value, set_input_value) = create_signal(String::new());
//     let (is_loading, set_is_loading) = create_signal(false);
// 
//     let send_message = create_action(move |content: &String| {
//         let content = content.clone();
//         async move {
//             if content.trim().is_empty() {
//                 return;
//             }
// 
//             set_is_loading.set(true);
// 
//             // Add user message to chat
//             let user_message = Message {
//                 id: Uuid::new_v4().to_string(),
//                 role: "user".to_string(),
//                 content: content.clone(),
//                 timestamp: Date::now(),
//             };
// 
//             set_messages.update(|msgs| msgs.push_back(user_message.clone()));
//             set_input_value.set(String::new());
// 
//             let mut chat_messages = Vec::new();
// 
//             // Add system message
//             let system_message = ChatCompletionRequestSystemMessageArgs::default()
//                 .content("You are a helpful assistant.")
//                 .build()
//                 .expect("failed to build system message");
//             chat_messages.push(system_message.into());
// 
//             // Add history messages
//             messages.with(|msgs| {
//                 for msg in msgs.iter() {
//                     let message = ChatCompletionRequestUserMessageArgs::default()
//                         .content(msg.content.clone().into())
//                         .build()
//                         .expect("failed to build message");
//                     chat_messages.push(message.into());
//                 }
//             });
// 
//             // Add current user message
//             let message = ChatCompletionRequestUserMessageArgs::default()
//                 .content(user_message.content.clone().into())
//                 .build()
//                 .expect("failed to build user message");
//             chat_messages.push(message.into());
// 
//             let request = CreateChatCompletionRequestArgs::default()
//                 .model("gemma-2b-it")
//                 .max_tokens(512u32)
//                 .messages(chat_messages)
//                 .build()
//                 .expect("failed to build request");
// 
//             // Send request
//             let config = OpenAIConfig::new().with_api_base("http://localhost:8080".to_string());
//             let client = Client::with_config(config);
// 
//             match client
//                 .chat()
//                 .create_stream(request)
//                 .await
//             {
//                 Ok(chat_response) => {
// 
// 
//                     // if let Some(choice) = chat_response {
//                     //     // Extract content from the message
//                     //     let content_text = match &choice.message.content {
//                     //         Some(message_content) => {
//                     //             match &message_content.0 {
//                     //                 either::Either::Left(text) => text.clone(),
//                     //                 either::Either::Right(_) => "Complex content not supported".to_string(),
//                     //             }
//                     //         }
//                     //         None => "No content provided".to_string(),
//                     //     };
//                     //
//                     //     let assistant_message = Message {
//                     //         id: Uuid::new_v4().to_string(),
//                     //         role: "assistant".to_string(),
//                     //         content: content_text,
//                     //         timestamp: Date::now(),
//                     //     };
//                     //     set_messages.update(|msgs| msgs.push_back(assistant_message));
//                     //
//                     //
//                     //
//                     //     // Log token usage information
//                     //     log::debug!("Token usage - Prompt: {}, Completion: {}, Total: {}",
//                     //         chat_response.usage.prompt_tokens,
//                     //         chat_response.usage.completion_tokens,
//                     //         chat_response.usage.total_tokens);
//                     // }
//                 }
//                 Err(e) => {
//                     log::error!("Failed to send request: {:?}", e);
//                     let error_message = Message {
//                         id: Uuid::new_v4().to_string(),
//                         role: "system".to_string(),
//                         content: "Error: Failed to connect to server".to_string(),
//                         timestamp: Date::now(),
//                     };
//                     set_messages.update(|msgs| msgs.push_back(error_message));
//                 }
//             }
// 
//             set_is_loading.set(false);
//         }
//     });
// 
//     let on_input = move |ev| {
//         let input = event_target::<HtmlInputElement>(&ev);
//         set_input_value.set(input.value());
//     };
// 
//     let on_submit = move |ev: SubmitEvent| {
//         ev.prevent_default();
//         let content = input_value.get();
//         send_message.dispatch(content);
//     };
// 
//     let on_keypress = move |ev: KeyboardEvent| {
//         if ev.key() == "Enter" && !ev.shift_key() {
//             ev.prevent_default();
//             let content = input_value.get();
//             send_message.dispatch(content);
//         }
//     };
// 
//     let messages_list = move || {
//         messages.get()
//             .into_iter()
//             .map(|message| {
//                 let role_class = match message.role.as_str() {
//                     "user" => "user-message",
//                     "assistant" => "assistant-message",
//                     _ => "system-message",
//                 };
// 
//                 view! {
//                     <div class=format!("message {}", role_class)>
//                         <div class="message-role">{message.role}</div>
//                         <div class="message-content">{message.content}</div>
//                     </div>
//                 }
//             })
//             .collect_view()
//     };
// 
//     let loading_indicator = move || {
//         is_loading.get().then(|| {
//             view! {
//                 <div class="message assistant-message">
//                     <div class="message-role">"assistant"</div>
//                     <div class="message-content">"Thinking..."</div>
//                 </div>
//             }
//         })
//     };
// 
//     view! {
//         <div class="chat-container">
//             <h1>"Chat Interface"</h1>
//             <div class="messages-container">
//                 {messages_list}
//                 {loading_indicator}
//             </div>
//             <form class="input-form" on:submit=on_submit>
//                 <input
//                     type="text"
//                     class="message-input"
//                     placeholder="Type your message here..."
//                     prop:value=input_value
//                     on:input=on_input
//                     on:keypress=on_keypress
//                     prop:disabled=is_loading
//                 />
//                 <button
//                     type="submit"
//                     class="send-button"
//                     prop:disabled=move || is_loading.get() || input_value.get().trim().is_empty()
//                 >
//                     "Send"
//                 </button>
//             </form>
//         </div>
//     }
// }

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    // Set up error handling and logging for WebAssembly
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("error initializing logger");

    // Mount the App component to the document body

    leptos::mount_to_body(App)
}