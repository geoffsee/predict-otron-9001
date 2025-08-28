use leptos::prelude::*;
use leptos_meta::{provide_meta_context, MetaTags, Stylesheet, Title};
use leptos_router::{
    components::{Route, Router, Routes},
    StaticSegment,
};

#[cfg(feature = "hydrate")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "hydrate")]
use std::collections::VecDeque;
#[cfg(feature = "hydrate")]
use uuid::Uuid;
#[cfg(feature = "hydrate")]
use js_sys::Date;
#[cfg(feature = "hydrate")]
use web_sys::{HtmlInputElement, KeyboardEvent, SubmitEvent};
#[cfg(feature = "hydrate")]
use futures_util::StreamExt;
#[cfg(feature = "hydrate")]
use async_openai_wasm::{
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, Model as OpenAIModel,
    },
    Client,
};
#[cfg(feature = "hydrate")]
use async_openai_wasm::config::OpenAIConfig;
#[cfg(feature = "hydrate")]
use async_openai_wasm::types::{Role, FinishReason};
#[cfg(feature = "hydrate")]
use leptos::task::spawn_local;

#[cfg(feature = "hydrate")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: f64,
}

#[cfg(feature = "hydrate")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContent(pub either::Either<String, Vec<std::collections::HashMap<String, MessageInnerContent>>>);

#[cfg(feature = "hydrate")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageInnerContent(pub either::Either<String, std::collections::HashMap<String, String>>);

#[cfg(feature = "hydrate")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<MessageContent>,
    pub name: Option<String>,
}

#[cfg(feature = "hydrate")]
const DEFAULT_MODEL: &str = "default";

#[cfg(feature = "hydrate")]
async fn fetch_available_models() -> Result<Vec<OpenAIModel>, String> {
    leptos::logging::log!("[DEBUG_LOG] fetch_available_models: Starting model fetch from http://localhost:8080/v1");
    
    let config = OpenAIConfig::new().with_api_base("http://localhost:8080/v1".to_string());
    let client = Client::with_config(config);
    
    match client.models().list().await {
        Ok(response) => {
            let model_count = response.data.len();
            leptos::logging::log!("[DEBUG_LOG] fetch_available_models: Successfully fetched {} models", model_count);
            
            if model_count > 0 {
                let model_names: Vec<String> = response.data.iter().map(|m| m.id.clone()).collect();
                leptos::logging::log!("[DEBUG_LOG] fetch_available_models: Available models: {:?}", model_names);
            } else {
                leptos::logging::log!("[DEBUG_LOG] fetch_available_models: No models returned by server");
            }
            
            Ok(response.data)
        },
        Err(e) => {
            leptos::logging::log!("[DEBUG_LOG] fetch_available_models: Failed to fetch models: {:?}", e);
            Err(format!("Failed to fetch models: {}", e))
        }
    }
}

pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <AutoReload options=options.clone() />
                <HydrationScripts options/>
                <MetaTags/>
            </head>
            <body>
                <App/>
            </body>
        </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context();

    view! {
        // injects a stylesheet into the document <head>
        // id=leptos means cargo-leptos will hot-reload this stylesheet
        <Stylesheet id="leptos" href="/pkg/leptos-app.css"/>

        // sets the document title
        <Title text="Chat Interface"/>

        // content for this chat interface
        <Router>
            <main>
                <Routes fallback=|| "Page not found.".into_view()>
                    <Route path=StaticSegment("") view=ChatInterface/>
                </Routes>
            </main>
        </Router>
    }
}

/// Renders the home page of your application.
#[component]
fn HomePage() -> impl IntoView {
    // Creates a reactive value to update the button
    let count = RwSignal::new(0);
    let on_click = move |_| *count.write() += 1;

    view! {
        <h1>"Welcome to Leptos!"</h1>
        <button on:click=on_click>"Click Me: " {count}</button>
    }
}

/// Renders the chat interface
#[component]
fn ChatInterface() -> impl IntoView {
    #[cfg(feature = "hydrate")]
    {
        ChatInterfaceImpl()
    }
    
    #[cfg(not(feature = "hydrate"))]
    {
        view! {
            <div class="chat-container">
                <h1>"Chat Interface"</h1>
                <p>"Loading chat interface..."</p>
            </div>
        }
    }
}

#[cfg(feature = "hydrate")]
#[component]
fn ChatInterfaceImpl() -> impl IntoView {
    let (messages, set_messages) = RwSignal::new(VecDeque::<Message>::new()).split();
    let (input_value, set_input_value) = RwSignal::new(String::new()).split();
    let (is_loading, set_is_loading) = RwSignal::new(false).split();
    let (available_models, set_available_models) = RwSignal::new(Vec::<OpenAIModel>::new()).split();
    let (selected_model, set_selected_model) = RwSignal::new(DEFAULT_MODEL.to_string()).split();
    let (models_loading, set_models_loading) = RwSignal::new(false).split();

    // Fetch models on component initialization
    Effect::new(move |_| {
        spawn_local(async move {
            set_models_loading.set(true);
            match fetch_available_models().await {
                Ok(models) => {
                    set_available_models.set(models);
                    set_models_loading.set(false);
                }
                Err(e) => {
                    leptos::logging::log!("Failed to fetch models: {}", e);
                    set_available_models.set(vec![]);
                    set_models_loading.set(false);
                }
            }
        });
    });

    let send_message = Action::new_unsync(move |content: &String| {
        let content = content.clone();
        async move {
            if content.trim().is_empty() {
                leptos::logging::log!("[DEBUG_LOG] send_message: Empty content, skipping");
                return;
            }

            leptos::logging::log!("[DEBUG_LOG] send_message: Starting message send process");
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
            let history_count = messages.get_untracked().len();
            for msg in messages.get_untracked().iter() {
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
                    _ => {}
                }
            }

            // Add current user message
            let message = ChatCompletionRequestUserMessageArgs::default()
                .content(user_message.content.clone())
                .build()
                .expect("failed to build user message");
            chat_messages.push(message.into());

            let current_model = selected_model.get_untracked();
            let total_messages = chat_messages.len();
            
            leptos::logging::log!("[DEBUG_LOG] send_message: Preparing request - model: '{}', history_count: {}, total_messages: {}", 
                          current_model, history_count, total_messages);

            let request = CreateChatCompletionRequestArgs::default()
                .model(current_model.as_str())
                .max_tokens(512u32)
                .messages(chat_messages)
                .stream(true)
                .build()
                .expect("failed to build request");

            // Send request
            let config = OpenAIConfig::new().with_api_base("http://localhost:8080/v1".to_string());
            let client = Client::with_config(config);
            
            leptos::logging::log!("[DEBUG_LOG] send_message: Sending request to http://localhost:8080/v1 with model: '{}'", current_model);

            match client.chat().create_stream(request).await {
                Ok(mut stream) => {
                    leptos::logging::log!("[DEBUG_LOG] send_message: Successfully created stream");
                    
                    let mut assistant_created = false;
                    let mut content_appended = false;
                    let mut chunks_received = 0;
                    
                    while let Some(next) = stream.next().await {
                        match next {
                            Ok(chunk) => {
                                chunks_received += 1;
                                if let Some(choice) = chunk.choices.get(0) {
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

                                    if let Some(content) = &choice.delta.content {
                                        if !content.is_empty() {
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

                                    if let Some(reason) = &choice.finish_reason {
                                        if reason == &FinishReason::Stop {
                                            leptos::logging::log!("[DEBUG_LOG] send_message: Received finish_reason=stop after {} chunks", chunks_received);
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                leptos::logging::log!("[DEBUG_LOG] send_message: Stream error after {} chunks: {:?}", chunks_received, e);
                                set_messages.update(|msgs| {
                                    msgs.push_back(Message {
                                        id: Uuid::new_v4().to_string(),
                                        role: "system".to_string(),
                                        content: format!("Stream error: {}", e),
                                        timestamp: Date::now(),
                                    });
                                });
                                break;
                            }
                        }
                    }

                    if assistant_created && !content_appended {
                        set_messages.update(|msgs| {
                            let should_pop = msgs
                                .back()
                                .map(|m| m.role == "assistant" && m.content.is_empty())
                                .unwrap_or(false);
                            if should_pop {
                                msgs.pop_back();
                            }
                        });
                    }

                    leptos::logging::log!("[DEBUG_LOG] send_message: Stream completed successfully, received {} chunks", chunks_received);
                }
                Err(e) => {
                    leptos::logging::log!("[DEBUG_LOG] send_message: Request failed with error: {:?}", e);
                    let error_message = Message {
                        id: Uuid::new_v4().to_string(),
                        role: "system".to_string(),
                        content: format!("Error: Request failed - {}", e),
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
            .collect::<Vec<_>>()
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
                            vec![view! {
                                <option value={String::from("")} selected=false>{String::from("Loading models...")}</option>
                            }]
                        } else {
                            let models = available_models.get();
                            if models.is_empty() {
                                vec![view! {
                                    <option value={String::from("default")} selected=true>{String::from("default")}</option>
                                }]
                            } else {
                                models.into_iter().map(|model| {
                                    view! {
                                        <option value=model.id.clone() selected={model.id == DEFAULT_MODEL}>{model.id.clone()}</option>
                                    }
                                }).collect::<Vec<_>>()
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
