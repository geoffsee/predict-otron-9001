#[cfg(feature = "ssr")]
use axum::Router;
#[cfg(feature = "ssr")]
use leptos::prelude::LeptosOptions;
#[cfg(feature = "ssr")]
use leptos_axum::{generate_route_list, LeptosRoutes};

pub struct AppConfig {
    pub config: ConfFile,
    pub address: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        let conf = get_configuration(Some(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")))
            .expect("failed to read config");

        let addr = conf.leptos_options.site_addr;

        AppConfig {
            config: conf, // or whichever field/string representation you need
            address: addr.to_string(),
        }
    }
}

/// Build the Axum router for this app, including routes, fallback, and state.
/// Call this from another crate (or your bin) when running the server.
#[cfg(feature = "ssr")]
pub fn create_router(leptos_options: LeptosOptions) -> Router {
    // Generate the list of routes in your Leptos App
    let routes = generate_route_list(App);

    Router::new()
        .leptos_routes(&leptos_options, routes, {
            let leptos_options = leptos_options.clone();
            move || shell(leptos_options.clone())
        })
        .fallback(leptos_axum::file_and_error_handler(shell))
        .with_state(leptos_options)
}

use gloo_net::http::Request;
use leptos::prelude::*;
use leptos_meta::{provide_meta_context, MetaTags, Stylesheet, Title};
use leptos_router::{
    components::{Route, Router, Routes},
    StaticSegment,
};
use serde::{Deserialize, Serialize};
use web_sys::console;
// Remove spawn_local import as we'll use different approach

// Data structures for OpenAI-compatible API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

// Data structures for models API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

// API client function to fetch available models
pub async fn fetch_models() -> Result<Vec<ModelInfo>, String> {
    let response = Request::get("/v1/models")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {:?}", e))?;

    if response.ok() {
        let models_response: ModelsResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse models response: {:?}", e))?;
        Ok(models_response.data)
    } else {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(format!("Failed to fetch models {}: {}", status, error_text))
    }
}

// API client function to send chat completion requests
pub async fn send_chat_completion(
    messages: Vec<ChatMessage>,
    model: String,
) -> Result<String, String> {
    let request = ChatRequest {
        model,
        messages,
        max_tokens: Some(1024),
        stream: Some(false),
    };

    let response = Request::post("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .json(&request)
        .map_err(|e| format!("Failed to create request: {:?}", e))?
        .send()
        .await
        .map_err(|e| format!("Failed to send request: {:?}", e))?;

    if response.ok() {
        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {:?}", e))?;

        if let Some(choice) = chat_response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err("No response choices available".to_string())
        }
    } else {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(format!("Server error {}: {}", status, error_text))
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
        <Stylesheet id="leptos" href="/pkg/chat-ui.css"/>

        // sets the document title
        <Title text="Predict-Otron-9000 Chat"/>

        // content for this welcome page
        <Router>
            <main>
                <Routes fallback=|| "Page not found.".into_view()>
                    <Route path=StaticSegment("") view=ChatPage/>
                </Routes>
            </main>
        </Router>
    }
}

/// Renders the chat interface page
#[component]
fn ChatPage() -> impl IntoView {
    // State for conversation messages
    let messages = RwSignal::new(Vec::<ChatMessage>::new());

    // State for current user input
    let input_text = RwSignal::new(String::new());

    // State for loading indicator
    let is_loading = RwSignal::new(false);

    // State for error messages
    let error_message = RwSignal::new(Option::<String>::None);

    // State for available models and selected model
    let available_models = RwSignal::new(Vec::<ModelInfo>::new());
    let selected_model = RwSignal::new(String::from("gemma-3-1b-it")); // Default model

    // Client-side only: Fetch models on component mount
    #[cfg(target_arch = "wasm32")]
    {
        use leptos::task::spawn_local;
        spawn_local(async move {
            match fetch_models().await {
                Ok(models) => {
                    available_models.set(models);
                }
                Err(error) => {
                    console::log_1(&format!("Failed to fetch models: {}", error).into());
                    error_message.set(Some(format!("Failed to load models: {}", error)));
                }
            }
        });
    }

    // Shared logic for sending a message
    let send_message_logic = move || {
        let user_input = input_text.get();
        if user_input.trim().is_empty() {
            return;
        }

        // Add user message to conversation
        let user_message = ChatMessage {
            role: "user".to_string(),
            content: user_input.clone(),
        };

        messages.update(|msgs| msgs.push(user_message.clone()));
        input_text.set(String::new());
        is_loading.set(true);
        error_message.set(None);

        // Client-side only: Send chat completion request
        #[cfg(target_arch = "wasm32")]
        {
            use leptos::task::spawn_local;

            // Prepare messages for API call
            let current_messages = messages.get();
            let current_model = selected_model.get();

            // Spawn async task to call API
            spawn_local(async move {
                match send_chat_completion(current_messages, current_model).await {
                    Ok(response_content) => {
                        let assistant_message = ChatMessage {
                            role: "assistant".to_string(),
                            content: response_content,
                        };
                        messages.update(|msgs| msgs.push(assistant_message));
                        is_loading.set(false);
                    }
                    Err(error) => {
                        console::log_1(&format!("API Error: {}", error).into());
                        error_message.set(Some(error));
                        is_loading.set(false);
                    }
                }
            });
        }
    };

    // Button click handler
    let on_button_click = {
        let send_logic = send_message_logic.clone();
        move |_: web_sys::MouseEvent| {
            send_logic();
        }
    };

    // Handle enter key press in input field
    let on_key_down = move |ev: web_sys::KeyboardEvent| {
        if ev.key() == "Enter" && !ev.shift_key() {
            ev.prevent_default();
            send_message_logic();
        }
    };

    view! {
        <div class="chat-container">
            <div class="chat-header">
                <h1>"Predict-Otron-9000 Chat"</h1>
                <div class="model-selector">
                    <label for="model-select">"Model:"</label>
                    <select
                        id="model-select"
                        prop:value=move || selected_model.get()
                        on:change=move |ev| {
                            let new_model = event_target_value(&ev);
                            selected_model.set(new_model);
                        }
                    >
                        <For
                            each=move || available_models.get().into_iter()
                            key=|model| model.id.clone()
                            children=move |model| {
                                view! {
                                    <option value=model.id.clone()>
                                        {format!("{} ({})", model.id, model.owned_by)}
                                    </option>
                                }
                            }
                        />
                    </select>
                </div>
            </div>

            <div class="chat-messages">
                <For
                    each=move || messages.get().into_iter().enumerate()
                    key=|(i, _)| *i
                    children=move |(_, message)| {
                        let role_class = if message.role == "user" { "user-message" } else { "assistant-message" };
                        view! {
                            <div class=format!("message {}", role_class)>
                                <div class="message-role">{message.role.clone()}</div>
                                <div class="message-content">{message.content.clone()}</div>
                            </div>
                        }
                    }
                />

                {move || {
                    if is_loading.get() {
                        view! {
                            <div class="message assistant-message loading">
                                <div class="message-role">"assistant"</div>
                                <div class="message-content">"Thinking..."</div>
                            </div>
                        }.into_any()
                    } else {
                        view! {}.into_any()
                    }
                }}
            </div>

            {move || {
                if let Some(error) = error_message.get() {
                    view! {
                        <div class="error-message">
                            "Error: " {error}
                        </div>
                    }.into_any()
                } else {
                    view! {}.into_any()
                }
            }}

            <div class="chat-input">
                <textarea
                    placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
                    prop:value=move || input_text.get()
                    on:input=move |ev| input_text.set(event_target_value(&ev))
                    on:keydown=on_key_down
                    class:disabled=move || is_loading.get()
                />
                <button
                    on:click=on_button_click
                    class:disabled=move || is_loading.get() || input_text.get().trim().is_empty()
                >
                    "Send"
                </button>
            </div>
        </div>
    }
}
