use std::path::PathBuf;

pub mod app;

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    use crate::app::*;
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(App);
}

#[cfg(feature = "ssr")]
pub fn create_leptos_router() -> axum::Router {
    use crate::app::*;
    use axum::Router;
    use leptos::prelude::*;
    use leptos_axum::{generate_route_list, LeptosRoutes};


    // Build an absolute path to THIS crate's Cargo.toml
    let mut cargo_toml = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    cargo_toml.push("Cargo.toml");

    let conf = get_configuration(Some(
        cargo_toml.to_str().expect("valid utf-8 path to Cargo.toml"),
    ))
        .expect("load leptos config");

    let conf = get_configuration(Some(cargo_toml.to_str().unwrap())).unwrap();
    let leptos_options = conf.leptos_options;
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
