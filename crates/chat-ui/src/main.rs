#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    use axum::Router;
    use chat_ui::app::*;
    use leptos::logging::log;
    use leptos::prelude::*;
    use leptos_axum::{generate_route_list, LeptosRoutes};

    let conf = get_configuration(None).expect("failed to read config");
    let addr = conf.leptos_options.site_addr;

    // Build the app router with your extracted function
    let app: Router = create_router(conf.leptos_options);

    log!("listening on http://{}", &addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

#[cfg(not(feature = "ssr"))]
pub fn main() {
    // no client-side main function
}
