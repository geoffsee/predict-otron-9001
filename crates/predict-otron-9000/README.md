# predict-otron-9000

This is an extensible axum/tokio hybrid combining [embeddings-engine](../embeddings-engine), [inference-engine](../inference-engine), and [leptos-app](../leptos-app).


# Notes
- When `server_mode` is Standalone (default), the instance contains all components necessary for inference.
- When `server_mode` is HighAvailability, automatic scaling of inference and embeddings; proxies to inference and embeddings services via dns 
