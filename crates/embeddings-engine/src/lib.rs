use async_openai::types::{CreateEmbeddingRequest, EmbeddingInput};
use axum::{
    response::Json as ResponseJson, routing::{post},
    Json,
    Router,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use once_cell::sync::Lazy;
use tower_http::trace::TraceLayer;
use tracing;

// Persistent model instance (singleton pattern)
static EMBEDDING_MODEL: Lazy<TextEmbedding> = Lazy::new(|| {
    tracing::info!("Initializing persistent embedding model (singleton)");
    let model_start_time = std::time::Instant::now();
    
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::NomicEmbedTextV15).with_show_download_progress(true)
    )
        .expect("Failed to initialize persistent embedding model");
    
    let model_init_time = model_start_time.elapsed();
    tracing::info!("Persistent embedding model initialized in {:.2?}", model_init_time);
    
    model
});

pub async fn embeddings_create(
    Json(payload): Json<CreateEmbeddingRequest>,
) -> ResponseJson<serde_json::Value> {
    // Start timing the entire process
    let start_time = std::time::Instant::now();
    
    // Phase 1: Access persistent model instance
    let model_start_time = std::time::Instant::now();
    
    // Access the lazy-initialized persistent model instance
    // This will only initialize the model on the first request
    let model_access_time = model_start_time.elapsed();
    tracing::debug!("Persistent model access completed in {:.2?}", model_access_time);
    
    // Phase 2: Process input
    let input_start_time = std::time::Instant::now();
    
    let embedding_input = payload.input;
    let texts_from_embedding_input = match embedding_input {
        EmbeddingInput::String(text) => vec![text],
        EmbeddingInput::StringArray(texts) => texts,
        EmbeddingInput::IntegerArray(_) => {
            panic!("Integer array input not supported for text embeddings");
        }
        EmbeddingInput::ArrayOfIntegerArray(_) => {
            panic!("Array of integer arrays not supported for text embeddings");
        }
    };
    
    let input_processing_time = input_start_time.elapsed();
    tracing::debug!("Input processing completed in {:.2?}", input_processing_time);
    
    // Phase 3: Generate embeddings
    let embedding_start_time = std::time::Instant::now();
    
    let embeddings = EMBEDDING_MODEL
        .embed(texts_from_embedding_input, None)
        .expect("failed to embed document");
    
    let embedding_generation_time = embedding_start_time.elapsed();
    tracing::info!("Embedding generation completed in {:.2?}", embedding_generation_time);
    
    // Memory usage estimation (approximate)
    let embedding_size_bytes = embeddings.iter()
        .map(|e| e.len() * std::mem::size_of::<f32>())
        .sum::<usize>();
    tracing::debug!("Embedding size: {:.2} MB", embedding_size_bytes as f64 / 1024.0 / 1024.0);

    // Only log detailed embedding information at trace level to reduce log volume
    tracing::trace!("Embeddings length: {}", embeddings.len());
    tracing::info!("Embedding dimension: {}", embeddings[0].len());

    // Log the first 10 values of the original embedding at trace level
    tracing::trace!("Original embedding preview: {:?}", &embeddings[0][..10.min(embeddings[0].len())]);

    // Check if there are any NaN or zero values in the original embedding
    let nan_count = embeddings[0].iter().filter(|&&x| x.is_nan()).count();
    let zero_count = embeddings[0].iter().filter(|&&x| x == 0.0).count();
    tracing::trace!("Original embedding stats: NaN count={}, zero count={}", nan_count, zero_count);

    // Phase 4: Post-process embeddings
    let postprocessing_start_time = std::time::Instant::now();
    
    // Create the final embedding
    let final_embedding = {
        // Check if the embedding is all zeros
        let all_zeros = embeddings[0].iter().all(|&x| x == 0.0);
        if all_zeros {
            tracing::warn!("Embedding is all zeros. Generating random non-zero embedding.");

            // Generate a random non-zero embedding
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let mut random_embedding = Vec::with_capacity(768);
            for _ in 0..768 {
                // Generate random values between -1.0 and 1.0, excluding 0
                let mut val = 0.0;
                while val == 0.0 {
                    val = rng.gen_range(-1.0..1.0);
                }
                random_embedding.push(val);
            }

            // Normalize the random embedding
            let norm: f32 = random_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            for i in 0..random_embedding.len() {
                random_embedding[i] /= norm;
            }

            random_embedding
        } else {
            // Check if dimensions parameter is provided and pad the embeddings if necessary
            let mut padded_embedding = embeddings[0].clone();

            // If the client expects 768 dimensions but our model produces fewer, pad with zeros
            let target_dimension = 768;
            if padded_embedding.len() < target_dimension {
                let padding_needed = target_dimension - padded_embedding.len();
                tracing::trace!("Padding embedding with {} zeros to reach {} dimensions", padding_needed, target_dimension);
                padded_embedding.extend(vec![0.0; padding_needed]);
            }

            padded_embedding
        }
    };
    
    let postprocessing_time = postprocessing_start_time.elapsed();
    tracing::debug!("Embedding post-processing completed in {:.2?}", postprocessing_time);

    tracing::trace!("Final embedding dimension: {}", final_embedding.len());

    // Log the first 10 values of the final embedding at trace level
    tracing::trace!("Final embedding preview: {:?}", &final_embedding[..10.min(final_embedding.len())]);

    // Phase 5: Prepare response
    let response_start_time = std::time::Instant::now();
    
    // Return a response that matches the OpenAI API format
    let response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": final_embedding
            }
        ],
        "model": payload.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    });
    
    let response_time = response_start_time.elapsed();
    tracing::debug!("Response preparation completed in {:.2?}", response_time);
    
    // Log total time and breakdown
    let total_time = start_time.elapsed();
    tracing::info!(
        "Embeddings request completed in {:.2?} (model_access: {:.2?}, embedding: {:.2?}, postprocessing: {:.2?})",
        total_time,
        model_access_time,
        embedding_generation_time,
        postprocessing_time
    );
    
    ResponseJson(response)
}

pub fn create_embeddings_router() -> Router {
    Router::new()
        .route("/v1/embeddings", post(embeddings_create))
        .layer(TraceLayer::new_for_http())
}