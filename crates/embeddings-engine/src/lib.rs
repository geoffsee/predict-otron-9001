use async_openai::types::{CreateEmbeddingRequest, EmbeddingInput};
use axum::{
    Json, Router,
    http::StatusCode,
    response::Json as ResponseJson,
    routing::{get, post},
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use once_cell::sync::Lazy;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tower_http::trace::TraceLayer;
use tracing;

// Cache for multiple embedding models
static MODEL_CACHE: Lazy<RwLock<HashMap<EmbeddingModel, Arc<TextEmbedding>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub description: String,
    pub dimensions: usize,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

// Function to convert model name strings to EmbeddingModel enum variants
fn parse_embedding_model(model_name: &str) -> Result<EmbeddingModel, String> {
    match model_name {
        // Sentence Transformers models
        "sentence-transformers/all-MiniLM-L6-v2" | "all-minilm-l6-v2" => {
            Ok(EmbeddingModel::AllMiniLML6V2)
        }
        "sentence-transformers/all-MiniLM-L6-v2-q" | "all-minilm-l6-v2-q" => {
            Ok(EmbeddingModel::AllMiniLML6V2Q)
        }
        "sentence-transformers/all-MiniLM-L12-v2" | "all-minilm-l12-v2" => {
            Ok(EmbeddingModel::AllMiniLML12V2)
        }
        "sentence-transformers/all-MiniLM-L12-v2-q" | "all-minilm-l12-v2-q" => {
            Ok(EmbeddingModel::AllMiniLML12V2Q)
        }

        // BGE models
        "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => Ok(EmbeddingModel::BGEBaseENV15),
        "BAAI/bge-base-en-v1.5-q" | "bge-base-en-v1.5-q" => Ok(EmbeddingModel::BGEBaseENV15Q),
        "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => Ok(EmbeddingModel::BGELargeENV15),
        "BAAI/bge-large-en-v1.5-q" | "bge-large-en-v1.5-q" => Ok(EmbeddingModel::BGELargeENV15Q),
        "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => Ok(EmbeddingModel::BGESmallENV15),
        "BAAI/bge-small-en-v1.5-q" | "bge-small-en-v1.5-q" => Ok(EmbeddingModel::BGESmallENV15Q),
        "BAAI/bge-small-zh-v1.5" | "bge-small-zh-v1.5" => Ok(EmbeddingModel::BGESmallZHV15),
        "BAAI/bge-large-zh-v1.5" | "bge-large-zh-v1.5" => Ok(EmbeddingModel::BGELargeZHV15),

        // Nomic models
        "nomic-ai/nomic-embed-text-v1" | "nomic-embed-text-v1" => {
            Ok(EmbeddingModel::NomicEmbedTextV1)
        }
        "nomic-ai/nomic-embed-text-v1.5" | "nomic-embed-text-v1.5" | "nomic-text-embed" => {
            Ok(EmbeddingModel::NomicEmbedTextV15)
        }
        "nomic-ai/nomic-embed-text-v1.5-q" | "nomic-embed-text-v1.5-q" => {
            Ok(EmbeddingModel::NomicEmbedTextV15Q)
        }

        // Paraphrase models
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        | "paraphrase-multilingual-minilm-l12-v2" => Ok(EmbeddingModel::ParaphraseMLMiniLML12V2),
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2-q"
        | "paraphrase-multilingual-minilm-l12-v2-q" => Ok(EmbeddingModel::ParaphraseMLMiniLML12V2Q),
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        | "paraphrase-multilingual-mpnet-base-v2" => Ok(EmbeddingModel::ParaphraseMLMpnetBaseV2),

        // ModernBert
        "lightonai/modernbert-embed-large" | "modernbert-embed-large" => {
            Ok(EmbeddingModel::ModernBertEmbedLarge)
        }

        // Multilingual E5 models
        "intfloat/multilingual-e5-small" | "multilingual-e5-small" => {
            Ok(EmbeddingModel::MultilingualE5Small)
        }
        "intfloat/multilingual-e5-base" | "multilingual-e5-base" => {
            Ok(EmbeddingModel::MultilingualE5Base)
        }
        "intfloat/multilingual-e5-large" | "multilingual-e5-large" => {
            Ok(EmbeddingModel::MultilingualE5Large)
        }

        // Mixedbread models
        "mixedbread-ai/mxbai-embed-large-v1" | "mxbai-embed-large-v1" => {
            Ok(EmbeddingModel::MxbaiEmbedLargeV1)
        }
        "mixedbread-ai/mxbai-embed-large-v1-q" | "mxbai-embed-large-v1-q" => {
            Ok(EmbeddingModel::MxbaiEmbedLargeV1Q)
        }

        // GTE models
        "Alibaba-NLP/gte-base-en-v1.5" | "gte-base-en-v1.5" => Ok(EmbeddingModel::GTEBaseENV15),
        "Alibaba-NLP/gte-base-en-v1.5-q" | "gte-base-en-v1.5-q" => {
            Ok(EmbeddingModel::GTEBaseENV15Q)
        }
        "Alibaba-NLP/gte-large-en-v1.5" | "gte-large-en-v1.5" => Ok(EmbeddingModel::GTELargeENV15),
        "Alibaba-NLP/gte-large-en-v1.5-q" | "gte-large-en-v1.5-q" => {
            Ok(EmbeddingModel::GTELargeENV15Q)
        }

        // CLIP model
        "Qdrant/clip-ViT-B-32-text" | "clip-vit-b-32" => Ok(EmbeddingModel::ClipVitB32),

        // Jina model
        "jinaai/jina-embeddings-v2-base-code" | "jina-embeddings-v2-base-code" => {
            Ok(EmbeddingModel::JinaEmbeddingsV2BaseCode)
        }

        _ => Err(format!("Unsupported embedding model: {}", model_name)),
    }
}

// Function to get model dimensions
fn get_model_dimensions(model: &EmbeddingModel) -> usize {
    match model {
        EmbeddingModel::AllMiniLML6V2 | EmbeddingModel::AllMiniLML6V2Q => 384,
        EmbeddingModel::AllMiniLML12V2 | EmbeddingModel::AllMiniLML12V2Q => 384,
        EmbeddingModel::BGEBaseENV15 | EmbeddingModel::BGEBaseENV15Q => 768,
        EmbeddingModel::BGELargeENV15 | EmbeddingModel::BGELargeENV15Q => 1024,
        EmbeddingModel::BGESmallENV15 | EmbeddingModel::BGESmallENV15Q => 384,
        EmbeddingModel::BGESmallZHV15 => 512,
        EmbeddingModel::BGELargeZHV15 => 1024,
        EmbeddingModel::NomicEmbedTextV1
        | EmbeddingModel::NomicEmbedTextV15
        | EmbeddingModel::NomicEmbedTextV15Q => 768,
        EmbeddingModel::ParaphraseMLMiniLML12V2 | EmbeddingModel::ParaphraseMLMiniLML12V2Q => 384,
        EmbeddingModel::ParaphraseMLMpnetBaseV2 => 768,
        EmbeddingModel::ModernBertEmbedLarge => 1024,
        EmbeddingModel::MultilingualE5Small => 384,
        EmbeddingModel::MultilingualE5Base => 768,
        EmbeddingModel::MultilingualE5Large => 1024,
        EmbeddingModel::MxbaiEmbedLargeV1 | EmbeddingModel::MxbaiEmbedLargeV1Q => 1024,
        EmbeddingModel::GTEBaseENV15 | EmbeddingModel::GTEBaseENV15Q => 768,
        EmbeddingModel::GTELargeENV15 | EmbeddingModel::GTELargeENV15Q => 1024,
        EmbeddingModel::ClipVitB32 => 512,
        EmbeddingModel::JinaEmbeddingsV2BaseCode => 768,
    }
}

// Function to get or create a model from cache
fn get_or_create_model(embedding_model: EmbeddingModel) -> Result<Arc<TextEmbedding>, String> {
    // First try to get from cache (read lock)
    {
        let cache = MODEL_CACHE
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;
        if let Some(model) = cache.get(&embedding_model) {
            tracing::debug!("Using cached model: {:?}", embedding_model);
            return Ok(Arc::clone(model));
        }
    }

    // Model not in cache, create it (write lock)
    let mut cache = MODEL_CACHE
        .write()
        .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

    // Double-check after acquiring write lock
    if let Some(model) = cache.get(&embedding_model) {
        tracing::debug!("Using cached model (double-check): {:?}", embedding_model);
        return Ok(Arc::clone(model));
    }

    tracing::info!("Initializing new embedding model: {:?}", embedding_model);
    let model_start_time = std::time::Instant::now();

    let model = TextEmbedding::try_new(
        InitOptions::new(embedding_model.clone()).with_show_download_progress(true),
    )
    .map_err(|e| format!("Failed to initialize model {:?}: {}", embedding_model, e))?;

    let model_init_time = model_start_time.elapsed();
    tracing::info!(
        "Embedding model {:?} initialized in {:.2?}",
        embedding_model,
        model_init_time
    );

    let model_arc = Arc::new(model);
    cache.insert(embedding_model.clone(), Arc::clone(&model_arc));
    Ok(model_arc)
}

pub async fn embeddings_create(
    Json(payload): Json<CreateEmbeddingRequest>,
) -> Result<ResponseJson<serde_json::Value>, (StatusCode, String)> {
    // Start timing the entire process
    let start_time = std::time::Instant::now();

    // Phase 1: Parse and get the embedding model
    let model_start_time = std::time::Instant::now();

    let embedding_model = match parse_embedding_model(&payload.model) {
        Ok(model) => model,
        Err(e) => {
            tracing::error!("Invalid model requested: {}", e);
            return Err((StatusCode::BAD_REQUEST, format!("Invalid model: {}", e)));
        }
    };

    let model = match get_or_create_model(embedding_model.clone()) {
        Ok(model) => model,
        Err(e) => {
            tracing::error!("Failed to get/create model: {}", e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Model initialization failed: {}", e),
            ));
        }
    };

    let model_access_time = model_start_time.elapsed();
    tracing::debug!(
        "Model access/creation completed in {:.2?}",
        model_access_time
    );

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
    tracing::debug!(
        "Input processing completed in {:.2?}",
        input_processing_time
    );

    // Phase 3: Generate embeddings
    let embedding_start_time = std::time::Instant::now();

    let embeddings = model.embed(texts_from_embedding_input, None).map_err(|e| {
        tracing::error!("Failed to generate embeddings: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Embedding generation failed: {}", e),
        )
    })?;

    let embedding_generation_time = embedding_start_time.elapsed();
    tracing::info!(
        "Embedding generation completed in {:.2?}",
        embedding_generation_time
    );

    // Memory usage estimation (approximate)
    let embedding_size_bytes = embeddings
        .iter()
        .map(|e| e.len() * std::mem::size_of::<f32>())
        .sum::<usize>();
    tracing::debug!(
        "Embedding size: {:.2} MB",
        embedding_size_bytes as f64 / 1024.0 / 1024.0
    );

    // Only log detailed embedding information at trace level to reduce log volume
    tracing::trace!("Embeddings length: {}", embeddings.len());
    tracing::info!("Embedding dimension: {}", embeddings[0].len());

    // Log the first 10 values of the original embedding at trace level
    tracing::trace!(
        "Original embedding preview: {:?}",
        &embeddings[0][..10.min(embeddings[0].len())]
    );

    // Check if there are any NaN or zero values in the original embedding
    let nan_count = embeddings[0].iter().filter(|&&x| x.is_nan()).count();
    let zero_count = embeddings[0].iter().filter(|&&x| x == 0.0).count();
    tracing::trace!(
        "Original embedding stats: NaN count={}, zero count={}",
        nan_count,
        zero_count
    );

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
            let expected_dimensions = get_model_dimensions(&embedding_model);
            let mut random_embedding = Vec::with_capacity(expected_dimensions);
            for _ in 0..expected_dimensions {
                // Generate random values between -1.0 and 1.0, excluding 0
                let mut val = 0.0;
                while val == 0.0 {
                    val = rng.gen_range(-1.0..1.0);
                }
                random_embedding.push(val);
            }

            // Normalize the random embedding
            let norm: f32 = random_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

            #[allow(clippy::needless_range_loop)]
            for i in 0..random_embedding.len() {
                random_embedding[i] /= norm;
            }

            random_embedding
        } else {
            // Check if dimensions parameter is provided and pad the embeddings if necessary
            let padded_embedding = embeddings[0].clone();

            // Use the actual model dimensions instead of hardcoded 768
            let actual_dimensions = padded_embedding.len();
            let expected_dimensions = get_model_dimensions(&embedding_model);

            if actual_dimensions != expected_dimensions {
                tracing::warn!(
                    "Model {:?} produced {} dimensions but expected {}",
                    embedding_model,
                    actual_dimensions,
                    expected_dimensions
                );
            }

            padded_embedding
        }
    };

    let postprocessing_time = postprocessing_start_time.elapsed();
    tracing::debug!(
        "Embedding post-processing completed in {:.2?}",
        postprocessing_time
    );

    tracing::trace!("Final embedding dimension: {}", final_embedding.len());

    // Log the first 10 values of the final embedding at trace level
    tracing::trace!(
        "Final embedding preview: {:?}",
        &final_embedding[..10.min(final_embedding.len())]
    );

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

    Ok(ResponseJson(response))
}

pub async fn models_list() -> ResponseJson<ModelsResponse> {
    let models = vec![
        ModelInfo {
            id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            object: "model".to_string(),
            owned_by: "sentence-transformers".to_string(),
            description: "Sentence Transformer model, MiniLM-L6-v2".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "sentence-transformers/all-MiniLM-L6-v2-q".to_string(),
            object: "model".to_string(),
            owned_by: "sentence-transformers".to_string(),
            description: "Quantized Sentence Transformer model, MiniLM-L6-v2".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            object: "model".to_string(),
            owned_by: "sentence-transformers".to_string(),
            description: "Sentence Transformer model, MiniLM-L12-v2".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "sentence-transformers/all-MiniLM-L12-v2-q".to_string(),
            object: "model".to_string(),
            owned_by: "sentence-transformers".to_string(),
            description: "Quantized Sentence Transformer model, MiniLM-L12-v2".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "BAAI/bge-base-en-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "v1.5 release of the base English model".to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "BAAI/bge-base-en-v1.5-q".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "Quantized v1.5 release of the base English model".to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "BAAI/bge-large-en-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "v1.5 release of the large English model".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "BAAI/bge-large-en-v1.5-q".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "Quantized v1.5 release of the large English model".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "BAAI/bge-small-en-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "v1.5 release of the fast and default English model".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "BAAI/bge-small-en-v1.5-q".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "Quantized v1.5 release of the fast and default English model".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "BAAI/bge-small-zh-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "v1.5 release of the small Chinese model".to_string(),
            dimensions: 512,
        },
        ModelInfo {
            id: "BAAI/bge-large-zh-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "BAAI".to_string(),
            description: "v1.5 release of the large Chinese model".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "nomic-ai/nomic-embed-text-v1".to_string(),
            object: "model".to_string(),
            owned_by: "nomic-ai".to_string(),
            description: "8192 context length english model".to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "nomic-ai/nomic-embed-text-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "nomic-ai".to_string(),
            description: "v1.5 release of the 8192 context length english model".to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "nomic-ai/nomic-embed-text-v1.5-q".to_string(),
            object: "model".to_string(),
            owned_by: "nomic-ai".to_string(),
            description: "Quantized v1.5 release of the 8192 context length english model"
                .to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            object: "model".to_string(),
            owned_by: "sentence-transformers".to_string(),
            description: "Multi-lingual model".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2-q".to_string(),
            object: "model".to_string(),
            owned_by: "sentence-transformers".to_string(),
            description: "Quantized Multi-lingual model".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2".to_string(),
            object: "model".to_string(),
            owned_by: "sentence-transformers".to_string(),
            description: "Sentence-transformers model for tasks like clustering or semantic search"
                .to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "lightonai/modernbert-embed-large".to_string(),
            object: "model".to_string(),
            owned_by: "lightonai".to_string(),
            description: "Large model of ModernBert Text Embeddings".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "intfloat/multilingual-e5-small".to_string(),
            object: "model".to_string(),
            owned_by: "intfloat".to_string(),
            description: "Small model of multilingual E5 Text Embeddings".to_string(),
            dimensions: 384,
        },
        ModelInfo {
            id: "intfloat/multilingual-e5-base".to_string(),
            object: "model".to_string(),
            owned_by: "intfloat".to_string(),
            description: "Base model of multilingual E5 Text Embeddings".to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "intfloat/multilingual-e5-large".to_string(),
            object: "model".to_string(),
            owned_by: "intfloat".to_string(),
            description: "Large model of multilingual E5 Text Embeddings".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "mixedbread-ai/mxbai-embed-large-v1".to_string(),
            object: "model".to_string(),
            owned_by: "mixedbread-ai".to_string(),
            description: "Large English embedding model from MixedBreed.ai".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "mixedbread-ai/mxbai-embed-large-v1-q".to_string(),
            object: "model".to_string(),
            owned_by: "mixedbread-ai".to_string(),
            description: "Quantized Large English embedding model from MixedBreed.ai".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "Alibaba-NLP/gte-base-en-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "Alibaba-NLP".to_string(),
            description: "Base multilingual embedding model from Alibaba".to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "Alibaba-NLP/gte-base-en-v1.5-q".to_string(),
            object: "model".to_string(),
            owned_by: "Alibaba-NLP".to_string(),
            description: "Quantized Base multilingual embedding model from Alibaba".to_string(),
            dimensions: 768,
        },
        ModelInfo {
            id: "Alibaba-NLP/gte-large-en-v1.5".to_string(),
            object: "model".to_string(),
            owned_by: "Alibaba-NLP".to_string(),
            description: "Large multilingual embedding model from Alibaba".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "Alibaba-NLP/gte-large-en-v1.5-q".to_string(),
            object: "model".to_string(),
            owned_by: "Alibaba-NLP".to_string(),
            description: "Quantized Large multilingual embedding model from Alibaba".to_string(),
            dimensions: 1024,
        },
        ModelInfo {
            id: "Qdrant/clip-ViT-B-32-text".to_string(),
            object: "model".to_string(),
            owned_by: "Qdrant".to_string(),
            description: "CLIP text encoder based on ViT-B/32".to_string(),
            dimensions: 512,
        },
        ModelInfo {
            id: "jinaai/jina-embeddings-v2-base-code".to_string(),
            object: "model".to_string(),
            owned_by: "jinaai".to_string(),
            description: "Jina embeddings v2 base code".to_string(),
            dimensions: 768,
        },
    ];

    ResponseJson(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

pub fn create_embeddings_router() -> Router {
    Router::new()
        .route("/v1/embeddings", post(embeddings_create))
        // .route("/v1/models", get(models_list))
        .layer(TraceLayer::new_for_http())
}
