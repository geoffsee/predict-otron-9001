use crate::EOS_TOKEN;
use anyhow::{bail, Error as E};
use candle_core::{utils, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;
use candle_transformers::models::llama::{Llama, LlamaConfig};
use clap::ValueEnum;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use std::sync::mpsc::{self, Receiver};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum WhichModel {
    #[value(name = "llama-3.2-1b")]
    #[default]
    Llama32_1B,
    #[value(name = "llama-3.2-1b-instruct")]
    Llama32_1BInstruct,
    #[value(name = "llama-3.2-3b")]
    Llama32_3B,
    #[value(name = "llama-3.2-3b-instruct")]
    Llama32_3BInstruct,
    #[value(name = "smollm2-135m")]
    SmolLM2_135M,
    #[value(name = "smollm2-135m-instruct")]
    SmolLM2_135MInstruct,
    #[value(name = "smollm2-360m")]
    SmolLM2_360M,
    #[value(name = "smollm2-360m-instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "smollm2-1.7b")]
    SmolLM2_1_7B,
    #[value(name = "smollm2-1.7b-instruct")]
    SmolLM2_1_7BInstruct,
    #[value(name = "tinyllama-1.1b-chat")]
    TinyLlama1_1BChat,
}

#[derive(Debug, Clone)]
pub struct LlamaInferenceConfig {
    pub prompt: String,

    pub model: WhichModel,
    pub cpu: bool,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub max_tokens: usize,
    pub no_kv_cache: bool,
    pub dtype: Option<String>,
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub use_flash_attn: bool,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl LlamaInferenceConfig {
    pub fn new(model: WhichModel) -> Self {
        Self {
            prompt: String::new(),
            model,
            cpu: false,
            temperature: 1.0,
            top_p: None,
            top_k: None,
            seed: 42,
            max_tokens: 512,
            no_kv_cache: false,
            dtype: None,
            model_id: None,
            revision: None,
            use_flash_attn: true,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}
impl Default for LlamaInferenceConfig {
    fn default() -> Self {
        Self {
            // Leave prompt empty by default; let call sites set it.
            prompt: String::new(),

            // Keep your existing model choice; swap at call-site if needed.
            model: WhichModel::Llama32_1BInstruct,

            // Prefer GPU if available.
            cpu: false,

            // Sampling: balanced + stable
            temperature: 0.7,
            top_p: Some(0.95),
            top_k: Some(50),

            // Reproducible by default; override for variability.
            seed: 42,

            // Don’t run unbounded generations.
            max_tokens: 512,

            // Performance flags
            no_kv_cache: false,    // keep cache ON for speed
            use_flash_attn: false, // great speed boost if supported

            // Precision: bf16 is a good default on Ampere+; fallback to fp16 if needed.
            dtype: Some("bf16".to_string()),

            // Optional model source pinning (None = app defaults)
            model_id: None,
            revision: None,

            // Anti-repeat heuristics
            repeat_penalty: 1.15,
            repeat_last_n: 128,
        }
    }
}

fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

fn hub_load_safetensors(
    api: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let json_file = api.get(json_file)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value = serde_json::from_reader(&json_file)?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| api.get(v))
        .collect::<anyhow::Result<Vec<_>, _>>()?;
    Ok(safetensors_files)
}

pub fn run_llama_inference(
    cfg: LlamaInferenceConfig,
) -> anyhow::Result<Receiver<anyhow::Result<String>>, anyhow::Error> {
    // ---- Device & dtype -----------------------------------------------------
    let device = device(cfg.cpu)?;
    println!("Device: {:?}", device);

    let dtype = match cfg.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    println!("Using dtype: {:?}", dtype);

    // ---- Load model & tokenizer --------------------------------------------
    let (llama, tokenizer, mut cache) = {
        let api = Api::new()?;
        let model_id = cfg.model_id.clone().unwrap_or_else(|| {
            match cfg.model {
                WhichModel::Llama32_1B => "meta-llama/Llama-3.2-1B",
                WhichModel::Llama32_1BInstruct => "meta-llama/Llama-3.2-1B-Instruct",
                WhichModel::Llama32_3B => "meta-llama/Llama-3.2-3B",
                WhichModel::Llama32_3BInstruct => "meta-llama/Llama-3.2-3B-Instruct",
                WhichModel::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M",
                WhichModel::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct",
                WhichModel::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M",
                WhichModel::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
                WhichModel::SmolLM2_1_7B => "HuggingFaceTB/SmolLM2-1.7B",
                WhichModel::SmolLM2_1_7BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                WhichModel::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            }
            .to_string()
        });
        println!("Loading model: {}", model_id);
        let revision = cfg.revision.clone().unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(cfg.use_flash_attn);

        let filenames = match cfg.model {
            WhichModel::Llama32_3B | WhichModel::Llama32_3BInstruct => {
                hub_load_safetensors(&api, "model.safetensors.index.json")?
            }
            _ => vec![api.get("model.safetensors")?],
        };

        let cache = model::Cache::new(!cfg.no_kv_cache, dtype, &config, &device)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let llama = Llama::load(vb, &config)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        (llama, tokenizer, cache)
    };

    // ---- Prepare prompt & sampler ------------------------------------------
    let eos_token_id = tokenizer
        .token_to_id(EOS_TOKEN)
        .map(model::LlamaEosToks::Single);

    let mut tokens = tokenizer
        .encode(cfg.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("Starting inference...");

    let mut logits_processor = {
        let temperature = cfg.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (cfg.top_k, cfg.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(cfg.seed, sampling)
    };

    // Channel for streaming decoded fragments to the caller.
    let (tx, rx) = mpsc::channel::<anyhow::Result<String>>();

    // ---- Spawn generation thread -------------------------------------------
    std::thread::spawn(move || {
        let start_gen = std::time::Instant::now();
        let mut index_pos = 0usize;
        let mut token_generated = 0usize;

        for index in 0..cfg.max_tokens {
            // Use KV-cache for single-token step after the first pass.
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = match Tensor::new(ctxt, &device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(Err(e.into()));
                    break;
                }
            };

            let logits = match llama.forward(&input, context_index, &mut cache) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Err(e.into()));
                    break;
                }
            };
            let logits = match logits.squeeze(0) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Err(e.into()));
                    break;
                }
            };

            let logits = if cfg.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(cfg.repeat_last_n);
                match candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    cfg.repeat_penalty,
                    &tokens[start_at..],
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.send(Err(e.into()));
                        break;
                    }
                }
            };

            index_pos += ctxt.len();

            let next_token = match logits_processor.sample(&logits) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(Err(e.into()));
                    break;
                }
            };

            token_generated += 1;
            tokens.push(next_token);

            // Early stop on EOS.
            let stop = match eos_token_id {
                Some(model::LlamaEosToks::Single(eos_tok_id)) => next_token == eos_tok_id,
                Some(model::LlamaEosToks::Multiple(ref eos_ids)) => eos_ids.contains(&next_token),
                None => false,
            };
            if stop {
                break;
            }

            // Decode this token's text and stream it out.
            match tokenizer.decode(&[next_token], false) {
                Ok(text) => {
                    if !text.is_empty() {
                        // Best-effort send; if receiver is gone, just stop.
                        if tx.send(Ok(text)).is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("{}", e)));
                    break;
                }
            }
        }

        // Optional: final stats as a debug line (not sent through the stream).
        let dt = start_gen.elapsed();
        eprintln!(
            "[llama-runner] {} tokens generated ({:.2} tokens/s)",
            token_generated,
            token_generated as f64 / dt.as_secs_f64(),
        );
        // Dropping tx closes the stream.
    });

    Ok(rx)
}
