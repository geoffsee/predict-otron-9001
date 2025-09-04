use anyhow::{Error as E, Result};
use candle_transformers::models::gemma::{Config as Config1, Model as Model1};
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};

// Removed gemma_cli import as it's not needed for the API
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;

use std::fmt;
use std::str::FromStr;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use tokenizers::Tokenizer;
use utils::hub_load_safetensors;
use utils::token_output_stream::TokenOutputStream;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WhichModel {
    #[value(name = "gemma-2b")]
    Base2B,
    #[value(name = "gemma-7b")]
    Base7B,
    #[value(name = "gemma-2b-it")]
    Instruct2B,
    #[value(name = "gemma-7b-it")]
    Instruct7B,
    #[value(name = "gemma-1.1-2b-it")]
    InstructV1_1_2B,
    #[value(name = "gemma-1.1-7b-it")]
    InstructV1_1_7B,
    #[value(name = "codegemma-2b")]
    CodeBase2B,
    #[value(name = "codegemma-7b")]
    CodeBase7B,
    #[value(name = "codegemma-2b-it")]
    CodeInstruct2B,
    #[value(name = "codegemma-7b-it")]
    CodeInstruct7B,
    #[value(name = "gemma-2-2b")]
    BaseV2_2B,
    #[value(name = "gemma-2-2b-it")]
    InstructV2_2B,
    #[value(name = "gemma-2-9b")]
    BaseV2_9B,
    #[value(name = "gemma-2-9b-it")]
    InstructV2_9B,
    #[value(name = "gemma-3-1b")]
    BaseV3_1B,
    #[value(name = "gemma-3-1b-it")]
    InstructV3_1B,
}

impl FromStr for WhichModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gemma-2b" => Ok(Self::Base2B),
            "gemma-7b" => Ok(Self::Base7B),
            "gemma-2b-it" => Ok(Self::Instruct2B),
            "gemma-7b-it" => Ok(Self::Instruct7B),
            "gemma-1.1-2b-it" => Ok(Self::InstructV1_1_2B),
            "gemma-1.1-7b-it" => Ok(Self::InstructV1_1_7B),
            "codegemma-2b" => Ok(Self::CodeBase2B),
            "codegemma-7b" => Ok(Self::CodeBase7B),
            "codegemma-2b-it" => Ok(Self::CodeInstruct2B),
            "codegemma-7b-it" => Ok(Self::CodeInstruct7B),
            "gemma-2-2b" => Ok(Self::BaseV2_2B),
            "gemma-2-2b-it" => Ok(Self::InstructV2_2B),
            "gemma-2-9b" => Ok(Self::BaseV2_9B),
            "gemma-2-9b-it" => Ok(Self::InstructV2_9B),
            "gemma-3-1b" => Ok(Self::BaseV3_1B),
            "gemma-3-1b-it" => Ok(Self::InstructV3_1B),
            _ => Err(format!("Unknown model: {}", s)),
        }
    }
}

impl fmt::Display for WhichModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Base2B => "gemma-2b",
            Self::Base7B => "gemma-7b",
            Self::Instruct2B => "gemma-2b-it",
            Self::Instruct7B => "gemma-7b-it",
            Self::InstructV1_1_2B => "gemma-1.1-2b-it",
            Self::InstructV1_1_7B => "gemma-1.1-7b-it",
            Self::CodeBase2B => "codegemma-2b",
            Self::CodeBase7B => "codegemma-7b",
            Self::CodeInstruct2B => "codegemma-2b-it",
            Self::CodeInstruct7B => "codegemma-7b-it",
            Self::BaseV2_2B => "gemma-2-2b",
            Self::InstructV2_2B => "gemma-2-2b-it",
            Self::BaseV2_9B => "gemma-2-9b",
            Self::InstructV2_9B => "gemma-2-9b-it",
            Self::BaseV3_1B => "gemma-3-1b",
            Self::InstructV3_1B => "gemma-3-1b-it",
        };
        write!(f, "{}", name)
    }
}

enum Model {
    V1(Model1),
    V2(Model2),
    V3(Model3),
}

impl Model {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::V1(m) => m.forward(input_ids, pos),
            Self::V2(m) => m.forward(input_ids, pos),
            Self::V3(m) => m.forward(input_ids, pos),
        }
    }
}

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: tokenizers::Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    /// Stream-only generation: sends freshly generated token strings over `tx`.
    /// (Does not send the prompt tokens; only newly generated model tokens.)
    fn run_stream(
        &mut self,
        prompt: &str,
        sample_len: usize,
        tx: Sender<Result<String>>,
    ) -> Result<()> {
        self.tokenizer.clear();

        // Encode prompt (context only; do not emit prompt tokens to the stream).
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // Warm the tokenizer's internal state with prompt tokens (so merges are correct),
        // but do not send them to the receiver.
        for &t in tokens.iter() {
            let _ = self.tokenizer.next_token(t)?;
        }
        // Make sure stdout isn't holding anything (if caller also prints).
        std::io::stdout().flush()?;

        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };
        let eot_token = match self.tokenizer.get_token("<end_of_turn>") {
            Some(token) => token,
            None => {
                eprintln!("Warning: <end_of_turn> token not found, using <eos> as backup");
                eos_token
            }
        };

        let start_gen = std::time::Instant::now();

        for (_generated_tokens, index) in (0..sample_len).enumerate() {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);

            if next_token == eos_token || next_token == eot_token {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                // Best-effort send; ignore if receiver dropped.
                let _ = tx.send(Ok(t));
            }
        }

        let _dt = start_gen.elapsed();

        // Flush any remaining buffered bytes as one final chunk.
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            let _ = tx.send(Ok(rest));
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GemmaInferenceConfig {
    pub tracing: bool,
    pub prompt: String,
    pub model: Option<WhichModel>,
    pub cpu: bool,
    pub dtype: Option<String>,
    pub model_id: Option<String>,
    pub revision: String,
    pub use_flash_attn: bool,
    pub seed: u64,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub max_tokens: usize,
}

impl Default for GemmaInferenceConfig {
    fn default() -> Self {
        Self {
            tracing: false,
            prompt: "Hello".to_string(),
            model: Some(WhichModel::InstructV2_2B),
            cpu: false,
            dtype: None,
            model_id: None,
            revision: "main".to_string(),
            use_flash_attn: false,
            seed: 299792458,
            temperature: 0.8,
            top_p: None,
            repeat_penalty: 1.1,
            repeat_last_n: 128,
            max_tokens: 100,
        }
    }
}

// Removed From<Args> implementation as Args is not available and not needed for API usage

/// Builds the model and returns a channel that streams generated token strings.
/// If model setup fails, the `Result` is returned immediately.
pub fn run_gemma_api(cfg: GemmaInferenceConfig) -> Result<Receiver<Result<String>>> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let _guard = if cfg.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let device = device(cfg.cpu)?;
    println!("Device: {:?}", device);

    let dtype = match cfg.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => anyhow::bail!("Unsupported dtype {dtype}"),
        None => {
            if device.is_cuda() {
                DType::BF16
            } else {
                DType::F16
            }
        }
    };
    println!("Using dtype: {:?}", dtype);
    println!("Raw model string: {:?}", cfg.model_id);

    let start = std::time::Instant::now();
    let api = Api::new()?;

    let model_id = cfg.model_id.unwrap_or_else(|| {
        match cfg.model {
            Some(WhichModel::Base2B) => "google/gemma-2b",
            Some(WhichModel::Base7B) => "google/gemma-7b",
            Some(WhichModel::Instruct2B) => "google/gemma-2b-it",
            Some(WhichModel::Instruct7B) => "google/gemma-7b-it",
            Some(WhichModel::InstructV1_1_2B) => "google/gemma-1.1-2b-it",
            Some(WhichModel::InstructV1_1_7B) => "google/gemma-1.1-7b-it",
            Some(WhichModel::CodeBase2B) => "google/codegemma-2b",
            Some(WhichModel::CodeBase7B) => "google/codegemma-7b",
            Some(WhichModel::CodeInstruct2B) => "google/codegemma-2b-it",
            Some(WhichModel::CodeInstruct7B) => "google/codegemma-7b-it",
            Some(WhichModel::BaseV2_2B) => "google/gemma-2-2b",
            Some(WhichModel::InstructV2_2B) => "google/gemma-2-2b-it",
            Some(WhichModel::BaseV2_9B) => "google/gemma-2-9b",
            Some(WhichModel::InstructV2_9B) => "google/gemma-2-9b-it",
            Some(WhichModel::BaseV3_1B) => "google/gemma-3-1b-pt",
            Some(WhichModel::InstructV3_1B) => "google/gemma-3-1b-it",
            None => "google/gemma-2-2b-it", // default fallback
        }
        .to_string()
    });

    println!("Loading model: {}", &model_id);

    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, cfg.revision));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let filenames = match cfg.model {
        Some(WhichModel::BaseV3_1B) | Some(WhichModel::InstructV3_1B) => {
            vec![repo.get("model.safetensors")?]
        }
        _ => hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    println!("Retrieved files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let model: Model = match cfg.model {
        Some(WhichModel::Base2B)
        | Some(WhichModel::Base7B)
        | Some(WhichModel::Instruct2B)
        | Some(WhichModel::Instruct7B)
        | Some(WhichModel::InstructV1_1_2B)
        | Some(WhichModel::InstructV1_1_7B)
        | Some(WhichModel::CodeBase2B)
        | Some(WhichModel::CodeBase7B)
        | Some(WhichModel::CodeInstruct2B)
        | Some(WhichModel::CodeInstruct7B) => {
            let config: Config1 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model1::new(cfg.use_flash_attn, &config, vb)?;
            Model::V1(model)
        }
        Some(WhichModel::BaseV2_2B)
        | Some(WhichModel::InstructV2_2B)
        | Some(WhichModel::BaseV2_9B)
        | Some(WhichModel::InstructV2_9B)
        | None => {
            // default to V2 model
            let config: Config2 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model2::new(cfg.use_flash_attn, &config, vb)?;
            Model::V2(model)
        }
        Some(WhichModel::BaseV3_1B) | Some(WhichModel::InstructV3_1B) => {
            let config: Config3 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model3::new(cfg.use_flash_attn, &config, vb)?;
            Model::V3(model)
        }
    };
    println!("Loaded model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        cfg.seed,
        cfg.temperature.into(),
        cfg.top_p,
        cfg.repeat_penalty,
        cfg.repeat_last_n,
        &device,
    );

    let prompt = match cfg.model {
        Some(WhichModel::InstructV3_1B) => {
            format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                cfg.prompt
            )
        }
        _ => cfg.prompt,
    };

    println!("Starting inference...");

    // Create the channel after successful setup.
    let (tx, rx) = mpsc::channel::<Result<String>>();

    // Spawn generation thread; send tokens to the channel.
    thread::spawn(move || {
        // If generation fails, forward the error once.
        if let Err(e) = pipeline.run_stream(&prompt, cfg.max_tokens, tx.clone()) {
            let _ = tx.send(Err(e));
        }
        // Channel closes when tx is dropped.
    });

    Ok(rx)
}
