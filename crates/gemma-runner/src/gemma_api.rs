#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle_transformers::models::gemma::{Config as Config1, Model as Model1};
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};
use clap::ValueEnum;

// Removed gemma_cli import as it's not needed for the API
use candle_core::{utils, DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
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
    } else if utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
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

        let mut generated_tokens = 0usize;

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

        for index in 0..sample_len {
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
            generated_tokens += 1;

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
    pub model: WhichModel,
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
            model: WhichModel::InstructV2_2B,
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
        utils::with_avx(),
        utils::with_neon(),
        utils::with_simd128(),
        utils::with_f16c()
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

    let start = std::time::Instant::now();
    let api = Api::new()?;

    let model_id = cfg.model_id.unwrap_or_else(|| {
        match cfg.model {
            WhichModel::Base2B => "google/gemma-2b",
            WhichModel::Base7B => "google/gemma-7b",
            WhichModel::Instruct2B => "google/gemma-2b-it",
            WhichModel::Instruct7B => "google/gemma-7b-it",
            WhichModel::InstructV1_1_2B => "google/gemma-1.1-2b-it",
            WhichModel::InstructV1_1_7B => "google/gemma-1.1-7b-it",
            WhichModel::CodeBase2B => "google/codegemma-2b",
            WhichModel::CodeBase7B => "google/codegemma-7b",
            WhichModel::CodeInstruct2B => "google/codegemma-2b-it",
            WhichModel::CodeInstruct7B => "google/codegemma-7b-it",
            WhichModel::BaseV2_2B => "google/gemma-2-2b",
            WhichModel::InstructV2_2B => "google/gemma-2-2b-it",
            WhichModel::BaseV2_9B => "google/gemma-2-9b",
            WhichModel::InstructV2_9B => "google/gemma-2-9b-it",
            WhichModel::BaseV3_1B => "google/gemma-3-1b-pt",
            WhichModel::InstructV3_1B => "google/gemma-3-1b-it",
        }
        .to_string()
    });

    println!("Loading model: {}", &model_id);

    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, cfg.revision));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let filenames = match cfg.model {
        WhichModel::BaseV3_1B | WhichModel::InstructV3_1B => vec![repo.get("model.safetensors")?],
        _ => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    println!("Retrieved files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let model: Model = match cfg.model {
        WhichModel::Base2B
        | WhichModel::Base7B
        | WhichModel::Instruct2B
        | WhichModel::Instruct7B
        | WhichModel::InstructV1_1_2B
        | WhichModel::InstructV1_1_7B
        | WhichModel::CodeBase2B
        | WhichModel::CodeBase7B
        | WhichModel::CodeInstruct2B
        | WhichModel::CodeInstruct7B => {
            let config: Config1 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model1::new(cfg.use_flash_attn, &config, vb)?;
            Model::V1(model)
        }
        WhichModel::BaseV2_2B
        | WhichModel::InstructV2_2B
        | WhichModel::BaseV2_9B
        | WhichModel::InstructV2_9B => {
            let config: Config2 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model2::new(cfg.use_flash_attn, &config, vb)?;
            Model::V2(model)
        }
        WhichModel::BaseV3_1B | WhichModel::InstructV3_1B => {
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
        WhichModel::InstructV3_1B => {
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
