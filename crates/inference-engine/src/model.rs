use candle_core::Tensor;
use candle_transformers::models::gemma::{Config as Config1, Model as Model1};
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Which {
    #[value(name = "2b")]
    Base2B,
    #[value(name = "7b")]
    Base7B,
    #[value(name = "2b-it")]
    Instruct2B,
    #[value(name = "7b-it")]
    Instruct7B,
    #[value(name = "1.1-2b-it")]
    InstructV1_1_2B,
    #[value(name = "1.1-7b-it")]
    InstructV1_1_7B,
    #[value(name = "code-2b")]
    CodeBase2B,
    #[value(name = "code-7b")]
    CodeBase7B,
    #[value(name = "code-2b-it")]
    CodeInstruct2B,
    #[value(name = "code-7b-it")]
    CodeInstruct7B,
    #[value(name = "2-2b")]
    BaseV2_2B,
    #[value(name = "2-2b-it")]
    InstructV2_2B,
    #[value(name = "2-9b")]
    BaseV2_9B,
    #[value(name = "2-9b-it")]
    InstructV2_9B,
    #[value(name = "3-1b")]
    BaseV3_1B,
    #[value(name = "3-1b-it")]
    InstructV3_1B,
}

pub enum Model {
    V1(Model1),
    V2(Model2),
    V3(Model3),
}

impl Model {
    pub fn forward(&mut self, input_ids: &candle_core::Tensor, pos: usize) -> candle_core::Result<candle_core::Tensor> {
        match self {
            Self::V1(m) => m.forward(input_ids, pos),
            Self::V2(m) => m.forward(input_ids, pos),
            Self::V3(m) => m.forward(input_ids, pos),
        }
    }
}

impl Which {
    pub fn to_model_id(&self) -> String {
        match self {
            Self::InstructV1_1_2B => "google/gemma-1.1-2b-it".to_string(),
            Self::InstructV1_1_7B => "google/gemma-1.1-7b-it".to_string(),
            Self::Base2B => "google/gemma-2b".to_string(),
            Self::Base7B => "google/gemma-7b".to_string(),
            Self::Instruct2B => "google/gemma-2b-it".to_string(),
            Self::Instruct7B => "google/gemma-7b-it".to_string(),
            Self::CodeBase2B => "google/codegemma-2b".to_string(),
            Self::CodeBase7B => "google/codegemma-7b".to_string(),
            Self::CodeInstruct2B => "google/codegemma-2b-it".to_string(),
            Self::CodeInstruct7B => "google/codegemma-7b-it".to_string(),
            Self::BaseV2_2B => "google/gemma-2-2b".to_string(),
            Self::InstructV2_2B => "google/gemma-2-2b-it".to_string(),
            Self::BaseV2_9B => "google/gemma-2-9b".to_string(),
            Self::InstructV2_9B => "google/gemma-2-9b-it".to_string(),
            Self::BaseV3_1B => "google/gemma-3-1b-pt".to_string(),
            Self::InstructV3_1B => "google/gemma-3-1b-it".to_string(),
        }
    }

    pub fn is_instruct_model(&self) -> bool {
        match self {
            Self::Base2B | Self::Base7B | Self::CodeBase2B | Self::CodeBase7B | Self::BaseV2_2B | Self::BaseV2_9B | Self::BaseV3_1B => false,
            _ => true,
        }
    }

    pub fn is_v3_model(&self) -> bool {
        matches!(self, Self::BaseV3_1B | Self::InstructV3_1B)
    }
}