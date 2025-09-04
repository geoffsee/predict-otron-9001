use candle_transformers::models::csm::{LlamaConfig, LlamaModel};
use candle_transformers::models::gemma::{Config as Config1, Model as Model1};
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};

#[derive(Clone, Debug)]
pub enum Model {
    V1(Model1),
    V2(Model2),
    V3(Model3),
    Llama(LlamaModel),
}

impl Model {
    pub fn forward(
        &mut self,
        input_ids: &candle_core::Tensor,
        pos: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        match self {
            Self::V1(m) => m.forward(input_ids, pos),
            Self::V2(m) => m.forward(input_ids, pos),
            Self::V3(m) => m.forward(input_ids, pos),
            Self::Llama(m) => m.forward(input_ids, pos),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    GemmaV1,
    GemmaV2,
    GemmaV3,
    Llama,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelMeta {
    pub id: &'static str,
    pub family: Family,
    pub instruct: bool,
}

const fn m(id: &'static str, family: Family, instruct: bool) -> ModelMeta {
    ModelMeta {
        id,
        family,
        instruct,
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Which {
    // Gemma 1.x
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

    // CodeGemma
    #[value(name = "code-2b")]
    CodeBase2B,
    #[value(name = "code-7b")]
    CodeBase7B,
    #[value(name = "code-2b-it")]
    CodeInstruct2B,
    #[value(name = "code-7b-it")]
    CodeInstruct7B,

    // Gemma 2
    #[value(name = "2-2b")]
    BaseV2_2B,
    #[value(name = "2-2b-it")]
    InstructV2_2B,
    #[value(name = "2-9b")]
    BaseV2_9B,
    #[value(name = "2-9b-it")]
    InstructV2_9B,

    // Gemma 3
    #[value(name = "3-1b")]
    BaseV3_1B,
    #[value(name = "3-1b-it")]
    InstructV3_1B,

    // Llama 3.2 (use aliases instead of duplicate variants)
    #[value(name = "llama-3.2-1b")]
    Llama32_1B,
    #[value(name = "llama-3.2-1b-it", alias = "llama-3.2-1b-instruct")]
    Llama32_1BInstruct,
    #[value(name = "llama-3.2-3b")]
    Llama32_3B,
    #[value(name = "llama-3.2-3b-it", alias = "llama-3.2-3b-instruct")]
    Llama32_3BInstruct,
}

impl Which {
    pub const fn meta(&self) -> ModelMeta {
        use Family::*;
        match self {
            // Gemma 1.x
            Self::Base2B => m("google/gemma-2b", GemmaV1, false),
            Self::Base7B => m("google/gemma-7b", GemmaV1, false),
            Self::Instruct2B => m("google/gemma-2b-it", GemmaV1, true),
            Self::Instruct7B => m("google/gemma-7b-it", GemmaV1, true),
            Self::InstructV1_1_2B => m("google/gemma-1.1-2b-it", GemmaV1, true),
            Self::InstructV1_1_7B => m("google/gemma-1.1-7b-it", GemmaV1, true),

            // CodeGemma
            Self::CodeBase2B => m("google/codegemma-2b", GemmaV1, false),
            Self::CodeBase7B => m("google/codegemma-7b", GemmaV1, false),
            Self::CodeInstruct2B => m("google/codegemma-2b-it", GemmaV1, true),
            Self::CodeInstruct7B => m("google/codegemma-7b-it", GemmaV1, true),

            // Gemma 2
            Self::BaseV2_2B => m("google/gemma-2-2b", GemmaV2, false),
            Self::InstructV2_2B => m("google/gemma-2-2b-it", GemmaV2, true),
            Self::BaseV2_9B => m("google/gemma-2-9b", GemmaV2, false),
            Self::InstructV2_9B => m("google/gemma-2-9b-it", GemmaV2, true),

            // Gemma 3
            Self::BaseV3_1B => m("google/gemma-3-1b-pt", GemmaV3, false),
            Self::InstructV3_1B => m("google/gemma-3-1b-it", GemmaV3, true),

            // Llama 3.2
            Self::Llama32_1B => m("meta-llama/Llama-3.2-1B", Llama, false),
            Self::Llama32_1BInstruct => m("meta-llama/Llama-3.2-1B-Instruct", Llama, true),
            Self::Llama32_3B => m("meta-llama/Llama-3.2-3B", Llama, false),
            Self::Llama32_3BInstruct => m("meta-llama/Llama-3.2-3B-Instruct", Llama, true),
        }
    }

    pub fn to_model_id(&self) -> String {
        self.meta().id.to_string()
    }

    pub fn is_instruct_model(&self) -> bool {
        self.meta().instruct
    }

    pub fn is_v3_model(&self) -> bool {
        matches!(self.meta().family, Family::GemmaV3)
    }

    pub fn is_llama_model(&self) -> bool {
        matches!(self.meta().family, Family::Llama)
    }
}
