use inference_engine::model::{Model, Which};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_which_to_model_id() {
        // Test a few representative model variants
        assert_eq!(Which::Base2B.to_model_id(), "google/gemma-2b");
        assert_eq!(Which::Instruct7B.to_model_id(), "google/gemma-7b-it");
        assert_eq!(Which::InstructV1_1_2B.to_model_id(), "google/gemma-1.1-2b-it");
        assert_eq!(Which::CodeBase2B.to_model_id(), "google/codegemma-2b");
        assert_eq!(Which::BaseV2_2B.to_model_id(), "google/gemma-2-2b");
        assert_eq!(Which::InstructV3_1B.to_model_id(), "google/gemma-3-1b-it");
    }

    #[test]
    fn test_which_is_instruct_model() {
        // Test base models (should return false)
        assert!(!Which::Base2B.is_instruct_model());
        assert!(!Which::Base7B.is_instruct_model());
        assert!(!Which::CodeBase2B.is_instruct_model());
        assert!(!Which::CodeBase7B.is_instruct_model());
        assert!(!Which::BaseV2_2B.is_instruct_model());
        assert!(!Which::BaseV2_9B.is_instruct_model());
        assert!(!Which::BaseV3_1B.is_instruct_model());

        // Test instruct models (should return true)
        assert!(Which::Instruct2B.is_instruct_model());
        assert!(Which::Instruct7B.is_instruct_model());
        assert!(Which::InstructV1_1_2B.is_instruct_model());
        assert!(Which::InstructV1_1_7B.is_instruct_model());
        assert!(Which::CodeInstruct2B.is_instruct_model());
        assert!(Which::CodeInstruct7B.is_instruct_model());
        assert!(Which::InstructV2_2B.is_instruct_model());
        assert!(Which::InstructV2_9B.is_instruct_model());
        assert!(Which::InstructV3_1B.is_instruct_model());
    }

    #[test]
    fn test_which_is_v3_model() {
        // Test non-v3 models (should return false)
        assert!(!Which::Base2B.is_v3_model());
        assert!(!Which::Base7B.is_v3_model());
        assert!(!Which::Instruct2B.is_v3_model());
        assert!(!Which::Instruct7B.is_v3_model());
        assert!(!Which::InstructV1_1_2B.is_v3_model());
        assert!(!Which::InstructV1_1_7B.is_v3_model());
        assert!(!Which::CodeBase2B.is_v3_model());
        assert!(!Which::CodeBase7B.is_v3_model());
        assert!(!Which::CodeInstruct2B.is_v3_model());
        assert!(!Which::CodeInstruct7B.is_v3_model());
        assert!(!Which::BaseV2_2B.is_v3_model());
        assert!(!Which::InstructV2_2B.is_v3_model());
        assert!(!Which::BaseV2_9B.is_v3_model());
        assert!(!Which::InstructV2_9B.is_v3_model());

        // Test v3 models (should return true)
        assert!(Which::BaseV3_1B.is_v3_model());
        assert!(Which::InstructV3_1B.is_v3_model());
    }

    // Note: Testing the Model enum's forward method would require creating actual model instances,
    // which is complex and would require loading model weights. This is better suited for
    // integration tests or mocking the models.
}