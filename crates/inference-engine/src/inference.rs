use anyhow::Result;
use candle_core::Tensor;

/// ModelInference trait defines the common interface for model inference operations
///
/// This trait serves as an abstraction for different model implementations (Gemma and Llama)
/// to provide a unified interface for the inference engine.
pub trait ModelInference {
    /// Perform model inference for the given input tensor starting at the specified position
    ///
    /// # Arguments
    ///
    /// * `input_ids` - The input tensor containing token IDs
    /// * `pos` - The position to start generation from
    ///
    /// # Returns
    ///
    /// A tensor containing the logits for the next token prediction
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor>;

    /// Reset the model's internal state, if applicable
    ///
    /// This method can be used to clear any cached state between inference requests
    fn reset_state(&mut self) -> Result<()>;

    /// Get the model type name
    ///
    /// Returns a string identifier for the model type (e.g., "Gemma", "Llama")
    fn model_type(&self) -> &'static str;
}

/// Factory function type for creating model inference implementations
pub type ModelInferenceFactory = fn() -> Result<Box<dyn ModelInference>>;