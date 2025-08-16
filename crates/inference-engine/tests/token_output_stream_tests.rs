use inference_engine::token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;
use std::path::PathBuf;
use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a simple tokenizer for testing
    fn create_test_tokenizer() -> Result<Tokenizer> {
        // Create a simple tokenizer from the pretrained model
        // This uses the tokenizer from the Hugging Face hub
        let tokenizer = Tokenizer::from_pretrained("google/gemma-2b", None).unwrap();
        Ok(tokenizer)
    }

    #[test]
    fn test_new_token_output_stream() -> Result<()> {
        let tokenizer = create_test_tokenizer()?;
        let token_stream = TokenOutputStream::new(tokenizer);
        
        // Check that the token stream was created successfully
        assert!(token_stream.tokenizer().get_vocab(true).len() > 0);
        Ok(())
    }

    #[test]
    fn test_clear() -> Result<()> {
        let tokenizer = create_test_tokenizer()?;
        let mut token_stream = TokenOutputStream::new(tokenizer);
        
        // Add a token
        let token_id = token_stream.get_token("<eos>").unwrap();
        token_stream.next_token(token_id)?;
        
        // Clear the stream
        token_stream.clear();
        
        // Check that the stream is empty by trying to decode all
        let decoded = token_stream.decode_all()?;
        assert_eq!(decoded, "");
        
        Ok(())
    }

    #[test]
    fn test_get_token() -> Result<()> {
        let tokenizer = create_test_tokenizer()?;
        let token_stream = TokenOutputStream::new(tokenizer);
        
        // Get a token that should exist
        let eos_token = token_stream.get_token("<eos>");
        assert!(eos_token.is_some());
        
        // Get a token that shouldn't exist
        let nonexistent_token = token_stream.get_token("<this_token_does_not_exist>");
        assert!(nonexistent_token.is_none());
        
        Ok(())
    }

    #[test]
    fn test_next_token_and_decode() -> Result<()> {
        let tokenizer = create_test_tokenizer()?;
        let mut token_stream = TokenOutputStream::new(tokenizer);
        
        // Get some tokens
        let hello_tokens = token_stream.tokenizer().encode("Hello world", true).unwrap();
        let token_ids = hello_tokens.get_ids();
        
        // Add tokens one by one
        let mut output = String::new();
        for &token_id in token_ids {
            if let Some(text) = token_stream.next_token(token_id)? {
                output.push_str(&text);
            }
        }
        
        // Get any remaining text
        if let Some(rest) = token_stream.decode_rest()? {
            output.push_str(&rest);
        }
        
        // Check the output
        assert!(!output.is_empty());
        assert_eq!(output.trim(), "Hello world");
        
        Ok(())
    }

    #[test]
    fn test_decode_all() -> Result<()> {
        let tokenizer = create_test_tokenizer()?;
        let mut token_stream = TokenOutputStream::new(tokenizer);
        
        // Get some tokens
        let hello_tokens = token_stream.tokenizer().encode("Hello world", true).unwrap();
        let token_ids = hello_tokens.get_ids();
        
        // Add tokens one by one
        for &token_id in token_ids {
            token_stream.next_token(token_id)?;
        }
        
        // Decode all
        let decoded = token_stream.decode_all()?;
        
        // Check the output
        assert_eq!(decoded.trim(), "Hello world");
        
        Ok(())
    }

    #[test]
    fn test_into_inner() -> Result<()> {
        let tokenizer = create_test_tokenizer()?;
        let token_stream = TokenOutputStream::new(tokenizer);
        
        // Get the inner tokenizer
        let inner_tokenizer = token_stream.into_inner();
        
        // Check that the inner tokenizer works
        let encoded = inner_tokenizer.encode("Test", true).unwrap();
        assert!(encoded.get_ids().len() > 0);
        
        Ok(())
    }
}