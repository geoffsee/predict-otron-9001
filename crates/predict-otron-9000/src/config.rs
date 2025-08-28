use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerConfig {
    #[serde(default = "default_server_host")]
    pub server_host: String,
    #[serde(default = "default_server_port")]
    pub server_port: u16,
    pub server_mode: ServerMode,
    #[serde(default)]
    pub services: Services,
}

fn default_server_host() -> String {
    "127.0.0.1".to_string()
}

fn default_server_port() -> u16 { 8080 }

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum ServerMode {
    Standalone,
    HighAvailability,
}

impl Default for ServerMode {
    fn default() -> Self {
        Self::Standalone
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Services {
    #[serde(default = "inference_service_url")]
    pub inference_url: String,
    #[serde(default = "embeddings_service_url")]
    pub embeddings_url: String,
}

impl Default for Services {
    fn default() -> Self {
        Self {
            inference_url: inference_service_url(),
            embeddings_url: embeddings_service_url(),
        }
    }
}

fn inference_service_url() -> String {
    "http://inference-service:8080".to_string()
}

fn embeddings_service_url() -> String {
    "http://embeddings-service:8080".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server_host: "127.0.0.1".to_string(),
            server_port: 8080,
            server_mode: ServerMode::Standalone,
            services: Services::default(),
        }
    }
}

impl ServerConfig {
    /// Load configuration from SERVER_CONFIG environment variable
    /// Falls back to default (Local mode) if not set or invalid
    pub fn from_env() -> Self {
        match env::var("SERVER_CONFIG") {
            Ok(config_str) => {
                match serde_json::from_str::<ServerConfig>(&config_str) {
                    Ok(config) => {
                        tracing::info!("Loaded server configuration: {:?}", config);
                        config
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse SERVER_CONFIG environment variable: {}. Using default configuration.", 
                            e
                        );
                        ServerConfig::default()
                    }
                }
            }
            Err(_) => {
                tracing::info!("SERVER_CONFIG not set, Standalone mode active");
                ServerConfig::default()
            }
        }
    }

    /// Check if the server should run in high availability mode
    pub fn is_high_availability(&self) -> bool {
        self.server_mode == ServerMode::HighAvailability
    }

    /// Get the inference service URL for proxying
    pub fn inference_url(&self) -> &str {
        &self.services.inference_url
    }

    /// Get the embeddings service URL for proxying
    pub fn embeddings_url(&self) -> &str {
        &self.services.embeddings_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.server_mode, ServerMode::Standalone);
        assert!(!config.is_high_availability());
    }

    #[test]
    fn test_high_availability_config() {
        let config_json = r#"{
            "serverMode": "HighAvailability",
            "services": {
                "inference_url": "http://inference-service:8080",
                "embeddings_url": "http://embeddings-service:8080"
            }
        }"#;

        let config: ServerConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.server_mode, ServerMode::HighAvailability);
        assert!(config.is_high_availability());
        assert_eq!(config.inference_url(), "http://inference-service:8080");
        assert_eq!(config.embeddings_url(), "http://embeddings-service:8080");
    }

    #[test]
    fn test_local_mode_config() {
        let config_json = r#"{
            "serverMode": "Local"
        }"#;

        let config: ServerConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.server_mode, ServerMode::Standalone);
        assert!(!config.is_high_availability());
        // Should use default URLs
        assert_eq!(config.inference_url(), "http://inference-service:8080");
        assert_eq!(config.embeddings_url(), "http://embeddings-service:8080");
    }

    #[test]
    fn test_custom_urls() {
        let config_json = r#"{
            "serverMode": "HighAvailability",
            "services": {
                "inference_url": "http://custom-inference:9000",
                "embeddings_url": "http://custom-embeddings:9001"
            }
        }"#;

        let config: ServerConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.inference_url(), "http://custom-inference:9000");
        assert_eq!(config.embeddings_url(), "http://custom-embeddings:9001");
    }

    #[test]
    fn test_minimal_high_availability_config() {
        let config_json = r#"{"serverMode": "HighAvailability"}"#;
        let config: ServerConfig = serde_json::from_str(config_json).unwrap();
        assert!(config.is_high_availability());
        // Should use default URLs
        assert_eq!(config.inference_url(), "http://inference-service:8080");
        assert_eq!(config.embeddings_url(), "http://embeddings-service:8080");
    }
}