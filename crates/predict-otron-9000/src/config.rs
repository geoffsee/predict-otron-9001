use serde::{Deserialize, Serialize};
use std::env;
use tracing::info;
use tracing::log::error;
/// # Generating `SERVER_CONFIG` with Node
// # const server_config = {serverMode: "HighAvailability", services: {inference_url: "http://custom-inference:9000", embeddings_url: "http://custom-embeddings:9001"} };
// # console.log(JSON.stringify(server_config).replace(/"/g, '\\"'));
///
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ServerConfig {
    #[serde(default = "default_server_host")]
    pub server_host: String,
    #[serde(default = "default_server_port")]
    pub server_port: u16,
    pub server_mode: ServerMode,
    #[serde(default)]
    pub services: Option<Services>,
}

fn default_server_host() -> String {
    "127.0.0.1".to_string()
}

fn default_server_port() -> u16 {
    8080
}

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

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Services {
    pub inference_url: Option<String>,
    pub embeddings_url: Option<String>,
}



impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server_host: "127.0.0.1".to_string(),
            server_port: 8080,
            server_mode: ServerMode::Standalone,
            services: Some(Services::default()),
        }
    }
}

impl ServerConfig {
    /// Load configuration from SERVER_CONFIG environment variable
    /// Falls back to default (Local mode) if not set or invalid
    pub fn from_env() -> Self {
        match env::var("SERVER_CONFIG") {
            Ok(config_str) => match serde_json::from_str::<ServerConfig>(&config_str) {
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
            },
            Err(_) => {
                tracing::info!("SERVER_CONFIG not set, Standalone mode active");
                ServerConfig::default()
            }
        }
    }

    /// Check if the server should run in high availability mode
    pub fn is_high_availability(&self) -> Result<bool, std::io::Error> {
        if self.server_mode == ServerMode::HighAvailability {
            let services_well_defined: bool = self.clone().services.is_some();

            let inference_url_well_defined: bool =
                services_well_defined && self.clone().services.unwrap().inference_url.is_some();

            let embeddings_well_defined: bool =
                services_well_defined && self.clone().services.unwrap().embeddings_url.is_some();

            let is_well_defined_for_ha =
                services_well_defined && inference_url_well_defined && embeddings_well_defined;

            if !is_well_defined_for_ha {
                let config_string = serde_json::to_string_pretty(&self).unwrap();
                error!(
                    "HighAvailability mode configured but services not well defined! \n## Config Used:\n {}",
                    config_string
                );
                let err = std::io::Error::other(
                    "HighAvailability mode configured but services not well defined!",
                );
                return Err(err);
            }
        }

        Ok(self.server_mode == ServerMode::HighAvailability)
    }

    /// Get the inference service URL for proxying
    pub fn inference_url(&self) -> Option<String> {
        if self.services.is_some() {
            self.services.clone()?.inference_url
        } else {
            None
        }
    }

    /// Get the embeddings service URL for proxying
    pub fn embeddings_url(&self) -> Option<String> {
        if self.services.is_some() {
            self.services.clone()?.embeddings_url
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.server_mode, ServerMode::Standalone);
        assert!(!config.is_high_availability().unwrap());
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
        assert!(config.is_high_availability().unwrap());
        assert_eq!(
            config.inference_url().unwrap(),
            "http://inference-service:8080"
        );
        assert_eq!(
            config.embeddings_url().unwrap(),
            "http://embeddings-service:8080"
        );
    }

    #[test]
    fn test_local_mode_config() {
        let config_json = r#"{
            "serverMode": "Standalone"
        }"#;

        let config: ServerConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.server_mode, ServerMode::Standalone);
        assert!(!config.is_high_availability().unwrap());
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
        assert_eq!(
            config.inference_url().unwrap(),
            "http://custom-inference:9000"
        );
        assert_eq!(
            config.embeddings_url().unwrap(),
            "http://custom-embeddings:9001"
        );
    }

    #[test]
    fn test_minimal_high_availability_config_error() {
        let config_json = r#"{"serverMode": "HighAvailability"}"#;
        let config: ServerConfig = serde_json::from_str(config_json).unwrap();

        let is_high_availability = config.is_high_availability();

        assert!(is_high_availability.is_err());
        // // Should use default URLs
        // assert_eq!(config.inference_url().unwrap(), "http://inference-service:8080");
        // assert_eq!(config.embeddings_url().unwrap(), "http://embeddings-service:8080");
    }
}
