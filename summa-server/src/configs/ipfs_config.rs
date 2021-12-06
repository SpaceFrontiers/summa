use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct IpfsConfig {
    pub api_endpoint: String,
    pub default_hash: Option<String>,
    pub default_chunker: Option<String>,
}

impl Default for IpfsConfig {
    fn default() -> Self {
        IpfsConfig {
            api_endpoint: "127.0.0.1:8080".to_owned(),
            default_hash: Some("blake3".to_string()),
            default_chunker: Some("size-65536".to_string()),
        }
    }
}
