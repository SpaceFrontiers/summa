use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    pub grpc_endpoint: String,
    pub http_endpoint: Option<String>,
    pub max_frame_size_bytes: Option<u32>,
    pub workers: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            grpc_endpoint: "127.0.0.1:8082".to_string(),
            http_endpoint: None,
            max_frame_size_bytes: Some(40 * 1024 * 1024),
            workers: 1,
        }
    }
}
