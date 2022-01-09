use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    /// Summa API GRPC address in format: `127.0.0.1:8082`
    pub grpc_endpoint: String,
    /// Summa API GRPC-web address in format: `127.0.0.1:8081`
    pub http_endpoint: Option<String>,
    /// Maximum frame size in bytes for HTTP2 transport
    pub max_frame_size_bytes: Option<u32>,
    /// Maximum number of in-flight requests
    pub concurrency_limit: usize,
    /// Maximum number of buffered requests
    pub buffer: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            grpc_endpoint: "127.0.0.1:8082".to_string(),
            http_endpoint: None,
            max_frame_size_bytes: Some(40 * 1024 * 1024),
            concurrency_limit: 10,
            buffer: 100,
        }
    }
}
