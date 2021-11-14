use serde::{Deserialize, Serialize};

use crate::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct GrpcConfig {
    pub endpoint: String,
    pub max_frame_size_bytes: Option<u32>,
    pub workers: usize,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        GrpcConfig {
            endpoint: "127.0.0.1:8082".to_owned(),
            max_frame_size_bytes: Some(40 * 1024 * 1024),
            workers: 1,
        }
    }
}
