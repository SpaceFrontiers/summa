use serde::{Deserialize, Serialize};

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default)]
pub struct GrpcConfig {
    pub endpoint: String,
    pub workers: usize,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        GrpcConfig {
            endpoint: "127.0.0.1:8082".to_owned(),
            workers: 1,
        }
    }
}
