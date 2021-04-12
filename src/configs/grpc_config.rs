use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrpcConfig {
    pub endpoint: String,
    pub timeout_seconds: u64,
    pub workers: usize,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        GrpcConfig {
            endpoint: "127.0.0.1:8082".to_owned(),
            timeout_seconds: 10,
            workers: 1,
        }
    }
}
