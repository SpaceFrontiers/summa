use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct MetricsConfig {
    pub endpoint: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        MetricsConfig {
            endpoint: "127.0.0.1:8084".to_owned(),
        }
    }
}
