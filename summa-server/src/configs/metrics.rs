use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    /// Summa Metrics endpoint in format: `127.0.0.1:8084`
    pub endpoint: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            endpoint: "127.0.0.1:8084".to_owned(),
        }
    }
}
