use std::collections::HashMap;
use std::default::Default;

use serde::{Deserialize, Serialize};
use summa_proto::proto::IndexEngineConfig;

use crate::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct ApplicationConfig {
    #[serde(default = "HashMap::new")]
    pub indices: HashMap<String, IndexEngineConfig>,
    #[builder(default = "None")]
    pub autocommit_interval_ms: Option<u64>,
    #[builder(default = "128 * 1024 * 1024")]
    pub writer_heap_size_bytes: u64,
    #[builder(default = "1")]
    pub writer_threads: u64,
}

impl Default for ApplicationConfig {
    fn default() -> Self {
        ApplicationConfig {
            indices: HashMap::new(),
            autocommit_interval_ms: None,
            writer_heap_size_bytes: 128 * 1024 * 1024,
            writer_threads: 1,
        }
    }
}
