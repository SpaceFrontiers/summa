use serde::{Deserialize, Serialize};

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default)]
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
