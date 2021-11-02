use serde::{Deserialize, Serialize};

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default)]
pub struct IpfsConfig {
    pub api_endpoint: String,
}

impl Default for IpfsConfig {
    fn default() -> Self {
        IpfsConfig {
            api_endpoint: "127.0.0.1:8080".to_owned(),
        }
    }
}
