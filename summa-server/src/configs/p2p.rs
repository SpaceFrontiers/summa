use std::path::{Path, PathBuf};

use path_absolutize::Absolutize;
use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

fn default_http_gateways() -> Vec<String> {
    vec![
        "w3s.link".to_string(),
        "https://ipfs.io/ipfs/".to_string(),
        "https://cloudflare-ipfs.com/ipfs/".to_string(),
        "https://gateway.pinata.cloud/ipfs/".to_string(),
    ]
}

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    pub endpoint: String,
    #[builder(setter(custom))]
    pub key_store_path: PathBuf,
    #[builder(default)]
    pub bootstrap: Option<Vec<String>>,
    #[builder(default = "default_http_gateways()")]
    #[serde(default)]
    pub http_gateways: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            endpoint: "irpc://127.0.0.1:4401".to_string(),
            key_store_path: PathBuf::new(),
            bootstrap: None,
            http_gateways: default_http_gateways(),
        }
    }
}

impl ConfigBuilder {
    pub fn key_store_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.key_store_path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }
}
