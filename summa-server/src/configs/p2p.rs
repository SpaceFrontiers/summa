use std::path::{Path, PathBuf};

use path_absolutize::Absolutize;
use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    pub endpoint: String,
    #[builder(setter(custom))]
    pub key_store_path: PathBuf,
    #[builder(default)]
    #[serde(default = "Vec::new")]
    pub bootstrap: Vec<String>,
    #[builder(default)]
    #[serde(default = "Vec::new")]
    pub http_gateways: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            endpoint: "irpc://127.0.0.1:4401".to_string(),
            key_store_path: PathBuf::new(),
            bootstrap: vec![],
            http_gateways: vec!["https://ipfs.io/ipfs/".to_string(), "https://cloudflare-ipfs.com/ipfs/".to_string()],
        }
    }
}

impl ConfigBuilder {
    pub fn key_store_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.key_store_path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }
}
