use std::path::{Path, PathBuf};

use path_absolutize::Absolutize;
use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    /// Default chunk size used in `Store`
    #[builder(default = "64 * 1024")]
    pub default_chunk_size: u64,
    /// Iroh Store RPC endpoint in format: `127.0.0.1:4402`
    pub endpoint: String,
    /// Path to store files
    #[builder(setter(custom))]
    pub path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            default_chunk_size: 64 * 1024,
            endpoint: "127.0.0.1:4402".to_string(),
            path: PathBuf::new(),
        }
    }
}

impl ConfigBuilder {
    pub fn path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }
}
