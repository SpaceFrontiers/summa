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

fn default_listening_multiaddrs() -> Vec<String> {
    vec!["/ip4/0.0.0.0/tcp/4444".to_string(), "/ip4/0.0.0.0/udp/4445/quic-v1".to_string()]
}

fn default_max_conns_out() -> u32 {
    256
}

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    /// Iroh P2P RPC endpoint in format: `127.0.0.1:4401`
    pub endpoint: String,
    /// Path to crypto keys
    #[builder(setter(custom))]
    pub key_store_path: PathBuf,
    /// Bootstrap nodes. Use `~` for defaults and [] for disabling bootstrap nodes.
    #[builder(default)]
    pub bootstrap: Option<Vec<String>>,
    /// HTTP gateways that used as fallback if P2P is not available or disabled
    #[builder(default = "default_http_gateways()")]
    #[serde(default)]
    pub http_gateways: Vec<String>,
    /// Maximum number of connected peers
    #[builder(default = "default_max_conns_out()")]
    #[serde(default = "default_max_conns_out")]
    pub max_conns_out: u32,
    /// Listening P2P address in multiaddr format, for example: /ip4/0.0.0.0/tcp/4444
    #[builder(default = "default_listening_multiaddrs()")]
    #[serde(default)]
    pub listening_multiaddrs: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            endpoint: "127.0.0.1:4401".to_string(),
            key_store_path: PathBuf::new(),
            bootstrap: None,
            http_gateways: default_http_gateways(),
            max_conns_out: 256,
            listening_multiaddrs: default_listening_multiaddrs(),
        }
    }
}

impl ConfigBuilder {
    pub fn key_store_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.key_store_path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }
}
