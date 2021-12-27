use std::net::SocketAddr;

use iroh_rpc_types::gateway::GatewayAddr;
use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    pub http_endpoint: String,
    pub p2p_endpoint: String,
    pub dns_resolver: iroh_resolver::dns_resolver::Config,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            http_endpoint: "127.0.0.1:8080".to_string(),
            p2p_endpoint: "irpc://127.0.0.1:4400".to_string(),
            dns_resolver: iroh_resolver::dns_resolver::Config::default(),
        }
    }
}

impl Config {
    pub fn derive_iroh_gateway_config(&self) -> iroh_gateway::config::Config {
        let gateway_addr = self.p2p_endpoint.parse::<GatewayAddr>().unwrap();
        let mut config = iroh_gateway::config::Config::default();
        config.rpc_client.gateway_addr = Some(gateway_addr.clone());
        config.port = self.http_endpoint.parse::<SocketAddr>().unwrap().port();
        config.redirect_to_subdomain = true;
        config
    }
}
