use std::collections::HashMap;
use std::net::SocketAddr;
use std::ops::Add;
use std::str::FromStr;

use hyper::header::{HeaderName, HeaderValue};
use hyper::HeaderMap;
use iroh_rpc_types::gateway::GatewayAddr;
use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;

use crate::errors::SummaServerResult;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    pub http_endpoint: String,
    pub p2p_endpoint: String,
    pub dns_resolver: iroh_resolver::dns_resolver::Config,
    pub headers: HashMap<String, String>,
    pub public_url_base: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            http_endpoint: "127.0.0.1:8080".to_string(),
            p2p_endpoint: "irpc://127.0.0.1:4400".to_string(),
            dns_resolver: iroh_resolver::dns_resolver::Config::default(),
            headers: HashMap::default(),
            public_url_base: "http://localhost:8080/".to_string(),
        }
    }
}

impl Config {
    pub fn derive_iroh_gateway_config(
        &self,
        store_service: &crate::services::Store,
        p2p_service: Option<&crate::services::P2p>,
    ) -> SummaServerResult<iroh_gateway::config::Config> {
        let gateway_addr = self.p2p_endpoint.parse::<GatewayAddr>()?;
        let mut config = iroh_gateway::config::Config::default();
        config.rpc_client.gateway_addr = Some(gateway_addr);
        config.rpc_client.store_addr = Some(store_service.rpc_addr().clone());
        config.rpc_client.p2p_addr = p2p_service.map(|p2p_service| p2p_service.rpc_addr().clone());
        config.port = self.http_endpoint.parse::<SocketAddr>()?.port();
        config.public_url_base = match self.public_url_base.ends_with('/') {
            true => self.public_url_base.to_string(),
            false => self.public_url_base.clone().add("/"),
        };
        config.redirect_to_subdomain = true;
        config.headers = HeaderMap::from_iter(
            self.headers
                .iter()
                .map(|(header_name, header_value)| {
                    Ok((
                        HeaderName::from_str(header_name).map_err(crate::errors::ValidationError::from)?,
                        HeaderValue::from_str(header_value).map_err(crate::errors::ValidationError::from)?,
                    ))
                })
                .collect::<SummaServerResult<Vec<_>>>()?
                .into_iter(),
        );
        Ok(config)
    }
}
