use std::collections::HashMap;
use std::net::SocketAddr;
use std::ops::Add;
use std::str::FromStr;

use headers::{
    AcceptRanges, AccessControlAllowHeaders, AccessControlAllowMethods, AccessControlAllowOrigin, AccessControlExposeHeaders, HeaderMapExt, HeaderName,
    HeaderValue,
};
use hyper::header::{ACCEPT, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, IF_NONE_MATCH, RANGE, USER_AGENT};
use hyper::{HeaderMap, Method};
use iroh_gateway::constants::{
    HEADER_SERVICE_WORKER, HEADER_X_CHUNKED_OUTPUT, HEADER_X_IPFS_PATH, HEADER_X_IPFS_ROOTS, HEADER_X_REQUESTED_WITH, HEADER_X_STREAM_OUTPUT,
};
use iroh_rpc_types::gateway::GatewayAddr;
use serde::{Deserialize, Serialize};
use summa_core::errors::BuilderError;
use summa_core::utils::parse_endpoint;

use crate::errors::SummaServerResult;

fn default_headers() -> HashMap<String, String> {
    let mut headers = HeaderMap::new();
    headers.typed_insert(AccessControlAllowOrigin::ANY);
    headers.typed_insert(AcceptRanges::bytes());
    headers.typed_insert(
        [Method::GET, Method::PUT, Method::POST, Method::DELETE, Method::HEAD, Method::OPTIONS]
            .into_iter()
            .collect::<AccessControlAllowMethods>(),
    );
    headers.typed_insert(
        [
            ACCEPT,
            CACHE_CONTROL,
            CONTENT_TYPE,
            CONTENT_LENGTH,
            CONTENT_RANGE,
            HEADER_SERVICE_WORKER.clone(),
            HEADER_X_REQUESTED_WITH.clone(),
            IF_NONE_MATCH,
            RANGE,
            USER_AGENT,
        ]
        .into_iter()
        .collect::<AccessControlAllowHeaders>(),
    );
    headers.typed_insert(
        [
            CONTENT_TYPE,
            CONTENT_LENGTH,
            CONTENT_RANGE,
            HEADER_X_IPFS_PATH.clone(),
            HEADER_X_IPFS_ROOTS.clone(),
            HEADER_X_CHUNKED_OUTPUT.clone(),
            HEADER_X_STREAM_OUTPUT.clone(),
        ]
        .into_iter()
        .collect::<AccessControlExposeHeaders>(),
    );
    headers
        .iter()
        .map(|(header_name, header_value)| (header_name.to_string(), header_value.to_str().expect("default headers seems wrong").to_string()))
        .collect()
}

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    /// Iroh Gateway HTTP endpoint in format: `127.0.0.1:8080`
    pub http_endpoint: String,
    /// Iroh Gateway RPC endpoint in format: `127.0.0.1:4400`
    pub p2p_endpoint: String,
    /// TLD resolvers for specific domains
    pub dns_resolver: iroh_resolver::dns_resolver::Config,
    /// Headers that will be added to each HTTP response
    #[builder(default = "default_headers()")]
    pub headers: HashMap<String, String>,
    /// Public URL base
    pub public_url_base: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            http_endpoint: "127.0.0.1:8080".to_string(),
            p2p_endpoint: "127.0.0.1:4400".to_string(),
            dns_resolver: iroh_resolver::dns_resolver::Config::default(),
            headers: default_headers(),
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
        let gateway_addr: GatewayAddr = parse_endpoint(&self.p2p_endpoint)?;
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
