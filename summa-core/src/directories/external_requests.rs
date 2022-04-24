use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::ops::Range;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use strfmt::strfmt;
use summa_proto::proto::RemoteEngineConfig;

use crate::errors::ValidationError;
use crate::Error;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Header {
    pub name: String,
    pub value: String,
}

impl Header {
    pub fn new(name: &str, value: &str) -> Header {
        Header {
            name: name.to_string(),
            value: value.to_string(),
        }
    }
}

/// Using in `NetworkDirectory` for making requests over network
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ExternalResponse {
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    pub headers: Vec<Header>,
}

#[derive(thiserror::Error, Debug)]
pub enum RequestError {
    #[error("external: {0}")]
    External(String),
    #[cfg(feature = "hyper-external-request")]
    #[error("http_error: {0}")]
    HttpError(#[from] hyper::http::Error),
    #[cfg(feature = "hyper-external-request")]
    #[error("hyper_error: {0}")]
    HyperError(#[from] hyper::Error),
    #[cfg(feature = "hyper-external-request")]
    #[error("invalid_method: {0}")]
    InvalidMethod(#[from] hyper::http::method::InvalidMethod),
    #[error("io_error: {0} {1}")]
    IoError(io::Error, PathBuf),
    #[error("not_found: {0}")]
    NotFound(PathBuf),
}

#[async_trait]
pub trait ExternalRequest: Debug + Send + Sync {
    fn new(method: &str, url: &str, headers: &[Header]) -> Self
    where
        Self: Sized;
    fn request(self) -> Result<ExternalResponse, RequestError>;
    async fn request_async(self) -> Result<ExternalResponse, RequestError>;
    fn url(&self) -> &str;
}

pub trait ExternalRequestGenerator<TExternalRequest: ExternalRequest>: ExternalRequestGeneratorClone<TExternalRequest> + Debug + Send + Sync {
    fn new(network_config: RemoteEngineConfig) -> Self
    where
        Self: Sized;
    fn generate_range_request(&self, file_name: &str, range: Option<Range<u64>>) -> TExternalRequest;
    fn generate_length_request(&self, file_name: &str) -> TExternalRequest;
}

pub trait ExternalRequestGeneratorClone<TExternalRequest: ExternalRequest> {
    fn box_clone(&self) -> Box<dyn ExternalRequestGenerator<TExternalRequest>>;
}

#[derive(Clone, Debug)]
pub struct DefaultExternalRequestGenerator<TExternalRequest: ExternalRequest + Clone> {
    remote_engine_config: RemoteEngineConfig,
    _pd: PhantomData<TExternalRequest>,
}

impl<TExternalRequest: ExternalRequest + Clone + 'static> ExternalRequestGeneratorClone<TExternalRequest>
    for DefaultExternalRequestGenerator<TExternalRequest>
{
    fn box_clone(&self) -> Box<dyn ExternalRequestGenerator<TExternalRequest>> {
        Box::new((*self).clone())
    }
}

impl<TExternalRequest: ExternalRequest + Clone + 'static> ExternalRequestGenerator<TExternalRequest> for DefaultExternalRequestGenerator<TExternalRequest> {
    fn new(remote_engine_config: RemoteEngineConfig) -> DefaultExternalRequestGenerator<TExternalRequest> {
        DefaultExternalRequestGenerator {
            remote_engine_config,
            _pd: PhantomData,
        }
    }

    fn generate_range_request(&self, file_name: &str, range: Option<Range<u64>>) -> TExternalRequest {
        let mut vars = HashMap::new();
        vars.insert("file_name".to_string(), file_name.to_string());
        if let Some(range) = &range {
            let start = range.start.to_string();
            let end = (range.end - 1).to_string();
            vars.insert("start".to_string(), start);
            vars.insert("end".to_string(), end);
        } else {
            vars.insert("start".to_string(), "0".to_string());
            vars.insert("end".to_string(), "".to_string());
        }

        let mut headers = Vec::with_capacity(self.remote_engine_config.headers_template.len());
        for (header_name, header_value) in self.remote_engine_config.headers_template.iter() {
            // ToDo: temporary fix
            if range.is_none() && header_name == "range" {
                continue;
            }
            headers.push(Header {
                name: header_name.clone(),
                value: strfmt(header_value, &vars)
                    .map_err(|e| Error::Validation(Box::new(ValidationError::from(e))))
                    .expect("broken fmt"),
            });
        }
        TExternalRequest::new(
            &self.remote_engine_config.method,
            &strfmt(&self.remote_engine_config.url_template, &vars)
                .map_err(|e| Error::Validation(Box::new(ValidationError::from(e))))
                .expect("broken fmt"),
            &headers,
        )
    }

    fn generate_length_request(&self, file_name: &str) -> TExternalRequest {
        let mut vars = HashMap::new();
        vars.insert("file_name".to_string(), file_name);
        TExternalRequest::new(
            "HEAD",
            &strfmt(&self.remote_engine_config.url_template, &vars)
                .map_err(|e| Error::Validation(Box::new(ValidationError::from(e))))
                .expect("broken fmt"),
            &[],
        )
    }
}
