use std::path::PathBuf;

use hyper::client::HttpConnector;
use hyper::{Client, Method, Request};
use summa_core::directories::{ExternalRequest, ExternalResponse, Header, RequestError};
use tonic::async_trait;
use tracing::info;

use crate::errors::Error;

#[derive(Clone, Debug)]
pub struct HyperExternalRequest {
    pub client: Client<HttpConnector>,
    pub method: String,
    pub url: String,
    pub headers: Vec<Header>,
}

#[async_trait]
impl ExternalRequest for HyperExternalRequest {
    fn new(method: &str, url: &str, headers: &[Header]) -> Self
    where
        Self: Sized,
    {
        HyperExternalRequest {
            client: Client::new(),
            method: method.to_string(),
            url: url.to_string(),
            headers: Vec::from_iter(headers.iter().cloned()),
        }
    }

    fn request(self) -> Result<ExternalResponse, RequestError> {
        let (s, r) = tokio::sync::oneshot::channel();
        tokio::spawn(async move { s.send(self.request_async().await) });
        r.blocking_recv().expect("channel failed")
    }

    async fn request_async(self) -> Result<ExternalResponse, RequestError> {
        let mut request = Request::builder().uri(&self.url).method(Method::from_bytes(self.method.as_bytes()).unwrap());
        for header in self.headers.iter() {
            request = request.header(&header.name, &header.value);
        }
        info!(action = "network_request", request = ?request);
        let response = self.client.request(request.body(hyper::Body::empty()).unwrap()).await.unwrap();
        if response.status() == 404 {
            return Err(RequestError::NotFound(PathBuf::from(self.url)));
        }
        let headers = response
            .headers()
            .iter()
            .map(|header_value| Header {
                name: header_value.0.to_string(),
                value: header_value.1.to_str().expect("wrong header value").to_string(),
            })
            .collect();
        Ok(ExternalResponse {
            data: hyper::body::to_bytes(response).await.map_err(Error::from).unwrap().to_vec(),
            headers,
        })
    }

    fn url(&self) -> &str {
        &self.url
    }
}
