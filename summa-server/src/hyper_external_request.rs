use hyper::client::HttpConnector;
use hyper::{Client, Method, Request};
use summa_core::directories::{ExternalRequest, ExternalResponse, Header};
use summa_core::errors::{SummaResult, ValidationError};
use tonic::async_trait;

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

    fn request(&self) -> SummaResult<ExternalResponse> {
        todo!()
    }

    async fn request_async(&self) -> SummaResult<ExternalResponse> {
        let mut request = Request::builder().uri(&self.url).method(
            Method::from_bytes(self.method.as_bytes())
                .map_err(|_| summa_core::Error::Validation(ValidationError::InvalidHttpMethod(self.method.to_string())))?,
        );
        for header in self.headers.iter() {
            request = request.header(&header.name, &header.value);
        }
        let response = self
            .client
            .request(request.body(hyper::Body::empty()).map_err(Error::from)?)
            .await
            .map_err(Error::from)?;
        let headers = response
            .headers()
            .iter()
            .map(|header_value| Header {
                name: header_value.0.to_string(),
                value: header_value.1.to_str().unwrap().to_string(),
            })
            .collect();
        Ok(ExternalResponse {
            data: hyper::body::to_bytes(response).await.map_err(Error::from)?.to_vec(),
            headers,
        })
    }
}
