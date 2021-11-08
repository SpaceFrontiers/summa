use hyper::client::HttpConnector;
use hyper::{Client, Method, Request};
use summa_core::directories::{ExternalRequest, Header};
use summa_core::errors::{SummaResult, ValidationError};
use summa_core::Error;
use tonic::async_trait;

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

    fn request(&self) -> SummaResult<Vec<u8>> {
        todo!()
    }

    async fn request_async(&self) -> SummaResult<Vec<u8>> {
        let mut request = Request::builder()
            .uri(&self.url)
            .method(Method::from_bytes(self.method.as_bytes()).map_err(|_| Error::Validation(ValidationError::InvalidHttpMethod(self.method.to_string())))?);
        for header in self.headers.iter() {
            request = request.header(&header.name, &header.value);
        }
        let response = self
            .client
            .request(
                request
                    .body(hyper::Body::empty())
                    .map_err(|e| summa_core::Error::External(format!("{:?}", e)))?,
            )
            .await
            .map_err(|e| summa_core::Error::External(format!("{:?}", e)))?;
        Ok(hyper::body::to_bytes(response)
            .await
            .map_err(|e| summa_core::Error::External(format!("{:?}", e)))?
            .to_vec())
    }
}
