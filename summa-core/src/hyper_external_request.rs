use std::path::PathBuf;

use hyper::client::HttpConnector;
use hyper::{Client, Method, Request};
use hyper_tls::HttpsConnector;
use tracing::info;

use crate::directories::{ExternalRequest, ExternalResponse, Header, RequestError};

#[derive(Clone, Debug)]
pub struct HyperExternalRequest {
    pub client: Client<HttpsConnector<HttpConnector>>,
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
        let https = HttpsConnector::new();
        let client = Client::builder().build::<_, hyper::Body>(https);
        HyperExternalRequest {
            client,
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
        let mut request = Request::builder().uri(&self.url).method(Method::from_bytes(self.method.as_bytes())?);
        for header in self.headers.iter() {
            request = request.header(&header.name, &header.value);
        }
        info!(action = "network_request", request = ?request);
        let response = self.client.request(request.body(hyper::Body::empty())?).await?;
        if response.status() == 404 {
            return Err(RequestError::NotFound(PathBuf::from(self.url)));
        }
        if response.status().as_u16() < 200 || response.status().as_u16() >= 300 {
            return Err(RequestError::External(format!("status: {} for {}", response.status(), self.url)));
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
            data: hyper::body::to_bytes(response).await?.to_vec(),
            headers,
        })
    }

    fn url(&self) -> &str {
        &self.url
    }
}
