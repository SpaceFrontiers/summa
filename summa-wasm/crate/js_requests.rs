use summa_core::directories::{ExternalRequest, ExternalResponse, Header};
use summa_core::errors::SummaResult;
use summa_core::Error;
use tokio::sync::mpsc::unbounded_channel;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::spawn_local;

#[wasm_bindgen(raw_module = "../src/gate.ts")]
extern "C" {
    #[wasm_bindgen(catch)]
    pub fn request(method: String, url: String, headers: JsValue, timeout_ms: Option<u32>) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(catch)]
    pub async fn request_async(method: String, url: String, headers: JsValue, timeout_ms: Option<u32>) -> Result<JsValue, JsValue>;
}

#[derive(Clone, Debug)]
pub struct JsExternalRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<Header>,
    pub timeout_ms: Option<u32>,
}

#[async_trait]
impl ExternalRequest for JsExternalRequest {
    fn new(method: &str, url: &str, headers: &[Header], timeout_ms: Option<u32>) -> Self
    where
        Self: Sized,
    {
        let default_headers = &[];
        JsExternalRequest {
            method: method.to_string(),
            url: url.to_string(),
            headers: Vec::from_iter(headers.iter().chain(default_headers.iter()).cloned()),
            timeout_ms,
        }
    }

    fn request(self) -> SummaResult<ExternalResponse> {
        let response = request(
            self.method,
            self.url,
            serde_wasm_bindgen::to_value(&self.headers).map_err(|e| Error::External(e.to_string()))?,
            self.timeout_ms,
        )
        .map_err(|error| Error::External(format!("{error:?}")))?;
        let response = serde_wasm_bindgen::from_value(response).unwrap_throw();
        Ok(response)
    }

    async fn request_async(self) -> SummaResult<ExternalResponse> {
        let (sender, mut receiver) = unbounded_channel();
        spawn_local(async move {
            let headers = serde_wasm_bindgen::to_value(&self.headers).expect("headers are not serializable");
            let response = request_async(self.method, self.url, headers, self.timeout_ms).await;
            let response = response
                .map(|response| serde_wasm_bindgen::from_value(response).unwrap_throw())
                .map_err(|error| Error::External(format!("{error:?}")));
            sender.send(response).unwrap_throw();
        });
        let response = receiver.recv().await.unwrap_throw()?;
        Ok(response)
    }

    fn url(&self) -> &str {
        &self.url
    }
}
