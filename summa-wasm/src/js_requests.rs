use async_channel::bounded;
use js_sys::Uint8Array;
use summa_core::directories::{ExternalRequest, Header};
use summa_core::errors::SummaResult;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::spawn_local;

#[wasm_bindgen(raw_module = "./gate.ts")]
extern "C" {
    #[wasm_bindgen(catch)]
    pub fn request(method: String, url: String, headers: JsValue) -> Result<Uint8Array, JsValue>;

    #[wasm_bindgen(catch)]
    pub async fn request_async(method: String, url: String, headers: JsValue) -> Result<JsValue, JsValue>;
}

#[derive(Clone, Debug)]
pub struct JsExternalRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<Header>,
}

#[async_trait]
impl ExternalRequest for JsExternalRequest {
    fn new(method: &str, url: &str, headers: &[Header]) -> Self
    where
        Self: Sized,
    {
        JsExternalRequest {
            method: method.to_string(),
            url: url.to_string(),
            headers: Vec::from_iter(headers.iter().map(|header| header.clone())),
        }
    }

    fn request(&self) -> SummaResult<Vec<u8>> {
        let response = request(
            self.method.to_string(),
            self.url.to_string(),
            serde_wasm_bindgen::to_value(&self.headers).unwrap(),
        )
        .map_err(|e| summa_core::Error::External(format!("{:?}", e)))?;
        let array = Uint8Array::new(&response);
        Ok(array.to_vec())
    }

    async fn request_async(&self) -> SummaResult<Vec<u8>> {
        let (sender, receiver) = bounded(1);
        let method = self.method.to_string();
        let url = self.url.to_string();
        let headers = self.headers.clone();
        spawn_local(async move {
            let headers = serde_wasm_bindgen::to_value(&headers).unwrap();
            let response = request_async(method, url, headers)
                .await
                .map(|response| Uint8Array::new(&response).to_vec())
                .map_err(|e| summa_core::Error::External(format!("{:?}", e)));
            sender.send(response).await.unwrap_throw();
        });
        let array = receiver.recv().await.unwrap_throw()?;
        Ok(array)
    }
}
