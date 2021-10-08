use crate::errors::SummaWasmResult;
use crate::Header;
use js_sys::Uint8Array;
use std::collections::{Bound, HashMap};
use std::ops::RangeBounds;
use strfmt::strfmt;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

#[wasm_bindgen(raw_module = "./gate.ts")]
extern "C" {
    #[wasm_bindgen(catch)]
    pub fn request(method: String, url: String, headers: JsValue) -> Result<Uint8Array, JsValue>;

    #[wasm_bindgen(catch)]
    pub async fn request_async(method: String, url: String, headers: JsValue) -> Result<JsValue, JsValue>;
}

pub struct Request {
    pub method: String,
    pub url: String,
    pub headers: Option<Vec<Header>>,
}

impl Request {
    pub fn send(&self) -> SummaWasmResult<Uint8Array> {
        Ok(request(
            self.method.to_string(),
            self.url.to_string(),
            serde_wasm_bindgen::to_value(&self.headers)?,
        )?)
    }

    pub async fn send_async(&self) -> SummaWasmResult<Uint8Array> {
        let response = request_async(self.method.to_string(), self.url.to_string(), serde_wasm_bindgen::to_value(&self.headers)?).await?;
        let array = Uint8Array::new(&response);
        Ok(array)
    }
}

#[derive(Clone, Debug)]
pub struct RequestGenerator {
    method: String,
    url_template: String,
    headers_template: Option<Vec<Header>>,
}

impl RequestGenerator {
    pub fn new(method: String, url_template: String, headers_template: Option<Vec<Header>>) -> RequestGenerator {
        RequestGenerator {
            method,
            url_template,
            headers_template,
        }
    }

    pub fn generate(&self, file_name: &str, range: impl RangeBounds<usize>) -> SummaWasmResult<Request> {
        let mut vars = HashMap::new();
        let start = match range.start_bound() {
            Bound::Included(s) => s.to_string(),
            Bound::Excluded(s) => (s + 1).to_string(),
            Bound::Unbounded => 0.to_string(),
        };
        let end = match range.end_bound() {
            Bound::Included(e) => e.to_string(),
            Bound::Excluded(e) => (e - 1).to_string(),
            Bound::Unbounded => "".to_string(),
        };
        vars.insert("file_name".to_string(), file_name);
        vars.insert("start".to_string(), &start);
        vars.insert("end".to_string(), &end);
        let headers = match self.headers_template.as_ref() {
            None => None,
            Some(headers_template) => {
                let mut headers = Vec::with_capacity(headers_template.len());
                for header in headers_template.iter() {
                    headers.push(Header {
                        name: header.name.clone(),
                        value: strfmt(&header.value, &vars)?,
                    });
                }
                Some(headers)
            }
        };
        Ok(Request {
            method: self.method.to_string(),
            url: strfmt(&self.url_template, &vars)?,
            headers,
        })
    }
}
