use crate::request;
use js_sys::Uint8Array;
use std::collections::HashMap;
use std::ops::Range;
use strfmt::strfmt;

pub struct Request {
    pub method: String,
    pub url: String,
    pub headers: Option<HashMap<String, String>>,
}

impl Request {
    pub fn send(&self) -> Uint8Array {
        request(
            self.method.to_string(),
            self.url.to_string(),
            serde_wasm_bindgen::to_value(&self.headers).unwrap(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct RequestGenerator {
    method: String,
    url_template: String,
    headers_template: Option<HashMap<String, String>>,
}

impl RequestGenerator {
    pub fn new(method: String, url_template: String, headers_template: Option<HashMap<String, String>>) -> RequestGenerator {
        RequestGenerator {
            method,
            url_template,
            headers_template,
        }
    }

    pub fn generate(&self, file_name: &str, range: Range<usize>) -> Request {
        let mut vars = HashMap::new();
        vars.insert("file_name".to_string(), file_name.to_string());
        vars.insert("start".to_string(), range.start.to_string());
        vars.insert("end".to_string(), range.end.to_string());
        vars.insert("length".to_string(), (range.end - range.start).to_string());
        Request {
            method: self.method.to_string(),
            url: strfmt(&self.url_template, &vars).unwrap(),
            headers: self.headers_template.as_ref().map(|headers_template| {
                headers_template
                    .into_iter()
                    .map(|(header_name, header_value_template)| (header_name.to_string(), strfmt(&header_value_template, &vars).unwrap()))
                    .collect()
            }),
        }
    }
}
