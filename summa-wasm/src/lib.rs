#[macro_use]
extern crate async_trait;

mod errors;
mod network_directory;
mod rayon_helper;
mod requests;
mod web_index_inner;

use wasm_bindgen::prelude::*;

fn report_to_callback(callback: &js_sys::Function, type_: &str, message: &str) -> Result<JsValue, JsValue> {
    callback.call2(&JsValue::null(), &type_.into(), &message.into())
}
