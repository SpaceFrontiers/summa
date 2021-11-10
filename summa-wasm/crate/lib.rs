#[macro_use]
extern crate async_trait;

mod errors;
mod js_requests;
mod rayon_helper;
mod web_index_registry;

use once_cell::sync::Lazy;
use serde::Serialize;
use serde_wasm_bindgen::Serializer;
use summa_core::components::CACHE_METRICS;
use wasm_bindgen::prelude::*;

pub static SERIALIZER: Lazy<Serializer> = Lazy::new(|| Serializer::new().serialize_maps_as_objects(true));

#[wasm_bindgen]
pub async fn cache_metrics() -> Result<JsValue, JsValue> {
    Ok(CACHE_METRICS.serialize(&*SERIALIZER)?)
}
