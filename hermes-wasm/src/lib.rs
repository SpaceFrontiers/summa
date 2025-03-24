//! WASM bindings for Hermes search engine
//!
//! Provides browser-compatible search functionality with remote index loading.
//!
//! Supports two modes:
//! - HTTP: Uses hermes-core's HttpDirectory for standard HTTP servers
//! - IPFS: Uses JavaScript callbacks for fetching (e.g., via @helia/verified-fetch)

#![cfg(target_arch = "wasm32")]

mod idb;
mod ipfs_index;
mod local_index;
mod query;
mod registry;
mod remote_index;

use wasm_bindgen::prelude::*;

// Re-export public types
pub use ipfs_index::IpfsIndex;
pub use local_index::LocalIndex;
pub use registry::IndexRegistry;
pub use remote_index::RemoteIndex;

/// Default cache size: 10MB
pub(crate) const DEFAULT_CACHE_SIZE: usize = 10 * 1024 * 1024;

/// Initialize panic hook and logging (defaults to Warn level)
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Warn).ok();
}

/// Setup logging to browser console (can be called explicitly)
#[wasm_bindgen]
pub fn setup_logging() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Warn).ok();
}

/// Set log level: "error", "warn", "info", "debug", "trace", "off"
#[wasm_bindgen]
pub fn set_log_level(level: &str) {
    let filter = level
        .parse::<log::LevelFilter>()
        .unwrap_or(log::LevelFilter::Warn);
    log::set_max_level(filter);
}

/// Resolve field name strings to a set of field IDs.
pub(crate) fn resolve_field_ids(
    schema: &hermes_core::Schema,
    names: &[String],
) -> Result<rustc_hash::FxHashSet<u32>, JsValue> {
    let mut ids = rustc_hash::FxHashSet::default();
    for name in names {
        let field = schema
            .get_field(name)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", name)))?;
        ids.insert(field.0);
    }
    Ok(ids)
}

/// Fetch bytes from a URL using the Fetch API
pub(crate) async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, Response};

    let opts = RequestInit::new();
    opts.set_method("GET");

    let request = Request::new_with_str_and_init(url, &opts)?;

    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!("HTTP error: {}", resp.status())));
    }

    let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    Ok(uint8_array.to_vec())
}
