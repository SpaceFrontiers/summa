#![deny(
    non_shorthand_field_patterns,
    no_mangle_generic_items,
    overflowing_literals,
    path_statements,
    unused_allocation,
    unused_comparisons,
    unused_parens,
    while_true,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_must_use,
    clippy::unwrap_used
)]

#[macro_use]
extern crate async_trait;

mod errors;
mod js_requests;
mod tracker;
mod web_index_registry;
mod worker_helper;

use once_cell::sync::Lazy;
use serde_wasm_bindgen::Serializer;
pub use worker_helper::{worker_entry_point, ThreadPool};
pub static SERIALIZER: Lazy<Serializer> = Lazy::new(|| Serializer::new().serialize_maps_as_objects(true));
