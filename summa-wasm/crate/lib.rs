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

use parking_lot::Once;
use tracing_wasm::{ConsoleConfig, WASMLayerConfigBuilder};
use wasm_bindgen::prelude::*;

mod errors;
mod js_requests;
mod web_index_registry;
mod worker_helper;

#[wasm_bindgen]
pub fn setup_logging(max_level: String) {
    console_error_panic_hook::set_once();
    let logging_config = WASMLayerConfigBuilder::new()
        .set_max_level(max_level.parse().unwrap_or_else(|_| panic!("cannot parse log level: {max_level}")))
        .set_report_logs_in_timings(true)
        .set_console_config(ConsoleConfig::ReportWithConsoleColor)
        .build();
    tracing_wasm::set_as_global_default_with_config(logging_config);
}

#[wasm_bindgen]
pub fn reserve_heap() {
    static mut HEAP: Vec<u8> = Vec::new();
    static START: Once = Once::new();
    START.call_once(|| unsafe {
        HEAP.reserve(512 * (1 << 20));
        HEAP.shrink_to_fit();
    });
}
