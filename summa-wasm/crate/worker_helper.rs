use js_sys::Promise;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use web_sys::WorkerOptions;

#[wasm_bindgen(raw_module = "../src/search-worker.ts")]
extern "C" {
    #[wasm_bindgen(js_name = "start_worker")]
    // Returns Promise<Worker>
    fn start_worker(module: JsValue, memory: JsValue, shared_data: JsValue, opts: WorkerOptions) -> Promise;
}
