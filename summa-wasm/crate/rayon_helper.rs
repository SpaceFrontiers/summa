use js_sys::Promise;
use spmc::{channel, Receiver, Sender};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

// Naming is a workaround for https://github.com/rustwasm/wasm-bindgen/issues/2429
// and https://github.com/rustwasm/wasm-bindgen/issues/1762.
#[allow(non_camel_case_types)]
#[wasm_bindgen]
#[doc(hidden)]
pub struct wbg_rayon_PoolBuilder {
    num_threads: usize,
    sender: Sender<rayon::ThreadBuilder>,
    receiver: Receiver<rayon::ThreadBuilder>,
}

#[wasm_bindgen(raw_module = "../src/rayon-worker.ts")]
extern "C" {
    #[wasm_bindgen(js_name = start_workers)]
    fn start_workers(module: JsValue, memory: JsValue, builder: wbg_rayon_PoolBuilder) -> Promise;
}

#[wasm_bindgen]
impl wbg_rayon_PoolBuilder {
    fn new(num_threads: usize) -> Self {
        let (sender, receiver) = channel();
        Self { num_threads, sender, receiver }
    }

    #[wasm_bindgen(js_name = num_threads)]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn receiver(&self) -> *const Receiver<rayon::ThreadBuilder> {
        &self.receiver
    }

    pub fn build(&mut self) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .spawn_handler(move |thread| {
                self.sender.send(thread).unwrap_throw();
                Ok(())
            })
            .build_global()
            .unwrap_throw();
    }
}

#[wasm_bindgen(js_name = init_thread_pool)]
#[doc(hidden)]
pub fn init_thread_pool(num_threads: usize) -> Promise {
    console_error_panic_hook::set_once();
    start_workers(wasm_bindgen::module(), wasm_bindgen::memory(), wbg_rayon_PoolBuilder::new(num_threads))
}

#[wasm_bindgen]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[doc(hidden)]
pub fn wbg_rayon_start_worker(receiver: *const Receiver<rayon::ThreadBuilder>)
where
    // Statically assert that it's safe to accept `Receiver` from another thread.
    Receiver<rayon::ThreadBuilder>: Sync,
{
    // This is safe, because we know it came from a reference to PoolBuilder,
    // allocated on the heap by wasm-bindgen and dropped only once all the
    // threads are running.
    //
    // The only way to violate safety is if someone externally calls
    // `exports.wbg_rayon_start_worker(garbageValue)`, but then no Rust tools
    // would prevent us from issues anyway.
    let receiver = unsafe { &*receiver };
    // Wait for a task (`ThreadBuilder`) on the channel, and, once received,
    // start executing it.
    //
    // On practice this will start running Rayon's internal event loop.
    receiver.recv().unwrap_throw().run()
}
