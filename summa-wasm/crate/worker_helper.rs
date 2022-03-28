use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::future::BoxFuture;
use js_sys::Promise;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot::error::RecvError;
use tokio::sync::{mpsc, oneshot};
use tracing::trace;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{DedicatedWorkerGlobalScope, WorkerOptions, WorkerType};

#[wasm_bindgen(raw_module = "../src/search-worker.ts")]
extern "C" {
    #[wasm_bindgen(js_name = "start_worker")]
    // Returns Promise<Worker>
    fn start_worker(module: JsValue, memory: JsValue, shared_data: JsValue, opts: WorkerOptions) -> Promise;
}

pub struct ThreadPool {
    state: Arc<PoolState>,
}

impl Clone for ThreadPool {
    fn clone(&self) -> Self {
        self.state.cnt.fetch_add(1, Ordering::Relaxed);
        Self { state: self.state.clone() }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        if self.state.cnt.fetch_sub(1, Ordering::Relaxed) == 1 {
            for _ in 0..self.state.size {
                self.state.send(Message::Close);
            }
        }
    }
}

impl ThreadPool {
    /// Creates a new [`ThreadPool`] with the provided count of web workers. The returned future
    /// will resolve after all workers have spawned and are ready to accept work.
    pub async fn new(size: usize) -> Result<ThreadPool, JsValue> {
        let (tx, rx) = unbounded_channel();
        let pool = ThreadPool {
            state: Arc::new(PoolState {
                tx: parking_lot::Mutex::new(tx),
                rx: tokio::sync::Mutex::new(rx),
                cnt: AtomicUsize::new(1),
                size,
            }),
        };

        for idx in 0..size {
            let state = pool.state.clone();

            let mut opts = WorkerOptions::new();
            opts.type_(WorkerType::Module);
            opts.name(&format!("Worker-{idx}"));

            // With a worker spun up send it the module/memory so it can start
            // instantiating the wasm module. Later it might receive further
            // messages about code to run on the wasm module.
            let ptr = Arc::into_raw(state);
            let _worker =
                wasm_bindgen_futures::JsFuture::from(start_worker(wasm_bindgen::module(), wasm_bindgen::memory(), JsValue::from(ptr as u32), opts)).await?;
            // TODO: Check that workers actually spawned.
        }
        Ok(pool)
    }

    /// Creates a new [`ThreadPool`] with `Navigator.hardwareConcurrency` web workers.
    pub async fn max_threads() -> Result<Self, JsValue> {
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = navigator, js_name = hardwareConcurrency)]
            static HARDWARE_CONCURRENCY: usize;
        }
        let pool_size = std::cmp::max(*HARDWARE_CONCURRENCY, 1);
        Self::new(pool_size).await
    }

    /// Spawns a task that polls the given future with output `()` to
    /// completion.
    pub fn spawn_ok<Fut>(&self, future: Fut)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        trace!(action = "spawning_thread");
        self.state.send(Message::Run(Box::pin(future)));
    }

    /// Spawns a task. This function returns a future which eventually resolves to the output of
    /// the computation.
    /// Note: The provided future is polled on the thread pool, no matter whether the returned
    /// future is polled or not.
    pub fn spawn<Fut>(&self, future: Fut) -> impl Future<Output = Result<Fut::Output, RecvError>> + 'static
    where
        Fut: Future + Send + 'static,
        Fut::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let f = async move {
            let res = future.await;
            trace!(action = "receiving_future_result");
            let _ = tx.send(res);
        };
        self.spawn_ok(f);
        rx
    }
}

enum Message {
    Run(BoxFuture<'static, ()>),
    Close,
}

impl Debug for Message {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Message").finish()
    }
}

pub struct PoolState {
    tx: parking_lot::Mutex<mpsc::UnboundedSender<Message>>,
    rx: tokio::sync::Mutex<mpsc::UnboundedReceiver<Message>>,
    cnt: AtomicUsize,
    size: usize,
}

impl PoolState {
    fn send(&self, msg: Message) {
        self.tx.lock().send(msg).expect("cannot send")
    }

    fn work(slf: Arc<PoolState>) {
        let driver = async move {
            let global = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
            while let Some(msg) = slf.rx.lock().await.recv().await {
                match msg {
                    Message::Run(future) => wasm_bindgen_futures::spawn_local(future),
                    Message::Close => break,
                }
            }
            global.close();
        };
        wasm_bindgen_futures::spawn_local(driver);
    }
}

/// Entry point invoked by the web worker. The passed pointer will be unconditionally interpreted
/// as an `Arc<PoolState>`.
#[wasm_bindgen]
pub fn worker_entry_point(state_ptr: u32) {
    PoolState::work(unsafe { Arc::<PoolState>::from_raw(state_ptr as *const PoolState) });
}
