extern crate core;

use std::sync::Arc;

use pyo3::prelude::*;
use pythonize::depythonize;
use summa_server::configs::server::ConfigHolder;
use summa_server::{Server, SummaServerResult, ThreadHandler};
use tokio::sync::Mutex;

#[pyclass]
pub struct SummaEmbedServerBin {
    server_config: summa_server::configs::server::Config,
    thread_handler: Arc<Mutex<Option<ThreadHandler<SummaServerResult<()>>>>>,
}

#[pymethods]
impl SummaEmbedServerBin {
    #[new]
    pub fn new(config: &PyAny) -> Self {
        let server_config: summa_server::configs::server::Config = depythonize(config).unwrap();
        SummaEmbedServerBin {
            server_config,
            thread_handler: Arc::default(),
        }
    }

    fn start<'a>(&self, py: Python<'a>) -> PyResult<&'a PyAny> {
        let thread_handler = self.thread_handler.clone().lock_owned();
        let server_config = self.server_config.clone();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let mut thread_handler = thread_handler.await;
            if thread_handler.is_some() {
                return Ok(());
            }
            let server_config_holder = ConfigHolder::from_config(server_config);
            let server = Server::from_server_config(server_config_holder);
            let (sender, receiver) = async_broadcast::broadcast(1);
            let fut = server.serve(receiver).await.unwrap();
            let join_handle = tokio::spawn(fut);
            *thread_handler = Some(ThreadHandler::new(join_handle, sender));
            Ok(())
        })
    }

    fn stop<'a>(&self, py: Python<'a>) -> PyResult<&'a PyAny> {
        let thread_handler = self.thread_handler.clone().lock_owned();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let mut thread_handler = thread_handler.await;
            if let Some(thread_handler) = thread_handler.take() {
                thread_handler.force_stop().await.unwrap().unwrap();
            }
            Ok(())
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn summa_embed(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<SummaEmbedServerBin>()?;
    Ok(())
}
