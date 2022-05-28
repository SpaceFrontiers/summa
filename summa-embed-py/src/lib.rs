use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use prost::Message;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pythonize::pythonize;
use summa_core::components::{Driver, IndexHolder};
use summa_core::configs::{ConfigProxy, DirectProxy};
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_core::hyper_external_request::HyperExternalRequest;
use summa_proto::proto;
use tantivy::IndexBuilder;

struct SummaPyError(String);

impl From<summa_core::Error> for SummaPyError {
    fn from(value: summa_core::Error) -> Self {
        SummaPyError(format!("{:?}", value))
    }
}

impl From<SummaPyError> for PyErr {
    fn from(err: SummaPyError) -> PyErr {
        PyOSError::new_err(err.0)
    }
}

#[pyclass]
#[derive(Clone)]
struct IndexRegistry {
    index_registry: summa_core::components::IndexRegistry,
    core_config: Arc<dyn ConfigProxy<summa_core::configs::core::Config>>,
}

#[pymethods]
impl IndexRegistry {
    #[new]
    pub fn new() -> Self {
        let core_config = summa_core::configs::core::ConfigBuilder::default()
            .dedicated_compression_thread(false)
            .writer_threads(None)
            .build()
            .expect("cannot build");
        let core_config = Arc::new(DirectProxy::new(core_config)) as Arc<dyn ConfigProxy<_>>;
        IndexRegistry {
            index_registry: summa_core::components::IndexRegistry::new(&core_config),
            core_config,
        }
    }

    fn add<'a>(&'a self, py: Python<'a>, index_engine_config: &PyBytes, index_name: Option<String>) -> PyResult<&'a PyAny> {
        let index_engine_config = proto::IndexEngineConfig::decode(index_engine_config.as_bytes()).unwrap();
        let this = self.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let index = match &index_engine_config.config {
                Some(proto::index_engine_config::Config::Memory(memory_engine_config)) => {
                    let schema = serde_json::from_str(&memory_engine_config.schema).unwrap();
                    let index_builder = IndexBuilder::new().schema(schema);
                    IndexHolder::create_memory_index(index_builder).unwrap()
                }
                Some(proto::index_engine_config::Config::Remote(remote_engine_config)) => IndexHolder::open_remote_index::<
                    HyperExternalRequest,
                    DefaultExternalRequestGenerator<HyperExternalRequest>,
                >(remote_engine_config.clone(), true)
                .await
                .unwrap(),
                _ => unimplemented!(),
            };
            let core_config = this.core_config.read().await.get().clone();
            let index_holder = tokio::task::spawn_blocking(move || {
                IndexHolder::create_holder(
                    &core_config,
                    index,
                    index_name.as_deref(),
                    Arc::new(DirectProxy::new(index_engine_config)),
                    None,
                    HashMap::new(),
                    true,
                    Driver::current_tokio(),
                )
            })
            .await
            .unwrap()
            .unwrap();
            let index_attributes = index_holder.index_attributes().cloned();
            this.index_registry.add(index_holder).await.unwrap();
            Ok(Python::with_gil(|py| pythonize(py, &index_attributes).unwrap()))
        })
    }

    fn search<'a>(&'a self, py: Python<'a>, index_queries: &PyBytes) -> PyResult<&'a PyAny> {
        let search_request = proto::SearchRequest::decode(index_queries.as_bytes()).unwrap();
        let this = self.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let futures = this.index_registry.search_futures(search_request.index_queries).unwrap();
            let extraction_results = join_all(futures).await.into_iter().map(|r| r.expect("cannot receive")).collect::<Vec<_>>();
            let result = this.index_registry.finalize_extraction(extraction_results).await.unwrap();
            Ok(Python::with_gil(|py| pythonize(py, &result).unwrap()))
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "summa_embed_bin")]
fn summa_embed_bin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IndexRegistry>()?;
    Ok(())
}
