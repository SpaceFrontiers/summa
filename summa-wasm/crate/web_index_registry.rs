use std::sync::Arc;

use futures::future::join_all;
use serde::Serialize;
use summa_core::components::{IndexHolder, IndexRegistry, SummaDocument};
use summa_core::configs::{ConfigProxy, DirectProxy};
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_core::errors::SummaResult;
use summa_proto::proto;
use summa_proto::proto::{IndexAttributes, IndexEngineConfig, RemoteEngineConfig};
use tracing::info;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

use crate::errors::{Error, SummaWasmResult};
use crate::js_requests::JsExternalRequest;
use crate::{ThreadPool, SERIALIZER};

/// Hold `IndexRegistry` and wrap it for making WASM-compatible
#[wasm_bindgen]
#[derive(Clone)]
pub struct WebIndexRegistry {
    index_registry: IndexRegistry,
    core_config: Arc<dyn ConfigProxy<summa_core::configs::core::Config>>,
    thread_pool: Option<ThreadPool>,
}

#[wasm_bindgen]
impl WebIndexRegistry {
    /// Create new `WebIndexRegistry`
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebIndexRegistry {
        let core_config = summa_core::configs::core::ConfigBuilder::default()
            .dedicated_compression_thread(false)
            .writer_threads(None)
            .build()
            .expect("cannot build");
        let core_config = Arc::new(DirectProxy::new(core_config)) as Arc<dyn ConfigProxy<_>>;
        WebIndexRegistry {
            index_registry: IndexRegistry::new(&core_config),
            core_config,
            thread_pool: None,
        }
    }

    /// Configures WASM-pool for executing queries to different indices
    #[wasm_bindgen]
    pub async fn setup(&mut self, threads: usize) -> Result<(), JsValue> {
        self.thread_pool = Some(ThreadPool::new(threads).await?);
        Ok(())
    }

    /// Do pooled search
    #[wasm_bindgen]
    pub async fn search(&self, index_queries: JsValue) -> Result<JsValue, JsValue> {
        let index_queries: Vec<proto::IndexQuery> = serde_wasm_bindgen::from_value(index_queries)?;
        Ok(self.search_internal(index_queries).await.map_err(Error::from)?.serialize(&*SERIALIZER)?)
    }

    async fn search_internal(&self, index_queries: Vec<proto::IndexQuery>) -> SummaResult<Vec<proto::CollectorOutput>> {
        let futures = self.index_registry.search_futures(&index_queries).await?;
        info!(action = "search", index_queries = ?index_queries);
        let collectors_outputs = join_all(futures.into_iter().map(|f| self.thread_pool().spawn(f)))
            .await
            .into_iter()
            .map(|r| r.expect("cannot receive"))
            .collect::<SummaResult<Vec<_>>>()?;
        self.merge_responses(&collectors_outputs)
    }

    /// Add new index to `WebIndexRegistry`
    #[wasm_bindgen]
    pub async fn add(&self, remote_engine_config: JsValue, index_name: Option<String>) -> Result<JsValue, JsValue> {
        let remote_engine_config: RemoteEngineConfig = serde_wasm_bindgen::from_value(remote_engine_config)?;
        Ok(self
            .add_internal(remote_engine_config, index_name)
            .await
            .map_err(Error::from)?
            .serialize(&*SERIALIZER)?)
    }

    async fn add_internal(&self, remote_engine_config: RemoteEngineConfig, index_name: Option<String>) -> SummaWasmResult<Option<IndexAttributes>> {
        let index =
            IndexHolder::attach_remote_index::<JsExternalRequest, DefaultExternalRequestGenerator<JsExternalRequest>>(remote_engine_config.clone(), true)
                .await?;
        let index_holder = IndexHolder::create_holder(
            self.core_config.read().await.get(),
            index,
            index_name.as_deref(),
            Arc::new(DirectProxy::new(IndexEngineConfig {
                config: Some(proto::index_engine_config::Config::Remote(remote_engine_config)),
            })),
            true,
        )?;
        let index_attributes = index_holder.index_attributes().cloned();
        self.index_registry.add(index_holder).await;
        Ok(index_attributes)
    }

    /// Remove index from `WebIndexRegistry`
    #[wasm_bindgen]
    pub async fn delete(&mut self, index_name: String) {
        self.index_registry.delete(&index_name).await;
    }

    /// Do [partial warm up](IndexHolder::partial_warmup)
    #[wasm_bindgen]
    pub async fn warmup(&self, index_name: &str) -> Result<(), JsValue> {
        self.index_registry
            .get_index_holder_by_name(index_name)
            .await
            .map_err(Error::from)?
            .partial_warmup()
            .await
            .map_err(Error::from)?;
        Ok(())
    }

    /// Index new document
    #[wasm_bindgen]
    pub async fn index_document(&self, index_name: &str, document: &str) -> Result<(), JsValue> {
        let index_holder = self.index_registry.get_index_holder_by_name(index_name).await.map_err(Error::from)?;
        index_holder
            .index_document(SummaDocument::UnboundJsonBytes(document.as_bytes()))
            .await
            .map_err(Error::from)?;
        Ok(())
    }

    async fn commit_internal(&self, index_name: &str) -> SummaResult<()> {
        let index_holder = self.index_registry.get_index_holder_by_name(index_name).await?;
        index_holder.index_writer_holder()?.write().await.commit().await?;
        Ok(())
    }

    /// Make commit
    #[wasm_bindgen]
    pub async fn commit(&self, index_name: &str) -> Result<(), JsValue> {
        Ok(self.commit_internal(index_name).await.map_err(Error::from)?)
    }

    pub(crate) fn merge_responses(&self, collectors_outputs: &[Vec<proto::CollectorOutput>]) -> SummaResult<Vec<proto::CollectorOutput>> {
        self.index_registry.merge_responses(collectors_outputs)
    }

    pub(crate) fn thread_pool(&self) -> &ThreadPool {
        self.thread_pool
            .as_ref()
            .expect("thread_pool should be initialized through `setup` call before use")
    }
}

impl Default for WebIndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}
