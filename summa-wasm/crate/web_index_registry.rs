use std::sync::{Arc, Once};

use serde::Serialize;
use summa_core::components::{IndexHolder, IndexRegistry, SummaDocument};
use summa_core::configs::core::ExecutionStrategy;
use summa_core::configs::{ConfigProxy, DirectProxy};
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_core::errors::SummaResult;
use summa_proto::proto;
use summa_proto::proto::{IndexAttributes, IndexEngineConfig, RemoteEngineConfig};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

use crate::errors::{Error, SummaWasmResult};
use crate::js_requests::JsExternalRequest;
use crate::tracker::SubscribeTracker;
use crate::SERIALIZER;

#[wasm_bindgen]
#[derive(Clone, Default)]
pub struct JsTracker(SubscribeTracker);

#[wasm_bindgen]
impl JsTracker {
    #[wasm_bindgen]
    pub fn add_subscriber(&mut self, subscriber: js_sys::Function) {
        self.0.add_subscriber(Box::new(move |op| {
            subscriber
                .call1(&JsValue::null(), &serde_wasm_bindgen::to_value(&op).expect("cannot serialize"))
                .expect("set status failed");
        }))
    }
}

#[wasm_bindgen]
pub struct JsAddOperation {
    web_index_registry: WebIndexRegistry,
    remote_engine_config: RemoteEngineConfig,
    tracker: JsTracker,
}

#[wasm_bindgen]
impl JsAddOperation {
    pub(crate) fn new(web_index_registry: WebIndexRegistry, remote_engine_config: RemoteEngineConfig) -> JsAddOperation {
        JsAddOperation {
            web_index_registry,
            remote_engine_config,
            tracker: JsTracker::default(),
        }
    }
    pub fn tracker(&self) -> JsTracker {
        self.tracker.clone()
    }

    pub async fn execute(self) -> Result<JsValue, JsValue> {
        Ok(self
            .web_index_registry
            .add_internal(self.remote_engine_config, self.tracker.0)
            .await?
            .serialize(&*SERIALIZER)?)
    }
}

#[wasm_bindgen]
pub struct JsSearchOperation {
    web_index_registry: WebIndexRegistry,
    index_queries: Vec<proto::IndexQuery>,
    tracker: JsTracker,
}

#[wasm_bindgen]
impl JsSearchOperation {
    pub(crate) fn new(web_index_registry: WebIndexRegistry, index_queries: Vec<proto::IndexQuery>) -> JsSearchOperation {
        JsSearchOperation {
            web_index_registry,
            index_queries,
            tracker: JsTracker::default(),
        }
    }
    pub fn tracker(&self) -> JsTracker {
        self.tracker.clone()
    }

    pub async fn execute(self) -> Result<JsValue, JsValue> {
        Ok(self
            .web_index_registry
            .index_registry
            .search(&self.index_queries, self.tracker.0)
            .await
            .map_err(Error::from)?
            .serialize(&*SERIALIZER)?)
    }
}

#[wasm_bindgen]
pub struct JsWarmupOperation {
    web_index_registry: WebIndexRegistry,
    index_name: String,
    tracker: JsTracker,
}

#[wasm_bindgen]
impl JsWarmupOperation {
    pub(crate) fn new(web_index_registry: WebIndexRegistry, index_name: &str) -> JsWarmupOperation {
        JsWarmupOperation {
            web_index_registry,
            index_name: index_name.to_string(),
            tracker: JsTracker::default(),
        }
    }
    pub fn tracker(&self) -> JsTracker {
        self.tracker.clone()
    }

    pub async fn execute(self) -> Result<(), JsValue> {
        let index_holder = self
            .web_index_registry
            .index_registry
            .get_index_holder_by_name(&self.index_name)
            .await
            .map_err(Error::from)?;
        Ok(index_holder.partial_warmup(self.tracker.0).await.map_err(Error::from)?)
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebIndexRegistry {
    index_registry: IndexRegistry,
    core_config: Arc<dyn ConfigProxy<summa_core::configs::core::Config>>,
}

#[wasm_bindgen]
impl WebIndexRegistry {
    fn reserve_heap() {
        static mut HEAP: Vec<u8> = Vec::new();
        static START: Once = Once::new();
        START.call_once(|| unsafe {
            HEAP.reserve(512 * (1 << 20));
            HEAP.shrink_to_fit();
        });
    }

    #[wasm_bindgen(constructor)]
    pub fn new(multithreading: bool) -> WebIndexRegistry {
        console_error_panic_hook::set_once();
        WebIndexRegistry::reserve_heap();
        let core_config = summa_core::configs::core::ConfigBuilder::default()
            .dedicated_compression_thread(false)
            .writer_threads(0)
            .execution_strategy(match multithreading {
                true => ExecutionStrategy::GlobalPool,
                false => ExecutionStrategy::Async,
            })
            .build()
            .expect("cannot build");
        let core_config = Arc::new(DirectProxy::new(core_config)) as Arc<dyn ConfigProxy<_>>;
        WebIndexRegistry {
            index_registry: IndexRegistry::new(&core_config),
            core_config,
        }
    }

    #[wasm_bindgen]
    pub fn search(&self, index_queries: JsValue) -> Result<JsSearchOperation, JsValue> {
        let index_queries: Vec<proto::IndexQuery> = serde_wasm_bindgen::from_value(index_queries)?;
        Ok(JsSearchOperation::new(self.clone(), index_queries))
    }

    #[wasm_bindgen]
    pub fn add(&self, remote_engine_config: JsValue) -> Result<JsAddOperation, JsValue> {
        let remote_engine_config: RemoteEngineConfig = serde_wasm_bindgen::from_value(remote_engine_config)?;
        Ok(JsAddOperation::new(self.clone(), remote_engine_config))
    }

    async fn add_internal(&self, remote_engine_config: RemoteEngineConfig, tracker: SubscribeTracker) -> SummaWasmResult<Option<IndexAttributes>> {
        let index =
            IndexHolder::attach_remote_index::<JsExternalRequest, DefaultExternalRequestGenerator<JsExternalRequest>>(remote_engine_config.clone(), tracker)
                .await?;
        let core_config_value = self.core_config.read().await.get().clone();
        let index_holder = IndexHolder::create_holder(
            &self.core_config,
            &core_config_value,
            index,
            None,
            Arc::new(DirectProxy::new(IndexEngineConfig {
                config: Some(proto::index_engine_config::Config::Remote(remote_engine_config)),
            })),
            true,
        )?;
        let index_attributes = index_holder.index_attributes().cloned();
        self.index_registry.add(index_holder).await;
        Ok(index_attributes)
    }

    #[wasm_bindgen]
    pub async fn delete(&mut self, index_name: String) {
        self.index_registry.delete(&index_name).await;
    }

    #[wasm_bindgen]
    pub fn warmup(&self, index_name: &str) -> Result<JsWarmupOperation, JsValue> {
        Ok(JsWarmupOperation::new(self.clone(), index_name))
    }

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

    #[wasm_bindgen]
    pub async fn commit(&self, index_name: &str) -> Result<(), JsValue> {
        Ok(self.commit_internal(index_name).await.map_err(Error::from)?)
    }
}
