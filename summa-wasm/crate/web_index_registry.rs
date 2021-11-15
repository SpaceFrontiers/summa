use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use summa_core::components::{IndexHolder, IndexQuery, IndexRegistry, SummaDocument};
use summa_core::configs::{DirectProxy, IndexConfigBuilder, IndexEngine};
use summa_core::directories::DefaultExternalRequestGenerator;
use tantivy::Executor;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

use crate::errors::{Error, SummaWasmResult};
use crate::js_requests::JsExternalRequest;
use crate::SERIALIZER;

#[wasm_bindgen]
pub struct WebIndexRegistry {
    index_registry: IndexRegistry,
    multithreading: bool,
}

#[derive(Deserialize, Serialize)]

struct IndexPayload {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub default_fields: Vec<String>,
    #[serde(default)]
    pub multi_fields: HashSet<String>,
    #[serde(default)]
    pub unixtime: u32,
}

#[wasm_bindgen]
impl WebIndexRegistry {
    #[wasm_bindgen(constructor)]
    pub fn new(multithreading: bool) -> WebIndexRegistry {
        console_error_panic_hook::set_once();
        WebIndexRegistry {
            index_registry: IndexRegistry::default(),
            multithreading,
        }
    }

    #[wasm_bindgen]
    pub async fn search(&self, index_queries: JsValue) -> Result<JsValue, JsValue> {
        let index_queries: Vec<IndexQuery> = serde_wasm_bindgen::from_value(index_queries)?;
        Ok(self.index_registry.search(&index_queries).await.map_err(Error::from)?.serialize(&*SERIALIZER)?)
    }

    #[wasm_bindgen]
    pub async fn add(&mut self, network_config: JsValue) -> Result<JsValue, JsValue> {
        let index_engine: IndexEngine = serde_wasm_bindgen::from_value(network_config)?;
        Ok(self.add_internal(index_engine).await.map_err(Error::from)?.serialize(&*SERIALIZER)?)
    }

    #[wasm_bindgen]
    pub async fn get_index_payload(&self, index_name: &str) -> Result<JsValue, JsValue> {
        let index_holder = self.index_registry.get_index_holder_by_name(index_name).await.map_err(Error::from)?;
        let index_payload = index_holder.index_payload().ok().flatten();
        let index_payload = index_payload.and_then(|index_payload| serde_json::from_str::<IndexPayload>(&index_payload).ok());
        Ok(index_payload.serialize(&*SERIALIZER)?)
    }

    async fn add_internal(&mut self, index_engine: IndexEngine) -> SummaWasmResult<IndexPayload> {
        let mut index = IndexHolder::open::<JsExternalRequest, DefaultExternalRequestGenerator<JsExternalRequest>>(&index_engine)?;
        let index_payload: IndexPayload =
            serde_json::from_str(&index.load_metas()?.payload.ok_or(Error::IncorrectPayload)?).map_err(|_| Error::IncorrectPayload)?;
        let index_config = IndexConfigBuilder::default()
            .index_engine(index_engine)
            .default_fields(index_payload.default_fields.clone())
            .multi_fields(index_payload.multi_fields.clone())
            .writer_threads(0)
            .build()
            .map_err(|e| Error::Core(e.into()))?;

        index.set_multithread_executor(match self.multithreading {
            true => Executor::GlobalPool,
            false => Executor::SingleThread,
        })?;

        let index_config = Arc::new(DirectProxy::new(index_config));
        let index_holder = IndexHolder::setup(&index_payload.name, index, index_config).await?;

        self.index_registry.add(index_holder).await;
        Ok(index_payload)
    }

    #[wasm_bindgen]
    pub async fn delete(&mut self, index_name: String) {
        self.index_registry.delete(&index_name).await;
    }

    #[wasm_bindgen]
    pub async fn warmup(&self, index_name: &str) -> Result<(), JsValue> {
        let index_holder = self.index_registry.get_index_holder_by_name(index_name).await.map_err(Error::from)?;
        index_holder.warmup().await.map_err(Error::from)?;
        Ok(())
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

    #[wasm_bindgen]
    pub async fn commit(&self, index_name: &str) -> Result<(), JsValue> {
        let index_holder = self.index_registry.get_index_holder_by_name(index_name).await.map_err(Error::from)?;
        index_holder.commit(None).await.map_err(Error::from)?;
        Ok(())
    }
}
