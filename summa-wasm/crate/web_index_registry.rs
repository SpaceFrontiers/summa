use std::sync::Arc;

use serde::Serialize;
use summa_core::components::{IndexHolder, IndexQuery, IndexRegistry, SummaDocument};
use summa_core::configs::{ApplicationConfigBuilder, DirectProxy};
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_proto::proto::IndexEngineConfig;
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
    pub async fn add(&mut self, index_name: &str, index_engine: JsValue) -> Result<JsValue, JsValue> {
        let index_engine: IndexEngineConfig = serde_wasm_bindgen::from_value(index_engine)?;
        Ok(self
            .add_internal(index_name, index_engine)
            .await
            .map_err(Error::from)?
            .serialize(&*SERIALIZER)?)
    }

    async fn add_internal(&mut self, index_name: &str, index_engine: IndexEngineConfig) -> SummaWasmResult<()> {
        let mut index = IndexHolder::from_index_engine_config::<JsExternalRequest, DefaultExternalRequestGenerator<JsExternalRequest>>(&index_engine).await?;
        index.settings_mut().docstore_compress_dedicated_thread = false;

        let application_config = ApplicationConfigBuilder::default()
            .writer_threads(0)
            .build()
            .map_err(|e| Error::Core(e.into()))?;

        index.set_multithread_executor(match self.multithreading {
            true => Executor::GlobalPool,
            false => Executor::SingleThread,
        })?;

        let application_config = Arc::new(DirectProxy::new(application_config));
        let index_holder = IndexHolder::create_holder(application_config, index_engine, index_name, index).await?;

        self.index_registry.add(index_holder).await;
        Ok(())
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
