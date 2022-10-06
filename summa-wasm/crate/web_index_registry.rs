use std::sync::Arc;

use serde::Serialize;
use serde_wasm_bindgen::Serializer;
use summa_core::components::{IndexHolder, IndexRegistry, SummaDocument};
use summa_core::configs::{ConfigProxy, DirectProxy};
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_core::errors::SummaResult;
use summa_proto::proto;
use tantivy::IndexBuilder;
use tracing::{info, trace};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

use crate::errors::{Error, SummaWasmResult};
use crate::js_requests::JsExternalRequest;

/// Hold `IndexRegistry` and wrap it for making WASM-compatible
#[wasm_bindgen]
#[derive(Clone)]
pub struct WrappedIndexRegistry {
    index_registry: IndexRegistry,
    core_config: Arc<dyn ConfigProxy<summa_core::configs::core::Config>>,
}

#[wasm_bindgen]
impl WrappedIndexRegistry {
    /// Create new `WrappedIndexRegistry`
    #[wasm_bindgen(constructor)]
    pub fn new() -> WrappedIndexRegistry {
        let core_config = summa_core::configs::core::ConfigBuilder::default()
            .doc_store_compress_threads(0)
            .writer_threads(None)
            .build()
            .expect("cannot build");
        let core_config = Arc::new(DirectProxy::new(core_config)) as Arc<dyn ConfigProxy<_>>;
        WrappedIndexRegistry {
            index_registry: IndexRegistry::new(&core_config),
            core_config,
        }
    }

    /// Do pooled search
    #[wasm_bindgen]
    pub async fn search(&self, search_request: JsValue) -> Result<JsValue, JsValue> {
        let search_request: proto::SearchRequest = serde_wasm_bindgen::from_value(search_request)?;
        let serializer = Serializer::new().serialize_maps_as_objects(true).serialize_large_number_types_as_bigints(true);
        Ok(self.search_internal(search_request).await.map_err(Error::from)?.serialize(&serializer)?)
    }

    async fn search_internal(&self, search_request: proto::SearchRequest) -> SummaResult<Vec<proto::CollectorOutput>> {
        info!(action = "search", search_request = ?search_request);
        let index_holder = self.index_registry.get_index_holder(&search_request.index_alias).await?;
        let collector_outputs = index_holder
            .custom_search(
                &search_request.index_alias,
                search_request
                    .query
                    .and_then(|query| query.query)
                    .unwrap_or_else(|| proto::query::Query::All(proto::AllQuery {})),
                search_request.collectors,
                search_request.is_fieldnorms_scoring_enabled,
                search_request.load_cache,
                search_request.store_cache,
            )
            .await?;
        trace!(action = "searched");
        self.index_registry.finalize_extraction(collector_outputs).await
    }

    /// Add new index to `WrappedIndexRegistry`
    #[wasm_bindgen]
    pub async fn add(&self, index_name: &str, index_engine_config: JsValue) -> Result<JsValue, JsValue> {
        let index_engine_config: proto::IndexEngineConfig = serde_wasm_bindgen::from_value(index_engine_config)?;
        let serializer = Serializer::new().serialize_maps_as_objects(true).serialize_large_number_types_as_bigints(true);
        Ok(self
            .add_internal(index_name, index_engine_config)
            .await
            .map_err(Error::from)?
            .serialize(&serializer)?)
    }

    async fn add_internal(&self, index_name: &str, index_engine_config: proto::IndexEngineConfig) -> SummaWasmResult<Option<proto::IndexAttributes>> {
        let index = match &index_engine_config.config {
            Some(proto::index_engine_config::Config::Memory(memory_engine_config)) => {
                let schema = serde_wasm_bindgen::from_value(memory_engine_config.schema.clone().into()).expect("cannot parse schema");
                let index_builder = IndexBuilder::new().schema(schema);
                IndexHolder::create_memory_index(index_builder)?
            }
            Some(proto::index_engine_config::Config::Remote(remote_engine_config)) => {
                IndexHolder::open_remote_index::<JsExternalRequest, DefaultExternalRequestGenerator<JsExternalRequest>>(remote_engine_config.clone(), true)
                    .await?
            }
            _ => unimplemented!(),
        };
        let query_parser_config = index_engine_config.query_parser_config.as_ref().cloned().unwrap_or_default();
        let index_holder = IndexHolder::create_holder(
            self.core_config.read().await.get(),
            index,
            index_name,
            Arc::new(DirectProxy::new(index_engine_config)),
            None,
            query_parser_config,
        )?;
        let index_attributes = index_holder.index_attributes().cloned();
        self.index_registry.add(index_holder).await?;
        Ok(index_attributes)
    }

    /// Remove index from `WrappedIndexRegistry`
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
            .partial_warmup(false, &[] as &[&str; 0])
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

    /// Make commit
    #[wasm_bindgen]
    pub async fn commit(&self, index_name: &str) -> Result<(), JsValue> {
        Ok(self.commit_internal(index_name).await.map_err(Error::from)?)
    }

    async fn commit_internal(&self, index_name: &str) -> SummaResult<()> {
        let index_holder = self.index_registry.get_index_holder_by_name(index_name).await?;
        index_holder.index_writer_holder()?.write().await.commit()?;
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn get_index_field_names(&self, index_name: &str) -> Result<JsValue, JsValue> {
        let index_holder = self.index_registry.get_index_holder(index_name).await.map_err(Error::from)?;
        Ok(serde_wasm_bindgen::to_value(
            &index_holder
                .schema()
                .fields()
                .map(|(_, field_entry)| field_entry.name().to_string())
                .collect::<Vec<_>>(),
        )?)
    }
}

impl Default for WrappedIndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}
