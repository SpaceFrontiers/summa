#[macro_use]
extern crate async_trait;

mod errors;
mod network_directory;
mod rayon_helper;
mod requests;

use crate::errors::{Error, SummaWasmResult};
use crate::requests::RequestGenerator;
use futures::future::try_join_all;
use network_directory::NetworkDirectory;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::Serializer;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use summa_core::default_tokenizers;
use summa_core::directories::{ChunkedCachingDirectory, HotDirectory};
use summa_core::errors::SummaResult;
use summa_core::metrics::CacheMetrics;
use summa_core::QueryParser;
use summa_core::{build_fruit_extractor, FruitExtractor};
use summa_proto::proto;
use tantivy::collector::MultiCollector;
use tantivy::directory::OwnedBytes;
use tantivy::schema::Field;
use tantivy::schema::Value;
use tantivy::{Directory, Executor, Index, IndexReader, Score};
use wasm_bindgen::prelude::*;

pub static SERIALIZER: Lazy<Serializer> = Lazy::new(|| Serializer::new().serialize_maps_as_objects(true));
pub static CACHE_METRICS: Lazy<CacheMetrics> = Lazy::new(|| CacheMetrics::default());

#[wasm_bindgen]
pub async fn cache_metrics() -> Result<JsValue, JsValue> {
    Ok(CACHE_METRICS.serialize(&*SERIALIZER)?)
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
struct File {
    name: String,
    size: usize,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct Header {
    pub name: String,
    pub value: String,
}

#[derive(Deserialize)]
struct IndexPayload {
    name: String,
    description: String,
    default_fields: Vec<String>,
    multi_fields: HashSet<String>,
    unixtime: u32,
}

#[wasm_bindgen]
pub struct WebIndexInner {
    index: Index,
    index_reader: IndexReader,
    index_payload: IndexPayload,
    default_fields: Vec<Field>,
    multi_fields: HashSet<Field>,
    query_parser: QueryParser,
}

fn report_to_callback(callback: &js_sys::Function, type_: &str, message: &str) -> Result<JsValue, JsValue> {
    callback.call2(&JsValue::null(), &type_.into(), &message.into())
}

#[wasm_bindgen]
impl WebIndexInner {
    #[wasm_bindgen(constructor)]
    pub fn new(
        method: String,
        url_template: String,
        headers_template: JsValue,
        files: JsValue,
        status_callback: &js_sys::Function,
    ) -> Result<WebIndexInner, JsValue> {
        console_error_panic_hook::set_once();
        Ok(WebIndexInner::new_internal(method, url_template, headers_template, files, status_callback)?)
    }

    pub fn new_internal(
        method: String,
        url_template: String,
        headers_template: JsValue,
        files: JsValue,
        status_callback: &js_sys::Function,
    ) -> SummaWasmResult<WebIndexInner> {
        let headers_template: Option<Vec<Header>> = serde_wasm_bindgen::from_value(headers_template)?;
        let request_generator = RequestGenerator::new(method, url_template, headers_template);

        let files: Vec<File> = serde_wasm_bindgen::from_value(files)?;
        let files = HashMap::from_iter(files.into_iter().map(|file| (file.name, file.size)));

        let network_directory = Arc::new(NetworkDirectory::new(files, request_generator.clone()));
        let caching_directory = ChunkedCachingDirectory::new_with_capacity_in_bytes(network_directory, 16 * 1024, 1024 * 1024 * 1024, CACHE_METRICS.clone());
        report_to_callback(status_callback, "status", "retrieving hotcache.bin...")?;
        let index: Index = match caching_directory.atomic_read("hotcache.bin".as_ref()) {
            Ok(hotcache_bytes) => {
                let hotcache_directory = HotDirectory::open(caching_directory, OwnedBytes::new(hotcache_bytes.to_vec()))?;
                Index::open(hotcache_directory)?
            }
            Err(_) => {
                report_to_callback(status_callback, "status", "retrieving index...")?;
                Index::open(caching_directory)?
            }
        };

        for (tokenizer_name, tokenizer) in &default_tokenizers() {
            index.tokenizers().register(tokenizer_name, tokenizer.clone())
        }

        report_to_callback(status_callback, "status", "retrieving meta.json...")?;
        let index_payload: IndexPayload =
            serde_json::from_str(&index.load_metas()?.payload.ok_or(Error::IncorrectPayload)?).map_err(|_| Error::IncorrectPayload)?;
        let schema = index.schema();

        let default_fields = index_payload
            .default_fields
            .iter()
            .map(|field_name| {
                schema
                    .get_field(field_name)
                    .ok_or_else(|| summa_core::Error::Validation(summa_core::errors::ValidationError::MissingDefaultField(field_name.to_string())))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let multi_fields = index_payload
            .multi_fields
            .iter()
            .map(|field_name| {
                schema
                    .get_field(field_name)
                    .ok_or_else(|| summa_core::Error::Validation(summa_core::errors::ValidationError::MissingDefaultField(field_name.to_string())))
            })
            .collect::<Result<HashSet<_>, _>>()?;

        let query_parser = QueryParser::for_index(&index, default_fields.clone());
        let index_reader = index.reader()?;

        Ok(WebIndexInner {
            index,
            index_reader,
            index_payload,
            default_fields,
            multi_fields,
            query_parser,
        })
    }

    #[wasm_bindgen]
    pub async fn warmup(&self) -> Result<(), JsValue> {
        Ok(self.warmup_internal().await?)
    }

    async fn warmup_internal(&self) -> SummaWasmResult<()> {
        let searcher = self.index_reader.searcher();

        let mut warm_up_futures = Vec::new();
        for field in self.default_fields.iter() {
            for segment_reader in searcher.segment_readers() {
                let inverted_index = segment_reader.inverted_index(*field)?.clone();
                warm_up_futures.push(async move {
                    let dict = inverted_index.terms();
                    dict.warm_up_dictionary().await
                });
            }
        }
        try_join_all(warm_up_futures).await?;
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn search(&self, index_name: String, query: JsValue, collectors: JsValue) -> Result<JsValue, JsValue> {
        let query = serde_wasm_bindgen::from_value(query)?;
        let collectors: Vec<_> = serde_wasm_bindgen::from_value(collectors)?;
        Ok(self.search_inner(&index_name, query, collectors).await?)
    }

    async fn search_inner(&self, external_index_alias: &str, query: proto::Query, collectors: Vec<proto::Collector>) -> SummaWasmResult<JsValue> {
        let searcher = self.index_reader.searcher();
        let parsed_query = self.query_parser.parse_query(&query)?;
        let schema = self.index.schema();
        let mut multi_collector = MultiCollector::new();
        let extractors: Vec<Box<dyn FruitExtractor>> = collectors
            .into_iter()
            .map(|collector_proto| build_fruit_extractor(collector_proto, &parsed_query, &schema, &mut multi_collector))
            .collect::<SummaResult<_>>()?;
        let external_index_alias = external_index_alias.to_string();
        let mut multi_fruit = searcher.search_with_executor(&parsed_query, &multi_collector, &Executor::GlobalPool)?;
        let mut collector_outputs = Vec::with_capacity(extractors.len());
        for extractor in extractors.into_iter() {
            collector_outputs.push(
                extractor
                    .extract_async(&external_index_alias, &mut multi_fruit, &searcher, &self.multi_fields)
                    .await,
            )
        }
        Ok(collector_outputs.serialize(&*SERIALIZER)?)
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.index_payload.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.index_payload.description.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn unixtime(&self) -> u32 {
        self.index_payload.unixtime
    }
}

#[derive(Serialize)]
struct ScoredDocument {
    score: Score,
    document: HashMap<String, Vec<Value>>,
}
