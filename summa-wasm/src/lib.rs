mod network_directory;
mod request_generator;

use summa_core::errors::SummaResult;
use crate::request_generator::RequestGenerator;
use js_sys::Uint8Array;
use network_directory::NetworkDirectory;
use serde::Deserialize;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use summa_core::proto;
use summa_core::QueryParser;
use summa_core::default_tokenizers::default_tokenizers;
use summa_directory::caching_directory::CachingDirectory;
use summa_directory::hot_cache_directory::HotDirectory;
use tantivy::directory::OwnedBytes;
use tantivy::schema::Value;
use summa_core::fruit_extractors::{build_fruit_extractor, FruitExtractor};
use tantivy::{Index, Score, IndexReader};
use tantivy::collector::MultiCollector;
use tantivy::schema::Field;
use wasm_bindgen::prelude::*;
use web_sys::console::log_1;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct FileSize {
    name: String,
    size: usize,
}

#[wasm_bindgen(raw_module = "./gate")]
extern "C" {
    #[wasm_bindgen]
    pub fn request(method: String, url: String, headers: JsValue) -> Uint8Array;
}

#[wasm_bindgen]
pub struct SummaIndexWrapper {
    index: Index,
    index_reader: IndexReader,
    query_parser: summa_core::QueryParser,
    multi_fields: HashSet<Field>,
}

#[wasm_bindgen]
impl SummaIndexWrapper {
    #[wasm_bindgen(constructor)]
    pub fn new(method: String, url_template: String, headers_template: JsValue, files: JsValue, default_fields: &str, multi_fields: &str) -> Result<SummaIndexWrapper, JsValue> {
        let headers_template: Option<HashMap<String, String>> = serde_wasm_bindgen::from_value(headers_template).map_err(to_js_err)?;
        let file_sizes: Vec<FileSize> = serde_wasm_bindgen::from_value(files).map_err(to_js_err)?;
        let file_sizes = HashMap::from_iter(file_sizes.into_iter().map(|file_size| (file_size.name, file_size.size)));

        let request_generator = RequestGenerator::new(method, url_template, headers_template);

        let network_directory = Arc::new(NetworkDirectory::new(request_generator.clone(), file_sizes.clone()));
        let caching_directory = CachingDirectory::new_with_capacity_in_bytes(network_directory, 1024 * 1024 * 1024);

        let index = match file_sizes.get("hotcache.bin") {
            None => Index::open(caching_directory).map_err(to_js_err),
            Some(hotcache_bytes_size) => {
                let request = request_generator.generate(&"hotcache.bin", 0..*hotcache_bytes_size);
                let hotcache_bytes = request.send();
                let hotcache_directory = HotDirectory::open(caching_directory, OwnedBytes::new(hotcache_bytes.to_vec())).map_err(to_js_err)?;
                Index::open(hotcache_directory).map_err(to_js_err)
            }
        }?;

        for (tokenizer_name, tokenizer) in &default_tokenizers() {
            index.tokenizers().register(tokenizer_name, tokenizer.clone())
        }

        let schema = index.schema();
        let default_fields = default_fields
            .split(",")
            .map(|field_name| {
                schema
                    .get_field(field_name)
                    .ok_or_else(|| JsValue::from(format!("no default field {}", field_name)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let multi_fields = multi_fields
            .split(",")
            .map(|field_name| {
                schema
                    .get_field(field_name)
                    .ok_or_else(|| JsValue::from(format!("no default field {}", field_name)))
            })
            .collect::<Result<HashSet<_>, _>>()?;
        let query_parser = QueryParser::for_index(&index, default_fields);
        let index_reader = index.reader().map_err(to_js_err)?;
        Ok(SummaIndexWrapper { index, index_reader, query_parser, multi_fields })
    }

    #[wasm_bindgen]
    pub fn search(&self, index_name: String, query: JsValue, collectors: JsValue) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        log_1(&format!("rq {:?}", query).into());
        log_1(&format!("rc {:?}", collectors).into());
        let query = serde_wasm_bindgen::from_value(query).map_err(to_js_err)?;
        log_1(&format!("q {:?}", query).into());
        let collectors: Vec<_> = serde_wasm_bindgen::from_value(collectors).map_err(to_js_err)?;
        log_1(&format!("c {:?}", collectors).into());
        self.search_inner(&index_name, query, collectors).map_err(to_js_err)
    }

    fn search_inner(
        &self,
        external_index_alias: &str,
        query: proto::Query,
        collectors: Vec<proto::Collector>,
    ) -> SummaResult<JsValue> {
        let searcher = self.index_reader.searcher();
        let parsed_query = self.query_parser.parse_query(&query)?;
        let schema = self.index.schema();
        let mut multi_collector = MultiCollector::new();
        let mut extractors: Vec<Box<dyn FruitExtractor>> = collectors
            .into_iter()
            .map(|collector_proto| build_fruit_extractor(collector_proto, &parsed_query, &schema, &mut multi_collector))
            .collect::<SummaResult<_>>()?;
        let external_index_alias = external_index_alias.to_string();
        let mut multi_fruit = searcher.search(&parsed_query, &multi_collector)?;
        let collector_outputs: Vec<proto::CollectorOutput> = extractors
            .drain(..)
            .map(|e| e.extract(&external_index_alias, &mut multi_fruit, &searcher, &self.multi_fields))
            .collect();
        Ok(serde_wasm_bindgen::to_value(&collector_outputs).unwrap())
    }
}

fn to_js_err(e: impl std::fmt::Debug) -> JsValue {
    return JsValue::from(format!("{:?}", e));
}

#[derive(Serialize)]
struct ScoredDocument {
    score: Score,
    document: HashMap<String, Vec<Value>>,
}
