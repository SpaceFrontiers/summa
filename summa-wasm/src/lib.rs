mod network_directory;
mod request_generator;

use crate::request_generator::RequestGenerator;
use js_sys::Uint8Array;
use network_directory::NetworkDirectory;
use serde::Deserialize;
use serde::Serialize;
use serde_wasm_bindgen::Serializer;
use std::collections::HashMap;
use std::sync::Arc;
use summa_directory::caching_directory::CachingDirectory;
use summa_directory::hot_cache_directory::HotDirectory;
use tantivy::directory::OwnedBytes;
use tantivy::schema::Value;
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer};
use tantivy::{collector::TopDocs, query::QueryParser, Document, Index, Score};
use wasm_bindgen::prelude::*;

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
pub struct WebIndex {
    index: Index,
}

#[wasm_bindgen]
impl WebIndex {
    #[wasm_bindgen(constructor)]
    pub fn new(method: String, url_template: String, headers_template: JsValue, files: JsValue) -> Result<WebIndex, JsValue> {
        let headers_template: Option<HashMap<String, String>> = serde_wasm_bindgen::from_value(headers_template).unwrap();
        let file_sizes: Vec<FileSize> = serde_wasm_bindgen::from_value(files).unwrap();
        let file_sizes = HashMap::from_iter(file_sizes.into_iter().map(|file_size| (file_size.name, file_size.size)));

        let request_generator = RequestGenerator::new(method, url_template, headers_template);
        let hotbytes_size = file_sizes.get("hotcache.bin").expect("no hotcache");
        let request = request_generator.generate(&"hotcache.bin", 0..*hotbytes_size);
        let hotcache_bytes = request.send();

        let network_directory = Arc::new(NetworkDirectory::new(request_generator, file_sizes));
        let caching_directory = CachingDirectory::new_with_capacity_in_bytes(network_directory, 1024 * 1024 * 1024);
        let hotcache_directory = HotDirectory::open(caching_directory, OwnedBytes::new(hotcache_bytes.to_vec())).unwrap();

        let index = Index::open(hotcache_directory).map_err(to_js_err)?;
        let default_tokenizer = TextAnalyzer::from(SimpleTokenizer).filter(RemoveLongFilter::limit(100)).filter(LowerCaser);
        index.tokenizers().register("summa", default_tokenizer);
        Ok(WebIndex { index })
    }

    #[wasm_bindgen]
    pub fn search(&self, query: String) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        self.search_inner(query).map_err(to_js_err)
    }

    fn search_inner(&self, query: String) -> tantivy::Result<JsValue> {
        let schema = self.index.schema();
        let default_fields = [schema.get_field("title").unwrap(), schema.get_field("description").unwrap()].to_vec();
        let mut query_parser = QueryParser::new(schema.clone(), default_fields, self.index.tokenizers().clone());

        let query = query_parser.parse_query(&query)?;
        let searcher = self.index.reader()?.searcher();

        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        let read_results = results.iter().map(|(_, a)| searcher.doc(*a).unwrap()).collect::<Vec<Document>>();

        let mut documents = Vec::new();
        for ((score, doc_address), doc) in results.iter().zip(read_results) {
            let serilalized_document = schema.to_named_doc(&doc);
            let document = ScoredDocument {
                score: *score,
                document: HashMap::from_iter(serilalized_document.0.into_iter()),
            };
            documents.push(document);
        }
        let serializer = Serializer::new().serialize_maps_as_objects(true);
        Ok(documents.serialize(&serializer).unwrap())
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
