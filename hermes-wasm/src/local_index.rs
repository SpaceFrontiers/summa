//! In-browser index with create, index, search capabilities.
//!
//! Uses RamDirectory for in-memory storage. Documents are indexed via
//! the single-threaded WasmIndexWriter and searchable after commit().

use std::sync::Arc;

use hermes_core::directories::RamDirectory;
use hermes_core::dsl::Schema;
use hermes_core::{IndexConfig, IndexMetadata, Searcher, WasmIndexWriter};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// In-browser local index — create, index documents, search, all in WASM.
///
/// ```js
/// const index = await LocalIndex.create("index articles { field title: text<en_stem> [indexed, stored] }");
/// index.addDocument({ title: "Hello World" });
/// await index.commit();
/// const results = await index.search("hello", 10);
/// ```
#[wasm_bindgen]
pub struct LocalIndex {
    writer: Option<WasmIndexWriter<RamDirectory>>,
    searcher: Option<Searcher<RamDirectory>>,
    directory: Arc<RamDirectory>,
    schema: Arc<Schema>,
}

#[wasm_bindgen]
impl LocalIndex {
    /// Create a new in-memory index from an SDL schema string.
    ///
    /// Example SDL: `"index articles { field title: text<en_stem> [indexed, stored] }"`
    #[wasm_bindgen]
    pub async fn create(schema_sdl: String) -> Result<LocalIndex, JsValue> {
        let index_def = hermes_core::parse_single_index(&schema_sdl)
            .map_err(|e| JsValue::from_str(&format!("Schema parse error: {}", e)))?;

        let schema = index_def.to_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig {
            max_indexing_memory_bytes: 32 * 1024 * 1024, // 32 MB for browser
            ..IndexConfig::default()
        };

        let writer = WasmIndexWriter::create(dir.clone(), schema.clone(), config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Create error: {}", e)))?;

        Ok(LocalIndex {
            schema: Arc::new(schema),
            directory: Arc::new(dir),
            writer: Some(writer),
            searcher: None,
        })
    }

    /// Add a single document (JSON object with field names matching the schema).
    #[wasm_bindgen(js_name = "addDocument")]
    pub async fn add_document(&mut self, doc_json: JsValue) -> Result<(), JsValue> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Index not writable"))?;

        let json_value: serde_json::Value = serde_wasm_bindgen::from_value(doc_json)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

        let doc = json_to_document(&json_value, &self.schema)
            .map_err(|e| JsValue::from_str(&format!("Document error: {}", e)))?;

        writer
            .add_document(doc)
            .await
            .map_err(|e| JsValue::from_str(&format!("Index error: {}", e)))
    }

    /// Add multiple documents at once (array of JSON objects).
    #[wasm_bindgen(js_name = "addDocuments")]
    pub async fn add_documents(&mut self, docs_json: JsValue) -> Result<u32, JsValue> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Index not writable"))?;

        let json_array: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(docs_json)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

        let mut count = 0u32;
        for json_value in &json_array {
            let doc = json_to_document(json_value, &self.schema)
                .map_err(|e| JsValue::from_str(&format!("Document error: {}", e)))?;
            writer
                .add_document(doc)
                .await
                .map_err(|e| JsValue::from_str(&format!("Index error: {}", e)))?;
            count += 1;
        }

        Ok(count)
    }

    /// Commit pending documents — builds segments and updates metadata.
    /// Must be called before search() will return the new documents.
    #[wasm_bindgen]
    pub async fn commit(&mut self) -> Result<bool, JsValue> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Index not writable"))?;

        let committed = writer
            .commit()
            .await
            .map_err(|e| JsValue::from_str(&format!("Commit error: {}", e)))?;

        if committed {
            // Refresh searcher to pick up new segments
            self.refresh_searcher().await?;
        }

        Ok(committed)
    }

    /// Search the index.
    ///
    /// Returns `{ hits: [{ address: { segment_id, doc_id }, score }], total_hits }`.
    #[wasm_bindgen]
    pub async fn search(&self, query_str: String, limit: usize) -> Result<JsValue, JsValue> {
        self.search_offset(query_str, limit, 0).await
    }

    /// Search with offset for pagination.
    #[wasm_bindgen(js_name = "searchOffset")]
    pub async fn search_offset(
        &self,
        query_str: String,
        limit: usize,
        offset: usize,
    ) -> Result<JsValue, JsValue> {
        let searcher = self
            .searcher
            .as_ref()
            .ok_or_else(|| JsValue::from_str("No committed data — call commit() first"))?;

        let response = searcher
            .query_offset(&query_str, limit, offset)
            .await
            .map_err(|e| JsValue::from_str(&format!("Search error: {}", e)))?;

        response
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Get a document by its address.
    #[wasm_bindgen(js_name = "getDocument")]
    pub async fn get_document(&self, segment_id: String, doc_id: u32) -> Result<JsValue, JsValue> {
        let searcher = self
            .searcher
            .as_ref()
            .ok_or_else(|| JsValue::from_str("No committed data"))?;

        let segment_id_u128 = u128::from_str_radix(&segment_id, 16)
            .map_err(|e| JsValue::from_str(&format!("Invalid segment_id: {}", e)))?;
        let address = hermes_core::query::DocAddress::new(segment_id_u128, doc_id);

        let doc = searcher
            .get_document(&address)
            .await
            .map_err(|e| JsValue::from_str(&format!("Get document error: {}", e)))?;

        match doc {
            Some(document) => {
                let json = document.to_json(searcher.schema());
                json.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Number of indexed documents (across all committed segments).
    #[wasm_bindgen(js_name = "numDocs")]
    pub fn num_docs(&self) -> u32 {
        self.searcher.as_ref().map(|s| s.num_docs()).unwrap_or(0)
    }

    /// Number of documents pending (not yet committed).
    #[wasm_bindgen(js_name = "pendingDocs")]
    pub fn pending_docs(&self) -> u32 {
        self.writer.as_ref().map(|w| w.pending_docs()).unwrap_or(0)
    }

    /// Get field names from the schema.
    #[wasm_bindgen(js_name = "fieldNames")]
    pub fn field_names(&self) -> JsValue {
        let names: Vec<String> = self.schema.fields().map(|(_, f)| f.name.clone()).collect();
        serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
    }

    /// Refresh the internal searcher to reflect committed segments.
    async fn refresh_searcher(&mut self) -> Result<(), JsValue> {
        let metadata = IndexMetadata::load(self.directory.as_ref())
            .await
            .map_err(|e| JsValue::from_str(&format!("Metadata load error: {}", e)))?;

        let segment_ids = metadata.segment_ids();

        let searcher = Searcher::open(
            Arc::clone(&self.directory),
            Arc::clone(&self.schema),
            &segment_ids,
            32,
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("Searcher open error: {}", e)))?;

        self.searcher = Some(searcher);
        Ok(())
    }
}

/// Convert a JSON value to a hermes Document using the schema.
fn json_to_document(
    json: &serde_json::Value,
    schema: &Schema,
) -> std::result::Result<hermes_core::Document, String> {
    let obj = json.as_object().ok_or("Document must be a JSON object")?;

    let mut doc = hermes_core::Document::new();

    for (key, value) in obj {
        let field = schema
            .get_field(key)
            .ok_or_else(|| format!("Unknown field: {}", key))?;
        let entry = schema
            .get_field_entry(field)
            .ok_or_else(|| format!("No field entry for: {}", key))?;

        match entry.field_type {
            hermes_core::FieldType::Text => {
                if let Some(text) = value.as_str() {
                    doc.add_text(field, text);
                } else if let Some(arr) = value.as_array() {
                    for v in arr {
                        if let Some(text) = v.as_str() {
                            doc.add_text(field, text);
                        }
                    }
                }
            }
            hermes_core::FieldType::U64 => {
                if let Some(n) = value.as_u64() {
                    doc.add_u64(field, n);
                }
            }
            hermes_core::FieldType::I64 => {
                if let Some(n) = value.as_i64() {
                    doc.add_i64(field, n);
                }
            }
            hermes_core::FieldType::F64 => {
                if let Some(n) = value.as_f64() {
                    doc.add_f64(field, n);
                }
            }
            _ => {} // Skip unsupported types for now
        }
    }

    Ok(doc)
}
