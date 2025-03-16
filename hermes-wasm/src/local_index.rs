//! In-browser index with create, index, search capabilities.
//!
//! Supports two storage modes:
//! - **In-memory** (`LocalIndex.create`) — fast, lost on page refresh
//! - **Persistent** (`LocalIndex.createPersistent` / `LocalIndex.open`) — backed by IndexedDB

use std::sync::Arc;

use hermes_core::directories::{Directory, RamDirectory};
use hermes_core::{IndexConfig, Searcher, WasmIndexWriter};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::idb;

/// Default term cache blocks for WASM searcher.
const WASM_TERM_CACHE_BLOCKS: usize = 32;

/// IDB object store for persisted indexes.
const IDB_INDEX_STORE: &str = "indexes";

/// In-browser local index — create, index documents, search, all in WASM.
///
/// ```js
/// // In-memory (ephemeral)
/// const index = await LocalIndex.create("index articles { field title: text<en_stem> [indexed, stored] }");
///
/// // Persistent (IndexedDB-backed)
/// const index = await LocalIndex.createPersistent("my-index", "index articles { ... }");
/// // ... later, on page reload:
/// const index = await LocalIndex.open("my-index");
/// ```
#[wasm_bindgen]
pub struct LocalIndex {
    writer: Option<WasmIndexWriter<RamDirectory>>,
    searcher: Option<Searcher<RamDirectory>>,
    /// Name for IDB persistence (None = in-memory only)
    persist_name: Option<String>,
}

#[wasm_bindgen]
impl LocalIndex {
    /// Create a new in-memory index from an SDL schema string.
    ///
    /// Data is lost on page refresh. Use `createPersistent()` for IndexedDB-backed storage.
    #[wasm_bindgen]
    pub async fn create(schema_sdl: String) -> Result<LocalIndex, JsValue> {
        Self::create_inner(schema_sdl, None).await
    }

    /// Create a new persistent index backed by IndexedDB.
    ///
    /// The index is automatically saved to IndexedDB on each `commit()`.
    /// Use `LocalIndex.open(name)` to reopen it after page refresh.
    #[wasm_bindgen(js_name = "createPersistent")]
    pub async fn create_persistent(
        name: String,
        schema_sdl: String,
    ) -> Result<LocalIndex, JsValue> {
        Self::create_inner(schema_sdl, Some(name)).await
    }

    /// Open an existing persistent index from IndexedDB.
    ///
    /// Returns an error if the index doesn't exist. Use `createPersistent()` to create one first.
    #[wasm_bindgen]
    pub async fn open(name: String) -> Result<LocalIndex, JsValue> {
        let idb_key = format!("index:{}", name);
        let data = idb::idb_get_from_store(IDB_INDEX_STORE, &idb_key)
            .await?
            .ok_or_else(|| {
                JsValue::from_str(&format!("Index '{}' not found in IndexedDB", name))
            })?;

        let dir = deserialize_ram_directory(&data)
            .map_err(|e| JsValue::from_str(&format!("Deserialize error: {}", e)))?;

        let config = IndexConfig {
            max_indexing_memory_bytes: 32 * 1024 * 1024,
            ..IndexConfig::default()
        };

        let writer = WasmIndexWriter::open(dir.clone(), config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Open error: {}", e)))?;

        let mut index = LocalIndex {
            writer: Some(writer),
            searcher: None,
            persist_name: Some(name),
        };
        index.refresh_searcher().await?;
        Ok(index)
    }

    /// Delete a persistent index from IndexedDB.
    #[wasm_bindgen(js_name = "deleteIndex")]
    pub async fn delete_index(name: String) -> Result<(), JsValue> {
        let idb_key = format!("index:{}", name);
        idb::idb_delete_from_store(IDB_INDEX_STORE, &idb_key).await
    }

    /// Check if a persistent index exists in IndexedDB.
    #[wasm_bindgen]
    pub async fn exists(name: String) -> Result<bool, JsValue> {
        let idb_key = format!("index:{}", name);
        let data = idb::idb_get_from_store(IDB_INDEX_STORE, &idb_key).await?;
        Ok(data.is_some())
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

        let doc = hermes_core::Document::from_json(&json_value, writer.schema())
            .ok_or_else(|| JsValue::from_str("Failed to parse document from JSON"))?;

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

        let schema = writer.schema().clone();
        let mut count = 0u32;
        for json_value in &json_array {
            let doc = hermes_core::Document::from_json(json_value, &schema)
                .ok_or_else(|| JsValue::from_str("Failed to parse document from JSON"))?;
            writer
                .add_document(doc)
                .await
                .map_err(|e| JsValue::from_str(&format!("Index error: {}", e)))?;
            count += 1;
        }

        Ok(count)
    }

    /// Commit pending documents — builds segments and updates metadata.
    ///
    /// For persistent indexes, also saves to IndexedDB automatically.
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
            self.refresh_searcher().await?;

            // Auto-save to IDB for persistent indexes
            if self.persist_name.is_some() {
                self.save_to_idb().await?;
            }
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
        let names: Vec<String> = self
            .writer
            .as_ref()
            .map(|w| w.schema().fields().map(|(_, f)| f.name.clone()).collect())
            .unwrap_or_default();
        serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
    }

    // ── Internal helpers ──

    async fn create_inner(
        schema_sdl: String,
        persist_name: Option<String>,
    ) -> Result<LocalIndex, JsValue> {
        let index_def = hermes_core::parse_single_index(&schema_sdl)
            .map_err(|e| JsValue::from_str(&format!("Schema parse error: {}", e)))?;

        let schema = index_def.to_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig {
            max_indexing_memory_bytes: 32 * 1024 * 1024,
            ..IndexConfig::default()
        };

        let writer = WasmIndexWriter::create(dir, schema, config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Create error: {}", e)))?;

        Ok(LocalIndex {
            writer: Some(writer),
            searcher: None,
            persist_name,
        })
    }

    async fn refresh_searcher(&mut self) -> Result<(), JsValue> {
        let writer = self
            .writer
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Index not writable"))?;

        let metadata = writer.metadata();
        let segment_ids = metadata.segment_ids();
        let schema = Arc::new(writer.schema().clone());

        let searcher = Searcher::open(
            writer.directory().clone(),
            schema,
            &segment_ids,
            WASM_TERM_CACHE_BLOCKS,
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("Searcher open error: {}", e)))?;

        self.searcher = Some(searcher);
        Ok(())
    }

    async fn save_to_idb(&self) -> Result<(), JsValue> {
        let writer = self.writer.as_ref().unwrap();
        let name = self.persist_name.as_ref().unwrap();
        let idb_key = format!("index:{}", name);

        let data = serialize_ram_directory(writer.directory())
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))?;

        idb::idb_put_to_store(IDB_INDEX_STORE, &idb_key, &data).await
    }
}

// ── RamDirectory serialization ──
// Simple format: [num_files: u32] [path_len: u32, path_bytes, data_len: u32, data_bytes] ...

fn serialize_ram_directory(dir: &RamDirectory) -> Result<Vec<u8>, String> {
    use std::path::Path;

    let files = dir
        .list_files_sync(Path::new(""))
        .map_err(|e| e.to_string())?;

    let mut buf = Vec::new();
    buf.extend_from_slice(&(files.len() as u32).to_le_bytes());

    for path in &files {
        let path_str = path.to_string_lossy();
        let path_bytes = path_str.as_bytes();
        buf.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(path_bytes);

        let data = dir.read_file_sync(path).map_err(|e| e.to_string())?;
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&data);
    }

    Ok(buf)
}

fn deserialize_ram_directory(data: &[u8]) -> Result<RamDirectory, String> {
    use std::path::PathBuf;

    let dir = RamDirectory::new();

    if data.len() < 4 {
        return Err("Data too short".into());
    }
    let num_files = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut offset = 4;

    for _ in 0..num_files {
        if offset + 4 > data.len() {
            return Err("Truncated path length".into());
        }
        let path_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if offset + path_len > data.len() {
            return Err("Truncated path".into());
        }
        let path_str =
            std::str::from_utf8(&data[offset..offset + path_len]).map_err(|e| e.to_string())?;
        offset += path_len;
        let path = PathBuf::from(path_str);

        if offset + 4 > data.len() {
            return Err("Truncated data length".into());
        }
        let data_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if offset + data_len > data.len() {
            return Err("Truncated file data".into());
        }
        let file_data = &data[offset..offset + data_len];
        offset += data_len;

        dir.write_sync(&path, file_data)
            .map_err(|e| e.to_string())?;
    }

    Ok(dir)
}
