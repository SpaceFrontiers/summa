//! In-browser index with create, index, search capabilities.
//!
//! Supports two storage modes:
//! - **In-memory** (`LocalIndex.create`) — fast, lost on page refresh
//! - **Persistent** (`LocalIndex.createPersistent` / `LocalIndex.open`) — backed by IndexedDB
//!
//! Persistent storage uses per-file IDB records: each segment file gets its
//! own key, so commits only write new/changed files (not the whole index).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use hermes_core::directories::RamDirectory;
use hermes_core::{IndexConfig, Searcher, WasmIndexWriter};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::idb;

/// Default term cache blocks for WASM searcher.
const WASM_TERM_CACHE_BLOCKS: usize = 32;

/// IDB object store for persisted index files.
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
    /// File paths already in IDB (for incremental sync)
    persisted_files: HashSet<String>,
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
    /// Each file in the index gets its own IDB record — commits only write
    /// new/changed files, not the entire index.
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
    /// Loads the file manifest and all index files into a RamDirectory.
    #[wasm_bindgen]
    pub async fn open(name: String) -> Result<LocalIndex, JsValue> {
        let manifest_key = idb_key(&name, "__manifest");
        let manifest_data = idb::idb_get_from_store(IDB_INDEX_STORE, &manifest_key)
            .await?
            .ok_or_else(|| {
                JsValue::from_str(&format!("Index '{}' not found in IndexedDB", name))
            })?;

        let manifest_str = String::from_utf8(manifest_data)
            .map_err(|e| JsValue::from_str(&format!("Manifest decode error: {}", e)))?;
        let file_paths: Vec<&str> = manifest_str.lines().filter(|l| !l.is_empty()).collect();

        // Load each file from IDB into RamDirectory
        let dir = RamDirectory::new();
        for path_str in &file_paths {
            let key = idb_key(&name, path_str);
            if let Some(data) = idb::idb_get_from_store(IDB_INDEX_STORE, &key).await? {
                dir.write_sync(Path::new(path_str), &data)
                    .map_err(|e| JsValue::from_str(&format!("Write error: {}", e)))?;
            }
        }

        let config = IndexConfig {
            max_indexing_memory_bytes: 32 * 1024 * 1024,
            ..IndexConfig::default()
        };

        let writer = WasmIndexWriter::open(dir.clone(), config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Open error: {}", e)))?;

        let persisted_files: HashSet<String> = file_paths.iter().map(|s| s.to_string()).collect();

        let mut index = LocalIndex {
            writer: Some(writer),
            searcher: None,
            persist_name: Some(name),
            persisted_files,
        };
        index.refresh_searcher().await?;
        Ok(index)
    }

    /// Delete a persistent index from IndexedDB.
    #[wasm_bindgen(js_name = "deleteIndex")]
    pub async fn delete_index(name: String) -> Result<(), JsValue> {
        // Load manifest to find all file keys
        let manifest_key = idb_key(&name, "__manifest");
        if let Some(manifest_data) = idb::idb_get_from_store(IDB_INDEX_STORE, &manifest_key).await?
        {
            if let Ok(manifest_str) = String::from_utf8(manifest_data) {
                for path_str in manifest_str.lines().filter(|l| !l.is_empty()) {
                    let key = idb_key(&name, path_str);
                    idb::idb_delete_from_store(IDB_INDEX_STORE, &key).await?;
                }
            }
        }
        idb::idb_delete_from_store(IDB_INDEX_STORE, &manifest_key).await
    }

    /// Check if a persistent index exists in IndexedDB.
    #[wasm_bindgen]
    pub async fn exists(name: String) -> Result<bool, JsValue> {
        let manifest_key = idb_key(&name, "__manifest");
        let data = idb::idb_get_from_store(IDB_INDEX_STORE, &manifest_key).await?;
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
    /// For persistent indexes, only writes new/changed files to IndexedDB.
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

            // Sync only new/changed files to IDB
            if self.persist_name.is_some() {
                self.sync_to_idb().await?;
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
            persisted_files: HashSet::new(),
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

    /// Sync new/changed files to IDB. Each file gets its own IDB record
    /// keyed by `index:{name}:{path}`. Only files not in `persisted_files`
    /// are written (plus metadata.json which changes every commit).
    async fn sync_to_idb(&mut self) -> Result<(), JsValue> {
        let writer = self.writer.as_ref().unwrap();
        let name = self.persist_name.as_ref().unwrap();
        let dir = writer.directory();

        let current_files = dir
            .list_files_sync(Path::new(""))
            .map_err(|e| JsValue::from_str(&format!("List files error: {}", e)))?;

        let current_set: HashSet<String> = current_files
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        // Write new files (not yet in IDB) and metadata.json (always changes)
        for path in &current_files {
            let path_str = path.to_string_lossy();
            if !self.persisted_files.contains(path_str.as_ref()) || path_str == "metadata.json" {
                let data = dir
                    .read_file_sync(path)
                    .map_err(|e| JsValue::from_str(&format!("Read error: {}", e)))?;
                let key = idb_key(name, &path_str);
                idb::idb_put_to_store(IDB_INDEX_STORE, &key, &data).await?;
            }
        }

        // Delete removed files from IDB (e.g. after merge)
        for old_path in &self.persisted_files {
            if !current_set.contains(old_path) {
                let key = idb_key(name, old_path);
                idb::idb_delete_from_store(IDB_INDEX_STORE, &key).await?;
            }
        }

        // Update manifest
        let manifest = current_set.iter().cloned().collect::<Vec<_>>().join("\n");
        let manifest_key = idb_key(name, "__manifest");
        idb::idb_put_to_store(IDB_INDEX_STORE, &manifest_key, manifest.as_bytes()).await?;

        self.persisted_files = current_set;
        Ok(())
    }
}

/// Build an IDB key for a named index file.
fn idb_key(index_name: &str, file_path: &str) -> String {
    format!("index:{}:{}", index_name, file_path)
}
