//! In-browser index with create, index, search capabilities.
//!
//! Supports two modes:
//! - **In-memory** (`LocalIndex.create`) — fast, lost on page refresh
//! - **Persistent** (`LocalIndex.withStorage`) — pluggable storage backend
//!
//! Storage implements a simple JS interface:
//! ```ts
//! interface IFilesStorage {
//!     write(name: string, buffer: ArrayBuffer): Promise<void>;
//!     get(name: string): Promise<ArrayBuffer | null>;
//!     delete(names: string[]): Promise<void>;
//!     list(): Promise<string[]>;
//! }
//! ```

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use hermes_core::directories::RamDirectory;
use hermes_core::{IndexConfig, Searcher, WasmIndexWriter};
use serde::Serialize;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Default term cache blocks for WASM searcher.
const WASM_TERM_CACHE_BLOCKS: usize = 32;

/// Wraps a JS object implementing `IFilesStorage`.
struct JsStorageAdapter {
    write_fn: js_sys::Function,
    get_fn: js_sys::Function,
    delete_fn: js_sys::Function,
    list_fn: js_sys::Function,
}

impl JsStorageAdapter {
    fn from_js(storage: &JsValue) -> Result<Self, JsValue> {
        let write_fn = js_sys::Reflect::get(storage, &"write".into())?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| JsValue::from_str("storage.write must be a function"))?;
        let get_fn = js_sys::Reflect::get(storage, &"get".into())?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| JsValue::from_str("storage.get must be a function"))?;
        let delete_fn = js_sys::Reflect::get(storage, &"delete".into())?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| JsValue::from_str("storage.delete must be a function"))?;
        let list_fn = js_sys::Reflect::get(storage, &"list".into())?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| JsValue::from_str("storage.list must be a function"))?;
        Ok(Self {
            write_fn,
            get_fn,
            delete_fn,
            list_fn,
        })
    }

    async fn write(&self, name: &str, data: &[u8]) -> Result<(), JsValue> {
        let this = JsValue::NULL;
        let js_name = JsValue::from_str(name);
        let js_buffer = js_sys::Uint8Array::from(data).buffer();
        let promise = self.write_fn.call2(&this, &js_name, &js_buffer)?;
        JsFuture::from(js_sys::Promise::from(promise)).await?;
        Ok(())
    }

    async fn get(&self, name: &str) -> Result<Option<Vec<u8>>, JsValue> {
        let this = JsValue::NULL;
        let js_name = JsValue::from_str(name);
        let promise = self.get_fn.call1(&this, &js_name)?;
        let result = JsFuture::from(js_sys::Promise::from(promise)).await?;
        if result.is_null() || result.is_undefined() {
            return Ok(None);
        }
        let array = js_sys::Uint8Array::new(&result);
        Ok(Some(array.to_vec()))
    }

    async fn delete(&self, names: &[String]) -> Result<(), JsValue> {
        let this = JsValue::NULL;
        let js_array = js_sys::Array::new();
        for name in names {
            js_array.push(&JsValue::from_str(name));
        }
        let promise = self.delete_fn.call1(&this, &js_array)?;
        JsFuture::from(js_sys::Promise::from(promise)).await?;
        Ok(())
    }

    async fn list(&self) -> Result<Vec<String>, JsValue> {
        let this = JsValue::NULL;
        let promise = self.list_fn.call0(&this)?;
        let result = JsFuture::from(js_sys::Promise::from(promise)).await?;
        let array: js_sys::Array = result
            .dyn_into()
            .map_err(|_| JsValue::from_str("storage.list() must return an array of strings"))?;
        let mut files = Vec::with_capacity(array.length() as usize);
        for i in 0..array.length() {
            if let Some(s) = array.get(i).as_string() {
                files.push(s);
            }
        }
        Ok(files)
    }
}

/// In-browser local index — create, index documents, search, all in WASM.
///
/// ```js
/// // In-memory (ephemeral)
/// const index = await LocalIndex.create("index articles { ... }");
///
/// // With pluggable storage (IDB, encrypted, remote, etc.)
/// const index = await LocalIndex.withStorage(myStorage, "index articles { ... }");
/// ```
#[wasm_bindgen]
pub struct LocalIndex {
    writer: Option<WasmIndexWriter<RamDirectory>>,
    searcher: Option<Searcher<RamDirectory>>,
    /// JS storage adapter (None = in-memory only)
    storage: Option<JsStorageAdapter>,
    /// File paths already persisted (for incremental sync)
    persisted_files: HashSet<String>,
}

#[wasm_bindgen]
impl LocalIndex {
    /// Create a new in-memory index from an SDL schema string.
    ///
    /// Data is lost on page refresh. Use `withStorage()` for persistence.
    #[wasm_bindgen]
    pub async fn create(schema_sdl: String) -> Result<LocalIndex, JsValue> {
        let writer = Self::make_writer(&schema_sdl).await?;
        Ok(LocalIndex {
            writer: Some(writer),
            searcher: None,
            storage: None,
            persisted_files: HashSet::new(),
        })
    }

    /// Create or open an index with a pluggable storage backend.
    ///
    /// If the storage already contains index files, the index is reopened.
    /// Otherwise a new index is created from the SDL schema.
    ///
    /// The storage object must implement:
    /// ```ts
    /// interface IFilesStorage {
    ///     write(name: string, buffer: ArrayBuffer): Promise<void>;
    ///     get(name: string): Promise<ArrayBuffer | null>;
    ///     delete(names: string[]): Promise<void>;
    ///     list(): Promise<string[]>;
    /// }
    /// ```
    #[wasm_bindgen(js_name = "withStorage")]
    pub async fn with_storage(storage: JsValue, schema_sdl: String) -> Result<LocalIndex, JsValue> {
        let adapter = JsStorageAdapter::from_js(&storage)?;
        let existing_files = adapter.list().await?;

        if existing_files.is_empty() {
            // New index
            let writer = Self::make_writer(&schema_sdl).await?;
            Ok(LocalIndex {
                writer: Some(writer),
                searcher: None,
                storage: Some(adapter),
                persisted_files: HashSet::new(),
            })
        } else {
            // Reopen from storage
            let dir = RamDirectory::new();
            for file_name in &existing_files {
                if let Some(data) = adapter.get(file_name).await? {
                    dir.write_sync(Path::new(file_name), &data)
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

            let persisted_files: HashSet<String> = existing_files.into_iter().collect();

            let mut index = LocalIndex {
                writer: Some(writer),
                searcher: None,
                storage: Some(adapter),
                persisted_files,
            };
            index.refresh_searcher().await?;
            Ok(index)
        }
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
    /// For persistent indexes, only writes new/changed files to storage.
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
            self.sync_to_storage().await?;
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

    async fn make_writer(schema_sdl: &str) -> Result<WasmIndexWriter<RamDirectory>, JsValue> {
        let index_def = hermes_core::parse_single_index(schema_sdl)
            .map_err(|e| JsValue::from_str(&format!("Schema parse error: {}", e)))?;
        let schema = index_def.to_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig {
            max_indexing_memory_bytes: 32 * 1024 * 1024,
            ..IndexConfig::default()
        };
        WasmIndexWriter::create(dir, schema, config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Create error: {}", e)))
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

    /// Sync new/changed files to storage (if any).
    async fn sync_to_storage(&mut self) -> Result<(), JsValue> {
        let adapter = match &self.storage {
            Some(a) => a,
            None => return Ok(()),
        };

        let writer = self.writer.as_ref().unwrap();
        let dir = writer.directory();

        let current_files = dir
            .list_files_sync(Path::new(""))
            .map_err(|e| JsValue::from_str(&format!("List files error: {}", e)))?;

        let current_set: HashSet<String> = current_files
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        // Write new files and metadata.json (always changes on commit)
        for path in &current_files {
            let path_str = path.to_string_lossy();
            if !self.persisted_files.contains(path_str.as_ref()) || path_str == "metadata.json" {
                let data = dir
                    .read_file_sync(path)
                    .map_err(|e| JsValue::from_str(&format!("Read error: {}", e)))?;
                adapter.write(&path_str, &data).await?;
            }
        }

        // Delete removed files (e.g. after merge)
        let removed: Vec<String> = self
            .persisted_files
            .iter()
            .filter(|p| !current_set.contains(p.as_str()))
            .cloned()
            .collect();
        if !removed.is_empty() {
            adapter.delete(&removed).await?;
        }

        self.persisted_files = current_set;
        Ok(())
    }
}
