//! HTTP-based remote index with slice caching

use hermes_core::directories::SliceCachingDirectory;
use hermes_core::{HttpDirectory, Index, IndexConfig, SLICE_CACHE_FILENAME};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::idb::{cache_key, idb_delete, idb_get, idb_put};
use crate::{DEFAULT_CACHE_SIZE, fetch_bytes};

/// Type alias for our cached HTTP directory
pub type CachedHttpDirectory = SliceCachingDirectory<HttpDirectory>;

/// Remote index that loads data via HTTP with slice caching
#[wasm_bindgen]
pub struct RemoteIndex {
    base_url: String,
    cache_size: usize,
    index: Option<Index<CachedHttpDirectory>>,
}

#[wasm_bindgen]
impl RemoteIndex {
    /// Create a new remote index pointing to a URL
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            cache_size: DEFAULT_CACHE_SIZE,
            index: None,
        }
    }

    /// Create with custom cache size (in bytes)
    #[wasm_bindgen]
    pub fn with_cache_size(base_url: String, cache_size: usize) -> Self {
        Self {
            base_url,
            cache_size,
            index: None,
        }
    }

    /// Load index from URL using hermes-core Index with slice caching
    ///
    /// Automatically attempts to load the slice cache file (index.slicecache)
    /// to prefill the cache with hot data, reducing cold-start latency.
    #[wasm_bindgen]
    pub async fn load(&mut self) -> Result<(), JsValue> {
        // Create HTTP directory and wrap with slice caching
        let http_dir = HttpDirectory::new(&self.base_url);
        let cached_dir = SliceCachingDirectory::new(http_dir, self.cache_size);

        // Try to load slice cache file to prefill the cache
        let cache_url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            SLICE_CACHE_FILENAME
        );
        if let Ok(cache_data) = fetch_bytes(&cache_url).await {
            let _ = cached_dir.deserialize(&cache_data);
        }

        let config = IndexConfig::default();

        let index = Index::open(cached_dir, config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to open index: {}", e)))?;

        self.index = Some(index);
        Ok(())
    }

    /// Get cache statistics
    #[wasm_bindgen]
    pub fn cache_stats(&self) -> JsValue {
        #[derive(Serialize)]
        struct CacheStatsJs {
            total_bytes: usize,
            max_bytes: usize,
            total_slices: usize,
            files_cached: usize,
        }

        if let Some(index) = &self.index {
            let stats = index.directory().stats();
            let js_stats = CacheStatsJs {
                total_bytes: stats.total_bytes,
                max_bytes: stats.max_bytes,
                total_slices: stats.total_slices,
                files_cached: stats.files_cached,
            };
            serde_wasm_bindgen::to_value(&js_stats).unwrap_or(JsValue::NULL)
        } else {
            JsValue::NULL
        }
    }

    /// Get network statistics (requests made, bytes transferred, etc.)
    #[wasm_bindgen]
    pub fn network_stats(&self) -> JsValue {
        if let Some(index) = &self.index {
            let http_stats = index.directory().inner().http_stats();
            serde_wasm_bindgen::to_value(&http_stats).unwrap_or(JsValue::NULL)
        } else {
            JsValue::NULL
        }
    }

    /// Reset network statistics
    #[wasm_bindgen]
    pub fn reset_network_stats(&self) {
        if let Some(index) = &self.index {
            index.directory().inner().reset_stats();
        }
    }

    /// Get number of documents
    #[wasm_bindgen]
    pub fn num_docs(&self) -> u32 {
        self.index.as_ref().map(|i| i.num_docs()).unwrap_or(0)
    }

    /// Get number of segments
    #[wasm_bindgen]
    pub fn num_segments(&self) -> usize {
        self.index
            .as_ref()
            .map(|i| i.segment_readers().len())
            .unwrap_or(0)
    }

    /// Get field names
    #[wasm_bindgen]
    pub fn field_names(&self) -> JsValue {
        let names: Vec<String> = self
            .index
            .as_ref()
            .map(|i| i.schema().fields().map(|(_, f)| f.name.clone()).collect())
            .unwrap_or_default();
        serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
    }

    /// Search the index
    ///
    /// Accepts both query language syntax (field:term, AND, OR, NOT, grouping)
    /// and simple text (tokenized and searched across default fields).
    /// Returns document addresses (segment_id + doc_id) without document content.
    #[wasm_bindgen]
    pub async fn search(&self, query_str: String, limit: usize) -> Result<JsValue, JsValue> {
        web_sys::console::log_1(&format!("=== SEARCH START: '{}' ===", query_str).into());

        let index = self
            .index
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        // Log segment info
        for (i, seg) in index.segment_readers().iter().enumerate() {
            let stats = seg.term_dict_stats();
            web_sys::console::log_1(
                &format!(
                    "  Segment {}: {} blocks, {} sparse entries, {} terms",
                    i, stats.num_blocks, stats.num_sparse_entries, stats.num_entries
                )
                .into(),
            );
        }

        // Reset network stats before search to see only this query's I/O
        index.directory().inner().reset_stats();

        let response = index
            .query(&query_str, limit)
            .await
            .map_err(|e| JsValue::from_str(&format!("Search error: {}", e)))?;

        // Log network stats after search
        let stats = index.directory().inner().http_stats();
        web_sys::console::log_1(
            &format!(
                "=== SEARCH END: {} requests, {} bytes ===",
                stats.total_requests, stats.total_bytes
            )
            .into(),
        );
        for op in &stats.operations {
            web_sys::console::log_1(
                &format!(
                    "  HTTP: {} bytes, {}ms, range={:?}, url={}",
                    op.bytes, op.duration_ms, op.range, op.url
                )
                .into(),
            );
        }

        // Use json_compatible serializer for proper plain object output
        response
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Get a document by its address (segment_id + doc_id)
    ///
    /// Returns the document as a JSON object, or null if not found.
    #[wasm_bindgen]
    pub async fn get_document(&self, segment_id: String, doc_id: u32) -> Result<JsValue, JsValue> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        let address = hermes_core::query::DocAddress { segment_id, doc_id };

        let doc = index
            .get_document(&address)
            .await
            .map_err(|e| JsValue::from_str(&format!("Get document error: {}", e)))?;

        match doc {
            Some(document) => {
                let json = document.to_json(index.schema());
                json.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Get default field names for query parsing
    #[wasm_bindgen]
    pub fn default_fields(&self) -> JsValue {
        let names: Vec<String> = self
            .index
            .as_ref()
            .map(|i| {
                i.default_fields()
                    .iter()
                    .filter_map(|f| i.schema().get_field_name(*f).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
    }

    /// Export the current slice cache as bytes
    ///
    /// Returns the serialized cache data that can be stored in IndexedDB
    /// or other persistent storage for later restoration.
    #[wasm_bindgen]
    pub fn export_cache(&self) -> Option<Vec<u8>> {
        self.index.as_ref().map(|i| i.directory().serialize())
    }

    /// Import a previously exported slice cache
    ///
    /// Merges the cached slices into the current cache, reducing
    /// network requests for previously fetched data.
    #[wasm_bindgen]
    pub fn import_cache(&self, data: &[u8]) -> Result<(), JsValue> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        index
            .directory()
            .deserialize(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to import cache: {}", e)))
    }

    /// Save the slice cache to IndexedDB for persistence across sessions
    ///
    /// The cache is stored under a key derived from the base URL.
    #[wasm_bindgen]
    pub async fn save_cache_to_idb(&self) -> Result<(), JsValue> {
        let cache_data = self
            .export_cache()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        let key = cache_key(&self.base_url);
        idb_put(&key, &cache_data).await
    }

    /// Load the slice cache from IndexedDB
    ///
    /// Call this after load() to restore cached data from a previous session.
    #[wasm_bindgen]
    pub async fn load_cache_from_idb(&self) -> Result<bool, JsValue> {
        let key = cache_key(&self.base_url);

        match idb_get(&key).await? {
            Some(data) => {
                self.import_cache(&data)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Clear the persisted cache from IndexedDB
    #[wasm_bindgen]
    pub async fn clear_idb_cache(&self) -> Result<(), JsValue> {
        let key = cache_key(&self.base_url);
        idb_delete(&key).await
    }
}
