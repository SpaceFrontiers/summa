//! IPFS-based index using JavaScript fetch callbacks

use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use hermes_core::directories::{
    Directory, FileSlice, LazyFileHandle, OwnedBytes, SliceCachingDirectory,
};
use hermes_core::{IndexMetadata, SLICE_CACHE_FILENAME, Searcher};
use parking_lot::RwLock;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::DEFAULT_CACHE_SIZE;
use crate::idb::{cache_key, idb_delete, idb_get, idb_put};

/// A single network request record
#[derive(Debug, Clone, Serialize)]
pub struct RequestRecord {
    pub path: String,
    pub bytes: u64,
    pub range_start: Option<u64>,
    pub range_end: Option<u64>,
}

/// Network statistics for IPFS fetching
#[derive(Debug, Default)]
pub struct IpfsNetworkStats {
    pub total_requests: AtomicU64,
    pub total_bytes: AtomicU64,
    pub requests: RwLock<Vec<RequestRecord>>,
}

impl IpfsNetworkStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, path: &str, bytes: u64, range: Option<Range<u64>>) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);

        let record = RequestRecord {
            path: path.to_string(),
            bytes,
            range_start: range.as_ref().map(|r| r.start),
            range_end: range.as_ref().map(|r| r.end),
        };
        self.requests.write().push(record);
    }

    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_bytes.store(0, Ordering::Relaxed);
        self.requests.write().clear();
    }

    pub fn get_requests(&self) -> Vec<RequestRecord> {
        self.requests.read().clone()
    }
}

/// A Directory implementation that calls back to JavaScript for fetching
///
/// This allows using custom fetch implementations like HTTP gateway fetching
/// for IPFS content retrieval.
pub struct JsFetchDirectory {
    base_path: String,
    /// JavaScript function: (path: string, rangeStart?: number, rangeEnd?: number) => Promise<Uint8Array>
    fetch_fn: js_sys::Function,
    /// JavaScript function: (path: string) => Promise<number> (file size)
    size_fn: js_sys::Function,
    /// Cache for file sizes
    size_cache: RwLock<HashMap<PathBuf, u64>>,
    /// Network statistics
    stats: Arc<IpfsNetworkStats>,
}

impl JsFetchDirectory {
    pub fn new(
        base_path: String,
        fetch_fn: js_sys::Function,
        size_fn: js_sys::Function,
        stats: Arc<IpfsNetworkStats>,
    ) -> Self {
        Self {
            base_path,
            fetch_fn,
            size_fn,
            size_cache: RwLock::new(HashMap::new()),
            stats,
        }
    }

    fn path_for(&self, path: &Path) -> String {
        if self.base_path.is_empty() {
            path.display().to_string()
        } else {
            format!(
                "{}/{}",
                self.base_path.trim_end_matches('/'),
                path.display()
            )
        }
    }

    /// Fetch full file content
    async fn call_fetch(&self, path: &str) -> io::Result<Vec<u8>> {
        self.call_fetch_range(path, None).await
    }

    /// Fetch file content with optional range
    async fn call_fetch_range(&self, path: &str, range: Option<Range<u64>>) -> io::Result<Vec<u8>> {
        let this = JsValue::NULL;
        let path_js = JsValue::from_str(path);

        let promise = match &range {
            Some(r) => {
                let start = JsValue::from_f64(r.start as f64);
                let end = JsValue::from_f64(r.end as f64);
                self.fetch_fn.call3(&this, &path_js, &start, &end)
            }
            None => self.fetch_fn.call1(&this, &path_js),
        }
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("JS fetch call failed: {:?}", e),
            )
        })?;

        let result = JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("JS fetch failed: {:?}", e))
            })?;

        let array = js_sys::Uint8Array::new(&result);
        let data = array.to_vec();

        // Record network stats with path and range
        self.stats.record(path, data.len() as u64, range);

        Ok(data)
    }

    async fn call_size(&self, path: &str) -> io::Result<u64> {
        let this = JsValue::NULL;
        let path_js = JsValue::from_str(path);

        let promise = self.size_fn.call1(&this, &path_js).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("JS size call failed: {:?}", e),
            )
        })?;

        let result = JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("JS size failed: {:?}", e))
            })?;

        result
            .as_f64()
            .map(|n| n as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Size is not a number"))
    }
}

#[async_trait(?Send)]
impl Directory for JsFetchDirectory {
    async fn exists(&self, _path: &Path) -> io::Result<bool> {
        // Assume files exist, will fail on actual read if not
        Ok(true)
    }

    async fn file_size(&self, path: &Path) -> io::Result<u64> {
        // Check cache first
        if let Some(&size) = self.size_cache.read().get(path) {
            return Ok(size);
        }

        let full_path = self.path_for(path);
        let size = self.call_size(&full_path).await?;

        self.size_cache.write().insert(path.to_path_buf(), size);
        Ok(size)
    }

    async fn open_read(&self, path: &Path) -> io::Result<FileSlice> {
        let full_path = self.path_for(path);
        let data = self.call_fetch(&full_path).await?;
        Ok(FileSlice::new(OwnedBytes::new(data)))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        let full_path = self.path_for(path);
        let data = self.call_fetch_range(&full_path, Some(range)).await?;
        Ok(OwnedBytes::new(data))
    }

    async fn list_files(&self, _prefix: &Path) -> io::Result<Vec<PathBuf>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "JsFetchDirectory does not support file listing",
        ))
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle> {
        // JsFetchDirectory doesn't support lazy loading directly because
        // js_sys::Function is not Send+Sync. Instead, we load the full file
        // and wrap it in a LazyFileHandle that serves from memory.
        let full_path = self.path_for(path);
        let data = self.call_fetch(&full_path).await?;
        let file_size = data.len() as u64;
        let data = Arc::new(data);

        // Create a simple range read function that reads from the cached data
        let read_fn: hermes_core::directories::RangeReadFn = Arc::new(move |range: Range<u64>| {
            let data = Arc::clone(&data);
            Box::pin(async move {
                let start = range.start as usize;
                let end = (range.end as usize).min(data.len());

                if start >= data.len() {
                    return Ok(OwnedBytes::new(vec![]));
                }

                Ok(OwnedBytes::new(data[start..end].to_vec()))
            })
        });

        Ok(LazyFileHandle::new(file_size, read_fn))
    }
}

/// Type alias for cached JS fetch directory
pub type CachedJsFetchDirectory = SliceCachingDirectory<JsFetchDirectory>;

/// IPFS Index that uses JavaScript verified-fetch for content retrieval
///
/// This allows loading indexes from IPFS without using gateways by
/// leveraging @helia/verified-fetch in JavaScript.
#[wasm_bindgen]
pub struct IpfsIndex {
    base_path: String,
    cache_size: usize,
    searcher: Option<Searcher<CachedJsFetchDirectory>>,
    directory: Option<Arc<CachedJsFetchDirectory>>,
    stats: Arc<IpfsNetworkStats>,
}

#[wasm_bindgen]
impl IpfsIndex {
    /// Create a new IPFS index
    ///
    /// @param base_path - The IPFS path (e.g., "/ipfs/Qm..." or "/ipns/...")
    #[wasm_bindgen(constructor)]
    pub fn new(base_path: String) -> Self {
        Self {
            base_path,
            cache_size: DEFAULT_CACHE_SIZE,
            searcher: None,
            directory: None,
            stats: Arc::new(IpfsNetworkStats::new()),
        }
    }

    /// Create with custom cache size
    #[wasm_bindgen]
    pub fn with_cache_size(base_path: String, cache_size: usize) -> Self {
        Self {
            base_path,
            cache_size,
            searcher: None,
            directory: None,
            stats: Arc::new(IpfsNetworkStats::new()),
        }
    }

    /// Load index using JavaScript fetch functions
    ///
    /// @param fetch_fn - JS function: (path: string) => Promise<Uint8Array>
    /// @param size_fn - JS function: (path: string) => Promise<number>
    #[wasm_bindgen]
    pub async fn load(
        &mut self,
        fetch_fn: js_sys::Function,
        size_fn: js_sys::Function,
    ) -> Result<(), JsValue> {
        let js_dir = JsFetchDirectory::new(
            self.base_path.clone(),
            fetch_fn.clone(),
            size_fn.clone(),
            Arc::clone(&self.stats),
        );
        let cached_dir = SliceCachingDirectory::new(js_dir, self.cache_size);

        // Try to load slice cache
        let cache_path = format!(
            "{}/{}",
            self.base_path.trim_end_matches('/'),
            SLICE_CACHE_FILENAME
        );

        // Try to fetch cache file via JS
        let this = JsValue::NULL;
        let cache_path_js = JsValue::from_str(&cache_path);
        if let Ok(promise) = fetch_fn.call1(&this, &cache_path_js) {
            if let Ok(result) = JsFuture::from(js_sys::Promise::from(promise)).await {
                let array = js_sys::Uint8Array::new(&result);
                let cache_data = array.to_vec();
                let _ = cached_dir.deserialize(&cache_data);
            }
        }

        let cached_dir = Arc::new(cached_dir);

        // Load metadata to get schema and segment IDs
        let metadata = IndexMetadata::load(cached_dir.as_ref())
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to load metadata: {}", e)))?;

        let schema = Arc::new(metadata.schema.clone());
        let segment_ids = metadata.segments.clone();

        // Create Searcher directly
        let searcher = Searcher::open(Arc::clone(&cached_dir), schema, &segment_ids, 32)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to open searcher: {}", e)))?;

        self.searcher = Some(searcher);
        self.directory = Some(cached_dir);
        Ok(())
    }

    /// Load index with IndexedDB cache pre-loaded
    ///
    /// This method first loads any cached data from IndexedDB, then opens the index.
    /// This allows previously cached slices to be used during index loading,
    /// reducing network requests on page refresh.
    ///
    /// @param fetch_fn - JS function: (path: string) => Promise<Uint8Array>
    /// @param size_fn - JS function: (path: string) => Promise<number>
    #[wasm_bindgen]
    pub async fn load_with_idb_cache(
        &mut self,
        fetch_fn: js_sys::Function,
        size_fn: js_sys::Function,
    ) -> Result<(), JsValue> {
        let js_dir = JsFetchDirectory::new(
            self.base_path.clone(),
            fetch_fn.clone(),
            size_fn.clone(),
            Arc::clone(&self.stats),
        );
        let cached_dir = SliceCachingDirectory::new(js_dir, self.cache_size);

        // First, try to restore cache from IndexedDB (accumulated from previous sessions)
        let idb_key = cache_key(&self.base_path);
        let mut idb_restored = false;
        if let Ok(Some(idb_data)) = idb_get(&idb_key).await {
            if cached_dir.deserialize(&idb_data).is_ok() {
                web_sys::console::log_1(&"Restored slice cache from IndexedDB".into());
                idb_restored = true;
            }
        }

        // Only fetch .slicecache from IPFS if we didn't restore from IndexedDB
        // (IndexedDB cache is more up-to-date since it includes search-time data)
        if !idb_restored {
            let cache_path = format!(
                "{}/{}",
                self.base_path.trim_end_matches('/'),
                SLICE_CACHE_FILENAME
            );

            let this = JsValue::NULL;
            let cache_path_js = JsValue::from_str(&cache_path);
            if let Ok(promise) = fetch_fn.call1(&this, &cache_path_js) {
                if let Ok(result) = JsFuture::from(js_sys::Promise::from(promise)).await {
                    let array = js_sys::Uint8Array::new(&result);
                    let cache_data = array.to_vec();
                    let _ = cached_dir.deserialize(&cache_data);
                }
            }
        }

        let cached_dir = Arc::new(cached_dir);

        // Load metadata to get schema and segment IDs
        let metadata = IndexMetadata::load(cached_dir.as_ref())
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to load metadata: {}", e)))?;

        let schema = Arc::new(metadata.schema.clone());
        let segment_ids = metadata.segments.clone();

        // Create Searcher directly
        let searcher = Searcher::open(Arc::clone(&cached_dir), schema, &segment_ids, 32)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to open searcher: {}", e)))?;

        self.searcher = Some(searcher);
        self.directory = Some(cached_dir);
        Ok(())
    }

    /// Get network statistics
    #[wasm_bindgen]
    pub fn network_stats(&self) -> JsValue {
        #[derive(Serialize)]
        struct NetworkStatsJs {
            total_requests: u64,
            total_bytes: u64,
            requests: Vec<RequestRecord>,
        }
        let stats = NetworkStatsJs {
            total_requests: self.stats.total_requests.load(Ordering::Relaxed),
            total_bytes: self.stats.total_bytes.load(Ordering::Relaxed),
            requests: self.stats.get_requests(),
        };
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }

    /// Reset network statistics
    #[wasm_bindgen]
    pub fn reset_network_stats(&self) {
        self.stats.reset();
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

        if let Some(directory) = &self.directory {
            let stats = directory.stats();
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

    /// Get number of documents
    #[wasm_bindgen]
    pub fn num_docs(&self) -> u32 {
        self.searcher.as_ref().map(|s| s.num_docs()).unwrap_or(0)
    }

    /// Get number of segments
    #[wasm_bindgen]
    pub fn num_segments(&self) -> usize {
        self.searcher
            .as_ref()
            .map(|s| s.segment_readers().len())
            .unwrap_or(0)
    }

    /// Get field names
    #[wasm_bindgen]
    pub fn field_names(&self) -> JsValue {
        let names: Vec<String> = self
            .searcher
            .as_ref()
            .map(|s| s.schema().fields().map(|(_, f)| f.name.clone()).collect())
            .unwrap_or_default();
        serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
    }

    /// Search the index
    #[wasm_bindgen]
    pub async fn search(&self, query_str: String, limit: usize) -> Result<JsValue, JsValue> {
        self.search_offset(query_str, limit, 0).await
    }

    /// Search the index with offset for pagination
    #[wasm_bindgen]
    pub async fn search_offset(
        &self,
        query_str: String,
        limit: usize,
        offset: usize,
    ) -> Result<JsValue, JsValue> {
        let searcher = self
            .searcher
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        let response = searcher
            .query_offset(&query_str, limit, offset)
            .await
            .map_err(|e| JsValue::from_str(&format!("Search error: {}", e)))?;

        response
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Get a document by address
    #[wasm_bindgen]
    pub async fn get_document(&self, segment_id: String, doc_id: u32) -> Result<JsValue, JsValue> {
        let searcher = self
            .searcher
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        let address = hermes_core::query::DocAddress { segment_id, doc_id };

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

    /// Get default fields
    #[wasm_bindgen]
    pub fn default_fields(&self) -> JsValue {
        let names: Vec<String> = self
            .searcher
            .as_ref()
            .map(|s| {
                s.default_fields()
                    .iter()
                    .filter_map(|f| s.schema().get_field_name(*f).map(|name| name.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
    }

    /// Export cache
    #[wasm_bindgen]
    pub fn export_cache(&self) -> Option<Vec<u8>> {
        self.directory.as_ref().map(|d| d.serialize())
    }

    /// Import cache
    #[wasm_bindgen]
    pub fn import_cache(&self, data: &[u8]) -> Result<(), JsValue> {
        let directory = self
            .directory
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        directory
            .deserialize(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to import cache: {}", e)))
    }

    /// Save cache to IndexedDB
    #[wasm_bindgen]
    pub async fn save_cache_to_idb(&self) -> Result<(), JsValue> {
        let cache_data = self
            .export_cache()
            .ok_or_else(|| JsValue::from_str("Index not loaded"))?;

        let key = cache_key(&self.base_path);
        idb_put(&key, &cache_data).await
    }

    /// Load cache from IndexedDB
    #[wasm_bindgen]
    pub async fn load_cache_from_idb(&self) -> Result<bool, JsValue> {
        let key = cache_key(&self.base_path);

        match idb_get(&key).await? {
            Some(data) => {
                self.import_cache(&data)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Clear IndexedDB cache
    #[wasm_bindgen]
    pub async fn clear_idb_cache(&self) -> Result<(), JsValue> {
        let key = cache_key(&self.base_path);
        idb_delete(&key).await
    }
}
