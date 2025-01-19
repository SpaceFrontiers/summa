//! HTTP-based directory for remote index access
//!
//! Uses reqwest for HTTP requests, works on both native and WASM.

use async_trait::async_trait;
use instant::Instant;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{Directory, FileSlice, LazyFileHandle, OwnedBytes, RangeReadFn};

/// A single network operation record
#[derive(Debug, Clone, serde::Serialize)]
pub struct NetworkOp {
    /// URL that was fetched
    pub url: String,
    /// Number of bytes transferred
    pub bytes: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Range request info: (start, end) if this was a range request, None for full file
    pub range: Option<(u64, u64)>,
}

/// Network statistics for HTTP directory
#[derive(Debug, Clone, serde::Serialize)]
pub struct HttpStats {
    /// Total number of requests made
    pub total_requests: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Individual operations log
    pub operations: Vec<NetworkOp>,
}

/// Internal stats tracker
struct StatsTracker {
    total_requests: AtomicU64,
    total_bytes: AtomicU64,
    operations: RwLock<Vec<NetworkOp>>,
}

impl StatsTracker {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            operations: RwLock::new(Vec::new()),
        }
    }

    fn record(&self, url: String, bytes: u64, duration_ms: u64, range: Option<(u64, u64)>) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.operations.write().push(NetworkOp {
            url,
            bytes,
            duration_ms,
            range,
        });
    }

    fn get_stats(&self) -> HttpStats {
        HttpStats {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            operations: self.operations.read().clone(),
        }
    }

    fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_bytes.store(0, Ordering::Relaxed);
        self.operations.write().clear();
    }
}

/// HTTP-based directory that fetches files from a remote server
///
/// Supports HTTP Range requests for efficient partial file reads.
/// Works on both native (with tokio) and WASM (with browser fetch).
pub struct HttpDirectory {
    base_url: String,
    client: reqwest::Client,
    /// Cache for fully loaded files (used by open_read)
    cache: RwLock<HashMap<PathBuf, Arc<Vec<u8>>>>,
    /// Network statistics tracker
    stats: Arc<StatsTracker>,
}

impl HttpDirectory {
    /// Create a new HTTP directory pointing to the given base URL
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
            cache: RwLock::new(HashMap::new()),
            stats: Arc::new(StatsTracker::new()),
        }
    }

    /// Create with a custom reqwest client
    pub fn with_client(base_url: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into(),
            client,
            cache: RwLock::new(HashMap::new()),
            stats: Arc::new(StatsTracker::new()),
        }
    }

    /// Get network statistics
    pub fn http_stats(&self) -> HttpStats {
        self.stats.get_stats()
    }

    /// Reset network statistics
    pub fn reset_stats(&self) {
        self.stats.reset()
    }

    fn url_for(&self, path: &Path) -> String {
        format!("{}/{}", self.base_url, path.display())
    }

    async fn fetch_bytes(&self, url: &str) -> io::Result<Vec<u8>> {
        let start_time = Instant::now();

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("HTTP {}: {}", response.status(), url),
            ));
        }

        let bytes = response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Record stats
        let duration_ms = start_time.elapsed().as_millis() as u64;
        self.stats
            .record(url.to_string(), bytes.len() as u64, duration_ms, None);

        Ok(bytes)
    }

    async fn fetch_range(&self, url: &str, range: Range<u64>) -> io::Result<Vec<u8>> {
        let start_time = Instant::now();
        let range_header = format!("bytes={}-{}", range.start, range.end - 1);

        let response = self
            .client
            .get(url)
            .header("Range", range_header)
            .send()
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Accept both 200 (full content) and 206 (partial content)
        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("HTTP {}: {}", response.status(), url),
            ));
        }

        let bytes = response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Record stats
        let duration_ms = start_time.elapsed().as_millis() as u64;
        self.stats.record(
            url.to_string(),
            bytes.len() as u64,
            duration_ms,
            Some((range.start, range.end)),
        );

        Ok(bytes)
    }

    async fn head_content_length(&self, url: &str) -> io::Result<u64> {
        let response = self
            .client
            .head(url)
            .send()
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("HTTP {}: {}", response.status(), url),
            ));
        }

        response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| io::Error::other("No Content-Length header"))
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Directory for HttpDirectory {
    async fn exists(&self, path: &Path) -> io::Result<bool> {
        if self.cache.read().contains_key(path) {
            return Ok(true);
        }
        // For HTTP, we assume files exist (will fail on actual read if not)
        Ok(true)
    }

    async fn file_size(&self, path: &Path) -> io::Result<u64> {
        // Check cache first
        if let Some(data) = self.cache.read().get(path) {
            return Ok(data.len() as u64);
        }

        // Use HEAD request to get Content-Length
        let url = self.url_for(path);
        self.head_content_length(&url).await
    }

    async fn open_read(&self, path: &Path) -> io::Result<FileSlice> {
        // Check cache first
        if let Some(data) = self.cache.read().get(path) {
            return Ok(FileSlice::new(OwnedBytes::new(data.as_ref().clone())));
        }

        // Fetch entire file
        let url = self.url_for(path);
        let data = self.fetch_bytes(&url).await?;

        // Cache it
        let data = Arc::new(data);
        self.cache
            .write()
            .insert(path.to_path_buf(), Arc::clone(&data));

        Ok(FileSlice::new(OwnedBytes::new(data.as_ref().clone())))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        // Check cache first
        if let Some(data) = self.cache.read().get(path) {
            let start = range.start as usize;
            let end = range.end as usize;
            if end <= data.len() {
                return Ok(OwnedBytes::new(data[start..end].to_vec()));
            }
        }

        // Fetch range from server
        let url = self.url_for(path);
        let data = self.fetch_range(&url, range).await?;
        Ok(OwnedBytes::new(data))
    }

    async fn list_files(&self, _prefix: &Path) -> io::Result<Vec<PathBuf>> {
        // HTTP directories don't support listing
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "HTTP directory does not support file listing",
        ))
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle> {
        // Get file size via HEAD request
        let file_size = self.file_size(path).await?;

        // Create the range read function
        let url = self.url_for(path);
        let client = self.client.clone();
        let stats = Arc::clone(&self.stats);

        let read_fn: RangeReadFn = Arc::new(move |range: Range<u64>| {
            let url = url.clone();
            let client = client.clone();
            let stats = Arc::clone(&stats);

            Box::pin(async move {
                let start_time = Instant::now();
                let range_header = format!("bytes={}-{}", range.start, range.end - 1);

                let response = client
                    .get(&url)
                    .header("Range", range_header)
                    .send()
                    .await
                    .map_err(|e| io::Error::other(e.to_string()))?;

                if !response.status().is_success() {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("HTTP {}", response.status()),
                    ));
                }

                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| io::Error::other(e.to_string()))?;

                // Record stats
                let duration_ms = start_time.elapsed().as_millis() as u64;
                stats.record(
                    url.clone(),
                    bytes.len() as u64,
                    duration_ms,
                    Some((range.start, range.end)),
                );

                Ok(OwnedBytes::new(bytes.to_vec()))
            })
        });

        Ok(LazyFileHandle::new(file_size, read_fn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_construction() {
        let dir = HttpDirectory::new("http://localhost:8080");
        assert_eq!(
            dir.url_for(Path::new("index/segment.bin")),
            "http://localhost:8080/index/segment.bin"
        );
    }
}
