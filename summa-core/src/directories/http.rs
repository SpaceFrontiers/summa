//! HTTP-based directory for remote index access
//!
//! Uses reqwest for HTTP requests, works on both native and WASM.

use async_trait::async_trait;
use instant::Instant;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{Directory, FileHandle, OwnedBytes, RangeReadFn};

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
    /// Total recorded response-body transfers
    pub total_requests: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Latest 256 completed operations, in recording order.
    pub operations: Vec<NetworkOp>,
    /// Older operations omitted from the bounded history; totals include them.
    pub omitted_operations: u64,
}

/// The diagnostics lock owns both totals and bounded history, including reset.
#[derive(Default)]
struct StatsTracker(RwLock<StatsState>);

#[derive(Default)]
struct StatsState {
    total_requests: u64,
    total_bytes: u64,
    operations: VecDeque<NetworkOp>,
}

impl StatsTracker {
    fn new() -> Self {
        Self::default()
    }

    fn record(&self, url: String, bytes: u64, duration_ms: u64, range: Option<(u64, u64)>) {
        let mut state = self.0.write();
        state.total_requests = state.total_requests.saturating_add(1);
        state.total_bytes = state.total_bytes.saturating_add(bytes);
        if state.operations.len() == 256 {
            state.operations.pop_front();
        }
        state.operations.push_back(NetworkOp {
            url,
            bytes,
            duration_ms,
            range,
        });
    }

    fn get_stats(&self) -> HttpStats {
        let state = self.0.read();
        HttpStats {
            total_requests: state.total_requests,
            total_bytes: state.total_bytes,
            omitted_operations: state
                .total_requests
                .saturating_sub(state.operations.len() as u64),
            operations: state.operations.iter().cloned().collect(),
        }
    }

    fn reset(&self) {
        let mut state = self.0.write();
        state.total_requests = 0;
        state.total_bytes = 0;
        state.operations.clear();
    }
}

/// HTTP-based directory that fetches files from a remote server
///
/// Supports HTTP Range requests for efficient partial file reads.
/// Works on both native (with tokio) and WASM (with browser fetch). Add a
/// [`super::SliceCachingDirectory`] for bounded caching; the transport retains no files.
pub struct HttpDirectory {
    base_url: String,
    client: reqwest::Client,
    /// Network statistics tracker
    stats: Arc<StatsTracker>,
    /// Index name for Directory-layer metric labels
    label: super::IndexLabel,
}

impl HttpDirectory {
    /// Create a new HTTP directory pointing to the given base URL
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_client(base_url, reqwest::Client::new())
    }

    /// Create with a custom reqwest client
    pub fn with_client(base_url: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into(),
            client,
            stats: Arc::new(StatsTracker::new()),
            label: super::IndexLabel::default(),
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

    async fn fetch_range(
        client: &reqwest::Client,
        stats: &StatsTracker,
        url: &str,
        range: Range<u64>,
    ) -> io::Result<OwnedBytes> {
        let expected = range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "reversed read range"))?;
        if expected == 0 {
            return Ok(OwnedBytes::empty());
        }
        let start_time = Instant::now();
        let response = client
            .get(url)
            .header("Range", format!("bytes={}-{}", range.start, range.end - 1))
            .send()
            .await
            .map_err(|error| io::Error::other(error.to_string()))?;
        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("HTTP {}: {}", response.status(), url),
            ));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| io::Error::other(error.to_string()))?;
        stats.record(
            url.to_owned(),
            bytes.len() as u64,
            start_time.elapsed().as_millis() as u64,
            Some((range.start, range.end)),
        );
        if bytes.len() as u64 != expected {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "HTTP range returned an incorrect byte count",
            ));
        }
        Ok(OwnedBytes::new(bytes.to_vec()))
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
    async fn exists(&self, _path: &Path) -> io::Result<bool> {
        // HTTP existence is checked by the subsequent actual request.
        Ok(true)
    }

    async fn file_size(&self, path: &Path) -> io::Result<u64> {
        self.head_content_length(&self.url_for(path)).await
    }

    async fn open_read(&self, path: &Path) -> io::Result<FileHandle> {
        let bytes = self.fetch_bytes(&self.url_for(path)).await?;
        Ok(FileHandle::from_bytes(OwnedBytes::new(bytes)))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        Self::fetch_range(&self.client, &self.stats, &self.url_for(path), range).await
    }

    async fn list_files(&self, _prefix: &Path) -> io::Result<Vec<PathBuf>> {
        // HTTP directories don't support listing
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "HTTP directory does not support file listing",
        ))
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle> {
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

            Box::pin(async move { Self::fetch_range(&client, &stats, &url, range).await })
        });

        Ok(FileHandle::lazy_labeled(
            file_size,
            read_fn,
            self.label.get(),
        ))
    }

    fn set_index_label(&self, label: &str) {
        self.label.set(label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn http_ranges_reject_reversed_ranges_before_network_io() {
        let directory = HttpDirectory::new("http://127.0.0.1:1");
        let error = directory
            .read_range(Path::new("data"), Range { start: 5, end: 3 })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn direct_http_ranges_reject_short_and_overlong_responses() {
        use std::io::{Read, Write};
        for actual in [1, 3] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = std::thread::spawn(move || {
                let (mut connection, _) = listener.accept().unwrap();
                connection
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let mut byte = [0];
                    connection.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                    assert!(request.len() <= 2048);
                }
                write!(connection, "HTTP/1.1 206 Partial Content\r\nContent-Length: {actual}\r\nConnection: close\r\n\r\n{}", "x".repeat(actual)).unwrap();
            });
            let directory = HttpDirectory::new(url);
            assert_eq!(
                directory
                    .read_range(Path::new("data"), 0..2)
                    .await
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::UnexpectedEof
            );
            server.join().unwrap();
        }
    }

    #[test]
    fn network_operation_history_is_bounded_without_losing_totals() {
        let stats = StatsTracker::new();
        for n in 0..1024 {
            stats.record(format!("http://example/{n}"), 3, 1, None);
        }
        let snapshot = stats.get_stats();
        assert!(snapshot.operations.len() <= 256);
        assert_eq!(snapshot.total_requests, 1024);
        assert_eq!(snapshot.omitted_operations, 768);
        assert_eq!(snapshot.total_bytes, 3072);
        assert_eq!(
            snapshot.operations.last().unwrap().url,
            "http://example/1023"
        );
        stats.reset();
        assert_eq!(stats.get_stats().total_requests, 0);
        assert!(stats.get_stats().operations.is_empty());
    }

    #[test]
    fn test_url_construction() {
        let dir = HttpDirectory::new("http://localhost:8080");
        assert_eq!(
            dir.url_for(Path::new("index/segment.bin")),
            "http://localhost:8080/index/segment.bin"
        );
    }
}
