//! Index registry for managing open indexes and writers

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::RwLock;
use tonic::Status;

use log::{info, warn};

use hermes_core::segment::{SegmentId, SegmentReader, delete_segment};
use hermes_core::structures::QueryWeighting;
use hermes_core::{Index, IndexConfig, IndexMetadata, IndexWriter, MmapDirectory, Schema};

/// Combined index + writer handle under a single registry entry
pub struct IndexHandle {
    pub index: Arc<Index<MmapDirectory>>,
    pub writer: Arc<tokio::sync::RwLock<IndexWriter<MmapDirectory>>>,
}

/// Exclusive per-name lease held for the complete delete transaction. It
/// serializes deletion with an index already being opened or created, closing
/// the race where an in-flight opener could reinsert a handle after eviction.
pub struct IndexDeleteLease {
    index_path: PathBuf,
    handle: Option<IndexHandle>,
    _open_guard: tokio::sync::OwnedMutexGuard<()>,
}

impl IndexDeleteLease {
    /// Finish deletion independently of the requesting RPC.
    ///
    /// Once the registry entry is evicted and `.deleting` is visible, no new
    /// raw handle can be issued. Wait for previously issued search/writer Arcs
    /// before shutting down lifecycle work and unlinking the directory.
    pub async fn complete(mut self) -> Result<(), Status> {
        if let Some(handle) = self.handle.take() {
            let mut next_log = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                let index_users = Arc::strong_count(&handle.index).saturating_sub(1);
                let writer_users = Arc::strong_count(&handle.writer).saturating_sub(1);
                if index_users == 0 && writer_users == 0 {
                    break;
                }
                if std::time::Instant::now() >= next_log {
                    log::debug!(
                        "[index_delete] waiting for {} search and {} writer handle(s)",
                        index_users,
                        writer_users,
                    );
                    next_log += std::time::Duration::from_secs(30);
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }

            let segment_manager = {
                let mut writer = handle.writer.write().await;
                let manager = Arc::clone(writer.segment_manager());
                writer
                    .shutdown()
                    .await
                    .map_err(crate::error::hermes_error_to_status)?;
                manager
            };

            // Cached readers and the writer disappear before the lifecycle
            // drain. Blocking merge phases cannot be canceled by aborting
            // their async wrapper, so wait for actual ownership to finish.
            drop(handle);
            segment_manager.wait_for_shutdown().await;
        }

        if self.index_path.exists() {
            let index_path = self.index_path.clone();
            tokio::task::spawn_blocking(move || std::fs::remove_dir_all(&index_path))
                .await
                .map_err(|e| Status::internal(format!("Delete task failed: {}", e)))?
                .map_err(|e| Status::internal(format!("Failed to delete index: {}", e)))?;
        }
        Ok(())
    }
}

/// Index registry holding all open indexes
pub struct IndexRegistry {
    /// Single map: name → handle (index + writer together)
    handles: RwLock<HashMap<String, IndexHandle>>,
    /// Per-index open locks to prevent concurrent Index::open for the same name
    open_locks: RwLock<HashMap<String, Weak<tokio::sync::Mutex<()>>>>,
    pub(crate) data_dir: PathBuf,
    config: IndexConfig,
    shutting_down: AtomicBool,
}

impl IndexRegistry {
    pub fn new(data_dir: PathBuf, config: IndexConfig) -> Self {
        Self {
            handles: RwLock::new(HashMap::new()),
            open_locks: RwLock::new(HashMap::new()),
            data_dir,
            config,
            shutting_down: AtomicBool::new(false),
        }
    }

    fn ensure_running(&self) -> Result<(), Status> {
        if self.shutting_down.load(Ordering::Acquire) {
            Err(Status::unavailable("Hermes server is shutting down"))
        } else {
            Ok(())
        }
    }

    /// Close lifecycle admission immediately when process shutdown starts.
    ///
    /// This method is synchronous so the signal future can stop new work
    /// before tonic finishes draining in-flight RPCs.
    pub fn begin_shutdown(&self) {
        if self.shutting_down.swap(true, Ordering::AcqRel) {
            return;
        }
        for handle in self.handles.read().values() {
            handle.index.segment_manager().begin_shutdown();
        }
    }

    /// Stop all index writers while Tokio is still alive, then drain every
    /// merge, reorder, metadata transaction, and deferred cleanup.
    pub async fn shutdown(&self) -> Result<(), Status> {
        self.begin_shutdown();
        let handles = std::mem::take(&mut *self.handles.write());
        info!(
            "[shutdown] stopping writers for {} open index(es)",
            handles.len()
        );
        let mut managers = Vec::with_capacity(handles.len());
        let mut first_error = None;

        for (name, handle) in handles {
            let manager = Arc::clone(handle.index.segment_manager());
            if let Err(error) = handle.writer.write().await.shutdown().await
                && first_error.is_none()
            {
                first_error = Some(crate::error::hermes_error_to_status(error));
            }
            drop(handle);
            managers.push((name, manager));
        }

        for (name, manager) in managers {
            manager.wait_for_shutdown().await;
            info!("[shutdown] index '{}' drained", name);
        }

        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn open_lock(&self, name: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.open_locks.write();
        // The registry must not retain one mutex and name forever for every
        // typo/404 ever requested. Weak entries still unify concurrent calls;
        // expired entries are pruned opportunistically under the same lock.
        locks.retain(|_, lock| lock.strong_count() > 0);
        if let Some(lock) = locks.get(name).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(name.to_string(), Arc::downgrade(&lock));
        lock
    }

    /// Validate index name to prevent path traversal and other issues.
    /// Allows alphanumeric characters, hyphens, underscores, and dots (not leading).
    fn validate_index_name(name: &str) -> Result<(), Status> {
        if name.is_empty() {
            return Err(Status::invalid_argument("Index name must not be empty"));
        }
        if name.len() > 255 {
            return Err(Status::invalid_argument(
                "Index name must not exceed 255 characters",
            ));
        }
        if name.contains('/') || name.contains('\\') || name.contains("..") {
            return Err(Status::invalid_argument(
                "Index name must not contain '/', '\\', or '..'",
            ));
        }
        if name.starts_with('.') || name.starts_with('-') {
            return Err(Status::invalid_argument(
                "Index name must not start with '.' or '-'",
            ));
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            return Err(Status::invalid_argument(
                "Index name must contain only alphanumeric characters, hyphens, underscores, or dots",
            ));
        }
        Ok(())
    }

    /// Get or open an index
    ///
    /// Uses a per-index mutex to prevent concurrent Index::open for the same name.
    /// Without this, two concurrent requests can both miss the cache and open the
    /// index twice (wasting ~30s loading segments redundantly).
    pub async fn get_or_open_index(&self, name: &str) -> Result<Arc<Index<MmapDirectory>>, Status> {
        self.ensure_running()?;
        Self::validate_index_name(name)?;

        // Fast path: already cached
        if let Some(h) = self.handles.read().get(name) {
            return Ok(Arc::clone(&h.index));
        }

        // Get or create per-index open lock
        let lock = self.open_lock(name);

        // Serialize open attempts for the same index name
        let _guard = lock.lock().await;
        self.ensure_running()?;

        // Re-check cache after acquiring lock (another task may have opened it)
        if let Some(h) = self.handles.read().get(name) {
            return Ok(Arc::clone(&h.index));
        }

        // Open from disk
        let index_path = self.data_dir.join(name);
        if !index_path.exists() || index_path.join(".deleting").exists() {
            return Err(Status::not_found(format!("Index '{}' not found", name)));
        }

        let dir = MmapDirectory::new(&index_path);
        let index = Index::open(dir, self.config.clone())
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        // This registry lock guarantees there is no second server-side
        // producer for the index while crash leftovers are swept. Keep this
        // out of the read-only core `Index::open` API: opening a search handle
        // must not delete files owned by an independently opened writer.
        let swept = index
            .segment_manager()
            .cleanup_orphan_segments()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        if swept > 0 {
            warn!(
                "[segment_cleanup] swept {} crash-leftover segment(s) while opening '{}'",
                swept, name
            );
        }

        let index = Arc::new(index);
        let mut w = index.writer();
        w.init_primary_key_dedup()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let writer = Arc::new(tokio::sync::RwLock::new(w));

        Self::precache_idf_files(&index);

        {
            let mut handles = self.handles.write();
            if !self.shutting_down.load(Ordering::Acquire) {
                handles.insert(
                    name.to_string(),
                    IndexHandle {
                        index: Arc::clone(&index),
                        writer,
                    },
                );
                return Ok(index);
            }
        }

        let manager = Arc::clone(index.segment_manager());
        manager.begin_shutdown();
        writer
            .write()
            .await
            .shutdown()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        drop(writer);
        drop(index);
        manager.wait_for_shutdown().await;
        Err(Status::unavailable("Hermes server is shutting down"))
    }

    /// Create a new index
    pub async fn create_index(&self, name: &str, schema: Schema) -> Result<(), Status> {
        self.ensure_running()?;
        Self::validate_index_name(name)?;
        let _open_guard = self.open_lock(name).lock_owned().await;
        self.ensure_running()?;
        let index_path = self.data_dir.join(name);

        if index_path.exists() {
            return Err(Status::already_exists(format!(
                "Index '{}' already exists",
                name
            )));
        }

        std::fs::create_dir_all(&index_path)
            .map_err(|e| Status::internal(format!("Failed to create directory: {}", e)))?;

        let dir = MmapDirectory::new(&index_path);
        let index = Index::create(dir, schema, self.config.clone())
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        let index = Arc::new(index);
        let mut w = index.writer();
        w.init_primary_key_dedup()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let writer = Arc::new(tokio::sync::RwLock::new(w));

        Self::precache_idf_files(&index);

        {
            let mut handles = self.handles.write();
            if !self.shutting_down.load(Ordering::Acquire) {
                handles.insert(
                    name.to_string(),
                    IndexHandle {
                        index: Arc::clone(&index),
                        writer,
                    },
                );
                return Ok(());
            }
        }

        let manager = Arc::clone(index.segment_manager());
        manager.begin_shutdown();
        writer
            .write()
            .await
            .shutdown()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        drop(writer);
        drop(index);
        manager.wait_for_shutdown().await;
        Err(Status::unavailable("Hermes server is shutting down"))
    }

    /// Get writer for an index (opens index if needed)
    pub async fn get_writer(
        &self,
        name: &str,
    ) -> Result<Arc<tokio::sync::RwLock<IndexWriter<MmapDirectory>>>, Status> {
        self.ensure_running()?;
        if let Some(h) = self.handles.read().get(name) {
            return Ok(Arc::clone(&h.writer));
        }

        // Open index first (creates writer too)
        self.get_or_open_index(name).await?;

        // Now the handle exists
        self.handles
            .read()
            .get(name)
            .map(|h| Arc::clone(&h.writer))
            .ok_or_else(|| Status::internal("Failed to create writer"))
    }

    /// Eagerly download and cache idf.json for all sparse vector fields
    /// that use `IdfFile` weighting. Runs in background threads so it doesn't
    /// block index open/create. On success the file is saved to the index
    /// directory; on failure a warning is logged (non-fatal).
    fn precache_idf_files(index: &Arc<Index<MmapDirectory>>) {
        let index_dir = index.directory().root().to_path_buf();

        // Collect model names that need IDF files
        let models: Vec<String> = index
            .schema()
            .fields()
            .filter_map(|(_, entry)| {
                let query_cfg = entry.sparse_vector_config.as_ref()?.query_config.as_ref()?;
                if query_cfg.weighting == QueryWeighting::IdfFile {
                    query_cfg.tokenizer.clone()
                } else {
                    None
                }
            })
            .collect();

        for name in models {
            let dir = index_dir.clone();
            // Keep this write visible to index deletion's issued-handle drain.
            // Capturing only the path let an untracked downloader write into a
            // directory after the registry had removed it.
            let index_lease = Arc::clone(index);
            std::thread::spawn(move || {
                let _index_lease = index_lease;
                hermes_core::tokenizer::idf_weights_cache().get_or_load(&name, Some(&dir));
            });
        }
    }

    /// Begin an index delete while holding the same per-name lock used by
    /// open/create. The marker is installed before eviction and the returned
    /// lease keeps the lock until filesystem removal completes.
    pub async fn begin_delete(&self, name: &str) -> Result<IndexDeleteLease, Status> {
        self.ensure_running()?;
        Self::validate_index_name(name)?;
        let open_guard = self.open_lock(name).lock_owned().await;
        let index_path = self.data_dir.join(name);

        if index_path.exists() {
            std::fs::File::create(index_path.join(".deleting")).map_err(|error| {
                Status::internal(format!("Failed to mark index for deletion: {}", error))
            })?;
        }

        let handle = self.handles.write().remove(name);
        Ok(IndexDeleteLease {
            index_path,
            handle,
            _open_guard: open_guard,
        })
    }

    /// Remove index directories left over from incomplete deletes.
    ///
    /// If the server crashed between placing the `.deleting` marker and
    /// finishing `remove_dir_all`, the directory is still on disk. This
    /// method cleans them up at startup.
    pub fn cleanup_incomplete_deletes(&self) {
        let entries = match std::fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if !ft.is_dir() {
                continue;
            }
            let path = entry.path();
            if path.join(".deleting").exists() {
                let name = entry.file_name();
                match std::fs::remove_dir_all(&path) {
                    Ok(_) => info!("Cleaned up incomplete delete: {:?}", name),
                    Err(e) => warn!("Failed to clean up {:?}: {}", name, e),
                }
            }
        }
    }

    /// Validate all indexes on disk, removing corrupt segments.
    ///
    /// For each index directory, loads metadata.json, tries to open every
    /// segment, and removes any that fail validation. Operates directly on
    /// metadata files — the index is not opened through the normal path.
    pub async fn doctor_all_indexes(&self) {
        info!("Doctor: scanning indexes in {:?}", self.data_dir);

        let entries = match std::fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(e) => {
                warn!("Doctor: cannot read data directory: {}", e);
                return;
            }
        };

        let mut total_removed = 0usize;
        let mut indexes_checked = 0usize;

        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if !ft.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().into_string().ok() else {
                continue;
            };
            // Skip directories still marked for deletion (cleanup may have
            // failed if files are locked — will retry next startup)
            if entry.path().join(".deleting").exists() {
                continue;
            }

            let index_path = self.data_dir.join(&name);
            let dir = MmapDirectory::new(&index_path);

            // Load metadata — skip directories that aren't indexes
            let meta = match IndexMetadata::load(&dir).await {
                Ok(m) => m,
                Err(e) => {
                    warn!("Doctor: {}: cannot load metadata, skipping ({})", name, e);
                    continue;
                }
            };

            indexes_checked += 1;
            let schema = Arc::new(meta.schema.clone());
            let segment_ids: Vec<String> = meta.segment_ids();

            if segment_ids.is_empty() {
                continue;
            }

            let mut bad_segments: Vec<String> = Vec::new();
            for seg_id_str in &segment_ids {
                let Some(seg_id) = SegmentId::from_hex(seg_id_str) else {
                    warn!(
                        "Doctor: {}: invalid segment id '{}', marking corrupt",
                        name, seg_id_str
                    );
                    bad_segments.push(seg_id_str.clone());
                    continue;
                };

                match SegmentReader::open(&dir, seg_id, Arc::clone(&schema), 0).await {
                    Ok(_reader) => {
                        // Segment is valid — drop the reader
                    }
                    Err(e) => {
                        warn!(
                            "Doctor: {}: segment {} is corrupt ({}), will remove",
                            name, seg_id_str, e
                        );
                        bad_segments.push(seg_id_str.clone());
                    }
                }
            }

            if bad_segments.is_empty() {
                info!("Doctor: {}: all {} segments OK", name, segment_ids.len());
                continue;
            }

            // Remove bad segments from metadata and save
            let mut meta = meta;
            for seg_id_str in &bad_segments {
                meta.remove_segment(seg_id_str);
            }
            if let Err(e) = meta.save(&dir).await {
                warn!("Doctor: {}: failed to save metadata: {}", name, e);
                continue;
            }

            // Delete orphan segment files
            for seg_id_str in &bad_segments {
                if let Some(seg_id) = SegmentId::from_hex(seg_id_str) {
                    let _ = delete_segment(&dir, seg_id).await;
                }
            }

            let removed = bad_segments.len();
            total_removed += removed;
            info!(
                "Doctor: {}: removed {} corrupt segment(s), {} remaining",
                name,
                removed,
                segment_ids.len() - removed,
            );
        }

        info!(
            "Doctor: done — checked {} index(es), removed {} corrupt segment(s)",
            indexes_checked, total_removed,
        );
    }

    /// List all indexes on disk.
    ///
    /// Filesystem I/O is done inside `spawn_blocking` to avoid stalling
    /// the tokio worker threads under heavy load.
    pub async fn list_indexes(&self) -> Result<Vec<String>, Status> {
        self.ensure_running()?;
        let data_dir = self.data_dir.clone();
        tokio::task::spawn_blocking(move || {
            let mut names: Vec<String> = std::fs::read_dir(&data_dir)
                .into_iter()
                .flatten()
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    if entry.file_type().ok()?.is_dir() {
                        let path = entry.path();
                        // Skip indexes that are being deleted
                        if path.join(".deleting").exists() {
                            return None;
                        }
                        entry.file_name().into_string().ok()
                    } else {
                        None
                    }
                })
                .collect();
            names.sort();
            names
        })
        .await
        .map_err(|e| Status::internal(format!("list_indexes task failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn delete_waits_for_previously_issued_index_handle() {
        let root = std::env::temp_dir().join(format!(
            "hermes_registry_delete_{}",
            SegmentId::new().to_hex()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let registry = IndexRegistry::new(
            root.clone(),
            IndexConfig {
                num_indexing_threads: 1,
                ..Default::default()
            },
        );
        registry
            .create_index("held", hermes_core::SchemaBuilder::default().build())
            .await
            .unwrap();

        let issued = registry.get_or_open_index("held").await.unwrap();
        let lease = registry.begin_delete("held").await.unwrap();
        let completion = tokio::spawn(async move { lease.complete().await });

        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert!(
            !completion.is_finished(),
            "filesystem deletion raced a previously issued search handle"
        );
        assert!(root.join("held").exists());

        drop(issued);
        tokio::time::timeout(std::time::Duration::from_secs(2), completion)
            .await
            .expect("delete did not resume after the final search handle dropped")
            .unwrap()
            .unwrap();
        assert!(!root.join("held").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn process_shutdown_drains_indexes_and_rejects_new_work() {
        let root = std::env::temp_dir().join(format!(
            "hermes_registry_shutdown_{}",
            SegmentId::new().to_hex()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let registry = IndexRegistry::new(
            root.clone(),
            IndexConfig {
                num_indexing_threads: 1,
                ..Default::default()
            },
        );
        registry
            .create_index("open", hermes_core::SchemaBuilder::default().build())
            .await
            .unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(2), registry.shutdown())
            .await
            .expect("registry shutdown timed out")
            .unwrap();

        assert_eq!(
            registry
                .get_or_open_index("open")
                .await
                .err()
                .expect("open must be rejected")
                .code(),
            tonic::Code::Unavailable,
        );
        assert_eq!(
            registry
                .create_index("new", hermes_core::SchemaBuilder::default().build())
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unavailable,
        );
        assert_eq!(
            registry.list_indexes().await.unwrap_err().code(),
            tonic::Code::Unavailable,
        );

        drop(registry);
        std::fs::remove_dir_all(root).unwrap();
    }
}
