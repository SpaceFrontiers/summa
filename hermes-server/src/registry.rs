//! Index registry for managing open indexes and writers

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

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

/// Index registry holding all open indexes
pub struct IndexRegistry {
    /// Single map: name → handle (index + writer together)
    handles: RwLock<HashMap<String, IndexHandle>>,
    /// Per-index open locks to prevent concurrent Index::open for the same name
    open_locks: RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    pub(crate) data_dir: PathBuf,
    config: IndexConfig,
}

impl IndexRegistry {
    pub fn new(data_dir: PathBuf, config: IndexConfig) -> Self {
        Self {
            handles: RwLock::new(HashMap::new()),
            open_locks: RwLock::new(HashMap::new()),
            data_dir,
            config,
        }
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
        Self::validate_index_name(name)?;

        // Fast path: already cached
        if let Some(h) = self.handles.read().get(name) {
            return Ok(Arc::clone(&h.index));
        }

        // Get or create per-index open lock
        let lock = {
            let mut locks = self.open_locks.write();
            Arc::clone(
                locks
                    .entry(name.to_string())
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
            )
        };

        // Serialize open attempts for the same index name
        let _guard = lock.lock().await;

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

        let index = Arc::new(index);
        let mut w = index.writer();
        w.init_primary_key_dedup()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let writer = Arc::new(tokio::sync::RwLock::new(w));

        Self::precache_idf_files(&index);

        self.handles.write().insert(
            name.to_string(),
            IndexHandle {
                index: Arc::clone(&index),
                writer,
            },
        );
        Ok(index)
    }

    /// Create a new index
    pub async fn create_index(&self, name: &str, schema: Schema) -> Result<(), Status> {
        Self::validate_index_name(name)?;
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

        self.handles.write().insert(
            name.to_string(),
            IndexHandle {
                index: Arc::clone(&index),
                writer,
            },
        );
        Ok(())
    }

    /// Get writer for an index (opens index if needed)
    pub async fn get_writer(
        &self,
        name: &str,
    ) -> Result<Arc<tokio::sync::RwLock<IndexWriter<MmapDirectory>>>, Status> {
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
    fn precache_idf_files(index: &Index<MmapDirectory>) {
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
            std::thread::spawn(move || {
                hermes_core::tokenizer::idf_weights_cache().get_or_load(&name, Some(&dir));
            });
        }
    }

    /// Evict an index from the registry atomically.
    ///
    /// Returns the handle so the caller can flush/wait on it before deleting files.
    /// Once evicted, no new operations can obtain references to this index.
    pub fn evict(&self, name: &str) -> Option<IndexHandle> {
        self.open_locks.write().remove(name);
        self.handles.write().remove(name)
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
