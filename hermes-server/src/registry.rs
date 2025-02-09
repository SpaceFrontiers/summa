//! Index registry for managing open indexes and writers

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use tonic::Status;

use hermes_core::{Index, IndexConfig, IndexWriter, MmapDirectory, Schema};

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

    /// Get or open an index
    ///
    /// Uses a per-index mutex to prevent concurrent Index::open for the same name.
    /// Without this, two concurrent requests can both miss the cache and open the
    /// index twice (wasting ~30s loading segments redundantly).
    pub async fn get_or_open_index(&self, name: &str) -> Result<Arc<Index<MmapDirectory>>, Status> {
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
        if !index_path.exists() {
            return Err(Status::not_found(format!("Index '{}' not found", name)));
        }

        let dir = MmapDirectory::new(&index_path);
        let index = Index::open(dir, self.config.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to open index: {}", e)))?;

        let index = Arc::new(index);
        let writer = Arc::new(tokio::sync::RwLock::new(index.writer()));

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
            .map_err(|e| Status::internal(format!("Failed to create index: {}", e)))?;

        let index = Arc::new(index);
        let writer = Arc::new(tokio::sync::RwLock::new(index.writer()));

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

    /// Evict an index from the registry atomically.
    ///
    /// Returns the handle so the caller can flush/wait on it before deleting files.
    /// Once evicted, no new operations can obtain references to this index.
    pub fn evict(&self, name: &str) -> Option<IndexHandle> {
        self.open_locks.write().remove(name);
        self.handles.write().remove(name)
    }

    /// List all indexes on disk
    pub fn list_indexes(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.data_dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_dir() {
                    entry.file_name().into_string().ok()
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        names
    }
}
