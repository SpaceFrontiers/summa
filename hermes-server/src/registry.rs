//! Index registry for managing open indexes and writers

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use tonic::Status;

use hermes_core::{FsDirectory, Index, IndexConfig, IndexWriter, Schema};

/// Index registry holding all open indexes
pub struct IndexRegistry {
    /// Open indexes (Index is the central concept)
    indexes: RwLock<HashMap<String, Arc<Index<FsDirectory>>>>,
    /// Cached writers - one per index, reused across requests to avoid segment fragmentation
    writers: RwLock<HashMap<String, Arc<tokio::sync::Mutex<IndexWriter<FsDirectory>>>>>,
    pub(crate) data_dir: PathBuf,
    config: IndexConfig,
}

impl IndexRegistry {
    pub fn new(data_dir: PathBuf, config: IndexConfig) -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            writers: RwLock::new(HashMap::new()),
            data_dir,
            config,
        }
    }

    /// Get or open an index
    pub async fn get_or_open_index(&self, name: &str) -> Result<Arc<Index<FsDirectory>>, Status> {
        // Check if already open
        if let Some(index) = self.indexes.read().get(name) {
            return Ok(Arc::clone(index));
        }

        // Open from disk
        let index_path = self.data_dir.join(name);
        if !index_path.exists() {
            return Err(Status::not_found(format!("Index '{}' not found", name)));
        }

        let dir = FsDirectory::new(&index_path);
        let index = Index::open(dir, self.config.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to open index: {}", e)))?;

        let index = Arc::new(index);
        self.indexes
            .write()
            .insert(name.to_string(), Arc::clone(&index));
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

        let dir = FsDirectory::new(&index_path);
        let index = Index::create(dir, schema, self.config.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to create index: {}", e)))?;

        let index = Arc::new(index);
        let writer = Arc::new(tokio::sync::Mutex::new(index.writer()));

        self.indexes
            .write()
            .insert(name.to_string(), Arc::clone(&index));
        self.writers.write().insert(name.to_string(), writer);
        Ok(())
    }

    /// Get or create a cached writer for an index
    pub async fn get_writer(
        &self,
        name: &str,
    ) -> Result<Arc<tokio::sync::Mutex<IndexWriter<FsDirectory>>>, Status> {
        // Check if writer already exists
        if let Some(writer) = self.writers.read().get(name) {
            return Ok(Arc::clone(writer));
        }

        // Need to create writer - first ensure index is open
        let index = self.get_or_open_index(name).await?;

        // Double-check and create writer if needed
        let mut writers = self.writers.write();
        if let Some(writer) = writers.get(name) {
            return Ok(Arc::clone(writer));
        }

        let writer = Arc::new(tokio::sync::Mutex::new(index.writer()));
        writers.insert(name.to_string(), Arc::clone(&writer));
        Ok(writer)
    }

    /// Get a cloned writer reference if one exists (for waiting on merges before delete)
    pub fn get_existing_writer(
        &self,
        name: &str,
    ) -> Option<Arc<tokio::sync::Mutex<IndexWriter<FsDirectory>>>> {
        self.writers.read().get(name).cloned()
    }

    /// Remove writer and index from registry
    pub fn remove(&self, name: &str) {
        self.writers.write().remove(name);
        self.indexes.write().remove(name);
    }
}
