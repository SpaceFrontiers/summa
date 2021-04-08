use crate::errors::{Error, SummaResult, ValidationError};
use std::path::{Path, PathBuf};
use tokio::fs::{create_dir_all, remove_dir_all};
use tracing::{info, instrument};

/// Physical index layout on the disk
#[derive(Clone, Debug)]
pub struct IndexLayout {
    data_path: PathBuf,
    index_config_path: PathBuf,
}

impl IndexLayout {
    /// Creates all directories and initializes paths
    pub async fn create(index_path: &Path) -> SummaResult<IndexLayout> {
        if index_path.exists() {
            Err(ValidationError::ExistingPathError(index_path.to_string_lossy().to_string()))?;
        }
        let index_layout = IndexLayout::new(index_path)?;
        create_dir_all(&index_layout.data_path())
            .await
            .map_err(|e| Error::IOError((e, Some(index_layout.data_path.clone()))))?;
        Ok(index_layout)
    }

    /// Open and validates layout
    pub async fn open(index_path: &Path) -> SummaResult<IndexLayout> {
        if !index_path.exists() {
            Err(ValidationError::MissingPathError(index_path.to_string_lossy().to_string()))?;
        }
        IndexLayout::new(index_path)
    }

    /// Create layout
    fn new(index_path: &Path) -> SummaResult<IndexLayout> {
        let data_path = index_path.join("data");
        let index_config_path = index_path.join("index.yaml");
        Ok(IndexLayout { data_path, index_config_path })
    }
    /// Directory for storing binary index data
    pub fn data_path(&self) -> &Path {
        &self.data_path
    }

    /// Path for storing an index config
    pub fn config_filepath(&self) -> &Path {
        &self.index_config_path
    }

    /// Delete all directories
    #[instrument]
    pub async fn delete(self) -> SummaResult<()> {
        let index_path = self.data_path.parent().unwrap();
        info!(action = "delete_directory");
        remove_dir_all(&index_path).await.map_err(|e| Error::IOError((e, Some(index_path.to_path_buf()))))
    }
}
