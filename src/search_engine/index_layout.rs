use crate::errors::{Error, SummaResult, ValidationError};
use std::path::{Path, PathBuf};
use tokio::fs::{create_dir_all, remove_dir_all};
use tracing::info;

/// Physical index layout on the disk
#[derive(Clone, Debug)]
pub struct IndexLayout {
    data_path: PathBuf,
    index_config_path: PathBuf,
}

impl IndexLayout {
    /// Creates all directories and initializes paths
    pub async fn setup(index_path: &Path) -> SummaResult<IndexLayout> {
        if index_path.exists() {
            Err(ValidationError::ExistingPathError(index_path.to_string_lossy().to_string()))?;
        };
        let data_path = index_path.join("data");
        let index_config_path = index_path.join("index.yaml");
        create_dir_all(&data_path).await.map_err(|e| Error::IOError((e, Some(data_path.clone()))))?;
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
    pub async fn delete(self) -> SummaResult<()> {
        let index_path = self.data_path.parent().unwrap();
        info!(action = "delete_directory", index_path = ?index_path);
        remove_dir_all(&index_path).await.map_err(|e| Error::IOError((e, Some(index_path.to_path_buf()))))
    }
}
