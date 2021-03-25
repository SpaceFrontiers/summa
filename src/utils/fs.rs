use crate::errors::{Error, SummaResult, ValidationError};
use std::path::{Path, PathBuf};

pub struct IndexLayout {
    data_path: PathBuf,
    index_config_path: PathBuf,
}

impl IndexLayout {
    pub fn setup(index_path: &Path) -> SummaResult<IndexLayout> {
        if index_path.exists() {
            Err(ValidationError::ExistingPathError(index_path.to_string_lossy().to_string()))?;
        };
        let data_path = index_path.join("data");
        let index_config_path = index_path.join("index.yaml");
        std::fs::create_dir_all(&data_path).map_err(|e| Error::IOError((e, Some(data_path.clone()))))?;
        Ok(IndexLayout { data_path, index_config_path })
    }
    pub fn data_path(&self) -> &Path {
        &self.data_path
    }
    pub fn config_filepath(&self) -> &Path {
        &self.index_config_path
    }
}
