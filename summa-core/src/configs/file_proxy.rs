use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};
use serde_yaml::to_writer;
use tokio::io::AsyncWriteExt;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::configs::{ConfigProxy, ConfigReadProxy, ConfigWriteProxy};
use crate::errors::{Error, SummaResult, ValidationError};

#[async_trait]
pub trait Persistable: Send + Sync {
    async fn save(&self) -> SummaResult<()>;
}

pub trait Loadable {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>) -> SummaResult<Self>
    where
        Self: Sized;
}

#[derive(Clone, Debug)]
pub struct FileProxy<TConfig: Send + Sync + Serialize> {
    config: Arc<RwLock<TConfig>>,
    config_filepath: PathBuf,
}

pub struct FileReadProxy<'a, TConfig: Send + Sync + Serialize> {
    config: RwLockReadGuard<'a, TConfig>,
}

pub struct FileWriteProxy<'a, TConfig: Send + Sync + Serialize> {
    config: RwLockWriteGuard<'a, TConfig>,
    config_filepath: PathBuf,
}

impl<'a, TConfig: Send + Sync + Serialize + Deserialize<'a>> FileProxy<TConfig> {
    pub fn file(config: TConfig, config_filepath: &Path) -> Self {
        FileProxy {
            config: Arc::new(RwLock::new(config)),
            config_filepath: config_filepath.to_path_buf(),
        }
    }
}

impl<'a, TConfig: Send + Sync + Serialize + Deserialize<'a>> Loadable for FileProxy<TConfig> {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>) -> SummaResult<FileProxy<TConfig>> {
        let mut s = Config::builder();
        if !config_filepath.exists() {
            return Err(ValidationError::MissingPath(config_filepath.to_path_buf()).into());
        }
        s = s.add_source(File::from(config_filepath));
        if let Some(env_prefix) = env_prefix {
            s = s.add_source(Environment::with_prefix(env_prefix).separator("."));
        }
        let config: TConfig = s.build()?.try_deserialize().map_err(Error::Config)?;
        let config_holder = FileProxy::file(config, config_filepath);
        Ok(config_holder)
    }
}

#[async_trait]
impl<TConfig: Send + Sync + Serialize> ConfigProxy<TConfig> for FileProxy<TConfig> {
    async fn read<'a>(&'a self) -> Box<dyn ConfigReadProxy<TConfig> + 'a> {
        Box::new(FileReadProxy {
            config: self.config.read().await,
        })
    }

    async fn write<'a>(&'a self) -> Box<dyn ConfigWriteProxy<TConfig> + 'a> {
        Box::new(FileWriteProxy {
            config: self.config.write().await,
            config_filepath: self.config_filepath.clone(),
        })
    }
}

impl<TConfig: Send + Sync + Serialize> ConfigReadProxy<TConfig> for FileReadProxy<'_, TConfig> {
    fn get(&self) -> &TConfig {
        &self.config
    }
}

#[async_trait]
impl<TConfig: Send + Sync + Serialize> ConfigWriteProxy<TConfig> for FileWriteProxy<'_, TConfig> {
    fn get(&self) -> &TConfig {
        self.config.deref()
    }

    fn get_mut(&mut self) -> &mut TConfig {
        self.config.deref_mut()
    }

    async fn commit(&self) -> SummaResult<()> {
        let mut vec = Vec::with_capacity(128);
        to_writer(&mut vec, self.config.deref())?;
        tokio::fs::File::create(&self.config_filepath)
            .await
            .map_err(|e| Error::IO((e, Some(self.config_filepath.to_path_buf()))))?
            .write_all(&vec)
            .await
            .map_err(|e| Error::IO((e, Some(self.config_filepath.to_path_buf()))))?;
        Ok(())
    }
}
