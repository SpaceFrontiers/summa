use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};
use serde_yaml::to_writer;

use crate::errors::{Error, SummaResult, ValidationError};

#[async_trait]
pub trait Persistable {
    fn save(&self) -> SummaResult<()>;
    //ToDo
    async fn save_async(&self) -> SummaResult<()> {
        self.save()
    }
}

pub trait Loadable {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>) -> SummaResult<Self>
    where
        Self: Sized;
}

pub struct AutosaveLockWriteGuard<'a, T: Persistable> {
    data: &'a mut T,
}

impl<'a, T: Persistable> Drop for AutosaveLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        match self.data.save() {
            Ok(_) | Err(Error::Validation(ValidationError::MissingPath(_))) => (),
            Err(e) => panic!("{:?}", e),
        };
    }
}

impl<'a, T: Persistable> Deref for AutosaveLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T: Persistable> DerefMut for AutosaveLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

#[derive(Clone, Debug)]
pub struct ConfigHolder<TConfig: Serialize> {
    config: TConfig,
    config_filepath: Option<PathBuf>,
}

impl<'a, TConfig: Serialize + Deserialize<'a>> ConfigHolder<TConfig> {
    pub fn file(config: TConfig, config_filepath: &Path) -> Self {
        ConfigHolder {
            config,
            config_filepath: Some(config_filepath.to_path_buf()),
        }
    }
    pub fn memory(config: TConfig) -> Self {
        ConfigHolder { config, config_filepath: None }
    }
}

impl<'a, TConfig: Serialize + Deserialize<'a>> Loadable for ConfigHolder<TConfig> {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>) -> SummaResult<ConfigHolder<TConfig>> {
        let mut s = Config::builder();
        if !config_filepath.exists() {
            return Err(ValidationError::MissingPath(config_filepath.to_path_buf()).into());
        }
        if config_filepath.exists() {
            s = s.add_source(File::from(config_filepath));
        }
        if let Some(env_prefix) = env_prefix {
            s = s.add_source(Environment::with_prefix(env_prefix).separator("."));
        }
        let config: TConfig = s.build()?.try_deserialize().map_err(Error::Config)?;
        let config_holder = ConfigHolder::file(config, config_filepath);
        Ok(config_holder)
    }
}

impl<TConfig: Serialize> Persistable for ConfigHolder<TConfig> {
    fn save(&self) -> SummaResult<()> {
        if let Some(config_filepath) = &self.config_filepath {
            let mut vec = Vec::with_capacity(128);
            to_writer(&mut vec, &self.config)?;
            std::fs::File::create(config_filepath)
                .and_then(|mut file| file.write_all(&vec))
                .map_err(|e| Error::IO((e, Some(config_filepath.to_path_buf()))))?;
        }
        Ok(())
    }
}

impl<TConfig: Serialize> Deref for ConfigHolder<TConfig> {
    type Target = TConfig;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

impl<TConfig: Serialize> DerefMut for ConfigHolder<TConfig> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.config
    }
}

impl<TConfig: Serialize> std::fmt::Display for ConfigHolder<TConfig> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_yaml::to_string(&self.config).expect("cannot serialize"))
    }
}
