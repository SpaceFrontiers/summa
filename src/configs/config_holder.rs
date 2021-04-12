use crate::errors::{Error, SummaResult};
use config::{Config, Environment, File};

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

pub trait Persistable {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>) -> SummaResult<Self>
    where
        Self: Sized;
    fn save(&self) -> SummaResult<&Self>;
}

pub struct AutosaveLockWriteGuard<'a, T: Persistable> {
    data: &'a mut T,
}

impl<'a, T: Persistable> AutosaveLockWriteGuard<'a, T> {
    pub fn new(data: &'a mut T) -> Self {
        AutosaveLockWriteGuard { data }
    }
}

impl<'a, T: Persistable> Drop for AutosaveLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.data.save().unwrap();
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
        &mut self.data
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
    pub fn autosave(&mut self) -> AutosaveLockWriteGuard<ConfigHolder<TConfig>> {
        AutosaveLockWriteGuard::new(self)
    }
}

impl<'a, TConfig: Serialize + Deserialize<'a>> Persistable for ConfigHolder<TConfig> {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>) -> SummaResult<ConfigHolder<TConfig>> {
        let mut s = Config::builder();
        if config_filepath.exists() {
            s = s.add_source(File::from(config_filepath));
        }
        if let Some(env_prefix) = env_prefix {
            s = s.add_source(Environment::with_prefix(env_prefix).separator("."));
        }
        let config: TConfig = s.build()?.try_deserialize().map_err(|e| Error::ConfigError(e))?;
        let config_holder = ConfigHolder::file(config, config_filepath);
        Ok(config_holder)
    }

    fn save(&self) -> SummaResult<&Self> {
        match self.config_filepath {
            Some(ref config_filepath) => {
                std::fs::File::create(config_filepath)
                    .and_then(|mut file| file.write_all(&serde_yaml::to_vec(&self.config).unwrap()))
                    .map_err(|e| Error::IOError((e, Some(config_filepath.to_path_buf()))))?;
            }
            None => (),
        };
        Ok(self)
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
        write!(f, "{}", serde_yaml::to_string(&self.config).unwrap())
    }
}
