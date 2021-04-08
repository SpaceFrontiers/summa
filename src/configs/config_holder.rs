use crate::errors::{Error, SummaResult, ValidationError};
use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

pub trait Persistable {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>, mandatory: bool) -> SummaResult<Self>
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
    config_filepath: PathBuf,
    config: TConfig,
}

impl<'a, TConfig: Serialize + Deserialize<'a>> ConfigHolder<TConfig> {
    pub fn new(config: TConfig, config_filepath: &Path) -> SummaResult<Self> {
        if !config_filepath.exists() {
            Ok(ConfigHolder {
                config_filepath: config_filepath.to_path_buf(),
                config,
            })
        } else {
            Err(Error::ValidationError(ValidationError::ExistingConfigError(config_filepath.to_path_buf())))
        }
    }
    pub fn get_config_filepath(&self) -> &Path {
        &self.config_filepath
    }
    pub fn autosave(&mut self) -> AutosaveLockWriteGuard<ConfigHolder<TConfig>> {
        AutosaveLockWriteGuard::new(self)
    }
}

impl<'a, TConfig: Serialize + Deserialize<'a>> Persistable for ConfigHolder<TConfig> {
    fn from_file(config_filepath: &Path, env_prefix: Option<&str>, mandatory: bool) -> SummaResult<ConfigHolder<TConfig>> {
        let mut s = Config::builder();
        if config_filepath.exists() || mandatory {
            s = s.add_source(File::from(config_filepath));
        }
        if let Some(env_prefix) = env_prefix {
            s = s.add_source(Environment::with_prefix(env_prefix).separator("."));
        }
        let config: TConfig = s.build()?.try_deserialize().map_err(|e| Error::ConfigError(e))?;
        let config_holder = ConfigHolder {
            config_filepath: config_filepath.to_path_buf(),
            config,
        };
        Ok(config_holder)
    }

    fn save(&self) -> SummaResult<&Self> {
        std::fs::File::create(&self.config_filepath)
            .and_then(|mut file| file.write_all(&serde_yaml::to_vec(&self.config).unwrap()))
            .map_err(|e| Error::IOError((e, Some(self.config_filepath.to_path_buf()))))?;
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
