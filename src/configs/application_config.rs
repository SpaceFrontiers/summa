use super::ConfigHolder;
use super::GrpcConfig;
use super::MetricsConfig;
use crate::configs::IndexConfig;
use crate::errors::{SummaResult, ValidationError};
use colored::Colorize;
use parking_lot::RwLock;
use path_absolutize::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use textwrap::indent;

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct ApplicationConfig {
    #[serde(default = "HashMap::new")]
    pub aliases: HashMap<String, String>,
    pub data_path: PathBuf,
    pub debug: bool,
    pub grpc: GrpcConfig,
    #[serde(default = "HashMap::new")]
    pub indices: HashMap<String, IndexConfig>,
    pub log_path: PathBuf,
    pub metrics: MetricsConfig,
}

impl ApplicationConfig {
    pub fn new<P>(data_path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let data_path = Absolutize::absolutize(data_path.as_ref()).unwrap();
        ApplicationConfig {
            aliases: HashMap::new(),
            data_path: data_path.join("bin"),
            debug: true,
            grpc: GrpcConfig::default(),
            indices: HashMap::new(),
            log_path: data_path.join("logs"),
            metrics: MetricsConfig::default(),
        }
    }

    pub fn get_path_for_index_data(&self, index_name: &str) -> PathBuf {
        self.data_path.join(index_name)
    }

    /// Copy aliases for the index
    pub fn get_index_aliases_for_index(&self, index_name: &str) -> Vec<String> {
        self.aliases.iter().filter(|(_, v)| *v == index_name).map(|(k, _)| k.clone()).collect::<Vec<String>>()
    }

    /// Find index by alias
    pub fn resolve_index_alias(&self, alias: &str) -> Option<String> {
        self.aliases.get(alias).cloned()
    }

    /// Set new alias for index
    pub fn set_index_alias(&mut self, alias: &str, index_name: &str) -> SummaResult<Option<String>> {
        if alias == "" {
            Err(ValidationError::EmptyArgument("alias".to_owned()))?
        }
        if !self.indices.contains_key(index_name) {
            Err(ValidationError::MissingIndexError(index_name.to_owned()))?
        }
        Ok(self.aliases.insert(alias.to_owned(), index_name.to_owned()))
    }

    /// Delete all aliases listed in `index_aliases`
    pub fn delete_index_aliases(&mut self, index_aliases: &Vec<String>) {
        for alias in index_aliases {
            self.aliases.remove(alias);
        }
    }
}

impl std::fmt::Display for ApplicationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}\n{}", "Summa Config:".green().bold(), indent(&serde_yaml::to_string(&self).unwrap(), "  "),)
    }
}

#[derive(Clone, Debug)]
pub struct ApplicationConfigHolder(Arc<RwLock<ConfigHolder<ApplicationConfig>>>);

impl ApplicationConfigHolder {
    pub fn new(config_holder: ConfigHolder<ApplicationConfig>) -> ApplicationConfigHolder {
        ApplicationConfigHolder(Arc::new(RwLock::new(config_holder)))
    }
}

impl Default for ApplicationConfigHolder {
    fn default() -> Self {
        ApplicationConfigHolder(Arc::new(RwLock::new(ConfigHolder::memory(ApplicationConfig::default()))))
    }
}

impl Deref for ApplicationConfigHolder {
    type Target = Arc<RwLock<ConfigHolder<ApplicationConfig>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ApplicationConfigHolder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
