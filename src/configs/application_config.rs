use super::ConfigHolder;
use super::GrpcConfig;
use super::MetricsConfig;
use crate::configs::{GrpcConfigBuilder, IndexConfig, Loadable, MetricsConfigBuilder};
use crate::errors::{Error, SummaResult, ValidationError};
use colored::Colorize;
use parking_lot::RwLock;
use path_absolutize::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::default::Default;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use textwrap::indent;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default)]
pub struct ApplicationConfig {
    #[serde(default = "HashMap::new")]
    pub aliases: HashMap<String, String>,
    #[builder(setter(custom))]
    pub data_path: PathBuf,
    pub debug: bool,
    pub grpc: GrpcConfig,
    #[serde(default = "HashMap::new")]
    pub indices: HashMap<String, IndexConfig>,
    #[builder(setter(custom))]
    pub log_path: PathBuf,
    pub metrics: MetricsConfig,
}

impl Default for ApplicationConfig {
    fn default() -> Self {
        ApplicationConfig {
            aliases: HashMap::new(),
            data_path: PathBuf::new(),
            debug: true,
            grpc: GrpcConfigBuilder::default().build().unwrap(),
            indices: HashMap::new(),
            log_path: PathBuf::new(),
            metrics: MetricsConfigBuilder::default().build().unwrap(),
        }
    }
}

impl ApplicationConfig {
    pub fn get_path_for_index_data(&self, index_name: &str) -> PathBuf {
        self.data_path.join(index_name)
    }

    /// Copy aliases for the index
    pub fn get_index_aliases_for_index(&self, index_name: &str) -> Vec<String> {
        self.aliases
            .iter()
            .filter(|(_, v)| *v == index_name)
            .map(|(k, _)| k.clone())
            .collect::<Vec<String>>()
    }

    /// Find index by alias
    pub fn resolve_index_alias(&self, alias: &str) -> Option<String> {
        self.aliases.get(alias).cloned()
    }

    /// Set new alias for index
    pub fn set_index_alias(&mut self, alias: &str, index_name: &str) -> SummaResult<Option<String>> {
        if alias.is_empty() {
            return Err(ValidationError::EmptyArgument("alias".to_owned()).into());
        }
        if !self.indices.contains_key(index_name) {
            return Err(ValidationError::MissingIndex(index_name.to_owned()).into());
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

impl ApplicationConfigBuilder {
    pub fn data_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.data_path = Some(Absolutize::absolutize(value.as_ref()).unwrap().to_path_buf());
        self
    }

    pub fn logs_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.log_path = Some(Absolutize::absolutize(value.as_ref()).unwrap().to_path_buf());
        self
    }
}

impl std::fmt::Display for ApplicationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}\n{}",
            "Summa Config:".green().bold(),
            indent(&serde_yaml::to_string(&self).unwrap(), "  "),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ApplicationConfigHolder(Arc<RwLock<ConfigHolder<ApplicationConfig>>>);

impl ApplicationConfigHolder {
    pub fn from_path<P: AsRef<Path>>(application_config_filepath: P) -> SummaResult<ApplicationConfigHolder> {
        let application_config = ConfigHolder::<ApplicationConfig>::from_file(application_config_filepath.as_ref(), None)?;
        std::fs::create_dir_all(&application_config.data_path).map_err(|e| Error::IO((e, Some(application_config.data_path.clone()))))?;
        Ok(ApplicationConfigHolder::from_config_holder(application_config))
    }
    pub fn with_path<P: AsRef<Path>>(application_config: ApplicationConfig, application_config_filepath: P) -> ApplicationConfigHolder {
        ApplicationConfigHolder(Arc::new(RwLock::new(ConfigHolder::file(
            application_config,
            application_config_filepath.as_ref(),
        ))))
    }
    pub fn from_path_or<P: AsRef<Path>, F: FnOnce() -> ApplicationConfig>(
        application_config_filepath: P,
        application_config_fn: F,
    ) -> SummaResult<ApplicationConfigHolder> {
        if application_config_filepath.as_ref().exists() {
            ApplicationConfigHolder::from_path(application_config_filepath)
        } else {
            Ok(ApplicationConfigHolder::with_path(application_config_fn(), application_config_filepath))
        }
    }
    pub fn from_config(application_config: ApplicationConfig) -> ApplicationConfigHolder {
        ApplicationConfigHolder(Arc::new(RwLock::new(ConfigHolder::memory(application_config))))
    }
    pub fn from_config_holder(config_holder: ConfigHolder<ApplicationConfig>) -> ApplicationConfigHolder {
        ApplicationConfigHolder(Arc::new(RwLock::new(config_holder)))
    }
}

impl Default for ApplicationConfigHolder {
    fn default() -> Self {
        ApplicationConfigHolder(Arc::new(RwLock::new(ConfigHolder::memory(
            ApplicationConfigBuilder::default().build().unwrap(),
        ))))
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

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::configs::GrpcConfigBuilder;
    use crate::configs::MetricsConfigBuilder;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static BASE_PORT: AtomicUsize = AtomicUsize::new(50000);

    pub(crate) fn create_test_application_config(data_path: &Path) -> ApplicationConfig {
        ApplicationConfigBuilder::default()
            .debug(true)
            .data_path(data_path.to_path_buf())
            .grpc(
                GrpcConfigBuilder::default()
                    .endpoint(format!("127.0.0.1:{}", BASE_PORT.fetch_add(1, Ordering::Relaxed)))
                    .build()
                    .unwrap(),
            )
            .metrics(
                MetricsConfigBuilder::default()
                    .endpoint(format!("127.0.0.1:{}", BASE_PORT.fetch_add(1, Ordering::Relaxed)))
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap()
    }

    pub(crate) fn create_test_application_config_holder(data_path: &Path) -> ApplicationConfigHolder {
        ApplicationConfigHolder::from_config(create_test_application_config(&data_path))
    }
}
