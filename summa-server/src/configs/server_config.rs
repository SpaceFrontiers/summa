use std::collections::HashMap;
use std::default::Default;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use path_absolutize::*;
use serde::{Deserialize, Serialize};
use summa_core::configs::{ApplicationConfig, ConfigProxy, DirectProxy, FileProxy, Loadable};
use summa_core::errors::BuilderError;

use super::{GrpcConfig, GrpcConfigBuilder};
use super::{MetricsConfig, MetricsConfigBuilder};
use crate::configs::{ConsumerConfig, IpfsConfig};
use crate::errors::{SummaServerResult, ValidationError};

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct ServerConfig {
    #[serde(default = "HashMap::new")]
    pub aliases: HashMap<String, String>,
    #[builder(setter(custom))]
    pub data_path: PathBuf,
    pub debug: bool,
    pub grpc: GrpcConfig,
    pub ipfs: Option<IpfsConfig>,
    #[builder(setter(custom))]
    pub log_path: PathBuf,
    pub metrics: MetricsConfig,

    #[builder(default = "HashMap::new()")]
    pub consumer_configs: HashMap<String, ConsumerConfig>,

    #[serde(default)]
    pub application: ApplicationConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            aliases: HashMap::new(),
            data_path: PathBuf::new(),
            debug: true,
            grpc: GrpcConfigBuilder::default().build().expect("cannot build default config"),
            ipfs: None,
            log_path: PathBuf::new(),
            metrics: MetricsConfigBuilder::default().build().expect("cannot build default config"),
            consumer_configs: HashMap::new(),
            application: ApplicationConfig::default(),
        }
    }
}

impl ServerConfig {
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
    pub fn set_index_alias(&mut self, alias: &str, index_name: &str) -> SummaServerResult<Option<String>> {
        if alias.is_empty() {
            return Err(ValidationError::EmptyArgument("alias".to_owned()).into());
        }
        if index_name.is_empty() {
            return Err(ValidationError::EmptyArgument("index_name".to_owned()).into());
        }
        if !self.application.indices.contains_key(index_name) {
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

impl ServerConfigBuilder {
    pub fn data_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.data_path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }

    pub fn logs_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.log_path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }
}

pub struct ServerConfigHolder {}

impl ServerConfigHolder {
    pub fn from_path<P: AsRef<Path>>(server_config_filepath: P) -> SummaServerResult<Arc<dyn ConfigProxy<ServerConfig>>> {
        let server_config = FileProxy::<ServerConfig>::from_file(server_config_filepath.as_ref(), None)?;
        Ok(Arc::new(server_config))
    }
    pub fn with_path<P: AsRef<Path>>(server_config: ServerConfig, server_config_filepath: P) -> Arc<dyn ConfigProxy<ServerConfig>> {
        Arc::new(FileProxy::file(server_config, server_config_filepath.as_ref()))
    }
    pub fn from_path_or<P: AsRef<Path>, F: FnOnce() -> ServerConfig>(
        server_config_filepath: P,
        server_config_fn: F,
    ) -> SummaServerResult<Arc<dyn ConfigProxy<ServerConfig>>> {
        if server_config_filepath.as_ref().exists() {
            ServerConfigHolder::from_path(server_config_filepath)
        } else {
            Ok(ServerConfigHolder::with_path(server_config_fn(), server_config_filepath))
        }
    }
    pub fn from_config(server_config: ServerConfig) -> Arc<dyn ConfigProxy<ServerConfig>> {
        Arc::new(DirectProxy::new(server_config))
    }
}

pub mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::configs::GrpcConfigBuilder;
    use crate::configs::MetricsConfigBuilder;

    static BASE_PORT: AtomicUsize = AtomicUsize::new(50000);

    pub fn create_test_server_config(data_path: &Path) -> ServerConfig {
        ServerConfigBuilder::default()
            .debug(true)
            .data_path(data_path)
            .grpc(
                GrpcConfigBuilder::default()
                    .endpoint(format!("127.0.0.1:{}", BASE_PORT.fetch_add(1, Ordering::Relaxed)))
                    .build()
                    .expect("cannot create grpc config"),
            )
            .metrics(
                MetricsConfigBuilder::default()
                    .endpoint(format!("127.0.0.1:{}", BASE_PORT.fetch_add(1, Ordering::Relaxed)))
                    .build()
                    .expect("cannot create metrics config"),
            )
            .build()
            .expect("cannot create server config")
    }

    pub fn create_test_server_config_holder(data_path: &Path) -> Arc<dyn ConfigProxy<ServerConfig>> {
        ServerConfigHolder::from_config(create_test_server_config(data_path))
    }
}
