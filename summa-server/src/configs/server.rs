use std::collections::HashMap;
use std::default::Default;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use path_absolutize::*;
use serde::{Deserialize, Serialize};
use summa_core::configs::{ConfigProxy, DirectProxy, FileProxy, Loadable};
use summa_core::errors::BuilderError;

use crate::errors::SummaServerResult;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    /// Base path for storing new indices and setting dependent components on initalization
    #[builder(setter(custom))]
    pub data_path: PathBuf,
    /// Enable debugging logs to `stdout`
    pub debug: bool,
    /// Summa API configuration
    pub api: crate::configs::api::Config,
    /// Path to directory with logs
    #[builder(setter(custom))]
    pub log_path: PathBuf,
    /// Summa Metrics configuration
    pub metrics: Option<crate::configs::metrics::Config>,
    /// Kafka consumers
    #[builder(default = "HashMap::new()")]
    pub consumers: HashMap<String, crate::configs::consumer::Config>,
    /// Summa options
    #[serde(default)]
    pub core: summa_core::configs::core::Config,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            data_path: PathBuf::new(),
            debug: true,
            api: crate::configs::api::Config::default(),
            log_path: PathBuf::new(),
            metrics: Some(crate::configs::metrics::Config::default()),
            consumers: HashMap::new(),
            core: summa_core::configs::core::Config::default(),
        }
    }
}

impl Config {
    pub fn get_path_for_index_data(&self, index_name: &str) -> PathBuf {
        self.data_path.join(index_name)
    }
}

impl ConfigBuilder {
    pub fn data_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.data_path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }

    pub fn logs_path<P: AsRef<Path>>(&mut self, value: P) -> &mut Self {
        self.log_path = Some(Absolutize::absolutize(value.as_ref()).expect("cannot get cwd").to_path_buf());
        self
    }
}

pub struct ConfigHolder {}

impl ConfigHolder {
    pub fn from_path<P: AsRef<Path>>(server_config_filepath: P) -> SummaServerResult<Arc<dyn ConfigProxy<Config>>> {
        let server_config = FileProxy::<Config>::from_file(server_config_filepath.as_ref(), None)?;
        Ok(Arc::new(server_config))
    }
    pub fn with_path<P: AsRef<Path>>(server_config: Config, server_config_filepath: P) -> Arc<dyn ConfigProxy<Config>> {
        Arc::new(FileProxy::file(server_config, server_config_filepath.as_ref()))
    }
    pub fn from_path_or<P: AsRef<Path>, F: FnOnce() -> Config>(
        server_config_filepath: P,
        server_config_fn: F,
    ) -> SummaServerResult<Arc<dyn ConfigProxy<Config>>> {
        if server_config_filepath.as_ref().exists() {
            ConfigHolder::from_path(server_config_filepath)
        } else {
            Ok(ConfigHolder::with_path(server_config_fn(), server_config_filepath))
        }
    }
    pub fn from_config(server_config: Config) -> Arc<dyn ConfigProxy<Config>> {
        Arc::new(DirectProxy::new(server_config))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::utils::tests::acquire_free_port;

    pub fn create_test_server_config(data_path: &Path) -> Config {
        ConfigBuilder::default()
            .debug(true)
            .data_path(data_path)
            .api(
                crate::configs::api::ConfigBuilder::default()
                    .grpc_endpoint(format!("127.0.0.1:{}", acquire_free_port()))
                    .build()
                    .expect("cannot create api config"),
            )
            .metrics(Some(
                crate::configs::metrics::ConfigBuilder::default()
                    .endpoint(format!("127.0.0.1:{}", acquire_free_port()))
                    .build()
                    .expect("cannot create metrics config"),
            ))
            .build()
            .expect("cannot create server config")
    }

    pub fn create_test_server_config_holder(data_path: &Path) -> Arc<dyn ConfigProxy<Config>> {
        ConfigHolder::from_config(create_test_server_config(data_path))
    }
}
