use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
#[cfg(feature = "index-updater")]
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tantivy::schema::Schema;
use tokio::sync::{RwLockReadGuard, RwLockWriteGuard};

use super::ConsumerConfig;
use crate::configs::ConfigProxy;
use crate::configs::ConfigReadProxy;
use crate::configs::ConfigWriteProxy;
use crate::configs::{ApplicationConfig, ApplicationConfigHolder, ConfigHolder, Persistable};
use crate::directories::Header;
use crate::errors::{BuilderError, SummaResult, ValidationError};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkedCacheConfig {
    pub chunk_size: usize,
    pub cache_size: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub method: String,
    pub url_template: String,
    pub headers_template: Vec<Header>,
    pub chunked_cache_config: Option<ChunkedCacheConfig>,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum IndexEngine {
    #[cfg(feature = "index-updater")]
    File(PathBuf),
    Memory(Schema),
    Remote(NetworkConfig),
}

impl Debug for IndexEngine {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                #[cfg(feature = "index-updater")]
                IndexEngine::File(_) => "File",
                IndexEngine::Memory(_) => "Memory",
                IndexEngine::Remote(_) => "Remote",
            }
        )
    }
}

#[derive(Builder, Clone, Serialize, Deserialize)]
#[builder(build_fn(error = "BuilderError"))]
pub struct IndexConfig {
    #[builder(default = "None")]
    pub autocommit_interval_ms: Option<u64>,
    #[builder(default = "HashMap::new()")]
    pub consumer_configs: HashMap<String, ConsumerConfig>,
    #[builder(default = "Vec::new()")]
    pub default_fields: Vec<String>,
    pub index_engine: IndexEngine,
    #[builder(default = "None")]
    pub primary_key: Option<String>,
    #[builder(default = "HashSet::new()")]
    pub multi_fields: HashSet<String>,
    #[builder(default = "128 * 1024 * 1024")]
    pub writer_heap_size_bytes: u64,
    #[builder(default = "1")]
    pub writer_threads: u64,
}

impl Debug for IndexConfig {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", &serde_yaml::to_string(&self).expect("Cannot serialize"))
    }
}

#[derive(Clone)]
pub struct IndexConfigFilePartProxy {
    application_config: ApplicationConfigHolder,
    index_name: String,
}

impl IndexConfigFilePartProxy {
    pub fn new(application_config: &ApplicationConfigHolder, index_name: &str) -> IndexConfigFilePartProxy {
        IndexConfigFilePartProxy {
            application_config: application_config.clone(),
            index_name: index_name.to_owned(),
        }
    }
}

#[async_trait]
impl ConfigProxy<IndexConfig> for IndexConfigFilePartProxy {
    async fn read<'a>(&'a self) -> Box<dyn ConfigReadProxy<IndexConfig> + 'a> {
        Box::new(IndexConfigFilePartReadProxy {
            application_config: self.application_config.read().await,
            index_name: self.index_name.to_owned(),
        })
    }

    async fn write<'a>(&'a self) -> Box<dyn ConfigWriteProxy<IndexConfig> + 'a> {
        Box::new(IndexConfigFilePartWriteProxy {
            application_config: self.application_config.write().await,
            index_name: self.index_name.to_owned(),
        })
    }

    async fn delete(self: Arc<Self>) -> SummaResult<IndexConfig> {
        let mut application_config = self.application_config.write().await;
        let index_config = application_config
            .indices
            .remove(&self.index_name)
            .ok_or_else(|| ValidationError::MissingIndex(self.index_name.to_string()))?;
        application_config.save()?;
        Ok(index_config)
    }
}

pub struct IndexConfigFilePartReadProxy<'a> {
    application_config: RwLockReadGuard<'a, ConfigHolder<ApplicationConfig>>,
    index_name: String,
}

impl<'a> IndexConfigFilePartReadProxy<'a> {
    pub fn application_config(&self) -> &ConfigHolder<ApplicationConfig> {
        self.application_config.deref()
    }
}

impl<'a> ConfigReadProxy<IndexConfig> for IndexConfigFilePartReadProxy<'a> {
    fn get(&self) -> &IndexConfig {
        self.application_config
            .indices
            .get(&self.index_name)
            .ok_or_else(|| ValidationError::MissingIndex(self.index_name.to_string()))
            .expect("missing index")
    }
}

impl<'a> Deref for IndexConfigFilePartReadProxy<'a> {
    type Target = IndexConfig;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

pub struct IndexConfigFilePartWriteProxy<'a> {
    application_config: RwLockWriteGuard<'a, ConfigHolder<ApplicationConfig>>,
    index_name: String,
}

impl<'a> IndexConfigFilePartWriteProxy<'a> {
    pub fn application_config(&self) -> &ConfigHolder<ApplicationConfig> {
        self.application_config.deref()
    }
}

impl<'a> ConfigWriteProxy<IndexConfig> for IndexConfigFilePartWriteProxy<'a> {
    fn get(&self) -> &IndexConfig {
        &self.application_config.indices[&self.index_name]
    }
    fn get_mut(&mut self) -> &mut IndexConfig {
        self.application_config.indices.get_mut(&self.index_name).expect("Not found IndexConfig")
    }
    fn commit(&self) -> SummaResult<()> {
        self.application_config.save()
    }
}

impl<'a> Deref for IndexConfigFilePartWriteProxy<'a> {
    type Target = IndexConfig;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}
