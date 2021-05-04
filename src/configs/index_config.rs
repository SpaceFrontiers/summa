use super::ConsumerConfig;
use crate::configs::config_holder::AutosaveLockWriteGuard;
use crate::configs::{ApplicationConfig, ApplicationConfigHolder, ConfigHolder, Persistable};
use crate::errors::SummaResult;
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::path::PathBuf;
use tantivy::schema::Schema;
use tantivy::IndexSortByField;

#[derive(Clone, Serialize, Deserialize)]
pub enum IndexEngine {
    File(PathBuf),
    Memory(Schema),
}

impl Debug for IndexEngine {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IndexEngine::File(_) => "File",
                IndexEngine::Memory(_) => "Memory",
            }
        )
    }
}

#[derive(Builder, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    #[builder(default = "None")]
    pub autocommit_interval_ms: Option<u64>,
    #[builder(default = "tantivy::store::Compressor::Brotli")]
    pub compression: tantivy::store::Compressor,
    #[builder(default = "HashMap::new()")]
    pub consumer_configs: HashMap<String, ConsumerConfig>,
    #[builder(default = "Vec::new()")]
    pub default_fields: Vec<String>,
    pub index_engine: IndexEngine,
    #[builder(default = "None")]
    pub primary_key: Option<String>,
    #[builder(default = "HashSet::new()")]
    pub multi_fields: HashSet<String>,
    #[builder(default = "None")]
    pub sort_by_field: Option<IndexSortByField>,
    #[builder(default = "128 * 1024 * 1024")]
    pub writer_heap_size_bytes: u64,
    #[builder(default = "1")]
    pub writer_threads: u64,
}

impl std::fmt::Debug for IndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &serde_yaml::to_string(&self).unwrap())
    }
}

#[derive(Clone)]
pub struct IndexConfigProxy {
    application_config: ApplicationConfigHolder,
    index_name: String,
}

impl IndexConfigProxy {
    pub fn new(application_config: &ApplicationConfigHolder, index_name: &str) -> IndexConfigProxy {
        IndexConfigProxy {
            application_config: application_config.clone(),
            index_name: index_name.to_owned(),
        }
    }

    pub fn read(&self) -> IndexConfigReadProxy {
        IndexConfigReadProxy {
            application_config: self.application_config.read(),
            index_name: self.index_name.to_owned(),
        }
    }

    pub fn write(&self) -> IndexConfigWriteProxy {
        IndexConfigWriteProxy {
            application_config: self.application_config.write(),
            index_name: self.index_name.to_owned(),
        }
    }

    pub fn delete(self) -> IndexConfig {
        self.application_config.write().autosave().indices.remove(&self.index_name).unwrap()
    }
}

pub struct IndexConfigReadProxy<'a> {
    application_config: RwLockReadGuard<'a, ConfigHolder<ApplicationConfig>>,
    index_name: String,
}

impl<'a> IndexConfigReadProxy<'a> {
    pub fn get(&self) -> &IndexConfig {
        self.application_config.indices.get(&self.index_name).unwrap()
    }

    pub fn application_config(&self) -> &ConfigHolder<ApplicationConfig> {
        self.application_config.deref()
    }
}

impl<'a> Deref for IndexConfigReadProxy<'a> {
    type Target = IndexConfig;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

pub struct IndexConfigWriteProxy<'a> {
    application_config: RwLockWriteGuard<'a, ConfigHolder<ApplicationConfig>>,
    index_name: String,
}

impl<'a> IndexConfigWriteProxy<'a> {
    pub fn get(&self) -> &IndexConfig {
        self.application_config.indices.get(&self.index_name).unwrap()
    }

    pub fn get_mut(&mut self) -> &mut IndexConfig {
        self.application_config.indices.get_mut(&self.index_name).unwrap()
    }

    pub fn application_config(&self) -> &ConfigHolder<ApplicationConfig> {
        self.application_config.deref()
    }

    pub fn autosave(&mut self) -> AutosaveLockWriteGuard<IndexConfigWriteProxy<'a>> {
        AutosaveLockWriteGuard::new(self)
    }
}

impl<'a> Persistable for IndexConfigWriteProxy<'a> {
    fn save(&self) -> SummaResult<&Self> {
        self.application_config.save()?;
        Ok(self)
    }
}

impl<'a> Deref for IndexConfigWriteProxy<'a> {
    type Target = IndexConfig;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}
