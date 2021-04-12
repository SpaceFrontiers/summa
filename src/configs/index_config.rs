use super::KafkaConsumerConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use tantivy::schema::{Field, Schema};
use tantivy::IndexSortByField;

#[derive(Clone, Serialize, Deserialize)]
pub enum IndexEngine {
    File,
    Memory(Schema),
}

impl Debug for IndexEngine {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IndexEngine::File => "File",
                IndexEngine::Memory(_) => "Memory",
            }
        )
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub autocommit_interval_ms: Option<u64>,
    pub compression: tantivy::store::Compressor,
    pub consumer_configs: HashMap<String, KafkaConsumerConfig>,
    pub default_fields: Vec<Field>,
    pub index_engine: IndexEngine,
    pub primary_key: Option<Field>,
    pub sort_by_field: Option<IndexSortByField>,
    pub writer_heap_size_bytes: u64,
    pub writer_threads: u64,
}

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig {
            autocommit_interval_ms: None,
            compression: tantivy::store::Compressor::Brotli,
            consumer_configs: HashMap::new(),
            default_fields: vec![],
            index_engine: IndexEngine::File,
            primary_key: None,
            sort_by_field: None,
            writer_heap_size_bytes: 128 * 1024 * 1024,
            writer_threads: 1,
        }
    }
}

impl std::fmt::Debug for IndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &serde_yaml::to_string(&self).unwrap())
    }
}
