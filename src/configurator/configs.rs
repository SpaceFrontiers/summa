use crate::configurator::ConfigHolder;
use crate::errors::{Error, SummaResult};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tantivy::schema::Field;
use textwrap::indent;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApplicationConfig {
    pub data_path: PathBuf,
    pub debug: bool,
    pub grpc: GrpcConfig,
    pub log_path: PathBuf,
    pub metrics: MetricsConfig,
}

impl Default for ApplicationConfig {
    fn default() -> Self {
        ApplicationConfig {
            data_path: PathBuf::from("./data/bin"),
            debug: true,
            grpc: GrpcConfig::default(),
            log_path: PathBuf::from("./data/logs"),
            metrics: MetricsConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrpcConfig {
    pub endpoint: String,
    pub workers: usize,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        GrpcConfig {
            endpoint: "127.0.0.1:8082".to_string(),
            workers: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub endpoint: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        MetricsConfig {
            endpoint: "127.0.0.1:8084".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexConfig {
    pub autocommit_interval_ms: Option<u64>,
    pub compression: tantivy::store::Compressor,
    pub default_fields: Vec<Field>,
    pub primary_key: Option<Field>,
    pub writer_heap_size_bytes: u64,
    pub writer_threads: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KafkaConsumerConfig {
    pub bootstrap_servers: Vec<String>,
    pub create_topics: bool,
    pub delete_topics: bool,
    pub group_id: String,
    pub index_name: String,
    pub topics: Vec<String>,
    pub threads: u32,
}

impl KafkaConsumerConfig {
    pub fn new(bootstrap_servers: &Vec<String>, group_id: &str, index_name: &str, mut threads: u32, topics: &Vec<String>) -> SummaResult<KafkaConsumerConfig> {
        if threads == 0 {
            threads = 1;
        }
        Ok(KafkaConsumerConfig {
            bootstrap_servers: bootstrap_servers.clone(),
            create_topics: true,
            delete_topics: true,
            group_id: group_id.to_string(),
            index_name: index_name.to_string(),
            threads: threads.try_into().map_err(|_| Error::InvalidConfigError("`threads` must be u32 sized".to_string()))?,
            topics: topics.clone(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "HashMap::new")]
    pub aliases: HashMap<String, String>,
    #[serde(default = "HashMap::new")]
    pub consumer_configs: HashMap<String, KafkaConsumerConfig>,
}

impl std::fmt::Display for ApplicationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}\n{}", "Summa Config:".green().bold(), indent(&serde_yaml::to_string(&self).unwrap(), "  "),)
    }
}

pub type ApplicationConfigHolder = ConfigHolder<ApplicationConfig>;
pub type IndexConfigHolder = ConfigHolder<IndexConfig>;
pub type RuntimeConfigHolder = ConfigHolder<RuntimeConfig>;
