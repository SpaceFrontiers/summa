use std::collections::HashMap;
use std::default::Default;

use serde::{Deserialize, Serialize};
use summa_proto::proto;
use summa_proto::proto::IndexEngineConfig;

use crate::errors::{BuilderError, SummaResult, ValidationError};

fn return_1() -> usize {
    1
}
fn return_100() -> usize {
    100
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriterThreads {
    SameThread,
    N(u64),
}

impl WriterThreads {
    pub fn threads(&self) -> u64 {
        match self {
            WriterThreads::SameThread => 0,
            WriterThreads::N(n) => *n,
        }
    }
}

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(default, build_fn(error = "BuilderError"))]
pub struct Config {
    #[serde(default = "HashMap::new")]
    pub aliases: HashMap<String, String>,
    #[builder(default = "None")]
    pub autocommit_interval_ms: Option<u64>,
    #[builder(default = "1")]
    #[serde(default = "return_1")]
    pub doc_store_compress_threads: usize,
    #[builder(default = "100")]
    #[serde(default = "return_100")]
    pub doc_store_cache_num_blocks: usize,
    #[serde(default = "HashMap::new")]
    pub indices: HashMap<String, IndexEngineConfig>,
    #[builder(default = "1024 * 1024 * 1024")]
    pub writer_heap_size_bytes: u64,
    #[builder(default = "Some(WriterThreads::N(1))")]
    pub writer_threads: Option<WriterThreads>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            aliases: HashMap::new(),
            autocommit_interval_ms: None,
            doc_store_compress_threads: 1,
            indices: HashMap::new(),
            writer_heap_size_bytes: 1024 * 1024 * 1024,
            writer_threads: Some(WriterThreads::N(1)),
            doc_store_cache_num_blocks: 100,
        }
    }
}

impl Config {
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
        if index_name.is_empty() {
            return Err(ValidationError::EmptyArgument("index_name".to_owned()).into());
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

#[derive(Debug, Clone)]
pub struct QueryParserConfig(pub proto::QueryParserConfig);
impl QueryParserConfig {
    pub fn from_default_fields(default_fields: Vec<String>) -> Self {
        QueryParserConfig(proto::QueryParserConfig {
            default_fields,
            ..Default::default()
        })
    }
    pub fn term_limit(&self) -> usize {
        if self.0.term_limit > 0 {
            self.0.term_limit as usize
        } else {
            16
        }
    }
    pub fn merge(&mut self, other: Self) {
        if !other.0.default_fields.is_empty() {
            self.0.default_fields = other.0.default_fields;
        }
        self.0.field_aliases.extend(other.0.field_aliases);
        self.0.term_field_mapper_configs.extend(other.0.term_field_mapper_configs);
        self.0.field_boosts.extend(other.0.field_boosts);
        self.0.morphology_configs.extend(other.0.morphology_configs);
        if other.0.term_limit > 0 {
            self.0.term_limit = other.0.term_limit;
        }
        if let Some(exact_matches_promoter) = other.0.exact_matches_promoter {
            self.0.exact_matches_promoter = Some(exact_matches_promoter)
        }
        if let Some(default_mode) = other.0.default_mode {
            self.0.default_mode = Some(default_mode)
        }
        if let Some(query_language) = other.0.query_language {
            self.0.query_language = Some(query_language)
        }
    }
}
