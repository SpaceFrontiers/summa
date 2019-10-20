use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::QueryParser;
use tantivy::schema::{Document, Schema};
use tantivy::tokenizer::{
    LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer,
};
use tantivy::{Index, IndexReader, ReloadPolicy};

use crate::config::SearchEngineConfig;
use crate::logging::create_logger;

use super::STOP_WORDS;

#[derive(Clone)]
pub struct SearchEngine {
    indices: HashMap<String, IndexKeeper>,
    logger: slog::Logger,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SchemaConfig {
    pub default_fields: Vec<String>,
    pub enabled: bool,
    pub multi_fields: Vec<String>,
    pub name: String,
    pub schema: Schema,
}

#[derive(Clone)]
pub struct IndexKeeper {
    pub index: Index,
    pub index_reader: IndexReader,
    pub query_parser: QueryParser,
    pub schema_config: SchemaConfig,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub documents: Vec<SearchResponseDocument>,
    pub has_next: bool,
}

#[derive(Serialize)]
pub struct SearchResponseDocument {
    pub document: BTreeMap<String, NamedFieldValue>,
    pub score: f32,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum NamedFieldValue {
    Value(tantivy::schema::Value),
    Vec(Vec<tantivy::schema::Value>)
}

impl IndexKeeper {
    fn to_named_doc(&self, document: &Document) -> BTreeMap<String, NamedFieldValue> {
        // ToDo: move it to Tantivy to eliminate copying
        let named_document = self.schema_config.schema.to_named_doc(document);
        let mut named_document_casted: BTreeMap<String, NamedFieldValue> = BTreeMap::new();
        let schema = &self.schema_config.schema;

        for (_, field_entry) in schema.fields() {
            match named_document.0.get(field_entry.name()).map(|x| x.get(0)) {
                Some(Some(value)) => {
                    named_document_casted.insert(
                        field_entry.name().to_string(),
                        NamedFieldValue::Value(value.clone())
                    );
                },
                _ => {}
            }
        }
        for field_name in self.schema_config.multi_fields.iter() {
            match named_document.0.get(field_name) {
                Some(value) => {
                    named_document_casted.insert(
                        field_name.to_string(),
                        NamedFieldValue::Vec(value.to_vec())
                    );
                },
                _ => {}
            }
        }
        named_document_casted
    }
}

impl SearchEngine {
    pub fn new(
        search_engine_config: &SearchEngineConfig,
        log_path: PathBuf,
    ) -> Result<SearchEngine, crate::errors::Error> {
        let mut search_engine = SearchEngine {
            indices: HashMap::new(),
            logger: create_logger(&Path::new(&log_path).join("statbox.log"))?,
        };
        search_engine.read_schema_files(search_engine_config)?;
        Ok(search_engine)
    }
    fn add_schema(
        &mut self,
        schema_path: PathBuf,
        data_path: PathBuf,
    ) -> Result<(), crate::errors::Error> {
        let schema_config_yaml = std::fs::read_to_string(schema_path)?;
        let schema_config: SchemaConfig = serde_yaml::from_str(&schema_config_yaml)
            .map_err(|e| crate::errors::ConfigError::YamlError(e))?;
        if !schema_config.enabled {
            return Ok(());
        }
        let schema = schema_config.schema.clone();
        let name = schema_config.name.clone();

        let index_path = Path::new(&data_path).join("index").join(&schema_config.name);
        let index = create_index(&index_path, &schema)?;
        let index_reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()
            .unwrap();
        let default_fields = schema_config
            .default_fields
            .iter()
            .map(|field_name| schema.get_field(&field_name).unwrap())
            .collect();
        let query_parser = QueryParser::for_index(&index, default_fields);

        let index_keeper = IndexKeeper {
            index,
            index_reader,
            query_parser,
            schema_config,
        };
        self.indices.insert(name, index_keeper);
        Ok(())
    }
    fn read_schema_files(
        &mut self,
        search_engine_config: &SearchEngineConfig,
    ) -> Result<(), crate::errors::Error> {
        let schema_path = std::path::Path::new(&search_engine_config.data_path).join("schema");
        fs::create_dir_all(&schema_path).map_err(
            |e| std::io::Error::new(
                e.kind(),
                format!("cannot create schema directory {}", &schema_path.to_str().unwrap())
            )
        )?;
        for entry in fs::read_dir(&schema_path)? {
            let entry = entry?;
            let schema_path = entry.path();
            if schema_path.is_file() {
                self.add_schema(schema_path, PathBuf::from(&search_engine_config.data_path))?;
            }
        }
        Ok(())
    }
    fn reindex(&self) {

    }
    pub fn search(
        &self,
        schema_name: &str,
        query: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, crate::errors::Error> {
        let index_keeper = self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?;
        let searcher = index_keeper.index_reader.searcher();
        let parsed_query = index_keeper
            .query_parser
            .parse_query(&query)
            .map_err(|e| crate::errors::Error::InvalidSyntaxError((e, String::from(query))))?;
        let top_docs = searcher.search(
            &parsed_query,
            &TopDocs::with_limit(limit + 1).and_offset(offset),
        )?;
        let has_next = top_docs.len() > limit;

        let mut documents = vec![];
        for (score, doc_address) in top_docs {
            let document = searcher.doc(doc_address)?;
            documents.push(SearchResponseDocument {
                document: index_keeper.to_named_doc(&document),
                score,
            })
        }
        let search_response = SearchResponse {
            documents,
            has_next,
        };

        Ok(search_response)
    }
}

fn create_index(path: &PathBuf, schema: &Schema) -> Result<Index, crate::errors::Error> {
    let mut index = Index::open_or_create(
        MmapDirectory::open(path).map_err(
            |_e| std::io::Error::new(std::io::ErrorKind::Other, "cannot mmap directory")
        )?,
        schema.clone()
    )?;
    index.set_default_multithread_executor().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::Other, "cannot set multithread executor")
    })?;
    index.tokenizers().register(
        "default",
        TextAnalyzer::from(SimpleTokenizer)
            .filter(RemoveLongFilter::limit(200))
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(
                STOP_WORDS.iter().map(|x| String::from(*x)).collect(),
            )),
    );
    Ok(index)
}
