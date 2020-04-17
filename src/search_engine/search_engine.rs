use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Document, Schema};
use tantivy::tokenizer::{
    LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer,
};
use tantivy::{Index, ReloadPolicy};

use crate::config::SearchEngineConfig;
use crate::ThreadHandler;

use super::{SchemaConfig, SchemaEngine, SummaTokenizer, STOP_WORDS};

#[derive(Clone)]
pub struct SearchEngine {
    search_engine_config: SearchEngineConfig,
    indices: HashMap<String, SchemaEngine>,
    logger: slog::Logger,
}

pub struct SearchResponse {
    schema_config: SchemaConfig,
    scored_documents: Vec<ScoredDocument>,
    has_next: bool,
}

pub struct ScoredDocument {
    pub document: Document,
    pub score: f32,
    pub position: u32,
}

impl SearchResponse {
    pub fn to_json(&self) -> Result<Vec<u8>, crate::errors::Error> {
        #[derive(Serialize)]
        #[serde(untagged)]
        enum NamedFieldValue {
            Value(tantivy::schema::Value),
            Vec(Vec<tantivy::schema::Value>),
        }

        #[derive(Serialize)]
        struct JsonScoredDocument {
            schema: String,
            document: BTreeMap<String, NamedFieldValue>,
            score: f32,
        }

        #[derive(Serialize)]
        struct JsonSearchResponse {
            has_next: bool,
            scored_documents: Vec<JsonScoredDocument>,
        }

        let json_scored_documents = self
            .scored_documents
            .iter()
            .map(|scored_document| {
                let named_document = self
                    .schema_config
                    .schema
                    .to_named_doc(&scored_document.document);
                let mut named_document_casted: BTreeMap<String, NamedFieldValue> = BTreeMap::new();
                let schema = &self.schema_config.schema;
                for (_, field_entry) in schema.fields() {
                    match named_document.0.get(field_entry.name()).map(|x| x.get(0)) {
                        Some(Some(value)) => {
                            named_document_casted.insert(
                                field_entry.name().to_string(),
                                NamedFieldValue::Value(value.clone()),
                            );
                        }
                        _ => {}
                    }
                }

                for field_name in self.schema_config.multi_fields.iter() {
                    match named_document.0.get(field_name) {
                        Some(value) => {
                            named_document_casted.insert(
                                field_name.to_string(),
                                NamedFieldValue::Vec(value.to_vec()),
                            );
                        }
                        _ => {}
                    }
                }

                JsonScoredDocument {
                    schema: self.schema_config.name.clone(),
                    document: named_document_casted,
                    score: scored_document.score,
                }
            })
            .collect();

        Ok(serde_json::to_vec(&JsonSearchResponse {
            has_next: self.has_next,
            scored_documents: json_scored_documents,
        })?)
    }
}

impl SearchEngine {
    pub fn get_config(&self) -> &SearchEngineConfig {
        &self.search_engine_config
    }
    pub fn new(
        search_engine_config: &SearchEngineConfig,
        logger: slog::Logger,
    ) -> Result<SearchEngine, crate::errors::Error> {
        let mut search_engine = SearchEngine {
            search_engine_config: search_engine_config.clone(),
            indices: HashMap::new(),
            logger,
        };
        search_engine.read_schema_files()?;
        Ok(search_engine)
    }
    fn add_schema(
        &mut self,
        schema_path: PathBuf,
        data_path: PathBuf,
        writer_threads: usize,
        writer_memory_mb: usize,
    ) -> Result<(), crate::errors::Error> {
        let schema_config_yaml = std::fs::read_to_string(schema_path)?;
        let schema_config: SchemaConfig = serde_yaml::from_str(&schema_config_yaml)
            .map_err(|e| crate::errors::Error::YamlError(e))?;
        if !schema_config.enabled {
            return Ok(());
        }
        let schema = schema_config.schema.clone();
        let name = schema_config.name.clone();

        let index_path = Path::new(&data_path)
            .join("index")
            .join(&schema_config.name);
        let index = open_index(&index_path)?;
        let index_reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()
            .unwrap();
        let index_writer = index
            .writer_with_num_threads(writer_threads, writer_memory_mb * 1024 * 1024)
            .unwrap();

        let default_fields = schema_config
            .default_fields
            .iter()
            .map(|field_name| schema.get_field(&field_name).unwrap())
            .collect();
        let query_parser = QueryParser::for_index(&index, default_fields);
        let (document_sender, document_receiver) = crossbeam_channel::unbounded();

        let schema_engine = SchemaEngine {
            document_receiver,
            document_sender,
            index,
            index_reader,
            index_writer: Arc::new(RwLock::new(index_writer)),
            logger: self.logger.clone(),
            query_parser,
            schema_config,
        };
        self.indices.insert(name, schema_engine);
        Ok(())
    }
    fn read_schema_files(&mut self) -> Result<(), crate::errors::Error> {
        let schema_path = std::path::Path::new(&self.get_config().data_path).join("schema");
        fs::create_dir_all(&schema_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "cannot create schema directory {}",
                    &schema_path.to_str().unwrap()
                ),
            )
        })?;
        for entry in fs::read_dir(&schema_path)? {
            let entry = entry?;
            let schema_path = entry.path();
            if schema_path.is_file() {
                self.add_schema(
                    schema_path,
                    PathBuf::from(&self.get_config().data_path),
                    self.get_config().writer_threads,
                    self.get_config().writer_memory_mb,
                )?;
            }
        }
        Ok(())
    }
    pub async fn search(
        &self,
        schema_name: &str,
        query: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, crate::errors::Error> {
        let schema_engine = self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?;
        let searcher = schema_engine.index_reader.searcher();
        let schema_config = schema_engine.schema_config.clone();

        let parsed_query = schema_engine.query_parser.parse_query(&query);
        let parsed_query = match parsed_query {
            Err(tantivy::query::QueryParserError::FieldDoesNotExist(_)) => {
                return Ok(SearchResponse {
                    schema_config,
                    scored_documents: vec![],
                    has_next: false,
                })
            }
            Err(e) => Err(crate::errors::Error::InvalidSyntaxError((
                e,
                String::from(query),
            ))),
            Ok(r) => Ok(r),
        }?;
        tokio::task::spawn_blocking(move || {
            let top_docs = searcher.search(
                &parsed_query,
                &TopDocs::with_limit(limit + 1).and_offset(offset),
            )?;
            let has_next = top_docs.len() > limit;
            Ok(SearchResponse {
                schema_config,
                scored_documents: top_docs[..std::cmp::min(limit, top_docs.len())]
                    .iter()
                    .enumerate()
                    .map(|(position, (score, doc_address))| {
                        let document = searcher.doc(*doc_address).unwrap();
                        ScoredDocument {
                            document,
                            score: *score,
                            position: position.try_into().unwrap(),
                        }
                    })
                    .collect(),
                has_next,
            })
        })
        .await?
    }
    pub fn start_auto_commit_threads(
        &self,
        timeout: Duration,
        wakeup_timeout: Duration,
    ) -> Vec<ThreadHandler> {
        self.indices
            .iter()
            .map(|(_, v)| v.start_auto_commit_thread(timeout, wakeup_timeout))
            .collect()
    }
    pub fn get_schema(&self, schema_name: &str) -> Result<&Schema, crate::errors::Error> {
        Ok(&self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?
            .schema_config
            .schema)
    }

    pub async fn commit(&self, schema_name: &str) -> Result<(), crate::errors::Error> {
        self.indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?
            .commit()
    }
    pub async fn put_document(
        &self,
        schema_name: &str,
        document: Document,
    ) -> Result<(), crate::errors::Error> {
        let schema_engine = self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?;
        schema_engine.document_sender.send(document)?;
        Ok(())
    }

    pub fn list_schemas(&self) -> Vec<String> {
        let mut result = self.indices.keys().cloned().collect::<Vec<String>>();
        result.sort();
        result
    }
}

fn open_index(path: &PathBuf) -> Result<Index, crate::errors::Error> {
    let index = Index::open_in_dir(path)?;
    index.tokenizers().register(
        "default",
        TextAnalyzer::from(SimpleTokenizer)
            .filter(RemoveLongFilter::limit(200))
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(
                STOP_WORDS.iter().map(|x| String::from(*x)).collect(),
            )),
    );
    index.tokenizers().register(
        "summa",
        TextAnalyzer::from(SummaTokenizer)
            .filter(RemoveLongFilter::limit(200))
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(
                STOP_WORDS.iter().map(|x| String::from(*x)).collect(),
            )),
    );
    Ok(index)
}
