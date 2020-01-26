use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, RwLock,
};
use std::time::Duration;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Document, Schema};
use tantivy::tokenizer::{
    LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, Term};

use crate::config::SearchEngineConfig;
use crate::logging::create_logger;
use crate::thread_handler::ThreadHandler;

use super::{SummaTokenizer, STOP_WORDS};

#[derive(Clone)]
pub struct SearchEngine {
    search_engine_config: SearchEngineConfig,
    indices: HashMap<String, IndexKeeper>,
    logger: slog::Logger,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SchemaConfig {
    pub default_fields: Vec<String>,
    pub enabled: bool,
    pub key_field: String,
    pub multi_fields: Vec<String>,
    pub name: String,
    pub schema: Schema,
}

#[derive(Clone)]
pub struct IndexKeeper {
    pub index: Index,
    pub index_reader: IndexReader,
    pub index_writer: Arc<RwLock<IndexWriter>>,
    pub query_parser: QueryParser,
    pub schema_config: SchemaConfig,
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
        log_path: PathBuf,
    ) -> Result<SearchEngine, crate::errors::Error> {
        let mut search_engine = SearchEngine {
            search_engine_config: search_engine_config.clone(),
            indices: HashMap::new(),
            logger: create_logger(&Path::new(&log_path).join("statbox.log"))?,
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
            .map_err(|e| crate::errors::ConfigError::YamlError(e))?;
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

        let index_keeper = IndexKeeper {
            index,
            index_reader,
            index_writer: Arc::new(RwLock::new(index_writer)),
            query_parser,
            schema_config,
        };
        self.indices.insert(name, index_keeper);
        Ok(())
    }
    fn read_schema_files(
        &mut self,
    ) -> Result<(), crate::errors::Error> {
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
        let index_keeper = self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?;
        let searcher = index_keeper.index_reader.searcher();
        let schema_config = index_keeper.schema_config.clone();

        let parsed_query = index_keeper.query_parser.parse_query(&query);
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
    pub fn get_schema(&self, schema_name: &str) -> Result<&Schema, crate::errors::Error> {
        Ok(&self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?
            .schema_config
            .schema)
    }
    pub async fn commit(&self, schema_name: &str) -> Result<(), crate::errors::Error> {
        let index_keeper = self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?;
        index_keeper.index_writer.write().unwrap().commit()?;
        Ok(())
    }
    pub async fn put_document(
        &self,
        schema_name: &str,
        document: Document,
    ) -> Result<(), crate::errors::Error> {
        let index_keeper = self
            .indices
            .get(schema_name)
            .ok_or(crate::errors::Error::UnknownSchemaError)?;
        let id_field = index_keeper
            .schema_config
            .schema
            .get_field(&index_keeper.schema_config.key_field)
            .unwrap();
        {
            let index_writer = index_keeper.index_writer.read().unwrap();
            index_writer.delete_term(Term::from_field_i64(
                id_field,
                document.get_first(id_field).unwrap().i64_value().unwrap(),
            ));
            index_writer.add_document(document);
        }

        Ok(())
    }
    pub fn start_auto_commit_thread(&self) -> ThreadHandler {
        let running = Arc::new(AtomicBool::new(true));
        let running_in_thread = running.clone();

        let timeout = Duration::from_secs(60);
        let logger = self.logger.clone();

        let indices = self.indices.clone();

        let join_handle = Box::new(std::thread::spawn(
            move || -> Result<(), crate::errors::Error> {
                while running_in_thread.load(Ordering::Acquire) {
                    std::thread::sleep(timeout);
                    for (_, index_keeper) in &indices {
                        index_keeper.index_writer.write().unwrap().commit()?;
                    }
                }
                for (_, index_keeper) in &indices {
                    index_keeper.index_writer.write().unwrap().commit()?;
                }
                info!(logger, "exit";
                    "action" => "exit",
                    "mode" => "update_thread",
                );
                Ok(())
            },
        ));
        ThreadHandler::new(join_handle, running)
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
