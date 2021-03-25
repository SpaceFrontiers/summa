use super::default_tokenizers::default_tokenizers;
use super::index_writer_holder::IndexWriterHolder;
use crate::configurator::configs::{IndexConfig, IndexConfigHolder};
use crate::configurator::ConfigHolder;
use crate::configurator::Persistable;
use crate::consumers::kafka::KafkaConsumer;
use crate::errors::{Error, SummaResult};
use crate::proto;
use crate::search_engine::IndexUpdater;
use crate::utils::thread_handler::ThreadHandler;
use parking_lot::RwLock;
use std::path::{Path, PathBuf};

use std::sync::Arc;
use std::time::Duration;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Schema;
use tantivy::space_usage::SearcherSpaceUsage;
use tantivy::{Index, IndexReader, IndexSettings};
use tokio::sync::oneshot;
use tokio::time;
use tracing::{info, instrument, warn};

pub struct IndexHolder {
    index_name: String,
    schema: Schema,
    data_path: PathBuf,
    index: Index,
    index_config: IndexConfigHolder,
    index_reader: IndexReader,
    index_updater: Arc<RwLock<IndexUpdater>>,
    autocommit_thread: Option<ThreadHandler>,
}

impl IndexHolder {
    pub(crate) fn new(
        index_name: &str,
        index_config: IndexConfig,
        config_filepath: &Path,
        data_path: &Path,
        schema: &Schema,
        consumers: Vec<KafkaConsumer>,
    ) -> SummaResult<IndexHolder> {
        IndexHolder::create(index_config, config_filepath, data_path, schema)?;
        IndexHolder::open(index_name, config_filepath, data_path, consumers)
    }

    #[instrument]
    fn create(index_config: IndexConfig, config_filepath: &Path, data_path: &Path, schema: &Schema) -> SummaResult<()> {
        let index_config_holder = ConfigHolder::new(index_config.clone(), &config_filepath)?;
        index_config_holder.save()?;
        let settings = IndexSettings {
            docstore_compression: index_config.compression,
            ..Default::default()
        };
        Index::builder().schema(schema.clone()).settings(settings).create_in_dir(data_path)?;
        Ok(())
    }

    #[instrument]
    pub(crate) fn open(index_name: &str, config_filepath: &Path, data_path: &Path, consumers: Vec<KafkaConsumer>) -> SummaResult<IndexHolder> {
        let index_config: IndexConfigHolder = ConfigHolder::from_file(config_filepath, None, true)?;

        let index = Index::open_in_dir(data_path)?;
        let schema = index.schema();
        for (tokenizer_name, tokenizer) in &default_tokenizers() {
            index.tokenizers().register(tokenizer_name, tokenizer.clone())
        }
        let index_reader = index.reader()?;
        let index_writer_holder = IndexWriterHolder::new(index_name, &index, &index_config)?;

        let mut index_updater = IndexUpdater::new(index_writer_holder);
        index_updater.add_consumers(consumers)?;
        let index_updater = Arc::new(RwLock::new(index_updater));

        let autocommit_interval_ms = index_config.autocommit_interval_ms;

        let autocommit_thread = match autocommit_interval_ms {
            Some(interval_ms) => {
                let index_updater = index_updater.clone();
                let (shutdown_trigger, shutdown_tripwire) = oneshot::channel();
                Some(ThreadHandler::new(
                    tokio::spawn(async move {
                        let inner_task = async move {
                            let mut interval = time::interval(Duration::from_millis(interval_ms));
                            loop {
                                interval.tick().await;
                                if let Err(error) = index_updater.write().commit(true).await {
                                    warn!(error = ?error)
                                }
                            }
                        };
                        tokio::select! {
                            _ = inner_task => {}
                            _ = shutdown_tripwire => {
                                info!(action = "shutdown");
                            }
                        }
                        Ok(())
                    }),
                    shutdown_trigger,
                ))
            }
            None => None,
        };

        Ok(IndexHolder {
            index_name: String::from(index_name),
            schema,
            data_path: data_path.to_path_buf(),
            index,
            index_reader,
            index_config,
            index_updater,
            autocommit_thread,
        })
    }

    pub(crate) async fn stop(self) -> SummaResult<()> {
        if let Some(autocommit_thread) = self.autocommit_thread {
            autocommit_thread.stop().await?;
        }
        self.index_updater.write().commit(false).await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub(crate) async fn delete(self) -> SummaResult<()> {
        let data_path = self.data_path.clone();
        self.stop().await?;
        let index_path = data_path.parent().unwrap();
        info!(action = "delete_directory", index_path = ?index_path);
        std::fs::remove_dir_all(&index_path).map_err(|e| Error::IOError((e, Some(index_path.to_path_buf()))))?;
        Ok(())
    }

    pub(crate) fn index_name(&self) -> &str {
        &self.index_name
    }

    pub(crate) fn count(&self) -> u64 {
        self.index_reader.searcher().num_docs()
    }

    pub(crate) fn space_usage(&self) -> SearcherSpaceUsage {
        self.index_reader.searcher().space_usage().unwrap()
    }

    pub(crate) fn index_updater(&self) -> &Arc<RwLock<IndexUpdater>> {
        &self.index_updater
    }

    #[instrument(skip(self))]
    pub(crate) async fn search(&self, query: &str, limit: usize, offset: usize) -> SummaResult<proto::SearchResponse> {
        let schema = self.schema.clone();
        let searcher = self.index_reader.searcher();

        let index_name = self.index_name.to_string();
        let default_fields = self.index_config.default_fields.clone();

        let query_parser = QueryParser::for_index(&self.index, default_fields);
        let parsed_query = query_parser.parse_query(&query);

        let parsed_query = match parsed_query {
            // ToDo: More clever processing
            Err(tantivy::query::QueryParserError::FieldDoesNotExist(_)) => {
                return Ok(proto::SearchResponse {
                    index_name: index_name.to_string(),
                    scored_documents: vec![],
                    has_next: false,
                })
            }
            Err(e) => Err(Error::InvalidSyntaxError((e, query.to_string()))),
            Ok(r) => Ok(r),
        }?;

        let query = query.to_string();
        tokio::task::spawn_blocking(move || {
            let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit + 1).and_offset(offset))?;
            let right_bound = std::cmp::min(limit, top_docs.len());
            let has_next = top_docs.len() > limit;

            info!(action = "search", index_name = ?index_name, query = ?query, limit = limit, offset = offset, count = ?right_bound);
            Ok(proto::SearchResponse {
                scored_documents: top_docs[..right_bound]
                    .iter()
                    .enumerate()
                    .map(|(position, (score, doc_address))| {
                        let document = searcher.doc(*doc_address).unwrap();
                        proto::ScoredDocument {
                            document: schema.to_json(&document),
                            score: *score,
                            position: position.try_into().unwrap(),
                        }
                    })
                    .collect(),
                has_next,
                index_name: index_name.to_string(),
            })
        })
        .await?
    }
}

impl std::fmt::Debug for IndexHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "IndexHolder({:?})", self.index)
    }
}
