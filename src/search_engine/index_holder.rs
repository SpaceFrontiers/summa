use super::default_tokenizers::default_tokenizers;
use super::index_writer_holder::IndexWriterHolder;
use crate::configs::Persistable;
use crate::configs::{ConfigHolder, KafkaConsumerConfig};
use crate::configs::{IndexConfig, IndexConfigHolder};
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use crate::search_engine::index_layout::IndexLayout;
use crate::search_engine::IndexUpdater;
use crate::utils::sync::{Handler, OwningHandler};
use crate::utils::thread_handler::ThreadHandler;
use std::time::Duration;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema};
use tantivy::{Index, IndexReader, IndexSettings, LeasedItem, Opstamp, Searcher};
use tokio::sync::oneshot;
use tokio::time;
use tracing::{info, info_span, instrument, warn, Instrument};

pub struct IndexHolder {
    index_name: String,
    index_layout: IndexLayout,
    default_fields: Vec<Field>,
    schema: Schema,
    index: Index,
    index_reader: IndexReader,
    /// All modifying operations are isolated inside `index_updater`
    index_updater: OwningHandler<IndexUpdater>,
    autocommit_thread: Option<ThreadHandler>,
}

impl IndexHolder {
    pub(crate) fn new(index_name: &str, index_config: IndexConfig, index_layout: &IndexLayout, schema: &Schema) -> SummaResult<IndexHolder> {
        IndexHolder::create(index_config, index_layout, schema)?;
        IndexHolder::open(index_name, index_layout)
    }

    #[instrument]
    fn create(index_config: IndexConfig, index_layout: &IndexLayout, schema: &Schema) -> SummaResult<()> {
        let index_config_holder = ConfigHolder::new(index_config.clone(), index_layout.config_filepath())?;
        index_config_holder.save()?;
        let settings = IndexSettings {
            docstore_compression: index_config.compression,
            sort_by_field: index_config.sort_by_field,
        };
        Index::builder().schema(schema.clone()).settings(settings).create_in_dir(index_layout.data_path())?;
        Ok(())
    }

    #[instrument(skip(index_layout))]
    pub(crate) fn open(index_name: &str, index_layout: &IndexLayout) -> SummaResult<IndexHolder> {
        let index_config: IndexConfigHolder = ConfigHolder::from_file(index_layout.config_filepath(), None, true)?;

        let index = Index::open_in_dir(index_layout.data_path())?;
        let schema = index.schema();
        for (tokenizer_name, tokenizer) in &default_tokenizers() {
            index.tokenizers().register(tokenizer_name, tokenizer.clone())
        }

        let autocommit_interval_ms = index_config.autocommit_interval_ms.clone();
        let default_fields = index_config.default_fields.clone();
        let index_reader = index.reader()?;
        let index_writer_holder = IndexWriterHolder::new(index_name, &index, &index_config)?;
        let index_updater = OwningHandler::new(IndexUpdater::new(index_config, index_writer_holder)?);

        let autocommit_thread = match autocommit_interval_ms {
            Some(interval_ms) => {
                let index_updater = index_updater.handler().clone();
                let (shutdown_trigger, mut shutdown_tripwire) = oneshot::channel();
                let mut tick_task = time::interval(Duration::from_millis(interval_ms));
                Some(ThreadHandler::new(
                    tokio::spawn(
                        async move {
                            info!(action = "spawning_autocommit_thread", interval_ms = interval_ms);
                            // The first tick ticks immediately so we skip it
                            tick_task.tick().await;
                            loop {
                                tokio::select! {
                                    _ = tick_task.tick() => {
                                        info!(action = "autocommit_thread_tick");
                                        index_updater.try_commit_and_log(true).await;
                                    }
                                    _ = &mut shutdown_tripwire => {
                                        info!(action = "shutdown_autocommit_thread");
                                        break;
                                    }
                                }
                            }
                            Ok(())
                        }
                        .instrument(info_span!(parent: None, "autocommit_thread")),
                    ),
                    shutdown_trigger,
                ))
            }
            None => None,
        };

        Ok(IndexHolder {
            index_name: String::from(index_name),
            default_fields,
            schema,
            index_layout: index_layout.clone(),
            index,
            index_reader,
            index_updater,
            autocommit_thread,
        })
    }

    pub(crate) async fn create_consumer(&self, consumer_name: &str, consumer_config: KafkaConsumerConfig) -> SummaResult<()> {
        self.index_updater().create_consumer(consumer_name, consumer_config).await?;
        Ok(())
    }

    pub(crate) async fn delete_consumer(&self, consumer_name: &str) -> SummaResult<()> {
        self.index_updater().delete_consumer(consumer_name).await?;
        Ok(())
    }

    pub(crate) async fn stop(self) -> SummaResult<()> {
        if let Some(autocommit_thread) = self.autocommit_thread {
            autocommit_thread.stop().await?;
        }
        self.index_updater.into_inner().last_commit().await?;
        Ok(())
    }

    #[instrument(skip_all)]
    pub(crate) async fn delete(self) -> SummaResult<()> {
        let index_layout = self.index_layout.clone();
        self.stop().await?;
        index_layout.delete().await?;
        Ok(())
    }

    pub(crate) fn get_consumer_config(&self, consumer_name: &str) -> SummaResult<KafkaConsumerConfig> {
        Ok(self
            .index_updater
            .get_consumer_config(consumer_name)
            .ok_or_else(|| ValidationError::MissingConsumerError(consumer_name.to_string()))?)
    }

    pub(crate) fn get_consumers_names(&self) -> Vec<String> {
        self.index_updater.get_consumers_names()
    }

    pub(crate) fn index_name(&self) -> &str {
        &self.index_name
    }

    fn index_updater(&self) -> Handler<IndexUpdater> {
        self.index_updater.handler()
    }

    pub(crate) fn schema(&self) -> &Schema {
        &self.schema
    }

    pub(crate) fn searcher(&self) -> LeasedItem<Searcher> {
        self.index_reader.searcher()
    }

    pub async fn delete_all_consumers(&self) -> SummaResult<()> {
        self.index_updater().delete_all_consumers().await
    }

    pub(crate) async fn index_document(&self, document: &[u8], reindex: bool) -> SummaResult<Opstamp> {
        self.index_updater().index_document(document, reindex).await
    }

    pub async fn commit(&self) -> SummaResult<Opstamp> {
        self.index_updater().commit().await
    }

    pub async fn try_commit_and_log(&self, restart_consumers: bool) -> Option<Opstamp> {
        self.index_updater().try_commit_and_log(restart_consumers).await
    }

    pub(crate) async fn search(&self, query: &str, limit: usize, offset: usize) -> SummaResult<proto::SearchResponse> {
        let schema = self.schema.clone();
        let searcher = self.index_reader.searcher();

        let index_name = self.index_name.to_string();
        let default_fields = self.default_fields.clone();

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
        info!(target: "query", index_name=?index_name, query = ?query, limit = limit, offset = offset);

        tokio::task::spawn_blocking(move || {
            let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit + 1).and_offset(offset))?;
            let right_bound = std::cmp::min(limit, top_docs.len());
            let has_next = top_docs.len() > limit;

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
