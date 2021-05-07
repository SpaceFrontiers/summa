use super::default_tokenizers::default_tokenizers;
use super::index_writer_holder::IndexWriterHolder;
use crate::configs::IndexEngine;
use crate::configs::{IndexConfig, IndexConfigProxy};
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use crate::search_engine::collectors::ReservoirSampling;
use crate::search_engine::fruit_extractors::{self, FruitExtractor};
use crate::search_engine::query_parser::QueryParser;
use crate::search_engine::IndexUpdater;
use crate::utils::sync::{Handler, OwningHandler};
use crate::utils::thread_handler::ThreadHandler;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::ops::Deref;
use std::time::Duration;
use tantivy::collector::{Count, MultiCollector, TopDocs};
use tantivy::schema::{Field, Schema};
use tantivy::{Index, IndexReader, IndexSettings, Opstamp, ReloadPolicy};
use tokio::fs::remove_dir_all;
use tokio::sync::oneshot;
use tokio::time;
use tracing::{info, info_span, instrument, warn, Instrument};

pub struct IndexHolder {
    index_name: String,
    index_config_proxy: IndexConfigProxy,
    cached_schema: Schema,
    index_reader: IndexReader,
    query_parser: QueryParser,
    multi_fields: HashSet<Field>,
    /// All modifying operations are isolated inside `index_updater`
    index_updater: OwningHandler<RwLock<IndexUpdater>>,
    autocommit_thread: Option<ThreadHandler>,
}

impl IndexHolder {
    /// Sets up standard Summa tokenizers
    ///
    /// The set of tokenizers includes standard Tantivy tokenizers as well as `SummaTokenizer` that supports CJK
    fn register_default_tokenizers(index: &Index, index_config: &IndexConfig) {
        for (tokenizer_name, tokenizer) in &default_tokenizers(index_config) {
            index.tokenizers().register(tokenizer_name, tokenizer.clone())
        }
    }

    /// Sets up `IndexHolder`
    ///
    /// Creates the auto committing thread and consumers
    async fn setup(index_name: &str, index: Index, index_config_proxy: IndexConfigProxy) -> SummaResult<IndexHolder> {
        let index_config = index_config_proxy.read().deref().clone();
        IndexHolder::register_default_tokenizers(&index, &index_config);
        let index_reader = index.reader_builder().reload_policy(ReloadPolicy::OnCommit).try_into()?;
        let index_updater = OwningHandler::new(RwLock::new(IndexUpdater::new(
            &index.schema(),
            index_config_proxy.clone(),
            IndexWriterHolder::new(
                index.writer_with_num_threads(index_config.writer_threads.try_into().unwrap(), index_config.writer_heap_size_bytes.try_into().unwrap())?,
                match index_config.primary_key {
                    Some(ref primary_key) => index.schema().get_field(primary_key).clone(),
                    None => None,
                },
            )?,
        )?));

        let autocommit_thread = match index_config.autocommit_interval_ms.clone() {
            Some(interval_ms) => {
                let index_updater = index_updater.handler().clone();
                let index_name = index_name.to_owned();
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
                                        if let Some(mut index_updater) = index_updater.try_write() {
                                            if let Err(error) = index_updater.commit().await {
                                                warn!(error = ?error);
                                            }
                                        }
                                    }
                                    _ = &mut shutdown_tripwire => {
                                        info!(action = "shutdown_autocommit_thread");
                                        break;
                                    }
                                }
                            }
                            Ok(())
                        }
                        .instrument(info_span!(parent: None, "autocommit_thread", index_name = ?index_name)),
                    ),
                    shutdown_trigger,
                ))
            }
            None => None,
        };

        let cached_schema = index.schema();
        Ok(IndexHolder {
            index_name: String::from(index_name),
            autocommit_thread,
            query_parser: QueryParser::for_index(&index, index_config.default_fields.iter().map(|x| cached_schema.get_field(x).unwrap()).collect()),
            multi_fields: index_config.multi_fields.iter().map(|x| cached_schema.get_field(x).unwrap()).collect(),
            cached_schema,
            index_reader,
            index_updater,
            index_config_proxy,
        })
    }

    /// Creates index and sets it up via `setup`
    #[instrument(skip(schema, index_config_proxy))]
    pub(crate) async fn create(index_name: &str, schema: &Schema, index_config_proxy: IndexConfigProxy) -> SummaResult<IndexHolder> {
        let index = {
            let index_config = index_config_proxy.read();
            info!(action = "create", engine = ?index_config.index_engine);
            match &index_config.index_engine {
                IndexEngine::Memory(schema) => Index::create_in_ram(schema.clone()),
                IndexEngine::File(index_path) => {
                    if index_path.exists() {
                        Err(ValidationError::ExistingPathError(index_path.to_owned()))?;
                    }
                    std::fs::create_dir_all(index_path)?;
                    let settings = IndexSettings {
                        docstore_compression: index_config.compression.clone(),
                        sort_by_field: index_config.sort_by_field.clone(),
                    };
                    Index::builder().schema(schema.clone()).settings(settings).create_in_dir(index_path)?
                }
            }
        };
        IndexHolder::setup(index_name, index, index_config_proxy).await
    }

    /// Opens index within `index_path` directory and sets it up via `setup`
    #[instrument(skip(index_config_proxy))]
    pub(crate) async fn open(index_name: &str, index_config_proxy: IndexConfigProxy) -> SummaResult<IndexHolder> {
        let index = match &index_config_proxy.read().index_engine {
            IndexEngine::Memory(schema) => Index::create_in_ram(schema.clone()),
            IndexEngine::File(index_path) => Index::open_in_dir(index_path)?,
        };
        IndexHolder::setup(index_name, index, index_config_proxy).await
    }

    /// Index name
    pub(crate) fn index_name(&self) -> &str {
        &self.index_name
    }

    /// `IndexReader` singleton
    pub(crate) fn index_reader(&self) -> &IndexReader {
        &self.index_reader
    }

    /// `IndexUpdater` handler
    pub(crate) fn index_updater(&self) -> Handler<RwLock<IndexUpdater>> {
        self.index_updater.handler()
    }

    /// Index schema
    pub(crate) fn schema(&self) -> &Schema {
        &self.cached_schema
    }

    /// Stops `IndexHolder` instance
    ///
    /// Both autocommitting thread and consumers are stopped
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    pub(crate) async fn stop(self) -> SummaResult<Opstamp> {
        if let Some(autocommit_thread) = self.autocommit_thread {
            autocommit_thread.stop().await?;
        }
        self.index_updater.into_inner().into_inner().stop_consumers_and_commit().await
    }

    /// Delete `IndexHolder` instance
    ///
    /// Both autocommitting thread and consumers are stopped, then `IndexConfig` is removed from `ApplicationConfig`
    /// and then directory with the index is deleted.
    pub(crate) async fn delete(self) -> SummaResult<()> {
        if let Some(autocommit_thread) = self.autocommit_thread {
            autocommit_thread.stop().await?;
        };
        self.index_updater.into_inner().into_inner().stop().await?;
        match self.index_config_proxy.delete().index_engine {
            IndexEngine::Memory(_) => (),
            IndexEngine::File(ref index_path) => {
                info!(action = "delete_directory");
                remove_dir_all(index_path).await.map_err(|e| Error::IOError((e, Some(index_path.to_path_buf()))))?;
            }
        };
        Ok(())
    }

    /// Search `query` in the `IndexHolder` and collecting `Fruit` with a list of `collectors`
    pub(crate) async fn search(&self, query: &proto::Query, collectors: &Vec<proto::Collector>) -> SummaResult<Vec<proto::CollectorResult>> {
        let searcher = self.index_reader.searcher();
        let parsed_query = self.query_parser.parse_query(query)?;
        let mut multi_collector = MultiCollector::new();
        let mut extractors: Vec<Box<dyn FruitExtractor>> = collectors
            .iter()
            .map(|proto::Collector { collector, .. }| match &collector {
                Some(proto::collector::Collector::TopDocs(top_docs_collector_proto)) => {
                    let top_docs_collector: TopDocs = top_docs_collector_proto.into();
                    Box::new(fruit_extractors::TopDocs::new(
                        multi_collector.add_collector(top_docs_collector),
                        top_docs_collector_proto.limit.try_into().unwrap(),
                    )) as Box<dyn FruitExtractor>
                }
                Some(proto::collector::Collector::ReservoirSampling(reservoir_sampling_collector_proto)) => {
                    let reservoir_sampling_collector: ReservoirSampling = reservoir_sampling_collector_proto.into();
                    Box::new(fruit_extractors::ReservoirSampling(multi_collector.add_collector(reservoir_sampling_collector))) as Box<dyn FruitExtractor>
                }
                Some(proto::collector::Collector::Count(count_collector_proto)) => {
                    let count_collector: Count = count_collector_proto.into();
                    Box::new(fruit_extractors::Count(multi_collector.add_collector(count_collector))) as Box<dyn FruitExtractor>
                }
                None => Box::new(fruit_extractors::Count(multi_collector.add_collector(Count))) as Box<dyn FruitExtractor>,
            })
            .collect();
        info!(target: "query", index_name = ?self.index_name, query = ?parsed_query);
        let multi_fields = self.multi_fields.clone();
        Ok(tokio::task::spawn_blocking(move || -> SummaResult<Vec<proto::CollectorResult>> {
            let mut multifruit = searcher.search(&parsed_query, &multi_collector)?;
            Ok(extractors.drain(..).map(|e| e.extract(&mut multifruit, &searcher, &multi_fields)).collect())
        })
        .await??)
    }
}

impl std::fmt::Debug for IndexHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("IndexHolder").field("index_name", &self.index_name).finish()
    }
}

impl Into<proto::Index> for &IndexHolder {
    fn into(self) -> proto::Index {
        let index_config_proxy = self.index_config_proxy.read();
        proto::Index {
            index_aliases: index_config_proxy.application_config().get_index_aliases_for_index(&self.index_name),
            index_name: self.index_name.to_owned(),
            index_engine: format!("{:?}", index_config_proxy.index_engine),
            num_docs: self.index_reader().searcher().num_docs(),
        }
    }
}
