use super::default_tokenizers::default_tokenizers;
use super::index_writer_holder::IndexWriterHolder;
use crate::configs::{ApplicationConfig, KafkaConsumerConfig};
use crate::configs::{IndexConfig, IndexEngine};
use crate::errors::{Error, SummaResult};
use crate::proto;
use crate::search_engine::{IndexUpdater, SummaDocument};
use crate::utils::sync::OwningHandler;
use crate::utils::thread_handler::ThreadHandler;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tantivy::collector::TopDocs;
use tantivy::directory::{GarbageCollectionResult, INDEX_WRITER_LOCK};
use tantivy::query::QueryParser;
use tantivy::schema::Schema;
use tantivy::{Directory, Index, IndexReader, IndexSettings, LeasedItem, Opstamp, ReloadPolicy, Searcher, SegmentId, SegmentMeta};
use tokio::sync::oneshot;
use tokio::time;
use tracing::{info, info_span, instrument, warn, Instrument};

pub struct IndexHolder {
    index_name: String,
    index: Index,
    index_reader: IndexReader,
    query_parser: QueryParser,
    /// All modifying operations are isolated inside `index_updater`
    index_updater: OwningHandler<IndexUpdater>,
    autocommit_thread: Option<ThreadHandler>,
}

fn check_orphan_files<P: AsRef<Path>>(index_path: P, index: &Index) {
    let directory = index.directory();
    let _ = directory.acquire_lock(&INDEX_WRITER_LOCK).unwrap();

    let tantivy_files = directory.list_managed_files();
    let tantivy_files: HashSet<PathBuf> = tantivy_files
        .union(
            &index
                .searchable_segment_metas()
                .unwrap()
                .into_iter()
                .flat_map(|segment_meta| segment_meta.list_files())
                .collect(),
        )
        .map(|x| x.to_owned())
        .collect();

    let mut orphan_files = Vec::new();
    for entry in std::fs::read_dir(&index_path).unwrap() {
        let file_name = entry.unwrap().file_name().to_string_lossy().to_string();
        if file_name == ".managed.json" || file_name == ".tantivy-meta.lock" || file_name == ".tantivy-writer.lock" {
            continue;
        }
        if !tantivy_files.contains(&PathBuf::from(file_name.clone())) {
            orphan_files.push(file_name);
        }
    }
    if orphan_files.len() > 0 {
        let mut tantivy_files: Vec<_> = tantivy_files.into_iter().collect();
        tantivy_files.sort();
        orphan_files.sort();
        warn!(action = "tantivy_files", index_path = ?index_path.as_ref(), tantivy_files = ?tantivy_files);
        warn!(action = "orphan_files", index_path = ?index_path.as_ref(), orphan_files = ?orphan_files);
    };
}

impl IndexHolder {
    fn register_default_tokenizers(index: &Index) {
        for (tokenizer_name, tokenizer) in &default_tokenizers() {
            index.tokenizers().register(tokenizer_name, tokenizer.clone())
        }
    }

    fn setup(index_name: &str, index: Index, index_config: &IndexConfig) -> SummaResult<IndexHolder> {
        IndexHolder::register_default_tokenizers(&index);
        let index_reader = index.reader_builder().reload_policy(ReloadPolicy::OnCommit).try_into()?;
        let index_updater = OwningHandler::new(IndexUpdater::new(
            &index.schema(),
            index_config,
            IndexWriterHolder::new(
                index.writer_with_num_threads(index_config.writer_threads.try_into().unwrap(), index_config.writer_heap_size_bytes.try_into().unwrap())?,
                index_config.primary_key.clone(),
            )?,
        )?);

        let autocommit_thread = match index_config.autocommit_interval_ms.clone() {
            Some(interval_ms) => {
                let index_updater = index_updater.handler().clone();
                let index_name = index_name.to_owned();
                let (shutdown_trigger, mut shutdown_tripwire) = oneshot::channel();
                let mut tick_task = time::interval(Duration::from_millis(interval_ms));
                Some(ThreadHandler::new(
                    tokio::spawn(
                        async move {
                            info!(action = "spawning_autocommit_thread", index_name = ?index_name, interval_ms = interval_ms);
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
            autocommit_thread,
            query_parser: QueryParser::for_index(&index, index_config.default_fields.clone()),
            index,
            index_reader,
            index_updater,
        })
    }

    #[instrument(skip(index_config, schema))]
    pub(crate) async fn create(index_name: &str, index_path: &Path, index_config: &IndexConfig, schema: &Schema) -> SummaResult<IndexHolder> {
        info!(action = "create", engine = ?index_config.index_engine);
        let index = match &index_config.index_engine {
            IndexEngine::Memory(schema) => Index::create_in_ram(schema.clone()),
            IndexEngine::File => {
                let settings = IndexSettings {
                    docstore_compression: index_config.compression.clone(),
                    sort_by_field: index_config.sort_by_field.clone(),
                };
                Index::builder().schema(schema.clone()).settings(settings).create_in_dir(index_path)?
            }
        };
        IndexHolder::setup(index_name, index, index_config)
    }

    #[instrument(skip_all)]
    pub(crate) async fn open(index_name: &str, application_config: &ApplicationConfig, index_config: &IndexConfig) -> SummaResult<IndexHolder> {
        let index = match &index_config.index_engine {
            IndexEngine::Memory(schema) => Index::create_in_ram(schema.clone()),
            IndexEngine::File => {
                // ToDo: remove after introducing orphan files
                let index_path = application_config.get_path_for_index_data(index_name);
                let index = Index::open_in_dir(&index_path)?;
                check_orphan_files(index_path, &index);
                index
            }
        };
        IndexHolder::setup(index_name, index, index_config)
    }

    pub(crate) fn index_name(&self) -> &str {
        &self.index_name
    }

    pub(crate) fn num_docs(&self) -> u64 {
        self.index_reader.searcher().num_docs()
    }

    pub(crate) fn has_consumers(&self) -> bool {
        self.index_updater.has_consumers()
    }

    pub(crate) fn schema(&self) -> Schema {
        self.index.schema()
    }

    pub(crate) fn searcher(&self) -> LeasedItem<Searcher> {
        self.index_reader.searcher()
    }

    pub(crate) fn reload(&self) -> SummaResult<()> {
        Ok(self.index_reader.reload()?)
    }

    pub(crate) async fn vacuum(&self) -> SummaResult<GarbageCollectionResult> {
        self.index_updater.vacuum().await
    }

    pub(crate) async fn stop(self) -> SummaResult<Opstamp> {
        if let Some(autocommit_thread) = self.autocommit_thread {
            autocommit_thread.stop().await?;
        }
        self.index_updater.into_inner().last_commit().await
    }

    pub async fn commit(&self) -> SummaResult<Opstamp> {
        self.index_updater.commit().await
    }

    pub(crate) async fn create_consumer(&self, consumer_name: &str, consumer_config: KafkaConsumerConfig) -> SummaResult<()> {
        self.index_updater.create_consumer(consumer_name, consumer_config).await
    }

    pub(crate) async fn delete_consumer(&self, consumer_name: &str) -> SummaResult<()> {
        self.index_updater.delete_consumer(consumer_name).await
    }

    pub async fn delete_all_consumers(&self) -> SummaResult<()> {
        self.index_updater.delete_all_consumers().await
    }

    pub(crate) fn index_document(&self, document: SummaDocument<'_>, reindex: bool) -> SummaResult<Opstamp> {
        self.index_updater.index_document(document, reindex)
    }

    pub async fn merge(&self, segment_ids: &[SegmentId]) -> SummaResult<SegmentMeta> {
        self.index_updater.merge(segment_ids).await
    }

    pub async fn try_commit_and_log(&self, restart_consumers: bool) -> Option<Opstamp> {
        self.index_updater.try_commit_and_log(restart_consumers).await
    }

    pub(crate) async fn search(&self, query: &str, limit: usize, offset: usize) -> SummaResult<proto::SearchResponse> {
        let schema = self.schema();
        let searcher = self.index_reader.searcher();

        let index_name = self.index_name.to_owned();

        let parsed_query = self.query_parser.parse_query(&query);
        let parsed_query = match parsed_query {
            // ToDo: More clever processing
            Err(tantivy::query::QueryParserError::FieldDoesNotExist(_)) => {
                return Ok(proto::SearchResponse {
                    index_name: index_name.to_owned(),
                    scored_documents: vec![],
                    has_next: false,
                })
            }
            Err(e) => Err(Error::InvalidSyntaxError((e, query.to_owned()))),
            Ok(r) => Ok(r),
        }?;

        let query = query.to_owned();
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
                index_name: index_name.to_owned(),
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
