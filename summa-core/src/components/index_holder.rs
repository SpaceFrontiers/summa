use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use futures::future::{join_all, try_join_all};
#[cfg(feature = "metrics")]
use instant::Instant;
#[cfg(feature = "metrics")]
use opentelemetry::{
    global,
    metrics::{Histogram, Unit},
    Context, KeyValue,
};
use summa_proto::proto;
use tantivy::collector::{Collector, MultiCollector};
use tantivy::directory::OwnedBytes;
use tantivy::query::{EnableScoring, Query, Weight};
use tantivy::schema::Schema;
use tantivy::{Directory, Executor, Index, IndexBuilder, IndexReader, ReloadPolicy};
use tokio::sync::RwLock;
use tracing::{info, warn};
use tracing::{instrument, trace};

use super::SummaSegmentAttributes;
use super::{build_fruit_extractor, default_tokenizers, FruitExtractor, QueryParser};
use crate::components::segment_attributes::SegmentAttributesMergerImpl;
use crate::components::tracker::Tracker;
use crate::components::{IndexWriterHolder, SummaDocument, TrackerEvent, CACHE_METRICS};
use crate::configs::core::ExecutionStrategy;
use crate::configs::ConfigProxy;
use crate::directories::{ChunkedCachingDirectory, ExternalRequest, ExternalRequestGenerator, FileStats, HotDirectory, NetworkDirectory, StaticDirectoryCache};
use crate::errors::{SummaResult, ValidationError};
use crate::Error;

pub struct IndexHolder {
    core_config_holder: Arc<dyn ConfigProxy<crate::configs::core::Config>>,
    index_engine_config: Arc<dyn ConfigProxy<proto::IndexEngineConfig>>,
    index_name: String,
    index: Index,
    cached_index_attributes: Option<proto::IndexAttributes>,
    cached_schema: Schema,
    index_reader: IndexReader,
    index_writer_holder: Option<Arc<RwLock<IndexWriterHolder>>>,
    query_parser: RwLock<QueryParser>,
    /// Counters
    #[cfg(feature = "metrics")]
    search_times_meter: Histogram<f64>,
}

impl Hash for IndexHolder {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index_name.hash(state)
    }
}

impl PartialEq<Self> for IndexHolder {
    fn eq(&self, other: &Self) -> bool {
        self.index_name.eq(&other.index_name)
    }
}

impl Debug for IndexHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("IndexHolder")
            .field("index_name", &self.index_name)
            .field("index_settings", &self.index.settings())
            .finish()
    }
}

/// Sets up standard Summa tokenizers
///
/// The set of tokenizers includes standard Tantivy tokenizers as well as `SummaTokenizer` that supports CJK
fn register_default_tokenizers(index: &Index) {
    for (tokenizer_name, tokenizer) in &default_tokenizers() {
        index.tokenizers().register(tokenizer_name, tokenizer.clone())
    }
}

fn wrap_with_caches<D: Directory>(
    directory: D,
    hotcache_bytes: Option<OwnedBytes>,
    chunked_cache_config: Option<&proto::ChunkedCacheConfig>,
) -> SummaResult<Box<dyn Directory>> {
    let static_cache = hotcache_bytes.map(StaticDirectoryCache::open).transpose()?;
    // ToDo: Hotcache should use same cache
    let file_lengths = static_cache
        .as_ref()
        .map(|static_cache| static_cache.file_lengths().clone())
        .unwrap_or_default();
    let file_stats = FileStats::from_file_lengths(file_lengths);
    let next_directory = match &chunked_cache_config {
        Some(chunked_cache_config) => match chunked_cache_config.cache_size {
            Some(cache_size) => Box::new(ChunkedCachingDirectory::new_with_capacity_in_bytes(
                Box::new(directory),
                chunked_cache_config.chunk_size as usize,
                cache_size as usize,
                CACHE_METRICS.clone(),
                file_stats,
            )) as Box<dyn Directory>,
            None => Box::new(ChunkedCachingDirectory::new(
                Box::new(directory),
                chunked_cache_config.chunk_size as usize,
                file_stats,
            )) as Box<dyn Directory>,
        },
        None => Box::new(directory) as Box<dyn Directory>,
    };
    Ok(match static_cache {
        Some(static_cache) => Box::new(HotDirectory::open(next_directory, static_cache)?),
        None => next_directory,
    })
}

impl IndexHolder {
    /// Sets up `IndexHolder`
    pub fn create_holder(
        core_config_holder: &Arc<dyn ConfigProxy<crate::configs::core::Config>>,
        core_config: &crate::configs::core::Config,
        mut index: Index,
        index_name: Option<&str>,
        index_engine_config: Arc<dyn ConfigProxy<proto::IndexEngineConfig>>,
        read_only: bool,
    ) -> SummaResult<IndexHolder> {
        register_default_tokenizers(&index);

        index.settings_mut().docstore_compress_dedicated_thread = core_config.dedicated_compression_thread;
        index.set_segment_attributes_merger(Arc::new(SegmentAttributesMergerImpl::<SummaSegmentAttributes>::new()));

        let metas = index.load_metas()?;
        let index_name = index_name.map(|x| x.to_string()).unwrap_or_else(|| {
            serde_json::from_value::<proto::IndexAttributes>(metas.index_attributes().expect("no attributes").expect("no attributes"))
                .expect("wrong json")
                .default_index_name
                .expect("no index name")
        });
        let cached_schema = index.schema();
        let query_parser = RwLock::new(QueryParser::for_index(&index_name, &index)?);
        let index_reader = index.reader_builder().reload_policy(ReloadPolicy::OnCommit).try_into()?;
        index_reader.reload()?;

        let index_writer_holder = if !read_only {
            Some(Arc::new(RwLock::new(IndexWriterHolder::from_config(&index, core_config)?)))
        } else {
            None
        };

        Ok(IndexHolder {
            core_config_holder: core_config_holder.clone(),
            index_engine_config,
            index_name,
            index: index.clone(),
            query_parser,
            cached_schema,
            cached_index_attributes: metas.index_attributes()?,
            index_reader,
            index_writer_holder,
            #[cfg(feature = "metrics")]
            search_times_meter: global::meter("summa")
                .f64_histogram("search_times")
                .with_unit(Unit::new("seconds"))
                .with_description("Search times")
                .init(),
        })
    }

    pub async fn reload_query_parser(&self) -> SummaResult<()> {
        *self.query_parser.write().await = QueryParser::for_index(&self.index_name, self.index_reader().searcher().index())?;
        Ok(())
    }

    /// Creates index and sets it up via `setup`
    #[instrument(skip_all)]
    pub async fn create_memory_index<
        TExternalRequest: ExternalRequest + 'static,
        TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static,
    >(
        index_builder: IndexBuilder,
    ) -> SummaResult<Index> {
        let index = index_builder.create_in_ram()?;
        info!(action = "created", index = ?index);
        Ok(index)
    }

    /// Creates index and sets it up via `setup`
    #[instrument(skip_all)]
    #[cfg(feature = "fs")]
    pub async fn create_file_index(index_path: &std::path::Path, index_builder: IndexBuilder) -> SummaResult<Index> {
        if index_path.exists() {
            return Err(ValidationError::ExistingPath(index_path.to_owned()).into());
        }
        tokio::fs::create_dir_all(index_path).await?;
        let index = index_builder.create_in_dir(index_path)?;
        info!(action = "created", index = ?index);
        Ok(index)
    }

    /// Attaches index and sets it up via `setup`
    #[instrument(skip_all)]
    #[cfg(feature = "fs")]
    pub async fn attach_file_index(file_engine_config: &proto::FileEngineConfig) -> SummaResult<Index> {
        let index = Index::open_in_dir(&file_engine_config.path)?;
        info!(action = "attached", config = ?file_engine_config);
        Ok(index)
    }

    pub async fn attach_remote_index<
        TExternalRequest: ExternalRequest + 'static,
        TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static,
    >(
        remote_engine_config: proto::RemoteEngineConfig,
        tracker: impl Tracker,
        read_only: bool,
    ) -> SummaResult<Index> {
        let network_directory = NetworkDirectory::open(Box::new(TExternalRequestGenerator::new(remote_engine_config.clone())));
        tracker.send_event(TrackerEvent::ReadingHotcache);
        let hotcache_bytes = match network_directory
            .get_network_file_handle("hotcache.bin".as_ref())
            .do_read_bytes_async(None, tracker)
            .await
        {
            Ok(hotcache_bytes) => {
                if read_only {
                    info!(action = "read_hotcache");
                    Some(hotcache_bytes)
                } else {
                    warn!(action = "omit_hotcache");
                    None
                }
            }
            Err(error) => {
                warn!(action = "error_hotcache", error = ?error);
                None
            }
        };
        let directory = wrap_with_caches(network_directory, hotcache_bytes, remote_engine_config.chunked_cache_config.as_ref())?;
        Ok(Index::open(directory)?)
    }

    /// Attaches index and sets it up via `setup`
    #[cfg(feature = "ipfs")]
    #[instrument(skip_all)]
    pub async fn attach_ipfs_index(
        ipfs_engine_config: &proto::IpfsEngineConfig,
        content_loader: &iroh_unixfs::content_loader::FullLoader,
        read_only: bool,
    ) -> SummaResult<Index> {
        let index_path = std::path::Path::new(&ipfs_engine_config.path);
        if index_path.exists() {
            tokio::fs::remove_dir_all(index_path).await?;
        }
        tokio::fs::create_dir_all(index_path).await?;
        let mmap_directory = tantivy::directory::MmapDirectory::open(index_path)?;
        let iroh_directory = crate::directories::IrohDirectory::new(mmap_directory, content_loader.clone(), &ipfs_engine_config.cid).await?;
        let chunked_cache_config = ipfs_engine_config.chunked_cache_config.clone();
        let hotcache_bytes = match iroh_directory.get_file_handle("hotcache.bin".as_ref()) {
            Ok(hotcache_handle) => {
                if read_only {
                    info!(action = "read_hotcache");
                    hotcache_handle.read_bytes_async(0..hotcache_handle.len()).await.ok()
                } else {
                    warn!(action = "omit_hotcache");
                    None
                }
            }
            Err(error) => {
                warn!(action = "error_hotcache", error = ?error);
                None
            }
        };
        let index = tokio::task::spawn_blocking(move || {
            Ok::<Index, Error>(Index::open(wrap_with_caches(iroh_directory, hotcache_bytes, chunked_cache_config.as_ref())?)?)
        })
        .await??;
        info!(action = "attached", config = ?ipfs_engine_config);
        Ok(index)
    }

    /// Compression
    pub fn compression(&self) -> proto::Compression {
        self.index_reader().searcher().index().settings().docstore_compression.into()
    }

    /// Load index attributes from meta.json
    pub fn index_attributes(&self) -> Option<&proto::IndexAttributes> {
        self.cached_index_attributes.as_ref()
    }

    /// Index name
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// `IndexReader` singleton
    pub fn index_reader(&self) -> &IndexReader {
        &self.index_reader
    }

    pub async fn index_payload(&self) -> SummaResult<Option<String>> {
        Ok(self.index.load_metas_async().await?.payload)
    }

    pub fn index_engine_config(&self) -> &Arc<dyn ConfigProxy<proto::IndexEngineConfig>> {
        &self.index_engine_config
    }

    pub fn index_writer_holder(&self) -> SummaResult<&Arc<RwLock<IndexWriterHolder>>> {
        self.index_writer_holder
            .as_ref()
            .ok_or_else(|| Error::ReadOnlyIndex(self.index_name.to_string()))
    }

    /// Index schema
    pub fn schema(&self) -> &Schema {
        &self.cached_schema
    }

    pub async fn partial_warmup(&self, tracker: impl Tracker) -> SummaResult<()> {
        let searcher = self.index_reader().searcher();
        let mut warm_up_futures = Vec::new();
        let index_attributes = self.index_attributes();
        let default_fields = index_attributes
            .map(|index_attributes| {
                index_attributes
                    .default_fields
                    .iter()
                    .map(|field_name| {
                        self.cached_schema
                            .get_field(field_name)
                            .ok_or_else(|| ValidationError::MissingField(field_name.to_string()).into())
                    })
                    .collect::<SummaResult<Vec<_>>>()
            })
            .transpose()?;
        if let Some(default_fields) = default_fields {
            for field in default_fields {
                for segment_reader in searcher.segment_readers() {
                    let inverted_index = segment_reader.inverted_index(field)?.clone();
                    warm_up_futures.push(async move {
                        let dict = inverted_index.terms();
                        dict.warm_up_dictionary().await
                    });
                }
            }
            tracker.send_event(TrackerEvent::WarmingUp);
            try_join_all(warm_up_futures).await?;
        }
        Ok(())
    }

    pub async fn full_warmup(&self, tracker: impl Tracker) -> SummaResult<()> {
        let managed_directory = self.index.directory();
        tracker.send_event(TrackerEvent::WarmingUp);
        join_all(managed_directory.list_managed_files().into_iter().map(move |file| {
            let tracker = tracker.clone();
            let file_name = file.to_string_lossy().to_string();
            async move {
                tracker.send_event(TrackerEvent::StartReadingFile(file_name.clone()));
                managed_directory.atomic_read_async(&file).await.map_err(|e| Error::Tantivy(e.into()))?;
                tracker.send_event(TrackerEvent::FinishReadingFile(file_name));
                Ok(())
            }
        }))
        .await
        .into_iter()
        .collect::<SummaResult<Vec<_>>>()?;
        Ok(())
    }

    /// Delete `IndexHolder` instance
    ///
    /// Consumers are stopped, then `IndexConfig` is removed from `CoreConfig`
    /// and then directory with the index is deleted.
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    pub async fn delete(self) -> SummaResult<()> {
        let mut config = self.core_config_holder.write().await;
        let index_engine_config = config.get_mut().indices.remove(self.index_name()).expect("cannot retrieve config");
        match index_engine_config.config {
            #[cfg(feature = "fs")]
            Some(proto::index_engine_config::Config::File(ref config)) => {
                info!(action = "delete_directory", directory = ?config.path);
                tokio::fs::remove_dir_all(&config.path)
                    .await
                    .map_err(|e| Error::IO((e, Some(std::path::PathBuf::from(&config.path)))))?;
            }
            #[cfg(feature = "fs")]
            Some(proto::index_engine_config::Config::Ipfs(ref config)) => {
                info!(action = "delete_directory", directory = ?config.path);
                tokio::fs::remove_dir_all(&config.path)
                    .await
                    .map_err(|e| Error::IO((e, Some(std::path::PathBuf::from(&config.path)))))?;
            }
            _ => (),
        };
        config.commit().await?;
        Ok(())
    }

    /// Search `query` in the `IndexHolder` and collecting `Fruit` with a list of `collectors`
    pub async fn search(&self, external_index_alias: &str, query: &proto::Query, collectors: &[proto::Collector]) -> SummaResult<Vec<proto::CollectorOutput>> {
        let execution_strategy = self.core_config_holder.read().await.get().execution_strategy.clone();
        let searcher = self.index_reader().searcher();
        let parsed_query = self.query_parser.read().await.parse_query(query)?;
        let mut multi_collector = MultiCollector::new();
        let extractors: Vec<Box<dyn FruitExtractor>> = collectors
            .iter()
            .map(|collector_proto| build_fruit_extractor(collector_proto, &parsed_query, &self.cached_schema, &mut multi_collector))
            .collect::<SummaResult<_>>()?;
        trace!(
            target: "query",
            index_name = ?self.index_name,
            query = ?query,
            parsed_query = ?parsed_query,
            execution_strategy = ?execution_strategy
        );

        #[cfg(feature = "metrics")]
        let start_time = Instant::now();
        let mut multi_fruit = match execution_strategy {
            ExecutionStrategy::Async => searcher.search_async(&parsed_query, &multi_collector).await?,
            // ToDo: Reduce code here or in Tantivy `search_with_executor`
            ExecutionStrategy::GlobalPool => {
                let segment_readers = searcher.segment_readers();
                let multi_collector_ref = Arc::new(multi_collector);
                let mut fruit_receiver = {
                    let (fruit_sender, fruit_receiver) = tokio::sync::mpsc::unbounded_channel();
                    let weight: Arc<dyn Weight> = Arc::from(parsed_query.weight_async(EnableScoring::Enabled(&searcher)).await?);
                    for (segment_ord, segment_reader) in segment_readers.iter().enumerate() {
                        let fruit_sender_ref = fruit_sender.clone();
                        let multi_collector_ref = multi_collector_ref.clone();
                        let weight = weight.clone();
                        let segment_reader = segment_reader.clone();
                        rayon::spawn(move || {
                            fruit_sender_ref
                                .send(multi_collector_ref.collect_segment(weight.as_ref(), segment_ord as u32, &segment_reader))
                                .map_err(|_| Error::Internal)
                                .expect("Send error");
                        });
                    }
                    fruit_receiver
                };
                let mut fruits = vec![];
                while let Some(fruit) = fruit_receiver.recv().await {
                    fruits.push(fruit?);
                }
                multi_collector_ref.merge_fruits(fruits)?
            }
            ExecutionStrategy::SingleThread => searcher.search_with_executor(&parsed_query, &multi_collector, &Executor::SingleThread)?,
        };
        #[cfg(feature = "metrics")]
        self.search_times_meter.record(
            &Context::current(),
            start_time.elapsed().as_secs_f64(),
            &[KeyValue::new("index_name", self.index_name.to_owned())],
        );
        let mut collector_outputs = Vec::with_capacity(extractors.len());
        let multi_fields = self
            .index_attributes()
            .map(|index_attributes| {
                index_attributes
                    .multi_fields
                    .iter()
                    .map(|field_name| self.cached_schema.get_field(field_name).expect("Field not found"))
                    .collect()
            })
            .unwrap_or_else(HashSet::new);
        for extractor in extractors.into_iter() {
            collector_outputs.push(
                extractor
                    .extract_async(external_index_alias, &mut multi_fruit, &searcher, &multi_fields)
                    .await?,
            )
        }
        Ok(collector_outputs)
    }

    /// Delete `SummaDocument` by `primary_key`
    pub async fn delete_document(&self, primary_key_value: Option<proto::PrimaryKey>) -> SummaResult<()> {
        self.index_writer_holder()?.read().await.delete_document_by_primary_key(primary_key_value)
    }

    /// Index generic `SummaDocument`
    ///
    /// `IndexUpdater` bounds unbounded `SummaDocument` inside
    pub async fn index_document(&self, document: SummaDocument<'_>) -> SummaResult<()> {
        let document = document.bound_with(&self.index.schema()).try_into()?;
        self.index_writer_holder()?.read().await.index_document(document)
    }

    /// Index multiple documents at a time
    pub async fn index_bulk(&self, documents: &Vec<Vec<u8>>) -> SummaResult<(u64, u64)> {
        let (mut success_docs, mut failed_docs) = (0u64, 0u64);
        let index_writer_holder = self.index_writer_holder()?.read().await;
        for document in documents {
            match SummaDocument::UnboundJsonBytes(document)
                .bound_with(&self.index.schema())
                .try_into()
                .and_then(|document| index_writer_holder.index_document(document))
            {
                Ok(_) => success_docs += 1,
                Err(error) => {
                    warn!(action = "error", error = ?error);
                    failed_docs += 1
                }
            }
        }
        Ok((success_docs, failed_docs))
    }
}
