use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::try_join_all;
#[cfg(feature = "metrics")]
use instant::Instant;
#[cfg(feature = "metrics")]
use opentelemetry::{
    global,
    metrics::{Histogram, Unit},
    Context, KeyValue,
};
use summa_proto::proto;
use tantivy::collector::MultiCollector;
use tantivy::directory::OwnedBytes;
use tantivy::schema::Schema;
use tantivy::{Directory, Executor, Index, IndexBuilder, IndexReader, ReloadPolicy};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::RwLock;
use tracing::{info, warn};
use tracing::{instrument, trace};

use super::SummaSegmentAttributes;
use super::{build_fruit_extractor, default_tokenizers, FruitExtractor, QueryParser};
use crate::components::segment_attributes::SegmentAttributesMergerImpl;
use crate::components::{IndexWriterHolder, SummaDocument, CACHE_METRICS};
use crate::configs::{ApplicationConfig, ConfigProxy};
use crate::directories::{ChunkedCachingDirectory, ExternalRequest, ExternalRequestGenerator, HotDirectory, NetworkDirectory};
use crate::errors::{SummaResult, ValidationError};
use crate::Error;

pub struct IndexHolder {
    application_config_holder: Arc<dyn ConfigProxy<ApplicationConfig>>,
    index_engine_config: proto::IndexEngineConfig,
    index_name: String,
    index: Index,
    cached_schema: Schema,
    index_reader: IndexReader,
    index_writer_holder: Arc<RwLock<IndexWriterHolder>>,
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
            .field("index_engine_config", &self.index_engine_config)
            .field("index_settings", &self.index.settings())
            .field("index_attributes", &self.index_attributes())
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

async fn open_network_index<TExternalRequest: ExternalRequest + 'static, TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static>(
    remote_engine_config: &proto::RemoteEngineConfig,
) -> SummaResult<Index> {
    let file_lengths = Arc::new(std::sync::RwLock::new(HashMap::<PathBuf, u64>::new()));
    let network_directory = NetworkDirectory::open(Box::new(TExternalRequestGenerator::new(remote_engine_config.clone())), file_lengths.clone());

    let hotcache_bytes = match network_directory.get_file_handle("hotcache.bin".as_ref()) {
        Ok(hotcache_handle) => Some(hotcache_handle.read_bytes_async(0..hotcache_handle.len()).await?),
        Err(_) => None,
    };

    let next_directory = match &remote_engine_config.chunked_cache_config {
        Some(chunked_cache_config) => match chunked_cache_config.cache_size {
            Some(cache_size) => Box::new(ChunkedCachingDirectory::new_with_capacity_in_bytes(
                Arc::new(network_directory),
                chunked_cache_config.chunk_size as usize,
                cache_size as usize,
                CACHE_METRICS.clone(),
            )) as Box<dyn Directory>,
            None => Box::new(ChunkedCachingDirectory::new(
                Arc::new(network_directory),
                chunked_cache_config.chunk_size as usize,
            )) as Box<dyn Directory>,
        },
        None => Box::new(network_directory) as Box<dyn Directory>,
    };

    Ok(match hotcache_bytes {
        Some(hotcache_bytes) => {
            let hotcache_directory = HotDirectory::open(next_directory, OwnedBytes::new(hotcache_bytes.to_vec()), file_lengths)?;
            Index::open(hotcache_directory)?
        }
        None => Index::open(next_directory)?,
    })
}

impl IndexHolder {
    /// Sets up `IndexHolder`
    pub async fn create_holder(
        application_config_holder: Arc<dyn ConfigProxy<ApplicationConfig>>,
        index_engine_config: proto::IndexEngineConfig,
        index_name: &str,
        mut index: Index,
    ) -> SummaResult<IndexHolder> {
        register_default_tokenizers(&index);
        index.set_segment_attributes_merger(Arc::new(SegmentAttributesMergerImpl::<SummaSegmentAttributes>::new()));

        let cached_schema = index.schema();
        let query_parser = RwLock::new(QueryParser::for_index(index_name, &index)?);
        let index_reader = index.reader_builder().reload_policy(ReloadPolicy::OnCommit).try_into()?;
        let index_writer_holder = Arc::new(RwLock::new(IndexWriterHolder::from_config(
            &index,
            application_config_holder.read().await.get(),
        )?));

        Ok(IndexHolder {
            application_config_holder: application_config_holder.clone(),
            index_engine_config,
            index_name: String::from(index_name),
            index: index.clone(),
            query_parser,
            cached_schema,
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
    pub async fn create_file_index(index_path: &Path, index_builder: IndexBuilder) -> SummaResult<Index> {
        if index_path.exists() {
            return Err(ValidationError::ExistingPath(index_path.to_owned()).into());
        }
        tokio::fs::create_dir_all(index_path).await?;
        let index = index_builder.create_in_dir(index_path)?;
        info!(action = "created", index = ?index);
        Ok(index)
    }

    /// Opens index and sets it up via `setup`
    #[instrument(skip_all)]
    pub async fn from_index_engine_config<
        TExternalRequest: ExternalRequest + 'static,
        TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static,
    >(
        index_engine_config: &proto::IndexEngineConfig,
    ) -> SummaResult<Index> {
        let index = match index_engine_config {
            #[cfg(feature = "fs")]
            proto::IndexEngineConfig {
                config: Some(proto::index_engine_config::Config::File(config)),
            } => Index::open_in_dir(&config.path)?,
            proto::IndexEngineConfig {
                config: Some(proto::index_engine_config::Config::Memory(config)),
            } => IndexBuilder::new().schema(serde_yaml::from_str(&config.schema)?).create_in_ram()?,
            proto::IndexEngineConfig {
                config: Some(proto::index_engine_config::Config::Remote(config)),
            } => open_network_index::<TExternalRequest, TExternalRequestGenerator>(config).await?,
            proto::IndexEngineConfig {
                config: Some(proto::index_engine_config::Config::Ipfs(_)),
            } => todo!(),
            _ => panic!(),
        };
        info!(action = "from_config", index = ?index);
        Ok(index)
    }

    /// Attaches index and sets it up via `setup`
    #[instrument(skip_all)]
    #[cfg(feature = "fs")]
    pub async fn attach_file_index(index_path: &Path) -> SummaResult<Index> {
        let index = Index::open_in_dir(index_path)?;
        info!(action = "attached", index = ?index);
        Ok(index)
    }

    /// Compression
    pub fn compression(&self) -> proto::Compression {
        self.index_reader().searcher().index().settings().docstore_compression.into()
    }

    /// Load index attributes from meta.json
    pub fn index_attributes(&self) -> SummaResult<Option<proto::IndexAttributes>> {
        Ok(self.index.load_metas()?.attributes()?)
    }

    /// Index name
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// `IndexReader` singleton
    pub fn index_reader(&self) -> &IndexReader {
        &self.index_reader
    }

    pub fn index_payload(&self) -> SummaResult<Option<String>> {
        Ok(self.index.load_metas()?.payload)
    }

    pub fn index_engine_config(&self) -> &proto::IndexEngineConfig {
        &self.index_engine_config
    }

    pub fn index_writer_holder(&self) -> &Arc<RwLock<IndexWriterHolder>> {
        &self.index_writer_holder
    }

    /// Index schema
    pub fn schema(&self) -> &Schema {
        &self.cached_schema
    }

    pub async fn warmup(&self) -> SummaResult<()> {
        let searcher = self.index_reader().searcher();
        let mut warm_up_futures = Vec::new();
        let index_attributes = self.index_attributes();
        let default_fields = index_attributes?
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
            try_join_all(warm_up_futures).await?;
        }
        Ok(())
    }

    /// Delete `IndexHolder` instance
    ///
    /// Consumers are stopped, then `IndexConfig` is removed from `ApplicationConfig`
    /// and then directory with the index is deleted.
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    pub async fn delete(self) -> SummaResult<()> {
        info!(action = "delete_directory");
        self.application_config_holder.write().await.get_mut().indices.remove(self.index_name());
        match self.index_engine_config {
            #[cfg(feature = "fs")]
            proto::IndexEngineConfig {
                config: Some(proto::index_engine_config::Config::File(ref config)),
            } => {
                tokio::fs::remove_dir_all(&config.path)
                    .await
                    .map_err(|e| Error::IO((e, Some(PathBuf::from(&config.path)))))?;
            }
            _ => (),
        };
        Ok(())
    }

    fn spawn<R: Send + 'static, F: Sized + Send + Sync + FnOnce() -> R + 'static>(&self, f: F) -> UnboundedReceiver<R> {
        let (fruit_sender, fruit_receiver) = tokio::sync::mpsc::unbounded_channel();
        match self.index.search_executor() {
            Executor::SingleThread => {
                fruit_sender.send(f()).map_err(|_| Error::Internal).expect("Send error");
            }
            Executor::GlobalPool => {
                rayon::spawn(move || {
                    fruit_sender.send(f()).map_err(|_| Error::Internal).expect("Send error");
                });
            }
            Executor::ThreadPool(pool) => {
                pool.spawn(move || {
                    fruit_sender.send(f()).map_err(|_| Error::Internal).expect("Send error");
                });
            }
        };
        fruit_receiver
    }

    /// Search `query` in the `IndexHolder` and collecting `Fruit` with a list of `collectors`
    pub async fn search(&self, external_index_alias: &str, query: &proto::Query, collectors: &[proto::Collector]) -> SummaResult<Vec<proto::CollectorOutput>> {
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
            parsed_query = ?parsed_query
        );
        #[cfg(feature = "metrics")]
        let start_time = Instant::now();
        let mut multi_fruit = searcher.search_async(&parsed_query, &multi_collector).await?;
        #[cfg(feature = "metrics")]
        self.search_times_meter.record(
            &Context::current(),
            start_time.elapsed().as_secs_f64(),
            &[KeyValue::new("index_name", self.index_name.to_owned())],
        );
        let mut collector_outputs = Vec::with_capacity(extractors.len());
        let multi_fields = self
            .index_attributes()?
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
        self.index_writer_holder.read().await.delete_document_by_primary_key(primary_key_value)
    }

    /// Index generic `SummaDocument`
    ///
    /// `IndexUpdater` bounds unbounded `SummaDocument` inside
    pub async fn index_document(&self, document: SummaDocument<'_>) -> SummaResult<()> {
        let document = document.bound_with(&self.index.schema()).try_into()?;
        self.index_writer_holder.read().await.index_document(document)
    }

    /// Index multiple documents at a time
    pub async fn index_bulk(&self, documents: &Vec<Vec<u8>>) -> (u64, u64) {
        let (mut success_docs, mut failed_docs) = (0u64, 0u64);
        let index_writer_holder = self.index_writer_holder.read().await;
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
        (success_docs, failed_docs)
    }

    /// Commits index
    #[instrument(skip(self))]
    pub async fn commit(&self, payload: Option<String>) -> SummaResult<()> {
        self.index_writer_holder.write().await.commit(payload).await
    }
}
