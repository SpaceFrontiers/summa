use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
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
use tantivy::{Directory, Executor, Index, IndexReader, IndexSettings, ReloadPolicy};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::RwLock;
use tracing::{info, warn};
use tracing::{instrument, trace};

#[cfg(feature = "consume")]
use super::IndexUpdater;
use super::SummaSegmentAttributes;
use super::{build_fruit_extractor, default_tokenizers, FruitExtractor, QueryParser};
use crate::components::{IndexWriterHolder, SummaDocument, CACHE_METRICS};
use crate::configs::{ConfigProxy, ConsumerConfig, IndexConfig, IndexEngine, NetworkConfig};
use crate::directories::{ChunkedCachingDirectory, ExternalRequest, ExternalRequestGenerator, HotDirectory, NetworkDirectory};
use crate::errors::{SummaResult, ValidationError};
use crate::Error;

pub struct IndexHolder {
    index_name: String,
    index: Index,
    index_config_proxy: Arc<dyn ConfigProxy<IndexConfig>>,
    cached_schema: Schema,
    index_reader: IndexReader,
    index_writer_holder: Arc<RwLock<IndexWriterHolder>>,
    query_parser: RwLock<QueryParser>,
    /// Counters
    #[cfg(feature = "metrics")]
    search_times_meter: Histogram<f64>,
    /// All modifying operations are isolated inside `index_updater`
    #[cfg(feature = "consume")]
    index_updater: IndexUpdater,
}

/// Sets up standard Summa tokenizers
///
/// The set of tokenizers includes standard Tantivy tokenizers as well as `SummaTokenizer` that supports CJK
fn register_default_tokenizers(index: &Index) {
    for (tokenizer_name, tokenizer) in &default_tokenizers() {
        index.tokenizers().register(tokenizer_name, tokenizer.clone())
    }
}

/// Sets up query parser
fn setup_query_parser(index_name: &str, index: &Index, index_config: &IndexConfig, schema: &Schema) -> SummaResult<QueryParser> {
    Ok(QueryParser::for_index(
        index_name,
        index,
        index_config
            .default_fields
            .iter()
            .map(|field_name| {
                schema
                    .get_field(field_name)
                    .ok_or_else(|| ValidationError::MissingField(field_name.to_string()).into())
            })
            .collect::<SummaResult<_>>()?,
    ))
}

fn create_network_index<TExternalRequest: ExternalRequest + 'static, TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static>(
    network_config: &NetworkConfig,
) -> SummaResult<Index> {
    let file_lengths = Arc::new(parking_lot::RwLock::new(HashMap::<PathBuf, u64>::new()));
    let network_directory = NetworkDirectory::open(Box::new(TExternalRequestGenerator::new(network_config.clone())), file_lengths.clone());
    let hotcache_bytes = network_directory.atomic_read("hotcache.bin".as_ref());

    let next_directory = match &network_config.chunked_cache_config {
        Some(chunked_cache_config) => match chunked_cache_config.cache_size {
            Some(cache_size) => Box::new(ChunkedCachingDirectory::new_with_capacity_in_bytes(
                Arc::new(network_directory),
                chunked_cache_config.chunk_size,
                cache_size,
                CACHE_METRICS.clone(),
            )) as Box<dyn Directory>,
            None => Box::new(ChunkedCachingDirectory::new(Arc::new(network_directory), chunked_cache_config.chunk_size)) as Box<dyn Directory>,
        },
        None => Box::new(network_directory) as Box<dyn Directory>,
    };

    Ok(match hotcache_bytes {
        Ok(hotcache_bytes) => {
            let hotcache_directory = HotDirectory::open(next_directory, OwnedBytes::new(hotcache_bytes.to_vec()), file_lengths)?;
            Index::open(hotcache_directory)?
        }
        Err(_) => Index::open(next_directory)?,
    })
}

impl IndexHolder {
    /// Sets up `IndexHolder`
    pub async fn setup(index_name: &str, mut index: Index, index_config_proxy: Arc<dyn ConfigProxy<IndexConfig>>) -> SummaResult<IndexHolder> {
        let index_config = index_config_proxy.read().await.get().clone();
        register_default_tokenizers(&index);
        index.set_segment_attributes_merger::<SummaSegmentAttributes>();

        let cached_schema = index.schema();
        let query_parser = RwLock::new(setup_query_parser(index_name, &index, &index_config, &cached_schema)?);
        let index_reader = index.reader_builder().reload_policy(ReloadPolicy::OnCommit).try_into()?;
        let index_writer_holder = Arc::new(RwLock::new(IndexWriterHolder::from_config(&index, &index_config)?));

        Ok(IndexHolder {
            index_name: String::from(index_name),
            index: index.clone(),
            query_parser,
            cached_schema,
            index_reader,
            index_writer_holder: index_writer_holder.clone(),
            index_config_proxy: index_config_proxy.clone(),
            #[cfg(feature = "metrics")]
            search_times_meter: global::meter("summa")
                .f64_histogram("search_times")
                .with_unit(Unit::new("seconds"))
                .with_description("Search times")
                .init(),
            #[cfg(feature = "consume")]
            index_updater: IndexUpdater::new(index, index_writer_holder, index_name, index_config_proxy).await?,
        })
    }

    pub async fn reload_query_parser(&self) -> SummaResult<()> {
        *self.query_parser.write().await = setup_query_parser(
            &self.index_name,
            self.index_reader.searcher().index(),
            self.index_config_proxy.read().await.get(),
            &self.cached_schema,
        )?;
        Ok(())
    }

    /// Creates index and sets it up via `setup`
    #[instrument(skip_all)]
    pub async fn create<TExternalRequest: ExternalRequest + 'static, TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static>(
        schema: &Schema,
        index_settings: IndexSettings,
        index_engine: &IndexEngine,
    ) -> SummaResult<Index> {
        let index_builder = Index::builder().schema(schema.clone()).settings(index_settings);
        let index = match index_engine {
            #[cfg(feature = "fs")]
            IndexEngine::File(index_path) => {
                if index_path.exists() {
                    return Err(ValidationError::ExistingPath(index_path.to_owned()).into());
                }
                tokio::fs::create_dir_all(index_path).await?;
                index_builder.create_in_dir(index_path)?
            }
            IndexEngine::Memory(_) => index_builder.create_in_ram()?,
            IndexEngine::Remote(network_config) => create_network_index::<TExternalRequest, TExternalRequestGenerator>(network_config)?,
        };
        info!(action = "created", index = ?index);
        Ok(index)
    }

    /// Opens index and sets it up via `setup`
    #[instrument(skip_all)]
    pub fn open<TExternalRequest: ExternalRequest + 'static, TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static>(
        index_engine: &IndexEngine,
    ) -> SummaResult<Index> {
        let index = match index_engine {
            #[cfg(feature = "fs")]
            IndexEngine::File(index_path) => Index::open_in_dir(index_path)?,
            IndexEngine::Memory(schema) => Index::create_in_ram(schema.clone()),
            IndexEngine::Remote(network_config) => create_network_index::<TExternalRequest, TExternalRequestGenerator>(network_config)?,
        };
        info!(action = "opened", index = ?index);
        Ok(index)
    }

    /// Compression
    pub fn compression(&self) -> proto::Compression {
        self.index_reader.searcher().index().settings().docstore_compression.into()
    }

    /// Index name
    pub fn index_config_proxy(&self) -> &Arc<dyn ConfigProxy<IndexConfig>> {
        &self.index_config_proxy
    }

    /// Index name
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// `IndexReader` singleton
    pub fn index_reader(&self) -> &IndexReader {
        &self.index_reader
    }

    /// `IndexUpdater` handler
    #[cfg(feature = "consume")]
    pub fn index_updater(&self) -> &IndexUpdater {
        &self.index_updater
    }

    pub fn index_payload(&self) -> SummaResult<Option<String>> {
        Ok(self.index.load_metas()?.payload)
    }

    /// Index schema
    pub fn schema(&self) -> &Schema {
        &self.cached_schema
    }

    /// Consumer configs
    pub async fn get_consumer_config(&self, consumer_name: &str) -> SummaResult<ConsumerConfig> {
        Ok(self
            .index_config_proxy
            .read()
            .await
            .get()
            .consumer_configs
            .get(consumer_name)
            .ok_or_else(|| ValidationError::MissingConsumer(consumer_name.to_owned()))?
            .clone())
    }

    pub async fn warmup(&self) -> SummaResult<()> {
        let searcher = self.index_reader.searcher();
        let mut warm_up_futures = Vec::new();
        let fields = self
            .index_config_proxy
            .read()
            .await
            .get()
            .default_fields
            .iter()
            .map(|field_name| {
                self.cached_schema
                    .get_field(field_name)
                    .ok_or_else(|| ValidationError::MissingField(field_name.to_string()).into())
            })
            .collect::<SummaResult<Vec<_>>>()?;
        for field in fields {
            for segment_reader in searcher.segment_readers() {
                let inverted_index = segment_reader.inverted_index(field)?.clone();
                warm_up_futures.push(async move {
                    let dict = inverted_index.terms();
                    dict.warm_up_dictionary().await
                });
            }
        }
        try_join_all(warm_up_futures).await?;
        Ok(())
    }

    /// Stops `IndexHolder` instance
    #[cfg(feature = "consume")]
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    pub async fn stop(self) -> SummaResult<()> {
        self.index_updater.stop_updates_and_commit().await
    }

    /// Delete `IndexHolder` instance
    ///
    /// Consumers are stopped, then `IndexConfig` is removed from `ApplicationConfig`
    /// and then directory with the index is deleted.
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    pub async fn delete(mut self) -> SummaResult<()> {
        #[cfg(feature = "consume")]
        self.index_updater.stop_updates().await?;
        info!(action = "delete_directory");
        match self.index_config_proxy.delete().await?.index_engine {
            #[cfg(feature = "fs")]
            IndexEngine::File(ref index_path) => {
                tokio::fs::remove_dir_all(index_path)
                    .await
                    .map_err(|e| Error::IO((e, Some(index_path.to_path_buf()))))?;
            }
            IndexEngine::Memory(_) => (),
            IndexEngine::Remote(_) => (),
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
        let searcher = self.index_reader.searcher();
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
        let mut receiver = self.spawn(move || (searcher.search(&parsed_query, &multi_collector), searcher));
        let (multi_fruit, searcher) = receiver.recv().await.expect("channel unexpectedly closed");
        let mut multi_fruit = multi_fruit?;
        #[cfg(feature = "metrics")]
        self.search_times_meter.record(
            &Context::current(),
            start_time.elapsed().as_secs_f64(),
            &[KeyValue::new("index_name", self.index_name.to_owned())],
        );
        let mut collector_outputs = Vec::with_capacity(extractors.len());
        let multi_fields = self
            .index_config_proxy
            .read()
            .await
            .get()
            .multi_fields
            .iter()
            .map(|field_name| self.cached_schema.get_field(field_name).expect("Field not found"))
            .collect();
        for extractor in extractors.into_iter() {
            collector_outputs.push(extractor.extract(external_index_alias, &mut multi_fruit, &searcher, &multi_fields)?)
        }
        Ok(collector_outputs)
    }

    /// Delete `SummaDocument` by `primary_key`
    pub async fn delete_document(&self, primary_key_value: i64) -> SummaResult<()> {
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

impl Debug for IndexHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("IndexHolder").field("index_name", &self.index_name).finish()
    }
}
