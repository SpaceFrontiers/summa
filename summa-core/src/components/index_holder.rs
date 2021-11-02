#[cfg(feature = "index-updater")]
use super::IndexUpdater;
use super::SummaSegmentAttributes;
use super::{build_fruit_extractor, default_tokenizers, FruitExtractor, QueryParser};
use crate::components::CACHE_METRICS;
use crate::configs::{ConfigProxy, ConsumerConfig, IndexConfig, IndexEngine, NetworkConfig};
use crate::directories::{ChunkedCachingDirectory, ExternalRequest, ExternalRequestGenerator, HotDirectory, NetworkDirectory};
use crate::errors::{self, SummaResult, ValidationError};
use futures::future::try_join_all;
#[cfg(feature = "metrics")]
use instant::Instant;
#[cfg(feature = "metrics")]
use opentelemetry::{
    global,
    metrics::{Histogram, Unit},
    Context, KeyValue,
};
use std::sync::Arc;
use summa_proto::proto;
use tantivy::collector::MultiCollector;
use tantivy::directory::OwnedBytes;
use tantivy::schema::Schema;
#[cfg(feature = "index-updater")]
use tantivy::Opstamp;
use tantivy::{Directory, Index, IndexReader, IndexSettings, ReloadPolicy};
#[cfg(feature = "index-updater")]
use tokio::fs::remove_dir_all;
use tokio::sync::RwLock;
use tracing::info;
use tracing::{instrument, trace};

pub struct IndexHolder {
    index_name: String,
    index: Index,
    index_config_proxy: Arc<dyn ConfigProxy<IndexConfig>>,
    cached_schema: Schema,
    index_reader: IndexReader,
    query_parser: RwLock<QueryParser>,
    /// Counters
    #[cfg(feature = "metrics")]
    search_times_meter: Histogram<f64>,
    /// All modifying operations are isolated inside `index_updater`
    #[cfg(feature = "index-updater")]
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

fn create_index_with_hotcache<D: Directory>(directory: D) -> SummaResult<Index> {
    Ok(match directory.atomic_read("hotcache.bin".as_ref()) {
        Ok(hotcache_bytes) => {
            let hotcache_directory = HotDirectory::open(directory, OwnedBytes::new(hotcache_bytes.to_vec()))?;
            Index::open(hotcache_directory)?
        }
        Err(_) => Index::open(directory)?,
    })
}

fn create_network_index<TExternalRequest: ExternalRequest + 'static, TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static>(
    network_config: &NetworkConfig,
) -> SummaResult<Index> {
    let network_directory = NetworkDirectory::new(network_config.files.clone(), Box::new(TExternalRequestGenerator::new(network_config.clone())));
    Ok(match &network_config.caching_config {
        Some(caching_config) => {
            let chunking_directory = ChunkedCachingDirectory::new_with_capacity_in_bytes(
                Arc::new(network_directory),
                caching_config.chunk_size,
                caching_config.cache_size,
                CACHE_METRICS.clone(),
            );
            create_index_with_hotcache(chunking_directory)?
        }
        None => create_index_with_hotcache(network_directory)?,
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

        Ok(IndexHolder {
            index_name: String::from(index_name),
            index: index.clone(),
            query_parser,
            cached_schema,
            index_reader,
            index_config_proxy: index_config_proxy.clone(),
            #[cfg(feature = "metrics")]
            search_times_meter: global::meter("summa")
                .f64_histogram("search_times")
                .with_unit(Unit::new("seconds"))
                .with_description("Search times")
                .init(),
            #[cfg(feature = "index-updater")]
            index_updater: IndexUpdater::new(index, index_name, index_config_proxy).await?,
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
    pub fn create<TExternalRequest: ExternalRequest + 'static, TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static>(
        schema: &Schema,
        index_settings: IndexSettings,
        index_engine: &IndexEngine,
    ) -> SummaResult<Index> {
        let index_builder = Index::builder().schema(schema.clone()).settings(index_settings);
        let index = match index_engine {
            #[cfg(feature = "index-updater")]
            IndexEngine::File(index_path) => {
                if index_path.exists() {
                    return Err(ValidationError::ExistingPath(index_path.to_owned()).into());
                }
                std::fs::create_dir_all(index_path)?;
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
            #[cfg(feature = "index-updater")]
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
    #[cfg(feature = "index-updater")]
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
            .ok_or_else(|| errors::ValidationError::MissingConsumer(consumer_name.to_owned()))?
            .clone())
    }

    pub async fn warmup_internal(&self) -> SummaResult<()> {
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
    #[cfg(feature = "index-updater")]
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    pub async fn stop(self) -> SummaResult<Opstamp> {
        self.index_updater.stop_updates_and_commit().await
    }

    /// Delete `IndexHolder` instance
    ///
    /// Consumers are stopped, then `IndexConfig` is removed from `ApplicationConfig`
    /// and then directory with the index is deleted.
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    #[cfg(feature = "index-updater")]
    pub async fn delete(mut self) -> SummaResult<()> {
        self.index_updater.stop_updates().await?;
        match self.index_config_proxy.delete().await?.index_engine {
            IndexEngine::File(ref index_path) => {
                info!(action = "delete_directory");
                remove_dir_all(index_path)
                    .await
                    .map_err(|e| errors::Error::IO((e, Some(index_path.to_path_buf()))))?;
            }
            IndexEngine::Memory(_) => (),
            IndexEngine::Remote(_) => (),
        };
        Ok(())
    }

    /// Search `query` in the `IndexHolder` and collecting `Fruit` with a list of `collectors`
    pub async fn search(
        &self,
        external_index_alias: &str,
        query: &proto::Query,
        collectors: Vec<proto::Collector>,
    ) -> SummaResult<Vec<proto::CollectorOutput>> {
        let searcher = self.index_reader.searcher();
        let parsed_query = self.query_parser.read().await.parse_query(&query)?;
        let mut multi_collector = MultiCollector::new();
        let extractors: Vec<Box<dyn FruitExtractor>> = collectors
            .into_iter()
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
        #[cfg(feature = "async-search")]
        let (multi_fruit, searcher) = tokio::task::spawn_blocking(move || (searcher.search(&parsed_query, &multi_collector), searcher)).await?;
        #[cfg(not(feature = "async-search"))]
        let multi_fruit = searcher.search(&parsed_query, &multi_collector);
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
            .map(|field_name| self.cached_schema.get_field(field_name).unwrap())
            .collect();
        for extractor in extractors.into_iter() {
            collector_outputs.push(
                extractor
                    .extract_async(external_index_alias, &mut multi_fruit, &searcher, &multi_fields)
                    .await?,
            )
        }
        Ok(collector_outputs)
    }
}

impl std::fmt::Debug for IndexHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("IndexHolder").field("index_name", &self.index_name).finish()
    }
}
