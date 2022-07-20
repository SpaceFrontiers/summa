use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::path::Path;
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
use serde::Deserialize;
use summa_proto::proto;
use summa_proto::proto::IndexAttributes;
use tantivy::collector::{Collector, MultiCollector, MultiFruit};
use tantivy::directory::OwnedBytes;
use tantivy::query::{EnableScoring, Query};
use tantivy::schema::{Field, Schema};
use tantivy::space_usage::SearcherSpaceUsage;
use tantivy::{Directory, Index, IndexBuilder, IndexReader, Opstamp, ReloadPolicy, Searcher};
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use tracing::{instrument, trace};

use super::SummaSegmentAttributes;
use super::{build_fruit_extractor, default_tokenizers, FruitExtractor, ProtoQueryParser};
use crate::components::fruit_extractors::IntermediateExtractionResult;
use crate::components::segment_attributes::SegmentAttributesMergerImpl;
use crate::components::{IndexWriterHolder, SummaDocument};
use crate::configs::ConfigProxy;
use crate::directories::{CachingDirectory, ExternalRequest, ExternalRequestGenerator, FileStats, HotDirectory, NetworkDirectory, StaticDirectoryCache};
use crate::errors::{SummaResult, ValidationError};
use crate::proto_traits::Wrapper;
use crate::Error;

pub struct IndexHolder {
    index_engine_config: Arc<dyn ConfigProxy<proto::IndexEngineConfig>>,
    index_name: String,
    index: Index,
    cached_index_attributes: Option<IndexAttributes>,
    cached_schema: Schema,
    cached_multi_fields: HashSet<Field>,
    index_reader: IndexReader,
    index_writer_holder: Option<Arc<RwLock<IndexWriterHolder>>>,
    query_parser: ProtoQueryParser,
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

#[derive(Deserialize)]
struct LightMeta {
    pub opstamp: Opstamp,
}

pub async fn read_opstamp<D: Directory>(directory: &D) -> SummaResult<Opstamp> {
    let meta = directory.atomic_read_async(Path::new("meta.json")).await.map_err(|e| Error::Anyhow(e.into()))?;
    let meta_string = String::from_utf8(meta).map_err(|e| Error::Anyhow(e.into()))?;
    let meta_json: LightMeta = match serde_json::from_str(&meta_string) {
        Ok(meta_json) => meta_json,
        Err(e) => {
            error!(action = "invalid_json", json = meta_string);
            Err(e)?
        }
    };
    Ok(meta_json.opstamp)
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
pub fn register_default_tokenizers(index: &Index) {
    for (tokenizer_name, tokenizer) in &default_tokenizers() {
        index.tokenizers().register(tokenizer_name, tokenizer.clone())
    }
}

/// Cleanup after index deletion
///
/// Consumers are stopped, then `IndexConfig` is removed from `CoreConfig`
/// and then directory with the index is deleted.
pub async fn cleanup_index(index_engine_config: proto::IndexEngineConfig) -> SummaResult<()> {
    match index_engine_config.config {
        #[cfg(feature = "fs")]
        Some(proto::index_engine_config::Config::File(ref config)) => {
            info!(action = "delete_directory", directory = ?config.path);
            tokio::fs::remove_dir_all(&config.path)
                .await
                .map_err(|e| Error::IO((e, Some(std::path::PathBuf::from(&config.path)))))?;
        }
        _ => (),
    };
    Ok(())
}

fn wrap_with_caches<D: Directory>(
    directory: D,
    hotcache_bytes: Option<OwnedBytes>,
    cache_config: Option<proto::CacheConfig>,
    opstamp: Opstamp,
) -> SummaResult<Box<dyn Directory>> {
    let static_cache = hotcache_bytes
        .map(|hotcache_bytes| StaticDirectoryCache::open(hotcache_bytes, opstamp))
        .transpose()?;
    info!(action = "opened_static_cache", static_cache = ?static_cache);
    let file_lengths = static_cache
        .as_ref()
        .map(|static_cache| static_cache.file_lengths().clone())
        .unwrap_or_default();

    info!(action = "read_file_lengths", file_lengths = ?file_lengths);
    let file_stats = FileStats::from_file_lengths(file_lengths);
    let next_directory = if let Some(cache_config) = cache_config {
        Box::new(CachingDirectory::bounded(Arc::new(directory), cache_config.cache_size as usize, file_stats)) as Box<dyn Directory>
    } else {
        Box::new(directory) as Box<dyn Directory>
    };
    info!(action = "wrapped_with_cache");
    Ok(match static_cache {
        Some(static_cache) => {
            let hot_directory = HotDirectory::open(next_directory, static_cache);
            info!(action = "opened_hotcache", hot_directory = ?hot_directory);
            Box::new(hot_directory?)
        }
        None => next_directory,
    })
}

impl IndexHolder {
    /// Sets up `IndexHolder`
    pub fn create_holder(
        core_config: &crate::configs::core::Config,
        mut index: Index,
        index_name: &str,
        index_engine_config: Arc<dyn ConfigProxy<proto::IndexEngineConfig>>,
        merge_policy: Option<proto::MergePolicy>,
        query_parser_config: proto::QueryParserConfig,
    ) -> SummaResult<IndexHolder> {
        register_default_tokenizers(&index);

        index.settings_mut().docstore_compress_threads = core_config.doc_store_compress_threads;
        index.set_segment_attributes_merger(Arc::new(SegmentAttributesMergerImpl::<SummaSegmentAttributes>::new()));

        let metas = index.load_metas()?;
        let cached_schema = index.schema();
        let cached_index_attributes: Option<IndexAttributes> = metas.index_attributes()?;
        let cached_multi_fields = cached_index_attributes
            .as_ref()
            .map(|index_attributes| {
                index_attributes
                    .multi_fields
                    .iter()
                    .map(|field_name| Ok::<_, Error>(cached_schema.get_field(field_name)?))
                    .collect()
            })
            .transpose()?
            .unwrap_or_default();

        let query_parser = ProtoQueryParser::for_index(index_name, &index, query_parser_config)?;
        let index_reader = index
            .reader_builder()
            .doc_store_cache_num_blocks(core_config.doc_store_cache_num_blocks)
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()?;
        index_reader.reload()?;

        let index_writer_holder = if let Some(writer_threads) = &core_config.writer_threads {
            let merge_policy = Wrapper::from(merge_policy).into();
            info!(action = "create_index_writer", merge_policy = ?merge_policy, writer_threads = ?writer_threads, writer_heap_size_bytes = core_config.writer_heap_size_bytes);
            Some(Arc::new(RwLock::new(IndexWriterHolder::create(
                &index,
                writer_threads.clone(),
                core_config.writer_heap_size_bytes as usize,
                merge_policy,
            )?)))
        } else {
            None
        };

        Ok(IndexHolder {
            index_engine_config,
            index_name: index_name.to_string(),
            index: index.clone(),
            query_parser,
            cached_schema,
            cached_index_attributes,
            cached_multi_fields,
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

    /// Creates index and sets it up via `setup`
    #[instrument(skip_all)]
    pub fn create_memory_index(index_builder: IndexBuilder) -> SummaResult<Index> {
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

    /// Attaches index and sets it up via `setup`
    #[instrument(skip_all)]
    #[cfg(feature = "fs")]
    pub async fn open_file_index(file_engine_config: &proto::FileEngineConfig) -> SummaResult<Index> {
        let index = Index::open_in_dir(&file_engine_config.path)?;
        info!(action = "opened", config = ?file_engine_config);
        Ok(index)
    }

    pub async fn open_remote_index<
        TExternalRequest: ExternalRequest + 'static,
        TExternalRequestGenerator: ExternalRequestGenerator<TExternalRequest> + 'static,
    >(
        remote_engine_config: proto::RemoteEngineConfig,
        read_hotcache: bool,
    ) -> SummaResult<Index> {
        info!(action = "opening_network_directory", config = ?remote_engine_config, read_hotcache = read_hotcache);
        let network_directory = NetworkDirectory::open(Box::new(TExternalRequestGenerator::new(remote_engine_config.clone())));
        let opstamp = read_opstamp(&network_directory).await?;
        let hotcache_bytes = match network_directory
            .get_network_file_handle(format!("hotcache.{}.bin", opstamp).as_ref())
            .do_read_bytes_async(None)
            .await
        {
            Ok(hotcache_bytes) => {
                if read_hotcache {
                    info!(action = "read_hotcache", len = hotcache_bytes.len());
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
        let directory = wrap_with_caches(network_directory, hotcache_bytes, remote_engine_config.cache_config, opstamp)?;
        Ok(Index::open_async(directory).await?)
    }

    /// Compression
    pub fn compression(&self) -> proto::Compression {
        Wrapper::from(self.index_reader().searcher().index().settings().docstore_compression).into_inner()
    }

    /// Load index attributes from meta.json
    pub fn index_attributes(&self) -> Option<&IndexAttributes> {
        self.cached_index_attributes.as_ref()
    }

    /// How to resolve conflicts on inserting new documents
    pub fn conflict_strategy(&self) -> proto::ConflictStrategy {
        self.index_attributes().as_ref().map(|c| c.conflict_strategy()).unwrap_or_default()
    }

    /// Index name
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// `IndexReader` singleton
    pub fn index_reader(&self) -> &IndexReader {
        &self.index_reader
    }

    /// Return internal Tantivy index
    pub fn index(&self) -> &Index {
        &self.index
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

    /// Multi fields
    pub fn multi_fields(&self) -> &HashSet<Field> {
        &self.cached_multi_fields
    }

    /// Return internal Tantivy index
    pub fn real_directory(&self) -> &dyn Directory {
        self.index.directory().real_directory()
    }

    /// Load term dictionaries into memory
    pub async fn partial_warmup<T: AsRef<str>>(&self, load_dictionaries: bool, fields: &[T]) -> SummaResult<()> {
        let searcher = self.index_reader().searcher();
        let mut warm_up_futures = Vec::new();
        let default_fields = fields
            .iter()
            .map(|field_name| {
                self.cached_schema
                    .find_field(field_name.as_ref())
                    .map(|x| x.0)
                    .ok_or_else(|| ValidationError::MissingField(field_name.as_ref().to_string()))
            })
            .collect::<Result<HashSet<_>, _>>()?;
        for field in default_fields {
            for segment_reader in searcher.segment_readers() {
                let inverted_index = segment_reader.inverted_index_async(field).await?.clone();
                if load_dictionaries {
                    warm_up_futures.push(async move {
                        let dict = inverted_index.terms();
                        info!(action = "warming_up_dictionary", index_name = ?self.index_name());
                        dict.warm_up_dictionary_async().await
                    });
                }
            }
        }
        info!(action = "warming_up");
        try_join_all(warm_up_futures).await?;
        Ok(())
    }

    /// Load all index files into memory
    pub async fn full_warmup(&self) -> SummaResult<()> {
        let managed_directory = self.index.directory();
        info!(action = "warming_up");
        join_all(managed_directory.list_managed_files().into_iter().map(move |file| {
            let file_name = file.to_string_lossy().to_string();
            async move {
                info!(action = "start_reading_file", index_name = ?self.index_name(), file_name = ?file_name);
                managed_directory.atomic_read_async(&file).await.map_err(|e| Error::Tantivy(e.into()))?;
                info!(action = "finished_reading_file", index_name = ?self.index_name(), file_name = ?file_name);
                Ok(())
            }
        }))
        .await
        .into_iter()
        .collect::<SummaResult<Vec<_>>>()?;
        Ok(())
    }

    /// Runs a query on the segment readers wrapped by the searcher asynchronously.
    pub async fn search_in_segments(
        &self,
        searcher: &Searcher,
        query: &dyn Query,
        collector: &MultiCollector<'_>,
        is_fieldnorms_scoring_enabled: Option<bool>,
    ) -> tantivy::Result<MultiFruit> {
        let enabled_scoring = match (is_fieldnorms_scoring_enabled, collector.requires_scoring()) {
            (Some(true), true) | (None, true) => EnableScoring::enabled_from_searcher(searcher),
            (Some(false), true) => EnableScoring::enabled_from_searcher_without_fieldnorms(searcher),
            (_, false) => EnableScoring::disabled_from_searcher(searcher),
        };
        let segment_readers = searcher.segment_readers();
        trace!(index_name = ?self.index_name, action = "weight");
        let weight = query.weight_async(enabled_scoring).await?;
        trace!(index_name = ?self.index_name, action = "collect_segment");
        let fruits = join_all(segment_readers.iter().enumerate().map(|(segment_ord, segment_reader)| {
            let weight_ref = weight.as_ref();
            collector.collect_segment_async(weight_ref, segment_ord as u32, segment_reader)
        }))
        .await
        .into_iter()
        .collect::<tantivy::Result<Vec<_>>>()?;
        collector.merge_fruits(fruits)
    }

    /// Search `query` in the `IndexHolder` and collecting `Fruit` with a list of `collectors`
    pub async fn search(
        &self,
        index_alias: &str,
        query: proto::query::Query,
        collectors: Vec<proto::Collector>,
        is_fieldnorms_scoring_enabled: Option<bool>,
    ) -> SummaResult<Vec<IntermediateExtractionResult>> {
        let searcher = self.index_reader().searcher();
        trace!(action = "parse_query", index_name = ?self.index_name, query = ?query);
        #[cfg(feature = "tokio-rt")]
        let parsed_query = {
            let query_parser = self.query_parser.clone();
            tokio::task::spawn_blocking(move || query_parser.parse_query(query)).await??
        };
        #[cfg(not(feature = "tokio-rt"))]
        let parsed_query = self.query_parser.parse_query(query)?;
        let mut multi_collector = MultiCollector::new();
        trace!(action = "build_extractors", index_name = ?self.index_name);
        let extractors: Vec<Box<dyn FruitExtractor>> = collectors
            .into_iter()
            .map(|collector_proto| build_fruit_extractor(self, index_alias, searcher.clone(), collector_proto, &parsed_query, &mut multi_collector))
            .collect::<SummaResult<_>>()?;
        trace!(
            target: "query",
            index_name = ?self.index_name,
            parsed_query = ?parsed_query,
            is_fieldnorms_scoring_enabled = is_fieldnorms_scoring_enabled,
        );
        #[cfg(feature = "metrics")]
        let start_time = Instant::now();
        let mut multi_fruit = self
            .search_in_segments(&searcher, &parsed_query, &multi_collector, is_fieldnorms_scoring_enabled)
            .await?;
        #[cfg(feature = "metrics")]
        self.search_times_meter.record(
            &Context::current(),
            start_time.elapsed().as_secs_f64(),
            &[KeyValue::new("index_name", self.index_name.to_owned())],
        );
        let mut collector_outputs = Vec::with_capacity(extractors.len());
        trace!(action = "extract");
        for extractor in extractors.into_iter() {
            collector_outputs.push(extractor.extract(&mut multi_fruit)?)
        }
        trace!(action = "extracted");
        Ok(collector_outputs)
    }

    /// Delete `SummaDocument` by `unq`
    pub async fn delete_documents(&self, query: proto::query::Query) -> SummaResult<u64> {
        #[cfg(feature = "tokio-rt")]
        let parsed_query = {
            let query_parser = self.query_parser.clone();
            tokio::task::spawn_blocking(move || query_parser.parse_query(query)).await??
        };
        #[cfg(not(feature = "tokio-rt"))]
        let parsed_query = self.query_parser.parse_query(query)?;
        self.index_writer_holder()?.read().await.delete_by_query(parsed_query)
    }

    /// Index generic `SummaDocument`
    ///
    /// `IndexUpdater` bounds unbounded `SummaDocument` inside
    pub async fn index_document(&self, document: SummaDocument<'_>) -> SummaResult<()> {
        let document = document.bound_with(&self.index.schema()).try_into()?;
        self.index_writer_holder()?.read().await.index_document(document, self.conflict_strategy())
    }

    /// Index multiple documents at a time
    pub async fn index_bulk(&self, documents: &Vec<Vec<u8>>, conflict_strategy: Option<proto::ConflictStrategy>) -> SummaResult<(u64, u64)> {
        let (mut success_docs, mut failed_docs) = (0u64, 0u64);
        let index_writer_holder = self.index_writer_holder()?.read().await;
        let conflict_strategy = conflict_strategy.unwrap_or_else(|| self.conflict_strategy());
        for document in documents {
            match SummaDocument::UnboundJsonBytes(document)
                .bound_with(&self.index.schema())
                .try_into()
                .and_then(|document| index_writer_holder.index_document(document, conflict_strategy))
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

    #[cfg(feature = "tokio-rt")]
    pub fn documents<O: Send + 'static>(
        &self,
        searcher: &Searcher,
        documents_modifier: impl Fn(tantivy::schema::Document) -> Option<O> + Clone + Send + Sync + 'static,
    ) -> SummaResult<tokio::sync::mpsc::Receiver<O>> {
        let segment_readers = searcher.segment_readers();
        let (tx, rx) = tokio::sync::mpsc::channel(segment_readers.len() * 2 + 1);
        for segment_reader in segment_readers {
            let documents_modifier = documents_modifier.clone();
            let tx = tx.clone();
            let segment_reader = segment_reader.clone();
            let span = tracing::Span::current();
            tokio::task::spawn_blocking(move || {
                span.in_scope(|| {
                    let store_reader = segment_reader.get_store_reader(1)?;
                    for document in store_reader.iter(segment_reader.alive_bitset()) {
                        let Ok(document) = document
                        else {
                            info!(action = "broken_document", document = ?document);
                            return Ok::<_, Error>(());
                        };
                        if let Some(document) = documents_modifier(document) {
                            if tx.blocking_send(document).is_err() {
                                info!(action = "documents_client_dropped");
                                return Ok::<_, Error>(());
                            }
                        }
                    }
                    Ok(())
                })
            });
        }
        Ok(rx)
    }

    pub fn space_usage(&self) -> SummaResult<SearcherSpaceUsage> {
        let index_reader = self.index_reader();
        index_reader.reload()?;
        Ok(index_reader.searcher().space_usage()?)
    }
}

#[cfg(test)]
pub mod tests {
    use std::error::Error;
    use std::sync::Arc;

    use serde_json::json;
    use summa_proto::proto;
    use summa_proto::proto::ConflictStrategy;
    use tantivy::collector::{Count, TopDocs};
    use tantivy::query::{AllQuery, TermQuery};
    use tantivy::schema::IndexRecordOption;
    use tantivy::{doc, Document, IndexBuilder, Term};

    use crate::components::index_holder::register_default_tokenizers;
    use crate::components::test_utils::{create_test_schema, generate_documents};
    use crate::components::IndexWriterHolder;
    use crate::configs::core::WriterThreads;

    #[test]
    fn test_deletes() -> Result<(), Box<dyn Error>> {
        let schema = create_test_schema();
        let index = IndexBuilder::new()
            .schema(schema.clone())
            .index_attributes(proto::IndexAttributes {
                unique_fields: vec!["id".to_string()],
                ..Default::default()
            })
            .create_in_ram()?;
        register_default_tokenizers(&index);
        let mut index_writer_holder = IndexWriterHolder::create(
            &index,
            WriterThreads::N(12),
            1024 * 1024 * 1024,
            Arc::new(tantivy::merge_policy::LogMergePolicy::default()),
        )?;
        let mut last_document = None;
        for document in generate_documents(&schema, 10000) {
            let document: Document = document.bound_with(&schema).try_into()?;
            last_document = Some(document.clone());
            index_writer_holder.index_document(document, ConflictStrategy::Merge)?;
        }
        let last_document = last_document.clone().unwrap();
        let id = last_document.get_first(schema.get_field("id").unwrap()).unwrap().as_i64().unwrap();
        let modified_last_document = doc!(
            schema.get_field("id").unwrap() => id,
            schema.get_field("issued_at").unwrap() => 100i64
        );
        index_writer_holder.commit()?;
        for document in generate_documents(&schema, 1000) {
            let document = document.bound_with(&schema).try_into()?;
            index_writer_holder.index_document(modified_last_document.clone(), ConflictStrategy::Merge)?;
            index_writer_holder.index_document(document, ConflictStrategy::Merge)?;
        }
        index_writer_holder.commit()?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let counter = searcher.search(
            &TermQuery::new(Term::from_field_i64(schema.get_field("id").unwrap(), id), IndexRecordOption::WithFreqs),
            &Count,
        )?;
        assert_eq!(counter, 1);
        Ok(())
    }

    #[test]
    fn test_mapped_fields() -> Result<(), Box<dyn Error>> {
        let schema = create_test_schema();
        let metadata_field = schema.get_field("metadata").expect("no field");
        let tags_field = schema.get_field("tags").expect("no field");
        let title_field = schema.get_field("title").expect("no field");
        let extra_field = schema.get_field("extra").expect("no field");
        let index = IndexBuilder::new()
            .schema(schema.clone())
            .index_attributes(proto::IndexAttributes {
                mapped_fields: vec![
                    proto::MappedField {
                        source_field: "metadata.isbn".to_string(),
                        target_field: "extra".to_string(),
                    },
                    proto::MappedField {
                        source_field: "tags".to_string(),
                        target_field: "extra".to_string(),
                    },
                    proto::MappedField {
                        source_field: "title".to_string(),
                        target_field: "extra".to_string(),
                    },
                ],
                ..Default::default()
            })
            .create_in_ram()?;
        register_default_tokenizers(&index);
        let mut index_writer_holder = IndexWriterHolder::create(
            &index,
            WriterThreads::N(12),
            1024 * 1024 * 1024,
            Arc::new(tantivy::merge_policy::LogMergePolicy::default()),
        )?;
        index_writer_holder.index_document(
            doc!(
                title_field => "Hitchhiker's guide",
                metadata_field => json!({"isbn": ["100", "200", "300", "500", "600"], "another_title": "Super Title"}),
                tags_field => "reality",
            ),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.index_document(
            doc!(
                title_field => "Envy of the Stars",
                metadata_field => json!({"isbn": "500"}),
                tags_field => "scifi",
                tags_field => "novel",
            ),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.commit()?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let docs = searcher
            .search(
                &TermQuery::new(Term::from_field_text(extra_field, "500"), IndexRecordOption::Basic),
                &TopDocs::with_limit(10),
            )?
            .into_iter()
            .map(|x| {
                searcher
                    .doc(x.1)
                    .unwrap()
                    .get_first(title_field)
                    .unwrap()
                    .as_text()
                    .map(|x| x.to_string())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(format!("{:?}", docs), "[\"Envy of the Stars\", \"Hitchhiker's guide\"]");
        let docs = searcher
            .search(
                &TermQuery::new(Term::from_field_text(extra_field, "scifi"), IndexRecordOption::Basic),
                &TopDocs::with_limit(10),
            )?
            .into_iter()
            .map(|x| {
                searcher
                    .doc(x.1)
                    .unwrap()
                    .get_first(title_field)
                    .unwrap()
                    .as_text()
                    .map(|x| x.to_string())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(format!("{:?}", docs), "[\"Envy of the Stars\"]");
        Ok(())
    }

    #[test]
    fn test_unique_json_fields() -> Result<(), Box<dyn Error>> {
        let schema = create_test_schema();
        let index = IndexBuilder::new()
            .schema(schema.clone())
            .index_attributes(proto::IndexAttributes {
                unique_fields: vec!["metadata.id".to_string()],
                ..Default::default()
            })
            .create_in_ram()?;
        register_default_tokenizers(&index);
        let mut index_writer_holder = IndexWriterHolder::create(
            &index,
            WriterThreads::N(12),
            1024 * 1024 * 1024,
            Arc::new(tantivy::merge_policy::LogMergePolicy::default()),
        )?;

        index_writer_holder.index_document(
            doc!(schema.get_field("metadata").expect("no field") => json!({"id": 1})),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.index_document(
            doc!(schema.get_field("metadata").expect("no field") => json!({"id": 2})),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.index_document(
            doc!(schema.get_field("metadata").expect("no field") => json!({"id": 3})),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.commit()?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let counter = searcher.search(&AllQuery, &Count)?;
        assert_eq!(counter, 3);
        index_writer_holder.index_document(
            doc!(schema.get_field("metadata").expect("no field") => json!({"id": "g"})),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.commit()?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let counter = searcher.search(&AllQuery, &Count)?;
        assert_eq!(counter, 4);
        index_writer_holder.index_document(
            doc!(schema.get_field("metadata").expect("no field") => json!({"id": "g"})),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.commit()?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let counter = searcher.search(&AllQuery, &Count)?;
        assert_eq!(counter, 4);
        index_writer_holder.index_document(
            doc!(schema.get_field("metadata").expect("no field") => json!({"id": 2})),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.index_document(
            doc!(schema.get_field("metadata").expect("no field") => json!({"id": 4})),
            ConflictStrategy::Merge,
        )?;
        index_writer_holder.commit()?;
        let reader = index.reader()?;
        reader.reload()?;
        let searcher = reader.searcher();
        let counter = searcher.search(&AllQuery, &Count)?;
        assert_eq!(counter, 5);
        Ok(())
    }
}
