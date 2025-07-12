//! Index - multi-segment async search index
//!
//! The `Index` is the central concept that provides:
//! - `Index::create()` / `Index::open()` - create or open an index
//! - `index.writer()` - get an IndexWriter for adding documents
//! - `index.reader()` - get an IndexReader for searching (with reload policy)
//!
//! The Index owns the SegmentManager which handles segment lifecycle and tracking.

#[cfg(feature = "native")]
use crate::dsl::Schema;
#[cfg(feature = "native")]
use crate::error::Result;
#[cfg(feature = "native")]
use std::sync::Arc;

mod searcher;
pub use searcher::Searcher;

#[cfg(feature = "native")]
mod primary_key;
#[cfg(feature = "native")]
mod reader;
#[cfg(feature = "native")]
mod vector_builder;
#[cfg(all(feature = "wasm", not(feature = "native")))]
mod wasm_writer;
#[cfg(feature = "native")]
mod writer;
#[cfg(feature = "native")]
pub use primary_key::PrimaryKeyIndex;
#[cfg(feature = "native")]
pub use reader::IndexReader;
#[cfg(all(feature = "wasm", not(feature = "native")))]
pub use wasm_writer::IndexWriter as WasmIndexWriter;
#[cfg(feature = "native")]
pub use writer::{IndexWriter, PreparedCommit};

mod metadata;
pub use metadata::{FieldVectorMeta, INDEX_META_FILENAME, IndexMetadata, VectorIndexState};

#[cfg(feature = "native")]
mod helpers;
#[cfg(feature = "native")]
pub use helpers::{
    IndexingStats, SchemaConfig, SchemaFieldConfig, create_index_at_path, create_index_from_sdl,
    index_documents_from_reader, index_json_document, parse_schema,
};

/// Default file name for the slice cache
pub const SLICE_CACHE_FILENAME: &str = "index.slicecache";

/// Index configuration
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Number of threads for CPU-intensive tasks (search parallelism)
    pub num_threads: usize,
    /// Number of parallel segment builders (documents distributed round-robin)
    pub num_indexing_threads: usize,
    /// Number of threads for parallel block compression within each segment
    pub num_compression_threads: usize,
    /// Block cache size for term dictionary per segment
    pub term_cache_blocks: usize,
    /// Block cache size for document store per segment
    pub store_cache_blocks: usize,
    /// Max memory (bytes) across all builders before auto-commit (global limit)
    pub max_indexing_memory_bytes: usize,
    /// Merge policy for background segment merging
    pub merge_policy: Box<dyn crate::merge::MergePolicy>,
    /// Index optimization mode (adaptive, size-optimized, performance-optimized)
    pub optimization: crate::structures::IndexOptimization,
    /// Reload interval in milliseconds for IndexReader (how often to check for new segments)
    pub reload_interval_ms: u64,
    /// Maximum number of concurrent background merges (default: 4)
    pub max_concurrent_merges: usize,
    /// Wall-clock budget for merge-time BP reorder per field (only applies
    /// when the index has `reorder_on_merge`). A truncated pass still writes
    /// a valid, better-ordered segment; it is marked `bp_converged = false`
    /// and the background optimizer deepens it later (warm-started).
    /// `None` = unbudgeted (BP runs to full depth inside the merge, which can
    /// hold a merge slot for 10-30+ minutes on 10M+ doc outputs).
    pub merge_bp_time_budget: Option<std::time::Duration>,
    /// Memory budget (bytes) for the BP forward index during reorder passes
    /// (merge-time and background). When a large segment's forward index
    /// would exceed this, the highest-df dims are dropped from BP's input
    /// (logged loudly) — clustering quality degrades gracefully. Production
    /// evidence: 18M-doc merges exceed the 2 GB default and drop ~10% of
    /// eligible dims; hosts with headroom should raise this.
    pub bp_memory_budget_bytes: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        #[cfg(feature = "native")]
        let indexing_threads = crate::default_indexing_threads();
        #[cfg(not(feature = "native"))]
        let indexing_threads = 1;

        #[cfg(feature = "native")]
        let compression_threads = crate::default_compression_threads();
        #[cfg(not(feature = "native"))]
        let compression_threads = 1;

        Self {
            num_threads: indexing_threads,
            num_indexing_threads: 1, // Increase to 2+ for production to avoid stalls during segment build
            num_compression_threads: compression_threads,
            term_cache_blocks: 256,
            store_cache_blocks: 32,
            max_indexing_memory_bytes: 256 * 1024 * 1024, // 256 MB default
            // large_scale: wide fan-in + budget/scored selection. Safe for
            // small indexes too (tier floors only shape *when* segments
            // merge); merge-time BP is wall-clock budgeted, so giant merges
            // cannot hold slots indefinitely.
            merge_policy: Box::new(crate::merge::TieredMergePolicy::large_scale()),
            optimization: crate::structures::IndexOptimization::default(),
            reload_interval_ms: 1000, // 1 second default
            max_concurrent_merges: 4,
            merge_bp_time_budget: Some(std::time::Duration::from_secs(600)),
            // 24 GB — mirrors segment::reorder::DEFAULT_MEMORY_BUDGET (that
            // module is native-only; IndexConfig also compiles for wasm).
            // A cap, not an allocation: usage is proportional to the segment
            // being reordered (~4 B/posting + ~28 B/doc). Sized from prod
            // evidence: a 58M-doc/5B-posting pass estimated 20.1 GB, which
            // 8/16 GB budgets trimmed by dropping highest-df dims.
            // 24 GB overflows 32-bit usize (wasm32) — reorder never runs
            // there, so any large value works; use usize::MAX.
            #[cfg(target_pointer_width = "64")]
            bp_memory_budget_bytes: 24 * 1024 * 1024 * 1024,
            #[cfg(not(target_pointer_width = "64"))]
            bp_memory_budget_bytes: usize::MAX,
        }
    }
}

/// Multi-segment async Index
///
/// The central concept for search. Owns segment lifecycle and provides:
/// - `Index::create()` / `Index::open()` - create or open an index
/// - `index.writer()` - get an IndexWriter for adding documents
/// - `index.reader()` - get an IndexReader for searching with reload policy
///
/// All segment management is delegated to SegmentManager.
#[cfg(feature = "native")]
pub struct Index<D: crate::directories::DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    config: IndexConfig,
    /// Segment manager - owns segments, tracker, metadata, and trained structures
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Cached reader (created lazily, reused across calls)
    cached_reader: tokio::sync::OnceCell<IndexReader<D>>,
}

#[cfg(feature = "native")]
impl<D: crate::directories::DirectoryWriter + 'static> Index<D> {
    /// Create a new index in the directory
    pub async fn create(directory: D, schema: Schema, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);
        let schema = Arc::new(schema);
        // Directory-layer metrics (cold writes, lazy reads) carry the index label
        directory.set_index_label(schema.index_label());
        let metadata = IndexMetadata::new((*schema).clone());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
            config.merge_bp_time_budget,
            config.bp_memory_budget_bytes,
        ));

        // Save initial metadata
        segment_manager.update_metadata(|_| {}).await?;

        Ok(Self {
            directory,
            schema,
            config,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Open an existing index from a directory
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let directory = Arc::new(directory);

        // Load metadata (includes schema)
        let metadata = IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());
        // Directory-layer metrics (cold writes, lazy reads) carry the index label
        directory.set_index_label(schema.index_label());

        let segment_manager = Arc::new(crate::merge::SegmentManager::new(
            Arc::clone(&directory),
            Arc::clone(&schema),
            metadata,
            config.merge_policy.clone_box(),
            config.term_cache_blocks,
            config.max_concurrent_merges,
            config.merge_bp_time_budget,
            config.bp_memory_budget_bytes,
        ));

        // Load trained structures into SegmentManager's ArcSwap
        segment_manager.load_and_publish_trained().await;

        Ok(Self {
            directory,
            schema,
            config,
            segment_manager,
            cached_reader: tokio::sync::OnceCell::new(),
        })
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the schema as an Arc reference (avoids clone when Arc is needed)
    pub fn schema_arc(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Get a reference to the underlying directory
    pub fn directory(&self) -> &D {
        &self.directory
    }

    /// Get the segment manager
    pub fn segment_manager(&self) -> &Arc<crate::merge::SegmentManager<D>> {
        &self.segment_manager
    }

    /// Get an IndexReader for searching (with reload policy)
    ///
    /// The reader is cached and reused across calls. The reader's internal
    /// searcher will reload segments based on its reload interval (configurable via IndexConfig).
    pub async fn reader(&self) -> Result<&IndexReader<D>> {
        self.cached_reader
            .get_or_try_init(|| async {
                IndexReader::from_segment_manager(
                    Arc::clone(&self.schema),
                    Arc::clone(&self.segment_manager),
                    self.config.term_cache_blocks,
                    self.config.reload_interval_ms,
                )
                .await
            })
            .await
    }

    /// Get the config
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Get segment readers for query execution (convenience method)
    pub async fn segment_readers(&self) -> Result<Vec<Arc<crate::segment::SegmentReader>>> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        Ok(searcher.segment_readers().to_vec())
    }

    /// Total number of documents across all segments
    pub async fn num_docs(&self) -> Result<u32> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        Ok(searcher.num_docs())
    }

    /// Get default fields for search
    pub fn default_fields(&self) -> Vec<crate::Field> {
        if !self.schema.default_fields().is_empty() {
            self.schema.default_fields().to_vec()
        } else {
            self.schema
                .fields()
                .filter(|(_, entry)| {
                    entry.indexed && entry.field_type == crate::dsl::FieldType::Text
                })
                .map(|(field, _)| field)
                .collect()
        }
    }

    /// Get tokenizer registry
    pub fn tokenizers(&self) -> Arc<crate::tokenizer::TokenizerRegistry> {
        Arc::new(crate::tokenizer::TokenizerRegistry::default())
    }

    /// Create a query parser for this index
    pub fn query_parser(&self) -> crate::dsl::QueryLanguageParser {
        let default_fields = self.default_fields();
        let tokenizers = self.tokenizers();

        let query_routers = self.schema.query_routers();
        if !query_routers.is_empty()
            && let Ok(router) = crate::dsl::QueryFieldRouter::from_rules(query_routers)
        {
            return crate::dsl::QueryLanguageParser::with_router(
                Arc::clone(&self.schema),
                default_fields,
                tokenizers,
                router,
            );
        }

        crate::dsl::QueryLanguageParser::new(Arc::clone(&self.schema), default_fields, tokenizers)
    }

    /// Parse and search using a query string
    pub async fn query(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<crate::query::SearchResponse> {
        self.query_offset(query_str, limit, 0).await
    }

    /// Query with offset for pagination
    pub async fn query_offset(
        &self,
        query_str: &str,
        limit: usize,
        offset: usize,
    ) -> Result<crate::query::SearchResponse> {
        let parser = self.query_parser();
        let query = parser
            .parse(query_str)
            .map_err(crate::error::Error::Query)?;
        self.search_offset(query.as_ref(), limit, offset).await
    }

    /// Search and return results
    pub async fn search(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<crate::query::SearchResponse> {
        self.search_offset(query, limit, 0).await
    }

    /// Search with offset for pagination
    pub async fn search_offset(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<crate::query::SearchResponse> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;

        #[cfg(feature = "sync")]
        let (results, total_seen) = {
            // Sync search: rayon handles segment parallelism internally.
            // On multi-threaded tokio, use block_in_place to yield the worker;
            // on single-threaded (tests), call directly.
            let runtime_flavor = tokio::runtime::Handle::current().runtime_flavor();
            if runtime_flavor == tokio::runtime::RuntimeFlavor::MultiThread {
                tokio::task::block_in_place(|| {
                    searcher.search_with_offset_and_count_sync(query, limit, offset)
                })?
            } else {
                searcher.search_with_offset_and_count_sync(query, limit, offset)?
            }
        };

        #[cfg(not(feature = "sync"))]
        let (results, total_seen) = {
            searcher
                .search_with_offset_and_count(query, limit, offset)
                .await?
        };

        let total_hits = total_seen;
        let hits: Vec<crate::query::SearchHit> = results
            .into_iter()
            .map(|result| crate::query::SearchHit {
                address: crate::query::DocAddress::new(result.segment_id, result.doc_id),
                score: result.score,
                matched_fields: result.extract_ordinals(),
            })
            .collect();

        Ok(crate::query::SearchResponse { hits, total_hits })
    }

    /// Get a document by its unique address
    pub async fn get_document(
        &self,
        address: &crate::query::DocAddress,
    ) -> Result<Option<crate::dsl::Document>> {
        let reader = self.reader().await?;
        let searcher = reader.searcher().await?;
        searcher.get_document(address).await
    }

    /// Get posting lists for a term across all segments
    pub async fn get_postings(
        &self,
        field: crate::Field,
        term: &[u8],
    ) -> Result<
        Vec<(
            Arc<crate::segment::SegmentReader>,
            crate::structures::BlockPostingList,
        )>,
    > {
        let segments = self.segment_readers().await?;
        let mut results = Vec::new();

        for segment in segments {
            if let Some(postings) = segment.get_postings(field, term).await? {
                results.push((segment, postings));
            }
        }

        Ok(results)
    }
}

/// Native-only methods for Index
#[cfg(feature = "native")]
impl<D: crate::directories::DirectoryWriter + 'static> Index<D> {
    /// Get an IndexWriter for adding documents
    pub fn writer(&self) -> writer::IndexWriter<D> {
        writer::IndexWriter::from_index(self)
    }
}

#[cfg(test)]
mod tests;

// (tests moved to index/tests/ module)
