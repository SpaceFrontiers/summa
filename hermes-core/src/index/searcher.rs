//! Searcher - read-only search over pre-built segments
//!
//! This module provides `Searcher` for read-only search access to indexes.
//! It can be used standalone (for wasm/read-only) or via `IndexReader` (for native).

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::Directory;
use crate::dsl::Schema;
use crate::error::Result;
use crate::query::LazyGlobalStats;
use crate::segment::{SegmentId, SegmentReader, TrainedVectorStructures};
#[cfg(feature = "native")]
use crate::segment::{SegmentSnapshot, SegmentTracker};

/// Immutable resources that must stay identical across `IndexReader` reloads.
/// The search pool exists only when synchronous scoring is compiled in; native
/// async-only builds retain the cache policy without spawning unused threads.
#[cfg(feature = "native")]
#[derive(Clone)]
pub(crate) struct SearcherResources {
    pub(crate) term_cache_blocks: usize,
    pub(crate) store_cache: Arc<crate::segment::SharedStoreCache>,
    pub(crate) bmp_io_gate: Arc<super::BmpIoGate>,
    pub(crate) bmp_io_concurrency: usize,
    #[cfg(feature = "sync")]
    pub(crate) search_pool: Arc<rayon::ThreadPool>,
}

#[cfg(feature = "native")]
impl SearcherResources {
    pub(crate) fn new(
        term_cache_blocks: usize,
        store_cache_budget_bytes: usize,
        num_threads: usize,
        bmp_io_concurrency: usize,
    ) -> Result<Self> {
        if num_threads == 0 {
            return Err(crate::Error::Internal(
                "IndexConfig.num_threads must be greater than zero".into(),
            ));
        }
        if bmp_io_concurrency == 0 {
            return Err(crate::Error::Internal(
                "IndexConfig.bmp_io_concurrency must be greater than zero".into(),
            ));
        }

        #[cfg(feature = "sync")]
        let search_pool = super::shared_search_pool(num_threads)?;

        Ok(Self {
            term_cache_blocks,
            store_cache: super::shared_store_cache(store_cache_budget_bytes),
            bmp_io_gate: super::shared_bmp_io_gate(bmp_io_concurrency),
            bmp_io_concurrency,
            #[cfg(feature = "sync")]
            search_pool,
        })
    }
}

/// Searcher - provides search over loaded segments
///
/// For wasm/read-only use, create via `Searcher::open()`.
/// For native use with Index, this is created via `IndexReader`.
pub struct Searcher<D: Directory + 'static> {
    /// Segment snapshot holding refs - prevents deletion during native use
    #[cfg(feature = "native")]
    _snapshot: SegmentSnapshot,
    /// PhantomData for the directory generic
    _phantom: std::marker::PhantomData<D>,
    /// Loaded segment readers
    segments: Vec<Arc<SegmentReader>>,
    /// Schema
    schema: Arc<Schema>,
    /// Default fields for search
    default_fields: Vec<crate::Field>,
    /// Tokenizers
    tokenizers: Arc<crate::tokenizer::TokenizerRegistry>,
    /// One immutable generation of all index-global ANN artifacts.
    trained_vectors: Arc<TrainedVectorStructures>,
    /// Lazy global statistics for cross-segment IDF computation
    global_stats: Arc<LazyGlobalStats>,
    /// O(1) segment lookup by segment_id
    segment_map: FxHashMap<u128, usize>,
    /// Total document count across all segments
    total_docs: u32,
    /// Bounded process-wide-by-width pool for the complete nested search tree.
    #[cfg(feature = "sync")]
    search_pool: Arc<rayon::ThreadPool>,
    /// Shared random-I/O gate and per-query wave width for BMP.
    #[cfg(feature = "native")]
    bmp_io_gate: Arc<super::BmpIoGate>,
    #[cfg(feature = "native")]
    bmp_io_concurrency: usize,
}

impl<D: Directory + 'static> Searcher<D> {
    /// Create a Searcher directly from segment IDs
    ///
    /// This is a simpler initialization path that doesn't require SegmentManager.
    /// Use this for read-only access to pre-built indexes.
    pub async fn open(
        directory: Arc<D>,
        schema: Arc<Schema>,
        segment_ids: &[String],
        term_cache_blocks: usize,
    ) -> Result<Self> {
        const STANDALONE_STORE_CACHE_BYTES: usize = 32 * 1024 * 1024;
        #[cfg(feature = "native")]
        let store_cache = super::shared_store_cache(STANDALONE_STORE_CACHE_BYTES);
        #[cfg(not(feature = "native"))]
        let store_cache = Arc::new(crate::segment::SharedStoreCache::new(
            STANDALONE_STORE_CACHE_BYTES,
        ));
        Self::create(
            directory,
            schema,
            segment_ids,
            Arc::new(TrainedVectorStructures::default()),
            term_cache_blocks,
            store_cache,
        )
        .await
    }

    /// Create from a snapshot (for native IndexReader use)
    #[cfg(feature = "native")]
    pub(crate) async fn from_snapshot(
        directory: Arc<D>,
        schema: Arc<Schema>,
        snapshot: SegmentSnapshot,
        trained_vectors: Arc<TrainedVectorStructures>,
        resources: SearcherResources,
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            snapshot.segment_ids(),
            &trained_vectors,
            resources.term_cache_blocks,
            Arc::clone(&resources.store_cache),
            &[],
        )
        .await?;

        Ok(Self {
            _snapshot: snapshot,
            _phantom: std::marker::PhantomData,
            segments,
            schema,
            default_fields,
            tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
            trained_vectors,
            global_stats,
            segment_map,
            total_docs,
            #[cfg(feature = "sync")]
            search_pool: resources.search_pool,
            bmp_io_gate: resources.bmp_io_gate,
            bmp_io_concurrency: resources.bmp_io_concurrency,
        })
    }

    /// Create from a snapshot, reusing existing segment readers for unchanged segments.
    /// This avoids re-opening mmaps, fast fields, sparse indexes, etc. for segments
    /// that weren't touched by merge.
    #[cfg(feature = "native")]
    pub(crate) async fn from_snapshot_reuse(
        directory: Arc<D>,
        schema: Arc<Schema>,
        snapshot: SegmentSnapshot,
        trained_vectors: Arc<TrainedVectorStructures>,
        resources: SearcherResources,
        existing_segments: &[Arc<SegmentReader>],
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            snapshot.segment_ids(),
            &trained_vectors,
            resources.term_cache_blocks,
            Arc::clone(&resources.store_cache),
            existing_segments,
        )
        .await?;

        Ok(Self {
            _snapshot: snapshot,
            _phantom: std::marker::PhantomData,
            segments,
            schema,
            default_fields,
            tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
            trained_vectors,
            global_stats,
            segment_map,
            total_docs,
            #[cfg(feature = "sync")]
            search_pool: resources.search_pool,
            bmp_io_gate: resources.bmp_io_gate,
            bmp_io_concurrency: resources.bmp_io_concurrency,
        })
    }

    /// Internal create method
    async fn create(
        directory: Arc<D>,
        schema: Arc<Schema>,
        segment_ids: &[String],
        trained_vectors: Arc<TrainedVectorStructures>,
        term_cache_blocks: usize,
        store_cache: Arc<crate::segment::SharedStoreCache>,
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            segment_ids,
            &trained_vectors,
            term_cache_blocks,
            store_cache,
            &[],
        )
        .await?;

        #[cfg(feature = "native")]
        let _snapshot = {
            let tracker = Arc::new(SegmentTracker::new());
            SegmentSnapshot::new(tracker, segment_ids.to_vec())
        };

        #[cfg(feature = "sync")]
        let search_pool = super::shared_search_pool(crate::default_search_threads())?;
        #[cfg(feature = "native")]
        let bmp_io_concurrency = 4;
        #[cfg(feature = "native")]
        let bmp_io_gate = super::shared_bmp_io_gate(bmp_io_concurrency);

        let _ = directory; // suppress unused warning on wasm
        Ok(Self {
            #[cfg(feature = "native")]
            _snapshot,
            _phantom: std::marker::PhantomData,
            segments,
            schema,
            default_fields,
            tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
            trained_vectors,
            global_stats,
            segment_map,
            total_docs,
            #[cfg(feature = "sync")]
            search_pool,
            #[cfg(feature = "native")]
            bmp_io_gate,
            #[cfg(feature = "native")]
            bmp_io_concurrency,
        })
    }

    /// Common loading logic shared by create and from_snapshot
    async fn load_common(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        segment_ids: &[String],
        trained_vectors: &Arc<TrainedVectorStructures>,
        term_cache_blocks: usize,
        store_cache: Arc<crate::segment::SharedStoreCache>,
        existing_segments: &[Arc<SegmentReader>],
    ) -> Result<(
        Vec<Arc<SegmentReader>>,
        Vec<crate::Field>,
        Arc<LazyGlobalStats>,
        FxHashMap<u128, usize>,
        u32,
    )> {
        let segments = Self::load_segments(
            directory,
            schema,
            segment_ids,
            trained_vectors,
            term_cache_blocks,
            store_cache,
            existing_segments,
        )
        .await?;
        let default_fields = Self::build_default_fields(schema);
        let global_stats = Arc::new(LazyGlobalStats::new(segments.clone()));
        let (segment_map, total_docs) = Self::build_lookup_tables(&segments);
        Ok((
            segments,
            default_fields,
            global_stats,
            segment_map,
            total_docs,
        ))
    }

    /// Load segment readers from IDs (parallel loading for performance).
    /// Reuses existing segment readers for unchanged segments when `existing_segments`
    /// is non-empty — avoids re-opening mmaps, fast fields, sparse indexes, etc.
    async fn load_segments(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        segment_ids: &[String],
        trained_vectors: &Arc<TrainedVectorStructures>,
        term_cache_blocks: usize,
        store_cache: Arc<crate::segment::SharedStoreCache>,
        existing_segments: &[Arc<SegmentReader>],
    ) -> Result<Vec<Arc<SegmentReader>>> {
        // Build lookup from existing segment readers for reuse
        let existing_map: FxHashMap<u128, Arc<SegmentReader>> = existing_segments
            .iter()
            .map(|seg| (seg.meta().id, Arc::clone(seg)))
            .collect();

        // Parse segment IDs from metadata. A key that fails to parse means the
        // metadata is corrupt; fail loud instead of silently serving results
        // without that segment's documents (the merge path errors with
        // Corruption on the same input — search must not disagree).
        let mut valid_segments: Vec<(usize, SegmentId)> = Vec::with_capacity(segment_ids.len());
        for (idx, id_str) in segment_ids.iter().enumerate() {
            let sid = SegmentId::from_hex(id_str).ok_or_else(|| {
                crate::error::Error::Corruption(format!(
                    "Invalid segment ID in metadata: {id_str:?}"
                ))
            })?;
            valid_segments.push((idx, sid));
        }

        // Separate into reusable and new segments
        let mut reused: Vec<(usize, Arc<SegmentReader>)> = Vec::new();
        let mut to_load: Vec<(usize, SegmentId)> = Vec::new();
        for (idx, sid) in &valid_segments {
            if let Some(existing) = existing_map.get(&sid.0) {
                reused.push((*idx, Arc::clone(existing)));
            } else {
                to_load.push((*idx, *sid));
            }
        }

        if !existing_segments.is_empty() {
            log::info!(
                "[searcher] reusing {} segment readers, loading {} new",
                reused.len(),
                to_load.len(),
            );
        }

        // Include the live directory allocation in document-cache keys.
        // Segment IDs can legitimately repeat when two independent indexes
        // are copied/opened in the same process.
        let store_cache_directory_namespace = Arc::as_ptr(directory) as usize;

        // Load only NEW segments in parallel
        let futures: Vec<_> = to_load
            .iter()
            .map(|(_, segment_id)| {
                let dir = Arc::clone(directory);
                let sch = Arc::clone(schema);
                let store_cache = Arc::clone(&store_cache);
                let sid = *segment_id;
                async move {
                    SegmentReader::open_with_store_cache(
                        dir.as_ref(),
                        sid,
                        sch,
                        term_cache_blocks,
                        store_cache_directory_namespace,
                        store_cache,
                    )
                    .await
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        // Collect newly loaded results — fail fast if any segment fails to open
        let mut loaded: Vec<(usize, Arc<SegmentReader>)> = Vec::with_capacity(valid_segments.len());

        // Add reused segments
        loaded.extend(reused);

        // Add newly loaded segments
        for ((idx, sid), result) in to_load.into_iter().zip(results) {
            match result {
                Ok(mut reader) => {
                    // Inject the single immutable index-level artifact generation.
                    reader.set_trained_vectors(Arc::clone(trained_vectors));
                    loaded.push((idx, Arc::new(reader)));
                }
                Err(e) => {
                    return Err(crate::error::Error::Internal(format!(
                        "Failed to open segment {:016x}: {:?}",
                        sid.0, e
                    )));
                }
            }
        }

        // Sort by original index to maintain deterministic ordering
        loaded.sort_by_key(|(idx, _)| *idx);

        let segments: Vec<Arc<SegmentReader>> = loaded.into_iter().map(|(_, seg)| seg).collect();

        // Keep heap, file-backed address space, and pinned residency separate.
        // Mapped bytes are not necessarily resident; process RSS is the
        // authoritative whole-process residency measurement.
        let total_docs: u64 = segments.iter().map(|s| s.meta().num_docs as u64).sum();
        let mut total_heap = 0usize;
        let mut total_file_backed = 0u64;
        let mut total_pinned = 0u64;
        let mut total_pin_intended = 0u64;
        for seg in &segments {
            let stats = seg.memory_stats();
            let heap = stats.estimated_heap_bytes();
            let file_backed = stats.file_backed_bytes();
            total_heap = total_heap.saturating_add(heap);
            total_file_backed = total_file_backed.saturating_add(file_backed);
            total_pinned = total_pinned.saturating_add(stats.pinned_metadata_bytes);
            total_pin_intended = total_pin_intended.saturating_add(stats.pin_intended_bytes);
            log::info!(
                "[searcher] segment {:016x}: docs={}, heap_estimate={} \
                 (term_cache={}, store_cache={}, sparse_vectors={}, dense_vectors={}), \
                 file_backed={} (term_bloom={}, sparse_vectors={}, dense_vectors={}), \
                 pinned_metadata={} of {} eligible \
                 (sparse_vectors={} of {}, dense_vectors={} of {})",
                stats.segment_id,
                stats.num_docs,
                crate::format_bytes(heap as u64),
                crate::format_bytes(stats.term_dict_cache_bytes as u64),
                crate::format_bytes(stats.store_cache_bytes as u64),
                crate::format_bytes(stats.sparse_heap_bytes as u64),
                crate::format_bytes(stats.dense_heap_bytes as u64),
                crate::format_bytes(file_backed),
                crate::format_bytes(stats.term_bloom_file_bytes),
                crate::format_bytes(stats.sparse_file_backed_bytes),
                crate::format_bytes(stats.dense_file_backed_bytes),
                crate::format_bytes(stats.pinned_metadata_bytes),
                crate::format_bytes(stats.pin_intended_bytes),
                crate::format_bytes(stats.sparse_pinned_metadata_bytes),
                crate::format_bytes(stats.sparse_pin_intended_bytes),
                crate::format_bytes(stats.dense_pinned_metadata_bytes),
                crate::format_bytes(stats.dense_pin_intended_bytes),
            );
        }
        // Log process RSS if available (helps diagnose OOM)
        let rss_bytes = process_rss_bytes();
        log::info!(
            "[searcher] loaded {} segments: total_docs={}, heap_estimate={}, \
             file_backed={}, pinned_metadata={} of {} eligible, \
             shared_store_cache={} in {} blocks, process_rss={}",
            segments.len(),
            total_docs,
            crate::format_bytes(total_heap as u64),
            crate::format_bytes(total_file_backed),
            crate::format_bytes(total_pinned),
            crate::format_bytes(total_pin_intended),
            crate::format_bytes(store_cache.total_bytes() as u64),
            store_cache.total_blocks(),
            crate::format_bytes(rss_bytes),
        );

        Ok(segments)
    }

    /// Build default fields from schema
    fn build_default_fields(schema: &Schema) -> Vec<crate::Field> {
        if !schema.default_fields().is_empty() {
            schema.default_fields().to_vec()
        } else {
            schema
                .fields()
                .filter(|(_, entry)| {
                    entry.indexed && entry.field_type == crate::dsl::FieldType::Text
                })
                .map(|(field, _)| field)
                .collect()
        }
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get segment readers
    pub fn segment_readers(&self) -> &[Arc<SegmentReader>] {
        &self.segments
    }

    /// Get default fields for search
    pub fn default_fields(&self) -> &[crate::Field] {
        &self.default_fields
    }

    /// Get tokenizer registry
    pub fn tokenizers(&self) -> &crate::tokenizer::TokenizerRegistry {
        &self.tokenizers
    }

    /// Get trained centroids
    pub fn trained_centroids(&self) -> &FxHashMap<u32, Arc<crate::structures::CoarseCentroids>> {
        &self.trained_vectors.centroids
    }

    pub fn trained_binary_quantizers(
        &self,
    ) -> &FxHashMap<u32, Arc<crate::structures::BinaryCoarseQuantizer>> {
        &self.trained_vectors.binary_quantizers
    }

    /// Get lazy global statistics for cross-segment IDF computation
    pub fn global_stats(&self) -> &Arc<LazyGlobalStats> {
        &self.global_stats
    }

    /// Build O(1) lookup tables from loaded segments
    fn build_lookup_tables(segments: &[Arc<SegmentReader>]) -> (FxHashMap<u128, usize>, u32) {
        let mut segment_map = FxHashMap::default();
        let mut total = 0u32;
        for (i, seg) in segments.iter().enumerate() {
            segment_map.insert(seg.meta().id, i);
            total = total.saturating_add(seg.meta().num_docs);
        }
        (segment_map, total)
    }

    /// Get total document count across all segments
    pub fn num_docs(&self) -> u32 {
        self.total_docs
    }

    /// Get O(1) segment_id → index map (used by reranker)
    pub fn segment_map(&self) -> &FxHashMap<u128, usize> {
        &self.segment_map
    }

    /// Run a bounded piece of CPU work inside this index's shared search pool.
    #[cfg(feature = "sync")]
    pub(crate) fn install_search_cpu<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.search_pool.install(operation)
    }

    /// Async-only/WASM builds execute inline because Rayon is not available.
    /// Keeping this overload free of `Send` bounds allows browser-backed file
    /// handles, whose callbacks are deliberately thread-local, to be scored.
    #[cfg(not(feature = "sync"))]
    pub(crate) fn install_search_cpu<R>(&self, operation: impl FnOnce() -> R) -> R {
        operation()
    }

    /// Get number of segments
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get a document by (segment_id, local_doc_id)
    pub async fn doc(&self, segment_id: u128, doc_id: u32) -> Result<Option<crate::dsl::Document>> {
        if let Some(&idx) = self.segment_map.get(&segment_id) {
            return self.segments[idx].doc(doc_id).await;
        }
        Ok(None)
    }

    /// Search across all segments and return aggregated results
    pub async fn search(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<Vec<crate::query::SearchResult>> {
        let (results, _) = self.search_with_count(query, limit).await?;
        Ok(results)
    }

    /// Search across all segments and return (results, total_seen)
    /// total_seen is the number of documents that were scored across all segments
    pub async fn search_with_count(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        self.search_with_offset_and_count(query, limit, 0).await
    }

    /// Search with offset for pagination
    pub async fn search_with_offset(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<crate::query::SearchResult>> {
        let (results, _) = self
            .search_with_offset_and_count(query, limit, offset)
            .await?;
        Ok(results)
    }

    /// Search with offset and return (results, total_seen)
    pub async fn search_with_offset_and_count(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        self.search_internal(query, limit, offset, false).await
    }

    /// Search with positions (ordinal tracking) and return (results, total_seen)
    ///
    /// Use this when you need per-ordinal scores for multi-valued fields.
    pub async fn search_with_positions(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        self.search_internal(query, limit, 0, true).await
    }

    /// Build the paper's single query-level top-γ superblock set, then project
    /// it back onto segment-local plans.
    ///
    /// Treating every immutable segment as an independent LSP index would
    /// multiply work by the segment count. The prepass retains one global γ
    /// while preserving Hermes's streaming segment architecture.
    fn prepare_global_lsp(
        &self,
        query: &dyn crate::query::Query,
        retrieval_depth: usize,
        parallel: bool,
    ) -> Result<Vec<Option<std::sync::Arc<crate::query::bmp::LspSegmentPlan>>>> {
        let total_start = crate::observe::WallTimer::start();
        let empty = || vec![None; self.segments.len()];
        if retrieval_depth == 0 {
            return Ok(empty());
        }
        let crate::query::QueryDecomposition::SparseTerms(infos) = query.decompose() else {
            return Ok(empty());
        };
        let Some(&first) = infos.first() else {
            return Ok(empty());
        };
        if infos
            .iter()
            .any(|info| info.field != first.field || info.lsp_gamma != first.lsp_gamma)
        {
            return Ok(empty());
        }
        let field = first.field;
        let field_label = self.schema.get_field_name(field).unwrap_or("?");
        let (total_superblocks, total_coarse_groups, planning_depth) = self
            .segments
            .iter()
            .filter_map(|segment| segment.bmp_index(field))
            .fold(
                (0usize, 0usize, retrieval_depth),
                |(total, coarse, depth), bmp| {
                    (
                        total.saturating_add(bmp.num_superblocks as usize),
                        coarse.saturating_add(bmp.num_coarse_groups as usize),
                        depth.max(crate::query::bmp_executor_limit(
                            retrieval_depth,
                            first.over_fetch_factor,
                            bmp,
                        )),
                    )
                },
            );
        let Some(reference_bmp) = self
            .segments
            .iter()
            .find_map(|segment| segment.bmp_index(field))
        else {
            return Ok(empty());
        };
        if !infos.iter().any(|info| info.candidate) {
            return Ok(empty());
        }
        let prepare_start = crate::observe::WallTimer::start();
        let Some(prepared_query) =
            crate::query::bmp::prepare_bmp_query_infos(reference_bmp.dims(), &infos)?
        else {
            return Ok(empty());
        };
        let infos: std::sync::Arc<[crate::query::SparseTermQueryInfo]> = infos.into();
        let prepared_query = std::sync::Arc::new(prepared_query);
        let prepare_secs = prepare_start.secs();

        let local_plans = || {
            let plan = std::sync::Arc::new(crate::query::bmp::LspSegmentPlan {
                infos: std::sync::Arc::clone(&infos),
                prepared_query: std::sync::Arc::clone(&prepared_query),
                selection: None,
            });
            self.segments
                .iter()
                .map(|segment| {
                    segment
                        .bmp_index(field)
                        .map(|_| std::sync::Arc::clone(&plan))
                })
                .collect()
        };
        let gamma = first
            .lsp_gamma
            .unwrap_or_else(|| crate::query::bmp::recommended_lsp_gamma(planning_depth));
        if gamma == 0 || gamma >= total_superblocks {
            // A cap covering the whole index is exhaustive. Let each segment
            // compute and traverse its local order once instead of building a
            // query-global heap and retaining an all-superblock selection.
            crate::observe::bmp_lsp(
                self.schema.index_label(),
                field_label,
                total_start.secs(),
                prepare_secs,
                0.0,
                0.0,
                total_superblocks,
                gamma,
                total_coarse_groups,
                0,
                0,
            );
            return Ok(local_plans());
        }
        let hierarchy_scan_start = crate::observe::WallTimer::start();
        let prepare = |segment: &std::sync::Arc<crate::segment::SegmentReader>| {
            segment
                .bmp_index(field)
                .map(|bmp| crate::query::bmp::prepare_lsp_coarse_ubs(bmp, &prepared_query))
                .transpose()
        };

        #[cfg(feature = "sync")]
        let coarse_bounds: Vec<Option<Vec<f32>>> = if parallel {
            use rayon::prelude::*;
            self.search_pool.install(|| {
                self.segments
                    .par_iter()
                    .map(prepare)
                    .collect::<Result<Vec<_>>>()
            })?
        } else {
            self.segments
                .iter()
                .map(prepare)
                .collect::<Result<Vec<_>>>()?
        };
        #[cfg(not(feature = "sync"))]
        let coarse_bounds: Vec<Option<Vec<f32>>> = {
            let _ = parallel;
            self.segments
                .iter()
                .map(prepare)
                .collect::<Result<Vec<_>>>()?
        };
        let hierarchy_scan_secs = hierarchy_scan_start.secs();

        let select_start = crate::observe::WallTimer::start();
        let selection =
            select_global_lsp_hierarchical(&coarse_bounds, gamma, |segment, group, out| {
                let bmp = self.segments[segment].bmp_index(field).ok_or_else(|| {
                    crate::Error::Internal(
                        "BMP coarse plan references a segment without the sparse field".into(),
                    )
                })?;
                crate::query::bmp::expand_lsp_coarse_group(bmp, &prepared_query, group, out)
            })?;
        let mut plans = Vec::with_capacity(self.segments.len());
        for (segment, selected) in selection.selected.into_iter().enumerate() {
            if self.segments[segment].bmp_index(field).is_none() {
                plans.push(None);
                continue;
            }
            let (selected_superblocks, selected_bounds): (Vec<_>, Vec<_>) =
                selected.into_iter().unzip();
            plans.push(Some(std::sync::Arc::new(
                crate::query::bmp::LspSegmentPlan {
                    infos: std::sync::Arc::clone(&infos),
                    prepared_query: std::sync::Arc::clone(&prepared_query),
                    selection: Some(crate::query::bmp::LspSelection {
                        sb_ubs: selected_bounds,
                        sb_order: selected_superblocks,
                    }),
                },
            )));
        }
        let select_secs = select_start.secs();
        crate::observe::bmp_lsp(
            self.schema.index_label(),
            field_label,
            total_start.secs(),
            prepare_secs,
            hierarchy_scan_secs,
            select_secs,
            total_superblocks,
            gamma,
            total_coarse_groups,
            selection.expanded_groups,
            selection.evaluated_superblocks,
        );
        log::debug!(
            "BMP hierarchical LSP: field={}, coarse_groups={}/{}, E_superblocks={}/{}, gamma={}",
            field_label,
            selection.expanded_groups,
            coarse_bounds
                .iter()
                .filter_map(Option::as_ref)
                .map(Vec::len)
                .sum::<usize>(),
            selection.evaluated_superblocks,
            total_superblocks,
            gamma,
        );
        Ok(plans)
    }

    fn ordered_lsp_segments(
        &self,
        plans: &[Option<std::sync::Arc<crate::query::bmp::LspSegmentPlan>>],
    ) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.segments.len()).collect();
        order.sort_unstable_by(|&left, &right| {
            let left_priority = plans[left]
                .as_ref()
                .map_or(f32::NEG_INFINITY, |plan| plan.priority());
            let right_priority = plans[right]
                .as_ref()
                .map_or(f32::NEG_INFINITY, |plan| plan.priority());
            right_priority
                .total_cmp(&left_priority)
                .then_with(|| {
                    self.segments[right]
                        .num_docs()
                        .cmp(&self.segments[left].num_docs())
                })
                .then_with(|| {
                    self.segments[left]
                        .meta()
                        .id
                        .cmp(&self.segments[right].meta().id)
                })
        });
        order
    }

    #[inline]
    fn bmp_wave_width(&self) -> usize {
        #[cfg(feature = "native")]
        {
            self.bmp_io_concurrency
        }
        #[cfg(not(feature = "native"))]
        {
            4
        }
    }

    /// Internal search implementation
    async fn search_internal(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
        collect_positions: bool,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        let fetch_limit = checked_search_window(limit, offset)?;

        // Use rayon + block_in_place for CPU-bound scoring (sync feature required).
        // Offloads the scoring loop from tokio workers so search doesn't starve
        // other async tasks. Works for any segment count (rayon degrades gracefully
        // to inline execution for a single segment).
        // Only works on multi-threaded tokio runtime (block_in_place panics on current_thread).
        #[cfg(feature = "sync")]
        if !self.segments.is_empty()
            && tokio::runtime::Handle::current().runtime_flavor()
                == tokio::runtime::RuntimeFlavor::MultiThread
        {
            return self.search_internal_parallel(query, fetch_limit, offset, collect_positions);
        }

        // No segments, no sync feature, or current_thread runtime: use an
        // explicitly bounded async stream. Starting every segment at once can
        // retain `segments × top_k` results while the slowest I/O completes.
        const MAX_ASYNC_SEGMENT_SEARCHES: usize = 8;
        use futures::StreamExt;
        use futures::TryStreamExt;
        // Cross-segment top-k floor (see search_internal_sync). Concurrent
        // segments share it via an atomic; ordering is best-effort.
        let shared = crate::query::SharedThreshold::new();
        let lsp_plans = self.prepare_global_lsp(query, fetch_limit, false)?;
        let mut total_seen: u32 = 0;
        let mut merged = Vec::new();
        let mut merge_scratch = Vec::new();
        let bmp_planned = lsp_plans.iter().any(Option::is_some);
        let order = if bmp_planned {
            self.ordered_lsp_segments(&lsp_plans)
        } else {
            (0..self.segments.len()).collect()
        };
        #[cfg(feature = "native")]
        let bmp_io_gate = Arc::clone(&self.bmp_io_gate);
        let run_segment = |segment_index: usize| {
            let segment = Arc::clone(&self.segments[segment_index]);
            let lsp_plan = lsp_plans[segment_index].clone();
            let shared = shared.clone();
            #[cfg(feature = "native")]
            let bmp_io_gate = Arc::clone(&bmp_io_gate);
            async move {
                if lsp_plan.as_ref().is_some_and(|plan| !plan.has_work()) {
                    return Ok((Vec::new(), 0u32));
                }
                #[cfg(feature = "native")]
                let _io_permit = if lsp_plan.is_some() {
                    Some(bmp_io_gate.acquire_async().await)
                } else {
                    None
                };
                let sid = segment.meta().id;
                let (mut results, segment_seen) = crate::query::search_segment_shared_planned(
                    segment.as_ref(),
                    query,
                    fetch_limit,
                    collect_positions,
                    shared.clone(),
                    lsp_plan,
                )
                .await?;
                if fetch_limit > 0 && results.len() >= fetch_limit {
                    shared.raise(results[fetch_limit - 1].score);
                }
                for result in &mut results {
                    result.segment_id = sid;
                }
                Ok::<_, crate::error::Error>((results, segment_seen))
            }
        };

        let mut remainder = order.as_slice();
        if bmp_planned {
            // Match the synchronous policy: score the highest-bound pilot
            // first so lower-bound async work starts with a useful theta.
            if let Some((&pilot, rest)) = order.split_first() {
                let (batch, segment_seen) = run_segment(pilot).await?;
                total_seen = total_seen.saturating_add(segment_seen);
                merge_ranked_reuse(&mut merged, batch, fetch_limit, &mut merge_scratch);
                remainder = rest;
            }
        }
        let concurrency = if bmp_planned {
            self.bmp_wave_width()
        } else {
            MAX_ASYNC_SEGMENT_SEARCHES
        };
        let searches = futures::stream::iter(remainder.iter().copied().map(run_segment))
            .buffer_unordered(concurrency);
        futures::pin_mut!(searches);
        while let Some((batch, segment_seen)) = searches.try_next().await? {
            total_seen = total_seen.saturating_add(segment_seen);
            merge_ranked_reuse(&mut merged, batch, fetch_limit, &mut merge_scratch);
        }

        let results = apply_result_offset(merged, fetch_limit, offset);
        Ok((results, total_seen))
    }

    /// Multi-segment parallel search using rayon (CPU-bound scoring on thread pool).
    ///
    /// `block_in_place` tells tokio this worker is occupied so it can steal tasks.
    /// `rayon::par_iter` distributes segment scoring across the rayon thread pool.
    #[cfg(feature = "sync")]
    fn search_internal_parallel(
        &self,
        query: &dyn crate::query::Query,
        fetch_limit: usize,
        offset: usize,
        collect_positions: bool,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        tokio::task::block_in_place(|| {
            self.search_internal_sync(query, fetch_limit, offset, collect_positions)
        })
    }

    /// Sync body of the parallel search: rayon par_iter over segments.
    /// Callers must already be off the async reactor (block_in_place or a
    /// rayon/blocking thread) — safe to nest inside another par_iter
    /// (rayon work-stealing composes).
    #[cfg(feature = "sync")]
    fn search_internal_sync(
        &self,
        query: &dyn crate::query::Query,
        fetch_limit: usize,
        offset: usize,
        collect_positions: bool,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        let (merged, total_seen) =
            self.search_segments_sync(query, fetch_limit, collect_positions)?;

        let results = apply_result_offset(merged, fetch_limit, offset);
        Ok((results, total_seen))
    }

    /// Score all segments with one shared threshold.
    ///
    /// Ordinary queries retain full CPU parallelism. BMP uses a highest-bound
    /// pilot followed by bounded waves, and every active BMP segment also
    /// holds a process-wide random-I/O permit. This lets theta mature before
    /// lower-bound segments touch pageable D/payload pages and prevents
    /// concurrent queries from multiplying the wave width.
    #[cfg(feature = "sync")]
    fn search_segments_sync(
        &self,
        query: &dyn crate::query::Query,
        fetch_limit: usize,
        collect_positions: bool,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        use rayon::prelude::*;

        let lsp_plans = self.prepare_global_lsp(query, fetch_limit, true)?;
        let shared = crate::query::SharedThreshold::new();
        let run_segment = |segment_index: &usize| {
            let segment = &self.segments[*segment_index];
            let lsp_plan = lsp_plans[*segment_index].clone();
            if lsp_plan.as_ref().is_some_and(|plan| !plan.has_work()) {
                return Ok((Vec::new(), 0u32));
            }
            let _io_permit = lsp_plan.as_ref().map(|_| self.bmp_io_gate.acquire());
            let sid = segment.meta().id;
            let (mut results, segment_seen) = crate::query::search_segment_shared_sync_planned(
                segment.as_ref(),
                query,
                fetch_limit,
                collect_positions,
                shared.clone(),
                lsp_plan,
            )?;
            if fetch_limit > 0 && results.len() >= fetch_limit {
                shared.raise(results[fetch_limit - 1].score);
            }
            for result in &mut results {
                result.segment_id = sid;
            }
            Ok::<_, crate::Error>((results, segment_seen))
        };

        if !lsp_plans.iter().any(Option::is_some) {
            return self.search_pool.install(|| {
                (0..self.segments.len())
                    .into_par_iter()
                    .map(|segment| run_segment(&segment))
                    .try_reduce(
                        || (Vec::new(), 0u32),
                        |(left, left_seen), (right, right_seen)| {
                            Ok((
                                merge_two_ranked(left, right, fetch_limit),
                                left_seen.saturating_add(right_seen),
                            ))
                        },
                    )
            });
        }

        let order = self.ordered_lsp_segments(&lsp_plans);

        let mut merged = Vec::new();
        let mut merge_scratch = Vec::new();
        let mut total_seen = 0u32;
        if let Some((&pilot, rest)) = order.split_first() {
            let (pilot_results, pilot_seen) = self.search_pool.install(|| run_segment(&pilot))?;
            merge_ranked_reuse(&mut merged, pilot_results, fetch_limit, &mut merge_scratch);
            total_seen = total_seen.saturating_add(pilot_seen);

            for wave in rest.chunks(self.bmp_wave_width()) {
                let batches = self
                    .search_pool
                    .install(|| wave.par_iter().map(run_segment).collect::<Result<Vec<_>>>())?;
                for (results, seen) in batches {
                    merge_ranked_reuse(&mut merged, results, fetch_limit, &mut merge_scratch);
                    total_seen = total_seen.saturating_add(seen);
                }
            }
        }
        Ok((merged, total_seen))
    }

    /// Synchronous search across all segments using rayon for parallelism.
    ///
    /// This is the async-free boundary — no tokio involvement from here down.
    #[cfg(feature = "sync")]
    pub fn search_with_offset_and_count_sync(
        &self,
        query: &dyn crate::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        let fetch_limit = checked_search_window(limit, offset)?;
        let (merged, total_seen) = self.search_segments_sync(query, fetch_limit, false)?;

        let results = apply_result_offset(merged, fetch_limit, offset);
        Ok((results, total_seen))
    }

    /// Hybrid search: run several queries independently and fuse their
    /// ranked lists (union) into a single top-`limit` result.
    ///
    /// Unlike [`Self::search_and_rerank`] — which can only re-score
    /// documents the first-stage query already found — fusion keeps
    /// documents found by *any* of the queries. Typical use is sparse
    /// (BM25/SPLADE) + dense vector hybrid retrieval with
    /// `FusionMethod::Rrf { k: 60.0 }`.
    ///
    /// Fusion happens at **chunk granularity**: per-ordinal scores are
    /// collected from each sub-query, fused per `(doc, ordinal)` key, then
    /// combined into a doc score with `combiner`
    /// (`MultiValueCombiner::Max` recommended — same-chunk corroboration
    /// across verticals compounds, scattered noise does not). Fused results
    /// carry per-chunk `positions`.
    ///
    /// Each query is paired with a weight scaling its contribution.
    /// `fetch_limit` is the per-query candidate depth. Request-facing adapters
    /// default to at most [`crate::query::MAX_CANDIDATE_OVERSUBSCRIPTION`] times
    /// the result window and never multiply an existing rerank pool again.
    pub async fn search_fused(
        &self,
        queries: &[(&dyn crate::query::Query, f32)],
        fetch_limit: usize,
        limit: usize,
        method: crate::query::FusionMethod,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<crate::query::SearchResult>> {
        let (results, _) = self
            .search_fused_with_count(queries, fetch_limit, limit, method, combiner)
            .await?;
        Ok(results)
    }

    /// Fusion variant that also returns the aggregate number of documents
    /// scored by all sub-queries. This lets request-facing callers use the
    /// parallel fusion path without rerunning sub-queries for observability.
    pub async fn search_fused_with_count(
        &self,
        queries: &[(&dyn crate::query::Query, f32)],
        fetch_limit: usize,
        limit: usize,
        method: crate::query::FusionMethod,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        if queries.is_empty() {
            return Err(crate::Error::Query(
                "fusion requires at least one sub-query".to_string(),
            ));
        }
        if queries.len() > crate::query::MAX_FUSION_SUB_QUERIES {
            return Err(crate::Error::Query(format!(
                "fusion supports at most {} sub-queries, got {}",
                crate::query::MAX_FUSION_SUB_QUERIES,
                queries.len()
            )));
        }
        if fetch_limit == 0 {
            return Err(crate::Error::Query(
                "fusion fetch_limit must be greater than zero".to_string(),
            ));
        }
        let candidate_slots = fetch_limit
            .checked_mul(queries.len())
            .ok_or_else(|| crate::Error::Query("fusion candidate budget overflow".to_string()))?;
        if candidate_slots > crate::query::MAX_FUSION_CANDIDATE_SLOTS {
            return Err(crate::Error::Query(format!(
                "fusion candidate budget must not exceed {}, got {candidate_slots}",
                crate::query::MAX_FUSION_CANDIDATE_SLOTS
            )));
        }
        for (index, &(_, weight)) in queries.iter().enumerate() {
            if !weight.is_finite() || weight < 0.0 {
                return Err(crate::Error::Query(format!(
                    "fusion query weight at index {index} must be finite and non-negative, \
                     got {weight}"
                )));
            }
        }
        if let crate::query::FusionMethod::Rrf { k } = method
            && (!k.is_finite() || k < 0.0)
        {
            return Err(crate::Error::Query(format!(
                "fusion RRF k must be finite and non-negative, got {k}"
            )));
        }
        combiner.validate().map_err(crate::Error::Query)?;

        // Each sub-query already fans out across every segment. Keep fusion
        // sequential at the outer level so queries do not contend for the
        // same rayon pool, mmap pages, and memory bandwidth, and so each
        // query's shared threshold converges as early as possible.
        #[cfg(feature = "sync")]
        if !self.segments.is_empty()
            && tokio::runtime::Handle::current().runtime_flavor()
                == tokio::runtime::RuntimeFlavor::MultiThread
        {
            let lists: Vec<(Vec<crate::query::SearchResult>, f32, u32)> =
                tokio::task::block_in_place(|| {
                    queries
                        .iter()
                        .map(|&(query, weight)| {
                            let (results, seen) =
                                self.search_internal_sync(query, fetch_limit, 0, true)?;
                            Ok((results, weight, seen))
                        })
                        .collect::<Result<Vec<_>>>()
                })?;
            let mut total_seen = 0u32;
            let ranked_lists = lists
                .into_iter()
                .map(|(results, weight, seen)| {
                    total_seen = total_seen.saturating_add(seen);
                    (results, weight)
                })
                .collect();
            let fused =
                crate::query::try_fuse_ranked_lists_chunked(ranked_lists, method, combiner, limit)
                    .map_err(crate::Error::Query)?;
            return Ok((fused, total_seen));
        }

        // Async/current-thread fallback uses the same outer execution shape
        // and preserves input list order for deterministic rank ties.
        let mut lists = Vec::with_capacity(queries.len());
        for &(query, weight) in queries {
            let (results, seen) = self.search_with_positions(query, fetch_limit).await?;
            lists.push((results, weight, seen));
        }
        let mut total_seen = 0u32;
        let ranked_lists = lists
            .into_iter()
            .map(|(results, weight, seen)| {
                total_seen = total_seen.saturating_add(seen);
                (results, weight)
            })
            .collect();
        let fused =
            crate::query::try_fuse_ranked_lists_chunked(ranked_lists, method, combiner, limit)
                .map_err(crate::Error::Query)?;
        Ok((fused, total_seen))
    }

    /// Two-stage search: L1 retrieval + L2 dense vector reranking
    ///
    /// Runs the query to get `l1_limit` candidates, then reranks by exact
    /// dense vector distance and returns the top `final_limit` results.
    pub async fn search_and_rerank(
        &self,
        query: &dyn crate::query::Query,
        l1_limit: usize,
        final_limit: usize,
        config: &crate::query::RerankerConfig,
    ) -> Result<(Vec<crate::query::SearchResult>, u32)> {
        let (candidates, total_seen) = self.search_with_count(query, l1_limit).await?;
        let reranked = crate::query::rerank(self, &candidates, config, final_limit).await?;
        Ok((reranked, total_seen))
    }

    /// Parse query string and search (convenience method)
    pub async fn query(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<crate::query::SearchResponse> {
        self.query_offset(query_str, limit, 0).await
    }

    /// Parse query string and search with offset (convenience method)
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

        let (results, _total_seen) = self
            .search_internal(query.as_ref(), limit, offset, false)
            .await?;

        let total_hits = results.len() as u32;
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

    /// Get query parser for this searcher
    pub fn query_parser(&self) -> crate::dsl::QueryLanguageParser {
        let query_routers = self.schema.query_routers();
        if !query_routers.is_empty()
            && let Ok(router) = crate::dsl::QueryFieldRouter::from_rules(query_routers)
        {
            return crate::dsl::QueryLanguageParser::with_router(
                Arc::clone(&self.schema),
                self.default_fields.clone(),
                Arc::clone(&self.tokenizers),
                router,
            );
        }

        crate::dsl::QueryLanguageParser::new(
            Arc::clone(&self.schema),
            self.default_fields.clone(),
            Arc::clone(&self.tokenizers),
        )
    }

    /// Get a document by address (segment_id + local doc_id)
    pub async fn get_document(
        &self,
        address: &crate::query::DocAddress,
    ) -> Result<Option<crate::dsl::Document>> {
        self.get_document_with_fields(address, None).await
    }

    /// Get a document by address, hydrating only the specified field IDs.
    ///
    /// If `fields` is `None`, all fields are hydrated (including dense vectors).
    /// If `fields` is `Some(set)`, only dense vector fields in the set are read
    /// from flat storage — skipping expensive mmap reads for unrequested vectors.
    pub async fn get_document_with_fields(
        &self,
        address: &crate::query::DocAddress,
        fields: Option<&rustc_hash::FxHashSet<u32>>,
    ) -> Result<Option<crate::dsl::Document>> {
        let segment_id = address.segment_id_u128().ok_or_else(|| {
            crate::error::Error::Query(format!("Invalid segment ID: {}", address.segment_id()))
        })?;

        if let Some(&idx) = self.segment_map.get(&segment_id) {
            return self.segments[idx]
                .doc_with_fields(address.doc_id, fields)
                .await;
        }

        Ok(None)
    }
}

struct HierarchicalLspSelection {
    selected: Vec<Vec<(u32, f32)>>,
    expanded_groups: usize,
    evaluated_superblocks: usize,
}

/// Select the exact global top-gamma E cells through safe H upper bounds.
///
/// Best-first expansion stops only when the next H bound is strictly below
/// the current gamma-th E bound. Equality must still expand because the
/// deterministic `(score, segment, superblock)` tie-break can change
/// membership. Memory is O(number of H cells + gamma), never O(all E cells).
fn select_global_lsp_hierarchical(
    coarse_bounds: &[Option<Vec<f32>>],
    gamma: usize,
    mut expand: impl FnMut(usize, u32, &mut Vec<(u32, f32)>) -> Result<()>,
) -> Result<HierarchicalLspSelection> {
    if gamma == 0 {
        return Err(crate::Error::Internal(
            "hierarchical LSP selection requires a positive gamma".into(),
        ));
    }
    let mut frontier = std::collections::BinaryHeap::<(u32, usize, u32)>::new();
    for (segment, bounds) in coarse_bounds.iter().enumerate() {
        let Some(bounds) = bounds else {
            continue;
        };
        for (coarse_group, &bound) in bounds.iter().enumerate() {
            debug_assert!(bound.is_finite() && bound >= 0.0);
            if bound > 0.0 {
                frontier.push((bound.to_bits(), segment, coarse_group as u32));
            }
        }
    }
    let mut top =
        std::collections::BinaryHeap::<std::cmp::Reverse<(u32, usize, u32)>>::with_capacity(
            gamma.min(65_536),
        );
    let mut expanded =
        Vec::with_capacity(crate::segment::reader::bmp::BMP_COARSE_SUPERBLOCKS as usize);
    let mut expanded_groups = 0usize;
    let mut evaluated_superblocks = 0usize;
    while let Some((coarse_bound, segment, coarse_group)) = frontier.pop() {
        if top.len() == gamma && top.peek().is_some_and(|minimum| coarse_bound < minimum.0.0) {
            break;
        }
        expand(segment, coarse_group, &mut expanded)?;
        expanded_groups += 1;
        evaluated_superblocks = evaluated_superblocks.saturating_add(expanded.len());
        for &(superblock, bound) in &expanded {
            debug_assert!(bound.is_finite() && bound >= 0.0);
            if bound <= 0.0 {
                continue;
            }
            let candidate = (bound.to_bits(), segment, superblock);
            if top.len() < gamma {
                top.push(std::cmp::Reverse(candidate));
            } else if top.peek().is_some_and(|minimum| candidate > minimum.0) {
                top.pop();
                top.push(std::cmp::Reverse(candidate));
            }
        }
    }

    let mut selected = vec![Vec::<(u32, f32)>::new(); coarse_bounds.len()];
    for std::cmp::Reverse((bound, segment, superblock)) in top {
        selected[segment].push((superblock, f32::from_bits(bound)));
    }
    for segment in &mut selected {
        segment.sort_unstable_by(|&(left_sb, left_bound), &(right_sb, right_bound)| {
            right_bound
                .total_cmp(&left_bound)
                .then_with(|| left_sb.cmp(&right_sb))
        });
    }
    Ok(HierarchicalLspSelection {
        selected,
        expanded_groups,
        evaluated_superblocks,
    })
}

/// Select one query-level top-γ set without allocating one tuple per
/// superblock. Prepared BMP bounds are finite and non-negative, so their f32
/// bit patterns have the same order as their numeric values.
#[cfg(test)]
fn select_global_lsp_superblocks(bounds: &[Option<Vec<f32>>], gamma: usize) -> Vec<Vec<u32>> {
    let mut top =
        std::collections::BinaryHeap::<std::cmp::Reverse<(u32, usize, u32)>>::with_capacity(
            gamma.min(65_536),
        );
    for (segment, segment_bounds) in bounds.iter().enumerate() {
        let Some(segment_bounds) = segment_bounds else {
            continue;
        };
        for (superblock, &bound) in segment_bounds.iter().enumerate() {
            if bound <= 0.0 {
                continue;
            }
            let candidate = (bound.to_bits(), segment, superblock as u32);
            if top.len() < gamma {
                top.push(std::cmp::Reverse(candidate));
            } else if top.peek().is_some_and(|minimum| candidate > minimum.0) {
                top.pop();
                top.push(std::cmp::Reverse(candidate));
            }
        }
    }

    let mut selected = vec![Vec::<u32>::new(); bounds.len()];
    for std::cmp::Reverse((_, segment, superblock)) in top {
        selected[segment].push(superblock);
    }
    selected
}

/// Merge one segment batch into the running top-k while alternating two
/// retained buffers. Search results are moved, not cloned, and segment fan-out
/// no longer allocates a fresh `limit`-sized vector for every merge.
fn merge_ranked_reuse(
    merged: &mut Vec<crate::query::SearchResult>,
    batch: Vec<crate::query::SearchResult>,
    limit: usize,
    scratch: &mut Vec<crate::query::SearchResult>,
) {
    scratch.clear();
    let output_len = limit.min(merged.len().saturating_add(batch.len()));
    if scratch.capacity() < output_len {
        scratch.reserve_exact(output_len);
    }
    {
        let mut left = merged.drain(..).peekable();
        let mut right = batch.into_iter().peekable();
        while scratch.len() < output_len {
            let take_left = match (left.peek(), right.peek()) {
                (Some(left), Some(right)) => {
                    !crate::query::compare_search_results_desc(left, right).is_gt()
                }
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if take_left {
                scratch.push(left.next().expect("peeked left result"));
            } else {
                scratch.push(right.next().expect("peeked right result"));
            }
        }
    }
    std::mem::swap(merged, scratch);
}

/// Merge two canonically sorted batches while moving (not cloning) hits.
/// Native parallel reductions use this eagerly, so retained cross-segment
/// results stay O(k) instead of O(number_of_segments × k).
#[cfg(any(feature = "native", test))]
fn merge_two_ranked(
    left: Vec<crate::query::SearchResult>,
    right: Vec<crate::query::SearchResult>,
    limit: usize,
) -> Vec<crate::query::SearchResult> {
    let mut left = left.into_iter().peekable();
    let mut right = right.into_iter().peekable();
    let mut merged = Vec::with_capacity(limit.min(left.len().saturating_add(right.len())));

    while merged.len() < limit {
        let take_left = match (left.peek(), right.peek()) {
            (Some(left), Some(right)) => {
                !crate::query::compare_search_results_desc(left, right).is_gt()
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_left {
            merged.push(left.next().expect("peeked left result"));
        } else {
            merged.push(right.next().expect("peeked right result"));
        }
    }
    merged
}

fn apply_result_offset(
    mut results: Vec<crate::query::SearchResult>,
    fetch_limit: usize,
    offset: usize,
) -> Vec<crate::query::SearchResult> {
    if offset == 0 {
        results.truncate(fetch_limit);
        return results;
    }
    // Pagination usually returns a small window from a much larger fetch.
    // Allocate that small result rather than retaining the fetch-sized backing
    // allocation through response serialization.
    results
        .into_iter()
        .skip(offset)
        .take(fetch_limit.saturating_sub(offset))
        .collect()
}

fn checked_search_window(limit: usize, offset: usize) -> Result<usize> {
    offset
        .checked_add(limit)
        .ok_or_else(|| crate::Error::Query("search offset + limit overflow".into()))
}

/// Get current process RSS in bytes (best-effort, returns zero on failure).
fn process_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // Read from /proc/self/status — VmRSS line
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    let kib: u64 = rest
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    return kib.saturating_mul(1024);
                }
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        // Use mach_task_self / task_info via raw syscall
        use std::mem;
        #[repr(C)]
        struct TaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(task: u32, flavor: u32, info: *mut TaskBasicInfo, count: *mut u32) -> i32;
        }
        const MACH_TASK_BASIC_INFO: u32 = 20;
        let mut info: TaskBasicInfo = unsafe { mem::zeroed() };
        let mut count = (mem::size_of::<TaskBasicInfo>() / mem::size_of::<u32>()) as u32;
        let ret = unsafe {
            task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info,
                &mut count,
            )
        };
        if ret == 0 { info.resident_size } else { 0 }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

#[cfg(test)]
mod load_segments_tests {
    use super::*;

    #[tokio::test]
    async fn searcher_open_fails_loud_on_corrupt_metadata_segment_id() {
        let directory = Arc::new(crate::directories::RamDirectory::new());
        let schema = Arc::new(crate::dsl::SchemaBuilder::default().build());

        let result =
            Searcher::open(directory, schema, &["not-a-hex-segment-id".to_string()], 8).await;

        match result {
            Ok(searcher) => panic!(
                "corrupt segment ID must fail loud instead of silently serving {} segments",
                searcher.segment_readers().len()
            ),
            Err(crate::error::Error::Corruption(message)) => {
                assert!(message.contains("not-a-hex-segment-id"), "{message}");
            }
            Err(other) => panic!("expected Corruption error for invalid segment ID, got: {other}"),
        }
    }
}

#[cfg(test)]
mod search_window_tests {
    use super::{
        apply_result_offset, checked_search_window, merge_ranked_reuse, merge_two_ranked,
        select_global_lsp_hierarchical, select_global_lsp_superblocks,
    };
    use crate::query::SearchResult;

    fn result(segment_id: u128, doc_id: u32, score: f32) -> SearchResult {
        SearchResult {
            doc_id,
            score,
            segment_id,
            positions: Vec::new(),
        }
    }

    #[test]
    fn search_window_is_checked() {
        assert_eq!(checked_search_window(7, 5).unwrap(), 12);
        assert!(checked_search_window(1, usize::MAX).is_err());
    }

    #[test]
    fn bounded_merge_preserves_canonical_order_and_ties() {
        let left = vec![result(2, 9, 10.0), result(2, 3, 7.0)];
        let right = vec![result(1, 8, 10.0), result(1, 2, 7.0)];

        let expected = merge_two_ranked(left.clone(), right.clone(), 3);
        let mut merged = left;
        let mut scratch = Vec::new();
        merge_ranked_reuse(&mut merged, right, 3, &mut scratch);
        assert_eq!(merged, expected);
        assert!(scratch.is_empty());
        let keys: Vec<_> = merged
            .iter()
            .map(|result| (result.score, result.segment_id, result.doc_id))
            .collect();
        assert_eq!(keys, vec![(10.0, 1, 8), (10.0, 2, 9), (7.0, 1, 2)]);
    }

    #[test]
    fn result_offset_returns_only_the_requested_window() {
        let results = (0..8)
            .map(|doc_id| result(1, doc_id, 8.0 - doc_id as f32))
            .collect();

        let page = apply_result_offset(results, 5, 2);
        assert_eq!(
            page.iter().map(|result| result.doc_id).collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn lsp_gamma_is_global_not_per_segment() {
        let bounds = vec![
            Some(vec![9.0, 1.0, 8.0]),
            Some(vec![7.0, 6.0, 0.0]),
            None,
            Some(vec![5.0, 4.0]),
        ];
        let mut selected = select_global_lsp_superblocks(&bounds, 4);
        for segment in &mut selected {
            segment.sort_unstable();
        }
        assert_eq!(selected.iter().map(Vec::len).sum::<usize>(), 4);
        assert_eq!(selected[0], vec![0, 2]);
        assert_eq!(selected[1], vec![0, 1]);
        assert!(selected[2].is_empty());
        assert!(selected[3].is_empty());
    }

    #[test]
    fn hierarchical_lsp_matches_full_e_scan_exactly() {
        const GROUP: usize = crate::segment::reader::bmp::BMP_COARSE_SUPERBLOCKS as usize;
        let full = vec![
            Some(
                (0..700)
                    .map(|index| 1_000.0 - index as f32)
                    .collect::<Vec<_>>(),
            ),
            Some(
                (0..530)
                    .map(|index| 995.0 - index as f32 * 1.25)
                    .collect::<Vec<_>>(),
            ),
            None,
        ];
        let coarse: Vec<Option<Vec<f32>>> = full
            .iter()
            .map(|segment| {
                segment.as_ref().map(|bounds| {
                    bounds
                        .chunks(GROUP)
                        .map(|group| group.iter().copied().fold(0.0, f32::max))
                        .collect()
                })
            })
            .collect();
        let gamma = 37;
        let hierarchy = select_global_lsp_hierarchical(&coarse, gamma, |segment, group, output| {
            output.clear();
            let Some(bounds) = &full[segment] else {
                return Ok(());
            };
            let start = group as usize * GROUP;
            output.extend(
                bounds[start..(start + GROUP).min(bounds.len())]
                    .iter()
                    .enumerate()
                    .map(|(within, &bound)| ((start + within) as u32, bound)),
            );
            Ok(())
        })
        .unwrap();
        let mut expected = select_global_lsp_superblocks(&full, gamma);
        for segment in &mut expected {
            segment.sort_unstable();
        }
        let mut actual: Vec<Vec<u32>> = hierarchy
            .selected
            .iter()
            .map(|segment| {
                let mut ids: Vec<_> = segment.iter().map(|&(id, _)| id).collect();
                ids.sort_unstable();
                ids
            })
            .collect();
        for segment in &mut actual {
            segment.sort_unstable();
        }
        assert_eq!(actual, expected);
        assert_eq!(actual.iter().map(Vec::len).sum::<usize>(), gamma);
        assert!(
            hierarchy.evaluated_superblocks
                < full.iter().filter_map(Option::as_ref).map(Vec::len).sum()
        );
    }
}

#[cfg(test)]
mod fusion_parallelism_tests {
    #[test]
    fn fusion_keeps_parallelism_at_the_segment_level() {
        let source = include_str!("searcher.rs");
        let body = source
            .split("pub async fn search_fused_with_count")
            .nth(1)
            .and_then(|tail| tail.split("/// Two-stage search").next())
            .expect("bounded fusion search implementation");

        assert!(
            !body.contains(".par_iter()"),
            "sub-query parallelism nests over segment parallelism"
        );
        assert!(
            !body.contains(".buffered("),
            "async fusion fallback must preserve the same bounded execution shape"
        );
    }
}
