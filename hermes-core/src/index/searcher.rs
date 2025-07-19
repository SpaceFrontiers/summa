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
use crate::segment::{SegmentId, SegmentReader};
#[cfg(feature = "native")]
use crate::segment::{SegmentSnapshot, SegmentTracker};
use crate::structures::CoarseCentroids;

/// Immutable resources that must stay identical across `IndexReader` reloads.
/// The search pool exists only when synchronous scoring is compiled in; native
/// async-only builds retain the cache policy without spawning unused threads.
#[cfg(feature = "native")]
#[derive(Clone)]
pub(crate) struct SearcherResources {
    pub(crate) term_cache_blocks: usize,
    pub(crate) store_cache_blocks: usize,
    #[cfg(feature = "sync")]
    pub(crate) search_pool: Arc<rayon::ThreadPool>,
}

#[cfg(feature = "native")]
impl SearcherResources {
    pub(crate) fn new(
        term_cache_blocks: usize,
        store_cache_blocks: usize,
        num_threads: usize,
    ) -> Result<Self> {
        if num_threads == 0 {
            return Err(crate::Error::Internal(
                "IndexConfig.num_threads must be greater than zero".into(),
            ));
        }

        #[cfg(feature = "sync")]
        let search_pool = super::shared_search_pool(num_threads)?;

        Ok(Self {
            term_cache_blocks,
            store_cache_blocks,
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
    /// Trained centroids per field (injected into segment readers for IVF/ScaNN search)
    trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Lazy global statistics for cross-segment IDF computation
    global_stats: Arc<LazyGlobalStats>,
    /// O(1) segment lookup by segment_id
    segment_map: FxHashMap<u128, usize>,
    /// Total document count across all segments
    total_docs: u32,
    /// Bounded process-wide-by-width pool for the complete nested search tree.
    #[cfg(feature = "sync")]
    search_pool: Arc<rayon::ThreadPool>,
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
        Self::open_with_cache_blocks(
            directory,
            schema,
            segment_ids,
            term_cache_blocks,
            term_cache_blocks,
        )
        .await
    }

    /// Create a read-only searcher with independent cache capacities.
    pub async fn open_with_cache_blocks(
        directory: Arc<D>,
        schema: Arc<Schema>,
        segment_ids: &[String],
        term_cache_blocks: usize,
        store_cache_blocks: usize,
    ) -> Result<Self> {
        Self::create(
            directory,
            schema,
            segment_ids,
            FxHashMap::default(),
            term_cache_blocks,
            store_cache_blocks,
        )
        .await
    }

    /// Create from a snapshot (for native IndexReader use)
    #[cfg(feature = "native")]
    pub(crate) async fn from_snapshot(
        directory: Arc<D>,
        schema: Arc<Schema>,
        snapshot: SegmentSnapshot,
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        resources: SearcherResources,
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            snapshot.segment_ids(),
            &trained_centroids,
            resources.term_cache_blocks,
            resources.store_cache_blocks,
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
            trained_centroids,
            global_stats,
            segment_map,
            total_docs,
            #[cfg(feature = "sync")]
            search_pool: resources.search_pool,
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
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        resources: SearcherResources,
        existing_segments: &[Arc<SegmentReader>],
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            snapshot.segment_ids(),
            &trained_centroids,
            resources.term_cache_blocks,
            resources.store_cache_blocks,
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
            trained_centroids,
            global_stats,
            segment_map,
            total_docs,
            #[cfg(feature = "sync")]
            search_pool: resources.search_pool,
        })
    }

    /// Internal create method
    async fn create(
        directory: Arc<D>,
        schema: Arc<Schema>,
        segment_ids: &[String],
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
        store_cache_blocks: usize,
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            segment_ids,
            &trained_centroids,
            term_cache_blocks,
            store_cache_blocks,
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

        let _ = directory; // suppress unused warning on wasm
        Ok(Self {
            #[cfg(feature = "native")]
            _snapshot,
            _phantom: std::marker::PhantomData,
            segments,
            schema,
            default_fields,
            tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
            trained_centroids,
            global_stats,
            segment_map,
            total_docs,
            #[cfg(feature = "sync")]
            search_pool,
        })
    }

    /// Common loading logic shared by create and from_snapshot
    async fn load_common(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        segment_ids: &[String],
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
        store_cache_blocks: usize,
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
            trained_centroids,
            term_cache_blocks,
            store_cache_blocks,
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
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
        store_cache_blocks: usize,
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

        // Load only NEW segments in parallel
        let futures: Vec<_> = to_load
            .iter()
            .map(|(_, segment_id)| {
                let dir = Arc::clone(directory);
                let sch = Arc::clone(schema);
                let sid = *segment_id;
                async move {
                    SegmentReader::open_with_cache_blocks(
                        dir.as_ref(),
                        sid,
                        sch,
                        term_cache_blocks,
                        store_cache_blocks,
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
                    // Inject per-field centroids into reader for IVF/ScaNN search
                    if !trained_centroids.is_empty() {
                        reader.set_coarse_centroids(trained_centroids.clone());
                    }
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

        // Log searcher loading summary with per-segment memory breakdown
        let total_docs: u64 = segments.iter().map(|s| s.meta().num_docs as u64).sum();
        let mut total_mem = 0usize;
        for seg in &segments {
            let stats = seg.memory_stats();
            let seg_total = stats.total_bytes();
            total_mem += seg_total;
            log::info!(
                "[searcher] segment {:016x}: docs={}, mem={:.2} MB \
                 (term_dict={:.2} MB, store={:.2} MB, sparse={:.2} MB, dense={:.2} MB, bloom={:.2} MB)",
                stats.segment_id,
                stats.num_docs,
                seg_total as f64 / (1024.0 * 1024.0),
                stats.term_dict_cache_bytes as f64 / (1024.0 * 1024.0),
                stats.store_cache_bytes as f64 / (1024.0 * 1024.0),
                stats.sparse_index_bytes as f64 / (1024.0 * 1024.0),
                stats.dense_index_bytes as f64 / (1024.0 * 1024.0),
                stats.bloom_filter_bytes as f64 / (1024.0 * 1024.0),
            );
        }
        // Log process RSS if available (helps diagnose OOM)
        let rss_mb = process_rss_mb();
        log::info!(
            "[searcher] loaded {} segments: total_docs={}, estimated_mem={:.2} MB, process_rss={:.1} MB",
            segments.len(),
            total_docs,
            total_mem as f64 / (1024.0 * 1024.0),
            rss_mb,
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
    pub fn trained_centroids(&self) -> &FxHashMap<u32, Arc<CoarseCentroids>> {
        &self.trained_centroids
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
        let searches = futures::stream::iter(self.segments.iter().cloned().map(|segment| {
            let sid = segment.meta().id;
            let shared = shared.clone();
            async move {
                let (mut results, segment_seen) = crate::query::search_segment_shared(
                    segment.as_ref(),
                    query,
                    fetch_limit,
                    collect_positions,
                    shared.clone(),
                )
                .await?;
                if fetch_limit > 0 && results.len() >= fetch_limit {
                    shared.raise(results[fetch_limit - 1].score);
                }
                // Stamp segment_id on each result
                for r in &mut results {
                    r.segment_id = sid;
                }
                Ok::<_, crate::error::Error>((results, segment_seen))
            }
        }))
        .buffer_unordered(MAX_ASYNC_SEGMENT_SEARCHES);
        futures::pin_mut!(searches);

        let mut total_seen: u32 = 0;
        let mut merged = Vec::new();
        while let Some((batch, segment_seen)) = searches.try_next().await? {
            total_seen = total_seen.saturating_add(segment_seen);
            merged = merge_two_ranked(merged, batch, fetch_limit);
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
        use rayon::prelude::*;

        // Cross-segment top-k floor: each segment seeds its pruning from the
        // running global k-th score and raises it once it fills its own heap.
        let shared = crate::query::SharedThreshold::new();
        let (merged, total_seen) = self.search_pool.install(|| {
            self.segments
                .par_iter()
                .map(|segment| {
                    let sid = segment.meta().id;
                    let (mut results, segment_seen) = crate::query::search_segment_shared_sync(
                        segment.as_ref(),
                        query,
                        fetch_limit,
                        collect_positions,
                        shared.clone(),
                    )?;
                    if fetch_limit > 0 && results.len() >= fetch_limit {
                        shared.raise(results[fetch_limit - 1].score);
                    }
                    for r in &mut results {
                        r.segment_id = sid;
                    }
                    Ok::<_, crate::Error>((results, segment_seen))
                })
                .try_reduce(
                    || (Vec::new(), 0u32),
                    |(left, left_seen), (right, right_seen)| {
                        Ok((
                            merge_two_ranked(left, right, fetch_limit),
                            left_seen.saturating_add(right_seen),
                        ))
                    },
                )
        })?;

        let results = apply_result_offset(merged, fetch_limit, offset);
        Ok((results, total_seen))
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
        use rayon::prelude::*;

        let fetch_limit = checked_search_window(limit, offset)?;

        // Cross-segment top-k floor (see search_internal_sync).
        let shared = crate::query::SharedThreshold::new();
        let (merged, total_seen) = self.search_pool.install(|| {
            self.segments
                .par_iter()
                .map(|segment| {
                    let sid = segment.meta().id;
                    let (mut results, segment_seen) = crate::query::search_segment_shared_sync(
                        segment.as_ref(),
                        query,
                        fetch_limit,
                        false,
                        shared.clone(),
                    )?;
                    if fetch_limit > 0 && results.len() >= fetch_limit {
                        shared.raise(results[fetch_limit - 1].score);
                    }
                    for r in &mut results {
                        r.segment_id = sid;
                    }
                    Ok::<_, crate::Error>((results, segment_seen))
                })
                .try_reduce(
                    || (Vec::new(), 0u32),
                    |(left, left_seen), (right, right_seen)| {
                        Ok((
                            merge_two_ranked(left, right, fetch_limit),
                            left_seen.saturating_add(right_seen),
                        ))
                    },
                )
        })?;

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
    /// `fetch_limit` is the per-query candidate depth; a common choice is
    /// `4 * limit` (min 50) for good rank resolution.
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

        // Sub-queries are independent — fan them out on rayon under a single
        // block_in_place (each also par_iters its segments; rayon
        // work-stealing composes the two levels). Sequential fallback for
        // current_thread runtimes / non-sync builds.
        #[cfg(feature = "sync")]
        if !self.segments.is_empty()
            && tokio::runtime::Handle::current().runtime_flavor()
                == tokio::runtime::RuntimeFlavor::MultiThread
        {
            use rayon::prelude::*;
            let lists: Vec<Result<(Vec<crate::query::SearchResult>, f32, u32)>> =
                tokio::task::block_in_place(|| {
                    self.search_pool.install(|| {
                        queries
                            .par_iter()
                            .map(|&(query, weight)| {
                                let (results, seen) =
                                    self.search_internal_sync(query, fetch_limit, 0, true)?;
                                Ok((results, weight, seen))
                            })
                            .collect()
                    })
                });
            let lists = lists.into_iter().collect::<Result<Vec<_>>>()?;
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

        // Async/current-thread fallback retains bounded I/O concurrency and
        // preserves input list order for deterministic rank ties.
        const MAX_ASYNC_FUSION_SEARCHES: usize = 4;
        use futures::{StreamExt, TryStreamExt};
        let mut pending = Vec::with_capacity(queries.len());
        for &(query, weight) in queries {
            pending.push(async move {
                let (results, seen) = self.search_with_positions(query, fetch_limit).await?;
                Ok::<_, crate::Error>((results, weight, seen))
            });
        }
        let searches = futures::stream::iter(pending).buffered(MAX_ASYNC_FUSION_SEARCHES);
        let lists: Vec<_> = searches.try_collect().await?;
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

/// Merge two canonically sorted batches while moving (not cloning) hits.
/// Reductions use this eagerly, so retained cross-segment results stay O(k)
/// instead of O(number_of_segments × k).
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
    results: Vec<crate::query::SearchResult>,
    fetch_limit: usize,
    offset: usize,
) -> Vec<crate::query::SearchResult> {
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

/// Get current process RSS in MB (best-effort, returns 0.0 on failure)
fn process_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        // Read from /proc/self/status — VmRSS line
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    let kb: f64 = rest
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse()
                        .unwrap_or(0.0);
                    return kb / 1024.0;
                }
            }
        }
        0.0
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
        if ret == 0 {
            info.resident_size as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0.0
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
    use super::{apply_result_offset, checked_search_window, merge_two_ranked};
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

        let merged = merge_two_ranked(left, right, 3);
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
}
