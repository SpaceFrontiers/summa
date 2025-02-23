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
        Self::create(
            directory,
            schema,
            segment_ids,
            FxHashMap::default(),
            term_cache_blocks,
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
        term_cache_blocks: usize,
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            snapshot.segment_ids(),
            &trained_centroids,
            term_cache_blocks,
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
        term_cache_blocks: usize,
        existing_segments: &[Arc<SegmentReader>],
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            snapshot.segment_ids(),
            &trained_centroids,
            term_cache_blocks,
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
        })
    }

    /// Internal create method
    async fn create(
        directory: Arc<D>,
        schema: Arc<Schema>,
        segment_ids: &[String],
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
    ) -> Result<Self> {
        let (segments, default_fields, global_stats, segment_map, total_docs) = Self::load_common(
            &directory,
            &schema,
            segment_ids,
            &trained_centroids,
            term_cache_blocks,
            &[],
        )
        .await?;

        #[cfg(feature = "native")]
        let _snapshot = {
            let tracker = Arc::new(SegmentTracker::new());
            SegmentSnapshot::new(tracker, segment_ids.to_vec())
        };

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
        })
    }

    /// Common loading logic shared by create and from_snapshot
    async fn load_common(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        segment_ids: &[String],
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
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
        existing_segments: &[Arc<SegmentReader>],
    ) -> Result<Vec<Arc<SegmentReader>>> {
        // Build lookup from existing segment readers for reuse
        let existing_map: FxHashMap<u128, Arc<SegmentReader>> = existing_segments
            .iter()
            .map(|seg| (seg.meta().id, Arc::clone(seg)))
            .collect();

        // Parse segment IDs and filter invalid ones
        let valid_segments: Vec<(usize, SegmentId)> = segment_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, id_str)| SegmentId::from_hex(id_str).map(|sid| (idx, sid)))
            .collect();

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
                async move { SegmentReader::open(dir.as_ref(), sid, sch, term_cache_blocks).await }
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
        let fetch_limit = offset + limit;

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

        // No segments, no sync feature, or current_thread runtime: use async path
        let futures: Vec<_> = self
            .segments
            .iter()
            .map(|segment| {
                let sid = segment.meta().id;
                async move {
                    let (mut results, segment_seen) = if collect_positions {
                        crate::query::search_segment_with_positions_and_count(
                            segment.as_ref(),
                            query,
                            fetch_limit,
                        )
                        .await?
                    } else {
                        crate::query::search_segment_with_count(
                            segment.as_ref(),
                            query,
                            fetch_limit,
                        )
                        .await?
                    };
                    // Stamp segment_id on each result
                    for r in &mut results {
                        r.segment_id = sid;
                    }
                    Ok::<_, crate::error::Error>((results, segment_seen))
                }
            })
            .collect();

        let batches = futures::future::try_join_all(futures).await?;
        let mut total_seen: u32 = 0;

        let mut sorted_batches: Vec<Vec<crate::query::SearchResult>> =
            Vec::with_capacity(batches.len());
        for (batch, segment_seen) in batches {
            total_seen = total_seen.saturating_add(segment_seen);
            if !batch.is_empty() {
                sorted_batches.push(batch);
            }
        }

        let results = merge_segment_results(sorted_batches, fetch_limit, offset);
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
        use rayon::prelude::*;

        let batches: Vec<Result<(Vec<crate::query::SearchResult>, u32)>> =
            tokio::task::block_in_place(|| {
                self.segments
                    .par_iter()
                    .map(|segment| {
                        let sid = segment.meta().id;
                        let (mut results, segment_seen) = if collect_positions {
                            crate::query::search_segment_with_positions_and_count_sync(
                                segment.as_ref(),
                                query,
                                fetch_limit,
                            )?
                        } else {
                            crate::query::search_segment_with_count_sync(
                                segment.as_ref(),
                                query,
                                fetch_limit,
                            )?
                        };
                        for r in &mut results {
                            r.segment_id = sid;
                        }
                        Ok((results, segment_seen))
                    })
                    .collect()
            });

        let mut total_seen: u32 = 0;
        let mut sorted_batches: Vec<Vec<crate::query::SearchResult>> =
            Vec::with_capacity(batches.len());
        for result in batches {
            let (batch, segment_seen) = result?;
            total_seen = total_seen.saturating_add(segment_seen);
            if !batch.is_empty() {
                sorted_batches.push(batch);
            }
        }

        let results = merge_segment_results(sorted_batches, fetch_limit, offset);
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

        let fetch_limit = offset + limit;

        let batches: Vec<Result<(Vec<crate::query::SearchResult>, u32)>> = self
            .segments
            .par_iter()
            .map(|segment| {
                let sid = segment.meta().id;
                let (mut results, segment_seen) = crate::query::search_segment_with_count_sync(
                    segment.as_ref(),
                    query,
                    fetch_limit,
                )?;
                for r in &mut results {
                    r.segment_id = sid;
                }
                Ok((results, segment_seen))
            })
            .collect();

        let mut total_seen: u32 = 0;
        let mut sorted_batches: Vec<Vec<crate::query::SearchResult>> =
            Vec::with_capacity(batches.len());
        for result in batches {
            let (batch, segment_seen) = result?;
            total_seen = total_seen.saturating_add(segment_seen);
            if !batch.is_empty() {
                sorted_batches.push(batch);
            }
        }

        let results = merge_segment_results(sorted_batches, fetch_limit, offset);
        Ok((results, total_seen))
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

/// K-way merge of pre-sorted segment result batches.
///
/// Each batch is sorted by score descending. Uses a max-heap of
/// (score, batch_idx, position) to merge in O(N log K).
fn merge_segment_results(
    sorted_batches: Vec<Vec<crate::query::SearchResult>>,
    fetch_limit: usize,
    offset: usize,
) -> Vec<crate::query::SearchResult> {
    use std::cmp::Ordering;

    struct MergeEntry {
        score: f32,
        batch_idx: usize,
        pos: usize,
    }
    impl PartialEq for MergeEntry {
        fn eq(&self, other: &Self) -> bool {
            self.score == other.score
        }
    }
    impl Eq for MergeEntry {}
    impl PartialOrd for MergeEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for MergeEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            self.score
                .partial_cmp(&other.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = std::collections::BinaryHeap::with_capacity(sorted_batches.len());
    for (i, batch) in sorted_batches.iter().enumerate() {
        if !batch.is_empty() {
            heap.push(MergeEntry {
                score: batch[0].score,
                batch_idx: i,
                pos: 0,
            });
        }
    }

    let mut results = Vec::with_capacity(fetch_limit.min(64));
    let mut emitted = 0usize;
    while let Some(entry) = heap.pop() {
        if emitted >= fetch_limit {
            break;
        }
        let batch = &sorted_batches[entry.batch_idx];
        if emitted >= offset {
            results.push(batch[entry.pos].clone());
        }
        emitted += 1;
        let next_pos = entry.pos + 1;
        if next_pos < batch.len() {
            heap.push(MergeEntry {
                score: batch[next_pos].score,
                batch_idx: entry.batch_idx,
                pos: next_pos,
            });
        }
    }

    results
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
