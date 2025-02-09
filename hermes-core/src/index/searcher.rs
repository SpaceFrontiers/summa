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
    _snapshot: SegmentSnapshot<D>,
    /// Phantom data for wasm builds
    #[cfg(not(feature = "native"))]
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
    /// Cumulative doc counts for binary-search in doc(global_id)
    doc_id_cumulative: Vec<u32>,
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
        snapshot: SegmentSnapshot<D>,
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
    ) -> Result<Self> {
        let (segments, default_fields, global_stats) = Self::load_common(
            &directory,
            &schema,
            snapshot.segment_ids(),
            &trained_centroids,
            term_cache_blocks,
        )
        .await;

        let (segment_map, doc_id_cumulative) = Self::build_lookup_tables(&segments);
        Ok(Self {
            _snapshot: snapshot,
            segments,
            schema,
            default_fields,
            tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
            trained_centroids,
            global_stats,
            segment_map,
            doc_id_cumulative,
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
        let (segments, default_fields, global_stats) = Self::load_common(
            &directory,
            &schema,
            segment_ids,
            &trained_centroids,
            term_cache_blocks,
        )
        .await;

        #[cfg(feature = "native")]
        {
            let tracker = Arc::new(SegmentTracker::new());
            let snapshot = SegmentSnapshot::new(tracker, segment_ids.to_vec());
            let (segment_map, doc_id_cumulative) = Self::build_lookup_tables(&segments);
            Ok(Self {
                _snapshot: snapshot,
                segments,
                schema,
                default_fields,
                tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
                trained_centroids,
                global_stats,
                segment_map,
                doc_id_cumulative,
            })
        }

        #[cfg(not(feature = "native"))]
        {
            let _ = directory; // suppress unused warning
            let (segment_map, doc_id_cumulative) = Self::build_lookup_tables(&segments);
            Ok(Self {
                _phantom: std::marker::PhantomData,
                segments,
                schema,
                default_fields,
                tokenizers: Arc::new(crate::tokenizer::TokenizerRegistry::default()),
                trained_centroids,
                global_stats,
                segment_map,
                doc_id_cumulative,
            })
        }
    }

    /// Common loading logic shared by create and from_snapshot
    async fn load_common(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        segment_ids: &[String],
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
    ) -> (
        Vec<Arc<SegmentReader>>,
        Vec<crate::Field>,
        Arc<LazyGlobalStats>,
    ) {
        let segments = Self::load_segments(
            directory,
            schema,
            segment_ids,
            trained_centroids,
            term_cache_blocks,
        )
        .await;
        let default_fields = Self::build_default_fields(schema);
        let global_stats = Arc::new(LazyGlobalStats::new(segments.clone()));
        (segments, default_fields, global_stats)
    }

    /// Load segment readers from IDs (parallel loading for performance)
    async fn load_segments(
        directory: &Arc<D>,
        schema: &Arc<Schema>,
        segment_ids: &[String],
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
    ) -> Vec<Arc<SegmentReader>> {
        // Parse segment IDs and filter invalid ones
        let valid_segments: Vec<(usize, SegmentId)> = segment_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, id_str)| SegmentId::from_hex(id_str).map(|sid| (idx, sid)))
            .collect();

        // Load all segments in parallel with offset=0
        let futures: Vec<_> =
            valid_segments
                .iter()
                .map(|(_, segment_id)| {
                    let dir = Arc::clone(directory);
                    let sch = Arc::clone(schema);
                    let sid = *segment_id;
                    async move {
                        SegmentReader::open(dir.as_ref(), sid, sch, 0, term_cache_blocks).await
                    }
                })
                .collect();

        let results = futures::future::join_all(futures).await;

        // Collect successful results with their original index for ordering
        let mut loaded: Vec<(usize, SegmentReader)> = valid_segments
            .into_iter()
            .zip(results)
            .filter_map(|((idx, _), result)| match result {
                Ok(reader) => Some((idx, reader)),
                Err(e) => {
                    log::warn!("Failed to open segment: {:?}", e);
                    None
                }
            })
            .collect();

        // Sort by original index to maintain deterministic ordering
        loaded.sort_by_key(|(idx, _)| *idx);

        // Calculate and assign doc_id_offsets sequentially
        let mut doc_id_offset = 0u32;
        let mut segments = Vec::with_capacity(loaded.len());
        for (_, mut reader) in loaded {
            reader.set_doc_id_offset(doc_id_offset);
            doc_id_offset += reader.meta().num_docs;
            // Inject per-field centroids into reader for IVF/ScaNN search
            if !trained_centroids.is_empty() {
                reader.set_coarse_centroids(trained_centroids.clone());
            }
            segments.push(Arc::new(reader));
        }

        // Log searcher loading summary
        let total_docs: u32 = segments.iter().map(|s| s.meta().num_docs).sum();
        let total_sparse_mem: usize = segments
            .iter()
            .flat_map(|s| s.sparse_indexes().values())
            .map(|idx| idx.num_dimensions() * 12)
            .sum();
        log::info!(
            "[searcher] loaded {} segments: total_docs={}, sparse_index_mem={:.2} MB",
            segments.len(),
            total_docs,
            total_sparse_mem as f64 / (1024.0 * 1024.0)
        );

        segments
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
    fn build_lookup_tables(segments: &[Arc<SegmentReader>]) -> (FxHashMap<u128, usize>, Vec<u32>) {
        let mut segment_map = FxHashMap::default();
        let mut cumulative = Vec::with_capacity(segments.len());
        let mut acc = 0u32;
        for (i, seg) in segments.iter().enumerate() {
            segment_map.insert(seg.meta().id, i);
            acc += seg.meta().num_docs;
            cumulative.push(acc);
        }
        (segment_map, cumulative)
    }

    /// Get total document count across all segments
    pub fn num_docs(&self) -> u32 {
        self.doc_id_cumulative.last().copied().unwrap_or(0)
    }

    /// Get O(1) segment_id → index map (used by reranker)
    pub fn segment_map(&self) -> &FxHashMap<u128, usize> {
        &self.segment_map
    }

    /// Get number of segments
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get a document by global doc_id (binary search on cumulative offsets)
    pub async fn doc(&self, doc_id: u32) -> Result<Option<crate::dsl::Document>> {
        let idx = self.doc_id_cumulative.partition_point(|&cum| cum <= doc_id);
        if idx >= self.segments.len() {
            return Ok(None);
        }
        let base = if idx == 0 {
            0
        } else {
            self.doc_id_cumulative[idx - 1]
        };
        self.segments[idx].doc(doc_id - base).await
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
        let mut all_results: Vec<crate::query::SearchResult> = Vec::new();
        let mut total_seen: u32 = 0;
        for (batch, segment_seen) in batches {
            total_seen += segment_seen;
            all_results.extend(batch);
        }

        // Sort by score descending
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply offset and limit
        let results = all_results.into_iter().skip(offset).take(limit).collect();

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

    /// Get a document by address (segment_id + global doc_id)
    ///
    /// The doc_id in the address is a global doc_id (with doc_id_offset applied).
    /// This method converts it back to a segment-local doc_id for the store lookup.
    pub async fn get_document(
        &self,
        address: &crate::query::DocAddress,
    ) -> Result<Option<crate::dsl::Document>> {
        let segment_id = address.segment_id_u128().ok_or_else(|| {
            crate::error::Error::Query(format!("Invalid segment ID: {}", address.segment_id))
        })?;

        if let Some(&idx) = self.segment_map.get(&segment_id) {
            let segment = &self.segments[idx];
            let local_doc_id = address.doc_id.wrapping_sub(segment.doc_id_offset());
            return segment.doc(local_doc_id).await;
        }

        Ok(None)
    }
}
