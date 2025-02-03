//! Async segment reader with lazy loading

mod loader;
mod types;

pub use types::{SparseIndex, VectorIndex, VectorSearchResult};

/// Memory statistics for a single segment
#[derive(Debug, Clone, Default)]
pub struct SegmentMemoryStats {
    /// Segment ID
    pub segment_id: u128,
    /// Number of documents in segment
    pub num_docs: u32,
    /// Term dictionary block cache bytes
    pub term_dict_cache_bytes: usize,
    /// Document store block cache bytes
    pub store_cache_bytes: usize,
    /// Sparse vector index bytes (in-memory posting lists)
    pub sparse_index_bytes: usize,
    /// Dense vector index bytes (cluster assignments, quantized codes)
    pub dense_index_bytes: usize,
    /// Bloom filter bytes
    pub bloom_filter_bytes: usize,
}

impl SegmentMemoryStats {
    /// Total estimated memory for this segment
    pub fn total_bytes(&self) -> usize {
        self.term_dict_cache_bytes
            + self.store_cache_bytes
            + self.sparse_index_bytes
            + self.dense_index_bytes
            + self.bloom_filter_bytes
    }
}

use crate::structures::BlockSparsePostingList;

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::{AsyncFileRead, Directory, LazyFileHandle, LazyFileSlice};
use crate::dsl::{Document, Field, Schema};
use crate::structures::{
    AsyncSSTableReader, BlockPostingList, CoarseCentroids, IVFPQIndex, IVFRaBitQIndex, PQCodebook,
    RaBitQIndex, SSTableStats, TermInfo,
};
use crate::{DocId, Error, Result};

use super::store::{AsyncStoreReader, RawStoreBlock};
use super::types::{SegmentFiles, SegmentId, SegmentMeta};

/// Async segment reader with lazy loading
///
/// - Term dictionary: only index loaded, blocks loaded on-demand
/// - Postings: loaded on-demand per term via HTTP range requests
/// - Document store: only index loaded, blocks loaded on-demand via HTTP range requests
pub struct AsyncSegmentReader {
    meta: SegmentMeta,
    /// Term dictionary with lazy block loading
    term_dict: Arc<AsyncSSTableReader<TermInfo>>,
    /// Postings file handle - fetches ranges on demand
    postings_handle: LazyFileHandle,
    /// Document store with lazy block loading
    store: Arc<AsyncStoreReader>,
    schema: Arc<Schema>,
    /// Base doc_id offset for this segment
    doc_id_offset: DocId,
    /// Dense vector indexes per field (RaBitQ or IVF-RaBitQ)
    vector_indexes: FxHashMap<u32, VectorIndex>,
    /// Shared coarse centroids for IVF search (loaded once)
    coarse_centroids: Option<Arc<CoarseCentroids>>,
    /// Sparse vector indexes per field
    sparse_indexes: FxHashMap<u32, SparseIndex>,
    /// Position file handle for phrase queries (lazy loading)
    positions_handle: Option<LazyFileHandle>,
}

impl AsyncSegmentReader {
    /// Open a segment with lazy loading
    pub async fn open<D: Directory>(
        dir: &D,
        segment_id: SegmentId,
        schema: Arc<Schema>,
        doc_id_offset: DocId,
        cache_blocks: usize,
    ) -> Result<Self> {
        let files = SegmentFiles::new(segment_id.0);

        // Read metadata (small, always loaded)
        let meta_slice = dir.open_read(&files.meta).await?;
        let meta_bytes = meta_slice.read_bytes().await?;
        let meta = SegmentMeta::deserialize(meta_bytes.as_slice())?;
        debug_assert_eq!(meta.id, segment_id.0);

        // Open term dictionary with lazy loading (fetches ranges on demand)
        let term_dict_handle = dir.open_lazy(&files.term_dict).await?;
        let term_dict = AsyncSSTableReader::open(term_dict_handle, cache_blocks).await?;

        // Get postings file handle (lazy - fetches ranges on demand)
        let postings_handle = dir.open_lazy(&files.postings).await?;

        // Open store with lazy loading
        let store_handle = dir.open_lazy(&files.store).await?;
        let store = AsyncStoreReader::open(store_handle, cache_blocks).await?;

        // Load dense vector indexes from unified .vectors file
        let (vector_indexes, coarse_centroids) =
            loader::load_vectors_file(dir, &files, &schema).await?;

        // Load sparse vector indexes from .sparse file
        let sparse_indexes = loader::load_sparse_file(dir, &files, meta.num_docs, &schema).await?;

        // Open positions file handle (if exists) - offsets are now in TermInfo
        let positions_handle = loader::open_positions_file(dir, &files, &schema).await?;

        // Log segment loading stats (compact format: ~24 bytes per active dim in hashmap)
        let sparse_dims: usize = sparse_indexes.values().map(|s| s.num_dimensions()).sum();
        let sparse_mem = sparse_dims * 24; // HashMap entry overhead
        log::debug!(
            "[segment] loaded {:016x}: docs={}, sparse_dims={}, sparse_mem={:.2} KB, vectors={}",
            segment_id.0,
            meta.num_docs,
            sparse_dims,
            sparse_mem as f64 / 1024.0,
            vector_indexes.len()
        );

        Ok(Self {
            meta,
            term_dict: Arc::new(term_dict),
            postings_handle,
            store: Arc::new(store),
            schema,
            doc_id_offset,
            vector_indexes,
            coarse_centroids,
            sparse_indexes,
            positions_handle,
        })
    }

    pub fn meta(&self) -> &SegmentMeta {
        &self.meta
    }

    pub fn num_docs(&self) -> u32 {
        self.meta.num_docs
    }

    /// Get average field length for BM25F scoring
    pub fn avg_field_len(&self, field: Field) -> f32 {
        self.meta.avg_field_len(field)
    }

    pub fn doc_id_offset(&self) -> DocId {
        self.doc_id_offset
    }

    /// Set the doc_id_offset (used for parallel segment loading)
    pub fn set_doc_id_offset(&mut self, offset: DocId) {
        self.doc_id_offset = offset;
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get sparse indexes for all fields
    pub fn sparse_indexes(&self) -> &FxHashMap<u32, SparseIndex> {
        &self.sparse_indexes
    }

    /// Get vector indexes for all fields
    pub fn vector_indexes(&self) -> &FxHashMap<u32, VectorIndex> {
        &self.vector_indexes
    }

    /// Get term dictionary stats for debugging
    pub fn term_dict_stats(&self) -> SSTableStats {
        self.term_dict.stats()
    }

    /// Estimate memory usage of this segment reader
    pub fn memory_stats(&self) -> SegmentMemoryStats {
        let term_dict_stats = self.term_dict.stats();

        // Term dict cache: num_blocks * avg_block_size (estimate 4KB per cached block)
        let term_dict_cache_bytes = self.term_dict.cached_blocks() * 4096;

        // Store cache: similar estimate
        let store_cache_bytes = self.store.cached_blocks() * 4096;

        // Sparse index: each dimension has a posting list in memory
        // Estimate: ~24 bytes per active dimension (HashMap entry overhead)
        let sparse_index_bytes: usize = self
            .sparse_indexes
            .values()
            .map(|s| s.num_dimensions() * 24)
            .sum();

        // Dense index: vectors are memory-mapped, but we track index structures
        // RaBitQ/IVF indexes have cluster assignments in memory
        let dense_index_bytes: usize = self
            .vector_indexes
            .values()
            .map(|v| v.estimated_memory_bytes())
            .sum();

        SegmentMemoryStats {
            segment_id: self.meta.id,
            num_docs: self.meta.num_docs,
            term_dict_cache_bytes,
            store_cache_bytes,
            sparse_index_bytes,
            dense_index_bytes,
            bloom_filter_bytes: term_dict_stats.bloom_filter_size,
        }
    }

    /// Get posting list for a term (async - loads on demand)
    ///
    /// For small posting lists (1-3 docs), the data is inlined in the term dictionary
    /// and no additional I/O is needed. For larger lists, reads from .post file.
    pub async fn get_postings(
        &self,
        field: Field,
        term: &[u8],
    ) -> Result<Option<BlockPostingList>> {
        log::debug!(
            "SegmentReader::get_postings field={} term_len={}",
            field.0,
            term.len()
        );

        // Build key: field_id + term
        let mut key = Vec::with_capacity(4 + term.len());
        key.extend_from_slice(&field.0.to_le_bytes());
        key.extend_from_slice(term);

        // Look up in term dictionary
        let term_info = match self.term_dict.get(&key).await? {
            Some(info) => {
                log::debug!("SegmentReader::get_postings found term_info");
                info
            }
            None => {
                log::debug!("SegmentReader::get_postings term not found");
                return Ok(None);
            }
        };

        // Check if posting list is inlined
        if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
            // Build BlockPostingList from inline data (no I/O needed!)
            let mut posting_list = crate::structures::PostingList::with_capacity(doc_ids.len());
            for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs.into_iter()) {
                posting_list.push(doc_id, tf);
            }
            let block_list = BlockPostingList::from_posting_list(&posting_list)?;
            return Ok(Some(block_list));
        }

        // External posting list - read from postings file handle (lazy - HTTP range request)
        let (posting_offset, posting_len) = term_info.external_info().ok_or_else(|| {
            Error::Corruption("TermInfo has neither inline nor external data".to_string())
        })?;

        let start = posting_offset;
        let end = start + posting_len as u64;

        if end > self.postings_handle.len() {
            return Err(Error::Corruption(
                "Posting offset out of bounds".to_string(),
            ));
        }

        let posting_bytes = self.postings_handle.read_bytes_range(start..end).await?;
        let block_list = BlockPostingList::deserialize(&mut posting_bytes.as_slice())?;

        Ok(Some(block_list))
    }

    /// Get document by local doc_id (async - loads on demand)
    pub async fn doc(&self, local_doc_id: DocId) -> Result<Option<Document>> {
        self.store
            .get(local_doc_id, &self.schema)
            .await
            .map_err(Error::from)
    }

    /// Prefetch term dictionary blocks for a key range
    pub async fn prefetch_terms(
        &self,
        field: Field,
        start_term: &[u8],
        end_term: &[u8],
    ) -> Result<()> {
        let mut start_key = Vec::with_capacity(4 + start_term.len());
        start_key.extend_from_slice(&field.0.to_le_bytes());
        start_key.extend_from_slice(start_term);

        let mut end_key = Vec::with_capacity(4 + end_term.len());
        end_key.extend_from_slice(&field.0.to_le_bytes());
        end_key.extend_from_slice(end_term);

        self.term_dict.prefetch_range(&start_key, &end_key).await?;
        Ok(())
    }

    /// Check if store uses dictionary compression (incompatible with raw merging)
    pub fn store_has_dict(&self) -> bool {
        self.store.has_dict()
    }

    /// Get raw store blocks for optimized merging
    pub fn store_raw_blocks(&self) -> Vec<RawStoreBlock> {
        self.store.raw_blocks()
    }

    /// Get store data slice for raw block access
    pub fn store_data_slice(&self) -> &LazyFileSlice {
        self.store.data_slice()
    }

    /// Get all terms from this segment (for merge)
    pub async fn all_terms(&self) -> Result<Vec<(Vec<u8>, TermInfo)>> {
        self.term_dict.all_entries().await.map_err(Error::from)
    }

    /// Get all terms with parsed field and term string (for statistics aggregation)
    ///
    /// Returns (field, term_string, doc_freq) for each term in the dictionary.
    /// Skips terms that aren't valid UTF-8.
    pub async fn all_terms_with_stats(&self) -> Result<Vec<(Field, String, u32)>> {
        let entries = self.term_dict.all_entries().await?;
        let mut result = Vec::with_capacity(entries.len());

        for (key, term_info) in entries {
            // Key format: field_id (4 bytes little-endian) + term bytes
            if key.len() > 4 {
                let field_id = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
                let term_bytes = &key[4..];
                if let Ok(term_str) = std::str::from_utf8(term_bytes) {
                    result.push((Field(field_id), term_str.to_string(), term_info.doc_freq()));
                }
            }
        }

        Ok(result)
    }

    /// Get streaming iterator over term dictionary (for memory-efficient merge)
    pub fn term_dict_iter(&self) -> crate::structures::AsyncSSTableIterator<'_, TermInfo> {
        self.term_dict.iter()
    }

    /// Read raw posting bytes at offset
    pub async fn read_postings(&self, offset: u64, len: u32) -> Result<Vec<u8>> {
        let start = offset;
        let end = start + len as u64;
        let bytes = self.postings_handle.read_bytes_range(start..end).await?;
        Ok(bytes.to_vec())
    }

    /// Search dense vectors using RaBitQ
    ///
    /// Returns VectorSearchResult with ordinal tracking for multi-value fields.
    /// The doc_ids are adjusted by doc_id_offset for this segment.
    /// If mrl_dim is configured, the query vector is automatically trimmed.
    /// For multi-valued documents, scores are combined using the specified combiner.
    pub fn search_dense_vector(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        let index = self
            .vector_indexes
            .get(&field.0)
            .ok_or_else(|| Error::Schema(format!("No dense vector index for field {}", field.0)))?;

        // Get mrl_dim from config to trim query vector if needed
        let mrl_dim = self
            .schema
            .get_field_entry(field)
            .and_then(|e| e.dense_vector_config.as_ref())
            .and_then(|c| c.mrl_dim);

        // Trim query vector if mrl_dim is set
        let query_vec: Vec<f32>;
        let effective_query = if let Some(trim_dim) = mrl_dim {
            if trim_dim < query.len() {
                query_vec = query[..trim_dim].to_vec();
                query_vec.as_slice()
            } else {
                query
            }
        } else {
            query
        };

        // Results include (doc_id, ordinal, distance)
        let results: Vec<(u32, u16, f32)> = match index {
            VectorIndex::Flat(flat_data) => {
                // Brute-force search over raw vectors using SIMD-accelerated distance
                use crate::structures::simd::squared_euclidean_distance;

                let mut candidates: Vec<(u32, u16, f32)> = flat_data
                    .vectors
                    .iter()
                    .zip(flat_data.doc_ids.iter())
                    .map(|(vec, &(doc_id, ordinal))| {
                        let dist = squared_euclidean_distance(effective_query, vec);
                        (doc_id, ordinal, dist)
                    })
                    .collect();
                candidates
                    .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
                candidates.truncate(k);
                candidates
            }
            VectorIndex::RaBitQ(rabitq) => rabitq.search(effective_query, k, rerank_factor),
            VectorIndex::IVF { index, codebook } => {
                let centroids = self.coarse_centroids.as_ref().ok_or_else(|| {
                    Error::Schema("IVF index requires coarse centroids".to_string())
                })?;
                let nprobe = rerank_factor.max(32); // Use rerank_factor as nprobe hint
                index
                    .search(centroids, codebook, effective_query, k, Some(nprobe))
                    .into_iter()
                    .map(|(doc_id, dist)| (doc_id, 0u16, dist)) // IVF doesn't track ordinals yet
                    .collect()
            }
            VectorIndex::ScaNN { index, codebook } => {
                let centroids = self.coarse_centroids.as_ref().ok_or_else(|| {
                    Error::Schema("ScaNN index requires coarse centroids".to_string())
                })?;
                let nprobe = rerank_factor.max(32);
                index
                    .search(centroids, codebook, effective_query, k, Some(nprobe))
                    .into_iter()
                    .map(|(doc_id, dist)| (doc_id, 0u16, dist)) // ScaNN doesn't track ordinals yet
                    .collect()
            }
        };

        // Convert distance to score (smaller distance = higher score)
        // and adjust doc_ids by segment offset
        // Track ordinals with individual scores for each doc_id
        let mut doc_ordinals: rustc_hash::FxHashMap<DocId, Vec<(u32, f32)>> =
            rustc_hash::FxHashMap::default();
        for (doc_id, ordinal, dist) in results {
            let doc_id = doc_id as DocId + self.doc_id_offset;
            let score = 1.0 / (1.0 + dist); // Convert distance to similarity score
            let ordinals = doc_ordinals.entry(doc_id).or_default();
            ordinals.push((ordinal as u32, score));
        }

        // Combine scores and build results with ordinal tracking
        let mut final_results: Vec<VectorSearchResult> = doc_ordinals
            .into_iter()
            .map(|(doc_id, ordinals)| {
                let combined_score = combiner.combine(&ordinals);
                VectorSearchResult::new(doc_id, combined_score, ordinals)
            })
            .collect();

        // Sort by score descending and take top k
        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_results.truncate(k);

        Ok(final_results)
    }

    /// Check if this segment has a dense vector index for the given field
    pub fn has_dense_vector_index(&self, field: Field) -> bool {
        self.vector_indexes.contains_key(&field.0)
    }

    /// Get the dense vector index for a field (if available)
    pub fn get_dense_vector_index(&self, field: Field) -> Option<Arc<RaBitQIndex>> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::RaBitQ(idx)) => Some(idx.clone()),
            _ => None,
        }
    }

    /// Get the IVF vector index for a field (if available)
    pub fn get_ivf_vector_index(&self, field: Field) -> Option<Arc<IVFRaBitQIndex>> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::IVF { index, .. }) => Some(index.clone()),
            _ => None,
        }
    }

    /// Get the ScaNN vector index for a field (if available)
    pub fn get_scann_vector_index(
        &self,
        field: Field,
    ) -> Option<(Arc<IVFPQIndex>, Arc<PQCodebook>)> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::ScaNN { index, codebook }) => Some((index.clone(), codebook.clone())),
            _ => None,
        }
    }

    /// Get the vector index type for a field
    pub fn get_vector_index(&self, field: Field) -> Option<&VectorIndex> {
        self.vector_indexes.get(&field.0)
    }

    /// Search for similar sparse vectors using dedicated sparse posting lists
    ///
    /// Uses shared `WandExecutor` with `SparseTermScorer` for efficient top-k retrieval.
    /// Optimizations (via WandExecutor):
    /// 1. **MaxScore pruning**: Dimensions sorted by max contribution
    /// 2. **Block-Max WAND**: Skips blocks where max contribution < threshold
    /// 3. **Top-K heap**: Efficient score collection
    ///
    /// Returns VectorSearchResult with ordinal tracking for multi-value fields.
    pub async fn search_sparse_vector(
        &self,
        field: Field,
        vector: &[(u32, f32)],
        limit: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        use crate::query::{SparseTermScorer, WandExecutor};

        let query_tokens = vector.len();

        // Get sparse index for this field
        let sparse_index = match self.sparse_indexes.get(&field.0) {
            Some(idx) => idx,
            None => {
                log::debug!(
                    "Sparse vector search: no index for field {}, returning empty",
                    field.0
                );
                return Ok(Vec::new());
            }
        };

        let index_dimensions = sparse_index.num_dimensions();

        // Build scorers for each dimension that exists in the index
        // Load posting lists on-demand (lazy loading via mmap)
        // Keep Arc references alive for the duration of scoring
        let mut matched_tokens = Vec::new();
        let mut missing_tokens = Vec::new();
        let mut posting_lists: Vec<(u32, f32, Arc<BlockSparsePostingList>)> =
            Vec::with_capacity(vector.len());

        for &(dim_id, query_weight) in vector {
            // Check if dimension exists before loading
            if !sparse_index.has_dimension(dim_id) {
                missing_tokens.push(dim_id);
                continue;
            }

            // Load posting list on-demand (async, uses mmap)
            match sparse_index.get_posting(dim_id).await? {
                Some(pl) => {
                    matched_tokens.push(dim_id);
                    posting_lists.push((dim_id, query_weight, pl));
                }
                None => {
                    missing_tokens.push(dim_id);
                }
            }
        }

        // Create scorers from the loaded posting lists (borrows from posting_lists)
        let scorers: Vec<SparseTermScorer> = posting_lists
            .iter()
            .map(|(_, query_weight, pl)| SparseTermScorer::from_arc(pl, *query_weight))
            .collect();

        log::debug!(
            "Sparse vector search: query_tokens={}, matched={}, missing={}, index_dimensions={}",
            query_tokens,
            matched_tokens.len(),
            missing_tokens.len(),
            index_dimensions
        );

        // Log query tokens with their IDs and weights
        if log::log_enabled!(log::Level::Debug) {
            let query_details: Vec<_> = vector
                .iter()
                .take(30)
                .map(|(id, w)| format!("{}:{:.3}", id, w))
                .collect();
            log::debug!("Query tokens (id:weight): [{}]", query_details.join(", "));
        }

        if !matched_tokens.is_empty() {
            log::debug!(
                "Matched token IDs: {:?}",
                matched_tokens.iter().take(20).collect::<Vec<_>>()
            );
        }

        if !missing_tokens.is_empty() {
            log::debug!(
                "Missing token IDs (not in index): {:?}",
                missing_tokens.iter().take(20).collect::<Vec<_>>()
            );
        }

        if scorers.is_empty() {
            log::debug!("Sparse vector search: no matching tokens, returning empty");
            return Ok(Vec::new());
        }

        // Use shared WandExecutor for top-k retrieval
        // Note: For multi-valued fields, same doc_id may appear multiple times
        // with different scores that need to be combined
        let raw_results = WandExecutor::new(scorers, limit * 2).execute(); // Over-fetch for combining

        log::trace!(
            "Sparse WAND returned {} raw results for segment (doc_id_offset={})",
            raw_results.len(),
            self.doc_id_offset
        );
        if log::log_enabled!(log::Level::Trace) && !raw_results.is_empty() {
            for r in raw_results.iter().take(5) {
                log::trace!(
                    "  Raw result: doc_id={} (global={}), score={:.4}, ordinal={}",
                    r.doc_id,
                    r.doc_id + self.doc_id_offset,
                    r.score,
                    r.ordinal
                );
            }
        }

        // Track ordinals with individual scores for each doc_id
        // Now using real ordinals from the posting lists
        let mut doc_ordinals: rustc_hash::FxHashMap<u32, Vec<(u32, f32)>> =
            rustc_hash::FxHashMap::default();
        for r in raw_results {
            let ordinals = doc_ordinals.entry(r.doc_id).or_default();
            ordinals.push((r.ordinal as u32, r.score));
        }

        // Combine scores and build results with ordinal tracking
        // Add doc_id_offset to convert segment-local IDs to global IDs
        let mut results: Vec<VectorSearchResult> = doc_ordinals
            .into_iter()
            .map(|(doc_id, ordinals)| {
                let global_doc_id = doc_id + self.doc_id_offset;
                let combined_score = combiner.combine(&ordinals);
                VectorSearchResult::new(global_doc_id, combined_score, ordinals)
            })
            .collect();

        // Sort by score descending and take top limit
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    /// Get positions for a term (for phrase queries)
    ///
    /// Position offsets are now embedded in TermInfo, so we first look up
    /// the term to get its TermInfo, then use position_info() to get the offset.
    pub async fn get_positions(
        &self,
        field: Field,
        term: &[u8],
    ) -> Result<Option<crate::structures::PositionPostingList>> {
        use std::io::Cursor;

        // Get positions handle
        let handle = match &self.positions_handle {
            Some(h) => h,
            None => return Ok(None),
        };

        // Build key: field_id + term
        let mut key = Vec::with_capacity(4 + term.len());
        key.extend_from_slice(&field.0.to_le_bytes());
        key.extend_from_slice(term);

        // Look up term in dictionary to get TermInfo with position offset
        let term_info = match self.term_dict.get(&key).await? {
            Some(info) => info,
            None => return Ok(None),
        };

        // Get position offset from TermInfo
        let (offset, length) = match term_info.position_info() {
            Some((o, l)) => (o, l),
            None => return Ok(None),
        };

        // Read the position data
        let slice = handle.slice(offset..offset + length as u64);
        let data = slice.read_bytes().await?;

        // Deserialize
        let mut cursor = Cursor::new(data.as_slice());
        let pos_list = crate::structures::PositionPostingList::deserialize(&mut cursor)?;

        Ok(Some(pos_list))
    }

    /// Check if positions are available for a field
    pub fn has_positions(&self, field: Field) -> bool {
        // Check schema for position mode on this field
        if let Some(entry) = self.schema.get_field_entry(field) {
            entry.positions.is_some()
        } else {
            false
        }
    }
}

/// Alias for AsyncSegmentReader
pub type SegmentReader = AsyncSegmentReader;
