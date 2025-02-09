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

use super::vector_data::LazyFlatVectorData;
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
    /// Dense vector indexes per field (RaBitQ or IVF-RaBitQ) — for search
    vector_indexes: FxHashMap<u32, VectorIndex>,
    /// Lazy flat vectors per field — for reranking and merge (doc_ids in memory, vectors via mmap)
    flat_vectors: FxHashMap<u32, LazyFlatVectorData>,
    /// Per-field coarse centroids for IVF/ScaNN search
    coarse_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
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
        let vectors_data = loader::load_vectors_file(dir, &files, &schema).await?;
        let vector_indexes = vectors_data.indexes;
        let flat_vectors = vectors_data.flat_vectors;

        // Load sparse vector indexes from .sparse file
        let sparse_indexes = loader::load_sparse_file(dir, &files, meta.num_docs, &schema).await?;

        // Open positions file handle (if exists) - offsets are now in TermInfo
        let positions_handle = loader::open_positions_file(dir, &files, &schema).await?;

        // Log segment loading stats (compact format: ~24 bytes per active dim in hashmap)
        let sparse_dims: usize = sparse_indexes.values().map(|s| s.num_dimensions()).sum();
        let sparse_mem = sparse_dims * 24; // HashMap entry overhead
        log::debug!(
            "[segment] loaded {:016x}: docs={}, sparse_dims={}, sparse_mem={:.2} KB, dense_flat={}, dense_ann={}",
            segment_id.0,
            meta.num_docs,
            sparse_dims,
            sparse_mem as f64 / 1024.0,
            flat_vectors.len(),
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
            flat_vectors,
            coarse_centroids: FxHashMap::default(),
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

    /// Get lazy flat vectors for all fields (for reranking and merge)
    pub fn flat_vectors(&self) -> &FxHashMap<u32, LazyFlatVectorData> {
        &self.flat_vectors
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

    /// Get document by local doc_id (async - loads on demand).
    ///
    /// Dense vector fields are hydrated from LazyFlatVectorData (not stored in .store).
    /// Uses binary search on sorted doc_ids for O(log N) lookup.
    pub async fn doc(&self, local_doc_id: DocId) -> Result<Option<Document>> {
        let mut doc = match self.store.get(local_doc_id, &self.schema).await {
            Ok(Some(d)) => d,
            Ok(None) => return Ok(None),
            Err(e) => return Err(Error::from(e)),
        };

        // Hydrate dense vector fields from flat vector data
        for (&field_id, lazy_flat) in &self.flat_vectors {
            let (start, entries) = lazy_flat.flat_indexes_for_doc(local_doc_id);
            for (j, &(_doc_id, _ordinal)) in entries.iter().enumerate() {
                let flat_idx = start + j;
                match lazy_flat.get_vector(flat_idx).await {
                    Ok(vec) => {
                        doc.add_dense_vector(Field(field_id), vec);
                    }
                    Err(e) => {
                        log::warn!("Failed to hydrate vector field {}: {}", field_id, e);
                    }
                }
            }
        }

        Ok(Some(doc))
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

    /// Get store reference for merge operations
    pub fn store(&self) -> &super::store::AsyncStoreReader {
        &self.store
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

    /// Prefetch all term dictionary blocks in a single bulk I/O call.
    ///
    /// Call before merge iteration to eliminate per-block cache misses.
    pub async fn prefetch_term_dict(&self) -> crate::Result<()> {
        self.term_dict
            .prefetch_all_data_bulk()
            .await
            .map_err(crate::Error::from)
    }

    /// Read raw posting bytes at offset
    pub async fn read_postings(&self, offset: u64, len: u32) -> Result<Vec<u8>> {
        let start = offset;
        let end = start + len as u64;
        let bytes = self.postings_handle.read_bytes_range(start..end).await?;
        Ok(bytes.to_vec())
    }

    /// Read raw position bytes at offset (for merge)
    pub async fn read_position_bytes(&self, offset: u64, len: u32) -> Result<Option<Vec<u8>>> {
        let handle = match &self.positions_handle {
            Some(h) => h,
            None => return Ok(None),
        };
        let start = offset;
        let end = start + len as u64;
        let bytes = handle.read_bytes_range(start..end).await?;
        Ok(Some(bytes.to_vec()))
    }

    /// Check if this segment has a positions file
    pub fn has_positions_file(&self) -> bool {
        self.positions_handle.is_some()
    }

    /// Batch cosine scoring on raw quantized bytes.
    ///
    /// Dispatches to the appropriate SIMD scorer based on quantization type.
    /// Vectors file uses data-first layout (offset 0) with 8-byte padding between
    /// fields, so mmap slices are always properly aligned for f32/f16/u8 access.
    fn score_quantized_batch(
        query: &[f32],
        raw: &[u8],
        quant: crate::dsl::DenseVectorQuantization,
        dim: usize,
        scores: &mut [f32],
    ) {
        match quant {
            crate::dsl::DenseVectorQuantization::F32 => {
                let num_floats = scores.len() * dim;
                debug_assert!(
                    (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
                    "f32 vector data not 4-byte aligned — vectors file may use legacy format"
                );
                let vectors: &[f32] =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
                crate::structures::simd::batch_cosine_scores(query, vectors, dim, scores);
            }
            crate::dsl::DenseVectorQuantization::F16 => {
                crate::structures::simd::batch_cosine_scores_f16(query, raw, dim, scores);
            }
            crate::dsl::DenseVectorQuantization::UInt8 => {
                crate::structures::simd::batch_cosine_scores_u8(query, raw, dim, scores);
            }
        }
    }

    /// Search dense vectors using RaBitQ
    ///
    /// Returns VectorSearchResult with ordinal tracking for multi-value fields.
    /// The doc_ids are adjusted by doc_id_offset for this segment.
    /// For multi-valued documents, scores are combined using the specified combiner.
    pub async fn search_dense_vector(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        let ann_index = self.vector_indexes.get(&field.0);
        let lazy_flat = self.flat_vectors.get(&field.0);

        // No vectors at all for this field
        if ann_index.is_none() && lazy_flat.is_none() {
            return Ok(Vec::new());
        }

        /// Batch size for brute-force scoring (4096 vectors × 768 dims × 4 bytes ≈ 12MB)
        const BRUTE_FORCE_BATCH: usize = 4096;

        // Results are (doc_id, ordinal, score) where score = similarity (higher = better)
        let mut results: Vec<(u32, u16, f32)> = if let Some(index) = ann_index {
            // ANN search (RaBitQ, IVF, ScaNN)
            match index {
                VectorIndex::RaBitQ(rabitq) => {
                    let fetch_k = k * rerank_factor.max(1);
                    rabitq
                        .search(query, fetch_k, rerank_factor)
                        .into_iter()
                        .map(|(doc_id, ordinal, dist)| (doc_id, ordinal, 1.0 / (1.0 + dist)))
                        .collect()
                }
                VectorIndex::IVF(lazy) => {
                    let (index, codebook) = lazy.get().ok_or_else(|| {
                        Error::Schema("IVF index deserialization failed".to_string())
                    })?;
                    let centroids = self.coarse_centroids.get(&field.0).ok_or_else(|| {
                        Error::Schema(format!(
                            "IVF index requires coarse centroids for field {}",
                            field.0
                        ))
                    })?;
                    let effective_nprobe = if nprobe > 0 { nprobe } else { 32 };
                    let fetch_k = k * rerank_factor.max(1);
                    index
                        .search(centroids, codebook, query, fetch_k, Some(effective_nprobe))
                        .into_iter()
                        .map(|(doc_id, ordinal, dist)| (doc_id, ordinal, 1.0 / (1.0 + dist)))
                        .collect()
                }
                VectorIndex::ScaNN(lazy) => {
                    let (index, codebook) = lazy.get().ok_or_else(|| {
                        Error::Schema("ScaNN index deserialization failed".to_string())
                    })?;
                    let centroids = self.coarse_centroids.get(&field.0).ok_or_else(|| {
                        Error::Schema(format!(
                            "ScaNN index requires coarse centroids for field {}",
                            field.0
                        ))
                    })?;
                    let effective_nprobe = if nprobe > 0 { nprobe } else { 32 };
                    let fetch_k = k * rerank_factor.max(1);
                    index
                        .search(centroids, codebook, query, fetch_k, Some(effective_nprobe))
                        .into_iter()
                        .map(|(doc_id, ordinal, dist)| (doc_id, ordinal, 1.0 / (1.0 + dist)))
                        .collect()
                }
            }
        } else if let Some(lazy_flat) = lazy_flat {
            // Batched brute-force from lazy flat vectors (native-precision SIMD scoring)
            // Uses a top-k heap to avoid collecting and sorting all N candidates.
            let dim = lazy_flat.dim;
            let n = lazy_flat.num_vectors;
            let quant = lazy_flat.quantization;
            let fetch_k = k * rerank_factor.max(1);
            let mut collector = crate::query::ScoreCollector::new(fetch_k);

            for batch_start in (0..n).step_by(BRUTE_FORCE_BATCH) {
                let batch_count = BRUTE_FORCE_BATCH.min(n - batch_start);
                let batch_bytes = lazy_flat
                    .read_vectors_batch(batch_start, batch_count)
                    .await
                    .map_err(crate::Error::Io)?;
                let raw = batch_bytes.as_slice();

                let mut scores = vec![0f32; batch_count];
                Self::score_quantized_batch(query, raw, quant, dim, &mut scores);

                for (i, &score) in scores.iter().enumerate().take(batch_count) {
                    let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
                    collector.insert_with_ordinal(doc_id, score, ordinal);
                }
            }

            collector
                .into_sorted_results()
                .into_iter()
                .map(|(doc_id, score, ordinal)| (doc_id, ordinal, score))
                .collect()
        } else {
            return Ok(Vec::new());
        };

        // Rerank ANN candidates using raw vectors from lazy flat (binary search lookup)
        // Uses native-precision SIMD scoring on quantized bytes — no dequantization overhead.
        if ann_index.is_some()
            && !results.is_empty()
            && let Some(lazy_flat) = lazy_flat
        {
            let dim = lazy_flat.dim;
            let quant = lazy_flat.quantization;
            let vbs = lazy_flat.vector_byte_size();

            // Resolve flat indexes for each candidate via binary search
            let mut resolved: Vec<(usize, usize)> = Vec::new(); // (result_idx, flat_idx)
            for (ri, c) in results.iter().enumerate() {
                let (start, entries) = lazy_flat.flat_indexes_for_doc(c.0);
                for (j, &(_, ord)) in entries.iter().enumerate() {
                    if ord == c.1 {
                        resolved.push((ri, start + j));
                        break;
                    }
                }
            }

            if !resolved.is_empty() {
                // Sort by flat_idx for sequential mmap access (better page locality)
                resolved.sort_unstable_by_key(|&(_, flat_idx)| flat_idx);

                // Batch-read raw quantized bytes into contiguous buffer
                let mut raw_buf = vec![0u8; resolved.len() * vbs];
                for (buf_idx, &(_, flat_idx)) in resolved.iter().enumerate() {
                    let _ = lazy_flat
                        .read_vector_raw_into(
                            flat_idx,
                            &mut raw_buf[buf_idx * vbs..(buf_idx + 1) * vbs],
                        )
                        .await;
                }

                // Native-precision batch SIMD cosine scoring
                let mut scores = vec![0f32; resolved.len()];
                Self::score_quantized_batch(query, &raw_buf, quant, dim, &mut scores);

                // Write scores back to results
                for (buf_idx, &(ri, _)) in resolved.iter().enumerate() {
                    results[ri].2 = scores[buf_idx];
                }
            }

            results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(k * rerank_factor.max(1));
        }

        // Track ordinals with individual scores for each doc_id
        // Note: doc_id_offset is NOT applied here - the collector applies it uniformly
        let mut doc_ordinals: rustc_hash::FxHashMap<DocId, Vec<(u32, f32)>> =
            rustc_hash::FxHashMap::default();
        for (doc_id, ordinal, score) in results {
            let ordinals = doc_ordinals.entry(doc_id as DocId).or_default();
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

    /// Check if this segment has dense vectors for the given field
    pub fn has_dense_vector_index(&self, field: Field) -> bool {
        self.vector_indexes.contains_key(&field.0) || self.flat_vectors.contains_key(&field.0)
    }

    /// Get the dense vector index for a field (if available)
    pub fn get_dense_vector_index(&self, field: Field) -> Option<Arc<RaBitQIndex>> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::RaBitQ(idx)) => Some(idx.clone()),
            _ => None,
        }
    }

    /// Get the IVF vector index for a field (if available)
    pub fn get_ivf_vector_index(
        &self,
        field: Field,
    ) -> Option<(Arc<IVFRaBitQIndex>, Arc<crate::structures::RaBitQCodebook>)> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::IVF(lazy)) => lazy.get().map(|(i, c)| (i.clone(), c.clone())),
            _ => None,
        }
    }

    /// Get coarse centroids for a field
    pub fn coarse_centroids(&self, field_id: u32) -> Option<&Arc<CoarseCentroids>> {
        self.coarse_centroids.get(&field_id)
    }

    /// Set per-field coarse centroids from index-level trained structures
    pub fn set_coarse_centroids(&mut self, centroids: FxHashMap<u32, Arc<CoarseCentroids>>) {
        self.coarse_centroids = centroids;
    }

    /// Get the ScaNN vector index for a field (if available)
    pub fn get_scann_vector_index(
        &self,
        field: Field,
    ) -> Option<(Arc<IVFPQIndex>, Arc<PQCodebook>)> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::ScaNN(lazy)) => lazy.get().map(|(i, c)| (i.clone(), c.clone())),
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
        heap_factor: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        use crate::query::{BlockMaxScoreExecutor, BmpExecutor, SparseTermScorer};

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

        // Filter query terms to only those present in the index
        let mut matched_terms: Vec<(u32, f32)> = Vec::with_capacity(vector.len());
        let mut missing_tokens = Vec::new();

        for &(dim_id, query_weight) in vector {
            if sparse_index.has_dimension(dim_id) {
                matched_terms.push((dim_id, query_weight));
            } else {
                missing_tokens.push(dim_id);
            }
        }

        log::debug!(
            "Sparse vector search: query_tokens={}, matched={}, missing={}, index_dimensions={}",
            query_tokens,
            matched_terms.len(),
            missing_tokens.len(),
            index_dimensions
        );

        if log::log_enabled!(log::Level::Debug) {
            let query_details: Vec<_> = vector
                .iter()
                .take(30)
                .map(|(id, w)| format!("{}:{:.3}", id, w))
                .collect();
            log::debug!("Query tokens (id:weight): [{}]", query_details.join(", "));
        }

        if !missing_tokens.is_empty() {
            log::debug!(
                "Missing token IDs (not in index): {:?}",
                missing_tokens.iter().take(20).collect::<Vec<_>>()
            );
        }

        if matched_terms.is_empty() {
            log::debug!("Sparse vector search: no matching tokens, returning empty");
            return Ok(Vec::new());
        }

        // Select executor based on number of query terms:
        // - 12+ terms: BMP (block-at-a-time, lazy block loading, best for SPLADE)
        // - 1-11 terms: BlockMaxScoreExecutor (unified MaxScore + block-max + conjunction)
        let num_terms = matched_terms.len();
        let over_fetch = limit * 2; // Over-fetch for multi-value combining
        let raw_results = if num_terms > 12 {
            // BMP: lazy block loading — only skip entries in memory, blocks loaded on-demand
            BmpExecutor::new(sparse_index, matched_terms, over_fetch, heap_factor)
                .execute()
                .await?
        } else {
            // Load posting lists only for the few terms (1-11) used by BlockMaxScore
            let mut posting_lists: Vec<(u32, f32, Arc<BlockSparsePostingList>)> =
                Vec::with_capacity(num_terms);
            for &(dim_id, query_weight) in &matched_terms {
                if let Some(pl) = sparse_index.get_posting(dim_id).await? {
                    posting_lists.push((dim_id, query_weight, pl));
                }
            }
            let scorers: Vec<SparseTermScorer> = posting_lists
                .iter()
                .map(|(_, query_weight, pl)| SparseTermScorer::from_arc(pl, *query_weight))
                .collect();
            if scorers.is_empty() {
                return Ok(Vec::new());
            }
            BlockMaxScoreExecutor::with_heap_factor(scorers, over_fetch, heap_factor).execute()
        };

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
        // Note: doc_id_offset is NOT applied here - the collector applies it uniformly
        let mut results: Vec<VectorSearchResult> = doc_ordinals
            .into_iter()
            .map(|(doc_id, ordinals)| {
                let combined_score = combiner.combine(&ordinals);
                VectorSearchResult::new(doc_id, combined_score, ordinals)
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
