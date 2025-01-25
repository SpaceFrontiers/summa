//! Async segment reader with lazy loading

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::{AsyncFileRead, Directory, LazyFileHandle, LazyFileSlice};
use crate::dsl::{Document, Field, Schema};
use crate::structures::{
    AsyncSSTableReader, BlockPostingList, BlockSparsePostingList, CoarseCentroids, IVFPQIndex,
    IVFRaBitQIndex, PQCodebook, RaBitQCodebook, RaBitQIndex, SSTableStats, TermInfo,
};
use crate::{DocId, Error, Result};

use super::store::{AsyncStoreReader, RawStoreBlock};
use super::types::{SegmentFiles, SegmentId, SegmentMeta};
use super::vector_data::FlatVectorData;

/// Vector index type - Flat, RaBitQ, IVF-RaBitQ, or ScaNN (IVF-PQ)
#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum VectorIndex {
    /// Flat - brute-force search over raw vectors (accumulating state)
    Flat(Arc<FlatVectorData>),
    /// RaBitQ - binary quantization, good for small datasets
    RaBitQ(Arc<RaBitQIndex>),
    /// IVF-RaBitQ - inverted file with RaBitQ, good for medium datasets
    IVF {
        index: Arc<IVFRaBitQIndex>,
        codebook: Arc<RaBitQCodebook>,
    },
    /// ScaNN (IVF-PQ) - product quantization with OPQ, best for large datasets
    ScaNN {
        index: Arc<IVFPQIndex>,
        codebook: Arc<PQCodebook>,
    },
}

/// Sparse vector index for a field: direct-indexed by dimension ID
#[derive(Clone)]
pub struct SparseIndex {
    /// Posting lists indexed directly by dimension ID (O(1) lookup)
    /// None means dimension not present in index
    pub postings: Vec<Option<Arc<BlockSparsePostingList>>>,
    /// Total document count in this segment (for IDF computation)
    pub total_docs: u32,
}

impl SparseIndex {
    /// Compute IDF (inverse document frequency) for a dimension
    ///
    /// IDF = log(N / df) where N = total docs, df = docs containing dimension
    /// Returns 0.0 if dimension not present
    #[inline]
    pub fn idf(&self, dim_id: u32) -> f32 {
        if let Some(Some(pl)) = self.postings.get(dim_id as usize) {
            let df = pl.doc_count() as f32;
            if df > 0.0 {
                (self.total_docs as f32 / df).ln()
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Get IDF weights for multiple dimensions
    pub fn idf_weights(&self, dim_ids: &[u32]) -> Vec<f32> {
        dim_ids.iter().map(|&d| self.idf(d)).collect()
    }
}

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
            Self::load_vectors_file(dir, &files, &schema).await?;

        // Load sparse vector indexes from .sparse file
        let sparse_indexes = Self::load_sparse_file(dir, &files, meta.num_docs).await?;

        // Open positions file handle (if exists) - offsets are now in TermInfo
        let positions_handle = Self::open_positions_file(dir, &files).await?;

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
    /// Returns (doc_id, score) pairs sorted by score (descending).
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
    ) -> Result<Vec<(DocId, f32)>> {
        use crate::query::MultiValueCombiner;
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

        let results: Vec<(u32, f32)> = match index {
            VectorIndex::Flat(flat_data) => {
                // Brute-force search over raw vectors using SIMD-accelerated distance
                use crate::structures::simd::squared_euclidean_distance;

                let mut candidates: Vec<(u32, f32)> = flat_data
                    .vectors
                    .iter()
                    .zip(flat_data.doc_ids.iter())
                    .map(|(vec, &doc_id)| {
                        let dist = squared_euclidean_distance(effective_query, vec);
                        (doc_id, dist)
                    })
                    .collect();
                candidates
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates.truncate(k);
                candidates
            }
            VectorIndex::RaBitQ(rabitq) => rabitq
                .search(effective_query, k, rerank_factor)
                .into_iter()
                .map(|(idx, dist)| (idx as u32, dist))
                .collect(),
            VectorIndex::IVF { index, codebook } => {
                let centroids = self.coarse_centroids.as_ref().ok_or_else(|| {
                    Error::Schema("IVF index requires coarse centroids".to_string())
                })?;
                let nprobe = rerank_factor.max(32); // Use rerank_factor as nprobe hint
                index.search(centroids, codebook, effective_query, k, Some(nprobe))
            }
            VectorIndex::ScaNN { index, codebook } => {
                let centroids = self.coarse_centroids.as_ref().ok_or_else(|| {
                    Error::Schema("ScaNN index requires coarse centroids".to_string())
                })?;
                let nprobe = rerank_factor.max(32);
                index.search(centroids, codebook, effective_query, k, Some(nprobe))
            }
        };

        // Convert distance to score (smaller distance = higher score)
        // and adjust doc_ids by segment offset
        let raw_results: Vec<(DocId, f32)> = results
            .into_iter()
            .map(|(idx, dist)| {
                let doc_id = idx as DocId + self.doc_id_offset;
                let score = 1.0 / (1.0 + dist); // Convert distance to similarity score
                (doc_id, score)
            })
            .collect();

        // Combine scores for duplicate doc_ids (multi-valued documents)
        let mut combined: rustc_hash::FxHashMap<DocId, (f32, u32)> =
            rustc_hash::FxHashMap::default();
        for (doc_id, score) in raw_results {
            combined
                .entry(doc_id)
                .and_modify(|(acc_score, count)| match combiner {
                    MultiValueCombiner::Sum => *acc_score += score,
                    MultiValueCombiner::Max => *acc_score = acc_score.max(score),
                    MultiValueCombiner::Avg => {
                        *acc_score += score;
                        *count += 1;
                    }
                })
                .or_insert((score, 1));
        }

        // Finalize averages and collect results
        let mut final_results: Vec<(DocId, f32)> = combined
            .into_iter()
            .map(|(doc_id, (score, count))| {
                let final_score = if combiner == MultiValueCombiner::Avg {
                    score / count as f32
                } else {
                    score
                };
                (doc_id, final_score)
            })
            .collect();

        // Sort by score descending and take top k
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
    /// Returns (doc_id, score) pairs sorted by score descending.
    pub async fn search_sparse_vector(
        &self,
        field: Field,
        vector: &[(u32, f32)],
        limit: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<(u32, f32)>> {
        use crate::query::{MultiValueCombiner, SparseTermScorer, WandExecutor};

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

        let index_dimensions = sparse_index.postings.len();

        // Build scorers for each dimension that exists in the index
        let mut matched_tokens = Vec::new();
        let mut missing_tokens = Vec::new();

        let scorers: Vec<SparseTermScorer> = vector
            .iter()
            .filter_map(|&(dim_id, query_weight)| {
                // Direct indexing: O(1) lookup
                match sparse_index
                    .postings
                    .get(dim_id as usize)
                    .and_then(|opt| opt.as_ref())
                {
                    Some(pl) => {
                        matched_tokens.push(dim_id);
                        Some(SparseTermScorer::from_arc(pl, query_weight))
                    }
                    None => {
                        missing_tokens.push(dim_id);
                        None
                    }
                }
            })
            .collect();

        log::debug!(
            "Sparse vector search: query_tokens={}, matched={}, missing={}, index_dimensions={}",
            query_tokens,
            matched_tokens.len(),
            missing_tokens.len(),
            index_dimensions
        );

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

        // Combine scores for duplicate doc_ids based on combiner strategy
        let mut combined: rustc_hash::FxHashMap<u32, (f32, u32)> = rustc_hash::FxHashMap::default();
        for r in raw_results {
            combined
                .entry(r.doc_id)
                .and_modify(|(score, count)| match combiner {
                    MultiValueCombiner::Sum => *score += r.score,
                    MultiValueCombiner::Max => *score = score.max(r.score),
                    MultiValueCombiner::Avg => {
                        *score += r.score;
                        *count += 1;
                    }
                })
                .or_insert((r.score, 1));
        }

        // Finalize averages and collect results
        let mut results: Vec<(u32, f32)> = combined
            .into_iter()
            .map(|(doc_id, (score, count))| {
                let final_score = if combiner == MultiValueCombiner::Avg {
                    score / count as f32
                } else {
                    score
                };
                (doc_id, final_score)
            })
            .collect();

        // Sort by score descending and take top limit
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Load dense vector indexes from unified .vectors file
    ///
    /// Supports RaBitQ (type 0), IVF-RaBitQ (type 1), and ScaNN (type 2).
    /// Also loads coarse centroids and PQ codebook as needed.
    ///
    /// Memory optimization: Uses lazy range reads to load each index separately,
    /// avoiding loading the entire vectors file into memory at once.
    async fn load_vectors_file<D: Directory>(
        dir: &D,
        files: &SegmentFiles,
        schema: &Schema,
    ) -> Result<(FxHashMap<u32, VectorIndex>, Option<Arc<CoarseCentroids>>)> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;

        let mut indexes = FxHashMap::default();
        let mut coarse_centroids: Option<Arc<CoarseCentroids>> = None;

        // Skip loading vectors file if schema has no dense vector fields
        let has_dense_vectors = schema
            .fields()
            .any(|(_, entry)| entry.dense_vector_config.is_some());
        if !has_dense_vectors {
            return Ok((indexes, None));
        }

        // Try to open vectors file (may not exist if no vectors were indexed)
        let handle = match dir.open_lazy(&files.vectors).await {
            Ok(h) => h,
            Err(_) => return Ok((indexes, None)),
        };

        // Read only the header first (4 bytes for num_fields)
        let header_bytes = match handle.read_bytes_range(0..4).await {
            Ok(b) => b,
            Err(_) => return Ok((indexes, None)),
        };

        if header_bytes.is_empty() {
            return Ok((indexes, None));
        }

        let mut cursor = Cursor::new(header_bytes.as_slice());
        let num_fields = cursor.read_u32::<LittleEndian>()?;

        if num_fields == 0 {
            return Ok((indexes, None));
        }

        // Read field entries header: (field_id: 4, index_type: 1, offset: 8, length: 8) = 21 bytes per field
        let entries_size = num_fields as u64 * 21;
        let entries_bytes = handle.read_bytes_range(4..4 + entries_size).await?;
        let mut cursor = Cursor::new(entries_bytes.as_slice());

        // Read field entries (field_id, index_type, offset, length)
        let mut entries = Vec::with_capacity(num_fields as usize);
        for _ in 0..num_fields {
            let field_id = cursor.read_u32::<LittleEndian>()?;
            // Try to read index_type - if this fails, assume old format without type
            let index_type = cursor.read_u8().unwrap_or(255); // 255 = unknown/legacy
            let offset = cursor.read_u64::<LittleEndian>()?;
            let length = cursor.read_u64::<LittleEndian>()?;
            entries.push((field_id, index_type, offset, length));
        }

        // Load each index on-demand using range reads (memory efficient)
        for (field_id, index_type, offset, length) in entries {
            // Read only this index's data
            let data = handle.read_bytes_range(offset..offset + length).await?;
            let _field = crate::dsl::Field(field_id);

            match index_type {
                3 => {
                    // Flat (brute-force) - raw vectors for accumulating state
                    if let Ok(flat_data) = serde_json::from_slice::<FlatVectorData>(data.as_slice())
                    {
                        indexes.insert(field_id, VectorIndex::Flat(Arc::new(flat_data)));
                    }
                }
                2 => {
                    // ScaNN (IVF-PQ) with embedded centroids and codebook
                    use super::vector_data::ScaNNIndexData;
                    if let Ok(scann_data) = ScaNNIndexData::from_bytes(data.as_slice()) {
                        coarse_centroids = Some(Arc::new(scann_data.centroids));
                        indexes.insert(
                            field_id,
                            VectorIndex::ScaNN {
                                index: Arc::new(scann_data.index),
                                codebook: Arc::new(scann_data.codebook),
                            },
                        );
                    }
                }
                1 => {
                    // IVF-RaBitQ with embedded centroids and codebook
                    use super::vector_data::IVFRaBitQIndexData;
                    if let Ok(ivf_data) = IVFRaBitQIndexData::from_bytes(data.as_slice()) {
                        coarse_centroids = Some(Arc::new(ivf_data.centroids));
                        indexes.insert(
                            field_id,
                            VectorIndex::IVF {
                                index: Arc::new(ivf_data.index),
                                codebook: Arc::new(ivf_data.codebook),
                            },
                        );
                    }
                }
                0 => {
                    // RaBitQ (standalone)
                    if let Ok(rabitq_index) = serde_json::from_slice::<RaBitQIndex>(data.as_slice())
                    {
                        indexes.insert(field_id, VectorIndex::RaBitQ(Arc::new(rabitq_index)));
                    }
                }
                _ => {
                    // Unknown type - try Flat first (most common in new indexes)
                    if let Ok(flat_data) = serde_json::from_slice::<FlatVectorData>(data.as_slice())
                    {
                        indexes.insert(field_id, VectorIndex::Flat(Arc::new(flat_data)));
                    } else if let Ok(rabitq_index) =
                        serde_json::from_slice::<RaBitQIndex>(data.as_slice())
                    {
                        indexes.insert(field_id, VectorIndex::RaBitQ(Arc::new(rabitq_index)));
                    }
                }
            }
        }

        Ok((indexes, coarse_centroids))
    }

    /// Load sparse vector indexes from .sparse file
    ///
    /// File format (direct-indexed table for O(1) dimension lookup):
    /// - Header: num_fields (u32)
    /// - For each field:
    ///   - field_id (u32)
    ///   - quantization (u8)
    ///   - max_dim_id (u32)          ← table size
    ///   - table: [(offset: u64, length: u32)] × max_dim_id  ← direct indexed
    ///     (offset=0, length=0 means dimension not present)
    /// - Data: concatenated serialized BlockSparsePostingList
    async fn load_sparse_file<D: Directory>(
        dir: &D,
        files: &SegmentFiles,
        total_docs: u32,
    ) -> Result<FxHashMap<u32, SparseIndex>> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;

        let mut indexes = FxHashMap::default();

        // Try to open sparse file (may not exist if no sparse vectors were indexed)
        let handle = match dir.open_lazy(&files.sparse).await {
            Ok(h) => h,
            Err(_) => return Ok(indexes),
        };

        // Read the entire file (sparse files are typically small enough)
        let data = match handle.read_bytes().await {
            Ok(d) => d,
            Err(_) => return Ok(indexes),
        };

        if data.len() < 4 {
            return Ok(indexes);
        }

        let mut cursor = Cursor::new(data.as_slice());
        let num_fields = cursor.read_u32::<LittleEndian>()?;

        if num_fields == 0 {
            return Ok(indexes);
        }

        // Read field entries and build indexes
        for _ in 0..num_fields {
            let field_id = cursor.read_u32::<LittleEndian>()?;
            let _quantization = cursor.read_u8()?; // Already stored in each BlockSparsePostingList
            let max_dim_id = cursor.read_u32::<LittleEndian>()?;

            // Read direct-indexed table
            let mut postings: Vec<Option<Arc<BlockSparsePostingList>>> =
                vec![None; max_dim_id as usize];

            for dim_id in 0..max_dim_id {
                let offset = cursor.read_u64::<LittleEndian>()?;
                let length = cursor.read_u32::<LittleEndian>()?;

                // offset=0, length=0 means dimension not present
                if length > 0 {
                    let start = offset as usize;
                    let end = start + length as usize;
                    if end <= data.len() {
                        let posting_data = &data.as_slice()[start..end];
                        if let Ok(posting_list) =
                            BlockSparsePostingList::deserialize(&mut Cursor::new(posting_data))
                        {
                            postings[dim_id as usize] = Some(Arc::new(posting_list));
                        }
                    }
                }
            }

            indexes.insert(
                field_id,
                SparseIndex {
                    postings,
                    total_docs,
                },
            );
        }

        Ok(indexes)
    }

    /// Load position index header from .pos file
    ///
    /// File format:
    /// Open positions file handle (no header parsing needed - offsets are in TermInfo)
    async fn open_positions_file<D: Directory>(
        dir: &D,
        files: &SegmentFiles,
    ) -> Result<Option<LazyFileHandle>> {
        // Try to open positions file (may not exist if no positions were indexed)
        match dir.open_lazy(&files.positions).await {
            Ok(h) => Ok(Some(h)),
            Err(_) => Ok(None),
        }
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
