//! Types for segment reader

use std::io::Cursor;
use std::sync::{Arc, OnceLock};

use crate::DocId;
use crate::directories::{AsyncFileRead, LazyFileHandle, OwnedBytes};
use crate::structures::{
    BlockSparsePostingList, IVFPQIndex, IVFRaBitQIndex, PQCodebook, RaBitQCodebook, RaBitQIndex,
    SparseBlock, SparseSkipEntry,
};

/// Vector index type - RaBitQ, IVF-RaBitQ, or ScaNN (IVF-PQ)
///
/// Raw flat vectors are stored separately in [`LazyFlatVectorData`] and accessed
/// via mmap for reranking and merge. This enum only holds ANN indexes.
///
/// IVF and ScaNN variants are **lazy**: `OwnedBytes` (zero-copy mmap ref) are
/// stored on construction, bincode deserialization is deferred to first search
/// access via `OnceLock`. No heap copies during segment load.
#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum VectorIndex {
    /// RaBitQ - binary quantization, good for small datasets
    RaBitQ(Arc<RaBitQIndex>),
    /// IVF-RaBitQ - lazy deserialization on first access
    IVF(Arc<LazyIVF>),
    /// ScaNN (IVF-PQ) - lazy deserialization on first access
    ScaNN(Arc<LazyScaNN>),
}

/// Lazy IVF-RaBitQ index — defers bincode deserialization to first access
///
/// Stores `OwnedBytes` which for mmap directories is a zero-copy reference.
/// The mmap pages are only paged into physical RAM when deserialization happens.
pub struct LazyIVF {
    raw: OwnedBytes,
    resolved: OnceLock<Option<(Arc<IVFRaBitQIndex>, Arc<RaBitQCodebook>)>>,
}

impl LazyIVF {
    pub fn new(raw: OwnedBytes) -> Self {
        Self {
            raw,
            resolved: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Option<(&Arc<IVFRaBitQIndex>, &Arc<RaBitQCodebook>)> {
        self.resolved
            .get_or_init(|| {
                match super::super::vector_data::IVFRaBitQIndexData::from_bytes(&self.raw) {
                    Ok(data) => Some((Arc::new(data.index), Arc::new(data.codebook))),
                    Err(e) => {
                        log::warn!("[lazy_ivf] deserialization failed: {}", e);
                        None
                    }
                }
            })
            .as_ref()
            .map(|(i, c)| (i, c))
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        match self.resolved.get() {
            Some(Some((index, codebook))) => {
                index.estimated_memory_bytes() + codebook.estimated_memory_bytes()
            }
            _ => self.raw.len(),
        }
    }
}

/// Lazy ScaNN (IVF-PQ) index — defers bincode deserialization to first access
///
/// Stores `OwnedBytes` which for mmap directories is a zero-copy reference.
/// The mmap pages are only paged into physical RAM when deserialization happens.
pub struct LazyScaNN {
    raw: OwnedBytes,
    resolved: OnceLock<Option<(Arc<IVFPQIndex>, Arc<PQCodebook>)>>,
}

impl LazyScaNN {
    pub fn new(raw: OwnedBytes) -> Self {
        Self {
            raw,
            resolved: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Option<(&Arc<IVFPQIndex>, &Arc<PQCodebook>)> {
        self.resolved
            .get_or_init(|| {
                match super::super::vector_data::ScaNNIndexData::from_bytes(&self.raw) {
                    Ok(data) => Some((Arc::new(data.index), Arc::new(data.codebook))),
                    Err(e) => {
                        log::warn!("[lazy_scann] deserialization failed: {}", e);
                        None
                    }
                }
            })
            .as_ref()
            .map(|(i, c)| (i, c))
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        match self.resolved.get() {
            Some(Some((index, codebook))) => {
                index.estimated_memory_bytes() + codebook.estimated_memory_bytes()
            }
            _ => self.raw.len(),
        }
    }
}

impl VectorIndex {
    /// Estimate memory usage of this vector index
    pub fn estimated_memory_bytes(&self) -> usize {
        match self {
            VectorIndex::RaBitQ(idx) => idx.estimated_memory_bytes(),
            VectorIndex::IVF(lazy) => lazy.estimated_memory_bytes(),
            VectorIndex::ScaNN(lazy) => lazy.estimated_memory_bytes(),
        }
    }
}

/// Raw dimension data for zero-copy merge (no block deserialization).
///
/// Contains the skip entries (already in memory from segment open),
/// metadata, and raw block data bytes read directly from mmap.
pub struct DimRawData<'a> {
    /// Skip entries for this dimension (borrowed from SparseIndex)
    pub skip_entries: &'a [SparseSkipEntry],
    /// Total document count in this dimension's posting list
    pub doc_count: u32,
    /// Global max weight across all blocks
    pub global_max_weight: f32,
    /// Raw block data bytes (single contiguous mmap read)
    pub raw_block_data: crate::directories::OwnedBytes,
}

/// Sparse vector index for a field: lazy block loading via mmap
///
/// Stores skip list per dimension (block metadata). Blocks loaded on-demand.
/// Memory: only skip list headers in RAM, block data via mmap (OS page cache).
///
/// Skip entries are stored in a single flat array shared across all dimensions
/// (eliminates per-dimension heap allocations — one alloc instead of 73K+).
#[derive(Clone)]
pub struct SparseIndex {
    /// Mmap file handle for sparse data
    handle: LazyFileHandle,
    /// Per-dimension data, sorted by dim_id for binary search
    dimensions: Arc<Vec<DimensionEntry>>,
    /// All skip entries in a single flat array (dimensions reference slices via skip_start/skip_count)
    skip_entries: Arc<Vec<SparseSkipEntry>>,
    /// Total document count in this segment (for IDF computation)
    pub total_docs: u32,
    /// Total sparse vectors in this segment (for multi-valued IDF)
    pub total_vectors: u32,
}

/// Per-dimension skip list data
#[derive(Clone)]
pub struct DimensionEntry {
    pub dim_id: u32,
    /// Base offset in file where block data starts for this dimension
    pub data_offset: u64,
    /// Total documents in this dimension's posting list
    pub doc_count: u32,
    /// Global max weight across all blocks
    pub global_max_weight: f32,
    /// Start index into the shared skip_entries flat array
    pub skip_start: u32,
    /// Number of skip entries for this dimension
    pub skip_count: u32,
}

impl SparseIndex {
    /// Create a new sparse index with lazy block loading
    pub fn new(
        handle: LazyFileHandle,
        dimensions: Vec<DimensionEntry>,
        skip_entries: Vec<SparseSkipEntry>,
        total_docs: u32,
        total_vectors: u32,
    ) -> Self {
        Self {
            handle,
            dimensions: Arc::new(dimensions),
            skip_entries: Arc::new(skip_entries),
            total_docs,
            total_vectors,
        }
    }

    /// Find dimension entry via binary search
    #[inline]
    fn get_dimension(&self, dim_id: u32) -> Option<&DimensionEntry> {
        self.dimensions
            .binary_search_by_key(&dim_id, |d| d.dim_id)
            .ok()
            .map(|idx| &self.dimensions[idx])
    }

    /// Get the skip entries slice for a dimension
    #[inline]
    fn dim_skip_entries(&self, dim: &DimensionEntry) -> &[SparseSkipEntry] {
        let start = dim.skip_start as usize;
        let end = start + dim.skip_count as usize;
        &self.skip_entries[start..end]
    }

    /// Load a single block via mmap (OS page cache handles caching)
    async fn load_block(
        &self,
        dim: &DimensionEntry,
        block_idx: usize,
    ) -> crate::Result<SparseBlock> {
        let skip = self.dim_skip_entries(dim);
        let entry = &skip[block_idx];
        let abs_offset = dim.data_offset + entry.offset as u64;
        let data = self
            .handle
            .read_bytes_range(abs_offset..abs_offset + entry.length as u64)
            .await
            .map_err(crate::Error::Io)?;

        Ok(SparseBlock::read(&mut Cursor::new(data.as_slice()))?)
    }

    /// Get posting list for a dimension (loads all blocks via mmap)
    pub async fn get_posting(
        &self,
        dim_id: u32,
    ) -> crate::Result<Option<Arc<BlockSparsePostingList>>> {
        let dim = match self.get_dimension(dim_id) {
            Some(d) => d,
            None => return Ok(None),
        };

        // Load all blocks
        let skip = self.dim_skip_entries(dim);
        let mut blocks = Vec::with_capacity(skip.len());
        for i in 0..skip.len() {
            let block = self.load_block(dim, i).await?;
            blocks.push(block);
        }

        Ok(Some(Arc::new(BlockSparsePostingList {
            doc_count: dim.doc_count,
            blocks,
        })))
    }

    /// Get skip list for a dimension (for block-max iteration without loading blocks)
    pub fn get_skip_list(&self, dim_id: u32) -> Option<(&[SparseSkipEntry], f32)> {
        self.get_dimension(dim_id)
            .map(|d| (self.dim_skip_entries(d), d.global_max_weight))
    }

    /// Load specific block for a dimension
    pub async fn get_block(
        &self,
        dim_id: u32,
        block_idx: usize,
    ) -> crate::Result<Option<SparseBlock>> {
        let dim = match self.get_dimension(dim_id) {
            Some(d) => d,
            None => return Ok(None),
        };
        let skip = self.dim_skip_entries(dim);
        if block_idx >= skip.len() {
            return Ok(None);
        }
        Ok(Some(self.load_block(dim, block_idx).await?))
    }

    /// Check if dimension exists
    #[inline]
    pub fn has_dimension(&self, dim_id: u32) -> bool {
        self.get_dimension(dim_id).is_some()
    }

    /// Get the number of dimensions in the index
    #[inline]
    pub fn num_dimensions(&self) -> usize {
        self.dimensions.len()
    }

    /// Iterate over all active dimension IDs
    pub fn active_dimensions(&self) -> impl Iterator<Item = u32> + '_ {
        self.dimensions.iter().map(|d| d.dim_id)
    }

    /// Get doc count for dimension (from skip list, no I/O)
    pub fn doc_count(&self, dim_id: u32) -> u32 {
        self.get_dimension(dim_id).map(|d| d.doc_count).unwrap_or(0)
    }

    /// Compute IDF using doc_count from skip list
    ///
    /// doc_count tracks unique documents per dimension (not ordinals),
    /// so df <= total_docs is always true. max(total_vectors, total_docs)
    /// is a safety invariant.
    #[inline]
    pub fn idf(&self, dim_id: u32) -> f32 {
        let df = self.doc_count(dim_id) as f32;
        if df > 0.0 {
            let n = self.total_vectors.max(self.total_docs) as f32;
            (n / df).ln().max(0.0)
        } else {
            0.0
        }
    }

    /// Get IDF weights for multiple dimensions
    pub fn idf_weights(&self, dim_ids: &[u32]) -> Vec<f32> {
        dim_ids.iter().map(|&d| self.idf(d)).collect()
    }

    /// Get merge-level info for a dimension: skip entries, doc_count, global_max_weight,
    /// and raw block data bytes (single mmap read, zero deserialization).
    ///
    /// Used by sparse merge for O(blocks) byte-level block stacking: the caller
    /// writes a new header + adjusted skip entries + copies raw bytes directly,
    /// avoiding deserialize → clone → re-serialize overhead entirely.
    pub async fn read_dim_raw(&self, dim_id: u32) -> crate::Result<Option<DimRawData<'_>>> {
        let dim = match self.get_dimension(dim_id) {
            Some(d) => d,
            None => return Ok(None),
        };
        let skip = self.dim_skip_entries(dim);
        if skip.is_empty() {
            return Ok(None);
        }
        // Total block data size: last entry's (offset + length)
        let last = &skip[skip.len() - 1];
        let total_bytes = last.offset as u64 + last.length as u64;
        let raw = self
            .handle
            .read_bytes_range(dim.data_offset..dim.data_offset + total_bytes)
            .await
            .map_err(crate::Error::Io)?;
        Ok(Some(DimRawData {
            skip_entries: skip,
            doc_count: dim.doc_count,
            global_max_weight: dim.global_max_weight,
            raw_block_data: raw,
        }))
    }
}

/// Vector search result with ordinal tracking for multi-value fields
///
/// Each result contains the combined score and individual contributions
/// from each ordinal (for multi-valued vector fields)
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Document ID
    pub doc_id: DocId,
    /// Combined score (after applying combiner: Sum/Max/Avg)
    pub score: f32,
    /// Individual ordinal contributions: (ordinal, score)
    /// For single-value fields, this will have one entry with ordinal 0
    pub ordinals: Vec<(u32, f32)>,
}

impl VectorSearchResult {
    pub fn new(doc_id: DocId, score: f32, ordinals: Vec<(u32, f32)>) -> Self {
        Self {
            doc_id,
            score,
            ordinals,
        }
    }
}
