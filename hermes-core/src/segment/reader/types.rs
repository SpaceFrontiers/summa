//! Types for segment reader

use std::io::Cursor;
use std::sync::Arc;

use crate::DocId;
use crate::directories::{AsyncFileRead, LazyFileHandle};
use crate::structures::{
    BlockSparsePostingList, IVFPQIndex, IVFRaBitQIndex, PQCodebook, RaBitQCodebook, RaBitQIndex,
    SparseBlock, SparseSkipEntry,
};

use super::super::vector_data::FlatVectorData;

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

impl VectorIndex {
    /// Estimate memory usage of this vector index
    pub fn estimated_memory_bytes(&self) -> usize {
        match self {
            VectorIndex::Flat(data) => data.estimated_memory_bytes(),
            VectorIndex::RaBitQ(idx) => idx.estimated_memory_bytes(),
            VectorIndex::IVF { index, codebook } => {
                index.estimated_memory_bytes() + codebook.estimated_memory_bytes()
            }
            VectorIndex::ScaNN { index, codebook } => {
                index.estimated_memory_bytes() + codebook.estimated_memory_bytes()
            }
        }
    }
}

/// Sparse vector index for a field: lazy block loading via mmap
///
/// Stores skip list per dimension (block metadata). Blocks loaded on-demand.
/// Memory: only skip list headers in RAM, block data via mmap (OS page cache).
#[derive(Clone)]
pub struct SparseIndex {
    /// Mmap file handle for sparse data
    handle: LazyFileHandle,
    /// Per-dimension data: [(dim_id, data_offset, doc_count, global_max_weight, skip_entries)]
    /// Sorted by dim_id for binary search
    dimensions: Arc<Vec<DimensionEntry>>,
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
    /// Skip list entries (block metadata)
    pub skip_entries: Vec<SparseSkipEntry>,
}

impl SparseIndex {
    /// Create a new sparse index with lazy block loading
    pub fn new(
        handle: LazyFileHandle,
        dimensions: Vec<DimensionEntry>,
        total_docs: u32,
        total_vectors: u32,
    ) -> Self {
        Self {
            handle,
            dimensions: Arc::new(dimensions),
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

    /// Load a single block via mmap (OS page cache handles caching)
    async fn load_block(
        &self,
        dim: &DimensionEntry,
        block_idx: usize,
    ) -> crate::Result<SparseBlock> {
        let entry = &dim.skip_entries[block_idx];
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
        let mut blocks = Vec::with_capacity(dim.skip_entries.len());
        for i in 0..dim.skip_entries.len() {
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
            .map(|d| (d.skip_entries.as_slice(), d.global_max_weight))
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
        if block_idx >= dim.skip_entries.len() {
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
