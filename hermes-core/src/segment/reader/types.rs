//! Types for segment reader

use std::sync::Arc;

use crate::DocId;
use crate::structures::{
    BlockSparsePostingList, IVFPQIndex, IVFRaBitQIndex, PQCodebook, RaBitQCodebook, RaBitQIndex,
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

/// Sparse vector index for a field: direct-indexed by dimension ID
#[derive(Clone)]
pub struct SparseIndex {
    /// Posting lists indexed directly by dimension ID (O(1) lookup)
    /// None means dimension not present in index
    pub postings: Vec<Option<Arc<BlockSparsePostingList>>>,
    /// Total document count in this segment (for IDF computation)
    pub total_docs: u32,
    /// Total sparse vectors in this segment (for multi-valued IDF)
    /// For single-valued fields, this equals total_docs.
    /// For multi-valued fields, this is the sum of all vectors across all docs.
    pub total_vectors: u32,
}

impl SparseIndex {
    /// Compute IDF (inverse document frequency) for a dimension
    ///
    /// For multi-valued fields, uses total_vectors instead of total_docs
    /// to properly handle cases where df can exceed total_docs.
    /// IDF = log(N / df), clamped to >= 0
    /// Returns 0.0 if dimension not present
    #[inline]
    pub fn idf(&self, dim_id: u32) -> f32 {
        if let Some(Some(pl)) = self.postings.get(dim_id as usize) {
            let df = pl.doc_count() as f32;
            if df > 0.0 {
                // Use total_vectors for proper IDF with multi-valued fields
                let n = self.total_vectors.max(self.total_docs) as f32;
                (n / df).ln().max(0.0)
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
