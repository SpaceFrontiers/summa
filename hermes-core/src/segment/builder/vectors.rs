//! Vector builders for dense and sparse vectors

use rustc_hash::FxHashMap;

use crate::DocId;

/// Builder for dense vector index
///
/// Collects vectors with ordinal tracking for multi-valued fields.
pub(super) struct DenseVectorBuilder {
    /// Dimension of vectors
    pub dim: usize,
    /// Document IDs with ordinals: (doc_id, ordinal)
    pub doc_ids: Vec<(DocId, u16)>,
    /// Flat vector storage (doc_ids.len() * dim floats)
    pub vectors: Vec<f32>,
}

impl DenseVectorBuilder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            doc_ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    pub fn add(&mut self, doc_id: DocId, ordinal: u16, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");
        self.doc_ids.push((doc_id, ordinal));
        self.vectors.extend_from_slice(vector);
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Get all vectors as Vec<Vec<f32>> for RaBitQ indexing
    pub fn get_vectors(&self) -> Vec<Vec<f32>> {
        self.doc_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = i * self.dim;
                self.vectors[start..start + self.dim].to_vec()
            })
            .collect()
    }

    /// Get vectors trimmed to specified dimension for matryoshka/MRL indexing
    pub fn get_vectors_trimmed(&self, trim_dim: usize) -> Vec<Vec<f32>> {
        debug_assert!(trim_dim <= self.dim, "trim_dim must be <= dim");
        self.doc_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = i * self.dim;
                self.vectors[start..start + trim_dim].to_vec()
            })
            .collect()
    }
}

/// Builder for sparse vector index using BlockSparsePostingList
///
/// Collects (doc_id, ordinal, weight) postings per dimension, then builds
/// BlockSparsePostingList with proper quantization during commit.
pub(super) struct SparseVectorBuilder {
    /// Postings per dimension: dim_id -> Vec<(doc_id, ordinal, weight)>
    pub postings: FxHashMap<u32, Vec<(DocId, u16, f32)>>,
}

impl SparseVectorBuilder {
    pub fn new() -> Self {
        Self {
            postings: FxHashMap::default(),
        }
    }

    /// Add a sparse vector entry with ordinal tracking
    #[inline]
    pub fn add(&mut self, dim_id: u32, doc_id: DocId, ordinal: u16, weight: f32) {
        self.postings
            .entry(dim_id)
            .or_default()
            .push((doc_id, ordinal, weight));
    }

    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
}
