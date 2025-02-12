//! Standalone RaBitQ index (without IVF)
//!
//! For small datasets where IVF overhead isn't worth it.
//! Uses brute-force search over all quantized vectors.

use std::io;

use serde::{Deserialize, Serialize};

use crate::structures::vector::ivf::QuantizedCode;
use crate::structures::vector::quantization::{
    QuantizedQuery, QuantizedVector, RaBitQCodebook, RaBitQConfig,
};

/// Standalone RaBitQ index for small datasets
///
/// Uses brute-force search over all quantized vectors.
/// For larger datasets, use `IVFRaBitQIndex` instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQIndex {
    /// RaBitQ codebook (random transform parameters)
    pub codebook: RaBitQCodebook,
    /// Centroid of all indexed vectors
    pub centroid: Vec<f32>,
    /// Document IDs
    pub doc_ids: Vec<u32>,
    /// Element ordinals for multi-valued fields (0 for single-valued)
    pub ordinals: Vec<u16>,
    /// Quantized vectors
    pub vectors: Vec<QuantizedVector>,
}

impl RaBitQIndex {
    /// Create a new empty RaBitQ index
    pub fn new(config: RaBitQConfig) -> Self {
        let dim = config.dim;
        let codebook = RaBitQCodebook::new(config);

        Self {
            codebook,
            centroid: vec![0.0; dim],
            doc_ids: Vec::new(),
            ordinals: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Build index from vectors with doc IDs and ordinals
    pub fn build_with_ids(
        config: RaBitQConfig,
        vectors: &[(u32, u16, Vec<f32>)], // (doc_id, ordinal, vector)
    ) -> Self {
        let n = vectors.len();
        let dim = config.dim;

        assert!(n > 0, "Cannot build index from empty vector set");
        assert!(vectors[0].2.len() == dim, "Vector dimension mismatch");

        let mut index = Self::new(config);

        // Compute centroid
        index.centroid = vec![0.0; dim];
        for (_, _, v) in vectors {
            for (i, &val) in v.iter().enumerate() {
                index.centroid[i] += val;
            }
        }
        for c in &mut index.centroid {
            *c /= n as f32;
        }

        // Store doc_ids, ordinals and quantize vectors
        index.doc_ids = vectors.iter().map(|(doc_id, _, _)| *doc_id).collect();
        index.ordinals = vectors.iter().map(|(_, ordinal, _)| *ordinal).collect();
        index.vectors = vectors
            .iter()
            .map(|(_, _, v)| index.codebook.encode(v, Some(&index.centroid)))
            .collect();

        index
    }

    /// Build index from a set of vectors (legacy, uses doc_id = index, ordinal = 0)
    pub fn build(config: RaBitQConfig, vectors: &[Vec<f32>]) -> Self {
        let with_ids: Vec<(u32, u16, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, 0u16, v.clone()))
            .collect();
        Self::build_with_ids(config, &with_ids)
    }

    /// Add a single vector to the index
    pub fn add_vector(&mut self, doc_id: u32, ordinal: u16, vector: &[f32]) {
        self.doc_ids.push(doc_id);
        self.ordinals.push(ordinal);
        self.vectors
            .push(self.codebook.encode(vector, Some(&self.centroid)));
    }

    /// Prepare a query for fast distance estimation
    pub fn prepare_query(&self, query: &[f32]) -> QuantizedQuery {
        self.codebook.prepare_query(query, Some(&self.centroid))
    }

    /// Estimate squared distance between query and a quantized vector
    pub fn estimate_distance(&self, query: &QuantizedQuery, vec_idx: usize) -> f32 {
        self.codebook
            .estimate_distance(query, &self.vectors[vec_idx])
    }

    /// Search for k nearest neighbors, returns (doc_id, ordinal, distance)
    pub fn search(&self, query: &[f32], k: usize, _fetch_k: usize) -> Vec<(u32, u16, f32)> {
        let prepared = self.prepare_query(query);

        // Phase 1: Estimate distances for all vectors
        let mut candidates: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, _)| (i, self.estimate_distance(&prepared, i)))
            .collect();

        // Sort by estimated distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates.truncate(k);

        // Map indices to (doc_id, ordinal, dist)
        candidates
            .into_iter()
            .map(|(idx, dist)| (self.doc_ids[idx], self.ordinals[idx], dist))
            .collect()
    }

    /// Number of indexed vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        use std::mem::size_of;

        let vectors_size: usize = self.vectors.iter().map(|v| v.size_bytes()).sum();
        let centroid_size = self.centroid.len() * size_of::<f32>();
        let doc_ids_size = self.doc_ids.len() * size_of::<u32>();
        let ordinals_size = self.ordinals.len() * size_of::<u16>();
        let codebook_size = self.codebook.size_bytes();
        vectors_size + centroid_size + doc_ids_size + ordinals_size + codebook_size
    }

    /// Estimated memory usage in bytes (alias for size_bytes)
    pub fn estimated_memory_bytes(&self) -> usize {
        self.size_bytes()
    }

    /// Compression ratio compared to raw float32 vectors
    pub fn compression_ratio(&self) -> f32 {
        if self.vectors.is_empty() {
            return 1.0;
        }

        let dim = self.codebook.config.dim;
        let raw_size = self.vectors.len() * dim * 4;
        let compressed_size: usize = self.vectors.iter().map(|v| v.size_bytes()).sum();

        raw_size as f32 / compressed_size as f32
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        serde_json::from_slice(data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_rabitq_basic() {
        let dim = 128;
        let n = 100;

        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let config = RaBitQConfig::new(dim);
        let index = RaBitQIndex::build(config, &vectors);

        assert_eq!(index.len(), n);
        println!("Compression ratio: {:.1}x", index.compression_ratio());
    }

    #[test]
    fn test_rabitq_search() {
        let dim = 64;
        let n = 1000;
        let k = 10;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let config = RaBitQConfig::new(dim);
        let index = RaBitQIndex::build(config, &vectors);

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let results = index.search(&query, k, 10);

        assert_eq!(results.len(), k);

        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }
}
