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
    /// Quantized vectors
    pub vectors: Vec<QuantizedVector>,
    /// Raw vectors for re-ranking (optional)
    pub raw_vectors: Option<Vec<Vec<f32>>>,
}

impl RaBitQIndex {
    /// Create a new empty RaBitQ index
    pub fn new(config: RaBitQConfig) -> Self {
        let dim = config.dim;
        let codebook = RaBitQCodebook::new(config);

        Self {
            codebook,
            centroid: vec![0.0; dim],
            vectors: Vec::new(),
            raw_vectors: None,
        }
    }

    /// Build index from a set of vectors
    pub fn build(config: RaBitQConfig, vectors: &[Vec<f32>], store_raw: bool) -> Self {
        let n = vectors.len();
        let dim = config.dim;

        assert!(n > 0, "Cannot build index from empty vector set");
        assert!(vectors[0].len() == dim, "Vector dimension mismatch");

        let mut index = Self::new(config);

        // Compute centroid
        index.centroid = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                index.centroid[i] += val;
            }
        }
        for c in &mut index.centroid {
            *c /= n as f32;
        }

        // Quantize each vector relative to centroid
        index.vectors = vectors
            .iter()
            .map(|v| index.codebook.encode(v, Some(&index.centroid)))
            .collect();

        if store_raw {
            index.raw_vectors = Some(vectors.to_vec());
        }

        index
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

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, rerank_factor: usize) -> Vec<(usize, f32)> {
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

        // Phase 2: Re-rank top candidates with exact distances
        let rerank_count = (k * rerank_factor).min(candidates.len());

        if let Some(ref raw_vectors) = self.raw_vectors {
            let mut reranked: Vec<(usize, f32)> = candidates[..rerank_count]
                .iter()
                .map(|&(idx, _)| {
                    let exact_dist = euclidean_distance_squared(query, &raw_vectors[idx]);
                    (idx, exact_dist)
                })
                .collect();

            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            reranked.truncate(k);
            reranked
        } else {
            candidates.truncate(k);
            candidates
        }
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
        let vectors_size: usize = self.vectors.iter().map(|v| v.size_bytes()).sum();
        let centroid_size = self.centroid.len() * 4;
        let codebook_size = self.codebook.size_bytes();
        let raw_size = self
            .raw_vectors
            .as_ref()
            .map(|vecs| vecs.iter().map(|v| v.len() * 4).sum())
            .unwrap_or(0);

        vectors_size + centroid_size + codebook_size + raw_size
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

/// Compute squared Euclidean distance
#[inline]
fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
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
        let index = RaBitQIndex::build(config, &vectors, true);

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
        let index = RaBitQIndex::build(config, &vectors, true);

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let results = index.search(&query, k, 10);

        assert_eq!(results.len(), k);

        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }
}
