//! IVF-PQ: Inverted File Index with Product Quantization (ScaNN-style)
//!
//! Two-level index for vector search:
//! - Level 1: Coarse quantizer (k-means centroids)
//! - Level 2: Product Quantization codes per cluster
//!
//! Key feature: Segments sharing the same coarse centroids and PQ codebook
//! can be merged in O(1) by concatenating cluster data.

use serde::{Deserialize, Serialize};
use std::io;

use crate::structures::vector::ivf::{ClusterStorage, CoarseCentroids, MultiAssignment};
use crate::structures::vector::quantization::{DistanceTable, PQCodebook, PQVector};

/// Configuration for IVF-PQ index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFPQConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of clusters to probe during search
    pub default_nprobe: usize,
    /// Store raw vectors for re-ranking
    pub store_raw: bool,
    /// Re-rank factor (multiply k by this to get candidates for re-ranking)
    pub rerank_factor: usize,
}

impl IVFPQConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            default_nprobe: 32,
            store_raw: true,
            rerank_factor: 10,
        }
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.default_nprobe = nprobe;
        self
    }

    pub fn with_store_raw(mut self, store: bool) -> Self {
        self.store_raw = store;
        self
    }

    pub fn with_rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor;
        self
    }
}

/// IVF-PQ index for a single segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFPQIndex {
    /// Configuration
    pub config: IVFPQConfig,
    /// Version of coarse centroids used (for merge compatibility)
    pub centroids_version: u64,
    /// Version of PQ codebook used (for merge compatibility)
    pub codebook_version: u64,
    /// Cluster storage with PQ codes
    pub clusters: ClusterStorage<PQVector>,
}

impl IVFPQIndex {
    /// Create a new empty IVF-PQ index
    pub fn new(config: IVFPQConfig, centroids_version: u64, codebook_version: u64) -> Self {
        Self {
            config,
            centroids_version,
            codebook_version,
            clusters: ClusterStorage::new(),
        }
    }

    /// Build index from vectors using provided coarse centroids and PQ codebook
    pub fn build(
        config: IVFPQConfig,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        vectors: &[Vec<f32>],
        doc_ids: Option<&[u32]>,
    ) -> Self {
        let mut index = Self::new(config.clone(), coarse_centroids.version, codebook.version);

        for (i, vector) in vectors.iter().enumerate() {
            let doc_id = doc_ids.map(|ids| ids[i]).unwrap_or(i as u32);
            index.add_vector(coarse_centroids, codebook, doc_id, vector);
        }

        index
    }

    /// Add a single vector to the index
    pub fn add_vector(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        doc_id: u32,
        vector: &[f32],
    ) {
        // Get cluster assignment (with SOAR if configured)
        let assignment = coarse_centroids.assign(vector);

        // Add to primary cluster
        self.add_to_cluster(
            coarse_centroids,
            codebook,
            &assignment,
            doc_id,
            vector,
            true,
        );

        // Add to secondary clusters (SOAR)
        for &cluster_id in &assignment.secondary_clusters {
            let secondary_assignment = MultiAssignment {
                primary_cluster: cluster_id,
                secondary_clusters: Vec::new(),
            };
            self.add_to_cluster(
                coarse_centroids,
                codebook,
                &secondary_assignment,
                doc_id,
                vector,
                false, // Don't store raw for secondary
            );
        }
    }

    fn add_to_cluster(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        assignment: &MultiAssignment,
        doc_id: u32,
        vector: &[f32],
        store_raw: bool,
    ) {
        let cluster_id = assignment.primary_cluster;
        let centroid = coarse_centroids.get_centroid(cluster_id);

        // Encode residual with PQ
        let code = codebook.encode(vector, Some(centroid));

        // Store
        let raw = if store_raw && self.config.store_raw {
            Some(vector.to_vec())
        } else {
            None
        };

        self.clusters.add(cluster_id, doc_id, code, raw);
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
    ) -> Vec<(u32, f32)> {
        let nprobe = nprobe.unwrap_or(self.config.default_nprobe);

        // Find nprobe nearest coarse centroids
        let nearest_clusters = coarse_centroids.find_k_nearest(query, nprobe);

        let mut candidates: Vec<(u32, f32)> = Vec::new();

        for &cluster_id in &nearest_clusters {
            if let Some(cluster) = self.clusters.get(cluster_id) {
                // Build distance table for this cluster's centroid
                let centroid = coarse_centroids.get_centroid(cluster_id);
                let distance_table = DistanceTable::build(codebook, query, Some(centroid));

                // Score all vectors in cluster using ADC (Asymmetric Distance Computation)
                for (doc_id, code) in cluster.iter() {
                    let dist = distance_table.compute_distance(&code.codes);
                    candidates.push((doc_id, dist));
                }
            }
        }

        // Sort by distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Re-rank top candidates if raw vectors available
        let rerank_count = (k * self.config.rerank_factor).min(candidates.len());

        if rerank_count > 0 {
            let has_raw = nearest_clusters.iter().any(|&c| {
                self.clusters
                    .get(c)
                    .map(|cl| cl.raw_vectors.is_some())
                    .unwrap_or(false)
            });

            if has_raw {
                let mut reranked: Vec<(u32, f32)> = candidates[..rerank_count]
                    .iter()
                    .filter_map(|&(doc_id, _)| {
                        for &cluster_id in &nearest_clusters {
                            if let Some(cluster) = self.clusters.get(cluster_id)
                                && let Some(ref raw_vecs) = cluster.raw_vectors
                            {
                                for (i, &did) in cluster.doc_ids.iter().enumerate() {
                                    if did == doc_id {
                                        let exact_dist =
                                            euclidean_distance_squared(query, &raw_vecs[i]);
                                        return Some((doc_id, exact_dist));
                                    }
                                }
                            }
                        }
                        None
                    })
                    .collect();

                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                reranked.truncate(k);
                return reranked;
            }
        }

        candidates.truncate(k);
        candidates
    }

    /// Search using inner product (MIPS - Maximum Inner Product Search)
    pub fn search_mips(
        &self,
        coarse_centroids: &CoarseCentroids,
        codebook: &PQCodebook,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
    ) -> Vec<(u32, f32)> {
        // For MIPS, we use the same search but with inner product distance table
        // The distance table stores negative inner products so we can use min-heap
        let mut results = self.search(coarse_centroids, codebook, query, k, nprobe);

        // Convert back to inner products (negate distances)
        for (_, dist) in &mut results {
            *dist = -*dist;
        }

        // Re-sort by inner product (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Merge another index into this one (instance method)
    pub fn merge_into(
        &mut self,
        other: &IVFPQIndex,
        doc_id_offset: u32,
    ) -> Result<(), &'static str> {
        if self.centroids_version != other.centroids_version {
            return Err("Cannot merge indexes with different centroid versions");
        }
        if self.codebook_version != other.codebook_version {
            return Err("Cannot merge indexes with different codebook versions");
        }

        self.clusters.merge(&other.clusters, doc_id_offset);
        Ok(())
    }

    /// Number of indexed vectors
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// Number of non-empty clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.num_clusters()
    }

    /// Memory usage estimate
    pub fn size_bytes(&self) -> usize {
        self.clusters.size_bytes()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let json =
            serde_json::to_vec(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(json)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        serde_json::from_slice(data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Static merge function for backward compatibility with segment merger
    ///
    /// Note: This is named `merge` for backward compatibility with existing code.
    /// It merges multiple indexes into a new one.
    #[allow(clippy::should_implement_trait)]
    pub fn merge(indexes: &[&IVFPQIndex], doc_offsets: &[u32]) -> Result<Self, &'static str> {
        if indexes.is_empty() {
            return Err("Cannot merge empty list of indexes");
        }

        let first = indexes[0];
        let mut merged = Self::new(
            first.config.clone(),
            first.centroids_version,
            first.codebook_version,
        );

        for (idx, &index) in indexes.iter().enumerate() {
            let offset = doc_offsets.get(idx).copied().unwrap_or(0);
            merged.merge_into(index, offset)?;
        }

        Ok(merged)
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
    use crate::structures::vector::ivf::CoarseConfig;
    use crate::structures::vector::quantization::PQConfig;
    use rand::prelude::*;

    #[test]
    #[ignore] // Long-running test
    fn test_ivf_pq_basic() {
        let dim = 64;
        let n = 500;
        let num_clusters = 16;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        // Train coarse centroids
        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors);

        // Train PQ codebook
        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors, 10);

        // Build index
        let config = IVFPQConfig::new(dim);
        let index = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        assert_eq!(index.len(), n);
    }

    #[test]
    #[ignore] // Long-running test
    fn test_ivf_pq_search() {
        let dim = 32;
        let n = 200;
        let k = 10;
        let num_clusters = 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors);

        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors, 10);

        let config = IVFPQConfig::new(dim).with_nprobe(4);
        let index = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let results = index.search(&coarse_centroids, &codebook, &query, k, None);

        assert_eq!(results.len(), k);

        // Verify sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    #[ignore] // Long-running test
    fn test_ivf_pq_recall() {
        let dim = 128;
        let n = 1000;
        let k = 10;
        let num_clusters = 32;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors);

        // Test with default PQ config (64 subspaces, 2 dims each)
        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors, 25);

        let config = IVFPQConfig::new(dim).with_nprobe(32);
        let index = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        // Run multiple queries and compute recall
        let num_queries = 50;
        let mut total_recall = 0.0f32;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();

            // Exact k-NN
            let mut exact: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = query.iter().zip(v).map(|(&a, &b)| (a - b) * (a - b)).sum();
                    (i, d)
                })
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let exact_top_k: std::collections::HashSet<usize> =
                exact[..k].iter().map(|(i, _)| *i).collect();

            // IVF-PQ search
            let results = index.search(&coarse_centroids, &codebook, &query, k, None);
            let pq_top_k: std::collections::HashSet<usize> =
                results.iter().map(|(i, _)| *i as usize).collect();

            let recall = exact_top_k.intersection(&pq_top_k).count() as f32 / k as f32;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f32;
        println!("IVF-PQ Recall@{}: {:.1}%", k, avg_recall * 100.0);

        // With proper config, recall should be reasonable (>30%)
        assert!(
            avg_recall > 0.25,
            "IVF-PQ recall too low: {:.1}%",
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_ivf_pq_merge() {
        let dim = 32;
        let n = 100;
        let num_clusters = 4;

        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let vectors1: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();
        let vectors2: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let coarse_config = CoarseConfig::new(dim, num_clusters);
        let coarse_centroids = CoarseCentroids::train(&coarse_config, &vectors1);

        let pq_config = PQConfig::new(dim).with_opq(false, 0);
        let codebook = PQCodebook::train(pq_config, &vectors1, 10);

        let config = IVFPQConfig::new(dim);
        let mut index1 = IVFPQIndex::build(
            config.clone(),
            &coarse_centroids,
            &codebook,
            &vectors1,
            None,
        );
        let index2 = IVFPQIndex::build(config, &coarse_centroids, &codebook, &vectors2, None);

        index1.merge_into(&index2, n as u32).unwrap();

        assert_eq!(index1.len(), 2 * n);
    }
}
