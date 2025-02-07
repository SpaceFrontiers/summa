//! IVF-RaBitQ: Inverted File Index with RaBitQ quantization
//!
//! Two-level index for vector search:
//! - Level 1: Coarse quantizer (k-means centroids)
//! - Level 2: RaBitQ binary codes per cluster
//!
//! Key feature: Segments sharing the same coarse centroids can be merged
//! in O(1) by concatenating cluster data - no re-quantization needed.

use std::io;

use serde::{Deserialize, Serialize};

use crate::structures::vector::ivf::{ClusterStorage, CoarseCentroids, MultiAssignment};
use crate::structures::vector::quantization::{QuantizedVector, RaBitQCodebook};

/// Configuration for IVF-RaBitQ index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFRaBitQConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of clusters to probe during search
    pub default_nprobe: usize,
    /// Store raw vectors for re-ranking
    pub store_raw: bool,
    /// Re-rank factor (multiply k by this to get candidates for re-ranking)
    pub rerank_factor: usize,
}

impl IVFRaBitQConfig {
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

/// IVF-RaBitQ index for a single segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFRaBitQIndex {
    /// Configuration
    pub config: IVFRaBitQConfig,
    /// Version of coarse centroids used (for merge compatibility)
    pub centroids_version: u64,
    /// Version of RaBitQ codebook used (for merge compatibility)
    pub codebook_version: u64,
    /// Cluster storage with RaBitQ codes
    pub clusters: ClusterStorage<QuantizedVector>,
}

impl IVFRaBitQIndex {
    /// Create a new empty IVF-RaBitQ index
    pub fn new(config: IVFRaBitQConfig, centroids_version: u64, codebook_version: u64) -> Self {
        Self {
            config,
            centroids_version,
            codebook_version,
            clusters: ClusterStorage::new(),
        }
    }

    /// Build index from vectors using provided coarse centroids and codebook
    ///
    /// `doc_id_ordinals`: (doc_id, ordinal) pairs. If None, uses (index, 0).
    pub fn build(
        config: IVFRaBitQConfig,
        coarse_centroids: &CoarseCentroids,
        codebook: &RaBitQCodebook,
        vectors: &[Vec<f32>],
        doc_id_ordinals: Option<&[(u32, u16)]>,
    ) -> Self {
        let mut index = Self::new(config.clone(), coarse_centroids.version, codebook.version);

        for (i, vector) in vectors.iter().enumerate() {
            let (doc_id, ordinal) = doc_id_ordinals.map(|ids| ids[i]).unwrap_or((i as u32, 0));
            index.add_vector(coarse_centroids, codebook, doc_id, ordinal, vector);
        }

        index
    }

    /// Add a single vector to the index
    pub fn add_vector(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        codebook: &RaBitQCodebook,
        doc_id: u32,
        ordinal: u16,
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
            ordinal,
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
                ordinal,
                vector,
                false, // Don't store raw for secondary
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_to_cluster(
        &mut self,
        coarse_centroids: &CoarseCentroids,
        codebook: &RaBitQCodebook,
        assignment: &MultiAssignment,
        doc_id: u32,
        ordinal: u16,
        vector: &[f32],
        store_raw: bool,
    ) {
        let cluster_id = assignment.primary_cluster;
        let centroid = coarse_centroids.get_centroid(cluster_id);

        // Quantize relative to cluster centroid
        let code = codebook.encode(vector, Some(centroid));

        // Store
        let raw = if store_raw && self.config.store_raw {
            Some(vector.to_vec())
        } else {
            None
        };

        self.clusters.add(cluster_id, doc_id, ordinal, code, raw);
    }

    /// Search for k nearest neighbors, returns (doc_id, ordinal, distance)
    pub fn search(
        &self,
        coarse_centroids: &CoarseCentroids,
        codebook: &RaBitQCodebook,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
    ) -> Vec<(u32, u16, f32)> {
        let nprobe = nprobe.unwrap_or(self.config.default_nprobe);

        // Find nprobe nearest coarse centroids
        let nearest_clusters = coarse_centroids.find_k_nearest(query, nprobe);

        let mut candidates: Vec<(u32, u16, f32)> = Vec::new();

        for &cluster_id in &nearest_clusters {
            if let Some(cluster) = self.clusters.get(cluster_id) {
                // Prepare query relative to cluster centroid
                let centroid = coarse_centroids.get_centroid(cluster_id);
                let prepared_query = codebook.prepare_query(query, Some(centroid));

                // Score all vectors in cluster
                for (doc_id, ordinal, code) in cluster.iter() {
                    let dist = codebook.estimate_distance(&prepared_query, code);
                    candidates.push((doc_id, ordinal, dist));
                }
            }
        }

        // Sort by distance
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Re-rank top candidates if raw vectors available
        let rerank_count = (k * self.config.rerank_factor).min(candidates.len());

        if rerank_count > 0 {
            // Check if we have raw vectors for re-ranking
            let has_raw = nearest_clusters.iter().any(|&c| {
                self.clusters
                    .get(c)
                    .map(|cl| cl.raw_vectors.is_some())
                    .unwrap_or(false)
            });

            if has_raw {
                let mut reranked: Vec<(u32, u16, f32)> = candidates[..rerank_count]
                    .iter()
                    .filter_map(|&(doc_id, ordinal, _)| {
                        // Find raw vector for this doc_id
                        for &cluster_id in &nearest_clusters {
                            if let Some(cluster) = self.clusters.get(cluster_id)
                                && let Some(ref raw_vecs) = cluster.raw_vectors
                            {
                                for (i, &did) in cluster.doc_ids.iter().enumerate() {
                                    if did == doc_id {
                                        let exact_dist =
                                            euclidean_distance_squared(query, &raw_vecs[i]);
                                        return Some((doc_id, ordinal, exact_dist));
                                    }
                                }
                            }
                        }
                        None
                    })
                    .collect();

                reranked.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                reranked.truncate(k);
                return reranked;
            }
        }

        candidates.truncate(k);
        candidates
    }

    /// Merge another index into this one (instance method)
    pub fn merge_into(
        &mut self,
        other: &IVFRaBitQIndex,
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

    /// Estimated memory usage in bytes
    pub fn estimated_memory_bytes(&self) -> usize {
        self.clusters.estimated_memory_bytes()
    }

    /// Number of non-empty clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.num_clusters()
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
    #[allow(clippy::should_implement_trait)]
    pub fn merge(indexes: &[&IVFRaBitQIndex], doc_offsets: &[u32]) -> Result<Self, &'static str> {
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
    use crate::structures::vector::quantization::RaBitQConfig;
    use rand::prelude::*;

    #[test]
    fn test_ivf_rabitq_basic() {
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

        // Create RaBitQ codebook
        let rabitq_config = RaBitQConfig::new(dim);
        let codebook = RaBitQCodebook::new(rabitq_config);

        // Build index
        let config = IVFRaBitQConfig::new(dim);
        let index = IVFRaBitQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        assert_eq!(index.len(), n);
    }

    #[test]
    fn test_ivf_rabitq_search() {
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

        let rabitq_config = RaBitQConfig::new(dim);
        let codebook = RaBitQCodebook::new(rabitq_config);

        let config = IVFRaBitQConfig::new(dim).with_nprobe(4);
        let index = IVFRaBitQIndex::build(config, &coarse_centroids, &codebook, &vectors, None);

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let results = index.search(&coarse_centroids, &codebook, &query, k, None);

        assert_eq!(results.len(), k);

        // Verify sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_ivf_rabitq_merge() {
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

        let rabitq_config = RaBitQConfig::new(dim);
        let codebook = RaBitQCodebook::new(rabitq_config);

        let config = IVFRaBitQConfig::new(dim);
        let mut index1 = IVFRaBitQIndex::build(
            config.clone(),
            &coarse_centroids,
            &codebook,
            &vectors1,
            None,
        );
        let index2 = IVFRaBitQIndex::build(config, &coarse_centroids, &codebook, &vectors2, None);

        index1.merge_into(&index2, n as u32).unwrap();

        assert_eq!(index1.len(), 2 * n);
    }
}
