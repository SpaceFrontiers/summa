//! Coarse centroids for IVF partitioning
//!
//! Provides k-means clustering for the first level of IVF indexing.
//! Trained once, shared across all segments for O(1) merge compatibility.

use serde::{Deserialize, Serialize};

use super::routing::{
    HNSW_AUTO_THRESHOLD, HnswRoutingGraph, IvfProbePlan, IvfRoutingTopology,
    allocate_child_clusters, effective_routing_mode, float_probe_fingerprint, parent_probe_count,
    routing_parent_count, select_best, select_best_candidates,
};
use super::soar::{MultiAssignment, SoarConfig};
use crate::dsl::IvfRoutingMode;

/// Configuration for coarse quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarseConfig {
    /// Number of clusters
    pub num_clusters: usize,
    /// Vector dimension
    pub dim: usize,
    /// Maximum k-means iterations
    pub max_iters: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// SOAR configuration (optional)
    pub soar: Option<SoarConfig>,
    /// Flat, two-level, or HNSW routing. Auto chooses from the final leaf count.
    pub routing: IvfRoutingMode,
}

impl CoarseConfig {
    pub fn new(dim: usize, num_clusters: usize) -> Self {
        Self {
            num_clusters,
            dim,
            max_iters: 25,
            seed: 42,
            soar: None,
            routing: IvfRoutingMode::Auto,
        }
    }

    pub fn with_soar(mut self, config: SoarConfig) -> Self {
        self.soar = Some(config);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_max_iters(mut self, iters: usize) -> Self {
        self.max_iters = iters;
        self
    }

    pub fn with_routing(mut self, routing: IvfRoutingMode) -> Self {
        self.routing = routing;
        self
    }
}

/// Coarse centroids for IVF - trained once, shared across all segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarseCentroids {
    /// Number of clusters
    pub num_clusters: u32,
    /// Vector dimension
    pub dim: usize,
    /// Centroids stored as flat array (num_clusters × dim)
    pub centroids: Vec<f32>,
    /// Version for compatibility checking during merge
    pub version: u64,
    /// SOAR configuration (if enabled)
    pub soar_config: Option<SoarConfig>,
    /// Persisted parent centroids and topology for sublinear leaf routing.
    pub(crate) routing_index: Option<FloatCentroidRouter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum FloatCentroidRouter {
    TwoLevel {
        parent_centroids: Vec<f32>,
        topology: IvfRoutingTopology,
    },
    Hnsw(HnswRoutingGraph),
}

impl CoarseCentroids {
    /// Train coarse centroids using k-means algorithm
    ///
    /// Uses deterministic k-means++ seeding and Lloyd refinement.
    pub fn train(config: &CoarseConfig, vectors: &[Vec<f32>]) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(config.num_clusters > 0, "Need at least 1 cluster");
        assert!(vectors.iter().all(|vector| vector.len() == config.dim));

        let actual_clusters = config.num_clusters.min(vectors.len());
        let (centroids, routing_index) =
            match effective_routing_mode(config.routing, actual_clusters) {
                IvfRoutingMode::TwoLevel => {
                    let (leaves, router) =
                        Self::train_hierarchical(config, vectors, actual_clusters);
                    (leaves, Some(router))
                }
                IvfRoutingMode::Hnsw => {
                    let leaves = if actual_clusters >= HNSW_AUTO_THRESHOLD {
                        Self::train_hierarchical(config, vectors, actual_clusters).0
                    } else {
                        Self::train_flat(config, vectors, actual_clusters).0
                    };
                    let graph = HnswRoutingGraph::build(
                        actual_clusters,
                        |left, right| {
                            let left = left as usize * config.dim;
                            let right = right as usize * config.dim;
                            squared_l2(
                                &leaves[left..left + config.dim],
                                &leaves[right..right + config.dim],
                            )
                        },
                        config.seed,
                    );
                    (leaves, Some(FloatCentroidRouter::Hnsw(graph)))
                }
                IvfRoutingMode::Flat | IvfRoutingMode::Auto => {
                    (Self::train_flat(config, vectors, actual_clusters).0, None)
                }
            };

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            num_clusters: actual_clusters as u32,
            dim: config.dim,
            centroids,
            version,
            soar_config: config.soar.clone(),
            routing_index,
        }
    }

    fn train_flat(
        config: &CoarseConfig,
        vectors: &[Vec<f32>],
        clusters: usize,
    ) -> (Vec<f32>, Vec<Vec<usize>>) {
        let dim = config.dim;
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let trained = crate::structures::vector::kmeans::train_euclidean_kmeans(
            &flat,
            vectors.len(),
            dim,
            clusters,
            config.max_iters,
            config.seed,
        );
        let mut groups = vec![Vec::new(); clusters];
        for (point, cluster) in trained.assignments.into_iter().enumerate() {
            groups[cluster].push(point);
        }
        (trained.centroids, groups)
    }

    fn train_hierarchical(
        config: &CoarseConfig,
        vectors: &[Vec<f32>],
        leaf_count: usize,
    ) -> (Vec<f32>, FloatCentroidRouter) {
        let parent_count = routing_parent_count(leaf_count).min(vectors.len());
        let mut parent_config = config.clone();
        parent_config.routing = IvfRoutingMode::Flat;
        parent_config.num_clusters = parent_count;
        let (parent_centroids, groups) = Self::train_flat(&parent_config, vectors, parent_count);
        let group_sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
        let child_counts = allocate_child_clusters(&group_sizes, leaf_count);
        let mut leaves = Vec::with_capacity(leaf_count.saturating_mul(config.dim));
        let mut children = vec![Vec::new(); parent_count];

        for (parent, (indices, &child_count)) in groups.iter().zip(&child_counts).enumerate() {
            if child_count == 0 {
                continue;
            }
            let group_vectors: Vec<Vec<f32>> = indices
                .iter()
                .map(|&index| vectors[index].clone())
                .collect();
            let mut child_config = config.clone();
            child_config.routing = IvfRoutingMode::Flat;
            child_config.num_clusters = child_count;
            child_config.seed = config.seed ^ (parent as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let first_leaf = leaves.len() / config.dim;
            leaves
                .extend_from_slice(&Self::train_flat(&child_config, &group_vectors, child_count).0);
            children[parent].extend((first_leaf..first_leaf + child_count).map(|leaf| leaf as u32));
        }
        debug_assert_eq!(leaves.len(), leaf_count * config.dim);

        (
            leaves,
            FloatCentroidRouter::TwoLevel {
                parent_centroids,
                topology: IvfRoutingTopology::from_children(&children),
            },
        )
    }

    /// Find nearest centroid index for a vector (static helper)
    fn find_nearest_idx_static(vector: &[f32], centroids: &[f32], dim: usize) -> usize {
        let num_clusters = centroids.len() / dim;
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for c in 0..num_clusters {
            let offset = c * dim;
            let dist: f32 = vector
                .iter()
                .zip(&centroids[offset..offset + dim])
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Find nearest cluster for a query vector
    pub fn find_nearest(&self, vector: &[f32]) -> u32 {
        Self::find_nearest_idx_static(vector, &self.centroids, self.dim) as u32
    }

    /// Find k nearest clusters for a query vector
    pub fn find_k_nearest(&self, vector: &[f32], k: usize) -> Vec<u32> {
        let mut distances: Vec<(u32, f32)> = (0..self.num_clusters)
            .map(|c| {
                let offset = c as usize * self.dim;
                let dist: f32 = vector
                    .iter()
                    .zip(&self.centroids[offset..offset + self.dim])
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (c, dist)
            })
            .collect();

        // Partial sort: O(n + k log k) instead of O(n log n)
        if distances.len() > k {
            distances.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
            distances.truncate(k);
        }
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        distances.into_iter().map(|(c, _)| c).collect()
    }

    /// Build a versioned probe plan using flat or two-level routing.
    ///
    /// The returned leaf IDs are independent of segment contents and can be
    /// reused across every segment built from this global codebook.
    pub fn probe(&self, vector: &[f32], k: usize, mode: IvfRoutingMode) -> IvfProbePlan {
        let take = k.clamp(1, self.num_clusters as usize);
        let clusters = match effective_routing_mode(mode, self.num_clusters as usize) {
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => self.find_k_nearest(vector, take),
            IvfRoutingMode::TwoLevel => self.find_k_nearest_two_level(vector, take),
            IvfRoutingMode::Hnsw => self.find_k_nearest_hnsw(vector, take),
        };
        IvfProbePlan::new(
            self.version,
            float_probe_fingerprint(vector, take, mode),
            clusters,
        )
    }

    pub fn validate_routing(&self, mode: IvfRoutingMode) -> Result<(), String> {
        match effective_routing_mode(mode, self.num_clusters as usize) {
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => Ok(()),
            IvfRoutingMode::TwoLevel => {
                let Some(FloatCentroidRouter::TwoLevel {
                    parent_centroids,
                    topology,
                }) = self.routing_index.as_ref()
                else {
                    return Err(
                        "two-level IVF routing was requested but the global codebook has no matching router"
                            .to_string(),
                    );
                };
                let parent_count = topology.parent_count();
                if parent_count == 0
                    || parent_centroids.len() != parent_count.saturating_mul(self.dim)
                    || !topology.validate(self.num_clusters as usize)
                    || parent_centroids.iter().any(|value| !value.is_finite())
                {
                    return Err("invalid float two-level IVF routing index".to_string());
                }
                Ok(())
            }
            IvfRoutingMode::Hnsw => {
                let Some(FloatCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
                    return Err(
                        "HNSW IVF routing was requested but the global codebook has no HNSW graph"
                            .to_string(),
                    );
                };
                if !graph.validate(self.num_clusters as usize) {
                    return Err("invalid float HNSW routing graph".to_string());
                }
                Ok(())
            }
        }
    }

    fn find_k_nearest_two_level(&self, vector: &[f32], k: usize) -> Vec<u32> {
        let Some(FloatCentroidRouter::TwoLevel {
            parent_centroids,
            topology,
        }) = self.routing_index.as_ref()
        else {
            return self.find_k_nearest(vector, k);
        };
        if topology.parent_count() <= 1 {
            return self.find_k_nearest(vector, k);
        }

        let mut parent_scores = vec![0.0; topology.parent_count()];
        for (parent_id, score) in parent_scores.iter_mut().enumerate() {
            let offset = parent_id * self.dim;
            *score = squared_l2(vector, &parent_centroids[offset..offset + self.dim]);
        }
        let parent_take =
            parent_probe_count(k, self.num_clusters as usize, topology.parent_count());
        let parents = select_best::<false>(&parent_scores, parent_take);
        let candidate_capacity = parents
            .iter()
            .map(|&parent| topology.children(parent as usize).len())
            .sum();
        let mut candidates = Vec::with_capacity(candidate_capacity);
        for parent in parents {
            for &leaf in topology.children(parent as usize) {
                candidates.push((leaf, squared_l2(vector, self.get_centroid(leaf))));
            }
        }
        select_best_candidates::<false>(&mut candidates, k)
    }

    fn find_k_nearest_hnsw(&self, vector: &[f32], k: usize) -> Vec<u32> {
        let Some(FloatCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_k_nearest(vector, k);
        };
        graph.search(|leaf| squared_l2(vector, self.get_centroid(leaf)), k)
    }

    fn find_nearest_hnsw(&self, vector: &[f32]) -> u32 {
        let Some(FloatCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_nearest(vector);
        };
        graph.search_one(|leaf| squared_l2(vector, self.get_centroid(leaf)))
    }

    /// Find k nearest clusters with their distances
    pub fn find_k_nearest_with_distances(&self, vector: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut distances: Vec<(u32, f32)> = (0..self.num_clusters)
            .map(|c| {
                let offset = c as usize * self.dim;
                let dist: f32 = vector
                    .iter()
                    .zip(&self.centroids[offset..offset + self.dim])
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (c, dist)
            })
            .collect();

        // Partial sort: O(n + k log k) instead of O(n log n)
        if distances.len() > k {
            distances.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
            distances.truncate(k);
        }
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        distances
    }

    /// Assign vector with SOAR (if configured) or standard assignment
    pub fn assign(&self, vector: &[f32]) -> MultiAssignment {
        self.assign_with_routing(vector, IvfRoutingMode::Flat)
    }

    /// Assign during segment construction through the same persisted router
    /// used at query time. Large codebooks therefore avoid an O(K) scan for
    /// every indexed vector.
    pub fn assign_with_routing(&self, vector: &[f32], routing: IvfRoutingMode) -> MultiAssignment {
        if let Some(ref soar_config) = self.soar_config {
            self.assign_with_soar_and_routing(vector, soar_config, routing)
        } else {
            let primary_cluster = match effective_routing_mode(routing, self.num_clusters as usize)
            {
                IvfRoutingMode::Hnsw => self.find_nearest_hnsw(vector),
                IvfRoutingMode::TwoLevel => self.find_k_nearest_two_level(vector, 1)[0],
                IvfRoutingMode::Flat | IvfRoutingMode::Auto => self.find_nearest(vector),
            };
            MultiAssignment {
                primary_cluster,
                secondary_clusters: Vec::new(),
            }
        }
    }

    /// SOAR-style assignment: find secondary clusters with orthogonal residuals
    pub fn assign_with_soar(&self, vector: &[f32], config: &SoarConfig) -> MultiAssignment {
        self.assign_with_soar_and_routing(vector, config, IvfRoutingMode::Flat)
    }

    fn assign_with_soar_and_routing(
        &self,
        vector: &[f32],
        config: &SoarConfig,
        routing: IvfRoutingMode,
    ) -> MultiAssignment {
        let leaf_ids: Vec<u32> = match effective_routing_mode(routing, self.num_clusters as usize) {
            IvfRoutingMode::TwoLevel => {
                self.two_level_candidate_leaves(vector, config.num_secondary + 1)
            }
            IvfRoutingMode::Hnsw => self.find_k_nearest_hnsw(
                vector,
                (config.num_secondary + 1)
                    .saturating_mul(16)
                    .max(32)
                    .min(self.num_clusters as usize),
            ),
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => (0..self.num_clusters).collect(),
        };
        let primary = leaf_ids
            .iter()
            .copied()
            .min_by(|&left, &right| {
                squared_l2(vector, self.get_centroid(left))
                    .total_cmp(&squared_l2(vector, self.get_centroid(right)))
                    .then_with(|| left.cmp(&right))
            })
            .unwrap_or(0);
        let primary_centroid = self.get_centroid(primary);

        // 2. Compute primary residual r = x - c
        let residual: Vec<f32> = vector
            .iter()
            .zip(primary_centroid)
            .map(|(v, c)| v - c)
            .collect();

        let residual_norm_sq: f32 = residual.iter().map(|x| x * x).sum();

        // 3. Check if we should spill (selective spilling)
        if config.selective && residual_norm_sq < config.spill_threshold * config.spill_threshold {
            return MultiAssignment {
                primary_cluster: primary,
                secondary_clusters: Vec::new(),
            };
        }

        // 4. Find secondary clusters that MINIMIZE |⟨r, r'⟩| (orthogonal residuals)
        let mut candidates: Vec<(u32, f32)> = leaf_ids
            .into_iter()
            .filter(|&c| c != primary)
            .map(|c| {
                let centroid = self.get_centroid(c);
                // Compute r' = x - c'
                // Then compute |⟨r, r'⟩| - we want this SMALL (orthogonal)
                let dot: f32 = vector
                    .iter()
                    .zip(centroid)
                    .zip(&residual)
                    .map(|((v, c), r)| (v - c) * r)
                    .sum();
                (c, dot.abs())
            })
            .collect();

        // Partial sort by orthogonality (smallest dot product first)
        let take = config.num_secondary.min(candidates.len());
        if candidates.len() > take {
            candidates.select_nth_unstable_by(take, |a, b| a.1.total_cmp(&b.1));
            candidates.truncate(take);
        }

        MultiAssignment {
            primary_cluster: primary,
            secondary_clusters: candidates
                .iter()
                .take(config.num_secondary)
                .map(|(c, _)| *c)
                .collect(),
        }
    }

    fn two_level_candidate_leaves(&self, vector: &[f32], k: usize) -> Vec<u32> {
        let Some(FloatCentroidRouter::TwoLevel {
            parent_centroids,
            topology,
        }) = self.routing_index.as_ref()
        else {
            return (0..self.num_clusters).collect();
        };
        let mut parent_scores = vec![0.0; topology.parent_count()];
        for (parent_id, score) in parent_scores.iter_mut().enumerate() {
            let offset = parent_id * self.dim;
            *score = squared_l2(vector, &parent_centroids[offset..offset + self.dim]);
        }
        let parents = select_best::<false>(
            &parent_scores,
            parent_probe_count(k, self.num_clusters as usize, topology.parent_count()),
        );
        let capacity = parents
            .iter()
            .map(|&parent| topology.children(parent as usize).len())
            .sum();
        let mut leaves = Vec::with_capacity(capacity);
        for parent in parents {
            leaves.extend_from_slice(topology.children(parent as usize));
        }
        leaves
    }

    /// Get centroid for a cluster
    pub fn get_centroid(&self, cluster_id: u32) -> &[f32] {
        let offset = cluster_id as usize * self.dim;
        &self.centroids[offset..offset + self.dim]
    }

    /// Compute residual vector (vector - centroid)
    pub fn compute_residual(&self, vector: &[f32], cluster_id: u32) -> Vec<f32> {
        let centroid = self.get_centroid(cluster_id);
        vector.iter().zip(centroid).map(|(&v, &c)| v - c).collect()
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        let routing_bytes = self
            .routing_index
            .as_ref()
            .map_or(0, |router| match router {
                FloatCentroidRouter::TwoLevel {
                    parent_centroids,
                    topology,
                } => {
                    parent_centroids.len() * size_of::<f32>()
                        + topology.parent_count() * size_of::<u32>()
                        + self.num_clusters as usize * size_of::<u32>()
                }
                FloatCentroidRouter::Hnsw(graph) => graph.size_bytes(),
            });
        self.centroids.len() * size_of::<f32>() + routing_bytes + 64
    }

    /// Visit compact routing topology and parent arrays before the potentially
    /// much larger leaf centroid matrix.
    pub(crate) fn visit_routing_regions(&self, visit: &mut dyn FnMut(&'static str, &[u8])) {
        if let Some(router) = &self.routing_index {
            match router {
                FloatCentroidRouter::TwoLevel {
                    parent_centroids,
                    topology,
                } => {
                    topology.visit_resident_regions(visit);
                    visit(
                        "float parent centroids",
                        super::routing::bytes_of_slice(parent_centroids),
                    );
                }
                FloatCentroidRouter::Hnsw(graph) => graph.visit_resident_regions(visit),
            }
        }
    }

    pub(crate) fn visit_leaf_centroid_region(&self, visit: &mut dyn FnMut(&'static str, &[u8])) {
        visit(
            "float leaf centroids",
            super::routing::bytes_of_slice(&self.centroids),
        );
    }

    /// Encode the current index-level centroid artifact format.
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }
}

#[inline]
fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_coarse_centroids_basic() {
        let dim = 64;
        let n = 1000;
        let num_clusters = 16;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let config = CoarseConfig::new(dim, num_clusters);
        let centroids = CoarseCentroids::train(&config, &vectors);

        assert_eq!(centroids.num_clusters, num_clusters as u32);
        assert_eq!(centroids.dim, dim);
    }

    #[test]
    fn test_find_nearest() {
        let dim = 32;
        let n = 500;
        let num_clusters = 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let config = CoarseConfig::new(dim, num_clusters);
        let centroids = CoarseCentroids::train(&config, &vectors);

        // Test that find_nearest returns valid cluster IDs
        for v in &vectors {
            let cluster = centroids.find_nearest(v);
            assert!(cluster < centroids.num_clusters);
        }
    }

    #[test]
    fn test_soar_assignment() {
        let dim = 32;
        let n = 100;
        let num_clusters = 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let soar_config = SoarConfig {
            num_secondary: 2,
            selective: false,
            spill_threshold: 0.0,
        };
        let config = CoarseConfig::new(dim, num_clusters).with_soar(soar_config);
        let centroids = CoarseCentroids::train(&config, &vectors);

        // Test SOAR assignment
        let assignment = centroids.assign(&vectors[0]);
        assert!(assignment.primary_cluster < centroids.num_clusters);
        assert_eq!(assignment.secondary_clusters.len(), 2);

        // Secondary clusters should be different from primary
        for &sec in &assignment.secondary_clusters {
            assert_ne!(sec, assignment.primary_cluster);
        }
    }

    #[test]
    fn test_serialization() {
        let dim = 16;
        let n = 50;
        let num_clusters = 4;

        let mut rng = rand::rngs::StdRng::seed_from_u64(789);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        let config = CoarseConfig::new(dim, num_clusters);
        let centroids = CoarseCentroids::train(&config, &vectors);

        // Serialize and deserialize
        let bytes = bincode::serde::encode_to_vec(&centroids, bincode::config::standard()).unwrap();
        let (loaded, consumed): (CoarseCentroids, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());

        assert_eq!(loaded.num_clusters, centroids.num_clusters);
        assert_eq!(loaded.dim, centroids.dim);
        assert_eq!(loaded.centroids.len(), centroids.centroids.len());
    }

    #[test]
    fn persisted_hnsw_and_two_level_routers_are_valid() {
        let dim = 4;
        let mut rng = rand::rngs::StdRng::seed_from_u64(991);
        let vectors: Vec<Vec<f32>> = (0..256)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        for routing in [IvfRoutingMode::Hnsw, IvfRoutingMode::TwoLevel] {
            let trained =
                CoarseCentroids::train(&CoarseConfig::new(dim, 16).with_routing(routing), &vectors);
            trained.validate_routing(routing).unwrap();
            let plan = trained.probe(&vectors[0], 8, routing);
            assert_eq!(plan.cluster_ids.len(), 8);
            assert!(
                plan.cluster_ids
                    .iter()
                    .all(|&cluster| cluster < trained.num_clusters)
            );

            let bytes =
                bincode::serde::encode_to_vec(&trained, bincode::config::standard()).unwrap();
            let (loaded, consumed): (CoarseCentroids, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(consumed, bytes.len());
            loaded.validate_routing(routing).unwrap();
        }
    }
}
