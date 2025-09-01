//! Coarse centroids for IVF partitioning
//!
//! Provides k-means clustering for the first level of IVF indexing.
//! Trained once, shared across all segments for O(1) merge compatibility.

use serde::{Deserialize, Serialize};

use super::routing::{
    HIERARCHICAL_TRAINING_THRESHOLD, HnswRoutingGraph, IvfProbePlan, IvfRoutingTopology,
    allocate_child_clusters, effective_routing_mode, float_probe_fingerprint, routing_parent_count,
    select_best_candidates, select_parent_beam, select_parent_beam_for_build,
};
use super::soar::{MultiAssignment, SoarConfig};
use crate::dsl::IvfRoutingMode;

// The SOAR paper evaluates lambda = 1. Keep it private until a different
// value has recall/latency evidence and can be added without changing the
// serialized public configuration.
const SOAR_LAMBDA: f32 = 1.0;
/// Selective spilling is intended for boundary vectors, whose primary and
/// secondary residuals have comparable distortion. Encoding a residual to a
/// much farther secondary centroid both wastes a posting and amplifies TQ
/// estimation error enough to crowd better exact-rerank candidates out.
///
/// Compare squared distances, so `4` permits a secondary residual up to twice
/// the primary residual norm. Full spilling remains unconditional.
const MAX_SELECTIVE_SECONDARY_TO_PRIMARY_DISTANCE_RATIO_SQ: f32 = 4.0;
/// Construction is offline, so expand more of a hierarchical router than a
/// latency-sensitive one-leaf query. This reduces permanent misassignment
/// without returning to an O(K) centroid scan for large codebooks.
const BUILD_ASSIGNMENT_CANDIDATES: usize = 128;

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

thread_local! {
    /// Flat probing scores every leaf centroid; at production cluster counts
    /// that buffer is hundreds of KiB per query, so it is retained per thread
    /// (mirrors `binary_ivf::CENTROID_SCORE_SCRATCH`).
    static CENTROID_SCORE_SCRATCH: std::cell::RefCell<Vec<(u32, f32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct FlatClusterMemberships {
    offsets: Vec<usize>,
    members: Vec<usize>,
}

impl FlatClusterMemberships {
    fn cluster(&self, cluster: usize) -> &[usize] {
        &self.members[self.offsets[cluster]..self.offsets[cluster + 1]]
    }
}

impl CoarseCentroids {
    /// Train coarse centroids using k-means algorithm
    ///
    /// Uses deterministic adaptive D² seeding and Lloyd refinement.
    pub fn train(config: &CoarseConfig, vectors: &[Vec<f32>], index_label: &str) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(config.num_clusters > 0, "Need at least 1 cluster");
        assert!(vectors.iter().all(|vector| vector.len() == config.dim));

        // Keep the public row-oriented API for callers and tests, but funnel
        // production training through one contiguous matrix so the build path
        // does not allocate one heap object per sampled vector.
        let flat = vectors
            .iter()
            .flat_map(|vector| vector.iter().copied())
            .collect::<Vec<_>>();
        Self::train_contiguous(config, &flat, vectors.len(), index_label)
    }

    /// Train directly from a contiguous row-major matrix.
    pub(crate) fn train_contiguous(
        config: &CoarseConfig,
        vectors: &[f32],
        vector_count: usize,
        index_label: &str,
    ) -> Self {
        assert!(vector_count > 0, "Cannot train on empty vector set");
        assert!(config.num_clusters > 0, "Need at least 1 cluster");
        assert_eq!(vectors.len(), vector_count.saturating_mul(config.dim));

        let actual_clusters = config.num_clusters.min(vector_count);
        let (centroids, routing_index) =
            match effective_routing_mode(config.routing, actual_clusters) {
                IvfRoutingMode::TwoLevel => {
                    let (leaves, router) =
                        Self::train_hierarchical(config, vectors, vector_count, actual_clusters);
                    (leaves, Some(router))
                }
                IvfRoutingMode::Hnsw => {
                    let leaves = if actual_clusters >= HIERARCHICAL_TRAINING_THRESHOLD {
                        Self::train_hierarchical(config, vectors, vector_count, actual_clusters).0
                    } else {
                        Self::train_flat(config, vectors, vector_count, actual_clusters)
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
                        index_label,
                    );
                    (leaves, Some(FloatCentroidRouter::Hnsw(graph)))
                }
                IvfRoutingMode::Flat | IvfRoutingMode::Auto => (
                    Self::train_flat(config, vectors, vector_count, actual_clusters),
                    None,
                ),
            };

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut soar_config = config.soar.clone();
        if let Some(soar) = &mut soar_config
            && soar.num_secondary > 1
        {
            log::warn!(
                "SOAR currently implements the published primary + one-secondary objective; \
                 clamping {} requested secondary assignments to one",
                soar.num_secondary,
            );
            soar.num_secondary = 1;
        }
        let calibration_target = soar_config
            .as_ref()
            .and_then(SoarConfig::calibration_target);
        let mut trained = Self {
            num_clusters: actual_clusters as u32,
            dim: config.dim,
            centroids,
            version,
            // Install the SOAR policy after calibration so primary assignment
            // below cannot recursively request secondary leaves.
            soar_config: None,
            routing_index,
        };
        if let Some(target) = calibration_target {
            let threshold = trained.calibrate_selective_spill_threshold(
                vectors,
                vector_count,
                config.routing,
                target,
            );
            if let Some(soar) = &mut soar_config {
                soar.spill_threshold = threshold;
            }
            log::info!(
                "Calibrated SOAR selective spilling to at most {:.1}% of the training sample \
                 (residual threshold {:.6})",
                target * 100.0,
                threshold,
            );
        }
        trained.soar_config = soar_config;
        trained
    }

    /// Build a codebook around an existing leaf-centroid matrix without
    /// running k-means.
    ///
    /// Benchmarks and tests use this to route over synthetic centroids of a
    /// chosen size; production codebooks come from [`Self::train`]. Only
    /// `Flat`/`Auto` (no router) and `Hnsw` (graph over the given leaves) are
    /// supported: a two-level router needs trained parent cells.
    pub fn from_leaf_centroids(
        dim: usize,
        centroids: Vec<f32>,
        routing: IvfRoutingMode,
        seed: u64,
        index_label: &str,
    ) -> Self {
        assert!(dim > 0, "centroid dimension must be positive");
        assert!(
            !centroids.is_empty() && centroids.len().is_multiple_of(dim),
            "centroid matrix length {} is not a non-empty multiple of dim {dim}",
            centroids.len()
        );
        let num_clusters = centroids.len() / dim;
        let routing_index = match effective_routing_mode(routing, num_clusters) {
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => None,
            IvfRoutingMode::Hnsw => Some(FloatCentroidRouter::Hnsw(HnswRoutingGraph::build(
                num_clusters,
                |left, right| {
                    let left = left as usize * dim;
                    let right = right as usize * dim;
                    squared_l2(&centroids[left..left + dim], &centroids[right..right + dim])
                },
                seed,
                index_label,
            ))),
            IvfRoutingMode::TwoLevel => panic!(
                "two-level routing needs trained parent centroids; use CoarseCentroids::train"
            ),
        };
        Self {
            num_clusters: num_clusters as u32,
            dim,
            centroids,
            version: seed,
            soar_config: None,
            routing_index,
        }
    }

    fn calibrate_selective_spill_threshold(
        &self,
        vectors: &[f32],
        vector_count: usize,
        routing: IvfRoutingMode,
        target_fraction: f32,
    ) -> f32 {
        // Bound calibration independently of the caller's raw training
        // sample. Flat routing pays O(KD) per selected vector; hierarchical
        // routers can afford a larger validation slice.
        let routing = effective_routing_mode(routing, self.num_clusters as usize);
        let sample_limit = match routing {
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => {
                let work_per_vector = (self.num_clusters as usize).saturating_mul(self.dim).max(1);
                (64_000_000usize / work_per_vector).clamp(32, 4_096)
            }
            IvfRoutingMode::TwoLevel | IvfRoutingMode::Hnsw => 8_192,
        }
        .min(vector_count);
        let mut residual_norms = Vec::with_capacity(sample_limit);
        for sample in 0..sample_limit {
            let vector_index = sample.saturating_mul(vector_count) / sample_limit;
            let offset = vector_index * self.dim;
            let vector = &vectors[offset..offset + self.dim];
            let primary = self.assign_with_routing(vector, routing).primary_cluster;
            residual_norms.push(squared_l2(vector, self.get_centroid(primary)).sqrt());
        }
        residual_norms.sort_unstable_by(f32::total_cmp);
        if residual_norms.is_empty() {
            return 0.0;
        }
        let spill_count = ((residual_norms.len() as f32 * target_fraction.clamp(0.0, 1.0)).round()
            as usize)
            .min(residual_norms.len());
        if spill_count == 0 {
            let largest = residual_norms.last().copied().unwrap_or(0.0);
            return threshold_strictly_above(largest);
        }
        let boundary = residual_norms[residual_norms.len() - spill_count];
        let at_or_above =
            residual_norms.len() - residual_norms.partition_point(|&norm| norm < boundary);
        if at_or_above > spill_count {
            // Equality spills, so a quantile that lands in a tie could exceed
            // the configured storage budget—up to 100% for identical
            // residuals. Move strictly above the tied value; underfilling is
            // preferable to an unbounded posting expansion.
            threshold_strictly_above(boundary)
        } else {
            boundary
        }
    }

    fn train_flat(
        config: &CoarseConfig,
        vectors: &[f32],
        vector_count: usize,
        clusters: usize,
    ) -> Vec<f32> {
        Self::run_kmeans(config, vectors, vector_count, clusters).centroids
    }

    fn train_flat_with_memberships(
        config: &CoarseConfig,
        vectors: &[f32],
        vector_count: usize,
        clusters: usize,
    ) -> (Vec<f32>, FlatClusterMemberships) {
        let trained = Self::run_kmeans(config, vectors, vector_count, clusters);
        (
            trained.centroids,
            FlatClusterMemberships {
                offsets: trained.member_offsets,
                members: trained.members,
            },
        )
    }

    fn run_kmeans(
        config: &CoarseConfig,
        vectors: &[f32],
        vector_count: usize,
        clusters: usize,
    ) -> crate::structures::vector::kmeans::EuclideanKMeans {
        crate::structures::vector::kmeans::train_euclidean_kmeans(
            vectors,
            vector_count,
            config.dim,
            clusters,
            config.max_iters,
            config.seed,
        )
    }

    fn train_hierarchical(
        config: &CoarseConfig,
        vectors: &[f32],
        vector_count: usize,
        leaf_count: usize,
    ) -> (Vec<f32>, FloatCentroidRouter) {
        let parent_count = routing_parent_count(leaf_count).min(vector_count);
        let mut parent_config = config.clone();
        parent_config.routing = IvfRoutingMode::Flat;
        parent_config.num_clusters = parent_count;
        let (parent_centroids, groups) =
            Self::train_flat_with_memberships(&parent_config, vectors, vector_count, parent_count);
        let group_sizes: Vec<usize> = (0..parent_count)
            .map(|parent| groups.cluster(parent).len())
            .collect();
        let child_counts = allocate_child_clusters(&group_sizes, leaf_count);
        let mut leaves = Vec::with_capacity(leaf_count.saturating_mul(config.dim));
        let mut children = vec![Vec::new(); parent_count];
        let mut group_vectors = Vec::new();

        for (parent, &child_count) in child_counts.iter().enumerate() {
            if child_count == 0 {
                continue;
            }
            let indices = groups.cluster(parent);
            group_vectors.clear();
            group_vectors.reserve(indices.len().saturating_mul(config.dim));
            for &index in indices {
                let offset = index * config.dim;
                group_vectors.extend_from_slice(&vectors[offset..offset + config.dim]);
            }
            let mut child_config = config.clone();
            child_config.routing = IvfRoutingMode::Flat;
            child_config.num_clusters = child_count;
            child_config.seed = config.seed ^ (parent as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let first_leaf = leaves.len() / config.dim;
            leaves.extend_from_slice(&Self::train_flat(
                &child_config,
                &group_vectors,
                indices.len(),
                child_count,
            ));
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
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (c, centroid) in centroids.chunks_exact(dim).enumerate() {
            let dist = squared_l2(vector, centroid);
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
        self.flat_k_nearest_with_distances(vector, k, |distances| {
            distances.iter().map(|&(c, _)| c).collect()
        })
    }

    /// Exact flat pass over every leaf centroid, retaining the `k` nearest in
    /// ascending distance order. The O(num_clusters) score buffer lives in
    /// thread-local scratch: at production codebook sizes it is hundreds of
    /// KiB per query and this runs once per query per routed field.
    fn flat_k_nearest_with_distances<T>(
        &self,
        vector: &[f32],
        k: usize,
        finish: impl FnOnce(&[(u32, f32)]) -> T,
    ) -> T {
        CENTROID_SCORE_SCRATCH.with(|scratch| {
            let mut distances = scratch.borrow_mut();
            distances.clear();
            distances.extend(
                (0..self.num_clusters).map(|c| (c, squared_l2(vector, self.get_centroid(c)))),
            );

            // Partial sort: O(n + k log k) instead of O(n log n)
            if distances.len() > k {
                distances.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
                distances.truncate(k);
            }
            distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            finish(&distances)
        })
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
        self.find_k_nearest_two_level_impl::<false>(vector, k)
    }

    fn find_k_nearest_two_level_for_build(&self, vector: &[f32], k: usize) -> Vec<u32> {
        self.find_k_nearest_two_level_impl::<true>(vector, k)
    }

    fn find_k_nearest_two_level_impl<const FOR_BUILD: bool>(
        &self,
        vector: &[f32],
        k: usize,
    ) -> Vec<u32> {
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
        let parents = if FOR_BUILD {
            select_parent_beam_for_build::<false>(&parent_scores, topology, k)
        } else {
            select_parent_beam::<false>(&parent_scores, topology, k)
        };
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

    /// Find k nearest clusters with their distances
    pub fn find_k_nearest_with_distances(&self, vector: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.flat_k_nearest_with_distances(vector, k, <[(u32, f32)]>::to_vec)
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
                IvfRoutingMode::Hnsw => self.find_nearest_hnsw_for_build(vector),
                IvfRoutingMode::TwoLevel => self
                    .find_k_nearest_two_level_for_build(
                        vector,
                        BUILD_ASSIGNMENT_CANDIDATES.min(self.num_clusters as usize),
                    )
                    .first()
                    .copied()
                    .unwrap_or(0),
                IvfRoutingMode::Flat | IvfRoutingMode::Auto => self.find_nearest(vector),
            };
            MultiAssignment {
                primary_cluster,
                secondary_clusters: Vec::new(),
            }
        }
    }

    /// SOAR-style assignment: balance secondary distortion and residual orthogonality
    pub fn assign_with_soar(&self, vector: &[f32], config: &SoarConfig) -> MultiAssignment {
        self.assign_with_soar_and_routing(vector, config, IvfRoutingMode::Flat)
    }

    fn assign_with_soar_and_routing(
        &self,
        vector: &[f32],
        config: &SoarConfig,
        routing: IvfRoutingMode,
    ) -> MultiAssignment {
        // The implemented SOAR loss is the published primary + one-secondary
        // objective. Treat larger manually constructed values the same as the
        // trained/config-parsed path instead of pretending repeated independent
        // minimization implements the generalized multi-spill objective.
        let num_secondary = config.num_secondary.min(1);
        // Secondary assignment needs a meaningfully larger candidate pool
        // than the number of requested spills; otherwise a skewed two-level
        // topology can leave the SOAR loss no alternatives to rank.
        let candidate_budget =
            soar_build_candidate_budget(num_secondary, self.num_clusters as usize);
        let leaf_ids: Vec<u32> = match effective_routing_mode(routing, self.num_clusters as usize) {
            IvfRoutingMode::TwoLevel => {
                self.two_level_candidate_leaves_for_build(vector, candidate_budget)
            }
            IvfRoutingMode::Hnsw => self.find_k_nearest_hnsw_for_build(vector, candidate_budget),
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => (0..self.num_clusters).collect(),
        };
        // Compute every candidate distance once. Reuse it both for primary
        // selection and as the distortion term in the secondary SOAR loss.
        let leaf_distances: Vec<(u32, f32)> = leaf_ids
            .into_iter()
            .map(|cluster| (cluster, squared_l2(vector, self.get_centroid(cluster))))
            .collect();
        let primary = leaf_distances
            .iter()
            .min_by(|left, right| scored_cluster_order(left, right))
            .map(|&(cluster, _)| cluster)
            .unwrap_or(0);
        let primary_centroid = self.get_centroid(primary);

        // 2. Compute primary residual r = x - c
        let residual: Vec<f32> = vector
            .iter()
            .zip(primary_centroid)
            .map(|(v, c)| v - c)
            .collect();

        let residual_norm_sq = crate::structures::simd::norm_squared_f32(&residual);

        // 3. Check if we should spill (selective spilling)
        if config.selective && residual_norm_sq < config.spill_threshold * config.spill_threshold {
            return MultiAssignment {
                primary_cluster: primary,
                secondary_clusters: Vec::new(),
            };
        }

        // 4. Minimize the published lambda=1 SOAR objective:
        //
        //      ||r'||² + lambda * ||proj_r(r')||²
        //
        // This retains ordinary secondary quantization quality while penalizing
        // correlation with the primary residual. Optimizing only the projection
        // term can otherwise select an arbitrarily distant orthogonal centroid.
        let mut candidates: Vec<(u32, f32)> = leaf_distances
            .into_iter()
            .filter(|&(cluster, secondary_residual_norm_sq)| {
                cluster != primary
                    && (!config.selective
                        || secondary_residual_norm_sq
                            <= MAX_SELECTIVE_SECONDARY_TO_PRIMARY_DISTANCE_RATIO_SQ
                                * residual_norm_sq)
            })
            .map(|(cluster, secondary_residual_norm_sq)| {
                (
                    cluster,
                    soar_secondary_loss(
                        vector,
                        self.get_centroid(cluster),
                        &residual,
                        residual_norm_sq,
                        secondary_residual_norm_sq,
                    ),
                )
            })
            .collect();

        // Select by loss, then sort the retained prefix so ties and assignment
        // order are deterministic across platforms and repeated builds.
        let take = num_secondary.min(candidates.len());
        if candidates.len() > take {
            candidates.select_nth_unstable_by(take, scored_cluster_order);
            candidates.truncate(take);
        }
        candidates.sort_unstable_by(scored_cluster_order);

        MultiAssignment {
            primary_cluster: primary,
            secondary_clusters: candidates
                .iter()
                .take(num_secondary)
                .map(|(c, _)| *c)
                .collect(),
        }
    }

    fn two_level_candidate_leaves_for_build(&self, vector: &[f32], k: usize) -> Vec<u32> {
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
        let parents = select_parent_beam_for_build::<false>(&parent_scores, topology, k);
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

    fn find_k_nearest_hnsw_for_build(&self, vector: &[f32], k: usize) -> Vec<u32> {
        let Some(FloatCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_k_nearest(vector, k);
        };
        graph.search_for_build(|leaf| squared_l2(vector, self.get_centroid(leaf)), k)
    }

    /// Single-leaf assignment. Construction routes every vector through here,
    /// so it avoids the one-element result `Vec` the ranked form allocates.
    fn find_nearest_hnsw_for_build(&self, vector: &[f32]) -> u32 {
        let Some(FloatCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_nearest(vector);
        };
        graph
            .search_best_for_build(|leaf| squared_l2(vector, self.get_centroid(leaf)))
            .unwrap_or(0)
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
    #[cfg(feature = "native")]
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

    #[cfg(feature = "native")]
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
    crate::structures::simd::squared_l2_f32(left, right)
}

#[inline]
fn threshold_strictly_above(value: f32) -> f32 {
    if !value.is_finite() {
        return f32::INFINITY;
    }
    // A single ULP is not sufficient near zero because the assignment path
    // compares squared norms and a subnormal threshold can square back to
    // zero. A small relative-or-absolute margin remains negligible for
    // normalized vectors while guaranteeing a strictly larger squared bound.
    let next = value + value.abs().max(1.0) * (4.0 * f32::EPSILON);
    if next > value { next } else { f32::INFINITY }
}

#[inline]
fn soar_build_candidate_budget(num_secondary: usize, num_clusters: usize) -> usize {
    num_secondary
        .saturating_add(1)
        .saturating_mul(64)
        .max(BUILD_ASSIGNMENT_CANDIDATES)
        .min(num_clusters)
}

#[inline]
fn soar_secondary_loss(
    vector: &[f32],
    secondary_centroid: &[f32],
    primary_residual: &[f32],
    primary_residual_norm_sq: f32,
    secondary_residual_norm_sq: f32,
) -> f32 {
    let residual_dot = vector
        .iter()
        .zip(secondary_centroid)
        .zip(primary_residual)
        .fold(0.0f32, |acc, ((&value, &centroid), &primary)| {
            acc.algebraic_add(primary.algebraic_mul(value - centroid))
        });

    // A zero primary residual already has no correlated score error. Define
    // its projection penalty as zero rather than producing 0/0.
    let projection_norm_sq = if primary_residual_norm_sq > 0.0 {
        residual_dot * residual_dot / primary_residual_norm_sq
    } else {
        0.0
    };
    secondary_residual_norm_sq + SOAR_LAMBDA * projection_norm_sq
}

#[inline]
fn scored_cluster_order(left: &(u32, f32), right: &(u32, f32)) -> std::cmp::Ordering {
    left.1
        .total_cmp(&right.1)
        .then_with(|| left.0.cmp(&right.0))
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
        let centroids = CoarseCentroids::train(&config, &vectors, "test");

        assert_eq!(centroids.num_clusters, num_clusters as u32);
        assert_eq!(centroids.dim, dim);
    }

    #[test]
    fn contiguous_training_matches_row_wrapper() {
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..128)
            .map(|row| {
                (0..dim)
                    .map(|column| ((row * 31 + column * 17) % 101) as f32 / 101.0)
                    .collect()
            })
            .collect();
        let flat = vectors.iter().flatten().copied().collect::<Vec<_>>();
        let config = CoarseConfig::new(dim, 12)
            .with_seed(91)
            .with_routing(IvfRoutingMode::TwoLevel);

        let rows = CoarseCentroids::train(&config, &vectors, "test");
        let contiguous = CoarseCentroids::train_contiguous(&config, &flat, vectors.len(), "test");

        assert_eq!(rows.centroids, contiguous.centroids);
        assert_eq!(
            bincode::serde::encode_to_vec(&rows.routing_index, bincode::config::standard())
                .unwrap(),
            bincode::serde::encode_to_vec(&contiguous.routing_index, bincode::config::standard())
                .unwrap(),
        );
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
        let centroids = CoarseCentroids::train(&config, &vectors, "test");

        // Test that find_nearest returns valid cluster IDs
        for v in &vectors {
            let cluster = centroids.find_nearest(v);
            assert!(cluster < centroids.num_clusters);
        }
    }

    #[test]
    fn scaled_l2_probes_keep_distinct_cache_identities() {
        let centroids = CoarseCentroids {
            num_clusters: 2,
            dim: 2,
            centroids: vec![1.0, 0.0, 10.0, 0.0],
            version: 7,
            soar_config: None,
            routing_index: None,
        };

        let near = centroids.probe(&[1.0, 0.0], 1, IvfRoutingMode::Flat);
        let scaled = centroids.probe(&[100.0, 0.0], 1, IvfRoutingMode::Flat);

        assert_eq!(&*near.cluster_ids, &[0]);
        assert_eq!(&*scaled.cluster_ids, &[1]);
        assert_ne!(near.request_fingerprint, scaled.request_fingerprint);
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
        let centroids = CoarseCentroids::train(&config, &vectors, "test");

        // Test SOAR assignment
        let assignment = centroids.assign(&vectors[0]);
        assert!(assignment.primary_cluster < centroids.num_clusters);
        assert_eq!(centroids.soar_config.as_ref().unwrap().num_secondary, 1);
        assert_eq!(assignment.secondary_clusters.len(), 1);

        // Secondary clusters should be different from primary
        for &sec in &assignment.secondary_clusters {
            assert_ne!(sec, assignment.primary_cluster);
        }
    }

    #[test]
    fn soar_loss_includes_distortion_and_normalized_projection() {
        let vector = [0.0, 0.0];
        let primary_residual = [2.0, 0.0];
        let secondary_centroid = [-3.0, -4.0];

        // r' = [3, 4], so ||r'||² = 25 and
        // ||proj_r(r')||² = <r,r'>² / ||r||² = 36 / 4 = 9.
        let loss = soar_secondary_loss(&vector, &secondary_centroid, &primary_residual, 4.0, 25.0);
        assert!((loss - 34.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn soar_routing_keeps_an_oversampled_secondary_candidate_pool() {
        assert_eq!(soar_build_candidate_budget(1, 1_000), 128);
        assert_eq!(soar_build_candidate_budget(2, 1_000), 192);
        assert_eq!(soar_build_candidate_budget(8, 64), 64);
    }

    #[test]
    fn two_level_build_assignment_checks_four_parents_for_primary_and_soar() {
        let leaves_per_parent = 512;
        let children: Vec<Vec<u32>> = (0..4)
            .map(|parent| {
                let first = parent * leaves_per_parent;
                (first..first + leaves_per_parent)
                    .map(|leaf| leaf as u32)
                    .collect()
            })
            .collect();
        let mut leaf_centroids = vec![1.0; 4 * leaves_per_parent];
        let best_leaf = 3 * leaves_per_parent;
        leaf_centroids[best_leaf] = 0.0;
        let centroids = CoarseCentroids {
            num_clusters: leaf_centroids.len() as u32,
            dim: 1,
            centroids: leaf_centroids,
            version: 1,
            soar_config: None,
            routing_index: Some(FloatCentroidRouter::TwoLevel {
                parent_centroids: vec![0.0, 10.0, 20.0, 30.0],
                topology: IvfRoutingTopology::from_children(&children),
            }),
        };

        let query = [0.0];
        assert_eq!(
            &*centroids
                .probe(&query, 1, IvfRoutingMode::TwoLevel)
                .cluster_ids,
            &[0]
        );
        assert_eq!(
            centroids
                .assign_with_routing(&query, IvfRoutingMode::TwoLevel)
                .primary_cluster,
            best_leaf as u32
        );
        assert_eq!(
            centroids
                .assign_with_soar_and_routing(
                    &query,
                    &SoarConfig::full(),
                    IvfRoutingMode::TwoLevel,
                )
                .primary_cluster,
            best_leaf as u32
        );
    }

    #[test]
    fn selective_soar_calibrates_to_a_storage_budget() {
        let centroids = CoarseCentroids {
            num_clusters: 2,
            dim: 2,
            centroids: vec![0.0, 0.0, 10.0, 0.0],
            version: 1,
            soar_config: None,
            routing_index: None,
        };
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|index| vec![index as f32 / 100.0, 0.0])
            .collect();
        let flat = vectors.iter().flatten().copied().collect::<Vec<_>>();
        let threshold = centroids.calibrate_selective_spill_threshold(
            &flat,
            vectors.len(),
            IvfRoutingMode::Flat,
            0.30,
        );
        let spilled = vectors
            .iter()
            .filter(|vector| squared_l2(vector, centroids.get_centroid(0)).sqrt() >= threshold)
            .count();
        assert!((29..=31).contains(&spilled), "{spilled}");
    }

    #[test]
    fn selective_soar_never_exceeds_budget_when_residuals_tie() {
        let centroids = CoarseCentroids {
            num_clusters: 2,
            dim: 2,
            centroids: vec![0.0, 0.0, 10.0, 0.0],
            version: 1,
            soar_config: None,
            routing_index: None,
        };
        let vectors = [1.0f32, 0.0].repeat(100);
        let threshold = centroids.calibrate_selective_spill_threshold(
            &vectors,
            100,
            IvfRoutingMode::Flat,
            0.30,
        );
        let config = SoarConfig::new().threshold(threshold);
        let spilled = vectors
            .chunks_exact(2)
            .filter(|vector| centroids.assign_with_soar(vector, &config).is_spilled())
            .count();

        assert!(threshold > 1.0);
        assert!(spilled <= 30, "{spilled}");
    }

    #[test]
    fn selective_soar_preserves_boundary_query_candidate_recall_with_bounded_postings() {
        const DIM: usize = 16;
        const CLUSTERS: usize = 16;
        const MEMBERS_PER_CLUSTER: usize = 128;
        const TOP_K: usize = 20;
        const TARGET_SPILL: f32 = 0.30;

        fn normalize(values: &mut [f32]) {
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            for value in values {
                *value /= norm;
            }
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x50a4_5eed);
        let source_centers: Vec<Vec<f32>> = (0..CLUSTERS)
            .map(|_| {
                let mut center: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect();
                normalize(&mut center);
                center
            })
            .collect();
        let corpus: Vec<Vec<f32>> = source_centers
            .iter()
            .flat_map(|center| {
                (0..MEMBERS_PER_CLUSTER)
                    .map(|_| {
                        let mut noise: Vec<f32> =
                            (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect();
                        normalize(&mut noise);
                        let mut vector: Vec<f32> = center
                            .iter()
                            .zip(noise)
                            .map(|(&value, noise)| value + 0.75 * noise)
                            .collect();
                        normalize(&mut vector);
                        vector
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let selective = CoarseCentroids::train(
            &CoarseConfig::new(DIM, CLUSTERS)
                .with_seed(0x1f4)
                .with_routing(IvfRoutingMode::Flat)
                .with_soar(SoarConfig::new().target_spill_fraction(TARGET_SPILL)),
            &corpus,
            "test",
        );
        // Share the exact trained codebook so the only variable is whether
        // documents receive selective secondary postings.
        let mut primary_only = selective.clone();
        primary_only.soar_config = None;

        let primary_assignments: Vec<MultiAssignment> = corpus
            .iter()
            .map(|vector| primary_only.assign(vector))
            .collect();
        let selective_assignments: Vec<MultiAssignment> = corpus
            .iter()
            .map(|vector| selective.assign(vector))
            .collect();
        for (primary, spilled) in primary_assignments.iter().zip(&selective_assignments) {
            assert_eq!(
                primary.primary_cluster, spilled.primary_cluster,
                "SOAR policy must not change the primary posting"
            );
        }

        let spilled = selective_assignments
            .iter()
            .filter(|assignment| assignment.is_spilled())
            .count();
        let posting_factor = selective_assignments
            .iter()
            .map(MultiAssignment::num_assignments)
            .sum::<usize>() as f32
            / corpus.len() as f32;
        let spill_budget = (corpus.len() as f32 * TARGET_SPILL).round() as usize;
        assert!(
            spilled > 0,
            "the smoke corpus must exercise selective spilling"
        );
        assert!(
            spilled <= spill_budget,
            "{spilled} spilled vectors exceeded the calibrated budget of {spill_budget}"
        );
        assert!(
            posting_factor <= 1.0 + TARGET_SPILL + f32::EPSILON,
            "posting amplification {posting_factor:.4} exceeded the 1.30 calibration target"
        );

        // Midpoints between each learned centroid and its nearest peer stress
        // the exact partition boundaries where a single probe loses the most
        // candidates and selective secondary postings should help.
        let queries: Vec<Vec<f32>> = (0..selective.num_clusters)
            .map(|left| {
                let left_centroid = selective.get_centroid(left);
                let right = (0..selective.num_clusters)
                    .filter(|&candidate| candidate != left)
                    .min_by(|&a, &b| {
                        squared_l2(left_centroid, selective.get_centroid(a))
                            .total_cmp(&squared_l2(left_centroid, selective.get_centroid(b)))
                            .then_with(|| a.cmp(&b))
                    })
                    .unwrap();
                let mut query: Vec<f32> = left_centroid
                    .iter()
                    .zip(selective.get_centroid(right))
                    .map(|(&a, &b)| 0.51 * a + 0.49 * b)
                    .collect();
                normalize(&mut query);
                query
            })
            .collect();

        let mut gained_queries = 0usize;
        for nprobe in [1, 2] {
            let mut primary_hits = 0usize;
            let mut selective_hits = 0usize;
            for query in &queries {
                let primary_plan = primary_only.probe(query, nprobe, IvfRoutingMode::Flat);
                let selective_plan = selective.probe(query, nprobe, IvfRoutingMode::Flat);
                assert_eq!(
                    primary_plan.cluster_ids, selective_plan.cluster_ids,
                    "SOAR must not alter query routing"
                );

                let mut truth: Vec<(usize, f32)> = corpus
                    .iter()
                    .enumerate()
                    .map(|(document, vector)| (document, squared_l2(query, vector)))
                    .collect();
                truth.select_nth_unstable_by(TOP_K, |left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.cmp(&right.0))
                });
                truth.truncate(TOP_K);

                let query_primary_hits = truth
                    .iter()
                    .filter(|&&(document, _)| {
                        primary_assignments[document]
                            .all_clusters()
                            .any(|cluster| primary_plan.cluster_ids.contains(&cluster))
                    })
                    .count();
                let query_selective_hits = truth
                    .iter()
                    .filter(|&&(document, _)| {
                        selective_assignments[document]
                            .all_clusters()
                            .any(|cluster| selective_plan.cluster_ids.contains(&cluster))
                    })
                    .count();
                primary_hits += query_primary_hits;
                selective_hits += query_selective_hits;
                gained_queries += usize::from(query_selective_hits > query_primary_hits);
            }

            let denominator = (queries.len() * TOP_K) as f32;
            let primary_recall = primary_hits as f32 / denominator;
            let selective_recall = selective_hits as f32 / denominator;
            assert!(
                selective_recall + 0.005 >= primary_recall,
                "selective SOAR candidate recall regressed at nprobe={nprobe}: \
                 {selective_recall:.4} vs {primary_recall:.4}"
            );
            if nprobe == 1 {
                assert!(
                    selective_recall >= primary_recall + 0.01,
                    "selective SOAR must recover boundary candidates at nprobe=1: \
                     {selective_recall:.4} vs {primary_recall:.4}"
                );
            }
        }
        assert!(
            gained_queries > 0,
            "boundary-query smoke did not exercise a selective SOAR recall gain"
        );
    }

    #[test]
    fn soar_does_not_choose_an_arbitrarily_distant_orthogonal_centroid() {
        let centroids = CoarseCentroids {
            num_clusters: 3,
            dim: 2,
            // For x=[0,0], cluster 0 is primary with r=[1,0].
            // Cluster 1 is perfectly orthogonal but extremely distant.
            // Cluster 2 is slightly farther than the primary and parallel:
            // its complete SOAR loss is 1.21 + 1.21 = 2.42.
            centroids: vec![-1.0, 0.0, 0.0, -100.0, 1.1, 0.0],
            version: 1,
            soar_config: None,
            routing_index: None,
        };

        let assignment = centroids.assign_with_soar(&[0.0, 0.0], &SoarConfig::full());
        assert_eq!(assignment.primary_cluster, 0);
        assert_eq!(assignment.secondary_clusters, vec![2]);
    }

    #[test]
    fn selective_soar_rejects_a_far_secondary_residual() {
        let centroids = CoarseCentroids {
            num_clusters: 2,
            dim: 2,
            centroids: vec![0.0, 0.0, 10.0, 0.0],
            version: 1,
            soar_config: None,
            routing_index: None,
        };
        let config = SoarConfig::new().threshold(0.0);

        // Primary squared distance is 1; the only secondary is 81 away.
        // Selective SOAR must not create a low-quality far-leaf posting.
        let assignment = centroids.assign_with_soar(&[1.0, 0.0], &config);
        assert_eq!(assignment.primary_cluster, 0);
        assert!(assignment.secondary_clusters.is_empty());
    }

    #[test]
    fn selective_soar_keeps_a_comparable_boundary_secondary() {
        let centroids = CoarseCentroids {
            num_clusters: 2,
            dim: 2,
            centroids: vec![0.0, 0.0, 2.0, 0.0],
            version: 1,
            soar_config: None,
            routing_index: None,
        };
        let config = SoarConfig::new().threshold(0.0);

        // The point is close to the Voronoi boundary: primary and secondary
        // squared distances are 0.82 and 1.22, comfortably within the gate.
        let assignment = centroids.assign_with_soar(&[0.9, 0.1], &config);
        assert_eq!(assignment.primary_cluster, 0);
        assert_eq!(assignment.secondary_clusters, vec![1]);
    }

    #[test]
    fn soar_secondary_ties_are_ordered_by_cluster_id_and_capped_to_one() {
        let centroids = CoarseCentroids {
            num_clusters: 3,
            dim: 2,
            centroids: vec![0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            version: 1,
            soar_config: None,
            routing_index: None,
        };
        let config = SoarConfig {
            num_secondary: 2,
            selective: false,
            spill_threshold: 0.0,
        };

        let assignment = centroids.assign_with_soar(&[0.0, 0.0], &config);
        assert_eq!(assignment.primary_cluster, 0);
        assert_eq!(assignment.secondary_clusters, vec![1]);
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
        let centroids = CoarseCentroids::train(&config, &vectors, "test");

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
            let trained = CoarseCentroids::train(
                &CoarseConfig::new(dim, 16).with_routing(routing),
                &vectors,
                "test",
            );
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
