//! Binary IVF: inverted-file index for packed-bit vectors under Hamming distance.
//!
//! Coarse centroids are trained with k-majority clustering (the Hamming-space
//! analog of k-means: assignment by Hamming distance, centroid update by
//! per-bit majority vote). Cluster payloads store the *exact* packed codes in
//! one contiguous buffer per cluster, so within-cluster scanning uses the same
//! SIMD Hamming kernel as brute force and scanned candidates have exact
//! distances — the only approximation is which clusters get probed (`nprobe`).
//!
//! Brute-force Hamming is fast (~10-50M vectors/s/core), so this index pays
//! off for segments past a few million vectors, or when many binary fields
//! are queried concurrently.

use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;

use crate::dsl::IvfRoutingMode;
use crate::structures::simd::{HammingKernel, scores_from_hamming};
#[cfg(test)]
use crate::structures::simd::{batch_hamming_scores, hamming_distance};
#[cfg(test)]
use crate::structures::vector::ivf::routing::select_best;
use crate::structures::vector::ivf::routing::{
    HIERARCHICAL_TRAINING_THRESHOLD, HnswRoutingGraph, IvfProbePlan, IvfRoutingTopology,
    PairDistance, QueryDistance, allocate_child_clusters,
    binary_probe_fingerprint_with_parent_beam, effective_binary_routing_mode, routing_parent_count,
    select_best_candidates, select_parent_beam_for_build_with_oversample,
    select_parent_beam_with_oversample,
};
use crate::structures::vector::progress::PhaseProgress;

/// Hamming distance from one query code to graph nodes backed by a packed
/// centroid matrix.
///
/// Routing visits scattered centroid rows, so the batched form gathers a whole
/// neighbour list through one SIMD dispatch instead of one call per node.
struct BinaryCentroidDistance<'a> {
    query: &'a [u8],
    centroids: &'a [u8],
    byte_len: usize,
    kernel: HammingKernel,
}

impl<'a> BinaryCentroidDistance<'a> {
    #[inline]
    fn new(query: &'a [u8], centroids: &'a [u8], byte_len: usize) -> Self {
        Self {
            query,
            centroids,
            byte_len,
            kernel: HammingKernel::resolve(),
        }
    }
}

impl QueryDistance for BinaryCentroidDistance<'_> {
    #[inline]
    fn distance(&self, node: u32) -> f32 {
        let offset = node as usize * self.byte_len;
        self.kernel
            .distance(self.query, &self.centroids[offset..offset + self.byte_len]) as f32
    }

    fn distances(&self, nodes: &[u32], out: &mut [f32]) {
        let mut distances = [0u32; ROUTING_GATHER_BLOCK];
        for (block, scores) in nodes
            .chunks(ROUTING_GATHER_BLOCK)
            .zip(out.chunks_mut(ROUTING_GATHER_BLOCK))
        {
            let gathered = &mut distances[..block.len()];
            self.kernel.gather_distances(
                self.query,
                self.centroids,
                self.byte_len,
                block,
                gathered,
            );
            for (slot, &distance) in scores.iter_mut().zip(gathered.iter()) {
                *slot = distance as f32;
            }
        }
    }
}

/// Pairwise centroid distance used while building the routing graph.
struct BinaryCentroidPairDistance<'a> {
    centroids: &'a [u8],
    byte_len: usize,
    kernel: HammingKernel,
}

impl PairDistance for BinaryCentroidPairDistance<'_> {
    #[inline]
    fn distance(&self, left: u32, right: u32) -> f32 {
        let left_offset = left as usize * self.byte_len;
        let right_offset = right as usize * self.byte_len;
        self.kernel.distance(
            &self.centroids[left_offset..left_offset + self.byte_len],
            &self.centroids[right_offset..right_offset + self.byte_len],
        ) as f32
    }

    fn distances_from(&self, left: u32, rights: &[u32], out: &mut [f32]) {
        let offset = left as usize * self.byte_len;
        BinaryCentroidDistance {
            query: &self.centroids[offset..offset + self.byte_len],
            centroids: self.centroids,
            byte_len: self.byte_len,
            kernel: self.kernel,
        }
        .distances(rights, out);
    }
}

/// Rows gathered per kernel dispatch while routing. Covers a full level-0
/// neighbour list (`2 * HNSW_M`) without touching the allocator.
const ROUTING_GATHER_BLOCK: usize = 128;

/// Nearest centroid and its exact Hamming distance.
///
/// Ranking stays integral: the winner's distance falls out of the same scan
/// rather than costing a second full-width distance, and equal distances keep
/// the lowest cluster ID to match query-time centroid ordering.
#[inline]
fn nearest_binary_centroid_with_distance(
    kernel: HammingKernel,
    code: &[u8],
    centroids: &[u8],
    byte_len: usize,
    distances: &mut [u32],
) -> (u32, u32) {
    kernel.distances(code, centroids, byte_len, distances);
    let mut best = (u32::MAX, 0u32);
    for (cluster, &distance) in distances.iter().enumerate() {
        if distance < best.0 {
            best = (distance, cluster as u32);
        }
    }
    (best.1, best.0)
}

const MAX_BINARY_IVF_CLUSTERS: usize = 1_048_576;
#[cfg(test)]
const BINARY_IVF_SCORE_BATCH: usize = 8_192;
const BUILD_ASSIGNMENT_CANDIDATES: usize = 128;
/// Two-level binary routing can spend more leaf-centroid work than float IVF:
/// packed centroids are narrow and exact Hamming scans vectorize well. Widen
/// only once the direct-training crossover is reached, then again for truly
/// large codebooks where a fixed four-times beam becomes nearly greedy.
pub const BINARY_PARENT_BEAM_OVERSAMPLE_MEDIUM: usize = 8;
pub const BINARY_PARENT_BEAM_OVERSAMPLE_LARGE: usize = 12;

/// Default runtime parent coverage for a binary codebook. Callers that expose
/// an expert knob can override this through `probe_with_parent_beam` without
/// changing the serialized quantizer format.
pub fn adaptive_binary_parent_beam_oversample(num_clusters: usize) -> usize {
    match num_clusters {
        65_536.. => BINARY_PARENT_BEAM_OVERSAMPLE_LARGE,
        HIERARCHICAL_TRAINING_THRESHOLD.. => BINARY_PARENT_BEAM_OVERSAMPLE_MEDIUM,
        _ => crate::structures::vector::ivf::routing::DEFAULT_PARENT_BEAM_OVERSAMPLE,
    }
}

#[inline]
fn uses_hierarchical_binary_training(num_clusters: usize) -> bool {
    num_clusters >= HIERARCHICAL_TRAINING_THRESHOLD
}

/// A code with no set bits carries no information: its Hamming distance to any
/// query is a constant `popcount(query)`, so it scores mid-range against
/// *everything* while matching nothing.
///
/// It is also a latency cliff. Equal distances resolve to the lowest cluster
/// ID, so every zero code lands in the same leaf: one production field
/// accumulated 31% of its vectors — 20.1M codes, 6.0 GiB — in leaf 0, and any
/// query probing that leaf scanned all of it.
///
/// They are still indexed, because the byte-copy merge path requires an ANN
/// payload to hold exactly as many vectors as flat storage; withholding them
/// needs a versioned header field and is deliberately not done here. What the
/// build does instead is count and report them, so a producer regression is
/// loud rather than a silent scan cliff.
#[inline]
fn is_zero_code(code: &[u8]) -> bool {
    code.iter().all(|&byte| byte == 0)
}

/// The saturated twin of [`is_zero_code`]: every bit set. Produced by packers
/// that route NaN through a signbit test (`~signbit(NaN)` is true), where the
/// `x > 0` convention produces zeros instead. One production corpus carried
/// this face for two years — 36% of a field's vectors were the identical
/// all-ones code, with a codebook centroid trained to exactly 0xFF — so it
/// gets the same count-and-report treatment. `dim_bits` is validated to be a
/// multiple of 8, so a byte-level check is exact (no pad bits).
#[inline]
fn is_ones_code(code: &[u8]) -> bool {
    code.iter().all(|&byte| byte == 0xff)
}

/// A leaf this many times larger than the average is a scan cliff regardless of
/// what produced it, so segment builds report it.
#[cfg(feature = "native")]
const LEAF_SKEW_WARN_RATIO: usize = 100;
/// Below this many vectors a skewed leaf cannot cost enough to be worth a line.
#[cfg(feature = "native")]
const LEAF_SKEW_WARN_MINIMUM: usize = 10_000;

/// Global Hamming coarse quantizer shared by every segment of a field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryCoarseQuantizer {
    pub dim_bits: usize,
    pub num_clusters: u32,
    /// Packed leaf centroids (`num_clusters × byte_len`).
    centroids: Vec<u8>,
    pub version: u64,
    routing_index: Option<BinaryCentroidRouter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum BinaryCentroidRouter {
    TwoLevel {
        parent_centroids: Vec<u8>,
        topology: IvfRoutingTopology,
    },
    Hnsw(HnswRoutingGraph),
}

impl BinaryCoarseQuantizer {
    pub fn train(
        mut config: BinaryIvfConfig,
        codes: &[u8],
        num_vectors: usize,
        index_label: &str,
    ) -> io::Result<Self> {
        config
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        let expected = num_vectors.checked_mul(config.byte_len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "binary training size overflow")
        })?;
        if num_vectors == 0 || codes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary coarse training requires a non-empty, contiguous code matrix",
            ));
        }
        config.num_clusters = config.num_clusters.clamp(1, num_vectors);
        let (centroids, routing_index) =
            match effective_binary_routing_mode(config.routing, config.num_clusters) {
                IvfRoutingMode::TwoLevel => {
                    let (leaves, router) =
                        train_k_majority_hierarchical(&config, codes, num_vectors, index_label);
                    (leaves, Some(router))
                }
                IvfRoutingMode::Hnsw => {
                    let leaves = if uses_hierarchical_binary_training(config.num_clusters) {
                        train_k_majority_hierarchical(&config, codes, num_vectors, index_label).0
                    } else {
                        train_k_majority(&config, codes, num_vectors, index_label)
                    };
                    let byte_len = config.byte_len();
                    let graph = HnswRoutingGraph::build(
                        config.num_clusters,
                        BinaryCentroidPairDistance {
                            centroids: &leaves,
                            byte_len,
                            kernel: HammingKernel::resolve(),
                        },
                        config.seed,
                        index_label,
                    );
                    (leaves, Some(BinaryCentroidRouter::Hnsw(graph)))
                }
                IvfRoutingMode::Flat | IvfRoutingMode::Auto => {
                    // Routing and training have different crossover points.
                    // Packed centroids remain cheap enough for exact flat
                    // probing well past 4K leaves, while direct k-majority
                    // seeding is already O(N*K) and impractical there.
                    let leaves = if uses_hierarchical_binary_training(config.num_clusters) {
                        train_k_majority_hierarchical(&config, codes, num_vectors, index_label).0
                    } else {
                        train_k_majority(&config, codes, num_vectors, index_label)
                    };
                    (leaves, None)
                }
            };
        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Ok(Self {
            dim_bits: config.dim_bits,
            num_clusters: config.num_clusters as u32,
            centroids,
            version,
            routing_index,
        })
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.dim_bits.div_ceil(8)
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = (self.num_clusters as usize)
            .checked_mul(self.byte_len())
            .ok_or_else(|| "binary coarse centroid size overflow".to_string())?;
        if self.dim_bits == 0
            || !self.dim_bits.is_multiple_of(8)
            || self.num_clusters == 0
            || self.centroids.len() != expected
        {
            return Err("invalid binary coarse quantizer shape".to_string());
        }
        if let Some(router) = &self.routing_index {
            match router {
                BinaryCentroidRouter::TwoLevel {
                    parent_centroids,
                    topology,
                } => {
                    let parent_count = topology.parent_count();
                    if parent_count == 0
                        || parent_centroids.len() != parent_count.saturating_mul(self.byte_len())
                        || !topology.validate(self.num_clusters as usize)
                    {
                        return Err("invalid binary two-level routing index".to_string());
                    }
                }
                BinaryCentroidRouter::Hnsw(graph)
                    if !graph.validate(self.num_clusters as usize) =>
                {
                    return Err("invalid binary HNSW routing graph".to_string());
                }
                BinaryCentroidRouter::Hnsw(_) => {}
            }
        }
        Ok(())
    }

    pub fn validate_routing(&self, mode: IvfRoutingMode) -> Result<(), String> {
        match effective_binary_routing_mode(mode, self.num_clusters as usize) {
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => {}
            IvfRoutingMode::TwoLevel
                if !matches!(
                    self.routing_index,
                    Some(BinaryCentroidRouter::TwoLevel { .. })
                ) =>
            {
                return Err(
                    "two-level IVF routing was requested but the global binary quantizer has no matching router"
                        .to_string(),
                );
            }
            IvfRoutingMode::Hnsw
                if !matches!(self.routing_index, Some(BinaryCentroidRouter::Hnsw(_))) =>
            {
                return Err(
                    "HNSW IVF routing was requested but the global binary quantizer has no HNSW graph"
                        .to_string(),
                );
            }
            IvfRoutingMode::TwoLevel | IvfRoutingMode::Hnsw => {}
        }
        self.validate()
    }

    /// Visit compact routing topology and parent arrays before the potentially
    /// much larger leaf centroid matrix.
    #[cfg(feature = "native")]
    pub(crate) fn visit_routing_regions(&self, visit: &mut dyn FnMut(&'static str, &[u8])) {
        if let Some(router) = &self.routing_index {
            match router {
                BinaryCentroidRouter::TwoLevel {
                    parent_centroids,
                    topology,
                } => {
                    topology.visit_resident_regions(visit);
                    visit("binary parent centroids", parent_centroids);
                }
                BinaryCentroidRouter::Hnsw(graph) => graph.visit_resident_regions(visit),
            }
        }
    }

    #[cfg(feature = "native")]
    pub(crate) fn visit_leaf_centroid_region(&self, visit: &mut dyn FnMut(&'static str, &[u8])) {
        visit("binary leaf centroids", &self.centroids);
    }

    pub fn probe(&self, query: &[u8], k: usize, mode: IvfRoutingMode) -> io::Result<IvfProbePlan> {
        self.probe_with_parent_beam(
            query,
            k,
            mode,
            adaptive_binary_parent_beam_oversample(self.num_clusters as usize),
        )
    }

    /// Cache key for the default probe policy. This must stay paired with
    /// [`Self::probe`]: automatic routing and adaptive parent beams are both
    /// resolved here before hashing.
    pub fn request_fingerprint(&self, query: &[u8], k: usize, mode: IvfRoutingMode) -> u64 {
        let take = k.clamp(1, self.num_clusters as usize);
        let effective_mode = effective_binary_routing_mode(mode, self.num_clusters as usize);
        binary_probe_fingerprint_with_parent_beam(
            query,
            take,
            effective_mode,
            adaptive_binary_parent_beam_oversample(self.num_clusters as usize),
        )
    }

    /// Probe with an explicit two-level parent coverage multiplier.
    ///
    /// The routing layer clamps the multiplier to its hard work bound. Flat
    /// and HNSW modes ignore it, so a runtime integration can pass one policy
    /// uniformly without changing artifacts or exact leaf scoring.
    pub fn probe_with_parent_beam(
        &self,
        query: &[u8],
        k: usize,
        mode: IvfRoutingMode,
        parent_beam_oversample: usize,
    ) -> io::Result<IvfProbePlan> {
        self.check_code_len(query, "probe")?;
        let take = k.clamp(1, self.num_clusters as usize);
        let effective_mode = effective_binary_routing_mode(mode, self.num_clusters as usize);
        let cluster_ids = match effective_mode {
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => self.find_k_nearest(query, take)?,
            IvfRoutingMode::TwoLevel => {
                self.find_k_nearest_two_level(query, take, parent_beam_oversample)?
            }
            IvfRoutingMode::Hnsw => self.find_k_nearest_hnsw(query, take)?,
        };
        Ok(IvfProbePlan::new(
            self.version,
            binary_probe_fingerprint_with_parent_beam(
                query,
                take,
                effective_mode,
                parent_beam_oversample,
            ),
            cluster_ids,
        ))
    }

    /// Assign one code to its leaf.
    ///
    /// Segment construction routes every vector through here, so this path must
    /// stay allocation-free. A code of the wrong width, or a router that cannot
    /// produce a leaf, is an error: silently assigning cluster 0 would build a
    /// payload that scores as valid but never finds those vectors.
    pub fn assign(&self, code: &[u8], mode: IvfRoutingMode) -> io::Result<u32> {
        self.check_code_len(code, "assign")?;
        let parent_beam_oversample =
            adaptive_binary_parent_beam_oversample(self.num_clusters as usize);
        match effective_binary_routing_mode(mode, self.num_clusters as usize) {
            IvfRoutingMode::Hnsw => self.find_nearest_hnsw_for_build(code)?.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "binary HNSW centroid router produced no leaf for a valid code",
                )
            }),
            IvfRoutingMode::TwoLevel => self
                .find_k_nearest_two_level_for_build(
                    code,
                    BUILD_ASSIGNMENT_CANDIDATES.min(self.num_clusters as usize),
                    parent_beam_oversample,
                )?
                .first()
                .copied()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "binary two-level centroid router produced no leaf for a valid code",
                    )
                }),
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => self.find_nearest(code),
        }
    }

    fn check_code_len(&self, code: &[u8], operation: &str) -> io::Result<()> {
        if code.len() != self.byte_len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "binary IVF {operation}: code has {} bytes but the quantizer expects {} \
                     ({} bits)",
                    code.len(),
                    self.byte_len(),
                    self.dim_bits
                ),
            ));
        }
        Ok(())
    }

    fn find_nearest(&self, query: &[u8]) -> io::Result<u32> {
        let byte_len = self.byte_len();
        self.check_code_len(query, "find_nearest")?;
        let kernel = HammingKernel::resolve();
        let mut distances = [0u32; BINARY_CENTROID_SCAN_BLOCK];
        let mut best = (u32::MAX, 0u32);
        for (block_index, block) in self
            .centroids
            .chunks(BINARY_CENTROID_SCAN_BLOCK * byte_len)
            .enumerate()
        {
            let rows = block.len() / byte_len;
            let scored = &mut distances[..rows];
            kernel.distances(query, block, byte_len, scored);
            for (row, &distance) in scored.iter().enumerate() {
                // Ties keep the lowest cluster ID, matching query-time ordering.
                if distance < best.0 {
                    best = (
                        distance,
                        (block_index * BINARY_CENTROID_SCAN_BLOCK + row) as u32,
                    );
                }
            }
        }
        Ok(best.1)
    }

    /// Exact flat probe: every centroid is scored, then the `k` nearest are
    /// selected on integers. Hamming distances stay `u32` and are packed with
    /// the cluster ID into one `u64` (`distance << 32 | id`), so an unstable
    /// nth-element partition on plain integers yields exactly the ordering the
    /// float `1 - d / dim` score with an ID tie-break produced, without the
    /// float round trip or a `num_clusters`-wide permutation per query.
    fn find_k_nearest(&self, query: &[u8], k: usize) -> io::Result<Vec<u32>> {
        self.check_code_len(query, "find_k_nearest")?;
        let num_clusters = self.num_clusters as usize;
        let take = k.min(num_clusters);
        if take == 0 {
            return Ok(Vec::new());
        }
        Ok(with_binary_probe_scratch(|scratch| {
            scratch.distances.clear();
            scratch.distances.resize(num_clusters, 0);
            HammingKernel::resolve().distances(
                query,
                &self.centroids,
                self.byte_len(),
                &mut scratch.distances,
            );
            scratch.packed.clear();
            scratch.packed.extend(
                scratch
                    .distances
                    .iter()
                    .enumerate()
                    .map(|(id, &distance)| (u64::from(distance) << 32) | id as u64),
            );
            select_best_packed(&mut scratch.packed, take)
        }))
    }

    fn find_k_nearest_two_level(
        &self,
        query: &[u8],
        k: usize,
        parent_beam_oversample: usize,
    ) -> io::Result<Vec<u32>> {
        self.find_k_nearest_two_level_impl::<false>(query, k, parent_beam_oversample)
    }

    fn find_k_nearest_two_level_for_build(
        &self,
        query: &[u8],
        k: usize,
        parent_beam_oversample: usize,
    ) -> io::Result<Vec<u32>> {
        self.find_k_nearest_two_level_impl::<true>(query, k, parent_beam_oversample)
    }

    fn find_k_nearest_two_level_impl<const FOR_BUILD: bool>(
        &self,
        query: &[u8],
        k: usize,
        parent_beam_oversample: usize,
    ) -> io::Result<Vec<u32>> {
        let Some(BinaryCentroidRouter::TwoLevel {
            parent_centroids,
            topology,
        }) = self.routing_index.as_ref()
        else {
            return self.find_k_nearest(query, k);
        };
        if topology.parent_count() <= 1 {
            return self.find_k_nearest(query, k);
        }
        self.check_code_len(query, "two-level probe")?;
        let byte_len = self.byte_len();
        let kernel = HammingKernel::resolve();
        Ok(with_binary_probe_scratch(|scratch| {
            let parent_scores = &mut scratch.parent_scores;
            parent_scores.clear();
            parent_scores.resize(topology.parent_count(), 0.0);
            scores_from_hamming(
                kernel,
                query,
                parent_centroids,
                byte_len,
                self.dim_bits,
                parent_scores,
            );
            let parents = if FOR_BUILD {
                select_parent_beam_for_build_with_oversample::<true>(
                    parent_scores,
                    topology,
                    k,
                    parent_beam_oversample,
                    parent_beam_oversample.min(BINARY_PARENT_BEAM_OVERSAMPLE_MEDIUM),
                )
            } else {
                select_parent_beam_with_oversample::<true>(
                    parent_scores,
                    topology,
                    k,
                    parent_beam_oversample,
                )
            };
            let candidate_count = parents
                .iter()
                .map(|&parent| topology.children(parent as usize).len())
                .sum();
            let candidates = &mut scratch.candidates;
            candidates.clear();
            candidates.reserve(candidate_count);
            // Each parent owns a contiguous leaf run, so its children are one
            // batched pass over the centroid matrix rather than one kernel call
            // (and one runtime feature detection) per leaf.
            let leaf_scores = &mut scratch.leaf_scores;
            for parent in parents {
                let children = topology.children(parent as usize);
                match topology.children_run(parent as usize) {
                    Some((first_leaf, count)) => {
                        leaf_scores.clear();
                        leaf_scores.resize(count, 0.0);
                        let start = first_leaf as usize * byte_len;
                        scores_from_hamming(
                            kernel,
                            query,
                            &self.centroids[start..start + count * byte_len],
                            byte_len,
                            self.dim_bits,
                            leaf_scores,
                        );
                        candidates.extend(
                            (first_leaf..first_leaf + count as u32)
                                .zip(leaf_scores.iter().copied()),
                        );
                    }
                    None => {
                        for &leaf in children {
                            let offset = leaf as usize * byte_len;
                            let distance =
                                kernel.distance(query, &self.centroids[offset..offset + byte_len]);
                            candidates.push((leaf, 1.0 - distance as f32 / self.dim_bits as f32));
                        }
                    }
                }
            }
            select_best_candidates::<true>(candidates, k)
        }))
    }

    fn find_k_nearest_hnsw(&self, query: &[u8], k: usize) -> io::Result<Vec<u32>> {
        let Some(BinaryCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_k_nearest(query, k);
        };
        self.check_code_len(query, "HNSW probe")?;
        Ok(graph.search(
            BinaryCentroidDistance::new(query, &self.centroids, self.byte_len()),
            k,
        ))
    }

    fn find_nearest_hnsw_for_build(&self, query: &[u8]) -> io::Result<Option<u32>> {
        let Some(BinaryCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_nearest(query).map(Some);
        };
        self.check_code_len(query, "HNSW assign")?;
        Ok(graph.search_best_for_build(BinaryCentroidDistance::new(
            query,
            &self.centroids,
            self.byte_len(),
        )))
    }
}

/// Centroid rows scored per stack block during a flat scan.
const BINARY_CENTROID_SCAN_BLOCK: usize = 64;

/// Per-thread probe buffers. A flat probe scores every centroid (hundreds of
/// KiB at production cluster counts) and the two-level probe needs parent
/// scores, the child candidate list and a per-parent leaf score block; none
/// of them should be allocated per query.
#[derive(Default)]
struct BinaryProbeScratch {
    distances: Vec<u32>,
    packed: Vec<u64>,
    parent_scores: Vec<f32>,
    candidates: Vec<(u32, f32)>,
    leaf_scores: Vec<f32>,
}

thread_local! {
    static BINARY_PROBE_SCRATCH: std::cell::RefCell<BinaryProbeScratch> =
        std::cell::RefCell::new(BinaryProbeScratch::default());
}

fn with_binary_probe_scratch<T>(scope: impl FnOnce(&mut BinaryProbeScratch) -> T) -> T {
    BINARY_PROBE_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => scope(&mut scratch),
        // Re-entrant use is not expected; a private buffer keeps the probe
        // correct rather than panicking inside a query.
        Err(_) => scope(&mut BinaryProbeScratch::default()),
    })
}

/// Select the `take` smallest `(distance << 32 | id)` keys and return their
/// IDs in ascending `(distance, id)` order. Integer keys make the nth-element
/// partition branch-cheap and the tie-break exact.
fn select_best_packed(packed: &mut Vec<u64>, take: usize) -> Vec<u32> {
    let take = take.min(packed.len());
    if take == 0 {
        return Vec::new();
    }
    if take < packed.len() {
        packed.select_nth_unstable(take);
        packed.truncate(take);
    }
    packed.sort_unstable();
    packed.iter().map(|&key| key as u32).collect()
}

fn default_max_train_samples() -> usize {
    100_000
}

fn default_hierarchical_parent_restarts() -> usize {
    2
}

const MAX_HIERARCHICAL_PARENT_RESTARTS: usize = 4;

/// Configuration for a binary IVF index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryIvfConfig {
    /// Number of bits per vector (must be a multiple of 8)
    pub dim_bits: usize,
    /// Number of clusters
    pub num_clusters: usize,
    /// Flat, two-level, or HNSW coarse routing. Auto chooses from the leaf count.
    pub routing: IvfRoutingMode,
    /// k-majority training iterations
    pub train_iters: usize,
    /// Cap on vectors used for centroid training (assignment still covers
    /// all vectors). Bounds merge-time training cost on huge segments.
    #[serde(default = "default_max_train_samples")]
    pub max_train_samples: usize,
    /// Deterministic parent-codebook candidates evaluated by exact Hamming
    /// loss before hierarchical child training. This strengthens large IVF
    /// training without multiplying the much larger child-codebook work.
    #[serde(default = "default_hierarchical_parent_restarts")]
    pub hierarchical_parent_restarts: usize,
    /// RNG seed for centroid initialization (deterministic builds)
    pub seed: u64,
}

impl BinaryIvfConfig {
    pub fn new(dim_bits: usize, num_clusters: usize) -> Self {
        Self {
            dim_bits,
            num_clusters,
            routing: IvfRoutingMode::Auto,
            train_iters: 10,
            max_train_samples: default_max_train_samples(),
            hierarchical_parent_restarts: default_hierarchical_parent_restarts(),
            seed: 42,
        }
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.dim_bits.div_ceil(8)
    }

    fn validate(&self) -> Result<(), String> {
        if self.dim_bits == 0 || !self.dim_bits.is_multiple_of(8) {
            return Err(format!(
                "binary IVF dimension must be a positive multiple of 8, got {}",
                self.dim_bits
            ));
        }
        if !(1..=MAX_BINARY_IVF_CLUSTERS).contains(&self.num_clusters) {
            return Err(format!(
                "binary IVF cluster count must be in 1..={MAX_BINARY_IVF_CLUSTERS}, got {}",
                self.num_clusters
            ));
        }
        self.num_clusters
            .checked_mul(self.byte_len())
            .ok_or_else(|| "binary IVF centroid size overflow".to_string())?;
        self.num_clusters
            .checked_mul(self.dim_bits)
            .ok_or_else(|| "binary IVF training scratch size overflow".to_string())?;
        if !(1..=MAX_HIERARCHICAL_PARENT_RESTARTS).contains(&self.hierarchical_parent_restarts) {
            return Err(format!(
                "binary IVF hierarchical parent restarts must be in 1..={MAX_HIERARCHICAL_PARENT_RESTARTS}, got {}",
                self.hierarchical_parent_restarts
            ));
        }
        Ok(())
    }
}

/// One cluster: SoA layout with contiguous packed codes for SIMD scanning.
#[derive(Debug, Clone, Default)]
pub(crate) struct BinaryCluster {
    pub(crate) doc_ids: Vec<u32>,
    pub(crate) ordinals: Vec<u16>,
    /// Packed codes, `byte_len` bytes per entry, contiguous
    pub(crate) codes: Vec<u8>,
}

#[cfg(test)]
fn visit_binary_cluster(
    cluster: &BinaryCluster,
    dim_bits: usize,
    query: &[u8],
    scores: &mut [f32],
    visit: &mut impl FnMut(u32, u16, f32),
) {
    let byte_len = dim_bits.div_ceil(8);
    let count = cluster.doc_ids.len();
    for batch_start in (0..count).step_by(BINARY_IVF_SCORE_BATCH) {
        let batch_count = BINARY_IVF_SCORE_BATCH.min(count - batch_start);
        let code_start = batch_start * byte_len;
        let code_end = (batch_start + batch_count) * byte_len;
        batch_hamming_scores(
            query,
            &cluster.codes[code_start..code_end],
            byte_len,
            dim_bits,
            &mut scores[..batch_count],
        );
        for (batch_idx, &score) in scores.iter().enumerate().take(batch_count) {
            let i = batch_start + batch_idx;
            visit(cluster.doc_ids[i], cluster.ordinals[i], score);
        }
    }
}

/// Centroid-free binary IVF payload for one segment. The global quantizer is
/// loaded once at index scope; segments only retain exact codes partitioned by
/// leaf ID, making compatible merges O(number of non-empty clusters).
#[derive(Debug, Clone)]
pub struct BinaryIvfIndex {
    pub dim_bits: usize,
    pub quantizer_version: u64,
    pub num_clusters: u32,
    /// Sorted non-empty `(leaf_id, payload)` pairs. Empty cells cost no
    /// per-segment heap memory even when the global codebook has millions of
    /// leaves.
    pub(crate) clusters: Vec<(u32, BinaryCluster)>,
    len: usize,
    /// Indexed codes that carry no information (all bits clear).
    zero_codes: usize,
    /// Indexed codes that carry no information (all bits set).
    ones_codes: usize,
}

/// Streaming build state used by vector-generation rewrites. Only the exact
/// compressed output plus one bounded assignment batch is resident; source
/// flat vectors are never accumulated in memory.
pub(crate) struct BinaryIvfBuilder {
    dim_bits: usize,
    quantizer_version: u64,
    num_clusters: u32,
    routing: IvfRoutingMode,
    clusters: rustc_hash::FxHashMap<u32, BinaryCluster>,
    len: usize,
    zero_codes: usize,
    ones_codes: usize,
}

impl BinaryIvfBuilder {
    pub(crate) fn new(
        quantizer: &BinaryCoarseQuantizer,
        routing: IvfRoutingMode,
    ) -> io::Result<Self> {
        quantizer
            .validate_routing(routing)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        Ok(Self {
            dim_bits: quantizer.dim_bits,
            quantizer_version: quantizer.version,
            num_clusters: quantizer.num_clusters,
            routing,
            clusters: rustc_hash::FxHashMap::default(),
            len: 0,
            zero_codes: 0,
            ones_codes: 0,
        })
    }

    pub(crate) fn add_batch(
        &mut self,
        quantizer: &BinaryCoarseQuantizer,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
    ) -> io::Result<()> {
        if quantizer.dim_bits != self.dim_bits
            || quantizer.version != self.quantizer_version
            || quantizer.num_clusters != self.num_clusters
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary IVF build batch uses a different quantizer generation",
            ));
        }
        let byte_len = quantizer.byte_len();
        let expected = doc_id_ordinals.len().checked_mul(byte_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "binary IVF batch overflows")
        })?;
        if codes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary IVF code/label batch is inconsistent",
            ));
        }
        #[cfg(feature = "native")]
        let assignments: Vec<u32> = {
            use rayon::prelude::*;
            codes
                .par_chunks_exact(byte_len)
                .map(|code| quantizer.assign(code, self.routing))
                .collect::<io::Result<Vec<u32>>>()?
        };
        #[cfg(not(feature = "native"))]
        let assignments: Vec<u32> = codes
            .chunks_exact(byte_len)
            .map(|code| quantizer.assign(code, self.routing))
            .collect::<io::Result<Vec<u32>>>()?;

        // Insert grouped by leaf: one map lookup and one reservation per
        // distinct leaf in the batch instead of per code. Sorting by
        // `(cluster, index)` keeps each leaf's entries in ascending batch order,
        // so the serialized payload is byte-identical to per-code insertion.
        for code in codes.chunks_exact(byte_len) {
            if is_zero_code(code) {
                self.zero_codes += 1;
            } else if is_ones_code(code) {
                self.ones_codes += 1;
            }
        }
        let mut order: Vec<u32> = (0..assignments.len() as u32).collect();
        order.sort_unstable_by_key(|&index| (assignments[index as usize], index));
        let mut run_start = 0usize;
        while run_start < order.len() {
            let cluster_id = assignments[order[run_start] as usize];
            let mut run_end = run_start + 1;
            while run_end < order.len() && assignments[order[run_end] as usize] == cluster_id {
                run_end += 1;
            }
            let run = &order[run_start..run_end];
            let cluster = self.clusters.entry(cluster_id).or_default();
            cluster.doc_ids.reserve(run.len());
            cluster.ordinals.reserve(run.len());
            cluster.codes.reserve(run.len() * byte_len);
            for &index in run {
                let index = index as usize;
                let (doc_id, ordinal) = doc_id_ordinals[index];
                cluster.doc_ids.push(doc_id);
                cluster.ordinals.push(ordinal);
                cluster
                    .codes
                    .extend_from_slice(&codes[index * byte_len..(index + 1) * byte_len]);
            }
            run_start = run_end;
        }
        self.len = self.len.checked_add(order.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "binary IVF count overflows")
        })?;
        Ok(())
    }

    pub(crate) fn finish(self) -> io::Result<BinaryIvfIndex> {
        let mut clusters: Vec<_> = self.clusters.into_iter().collect();
        clusters.sort_unstable_by_key(|(cluster_id, _)| *cluster_id);
        let index = BinaryIvfIndex {
            dim_bits: self.dim_bits,
            quantizer_version: self.quantizer_version,
            num_clusters: self.num_clusters,
            clusters,
            len: self.len,
            zero_codes: self.zero_codes,
            ones_codes: self.ones_codes,
        };
        index
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(index)
    }
}

impl BinaryIvfIndex {
    pub fn build(
        quantizer: &BinaryCoarseQuantizer,
        routing: IvfRoutingMode,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
    ) -> io::Result<Self> {
        let mut builder = BinaryIvfBuilder::new(quantizer, routing)?;
        builder.add_batch(quantizer, codes, doc_id_ordinals)?;
        builder.finish()
    }

    fn validate(&self) -> Result<(), String> {
        if self.dim_bits == 0 || !self.dim_bits.is_multiple_of(8) || self.num_clusters == 0 {
            return Err("invalid global binary IVF metadata".to_string());
        }
        let byte_len = self.dim_bits.div_ceil(8);
        let mut total = 0usize;
        let mut previous = None;
        for (cluster_id, cluster) in &self.clusters {
            if *cluster_id >= self.num_clusters || previous.is_some_and(|id| id >= *cluster_id) {
                return Err("global binary IVF cluster IDs are invalid or unsorted".to_string());
            }
            previous = Some(*cluster_id);
            let count = cluster.doc_ids.len();
            if cluster.ordinals.len() != count
                || cluster.codes.len() != count.saturating_mul(byte_len)
            {
                return Err("global binary IVF cluster columns are inconsistent".to_string());
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| "global binary IVF vector count overflow".to_string())?;
        }
        if total != self.len {
            return Err("global binary IVF vector count is inconsistent".to_string());
        }
        Ok(())
    }

    /// Score the requested clusters of the in-memory build product.
    ///
    /// Queries never reach this: they scan the mmap-backed `AnnDiskIndex` this
    /// structure serialises into. It exists so build-side tests can assert the
    /// partitioning directly, and is therefore test-only — production scanning
    /// lives in `ann_disk::score_binary_cluster_runs`.
    #[cfg(test)]
    pub fn search_in_clusters(
        &self,
        query: &[u8],
        k: usize,
        cluster_ids: &[u32],
    ) -> Vec<(u32, u16, f32)> {
        let mut collector = super::BoundedAnnCollector::<false, true>::new(k);
        if query.len() == self.dim_bits.div_ceil(8) && self.len > 0 {
            let mut scores = vec![0.0; BINARY_IVF_SCORE_BATCH.min(self.len)];
            for &cluster_id in cluster_ids {
                if let Ok(position) = self
                    .clusters
                    .binary_search_by_key(&cluster_id, |(id, _)| *id)
                {
                    visit_binary_cluster(
                        &self.clusters[position].1,
                        self.dim_bits,
                        query,
                        &mut scores,
                        &mut |doc_id, ordinal, score| collector.insert(doc_id, ordinal, score),
                    );
                }
            }
        }
        collector.into_sorted_results()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Indexed codes that were all-zero.
    pub fn zero_codes(&self) -> usize {
        self.zero_codes
    }

    /// Indexed codes that were all-ones.
    pub fn ones_codes(&self) -> usize {
        self.ones_codes
    }

    /// Largest leaf as `(cluster_id, count)`, for skew reporting.
    pub fn largest_cluster(&self) -> Option<(u32, usize)> {
        self.clusters
            .iter()
            .map(|(cluster_id, cluster)| (*cluster_id, cluster.doc_ids.len()))
            .max_by_key(|&(cluster_id, count)| (count, std::cmp::Reverse(cluster_id)))
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        self.clusters
            .iter()
            .map(|(_, cluster)| cluster.codes.len() + cluster.doc_ids.len() * 6)
            .sum()
    }
}

/// Report build-quality problems for one segment's binary payload.
///
/// Both are silent-degradation modes that cost search latency and recall long
/// before anything looks broken, so they are surfaced at every build, merge and
/// rewrite rather than left to be discovered from disk.
#[cfg(feature = "native")]
pub(crate) fn report_binary_build_quality(
    index_label: &str,
    field_id: u32,
    index: &BinaryIvfIndex,
) {
    let indexed = index.len();
    let zero = index.zero_codes();
    if zero > 0 {
        log::warn!(
            "[binary_ivf] index={index_label} field={field_id}: {zero} of {indexed} indexed \
             vectors ({:.1}%) are all-zero — they match nothing, collapse into a single leaf, \
             and every query probing that leaf scans them; check the embedding producer",
            100.0 * zero as f64 / indexed.max(1) as f64,
        );
    }
    let ones = index.ones_codes();
    if ones > 0 {
        log::warn!(
            "[binary_ivf] index={index_label} field={field_id}: {ones} of {indexed} indexed \
             vectors ({:.1}%) are all-ones — the saturated twin of the zero code (NaN packed \
             through a signbit test): they match nothing, collapse into a single leaf, and \
             every query probing that leaf scans them; check the embedding producer",
            100.0 * ones as f64 / indexed.max(1) as f64,
        );
    }
    if let Some((cluster_id, count)) = index.largest_cluster()
        && count >= LEAF_SKEW_WARN_MINIMUM
        && index.num_clusters > 1
        && count.saturating_mul(index.num_clusters as usize)
            > indexed.saturating_mul(LEAF_SKEW_WARN_RATIO)
    {
        log::warn!(
            "[binary_ivf] index={index_label} field={field_id}: leaf {cluster_id} holds {count}              of {indexed} vectors ({:.1}%, {:.0}x the average) — every query probing it scans              that leaf in full",
            100.0 * count as f64 / indexed.max(1) as f64,
            count as f64 * index.num_clusters as f64 / indexed.max(1) as f64,
        );
    }
}

/// Lloyd-style k-majority clustering in Hamming space.
fn train_k_majority(
    config: &BinaryIvfConfig,
    codes: &[u8],
    n: usize,
    index_label: &str,
) -> Vec<u8> {
    train_k_majority_reporting(config, codes, n, index_label, true)
}

/// Train one flat packed-bit k-majority codebook for another global routing
/// implementation.
///
/// ScaNN's binary tree deliberately reuses the same deterministic trainer as
/// binary IVF. Keeping this narrow adapter here prevents a second Hamming
/// clustering implementation (and, critically, any conversion through
/// floating-point vectors).
pub(crate) fn train_binary_k_majority_codebook(
    config: &BinaryIvfConfig,
    codes: &[u8],
    num_vectors: usize,
    index_label: &str,
) -> io::Result<Vec<u8>> {
    config
        .validate()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let expected = num_vectors.checked_mul(config.byte_len()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "binary k-majority training size overflow",
        )
    })?;
    if num_vectors == 0 || config.num_clusters > num_vectors || codes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "binary k-majority needs a contiguous matrix with at least one vector per centroid",
        ));
    }
    Ok(train_k_majority_reporting(
        config,
        codes,
        num_vectors,
        index_label,
        false,
    ))
}

/// As [`train_k_majority`], with progress reporting suppressed for the hundreds
/// of small child codebooks a hierarchical build trains.
fn train_k_majority_reporting(
    config: &BinaryIvfConfig,
    codes: &[u8],
    n: usize,
    index_label: &str,
    report: bool,
) -> Vec<u8> {
    let byte_len = config.byte_len();
    let k = config.num_clusters;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    // Seeding and Lloyd assignment together dominate training; resolve the
    // architecture kernel once instead of re-detecting CPU features for every
    // one of the O(N*K) code pairs.
    let kernel = HammingKernel::resolve();

    // Bound training cost: iterate over a sample, assign everything later. Do
    // not materialize a random permutation when the complete input is used:
    // the indirection destroys locality for the billion-scale global training
    // sample without changing which points participate.
    let sample_len = config.max_train_samples.max(k).min(n);
    let sample =
        (sample_len < n).then(|| rand::seq::index::sample(&mut rng, n, sample_len).into_vec());
    let n = sample_len;
    let vec_at = |i: usize| -> &[u8] {
        let vi = sample.as_ref().map_or(i, |sample| sample[i]);
        &codes[vi * byte_len..(vi + 1) * byte_len]
    };

    let mut centroids = vec![0u8; k * byte_len];
    // k-means++ seeding in Hamming space. Random-first-k is especially prone
    // to duplicate/near-duplicate cells on skewed embedding distributions.
    let first = rng.random_range(0..n);
    centroids[..byte_len].copy_from_slice(vec_at(first));
    let mut minimum_weights = vec![f64::INFINITY; n];
    // Seeding is one full pass over the sample per centroid, so it is O(N*K)
    // and by far the longest silent stretch of a large codebook.
    let mut seeding = PhaseProgress::start_if(
        report,
        index_label,
        "k-majority seeding",
        format!("{k} centroids over {n} samples"),
        k,
    );
    for centroid_id in 1..k {
        seeding.advance(centroid_id);
        let previous = &centroids[(centroid_id - 1) * byte_len..centroid_id * byte_len];
        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            minimum_weights
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, minimum_weight)| {
                    let distance = kernel.distance(vec_at(index), previous);
                    // Hamming distance is already squared Euclidean distance
                    // for vectors in {0,1}^d, so D² sampling uses it directly.
                    *minimum_weight = minimum_weight.min(distance as f64);
                });
        }
        #[cfg(not(feature = "native"))]
        for (index, minimum_weight) in minimum_weights.iter_mut().enumerate() {
            let distance = kernel.distance(vec_at(index), previous);
            *minimum_weight = minimum_weight.min(distance as f64);
        }

        let chosen = if let Some(chosen) = crate::structures::vector::kmeans::weighted_sample_index(
            &minimum_weights,
            rng.random::<f64>(),
        ) {
            chosen
        } else {
            rng.random_range(0..n)
        };
        centroids[centroid_id * byte_len..(centroid_id + 1) * byte_len]
            .copy_from_slice(vec_at(chosen));
    }

    seeding.finish();

    let mut assignment = vec![u32::MAX; n];
    let mut assignment_distances = vec![u32::MAX; n];
    let mut members: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();

    let mut lloyd = PhaseProgress::start_if(
        report,
        index_label,
        "k-majority refinement",
        format!(
            "{k} centroids, {n} samples, <= {} iters",
            config.train_iters
        ),
        config.train_iters,
    );
    for _iter in 0..config.train_iters {
        lloyd.advance(_iter);
        // Assignment is point-independent. Give every rayon worker its own
        // score buffer so the O(N*K) scan uses all available training CPUs
        // without synchronization or per-point allocation.
        #[cfg(feature = "native")]
        let changed: usize = {
            use rayon::prelude::*;
            assignment
                .par_iter_mut()
                .zip(assignment_distances.par_iter_mut())
                .enumerate()
                .map_init(
                    || vec![0u32; k],
                    |distances, (i, (slot, assignment_distance))| {
                        let (best, distance) = nearest_binary_centroid_with_distance(
                            kernel,
                            vec_at(i),
                            &centroids,
                            byte_len,
                            distances,
                        );
                        *assignment_distance = distance;
                        usize::from(std::mem::replace(slot, best) != best)
                    },
                )
                .sum()
        };
        #[cfg(not(feature = "native"))]
        let changed: usize = {
            let mut distances = vec![0u32; k];
            assignment
                .iter_mut()
                .zip(&mut assignment_distances)
                .enumerate()
                .map(|(i, (slot, assignment_distance))| {
                    let (best, distance) = nearest_binary_centroid_with_distance(
                        kernel,
                        vec_at(i),
                        &centroids,
                        byte_len,
                        &mut distances,
                    );
                    *assignment_distance = distance;
                    usize::from(std::mem::replace(slot, best) != best)
                })
                .sum()
        };
        if changed == 0 {
            break;
        }

        // Group point IDs once, in input order. Updating one centroid per task
        // is both deterministic and much more cache-friendly than writing a
        // K*D counter matrix at a data-dependent cluster offset for every bit.
        for cluster_members in &mut members {
            cluster_members.clear();
        }
        for (i, &slot) in assignment.iter().enumerate().take(n) {
            members[slot as usize].push(i);
        }

        reseed_empty_binary_centroids(
            &mut centroids,
            &mut members,
            &mut assignment,
            &mut assignment_distances,
            sample.as_deref(),
            codes,
            byte_len,
        );

        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            centroids
                .par_chunks_mut(byte_len)
                .zip(members.par_iter())
                .filter(|(_, cluster_members)| !cluster_members.is_empty())
                .for_each_init(
                    BinaryMajorityScratch::default,
                    |scratch, (centroid, cluster_members)| {
                        update_binary_centroid(
                            centroid,
                            cluster_members,
                            sample.as_deref(),
                            codes,
                            byte_len,
                            scratch,
                        );
                    },
                );
        }
        #[cfg(not(feature = "native"))]
        {
            let mut scratch = BinaryMajorityScratch::default();
            for (centroid, cluster_members) in centroids.chunks_mut(byte_len).zip(&members) {
                if !cluster_members.is_empty() {
                    update_binary_centroid(
                        centroid,
                        cluster_members,
                        sample.as_deref(),
                        codes,
                        byte_len,
                        &mut scratch,
                    );
                }
            }
        }
    }
    lloyd.finish();

    centroids
}

/// Re-seed preference: farthest assignment distance first, then lowest point ID.
type ReseedPriority = (u32, Reverse<usize>);

/// Re-seed empty cells from the represented points with the largest assignment
/// error. One closest point is retained in every donor cell, so each selected
/// point is unique and re-assignment cannot create a new empty cell. Empty IDs
/// and equal-distance candidates are resolved in ascending order.
fn reseed_empty_binary_centroids(
    centroids: &mut [u8],
    members: &mut [Vec<usize>],
    assignment: &mut [u32],
    assignment_distances: &mut [u32],
    sample: Option<&[usize]>,
    codes: &[u8],
    byte_len: usize,
) {
    let empty_clusters: Vec<usize> = members
        .iter()
        .enumerate()
        .filter_map(|(cluster, cluster_members)| cluster_members.is_empty().then_some(cluster))
        .collect();
    if empty_clusters.is_empty() {
        return;
    }

    // Retain the closest (then lowest-index) represented point in each donor
    // cell. Every other point is safe to move without emptying its donor.
    //
    // Only `empty_clusters.len()` of those points are ever used, so selection
    // runs through a bounded heap: the previous full candidate list was an
    // allocation proportional to the whole training sample, on every Lloyd
    // iteration that had even one empty cell.
    let wanted = empty_clusters.len();
    let priority =
        |point: usize| -> ReseedPriority { (assignment_distances[point], Reverse(point)) };
    // The heap keeps the *least* preferred survivor on top, so a bounded push/pop
    // retains exactly the `wanted` most preferred points.
    let mut selected: BinaryHeap<Reverse<ReseedPriority>> = BinaryHeap::with_capacity(wanted + 1);
    for cluster_members in members.iter().filter(|members| !members.is_empty()) {
        let keeper = *cluster_members
            .iter()
            .min_by_key(|&&point| (assignment_distances[point], point))
            .expect("non-empty binary cluster must have a represented point");
        for &point in cluster_members.iter().filter(|&&point| point != keeper) {
            selected.push(Reverse(priority(point)));
            if selected.len() > wanted {
                selected.pop();
            }
        }
    }
    debug_assert!(selected.len() >= wanted.min(selected.capacity()));

    let mut candidates: Vec<usize> = selected
        .into_iter()
        .map(|Reverse((_, Reverse(point)))| point)
        .collect();
    candidates.sort_unstable_by_key(|&point| Reverse(priority(point)));

    for (&empty_cluster, &point) in empty_clusters.iter().zip(candidates.iter()) {
        let donor = assignment[point] as usize;
        assignment[point] = empty_cluster as u32;
        assignment_distances[point] = 0;
        let vector_index = sample.map_or(point, |sample| sample[point]);
        centroids[empty_cluster * byte_len..(empty_cluster + 1) * byte_len]
            .copy_from_slice(&codes[vector_index * byte_len..(vector_index + 1) * byte_len]);

        // Keep the membership view consistent with the assignments before the
        // centroid-update phase: a donor cell must not retain a point that
        // moved into a newly seeded cell. Only moved points are touched, so
        // this costs one donor scan each instead of rebuilding every cell.
        if let Some(position) = members[donor].iter().position(|&member| member == point) {
            members[donor].remove(position);
        }
        members[empty_cluster].push(point);
    }
}

/// Per-worker counters for Hamming-majority centroid updates.
///
/// Counts live as eight bit-planes of one byte-counter per code byte, so the
/// accumulation loop is a flat `u8` add over a contiguous plane and vectorises.
/// The narrow planes are folded into wide counters before they can overflow,
/// which keeps one 20 KiB-scale allocation per worker instead of one per
/// centroid per Lloyd iteration.
#[derive(Default)]
struct BinaryMajorityScratch {
    planes: Vec<u8>,
    counts: Vec<u32>,
}

/// Members accumulated into `u8` planes before folding. A plane slot counts one
/// bit per member, so 255 members is the overflow bound.
const MAJORITY_PLANE_FLUSH: usize = 255;

impl BinaryMajorityScratch {
    fn reset(&mut self, byte_len: usize) {
        let slots = byte_len * 8;
        self.planes.clear();
        self.planes.resize(slots, 0);
        self.counts.clear();
        self.counts.resize(slots, 0);
    }

    #[inline]
    fn accumulate(&mut self, vector: &[u8], byte_len: usize) {
        for bit in 0..8 {
            let plane = &mut self.planes[bit * byte_len..(bit + 1) * byte_len];
            for (slot, &byte) in plane.iter_mut().zip(vector) {
                *slot += (byte >> bit) & 1;
            }
        }
    }

    #[inline]
    fn fold(&mut self) {
        for (count, plane) in self.counts.iter_mut().zip(self.planes.iter_mut()) {
            *count += u32::from(*plane);
            *plane = 0;
        }
    }

    #[inline]
    fn count(&self, bit: usize, byte_index: usize, byte_len: usize) -> u32 {
        self.counts[bit * byte_len + byte_index]
    }
}

/// Compute one Hamming-space centroid using a per-byte bit histogram. Member
/// IDs arrive in input order, so the result is deterministic regardless of
/// which centroids rayon updates concurrently.
fn update_binary_centroid(
    centroid: &mut [u8],
    members: &[usize],
    sample: Option<&[usize]>,
    codes: &[u8],
    byte_len: usize,
    scratch: &mut BinaryMajorityScratch,
) {
    scratch.reset(byte_len);
    for (position, &member) in members.iter().enumerate() {
        let vector_index = sample.map_or(member, |sample| sample[member]);
        let vector = &codes[vector_index * byte_len..(vector_index + 1) * byte_len];
        scratch.accumulate(vector, byte_len);
        if (position + 1).is_multiple_of(MAJORITY_PLANE_FLUSH) {
            scratch.fold();
        }
    }
    scratch.fold();

    let total = members.len() as u32;
    for (byte_index, byte) in centroid.iter_mut().enumerate() {
        let previous = *byte;
        let mut packed = 0u8;
        for bit in 0..8 {
            let ones = scratch.count(bit, byte_index, byte_len);
            let value = match ones.cmp(&(total - ones)) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => 0,
                // Majority ties keep the previous bit rather than biasing to 0.
                std::cmp::Ordering::Equal => (previous >> bit) & 1,
            };
            packed |= value << bit;
        }
        *byte = packed;
    }
}

/// Exact total Hamming assignment loss for deterministic parent-model
/// selection. Summation is integer and therefore independent of rayon task
/// ordering.
fn binary_codebook_loss(codes: &[u8], centroids: &[u8], byte_len: usize) -> u64 {
    debug_assert!(byte_len > 0 && codes.len().is_multiple_of(byte_len));
    let clusters = centroids.len() / byte_len;
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        codes
            .par_chunks_exact(byte_len)
            .map_init(
                || vec![0u32; clusters],
                |distances, code| {
                    u64::from(
                        nearest_binary_centroid_with_distance(
                            HammingKernel::resolve(),
                            code,
                            centroids,
                            byte_len,
                            distances,
                        )
                        .1,
                    )
                },
            )
            .sum()
    }
    #[cfg(not(feature = "native"))]
    {
        let kernel = HammingKernel::resolve();
        let mut distances = vec![0u32; clusters];
        codes
            .chunks_exact(byte_len)
            .map(|code| {
                u64::from(
                    nearest_binary_centroid_with_distance(
                        kernel,
                        code,
                        centroids,
                        byte_len,
                        &mut distances,
                    )
                    .1,
                )
            })
            .sum()
    }
}

/// Hierarchical k-majority training keeps large global codebooks tractable:
/// train sqrt(K) parent cells, partition the sample once, then train each
/// child codebook independently. Complexity is O(N·sqrt(K)) rather than
/// O(N·K), while leaf centroids remain ordinary Hamming-majority centroids.
fn train_k_majority_hierarchical(
    config: &BinaryIvfConfig,
    codes: &[u8],
    n: usize,
    index_label: &str,
) -> (Vec<u8>, BinaryCentroidRouter) {
    let byte_len = config.byte_len();
    let parent_count = routing_parent_count(config.num_clusters).min(n);
    let mut parent_config = config.clone();
    parent_config.num_clusters = parent_count;
    parent_config.max_train_samples = config.max_train_samples.min(n);
    let mut selected_parent: Option<(u64, Vec<u8>, usize)> = None;
    for restart in 0..config.hierarchical_parent_restarts {
        parent_config.seed = if restart == 0 {
            config.seed
        } else {
            config
                .seed
                .wrapping_add((restart as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        };
        let candidate = train_k_majority(&parent_config, codes, n, index_label);
        let loss = binary_codebook_loss(codes, &candidate, byte_len);
        let replace = match selected_parent.as_ref() {
            None => true,
            Some((best_loss, best_centroids, _)) => {
                loss < *best_loss || (loss == *best_loss && candidate < *best_centroids)
            }
        };
        if replace {
            selected_parent = Some((loss, candidate, restart));
        }
    }
    let (parent_loss, parents, selected_restart) =
        selected_parent.expect("validated binary IVF has at least one parent restart");
    log::info!(
        "[binary_ivf] index={index_label}: selected hierarchical parent restart {selected_restart}/{} with exact Hamming loss {parent_loss}",
        config.hierarchical_parent_restarts,
    );

    let mut assignments = vec![0u32; n];
    let mut group_sizes = vec![0usize; parent_count];
    let kernel = HammingKernel::resolve();
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        assignments.par_iter_mut().enumerate().for_each_init(
            || vec![0u32; parent_count],
            |distances, (index, assignment)| {
                *assignment = nearest_binary_centroid_with_distance(
                    kernel,
                    &codes[index * byte_len..(index + 1) * byte_len],
                    &parents,
                    byte_len,
                    distances,
                )
                .0;
            },
        );
    }
    #[cfg(not(feature = "native"))]
    {
        let mut distances = vec![0u32; parent_count];
        for (index, assignment) in assignments.iter_mut().enumerate() {
            *assignment = nearest_binary_centroid_with_distance(
                kernel,
                &codes[index * byte_len..(index + 1) * byte_len],
                &parents,
                byte_len,
                &mut distances,
            )
            .0;
        }
    }
    for &assignment in &assignments {
        group_sizes[assignment as usize] += 1;
    }
    let child_counts = allocate_child_clusters(&group_sizes, config.num_clusters);
    let mut groups: Vec<Vec<u8>> = group_sizes
        .iter()
        .map(|&size| Vec::with_capacity(size.saturating_mul(byte_len)))
        .collect();
    for (index, &assignment) in assignments.iter().enumerate() {
        groups[assignment as usize]
            .extend_from_slice(&codes[index * byte_len..(index + 1) * byte_len]);
    }
    drop(assignments);

    // Child codebooks share nothing: each trains from its own parent's group
    // with its own derived seed. Training them concurrently is the level that
    // actually fills the machine — one child covers only a few tens of
    // thousands of codes, far too little work for its internal rayon loops to
    // saturate a 48-core box, so a sequential loop over parents left most of
    // the machine idle for the bulk of a large codebook's build.
    let populated = child_counts.iter().filter(|&&count| count > 0).count();
    let child_phase = PhaseProgress::start(
        index_label,
        "child codebooks",
        format!("{populated} parents -> {} leaves", config.num_clusters),
        populated,
    );
    let shared = child_phase.shared();
    let train_child = |(parent, (group, &child_count)): (usize, (&Vec<u8>, &usize))| -> Vec<u8> {
        if child_count == 0 {
            return Vec::new();
        }
        let mut child_config = config.clone();
        child_config.num_clusters = child_count;
        child_config.max_train_samples = group.len() / byte_len;
        child_config.seed = config.seed ^ (parent as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let trained = train_k_majority_reporting(
            &child_config,
            group,
            group.len() / byte_len,
            index_label,
            false,
        );
        shared.complete_one();
        trained
    };
    #[cfg(feature = "native")]
    let trained_children: Vec<Vec<u8>> = {
        use rayon::prelude::*;
        groups
            .par_iter()
            .zip(child_counts.par_iter())
            .enumerate()
            .map(train_child)
            .collect()
    };
    #[cfg(not(feature = "native"))]
    let trained_children: Vec<Vec<u8>> = groups
        .iter()
        .zip(child_counts.iter())
        .enumerate()
        .map(train_child)
        .collect();
    child_phase.finish();

    // Assemble in parent order, so the leaf matrix is byte-identical to the
    // sequential build and every parent still owns one contiguous leaf run.
    let mut leaves = Vec::with_capacity(config.num_clusters.saturating_mul(byte_len));
    let mut children = vec![Vec::new(); parent_count];
    for (parent, (trained, &child_count)) in trained_children.iter().zip(&child_counts).enumerate()
    {
        if child_count == 0 {
            continue;
        }
        let first_leaf = leaves.len() / byte_len;
        leaves.extend_from_slice(trained);
        children[parent].extend((first_leaf..first_leaf + child_count).map(|leaf| leaf as u32));
    }
    debug_assert_eq!(leaves.len(), config.num_clusters * byte_len);
    (
        leaves,
        BinaryCentroidRouter::TwoLevel {
            parent_centroids: parents,
            topology: IvfRoutingTopology::from_children(&children),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained_index(
        dim_bits: usize,
        clusters: usize,
        codes: &[u8],
        labels: &[(u32, u16)],
    ) -> (BinaryCoarseQuantizer, BinaryIvfIndex) {
        let mut config = BinaryIvfConfig::new(dim_bits, clusters);
        config.train_iters = 4;
        config.max_train_samples = labels.len();
        let quantizer = BinaryCoarseQuantizer::train(config, codes, labels.len(), "test").unwrap();
        let index = BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, codes, labels).unwrap();
        (quantizer, index)
    }

    #[test]
    fn full_probe_matches_exact_hamming_and_preserves_ties() {
        let dim = 64;
        let byte_len = dim / 8;
        let n = 300;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let codes: Vec<u8> = (0..n * byte_len).map(|_| rng.random()).collect();
        let labels: Vec<_> = (0..n as u32).map(|doc_id| (doc_id, 0)).collect();
        let query: Vec<u8> = (0..byte_len).map(|_| rng.random()).collect();
        let (quantizer, index) = trained_index(dim, 8, &codes, &labels);
        let plan = quantizer.probe(&query, 8, IvfRoutingMode::Flat).unwrap();
        let actual = index.search_in_clusters(&query, 20, &plan.cluster_ids);

        let mut scores = vec![0.0; n];
        batch_hamming_scores(&query, &codes, byte_len, dim, &mut scores);
        let mut expected: Vec<_> = scores
            .into_iter()
            .enumerate()
            .map(|(doc_id, score)| (doc_id as u32, 0, score))
            .collect();
        expected.sort_unstable_by(|left, right| {
            right
                .2
                .total_cmp(&left.2)
                .then_with(|| left.0.cmp(&right.0))
        });
        expected.truncate(20);
        assert_eq!(actual, expected);
    }

    /// Build a clustered binary corpus: `groups` random anchors, each with
    /// `per_group` codes at a small Hamming radius. Real embedding corpora are
    /// clustered, and uniform noise would make every routing budget look alike.
    fn clustered_binary_codes(
        byte_len: usize,
        groups: usize,
        per_group: usize,
        flips: usize,
        seed: u64,
    ) -> Vec<u8> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut codes = Vec::with_capacity(groups * per_group * byte_len);
        for _ in 0..groups {
            let mut anchor = vec![0u8; byte_len];
            rng.fill_bytes(&mut anchor);
            for _ in 0..per_group {
                let mut code = anchor.clone();
                for _ in 0..flips {
                    let bit = rng.random_range(0..byte_len * 8);
                    code[bit / 8] ^= 1 << (bit % 8);
                }
                codes.extend_from_slice(&code);
            }
        }
        codes
    }

    #[test]
    fn binary_parent_beam_adapts_at_large_codebook_boundaries() {
        assert_eq!(adaptive_binary_parent_beam_oversample(4_095), 4);
        assert_eq!(
            adaptive_binary_parent_beam_oversample(4_096),
            BINARY_PARENT_BEAM_OVERSAMPLE_MEDIUM
        );
        assert_eq!(
            adaptive_binary_parent_beam_oversample(65_536),
            BINARY_PARENT_BEAM_OVERSAMPLE_LARGE
        );
        assert!(!uses_hierarchical_binary_training(4_095));
        assert!(uses_hierarchical_binary_training(4_096));
    }

    #[test]
    fn wider_binary_parent_beam_recovers_a_cross_parent_nearest_leaf() {
        let (dim_bits, byte_len, parent_count, leaves_per_parent) = (64, 8, 64, 64);
        let leaves = parent_count * leaves_per_parent;
        let mut parent_centroids = vec![0u8; parent_count * byte_len];
        for parent in 0..parent_count {
            for bit in 0..parent.min(dim_bits) {
                parent_centroids[parent * byte_len + bit / 8] |= 1 << (bit % 8);
            }
        }
        let children: Vec<Vec<u32>> = (0..parent_count)
            .map(|parent| {
                let first = parent * leaves_per_parent;
                (first..first + leaves_per_parent)
                    .map(|leaf| leaf as u32)
                    .collect()
            })
            .collect();
        let target = 3 * leaves_per_parent;
        let mut centroids = vec![0xff; leaves * byte_len];
        centroids[target * byte_len..(target + 1) * byte_len].fill(0);
        let quantizer = BinaryCoarseQuantizer {
            dim_bits,
            num_clusters: leaves as u32,
            centroids,
            version: 1,
            routing_index: Some(BinaryCentroidRouter::TwoLevel {
                parent_centroids,
                topology: IvfRoutingTopology::from_children(&children),
            }),
        };
        quantizer.validate().unwrap();

        let query = [0u8; 8];
        let narrow = quantizer
            .probe_with_parent_beam(&query, 32, IvfRoutingMode::TwoLevel, 4)
            .unwrap();
        let wide = quantizer
            .probe_with_parent_beam(&query, 32, IvfRoutingMode::TwoLevel, 8)
            .unwrap();
        assert!(!narrow.cluster_ids.contains(&(target as u32)));
        assert_eq!(wide.cluster_ids.first(), Some(&(target as u32)));
        assert_eq!(wide.cluster_ids.len(), 32);
        assert_ne!(narrow.request_fingerprint, wide.request_fingerprint);

        let default_plan = quantizer.probe(&query, 32, IvfRoutingMode::Auto).unwrap();
        assert_eq!(
            default_plan.request_fingerprint,
            quantizer.request_fingerprint(&query, 32, IvfRoutingMode::Auto),
            "reader-side cache lookup must use the resolved adaptive policy"
        );
    }

    #[test]
    fn deterministic_parent_restarts_cannot_worsen_exact_hamming_loss() {
        let dim_bits = 64;
        let byte_len = dim_bits / 8;
        let codes = clustered_binary_codes(byte_len, 32, 16, 5, 0x55aa_0102);
        let points = codes.len() / byte_len;
        let train = |restarts| {
            let mut config = BinaryIvfConfig::new(dim_bits, 64);
            config.routing = IvfRoutingMode::TwoLevel;
            config.train_iters = 2;
            config.max_train_samples = points;
            config.hierarchical_parent_restarts = restarts;
            BinaryCoarseQuantizer::train(config, &codes, points, "test").unwrap()
        };
        let single = train(1);
        let multi = train(3);
        let parent_loss = |quantizer: &BinaryCoarseQuantizer| {
            let Some(BinaryCentroidRouter::TwoLevel {
                parent_centroids, ..
            }) = quantizer.routing_index.as_ref()
            else {
                panic!("two-level parent router expected");
            };
            binary_codebook_loss(&codes, parent_centroids, byte_len)
        };
        assert!(parent_loss(&multi) <= parent_loss(&single));

        let repeated = train(3);
        assert_eq!(multi.centroids, repeated.centroids);
        let (
            Some(BinaryCentroidRouter::TwoLevel {
                parent_centroids: multi_parents,
                topology: multi_topology,
            }),
            Some(BinaryCentroidRouter::TwoLevel {
                parent_centroids: repeated_parents,
                topology: repeated_topology,
            }),
        ) = (
            multi.routing_index.as_ref(),
            repeated.routing_index.as_ref(),
        )
        else {
            panic!("two-level parent routers expected");
        };
        assert_eq!(multi_parents, repeated_parents);
        assert_eq!(multi_topology, repeated_topology);
    }

    /// The construction beam is a *floor* paid by every assigned vector, so its
    /// value has to be justified by assignment quality rather than by "build can
    /// afford more". Assignment recall against exact centroid scan saturates
    /// well below the floor: widening past it only multiplies the distance work
    /// each rebuilt segment pays per vector.
    ///
    /// Referenced by `HNSW_MIN_EF_BUILD` in `ivf/routing.rs`.
    #[cfg(feature = "native")]
    #[test]
    fn hnsw_build_beam_recall_saturates_before_the_floor() {
        use crate::structures::vector::ivf::routing::HNSW_MIN_EF_BUILD;

        let dim_bits = 256;
        let byte_len = dim_bits / 8;
        let codes = clustered_binary_codes(byte_len, 512, 24, 12, 0x5eed_1234);
        let points = codes.len() / byte_len;

        let mut config = BinaryIvfConfig::new(dim_bits, 4_096);
        config.train_iters = 3;
        config.max_train_samples = points;
        config.routing = IvfRoutingMode::Hnsw;
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, points, "test").unwrap();
        let Some(BinaryCentroidRouter::Hnsw(graph)) = quantizer.routing_index.as_ref() else {
            panic!("HNSW routing expected at 4096 clusters");
        };

        let probes = clustered_binary_codes(byte_len, 64, 4, 14, 0x9ab_cdef);
        let exact: Vec<u32> = probes
            .chunks_exact(byte_len)
            .map(|code| quantizer.find_nearest(code).unwrap())
            .collect();

        let recall_at = |ef: usize| -> f64 {
            let hits = probes
                .chunks_exact(byte_len)
                .zip(exact.iter())
                .filter(|&(code, &want)| {
                    graph.search_best_with_ef(
                        BinaryCentroidDistance::new(code, &quantizer.centroids, byte_len),
                        ef,
                    ) == Some(want)
                })
                .count();
            hits as f64 / exact.len() as f64
        };

        let (narrow, floor, wide) = (recall_at(32), recall_at(HNSW_MIN_EF_BUILD), recall_at(512));

        // Cost side of the same comparison: count centroid distance
        // evaluations, which is deterministic where wall-clock is not.
        struct CountingDistance<'a> {
            inner: BinaryCentroidDistance<'a>,
            calls: &'a std::cell::Cell<usize>,
        }
        impl QueryDistance for CountingDistance<'_> {
            fn distance(&self, node: u32) -> f32 {
                self.calls.set(self.calls.get() + 1);
                self.inner.distance(node)
            }

            fn distances(&self, nodes: &[u32], out: &mut [f32]) {
                self.calls.set(self.calls.get() + nodes.len());
                self.inner.distances(nodes, out);
            }
        }
        let work_at = |ef: usize| -> usize {
            let calls = std::cell::Cell::new(0usize);
            for code in probes.chunks_exact(byte_len) {
                graph.search_best_with_ef(
                    CountingDistance {
                        inner: BinaryCentroidDistance::new(code, &quantizer.centroids, byte_len),
                        calls: &calls,
                    },
                    ef,
                );
            }
            calls.get()
        };
        let (floor_work, wide_work) = (work_at(HNSW_MIN_EF_BUILD), work_at(512));
        println!(
            "assignment recall@1: ef=32 {narrow:.4}, ef={HNSW_MIN_EF_BUILD} {floor:.4}, ef=512 {wide:.4}\n\
             assignment distance evaluations: ef={HNSW_MIN_EF_BUILD} {floor_work}, ef=512 {wide_work} \
             ({:.2}x)",
            wide_work as f64 / floor_work as f64
        );
        assert!(
            wide_work > floor_work,
            "a wider beam must cost more work: {wide_work} vs {floor_work}"
        );

        // The floor must be worth paying for over a clearly narrow beam...
        assert!(
            floor >= narrow,
            "floor beam {HNSW_MIN_EF_BUILD} must not route worse than ef=32: {floor:.4} vs {narrow:.4}"
        );
        // ...and 4x more work than the floor must not be buying recall, which is
        // what makes the previous 512 floor pure overhead.
        assert!(
            wide - floor <= 0.005,
            "ef=512 bought {:.4} recall over ef={HNSW_MIN_EF_BUILD}; the floor is too low",
            wide - floor
        );
    }

    /// Child codebooks train concurrently, so the leaf matrix and topology must
    /// still be byte-identical whatever the pool width — the quantizer is a
    /// shared artifact and every segment's postings are keyed to its leaf IDs.
    #[cfg(feature = "native")]
    #[test]
    fn hierarchical_training_is_deterministic_across_thread_counts() {
        let dim_bits = 64;
        let byte_len = dim_bits / 8;
        let points = 4_000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234_5678);
        let mut codes = vec![0u8; points * byte_len];
        rng.fill_bytes(&mut codes);

        let mut config = BinaryIvfConfig::new(dim_bits, 96);
        config.train_iters = 3;
        config.max_train_samples = points;

        let train = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| train_k_majority_hierarchical(&config, &codes, points, "test"))
        };
        let (one_leaves, one_router) = train(1);
        let (eight_leaves, eight_router) = train(8);

        assert_eq!(one_leaves, eight_leaves, "leaf centroids diverged");
        let (
            BinaryCentroidRouter::TwoLevel { topology: one, .. },
            BinaryCentroidRouter::TwoLevel {
                topology: eight, ..
            },
        ) = (&one_router, &eight_router)
        else {
            panic!("hierarchical training must produce a two-level router");
        };
        assert_eq!(one, eight, "parent topology diverged");
        assert!(one.validate(config.num_clusters), "topology invalid");
    }

    #[cfg(feature = "native")]
    #[test]
    fn k_majority_is_deterministic_across_thread_counts() {
        let dim_bits = 128;
        let byte_len = dim_bits / 8;
        let points = 1_024;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xfeed_cafe);
        let mut codes = vec![0u8; points * byte_len];
        rng.fill_bytes(&mut codes);

        let mut config = BinaryIvfConfig::new(dim_bits, 32);
        config.train_iters = 4;
        config.max_train_samples = points;
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| train_k_majority(&config, &codes, points, "test"));
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| train_k_majority(&config, &codes, points, "test"));

        assert_eq!(one_thread, four_threads);
    }

    #[test]
    fn k_majority_seeding_uses_raw_hamming_d2_weights() {
        let codes = [0x00, 0x01, 0x03, 0x0f, 0x3f, 0x7f, 0xff];
        let (seed, first, raw_choice, squared_choice) = (0u64..10_000)
            .find_map(|seed| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let first = rng.random_range(0..codes.len());
                let draw = rng.random::<f64>();
                let raw_weights: Vec<f64> = codes
                    .iter()
                    .map(|code| {
                        f64::from(hamming_distance(
                            std::slice::from_ref(code),
                            std::slice::from_ref(&codes[first]),
                        ))
                    })
                    .collect();
                let squared_weights: Vec<f64> =
                    raw_weights.iter().map(|weight| weight * weight).collect();
                let raw_choice =
                    crate::structures::vector::kmeans::weighted_sample_index(&raw_weights, draw)?;
                let squared_choice = crate::structures::vector::kmeans::weighted_sample_index(
                    &squared_weights,
                    draw,
                )?;
                (raw_choice != squared_choice).then_some((seed, first, raw_choice, squared_choice))
            })
            .expect("test data must distinguish Hamming from Hamming-squared sampling");

        let mut config = BinaryIvfConfig::new(8, 2);
        config.seed = seed;
        config.train_iters = 0;
        config.max_train_samples = codes.len();
        let centroids = train_k_majority(&config, &codes, codes.len(), "test");

        assert_eq!(centroids[0], codes[first]);
        assert_eq!(centroids[1], codes[raw_choice]);
        assert_ne!(centroids[1], codes[squared_choice]);
    }

    #[test]
    fn empty_binary_centroids_take_farthest_movable_points_deterministically() {
        let codes = [0x0f, 0x01, 0xf0, 0xf1, 0xfe];
        let mut centroids = vec![0x00, 0xff, 0x55, 0xaa];
        let mut members = vec![vec![0, 1], vec![2, 3, 4], vec![], vec![]];
        let mut assignment = vec![0, 0, 1, 1, 1];
        let mut assignment_distances = vec![4, 1, 4, 3, 1];

        reseed_empty_binary_centroids(
            &mut centroids,
            &mut members,
            &mut assignment,
            &mut assignment_distances,
            None,
            &codes,
            1,
        );

        // Points 0 and 2 have the largest movable distance. The equal-distance
        // tie and empty cluster IDs are both resolved from lowest to highest.
        assert_eq!(centroids, vec![0x00, 0xff, 0x0f, 0xf0]);
        assert_eq!(assignment, vec![2, 0, 3, 1, 1]);
        assert_eq!(assignment_distances, vec![0, 1, 0, 3, 1]);
        assert_eq!(members, vec![vec![1], vec![3, 4], vec![0], vec![2]]);
    }

    #[test]
    fn binary_centroid_majority_ties_keep_previous_bits() {
        let mut scratch = BinaryMajorityScratch::default();
        let codes = [0b1010_1010, 0b0101_0101];
        let mut centroid = [0b1100_0011];
        update_binary_centroid(&mut centroid, &[0, 1], None, &codes, 1, &mut scratch);
        assert_eq!(centroid, [0b1100_0011]);

        let decisive_codes = [0b1111_0000, 0b1111_0000, 0b0000_1111];
        update_binary_centroid(
            &mut centroid,
            &[0, 1, 2],
            None,
            &decisive_codes,
            1,
            &mut scratch,
        );
        assert_eq!(centroid, [0b1111_0000]);
    }

    /// Bit counts accumulate in `u8` planes that must be folded into the wide
    /// counters before 256 members can overflow them, and the scratch is reused
    /// across centroids so it must not carry counts between calls.
    #[test]
    fn binary_centroid_majority_survives_plane_overflow_and_scratch_reuse() {
        let byte_len = 3;
        let members: Vec<usize> = (0..MAJORITY_PLANE_FLUSH * 2 + 7).collect();
        // Every member is all-ones, so the majority must be all-ones however
        // many times the planes were folded.
        let ones = vec![0xffu8; members.len() * byte_len];
        let mut scratch = BinaryMajorityScratch::default();
        let mut centroid = vec![0x00u8; byte_len];
        update_binary_centroid(&mut centroid, &members, None, &ones, byte_len, &mut scratch);
        assert_eq!(centroid, vec![0xff; byte_len]);

        // Reusing the same scratch for an all-zero cluster must yield zeros.
        let zeros = vec![0x00u8; members.len() * byte_len];
        let mut next = vec![0xffu8; byte_len];
        update_binary_centroid(&mut next, &members, None, &zeros, byte_len, &mut scratch);
        assert_eq!(next, vec![0x00; byte_len]);

        // A per-bit split just past the fold boundary resolves by true majority.
        let split_members: Vec<usize> = (0..MAJORITY_PLANE_FLUSH + 3).collect();
        let mut split_codes = vec![0x00u8; split_members.len() * byte_len];
        let ones_majority = MAJORITY_PLANE_FLUSH / 2 + 3;
        for (position, chunk) in split_codes.chunks_mut(byte_len).enumerate() {
            if position < ones_majority {
                chunk.fill(0b0000_1111);
            }
        }
        assert!(ones_majority * 2 > split_members.len());
        let mut split = vec![0x00u8; byte_len];
        update_binary_centroid(
            &mut split,
            &split_members,
            None,
            &split_codes,
            byte_len,
            &mut scratch,
        );
        assert_eq!(split, vec![0b0000_1111; byte_len]);
    }

    #[test]
    fn streaming_builder_appends_batches_without_retaining_inputs() {
        let codes = [0x00, 0x01, 0x02, 0xf0, 0xf1, 0xf2];
        let labels = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)];
        let config = BinaryIvfConfig::new(8, 2);
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, codes.len(), "test").unwrap();
        let mut builder = BinaryIvfBuilder::new(&quantizer, IvfRoutingMode::Flat).unwrap();
        builder
            .add_batch(&quantizer, &codes[..3], &labels[..3])
            .unwrap();
        builder
            .add_batch(&quantizer, &codes[3..], &labels[3..])
            .unwrap();
        let index = builder.finish().unwrap();
        assert_eq!(index.len(), 6);
        let plan = quantizer.probe(&[0xf0], 2, IvfRoutingMode::Flat).unwrap();
        assert_eq!(
            index.search_in_clusters(&[0xf0], 1, &plan.cluster_ids)[0].0,
            3
        );

        // Batched inserts group by leaf, so pin the payload layout: one batch
        // must produce the same columns as two, and each leaf must keep its
        // entries in ascending input order.
        let single =
            BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, &codes, &labels).unwrap();
        assert_eq!(index.clusters.len(), single.clusters.len());
        for ((left_id, left), (right_id, right)) in index.clusters.iter().zip(&single.clusters) {
            assert_eq!(left_id, right_id);
            assert_eq!(left.doc_ids, right.doc_ids);
            assert_eq!(left.ordinals, right.ordinals);
            assert_eq!(left.codes, right.codes);
            assert!(
                left.doc_ids.windows(2).all(|pair| pair[0] < pair[1]),
                "leaf {left_id} lost input order: {:?}",
                left.doc_ids
            );
        }
    }

    /// All-zero codes are counted and reported. They are still indexed —
    /// the byte-copy merge requires the payload to hold exactly as many
    /// vectors as flat storage — so the guard is detection, not removal.
    #[test]
    fn all_zero_codes_are_counted_and_reported() {
        let codes = [0x00u8, 0xf0, 0x00, 0x00, 0x0f, 0x00];
        let labels = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)];
        let mut config = BinaryIvfConfig::new(8, 2);
        config.train_iters = 2;
        config.max_train_samples = labels.len();
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, labels.len(), "test").unwrap();
        let index =
            BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, &codes, &labels).unwrap();

        assert_eq!(index.zero_codes(), 4, "four zero codes must be counted");
        assert_eq!(index.len(), labels.len(), "every vector stays indexed");
        // Ties resolve to the lowest cluster ID, so the zeros share one leaf:
        // this is exactly the collapse the report warns about.
        let (_, largest) = index.largest_cluster().expect("a populated leaf");
        assert!(
            largest >= 4,
            "all-zero codes should share one leaf, got {largest}"
        );
        report_binary_build_quality("test-index", 7, &index);
    }

    /// All-ones codes — signbit-packed NaN — are counted and reported the
    /// same way as zeros, and separately from them: the two faces attribute
    /// to different producer revisions.
    #[test]
    fn all_ones_codes_are_counted_and_reported() {
        let codes = [0xffu8, 0xf0, 0xff, 0xff, 0x0f, 0x00];
        let labels = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)];
        let mut config = BinaryIvfConfig::new(8, 2);
        config.train_iters = 2;
        config.max_train_samples = labels.len();
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, labels.len(), "test").unwrap();
        let index =
            BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, &codes, &labels).unwrap();

        assert_eq!(
            index.ones_codes(),
            3,
            "three all-ones codes must be counted"
        );
        assert_eq!(index.zero_codes(), 1, "the zero code is counted on its own");
        assert_eq!(index.len(), labels.len(), "every vector stays indexed");
        // Identical codes share a nearest centroid, so the constant collapses
        // into one leaf: the same topology the ann_health warning names.
        let (_, largest) = index.largest_cluster().expect("a populated leaf");
        assert!(
            largest >= 3,
            "all-ones codes should share one leaf, got {largest}"
        );
        report_binary_build_quality("test-index", 7, &index);
    }

    /// A field that is entirely zero still produces a payload, so merges that
    /// require one keep working.
    #[test]
    fn an_entirely_zero_field_still_produces_a_payload() {
        let codes = [0x00u8; 4];
        let labels = [(0, 0), (1, 0), (2, 0), (3, 0)];
        let mut config = BinaryIvfConfig::new(8, 2);
        config.train_iters = 1;
        config.max_train_samples = labels.len();
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, labels.len(), "test").unwrap();
        let index =
            BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, &codes, &labels).unwrap();
        assert!(!index.is_empty(), "payload must cover every flat vector");
        assert_eq!(index.len(), labels.len());
        assert_eq!(index.zero_codes(), 4);
    }

    #[test]
    fn largest_cluster_reports_the_dominant_leaf() {
        let codes = [0x0fu8, 0x0f, 0x0e, 0x0f, 0x0d, 0x0f, 0xf0];
        let labels = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)];
        let mut config = BinaryIvfConfig::new(8, 2);
        config.train_iters = 4;
        config.max_train_samples = labels.len();
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, labels.len(), "test").unwrap();
        let index =
            BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, &codes, &labels).unwrap();
        let (_, count) = index.largest_cluster().expect("a populated leaf");
        assert_eq!(count, 6, "the dominant leaf holds every near-duplicate");
    }

    #[test]
    fn sparse_payload_does_not_allocate_empty_leaf_columns() {
        let codes = [0x00, 0xff];
        let labels = [(0, 0), (1, 0)];
        let (quantizer, index) = trained_index(8, 2, &codes, &labels);
        assert!(index.clusters.len() <= 2);
        assert_eq!(index.quantizer_version, quantizer.version);
    }

    #[test]
    fn child_allocation_is_exact_and_never_exceeds_group_size() {
        let sizes = [100, 30, 0, 7];
        let allocation = allocate_child_clusters(&sizes, 64);
        assert_eq!(allocation.iter().sum::<usize>(), 64);
        assert!(
            allocation
                .iter()
                .zip(sizes)
                .all(|(&cells, size)| cells <= size)
        );
        assert_eq!(allocation[2], 0);
    }

    #[test]
    fn binary_two_level_build_assignment_checks_four_parents() {
        let leaves_per_parent = 512;
        let children: Vec<Vec<u32>> = (0..4)
            .map(|parent| {
                let first = parent * leaves_per_parent;
                (first..first + leaves_per_parent)
                    .map(|leaf| leaf as u32)
                    .collect()
            })
            .collect();
        let mut leaf_centroids = vec![0x01; 4 * leaves_per_parent];
        let best_leaf = 3 * leaves_per_parent;
        leaf_centroids[best_leaf] = 0x00;
        let quantizer = BinaryCoarseQuantizer {
            dim_bits: 8,
            num_clusters: leaf_centroids.len() as u32,
            centroids: leaf_centroids,
            version: 1,
            routing_index: Some(BinaryCentroidRouter::TwoLevel {
                parent_centroids: vec![0x00, 0xff, 0xff, 0xff],
                topology: IvfRoutingTopology::from_children(&children),
            }),
        };

        assert_eq!(
            &*quantizer
                .probe(&[0x00], 1, IvfRoutingMode::TwoLevel)
                .unwrap()
                .cluster_ids,
            &[0]
        );
        assert_eq!(
            quantizer.assign(&[0x00], IvfRoutingMode::TwoLevel).unwrap(),
            best_leaf as u32
        );
    }

    /// A query or code of the wrong byte width used to probe zero clusters or
    /// assign cluster 0 silently; both must now be loud errors.
    #[test]
    fn binary_probe_and_assign_reject_codes_with_the_wrong_byte_length() {
        let codes = [0x00, 0x01, 0x02, 0xf0, 0xf1, 0xf2];
        let config = BinaryIvfConfig::new(8, 2);
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, codes.len(), "test").unwrap();
        for mode in [
            IvfRoutingMode::Flat,
            IvfRoutingMode::Auto,
            IvfRoutingMode::TwoLevel,
            IvfRoutingMode::Hnsw,
        ] {
            let error = quantizer.probe(&[0xf0, 0x0f], 2, mode).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{mode:?}");
            assert!(error.to_string().contains("2 bytes"), "{error}");
            let error = quantizer.assign(&[], mode).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{mode:?}");
            assert!(error.to_string().contains("expects 1"), "{error}");
        }
        let mut builder = BinaryIvfBuilder::new(&quantizer, IvfRoutingMode::Flat).unwrap();
        assert!(
            builder
                .add_batch(&quantizer, &codes[..4], &[(0, 0), (1, 0)])
                .is_err()
        );
        assert_eq!(builder.len, 0);
    }

    /// The integer `(distance << 32 | id)` selection must reproduce the float
    /// `1 - d / dim` ordering with ID tie-breaks exactly, for every `k`.
    #[test]
    fn flat_probe_integer_selection_matches_float_score_ordering() {
        let dim = 64;
        let byte_len = dim / 8;
        let mut rng = rand::rngs::StdRng::seed_from_u64(29);
        let codes: Vec<u8> = (0..4_000 * byte_len).map(|_| rng.random()).collect();
        let mut config = BinaryIvfConfig::new(dim, 64);
        config.routing = IvfRoutingMode::Flat;
        config.train_iters = 3;
        config.max_train_samples = 4_000;
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, 4_000, "test").unwrap();
        for query in codes.chunks_exact(byte_len).take(16) {
            let mut scores = vec![0.0f32; quantizer.num_clusters as usize];
            batch_hamming_scores(query, &quantizer.centroids, byte_len, dim, &mut scores);
            for k in [1usize, 3, 8, 64] {
                let expected = select_best::<true>(&scores, k);
                let got = quantizer.find_k_nearest(query, k).unwrap();
                assert_eq!(got, expected, "k={k}");
            }
        }
    }

    #[test]
    fn persisted_binary_hnsw_and_two_level_routers_are_valid() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let codes: Vec<u8> = (0..256 * 8).map(|_| rng.random()).collect();
        for routing in [IvfRoutingMode::Hnsw, IvfRoutingMode::TwoLevel] {
            let mut config = BinaryIvfConfig::new(64, 16);
            config.routing = routing;
            config.train_iters = 3;
            config.max_train_samples = 256;
            let quantizer = BinaryCoarseQuantizer::train(config, &codes, 256, "test").unwrap();
            quantizer.validate_routing(routing).unwrap();
            let plan = quantizer.probe(&codes[..8], 8, routing).unwrap();
            assert_eq!(plan.cluster_ids.len(), 8);
            assert!(
                plan.cluster_ids
                    .iter()
                    .all(|&cluster| cluster < quantizer.num_clusters)
            );

            let bytes =
                bincode::serde::encode_to_vec(&quantizer, bincode::config::standard()).unwrap();
            let (loaded, consumed): (BinaryCoarseQuantizer, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(consumed, bytes.len());
            loaded.validate_routing(routing).unwrap();
        }
    }
}
