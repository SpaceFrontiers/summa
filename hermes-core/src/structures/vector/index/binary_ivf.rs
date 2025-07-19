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
use std::io;

use crate::dsl::IvfRoutingMode;
use crate::structures::simd::batch_hamming_scores;
use crate::structures::vector::ivf::routing::{
    HNSW_AUTO_THRESHOLD, HnswRoutingGraph, IvfProbePlan, IvfRoutingTopology,
    allocate_child_clusters, binary_probe_fingerprint, effective_routing_mode, parent_probe_count,
    routing_parent_count, select_best, select_best_candidates,
};

fn argmax_score_lowest_index(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                // `max_by` keeps the greater element; reverse the index
                // comparison so equal scores consistently prefer the lowest
                // cluster ID, matching query-time centroid ordering.
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

const MAX_BINARY_IVF_CLUSTERS: usize = 1_048_576;
const BINARY_IVF_SCORE_BATCH: usize = 8_192;

#[inline]
fn packed_hamming_distance(left: &[u8], right: &[u8]) -> u32 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a ^ b).count_ones())
        .sum()
}

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
            match effective_routing_mode(config.routing, config.num_clusters) {
                IvfRoutingMode::TwoLevel => {
                    let (leaves, router) =
                        train_k_majority_hierarchical(&config, codes, num_vectors);
                    (leaves, Some(router))
                }
                IvfRoutingMode::Hnsw => {
                    let leaves = if config.num_clusters >= HNSW_AUTO_THRESHOLD {
                        train_k_majority_hierarchical(&config, codes, num_vectors).0
                    } else {
                        train_k_majority(&config, codes, num_vectors)
                    };
                    let byte_len = config.byte_len();
                    let graph = HnswRoutingGraph::build(
                        config.num_clusters,
                        |left, right| {
                            packed_hamming_distance(
                                &leaves[left as usize * byte_len..(left as usize + 1) * byte_len],
                                &leaves[right as usize * byte_len..(right as usize + 1) * byte_len],
                            ) as f32
                        },
                        config.seed,
                    );
                    (leaves, Some(BinaryCentroidRouter::Hnsw(graph)))
                }
                IvfRoutingMode::Flat | IvfRoutingMode::Auto => {
                    (train_k_majority(&config, codes, num_vectors), None)
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
        match effective_routing_mode(mode, self.num_clusters as usize) {
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

    pub fn probe(&self, query: &[u8], k: usize, mode: IvfRoutingMode) -> IvfProbePlan {
        let take = k.clamp(1, self.num_clusters as usize);
        let cluster_ids = match effective_routing_mode(mode, self.num_clusters as usize) {
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => self.find_k_nearest(query, take),
            IvfRoutingMode::TwoLevel => self.find_k_nearest_two_level(query, take),
            IvfRoutingMode::Hnsw => self.find_k_nearest_hnsw(query, take),
        };
        IvfProbePlan::new(
            self.version,
            binary_probe_fingerprint(query, take, mode),
            cluster_ids,
        )
    }

    pub fn assign(&self, code: &[u8], mode: IvfRoutingMode) -> u32 {
        match effective_routing_mode(mode, self.num_clusters as usize) {
            IvfRoutingMode::Hnsw => self.find_nearest_hnsw(code),
            IvfRoutingMode::TwoLevel => self.find_k_nearest_two_level(code, 1)[0],
            IvfRoutingMode::Flat | IvfRoutingMode::Auto => self.find_nearest(code),
        }
    }

    fn find_nearest(&self, query: &[u8]) -> u32 {
        (0..self.num_clusters)
            .min_by_key(|&cluster| {
                let offset = cluster as usize * self.byte_len();
                packed_hamming_distance(query, &self.centroids[offset..offset + self.byte_len()])
            })
            .unwrap_or(0)
    }

    fn find_k_nearest(&self, query: &[u8], k: usize) -> Vec<u32> {
        if query.len() != self.byte_len() {
            return Vec::new();
        }
        let mut scores = vec![0.0; self.num_clusters as usize];
        batch_hamming_scores(
            query,
            &self.centroids,
            self.byte_len(),
            self.dim_bits,
            &mut scores,
        );
        select_best::<true>(&scores, k)
    }

    fn find_k_nearest_two_level(&self, query: &[u8], k: usize) -> Vec<u32> {
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
        let mut parent_scores = vec![0.0; topology.parent_count()];
        batch_hamming_scores(
            query,
            parent_centroids,
            self.byte_len(),
            self.dim_bits,
            &mut parent_scores,
        );
        let parent_take =
            parent_probe_count(k, self.num_clusters as usize, topology.parent_count());
        let parents = select_best::<true>(&parent_scores, parent_take);
        let candidate_count = parents
            .iter()
            .map(|&parent| topology.children(parent as usize).len())
            .sum();
        let mut candidates = Vec::with_capacity(candidate_count);
        let mut score = [0.0];
        for parent in parents {
            for &leaf in topology.children(parent as usize) {
                let offset = leaf as usize * self.byte_len();
                batch_hamming_scores(
                    query,
                    &self.centroids[offset..offset + self.byte_len()],
                    self.byte_len(),
                    self.dim_bits,
                    &mut score,
                );
                candidates.push((leaf, score[0]));
            }
        }
        select_best_candidates::<true>(&mut candidates, k)
    }

    fn find_k_nearest_hnsw(&self, query: &[u8], k: usize) -> Vec<u32> {
        let Some(BinaryCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_k_nearest(query, k);
        };
        let byte_len = self.byte_len();
        graph.search(
            |leaf| {
                packed_hamming_distance(
                    query,
                    &self.centroids[leaf as usize * byte_len..(leaf as usize + 1) * byte_len],
                ) as f32
            },
            k,
        )
    }

    fn find_nearest_hnsw(&self, query: &[u8]) -> u32 {
        let Some(BinaryCentroidRouter::Hnsw(graph)) = self.routing_index.as_ref() else {
            return self.find_nearest(query);
        };
        let byte_len = self.byte_len();
        graph.search_one(|leaf| {
            packed_hamming_distance(
                query,
                &self.centroids[leaf as usize * byte_len..(leaf as usize + 1) * byte_len],
            ) as f32
        })
    }
}

fn default_max_train_samples() -> usize {
    100_000
}

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
        Ok(())
    }
}

/// One cluster: SoA layout with contiguous packed codes for SIMD scanning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct BinaryCluster {
    doc_ids: Vec<u32>,
    ordinals: Vec<u16>,
    /// Packed codes, `byte_len` bytes per entry, contiguous
    codes: Vec<u8>,
}

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryIvfIndex {
    pub dim_bits: usize,
    pub quantizer_version: u64,
    pub num_clusters: u32,
    /// Sorted non-empty `(leaf_id, payload)` pairs. Empty cells cost no
    /// per-segment heap memory even when the global codebook has millions of
    /// leaves.
    clusters: Vec<(u32, BinaryCluster)>,
    len: usize,
}

impl BinaryIvfIndex {
    pub fn build(
        quantizer: &BinaryCoarseQuantizer,
        routing: IvfRoutingMode,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
    ) -> io::Result<Self> {
        quantizer
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        let byte_len = quantizer.byte_len();
        let expected = doc_id_ordinals.len().checked_mul(byte_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "binary IVF code size overflow")
        })?;
        if codes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary IVF code/label matrix is inconsistent",
            ));
        }
        let n = doc_id_ordinals.len();
        #[cfg(feature = "native")]
        let assignments: Vec<u32> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|i| quantizer.assign(&codes[i * byte_len..(i + 1) * byte_len], routing))
                .collect()
        };
        #[cfg(not(feature = "native"))]
        let assignments: Vec<u32> = (0..n)
            .map(|i| quantizer.assign(&codes[i * byte_len..(i + 1) * byte_len], routing))
            .collect();

        let mut by_cluster: rustc_hash::FxHashMap<u32, BinaryCluster> =
            rustc_hash::FxHashMap::default();
        for (i, &(doc_id, ordinal)) in doc_id_ordinals.iter().enumerate() {
            let cluster = by_cluster.entry(assignments[i]).or_default();
            cluster.doc_ids.push(doc_id);
            cluster.ordinals.push(ordinal);
            cluster
                .codes
                .extend_from_slice(&codes[i * byte_len..(i + 1) * byte_len]);
        }
        let mut clusters: Vec<_> = by_cluster.into_iter().collect();
        clusters.sort_unstable_by_key(|(cluster_id, _)| *cluster_id);
        Ok(Self {
            dim_bits: quantizer.dim_bits,
            quantizer_version: quantizer.version,
            num_clusters: quantizer.num_clusters,
            clusters,
            len: n,
        })
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

    pub fn search_in_clusters(
        &self,
        query: &[u8],
        k: usize,
        cluster_ids: &[u32],
    ) -> Vec<(u32, u16, f32)> {
        self.search_impl::<false>(query, k, cluster_ids)
    }

    pub fn search_distinct_documents_in_clusters(
        &self,
        query: &[u8],
        k: usize,
        cluster_ids: &[u32],
    ) -> Vec<(u32, u16, f32)> {
        self.search_impl::<true>(query, k, cluster_ids)
    }

    fn search_impl<const BY_DOCUMENT: bool>(
        &self,
        query: &[u8],
        k: usize,
        cluster_ids: &[u32],
    ) -> Vec<(u32, u16, f32)> {
        let mut collector = super::BoundedAnnCollector::<BY_DOCUMENT, true>::new(k);
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

    pub fn merge_into(&mut self, other: &Self, doc_id_offset: u32) -> Result<(), &'static str> {
        if self.quantizer_version != other.quantizer_version
            || self.dim_bits != other.dim_bits
            || self.num_clusters != other.num_clusters
        {
            return Err("cannot merge binary IVF payloads from different quantizers");
        }
        for (cluster_id, source) in &other.clusters {
            let position = match self
                .clusters
                .binary_search_by_key(cluster_id, |(id, _)| *id)
            {
                Ok(position) => position,
                Err(position) => {
                    self.clusters
                        .insert(position, (*cluster_id, BinaryCluster::default()));
                    position
                }
            };
            let target = &mut self.clusters[position].1;
            target.doc_ids.reserve(source.doc_ids.len());
            target.ordinals.reserve(source.ordinals.len());
            target.codes.reserve(source.codes.len());
            target
                .doc_ids
                .extend(source.doc_ids.iter().map(|doc_id| doc_id + doc_id_offset));
            target.ordinals.extend_from_slice(&source.ordinals);
            target.codes.extend_from_slice(&source.codes);
        }
        self.len += other.len;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        bincode::serde::encode_into_std_write(self, writer, bincode::config::standard())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }

    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }

    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let index: Self =
            crate::structures::vector::decode_ann_bincode_exact(data, "global binary IVF")?;
        index
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(index)
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        self.clusters
            .iter()
            .map(|(_, cluster)| cluster.codes.len() + cluster.doc_ids.len() * 6)
            .sum()
    }
}

/// Lloyd-style k-majority clustering in Hamming space.
fn train_k_majority(config: &BinaryIvfConfig, codes: &[u8], n: usize) -> Vec<u8> {
    let byte_len = config.byte_len();
    let k = config.num_clusters;
    let dim_bits = config.dim_bits;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // Bound training cost: iterate over a sample, assign everything later
    let sample_len = config.max_train_samples.max(k).min(n);
    let sample = rand::seq::index::sample(&mut rng, n, sample_len).into_vec();
    let n = sample.len();
    let vec_at = |i: usize| -> &[u8] {
        let vi = sample[i];
        &codes[vi * byte_len..(vi + 1) * byte_len]
    };

    let mut centroids = vec![0u8; k * byte_len];
    // k-means++ seeding in Hamming space. Random-first-k is especially prone
    // to duplicate/near-duplicate cells on skewed embedding distributions.
    let first = rng.random_range(0..n);
    centroids[..byte_len].copy_from_slice(vec_at(first));
    let mut min_dist_sq = vec![f64::INFINITY; n];
    for centroid_id in 1..k {
        let previous = &centroids[(centroid_id - 1) * byte_len..centroid_id * byte_len];
        let mut total_weight = 0.0;
        for (index, min_distance) in min_dist_sq.iter_mut().enumerate() {
            let distance: u32 = vec_at(index)
                .iter()
                .zip(previous)
                .map(|(&left, &right)| (left ^ right).count_ones())
                .sum();
            *min_distance = min_distance.min((distance as f64) * (distance as f64));
            total_weight += *min_distance;
        }
        let chosen = if total_weight > 0.0 {
            let mut target = rng.random::<f64>() * total_weight;
            min_dist_sq
                .iter()
                .position(|weight| {
                    target -= *weight;
                    target <= 0.0
                })
                .unwrap_or(n - 1)
        } else {
            rng.random_range(0..n)
        };
        centroids[centroid_id * byte_len..(centroid_id + 1) * byte_len]
            .copy_from_slice(vec_at(chosen));
    }

    let mut assignment = vec![u32::MAX; n];
    let mut scores = vec![0f32; k];

    for _iter in 0..config.train_iters {
        // Assign
        let mut changed = 0usize;
        for (i, slot) in assignment.iter_mut().enumerate().take(n) {
            let code = vec_at(i);
            batch_hamming_scores(code, &centroids, byte_len, dim_bits, &mut scores);
            let best = argmax_score_lowest_index(&scores) as u32;
            if *slot != best {
                *slot = best;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }

        // Update: per-bit majority vote
        let mut bit_counts = vec![0u32; k * dim_bits];
        let mut member_counts = vec![0u32; k];
        for (i, &slot) in assignment.iter().enumerate().take(n) {
            let c = slot as usize;
            member_counts[c] += 1;
            let code = vec_at(i);
            for bit in 0..dim_bits {
                if (code[bit / 8] >> (bit % 8)) & 1 == 1 {
                    bit_counts[c * dim_bits + bit] += 1;
                }
            }
        }

        for c in 0..k {
            let members = member_counts[c];
            if members == 0 {
                // Re-seed empty cluster with a random sampled vector
                let vi = sample[rng.random_range(0..n)];
                centroids[c * byte_len..(c + 1) * byte_len]
                    .copy_from_slice(&codes[vi * byte_len..(vi + 1) * byte_len]);
                continue;
            }
            let half = members / 2;
            let centroid = &mut centroids[c * byte_len..(c + 1) * byte_len];
            centroid.fill(0);
            for bit in 0..dim_bits {
                if bit_counts[c * dim_bits + bit] > half {
                    centroid[bit / 8] |= 1 << (bit % 8);
                }
            }
        }
    }

    centroids
}

/// Hierarchical k-majority training keeps large global codebooks tractable:
/// train sqrt(K) parent cells, partition the sample once, then train each
/// child codebook independently. Complexity is O(N·sqrt(K)) rather than
/// O(N·K), while leaf centroids remain ordinary Hamming-majority centroids.
fn train_k_majority_hierarchical(
    config: &BinaryIvfConfig,
    codes: &[u8],
    n: usize,
) -> (Vec<u8>, BinaryCentroidRouter) {
    let byte_len = config.byte_len();
    let parent_count = routing_parent_count(config.num_clusters).min(n);
    let mut parent_config = config.clone();
    parent_config.num_clusters = parent_count;
    parent_config.max_train_samples = config.max_train_samples.min(n);
    let parents = train_k_majority(&parent_config, codes, n);

    let mut assignments = vec![0u32; n];
    let mut group_sizes = vec![0usize; parent_count];
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        assignments.par_iter_mut().enumerate().for_each_init(
            || vec![0.0; parent_count],
            |scores, (index, assignment)| {
                batch_hamming_scores(
                    &codes[index * byte_len..(index + 1) * byte_len],
                    &parents,
                    byte_len,
                    config.dim_bits,
                    scores,
                );
                *assignment = argmax_score_lowest_index(scores) as u32;
            },
        );
    }
    #[cfg(not(feature = "native"))]
    {
        let mut scores = vec![0.0; parent_count];
        for (index, assignment) in assignments.iter_mut().enumerate() {
            batch_hamming_scores(
                &codes[index * byte_len..(index + 1) * byte_len],
                &parents,
                byte_len,
                config.dim_bits,
                &mut scores,
            );
            *assignment = argmax_score_lowest_index(&scores) as u32;
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

    let mut leaves = Vec::with_capacity(config.num_clusters.saturating_mul(byte_len));
    let mut children = vec![Vec::new(); parent_count];
    for (parent, (group, &child_count)) in groups.iter().zip(&child_counts).enumerate() {
        if child_count == 0 {
            continue;
        }
        let mut child_config = config.clone();
        child_config.num_clusters = child_count;
        child_config.max_train_samples = group.len() / byte_len;
        child_config.seed = config.seed ^ (parent as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let first_leaf = leaves.len() / byte_len;
        leaves.extend_from_slice(&train_k_majority(
            &child_config,
            group,
            group.len() / byte_len,
        ));
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
        let quantizer = BinaryCoarseQuantizer::train(config, codes, labels.len()).unwrap();
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
        let plan = quantizer.probe(&query, 8, IvfRoutingMode::Flat);
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

    #[test]
    fn payload_merge_is_lossless_and_generation_checked() {
        let codes = [0x00, 0x01, 0x02, 0xf0, 0xf1, 0xf2];
        let labels = [(0, 0), (1, 0), (2, 0), (0, 0), (1, 0), (2, 0)];
        let config = BinaryIvfConfig::new(8, 2);
        let quantizer = BinaryCoarseQuantizer::train(config, &codes, codes.len()).unwrap();
        let mut left =
            BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, &codes[..3], &labels[..3])
                .unwrap();
        let right =
            BinaryIvfIndex::build(&quantizer, IvfRoutingMode::Flat, &codes[3..], &labels[3..])
                .unwrap();
        left.merge_into(&right, 3).unwrap();
        assert_eq!(left.len(), 6);
        let plan = quantizer.probe(&[0xf0], 2, IvfRoutingMode::Flat);
        assert_eq!(
            left.search_in_clusters(&[0xf0], 1, &plan.cluster_ids)[0].0,
            3
        );
    }

    #[test]
    fn sparse_payload_and_serialization_do_not_allocate_empty_leaf_columns() {
        let codes = [0x00, 0xff];
        let labels = [(0, 0), (1, 0)];
        let (quantizer, index) = trained_index(8, 2, &codes, &labels);
        assert!(index.clusters.len() <= 2);
        let bytes = index.to_bytes().unwrap();
        let decoded = BinaryIvfIndex::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.quantizer_version, quantizer.version);
        let mut trailing = bytes;
        trailing.push(0);
        assert!(BinaryIvfIndex::from_bytes(&trailing).is_err());
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
    fn persisted_binary_hnsw_and_two_level_routers_are_valid() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let codes: Vec<u8> = (0..256 * 8).map(|_| rng.random()).collect();
        for routing in [IvfRoutingMode::Hnsw, IvfRoutingMode::TwoLevel] {
            let mut config = BinaryIvfConfig::new(64, 16);
            config.routing = routing;
            config.train_iters = 3;
            config.max_train_samples = 256;
            let quantizer = BinaryCoarseQuantizer::train(config, &codes, 256).unwrap();
            quantizer.validate_routing(routing).unwrap();
            let plan = quantizer.probe(&codes[..8], 8, routing);
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
