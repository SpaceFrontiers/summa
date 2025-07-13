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

use crate::structures::simd::batch_hamming_scores;

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

const MAX_BINARY_IVF_CLUSTERS: usize = 4_096;
const BINARY_IVF_SCORE_BATCH: usize = 8_192;

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
    /// Clusters probed at query time when the caller passes none
    pub default_nprobe: usize,
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
            default_nprobe: 32,
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

/// IVF index over packed binary vectors (Hamming distance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryIvfIndex {
    pub config: BinaryIvfConfig,
    /// Packed centroids: `num_clusters × byte_len` bytes, contiguous
    centroids: Vec<u8>,
    clusters: Vec<BinaryCluster>,
    len: usize,
}

impl BinaryIvfIndex {
    fn validate(&self) -> Result<(), String> {
        let config = &self.config;
        config.validate()?;
        let byte_len = config.byte_len();
        let expected_centroids = config
            .num_clusters
            .checked_mul(byte_len)
            .ok_or_else(|| "binary IVF centroid size overflow".to_string())?;
        if self.centroids.len() != expected_centroids || self.clusters.len() != config.num_clusters
        {
            return Err("binary IVF centroid/cluster columns are inconsistent".to_string());
        }

        let mut total = 0usize;
        for cluster in &self.clusters {
            let count = cluster.doc_ids.len();
            let expected_codes = count
                .checked_mul(byte_len)
                .ok_or_else(|| "binary IVF code size overflow".to_string())?;
            if cluster.ordinals.len() != count || cluster.codes.len() != expected_codes {
                return Err("binary IVF cluster columns are inconsistent".to_string());
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| "binary IVF vector count overflow".to_string())?;
        }
        if total != self.len {
            return Err(format!(
                "binary IVF vector count is {}, metadata says {}",
                total, self.len
            ));
        }
        Ok(())
    }

    /// Train centroids via k-majority and build the index from packed vectors.
    ///
    /// `codes` is `n × byte_len` contiguous packed vectors;
    /// `doc_id_ordinals[i]` labels vector `i`.
    pub fn build(
        mut config: BinaryIvfConfig,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
    ) -> io::Result<Self> {
        config
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        let byte_len = config.byte_len();
        let n = doc_id_ordinals.len();
        let expected_codes = n.checked_mul(byte_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "binary IVF code size overflow")
        })?;
        if codes.len() != expected_codes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "binary IVF needs {expected_codes} code bytes for {n} labels, got {}",
                    codes.len()
                ),
            ));
        }

        // Can't have more clusters than vectors
        config.num_clusters = config
            .num_clusters
            .clamp(1, n.max(1))
            .min(u32::MAX as usize);
        let k = config.num_clusters;

        let centroids = train_k_majority(&config, codes, n);

        let mut index = Self {
            config,
            centroids,
            clusters: vec![BinaryCluster::default(); k],
            len: 0,
        };

        // Phase 1: nearest-centroid assignment — embarrassingly parallel,
        // dominates build time at O(n × k) Hamming comparisons.
        #[cfg(feature = "native")]
        let assignments: Vec<u32> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map_init(
                    || vec![0f32; index.config.num_clusters],
                    |scores, i| {
                        index.nearest_centroid(&codes[i * byte_len..(i + 1) * byte_len], scores)
                            as u32
                    },
                )
                .collect()
        };
        #[cfg(not(feature = "native"))]
        let assignments: Vec<u32> = {
            let mut scores = vec![0f32; index.config.num_clusters];
            (0..n)
                .map(|i| {
                    index.nearest_centroid(&codes[i * byte_len..(i + 1) * byte_len], &mut scores)
                        as u32
                })
                .collect()
        };

        // Reserve exact cluster payload sizes before copying. Geometric Vec
        // growth otherwise keeps substantial over-capacity for large clusters
        // while the complete input code buffer is still resident.
        let mut cluster_sizes = vec![0usize; k];
        for &assignment in &assignments {
            cluster_sizes[assignment as usize] += 1;
        }
        for (cluster, count) in index.clusters.iter_mut().zip(cluster_sizes) {
            cluster.doc_ids.reserve_exact(count);
            cluster.ordinals.reserve_exact(count);
            cluster.codes.reserve_exact(count.saturating_mul(byte_len));
        }

        // Phase 2: sequential append into SoA cluster storage
        for i in 0..n {
            let code = &codes[i * byte_len..(i + 1) * byte_len];
            let (doc_id, ordinal) = doc_id_ordinals[i];
            let c = &mut index.clusters[assignments[i] as usize];
            c.doc_ids.push(doc_id);
            c.ordinals.push(ordinal);
            c.codes.extend_from_slice(code);
        }
        index.len = n;

        Ok(index)
    }

    /// Assign a packed code to its nearest centroid and append it.
    /// `centroid_scores` is a reusable scratch buffer of `num_clusters` floats.
    fn add_assigned(&mut self, code: &[u8], doc_id: u32, ordinal: u16, scores: &mut [f32]) {
        let cluster = self.nearest_centroid(code, scores);
        let c = &mut self.clusters[cluster];
        c.doc_ids.push(doc_id);
        c.ordinals.push(ordinal);
        c.codes.extend_from_slice(code);
        self.len += 1;
    }

    /// Index of the nearest centroid by Hamming distance.
    fn nearest_centroid(&self, code: &[u8], scores: &mut [f32]) -> usize {
        let byte_len = self.config.byte_len();
        batch_hamming_scores(
            code,
            &self.centroids,
            byte_len,
            self.config.dim_bits,
            scores,
        );
        // batch_hamming_scores returns similarity (higher = closer)
        argmax_score_lowest_index(scores)
    }

    /// Search: probe `nprobe` nearest clusters, exact Hamming within each.
    ///
    /// Returns `(doc_id, ordinal, similarity)` with similarity = 1 - hamming/dim,
    /// sorted descending — exact for every scanned vector.
    pub fn search(&self, query: &[u8], k: usize, nprobe: Option<usize>) -> Vec<(u32, u16, f32)> {
        let byte_len = self.config.byte_len();
        if query.len() != byte_len || self.len == 0 {
            return Vec::new();
        }
        let nprobe = nprobe
            .unwrap_or(self.config.default_nprobe)
            .clamp(1, self.config.num_clusters);

        // Rank centroids by similarity
        let mut centroid_scores = vec![0f32; self.config.num_clusters];
        batch_hamming_scores(
            query,
            &self.centroids,
            byte_len,
            self.config.dim_bits,
            &mut centroid_scores,
        );
        let mut order: Vec<usize> = (0..self.config.num_clusters).collect();
        if nprobe < order.len() {
            order.select_nth_unstable_by(nprobe, |&a, &b| {
                centroid_scores[b]
                    .total_cmp(&centroid_scores[a])
                    .then_with(|| a.cmp(&b))
            });
            order.truncate(nprobe);
        }

        // Scan probed clusters with exact SIMD Hamming
        let mut collector = crate::query::ScoreCollector::new(k);
        let mut scores = vec![0.0f32; BINARY_IVF_SCORE_BATCH.min(self.len)];
        for &cluster_id in &order {
            let cluster = &self.clusters[cluster_id];
            let count = cluster.doc_ids.len();
            if count == 0 {
                continue;
            }
            for batch_start in (0..count).step_by(BINARY_IVF_SCORE_BATCH) {
                let batch_count = BINARY_IVF_SCORE_BATCH.min(count - batch_start);
                let code_start = batch_start * byte_len;
                let code_end = (batch_start + batch_count) * byte_len;
                batch_hamming_scores(
                    query,
                    &cluster.codes[code_start..code_end],
                    byte_len,
                    self.config.dim_bits,
                    &mut scores[..batch_count],
                );
                for (batch_idx, &score) in scores.iter().enumerate().take(batch_count) {
                    let i = batch_start + batch_idx;
                    let doc_id = cluster.doc_ids[i];
                    let ordinal = cluster.ordinals[i];
                    if collector.would_enter_candidate(doc_id, score, ordinal) {
                        collector.insert_with_ordinal(doc_id, score, ordinal);
                    }
                }
            }
        }

        collector
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score, ordinal)| (doc_id, ordinal, score))
            .collect()
    }

    /// Merge another index into this one, re-assigning its vectors to this
    /// index's centroids. Lossless: cluster payloads are the exact codes.
    pub fn merge_into(&mut self, other: &BinaryIvfIndex, doc_id_offset: u32) {
        let byte_len = self.config.byte_len();
        let mut scores = vec![0f32; self.config.num_clusters];
        for cluster in &other.clusters {
            for i in 0..cluster.doc_ids.len() {
                let code = &cluster.codes[i * byte_len..(i + 1) * byte_len];
                self.add_assigned(
                    code,
                    cluster.doc_ids[i] + doc_id_offset,
                    cluster.ordinals[i],
                    &mut scores,
                );
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_clusters(&self) -> usize {
        self.config.num_clusters
    }

    /// Serialize to compact bytes (bincode).
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Serialize directly into an output stream without materializing a second
    /// complete ANN blob in memory.
    pub fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        bincode::serde::encode_into_std_write(self, writer, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let index: Self = crate::structures::vector::decode_ann_bincode_exact(data, "binary IVF")?;
        index
            .validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(index)
    }

    /// Estimated memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> usize {
        self.centroids.len()
            + self
                .clusters
                .iter()
                .map(|c| c.codes.len() + c.doc_ids.len() * 6)
                .sum::<usize>()
    }
}

/// k-majority clustering: Hamming-space k-means.
///
/// Init: k distinct vectors sampled without replacement. Iterate: assign every
/// vector to its nearest centroid, then set each centroid bit to the majority
/// value among its members. Empty clusters are re-seeded with random vectors.
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

    // Init: sample k distinct vector indexes
    let init: Vec<usize> = sample.iter().copied().take(k).collect();

    let mut centroids = vec![0u8; k * byte_len];
    for (c, &vi) in init.iter().enumerate() {
        centroids[c * byte_len..(c + 1) * byte_len]
            .copy_from_slice(&codes[vi * byte_len..(vi + 1) * byte_len]);
    }
    drop(init);

    let mut assignment = vec![0u32; n];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_ties_prefer_the_lowest_cluster_id() {
        assert_eq!(argmax_score_lowest_index(&[0.5, 1.0, 1.0, 0.25]), 1);
    }

    #[test]
    fn duplicate_centroids_remain_searchable_with_one_probe() {
        let codes = vec![0u8; 32];
        let labels: Vec<_> = (0..32).map(|doc_id| (doc_id, 0)).collect();
        let index = BinaryIvfIndex::build(BinaryIvfConfig::new(8, 4), &codes, &labels).unwrap();

        let results = index.search(&[0], 5, Some(1));
        assert_eq!(results.len(), 5);
        assert_eq!(
            results.iter().map(|result| result.0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    fn pack(bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }

    /// Two well-separated bit clusters must be recovered and searched exactly.
    #[test]
    fn test_binary_ivf_clusters_and_search() {
        let dim = 64;
        let byte_len = dim / 8;
        let n = 200;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);

        // Cluster A: mostly ones; Cluster B: mostly zeros
        let mut codes = Vec::with_capacity(n * byte_len);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let base = if i % 2 == 0 { 1u8 } else { 0u8 };
            let bits: Vec<u8> = (0..dim)
                .map(|_| {
                    if rng.random::<f32>() < 0.1 {
                        1 - base
                    } else {
                        base
                    }
                })
                .collect();
            codes.extend_from_slice(&pack(&bits));
            labels.push((i as u32, 0u16));
        }

        let config = BinaryIvfConfig::new(dim, 2);
        let index = BinaryIvfIndex::build(config, &codes, &labels).unwrap();
        assert_eq!(index.len(), n);

        // Query: all ones — must retrieve cluster-A (even doc_ids) members
        let query = vec![0xFFu8; byte_len];
        let results = index.search(&query, 10, Some(1));
        assert_eq!(results.len(), 10);
        for &(doc_id, _, score) in &results {
            assert_eq!(doc_id % 2, 0, "expected mostly-ones cluster members");
            assert!(score > 0.7);
        }
    }

    #[test]
    fn search_retains_zero_score_ties_in_doc_id_order() {
        let codes = vec![0xff; 5];
        let labels: Vec<_> = (0..5).map(|doc_id| (doc_id, 0)).collect();
        let index = BinaryIvfIndex::build(BinaryIvfConfig::new(8, 1), &codes, &labels).unwrap();

        let results = index.search(&[0], 3, Some(1));
        assert_eq!(results, vec![(0, 0, 0.0), (1, 0, 0.0), (2, 0, 0.0)]);
    }

    #[test]
    fn deserialize_rejects_inconsistent_cluster_columns() {
        let mut index = BinaryIvfIndex::build(BinaryIvfConfig::new(8, 1), &[0], &[(0, 0)]).unwrap();
        index.clusters[0].ordinals.clear();
        let bytes = index.to_bytes().unwrap();

        assert!(BinaryIvfIndex::from_bytes(&bytes).is_err());
    }

    #[test]
    fn build_rejects_invalid_config_and_code_lengths() {
        let labels = [(0, 0)];

        assert!(BinaryIvfIndex::build(BinaryIvfConfig::new(0, 1), &[], &labels).is_err());
        assert!(BinaryIvfIndex::build(BinaryIvfConfig::new(7, 1), &[0], &labels).is_err());
        assert!(BinaryIvfIndex::build(BinaryIvfConfig::new(8, 0), &[0], &labels).is_err());
        assert!(
            BinaryIvfIndex::build(
                BinaryIvfConfig::new(8, MAX_BINARY_IVF_CLUSTERS + 1),
                &[0],
                &labels,
            )
            .is_err()
        );
        assert!(BinaryIvfIndex::build(BinaryIvfConfig::new(16, 1), &[0], &labels).is_err());
    }

    #[test]
    fn search_scores_large_clusters_in_equivalent_chunks() {
        let n = BINARY_IVF_SCORE_BATCH + 37;
        let codes: Vec<u8> = (0..n).map(|i| i as u8).collect();
        let labels: Vec<(u32, u16)> = (0..n as u32).map(|doc_id| (doc_id, 0)).collect();
        let query = [0x5a];
        let k = 64;

        let index = BinaryIvfIndex::build(BinaryIvfConfig::new(8, 1), &codes, &labels).unwrap();
        let actual = index.search(&query, k, Some(1));

        let mut scores = vec![0.0; n];
        batch_hamming_scores(&query, &codes, 1, 8, &mut scores);
        let mut expected: Vec<_> = scores
            .into_iter()
            .enumerate()
            .map(|(doc_id, score)| (doc_id as u32, 0, score))
            .collect();
        expected.sort_unstable_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        expected.truncate(k);

        assert_eq!(actual, expected);
    }

    /// Full probing must match brute force exactly (scanned distances are exact).
    #[test]
    fn test_binary_ivf_full_probe_equals_brute_force() {
        let dim = 128;
        let byte_len = dim / 8;
        let n = 300;
        let k = 15;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);

        let codes: Vec<u8> = (0..n * byte_len).map(|_| rng.random::<u8>()).collect();
        let labels: Vec<(u32, u16)> = (0..n as u32).map(|i| (i, 0)).collect();
        let query: Vec<u8> = (0..byte_len).map(|_| rng.random::<u8>()).collect();

        // Brute force top-k
        let mut scores = vec![0f32; n];
        batch_hamming_scores(&query, &codes, byte_len, dim, &mut scores);
        let mut truth: Vec<(u32, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .collect();
        truth.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        let config = BinaryIvfConfig::new(dim, 8);
        let index = BinaryIvfIndex::build(config, &codes, &labels).unwrap();
        let results = index.search(&query, k, Some(8)); // probe all clusters

        assert_eq!(results.len(), k);
        let truth_scores: Vec<f32> = truth[..k].iter().map(|&(_, s)| s).collect();
        let ivf_scores: Vec<f32> = results.iter().map(|&(_, _, s)| s).collect();
        assert_eq!(
            ivf_scores, truth_scores,
            "full-probe IVF must equal brute force"
        );
    }

    #[test]
    fn test_binary_ivf_merge() {
        let dim = 64;
        let byte_len = dim / 8;
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);

        let make = |n: usize, rng: &mut rand::rngs::StdRng| -> (Vec<u8>, Vec<(u32, u16)>) {
            let codes: Vec<u8> = (0..n * byte_len).map(|_| rng.random::<u8>()).collect();
            let labels: Vec<(u32, u16)> = (0..n as u32).map(|i| (i, 0)).collect();
            (codes, labels)
        };

        let (codes1, labels1) = make(100, &mut rng);
        let (codes2, labels2) = make(80, &mut rng);

        let mut index1 =
            BinaryIvfIndex::build(BinaryIvfConfig::new(dim, 4), &codes1, &labels1).unwrap();
        let index2 =
            BinaryIvfIndex::build(BinaryIvfConfig::new(dim, 4), &codes2, &labels2).unwrap();

        index1.merge_into(&index2, 100);
        assert_eq!(index1.len(), 180);

        // Merged docs are findable
        let query = &codes2[..byte_len]; // first vector of segment 2 → doc 100
        let results = index1.search(query, 1, Some(4));
        assert_eq!(results[0].0, 100);
        assert!((results[0].2 - 1.0).abs() < 1e-6, "exact self-match");
    }

    #[test]
    fn test_binary_ivf_serde_roundtrip() {
        let dim = 64;
        let byte_len = dim / 8;
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let codes: Vec<u8> = (0..50 * byte_len).map(|_| rng.random::<u8>()).collect();
        let labels: Vec<(u32, u16)> = (0..50u32).map(|i| (i, 0)).collect();

        let index = BinaryIvfIndex::build(BinaryIvfConfig::new(dim, 4), &codes, &labels).unwrap();
        let bytes = index.to_bytes().unwrap();
        let mut streamed = Vec::new();
        let written = index.write_to(&mut streamed).unwrap();
        assert_eq!(written, streamed.len());
        assert_eq!(streamed, bytes);
        let back = BinaryIvfIndex::from_bytes(&bytes).unwrap();
        assert_eq!(back.len(), index.len());
        let mut with_trailing_data = bytes.clone();
        with_trailing_data.push(0);
        assert!(BinaryIvfIndex::from_bytes(&with_trailing_data).is_err());

        let query = &codes[..byte_len];
        assert_eq!(index.search(query, 5, None), back.search(query, 5, None));
    }
}
