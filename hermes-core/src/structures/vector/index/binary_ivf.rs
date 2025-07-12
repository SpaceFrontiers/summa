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

use crate::structures::simd::batch_hamming_scores;

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
    /// Train centroids via k-majority and build the index from packed vectors.
    ///
    /// `codes` is `n × byte_len` contiguous packed vectors;
    /// `doc_id_ordinals[i]` labels vector `i`.
    pub fn build(
        mut config: BinaryIvfConfig,
        codes: &[u8],
        doc_id_ordinals: &[(u32, u16)],
    ) -> Self {
        let byte_len = config.byte_len();
        let n = doc_id_ordinals.len();
        debug_assert_eq!(codes.len(), n * byte_len);

        // Can't have more clusters than vectors
        config.num_clusters = config.num_clusters.clamp(1, n.max(1));
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
        let assignments: Vec<usize> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map_init(
                    || vec![0f32; index.config.num_clusters],
                    |scores, i| {
                        index.nearest_centroid(&codes[i * byte_len..(i + 1) * byte_len], scores)
                    },
                )
                .collect()
        };
        #[cfg(not(feature = "native"))]
        let assignments: Vec<usize> = {
            let mut scores = vec![0f32; index.config.num_clusters];
            (0..n)
                .map(|i| {
                    index.nearest_centroid(&codes[i * byte_len..(i + 1) * byte_len], &mut scores)
                })
                .collect()
        };

        // Phase 2: sequential append into SoA cluster storage
        for i in 0..n {
            let code = &codes[i * byte_len..(i + 1) * byte_len];
            let (doc_id, ordinal) = doc_id_ordinals[i];
            let c = &mut index.clusters[assignments[i]];
            c.doc_ids.push(doc_id);
            c.ordinals.push(ordinal);
            c.codes.extend_from_slice(code);
        }
        index.len = n;

        index
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
        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
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
                centroid_scores[b].total_cmp(&centroid_scores[a])
            });
            order.truncate(nprobe);
        }

        // Scan probed clusters with exact SIMD Hamming
        let mut collector = crate::query::ScoreCollector::new(k);
        let mut scores: Vec<f32> = Vec::new();
        for &cluster_id in &order {
            let cluster = &self.clusters[cluster_id];
            let count = cluster.doc_ids.len();
            if count == 0 {
                continue;
            }
            scores.resize(count, 0.0);
            batch_hamming_scores(
                query,
                &cluster.codes,
                byte_len,
                self.config.dim_bits,
                &mut scores[..count],
            );
            let threshold = collector.threshold();
            for (i, &score) in scores.iter().enumerate().take(count) {
                if score > threshold {
                    collector.insert_with_ordinal(cluster.doc_ids[i], score, cluster.ordinals[i]);
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

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
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
    let mut sample: Vec<usize> = (0..n).collect();
    sample.shuffle(&mut rng);
    sample.truncate(config.max_train_samples.max(k));
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
            let best = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(c, _)| c as u32)
                .unwrap_or(0);
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
        let index = BinaryIvfIndex::build(config, &codes, &labels);
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
        let index = BinaryIvfIndex::build(config, &codes, &labels);
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

        let mut index1 = BinaryIvfIndex::build(BinaryIvfConfig::new(dim, 4), &codes1, &labels1);
        let index2 = BinaryIvfIndex::build(BinaryIvfConfig::new(dim, 4), &codes2, &labels2);

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

        let index = BinaryIvfIndex::build(BinaryIvfConfig::new(dim, 4), &codes, &labels);
        let bytes = index.to_bytes().unwrap();
        let back = BinaryIvfIndex::from_bytes(&bytes).unwrap();
        assert_eq!(back.len(), index.len());

        let query = &codes[..byte_len];
        assert_eq!(index.search(query, 5, None), back.search(query, 5, None));
    }
}
