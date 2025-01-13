//! IVF-RaBitQ: Inverted File Index with RaBitQ quantization
//!
//! Two-level index for billion-scale vector search:
//! - Level 1: Coarse quantizer (k-means centroids)
//! - Level 2: RaBitQ binary codes per cluster
//!
//! Key feature: Segments sharing the same coarse centroids can be merged
//! in O(1) by concatenating cluster data - no re-quantization needed.

use std::collections::HashMap;
use std::io::{self, Cursor, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::rabitq::QuantizedVector;

/// Magic number for coarse centroids file
const CENTROIDS_MAGIC: u32 = 0x48435643; // "CVCH" - Coarse Vector Centroids Hermes

/// Magic number for IVF-RaBitQ segment file
#[allow(dead_code)]
const IVF_MAGIC: u32 = 0x49565651; // "IVFQ"

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
}

impl CoarseCentroids {
    /// Train coarse centroids using k-means algorithm
    ///
    /// Uses kmeans crate with SIMD acceleration (native feature).
    #[cfg(feature = "native")]
    pub fn train(vectors: &[Vec<f32>], num_clusters: usize, max_iters: usize, _seed: u64) -> Self {
        use kmeans::{EuclideanDistance, KMeans, KMeansConfig};

        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(num_clusters > 0, "Need at least 1 cluster");

        let actual_clusters = num_clusters.min(vectors.len());
        let dim = vectors[0].len();

        // Flatten vectors for kmeans crate (expects flat slice)
        let samples: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        // Run k-means with k-means++ initialization
        // KMeans<f32, 8, _> uses 8-lane SIMD (AVX256)
        let kmean: KMeans<f32, 8, _> = KMeans::new(&samples, vectors.len(), dim, EuclideanDistance);
        let result = kmean.kmeans_lloyd(
            actual_clusters,
            max_iters,
            KMeans::init_kmeanplusplus,
            &KMeansConfig::default(),
        );

        // Extract centroids from StrideBuffer to flat Vec
        let centroids: Vec<f32> = result
            .centroids
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            num_clusters: actual_clusters as u32,
            dim,
            centroids,
            version,
        }
    }

    /// Fallback k-means for non-native builds (WASM)
    #[cfg(not(feature = "native"))]
    pub fn train(vectors: &[Vec<f32>], num_clusters: usize, max_iters: usize, seed: u64) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(num_clusters > 0, "Need at least 1 cluster");

        let actual_clusters = num_clusters.min(vectors.len());
        let dim = vectors[0].len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Simple random initialization
        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        indices.shuffle(&mut rng);

        let mut centroids: Vec<f32> = indices[..actual_clusters]
            .iter()
            .flat_map(|&i| vectors[i].iter().copied())
            .collect();

        // K-means iterations
        for _ in 0..max_iters {
            let assignments: Vec<usize> = vectors
                .iter()
                .map(|v| Self::find_nearest_centroid_idx(v, &centroids, dim))
                .collect();

            let mut new_centroids = vec![0.0f32; actual_clusters * dim];
            let mut counts = vec![0usize; actual_clusters];

            for (vec_idx, &cluster_id) in assignments.iter().enumerate() {
                counts[cluster_id] += 1;
                let offset = cluster_id * dim;
                for (i, &val) in vectors[vec_idx].iter().enumerate() {
                    new_centroids[offset + i] += val;
                }
            }

            for cluster_id in 0..actual_clusters {
                if counts[cluster_id] > 0 {
                    let offset = cluster_id * dim;
                    for i in 0..dim {
                        new_centroids[offset + i] /= counts[cluster_id] as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        let version = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            num_clusters: actual_clusters as u32,
            dim,
            centroids,
            version,
        }
    }

    /// K-means++ initialization for better starting centroids
    #[allow(dead_code)]
    fn kmeans_plusplus_init(
        vectors: &[Vec<f32>],
        num_clusters: usize,
        rng: &mut impl Rng,
    ) -> Vec<f32> {
        let dim = vectors[0].len();
        let mut centroids = Vec::with_capacity(num_clusters * dim);

        // First centroid: random
        let first_idx = rng.random_range(0..vectors.len());
        centroids.extend_from_slice(&vectors[first_idx]);

        // Remaining centroids: weighted by distance to nearest existing centroid
        for _ in 1..num_clusters {
            let mut distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    let mut min_dist = f32::MAX;
                    for c in 0..(centroids.len() / dim) {
                        let offset = c * dim;
                        let dist: f32 = v
                            .iter()
                            .zip(&centroids[offset..offset + dim])
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum();
                        min_dist = min_dist.min(dist);
                    }
                    min_dist
                })
                .collect();

            // Normalize to probabilities
            let total: f32 = distances.iter().sum();
            if total > 0.0 {
                for d in &mut distances {
                    *d /= total;
                }
            }

            // Sample proportional to distance squared
            let r: f32 = rng.random();
            let mut cumsum = 0.0;
            let mut chosen_idx = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= r {
                    chosen_idx = i;
                    break;
                }
            }

            centroids.extend_from_slice(&vectors[chosen_idx]);
        }

        centroids
    }

    /// Find nearest centroid index for a vector
    fn find_nearest_centroid_idx(vector: &[f32], centroids: &[f32], dim: usize) -> usize {
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
        Self::find_nearest_centroid_idx(vector, &self.centroids, self.dim) as u32
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

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances.into_iter().map(|(c, _)| c).collect()
    }

    /// Get centroid for a cluster
    pub fn get_centroid(&self, cluster_id: u32) -> &[f32] {
        let offset = cluster_id as usize * self.dim;
        &self.centroids[offset..offset + self.dim]
    }

    /// Save to binary file
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }

    /// Write to any writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(CENTROIDS_MAGIC)?;
        writer.write_u32::<LittleEndian>(1)?; // version
        writer.write_u64::<LittleEndian>(self.version)?;
        writer.write_u32::<LittleEndian>(self.num_clusters)?;
        writer.write_u32::<LittleEndian>(self.dim as u32)?;

        for &val in &self.centroids {
            writer.write_f32::<LittleEndian>(val)?;
        }

        Ok(())
    }

    /// Load from binary file
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::read_from(&mut Cursor::new(data))
    }

    /// Read from any reader
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != CENTROIDS_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid centroids file magic",
            ));
        }

        let _file_version = reader.read_u32::<LittleEndian>()?;
        let version = reader.read_u64::<LittleEndian>()?;
        let num_clusters = reader.read_u32::<LittleEndian>()?;
        let dim = reader.read_u32::<LittleEndian>()? as usize;

        let mut centroids = vec![0.0f32; num_clusters as usize * dim];
        for val in &mut centroids {
            *val = reader.read_f32::<LittleEndian>()?;
        }

        Ok(Self {
            num_clusters,
            dim,
            centroids,
            version,
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.write_to(&mut buf)?;
        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        Self::read_from(&mut Cursor::new(data))
    }
}

/// Data for a single cluster within a segment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterData {
    /// Document IDs (local to segment)
    pub doc_ids: Vec<u32>,
    /// Binary quantized vectors
    pub binary_codes: Vec<QuantizedVector>,
    /// Raw vectors for re-ranking (optional)
    pub raw_vectors: Option<Vec<Vec<f32>>>,
}

impl ClusterData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Append another cluster's data (for merging)
    pub fn append(&mut self, other: &ClusterData, doc_id_offset: u32) {
        for &doc_id in &other.doc_ids {
            self.doc_ids.push(doc_id + doc_id_offset);
        }
        self.binary_codes.extend(other.binary_codes.iter().cloned());

        if let Some(ref other_raw) = other.raw_vectors {
            let raw = self.raw_vectors.get_or_insert_with(Vec::new);
            raw.extend(other_raw.iter().cloned());
        }
    }
}

/// IVF-RaBitQ index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFConfig {
    /// Vector dimension
    pub dim: usize,
    /// Random seed for reproducible transforms
    pub seed: u64,
    /// Number of bits for query quantization (usually 4)
    pub query_bits: u8,
    /// Store raw vectors for re-ranking
    pub store_raw: bool,
    /// Number of clusters to probe during search
    pub default_nprobe: usize,
}

impl IVFConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            seed: 42,
            query_bits: 4,
            store_raw: true,
            default_nprobe: 32,
        }
    }
}

/// IVF-RaBitQ index for a single segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFRaBitQIndex {
    /// Configuration
    pub config: IVFConfig,
    /// Version of coarse centroids used (for merge compatibility)
    pub centroids_version: u64,
    /// Random signs for transform (+1 or -1)
    pub random_signs: Vec<i8>,
    /// Random permutation for transform
    pub random_perm: Vec<u32>,
    /// Cluster data (sparse - only populated clusters)
    pub clusters: HashMap<u32, ClusterData>,
    /// Total number of vectors indexed
    pub num_vectors: usize,
}

impl IVFRaBitQIndex {
    /// Create a new empty IVF index
    pub fn new(config: IVFConfig, centroids_version: u64) -> Self {
        let dim = config.dim;
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        // Generate random signs
        let random_signs: Vec<i8> = (0..dim)
            .map(|_| if rng.random::<bool>() { 1 } else { -1 })
            .collect();

        // Generate random permutation
        let mut random_perm: Vec<u32> = (0..dim as u32).collect();
        for i in (1..dim).rev() {
            let j = rng.random_range(0..=i);
            random_perm.swap(i, j);
        }

        Self {
            config,
            centroids_version,
            random_signs,
            random_perm,
            clusters: HashMap::new(),
            num_vectors: 0,
        }
    }

    /// Build index from vectors using provided coarse centroids
    pub fn build(
        config: IVFConfig,
        coarse_centroids: &CoarseCentroids,
        vectors: &[Vec<f32>],
        doc_ids: Option<&[u32]>,
    ) -> Self {
        let mut index = Self::new(config.clone(), coarse_centroids.version);

        for (i, vector) in vectors.iter().enumerate() {
            let doc_id = doc_ids.map(|ids| ids[i]).unwrap_or(i as u32);
            index.add_vector(coarse_centroids, doc_id, vector);
        }

        index
    }

    /// Add a single vector to the index
    pub fn add_vector(&mut self, coarse_centroids: &CoarseCentroids, doc_id: u32, vector: &[f32]) {
        // Find nearest cluster
        let cluster_id = coarse_centroids.find_nearest(vector);

        // Get cluster centroid
        let centroid = coarse_centroids.get_centroid(cluster_id);

        // Quantize relative to cluster centroid
        let binary_code = self.quantize_vector(vector, centroid);

        // Store in cluster
        let cluster = self.clusters.entry(cluster_id).or_default();
        cluster.doc_ids.push(doc_id);
        cluster.binary_codes.push(binary_code);

        if self.config.store_raw {
            cluster
                .raw_vectors
                .get_or_insert_with(Vec::new)
                .push(vector.to_vec());
        }

        self.num_vectors += 1;
    }

    /// Quantize a vector relative to a centroid
    fn quantize_vector(&self, raw: &[f32], centroid: &[f32]) -> QuantizedVector {
        let dim = self.config.dim;

        // Subtract centroid and compute norm
        let mut centered: Vec<f32> = raw.iter().zip(centroid).map(|(&v, &c)| v - c).collect();

        let norm: f32 = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dist_to_centroid = norm;

        // Normalize
        if norm > 1e-10 {
            for x in &mut centered {
                *x /= norm;
            }
        }

        // Apply random transform
        let transformed: Vec<f32> = (0..dim)
            .map(|i| {
                let src_idx = self.random_perm[i] as usize;
                centered[src_idx] * self.random_signs[src_idx] as f32
            })
            .collect();

        // Binary quantize
        let num_bytes = dim.div_ceil(8);
        let mut bits = vec![0u8; num_bytes];
        let mut popcount = 0u32;

        for i in 0..dim {
            if transformed[i] >= 0.0 {
                bits[i / 8] |= 1 << (i % 8);
                popcount += 1;
            }
        }

        // Compute self dot product
        let scale = 1.0 / (dim as f32).sqrt();
        let mut self_dot = 0.0f32;
        for i in 0..dim {
            let o_bar_i = if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                scale
            } else {
                -scale
            };
            self_dot += transformed[i] * o_bar_i;
        }

        QuantizedVector {
            bits,
            dist_to_centroid,
            self_dot,
            popcount,
        }
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        coarse_centroids: &CoarseCentroids,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Vec<(u32, f32)> {
        // Find nprobe nearest clusters
        let nearest_clusters = coarse_centroids.find_k_nearest(query, nprobe);

        // Collect candidates from all probed clusters
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        for cluster_id in nearest_clusters {
            if let Some(cluster) = self.clusters.get(&cluster_id) {
                let centroid = coarse_centroids.get_centroid(cluster_id);
                let prepared = self.prepare_query(query, centroid);

                for (i, binary_code) in cluster.binary_codes.iter().enumerate() {
                    let dist = self.estimate_distance(&prepared, binary_code);
                    candidates.push((cluster.doc_ids[i], dist));
                }
            }
        }

        // Sort by estimated distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Re-rank top candidates with exact distance if raw vectors available
        let rerank_count = (k * 3).min(candidates.len());
        if rerank_count > 0 {
            let mut reranked: Vec<(u32, f32)> = Vec::with_capacity(rerank_count);

            for &(doc_id, _) in candidates.iter().take(rerank_count) {
                // Find the vector in clusters
                for cluster in self.clusters.values() {
                    if let Some(pos) = cluster.doc_ids.iter().position(|&d| d == doc_id) {
                        if let Some(ref raw_vecs) = cluster.raw_vectors {
                            let raw_vec = &raw_vecs[pos];
                            let dist: f32 = query
                                .iter()
                                .zip(raw_vec.iter())
                                .map(|(&a, &b)| (a - b).powi(2))
                                .sum();
                            reranked.push((doc_id, dist));
                        }
                        break;
                    }
                }
            }

            if !reranked.is_empty() {
                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                reranked.truncate(k);
                return reranked;
            }
        }

        // Sort by distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Prepare query for fast distance estimation (matches RaBitQ algorithm)
    fn prepare_query(&self, raw_query: &[f32], centroid: &[f32]) -> PreparedQuery {
        let dim = self.config.dim;

        // Subtract centroid
        let mut centered: Vec<f32> = raw_query
            .iter()
            .zip(centroid)
            .map(|(&v, &c)| v - c)
            .collect();

        let norm: f32 = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dist_to_centroid = norm;

        // Normalize
        if norm > 1e-10 {
            for x in &mut centered {
                *x /= norm;
            }
        }

        // Apply random transform
        let transformed: Vec<f32> = (0..dim)
            .map(|i| {
                let src_idx = self.random_perm[i] as usize;
                centered[src_idx] * self.random_signs[src_idx] as f32
            })
            .collect();

        // Scalar quantize to 4-bit (same as RaBitQ)
        let min_val = transformed.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = transformed
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let lower = min_val;
        let width = if max_val > min_val {
            max_val - min_val
        } else {
            1.0
        };

        // Quantize to 0-15 range
        let quantized_vals: Vec<u8> = transformed
            .iter()
            .map(|&x| {
                let normalized = (x - lower) / width;
                (normalized * 15.0).round().clamp(0.0, 15.0) as u8
            })
            .collect();

        // Compute sum of quantized values
        let sum: u32 = quantized_vals.iter().map(|&x| x as u32).sum();

        // Build LUTs for fast dot product
        let num_luts = dim.div_ceil(4);
        let mut luts = vec![[0u16; 16]; num_luts];

        for (lut_idx, lut) in luts.iter_mut().enumerate() {
            let base_dim = lut_idx * 4;
            for pattern in 0u8..16 {
                let mut dot = 0u16;
                for bit in 0..4 {
                    let dim_idx = base_dim + bit;
                    if dim_idx < dim && (pattern >> bit) & 1 == 1 {
                        dot += quantized_vals[dim_idx] as u16;
                    }
                }
                lut[pattern as usize] = dot;
            }
        }

        PreparedQuery {
            dist_to_centroid,
            lower,
            width,
            sum,
            luts,
        }
    }

    /// Estimate distance using LUT-based dot product (matches RaBitQ algorithm)
    fn estimate_distance(&self, query: &PreparedQuery, vec: &QuantizedVector) -> f32 {
        let dim = self.config.dim;

        // LUT-based dot product
        let mut dot_sum = 0u32;
        for (lut_idx, lut) in query.luts.iter().enumerate() {
            let base_bit = lut_idx * 4;
            let byte_idx = base_bit / 8;
            let bit_offset = base_bit % 8;

            let byte = vec.bits.get(byte_idx).copied().unwrap_or(0);
            let next_byte = vec.bits.get(byte_idx + 1).copied().unwrap_or(0);

            let pattern = if bit_offset <= 4 {
                (byte >> bit_offset) & 0x0F
            } else {
                ((byte >> bit_offset) | (next_byte << (8 - bit_offset))) & 0x0F
            };

            dot_sum += lut[pattern as usize] as u32;
        }

        // Dequantize using RaBitQ formula
        let scale = 1.0 / (dim as f32).sqrt();

        // sum_positive = sum of q[i] where bit[i] = 1
        // = popcount * lower + (dot_sum * width / 15)
        let sum_positive = vec.popcount as f32 * query.lower + dot_sum as f32 * query.width / 15.0;

        // sum_all = D * lower + sum_q * width / 15
        let sum_all = dim as f32 * query.lower + query.sum as f32 * query.width / 15.0;

        // <q, o_bar> = scale * (2 * sum_positive - sum_all)
        let q_obar_dot = scale * (2.0 * sum_positive - sum_all);

        // Estimate <q, o> using the corrective factor <o, o_bar>
        let q_o_estimate = if vec.self_dot.abs() > 1e-6 {
            q_obar_dot / vec.self_dot
        } else {
            q_obar_dot
        };

        // Clamp to valid range
        let q_o_clamped = q_o_estimate.clamp(-1.0, 1.0);

        // Distance formula: ||o - q||^2 = ||o||^2 + ||q||^2 - 2*||o||*||q||*<o,q>
        let dist_sq = vec.dist_to_centroid * vec.dist_to_centroid
            + query.dist_to_centroid * query.dist_to_centroid
            - 2.0 * vec.dist_to_centroid * query.dist_to_centroid * q_o_clamped;

        dist_sq.max(0.0)
    }

    /// Merge multiple IVF indexes (O(1) per cluster - just concatenate)
    pub fn merge(
        indexes: &[&IVFRaBitQIndex],
        doc_id_offsets: &[u32],
    ) -> Result<Self, &'static str> {
        if indexes.is_empty() {
            return Err("No indexes to merge");
        }

        // Verify all indexes use same centroids version
        let version = indexes[0].centroids_version;
        for idx in indexes.iter().skip(1) {
            if idx.centroids_version != version {
                return Err("Cannot merge indexes with different centroid versions");
            }
        }

        let config = indexes[0].config.clone();
        let mut merged = Self::new(config, version);

        // Merge clusters
        for (seg_idx, index) in indexes.iter().enumerate() {
            let offset = doc_id_offsets[seg_idx];

            for (&cluster_id, cluster_data) in &index.clusters {
                let merged_cluster = merged.clusters.entry(cluster_id).or_default();

                merged_cluster.append(cluster_data, offset);
            }

            merged.num_vectors += index.num_vectors;
        }

        Ok(merged)
    }

    /// Get number of populated clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Get total number of vectors
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

/// Prepared query for fast distance estimation
struct PreparedQuery {
    dist_to_centroid: f32,
    lower: f32,
    width: f32,
    #[allow(dead_code)]
    sum: u32,
    luts: Vec<[u16; 16]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coarse_centroids_train() {
        // Generate random vectors
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..64).map(|_| rng.random::<f32>()).collect())
            .collect();

        let centroids = CoarseCentroids::train(&vectors, 16, 10, 42);

        assert_eq!(centroids.num_clusters, 16);
        assert_eq!(centroids.dim, 64);
        assert_eq!(centroids.centroids.len(), 16 * 64);
    }

    #[test]
    fn test_coarse_centroids_save_load() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..32).map(|_| rng.random::<f32>()).collect())
            .collect();

        let centroids = CoarseCentroids::train(&vectors, 8, 5, 42);
        let bytes = centroids.to_bytes().unwrap();
        let loaded = CoarseCentroids::from_bytes(&bytes).unwrap();

        assert_eq!(centroids.num_clusters, loaded.num_clusters);
        assert_eq!(centroids.dim, loaded.dim);
        assert_eq!(centroids.centroids, loaded.centroids);
    }

    #[test]
    fn test_ivf_build_and_search() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dim = 64;

        // Generate vectors
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        // Train centroids
        let centroids = CoarseCentroids::train(&vectors, 16, 10, 42);

        // Build index
        let config = IVFConfig::new(dim);
        let index = IVFRaBitQIndex::build(config, &centroids, &vectors, None);

        assert_eq!(index.len(), 1000);
        assert!(index.num_clusters() <= 16);

        // Search
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let results = index.search(&centroids, &query, 10, 4);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_ivf_merge() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dim = 32;

        // Generate vectors for two segments
        let vectors1: Vec<Vec<f32>> = (0..500)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();
        let vectors2: Vec<Vec<f32>> = (0..500)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        // Train centroids (shared)
        let all_vectors: Vec<Vec<f32>> = vectors1.iter().chain(vectors2.iter()).cloned().collect();
        let centroids = CoarseCentroids::train(&all_vectors, 8, 10, 42);

        // Build two indexes
        let config = IVFConfig::new(dim);
        let index1 = IVFRaBitQIndex::build(config.clone(), &centroids, &vectors1, None);
        let index2 = IVFRaBitQIndex::build(config, &centroids, &vectors2, None);

        // Merge
        let merged = IVFRaBitQIndex::merge(&[&index1, &index2], &[0, 500]).unwrap();

        assert_eq!(merged.len(), 1000);

        // Search merged index
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let results = merged.search(&centroids, &query, 10, 4);

        assert_eq!(results.len(), 10);
    }
}
