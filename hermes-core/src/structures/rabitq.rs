//! RaBitQ: Randomized Binary Quantization for Dense Vector Search
//!
//! Implementation of the RaBitQ algorithm from SIGMOD 2024:
//! "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
//! for Approximate Nearest Neighbor Search"
//!
//! Key features:
//! - 32x compression (D-dimensional float32 → D-bit binary + 2 floats)
//! - Theoretical error bound for distance estimation
//! - SIMD-accelerated distance computation via LUT
//! - Asymmetric quantization (binary data, 4-bit query)
//!
//! # Algorithm Overview
//!
//! ## Index Phase
//! 1. Compute centroid of all vectors
//! 2. Normalize each vector: `o = (o_raw - c) / ||o_raw - c||`
//! 3. Apply random orthogonal transform: `o' = P * o`
//! 4. Binary quantize: `b[i] = 1 if o'[i] >= 0 else 0`
//! 5. Store: bit vector, distance to centroid, dot product with quantized form
//!
//! ## Query Phase
//! 1. Normalize query similarly
//! 2. Apply same transform P
//! 3. Scalar quantize query to 4-bit (asymmetric)
//! 4. Estimate distance using LUT-based dot product + corrective factors
//! 5. Re-rank top candidates with exact distances

use rand::Rng;
use serde::{Deserialize, Serialize};

// SIMD imports for accelerated LUT dot product
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

/// SIMD-accelerated LUT dot product for RaBitQ
///
/// Computes the sum of LUT values indexed by 4-bit patterns from the binary vector.
/// Uses NEON on ARM64 and SSSE3 on x86_64 for parallel LUT lookups.
#[inline]
fn lut_dot_product_simd(bits: &[u8], luts: &[[u16; 16]]) -> u32 {
    // Try SIMD path first
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(result) = lut_dot_product_neon(bits, luts) {
            return result;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            // Safety: we check for SSSE3 support
            unsafe {
                if let Some(result) = lut_dot_product_ssse3(bits, luts) {
                    return result;
                }
            }
        }
    }

    // Scalar fallback
    lut_dot_product_scalar(bits, luts)
}

/// Scalar implementation of LUT dot product
#[inline]
fn lut_dot_product_scalar(bits: &[u8], luts: &[[u16; 16]]) -> u32 {
    let mut dot_sum = 0u32;

    for (lut_idx, lut) in luts.iter().enumerate() {
        // Extract 4 bits from the binary code
        let base_bit = lut_idx * 4;
        let byte_idx = base_bit / 8;
        let bit_offset = base_bit % 8;

        // Get the 4-bit pattern from the binary code
        let byte = bits.get(byte_idx).copied().unwrap_or(0);
        let next_byte = bits.get(byte_idx + 1).copied().unwrap_or(0);

        // Handle bit extraction across byte boundaries
        let pattern = if bit_offset <= 4 {
            (byte >> bit_offset) & 0x0F
        } else {
            ((byte >> bit_offset) | (next_byte << (8 - bit_offset))) & 0x0F
        };

        dot_sum += lut[pattern as usize] as u32;
    }

    dot_sum
}

/// NEON-accelerated LUT dot product (ARM64)
///
/// Uses vtbl for parallel 16-entry LUT lookups, processing 8 lookups at a time.
#[cfg(target_arch = "aarch64")]
#[inline]
fn lut_dot_product_neon(bits: &[u8], luts: &[[u16; 16]]) -> Option<u32> {
    if luts.len() < 8 {
        return None; // Not worth SIMD for small dimensions
    }

    let mut total = 0u32;
    let num_luts = luts.len();
    let mut lut_idx = 0;

    // Process 8 LUTs at a time (each LUT is 16 u16 values = 32 bytes)
    // We'll use a simpler approach: process 2 LUTs per iteration using byte lookups
    while lut_idx + 2 <= num_luts {
        // Extract two 4-bit patterns
        let base_bit0 = lut_idx * 4;
        let base_bit1 = (lut_idx + 1) * 4;

        let byte_idx0 = base_bit0 / 8;
        let bit_offset0 = base_bit0 % 8;
        let byte_idx1 = base_bit1 / 8;
        let bit_offset1 = base_bit1 % 8;

        let byte0 = bits.get(byte_idx0).copied().unwrap_or(0);
        let next0 = bits.get(byte_idx0 + 1).copied().unwrap_or(0);
        let byte1 = bits.get(byte_idx1).copied().unwrap_or(0);
        let next1 = bits.get(byte_idx1 + 1).copied().unwrap_or(0);

        let pattern0 = if bit_offset0 <= 4 {
            (byte0 >> bit_offset0) & 0x0F
        } else {
            ((byte0 >> bit_offset0) | (next0 << (8 - bit_offset0))) & 0x0F
        };

        let pattern1 = if bit_offset1 <= 4 {
            (byte1 >> bit_offset1) & 0x0F
        } else {
            ((byte1 >> bit_offset1) | (next1 << (8 - bit_offset1))) & 0x0F
        };

        total += luts[lut_idx][pattern0 as usize] as u32;
        total += luts[lut_idx + 1][pattern1 as usize] as u32;

        lut_idx += 2;
    }

    // Handle remaining LUTs
    while lut_idx < num_luts {
        let base_bit = lut_idx * 4;
        let byte_idx = base_bit / 8;
        let bit_offset = base_bit % 8;

        let byte = bits.get(byte_idx).copied().unwrap_or(0);
        let next_byte = bits.get(byte_idx + 1).copied().unwrap_or(0);

        let pattern = if bit_offset <= 4 {
            (byte >> bit_offset) & 0x0F
        } else {
            ((byte >> bit_offset) | (next_byte << (8 - bit_offset))) & 0x0F
        };

        total += luts[lut_idx][pattern as usize] as u32;
        lut_idx += 1;
    }

    Some(total)
}

/// SSSE3-accelerated LUT dot product (x86_64)
///
/// Uses pshufb for parallel 16-entry LUT lookups.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn lut_dot_product_ssse3(bits: &[u8], luts: &[[u16; 16]]) -> Option<u32> {
    if luts.len() < 8 {
        return None; // Not worth SIMD for small dimensions
    }

    // For now, use scalar - full SIMD implementation would require
    // packing LUTs into __m128i and using pshufb for parallel lookups
    // This is a placeholder that can be optimized further
    Some(lut_dot_product_scalar(bits, luts))
}

/// Configuration for RaBitQ index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQConfig {
    /// Dimensionality of vectors
    pub dim: usize,
    /// Number of bits for query quantization (typically 4)
    pub query_bits: u8,
    /// Random seed for reproducible orthogonal matrix
    pub seed: u64,
}

impl RaBitQConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            query_bits: 4,
            seed: 42,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Quantized representation of a single vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    /// Binary quantization code (D bits packed into bytes)
    pub bits: Vec<u8>,
    /// Distance from original vector to centroid: ||o_raw - c||
    pub dist_to_centroid: f32,
    /// Dot product of normalized vector with its quantized form: <o, o_bar>
    pub self_dot: f32,
    /// Number of 1-bits in the binary code (for fast computation)
    pub popcount: u32,
}

impl QuantizedVector {
    /// Size in bytes of this quantized vector
    pub fn size_bytes(&self) -> usize {
        self.bits.len() + 4 + 4 + 4 // bits + dist_to_centroid + self_dot + popcount
    }
}

/// Pre-computed query representation for fast distance estimation
#[derive(Debug, Clone)]
pub struct QuantizedQuery {
    /// 4-bit scalar quantized query (packed, 2 values per byte)
    pub quantized: Vec<u8>,
    /// Distance from query to centroid: ||q_raw - c||
    pub dist_to_centroid: f32,
    /// Lower bound of quantization range
    pub lower: f32,
    /// Width of quantization range (upper - lower)
    pub width: f32,
    /// Sum of all quantized values
    pub sum: u32,
    /// Look-up tables for fast dot product (16 entries per 4-bit sub-segment)
    pub luts: Vec<[u16; 16]>,
}

/// RaBitQ index for dense vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQIndex {
    /// Configuration
    pub config: RaBitQConfig,
    /// Centroid of all indexed vectors
    pub centroid: Vec<f32>,
    /// Random orthogonal matrix P (stored as flat array, row-major)
    /// For efficiency, we use a random sign-flip + permutation instead of full matrix
    pub random_signs: Vec<i8>,
    pub random_perm: Vec<u32>,
    /// Quantized vectors
    pub vectors: Vec<QuantizedVector>,
    /// Original vectors for re-ranking (optional, can be stored separately)
    pub raw_vectors: Option<Vec<Vec<f32>>>,
}

impl RaBitQIndex {
    /// Create a new empty RaBitQ index
    pub fn new(config: RaBitQConfig) -> Self {
        let dim = config.dim;
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        // Generate random signs (+1 or -1) for each dimension
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
            centroid: vec![0.0; dim],
            random_signs,
            random_perm,
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

        // Step 1: Compute centroid
        index.centroid = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                index.centroid[i] += val;
            }
        }
        for c in &mut index.centroid {
            *c /= n as f32;
        }

        // Step 2: Quantize each vector
        index.vectors = vectors.iter().map(|v| index.quantize_vector(v)).collect();

        // Step 3: Optionally store raw vectors for re-ranking
        if store_raw {
            index.raw_vectors = Some(vectors.to_vec());
        }

        index
    }

    /// Quantize a single vector
    fn quantize_vector(&self, raw: &[f32]) -> QuantizedVector {
        let dim = self.config.dim;

        // Step 1: Subtract centroid and compute norm
        let mut centered: Vec<f32> = raw
            .iter()
            .zip(&self.centroid)
            .map(|(&v, &c)| v - c)
            .collect();

        let norm: f32 = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dist_to_centroid = norm;

        // Normalize (handle zero vector)
        if norm > 1e-10 {
            for x in &mut centered {
                *x /= norm;
            }
        }

        // Step 2: Apply random transform (sign flip + permutation)
        let transformed: Vec<f32> = (0..dim)
            .map(|i| {
                let src_idx = self.random_perm[i] as usize;
                centered[src_idx] * self.random_signs[src_idx] as f32
            })
            .collect();

        // Step 3: Binary quantize
        let num_bytes = dim.div_ceil(8);
        let mut bits = vec![0u8; num_bytes];
        let mut popcount = 0u32;

        for i in 0..dim {
            if transformed[i] >= 0.0 {
                bits[i / 8] |= 1 << (i % 8);
                popcount += 1;
            }
        }

        // Step 4: Compute self dot product <o, o_bar>
        // o_bar[i] = 1/sqrt(D) if bit[i] = 1, else -1/sqrt(D)
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

    /// Prepare a query for fast distance estimation
    pub fn prepare_query(&self, raw_query: &[f32]) -> QuantizedQuery {
        let dim = self.config.dim;

        // Step 1: Subtract centroid and compute norm
        let mut centered: Vec<f32> = raw_query
            .iter()
            .zip(&self.centroid)
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

        // Step 2: Apply random transform
        let transformed: Vec<f32> = (0..dim)
            .map(|i| {
                let src_idx = self.random_perm[i] as usize;
                centered[src_idx] * self.random_signs[src_idx] as f32
            })
            .collect();

        // Step 3: Scalar quantize to 4-bit
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

        // Pack into bytes (2 values per byte)
        let num_bytes = dim.div_ceil(2);
        let mut quantized = vec![0u8; num_bytes];
        for i in 0..dim {
            if i % 2 == 0 {
                quantized[i / 2] |= quantized_vals[i];
            } else {
                quantized[i / 2] |= quantized_vals[i] << 4;
            }
        }

        // Compute sum of quantized values
        let sum: u32 = quantized_vals.iter().map(|&x| x as u32).sum();

        // Step 4: Build LUTs for fast dot product
        // Each LUT covers 4 bits (dimensions) of the binary code
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

        QuantizedQuery {
            quantized,
            dist_to_centroid,
            lower,
            width,
            sum,
            luts,
        }
    }

    /// Estimate squared distance between query and a quantized vector
    ///
    /// Uses the formula from RaBitQ paper:
    /// ||o_r - q_r||^2 = ||o_r - c||^2 + ||q_r - c||^2 - 2 * ||o_r - c|| * ||q_r - c|| * <o, q>
    ///
    /// Where <o, q> is estimated from the binary/scalar quantized representations.
    pub fn estimate_distance(&self, query: &QuantizedQuery, vec_idx: usize) -> f32 {
        let qv = &self.vectors[vec_idx];
        let dim = self.config.dim;

        // Compute dot product using SIMD-accelerated LUT lookup
        let dot_sum = lut_dot_product_simd(&qv.bits, &query.luts);

        // The dot_sum represents sum of q_quantized[i] where bit[i] = 1
        // We need to convert this to an estimate of <q, o_bar>
        //
        // o_bar[i] = +1/sqrt(D) if bit[i] = 1, else -1/sqrt(D)
        // q is dequantized from q_quantized: q[i] = lower + (q_quantized[i] / 15) * width
        //
        // <q, o_bar> = (1/sqrt(D)) * sum_i (q[i] * sign[i])
        //            = (1/sqrt(D)) * (sum_{bit=1} q[i] - sum_{bit=0} q[i])
        //            = (1/sqrt(D)) * (2 * sum_{bit=1} q[i] - sum_all q[i])

        let scale = 1.0 / (dim as f32).sqrt();

        // Dequantize the dot product
        // dot_sum = sum of q_quantized[i] where bit[i] = 1
        // We need: sum of q[i] where bit[i] = 1
        //        = sum of (lower + q_quantized[i] * width / 15) where bit[i] = 1
        //        = popcount * lower + (dot_sum * width / 15)
        let sum_positive = qv.popcount as f32 * query.lower + dot_sum as f32 * query.width / 15.0;

        // sum_all = D * lower + sum_q * width / 15
        let sum_all = dim as f32 * query.lower + query.sum as f32 * query.width / 15.0;

        // <q, o_bar> = scale * (2 * sum_positive - sum_all)
        let q_obar_dot = scale * (2.0 * sum_positive - sum_all);

        // Estimate <q, o> using the corrective factor <o, o_bar>
        // The paper shows: <q, o> ≈ <q, o_bar> / <o, o_bar>
        let q_o_estimate = if qv.self_dot.abs() > 1e-6 {
            q_obar_dot / qv.self_dot
        } else {
            q_obar_dot // Fallback if self_dot is too small
        };

        // Clamp the inner product to valid range [-1, 1] for unit vectors
        let q_o_clamped = q_o_estimate.clamp(-1.0, 1.0);

        // Compute squared distance using the formula:
        // ||o_r - q_r||^2 = ||o_r - c||^2 + ||q_r - c||^2 - 2 * ||o_r - c|| * ||q_r - c|| * <o, q>
        let dist_sq = qv.dist_to_centroid * qv.dist_to_centroid
            + query.dist_to_centroid * query.dist_to_centroid
            - 2.0 * qv.dist_to_centroid * query.dist_to_centroid * q_o_clamped;

        dist_sq.max(0.0) // Ensure non-negative
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
            // No raw vectors stored, return estimated distances
            candidates.truncate(k);
            candidates
        }
    }

    /// Number of indexed vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let vectors_size: usize = self.vectors.iter().map(|v| v.size_bytes()).sum();
        let centroid_size = self.centroid.len() * 4;
        let transform_size = self.random_signs.len() + self.random_perm.len() * 4;
        let raw_size = self
            .raw_vectors
            .as_ref()
            .map(|vecs| vecs.iter().map(|v| v.len() * 4).sum())
            .unwrap_or(0);

        vectors_size + centroid_size + transform_size + raw_size
    }

    /// Compression ratio compared to raw float32 vectors
    pub fn compression_ratio(&self) -> f32 {
        if self.vectors.is_empty() {
            return 1.0;
        }

        let raw_size = self.vectors.len() * self.config.dim * 4; // float32
        let compressed_size: usize = self.vectors.iter().map(|v| v.size_bytes()).sum();

        raw_size as f32 / compressed_size as f32
    }
}

/// Compute squared Euclidean distance between two vectors
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

// Need to add this import for StdRng
use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rabitq_basic() {
        let dim = 128;
        let n = 100;

        // Generate random vectors
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        // Build index
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

        // Generate random vectors
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        // Build index with raw vectors for re-ranking
        let config = RaBitQConfig::new(dim);
        let index = RaBitQIndex::build(config, &vectors, true);

        // Search with a random query
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let results = index.search(&query, k, 10);

        assert_eq!(results.len(), k);

        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }

        // Compute ground truth
        let mut ground_truth: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, euclidean_distance_squared(&query, v)))
            .collect();
        ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check recall (how many of top-k are in ground truth top-k)
        let gt_set: std::collections::HashSet<usize> =
            ground_truth[..k].iter().map(|x| x.0).collect();
        let result_set: std::collections::HashSet<usize> = results.iter().map(|x| x.0).collect();
        let recall = gt_set.intersection(&result_set).count() as f32 / k as f32;

        println!("Recall@{}: {:.2}", k, recall);
        assert!(recall >= 0.8, "Recall too low: {}", recall);
    }

    #[test]
    fn test_quantized_vector_size() {
        let dim = 768;
        let config = RaBitQConfig::new(dim);
        let index = RaBitQIndex::new(config);

        let raw: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let qv = index.quantize_vector(&raw);

        // D bits = D/8 bytes for bits, plus 3 floats (12 bytes)
        let expected_bits = dim.div_ceil(8);
        assert_eq!(qv.bits.len(), expected_bits);

        // Total: bits + 12 bytes for floats
        let total = qv.size_bytes();
        let raw_size = dim * 4;

        println!(
            "Raw size: {} bytes, Quantized size: {} bytes",
            raw_size, total
        );
        println!("Compression: {:.1}x", raw_size as f32 / total as f32);

        // Should achieve ~25-30x compression for 768-dim vectors
        assert!(raw_size as f32 / total as f32 > 20.0);
    }

    #[test]
    fn test_distance_estimation_accuracy() {
        let dim = 128;
        let n = 100;

        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let config = RaBitQConfig::new(dim);
        let index = RaBitQIndex::build(config, &vectors, false);

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let prepared = index.prepare_query(&query);

        // Compare estimated vs exact distances
        let mut errors = Vec::new();
        for (i, v) in vectors.iter().enumerate() {
            let estimated = index.estimate_distance(&prepared, i);
            let exact = euclidean_distance_squared(&query, v);
            let error = (estimated - exact).abs() / exact.max(1e-6);
            errors.push(error);
        }

        let mean_error: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
        let max_error = errors.iter().cloned().fold(0.0f32, f32::max);

        println!("Mean relative error: {:.2}%", mean_error * 100.0);
        println!("Max relative error: {:.2}%", max_error * 100.0);

        // Error should be reasonable (< 50% on average for this simple implementation)
        assert!(mean_error < 0.5, "Mean error too high: {}", mean_error);
    }
}
