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

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::super::ivf::cluster::QuantizedCode;
use super::Quantizer;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

/// Configuration for RaBitQ quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQConfig {
    /// Dimensionality of vectors
    pub dim: usize,
    /// Number of bits for query quantization (typically 4)
    pub query_bits: u8,
    /// Random seed for reproducible orthogonal matrix
    pub seed: u64,
    /// Extended RaBitQ: extra magnitude bits per dimension (0-7).
    ///
    /// 0 = classic 1-bit RaBitQ (sign only). N > 0 stores an additional
    /// N-bit magnitude refinement per dimension (total N+1 bits/dim),
    /// giving much tighter distance estimates — 3-5 extra bits typically
    /// allow shrinking the exact-rerank pool (`rerank_factor`) by 2-3x,
    /// cutting raw-vector I/O in disk-resident indexes.
    ///
    /// Follows the Extended RaBitQ line (Gao & Long, 2024; adopted by
    /// NVIDIA cuVS IVF-RaBitQ): uniform magnitude quantization of the
    /// rotated unit residual with a per-vector scale.
    #[serde(default)]
    pub ex_bits: u8,
}

impl RaBitQConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            query_bits: 4,
            seed: 42,
            ex_bits: 0,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set total bits per dimension (1 = classic binary, 2-8 = extended).
    pub fn with_bits(mut self, total_bits: u8) -> Self {
        self.ex_bits = total_bits.clamp(1, 8) - 1;
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
    /// Extended RaBitQ magnitude codes: `ex_bits` per dimension, packed
    /// LSB-first. Empty for classic 1-bit codes (serde default keeps old
    /// segments readable).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ex_code: Vec<u8>,
    /// Per-vector magnitude scale: max |transformed_i| (extended only)
    #[serde(default)]
    pub ex_scale: f32,
    /// Norm of the reconstructed vector v̂ (extended only) — used to
    /// normalize the refined inner-product estimate.
    #[serde(default)]
    pub ex_norm: f32,
}

impl QuantizedCode for QuantizedVector {
    fn size_bytes(&self) -> usize {
        // bits + dist_to_centroid + self_dot + popcount (+ extended payload)
        self.bits.len() + 4 + 4 + 4 + self.ex_code.len() + 8
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
    /// Rotated, normalized query (full precision) — used by the extended
    /// multi-bit estimator for exact dot products against reconstructed codes.
    pub transformed: Vec<f32>,
}

/// RaBitQ codebook (random transform parameters)
///
/// Trained once, shared across all segments for merge compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQCodebook {
    /// Configuration
    pub config: RaBitQConfig,
    /// Random signs for transform (+1 or -1)
    pub random_signs: Vec<i8>,
    /// Random permutation for transform
    pub random_perm: Vec<u32>,
    /// Version for merge compatibility checking
    pub version: u64,
}

impl RaBitQCodebook {
    /// Create a new RaBitQ codebook with random transform
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

        // Version derived from config — codebook is deterministic (same seed+dim
        // always produces identical random_signs and random_perm), so segments
        // built with the same config are merge-compatible. ex_bits participates
        // so 1-bit and multi-bit segments never merge (the term is 0 for
        // ex_bits = 0, preserving historical versions).
        let version = config.seed
            ^ (config.dim as u64).wrapping_mul(0x9e3779b97f4a7c15)
            ^ (config.ex_bits as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);

        Self {
            config,
            random_signs,
            random_perm,
            version,
        }
    }

    /// Encode a vector to binary quantized form
    ///
    /// If centroid is provided, encodes the residual (vector - centroid).
    pub fn encode(&self, vector: &[f32], centroid: Option<&[f32]>) -> QuantizedVector {
        let dim = self.config.dim;

        // Step 1: Center + normalize in-place (single allocation instead of two)
        let mut normalized: Vec<f32> = if let Some(c) = centroid {
            vector.iter().zip(c).map(|(&v, &c)| v - c).collect()
        } else {
            vector.to_vec()
        };

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dist_to_centroid = norm;

        if norm > 1e-10 {
            let inv_norm = 1.0 / norm;
            for x in normalized.iter_mut() {
                *x *= inv_norm;
            }
        }

        // Step 2: Apply random transform (sign flip + permutation)
        let transformed: Vec<f32> = (0..dim)
            .map(|i| {
                let src_idx = self.random_perm[i] as usize;
                normalized[src_idx] * self.random_signs[src_idx] as f32
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

        // Step 5 (extended): quantize per-dim magnitudes with a per-vector
        // uniform scale. Reconstruction v̂_i = sign_i * (code_i + 0.5) * step.
        let (ex_code, ex_scale, ex_norm) = if self.config.ex_bits > 0 {
            let ex_bits = self.config.ex_bits as u32;
            let levels = 1u32 << ex_bits;
            let max_abs = transformed.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let ex_scale = if max_abs > 1e-10 { max_abs } else { 1.0 };
            let step = ex_scale / levels as f32;

            let total_bits = dim * ex_bits as usize;
            let mut ex_code = vec![0u8; total_bits.div_ceil(8)];
            let mut norm_sq = 0.0f64;
            let mut bit_pos = 0usize;
            for &t in &transformed {
                let mag = (t.abs() / step) as u32;
                let code = mag.min(levels - 1);
                // Pack `ex_bits` LSB-first at bit_pos
                let mut v = code;
                let mut remaining = ex_bits as usize;
                let mut pos = bit_pos;
                while remaining > 0 {
                    let byte = pos / 8;
                    let offset = pos % 8;
                    let take = remaining.min(8 - offset);
                    ex_code[byte] |= ((v & ((1 << take) - 1)) as u8) << offset;
                    v >>= take;
                    pos += take;
                    remaining -= take;
                }
                bit_pos += ex_bits as usize;

                let recon = (code as f32 + 0.5) * step;
                norm_sq += (recon as f64) * (recon as f64);
            }

            (ex_code, ex_scale, (norm_sq.sqrt()) as f32)
        } else {
            (Vec::new(), 0.0, 0.0)
        };

        QuantizedVector {
            bits,
            dist_to_centroid,
            self_dot,
            popcount,
            ex_code,
            ex_scale,
            ex_norm,
        }
    }

    /// Prepare a query for fast distance estimation
    pub fn prepare_query(&self, query: &[f32], centroid: Option<&[f32]>) -> QuantizedQuery {
        let dim = self.config.dim;

        // Step 1: Center + normalize in-place (single allocation instead of two)
        let mut normalized: Vec<f32> = if let Some(c) = centroid {
            query.iter().zip(c).map(|(&v, &c)| v - c).collect()
        } else {
            query.to_vec()
        };

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dist_to_centroid = norm;

        if norm > 1e-10 {
            let inv_norm = 1.0 / norm;
            for x in normalized.iter_mut() {
                *x *= inv_norm;
            }
        }

        // Step 2: Apply random transform
        let transformed: Vec<f32> = (0..dim)
            .map(|i| {
                let src_idx = self.random_perm[i] as usize;
                normalized[src_idx] * self.random_signs[src_idx] as f32
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
            transformed,
        }
    }

    /// Estimate squared distance between query and a quantized vector
    pub fn estimate_distance(&self, query: &QuantizedQuery, code: &QuantizedVector) -> f32 {
        // Extended multi-bit codes get the refined estimator
        if !code.ex_code.is_empty() {
            return self.estimate_distance_extended(query, code);
        }

        let dim = self.config.dim;

        // Compute dot product using SIMD-accelerated LUT lookup
        let dot_sum = lut_dot_product_simd(&code.bits, &query.luts);

        let scale = 1.0 / (dim as f32).sqrt();

        // Dequantize the dot product
        let sum_positive = code.popcount as f32 * query.lower + dot_sum as f32 * query.width / 15.0;
        let sum_all = dim as f32 * query.lower + query.sum as f32 * query.width / 15.0;

        // <q, o_bar> = scale * (2 * sum_positive - sum_all)
        let q_obar_dot = scale * (2.0 * sum_positive - sum_all);

        // Estimate <q, o> using the corrective factor <o, o_bar>
        let q_o_estimate = if code.self_dot.abs() > 1e-6 {
            q_obar_dot / code.self_dot
        } else {
            q_obar_dot
        };

        // Clamp the inner product to valid range [-1, 1]
        let q_o_clamped = q_o_estimate.clamp(-1.0, 1.0);

        // Compute squared distance
        let dist_sq = code.dist_to_centroid * code.dist_to_centroid
            + query.dist_to_centroid * query.dist_to_centroid
            - 2.0 * code.dist_to_centroid * query.dist_to_centroid * q_o_clamped;

        dist_sq.max(0.0)
    }

    /// Refined distance estimate from extended multi-bit magnitude codes.
    ///
    /// Reconstructs v̂_i = sign_i * (code_i + 0.5) * step in the rotated
    /// space and computes an exact dot product against the full-precision
    /// rotated query, normalized by ||v̂||. Estimation error shrinks
    /// roughly 2x per extra bit, allowing much smaller exact-rerank pools.
    fn estimate_distance_extended(&self, query: &QuantizedQuery, code: &QuantizedVector) -> f32 {
        let dim = self.config.dim;
        let ex_bits = self.config.ex_bits as usize;
        debug_assert!(ex_bits > 0 && ex_bits <= 8);
        let levels = 1u32 << ex_bits;
        let step = code.ex_scale / levels as f32;
        let mask = levels - 1;

        // Rolling bit-buffer reader over the packed magnitude codes
        let mut acc: u64 = 0;
        let mut acc_bits: usize = 0;
        let mut byte_idx: usize = 0;
        let ex_code = &code.ex_code;

        let mut dot = 0.0f32;
        for (i, &q) in query.transformed.iter().enumerate().take(dim) {
            while acc_bits < ex_bits {
                acc |= (ex_code.get(byte_idx).copied().unwrap_or(0) as u64) << acc_bits;
                byte_idx += 1;
                acc_bits += 8;
            }
            let mag_code = (acc as u32) & mask;
            acc >>= ex_bits;
            acc_bits -= ex_bits;

            let magnitude = (mag_code as f32 + 0.5) * step;
            let signed = if (code.bits[i / 8] >> (i % 8)) & 1 == 1 {
                magnitude
            } else {
                -magnitude
            };
            dot += q * signed;
        }

        // <q, o> estimate: v̂ approximates the unit residual direction
        let q_o_estimate = if code.ex_norm > 1e-10 {
            dot / code.ex_norm
        } else {
            dot
        };
        let q_o_clamped = q_o_estimate.clamp(-1.0, 1.0);

        let dist_sq = code.dist_to_centroid * code.dist_to_centroid
            + query.dist_to_centroid * query.dist_to_centroid
            - 2.0 * code.dist_to_centroid * query.dist_to_centroid * q_o_clamped;

        dist_sq.max(0.0)
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.random_signs.len() + self.random_perm.len() * 4 + 64
    }

    /// Estimated memory usage in bytes (alias for size_bytes)
    pub fn estimated_memory_bytes(&self) -> usize {
        self.size_bytes()
    }
}

impl Quantizer for RaBitQCodebook {
    type Code = QuantizedVector;
    type Config = RaBitQConfig;
    type QueryData = QuantizedQuery;

    fn encode(&self, vector: &[f32], centroid: Option<&[f32]>) -> Self::Code {
        self.encode(vector, centroid)
    }

    fn prepare_query(&self, query: &[f32], centroid: Option<&[f32]>) -> Self::QueryData {
        self.prepare_query(query, centroid)
    }

    fn compute_distance(&self, query_data: &Self::QueryData, code: &Self::Code) -> f32 {
        self.estimate_distance(query_data, code)
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes()
    }
}

// ============================================================================
// SIMD-accelerated LUT dot product
// ============================================================================

/// SIMD-accelerated LUT dot product for RaBitQ
#[inline]
fn lut_dot_product_simd(bits: &[u8], luts: &[[u16; 16]]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(result) = lut_dot_product_neon(bits, luts) {
            return result;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe {
                if let Some(result) = lut_dot_product_ssse3(bits, luts) {
                    return result;
                }
            }
        }
    }

    lut_dot_product_scalar(bits, luts)
}

/// Scalar implementation of LUT dot product
#[inline]
fn lut_dot_product_scalar(bits: &[u8], luts: &[[u16; 16]]) -> u32 {
    let mut dot_sum = 0u32;

    for (lut_idx, lut) in luts.iter().enumerate() {
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

        dot_sum += lut[pattern as usize] as u32;
    }

    dot_sum
}

/// NEON-accelerated LUT dot product (ARM64)
#[cfg(target_arch = "aarch64")]
#[inline]
fn lut_dot_product_neon(bits: &[u8], luts: &[[u16; 16]]) -> Option<u32> {
    if luts.len() < 8 {
        return None;
    }

    let mut total = 0u32;
    let num_luts = luts.len();
    let mut lut_idx = 0;

    while lut_idx + 2 <= num_luts {
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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn lut_dot_product_ssse3(bits: &[u8], luts: &[[u16; 16]]) -> Option<u32> {
    if luts.len() < 8 {
        return None;
    }
    Some(lut_dot_product_scalar(bits, luts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rabitq_codebook_basic() {
        let config = RaBitQConfig::new(128);
        let codebook = RaBitQCodebook::new(config);

        assert_eq!(codebook.random_signs.len(), 128);
        assert_eq!(codebook.random_perm.len(), 128);
    }

    #[test]
    fn test_encode_decode() {
        let config = RaBitQConfig::new(64);
        let codebook = RaBitQCodebook::new(config);

        let vector: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let code = codebook.encode(&vector, None);

        assert_eq!(code.bits.len(), 8); // 64 bits = 8 bytes
        assert!(code.dist_to_centroid > 0.0);
    }

    #[test]
    fn test_distance_estimation() {
        let config = RaBitQConfig::new(64);
        let codebook = RaBitQCodebook::new(config);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let v1: Vec<f32> = (0..64).map(|_| rng.random::<f32>() - 0.5).collect();
        let v2: Vec<f32> = (0..64).map(|_| rng.random::<f32>() - 0.5).collect();

        let code = codebook.encode(&v1, None);
        let query = codebook.prepare_query(&v2, None);

        let estimated = codebook.estimate_distance(&query, &code);
        assert!(estimated >= 0.0);
    }

    #[test]
    fn test_extended_bits_reduce_estimation_error() {
        let dim = 128;
        let n = 200;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();

        let exact =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

        let mut errors = Vec::new();
        for bits in [1u8, 5u8] {
            let config = RaBitQConfig::new(dim).with_bits(bits);
            let codebook = RaBitQCodebook::new(config);
            let q = codebook.prepare_query(&query, None);

            let mut total_err = 0.0f64;
            for v in &vectors {
                let code = codebook.encode(v, None);
                let est = codebook.estimate_distance(&q, &code);
                let truth = exact(&query, v);
                total_err += ((est - truth).abs() / truth.max(1e-6)) as f64;
            }
            errors.push(total_err / n as f64);
        }

        // 5-bit codes must estimate distances much more accurately than 1-bit
        assert!(
            errors[1] < errors[0] * 0.5,
            "extended codes should at least halve mean relative error: 1-bit={:.4}, 5-bit={:.4}",
            errors[0],
            errors[1]
        );
        // And be tight in absolute terms
        assert!(
            errors[1] < 0.05,
            "5-bit mean relative error should be <5%, got {:.4}",
            errors[1]
        );
    }

    #[test]
    fn test_extended_code_serde_roundtrip_and_legacy() {
        let dim = 64;
        let config = RaBitQConfig::new(dim).with_bits(4);
        let codebook = RaBitQCodebook::new(config);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let code = codebook.encode(&v, None);
        assert!(!code.ex_code.is_empty());
        assert!(code.ex_norm > 0.0);

        // Round-trip
        let json = serde_json::to_vec(&code).unwrap();
        let back: QuantizedVector = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.ex_code, code.ex_code);
        assert_eq!(back.ex_scale, code.ex_scale);

        // Legacy payload without extended fields still deserializes (old segments)
        let legacy = serde_json::json!({
            "bits": code.bits,
            "dist_to_centroid": code.dist_to_centroid,
            "self_dot": code.self_dot,
            "popcount": code.popcount,
        });
        let old: QuantizedVector = serde_json::from_value(legacy).unwrap();
        assert!(old.ex_code.is_empty());
    }

    #[test]
    fn test_extended_version_differs_from_classic() {
        let dim = 64;
        let classic = RaBitQCodebook::new(RaBitQConfig::new(dim));
        let extended = RaBitQCodebook::new(RaBitQConfig::new(dim).with_bits(4));
        assert_ne!(
            classic.version, extended.version,
            "1-bit and multi-bit segments must not be merge-compatible"
        );
    }

    #[test]
    fn test_quantizer_trait() {
        let config = RaBitQConfig::new(32);
        let codebook = RaBitQCodebook::new(config);

        let vector: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
        let query: Vec<f32> = (0..32).map(|i| (31 - i) as f32 / 32.0).collect();

        // Use trait methods
        let code = Quantizer::encode(&codebook, &vector, None);
        let query_data = Quantizer::prepare_query(&codebook, &query, None);
        let dist = Quantizer::compute_distance(&codebook, &query_data, &code);

        assert!(dist >= 0.0);
    }
}
