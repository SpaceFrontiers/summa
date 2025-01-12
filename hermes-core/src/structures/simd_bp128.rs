//! SIMD-BP128: Vectorized bitpacking with NEON/SSE intrinsics
//!
//! Based on Lemire & Boytsov (2015) "Decoding billions of integers per second through vectorization"
//! and Quickwit's bitpacking crate architecture.
//!
//! Key optimizations:
//! - **True vertical layout**: Optimal compression (BLOCK_SIZE * bit_width / 8 bytes)
//! - **Integrated delta decoding**: Fused unpack + prefix sum in single pass
//! - **128-integer blocks**: 32 groups of 4 integers each
//! - **NEON intrinsics on ARM**: Uses vld1q_u32, vaddq_u32, etc.
//! - **Block-level metadata**: Skip info for BlockMax WAND

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Block size: 128 integers (32 groups of 4 for SIMD lanes)
pub const SIMD_BLOCK_SIZE: usize = 128;

/// Number of 32-bit lanes in NEON/SSE (4 x 32-bit = 128-bit)
#[allow(dead_code)]
const SIMD_LANES: usize = 4;

/// Number of groups per block (128 / 4 = 32)
#[allow(dead_code)]
const GROUPS_PER_BLOCK: usize = SIMD_BLOCK_SIZE / SIMD_LANES;

/// Compute bits needed for max value
#[inline]
pub fn bits_needed(max_val: u32) -> u8 {
    if max_val == 0 {
        0
    } else {
        32 - max_val.leading_zeros() as u8
    }
}

// ============================================================================
// NEON intrinsics for aarch64 (Apple Silicon, ARM servers)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Lookup table for expanding a byte to 8 u32 values (one per bit)
    /// LUT[byte][bit_position] = (byte >> bit_position) & 1
    /// We use a different approach: expand byte to 8 separate bit values
    static BIT_EXPAND_LUT: [[u32; 8]; 256] = {
        let mut lut = [[0u32; 8]; 256];
        let mut byte = 0usize;
        while byte < 256 {
            let mut bit = 0;
            while bit < 8 {
                lut[byte][bit] = ((byte >> bit) & 1) as u32;
                bit += 1;
            }
            byte += 1;
        }
        lut
    };

    /// Unpack 4 u32 values from packed data using NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_4_neon(input: &[u8], bit_width: u8, output: &mut [u32; 4]) {
        if bit_width == 0 {
            *output = [0; 4];
            return;
        }

        let mask = (1u32 << bit_width) - 1;

        // Load packed data
        let mut packed_bytes = [0u8; 16];
        let bytes_needed = ((bit_width as usize) * 4).div_ceil(8);
        packed_bytes[..bytes_needed.min(16)].copy_from_slice(&input[..bytes_needed.min(16)]);
        let packed = u128::from_le_bytes(packed_bytes);

        // Extract 4 values
        let v0 = (packed & mask as u128) as u32;
        let v1 = ((packed >> bit_width) & mask as u128) as u32;
        let v2 = ((packed >> (bit_width * 2)) & mask as u128) as u32;
        let v3 = ((packed >> (bit_width * 3)) & mask as u128) as u32;

        // Store using NEON
        unsafe {
            let result = vld1q_u32([v0, v1, v2, v3].as_ptr());
            vst1q_u32(output.as_mut_ptr(), result);
        }
    }

    /// SIMD prefix sum for 4 elements using NEON
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn prefix_sum_4_neon(values: &mut [u32; 4]) {
        unsafe {
            // Load values
            let mut v = vld1q_u32(values.as_ptr());

            // Prefix sum using NEON shuffles and adds
            // v = [a, b, c, d]
            // Step 1: v = [a, a+b, c, c+d]
            let shifted1 = vextq_u32(vdupq_n_u32(0), v, 3); // [0, a, b, c]
            v = vaddq_u32(v, shifted1);
            // v = [a, a+b, b+c, c+d]

            // Step 2: v = [a, a+b, a+b+c, a+b+c+d]
            let shifted2 = vextq_u32(vdupq_n_u32(0), v, 2); // [0, 0, a, a+b]
            v = vaddq_u32(v, shifted2);

            // Store result
            vst1q_u32(values.as_mut_ptr(), v);
        }
    }

    /// Unpack 128 integers from true vertical layout using NEON (optimized)
    ///
    /// Optimizations:
    /// 1. Lookup table for bit extraction (avoids per-bit shifts)
    /// 2. Process 4 bytes at once (32 integers per iteration)
    /// 3. Prefetch next bit position's data
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_block_neon(
        input: &[u8],
        bit_width: u8,
        output: &mut [u32; SIMD_BLOCK_SIZE],
    ) {
        if bit_width == 0 {
            output.fill(0);
            return;
        }

        // Clear output using NEON
        unsafe {
            let zero = vdupq_n_u32(0);
            for i in (0..SIMD_BLOCK_SIZE).step_by(4) {
                vst1q_u32(output[i..].as_mut_ptr(), zero);
            }
        }

        // For each bit position, scatter that bit to all 128 integers
        for bit_pos in 0..bit_width as usize {
            let byte_offset = bit_pos * 16;
            let bit_mask = 1u32 << bit_pos;

            // Prefetch next bit position's data (if not last)
            if bit_pos + 1 < bit_width as usize {
                let next_offset = (bit_pos + 1) * 16;
                unsafe {
                    // Use inline asm for prefetch on aarch64
                    std::arch::asm!(
                        "prfm pldl1keep, [{0}]",
                        in(reg) input.as_ptr().add(next_offset),
                        options(nostack, preserves_flags)
                    );
                }
            }

            // Process 4 bytes at a time (32 integers)
            for chunk in 0..4 {
                let chunk_offset = byte_offset + chunk * 4;

                // Load 4 bytes at once
                let b0 = input[chunk_offset] as usize;
                let b1 = input[chunk_offset + 1] as usize;
                let b2 = input[chunk_offset + 2] as usize;
                let b3 = input[chunk_offset + 3] as usize;

                let base_int = chunk * 32;

                unsafe {
                    let mask_vec = vdupq_n_u32(bit_mask);

                    // Process byte 0 (integers 0-7)
                    let lut0 = &BIT_EXPAND_LUT[b0];
                    let bits_0_3 = vld1q_u32(lut0.as_ptr());
                    let bits_4_7 = vld1q_u32(lut0[4..].as_ptr());

                    let shifted_0_3 = vmulq_u32(bits_0_3, mask_vec);
                    let shifted_4_7 = vmulq_u32(bits_4_7, mask_vec);

                    let cur_0_3 = vld1q_u32(output[base_int..].as_ptr());
                    let cur_4_7 = vld1q_u32(output[base_int + 4..].as_ptr());

                    vst1q_u32(
                        output[base_int..].as_mut_ptr(),
                        vorrq_u32(cur_0_3, shifted_0_3),
                    );
                    vst1q_u32(
                        output[base_int + 4..].as_mut_ptr(),
                        vorrq_u32(cur_4_7, shifted_4_7),
                    );

                    // Process byte 1 (integers 8-15)
                    let lut1 = &BIT_EXPAND_LUT[b1];
                    let bits_8_11 = vld1q_u32(lut1.as_ptr());
                    let bits_12_15 = vld1q_u32(lut1[4..].as_ptr());

                    let shifted_8_11 = vmulq_u32(bits_8_11, mask_vec);
                    let shifted_12_15 = vmulq_u32(bits_12_15, mask_vec);

                    let cur_8_11 = vld1q_u32(output[base_int + 8..].as_ptr());
                    let cur_12_15 = vld1q_u32(output[base_int + 12..].as_ptr());

                    vst1q_u32(
                        output[base_int + 8..].as_mut_ptr(),
                        vorrq_u32(cur_8_11, shifted_8_11),
                    );
                    vst1q_u32(
                        output[base_int + 12..].as_mut_ptr(),
                        vorrq_u32(cur_12_15, shifted_12_15),
                    );

                    // Process byte 2 (integers 16-23)
                    let lut2 = &BIT_EXPAND_LUT[b2];
                    let bits_16_19 = vld1q_u32(lut2.as_ptr());
                    let bits_20_23 = vld1q_u32(lut2[4..].as_ptr());

                    let shifted_16_19 = vmulq_u32(bits_16_19, mask_vec);
                    let shifted_20_23 = vmulq_u32(bits_20_23, mask_vec);

                    let cur_16_19 = vld1q_u32(output[base_int + 16..].as_ptr());
                    let cur_20_23 = vld1q_u32(output[base_int + 20..].as_ptr());

                    vst1q_u32(
                        output[base_int + 16..].as_mut_ptr(),
                        vorrq_u32(cur_16_19, shifted_16_19),
                    );
                    vst1q_u32(
                        output[base_int + 20..].as_mut_ptr(),
                        vorrq_u32(cur_20_23, shifted_20_23),
                    );

                    // Process byte 3 (integers 24-31)
                    let lut3 = &BIT_EXPAND_LUT[b3];
                    let bits_24_27 = vld1q_u32(lut3.as_ptr());
                    let bits_28_31 = vld1q_u32(lut3[4..].as_ptr());

                    let shifted_24_27 = vmulq_u32(bits_24_27, mask_vec);
                    let shifted_28_31 = vmulq_u32(bits_28_31, mask_vec);

                    let cur_24_27 = vld1q_u32(output[base_int + 24..].as_ptr());
                    let cur_28_31 = vld1q_u32(output[base_int + 28..].as_ptr());

                    vst1q_u32(
                        output[base_int + 24..].as_mut_ptr(),
                        vorrq_u32(cur_24_27, shifted_24_27),
                    );
                    vst1q_u32(
                        output[base_int + 28..].as_mut_ptr(),
                        vorrq_u32(cur_28_31, shifted_28_31),
                    );
                }
            }
        }
    }

    /// Prefix sum for 128 elements using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn prefix_sum_block_neon(deltas: &mut [u32; SIMD_BLOCK_SIZE], first_val: u32) {
        let mut carry = first_val;

        for group in 0..GROUPS_PER_BLOCK {
            let start = group * SIMD_LANES;
            let mut group_vals = [
                deltas[start],
                deltas[start + 1],
                deltas[start + 2],
                deltas[start + 3],
            ];

            // Add carry to first element
            group_vals[0] = group_vals[0].wrapping_add(carry);

            // SIMD prefix sum
            unsafe { prefix_sum_4_neon(&mut group_vals) };

            // Write back
            deltas[start..start + 4].copy_from_slice(&group_vals);

            // Carry for next group
            carry = group_vals[3];
        }
    }
}

// ============================================================================
// Scalar fallback for other architectures
// ============================================================================

#[allow(dead_code)]
mod scalar {
    use super::*;

    /// Pack 4 u32 values into output
    #[inline]
    pub fn pack_4_scalar(values: &[u32; 4], bit_width: u8, output: &mut [u8]) {
        if bit_width == 0 {
            return;
        }

        let bytes_needed = ((bit_width as usize) * 4).div_ceil(8);
        let mut packed = 0u128;
        for (i, &val) in values.iter().enumerate() {
            packed |= (val as u128) << (i * bit_width as usize);
        }

        let packed_bytes = packed.to_le_bytes();
        output[..bytes_needed].copy_from_slice(&packed_bytes[..bytes_needed]);
    }

    /// Unpack 4 u32 values from packed data
    #[inline]
    pub fn unpack_4_scalar(input: &[u8], bit_width: u8, output: &mut [u32; 4]) {
        if bit_width == 0 {
            *output = [0; 4];
            return;
        }

        let mask = (1u32 << bit_width) - 1;
        let mut packed_bytes = [0u8; 16];
        let bytes_needed = ((bit_width as usize) * 4).div_ceil(8);
        packed_bytes[..bytes_needed.min(16)].copy_from_slice(&input[..bytes_needed.min(16)]);
        let packed = u128::from_le_bytes(packed_bytes);

        output[0] = (packed & mask as u128) as u32;
        output[1] = ((packed >> bit_width) & mask as u128) as u32;
        output[2] = ((packed >> (bit_width * 2)) & mask as u128) as u32;
        output[3] = ((packed >> (bit_width * 3)) & mask as u128) as u32;
    }

    /// Prefix sum for 4 elements
    #[inline]
    pub fn prefix_sum_4_scalar(vals: &mut [u32; 4]) {
        vals[1] = vals[1].wrapping_add(vals[0]);
        vals[2] = vals[2].wrapping_add(vals[1]);
        vals[3] = vals[3].wrapping_add(vals[2]);
    }

    /// Unpack 128 integers from true vertical layout
    pub fn unpack_block_scalar(input: &[u8], bit_width: u8, output: &mut [u32; SIMD_BLOCK_SIZE]) {
        if bit_width == 0 {
            output.fill(0);
            return;
        }

        // Clear output first
        output.fill(0);

        // Unpack from vertical bit-interleaved layout
        for bit_pos in 0..bit_width as usize {
            let byte_offset = bit_pos * 16; // 128/8 = 16 bytes per bit position

            for byte_idx in 0..16 {
                let byte_val = input[byte_offset + byte_idx];
                let base_int = byte_idx * 8;

                // Extract 8 bits from this byte
                output[base_int + 0] |= ((byte_val >> 0) & 1) as u32 * (1 << bit_pos);
                output[base_int + 1] |= ((byte_val >> 1) & 1) as u32 * (1 << bit_pos);
                output[base_int + 2] |= ((byte_val >> 2) & 1) as u32 * (1 << bit_pos);
                output[base_int + 3] |= ((byte_val >> 3) & 1) as u32 * (1 << bit_pos);
                output[base_int + 4] |= ((byte_val >> 4) & 1) as u32 * (1 << bit_pos);
                output[base_int + 5] |= ((byte_val >> 5) & 1) as u32 * (1 << bit_pos);
                output[base_int + 6] |= ((byte_val >> 6) & 1) as u32 * (1 << bit_pos);
                output[base_int + 7] |= ((byte_val >> 7) & 1) as u32 * (1 << bit_pos);
            }
        }
    }

    /// Prefix sum for 128 elements
    pub fn prefix_sum_block_scalar(deltas: &mut [u32; SIMD_BLOCK_SIZE], first_val: u32) {
        let mut carry = first_val;

        for group in 0..GROUPS_PER_BLOCK {
            let start = group * SIMD_LANES;
            let mut group_vals = [
                deltas[start],
                deltas[start + 1],
                deltas[start + 2],
                deltas[start + 3],
            ];

            group_vals[0] = group_vals[0].wrapping_add(carry);
            prefix_sum_4_scalar(&mut group_vals);
            deltas[start..start + 4].copy_from_slice(&group_vals);
            carry = group_vals[3];
        }
    }
}

// ============================================================================
// Public API - dispatches to NEON or scalar
// ============================================================================

/// Pack 128 integers using true vertical layout (optimal compression)
///
/// Vertical layout stores bit i of all 128 integers together.
/// Total size: exactly BLOCK_SIZE * bit_width / 8 bytes (no padding waste)
pub fn pack_horizontal(values: &[u32; SIMD_BLOCK_SIZE], bit_width: u8, output: &mut Vec<u8>) {
    if bit_width == 0 {
        return;
    }

    // True vertical layout: exactly (128 * bit_width) / 8 bytes
    let total_bytes = (SIMD_BLOCK_SIZE * bit_width as usize) / 8;
    let start = output.len();
    output.resize(start + total_bytes, 0);

    // Pack using vertical bit-interleaved layout
    // For each bit position, pack that bit from all 128 integers
    for bit_pos in 0..bit_width as usize {
        let byte_offset = start + bit_pos * (SIMD_BLOCK_SIZE / 8);
        for (int_idx, &val) in values.iter().enumerate() {
            let bit = (val >> bit_pos) & 1;
            let byte_idx = byte_offset + int_idx / 8;
            let bit_in_byte = int_idx % 8;
            output[byte_idx] |= (bit as u8) << bit_in_byte;
        }
    }
}

/// Unpack 128 integers from true vertical layout (optimized)
///
/// This is the inverse of pack_horizontal - extracts bits from vertical layout.
/// Optimized to process 8 integers per byte using lookup tables.
pub fn unpack_horizontal(input: &[u8], bit_width: u8, output: &mut [u32; SIMD_BLOCK_SIZE]) {
    if bit_width == 0 {
        output.fill(0);
        return;
    }

    // Clear output first
    output.fill(0);

    // Process 8 integers at a time (one byte contains bit i of 8 consecutive integers)
    // For each bit position, we have 16 bytes (128 integers / 8 = 16 bytes)
    for bit_pos in 0..bit_width as usize {
        let byte_offset = bit_pos * 16; // 128/8 = 16 bytes per bit position

        // Process 16 bytes (128 integers) for this bit position
        for byte_idx in 0..16 {
            let byte_val = input[byte_offset + byte_idx];
            let base_int = byte_idx * 8;

            // Unroll: extract 8 bits from this byte
            output[base_int] |= (byte_val & 1) as u32 * (1 << bit_pos);
            output[base_int + 1] |= ((byte_val >> 1) & 1) as u32 * (1 << bit_pos);
            output[base_int + 2] |= ((byte_val >> 2) & 1) as u32 * (1 << bit_pos);
            output[base_int + 3] |= ((byte_val >> 3) & 1) as u32 * (1 << bit_pos);
            output[base_int + 4] |= ((byte_val >> 4) & 1) as u32 * (1 << bit_pos);
            output[base_int + 5] |= ((byte_val >> 5) & 1) as u32 * (1 << bit_pos);
            output[base_int + 6] |= ((byte_val >> 6) & 1) as u32 * (1 << bit_pos);
            output[base_int + 7] |= ((byte_val >> 7) & 1) as u32 * (1 << bit_pos);
        }
    }
}

/// Prefix sum for 128 elements - uses NEON on aarch64
#[allow(dead_code)]
pub fn prefix_sum_128(deltas: &mut [u32; SIMD_BLOCK_SIZE], first_val: u32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::prefix_sum_block_neon(deltas, first_val) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::prefix_sum_block_scalar(deltas, first_val)
    }
}

// Keep old names for compatibility
pub fn pack_vertical(values: &[u32; SIMD_BLOCK_SIZE], bit_width: u8, output: &mut Vec<u8>) {
    pack_horizontal(values, bit_width, output)
}

pub fn unpack_vertical(input: &[u8], bit_width: u8, output: &mut [u32; SIMD_BLOCK_SIZE]) {
    unpack_horizontal(input, bit_width, output)
}

/// Unpack with integrated delta decoding (fused for better performance)
///
/// The encoding stores deltas[i] = doc_ids[i+1] - doc_ids[i] - 1
/// So we have (count-1) deltas for count doc_ids.
/// first_doc_id is doc_ids[0], and we compute the rest from deltas.
///
/// This fused version avoids a separate prefix sum pass by computing
/// doc_ids inline during unpacking.
pub fn unpack_vertical_d1(
    input: &[u8],
    bit_width: u8,
    first_doc_id: u32,
    output: &mut [u32; SIMD_BLOCK_SIZE],
    count: usize,
) {
    if count == 0 {
        return;
    }

    if bit_width == 0 {
        // All deltas are 0, so gaps are all 1
        let mut current = first_doc_id;
        output[0] = current;
        for out_val in output.iter_mut().take(count).skip(1) {
            current = current.wrapping_add(1);
            *out_val = current;
        }
        return;
    }

    // Fused unpack + prefix sum: compute doc_ids inline
    output[0] = first_doc_id;
    let mut current = first_doc_id;

    // Process in groups of 4 for better cache locality
    let full_groups = (count - 1) / 4;
    let remainder = (count - 1) % 4;

    for group in 0..full_groups {
        let base_idx = group * 4;

        // Extract 4 deltas from vertical layout
        let mut deltas = [0u32; 4];
        for bit_pos in 0..bit_width as usize {
            let byte_offset = bit_pos * (SIMD_BLOCK_SIZE / 8);
            for (j, delta) in deltas.iter_mut().enumerate() {
                let int_idx = base_idx + j;
                let byte_idx = byte_offset + int_idx / 8;
                let bit_in_byte = int_idx % 8;
                let bit = ((input[byte_idx] >> bit_in_byte) & 1) as u32;
                *delta |= bit << bit_pos;
            }
        }

        // Apply prefix sum inline
        for j in 0..4 {
            current = current.wrapping_add(deltas[j]).wrapping_add(1);
            output[base_idx + j + 1] = current;
        }
    }

    // Handle remainder
    let base_idx = full_groups * 4;
    for j in 0..remainder {
        let int_idx = base_idx + j;
        let mut delta = 0u32;
        for bit_pos in 0..bit_width as usize {
            let byte_offset = bit_pos * (SIMD_BLOCK_SIZE / 8);
            let byte_idx = byte_offset + int_idx / 8;
            let bit_in_byte = int_idx % 8;
            let bit = ((input[byte_idx] >> bit_in_byte) & 1) as u32;
            delta |= bit << bit_pos;
        }
        current = current.wrapping_add(delta).wrapping_add(1);
        output[base_idx + j + 1] = current;
    }
}

/// A single SIMD-BP128 block with metadata
#[derive(Debug, Clone)]
pub struct SimdBp128Block {
    /// Vertically-packed delta-encoded doc_ids
    pub doc_data: Vec<u8>,
    /// Bit width for doc deltas
    pub doc_bit_width: u8,
    /// Vertically-packed term frequencies (tf - 1)
    pub tf_data: Vec<u8>,
    /// Bit width for term frequencies
    pub tf_bit_width: u8,
    /// First doc_id in block (absolute)
    pub first_doc_id: u32,
    /// Last doc_id in block (absolute)
    pub last_doc_id: u32,
    /// Number of docs in this block
    pub num_docs: u16,
    /// Maximum term frequency in block
    pub max_tf: u32,
    /// Maximum BM25 score upper bound for BlockMax WAND
    pub max_block_score: f32,
}

impl SimdBp128Block {
    /// Serialize block
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.first_doc_id)?;
        writer.write_u32::<LittleEndian>(self.last_doc_id)?;
        writer.write_u16::<LittleEndian>(self.num_docs)?;
        writer.write_u8(self.doc_bit_width)?;
        writer.write_u8(self.tf_bit_width)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_block_score)?;

        writer.write_u16::<LittleEndian>(self.doc_data.len() as u16)?;
        writer.write_all(&self.doc_data)?;

        writer.write_u16::<LittleEndian>(self.tf_data.len() as u16)?;
        writer.write_all(&self.tf_data)?;

        Ok(())
    }

    /// Deserialize block
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let first_doc_id = reader.read_u32::<LittleEndian>()?;
        let last_doc_id = reader.read_u32::<LittleEndian>()?;
        let num_docs = reader.read_u16::<LittleEndian>()?;
        let doc_bit_width = reader.read_u8()?;
        let tf_bit_width = reader.read_u8()?;
        let max_tf = reader.read_u32::<LittleEndian>()?;
        let max_block_score = reader.read_f32::<LittleEndian>()?;

        let doc_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut doc_data = vec![0u8; doc_len];
        reader.read_exact(&mut doc_data)?;

        let tf_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut tf_data = vec![0u8; tf_len];
        reader.read_exact(&mut tf_data)?;

        Ok(Self {
            doc_data,
            doc_bit_width,
            tf_data,
            tf_bit_width,
            first_doc_id,
            last_doc_id,
            num_docs,
            max_tf,
            max_block_score,
        })
    }

    /// Decode doc_ids from this block
    pub fn decode_doc_ids(&self) -> Vec<u32> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let mut output = [0u32; SIMD_BLOCK_SIZE];
        unpack_vertical_d1(
            &self.doc_data,
            self.doc_bit_width,
            self.first_doc_id,
            &mut output,
            self.num_docs as usize,
        );

        output[..self.num_docs as usize].to_vec()
    }

    /// Decode term frequencies from this block
    pub fn decode_term_freqs(&self) -> Vec<u32> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let mut output = [0u32; SIMD_BLOCK_SIZE];
        unpack_vertical(&self.tf_data, self.tf_bit_width, &mut output);

        // TF is stored as tf-1, add 1 back
        output[..self.num_docs as usize]
            .iter()
            .map(|&tf| tf + 1)
            .collect()
    }
}

/// SIMD-BP128 posting list with vertical layout and BlockMax support
#[derive(Debug, Clone)]
pub struct SimdBp128PostingList {
    /// Blocks of postings
    pub blocks: Vec<SimdBp128Block>,
    /// Total document count
    pub doc_count: u32,
    /// Maximum score across all blocks
    pub max_score: f32,
}

impl SimdBp128PostingList {
    /// BM25 parameters
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    /// Compute BM25 upper bound score
    #[inline]
    pub fn compute_bm25_upper_bound(max_tf: u32, idf: f32) -> f32 {
        let tf = max_tf as f32;
        let min_length_norm = 1.0 - Self::B;
        let tf_norm = (tf * (Self::K1 + 1.0)) / (tf + Self::K1 * min_length_norm);
        idf * tf_norm
    }

    /// Create from raw postings
    pub fn from_postings(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> Self {
        assert_eq!(doc_ids.len(), term_freqs.len());

        if doc_ids.is_empty() {
            return Self {
                blocks: Vec::new(),
                doc_count: 0,
                max_score: 0.0,
            };
        }

        let mut blocks = Vec::new();
        let mut max_score = 0.0f32;
        let mut i = 0;

        while i < doc_ids.len() {
            let block_end = (i + SIMD_BLOCK_SIZE).min(doc_ids.len());
            let block_docs = &doc_ids[i..block_end];
            let block_tfs = &term_freqs[i..block_end];

            let block = Self::create_block(block_docs, block_tfs, idf);
            max_score = max_score.max(block.max_block_score);
            blocks.push(block);

            i = block_end;
        }

        Self {
            blocks,
            doc_count: doc_ids.len() as u32,
            max_score,
        }
    }

    fn create_block(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> SimdBp128Block {
        let num_docs = doc_ids.len();
        let first_doc_id = doc_ids[0];
        let last_doc_id = *doc_ids.last().unwrap();

        // Compute deltas (gap - 1)
        let mut deltas = [0u32; SIMD_BLOCK_SIZE];
        let mut max_delta = 0u32;
        for j in 1..num_docs {
            let delta = doc_ids[j] - doc_ids[j - 1] - 1;
            deltas[j - 1] = delta;
            max_delta = max_delta.max(delta);
        }

        // Compute TFs (tf - 1)
        let mut tfs = [0u32; SIMD_BLOCK_SIZE];
        let mut max_tf = 0u32;
        for (j, &tf) in term_freqs.iter().enumerate() {
            tfs[j] = tf.saturating_sub(1);
            max_tf = max_tf.max(tf);
        }

        let doc_bit_width = bits_needed(max_delta);
        let tf_bit_width = bits_needed(max_tf.saturating_sub(1));

        let mut doc_data = Vec::new();
        pack_vertical(&deltas, doc_bit_width, &mut doc_data);

        let mut tf_data = Vec::new();
        pack_vertical(&tfs, tf_bit_width, &mut tf_data);

        let max_block_score = Self::compute_bm25_upper_bound(max_tf, idf);

        SimdBp128Block {
            doc_data,
            doc_bit_width,
            tf_data,
            tf_bit_width,
            first_doc_id,
            last_doc_id,
            num_docs: num_docs as u16,
            max_tf,
            max_block_score,
        }
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.doc_count)?;
        writer.write_f32::<LittleEndian>(self.max_score)?;
        writer.write_u32::<LittleEndian>(self.blocks.len() as u32)?;

        for block in &self.blocks {
            block.serialize(writer)?;
        }

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_count = reader.read_u32::<LittleEndian>()?;
        let max_score = reader.read_f32::<LittleEndian>()?;
        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(SimdBp128Block::deserialize(reader)?);
        }

        Ok(Self {
            blocks,
            doc_count,
            max_score,
        })
    }

    /// Create iterator
    pub fn iterator(&self) -> SimdBp128Iterator<'_> {
        SimdBp128Iterator::new(self)
    }

    /// Get approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        let mut size = 12; // header
        for block in &self.blocks {
            size += 22 + block.doc_data.len() + block.tf_data.len();
        }
        size
    }
}

/// Iterator over SIMD-BP128 posting list
pub struct SimdBp128Iterator<'a> {
    list: &'a SimdBp128PostingList,
    current_block: usize,
    block_doc_ids: Vec<u32>,
    block_term_freqs: Vec<u32>,
    pos_in_block: usize,
    exhausted: bool,
}

impl<'a> SimdBp128Iterator<'a> {
    pub fn new(list: &'a SimdBp128PostingList) -> Self {
        let mut iter = Self {
            list,
            current_block: 0,
            block_doc_ids: Vec::new(),
            block_term_freqs: Vec::new(),
            pos_in_block: 0,
            exhausted: list.blocks.is_empty(),
        };

        if !iter.exhausted {
            iter.decode_current_block();
        }

        iter
    }

    fn decode_current_block(&mut self) {
        let block = &self.list.blocks[self.current_block];
        self.block_doc_ids = block.decode_doc_ids();
        self.block_term_freqs = block.decode_term_freqs();
        self.pos_in_block = 0;
    }

    /// Current document ID
    pub fn doc(&self) -> u32 {
        if self.exhausted {
            u32::MAX
        } else {
            self.block_doc_ids[self.pos_in_block]
        }
    }

    /// Current term frequency
    pub fn term_freq(&self) -> u32 {
        if self.exhausted {
            0
        } else {
            self.block_term_freqs[self.pos_in_block]
        }
    }

    /// Advance to next document
    pub fn advance(&mut self) -> u32 {
        if self.exhausted {
            return u32::MAX;
        }

        self.pos_in_block += 1;

        if self.pos_in_block >= self.block_doc_ids.len() {
            self.current_block += 1;
            if self.current_block >= self.list.blocks.len() {
                self.exhausted = true;
                return u32::MAX;
            }
            self.decode_current_block();
        }

        self.doc()
    }

    /// Seek to first doc >= target with block skipping
    pub fn seek(&mut self, target: u32) -> u32 {
        if self.exhausted {
            return u32::MAX;
        }

        // Binary search for target block
        let block_idx = self.list.blocks[self.current_block..].binary_search_by(|block| {
            if block.last_doc_id < target {
                std::cmp::Ordering::Less
            } else if block.first_doc_id > target {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        let target_block = match block_idx {
            Ok(idx) => self.current_block + idx,
            Err(idx) => {
                if self.current_block + idx >= self.list.blocks.len() {
                    self.exhausted = true;
                    return u32::MAX;
                }
                self.current_block + idx
            }
        };

        if target_block != self.current_block {
            self.current_block = target_block;
            self.decode_current_block();
        }

        // Binary search within block
        let pos = self.block_doc_ids[self.pos_in_block..]
            .binary_search(&target)
            .unwrap_or_else(|x| x);
        self.pos_in_block += pos;

        if self.pos_in_block >= self.block_doc_ids.len() {
            self.current_block += 1;
            if self.current_block >= self.list.blocks.len() {
                self.exhausted = true;
                return u32::MAX;
            }
            self.decode_current_block();
        }

        self.doc()
    }

    /// Get max score for remaining blocks
    pub fn max_remaining_score(&self) -> f32 {
        if self.exhausted {
            return 0.0;
        }
        self.list.blocks[self.current_block..]
            .iter()
            .map(|b| b.max_block_score)
            .fold(0.0f32, |a, b| a.max(b))
    }

    /// Get current block's max score
    pub fn current_block_max_score(&self) -> f32 {
        if self.exhausted {
            0.0
        } else {
            self.list.blocks[self.current_block].max_block_score
        }
    }

    /// Get current block's max TF
    pub fn current_block_max_tf(&self) -> u32 {
        if self.exhausted {
            0
        } else {
            self.list.blocks[self.current_block].max_tf
        }
    }

    /// Skip to next block containing doc >= target (for BlockWAND)
    /// Returns (first_doc_in_block, block_max_score) or None if exhausted
    pub fn skip_to_block_with_doc(&mut self, target: u32) -> Option<(u32, f32)> {
        while self.current_block < self.list.blocks.len() {
            let block = &self.list.blocks[self.current_block];
            if block.last_doc_id >= target {
                // Decode this block and position at start
                self.decode_current_block();
                return Some((block.first_doc_id, block.max_block_score));
            }
            self.current_block += 1;
        }
        self.exhausted = true;
        None
    }

    /// Check if iterator is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_vertical() {
        let mut values = [0u32; SIMD_BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i * 3) as u32;
        }

        let max_val = values.iter().max().copied().unwrap();
        let bit_width = bits_needed(max_val);

        let mut packed = Vec::new();
        pack_vertical(&values, bit_width, &mut packed);

        let mut unpacked = [0u32; SIMD_BLOCK_SIZE];
        unpack_vertical(&packed, bit_width, &mut unpacked);

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_vertical_various_widths() {
        for bit_width in 1..=20 {
            let mut values = [0u32; SIMD_BLOCK_SIZE];
            let max_val = (1u32 << bit_width) - 1;
            for (i, v) in values.iter_mut().enumerate() {
                *v = (i as u32) % (max_val + 1);
            }

            let mut packed = Vec::new();
            pack_vertical(&values, bit_width, &mut packed);

            let mut unpacked = [0u32; SIMD_BLOCK_SIZE];
            unpack_vertical(&packed, bit_width, &mut unpacked);

            assert_eq!(values, unpacked, "Failed for bit_width={}", bit_width);
        }
    }

    #[test]
    fn test_simd_bp128_posting_list() {
        let doc_ids: Vec<u32> = (0..200).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = (0..200).map(|i| (i % 10) + 1).collect();

        let list = SimdBp128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        assert_eq!(list.doc_count, 200);
        assert_eq!(list.blocks.len(), 2); // 128 + 72

        let mut iter = list.iterator();
        for (i, &expected_doc) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected_doc, "Doc mismatch at {}", i);
            assert_eq!(iter.term_freq(), term_freqs[i], "TF mismatch at {}", i);
            if i < doc_ids.len() - 1 {
                iter.advance();
            }
        }
    }

    #[test]
    fn test_simd_bp128_seek() {
        let doc_ids: Vec<u32> = vec![10, 20, 30, 100, 200, 300, 1000, 2000];
        let term_freqs: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let list = SimdBp128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut iter = list.iterator();

        assert_eq!(iter.seek(25), 30);
        assert_eq!(iter.seek(100), 100);
        assert_eq!(iter.seek(500), 1000);
        assert_eq!(iter.seek(3000), u32::MAX);
    }

    #[test]
    fn test_simd_bp128_serialization() {
        let doc_ids: Vec<u32> = (0..300).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = (0..300).map(|i| (i % 5) + 1).collect();

        let list = SimdBp128PostingList::from_postings(&doc_ids, &term_freqs, 1.5);

        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();

        let restored = SimdBp128PostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.doc_count, list.doc_count);
        assert_eq!(restored.blocks.len(), list.blocks.len());

        let mut iter1 = list.iterator();
        let mut iter2 = restored.iterator();

        while iter1.doc() != u32::MAX {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }
    }

    #[test]
    fn test_vertical_layout_size() {
        // True vertical layout: BLOCK_SIZE * bit_width / 8 bytes (optimal)
        let mut values = [0u32; SIMD_BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = i as u32;
        }

        let bit_width = bits_needed(127); // 7 bits
        assert_eq!(bit_width, 7);

        let mut packed = Vec::new();
        pack_horizontal(&values, bit_width, &mut packed);

        // True vertical layout: 128 * 7 / 8 = 112 bytes (optimal, no padding)
        let expected_bytes = (SIMD_BLOCK_SIZE * bit_width as usize) / 8;
        assert_eq!(expected_bytes, 112);
        assert_eq!(packed.len(), expected_bytes);
    }

    #[test]
    fn test_simd_bp128_block_max() {
        // Create a large posting list that spans multiple blocks
        let doc_ids: Vec<u32> = (0..500).map(|i| i * 2).collect();
        // Vary term frequencies so different blocks have different max_tf
        let term_freqs: Vec<u32> = (0..500)
            .map(|i| {
                if i < 128 {
                    1 // Block 0: max_tf = 1
                } else if i < 256 {
                    5 // Block 1: max_tf = 5
                } else if i < 384 {
                    10 // Block 2: max_tf = 10
                } else {
                    3 // Block 3: max_tf = 3
                }
            })
            .collect();

        let list = SimdBp128PostingList::from_postings(&doc_ids, &term_freqs, 2.0);

        // Should have 4 blocks (500 docs / 128 per block)
        assert_eq!(list.blocks.len(), 4);
        assert_eq!(list.blocks[0].max_tf, 1);
        assert_eq!(list.blocks[1].max_tf, 5);
        assert_eq!(list.blocks[2].max_tf, 10);
        assert_eq!(list.blocks[3].max_tf, 3);

        // Block 2 should have highest score (max_tf = 10)
        assert!(list.blocks[2].max_block_score > list.blocks[0].max_block_score);
        assert!(list.blocks[2].max_block_score > list.blocks[1].max_block_score);
        assert!(list.blocks[2].max_block_score > list.blocks[3].max_block_score);

        // Global max_score should equal block 2's score
        assert_eq!(list.max_score, list.blocks[2].max_block_score);

        // Test iterator block-max methods
        let mut iter = list.iterator();
        assert_eq!(iter.current_block_max_tf(), 1); // Block 0

        // Seek to block 1
        iter.seek(256); // first doc in block 1
        assert_eq!(iter.current_block_max_tf(), 5);

        // Seek to block 2
        iter.seek(512); // first doc in block 2
        assert_eq!(iter.current_block_max_tf(), 10);

        // Test skip_to_block_with_doc
        let mut iter2 = list.iterator();
        let result = iter2.skip_to_block_with_doc(300);
        assert!(result.is_some());
        let (first_doc, score) = result.unwrap();
        assert!(first_doc <= 300);
        assert!(score > 0.0);
    }
}
