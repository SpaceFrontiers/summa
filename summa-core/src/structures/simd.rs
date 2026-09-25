//! Shared SIMD-accelerated functions for posting list compression
//!
//! This module provides platform-optimized implementations for common operations:
//! - **Unpacking**: Convert packed 8/16/32-bit values to u32 arrays
//! - **Delta decoding**: Prefix sum for converting deltas to absolute values
//! - **Add one**: Increment all values in an array (for TF decoding)
//! - **Float reductions**: dot product, squared L2, squared norm
//!
//! Supports:
//! - **NEON** on aarch64 (Apple Silicon, ARM servers)
//! - **SSE/SSE4.1** on x86_64 (Intel/AMD)
//! - **Scalar fallback** for other architectures (algebraic float ops, see below)

// ============================================================================
// NEON intrinsics for aarch64 (Apple Silicon, ARM servers)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod neon {
    use std::arch::aarch64::*;

    /// SIMD unpack for 8-bit values using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_8bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 16;
        let remainder = count % 16;

        for chunk in 0..chunks {
            let base = chunk * 16;
            let in_ptr = input.as_ptr().add(base);

            // Load 16 bytes
            let bytes = vld1q_u8(in_ptr);

            // Widen u8 -> u16 -> u32
            let low8 = vget_low_u8(bytes);
            let high8 = vget_high_u8(bytes);

            let low16 = vmovl_u8(low8);
            let high16 = vmovl_u8(high8);

            let v0 = vmovl_u16(vget_low_u16(low16));
            let v1 = vmovl_u16(vget_high_u16(low16));
            let v2 = vmovl_u16(vget_low_u16(high16));
            let v3 = vmovl_u16(vget_high_u16(high16));

            let out_ptr = output.as_mut_ptr().add(base);
            vst1q_u32(out_ptr, v0);
            vst1q_u32(out_ptr.add(4), v1);
            vst1q_u32(out_ptr.add(8), v2);
            vst1q_u32(out_ptr.add(12), v3);
        }

        // Handle remainder
        let base = chunks * 16;
        for i in 0..remainder {
            output[base + i] = input[base + i] as u32;
        }
    }

    /// SIMD unpack for 16-bit values using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_16bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 8;
        let remainder = count % 8;

        for chunk in 0..chunks {
            let base = chunk * 8;
            let in_ptr = input.as_ptr().add(base * 2) as *const u16;

            let vals = vld1q_u16(in_ptr);
            let low = vmovl_u16(vget_low_u16(vals));
            let high = vmovl_u16(vget_high_u16(vals));

            let out_ptr = output.as_mut_ptr().add(base);
            vst1q_u32(out_ptr, low);
            vst1q_u32(out_ptr.add(4), high);
        }

        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            let idx = (base + i) * 2;
            output[base + i] = u16::from_le_bytes([input[idx], input[idx + 1]]) as u32;
        }
    }

    /// SIMD unpack for 32-bit values using NEON (fast copy)
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_32bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 4;
        let remainder = count % 4;

        let in_ptr = input.as_ptr() as *const u32;
        let out_ptr = output.as_mut_ptr();

        for chunk in 0..chunks {
            let vals = vld1q_u32(in_ptr.add(chunk * 4));
            vst1q_u32(out_ptr.add(chunk * 4), vals);
        }

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            let idx = (base + i) * 4;
            output[base + i] =
                u32::from_le_bytes([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
        }
    }

    /// SIMD prefix sum for 4 u32 values using NEON
    /// Input:  [a, b, c, d]
    /// Output: [a, a+b, a+b+c, a+b+c+d]
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn prefix_sum_4(v: uint32x4_t) -> uint32x4_t {
        // Step 1: shift by 1 and add
        // [a, b, c, d] + [0, a, b, c] = [a, a+b, b+c, c+d]
        let shifted1 = vextq_u32(vdupq_n_u32(0), v, 3);
        let sum1 = vaddq_u32(v, shifted1);

        // Step 2: shift by 2 and add
        // [a, a+b, b+c, c+d] + [0, 0, a, a+b] = [a, a+b, a+b+c, a+b+c+d]
        let shifted2 = vextq_u32(vdupq_n_u32(0), sum1, 2);
        vaddq_u32(sum1, shifted2)
    }

    /// SIMD delta decode: convert deltas to absolute doc IDs
    /// deltas[i] stores (gap - 1), output[i] = first + sum(gaps[0..i])
    /// Uses NEON SIMD prefix sum for high throughput
    #[target_feature(enable = "neon")]
    pub unsafe fn delta_decode(
        output: &mut [u32],
        deltas: &[u32],
        first_doc_id: u32,
        count: usize,
    ) {
        if count == 0 {
            return;
        }

        output[0] = first_doc_id;
        if count == 1 {
            return;
        }

        let ones = vdupq_n_u32(1);
        let mut carry = vdupq_n_u32(first_doc_id);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;

            // Load 4 deltas and add 1 (since we store gap-1)
            let d = vld1q_u32(deltas[base..].as_ptr());
            let gaps = vaddq_u32(d, ones);

            // Compute prefix sum within the 4 elements
            let prefix = prefix_sum_4(gaps);

            // Add carry (broadcast last element of previous group)
            let result = vaddq_u32(prefix, carry);

            // Store result
            vst1q_u32(output[base + 1..].as_mut_ptr(), result);

            // Update carry: broadcast the last element for next iteration
            carry = vdupq_n_u32(vgetq_lane_u32(result, 3));
        }

        // Handle remainder
        let base = full_groups * 4;
        let mut scalar_carry = vgetq_lane_u32(carry, 0);
        for j in 0..remainder {
            scalar_carry = scalar_carry.wrapping_add(deltas[base + j]).wrapping_add(1);
            output[base + j + 1] = scalar_carry;
        }
    }

    /// SIMD add 1 to all values (for TF decoding: stored as tf-1)
    #[target_feature(enable = "neon")]
    pub unsafe fn add_one(values: &mut [u32], count: usize) {
        let ones = vdupq_n_u32(1);
        let chunks = count / 4;
        let remainder = count % 4;

        for chunk in 0..chunks {
            let base = chunk * 4;
            let ptr = values.as_mut_ptr().add(base);
            let v = vld1q_u32(ptr);
            let result = vaddq_u32(v, ones);
            vst1q_u32(ptr, result);
        }

        let base = chunks * 4;
        for i in 0..remainder {
            values[base + i] += 1;
        }
    }

    /// Fused unpack 8-bit + delta decode using NEON. Processes 4 values at a
    /// time, fusing unpack and prefix sum; `OFFSET` is added to every gap.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `input.len() >= count - 1` (one byte per
    /// delta) and `output.len() >= count`; the vector loads/stores are
    /// unchecked. The safe dispatchers in this module assert this once per
    /// block. `count == 0` is not allowed (`output[0]` is written).
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_8bit_delta_decode_with_offset<const OFFSET: u32>(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = vdupq_n_u32(OFFSET);
        let mut carry = vdupq_n_u32(first_value);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;

            // Load 4 bytes as a u32, then widen u8→u16→u32 via NEON
            let raw = std::ptr::read_unaligned(input.as_ptr().add(base) as *const u32);
            let bytes = vreinterpret_u8_u32(vdup_n_u32(raw));
            let u16s = vmovl_u8(bytes); // 8×u8 → 8×u16 (only low 4 matter)
            let d = vmovl_u16(vget_low_u16(u16s)); // 4×u16 → 4×u32

            // Add the format's gap offset
            let gaps = vaddq_u32(d, ones);

            // Compute prefix sum within the 4 elements
            let prefix = prefix_sum_4(gaps);

            // Add carry
            let result = vaddq_u32(prefix, carry);

            // Store result
            vst1q_u32(output[base + 1..].as_mut_ptr(), result);

            // Update carry
            carry = vdupq_n_u32(vgetq_lane_u32(result, 3));
        }

        // Handle remainder: re-decode from output[base] (== carry) onward.
        let base = full_groups * 4;
        super::scalar::delta_decode_with_offset::<OFFSET, 1>(
            &input[base..],
            &mut output[base..],
            vgetq_lane_u32(carry, 0),
            remainder + 1,
        );
    }

    /// Fused unpack 16-bit + delta decode using NEON; `OFFSET` is added to
    /// every gap.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `input.len() >= (count - 1) * 2` (two bytes
    /// per delta) and `output.len() >= count`; the vector loads/stores are
    /// unchecked. The safe dispatchers in this module assert this once per
    /// block. `count == 0` is not allowed (`output[0]` is written).
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_16bit_delta_decode_with_offset<const OFFSET: u32>(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = vdupq_n_u32(OFFSET);
        let mut carry = vdupq_n_u32(first_value);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;
            let in_ptr = input.as_ptr().add(base * 2) as *const u16;

            // Load 4 u16 values and widen to u32
            let vals = vld1_u16(in_ptr);
            let d = vmovl_u16(vals);

            // Add the format's gap offset
            let gaps = vaddq_u32(d, ones);

            // Compute prefix sum within the 4 elements
            let prefix = prefix_sum_4(gaps);

            // Add carry
            let result = vaddq_u32(prefix, carry);

            // Store result
            vst1q_u32(output[base + 1..].as_mut_ptr(), result);

            // Update carry
            carry = vdupq_n_u32(vgetq_lane_u32(result, 3));
        }

        // Handle remainder: re-decode from output[base] (== carry) onward.
        let base = full_groups * 4;
        super::scalar::delta_decode_with_offset::<OFFSET, 2>(
            &input[base * 2..],
            &mut output[base..],
            vgetq_lane_u32(carry, 0),
            remainder + 1,
        );
    }

    /// NEON Hamming distance: XOR + byte popcount + horizontal sum.
    /// Processes 16 bytes per iteration (vs 8 for scalar u64 path).
    #[target_feature(enable = "neon")]
    pub unsafe fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        let len = a.len();
        let chunks16 = len / 16;
        let mut total = 0u32;

        // Process 16 bytes at a time, flush u8 accumulators every 31 iters
        // (vcntq_u8 returns 0-8 per lane; 31 * 8 = 248 ≤ 255, avoiding u8 overflow)
        let mut i = 0;
        while i < chunks16 {
            let batch_end = (i + 31).min(chunks16);
            let mut acc = vdupq_n_u8(0);
            for j in i..batch_end {
                let off = j * 16;
                let va = vld1q_u8(a.as_ptr().add(off));
                let vb = vld1q_u8(b.as_ptr().add(off));
                let popcnt = vcntq_u8(veorq_u8(va, vb));
                acc = vaddq_u8(acc, popcnt);
            }
            // Widen u8 -> u16 -> u32 -> u64 and horizontal sum
            let sum64 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc)));
            total += vgetq_lane_u64(sum64, 0) as u32 + vgetq_lane_u64(sum64, 1) as u32;
            i = batch_end;
        }

        // Remainder bytes (< 16)
        let base = chunks16 * 16;
        for k in base..len {
            total += (a[k] ^ b[k]).count_ones();
        }

        total
    }

    /// Four-row NEON Hamming distance.
    ///
    /// The query chunk load and the horizontal reduction are shared across the
    /// rows, and the four accumulator chains overlap instead of serialising on
    /// `vcntq_u8`/`vaddq_u8` latency.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn hamming_distance_x4(query: &[u8], rows: [&[u8]; 4]) -> [u32; 4] {
        let len = query.len();
        let chunks16 = len / 16;
        let mut total = [0u32; 4];

        let mut i = 0;
        while i < chunks16 {
            let batch_end = (i + 31).min(chunks16);
            let mut acc = [vdupq_n_u8(0); 4];
            for j in i..batch_end {
                let off = j * 16;
                let vq = vld1q_u8(query.as_ptr().add(off));
                for r in 0..4 {
                    let vr = vld1q_u8(rows[r].as_ptr().add(off));
                    acc[r] = vaddq_u8(acc[r], vcntq_u8(veorq_u8(vq, vr)));
                }
            }
            for r in 0..4 {
                let sum64 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc[r])));
                total[r] += vgetq_lane_u64(sum64, 0) as u32 + vgetq_lane_u64(sum64, 1) as u32;
            }
            i = batch_end;
        }

        // Remainder through the u64 scalar path: a per-byte tail would dominate
        // for code widths narrower than one vector (e.g. 64-bit fields).
        let base = chunks16 * 16;
        if base < len {
            let tail = &query[base..];
            for r in 0..4 {
                total[r] += super::hamming_distance_scalar(tail, &rows[r][base..]);
            }
        }

        total
    }

    /// Check if NEON is available (always true on aarch64)
    #[inline]
    pub fn is_available() -> bool {
        true
    }
}

// ============================================================================
// SSE intrinsics for x86_64 (Intel/AMD)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod sse {
    use std::arch::x86_64::*;

    /// SIMD unpack for 8-bit values using SSE
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    pub unsafe fn unpack_8bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 16;
        let remainder = count % 16;

        for chunk in 0..chunks {
            let base = chunk * 16;
            let in_ptr = input.as_ptr().add(base);

            let bytes = _mm_loadu_si128(in_ptr as *const __m128i);

            // Zero extend u8 -> u32 using SSE4.1 pmovzx
            let v0 = _mm_cvtepu8_epi32(bytes);
            let v1 = _mm_cvtepu8_epi32(_mm_srli_si128(bytes, 4));
            let v2 = _mm_cvtepu8_epi32(_mm_srli_si128(bytes, 8));
            let v3 = _mm_cvtepu8_epi32(_mm_srli_si128(bytes, 12));

            let out_ptr = output.as_mut_ptr().add(base);
            _mm_storeu_si128(out_ptr as *mut __m128i, v0);
            _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, v1);
            _mm_storeu_si128(out_ptr.add(8) as *mut __m128i, v2);
            _mm_storeu_si128(out_ptr.add(12) as *mut __m128i, v3);
        }

        let base = chunks * 16;
        for i in 0..remainder {
            output[base + i] = input[base + i] as u32;
        }
    }

    /// SIMD unpack for 16-bit values using SSE
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    pub unsafe fn unpack_16bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 8;
        let remainder = count % 8;

        for chunk in 0..chunks {
            let base = chunk * 8;
            let in_ptr = input.as_ptr().add(base * 2);

            let vals = _mm_loadu_si128(in_ptr as *const __m128i);
            let low = _mm_cvtepu16_epi32(vals);
            let high = _mm_cvtepu16_epi32(_mm_srli_si128(vals, 8));

            let out_ptr = output.as_mut_ptr().add(base);
            _mm_storeu_si128(out_ptr as *mut __m128i, low);
            _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, high);
        }

        let base = chunks * 8;
        for i in 0..remainder {
            let idx = (base + i) * 2;
            output[base + i] = u16::from_le_bytes([input[idx], input[idx + 1]]) as u32;
        }
    }

    /// SIMD unpack for 32-bit values using SSE (fast copy)
    #[target_feature(enable = "sse2")]
    pub unsafe fn unpack_32bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 4;
        let remainder = count % 4;

        let in_ptr = input.as_ptr() as *const __m128i;
        let out_ptr = output.as_mut_ptr() as *mut __m128i;

        for chunk in 0..chunks {
            let vals = _mm_loadu_si128(in_ptr.add(chunk));
            _mm_storeu_si128(out_ptr.add(chunk), vals);
        }

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            let idx = (base + i) * 4;
            output[base + i] =
                u32::from_le_bytes([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
        }
    }

    /// SIMD prefix sum for 4 u32 values using SSE
    /// Input:  [a, b, c, d]
    /// Output: [a, a+b, a+b+c, a+b+c+d]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn prefix_sum_4(v: __m128i) -> __m128i {
        // Step 1: shift by 1 element (4 bytes) and add
        // [a, b, c, d] + [0, a, b, c] = [a, a+b, b+c, c+d]
        let shifted1 = _mm_slli_si128(v, 4);
        let sum1 = _mm_add_epi32(v, shifted1);

        // Step 2: shift by 2 elements (8 bytes) and add
        // [a, a+b, b+c, c+d] + [0, 0, a, a+b] = [a, a+b, a+b+c, a+b+c+d]
        let shifted2 = _mm_slli_si128(sum1, 8);
        _mm_add_epi32(sum1, shifted2)
    }

    /// SIMD delta decode using SSE with true SIMD prefix sum
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    pub unsafe fn delta_decode(
        output: &mut [u32],
        deltas: &[u32],
        first_doc_id: u32,
        count: usize,
    ) {
        if count == 0 {
            return;
        }

        output[0] = first_doc_id;
        if count == 1 {
            return;
        }

        let ones = _mm_set1_epi32(1);
        let mut carry = _mm_set1_epi32(first_doc_id as i32);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;

            // Load 4 deltas and add 1 (since we store gap-1)
            let d = _mm_loadu_si128(deltas[base..].as_ptr() as *const __m128i);
            let gaps = _mm_add_epi32(d, ones);

            // Compute prefix sum within the 4 elements
            let prefix = prefix_sum_4(gaps);

            // Add carry (broadcast last element of previous group)
            let result = _mm_add_epi32(prefix, carry);

            // Store result
            _mm_storeu_si128(output[base + 1..].as_mut_ptr() as *mut __m128i, result);

            // Update carry: broadcast the last element for next iteration
            carry = _mm_shuffle_epi32(result, 0xFF); // broadcast lane 3
        }

        // Handle remainder
        let base = full_groups * 4;
        let mut scalar_carry = _mm_extract_epi32(carry, 0) as u32;
        for j in 0..remainder {
            scalar_carry = scalar_carry.wrapping_add(deltas[base + j]).wrapping_add(1);
            output[base + j + 1] = scalar_carry;
        }
    }

    /// SIMD add 1 to all values using SSE
    #[target_feature(enable = "sse2")]
    pub unsafe fn add_one(values: &mut [u32], count: usize) {
        let ones = _mm_set1_epi32(1);
        let chunks = count / 4;
        let remainder = count % 4;

        for chunk in 0..chunks {
            let base = chunk * 4;
            let ptr = values.as_mut_ptr().add(base) as *mut __m128i;
            let v = _mm_loadu_si128(ptr);
            let result = _mm_add_epi32(v, ones);
            _mm_storeu_si128(ptr, result);
        }

        let base = chunks * 4;
        for i in 0..remainder {
            values[base + i] += 1;
        }
    }

    /// Fused unpack 8-bit + delta decode using SSE4.1; `OFFSET` is added to
    /// every gap.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `input.len() >= count - 1` (one byte per
    /// delta) and `output.len() >= count`; the vector loads/stores are
    /// unchecked. The safe dispatchers in this module assert this once per
    /// block. `count == 0` is not allowed (`output[0]` is written).
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn unpack_8bit_delta_decode_with_offset<const OFFSET: u32>(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = _mm_set1_epi32(OFFSET as i32);
        let mut carry = _mm_set1_epi32(first_value as i32);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;

            // Load 4 bytes (unaligned) and zero-extend to u32
            let bytes = _mm_cvtsi32_si128(std::ptr::read_unaligned(
                input.as_ptr().add(base) as *const i32
            ));
            let d = _mm_cvtepu8_epi32(bytes);

            // Add the format's gap offset
            let gaps = _mm_add_epi32(d, ones);

            // Compute prefix sum within the 4 elements
            let prefix = prefix_sum_4(gaps);

            // Add carry
            let result = _mm_add_epi32(prefix, carry);

            // Store result
            _mm_storeu_si128(output[base + 1..].as_mut_ptr() as *mut __m128i, result);

            // Update carry: broadcast the last element
            carry = _mm_shuffle_epi32(result, 0xFF);
        }

        // Handle remainder: re-decode from output[base] (== carry) onward.
        let base = full_groups * 4;
        super::scalar::delta_decode_with_offset::<OFFSET, 1>(
            &input[base..],
            &mut output[base..],
            _mm_extract_epi32(carry, 0) as u32,
            remainder + 1,
        );
    }

    /// Fused unpack 16-bit + delta decode using SSE4.1; `OFFSET` is added to
    /// every gap.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `input.len() >= (count - 1) * 2` (two bytes
    /// per delta) and `output.len() >= count`; the vector loads/stores are
    /// unchecked. The safe dispatchers in this module assert this once per
    /// block. `count == 0` is not allowed (`output[0]` is written).
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn unpack_16bit_delta_decode_with_offset<const OFFSET: u32>(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = _mm_set1_epi32(OFFSET as i32);
        let mut carry = _mm_set1_epi32(first_value as i32);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;
            let in_ptr = input.as_ptr().add(base * 2);

            // Load 8 bytes (4 u16 values, unaligned) and zero-extend to u32
            let vals = _mm_loadl_epi64(in_ptr as *const __m128i); // loadl_epi64 supports unaligned
            let d = _mm_cvtepu16_epi32(vals);

            // Add the format's gap offset
            let gaps = _mm_add_epi32(d, ones);

            // Compute prefix sum within the 4 elements
            let prefix = prefix_sum_4(gaps);

            // Add carry
            let result = _mm_add_epi32(prefix, carry);

            // Store result
            _mm_storeu_si128(output[base + 1..].as_mut_ptr() as *mut __m128i, result);

            // Update carry: broadcast the last element
            carry = _mm_shuffle_epi32(result, 0xFF);
        }

        // Handle remainder: re-decode from output[base] (== carry) onward.
        let base = full_groups * 4;
        super::scalar::delta_decode_with_offset::<OFFSET, 2>(
            &input[base * 2..],
            &mut output[base..],
            _mm_extract_epi32(carry, 0) as u32,
            remainder + 1,
        );
    }

    /// Check if SSE4.1 is available at runtime
    #[inline]
    pub fn is_available() -> bool {
        is_x86_feature_detected!("sse4.1")
    }
}

// ============================================================================
// AVX2 intrinsics for x86_64 (Intel/AMD with 256-bit registers)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx2 {
    use std::arch::x86_64::*;

    /// AVX2 unpack for 8-bit values (processes 32 bytes at a time)
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_8bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 32;
        let remainder = count % 32;

        for chunk in 0..chunks {
            let base = chunk * 32;
            let in_ptr = input.as_ptr().add(base);

            // Load 32 bytes (two 128-bit loads, then combine)
            let bytes_lo = _mm_loadu_si128(in_ptr as *const __m128i);
            let bytes_hi = _mm_loadu_si128(in_ptr.add(16) as *const __m128i);

            // Zero extend first 16 bytes: u8 -> u32
            let v0 = _mm256_cvtepu8_epi32(bytes_lo);
            let v1 = _mm256_cvtepu8_epi32(_mm_srli_si128(bytes_lo, 8));
            let v2 = _mm256_cvtepu8_epi32(bytes_hi);
            let v3 = _mm256_cvtepu8_epi32(_mm_srli_si128(bytes_hi, 8));

            let out_ptr = output.as_mut_ptr().add(base);
            _mm256_storeu_si256(out_ptr as *mut __m256i, v0);
            _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, v1);
            _mm256_storeu_si256(out_ptr.add(16) as *mut __m256i, v2);
            _mm256_storeu_si256(out_ptr.add(24) as *mut __m256i, v3);
        }

        // Handle remainder with SSE
        let base = chunks * 32;
        for i in 0..remainder {
            output[base + i] = input[base + i] as u32;
        }
    }

    /// AVX2 unpack for 16-bit values (processes 16 values at a time)
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_16bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 16;
        let remainder = count % 16;

        for chunk in 0..chunks {
            let base = chunk * 16;
            let in_ptr = input.as_ptr().add(base * 2);

            // Load 32 bytes (16 u16 values)
            let vals_lo = _mm_loadu_si128(in_ptr as *const __m128i);
            let vals_hi = _mm_loadu_si128(in_ptr.add(16) as *const __m128i);

            // Zero extend u16 -> u32
            let v0 = _mm256_cvtepu16_epi32(vals_lo);
            let v1 = _mm256_cvtepu16_epi32(vals_hi);

            let out_ptr = output.as_mut_ptr().add(base);
            _mm256_storeu_si256(out_ptr as *mut __m256i, v0);
            _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, v1);
        }

        // Handle remainder
        let base = chunks * 16;
        for i in 0..remainder {
            let idx = (base + i) * 2;
            output[base + i] = u16::from_le_bytes([input[idx], input[idx + 1]]) as u32;
        }
    }

    /// AVX2 unpack for 32-bit values (fast copy, 8 values at a time)
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_32bit(input: &[u8], output: &mut [u32], count: usize) {
        let chunks = count / 8;
        let remainder = count % 8;

        let in_ptr = input.as_ptr() as *const __m256i;
        let out_ptr = output.as_mut_ptr() as *mut __m256i;

        for chunk in 0..chunks {
            let vals = _mm256_loadu_si256(in_ptr.add(chunk));
            _mm256_storeu_si256(out_ptr.add(chunk), vals);
        }

        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            let idx = (base + i) * 4;
            output[base + i] =
                u32::from_le_bytes([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
        }
    }

    /// AVX2 add 1 to all values (8 values at a time)
    #[target_feature(enable = "avx2")]
    pub unsafe fn add_one(values: &mut [u32], count: usize) {
        let ones = _mm256_set1_epi32(1);
        let chunks = count / 8;
        let remainder = count % 8;

        for chunk in 0..chunks {
            let base = chunk * 8;
            let ptr = values.as_mut_ptr().add(base) as *mut __m256i;
            let v = _mm256_loadu_si256(ptr);
            let result = _mm256_add_epi32(v, ones);
            _mm256_storeu_si256(ptr, result);
        }

        let base = chunks * 8;
        for i in 0..remainder {
            values[base + i] += 1;
        }
    }

    /// AVX2 prefix sum for 8 u32 values (Hillis-Steele)
    /// Input:  [a, b, c, d, e, f, g, h]
    /// Output: [a, a+b, a+b+c, ..., a+b+c+d+e+f+g+h]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn prefix_sum_8(v: __m256i) -> __m256i {
        // Step 1: intra-lane shift by 1 element (4 bytes) and add
        let s1 = _mm256_slli_si256(v, 4);
        let r1 = _mm256_add_epi32(v, s1);

        // Step 2: intra-lane shift by 2 elements (8 bytes) and add
        let s2 = _mm256_slli_si256(r1, 8);
        let r2 = _mm256_add_epi32(r1, s2);

        // Step 3: propagate lower lane sum to upper lane
        // Broadcast element 3 (lower lane sum) within each lane
        let lo_sum = _mm256_shuffle_epi32(r2, 0xFF);
        // Duplicate lane 0 to both lanes
        let carry = _mm256_permute2x128_si256(lo_sum, lo_sum, 0x00);
        // Zero carry for lower lane, keep for upper
        let carry_hi = _mm256_blend_epi32::<0xF0>(_mm256_setzero_si256(), carry);
        _mm256_add_epi32(r2, carry_hi)
    }

    /// AVX2 fused unpack 8-bit + delta decode (processes 8 values at a time);
    /// `OFFSET` is added to every gap.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `input.len() >= count - 1` (one byte per
    /// delta) and `output.len() >= count`; the vector loads/stores are
    /// unchecked. The safe dispatchers in this module assert this once per
    /// block. `count == 0` is not allowed (`output[0]` is written).
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_8bit_delta_decode_with_offset<const OFFSET: u32>(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = _mm256_set1_epi32(OFFSET as i32);
        let mut carry = _mm256_set1_epi32(first_value as i32);
        let broadcast_idx = _mm256_set1_epi32(7);

        let full_groups = (count - 1) / 8;
        let remainder = (count - 1) % 8;

        for group in 0..full_groups {
            let base = group * 8;

            // Load 8 bytes and zero-extend to 8×u32
            let bytes = _mm_loadl_epi64(input.as_ptr().add(base) as *const __m128i);
            let d = _mm256_cvtepu8_epi32(bytes);

            // Add the format's gap offset
            let gaps = _mm256_add_epi32(d, ones);

            // Compute prefix sum within 8 elements
            let prefix = prefix_sum_8(gaps);

            // Add carry from previous group
            let result = _mm256_add_epi32(prefix, carry);

            // Store 8 results
            _mm256_storeu_si256(output[base + 1..].as_mut_ptr() as *mut __m256i, result);

            // Update carry: broadcast element 7 to all positions
            carry = _mm256_permutevar8x32_epi32(result, broadcast_idx);
        }

        // Handle remainder: re-decode from output[base] (== carry) onward.
        let base = full_groups * 8;
        super::scalar::delta_decode_with_offset::<OFFSET, 1>(
            &input[base..],
            &mut output[base..],
            _mm256_extract_epi32::<0>(carry) as u32,
            remainder + 1,
        );
    }

    /// AVX2 fused unpack 16-bit + delta decode (processes 8 values at a time);
    /// `OFFSET` is added to every gap.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `input.len() >= (count - 1) * 2` (two bytes
    /// per delta) and `output.len() >= count`; the vector loads/stores are
    /// unchecked. The safe dispatchers in this module assert this once per
    /// block. `count == 0` is not allowed (`output[0]` is written).
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_16bit_delta_decode_with_offset<const OFFSET: u32>(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = _mm256_set1_epi32(OFFSET as i32);
        let mut carry = _mm256_set1_epi32(first_value as i32);
        let broadcast_idx = _mm256_set1_epi32(7);

        let full_groups = (count - 1) / 8;
        let remainder = (count - 1) % 8;

        for group in 0..full_groups {
            let base = group * 8;
            let in_ptr = input.as_ptr().add(base * 2);

            // Load 16 bytes (8 u16 values) and zero-extend to 8×u32
            let vals = _mm_loadu_si128(in_ptr as *const __m128i);
            let d = _mm256_cvtepu16_epi32(vals);

            // Add the format's gap offset
            let gaps = _mm256_add_epi32(d, ones);

            // Compute prefix sum within 8 elements
            let prefix = prefix_sum_8(gaps);

            // Add carry from previous group
            let result = _mm256_add_epi32(prefix, carry);

            // Store 8 results
            _mm256_storeu_si256(output[base + 1..].as_mut_ptr() as *mut __m256i, result);

            // Update carry: broadcast element 7 to all positions
            carry = _mm256_permutevar8x32_epi32(result, broadcast_idx);
        }

        // Handle remainder: re-decode from output[base] (== carry) onward.
        let base = full_groups * 8;
        super::scalar::delta_decode_with_offset::<OFFSET, 2>(
            &input[base * 2..],
            &mut output[base..],
            _mm256_extract_epi32::<0>(carry) as u32,
            remainder + 1,
        );
    }

    /// AVX2 Hamming distance using VPSHUFB-based popcount (Muła algorithm).
    /// Processes 32 bytes per iteration with a nibble lookup table.
    #[target_feature(enable = "avx2")]
    pub unsafe fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        let len = a.len();
        let chunks32 = len / 32;
        let low_mask = _mm256_set1_epi8(0x0f);
        // Nibble popcount lookup table: popcount(0..15)
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let mut total = 0u64;

        let mut i = 0;
        while i < chunks32 {
            // Accumulate in u8 lanes, flush every 31 iters to avoid overflow
            // (nibble popcount gives 0-8 per lane; 31 * 8 = 248 ≤ 255)
            let batch_end = (i + 31).min(chunks32);
            let mut acc = _mm256_setzero_si256();
            for j in i..batch_end {
                let off = j * 32;
                let va = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
                let vb = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);
                let xored = _mm256_xor_si256(va, vb);
                // VPSHUFB popcount: count bits per byte via nibble lookup
                let lo = _mm256_and_si256(xored, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(xored, 4), low_mask);
                let popcnt = _mm256_add_epi8(
                    _mm256_shuffle_epi8(lookup, lo),
                    _mm256_shuffle_epi8(lookup, hi),
                );
                acc = _mm256_add_epi8(acc, popcnt);
            }
            // Horizontal sum: u8 -> u64 via SAD against zero
            let sad = _mm256_sad_epu8(acc, _mm256_setzero_si256());
            total += _mm256_extract_epi64(sad, 0) as u64
                + _mm256_extract_epi64(sad, 1) as u64
                + _mm256_extract_epi64(sad, 2) as u64
                + _mm256_extract_epi64(sad, 3) as u64;
            i = batch_end;
        }

        // Remainder bytes (< 32)
        let base = chunks32 * 32;
        for k in base..len {
            total += (a[k] ^ b[k]).count_ones() as u64;
        }

        total as u32
    }

    /// Four-row AVX2 Hamming distance.
    ///
    /// The query chunk load, the nibble lookup table and the horizontal
    /// reduction are shared across the rows, and the four accumulator chains
    /// overlap instead of serialising on popcount latency.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn hamming_distance_x4(query: &[u8], rows: [&[u8]; 4]) -> [u32; 4] {
        let len = query.len();
        let chunks32 = len / 32;
        let low_mask = _mm256_set1_epi8(0x0f);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let mut total = [0u64; 4];

        let mut i = 0;
        while i < chunks32 {
            let batch_end = (i + 31).min(chunks32);
            let mut acc = [_mm256_setzero_si256(); 4];
            for j in i..batch_end {
                let off = j * 32;
                let vq = _mm256_loadu_si256(query.as_ptr().add(off) as *const __m256i);
                for r in 0..4 {
                    let vr = _mm256_loadu_si256(rows[r].as_ptr().add(off) as *const __m256i);
                    let xored = _mm256_xor_si256(vq, vr);
                    let lo = _mm256_and_si256(xored, low_mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(xored, 4), low_mask);
                    acc[r] = _mm256_add_epi8(
                        acc[r],
                        _mm256_add_epi8(
                            _mm256_shuffle_epi8(lookup, lo),
                            _mm256_shuffle_epi8(lookup, hi),
                        ),
                    );
                }
            }
            for r in 0..4 {
                let sad = _mm256_sad_epu8(acc[r], _mm256_setzero_si256());
                total[r] += _mm256_extract_epi64(sad, 0) as u64
                    + _mm256_extract_epi64(sad, 1) as u64
                    + _mm256_extract_epi64(sad, 2) as u64
                    + _mm256_extract_epi64(sad, 3) as u64;
            }
            i = batch_end;
        }

        // Remainder through the u64 scalar path: a per-byte tail would dominate
        // for code widths narrower than one vector (e.g. 64-bit fields).
        let base = chunks32 * 32;
        if base < len {
            let tail = &query[base..];
            for r in 0..4 {
                total[r] += u64::from(super::hamming_distance_scalar(tail, &rows[r][base..]));
            }
        }

        [
            total[0] as u32,
            total[1] as u32,
            total[2] as u32,
            total[3] as u32,
        ]
    }

    /// Check if AVX2 is available at runtime
    #[inline]
    pub fn is_available() -> bool {
        is_x86_feature_detected!("avx2")
    }
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================

mod scalar {
    /// Scalar unpack for 8-bit values
    #[inline]
    pub fn unpack_8bit(input: &[u8], output: &mut [u32], count: usize) {
        for i in 0..count {
            output[i] = input[i] as u32;
        }
    }

    /// Scalar unpack for 16-bit values
    #[inline]
    pub fn unpack_16bit(input: &[u8], output: &mut [u32], count: usize) {
        for (i, out) in output.iter_mut().enumerate().take(count) {
            let idx = i * 2;
            *out = u16::from_le_bytes([input[idx], input[idx + 1]]) as u32;
        }
    }

    /// Scalar unpack for 32-bit values
    #[inline]
    pub fn unpack_32bit(input: &[u8], output: &mut [u32], count: usize) {
        for (i, out) in output.iter_mut().enumerate().take(count) {
            let idx = i * 4;
            *out = u32::from_le_bytes([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]]);
        }
    }

    /// Scalar delta decode
    #[inline]
    pub fn delta_decode(output: &mut [u32], deltas: &[u32], first_doc_id: u32, count: usize) {
        if count == 0 {
            return;
        }

        output[0] = first_doc_id;
        let mut carry = first_doc_id;

        for i in 0..count - 1 {
            carry = carry.wrapping_add(deltas[i]).wrapping_add(1);
            output[i + 1] = carry;
        }
    }

    /// Scalar add 1 to all values
    #[inline]
    pub fn add_one(values: &mut [u32], count: usize) {
        for val in values.iter_mut().take(count) {
            *val += 1;
        }
    }

    /// Fused unpack + delta decode of `count` values: `output[0] = first_value`
    /// and `output[i + 1] = output[i] + delta[i] + OFFSET` (wrapping), where
    /// each little-endian delta occupies `BYTES` (1 or 2) bytes of `input`.
    ///
    /// This is the single scalar definition of the fused kernels: the SIMD
    /// kernels call it for their sub-vector tails (with `first_value` set to
    /// the last vector result and the slices advanced to it) and the
    /// dispatchers use it as the non-SIMD fallback. `count == 0` is a no-op.
    #[inline]
    pub fn delta_decode_with_offset<const OFFSET: u32, const BYTES: usize>(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        const {
            assert!(
                BYTES == 1 || BYTES == 2,
                "scalar delta decode supports 8/16-bit deltas"
            );
        }
        if count == 0 {
            return;
        }
        output[0] = first_value;
        let mut carry = first_value;
        for i in 0..count - 1 {
            let idx = i * BYTES;
            let delta = if BYTES == 1 {
                input[idx] as u32
            } else {
                u16::from_le_bytes([input[idx], input[idx + 1]]) as u32
            };
            carry = carry.wrapping_add(delta).wrapping_add(OFFSET);
            output[i + 1] = carry;
        }
    }
}

// ============================================================================
// Public dispatch functions that select SIMD or scalar at runtime
// ============================================================================

/// Unpack 8-bit packed values to u32 with SIMD acceleration
#[inline]
pub fn unpack_8bit(input: &[u8], output: &mut [u32], count: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                neon::unpack_8bit(input, output, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX2 (256-bit) over SSE (128-bit) when available
        if avx2::is_available() {
            unsafe {
                avx2::unpack_8bit(input, output, count);
            }
            return;
        }
        if sse::is_available() {
            unsafe {
                sse::unpack_8bit(input, output, count);
            }
            return;
        }
    }

    scalar::unpack_8bit(input, output, count);
}

/// Unpack 16-bit packed values to u32 with SIMD acceleration
#[inline]
pub fn unpack_16bit(input: &[u8], output: &mut [u32], count: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                neon::unpack_16bit(input, output, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX2 (256-bit) over SSE (128-bit) when available
        if avx2::is_available() {
            unsafe {
                avx2::unpack_16bit(input, output, count);
            }
            return;
        }
        if sse::is_available() {
            unsafe {
                sse::unpack_16bit(input, output, count);
            }
            return;
        }
    }

    scalar::unpack_16bit(input, output, count);
}

/// Unpack 32-bit packed values to u32 with SIMD acceleration
#[inline]
pub fn unpack_32bit(input: &[u8], output: &mut [u32], count: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                neon::unpack_32bit(input, output, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX2 (256-bit) over SSE (128-bit) when available
        if avx2::is_available() {
            unsafe {
                avx2::unpack_32bit(input, output, count);
            }
            return;
        }
        if sse::is_available() {
            unsafe {
                sse::unpack_32bit(input, output, count);
            }
            return;
        }
    }

    scalar::unpack_32bit(input, output, count);
}

/// Delta decode with SIMD acceleration
///
/// Converts delta-encoded values to absolute values.
/// Input: `deltas[i] = value[i + 1] - value[i] - 1` (gap minus one)
/// Output: absolute values starting from first_value
#[inline]
pub fn delta_decode(output: &mut [u32], deltas: &[u32], first_value: u32, count: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                neon::delta_decode(output, deltas, first_value, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            unsafe {
                sse::delta_decode(output, deltas, first_value, count);
            }
            return;
        }
    }

    scalar::delta_decode(output, deltas, first_value, count);
}

/// Add 1 to all values with SIMD acceleration
///
/// Used for TF decoding where values are stored as (tf - 1)
#[inline]
pub fn add_one(values: &mut [u32], count: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                neon::add_one(values, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX2 (256-bit) over SSE (128-bit) when available
        if avx2::is_available() {
            unsafe {
                avx2::add_one(values, count);
            }
            return;
        }
        if sse::is_available() {
            unsafe {
                sse::add_one(values, count);
            }
            return;
        }
    }

    scalar::add_one(values, count);
}

/// Compute the number of bits needed to represent a value
#[inline]
pub fn bits_needed(val: u32) -> u8 {
    if val == 0 {
        0
    } else {
        32 - val.leading_zeros() as u8
    }
}

// ============================================================================
// Rounded bitpacking for truly vectorized encoding/decoding
// ============================================================================
//
// Instead of using arbitrary bit widths (1-32), we round up to SIMD-friendly
// widths: 0, 8, 16, or 32 bits. This trades ~10-20% more space for much faster
// decoding since we can use direct SIMD widening instructions (pmovzx) without
// any bit-shifting or masking.
//
// Bit width mapping:
//   0      -> 0  (all zeros)
//   1-8    -> 8  (u8)
//   9-16   -> 16 (u16)
//   17-32  -> 32 (u32)

/// Rounded bit width type for SIMD-friendly encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundedBitWidth {
    Zero = 0,
    Bits8 = 8,
    Bits16 = 16,
    Bits32 = 32,
}

impl RoundedBitWidth {
    /// Round an exact bit width to the nearest SIMD-friendly width
    #[inline]
    pub fn from_exact(bits: u8) -> Self {
        match bits {
            0 => RoundedBitWidth::Zero,
            1..=8 => RoundedBitWidth::Bits8,
            9..=16 => RoundedBitWidth::Bits16,
            _ => RoundedBitWidth::Bits32,
        }
    }

    /// Convert from a stored u8 value; `None` unless it is exactly 0, 8, 16,
    /// or 32. Readers of persisted headers must use this and surface `None`
    /// as corruption instead of guessing a width.
    #[inline]
    pub fn try_from_u8(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(RoundedBitWidth::Zero),
            8 => Some(RoundedBitWidth::Bits8),
            16 => Some(RoundedBitWidth::Bits16),
            32 => Some(RoundedBitWidth::Bits32),
            _ => None,
        }
    }

    /// Convert from a stored u8 value that has already been validated to be
    /// 0, 8, 16, or 32 (see [`Self::try_from_u8`]). Any other value maps to
    /// `Bits32`, which is only acceptable after the header has been checked.
    #[inline]
    pub fn from_u8(bits: u8) -> Self {
        Self::try_from_u8(bits).unwrap_or(RoundedBitWidth::Bits32)
    }

    /// Get the byte size per value
    #[inline]
    pub fn bytes_per_value(self) -> usize {
        match self {
            RoundedBitWidth::Zero => 0,
            RoundedBitWidth::Bits8 => 1,
            RoundedBitWidth::Bits16 => 2,
            RoundedBitWidth::Bits32 => 4,
        }
    }

    /// Get the raw bit width value
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Round a bit width to the nearest SIMD-friendly width (0, 8, 16, or 32)
#[inline]
pub fn round_bit_width(bits: u8) -> u8 {
    RoundedBitWidth::from_exact(bits).as_u8()
}

/// Pack values using rounded bit width (SIMD-friendly)
///
/// This is much simpler than arbitrary bitpacking since values are byte-aligned.
/// Returns the number of bytes written.
#[inline]
pub fn pack_rounded(values: &[u32], bit_width: RoundedBitWidth, output: &mut [u8]) -> usize {
    let count = values.len();
    match bit_width {
        RoundedBitWidth::Zero => 0,
        RoundedBitWidth::Bits8 => {
            for (i, &v) in values.iter().enumerate() {
                output[i] = v as u8;
            }
            count
        }
        RoundedBitWidth::Bits16 => {
            for (i, &v) in values.iter().enumerate() {
                let bytes = (v as u16).to_le_bytes();
                output[i * 2] = bytes[0];
                output[i * 2 + 1] = bytes[1];
            }
            count * 2
        }
        RoundedBitWidth::Bits32 => {
            for (i, &v) in values.iter().enumerate() {
                let bytes = v.to_le_bytes();
                output[i * 4] = bytes[0];
                output[i * 4 + 1] = bytes[1];
                output[i * 4 + 2] = bytes[2];
                output[i * 4 + 3] = bytes[3];
            }
            count * 4
        }
    }
}

/// Unpack values using rounded bit width with SIMD acceleration
///
/// This is the fast path - no bit manipulation needed, just widening.
#[inline]
pub fn unpack_rounded(input: &[u8], bit_width: RoundedBitWidth, output: &mut [u32], count: usize) {
    match bit_width {
        RoundedBitWidth::Zero => {
            for out in output.iter_mut().take(count) {
                *out = 0;
            }
        }
        RoundedBitWidth::Bits8 => unpack_8bit(input, output, count),
        RoundedBitWidth::Bits16 => unpack_16bit(input, output, count),
        RoundedBitWidth::Bits32 => unpack_32bit(input, output, count),
    }
}

/// Decode actual rounded gaps (unlike legacy gap-minus-one streams).
/// Shares the ISA kernels; no intermediate unpacked delta buffer is needed.
#[inline]
pub(crate) fn unpack_rounded_raw_delta_decode(
    input: &[u8],
    bit_width: RoundedBitWidth,
    output: &mut [u32],
    first_value: u32,
    count: usize,
) {
    match bit_width {
        RoundedBitWidth::Zero => output.iter_mut().take(count).for_each(|v| *v = first_value),
        RoundedBitWidth::Bits8 => {
            unpack_8bit_delta_decode_with_offset::<0>(input, output, first_value, count)
        }
        RoundedBitWidth::Bits16 => {
            unpack_16bit_delta_decode_with_offset::<0>(input, output, first_value, count)
        }
        RoundedBitWidth::Bits32 => {
            if count > 0 {
                output[0] = first_value;
                let mut carry = first_value;
                for i in 0..count - 1 {
                    let offset = i * 4;
                    let delta = u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap());
                    carry = carry.wrapping_add(delta);
                    output[i + 1] = carry;
                }
            }
        }
    }
}

/// Fused unpack + delta decode using rounded bit width
///
/// Combines unpacking and prefix sum in a single pass for better cache utilization.
#[inline]
pub fn unpack_rounded_delta_decode(
    input: &[u8],
    bit_width: RoundedBitWidth,
    output: &mut [u32],
    first_value: u32,
    count: usize,
) {
    match bit_width {
        RoundedBitWidth::Zero => {
            // All deltas are 0, meaning gaps of 1
            let mut val = first_value;
            for out in output.iter_mut().take(count) {
                *out = val;
                val = val.wrapping_add(1);
            }
        }
        RoundedBitWidth::Bits8 => unpack_8bit_delta_decode(input, output, first_value, count),
        RoundedBitWidth::Bits16 => unpack_16bit_delta_decode(input, output, first_value, count),
        RoundedBitWidth::Bits32 => {
            // Unpack count-1 deltas from input, then prefix sum to absolute values
            if count > 0 {
                output[0] = first_value;
                let mut carry = first_value;
                for i in 0..count - 1 {
                    let idx = i * 4;
                    let delta = u32::from_le_bytes([
                        input[idx],
                        input[idx + 1],
                        input[idx + 2],
                        input[idx + 3],
                    ]);
                    carry = carry.wrapping_add(delta).wrapping_add(1);
                    output[i + 1] = carry;
                }
            }
        }
    }
}

// ============================================================================
// Fused operations for better cache utilization
// ============================================================================

/// Fused unpack 8-bit + delta decode in a single pass
///
/// This avoids writing the intermediate unpacked values to memory,
/// improving cache utilization for large blocks.
#[inline]
pub fn unpack_8bit_delta_decode(input: &[u8], output: &mut [u32], first_value: u32, count: usize) {
    unpack_8bit_delta_decode_with_offset::<1>(input, output, first_value, count);
}

/// Fused unpack 8-bit + delta decode with a configurable per-gap `OFFSET`.
///
/// Safe boundary for the unchecked ISA kernels: panics (once per block, not
/// per value) unless `input` holds `count - 1` delta bytes and `output` holds
/// `count` values.
#[inline]
pub(crate) fn unpack_8bit_delta_decode_with_offset<const OFFSET: u32>(
    input: &[u8],
    output: &mut [u32],
    first_value: u32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    assert_delta_decode_bounds(input.len(), output.len(), count, 1);

    output[0] = first_value;
    if count == 1 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            // SAFETY: bounds asserted above; NEON availability checked.
            unsafe {
                neon::unpack_8bit_delta_decode_with_offset::<OFFSET>(
                    input,
                    output,
                    first_value,
                    count,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            // SAFETY: bounds asserted above; AVX2 availability checked.
            unsafe {
                avx2::unpack_8bit_delta_decode_with_offset::<OFFSET>(
                    input,
                    output,
                    first_value,
                    count,
                );
            }
            return;
        }
        if sse::is_available() {
            // SAFETY: bounds asserted above; SSE4.1 availability checked.
            unsafe {
                sse::unpack_8bit_delta_decode_with_offset::<OFFSET>(
                    input,
                    output,
                    first_value,
                    count,
                );
            }
            return;
        }
    }

    scalar::delta_decode_with_offset::<OFFSET, 1>(input, output, first_value, count);
}

/// Fused unpack 16-bit + delta decode in a single pass
#[inline]
pub fn unpack_16bit_delta_decode(input: &[u8], output: &mut [u32], first_value: u32, count: usize) {
    unpack_16bit_delta_decode_with_offset::<1>(input, output, first_value, count);
}

/// Fused unpack 16-bit + delta decode with a configurable per-gap `OFFSET`.
///
/// Safe boundary for the unchecked ISA kernels: panics (once per block, not
/// per value) unless `input` holds `(count - 1) * 2` delta bytes and
/// `output` holds `count` values.
#[inline]
pub(crate) fn unpack_16bit_delta_decode_with_offset<const OFFSET: u32>(
    input: &[u8],
    output: &mut [u32],
    first_value: u32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    assert_delta_decode_bounds(input.len(), output.len(), count, 2);

    output[0] = first_value;
    if count == 1 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            // SAFETY: bounds asserted above; NEON availability checked.
            unsafe {
                neon::unpack_16bit_delta_decode_with_offset::<OFFSET>(
                    input,
                    output,
                    first_value,
                    count,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            // SAFETY: bounds asserted above; AVX2 availability checked.
            unsafe {
                avx2::unpack_16bit_delta_decode_with_offset::<OFFSET>(
                    input,
                    output,
                    first_value,
                    count,
                );
            }
            return;
        }
        if sse::is_available() {
            // SAFETY: bounds asserted above; SSE4.1 availability checked.
            unsafe {
                sse::unpack_16bit_delta_decode_with_offset::<OFFSET>(
                    input,
                    output,
                    first_value,
                    count,
                );
            }
            return;
        }
    }

    scalar::delta_decode_with_offset::<OFFSET, 2>(input, output, first_value, count);
}

/// Bounds contract shared by the fused delta-decode dispatchers: the ISA
/// kernels read `(count - 1) * bytes_per_delta` input bytes and write `count`
/// outputs without checks, so a short slice must fail here, loudly.
#[inline]
fn assert_delta_decode_bounds(input_len: usize, output_len: usize, count: usize, bytes: usize) {
    assert!(
        output_len >= count,
        "fused delta decode: output holds {output_len} values, block needs {count}"
    );
    let needed = (count - 1) * bytes;
    assert!(
        input_len >= needed,
        "fused delta decode: input holds {input_len} bytes, block needs {needed}"
    );
}

/// Fused unpack + delta decode for arbitrary bit widths
///
/// Combines unpacking and prefix sum in a single pass, avoiding intermediate buffer.
/// Uses SIMD-accelerated paths for 8/16-bit widths, scalar for others.
#[inline]
pub fn unpack_delta_decode(
    input: &[u8],
    bit_width: u8,
    output: &mut [u32],
    first_value: u32,
    count: usize,
) {
    if count == 0 {
        return;
    }

    output[0] = first_value;
    if count == 1 {
        return;
    }

    // Fast paths for SIMD-friendly bit widths
    match bit_width {
        0 => {
            // All zeros = consecutive doc IDs (gap of 1)
            let mut val = first_value;
            for item in output.iter_mut().take(count).skip(1) {
                val = val.wrapping_add(1);
                *item = val;
            }
        }
        8 => unpack_8bit_delta_decode(input, output, first_value, count),
        16 => unpack_16bit_delta_decode(input, output, first_value, count),
        32 => {
            // 32-bit: unpack inline and delta decode
            let mut carry = first_value;
            for i in 0..count - 1 {
                let idx = i * 4;
                let delta = u32::from_le_bytes([
                    input[idx],
                    input[idx + 1],
                    input[idx + 2],
                    input[idx + 3],
                ]);
                carry = carry.wrapping_add(delta).wrapping_add(1);
                output[i + 1] = carry;
            }
        }
        _ => {
            // Generic bit width: fused unpack + delta decode
            let mask = (1u64 << bit_width) - 1;
            let bit_width_usize = bit_width as usize;
            let mut bit_pos = 0usize;
            let input_ptr = input.as_ptr();
            let mut carry = first_value;

            for i in 0..count - 1 {
                let byte_idx = bit_pos >> 3;
                let bit_offset = bit_pos & 7;

                // SAFETY: Caller guarantees input has enough data
                let word = unsafe { (input_ptr.add(byte_idx) as *const u64).read_unaligned() };
                let delta = ((word >> bit_offset) & mask) as u32;

                carry = carry.wrapping_add(delta).wrapping_add(1);
                output[i + 1] = carry;
                bit_pos += bit_width_usize;
            }
        }
    }
}

// ============================================================================
// Sparse Vector SIMD Functions
// ============================================================================

/// Dequantize UInt8 weights to f32 with SIMD acceleration
///
/// Computes `output[i] = input[i] as f32 * scale + min_val`.
#[inline]
pub fn dequantize_uint8(input: &[u8], output: &mut [f32], scale: f32, min_val: f32, count: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                dequantize_uint8_neon(input, output, scale, min_val, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            unsafe {
                dequantize_uint8_sse(input, output, scale, min_val, count);
            }
            return;
        }
    }

    // Scalar fallback
    for i in 0..count {
        output[i] = input[i] as f32 * scale + min_val;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dequantize_uint8_neon(
    input: &[u8],
    output: &mut [f32],
    scale: f32,
    min_val: f32,
    count: usize,
) {
    use std::arch::aarch64::*;

    let scale_v = vdupq_n_f32(scale);
    let min_v = vdupq_n_f32(min_val);

    let chunks = count / 16;
    let remainder = count % 16;

    for chunk in 0..chunks {
        let base = chunk * 16;
        let in_ptr = input.as_ptr().add(base);

        // Load 16 bytes
        let bytes = vld1q_u8(in_ptr);

        // Widen u8 -> u16 -> u32 -> f32
        let low8 = vget_low_u8(bytes);
        let high8 = vget_high_u8(bytes);

        let low16 = vmovl_u8(low8);
        let high16 = vmovl_u8(high8);

        // Process 4 values at a time
        let u32_0 = vmovl_u16(vget_low_u16(low16));
        let u32_1 = vmovl_u16(vget_high_u16(low16));
        let u32_2 = vmovl_u16(vget_low_u16(high16));
        let u32_3 = vmovl_u16(vget_high_u16(high16));

        // Convert to f32 and apply scale + min_val
        let f32_0 = vfmaq_f32(min_v, vcvtq_f32_u32(u32_0), scale_v);
        let f32_1 = vfmaq_f32(min_v, vcvtq_f32_u32(u32_1), scale_v);
        let f32_2 = vfmaq_f32(min_v, vcvtq_f32_u32(u32_2), scale_v);
        let f32_3 = vfmaq_f32(min_v, vcvtq_f32_u32(u32_3), scale_v);

        let out_ptr = output.as_mut_ptr().add(base);
        vst1q_f32(out_ptr, f32_0);
        vst1q_f32(out_ptr.add(4), f32_1);
        vst1q_f32(out_ptr.add(8), f32_2);
        vst1q_f32(out_ptr.add(12), f32_3);
    }

    // Handle remainder
    let base = chunks * 16;
    for i in 0..remainder {
        output[base + i] = input[base + i] as f32 * scale + min_val;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dequantize_uint8_sse(
    input: &[u8],
    output: &mut [f32],
    scale: f32,
    min_val: f32,
    count: usize,
) {
    use std::arch::x86_64::*;

    let scale_v = _mm_set1_ps(scale);
    let min_v = _mm_set1_ps(min_val);

    let chunks = count / 4;
    let remainder = count % 4;

    for chunk in 0..chunks {
        let base = chunk * 4;

        // Load 4 bytes as a single i32 and zero-extend u8→u32 via SSE4.1
        let bytes = _mm_cvtsi32_si128(std::ptr::read_unaligned(
            input.as_ptr().add(base) as *const i32
        ));
        let ints = _mm_cvtepu8_epi32(bytes);
        let floats = _mm_cvtepi32_ps(ints);

        // Apply scale and min_val: result = floats * scale + min_val
        let scaled = _mm_add_ps(_mm_mul_ps(floats, scale_v), min_v);

        _mm_storeu_ps(output.as_mut_ptr().add(base), scaled);
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        output[base + i] = input[base + i] as f32 * scale + min_val;
    }
}

// ============================================================================
// Algebraic float reductions
// ============================================================================
//
// `f32::algebraic_add` / `algebraic_mul` (stable since Rust 1.98) permit the
// compiler to reassociate a floating-point reduction. That permission is the
// whole point: with strict IEEE `+` the loop-carried dependency on the
// accumulator pins these reductions to one scalar add per iteration, and LLVM
// is not allowed to split them into independent accumulator chains or vector
// lanes. With algebraic ops it vectorizes them the same way the hand-written
// NEON/AVX kernels below do by hand.
//
// Measured on aarch64 (Apple Silicon, rustc 1.98.0, opt-level=3, baseline
// target-cpu), before -> after ns/op:
//
//   dim    squared_l2       scalar dot     fused dot+norm   SOAR loss
//   128     55.5 -> 17.0     108 ->  16      145 ->  27      76.6 -> 18.8
//   384    220.8 -> 26.1     360 ->  32      365 ->  30      290  -> 80.0
//   768    592.8 -> 63.2     956 ->  55      752 ->  52      538  -> 67.7
//   1536  1827.8 -> 123.6   2032 -> 131     1404 -> 123     1464  -> 287
//
// i.e. 3-17x depending on width, largest at embedding-sized dimensions.
// Relative error against a strict f64 reference stays under 1e-6.
//
// End to end on `benches/vector_indexing.rs` (criterion, source-only diff):
// ivf_coarse_training/257 clusters -29.1%, /64 clusters -16.4%,
// ivf_tq_plan/64 -21.9%, ivf_tq_plan/16 -9.4%. See
// docs/algebraic-float-reductions.md.
//
// These operations are always safe (never UB), but they are *not*
// bit-reproducible across builds: a different rustc version, target CPU, or
// inlining decision may pick a different reduction order and move the last few
// ULPs. That variance already exists in every kernel in this module —
// `dot_product_f32` rounds differently on NEON (4 accumulators), AVX2 (4),
// AVX-512 (4) and the scalar path (1), so a query scored on an Apple Silicon
// replica already does not bit-match the same query on an AVX-512 replica.
// Using algebraic ops in the scalar paths therefore adds no new *class* of
// variance, only the same one at vector speed.
//
// Do not use these where a float is compared for bit-exact equality, hashed, or
// written into a content-addressed artifact.

/// Dot product of two equal-length f32 slices, reassociated for vectorization.
#[inline]
fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |acc, (&x, &y)| {
        acc.algebraic_add(x.algebraic_mul(y))
    })
}

/// Fused dot(a, b) and dot(b, b) over equal-length f32 slices in one pass.
#[inline]
fn fused_dot_norm_scalar(a: &[f32], b: &[f32]) -> (f32, f32) {
    a.iter()
        .zip(b)
        .fold((0.0f32, 0.0f32), |(dot, norm), (&x, &y)| {
            (
                dot.algebraic_add(x.algebraic_mul(y)),
                norm.algebraic_add(y.algebraic_mul(y)),
            )
        })
}

/// Squared L2 distance `||a - b||^2` between two equal-length f32 slices.
///
/// The element-wise subtraction stays strict IEEE; only the summation is
/// reassociated. Iterates over `min(a.len(), b.len())` elements.
#[inline]
pub fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |acc, (&x, &y)| {
        let delta = x - y;
        acc.algebraic_add(delta.algebraic_mul(delta))
    })
}

/// Squared L2 norm `||v||^2` of an f32 slice.
#[inline]
pub fn norm_squared_f32(v: &[f32]) -> f32 {
    v.iter()
        .fold(0.0f32, |acc, &x| acc.algebraic_add(x.algebraic_mul(x)))
}

/// L2 norm `||v||` of an f32 slice.
#[inline]
pub fn norm_f32(v: &[f32]) -> f32 {
    norm_squared_f32(v).sqrt()
}

/// f32 dot / fused dot+norm kernel resolved once for a whole batch.
///
/// The batch scorers (`batch_*_precomp`) call a kernel once per stored
/// vector; resolving it up front keeps runtime feature detection out of that
/// loop (mirrors [`HammingKernel`]). Every variant has the scalar fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseF32Kernel {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    #[cfg(target_arch = "x86_64")]
    Sse,
    Scalar,
}

impl DenseF32Kernel {
    /// Detect the widest kernel this CPU supports.
    #[inline]
    pub fn resolve() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if neon::is_available() {
                Self::Neon
            } else {
                Self::Scalar
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self::Avx2Fma;
            }
            if sse::is_available() {
                return Self::Sse;
            }
            Self::Scalar
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }

    /// `dot(a[..count], b[..count])`. Callers guarantee `count` is in bounds.
    #[inline]
    pub fn dot(self, a: &[f32], b: &[f32], count: usize) -> f32 {
        debug_assert!(count <= a.len() && count <= b.len());
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { dot_product_f32_neon(a, b, count) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => unsafe { dot_product_f32_avx512(a, b, count) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx2Fma => unsafe { dot_product_f32_avx2(a, b, count) },
            #[cfg(target_arch = "x86_64")]
            Self::Sse => unsafe { dot_product_f32_sse(a, b, count) },
            Self::Scalar => dot_product_f32_scalar(&a[..count], &b[..count]),
        }
    }

    /// `(dot(a, b), dot(b, b))` over the first `count` elements.
    #[inline]
    pub fn fused_dot_norm(self, a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
        debug_assert!(count <= a.len() && count <= b.len());
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { fused_dot_norm_neon(a, b, count) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => unsafe { fused_dot_norm_avx512(a, b, count) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx2Fma => unsafe { fused_dot_norm_avx2(a, b, count) },
            #[cfg(target_arch = "x86_64")]
            Self::Sse => unsafe { fused_dot_norm_sse(a, b, count) },
            Self::Scalar => fused_dot_norm_scalar(&a[..count], &b[..count]),
        }
    }
}

/// Compute dot product of two f32 arrays with SIMD acceleration
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32], count: usize) -> f32 {
    assert!(
        count <= a.len() && count <= b.len(),
        "dot_product_f32 count {count} exceeds input lengths ({}, {})",
        a.len(),
        b.len()
    );
    DenseF32Kernel::resolve().dot(a, b, count)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::aarch64::*;

    let chunks16 = count / 16;
    let remainder = count % 16;

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    for c in 0..chunks16 {
        let base = c * 16;
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(a.as_ptr().add(base)),
            vld1q_f32(b.as_ptr().add(base)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(a.as_ptr().add(base + 4)),
            vld1q_f32(b.as_ptr().add(base + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(a.as_ptr().add(base + 8)),
            vld1q_f32(b.as_ptr().add(base + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(a.as_ptr().add(base + 12)),
            vld1q_f32(b.as_ptr().add(base + 12)),
        );
    }

    let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut sum = vaddvq_f32(acc);

    // Up to 15 trailing elements: one 4-lane accumulator over the remaining
    // full lane groups, then an algebraic scalar tail of at most 3.
    let mut base = chunks16 * 16;
    // LLVM's algebraic scalar loop lowers an exactly 8-lane remainder to a
    // better two-vector reduction than this generic accumulator on NEON.
    // Keep the manual tail for 4/12 lanes, where it wins.
    if remainder >= 4 && remainder != 8 {
        let mut tail = vdupq_n_f32(0.0);
        while base + 4 <= count {
            tail = vfmaq_f32(
                tail,
                vld1q_f32(a.as_ptr().add(base)),
                vld1q_f32(b.as_ptr().add(base)),
            );
            base += 4;
        }
        sum += vaddvq_f32(tail);
    }
    for i in base..count {
        sum = sum.algebraic_add(a[i].algebraic_mul(b[i]));
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks32 = count / 32;
    let remainder = count % 32;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    for c in 0..chunks32 {
        let base = c * 32;
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(base)),
            _mm256_loadu_ps(b.as_ptr().add(base)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(base + 8)),
            _mm256_loadu_ps(b.as_ptr().add(base + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(base + 16)),
            _mm256_loadu_ps(b.as_ptr().add(base + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(base + 24)),
            _mm256_loadu_ps(b.as_ptr().add(base + 24)),
            acc3,
        );
    }

    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    // Horizontal sum: 256-bit → 128-bit → scalar
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);

    let mut sum = _mm_cvtss_f32(final_sum);

    // Up to 31 trailing elements: one 8-lane accumulator over the remaining
    // full lane groups, then an algebraic scalar tail of at most 7.
    let mut base = chunks32 * 32;
    if remainder >= 8 {
        let mut tail = _mm256_setzero_ps();
        while base + 8 <= count {
            tail = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(base)),
                _mm256_loadu_ps(b.as_ptr().add(base)),
                tail,
            );
            base += 8;
        }
        let hi = _mm256_extractf128_ps(tail, 1);
        let lo = _mm256_castps256_ps128(tail);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        sum += _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    }
    for i in base..count {
        sum = sum.algebraic_add(a[i].algebraic_mul(b[i]));
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_f32_sse(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut acc = _mm_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let vb = _mm_loadu_ps(b.as_ptr().add(base));
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }

    // Horizontal sum: [a, b, c, d] -> a + b + c + d
    let shuf = _mm_shuffle_ps(acc, acc, 0b10_11_00_01); // [b, a, d, c]
    let sums = _mm_add_ps(acc, shuf); // [a+b, a+b, c+d, c+d]
    let shuf2 = _mm_movehl_ps(sums, sums); // [c+d, c+d, ?, ?]
    let final_sum = _mm_add_ss(sums, shuf2); // [a+b+c+d, ?, ?, ?]

    let mut sum = _mm_cvtss_f32(final_sum);

    // Handle remainder (at most 3 elements)
    let base = chunks * 4;
    for i in 0..remainder {
        sum = sum.algebraic_add(a[base + i].algebraic_mul(b[base + i]));
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_f32_avx512(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks64 = count / 64;
    let remainder = count % 64;

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();

    for c in 0..chunks64 {
        let base = c * 64;
        acc0 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a.as_ptr().add(base)),
            _mm512_loadu_ps(b.as_ptr().add(base)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a.as_ptr().add(base + 16)),
            _mm512_loadu_ps(b.as_ptr().add(base + 16)),
            acc1,
        );
        acc2 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a.as_ptr().add(base + 32)),
            _mm512_loadu_ps(b.as_ptr().add(base + 32)),
            acc2,
        );
        acc3 = _mm512_fmadd_ps(
            _mm512_loadu_ps(a.as_ptr().add(base + 48)),
            _mm512_loadu_ps(b.as_ptr().add(base + 48)),
            acc3,
        );
    }

    let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut sum = _mm512_reduce_add_ps(acc);

    // Up to 63 trailing elements: one 16-lane accumulator over the remaining
    // full lane groups, then an algebraic scalar tail of at most 15.
    let mut base = chunks64 * 64;
    if remainder >= 16 {
        let mut tail = _mm512_setzero_ps();
        while base + 16 <= count {
            tail = _mm512_fmadd_ps(
                _mm512_loadu_ps(a.as_ptr().add(base)),
                _mm512_loadu_ps(b.as_ptr().add(base)),
                tail,
            );
            base += 16;
        }
        sum += _mm512_reduce_add_ps(tail);
    }
    for i in base..count {
        sum = sum.algebraic_add(a[i].algebraic_mul(b[i]));
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_avx512(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let chunks64 = count / 64;
    let remainder = count % 64;

    let mut d0 = _mm512_setzero_ps();
    let mut d1 = _mm512_setzero_ps();
    let mut d2 = _mm512_setzero_ps();
    let mut d3 = _mm512_setzero_ps();
    let mut n0 = _mm512_setzero_ps();
    let mut n1 = _mm512_setzero_ps();
    let mut n2 = _mm512_setzero_ps();
    let mut n3 = _mm512_setzero_ps();

    for c in 0..chunks64 {
        let base = c * 64;
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(base));
        d0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(base)), vb0, d0);
        n0 = _mm512_fmadd_ps(vb0, vb0, n0);
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(base + 16));
        d1 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(base + 16)), vb1, d1);
        n1 = _mm512_fmadd_ps(vb1, vb1, n1);
        let vb2 = _mm512_loadu_ps(b.as_ptr().add(base + 32));
        d2 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(base + 32)), vb2, d2);
        n2 = _mm512_fmadd_ps(vb2, vb2, n2);
        let vb3 = _mm512_loadu_ps(b.as_ptr().add(base + 48));
        d3 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(base + 48)), vb3, d3);
        n3 = _mm512_fmadd_ps(vb3, vb3, n3);
    }

    let acc_dot = _mm512_add_ps(_mm512_add_ps(d0, d1), _mm512_add_ps(d2, d3));
    let acc_norm = _mm512_add_ps(_mm512_add_ps(n0, n1), _mm512_add_ps(n2, n3));
    let mut dot = _mm512_reduce_add_ps(acc_dot);
    let mut norm = _mm512_reduce_add_ps(acc_norm);

    let mut base = chunks64 * 64;
    if remainder >= 16 {
        let mut tail_dot = _mm512_setzero_ps();
        let mut tail_norm = _mm512_setzero_ps();
        while base + 16 <= count {
            let vb = _mm512_loadu_ps(b.as_ptr().add(base));
            tail_dot = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(base)), vb, tail_dot);
            tail_norm = _mm512_fmadd_ps(vb, vb, tail_norm);
            base += 16;
        }
        dot += _mm512_reduce_add_ps(tail_dot);
        norm += _mm512_reduce_add_ps(tail_norm);
    }
    for i in base..count {
        dot = dot.algebraic_add(a[i].algebraic_mul(b[i]));
        norm = norm.algebraic_add(b[i].algebraic_mul(b[i]));
    }

    (dot, norm)
}

// ============================================================================
// Batched Cosine Similarity for Dense Vector Search
// ============================================================================

/// Fused dot-product + self-norm in a single pass (SIMD accelerated).
///
/// Returns (dot(a, b), dot(b, b)) — i.e. the dot product of a·b and ||b||².
/// Loads `b` only once (halves memory bandwidth vs two separate dot products).
#[inline]
fn fused_dot_norm(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    DenseF32Kernel::resolve().fused_dot_norm(a, b, count)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_neon(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    use std::arch::aarch64::*;

    let chunks16 = count / 16;
    let remainder = count % 16;

    let mut d0 = vdupq_n_f32(0.0);
    let mut d1 = vdupq_n_f32(0.0);
    let mut d2 = vdupq_n_f32(0.0);
    let mut d3 = vdupq_n_f32(0.0);
    let mut n0 = vdupq_n_f32(0.0);
    let mut n1 = vdupq_n_f32(0.0);
    let mut n2 = vdupq_n_f32(0.0);
    let mut n3 = vdupq_n_f32(0.0);

    for c in 0..chunks16 {
        let base = c * 16;
        let va0 = vld1q_f32(a.as_ptr().add(base));
        let vb0 = vld1q_f32(b.as_ptr().add(base));
        d0 = vfmaq_f32(d0, va0, vb0);
        n0 = vfmaq_f32(n0, vb0, vb0);
        let va1 = vld1q_f32(a.as_ptr().add(base + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(base + 4));
        d1 = vfmaq_f32(d1, va1, vb1);
        n1 = vfmaq_f32(n1, vb1, vb1);
        let va2 = vld1q_f32(a.as_ptr().add(base + 8));
        let vb2 = vld1q_f32(b.as_ptr().add(base + 8));
        d2 = vfmaq_f32(d2, va2, vb2);
        n2 = vfmaq_f32(n2, vb2, vb2);
        let va3 = vld1q_f32(a.as_ptr().add(base + 12));
        let vb3 = vld1q_f32(b.as_ptr().add(base + 12));
        d3 = vfmaq_f32(d3, va3, vb3);
        n3 = vfmaq_f32(n3, vb3, vb3);
    }

    let acc_dot = vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3));
    let acc_norm = vaddq_f32(vaddq_f32(n0, n1), vaddq_f32(n2, n3));
    let mut dot = vaddvq_f32(acc_dot);
    let mut norm = vaddvq_f32(acc_norm);

    let mut base = chunks16 * 16;
    if remainder >= 4 {
        let mut tail_dot = vdupq_n_f32(0.0);
        let mut tail_norm = vdupq_n_f32(0.0);
        while base + 4 <= count {
            let vb = vld1q_f32(b.as_ptr().add(base));
            tail_dot = vfmaq_f32(tail_dot, vld1q_f32(a.as_ptr().add(base)), vb);
            tail_norm = vfmaq_f32(tail_norm, vb, vb);
            base += 4;
        }
        dot += vaddvq_f32(tail_dot);
        norm += vaddvq_f32(tail_norm);
    }
    for i in base..count {
        dot = dot.algebraic_add(a[i].algebraic_mul(b[i]));
        norm = norm.algebraic_add(b[i].algebraic_mul(b[i]));
    }

    (dot, norm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_avx2(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let chunks32 = count / 32;
    let remainder = count % 32;

    let mut d0 = _mm256_setzero_ps();
    let mut d1 = _mm256_setzero_ps();
    let mut d2 = _mm256_setzero_ps();
    let mut d3 = _mm256_setzero_ps();
    let mut n0 = _mm256_setzero_ps();
    let mut n1 = _mm256_setzero_ps();
    let mut n2 = _mm256_setzero_ps();
    let mut n3 = _mm256_setzero_ps();

    for c in 0..chunks32 {
        let base = c * 32;
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(base));
        d0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(base)), vb0, d0);
        n0 = _mm256_fmadd_ps(vb0, vb0, n0);
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
        d1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(base + 8)), vb1, d1);
        n1 = _mm256_fmadd_ps(vb1, vb1, n1);
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(base + 16));
        d2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(base + 16)), vb2, d2);
        n2 = _mm256_fmadd_ps(vb2, vb2, n2);
        let vb3 = _mm256_loadu_ps(b.as_ptr().add(base + 24));
        d3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(base + 24)), vb3, d3);
        n3 = _mm256_fmadd_ps(vb3, vb3, n3);
    }

    let acc_dot = _mm256_add_ps(_mm256_add_ps(d0, d1), _mm256_add_ps(d2, d3));
    let acc_norm = _mm256_add_ps(_mm256_add_ps(n0, n1), _mm256_add_ps(n2, n3));

    // Horizontal sums: 256→128→scalar
    let hi_d = _mm256_extractf128_ps(acc_dot, 1);
    let lo_d = _mm256_castps256_ps128(acc_dot);
    let sum_d = _mm_add_ps(lo_d, hi_d);
    let shuf_d = _mm_shuffle_ps(sum_d, sum_d, 0b10_11_00_01);
    let sums_d = _mm_add_ps(sum_d, shuf_d);
    let shuf2_d = _mm_movehl_ps(sums_d, sums_d);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums_d, shuf2_d));

    let hi_n = _mm256_extractf128_ps(acc_norm, 1);
    let lo_n = _mm256_castps256_ps128(acc_norm);
    let sum_n = _mm_add_ps(lo_n, hi_n);
    let shuf_n = _mm_shuffle_ps(sum_n, sum_n, 0b10_11_00_01);
    let sums_n = _mm_add_ps(sum_n, shuf_n);
    let shuf2_n = _mm_movehl_ps(sums_n, sums_n);
    let mut norm = _mm_cvtss_f32(_mm_add_ss(sums_n, shuf2_n));

    let mut base = chunks32 * 32;
    if remainder >= 8 {
        let mut tail_dot = _mm256_setzero_ps();
        let mut tail_norm = _mm256_setzero_ps();
        while base + 8 <= count {
            let vb = _mm256_loadu_ps(b.as_ptr().add(base));
            tail_dot = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(base)), vb, tail_dot);
            tail_norm = _mm256_fmadd_ps(vb, vb, tail_norm);
            base += 8;
        }
        let reduce = |v: __m256| -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
        };
        dot += reduce(tail_dot);
        norm += reduce(tail_norm);
    }
    for i in base..count {
        dot = dot.algebraic_add(a[i].algebraic_mul(b[i]));
        norm = norm.algebraic_add(b[i].algebraic_mul(b[i]));
    }

    (dot, norm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_sse(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut acc_dot = _mm_setzero_ps();
    let mut acc_norm = _mm_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let vb = _mm_loadu_ps(b.as_ptr().add(base));
        acc_dot = _mm_add_ps(acc_dot, _mm_mul_ps(va, vb));
        acc_norm = _mm_add_ps(acc_norm, _mm_mul_ps(vb, vb));
    }

    // Horizontal sums
    let shuf_d = _mm_shuffle_ps(acc_dot, acc_dot, 0b10_11_00_01);
    let sums_d = _mm_add_ps(acc_dot, shuf_d);
    let shuf2_d = _mm_movehl_ps(sums_d, sums_d);
    let final_d = _mm_add_ss(sums_d, shuf2_d);
    let mut dot = _mm_cvtss_f32(final_d);

    let shuf_n = _mm_shuffle_ps(acc_norm, acc_norm, 0b10_11_00_01);
    let sums_n = _mm_add_ps(acc_norm, shuf_n);
    let shuf2_n = _mm_movehl_ps(sums_n, sums_n);
    let final_n = _mm_add_ss(sums_n, shuf2_n);
    let mut norm = _mm_cvtss_f32(final_n);

    let base = chunks * 4;
    for i in 0..remainder {
        dot = dot.algebraic_add(a[base + i].algebraic_mul(b[base + i]));
        norm = norm.algebraic_add(b[base + i].algebraic_mul(b[base + i]));
    }

    (dot, norm)
}

/// Fast approximate reciprocal square root: 1/sqrt(x).
///
/// Uses the IEEE 754 bit trick (Quake III) + one Newton-Raphson iteration
/// for ~23-bit precision — sufficient for cosine similarity scoring.
/// ~3-5x faster than `1.0 / x.sqrt()` on most architectures.
#[inline]
pub fn fast_inv_sqrt(x: f32) -> f32 {
    let half = 0.5 * x;
    let i = 0x5F37_5A86_u32.wrapping_sub(x.to_bits() >> 1);
    let y = f32::from_bits(i);
    let y = y * (1.5 - half * y * y); // first Newton-Raphson step
    y * (1.5 - half * y * y) // second step: ~23-bit precision
}

/// Batch cosine similarity: query vs N contiguous vectors.
///
/// `vectors` is a contiguous buffer of `n * dim` floats (row-major).
/// `scores` must have length >= n.
///
/// Optimizations over calling `cosine_similarity` N times:
/// 1. Query norm computed once (not N times)
/// 2. Fused dot+norm kernel — each vector loaded once (halves bandwidth)
/// 3. No per-call overhead (branch prediction, function calls)
/// 4. Fast reciprocal square root (~3-5x faster than 1/sqrt)
#[inline]
pub fn batch_cosine_scores(query: &[f32], vectors: &[f32], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
    let required = n
        .checked_mul(dim)
        .expect("batch cosine vector length overflow");
    assert_eq!(query.len(), dim, "batch cosine query dimension mismatch");
    assert!(
        vectors.len() >= required,
        "batch cosine vectors are truncated: need {required}, got {}",
        vectors.len()
    );

    if dim == 0 || n == 0 {
        return;
    }

    // Pre-compute query inverse norm once
    let norm_q_sq = dot_product_f32(query, query, dim);
    if norm_q_sq < f32::EPSILON {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    let inv_norm_q = fast_inv_sqrt(norm_q_sq);

    for i in 0..n {
        let vec = &vectors[i * dim..(i + 1) * dim];
        let (dot, norm_v_sq) = fused_dot_norm(query, vec, dim);
        if norm_v_sq < f32::EPSILON {
            scores[i] = 0.0;
        } else {
            scores[i] = dot * inv_norm_q * fast_inv_sqrt(norm_v_sq);
        }
    }
}

// ============================================================================
// f16 (IEEE 754 half-precision) conversion
// ============================================================================

/// Convert f32 to f16 (IEEE 754 half-precision), stored as u16
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exp == 255 {
        // Inf/NaN
        return (sign | 0x7C00 | ((mantissa >> 13) & 0x3FF)) as u16;
    }

    let exp16 = exp - 127 + 15;

    if exp16 >= 31 {
        return (sign | 0x7C00) as u16; // overflow → infinity
    }

    if exp16 <= 0 {
        if exp16 < -10 {
            return sign as u16; // too small → zero
        }
        let shift = (1 - exp16) as u32;
        let m = (mantissa | 0x80_0000) >> shift;
        // Round-to-nearest-even
        let round_bit = (m >> 12) & 1;
        let sticky = m & 0xFFF;
        let m13 = m >> 13;
        let rounded = m13 + (round_bit & (m13 | if sticky != 0 { 1 } else { 0 }));
        return (sign | rounded) as u16;
    }

    // Round-to-nearest-even for normal numbers
    let round_bit = (mantissa >> 12) & 1;
    let sticky = mantissa & 0xFFF;
    let m13 = mantissa >> 13;
    let rounded = m13 + (round_bit & (m13 | if sticky != 0 { 1 } else { 0 }));
    // Check if rounding caused mantissa overflow (carry into exponent)
    if rounded > 0x3FF {
        let exp16_inc = exp16 as u32 + 1;
        if exp16_inc >= 31 {
            return (sign | 0x7C00) as u16; // overflow → infinity
        }
        (sign | (exp16_inc << 10)) as u16
    } else {
        (sign | ((exp16 as u32) << 10) | rounded) as u16
    }
}

/// Convert f16 (stored as u16) to f32
#[inline]
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: normalize
        let mut e = 0u32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        return f32::from_bits(sign | ((127 - 15 + 1 - e) << 23) | ((m & 0x3FF) << 13));
    }

    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13));
    }

    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mantissa << 13))
}

// ============================================================================
// uint8 scalar quantization for [-1, 1] range
// ============================================================================

const U8_SCALE: f32 = 127.5;
const U8_INV_SCALE: f32 = 1.0 / 127.5;

/// Quantize f32 in [-1, 1] to u8 [0, 255]
#[inline]
pub fn f32_to_u8_saturating(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) + 1.0) * U8_SCALE) as u8
}

/// Dequantize u8 [0, 255] to f32 in [-1, 1]
#[inline]
pub fn u8_to_f32(byte: u8) -> f32 {
    byte as f32 * U8_INV_SCALE - 1.0
}

// ============================================================================
// Batch conversion (used during builder write)
// ============================================================================

/// Batch convert f32 slice to f16 (stored as u16)
pub fn batch_f32_to_f16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_f16(*s);
    }
}

/// Batch convert an f32 slice to u8 with `[-1, 1]` to `[0, 255]` mapping.
pub fn batch_f32_to_u8(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_u8_saturating(*s);
    }
}

// ============================================================================
// NEON-accelerated fused dot+norm for quantized vectors
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod neon_quant {
    use std::arch::aarch64::*;

    /// Fused dot(query_f16, vec_f16) + norm(vec_f16) for f16 vectors on NEON.
    ///
    /// Both query and vectors are f16 (stored as u16). Uses hardware `vcvt_f32_f16`
    /// for SIMD f16→f32 conversion (replaces scalar bit manipulation), processes
    /// 8 elements per iteration with f32 accumulation for precision.
    #[allow(clippy::incompatible_msrv)]
    #[target_feature(enable = "neon")]
    pub unsafe fn fused_dot_norm_f16(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
        let chunks16 = dim / 16;
        let remainder = dim % 16;

        // 2 accumulator pairs to hide FMA latency (processes 16 f16 per iteration)
        let mut acc_dot0 = vdupq_n_f32(0.0);
        let mut acc_dot1 = vdupq_n_f32(0.0);
        let mut acc_norm0 = vdupq_n_f32(0.0);
        let mut acc_norm1 = vdupq_n_f32(0.0);

        for c in 0..chunks16 {
            let base = c * 16;

            // First 8 f16 elements
            let v_raw0 = vld1q_u16(vec_f16.as_ptr().add(base));
            let v_lo0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v_raw0)));
            let v_hi0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(v_raw0)));
            let q_raw0 = vld1q_u16(query_f16.as_ptr().add(base));
            let q_lo0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(q_raw0)));
            let q_hi0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(q_raw0)));

            acc_dot0 = vfmaq_f32(acc_dot0, q_lo0, v_lo0);
            acc_dot0 = vfmaq_f32(acc_dot0, q_hi0, v_hi0);
            acc_norm0 = vfmaq_f32(acc_norm0, v_lo0, v_lo0);
            acc_norm0 = vfmaq_f32(acc_norm0, v_hi0, v_hi0);

            // Second 8 f16 elements (independent accumulator chain)
            let v_raw1 = vld1q_u16(vec_f16.as_ptr().add(base + 8));
            let v_lo1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v_raw1)));
            let v_hi1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(v_raw1)));
            let q_raw1 = vld1q_u16(query_f16.as_ptr().add(base + 8));
            let q_lo1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(q_raw1)));
            let q_hi1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(q_raw1)));

            acc_dot1 = vfmaq_f32(acc_dot1, q_lo1, v_lo1);
            acc_dot1 = vfmaq_f32(acc_dot1, q_hi1, v_hi1);
            acc_norm1 = vfmaq_f32(acc_norm1, v_lo1, v_lo1);
            acc_norm1 = vfmaq_f32(acc_norm1, v_hi1, v_hi1);
        }

        // Combine accumulator pairs
        let mut dot = vaddvq_f32(vaddq_f32(acc_dot0, acc_dot1));
        let mut norm = vaddvq_f32(vaddq_f32(acc_norm0, acc_norm1));

        // Handle remainder
        let base = chunks16 * 16;
        for i in 0..remainder {
            let v = super::f16_to_f32(*vec_f16.get_unchecked(base + i));
            let q = super::f16_to_f32(*query_f16.get_unchecked(base + i));
            dot += q * v;
            norm += v * v;
        }

        (dot, norm)
    }

    /// Fused dot(query, vec) + norm(vec) for u8 vectors on NEON.
    /// Processes 16 u8 values per iteration using NEON widening chain.
    #[target_feature(enable = "neon")]
    pub unsafe fn fused_dot_norm_u8(query: &[f32], vec_u8: &[u8], dim: usize) -> (f32, f32) {
        let scale = vdupq_n_f32(super::U8_INV_SCALE);
        let offset = vdupq_n_f32(-1.0);

        let chunks16 = dim / 16;
        let remainder = dim % 16;

        let mut acc_dot = vdupq_n_f32(0.0);
        let mut acc_norm = vdupq_n_f32(0.0);

        for c in 0..chunks16 {
            let base = c * 16;

            // Load 16 u8 values
            let bytes = vld1q_u8(vec_u8.as_ptr().add(base));

            // Widen: 16×u8 → 2×8×u16 → 4×4×u32 → 4×4×f32
            let lo8 = vget_low_u8(bytes);
            let hi8 = vget_high_u8(bytes);
            let lo16 = vmovl_u8(lo8);
            let hi16 = vmovl_u8(hi8);

            let f0 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))), scale),
                offset,
            );
            let f1 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))), scale),
                offset,
            );
            let f2 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))), scale),
                offset,
            );
            let f3 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))), scale),
                offset,
            );

            let q0 = vld1q_f32(query.as_ptr().add(base));
            let q1 = vld1q_f32(query.as_ptr().add(base + 4));
            let q2 = vld1q_f32(query.as_ptr().add(base + 8));
            let q3 = vld1q_f32(query.as_ptr().add(base + 12));

            acc_dot = vfmaq_f32(acc_dot, q0, f0);
            acc_dot = vfmaq_f32(acc_dot, q1, f1);
            acc_dot = vfmaq_f32(acc_dot, q2, f2);
            acc_dot = vfmaq_f32(acc_dot, q3, f3);

            acc_norm = vfmaq_f32(acc_norm, f0, f0);
            acc_norm = vfmaq_f32(acc_norm, f1, f1);
            acc_norm = vfmaq_f32(acc_norm, f2, f2);
            acc_norm = vfmaq_f32(acc_norm, f3, f3);
        }

        let mut dot = vaddvq_f32(acc_dot);
        let mut norm = vaddvq_f32(acc_norm);

        let base = chunks16 * 16;
        for i in 0..remainder {
            let v = super::u8_to_f32(*vec_u8.get_unchecked(base + i));
            dot += *query.get_unchecked(base + i) * v;
            norm += v * v;
        }

        (dot, norm)
    }

    /// Dot product only for f16 vectors on NEON (no norm — for unit_norm vectors).
    #[allow(clippy::incompatible_msrv)]
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_f16(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> f32 {
        let chunks8 = dim / 8;
        let remainder = dim % 8;

        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks8 {
            let base = c * 8;
            let v_raw = vld1q_u16(vec_f16.as_ptr().add(base));
            let v_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v_raw)));
            let v_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(v_raw)));
            let q_raw = vld1q_u16(query_f16.as_ptr().add(base));
            let q_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(q_raw)));
            let q_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(q_raw)));
            acc = vfmaq_f32(acc, q_lo, v_lo);
            acc = vfmaq_f32(acc, q_hi, v_hi);
        }

        let mut dot = vaddvq_f32(acc);
        let base = chunks8 * 8;
        for i in 0..remainder {
            let v = super::f16_to_f32(*vec_f16.get_unchecked(base + i));
            let q = super::f16_to_f32(*query_f16.get_unchecked(base + i));
            dot += q * v;
        }
        dot
    }

    /// Dot product only for u8 vectors on NEON (no norm — for unit_norm vectors).
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_u8(query: &[f32], vec_u8: &[u8], dim: usize) -> f32 {
        let scale = vdupq_n_f32(super::U8_INV_SCALE);
        let offset = vdupq_n_f32(-1.0);
        let chunks16 = dim / 16;
        let remainder = dim % 16;

        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks16 {
            let base = c * 16;
            let bytes = vld1q_u8(vec_u8.as_ptr().add(base));
            let lo8 = vget_low_u8(bytes);
            let hi8 = vget_high_u8(bytes);
            let lo16 = vmovl_u8(lo8);
            let hi16 = vmovl_u8(hi8);
            let f0 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))), scale),
                offset,
            );
            let f1 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))), scale),
                offset,
            );
            let f2 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))), scale),
                offset,
            );
            let f3 = vaddq_f32(
                vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))), scale),
                offset,
            );
            let q0 = vld1q_f32(query.as_ptr().add(base));
            let q1 = vld1q_f32(query.as_ptr().add(base + 4));
            let q2 = vld1q_f32(query.as_ptr().add(base + 8));
            let q3 = vld1q_f32(query.as_ptr().add(base + 12));
            acc = vfmaq_f32(acc, q0, f0);
            acc = vfmaq_f32(acc, q1, f1);
            acc = vfmaq_f32(acc, q2, f2);
            acc = vfmaq_f32(acc, q3, f3);
        }

        let mut dot = vaddvq_f32(acc);
        let base = chunks16 * 16;
        for i in 0..remainder {
            let v = super::u8_to_f32(*vec_u8.get_unchecked(base + i));
            dot += *query.get_unchecked(base + i) * v;
        }
        dot
    }
}

// ============================================================================
// Scalar fallback for fused dot+norm on quantized vectors
// ============================================================================

fn fused_dot_norm_f16_scalar(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
    (0..dim).fold((0.0f32, 0.0f32), |(dot, norm), i| {
        let v = f16_to_f32(vec_f16[i]);
        let q = f16_to_f32(query_f16[i]);
        (
            dot.algebraic_add(q.algebraic_mul(v)),
            norm.algebraic_add(v.algebraic_mul(v)),
        )
    })
}

fn fused_dot_norm_u8_scalar(query: &[f32], vec_u8: &[u8], dim: usize) -> (f32, f32) {
    (0..dim).fold((0.0f32, 0.0f32), |(dot, norm), i| {
        let v = u8_to_f32(vec_u8[i]);
        (
            dot.algebraic_add(query[i].algebraic_mul(v)),
            norm.algebraic_add(v.algebraic_mul(v)),
        )
    })
}

fn dot_product_f16_scalar(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> f32 {
    (0..dim).fold(0.0f32, |dot, i| {
        dot.algebraic_add(f16_to_f32(query_f16[i]).algebraic_mul(f16_to_f32(vec_f16[i])))
    })
}

fn dot_product_u8_scalar(query: &[f32], vec_u8: &[u8], dim: usize) -> f32 {
    (0..dim).fold(0.0f32, |dot, i| {
        dot.algebraic_add(query[i].algebraic_mul(u8_to_f32(vec_u8[i])))
    })
}

// ============================================================================
// x86_64 SSE4.1 quantized fused dot+norm
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_f16_sse(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let chunks = dim / 4;
    let remainder = dim % 4;

    let mut acc_dot = _mm_setzero_ps();
    let mut acc_norm = _mm_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 4;
        // Load 4 f16 values and convert to f32 using scalar conversion
        let v0 = f16_to_f32(*vec_f16.get_unchecked(base));
        let v1 = f16_to_f32(*vec_f16.get_unchecked(base + 1));
        let v2 = f16_to_f32(*vec_f16.get_unchecked(base + 2));
        let v3 = f16_to_f32(*vec_f16.get_unchecked(base + 3));
        let vb = _mm_set_ps(v3, v2, v1, v0);

        let q0 = f16_to_f32(*query_f16.get_unchecked(base));
        let q1 = f16_to_f32(*query_f16.get_unchecked(base + 1));
        let q2 = f16_to_f32(*query_f16.get_unchecked(base + 2));
        let q3 = f16_to_f32(*query_f16.get_unchecked(base + 3));
        let va = _mm_set_ps(q3, q2, q1, q0);

        acc_dot = _mm_add_ps(acc_dot, _mm_mul_ps(va, vb));
        acc_norm = _mm_add_ps(acc_norm, _mm_mul_ps(vb, vb));
    }

    // Horizontal sums
    let shuf_d = _mm_shuffle_ps(acc_dot, acc_dot, 0b10_11_00_01);
    let sums_d = _mm_add_ps(acc_dot, shuf_d);
    let shuf2_d = _mm_movehl_ps(sums_d, sums_d);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums_d, shuf2_d));

    let shuf_n = _mm_shuffle_ps(acc_norm, acc_norm, 0b10_11_00_01);
    let sums_n = _mm_add_ps(acc_norm, shuf_n);
    let shuf2_n = _mm_movehl_ps(sums_n, sums_n);
    let mut norm = _mm_cvtss_f32(_mm_add_ss(sums_n, shuf2_n));

    let base = chunks * 4;
    for i in 0..remainder {
        let v = f16_to_f32(*vec_f16.get_unchecked(base + i));
        let q = f16_to_f32(*query_f16.get_unchecked(base + i));
        dot += q * v;
        norm += v * v;
    }

    (dot, norm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_u8_sse(query: &[f32], vec_u8: &[u8], dim: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let scale = _mm_set1_ps(U8_INV_SCALE);
    let offset = _mm_set1_ps(-1.0);

    let chunks = dim / 4;
    let remainder = dim % 4;

    let mut acc_dot = _mm_setzero_ps();
    let mut acc_norm = _mm_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 4;

        // Load 4 bytes, zero-extend to i32, convert to f32, dequantize
        let bytes = _mm_cvtsi32_si128(std::ptr::read_unaligned(
            vec_u8.as_ptr().add(base) as *const i32
        ));
        let ints = _mm_cvtepu8_epi32(bytes);
        let floats = _mm_cvtepi32_ps(ints);
        let vb = _mm_add_ps(_mm_mul_ps(floats, scale), offset);

        let va = _mm_loadu_ps(query.as_ptr().add(base));

        acc_dot = _mm_add_ps(acc_dot, _mm_mul_ps(va, vb));
        acc_norm = _mm_add_ps(acc_norm, _mm_mul_ps(vb, vb));
    }

    // Horizontal sums
    let shuf_d = _mm_shuffle_ps(acc_dot, acc_dot, 0b10_11_00_01);
    let sums_d = _mm_add_ps(acc_dot, shuf_d);
    let shuf2_d = _mm_movehl_ps(sums_d, sums_d);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums_d, shuf2_d));

    let shuf_n = _mm_shuffle_ps(acc_norm, acc_norm, 0b10_11_00_01);
    let sums_n = _mm_add_ps(acc_norm, shuf_n);
    let shuf2_n = _mm_movehl_ps(sums_n, sums_n);
    let mut norm = _mm_cvtss_f32(_mm_add_ss(sums_n, shuf2_n));

    let base = chunks * 4;
    for i in 0..remainder {
        let v = u8_to_f32(*vec_u8.get_unchecked(base + i));
        dot += *query.get_unchecked(base + i) * v;
        norm += v * v;
    }

    (dot, norm)
}

// ============================================================================
// x86_64 F16C + AVX + FMA accelerated f16 scoring
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "f16c", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_f16_f16c(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let chunks16 = dim / 16;
    let remainder = dim % 16;

    // 2 accumulator pairs to hide FMA latency (processes 16 f16 per iteration)
    let mut acc_dot0 = _mm256_setzero_ps();
    let mut acc_dot1 = _mm256_setzero_ps();
    let mut acc_norm0 = _mm256_setzero_ps();
    let mut acc_norm1 = _mm256_setzero_ps();

    for c in 0..chunks16 {
        let base = c * 16;

        // First 8 f16 elements
        let v_raw0 = _mm_loadu_si128(vec_f16.as_ptr().add(base) as *const __m128i);
        let vb0 = _mm256_cvtph_ps(v_raw0);
        let q_raw0 = _mm_loadu_si128(query_f16.as_ptr().add(base) as *const __m128i);
        let qa0 = _mm256_cvtph_ps(q_raw0);
        acc_dot0 = _mm256_fmadd_ps(qa0, vb0, acc_dot0);
        acc_norm0 = _mm256_fmadd_ps(vb0, vb0, acc_norm0);

        // Second 8 f16 elements (independent accumulator chain)
        let v_raw1 = _mm_loadu_si128(vec_f16.as_ptr().add(base + 8) as *const __m128i);
        let vb1 = _mm256_cvtph_ps(v_raw1);
        let q_raw1 = _mm_loadu_si128(query_f16.as_ptr().add(base + 8) as *const __m128i);
        let qa1 = _mm256_cvtph_ps(q_raw1);
        acc_dot1 = _mm256_fmadd_ps(qa1, vb1, acc_dot1);
        acc_norm1 = _mm256_fmadd_ps(vb1, vb1, acc_norm1);
    }

    // Combine accumulator pairs
    let acc_dot = _mm256_add_ps(acc_dot0, acc_dot1);
    let acc_norm = _mm256_add_ps(acc_norm0, acc_norm1);

    // Horizontal sum 256→128→scalar
    let hi_d = _mm256_extractf128_ps(acc_dot, 1);
    let lo_d = _mm256_castps256_ps128(acc_dot);
    let sum_d = _mm_add_ps(lo_d, hi_d);
    let shuf_d = _mm_shuffle_ps(sum_d, sum_d, 0b10_11_00_01);
    let sums_d = _mm_add_ps(sum_d, shuf_d);
    let shuf2_d = _mm_movehl_ps(sums_d, sums_d);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums_d, shuf2_d));

    let hi_n = _mm256_extractf128_ps(acc_norm, 1);
    let lo_n = _mm256_castps256_ps128(acc_norm);
    let sum_n = _mm_add_ps(lo_n, hi_n);
    let shuf_n = _mm_shuffle_ps(sum_n, sum_n, 0b10_11_00_01);
    let sums_n = _mm_add_ps(sum_n, shuf_n);
    let shuf2_n = _mm_movehl_ps(sums_n, sums_n);
    let mut norm = _mm_cvtss_f32(_mm_add_ss(sums_n, shuf2_n));

    let base = chunks16 * 16;
    for i in 0..remainder {
        let v = f16_to_f32(*vec_f16.get_unchecked(base + i));
        let q = f16_to_f32(*query_f16.get_unchecked(base + i));
        dot += q * v;
        norm += v * v;
    }

    (dot, norm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "f16c", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_f16_f16c(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = dim / 8;
    let remainder = dim % 8;
    let mut acc = _mm256_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 8;
        let v_raw = _mm_loadu_si128(vec_f16.as_ptr().add(base) as *const __m128i);
        let vb = _mm256_cvtph_ps(v_raw);
        let q_raw = _mm_loadu_si128(query_f16.as_ptr().add(base) as *const __m128i);
        let qa = _mm256_cvtph_ps(q_raw);
        acc = _mm256_fmadd_ps(qa, vb, acc);
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum = _mm_add_ps(lo, hi);
    let shuf = _mm_shuffle_ps(sum, sum, 0b10_11_00_01);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

    let base = chunks * 8;
    for i in 0..remainder {
        let v = f16_to_f32(*vec_f16.get_unchecked(base + i));
        let q = f16_to_f32(*query_f16.get_unchecked(base + i));
        dot += q * v;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2", enable = "sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_u8_sse(query: &[f32], vec_u8: &[u8], dim: usize) -> f32 {
    use std::arch::x86_64::*;

    let scale = _mm_set1_ps(U8_INV_SCALE);
    let offset = _mm_set1_ps(-1.0);
    let chunks = dim / 4;
    let remainder = dim % 4;
    let mut acc = _mm_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 4;
        let bytes = _mm_cvtsi32_si128(std::ptr::read_unaligned(
            vec_u8.as_ptr().add(base) as *const i32
        ));
        let ints = _mm_cvtepu8_epi32(bytes);
        let floats = _mm_cvtepi32_ps(ints);
        let vb = _mm_add_ps(_mm_mul_ps(floats, scale), offset);
        let va = _mm_loadu_ps(query.as_ptr().add(base));
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }

    let shuf = _mm_shuffle_ps(acc, acc, 0b10_11_00_01);
    let sums = _mm_add_ps(acc, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

    let base = chunks * 4;
    for i in 0..remainder {
        dot += *query.get_unchecked(base + i) * u8_to_f32(*vec_u8.get_unchecked(base + i));
    }
    dot
}

// ============================================================================
// Platform dispatch
// ============================================================================

/// f16 scoring kernel resolved once per batch (see [`DenseF32Kernel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantF16Kernel {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    F16c,
    /// SSE4.1 fused kernel; the dot-only form has no SSE variant and falls
    /// back to scalar, exactly as the previous per-call dispatch did.
    #[cfg(target_arch = "x86_64")]
    Sse,
    Scalar,
}

impl QuantF16Kernel {
    #[inline]
    pub fn resolve() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("fma") {
                return Self::F16c;
            }
            if sse::is_available() {
                return Self::Sse;
            }
            Self::Scalar
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }

    #[inline]
    pub fn fused_dot_norm(self, query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon_quant::fused_dot_norm_f16(query_f16, vec_f16, dim) },
            #[cfg(target_arch = "x86_64")]
            Self::F16c => unsafe { fused_dot_norm_f16_f16c(query_f16, vec_f16, dim) },
            #[cfg(target_arch = "x86_64")]
            Self::Sse => unsafe { fused_dot_norm_f16_sse(query_f16, vec_f16, dim) },
            Self::Scalar => fused_dot_norm_f16_scalar(query_f16, vec_f16, dim),
        }
    }

    #[inline]
    pub fn dot(self, query_f16: &[u16], vec_f16: &[u16], dim: usize) -> f32 {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon_quant::dot_product_f16(query_f16, vec_f16, dim) },
            #[cfg(target_arch = "x86_64")]
            Self::F16c => unsafe { dot_product_f16_f16c(query_f16, vec_f16, dim) },
            #[cfg(target_arch = "x86_64")]
            Self::Sse => dot_product_f16_scalar(query_f16, vec_f16, dim),
            Self::Scalar => dot_product_f16_scalar(query_f16, vec_f16, dim),
        }
    }
}

/// u8 scoring kernel resolved once per batch (see [`DenseF32Kernel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantU8Kernel {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Sse,
    Scalar,
}

impl QuantU8Kernel {
    #[inline]
    pub fn resolve() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if sse::is_available() {
                return Self::Sse;
            }
            Self::Scalar
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }

    #[inline]
    pub fn fused_dot_norm(self, query: &[f32], vec_u8: &[u8], dim: usize) -> (f32, f32) {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon_quant::fused_dot_norm_u8(query, vec_u8, dim) },
            #[cfg(target_arch = "x86_64")]
            Self::Sse => unsafe { fused_dot_norm_u8_sse(query, vec_u8, dim) },
            Self::Scalar => fused_dot_norm_u8_scalar(query, vec_u8, dim),
        }
    }

    #[inline]
    pub fn dot(self, query: &[f32], vec_u8: &[u8], dim: usize) -> f32 {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon_quant::dot_product_u8(query, vec_u8, dim) },
            #[cfg(target_arch = "x86_64")]
            Self::Sse => unsafe { dot_product_u8_sse(query, vec_u8, dim) },
            Self::Scalar => dot_product_u8_scalar(query, vec_u8, dim),
        }
    }
}

#[inline]
fn fused_dot_norm_f16(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
    QuantF16Kernel::resolve().fused_dot_norm(query_f16, vec_f16, dim)
}

#[inline]
fn fused_dot_norm_u8(query: &[f32], vec_u8: &[u8], dim: usize) -> (f32, f32) {
    QuantU8Kernel::resolve().fused_dot_norm(query, vec_u8, dim)
}

// ── Dot-product-only dispatch (for unit_norm vectors) ─────────────────────

#[inline]
fn dot_product_f16_quant(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> f32 {
    QuantF16Kernel::resolve().dot(query_f16, vec_f16, dim)
}

#[inline]
fn dot_product_u8_quant(query: &[f32], vec_u8: &[u8], dim: usize) -> f32 {
    QuantU8Kernel::resolve().dot(query, vec_u8, dim)
}

// ============================================================================
// Public batch cosine scoring for quantized vectors
// ============================================================================

/// Batch cosine similarity: f32 query vs N contiguous f16 vectors.
///
/// `vectors_raw` is raw bytes: N vectors × dim × 2 bytes (f16 stored as u16).
/// Query is quantized to f16 once, then both query and vectors are scored in
/// f16 space using hardware SIMD conversion (8 elements/iteration on NEON).
/// Memory bandwidth is halved for both query and vector loads.
#[inline]
pub fn batch_cosine_scores_f16(query: &[f32], vectors_raw: &[u8], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
    let vec_bytes = dim.checked_mul(2).expect("f16 vector size overflow");
    let required = n
        .checked_mul(vec_bytes)
        .expect("f16 batch byte length overflow");
    assert_eq!(
        query.len(),
        dim,
        "f16 batch cosine query dimension mismatch"
    );
    assert!(
        vectors_raw.len() >= required,
        "f16 batch cosine vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    if required > 0 {
        assert!(
            (vectors_raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()),
            "f16 batch cosine vectors are not 2-byte aligned"
        );
    }
    if dim == 0 || n == 0 {
        return;
    }

    // Compute query inverse norm in f32 (full precision, before quantization)
    let norm_q_sq = dot_product_f32(query, query, dim);
    if norm_q_sq < f32::EPSILON {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    let inv_norm_q = fast_inv_sqrt(norm_q_sq);

    // Quantize query to f16 once (O(dim)), reused for all N vector scorings
    let query_f16: Vec<u16> = query.iter().map(|&v| f32_to_f16(v)).collect();

    for i in 0..n {
        let raw = &vectors_raw[i * vec_bytes..(i + 1) * vec_bytes];
        let f16_slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, dim) };

        let (dot, norm_v_sq) = fused_dot_norm_f16(&query_f16, f16_slice, dim);
        scores[i] = if norm_v_sq < f32::EPSILON {
            0.0
        } else {
            dot * inv_norm_q * fast_inv_sqrt(norm_v_sq)
        };
    }
}

/// Batch cosine similarity: f32 query vs N contiguous u8 vectors.
///
/// `vectors_raw` is raw bytes: N vectors × dim bytes (u8, mapping
/// `[-1, 1]` to `[0, 255]`).
/// Converts u8→f32 using NEON widening chain (16 values/iteration), scores with FMA.
/// Memory bandwidth is quartered compared to f32 scoring.
#[inline]
pub fn batch_cosine_scores_u8(query: &[f32], vectors_raw: &[u8], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
    let required = n.checked_mul(dim).expect("u8 batch byte length overflow");
    assert_eq!(query.len(), dim, "u8 batch cosine query dimension mismatch");
    assert!(
        vectors_raw.len() >= required,
        "u8 batch cosine vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    if dim == 0 || n == 0 {
        return;
    }

    let norm_q_sq = dot_product_f32(query, query, dim);
    if norm_q_sq < f32::EPSILON {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    let inv_norm_q = fast_inv_sqrt(norm_q_sq);

    for i in 0..n {
        let u8_slice = &vectors_raw[i * dim..(i + 1) * dim];

        let (dot, norm_v_sq) = fused_dot_norm_u8(query, u8_slice, dim);
        scores[i] = if norm_v_sq < f32::EPSILON {
            0.0
        } else {
            dot * inv_norm_q * fast_inv_sqrt(norm_v_sq)
        };
    }
}

// ============================================================================
// Batch dot-product scoring for unit-norm vectors
// ============================================================================

/// Batch dot-product scoring: f32 query vs N contiguous f32 unit-norm vectors.
///
/// For pre-normalized vectors (||v|| = 1), cosine = dot(q, v) / ||q||.
/// Skips per-vector norm computation — ~40% less work than `batch_cosine_scores`.
#[inline]
pub fn batch_dot_scores(query: &[f32], vectors: &[f32], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
    let required = n
        .checked_mul(dim)
        .expect("batch dot vector length overflow");
    assert_eq!(query.len(), dim, "batch dot query dimension mismatch");
    assert!(
        vectors.len() >= required,
        "batch dot vectors are truncated: need {required}, got {}",
        vectors.len()
    );

    if dim == 0 || n == 0 {
        return;
    }

    let norm_q_sq = dot_product_f32(query, query, dim);
    if norm_q_sq < f32::EPSILON {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    let inv_norm_q = fast_inv_sqrt(norm_q_sq);

    for i in 0..n {
        let vec = &vectors[i * dim..(i + 1) * dim];
        let dot = dot_product_f32(query, vec, dim);
        scores[i] = dot * inv_norm_q;
    }
}

/// Batch dot-product scoring: f32 query vs N contiguous f16 unit-norm vectors.
///
/// For pre-normalized vectors (||v|| = 1), cosine = dot(q, v) / ||q||.
/// Uses F16C/NEON hardware conversion + dot-only kernel.
#[inline]
pub fn batch_dot_scores_f16(query: &[f32], vectors_raw: &[u8], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
    let vec_bytes = dim.checked_mul(2).expect("f16 vector size overflow");
    let required = n
        .checked_mul(vec_bytes)
        .expect("f16 batch byte length overflow");
    assert_eq!(query.len(), dim, "f16 batch dot query dimension mismatch");
    assert!(
        vectors_raw.len() >= required,
        "f16 batch dot vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    if required > 0 {
        assert!(
            (vectors_raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()),
            "f16 batch dot vectors are not 2-byte aligned"
        );
    }
    if dim == 0 || n == 0 {
        return;
    }

    let norm_q_sq = dot_product_f32(query, query, dim);
    if norm_q_sq < f32::EPSILON {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    let inv_norm_q = fast_inv_sqrt(norm_q_sq);

    let query_f16: Vec<u16> = query.iter().map(|&v| f32_to_f16(v)).collect();
    for i in 0..n {
        let raw = &vectors_raw[i * vec_bytes..(i + 1) * vec_bytes];
        let f16_slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, dim) };
        let dot = dot_product_f16_quant(&query_f16, f16_slice, dim);
        scores[i] = dot * inv_norm_q;
    }
}

/// Batch dot-product scoring: f32 query vs N contiguous u8 unit-norm vectors.
///
/// For pre-normalized vectors (||v|| = 1), cosine = dot(q, v) / ||q||.
/// Uses NEON/SSE widening chain for u8→f32 conversion + dot-only kernel.
#[inline]
pub fn batch_dot_scores_u8(query: &[f32], vectors_raw: &[u8], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
    let required = n.checked_mul(dim).expect("u8 batch byte length overflow");
    assert_eq!(query.len(), dim, "u8 batch dot query dimension mismatch");
    assert!(
        vectors_raw.len() >= required,
        "u8 batch dot vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    if dim == 0 || n == 0 {
        return;
    }

    let norm_q_sq = dot_product_f32(query, query, dim);
    if norm_q_sq < f32::EPSILON {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    let inv_norm_q = fast_inv_sqrt(norm_q_sq);

    for i in 0..n {
        let u8_slice = &vectors_raw[i * dim..(i + 1) * dim];
        let dot = dot_product_u8_quant(query, u8_slice, dim);
        scores[i] = dot * inv_norm_q;
    }
}

// ============================================================================
// Precomputed-norm batch scoring (avoids redundant query norm + f16 conversion)
// ============================================================================

/// Batch cosine: f32 query vs N f32 vectors, with precomputed `inv_norm_q`.
#[inline]
pub fn batch_cosine_scores_precomp(
    query: &[f32],
    vectors: &[f32],
    dim: usize,
    scores: &mut [f32],
    inv_norm_q: f32,
) {
    let n = scores.len();
    let required = n
        .checked_mul(dim)
        .expect("precomputed cosine vector length overflow");
    assert_eq!(
        query.len(),
        dim,
        "precomputed cosine query dimension mismatch"
    );
    assert!(
        vectors.len() >= required,
        "precomputed cosine vectors are truncated: need {required}, got {}",
        vectors.len()
    );
    // Dispatch outside the row loop. Calling `DenseF32Kernel::fused_dot_norm`
    // through a loop-carried enum was measurably slower on Sapphire Rapids;
    // these arms preserve a direct target-feature call without repeating CPU
    // detection for every stored vector.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    macro_rules! score_simd_rows {
        ($kernel:path) => {{
            for i in 0..n {
                let vec = &vectors[i * dim..(i + 1) * dim];
                let (dot, norm_v_sq) = unsafe { $kernel(query, vec, dim) };
                scores[i] = if norm_v_sq < f32::EPSILON {
                    0.0
                } else {
                    dot * inv_norm_q * fast_inv_sqrt(norm_v_sq)
                };
            }
        }};
    }
    match DenseF32Kernel::resolve() {
        #[cfg(target_arch = "aarch64")]
        DenseF32Kernel::Neon => score_simd_rows!(fused_dot_norm_neon),
        #[cfg(target_arch = "x86_64")]
        DenseF32Kernel::Avx512 => score_simd_rows!(fused_dot_norm_avx512),
        #[cfg(target_arch = "x86_64")]
        DenseF32Kernel::Avx2Fma => score_simd_rows!(fused_dot_norm_avx2),
        #[cfg(target_arch = "x86_64")]
        DenseF32Kernel::Sse => score_simd_rows!(fused_dot_norm_sse),
        DenseF32Kernel::Scalar => {
            for i in 0..n {
                let vec = &vectors[i * dim..(i + 1) * dim];
                let (dot, norm_v_sq) = fused_dot_norm_scalar(query, vec);
                scores[i] = if norm_v_sq < f32::EPSILON {
                    0.0
                } else {
                    dot * inv_norm_q * fast_inv_sqrt(norm_v_sq)
                };
            }
        }
    }
}

/// Batch cosine: precomputed `inv_norm_q` + `query_f16` vs N f16 vectors.
#[inline]
pub fn batch_cosine_scores_f16_precomp(
    query_f16: &[u16],
    vectors_raw: &[u8],
    dim: usize,
    scores: &mut [f32],
    inv_norm_q: f32,
) {
    let n = scores.len();
    let vec_bytes = dim.checked_mul(2).expect("f16 vector size overflow");
    let required = n
        .checked_mul(vec_bytes)
        .expect("precomputed f16 cosine batch byte length overflow");
    assert_eq!(
        query_f16.len(),
        dim,
        "precomputed f16 cosine query dimension mismatch"
    );
    assert!(
        vectors_raw.len() >= required,
        "precomputed f16 cosine vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    if required > 0 {
        assert!(
            (vectors_raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()),
            "precomputed f16 cosine vectors are not 2-byte aligned"
        );
    }
    let kernel = QuantF16Kernel::resolve();
    for i in 0..n {
        let raw = &vectors_raw[i * vec_bytes..(i + 1) * vec_bytes];
        let f16_slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, dim) };
        let (dot, norm_v_sq) = kernel.fused_dot_norm(query_f16, f16_slice, dim);
        scores[i] = if norm_v_sq < f32::EPSILON {
            0.0
        } else {
            dot * inv_norm_q * fast_inv_sqrt(norm_v_sq)
        };
    }
}

/// Batch cosine: precomputed `inv_norm_q` vs N u8 vectors.
#[inline]
pub fn batch_cosine_scores_u8_precomp(
    query: &[f32],
    vectors_raw: &[u8],
    dim: usize,
    scores: &mut [f32],
    inv_norm_q: f32,
) {
    let n = scores.len();
    let required = n
        .checked_mul(dim)
        .expect("precomputed u8 cosine batch byte length overflow");
    assert_eq!(
        query.len(),
        dim,
        "precomputed u8 cosine query dimension mismatch"
    );
    assert!(
        vectors_raw.len() >= required,
        "precomputed u8 cosine vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    let kernel = QuantU8Kernel::resolve();
    for i in 0..n {
        let u8_slice = &vectors_raw[i * dim..(i + 1) * dim];
        let (dot, norm_v_sq) = kernel.fused_dot_norm(query, u8_slice, dim);
        scores[i] = if norm_v_sq < f32::EPSILON {
            0.0
        } else {
            dot * inv_norm_q * fast_inv_sqrt(norm_v_sq)
        };
    }
}

/// Batch dot-product: precomputed `inv_norm_q` vs N f32 unit-norm vectors.
#[inline]
pub fn batch_dot_scores_precomp(
    query: &[f32],
    vectors: &[f32],
    dim: usize,
    scores: &mut [f32],
    inv_norm_q: f32,
) {
    let n = scores.len();
    let required = n
        .checked_mul(dim)
        .expect("precomputed dot vector length overflow");
    assert_eq!(query.len(), dim, "precomputed dot query dimension mismatch");
    assert!(
        vectors.len() >= required,
        "precomputed dot vectors are truncated: need {required}, got {}",
        vectors.len()
    );
    let kernel = DenseF32Kernel::resolve();
    for i in 0..n {
        let vec = &vectors[i * dim..(i + 1) * dim];
        scores[i] = kernel.dot(query, vec, dim) * inv_norm_q;
    }
}

/// Batch dot-product: precomputed `inv_norm_q` + `query_f16` vs N f16 unit-norm vectors.
#[inline]
pub fn batch_dot_scores_f16_precomp(
    query_f16: &[u16],
    vectors_raw: &[u8],
    dim: usize,
    scores: &mut [f32],
    inv_norm_q: f32,
) {
    let n = scores.len();
    let vec_bytes = dim.checked_mul(2).expect("f16 vector size overflow");
    let required = n
        .checked_mul(vec_bytes)
        .expect("precomputed f16 dot batch byte length overflow");
    assert_eq!(
        query_f16.len(),
        dim,
        "precomputed f16 dot query dimension mismatch"
    );
    assert!(
        vectors_raw.len() >= required,
        "precomputed f16 dot vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    if required > 0 {
        assert!(
            (vectors_raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()),
            "precomputed f16 dot vectors are not 2-byte aligned"
        );
    }
    let kernel = QuantF16Kernel::resolve();
    for i in 0..n {
        let raw = &vectors_raw[i * vec_bytes..(i + 1) * vec_bytes];
        let f16_slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, dim) };
        scores[i] = kernel.dot(query_f16, f16_slice, dim) * inv_norm_q;
    }
}

/// Batch dot-product: precomputed `inv_norm_q` vs N u8 unit-norm vectors.
#[inline]
pub fn batch_dot_scores_u8_precomp(
    query: &[f32],
    vectors_raw: &[u8],
    dim: usize,
    scores: &mut [f32],
    inv_norm_q: f32,
) {
    let n = scores.len();
    let required = n
        .checked_mul(dim)
        .expect("precomputed u8 dot batch byte length overflow");
    assert_eq!(
        query.len(),
        dim,
        "precomputed u8 dot query dimension mismatch"
    );
    assert!(
        vectors_raw.len() >= required,
        "precomputed u8 dot vectors are truncated: need {required} bytes, got {}",
        vectors_raw.len()
    );
    let kernel = QuantU8Kernel::resolve();
    for i in 0..n {
        let u8_slice = &vectors_raw[i * dim..(i + 1) * dim];
        scores[i] = kernel.dot(query, u8_slice, dim) * inv_norm_q;
    }
}

/// Compute cosine similarity between two f32 vectors with SIMD acceleration
///
/// Returns dot(a,b) / (||a|| * ||b||), range [-1, 1]
/// Returns 0.0 if either vector has zero norm.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine vector dimension mismatch");
    let count = a.len();

    if count == 0 {
        return 0.0;
    }

    let dot = dot_product_f32(a, b, count);
    let norm_a = dot_product_f32(a, a, count);
    let norm_b = dot_product_f32(b, b, count);

    let denom = (norm_a * norm_b).sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }

    dot / denom
}

// ============================================================================
// Hamming distance for binary dense vectors
// ============================================================================

/// AVX-512 Hamming distance using `VPOPCNTDQ`.
///
/// Processes 64 bytes per iteration with a single hardware popcount per lane
/// group, which removes the nibble-lookup shuffles the AVX2 path needs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hamming_distance_avx512(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks64 = len / 64;
    let mut acc = _mm512_setzero_si512();

    for c in 0..chunks64 {
        let off = c * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(off) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(off) as *const __m512i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
    }

    let base = chunks64 * 64;
    _mm512_reduce_add_epi64(acc) as u32 + hamming_distance_scalar(&a[base..], &b[base..])
}

/// Four-row AVX-512 Hamming distance sharing the query load across rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn hamming_distance_x4_avx512(query: &[u8], rows: [&[u8]; 4]) -> [u32; 4] {
    use std::arch::x86_64::*;

    let len = query.len();
    let chunks64 = len / 64;
    let mut acc = [_mm512_setzero_si512(); 4];

    for c in 0..chunks64 {
        let off = c * 64;
        let vq = _mm512_loadu_si512(query.as_ptr().add(off) as *const __m512i);
        for r in 0..4 {
            let vr = _mm512_loadu_si512(rows[r].as_ptr().add(off) as *const __m512i);
            acc[r] = _mm512_add_epi64(acc[r], _mm512_popcnt_epi64(_mm512_xor_si512(vq, vr)));
        }
    }

    let base = chunks64 * 64;
    let tail = &query[base..];
    [
        _mm512_reduce_add_epi64(acc[0]) as u32 + hamming_distance_scalar(tail, &rows[0][base..]),
        _mm512_reduce_add_epi64(acc[1]) as u32 + hamming_distance_scalar(tail, &rows[1][base..]),
        _mm512_reduce_add_epi64(acc[2]) as u32 + hamming_distance_scalar(tail, &rows[2][base..]),
        _mm512_reduce_add_epi64(acc[3]) as u32 + hamming_distance_scalar(tail, &rows[3][base..]),
    ]
}

/// Four-row scalar Hamming distance sharing the query load across rows.
#[inline]
fn hamming_distance_x4_scalar(query: &[u8], rows: [&[u8]; 4]) -> [u32; 4] {
    let len = query.len();
    let chunks = len / 8;
    let mut total = [0u32; 4];

    for i in 0..chunks {
        let off = i * 8;
        let vq = unsafe { std::ptr::read_unaligned(query.as_ptr().add(off) as *const u64) };
        for r in 0..4 {
            let vr = unsafe { std::ptr::read_unaligned(rows[r].as_ptr().add(off) as *const u64) };
            total[r] += (vq ^ vr).count_ones();
        }
    }

    let base = chunks * 8;
    for k in base..len {
        let q = query[k];
        for r in 0..4 {
            total[r] += (q ^ rows[r][k]).count_ones();
        }
    }

    total
}

/// Rows scored per kernel invocation. Sharing the query load, the AVX2 nibble
/// lookup table and the horizontal reduction across four rows amortises the
/// non-inlinable `#[target_feature]` call and overlaps the popcount chains.
const HAMMING_ROWS_PER_KERNEL: usize = 4;

/// Architecture kernel resolved once for a whole scan.
///
/// Hot binary paths — HNSW centroid routing, k-majority assignment, leaf
/// scanning — score millions of code pairs against one query. Resolving the
/// kernel up front keeps runtime feature detection out of the inner loop, and
/// the row-batched entry points let one dispatch cover a whole neighbour list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HammingKernel {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

impl HammingKernel {
    /// Detect the widest kernel this CPU supports.
    #[inline]
    pub fn resolve() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
                return Self::Avx512;
            }
            if avx2::is_available() {
                return Self::Avx2;
            }
            Self::Scalar
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }

    /// Bytes consumed per vector iteration; below this width the SIMD kernel
    /// has no full vector to work on and runs entirely in its remainder loop.
    #[inline]
    fn vector_bytes(self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => 64,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => 32,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => 16,
            Self::Scalar => 8,
        }
    }

    /// Kernel to use for `byte_len`-byte codes. Codes narrower than one SIMD
    /// vector (e.g. 64-bit fields) go through the plain `u64::count_ones`
    /// loop: measured on aarch64/NEON, 1,024 rows of 8-byte codes score in
    /// 0.98 µs through the scalar loop against 1.21 µs through the NEON entry
    /// point, which was only executing its per-byte remainder. At one full
    /// vector or more the SIMD kernel wins (32 bytes: 1.23 vs 1.28 µs;
    /// 128 bytes: 2.73 vs 3.60 µs). AVX2/AVX-512 widths are not measured here;
    /// the rule is the same "no full vector, no SIMD" and cannot be slower than
    /// running the remainder loop alone.
    #[inline]
    fn for_byte_len(self, byte_len: usize) -> Self {
        if byte_len < self.vector_bytes() {
            Self::Scalar
        } else {
            self
        }
    }

    /// Hamming distance between two equal-length packed-bit vectors.
    #[inline]
    pub fn distance(self, a: &[u8], b: &[u8]) -> u32 {
        debug_assert_eq!(a.len(), b.len(), "Hamming vector byte length mismatch");
        match self.for_byte_len(a.len()) {
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => unsafe { hamming_distance_avx512(a, b) },
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { avx2::hamming_distance(a, b) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::hamming_distance(a, b) },
            Self::Scalar => hamming_distance_scalar(a, b),
        }
    }

    /// `out[i]` receives the distance from `query` to row `i` of `db`.
    pub fn distances(self, query: &[u8], db: &[u8], byte_len: usize, out: &mut [u32]) {
        // A literal width lets LLVM unroll the shared kernel for 256-bit
        // codes; retain the same dispatch, bounds checks, and tail handling.
        if byte_len == 32 {
            self.score_rows(query, db, 32, out, |index| index);
        } else {
            self.score_rows(query, db, byte_len, out, |index| index);
        }
    }

    /// `out[i]` receives the distance from `query` to row `ids[i]` of `db`.
    ///
    /// Graph routing visits scattered centroid rows; gathering them through one
    /// dispatch keeps the batched kernel usable there.
    pub fn gather_distances(
        self,
        query: &[u8],
        db: &[u8],
        byte_len: usize,
        ids: &[u32],
        out: &mut [u32],
    ) {
        assert_eq!(
            ids.len(),
            out.len(),
            "Hamming gather needs one output slot per row id"
        );
        self.score_rows(query, db, byte_len, out, |index| ids[index] as usize);
    }

    #[inline]
    fn score_rows(
        self,
        query: &[u8],
        db: &[u8],
        byte_len: usize,
        out: &mut [u32],
        index_of: impl Fn(usize) -> usize,
    ) {
        assert_eq!(query.len(), byte_len, "Hamming query byte length mismatch");
        if byte_len == 0 || out.is_empty() {
            return;
        }
        let row = |index: usize| -> &[u8] {
            let start = index * byte_len;
            &db[start..start + byte_len]
        };
        let kernel = self.for_byte_len(byte_len);
        macro_rules! score_with {
            ($one:expr, $four:expr) => {{
                let mut i = 0;
                while i + HAMMING_ROWS_PER_KERNEL <= out.len() {
                    let quad = [
                        row(index_of(i)),
                        row(index_of(i + 1)),
                        row(index_of(i + 2)),
                        row(index_of(i + 3)),
                    ];
                    out[i..i + HAMMING_ROWS_PER_KERNEL].copy_from_slice(&$four(query, quad));
                    i += HAMMING_ROWS_PER_KERNEL;
                }
                while i < out.len() {
                    out[i] = $one(query, row(index_of(i)));
                    i += 1;
                }
            }};
        }
        match kernel {
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => score_with!(
                |query, row| unsafe { hamming_distance_avx512(query, row) },
                |query, rows| unsafe { hamming_distance_x4_avx512(query, rows) }
            ),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => score_with!(
                |query, row| unsafe { avx2::hamming_distance(query, row) },
                |query, rows| unsafe { avx2::hamming_distance_x4(query, rows) }
            ),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => score_with!(
                |query, row| unsafe { neon::hamming_distance(query, row) },
                |query, rows| unsafe { neon::hamming_distance_x4(query, rows) }
            ),
            Self::Scalar => score_with!(hamming_distance_scalar, hamming_distance_x4_scalar),
        }
    }
}

/// Compute Hamming distance between two packed-bit vectors.
/// Returns the number of differing bits.
///
/// Uses NEON on aarch64 and VPOPCNTDQ/AVX2 on x86_64, with a scalar fallback.
/// Loops over many pairs should resolve a [`HammingKernel`] once instead of
/// paying feature detection here per pair.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "Hamming vector byte length mismatch");
    HammingKernel::resolve().distance(a, b)
}

/// Scalar Hamming distance using u64 chunks + count_ones().
/// On x86_64, count_ones() compiles to POPCNT when target-cpu supports it.
#[inline]
fn hamming_distance_scalar(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;
    let mut total = 0u32;

    for i in 0..chunks {
        let off = i * 8;
        let va = unsafe { std::ptr::read_unaligned(a.as_ptr().add(off) as *const u64) };
        let vb = unsafe { std::ptr::read_unaligned(b.as_ptr().add(off) as *const u64) };
        total += (va ^ vb).count_ones();
    }

    let base = chunks * 8;
    for i in 0..remainder {
        total += (a[base + i] ^ b[base + i]).count_ones();
    }

    total
}

/// Batch Hamming scoring: compute similarity scores for multiple binary vectors.
///
/// `query` and each vector in `db` are packed-bit vectors of `byte_len` bytes each.
/// `dim_bits` is the number of bits (dimensions) for normalization.
/// Score = 1.0 - hamming_distance / dim_bits (range [0.0, 1.0]).
pub fn batch_hamming_scores(
    query: &[u8],
    db: &[u8],
    byte_len: usize,
    dim_bits: usize,
    scores: &mut [f32],
) {
    let n = scores.len();
    let required = n
        .checked_mul(byte_len)
        .expect("Hamming batch byte length overflow");
    assert_eq!(query.len(), byte_len, "Hamming query byte length mismatch");
    assert!(
        db.len() >= required,
        "Hamming batch is truncated: need {required} bytes, got {}",
        db.len()
    );

    if byte_len == 0 || n == 0 || dim_bits == 0 {
        return;
    }

    scores_from_hamming(
        HammingKernel::resolve(),
        query,
        db,
        byte_len,
        dim_bits,
        scores,
    );
}

/// Batch Hamming scoring with a caller-resolved kernel.
///
/// Scans that already hold a [`HammingKernel`] (leaf scanning, Lloyd
/// assignment) use this to keep feature detection out of the loop entirely.
pub fn scores_from_hamming(
    kernel: HammingKernel,
    query: &[u8],
    db: &[u8],
    byte_len: usize,
    dim_bits: usize,
    scores: &mut [f32],
) {
    if byte_len == 0 || scores.is_empty() || dim_bits == 0 {
        return;
    }
    let inv_dim = 1.0 / dim_bits as f32;
    // Distances stay integral until the very last step; the stack block keeps
    // the row-batched kernel reachable without a per-scan allocation.
    let mut distances = [0u32; HAMMING_DISTANCE_BLOCK];
    for (block_index, block) in scores.chunks_mut(HAMMING_DISTANCE_BLOCK).enumerate() {
        let rows = &mut distances[..block.len()];
        kernel.distances(
            query,
            &db[block_index * HAMMING_DISTANCE_BLOCK * byte_len..],
            byte_len,
            rows,
        );
        for (score, &distance) in block.iter_mut().zip(rows.iter()) {
            *score = 1.0 - distance as f32 * inv_dim;
        }
    }
}

/// Rows per stack block when converting batched distances into scores.
const HAMMING_DISTANCE_BLOCK: usize = 64;

/// Batch Hamming distances (exact bit counts) for `out.len()` rows of `db`.
///
/// Callers that rank by distance — coarse assignment, routing — avoid the
/// float round-trip entirely.
pub fn batch_hamming_distances(query: &[u8], db: &[u8], byte_len: usize, out: &mut [u32]) {
    HammingKernel::resolve().distances(query, db, byte_len, out);
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_block_seek_preserves_suffix_lower_bounds_at_unsigned_extremes() {
        for length in 0..=128 {
            for base in [0u32, 1 << 31, u32::MAX - 512] {
                let docs: Vec<_> = (0..length).map(|i| base + i as u32 * 3).collect();
                let targets = [0, base, base + 1, base + 127, base + 383, u32::MAX];
                for from in 0..=length {
                    for target in targets {
                        assert_eq!(
                            super::find_first_ge_block_from(&docs, from, target),
                            from + docs[from..].partition_point(|&doc| doc < target),
                            "length={length} from={from} target={target}"
                        );
                    }
                }
            }
        }
        for from in 0..=128 {
            for target in [0, 7, 8, u32::MAX] {
                let docs = [7; 128];
                assert_eq!(
                    super::find_first_ge_block_from(&docs, from, target),
                    from + docs[from..].partition_point(|&doc| doc < target)
                );
            }
        }
    }

    #[test]
    fn posting_block_intersection_preserves_suffixes_partial_outputs_and_unsigned_ids() {
        for base in [0u32, 1 << 31, u32::MAX - 4096] {
            for trial in 0..32 {
                let all_left: Vec<_> = (0..128).map(|i| base + i * (trial % 7 + 1)).collect();
                let all_right: Vec<_> = (0..128)
                    .map(|i| base + i * (trial % 11 + 1) + trial % 3)
                    .collect();
                for len_a in [0, 1, 7, 8, 9, 127, 128] {
                    for len_b in [0, 1, 7, 8, 9, 127, 128] {
                        let left = &all_left[..len_a];
                        let right = &all_right[..len_b];
                        for (from_a, from_b) in [(0, 0), (len_a / 2, len_b / 3), (len_a, len_b)] {
                            let expected: Vec<_> = left[from_a..]
                                .iter()
                                .copied()
                                .filter(|value| right[from_b..].binary_search(value).is_ok())
                                .collect();
                            for limit in [1, 7, 128] {
                                let (mut a, mut b) = (from_a, from_b);
                                let mut actual = Vec::new();
                                let mut pairs = [(0u8, 0u8); 128];
                                while a < len_a && b < len_b {
                                    let previous = (a, b);
                                    let count = super::intersect_posting_blocks(
                                        left,
                                        &mut a,
                                        right,
                                        &mut b,
                                        &mut pairs[..limit],
                                    );
                                    assert!(a > previous.0 || b > previous.1);
                                    for &(l, r) in &pairs[..count] {
                                        assert_eq!(left[l as usize], right[r as usize]);
                                        actual.push(left[l as usize]);
                                    }
                                }
                                assert_eq!(actual, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The single scalar fused kernel (used for every SIMD tail and as the
    /// non-SIMD fallback) must match a naive reference for every count that
    /// crosses the 4/8/16-lane group boundaries, both widths and both offsets.
    #[test]
    fn scalar_delta_decode_with_offset_matches_naive_reference_for_all_counts() {
        fn naive<const OFFSET: u32>(deltas: &[u32], first: u32) -> Vec<u32> {
            let mut out = vec![first];
            for &d in deltas {
                out.push(out.last().unwrap().wrapping_add(d).wrapping_add(OFFSET));
            }
            out
        }
        for count in 0..=257usize {
            let deltas: Vec<u32> = (0..count.saturating_sub(1))
                .map(|i| [0, 1, 255, 65535, 42, 17][i % 6])
                .collect();
            for first in [0u32, 7, u32::MAX - 3] {
                for bytes in [1usize, 2] {
                    let mask = if bytes == 1 { 0xFF } else { 0xFFFF };
                    let masked: Vec<u32> = deltas.iter().map(|d| d & mask).collect();
                    let mut input = Vec::new();
                    for d in &masked {
                        input.extend_from_slice(&d.to_le_bytes()[..bytes]);
                    }
                    let (expected0, expected1) = if count == 0 {
                        (Vec::new(), Vec::new())
                    } else {
                        (naive::<0>(&masked, first), naive::<1>(&masked, first))
                    };
                    let mut out0 = vec![0xDEAD_BEEF; count + 2];
                    let mut out1 = vec![0xDEAD_BEEF; count + 2];
                    if bytes == 1 {
                        super::scalar::delta_decode_with_offset::<0, 1>(
                            &input,
                            &mut out0[..count],
                            first,
                            count,
                        );
                        super::scalar::delta_decode_with_offset::<1, 1>(
                            &input,
                            &mut out1[..count],
                            first,
                            count,
                        );
                    } else {
                        super::scalar::delta_decode_with_offset::<0, 2>(
                            &input,
                            &mut out0[..count],
                            first,
                            count,
                        );
                        super::scalar::delta_decode_with_offset::<1, 2>(
                            &input,
                            &mut out1[..count],
                            first,
                            count,
                        );
                    }
                    assert_eq!(
                        &out0[..count],
                        expected0,
                        "offset 0 bytes={bytes} count={count}"
                    );
                    assert_eq!(
                        &out1[..count],
                        expected1,
                        "offset 1 bytes={bytes} count={count}"
                    );
                    assert_eq!(&out0[count..], &[0xDEAD_BEEF; 2]);
                    assert_eq!(&out1[count..], &[0xDEAD_BEEF; 2]);
                    // The public dispatchers (SIMD where available, scalar
                    // otherwise) must agree with the scalar definition.
                    if bytes == 1 {
                        let mut simd_out = vec![0; count];
                        super::unpack_8bit_delta_decode_with_offset::<0>(
                            &input,
                            &mut simd_out,
                            first,
                            count,
                        );
                        assert_eq!(simd_out, expected0, "dispatch 8-bit count={count}");
                    } else {
                        let mut simd_out = vec![0; count];
                        super::unpack_16bit_delta_decode_with_offset::<1>(
                            &input,
                            &mut simd_out,
                            first,
                            count,
                        );
                        assert_eq!(simd_out, expected1, "dispatch 16-bit count={count}");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "fused delta decode: input holds")]
    fn fused_delta_decode_rejects_short_input_before_touching_the_kernels() {
        let input = [1u8; 3];
        let mut output = [0u32; 8];
        super::unpack_8bit_delta_decode(&input, &mut output, 0, 8);
    }

    #[test]
    #[should_panic(expected = "fused delta decode: output holds")]
    fn fused_delta_decode_rejects_short_output_before_touching_the_kernels() {
        let input = [1u8; 16];
        let mut output = [0u32; 4];
        super::unpack_16bit_delta_decode(&input, &mut output, 0, 8);
    }

    #[test]
    fn rounded_bit_width_try_from_u8_rejects_unrounded_widths() {
        use super::RoundedBitWidth;
        assert_eq!(RoundedBitWidth::try_from_u8(0), Some(RoundedBitWidth::Zero));
        assert_eq!(
            RoundedBitWidth::try_from_u8(8),
            Some(RoundedBitWidth::Bits8)
        );
        assert_eq!(
            RoundedBitWidth::try_from_u8(16),
            Some(RoundedBitWidth::Bits16)
        );
        assert_eq!(
            RoundedBitWidth::try_from_u8(32),
            Some(RoundedBitWidth::Bits32)
        );
        for bad in (1..=255u8).filter(|b| ![8, 16, 32].contains(b)) {
            assert_eq!(RoundedBitWidth::try_from_u8(bad), None, "width {bad}");
        }
    }

    #[test]
    fn raw_rounded_gaps_and_legacy_gaps_agree_with_scalar_at_all_tails() {
        use super::*;
        for (width, mask) in [
            (RoundedBitWidth::Zero, 0u32),
            (RoundedBitWidth::Bits8, 255),
            (RoundedBitWidth::Bits16, 65535),
            (RoundedBitWidth::Bits32, u32::MAX),
        ] {
            for count in 0..=257usize {
                let first = u32::MAX - 17;
                let gaps: Vec<_> = (0..count.saturating_sub(1))
                    .map(|i| [0, 1, mask, mask / 2][i % 4] & mask)
                    .collect();
                let mut input = Vec::new();
                for &gap in &gaps {
                    let bytes = gap.to_le_bytes();
                    input.extend_from_slice(&bytes[..width.bytes_per_value()]);
                }
                let mut expected = Vec::with_capacity(count);
                if count > 0 {
                    expected.push(first);
                    for &gap in &gaps {
                        expected.push(expected.last().unwrap().wrapping_add(gap));
                    }
                }
                let mut actual = vec![0xDEADBEEF; count + 4];
                unpack_rounded_raw_delta_decode(&input, width, &mut actual[..count], first, count);
                assert_eq!(&actual[..count], expected, "width={width:?} count={count}");
                assert_eq!(&actual[count..], &[0xDEADBEEF; 4]);
                let mut legacy = vec![0; count];
                unpack_rounded_delta_decode(&input, width, &mut legacy, first, count);
                let biased: Vec<_> = expected
                    .iter()
                    .enumerate()
                    .map(|(i, &doc)| doc.wrapping_add(i as u32))
                    .collect();
                assert_eq!(legacy, biased, "legacy width={width:?} count={count}");
                if count == 0 {
                    continue;
                }
                // Exercise each available ISA, including SSE on AVX2 hosts.
                #[cfg(target_arch = "x86_64")]
                for (available, kernels) in [
                    (
                        sse::is_available(),
                        [
                            sse::unpack_8bit_delta_decode_with_offset::<0>
                                as unsafe fn(&[u8], &mut [u32], u32, usize),
                            sse::unpack_16bit_delta_decode_with_offset::<0>,
                        ],
                    ),
                    (
                        avx2::is_available(),
                        [
                            avx2::unpack_8bit_delta_decode_with_offset::<0>
                                as unsafe fn(&[u8], &mut [u32], u32, usize),
                            avx2::unpack_16bit_delta_decode_with_offset::<0>,
                        ],
                    ),
                ] {
                    let index = match width {
                        RoundedBitWidth::Bits8 => Some(0),
                        RoundedBitWidth::Bits16 => Some(1),
                        _ => None,
                    };
                    if available && let Some(index) = index {
                        let mut decoded = vec![0; count];
                        unsafe {
                            kernels[index](&input, &mut decoded, first, count);
                        }
                        assert_eq!(decoded, expected);
                    }
                }
            }
        }
    }
    use super::*;

    #[test]
    fn vector_simd_boundaries_reject_dimension_mismatches() {
        let vectors = vec![1.0f32; 6];
        let raw_f16 = vec![0u8; 12];
        let raw_u8 = vec![0u8; 6];
        let mut scores = vec![0.0f32; 2];

        for invalid_query in [vec![1.0, 2.0], vec![1.0, 2.0, 3.0, 4.0]] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    batch_cosine_scores(&invalid_query, &vectors, 3, &mut scores)
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    batch_dot_scores_f16(&invalid_query, &raw_f16, 3, &mut scores)
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    batch_cosine_scores_u8(&invalid_query, &raw_u8, 3, &mut scores)
                }))
                .is_err()
            );
        }
    }

    #[test]
    fn vector_simd_boundaries_reject_truncated_storage() {
        let query = [1.0f32, 2.0, 3.0];
        let mut scores = [0.0f32; 2];

        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                batch_dot_scores(&query, &[0.0; 5], 3, &mut scores)
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                batch_cosine_scores_f16(&query, &[0u8; 11], 3, &mut scores)
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dot_product_f32(&query, &query, 4)
            }))
            .is_err()
        );
    }

    #[test]
    fn test_unpack_8bit() {
        let input: Vec<u8> = (0..128).collect();
        let mut output = vec![0u32; 128];
        unpack_8bit(&input, &mut output, 128);

        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, i as u32);
        }
    }

    #[test]
    fn test_unpack_16bit() {
        let mut input = vec![0u8; 256];
        for i in 0..128 {
            let val = (i * 100) as u16;
            input[i * 2] = val as u8;
            input[i * 2 + 1] = (val >> 8) as u8;
        }

        let mut output = vec![0u32; 128];
        unpack_16bit(&input, &mut output, 128);

        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, (i * 100) as u32);
        }
    }

    #[test]
    fn test_unpack_32bit() {
        let mut input = vec![0u8; 512];
        for i in 0..128 {
            let val = (i * 1000) as u32;
            let bytes = val.to_le_bytes();
            input[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }

        let mut output = vec![0u32; 128];
        unpack_32bit(&input, &mut output, 128);

        for (i, &v) in output.iter().enumerate() {
            assert_eq!(v, (i * 1000) as u32);
        }
    }

    #[test]
    fn test_delta_decode() {
        // doc_ids: [10, 15, 20, 30, 50]
        // gaps: [5, 5, 10, 20]
        // deltas (gap-1): [4, 4, 9, 19]
        let deltas = vec![4u32, 4, 9, 19];
        let mut output = vec![0u32; 5];

        delta_decode(&mut output, &deltas, 10, 5);

        assert_eq!(output, vec![10, 15, 20, 30, 50]);
    }

    #[test]
    fn test_add_one() {
        let mut values = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
        add_one(&mut values, 8);

        assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_bits_needed() {
        assert_eq!(bits_needed(0), 0);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 2);
        assert_eq!(bits_needed(3), 2);
        assert_eq!(bits_needed(4), 3);
        assert_eq!(bits_needed(255), 8);
        assert_eq!(bits_needed(256), 9);
        assert_eq!(bits_needed(u32::MAX), 32);
    }

    #[test]
    fn test_unpack_8bit_delta_decode() {
        // doc_ids: [10, 15, 20, 30, 50]
        // gaps: [5, 5, 10, 20]
        // deltas (gap-1): [4, 4, 9, 19] stored as u8
        let input: Vec<u8> = vec![4, 4, 9, 19];
        let mut output = vec![0u32; 5];

        unpack_8bit_delta_decode(&input, &mut output, 10, 5);

        assert_eq!(output, vec![10, 15, 20, 30, 50]);
    }

    #[test]
    fn test_unpack_16bit_delta_decode() {
        // doc_ids: [100, 600, 1100, 2100, 4100]
        // gaps: [500, 500, 1000, 2000]
        // deltas (gap-1): [499, 499, 999, 1999] stored as u16
        let mut input = vec![0u8; 8];
        for (i, &delta) in [499u16, 499, 999, 1999].iter().enumerate() {
            input[i * 2] = delta as u8;
            input[i * 2 + 1] = (delta >> 8) as u8;
        }
        let mut output = vec![0u32; 5];

        unpack_16bit_delta_decode(&input, &mut output, 100, 5);

        assert_eq!(output, vec![100, 600, 1100, 2100, 4100]);
    }

    #[test]
    fn test_fused_vs_separate_8bit() {
        // Test that fused and separate operations produce the same result
        let input: Vec<u8> = (0..127).collect();
        let first_value = 1000u32;
        let count = 128;

        // Separate: unpack then delta_decode
        let mut unpacked = vec![0u32; 128];
        unpack_8bit(&input, &mut unpacked, 127);
        let mut separate_output = vec![0u32; 128];
        delta_decode(&mut separate_output, &unpacked, first_value, count);

        // Fused
        let mut fused_output = vec![0u32; 128];
        unpack_8bit_delta_decode(&input, &mut fused_output, first_value, count);

        assert_eq!(separate_output, fused_output);
    }

    #[test]
    fn test_round_bit_width() {
        assert_eq!(round_bit_width(0), 0);
        assert_eq!(round_bit_width(1), 8);
        assert_eq!(round_bit_width(5), 8);
        assert_eq!(round_bit_width(8), 8);
        assert_eq!(round_bit_width(9), 16);
        assert_eq!(round_bit_width(12), 16);
        assert_eq!(round_bit_width(16), 16);
        assert_eq!(round_bit_width(17), 32);
        assert_eq!(round_bit_width(24), 32);
        assert_eq!(round_bit_width(32), 32);
    }

    #[test]
    fn test_rounded_bitwidth_from_exact() {
        assert_eq!(RoundedBitWidth::from_exact(0), RoundedBitWidth::Zero);
        assert_eq!(RoundedBitWidth::from_exact(1), RoundedBitWidth::Bits8);
        assert_eq!(RoundedBitWidth::from_exact(8), RoundedBitWidth::Bits8);
        assert_eq!(RoundedBitWidth::from_exact(9), RoundedBitWidth::Bits16);
        assert_eq!(RoundedBitWidth::from_exact(16), RoundedBitWidth::Bits16);
        assert_eq!(RoundedBitWidth::from_exact(17), RoundedBitWidth::Bits32);
        assert_eq!(RoundedBitWidth::from_exact(32), RoundedBitWidth::Bits32);
    }

    #[test]
    fn test_pack_unpack_rounded_8bit() {
        let values: Vec<u32> = (0..128).map(|i| i % 256).collect();
        let mut packed = vec![0u8; 128];

        let bytes_written = pack_rounded(&values, RoundedBitWidth::Bits8, &mut packed);
        assert_eq!(bytes_written, 128);

        let mut unpacked = vec![0u32; 128];
        unpack_rounded(&packed, RoundedBitWidth::Bits8, &mut unpacked, 128);

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_rounded_16bit() {
        let values: Vec<u32> = (0..128).map(|i| i * 100).collect();
        let mut packed = vec![0u8; 256];

        let bytes_written = pack_rounded(&values, RoundedBitWidth::Bits16, &mut packed);
        assert_eq!(bytes_written, 256);

        let mut unpacked = vec![0u32; 128];
        unpack_rounded(&packed, RoundedBitWidth::Bits16, &mut unpacked, 128);

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_rounded_32bit() {
        let values: Vec<u32> = (0..128).map(|i| i * 100000).collect();
        let mut packed = vec![0u8; 512];

        let bytes_written = pack_rounded(&values, RoundedBitWidth::Bits32, &mut packed);
        assert_eq!(bytes_written, 512);

        let mut unpacked = vec![0u32; 128];
        unpack_rounded(&packed, RoundedBitWidth::Bits32, &mut unpacked, 128);

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_unpack_rounded_delta_decode() {
        // Test 8-bit rounded delta decode
        // doc_ids: [10, 15, 20, 30, 50]
        // gaps: [5, 5, 10, 20]
        // deltas (gap-1): [4, 4, 9, 19] stored as u8
        let input: Vec<u8> = vec![4, 4, 9, 19];
        let mut output = vec![0u32; 5];

        unpack_rounded_delta_decode(&input, RoundedBitWidth::Bits8, &mut output, 10, 5);

        assert_eq!(output, vec![10, 15, 20, 30, 50]);
    }

    #[test]
    fn test_unpack_rounded_delta_decode_zero() {
        // All zeros means gaps of 1 (consecutive doc IDs)
        let input: Vec<u8> = vec![];
        let mut output = vec![0u32; 5];

        unpack_rounded_delta_decode(&input, RoundedBitWidth::Zero, &mut output, 100, 5);

        assert_eq!(output, vec![100, 101, 102, 103, 104]);
    }

    // ========================================================================
    // Sparse Vector SIMD Tests
    // ========================================================================

    #[test]
    fn test_dequantize_uint8() {
        let input: Vec<u8> = vec![0, 128, 255, 64, 192];
        let mut output = vec![0.0f32; 5];
        let scale = 0.1;
        let min_val = 1.0;

        dequantize_uint8(&input, &mut output, scale, min_val, 5);

        // Expected: input[i] * scale + min_val
        assert!((output[0] - 1.0).abs() < 1e-6); // 0 * 0.1 + 1.0 = 1.0
        assert!((output[1] - 13.8).abs() < 1e-6); // 128 * 0.1 + 1.0 = 13.8
        assert!((output[2] - 26.5).abs() < 1e-6); // 255 * 0.1 + 1.0 = 26.5
        assert!((output[3] - 7.4).abs() < 1e-6); // 64 * 0.1 + 1.0 = 7.4
        assert!((output[4] - 20.2).abs() < 1e-6); // 192 * 0.1 + 1.0 = 20.2
    }

    #[test]
    fn test_dequantize_uint8_large() {
        // Test with 128 values (full SIMD block)
        let input: Vec<u8> = (0..128).collect();
        let mut output = vec![0.0f32; 128];
        let scale = 2.0;
        let min_val = -10.0;

        dequantize_uint8(&input, &mut output, scale, min_val, 128);

        for (i, &out) in output.iter().enumerate().take(128) {
            let expected = i as f32 * scale + min_val;
            assert!(
                (out - expected).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i,
                expected,
                out
            );
        }
    }

    #[test]
    fn test_dot_product_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];

        let result = dot_product_f32(&a, &b, 5);

        // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_f32_large() {
        // Test with 128 values
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let result = dot_product_f32(&a, &b, 128);

        // Compute expected
        let expected: f32 = (0..128).map(|i| (i as f32) * ((i + 1) as f32)).sum();
        assert!(
            (result - expected).abs() < 1e-3,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_fused_dot_norm() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (dot, norm_b) = fused_dot_norm(&a, &b, a.len());

        let expected_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let expected_norm: f32 = b.iter().map(|x| x * x).sum();
        assert!(
            (dot - expected_dot).abs() < 1e-5,
            "dot: expected {}, got {}",
            expected_dot,
            dot
        );
        assert!(
            (norm_b - expected_norm).abs() < 1e-5,
            "norm: expected {}, got {}",
            expected_norm,
            norm_b
        );
    }

    #[test]
    fn test_fused_dot_norm_large() {
        let a: Vec<f32> = (0..768).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32) * 0.02 + 0.5).collect();
        let (dot, norm_b) = fused_dot_norm(&a, &b, a.len());

        let expected_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let expected_norm: f32 = b.iter().map(|x| x * x).sum();
        assert!(
            (dot - expected_dot).abs() < 1.0,
            "dot: expected {}, got {}",
            expected_dot,
            dot
        );
        assert!(
            (norm_b - expected_norm).abs() < 1.0,
            "norm: expected {}, got {}",
            expected_norm,
            norm_b
        );
    }

    #[test]
    fn test_batch_cosine_scores() {
        // 4 vectors of dim 3
        let query = vec![1.0f32, 0.0, 0.0];
        let vectors = vec![
            1.0, 0.0, 0.0, // identical to query
            0.0, 1.0, 0.0, // orthogonal
            -1.0, 0.0, 0.0, // opposite
            0.5, 0.5, 0.0, // 45 degrees
        ];
        let mut scores = vec![0f32; 4];
        batch_cosine_scores(&query, &vectors, 3, &mut scores);

        assert!((scores[0] - 1.0).abs() < 1e-5, "identical: {}", scores[0]);
        assert!(scores[1].abs() < 1e-5, "orthogonal: {}", scores[1]);
        assert!((scores[2] - (-1.0)).abs() < 1e-5, "opposite: {}", scores[2]);
        let expected_45 = 0.5f32 / (0.5f32.powi(2) + 0.5f32.powi(2)).sqrt();
        assert!(
            (scores[3] - expected_45).abs() < 1e-5,
            "45deg: expected {}, got {}",
            expected_45,
            scores[3]
        );
    }

    #[test]
    fn test_batch_cosine_scores_matches_individual() {
        let query: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let n = 50;
        let dim = 128;
        let vectors: Vec<f32> = (0..n * dim).map(|i| ((i * 7 + 3) as f32) * 0.01).collect();

        let mut batch_scores = vec![0f32; n];
        batch_cosine_scores(&query, &vectors, dim, &mut batch_scores);

        for i in 0..n {
            let vec_i = &vectors[i * dim..(i + 1) * dim];
            let individual = cosine_similarity(&query, vec_i);
            assert!(
                (batch_scores[i] - individual).abs() < 1e-5,
                "vec {}: batch={}, individual={}",
                i,
                batch_scores[i],
                individual
            );
        }
    }

    #[test]
    fn test_batch_cosine_scores_empty() {
        let query = vec![1.0f32, 2.0, 3.0];
        let vectors: Vec<f32> = vec![];
        let mut scores: Vec<f32> = vec![];
        batch_cosine_scores(&query, &vectors, 3, &mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_batch_cosine_scores_zero_query() {
        let query = vec![0.0f32, 0.0, 0.0];
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut scores = vec![0f32; 2];
        batch_cosine_scores(&query, &vectors, 3, &mut scores);
        assert_eq!(scores[0], 0.0);
        assert_eq!(scores[1], 0.0);
    }

    // ================================================================
    // f16 conversion tests
    // ================================================================

    #[test]
    fn test_f16_roundtrip_normal() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.5, 0.333, 65504.0] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            let err = (back - v).abs() / v.abs().max(1e-6);
            assert!(
                err < 0.002,
                "f16 roundtrip {v} → {h:#06x} → {back}, rel err {err}"
            );
        }
    }

    #[test]
    fn test_f16_special() {
        // Zero
        assert_eq!(f16_to_f32(f32_to_f16(0.0)), 0.0);
        // Negative zero
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        // Infinity
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        // NaN
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn test_f16_embedding_range() {
        // Typical embedding values in [-1, 1]
        let values: Vec<f32> = (-100..=100).map(|i| i as f32 / 100.0).collect();
        for &v in &values {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() < 0.001, "f16 error for {v}: got {back}");
        }
    }

    // ================================================================
    // u8 conversion tests
    // ================================================================

    #[test]
    fn test_u8_roundtrip() {
        // Boundary values
        assert_eq!(f32_to_u8_saturating(-1.0), 0);
        assert_eq!(f32_to_u8_saturating(1.0), 255);
        assert_eq!(f32_to_u8_saturating(0.0), 127); // ~127.5 truncated

        // Saturation
        assert_eq!(f32_to_u8_saturating(-2.0), 0);
        assert_eq!(f32_to_u8_saturating(2.0), 255);
    }

    #[test]
    fn test_u8_dequantize() {
        assert!((u8_to_f32(0) - (-1.0)).abs() < 0.01);
        assert!((u8_to_f32(255) - 1.0).abs() < 0.01);
        assert!((u8_to_f32(127) - 0.0).abs() < 0.01);
    }

    // ================================================================
    // Batch scoring tests for quantized vectors
    // ================================================================

    #[test]
    fn test_batch_cosine_scores_f16() {
        let query = vec![0.6f32, 0.8, 0.0, 0.0];
        let dim = 4;
        let vecs_f32 = vec![
            0.6f32, 0.8, 0.0, 0.0, // identical to query
            0.0, 0.0, 0.6, 0.8, // orthogonal
        ];

        // Quantize to f16
        let mut f16_buf = vec![0u16; 8];
        batch_f32_to_f16(&vecs_f32, &mut f16_buf);
        let raw: &[u8] =
            unsafe { std::slice::from_raw_parts(f16_buf.as_ptr() as *const u8, f16_buf.len() * 2) };

        let mut scores = vec![0f32; 2];
        batch_cosine_scores_f16(&query, raw, dim, &mut scores);

        assert!(
            (scores[0] - 1.0).abs() < 0.01,
            "identical vectors: {}",
            scores[0]
        );
        assert!(scores[1].abs() < 0.01, "orthogonal vectors: {}", scores[1]);
    }

    #[test]
    fn test_batch_cosine_scores_u8() {
        let query = vec![0.6f32, 0.8, 0.0, 0.0];
        let dim = 4;
        let vecs_f32 = vec![
            0.6f32, 0.8, 0.0, 0.0, // ~identical to query
            -0.6, -0.8, 0.0, 0.0, // opposite
        ];

        // Quantize to u8
        let mut u8_buf = vec![0u8; 8];
        batch_f32_to_u8(&vecs_f32, &mut u8_buf);

        let mut scores = vec![0f32; 2];
        batch_cosine_scores_u8(&query, &u8_buf, dim, &mut scores);

        assert!(scores[0] > 0.95, "similar vectors: {}", scores[0]);
        assert!(scores[1] < -0.95, "opposite vectors: {}", scores[1]);
    }

    #[test]
    fn test_batch_cosine_scores_f16_large_dim() {
        // Test with typical embedding dimension
        let dim = 768;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32) - 0.5).collect();
        let vec2: Vec<f32> = query.iter().map(|x| x * 0.9 + 0.01).collect();

        let mut all_vecs = query.clone();
        all_vecs.extend_from_slice(&vec2);

        let mut f16_buf = vec![0u16; all_vecs.len()];
        batch_f32_to_f16(&all_vecs, &mut f16_buf);
        let raw: &[u8] =
            unsafe { std::slice::from_raw_parts(f16_buf.as_ptr() as *const u8, f16_buf.len() * 2) };

        let mut scores = vec![0f32; 2];
        batch_cosine_scores_f16(&query, raw, dim, &mut scores);

        // Self-similarity should be ~1.0
        assert!((scores[0] - 1.0).abs() < 0.01, "self-sim: {}", scores[0]);
        // High similarity with scaled version
        assert!(scores[1] > 0.99, "scaled-sim: {}", scores[1]);
    }

    // ================================================================
    // Hamming distance tests
    // ================================================================

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0xAA; 64];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let a = vec![0xFF; 32];
        let b = vec![0x00; 32];
        assert_eq!(hamming_distance(&a, &b), 256);
    }

    #[test]
    fn test_hamming_distance_known() {
        // Single byte: 0b10101010 vs 0b01010101 = 8 bits differ
        let a = vec![0xAA];
        let b = vec![0x55];
        assert_eq!(hamming_distance(&a, &b), 8);

        // Two bytes
        let a = vec![0xFF, 0x00];
        let b = vec![0x00, 0x00];
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn test_hamming_distance_single_bit() {
        let a = vec![0x00; 16];
        let mut b = vec![0x00; 16];
        b[7] = 0x01; // flip one bit
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_hamming_distance_empty() {
        let a: Vec<u8> = vec![];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_distance_remainder_path() {
        // 17 bytes: not aligned to 16 (NEON) or 32 (AVX2)
        let a = vec![0xFF; 17];
        let b = vec![0x00; 17];
        assert_eq!(hamming_distance(&a, &b), 136); // 17 * 8

        // 33 bytes: tests 32-byte chunk + 1 remainder for AVX2
        let a = vec![0xFF; 33];
        let b = vec![0x00; 33];
        assert_eq!(hamming_distance(&a, &b), 264); // 33 * 8
    }

    #[test]
    fn test_hamming_distance_large() {
        // 4096 bytes = 32768 bits, all differing
        let a = vec![0xFF; 4096];
        let b = vec![0x00; 4096];
        assert_eq!(hamming_distance(&a, &b), 32768);
    }

    #[test]
    fn test_hamming_distance_scalar_matches() {
        // Verify SIMD path matches scalar for various sizes
        for size in [1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 128, 255, 256] {
            let a: Vec<u8> = (0..size).map(|i| (i * 37 + 13) as u8).collect();
            let b: Vec<u8> = (0..size).map(|i| (i * 53 + 7) as u8).collect();
            let expected = hamming_distance_scalar(&a, &b);
            let got = hamming_distance(&a, &b);
            assert_eq!(got, expected, "mismatch at size {size}");
        }
    }

    // ================================================================
    // Batch Hamming scoring tests
    // ================================================================

    #[test]
    fn test_batch_hamming_scores_identical() {
        let query = vec![0xAA; 16];
        let db = vec![0xAA; 16]; // one vector, identical
        let mut scores = vec![0f32; 1];
        batch_hamming_scores(&query, &db, 16, 128, &mut scores);
        assert!((scores[0] - 1.0).abs() < 1e-6, "identical: {}", scores[0]);
    }

    #[test]
    fn test_batch_hamming_scores_opposite() {
        let query = vec![0xFF; 16];
        let db = vec![0x00; 16];
        let mut scores = vec![0f32; 1];
        batch_hamming_scores(&query, &db, 16, 128, &mut scores);
        assert!((scores[0] - 0.0).abs() < 1e-6, "opposite: {}", scores[0]);
    }

    #[test]
    fn test_batch_hamming_scores_multiple() {
        let byte_len = 8;
        let dim_bits = 64;
        let query = vec![0xFF; byte_len];
        let mut db = Vec::new();
        db.extend_from_slice(&vec![0xFF; byte_len]); // identical → 1.0
        db.extend_from_slice(&vec![0x00; byte_len]); // opposite → 0.0
        db.extend_from_slice(&vec![0x0F; byte_len]); // half bits differ → 0.5

        let mut scores = vec![0f32; 3];
        batch_hamming_scores(&query, &db, byte_len, dim_bits, &mut scores);

        assert!((scores[0] - 1.0).abs() < 1e-6, "identical: {}", scores[0]);
        assert!((scores[1] - 0.0).abs() < 1e-6, "opposite: {}", scores[1]);
        assert!((scores[2] - 0.5).abs() < 1e-6, "half: {}", scores[2]);
    }

    #[test]
    fn test_batch_hamming_scores_empty() {
        let query = vec![0xFF; 8];
        let db: Vec<u8> = vec![];
        let mut scores: Vec<f32> = vec![];
        batch_hamming_scores(&query, &db, 8, 64, &mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_batch_hamming_scores_zero_byte_len() {
        let query: Vec<u8> = vec![];
        let db: Vec<u8> = vec![];
        let mut scores = vec![0f32; 1];
        batch_hamming_scores(&query, &db, 0, 0, &mut scores);
        // Should return early without modifying scores
        assert_eq!(scores[0], 0.0);
    }

    // ================================================================
    // Resolved-kernel and row-batched Hamming tests
    // ================================================================

    fn hamming_matrix(rows: usize, byte_len: usize) -> (Vec<u8>, Vec<u8>) {
        let query: Vec<u8> = (0..byte_len).map(|i| (i * 31 + 5) as u8).collect();
        let db: Vec<u8> = (0..rows * byte_len)
            .map(|i| (i * 97 + i / byte_len * 11 + 3) as u8)
            .collect();
        (query, db)
    }

    /// The row-batched kernels share query loads and accumulators across four
    /// rows; every width must still agree bit-for-bit with the scalar loop.
    #[test]
    fn batched_hamming_distances_match_scalar_for_every_row_count() {
        let kernels = [HammingKernel::resolve(), HammingKernel::Scalar];
        // Cover both multiples of the quad width and every tail remainder, and
        // byte lengths that exercise 16/32/64-byte chunking plus odd tails.
        for byte_len in [1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 128, 320] {
            for rows in [1, 2, 3, 4, 5, 7, 8, 9, 64, 70] {
                let (query, db) = hamming_matrix(rows, byte_len);
                let mut got = vec![0u32; rows];
                for kernel in kernels {
                    kernel.distances(&query, &db, byte_len, &mut got);
                    for (row, &distance) in got.iter().enumerate() {
                        let expected = hamming_distance_scalar(
                            &query,
                            &db[row * byte_len..(row + 1) * byte_len],
                        );
                        assert_eq!(
                            distance, expected,
                            "{kernel:?}: row {row} of {rows} at byte_len {byte_len}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gathered_hamming_distances_follow_row_ids() {
        let kernel = HammingKernel::resolve();
        let byte_len = 320;
        let rows = 37;
        let (query, db) = hamming_matrix(rows, byte_len);
        // Scattered, repeated and reversed ids: routing visits rows in graph
        // order, not storage order.
        let ids: Vec<u32> = [36, 0, 17, 17, 5, 31, 2, 9, 9, 36, 1].into_iter().collect();
        let mut got = vec![0u32; ids.len()];
        kernel.gather_distances(&query, &db, byte_len, &ids, &mut got);
        for (slot, &id) in ids.iter().enumerate() {
            let start = id as usize * byte_len;
            let expected = hamming_distance_scalar(&query, &db[start..start + byte_len]);
            assert_eq!(got[slot], expected, "slot {slot} for row {id}");
        }
    }

    #[test]
    fn resolved_kernel_matches_scalar_pairwise() {
        let kernel = HammingKernel::resolve();
        for byte_len in [1, 8, 32, 64, 65, 320, 4096] {
            let (query, db) = hamming_matrix(1, byte_len);
            assert_eq!(
                kernel.distance(&query, &db),
                hamming_distance_scalar(&query, &db),
                "byte_len {byte_len}"
            );
        }
    }

    /// Codes narrower than one SIMD vector must take the `u64::count_ones`
    /// loop; codes at least one vector wide keep the resolved kernel.
    #[test]
    fn hamming_kernel_routes_sub_vector_codes_to_the_scalar_loop() {
        let kernel = HammingKernel::resolve();
        let width = kernel.vector_bytes();
        assert_eq!(HammingKernel::Scalar.for_byte_len(1), HammingKernel::Scalar);
        assert_eq!(kernel.for_byte_len(width - 1), HammingKernel::Scalar);
        assert_eq!(kernel.for_byte_len(width), kernel);
        assert_eq!(kernel.for_byte_len(width * 5 + 3), kernel);
        // The routed kernel is exact at every width around the cut.
        for byte_len in [width - 1, width, width + 1] {
            let (query, db) = hamming_matrix(9, byte_len);
            let mut got = vec![0u32; 9];
            kernel.distances(&query, &db, byte_len, &mut got);
            for (row, &distance) in got.iter().enumerate() {
                assert_eq!(
                    distance,
                    hamming_distance_scalar(&query, &db[row * byte_len..(row + 1) * byte_len]),
                    "row {row} at byte_len {byte_len}"
                );
            }
        }
    }

    #[test]
    fn scores_from_hamming_matches_batch_scores_across_blocks() {
        let kernel = HammingKernel::resolve();
        let byte_len = 320;
        let dim_bits = byte_len * 8;
        // More rows than one stack block so the block seam is covered.
        let rows = HAMMING_DISTANCE_BLOCK * 2 + 3;
        let (query, db) = hamming_matrix(rows, byte_len);
        let mut expected = vec![0f32; rows];
        for (row, score) in expected.iter_mut().enumerate() {
            let distance =
                hamming_distance_scalar(&query, &db[row * byte_len..(row + 1) * byte_len]);
            *score = 1.0 - distance as f32 / dim_bits as f32;
        }
        let mut got = vec![0f32; rows];
        scores_from_hamming(kernel, &query, &db, byte_len, dim_bits, &mut got);
        for (row, (&got, &want)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "row {row}: {got} vs {want}");
        }
        let mut public = vec![0f32; rows];
        batch_hamming_scores(&query, &db, byte_len, dim_bits, &mut public);
        assert_eq!(got, public);
    }
}

// ============================================================================
// SIMD-accelerated linear scan for sorted u32 slices (within-block seek)
// ============================================================================

/// Intersect two strictly increasing decoded posting blocks. Return index pairs
/// in document order and resume positions for the unconsumed suffixes. Each
/// input is at most 128 IDs; no document-space scratch or allocation is needed.
#[inline]
pub(crate) fn intersect_posting_blocks(
    left: &[u32],
    a: &mut usize,
    right: &[u32],
    b: &mut usize,
    pairs: &mut [(u8, u8)],
) -> usize {
    assert!(left.len() <= 128 && right.len() <= 128);
    assert!(*a <= left.len() && *b <= right.len());
    let (mut left_pos, mut right_pos) = (*a, *b);
    let mut count = 0;
    while left_pos < left.len() && right_pos < right.len() && count < pairs.len() {
        let doc = left[left_pos];
        if let Some(group) = right.get(right_pos..right_pos + 8) {
            let group: &[u32; 8] = group.try_into().unwrap();
            if group[7] < doc {
                right_pos += 8;
                continue;
            }
            if doc < group[0] {
                left_pos += find_first_ge_u32(&left[left_pos..], group[0]);
                continue;
            }
            if let Some(lane) = equal_lane_8(group, doc) {
                pairs[count] = (left_pos as u8, (right_pos + lane) as u8);
                count += 1;
                left_pos += 1;
                if count == pairs.len() {
                    right_pos += lane + 1;
                    break;
                }
            } else {
                left_pos += 1;
            }
        } else {
            match doc.cmp(&right[right_pos]) {
                std::cmp::Ordering::Less => left_pos += 1,
                std::cmp::Ordering::Greater => right_pos += 1,
                std::cmp::Ordering::Equal => {
                    pairs[count] = (left_pos as u8, right_pos as u8);
                    count += 1;
                    left_pos += 1;
                    right_pos += 1;
                }
            }
        }
    }
    *a = left_pos;
    *b = right_pos;
    count
}

#[inline]
fn equal_lane_8(values: &[u32; 8], target: u32) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if avx2::is_available() {
        // SAFETY: the fixed input has eight lanes and AVX2 was checked.
        let mask = unsafe { equal_mask_8_avx2(values, target) };
        return (mask != 0).then(|| mask.trailing_zeros() as usize);
    }
    #[cfg(target_arch = "aarch64")]
    if neon::is_available() {
        // SAFETY: the fixed input has eight lanes and NEON was checked.
        let mask = unsafe { equal_mask_8_neon(values, target) };
        return (mask != 0).then(|| mask.trailing_zeros() as usize);
    }
    values.iter().position(|&value| value == target)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn equal_mask_8_avx2(values: &[u32; 8], target: u32) -> u32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller supplies all eight lanes; the feature is enabled here.
    let values = unsafe { _mm256_loadu_si256(values.as_ptr().cast()) };
    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
        values,
        _mm256_set1_epi32(target as i32),
    ))) as u32
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn equal_mask_8_neon(values: &[u32; 8], target: u32) -> u32 {
    use std::arch::aarch64::*;
    // SAFETY: each load addresses four lanes of the fixed eight-lane input.
    unsafe {
        let target = vdupq_n_u32(target);
        let weights = [1u32, 2, 4, 8];
        let weights = vld1q_u32(weights.as_ptr());
        let lo = vceqq_u32(vld1q_u32(values.as_ptr()), target);
        let hi = vceqq_u32(vld1q_u32(values.as_ptr().add(4)), target);
        vaddvq_u32(vandq_u32(lo, weights)) | (vaddvq_u32(vandq_u32(hi, weights)) << 4)
    }
}

/// Lower bound at or after `from` in a decoded posting block. Full blocks
/// expose their fixed geometry to LLVM; tails keep the existing SIMD search.
#[inline]
pub(crate) fn find_first_ge_block_from(docs: &[u32], from: usize, target: u32) -> usize {
    debug_assert!(from <= docs.len());
    if let Ok(block) = <&[u32; 128]>::try_from(docs) {
        if block[127] < target {
            return 128;
        }
        let mut base = 0;
        let mut step = 64;
        while step != 0 {
            base += usize::from(block[base + step - 1] < target) * step;
            step >>= 1;
        }
        base.max(from)
    } else {
        from + find_first_ge_u32(&docs[from..], target)
    }
}

/// Find index of first element >= `target` in a sorted `u32` slice.
///
/// Equivalent to `slice.partition_point(|&d| d < target)` but uses SIMD to
/// scan 4 elements per cycle. Faster than binary search for slices ≤ 256
/// elements because it avoids the data-dependency chain inherent in binary
/// search (~8-10 cycles/iteration vs ~1-2 cycles/iteration for SIMD scan).
///
/// Returns `slice.len()` if no element >= `target`.
#[inline]
pub fn find_first_ge_u32(slice: &[u32], target: u32) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            // SAFETY: NEON availability checked; the kernel stays in bounds.
            return unsafe { find_first_ge_u32_neon(slice, target) };
        }
        slice.partition_point(|&d| d < target)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            // SAFETY: AVX2 availability checked; the kernel stays in bounds.
            return unsafe { find_first_ge_u32_avx2(slice, target) };
        }
        // The kernel only needs SSE2, which is part of the x86_64 baseline,
        // so no runtime feature detection is required.
        // SAFETY: SSE2 is always available on x86_64; the kernel stays in bounds.
        unsafe { find_first_ge_u32_sse(slice, target) }
    }

    // Scalar fallback (WASM, other architectures)
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        slice.partition_point(|&d| d < target)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn find_first_ge_u32_neon(slice: &[u32], target: u32) -> usize {
    use std::arch::aarch64::*;

    let n = slice.len();
    let ptr = slice.as_ptr();
    let target_vec = vdupq_n_u32(target);
    // Bit positions for each lane: [1, 2, 4, 8]
    let bit_mask: uint32x4_t = core::mem::transmute([1u32, 2u32, 4u32, 8u32]);

    let chunks = n / 16;
    let mut base = 0usize;

    // Process 16 elements per iteration (4 × 4-wide NEON compares)
    for _ in 0..chunks {
        let v0 = vld1q_u32(ptr.add(base));
        let v1 = vld1q_u32(ptr.add(base + 4));
        let v2 = vld1q_u32(ptr.add(base + 8));
        let v3 = vld1q_u32(ptr.add(base + 12));

        let c0 = vcgeq_u32(v0, target_vec);
        let c1 = vcgeq_u32(v1, target_vec);
        let c2 = vcgeq_u32(v2, target_vec);
        let c3 = vcgeq_u32(v3, target_vec);

        let m0 = vaddvq_u32(vandq_u32(c0, bit_mask));
        if m0 != 0 {
            return base + m0.trailing_zeros() as usize;
        }
        let m1 = vaddvq_u32(vandq_u32(c1, bit_mask));
        if m1 != 0 {
            return base + 4 + m1.trailing_zeros() as usize;
        }
        let m2 = vaddvq_u32(vandq_u32(c2, bit_mask));
        if m2 != 0 {
            return base + 8 + m2.trailing_zeros() as usize;
        }
        let m3 = vaddvq_u32(vandq_u32(c3, bit_mask));
        if m3 != 0 {
            return base + 12 + m3.trailing_zeros() as usize;
        }
        base += 16;
    }

    // Process remaining 4 elements at a time
    while base + 4 <= n {
        let vals = vld1q_u32(ptr.add(base));
        let cmp = vcgeq_u32(vals, target_vec);
        let mask = vaddvq_u32(vandq_u32(cmp, bit_mask));
        if mask != 0 {
            return base + mask.trailing_zeros() as usize;
        }
        base += 4;
    }

    // Scalar remainder (0-3 elements)
    while base < n {
        if *slice.get_unchecked(base) >= target {
            return base;
        }
        base += 1;
    }
    n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn find_first_ge_u32_sse(slice: &[u32], target: u32) -> usize {
    use std::arch::x86_64::*;

    let n = slice.len();
    let ptr = slice.as_ptr();

    // For unsigned >= comparison: XOR with 0x80000000 converts to signed domain
    let sign_flip = _mm_set1_epi32(i32::MIN);
    let target_xor = _mm_xor_si128(_mm_set1_epi32(target as i32), sign_flip);

    let chunks = n / 16;
    let mut base = 0usize;

    // Process 16 elements per iteration (4 × 4-wide SSE compares)
    for _ in 0..chunks {
        let v0 = _mm_xor_si128(_mm_loadu_si128(ptr.add(base) as *const __m128i), sign_flip);
        let v1 = _mm_xor_si128(
            _mm_loadu_si128(ptr.add(base + 4) as *const __m128i),
            sign_flip,
        );
        let v2 = _mm_xor_si128(
            _mm_loadu_si128(ptr.add(base + 8) as *const __m128i),
            sign_flip,
        );
        let v3 = _mm_xor_si128(
            _mm_loadu_si128(ptr.add(base + 12) as *const __m128i),
            sign_flip,
        );

        // ge = eq | gt (in signed domain after XOR)
        let ge0 = _mm_or_si128(
            _mm_cmpeq_epi32(v0, target_xor),
            _mm_cmpgt_epi32(v0, target_xor),
        );
        let m0 = _mm_movemask_ps(_mm_castsi128_ps(ge0)) as u32;
        if m0 != 0 {
            return base + m0.trailing_zeros() as usize;
        }

        let ge1 = _mm_or_si128(
            _mm_cmpeq_epi32(v1, target_xor),
            _mm_cmpgt_epi32(v1, target_xor),
        );
        let m1 = _mm_movemask_ps(_mm_castsi128_ps(ge1)) as u32;
        if m1 != 0 {
            return base + 4 + m1.trailing_zeros() as usize;
        }

        let ge2 = _mm_or_si128(
            _mm_cmpeq_epi32(v2, target_xor),
            _mm_cmpgt_epi32(v2, target_xor),
        );
        let m2 = _mm_movemask_ps(_mm_castsi128_ps(ge2)) as u32;
        if m2 != 0 {
            return base + 8 + m2.trailing_zeros() as usize;
        }

        let ge3 = _mm_or_si128(
            _mm_cmpeq_epi32(v3, target_xor),
            _mm_cmpgt_epi32(v3, target_xor),
        );
        let m3 = _mm_movemask_ps(_mm_castsi128_ps(ge3)) as u32;
        if m3 != 0 {
            return base + 12 + m3.trailing_zeros() as usize;
        }
        base += 16;
    }

    // Process remaining 4 elements at a time
    while base + 4 <= n {
        let vals = _mm_xor_si128(_mm_loadu_si128(ptr.add(base) as *const __m128i), sign_flip);
        let ge = _mm_or_si128(
            _mm_cmpeq_epi32(vals, target_xor),
            _mm_cmpgt_epi32(vals, target_xor),
        );
        let mask = _mm_movemask_ps(_mm_castsi128_ps(ge)) as u32;
        if mask != 0 {
            return base + mask.trailing_zeros() as usize;
        }
        base += 4;
    }

    // Scalar remainder (0-3 elements)
    while base < n {
        if *slice.get_unchecked(base) >= target {
            return base;
        }
        base += 1;
    }
    n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn find_first_ge_u32_avx2(slice: &[u32], target: u32) -> usize {
    use std::arch::x86_64::*;
    let target_vec = _mm256_set1_epi32(target as i32);
    let mut base = 0;
    while slice.len() - base >= 8 {
        let values = _mm256_loadu_si256(slice.as_ptr().add(base).cast());
        // min(value, target) equals target precisely when value >= target.
        // Keep this unsigned comparison packed through the mask extraction.
        let ge = _mm256_cmpeq_epi32(_mm256_min_epu32(values, target_vec), target_vec);
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(ge)) as u32;
        if mask != 0 {
            return base + mask.trailing_zeros() as usize;
        }
        base += 8;
    }
    base + find_first_ge_u32_sse(&slice[base..], target)
}

#[cfg(test)]
mod find_first_ge_tests {
    use super::find_first_ge_u32;

    #[test]
    fn test_find_first_ge_basic() {
        let data: Vec<u32> = (0..128).map(|i| i * 3).collect(); // [0, 3, 6, ..., 381]
        assert_eq!(find_first_ge_u32(&data, 0), 0);
        assert_eq!(find_first_ge_u32(&data, 1), 1); // first >= 1 is 3 at idx 1
        assert_eq!(find_first_ge_u32(&data, 3), 1);
        assert_eq!(find_first_ge_u32(&data, 4), 2); // first >= 4 is 6 at idx 2
        assert_eq!(find_first_ge_u32(&data, 381), 127);
        assert_eq!(find_first_ge_u32(&data, 382), 128); // past end
    }

    #[test]
    fn test_find_first_ge_matches_partition_point() {
        let data: Vec<u32> = vec![1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75];
        for target in 0..80 {
            let expected = data.partition_point(|&d| d < target);
            let actual = find_first_ge_u32(&data, target);
            assert_eq!(actual, expected, "target={}", target);
        }
    }

    #[test]
    fn test_find_first_ge_small_slices() {
        // Empty
        assert_eq!(find_first_ge_u32(&[], 5), 0);
        // Single element
        assert_eq!(find_first_ge_u32(&[10], 5), 0);
        assert_eq!(find_first_ge_u32(&[10], 10), 0);
        assert_eq!(find_first_ge_u32(&[10], 11), 1);
        // Three elements (< SIMD width)
        assert_eq!(find_first_ge_u32(&[2, 4, 6], 5), 2);
    }

    #[test]
    fn test_find_first_ge_full_block() {
        // Simulate a full 128-entry block
        let data: Vec<u32> = (100..228).collect();
        assert_eq!(find_first_ge_u32(&data, 100), 0);
        assert_eq!(find_first_ge_u32(&data, 150), 50);
        assert_eq!(find_first_ge_u32(&data, 227), 127);
        assert_eq!(find_first_ge_u32(&data, 228), 128);
        assert_eq!(find_first_ge_u32(&data, 99), 0);
    }

    #[test]
    fn test_find_first_ge_u32_max() {
        // Test with large u32 values (unsigned correctness)
        let data = vec![u32::MAX - 10, u32::MAX - 5, u32::MAX - 1, u32::MAX];
        assert_eq!(find_first_ge_u32(&data, u32::MAX - 10), 0);
        assert_eq!(find_first_ge_u32(&data, u32::MAX - 7), 1);
        assert_eq!(find_first_ge_u32(&data, u32::MAX), 3);
    }

    /// Every slice length that exercises the 16-wide, 4-wide and scalar
    /// remainder paths, with duplicate runs, sign-bit crossings and
    /// `u32::MAX`, against `partition_point` for every interesting target.
    #[test]
    fn find_first_ge_matches_partition_point_for_every_length_and_target() {
        let mut state = 0x9E37_79B9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for n in 0..=140usize {
            let mut data: Vec<u32> = Vec::with_capacity(n);
            let mut value = next() % 8;
            for i in 0..n {
                // Duplicate runs, occasional big jumps across the sign bit,
                // and a saturating tail so u32::MAX appears (possibly repeated).
                let step = match next() % 5 {
                    0 | 1 => 0,
                    2 => 1,
                    3 => next() % 1000,
                    _ => 0x4000_0000 + next() % 0x1000_0000,
                };
                value = value.saturating_add(step);
                if i + 3 >= n && n > 8 {
                    value = u32::MAX;
                }
                data.push(value);
            }
            assert!(data.windows(2).all(|w| w[0] <= w[1]));
            let mut targets: Vec<u32> =
                vec![0, 1, u32::MAX - 1, u32::MAX, i32::MAX as u32, 1 << 31];
            for &d in &data {
                targets.extend([d.saturating_sub(1), d, d.saturating_add(1)]);
            }
            for target in targets {
                let expected = data.partition_point(|&d| d < target);
                assert_eq!(
                    find_first_ge_u32(&data, target),
                    expected,
                    "n={n} target={target} data={data:?}"
                );
            }
        }
    }
}

/// Regression coverage for the algebraic (reassociation-permitting) float
/// reductions. Pins them against a strict f64 reference and against the
/// hand-written SIMD kernels they must stay interchangeable with.
#[cfg(test)]
mod algebraic_reduction_tests {
    use super::*;

    fn algebraic_test_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
            })
            .collect()
    }

    /// Dimensions that straddle every SIMD chunk width used in this module
    /// (4/8/16 lanes) plus their tails, and realistic embedding widths.
    const ALGEBRAIC_TEST_DIMS: [usize; 18] = [
        0, 1, 3, 4, 7, 8, 15, 16, 17, 64, 100, 127, 200, 300, 384, 768, 1000, 1536,
    ];

    #[test]
    fn test_algebraic_squared_l2_matches_f64_reference() {
        for dim in ALGEBRAIC_TEST_DIMS {
            let a = algebraic_test_vector(dim, 0x51ed_0001);
            let b = algebraic_test_vector(dim, 0x51ed_0002);
            let reference: f64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| {
                    let delta = f64::from(x) - f64::from(y);
                    delta * delta
                })
                .sum();
            let actual = squared_l2_f32(&a, &b);
            let tolerance = (reference * 1e-5).max(1e-6);
            assert!(
                (f64::from(actual) - reference).abs() <= tolerance,
                "dim {dim}: squared_l2_f32 {actual} drifted from f64 reference {reference}"
            );
        }
    }

    #[test]
    fn test_algebraic_squared_l2_uses_shorter_length() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 4.0];
        assert_eq!(squared_l2_f32(&a, &b), 4.0);
        assert_eq!(squared_l2_f32(&b, &a), 4.0);
    }

    #[test]
    fn test_algebraic_norm_squared_matches_f64_reference() {
        for dim in ALGEBRAIC_TEST_DIMS {
            let v = algebraic_test_vector(dim, 0x51ed_0003);
            let reference: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let actual = norm_squared_f32(&v);
            let tolerance = (reference * 1e-5).max(1e-6);
            assert!(
                (f64::from(actual) - reference).abs() <= tolerance,
                "dim {dim}: norm_squared_f32 {actual} drifted from f64 reference {reference}"
            );
            assert!(
                (f64::from(norm_f32(&v)) - reference.sqrt()).abs() <= tolerance.sqrt().max(1e-5)
            );
        }
    }

    #[test]
    fn test_algebraic_dot_scalar_matches_simd_dispatch() {
        for dim in ALGEBRAIC_TEST_DIMS {
            let a = algebraic_test_vector(dim, 0x51ed_0004);
            let b = algebraic_test_vector(dim, 0x51ed_0005);
            let dispatched = dot_product_f32(&a, &b, dim);
            let scalar = dot_product_f32_scalar(&a, &b);
            let tolerance = (dispatched.abs() * 1e-5).max(1e-5);
            assert!(
                (dispatched - scalar).abs() <= tolerance,
                "dim {dim}: scalar dot {scalar} disagrees with dispatched {dispatched}"
            );

            let (fused_dot, fused_norm) = fused_dot_norm(&a, &b, dim);
            let (scalar_dot, scalar_norm) = fused_dot_norm_scalar(&a, &b);
            assert!((fused_dot - scalar_dot).abs() <= tolerance);
            assert!((fused_norm - scalar_norm).abs() <= (fused_norm.abs() * 1e-5).max(1e-5));
        }
    }

    /// The SIMD remainder pass must agree with an f64 reference at every
    /// tail length (4/8/12 trailing lanes plus a scalar rest), and the
    /// batch-resolved kernels must be the very same code paths as the
    /// per-call dispatchers.
    #[test]
    fn test_simd_tail_dims_match_f64_reference_and_resolved_kernels() {
        for dim in ALGEBRAIC_TEST_DIMS {
            let a = algebraic_test_vector(dim, 0x51ed_0006);
            let b = algebraic_test_vector(dim, 0x51ed_0007);
            let reference: f64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            let dispatched = dot_product_f32(&a, &b, dim);
            let tolerance = (reference.abs() * 1e-5).max(1e-5);
            assert!(
                (f64::from(dispatched) - reference).abs() <= tolerance,
                "dim {dim}: dot {dispatched} drifted from f64 reference {reference}"
            );
            let kernel = DenseF32Kernel::resolve();
            assert_eq!(kernel.dot(&a, &b, dim).to_bits(), dispatched.to_bits());
            let (fused_dot, fused_norm) = fused_dot_norm(&a, &b, dim);
            let (kernel_dot, kernel_norm) = kernel.fused_dot_norm(&a, &b, dim);
            assert_eq!(kernel_dot.to_bits(), fused_dot.to_bits());
            assert_eq!(kernel_norm.to_bits(), fused_norm.to_bits());
            let norm_reference: f64 = b.iter().map(|&y| f64::from(y) * f64::from(y)).sum();
            assert!(
                (f64::from(fused_norm) - norm_reference).abs() <= (norm_reference * 1e-5).max(1e-5),
                "dim {dim}: fused norm {fused_norm} drifted from {norm_reference}"
            );

            let query_f16: Vec<u16> = a.iter().map(|&v| f32_to_f16(v)).collect();
            let vec_f16: Vec<u16> = b.iter().map(|&v| f32_to_f16(v)).collect();
            let f16_kernel = QuantF16Kernel::resolve();
            let (d, n) = fused_dot_norm_f16(&query_f16, &vec_f16, dim);
            let (kd, kn) = f16_kernel.fused_dot_norm(&query_f16, &vec_f16, dim);
            assert_eq!((kd.to_bits(), kn.to_bits()), (d.to_bits(), n.to_bits()));
            assert_eq!(
                f16_kernel.dot(&query_f16, &vec_f16, dim).to_bits(),
                dot_product_f16_quant(&query_f16, &vec_f16, dim).to_bits()
            );

            let vec_u8: Vec<u8> = b.iter().map(|&v| f32_to_u8_saturating(v)).collect();
            let u8_kernel = QuantU8Kernel::resolve();
            let (d, n) = fused_dot_norm_u8(&a, &vec_u8, dim);
            let (kd, kn) = u8_kernel.fused_dot_norm(&a, &vec_u8, dim);
            assert_eq!((kd.to_bits(), kn.to_bits()), (d.to_bits(), n.to_bits()));
            assert_eq!(
                u8_kernel.dot(&a, &vec_u8, dim).to_bits(),
                dot_product_u8_quant(&a, &vec_u8, dim).to_bits()
            );
        }
    }

    /// Rust's algebraic operations differ from `-ffast-math`: they permit
    /// reassociation but never assume finite inputs. NaN and infinity must
    /// still propagate, otherwise a degenerate stored vector would silently
    /// score as a finite number instead of being rejected downstream.
    #[test]
    fn test_algebraic_reductions_propagate_non_finite() {
        let finite = vec![1.0f32; 8];

        let mut with_nan = finite.clone();
        with_nan[5] = f32::NAN;
        assert!(norm_squared_f32(&with_nan).is_nan());
        assert!(squared_l2_f32(&with_nan, &finite).is_nan());
        assert!(dot_product_f32_scalar(&with_nan, &finite).is_nan());

        let mut with_inf = finite.clone();
        with_inf[2] = f32::INFINITY;
        assert!(norm_squared_f32(&with_inf).is_infinite());
        assert!(squared_l2_f32(&with_inf, &finite).is_infinite());
        assert!(dot_product_f32_scalar(&with_inf, &finite).is_infinite());
    }
}
