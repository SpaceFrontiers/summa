//! Shared SIMD-accelerated functions for posting list compression
//!
//! This module provides platform-optimized implementations for common operations:
//! - **Unpacking**: Convert packed 8/16/32-bit values to u32 arrays
//! - **Delta decoding**: Prefix sum for converting deltas to absolute values
//! - **Add one**: Increment all values in an array (for TF decoding)
//!
//! Supports:
//! - **NEON** on aarch64 (Apple Silicon, ARM servers)
//! - **SSE/SSE4.1** on x86_64 (Intel/AMD)
//! - **Scalar fallback** for other architectures

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

    /// Fused unpack 8-bit + delta decode using NEON
    /// Processes 4 values at a time, fusing unpack and prefix sum
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_8bit_delta_decode(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = vdupq_n_u32(1);
        let mut carry = vdupq_n_u32(first_value);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;

            // Load 4 bytes and widen to u32
            let b0 = input[base] as u32;
            let b1 = input[base + 1] as u32;
            let b2 = input[base + 2] as u32;
            let b3 = input[base + 3] as u32;
            let deltas = [b0, b1, b2, b3];
            let d = vld1q_u32(deltas.as_ptr());

            // Add 1 (since we store gap-1)
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

        // Handle remainder
        let base = full_groups * 4;
        let mut scalar_carry = vgetq_lane_u32(carry, 0);
        for j in 0..remainder {
            scalar_carry = scalar_carry
                .wrapping_add(input[base + j] as u32)
                .wrapping_add(1);
            output[base + j + 1] = scalar_carry;
        }
    }

    /// Fused unpack 16-bit + delta decode using NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_16bit_delta_decode(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = vdupq_n_u32(1);
        let mut carry = vdupq_n_u32(first_value);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;
            let in_ptr = input.as_ptr().add(base * 2) as *const u16;

            // Load 4 u16 values and widen to u32
            let vals = vld1_u16(in_ptr);
            let d = vmovl_u16(vals);

            // Add 1 (since we store gap-1)
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

        // Handle remainder
        let base = full_groups * 4;
        let mut scalar_carry = vgetq_lane_u32(carry, 0);
        for j in 0..remainder {
            let idx = (base + j) * 2;
            let delta = u16::from_le_bytes([input[idx], input[idx + 1]]) as u32;
            scalar_carry = scalar_carry.wrapping_add(delta).wrapping_add(1);
            output[base + j + 1] = scalar_carry;
        }
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

    /// Fused unpack 8-bit + delta decode using SSE
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    pub unsafe fn unpack_8bit_delta_decode(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = _mm_set1_epi32(1);
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

            // Add 1 (since we store gap-1)
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

        // Handle remainder
        let base = full_groups * 4;
        let mut scalar_carry = _mm_extract_epi32(carry, 0) as u32;
        for j in 0..remainder {
            scalar_carry = scalar_carry
                .wrapping_add(input[base + j] as u32)
                .wrapping_add(1);
            output[base + j + 1] = scalar_carry;
        }
    }

    /// Fused unpack 16-bit + delta decode using SSE
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    pub unsafe fn unpack_16bit_delta_decode(
        input: &[u8],
        output: &mut [u32],
        first_value: u32,
        count: usize,
    ) {
        output[0] = first_value;
        if count <= 1 {
            return;
        }

        let ones = _mm_set1_epi32(1);
        let mut carry = _mm_set1_epi32(first_value as i32);

        let full_groups = (count - 1) / 4;
        let remainder = (count - 1) % 4;

        for group in 0..full_groups {
            let base = group * 4;
            let in_ptr = input.as_ptr().add(base * 2);

            // Load 8 bytes (4 u16 values, unaligned) and zero-extend to u32
            let vals = _mm_loadl_epi64(in_ptr as *const __m128i); // loadl_epi64 supports unaligned
            let d = _mm_cvtepu16_epi32(vals);

            // Add 1 (since we store gap-1)
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

        // Handle remainder
        let base = full_groups * 4;
        let mut scalar_carry = _mm_extract_epi32(carry, 0) as u32;
        for j in 0..remainder {
            let idx = (base + j) * 2;
            let delta = u16::from_le_bytes([input[idx], input[idx + 1]]) as u32;
            scalar_carry = scalar_carry.wrapping_add(delta).wrapping_add(1);
            output[base + j + 1] = scalar_carry;
        }
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

    /// Check if AVX2 is available at runtime
    #[inline]
    pub fn is_available() -> bool {
        is_x86_feature_detected!("avx2")
    }
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================

#[allow(dead_code)]
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
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX2 (256-bit) over SSE (128-bit) when available
        if avx2::is_available() {
            unsafe {
                avx2::unpack_32bit(input, output, count);
            }
        } else {
            // SSE2 is always available on x86_64
            unsafe {
                sse::unpack_32bit(input, output, count);
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::unpack_32bit(input, output, count);
    }
}

/// Delta decode with SIMD acceleration
///
/// Converts delta-encoded values to absolute values.
/// Input: deltas[i] = value[i+1] - value[i] - 1 (gap minus one)
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
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX2 (256-bit) over SSE (128-bit) when available
        if avx2::is_available() {
            unsafe {
                avx2::add_one(values, count);
            }
        } else {
            // SSE2 is always available on x86_64
            unsafe {
                sse::add_one(values, count);
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::add_one(values, count);
    }
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

    /// Convert from stored u8 value (must be 0, 8, 16, or 32)
    #[inline]
    pub fn from_u8(bits: u8) -> Self {
        match bits {
            0 => RoundedBitWidth::Zero,
            8 => RoundedBitWidth::Bits8,
            16 => RoundedBitWidth::Bits16,
            32 => RoundedBitWidth::Bits32,
            _ => RoundedBitWidth::Bits32, // Fallback for invalid values
        }
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
            // For 32-bit, unpack then delta decode (no fused version needed)
            unpack_32bit(input, output, count);
            // Delta decode in place - but we need the deltas separate
            // Actually for 32-bit we should just unpack and delta decode separately
            if count > 0 {
                let mut carry = first_value;
                output[0] = first_value;
                for item in output.iter_mut().take(count).skip(1) {
                    // item currently holds delta (gap-1)
                    carry = carry.wrapping_add(*item).wrapping_add(1);
                    *item = carry;
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
    if count == 0 {
        return;
    }

    output[0] = first_value;
    if count == 1 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                neon::unpack_8bit_delta_decode(input, output, first_value, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            unsafe {
                sse::unpack_8bit_delta_decode(input, output, first_value, count);
            }
            return;
        }
    }

    // Scalar fallback
    let mut carry = first_value;
    for i in 0..count - 1 {
        carry = carry.wrapping_add(input[i] as u32).wrapping_add(1);
        output[i + 1] = carry;
    }
}

/// Fused unpack 16-bit + delta decode in a single pass
#[inline]
pub fn unpack_16bit_delta_decode(input: &[u8], output: &mut [u32], first_value: u32, count: usize) {
    if count == 0 {
        return;
    }

    output[0] = first_value;
    if count == 1 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            unsafe {
                neon::unpack_16bit_delta_decode(input, output, first_value, count);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            unsafe {
                sse::unpack_16bit_delta_decode(input, output, first_value, count);
            }
            return;
        }
    }

    // Scalar fallback
    let mut carry = first_value;
    for i in 0..count - 1 {
        let idx = i * 2;
        let delta = u16::from_le_bytes([input[idx], input[idx + 1]]) as u32;
        carry = carry.wrapping_add(delta).wrapping_add(1);
        output[i + 1] = carry;
    }
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
/// Computes: output[i] = input[i] as f32 * scale + min_val
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

        // Load 4 bytes and zero-extend to 32-bit
        let b0 = input[base] as i32;
        let b1 = input[base + 1] as i32;
        let b2 = input[base + 2] as i32;
        let b3 = input[base + 3] as i32;

        let ints = _mm_set_epi32(b3, b2, b1, b0);
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

/// Compute dot product of two f32 arrays with SIMD acceleration
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32], count: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            return unsafe { dot_product_f32_neon(a, b, count) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            return unsafe { dot_product_f32_sse(a, b, count) };
        }
    }

    // Scalar fallback
    let mut sum = 0.0f32;
    for i in 0..count {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::aarch64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut acc = vdupq_n_f32(0.0);

    for chunk in 0..chunks {
        let base = chunk * 4;
        let va = vld1q_f32(a.as_ptr().add(base));
        let vb = vld1q_f32(b.as_ptr().add(base));
        acc = vfmaq_f32(acc, va, vb);
    }

    // Horizontal sum
    let mut sum = vaddvq_f32(acc);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
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

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

/// Find maximum value in f32 array with SIMD acceleration
#[inline]
pub fn max_f32(values: &[f32], count: usize) -> f32 {
    if count == 0 {
        return f32::NEG_INFINITY;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            return unsafe { max_f32_neon(values, count) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            return unsafe { max_f32_sse(values, count) };
        }
    }

    // Scalar fallback
    values[..count]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn max_f32_neon(values: &[f32], count: usize) -> f32 {
    use std::arch::aarch64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);

    for chunk in 0..chunks {
        let base = chunk * 4;
        let v = vld1q_f32(values.as_ptr().add(base));
        max_v = vmaxq_f32(max_v, v);
    }

    // Horizontal max
    let mut max_val = vmaxvq_f32(max_v);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        max_val = max_val.max(values[base + i]);
    }

    max_val
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn max_f32_sse(values: &[f32], count: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut max_v = _mm_set1_ps(f32::NEG_INFINITY);

    for chunk in 0..chunks {
        let base = chunk * 4;
        let v = _mm_loadu_ps(values.as_ptr().add(base));
        max_v = _mm_max_ps(max_v, v);
    }

    // Horizontal max: [a, b, c, d] -> max(a, b, c, d)
    let shuf = _mm_shuffle_ps(max_v, max_v, 0b10_11_00_01); // [b, a, d, c]
    let max1 = _mm_max_ps(max_v, shuf); // [max(a,b), max(a,b), max(c,d), max(c,d)]
    let shuf2 = _mm_movehl_ps(max1, max1); // [max(c,d), max(c,d), ?, ?]
    let final_max = _mm_max_ss(max1, shuf2); // [max(a,b,c,d), ?, ?, ?]

    let mut max_val = _mm_cvtss_f32(final_max);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        max_val = max_val.max(values[base + i]);
    }

    max_val
}

// ============================================================================
// Squared Euclidean Distance for Dense Vector Search
// ============================================================================

/// Compute cosine similarity between two f32 vectors with SIMD acceleration
///
/// Returns dot(a,b) / (||a|| * ||b||), range [-1, 1]
/// Returns 0.0 if either vector has zero norm.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
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

/// Compute squared Euclidean distance between two f32 vectors with SIMD acceleration
///
/// Returns sum((a[i] - b[i])^2) for all i
#[inline]
pub fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let count = a.len();

    if count == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            return unsafe { squared_euclidean_neon(a, b, count) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            return unsafe { squared_euclidean_avx2(a, b, count) };
        }
        if sse::is_available() {
            return unsafe { squared_euclidean_sse(a, b, count) };
        }
    }

    // Scalar fallback
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn squared_euclidean_neon(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::aarch64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut acc = vdupq_n_f32(0.0);

    for chunk in 0..chunks {
        let base = chunk * 4;
        let va = vld1q_f32(a.as_ptr().add(base));
        let vb = vld1q_f32(b.as_ptr().add(base));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff); // acc += diff * diff (fused multiply-add)
    }

    // Horizontal sum
    let mut sum = vaddvq_f32(acc);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn squared_euclidean_sse(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut acc = _mm_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(base));
        let vb = _mm_loadu_ps(b.as_ptr().add(base));
        let diff = _mm_sub_ps(va, vb);
        acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
    }

    // Horizontal sum: [a, b, c, d] -> a + b + c + d
    let shuf = _mm_shuffle_ps(acc, acc, 0b10_11_00_01); // [b, a, d, c]
    let sums = _mm_add_ps(acc, shuf); // [a+b, a+b, c+d, c+d]
    let shuf2 = _mm_movehl_ps(sums, sums); // [c+d, c+d, ?, ?]
    let final_sum = _mm_add_ss(sums, shuf2); // [a+b+c+d, ?, ?, ?]

    let mut sum = _mm_cvtss_f32(final_sum);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn squared_euclidean_avx2(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = count / 8;
    let remainder = count % 8;

    let mut acc = _mm256_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc); // acc += diff * diff (FMA)
    }

    // Horizontal sum of 8 floats
    // First, add high 128 bits to low 128 bits
    let high = _mm256_extractf128_ps(acc, 1);
    let low = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(low, high);

    // Now sum the 4 floats in sum128
    let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);

    let mut sum = _mm_cvtss_f32(final_sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }

    sum
}

/// Batch compute squared Euclidean distances from one query to multiple vectors
///
/// Returns distances[i] = squared_euclidean_distance(query, vectors[i])
/// This is more efficient than calling squared_euclidean_distance in a loop
/// because we can keep the query in registers.
#[inline]
pub fn batch_squared_euclidean_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    distances: &mut [f32],
) {
    debug_assert_eq!(vectors.len(), distances.len());

    #[cfg(target_arch = "x86_64")]
    {
        if avx2::is_available() {
            for (i, vec) in vectors.iter().enumerate() {
                distances[i] = unsafe { squared_euclidean_avx2(query, vec, query.len()) };
            }
            return;
        }
    }

    // Fallback to individual calls
    for (i, vec) in vectors.iter().enumerate() {
        distances[i] = squared_euclidean_distance(query, vec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_max_f32() {
        let values = vec![1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0];
        let result = max_f32(&values, 6);
        assert!((result - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_f32_large() {
        // Test with 128 values, max at position 77
        let mut values: Vec<f32> = (0..128).map(|i| i as f32).collect();
        values[77] = 1000.0;

        let result = max_f32(&values, 128);
        assert!((result - 1000.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_f32_negative() {
        let values = vec![-5.0f32, -2.0, -10.0, -1.0, -3.0];
        let result = max_f32(&values, 5);
        assert!((result - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_max_f32_empty() {
        let values: Vec<f32> = vec![];
        let result = max_f32(&values, 0);
        assert_eq!(result, f32::NEG_INFINITY);
    }

    #[test]
    fn test_squared_euclidean_distance() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let result = squared_euclidean_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_squared_euclidean_distance_large() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1) + 0.5).collect();
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let result = squared_euclidean_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-3,
            "expected {}, got {}",
            expected,
            result
        );
    }
}
