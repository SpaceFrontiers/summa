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

            // Load 4 bytes and zero-extend to u32
            let bytes = _mm_cvtsi32_si128(*(input.as_ptr().add(base) as *const i32));
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

            // Load 8 bytes (4 u16 values) and zero-extend to u32
            let vals = _mm_loadl_epi64(in_ptr as *const __m128i);
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
}
