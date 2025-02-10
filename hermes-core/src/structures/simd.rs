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
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_f32_avx2(a, b, count) };
        }
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
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32], count: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = count / 8;
    let remainder = count % 8;

    let mut acc = _mm256_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    // Horizontal sum: 256-bit → 128-bit → scalar
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0b10_11_00_01);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);

    let mut sum = _mm_cvtss_f32(final_sum);

    let base = chunks * 8;
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
// Batched Cosine Similarity for Dense Vector Search
// ============================================================================

/// Fused dot-product + self-norm in a single pass (SIMD accelerated).
///
/// Returns (dot(a, b), dot(b, b)) — i.e. the dot product of a·b and ||b||².
/// Loads `b` only once (halves memory bandwidth vs two separate dot products).
#[inline]
fn fused_dot_norm(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if neon::is_available() {
            return unsafe { fused_dot_norm_neon(a, b, count) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { fused_dot_norm_avx2(a, b, count) };
        }
        if sse::is_available() {
            return unsafe { fused_dot_norm_sse(a, b, count) };
        }
    }

    // Scalar fallback
    let mut dot = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..count {
        dot += a[i] * b[i];
        norm_b += b[i] * b[i];
    }
    (dot, norm_b)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_neon(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    use std::arch::aarch64::*;

    let chunks = count / 4;
    let remainder = count % 4;

    let mut acc_dot = vdupq_n_f32(0.0);
    let mut acc_norm = vdupq_n_f32(0.0);

    for chunk in 0..chunks {
        let base = chunk * 4;
        let va = vld1q_f32(a.as_ptr().add(base));
        let vb = vld1q_f32(b.as_ptr().add(base));
        acc_dot = vfmaq_f32(acc_dot, va, vb);
        acc_norm = vfmaq_f32(acc_norm, vb, vb);
    }

    let mut dot = vaddvq_f32(acc_dot);
    let mut norm = vaddvq_f32(acc_norm);

    let base = chunks * 4;
    for i in 0..remainder {
        dot += a[base + i] * b[base + i];
        norm += b[base + i] * b[base + i];
    }

    (dot, norm)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fused_dot_norm_avx2(a: &[f32], b: &[f32], count: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    let chunks = count / 8;
    let remainder = count % 8;

    let mut acc_dot = _mm256_setzero_ps();
    let mut acc_norm = _mm256_setzero_ps();

    for chunk in 0..chunks {
        let base = chunk * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
        acc_norm = _mm256_fmadd_ps(vb, vb, acc_norm);
    }

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

    let base = chunks * 8;
    for i in 0..remainder {
        dot += a[base + i] * b[base + i];
        norm += b[base + i] * b[base + i];
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
        dot += a[base + i] * b[base + i];
        norm += b[base + i] * b[base + i];
    }

    (dot, norm)
}

/// Fast approximate reciprocal square root: 1/sqrt(x).
///
/// Uses the IEEE 754 bit trick (Quake III) + one Newton-Raphson iteration
/// for ~23-bit precision — sufficient for cosine similarity scoring.
/// ~3-5x faster than `1.0 / x.sqrt()` on most architectures.
#[inline]
fn fast_inv_sqrt(x: f32) -> f32 {
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
    debug_assert!(vectors.len() >= n * dim);
    debug_assert_eq!(query.len(), dim);

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
        let m = (mantissa | 0x80_0000) >> (1 - exp16);
        return (sign | (m >> 13)) as u16;
    }

    (sign | ((exp16 as u32) << 10) | (mantissa >> 13)) as u16
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

/// Batch convert f32 slice to u8 with [-1,1] → [0,255] mapping
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
    #[target_feature(enable = "neon")]
    pub unsafe fn fused_dot_norm_f16(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
        let chunks8 = dim / 8;
        let remainder = dim % 8;

        let mut acc_dot = vdupq_n_f32(0.0);
        let mut acc_norm = vdupq_n_f32(0.0);

        for c in 0..chunks8 {
            let base = c * 8;

            // Load 8 f16 vector values, hardware-convert to 2×4 f32
            let v_raw = vld1q_u16(vec_f16.as_ptr().add(base));
            let v_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v_raw)));
            let v_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(v_raw)));

            // Load 8 f16 query values, hardware-convert to 2×4 f32
            let q_raw = vld1q_u16(query_f16.as_ptr().add(base));
            let q_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(q_raw)));
            let q_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(q_raw)));

            acc_dot = vfmaq_f32(acc_dot, q_lo, v_lo);
            acc_dot = vfmaq_f32(acc_dot, q_hi, v_hi);
            acc_norm = vfmaq_f32(acc_norm, v_lo, v_lo);
            acc_norm = vfmaq_f32(acc_norm, v_hi, v_hi);
        }

        let mut dot = vaddvq_f32(acc_dot);
        let mut norm = vaddvq_f32(acc_norm);

        let base = chunks8 * 8;
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
}

// ============================================================================
// Scalar fallback for fused dot+norm on quantized vectors
// ============================================================================

#[allow(dead_code)]
fn fused_dot_norm_f16_scalar(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
    let mut dot = 0.0f32;
    let mut norm = 0.0f32;
    for i in 0..dim {
        let v = f16_to_f32(vec_f16[i]);
        let q = f16_to_f32(query_f16[i]);
        dot += q * v;
        norm += v * v;
    }
    (dot, norm)
}

#[allow(dead_code)]
fn fused_dot_norm_u8_scalar(query: &[f32], vec_u8: &[u8], dim: usize) -> (f32, f32) {
    let mut dot = 0.0f32;
    let mut norm = 0.0f32;
    for i in 0..dim {
        let v = u8_to_f32(vec_u8[i]);
        dot += query[i] * v;
        norm += v * v;
    }
    (dot, norm)
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
// Platform dispatch
// ============================================================================

#[inline]
fn fused_dot_norm_f16(query_f16: &[u16], vec_f16: &[u16], dim: usize) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_quant::fused_dot_norm_f16(query_f16, vec_f16, dim) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            return unsafe { fused_dot_norm_f16_sse(query_f16, vec_f16, dim) };
        }
    }

    #[allow(unreachable_code)]
    fused_dot_norm_f16_scalar(query_f16, vec_f16, dim)
}

#[inline]
fn fused_dot_norm_u8(query: &[f32], vec_u8: &[u8], dim: usize) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_quant::fused_dot_norm_u8(query, vec_u8, dim) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            return unsafe { fused_dot_norm_u8_sse(query, vec_u8, dim) };
        }
    }

    #[allow(unreachable_code)]
    fused_dot_norm_u8_scalar(query, vec_u8, dim)
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

    let vec_bytes = dim * 2;
    debug_assert!(vectors_raw.len() >= n * vec_bytes);

    // Vectors file uses data-first layout with 8-byte padding between fields,
    // so mmap slices are always 2-byte aligned for u16 access.
    debug_assert!(
        (vectors_raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()),
        "f16 vector data not 2-byte aligned"
    );

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
/// `vectors_raw` is raw bytes: N vectors × dim bytes (u8, mapping [-1,1]→[0,255]).
/// Converts u8→f32 using NEON widening chain (16 values/iteration), scores with FMA.
/// Memory bandwidth is quartered compared to f32 scoring.
#[inline]
pub fn batch_cosine_scores_u8(query: &[f32], vectors_raw: &[u8], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
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

    debug_assert!(vectors_raw.len() >= n * dim);

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
}

// ============================================================================
// SIMD-accelerated linear scan for sorted u32 slices (within-block seek)
// ============================================================================

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
            return unsafe { find_first_ge_u32_neon(slice, target) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if sse::is_available() {
            return unsafe { find_first_ge_u32_sse(slice, target) };
        }
    }

    // Scalar fallback (WASM, other architectures)
    slice.partition_point(|&d| d < target)
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
#[target_feature(enable = "sse2", enable = "sse4.1")]
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
}
