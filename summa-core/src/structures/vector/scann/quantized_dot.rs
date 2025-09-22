//! `f32 x u8` dot products over the fixed-point routing centroid plane.
//!
//! Quantized float ScaNN routing scores a query against thousands of
//! centroids whose coordinates are stored as one `u8` code per dimension. The
//! previous implementation decoded every coordinate (`min + step * code`) and
//! folded the squared difference with strict IEEE adds, which pins the loop
//! to one dependent scalar add per element. Expanding the square instead
//! leaves a single `f32 x u8` dot product per centroid whose remaining terms
//! are either query-only or centroid-only constants:
//!
//! ```text
//! ||q - c||^2 = sum((q_i - min_i) - step_i * code_i)^2
//!             = C - 2 * dot(q', code) + N_c
//! q'_i = (q_i - min_i) * step_i        query-only, once per level
//! C    = sum (q_i - min_i)^2           query-only, once per level
//! N_c  = sum step_i^2 * code_i^2       centroid-only, once per artifact open
//! ```
//!
//! The kernel is resolved once per model and dispatched without per-call
//! feature detection. NEON and AVX2+FMA kernels keep four independent
//! accumulator chains; the scalar fallback uses algebraic ops so LLVM can
//! vectorize it on targets without a hand-written kernel (including WASM).

/// Resolved `f32 x u8` dot kernel for one process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantizedDotKernel {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    /// Algebraic-op fallback; the only kernel on targets without NEON/AVX2.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    Scalar,
}

impl QuantizedDotKernel {
    /// Detect the widest kernel this CPU supports.
    pub(crate) fn resolve() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Self::Avx2Fma;
            }
            Self::Scalar
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }

    /// `sum weights[i] * codes[i]` over equal-length slices.
    #[inline]
    pub(crate) fn dot(self, weights: &[f32], codes: &[u8]) -> f32 {
        debug_assert_eq!(weights.len(), codes.len());
        let len = weights.len().min(codes.len());
        match self {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is baseline on aarch64; the kernel bounds every
            // load by `len`, which is the shorter of the two slices.
            Self::Neon => unsafe { dot_neon(weights, codes, len) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: guarded by runtime detection of avx2 + fma in
            // `resolve`; loads are unaligned and bounded by `len`.
            Self::Avx2Fma => unsafe { dot_avx2_fma(weights, codes, len) },
            Self::Scalar => dot_scalar(&weights[..len], &codes[..len]),
        }
    }
}

#[inline]
pub(crate) fn dot_scalar(weights: &[f32], codes: &[u8]) -> f32 {
    weights
        .iter()
        .zip(codes)
        .fold(0.0f32, |acc, (&weight, &code)| {
            acc.algebraic_add(weight.algebraic_mul(f32::from(code)))
        })
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(weights: &[f32], codes: &[u8], len: usize) -> f32 {
    use std::arch::aarch64::*;

    let chunks = len / 16;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    // SAFETY (caller): `len <= weights.len()` and `len <= codes.len()`, and
    // every offset below stays under `chunks * 16 <= len`.
    unsafe {
        for chunk in 0..chunks {
            let base = chunk * 16;
            let bytes = vld1q_u8(codes.as_ptr().add(base));
            let low = vmovl_u8(vget_low_u8(bytes));
            let high = vmovl_high_u8(bytes);
            let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(low)));
            let f1 = vcvtq_f32_u32(vmovl_high_u16(low));
            let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(high)));
            let f3 = vcvtq_f32_u32(vmovl_high_u16(high));
            acc0 = vfmaq_f32(acc0, vld1q_f32(weights.as_ptr().add(base)), f0);
            acc1 = vfmaq_f32(acc1, vld1q_f32(weights.as_ptr().add(base + 4)), f1);
            acc2 = vfmaq_f32(acc2, vld1q_f32(weights.as_ptr().add(base + 8)), f2);
            acc3 = vfmaq_f32(acc3, vld1q_f32(weights.as_ptr().add(base + 12)), f3);
        }
    }
    let total = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    let base = chunks * 16;
    total.algebraic_add(dot_scalar(&weights[base..len], &codes[base..len]))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(weights: &[f32], codes: &[u8], len: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = len / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    // SAFETY (caller): `len <= weights.len()` and `len <= codes.len()`; each
    // 8-byte code load and 8-float weight load stays under `chunks * 32`.
    unsafe {
        for chunk in 0..chunks {
            let base = chunk * 32;
            let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                codes.as_ptr().add(base).cast(),
            )));
            let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                codes.as_ptr().add(base + 8).cast(),
            )));
            let c2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                codes.as_ptr().add(base + 16).cast(),
            )));
            let c3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                codes.as_ptr().add(base + 24).cast(),
            )));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(weights.as_ptr().add(base)), c0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(weights.as_ptr().add(base + 8)), c1, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(weights.as_ptr().add(base + 16)), c2, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(weights.as_ptr().add(base + 24)), c3, acc3);
        }
    }
    let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    let quad = _mm_add_ps(low, high);
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    let total = _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps::<0b01>(pair, pair)));
    let base = chunks * 32;
    total.algebraic_add(dot_scalar(&weights[base..len], &codes[base..len]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(weights: &[f32], codes: &[u8]) -> f64 {
        weights
            .iter()
            .zip(codes)
            .map(|(&weight, &code)| f64::from(weight) * f64::from(code))
            .sum()
    }

    #[test]
    fn resolved_quantized_dot_matches_f64_reference_for_every_tail() {
        let kernel = QuantizedDotKernel::resolve();
        for len in [
            0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 100, 768, 1024, 1027,
        ] {
            let weights: Vec<f32> = (0..len)
                .map(|index| ((index * 37 % 101) as f32 / 50.0 - 1.0) * 0.01)
                .collect();
            let codes: Vec<u8> = (0..len).map(|index| (index * 73 % 256) as u8).collect();
            let expected = reference(&weights, &codes);
            let got = kernel.dot(&weights, &codes);
            let scalar = dot_scalar(&weights, &codes);
            let tolerance = 1e-5 * (1.0 + expected.abs());
            assert!(
                (f64::from(got) - expected).abs() <= tolerance,
                "len {len}: kernel {got} vs reference {expected}"
            );
            assert!(
                (f64::from(scalar) - expected).abs() <= tolerance,
                "len {len}: scalar {scalar} vs reference {expected}"
            );
        }
    }

    #[test]
    fn quantized_dot_propagates_non_finite_weights() {
        let kernel = QuantizedDotKernel::resolve();
        let weights = vec![f32::NAN; 40];
        let codes = vec![1u8; 40];
        assert!(kernel.dot(&weights, &codes).is_nan());
    }
}
