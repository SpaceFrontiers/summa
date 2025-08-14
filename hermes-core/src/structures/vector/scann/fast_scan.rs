//! 32-lane ScaNN AH lookup scoring with scalar, AVX2, and NEON dispatch.

use super::{AhQuery, CENTERS_PER_BLOCK, ScannFormatError, ScannResult};

pub const FAST_SCAN_LANES: usize = 32;

/// Transpose exactly 32 rows of unpacked 4-bit codes into block-major FastScan
/// layout (16 bytes per AH block).
pub fn pack_fast_scan_block(rows: &[u8], blocks: usize, output: &mut Vec<u8>) -> ScannResult<()> {
    if blocks == 0
        || rows.len() != FAST_SCAN_LANES * blocks
        || rows.iter().any(|&code| code as usize >= CENTERS_PER_BLOCK)
    {
        return Err(ScannFormatError::new("invalid ScaNN FastScan source block"));
    }
    output.reserve(blocks * FAST_SCAN_LANES / 2);
    for block in 0..blocks {
        for pair in 0..FAST_SCAN_LANES / 2 {
            let low = rows[(pair * 2) * blocks + block];
            let high = rows[(pair * 2 + 1) * blocks + block];
            output.push(low | (high << 4));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct FastScanQuery {
    blocks: usize,
    lookup: Vec<i8>,
    inverse_multiplier: f32,
}

impl FastScanQuery {
    pub fn new(query: &AhQuery) -> Self {
        let maximum = query
            .values()
            .iter()
            .fold(0.0f32, |current, &value| current.max(value.abs()));
        let multiplier = if maximum > f32::EPSILON {
            127.0 / maximum
        } else {
            1.0
        };
        Self {
            blocks: query.blocks(),
            lookup: query
                .values()
                .iter()
                .map(|&value| (value * multiplier).round().clamp(-127.0, 127.0) as i8)
                .collect(),
            inverse_multiplier: multiplier.recip(),
        }
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub fn score_block(
        &self,
        codes: &[u8],
        centroid_dot: f32,
    ) -> ScannResult<[f32; FAST_SCAN_LANES]> {
        if codes.len() != self.blocks * FAST_SCAN_LANES / 2 {
            return Err(ScannFormatError::new(
                "invalid ScaNN FastScan encoded block",
            ));
        }
        let mut integer_scores = [0i32; FAST_SCAN_LANES];
        accumulate(self, codes, &mut integer_scores);
        Ok(integer_scores.map(|score| centroid_dot + score as f32 * self.inverse_multiplier))
    }
}

fn accumulate(query: &FastScanQuery, codes: &[u8], scores: &mut [i32; FAST_SCAN_LANES]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline for aarch64 and all 16-byte loads are
        // bounded by the validated block shape.
        unsafe { accumulate_neon(query, codes, scores) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by runtime feature detection. Loads are unaligned
        // and stay within validated 16-byte tables/code blocks.
        unsafe { accumulate_avx2(query, codes, scores) };
        return;
    }
    #[allow(unreachable_code)]
    accumulate_scalar(query, codes, scores);
}

fn accumulate_scalar(query: &FastScanQuery, codes: &[u8], scores: &mut [i32; FAST_SCAN_LANES]) {
    scores.fill(0);
    for block in 0..query.blocks {
        let table = &query.lookup[block * CENTERS_PER_BLOCK..(block + 1) * CENTERS_PER_BLOCK];
        let packed = &codes[block * 16..(block + 1) * 16];
        for (pair, &byte) in packed.iter().enumerate() {
            scores[pair * 2] += i32::from(table[(byte & 0x0f) as usize]);
            scores[pair * 2 + 1] += i32::from(table[(byte >> 4) as usize]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_neon(
    query: &FastScanQuery,
    codes: &[u8],
    scores: &mut [i32; FAST_SCAN_LANES],
) {
    use std::arch::aarch64::*;

    scores.fill(0);
    unsafe {
        let mask = vdupq_n_u8(0x0f);
        let zero = vdupq_n_s32(0);
        let mut low_acc = [zero; 4];
        let mut high_acc = [zero; 4];
        for block in 0..query.blocks {
            let table = vld1q_s8(query.lookup.as_ptr().add(block * CENTERS_PER_BLOCK));
            let packed = vld1q_u8(codes.as_ptr().add(block * 16));
            let low = vqtbl1q_s8(table, vandq_u8(packed, mask));
            let high = vqtbl1q_s8(table, vshrq_n_u8::<4>(packed));
            let low_lo = vmovl_s8(vget_low_s8(low));
            let low_hi = vmovl_high_s8(low);
            let high_lo = vmovl_s8(vget_low_s8(high));
            let high_hi = vmovl_high_s8(high);
            low_acc[0] = vaddq_s32(low_acc[0], vmovl_s16(vget_low_s16(low_lo)));
            low_acc[1] = vaddq_s32(low_acc[1], vmovl_high_s16(low_lo));
            low_acc[2] = vaddq_s32(low_acc[2], vmovl_s16(vget_low_s16(low_hi)));
            low_acc[3] = vaddq_s32(low_acc[3], vmovl_high_s16(low_hi));
            high_acc[0] = vaddq_s32(high_acc[0], vmovl_s16(vget_low_s16(high_lo)));
            high_acc[1] = vaddq_s32(high_acc[1], vmovl_high_s16(high_lo));
            high_acc[2] = vaddq_s32(high_acc[2], vmovl_s16(vget_low_s16(high_hi)));
            high_acc[3] = vaddq_s32(high_acc[3], vmovl_high_s16(high_hi));
        }
        let mut low_values = [0i32; 16];
        let mut high_values = [0i32; 16];
        for chunk in 0..4 {
            vst1q_s32(low_values.as_mut_ptr().add(chunk * 4), low_acc[chunk]);
            vst1q_s32(high_values.as_mut_ptr().add(chunk * 4), high_acc[chunk]);
        }
        for lane in 0..16 {
            scores[lane * 2] = low_values[lane];
            scores[lane * 2 + 1] = high_values[lane];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(
    query: &FastScanQuery,
    codes: &[u8],
    scores: &mut [i32; FAST_SCAN_LANES],
) {
    use std::arch::x86_64::*;

    scores.fill(0);
    unsafe {
        let mask = _mm_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut low_acc = [zero; 2];
        let mut high_acc = [zero; 2];
        for block in 0..query.blocks {
            let table =
                _mm_loadu_si128(query.lookup.as_ptr().add(block * CENTERS_PER_BLOCK).cast());
            let packed = _mm_loadu_si128(codes.as_ptr().add(block * 16).cast());
            let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
            let high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16::<4>(packed), mask));
            let low_wide = _mm256_cvtepi8_epi16(low);
            let high_wide = _mm256_cvtepi8_epi16(high);
            low_acc[0] = _mm256_add_epi32(
                low_acc[0],
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(low_wide)),
            );
            low_acc[1] = _mm256_add_epi32(
                low_acc[1],
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(low_wide)),
            );
            high_acc[0] = _mm256_add_epi32(
                high_acc[0],
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(high_wide)),
            );
            high_acc[1] = _mm256_add_epi32(
                high_acc[1],
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(high_wide)),
            );
        }
        let mut low_values = [0i32; 16];
        let mut high_values = [0i32; 16];
        _mm256_storeu_si256(low_values.as_mut_ptr().cast(), low_acc[0]);
        _mm256_storeu_si256(low_values.as_mut_ptr().add(8).cast(), low_acc[1]);
        _mm256_storeu_si256(high_values.as_mut_ptr().cast(), high_acc[0]);
        _mm256_storeu_si256(high_values.as_mut_ptr().add(8).cast(), high_acc[1]);
        for lane in 0..16 {
            scores[lane * 2] = low_values[lane];
            scores[lane * 2 + 1] = high_values[lane];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::vector::scann::AhCodebook;

    #[test]
    fn fast_scan_dispatch_matches_scalar_scores() {
        let data: Vec<f32> = (0..64 * 8)
            .map(|value| (value % 37) as f32 / 20.0 - 0.9)
            .collect();
        let codebook = AhCodebook::train(&data, 64, 8, 2, 4, 73).unwrap();
        let query = codebook.query_dot_product(&data[..8]).unwrap();
        let fast_query = FastScanQuery::new(&query);
        let mut rows = Vec::with_capacity(FAST_SCAN_LANES * codebook.blocks());
        for vector in data.chunks_exact(8).take(FAST_SCAN_LANES) {
            let mut codes = vec![0u8; codebook.blocks()];
            codebook.encode(vector, vector, 0.2, &mut codes).unwrap();
            rows.extend(codes);
        }
        let mut packed = Vec::new();
        pack_fast_scan_block(&rows, codebook.blocks(), &mut packed).unwrap();
        let mut scalar = [0i32; FAST_SCAN_LANES];
        accumulate_scalar(&fast_query, &packed, &mut scalar);
        let mut dispatched = [0i32; FAST_SCAN_LANES];
        accumulate(&fast_query, &packed, &mut dispatched);
        assert_eq!(dispatched, scalar);
        assert!(
            fast_query
                .score_block(&packed, 0.0)
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
