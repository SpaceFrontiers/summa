//! 32-lane ScaNN AH lookup scoring with scalar, AVX2, and NEON dispatch.
//!
//! Layout v2 (see `docs/fast-scan-layout-v2.md`): a complete 32-row group
//! stores two AH blocks per 32-byte word. Byte `lane` of word `j` holds the
//! code of block `2j` in its low nibble and block `2j + 1` in its high nibble
//! for row `lane`. An odd block count is padded with a zero block whose
//! lookup table is all zeros. Lookup tables are unsigned bytes with the
//! per-block minimum subtracted, so the full `0..=255` range carries signal;
//! the summed minimums are added back once per row.

use super::{AhQuery, CENTERS_PER_BLOCK, ScannFormatError, ScannResult};

pub const FAST_SCAN_LANES: usize = 32;

/// On-disk FastScan group layout revision carried by
/// `SCANN_SEGMENT_PAYLOAD_VERSION`. Readers refuse other revisions.
pub const FAST_SCAN_LAYOUT_VERSION: u16 = 2;

/// AH blocks stored per 32-byte word (low nibble / high nibble).
const BLOCKS_PER_WORD: usize = 2;

/// Number of blocks actually stored for `blocks` AH blocks: odd counts are
/// padded to the next block pair.
#[inline]
pub fn padded_blocks(blocks: usize) -> usize {
    blocks.div_ceil(BLOCKS_PER_WORD) * BLOCKS_PER_WORD
}

/// Bytes of one packed 32-row FastScan group for `blocks` AH blocks.
#[inline]
pub fn packed_block_bytes(blocks: usize) -> Option<usize> {
    blocks
        .div_ceil(BLOCKS_PER_WORD)
        .checked_mul(FAST_SCAN_LANES)
}

/// Transpose exactly 32 rows of unpacked 4-bit codes (row-major, `blocks`
/// codes per row) into the v2 block-pair-major FastScan layout.
pub fn pack_fast_scan_block(rows: &[u8], blocks: usize, output: &mut Vec<u8>) -> ScannResult<()> {
    if blocks == 0
        || rows.len() != FAST_SCAN_LANES * blocks
        || rows.iter().any(|&code| code as usize >= CENTERS_PER_BLOCK)
    {
        return Err(ScannFormatError::new("invalid ScaNN FastScan source block"));
    }
    let words = blocks.div_ceil(BLOCKS_PER_WORD);
    output.reserve(words * FAST_SCAN_LANES);
    for word in 0..words {
        let low_block = word * BLOCKS_PER_WORD;
        let high_block = low_block + 1;
        for lane in 0..FAST_SCAN_LANES {
            let low = rows[lane * blocks + low_block];
            let high = if high_block < blocks {
                rows[lane * blocks + high_block]
            } else {
                0
            };
            output.push(low | (high << 4));
        }
    }
    Ok(())
}

/// Byte offset and nibble position of `(row, block)` inside one packed
/// 32-row group. Shared by the compaction unpacker so the layout has exactly
/// one definition.
#[inline]
pub fn packed_code_position(row_in_group: usize, block: usize) -> (usize, bool) {
    debug_assert!(row_in_group < FAST_SCAN_LANES);
    (
        (block / BLOCKS_PER_WORD) * FAST_SCAN_LANES + row_in_group,
        !block.is_multiple_of(BLOCKS_PER_WORD),
    )
}

/// Accumulation kernel resolved once per query instead of once per 32-row
/// block. The float leaf scan calls `accumulate` for every block of every
/// probed run, so per-call `is_x86_feature_detected!` was pure overhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastScanKernel {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Scalar,
}

impl FastScanKernel {
    pub fn resolve() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            Self::Scalar
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }
}

/// Block pairs accumulated in 16-bit lanes before folding into 32-bit lanes.
/// Every lookup byte is `<= 255`, so one pair adds at most 510 per lane and
/// `64 * 510 = 32,640` stays inside `u16`; the fold happens once per 128
/// blocks instead of once per block.
#[cfg_attr(
    not(any(target_arch = "aarch64", target_arch = "x86_64")),
    allow(dead_code)
)]
const U16_FLUSH_WORDS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct FastScanQuery {
    blocks: usize,
    /// `padded_blocks(blocks) * 16` unsigned entries; padding blocks are zero.
    lookup: Vec<u8>,
    /// Sum of the per-block minimums removed from `lookup`, in score units.
    bias: f32,
    multiplier: f32,
    inverse_multiplier: f32,
    kernel: FastScanKernel,
    /// Every AH block had a constant lookup table, so every lane scores as
    /// its centroid dot plus `bias`. Reported instead of silently using a
    /// unit multiplier.
    degenerate: bool,
}

impl FastScanQuery {
    pub fn new(query: &AhQuery) -> Self {
        let blocks = query.blocks();
        let values = query.values();
        let mut minimums = Vec::with_capacity(blocks);
        let mut widest_range = 0.0f32;
        for table in values.chunks_exact(CENTERS_PER_BLOCK) {
            let (minimum, maximum) = table
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
                    (low.min(value), high.max(value))
                });
            minimums.push(minimum);
            widest_range = widest_range.max(maximum - minimum);
        }
        // `AhQuery` values are validated finite, so a non-positive range means
        // every block table is constant.
        let degenerate = widest_range <= f32::EPSILON;
        let multiplier = if degenerate {
            1.0
        } else {
            255.0 / widest_range
        };
        let mut lookup = vec![0u8; padded_blocks(blocks) * CENTERS_PER_BLOCK];
        for (block, table) in values.chunks_exact(CENTERS_PER_BLOCK).enumerate() {
            let minimum = minimums[block];
            for (slot, &value) in lookup[block * CENTERS_PER_BLOCK..].iter_mut().zip(table) {
                *slot = ((value - minimum) * multiplier).round().clamp(0.0, 255.0) as u8;
            }
        }
        // Bias in f64: it is the exact sum of up to thousands of minimums and
        // is added back to every row score.
        let bias = minimums
            .iter()
            .fold(0.0f64, |acc, &minimum| acc + f64::from(minimum)) as f32;
        Self {
            blocks,
            lookup,
            bias,
            multiplier,
            inverse_multiplier: multiplier.recip(),
            kernel: FastScanKernel::resolve(),
            degenerate,
        }
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Bytes of one packed 32-row group under this query's block count.
    pub fn packed_block_bytes(&self) -> usize {
        self.lookup.len() / CENTERS_PER_BLOCK / BLOCKS_PER_WORD * FAST_SCAN_LANES
    }

    /// True when every AH table was constant (see `degenerate`).
    pub fn is_degenerate(&self) -> bool {
        self.degenerate
    }

    /// Integer lane sums for one packed 32-row group; no float conversion.
    #[inline]
    pub fn accumulate_block(
        &self,
        codes: &[u8],
        scores: &mut [i32; FAST_SCAN_LANES],
    ) -> ScannResult<()> {
        if codes.len() != self.packed_block_bytes() {
            return Err(ScannFormatError::new(
                "invalid ScaNN FastScan encoded block",
            ));
        }
        accumulate(self, codes, scores);
        Ok(())
    }

    /// Convert one integer lane sum into the approximate dot product.
    #[inline]
    pub fn lane_score(&self, lane: i32, centroid_dot: f32) -> f32 {
        centroid_dot
            .algebraic_add(self.bias)
            .algebraic_add((lane as f32).algebraic_mul(self.inverse_multiplier))
    }

    /// Largest integer lane sum that can *not* beat `threshold` for a row in a
    /// leaf with `centroid_dot`. Lanes `<= cutoff` may be skipped before any
    /// float work; the bound is conservative by one integer step so rounding
    /// in `multiplier * inverse_multiplier` can never drop a competitive row.
    #[inline]
    pub fn lane_cutoff(&self, threshold: f32, centroid_dot: f32) -> i32 {
        let exact = ((threshold - centroid_dot - self.bias) * self.multiplier).floor() - 1.0;
        if exact >= i32::MAX as f32 {
            i32::MAX
        } else if exact <= i32::MIN as f32 {
            i32::MIN
        } else {
            exact as i32
        }
    }

    pub fn score_block(
        &self,
        codes: &[u8],
        centroid_dot: f32,
    ) -> ScannResult<[f32; FAST_SCAN_LANES]> {
        let mut integer_scores = [0i32; FAST_SCAN_LANES];
        self.accumulate_block(codes, &mut integer_scores)?;
        Ok(integer_scores.map(|score| self.lane_score(score, centroid_dot)))
    }

    #[inline]
    fn words(&self) -> usize {
        self.lookup.len() / CENTERS_PER_BLOCK / BLOCKS_PER_WORD
    }
}

#[inline]
fn accumulate(query: &FastScanQuery, codes: &[u8], scores: &mut [i32; FAST_SCAN_LANES]) {
    match query.kernel {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline for aarch64 and all 16-byte loads are
        // bounded by the validated group shape (`codes.len() == words * 32`,
        // `lookup.len() == words * 32`).
        FastScanKernel::Neon => unsafe { accumulate_neon(query, codes, scores) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the kernel was resolved by runtime feature detection. Loads
        // are unaligned and bounded by the validated group shape.
        FastScanKernel::Avx2 => unsafe { accumulate_avx2(query, codes, scores) },
        FastScanKernel::Scalar => accumulate_scalar(query, codes, scores),
    }
}

fn accumulate_scalar(query: &FastScanQuery, codes: &[u8], scores: &mut [i32; FAST_SCAN_LANES]) {
    scores.fill(0);
    for word in 0..query.words() {
        let tables =
            &query.lookup[word * 2 * CENTERS_PER_BLOCK..(word + 1) * 2 * CENTERS_PER_BLOCK];
        let (low_table, high_table) = tables.split_at(CENTERS_PER_BLOCK);
        let packed = &codes[word * FAST_SCAN_LANES..(word + 1) * FAST_SCAN_LANES];
        for (lane, &byte) in packed.iter().enumerate() {
            scores[lane] += i32::from(low_table[(byte & 0x0f) as usize])
                + i32::from(high_table[(byte >> 4) as usize]);
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

    unsafe {
        let mask = vdupq_n_u8(0x0f);
        let mut acc32 = [vdupq_n_u32(0); 8];
        let words = query.words();
        let mut word = 0usize;
        while word < words {
            let flush_end = (word + U16_FLUSH_WORDS).min(words);
            // Four u16x8 accumulators cover rows 0..7, 8..15, 16..23, 24..31.
            let mut acc16 = [vdupq_n_u16(0); 4];
            for inner in word..flush_end {
                let table_base = query.lookup.as_ptr().add(inner * 2 * CENTERS_PER_BLOCK);
                let low_table = vld1q_u8(table_base);
                let high_table = vld1q_u8(table_base.add(CENTERS_PER_BLOCK));
                let code_base = codes.as_ptr().add(inner * FAST_SCAN_LANES);
                let first = vld1q_u8(code_base);
                let second = vld1q_u8(code_base.add(16));
                let first_low = vqtbl1q_u8(low_table, vandq_u8(first, mask));
                let first_high = vqtbl1q_u8(high_table, vshrq_n_u8::<4>(first));
                let second_low = vqtbl1q_u8(low_table, vandq_u8(second, mask));
                let second_high = vqtbl1q_u8(high_table, vshrq_n_u8::<4>(second));
                acc16[0] = vaddw_u8(acc16[0], vget_low_u8(first_low));
                acc16[0] = vaddw_u8(acc16[0], vget_low_u8(first_high));
                acc16[1] = vaddw_high_u8(acc16[1], first_low);
                acc16[1] = vaddw_high_u8(acc16[1], first_high);
                acc16[2] = vaddw_u8(acc16[2], vget_low_u8(second_low));
                acc16[2] = vaddw_u8(acc16[2], vget_low_u8(second_high));
                acc16[3] = vaddw_high_u8(acc16[3], second_low);
                acc16[3] = vaddw_high_u8(acc16[3], second_high);
            }
            for chunk in 0..4 {
                acc32[chunk * 2] = vaddw_u16(acc32[chunk * 2], vget_low_u16(acc16[chunk]));
                acc32[chunk * 2 + 1] = vaddw_high_u16(acc32[chunk * 2 + 1], acc16[chunk]);
            }
            word = flush_end;
        }
        for (chunk, acc) in acc32.iter().enumerate() {
            vst1q_s32(
                scores.as_mut_ptr().add(chunk * 4),
                vreinterpretq_s32_u32(*acc),
            );
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

    unsafe {
        let mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        // Rows 0..7, 8..15, 16..23, 24..31 as u32x8 accumulators.
        let mut acc32 = [zero; 4];
        let words = query.words();
        let mut word = 0usize;
        while word < words {
            let flush_end = (word + U16_FLUSH_WORDS).min(words);
            // Rows 0..15 and 16..31 as u16x16 accumulators.
            let mut acc16 = [zero; 2];
            for inner in word..flush_end {
                let table_base = query.lookup.as_ptr().add(inner * 2 * CENTERS_PER_BLOCK);
                let low_table = _mm256_broadcastsi128_si256(_mm_loadu_si128(table_base.cast()));
                let high_table = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    table_base.add(CENTERS_PER_BLOCK).cast(),
                ));
                let packed = _mm256_loadu_si256(codes.as_ptr().add(inner * FAST_SCAN_LANES).cast());
                // One 256-bit shuffle covers all 32 rows of one block.
                let low = _mm256_shuffle_epi8(low_table, _mm256_and_si256(packed, mask));
                let high = _mm256_shuffle_epi8(
                    high_table,
                    _mm256_and_si256(_mm256_srli_epi16::<4>(packed), mask),
                );
                acc16[0] = _mm256_add_epi16(
                    acc16[0],
                    _mm256_add_epi16(
                        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(low)),
                        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(high)),
                    ),
                );
                acc16[1] = _mm256_add_epi16(
                    acc16[1],
                    _mm256_add_epi16(
                        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(low)),
                        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(high)),
                    ),
                );
            }
            for half in 0..2 {
                acc32[half * 2] = _mm256_add_epi32(
                    acc32[half * 2],
                    _mm256_cvtepu16_epi32(_mm256_castsi256_si128(acc16[half])),
                );
                acc32[half * 2 + 1] = _mm256_add_epi32(
                    acc32[half * 2 + 1],
                    _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(acc16[half])),
                );
            }
            word = flush_end;
        }
        for (chunk, acc) in acc32.iter().enumerate() {
            _mm256_storeu_si256(scores.as_mut_ptr().add(chunk * 8).cast(), *acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::vector::scann::AhCodebook;

    fn synthetic_vectors(count: usize, dim: usize, seed: usize) -> Vec<f32> {
        (0..count * dim)
            .map(|value| (((value + seed) * 7919) % 1009) as f32 / 504.5 - 1.0)
            .collect()
    }

    fn encoded_rows(codebook: &AhCodebook, data: &[f32], dim: usize) -> Vec<u8> {
        let mut rows = Vec::with_capacity(FAST_SCAN_LANES * codebook.blocks());
        for vector in data.chunks_exact(dim).take(FAST_SCAN_LANES) {
            let mut codes = vec![0u8; codebook.blocks()];
            codebook.encode(vector, vector, 0.2, &mut codes).unwrap();
            rows.extend(codes);
        }
        rows
    }

    fn packed_rows(codebook: &AhCodebook, data: &[f32], dim: usize) -> Vec<u8> {
        let rows = encoded_rows(codebook, data, dim);
        let mut packed = Vec::new();
        pack_fast_scan_block(&rows, codebook.blocks(), &mut packed).unwrap();
        packed
    }

    #[test]
    fn fast_scan_dispatch_matches_scalar_scores() {
        let data: Vec<f32> = (0..64 * 8)
            .map(|value| (value % 37) as f32 / 20.0 - 0.9)
            .collect();
        let codebook = AhCodebook::train(&data, 64, 8, 2, 4, 73).unwrap();
        let query = codebook.query_dot_product(&data[..8]).unwrap();
        let fast_query = FastScanQuery::new(&query);
        let packed = packed_rows(&codebook, &data, 8);
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

    /// An odd block count pads a zero block into the last word; the padding
    /// must contribute nothing and every kernel must agree.
    #[test]
    fn fast_scan_odd_block_count_pads_a_zero_block() {
        let dim = 10; // 5 blocks of 2 dims
        let data = synthetic_vectors(64, dim, 3);
        let codebook = AhCodebook::train(&data, 64, dim, 2, 3, 11).unwrap();
        assert_eq!(codebook.blocks(), 5);
        assert_eq!(packed_block_bytes(5), Some(3 * FAST_SCAN_LANES));
        let query = codebook.query_dot_product(&data[..dim]).unwrap();
        let fast_query = FastScanQuery::new(&query);
        assert_eq!(fast_query.packed_block_bytes(), 96);
        let rows = encoded_rows(&codebook, &data, dim);
        let packed = packed_rows(&codebook, &data, dim);
        assert!(packed[64..].iter().all(|byte| byte >> 4 == 0));
        let mut scalar = [0i32; FAST_SCAN_LANES];
        accumulate_scalar(&fast_query, &packed, &mut scalar);
        let mut dispatched = [0i32; FAST_SCAN_LANES];
        accumulate(&fast_query, &packed, &mut dispatched);
        assert_eq!(dispatched, scalar);
        // Every lane must reproduce the exact f32 AH score to within the
        // 8-bit table quantization.
        let scores = fast_query.score_block(&packed, 0.5).unwrap();
        for (lane, row) in rows.chunks_exact(codebook.blocks()).enumerate() {
            let exact = query.score_unpacked(row, 0.5).unwrap();
            assert!(
                (scores[lane] - exact).abs() <= 5.0 * fast_query.inverse_multiplier,
                "lane {lane}: fast {} vs exact {exact}",
                scores[lane]
            );
        }
    }

    /// The 16-bit accumulators fold into 32-bit lanes every 64 words; a
    /// query with more words than one flush window and saturated table
    /// entries must still match the scalar reference exactly.
    #[test]
    fn fast_scan_u16_accumulation_matches_scalar_past_flush_window() {
        let dim = 2 * (U16_FLUSH_WORDS * 2 * 2 + 37);
        let data = synthetic_vectors(64, dim, 0);
        let codebook = AhCodebook::train(&data, 64, dim, 2, 2, 5).unwrap();
        assert!(codebook.blocks() > U16_FLUSH_WORDS * 2 * 2);
        let query = codebook.query_dot_product(&data[..dim]).unwrap();
        let fast_query = FastScanQuery::new(&query);
        // The aligned query saturates entries at 255 for its own centers.
        assert!(fast_query.lookup.contains(&255));
        let packed = packed_rows(&codebook, &data, dim);
        let mut scalar = [0i32; FAST_SCAN_LANES];
        accumulate_scalar(&fast_query, &packed, &mut scalar);
        let mut dispatched = [0i32; FAST_SCAN_LANES];
        accumulate(&fast_query, &packed, &mut dispatched);
        assert_eq!(dispatched, scalar);
    }

    /// FastScan is the only leaf scorer for complete groups, so its error
    /// against the exact f32 AH score is bounded by the table quantization:
    /// at most half a step per block, added over `blocks` blocks.
    #[test]
    fn fast_scan_scores_track_exact_ah_scores_within_table_quantization() {
        for dim in [16usize, 96, 384] {
            let data = synthetic_vectors(64, dim, dim);
            let codebook = AhCodebook::train(&data, 64, dim, 2, 3, 17).unwrap();
            let query = codebook.query_dot_product(&data[dim..2 * dim]).unwrap();
            let fast_query = FastScanQuery::new(&query);
            let rows = encoded_rows(&codebook, &data, dim);
            let packed = packed_rows(&codebook, &data, dim);
            let scores = fast_query.score_block(&packed, 0.0).unwrap();
            let bound = 0.5 * codebook.blocks() as f32 * fast_query.inverse_multiplier + 1e-4;
            for (lane, row) in rows.chunks_exact(codebook.blocks()).enumerate() {
                let exact = query.score_unpacked(row, 0.0).unwrap();
                assert!(
                    (scores[lane] - exact).abs() <= bound,
                    "dim {dim} lane {lane}: fast {} vs exact {exact} (bound {bound})",
                    scores[lane]
                );
            }
        }
    }

    /// The per-block bias lets every table use the full `0..=255` range. When
    /// some block's scores span both signs symmetrically the v2 step equals
    /// the v1 signed `[-127, 127]` step, so the two quantizations are equally
    /// accurate and differ only by rounding luck; v2 must never be
    /// systematically worse than v1 (emulated here). Run with `--nocapture`
    /// to see the measured worst-case errors per dimension.
    #[test]
    fn fast_scan_v2_bias_tables_track_v1_signed_table_accuracy() {
        let mut total_v1 = 0.0f64;
        let mut total_v2 = 0.0f64;
        for dim in [16usize, 96, 384, 768] {
            let data = synthetic_vectors(64, dim, dim + 1);
            let codebook = AhCodebook::train(&data, 64, dim, 2, 3, 23).unwrap();
            let query = codebook.query_dot_product(&data[dim..2 * dim]).unwrap();
            let fast_query = FastScanQuery::new(&query);
            let rows = encoded_rows(&codebook, &data, dim);
            let packed = packed_rows(&codebook, &data, dim);
            let v2_scores = fast_query.score_block(&packed, 0.0).unwrap();
            // v1 emulation: one global signed multiplier, no per-block bias.
            let maximum = query
                .values()
                .iter()
                .fold(0.0f32, |current, &value| current.max(value.abs()));
            let v1_multiplier = 127.0 / maximum;
            let v1_table: Vec<i32> = query
                .values()
                .iter()
                .map(|&value| (value * v1_multiplier).round().clamp(-127.0, 127.0) as i32)
                .collect();
            let mut worst_v1 = 0.0f32;
            let mut worst_v2 = 0.0f32;
            for (lane, row) in rows.chunks_exact(codebook.blocks()).enumerate() {
                let exact = query.score_unpacked(row, 0.0).unwrap();
                let v1_sum: i32 = row
                    .iter()
                    .enumerate()
                    .map(|(block, &code)| v1_table[block * CENTERS_PER_BLOCK + code as usize])
                    .sum();
                let v1 = v1_sum as f32 / v1_multiplier;
                worst_v1 = worst_v1.max((v1 - exact).abs());
                worst_v2 = worst_v2.max((v2_scores[lane] - exact).abs());
            }
            println!("dim {dim}: max |error| v1 {worst_v1:.6} v2 {worst_v2:.6}");
            total_v1 += f64::from(worst_v1);
            total_v2 += f64::from(worst_v2);
            // One v2 table step of slack absorbs rounding luck on 32 rows.
            assert!(
                worst_v2 <= worst_v1 * 1.5 + fast_query.inverse_multiplier,
                "dim {dim}: v2 error {worst_v2} is systematically worse than v1 error {worst_v1}"
            );
        }
        assert!(
            total_v2 <= total_v1 * 1.25,
            "v2 tables lost accuracy overall: {total_v2} vs {total_v1}"
        );
    }

    #[test]
    fn fast_scan_reports_degenerate_constant_lookup_tables() {
        let dim = 8;
        let data: Vec<f32> = (0..64 * dim)
            .map(|value| (value % 37) as f32 / 20.0 - 0.9)
            .collect();
        let codebook = AhCodebook::train(&data, 64, dim, 2, 4, 73).unwrap();
        let live = FastScanQuery::new(&codebook.query_dot_product(&data[..dim]).unwrap());
        assert!(!live.is_degenerate());
        let zero = FastScanQuery::new(&codebook.query_dot_product(&vec![0.0; dim]).unwrap());
        assert!(zero.is_degenerate());
        assert!(zero.lookup.iter().all(|&entry| entry == 0));
    }

    /// Any lane at or below `lane_cutoff` must score at or below the
    /// threshold, so pruning on the integer sum can never drop a candidate
    /// that the full float comparison would have kept.
    #[test]
    fn fast_scan_lane_cutoff_is_conservative() {
        let dim = 8;
        let data: Vec<f32> = (0..64 * dim)
            .map(|value| (value % 37) as f32 / 20.0 - 0.9)
            .collect();
        let codebook = AhCodebook::train(&data, 64, dim, 2, 4, 73).unwrap();
        let fast_query = FastScanQuery::new(&codebook.query_dot_product(&data[..dim]).unwrap());
        for centroid_dot in [-0.75f32, 0.0, 0.3125, 2.5] {
            for threshold in [-1.0f32, -0.1, 0.0, 0.05, 0.4, 3.0] {
                let cutoff = fast_query.lane_cutoff(threshold, centroid_dot);
                assert!(
                    fast_query.lane_score(cutoff, centroid_dot) <= threshold,
                    "cutoff {cutoff} beats threshold {threshold} at centroid dot {centroid_dot}"
                );
                // Never over-prune by more than a couple of integer steps.
                assert!(
                    fast_query.lane_score(cutoff.saturating_add(3), centroid_dot) > threshold,
                    "cutoff {cutoff} is too loose for threshold {threshold}"
                );
            }
        }
    }

    #[test]
    fn packed_code_position_round_trips_through_pack() {
        let blocks = 7usize;
        let rows: Vec<u8> = (0..FAST_SCAN_LANES * blocks)
            .map(|index| (index * 5 % 16) as u8)
            .collect();
        let mut packed = Vec::new();
        pack_fast_scan_block(&rows, blocks, &mut packed).unwrap();
        assert_eq!(packed.len(), packed_block_bytes(blocks).unwrap());
        for row in 0..FAST_SCAN_LANES {
            for block in 0..blocks {
                let (offset, high) = packed_code_position(row, block);
                let byte = packed[offset];
                let code = if high { byte >> 4 } else { byte & 0x0f };
                assert_eq!(code, rows[row * blocks + block], "row {row} block {block}");
            }
        }
    }
}
