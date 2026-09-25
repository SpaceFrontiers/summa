//! Conservative phrase admission over decoded posting blocks.

use super::Lengths;
use crate::query::bm25::PreparedBounds;
use crate::structures::BlockPostingIterator;
use crate::{DocId, Score};

/// First noncompetitive length per small TF: canonical singleton scores and
/// conservative inflated bounds for larger frequencies. The table is refreshed
/// only when the heap floor changes. An inverse seeds the search; authoritative
/// neighboring score checks establish every integer cutoff.
pub(super) struct CompetitiveLengths {
    pub(super) minimum: Score,
    pub(super) allow_equal: bool,
    pub(super) limits: [u32; 32],
}

impl CompetitiveLengths {
    pub(super) fn new(
        bounds: &PreparedBounds,
        minimum: Score,
        allow_equal: bool,
        singleton_score: impl Fn(u32) -> Score,
    ) -> Self {
        Self {
            minimum,
            allow_equal,
            limits: std::array::from_fn(|index| {
                let tf = index as u32 + 1;
                let competitive = |length| {
                    let score = if tf == 1 {
                        singleton_score(length)
                    } else {
                        bounds.pair(tf, length)
                    };
                    score > minimum || (allow_equal && score == minimum)
                };
                let mut low = 1;
                let mut high = u32::from(u16::MAX) + 1;
                if let Some(hint) = bounds.length_cutoff_hint(tf, minimum) {
                    if hint == high || !competitive(hint) {
                        if hint == 1 || competitive(hint - 1) {
                            return hint;
                        }
                        high = hint - 1;
                    } else {
                        low = hint + 1;
                    }
                }
                while low < high {
                    let middle = low + (high - low) / 2;
                    if competitive(middle) {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                low
            }),
        }
    }

    pub(super) fn find_candidate(
        &self,
        bounds: &PreparedBounds,
        lengths: &Lengths,
        postings: &mut BlockPostingIterator<'_>,
    ) -> Option<DocId> {
        let raw = match lengths {
            Lengths::Chunks(map) => Some((map.length_bytes(), map.length_floor().max(1))),
            Lengths::Docs(column) if !column.is_quantized() => Some((column.length_bytes(), 1)),
            _ => None,
        };
        if let Some((bytes, floor)) = raw {
            let block_minimum = postings
                .current_block_metadata()
                .and_then(|(list, block)| list.block_bounds(block))
                .and_then(|(_, minimum)| minimum)
                .unwrap_or(1);
            let floor = floor.max(block_minimum);
            let pairs = bytes.as_chunks::<2>().0;
            postings.find_in_block(|docs, tfs| {
                let length = |doc: u32| {
                    pairs.get(doc as usize).map_or_else(
                        || lengths.length(doc).max(1),
                        |&pair| u32::from(u16::from_le_bytes(pair)).max(floor),
                    )
                };
                #[cfg(target_arch = "x86_64")]
                if floor <= u32::from(u16::MAX)
                    && docs.last().is_some_and(|&doc| {
                        (doc as usize) < pairs.len().saturating_sub(1).min(i32::MAX as usize)
                    })
                    && std::arch::is_x86_feature_detected!("avx2")
                {
                    let mut start = 0;
                    while start < docs.len() {
                        // Sorted writer-produced IDs establish the gather's
                        // signed index range and its four-byte read extent.
                        // Both posting slices have the same bounded length.
                        let offset = unsafe {
                            if std::arch::is_x86_feature_detected!("avx512f") {
                                small_tf_candidates_avx512(
                                    &docs[start..],
                                    &tfs[start..],
                                    pairs,
                                    floor,
                                    &self.limits,
                                )
                            } else {
                                small_tf_candidates_avx2(
                                    &docs[start..],
                                    &tfs[start..],
                                    pairs,
                                    floor,
                                    &self.limits,
                                )
                            }
                        };
                        let at = start + offset;
                        if at == docs.len() {
                            return None;
                        }
                        if (1..=32).contains(&tfs[at])
                            || self.accepts(bounds, tfs[at], length(docs[at]))
                        {
                            return Some(at);
                        }
                        start = at + 1;
                    }
                    return None;
                }
                docs.iter().zip(tfs).position(|(&doc, &tf)| {
                    if tf
                        .checked_sub(1)
                        .and_then(|index| self.limits.get(index as usize))
                        .is_some_and(|&limit| limit <= floor)
                    {
                        return false;
                    }
                    self.accepts(bounds, tf, length(doc))
                })
            })
        } else {
            postings.find_in_block(|docs, tfs| {
                docs.iter()
                    .zip(tfs)
                    .position(|(&doc, &tf)| self.accepts(bounds, tf, lengths.length(doc).max(1)))
            })
        }
    }

    pub(super) fn accepts(&self, bounds: &PreparedBounds, tf: u32, length: u32) -> bool {
        if length <= u32::from(u16::MAX)
            && let Some(limit) = tf
                .checked_sub(1)
                .and_then(|index| self.limits.get(index as usize))
        {
            length < *limit
        } else {
            let score = bounds.pair(tf, length);
            score > self.minimum || (self.allow_equal && score == self.minimum)
        }
    }
}

/// Return the first possible candidate, or the suffix length. Frequencies
/// 1–32 use the exact cutoff table; larger/zero TFs retain scalar admission.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn small_tf_candidates_avx2(
    docs: &[u32],
    tfs: &[u32],
    lengths: &[[u8; 2]],
    floor: u32,
    limits: &[u32; 32],
) -> usize {
    use std::arch::x86_64::*;
    let mut base = 0;
    // SAFETY: the caller guarantees AVX2, equal slice lengths, sorted indices
    // below i32::MAX, and one extra u16 after the largest addressed norm. Thus
    // every four-byte gather stays inside the immutable length column.
    unsafe {
        let first_frequency = limits
            .iter()
            .position(|&limit| limit > floor)
            .unwrap_or(limits.len())
            + 1;
        let minimum_tf = _mm256_set1_epi32(first_frequency as i32 - 1);
        let ones = _mm256_set1_epi32(1);
        let mask16 = _mm256_set1_epi32(65535);
        let floor = _mm256_set1_epi32(floor as i32);
        let max_index = _mm256_set1_epi32(31);
        while base + 8 <= docs.len() {
            let frequencies = _mm256_loadu_si256(tfs.as_ptr().add(base).cast());
            // Unknown zero/large frequencies retain scalar admission. Reject
            // an entire common low-TF batch before either indexed gather.
            let possibly_large = _mm256_or_si256(
                _mm256_cmpgt_epi32(frequencies, minimum_tf),
                _mm256_cmpgt_epi32(ones, frequencies),
            );
            if _mm256_movemask_ps(_mm256_castsi256_ps(possibly_large)) == 0 {
                base += 8;
                continue;
            }
            let tf_indices = _mm256_sub_epi32(frequencies, ones);
            let indices = _mm256_min_epu32(tf_indices, max_index);
            let cutoffs = _mm256_i32gather_epi32::<4>(limits.as_ptr().cast(), indices);
            let small_tf = _mm256_cmpeq_epi32(indices, tf_indices);
            let outside = _mm256_xor_si256(small_tf, _mm256_set1_epi32(-1));
            let possible = _mm256_or_si256(_mm256_cmpgt_epi32(cutoffs, floor), outside);
            if _mm256_movemask_ps(_mm256_castsi256_ps(possible)) == 0 {
                base += 8;
                continue;
            }
            let ids = _mm256_loadu_si256(docs.as_ptr().add(base).cast());
            let gathered =
                _mm256_mask_i32gather_epi32::<2>(floor, lengths.as_ptr().cast(), ids, possible);
            let norms = _mm256_max_epi32(_mm256_and_si256(gathered, mask16), floor);
            let competitive = _mm256_cmpgt_epi32(cutoffs, norms);
            let eligible = _mm256_and_si256(_mm256_or_si256(competitive, outside), possible);
            let bits = _mm256_movemask_ps(_mm256_castsi256_ps(eligible));
            if bits != 0 {
                return base + bits.trailing_zeros() as usize;
            }
            base += 8;
        }
    }
    while base < docs.len() {
        let length = u32::from(u16::from_le_bytes(lengths[docs[base] as usize])).max(floor);
        if tfs[base]
            .checked_sub(1)
            .and_then(|index| limits.get(index as usize))
            .is_none_or(|&limit| length < limit)
        {
            break;
        }
        base += 1;
    }
    base
}

/// Sixteen-lane equivalent of the AVX2 kernel; short tails share that kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn small_tf_candidates_avx512(
    docs: &[u32],
    tfs: &[u32],
    lengths: &[[u8; 2]],
    floor: u32,
    limits: &[u32; 32],
) -> usize {
    use std::arch::x86_64::*;
    let mut base = 0;
    // SAFETY: identical slice, index and spare-u16 requirements to the AVX2
    // kernel; dispatch additionally establishes AVX-512F availability.
    unsafe {
        let minimum_tf = _mm512_set1_epi32(
            limits
                .iter()
                .position(|&limit| limit > floor)
                .unwrap_or(limits.len()) as i32,
        );
        let zero = _mm512_setzero_si512();
        let ones = _mm512_set1_epi32(1);
        let mask16 = _mm512_set1_epi32(65535);
        let floor_vector = _mm512_set1_epi32(floor as i32);
        let max_index = _mm512_set1_epi32(31);
        while base + 16 <= docs.len() {
            let frequencies = _mm512_loadu_si512(tfs.as_ptr().add(base).cast());
            let possible_tf = _mm512_cmpgt_epu32_mask(frequencies, minimum_tf)
                | _mm512_cmpeq_epi32_mask(frequencies, zero);
            if possible_tf == 0 {
                base += 16;
                continue;
            }
            let tf_indices = _mm512_sub_epi32(frequencies, ones);
            let indices = _mm512_min_epu32(tf_indices, max_index);
            let cutoffs = _mm512_i32gather_epi32::<4>(indices, limits.as_ptr().cast());
            let outside = !_mm512_cmpeq_epi32_mask(indices, tf_indices);
            let possible = _mm512_cmpgt_epu32_mask(cutoffs, floor_vector) | outside;
            if possible == 0 {
                base += 16;
                continue;
            }
            let ids = _mm512_loadu_si512(docs.as_ptr().add(base).cast());
            let gathered = _mm512_mask_i32gather_epi32::<2>(
                floor_vector,
                possible,
                ids,
                lengths.as_ptr().cast(),
            );
            let norms = _mm512_max_epu32(_mm512_and_si512(gathered, mask16), floor_vector);
            let eligible = (_mm512_cmpgt_epu32_mask(cutoffs, norms) | outside) & possible;
            if eligible != 0 {
                return base + eligible.trailing_zeros() as usize;
            }
            base += 16;
        }
        base + small_tf_candidates_avx2(&docs[base..], &tfs[base..], lengths, floor, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_seeded_cutoffs_match_binary_search_at_score_rounding_boundaries() {
        use crate::query::Bm25Params;
        for params in [
            Bm25Params::default(),
            Bm25Params { k1: 0.0, b: 1.0 },
            Bm25Params { k1: 7e20, b: 0.0 },
            Bm25Params { k1: 3.0, b: 1.0 },
        ] {
            for average in [0.0, 17.5, 10000.0] {
                let bounds = PreparedBounds::new(params, u32::MAX, 1.2345, average).unwrap();
                let score = |tf, length| {
                    if tf == 1 {
                        params.score(1.0, 1.2345, length as f32, average)
                    } else {
                        bounds.pair(tf, length)
                    }
                };
                for pivot_tf in [1, 2, 17, 32] {
                    for pivot_length in [1, 2, 255, 1023, 65535] {
                        let pivot = score(pivot_tf, pivot_length);
                        for minimum in [pivot.next_down(), pivot, pivot.next_up()] {
                            for allow_equal in [false, true] {
                                let table = CompetitiveLengths::new(
                                    &bounds,
                                    minimum,
                                    allow_equal,
                                    |length| score(1, length),
                                );
                                for tf in 1..=32 {
                                    let mut low = 1;
                                    let mut high = 65536;
                                    while low < high {
                                        let middle = low + (high - low) / 2;
                                        let value = score(tf, middle);
                                        if value > minimum || (allow_equal && value == minimum) {
                                            low = middle + 1;
                                        } else {
                                            high = middle;
                                        }
                                    }
                                    assert_eq!(
                                        table.limits[tf as usize - 1],
                                        low,
                                        "{params:?} avg={average} tf={tf} floor={minimum} equal={allow_equal}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vector_tf_admission_matches_scalar_for_every_lane_and_tail() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for count in 0..=129 {
            let docs: Vec<u32> = (0..count).map(|i| i * 3 + 1).collect();
            for first in 0..=count {
                let mut lengths = vec![65535u16.to_le_bytes(); count as usize * 3 + 4];
                let mut tfs = vec![1; count as usize];
                if first < count {
                    lengths[docs[first as usize] as usize] = 1u16.to_le_bytes();
                }
                for mode in 0..3 {
                    if first < count {
                        tfs[first as usize] = [1, 0, u32::MAX][mode];
                    }
                    for (floor, limit) in [(1, 0), (1, 1), (1, 2), (17, 16), (65535, 65536)] {
                        let limits = [limit; 32];
                        let expected = docs
                            .iter()
                            .zip(&tfs)
                            .position(|(&doc, &tf)| {
                                !(1..=32).contains(&tf)
                                    || u32::from(u16::from_le_bytes(lengths[doc as usize]))
                                        .max(floor)
                                        < limit
                            })
                            .unwrap_or(docs.len());
                        // The fixture has sorted positive indices, equal posting
                        // lengths and a spare u16 after every addressed value.
                        let actual = unsafe {
                            small_tf_candidates_avx2(&docs, &tfs, &lengths, floor, &limits)
                        };
                        assert_eq!(
                            actual, expected,
                            "count={count}, first={first}, mode={mode}"
                        );
                        if std::arch::is_x86_feature_detected!("avx512f") {
                            // Same fixture and bounds; this CPU also supports AVX2.
                            let wide = unsafe {
                                small_tf_candidates_avx512(&docs, &tfs, &lengths, floor, &limits)
                            };
                            assert_eq!(
                                wide, expected,
                                "wide count={count}, first={first}, mode={mode}"
                            );
                        }
                    }
                }
            }
        }
    }
}
