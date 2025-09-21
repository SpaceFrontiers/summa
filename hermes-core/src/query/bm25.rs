//! BM25/BM25F scoring constants and utilities
//!
//! Shared BM25 parameters used across full-text scoring implementations.
//! All posting list formats and scoring executors should use these functions.

/// BM25 k1 parameter - controls term frequency saturation
/// Higher values give more weight to term frequency
pub const BM25_K1: f32 = 1.2;

/// BM25 b parameter - controls length normalization
/// 0 = no length normalization, 1 = full normalization
pub const BM25_B: f32 = 0.75;

/// Query-local normalization for byte norms, using the canonical arithmetic.
/// One KiB per active quantized text cursor; no index-sized decoded cache.
pub(super) struct NormTable([f32; 256]);

impl NormTable {
    pub(super) fn new(params: Bm25Params, average: f32) -> Self {
        crate::observe::search_work!(norm_tables += 1);
        Self(std::array::from_fn(|code| {
            let length = crate::segment::norms::decode(code as u8) as f32;
            params.k1 * (1.0 - params.b + params.b * (length / average.max(1.0)))
        }))
    }

    /// Gather query-normalized lengths before scoring contiguous frequencies.
    /// The scratch is bounded by one posting block, including short tails.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_batch(
        &self,
        params: Bm25Params,
        idf: f32,
        average: f32,
        boost: f32,
        codes: impl ExactSizeIterator<Item = u8>,
        tfs: &[u32],
        scores: &mut [f32],
    ) {
        debug_assert_eq!(tfs.len(), scores.len());
        debug_assert_eq!(tfs.len(), codes.len());
        if boost == 0.0 {
            scores.fill(0.0);
            return;
        }
        let mut norms = [0.0; crate::structures::postings::POSTING_BLOCK_SIZE];
        for ((norm, &tf), code) in norms[..tfs.len()].iter_mut().zip(tfs).zip(codes) {
            *norm = if code == 0 {
                params.k1 * (1.0 - params.b + params.b * (tf as f32 / average.max(1.0)))
            } else {
                self.0[usize::from(code)]
            };
        }
        for ((score, &tf), &norm) in scores.iter_mut().zip(tfs).zip(&norms) {
            *score = if tf == 0 {
                0.0
            } else {
                let boosted = tf as f32 * boost;
                idf * ((boosted * (params.k1 + 1.0)) / (boosted + norm))
            };
        }
    }

    #[cfg(test)]
    pub(super) fn score(
        &self,
        params: Bm25Params,
        tf: f32,
        idf: f32,
        code: u8,
        average: f32,
    ) -> f32 {
        if code == 0 {
            return params.score(tf, idf, tf, average);
        }
        if tf == 0.0 {
            return 0.0;
        }
        idf * ((tf * (params.k1 + 1.0)) / (tf + self.0[usize::from(code)]))
    }

    pub(super) fn score_boosted(
        &self,
        params: Bm25Params,
        tf: f32,
        idf: f32,
        code: u8,
        average: f32,
        boost: f32,
    ) -> f32 {
        if code == 0 {
            return params.score_boosted(tf, idf, tf, average, boost);
        }
        if tf == 0.0 || boost == 0.0 {
            return 0.0;
        }
        idf * ((tf * boost * (params.k1 + 1.0)) / (tf * boost + self.0[usize::from(code)]))
    }
}

/// Per-field BM25 parameters (`indexed<k1: ..., b: ...>` in the schema).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: BM25_K1,
            b: BM25_B,
        }
    }
}

impl Bm25Params {
    /// Resolve a field's parameters from its schema entry.
    pub fn for_field(schema: &crate::dsl::Schema, field: crate::dsl::Field) -> Self {
        let entry = schema.get_field_entry(field);
        Self {
            k1: entry.and_then(|e| e.bm25_k1).unwrap_or(BM25_K1),
            b: entry.and_then(|e| e.bm25_b).unwrap_or(BM25_B),
        }
    }

    /// BM25 score of one term occurrence set.
    #[inline]
    pub fn score(self, tf: f32, idf: f32, doc_len: f32, avg_doc_len: f32) -> f32 {
        // Zero-frequency postings retain membership but contribute no score,
        // including k1=0 and the zero-length b=1 boundary (otherwise 0/0).
        if tf == 0.0 {
            return 0.0;
        }
        let length_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len.max(1.0));
        let tf_norm = (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm);
        idf * tf_norm
    }

    /// BM25F score with a field boost.
    #[inline]
    pub fn score_boosted(
        self,
        tf: f32,
        idf: f32,
        doc_len: f32,
        avg_doc_len: f32,
        field_boost: f32,
    ) -> f32 {
        if tf == 0.0 || field_boost == 0.0 {
            return 0.0;
        }
        let length_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len.max(1.0));
        let tf_norm =
            (tf * field_boost * (self.k1 + 1.0)) / (tf * field_boost + self.k1 * length_norm);
        idf * tf_norm
    }

    /// Upper bound from a downward-rounded minimum length/TF ratio.
    /// The scorer remains strict f32; evaluate the real-arithmetic envelope
    /// in f64 and inflate for its at-most eight positive f32 operations.
    /// Unsupported parameters disable this optional bound.
    #[cfg(test)]
    pub(crate) fn upper_bound_with_ratio(
        self,
        max_tf: u32,
        idf: f32,
        ratio: f32,
        avg_len: f32,
    ) -> f32 {
        PreparedBounds::new(self, max_tf, idf, avg_len)
            .map_or(f32::INFINITY, |bounds| bounds.ratio(max_tf, ratio))
    }

    /// Bound the canonical f32 score using a complete frequency/length envelope.
    #[cfg(test)]
    pub(crate) fn upper_bound_with_impacts(
        self,
        max_tf: u32,
        idf: f32,
        avg_len: f32,
        minimum: impl FnOnce(f64, f64) -> Option<f64>,
    ) -> f32 {
        PreparedBounds::new(self, max_tf, idf, avg_len)
            .map_or(f32::INFINITY, |bounds| bounds.impacts(minimum))
    }

    /// Upper bound with the shortest possible unit (length 0).
    #[inline]
    pub fn upper_bound(self, max_tf: f32, idf: f32) -> f32 {
        if max_tf == 0.0 {
            return 0.0;
        }
        let min_length_norm = 1.0 - self.b;
        let tf_norm = (max_tf * (self.k1 + 1.0)) / (max_tf + self.k1 * min_length_norm);
        idf * tf_norm
    }

    /// Upper bound with a known minimum unit length.
    #[inline]
    pub fn upper_bound_with_len(self, max_tf: f32, idf: f32, min_len: f32, avg_len: f32) -> f32 {
        if max_tf == 0.0 {
            return 0.0;
        }
        let length_norm = 1.0 - self.b + self.b * (min_len / avg_len.max(1.0));
        let tf_norm = (max_tf * (self.k1 + 1.0)) / (max_tf + self.k1 * length_norm);
        idf * tf_norm
    }
}

/// Query-constant coefficients for conservative metadata bounds. Construction
/// checks the whole list's maximum TF, which also covers every block/group.
/// Actual document scores continue to use the canonical f32 implementation.
pub(super) struct PreparedBounds {
    numerator: f64,
    k: f64,
    reciprocal: f64,
    length_ratio: f64,
}

impl PreparedBounds {
    pub(super) fn new(params: Bm25Params, max_tf: u32, idf: f32, average: f32) -> Option<Self> {
        if max_tf == 0
            || !idf.is_finite()
            || idf <= 0.0
            || !average.is_finite()
            || !(max_tf as f32 * (params.k1 + 1.0)).is_finite()
            || !params.k1.is_finite()
            || params.k1 < 0.0
            || !(0.0..=1.0).contains(&params.b)
        {
            return None;
        }
        let b = f64::from(params.b);
        let k = f64::from(params.k1);
        Some(Self {
            numerator: f64::from(idf) * (k + 1.0),
            k,
            reciprocal: 1.0 - b,
            length_ratio: b / f64::from(average.max(1.0)),
        })
    }

    fn score(&self, minimum: f64) -> f32 {
        let bound = self.numerator / (1.0 + self.k * minimum);
        Self::inflate(bound)
    }

    fn inflate(bound: f64) -> f32 {
        // Covers the canonical positive f32 operations, integer casts and
        // subnormal rounding. Moving f64 query constants does not remove it.
        (bound * (1.0 + 16.0 * f64::from(f32::EPSILON)) + 16.0 * f64::from(f32::MIN_POSITIVE))
            as f32
    }

    /// A single frequency/length pair needs only one division. All terms are
    /// nonnegative under the constructor's contract. f64 rounding remains
    /// far below the margin covering the canonical f32 score operations.
    pub(super) fn pair(&self, max_tf: u32, length: u32) -> f32 {
        if max_tf == 0 {
            return f32::INFINITY;
        }
        let tf = f64::from(max_tf);
        let norm = self.k * (self.reciprocal + self.length_ratio * f64::from(length));
        Self::inflate(self.numerator * (tf / (tf + norm)))
    }

    /// Approximate inverse used only to seed an integer cutoff search. Callers
    /// must verify its neighboring lengths with the authoritative score predicate.
    pub(super) fn length_cutoff_hint(&self, tf: u32, score: f32) -> Option<u32> {
        if self.k == 0.0 || self.length_ratio == 0.0 || score <= 0.0 {
            return None;
        }
        let inflation = 1.0 + 16.0 * f64::from(f32::EPSILON);
        let length = (f64::from(tf) * (self.numerator * inflation / f64::from(score) - 1.0)
            / self.k
            - self.reciprocal)
            / self.length_ratio;
        length
            .is_finite()
            .then(|| (length.floor() + 1.0).clamp(1.0, 65536.0) as u32)
    }

    pub(super) fn ratio(&self, max_tf: u32, ratio: f32) -> f32 {
        if max_tf == 0 || ratio <= 0.0 || !ratio.is_finite() {
            return f32::INFINITY;
        }
        self.score(self.reciprocal / f64::from(max_tf) + self.length_ratio * f64::from(ratio))
    }

    pub(super) fn impacts(&self, minimum: impl FnOnce(f64, f64) -> Option<f64>) -> f32 {
        match minimum(self.reciprocal, self.length_ratio) {
            Some(value) if value.is_finite() && value >= 0.0 => self.score(value),
            _ => f32::INFINITY,
        }
    }
}

/// Compute BM25 score for a term occurrence
///
/// # Arguments
/// * `tf` - Term frequency in document
/// * `idf` - Inverse document frequency
/// * `doc_len` - Document length (or field length)
/// * `avg_doc_len` - Average document length
#[inline]
pub fn bm25_score(tf: f32, idf: f32, doc_len: f32, avg_doc_len: f32) -> f32 {
    Bm25Params::default().score(tf, idf, doc_len, avg_doc_len)
}

/// Compute BM25F score with field boost
///
/// # Arguments
/// * `tf` - Term frequency in document
/// * `idf` - Inverse document frequency
/// * `doc_len` - Document length (or field length)
/// * `avg_doc_len` - Average document length
/// * `field_boost` - Field-specific boost factor
#[inline]
pub fn bm25f_score(tf: f32, idf: f32, doc_len: f32, avg_doc_len: f32, field_boost: f32) -> f32 {
    Bm25Params::default().score_boosted(tf, idf, doc_len, avg_doc_len, field_boost)
}

/// Compute BM25 upper bound score for MaxScore pruning
///
/// Uses conservative assumptions for maximum possible score:
/// - Maximum TF from posting list
/// - Minimum length normalization (shortest possible document)
#[inline]
pub fn bm25_upper_bound(max_tf: f32, idf: f32) -> f32 {
    Bm25Params::default().upper_bound(max_tf, idf)
}

/// BM25 upper bound with a known minimum length of the scoring units the
/// bound covers (a block or a whole list): the shortest unit has the
/// weakest length normalisation, so it bounds every longer one.
#[inline]
pub fn bm25_upper_bound_with_len(max_tf: f32, idf: f32, min_len: f32, avg_len: f32) -> f32 {
    Bm25Params::default().upper_bound_with_len(max_tf, idf, min_len, avg_len)
}

/// Compute BM25F upper bound score for MaxScore pruning with field boost
///
/// Uses conservative assumptions for maximum possible score:
/// - Maximum TF from posting list
/// - Minimum length normalization (shortest possible document)
/// - Field boost factor
#[inline]
pub fn bm25f_upper_bound(max_tf: f32, idf: f32, field_boost: f32) -> f32 {
    Bm25Params::default().score_boosted(max_tf, idf, 0.0, 1.0, field_boost)
}

/// Compute IDF (Inverse Document Frequency) using BM25 variant
///
/// # Arguments
/// * `doc_freq` - Number of documents containing the term
/// * `total_docs` - Total number of documents in collection
#[inline]
pub fn bm25_idf(doc_freq: f32, total_docs: f32) -> f32 {
    ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln()
}

#[cfg(test)]
mod norm_lookup_tests {
    use super::*;
    #[test]
    fn batch_normalization_preserves_scalar_scores_and_tails() {
        for params in [
            Bm25Params::default(),
            Bm25Params { k1: 0.0, b: 1.0 },
            Bm25Params { k1: 2.3, b: 0.0 },
            Bm25Params { k1: 1.2, b: 1.0 },
        ] {
            for average in [0.0, 1.0, 173.25, 65535.0] {
                let table = NormTable::new(params, average);
                for count in [0, 1, 17, 127, 128] {
                    for start in [0u8, 128] {
                        for boost in [0.0, 1.0, 2.3, -1.0] {
                            let tfs: Vec<_> = (0..count)
                                .map(|i| [0, 1, 2, 17, 65535, u32::MAX][i % 6])
                                .collect();
                            let codes: Vec<_> = (0..count).map(|i| start + i as u8).collect();
                            let mut scores = vec![0.0; count];
                            table.score_batch(
                                params,
                                1.37,
                                average,
                                boost,
                                codes.iter().copied(),
                                &tfs,
                                &mut scores,
                            );
                            for i in 0..count {
                                assert_eq!(
                                    scores[i].to_bits(),
                                    table
                                        .score_boosted(
                                            params,
                                            tfs[i] as f32,
                                            1.37,
                                            codes[i],
                                            average,
                                            boost,
                                        )
                                        .to_bits()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lookup_normalization_matches_canonical_arithmetic_for_every_byte_norm() {
        for params in [
            Bm25Params::default(),
            Bm25Params { k1: 0.0, b: 1.0 },
            Bm25Params { k1: 2.3, b: 0.0 },
            Bm25Params { k1: 1.2, b: 1.0 },
        ] {
            for avg in [0.0, 1.0, 173.25, 65535.0] {
                let table = NormTable::new(params, avg);
                for code in 0..=255 {
                    for tf in [0.0, 1.0, 2.0, 17.0, 65535.0] {
                        let length = if code == 0 {
                            tf
                        } else {
                            crate::segment::norms::decode(code) as f32
                        };
                        assert_eq!(
                            table.score(params, tf, 1.37, code, avg).to_bits(),
                            params.score(tf, 1.37, length, avg).to_bits()
                        );
                        for boost in [0.0, 1.0, 2.3] {
                            assert_eq!(
                                table
                                    .score_boosted(params, tf, 1.37, code, avg, boost)
                                    .to_bits(),
                                params.score_boosted(tf, 1.37, length, avg, boost).to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod ratio_tests {
    use super::*;

    #[test]
    fn ratio_bound_covers_canonical_scores_across_frequency_length_and_parameter_extremes() {
        let values = [
            1u32,
            2,
            3,
            127,
            255,
            65535,
            65536,
            100_000,
            16_777_217,
            u32::MAX,
        ];
        for tf in values {
            for length in values {
                let ratio = ((length as f64 / tf as f64) as f32).next_down();
                for max_tf in [tf, tf.saturating_mul(7), u32::MAX] {
                    for k1 in [0.0, 0.1, 1.2, 16.0, 1e6] {
                        for b in [0.0, 0.01, 0.75, 1.0 - f32::EPSILON, 1.0] {
                            for avg in [0.0, 1.0, 500.3, 1e8] {
                                for idf in [f32::MIN_POSITIVE, 0.001, 1.0, 20.0] {
                                    let params = Bm25Params { k1, b };
                                    let bound =
                                        params.upper_bound_with_ratio(max_tf, idf, ratio, avg);
                                    let score = params.score(tf as f32, idf, length as f32, avg);
                                    let pair = PreparedBounds::new(params, max_tf, idf, avg)
                                        .unwrap()
                                        .pair(max_tf, length);
                                    assert!(
                                        pair >= score,
                                        "pair {params:?} tf={tf} len={length} max={max_tf} avg={avg} idf={idf} score={score} bound={pair}"
                                    );
                                    assert!(
                                        bound >= score,
                                        "{params:?} tf={tf} len={length} max={max_tf} avg={avg} idf={idf} score={score} bound={bound}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ratio_bound_does_not_pair_a_long_documents_tf_with_a_short_documents_length() {
        let params = Bm25Params::default();
        let loose = params.upper_bound_with_len(100.0, 1.0, 10.0, 500.0);
        let tight = params.upper_bound_with_ratio(100, 1.0, 10.0f32.next_down(), 500.0);
        assert!(tight < loose);
        for (tf, len) in [(1.0, 10.0), (100.0, 1000.0)] {
            assert!(tight >= params.score(tf, 1.0, len, 500.0));
        }
        for (k1, b) in [(-1.0, 0.5), (1.0, 2.0), (f32::NAN, 0.5)] {
            assert!(
                Bm25Params { k1, b }
                    .upper_bound_with_ratio(1, 1.0, 1.0, 1.0)
                    .is_infinite()
            );
        }
    }
    #[test]
    fn complete_impact_envelopes_bound_canonical_f32_scores_and_disable_unsupported_parameters() {
        use crate::structures::{BlockPostingList, PostingCodec, PostingList};
        let mut seed = 17u64;
        for trial in 0..128 {
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 32) as u32
            };
            let values = [1, 2, 3, 127, 65535, 65536, 16777217, u32::MAX];
            let mut lengths = [1u32; 256];
            let mut postings = PostingList::new();
            for doc in 0..256 {
                let (tf, length) = if trial % 3 == 0 {
                    let tf = doc % 8 + 1;
                    (tf, tf * tf)
                } else if trial % 3 == 1 {
                    (values[next() as usize % 8], values[next() as usize % 8])
                } else {
                    (next().max(1), next().max(1))
                };
                lengths[doc as usize] = length;
                postings.push(doc, tf);
            }
            let list = BlockPostingList::from_posting_list_with_impact_bounds(
                &postings,
                false,
                Some(&|d| lengths[d as usize]),
                PostingCodec::Rounded,
            )
            .unwrap();
            for k1 in [0.0, 0.9, 1.2, 100.0, f32::MAX] {
                for b in [0.0, 0.25, 0.75, 1.0f32.next_down(), 1.0] {
                    let params = Bm25Params { k1, b };
                    for avg in [0.0, 1.0, 37.5, 65535.0, f32::MAX] {
                        for idf in [f32::MIN_POSITIVE, 0.001, 1.7, f32::MAX] {
                            for block in 0..2 {
                                let bound = params.upper_bound_with_impacts(
                                    list.block_max_tf(block).unwrap(),
                                    idf,
                                    avg,
                                    |a, c| list.block_impact_minimum(block, a, c),
                                );
                                let group_bound = params.upper_bound_with_impacts(
                                    list.max_tf(),
                                    idf,
                                    avg,
                                    |a, c| list.group_impact_minimum(block, a, c),
                                );
                                for posting in postings.iter().skip(block * 128).take(128) {
                                    let score = params.score(
                                        posting.term_freq as f32,
                                        idf,
                                        lengths[posting.doc_id as usize] as f32,
                                        avg,
                                    );
                                    assert!(
                                        group_bound == f32::INFINITY || score <= group_bound,
                                        "group trial={trial} params={params:?} avg={avg} idf={idf} score={score} bound={group_bound}"
                                    );
                                    assert!(
                                        bound == f32::INFINITY || score <= bound,
                                        "trial={trial} params={params:?} avg={avg} idf={idf} score={score} bound={bound}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        for params in [
            Bm25Params { k1: -1.0, b: 0.5 },
            Bm25Params {
                k1: f32::NAN,
                b: 0.5,
            },
            Bm25Params { k1: 1.2, b: -0.1 },
            Bm25Params {
                k1: 1.2,
                b: f32::NAN,
            },
        ] {
            assert_eq!(
                params.upper_bound_with_impacts(1, 1.0, 1.0, |_, _| panic!(
                    "unsupported scorer must not decode envelopes"
                )),
                f32::INFINITY
            );
        }
        let p = Bm25Params::default();
        for (tf, idf, avg) in [
            (0, 1.0, 1.0),
            (1, -1.0, 1.0),
            (1, f32::NAN, 1.0),
            (1, 1.0, f32::INFINITY),
        ] {
            assert_eq!(
                p.upper_bound_with_impacts(tf, idf, avg, |_, _| panic!(
                    "invalid input must not decode envelopes"
                )),
                f32::INFINITY
            );
        }
        assert_eq!(
            p.upper_bound_with_impacts(1, 1.0, 1.0, |_, _| None),
            f32::INFINITY
        );
    }
}

#[cfg(test)]
mod zero_frequency_tests {
    use super::*;

    #[test]
    fn zero_frequency_and_zero_boost_produce_finite_zero_scores_and_bounds() {
        for params in [
            Bm25Params::default(),
            Bm25Params { k1: 0.0, b: 1.0 },
            Bm25Params { k1: 1.2, b: 1.0 },
        ] {
            for length in [0.0, 1.0, 100.0] {
                for average in [0.0, 1.0, 175.0] {
                    assert_eq!(
                        params.score(0.0, 2.3, length, average).to_bits(),
                        0,
                        "{params:?} length={length}"
                    );
                    assert_eq!(
                        params
                            .score_boosted(0.0, 2.3, length, average, 7.0)
                            .to_bits(),
                        0
                    );
                    assert_eq!(
                        params
                            .score_boosted(13.0, 2.3, length, average, 0.0)
                            .to_bits(),
                        0
                    );
                    assert_eq!(params.upper_bound(0.0, 2.3).to_bits(), 0);
                    assert_eq!(
                        params
                            .upper_bound_with_len(0.0, 2.3, length, average)
                            .to_bits(),
                        0
                    );
                }
            }
        }
        assert_eq!(bm25_score(0.0, 2.3, 0.0, 0.0).to_bits(), 0);
        assert_eq!(bm25f_score(13.0, 2.3, 0.0, 0.0, 0.0).to_bits(), 0);
        assert_eq!(bm25_upper_bound(0.0, 2.3).to_bits(), 0);
        assert_eq!(bm25_upper_bound_with_len(0.0, 2.3, 0.0, 0.0).to_bits(), 0);
        assert_eq!(bm25f_upper_bound(13.0, 2.3, 0.0).to_bits(), 0);
    }
}
