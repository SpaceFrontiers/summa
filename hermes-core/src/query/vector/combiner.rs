//! Multi-value score combination strategies for vector search

/// Strategy for combining scores when a document has multiple values for the same field
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultiValueCombiner {
    /// Sum all scores (accumulates dot product contributions)
    Sum,
    /// Take the maximum score
    Max,
    /// Take the average score
    Avg,
    /// Softmax-weighted smooth maximum (default)
    /// `score = Σ softmax(t * sᵢ) * sᵢ`
    /// Higher temperature → closer to max; lower → closer to mean.
    ///
    /// Deliberately NOT the raw `(1/t)·log(Σ exp(t·sᵢ))`: that form adds
    /// `ln(n)/t` per document, so chunk *count* outranked chunk quality —
    /// a 300-chunk compendium of mediocre matches beat every focused paper
    /// (score 0.55 + ln(300)/1.5 ≈ 4.3 vs 0.72 + ln(3)/1.5 ≈ 1.4). The
    /// softmax weighting is count-invariant (n identical scores combine to
    /// that score), bounded by the max, and still tracks a dominant score
    /// at any scale.
    LogSumExp {
        /// Temperature parameter (default: 1.5)
        temperature: f32,
    },
    /// Weighted Top-K: weight top scores with exponential decay
    /// `score = Σ wᵢ * sorted_scores[i]` where `wᵢ = decay^i`
    WeightedTopK {
        /// Number of top scores to consider (default: 5)
        k: usize,
        /// Decay factor per rank (default: 0.7)
        decay: f32,
    },
}

impl Default for MultiValueCombiner {
    fn default() -> Self {
        // LogSumExp with temperature 1.5 provides good balance between
        // max (best relevance) and sum (saturation from multiple matches)
        MultiValueCombiner::LogSumExp { temperature: 1.5 }
    }
}

impl MultiValueCombiner {
    pub(crate) fn validate(self) -> Result<(), String> {
        match self {
            Self::LogSumExp { temperature } if !temperature.is_finite() || temperature <= 0.0 => {
                Err(format!(
                    "LogSumExp temperature must be finite and greater than zero, got {temperature}"
                ))
            }
            Self::WeightedTopK { k: 0, .. } => {
                Err("WeightedTopK k must be greater than zero".to_string())
            }
            Self::WeightedTopK { decay, .. }
                if !decay.is_finite() || !(0.0..=1.0).contains(&decay) =>
            {
                Err(format!(
                    "WeightedTopK decay must be finite and in [0, 1], got {decay}"
                ))
            }
            _ => Ok(()),
        }
    }

    /// Create LogSumExp combiner with default temperature (1.5)
    pub fn log_sum_exp() -> Self {
        Self::LogSumExp { temperature: 1.5 }
    }

    /// Create LogSumExp combiner with custom temperature
    pub fn log_sum_exp_with_temperature(temperature: f32) -> Self {
        Self::LogSumExp { temperature }
    }

    /// Create WeightedTopK combiner with defaults (k=5, decay=0.7)
    pub fn weighted_top_k() -> Self {
        Self::WeightedTopK { k: 5, decay: 0.7 }
    }

    /// Create WeightedTopK combiner with custom parameters
    pub fn weighted_top_k_with_params(k: usize, decay: f32) -> Self {
        Self::WeightedTopK { k, decay }
    }

    /// Combine multiple scores into a single score
    pub fn combine(&self, scores: &[(u32, f32)]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }

        // Strict IEEE accumulation on purpose: these reductions run over a
        // handful of ordinals per document, and measured `algebraic_add`
        // (docs/algebraic-float-reductions.md) made the exp-bound LogSumExp
        // loop slower at n=5 (16.5 → 20.4 ns) while changing rounding.
        match self {
            MultiValueCombiner::Sum => scores.iter().map(|(_, s)| s).sum(),
            MultiValueCombiner::Max => scores
                .iter()
                .map(|(_, s)| *s)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(0.0),
            MultiValueCombiner::Avg => {
                let sum: f32 = scores.iter().map(|(_, s)| s).sum();
                sum / scores.len() as f32
            }
            MultiValueCombiner::LogSumExp { temperature } => {
                // Softmax-weighted average, numerically stabilized by
                // subtracting the max before exponentiation. The max's own
                // weight is exp(0) = 1, so the denominator is never zero.
                let t = *temperature;
                let max_score = scores
                    .iter()
                    .map(|(_, s)| *s)
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(0.0);

                let mut weight_sum = 0.0f32;
                let mut weighted = 0.0f32;
                for &(_, s) in scores {
                    let weight = (t * (s - max_score)).exp();
                    weight_sum += weight;
                    weighted += weight * s;
                }
                weighted / weight_sum
            }
            MultiValueCombiner::WeightedTopK { k, decay } => {
                let k = (*k).min(scores.len());
                if k == 0 {
                    return 0.0;
                }
                // Select the top k scores without allocating for the common
                // small-document case: a stack buffer for up to 16 values,
                // `select_nth_unstable` to partition when k < len, then sort
                // only the top k (the decay weights are rank-dependent).
                const INLINE: usize = 16;
                let mut inline = [0.0f32; INLINE];
                let mut spilled: Vec<f32>;
                let values: &mut [f32] = if scores.len() <= INLINE {
                    for (slot, &(_, s)) in inline.iter_mut().zip(scores) {
                        *slot = s;
                    }
                    &mut inline[..scores.len()]
                } else {
                    spilled = scores.iter().map(|&(_, s)| s).collect();
                    spilled.as_mut_slice()
                };
                if k < values.len() {
                    values.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
                }
                let top = &mut values[..k];
                top.sort_unstable_by(|a, b| b.total_cmp(a));

                // Apply exponential decay weights
                let mut weight = 1.0f32;
                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                for &score in top.iter() {
                    weighted_sum += weight * score;
                    weight_total += weight;
                    weight *= decay;
                }

                if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combiner_sum() {
        let scores = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let combiner = MultiValueCombiner::Sum;
        assert!((combiner.combine(&scores) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_combiner_max() {
        let scores = vec![(0, 1.0), (1, 3.0), (2, 2.0)];
        let combiner = MultiValueCombiner::Max;
        assert!((combiner.combine(&scores) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_combiner_avg() {
        let scores = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let combiner = MultiValueCombiner::Avg;
        assert!((combiner.combine(&scores) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_combiner_log_sum_exp() {
        let scores = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let combiner = MultiValueCombiner::log_sum_exp();
        let result = combiner.combine(&scores);
        // A smooth maximum lives between the mean and the max, weighted
        // toward the max.
        assert!(result > 2.0, "must exceed the mean, got {result}");
        assert!(result <= 3.0, "must never exceed the max, got {result}");
    }

    /// The production incident this pins: a 300-chunk compendium whose chunks
    /// all score ~0.5 must not outrank a 3-chunk paper whose chunks score
    /// ~0.7. The additive `ln(n)/t` count term of the raw log-sum-exp did
    /// exactly that (0.55 + ln(300)/1.5 ≈ 4.3 vs 0.72 + ln(3)/1.5 ≈ 1.4),
    /// which buried every relevant result for "off-label aripiprazole usage"
    /// under generic long documents.
    #[test]
    fn log_sum_exp_is_count_invariant_and_bounded_by_max() {
        let combiner = MultiValueCombiner::log_sum_exp();

        // n identical scores combine to that score, regardless of n.
        let identical: Vec<(u32, f32)> = (0..300).map(|i| (i, 0.7)).collect();
        let combined = combiner.combine(&identical);
        assert!(
            (combined - 0.7).abs() < 1e-3,
            "300 identical 0.7 chunks must combine to 0.7, got {combined}"
        );

        // Many mediocre chunks never outrank a few strong ones.
        let mut compendium: Vec<(u32, f32)> = (0..300).map(|i| (i, 0.5)).collect();
        compendium.push((300, 0.55));
        let paper = vec![(0, 0.72), (1, 0.70), (2, 0.65)];
        let compendium_score = combiner.combine(&compendium);
        let paper_score = combiner.combine(&paper);
        assert!(
            paper_score > compendium_score,
            "3 strong chunks ({paper_score}) must beat 301 mediocre ones ({compendium_score})"
        );
    }

    /// The reason the fix is a softmax-weighted average rather than
    /// `LSE - ln(n)/t`: subtracting the count term turns the combiner into a
    /// near-average in the peaked regime, collapsing a document whose single
    /// chunk scores 15 among 299 zeros to ~6.4. The softmax weighting keeps
    /// it at the dominant score.
    #[test]
    fn log_sum_exp_tracks_a_dominant_score_at_sparse_scale() {
        let combiner = MultiValueCombiner::log_sum_exp_with_temperature(0.7);
        let mut scores: Vec<(u32, f32)> = (0..299).map(|i| (i, 0.0)).collect();
        scores.push((299, 15.0));
        let combined = combiner.combine(&scores);
        assert!(
            (combined - 15.0).abs() < 0.3,
            "one dominant sparse chunk must keep its score, got {combined}"
        );
    }

    #[test]
    fn test_combiner_log_sum_exp_approaches_max_with_high_temp() {
        let scores = vec![(0, 1.0), (1, 5.0), (2, 2.0)];
        // High temperature should approach max
        let combiner = MultiValueCombiner::log_sum_exp_with_temperature(10.0);
        let result = combiner.combine(&scores);
        // Should be very close to max (5.0)
        assert!((result - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_combiner_weighted_top_k() {
        let scores = vec![(0, 5.0), (1, 3.0), (2, 1.0), (3, 0.5)];
        let combiner = MultiValueCombiner::weighted_top_k_with_params(3, 0.5);
        let result = combiner.combine(&scores);
        // Top 3: 5.0, 3.0, 1.0 with weights 1.0, 0.5, 0.25
        // weighted_sum = 5*1 + 3*0.5 + 1*0.25 = 6.75
        // weight_total = 1.75
        // result = 6.75 / 1.75 ≈ 3.857
        assert!((result - 3.857).abs() < 0.01);
    }

    #[test]
    fn test_combiner_weighted_top_k_less_than_k() {
        let scores = vec![(0, 2.0), (1, 1.0)];
        let combiner = MultiValueCombiner::weighted_top_k_with_params(5, 0.7);
        let result = combiner.combine(&scores);
        // Only 2 scores, weights 1.0 and 0.7
        // weighted_sum = 2*1 + 1*0.7 = 2.7
        // weight_total = 1.7
        // result = 2.7 / 1.7 ≈ 1.588
        assert!((result - 1.588).abs() < 0.01);
    }

    /// The stack/select path must agree with a plain full sort for every
    /// length around the inline buffer boundary and every k, including
    /// ties and k larger than the input.
    #[test]
    fn weighted_top_k_selection_matches_full_sort_across_inline_boundary() {
        for len in [1usize, 2, 5, 15, 16, 17, 40] {
            let scores: Vec<(u32, f32)> = (0..len)
                .map(|i| (i as u32, ((i * 7919) % 13) as f32 / 13.0))
                .collect();
            for k in [1usize, 2, 3, 5, 16, 17, 64] {
                let combiner = MultiValueCombiner::weighted_top_k_with_params(k, 0.7);
                let actual = combiner.combine(&scores);

                let mut sorted: Vec<f32> = scores.iter().map(|&(_, s)| s).collect();
                sorted.sort_unstable_by(|a, b| b.total_cmp(a));
                sorted.truncate(k);
                let (mut w, mut ws, mut wt) = (1.0f32, 0.0f32, 0.0f32);
                for s in sorted {
                    ws += w * s;
                    wt += w;
                    w *= 0.7;
                }
                let expected = ws / wt;
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "len {len} k {k}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn test_combiner_empty_scores() {
        let scores: Vec<(u32, f32)> = vec![];
        assert_eq!(MultiValueCombiner::Sum.combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::Max.combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::Avg.combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::log_sum_exp().combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::weighted_top_k().combine(&scores), 0.0);
    }

    #[test]
    fn test_combiner_single_score() {
        let scores = vec![(0, 5.0)];
        // All combiners should return 5.0 for a single score
        assert!((MultiValueCombiner::Sum.combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::Max.combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::Avg.combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::log_sum_exp().combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::weighted_top_k().combine(&scores) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_combiner_is_log_sum_exp() {
        let combiner = MultiValueCombiner::default();
        match combiner {
            MultiValueCombiner::LogSumExp { temperature } => {
                assert!((temperature - 1.5).abs() < 1e-6);
            }
            _ => panic!("Default combiner should be LogSumExp"),
        }
    }
}
