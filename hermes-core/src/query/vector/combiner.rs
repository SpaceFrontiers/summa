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
    /// Log-Sum-Exp: smooth maximum approximation (default)
    /// `score = (1/t) * log(Σ exp(t * sᵢ))`
    /// Higher temperature → closer to max; lower → closer to mean
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

        match self {
            MultiValueCombiner::Sum => scores.iter().map(|(_, s)| s).sum(),
            MultiValueCombiner::Max => scores
                .iter()
                .map(|(_, s)| *s)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0),
            MultiValueCombiner::Avg => {
                let sum: f32 = scores.iter().map(|(_, s)| s).sum();
                sum / scores.len() as f32
            }
            MultiValueCombiner::LogSumExp { temperature } => {
                // Numerically stable log-sum-exp:
                // LSE(x) = max(x) + log(Σ exp(xᵢ - max(x)))
                let t = *temperature;
                let max_score = scores
                    .iter()
                    .map(|(_, s)| *s)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                let sum_exp: f32 = scores
                    .iter()
                    .map(|(_, s)| (t * (s - max_score)).exp())
                    .sum();

                max_score + sum_exp.ln() / t
            }
            MultiValueCombiner::WeightedTopK { k, decay } => {
                // Sort scores descending and take top k
                let mut sorted: Vec<f32> = scores.iter().map(|(_, s)| *s).collect();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                sorted.truncate(*k);

                // Apply exponential decay weights
                let mut weight = 1.0f32;
                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                for score in sorted {
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
        // LogSumExp should be between max (3.0) and max + log(n)/t
        assert!(result >= 3.0);
        assert!(result <= 3.0 + (3.0_f32).ln() / 1.5);
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
