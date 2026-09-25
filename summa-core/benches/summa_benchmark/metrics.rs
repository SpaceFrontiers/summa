//! Benchmark metrics computation

use std::collections::HashSet;
use std::time::Duration;

/// Latency statistics
#[derive(Clone, Debug)]
pub struct LatencyStats {
    pub avg_us: f64,
}

impl LatencyStats {
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self { avg_us: 0.0 };
        }

        let micros: Vec<f64> = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1_000_000.0)
            .collect();
        let avg_us = micros.iter().sum::<f64>() / micros.len() as f64;

        Self { avg_us }
    }
}

/// Compute MRR@10 against relevance judgments.
pub fn mrr_at_10(predicted: &[usize], relevant: &[u32]) -> f32 {
    let relevant_set: HashSet<u32> = relevant.iter().copied().collect();
    for (rank, &idx) in predicted.iter().take(10).enumerate() {
        if relevant_set.contains(&(idx as u32)) {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain)
pub fn ndcg_at_k(predicted: &[usize], relevant: &[u32], k: usize) -> f32 {
    let relevant_set: HashSet<u32> = relevant.iter().copied().collect();

    // DCG
    let mut dcg = 0.0f32;
    for (i, &idx) in predicted.iter().take(k).enumerate() {
        if relevant_set.contains(&(idx as u32)) {
            dcg += 1.0 / (i as f32 + 2.0).log2();
        }
    }

    // Ideal DCG (all relevant docs at top)
    let num_relevant = relevant.len().min(k);
    let mut idcg = 0.0f32;
    for i in 0..num_relevant {
        idcg += 1.0 / (i as f32 + 2.0).log2();
    }

    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

/// Recall@k against the first k exact neighbors, independent of file depth.
pub fn recall_at_k(predicted: &[usize], neighbors: &[u32], k: usize) -> f32 {
    let relevant: HashSet<u32> = neighbors.iter().take(k).copied().collect();
    if relevant.is_empty() {
        return 0.0;
    }
    let found: HashSet<u32> = predicted.iter().take(k).map(|&id| id as u32).collect();
    found.intersection(&relevant).count() as f32 / relevant.len() as f32
}
