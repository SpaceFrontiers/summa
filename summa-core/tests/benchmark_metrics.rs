#[path = "../benches/summa_benchmark/metrics.rs"]
mod metrics;

use std::time::Duration;

#[test]
fn mrr_at_ten_excludes_relevant_documents_below_the_cutoff() {
    let predicted: Vec<usize> = (0..100).collect();
    assert_eq!(metrics::mrr_at_10(&predicted, &[10]), 0.0);
    assert_eq!(metrics::mrr_at_10(&predicted, &[9]), 0.1);
    assert_eq!(metrics::mrr_at_10(&predicted, &[0]), 1.0);
    assert_eq!(metrics::mrr_at_10(&[], &[0]), 0.0);
}

#[test]
fn latency_preserves_sub_microsecond_measurements() {
    let stats = metrics::LatencyStats::from_durations(&[
        Duration::from_nanos(250),
        Duration::from_nanos(750),
    ]);
    assert_eq!(stats.avg_us, 0.5);
}

#[test]
fn ndcg_uses_the_requested_cutoff_and_handles_no_relevance() {
    assert_eq!(metrics::ndcg_at_k(&[3, 2, 1], &[3, 2], 2), 1.0);
    assert_eq!(metrics::ndcg_at_k(&[3, 2, 1], &[1], 2), 0.0);
    assert_eq!(metrics::ndcg_at_k(&[3], &[], 10), 0.0);
}

#[test]
fn recall_at_ten_does_not_count_neighbors_below_ten_or_duplicate_hits() {
    let neighbors: Vec<u32> = (0..100).collect();
    assert_eq!(metrics::recall_at_k(&[10, 11, 12], &neighbors, 10), 0.0);
    assert_eq!(metrics::recall_at_k(&[0, 0, 1], &neighbors, 10), 0.2);
    assert_eq!(metrics::recall_at_k(&[0, 1], &[0, 1], 10), 1.0);
    assert_eq!(metrics::recall_at_k(&[0], &[], 10), 0.0);
}
