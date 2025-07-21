//! Vector index implementations
//!
//! This module provides ready-to-use indexes:
//! - `IVFPQIndex` - global IVF with residual product quantization

mod binary_ivf;
mod ivf_pq;
mod ivf_tq;

#[cfg(feature = "native")]
pub(crate) use binary_ivf::BinaryIvfBuilder;
pub use binary_ivf::{BinaryCoarseQuantizer, BinaryIvfConfig, BinaryIvfIndex};
pub use ivf_pq::{IVFPQConfig, IVFPQIndex, IvfPqQueryPlan};
pub use ivf_tq::{IvfTqIndex, TqIvfEncodeScratch, TqIvfQueryPlan};

#[derive(Clone, Copy)]
struct RankedEntry {
    doc_id: u32,
    ordinal: u16,
    /// Lower is always better. Similarities are stored negated.
    rank: f32,
}

impl PartialEq for RankedEntry {
    fn eq(&self, other: &Self) -> bool {
        self.rank.to_bits() == other.rank.to_bits()
            && self.doc_id == other.doc_id
            && self.ordinal == other.ordinal
    }
}

impl Eq for RankedEntry {}

impl PartialOrd for RankedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank
            .total_cmp(&other.rank)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
            .then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

/// Shared ANN top-k engine.
///
/// `BY_DOCUMENT` selects whether duplicate keys are `(doc_id, ordinal)` or
/// just `doc_id`; `HIGHER_IS_BETTER` normalizes similarities into the same
/// lower-is-better ordering as distances. The retained map and heap are O(k),
/// including under arbitrarily skewed multi-value input.
pub(crate) struct BoundedAnnCollector<const BY_DOCUMENT: bool, const HIGHER_IS_BETTER: bool> {
    k: usize,
    heap: std::collections::BinaryHeap<RankedEntry>,
    best: rustc_hash::FxHashMap<u64, RankedEntry>,
}

impl<const BY_DOCUMENT: bool, const HIGHER_IS_BETTER: bool>
    BoundedAnnCollector<BY_DOCUMENT, HIGHER_IS_BETTER>
{
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            heap: std::collections::BinaryHeap::with_capacity(k.min(8_192)),
            best: rustc_hash::FxHashMap::with_capacity_and_hasher(k.min(8_192), Default::default()),
        }
    }

    #[inline]
    fn key(doc_id: u32, ordinal: u16) -> u64 {
        if BY_DOCUMENT {
            doc_id as u64
        } else {
            ((doc_id as u64) << 16) | ordinal as u64
        }
    }

    #[inline]
    fn rank(value: f32) -> f32 {
        if HIGHER_IS_BETTER { -value } else { value }
    }

    #[inline]
    fn value(rank: f32) -> f32 {
        if HIGHER_IS_BETTER { -rank } else { rank }
    }

    fn discard_stale_top(&mut self) {
        while let Some(entry) = self.heap.peek() {
            let key = Self::key(entry.doc_id, entry.ordinal);
            if self.best.get(&key).is_some_and(|best| best == entry) {
                break;
            }
            self.heap.pop();
        }
    }

    fn rebuild_if_needed(&mut self) {
        // A better duplicate leaves a stale heap entry. Rebuild occasionally
        // so even adversarial streams stay O(k).
        if self.heap.len() > self.k.saturating_mul(2).max(16) {
            self.heap = self.best.values().copied().collect();
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, doc_id: u32, ordinal: u16, value: f32) {
        if self.k == 0 || !value.is_finite() {
            return;
        }
        let candidate = RankedEntry {
            doc_id,
            ordinal,
            rank: Self::rank(value),
        };

        // A stale root is always worse than its current map entry, so this is
        // a conservative rejection even before stale entries are cleaned.
        // It avoids a hash lookup for the noncompetitive tail of a scan.
        if self.best.len() >= self.k && self.heap.peek().is_some_and(|worst| candidate >= *worst) {
            return;
        }

        let key = Self::key(doc_id, ordinal);
        if let Some(current) = self.best.get_mut(&key) {
            if candidate < *current {
                *current = candidate;
                self.heap.push(candidate);
                self.rebuild_if_needed();
            }
            return;
        }

        if self.best.len() >= self.k {
            self.discard_stale_top();
            let Some(worst) = self.heap.peek().copied() else {
                return;
            };
            if candidate >= worst {
                return;
            }
            self.heap.pop();
            self.best.remove(&Self::key(worst.doc_id, worst.ordinal));
        }
        self.best.insert(key, candidate);
        self.heap.push(candidate);
        self.rebuild_if_needed();
    }

    pub(crate) fn into_sorted_results(self) -> Vec<(u32, u16, f32)> {
        let mut results: Vec<_> = self
            .best
            .into_values()
            .map(|entry| (entry.doc_id, entry.ordinal, Self::value(entry.rank)))
            .collect();
        results.sort_unstable_by(|a, b| {
            let order = if HIGHER_IS_BETTER {
                b.2.total_cmp(&a.2)
            } else {
                a.2.total_cmp(&b.2)
            };
            order
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        results
    }
}

#[cfg(test)]
mod tests {
    use super::BoundedAnnCollector;
    use rand::{Rng, SeedableRng};

    type BoundedDistanceCollector = BoundedAnnCollector<false, false>;
    type BoundedDocumentDistanceCollector = BoundedAnnCollector<true, false>;
    type BoundedDocumentScoreCollector = BoundedAnnCollector<true, true>;

    #[test]
    fn bounded_collector_deduplicates_and_orders_ties() {
        let mut collector = BoundedDistanceCollector::new(3);
        collector.insert(9, 0, 2.0);
        collector.insert(2, 1, 1.0);
        collector.insert(2, 0, 1.0);
        collector.insert(7, 0, 3.0);
        collector.insert(9, 0, 0.5);
        collector.insert(3, 0, 1.0);

        assert_eq!(
            collector.into_sorted_results(),
            vec![(9, 0, 0.5), (2, 0, 1.0), (2, 1, 1.0)]
        );
    }

    #[test]
    fn bounded_collector_handles_zero_k_and_non_finite_distances() {
        let mut zero = BoundedDistanceCollector::new(0);
        zero.insert(1, 0, 1.0);
        assert!(zero.into_sorted_results().is_empty());

        let mut collector = BoundedDistanceCollector::new(2);
        collector.insert(1, 0, f32::NAN);
        collector.insert(2, 0, f32::INFINITY);
        collector.insert(3, 0, 1.0);
        assert_eq!(collector.into_sorted_results(), vec![(3, 0, 1.0)]);
    }

    #[test]
    fn document_collector_keeps_one_best_vector_per_document() {
        let mut collector = BoundedDocumentDistanceCollector::new(3);
        for ordinal in 0..100 {
            collector.insert(1, ordinal, 1.0 - ordinal as f32 / 1_000.0);
        }
        collector.insert(2, 4, 0.5);
        collector.insert(3, 2, 0.5);
        collector.insert(4, 0, 0.6);

        assert_eq!(
            collector.into_sorted_results(),
            vec![(2, 4, 0.5), (3, 2, 0.5), (4, 0, 0.6)]
        );
    }

    #[test]
    fn document_collector_bounds_stale_heap_entries() {
        let mut collector = BoundedDocumentDistanceCollector::new(5);
        for ordinal in (0..10_000u16).rev() {
            collector.insert(1, ordinal, ordinal as f32);
        }
        assert!(collector.heap.len() <= 16);
        assert_eq!(collector.best.len(), 1);
        assert_eq!(collector.into_sorted_results(), vec![(1, 0, 0.0)]);
    }

    #[test]
    fn document_score_collector_uses_descending_similarity_order() {
        let mut collector = BoundedDocumentScoreCollector::new(2);
        collector.insert(4, 1, 0.5);
        collector.insert(4, 0, 0.75);
        collector.insert(3, 0, 0.75);
        collector.insert(2, 0, 0.25);
        assert_eq!(
            collector.into_sorted_results(),
            vec![(3, 0, 0.75), (4, 0, 0.75)]
        );
    }

    #[test]
    fn document_collectors_match_brute_force_on_skewed_streams() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(91);
        let mut stream = Vec::new();
        for _ in 0..50_000 {
            // A small document population forces many updates, evictions,
            // ties, and stale heap entries.
            stream.push((
                rng.random_range(0..500u32),
                rng.random_range(0..64u16),
                rng.random_range(0..100u32) as f32 / 100.0,
            ));
        }

        let k = 37;
        let mut expected_distance: rustc_hash::FxHashMap<u32, (u16, f32)> =
            rustc_hash::FxHashMap::default();
        let mut expected_score: rustc_hash::FxHashMap<u32, (u16, f32)> =
            rustc_hash::FxHashMap::default();
        let mut distances = BoundedDocumentDistanceCollector::new(k);
        let mut scores = BoundedDocumentScoreCollector::new(k);
        for &(doc_id, ordinal, value) in &stream {
            distances.insert(doc_id, ordinal, value);
            scores.insert(doc_id, ordinal, value);
            expected_distance
                .entry(doc_id)
                .and_modify(|current| {
                    if value.total_cmp(&current.1).is_lt()
                        || (value.to_bits() == current.1.to_bits() && ordinal < current.0)
                    {
                        *current = (ordinal, value);
                    }
                })
                .or_insert((ordinal, value));
            expected_score
                .entry(doc_id)
                .and_modify(|current| {
                    if value.total_cmp(&current.1).is_gt()
                        || (value.to_bits() == current.1.to_bits() && ordinal < current.0)
                    {
                        *current = (ordinal, value);
                    }
                })
                .or_insert((ordinal, value));
        }

        let mut expected_distance: Vec<_> = expected_distance
            .into_iter()
            .map(|(doc_id, (ordinal, value))| (doc_id, ordinal, value))
            .collect();
        expected_distance.sort_unstable_by(|a, b| {
            a.2.total_cmp(&b.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        expected_distance.truncate(k);

        let mut expected_score: Vec<_> = expected_score
            .into_iter()
            .map(|(doc_id, (ordinal, value))| (doc_id, ordinal, value))
            .collect();
        expected_score.sort_unstable_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        expected_score.truncate(k);

        assert!(distances.best.len() <= k);
        assert!(distances.heap.len() <= k.saturating_mul(2).max(16));
        assert!(scores.best.len() <= k);
        assert!(scores.heap.len() <= k.saturating_mul(2).max(16));
        assert_eq!(distances.into_sorted_results(), expected_distance);
        assert_eq!(scores.into_sorted_results(), expected_score);
    }
}
