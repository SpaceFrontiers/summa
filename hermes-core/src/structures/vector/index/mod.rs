//! Vector index implementations
//!
//! This module provides ready-to-use indexes:
//! - `RaBitQIndex` - Standalone RaBitQ for small datasets
//! - `IVFRaBitQIndex` - IVF with RaBitQ binary quantization
//! - `IVFPQIndex` - IVF with Product Quantization (ScaNN-style)

mod binary_ivf;
mod ivf_pq;
mod ivf_rabitq;
mod rabitq;

pub use binary_ivf::{BinaryIvfConfig, BinaryIvfIndex};
pub use ivf_pq::{IVFPQConfig, IVFPQIndex};
pub use ivf_rabitq::{IVFRaBitQConfig, IVFRaBitQIndex};
pub use rabitq::RaBitQIndex;

#[derive(Clone, Copy)]
struct DistanceEntry {
    doc_id: u32,
    ordinal: u16,
    distance: f32,
}

impl PartialEq for DistanceEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits()
            && self.doc_id == other.doc_id
            && self.ordinal == other.ordinal
    }
}

impl Eq for DistanceEntry {}

impl PartialOrd for DistanceEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
            .then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

/// Streaming, deduplicating top-k distance collector. IVF previously retained
/// every vector in every probed cluster before truncation, so a full nprobe
/// scan allocated O(segment vectors) per query.
pub(crate) struct BoundedDistanceCollector {
    k: usize,
    heap: std::collections::BinaryHeap<DistanceEntry>,
    best: rustc_hash::FxHashMap<(u32, u16), f32>,
}

impl BoundedDistanceCollector {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            heap: std::collections::BinaryHeap::with_capacity(k.min(8_192)),
            best: rustc_hash::FxHashMap::with_capacity_and_hasher(k.min(8_192), Default::default()),
        }
    }

    fn discard_stale_top(&mut self) {
        while let Some(entry) = self.heap.peek() {
            let current = self.best.get(&(entry.doc_id, entry.ordinal));
            if current.is_some_and(|distance| distance.to_bits() == entry.distance.to_bits()) {
                break;
            }
            self.heap.pop();
        }
    }

    fn rebuild_if_needed(&mut self) {
        // Improved duplicate entries leave stale heap nodes. Rebuild
        // occasionally so even adversarial duplicate streams stay O(k).
        if self.heap.len() > self.k.saturating_mul(2).max(16) {
            self.heap = self
                .best
                .iter()
                .map(|(&(doc_id, ordinal), &distance)| DistanceEntry {
                    doc_id,
                    ordinal,
                    distance,
                })
                .collect();
        }
    }

    pub(crate) fn insert(&mut self, doc_id: u32, ordinal: u16, distance: f32) {
        if self.k == 0 || !distance.is_finite() {
            return;
        }
        let key = (doc_id, ordinal);
        if let Some(current) = self.best.get_mut(&key) {
            if distance.total_cmp(current).is_lt() {
                *current = distance;
                self.heap.push(DistanceEntry {
                    doc_id,
                    ordinal,
                    distance,
                });
                self.rebuild_if_needed();
            }
            return;
        }

        let candidate = DistanceEntry {
            doc_id,
            ordinal,
            distance,
        };
        if self.best.len() >= self.k {
            self.discard_stale_top();
            let Some(worst) = self.heap.peek().copied() else {
                return;
            };
            if candidate >= worst {
                return;
            }
            self.heap.pop();
            self.best.remove(&(worst.doc_id, worst.ordinal));
        }
        self.best.insert(key, distance);
        self.heap.push(candidate);
        self.rebuild_if_needed();
    }

    pub(crate) fn into_sorted_results(self) -> Vec<(u32, u16, f32)> {
        let mut results: Vec<_> = self
            .best
            .into_iter()
            .map(|((doc_id, ordinal), distance)| (doc_id, ordinal, distance))
            .collect();
        results.sort_unstable_by(|a, b| {
            a.2.total_cmp(&b.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        results
    }
}

#[cfg(test)]
mod tests {
    use super::BoundedDistanceCollector;

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
}
