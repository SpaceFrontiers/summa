//! Vector index implementations
//!
//! This module provides ready-to-use indexes:
//! - `RaBitQIndex` - Standalone RaBitQ for small datasets
//! - `IVFRaBitQIndex` - IVF with RaBitQ binary quantization
//! - `IVFPQIndex` - IVF with Product Quantization (ScaNN-style)

mod ivf_pq;
mod ivf_rabitq;
mod rabitq;

pub use ivf_pq::{IVFPQConfig, IVFPQIndex};
pub use ivf_rabitq::{IVFRaBitQConfig, IVFRaBitQIndex};
pub use rabitq::RaBitQIndex;

/// Collapse duplicate `(doc_id, ordinal)` candidates to their best (smallest)
/// distance, in place with no allocation. Needed when SOAR multi-assignment
/// is enabled: a vector spilled into a secondary cluster appears once per
/// probed cluster it lives in.
pub(crate) fn dedup_multi_assigned(candidates: &mut Vec<(u32, u16, f32)>) {
    if candidates.len() < 2 {
        return;
    }
    // Sort by key with distance ascending as tiebreaker, then keep the first
    // (= smallest distance) entry per (doc_id, ordinal).
    candidates.sort_unstable_by(|a, b| {
        (a.0, a.1)
            .cmp(&(b.0, b.1))
            .then_with(|| a.2.total_cmp(&b.2))
    });
    candidates.dedup_by_key(|c| (c.0, c.1));
}

/// Shared search epilogue for IVF-based indexes: optionally dedup SOAR-spilled
/// candidates, then select the top-k by ascending distance.
pub(crate) fn finalize_candidates(
    candidates: &mut Vec<(u32, u16, f32)>,
    k: usize,
    dedup_spilled: bool,
) {
    if dedup_spilled {
        dedup_multi_assigned(candidates);
    }
    // Partial sort: O(n + k log k) instead of O(n log n)
    if candidates.len() > k {
        candidates.select_nth_unstable_by(k, |a, b| a.2.partial_cmp(&b.2).unwrap());
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
}
