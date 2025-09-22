//! Streaming ScaNN primitives.
//!
//! The trained routing tree and asymmetric-hashing codebook belong to the
//! index generation, not to an individual immutable segment. Segment payloads
//! therefore carry only a compatibility fingerprint and leaf-local encoded
//! rows. Compatible segment merges rebase document IDs and move leaf runs;
//! they never retrain or re-encode vector codes.

mod ah;
mod artifact;
mod binary;
mod config;
mod engine;
mod fast_scan;
mod geometry;
mod payload;
mod quantized_dot;

pub use ah::{
    AhCodebook, AhEncodeScratch, AhQuery, CENTERS_PER_BLOCK, DEFAULT_ANISOTROPIC_THRESHOLD,
};
pub use artifact::{
    SCANN_GLOBAL_ARTIFACT_VERSION, ScannAhCodebook, ScannAhCodebookRef, ScannRoutingLevel,
    ScannRoutingLevelRef, ScannTrainedArtifact, ScannTrainedArtifactView,
};
pub use binary::{
    BinaryScannHit, BinaryScannModel, BinaryScannProbePlan, BinaryScannSearchScratch,
    BinaryScannSegment, BinaryScannSpillAssignment, BinaryScannTraining, BinaryScannTrainingStats,
    QuantizedBinaryScannModel, QuantizedBinaryScannModelView,
};
pub use config::{
    DEFAULT_TRAINING_SAMPLE_SIZE, MAX_SCANN_LEAVES, MAX_SCANN_TREE_LEVELS,
    MIN_PARTITION_TRAINING_POINTS_PER_LEAF, MIN_POINTS_FOR_PARTITIONING,
    PARTITION_TRAINING_POINTS_PER_CENTROID, SCANN_FAST_SCAN_LANES, ScannConfig, ScannEncoding,
    ScannTrainingState,
};
pub use engine::{
    EncodedFloatVector, FloatEncodeScratch, FloatRoutingTree, FloatScannModel, FloatScannQuery,
    QuantizedFloatScannModel, QuantizedFloatScannModelView, RoutedLeaf, RoutingScratch,
    RoutingTraining, RoutingTrainingStats, train_routing_tree,
};
pub use fast_scan::{
    FAST_SCAN_LANES, FAST_SCAN_LAYOUT_VERSION, FastScanKernel, FastScanQuery, pack_fast_scan_block,
    packed_block_bytes, packed_code_position, padded_blocks,
};
pub use geometry::{
    QUALITY_OPTIMIZED_POINTS_PER_LEAF, ScannGeometry, derive_geometry, derive_geometry_with_levels,
    derive_geometry_with_levels_and_sample_limit, desired_training_sample, geometry_for_leaves,
    geometry_for_leaves_with_auto_depth,
};
pub use payload::{SCANN_SEGMENT_PAYLOAD_VERSION, ScannLeafRun, ScannSegmentPayload};

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Count one occurrence of a degraded condition and emit `log::warn!` for the
/// 1st, 10th, 100th, ... occurrence so a hot path can stay observable without
/// flooding the log. Returns the running total.
pub(crate) fn warn_rate_limited(counter: &AtomicU64, message: impl FnOnce(u64) -> String) -> u64 {
    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
    if 10u64.pow(count.ilog10()) == count {
        log::warn!("{}", message(count));
    }
    count
}

/// Retain the existing recall-oriented parent beam, then widen it only when
/// that prefix cannot expose the requested number of children. Ranked nodes
/// may come from a subset of the level, so the target is capped to the child
/// coverage that is actually reachable from this frontier.
pub(crate) fn routing_prefix_for_child_coverage<T>(
    ranked: &[T],
    child_offsets: &[u32],
    minimum_width: usize,
    requested_children: usize,
    node_id: impl Fn(&T) -> usize,
) -> usize {
    let minimum_width = minimum_width.min(ranked.len());
    let reachable = ranked.iter().fold(0usize, |total, item| {
        let node = node_id(item);
        total.saturating_add(
            child_offsets[node + 1]
                .saturating_sub(child_offsets[node])
                .try_into()
                .unwrap_or(usize::MAX),
        )
    });
    let target = requested_children.min(reachable);
    let mut covered = 0usize;
    for (index, item) in ranked.iter().enumerate() {
        let node = node_id(item);
        covered = covered.saturating_add(
            child_offsets[node + 1]
                .saturating_sub(child_offsets[node])
                .try_into()
                .unwrap_or(usize::MAX),
        );
        let width = index + 1;
        if width >= minimum_width && covered >= target {
            return width;
        }
    }
    ranked.len()
}

/// Validation or compatibility error for a ScaNN artifact or segment payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannFormatError(String);

impl ScannFormatError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ScannFormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ScannFormatError {}

pub type ScannResult<T> = Result<T, ScannFormatError>;

#[cfg(test)]
mod tests {
    use super::routing_prefix_for_child_coverage;

    #[test]
    fn routing_prefix_widens_until_requested_children_are_reachable() {
        let ranked = [2usize, 0, 1];
        let offsets = [0u32, 2, 5, 9];
        assert_eq!(
            routing_prefix_for_child_coverage(&ranked, &offsets, 1, 6, |node| *node),
            2
        );
        assert_eq!(
            routing_prefix_for_child_coverage(&ranked, &offsets, 1, 9, |node| *node),
            3
        );
        assert_eq!(
            routing_prefix_for_child_coverage(&ranked, &offsets, 2, 1, |node| *node),
            2
        );
    }
}
