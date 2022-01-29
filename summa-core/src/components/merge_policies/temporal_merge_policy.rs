use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use tantivy::merge_policy::{MergeCandidate, MergePolicy};
use tantivy::{SegmentId, SegmentMeta};

use crate::components::SummaSegmentAttributes;
use crate::utils::current_time;

/// `TemporalMergePolicy` collapses segments old enough into a single one
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TemporalMergePolicy {
    merge_older_then_secs: u64,
}

impl TemporalMergePolicy {
    pub fn new(merge_older_then_secs: u64) -> TemporalMergePolicy {
        TemporalMergePolicy { merge_older_then_secs }
    }
}

impl MergePolicy for TemporalMergePolicy {
    fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<MergeCandidate> {
        let merge_pivot = current_time() - self.merge_older_then_secs;
        let old_segments = segments
            .iter()
            .filter(|segment_meta| {
                let segment_attributes = segment_meta.segment_attributes();
                let is_old = segment_attributes
                    .as_ref()
                    .map(|segment_attributes| {
                        let parsed_attributes = serde_json::from_value::<SummaSegmentAttributes>(segment_attributes.clone());
                        parsed_attributes
                            .ok()
                            .and_then(|v| v.created_at.map(|created_at| created_at < merge_pivot))
                            .unwrap_or(true)
                    })
                    .unwrap_or(true);
                is_old
            })
            .map(|segment| segment.id())
            .collect::<Vec<SegmentId>>();
        vec![MergeCandidate(old_segments)]
    }
}
