use std::fmt::Debug;
use std::time::{Duration, UNIX_EPOCH};

use instant::SystemTime;
use serde::{Deserialize, Serialize};
use tantivy::merge_policy::{MergeCandidate, MergePolicy};
use tantivy::{SegmentId, SegmentMeta};

use super::SummaSegmentAttributes;

/// `TemporalMergePolicy` collapses segments old enough into a single one
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TemporalMergePolicy {
    merge_older_then_n_secs: u64,
}

impl TemporalMergePolicy {
    pub fn merge_before(merge_older_then: Duration) -> TemporalMergePolicy {
        TemporalMergePolicy {
            merge_older_then_n_secs: merge_older_then.as_secs(),
        }
    }
}

impl MergePolicy for TemporalMergePolicy {
    fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<MergeCandidate> {
        let merge_pivot = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time goes backward").as_secs() - self.merge_older_then_n_secs;
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
