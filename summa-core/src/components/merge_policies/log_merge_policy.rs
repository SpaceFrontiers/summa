use std::fmt::Debug;

use tantivy::merge_policy::{MergeCandidate, MergePolicy};
use tantivy::SegmentMeta;

use crate::components::SummaSegmentAttributes;

/// `FrozenLogMergePolicy` is the same as `LogMergePolicy` except supporting of excluding
/// some segments marked as `is_frozen` from merging
#[derive(Debug, Default)]
pub struct LogMergePolicy {
    pub inner: tantivy::merge_policy::LogMergePolicy,
    pub is_frozen: bool,
}

impl LogMergePolicy {
    pub fn frozen() -> LogMergePolicy {
        LogMergePolicy {
            inner: tantivy::merge_policy::LogMergePolicy::default(),
            is_frozen: true,
        }
    }
}

impl MergePolicy for LogMergePolicy {
    fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<MergeCandidate> {
        if self.is_frozen {
            let filtered_segments = segments
                .iter()
                .filter(|segment_meta| {
                    let segment_attributes = segment_meta.segment_attributes();
                    let is_frozen = segment_attributes
                        .as_ref()
                        .map(|segment_attributes| {
                            let parsed_attributes = serde_json::from_value::<SummaSegmentAttributes>(segment_attributes.clone());
                            parsed_attributes.map(|v| v.is_frozen).unwrap_or(false)
                        })
                        .unwrap_or(false);
                    !is_frozen
                })
                .cloned()
                .collect::<Vec<SegmentMeta>>();
            self.inner.compute_merge_candidates(&filtered_segments)
        } else {
            self.inner.compute_merge_candidates(segments)
        }
    }
}
