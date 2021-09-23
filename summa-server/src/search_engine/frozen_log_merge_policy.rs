use std::fmt::Debug;
use tantivy::merge_policy::{LogMergePolicy, MergeCandidate, MergePolicy};
use tantivy::{SegmentAttribute, SegmentMeta};

#[derive(Debug, Default)]
pub struct FrozenLogMergePolicy(LogMergePolicy);

impl MergePolicy for FrozenLogMergePolicy {
    fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<MergeCandidate> {
        let filtered_segments = segments
            .iter()
            .filter(|segment_meta| match segment_meta.segment_attributes().get("is_frozen") {
                None => true,
                Some(is_frozen) => match is_frozen {
                    SegmentAttribute::ConjunctiveBool(value) => !*value,
                    _ => unreachable!(),
                },
            })
            .cloned()
            .collect::<Vec<SegmentMeta>>();
        self.0.compute_merge_candidates(&filtered_segments)
    }
}
