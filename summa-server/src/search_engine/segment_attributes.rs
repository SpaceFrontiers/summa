use serde::{Deserialize, Serialize};
use tantivy::SegmentAttributes;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SummaSegmentAttributes {
    pub is_frozen: bool,
}

impl SegmentAttributes for SummaSegmentAttributes {
    fn merge(segments_attributes: Vec<Self>) -> Self {
        SummaSegmentAttributes {
            is_frozen: segments_attributes.into_iter().map(|v| v.is_frozen).reduce(|a, b| a && b).unwrap_or(false),
        }
    }
}
