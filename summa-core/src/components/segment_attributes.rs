use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tantivy::SegmentAttributesMerger;

use crate::utils::current_time;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SummaSegmentAttributes {
    pub is_frozen: bool,
    pub created_at: Option<u64>,
}

impl Default for SummaSegmentAttributes {
    fn default() -> Self {
        SummaSegmentAttributes {
            is_frozen: false,
            created_at: Some(current_time()),
        }
    }
}

impl SummaSegmentAttributes {
    pub fn frozen() -> Self {
        SummaSegmentAttributes {
            is_frozen: true,
            created_at: Some(current_time()),
        }
    }
}

/// SegmentAttributes implementation owns custom segment attributes and its merging behavior
pub trait SegmentAttributes: Default + Serialize + DeserializeOwned + Send + Sync + Clone {
    /// Must be implemented for defining how to merge `SegmentAttributes` from
    /// different segments
    fn merge(segments_attributes: Vec<Self>) -> Self;
}

#[derive(Clone)]
pub struct SegmentAttributesMergerImpl<S: SegmentAttributes> {
    _phantom: PhantomData<S>,
}

impl<S: SegmentAttributes> SegmentAttributesMergerImpl<S> {
    pub fn new() -> SegmentAttributesMergerImpl<S> {
        SegmentAttributesMergerImpl { _phantom: PhantomData }
    }
}

impl<S: SegmentAttributes + 'static> SegmentAttributesMerger for SegmentAttributesMergerImpl<S> {
    fn merge_json(&self, segment_attributes_json: Vec<&serde_json::Value>) -> serde_json::Value {
        let segment_attributes: Vec<_> = segment_attributes_json.into_iter().flat_map(|v| serde_json::from_value(v.clone())).collect();
        serde_json::to_value(S::merge(segment_attributes)).expect("not serializable")
    }

    fn default(&self) -> serde_json::Value {
        serde_json::to_value(S::default()).expect("not serializable")
    }
}

impl SegmentAttributes for SummaSegmentAttributes {
    fn merge(segments_attributes: Vec<Self>) -> Self {
        SummaSegmentAttributes {
            is_frozen: segments_attributes.into_iter().map(|v| v.is_frozen).reduce(|a, b| a && b).unwrap_or(false),
            created_at: Some(current_time()),
        }
    }
}
