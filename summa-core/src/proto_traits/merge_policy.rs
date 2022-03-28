use std::sync::Arc;

use summa_proto::proto;

use crate::proto_traits::Wrapper;

impl From<Wrapper<Option<proto::MergePolicy>>> for Arc<dyn tantivy::merge_policy::MergePolicy> {
    fn from(merge_policy: Wrapper<Option<proto::MergePolicy>>) -> Self {
        match merge_policy.into_inner() {
            None | Some(proto::MergePolicy { merge_policy: None }) => Arc::new(tantivy::merge_policy::NoMergePolicy),
            Some(proto::MergePolicy {
                merge_policy: Some(proto::merge_policy::MergePolicy::Log(c)),
            }) => {
                if c.is_frozen {
                    Arc::new(crate::components::merge_policies::LogMergePolicy::frozen())
                } else {
                    Arc::<crate::components::merge_policies::LogMergePolicy>::default()
                }
            }
            Some(proto::MergePolicy {
                merge_policy: Some(proto::merge_policy::MergePolicy::Temporal(c)),
            }) => Arc::new(crate::components::merge_policies::TemporalMergePolicy::new(c.merge_older_then_secs)),
        }
    }
}
