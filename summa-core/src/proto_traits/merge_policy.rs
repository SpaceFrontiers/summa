use summa_proto::proto;

use crate::proto_traits::Wrapper;

impl From<Wrapper<Option<proto::MergePolicy>>> for Box<dyn tantivy::merge_policy::MergePolicy> {
    fn from(order: Wrapper<Option<proto::MergePolicy>>) -> Self {
        match order.into_inner() {
            None | Some(proto::MergePolicy { merge_policy: None }) => Box::new(tantivy::merge_policy::NoMergePolicy),
            Some(proto::MergePolicy {
                merge_policy: Some(proto::merge_policy::MergePolicy::Log(c)),
            }) => {
                if c.is_frozen {
                    Box::new(crate::components::merge_policies::LogMergePolicy::frozen())
                } else {
                    Box::<crate::components::merge_policies::LogMergePolicy>::default()
                }
            }
            Some(proto::MergePolicy {
                merge_policy: Some(proto::merge_policy::MergePolicy::Temporal(c)),
            }) => Box::new(crate::components::merge_policies::TemporalMergePolicy::new(c.merge_older_then_secs)),
        }
    }
}
