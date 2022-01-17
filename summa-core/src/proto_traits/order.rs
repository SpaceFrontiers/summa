use summa_proto::proto;

use crate::proto_traits::Wrapper;

impl From<Wrapper<proto::Order>> for tantivy::Order {
    fn from(order: Wrapper<proto::Order>) -> Self {
        match order.into_inner() {
            proto::Order::Asc => tantivy::Order::Asc,
            proto::Order::Desc => tantivy::Order::Desc,
        }
    }
}

impl From<Wrapper<proto::Order>> for tantivy::aggregation::bucket::Order {
    fn from(order: Wrapper<proto::Order>) -> Self {
        match order.into_inner() {
            proto::Order::Asc => tantivy::aggregation::bucket::Order::Asc,
            proto::Order::Desc => tantivy::aggregation::bucket::Order::Desc,
        }
    }
}
