use crate::proto;

impl From<proto::Order> for tantivy::Order {
    fn from(order: proto::Order) -> Self {
        match order {
            proto::Order::Asc => tantivy::Order::Asc,
            proto::Order::Desc => tantivy::Order::Desc,
        }
    }
}

impl From<proto::Order> for tantivy::aggregation::bucket::Order {
    fn from(order: proto::Order) -> Self {
        match order {
            proto::Order::Asc => tantivy::aggregation::bucket::Order::Asc,
            proto::Order::Desc => tantivy::aggregation::bucket::Order::Desc,
        }
    }
}
