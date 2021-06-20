use crate::proto;
use tantivy::IndexSortByField;

impl From<proto::SortByField> for IndexSortByField {
    fn from(sort_by_field: proto::SortByField) -> Self {
        IndexSortByField {
            field: sort_by_field.field.clone(),
            order: match proto::Order::from_i32(sort_by_field.order) {
                None => tantivy::Order::Asc,
                Some(order) => order.into(),
            },
        }
    }
}

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
