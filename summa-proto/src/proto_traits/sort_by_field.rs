use tantivy::IndexSortByField;

use crate::proto;

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
