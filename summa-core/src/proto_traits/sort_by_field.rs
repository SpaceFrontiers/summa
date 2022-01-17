use summa_proto::proto;

use crate::proto_traits::Wrapper;

impl From<Wrapper<proto::SortByField>> for tantivy::IndexSortByField {
    fn from(sort_by_field: Wrapper<proto::SortByField>) -> Self {
        let sort_by_field = sort_by_field.into_inner();
        tantivy::IndexSortByField {
            field: sort_by_field.field.clone(),
            order: match proto::Order::from_i32(sort_by_field.order) {
                None => tantivy::Order::Asc,
                Some(order) => Wrapper::from(order).into(),
            },
        }
    }
}
