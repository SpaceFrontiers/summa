use summa_proto::proto;

use crate::proto_traits::Wrapper;

impl From<Wrapper<proto::SortByField>> for tantivy::IndexSortByField {
    fn from(sort_by_field: Wrapper<proto::SortByField>) -> Self {
        let sort_by_field = sort_by_field.into_inner();
        tantivy::IndexSortByField {
            order: Wrapper::from(sort_by_field.order()).into(),
            field: sort_by_field.field,
        }
    }
}
