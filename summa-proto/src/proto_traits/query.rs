pub mod shortcuts {
    use crate::proto;

    pub fn match_query(value: &str, default_fields: Vec<String>) -> proto::query::Query {
        proto::query::Query::Match(proto::MatchQuery {
            value: value.to_owned(),
            default_fields,
        })
    }
}
