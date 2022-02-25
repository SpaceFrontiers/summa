pub mod shortcuts {
    use std::collections::HashMap;

    use crate::proto;

    pub fn match_query(value: &str, default_fields: Vec<String>) -> proto::query::Query {
        proto::query::Query::Match(proto::MatchQuery {
            value: value.to_owned(),
            default_fields,
            default_mode: None,
            field_boosts: HashMap::new(),
            exact_matches_promoter: None,
        })
    }
}
