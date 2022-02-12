pub mod shortcuts {
    use crate::proto;

    pub fn match_query(value: &str) -> proto::query::Query {
        proto::query::Query::Match(proto::MatchQuery { value: value.to_owned() })
    }
}
