#[cfg(test)]
pub mod shortcuts {
    use crate::proto;

    pub fn match_query(value: &str) -> proto::Query {
        proto::Query {
            query: Some(proto::query::Query::Match(proto::MatchQuery { value: value.to_owned() })),
        }
    }
}
