pub mod shortcuts {
    use crate::proto;

    pub fn match_query(value: &str, default_fields: Vec<String>) -> proto::query::Query {
        proto::query::Query::Match(proto::MatchQuery {
            value: value.to_owned(),
            query_parser_config: Some(proto::QueryParserConfig {
                default_fields,
                ..Default::default()
            }),
        })
    }
}
