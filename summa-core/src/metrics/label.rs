use summa_proto::proto;

/// Trait is used for converting data to Prometheus labels
pub trait ToLabel {
    fn to_label(&self) -> String;
}

impl ToLabel for proto::query::Query {
    fn to_label(&self) -> String {
        match &self {
            proto::query::Query::All(_) => "all",
            proto::query::Query::Boolean(_) => "boolean",
            proto::query::Query::Empty(_) => "empty",
            proto::query::Query::Match(_) => "match",
            proto::query::Query::Range(_) => "range",
            proto::query::Query::Boost(_) => "boost",
            proto::query::Query::Regex(_) => "regex",
            proto::query::Query::Phrase(_) => "phrase",
            proto::query::Query::Term(_) => "term",
            proto::query::Query::MoreLikeThis(_) => "more_like_this",
            proto::query::Query::DisjunctionMax(_) => "disjunction_max",
            proto::query::Query::Exists(_) => "exists",
        }
        .to_owned()
    }
}
