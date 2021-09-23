#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("invalid_aggregation")]
    InvalidAggregation
}
