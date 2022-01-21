use std::str::FromStr;

use crate::errors::SummaResult;

pub mod random;
pub mod sync;
pub mod thread_handler;

/// Parse `iroh` endpoints.
pub fn parse_endpoint<P: FromStr>(endpoint: &str) -> SummaResult<P>
where
    crate::errors::Error: From<<P as FromStr>::Err>,
{
    if endpoint.starts_with("irpc://") {
        Ok(endpoint.parse()?)
    } else {
        Ok(format!("irpc://{endpoint}").parse()?)
    }
}
