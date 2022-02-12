use std::str::FromStr;

use crate::errors::SummaResult;

pub mod random;
pub mod sync;

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

pub fn current_time() -> u64 {
    (instant::now() / 1000.0) as u64
}

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| iters.iter_mut().map(|n| n.next().expect("wrong length")).collect::<Vec<T>>())
        .collect()
}
