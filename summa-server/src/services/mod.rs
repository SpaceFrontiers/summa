//! Services responsible for various aspects a search engine like indices management or aliasing

pub(crate) mod api;
pub(crate) mod index;
#[cfg(feature = "metrics")]
pub(crate) mod metrics;

pub use api::Api;
pub use index::Index;
#[cfg(feature = "metrics")]
pub use metrics::Metrics;
