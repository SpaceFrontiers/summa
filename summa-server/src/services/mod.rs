//! Services responsible for various aspects a search engine like indices management or aliasing

pub(crate) mod api;
pub(crate) mod index;
pub(crate) mod metrics;

pub use api::Api;
pub use index::Index;
pub use metrics::Metrics;
