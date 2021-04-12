//! Services responsible for various aspects a search engine like indices management or aliasing

mod index_service;
mod metrics_service;

pub use index_service::IndexService;
pub use metrics_service::MetricsService;
