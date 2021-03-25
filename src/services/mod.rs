//! Services responsible for various aspects a search engine like indices management or aliasing

mod alias_service;
mod index_service;
mod metrics_service;

pub use alias_service::AliasService;
pub use index_service::IndexService;
pub use metrics_service::MetricsService;
