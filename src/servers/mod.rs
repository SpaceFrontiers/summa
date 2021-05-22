//! Server modules that expose a search engine to the outside

mod base;
mod grpc;
mod metrics;

pub use grpc::GrpcServer;
pub use metrics::MetricsServer;
