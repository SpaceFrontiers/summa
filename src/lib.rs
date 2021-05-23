//! Fast full-text search server
//!
//! ### Features
//! - Fine CLI and asynchronous client library [aiosumma](aiosumma/README.md) written in Python
//! - Simple GRPC API for managing multiple indices and for search
//! - Extendable query parsing on Python client side
//! - Ranking documents with custom and/or user-defined scoring functions
//! - Indexing documents through Kafka or directly
//! - Tracing with [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-rust) and exposing metrics in Prometheus format
//! - Reflection API for deep insights about text data
//! - Configurable tokenizers (including CJK)
//! - IPFS API for replication (soon)
//!
//! ## Getting started
//! - Look at [examples](crate::examples)
//! - Explore [Docs.rs](https://docs.rs/crate/summa/latest)

#[macro_use]
extern crate derive_builder;
#[macro_use]
extern crate lazy_static;
extern crate core;

pub mod apis;
pub mod application;
pub mod configs;
pub mod consumers;
pub mod errors;
mod examples;
pub(crate) mod logging;
mod metrics;
mod proto_traits;
pub mod requests;
pub mod search_engine;
pub mod servers;
pub mod services;
mod utils;

pub use application::Application;

pub mod proto {
    tonic::include_proto!("summa.proto");
    pub(crate) const FILE_DESCRIPTOR_SET: &'static [u8] = tonic::include_file_descriptor_set!("summa");
}
