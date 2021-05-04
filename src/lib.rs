#![crate_name = "summa"]
//! Fast full-text search server
//!
//! ### Features
//! - Simple GRPC API for managing multiple indices and for search
//! - Indexing documents through Kafka or directly
//! - Tracing with [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-rust) and exposing metrics in Prometheus format
//! - Various configurable tokenizers (including CJK)
//! - Fine CLI and asynchronous client library [aiosumma](aiosumma/README.md) written in Python
//! - IPFS API for replication (soon)
//!
//! ## Getting started
//! - Look at [examples](crate::examples)
//! - Explore [Docs.rs](https://docs.rs/crate/summa/latest)

#[macro_use]
extern crate derive_builder;
#[macro_use]
extern crate lazy_static;

pub mod apis;
pub mod configs;
pub mod consumers;
pub mod errors;
mod examples;
pub mod requests;
pub mod search_engine;
pub mod servers;
pub mod services;
mod utils;

pub mod proto {
    tonic::include_proto!("summa.proto");
}
