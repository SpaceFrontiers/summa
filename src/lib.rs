#![crate_name = "summa"]
//! Fast full-text search server
//!
//! ## Features
//! - Simple GRPC API for managing multiple indices
//! - Indexing documents through Kafka or directly
//! - IPFS API for replication (oncoming)
//! - Exposing metrics in Prometheus format
//! - Various configurable tokenizers (including CJK)
//! - Fine CLI on Python
//!
//! ## Getting started
//! - Look at [examples](crate::examples)
//! - Explore [Docs.rs](https://docs.rs/crate/summa/latest)

#[macro_use]
extern crate lazy_static;

pub mod apis;
pub mod configurator;
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
