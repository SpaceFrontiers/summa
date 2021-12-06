#![deny(
    non_shorthand_field_patterns,
    no_mangle_generic_items,
    overflowing_literals,
    path_statements,
    unused_allocation,
    unused_comparisons,
    unused_parens,
    while_true,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_must_use,
    clippy::unwrap_used
)]

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

pub mod apis;
pub mod components;
pub mod configs;
pub mod errors;
pub(crate) mod hyper_external_request;
pub(crate) mod logging;
pub mod requests;
pub mod server;
pub mod servers;
pub mod services;
pub(crate) mod utils;

pub use server::Server;
