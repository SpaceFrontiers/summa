#![crate_name = "summa"]
//! Fast full-text search server with following features:
//!
//! - Simple GRPC API
//! - Indexing documents through Kafka
//!
//! ## Quickstart
//! #### Launch server
//! ```bash
//! summa-server serve
//! ```
//! #### Create index schema
//! ```yaml
//! ---
//! # yamllint disable rule:key-ordering
//! - name: id
//!   type: i64
//!   options:
//!     fast: single
//!     fieldnorms: false
//!     indexed: true
//!     stored: true
//! - name: body
//!   type: text
//!   options:
//!     indexing:
//!       fieldnorms: true
//!       record: position
//!       tokenizer: summa
//!     stored: true
//! - name: tags
//!   type: text
//!   options:
//!     indexing:
//!       fieldnorms: true
//!       record: position
//!       tokenizer: summa
//!     stored: true
//! - name: title
//!   type: text
//!   options:
//!     indexing:
//!       fieldnorms: true
//!       record: position
//!       tokenizer: summa
//!     stored: true
//! ```
//! #### Create index
//! #### Index documents
//! #### Search

#[macro_use]
extern crate lazy_static;

mod apis;
pub mod configurator;
mod consumers;
pub mod errors;
mod requests;
mod search_engine;
pub mod servers;
mod services;
mod utils;

pub mod proto {
    tonic::include_proto!("summa");
}
