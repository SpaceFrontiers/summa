#![crate_name = "summa"]
//! Fast full-text search server with following features:
//!
//! - Simple GRPC API
//! - Indexing documents through Kafka
//!
//! ## Quickstart
//! #### Install
//! ```bash
//! cargo install summa
//! pip install aiosumma
//! ```
//! #### Generate default config file
//! ```bash
//! mkdir data
//! summa-server generate-config > data/summa.yaml
//! ```
//! #### Launch server
//! ```bash
//! summa-server serve data/summa.yaml
//! ```
//! #### Create index
//! ```bash
//! summa-client localhost:8082 create-index test-index --default-fields='["title", "body"]' "
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
//! "
//! ```
//! #### Index documents
//! ```bash
//! summa-client localhost:8082 index-document test-index '{"id": 1, "title": "What We Know About Extraterrestrial Intelligence: Foundations of Xenology", \
//! "body": "Have you ever wondered what could happen when we discover another communicating species outside the Earth? \
//! This book addresses this question in all its complexity. In addition to the physical barriers for communication, \
//! such as the enormous distances where a message can take centuries to reach its recipient, the book also examines \
//! the biological problems of communicating between species, the problems of identifying a non-Terrestrial intelligence, \
//! and the ethical, religious, legal and other problems of conducting discussions across light years. Most of the book is concerned \
//! with issues that could impinge on your life: how do we share experiences with ETI? Can we make shared laws? Could we trade? \
//! Would they have religion? The book addresses these and related issues, identifying potential barriers to communication and \
//! suggesting ways we can overcome them. The book explores this topic through reference to human experience, through analogy and thought \
//! experiment, while relying on what is known to-date about ourselves, our world, and the cosmos we live in.", "tags": ["science", "xenology"]}'
//! ```
//! #### Search
//! ```bash
//! summa-client localhost:8082 commit-index test-index
//! summa-client localhost:8082 search test-index "xenology"
//! ```

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
