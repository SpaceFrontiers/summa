//! ## Quickstart
//! ### Install
//! ```bash
//! # Install server and `summa-server` binary
//! cargo install summa
//! # Install Python client and `summa-client` binary
//! pip install aiosumma
//! ```
//! ### Generate default config file
//! ```bash
//! # Make directory where index will reside to
//! mkdir data
//! # Generate default config
//! summa-server generate-config > data/summa.yaml
//! ```
//! ### Launch server
//! ```bash
//! # Launch the server and start serving
//! summa-server serve data/summa.yaml
//! ```
//! ### Create index
//! ```bash
//! # Create index through `summa-client`
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
//! ### Index documents
//! ```bash
//! # Index document by passing content directly through `summa-client`
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
//! # Commit `test-index` to make the indexed document visible
//! summa-client localhost:8082 commit-index test-index
//! ```
//! ### Search
//! ```bash
//! summa-client localhost:8082 search test-index "xenology"
//! ```
