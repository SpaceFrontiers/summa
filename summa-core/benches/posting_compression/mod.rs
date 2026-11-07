//! Comprehensive benchmarks for posting list compression methods
//!
//! Compares: HorizBP, VertBP128, Elias-Fano, Partitioned EF, Roaring, OptP4D
//!
//! Tests multiple doc_id distributions:
//! - Sparse (1% density): rare terms
//! - Medium (10% density): typical terms
//! - Dense (50% density): common terms
//! - Clustered: docs grouped in ranges (locality)
//! - Sequential: consecutive doc_ids (best case)
//!
//! Metrics measured:
//! - Compression ratio (bytes per doc_id)
//! - Encoding speed (elements/sec)
//! - Decoding/iteration speed (elements/sec)
//! - Seek speed (seeks/sec)
//! - Serialization/deserialization speed

mod access_patterns;
mod block_decompression;
mod common;
mod deserialization;
mod distribution;
mod encoding;
mod iteration;
mod seek;
mod summary;

pub use access_patterns::*;
pub use block_decompression::*;
pub use common::*;
pub use deserialization::*;
pub use distribution::*;
pub use encoding::*;
pub use iteration::*;
pub use seek::*;
pub use summary::*;
