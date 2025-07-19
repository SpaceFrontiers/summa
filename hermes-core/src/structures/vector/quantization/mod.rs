//! Vector quantization for IVF indexes
//!
//! Residual Product Quantization with OPQ is the production float codec.
//!
mod pq;

pub use pq::{DistanceTable, PQCodebook, PQConfig};
