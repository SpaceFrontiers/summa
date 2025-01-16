//! Vector quantization methods for IVF indexes
//!
//! This module provides different quantization strategies:
//! - `rabitq` - RaBitQ binary quantization (32x compression)
//! - `pq` - Product Quantization with OPQ and anisotropic loss (ScaNN-style)
//!
//! All quantizers implement the `Quantizer` trait for use with IVF indexes.

mod pq;
mod rabitq;

pub use pq::{DistanceTable, PQCodebook, PQConfig, PQVector};
pub use rabitq::{QuantizedQuery, QuantizedVector, RaBitQCodebook, RaBitQConfig};

use super::ivf::cluster::QuantizedCode;

/// Trait for vector quantization methods
///
/// Quantizers encode vectors into compact codes and provide
/// fast approximate distance computation.
pub trait Quantizer: Clone + Send + Sync {
    /// The quantized code type
    type Code: QuantizedCode;

    /// Configuration type
    type Config: Clone;

    /// Query-specific precomputed data for fast distance computation
    type QueryData;

    /// Encode a vector (optionally relative to a centroid)
    fn encode(&self, vector: &[f32], centroid: Option<&[f32]>) -> Self::Code;

    /// Prepare query-specific data for fast distance computation
    fn prepare_query(&self, query: &[f32], centroid: Option<&[f32]>) -> Self::QueryData;

    /// Compute approximate distance using precomputed query data
    fn compute_distance(&self, query_data: &Self::QueryData, code: &Self::Code) -> f32;

    /// Decode a code back to approximate vector (if supported)
    fn decode(&self, _code: &Self::Code) -> Option<Vec<f32>> {
        None
    }

    /// Memory usage of the quantizer itself (codebooks, etc.)
    fn size_bytes(&self) -> usize;
}
