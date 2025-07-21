//! Vector indexing data structures
//!
//! This module provides a modular architecture for vector search:
//!
//! ## Module Structure
//!
//! - `ivf` - Core IVF (Inverted File Index) infrastructure
//!   - `CoarseCentroids` - k-means clustering for coarse quantization
//!   - `SoarConfig` / `MultiAssignment` - SOAR geometry-aware assignment
//!
//! - `quantization` - TurboQuant training-free codec
//!   - `TqCodec` - derived rotation/codebook, no trained artifacts
//!
//! - `index` - Segment payloads for the production ANN implementations
//!   - `IvfTqIndex` - float vectors with TQ-coded centroid residuals
//!   - `BinaryIvfIndex` - exact packed binary vectors
//!
//! ## SOAR (Spilling with Orthogonality-Amplified Residuals)
//!
//! The IVF module includes Google's SOAR algorithm for improved recall:
//! - Assigns vectors to multiple clusters (primary + secondary)
//! - Secondary clusters chosen to have orthogonal residuals
//! - Improves recall by 5-15% with ~1.3-2x storage overhead

pub mod index;
pub mod ivf;
mod kmeans;
pub mod quantization;

// IVF core
pub use ivf::{CoarseCentroids, CoarseConfig, IvfProbePlan, MultiAssignment, SoarConfig};

// Quantization
#[cfg(feature = "native")]
pub use quantization::TqFlatBuilder;
pub use quantization::{TqCodec, TqQueryPlan};

// Indexes
pub use index::{
    BinaryCoarseQuantizer, BinaryIvfConfig, BinaryIvfIndex, IvfTqIndex, TqIvfEncodeScratch,
    TqIvfQueryPlan,
};
