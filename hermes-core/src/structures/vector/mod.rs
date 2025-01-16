//! Vector indexing data structures
//!
//! This module provides a modular architecture for vector search:
//!
//! ## Module Structure
//!
//! - `ivf` - Core IVF (Inverted File Index) infrastructure
//!   - `CoarseCentroids` - k-means clustering for coarse quantization
//!   - `ClusterData` / `ClusterStorage` - generic cluster storage
//!   - `SoarConfig` / `MultiAssignment` - SOAR geometry-aware assignment
//!
//! - `quantization` - Vector quantization methods
//!   - `RaBitQCodebook` - RaBitQ binary quantization (32x compression)
//!   - `PQCodebook` - Product Quantization with OPQ (ScaNN-style)
//!   - `Quantizer` trait - common interface for quantizers
//!
//! - `index` - Ready-to-use IVF indexes
//!   - `IVFRaBitQIndex` - IVF + RaBitQ
//!   - `IVFPQIndex` - IVF + PQ
//!
//! ## SOAR (Spilling with Orthogonality-Amplified Residuals)
//!
//! The IVF module includes Google's SOAR algorithm for improved recall:
//! - Assigns vectors to multiple clusters (primary + secondary)
//! - Secondary clusters chosen to have orthogonal residuals
//! - Improves recall by 5-15% with ~1.3-2x storage overhead

pub mod index;
pub mod ivf;
pub mod quantization;

// IVF core
pub use ivf::{
    ClusterData, ClusterStorage, CoarseCentroids, CoarseConfig, MultiAssignment, QuantizedCode,
    SoarConfig,
};

// Quantization
pub use quantization::{
    DistanceTable, PQCodebook, PQConfig, PQVector, QuantizedQuery, QuantizedVector, Quantizer,
    RaBitQCodebook, RaBitQConfig,
};

// Indexes
pub use index::{IVFPQConfig, IVFPQIndex, IVFRaBitQConfig, IVFRaBitQIndex, RaBitQIndex};
