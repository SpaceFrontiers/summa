//! IVF (Inverted File Index) module for vector search
//!
//! This module provides the core IVF infrastructure that can be combined
//! with different quantization methods (RaBitQ, PQ, etc.):
//!
//! - `coarse` - Coarse centroids for IVF partitioning (k-means clustering)
//! - `cluster` - Generic cluster data storage
//! - `soar` - SOAR (Spilling with Orthogonality-Amplified Residuals) for better recall

pub mod cluster;
mod coarse;
mod soar;

pub use cluster::{ClusterData, ClusterStorage, QuantizedCode};
pub use coarse::{CoarseCentroids, CoarseConfig};
pub use soar::{MultiAssignment, SoarConfig};
