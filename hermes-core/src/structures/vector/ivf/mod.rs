//! IVF (Inverted File Index) module for vector search
//!
//! This module provides routing and assignment infrastructure shared by float
//! PQ payloads and exact packed binary payloads:
//!
//! - `coarse` - Coarse centroids for IVF partitioning (k-means clustering)
//! - `soar` - SOAR (Spilling with Orthogonality-Amplified Residuals) for better recall

mod coarse;
pub(crate) mod routing;
mod soar;

pub use coarse::{CoarseCentroids, CoarseConfig};
pub use routing::{HNSW_AUTO_THRESHOLD, IvfProbePlan};
pub use soar::{MultiAssignment, SoarConfig};
