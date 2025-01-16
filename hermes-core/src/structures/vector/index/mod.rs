//! Vector index implementations
//!
//! This module provides ready-to-use indexes:
//! - `RaBitQIndex` - Standalone RaBitQ for small datasets
//! - `IVFRaBitQIndex` - IVF with RaBitQ binary quantization
//! - `IVFPQIndex` - IVF with Product Quantization (ScaNN-style)

mod ivf_pq;
mod ivf_rabitq;
mod rabitq;

pub use ivf_pq::{IVFPQConfig, IVFPQIndex};
pub use ivf_rabitq::{IVFRaBitQConfig, IVFRaBitQIndex};
pub use rabitq::RaBitQIndex;
