//! Shared ANN index constants and construction helpers.
//!
//! Constants are available on all platforms (including WASM).
//! Builder/serialization functions are native-only.

/// Index type discriminants stored in the vectors file TOC.
pub const IVF_PQ_TYPE: u8 = 2;
pub const FLAT_TYPE: u8 = 4;
/// Binary IVF payload backed by an index-level global quantizer.
pub const BINARY_IVF_TYPE: u8 = 6;

// --- Native-only builder/serialization functions ---

#[cfg(feature = "native")]
use crate::structures::{CoarseCentroids, IVFPQConfig, IVFPQIndex, PQCodebook};

/// Create a fresh IVF-PQ index ready for vector insertion.
#[cfg(feature = "native")]
pub fn new_ivf_pq(
    dim: usize,
    routing: crate::dsl::IvfRoutingMode,
    centroids: &CoarseCentroids,
    codebook: &PQCodebook,
) -> IVFPQIndex {
    let config = IVFPQConfig::new(dim, codebook.config.num_subspaces).with_routing(routing);
    IVFPQIndex::new(config, centroids.version, codebook.version)
}

/// Serialize a populated IVF-PQ index to bytes.
#[cfg(feature = "native")]
pub fn serialize_ivf_pq(index: IVFPQIndex) -> crate::Result<Vec<u8>> {
    index
        .to_bytes()
        .map_err(|e| crate::Error::Serialization(e.to_string()))
}
