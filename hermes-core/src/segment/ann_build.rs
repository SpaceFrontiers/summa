//! Shared ANN index constants and construction helpers.
//!
//! Constants are available on all platforms (including WASM).
//! Builder/serialization functions are native-only.

/// Index type discriminants stored in the vectors file TOC.
pub const RABITQ_TYPE: u8 = 0;
pub const IVF_RABITQ_TYPE: u8 = 1;
pub const SCANN_TYPE: u8 = 2;
pub const FLAT_TYPE: u8 = 4;

// --- Native-only builder/serialization functions ---

#[cfg(feature = "native")]
use crate::structures::{
    CoarseCentroids, IVFPQConfig, IVFPQIndex, IVFRaBitQConfig, IVFRaBitQIndex, PQCodebook,
    RaBitQCodebook, RaBitQConfig,
};

/// Create a fresh IVF-RaBitQ index and codebook ready for vector insertion.
#[cfg(feature = "native")]
pub fn new_ivf_rabitq(dim: usize, centroids: &CoarseCentroids) -> (IVFRaBitQIndex, RaBitQCodebook) {
    let rabitq_config = RaBitQConfig::new(dim);
    let codebook = RaBitQCodebook::new(rabitq_config);
    let ivf_config = IVFRaBitQConfig::new(dim);
    let index = IVFRaBitQIndex::new(ivf_config, centroids.version, codebook.version);
    (index, codebook)
}

/// Serialize a populated IVF-RaBitQ index to bytes.
#[cfg(feature = "native")]
pub fn serialize_ivf_rabitq(
    index: IVFRaBitQIndex,
    codebook: RaBitQCodebook,
) -> crate::Result<Vec<u8>> {
    let data = super::IVFRaBitQIndexData { codebook, index };
    data.to_bytes()
        .map_err(|e| crate::Error::Serialization(e.to_string()))
}

/// Create a fresh ScaNN (IVF-PQ) index ready for vector insertion.
#[cfg(feature = "native")]
pub fn new_scann(dim: usize, centroids: &CoarseCentroids, codebook: &PQCodebook) -> IVFPQIndex {
    let config = IVFPQConfig::new(dim);
    IVFPQIndex::new(config, centroids.version, codebook.version)
}

/// Serialize a populated ScaNN index to bytes.
#[cfg(feature = "native")]
pub fn serialize_scann(index: IVFPQIndex, codebook: &PQCodebook) -> crate::Result<Vec<u8>> {
    let data = super::ScaNNIndexData {
        codebook: codebook.clone(),
        index,
    };
    data.to_bytes()
        .map_err(|e| crate::Error::Serialization(e.to_string()))
}
