//! Shared ANN index constants and construction helpers.
//!
//! Constants are available on all platforms (including WASM).
//! Builder/serialization functions are native-only.

/// Index type discriminants stored in the vectors file TOC.
/// Type 2 was IVF-PQ, removed after IVF-TQ superseded it; the loader still
/// recognizes it to fail with an actionable message. Never reuse it.
pub const LEGACY_IVF_PQ_TYPE: u8 = 2;
pub const FLAT_TYPE: u8 = 4;
/// Binary IVF payload backed by an index-level global quantizer.
pub const BINARY_IVF_TYPE: u8 = 6;
/// TurboQuant flat payload; training-free, no global artifacts.
pub const TQ_FLAT_TYPE: u8 = 7;
/// IVF-TQ payload: trained coarse router, TurboQuant residual leaves.
pub const IVF_TQ_TYPE: u8 = 8;

// --- Native-only builder/serialization functions ---

#[cfg(feature = "native")]
use crate::structures::CoarseCentroids;

/// Encode one segment's vectors into IVF-TQ leaves against the trained
/// global coarse centroids.
#[cfg(feature = "native")]
pub fn build_ivf_tq(
    dim: usize,
    routing: crate::dsl::IvfRoutingMode,
    centroids: &CoarseCentroids,
    doc_id_ordinals: &[(u32, u16)],
    vectors: &[f32],
) -> crate::Result<Vec<u8>> {
    let codec = crate::structures::vector::quantization::tq_shared_codec(dim);
    let mut index = crate::structures::IvfTqIndex::new(dim, routing, centroids.version, codec);
    index
        .add_vectors_parallel(centroids, doc_id_ordinals, vectors)
        .map_err(|error| crate::Error::Internal(format!("IVF-TQ encode failed: {error}")))?;
    let mut bytes = Vec::new();
    crate::segment::ann_disk::write_built_ivf_tq(&index, centroids.num_clusters, &mut bytes)
        .map_err(crate::Error::Io)?;
    Ok(bytes)
}

/// Encode one segment's vectors with the training-free TurboQuant codec.
#[cfg(feature = "native")]
pub fn build_tq_flat(
    dim: usize,
    doc_id_ordinals: &[(u32, u16)],
    vectors: &[f32],
) -> crate::Result<Vec<u8>> {
    let codec = crate::structures::vector::quantization::tq_shared_codec(dim);
    let mut builder = crate::structures::TqFlatBuilder::new(codec);
    builder
        .add_batch(doc_id_ordinals, vectors)
        .map_err(|error| crate::Error::Internal(format!("TQ encode failed: {error}")))?;
    builder.finish();
    let mut bytes = Vec::new();
    crate::segment::ann_disk::write_built_tq_flat(&builder, &mut bytes)
        .map_err(crate::Error::Io)?;
    Ok(bytes)
}
