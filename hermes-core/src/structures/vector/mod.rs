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

/// Hard ceiling for a single decoded ANN artifact. Bincode's limit accounts
/// for claimed container storage as well as bytes consumed, preventing a tiny
/// corrupt payload from advertising an effectively unbounded allocation.
///
/// The 4 GiB 64-bit ceiling still accommodates a 20M-document classic RaBitQ
/// segment (roughly 3.5 GB decoded at 768 dimensions), while preventing one
/// lazy decode from claiming an operationally unbounded fraction of RAM.
#[cfg(target_pointer_width = "64")]
pub(crate) const MAX_DENSE_ANN_DECODE_BYTES: usize = 4 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "64"))]
pub(crate) const MAX_DENSE_ANN_DECODE_BYTES: usize = 512 * 1024 * 1024;

const ANN_DECODE_EXPANSION_FACTOR: usize = 8;
const ANN_DECODE_FIXED_HEADROOM: usize = 16 * 1024 * 1024;

fn decode_ann_with_limit<T: serde::de::DeserializeOwned, const LIMIT: usize>(
    data: &[u8],
) -> std::io::Result<(T, usize)> {
    bincode::serde::decode_from_slice(data, bincode::config::standard().with_limit::<LIMIT>())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

#[cfg(target_pointer_width = "64")]
fn decode_ann_with_relative_limit<T: serde::de::DeserializeOwned>(
    data: &[u8],
    budget: usize,
) -> std::io::Result<(T, usize)> {
    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * MIB;
    if budget <= 32 * MIB {
        decode_ann_with_limit::<T, { 32 * MIB }>(data)
    } else if budget <= 128 * MIB {
        decode_ann_with_limit::<T, { 128 * MIB }>(data)
    } else if budget <= 512 * MIB {
        decode_ann_with_limit::<T, { 512 * MIB }>(data)
    } else if budget <= GIB {
        decode_ann_with_limit::<T, GIB>(data)
    } else if budget <= 2 * GIB {
        decode_ann_with_limit::<T, { 2 * GIB }>(data)
    } else {
        decode_ann_with_limit::<T, { 4 * GIB }>(data)
    }
}

#[cfg(not(target_pointer_width = "64"))]
fn decode_ann_with_relative_limit<T: serde::de::DeserializeOwned>(
    data: &[u8],
    budget: usize,
) -> std::io::Result<(T, usize)> {
    const MIB: usize = 1024 * 1024;
    if budget <= 32 * MIB {
        decode_ann_with_limit::<T, { 32 * MIB }>(data)
    } else if budget <= 128 * MIB {
        decode_ann_with_limit::<T, { 128 * MIB }>(data)
    } else {
        decode_ann_with_limit::<T, { 512 * MIB }>(data)
    }
}

/// Decode exactly one bincode-backed ANN artifact under a bounded allocation
/// budget. Trailing bytes are corruption rather than a second ignored object.
pub(crate) fn decode_ann_bincode_exact<T: serde::de::DeserializeOwned>(
    data: &[u8],
    description: &str,
) -> std::io::Result<T> {
    if data.len() > MAX_DENSE_ANN_DECODE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{description} payload is {} bytes, exceeding the {}-byte decode limit",
                data.len(),
                MAX_DENSE_ANN_DECODE_BYTES
            ),
        ));
    }
    let budget = data
        .len()
        .checked_mul(ANN_DECODE_EXPANSION_FACTOR)
        .and_then(|bytes| bytes.checked_add(ANN_DECODE_FIXED_HEADROOM))
        .unwrap_or(MAX_DENSE_ANN_DECODE_BYTES)
        .min(MAX_DENSE_ANN_DECODE_BYTES);
    let (value, consumed) = decode_ann_with_relative_limit(data, budget)?;
    if consumed != data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{description} payload contains {} trailing bytes",
                data.len() - consumed
            ),
        ));
    }
    Ok(value)
}

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
pub use index::{
    BinaryIvfConfig, BinaryIvfIndex, IVFPQConfig, IVFPQIndex, IVFRaBitQConfig, IVFRaBitQIndex,
    RaBitQIndex,
};

#[cfg(test)]
mod decode_tests {
    use super::decode_ann_bincode_exact;

    #[test]
    fn ann_bincode_decode_is_exact_and_rejects_large_claims_from_tiny_payloads() {
        let encoded =
            bincode::serde::encode_to_vec(vec![1u8, 2, 3], bincode::config::standard()).unwrap();
        assert_eq!(
            decode_ann_bincode_exact::<Vec<u8>>(&encoded, "test").unwrap(),
            [1, 2, 3]
        );

        let mut trailing = encoded;
        trailing.push(0);
        assert!(decode_ann_bincode_exact::<Vec<u8>>(&trailing, "test").is_err());

        // A serialized Vec starts with its usize length. This advertises a
        // 64 MiB allocation from only a few bytes and must hit the 32 MiB tier
        // before Vec can reserve it.
        let oversized_claim =
            bincode::serde::encode_to_vec(64usize * 1024 * 1024, bincode::config::standard())
                .unwrap();
        assert!(decode_ann_bincode_exact::<Vec<u8>>(&oversized_claim, "test").is_err());
    }
}
