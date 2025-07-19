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
//! - `quantization` - Residual product quantization
//!   - `PQCodebook` - the index-global OPQ codebook
//!
//! - `index` - Segment payloads for the production ANN implementations
//!   - `IVFPQIndex` - float vectors with residual PQ codes
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

/// Hard ceiling for a single decoded ANN artifact. Bincode's limit accounts
/// for claimed container storage as well as bytes consumed, preventing a tiny
/// corrupt payload from advertising an effectively unbounded allocation.
///
/// A billion-scale 2,560-bit multi-value binary segment can legitimately
/// exceed 8 GiB. Keep the serialized-input ceiling distinct from bincode's
/// decode budget: bincode charges both bytes consumed and requested container
/// allocations, so a valid 9 GiB payload needs more than a 9 GiB limit.
#[cfg(target_pointer_width = "64")]
pub(crate) const MAX_DENSE_ANN_PAYLOAD_BYTES: usize = 16 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "64"))]
pub(crate) const MAX_DENSE_ANN_PAYLOAD_BYTES: usize = 512 * 1024 * 1024;

#[cfg(target_pointer_width = "64")]
const MAX_DENSE_ANN_DECODE_BUDGET: usize = 32 * 1024 * 1024 * 1024;
#[cfg(not(target_pointer_width = "64"))]
const MAX_DENSE_ANN_DECODE_BUDGET: usize = MAX_DENSE_ANN_PAYLOAD_BYTES;

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
    } else if budget <= 4 * GIB {
        decode_ann_with_limit::<T, { 4 * GIB }>(data)
    } else if budget <= 8 * GIB {
        decode_ann_with_limit::<T, { 8 * GIB }>(data)
    } else if budget <= 16 * GIB {
        decode_ann_with_limit::<T, { 16 * GIB }>(data)
    } else {
        decode_ann_with_limit::<T, { 32 * GIB }>(data)
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
    if data.len() > MAX_DENSE_ANN_PAYLOAD_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{description} payload is {} bytes, exceeding the {}-byte decode limit",
                data.len(),
                MAX_DENSE_ANN_PAYLOAD_BYTES
            ),
        ));
    }
    let budget = data
        .len()
        .checked_mul(ANN_DECODE_EXPANSION_FACTOR)
        .and_then(|bytes| bytes.checked_add(ANN_DECODE_FIXED_HEADROOM))
        .unwrap_or(MAX_DENSE_ANN_DECODE_BUDGET)
        .min(MAX_DENSE_ANN_DECODE_BUDGET);
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
pub use ivf::{CoarseCentroids, CoarseConfig, IvfProbePlan, MultiAssignment, SoarConfig};

// Quantization
pub use quantization::{DistanceTable, PQCodebook, PQConfig};

// Indexes
pub use index::{
    BinaryCoarseQuantizer, BinaryIvfConfig, BinaryIvfIndex, IVFPQConfig, IVFPQIndex, IvfPqQueryPlan,
};

#[cfg(test)]
mod decode_tests {
    use super::decode_ann_bincode_exact;
    use super::{ANN_DECODE_EXPANSION_FACTOR, ANN_DECODE_FIXED_HEADROOM};
    use super::{MAX_DENSE_ANN_DECODE_BUDGET, MAX_DENSE_ANN_PAYLOAD_BYTES};

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

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn billion_scale_binary_payload_fits_checked_decode_budget() {
        // Observed production payload: 8,916,876,069 bytes. It must pass the
        // serialized-input ceiling, while bincode receives enough accounting
        // budget for bytes consumed plus the decoded SoA allocations.
        const OBSERVED_BINARY_IVF_BYTES: usize = 8_916_876_069;
        const { assert!(OBSERVED_BINARY_IVF_BYTES < MAX_DENSE_ANN_PAYLOAD_BYTES) };
        let budget = OBSERVED_BINARY_IVF_BYTES
            .checked_mul(ANN_DECODE_EXPANSION_FACTOR)
            .and_then(|bytes| bytes.checked_add(ANN_DECODE_FIXED_HEADROOM))
            .unwrap_or(MAX_DENSE_ANN_DECODE_BUDGET)
            .min(MAX_DENSE_ANN_DECODE_BUDGET);
        assert_eq!(budget, 32 * 1024 * 1024 * 1024);
    }
}
