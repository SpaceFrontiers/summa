use super::{ScannFormatError, ScannResult};

/// Maximum number of persisted routing centroid levels.
///
/// Matches the imported ScaNN artifact layout and AlloyDB-compatible knob.
/// Three levels support billion-scale trees with configurable branching.
pub const MAX_SCANN_TREE_LEVELS: u8 = 3;
/// On-disk leaf identifiers are u32, but the imported routing format is
/// intentionally capped to bound resident directories and route fan-out.
pub const MAX_SCANN_LEAVES: u32 = 30_000_000;
/// Below this many vectors, the fresh ScaNN implementation stays flat rather
/// than training a partition tree.
pub const MIN_POINTS_FOR_PARTITIONING: u64 = 100_000;
/// Hardcoded minimum number of sampled rows per terminal leaf.
///
/// This is deliberately not a schema or server setting: a routing tree with
/// fewer samples is not a viable trained geometry. Automatic geometry shrinks
/// to fit the builder's sample budget; explicitly requested geometry is
/// rejected when that budget cannot provide this floor.
pub const MIN_PARTITION_TRAINING_POINTS_PER_LEAF: u64 = 8;
/// Recall-oriented routing sample target per leaf centroid.
pub const PARTITION_TRAINING_POINTS_PER_CENTROID: u64 = 200;
/// Absolute routing/AH training sample target for small trained trees.
pub const DEFAULT_TRAINING_SAMPLE_SIZE: u64 = 100_000;
pub const SCANN_FAST_SCAN_LANES: usize = 32;

/// Leaf representation used by the segment-local ScaNN payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScannEncoding {
    /// Residual asymmetric-hash codes for floating-point embeddings.
    AsymmetricHash {
        dimensions_per_block: u16,
        bits_per_code: u8,
    },
    /// Exact packed binary embeddings scored with Hamming distance.
    BinaryHamming,
}

impl ScannEncoding {
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::AsymmetricHash { .. } => 1,
            Self::BinaryHamming => 2,
        }
    }

    pub(crate) fn parameters(self) -> (u16, u8) {
        match self {
            Self::AsymmetricHash {
                dimensions_per_block,
                bits_per_code,
            } => (dimensions_per_block, bits_per_code),
            Self::BinaryHamming => (0, 0),
        }
    }

    pub(crate) fn from_parts(tag: u8, dimensions_per_block: u16, bits: u8) -> ScannResult<Self> {
        match (tag, dimensions_per_block, bits) {
            (1, dimensions_per_block, bits_per_code) => Ok(Self::AsymmetricHash {
                dimensions_per_block,
                bits_per_code,
            }),
            (2, 0, 0) => Ok(Self::BinaryHamming),
            _ => Err(ScannFormatError::new(
                "invalid ScaNN leaf encoding or encoding parameters",
            )),
        }
    }

    pub fn row_code_bytes(self, dimension: u32) -> ScannResult<usize> {
        let dimension = usize::try_from(dimension)
            .map_err(|_| ScannFormatError::new("ScaNN dimension exceeds usize"))?;
        match self {
            Self::AsymmetricHash {
                dimensions_per_block,
                bits_per_code,
            } => {
                if dimensions_per_block == 0 || bits_per_code != 4 {
                    return Err(ScannFormatError::new(
                        "ScaNN AH encoding requires non-zero block dimensions and 4-bit codes",
                    ));
                }
                let blocks = dimension.div_ceil(usize::from(dimensions_per_block));
                blocks
                    .checked_mul(usize::from(bits_per_code))
                    .and_then(|bits| bits.checked_add(7))
                    .map(|bits| bits / 8)
                    .ok_or_else(|| ScannFormatError::new("ScaNN AH row size overflows usize"))
            }
            Self::BinaryHamming => {
                if !dimension.is_multiple_of(8) {
                    return Err(ScannFormatError::new(
                        "binary ScaNN dimension must be a multiple of eight bits",
                    ));
                }
                Ok(dimension / 8)
            }
        }
    }

    /// Encoded byte length for one leaf's corpus-sized code column. AH uses
    /// the 32-lane FastScan v2 layout (two blocks per 32-byte word, odd block
    /// counts padded; see `docs/fast-scan-layout-v2.md`) for complete groups
    /// and compact row-major packing for the tail.
    pub fn leaf_code_bytes(self, dimension: u32, rows: usize) -> ScannResult<usize> {
        match self {
            Self::BinaryHamming => self
                .row_code_bytes(dimension)?
                .checked_mul(rows)
                .ok_or_else(|| ScannFormatError::new("binary ScaNN leaf size overflows")),
            Self::AsymmetricHash {
                dimensions_per_block,
                ..
            } => {
                self.row_code_bytes(dimension)?;
                let blocks = (dimension as usize).div_ceil(usize::from(dimensions_per_block));
                let full_rows = rows / SCANN_FAST_SCAN_LANES;
                let tail_rows = rows % SCANN_FAST_SCAN_LANES;
                let full_block_bytes = super::packed_block_bytes(blocks)
                    .ok_or_else(|| ScannFormatError::new("ScaNN FastScan block size overflows"))?;
                let tail_row_bytes = blocks.div_ceil(2);
                full_rows
                    .checked_mul(full_block_bytes)
                    .and_then(|bytes| {
                        tail_rows
                            .checked_mul(tail_row_bytes)
                            .and_then(|tail| bytes.checked_add(tail))
                    })
                    .ok_or_else(|| ScannFormatError::new("ScaNN FastScan leaf size overflows"))
            }
        }
    }
}

/// Index-scoped ScaNN training and layout configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannConfig {
    /// Float dimensions for AH, bit dimensions for binary Hamming.
    pub dimension: u32,
    /// Configurable number of routing centroid levels.
    pub tree_levels: u8,
    /// Number of terminal leaves shared by every segment.
    pub num_leaves: u32,
    pub encoding: ScannEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScannTrainingState {
    AwaitingData { observed: u64, required: u64 },
    Ready { observed: u64, required: u64 },
}

impl ScannConfig {
    pub fn validate(&self) -> ScannResult<()> {
        if self.dimension == 0 {
            return Err(ScannFormatError::new(
                "ScaNN vector dimension must be positive",
            ));
        }
        if self.tree_levels == 0 || self.tree_levels > MAX_SCANN_TREE_LEVELS {
            return Err(ScannFormatError::new(format!(
                "ScaNN tree_levels must be in 1..={MAX_SCANN_TREE_LEVELS}"
            )));
        }
        if self.num_leaves < 2 || self.num_leaves > MAX_SCANN_LEAVES {
            return Err(ScannFormatError::new(
                "ScaNN num_leaves must be in 2..=30,000,000",
            ));
        }
        self.encoding.row_code_bytes(self.dimension)?;
        Ok(())
    }

    /// Minimum viable sample. A trainer may reduce the desired sample under a
    /// memory budget only while retaining at least this many rows.
    pub fn minimum_training_sample(&self) -> ScannResult<u64> {
        self.validate()?;
        u64::from(self.num_leaves)
            .checked_mul(MIN_PARTITION_TRAINING_POINTS_PER_LEAF)
            .ok_or_else(|| ScannFormatError::new("ScaNN minimum training sample overflows u64"))
    }

    /// Recall-oriented sample target from the fresh ScaNN builder. The target
    /// is capped by the observed corpus, but never silently changes geometry.
    pub fn desired_training_sample(&self, observed_vectors: u64) -> ScannResult<u64> {
        self.validate()?;
        let desired = u64::from(self.num_leaves)
            .checked_mul(PARTITION_TRAINING_POINTS_PER_CENTROID)
            .ok_or_else(|| ScannFormatError::new("ScaNN training sample target overflows u64"))?
            .max(DEFAULT_TRAINING_SAMPLE_SIZE);
        Ok(observed_vectors.min(desired))
    }

    /// Minimum corpus size at which a requested partition geometry may train.
    /// Before this threshold, serving must use the exact fallback.
    pub fn effective_training_threshold(&self) -> ScannResult<u64> {
        self.validate()?;
        Ok(MIN_POINTS_FOR_PARTITIONING.max(self.minimum_training_sample()?))
    }

    pub fn training_state(&self, observed_vectors: u64) -> ScannResult<ScannTrainingState> {
        let required = self.effective_training_threshold()?;
        Ok(if observed_vectors < required {
            ScannTrainingState::AwaitingData {
                observed: observed_vectors,
                required,
            }
        } else {
            ScannTrainingState::Ready {
                observed: observed_vectors,
                required,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(encoding: ScannEncoding) -> ScannConfig {
        ScannConfig {
            dimension: 128,
            tree_levels: 3,
            num_leaves: 1_000_000,
            encoding,
        }
    }

    #[test]
    fn scann_training_waits_for_partition_floor_and_minimum_leaf_sample() {
        let config = config(ScannEncoding::BinaryHamming);
        assert_eq!(config.effective_training_threshold().unwrap(), 8_000_000);
        assert_eq!(
            config.training_state(7_999_999).unwrap(),
            ScannTrainingState::AwaitingData {
                observed: 7_999_999,
                required: 8_000_000,
            }
        );
        assert!(matches!(
            config.training_state(8_000_000).unwrap(),
            ScannTrainingState::Ready { .. }
        ));
        assert_eq!(
            config.desired_training_sample(8_000_000).unwrap(),
            8_000_000
        );

        let mut smaller = config;
        smaller.num_leaves = 1_000;
        assert_eq!(smaller.effective_training_threshold().unwrap(), 100_000);
        assert_eq!(smaller.desired_training_sample(1_000_000).unwrap(), 200_000);
    }

    #[test]
    fn binary_scann_requires_a_byte_aligned_bit_dimension() {
        let mut config = config(ScannEncoding::BinaryHamming);
        config.dimension = 127;
        assert!(config.validate().is_err());
    }

    #[test]
    fn ah_row_size_is_derived_without_rounding_down() {
        let encoding = ScannEncoding::AsymmetricHash {
            dimensions_per_block: 2,
            bits_per_code: 4,
        };
        assert_eq!(encoding.row_code_bytes(5).unwrap(), 2);
        // FastScan v2 stores two blocks per 32-byte word: three blocks pad to
        // two words (64 bytes) per complete 32-row group, and the 33rd row is
        // a row-major tail of `ceil(3 / 2)` bytes.
        assert_eq!(encoding.leaf_code_bytes(5, 32).unwrap(), 64);
        assert_eq!(encoding.leaf_code_bytes(5, 33).unwrap(), 66);
        // An even block count needs no padding.
        assert_eq!(encoding.leaf_code_bytes(8, 32).unwrap(), 64);
        assert_eq!(encoding.leaf_code_bytes(8, 64).unwrap(), 128);
    }
}
