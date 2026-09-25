//! Recall-oriented ScaNN partition geometry.
//!
//! The constants mirror the freshly validated ScaNN builder: do not
//! expose a configurable minimum-training count. Readiness and sample size are
//! derived from the selected geometry.

use super::{
    MAX_SCANN_LEAVES, MAX_SCANN_TREE_LEVELS, MIN_PARTITION_TRAINING_POINTS_PER_LEAF,
    MIN_POINTS_FOR_PARTITIONING, ScannFormatError, ScannResult,
};

/// Higher-quality explicit geometry target documented by AlloyDB. Operators
/// can ask for `rows / QUALITY_OPTIMIZED_POINTS_PER_LEAF` leaves explicitly
/// when the additional build cost is justified by measurements on their
/// corpus.
pub const QUALITY_OPTIMIZED_POINTS_PER_LEAF: u64 = 100;
/// AlloyDB's recall-oriented balanced guidance changes the leaf exponent with
/// tree depth. Keep exactly one billion rows in the three-level band: it is a
/// useful, measured recall point, while the four-level band prioritizes build
/// scalability above that boundary.
const THREE_LEVEL_MIN_POINTS: u64 = 100_000_000;
const FOUR_LEVEL_MIN_POINTS_EXCLUSIVE: u64 = 1_000_000_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScannGeometry {
    pub centroid_levels: u8,
    pub num_leaves: u32,
    /// Centroid count at each routing level, ending in `num_leaves`.
    pub level_counts: Vec<u32>,
}

/// Derive automatic production geometry for an observed corpus.
pub fn derive_geometry(points: u64, dimension: u32) -> ScannResult<ScannGeometry> {
    derive_geometry_with_levels(points, dimension, None)
}

/// Derive geometry while allowing the schema-selected tree depth to override
/// automatic depth. A zero-level tree is returned below the hardcoded
/// partitioning floor; callers should keep serving the exact/flat generation.
pub fn derive_geometry_with_levels(
    points: u64,
    dimension: u32,
    requested_levels: Option<u8>,
) -> ScannResult<ScannGeometry> {
    derive_geometry_with_levels_and_sample_limit(points, dimension, requested_levels, points)
}

/// Derive automatic geometry and validate the number of training rows the
/// builder can actually retain. The selected topology depends only on corpus
/// geometry, never on a transient resource ceiling. A machine that cannot
/// retain the hardcoded minimum sample must fail/defer the build rather than
/// publish a different shared codebook shape.
pub fn derive_geometry_with_levels_and_sample_limit(
    points: u64,
    dimension: u32,
    requested_levels: Option<u8>,
    sample_limit: u64,
) -> ScannResult<ScannGeometry> {
    if points == 0 {
        return Err(ScannFormatError::new(
            "ScaNN geometry requires at least one vector",
        ));
    }
    if dimension == 0 {
        return Err(ScannFormatError::new(
            "ScaNN geometry requires a non-zero vector dimension",
        ));
    }
    if let Some(levels) = requested_levels
        && !(1..=MAX_SCANN_TREE_LEVELS).contains(&levels)
    {
        return Err(ScannFormatError::new(format!(
            "ScaNN tree levels must be in 1..={MAX_SCANN_TREE_LEVELS}"
        )));
    }
    if points < MIN_POINTS_FOR_PARTITIONING {
        return Ok(ScannGeometry {
            centroid_levels: 0,
            num_leaves: 1,
            level_counts: Vec::new(),
        });
    }

    let (desired_leaves, corpus_min_levels) = automatic_leaf_target(points);
    let leaves = desired_leaves.min(u64::from(MAX_SCANN_LEAVES)) as u32;
    let levels = requested_levels
        .unwrap_or_else(|| corpus_min_levels.max(width_required_levels(leaves, dimension)));
    let geometry = geometry_for_leaves(leaves, levels)?;
    let required_sample = u64::from(leaves)
        .checked_mul(MIN_PARTITION_TRAINING_POINTS_PER_LEAF)
        .ok_or_else(|| ScannFormatError::new("ScaNN minimum training sample overflows u64"))?
        .max(MIN_POINTS_FOR_PARTITIONING);
    let achievable_sample = points.min(sample_limit);
    if achievable_sample < required_sample {
        return Err(ScannFormatError::new(format!(
            "ScaNN automatic geometry selected {leaves} leaves and needs at least {required_sample} training samples (hardcoded {MIN_PARTITION_TRAINING_POINTS_PER_LEAF} samples/leaf), but the builder can supply {achievable_sample}"
        )));
    }
    Ok(geometry)
}

/// Construct and validate an explicit trained geometry.
pub fn geometry_for_leaves(num_leaves: u32, levels: u8) -> ScannResult<ScannGeometry> {
    if !(2..=MAX_SCANN_LEAVES).contains(&num_leaves) {
        return Err(ScannFormatError::new(format!(
            "ScaNN leaf count must be in 2..={MAX_SCANN_LEAVES}"
        )));
    }
    if !(1..=MAX_SCANN_TREE_LEVELS).contains(&levels) {
        return Err(ScannFormatError::new(format!(
            "ScaNN tree levels must be in 1..={MAX_SCANN_TREE_LEVELS}"
        )));
    }
    let factors = balanced_branching_factors(num_leaves, levels);
    let mut product = 1u64;
    let mut level_counts = Vec::with_capacity(usize::from(levels));
    for (level, factor) in factors.into_iter().enumerate() {
        product = product.saturating_mul(u64::from(factor));
        level_counts.push(if level + 1 == usize::from(levels) {
            num_leaves
        } else {
            product.min(u64::from(num_leaves)) as u32
        });
    }
    Ok(ScannGeometry {
        centroid_levels: levels,
        num_leaves,
        level_counts,
    })
}

/// Construct explicit leaf geometry, deriving only its routing depth when the
/// schema does not pin one. Explicit leaf counts must not inherit corpus-size
/// depth bands: those bands choose automatic leaf counts, not the shape of an
/// operator-selected codebook.
pub fn geometry_for_leaves_with_auto_depth(
    num_leaves: u32,
    dimension: u32,
    requested_levels: Option<u8>,
) -> ScannResult<ScannGeometry> {
    if dimension == 0 {
        return Err(ScannFormatError::new(
            "ScaNN geometry requires a non-zero vector dimension",
        ));
    }
    geometry_for_leaves(
        num_leaves,
        requested_levels.unwrap_or_else(|| width_required_levels(num_leaves, dimension)),
    )
}

/// Hardcoded recall-oriented training sample derived from geometry.
pub fn desired_training_sample(observed: u64, num_leaves: u32) -> u64 {
    let desired = u64::from(num_leaves)
        .saturating_mul(super::PARTITION_TRAINING_POINTS_PER_CENTROID)
        .max(super::DEFAULT_TRAINING_SAMPLE_SIZE);
    observed.min(desired)
}

fn automatic_leaf_target(points: u64) -> (u64, u8) {
    if points < THREE_LEVEL_MIN_POINTS {
        (fractional_power_ceil(points, 1, 2), 1)
    } else if points <= FOUR_LEVEL_MIN_POINTS_EXCLUSIVE {
        (fractional_power_ceil(points, 2, 3), 2)
    } else {
        (fractional_power_ceil(points, 3, 4), 3)
    }
}

fn width_required_levels(leaves: u32, dimension: u32) -> u8 {
    let max_branching = flat_training_width_bound(dimension);
    for levels in 1..MAX_SCANN_TREE_LEVELS {
        if nth_root_ceil(leaves, levels) <= max_branching {
            return levels;
        }
    }
    MAX_SCANN_TREE_LEVELS
}

/// ScaNN autopilot's bounded flat k-means work estimate.
fn flat_training_width_bound(dimension: u32) -> u32 {
    let numerator = 60.0 * 32.0 * 2.0e9;
    let denominator = f64::from(dimension) * super::PARTITION_TRAINING_POINTS_PER_CENTROID as f64;
    (numerator / denominator).sqrt().ceil().max(1.0) as u32
}

fn balanced_branching_factors(leaves: u32, levels: u8) -> Vec<u32> {
    let mut remaining = leaves;
    let mut factors = Vec::with_capacity(usize::from(levels));
    for levels_left in (1..=levels).rev() {
        let factor = nth_root_ceil(remaining, levels_left);
        factors.push(factor);
        remaining = remaining.div_ceil(factor);
    }
    factors
}

fn nth_root_ceil(value: u32, degree: u8) -> u32 {
    if value <= 1 || degree == 1 {
        return value;
    }
    let mut low = 1u32;
    let mut high = value;
    while low < high {
        let middle = low + (high - low) / 2;
        if pow_reaches(middle, degree, value) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

fn fractional_power_ceil(value: u64, numerator: u8, denominator: u8) -> u64 {
    debug_assert!(value > 0 && numerator > 0 && numerator < denominator);
    let target = saturating_pow_u128(u128::from(value), numerator);
    let mut low = 1u64;
    // Automatic geometry clamps at the format cap. One past that cap is enough
    // to tell the caller the fractional-power target is larger.
    let mut high = value.min(u64::from(MAX_SCANN_LEAVES) + 1).max(2);
    while low < high {
        let middle = low + (high - low) / 2;
        if saturating_pow_u128(u128::from(middle), denominator) >= target {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

fn saturating_pow_u128(base: u128, exponent: u8) -> u128 {
    (0..exponent).fold(1u128, |product, _| product.saturating_mul(base))
}

fn pow_reaches(base: u32, degree: u8, target: u32) -> bool {
    let mut product = 1u64;
    for _ in 0..degree {
        product = product.saturating_mul(u64::from(base));
        if product >= u64::from(target) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_billion_scale_balanced_width() {
        let geometry = derive_geometry(1_000_000_000, 1_024).unwrap();
        assert_eq!(geometry.centroid_levels, 2);
        assert_eq!(geometry.level_counts, [1_000, 1_000_000]);
        assert_eq!(
            desired_training_sample(1_000_000_000, 1_000_000),
            200_000_000
        );
    }

    #[test]
    fn billion_scale_float_geometry_rejects_an_inadequate_default_sample() {
        let sample_limit = (4_u64 * 1024 * 1024 * 1024) / (1_024 * 4);
        assert_eq!(sample_limit, 1_048_576);
        let error =
            derive_geometry_with_levels_and_sample_limit(1_000_000_000, 1_024, None, sample_limit)
                .unwrap_err()
                .to_string();
        assert!(error.contains("1000000 leaves"));
        assert!(error.contains("8000000 training samples"));
    }

    #[test]
    fn billion_scale_binary_geometry_fits_default_training_budget() {
        // The default 10M row cap is tighter than 4 GiB / 320-byte rows.
        let sample_limit = 10_000_000_u64;
        let geometry =
            derive_geometry_with_levels_and_sample_limit(1_000_000_000, 2_560, None, sample_limit)
                .unwrap();
        assert_eq!(sample_limit, 10_000_000);
        assert_eq!(geometry.centroid_levels, 2);
        assert_eq!(geometry.level_counts, [1_000, 1_000_000]);
        assert!(
            u64::from(geometry.num_leaves) * MIN_PARTITION_TRAINING_POINTS_PER_LEAF <= sample_limit
        );
    }

    #[test]
    fn fifteen_million_rows_use_the_measured_balanced_geometry() {
        let geometry = derive_geometry(15_000_000, 2_560).unwrap();
        assert_eq!(geometry.centroid_levels, 2);
        assert_eq!(geometry.level_counts, [63, 3_873]);
        assert_eq!(desired_training_sample(15_000_000, 3_873), 774_600);
    }

    #[test]
    fn automatic_geometry_follows_google_tree_depth_bands() {
        assert_eq!(
            derive_geometry(99_999_999, 2_560).unwrap().level_counts,
            [100, 10_000]
        );
        assert_eq!(
            derive_geometry(100_000_000, 2_560).unwrap().level_counts,
            [465, 215_444]
        );
        assert_eq!(
            derive_geometry(1_000_000_001, 2_560).unwrap().level_counts,
            [178, 31_684, 5_623_414]
        );
        assert_eq!(
            derive_geometry(10_000_000_000, 2_560).unwrap().level_counts,
            [311, 96_721, 30_000_000]
        );
    }

    #[test]
    fn sample_limit_never_changes_the_selected_topology() {
        let expected = derive_geometry(1_000_000_000, 2_560).unwrap();
        assert!(
            derive_geometry_with_levels_and_sample_limit(1_000_000_000, 2_560, None, 7_999_999,)
                .is_err()
        );
        assert_eq!(
            derive_geometry_with_levels_and_sample_limit(1_000_000_000, 2_560, None, 8_000_000,)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn fractional_power_is_exact_at_and_between_perfect_powers() {
        assert_eq!(fractional_power_ceil(1_000_000_000, 2, 3), 1_000_000);
        assert_eq!(fractional_power_ceil(1_000_000_001, 2, 3), 1_000_001);
        assert_eq!(fractional_power_ceil(1_000_000_000, 3, 4), 5_623_414);
        assert_eq!(
            fractional_power_ceil(u64::MAX, 3, 4),
            u64::from(MAX_SCANN_LEAVES) + 1
        );
    }

    #[test]
    fn geometry_defers_partitioning_below_hardcoded_floor() {
        let geometry = derive_geometry(99_999, 1_024).unwrap();
        assert_eq!(geometry.centroid_levels, 0);
        assert!(geometry.level_counts.is_empty());
    }

    #[test]
    fn explicit_depth_is_validated_and_balanced() {
        assert_eq!(
            geometry_for_leaves(10_000, 3).unwrap().level_counts,
            [22, 484, 10_000]
        );
        assert!(geometry_for_leaves(10_000, 4).is_err());
    }

    #[test]
    fn explicit_leaf_depth_depends_on_width_not_synthetic_corpus_bands() {
        assert_eq!(
            geometry_for_leaves_with_auto_depth(31_622, 2_560, None)
                .unwrap()
                .level_counts,
            [178, 31_622]
        );
        assert_eq!(
            geometry_for_leaves_with_auto_depth(31_623, 2_560, None)
                .unwrap()
                .level_counts,
            [178, 31_623]
        );
        assert_eq!(
            geometry_for_leaves_with_auto_depth(1_000_000, 1_024, None)
                .unwrap()
                .level_counts,
            [1_000, 1_000_000]
        );
    }

    #[test]
    fn automatic_geometry_partitions_at_the_exact_hardcoded_floor() {
        assert_eq!(
            derive_geometry(MIN_POINTS_FOR_PARTITIONING, 1_024)
                .unwrap()
                .level_counts,
            [317]
        );
    }
}
