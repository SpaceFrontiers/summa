//! Residual asymmetric hashing for float ScaNN.
//!
//! The anisotropic encoding objective is adapted from Google ScaNN 1.4.2,
//! `asymmetric_hashing_impl.cc`, Apache-2.0. See the adjacent `NOTICE` file.

use rand::{Rng, SeedableRng};

use super::{ScannAhCodebook, ScannAhCodebookRef, ScannFormatError, ScannResult};

pub const CENTERS_PER_BLOCK: usize = 16;
pub const DEFAULT_ANISOTROPIC_THRESHOLD: f32 = 0.2;

#[derive(Clone, Debug, PartialEq)]
pub struct AhCodebook {
    dimension: usize,
    dimensions_per_block: usize,
    /// Padded block-major, center-major, coordinate-major values.
    centers: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AhQuery {
    /// Block-major lookup table, 16 scores per block.
    values: Vec<f32>,
}

/// Reusable per-thread AH encoding workspace.
#[derive(Clone, Debug, Default)]
pub struct AhEncodeScratch {
    local_errors: Vec<[f32; CENTERS_PER_BLOCK]>,
    projections: Vec<[f32; CENTERS_PER_BLOCK]>,
}

impl AhCodebook {
    pub fn train(
        residuals: &[f32],
        points: usize,
        dimension: usize,
        dimensions_per_block: usize,
        iterations: usize,
        seed: u64,
    ) -> ScannResult<Self> {
        if points < CENTERS_PER_BLOCK
            || dimension == 0
            || dimensions_per_block == 0
            || dimensions_per_block > dimension
            || dimensions_per_block > u16::MAX as usize
            || residuals.len() != points.saturating_mul(dimension)
            || residuals.iter().any(|value| !value.is_finite())
        {
            return Err(ScannFormatError::new(
                "invalid ScaNN AH training data or block geometry",
            ));
        }
        let blocks = dimension.div_ceil(dimensions_per_block);
        let mut centers = Vec::with_capacity(blocks * CENTERS_PER_BLOCK * dimensions_per_block);
        for block in 0..blocks {
            let start = block * dimensions_per_block;
            let block_dimension = dimensions_per_block.min(dimension - start);
            let mut block_data = Vec::with_capacity(points * block_dimension);
            for row in residuals.chunks_exact(dimension) {
                block_data.extend_from_slice(&row[start..start + block_dimension]);
            }
            let trained = train_subspace(
                &block_data,
                points,
                block_dimension,
                iterations,
                seed.wrapping_add(0x517c_c1b7_2722_0a95u64.wrapping_mul(block as u64 + 1)),
            );
            for center in trained.chunks_exact(block_dimension) {
                centers.extend_from_slice(center);
                centers.resize(centers.len() + dimensions_per_block - block_dimension, 0.0);
            }
        }
        Ok(Self {
            dimension,
            dimensions_per_block,
            centers,
        })
    }

    /// Train residual AH codebooks directly from the original vectors and
    /// their terminal routing assignments. Only one AH subspace is
    /// materialized at a time, avoiding a second `points * dimension` float
    /// matrix beside the caller-owned training sample.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn train_from_assigned_vectors(
        data: &[f32],
        assignments: &[u32],
        leaf_centroids: &[f32],
        points: usize,
        dimension: usize,
        dimensions_per_block: usize,
        iterations: usize,
        seed: u64,
    ) -> ScannResult<Self> {
        if points < CENTERS_PER_BLOCK
            || dimension == 0
            || dimensions_per_block == 0
            || dimensions_per_block > dimension
            || dimensions_per_block > u16::MAX as usize
            || data.len() != points.saturating_mul(dimension)
            || assignments.len() != points
            || leaf_centroids.is_empty()
            || !leaf_centroids.len().is_multiple_of(dimension)
            || data.iter().any(|value| !value.is_finite())
            || leaf_centroids.iter().any(|value| !value.is_finite())
        {
            return Err(ScannFormatError::new(
                "invalid assigned ScaNN AH training data or block geometry",
            ));
        }
        let leaves = leaf_centroids.len() / dimension;
        if assignments.iter().any(|&leaf| leaf as usize >= leaves) {
            return Err(ScannFormatError::new(
                "ScaNN AH training assignment references a missing leaf",
            ));
        }
        Self::assigned_training_workspace_bytes(points, dimension, dimensions_per_block)?;

        let blocks = dimension.div_ceil(dimensions_per_block);
        let mut centers = Vec::with_capacity(blocks * CENTERS_PER_BLOCK * dimensions_per_block);
        for block in 0..blocks {
            let start = block * dimensions_per_block;
            let block_dimension = dimensions_per_block.min(dimension - start);
            let mut block_data = Vec::with_capacity(points * block_dimension);
            for (point, &leaf) in data.chunks_exact(dimension).zip(assignments) {
                let centroid =
                    &leaf_centroids[leaf as usize * dimension..(leaf as usize + 1) * dimension];
                block_data.extend(
                    (start..start + block_dimension)
                        .map(|coordinate| point[coordinate] - centroid[coordinate]),
                );
            }
            let trained = train_subspace(
                &block_data,
                points,
                block_dimension,
                iterations,
                seed.wrapping_add(0x517c_c1b7_2722_0a95u64.wrapping_mul(block as u64 + 1)),
            );
            for center in trained.chunks_exact(block_dimension) {
                centers.extend_from_slice(center);
                centers.resize(centers.len() + dimensions_per_block - block_dimension, 0.0);
            }
        }
        Ok(Self {
            dimension,
            dimensions_per_block,
            centers,
        })
    }

    /// Conservative peak temporary allocation for
    /// [`Self::train_from_assigned_vectors`], excluding caller-owned samples,
    /// routing assignments, centroids, and the returned codebook.
    pub(crate) fn assigned_training_workspace_bytes(
        points: usize,
        dimension: usize,
        dimensions_per_block: usize,
    ) -> ScannResult<usize> {
        if points < CENTERS_PER_BLOCK
            || dimension == 0
            || dimensions_per_block == 0
            || dimensions_per_block > dimension
        {
            return Err(ScannFormatError::new("invalid ScaNN AH workspace geometry"));
        }
        let block_dimension = dimensions_per_block.min(dimension);
        let block_values = points.checked_mul(block_dimension).ok_or_else(|| {
            ScannFormatError::new("ScaNN AH block workspace size overflows usize")
        })?;
        let center_values = CENTERS_PER_BLOCK
            .checked_mul(block_dimension)
            .ok_or_else(|| ScannFormatError::new("ScaNN AH center workspace overflows usize"))?;
        let nearest_bytes = points
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| ScannFormatError::new("ScaNN AH nearest workspace overflows usize"))?;
        let assignment_bytes = points
            .checked_mul(std::mem::size_of::<usize>())
            .ok_or_else(|| {
                ScannFormatError::new("ScaNN AH assignment workspace overflows usize")
            })?;
        let center_bytes = center_values
            .checked_mul(2 * std::mem::size_of::<f32>())
            .ok_or_else(|| ScannFormatError::new("ScaNN AH center workspace overflows usize"))?;
        let selected_and_count_bytes = 2 * CENTERS_PER_BLOCK * std::mem::size_of::<usize>();
        block_values
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(nearest_bytes))
            .and_then(|bytes| bytes.checked_add(assignment_bytes))
            .and_then(|bytes| bytes.checked_add(center_bytes))
            .and_then(|bytes| bytes.checked_add(selected_and_count_bytes))
            .ok_or_else(|| ScannFormatError::new("ScaNN AH workspace size overflows usize"))
    }

    pub fn from_artifact(dimension: usize, artifact: &ScannAhCodebook) -> ScannResult<Self> {
        if artifact.centers_per_block as usize != CENTERS_PER_BLOCK {
            return Err(ScannFormatError::new(
                "ScaNN AH artifact must have sixteen centers per block",
            ));
        }
        let dimensions_per_block = artifact.dimensions_per_block as usize;
        let blocks = dimension.div_ceil(dimensions_per_block.max(1));
        if dimension == 0
            || dimensions_per_block == 0
            || dimensions_per_block > dimension
            || artifact.centers.len() != blocks * CENTERS_PER_BLOCK * dimensions_per_block
            || artifact.centers.iter().any(|value| !value.is_finite())
        {
            return Err(ScannFormatError::new("invalid ScaNN AH artifact shape"));
        }
        Ok(Self {
            dimension,
            dimensions_per_block,
            centers: artifact.centers.clone(),
        })
    }

    /// Decode only the small AH codebook from a borrowed global-artifact view.
    /// Routing centroid planes remain mmap-backed in the executable model.
    pub fn from_artifact_ref(
        dimension: usize,
        artifact: ScannAhCodebookRef<'_>,
    ) -> ScannResult<Self> {
        if artifact.centers_per_block as usize != CENTERS_PER_BLOCK {
            return Err(ScannFormatError::new(
                "ScaNN AH artifact must have sixteen centers per block",
            ));
        }
        let dimensions_per_block = artifact.dimensions_per_block as usize;
        let blocks = dimension.div_ceil(dimensions_per_block.max(1));
        let centers: Vec<f32> = artifact.centers().collect();
        if dimension == 0
            || dimensions_per_block == 0
            || dimensions_per_block > dimension
            || centers.len() != blocks * CENTERS_PER_BLOCK * dimensions_per_block
            || centers.iter().any(|value| !value.is_finite())
        {
            return Err(ScannFormatError::new("invalid ScaNN AH artifact shape"));
        }
        Ok(Self {
            dimension,
            dimensions_per_block,
            centers,
        })
    }

    pub fn to_artifact(&self) -> ScannAhCodebook {
        ScannAhCodebook {
            dimensions_per_block: self.dimensions_per_block as u16,
            centers_per_block: CENTERS_PER_BLOCK as u16,
            centers: self.centers.clone(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn dimensions_per_block(&self) -> usize {
        self.dimensions_per_block
    }

    pub fn blocks(&self) -> usize {
        self.dimension.div_ceil(self.dimensions_per_block)
    }

    pub fn code_bytes(&self) -> usize {
        self.blocks().div_ceil(2)
    }

    pub fn estimated_memory_bytes(&self) -> usize {
        self.centers.len() * std::mem::size_of::<f32>()
    }

    /// Encode one residual with ScaNN's anisotropic direction-aware objective.
    /// Codes are unpacked nibbles to make streaming FastScan transposition cheap.
    pub fn encode(
        &self,
        residual: &[f32],
        original: &[f32],
        anisotropic_threshold: f32,
        codes: &mut [u8],
    ) -> ScannResult<()> {
        self.encode_with_scratch(
            residual,
            original,
            anisotropic_threshold,
            codes,
            &mut AhEncodeScratch::default(),
        )
    }

    pub fn encode_with_scratch(
        &self,
        residual: &[f32],
        original: &[f32],
        anisotropic_threshold: f32,
        codes: &mut [u8],
        scratch: &mut AhEncodeScratch,
    ) -> ScannResult<()> {
        if residual.len() != self.dimension
            || original.len() != self.dimension
            || codes.len() != self.blocks()
            || residual.iter().any(|value| !value.is_finite())
            || original.iter().any(|value| !value.is_finite())
            || !anisotropic_threshold.is_finite()
            || !(0.0..1.0).contains(&anisotropic_threshold)
        {
            return Err(ScannFormatError::new(
                "invalid ScaNN AH encode input or anisotropic threshold",
            ));
        }
        let norm = dot(original, original).sqrt();
        let inverse_norm = if norm.is_finite() && norm > f32::EPSILON {
            norm.recip()
        } else {
            0.0
        };
        scratch
            .local_errors
            .resize(self.blocks(), [0.0; CENTERS_PER_BLOCK]);
        scratch
            .projections
            .resize(self.blocks(), [0.0; CENTERS_PER_BLOCK]);
        let mut total_projection = 0.0f32;
        for (block, code) in codes.iter_mut().enumerate() {
            let (start, block_dimension) = self.block_shape(block);
            let mut best = (0usize, f32::INFINITY);
            for center in 0..CENTERS_PER_BLOCK {
                let center_values = self.center(block, center);
                let mut squared_error = 0.0;
                let mut projection = 0.0;
                for coordinate in 0..block_dimension {
                    let error = residual[start + coordinate] - center_values[coordinate];
                    squared_error += error * error;
                    projection += error * original[start + coordinate] * inverse_norm;
                }
                scratch.local_errors[block][center] = squared_error;
                scratch.projections[block][center] = projection;
                if squared_error < best.1 {
                    best = (center, squared_error);
                }
            }
            *code = best.0 as u8;
            total_projection += scratch.projections[block][best.0];
        }

        if inverse_norm == 0.0 || anisotropic_threshold == 0.0 {
            return Ok(());
        }
        let parallel_weight = anisotropic_threshold * anisotropic_threshold
            / (1.0 - anisotropic_threshold * anisotropic_threshold).max(f32::EPSILON)
            * self.dimension as f32;
        for _ in 0..3 {
            let mut changed = false;
            for (block, code) in codes.iter_mut().enumerate() {
                let current = *code as usize;
                let without_current = total_projection - scratch.projections[block][current];
                let mut best = current;
                let mut best_objective = scratch.local_errors[block][current]
                    + parallel_weight * total_projection * total_projection;
                for center in 0..CENTERS_PER_BLOCK {
                    let candidate_projection = without_current + scratch.projections[block][center];
                    let objective = scratch.local_errors[block][center]
                        + parallel_weight * candidate_projection * candidate_projection;
                    if objective < best_objective {
                        best = center;
                        best_objective = objective;
                    }
                }
                if best != current {
                    total_projection = without_current + scratch.projections[block][best];
                    *code = best as u8;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(())
    }

    pub fn encode_packed(
        &self,
        residual: &[f32],
        original: &[f32],
        anisotropic_threshold: f32,
        output: &mut [u8],
    ) -> ScannResult<()> {
        if output.len() != self.code_bytes() {
            return Err(ScannFormatError::new("invalid packed ScaNN AH code length"));
        }
        let mut codes = vec![0u8; self.blocks()];
        self.encode(residual, original, anisotropic_threshold, &mut codes)?;
        output.fill(0);
        for (block, &code) in codes.iter().enumerate() {
            output[block / 2] |= code << ((block % 2) * 4);
        }
        Ok(())
    }

    pub fn query_dot_product(&self, query: &[f32]) -> ScannResult<AhQuery> {
        if query.len() != self.dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(ScannFormatError::new("invalid ScaNN AH query vector"));
        }
        let mut values = Vec::with_capacity(self.blocks() * CENTERS_PER_BLOCK);
        for block in 0..self.blocks() {
            let (start, block_dimension) = self.block_shape(block);
            for center in 0..CENTERS_PER_BLOCK {
                values.push(dot(
                    &query[start..start + block_dimension],
                    &self.center(block, center)[..block_dimension],
                ));
            }
        }
        Ok(AhQuery { values })
    }

    fn block_shape(&self, block: usize) -> (usize, usize) {
        let start = block * self.dimensions_per_block;
        (start, self.dimensions_per_block.min(self.dimension - start))
    }

    fn center(&self, block: usize, center: usize) -> &[f32] {
        let offset = (block * CENTERS_PER_BLOCK + center) * self.dimensions_per_block;
        &self.centers[offset..offset + self.dimensions_per_block]
    }
}

impl AhQuery {
    pub fn blocks(&self) -> usize {
        self.values.len() / CENTERS_PER_BLOCK
    }

    pub fn score_unpacked(&self, codes: &[u8], centroid_dot: f32) -> ScannResult<f32> {
        if codes.len() != self.blocks()
            || codes.iter().any(|&code| code as usize >= CENTERS_PER_BLOCK)
        {
            return Err(ScannFormatError::new("invalid unpacked ScaNN AH codes"));
        }
        Ok(centroid_dot
            + codes
                .iter()
                .enumerate()
                .map(|(block, &code)| self.values[block * CENTERS_PER_BLOCK + code as usize])
                .sum::<f32>())
    }

    pub fn score_packed(&self, codes: &[u8], centroid_dot: f32) -> ScannResult<f32> {
        if codes.len() != self.blocks().div_ceil(2) {
            return Err(ScannFormatError::new("invalid packed ScaNN AH codes"));
        }
        let mut score = centroid_dot;
        for block in 0..self.blocks() {
            let code = (codes[block / 2] >> ((block % 2) * 4)) & 0x0f;
            score += self.values[block * CENTERS_PER_BLOCK + code as usize];
        }
        Ok(score)
    }

    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }
}

fn train_subspace(
    data: &[f32],
    points: usize,
    dimension: usize,
    iterations: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut selected = Vec::with_capacity(CENTERS_PER_BLOCK);
    selected.push(rng.random_range(0..points));
    let mut nearest = vec![f32::INFINITY; points];
    while selected.len() < CENTERS_PER_BLOCK {
        let last = &data[selected[selected.len() - 1] * dimension..][..dimension];
        for (row, point) in data.chunks_exact(dimension).enumerate() {
            nearest[row] = nearest[row].min(squared_l2(point, last));
        }
        let total: f64 = nearest.iter().map(|&value| f64::from(value)).sum();
        let next = if total > 0.0 && total.is_finite() {
            let target = rng.random::<f64>() * total;
            let mut sum = 0.0;
            nearest
                .iter()
                .position(|&value| {
                    sum += f64::from(value);
                    sum > target
                })
                .unwrap_or(points - 1)
        } else {
            (0..points)
                .find(|candidate| !selected.contains(candidate))
                .unwrap_or(0)
        };
        selected.push(next);
    }
    let mut centers: Vec<f32> = selected
        .iter()
        .flat_map(|&row| data[row * dimension..(row + 1) * dimension].iter().copied())
        .collect();
    let mut assignments = vec![0usize; points];
    for _ in 0..iterations.max(1) {
        for (row, point) in data.chunks_exact(dimension).enumerate() {
            assignments[row] = centers
                .chunks_exact(dimension)
                .enumerate()
                .map(|(center, values)| (center, squared_l2(point, values)))
                .min_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.cmp(&right.0))
                })
                .unwrap()
                .0;
        }
        let mut sums = vec![0.0f32; centers.len()];
        let mut counts = [0usize; CENTERS_PER_BLOCK];
        for (point, &center) in data.chunks_exact(dimension).zip(&assignments) {
            counts[center] += 1;
            for coordinate in 0..dimension {
                sums[center * dimension + coordinate] += point[coordinate];
            }
        }
        for center in 0..CENTERS_PER_BLOCK {
            if counts[center] == 0 {
                let replacement = selected[center];
                sums[center * dimension..(center + 1) * dimension]
                    .copy_from_slice(&data[replacement * dimension..(replacement + 1) * dimension]);
            } else {
                let inverse = (counts[center] as f32).recip();
                for value in &mut sums[center * dimension..(center + 1) * dimension] {
                    *value *= inverse;
                }
            }
        }
        centers = sums;
    }
    centers
}

#[inline]
fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| {
            let difference = a - b;
            difference * difference
        })
        .sum()
}

#[inline]
fn dot(left: &[f32], right: &[f32]) -> f32 {
    crate::structures::simd::dot_product_f32(left, right, left.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training_data() -> Vec<f32> {
        (0..256)
            .flat_map(|row| {
                (0..7)
                    .map(move |coordinate| ((row * 17 + coordinate * 29) % 101) as f32 / 50.0 - 1.0)
            })
            .collect()
    }

    #[test]
    fn ah_training_and_artifact_roundtrip_are_deterministic() {
        let data = training_data();
        let first = AhCodebook::train(&data, 256, 7, 2, 6, 42).unwrap();
        let second = AhCodebook::train(&data, 256, 7, 2, 6, 42).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            AhCodebook::from_artifact(7, &first.to_artifact()).unwrap(),
            first
        );
    }

    #[test]
    fn assigned_vector_training_matches_materialized_residual_training() {
        let data = training_data();
        let dimension = 7usize;
        let points = 256usize;
        let leaves = 4usize;
        let leaf_centroids: Vec<f32> = (0..leaves)
            .flat_map(|leaf| {
                (0..dimension)
                    .map(move |coordinate| (leaf as f32 - 1.5) * 0.1 + coordinate as f32 * 0.003)
            })
            .collect();
        let assignments: Vec<u32> = (0..points).map(|row| (row % leaves) as u32).collect();
        let residuals: Vec<f32> = data
            .chunks_exact(dimension)
            .zip(&assignments)
            .flat_map(|(point, &leaf)| {
                let centroid =
                    &leaf_centroids[leaf as usize * dimension..(leaf as usize + 1) * dimension];
                point
                    .iter()
                    .zip(centroid)
                    .map(|(&value, &center)| value - center)
            })
            .collect();

        let materialized = AhCodebook::train(&residuals, points, dimension, 2, 6, 42).unwrap();
        let blockwise = AhCodebook::train_from_assigned_vectors(
            &data,
            &assignments,
            &leaf_centroids,
            points,
            dimension,
            2,
            6,
            42,
        )
        .unwrap();
        assert_eq!(blockwise, materialized);
    }

    #[test]
    fn assigned_training_workspace_is_block_bounded_at_default_billion_scale_budget() {
        let points = 1_048_576usize;
        let dimension = 1_024usize;
        let dimensions_per_block = 2usize;
        let sample_bytes = points * dimension * std::mem::size_of::<f32>();
        let workspace =
            AhCodebook::assigned_training_workspace_bytes(points, dimension, dimensions_per_block)
                .unwrap();

        assert!(workspace < sample_bytes / 100);
        assert!(workspace < 24 * 1024 * 1024);
    }

    #[test]
    fn packed_and_unpacked_scores_match() {
        let data = training_data();
        let codebook = AhCodebook::train(&data, 256, 7, 2, 5, 9).unwrap();
        let vector = &data[..7];
        let mut unpacked = vec![0u8; codebook.blocks()];
        codebook
            .encode(vector, vector, DEFAULT_ANISOTROPIC_THRESHOLD, &mut unpacked)
            .unwrap();
        let mut packed = vec![0u8; codebook.code_bytes()];
        codebook
            .encode_packed(vector, vector, DEFAULT_ANISOTROPIC_THRESHOLD, &mut packed)
            .unwrap();
        let query = codebook.query_dot_product(vector).unwrap();
        assert_eq!(
            query.score_unpacked(&unpacked, 0.25).unwrap(),
            query.score_packed(&packed, 0.25).unwrap()
        );
    }
}
