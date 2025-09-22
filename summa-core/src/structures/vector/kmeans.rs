//! Deterministic Euclidean k-means shared by coarse and product quantizers.
//!
//! k-means|| seeding reduces the number of full-data synchronization passes
//! needed to initialize large codebooks. Lloyd assignment is parallel on
//! native builds, while candidate selection, objective reduction, and centroid
//! reduction stay deterministic.

use rand::prelude::*;

const WEIGHT_REDUCTION_BLOCK: usize = 16 * 1024;
const KMEANS_PARALLEL_ROUNDS: usize = 5;
const KMEANS_PARALLEL_TOTAL_OVERSAMPLING: usize = 2;
const KMEANS_PLUS_PLUS_MAX_CLUSTERS: usize = 256;
const RELATIVE_OBJECTIVE_TOLERANCE: f64 = 1.0e-5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InitializationStrategy {
    KMeansPlusPlus,
    KMeansParallel,
}

pub(crate) struct EuclideanKMeans {
    pub centroids: Vec<f32>,
    #[cfg(test)]
    pub assignments: Vec<usize>,
    /// Cluster-major ranges into `members`; length is `clusters + 1`.
    pub member_offsets: Vec<usize>,
    /// Every training point exactly once, grouped by final cluster.
    pub members: Vec<usize>,
}

#[inline]
fn initialization_strategy(clusters: usize) -> InitializationStrategy {
    if clusters <= KMEANS_PLUS_PLUS_MAX_CLUSTERS {
        InitializationStrategy::KMeansPlusPlus
    } else {
        InitializationStrategy::KMeansParallel
    }
}

/// Conservative full-Lloyd-pass equivalent for build-work admission control.
///
/// One unit is `points * clusters` point/centroid distance evaluations.
/// Small codebooks spend one such unit in k-means++ initialization; large
/// codebooks spend up to two units comparing points to the total 2K candidate
/// budget, plus `ceil(candidate_count / points)` for weighted candidate
/// reduction. Candidate weights reuse the oversampling assignments, so there
/// is no second points-by-candidates scan. The final unit keeps returned
/// assignments synchronized with the returned centroids.
#[cfg(any(feature = "native", test))]
pub(crate) fn estimated_euclidean_kmeans_distance_multiplier(
    points: usize,
    clusters: usize,
    max_iters: usize,
) -> usize {
    if points == 0 || clusters == 0 {
        return 0;
    }
    debug_assert!(clusters <= points);
    let initialization = match initialization_strategy(clusters) {
        InitializationStrategy::KMeansPlusPlus => 1,
        InitializationStrategy::KMeansParallel => {
            let candidates = kmeans_parallel_candidate_budget(points, clusters);
            candidates
                .div_ceil(clusters)
                .saturating_add(candidates.div_ceil(points))
        }
    };
    initialization
        .saturating_add(max_iters.max(1))
        .saturating_add(1)
}

#[inline]
fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    crate::structures::simd::squared_l2_f32(left, right)
}

#[inline]
fn nearest(centroids: &[f32], point: &[f32], dim: usize) -> (usize, f32) {
    let mut best = (0usize, f32::INFINITY);
    for (index, centroid) in centroids.chunks_exact(dim).enumerate() {
        let distance = squared_l2(point, centroid);
        if distance < best.1 || (distance == best.1 && index < best.0) {
            best = (index, distance);
        }
    }
    best
}

fn update_centroid(centroid: &mut [f32], members: &[usize], data: &[f32], dim: usize) {
    for &point_index in members {
        let point = &data[point_index * dim..(point_index + 1) * dim];
        for (sum, &value) in centroid.iter_mut().zip(point) {
            *sum += value;
        }
    }
    let inverse = 1.0 / members.len() as f32;
    for value in centroid {
        *value *= inverse;
    }
}

fn fixed_weight_block_totals(weights: &[f64]) -> Vec<f64> {
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        weights
            .par_chunks(WEIGHT_REDUCTION_BLOCK)
            .map(|block| block.iter().sum())
            .collect()
    }
    #[cfg(not(feature = "native"))]
    {
        weights
            .chunks(WEIGHT_REDUCTION_BLOCK)
            .map(|block| block.iter().sum())
            .collect()
    }
}

/// Draw from non-negative weights with a fixed reduction topology. Parallel
/// block sums avoid a serial O(N) reduction while producing the same choice
/// for every rayon thread count.
pub(crate) fn weighted_sample_index(weights: &[f64], draw: f64) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }
    let block_totals = fixed_weight_block_totals(weights);

    let total: f64 = block_totals.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mut target = draw * total;
    let block = block_totals
        .iter()
        .position(|weight| {
            if target < *weight {
                true
            } else {
                target -= *weight;
                false
            }
        })
        .unwrap_or(block_totals.len() - 1);
    let start = block * WEIGHT_REDUCTION_BLOCK;
    let end = (start + WEIGHT_REDUCTION_BLOCK).min(weights.len());
    weights[start..end]
        .iter()
        .position(|weight| {
            if target < *weight {
                true
            } else {
                target -= *weight;
                false
            }
        })
        .map(|index| start + index)
        .or_else(|| end.checked_sub(1))
}

fn initialize_kmeans_plus_plus(
    data: &[f32],
    points: usize,
    dim: usize,
    clusters: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut centroids = Vec::with_capacity(clusters.saturating_mul(dim));
    let first = rng.random_range(0..points);
    centroids.extend_from_slice(&data[first * dim..(first + 1) * dim]);
    let mut minimum_distances = vec![f64::INFINITY; points];

    while centroids.len() < clusters * dim {
        let latest = &centroids[centroids.len() - dim..];
        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            minimum_distances
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, minimum)| {
                    let point = &data[index * dim..(index + 1) * dim];
                    *minimum = minimum.min(squared_l2(point, latest) as f64);
                });
        }
        #[cfg(not(feature = "native"))]
        for (index, minimum) in minimum_distances.iter_mut().enumerate() {
            let point = &data[index * dim..(index + 1) * dim];
            *minimum = minimum.min(squared_l2(point, latest) as f64);
        }

        let selected = weighted_sample_index(&minimum_distances, rng.random::<f64>())
            .unwrap_or((centroids.len() / dim) % points);
        centroids.extend_from_slice(&data[selected * dim..(selected + 1) * dim]);
    }
    centroids
}

fn update_point_minimum_distances(
    data: &[f32],
    points: usize,
    dim: usize,
    center_indices: &[usize],
    center_offset: usize,
    minimum_distances: &mut [f64],
    nearest_centers: &mut [usize],
) {
    debug_assert_eq!(minimum_distances.len(), points);
    debug_assert_eq!(nearest_centers.len(), points);
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        minimum_distances
            .par_iter_mut()
            .zip(nearest_centers.par_iter_mut())
            .enumerate()
            .for_each(|(index, (minimum, nearest_center))| {
                let point = &data[index * dim..(index + 1) * dim];
                for (offset, &center_index) in center_indices.iter().enumerate() {
                    let center = &data[center_index * dim..(center_index + 1) * dim];
                    let distance = squared_l2(point, center) as f64;
                    let candidate = center_offset + offset;
                    if distance < *minimum || (distance == *minimum && candidate < *nearest_center)
                    {
                        *minimum = distance;
                        *nearest_center = candidate;
                    }
                }
            });
    }
    #[cfg(not(feature = "native"))]
    for (index, (minimum, nearest_center)) in minimum_distances
        .iter_mut()
        .zip(nearest_centers)
        .enumerate()
    {
        let point = &data[index * dim..(index + 1) * dim];
        for (offset, &center_index) in center_indices.iter().enumerate() {
            let center = &data[center_index * dim..(center_index + 1) * dim];
            let distance = squared_l2(point, center) as f64;
            let candidate = center_offset + offset;
            if distance < *minimum || (distance == *minimum && candidate < *nearest_center) {
                *minimum = distance;
                *nearest_center = candidate;
            }
        }
    }
}

fn update_candidate_minimum_distances(
    data: &[f32],
    dim: usize,
    candidate_indices: &[usize],
    selected_candidate: usize,
    minimum_distances: &mut [f64],
) {
    let center_index = candidate_indices[selected_candidate];
    let center = &data[center_index * dim..(center_index + 1) * dim];
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        minimum_distances
            .par_iter_mut()
            .enumerate()
            .for_each(|(candidate, minimum)| {
                let point_index = candidate_indices[candidate];
                let point = &data[point_index * dim..(point_index + 1) * dim];
                *minimum = minimum.min(squared_l2(point, center) as f64);
            });
    }
    #[cfg(not(feature = "native"))]
    for (candidate, minimum) in minimum_distances.iter_mut().enumerate() {
        let point_index = candidate_indices[candidate];
        let point = &data[point_index * dim..(point_index + 1) * dim];
        *minimum = minimum.min(squared_l2(point, center) as f64);
    }
}

fn reduce_kmeans_parallel_candidates(
    data: &[f32],
    dim: usize,
    clusters: usize,
    candidate_indices: &[usize],
    candidate_assignments: &[usize],
    rng: &mut rand::rngs::StdRng,
) -> Vec<f32> {
    debug_assert!(candidate_indices.len() >= clusters);
    debug_assert_eq!(candidate_assignments.len(), data.len() / dim);
    if candidate_indices.len() == clusters {
        return candidate_indices
            .iter()
            .flat_map(|&index| data[index * dim..(index + 1) * dim].iter().copied())
            .collect();
    }

    let mut candidate_weights = vec![0usize; candidate_indices.len()];
    for &candidate in candidate_assignments {
        candidate_weights[candidate] += 1;
    }
    let mut sampling_weights: Vec<f64> = candidate_weights
        .iter()
        .map(|&weight| weight as f64)
        .collect();
    let first = weighted_sample_index(&sampling_weights, rng.random::<f64>()).unwrap_or_default();
    let mut selected = vec![false; candidate_indices.len()];
    selected[first] = true;
    let mut selected_candidates = Vec::with_capacity(clusters);
    selected_candidates.push(first);
    let mut minimum_distances = vec![f64::INFINITY; candidate_indices.len()];

    while selected_candidates.len() < clusters {
        let latest = *selected_candidates.last().unwrap();
        update_candidate_minimum_distances(
            data,
            dim,
            candidate_indices,
            latest,
            &mut minimum_distances,
        );
        for (candidate, weight) in sampling_weights.iter_mut().enumerate() {
            *weight = if selected[candidate] {
                0.0
            } else {
                candidate_weights[candidate] as f64 * minimum_distances[candidate]
            };
        }
        let next = weighted_sample_index(&sampling_weights, rng.random::<f64>()).or_else(|| {
            // Identical candidates have zero D² weight. Retain deterministic
            // multiplicity by selecting the heaviest remaining candidate,
            // breaking ties by its stable candidate position.
            let mut best = None;
            for (candidate, &weight) in candidate_weights.iter().enumerate() {
                if selected[candidate] {
                    continue;
                }
                if best.is_none_or(|best_candidate| {
                    weight > candidate_weights[best_candidate]
                        || (weight == candidate_weights[best_candidate]
                            && candidate < best_candidate)
                }) {
                    best = Some(candidate);
                }
            }
            best
        });
        let Some(next) = next else {
            break;
        };
        selected[next] = true;
        selected_candidates.push(next);
    }

    selected_candidates
        .iter()
        .flat_map(|&candidate| {
            let point_index = candidate_indices[candidate];
            data[point_index * dim..(point_index + 1) * dim]
                .iter()
                .copied()
        })
        .collect()
}

#[inline]
fn kmeans_parallel_candidate_budget(points: usize, clusters: usize) -> usize {
    points.min(
        clusters
            .saturating_mul(KMEANS_PARALLEL_TOTAL_OVERSAMPLING)
            .max(clusters),
    )
}

fn initialize_kmeans_parallel(
    data: &[f32],
    points: usize,
    dim: usize,
    clusters: usize,
    seed: u64,
) -> (Vec<f32>, usize) {
    if clusters == points {
        return (data.to_vec(), clusters);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let first = rng.random_range(0..points);
    let candidate_budget = kmeans_parallel_candidate_budget(points, clusters);
    let mut candidate_indices = Vec::with_capacity(candidate_budget);
    candidate_indices.push(first);
    let mut selected = vec![false; points];
    selected[first] = true;
    let mut minimum_distances = vec![f64::INFINITY; points];
    let mut nearest_candidates = vec![usize::MAX; points];
    update_point_minimum_distances(
        data,
        points,
        dim,
        &[first],
        0,
        &mut minimum_distances,
        &mut nearest_candidates,
    );
    for round in 0..KMEANS_PARALLEL_ROUNDS {
        if candidate_indices.len() >= candidate_budget {
            break;
        }
        let potential: f64 = fixed_weight_block_totals(&minimum_distances)
            .into_iter()
            .sum();
        if !potential.is_finite() || potential <= 0.0 {
            break;
        }
        let remaining_rounds = KMEANS_PARALLEL_ROUNDS - round;
        let round_budget = (candidate_budget - candidate_indices.len()).div_ceil(remaining_rounds);
        let mut sampled = Vec::new();
        for (point, &distance) in minimum_distances.iter().enumerate() {
            if selected[point] {
                continue;
            }
            let probability = (round_budget as f64 * distance / potential).min(1.0);
            let draw = rng.random::<f64>();
            if probability > 0.0 && draw < probability {
                // Conditional on acceptance, draw / probability is uniform.
                // It provides an order-independent deterministic cap when a
                // Bernoulli round samples more than its allotted work.
                sampled.push((point, draw / probability));
            }
        }
        if sampled.len() > round_budget {
            sampled.sort_unstable_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            sampled.truncate(round_budget);
        }
        let mut round_candidates: Vec<usize> =
            sampled.into_iter().map(|(point, _)| point).collect();
        if round_candidates.is_empty()
            && let Some(point) = weighted_sample_index(&minimum_distances, rng.random::<f64>())
            && !selected[point]
        {
            round_candidates.push(point);
        }
        if round_candidates.is_empty() {
            break;
        }
        round_candidates.sort_unstable();
        for &point in &round_candidates {
            selected[point] = true;
        }
        let candidate_offset = candidate_indices.len();
        update_point_minimum_distances(
            data,
            points,
            dim,
            &round_candidates,
            candidate_offset,
            &mut minimum_distances,
            &mut nearest_candidates,
        );
        candidate_indices.extend(round_candidates);
    }

    if candidate_indices.len() < clusters {
        // Oversampling is probabilistic. Guarantee enough distinct source
        // points for the weighted reduction in one deterministic recovery pass.
        let mut remaining: Vec<usize> = (0..points).filter(|&point| !selected[point]).collect();
        remaining.sort_unstable_by(|&left, &right| {
            minimum_distances[right]
                .total_cmp(&minimum_distances[left])
                .then_with(|| left.cmp(&right))
        });
        let recovered: Vec<usize> = remaining
            .into_iter()
            .take(clusters - candidate_indices.len())
            .collect();
        let candidate_offset = candidate_indices.len();
        update_point_minimum_distances(
            data,
            points,
            dim,
            &recovered,
            candidate_offset,
            &mut minimum_distances,
            &mut nearest_candidates,
        );
        candidate_indices.extend(recovered);
    }
    debug_assert!(candidate_indices.len() <= candidate_budget);

    drop(minimum_distances);
    drop(selected);
    let candidate_count = candidate_indices.len();
    let centroids = reduce_kmeans_parallel_candidates(
        data,
        dim,
        clusters,
        &candidate_indices,
        &nearest_candidates,
        &mut rng,
    );
    (centroids, candidate_count)
}

fn initialize_euclidean_centroids(
    data: &[f32],
    points: usize,
    dim: usize,
    clusters: usize,
    seed: u64,
) -> Vec<f32> {
    match initialization_strategy(clusters) {
        InitializationStrategy::KMeansPlusPlus => {
            initialize_kmeans_plus_plus(data, points, dim, clusters, seed)
        }
        InitializationStrategy::KMeansParallel => {
            initialize_kmeans_parallel(data, points, dim, clusters, seed).0
        }
    }
}

fn assign_nearest_points(
    data: &[f32],
    dim: usize,
    centroids: &[f32],
    nearest_points: &mut [(usize, f32)],
) {
    debug_assert_eq!(data.len(), nearest_points.len().saturating_mul(dim));
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        nearest_points
            .par_iter_mut()
            .zip(data.par_chunks_exact(dim))
            .for_each(|(result, point)| *result = nearest(centroids, point, dim));
    }
    #[cfg(not(feature = "native"))]
    for (result, point) in nearest_points.iter_mut().zip(data.chunks_exact(dim)) {
        *result = nearest(centroids, point, dim);
    }
}

fn deterministic_objective(nearest_points: &[(usize, f32)]) -> f64 {
    #[cfg(feature = "native")]
    let block_totals: Vec<f64> = {
        use rayon::prelude::*;
        nearest_points
            .par_chunks(WEIGHT_REDUCTION_BLOCK)
            .map(|block| block.iter().map(|&(_, distance)| distance as f64).sum())
            .collect()
    };
    #[cfg(not(feature = "native"))]
    let block_totals: Vec<f64> = nearest_points
        .chunks(WEIGHT_REDUCTION_BLOCK)
        .map(|block| block.iter().map(|&(_, distance)| distance as f64).sum())
        .collect();
    block_totals.into_iter().sum()
}

#[inline]
fn objective_has_converged(previous: f64, current: f64) -> bool {
    previous.is_finite()
        && current.is_finite()
        && current <= previous
        && previous - current <= RELATIVE_OBJECTIVE_TOLERANCE * previous.max(f64::MIN_POSITIVE)
}

fn rebuild_flat_memberships(
    assignments: &[usize],
    counts: &mut [usize],
    offsets: &mut [usize],
    members: &mut [usize],
) {
    debug_assert_eq!(offsets.len(), counts.len() + 1);
    debug_assert_eq!(members.len(), assignments.len());
    counts.fill(0);
    for &cluster in assignments {
        counts[cluster] += 1;
    }
    offsets[0] = 0;
    for cluster in 0..counts.len() {
        offsets[cluster + 1] = offsets[cluster] + counts[cluster];
    }
    counts.fill(0);
    for (point, &cluster) in assignments.iter().enumerate() {
        members[offsets[cluster] + counts[cluster]] = point;
        counts[cluster] += 1;
    }
}

pub(crate) fn train_euclidean_kmeans(
    data: &[f32],
    points: usize,
    dim: usize,
    clusters: usize,
    max_iters: usize,
    seed: u64,
) -> EuclideanKMeans {
    assert!(points > 0 && dim > 0 && clusters > 0 && clusters <= points);
    assert_eq!(data.len(), points.saturating_mul(dim));
    assert!(data.iter().all(|value| value.is_finite()));

    let mut centroids = initialize_euclidean_centroids(data, points, dim, clusters, seed);
    let mut assignments = vec![usize::MAX; points];
    let mut nearest_points = vec![(0usize, f32::INFINITY); points];
    let mut counts = vec![0usize; clusters];
    let mut offsets = vec![0usize; clusters + 1];
    let mut members = vec![0usize; points];
    let mut next_centroids = vec![0.0f32; clusters * dim];
    let mut previous_objective = None;

    for _ in 0..max_iters.max(1) {
        assign_nearest_points(data, dim, &centroids, &mut nearest_points);

        let changed = assignments
            .iter()
            .zip(&nearest_points)
            .filter(|(current, (next, _))| **current != *next)
            .count();
        if changed == 0 {
            break;
        }
        let objective = deterministic_objective(&nearest_points);
        for (assignment, &(next, _)) in assignments.iter_mut().zip(&nearest_points) {
            *assignment = next;
        }

        rebuild_flat_memberships(&assignments, &mut counts, &mut offsets, &mut members);
        let has_empty_cluster = counts.contains(&0);
        if !has_empty_cluster
            && previous_objective
                .is_some_and(|previous| objective_has_converged(previous, objective))
        {
            break;
        }
        previous_objective = Some(objective);

        next_centroids.fill(0.0);
        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            next_centroids
                .par_chunks_mut(dim)
                .enumerate()
                .filter(|(cluster, _)| counts[*cluster] != 0)
                .for_each(|(cluster, centroid)| {
                    update_centroid(
                        centroid,
                        &members[offsets[cluster]..offsets[cluster + 1]],
                        data,
                        dim,
                    );
                });
        }
        #[cfg(not(feature = "native"))]
        for (cluster, centroid) in next_centroids.chunks_mut(dim).enumerate() {
            if counts[cluster] == 0 {
                continue;
            }
            update_centroid(
                centroid,
                &members[offsets[cluster]..offsets[cluster + 1]],
                data,
                dim,
            );
        }

        // Empty cells are re-seeded from the currently worst represented
        // points, a standard Lloyd recovery that avoids zero-vector cells.
        if has_empty_cluster {
            let mut farthest: Vec<usize> = (0..points).collect();
            farthest.sort_unstable_by(|&left, &right| {
                nearest_points[right]
                    .1
                    .total_cmp(&nearest_points[left].1)
                    .then_with(|| left.cmp(&right))
            });
            let mut replacement = 0usize;
            for (cluster, &count) in counts.iter().enumerate() {
                if count == 0 {
                    let point = farthest[replacement % farthest.len()];
                    replacement += 1;
                    next_centroids[cluster * dim..(cluster + 1) * dim]
                        .copy_from_slice(&data[point * dim..(point + 1) * dim]);
                }
            }
        }
        std::mem::swap(&mut centroids, &mut next_centroids);
    }

    // Ensure assignments correspond to the returned centroids when the
    // iteration budget, rather than convergence, stopped training.
    assign_nearest_points(data, dim, &centroids, &mut nearest_points);
    for (assignment, &(cluster, _)) in assignments.iter_mut().zip(&nearest_points) {
        *assignment = cluster;
    }
    rebuild_flat_memberships(&assignments, &mut counts, &mut offsets, &mut members);

    EuclideanKMeans {
        centroids,
        #[cfg(test)]
        assignments,
        member_offsets: offsets,
        members,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_memberships_match(trained: &EuclideanKMeans, clusters: usize) {
        assert_eq!(trained.member_offsets.len(), clusters + 1);
        assert_eq!(trained.member_offsets.first(), Some(&0));
        assert_eq!(
            trained.member_offsets.last(),
            Some(&trained.assignments.len())
        );
        let mut seen = vec![false; trained.assignments.len()];
        for cluster in 0..clusters {
            for &point in &trained.members
                [trained.member_offsets[cluster]..trained.member_offsets[cluster + 1]]
            {
                assert_eq!(trained.assignments[point], cluster);
                assert!(!std::mem::replace(&mut seen[point], true));
            }
        }
        assert!(seen.into_iter().all(|point| point));
    }

    #[test]
    fn weighted_sampling_selects_across_fixed_reduction_blocks() {
        let mut weights = vec![0.0; WEIGHT_REDUCTION_BLOCK * 2 + 3];
        weights[7] = 1.0;
        weights[WEIGHT_REDUCTION_BLOCK + 11] = 2.0;
        weights[WEIGHT_REDUCTION_BLOCK * 2 + 2] = 1.0;
        assert_eq!(weighted_sample_index(&weights, 0.10), Some(7));
        assert_eq!(weighted_sample_index(&weights, 0.0), Some(7));
        assert_eq!(
            weighted_sample_index(&weights, 0.50),
            Some(WEIGHT_REDUCTION_BLOCK + 11)
        );
        assert_eq!(
            weighted_sample_index(&weights, 0.99),
            Some(WEIGHT_REDUCTION_BLOCK * 2 + 2)
        );
        assert_eq!(weighted_sample_index(&[0.0, 0.0], 0.5), None);
    }

    #[test]
    fn deterministic_kmeans_plus_plus_finds_separated_groups() {
        let data = [0.0, 0.1, -0.1, 10.0, 9.9, 10.1];
        let first = train_euclidean_kmeans(&data, 6, 1, 2, 20, 42);
        let second = train_euclidean_kmeans(&data, 6, 1, 2, 20, 42);
        assert_eq!(first.centroids, second.centroids);
        assert_eq!(first.assignments, second.assignments);
        assert_memberships_match(&first, 2);
        for (point, &assignment) in data.iter().zip(&first.assignments) {
            assert_eq!(
                assignment,
                nearest(&first.centroids, std::slice::from_ref(point), 1).0
            );
        }
        let mut centroids = first.centroids;
        centroids.sort_unstable_by(f32::total_cmp);
        assert!((centroids[0] - 0.0).abs() < 0.01);
        assert!((centroids[1] - 10.0).abs() < 0.01);
    }

    #[test]
    fn initialization_strategy_and_distance_work_are_bounded() {
        assert_eq!(
            initialization_strategy(KMEANS_PLUS_PLUS_MAX_CLUSTERS),
            InitializationStrategy::KMeansPlusPlus
        );
        assert_eq!(
            initialization_strategy(KMEANS_PLUS_PLUS_MAX_CLUSTERS + 1),
            InitializationStrategy::KMeansParallel
        );
        assert_eq!(
            kmeans_parallel_candidate_budget(10_000, 300),
            300 * KMEANS_PARALLEL_TOTAL_OVERSAMPLING
        );
        assert_eq!(kmeans_parallel_candidate_budget(350, 300), 350);
        assert_eq!(
            estimated_euclidean_kmeans_distance_multiplier(
                10_000,
                KMEANS_PLUS_PLUS_MAX_CLUSTERS,
                25
            ),
            27
        );
        assert_eq!(
            estimated_euclidean_kmeans_distance_multiplier(
                10_000,
                KMEANS_PLUS_PLUS_MAX_CLUSTERS + 1,
                25
            ),
            29
        );
    }

    #[test]
    fn large_k_parallel_candidates_respect_total_budget_and_preserve_quality() {
        let clusters = KMEANS_PLUS_PLUS_MAX_CLUSTERS + 1;
        let mut data = Vec::with_capacity(clusters * 2);
        for cluster in 0..clusters {
            let center = cluster as f32 * 10.0;
            data.extend([center - 0.01, center + 0.01]);
        }
        let points = data.len();
        let (first, first_candidates) = initialize_kmeans_parallel(&data, points, 1, clusters, 42);
        let (second, second_candidates) =
            initialize_kmeans_parallel(&data, points, 1, clusters, 42);
        assert_eq!(first, second);
        assert_eq!(first_candidates, second_candidates);
        assert!(first_candidates >= clusters);
        assert!(first_candidates <= kmeans_parallel_candidate_budget(points, clusters));

        let trained = train_euclidean_kmeans(&data, points, 1, clusters, 10, 42);
        let objective: f32 = data
            .iter()
            .enumerate()
            .map(|(point, &value)| {
                let centroid = trained.centroids[trained.assignments[point]];
                (value - centroid) * (value - centroid)
            })
            .sum();
        assert!(objective < 0.1, "unexpected large-K objective {objective}");
        assert_memberships_match(&trained, clusters);
    }

    #[test]
    fn flat_memberships_are_cluster_major_and_stable() {
        let assignments = [2, 0, 2, 1, 0, 2];
        let mut counts = vec![0; 3];
        let mut offsets = vec![0; 4];
        let mut members = vec![usize::MAX; assignments.len()];
        rebuild_flat_memberships(&assignments, &mut counts, &mut offsets, &mut members);

        assert_eq!(counts, [2, 1, 3]);
        assert_eq!(offsets, [0, 2, 3, 6]);
        assert_eq!(members, [1, 4, 3, 0, 2, 5]);
    }

    #[test]
    fn farthest_recovery_keeps_identical_empty_centroids_representable() {
        let data = [7.0; 8];
        let trained = train_euclidean_kmeans(&data, data.len(), 1, 4, 1, 7);
        assert_eq!(trained.centroids, [7.0; 4]);
        assert!(trained.assignments.iter().all(|&cluster| cluster == 0));
        assert_memberships_match(&trained, 4);
    }

    #[test]
    fn objective_stopping_is_relative_and_never_accepts_regression() {
        assert!(objective_has_converged(100.0, 99.9995));
        assert!(!objective_has_converged(100.0, 99.9));
        assert!(!objective_has_converged(100.0, 100.0001));
        assert!(!objective_has_converged(f64::INFINITY, 1.0));
    }

    #[cfg(feature = "native")]
    #[test]
    fn training_is_deterministic_across_thread_counts() {
        let points = 1_024;
        let dim = 4;
        let clusters = KMEANS_PLUS_PLUS_MAX_CLUSTERS + 1;
        let data: Vec<f32> = (0..points * dim)
            .map(|index| {
                let mixed = (index as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left(17);
                (mixed as u32) as f32 / u32::MAX as f32
            })
            .collect();

        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| train_euclidean_kmeans(&data, points, dim, clusters, 3, 42));
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| train_euclidean_kmeans(&data, points, dim, clusters, 3, 42));

        assert_eq!(one_thread.centroids, four_threads.centroids);
        assert_eq!(one_thread.assignments, four_threads.assignments);
        assert_eq!(one_thread.member_offsets, four_threads.member_offsets);
        assert_eq!(one_thread.members, four_threads.members);
    }
}
