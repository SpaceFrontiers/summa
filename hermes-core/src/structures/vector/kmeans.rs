//! Deterministic Euclidean k-means shared by coarse and product quantizers.
//!
//! k-means++ seeding avoids the unstable random-first-k initialization that
//! previously existed in the native path. Lloyd assignment is parallel on
//! native builds, while centroid reduction stays deterministic.

use rand::prelude::*;

const WEIGHT_REDUCTION_BLOCK: usize = 16 * 1024;

pub(crate) struct EuclideanKMeans {
    pub centroids: Vec<f32>,
    pub assignments: Vec<usize>,
}

#[inline]
fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
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

/// Draw from non-negative weights with a fixed reduction topology. Parallel
/// block sums remove the serial O(N) bottleneck from every k-means++ seed while
/// producing the same choice for every rayon thread count.
pub(crate) fn weighted_sample_index(weights: &[f64], draw: f64) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }
    #[cfg(feature = "native")]
    let block_totals: Vec<f64> = {
        use rayon::prelude::*;
        weights
            .par_chunks(WEIGHT_REDUCTION_BLOCK)
            .map(|block| block.iter().sum())
            .collect()
    };
    #[cfg(not(feature = "native"))]
    let block_totals: Vec<f64> = weights
        .chunks(WEIGHT_REDUCTION_BLOCK)
        .map(|block| block.iter().sum())
        .collect();

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

        let selected = if let Some(selected) =
            weighted_sample_index(&minimum_distances, rng.random::<f64>())
        {
            selected
        } else {
            // All points are identical (or all remaining distances underflow).
            // A deterministic duplicate is the only representable centroid.
            (centroids.len() / dim) % points
        };
        centroids.extend_from_slice(&data[selected * dim..(selected + 1) * dim]);
    }
    centroids
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

    let mut centroids = initialize_kmeans_plus_plus(data, points, dim, clusters, seed);
    let mut assignments = vec![usize::MAX; points];
    let mut members: Vec<Vec<usize>> = (0..clusters).map(|_| Vec::new()).collect();

    for _ in 0..max_iters.max(1) {
        #[cfg(feature = "native")]
        let nearest_points: Vec<(usize, f32)> = {
            use rayon::prelude::*;
            data.par_chunks_exact(dim)
                .map(|point| nearest(&centroids, point, dim))
                .collect()
        };
        #[cfg(not(feature = "native"))]
        let nearest_points: Vec<(usize, f32)> = data
            .chunks_exact(dim)
            .map(|point| nearest(&centroids, point, dim))
            .collect();

        let changed = assignments
            .iter()
            .zip(&nearest_points)
            .filter(|(current, (next, _))| **current != *next)
            .count();
        if changed == 0 {
            break;
        }
        for (assignment, &(next, _)) in assignments.iter_mut().zip(&nearest_points) {
            *assignment = next;
        }

        for cluster_members in &mut members {
            cluster_members.clear();
        }
        for (point_index, &cluster) in assignments.iter().enumerate() {
            members[cluster].push(point_index);
        }

        let mut next_centroids = vec![0.0f32; clusters * dim];
        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            next_centroids
                .par_chunks_mut(dim)
                .zip(members.par_iter())
                .filter(|(_, cluster_members)| !cluster_members.is_empty())
                .for_each(|(centroid, cluster_members)| {
                    update_centroid(centroid, cluster_members, data, dim);
                });
        }
        #[cfg(not(feature = "native"))]
        for (centroid, cluster_members) in next_centroids.chunks_mut(dim).zip(&members) {
            if cluster_members.is_empty() {
                continue;
            }
            update_centroid(centroid, cluster_members, data, dim);
        }

        // Empty cells are re-seeded from the currently worst represented
        // points, a standard Lloyd recovery that avoids zero-vector cells.
        if members.iter().any(Vec::is_empty) {
            let mut farthest: Vec<usize> = (0..points).collect();
            farthest.sort_unstable_by(|&left, &right| {
                nearest_points[right]
                    .1
                    .total_cmp(&nearest_points[left].1)
                    .then_with(|| left.cmp(&right))
            });
            let mut replacement = 0usize;
            for (cluster, cluster_members) in members.iter().enumerate() {
                if cluster_members.is_empty() {
                    let point = farthest[replacement % farthest.len()];
                    replacement += 1;
                    next_centroids[cluster * dim..(cluster + 1) * dim]
                        .copy_from_slice(&data[point * dim..(point + 1) * dim]);
                }
            }
        }
        centroids = next_centroids;
    }

    // Ensure assignments correspond to the returned centroids when the
    // iteration budget, rather than convergence, stopped training.
    #[cfg(feature = "native")]
    {
        use rayon::prelude::*;
        assignments = data
            .par_chunks_exact(dim)
            .map(|point| nearest(&centroids, point, dim).0)
            .collect();
    }
    #[cfg(not(feature = "native"))]
    {
        assignments = data
            .chunks_exact(dim)
            .map(|point| nearest(&centroids, point, dim).0)
            .collect();
    }

    EuclideanKMeans {
        centroids,
        assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut centroids = first.centroids;
        centroids.sort_unstable_by(f32::total_cmp);
        assert!((centroids[0] - 0.0).abs() < 0.01);
        assert!((centroids[1] - 10.0).abs() < 0.01);
    }

    #[cfg(feature = "native")]
    #[test]
    fn training_is_deterministic_across_thread_counts() {
        let points = 2_048;
        let dim = 16;
        let clusters = 32;
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
            .install(|| train_euclidean_kmeans(&data, points, dim, clusters, 4, 42));
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| train_euclidean_kmeans(&data, points, dim, clusters, 4, 42));

        assert_eq!(one_thread.centroids, four_threads.centroids);
        assert_eq!(one_thread.assignments, four_threads.assignments);
    }
}
