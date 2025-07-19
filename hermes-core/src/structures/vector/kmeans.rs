//! Deterministic Euclidean k-means shared by coarse and product quantizers.
//!
//! k-means++ seeding avoids the unstable random-first-k initialization that
//! previously existed in the native path. Lloyd assignment is parallel on
//! native builds, while centroid reduction stays deterministic.

use rand::prelude::*;

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
        let mut total = 0.0f64;
        for (index, minimum) in minimum_distances.iter_mut().enumerate() {
            let point = &data[index * dim..(index + 1) * dim];
            *minimum = minimum.min(squared_l2(point, latest) as f64);
            total += *minimum;
        }
        let selected = if total.is_finite() && total > 0.0 {
            let mut target = rng.random::<f64>() * total;
            minimum_distances
                .iter()
                .position(|weight| {
                    target -= *weight;
                    target <= 0.0
                })
                .unwrap_or(points - 1)
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

        let mut next_centroids = vec![0.0f32; clusters * dim];
        let mut counts = vec![0usize; clusters];
        for (point_index, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            let source = &data[point_index * dim..(point_index + 1) * dim];
            let target = &mut next_centroids[cluster * dim..(cluster + 1) * dim];
            for (sum, &value) in target.iter_mut().zip(source) {
                *sum += value;
            }
        }
        for (cluster, &count) in counts.iter().enumerate() {
            if count > 0 {
                let inverse = 1.0 / count as f32;
                for value in &mut next_centroids[cluster * dim..(cluster + 1) * dim] {
                    *value *= inverse;
                }
            }
        }

        // Empty cells are re-seeded from the currently worst represented
        // points, a standard Lloyd recovery that avoids zero-vector cells.
        if counts.contains(&0) {
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
}
