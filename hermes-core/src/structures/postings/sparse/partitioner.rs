//! Optimal variable-size block partitioning for sparse posting lists.
//!
//! Implements the VBMW approach (Mallia et al., SIGIR 2017): partition a posting
//! list into variable-sized blocks to minimize "block error" — the sum of
//! (block_max - actual_weight) across all postings. Tighter block-max upper
//! bounds improve pruning in both BmpExecutor and SparseMaxScoreExecutor.
//!
//! Algorithm: 1-D DP over posting positions with allowed block sizes
//! [16, 32, 64, 128, 256]. Precomputed prefix sums and sparse-table RMQ
//! give O(1) cost evaluation per candidate split. Total: O(N × 5).

/// Allowed block sizes (SIMD-aligned powers of 2).
/// Smallest = 16 (matches minimum in from_postings_with_block_size).
/// Largest = 256 (MAX_BLOCK_SIZE).
const ALLOWED_SIZES: [usize; 5] = [16, 32, 64, 128, 256];

/// Compute an optimal variable-size block partition for a posting list.
///
/// Returns a `Vec<usize>` of block sizes whose sum equals `weights.len()`.
/// Each block size is one of [16, 32, 64, 128, 256].
///
/// For short lists (≤ 16), returns a single block covering everything.
/// For lists whose length isn't exactly coverable by allowed sizes,
/// a greedy tail-fill ensures full coverage.
///
/// `weights` must contain the **absolute** weights of postings, ordered by doc_id.
pub fn optimal_partition(weights: &[f32]) -> Vec<usize> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    // Short lists: single block
    if n <= ALLOWED_SIZES[0] {
        return vec![n];
    }

    // Prefix sums for O(1) range-sum queries
    let mut prefix_sum = vec![0.0f32; n + 1];
    for i in 0..n {
        prefix_sum[i + 1] = prefix_sum[i] + weights[i];
    }

    // Sparse table for O(1) range-max queries
    let rmq = SparseTableMax::new(weights);

    // DP: dp[i] = min total block error to cover postings 0..i
    // parent[i] = the block size used to reach position i
    let mut dp = vec![f64::MAX; n + 1];
    let mut parent = vec![0usize; n + 1];
    dp[0] = 0.0;

    for i in 1..=n {
        for &s in &ALLOWED_SIZES {
            if s > i {
                continue;
            }
            let start = i - s;
            if dp[start] == f64::MAX {
                continue;
            }
            // Block error = s * max(weights[start..i]) - sum(weights[start..i])
            let block_max = rmq.query(start, i - 1);
            let block_sum = (prefix_sum[i] - prefix_sum[start]) as f64;
            let cost = (s as f64) * (block_max as f64) - block_sum;
            let total = dp[start] + cost;
            if total < dp[i] {
                dp[i] = total;
                parent[i] = s;
            }
        }
    }

    // If dp[n] is reachable, backtrack to recover partition
    if dp[n] < f64::MAX {
        let mut partition = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let s = parent[pos];
            partition.push(s);
            pos -= s;
        }
        partition.reverse();
        return partition;
    }

    // Fallback: dp[n] unreachable (n not exactly coverable by allowed sizes).
    // Find the largest reachable position and greedily fill the tail.
    let mut best_pos = 0;
    for i in (1..n).rev() {
        if dp[i] < f64::MAX {
            best_pos = i;
            break;
        }
    }

    // Backtrack the reachable prefix
    let mut partition = Vec::new();
    let mut pos = best_pos;
    while pos > 0 {
        let s = parent[pos];
        partition.push(s);
        pos -= s;
    }
    partition.reverse();

    // Greedily cover the remaining tail
    let mut remaining = n - best_pos;
    while remaining > 0 {
        // Pick the largest allowed size that fits, or the smallest if none fit exactly
        let mut picked = remaining.min(ALLOWED_SIZES[0]);
        for &s in ALLOWED_SIZES.iter().rev() {
            if s <= remaining {
                picked = s;
                break;
            }
        }
        partition.push(picked);
        remaining -= picked;
    }

    partition
}

/// Compute the total block error for a given partition.
///
/// Error = Σ_block (block_size × max_weight_in_block - sum_of_weights_in_block)
#[cfg(test)]
fn partition_error(weights: &[f32], partition: &[usize]) -> f64 {
    let mut error = 0.0f64;
    let mut pos = 0;
    for &s in partition {
        let block = &weights[pos..pos + s];
        let block_max = block.iter().copied().fold(0.0f32, f32::max);
        let block_sum: f32 = block.iter().sum();
        error += (s as f64) * (block_max as f64) - (block_sum as f64);
        pos += s;
    }
    error
}

/// Sparse table for O(1) range-maximum queries over f32 values.
struct SparseTableMax {
    table: Vec<Vec<f32>>,
    log2: Vec<usize>,
}

impl SparseTableMax {
    fn new(data: &[f32]) -> Self {
        let n = data.len();
        if n == 0 {
            return Self {
                table: Vec::new(),
                log2: vec![0],
            };
        }

        // Precompute floor(log2) for each length
        let mut log2 = vec![0usize; n + 1];
        for i in 2..=n {
            log2[i] = log2[i / 2] + 1;
        }
        let max_log = log2[n] + 1;

        let mut table = vec![vec![0.0f32; n]; max_log];
        // Level 0: individual elements
        table[0][..n].copy_from_slice(&data[..n]);
        // Build higher levels
        for k in 1..max_log {
            let half = 1 << (k - 1);
            for i in 0..n {
                if i + half < n {
                    table[k][i] = f32::max(table[k - 1][i], table[k - 1][i + half]);
                } else {
                    table[k][i] = table[k - 1][i];
                }
            }
        }

        Self { table, log2 }
    }

    /// Query max over data[l..=r] (inclusive)
    #[inline]
    fn query(&self, l: usize, r: usize) -> f32 {
        if l > r {
            return 0.0;
        }
        let len = r - l + 1;
        let k = self.log2[len];
        let half = 1 << k;
        f32::max(self.table[k][l], self.table[k][r + 1 - half])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let partition = optimal_partition(&[]);
        assert!(partition.is_empty());
    }

    #[test]
    fn test_trivial_small() {
        // N ≤ 16: should return a single block
        let weights = vec![1.0; 10];
        let partition = optimal_partition(&weights);
        assert_eq!(partition.len(), 1);
        assert_eq!(partition[0], 10);
        assert_eq!(partition.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_exact_block_size() {
        // N = 128: exactly one block of 128
        let weights = vec![1.0; 128];
        let partition = optimal_partition(&weights);
        let total: usize = partition.iter().sum();
        assert_eq!(total, 128);
        // All blocks should be allowed sizes
        for &s in &partition {
            assert!(ALLOWED_SIZES.contains(&s), "block size {} not allowed", s);
        }
    }

    #[test]
    fn test_uniform_weights() {
        // Uniform weights: all blocks have zero error regardless of size.
        // DP should still produce a valid covering.
        let weights = vec![5.0; 256];
        let partition = optimal_partition(&weights);
        let total: usize = partition.iter().sum();
        assert_eq!(total, 256);
        for &s in &partition {
            assert!(ALLOWED_SIZES.contains(&s), "block size {} not allowed", s);
        }
        // Error should be ~0 for uniform weights
        let error = partition_error(&weights, &partition);
        assert!(error.abs() < 1e-6, "error should be ~0, got {}", error);
    }

    #[test]
    fn test_outlier_isolation() {
        // 128 postings: one outlier (weight=100) at position 16, rest are weight=1.
        // Optimal partition should isolate the outlier in a small block.
        let mut weights = vec![1.0f32; 128];
        weights[16] = 100.0;

        let adaptive_partition = optimal_partition(&weights);
        let adaptive_total: usize = adaptive_partition.iter().sum();
        assert_eq!(adaptive_total, 128);

        let adaptive_error = partition_error(&weights, &adaptive_partition);
        let fixed_error = partition_error(&weights, &[128]);

        // The outlier inflates the fixed block's error enormously.
        // Adaptive should be significantly better.
        assert!(
            adaptive_error < fixed_error,
            "adaptive error ({}) should be < fixed error ({})",
            adaptive_error,
            fixed_error
        );

        // The outlier at position 16 should end up in a block of size 16 or 32 max.
        // Find which block contains position 16.
        let mut pos = 0;
        let mut outlier_block_size = 0;
        for &s in &adaptive_partition {
            if pos <= 16 && 16 < pos + s {
                outlier_block_size = s;
                break;
            }
            pos += s;
        }
        assert!(
            outlier_block_size <= 32,
            "outlier should be in a small block, got size {}",
            outlier_block_size
        );
    }

    #[test]
    fn test_error_reduction_skewed() {
        // Skewed distribution: exponentially decaying weights
        let n = 512;
        let weights: Vec<f32> = (0..n).map(|i| 100.0 * (-0.01 * i as f32).exp()).collect();

        let adaptive_partition = optimal_partition(&weights);
        let adaptive_total: usize = adaptive_partition.iter().sum();
        assert_eq!(adaptive_total, n);

        let adaptive_error = partition_error(&weights, &adaptive_partition);

        // Fixed 128-block partition
        let fixed_partition: Vec<usize> = std::iter::repeat_n(128, n / 128).collect();
        let fixed_error = partition_error(&weights, &fixed_partition);

        assert!(
            adaptive_error < fixed_error,
            "adaptive error ({}) should be < fixed error ({})",
            adaptive_error,
            fixed_error
        );

        // Expect at least 20% reduction
        let reduction = 1.0 - adaptive_error / fixed_error;
        assert!(
            reduction > 0.20,
            "expected >20% error reduction, got {:.1}%",
            reduction * 100.0
        );
    }

    #[test]
    fn test_non_coverable_length() {
        // N = 100: not exactly coverable by [16,32,64,128,256]
        // 64 + 32 + 4 won't work, but 64 + 16 + 16 + 4 won't either.
        // Actually 64 + 32 + ... nope. Let's just verify it works.
        // Possible: 32 + 32 + 32 + ... or 64 + 16 + 16 + ... nah.
        // 16*6 = 96, need 4 more. Tail fill will add a block of 4.
        // Wait: 16 + 16 + 16 + 16 + 16 + 16 + 4 = 100. Tail gets 4.
        let weights = vec![1.0; 100];
        let partition = optimal_partition(&weights);
        let total: usize = partition.iter().sum();
        assert_eq!(total, 100);
        // The last block may be < 16 (tail fill for non-coverable remainder)
    }

    #[test]
    fn test_all_allowed_sizes_valid() {
        // For a coverable length, all blocks should be from ALLOWED_SIZES
        let weights = vec![1.0; 256];
        let partition = optimal_partition(&weights);
        for &s in &partition {
            assert!(ALLOWED_SIZES.contains(&s), "unexpected block size {}", s);
        }
    }

    #[test]
    fn test_large_list() {
        // Stress: 10K postings with random-ish weights
        let n = 10_000;
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                // Pseudo-random pattern
                let x = ((i * 7 + 13) % 97) as f32;
                x * 0.1 + 0.1
            })
            .collect();

        let partition = optimal_partition(&weights);
        let total: usize = partition.iter().sum();
        assert_eq!(total, n);

        // Should complete in negligible time (O(N×5))
        let adaptive_error = partition_error(&weights, &partition);
        assert!(adaptive_error >= 0.0);
    }

    #[test]
    fn test_sparse_table_rmq() {
        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let rmq = SparseTableMax::new(&data);

        assert_eq!(rmq.query(0, 0), 3.0);
        assert_eq!(rmq.query(0, 7), 9.0);
        assert_eq!(rmq.query(4, 5), 9.0);
        assert_eq!(rmq.query(2, 4), 5.0);
        assert_eq!(rmq.query(6, 7), 6.0);
        assert_eq!(rmq.query(3, 3), 1.0);
    }
}
