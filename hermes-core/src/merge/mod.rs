//! Merge policies for background segment merging
//!
//! Merge policies determine when and which segments should be merged together.
//! The default is a tiered/log-layered policy that groups segments by size tiers.

use std::fmt::Debug;

#[cfg(feature = "native")]
mod segment_manager;
#[cfg(feature = "native")]
pub use segment_manager::SegmentManager;

/// Information about a segment for merge decisions
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment ID (hex string)
    pub id: String,
    /// Number of documents in the segment
    pub num_docs: u32,
}

/// A merge operation specifying which segments to merge
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Segment IDs to merge together
    pub segment_ids: Vec<String>,
}

/// Trait for merge policies
///
/// Implementations decide when segments should be merged and which ones.
pub trait MergePolicy: Send + Sync + Debug {
    /// Given the current segments, return all eligible merge candidates.
    /// Multiple candidates can run concurrently as long as they don't share segments.
    fn find_merges(&self, segments: &[SegmentInfo]) -> Vec<MergeCandidate>;

    /// Clone the policy into a boxed trait object
    fn clone_box(&self) -> Box<dyn MergePolicy>;
}

impl Clone for Box<dyn MergePolicy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// No-op merge policy - never merges automatically
#[derive(Debug, Clone, Default)]
pub struct NoMergePolicy;

impl MergePolicy for NoMergePolicy {
    fn find_merges(&self, _segments: &[SegmentInfo]) -> Vec<MergeCandidate> {
        Vec::new()
    }

    fn clone_box(&self) -> Box<dyn MergePolicy> {
        Box::new(self.clone())
    }
}

/// Tiered/Log-layered merge policy
///
/// Groups segments into tiers based on document count. Segments in the same tier
/// are merged when there are enough of them. This creates a logarithmic structure
/// where larger segments are merged less frequently.
///
/// Tiers are defined by powers of `tier_factor`:
/// - Tier 0: 0 to tier_floor docs
/// - Tier 1: tier_floor to tier_floor * tier_factor docs
/// - Tier 2: tier_floor * tier_factor to tier_floor * tier_factor^2 docs
/// - etc.
#[derive(Debug, Clone)]
pub struct TieredMergePolicy {
    /// Minimum number of segments in a tier before merging (default: 10)
    pub segments_per_tier: usize,
    /// Maximum number of segments to merge at once (default: 10)
    pub max_merge_at_once: usize,
    /// Factor between tier sizes (default: 10.0)
    pub tier_factor: f64,
    /// Minimum segment size (docs) to consider for tiering (default: 1000)
    pub tier_floor: u32,
    /// Maximum total docs to merge at once (default: 5_000_000)
    pub max_merged_docs: u32,
}

impl Default for TieredMergePolicy {
    fn default() -> Self {
        Self {
            segments_per_tier: 10,
            max_merge_at_once: 100,
            tier_factor: 10.0,
            tier_floor: 1000,
            max_merged_docs: 5_000_000,
        }
    }
}

impl TieredMergePolicy {
    /// Create a new tiered merge policy with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an aggressive merge policy that merges more frequently
    ///
    /// - Merges when 3 segments in same tier (vs 10 default)
    /// - Lower tier floor (500 docs vs 1000)
    /// - Good for reducing segment count quickly
    pub fn aggressive() -> Self {
        Self {
            segments_per_tier: 3,
            max_merge_at_once: 10,
            tier_factor: 10.0,
            tier_floor: 500,
            max_merged_docs: 10_000_000,
        }
    }
}

impl MergePolicy for TieredMergePolicy {
    fn find_merges(&self, segments: &[SegmentInfo]) -> Vec<MergeCandidate> {
        if segments.len() < 2 {
            return Vec::new();
        }

        // Sort by size ascending — greedily merge from smallest.
        // This replaces per-tier grouping, allowing cross-tier promotion:
        // many small segments can jump several tiers in one merge.
        let mut sorted: Vec<&SegmentInfo> = segments.iter().collect();
        sorted.sort_by_key(|s| s.num_docs);

        let mut candidates = Vec::new();
        let mut used = vec![false; sorted.len()];
        let max_ratio = self.tier_factor as u64;

        let mut start = 0;
        loop {
            // Find next unused segment
            while start < sorted.len() && used[start] {
                start += 1;
            }
            if start >= sorted.len() {
                break;
            }

            // Build a merge group starting from the smallest unused segment.
            // Accumulate segments as long as:
            //   - group size < max_merge_at_once
            //   - total docs < max_merged_docs
            //   - the next segment isn't disproportionately larger than the group
            //     (ratio guard prevents rewriting a huge segment to absorb tiny ones)
            let mut group = vec![start];
            let mut total_docs: u64 = sorted[start].num_docs as u64;

            for j in (start + 1)..sorted.len() {
                if used[j] {
                    continue;
                }
                if group.len() >= self.max_merge_at_once {
                    break;
                }
                let next_docs = sorted[j].num_docs as u64;
                if total_docs + next_docs > self.max_merged_docs as u64 {
                    break;
                }
                // Ratio guard: don't include a segment that dwarfs the accumulated group.
                // Use tier_floor as minimum baseline so very small segments can always accumulate.
                let effective_total = total_docs.max(self.tier_floor as u64);
                if next_docs > effective_total * max_ratio {
                    break;
                }
                group.push(j);
                total_docs += next_docs;
            }

            if group.len() >= self.segments_per_tier && group.len() >= 2 {
                for &i in &group {
                    used[i] = true;
                }
                candidates.push(MergeCandidate {
                    segment_ids: group.iter().map(|&i| sorted[i].id.clone()).collect(),
                });
            }

            // Always advance past start (whether or not we formed a group)
            // so we try starting from the next unused segment.
            start += 1;
        }

        candidates
    }

    fn clone_box(&self) -> Box<dyn MergePolicy> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute tier for a segment (used only in tests to verify tier math)
    fn compute_tier(policy: &TieredMergePolicy, num_docs: u32) -> usize {
        if num_docs <= policy.tier_floor {
            return 0;
        }
        let ratio = num_docs as f64 / policy.tier_floor as f64;
        (ratio.log(policy.tier_factor).floor() as usize) + 1
    }

    #[test]
    fn test_tiered_policy_compute_tier() {
        let policy = TieredMergePolicy::default();

        // Tier 0: <= 1000 docs (tier_floor)
        assert_eq!(compute_tier(&policy, 500), 0);
        assert_eq!(compute_tier(&policy, 1000), 0);

        // Tier 1: 1001 - 9999 docs (ratio < 10)
        assert_eq!(compute_tier(&policy, 1001), 1);
        assert_eq!(compute_tier(&policy, 5000), 1);
        assert_eq!(compute_tier(&policy, 9999), 1);

        // Tier 2: 10000 - 99999 docs (ratio 10-100)
        assert_eq!(compute_tier(&policy, 10000), 2);
        assert_eq!(compute_tier(&policy, 50000), 2);

        // Tier 3: 100000+ docs
        assert_eq!(compute_tier(&policy, 100000), 3);
    }

    #[test]
    fn test_tiered_policy_no_merge_few_segments() {
        let policy = TieredMergePolicy::default();

        let segments = vec![
            SegmentInfo {
                id: "a".into(),
                num_docs: 100,
            },
            SegmentInfo {
                id: "b".into(),
                num_docs: 200,
            },
        ];

        assert!(policy.find_merges(&segments).is_empty());
    }

    #[test]
    fn test_tiered_policy_merge_same_size() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            ..Default::default()
        };

        // 5 small segments — all similar size, should merge into one group
        let segments: Vec<_> = (0..5)
            .map(|i| SegmentInfo {
                id: format!("seg_{}", i),
                num_docs: 100 + i * 10,
            })
            .collect();

        let candidates = policy.find_merges(&segments);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].segment_ids.len(), 5);
    }

    #[test]
    fn test_tiered_policy_cross_tier_promotion() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            tier_factor: 10.0,
            tier_floor: 1000,
            max_merge_at_once: 20,
            max_merged_docs: 5_000_000,
        };

        // 4 small (tier 0) + 3 medium (tier 1) — should merge ALL into one group
        // because the small segments accumulate and the medium ones pass the ratio check
        let mut segments: Vec<_> = (0..4)
            .map(|i| SegmentInfo {
                id: format!("small_{}", i),
                num_docs: 100 + i * 10,
            })
            .collect();
        for i in 0..3 {
            segments.push(SegmentInfo {
                id: format!("medium_{}", i),
                num_docs: 2000 + i * 500,
            });
        }

        let candidates = policy.find_merges(&segments);
        assert_eq!(
            candidates.len(),
            1,
            "should merge all into one cross-tier group"
        );
        assert_eq!(
            candidates[0].segment_ids.len(),
            7,
            "all 7 segments should be in the merge"
        );
    }

    #[test]
    fn test_tiered_policy_ratio_guard_separates_groups() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            tier_factor: 10.0,
            tier_floor: 100,
            max_merge_at_once: 20,
            max_merged_docs: 5_000_000,
        };

        // 4 tiny (10 docs) + 4 large (100_000 docs)
        // Ratio guard should prevent merging tiny with large:
        // group total after 4 tiny = 40, effective = max(40, 100) = 100
        // next segment is 100_000 > 100 * 10 = 1000 → blocked
        // So tiny segments (4) form one group, large segments (4) form another.
        let mut segments: Vec<_> = (0..4)
            .map(|i| SegmentInfo {
                id: format!("tiny_{}", i),
                num_docs: 10,
            })
            .collect();
        for i in 0..4 {
            segments.push(SegmentInfo {
                id: format!("large_{}", i),
                num_docs: 100_000 + i * 100,
            });
        }

        let candidates = policy.find_merges(&segments);
        assert_eq!(candidates.len(), 2, "should produce two separate groups");

        // First group: the 4 tiny segments
        assert_eq!(candidates[0].segment_ids.len(), 4);
        assert!(candidates[0].segment_ids[0].starts_with("tiny_"));

        // Second group: the 4 large segments
        assert_eq!(candidates[1].segment_ids.len(), 4);
        assert!(candidates[1].segment_ids[0].starts_with("large_"));
    }

    #[test]
    fn test_tiered_policy_small_segments_skip_to_large_group() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            tier_factor: 10.0,
            tier_floor: 1000,
            max_merge_at_once: 10,
            max_merged_docs: 5_000_000,
        };

        // 2 tiny segments (can't form a group) + 5 medium segments (can)
        // The tiny segments should be skipped, and the medium ones should merge.
        let mut segments = vec![
            SegmentInfo {
                id: "tiny_0".into(),
                num_docs: 10,
            },
            SegmentInfo {
                id: "tiny_1".into(),
                num_docs: 20,
            },
        ];
        for i in 0..5 {
            segments.push(SegmentInfo {
                id: format!("medium_{}", i),
                num_docs: 5000 + i * 100,
            });
        }

        let candidates = policy.find_merges(&segments);
        assert!(
            !candidates.is_empty(),
            "should find a merge even though tiny segments can't form a group"
        );
        // The medium segments should be merged (possibly with the tiny ones bridging in)
        let total_segs: usize = candidates.iter().map(|c| c.segment_ids.len()).sum();
        assert!(
            total_segs >= 5,
            "should merge at least the 5 medium segments"
        );
    }

    #[test]
    fn test_tiered_policy_respects_max_merged_docs() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            max_merge_at_once: 100,
            tier_factor: 10.0,
            tier_floor: 1000,
            max_merged_docs: 500,
        };

        // 10 segments of 100 docs each — total would be 1000 but max_merged_docs=500
        let segments: Vec<_> = (0..10)
            .map(|i| SegmentInfo {
                id: format!("seg_{}", i),
                num_docs: 100,
            })
            .collect();

        let candidates = policy.find_merges(&segments);
        for c in &candidates {
            let total: u64 = c
                .segment_ids
                .iter()
                .map(|id| segments.iter().find(|s| s.id == *id).unwrap().num_docs as u64)
                .sum();
            assert!(
                total <= 500,
                "merge total {} exceeds max_merged_docs 500",
                total
            );
        }
    }

    #[test]
    fn test_no_merge_policy() {
        let policy = NoMergePolicy;

        let segments = vec![
            SegmentInfo {
                id: "a".into(),
                num_docs: 100,
            },
            SegmentInfo {
                id: "b".into(),
                num_docs: 200,
            },
        ];

        assert!(policy.find_merges(&segments).is_empty());
    }
}
