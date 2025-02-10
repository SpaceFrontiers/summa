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

    /// Compute the tier for a segment based on its doc count
    fn compute_tier(&self, num_docs: u32) -> usize {
        if num_docs <= self.tier_floor {
            return 0;
        }

        let ratio = num_docs as f64 / self.tier_floor as f64;
        (ratio.log(self.tier_factor).floor() as usize) + 1
    }
}

impl MergePolicy for TieredMergePolicy {
    fn find_merges(&self, segments: &[SegmentInfo]) -> Vec<MergeCandidate> {
        if segments.len() < 2 {
            return Vec::new();
        }

        // Group segments by tier
        let mut tiers: std::collections::HashMap<usize, Vec<&SegmentInfo>> =
            std::collections::HashMap::new();

        for seg in segments {
            let tier = self.compute_tier(seg.num_docs);
            tiers.entry(tier).or_default().push(seg);
        }

        // Produce one candidate per qualifying tier (all can run concurrently)
        let mut candidates = Vec::new();
        for tier_segments in tiers.values() {
            if tier_segments.len() >= self.segments_per_tier {
                let mut sorted: Vec<_> = tier_segments.clone();
                sorted.sort_by_key(|s| s.num_docs);

                let chunk = &sorted[..sorted.len().min(self.max_merge_at_once)];
                if chunk.len() >= 2 {
                    let total_docs: u64 = chunk.iter().map(|s| s.num_docs as u64).sum();
                    if total_docs <= self.max_merged_docs as u64 {
                        candidates.push(MergeCandidate {
                            segment_ids: chunk.iter().map(|s| s.id.clone()).collect(),
                        });
                    }
                }
            }
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

    #[test]
    fn test_tiered_policy_compute_tier() {
        let policy = TieredMergePolicy::default();

        // Tier 0: <= 1000 docs (tier_floor)
        assert_eq!(policy.compute_tier(500), 0);
        assert_eq!(policy.compute_tier(1000), 0);

        // Tier 1: 1001 - 9999 docs (ratio < 10)
        assert_eq!(policy.compute_tier(1001), 1);
        assert_eq!(policy.compute_tier(5000), 1);
        assert_eq!(policy.compute_tier(9999), 1);

        // Tier 2: 10000 - 99999 docs (ratio 10-100)
        assert_eq!(policy.compute_tier(10000), 2);
        assert_eq!(policy.compute_tier(50000), 2);

        // Tier 3: 100000+ docs
        assert_eq!(policy.compute_tier(100000), 3);
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
    fn test_tiered_policy_merge_same_tier() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            ..Default::default()
        };

        // All in tier 0
        let segments: Vec<_> = (0..5)
            .map(|i| SegmentInfo {
                id: format!("seg_{}", i),
                num_docs: 100 + i * 10,
            })
            .collect();

        let candidates = policy.find_merges(&segments);
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].segment_ids.len() >= 3);
    }

    #[test]
    fn test_tiered_policy_multiple_tiers() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            ..Default::default()
        };

        // 4 segments in tier 0 + 3 segments in tier 1
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
            2,
            "should produce candidates for both tiers"
        );
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
