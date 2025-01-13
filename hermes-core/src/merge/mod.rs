//! Merge policies for background segment merging
//!
//! Merge policies determine when and which segments should be merged together.
//! The default is a tiered/log-layered policy that groups segments by size tiers.

use std::fmt::Debug;

#[cfg(feature = "native")]
mod scheduler;
#[cfg(feature = "native")]
pub use scheduler::SegmentManager;

/// Information about a segment for merge decisions
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment ID (hex string)
    pub id: String,
    /// Number of documents in the segment
    pub num_docs: u32,
    /// Approximate size in bytes (if known)
    pub size_bytes: Option<u64>,
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
    /// Given the current segments, return merge candidates (if any)
    ///
    /// Each `MergeCandidate` represents a group of segments that should be merged.
    /// Multiple candidates can be returned for parallel merging.
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
            max_merge_at_once: 10,
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

    /// Set segments per tier
    pub fn with_segments_per_tier(mut self, n: usize) -> Self {
        self.segments_per_tier = n;
        self
    }

    /// Set max merge at once
    pub fn with_max_merge_at_once(mut self, n: usize) -> Self {
        self.max_merge_at_once = n;
        self
    }

    /// Set tier factor
    pub fn with_tier_factor(mut self, factor: f64) -> Self {
        self.tier_factor = factor;
        self
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

        let mut candidates = Vec::new();

        // Find tiers with enough segments to merge
        for (_tier, tier_segments) in tiers {
            if tier_segments.len() >= self.segments_per_tier {
                // Sort by doc count (merge smaller ones first)
                let mut sorted: Vec<_> = tier_segments;
                sorted.sort_by_key(|s| s.num_docs);

                // Take up to max_merge_at_once segments
                let to_merge: Vec<_> = sorted.into_iter().take(self.max_merge_at_once).collect();

                // Check total docs limit
                let total_docs: u32 = to_merge.iter().map(|s| s.num_docs).sum();
                if total_docs <= self.max_merged_docs && to_merge.len() >= 2 {
                    candidates.push(MergeCandidate {
                        segment_ids: to_merge.into_iter().map(|s| s.id.clone()).collect(),
                    });
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
                size_bytes: None,
            },
            SegmentInfo {
                id: "b".into(),
                num_docs: 200,
                size_bytes: None,
            },
        ];

        let merges = policy.find_merges(&segments);
        assert!(merges.is_empty());
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
                size_bytes: None,
            })
            .collect();

        let merges = policy.find_merges(&segments);
        assert_eq!(merges.len(), 1);
        assert!(merges[0].segment_ids.len() >= 3);
    }

    #[test]
    fn test_no_merge_policy() {
        let policy = NoMergePolicy;

        let segments = vec![
            SegmentInfo {
                id: "a".into(),
                num_docs: 100,
                size_bytes: None,
            },
            SegmentInfo {
                id: "b".into(),
                num_docs: 200,
                size_bytes: None,
            },
        ];

        let merges = policy.find_merges(&segments);
        assert!(merges.is_empty());
    }
}
