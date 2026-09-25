//! Merge policies for background segment merging
//!
//! Merge policies determine when and which segments should be merged together.
//! The default is a tiered/log-layered policy that groups segments by size tiers.

use std::fmt::Debug;

#[cfg(feature = "native")]
mod segment_manager;
#[cfg(feature = "native")]
pub use segment_manager::SegmentManager;
#[cfg(feature = "native")]
pub(crate) use segment_manager::SegmentOperationGuard;

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

    /// Whether segment topology is far enough over its target that compaction
    /// latency should take precedence over optional merge-time optimization.
    ///
    /// The segment manager uses this signal only when `reorder_on_merge` is
    /// enabled: urgent merges copy encoded text blocks and leave the merged output
    /// for the standalone optimizer. This avoids many long BP passes holding
    /// every merge slot while an ingestion burst keeps publishing segments.
    fn has_severe_backlog(&self, _segments: &[SegmentInfo]) -> bool {
        false
    }

    /// Clone the policy into a boxed trait object
    fn clone_box(&self) -> Box<dyn MergePolicy>;

    /// Maximum number of documents a single segment should contain.
    /// Returns `None` for no limit (force_merge merges everything into one).
    fn max_segment_docs(&self) -> Option<u32> {
        None
    }
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
///
/// For large-scale indexes (10M-1B docs), use [`TieredMergePolicy::large_scale()`] or
/// [`TieredMergePolicy::bulk_indexing()`] presets which enable budget-aware triggering,
/// scored candidate selection, and oversized segment exclusion.
#[derive(Debug, Clone)]
pub struct TieredMergePolicy {
    /// Minimum number of segments in a tier before merging (default: 10)
    pub segments_per_tier: usize,
    /// Maximum number of segments to merge at once (default: 10).
    /// Should be close to segments_per_tier to prevent giant merges.
    pub max_merge_at_once: usize,
    /// Factor between tier sizes (default: 10.0)
    pub tier_factor: f64,
    /// Minimum segment size (docs) to consider for tiering (default: 1000)
    pub tier_floor: u32,
    /// Maximum total docs to merge at once (default: 5_000_000)
    pub max_merged_docs: u32,

    /// Floor size for scoring — tiny segments are treated as this size when
    /// computing merge scores. Prevents degenerate scores from near-empty segments.
    /// (default: 1000)
    pub floor_segment_docs: u32,
    /// Exclude segments larger than `max_merged_docs * oversized_threshold` from
    /// merge candidates. Prevents rewriting already-large segments. (default: 0.5)
    pub oversized_threshold: f64,
    /// Merge output must be >= `(1 + min_growth_ratio) * largest_input` docs.
    /// Rejects merges that rewrite a large segment just to absorb tiny ones.
    /// Set to 0.0 to disable. (default: 0.0)
    pub min_growth_ratio: f64,
    /// Only merge when segment count exceeds the ideal budget
    /// (`num_tiers * segments_per_tier`). Prevents unnecessary merges when
    /// the index is already well-structured. (default: false)
    pub budget_trigger: bool,
    /// Use Lucene-style skew scoring to pick the most balanced merge candidate
    /// instead of greedily taking the first valid group. (default: false)
    pub scored_selection: bool,
    /// Absolute maximum number of documents in a single segment.
    /// Respected by both automatic merging and `force_merge`. (default: 10_000_000)
    pub max_segment_docs: u32,
}

impl Default for TieredMergePolicy {
    fn default() -> Self {
        Self {
            segments_per_tier: 10,
            max_merge_at_once: 10,
            tier_factor: 10.0,
            tier_floor: 1000,
            max_merged_docs: 5_000_000,
            floor_segment_docs: 1000,
            oversized_threshold: 0.5,
            min_growth_ratio: 0.0,
            budget_trigger: false,
            scored_selection: false,
            max_segment_docs: 10_000_000,
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
            max_segment_docs: 10_000_000,
            ..Default::default()
        }
    }

    /// Large-scale merge policy for indexes with 100M-1B documents.
    ///
    /// Enables budget-aware triggering and scored candidate selection to avoid
    /// unnecessary IO on already well-structured indexes. Oversized segments
    /// (>2.5M docs) are excluded from merge candidates.
    ///
    /// Good for live read/write workloads at scale.
    pub fn large_scale() -> Self {
        Self {
            segments_per_tier: 10,
            // Wide fan-in absorbs continuous-ingestion floods of small
            // memtable segments in fewer passes (350-segment backlogs at
            // 30M docs were observed with fan-in 10). Giant merges are safe:
            // output docs are capped by max_merged_docs and merge-time BP is
            // wall-clock budgeted (IndexConfig::merge_bp_time_budget).
            max_merge_at_once: 24,
            tier_factor: 10.0,
            tier_floor: 50_000,
            // Bound replacement work and resident per-document metadata. The
            // historical five-million-document cap remains unchanged; tuning
            // it requires mixed-field ingest and maintenance measurements.
            max_merged_docs: 5_000_000,
            floor_segment_docs: 50_000,
            oversized_threshold: 0.5,
            min_growth_ratio: 0.5,
            budget_trigger: true,
            scored_selection: true,
            max_segment_docs: 5_000_000,
        }
    }

    /// Bulk-indexing merge policy for high-throughput initial loads.
    ///
    /// Uses larger merge batches and higher thresholds to maximize throughput.
    /// Call `force_merge()` after the bulk load is complete.
    pub fn bulk_indexing() -> Self {
        Self {
            segments_per_tier: 20,
            max_merge_at_once: 20,
            tier_factor: 10.0,
            tier_floor: 100_000,
            max_merged_docs: 50_000_000,
            floor_segment_docs: 100_000,
            oversized_threshold: 0.5,
            min_growth_ratio: 0.75,
            budget_trigger: true,
            scored_selection: true,
            max_segment_docs: 50_000_000,
        }
    }
}

impl TieredMergePolicy {
    /// Effective maximum docs for a merged segment, considering both
    /// `max_merged_docs` and `max_segment_docs`.
    fn effective_max_docs(&self) -> u32 {
        self.max_merged_docs.min(self.max_segment_docs)
    }

    /// Compute the ideal segment count for the given total document count.
    /// Based on Lucene's budget model: segments arrange in tiers of `tier_factor`
    /// width, with up to `segments_per_tier` segments per tier.
    fn compute_ideal_segment_count(&self, total_docs: u64) -> usize {
        if total_docs == 0 {
            return 0;
        }
        let floor = self.floor_segment_docs.max(1) as f64;
        // Number of tiers needed to cover total_docs
        let num_tiers = ((total_docs as f64 / floor).max(1.0))
            .log(self.tier_factor)
            .ceil() as usize;
        let num_tiers = num_tiers.max(1);
        num_tiers * self.segments_per_tier
    }

    fn eligible_segments<'a>(&self, segments: &'a [SegmentInfo]) -> Vec<&'a SegmentInfo> {
        let effective_max = self.effective_max_docs();
        let oversized_limit = (effective_max as f64 * self.oversized_threshold) as u64;
        segments
            .iter()
            .filter(|segment| (segment.num_docs as u64) <= oversized_limit || oversized_limit == 0)
            .collect()
    }

    /// Score a merge group. Lower score = better (more balanced) merge.
    /// Uses Lucene-style skew scoring: `skew * size_factor`.
    /// - skew = largest_floored / total_floored (1/N for perfectly balanced, 1.0 for singleton)
    /// - size_factor = total_floored^0.05 (mild preference for larger merges)
    fn score_candidate(&self, group: &[usize], sorted: &[&SegmentInfo]) -> f64 {
        let floor = self.floor_segment_docs.max(1) as f64;
        let mut total_floored = 0.0f64;
        let mut largest_floored = 0.0f64;
        for &idx in group {
            let floored = (sorted[idx].num_docs as f64).max(floor);
            total_floored += floored;
            if floored > largest_floored {
                largest_floored = floored;
            }
        }
        if total_floored == 0.0 {
            return f64::MAX;
        }
        let skew = largest_floored / total_floored;
        skew * total_floored.powf(0.05)
    }

    /// Check whether a merge group passes the minimum growth ratio.
    /// Returns true if the output (total docs) is at least `(1 + ratio) * largest_input`.
    fn passes_min_growth(&self, group: &[usize], sorted: &[&SegmentInfo]) -> bool {
        if self.min_growth_ratio <= 0.0 || group.len() < 2 {
            return true;
        }
        let largest = group
            .iter()
            .map(|&i| sorted[i].num_docs as u64)
            .max()
            .unwrap_or(0);
        let total: u64 = group.iter().map(|&i| sorted[i].num_docs as u64).sum();
        total as f64 >= (1.0 + self.min_growth_ratio) * largest as f64
    }

    /// Greedy merge selection — the original algorithm with min_growth_ratio added.
    fn find_merges_greedy(&self, sorted: &[&SegmentInfo]) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();
        let mut used = vec![false; sorted.len()];
        let max_ratio = self.tier_factor as u64;
        let effective_max = self.effective_max_docs() as u64;

        let mut start = 0;
        loop {
            while start < sorted.len() && used[start] {
                start += 1;
            }
            if start >= sorted.len() {
                break;
            }

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
                if total_docs + next_docs > effective_max {
                    break;
                }
                if next_docs > total_docs.max(1) * max_ratio {
                    break;
                }
                group.push(j);
                total_docs += next_docs;
            }

            if group.len() >= self.segments_per_tier
                && group.len() >= 2
                && self.passes_min_growth(&group, sorted)
            {
                for &i in &group {
                    used[i] = true;
                }
                candidates.push(MergeCandidate {
                    segment_ids: group.iter().map(|&i| sorted[i].id.clone()).collect(),
                });
            }

            start += 1;
        }

        candidates
    }

    /// Scored merge selection — evaluates all possible groups and picks the
    /// most balanced ones using skew scoring.
    fn find_merges_scored(&self, sorted: &[&SegmentInfo]) -> Vec<MergeCandidate> {
        let max_ratio = self.tier_factor as u64;
        let effective_max = self.effective_max_docs() as u64;

        // Build all valid merge groups with their scores
        let mut scored_groups: Vec<(f64, Vec<usize>)> = Vec::new();

        for start in 0..sorted.len() {
            let mut group = vec![start];
            let mut total_docs: u64 = sorted[start].num_docs as u64;

            for j in (start + 1)..sorted.len() {
                if group.len() >= self.max_merge_at_once {
                    break;
                }
                let next_docs = sorted[j].num_docs as u64;
                if total_docs + next_docs > effective_max {
                    break;
                }
                if next_docs > total_docs.max(1) * max_ratio {
                    break;
                }
                group.push(j);
                total_docs += next_docs;

                // Record every valid group (>= segments_per_tier)
                if group.len() >= self.segments_per_tier
                    && group.len() >= 2
                    && self.passes_min_growth(&group, sorted)
                {
                    let score = self.score_candidate(&group, sorted);
                    scored_groups.push((score, group.clone()));
                }
            }
        }

        // Sort by score ascending (best first)
        scored_groups.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Greedily select non-overlapping candidates
        let mut used = vec![false; sorted.len()];
        let mut candidates = Vec::new();

        for (_score, group) in scored_groups {
            if group.iter().any(|&i| used[i]) {
                continue;
            }
            for &i in &group {
                used[i] = true;
            }
            candidates.push(MergeCandidate {
                segment_ids: group.iter().map(|&i| sorted[i].id.clone()).collect(),
            });
        }

        candidates
    }
}

impl MergePolicy for TieredMergePolicy {
    fn find_merges(&self, segments: &[SegmentInfo]) -> Vec<MergeCandidate> {
        if segments.len() < 2 {
            return Vec::new();
        }

        // Phase 1: Filter oversized segments
        let eligible = self.eligible_segments(segments);

        if eligible.len() < 2 {
            return Vec::new();
        }

        // Phase 2: Budget check — skip merging if segment count is healthy
        if self.budget_trigger {
            let total_docs: u64 = segments.iter().map(|s| s.num_docs as u64).sum();
            let ideal = self.compute_ideal_segment_count(total_docs);
            if eligible.len() <= ideal {
                return Vec::new();
            }
        }

        // Sort eligible segments by size ascending
        let mut sorted = eligible;
        sorted.sort_by_key(|s| s.num_docs);

        // Phase 3: Select merge candidates
        if self.scored_selection {
            self.find_merges_scored(&sorted)
        } else {
            self.find_merges_greedy(&sorted)
        }
    }

    fn has_severe_backlog(&self, segments: &[SegmentInfo]) -> bool {
        if !self.budget_trigger {
            return false;
        }
        let eligible = self.eligible_segments(segments);
        if eligible.len() < 2 {
            return false;
        }
        let total_docs: u64 = segments
            .iter()
            .map(|segment| u64::from(segment.num_docs))
            .sum();
        let ideal = self.compute_ideal_segment_count(total_docs);
        eligible.len() > ideal.saturating_mul(2)
    }

    fn clone_box(&self) -> Box<dyn MergePolicy> {
        Box::new(self.clone())
    }

    fn max_segment_docs(&self) -> Option<u32> {
        Some(self.max_segment_docs)
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
    fn test_tiered_policy_large_segment_not_remerged_with_small() {
        // Simulates the user scenario: after merging, we have one large segment
        // and a few new small segments from recent commits. The large segment
        // should NOT be re-merged — only the small ones should merge together
        // once there are enough of them.
        let policy = TieredMergePolicy::default(); // segments_per_tier=10

        // 1 large segment (from previous merge) + 5 new small segments
        let mut segments = vec![SegmentInfo {
            id: "large_merged".into(),
            num_docs: 50_000,
        }];
        for i in 0..5 {
            segments.push(SegmentInfo {
                id: format!("new_{}", i),
                num_docs: 500,
            });
        }

        // Should NOT merge: only 5 small segments (< segments_per_tier=10),
        // and the large segment is too big to join their group.
        let candidates = policy.find_merges(&segments);
        assert!(
            candidates.is_empty(),
            "should not re-merge large segment with 5 small ones: {:?}",
            candidates
        );

        // Now add 5 more small segments (total 10) — those should merge together,
        // but the large segment should still be excluded.
        for i in 5..10 {
            segments.push(SegmentInfo {
                id: format!("new_{}", i),
                num_docs: 500,
            });
        }

        let candidates = policy.find_merges(&segments);
        assert_eq!(candidates.len(), 1, "should merge the 10 small segments");
        assert!(
            !candidates[0].segment_ids.contains(&"large_merged".into()),
            "large segment must NOT be in the merge group"
        );
        assert_eq!(
            candidates[0].segment_ids.len(),
            10,
            "all 10 small segments should be merged"
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

    #[test]
    fn test_oversized_exclusion() {
        // max_merged_docs=1M, oversized_threshold=0.5 → exclude segments > 500K
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            max_merged_docs: 1_000_000,
            oversized_threshold: 0.5,
            ..Default::default()
        };

        // 4 small segments + 2 oversized segments (600K each)
        let mut segments: Vec<_> = (0..4)
            .map(|i| SegmentInfo {
                id: format!("small_{}", i),
                num_docs: 1000,
            })
            .collect();
        segments.push(SegmentInfo {
            id: "oversized_0".into(),
            num_docs: 600_000,
        });
        segments.push(SegmentInfo {
            id: "oversized_1".into(),
            num_docs: 700_000,
        });

        let candidates = policy.find_merges(&segments);
        // Oversized segments must not appear in any merge candidate
        for c in &candidates {
            assert!(
                !c.segment_ids.contains(&"oversized_0".into()),
                "oversized_0 should be excluded"
            );
            assert!(
                !c.segment_ids.contains(&"oversized_1".into()),
                "oversized_1 should be excluded"
            );
        }
    }

    #[test]
    fn test_budget_trigger_prevents_unnecessary_merge() {
        let policy = TieredMergePolicy {
            segments_per_tier: 10,
            tier_factor: 10.0,
            tier_floor: 1000,
            floor_segment_docs: 1000,
            budget_trigger: true,
            ..Default::default()
        };

        // 5 segments of 10K docs each = 50K total
        // Budget: ceil(log10(50K/1000)) = ceil(log10(50)) = 2 tiers → 2*10 = 20 ideal segments
        // 5 segments < 20 ideal → should NOT merge
        let segments: Vec<_> = (0..5)
            .map(|i| SegmentInfo {
                id: format!("seg_{}", i),
                num_docs: 10_000,
            })
            .collect();

        let candidates = policy.find_merges(&segments);
        assert!(
            candidates.is_empty(),
            "should not merge when under budget: {:?}",
            candidates
        );
    }

    #[test]
    fn test_budget_trigger_allows_merge_when_over_budget() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            tier_factor: 10.0,
            tier_floor: 1000,
            floor_segment_docs: 1000,
            budget_trigger: true,
            ..Default::default()
        };

        // 10 segments of 1000 docs = 10K total
        // Budget: ceil(log10(10K/1000)) = 1 tier → 1*3 = 3 ideal segments
        // 10 segments > 3 ideal → should merge
        let segments: Vec<_> = (0..10)
            .map(|i| SegmentInfo {
                id: format!("seg_{}", i),
                num_docs: 1000,
            })
            .collect();

        let candidates = policy.find_merges(&segments);
        assert!(!candidates.is_empty(), "should merge when over budget");
    }

    #[test]
    fn test_severe_backlog_requires_more_than_twice_the_budget() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            tier_factor: 10.0,
            tier_floor: 1000,
            floor_segment_docs: 1000,
            budget_trigger: true,
            ..Default::default()
        };

        // Every segment is at the floor, so 3 segments is the ideal budget.
        // Ordinary pressure still uses reorder-on-merge.
        let six: Vec<_> = (0..6)
            .map(|i| SegmentInfo {
                id: format!("seg_{i}"),
                num_docs: 1000,
            })
            .collect();
        assert!(!policy.has_severe_backlog(&six));

        // Once the eligible population exceeds 2x budget, topology reduction
        // becomes urgent and merge-time BP should be deferred.
        let mut seven = six;
        seven.push(SegmentInfo {
            id: "seg_6".into(),
            num_docs: 1000,
        });
        assert!(policy.has_severe_backlog(&seven));
    }

    #[test]
    fn test_min_growth_ratio_rejects_wasteful_merge() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            min_growth_ratio: 0.5,
            max_merge_at_once: 10,
            ..Default::default()
        };

        // 1 large segment (100K) + 3 tiny segments (10 docs each)
        // Total = 100_030, largest = 100_000
        // Growth check: 100_030 >= 1.5 * 100_000 = 150_000? NO → reject
        let mut segments = vec![SegmentInfo {
            id: "big".into(),
            num_docs: 100_000,
        }];
        for i in 0..3 {
            segments.push(SegmentInfo {
                id: format!("tiny_{}", i),
                num_docs: 10,
            });
        }

        let candidates = policy.find_merges(&segments);
        // The 3 tiny segments alone can form a group, but the big one shouldn't
        // be merged with them. Let's verify no candidate includes "big".
        for c in &candidates {
            if c.segment_ids.contains(&"big".into()) {
                let total: u64 = c
                    .segment_ids
                    .iter()
                    .map(|id| segments.iter().find(|s| s.id == *id).unwrap().num_docs as u64)
                    .sum();
                let largest: u64 = c
                    .segment_ids
                    .iter()
                    .map(|id| segments.iter().find(|s| s.id == *id).unwrap().num_docs as u64)
                    .max()
                    .unwrap();
                assert!(
                    total as f64 >= 1.5 * largest as f64,
                    "merge with 'big' segment violates min_growth_ratio: total={}, largest={}",
                    total,
                    largest
                );
            }
        }
    }

    #[test]
    fn test_scored_selection_prefers_balanced_merge() {
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            max_merge_at_once: 5,
            scored_selection: true,
            ..Default::default()
        };

        // Group A: 3 balanced segments (1000, 1100, 1200)
        // Group B: 3 unbalanced segments (100, 100, 5000) — placed so greedy would pick them first
        let segments = vec![
            SegmentInfo {
                id: "unbal_0".into(),
                num_docs: 100,
            },
            SegmentInfo {
                id: "unbal_1".into(),
                num_docs: 100,
            },
            SegmentInfo {
                id: "bal_0".into(),
                num_docs: 1000,
            },
            SegmentInfo {
                id: "bal_1".into(),
                num_docs: 1100,
            },
            SegmentInfo {
                id: "bal_2".into(),
                num_docs: 1200,
            },
            SegmentInfo {
                id: "unbal_2".into(),
                num_docs: 5000,
            },
        ];

        let candidates = policy.find_merges(&segments);
        assert!(!candidates.is_empty(), "should find at least one merge");

        // The first candidate (best score) should be the balanced group
        let first = &candidates[0];
        let has_balanced = first.segment_ids.iter().any(|id| id.starts_with("bal_"));
        assert!(
            has_balanced,
            "scored selection should prefer balanced group, got: {:?}",
            first.segment_ids
        );
    }

    #[test]
    fn test_large_scale_preset_values() {
        let p = TieredMergePolicy::large_scale();
        assert_eq!(p.tier_floor, 50_000);
        // 5M cap (was 20M): BP reorder scales with sparse (doc, ordinal)
        // entries (~5x docs in prod), and a 20M-doc segment produced ~95M
        // entries that could never converge within the BP time/memory
        // budgets. See large_scale() comment.
        assert_eq!(p.max_merged_docs, 5_000_000);
        assert_eq!(p.max_segment_docs, 5_000_000);
        assert_eq!(p.floor_segment_docs, 50_000);
        assert!(p.budget_trigger);
        assert!(p.scored_selection);
        assert_eq!(p.segments_per_tier, 10);
        assert!((p.min_growth_ratio - 0.5).abs() < f64::EPSILON);
        assert!((p.oversized_threshold - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bulk_indexing_preset_values() {
        let p = TieredMergePolicy::bulk_indexing();
        assert_eq!(p.segments_per_tier, 20);
        assert_eq!(p.max_merge_at_once, 20);
        assert_eq!(p.tier_floor, 100_000);
        assert_eq!(p.max_merged_docs, 50_000_000);
        assert_eq!(p.max_segment_docs, 50_000_000);
        assert_eq!(p.floor_segment_docs, 100_000);
        assert!(p.budget_trigger);
        assert!(p.scored_selection);
        assert!((p.min_growth_ratio - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_max_segment_docs() {
        let p = TieredMergePolicy::default();
        assert_eq!(p.max_segment_docs, 10_000_000);
        assert_eq!(p.max_segment_docs().unwrap(), 10_000_000);
    }

    #[test]
    fn test_max_segment_docs_caps_merge_output() {
        // max_merged_docs=50M but max_segment_docs=5M → effective max is 5M
        let policy = TieredMergePolicy {
            segments_per_tier: 3,
            max_merge_at_once: 100,
            max_merged_docs: 50_000_000,
            max_segment_docs: 5_000_000,
            ..Default::default()
        };

        // 10 segments of 1M each — total would be 10M but max_segment_docs=5M
        let segments: Vec<_> = (0..10)
            .map(|i| SegmentInfo {
                id: format!("seg_{}", i),
                num_docs: 1_000_000,
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
                total <= 5_000_000,
                "merge total {} exceeds max_segment_docs 5M",
                total
            );
        }
    }
}
