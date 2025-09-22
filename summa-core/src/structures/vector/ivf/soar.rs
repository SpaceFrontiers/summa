//! SOAR: Spilling with Orthogonality-Amplified Residuals
//!
//! Implementation of Google's SOAR algorithm for improved IVF recall:
//! - Assigns vectors to multiple clusters (primary + secondary)
//! - Secondary clusters chosen to have orthogonal residuals
//! - When query is parallel to primary residual (high error), secondary has low error
//!
//! Reference: "SOAR: New algorithms for even faster vector search with ScaNN"
//! <https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/>

use serde::{Deserialize, Serialize};

const DEFAULT_SELECTIVE_SPILL_FRACTION: f32 = 0.30;

/// Configuration for SOAR (Spilling with Orthogonality-Amplified Residuals)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoarConfig {
    /// Number of secondary cluster assignments. Current trained generations
    /// use the published two-assignment objective, so builders clamp this to
    /// one secondary.
    pub num_secondary: usize,
    /// Use selective spilling (only spill vectors near cluster boundaries)
    pub selective: bool,
    /// Positive values are calibrated residual-norm thresholds. A negative
    /// value requests build-time calibration to the corresponding spill
    /// fraction; trained artifacts always persist a positive threshold. This
    /// tagged representation preserves the serialized structure layout.
    pub spill_threshold: f32,
}

impl Default for SoarConfig {
    fn default() -> Self {
        Self {
            num_secondary: 1,
            selective: true,
            spill_threshold: -DEFAULT_SELECTIVE_SPILL_FRACTION,
        }
    }
}

impl SoarConfig {
    /// Create SOAR config with 1 secondary assignment
    pub fn new() -> Self {
        Self::default()
    }

    /// Create SOAR config with specified number of secondary assignments
    pub fn with_secondary(num_secondary: usize) -> Self {
        Self {
            num_secondary: num_secondary.min(1),
            ..Default::default()
        }
    }

    /// Enable/disable selective spilling
    pub fn selective(mut self, enabled: bool) -> Self {
        self.selective = enabled;
        self
    }

    /// Set spill threshold for selective spilling
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.spill_threshold = threshold.max(0.0);
        self
    }

    /// Calibrate selective spilling during training to at most a target
    /// fraction of vectors receiving one secondary assignment.
    pub fn target_spill_fraction(mut self, fraction: f32) -> Self {
        self.selective = true;
        self.spill_threshold = -fraction.clamp(0.0, 1.0);
        self
    }

    /// Full spilling (no selectivity) - assigns all vectors to secondary clusters
    pub fn full() -> Self {
        Self {
            num_secondary: 1,
            selective: false,
            spill_threshold: 0.0,
        }
    }

    /// Compatibility alias for full one-secondary spilling. The generalized
    /// multi-secondary objective is intentionally not exposed until it is
    /// implemented and validated.
    pub fn aggressive() -> Self {
        Self {
            num_secondary: 1,
            selective: false,
            spill_threshold: 0.0,
        }
    }

    pub(crate) fn calibration_target(&self) -> Option<f32> {
        (self.selective && self.spill_threshold.is_sign_negative())
            .then(|| (-self.spill_threshold).clamp(0.0, 1.0))
    }
}

/// Multi-cluster assignment result from SOAR
#[derive(Debug, Clone)]
pub struct MultiAssignment {
    /// Primary cluster (nearest centroid)
    pub primary_cluster: u32,
    /// Secondary clusters (orthogonal residuals)
    pub secondary_clusters: Vec<u32>,
}

impl MultiAssignment {
    /// Create assignment with only primary cluster
    pub fn primary_only(cluster: u32) -> Self {
        Self {
            primary_cluster: cluster,
            secondary_clusters: Vec::new(),
        }
    }

    /// Get all clusters (primary + secondary)
    pub fn all_clusters(&self) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(self.primary_cluster).chain(self.secondary_clusters.iter().copied())
    }

    /// Total number of cluster assignments
    pub fn num_assignments(&self) -> usize {
        1 + self.secondary_clusters.len()
    }

    /// Check if this is a spilled assignment (has secondary clusters)
    pub fn is_spilled(&self) -> bool {
        !self.secondary_clusters.is_empty()
    }
}

/// Statistics for SOAR assignments
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SoarStats {
    /// Total vectors assigned
    pub total_vectors: usize,
    /// Vectors with secondary assignments (spilled)
    pub spilled_vectors: usize,
    /// Total cluster assignments (including secondary)
    pub total_assignments: usize,
}

#[allow(dead_code)]
impl SoarStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an assignment
    pub fn record(&mut self, assignment: &MultiAssignment) {
        self.total_vectors += 1;
        self.total_assignments += assignment.num_assignments();
        if assignment.is_spilled() {
            self.spilled_vectors += 1;
        }
    }

    /// Spill ratio (fraction of vectors with secondary assignments)
    pub fn spill_ratio(&self) -> f32 {
        if self.total_vectors == 0 {
            0.0
        } else {
            self.spilled_vectors as f32 / self.total_vectors as f32
        }
    }

    /// Average assignments per vector
    pub fn avg_assignments(&self) -> f32 {
        if self.total_vectors == 0 {
            0.0
        } else {
            self.total_assignments as f32 / self.total_vectors as f32
        }
    }

    /// Storage overhead factor (1.0 = no overhead, 2.0 = 2x storage)
    pub fn storage_factor(&self) -> f32 {
        self.avg_assignments()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soar_config_default() {
        let config = SoarConfig::default();
        assert_eq!(config.num_secondary, 1);
        assert!(config.selective);
        assert_eq!(config.calibration_target(), Some(0.30));
    }

    #[test]
    fn explicit_threshold_and_target_budget_have_distinct_tags() {
        let threshold = SoarConfig::new().threshold(0.42);
        assert_eq!(threshold.calibration_target(), None);
        assert_eq!(threshold.spill_threshold, 0.42);

        let target = SoarConfig::new().target_spill_fraction(0.25);
        assert_eq!(target.calibration_target(), Some(0.25));
    }

    #[test]
    fn test_multi_assignment() {
        let assignment = MultiAssignment {
            primary_cluster: 5,
            secondary_clusters: vec![2, 7],
        };

        assert_eq!(assignment.num_assignments(), 3);
        assert!(assignment.is_spilled());

        let all: Vec<u32> = assignment.all_clusters().collect();
        assert_eq!(all, vec![5, 2, 7]);
    }

    #[test]
    fn test_soar_stats() {
        let mut stats = SoarStats::new();

        // Primary only assignment
        stats.record(&MultiAssignment::primary_only(0));

        // Spilled assignment
        stats.record(&MultiAssignment {
            primary_cluster: 1,
            secondary_clusters: vec![2],
        });

        assert_eq!(stats.total_vectors, 2);
        assert_eq!(stats.spilled_vectors, 1);
        assert_eq!(stats.total_assignments, 3);
        assert_eq!(stats.spill_ratio(), 0.5);
        assert_eq!(stats.avg_assignments(), 1.5);
    }
}
