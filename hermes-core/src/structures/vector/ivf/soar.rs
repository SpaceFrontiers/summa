//! SOAR: Spilling with Orthogonality-Amplified Residuals
//!
//! Implementation of Google's SOAR algorithm for improved IVF recall:
//! - Assigns vectors to multiple clusters (primary + secondary)
//! - Secondary clusters chosen to have orthogonal residuals
//! - When query is parallel to primary residual (high error), secondary has low error
//!
//! Reference: "SOAR: New algorithms for even faster vector search with ScaNN"
//! https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/

use serde::{Deserialize, Serialize};

/// Configuration for SOAR (Spilling with Orthogonality-Amplified Residuals)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoarConfig {
    /// Number of secondary cluster assignments (typically 1-2)
    pub num_secondary: usize,
    /// Use selective spilling (only spill vectors near cluster boundaries)
    pub selective: bool,
    /// Threshold for selective spilling (residual norm must exceed this)
    pub spill_threshold: f32,
}

impl Default for SoarConfig {
    fn default() -> Self {
        Self {
            num_secondary: 1,
            selective: true,
            spill_threshold: 0.5,
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
            num_secondary,
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
        self.spill_threshold = threshold;
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

    /// Aggressive spilling with 2 secondary clusters
    pub fn aggressive() -> Self {
        Self {
            num_secondary: 2,
            selective: false,
            spill_threshold: 0.0,
        }
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
