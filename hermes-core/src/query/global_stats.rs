//! Global statistics for cross-segment IDF computation
//!
//! Provides cached aggregated statistics across multiple segments for:
//! - Sparse vector dimensions (for sparse vector queries)
//! - Full-text terms (for BM25/TF-IDF scoring)
//!
//! This implements a coordinator-style approach where statistics are gathered
//! from all segments, cached, and used for consistent IDF scoring.

use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::dsl::Field;
use crate::segment::SegmentReader;

/// Global statistics aggregated across all segments
///
/// Used for consistent IDF computation in multi-segment indexes.
/// Statistics are cached and invalidated when segments change.
#[derive(Debug)]
pub struct GlobalStats {
    /// Total documents across all segments
    total_docs: u64,
    /// Sparse vector statistics per field: field_id -> dimension stats
    sparse_stats: FxHashMap<u32, SparseFieldStats>,
    /// Full-text statistics per field: field_id -> term stats
    text_stats: FxHashMap<u32, TextFieldStats>,
    /// Generation counter for cache invalidation
    generation: u64,
}

/// Statistics for a sparse vector field
#[derive(Debug, Default)]
pub struct SparseFieldStats {
    /// Document frequency per dimension: dim_id -> doc_count
    pub doc_freqs: FxHashMap<u32, u64>,
}

/// Statistics for a full-text field
#[derive(Debug, Default)]
pub struct TextFieldStats {
    /// Document frequency per term: term -> doc_count
    pub doc_freqs: FxHashMap<String, u64>,
    /// Average field length (for BM25)
    pub avg_field_len: f32,
}

impl GlobalStats {
    /// Create empty stats
    pub fn new() -> Self {
        Self {
            total_docs: 0,
            sparse_stats: FxHashMap::default(),
            text_stats: FxHashMap::default(),
            generation: 0,
        }
    }

    /// Total documents in the index
    #[inline]
    pub fn total_docs(&self) -> u64 {
        self.total_docs
    }

    /// Compute IDF for a sparse vector dimension
    ///
    /// IDF = ln(N / df) where N = total docs, df = docs containing dimension
    #[inline]
    pub fn sparse_idf(&self, field: Field, dim_id: u32) -> f32 {
        if let Some(stats) = self.sparse_stats.get(&field.0)
            && let Some(&df) = stats.doc_freqs.get(&dim_id)
            && df > 0
        {
            return (self.total_docs as f32 / df as f32).ln();
        }
        0.0
    }

    /// Compute IDF weights for multiple sparse dimensions
    pub fn sparse_idf_weights(&self, field: Field, dim_ids: &[u32]) -> Vec<f32> {
        dim_ids.iter().map(|&d| self.sparse_idf(field, d)).collect()
    }

    /// Compute IDF for a full-text term
    ///
    /// IDF = ln((N - df + 0.5) / (df + 0.5) + 1) (BM25 variant)
    #[inline]
    pub fn text_idf(&self, field: Field, term: &str) -> f32 {
        if let Some(stats) = self.text_stats.get(&field.0)
            && let Some(&df) = stats.doc_freqs.get(term)
        {
            let n = self.total_docs as f32;
            let df = df as f32;
            // BM25 IDF formula
            return ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        }
        0.0
    }

    /// Get average field length for BM25
    #[inline]
    pub fn avg_field_len(&self, field: Field) -> f32 {
        self.text_stats
            .get(&field.0)
            .map(|s| s.avg_field_len)
            .unwrap_or(1.0)
    }

    /// Current generation (for cache invalidation)
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl Default for GlobalStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for aggregating statistics from multiple segments
pub struct GlobalStatsBuilder {
    /// Total documents across all segments
    pub total_docs: u64,
    sparse_stats: FxHashMap<u32, SparseFieldStats>,
    text_stats: FxHashMap<u32, TextFieldStats>,
}

impl GlobalStatsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            total_docs: 0,
            sparse_stats: FxHashMap::default(),
            text_stats: FxHashMap::default(),
        }
    }

    /// Add statistics from a segment reader
    pub fn add_segment(&mut self, reader: &SegmentReader) {
        self.total_docs += reader.num_docs() as u64;

        // Aggregate sparse vector statistics
        // Note: This requires access to sparse_indexes which may need to be exposed
    }

    /// Add sparse dimension document frequency
    pub fn add_sparse_df(&mut self, field: Field, dim_id: u32, doc_count: u64) {
        let stats = self.sparse_stats.entry(field.0).or_default();
        *stats.doc_freqs.entry(dim_id).or_insert(0) += doc_count;
    }

    /// Add text term document frequency
    pub fn add_text_df(&mut self, field: Field, term: String, doc_count: u64) {
        let stats = self.text_stats.entry(field.0).or_default();
        *stats.doc_freqs.entry(term).or_insert(0) += doc_count;
    }

    /// Set average field length for a text field
    pub fn set_avg_field_len(&mut self, field: Field, avg_len: f32) {
        let stats = self.text_stats.entry(field.0).or_default();
        stats.avg_field_len = avg_len;
    }

    /// Build the final GlobalStats
    pub fn build(self, generation: u64) -> GlobalStats {
        GlobalStats {
            total_docs: self.total_docs,
            sparse_stats: self.sparse_stats,
            text_stats: self.text_stats,
            generation,
        }
    }
}

impl Default for GlobalStatsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached global statistics with automatic invalidation
///
/// This is the main entry point for getting global IDF values.
/// It caches statistics and rebuilds them when the segment list changes.
pub struct GlobalStatsCache {
    /// Cached statistics
    stats: RwLock<Option<Arc<GlobalStats>>>,
    /// Current generation (incremented when segments change)
    generation: RwLock<u64>,
}

impl GlobalStatsCache {
    /// Create a new cache
    pub fn new() -> Self {
        Self {
            stats: RwLock::new(None),
            generation: RwLock::new(0),
        }
    }

    /// Invalidate the cache (call when segments are added/removed/merged)
    pub fn invalidate(&self) {
        let mut current_gen = self.generation.write();
        *current_gen += 1;
        let mut stats = self.stats.write();
        *stats = None;
    }

    /// Get current generation
    pub fn generation(&self) -> u64 {
        *self.generation.read()
    }

    /// Get cached stats if valid, or None if needs rebuild
    pub fn get(&self) -> Option<Arc<GlobalStats>> {
        self.stats.read().clone()
    }

    /// Update the cache with new stats
    pub fn set(&self, stats: GlobalStats) {
        let mut cached = self.stats.write();
        *cached = Some(Arc::new(stats));
    }

    /// Get or compute stats using the provided builder function (sync version)
    ///
    /// For basic stats that don't require async iteration.
    pub fn get_or_compute<F>(&self, compute: F) -> Arc<GlobalStats>
    where
        F: FnOnce(&mut GlobalStatsBuilder),
    {
        // Fast path: return cached if available
        if let Some(stats) = self.get() {
            return stats;
        }

        // Slow path: compute new stats
        let current_gen = self.generation();
        let mut builder = GlobalStatsBuilder::new();
        compute(&mut builder);
        let stats = Arc::new(builder.build(current_gen));

        // Cache the result
        let mut cached = self.stats.write();
        *cached = Some(Arc::clone(&stats));

        stats
    }

    /// Check if stats need to be rebuilt
    pub fn needs_rebuild(&self) -> bool {
        self.stats.read().is_none()
    }

    /// Set pre-built stats (for async computation)
    pub fn set_stats(&self, stats: GlobalStats) {
        let mut cached = self.stats.write();
        *cached = Some(Arc::new(stats));
    }
}

impl Default for GlobalStatsCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_idf_computation() {
        let mut builder = GlobalStatsBuilder::new();
        builder.total_docs = 1000;
        builder.add_sparse_df(Field(0), 42, 100); // dim 42 appears in 100 docs
        builder.add_sparse_df(Field(0), 43, 10); // dim 43 appears in 10 docs

        let stats = builder.build(1);

        // IDF = ln(N/df)
        let idf_42 = stats.sparse_idf(Field(0), 42);
        let idf_43 = stats.sparse_idf(Field(0), 43);

        // dim 43 should have higher IDF (rarer)
        assert!(idf_43 > idf_42);
        assert!((idf_42 - (1000.0_f32 / 100.0).ln()).abs() < 0.001);
        assert!((idf_43 - (1000.0_f32 / 10.0).ln()).abs() < 0.001);
    }

    #[test]
    fn test_text_idf_computation() {
        let mut builder = GlobalStatsBuilder::new();
        builder.total_docs = 10000;
        builder.add_text_df(Field(0), "common".to_string(), 5000);
        builder.add_text_df(Field(0), "rare".to_string(), 10);

        let stats = builder.build(1);

        let idf_common = stats.text_idf(Field(0), "common");
        let idf_rare = stats.text_idf(Field(0), "rare");

        // Rare term should have higher IDF
        assert!(idf_rare > idf_common);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = GlobalStatsCache::new();

        // Initially no stats
        assert!(cache.get().is_none());

        // Compute stats
        let stats = cache.get_or_compute(|builder| {
            builder.total_docs = 100;
        });
        assert_eq!(stats.total_docs(), 100);

        // Should be cached now
        assert!(cache.get().is_some());

        // Invalidate
        cache.invalidate();
        assert!(cache.get().is_none());
    }
}
