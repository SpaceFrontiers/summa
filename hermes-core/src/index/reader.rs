//! IndexReader - manages Searcher with reload policy (native only)
//!
//! The IndexReader periodically reloads its Searcher to pick up new segments.
//! Uses SegmentManager as authoritative source for segment state.

use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::directories::DirectoryWriter;
use crate::dsl::Schema;
use crate::error::Result;
use crate::structures::CoarseCentroids;

use super::Searcher;

/// IndexReader - manages Searcher with reload policy
///
/// The IndexReader periodically reloads its Searcher to pick up new segments.
/// Uses SegmentManager as authoritative source for segment state (avoids race conditions).
pub struct IndexReader<D: DirectoryWriter + 'static> {
    /// Schema
    schema: Arc<Schema>,
    /// Segment manager - authoritative source for segments
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Current searcher
    searcher: RwLock<Arc<Searcher<D>>>,
    /// Trained centroids (injected into segment readers for IVF/ScaNN search)
    trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Term cache blocks
    term_cache_blocks: usize,
    /// Last reload check time
    last_reload_check: RwLock<std::time::Instant>,
    /// Reload check interval (default 1 second)
    reload_check_interval: std::time::Duration,
    /// Current segment IDs (to detect changes)
    current_segment_ids: RwLock<Vec<String>>,
}

impl<D: DirectoryWriter + 'static> IndexReader<D> {
    /// Create a new searcher from a segment manager
    pub async fn from_segment_manager(
        schema: Arc<Schema>,
        segment_manager: Arc<crate::merge::SegmentManager<D>>,
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
    ) -> Result<Self> {
        Self::from_segment_manager_with_reload_interval(
            schema,
            segment_manager,
            trained_centroids,
            term_cache_blocks,
            1000, // default 1 second
        )
        .await
    }

    /// Create a new searcher from a segment manager with custom reload interval
    pub async fn from_segment_manager_with_reload_interval(
        schema: Arc<Schema>,
        segment_manager: Arc<crate::merge::SegmentManager<D>>,
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
        reload_interval_ms: u64,
    ) -> Result<Self> {
        // Get initial segment IDs
        let initial_segment_ids = segment_manager.get_segment_ids().await;

        let reader = Self::create_reader(
            &schema,
            &segment_manager,
            &trained_centroids,
            term_cache_blocks,
        )
        .await?;

        Ok(Self {
            schema,
            segment_manager,
            searcher: RwLock::new(Arc::new(reader)),
            trained_centroids,
            term_cache_blocks,
            last_reload_check: RwLock::new(std::time::Instant::now()),
            reload_check_interval: std::time::Duration::from_millis(reload_interval_ms),
            current_segment_ids: RwLock::new(initial_segment_ids),
        })
    }

    /// Create a new reader with fresh snapshot from segment manager
    /// This avoids race conditions by using SegmentManager's locked metadata
    async fn create_reader(
        schema: &Arc<Schema>,
        segment_manager: &Arc<crate::merge::SegmentManager<D>>,
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        term_cache_blocks: usize,
    ) -> Result<Searcher<D>> {
        // Use SegmentManager's acquire_snapshot - non-blocking RwLock read
        let snapshot = segment_manager.acquire_snapshot().await;

        Searcher::from_snapshot(
            segment_manager.directory(),
            Arc::clone(schema),
            snapshot,
            trained_centroids.clone(),
            term_cache_blocks,
        )
        .await
    }

    /// Set reload check interval
    pub fn set_reload_interval(&mut self, interval: std::time::Duration) {
        self.reload_check_interval = interval;
    }

    /// Get current searcher (reloads only if segments changed)
    pub async fn searcher(&self) -> Result<Arc<Searcher<D>>> {
        // Check if we should check for segment changes
        let should_check = {
            let last = self.last_reload_check.read();
            last.elapsed() >= self.reload_check_interval
        };

        if should_check {
            // Update check time first to avoid concurrent checks
            *self.last_reload_check.write() = std::time::Instant::now();

            // Get current segment IDs from segment manager (non-blocking)
            let new_segment_ids = self.segment_manager.get_segment_ids().await;

            // Check if segments actually changed
            let segments_changed = {
                let current = self.current_segment_ids.read();
                *current != new_segment_ids
            };

            if segments_changed {
                let old_count = self.current_segment_ids.read().len();
                let new_count = new_segment_ids.len();
                log::info!(
                    "[index_reload] old_count={} new_count={}",
                    old_count,
                    new_count
                );
                self.reload_with_segments(new_segment_ids).await?;
            }
        }

        Ok(Arc::clone(&*self.searcher.read()))
    }

    /// Force reload reader with fresh snapshot
    pub async fn reload(&self) -> Result<()> {
        let new_segment_ids = self.segment_manager.get_segment_ids().await;
        self.reload_with_segments(new_segment_ids).await
    }

    /// Internal reload with specific segment IDs
    async fn reload_with_segments(&self, new_segment_ids: Vec<String>) -> Result<()> {
        let new_reader = Self::create_reader(
            &self.schema,
            &self.segment_manager,
            &self.trained_centroids,
            self.term_cache_blocks,
        )
        .await?;

        // Swap in new searcher and update segment IDs
        *self.searcher.write() = Arc::new(new_reader);
        *self.current_segment_ids.write() = new_segment_ids;

        Ok(())
    }

    /// Get schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
