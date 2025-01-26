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
use crate::structures::{CoarseCentroids, PQCodebook};

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
    /// Trained centroids
    trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Trained codebooks
    trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
    /// Term cache blocks
    term_cache_blocks: usize,
    /// Last reload time
    last_reload: RwLock<std::time::Instant>,
    /// Reload interval (default 1 second)
    reload_interval: std::time::Duration,
}

impl<D: DirectoryWriter + 'static> IndexReader<D> {
    /// Create a new searcher from a segment manager
    pub async fn from_segment_manager(
        schema: Arc<Schema>,
        segment_manager: Arc<crate::merge::SegmentManager<D>>,
        trained_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
        trained_codebooks: FxHashMap<u32, Arc<PQCodebook>>,
        term_cache_blocks: usize,
    ) -> Result<Self> {
        let reader = Self::create_reader(
            &schema,
            &segment_manager,
            &trained_centroids,
            &trained_codebooks,
            term_cache_blocks,
        )
        .await?;

        Ok(Self {
            schema,
            segment_manager,
            searcher: RwLock::new(Arc::new(reader)),
            trained_centroids,
            trained_codebooks,
            term_cache_blocks,
            last_reload: RwLock::new(std::time::Instant::now()),
            reload_interval: std::time::Duration::from_secs(1),
        })
    }

    /// Create a new reader with fresh snapshot from segment manager
    /// This avoids race conditions by using SegmentManager's locked metadata
    async fn create_reader(
        schema: &Arc<Schema>,
        segment_manager: &Arc<crate::merge::SegmentManager<D>>,
        trained_centroids: &FxHashMap<u32, Arc<CoarseCentroids>>,
        trained_codebooks: &FxHashMap<u32, Arc<PQCodebook>>,
        term_cache_blocks: usize,
    ) -> Result<Searcher<D>> {
        // Use SegmentManager's acquire_snapshot - this locks metadata and tracker together
        let snapshot = segment_manager.acquire_snapshot().await;

        Searcher::from_snapshot(
            segment_manager.directory(),
            Arc::clone(schema),
            snapshot,
            trained_centroids.clone(),
            trained_codebooks.clone(),
            term_cache_blocks,
        )
        .await
    }

    /// Set reload interval
    pub fn set_reload_interval(&mut self, interval: std::time::Duration) {
        self.reload_interval = interval;
    }

    /// Get current searcher (reloads if interval exceeded)
    pub async fn searcher(&self) -> Result<Arc<Searcher<D>>> {
        // Check if reload needed
        let should_reload = {
            let last = self.last_reload.read();
            last.elapsed() >= self.reload_interval
        };

        if should_reload {
            self.reload().await?;
        }

        Ok(Arc::clone(&*self.searcher.read()))
    }

    /// Force reload reader with fresh snapshot
    pub async fn reload(&self) -> Result<()> {
        let new_reader = Self::create_reader(
            &self.schema,
            &self.segment_manager,
            &self.trained_centroids,
            &self.trained_codebooks,
            self.term_cache_blocks,
        )
        .await?;

        // Swap in new searcher (old one will be dropped when last ref released)
        *self.searcher.write() = Arc::new(new_reader);
        *self.last_reload.write() = std::time::Instant::now();

        Ok(())
    }

    /// Get schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
