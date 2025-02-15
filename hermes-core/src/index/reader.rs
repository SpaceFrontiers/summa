//! IndexReader - manages Searcher with reload policy (native only)
//!
//! The IndexReader periodically reloads its Searcher to pick up new segments.
//! Uses SegmentManager as authoritative source for segment state.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::directories::DirectoryWriter;
use crate::dsl::Schema;
use crate::error::Result;

use super::Searcher;

/// IndexReader - manages Searcher with reload policy
///
/// The IndexReader periodically reloads its Searcher to pick up new segments.
/// Uses SegmentManager as authoritative source for segment state (avoids race conditions).
/// Combined searcher + segment IDs, swapped atomically via ArcSwap (wait-free reads).
struct SearcherState<D: DirectoryWriter + 'static> {
    searcher: Arc<Searcher<D>>,
    segment_ids: Vec<String>,
}

pub struct IndexReader<D: DirectoryWriter + 'static> {
    /// Schema
    schema: Arc<Schema>,
    /// Segment manager - authoritative source for segments
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Current searcher + segment IDs (ArcSwap for wait-free reads)
    state: ArcSwap<SearcherState<D>>,
    /// Term cache blocks
    term_cache_blocks: usize,
    /// Last reload check time
    last_reload_check: RwLock<std::time::Instant>,
    /// Reload check interval (default 1 second)
    reload_check_interval: std::time::Duration,
    /// Guard against concurrent reloads
    reloading: AtomicBool,
}

impl<D: DirectoryWriter + 'static> IndexReader<D> {
    /// Create a new IndexReader from a segment manager
    ///
    /// Centroids are loaded dynamically from metadata on each reload,
    /// so the reader always picks up centroids trained after Index::create().
    pub async fn from_segment_manager(
        schema: Arc<Schema>,
        segment_manager: Arc<crate::merge::SegmentManager<D>>,
        term_cache_blocks: usize,
        reload_interval_ms: u64,
    ) -> Result<Self> {
        // Get initial segment IDs
        let initial_segment_ids = segment_manager.get_segment_ids().await;

        let reader = Self::create_reader(&schema, &segment_manager, term_cache_blocks).await?;

        Ok(Self {
            schema,
            segment_manager,
            state: ArcSwap::from_pointee(SearcherState {
                searcher: Arc::new(reader),
                segment_ids: initial_segment_ids,
            }),
            term_cache_blocks,
            last_reload_check: RwLock::new(std::time::Instant::now()),
            reload_check_interval: std::time::Duration::from_millis(reload_interval_ms),
            reloading: AtomicBool::new(false),
        })
    }

    /// Create a new reader with fresh snapshot from segment manager
    ///
    /// Reads trained centroids from SegmentManager's ArcSwap (lock-free).
    async fn create_reader(
        schema: &Arc<Schema>,
        segment_manager: &Arc<crate::merge::SegmentManager<D>>,
        term_cache_blocks: usize,
    ) -> Result<Searcher<D>> {
        // Read trained centroids from ArcSwap (lock-free)
        let trained = segment_manager.trained();
        let trained_centroids = trained
            .as_ref()
            .map(|t| t.centroids.clone())
            .unwrap_or_default();

        // Use SegmentManager's acquire_snapshot - non-blocking RwLock read
        let snapshot = segment_manager.acquire_snapshot().await;

        Searcher::from_snapshot(
            segment_manager.directory(),
            Arc::clone(schema),
            snapshot,
            trained_centroids,
            term_cache_blocks,
        )
        .await
    }

    /// Set reload check interval
    pub fn set_reload_interval(&mut self, interval: std::time::Duration) {
        self.reload_check_interval = interval;
    }

    /// Get current searcher (reloads only if segments changed)
    ///
    /// Wait-free read path via ArcSwap::load(). Reload checks are guarded
    /// by an AtomicBool to prevent concurrent reloads.
    pub async fn searcher(&self) -> Result<Arc<Searcher<D>>> {
        // Check if we should check for segment changes
        let should_check = {
            let last = self.last_reload_check.read();
            last.elapsed() >= self.reload_check_interval
        };

        if should_check {
            // Try to acquire the reload guard (non-blocking)
            if self
                .reloading
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // We won the race — do the reload check
                let result = self.do_reload_check().await;
                self.reloading.store(false, Ordering::Release);
                result?;
            }
            // Otherwise another reload is in progress — just return current searcher
        }

        // Wait-free load (no lock contention with reloads)
        Ok(Arc::clone(&self.state.load().searcher))
    }

    /// Actual reload check (called under the `reloading` guard)
    async fn do_reload_check(&self) -> Result<()> {
        *self.last_reload_check.write() = std::time::Instant::now();

        // Get current segment IDs from segment manager
        let new_segment_ids = self.segment_manager.get_segment_ids().await;

        // Check if segments actually changed (wait-free read)
        let segments_changed = {
            let state = self.state.load();
            state.segment_ids != new_segment_ids
        };

        if segments_changed {
            let old_count = self.state.load().segment_ids.len();
            let new_count = new_segment_ids.len();
            log::info!(
                "[index_reload] old_count={} new_count={}",
                old_count,
                new_count
            );
            self.reload_with_segments(new_segment_ids).await?;
        }
        Ok(())
    }

    /// Force reload reader with fresh snapshot.
    ///
    /// Waits for any in-progress reload (from `searcher()`) to finish, then
    /// performs its own reload with the latest segment IDs. This guarantees
    /// the reload actually happens — unlike `searcher()` which silently skips
    /// if another reload is in progress.
    pub async fn reload(&self) -> Result<()> {
        // Wait for any in-progress reload to finish, then acquire the guard.
        // This is critical: a concurrent do_reload_check() may have started
        // before a commit, so its reload won't see the new segments.
        loop {
            if self
                .reloading
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        let new_segment_ids = self.segment_manager.get_segment_ids().await;
        let result = self.reload_with_segments(new_segment_ids).await;
        self.reloading.store(false, Ordering::Release);
        result
    }

    /// Internal reload with specific segment IDs.
    /// Atomic swap via ArcSwap::store (wait-free for readers).
    async fn reload_with_segments(&self, new_segment_ids: Vec<String>) -> Result<()> {
        let new_reader =
            Self::create_reader(&self.schema, &self.segment_manager, self.term_cache_blocks)
                .await?;

        // Atomic swap — readers see old or new state, never a torn read
        self.state.store(Arc::new(SearcherState {
            searcher: Arc::new(new_reader),
            segment_ids: new_segment_ids,
        }));

        Ok(())
    }

    /// Get schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
