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
use super::searcher::SearcherResources;

/// IndexReader - manages Searcher with reload policy
///
/// The IndexReader periodically reloads its Searcher to pick up new segments.
/// Uses SegmentManager as authoritative source for segment state (avoids race conditions).
/// Combined searcher + segment IDs, swapped atomically via ArcSwap (wait-free reads).
struct SearcherState<D: DirectoryWriter + 'static> {
    searcher: Arc<Searcher<D>>,
    segment_ids: Vec<String>,
    publication_id: u64,
}

/// Cancellation-safe ownership of the reload flag. Async reload checks may be
/// dropped at any await point; resetting manually only on normal return leaves
/// every future reload disabled after request cancellation or panic.
struct ReloadGuard<'a>(&'a AtomicBool);

impl Drop for ReloadGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

pub struct IndexReader<D: DirectoryWriter + 'static> {
    /// Schema
    schema: Arc<Schema>,
    /// Segment manager - authoritative source for segments
    segment_manager: Arc<crate::merge::SegmentManager<D>>,
    /// Current searcher + segment IDs (ArcSwap for wait-free reads)
    state: ArcSwap<SearcherState<D>>,
    /// Cache and CPU policy preserved across every searcher reload.
    resources: SearcherResources,
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
        const STANDALONE_STORE_CACHE_BYTES: usize = 32 * 1024 * 1024;
        let resources = SearcherResources::new(
            term_cache_blocks,
            STANDALONE_STORE_CACHE_BYTES,
            crate::default_search_threads(),
            4,
        )?;
        Self::from_segment_manager_with_resources(
            schema,
            segment_manager,
            reload_interval_ms,
            resources,
        )
        .await
    }

    /// Internal constructor used by `Index` to preserve its configured cache
    /// and search CPU policy across reader reloads.
    pub(crate) async fn from_segment_manager_with_resources(
        schema: Arc<Schema>,
        segment_manager: Arc<crate::merge::SegmentManager<D>>,
        reload_interval_ms: u64,
        resources: SearcherResources,
    ) -> Result<Self> {
        // Get initial segment IDs
        let initial_segment_ids = segment_manager.get_segment_ids().await;

        let (reader, publication_id) =
            Self::create_reader(&schema, &segment_manager, resources.clone()).await?;

        Ok(Self {
            schema,
            segment_manager,
            state: ArcSwap::from_pointee(SearcherState {
                searcher: Arc::new(reader),
                segment_ids: initial_segment_ids,
                publication_id,
            }),
            resources,
            last_reload_check: RwLock::new(std::time::Instant::now()),
            reload_check_interval: std::time::Duration::from_millis(reload_interval_ms),
            reloading: AtomicBool::new(false),
        })
    }

    /// Create a new reader with fresh snapshot from segment manager
    ///
    /// Captures segment IDs and their trained vector generation together.
    async fn create_reader(
        schema: &Arc<Schema>,
        segment_manager: &Arc<crate::merge::SegmentManager<D>>,
        resources: SearcherResources,
    ) -> Result<(Searcher<D>, u64)> {
        let snapshot = segment_manager.acquire_snapshot().await;
        let generation = snapshot.published_generation();
        let snapshot_schema = generation
            .as_ref()
            .map(|generation| Arc::clone(&generation.schema))
            .unwrap_or_else(|| Arc::clone(schema));
        let trained = generation
            .as_ref()
            .and_then(|generation| generation.trained_vectors.clone())
            .unwrap_or_else(|| Arc::new(crate::segment::TrainedVectorStructures::default()));
        let publication_id = generation
            .as_ref()
            .map_or(0, |generation| generation.publication_id);

        let searcher = Searcher::from_snapshot(
            segment_manager.directory(),
            snapshot_schema,
            snapshot,
            trained,
            resources,
        )
        .await?;
        Ok((searcher, publication_id))
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
                let _reload_guard = ReloadGuard(&self.reloading);
                // We won the race — do the reload check
                self.do_reload_check().await?;
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
        let publication_id = self.segment_manager.publication_id();
        let generation_changed = {
            let state = self.state.load();
            state.segment_ids != new_segment_ids || state.publication_id != publication_id
        };

        if generation_changed {
            let old_count = self.state.load().segment_ids.len();
            let new_count = new_segment_ids.len();
            log::info!(
                "[index_reload] index={} old_count={} new_count={}",
                self.schema.index_label(),
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
        let _reload_guard = ReloadGuard(&self.reloading);
        let new_segment_ids = self.segment_manager.get_segment_ids().await;

        // Fast path: skip reload if segments haven't changed
        let publication_id = self.segment_manager.publication_id();
        let generation_changed = {
            let state = self.state.load();
            state.segment_ids != new_segment_ids || state.publication_id != publication_id
        };

        if generation_changed {
            self.reload_with_segments(new_segment_ids).await
        } else {
            log::debug!(
                "[reload] index={} segments unchanged, skipping",
                self.schema.index_label()
            );
            Ok(())
        }
    }

    /// Internal reload with specific segment IDs.
    /// Reuses existing segment readers for unchanged segments (avoids re-opening
    /// mmaps, fast fields, sparse indexes, etc.).
    /// Atomic swap via ArcSwap::store (wait-free for readers).
    async fn reload_with_segments(&self, new_segment_ids: Vec<String>) -> Result<()> {
        // Collect existing segment readers for reuse
        let existing_segments: Vec<Arc<crate::segment::SegmentReader>> =
            self.state.load().searcher.segment_readers().to_vec();

        let snapshot = self.segment_manager.acquire_snapshot().await;
        let generation = snapshot.published_generation();
        let schema = generation
            .as_ref()
            .map(|generation| Arc::clone(&generation.schema))
            .unwrap_or_else(|| Arc::clone(&self.schema));
        let trained = generation
            .as_ref()
            .and_then(|generation| generation.trained_vectors.clone())
            .unwrap_or_else(|| Arc::new(crate::segment::TrainedVectorStructures::default()));
        let publication_id = generation
            .as_ref()
            .map_or(0, |generation| generation.publication_id);

        let new_reader = Searcher::from_snapshot_reuse(
            self.segment_manager.directory(),
            schema,
            snapshot,
            trained,
            self.resources.clone(),
            &existing_segments,
        )
        .await?;

        // Atomic swap — readers see old or new state, never a torn read
        self.state.store(Arc::new(SearcherState {
            searcher: Arc::new(new_reader),
            segment_ids: new_segment_ids,
            publication_id,
        }));

        Ok(())
    }

    /// Get schema
    pub fn schema(&self) -> Arc<Schema> {
        self.schema_arc()
    }

    /// Schema of the currently published search generation.
    pub fn schema_arc(&self) -> Arc<Schema> {
        self.state.load().searcher.schema_arc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reload_guard_releases_flag_on_unwind() {
        let reloading = AtomicBool::new(true);
        let result = std::panic::catch_unwind(|| {
            let _guard = ReloadGuard(&reloading);
            panic!("cancel reload");
        });
        assert!(result.is_err());
        assert!(!reloading.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn schema_publication_is_atomic_for_old_and_new_searchers() {
        use crate::directories::RamDirectory;
        use crate::dsl::{DenseVectorConfig, SchemaBuilder, VectorIndexAlter, VectorIndexType};

        let mut builder = SchemaBuilder::default();
        let field = builder.add_dense_vector_field_with_config(
            "embedding",
            true,
            true,
            DenseVectorConfig::ivf_tq(4, Some(2), 1),
        );
        let directory = RamDirectory::new();
        let index = crate::Index::create(
            directory.clone(),
            builder.build(),
            crate::IndexConfig::default(),
        )
        .await
        .unwrap();
        let reader = index.reader().await.unwrap();
        let old_searcher = reader.searcher().await.unwrap();

        let mut target = DenseVectorConfig::ivf_tq(4, Some(2), 1);
        target.index_type = VectorIndexType::Scann;
        target.tree_levels = Some(1);
        target.soar = None;
        let next_schema = Arc::new(
            index
                .schema_arc()
                .with_vector_index_alter(field, VectorIndexAlter::Dense(target))
                .unwrap(),
        );
        let update = index
            .segment_manager()
            .begin_vector_artifact_update()
            .await
            .unwrap();
        index
            .segment_manager()
            .publish_vector_schema_only(&update, next_schema)
            .await
            .unwrap();
        drop(update);

        assert_eq!(
            old_searcher
                .schema()
                .get_field_entry(field)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .index_type,
            VectorIndexType::IvfTq
        );
        reader.reload().await.unwrap();
        let new_searcher = reader.searcher().await.unwrap();
        assert_eq!(
            new_searcher
                .schema()
                .get_field_entry(field)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .index_type,
            VectorIndexType::Scann
        );

        let reopened = crate::Index::open(directory, crate::IndexConfig::default())
            .await
            .unwrap();
        assert_eq!(
            reopened
                .schema_arc()
                .get_field_entry(field)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .index_type,
            VectorIndexType::Scann
        );
    }
}
