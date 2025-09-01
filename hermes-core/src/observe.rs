//! Metrics emission helpers (`metrics` facade, behind the `metrics` feature).
//!
//! Every helper is called at an aggregation point — query end, phase end, or
//! a single IO call — never inside per-block scoring loops, so the hot-path
//! atomics/allocation budget is untouched. With the feature off (or on wasm)
//! everything here compiles to nothing; with the feature on but no recorder
//! installed, the `metrics` macros are ~1ns no-ops.
//!
//! Metric names and semantics are documented in `docs/metrics.md`.

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(all(feature = "metrics", feature = "native")), allow(dead_code))]
pub(crate) struct BmpQueryPhases {
    pub prepare_secs: f64,
    pub d_grid_secs: f64,
    pub prefetch_secs: f64,
    pub block_score_secs: f64,
    pub docmap_secs: f64,
    pub prefetched_bytes: usize,
    pub prefetch_ranges: usize,
}

/// Wall-clock timer for diagnostics that must also compile on platforms where
/// `std::time::Instant` is unavailable at runtime (notably wasm32).
///
/// Native builds retain real timings; non-native builds fold the timer and its
/// readings away. Metrics-only call sites should use [`Timer`] instead so they
/// remain free when metrics are disabled.
pub(crate) struct WallTimer {
    #[cfg(feature = "native")]
    start: std::time::Instant,
}

impl WallTimer {
    #[inline]
    pub fn start() -> Self {
        Self {
            #[cfg(feature = "native")]
            start: std::time::Instant::now(),
        }
    }

    #[inline]
    pub fn secs(&self) -> f64 {
        #[cfg(feature = "native")]
        {
            self.start.elapsed().as_secs_f64()
        }
        #[cfg(not(feature = "native"))]
        {
            0.0
        }
    }
}

#[cfg(all(feature = "metrics", feature = "native"))]
mod imp {
    use super::BmpQueryPhases;
    use std::sync::Arc;

    #[inline]
    fn shared_label(value: &str) -> metrics::SharedString {
        Arc::<str>::from(value).into()
    }

    /// Wall-clock timer for phase measurements.
    pub struct Timer(std::time::Instant);

    impl Timer {
        #[inline]
        pub fn start() -> Self {
            Timer(std::time::Instant::now())
        }

        #[inline]
        pub fn secs(&self) -> f64 {
            self.0.elapsed().as_secs_f64()
        }
    }

    /// BMP executor finished one query on one segment/field.
    #[allow(clippy::too_many_arguments)]
    pub fn bmp_query(
        index: &str,
        field: &str,
        secs: f64,
        phases: BmpQueryPhases,
        sbs_scored: usize,
        sbs_total: usize,
        blocks_scored: usize,
        blocks_total: usize,
        docmap_lookups: usize,
    ) {
        let index = shared_label(index);
        let field = shared_label(field);
        metrics::histogram!("hermes_bmp_query_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(secs);
        metrics::histogram!("hermes_bmp_query_prepare_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(phases.prepare_secs);
        metrics::histogram!("hermes_bmp_d_grid_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(phases.d_grid_secs);
        metrics::histogram!("hermes_bmp_prefetch_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(phases.prefetch_secs);
        metrics::histogram!("hermes_bmp_block_score_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(phases.block_score_secs);
        metrics::histogram!("hermes_bmp_docmap_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(phases.docmap_secs);
        metrics::counter!("hermes_bmp_prefetched_bytes_total", "index" => index.clone(), "field" => field.clone())
            .increment(phases.prefetched_bytes as u64);
        metrics::counter!("hermes_bmp_prefetch_ranges_total", "index" => index.clone(), "field" => field.clone())
            .increment(phases.prefetch_ranges as u64);
        metrics::counter!("hermes_bmp_superblocks_visited_total", "index" => index.clone(), "field" => field.clone())
            .increment(sbs_scored as u64);
        metrics::counter!("hermes_bmp_superblocks_skipped_total", "index" => index.clone(), "field" => field.clone())
            .increment(sbs_total.saturating_sub(sbs_scored) as u64);
        metrics::counter!("hermes_bmp_blocks_scored_total", "index" => index.clone(), "field" => field.clone())
            .increment(blocks_scored as u64);
        metrics::counter!("hermes_bmp_blocks_skipped_total", "index" => index.clone(), "field" => field.clone())
            .increment(blocks_total.saturating_sub(blocks_scored) as u64);
        metrics::histogram!("hermes_bmp_blocks_scored_per_query", "index" => index.clone(), "field" => field.clone())
            .record(blocks_scored as f64);
        // Doc-map indirection cost: BMP reorder permutes only the BMP-internal
        // record order (doc ids resolve through a mapping, the rest of the
        // segment is NOT physically reordered) — every scored candidate pays a
        // scattered doc-map lookup.
        metrics::counter!("hermes_bmp_docmap_lookups_total", "index" => index.clone(), "field" => field.clone())
            .increment(docmap_lookups as u64);
        metrics::histogram!("hermes_bmp_docmap_lookups_per_query", "index" => index, "field" => field)
            .record(docmap_lookups as f64);
    }

    /// Query-global LSP planning, which runs outside every segment executor.
    #[allow(clippy::too_many_arguments)]
    pub fn bmp_lsp(
        index: &str,
        field: &str,
        total_secs: f64,
        prepare_secs: f64,
        hierarchy_scan_secs: f64,
        select_secs: f64,
        superblocks: usize,
        gamma: usize,
        coarse_groups: usize,
        coarse_groups_expanded: usize,
        superblocks_evaluated: usize,
    ) {
        let index = shared_label(index);
        let field = shared_label(field);
        metrics::histogram!("hermes_bmp_lsp_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(total_secs);
        metrics::histogram!("hermes_bmp_lsp_prepare_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(prepare_secs);
        metrics::histogram!("hermes_bmp_lsp_h_scan_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(hierarchy_scan_secs);
        metrics::histogram!("hermes_bmp_lsp_select_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(select_secs);
        metrics::histogram!("hermes_bmp_lsp_superblocks", "index" => index.clone(), "field" => field.clone())
            .record(superblocks as f64);
        metrics::histogram!("hermes_bmp_lsp_gamma", "index" => index.clone(), "field" => field.clone())
            .record(gamma as f64);
        metrics::histogram!("hermes_bmp_lsp_coarse_groups", "index" => index.clone(), "field" => field.clone())
            .record(coarse_groups as f64);
        metrics::histogram!("hermes_bmp_lsp_coarse_groups_expanded", "index" => index.clone(), "field" => field.clone())
            .record(coarse_groups_expanded as f64);
        metrics::histogram!("hermes_bmp_lsp_superblocks_evaluated", "index" => index, "field" => field)
            .record(superblocks_evaluated as f64);
    }

    /// Sparse DAAT MaxScore executor finished one query.
    pub fn maxscore_query(index: &str, field: &str, secs: f64, docs_returned: usize) {
        let index = shared_label(index);
        let field = shared_label(field);
        metrics::histogram!("hermes_sparse_maxscore_query_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(secs);
        metrics::histogram!("hermes_sparse_maxscore_docs_returned", "index" => index, "field" => field)
            .record(docs_returned as f64);
    }

    /// Dense vector L1 candidate generation (ANN or brute force) finished.
    pub fn dense_l1(index: &str, field: &str, kind: &'static str, secs: f64, candidates: usize) {
        let index = shared_label(index);
        let field = shared_label(field);
        metrics::histogram!("hermes_dense_l1_duration_seconds", "index" => index.clone(), "field" => field.clone(), "kind" => kind)
            .record(secs);
        metrics::histogram!("hermes_dense_l1_candidates", "index" => index, "field" => field, "kind" => kind)
            .record(candidates as f64);
    }

    /// Dense rerank phase finished (resolve + read + score).
    ///
    /// `resolve_secs` is the doc→flat-index indirection cost: like BMP's doc
    /// map, ANN results carry doc ids that must be mapped back to physical
    /// vector slots because the flat store is NOT reordered.
    pub fn dense_rerank(
        index: &str,
        field: &str,
        total_secs: f64,
        resolve_secs: f64,
        read_secs: f64,
        vectors: usize,
    ) {
        let index = shared_label(index);
        let field = shared_label(field);
        metrics::histogram!("hermes_dense_rerank_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(total_secs);
        metrics::histogram!("hermes_dense_rerank_resolve_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(resolve_secs);
        metrics::histogram!("hermes_dense_rerank_read_duration_seconds", "index" => index.clone(), "field" => field.clone())
            .record(read_secs);
        metrics::histogram!("hermes_dense_rerank_vectors", "index" => index, "field" => field)
            .record(vectors as f64);
    }

    /// One Directory-layer read completed.
    pub fn directory_read(index: &str, op: &'static str, secs: f64, bytes: usize) {
        let index = shared_label(index);
        metrics::histogram!("hermes_directory_read_duration_seconds", "index" => index.clone(), "op" => op)
            .record(secs);
        metrics::histogram!("hermes_directory_read_bytes", "index" => index, "op" => op)
            .record(bytes as f64);
    }

    /// One document store fetch completed.
    pub fn store_get(index: &str, secs: f64) {
        metrics::histogram!("hermes_store_get_duration_seconds", "index" => shared_label(index))
            .record(secs);
    }

    /// A cold (page-cache-dropping) writer finished one file.
    pub fn cold_write(index: &str, bytes: usize) {
        metrics::counter!("hermes_cold_write_bytes_total", "index" => shared_label(index))
            .increment(bytes as u64);
    }

    /// One reorder granularity decision was made (Auto or explicit).
    pub fn reorder_granularity(index: &str, field: &str, granularity: &'static str) {
        metrics::counter!(
            "hermes_reorder_granularity_total",
            "index" => shared_label(index),
            "field" => shared_label(field),
            "granularity" => granularity,
        )
        .increment(1);
    }

    /// Coherence measured for one `Auto` granularity decision (explicit
    /// granularity skips the scan and emits nothing here).
    pub fn reorder_coherence(index: &str, field: &str, coherence: f32, coherence_norm: f32) {
        let index = shared_label(index);
        let field = shared_label(field);
        metrics::histogram!("hermes_reorder_coherence", "index" => index.clone(), "field" => field.clone())
            .record(coherence as f64);
        metrics::histogram!("hermes_reorder_coherence_norm", "index" => index, "field" => field)
            .record(coherence_norm as f64);
    }

    /// Structural ANN health gauges, refreshed at every segment open.
    ///
    /// Gauges rather than log-only: leaf collapse and fragmentation build up
    /// over weeks, which is dashboard territory, not log-scraping territory.
    pub fn ann_health(
        index: &str,
        field: u32,
        imbalance: f64,
        fragmentation: f64,
        largest_leaf_share: f64,
    ) {
        let index = shared_label(index);
        let field = field.to_string();
        metrics::gauge!("hermes_ann_imbalance", "index" => index.clone(), "field" => field.clone())
            .set(imbalance);
        metrics::gauge!("hermes_ann_fragmentation", "index" => index.clone(), "field" => field.clone())
            .set(fragmentation);
        metrics::gauge!("hermes_ann_largest_leaf_share", "index" => index, "field" => field)
            .set(largest_leaf_share);
    }

    pub fn reorder_bp_started(index: &str, field: &str, entity_kind: &'static str) {
        metrics::gauge!("hermes_reorder_bp_active_passes", "index" => shared_label(index), "field" => shared_label(field), "entity_kind" => entity_kind)
            .increment(1.0);
    }

    pub fn reorder_bp_finished(index: &str, field: &str, entity_kind: &'static str) {
        metrics::gauge!("hermes_reorder_bp_active_passes", "index" => shared_label(index), "field" => shared_label(field), "entity_kind" => entity_kind)
            .decrement(1.0);
    }

    /// Final aggregate for one BP graph pass.
    #[allow(clippy::too_many_arguments)]
    pub fn reorder_bp_pass(
        index: &str,
        field: &str,
        entity_kind: &'static str,
        stop_reason: &'static str,
        secs: f64,
        entities: usize,
        postings: u64,
        partitions: u64,
        iterations: u64,
        entity_passes: u64,
        swaps: u64,
        converged: bool,
    ) {
        let index = shared_label(index);
        let field = shared_label(field);
        let converged = if converged { "true" } else { "false" };
        metrics::counter!("hermes_reorder_bp_passes_total", "index" => index.clone(), "field" => field.clone(), "entity_kind" => entity_kind, "stop_reason" => stop_reason, "converged" => converged)
            .increment(1);
        metrics::histogram!("hermes_reorder_bp_duration_seconds", "index" => index.clone(), "field" => field.clone(), "entity_kind" => entity_kind)
            .record(secs);
        metrics::histogram!("hermes_reorder_bp_entities", "index" => index.clone(), "field" => field.clone(), "entity_kind" => entity_kind)
            .record(entities as f64);
        metrics::histogram!("hermes_reorder_bp_postings", "index" => index.clone(), "field" => field.clone(), "entity_kind" => entity_kind)
            .record(postings as f64);
        metrics::histogram!("hermes_reorder_bp_partitions", "index" => index.clone(), "field" => field.clone(), "entity_kind" => entity_kind)
            .record(partitions as f64);
        metrics::histogram!("hermes_reorder_bp_iterations_per_pass", "index" => index.clone(), "field" => field.clone(), "entity_kind" => entity_kind)
            .record(iterations as f64);
        metrics::histogram!("hermes_reorder_bp_entity_passes_per_pass", "index" => index.clone(), "field" => field.clone(), "entity_kind" => entity_kind)
            .record(entity_passes as f64);
        metrics::histogram!("hermes_reorder_bp_swaps", "index" => index, "field" => field, "entity_kind" => entity_kind)
            .record(swaps as f64);
    }
}

#[cfg(not(all(feature = "metrics", feature = "native")))]
mod imp {
    use super::BmpQueryPhases;

    /// No-op timer — everything folds away at compile time.
    pub struct Timer;

    impl Timer {
        #[inline(always)]
        pub fn start() -> Self {
            Timer
        }

        #[inline(always)]
        pub fn secs(&self) -> f64 {
            0.0
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn bmp_query(
        _: &str,
        _: &str,
        _: f64,
        _: BmpQueryPhases,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) {
    }
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn bmp_lsp(
        _: &str,
        _: &str,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) {
    }
    #[inline(always)]
    pub fn maxscore_query(_: &str, _: &str, _: f64, _: usize) {}
    #[inline(always)]
    pub fn dense_l1(_: &str, _: &str, _: &'static str, _: f64, _: usize) {}
    #[inline(always)]
    pub fn ann_health(_: &str, _: u32, _: f64, _: f64, _: f64) {}
    #[inline(always)]
    pub fn dense_rerank(_: &str, _: &str, _: f64, _: f64, _: f64, _: usize) {}
    #[inline(always)]
    pub fn directory_read(_: &str, _: &'static str, _: f64, _: usize) {}
    #[inline(always)]
    pub fn store_get(_: &str, _: f64) {}
    // Caller is native-only directory code — dead on wasm.
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn cold_write(_: &str, _: usize) {}
    // Callers live in native-only modules (segment::reorder) — dead on wasm.
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn reorder_granularity(_: &str, _: &str, _: &'static str) {}
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn reorder_coherence(_: &str, _: &str, _: f32, _: f32) {}
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn reorder_bp_started(_: &str, _: &str, _: &'static str) {}
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn reorder_bp_finished(_: &str, _: &str, _: &'static str) {}
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn reorder_bp_pass(
        _: &str,
        _: &str,
        _: &'static str,
        _: &'static str,
        _: f64,
        _: usize,
        _: u64,
        _: u64,
        _: u64,
        _: u64,
        _: u64,
        _: bool,
    ) {
    }
}

pub(crate) use imp::*;

// ---- search plumbing (added by plumbing perf pass) ----

#[cfg(all(feature = "metrics", feature = "native"))]
mod plumbing {
    use std::sync::Arc;

    /// Slice cache hits, flushed in batches from an in-process counter (the
    /// hit path itself is one atomic increment, no label allocation).
    pub fn slice_cache_hits(index: &Arc<str>, hits: u64, last_bytes: usize) {
        metrics::counter!("hermes_slice_cache_hits_total", "index" => Arc::clone(index))
            .increment(hits);
        metrics::histogram!("hermes_slice_cache_hit_bytes", "index" => Arc::clone(index))
            .record(last_bytes as f64);
    }

    /// One slice cache miss (the range went to the inner directory).
    pub fn slice_cache_miss(index: &Arc<str>, bytes: usize) {
        metrics::counter!("hermes_slice_cache_misses_total", "index" => Arc::clone(index))
            .increment(1);
        metrics::histogram!("hermes_slice_cache_miss_bytes", "index" => Arc::clone(index))
            .record(bytes as f64);
    }

    /// One eviction pass of the slice cache finished.
    pub fn slice_cache_evicted(index: &Arc<str>, slices: u64, bytes: usize) {
        metrics::counter!("hermes_slice_cache_evicted_slices_total", "index" => Arc::clone(index))
            .increment(slices);
        metrics::counter!("hermes_slice_cache_evicted_bytes_total", "index" => Arc::clone(index))
            .increment(bytes as u64);
    }

    /// L2 rerank candidates dropped because their segment is gone or has no
    /// stored vectors for the field.
    pub fn rerank_candidates_skipped(index: &str, kind: &'static str, count: u64) {
        let index: metrics::SharedString = Arc::<str>::from(index).into();
        metrics::counter!("hermes_rerank_candidates_skipped_total", "index" => index, "kind" => kind)
            .increment(count);
    }
}

#[cfg(not(all(feature = "metrics", feature = "native")))]
mod plumbing {
    #[inline(always)]
    pub fn slice_cache_hits(_: &std::sync::Arc<str>, _: u64, _: usize) {}
    #[inline(always)]
    pub fn slice_cache_miss(_: &std::sync::Arc<str>, _: usize) {}
    #[inline(always)]
    pub fn slice_cache_evicted(_: &std::sync::Arc<str>, _: u64, _: usize) {}
    // Caller is the native-only L2 reranker — dead on wasm.
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn rerank_candidates_skipped(_: &str, _: &'static str, _: u64) {}
}

pub(crate) use plumbing::*;

// ---- dense ANN (added by dense perf pass) ----

/// Per-query diagnostics from one segment's dense ANN scan. Every field is a
/// fallback or a regime decision that used to be invisible: pruned blocks
/// only reached `log::debug!`, non-finite scores were dropped silently, the
/// serial-vs-Rayon choice left no trace, and combined-document scans
/// materialize every probed posting without saying how many.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DenseAnnScanStats {
    /// Blocks (or rows) whose scores were computed.
    pub scored_blocks: usize,
    /// Blocks skipped by the scale upper bound.
    pub pruned_blocks: usize,
    /// Postings in the probed leaves (what a combined-document scan buffers).
    pub posting_count: usize,
    /// Candidates dropped because their score was NaN/±inf.
    pub non_finite_dropped: usize,
    /// Whether the scan fanned out across Rayon workers.
    pub parallel: bool,
}

impl DenseAnnScanStats {
    /// Metric label for the serial-vs-Rayon decision.
    #[cfg_attr(not(all(feature = "metrics", feature = "native")), allow(dead_code))]
    #[inline]
    pub(crate) fn regime(&self) -> &'static str {
        if self.parallel { "parallel" } else { "serial" }
    }
}

#[cfg(all(feature = "metrics", feature = "native"))]
mod dense_ann_imp {
    use super::DenseAnnScanStats;
    use std::sync::Arc;

    /// One dense ANN segment scan finished (IVF-TQ, TQ flat, binary IVF).
    pub fn dense_ann_scan(index: &str, field: &str, kind: &'static str, stats: DenseAnnScanStats) {
        let index: metrics::SharedString = Arc::<str>::from(index).into();
        let field: metrics::SharedString = Arc::<str>::from(field).into();
        let regime = stats.regime();
        metrics::counter!("hermes_dense_ann_scans_total", "index" => index.clone(), "field" => field.clone(), "kind" => kind, "regime" => regime)
            .increment(1);
        metrics::histogram!("hermes_dense_ann_blocks_scored", "index" => index.clone(), "field" => field.clone(), "kind" => kind)
            .record(stats.scored_blocks as f64);
        metrics::histogram!("hermes_dense_ann_blocks_pruned", "index" => index.clone(), "field" => field.clone(), "kind" => kind)
            .record(stats.pruned_blocks as f64);
        metrics::histogram!("hermes_dense_ann_postings_probed", "index" => index.clone(), "field" => field.clone(), "kind" => kind)
            .record(stats.posting_count as f64);
        metrics::counter!("hermes_dense_ann_non_finite_dropped_total", "index" => index, "field" => field, "kind" => kind)
            .increment(stats.non_finite_dropped as u64);
    }
}

#[cfg(not(all(feature = "metrics", feature = "native")))]
mod dense_ann_imp {
    use super::DenseAnnScanStats;

    #[inline(always)]
    pub fn dense_ann_scan(_: &str, _: &str, _: &'static str, _: DenseAnnScanStats) {}
}

pub(crate) use dense_ann_imp::dense_ann_scan;

/// Rate-limited warning for dropped non-finite dense scores: a degenerate
/// stored vector (NaN/inf payload) must be visible in logs without one line
/// per query. Logs the 1st, 2nd, 4th, 8th, ... occurrence per process.
pub(crate) fn warn_non_finite_dense_scores(
    index: &str,
    field: &str,
    kind: &'static str,
    dropped: usize,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static OCCURRENCES: AtomicU64 = AtomicU64::new(0);
    if dropped == 0 {
        return;
    }
    let occurrence = OCCURRENCES.fetch_add(1, Ordering::Relaxed) + 1;
    if occurrence.is_power_of_two() {
        log::warn!(
            "[dense_ann] index={index} field={field} kind={kind}: dropped {dropped} candidates \
             with non-finite scores (occurrence #{occurrence}; a stored vector is likely \
             NaN/inf — see hermes_dense_ann_non_finite_dropped_total)"
        );
    }
}

// ---- integrity and degenerate-input counters (merge pass) ----

#[cfg(all(feature = "metrics", feature = "native"))]
mod integrity_imp {
    use std::sync::Arc;

    /// One BMP query hit at least one integrity fault; per-kind counts come
    /// from `BmpExecutionStats`.
    pub fn bmp_integrity_faults(
        index: &str,
        field: &str,
        corrupt_blocks: u64,
        corrupt_terms: u64,
        dropped_postings: u64,
        invalid_docmap_entries: u64,
    ) {
        let index: metrics::SharedString = Arc::<str>::from(index).into();
        let field: metrics::SharedString = Arc::<str>::from(field).into();
        metrics::counter!("hermes_bmp_integrity_fault_queries_total", "index" => index.clone(), "field" => field.clone())
            .increment(1);
        metrics::counter!("hermes_bmp_corrupt_blocks_total", "index" => index.clone(), "field" => field.clone())
            .increment(corrupt_blocks);
        metrics::counter!("hermes_bmp_corrupt_terms_total", "index" => index.clone(), "field" => field.clone())
            .increment(corrupt_terms);
        metrics::counter!("hermes_bmp_dropped_postings_total", "index" => index.clone(), "field" => field.clone())
            .increment(dropped_postings);
        metrics::counter!("hermes_bmp_invalid_docmap_entries_total", "index" => index, "field" => field)
            .increment(invalid_docmap_entries);
    }

    /// Non-finite leaf scores dropped before combining multi-valued ANN
    /// candidates (no index/field in scope at that layer).
    pub fn ann_non_finite_scores_dropped(dropped: u64) {
        metrics::counter!("hermes_ann_non_finite_scores_dropped_total").increment(dropped);
    }

    /// A FastScan lookup table collapsed to a constant (all-zero AH query):
    /// every leaf row scores as its centroid dot.
    pub fn scann_degenerate_fast_scan_query() {
        metrics::counter!("hermes_scann_fast_scan_degenerate_queries_total").increment(1);
    }
}

#[cfg(not(all(feature = "metrics", feature = "native")))]
mod integrity_imp {
    #[inline(always)]
    pub fn bmp_integrity_faults(_: &str, _: &str, _: u64, _: u64, _: u64, _: u64) {}
    // Callers live in the native-only ANN reader and ScaNN engine.
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn ann_non_finite_scores_dropped(_: u64) {}
    #[inline(always)]
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub fn scann_degenerate_fast_scan_query() {}
}

pub(crate) use integrity_imp::*;
