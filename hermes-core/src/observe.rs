//! Metrics emission helpers (`metrics` facade, behind the `metrics` feature).
//!
//! Every helper is called at an aggregation point — query end, phase end, or
//! a single IO call — never inside per-block scoring loops, so the hot-path
//! atomics/allocation budget is untouched. With the feature off (or on wasm)
//! everything here compiles to nothing; with the feature on but no recorder
//! installed, the `metrics` macros are ~1ns no-ops.
//!
//! Metric names and semantics are documented in `docs/metrics.md`.

#[cfg(all(feature = "metrics", feature = "native"))]
mod imp {
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
        field: u32,
        secs: f64,
        sbs_scored: usize,
        sbs_total: usize,
        blocks_scored: usize,
        blocks_total: usize,
        docmap_lookups: usize,
    ) {
        let field = field.to_string();
        metrics::histogram!("hermes_bmp_query_duration_seconds", "field" => field.clone())
            .record(secs);
        metrics::counter!("hermes_bmp_superblocks_visited_total", "field" => field.clone())
            .increment(sbs_scored as u64);
        metrics::counter!("hermes_bmp_superblocks_skipped_total", "field" => field.clone())
            .increment(sbs_total.saturating_sub(sbs_scored) as u64);
        metrics::counter!("hermes_bmp_blocks_scored_total", "field" => field.clone())
            .increment(blocks_scored as u64);
        metrics::counter!("hermes_bmp_blocks_skipped_total", "field" => field.clone())
            .increment(blocks_total.saturating_sub(blocks_scored) as u64);
        metrics::histogram!("hermes_bmp_blocks_scored_per_query", "field" => field.clone())
            .record(blocks_scored as f64);
        // Doc-map indirection cost: BMP reorder permutes only the BMP-internal
        // record order (doc ids resolve through a mapping, the rest of the
        // segment is NOT physically reordered) — every scored candidate pays a
        // scattered doc-map lookup.
        metrics::counter!("hermes_bmp_docmap_lookups_total", "field" => field.clone())
            .increment(docmap_lookups as u64);
        metrics::histogram!("hermes_bmp_docmap_lookups_per_query", "field" => field)
            .record(docmap_lookups as f64);
    }

    /// Sparse DAAT MaxScore executor finished one query.
    pub fn maxscore_query(secs: f64, docs_returned: usize) {
        metrics::histogram!("hermes_sparse_maxscore_query_duration_seconds").record(secs);
        metrics::histogram!("hermes_sparse_maxscore_docs_returned").record(docs_returned as f64);
    }

    /// Dense vector L1 candidate generation (ANN or brute force) finished.
    pub fn dense_l1(field: u32, kind: &'static str, secs: f64, candidates: usize) {
        let field = field.to_string();
        metrics::histogram!("hermes_dense_l1_duration_seconds", "field" => field.clone(), "kind" => kind)
            .record(secs);
        metrics::histogram!("hermes_dense_l1_candidates", "field" => field, "kind" => kind)
            .record(candidates as f64);
    }

    /// Dense rerank phase finished (resolve + read + score).
    ///
    /// `resolve_secs` is the doc→flat-index indirection cost: like BMP's doc
    /// map, ANN results carry doc ids that must be mapped back to physical
    /// vector slots because the flat store is NOT reordered.
    pub fn dense_rerank(
        field: u32,
        total_secs: f64,
        resolve_secs: f64,
        read_secs: f64,
        vectors: usize,
    ) {
        let field = field.to_string();
        metrics::histogram!("hermes_dense_rerank_duration_seconds", "field" => field.clone())
            .record(total_secs);
        metrics::histogram!("hermes_dense_rerank_resolve_duration_seconds", "field" => field.clone())
            .record(resolve_secs);
        metrics::histogram!("hermes_dense_rerank_read_duration_seconds", "field" => field.clone())
            .record(read_secs);
        metrics::histogram!("hermes_dense_rerank_vectors", "field" => field).record(vectors as f64);
    }

    /// One Directory-layer read completed.
    pub fn directory_read(op: &'static str, secs: f64, bytes: usize) {
        metrics::histogram!("hermes_directory_read_duration_seconds", "op" => op).record(secs);
        metrics::histogram!("hermes_directory_read_bytes", "op" => op).record(bytes as f64);
    }

    /// One document store fetch completed.
    pub fn store_get(secs: f64) {
        metrics::histogram!("hermes_store_get_duration_seconds").record(secs);
    }

    /// A cold (page-cache-dropping) writer finished one file.
    pub fn cold_write(bytes: usize) {
        metrics::counter!("hermes_cold_write_bytes_total").increment(bytes as u64);
    }

    /// One reorder granularity decision was made (Auto or explicit).
    pub fn reorder_granularity(field: u32, granularity: &'static str, coherence: f32) {
        let field = field.to_string();
        metrics::counter!(
            "hermes_reorder_granularity_total",
            "field" => field.clone(),
            "granularity" => granularity,
        )
        .increment(1);
        metrics::histogram!("hermes_reorder_coherence", "field" => field).record(coherence as f64);
    }
}

#[cfg(not(all(feature = "metrics", feature = "native")))]
mod imp {
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
    pub fn bmp_query(_: u32, _: f64, _: usize, _: usize, _: usize, _: usize, _: usize) {}
    #[inline(always)]
    pub fn maxscore_query(_: f64, _: usize) {}
    #[inline(always)]
    pub fn dense_l1(_: u32, _: &'static str, _: f64, _: usize) {}
    #[inline(always)]
    pub fn dense_rerank(_: u32, _: f64, _: f64, _: f64, _: usize) {}
    #[inline(always)]
    pub fn directory_read(_: &'static str, _: f64, _: usize) {}
    #[inline(always)]
    pub fn store_get(_: f64) {}
    #[inline(always)]
    pub fn cold_write(_: usize) {}
    #[inline(always)]
    pub fn reorder_granularity(_: u32, _: &'static str, _: f32) {}
}

pub(crate) use imp::*;
