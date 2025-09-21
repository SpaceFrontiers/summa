//! Opt-in work accounting for single-task query diagnosis.
//!
//! Requires `query-diagnostics`. Counts describe work, not production latency.
//! Synchronous Searcher segment workers inherit capture explicitly. Other spawned tasks
//! do not inherit it. Async scopes
//! are installed only while polling; nested captures exclude the nested work.

use std::cell::RefCell;
use std::future::{Future, poll_fn};

/// Fixed-size work counters. Repeated work is counted repeatedly, not deduplicated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct QueryWork {
    /// Seismic clusters visited, including repeated merged runs.
    pub seismic_clusters: u64,
    /// Candidate documents scored after eligibility and complete ordinal gathering.
    pub seismic_documents: u64,
    /// Forward vector rows scored, including exact backfill and winner hydration.
    pub seismic_forward_rows: u64,
    /// Encoded forward-vector bytes consumed by scoring.
    pub seismic_forward_bytes: u64,
    /// Selective or underfilled filtered queries resolved from forward values.
    pub seismic_filter_scans: u64,
    /// Nomination executions stopped at a finite expansion budget.
    pub seismic_budget_truncations: u64,
    /// Searcher segment executions captured, including failed executions.
    pub segment_runs: u64,
    /// Instrumented elapsed nanoseconds inside segment workers (summed across workers).
    /// This is diagnostic wall time, not production latency or CPU time.
    pub segment_elapsed_ns: u64,
    /// Completed shared search-pool installs observed on the capturing caller.
    pub search_pool_installs: u64,
    /// Wall time from submitting an install to entering its closure.
    pub search_pool_queue_ns: u64,
    /// Wall time from its closure finishing until the caller resumes.
    pub search_pool_return_ns: u64,
    /// Document-ID blocks decoded, including repeat decodes.
    pub doc_blocks: u64,
    /// Document IDs decoded, including repeat decodes.
    pub doc_values: u64,
    /// Encoded gap bytes consumed; excludes headers and directories.
    pub doc_payload_bytes: u64,
    /// Frequency blocks decoded, including repeat decodes.
    pub tf_blocks: u64,
    /// Frequencies decoded.
    pub tf_values: u64,
    /// Encoded frequency bytes consumed.
    pub tf_payload_bytes: u64,
    /// Position blocks decoded, including repeat decodes.
    pub position_blocks: u64,
    /// Position deltas decoded.
    pub position_values: u64,
    /// Encoded position bytes consumed, excluding headers.
    pub position_payload_bytes: u64,
    /// Posting iterator and typed term-cursor seek requests.
    pub posting_seeks: u64,
    /// Term-position requests, including cache hits.
    pub position_reads: u64,
    /// Sum of requested position counts.
    pub positions_requested: u64,
    /// Canonical BM25 evaluations in ordinary term scorers/cursors; excludes bounds
    /// and point reranking.
    pub exact_score_units: u64,
    /// Lookup BM25 evaluations in ordinary term scorers/cursors; excludes bounds.
    pub lookup_score_units: u64,
    /// Phrase-document score evaluations; excludes bounds.
    pub phrase_score_units: u64,
    /// Term score runs, including window runs; excludes scalar scoring.
    pub score_batches: u64,
    /// Query-local 256-entry norm tables constructed.
    pub norm_tables: u64,
    /// Typed text cursor block-bound requests, including memo hits.
    pub block_bound_calls: u64,
    /// Typed text cursor group-bound requests.
    pub group_bound_calls: u64,
    /// Phrase candidate upper-bound requests.
    pub phrase_bound_calls: u64,
    /// Actual phrase-position checks; excludes cached confirmations.
    pub phrase_confirmations: u64,
    /// Aligned conjunction candidates before final collection.
    pub conjunction_candidates: u64,
    /// MaxScore text windows visited.
    pub executor_windows: u64,
    /// MaxScore text windows skipped by score bounds.
    pub executor_windows_skipped: u64,
    /// MaxScore L1 groups skipped by score bounds.
    pub executor_groups_skipped: u64,
    /// Window candidates surviving to full scoring.
    pub executor_candidates: u64,
    /// Window documents reported as entering the heap.
    pub executor_heap_admissions: u64,
    /// Successful MaxScore heap pushes/replacements, including conjunctions.
    /// Excludes the outer segment/result collectors.
    pub maxscore_heap_updates: u64,
    /// Single-term MaxScore blocks scored.
    pub single_blocks_scored: u64,
    /// Single-term MaxScore blocks skipped by score bounds.
    pub single_blocks_skipped: u64,
    /// Posting-list envelopes opened.
    pub postings_opened: u64,
    /// Position-stream envelopes opened.
    pub positions_opened: u64,
}

impl QueryWork {
    fn merge(&mut self, other: Self) {
        self.search_pool_installs = self
            .search_pool_installs
            .saturating_add(other.search_pool_installs);
        self.search_pool_queue_ns = self
            .search_pool_queue_ns
            .saturating_add(other.search_pool_queue_ns);
        self.search_pool_return_ns = self
            .search_pool_return_ns
            .saturating_add(other.search_pool_return_ns);
        self.maxscore_heap_updates = self
            .maxscore_heap_updates
            .saturating_add(other.maxscore_heap_updates);
        self.seismic_filter_scans = self
            .seismic_filter_scans
            .saturating_add(other.seismic_filter_scans);
        self.seismic_clusters = self.seismic_clusters.saturating_add(other.seismic_clusters);
        self.seismic_documents = self
            .seismic_documents
            .saturating_add(other.seismic_documents);
        self.seismic_forward_rows = self
            .seismic_forward_rows
            .saturating_add(other.seismic_forward_rows);
        self.seismic_forward_bytes = self
            .seismic_forward_bytes
            .saturating_add(other.seismic_forward_bytes);
        self.seismic_budget_truncations = self
            .seismic_budget_truncations
            .saturating_add(other.seismic_budget_truncations);
        self.segment_runs = self.segment_runs.saturating_add(other.segment_runs);
        self.segment_elapsed_ns = self
            .segment_elapsed_ns
            .saturating_add(other.segment_elapsed_ns);
        self.doc_blocks = self.doc_blocks.saturating_add(other.doc_blocks);
        self.doc_values = self.doc_values.saturating_add(other.doc_values);
        self.doc_payload_bytes = self
            .doc_payload_bytes
            .saturating_add(other.doc_payload_bytes);
        self.tf_blocks = self.tf_blocks.saturating_add(other.tf_blocks);
        self.tf_values = self.tf_values.saturating_add(other.tf_values);
        self.tf_payload_bytes = self.tf_payload_bytes.saturating_add(other.tf_payload_bytes);
        self.position_blocks = self.position_blocks.saturating_add(other.position_blocks);
        self.position_values = self.position_values.saturating_add(other.position_values);
        self.position_payload_bytes = self
            .position_payload_bytes
            .saturating_add(other.position_payload_bytes);
        self.posting_seeks = self.posting_seeks.saturating_add(other.posting_seeks);
        self.position_reads = self.position_reads.saturating_add(other.position_reads);
        self.positions_requested = self
            .positions_requested
            .saturating_add(other.positions_requested);
        self.exact_score_units = self
            .exact_score_units
            .saturating_add(other.exact_score_units);
        self.lookup_score_units = self
            .lookup_score_units
            .saturating_add(other.lookup_score_units);
        self.phrase_score_units = self
            .phrase_score_units
            .saturating_add(other.phrase_score_units);
        self.score_batches = self.score_batches.saturating_add(other.score_batches);
        self.norm_tables = self.norm_tables.saturating_add(other.norm_tables);
        self.block_bound_calls = self
            .block_bound_calls
            .saturating_add(other.block_bound_calls);
        self.group_bound_calls = self
            .group_bound_calls
            .saturating_add(other.group_bound_calls);
        self.phrase_bound_calls = self
            .phrase_bound_calls
            .saturating_add(other.phrase_bound_calls);
        self.phrase_confirmations = self
            .phrase_confirmations
            .saturating_add(other.phrase_confirmations);
        self.conjunction_candidates = self
            .conjunction_candidates
            .saturating_add(other.conjunction_candidates);
        self.executor_windows = self.executor_windows.saturating_add(other.executor_windows);
        self.executor_windows_skipped = self
            .executor_windows_skipped
            .saturating_add(other.executor_windows_skipped);
        self.executor_groups_skipped = self
            .executor_groups_skipped
            .saturating_add(other.executor_groups_skipped);
        self.executor_candidates = self
            .executor_candidates
            .saturating_add(other.executor_candidates);
        self.executor_heap_admissions = self
            .executor_heap_admissions
            .saturating_add(other.executor_heap_admissions);
        self.single_blocks_scored = self
            .single_blocks_scored
            .saturating_add(other.single_blocks_scored);
        self.single_blocks_skipped = self
            .single_blocks_skipped
            .saturating_add(other.single_blocks_skipped);
        self.postings_opened = self.postings_opened.saturating_add(other.postings_opened);
        self.positions_opened = self.positions_opened.saturating_add(other.positions_opened);
    }
}

thread_local! {
    static CONTEXT: RefCell<Option<WorkContext>> = const { RefCell::new(None) };
    static ACTIVE: RefCell<Option<QueryWork>> = const { RefCell::new(None) };
}

#[inline]
pub(crate) fn update(f: impl FnOnce(&mut QueryWork)) {
    ACTIVE.with_borrow_mut(|active| {
        if let Some(work) = active {
            f(work);
        }
    });
}

#[derive(Clone, Default)]
pub(crate) struct WorkContext(std::sync::Arc<parking_lot::Mutex<QueryWork>>);
#[cfg(feature = "sync")]
impl WorkContext {
    pub(crate) fn current() -> Option<Self> {
        CONTEXT.with_borrow(Clone::clone)
    }

    pub(crate) fn enter(&self) -> WorkerScope {
        WorkerScope {
            started: std::time::Instant::now(),
            scope: Some(Scope::enter(QueryWork::default(), self)),
            context: self.clone(),
        }
    }
}

// One merge per completed segment, never a lock/atomic per decoded block.
#[cfg(feature = "sync")]
pub(crate) struct WorkerScope {
    started: std::time::Instant,
    scope: Option<Scope>,
    context: WorkContext,
}
#[cfg(feature = "sync")]
impl Drop for WorkerScope {
    fn drop(&mut self) {
        let mut work = self.scope.take().unwrap().finish();
        work.segment_runs += 1;
        work.segment_elapsed_ns = work
            .segment_elapsed_ns
            .saturating_add(self.started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
        self.context.0.lock().merge(work);
    }
}

struct Scope {
    previous: Option<QueryWork>,
    context: Option<WorkContext>,
}
impl Scope {
    fn enter(work: QueryWork, context: &WorkContext) -> Self {
        Self {
            previous: ACTIVE.replace(Some(work)),
            context: CONTEXT.replace(Some(context.clone())),
        }
    }
    fn finish(self) -> QueryWork {
        // Drop restores the outer scope, including on panic.
        ACTIVE.with_borrow_mut(|active| active.take().unwrap())
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.set(self.previous.take());
        CONTEXT.set(self.context.take());
    }
}

/// Capture work on the calling thread. Restores outer capture even on panic.
pub fn capture_sync<T>(run: impl FnOnce() -> T) -> (T, QueryWork) {
    let context = WorkContext::default();
    let scope = Scope::enter(QueryWork::default(), &context);
    let result = run();
    let mut work = scope.finish();
    work.merge(*context.0.lock());
    (result, work)
}

/// Capture only work performed while polling this future, on any polling thread.
/// Synchronous Searcher segment workers are included. Arbitrary spawned tasks
/// and other CPU-pool entry points are not.
/// Dropping a pending future
/// cannot leave an active capture behind.
pub async fn capture<F: Future>(future: F) -> (F::Output, QueryWork) {
    let mut future = std::pin::pin!(future);
    let mut work = QueryWork::default();
    let context = WorkContext::default();
    let output = poll_fn(|cx| {
        let scope = Scope::enter(work, &context);
        let result = future.as_mut().poll(cx);
        work = scope.finish();
        result
    })
    .await;
    work.merge(*context.0.lock());
    (output, work)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{Context, Poll, Waker};

    #[test]
    fn nested_and_panicking_captures_restore_outer_work() {
        let (_, outer) = capture_sync(|| {
            update(|w| w.posting_seeks += 1);
            let (_, inner) = capture_sync(|| update(|w| w.posting_seeks += 7));
            assert_eq!(inner.posting_seeks, 7);
            assert!(std::panic::catch_unwind(|| capture_sync(|| panic!("stop"))).is_err());
            update(|w| w.posting_seeks += 2);
        });
        assert_eq!(outer.posting_seeks, 3);
        assert!(ACTIVE.with_borrow(Option::is_none));
    }

    #[cfg(feature = "sync")]
    #[test]
    fn segment_workers_merge_once_and_keep_concurrent_queries_separate() {
        let queries: Vec<_> = [3, 17]
            .into_iter()
            .map(|n| {
                std::thread::spawn(move || {
                    let (_, work) = capture_sync(|| {
                        update(|w| w.doc_values += 1);
                        let context = WorkContext::current().unwrap();
                        std::thread::scope(|scope| {
                            for _ in 0..2 {
                                let context = context.clone();
                                scope.spawn(move || {
                                    let _scope = context.enter();
                                    update(|w| w.doc_values += n);
                                });
                            }
                        });
                    });
                    assert_eq!(work.doc_values, 1 + 2 * n);
                    assert_eq!(work.segment_runs, 2);
                    assert!(ACTIVE.with_borrow(Option::is_none));
                    assert!(WorkContext::current().is_none());
                })
            })
            .collect();
        for query in queries {
            query.join().unwrap();
        }
    }

    #[test]
    fn pending_cancelled_and_migrating_futures_do_not_capture_other_work() {
        let mut first = true;
        let future = poll_fn(move |_| {
            update(|w| w.doc_blocks += 1);
            if std::mem::take(&mut first) {
                Poll::Pending
            } else {
                Poll::Ready(42)
            }
        });
        let mut future = Box::pin(capture(future));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(future.as_mut().poll(&mut cx).is_pending());
        update(|w| w.doc_blocks += 100);
        assert!(ACTIVE.with_borrow(Option::is_none));
        let (value, work) = std::thread::spawn(move || {
            let mut cx = Context::from_waker(Waker::noop());
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(value) => value,
                Poll::Pending => panic!("second poll should finish"),
            }
        })
        .join()
        .unwrap();
        assert_eq!(value, 42);
        assert_eq!(work.doc_blocks, 2);
        let mut cancelled = Box::pin(capture(std::future::pending::<()>()));
        assert!(cancelled.as_mut().poll(&mut cx).is_pending());
        drop(cancelled);
        assert!(ACTIVE.with_borrow(Option::is_none));
        assert_eq!(capture_sync(|| ()).1, QueryWork::default());
    }
}
