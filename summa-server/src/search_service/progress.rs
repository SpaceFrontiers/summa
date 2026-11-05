//! Search admission progress and stall detection.
//!
//! An admitted search holds one admission permit until both its request and
//! any search-pool job sharing the permit have finished. If every permit stays
//! held while no search completes, the server rejects all searches with
//! `RESOURCE_EXHAUSTED` indefinitely: cancellation cannot free a permit held
//! by running or wedged pool work. [`StallMonitor`] detects that state and
//! reports where each admitted search is, so the owner can restart the
//! process and the report identifies the blocked stage.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Longest query description kept per admitted search.
const MAX_QUERY_DESCRIPTION_CHARS: usize = 240;

/// Where an admitted search is in the request pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SearchPhase {
    /// Opening the index, reader and searcher.
    Opening,
    /// L1 retrieval, candidate scoring and fusion.
    Retrieving,
    /// L2 reranking.
    Reranking,
    /// Loading stored fields into the response.
    Loading,
}

impl SearchPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::Retrieving => "retrieving",
            Self::Reranking => "reranking",
            Self::Loading => "loading",
        }
    }
}

#[derive(Clone, Debug)]
struct InFlight {
    index: String,
    admitted: Instant,
    phase: SearchPhase,
    query: String,
    request_ended: bool,
}

/// Admitted searches and the time of the last completed one.
#[derive(Debug)]
pub(crate) struct SearchProgress {
    epoch: Instant,
    last_completed_ms: AtomicU64,
    next_id: AtomicU64,
    in_flight: parking_lot::Mutex<BTreeMap<u64, InFlight>>,
}

impl SearchProgress {
    pub(crate) fn new() -> Arc<Self> {
        Self::with_epoch(Instant::now())
    }

    /// Progress whose clock started at `epoch`; tests backdate it.
    pub(crate) fn with_epoch(epoch: Instant) -> Arc<Self> {
        Arc::new(Self {
            epoch,
            last_completed_ms: AtomicU64::new(0),
            next_id: AtomicU64::new(0),
            in_flight: parking_lot::Mutex::new(BTreeMap::new()),
        })
    }

    /// Register an admitted search; the entry lives as long as the returned
    /// admission, which also owns the permit.
    pub(crate) fn admit(
        self: &Arc<Self>,
        permit: OwnedSemaphorePermit,
        index: &str,
    ) -> Arc<SearchAdmission> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.in_flight.lock().insert(
            id,
            InFlight {
                index: index.to_owned(),
                admitted: Instant::now(),
                phase: SearchPhase::Opening,
                query: String::new(),
                request_ended: false,
            },
        );
        Arc::new(SearchAdmission {
            progress: Arc::clone(self),
            id,
            _permit: permit,
        })
    }

    /// Record that an admitted search returned a response or an error.
    /// Cancelled requests did not finish their work and are not progress.
    pub(crate) fn record_completion(&self) {
        let elapsed = u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.last_completed_ms.fetch_max(elapsed, Ordering::Relaxed);
    }

    /// Time since the last completed search, or since startup if none has.
    fn since_last_completion(&self) -> Duration {
        let last = Duration::from_millis(self.last_completed_ms.load(Ordering::Relaxed));
        self.epoch.elapsed().saturating_sub(last)
    }

    fn update(&self, id: u64, apply: impl FnOnce(&mut InFlight)) {
        if let Some(entry) = self.in_flight.lock().get_mut(&id) {
            apply(entry);
        }
    }
}

/// One admitted search. Shared by the request and any search-pool job that
/// outlives it; the permit and the in-flight entry are released together.
#[derive(Debug)]
pub(crate) struct SearchAdmission {
    progress: Arc<SearchProgress>,
    id: u64,
    _permit: OwnedSemaphorePermit,
}

impl SearchAdmission {
    pub(crate) fn set_phase(&self, phase: SearchPhase) {
        self.progress.update(self.id, |entry| entry.phase = phase);
    }

    pub(crate) fn describe(&self, query: &str) {
        let query: String = query.chars().take(MAX_QUERY_DESCRIPTION_CHARS).collect();
        self.progress.update(self.id, |entry| entry.query = query);
    }

    /// Guard for the request future. Dropping it (completion or
    /// cancellation) marks that only shared work can still hold the permit.
    pub(crate) fn request_scope(self: &Arc<Self>) -> RequestScope {
        RequestScope(Arc::clone(self))
    }
}

impl Drop for SearchAdmission {
    fn drop(&mut self) {
        self.progress.in_flight.lock().remove(&self.id);
    }
}

pub(crate) struct RequestScope(Arc<SearchAdmission>);

impl Drop for RequestScope {
    fn drop(&mut self) {
        self.0
            .progress
            .update(self.0.id, |entry| entry.request_ended = true);
    }
}

/// Detects a server whose admission is exhausted without progress.
#[derive(Clone, Debug)]
pub struct StallMonitor {
    pub(crate) permits: Arc<Semaphore>,
    pub(crate) capacity: usize,
    pub(crate) progress: Arc<SearchProgress>,
    /// When [`Self::check`] first saw every permit held, since admission was
    /// last observed free.
    pub(crate) exhausted_since: Arc<parking_lot::Mutex<Option<Instant>>>,
}

impl StallMonitor {
    pub(crate) fn new(
        permits: Arc<Semaphore>,
        capacity: usize,
        progress: Arc<SearchProgress>,
    ) -> Self {
        Self {
            permits,
            capacity,
            progress,
            exhausted_since: Arc::default(),
        }
    }

    /// A report when every admission permit has been held, as observed by
    /// successive checks, for at least `stall_after` and no search has
    /// completed meanwhile. A burst that fills admission after a quiet period
    /// starts the clock at that check, not at the last completion.
    pub fn check(&self, stall_after: Duration) -> Option<StallReport> {
        let now = Instant::now();
        let mut exhausted_since = self.exhausted_since.lock();
        if self.permits.available_permits() > 0 {
            *exhausted_since = None;
            return None;
        }
        let exhausted_for = now.saturating_duration_since(*exhausted_since.get_or_insert(now));
        drop(exhausted_since);
        let idle = exhausted_for.min(self.progress.since_last_completion());
        if idle < stall_after {
            return None;
        }
        let mut in_flight: Vec<_> = self
            .progress
            .in_flight
            .lock()
            .values()
            .map(|entry| StalledSearch {
                index: entry.index.clone(),
                age: now.saturating_duration_since(entry.admitted),
                phase: entry.phase.as_str(),
                query: entry.query.clone(),
                request_ended: entry.request_ended,
            })
            .collect();
        in_flight.sort_by_key(|search| std::cmp::Reverse(search.age));
        Some(StallReport {
            capacity: self.capacity,
            idle,
            in_flight,
            runtime: summa_core::index::search_runtime_state(),
        })
    }
}

/// One admitted search at stall time.
#[derive(Clone, Debug)]
pub struct StalledSearch {
    pub index: String,
    pub age: Duration,
    pub phase: &'static str,
    pub query: String,
    /// The caller has gone; shared pool work still holds the permit.
    pub request_ended: bool,
}

/// What the server was doing when admission stalled.
#[derive(Clone, Debug)]
pub struct StallReport {
    pub capacity: usize,
    pub idle: Duration,
    pub in_flight: Vec<StalledSearch>,
    pub runtime: summa_core::index::SearchRuntimeState,
}

impl fmt::Display for StallReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let runtime = &self.runtime;
        write!(
            f,
            "all {} search permits held and no search completed for {}s; \
             pool jobs queued={} running={}, pool install waiters={}",
            self.capacity,
            self.idle.as_secs(),
            runtime.queued_pool_jobs,
            runtime.running_pool_jobs,
            runtime.pool_install_waiters,
        )?;
        for gate in &runtime.sparse_io_gates {
            write!(
                f,
                "; sparse I/O gate limit={} active={} waiting={}",
                gate.limit, gate.active, gate.waiting
            )?;
        }
        for search in &self.in_flight {
            write!(
                f,
                "\n  index={} age={}s phase={} request={} query={}",
                search.index,
                search.age.as_secs(),
                search.phase,
                if search.request_ended {
                    "ended"
                } else {
                    "waiting"
                },
                search.query,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STALL: Duration = Duration::from_millis(50);

    fn monitor(capacity: usize, progress: &Arc<SearchProgress>) -> StallMonitor {
        StallMonitor::new(
            Arc::new(Semaphore::new(capacity)),
            capacity,
            Arc::clone(progress),
        )
    }

    fn hour_old() -> Arc<SearchProgress> {
        SearchProgress::with_epoch(Instant::now() - Duration::from_secs(3600))
    }

    fn admit(monitor: &StallMonitor, index: &str) -> Arc<SearchAdmission> {
        monitor
            .progress
            .admit(monitor.permits.clone().try_acquire_owned().unwrap(), index)
    }

    /// Observe exhaustion, then check again after the stall interval.
    fn check_after_stall(monitor: &StallMonitor) -> Option<StallReport> {
        let first = monitor.check(STALL);
        assert!(
            first.is_none(),
            "the first observation only starts the clock"
        );
        std::thread::sleep(STALL + Duration::from_millis(20));
        monitor.check(STALL)
    }

    #[test]
    fn test_stall_requires_every_permit_held() {
        let progress = hour_old();
        let monitor = monitor(2, &progress);
        let _one = admit(&monitor, "a");
        assert!(monitor.check(Duration::ZERO).is_none());
        let _two = admit(&monitor, "a");
        assert!(check_after_stall(&monitor).is_some());
    }

    #[test]
    fn test_burst_after_quiet_period_is_not_a_stall() {
        // Nothing completed for an hour, then a burst fills admission: the
        // clock starts when exhaustion is observed, not at the last completion.
        let progress = hour_old();
        let monitor = monitor(1, &progress);
        let _burst = admit(&monitor, "a");
        assert!(monitor.check(Duration::from_secs(60)).is_none());
        assert!(monitor.check(Duration::from_secs(60)).is_none());
    }

    #[test]
    fn test_free_permit_resets_the_exhaustion_clock() {
        let progress = hour_old();
        let monitor = monitor(1, &progress);
        let held = admit(&monitor, "a");
        assert!(monitor.check(STALL).is_none());
        drop(held);
        std::thread::sleep(STALL + Duration::from_millis(20));
        assert!(monitor.check(STALL).is_none());
        let _held = admit(&monitor, "a");
        assert!(check_after_stall(&monitor).is_some());
    }

    #[test]
    fn test_completed_search_resets_the_stall_clock() {
        let progress = hour_old();
        let monitor = monitor(1, &progress);
        let _held = admit(&monitor, "a");
        assert!(check_after_stall(&monitor).is_some());
        progress.record_completion();
        assert!(monitor.check(STALL).is_none());
    }

    #[test]
    fn test_pool_work_keeps_permit_and_entry_after_request_ends() {
        let progress = hour_old();
        let monitor = monitor(1, &progress);
        let admission = admit(&monitor, "social");
        admission.set_phase(SearchPhase::Retrieving);
        admission.describe("Boolean(+Term +Range Sparse(4, dims=11))");
        let pool_job = Arc::clone(&admission);
        let request = admission.request_scope();
        drop(admission);
        drop(request);

        let report = check_after_stall(&monitor).unwrap();
        assert_eq!(report.in_flight.len(), 1);
        let search = &report.in_flight[0];
        assert_eq!(
            (search.index.as_str(), search.phase),
            ("social", "retrieving")
        );
        assert!(search.request_ended, "the cancelled caller is gone");
        let text = report.to_string();
        assert!(text.contains("all 1 search permits held"), "{text}");
        assert!(
            text.contains("phase=retrieving request=ended query=Boolean"),
            "{text}"
        );

        drop(pool_job);
        assert_eq!(monitor.permits.available_permits(), 1);
        assert!(progress.in_flight.lock().is_empty());
        assert!(monitor.check(Duration::ZERO).is_none());
    }

    #[test]
    fn test_query_description_is_bounded() {
        let progress = hour_old();
        let monitor = monitor(1, &progress);
        let admission = admit(&monitor, "a");
        admission.describe(&"x".repeat(10_000));
        let report = check_after_stall(&monitor).unwrap();
        assert_eq!(report.in_flight[0].query.len(), MAX_QUERY_DESCRIPTION_CHARS);
    }
}
