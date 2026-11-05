//! Process-wide occupancy of the shared search CPU pool and sparse random-I/O
//! gates, for stall diagnosis.
//!
//! A server whose admission permits are all held while no search completes
//! needs to say where the requests wait: queued for the pool, running on it,
//! blocked handing work to it, or parked on a sparse I/O gate. Each counter is
//! one relaxed atomic update per job or wait, maintained by RAII guards so
//! cancellation and panics cannot leak counts.

use std::sync::atomic::{AtomicUsize, Ordering};

static QUEUED_JOBS: AtomicUsize = AtomicUsize::new(0);
static RUNNING_JOBS: AtomicUsize = AtomicUsize::new(0);
static INSTALL_WAITERS: AtomicUsize = AtomicUsize::new(0);

/// Snapshot of one process-wide sparse random-I/O gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseIoGateState {
    /// Configured concurrent scorer limit.
    pub limit: usize,
    /// Permits currently held.
    pub active: usize,
    /// Scorers blocked or parked waiting for a permit.
    pub waiting: usize,
}

/// Snapshot of shared search execution state across every open index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchRuntimeState {
    /// Request-level jobs submitted to a search pool and not yet started.
    pub queued_pool_jobs: usize,
    /// Request-level jobs currently executing on a search pool.
    pub running_pool_jobs: usize,
    /// Threads outside a search pool blocked until work they handed to it
    /// returns (async workers inside `block_in_place`, other Rayon pools).
    pub pool_install_waiters: usize,
    /// One entry per live sparse I/O gate, ordered by limit.
    pub sparse_io_gates: Vec<SparseIoGateState>,
}

/// Current shared search execution state. Counters are sampled one at a
/// time, so a snapshot taken during transitions may be off by one job.
#[cfg(feature = "native")]
pub fn search_runtime_state() -> SearchRuntimeState {
    SearchRuntimeState {
        queued_pool_jobs: QUEUED_JOBS.load(Ordering::Relaxed),
        running_pool_jobs: RUNNING_JOBS.load(Ordering::Relaxed),
        pool_install_waiters: INSTALL_WAITERS.load(Ordering::Relaxed),
        sparse_io_gates: super::sparse_io_gate_states(),
    }
}

/// A request-level job between submission and start. Dropping it without
/// starting (the pool discarded the closure) releases the queued count.
#[cfg(feature = "sync")]
pub(crate) struct QueuedPoolJob(());

#[cfg(feature = "sync")]
impl QueuedPoolJob {
    pub(crate) fn new() -> Self {
        QUEUED_JOBS.fetch_add(1, Ordering::Relaxed);
        Self(())
    }

    pub(crate) fn start(self) -> RunningPoolJob {
        RUNNING_JOBS.fetch_add(1, Ordering::Relaxed);
        RunningPoolJob(())
    }
}

#[cfg(feature = "sync")]
impl Drop for QueuedPoolJob {
    fn drop(&mut self) {
        QUEUED_JOBS.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "sync")]
pub(crate) struct RunningPoolJob(());

#[cfg(feature = "sync")]
impl Drop for RunningPoolJob {
    fn drop(&mut self) {
        RUNNING_JOBS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A thread outside the pool blocked in `ThreadPool::install`.
#[cfg(feature = "sync")]
pub(crate) struct InstallWaiter(());

#[cfg(feature = "sync")]
impl InstallWaiter {
    pub(crate) fn new() -> Self {
        INSTALL_WAITERS.fetch_add(1, Ordering::Relaxed);
        Self(())
    }
}

#[cfg(feature = "sync")]
impl Drop for InstallWaiter {
    fn drop(&mut self) {
        INSTALL_WAITERS.fetch_sub(1, Ordering::Relaxed);
    }
}
