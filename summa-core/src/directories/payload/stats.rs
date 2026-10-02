//! Service counters shared by both execution backends.
use super::DEPTH;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Cumulative payload I/O accounting. Caller-retained bytes are excluded.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct PayloadReadStats {
    pub submitted: u64,
    pub completed: u64,
    pub skipped: u64,
    pub dropped: u64,
    pub errors: u64,
    pub bytes: u64,
    pub active: u64,
    pub max_active: u64,
    pub worker_failures: u64,
    /// Buffers retained for safety if exceptional kernel cancellation fails.
    pub quarantined_bytes: u64,
    /// Actual number of reads enqueued in each ring submission group.
    pub batches: [u64; DEPTH + 1],
    pub refills: u64,
    /// Queue notification CQEs, separate from completed payload reads.
    pub wakeups: u64,
    /// Fatal producer-notification failures; the service then refuses admission.
    pub notification_failures: u64,
    pub idle_buffer_bytes: usize,
    pub buffer_reuses: u64,
    /// Allocations while reuse is enabled; disabled mode avoids pool accounting.
    pub buffer_allocations: u64,
}
#[derive(Default)]
pub(super) struct Counters {
    submitted: AtomicU64,
    completed: AtomicU64,
    skipped: AtomicU64,
    dropped: AtomicU64,
    errors: AtomicU64,
    bytes: AtomicU64,
    active: AtomicU64,
    max_active: AtomicU64,
    worker_failures: AtomicU64,
    pub(super) quarantined_bytes: AtomicU64,
    batches: [AtomicU64; DEPTH + 1],
    refills: AtomicU64,
    wakeups: AtomicU64,
    notification_failures: AtomicU64,
}
impl Counters {
    pub(super) fn started(&self, len: usize) {
        self.submitted.fetch_add(1, Relaxed);
        self.bytes.fetch_add(len as u64, Relaxed);
        let active = self.active.fetch_add(1, Relaxed) + 1;
        self.max_active.fetch_max(active, Relaxed);
    }
    pub(super) fn finished(&self, error: bool) {
        self.completed.fetch_add(1, Relaxed);
        self.active.fetch_sub(1, Relaxed);
        if error {
            self.errors.fetch_add(1, Relaxed);
        }
    }
    pub(super) fn skipped(&self) {
        self.skipped.fetch_add(1, Relaxed);
    }
    pub(super) fn dropped(&self) {
        self.dropped.fetch_add(1, Relaxed);
    }
    pub(super) fn failed(&self) {
        self.worker_failures.fetch_add(1, Relaxed);
    }
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub(super) fn batch(&self, count: usize, refill: bool) {
        self.batches[count].fetch_add(1, Relaxed);
        if refill {
            self.refills.fetch_add(1, Relaxed);
        }
    }
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub(super) fn woke(&self) {
        self.wakeups.fetch_add(1, Relaxed);
    }
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub(super) fn notification_failed(&self) {
        self.notification_failures.fetch_add(1, Relaxed);
    }
    pub(super) fn snapshot(&self) -> PayloadReadStats {
        PayloadReadStats {
            submitted: self.submitted.load(Relaxed),
            completed: self.completed.load(Relaxed),
            skipped: self.skipped.load(Relaxed),
            dropped: self.dropped.load(Relaxed),
            errors: self.errors.load(Relaxed),
            bytes: self.bytes.load(Relaxed),
            active: self.active.load(Relaxed),
            max_active: self.max_active.load(Relaxed),
            worker_failures: self.worker_failures.load(Relaxed),
            quarantined_bytes: self.quarantined_bytes.load(Relaxed),
            batches: std::array::from_fn(|i| self.batches[i].load(Relaxed)),
            refills: self.refills.load(Relaxed),
            wakeups: self.wakeups.load(Relaxed),
            notification_failures: self.notification_failures.load(Relaxed),
            ..Default::default()
        }
    }
}
