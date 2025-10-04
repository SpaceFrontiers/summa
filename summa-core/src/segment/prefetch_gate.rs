//! Residency-sampled gate for `MADV_WILLNEED` prefetch of evictable mmap
//! sections.
//!
//! BMP issues bounded `MADV_WILLNEED` hints for the D-grid groups and block
//! payloads it is about to read, which hides major faults when the index does
//! not fit in the page cache. Each hint is a syscall (mmap lock, VMA lookup and
//! a page-cache probe per page) that is pure overhead when every page is
//! already cached: on a warm SPLADE-1M index it was 48–55% of BMP query time.
//!
//! The gate samples residency with `mincore` on one prefetch stage in
//! [`PROBE_INTERVAL`] (per thread) and closes after
//! [`CLOSE_AFTER_RESIDENT_PROBES`] consecutive probes found every sampled page
//! cached. Any probe that finds a missing page reopens it immediately, so
//! partially resident and cold segments keep today's prefetch behavior. A
//! closed gate still probes, which bounds the number of unhinted stages after
//! pages are evicted to `PROBE_INTERVAL - 1` per thread.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};

/// One prefetch stage in this many (per thread) samples page residency.
pub(crate) const PROBE_INTERVAL: u32 = 32;
/// Consecutive fully resident probes required before hints are skipped.
///
/// Within one cold query some probes see pages an earlier pipeline stage
/// already advised. A SPLADE-1M query runs about nine probes, so a cold
/// segment, whose every query starts with misses, cannot reach this streak;
/// with 64 the gate closed mid-query and took 27% more major faults.
pub(crate) const CLOSE_AFTER_RESIDENT_PROBES: u32 = 256;

thread_local! {
    static PREFETCH_TICK: Cell<u32> = const { Cell::new(0) };
    static WITHHELD_STAGES: Cell<usize> = const { Cell::new(0) };
}

/// Monotonic count of prefetch stages this thread withheld because sampled
/// pages were resident. Executors difference it around one query to report
/// `summa_bmp_prefetch_gated_total`.
#[inline]
pub(crate) fn withheld_stages() -> usize {
    WITHHELD_STAGES.with(Cell::get)
}

#[inline]
fn count_withheld() {
    WITHHELD_STAGES.with(|stages| stages.set(stages.get().wrapping_add(1)));
}

/// What one prefetch stage should do with its candidate ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefetchAction {
    /// The gate is closed and this is not a probe turn: issue no hints and
    /// skip building ranges.
    Skip,
    /// The gate is open: advise every range. `probe` also samples residency
    /// first, which is how an open gate learns that it may close.
    Advise { probe: bool },
    /// The gate is closed: sample residency and advise only when a sampled
    /// page is missing (which also reopens the gate).
    Probe,
}

/// Shared, lock-free residency state for one evictable mapping.
#[derive(Debug, Default)]
pub(crate) struct PrefetchGate {
    resident_probes: AtomicU32,
}

impl PrefetchGate {
    /// Decide this stage's action. Starts open: a freshly opened segment is
    /// usually cold.
    #[inline]
    pub(crate) fn action(&self) -> PrefetchAction {
        let probe = PREFETCH_TICK.with(|tick| {
            let next = tick.get().wrapping_add(1);
            tick.set(next);
            next.is_multiple_of(PROBE_INTERVAL)
        });
        let open = self.resident_probes.load(Ordering::Relaxed) < CLOSE_AFTER_RESIDENT_PROBES;
        match (open, probe) {
            (true, probe) => PrefetchAction::Advise { probe },
            (false, true) => PrefetchAction::Probe,
            (false, false) => {
                count_withheld();
                PrefetchAction::Skip
            }
        }
    }

    /// Record one probe. Concurrent updates race benignly: the worst case is
    /// one extra or one fewer resident probe before closing.
    #[inline]
    pub(crate) fn record_probe(&self, all_resident: bool) {
        if all_resident {
            let current = self.resident_probes.load(Ordering::Relaxed);
            if current < CLOSE_AFTER_RESIDENT_PROBES {
                self.resident_probes.store(current + 1, Ordering::Relaxed);
            }
        } else {
            self.resident_probes.store(0, Ordering::Relaxed);
        }
    }

    /// Apply `action` to coalesced `ranges`: probe with `resident` when the
    /// action asks for it, then return whether the ranges should be advised.
    #[inline]
    pub(crate) fn admit<R>(
        &self,
        action: PrefetchAction,
        ranges: &[R],
        mut resident: impl FnMut(&R) -> bool,
    ) -> bool {
        let probe = |resident: &mut dyn FnMut(&R) -> bool| ranges.iter().all(resident);
        match action {
            PrefetchAction::Skip => false,
            PrefetchAction::Advise { probe: false } => true,
            PrefetchAction::Advise { probe: true } => {
                self.record_probe(probe(&mut resident));
                true
            }
            PrefetchAction::Probe => {
                let all_resident = probe(&mut resident);
                self.record_probe(all_resident);
                if all_resident {
                    count_withheld();
                }
                !all_resident
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_stages(gate: &PrefetchGate, stages: u32, resident: bool) -> (u32, u32) {
        let (mut advised, mut skipped) = (0, 0);
        for _ in 0..stages {
            let action = gate.action();
            if gate.admit(action, &[()], |_| resident) {
                advised += 1;
            } else {
                skipped += 1;
            }
        }
        (advised, skipped)
    }

    #[test]
    fn prefetch_gate_closes_only_after_sustained_residency_and_reopens_on_a_miss() {
        let gate = PrefetchGate::default();
        let warm_up = PROBE_INTERVAL * CLOSE_AFTER_RESIDENT_PROBES;
        // Open until the required number of resident probes has been seen.
        let (advised, skipped) = run_stages(&gate, warm_up - 1, true);
        assert_eq!((advised, skipped), (warm_up - 1, 0));
        // The next stage is the closing probe; it still advises its ranges.
        assert_eq!(run_stages(&gate, 1, true), (1, 0));
        // Closed: only probe turns run, and resident probes advise nothing.
        let withheld = withheld_stages();
        let (advised, skipped) = run_stages(&gate, 4 * PROBE_INTERVAL, true);
        assert_eq!((advised, skipped), (0, 4 * PROBE_INTERVAL));
        assert_eq!(withheld_stages() - withheld, 4 * PROBE_INTERVAL as usize);
        // A miss is advised at the next probe turn (the last of these
        // stages) and reopens the gate.
        let (advised, skipped) = run_stages(&gate, PROBE_INTERVAL, false);
        assert_eq!((advised, skipped), (1, PROBE_INTERVAL - 1));
        assert!(matches!(gate.action(), PrefetchAction::Advise { .. }));
    }

    #[test]
    fn prefetch_gate_keeps_advising_while_any_probe_misses() {
        let gate = PrefetchGate::default();
        let stages = 8 * PROBE_INTERVAL * CLOSE_AFTER_RESIDENT_PROBES;
        let (mut advised, mut probes) = (0, 0u32);
        for _ in 0..stages {
            let action = gate.action();
            // Every eighth probe misses a page: never enough consecutive
            // resident probes to close.
            advised += u32::from(gate.admit(action, &[()], |_| {
                probes += 1;
                !probes.is_multiple_of(8)
            }));
        }
        assert_eq!(advised, stages);
        assert_eq!(probes, stages / PROBE_INTERVAL);
    }
}
