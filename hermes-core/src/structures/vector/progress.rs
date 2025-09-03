//! Progress reporting for long, otherwise silent training phases.
//!
//! Production codebook training runs for tens of minutes — a 163,342-cluster
//! binary field spent 1h24m between "training started" and "artifact saved"
//! with no output at all, which is indistinguishable from a hang. Every phase
//! that can run longer than a few seconds reports here, rate-limited so a long
//! phase produces a steady trickle rather than a flood.

#[cfg(feature = "native")]
use std::time::{Duration, Instant};

/// Minimum gap between progress lines within one phase.
#[cfg(feature = "native")]
const PROGRESS_INTERVAL: Duration = Duration::from_secs(30);

/// One named training phase with a known amount of work.
pub(crate) struct PhaseProgress {
    index: String,
    label: &'static str,
    detail: String,
    total: usize,
    report: bool,
    #[cfg(feature = "native")]
    started: Instant,
    #[cfg(feature = "native")]
    last_report: Instant,
}

impl PhaseProgress {
    /// Announce a phase. `total` is the unit count `advance` counts toward.
    ///
    /// `index` names the index being trained: several indexes train at once in
    /// production, and without it their interleaved lines are unattributable.
    pub(crate) fn start(index: &str, label: &'static str, detail: String, total: usize) -> Self {
        Self::start_if(true, index, label, detail, total)
    }

    /// As [`Self::start`], but silent when `report` is false.
    ///
    /// Hierarchical training runs hundreds of small child codebooks through the
    /// same code as one large one; only the outer phase should be logged.
    pub(crate) fn start_if(
        report: bool,
        index: &str,
        label: &'static str,
        detail: String,
        total: usize,
    ) -> Self {
        if report {
            log::info!("[vector_training] {label} started: index={index} {detail}");
        }
        Self {
            index: index.to_owned(),
            label,
            detail,
            total,
            report,
            #[cfg(feature = "native")]
            started: Instant::now(),
            #[cfg(feature = "native")]
            last_report: Instant::now(),
        }
    }

    /// Report cumulative progress, at most once per [`PROGRESS_INTERVAL`].
    ///
    /// Called from sequential outer loops only, so it never contends.
    #[allow(unused_variables)]
    pub(crate) fn advance(&mut self, done: usize) {
        if !self.report {
            return;
        }
        #[cfg(feature = "native")]
        {
            let now = Instant::now();
            if now.duration_since(self.last_report) < PROGRESS_INTERVAL || done == 0 {
                return;
            }
            self.last_report = now;
            let elapsed = now.duration_since(self.started);
            let percent = if self.total == 0 {
                0.0
            } else {
                100.0 * done as f64 / self.total as f64
            };
            // Extrapolate from the completed fraction; the remaining work is
            // homogeneous in every phase reporting here.
            let remaining = if done == 0 || done >= self.total {
                Duration::ZERO
            } else {
                elapsed.mul_f64((self.total - done) as f64 / done as f64)
            };
            log::info!(
                "[vector_training] {} index={} {}/{} ({percent:.1}%) in {:.1}s, ~{:.0}s left: {}",
                self.label,
                self.index,
                done,
                self.total,
                elapsed.as_secs_f64(),
                remaining.as_secs_f64(),
                self.detail,
            );
        }
    }

    /// Progress counter for a phase whose units complete on worker threads.
    ///
    /// Reporting is gated on completion count rather than elapsed time so it
    /// stays lock-free; the phase emits at most [`SHARED_PROGRESS_STEPS`] lines.
    pub(crate) fn shared(&self) -> SharedProgress {
        SharedProgress {
            index: self.index.clone(),
            label: self.label,
            total: self.total,
            done: std::sync::atomic::AtomicUsize::new(0),
            step: self.total.div_ceil(SHARED_PROGRESS_STEPS).max(1),
        }
    }

    /// Close the phase with its total wall time.
    pub(crate) fn finish(self) {
        if !self.report {
            return;
        }
        #[cfg(feature = "native")]
        log::info!(
            "[vector_training] {} complete in {:.1}s: index={} {}",
            self.label,
            self.started.elapsed().as_secs_f64(),
            self.index,
            self.detail,
        );
        #[cfg(not(feature = "native"))]
        log::info!(
            "[vector_training] {} complete: index={} {}",
            self.label,
            self.index,
            self.detail
        );
    }
}

/// Lines emitted by a [`SharedProgress`] over a whole phase.
const SHARED_PROGRESS_STEPS: usize = 20;

/// Completion counter safe to advance from worker threads.
pub(crate) struct SharedProgress {
    index: String,
    label: &'static str,
    total: usize,
    done: std::sync::atomic::AtomicUsize,
    step: usize,
}

impl SharedProgress {
    /// Record one completed unit, logging on every `step`-th completion.
    pub(crate) fn complete_one(&self) {
        let done = self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if done.is_multiple_of(self.step) && done < self.total {
            log::info!(
                "[vector_training] {} index={} {done}/{} ({:.0}%)",
                self.label,
                self.index,
                self.total,
                100.0 * done as f64 / self.total as f64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    static CAPTURED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

    struct CaptureLogger {
        thread: std::thread::ThreadId,
    }

    impl log::Log for CaptureLogger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            std::thread::current().id() == self.thread
        }
        fn log(&self, record: &log::Record<'_>) {
            if !self.enabled(record.metadata()) {
                return;
            }
            CAPTURED
                .get_or_init(Default::default)
                .lock()
                .unwrap()
                .push(record.args().to_string());
        }
        fn flush(&self) {}
    }

    /// Every training line must name its index: production runs several
    /// retrains at once and their output interleaves.
    #[test]
    fn training_progress_lines_carry_the_index_name() {
        let _ = log::set_boxed_logger(Box::new(CaptureLogger {
            thread: std::thread::current().id(),
        }));
        log::set_max_level(log::LevelFilter::Info);

        let phase = PhaseProgress::start(
            "documents_20260724",
            "k-majority seeding",
            "163342 centroids".into(),
            4,
        );
        phase.shared().complete_one();
        phase.finish();
        let silent = PhaseProgress::start_if(
            false,
            "social_20260724",
            "child codebooks",
            "quiet".into(),
            1,
        );
        silent.finish();

        let lines = CAPTURED
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .clone();
        assert!(
            lines
                .iter()
                .all(|line| line.contains("index=documents_20260724")),
            "unlabelled training line: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| line
                .contains("k-majority seeding started: index=documents_20260724 163342 centroids")),
            "start line shape changed: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("k-majority seeding complete in")),
            "missing completion line: {lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.contains("social")),
            "suppressed phase must stay silent: {lines:?}"
        );
    }
}
