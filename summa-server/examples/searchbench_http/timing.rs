//! Bounded, opt-in aggregate wall timings. Never used for throughput claims.
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{Value, json};
use summa_core::search_diagnostics::QueryWork;

const NAMES: [&str; 9] = [
    "blocking_queue",
    "parse",
    "search",
    "project",
    "encode_and_drop",
    "blocking_return",
    "core_pool_queue",
    "core_pool_return",
    "segment_work",
];

pub struct Trace {
    times: [u64; 9],
    mark: Instant,
}

impl Trace {
    pub fn start(submitted: Instant) -> Self {
        let now = Instant::now();
        let mut times = [0; 9];
        times[0] = nanos(now.duration_since(submitted));
        Self { times, mark: now }
    }

    pub fn mark(&mut self, stage: usize) {
        let now = Instant::now();
        self.times[stage] = nanos(now.duration_since(self.mark));
        self.mark = now;
    }

    pub fn work(&mut self, work: QueryWork) {
        self.times[6] = work.search_pool_queue_ns;
        self.times[7] = work.search_pool_return_ns;
        self.times[8] = work.segment_elapsed_ns;
    }
}

fn nanos(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

#[derive(Default)]
pub struct Totals(Mutex<Summary>);

#[derive(Default)]
struct Summary {
    requests: u64,
    errors: u64,
    sums: [u64; 9],
    maxima: [u64; 9],
}

impl Totals {
    pub fn complete(&self, mut trace: Trace) {
        trace.mark(5);
        let mut summary = self.0.lock().unwrap();
        summary.requests = summary.requests.saturating_add(1);
        for (i, value) in trace.times.into_iter().enumerate() {
            summary.sums[i] = summary.sums[i].saturating_add(value);
            summary.maxima[i] = summary.maxima[i].max(value);
        }
    }

    pub fn error(&self) {
        let mut summary = self.0.lock().unwrap();
        summary.errors = summary.errors.saturating_add(1);
    }

    pub fn snapshot(&self) -> Value {
        let summary = self.0.lock().unwrap();
        let stages: serde_json::Map<_, _> = NAMES.iter().enumerate().map(|(i, name)| {
            ((*name).into(), json!({
                "sum_ns":summary.sums[i], "max_ns":summary.maxima[i],
                "mean_us":if summary.requests == 0 { 0.0 } else { summary.sums[i] as f64 / summary.requests as f64 / 1000.0 },
            }))
        }).collect();
        json!({"successful_requests":summary.requests,"failed_workers":summary.errors,"stages":stages})
    }
}
