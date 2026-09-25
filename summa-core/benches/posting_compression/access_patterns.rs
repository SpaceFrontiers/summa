//! Access pattern benchmarks for posting lists
//!
//! Measures performance under realistic access patterns:
//! - Sequential iteration (full scan)
//! - Read first N documents
//! - Random seek access
//! - Skip-to pattern (conjunction simulation)
//! - Galloping search (intersection with small list)

use std::hint::black_box;
use std::time::Instant;
use summa_core::structures::{
    EliasFanoPostingList, HorizontalBP128PostingList, OptP4DPostingList, PartitionedEFPostingList,
    RoaringPostingList, VerticalBP128PostingList,
};

/// Type alias for the complex measurement function tuple
type MeasureFns = (
    usize,
    Box<dyn Fn() -> u64>,
    Box<dyn Fn() -> u64>,
    Box<dyn Fn() -> u64>,
    Box<dyn Fn() -> u64>,
    Box<dyn Fn() -> u64>,
    Box<dyn Fn() -> u64>,
);

/// Access pattern types
#[derive(Clone, Copy, Debug)]
pub enum AccessPattern {
    /// Read all documents sequentially
    FullScan,
    /// Read first N documents then stop
    TopN(usize),
    /// Random seeks to specific doc_ids
    RandomSeek,
    /// Skip pattern: advance by fixed intervals (simulates AND query)
    SkipInterval,
    /// Galloping: seek to targets from another list (intersection)
    Galloping,
}

impl AccessPattern {
    pub fn name(&self) -> &'static str {
        match self {
            AccessPattern::FullScan => "full_scan",
            AccessPattern::TopN(_) => "top_100",
            AccessPattern::RandomSeek => "random_seek",
            AccessPattern::SkipInterval => "skip_interval",
            AccessPattern::Galloping => "galloping",
        }
    }
}

/// Simple LCG for reproducible random numbers
fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// Generate random seek targets from the posting list
fn generate_seek_targets(doc_ids: &[u32], count: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut targets = Vec::with_capacity(count);
    let n = doc_ids.len();

    for _ in 0..count {
        let idx = (next_rand(&mut state) as usize) % n;
        targets.push(doc_ids[idx]);
    }

    // Sort for better cache behavior in some patterns
    targets.sort_unstable();
    targets
}

/// Generate skip interval targets (every Nth doc)
fn generate_skip_targets(doc_ids: &[u32], interval: usize) -> Vec<u32> {
    doc_ids.iter().step_by(interval).copied().collect()
}

/// Measure access pattern performance for all formats
/// Returns rates in elements per microsecond
pub fn measure_access_pattern(
    doc_ids: &[u32],
    term_freqs: &[u32],
    pattern: AccessPattern,
    iterations: usize,
) -> [f64; 6] {
    let horiz = HorizontalBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let vert = VerticalBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let ef = EliasFanoPostingList::from_postings(doc_ids, term_freqs);
    let pef = PartitionedEFPostingList::from_postings(doc_ids, term_freqs);
    let roar = RoaringPostingList::from_postings(doc_ids, term_freqs);
    let opt_p4d = OptP4DPostingList::from_postings(doc_ids, term_freqs, 1.0);

    // Pre-generate targets for seek-based patterns
    let seek_targets = generate_seek_targets(doc_ids, 100, 42);
    let skip_targets = generate_skip_targets(doc_ids, 10);
    let gallop_targets = generate_seek_targets(doc_ids, 100, 123);

    let (
        elements_per_iter,
        measure_horiz,
        measure_vert,
        measure_ef,
        measure_pef,
        measure_roar,
        measure_opt_p4d,
    ): MeasureFns = match pattern {
        AccessPattern::FullScan => {
            let n = doc_ids.len();
            (
                n,
                Box::new(move || {
                    let mut iter = horiz.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = vert.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = ef.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = pef.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = roar.iterator();
                    iter.init();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = opt_p4d.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
            )
        }
        AccessPattern::TopN(n) => {
            let count = n.min(doc_ids.len());
            (
                count,
                Box::new(move || {
                    let mut iter = horiz.iterator();
                    let mut sum = 0u64;
                    for _ in 0..count {
                        if iter.doc() == u32::MAX {
                            break;
                        }
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = vert.iterator();
                    let mut sum = 0u64;
                    for _ in 0..count {
                        if iter.doc() == u32::MAX {
                            break;
                        }
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = ef.iterator();
                    let mut sum = 0u64;
                    for _ in 0..count {
                        if iter.doc() == u32::MAX {
                            break;
                        }
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = pef.iterator();
                    let mut sum = 0u64;
                    for _ in 0..count {
                        if iter.doc() == u32::MAX {
                            break;
                        }
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = roar.iterator();
                    iter.init();
                    let mut sum = 0u64;
                    for _ in 0..count {
                        if iter.doc() == u32::MAX {
                            break;
                        }
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = opt_p4d.iterator();
                    let mut sum = 0u64;
                    for _ in 0..count {
                        if iter.doc() == u32::MAX {
                            break;
                        }
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    sum
                }),
            )
        }
        AccessPattern::RandomSeek => {
            let targets = seek_targets.clone();
            let n = targets.len();
            let t1 = targets.clone();
            let t2 = targets.clone();
            let t3 = targets.clone();
            let t4 = targets.clone();
            let t5 = targets.clone();
            let t6 = targets;
            (
                n,
                Box::new(move || {
                    let mut iter = horiz.iterator();
                    let mut sum = 0u64;
                    for &target in &t1 {
                        let doc = iter.seek(target);
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = vert.iterator();
                    let mut sum = 0u64;
                    for &target in &t2 {
                        let doc = iter.seek(target);
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = ef.iterator();
                    let mut sum = 0u64;
                    for &target in &t3 {
                        let doc = iter.seek(target);
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = pef.iterator();
                    let mut sum = 0u64;
                    for &target in &t4 {
                        let doc = iter.seek(target);
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = roar.iterator();
                    iter.init();
                    let mut sum = 0u64;
                    for &target in &t5 {
                        let doc = iter.seek(target);
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = opt_p4d.iterator();
                    let mut sum = 0u64;
                    for &target in &t6 {
                        let doc = iter.seek(target);
                        sum += doc as u64;
                    }
                    sum
                }),
            )
        }
        AccessPattern::SkipInterval => {
            let targets = skip_targets.clone();
            let n = targets.len();
            let t1 = targets.clone();
            let t2 = targets.clone();
            let t3 = targets.clone();
            let t4 = targets.clone();
            let t5 = targets.clone();
            let t6 = targets;
            (
                n,
                Box::new(move || {
                    let mut iter = horiz.iterator();
                    let mut sum = 0u64;
                    for &target in &t1 {
                        let doc = iter.seek(target);
                        if doc == u32::MAX {
                            break;
                        }
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = vert.iterator();
                    let mut sum = 0u64;
                    for &target in &t2 {
                        let doc = iter.seek(target);
                        if doc == u32::MAX {
                            break;
                        }
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = ef.iterator();
                    let mut sum = 0u64;
                    for &target in &t3 {
                        let doc = iter.seek(target);
                        if doc == u32::MAX {
                            break;
                        }
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = pef.iterator();
                    let mut sum = 0u64;
                    for &target in &t4 {
                        let doc = iter.seek(target);
                        if doc == u32::MAX {
                            break;
                        }
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = roar.iterator();
                    iter.init();
                    let mut sum = 0u64;
                    for &target in &t5 {
                        let doc = iter.seek(target);
                        if doc == u32::MAX {
                            break;
                        }
                        sum += doc as u64;
                    }
                    sum
                }),
                Box::new(move || {
                    let mut iter = opt_p4d.iterator();
                    let mut sum = 0u64;
                    for &target in &t6 {
                        let doc = iter.seek(target);
                        if doc == u32::MAX {
                            break;
                        }
                        sum += doc as u64;
                    }
                    sum
                }),
            )
        }
        AccessPattern::Galloping => {
            let targets = gallop_targets.clone();
            let n = targets.len();
            let t1 = targets.clone();
            let t2 = targets.clone();
            let t3 = targets.clone();
            let t4 = targets.clone();
            let t5 = targets.clone();
            let t6 = targets;
            (
                n,
                Box::new(move || {
                    let mut iter = horiz.iterator();
                    let mut sum = 0u64;
                    let mut found = 0u64;
                    for &target in &t1 {
                        let doc = iter.seek(target);
                        if doc == target {
                            sum += doc as u64;
                            found += 1;
                        }
                    }
                    sum + found
                }),
                Box::new(move || {
                    let mut iter = vert.iterator();
                    let mut sum = 0u64;
                    let mut found = 0u64;
                    for &target in &t2 {
                        let doc = iter.seek(target);
                        if doc == target {
                            sum += doc as u64;
                            found += 1;
                        }
                    }
                    sum + found
                }),
                Box::new(move || {
                    let mut iter = ef.iterator();
                    let mut sum = 0u64;
                    let mut found = 0u64;
                    for &target in &t3 {
                        let doc = iter.seek(target);
                        if doc == target {
                            sum += doc as u64;
                            found += 1;
                        }
                    }
                    sum + found
                }),
                Box::new(move || {
                    let mut iter = pef.iterator();
                    let mut sum = 0u64;
                    let mut found = 0u64;
                    for &target in &t4 {
                        let doc = iter.seek(target);
                        if doc == target {
                            sum += doc as u64;
                            found += 1;
                        }
                    }
                    sum + found
                }),
                Box::new(move || {
                    let mut iter = roar.iterator();
                    iter.init();
                    let mut sum = 0u64;
                    let mut found = 0u64;
                    for &target in &t5 {
                        let doc = iter.seek(target);
                        if doc == target {
                            sum += doc as u64;
                            found += 1;
                        }
                    }
                    sum + found
                }),
                Box::new(move || {
                    let mut iter = opt_p4d.iterator();
                    let mut sum = 0u64;
                    let mut found = 0u64;
                    for &target in &t6 {
                        let doc = iter.seek(target);
                        if doc == target {
                            sum += doc as u64;
                            found += 1;
                        }
                    }
                    sum + found
                }),
            )
        }
    };

    // Measure each format
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(measure_horiz());
    }
    let horiz_time = start.elapsed().as_micros() as f64;
    let horiz_rate = (elements_per_iter * iterations) as f64 / horiz_time;

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(measure_vert());
    }
    let vert_time = start.elapsed().as_micros() as f64;
    let vert_rate = (elements_per_iter * iterations) as f64 / vert_time;

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(measure_ef());
    }
    let ef_time = start.elapsed().as_micros() as f64;
    let ef_rate = (elements_per_iter * iterations) as f64 / ef_time;

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(measure_pef());
    }
    let pef_time = start.elapsed().as_micros() as f64;
    let pef_rate = (elements_per_iter * iterations) as f64 / pef_time;

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(measure_roar());
    }
    let roar_time = start.elapsed().as_micros() as f64;
    let roar_rate = (elements_per_iter * iterations) as f64 / roar_time;

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(measure_opt_p4d());
    }
    let opt_p4d_time = start.elapsed().as_micros() as f64;
    let opt_p4d_rate = (elements_per_iter * iterations) as f64 / opt_p4d_time;

    [
        horiz_rate,
        vert_rate,
        ef_rate,
        pef_rate,
        roar_rate,
        opt_p4d_rate,
    ]
}
