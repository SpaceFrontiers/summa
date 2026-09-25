//! Pure paired-timing summaries for the wake-tier benchmark.

use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ExecutionOrder {
    NonDueThenDue,
    DueThenNonDue,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct PairedSample {
    pub(super) iteration: usize,
    pub(super) order: ExecutionOrder,
    pub(super) non_due_ms: f64,
    pub(super) due_ms: f64,
    pub(super) due_to_non_due_ratio: f64,
}

impl PairedSample {
    pub(super) fn new(
        iteration: usize,
        order: ExecutionOrder,
        non_due_ms: f64,
        due_ms: f64,
    ) -> Result<Self> {
        ensure!(
            non_due_ms.is_finite() && non_due_ms > 0.0,
            "non-due duration must be finite and positive"
        );
        ensure!(
            due_ms.is_finite() && due_ms > 0.0,
            "due duration must be finite and positive"
        );
        let due_to_non_due_ratio = due_ms / non_due_ms;
        ensure!(
            due_to_non_due_ratio.is_finite() && due_to_non_due_ratio > 0.0,
            "due/non-due ratio must be finite and positive"
        );
        Ok(Self {
            iteration,
            order,
            non_due_ms,
            due_ms,
            due_to_non_due_ratio,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct PairedTrial {
    pub(super) seed: u64,
    pub(super) samples: Vec<PairedSample>,
    pub(super) non_due_median_ms: f64,
    pub(super) due_median_ms: f64,
    pub(super) paired_ratio_median: f64,
}

impl PairedTrial {
    pub(super) fn new(seed: u64, samples: Vec<PairedSample>) -> Result<Self> {
        ensure!(!samples.is_empty(), "paired trial must contain samples");
        Ok(Self {
            seed,
            non_due_median_ms: median(samples.iter().map(|sample| sample.non_due_ms))?,
            due_median_ms: median(samples.iter().map(|sample| sample.due_ms))?,
            paired_ratio_median: median(samples.iter().map(|sample| sample.due_to_non_due_ratio))?,
            samples,
        })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct PairedSummary {
    pub(super) seed_count: usize,
    pub(super) sample_count: usize,
    pub(super) non_due_median_ms: f64,
    pub(super) due_median_ms: f64,
    pub(super) paired_ratio_median: f64,
    pub(super) paired_ratio_p95: f64,
    pub(super) median_of_seed_median_ratios: f64,
    pub(super) worst_seed_median_ratio: f64,
}

pub(super) fn summarize(trials: &[PairedTrial]) -> Result<PairedSummary> {
    ensure!(!trials.is_empty(), "summary requires paired trials");
    ensure!(
        trials.iter().all(|trial| !trial.samples.is_empty()),
        "summary cannot include an empty paired trial"
    );
    let sample_count = trials.iter().map(|trial| trial.samples.len()).sum();
    let paired_ratios = trials.iter().flat_map(|trial| {
        trial
            .samples
            .iter()
            .map(|sample| sample.due_to_non_due_ratio)
    });
    let seed_ratios = trials.iter().map(|trial| trial.paired_ratio_median);
    Ok(PairedSummary {
        seed_count: trials.len(),
        sample_count,
        non_due_median_ms: median(
            trials
                .iter()
                .flat_map(|trial| trial.samples.iter().map(|sample| sample.non_due_ms)),
        )?,
        due_median_ms: median(
            trials
                .iter()
                .flat_map(|trial| trial.samples.iter().map(|sample| sample.due_ms)),
        )?,
        paired_ratio_median: median(paired_ratios.clone())?,
        paired_ratio_p95: percentile(paired_ratios, 0.95)?,
        median_of_seed_median_ratios: median(seed_ratios.clone())?,
        worst_seed_median_ratio: seed_ratios.fold(f64::NEG_INFINITY, f64::max),
    })
}

fn median(values: impl IntoIterator<Item = f64>) -> Result<f64> {
    percentile(values, 0.5)
}

fn percentile(values: impl IntoIterator<Item = f64>, probability: f64) -> Result<f64> {
    ensure!(
        probability.is_finite() && (0.0..=1.0).contains(&probability),
        "percentile probability must be in [0, 1]"
    );
    let mut values = values.into_iter().collect::<Vec<_>>();
    ensure!(!values.is_empty(), "cannot summarize an empty sample");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "cannot summarize non-finite samples"
    );
    values.sort_by(f64::total_cmp);
    let rank = probability * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        Ok(values[lower])
    } else {
        let fraction = rank - lower as f64;
        Ok(values[lower] + (values[upper] - values[lower]) * fraction)
    }
}
