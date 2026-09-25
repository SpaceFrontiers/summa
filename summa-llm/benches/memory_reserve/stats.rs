//! Pure paired-statistics support for the memory reserve benchmark.

use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOrder {
    ReferenceThenCandidate,
    CandidateThenReference,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairedSample {
    pub iteration: usize,
    pub order: ExecutionOrder,
    pub reference_ms: f64,
    pub candidate_ms: f64,
    pub candidate_to_reference_ratio: f64,
}

impl PairedSample {
    pub fn new(
        iteration: usize,
        order: ExecutionOrder,
        reference_ms: f64,
        candidate_ms: f64,
    ) -> Result<Self> {
        ensure!(
            reference_ms.is_finite() && reference_ms > 0.0,
            "reference duration must be finite and positive"
        );
        ensure!(
            candidate_ms.is_finite() && candidate_ms > 0.0,
            "candidate duration must be finite and positive"
        );
        let ratio = candidate_ms / reference_ms;
        ensure!(
            ratio.is_finite() && ratio > 0.0,
            "candidate/reference timing ratio must be finite and positive"
        );
        Ok(Self {
            iteration,
            order,
            reference_ms,
            candidate_ms,
            candidate_to_reference_ratio: ratio,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PairedTrial {
    pub seed: u64,
    pub samples: Vec<PairedSample>,
    pub reference_median_ms: f64,
    pub candidate_median_ms: f64,
    pub paired_ratio_median: f64,
}

impl PairedTrial {
    pub fn new(seed: u64, samples: Vec<PairedSample>) -> Result<Self> {
        ensure!(!samples.is_empty(), "paired trial must contain samples");
        let reference_median_ms = median(samples.iter().map(|sample| sample.reference_ms))?;
        let candidate_median_ms = median(samples.iter().map(|sample| sample.candidate_ms))?;
        let paired_ratio_median = median(
            samples
                .iter()
                .map(|sample| sample.candidate_to_reference_ratio),
        )?;
        Ok(Self {
            seed,
            samples,
            reference_median_ms,
            candidate_median_ms,
            paired_ratio_median,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct PairedSummary {
    pub seed_count: usize,
    pub sample_count: usize,
    pub reference_median_ms: f64,
    pub candidate_median_ms: f64,
    pub paired_ratio_median: f64,
    pub paired_ratio_p95: f64,
    pub median_of_seed_median_ratios: f64,
    pub worst_seed_median_ratio: f64,
}

pub fn summarize(trials: &[PairedTrial]) -> Result<PairedSummary> {
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
            .map(|sample| sample.candidate_to_reference_ratio)
    });
    let seed_ratios = trials.iter().map(|trial| trial.paired_ratio_median);
    Ok(PairedSummary {
        seed_count: trials.len(),
        sample_count,
        reference_median_ms: median(
            trials
                .iter()
                .flat_map(|trial| trial.samples.iter().map(|sample| sample.reference_ms)),
        )?,
        candidate_median_ms: median(
            trials
                .iter()
                .flat_map(|trial| trial.samples.iter().map(|sample| sample.candidate_ms)),
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
