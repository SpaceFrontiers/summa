//! Reproducible acceptance gates for training candidates.
//!
//! Public and sealed suites share the same metric contract. Benchmark evidence
//! retains sealed case ids and scores for audit, but the promotion report
//! exposes only their aggregate gate. Suite examples themselves are never
//! copied into either artifact.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteVisibility {
    Public,
    Sealed,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkFamily {
    Pretraining,
    Summarization,
    Retrieval,
    RetrievalPlanning,
    Reasoning,
    Preference,
    VerifiableRl,
    ContinualRetention,
    LongContext,
    FactualIncorporation,
    Dreaming,
    Throughput,
    Resume,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkCase {
    pub id: String,
    /// Stable catalog identity used to bind anchor policy independently of the
    /// suite-local public or sealed case id.
    pub catalog_id: String,
    pub family: BenchmarkFamily,
    pub visibility: SuiteVisibility,
    /// Higher-is-better metric written by the evaluator.
    pub metric: String,
    pub stable_anchor: bool,
}

pub const ACCEPTANCE_SUITE_VERSION: u32 = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptanceSuite {
    pub version: u32,
    pub cases: Vec<BenchmarkCase>,
}

impl AcceptanceSuite {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == ACCEPTANCE_SUITE_VERSION,
            "unsupported acceptance suite version {}",
            self.version
        );
        ensure!(!self.cases.is_empty(), "acceptance suite has no cases");
        let mut ids = BTreeSet::new();
        let mut catalog_entries = BTreeSet::new();
        for case in &self.cases {
            ensure!(!case.id.trim().is_empty(), "acceptance case id is empty");
            ensure!(
                ids.insert(case.id.as_str()),
                "duplicate acceptance case `{}`",
                case.id
            );
            ensure!(
                !case.catalog_id.trim().is_empty(),
                "acceptance case `{}` catalog_id is empty",
                case.id
            );
            ensure!(
                catalog_entries.insert((case.catalog_id.as_str(), case.visibility)),
                "acceptance suite repeats {:?} catalog entry `{}`",
                case.visibility,
                case.catalog_id
            );
            ensure!(
                !case.metric.trim().is_empty(),
                "acceptance case `{}` metric is empty",
                case.id
            );
        }
        ensure!(
            self.cases
                .iter()
                .any(|case| case.visibility == SuiteVisibility::Sealed),
            "acceptance suite must contain a sealed case"
        );
        ensure!(
            self.cases
                .iter()
                .any(|case| case.visibility == SuiteVisibility::Public),
            "acceptance suite must contain a public case"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedRun {
    pub seed: u64,
    pub baseline: f64,
    pub candidate: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub case_id: String,
    pub pairs: Vec<PairedRun>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceComparison {
    pub version: u32,
    pub baseline_id: String,
    pub candidate_id: String,
    /// This is checked against the baseline identity derived from all
    /// capacity- and compute-matched comparison runs.
    pub strongest_baseline_id: String,
    /// Identity of the fixed harness that emitted all raw measurements below.
    pub measurement_evaluator_id: String,
    /// Matched raw observations. Throughput and p95 latency are recomputed by
    /// promotion; evidence cannot submit either aggregate directly.
    pub wake_trials: Vec<PairedWakeTrial>,
    /// Cycle zero is the initial candidate and must be followed by every
    /// completed sleep cycle without gaps. Capacity envelopes are derived from
    /// this complete series.
    pub candidate_capacity: Vec<CapacityObservation>,
    pub grouped_mm_parity: KernelParityEvidence,
    pub pytorch_parity: KernelParityEvidence,
    pub exact_resume: ExactResumeEvidence,
}

pub const RESOURCE_COMPARISON_VERSION: u32 = 2;
pub const ACCEPTANCE_POLICY_VERSION: u32 = 2;
pub const RESOURCE_EXECUTION_PROTOCOL_VERSION: u32 = 2;
const MAX_ACCEPTANCE_PAIRED_SEEDS: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WakeMeasurement {
    pub tokens: u64,
    pub elapsed_seconds: f64,
    pub request_latency_ms: Vec<f64>,
}

impl WakeMeasurement {
    fn validate(&self, name: &str) -> Result<()> {
        ensure!(self.tokens > 0, "{name} wake measurement has no tokens");
        ensure!(
            self.elapsed_seconds.is_finite() && self.elapsed_seconds > 0.0,
            "{name} wake measurement has invalid elapsed seconds"
        );
        ensure!(
            !self.request_latency_ms.is_empty(),
            "{name} wake measurement has no request latencies"
        );
        ensure!(
            self.request_latency_ms
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "{name} wake measurement has an invalid request latency"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedWakeTrial {
    /// Zero-based, contiguous trial ordinal.
    pub trial: u64,
    pub baseline: WakeMeasurement,
    pub candidate: WakeMeasurement,
}

impl PairedWakeTrial {
    fn validate(&self) -> Result<()> {
        self.baseline.validate("baseline")?;
        self.candidate.validate("candidate")?;
        ensure!(
            self.baseline.tokens == self.candidate.tokens,
            "wake trial {} did not use the same token workload",
            self.trial
        );
        ensure!(
            self.baseline.request_latency_ms.len() == self.candidate.request_latency_ms.len(),
            "wake trial {} did not use the same request workload",
            self.trial
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityObservation {
    /// Zero denotes the initial wake model; positive values denote the state
    /// after that many completed sleep cycles.
    pub completed_sleep_cycles: u64,
    pub routed_active_parameters: u64,
    pub stored_parameters: u64,
    pub stored_bytes: u64,
}

impl CapacityObservation {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.routed_active_parameters > 0
                && self.stored_parameters > 0
                && self.stored_bytes > 0,
            "capacity observation {} contains an empty measurement",
            self.completed_sleep_cycles
        );
        ensure!(
            self.routed_active_parameters <= self.stored_parameters,
            "capacity observation {} routes more parameters than it stores",
            self.completed_sleep_cycles
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParitySample {
    pub reference: f64,
    pub candidate: f64,
}

/// Raw outputs produced from a fixed reference fixture. Promotion recomputes
/// both error bounds; tolerances never come from this evidence.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParityEvidence {
    pub samples: Vec<KernelParitySample>,
}

impl KernelParityEvidence {
    fn validate(&self, name: &str) -> Result<()> {
        ensure!(
            !self.samples.is_empty(),
            "{name} parity evaluated no values"
        );
        ensure!(
            self.samples
                .iter()
                .all(|sample| { sample.reference.is_finite() && sample.candidate.is_finite() }),
            "{name} parity contains a non-finite sample"
        );
        Ok(())
    }

    fn maximum_errors(&self) -> (f64, f64) {
        self.samples.iter().fold(
            (0.0_f64, 0.0_f64),
            |(maximum_absolute, maximum_relative), sample| {
                let absolute = (sample.candidate - sample.reference).abs();
                let relative = if sample.reference == 0.0 {
                    if absolute == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    absolute / sample.reference.abs()
                };
                (
                    maximum_absolute.max(absolute),
                    maximum_relative.max(relative),
                )
            },
        )
    }
}

/// One file used by the exact-resume verifier. Relative paths resolve against
/// the resource-comparison artifact that declared them.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExactResumeArtifact {
    pub path: PathBuf,
}

impl ExactResumeArtifact {
    fn validate(&self, name: &str) -> Result<()> {
        ensure!(
            !self.path.as_os_str().is_empty(),
            "{name} artifact path is empty"
        );
        Ok(())
    }
}

/// Artifacts proving that an interrupted run and an uninterrupted reference
/// converged to byte-identical state and semantic progress from the same
/// checkpoint. Promotion reads every referenced file; a comparison that names
/// files which do not exist is never promotable.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExactResumeEvidence {
    pub interrupted_checkpoint: ExactResumeArtifact,
    pub uninterrupted_final_state: ExactResumeArtifact,
    pub resumed_final_state: ExactResumeArtifact,
    pub uninterrupted_metrics: ExactResumeArtifact,
    pub resumed_metrics: ExactResumeArtifact,
    pub interruption_step: u64,
    pub resumed_from_step: u64,
}

impl ExactResumeEvidence {
    fn validate(&self) -> Result<()> {
        for (name, artifact) in [
            ("interrupted checkpoint", &self.interrupted_checkpoint),
            ("uninterrupted final state", &self.uninterrupted_final_state),
            ("resumed final state", &self.resumed_final_state),
            ("uninterrupted metrics", &self.uninterrupted_metrics),
            ("resumed metrics", &self.resumed_metrics),
        ] {
            artifact.validate(name)?;
        }
        ensure!(
            self.interruption_step > 0,
            "resume evidence must interrupt after at least one step"
        );
        ensure!(
            self.resumed_from_step == self.interruption_step,
            "resumed step must equal the interrupted checkpoint step"
        );
        Ok(())
    }
}

impl ResourceComparison {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == RESOURCE_COMPARISON_VERSION,
            "unsupported resource-comparison version {}",
            self.version
        );
        ensure!(!self.baseline_id.trim().is_empty(), "baseline_id is empty");
        ensure!(
            !self.candidate_id.trim().is_empty(),
            "candidate_id is empty"
        );
        ensure!(
            self.baseline_id != self.candidate_id,
            "baseline and candidate identities must differ"
        );
        ensure!(
            !self.strongest_baseline_id.trim().is_empty(),
            "strongest_baseline_id is empty"
        );
        ensure!(
            !self.measurement_evaluator_id.trim().is_empty(),
            "resource measurement evaluator id is empty"
        );
        ensure!(
            !self.wake_trials.is_empty(),
            "resource evidence has no wake trials"
        );
        for (index, trial) in self.wake_trials.iter().enumerate() {
            ensure!(
                trial.trial == index as u64,
                "wake trial ordinals must be zero-based and contiguous; expected {index}, got {}",
                trial.trial
            );
            trial.validate()?;
        }
        ensure!(
            self.candidate_capacity.len() >= 2,
            "wake capacity evidence must contain cycle zero and at least one completed sleep cycle"
        );
        for (index, observation) in self.candidate_capacity.iter().enumerate() {
            ensure!(
                observation.completed_sleep_cycles == index as u64,
                "capacity observations must cover cycle zero and every completed cycle; expected {index}, got {}",
                observation.completed_sleep_cycles
            );
            observation.validate()?;
        }
        self.grouped_mm_parity.validate("grouped-mm")?;
        self.pytorch_parity.validate("PyTorch")?;
        self.exact_resume.validate()?;
        Ok(())
    }

    pub(crate) fn derived_measurements(&self) -> Result<DerivedResourceMeasurements> {
        self.validate()?;
        let baseline_tokens = self.wake_trials.iter().try_fold(0_u128, |total, trial| {
            total
                .checked_add(u128::from(trial.baseline.tokens))
                .ok_or_else(|| anyhow::anyhow!("baseline wake token total overflowed"))
        })?;
        let candidate_tokens = self.wake_trials.iter().try_fold(0_u128, |total, trial| {
            total
                .checked_add(u128::from(trial.candidate.tokens))
                .ok_or_else(|| anyhow::anyhow!("candidate wake token total overflowed"))
        })?;
        let baseline_elapsed = self
            .wake_trials
            .iter()
            .map(|trial| trial.baseline.elapsed_seconds)
            .sum::<f64>();
        let candidate_elapsed = self
            .wake_trials
            .iter()
            .map(|trial| trial.candidate.elapsed_seconds)
            .sum::<f64>();
        ensure!(
            baseline_elapsed.is_finite() && candidate_elapsed.is_finite(),
            "wake elapsed-time aggregate overflowed"
        );
        let mut baseline_latencies = self
            .wake_trials
            .iter()
            .flat_map(|trial| trial.baseline.request_latency_ms.iter().copied())
            .collect::<Vec<_>>();
        let mut candidate_latencies = self
            .wake_trials
            .iter()
            .flat_map(|trial| trial.candidate.request_latency_ms.iter().copied())
            .collect::<Vec<_>>();
        let initial = self
            .candidate_capacity
            .first()
            .expect("resource validation requires capacity observations");
        let minimum_routed_active_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.routed_active_parameters)
            .min()
            .expect("resource validation requires capacity observations");
        let maximum_routed_active_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.routed_active_parameters)
            .max()
            .expect("resource validation requires capacity observations");
        let maximum_stored_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_parameters)
            .max()
            .expect("resource validation requires capacity observations");
        let minimum_stored_parameters = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_parameters)
            .min()
            .expect("resource validation requires capacity observations");
        let maximum_stored_bytes = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_bytes)
            .max()
            .expect("resource validation requires capacity observations");
        let minimum_stored_bytes = self
            .candidate_capacity
            .iter()
            .map(|observation| observation.stored_bytes)
            .min()
            .expect("resource validation requires capacity observations");
        let baseline_wake_tokens_per_second = baseline_tokens as f64 / baseline_elapsed;
        let candidate_wake_tokens_per_second = candidate_tokens as f64 / candidate_elapsed;
        let baseline_wake_p95_ms = nearest_rank_p95(&mut baseline_latencies);
        let candidate_wake_p95_ms = nearest_rank_p95(&mut candidate_latencies);
        let throughput_log_ratios = self
            .wake_trials
            .iter()
            // Paired trials are required to process the same token workload,
            // so the throughput ratio is exactly baseline/candidate elapsed
            // time. Avoiding an unnecessary u64-to-f64 token conversion also
            // preserves precision for very large workloads.
            .map(|trial| trial.baseline.elapsed_seconds.ln() - trial.candidate.elapsed_seconds.ln())
            .collect::<Vec<_>>();
        let latency_log_ratios = self
            .wake_trials
            .iter()
            .map(|trial| {
                let mut baseline = trial.baseline.request_latency_ms.clone();
                let mut candidate = trial.candidate.request_latency_ms.clone();
                nearest_rank_p95(&mut candidate).ln() - nearest_rank_p95(&mut baseline).ln()
            })
            .collect::<Vec<_>>();
        // A single trial yields an unbounded interval, which would otherwise
        // surface downstream as an opaque overflow rather than the actual
        // problem. The policy floor is checked separately and is never lower.
        ensure!(
            self.wake_trials.len() >= 2,
            "wake performance needs at least 2 paired trials to form a confidence interval, got {}",
            self.wake_trials.len()
        );
        let (_, throughput_log_lower_95, _) = paired_interval_95(&throughput_log_ratios)?;
        let (_, _, latency_log_upper_95) = paired_interval_95(&latency_log_ratios)?;
        let wake_throughput_ratio_lower_95 = throughput_log_lower_95.exp();
        let wake_latency_ratio_upper_95 = latency_log_upper_95.exp();
        ensure!(
            [
                baseline_wake_tokens_per_second,
                candidate_wake_tokens_per_second,
                baseline_wake_p95_ms,
                candidate_wake_p95_ms,
                wake_throughput_ratio_lower_95,
                wake_latency_ratio_upper_95,
            ]
            .iter()
            .all(|measurement| measurement.is_finite() && *measurement > 0.0),
            "derived wake measurements overflowed or underflowed"
        );
        Ok(DerivedResourceMeasurements {
            wake_trials: self.wake_trials.len(),
            wake_latency_samples: baseline_latencies.len(),
            baseline_wake_tokens_per_second,
            candidate_wake_tokens_per_second,
            baseline_wake_p95_ms,
            candidate_wake_p95_ms,
            wake_throughput_ratio_lower_95,
            wake_latency_ratio_upper_95,
            initial_routed_active_parameters: initial.routed_active_parameters,
            initial_stored_parameters: initial.stored_parameters,
            initial_stored_bytes: initial.stored_bytes,
            completed_sleep_cycles_observed: self.candidate_capacity.len() - 1,
            minimum_routed_active_parameters,
            maximum_routed_active_parameters,
            minimum_stored_parameters,
            maximum_stored_parameters,
            minimum_stored_bytes,
            maximum_stored_bytes,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DerivedResourceMeasurements {
    pub wake_trials: usize,
    pub wake_latency_samples: usize,
    pub baseline_wake_tokens_per_second: f64,
    pub candidate_wake_tokens_per_second: f64,
    pub baseline_wake_p95_ms: f64,
    pub candidate_wake_p95_ms: f64,
    /// Lower endpoint of a paired, trial-level 95% interval over log
    /// throughput ratios. Trial-level inference avoids treating requests from
    /// one run as independent replicates.
    pub wake_throughput_ratio_lower_95: f64,
    /// Upper endpoint of the analogous paired interval over per-trial p95
    /// latency ratios.
    pub wake_latency_ratio_upper_95: f64,
    pub initial_routed_active_parameters: u64,
    pub initial_stored_parameters: u64,
    pub initial_stored_bytes: u64,
    pub completed_sleep_cycles_observed: usize,
    pub minimum_routed_active_parameters: u64,
    pub maximum_routed_active_parameters: u64,
    pub minimum_stored_parameters: u64,
    pub maximum_stored_parameters: u64,
    pub minimum_stored_bytes: u64,
    pub maximum_stored_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelParityPolicy {
    pub minimum_samples: usize,
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
}

impl KernelParityPolicy {
    fn validate(&self, name: &str) -> Result<()> {
        ensure!(
            self.minimum_samples > 0,
            "{name} policy requires no parity samples"
        );
        ensure!(
            self.maximum_absolute_error.is_finite()
                && self.maximum_absolute_error >= 0.0
                && self.maximum_relative_error.is_finite()
                && self.maximum_relative_error >= 0.0,
            "{name} policy contains an invalid error ceiling"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptancePolicy {
    pub version: u32,
    pub minimum_paired_seeds: usize,
    pub maximum_anchor_regression: f64,
    pub stable_anchor_catalog_ids: BTreeSet<String>,
    pub resource_evaluator_id: String,
    pub minimum_wake_trials: usize,
    pub minimum_wake_latency_samples: usize,
    pub minimum_wake_throughput_ratio: f64,
    pub maximum_wake_latency_ratio: f64,
    pub grouped_mm_parity: KernelParityPolicy,
    pub pytorch_parity: KernelParityPolicy,
}

impl Default for AcceptancePolicy {
    fn default() -> Self {
        Self {
            version: ACCEPTANCE_POLICY_VERSION,
            minimum_paired_seeds: 3,
            maximum_anchor_regression: 0.01,
            stable_anchor_catalog_ids: [
                "pretraining_causal".to_owned(),
                "synthetic_retention_smoke".to_owned(),
                "clinc_continual_learning".to_owned(),
                "banking77_continual_learning".to_owned(),
            ]
            .into_iter()
            .collect(),
            resource_evaluator_id: "summa-resource-evaluator".to_owned(),
            minimum_wake_trials: 3,
            minimum_wake_latency_samples: 3,
            minimum_wake_throughput_ratio: 0.95,
            maximum_wake_latency_ratio: 1.05,
            grouped_mm_parity: KernelParityPolicy {
                minimum_samples: 1024,
                maximum_absolute_error: 2e-5,
                maximum_relative_error: 2e-4,
            },
            pytorch_parity: KernelParityPolicy {
                minimum_samples: 1024,
                maximum_absolute_error: 2e-5,
                maximum_relative_error: 2e-4,
            },
        }
    }
}

impl AcceptancePolicy {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == ACCEPTANCE_POLICY_VERSION,
            "unsupported acceptance-policy version {}",
            self.version
        );
        ensure!(
            (3..=MAX_ACCEPTANCE_PAIRED_SEEDS).contains(&self.minimum_paired_seeds),
            "promotion requires between 3 and {MAX_ACCEPTANCE_PAIRED_SEEDS} paired seeds"
        );
        ensure!(
            self.maximum_anchor_regression.is_finite() && self.maximum_anchor_regression >= 0.0,
            "maximum anchor regression must be finite and non-negative"
        );
        ensure!(
            !self.stable_anchor_catalog_ids.is_empty()
                && self
                    .stable_anchor_catalog_ids
                    .iter()
                    .all(|id| !id.trim().is_empty()),
            "acceptance policy must prescribe at least one non-empty stable-anchor catalog id"
        );
        ensure!(
            !self.resource_evaluator_id.trim().is_empty(),
            "resource evaluator id is empty"
        );
        ensure!(
            self.minimum_wake_trials >= 3 && self.minimum_wake_latency_samples >= 3,
            "wake performance gates require at least three trials and latency samples"
        );
        ensure!(
            self.minimum_wake_throughput_ratio.is_finite()
                && (0.0..=1.0).contains(&self.minimum_wake_throughput_ratio),
            "minimum wake throughput ratio must be in [0, 1]"
        );
        ensure!(
            self.maximum_wake_latency_ratio.is_finite() && self.maximum_wake_latency_ratio >= 1.0,
            "maximum wake latency ratio must be at least one"
        );
        self.grouped_mm_parity.validate("grouped-mm")?;
        self.pytorch_parity.validate("PyTorch")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseGate {
    pub case_id: String,
    pub mean_delta: f64,
    pub lower_confidence_bound: f64,
    pub allowed_anchor_regression: Option<f64>,
    pub passed: bool,
}

/// Sealed examples, ids, scores, and per-case deltas are deliberately absent
/// from the promotion report. Only this aggregate crosses the promotion
/// boundary; the verified benchmark-run evidence remains auditable.
#[derive(Clone, Debug, Serialize)]
pub struct SealedGate {
    pub case_count: usize,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PromotionReport {
    pub accepted: bool,
    /// Public cases only.  Sealed ids, metrics, and deltas are never retained
    /// in the promotion report.
    pub cases: Vec<CaseGate>,
    pub sealed: SealedGate,
    pub resource_gates: BTreeMap<String, bool>,
}

#[derive(Clone, Debug)]
pub(crate) struct VerifiedPromotionContext<'a> {
    pub strongest_baseline_id: &'a str,
    pub baseline_id: &'a str,
    pub candidate_id: &'a str,
    pub capacity_matched: bool,
    pub active_capacity_matched: bool,
    pub fixed_gpu_hour_measured: bool,
    pub baseline_stored_parameters: u64,
    pub candidate_stored_parameters: u64,
    pub baseline_stored_bytes: u64,
    pub candidate_stored_bytes: u64,
    pub candidate_routed_active_parameters: u64,
    pub exact_resume_verified: bool,
}

pub(crate) fn evaluate_with_verified_context(
    suite: &AcceptanceSuite,
    results: &[CaseResult],
    resources: &ResourceComparison,
    policy: &AcceptancePolicy,
    context: &VerifiedPromotionContext<'_>,
) -> Result<PromotionReport> {
    suite.validate()?;
    policy.validate()?;
    let measurements = resources.derived_measurements()?;

    ensure!(
        context.baseline_id == resources.baseline_id
            && context.candidate_id == resources.candidate_id,
        "resource evidence target identities do not match the benchmark run"
    );
    ensure!(
        resources.measurement_evaluator_id == policy.resource_evaluator_id,
        "resource evidence was not produced by the policy-named evaluator"
    );
    ensure!(
        measurements.wake_trials >= policy.minimum_wake_trials,
        "resource evidence has {} wake trials; policy requires {}",
        measurements.wake_trials,
        policy.minimum_wake_trials
    );
    ensure!(
        measurements.wake_latency_samples >= policy.minimum_wake_latency_samples,
        "resource evidence has {} paired wake latency samples; policy requires {}",
        measurements.wake_latency_samples,
        policy.minimum_wake_latency_samples
    );
    ensure!(
        resources.grouped_mm_parity.samples.len() >= policy.grouped_mm_parity.minimum_samples,
        "grouped-mm parity evidence has {} samples; policy requires {}",
        resources.grouped_mm_parity.samples.len(),
        policy.grouped_mm_parity.minimum_samples
    );
    ensure!(
        resources.pytorch_parity.samples.len() >= policy.pytorch_parity.minimum_samples,
        "PyTorch parity evidence has {} samples; policy requires {}",
        resources.pytorch_parity.samples.len(),
        policy.pytorch_parity.minimum_samples
    );

    let suite_catalog_ids = suite
        .cases
        .iter()
        .map(|case| case.catalog_id.as_str())
        .collect::<BTreeSet<_>>();
    for catalog_id in &policy.stable_anchor_catalog_ids {
        ensure!(
            suite_catalog_ids.contains(catalog_id.as_str()),
            "policy-prescribed stable-anchor catalog `{catalog_id}` is absent from the suite"
        );
    }

    let mut result_by_id = BTreeMap::new();
    for result in results {
        ensure!(
            result_by_id
                .insert(result.case_id.as_str(), result)
                .is_none(),
            "duplicate acceptance result `{}`",
            result.case_id
        );
    }
    ensure!(
        result_by_id.len() == suite.cases.len(),
        "acceptance results do not exactly match the suite"
    );
    let mut cases = Vec::with_capacity(suite.cases.len());
    let mut sealed_case_count = 0usize;
    let mut sealed_passed = true;
    let mut expected_seeds: Option<Vec<u64>> = None;
    for case in &suite.cases {
        let policy_anchor = policy
            .stable_anchor_catalog_ids
            .contains(case.catalog_id.as_str());
        ensure!(
            case.stable_anchor == policy_anchor,
            "acceptance case `{}` stable_anchor={} conflicts with policy for catalog `{}`",
            case.id,
            case.stable_anchor,
            case.catalog_id
        );
        let result = result_by_id
            .get(case.id.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing acceptance result `{}`", case.id))?;
        ensure!(
            (policy.minimum_paired_seeds..=MAX_ACCEPTANCE_PAIRED_SEEDS)
                .contains(&result.pairs.len()),
            "acceptance case `{}` has {} paired seeds; need {}..={MAX_ACCEPTANCE_PAIRED_SEEDS}",
            case.id,
            result.pairs.len(),
            policy.minimum_paired_seeds
        );
        let unique_seeds = result
            .pairs
            .iter()
            .map(|pair| pair.seed)
            .collect::<BTreeSet<_>>();
        ensure!(
            unique_seeds.len() == result.pairs.len(),
            "acceptance case `{}` repeats a seed",
            case.id
        );
        let seeds = result
            .pairs
            .iter()
            .map(|pair| pair.seed)
            .collect::<Vec<_>>();
        ensure!(
            seeds.windows(2).all(|pair| pair[0] < pair[1]),
            "acceptance case `{}` seeds are not strictly increasing",
            case.id
        );
        if let Some(expected) = &expected_seeds {
            ensure!(
                &seeds == expected,
                "acceptance case `{}` does not use the common paired seeds",
                case.id
            );
        } else {
            expected_seeds = Some(seeds);
        }
        ensure!(
            result
                .pairs
                .iter()
                .all(|pair| pair.baseline.is_finite() && pair.candidate.is_finite()),
            "acceptance case `{}` contains non-finite metrics",
            case.id
        );
        let deltas = result
            .pairs
            .iter()
            .map(|pair| pair.candidate - pair.baseline)
            .collect::<Vec<_>>();
        ensure!(
            deltas.iter().all(|delta| delta.is_finite()),
            "acceptance case `{}` metric deltas overflowed",
            case.id
        );
        ensure_representable_metrics(
            &case.id,
            &result
                .pairs
                .iter()
                .flat_map(|pair| [pair.baseline, pair.candidate])
                .collect::<Vec<_>>(),
        )?;
        let (mean_delta, lower_confidence_bound) = paired_lower_95(&deltas)?;
        // The tolerance comes from the addressed policy alone. Widening it by a
        // spread computed from the submitted pairs would let evidence choose its
        // own tolerance; paired-sample noise is already handled conservatively by
        // gating on the lower confidence bound rather than the mean.
        let allowed_anchor_regression = case
            .stable_anchor
            .then_some(policy.maximum_anchor_regression);
        let passed = if case.stable_anchor {
            lower_confidence_bound >= -allowed_anchor_regression.unwrap()
        } else {
            lower_confidence_bound > 0.0
        };
        if case.visibility == SuiteVisibility::Sealed {
            sealed_case_count += 1;
            sealed_passed &= passed;
        } else {
            cases.push(CaseGate {
                case_id: case.id.clone(),
                mean_delta,
                lower_confidence_bound,
                allowed_anchor_regression,
                passed,
            });
        }
    }

    let throughput_ratio = measurements.candidate_wake_tokens_per_second
        / measurements.baseline_wake_tokens_per_second;
    let latency_ratio = measurements.candidate_wake_p95_ms / measurements.baseline_wake_p95_ms;
    ensure!(
        throughput_ratio.is_finite()
            && throughput_ratio > 0.0
            && latency_ratio.is_finite()
            && latency_ratio > 0.0,
        "wake performance ratios overflowed or underflowed"
    );
    let grouped_mm_errors = resources.grouped_mm_parity.maximum_errors();
    let pytorch_errors = resources.pytorch_parity.maximum_errors();
    let resource_gates = BTreeMap::from([
        (
            "strongest_matched_baseline".into(),
            context.baseline_id == context.strongest_baseline_id,
        ),
        ("capacity_matched".into(), context.capacity_matched),
        (
            "active_capacity_matched".into(),
            context.active_capacity_matched,
        ),
        (
            "fixed_gpu_hour_measured".into(),
            context.fixed_gpu_hour_measured,
        ),
        ("exact_resume".into(), context.exact_resume_verified),
        (
            "constant_active_parameters".into(),
            measurements.completed_sleep_cycles_observed > 0
                && measurements.minimum_routed_active_parameters
                    == measurements.initial_routed_active_parameters
                && measurements.maximum_routed_active_parameters
                    == measurements.initial_routed_active_parameters
                && measurements.initial_routed_active_parameters
                    == context.candidate_routed_active_parameters,
        ),
        (
            "bounded_stored_parameters".into(),
            measurements.initial_stored_parameters == context.candidate_stored_parameters
                && measurements.minimum_stored_parameters == measurements.initial_stored_parameters
                && measurements.maximum_stored_parameters == measurements.initial_stored_parameters
                && context.candidate_stored_parameters <= context.baseline_stored_parameters,
        ),
        (
            "bounded_stored_bytes".into(),
            measurements.initial_stored_bytes == context.candidate_stored_bytes
                && measurements.minimum_stored_bytes == measurements.initial_stored_bytes
                && measurements.maximum_stored_bytes == measurements.initial_stored_bytes
                && context.candidate_stored_bytes <= context.baseline_stored_bytes,
        ),
        (
            "grouped_mm_parity".into(),
            grouped_mm_errors.0 <= policy.grouped_mm_parity.maximum_absolute_error
                && grouped_mm_errors.1 <= policy.grouped_mm_parity.maximum_relative_error,
        ),
        (
            "pytorch_parity".into(),
            pytorch_errors.0 <= policy.pytorch_parity.maximum_absolute_error
                && pytorch_errors.1 <= policy.pytorch_parity.maximum_relative_error,
        ),
        (
            "wake_throughput".into(),
            throughput_ratio >= policy.minimum_wake_throughput_ratio
                && measurements.wake_throughput_ratio_lower_95
                    >= policy.minimum_wake_throughput_ratio,
        ),
        (
            "wake_latency".into(),
            latency_ratio <= policy.maximum_wake_latency_ratio
                && measurements.wake_latency_ratio_upper_95 <= policy.maximum_wake_latency_ratio,
        ),
    ]);
    let accepted = cases.iter().all(|case| case.passed)
        && sealed_passed
        && resource_gates.values().all(|gate| *gate);
    Ok(PromotionReport {
        accepted,
        cases,
        sealed: SealedGate {
            case_count: sealed_case_count,
            passed: sealed_passed,
        },
        resource_gates,
    })
}

fn nearest_rank_p95(values: &mut [f64]) -> f64 {
    let rank = (95 * values.len()).div_ceil(100);
    let index = rank.saturating_sub(1);
    let (_, value, _) = values.select_nth_unstable_by(index, f64::total_cmp);
    *value
}

/// Reject raw metrics whose magnitudes cannot be aggregated, so extreme but
/// individually finite scores fail closed instead of gating on a delta that
/// happens to cancel.
fn ensure_representable_metrics(case_id: &str, values: &[f64]) -> Result<()> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    ensure!(
        mean.is_finite(),
        "acceptance case `{case_id}` metric mean overflowed"
    );
    let dispersion = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    ensure!(
        dispersion.is_finite(),
        "acceptance case `{case_id}` metric variance overflowed"
    );
    Ok(())
}

fn paired_lower_95(deltas: &[f64]) -> Result<(f64, f64)> {
    let (mean, lower, _) = paired_interval_95(deltas)?;
    Ok((mean, lower))
}

fn paired_interval_95(values: &[f64]) -> Result<(f64, f64, f64)> {
    ensure!(!values.is_empty(), "paired confidence interval is empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "paired confidence interval contains a non-finite delta"
    );
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    ensure!(
        mean.is_finite(),
        "paired confidence-interval mean overflowed"
    );
    if count == 1 {
        return Ok((mean, f64::NEG_INFINITY, f64::INFINITY));
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (count - 1) as f64;
    ensure!(
        variance.is_finite() && variance >= 0.0,
        "paired confidence-interval variance overflowed"
    );
    let standard_error = (variance / count as f64).sqrt();
    let half_width = t_critical_95(count - 1) * standard_error;
    let lower = mean - half_width;
    let upper = mean + half_width;
    ensure!(
        standard_error.is_finite() && lower.is_finite() && upper.is_finite(),
        "paired confidence interval overflowed"
    );
    Ok((mean, lower, upper))
}

/// Two-sided 95% Student-t critical values; the lower bound therefore uses
/// the conservative half-width expected by paired experiment reports.
fn t_critical_95(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    // The normal 1.96 limit is approached only asymptotically. Jumping from
    // t(30)=2.042 straight to 1.96 at df=31 makes the interval spuriously
    // narrower exactly when another paired seed is added. Keep the last
    // tabulated value for larger samples: it is monotone and conservative,
    // while avoiding a statistical dependency in the promotion boundary.
    TABLE
        .get(df.saturating_sub(1))
        .copied()
        .unwrap_or(*TABLE.last().expect("Student-t table is non-empty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_interval_does_not_narrow_discontinuously_after_t_table() {
        assert_eq!(t_critical_95(30), 2.042);
        assert_eq!(t_critical_95(31), 2.042);
        assert_eq!(t_critical_95(10_000), 2.042);
    }

    fn suite() -> AcceptanceSuite {
        AcceptanceSuite {
            version: ACCEPTANCE_SUITE_VERSION,
            cases: vec![
                BenchmarkCase {
                    id: "quality".into(),
                    catalog_id: "quality".into(),
                    family: BenchmarkFamily::Reasoning,
                    visibility: SuiteVisibility::Public,
                    metric: "accuracy".into(),
                    stable_anchor: false,
                },
                BenchmarkCase {
                    id: "retention".into(),
                    catalog_id: "retention".into(),
                    family: BenchmarkFamily::ContinualRetention,
                    visibility: SuiteVisibility::Sealed,
                    metric: "accuracy".into(),
                    stable_anchor: true,
                },
            ],
        }
    }

    fn policy() -> AcceptancePolicy {
        AcceptancePolicy {
            stable_anchor_catalog_ids: ["retention".to_owned()].into_iter().collect(),
            ..AcceptancePolicy::default()
        }
    }

    fn resources() -> ResourceComparison {
        let parity_samples = || {
            (0..1024)
                .map(|_| KernelParitySample {
                    reference: 1.0,
                    candidate: 1.0 + 1e-5,
                })
                .collect()
        };
        ResourceComparison {
            version: RESOURCE_COMPARISON_VERSION,
            baseline_id: "static-moe".into(),
            candidate_id: "sleep-candidate".into(),
            strongest_baseline_id: "static-moe".into(),
            measurement_evaluator_id: "summa-resource-evaluator".into(),
            wake_trials: (0..3)
                .map(|trial| PairedWakeTrial {
                    trial,
                    baseline: WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 10.0,
                        request_latency_ms: vec![10.0],
                    },
                    candidate: WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 1_000.0 / 96.0,
                        request_latency_ms: vec![10.4],
                    },
                })
                .collect(),
            candidate_capacity: (0..=4)
                .map(|completed_sleep_cycles| CapacityObservation {
                    completed_sleep_cycles,
                    routed_active_parameters: 80,
                    stored_parameters: 100,
                    stored_bytes: 1_000,
                })
                .collect(),
            grouped_mm_parity: KernelParityEvidence {
                samples: parity_samples(),
            },
            pytorch_parity: KernelParityEvidence {
                samples: parity_samples(),
            },
            exact_resume: ExactResumeEvidence {
                interrupted_checkpoint: ExactResumeArtifact {
                    path: "interrupted/generation-manifest.json".into(),
                },
                uninterrupted_final_state: ExactResumeArtifact {
                    path: "uninterrupted/generation-manifest.json".into(),
                },
                resumed_final_state: ExactResumeArtifact {
                    path: "resumed/generation-manifest.json".into(),
                },
                uninterrupted_metrics: ExactResumeArtifact {
                    path: "uninterrupted/metrics.jsonl".into(),
                },
                resumed_metrics: ExactResumeArtifact {
                    path: "resumed/metrics.jsonl".into(),
                },
                interruption_step: 17,
                resumed_from_step: 17,
            },
        }
    }

    fn verified_context<'a>(resources: &'a ResourceComparison) -> VerifiedPromotionContext<'a> {
        VerifiedPromotionContext {
            strongest_baseline_id: &resources.strongest_baseline_id,
            baseline_id: &resources.baseline_id,
            candidate_id: &resources.candidate_id,
            capacity_matched: true,
            active_capacity_matched: true,
            fixed_gpu_hour_measured: true,
            baseline_stored_parameters: 100,
            candidate_stored_parameters: 100,
            baseline_stored_bytes: 1_000,
            candidate_stored_bytes: 1_000,
            candidate_routed_active_parameters: 80,
            exact_resume_verified: true,
        }
    }

    #[test]
    fn promotion_requires_confident_gain_and_stable_anchors() {
        let results = vec![
            CaseResult {
                case_id: "quality".into(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.4,
                        candidate: 0.5,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.4,
                        candidate: 0.51,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.4,
                        candidate: 0.49,
                    },
                ],
            },
            CaseResult {
                case_id: "retention".into(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.8,
                        candidate: 0.795,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.8,
                        candidate: 0.796,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.8,
                        candidate: 0.794,
                    },
                ],
            },
        ];
        let resources = resources();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(report.accepted);
        assert!(report.cases.iter().all(|case| case.passed));
        assert_eq!(report.sealed.case_count, 1);
        assert!(report.sealed.passed);
        let report_json = serde_json::to_string(&report).unwrap();
        assert!(!report_json.contains("retention"));
    }

    #[test]
    fn stable_anchor_tolerance_ignores_the_spread_of_submitted_baselines() {
        let results = vec![
            CaseResult {
                case_id: "quality".into(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.4,
                        candidate: 0.5,
                    })
                    .collect(),
            },
            CaseResult {
                case_id: "retention".into(),
                // Widely spread baselines with a candidate that is consistently
                // worse by 0.15 — far beyond the 0.01 policy tolerance.
                pairs: [0.6, 0.8, 1.0]
                    .into_iter()
                    .enumerate()
                    .map(|(index, baseline)| PairedRun {
                        seed: (index + 1) as u64,
                        baseline,
                        candidate: baseline - 0.15,
                    })
                    .collect(),
            },
        ];
        let resources = resources();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();

        assert!(!report.sealed.passed);
        assert!(!report.accepted);
    }

    #[test]
    fn stable_anchor_uses_its_confidence_bound_not_only_the_mean() {
        let results = vec![
            CaseResult {
                case_id: "quality".into(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.4,
                        candidate: 0.5,
                    })
                    .collect(),
            },
            CaseResult {
                case_id: "retention".into(),
                pairs: [0.01, 0.01, -0.015]
                    .into_iter()
                    .enumerate()
                    .map(|(index, delta)| PairedRun {
                        seed: (index + 1) as u64,
                        baseline: 0.8,
                        candidate: 0.8 + delta,
                    })
                    .collect(),
            },
        ];
        let resources = resources();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();

        assert!(!report.sealed.passed);
        assert!(!report.accepted);
    }

    #[test]
    fn finite_samples_with_overflowing_statistics_fail_closed() {
        let quality = CaseResult {
            case_id: "quality".into(),
            pairs: [1, 2, 3]
                .into_iter()
                .map(|seed| PairedRun {
                    seed,
                    baseline: 0.0,
                    candidate: 1.0,
                })
                .collect(),
        };
        let retention = CaseResult {
            case_id: "retention".into(),
            pairs: [1, 2, 3]
                .into_iter()
                .map(|seed| PairedRun {
                    seed,
                    baseline: f64::MAX,
                    candidate: f64::MAX,
                })
                .collect(),
        };
        let resources = resources();
        let error = evaluate_with_verified_context(
            &suite(),
            &[quality, retention],
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("mean overflowed"), "{error}");

        let overflowed_delta = CaseResult {
            case_id: "quality".into(),
            pairs: [1, 2, 3]
                .into_iter()
                .map(|seed| PairedRun {
                    seed,
                    baseline: -f64::MAX,
                    candidate: f64::MAX,
                })
                .collect(),
        };
        let stable = CaseResult {
            case_id: "retention".into(),
            pairs: [1, 2, 3]
                .into_iter()
                .map(|seed| PairedRun {
                    seed,
                    baseline: 1.0,
                    candidate: 1.0,
                })
                .collect(),
        };
        let error = evaluate_with_verified_context(
            &suite(),
            &[overflowed_delta, stable],
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("metric deltas overflowed"), "{error}");
    }

    #[test]
    fn finite_wake_measurements_with_nonfinite_ratios_fail_closed() {
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.5,
                        candidate: if case.stable_anchor { 0.5 } else { 0.6 },
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let mut resources = resources();
        for trial in &mut resources.wake_trials {
            trial.baseline.tokens = 1;
            trial.baseline.elapsed_seconds = 1e289;
            trial.candidate.tokens = 1;
            trial.candidate.elapsed_seconds = 1e-20;
        }

        let error = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("overflowed or underflowed"), "{error}");
    }

    #[test]
    fn active_compute_growth_rejects_candidate() {
        let mut resources = resources();
        resources
            .candidate_capacity
            .last_mut()
            .unwrap()
            .routed_active_parameters += 1;
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.accepted);
        assert!(!report.resource_gates["constant_active_parameters"]);
    }

    #[test]
    fn first_activation_compute_growth_rejects_candidate() {
        let mut resources = resources();
        resources.candidate_capacity[0].routed_active_parameters -= 1;
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: vec![
                    PairedRun {
                        seed: 1,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 2,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                    PairedRun {
                        seed: 3,
                        baseline: 0.0,
                        candidate: 1.0,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.accepted);
        assert!(!report.resource_gates["constant_active_parameters"]);
    }

    #[test]
    fn topology_or_storage_shrink_across_cycles_rejects_candidate() {
        let mut resources = resources();
        let last = resources.candidate_capacity.last_mut().unwrap();
        last.stored_parameters -= 1;
        last.stored_bytes -= 1;
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.resource_gates["bounded_stored_parameters"]);
        assert!(!report.resource_gates["bounded_stored_bytes"]);
        assert!(!report.accepted);
    }

    #[test]
    fn wake_capacity_envelope_evidence_is_mandatory() {
        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .remove("candidate_capacity");
        assert!(
            serde_json::from_value::<ResourceComparison>(encoded).is_err(),
            "promotion evidence without capacity observations must fail closed"
        );

        let mut incomplete = resources();
        incomplete.candidate_capacity.truncate(1);
        assert!(incomplete.validate().is_err());
    }

    #[test]
    fn duplicate_seeds_fail_loudly() {
        let results = vec![CaseResult {
            case_id: "quality".into(),
            pairs: vec![
                PairedRun {
                    seed: 1,
                    baseline: 0.0,
                    candidate: 1.0,
                },
                PairedRun {
                    seed: 1,
                    baseline: 0.0,
                    candidate: 1.0,
                },
                PairedRun {
                    seed: 2,
                    baseline: 0.0,
                    candidate: 1.0,
                },
            ],
        }];
        let resources = resources();
        assert!(
            evaluate_with_verified_context(
                &suite(),
                &results,
                &resources,
                &policy(),
                &verified_context(&resources),
            )
            .is_err()
        );
    }

    #[test]
    fn wake_confidence_policy_requires_three_independent_trials() {
        let mut trial_policy = policy();
        trial_policy.minimum_wake_trials = 2;
        let error = trial_policy.validate().unwrap_err().to_string();
        assert!(error.contains("at least three trials"), "{error}");

        let mut latency_policy = policy();
        latency_policy.minimum_wake_latency_samples = 2;
        let error = latency_policy.validate().unwrap_err().to_string();
        assert!(error.contains("latency samples"), "{error}");

        let mut unbounded_seed_policy = policy();
        unbounded_seed_policy.minimum_paired_seeds = MAX_ACCEPTANCE_PAIRED_SEEDS + 1;
        let error = unbounded_seed_policy.validate().unwrap_err().to_string();
        assert!(error.contains("paired seeds"), "{error}");
    }

    #[test]
    fn parity_memory_and_resume_are_measured_gates() {
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let mut resources = resources();
        resources.grouped_mm_parity.samples[0].candidate = 2.0;
        resources
            .candidate_capacity
            .last_mut()
            .unwrap()
            .stored_bytes += 1;
        let mut context = verified_context(&resources);
        context.exact_resume_verified = false;
        let report =
            evaluate_with_verified_context(&suite(), &results, &resources, &policy(), &context)
                .unwrap();
        assert!(!report.resource_gates["grouped_mm_parity"]);
        assert!(!report.resource_gates["bounded_stored_bytes"]);
        assert!(!report.resource_gates["exact_resume"]);
        assert!(!report.accepted);
    }

    #[test]
    fn exact_resume_requires_complete_artifact_references() {
        let mut exact = resources().exact_resume;
        exact.validate().unwrap();
        exact.resumed_metrics.path = PathBuf::new();
        assert!(exact.validate().is_err());

        let mut exact = resources().exact_resume;
        exact.resumed_from_step += 1;
        assert!(exact.validate().is_err());

        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded.as_object_mut().unwrap().remove("exact_resume");
        assert!(serde_json::from_value::<ResourceComparison>(encoded).is_err());
    }

    #[test]
    fn suite_authors_cannot_expand_the_policy_anchor_set() {
        let mut forged_suite = suite();
        forged_suite.cases[0].stable_anchor = true;
        let results = forged_suite
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 1.0,
                        candidate: 0.999,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let resources = resources();
        let error = evaluate_with_verified_context(
            &forged_suite,
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("conflicts with policy"), "{error}");
    }

    #[test]
    fn resource_evidence_cannot_choose_parity_tolerances_or_submit_aggregates() {
        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded["grouped_mm_parity"]["absolute_tolerance"] = serde_json::json!(1e100);
        assert!(serde_json::from_value::<ResourceComparison>(encoded).is_err());

        let mut encoded = serde_json::to_value(resources()).unwrap();
        encoded["candidate_wake_tokens_per_second"] = serde_json::json!(1e100);
        assert!(serde_json::from_value::<ResourceComparison>(encoded).is_err());
    }

    #[test]
    fn wake_performance_is_recomputed_from_raw_trials() {
        let mut resources = resources();
        for trial in &mut resources.wake_trials {
            trial.candidate.elapsed_seconds = 20.0;
            trial.candidate.request_latency_ms = vec![20.0];
        }
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.resource_gates["wake_throughput"]);
        assert!(!report.resource_gates["wake_latency"]);
    }

    #[test]
    fn wake_performance_requires_paired_trial_confidence() {
        let mut resources = resources();
        for (trial, (elapsed, latency)) in
            resources
                .wake_trials
                .iter_mut()
                .zip([(9.0, 9.0), (9.0, 9.0), (13.0, 10.4)])
        {
            trial.candidate.elapsed_seconds = elapsed;
            trial.candidate.request_latency_ms = vec![latency];
        }
        let results = suite()
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.id.clone(),
                pairs: [1, 2, 3]
                    .into_iter()
                    .map(|seed| PairedRun {
                        seed,
                        baseline: 0.0,
                        candidate: 1.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let measurements = resources.derived_measurements().unwrap();
        assert!(
            measurements.candidate_wake_tokens_per_second
                / measurements.baseline_wake_tokens_per_second
                >= policy().minimum_wake_throughput_ratio
        );
        assert!(
            measurements.candidate_wake_p95_ms / measurements.baseline_wake_p95_ms
                <= policy().maximum_wake_latency_ratio
        );

        let report = evaluate_with_verified_context(
            &suite(),
            &results,
            &resources,
            &policy(),
            &verified_context(&resources),
        )
        .unwrap();
        assert!(!report.resource_gates["wake_throughput"]);
        assert!(!report.resource_gates["wake_latency"]);
        assert!(!report.accepted);
    }

    #[test]
    fn derived_wake_measurements_reject_finite_arithmetic_overflow() {
        let mut resources = resources();
        for trial in &mut resources.wake_trials {
            trial.baseline.elapsed_seconds = f64::MIN_POSITIVE;
        }
        let error = resources.derived_measurements().unwrap_err().to_string();
        assert!(error.contains("derived wake measurements"), "{error}");
    }

    #[test]
    fn v1_acceptance_artifacts_fail_closed() {
        let mut old_suite = suite();
        old_suite.version = 1;
        assert!(old_suite.validate().is_err());

        let mut old_resources = resources();
        old_resources.version = 1;
        assert!(old_resources.validate().is_err());

        let mut old_policy = policy();
        old_policy.version = 1;
        assert!(old_policy.validate().is_err());
    }
}
