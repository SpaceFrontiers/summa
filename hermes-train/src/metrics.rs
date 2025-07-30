//! Versioned, append-only training metrics for the W&B sidecar.
//!
//! Records deliberately use a closed schema instead of an arbitrary map.  The
//! common run/phase coordinates stay at the JSON root and the typed event is
//! stored as `{ "type": ..., "values": ... }`; a sidecar can flatten
//! `event.values` without guessing units or interpreting dynamically named
//! fields.  `MetricWriter::resume` validates the complete log before appending,
//! so a truncated record, sequence gap, different run, or rewound training
//! step cannot silently produce a misleading history.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Increment when the on-disk record contract changes incompatibly.
pub const METRIC_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricPhaseKind {
    Pretrain,
    ContinuedPretrain,
    Sft,
    Preference,
    Rl,
    Distillation,
    Sleep,
    Evaluation,
    Promotion,
    Quantization,
    CorpusPreparation,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricPhase {
    pub index: u32,
    pub name: String,
    pub kind: MetricPhaseKind,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricContext {
    pub global_step: u64,
    pub phase: MetricPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_hash: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseBoundary {
    Started,
    Progress,
    Completed,
    Failed,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseTimingMetrics {
    pub boundary: PhaseBoundary,
    pub elapsed_seconds: f64,
    pub input_wait_seconds: f64,
    pub forward_seconds: f64,
    pub backward_seconds: f64,
    pub optimizer_seconds: f64,
    pub checkpoint_seconds: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    Fast,
    Medium,
    Slow,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TierUpdateOutcome {
    Prospective,
    Committed,
    RolledBack,
    Skipped,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TierUpdateMetrics {
    pub transaction_id: String,
    pub tier: MemoryTier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receiver_tier: Option<MemoryTier>,
    pub tier_clock: u64,
    pub update_period: u64,
    pub accumulated_micro_steps: u64,
    pub outcome: TierUpdateOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update_l2_norm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve_slot: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve_generation: Option<u64>,
    pub optimizer_state_reset: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveCapacityMetrics {
    pub tier: MemoryTier,
    /// Persistent FFN/MoE expert pool; reserve slots are reported separately.
    pub active_base_experts: u32,
    pub active_reserve_experts: u32,
    pub dormant_reserve_experts: u32,
    /// Ordinary top-k of the persistent FFN MoE. Dream exploration never uses
    /// reserve slots to satisfy the additional-expert requirement.
    pub routed_top_k: u32,
    /// Fixed reserve route width. Before the first activation this lane is an
    /// untrainable zero-output low-rank fallback; afterward an activated slot
    /// replaces it. It therefore remains one while stored slots activate.
    pub reserve_routed_top_k: u32,
    /// Per-token routed expert parameter-equivalent. Reserve router scoring is
    /// excluded: candidate-pool overhead is bounded by stored capacity and is
    /// enforced by the independent wake throughput/latency gates.
    pub routed_active_parameters: u64,
    pub stored_parameters: u64,
    pub dream_generation: bool,
    pub random_extra_expert: bool,
}

/// Aggregate fixed-top-k wake-compute evidence across sleep cycles. The
/// initial sample includes the untrainable zero fallback route; every later
/// sample replaces that lane with one activated low-rank reserve route.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WakeCapacityEnvelopeMetrics {
    pub initial_wake_routed_active_parameters: u64,
    pub completed_sleep_cycles_observed: u64,
    pub minimum_observed_wake_routed_active_parameters: u64,
    pub maximum_observed_wake_routed_active_parameters: u64,
    pub stored_parameters: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistillationDivergenceMetrics {
    pub transaction_id: String,
    pub teacher_hash: String,
    pub student_hash: String,
    pub chunk_index: u32,
    pub chunk_count: u32,
    pub selected_tokens: u64,
    pub total_tokens: u64,
    pub forward_kl: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse_kl: Option<f64>,
    pub teacher_entropy: f64,
    pub student_entropy: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImitationRewardMetrics {
    pub transaction_id: String,
    pub semantic_judge_hash: String,
    pub samples: u64,
    pub semantic_score_mean: f64,
    pub normalized_levenshtein_mean: f64,
    pub levenshtein_threshold: f64,
    pub reward_mean: f64,
    pub reward_stddev: f64,
    pub grpo_kl: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DreamSelectionMetrics {
    pub transaction_id: String,
    pub selector_version: String,
    pub reference_set_hash: String,
    pub candidates_generated: u32,
    pub selected_by_alignment: u32,
    pub selected_random: u32,
    pub random_quota: u32,
    pub gradient_cosine_mean: f64,
    pub gradient_cosine_max: f64,
    pub selected_manifest_hash: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DreamTrialMetrics {
    pub transaction_id: String,
    pub candidate_hash: String,
    pub adapter_hash: String,
    pub evaluator_hash: String,
    pub lora_rank: u32,
    pub lora_alpha: f64,
    pub independent_task_delta: f64,
    pub reward: f64,
    pub elapsed_seconds: f64,
    pub accepted: bool,
    /// Must be true: the LoRA trial runs outside the shared candidate.
    pub isolated: bool,
    /// Must be true regardless of whether the trial is accepted.
    pub shared_checkpoint_unchanged: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionDeltaMetrics {
    pub transaction_id: String,
    pub suite: String,
    pub metric: String,
    pub evaluator_hash: String,
    pub direction: MetricDirection,
    pub baseline_score: f64,
    pub candidate_score: f64,
    /// Signed improvement: positive is always better, regardless of direction.
    pub improvement: f64,
    pub maximum_allowed_regression: f64,
    pub passed: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    BinaryG128,
    TernaryG128,
    TernaryEntropyG128,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationStage {
    Calibration,
    FakeQuantization,
    Export,
    Acceptance,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationMetrics {
    pub stage: QuantizationStage,
    pub format: QuantizationFormat,
    pub group_size: u32,
    pub progress_fraction: f64,
    pub tensors_quantized: u64,
    pub weights_quantized: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_bits_per_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_squared_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_absolute_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distillation_forward_kl: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_delta: Option<f64>,
    pub embeddings_quantized: bool,
    pub lm_head_quantized: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceUtilizationMetrics {
    /// Exact wall-clock time at which the background sampler collected this
    /// measurement. A writer may clamp the record timestamp to its durable
    /// monotonic watermark after a resume; this field is never clamped.
    pub sampled_at_unix_ms: u64,
    pub device_index: u32,
    pub sample_window_seconds: f64,
    pub gpu_utilization_percent: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_active_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_core_active_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_percent: Option<f64>,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_watts: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_celsius: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputMetrics {
    pub optimizer_steps: u64,
    pub compute_tokens: u64,
    pub supervised_tokens: u64,
    pub examples: u64,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
    pub examples_per_second: f64,
    pub input_wait_seconds: f64,
    pub host_to_device_seconds: f64,
    pub gpu_busy_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationMetrics {
    pub objective: String,
    pub loss: f64,
    pub optimized_loss: f64,
    pub weighted_loss: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_aux_loss: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_accuracy: Option<f64>,
    pub learning_rate: f64,
    pub muon_learning_rate: f64,
    pub gradient_norm: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_gradient_norms: Option<Vec<f64>>,
    pub sequence_length: u32,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
    pub compute_tokens: u64,
    pub supervised_tokens: u64,
    pub examples: u64,
    pub truncated_tokens: u64,
    pub retrieval_candidates: u64,
}

/// Exact objective executed by a native post-training update.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PostTrainingAlgorithm {
    Dpo,
    ForwardKl,
    Grpo,
}

/// One durably published DPO, forward-KL, or GRPO optimizer transaction.
/// Algorithm-specific measurements are optional at the wire level but are
/// validated as an exact, closed set for the selected `algorithm`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingUpdateMetrics {
    pub transaction_id: String,
    pub algorithm: PostTrainingAlgorithm,
    pub epoch: u64,
    pub first_record: u64,
    pub records: u64,
    pub optimizer_step: u64,
    pub rng_counter_start: u64,
    pub rng_counter_end: u64,
    pub loss: f64,
    pub checkpoint_sha256: String,
    pub optimizer_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preference_accuracy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implicit_reward_margin: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_kl: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_entropy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_agreement: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_reward: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_stddev: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_kl: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clipped_fraction: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "values", rename_all = "snake_case")]
pub enum MetricEvent {
    PhaseTiming(PhaseTimingMetrics),
    TierUpdate(TierUpdateMetrics),
    ActiveCapacity(ActiveCapacityMetrics),
    WakeCapacityEnvelope(WakeCapacityEnvelopeMetrics),
    DistillationDivergence(DistillationDivergenceMetrics),
    ImitationReward(ImitationRewardMetrics),
    DreamSelection(DreamSelectionMetrics),
    DreamTrial(DreamTrialMetrics),
    RetentionDelta(RetentionDeltaMetrics),
    Quantization(QuantizationMetrics),
    DeviceUtilization(DeviceUtilizationMetrics),
    Throughput(ThroughputMetrics),
    Optimization(OptimizationMetrics),
    PostTrainingUpdate(Box<PostTrainingUpdateMetrics>),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricRecord {
    pub schema_version: u32,
    pub sequence: u64,
    pub emitted_at_unix_ms: u64,
    pub run_id: String,
    pub global_step: u64,
    pub phase: MetricPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_hash: Option<String>,
    pub event: MetricEvent,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricLogState {
    pub records: u64,
    pub next_sequence: u64,
    pub last_global_step: Option<u64>,
    pub last_emitted_at_unix_ms: Option<u64>,
    pub run_id: Option<String>,
}

/// Measured allocation time represented in integer nanoseconds so re-reading
/// an already committed metric prefix cannot change its evidence bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommittedTrainingTime {
    pub optimizer_steps: u64,
    pub elapsed_nanoseconds: u64,
}

impl CommittedTrainingTime {
    /// The built-in trainer owns one accelerator, so allocated accelerator
    /// hours equal the sum of optimizer-step wall-clock windows.
    pub fn single_accelerator_hours(self) -> f64 {
        self.elapsed_nanoseconds as f64 / 3_600_000_000_000.0
    }
}

/// Separate identities for the raw audit journal and deterministic semantic
/// progress. Raw timing is intentionally not required to reproduce across an
/// interruption.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetricLogDigests {
    pub raw_sha256: String,
    pub semantic_progress_sha256: String,
    pub records: u64,
    pub last_global_step: Option<u64>,
}

/// Identity and mutation generation captured from an opened metric file.
///
/// Unix exposes a stable device/inode identity plus nanosecond mtime/ctime.
/// Other platforms retain the same fail-closed call structure using the
/// strongest portable metadata available from `std`.
#[derive(Clone, Debug, Eq, PartialEq)]
struct MetricFileStamp {
    len: u64,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    mtime: i64,
    #[cfg(unix)]
    mtime_nsec: i64,
    #[cfg(unix)]
    ctime: i64,
    #[cfg(unix)]
    ctime_nsec: i64,
    #[cfg(not(unix))]
    modified: Option<SystemTime>,
    #[cfg(not(unix))]
    created: Option<SystemTime>,
}

impl MetricFileStamp {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                len: metadata.len(),
                dev: metadata.dev(),
                ino: metadata.ino(),
                mtime: metadata.mtime(),
                mtime_nsec: metadata.mtime_nsec(),
                ctime: metadata.ctime(),
                ctime_nsec: metadata.ctime_nsec(),
            }
        }
        #[cfg(not(unix))]
        {
            Self {
                len: metadata.len(),
                modified: metadata.modified().ok(),
                created: metadata.created().ok(),
            }
        }
    }

    fn same_file(&self, other: &Self) -> bool {
        #[cfg(unix)]
        {
            self.dev == other.dev && self.ino == other.ino
        }
        #[cfg(not(unix))]
        {
            match (self.created, other.created) {
                (Some(left), Some(right)) => left == right,
                _ => self.same_version(other),
            }
        }
    }

    fn same_version(&self, other: &Self) -> bool {
        if !self.same_file_without_version_fallback(other) || self.len != other.len {
            return false;
        }
        #[cfg(unix)]
        {
            self.mtime == other.mtime
                && self.mtime_nsec == other.mtime_nsec
                && self.ctime == other.ctime
                && self.ctime_nsec == other.ctime_nsec
        }
        #[cfg(not(unix))]
        {
            self.modified == other.modified
        }
    }

    fn same_file_without_version_fallback(&self, other: &Self) -> bool {
        #[cfg(unix)]
        {
            self.dev == other.dev && self.ino == other.ino
        }
        #[cfg(not(unix))]
        {
            match (self.created, other.created) {
                (Some(left), Some(right)) => left == right,
                _ => true,
            }
        }
    }
}

/// Append-only writer that owns sequencing and validates every event.
pub struct MetricWriter {
    path: PathBuf,
    run_id: String,
    output: BufWriter<File>,
    state: MetricLogState,
}

impl MetricWriter {
    /// Create (or intentionally replace) a metric log for a new run.
    pub fn create(path: impl AsRef<Path>, run_id: impl Into<String>) -> Result<Self> {
        let path = path.as_ref();
        let run_id = run_id.into();
        validate_identifier("run_id", &run_id)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create metric directory {}", parent.display()))?;
        }
        let file = create_empty_metric_file(path, "metric log")?;
        Ok(Self {
            path: path.to_owned(),
            run_id,
            output: BufWriter::new(file),
            state: MetricLogState::default(),
        })
    }

    /// Validate an existing complete log and continue at its next sequence.
    pub fn resume(path: impl AsRef<Path>, run_id: impl Into<String>) -> Result<Self> {
        let path = path.as_ref();
        let run_id = run_id.into();
        validate_identifier("run_id", &run_id)?;
        let mut file = open_existing_metric_file(path, "metric log", true)?;
        let (bytes, stamp) = read_open_metric_file_stable(&mut file, path, "metric log")?;
        let state = validate_log_bytes(&bytes, Some(&run_id))
            .with_context(|| format!("validate metric log {}", path.display()))?;
        ensure_open_metric_file_unchanged(&file, path, &stamp, "metric log")?;
        Ok(Self {
            path: path.to_owned(),
            run_id,
            output: BufWriter::new(file),
            state,
        })
    }

    /// Resume at the metric sequence committed in checkpoint v2. Any later
    /// complete or torn records belong to optimizer work that was rolled back
    /// with the model checkpoint and are removed only after the committed
    /// prefix has been fully validated.
    pub fn resume_from_checkpoint(
        path: impl AsRef<Path>,
        run_id: impl Into<String>,
        committed_records: u64,
        committed_global_step: u64,
    ) -> Result<Self> {
        let path = path.as_ref();
        let run_id = run_id.into();
        validate_identifier("run_id", &run_id)?;
        let committed_records = usize::try_from(committed_records)
            .context("committed metric record count exceeds usize")?;
        let expected_last_step = (committed_records > 0).then_some(committed_global_step);
        Self::resume_validated_prefix(
            path,
            run_id,
            committed_records,
            expected_last_step,
            "checkpoint resume",
        )
    }

    /// Resume an external runtime at an exact committed metric prefix. Unlike
    /// [`Self::resume_from_checkpoint`], this verifies the last metric step for
    /// exact equality before truncating anything, so corrupt runtime metadata
    /// cannot mutate the journal as a side effect of a failed resume.
    pub fn resume_exact_prefix(
        path: impl AsRef<Path>,
        run_id: impl Into<String>,
        committed_records: u64,
        committed_last_global_step: Option<u64>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let run_id = run_id.into();
        validate_identifier("run_id", &run_id)?;
        let committed_records = usize::try_from(committed_records)
            .context("committed metric record count exceeds usize")?;
        Self::resume_validated_prefix(
            path,
            run_id,
            committed_records,
            committed_last_global_step,
            "exact resume",
        )
    }

    fn resume_validated_prefix(
        path: &Path,
        run_id: String,
        committed_records: usize,
        committed_last_global_step: Option<u64>,
        operation: &str,
    ) -> Result<Self> {
        let mut file = open_existing_metric_file(path, "metric log", true)
            .with_context(|| format!("open metric log {} for {operation}", path.display()))?;
        let (bytes, stamp) = read_open_metric_file_stable(&mut file, path, "metric log")
            .with_context(|| format!("read metric log {} for {operation}", path.display()))?;
        let (end, state) = validate_committed_prefix_bytes(
            &bytes,
            path,
            committed_records,
            committed_last_global_step,
            Some(&run_id),
        )?;

        // JSON validation can be substantial for a long journal. Reinspect
        // both the descriptor and its pathname after validation so an
        // in-place writer or path replacement cannot make us truncate bytes
        // other than the exact generation that was parsed above.
        ensure_open_metric_file_unchanged(&file, path, &stamp, "metric log")?;
        file.set_len(end as u64)
            .with_context(|| format!("truncate metric log {} for {operation}", path.display()))?;
        file.sync_all()
            .with_context(|| format!("sync metric log {} after {operation}", path.display()))?;
        let truncated = opened_metric_file_stamp(&file, "metric log")?;
        ensure!(
            truncated.same_file(&stamp) && truncated.len == end as u64,
            "metric log changed while its committed prefix was truncated"
        );
        ensure_metric_path_matches_stamp(path, &truncated, "metric log", true)?;

        Ok(Self {
            path: path.to_owned(),
            run_id,
            output: BufWriter::new(file),
            state,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn state(&self) -> &MetricLogState {
        &self.state
    }

    /// Append an event using a monotonic wall-clock timestamp.
    pub fn append(&mut self, context: MetricContext, event: MetricEvent) -> Result<u64> {
        let system_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_millis()
            .try_into()
            .context("Unix timestamp does not fit u64 milliseconds")?;
        let emitted_at = self
            .state
            .last_emitted_at_unix_ms
            .map_or(system_millis, |previous| previous.max(system_millis));
        self.append_at(context, event, emitted_at)
    }

    /// Append with an explicit timestamp, useful for importing monitored data.
    pub fn append_at(
        &mut self,
        context: MetricContext,
        event: MetricEvent,
        emitted_at_unix_ms: u64,
    ) -> Result<u64> {
        let sequence = self.state.next_sequence;
        let record = MetricRecord {
            schema_version: METRIC_SCHEMA_VERSION,
            sequence,
            emitted_at_unix_ms,
            run_id: self.run_id.clone(),
            global_step: context.global_step,
            phase: context.phase,
            checkpoint_hash: context.checkpoint_hash,
            event,
        };
        validate_record(&record)?;
        validate_record_position(&record, &self.state)?;

        // Validate the next state before touching the append-only file.  This
        // matters at the u64 boundary: an overflow error must not leave behind
        // a valid-looking record that the in-memory writer did not account for.
        let mut next_state = self.state.clone();
        advance_state(&mut next_state, &record)?;

        let mut encoded = serde_json::to_vec(&record).context("serialize metric record")?;
        encoded.push(b'\n');
        self.output
            .write_all(&encoded)
            .with_context(|| format!("append metric log {}", self.path.display()))?;
        self.state = next_state;
        Ok(sequence)
    }

    /// Push buffered records to the operating system.
    pub fn flush(&mut self) -> Result<()> {
        self.output
            .flush()
            .with_context(|| format!("flush metric log {}", self.path.display()))
    }

    /// Flush and ask the operating system to persist file data.
    pub fn sync_data(&mut self) -> Result<()> {
        self.flush()?;
        self.output
            .get_ref()
            .sync_data()
            .with_context(|| format!("sync metric data {}", self.path.display()))
    }

    /// Flush and persist both file data and metadata.
    pub fn sync_all(&mut self) -> Result<()> {
        self.flush()?;
        self.output
            .get_ref()
            .sync_all()
            .with_context(|| format!("sync metric log {}", self.path.display()))
    }

    /// Persist and measure exactly the prefix named by checkpoint state.
    pub fn committed_training_time(
        &mut self,
        committed_records: u64,
        committed_global_step: u64,
    ) -> Result<CommittedTrainingTime> {
        ensure!(
            self.state.records == committed_records,
            "metric writer has {} records but checkpoint commits {committed_records}",
            self.state.records
        );
        ensure!(
            self.state.last_global_step == Some(committed_global_step),
            "metric writer does not end at checkpoint global step {committed_global_step}"
        );
        self.sync_all()?;
        summarize_committed_training_time(
            &self.path,
            &self.run_id,
            committed_records,
            committed_global_step,
        )
    }
}

fn committed_prefix_end(bytes: &[u8], committed_records: usize) -> Result<usize> {
    if committed_records == 0 {
        return Ok(0);
    }
    bytes
        .iter()
        .enumerate()
        .filter_map(|(index, byte)| (*byte == b'\n').then_some(index + 1))
        .nth(committed_records - 1)
        .context("metric log has fewer records than the runtime checkpoint")
}

fn validate_committed_prefix_bytes(
    bytes: &[u8],
    path: &Path,
    committed_records: usize,
    committed_last_global_step: Option<u64>,
    expected_run_id: Option<&str>,
) -> Result<(usize, MetricLogState)> {
    validate_committed_prefix_bytes_with(
        bytes,
        path,
        committed_records,
        committed_last_global_step,
        expected_run_id,
        |_| Ok(()),
    )
}

fn validate_committed_prefix_bytes_with(
    bytes: &[u8],
    path: &Path,
    committed_records: usize,
    committed_last_global_step: Option<u64>,
    expected_run_id: Option<&str>,
    visitor: impl FnMut(&MetricRecord) -> Result<()>,
) -> Result<(usize, MetricLogState)> {
    let end = committed_prefix_end(bytes, committed_records)?;
    let state = visit_validated_records(&bytes[..end], expected_run_id, visitor)
        .with_context(|| format!("validate committed metric prefix in {}", path.display()))?;
    ensure!(
        state.records == committed_records as u64,
        "committed metric prefix record count is inconsistent"
    );
    ensure!(
        state.last_global_step == committed_last_global_step,
        "committed metric prefix global step is inconsistent"
    );
    Ok((end, state))
}

/// Validate a metric log without opening it for append.
pub fn validate_metric_log(
    path: impl AsRef<Path>,
    expected_run_id: Option<&str>,
) -> Result<MetricLogState> {
    if let Some(run_id) = expected_run_id {
        validate_identifier("expected run_id", run_id)?;
    }
    let path = path.as_ref();
    let mut file = open_existing_metric_file(path, "metric log", false)?;
    let (bytes, stamp) = read_open_metric_file_stable(&mut file, path, "metric log")?;
    let state = validate_log_bytes(&bytes, expected_run_id)
        .with_context(|| format!("validate metric log {}", path.display()))?;
    ensure_open_metric_file_unchanged(&file, path, &stamp, "metric log")?;
    Ok(state)
}

/// Validate the exact journal prefix committed by a training checkpoint
/// without truncating or otherwise mutating the live journal.
pub fn validate_metric_prefix(
    path: impl AsRef<Path>,
    committed_records: u64,
    committed_global_step: u64,
) -> Result<MetricLogState> {
    validate_metric_prefix_impl(
        path.as_ref(),
        committed_records,
        committed_global_step,
        None,
    )
}

/// Validate a committed journal prefix and bind every record to one run.
pub fn validate_metric_prefix_for_run(
    path: impl AsRef<Path>,
    expected_run_id: &str,
    committed_records: u64,
    committed_global_step: u64,
) -> Result<MetricLogState> {
    validate_identifier("expected run_id", expected_run_id)?;
    validate_metric_prefix_impl(
        path.as_ref(),
        committed_records,
        committed_global_step,
        Some(expected_run_id),
    )
}

fn validate_metric_prefix_impl(
    path: &Path,
    committed_records: u64,
    committed_global_step: u64,
    expected_run_id: Option<&str>,
) -> Result<MetricLogState> {
    let bytes = read_regular_file_stable(path, "metric log")?;
    let committed_records = usize::try_from(committed_records)
        .context("committed metric record count exceeds usize")?;
    let expected_last_step = (committed_records > 0).then_some(committed_global_step);
    let (_, state) = validate_committed_prefix_bytes(
        &bytes,
        path,
        committed_records,
        expected_last_step,
        expected_run_id,
    )?;
    Ok(state)
}

/// Validate an immutable metric snapshot that contains exactly the records
/// committed by its checkpoint, with no live-journal tail.
pub fn validate_metric_snapshot(
    path: impl AsRef<Path>,
    committed_records: u64,
    committed_global_step: u64,
) -> Result<MetricLogState> {
    validate_metric_snapshot_impl(
        path.as_ref(),
        committed_records,
        committed_global_step,
        None,
    )
}

/// Validate an immutable metric snapshot and bind every record to one run.
pub fn validate_metric_snapshot_for_run(
    path: impl AsRef<Path>,
    expected_run_id: &str,
    committed_records: u64,
    committed_global_step: u64,
) -> Result<MetricLogState> {
    validate_identifier("expected run_id", expected_run_id)?;
    validate_metric_snapshot_impl(
        path.as_ref(),
        committed_records,
        committed_global_step,
        Some(expected_run_id),
    )
}

fn validate_metric_snapshot_impl(
    path: &Path,
    committed_records: u64,
    committed_global_step: u64,
    expected_run_id: Option<&str>,
) -> Result<MetricLogState> {
    let bytes = read_regular_file_stable(path, "metric snapshot")?;
    let committed_records = usize::try_from(committed_records)
        .context("committed metric record count exceeds usize")?;
    let expected_last_step = (committed_records > 0).then_some(committed_global_step);
    let (end, state) = validate_committed_prefix_bytes(
        &bytes,
        path,
        committed_records,
        expected_last_step,
        expected_run_id,
    )?;
    ensure!(
        end == bytes.len(),
        "immutable metric snapshot contains an uncommitted tail"
    );
    Ok(state)
}

/// Measure committed single-accelerator work from a validated metric prefix.
/// Wake `Throughput.elapsed_seconds` covers complete optimizer-step windows;
/// non-overlapping sleep and final quantization-export `PhaseTiming` windows
/// are added explicitly. `gpu_busy_seconds` is a utilization diagnostic and
/// is deliberately not substituted for elapsed committed work.
pub fn summarize_committed_training_time(
    path: impl AsRef<Path>,
    expected_run_id: &str,
    committed_records: u64,
    committed_global_step: u64,
) -> Result<CommittedTrainingTime> {
    validate_identifier("expected run_id", expected_run_id)?;
    let path = path.as_ref();
    let bytes = read_regular_file_stable(path, "metric log for training evidence")?;
    let committed_records_usize = usize::try_from(committed_records)
        .context("committed metric record count exceeds usize")?;
    let mut optimizer_steps = 0_u64;
    let mut elapsed_nanoseconds = 0_u64;
    let (_, _) = validate_committed_prefix_bytes_with(
        &bytes,
        path,
        committed_records_usize,
        Some(committed_global_step),
        Some(expected_run_id),
        |record| {
            let elapsed = match &record.event {
                MetricEvent::Throughput(metric) => {
                    optimizer_steps = optimizer_steps
                        .checked_add(metric.optimizer_steps)
                        .context("measured optimizer-step count overflows u64")?;
                    Some(metric.elapsed_seconds)
                }
                MetricEvent::PhaseTiming(metric)
                    if matches!(
                        record.phase.kind,
                        MetricPhaseKind::Sleep | MetricPhaseKind::Quantization
                    ) && metric.boundary == PhaseBoundary::Completed =>
                {
                    Some(metric.elapsed_seconds)
                }
                _ => None,
            };
            if let Some(elapsed) = elapsed.filter(|elapsed| *elapsed > 0.0) {
                let nanoseconds = seconds_to_nanoseconds(elapsed)?;
                elapsed_nanoseconds = elapsed_nanoseconds
                    .checked_add(nanoseconds)
                    .context("measured training duration overflows u64 nanoseconds")?;
            }
            Ok(())
        },
    )?;
    ensure!(
        optimizer_steps == committed_global_step,
        "throughput metrics account for {optimizer_steps} optimizer steps, checkpoint records {committed_global_step}"
    );
    ensure!(
        elapsed_nanoseconds > 0,
        "committed throughput metrics contain no measurable training time"
    );
    Ok(CommittedTrainingTime {
        optimizer_steps,
        elapsed_nanoseconds,
    })
}

/// Hash both the exact JSONL bytes and a canonical semantic projection.
///
/// The semantic projection excludes record sequence, run id, timestamps,
/// device samples, and elapsed/throughput timing fields. It keeps optimizer
/// progress, losses, rewards, token counts, phase boundaries, and checkpoint
/// identities, allowing interrupted and uninterrupted executions to compare
/// deterministic state without pretending their wall-clock traces match.
pub fn metric_log_digests(
    path: impl AsRef<Path>,
    expected_run_id: Option<&str>,
) -> Result<MetricLogDigests> {
    if let Some(run_id) = expected_run_id {
        validate_identifier("expected run_id", run_id)?;
    }
    let path = path.as_ref();
    let bytes = read_regular_file_stable(path, "metric log for digest")?;
    metric_log_digests_from_bytes(&bytes, expected_run_id)
        .with_context(|| format!("validate metric log {} for digest", path.display()))
}

/// Verify and digest bytes already read through a stable artifact handle. This
/// lets higher-level evidence verification avoid a second path open between
/// checking an artifact's raw digest and projecting semantic progress.
pub(crate) fn metric_log_digests_from_bytes(
    bytes: &[u8],
    expected_run_id: Option<&str>,
) -> Result<MetricLogDigests> {
    if let Some(run_id) = expected_run_id {
        validate_identifier("expected run_id", run_id)?;
    }
    let mut semantic = Sha256::new();
    let state = visit_validated_records(bytes, expected_run_id, |record| {
        if let Some(value) = semantic_record(record)? {
            semantic.update(serde_json::to_vec(&canonicalize_json(value))?);
            semantic.update(b"\n");
        }
        Ok(())
    })?;
    Ok(MetricLogDigests {
        raw_sha256: hex_sha256(bytes),
        semantic_progress_sha256: format!("{:x}", semantic.finalize()),
        records: state.records,
        last_global_step: state.last_global_step,
    })
}

fn read_regular_file_stable(path: &Path, label: &str) -> Result<Vec<u8>> {
    let mut file = open_existing_metric_file(path, label, false)?;
    let (bytes, stamp) = read_open_metric_file_stable(&mut file, path, label)?;
    ensure_open_metric_file_unchanged(&file, path, &stamp, label)?;
    Ok(bytes)
}

fn create_empty_metric_file(path: &Path, label: &str) -> Result<File> {
    let existing = match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure_regular_metric_metadata(&metadata, path, label)?;
            Some(MetricFileStamp::from_metadata(&metadata))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| format!("inspect {label} {}", path.display()));
        }
    };

    let mut options = OpenOptions::new();
    options.read(true).write(true);
    if existing.is_none() {
        options.create_new(true);
    }
    harden_metric_open_options(&mut options, true);
    let file = options
        .open(path)
        .with_context(|| format!("open {label} {} for initialization", path.display()))?;
    let opened = opened_metric_file_stamp(&file, label)?;
    if let Some(existing) = existing {
        ensure!(
            opened.same_version(&existing),
            "{label} {} changed while it was opened for initialization",
            path.display()
        );
    }
    ensure_metric_path_matches_stamp(path, &opened, label, true)?;

    // Truncate only after the opened descriptor is proven to be the exact
    // regular file inspected above. A path swap can therefore make creation
    // fail, but can never redirect truncation into the replacement.
    ensure_open_metric_file_unchanged(&file, path, &opened, label)?;
    file.set_len(0)
        .with_context(|| format!("truncate {label} {}", path.display()))?;
    let empty = opened_metric_file_stamp(&file, label)?;
    ensure!(
        empty.same_file(&opened) && empty.len == 0,
        "{label} changed while it was initialized"
    );
    ensure_metric_path_matches_stamp(path, &empty, label, true)?;
    into_append_metric_file(file, path, label)
}

fn open_existing_metric_file(path: &Path, label: &str, append: bool) -> Result<File> {
    let before = metric_path_stamp(path, label)?;
    let mut options = OpenOptions::new();
    options.read(true);
    if append {
        options.append(true);
    }
    harden_metric_open_options(&mut options, false);
    let file = options
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))?;
    let opened = opened_metric_file_stamp(&file, label)?;
    ensure!(
        opened.same_version(&before),
        "{label} {} changed while it was opened",
        path.display()
    );
    ensure_metric_path_matches_stamp(path, &opened, label, true)?;
    Ok(file)
}

fn read_open_metric_file_stable(
    file: &mut File,
    path: &Path,
    label: &str,
) -> Result<(Vec<u8>, MetricFileStamp)> {
    let before = opened_metric_file_stamp(file, label)?;
    ensure_metric_path_matches_stamp(path, &before, label, true)?;
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("rewind opened {label} {}", path.display()))?;
    let capacity = usize::try_from(before.len).context("metric file length exceeds usize")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read opened {label} {}", path.display()))?;
    ensure!(
        bytes.len() as u64 == before.len,
        "{label} changed length while it was read"
    );
    ensure_open_metric_file_unchanged(file, path, &before, label)?;
    Ok((bytes, before))
}

fn ensure_open_metric_file_unchanged(
    file: &File,
    path: &Path,
    expected: &MetricFileStamp,
    label: &str,
) -> Result<()> {
    ensure_metric_path_matches_stamp(path, expected, label, false)?;
    let current = opened_metric_file_stamp(file, label)?;
    ensure!(
        current.same_version(expected),
        "{label} changed in place while it was being verified"
    );
    ensure_metric_path_matches_stamp(path, &current, label, true)
}

fn metric_path_stamp(path: &Path, label: &str) -> Result<MetricFileStamp> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    ensure_regular_metric_metadata(&metadata, path, label)?;
    Ok(MetricFileStamp::from_metadata(&metadata))
}

fn opened_metric_file_stamp(file: &File, label: &str) -> Result<MetricFileStamp> {
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect opened {label}"))?;
    ensure!(metadata.is_file(), "opened {label} must be a regular file");
    Ok(MetricFileStamp::from_metadata(&metadata))
}

fn ensure_regular_metric_metadata(metadata: &fs::Metadata, path: &Path, label: &str) -> Result<()> {
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{label} {} must be a regular non-symlink file",
        path.display()
    );
    Ok(())
}

fn ensure_metric_path_matches_stamp(
    path: &Path,
    expected: &MetricFileStamp,
    label: &str,
    require_same_version: bool,
) -> Result<()> {
    let current = metric_path_stamp(path, label)?;
    ensure!(
        current.same_file(expected),
        "{label} {} was replaced while it was open",
        path.display()
    );
    if require_same_version {
        ensure!(
            current.same_version(expected),
            "{label} {} changed in place while it was open",
            path.display()
        );
    }
    Ok(())
}

fn harden_metric_open_options(options: &mut OpenOptions, creating: bool) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
        if creating {
            options.mode(0o600);
        }
    }
    #[cfg(not(unix))]
    let _ = creating;
}

#[cfg(unix)]
fn into_append_metric_file(file: File, path: &Path, label: &str) -> Result<File> {
    use std::os::fd::AsRawFd;

    let initialized = opened_metric_file_stamp(&file, label)?;
    ensure_metric_path_matches_stamp(path, &initialized, label, true)?;
    // SAFETY: `file` owns this descriptor throughout both fcntl calls. F_GETFL
    // and F_SETFL do not consume it; O_APPEND is a supported status flag.
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    if flags == -1 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("read opened {label} descriptor flags"));
    }
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags | libc::O_APPEND) } == -1 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("set opened {label} descriptor append-only"));
    }
    ensure_open_metric_file_unchanged(&file, path, &initialized, label)?;
    Ok(file)
}

#[cfg(not(unix))]
fn into_append_metric_file(file: File, path: &Path, label: &str) -> Result<File> {
    let initialized = opened_metric_file_stamp(&file, label)?;
    let appended = open_existing_metric_file(path, label, true)?;
    let reopened = opened_metric_file_stamp(&appended, label)?;
    ensure!(
        reopened.same_version(&initialized),
        "{label} changed while it was reopened for append"
    );
    Ok(appended)
}

fn seconds_to_nanoseconds(seconds: f64) -> Result<u64> {
    finite("elapsed_seconds", seconds)?;
    ensure!(seconds > 0.0, "elapsed_seconds must be positive");
    let nanoseconds = seconds * 1_000_000_000.0;
    ensure!(
        nanoseconds.is_finite() && nanoseconds <= u64::MAX as f64,
        "elapsed_seconds cannot be represented as u64 nanoseconds"
    );
    let rounded = nanoseconds.round() as u64;
    ensure!(rounded > 0, "elapsed_seconds rounds to zero nanoseconds");
    Ok(rounded)
}

fn semantic_record(record: &MetricRecord) -> Result<Option<serde_json::Value>> {
    if matches!(record.event, MetricEvent::DeviceUtilization(_)) {
        return Ok(None);
    }
    let mut event = serde_json::to_value(&record.event)?;
    let event_object = event
        .as_object_mut()
        .context("serialized metric event is not an object")?;
    let values = event_object
        .get_mut("values")
        .and_then(serde_json::Value::as_object_mut)
        .context("serialized metric event has no values object")?;
    match &record.event {
        MetricEvent::PhaseTiming(_) => {
            for field in [
                "elapsed_seconds",
                "input_wait_seconds",
                "forward_seconds",
                "backward_seconds",
                "optimizer_seconds",
                "checkpoint_seconds",
            ] {
                values.remove(field);
            }
        }
        MetricEvent::Throughput(_) => {
            for field in [
                "elapsed_seconds",
                "tokens_per_second",
                "examples_per_second",
                "input_wait_seconds",
                "host_to_device_seconds",
                "gpu_busy_seconds",
            ] {
                values.remove(field);
            }
        }
        MetricEvent::DreamTrial(_) => {
            values.remove("elapsed_seconds");
        }
        _ => {}
    }
    Ok(Some(serde_json::json!({
        "schema_version": record.schema_version,
        "global_step": record.global_step,
        "phase": record.phase,
        "checkpoint_hash": record.checkpoint_hash,
        "event": event,
    })))
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(values) => {
            let mut entries = values.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize_json(value)))
                    .collect(),
            )
        }
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize_json).collect())
        }
        scalar => scalar,
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_log_bytes(bytes: &[u8], expected_run_id: Option<&str>) -> Result<MetricLogState> {
    visit_validated_records(bytes, expected_run_id, |_| Ok(()))
}

fn visit_validated_records(
    bytes: &[u8],
    expected_run_id: Option<&str>,
    mut visitor: impl FnMut(&MetricRecord) -> Result<()>,
) -> Result<MetricLogState> {
    if bytes.is_empty() {
        return Ok(MetricLogState::default());
    }
    ensure!(
        bytes.last() == Some(&b'\n'),
        "metric log ends with a partial record; refuse unsafe resume"
    );

    let mut state = MetricLogState::default();
    for (line_index, line) in bytes[..bytes.len() - 1]
        .split(|byte| *byte == b'\n')
        .enumerate()
    {
        ensure!(
            !line.is_empty(),
            "metric log contains an empty record at line {}",
            line_index + 1
        );
        let record: MetricRecord = serde_json::from_slice(line)
            .with_context(|| format!("decode metric record at line {}", line_index + 1))?;
        validate_record(&record)
            .with_context(|| format!("invalid metric record at line {}", line_index + 1))?;
        validate_record_position(&record, &state)
            .with_context(|| format!("invalid metric sequence at line {}", line_index + 1))?;
        if let Some(expected) = expected_run_id {
            ensure!(
                record.run_id == expected,
                "metric run_id `{}` does not match requested `{expected}`",
                record.run_id
            );
        }
        visitor(&record)
            .with_context(|| format!("process metric record at line {}", line_index + 1))?;
        advance_state(&mut state, &record)?;
    }
    Ok(state)
}

fn advance_state(state: &mut MetricLogState, record: &MetricRecord) -> Result<()> {
    state.records = state
        .records
        .checked_add(1)
        .context("metric record count overflows u64")?;
    state.next_sequence = record
        .sequence
        .checked_add(1)
        .context("metric sequence overflows u64")?;
    state.last_global_step = Some(record.global_step);
    state.last_emitted_at_unix_ms = Some(record.emitted_at_unix_ms);
    state.run_id = Some(record.run_id.clone());
    Ok(())
}

fn validate_record_position(record: &MetricRecord, state: &MetricLogState) -> Result<()> {
    ensure!(
        record.sequence == state.next_sequence,
        "expected metric sequence {}, got {}",
        state.next_sequence,
        record.sequence
    );
    if let Some(run_id) = &state.run_id {
        ensure!(
            record.run_id == *run_id,
            "metric run changed from `{run_id}` to `{}`",
            record.run_id
        );
    }
    if let Some(previous) = state.last_global_step {
        ensure!(
            record.global_step >= previous,
            "metric global step rewound from {previous} to {}",
            record.global_step
        );
    }
    if let Some(previous) = state.last_emitted_at_unix_ms {
        ensure!(
            record.emitted_at_unix_ms >= previous,
            "metric timestamp rewound from {previous} to {}",
            record.emitted_at_unix_ms
        );
    }
    Ok(())
}

pub fn validate_record(record: &MetricRecord) -> Result<()> {
    ensure!(
        record.schema_version == METRIC_SCHEMA_VERSION,
        "unsupported metric schema version {}; expected {METRIC_SCHEMA_VERSION}",
        record.schema_version
    );
    validate_identifier("run_id", &record.run_id)?;
    validate_identifier("phase name", &record.phase.name)?;
    if let Some(hash) = &record.checkpoint_hash {
        validate_sha256("checkpoint hash", hash)?;
    }
    record.event.validate()?;
    if let MetricEvent::DeviceUtilization(metric) = &record.event {
        ensure!(
            metric.sampled_at_unix_ms <= record.emitted_at_unix_ms,
            "device sample timestamp is later than its metric record"
        );
    }
    Ok(())
}

impl MetricEvent {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::PhaseTiming(metric) => {
                nonnegative("elapsed_seconds", metric.elapsed_seconds)?;
                nonnegative("input_wait_seconds", metric.input_wait_seconds)?;
                nonnegative("forward_seconds", metric.forward_seconds)?;
                nonnegative("backward_seconds", metric.backward_seconds)?;
                nonnegative("optimizer_seconds", metric.optimizer_seconds)?;
                nonnegative("checkpoint_seconds", metric.checkpoint_seconds)?;
                let accounted = metric.input_wait_seconds
                    + metric.forward_seconds
                    + metric.backward_seconds
                    + metric.optimizer_seconds
                    + metric.checkpoint_seconds;
                ensure!(
                    accounted <= metric.elapsed_seconds + timing_tolerance(metric.elapsed_seconds),
                    "phase component time {accounted} exceeds elapsed time {}",
                    metric.elapsed_seconds
                );
            }
            Self::TierUpdate(metric) => {
                validate_identifier("transaction_id", &metric.transaction_id)?;
                ensure!(metric.update_period > 0, "update_period must be positive");
                ensure!(
                    metric.tier_clock % metric.update_period == 0,
                    "tier_clock must be on its update-period boundary"
                );
                if let Some(norm) = metric.update_l2_norm {
                    nonnegative("update_l2_norm", norm)?;
                }
                ensure!(
                    metric.reserve_slot.is_some() == metric.reserve_generation.is_some(),
                    "reserve_slot and reserve_generation must be present together"
                );
                if matches!(metric.outcome, TierUpdateOutcome::RolledBack) {
                    ensure!(
                        !metric.optimizer_state_reset,
                        "a rolled-back tier update cannot reset optimizer state"
                    );
                }
            }
            Self::ActiveCapacity(metric) => {
                ensure!(metric.routed_top_k > 0, "routed_top_k must be positive");
                ensure!(
                    metric.reserve_routed_top_k == 1,
                    "sleep reserve route width must remain exactly one"
                );
                ensure!(
                    metric.routed_top_k <= metric.active_base_experts,
                    "routed_top_k exceeds persistent base experts"
                );
                ensure!(
                    metric.routed_active_parameters <= metric.stored_parameters,
                    "routed active parameters exceed stored parameters"
                );
                ensure!(
                    !metric.random_extra_expert || metric.dream_generation,
                    "random-extra-expert routing is only valid during dream generation"
                );
                if metric.random_extra_expert {
                    ensure!(
                        metric.routed_top_k < metric.active_base_experts,
                        "persistent base MoE has no expert outside ordinary top-k"
                    );
                }
            }
            Self::WakeCapacityEnvelope(metric) => {
                ensure!(
                    metric.initial_wake_routed_active_parameters > 0
                        && metric.minimum_observed_wake_routed_active_parameters > 0
                        && metric.maximum_observed_wake_routed_active_parameters > 0,
                    "wake capacity measurements must be positive"
                );
                ensure!(
                    metric.completed_sleep_cycles_observed > 0,
                    "wake capacity envelope requires completed sleep-cycle observations"
                );
                ensure!(
                    metric.minimum_observed_wake_routed_active_parameters
                        <= metric.maximum_observed_wake_routed_active_parameters,
                    "wake capacity envelope is inverted"
                );
                ensure!(
                    metric.maximum_observed_wake_routed_active_parameters
                        <= metric.stored_parameters,
                    "wake routed parameters exceed stored parameters"
                );
            }
            Self::DistillationDivergence(metric) => {
                validate_identifier("transaction_id", &metric.transaction_id)?;
                validate_sha256("teacher_hash", &metric.teacher_hash)?;
                validate_sha256("student_hash", &metric.student_hash)?;
                ensure!(metric.chunk_count > 0, "chunk_count must be positive");
                ensure!(
                    metric.chunk_index < metric.chunk_count,
                    "chunk_index must be less than chunk_count"
                );
                ensure!(metric.total_tokens > 0, "total_tokens must be positive");
                ensure!(
                    metric.selected_tokens > 0 && metric.selected_tokens <= metric.total_tokens,
                    "selected_tokens must be in 1..=total_tokens"
                );
                nonnegative("forward_kl", metric.forward_kl)?;
                if let Some(value) = metric.reverse_kl {
                    nonnegative("reverse_kl", value)?;
                }
                nonnegative("teacher_entropy", metric.teacher_entropy)?;
                nonnegative("student_entropy", metric.student_entropy)?;
            }
            Self::ImitationReward(metric) => {
                validate_identifier("transaction_id", &metric.transaction_id)?;
                validate_sha256("semantic_judge_hash", &metric.semantic_judge_hash)?;
                ensure!(metric.samples > 0, "imitation samples must be positive");
                unit_interval("semantic_score_mean", metric.semantic_score_mean)?;
                unit_interval(
                    "normalized_levenshtein_mean",
                    metric.normalized_levenshtein_mean,
                )?;
                unit_interval("levenshtein_threshold", metric.levenshtein_threshold)?;
                finite("reward_mean", metric.reward_mean)?;
                nonnegative("reward_stddev", metric.reward_stddev)?;
                nonnegative("grpo_kl", metric.grpo_kl)?;
            }
            Self::DreamSelection(metric) => {
                validate_identifier("transaction_id", &metric.transaction_id)?;
                validate_identifier("selector_version", &metric.selector_version)?;
                validate_sha256("reference_set_hash", &metric.reference_set_hash)?;
                validate_sha256("selected_manifest_hash", &metric.selected_manifest_hash)?;
                ensure!(
                    metric.selected_random <= metric.random_quota,
                    "selected random dreams exceed the configured quota"
                );
                let selected = metric
                    .selected_by_alignment
                    .checked_add(metric.selected_random)
                    .context("selected dream count overflows u32")?;
                ensure!(
                    selected <= metric.candidates_generated,
                    "selected dreams exceed generated candidates"
                );
                signed_unit_interval("gradient_cosine_mean", metric.gradient_cosine_mean)?;
                signed_unit_interval("gradient_cosine_max", metric.gradient_cosine_max)?;
                ensure!(
                    metric.gradient_cosine_max >= metric.gradient_cosine_mean,
                    "maximum gradient cosine is below the mean"
                );
            }
            Self::DreamTrial(metric) => {
                validate_identifier("transaction_id", &metric.transaction_id)?;
                validate_sha256("candidate_hash", &metric.candidate_hash)?;
                validate_sha256("adapter_hash", &metric.adapter_hash)?;
                validate_sha256("evaluator_hash", &metric.evaluator_hash)?;
                ensure!(metric.lora_rank > 0, "LoRA rank must be positive");
                ensure!(metric.lora_alpha > 0.0, "LoRA alpha must be positive");
                finite("lora_alpha", metric.lora_alpha)?;
                finite("independent_task_delta", metric.independent_task_delta)?;
                finite("reward", metric.reward)?;
                nonnegative("elapsed_seconds", metric.elapsed_seconds)?;
                ensure!(metric.isolated, "dream trial must be isolated");
                ensure!(
                    metric.shared_checkpoint_unchanged,
                    "dream trial mutated the shared candidate checkpoint"
                );
            }
            Self::RetentionDelta(metric) => {
                validate_identifier("transaction_id", &metric.transaction_id)?;
                validate_identifier("suite", &metric.suite)?;
                validate_identifier("metric", &metric.metric)?;
                validate_sha256("evaluator_hash", &metric.evaluator_hash)?;
                finite("baseline_score", metric.baseline_score)?;
                finite("candidate_score", metric.candidate_score)?;
                finite("improvement", metric.improvement)?;
                nonnegative(
                    "maximum_allowed_regression",
                    metric.maximum_allowed_regression,
                )?;
                let expected = match metric.direction {
                    MetricDirection::HigherIsBetter => {
                        metric.candidate_score - metric.baseline_score
                    }
                    MetricDirection::LowerIsBetter => {
                        metric.baseline_score - metric.candidate_score
                    }
                };
                ensure!(
                    approximately_equal(expected, metric.improvement),
                    "retention improvement {} does not match scores (expected {expected})",
                    metric.improvement
                );
                ensure!(
                    metric.passed == (metric.improvement >= -metric.maximum_allowed_regression),
                    "retention passed flag disagrees with the configured regression gate"
                );
            }
            Self::Quantization(metric) => {
                ensure!(
                    metric.group_size > 0,
                    "quantization group_size must be positive"
                );
                unit_interval("progress_fraction", metric.progress_fraction)?;
                if let Some(value) = metric.average_bits_per_weight {
                    ensure!(value > 0.0, "average_bits_per_weight must be positive");
                    finite("average_bits_per_weight", value)?;
                }
                if let Some(value) = metric.mean_squared_error {
                    nonnegative("mean_squared_error", value)?;
                }
                if let Some(value) = metric.max_absolute_error {
                    nonnegative("max_absolute_error", value)?;
                }
                if let Some(value) = metric.distillation_forward_kl {
                    nonnegative("distillation_forward_kl", value)?;
                }
                if let Some(value) = metric.acceptance_delta {
                    finite("acceptance_delta", value)?;
                }
                if metric.stage == QuantizationStage::Export {
                    ensure!(
                        metric.mean_squared_error.is_some() && metric.max_absolute_error.is_some(),
                        "quantization export requires error measurements"
                    );
                }
                if metric.stage == QuantizationStage::Export && metric.weights_quantized > 0 {
                    ensure!(
                        metric.average_bits_per_weight.is_some() && metric.packed_bytes.is_some(),
                        "quantization export requires packed size and bits-per-weight"
                    );
                }
            }
            Self::DeviceUtilization(metric) => {
                ensure!(
                    metric.sample_window_seconds > 0.0,
                    "device sample window must be positive"
                );
                finite("sample_window_seconds", metric.sample_window_seconds)?;
                percent("gpu_utilization_percent", metric.gpu_utilization_percent)?;
                if let Some(value) = metric.sm_active_percent {
                    percent("sm_active_percent", value)?;
                }
                if let Some(value) = metric.tensor_core_active_percent {
                    percent("tensor_core_active_percent", value)?;
                }
                if let Some(value) = metric.memory_bandwidth_percent {
                    percent("memory_bandwidth_percent", value)?;
                }
                ensure!(
                    metric.memory_used_bytes <= metric.memory_total_bytes,
                    "used GPU memory exceeds total GPU memory"
                );
                if let Some(value) = metric.power_watts {
                    nonnegative("power_watts", value)?;
                }
                if let Some(value) = metric.temperature_celsius {
                    finite("temperature_celsius", value)?;
                }
            }
            Self::Throughput(metric) => {
                ensure!(
                    metric.optimizer_steps > 0,
                    "optimizer_steps must be positive"
                );
                ensure!(
                    metric.elapsed_seconds > 0.0,
                    "elapsed_seconds must be positive"
                );
                finite("elapsed_seconds", metric.elapsed_seconds)?;
                nonnegative("tokens_per_second", metric.tokens_per_second)?;
                nonnegative("examples_per_second", metric.examples_per_second)?;
                nonnegative("input_wait_seconds", metric.input_wait_seconds)?;
                nonnegative("host_to_device_seconds", metric.host_to_device_seconds)?;
                nonnegative("gpu_busy_seconds", metric.gpu_busy_seconds)?;
                ensure!(
                    metric.input_wait_seconds <= metric.elapsed_seconds,
                    "input wait exceeds throughput window"
                );
                ensure!(
                    metric.host_to_device_seconds <= metric.elapsed_seconds,
                    "host-to-device time exceeds throughput window"
                );
                ensure!(
                    metric.gpu_busy_seconds <= metric.elapsed_seconds,
                    "GPU busy time exceeds throughput window"
                );
                let expected_tokens = metric.compute_tokens as f64 / metric.elapsed_seconds;
                let expected_examples = metric.examples as f64 / metric.elapsed_seconds;
                ensure!(
                    approximately_equal(expected_tokens, metric.tokens_per_second),
                    "tokens_per_second does not match compute_tokens / elapsed_seconds"
                );
                ensure!(
                    approximately_equal(expected_examples, metric.examples_per_second),
                    "examples_per_second does not match examples / elapsed_seconds"
                );
            }
            Self::Optimization(metric) => {
                validate_identifier("objective", &metric.objective)?;
                finite("loss", metric.loss)?;
                finite("optimized_loss", metric.optimized_loss)?;
                finite("weighted_loss", metric.weighted_loss)?;
                if let Some(loss) = metric.router_aux_loss {
                    finite("router_aux_loss", loss)?;
                }
                if let Some(accuracy) = metric.retrieval_accuracy {
                    unit_interval("retrieval_accuracy", accuracy)?;
                }
                nonnegative("learning_rate", metric.learning_rate)?;
                nonnegative("muon_learning_rate", metric.muon_learning_rate)?;
                nonnegative("gradient_norm", metric.gradient_norm)?;
                if let Some(norms) = &metric.layer_gradient_norms {
                    ensure!(!norms.is_empty(), "layer_gradient_norms must not be empty");
                    for norm in norms {
                        nonnegative("layer_gradient_norm", *norm)?;
                    }
                }
                ensure!(
                    metric.sequence_length > 0
                        && metric.batch_size > 0
                        && metric.gradient_accumulation > 0,
                    "optimization geometry must be positive"
                );
                ensure!(
                    metric.examples > 0 && metric.compute_tokens > 0,
                    "optimization batch is empty"
                );
                ensure!(
                    metric.supervised_tokens <= metric.compute_tokens,
                    "supervised tokens exceed compute tokens"
                );
            }
            Self::PostTrainingUpdate(metric) => {
                validate_sha256("transaction_id", &metric.transaction_id)?;
                validate_sha256("checkpoint_sha256", &metric.checkpoint_sha256)?;
                validate_sha256("optimizer_sha256", &metric.optimizer_sha256)?;
                ensure!(metric.records > 0, "post-training update is empty");
                ensure!(metric.optimizer_step > 0, "optimizer_step must be positive");
                ensure!(
                    metric.rng_counter_end >= metric.rng_counter_start,
                    "post-training RNG range is inverted"
                );
                finite("post_training_loss", metric.loss)?;
                let dpo = (metric.preference_accuracy, metric.implicit_reward_margin);
                let distillation = (
                    metric.forward_kl,
                    metric.teacher_entropy,
                    metric.top1_agreement,
                );
                let grpo = (
                    metric.mean_reward,
                    metric.reward_stddev,
                    metric.mean_kl,
                    metric.clipped_fraction,
                );
                match metric.algorithm {
                    PostTrainingAlgorithm::Dpo => {
                        ensure!(
                            dpo.0.is_some()
                                && dpo.1.is_some()
                                && distillation == (None, None, None)
                                && grpo == (None, None, None, None),
                            "DPO metrics contain a missing or foreign objective field"
                        );
                        unit_interval("preference_accuracy", dpo.0.expect("checked present"))?;
                        finite("implicit_reward_margin", dpo.1.expect("checked present"))?;
                        ensure!(
                            metric.rng_counter_start == metric.rng_counter_end,
                            "DPO must not consume rollout RNG"
                        );
                    }
                    PostTrainingAlgorithm::ForwardKl => {
                        ensure!(
                            dpo == (None, None)
                                && distillation.0.is_some()
                                && distillation.1.is_some()
                                && distillation.2.is_some()
                                && grpo == (None, None, None, None),
                            "forward-KL metrics contain a missing or foreign objective field"
                        );
                        nonnegative("forward_kl", distillation.0.expect("checked present"))?;
                        nonnegative("teacher_entropy", distillation.1.expect("checked present"))?;
                        unit_interval("top1_agreement", distillation.2.expect("checked present"))?;
                        ensure!(
                            metric.rng_counter_start == metric.rng_counter_end,
                            "forward-KL must not consume rollout RNG"
                        );
                    }
                    PostTrainingAlgorithm::Grpo => {
                        ensure!(
                            dpo == (None, None)
                                && distillation == (None, None, None)
                                && grpo.0.is_some()
                                && grpo.1.is_some()
                                && grpo.2.is_some()
                                && grpo.3.is_some(),
                            "GRPO metrics contain a missing or foreign objective field"
                        );
                        finite("mean_reward", grpo.0.expect("checked present"))?;
                        nonnegative("reward_stddev", grpo.1.expect("checked present"))?;
                        nonnegative("mean_kl", grpo.2.expect("checked present"))?;
                        unit_interval("clipped_fraction", grpo.3.expect("checked present"))?;
                        ensure!(
                            metric.rng_counter_end > metric.rng_counter_start,
                            "GRPO must reserve rollout RNG"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_identifier(name: &str, value: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{name} must not be empty");
    ensure!(
        !value.contains(['\n', '\r']),
        "{name} must not contain a newline"
    );
    Ok(())
}

fn validate_sha256(name: &str, value: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .with_context(|| format!("{name} must start with `sha256:`"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must contain 64 lowercase hexadecimal digits"
    );
    Ok(())
}

fn finite(name: &str, value: f64) -> Result<()> {
    ensure!(value.is_finite(), "{name} must be finite");
    Ok(())
}

fn nonnegative(name: &str, value: f64) -> Result<()> {
    finite(name, value)?;
    ensure!(value >= 0.0, "{name} must be non-negative");
    Ok(())
}

fn unit_interval(name: &str, value: f64) -> Result<()> {
    finite(name, value)?;
    ensure!((0.0..=1.0).contains(&value), "{name} must be in [0, 1]");
    Ok(())
}

fn signed_unit_interval(name: &str, value: f64) -> Result<()> {
    finite(name, value)?;
    ensure!((-1.0..=1.0).contains(&value), "{name} must be in [-1, 1]");
    Ok(())
}

fn percent(name: &str, value: f64) -> Result<()> {
    finite(name, value)?;
    ensure!((0.0..=100.0).contains(&value), "{name} must be in [0, 100]");
    Ok(())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 1e-9 * scale
}

fn timing_tolerance(elapsed: f64) -> f64 {
    1e-9 * elapsed.abs().max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase() -> MetricPhase {
        MetricPhase {
            index: 2,
            name: "sleep-02".to_owned(),
            kind: MetricPhaseKind::Sleep,
        }
    }

    fn context(step: u64) -> MetricContext {
        MetricContext {
            global_step: step,
            phase: phase(),
            checkpoint_hash: Some(format!("sha256:{:064x}", 42)),
        }
    }

    fn hash(digit: char) -> String {
        format!("sha256:{}", digit.to_string().repeat(64))
    }

    fn throughput() -> MetricEvent {
        MetricEvent::Throughput(ThroughputMetrics {
            optimizer_steps: 10,
            compute_tokens: 20_000,
            supervised_tokens: 19_000,
            examples: 100,
            elapsed_seconds: 2.0,
            tokens_per_second: 10_000.0,
            examples_per_second: 50.0,
            input_wait_seconds: 0.1,
            host_to_device_seconds: 0.02,
            gpu_busy_seconds: 1.8,
        })
    }

    fn one_step_throughput(seconds: f64) -> MetricEvent {
        MetricEvent::Throughput(ThroughputMetrics {
            optimizer_steps: 1,
            compute_tokens: 2_000,
            supervised_tokens: 1_900,
            examples: 10,
            elapsed_seconds: seconds,
            tokens_per_second: 2_000.0 / seconds,
            examples_per_second: 10.0 / seconds,
            input_wait_seconds: 0.1_f64.min(seconds),
            host_to_device_seconds: 0.02_f64.min(seconds),
            gpu_busy_seconds: (seconds * 0.9).min(seconds),
        })
    }

    fn valid_record(event: MetricEvent) -> MetricRecord {
        MetricRecord {
            schema_version: METRIC_SCHEMA_VERSION,
            sequence: 0,
            emitted_at_unix_ms: 100,
            run_id: "run-42".to_owned(),
            global_step: 9,
            phase: phase(),
            checkpoint_hash: None,
            event,
        }
    }

    #[test]
    fn event_json_is_versioned_and_strictly_tagged() {
        let record = valid_record(throughput());
        let encoded = serde_json::to_value(&record).unwrap();
        assert_eq!(encoded["schema_version"], 2);
        assert_eq!(encoded["event"]["type"], "throughput");
        assert_eq!(encoded["event"]["values"]["tokens_per_second"], 10_000.0);
        assert_eq!(
            serde_json::from_value::<MetricRecord>(encoded).unwrap(),
            record
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut encoded = serde_json::to_value(valid_record(throughput())).unwrap();
        encoded["surprise"] = serde_json::json!(true);
        assert!(serde_json::from_value::<MetricRecord>(encoded).is_err());

        let mut encoded = serde_json::to_value(valid_record(throughput())).unwrap();
        encoded["event"]["values"]["surprise"] = serde_json::json!(true);
        assert!(serde_json::from_value::<MetricRecord>(encoded).is_err());
    }

    #[test]
    fn checkpoint_hash_is_canonical_sha256() {
        let mut record = valid_record(throughput());
        record.checkpoint_hash = Some("sha256:not-a-digest".to_owned());
        assert!(validate_record(&record).is_err());
        record.checkpoint_hash = Some(format!("sha256:{:064x}", 7));
        validate_record(&record).unwrap();
    }

    #[test]
    fn creates_flushes_and_resumes_contiguous_sequence() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native/metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            assert_eq!(writer.append_at(context(1), throughput(), 100).unwrap(), 0);
            assert_eq!(writer.append_at(context(1), throughput(), 101).unwrap(), 1);
            writer.sync_data().unwrap();
            assert_eq!(writer.state().records, 2);
        }
        {
            let mut writer = MetricWriter::resume(&path, "run-a").unwrap();
            assert_eq!(writer.state().next_sequence, 2);
            assert_eq!(writer.append_at(context(2), throughput(), 102).unwrap(), 2);
            writer.sync_all().unwrap();
        }
        let state = validate_metric_log(&path, Some("run-a")).unwrap();
        assert_eq!(state.records, 3);
        assert_eq!(state.next_sequence, 3);
        assert_eq!(state.last_global_step, Some(2));
        assert_eq!(state.last_emitted_at_unix_ms, Some(102));
    }

    #[test]
    fn resume_rejects_partial_final_record() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        fs::write(&path, b"{\"partial\":true}").unwrap();
        let error = MetricWriter::resume(&path, "run-a").err().unwrap();
        assert!(format!("{error:#}").contains("partial record"));
    }

    #[cfg(unix)]
    #[test]
    fn metric_operations_reject_symlink_files() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.jsonl");
        fs::write(&target, b"").unwrap();
        let link = directory.path().join("metrics.jsonl");
        symlink(&target, &link).unwrap();

        assert!(MetricWriter::create(&link, "run-a").is_err());
        assert!(MetricWriter::resume(&link, "run-a").is_err());
        assert!(MetricWriter::resume_from_checkpoint(&link, "run-a", 0, 0).is_err());
        assert!(MetricWriter::resume_exact_prefix(&link, "run-a", 0, None).is_err());
        assert!(validate_metric_log(&link, None).is_err());
        assert!(validate_metric_prefix(&link, 0, 0).is_err());
        assert!(validate_metric_snapshot(&link, 0, 0).is_err());
        assert!(fs::read(&target).unwrap().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn opened_metric_handle_detects_path_replacement_before_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.sync_all().unwrap();
        }

        let mut opened = open_existing_metric_file(&path, "metric log", true).unwrap();
        let (verified, stamp) =
            read_open_metric_file_stable(&mut opened, &path, "metric log").unwrap();
        let displaced = directory.path().join("displaced.jsonl");
        fs::rename(&path, &displaced).unwrap();
        fs::write(&path, &verified).unwrap();
        let replacement_before = fs::read(&path).unwrap();

        let error = ensure_open_metric_file_unchanged(&opened, &path, &stamp, "metric log")
            .unwrap_err()
            .to_string();
        assert!(error.contains("replaced"), "{error}");
        assert_eq!(fs::read(&path).unwrap(), replacement_before);
    }

    #[cfg(unix)]
    #[test]
    fn opened_metric_handle_detects_same_length_in_place_mutation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.sync_all().unwrap();
        }

        let mut opened = open_existing_metric_file(&path, "metric log", true).unwrap();
        let (verified, stamp) =
            read_open_metric_file_stable(&mut opened, &path, "metric log").unwrap();
        let mut mutator = OpenOptions::new().write(true).open(&path).unwrap();
        mutator.seek(SeekFrom::Start(0)).unwrap();
        mutator.write_all(b"[").unwrap();
        mutator.sync_all().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), verified.len() as u64);

        let error = ensure_open_metric_file_unchanged(&opened, &path, &stamp, "metric log")
            .unwrap_err()
            .to_string();
        assert!(error.contains("changed in place"), "{error}");
    }

    #[test]
    fn checkpoint_resume_discards_only_uncommitted_metric_tail() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.append_at(context(2), throughput(), 101).unwrap();
            writer.sync_all().unwrap();
        }
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"torn\"")
            .unwrap();
        let mut writer = MetricWriter::resume_from_checkpoint(&path, "run-a", 1, 1).unwrap();
        assert_eq!(writer.state().records, 1);
        assert_eq!(writer.append_at(context(2), throughput(), 102).unwrap(), 1);
        writer.sync_all().unwrap();
        assert_eq!(
            validate_metric_log(&path, Some("run-a")).unwrap().records,
            2
        );
    }

    #[test]
    fn read_only_checkpoint_prefix_validation_rejects_nul_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.sync_all().unwrap();
        }
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = 0;
        fs::write(&path, bytes).unwrap();

        assert!(validate_metric_prefix(&path, 1, 1).is_err());
    }

    #[test]
    fn immutable_metric_snapshot_rejects_uncommitted_tail() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.append_at(context(2), throughput(), 101).unwrap();
            writer.sync_all().unwrap();
        }

        validate_metric_prefix(&path, 1, 1).unwrap();
        let error = validate_metric_snapshot(&path, 1, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("uncommitted tail"), "{error}");
        validate_metric_snapshot(&path, 2, 2).unwrap();
    }

    #[test]
    fn committed_prefix_and_snapshot_can_be_bound_to_the_expected_run() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.sync_all().unwrap();
        }

        validate_metric_prefix_for_run(&path, "run-a", 1, 1).unwrap();
        validate_metric_snapshot_for_run(&path, "run-a", 1, 1).unwrap();
        let prefix_error = format!(
            "{:#}",
            validate_metric_prefix_for_run(&path, "run-b", 1, 1).unwrap_err()
        );
        assert!(
            prefix_error.contains("does not match requested"),
            "{prefix_error}"
        );
        let snapshot_error = format!(
            "{:#}",
            validate_metric_snapshot_for_run(&path, "run-b", 1, 1).unwrap_err()
        );
        assert!(
            snapshot_error.contains("does not match requested"),
            "{snapshot_error}"
        );
    }

    #[test]
    fn stale_initial_writer_cannot_create_a_sparse_hole_after_truncation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let mut stale = MetricWriter::create(&path, "run-a").unwrap();
        stale.append_at(context(1), throughput(), 100).unwrap();
        stale.sync_all().unwrap();

        fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap()
            .sync_all()
            .unwrap();
        stale.append_at(context(2), throughput(), 101).unwrap();
        stale.sync_all().unwrap();

        let bytes = fs::read(&path).unwrap();
        assert!(
            !bytes.contains(&0),
            "stale descriptor created a sparse hole"
        );
        assert_eq!(bytes.first(), Some(&b'{'));
    }

    #[test]
    fn committed_training_time_uses_only_the_validated_checkpoint_prefix() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer
                .append_at(context(1), one_step_throughput(1.25), 100)
                .unwrap();
            writer.sync_all().unwrap();
        }

        let first = summarize_committed_training_time(&path, "run-a", 1, 1).unwrap();
        assert_eq!(first.optimizer_steps, 1);
        assert_eq!(first.elapsed_nanoseconds, 1_250_000_000);
        let mut writer = MetricWriter::resume_from_checkpoint(&path, "run-a", 1, 1).unwrap();
        writer
            .append_at(context(2), one_step_throughput(2.5), 101)
            .unwrap();
        let mut sleep_context = context(2);
        sleep_context.phase.kind = MetricPhaseKind::Sleep;
        writer
            .append_at(
                sleep_context.clone(),
                MetricEvent::PhaseTiming(PhaseTimingMetrics {
                    boundary: PhaseBoundary::Started,
                    elapsed_seconds: 99.0,
                    input_wait_seconds: 0.0,
                    forward_seconds: 0.0,
                    backward_seconds: 0.0,
                    optimizer_seconds: 0.0,
                    checkpoint_seconds: 0.0,
                }),
                102,
            )
            .unwrap();
        writer
            .append_at(
                sleep_context,
                MetricEvent::PhaseTiming(PhaseTimingMetrics {
                    boundary: PhaseBoundary::Completed,
                    elapsed_seconds: 0.5,
                    input_wait_seconds: 0.0,
                    forward_seconds: 0.0,
                    backward_seconds: 0.0,
                    optimizer_seconds: 0.0,
                    checkpoint_seconds: 0.0,
                }),
                103,
            )
            .unwrap();
        let complete = writer.committed_training_time(4, 2).unwrap();
        assert_eq!(complete.optimizer_steps, 2);
        assert_eq!(complete.elapsed_nanoseconds, 4_250_000_000);
        assert!((complete.single_accelerator_hours() - 4.25 / 3600.0).abs() < 1e-15);
    }

    #[test]
    fn semantic_digest_excludes_observation_timing_but_raw_digest_retains_it() {
        let directory = tempfile::tempdir().unwrap();
        let first_path = directory.path().join("first.jsonl");
        let second_path = directory.path().join("second.jsonl");
        let mut first = MetricWriter::create(&first_path, "uninterrupted").unwrap();
        first
            .append_at(context(1), one_step_throughput(2.0), 100)
            .unwrap();
        first.sync_all().unwrap();

        let mut second = MetricWriter::create(&second_path, "resumed").unwrap();
        second
            .append_at(
                context(0),
                MetricEvent::DeviceUtilization(DeviceUtilizationMetrics {
                    sampled_at_unix_ms: 1,
                    device_index: 0,
                    sample_window_seconds: 1.0,
                    gpu_utilization_percent: 50.0,
                    sm_active_percent: None,
                    tensor_core_active_percent: None,
                    memory_bandwidth_percent: None,
                    memory_used_bytes: 1,
                    memory_total_bytes: 2,
                    power_watts: None,
                    temperature_celsius: None,
                }),
                900,
            )
            .unwrap();
        second
            .append_at(context(1), one_step_throughput(4.0), 901)
            .unwrap();
        second.sync_all().unwrap();

        let first = metric_log_digests(&first_path, Some("uninterrupted")).unwrap();
        let second = metric_log_digests(&second_path, Some("resumed")).unwrap();
        assert_ne!(first.raw_sha256, second.raw_sha256);
        assert_eq!(
            first.semantic_progress_sha256,
            second.semantic_progress_sha256
        );
    }

    #[test]
    fn exact_prefix_resume_validates_commit_metadata_before_truncation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.append_at(context(2), throughput(), 101).unwrap();
            writer.sync_all().unwrap();
        }
        let original = fs::read(&path).unwrap();
        assert!(MetricWriter::resume_exact_prefix(&path, "run-a", 1, Some(999)).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn trainer_checkpoint_resume_requires_the_exact_last_step_before_truncation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        {
            let mut writer = MetricWriter::create(&path, "run-a").unwrap();
            writer.append_at(context(1), throughput(), 100).unwrap();
            writer.append_at(context(2), throughput(), 101).unwrap();
            writer.sync_all().unwrap();
        }
        let original = fs::read(&path).unwrap();
        assert!(MetricWriter::resume_from_checkpoint(&path, "run-a", 1, 2).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn validation_rejects_sequence_gaps_and_run_changes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let mut first = valid_record(throughput());
        first.run_id = "run-a".to_owned();
        let mut second = first.clone();
        second.sequence = 2;
        second.emitted_at_unix_ms = 101;
        let bytes = format!(
            "{}\n{}\n",
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap()
        );
        fs::write(&path, bytes).unwrap();
        let error = validate_metric_log(&path, Some("run-a")).unwrap_err();
        assert!(format!("{error:#}").contains("expected metric sequence 1, got 2"));

        fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&first).unwrap()),
        )
        .unwrap();
        let error = MetricWriter::resume(&path, "run-b").err().unwrap();
        assert!(format!("{error:#}").contains("does not match requested"));
    }

    #[test]
    fn append_rejects_rewound_step_and_timestamp_without_writing() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let mut writer = MetricWriter::create(&path, "run-a").unwrap();
        writer.append_at(context(10), throughput(), 100).unwrap();
        assert!(writer.append_at(context(9), throughput(), 101).is_err());
        assert!(writer.append_at(context(10), throughput(), 99).is_err());
        writer.sync_data().unwrap();
        assert_eq!(validate_metric_log(&path, None).unwrap().records, 1);
    }

    #[test]
    fn validates_phase_timing_accounting() {
        let mut metric = PhaseTimingMetrics {
            boundary: PhaseBoundary::Progress,
            elapsed_seconds: 2.0,
            input_wait_seconds: 0.2,
            forward_seconds: 0.7,
            backward_seconds: 0.8,
            optimizer_seconds: 0.2,
            checkpoint_seconds: 0.1,
        };
        MetricEvent::PhaseTiming(metric.clone()).validate().unwrap();
        metric.checkpoint_seconds = 0.2;
        assert!(MetricEvent::PhaseTiming(metric).validate().is_err());
    }

    #[test]
    fn validates_tier_boundary_capacity_and_dream_only_route() {
        let tier = TierUpdateMetrics {
            transaction_id: "txn-1".to_owned(),
            tier: MemoryTier::Fast,
            receiver_tier: Some(MemoryTier::Medium),
            tier_clock: 100,
            update_period: 10,
            accumulated_micro_steps: 10,
            outcome: TierUpdateOutcome::Committed,
            update_l2_norm: Some(0.25),
            reserve_slot: Some(1),
            reserve_generation: Some(3),
            optimizer_state_reset: true,
        };
        MetricEvent::TierUpdate(tier.clone()).validate().unwrap();
        let mut off_boundary = tier;
        off_boundary.tier_clock = 101;
        assert!(MetricEvent::TierUpdate(off_boundary).validate().is_err());

        let mut capacity = ActiveCapacityMetrics {
            tier: MemoryTier::Fast,
            active_base_experts: 8,
            active_reserve_experts: 1,
            dormant_reserve_experts: 3,
            routed_top_k: 2,
            reserve_routed_top_k: 1,
            routed_active_parameters: 20,
            stored_parameters: 100,
            dream_generation: true,
            random_extra_expert: true,
        };
        MetricEvent::ActiveCapacity(capacity.clone())
            .validate()
            .unwrap();
        let mut no_distinct_expert = capacity.clone();
        no_distinct_expert.active_base_experts = 2;
        no_distinct_expert.active_reserve_experts = 7;
        assert!(
            MetricEvent::ActiveCapacity(no_distinct_expert)
                .validate()
                .is_err()
        );
        capacity.dream_generation = false;
        assert!(MetricEvent::ActiveCapacity(capacity).validate().is_err());
    }

    #[test]
    fn wake_capacity_envelope_requires_initial_and_complete_cycle_samples() {
        let mut envelope = WakeCapacityEnvelopeMetrics {
            initial_wake_routed_active_parameters: 80,
            completed_sleep_cycles_observed: 4,
            minimum_observed_wake_routed_active_parameters: 80,
            maximum_observed_wake_routed_active_parameters: 80,
            stored_parameters: 100,
        };
        MetricEvent::WakeCapacityEnvelope(envelope.clone())
            .validate()
            .unwrap();

        envelope.completed_sleep_cycles_observed = 0;
        assert!(
            MetricEvent::WakeCapacityEnvelope(envelope.clone())
                .validate()
                .is_err(),
            "missing cycle observations must fail closed"
        );
        envelope.completed_sleep_cycles_observed = 4;
        envelope.minimum_observed_wake_routed_active_parameters = 81;
        envelope.maximum_observed_wake_routed_active_parameters = 80;
        assert!(
            MetricEvent::WakeCapacityEnvelope(envelope)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validates_distillation_and_imitation_ranges() {
        let divergence = DistillationDivergenceMetrics {
            transaction_id: "txn".to_owned(),
            teacher_hash: hash('1'),
            student_hash: hash('2'),
            chunk_index: 0,
            chunk_count: 2,
            selected_tokens: 64,
            total_tokens: 128,
            forward_kl: 0.1,
            reverse_kl: Some(0.2),
            teacher_entropy: 3.0,
            student_entropy: 3.1,
        };
        MetricEvent::DistillationDivergence(divergence)
            .validate()
            .unwrap();
        let imitation = ImitationRewardMetrics {
            transaction_id: "txn".to_owned(),
            semantic_judge_hash: hash('3'),
            samples: 8,
            semantic_score_mean: 0.8,
            normalized_levenshtein_mean: 0.6,
            levenshtein_threshold: 0.5,
            reward_mean: -0.1,
            reward_stddev: 0.2,
            grpo_kl: 0.01,
        };
        MetricEvent::ImitationReward(imitation).validate().unwrap();
    }

    #[test]
    fn validates_selection_and_isolated_dream_trial() {
        let selection = DreamSelectionMetrics {
            transaction_id: "txn".to_owned(),
            selector_version: "cosine-v1".to_owned(),
            reference_set_hash: hash('4'),
            candidates_generated: 10,
            selected_by_alignment: 3,
            selected_random: 2,
            random_quota: 2,
            gradient_cosine_mean: 0.2,
            gradient_cosine_max: 0.9,
            selected_manifest_hash: hash('5'),
        };
        MetricEvent::DreamSelection(selection).validate().unwrap();

        let mut trial = DreamTrialMetrics {
            transaction_id: "txn".to_owned(),
            candidate_hash: hash('6'),
            adapter_hash: hash('7'),
            evaluator_hash: hash('8'),
            lora_rank: 64,
            lora_alpha: 128.0,
            independent_task_delta: 0.03,
            reward: 0.7,
            elapsed_seconds: 5.0,
            accepted: true,
            isolated: true,
            shared_checkpoint_unchanged: true,
        };
        MetricEvent::DreamTrial(trial.clone()).validate().unwrap();
        trial.shared_checkpoint_unchanged = false;
        assert!(MetricEvent::DreamTrial(trial).validate().is_err());
    }

    #[test]
    fn retention_delta_is_direction_normalized_and_gate_checked() {
        let mut retention = RetentionDeltaMetrics {
            transaction_id: "txn".to_owned(),
            suite: "stable-anchor".to_owned(),
            metric: "loss".to_owned(),
            evaluator_hash: hash('9'),
            direction: MetricDirection::LowerIsBetter,
            baseline_score: 2.0,
            candidate_score: 1.9,
            improvement: 0.1,
            maximum_allowed_regression: 0.01,
            passed: true,
        };
        MetricEvent::RetentionDelta(retention.clone())
            .validate()
            .unwrap();
        retention.passed = false;
        assert!(MetricEvent::RetentionDelta(retention).validate().is_err());
    }

    #[test]
    fn quantization_export_requires_size_and_reports_strict_ranges() {
        let mut quantization = QuantizationMetrics {
            stage: QuantizationStage::Export,
            format: QuantizationFormat::BinaryG128,
            group_size: 128,
            progress_fraction: 1.0,
            tensors_quantized: 100,
            weights_quantized: 1_000,
            average_bits_per_weight: Some(1.125),
            packed_bytes: Some(141),
            mean_squared_error: Some(0.02),
            max_absolute_error: Some(0.5),
            distillation_forward_kl: Some(0.01),
            acceptance_delta: Some(-0.002),
            embeddings_quantized: true,
            lm_head_quantized: true,
        };
        MetricEvent::Quantization(quantization.clone())
            .validate()
            .unwrap();
        let valid_export = quantization.clone();
        quantization.packed_bytes = None;
        assert!(MetricEvent::Quantization(quantization).validate().is_err());

        let qat = QuantizationMetrics {
            stage: QuantizationStage::FakeQuantization,
            format: QuantizationFormat::BinaryG128,
            group_size: 128,
            progress_fraction: 0.5,
            tensors_quantized: 100,
            weights_quantized: 1_000,
            average_bits_per_weight: None,
            packed_bytes: None,
            mean_squared_error: None,
            max_absolute_error: None,
            distillation_forward_kl: None,
            acceptance_delta: None,
            embeddings_quantized: true,
            lm_head_quantized: true,
        };
        MetricEvent::Quantization(qat).validate().unwrap();

        let mut missing_error = valid_export;
        missing_error.mean_squared_error = None;
        assert!(MetricEvent::Quantization(missing_error).validate().is_err());
    }

    #[test]
    fn post_training_update_requires_exact_algorithm_fields_and_rng_contract() {
        let mut metric = PostTrainingUpdateMetrics {
            transaction_id: hash('a'),
            algorithm: PostTrainingAlgorithm::Grpo,
            epoch: 0,
            first_record: 4,
            records: 2,
            optimizer_step: 3,
            rng_counter_start: 8,
            rng_counter_end: 24,
            loss: -0.2,
            checkpoint_sha256: hash('b'),
            optimizer_sha256: hash('c'),
            preference_accuracy: None,
            implicit_reward_margin: None,
            forward_kl: None,
            teacher_entropy: None,
            top1_agreement: None,
            mean_reward: Some(0.7),
            reward_stddev: Some(0.2),
            mean_kl: Some(0.01),
            clipped_fraction: Some(0.1),
        };
        MetricEvent::PostTrainingUpdate(Box::new(metric.clone()))
            .validate()
            .unwrap();
        metric.rng_counter_end = metric.rng_counter_start;
        assert!(
            MetricEvent::PostTrainingUpdate(Box::new(metric.clone()))
                .validate()
                .is_err()
        );
        metric.rng_counter_end = 24;
        metric.preference_accuracy = Some(1.0);
        assert!(
            MetricEvent::PostTrainingUpdate(Box::new(metric))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn device_and_throughput_ranges_are_checked() {
        let mut device = DeviceUtilizationMetrics {
            sampled_at_unix_ms: 1,
            device_index: 0,
            sample_window_seconds: 10.0,
            gpu_utilization_percent: 95.0,
            sm_active_percent: Some(82.0),
            tensor_core_active_percent: Some(70.0),
            memory_bandwidth_percent: Some(60.0),
            memory_used_bytes: 80,
            memory_total_bytes: 100,
            power_watts: Some(300.0),
            temperature_celsius: Some(70.0),
        };
        MetricEvent::DeviceUtilization(device.clone())
            .validate()
            .unwrap();
        device.gpu_utilization_percent = 101.0;
        assert!(MetricEvent::DeviceUtilization(device).validate().is_err());

        let mut event = throughput();
        event.validate().unwrap();
        if let MetricEvent::Throughput(metric) = &mut event {
            metric.tokens_per_second = 9_000.0;
        }
        assert!(event.validate().is_err());
    }

    #[test]
    fn empty_log_is_valid_but_empty_run_id_is_not() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        File::create(&path).unwrap();
        assert_eq!(
            validate_metric_log(&path, None).unwrap(),
            MetricLogState::default()
        );
        assert!(MetricWriter::create(&path, "  ").is_err());
    }

    #[test]
    fn blank_lines_are_not_silently_ignored() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let mut file = File::create(&path).unwrap();
        writeln!(file).unwrap();
        let error = validate_metric_log(&path, None).unwrap_err();
        assert!(format!("{error:#}").contains("empty record"));
    }
}
