//! Strict version-2 training workflow configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[cfg(test)]
use std::fs;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use summa_llm::ModelDef;

use crate::artifact_io::{read_regular_bounded, validate_sha256_identity};
use crate::posttrain::PostTrainingConfig;
use crate::sleep::{DreamingConfig, ImitationConfig, KnowledgeSeedingConfig, SleepSchedule};
use crate::task::{TaskAdapter, TaskConfig, TaskExecution};
use crate::tensor_sleep::{RetentionGateConfig, TensorConsolidationConfig};
use crate::tier_optimizer::TierOptimizerConfig;

pub const WORKFLOW_VERSION: u32 = 2;
const MAX_WORKFLOW_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_WORKFLOW_PHASES: usize = 4_096;
const MAX_WORKFLOW_NAME_BYTES: usize = 1_024;
const MAX_PHASE_NAME_BYTES: usize = 256;
const MAX_PHASE_BATCH_SIZE: usize = 65_536;
const MAX_PHASE_GRADIENT_ACCUMULATION: usize = 1_048_576;
const MAX_PHASE_SHUFFLE_SAMPLES: usize = 262_144;
const MAX_PHASE_BATCH_TOKENS: usize = 8 * 1024 * 1024;
// The checked-in education curriculum intentionally shuffles 65,536 examples
// at 4,096 tokens. This is a documented high-memory (~2 GiB for one raw i64
// token vector per sample) production tradeoff, and forms the upper bound.
const MAX_PHASE_SHUFFLE_TOKENS: usize = 256 * 1024 * 1024;

fn default_one_f64() -> f64 {
    1.0
}

fn default_group_size() -> usize {
    128
}

fn default_true() -> bool {
    true
}

/// Coarse executor class. It keeps orchestration generic while making state
/// transitions and checkpoint boundaries explicit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseClass {
    Optimization,
    ModelMutation,
    Assessment,
    Release,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    Pretrain,
    ContinuedPretrain,
    Sft,
    Preference,
    Rl,
    Distillation,
    Sleep,
    Quantization,
    Evaluation,
    Promotion,
}

impl PhaseKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pretrain => "pretrain",
            Self::ContinuedPretrain => "continued_pretrain",
            Self::Sft => "sft",
            Self::Preference => "preference",
            Self::Rl => "rl",
            Self::Distillation => "distillation",
            Self::Sleep => "sleep",
            Self::Quantization => "quantization",
            Self::Evaluation => "evaluation",
            Self::Promotion => "promotion",
        }
    }

    pub fn class(self) -> PhaseClass {
        match self {
            Self::Pretrain
            | Self::ContinuedPretrain
            | Self::Sft
            | Self::Preference
            | Self::Rl
            | Self::Distillation => PhaseClass::Optimization,
            Self::Sleep | Self::Quantization => PhaseClass::ModelMutation,
            Self::Evaluation => PhaseClass::Assessment,
            Self::Promotion => PhaseClass::Release,
        }
    }

    pub fn uses_task_data(self) -> bool {
        !matches!(self, Self::Sleep | Self::Promotion)
    }

    pub fn updates_model(self) -> bool {
        matches!(
            self.class(),
            PhaseClass::Optimization | PhaseClass::ModelMutation
        )
    }

    /// Whether the phase applies gradient-based optimizer updates. Standalone
    /// sleep mutates a model transactionally but does not run a wake optimizer;
    /// QAT does, despite publishing a model-mutation candidate at completion.
    pub fn uses_optimizer(self) -> bool {
        matches!(
            self,
            Self::Pretrain
                | Self::ContinuedPretrain
                | Self::Sft
                | Self::Preference
                | Self::Rl
                | Self::Distillation
                | Self::Quantization
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    BinaryG128,
    TernaryG128,
    TernaryEntropyG128,
}

/// One local input to the built-in promotion gate.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionEvidenceRef {
    pub path: PathBuf,
}

impl PromotionEvidenceRef {
    fn validate(&self, label: &str, phase_name: &str) -> Result<()> {
        ensure!(
            !self.path.as_os_str().is_empty(),
            "workflow phase `{phase_name}` promotion {label} path is empty"
        );
        ensure!(
            !self
                .path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)),
            "workflow phase `{phase_name}` promotion {label} path must not contain `..`"
        );
        Ok(())
    }

    fn resolve_path(&mut self, base: &Path) {
        if self.path.is_relative() {
            self.path = base.join(&self.path);
        }
    }
}

/// Strict inputs for the trainer-owned WorkflowV2 promotion executor.
///
/// The selected benchmark and ten other runs form the complete fixed ablation
/// matrix. The policy is supplied as an evidence file rather than as mutable
/// command-line state.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionConfig {
    pub selected_run: PromotionEvidenceRef,
    pub comparison_runs: Vec<PromotionEvidenceRef>,
    pub resources: PromotionEvidenceRef,
    pub policy: PromotionEvidenceRef,
    pub artifact_directory: PathBuf,
}

impl PromotionConfig {
    pub(crate) fn validate(&self, phase_name: &str) -> Result<()> {
        self.selected_run.validate("selected_run", phase_name)?;
        ensure!(
            self.comparison_runs.len() == 10,
            "workflow phase `{phase_name}` promotion requires exactly ten comparison runs in addition to selected_run"
        );
        for (index, run) in self.comparison_runs.iter().enumerate() {
            run.validate(&format!("comparison_runs[{index}]"), phase_name)?;
        }
        self.resources.validate("resources", phase_name)?;
        self.policy.validate("policy", phase_name)?;
        ensure!(
            !self.artifact_directory.as_os_str().is_empty(),
            "workflow phase `{phase_name}` promotion artifact_directory is empty"
        );
        ensure!(
            !self
                .artifact_directory
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)),
            "workflow phase `{phase_name}` promotion artifact_directory must not contain `..`"
        );

        let mut paths = BTreeSet::new();
        let evidence = std::iter::once(&self.selected_run)
            .chain(self.comparison_runs.iter())
            .chain(std::iter::once(&self.resources))
            .chain(std::iter::once(&self.policy));
        for (index, evidence) in evidence.enumerate() {
            ensure!(
                paths.insert(&evidence.path),
                "workflow phase `{phase_name}` promotion repeats evidence path {index}: {}",
                evidence.path.display()
            );
        }
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        self.selected_run.resolve_path(base);
        for run in &mut self.comparison_runs {
            run.resolve_path(base);
        }
        self.resources.resolve_path(base);
        self.policy.resolve_path(base);
        if self.artifact_directory.is_relative() {
            self.artifact_directory = base.join(&self.artifact_directory);
        }
    }
}

/// Training recipe for a quantized candidate. The task and data remain on the
/// enclosing phase so QAT and distillation use the same adapter contracts as
/// every other optimization phase.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum QuantizationTraining {
    Qat {
        #[serde(default)]
        warmup_steps: usize,
        #[serde(default = "default_true")]
        straight_through: bool,
    },
    Distillation {
        teacher_checkpoint: PathBuf,
        /// Exact SHA-256 identity of the frozen teacher checkpoint.
        teacher_sha256: String,
        #[serde(default = "default_one_f64")]
        temperature: f64,
        #[serde(default = "default_one_f64")]
        loss_weight: f64,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConfig {
    pub format: QuantizationFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_format: Option<QuantizationFormat>,
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    #[serde(default)]
    pub start_step: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_step: Option<usize>,
    #[serde(default = "default_true")]
    pub embeddings: bool,
    #[serde(default = "default_true")]
    pub lm_head: bool,
    pub training: QuantizationTraining,
}

/// Trainer-owned in-model sleep controls. Memory topology and reserve tensor
/// geometry remain in MAL; tier ids and capacities are repeated here so a run
/// fails before mutation if workflow and checkpoint topology disagree.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InModelSleepConfig {
    pub schedule: SleepSchedule,
    /// Required for a standalone `sleep` phase; absent for periodic hooks,
    /// whose boundary comes from the enclosing wake optimizer/token clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standalone_trigger_clock: Option<u64>,
    pub knowledge_seeding: KnowledgeSeedingConfig,
    pub imitation: ImitationConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dreaming: Option<DreamingConfig>,
    pub retention_suite: PathBuf,
    /// Exact identity of the frozen retention input. The native sleep runner
    /// verifies the bytes before every initial execution and resume.
    pub retention_suite_sha256: String,
    pub retention: RetentionGateConfig,
    pub receiver_learning_rate: f64,
    pub receiver_weight_decay: f32,
    pub grpo_clip_epsilon: f64,
    pub grpo_advantage_epsilon: f64,
    pub grpo_kl_coefficient: f64,
    pub candidate_directory: PathBuf,
}

impl InModelSleepConfig {
    fn validate(&self, phase_name: &str) -> Result<()> {
        self.schedule
            .validate()
            .with_context(|| format!("invalid schedule in sleep phase `{phase_name}`"))?;
        self.knowledge_seeding
            .validate()
            .with_context(|| format!("invalid knowledge seeding in sleep phase `{phase_name}`"))?;
        self.imitation
            .validate()
            .with_context(|| format!("invalid imitation settings in sleep phase `{phase_name}`"))?;
        if let Some(dreaming) = &self.dreaming {
            dreaming.validate().with_context(|| {
                format!("invalid dreaming settings in sleep phase `{phase_name}`")
            })?;
        }
        ensure!(
            !self.retention_suite.as_os_str().is_empty()
                && !self.candidate_directory.as_os_str().is_empty(),
            "sleep phase `{phase_name}` requires retention_suite and candidate_directory"
        );
        validate_sha256_identity(
            &self.retention_suite_sha256,
            "sleep retention suite identity",
        )
        .with_context(|| {
            format!("sleep phase `{phase_name}` has an invalid retention-suite hash")
        })?;
        self.tensor_config().validate().with_context(|| {
            format!("invalid tensor sleep training settings in phase `{phase_name}`")
        })?;
        ensure!(
            self.retention.suite_hash == self.retention_suite_sha256,
            "sleep phase `{phase_name}` retention evaluator is not bound to retention_suite_sha256"
        );
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        if self.retention_suite.is_relative() {
            self.retention_suite = base.join(&self.retention_suite);
        }
        if self.candidate_directory.is_relative() {
            self.candidate_directory = base.join(&self.candidate_directory);
        }
    }

    pub fn tensor_config(&self) -> TensorConsolidationConfig {
        TensorConsolidationConfig {
            knowledge: self.knowledge_seeding.clone(),
            imitation: self.imitation.clone(),
            retention: self.retention.clone(),
            receiver_learning_rate: self.receiver_learning_rate,
            receiver_weight_decay: self.receiver_weight_decay,
            grpo_clip_epsilon: self.grpo_clip_epsilon,
            grpo_advantage_epsilon: self.grpo_advantage_epsilon,
            grpo_kl_coefficient: self.grpo_kl_coefficient,
        }
    }
}

impl QuantizationConfig {
    fn validate(&self, phase_name: &str) -> Result<()> {
        ensure!(
            self.group_size == 128,
            "workflow phase `{phase_name}` quantization group_size must be 128 for a *_g128 format"
        );
        ensure!(
            self.end_step
                .is_none_or(|end_step| end_step > self.start_step),
            "workflow phase `{phase_name}` quantization end_step must be greater than start_step"
        );
        match &self.training {
            QuantizationTraining::Qat { .. } => {}
            QuantizationTraining::Distillation {
                teacher_checkpoint,
                teacher_sha256,
                temperature,
                loss_weight,
            } => {
                ensure!(
                    !teacher_checkpoint.as_os_str().is_empty(),
                    "workflow phase `{phase_name}` quantization distillation requires teacher_checkpoint"
                );
                validate_sha256_identity(teacher_sha256, "quantization teacher identity")
                    .with_context(|| {
                        format!(
                            "workflow phase `{phase_name}` has an invalid quantization teacher hash"
                        )
                    })?;
                ensure!(
                    temperature.is_finite() && *temperature > 0.0,
                    "workflow phase `{phase_name}` quantization temperature must be finite and positive"
                );
                ensure!(
                    loss_weight.is_finite() && *loss_weight >= 0.0,
                    "workflow phase `{phase_name}` quantization loss_weight must be finite and non-negative"
                );
            }
        }
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        if let QuantizationTraining::Distillation {
            teacher_checkpoint, ..
        } = &mut self.training
            && teacher_checkpoint.is_relative()
        {
            *teacher_checkpoint = base.join(&*teacher_checkpoint);
        }
    }
}

/// Memory-tier update policy for optimizer-bearing workflow phases.
///
/// `wake_only` is the capacity- and cadence-matched sleep ablation. It keeps
/// each tier's independent optimizer and configured update period, but applies
/// due prospective base updates directly, fastest-to-slowest. It never runs
/// consolidation, transfers or reclaims reserve slots, or generates dreams.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum MemoryUpdateMode {
    WakeOnly {
        schedule: SleepSchedule,
        #[serde(default)]
        tier_optimizer: TierOptimizerConfig,
    },
}

impl MemoryUpdateMode {
    pub fn schedule(&self) -> &SleepSchedule {
        match self {
            Self::WakeOnly { schedule, .. } => schedule,
        }
    }

    pub fn tier_optimizer(&self) -> &TierOptimizerConfig {
        match self {
            Self::WakeOnly { tier_optimizer, .. } => tier_optimizer,
        }
    }

    pub fn validate(&self, phase_name: &str) -> Result<()> {
        match self {
            Self::WakeOnly {
                schedule,
                tier_optimizer,
            } => {
                schedule.validate().with_context(|| {
                    format!("invalid wake_only memory schedule in workflow phase `{phase_name}`")
                })?;
                tier_optimizer.validate().with_context(|| {
                    format!("invalid wake_only tier optimizer in workflow phase `{phase_name}`")
                })?;
            }
        }
        Ok(())
    }
}

/// Serialized phase definition. Algorithm-specific knobs are namespaced under
/// `parameters`; task packages never interpret them.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseV2 {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: PhaseKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_accumulation: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epochs: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_buffer: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<usize>,
    /// Cap a data-derived epoch plan without asserting that the corpus can
    /// supply exactly this many optimizer steps. Mutually exclusive with
    /// `steps`: `steps` is an exact request, while `max_steps` is
    /// `min(natural_epoch_steps, max_steps)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_rate_scale: Option<f64>,
    /// Strict, typed algorithm and frozen-model identities for preference,
    /// distillation, and RL phases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_training: Option<PostTrainingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sleep: Option<InModelSleepConfig>,
    /// Periodic in-model sleep hook evaluated at this optimization phase's
    /// optimizer-step or model-token boundaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub periodic_sleep: Option<InModelSleepConfig>,
    /// Explicit non-sleep update policy for memory tiers. Mutually exclusive
    /// with `periodic_sleep`; ordinary models leave both fields absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_update_mode: Option<MemoryUpdateMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<QuantizationConfig>,
    /// Content-addressed inputs for the trainer-owned release gate. Unlike
    /// algorithm-neutral `parameters`, this contract cannot be delegated to a
    /// phase worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotion: Option<PromotionConfig>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, serde_json::Value>,
}

impl PhaseV2 {
    pub fn epochs_or_default(&self) -> usize {
        self.epochs.unwrap_or(1)
    }

    pub fn shuffle_buffer_or_default(&self) -> usize {
        self.shuffle_buffer.unwrap_or(8192)
    }

    pub fn loss_weight_or_default(&self) -> f64 {
        self.loss_weight.unwrap_or(1.0)
    }

    pub fn learning_rate_scale_or_default(&self) -> f64 {
        self.learning_rate_scale.unwrap_or(1.0)
    }

    fn validate(&self) -> Result<()> {
        let name = &self.name;
        validate_workflow_identifier(name, "workflow phase name", MAX_PHASE_NAME_BYTES)?;
        if let Some(task) = &self.task {
            task.validate()
                .with_context(|| format!("invalid task in workflow phase `{name}`"))?;
        }

        validate_positive(self.sequence_length, "sequence_length", name)?;
        validate_positive(self.batch_size, "batch_size", name)?;
        validate_positive(self.gradient_accumulation, "gradient_accumulation", name)?;
        validate_positive(self.epochs, "epochs", name)?;
        validate_positive(self.shuffle_buffer, "shuffle_buffer", name)?;
        validate_positive(self.steps, "steps", name)?;
        validate_positive(self.max_steps, "max_steps", name)?;
        ensure!(
            self.steps.is_none() || self.max_steps.is_none(),
            "workflow phase `{name}` cannot set both steps and max_steps"
        );
        ensure!(
            self.max_steps.is_none()
                || matches!(
                    self.kind,
                    PhaseKind::Pretrain
                        | PhaseKind::ContinuedPretrain
                        | PhaseKind::Sft
                        | PhaseKind::Quantization
                ),
            "workflow phase `{name}` ({}) cannot set max_steps because its executor does not implement adaptive data-derived step caps",
            self.kind.name()
        );
        ensure!(
            self.batch_size
                .is_none_or(|value| value <= MAX_PHASE_BATCH_SIZE),
            "workflow phase `{name}` batch_size exceeds the {MAX_PHASE_BATCH_SIZE}-sample limit"
        );
        ensure!(
            self.gradient_accumulation
                .is_none_or(|value| value <= MAX_PHASE_GRADIENT_ACCUMULATION),
            "workflow phase `{name}` gradient_accumulation exceeds the {MAX_PHASE_GRADIENT_ACCUMULATION}-microbatch limit"
        );
        ensure!(
            self.shuffle_buffer
                .is_none_or(|value| value <= MAX_PHASE_SHUFFLE_SAMPLES),
            "workflow phase `{name}` shuffle_buffer exceeds the {MAX_PHASE_SHUFFLE_SAMPLES}-sample limit"
        );
        if let Some(loss_weight) = self.loss_weight {
            ensure!(
                loss_weight.is_finite() && loss_weight > 0.0,
                "workflow phase `{name}` loss_weight must be finite and positive"
            );
        }
        if let Some(scale) = self.learning_rate_scale {
            ensure!(
                scale.is_finite() && scale > 0.0,
                "workflow phase `{name}` learning_rate_scale must be finite and positive"
            );
        }

        if self.kind.uses_task_data() {
            ensure!(
                self.task.is_some(),
                "workflow phase `{name}` ({}) requires a task",
                self.kind.name()
            );
            ensure!(
                self.data
                    .as_ref()
                    .is_some_and(|path| !path.as_os_str().is_empty()),
                "workflow phase `{name}` ({}) requires a data path",
                self.kind.name()
            );
            ensure!(
                self.sequence_length.is_some(),
                "workflow phase `{name}` ({}) requires sequence_length",
                self.kind.name()
            );
            ensure!(
                self.batch_size.is_some(),
                "workflow phase `{name}` ({}) requires batch_size",
                self.kind.name()
            );
            let sequence_length = self.sequence_length.expect("required above");
            let batch_size = self.batch_size.expect("required above");
            let sequences_per_example = self
                .task
                .as_ref()
                .expect("required above")
                .maximum_model_sequences_per_example();
            ensure!(
                sequence_length
                    .checked_mul(batch_size)
                    .and_then(|tokens| tokens.checked_mul(sequences_per_example))
                    .is_some_and(|tokens| tokens <= MAX_PHASE_BATCH_TOKENS),
                "workflow phase `{name}` batch geometry exceeds the {MAX_PHASE_BATCH_TOKENS}-token materialization limit"
            );
            let shuffle_buffer = self.shuffle_buffer_or_default();
            ensure!(
                shuffle_buffer <= MAX_PHASE_SHUFFLE_SAMPLES
                    && sequence_length
                        .checked_mul(shuffle_buffer)
                        .and_then(|tokens| tokens.checked_mul(sequences_per_example))
                        .is_some_and(|tokens| tokens <= MAX_PHASE_SHUFFLE_TOKENS),
                "workflow phase `{name}` shuffle geometry exceeds the {MAX_PHASE_SHUFFLE_TOKENS}-token buffer limit"
            );
        } else {
            ensure!(
                self.task.is_none() && self.data.is_none(),
                "workflow phase `{name}` ({}) must not read task data",
                self.kind.name()
            );
        }

        match self.kind {
            PhaseKind::Pretrain
            | PhaseKind::ContinuedPretrain
            | PhaseKind::Sft
            | PhaseKind::Preference
            | PhaseKind::Rl
            | PhaseKind::Distillation
            | PhaseKind::Quantization => {
                ensure!(
                    self.gradient_accumulation.is_some(),
                    "workflow phase `{name}` ({}) requires gradient_accumulation",
                    self.kind.name()
                );
            }
            PhaseKind::Evaluation => {
                ensure!(
                    self.gradient_accumulation.is_none(),
                    "workflow phase `{name}` (evaluation) must not set gradient_accumulation"
                );
                ensure!(
                    self.loss_weight.is_none() && self.learning_rate_scale.is_none(),
                    "workflow phase `{name}` (evaluation) must not set optimizer weights"
                );
            }
            PhaseKind::Sleep | PhaseKind::Promotion => {
                ensure!(
                    self.sequence_length.is_none()
                        && self.batch_size.is_none()
                        && self.gradient_accumulation.is_none()
                        && self.epochs.is_none()
                        && self.shuffle_buffer.is_none()
                        && self.steps.is_none()
                        && self.max_steps.is_none()
                        && self.loss_weight.is_none()
                        && self.learning_rate_scale.is_none(),
                    "workflow phase `{name}` ({}) must not set task execution or optimizer geometry",
                    self.kind.name()
                );
            }
        }

        let execution = self.task.as_ref().map(|task| task.contract().execution);
        match self.kind {
            PhaseKind::Sft => ensure!(
                execution == Some(TaskExecution::SupervisedGeneration),
                "workflow phase `{name}` (sft) requires a supervised-generation task"
            ),
            PhaseKind::Distillation => ensure!(
                matches!(
                    execution,
                    Some(
                        TaskExecution::AutoregressiveTokenPrediction
                            | TaskExecution::SupervisedGeneration
                    )
                ),
                "workflow phase `{name}` (distillation) requires an autoregressive or supervised-generation task"
            ),
            PhaseKind::Preference => ensure!(
                execution == Some(TaskExecution::PairwisePreference),
                "workflow phase `{name}` (preference) requires a pairwise-preference task"
            ),
            PhaseKind::Rl => ensure!(
                execution == Some(TaskExecution::VerifiableReward),
                "workflow phase `{name}` (rl) requires a verifiable-reward task"
            ),
            _ => {}
        }

        match (&self.kind, &self.post_training) {
            (PhaseKind::Preference, Some(PostTrainingConfig::Dpo { .. }))
            | (PhaseKind::Distillation, Some(PostTrainingConfig::ForwardKl { .. }))
            | (PhaseKind::Rl, Some(PostTrainingConfig::Grpo { .. })) => {
                self.post_training
                    .as_ref()
                    .expect("matched present post-training config")
                    .validate()
                    .with_context(|| {
                        format!("invalid post-training settings in workflow phase `{name}`")
                    })?;
            }
            (PhaseKind::Preference, Some(_)) => {
                bail!("workflow phase `{name}` (preference) requires a DPO post-training algorithm")
            }
            (PhaseKind::Distillation, Some(_)) => bail!(
                "workflow phase `{name}` (distillation) requires a forward-KL post-training algorithm"
            ),
            (PhaseKind::Rl, Some(_)) => {
                bail!("workflow phase `{name}` (rl) requires a GRPO post-training algorithm")
            }
            (PhaseKind::Preference | PhaseKind::Distillation | PhaseKind::Rl, None) => bail!(
                "workflow phase `{name}` ({}) requires typed post_training settings",
                self.kind.name()
            ),
            (_, Some(_)) => bail!(
                "workflow phase `{name}` ({}) must not set post_training settings",
                self.kind.name()
            ),
            (_, None) => {}
        }
        if let Some(PostTrainingConfig::Grpo { sampling, .. }) = &self.post_training {
            ensure!(
                self.sequence_length
                    .is_some_and(|sequence_length| sampling.max_new_tokens <= sequence_length),
                "workflow phase `{name}` GRPO max_new_tokens must not exceed sequence_length"
            );
        }

        match (&self.kind, &self.quantization) {
            (PhaseKind::Quantization, Some(config)) => config.validate(name)?,
            (PhaseKind::Quantization, None) => {
                bail!("workflow phase `{name}` (quantization) requires quantization settings")
            }
            (_, Some(_)) => bail!(
                "workflow phase `{name}` ({}) must not set quantization settings",
                self.kind.name()
            ),
            (_, None) => {}
        }
        match (&self.kind, &self.promotion) {
            (PhaseKind::Promotion, Some(config)) => {
                config.validate(name)?;
                ensure!(
                    self.parameters.is_empty(),
                    "workflow phase `{name}` (promotion) must use typed promotion settings, not arbitrary parameters"
                );
            }
            (PhaseKind::Promotion, None) => {
                bail!("workflow phase `{name}` (promotion) requires typed promotion settings")
            }
            (_, Some(_)) => bail!(
                "workflow phase `{name}` ({}) must not set promotion settings",
                self.kind.name()
            ),
            (_, None) => {}
        }
        match (&self.kind, &self.sleep) {
            (PhaseKind::Sleep, Some(config)) => {
                config.validate(name)?;
                ensure!(
                    config
                        .standalone_trigger_clock
                        .is_some_and(|clock| clock > 0),
                    "workflow phase `{name}` (sleep) requires a positive standalone_trigger_clock"
                );
            }
            (PhaseKind::Sleep, None) => {
                bail!("workflow phase `{name}` (sleep) requires typed sleep settings")
            }
            (_, Some(_)) => bail!(
                "workflow phase `{name}` ({}) must not set sleep settings",
                self.kind.name()
            ),
            (_, None) => {}
        }
        if let Some(config) = &self.periodic_sleep {
            ensure!(
                matches!(
                    self.kind,
                    PhaseKind::Pretrain
                        | PhaseKind::ContinuedPretrain
                        | PhaseKind::Sft
                        | PhaseKind::Preference
                        | PhaseKind::Rl
                        | PhaseKind::Distillation
                        | PhaseKind::Quantization
                ),
                "workflow phase `{name}` ({}) cannot install a periodic sleep hook",
                self.kind.name()
            );
            config.validate(name)?;
            ensure!(
                config.standalone_trigger_clock.is_none(),
                "workflow phase `{name}` periodic_sleep must not set standalone_trigger_clock"
            );
        }
        if let Some(mode) = &self.memory_update_mode {
            ensure!(
                self.kind.uses_optimizer(),
                "workflow phase `{name}` ({}) cannot set memory_update_mode",
                self.kind.name()
            );
            ensure!(
                self.periodic_sleep.is_none(),
                "workflow phase `{name}` cannot combine memory_update_mode with periodic_sleep"
            );
            mode.validate(name)?;
        }
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        if let Some(data) = &mut self.data
            && data.is_relative()
        {
            *data = base.join(&*data);
        }
        if let Some(quantization) = &mut self.quantization {
            quantization.resolve_paths(base);
        }
        if let Some(post_training) = &mut self.post_training {
            post_training.resolve_paths(base);
        }
        if let Some(sleep) = &mut self.sleep {
            sleep.resolve_paths(base);
        }
        if let Some(sleep) = &mut self.periodic_sleep {
            sleep.resolve_paths(base);
        }
        if let Some(promotion) = &mut self.promotion {
            promotion.resolve_paths(base);
        }
    }
}

fn validate_positive(value: Option<usize>, field: &str, phase_name: &str) -> Result<()> {
    ensure!(
        value.is_none_or(|value| value > 0),
        "workflow phase `{phase_name}` {field} must be positive when set"
    );
    Ok(())
}

fn validate_workflow_identifier(value: &str, label: &str, maximum_bytes: usize) -> Result<()> {
    ensure!(!value.is_empty(), "{label} must not be empty");
    ensure!(
        value.trim() == value,
        "{label} must not have leading or trailing whitespace"
    );
    ensure!(
        !value.chars().any(char::is_control),
        "{label} must not contain control characters"
    );
    ensure!(
        value.len() <= maximum_bytes,
        "{label} exceeds the {maximum_bytes}-byte limit"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowV2 {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub phases: Vec<PhaseV2>,
}

impl WorkflowV2 {
    pub fn validate(&self) -> Result<()> {
        validate_workflow_parts(self.version, self.name.as_deref(), &self.phases)
    }

    pub fn resolve(mut self, source: &Path) -> Result<ResolvedWorkflow> {
        self.validate()?;
        let base = source.parent().unwrap_or_else(|| Path::new("."));
        for phase in &mut self.phases {
            phase.resolve_paths(base);
        }
        Ok(ResolvedWorkflow {
            version: self.version,
            name: self.name,
            phases: self.phases,
        })
    }
}

fn validate_workflow_parts(version: u32, name: Option<&str>, phases: &[PhaseV2]) -> Result<()> {
    ensure!(
        version == WORKFLOW_VERSION,
        "unsupported workflow version {}; this build supports version {WORKFLOW_VERSION}",
        version
    );
    if let Some(name) = name {
        validate_workflow_identifier(name, "workflow name", MAX_WORKFLOW_NAME_BYTES)?;
    }
    ensure!(!phases.is_empty(), "workflow contains no phases");
    ensure!(
        phases.len() <= MAX_WORKFLOW_PHASES,
        "workflow contains {} phases, exceeding the {MAX_WORKFLOW_PHASES}-phase limit",
        phases.len()
    );
    let mut names = BTreeSet::new();
    for phase in phases {
        ensure!(
            names.insert(phase.name.as_str()),
            "duplicate workflow phase name `{}`",
            phase.name
        );
        phase.validate()?;
    }
    Ok(())
}

/// Fully validated workflow with file references resolved against its source.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedWorkflow {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub phases: Vec<PhaseV2>,
}

impl ResolvedWorkflow {
    pub fn validate(&self) -> Result<()> {
        validate_workflow_parts(self.version, self.name.as_deref(), &self.phases)
    }
}

/// Validate a one-based retrieval read-out layer against a model topology.
///
/// `context` names the caller, such as a quoted workflow phase for training or
/// `eval` for forward-only evaluation, so both paths report the same failures
/// with the same wording. Retrieval embeddings are read after `layer`, so at
/// least one full-attention layer must precede the read-out.
pub fn validate_retrieval_layer_for_model(
    layer: usize,
    model: &ModelDef,
    context: &str,
) -> Result<()> {
    ensure!(
        layer > 0 && layer <= model.num_layers,
        "{context} requests retrieval layer {layer}, model has {} layers",
        model.num_layers
    );
    ensure!(
        (0..layer).any(|index| {
            let block = model.block_for_layer(index);
            !block.is_ssm() && block.attention.window_size.is_none()
        }),
        "{context} retrieval layer {layer} has no full-attention layer at or before it"
    );
    Ok(())
}

/// Validate one complete workflow against the exact MAL topology it will
/// mutate. This is deliberately separate from the generic WorkflowV2 schema:
/// orchestration can remain model-neutral, while launch tooling can fail before
/// creating runtime state when a recipe and model do not form one executable
/// training topology.
pub fn validate_workflow_for_model(workflow: &ResolvedWorkflow, model: &ModelDef) -> Result<()> {
    workflow.validate()?;
    ensure!(model.num_layers > 0, "model contains no layers");

    for phase in &workflow.phases {
        if let Some(sequence_length) = phase.sequence_length {
            ensure!(
                sequence_length <= model.max_seq_len,
                "workflow phase `{}` sequence_length {sequence_length} exceeds model max_seq_len {}",
                phase.name,
                model.max_seq_len
            );
        }
        if let Some(layer) = phase.task.as_ref().and_then(TaskConfig::retrieval_layer) {
            validate_retrieval_layer_for_model(
                layer,
                model,
                &format!("workflow phase `{}`", phase.name),
            )?;
        }
    }

    let memory_layers = (0..model.num_layers)
        .filter(|&layer| model.block_for_layer(layer).memory.is_some())
        .count();
    ensure!(
        memory_layers == 0 || memory_layers == model.num_layers,
        "model mixes {memory_layers} memory layers with {} ordinary-FFN layers; in-model sleep requires an explicit memory hierarchy in every layer and has no implicit topology transition",
        model.num_layers - memory_layers
    );

    let optimizer_phases = workflow
        .phases
        .iter()
        .filter(|phase| phase.kind.uses_optimizer())
        .collect::<Vec<_>>();
    let standalone_sleep = workflow
        .phases
        .iter()
        .filter_map(|phase| phase.sleep.as_ref().map(|sleep| (phase, sleep)))
        .collect::<Vec<_>>();
    if memory_layers == 0 {
        ensure!(
            optimizer_phases.iter().all(|phase| {
                phase.periodic_sleep.is_none() && phase.memory_update_mode.is_none()
            }) && standalone_sleep.is_empty(),
            "memory update modes and in-model sleep require a MAL model with an explicit memory hierarchy; no implicit topology upgrade is performed"
        );
        return Ok(());
    }

    if !optimizer_phases.is_empty() {
        let periodic_count = optimizer_phases
            .iter()
            .filter(|phase| phase.periodic_sleep.is_some())
            .count();
        let wake_only_count = optimizer_phases
            .iter()
            .filter(|phase| phase.memory_update_mode.is_some())
            .count();
        ensure!(
            periodic_count == optimizer_phases.len() || wake_only_count == optimizer_phases.len(),
            "memory MAL models require one identical update policy on every optimizer-bearing phase: periodic_sleep or memory_update_mode wake_only, including QAT"
        );
        if periodic_count == optimizer_phases.len() {
            let reference = optimizer_phases[0]
                .periodic_sleep
                .as_ref()
                .expect("checked every optimizer phase has periodic sleep");
            ensure!(
                optimizer_phases
                    .iter()
                    .all(|phase| phase.periodic_sleep.as_ref() == Some(reference)),
                "all optimizer-bearing phases in one memory-model workflow must use an identical periodic_sleep configuration"
            );
            validate_sleep_schedule_for_model(
                model,
                &reference.schedule,
                &optimizer_phases[0].name,
            )?;
            validate_dreaming_topology(model, reference, &optimizer_phases[0].name)?;
        } else {
            ensure!(
                standalone_sleep.is_empty(),
                "memory_update_mode wake_only is a no-sleep ablation and cannot be combined with standalone sleep phases"
            );
            let reference = optimizer_phases[0]
                .memory_update_mode
                .as_ref()
                .expect("checked every optimizer phase has a memory update mode");
            ensure!(
                optimizer_phases
                    .iter()
                    .all(|phase| phase.memory_update_mode.as_ref() == Some(reference)),
                "all optimizer-bearing phases in one memory-model workflow must use an identical memory_update_mode configuration"
            );
            validate_sleep_schedule_for_model(
                model,
                reference.schedule(),
                &optimizer_phases[0].name,
            )?;
        }
    }
    for (phase, sleep) in standalone_sleep {
        validate_sleep_schedule_for_model(model, &sleep.schedule, &phase.name)?;
        validate_dreaming_topology(model, sleep, &phase.name)?;
    }
    Ok(())
}

fn validate_dreaming_topology(
    model: &ModelDef,
    sleep: &InModelSleepConfig,
    phase_name: &str,
) -> Result<()> {
    if sleep.dreaming.is_none() {
        return Ok(());
    }
    let has_exploration_route = (0..model.num_layers).any(|layer| {
        model
            .block_for_layer(layer)
            .memory
            .as_ref()
            .is_some_and(|memory| {
                memory.tiers.iter().any(|tier| {
                    tier.ffn
                        .moe
                        .as_ref()
                        .is_some_and(|moe| moe.experts > moe.top_k)
                })
            })
    });
    ensure!(
        has_exploration_route,
        "workflow phase `{phase_name}` enables Dreaming, but the MAL memory hierarchy has no persistent FFN MoE with an expert outside ordinary top-k"
    );
    Ok(())
}

/// Verify that a sleep schedule names exactly the preallocated tiers and
/// reserve capacities present in every model layer.
pub fn validate_sleep_schedule_for_model(
    model: &ModelDef,
    schedule: &SleepSchedule,
    phase_name: &str,
) -> Result<()> {
    for layer in 0..model.num_layers {
        let memory = model
            .block_for_layer(layer)
            .memory
            .as_ref()
            .with_context(|| {
                format!(
                    "workflow phase `{phase_name}` enables sleep, but model layer {layer} has no memory hierarchy"
                )
            })?;
        ensure!(
            memory.tiers.len() == schedule.tiers.len(),
            "workflow phase `{phase_name}` declares {} memory tiers, but model layer {layer} defines {}",
            schedule.tiers.len(),
            memory.tiers.len()
        );
        for (tier_index, (configured, defined)) in
            schedule.tiers.iter().zip(&memory.tiers).enumerate()
        {
            ensure!(
                configured.id == defined.name,
                "workflow phase `{phase_name}` tier {tier_index} is `{}`, but model layer {layer} defines `{}`",
                configured.id,
                defined.name
            );
            ensure!(
                configured.reserve_slots == defined.reserve_experts.capacity,
                "workflow phase `{phase_name}` tier `{}` reserves {} slots, but model layer {layer} preallocates {}",
                configured.id,
                configured.reserve_slots,
                defined.reserve_experts.capacity
            );
        }
    }
    Ok(())
}

pub fn load_workflow(path: &Path) -> Result<ResolvedWorkflow> {
    let bytes = read_regular_bounded(path, MAX_WORKFLOW_JSON_BYTES, "workflow JSON")
        .with_context(|| format!("failed to read workflow {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid workflow JSON in {}", path.display()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("workflow {} has no integer `version`", path.display()))?;
    ensure!(
        version == u64::from(WORKFLOW_VERSION),
        "unsupported workflow version {version}; this build supports version {WORKFLOW_VERSION}"
    );
    let workflow: WorkflowV2 = serde_json::from_value(value)
        .with_context(|| format!("invalid workflow JSON in {}", path.display()))?;
    workflow.resolve(path)
}

#[cfg(test)]
pub(crate) fn test_promotion_config() -> serde_json::Value {
    let reference = |index: usize, stem: &str| serde_json::json!({ "path": format!("promotion/{stem}-{index}.json") });
    serde_json::json!({
        "selected_run": reference(0, "selected"),
        "comparison_runs": (1..=10)
            .map(|index| reference(index, "comparison"))
            .collect::<Vec<_>>(),
        "resources": reference(11, "resources"),
        "policy": reference(12, "policy"),
        "artifact_directory": "promotion/decisions"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training_phase(name: &str, kind: &str, task: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "type": kind,
            "task": task,
            "data": format!("data/{name}.jsonl"),
            "sequence_length": 512,
            "batch_size": 8,
            "gradient_accumulation": 2,
            "steps": 10
        })
    }

    #[test]
    fn workflow_v2_supports_the_complete_phase_lifecycle() {
        let phases = vec![
            training_phase(
                "pretrain",
                "pretrain",
                serde_json::json!({"type": "causal_lm"}),
            ),
            training_phase(
                "continued",
                "continued_pretrain",
                serde_json::json!({"type": "retrieval_representation"}),
            ),
            training_phase(
                "sft",
                "sft",
                serde_json::json!({"type": "instruction_tuning"}),
            ),
            {
                let mut phase = training_phase(
                    "preference",
                    "preference",
                    serde_json::json!({"type": "pairwise_preference"}),
                );
                phase["post_training"] = serde_json::json!({
                    "algorithm": "dpo",
                    "reference": {
                        "adapter": "summa_checkpoint",
                        "artifact": "reference",
                        "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                    }
                });
                phase
            },
            {
                let mut phase = training_phase(
                    "rl",
                    "rl",
                    serde_json::json!({
                        "type": "verifiable_rl",
                        "verifier": {"adapter": "exact_answer"}
                    }),
                );
                phase["post_training"] = serde_json::json!({
                    "algorithm": "grpo",
                    "reference": {
                        "adapter": "summa_checkpoint",
                        "artifact": "reference",
                        "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                    },
                    "sampling": {"max_new_tokens": 128}
                });
                phase
            },
            {
                let mut phase = training_phase(
                    "distill",
                    "distillation",
                    serde_json::json!({"type": "instruction_tuning"}),
                );
                phase["post_training"] = serde_json::json!({
                    "algorithm": "forward_kl",
                    "teacher": {
                        "adapter": "summa_checkpoint",
                        "artifact": "teacher",
                        "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                    }
                });
                phase
            },
            serde_json::json!({
                "name": "sleep",
                "type": "sleep",
                "sleep": {
                    "standalone_trigger_clock": 100,
                    "schedule": {
                        "clock": "optimizer_steps",
                        "terminal_consolidation": "distill_into_base_v1",
                        "tiers": [
                            {"id": "fast", "update_period": 100, "reserve_slots": 2},
                            {"id": "medium", "update_period": 400, "reserve_slots": 4},
                            {"id": "slow", "update_period": 3200, "reserve_slots": 8}
                        ]
                    },
                    "knowledge_seeding": {
                        "chunk_tokens": 512,
                        "teacher_rollouts": 4,
                        "detached_student_rollouts": 4,
                        "temperature": 1.0,
                        "forward_kl_weight": 1.0
                    },
                    "imitation": {
                        "semantic_judge_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                        "semantic_weight": 0.7,
                        "maximum_edit_distance": 32,
                        "grpo_group_size": 8
                    },
                    "dreaming": {
                        "candidate_count": 64,
                        "retain_top": 8,
                        "retain_random": 2,
                            "lora_rank": 64,
                            "lora_alpha": 128,
                            "restem_iterations": 1,
                            "selector_version": "gradient-cosine-v1",
                            "reference_set_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
                            "trial_evaluator_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
                    },
                    "retention_suite": "acceptance/retention.json",
                    "retention_suite_sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                    "retention": {
                        "evaluator_hash": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                        "suite_hash": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                        "max_anchor_forward_kl": 0.05,
                        "max_anchor_regression": 0.01,
                        "min_incorporation_gain": 0.0
                    },
                    "receiver_learning_rate": 0.0001,
                    "receiver_weight_decay": 0.01,
                    "grpo_clip_epsilon": 0.2,
                    "grpo_advantage_epsilon": 0.000001,
                    "grpo_kl_coefficient": 0.04,
                    "candidate_directory": "candidates"
                },
                "parameters": {"policy": "cms_v1"}
            }),
            {
                let mut phase = training_phase(
                    "quantize",
                    "quantization",
                    serde_json::json!({"type": "causal_lm"}),
                );
                phase["quantization"] = serde_json::json!({
                    "format": "binary_g128",
                    "training": {"type": "qat"}
                });
                phase
            },
            serde_json::json!({
                "name": "evaluate",
                "type": "evaluation",
                "task": {"type": "qa_reasoning"},
                "data": "data/eval.jsonl",
                "sequence_length": 512,
                "batch_size": 8
            }),
            serde_json::json!({
                "name": "promote",
                "type": "promotion",
                "promotion": test_promotion_config()
            }),
        ];
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "name": "full-lifecycle",
            "phases": phases
        }))
        .unwrap();

        workflow.validate().unwrap();
        assert_eq!(workflow.phases.len(), 10);
        assert_eq!(workflow.phases[6].kind.class(), PhaseClass::ModelMutation);
        assert_eq!(workflow.phases[9].kind.class(), PhaseClass::Release);
    }

    #[test]
    fn load_workflow_resolves_paths_and_applies_accessible_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "foundation",
                    "type": "pretrain",
                    "task": {"type": "causal_lm"},
                    "data": "data/foundation.jsonl",
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "steps": 10
                }]
            }"#,
        )
        .unwrap();

        let workflow = load_workflow(&path).unwrap();
        let phase = &workflow.phases[0];
        assert_eq!(
            phase.data.as_deref(),
            Some(dir.path().join("data/foundation.jsonl").as_path())
        );
        assert_eq!(phase.epochs_or_default(), 1);
        assert_eq!(phase.shuffle_buffer_or_default(), 8192);
        assert_eq!(phase.loss_weight_or_default(), 1.0);
        assert_eq!(phase.learning_rate_scale_or_default(), 1.0);
    }

    #[test]
    fn education_workflow_fits_the_bounded_shuffle_budget() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.education.example.json");
        load_workflow(&path)
            .unwrap_or_else(|error| panic!("education workflow is invalid: {error:#}"));
    }

    #[test]
    fn sft_retrieval_replay_is_adaptively_capped() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.sft.example.json");
        let workflow = load_workflow(&path)
            .unwrap_or_else(|error| panic!("SFT workflow is invalid: {error:#}"));
        let retrieval = workflow
            .phases
            .iter()
            .filter(|phase| {
                matches!(
                    phase.task.as_ref(),
                    Some(TaskConfig::RetrievalRepresentation { .. })
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(retrieval.len(), 4);
        assert!(retrieval.iter().all(|phase| phase.steps.is_none()));
        assert!(retrieval.iter().all(|phase| phase.max_steps == Some(150)));
    }

    #[test]
    fn incompatible_workflow_versions_are_rejected_without_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(&path, r#"{"version":1,"stages":[]}"#).unwrap();

        let error = load_workflow(&path).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("unsupported workflow version 1"), "{error}");
    }

    #[test]
    fn workflow_file_is_bounded_before_json_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized-workflow.json");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_WORKFLOW_JSON_BYTES + 1).unwrap();
        drop(file);

        let error = load_workflow(&path).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("byte limit"), "{error}");
    }

    #[test]
    fn memory_sized_phase_geometry_is_bounded_without_limiting_steps() {
        let phase = |field: &str, value: usize| {
            let mut phase = training_phase(
                "bounded",
                "pretrain",
                serde_json::json!({"type": "causal_lm"}),
            );
            phase[field] = serde_json::json!(value);
            let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
                "version": 2,
                "phases": [phase]
            }))
            .unwrap();
            workflow.validate().unwrap_err().to_string()
        };

        assert!(phase("batch_size", MAX_PHASE_BATCH_SIZE + 1).contains("batch_size exceeds"));
        assert!(
            phase("gradient_accumulation", MAX_PHASE_GRADIENT_ACCUMULATION + 1)
                .contains("gradient_accumulation exceeds")
        );
        assert!(
            phase("shuffle_buffer", MAX_PHASE_SHUFFLE_SAMPLES + 1)
                .contains("shuffle_buffer exceeds")
        );

        let mut enormous_batch = training_phase(
            "token-budget",
            "pretrain",
            serde_json::json!({"type": "causal_lm"}),
        );
        enormous_batch["sequence_length"] = serde_json::json!(MAX_PHASE_BATCH_TOKENS);
        enormous_batch["batch_size"] = serde_json::json!(2);
        enormous_batch["shuffle_buffer"] = serde_json::json!(1);
        enormous_batch["steps"] = serde_json::json!(usize::MAX);
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [enormous_batch]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("batch geometry"), "{error}");

        let mut retrieval = training_phase(
            "retrieval-memory",
            "continued_pretrain",
            serde_json::json!({"type": "retrieval_representation"}),
        );
        retrieval["sequence_length"] = serde_json::json!(4096);
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [retrieval]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("shuffle geometry"), "{error}");

        let mut long_run = training_phase(
            "long-run",
            "pretrain",
            serde_json::json!({"type": "causal_lm"}),
        );
        long_run["steps"] = serde_json::json!(usize::MAX);
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [long_run]
        }))
        .unwrap();
        workflow.validate().unwrap();
    }

    #[test]
    fn phase_names_and_fields_are_strictly_validated() {
        let duplicate: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [
                training_phase("same", "pretrain", serde_json::json!({"type": "causal_lm"})),
                training_phase("same", "pretrain", serde_json::json!({"type": "causal_lm"}))
            ]
        }))
        .unwrap();
        let error = duplicate.validate().unwrap_err().to_string();
        assert!(error.contains("duplicate workflow phase"), "{error}");

        let unknown = serde_json::from_value::<WorkflowV2>(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "train",
                "type": "pretrain",
                "task": {"type": "causal_lm"},
                "data": "data.jsonl",
                "sequence_length": 32,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "sequnce_length": 32
            }]
        }))
        .unwrap_err()
        .to_string();
        assert!(unknown.contains("sequnce_length"), "{unknown}");

        let mut zero_shuffle = training_phase(
            "zero-shuffle",
            "pretrain",
            serde_json::json!({"type": "causal_lm"}),
        );
        zero_shuffle["shuffle_buffer"] = serde_json::json!(0);
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [zero_shuffle]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("shuffle_buffer must be positive"), "{error}");

        for invalid_name in [" leading", "trailing ", "line\nbreak"] {
            let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
                "version": 2,
                "phases": [training_phase(
                    invalid_name,
                    "pretrain",
                    serde_json::json!({"type": "causal_lm"})
                )]
            }))
            .unwrap();
            let error = workflow.validate().unwrap_err().to_string();
            assert!(
                error.contains("whitespace") || error.contains("control characters"),
                "{error}"
            );
        }

        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [training_phase(
                &"x".repeat(MAX_PHASE_NAME_BYTES + 1),
                "pretrain",
                serde_json::json!({"type": "causal_lm"})
            )]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("byte limit"), "{error}");

        let valid_phase: PhaseV2 = serde_json::from_value(training_phase(
            "bounded",
            "pretrain",
            serde_json::json!({"type": "causal_lm"}),
        ))
        .unwrap();
        let workflow = WorkflowV2 {
            version: WORKFLOW_VERSION,
            name: None,
            phases: vec![valid_phase; MAX_WORKFLOW_PHASES + 1],
        };
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("phase limit"), "{error}");
    }

    #[test]
    fn exact_steps_and_natural_epoch_cap_are_mutually_exclusive() {
        let mut phase = training_phase(
            "bounded-retrieval",
            "continued_pretrain",
            serde_json::json!({"type": "retrieval_representation"}),
        );
        phase["max_steps"] = serde_json::json!(5);
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [phase]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("both steps and max_steps"), "{error}");

        let mut phase = training_phase(
            "bounded-retrieval",
            "continued_pretrain",
            serde_json::json!({"type": "retrieval_representation"}),
        );
        phase.as_object_mut().unwrap().remove("steps");
        phase["max_steps"] = serde_json::json!(0);
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [phase]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("max_steps must be positive"), "{error}");
    }

    #[test]
    fn algorithm_specific_phases_reject_incompatible_task_signals() {
        let preference: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [training_phase(
                "preference",
                "preference",
                serde_json::json!({"type": "causal_lm"})
            )]
        }))
        .unwrap();
        let error = preference.validate().unwrap_err().to_string();
        assert!(error.contains("pairwise-preference task"), "{error}");

        let rl: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [training_phase(
                "rl",
                "rl",
                serde_json::json!({"type": "qa_reasoning"})
            )]
        }))
        .unwrap();
        let error = rl.validate().unwrap_err().to_string();
        assert!(error.contains("verifiable-reward task"), "{error}");

        let mut invalid_distillation = training_phase(
            "distill-ranking",
            "distillation",
            serde_json::json!({"type": "retrieval_ranking"}),
        );
        invalid_distillation["post_training"] = serde_json::json!({
            "algorithm": "forward_kl",
            "teacher": {"adapter": "test", "revision": "teacher-v1"}
        });
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [invalid_distillation]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(
            error.contains("autoregressive or supervised-generation"),
            "{error}"
        );

        let mut supervised_distillation = training_phase(
            "distill-instructions",
            "distillation",
            serde_json::json!({"type": "instruction_tuning"}),
        );
        supervised_distillation["post_training"] = serde_json::json!({
            "algorithm": "forward_kl",
            "teacher": {"adapter": "test", "revision": "teacher-v1"}
        });
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [supervised_distillation]
        }))
        .unwrap();
        workflow.validate().unwrap();

        let missing_config: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [training_phase(
                "preference",
                "preference",
                serde_json::json!({"type": "pairwise_preference"})
            )]
        }))
        .unwrap();
        let error = missing_config.validate().unwrap_err().to_string();
        assert!(error.contains("typed post_training settings"), "{error}");

        let mut wrong_algorithm = training_phase(
            "preference",
            "preference",
            serde_json::json!({"type": "pairwise_preference"}),
        );
        wrong_algorithm["post_training"] = serde_json::json!({
            "algorithm": "grpo",
            "kl_coefficient": 0.0,
            "sampling": {"max_new_tokens": 64}
        });
        let wrong_algorithm: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [wrong_algorithm]
        }))
        .unwrap();
        let error = wrong_algorithm.validate().unwrap_err().to_string();
        assert!(error.contains("requires a DPO"), "{error}");
    }

    #[test]
    fn lifecycle_phases_cannot_silently_consume_or_optimize_data() {
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "sleep",
                "type": "sleep",
                "task": {"type": "causal_lm"},
                "data": "external.jsonl"
            }]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(error.contains("must not read task data"), "{error}");

        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "eval",
                "type": "evaluation",
                "task": {"type": "causal_lm"},
                "data": "eval.txt",
                "sequence_length": 32,
                "batch_size": 2,
                "gradient_accumulation": 1
            }]
        }))
        .unwrap();
        let error = workflow.validate().unwrap_err().to_string();
        assert!(
            error.contains("must not set gradient_accumulation"),
            "{error}"
        );
    }

    #[test]
    fn quantization_configuration_is_typed_and_resolves_teacher_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "quantize",
                    "type": "quantization",
                    "task": {"type": "causal_lm"},
                    "data": "calibration.jsonl",
                    "sequence_length": 128,
                    "batch_size": 4,
                    "gradient_accumulation": 1,
                    "steps": 20,
                    "quantization": {
                        "format": "ternary_g128",
                        "training": {
                            "type": "distillation",
                            "teacher_checkpoint": "teacher.safetensors",
                            "teacher_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                            "temperature": 2.0
                        }
                    }
                }]
            }"#,
        )
        .unwrap();

        let workflow = load_workflow(&path).unwrap();
        let quantization = workflow.phases[0].quantization.as_ref().unwrap();
        assert_eq!(quantization.group_size, 128);
        let QuantizationTraining::Distillation {
            teacher_checkpoint, ..
        } = &quantization.training
        else {
            panic!("expected distillation")
        };
        assert_eq!(teacher_checkpoint, &dir.path().join("teacher.safetensors"));
    }

    fn wake_only_workflow() -> ResolvedWorkflow {
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "wake",
                "type": "pretrain",
                "task": {"type": "causal_lm"},
                "data": "wake.jsonl",
                "sequence_length": 8,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "steps": 2,
                "memory_update_mode": {
                    "type": "wake_only",
                    "schedule": {
                        "clock": "optimizer_steps",
                        "terminal_consolidation": "distill_into_base_v1",
                        "tiers": [
                            {"id": "fast", "update_period": 1, "reserve_slots": 1},
                            {"id": "slow", "update_period": 2, "reserve_slots": 2}
                        ]
                    }
                }
            }]
        }))
        .unwrap();
        workflow.resolve(Path::new("workflow.json")).unwrap()
    }

    fn memory_model(mixed: bool) -> ModelDef {
        let topology = if mixed {
            "block: sleeping pattern: [ordinary, sleeping]"
        } else {
            "block: sleeping"
        };
        summa_llm::parse_mal(&format!(
            r#"
            ffn base {{ hidden_dim: 12 activation: swiglu dropout: 0.0 }}
            memory cms {{
                tier fast {{
                    ffn: base
                    reserve_experts {{ capacity: 1 rank: 3 top_k: 1 }}
                }}
                tier slow {{
                    ffn: base residual_init: zero
                    reserve_experts {{ capacity: 2 rank: 3 top_k: 1 }}
                }}
            }}
            block ordinary {{
                attention: {{ num_heads: 1 dropout: 0.0 position_encoding: none }}
                ffn: base dropout: 0.0
            }}
            block sleeping {{
                attention: {{ num_heads: 1 dropout: 0.0 position_encoding: none }}
                memory: cms dropout: 0.0
            }}
            model memory-test {{
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 2
                {topology}
            }}
            "#
        ))
        .unwrap()
    }

    fn dreaming_memory_model() -> ModelDef {
        summa_llm::parse_mal(
            r#"
            ffn dense { hidden_dim: 12 activation: swiglu dropout: 0.0 }
            ffn routed {
                hidden_dim: 12 activation: swiglu dropout: 0.0
                moe { experts: 2 top_k: 1 }
            }
            memory cms {
                tier fast {
                    ffn: routed
                    reserve_experts { capacity: 1 rank: 3 top_k: 1 }
                }
                tier slow {
                    ffn: dense residual_init: zero
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            block sleeping {
                attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
                memory: cms dropout: 0.0
            }
            model memory-test {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 2
                block: sleeping
            }
            "#,
        )
        .unwrap()
    }

    fn periodic_sleep_config(with_dreaming: bool) -> InModelSleepConfig {
        let mut value = serde_json::json!({
            "schedule": {
                "clock": "optimizer_steps",
                "terminal_consolidation": "distill_into_base_v1",
                "tiers": [
                    {"id": "fast", "update_period": 1, "reserve_slots": 1},
                    {"id": "slow", "update_period": 2, "reserve_slots": 2}
                ]
            },
            "knowledge_seeding": {
                "chunk_tokens": 8,
                "teacher_rollouts": 1,
                "detached_student_rollouts": 1,
                "temperature": 1.0,
                "forward_kl_weight": 1.0
            },
            "imitation": {
                "semantic_judge_hash": format!("sha256:{}", "0".repeat(64)),
                "semantic_weight": 1.0,
                "maximum_edit_distance": 8,
                "grpo_group_size": 2
            },
            "retention_suite": "retention.json",
            "retention_suite_sha256": format!("sha256:{}", "0".repeat(64)),
            "retention": {
                "evaluator_hash": format!("sha256:{}", "0".repeat(64)),
                "suite_hash": format!("sha256:{}", "0".repeat(64)),
                "max_anchor_forward_kl": 0.1,
                "max_anchor_regression": 0.1,
                "min_incorporation_gain": 0.0
            },
            "receiver_learning_rate": 0.0001,
            "receiver_weight_decay": 0.0,
            "grpo_clip_epsilon": 0.2,
            "grpo_advantage_epsilon": 0.000001,
            "grpo_kl_coefficient": 0.04,
            "candidate_directory": "candidates"
        });
        if with_dreaming {
            value["dreaming"] = serde_json::to_value(DreamingConfig::paper_reproduction()).unwrap();
        }
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn dreaming_requires_a_persistent_memory_moe_exploration_route() {
        let mut workflow = wake_only_workflow();
        workflow.phases[0].memory_update_mode = None;
        workflow.phases[0].periodic_sleep = Some(periodic_sleep_config(true));

        let error = validate_workflow_for_model(&workflow, &memory_model(false))
            .unwrap_err()
            .to_string();
        assert!(error.contains("persistent FFN MoE"), "{error}");

        validate_workflow_for_model(&workflow, &dreaming_memory_model()).unwrap();

        workflow.phases[0].periodic_sleep = Some(periodic_sleep_config(false));
        validate_workflow_for_model(&workflow, &memory_model(false)).unwrap();
    }

    #[test]
    fn wake_only_is_an_explicit_uniform_memory_update_policy() {
        let workflow = wake_only_workflow();
        validate_workflow_for_model(&workflow, &memory_model(false)).unwrap();

        let mut missing = workflow.clone();
        let mut second = missing.phases[0].clone();
        second.name = "missing-policy".into();
        second.memory_update_mode = None;
        missing.phases.push(second);
        let error = validate_workflow_for_model(&missing, &memory_model(false))
            .unwrap_err()
            .to_string();
        assert!(error.contains("one identical update policy"), "{error}");

        let mut combined = workflow.clone();
        combined.phases[0].periodic_sleep = Some(
            serde_json::from_value(serde_json::json!({
                "schedule": {
                    "clock": "optimizer_steps",
                    "terminal_consolidation": "distill_into_base_v1",
                    "tiers": [
                        {"id": "fast", "update_period": 1, "reserve_slots": 1},
                        {"id": "slow", "update_period": 2, "reserve_slots": 2}
                    ]
                },
                "knowledge_seeding": {
                    "chunk_tokens": 8,
                    "teacher_rollouts": 1,
                    "detached_student_rollouts": 1,
                    "temperature": 1.0,
                    "forward_kl_weight": 1.0
                },
                "imitation": {
                    "semantic_judge_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                    "semantic_weight": 1.0,
                    "maximum_edit_distance": 8,
                    "grpo_group_size": 2
                },
                "retention_suite": "retention.json",
                "retention_suite_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "retention": {
                    "evaluator_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                    "suite_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                    "max_anchor_forward_kl": 0.1,
                    "max_anchor_regression": 0.1,
                    "min_incorporation_gain": 0.0
                },
                "receiver_learning_rate": 0.0001,
                "receiver_weight_decay": 0.0,
                "grpo_clip_epsilon": 0.2,
                "grpo_advantage_epsilon": 0.000001,
                "grpo_kl_coefficient": 0.04,
                "candidate_directory": "candidates"
            }))
            .unwrap(),
        );
        let error = combined.validate().unwrap_err().to_string();
        assert!(error.contains("cannot combine"), "{error}");
    }

    #[test]
    fn mixed_memory_and_ordinary_layers_are_an_explicitly_unsupported_topology() {
        let error = validate_workflow_for_model(&wake_only_workflow(), &memory_model(true))
            .unwrap_err()
            .to_string();
        assert!(error.contains("model mixes"), "{error}");
        assert!(error.contains("every layer"), "{error}");
    }
}
