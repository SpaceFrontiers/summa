//! Backend-neutral post-training objectives and executor interfaces.
//!
//! This module owns the mathematics and validation for preference,
//! distillation, and verifiable-RL phases. The public scalar/tensor objectives
//! are correctness oracles; native execution uses mandatory device-owned batch
//! runtimes so logits, scores, and gradients never become trainer-side host
//! vectors. No task is converted to another task shape implicitly.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use anyhow::{Context, Result, bail, ensure};
use burn::prelude::Tensor;
use burn::tensor::activation::{log_sigmoid, log_softmax as tensor_log_softmax};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::artifact_io::{sha256_identity, sha256_identity_from_hex, validate_sha256_identity};
use crate::metrics::{
    MetricContext, MetricEvent, MetricPhase, MetricPhaseKind, PostTrainingAlgorithm,
    PostTrainingUpdateMetrics,
};
use crate::native_sleep::{NativeSleepCheckpoint, NativeSleepProgressSink};
use crate::runtime::{
    ImmutableArtifact, ImmutableModelCheckpoint, PhaseExecutionRequest, PhaseExecutionResult,
    PhaseExecutor, PhaseProduct, PhaseProgressSink,
};
use crate::sleep::{SleepPhase, UpdateClock};
use crate::task::{
    RewardSpec, TaskAdapter, TaskConfig, TaskDataFormat, TaskExample, TaskExecution, VerifierSpec,
};
use crate::workflow::{InModelSleepConfig, PhaseKind, PhaseV2};

/// A malformed input without a record delimiter must not make post-training
/// buffer an entire multi-gigabyte dataset. The limit is deliberately much
/// larger than any supported training context while keeping memory bounded.
const MAX_POST_TRAINING_RECORD_BYTES: usize = 64 * 1024 * 1024;
/// One native update must remain small enough to collect and authenticate
/// without an allocator abort. Larger effective batches should be split into
/// additional optimizer steps rather than represented by an enormous host
/// vector.
pub const MAX_POST_TRAINING_UPDATE_EXAMPLES: usize = 65_536;
/// Hard limit for the serialized input records retained by one update. This is
/// independent of the per-record framing bound above.
pub const MAX_POST_TRAINING_UPDATE_INPUT_BYTES: usize = 256 * 1024 * 1024;
/// Upper bound for host-visible GRPO token metadata across one device batch.
/// Numeric token state remains device-owned.
pub const MAX_POST_TRAINING_ROLLOUT_TOKENS: usize = 8_388_608;
pub const MAX_POST_TRAINING_SEQUENCE_TOKENS: usize = 1_048_576;
pub const MAX_POST_TRAINING_ROLLOUTS_PER_PROMPT: usize = 4_096;

fn default_dpo_beta() -> f64 {
    0.1
}

fn default_distillation_temperature() -> f64 {
    1.0
}

fn default_true() -> bool {
    true
}

fn default_grpo_group_size() -> usize {
    8
}

fn default_clip_epsilon() -> f64 {
    0.2
}

fn default_advantage_epsilon() -> f64 {
    1e-6
}

fn default_kl_coefficient() -> f64 {
    0.04
}

fn default_max_new_tokens() -> usize {
    512
}

fn default_rollout_temperature() -> f64 {
    0.7
}

fn default_top_p() -> f64 {
    0.95
}

/// An executor-owned frozen model. `adapter` selects a registered loader;
/// `artifact` and `revision` pin the exact teacher/reference identity.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenModelSpec {
    pub adapter: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<PathBuf>,
    /// Canonical `sha256:<64 lowercase hex>` identity of the exact artifact
    /// bytes. Required for every local artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, serde_json::Value>,
}

impl FrozenModelSpec {
    fn validate(&self, field: &str) -> Result<()> {
        validate_frozen_identifier(&self.adapter, &format!("`{field}.adapter`"))?;
        ensure!(
            self.artifact
                .as_ref()
                .is_none_or(|path| !path.as_os_str().is_empty()),
            "`{field}.artifact` must not be empty when set"
        );
        if let Some(revision) = self.revision.as_deref() {
            validate_frozen_identifier(revision, &format!("`{field}.revision`"))?;
        }
        ensure!(
            self.artifact.is_none() || self.sha256.is_some(),
            "`{field}.artifact` requires an exact `sha256`"
        );
        if let Some(sha256) = &self.sha256 {
            validate_sha256_identity(sha256, &format!("`{field}.sha256`"))?;
        }
        ensure!(
            self.sha256.is_some() || self.revision.is_some(),
            "`{field}` must pin an exact sha256 or immutable revision"
        );
        Ok(())
    }

    /// Stable identity required from the frozen side of a device runtime.
    pub fn immutable_identity(&self) -> Result<String> {
        self.validate("frozen_model")?;
        if let Some(sha256) = &self.sha256 {
            return Ok(sha256.clone());
        }
        let revision = self
            .revision
            .as_deref()
            .context("validated frozen model has no immutable identity")?;
        canonical_sha256(&(
            "hermes-frozen-revision-identity-v1",
            &self.adapter,
            revision,
            &self.parameters,
        ))
    }

    /// Verify local artifact bytes before an executor is allowed to load them.
    pub fn verify_local_artifact(&self) -> Result<()> {
        self.validate("frozen_model")?;
        let Some(path) = &self.artifact else {
            return Ok(());
        };
        let expected = self
            .sha256
            .as_deref()
            .expect("validated local artifact has sha256")
            .strip_prefix("sha256:")
            .expect("validated local artifact digest is canonical");
        let authenticated =
            AuthenticatedPostTrainingInput::open_labeled(path, "frozen model artifact")?;
        let actual = authenticated
            .identity
            .sha256
            .strip_prefix("sha256:")
            .expect("authenticated digest has a sha256 prefix");
        ensure!(
            actual == expected,
            "frozen model artifact {} sha256 mismatch: expected {expected}, got {actual}",
            path.display()
        );
        Ok(())
    }

    fn resolve_paths(&mut self, base: &Path) {
        if let Some(artifact) = &mut self.artifact
            && artifact.is_relative()
        {
            *artifact = base.join(&*artifact);
        }
    }
}

fn validate_frozen_identifier(value: &str, field: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control),
        "{field} must be non-empty, trimmed, and contain no control characters"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceReduction {
    #[default]
    Sum,
    Mean,
}

/// Typed algorithm selection for WorkflowV2 post-training phases.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "algorithm", rename_all = "snake_case", deny_unknown_fields)]
pub enum PostTrainingConfig {
    /// Direct Preference Optimization against an immutable reference policy.
    Dpo {
        reference: FrozenModelSpec,
        #[serde(default = "default_dpo_beta")]
        beta: f64,
        #[serde(default)]
        label_smoothing: f64,
        #[serde(default)]
        sequence_reduction: SequenceReduction,
    },
    /// Forward KL from a frozen teacher distribution to the student.
    ForwardKl {
        teacher: FrozenModelSpec,
        #[serde(default = "default_distillation_temperature")]
        temperature: f64,
        #[serde(default = "default_true")]
        scale_by_temperature_squared: bool,
    },
    /// Group Relative Policy Optimization with verifiable rewards.
    Grpo {
        #[serde(default = "default_grpo_group_size")]
        group_size: usize,
        #[serde(default = "default_clip_epsilon")]
        clip_epsilon: f64,
        #[serde(default = "default_advantage_epsilon")]
        advantage_epsilon: f64,
        #[serde(default = "default_kl_coefficient")]
        kl_coefficient: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reference: Option<FrozenModelSpec>,
        sampling: RolloutSampling,
    },
}

impl PostTrainingConfig {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Dpo {
                reference,
                beta,
                label_smoothing,
                ..
            } => {
                reference.validate("reference")?;
                ensure!(
                    beta.is_finite() && *beta > 0.0,
                    "DPO beta must be finite and positive"
                );
                ensure!(
                    label_smoothing.is_finite()
                        && *label_smoothing >= 0.0
                        && *label_smoothing < 0.5,
                    "DPO label_smoothing must be finite and in [0, 0.5)"
                );
            }
            Self::ForwardKl {
                teacher,
                temperature,
                ..
            } => {
                teacher.validate("teacher")?;
                ensure!(
                    temperature.is_finite() && *temperature > 0.0,
                    "distillation temperature must be finite and positive"
                );
            }
            Self::Grpo {
                group_size,
                clip_epsilon,
                advantage_epsilon,
                kl_coefficient,
                reference,
                sampling,
            } => {
                ensure!(
                    (2..=MAX_POST_TRAINING_ROLLOUTS_PER_PROMPT).contains(group_size),
                    "GRPO group_size must be in 2..={MAX_POST_TRAINING_ROLLOUTS_PER_PROMPT}"
                );
                ensure!(
                    clip_epsilon.is_finite() && *clip_epsilon > 0.0 && *clip_epsilon < 1.0,
                    "GRPO clip_epsilon must be finite and in (0, 1)"
                );
                ensure!(
                    advantage_epsilon.is_finite() && *advantage_epsilon > 0.0,
                    "GRPO advantage_epsilon must be finite and positive"
                );
                ensure!(
                    kl_coefficient.is_finite() && *kl_coefficient >= 0.0,
                    "GRPO kl_coefficient must be finite and non-negative"
                );
                if *kl_coefficient > 0.0 {
                    ensure!(
                        reference.is_some(),
                        "GRPO with positive kl_coefficient requires a frozen reference"
                    );
                }
                if let Some(reference) = reference {
                    reference.validate("reference")?;
                }
                sampling.validate()?;
            }
        }
        Ok(())
    }

    pub(crate) fn resolve_paths(&mut self, base: &Path) {
        match self {
            Self::Dpo { reference, .. } => reference.resolve_paths(base),
            Self::ForwardKl { teacher, .. } => teacher.resolve_paths(base),
            Self::Grpo {
                reference: Some(reference),
                ..
            } => reference.resolve_paths(base),
            Self::Grpo {
                reference: None, ..
            } => {}
        }
    }

    fn verify_local_artifacts(&self) -> Result<()> {
        match self {
            Self::Dpo { reference, .. } => reference.verify_local_artifact(),
            Self::ForwardKl { teacher, .. } => teacher.verify_local_artifact(),
            Self::Grpo {
                reference: Some(reference),
                ..
            } => reference.verify_local_artifact(),
            Self::Grpo {
                reference: None, ..
            } => Ok(()),
        }
    }
}

/// Sampling geometry is part of the persisted RL recipe rather than an
/// implementation-specific collection of unvalidated parameters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutSampling {
    pub max_new_tokens: usize,
    #[serde(default = "default_rollout_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
}

impl Default for RolloutSampling {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_new_tokens(),
            temperature: default_rollout_temperature(),
            top_p: default_top_p(),
        }
    }
}

impl RolloutSampling {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=MAX_POST_TRAINING_SEQUENCE_TOKENS).contains(&self.max_new_tokens),
            "max_new_tokens must be in 1..={MAX_POST_TRAINING_SEQUENCE_TOKENS}"
        );
        ensure!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "rollout temperature must be finite and positive"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "rollout top_p must be finite and in (0, 1]"
        );
        Ok(())
    }
}

/// Scalar correctness-oracle input. Native execution keeps these scores inside
/// [`DpoDeviceBatchRuntime`] and never constructs this value in trainer core.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairwiseLogProbabilities {
    pub policy_chosen: f64,
    pub policy_rejected: f64,
    pub reference_chosen: f64,
    pub reference_rejected: f64,
}

/// DPO scalar-oracle result. Derivatives target the aggregate policy sequence
/// log probabilities; this is not a native runtime transport type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpoLoss {
    pub loss: f64,
    pub policy_margin: f64,
    pub reference_margin: f64,
    pub implicit_reward_margin: f64,
    pub preference_correct: bool,
    pub d_policy_chosen: f64,
    pub d_policy_rejected: f64,
}

/// Numerically stable DPO objective:
/// `-(1-s) log sigmoid(z) - s log sigmoid(-z)`, where
/// `z = beta * ((pi_c-pi_r) - (ref_c-ref_r))`.
pub fn dpo_loss(
    scores: PairwiseLogProbabilities,
    beta: f64,
    label_smoothing: f64,
) -> Result<DpoLoss> {
    ensure_finite(
        &[
            scores.policy_chosen,
            scores.policy_rejected,
            scores.reference_chosen,
            scores.reference_rejected,
        ],
        "DPO log probabilities",
    )?;
    ensure!(
        beta.is_finite() && beta > 0.0,
        "DPO beta must be finite and positive"
    );
    ensure!(
        label_smoothing.is_finite() && (0.0..0.5).contains(&label_smoothing),
        "DPO label_smoothing must be finite and in [0, 0.5)"
    );

    let policy_margin = scores.policy_chosen - scores.policy_rejected;
    let reference_margin = scores.reference_chosen - scores.reference_rejected;
    let implicit_reward_margin = beta * (policy_margin - reference_margin);
    let loss = (1.0 - label_smoothing) * softplus(-implicit_reward_margin)
        + label_smoothing * softplus(implicit_reward_margin);
    let d_z = sigmoid(implicit_reward_margin) - (1.0 - label_smoothing);
    let result = DpoLoss {
        loss,
        policy_margin,
        reference_margin,
        implicit_reward_margin,
        preference_correct: policy_margin > reference_margin,
        d_policy_chosen: beta * d_z,
        d_policy_rejected: -beta * d_z,
    };
    ensure_finite(
        &[
            result.loss,
            result.policy_margin,
            result.reference_margin,
            result.implicit_reward_margin,
            result.d_policy_chosen,
            result.d_policy_rejected,
        ],
        "derived DPO objective values",
    )?;
    Ok(result)
}

/// Autodiff-preserving batched DPO loss over aggregate sequence log-probs.
/// Reference tensors are detached defensively; the returned scalar retains the
/// policy graph.
pub fn dpo_loss_tensor(
    policy_chosen: Tensor<1>,
    policy_rejected: Tensor<1>,
    reference_chosen: Tensor<1>,
    reference_rejected: Tensor<1>,
    beta: f64,
    label_smoothing: f64,
) -> Result<Tensor<1>> {
    let batch = policy_chosen.dims()[0];
    ensure!(batch > 0, "DPO tensor batch is empty");
    ensure!(
        policy_rejected.dims() == [batch]
            && reference_chosen.dims() == [batch]
            && reference_rejected.dims() == [batch],
        "DPO tensor score shapes must all be [{batch}]"
    );
    ensure!(
        policy_chosen.device() == policy_rejected.device()
            && policy_chosen.device() == reference_chosen.device()
            && policy_chosen.device() == reference_rejected.device(),
        "DPO tensors must be on the same device"
    );
    ensure!(
        beta.is_finite() && beta > 0.0,
        "DPO beta must be finite and positive"
    );
    ensure!(
        label_smoothing.is_finite() && (0.0..0.5).contains(&label_smoothing),
        "DPO label_smoothing must be finite and in [0, 0.5)"
    );
    let z = ((policy_chosen - policy_rejected)
        - (reference_chosen.detach() - reference_rejected.detach()))
    .mul_scalar(beta);
    let positive = log_sigmoid(z.clone()).mul_scalar(-(1.0 - label_smoothing));
    let negative = log_sigmoid(z.neg()).mul_scalar(-label_smoothing);
    Ok((positive + negative).mean())
}

/// Reduce per-token continuation log probabilities without silently changing
/// the length normalization selected by the workflow.
pub fn reduce_sequence_log_probs(values: &[f64], reduction: SequenceReduction) -> Result<f64> {
    ensure!(!values.is_empty(), "sequence log probabilities are empty");
    ensure_finite(values, "sequence log probabilities")?;
    let sum: f64 = values.iter().sum();
    Ok(match reduction {
        SequenceReduction::Sum => sum,
        SequenceReduction::Mean => sum / values.len() as f64,
    })
}

/// Host scalar-oracle row. Native forward-KL execution keeps full vocabulary
/// distributions inside [`ForwardKlDeviceBatchRuntime`].
#[derive(Clone, Debug, PartialEq)]
pub struct DistillationToken<'a> {
    pub teacher_logits: &'a [f64],
    pub student_logits: &'a [f64],
    /// Zero masks a token; fractional values support packed-example weights.
    pub weight: f64,
}

/// Scalar-oracle output. Its gradient vectors intentionally remain available
/// for golden tests, but are never returned by the native device-batch API.
#[derive(Clone, Debug, PartialEq)]
pub struct DistillationLoss {
    /// Temperature-scaled training objective.
    pub loss: f64,
    /// Unscaled forward KL, useful for comparable telemetry.
    pub mean_forward_kl: f64,
    pub teacher_entropy: f64,
    pub top1_agreement: f64,
    /// `d(loss)/d(student_logits)` in input row order.
    pub student_gradients: Vec<Vec<f64>>,
}

/// Forward `KL(teacher || student)` over weighted token rows. Log-softmax is
/// computed in f64 with max subtraction. When temperature-squared scaling is
/// enabled, gradients with respect to unscaled student logits are
/// `T * (p_student - p_teacher)`, before row reduction.
pub fn forward_kl_distillation(
    tokens: &[DistillationToken<'_>],
    temperature: f64,
    scale_by_temperature_squared: bool,
) -> Result<DistillationLoss> {
    ensure!(!tokens.is_empty(), "distillation batch is empty");
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "distillation temperature must be finite and positive"
    );

    let mut total_weight = 0.0;
    for (index, token) in tokens.iter().enumerate() {
        ensure!(
            token.teacher_logits.len() == token.student_logits.len(),
            "distillation token {index} teacher/student vocabulary sizes differ"
        );
        ensure!(
            token.teacher_logits.len() >= 2,
            "distillation token {index} vocabulary must contain at least two logits"
        );
        ensure!(
            token.weight.is_finite() && token.weight >= 0.0,
            "distillation token {index} weight must be finite and non-negative"
        );
        ensure_finite(token.teacher_logits, "teacher logits")?;
        ensure_finite(token.student_logits, "student logits")?;
        total_weight += token.weight;
    }
    ensure!(
        total_weight.is_finite() && total_weight > 0.0,
        "distillation batch has no positive token weight"
    );

    let loss_scale = if scale_by_temperature_squared {
        temperature * temperature
    } else {
        1.0
    };
    let gradient_scale = if scale_by_temperature_squared {
        temperature
    } else {
        1.0 / temperature
    };
    ensure!(
        loss_scale.is_finite() && gradient_scale.is_finite(),
        "distillation temperature scaling overflowed"
    );
    let mut weighted_kl = 0.0;
    let mut weighted_entropy = 0.0;
    let mut weighted_agreement = 0.0;
    let mut student_gradients = Vec::with_capacity(tokens.len());

    for token in tokens {
        let teacher_log_probs = log_softmax(token.teacher_logits, temperature);
        let student_log_probs = log_softmax(token.student_logits, temperature);
        ensure_finite(&teacher_log_probs, "teacher log probabilities")?;
        ensure_finite(&student_log_probs, "student log probabilities")?;
        let teacher_probs: Vec<f64> = teacher_log_probs.iter().map(|value| value.exp()).collect();
        let student_probs: Vec<f64> = student_log_probs.iter().map(|value| value.exp()).collect();
        let kl: f64 = teacher_probs
            .iter()
            .zip(&teacher_log_probs)
            .zip(&student_log_probs)
            .map(|((probability, teacher), student)| probability * (teacher - student))
            .sum();
        let entropy: f64 = teacher_probs
            .iter()
            .zip(&teacher_log_probs)
            .map(|(probability, log_probability)| -probability * log_probability)
            .sum();
        weighted_kl += token.weight * kl;
        weighted_entropy += token.weight * entropy;
        weighted_agreement +=
            token.weight * f64::from(argmax(token.teacher_logits) == argmax(token.student_logits));
        let row_scale = token.weight / total_weight * gradient_scale;
        let gradients = student_probs
            .iter()
            .zip(&teacher_probs)
            .map(|(student, teacher)| row_scale * (student - teacher))
            .collect::<Vec<_>>();
        ensure_finite(&gradients, "distillation student gradients")?;
        student_gradients.push(gradients);
    }

    let mean_forward_kl = weighted_kl / total_weight;
    let result = DistillationLoss {
        loss: loss_scale * mean_forward_kl,
        mean_forward_kl,
        teacher_entropy: weighted_entropy / total_weight,
        top1_agreement: weighted_agreement / total_weight,
        student_gradients,
    };
    ensure_finite(
        &[
            result.loss,
            result.mean_forward_kl,
            result.teacher_entropy,
            result.top1_agreement,
        ],
        "derived distillation objective values",
    )?;
    Ok(result)
}

/// Autodiff-preserving forward-KL tensor core for already selected target-token
/// rows. Teacher logits are detached. Masked/padded rows must be gathered out
/// before calling this function, so an invalid all-masked batch cannot be
/// silently normalized.
pub fn forward_kl_distillation_tensor(
    teacher_logits: Tensor<2>,
    student_logits: Tensor<2>,
    temperature: f64,
    scale_by_temperature_squared: bool,
) -> Result<Tensor<1>> {
    let [tokens, vocabulary] = teacher_logits.dims();
    ensure!(tokens > 0, "distillation tensor batch is empty");
    ensure!(
        vocabulary >= 2,
        "distillation tensor vocabulary must contain at least two logits"
    );
    ensure!(
        student_logits.dims() == [tokens, vocabulary],
        "distillation teacher/student tensor shapes differ"
    );
    ensure!(
        teacher_logits.device() == student_logits.device(),
        "distillation tensors must be on the same device"
    );
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "distillation temperature must be finite and positive"
    );
    let teacher_log_probs = tensor_log_softmax(teacher_logits.detach().div_scalar(temperature), 1);
    let student_log_probs = tensor_log_softmax(student_logits.div_scalar(temperature), 1);
    let forward_kl = (teacher_log_probs.clone().exp() * (teacher_log_probs - student_log_probs))
        .sum_dim(1)
        .mean();
    Ok(if scale_by_temperature_squared {
        forward_kl.mul_scalar(temperature * temperature)
    } else {
        forward_kl
    })
}

/// Scalar correctness-oracle rollout. Native GRPO scores and token ids stay
/// opaque inside [`GrpoDeviceBatchRuntime`].
#[derive(Clone, Debug, PartialEq)]
pub struct GrpoRollout<'a> {
    pub reward: f64,
    pub current_token_log_probs: &'a [f64],
    pub behavior_token_log_probs: &'a [f64],
    pub reference_token_log_probs: Option<&'a [f64]>,
}

/// Scalar-oracle output; native execution returns only reduced batch metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct GrpoLoss {
    pub loss: f64,
    pub mean_reward: f64,
    pub reward_stddev: f64,
    pub mean_kl: f64,
    pub clipped_fraction: f64,
    pub advantages: Vec<f64>,
    /// Per-rollout, per-token derivatives with respect to current log-probs.
    pub current_log_prob_gradients: Vec<Vec<f64>>,
}

/// Clipped GRPO objective with group-normalized rewards and the non-negative
/// unbiased KL estimator `exp(log p_ref - log p) - (log p_ref - log p) - 1`.
/// Each sequence contributes the mean of its token objectives and each member
/// contributes equally to the prompt group.
pub fn grpo_loss(
    rollouts: &[GrpoRollout<'_>],
    clip_epsilon: f64,
    advantage_epsilon: f64,
    kl_coefficient: f64,
) -> Result<GrpoLoss> {
    ensure!(
        rollouts.len() >= 2,
        "GRPO requires at least two rollouts per prompt"
    );
    ensure!(
        clip_epsilon.is_finite() && clip_epsilon > 0.0 && clip_epsilon < 1.0,
        "GRPO clip_epsilon must be finite and in (0, 1)"
    );
    ensure!(
        advantage_epsilon.is_finite() && advantage_epsilon > 0.0,
        "GRPO advantage_epsilon must be finite and positive"
    );
    ensure!(
        kl_coefficient.is_finite() && kl_coefficient >= 0.0,
        "GRPO kl_coefficient must be finite and non-negative"
    );

    let rewards: Vec<f64> = rollouts.iter().map(|rollout| rollout.reward).collect();
    ensure_finite(&rewards, "GRPO rewards")?;
    let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
    ensure!(mean_reward.is_finite(), "GRPO reward mean overflowed");
    let variance = rewards
        .iter()
        .map(|reward| (reward - mean_reward).powi(2))
        .sum::<f64>()
        / rewards.len() as f64;
    ensure!(variance.is_finite(), "GRPO reward normalization overflowed");
    let reward_stddev = variance.sqrt();
    let denominator = reward_stddev.max(advantage_epsilon);
    let advantages: Vec<f64> = rewards
        .iter()
        .map(|reward| (reward - mean_reward) / denominator)
        .collect();
    ensure_finite(&advantages, "GRPO normalized advantages")?;

    let group_scale = 1.0 / rollouts.len() as f64;
    let lower = 1.0 - clip_epsilon;
    let upper = 1.0 + clip_epsilon;
    let mut objective = 0.0;
    let mut total_kl = 0.0;
    let mut clipped = 0usize;
    let mut token_count = 0usize;
    let mut gradients = Vec::with_capacity(rollouts.len());

    for (rollout_index, (rollout, advantage)) in rollouts.iter().zip(&advantages).enumerate() {
        let token_len = rollout.current_token_log_probs.len();
        ensure!(
            token_len > 0,
            "GRPO rollout {rollout_index} has no generated tokens"
        );
        ensure!(
            rollout.behavior_token_log_probs.len() == token_len,
            "GRPO rollout {rollout_index} current/behavior token counts differ"
        );
        if kl_coefficient > 0.0 {
            ensure!(
                rollout.reference_token_log_probs.is_some(),
                "GRPO rollout {rollout_index} needs reference log-probs for positive KL coefficient"
            );
        }
        if let Some(reference) = rollout.reference_token_log_probs {
            ensure!(
                reference.len() == token_len,
                "GRPO rollout {rollout_index} current/reference token counts differ"
            );
            ensure_finite(reference, "GRPO reference log probabilities")?;
        }
        ensure_finite(
            rollout.current_token_log_probs,
            "GRPO current log probabilities",
        )?;
        ensure_finite(
            rollout.behavior_token_log_probs,
            "GRPO behavior log probabilities",
        )?;

        let token_scale = group_scale / token_len as f64;
        let mut row_gradients = Vec::with_capacity(token_len);
        for token_index in 0..token_len {
            let current = rollout.current_token_log_probs[token_index];
            let behavior = rollout.behavior_token_log_probs[token_index];
            let log_ratio = current - behavior;
            ensure!(
                log_ratio.is_finite() && log_ratio <= 80.0,
                "GRPO rollout {rollout_index} token {token_index} has an unsafe importance ratio"
            );
            let ratio = log_ratio.exp();
            let clipped_ratio = ratio.clamp(lower, upper);
            let unclipped_surrogate = ratio * advantage;
            let clipped_surrogate = clipped_ratio * advantage;
            let surrogate = unclipped_surrogate.min(clipped_surrogate);
            let is_clipped = surrogate == clipped_surrogate && ratio != clipped_ratio;
            clipped += usize::from(is_clipped);
            token_count += 1;

            let (kl, d_kl) = match rollout.reference_token_log_probs {
                Some(reference) => {
                    let log_ref_minus_policy = reference[token_index] - current;
                    ensure!(
                        log_ref_minus_policy.is_finite() && log_ref_minus_policy <= 80.0,
                        "GRPO rollout {rollout_index} token {token_index} has an unsafe KL ratio"
                    );
                    let ref_over_policy = log_ref_minus_policy.exp();
                    (
                        ref_over_policy - log_ref_minus_policy - 1.0,
                        1.0 - ref_over_policy,
                    )
                }
                None => (0.0, 0.0),
            };
            objective += token_scale * (surrogate - kl_coefficient * kl);
            total_kl += token_scale * kl;

            let surrogate_active =
                (*advantage >= 0.0 && ratio <= upper) || (*advantage < 0.0 && ratio >= lower);
            let d_surrogate = if surrogate_active {
                ratio * advantage
            } else {
                0.0
            };
            row_gradients.push(token_scale * (-d_surrogate + kl_coefficient * d_kl));
        }
        gradients.push(row_gradients);
    }

    ensure!(
        objective.is_finite() && total_kl.is_finite(),
        "GRPO objective overflowed"
    );
    for row in &gradients {
        ensure_finite(row, "GRPO current log-probability gradients")?;
    }

    Ok(GrpoLoss {
        loss: -objective,
        mean_reward,
        reward_stddev,
        mean_kl: total_kl,
        clipped_fraction: clipped as f64 / token_count as f64,
        advantages,
        current_log_prob_gradients: gradients,
    })
}

/// Autodiff-preserving GRPO tensor core for a fixed prompt group. Generated
/// token rows use shape `[group, max_tokens]`; `active_mask` is 1 for real
/// tokens and 0 for padding, with at least one real token in every row.
#[allow(clippy::too_many_arguments)]
pub fn grpo_loss_tensor(
    rewards: Tensor<1>,
    current_token_log_probs: Tensor<2>,
    behavior_token_log_probs: Tensor<2>,
    reference_token_log_probs: Option<Tensor<2>>,
    active_mask: Tensor<2>,
    clip_epsilon: f64,
    advantage_epsilon: f64,
    kl_coefficient: f64,
) -> Result<Tensor<1>> {
    let [group, tokens] = current_token_log_probs.dims();
    ensure!(
        group >= 2,
        "GRPO tensor group must contain at least two rollouts"
    );
    ensure!(tokens > 0, "GRPO tensor rollouts have no token columns");
    ensure!(
        rewards.dims() == [group],
        "GRPO rewards must have shape [{group}]"
    );
    for (name, dims) in [
        ("behavior", behavior_token_log_probs.dims()),
        ("active_mask", active_mask.dims()),
    ] {
        ensure!(
            dims == [group, tokens],
            "GRPO {name} tensor must have shape [{group}, {tokens}]"
        );
    }
    let device = current_token_log_probs.device();
    ensure!(
        rewards.device() == device
            && behavior_token_log_probs.device() == device
            && active_mask.device() == device,
        "GRPO tensors must be on the same device"
    );
    if let Some(reference) = &reference_token_log_probs {
        ensure!(
            reference.dims() == [group, tokens],
            "GRPO reference tensor must have shape [{group}, {tokens}]"
        );
        ensure!(
            reference.device() == device,
            "GRPO tensors must be on the same device"
        );
    } else {
        ensure!(
            kl_coefficient == 0.0,
            "GRPO tensor objective needs reference log-probs for positive KL coefficient"
        );
    }
    ensure!(
        clip_epsilon.is_finite() && clip_epsilon > 0.0 && clip_epsilon < 1.0,
        "GRPO clip_epsilon must be finite and in (0, 1)"
    );
    ensure!(
        advantage_epsilon.is_finite() && advantage_epsilon > 0.0,
        "GRPO advantage_epsilon must be finite and positive"
    );
    ensure!(
        kl_coefficient.is_finite() && kl_coefficient >= 0.0,
        "GRPO kl_coefficient must be finite and non-negative"
    );

    let rewards = rewards.detach();
    let behavior_token_log_probs = behavior_token_log_probs.detach();
    let reference_token_log_probs = reference_token_log_probs.map(Tensor::detach);
    let log_importance_ratio =
        current_token_log_probs.clone().detach() - behavior_token_log_probs.clone();
    let reward_mean = rewards.clone().mean();
    let centered_rewards = rewards.clone() - reward_mean.clone();
    let reward_variance = centered_rewards.clone().powi_scalar(2).mean();
    let reward_stddev = reward_variance.clone().sqrt().clamp_min(advantage_epsilon);

    // Build the differentiable objective before materializing validation
    // flags. This lets the same single device read that validates the inputs
    // also reject overflow introduced by exp/KL scaling before callers can
    // backpropagate a non-finite loss.
    let advantages =
        ((rewards.clone() - reward_mean.clone()) / reward_stddev.clone()).reshape([group, 1]);
    let ratio = (current_token_log_probs.clone() - behavior_token_log_probs.clone()).exp();
    let clipped_ratio = ratio.clone().clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon);
    let surrogate = (ratio * advantages.clone()).min_pair(clipped_ratio * advantages);
    let mask = active_mask.clone().detach();
    let active_per_rollout = mask.clone().sum_dim(1);
    let token_objective = match &reference_token_log_probs {
        Some(reference) => {
            let log_ref_minus_policy = reference.clone() - current_token_log_probs.clone();
            let kl = log_ref_minus_policy.clone().exp() - log_ref_minus_policy - 1;
            surrogate - kl.mul_scalar(kl_coefficient)
        }
        None => surrogate,
    };
    let sequence_objective = (token_objective * mask).sum_dim(1) / active_per_rollout;
    let loss = sequence_objective.mean().neg();

    // Collapse every device-side precondition into one tiny transfer instead
    // of synchronizing the accelerator once per check. The ordered labels
    // preserve precise diagnostics while keeping validation overhead bounded.
    let mut validation_names = vec![
        "GRPO rewards contain a non-finite value",
        "GRPO current log probabilities contain a non-finite value",
        "GRPO behavior log probabilities contain a non-finite value",
        "GRPO active_mask must contain only binary zero/one values",
        "GRPO active_mask must contain at least one active token in every rollout",
        "GRPO importance log-ratio is non-finite",
        "GRPO importance log-ratio exceeds the safe exponential domain",
        "GRPO reward normalization overflowed",
        "GRPO tensor objective overflowed",
    ];
    let mut validation_tensors = vec![
        rewards.clone().is_finite().all(),
        current_token_log_probs.clone().is_finite().all(),
        behavior_token_log_probs.clone().is_finite().all(),
        active_mask
            .clone()
            .equal_elem(0.0)
            .bool_or(active_mask.clone().equal_elem(1.0))
            .all(),
        active_mask.clone().sum_dim(1).greater_elem(0.0).all(),
        log_importance_ratio.clone().is_finite().all(),
        log_importance_ratio.clone().lower_equal_elem(80.0).all(),
        centered_rewards
            .clone()
            .is_finite()
            .all()
            .bool_and(reward_variance.clone().is_finite().all()),
        loss.clone().detach().is_finite().all(),
    ];
    if let Some(reference) = &reference_token_log_probs {
        let log_reference_ratio = reference.clone() - current_token_log_probs.clone().detach();
        validation_names.extend([
            "GRPO reference log probabilities contain a non-finite value",
            "GRPO reference log-ratio is non-finite",
            "GRPO reference log-ratio exceeds the safe exponential domain",
        ]);
        validation_tensors.extend([
            reference.clone().is_finite().all(),
            log_reference_ratio.clone().is_finite().all(),
            log_reference_ratio.lower_equal_elem(80.0).all(),
        ]);
    }
    let validation_values = Tensor::cat(validation_tensors, 0)
        .into_data()
        .to_vec::<bool>()
        .context("reading GRPO tensor validation flags")?;
    ensure!(
        validation_values.len() == validation_names.len(),
        "GRPO backend returned an incomplete validation result"
    );
    for (valid, message) in validation_values.into_iter().zip(validation_names) {
        ensure!(valid, message);
    }

    Ok(loss)
}

/// One deterministic model-execution substream. A device runtime must use the
/// same substream for a stochastic forward and its matching backward replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ModelExecutionRng {
    pub seed: u64,
    pub counter: u64,
}

/// Exact half-open model RNG range reserved by the durable trainer cursor.
/// Batch runtimes derive example-major substreams from this range without
/// asking trainer core to make one model call per example.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ModelExecutionRngRange {
    seed: u64,
    start: u64,
    end: u64,
}

impl ModelExecutionRngRange {
    fn reserve(seed: u64, start: u64, count: u64) -> Result<Self> {
        ensure!(count > 0, "model RNG range must not be empty");
        Ok(Self {
            seed,
            start,
            end: start
                .checked_add(count)
                .context("post-training model RNG counter overflows u64")?,
        })
    }

    pub fn len(self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub fn seed(self) -> u64 {
        self.seed
    }

    pub fn start(self) -> u64 {
        self.start
    }

    pub fn end(self) -> u64 {
        self.end
    }

    pub fn substream(self, offset: u64) -> Result<ModelExecutionRng> {
        ensure!(
            offset < self.len(),
            "model RNG substream offset is out of range"
        );
        Ok(ModelExecutionRng {
            seed: self.seed,
            counter: self
                .start
                .checked_add(offset)
                .context("post-training model RNG counter overflows u64")?,
        })
    }
}

/// Compact backend-owned evidence for a prepared device update. Implementors
/// hash their exact batched execution without materializing logits or
/// gradients in trainer memory. The digest must bind the input batch, base
/// model/optimizer generation, RNG range, objective configuration, reduced
/// metrics, and staged backward result; replay after restore must reproduce it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DeviceExecutionReceipt {
    pub execution_sha256: String,
    pub model_tokens: u64,
}

impl DeviceExecutionReceipt {
    fn validate(&self) -> Result<()> {
        validate_sha256_identity(&self.execution_sha256, "device execution receipt")?;
        ensure!(
            self.model_tokens > 0,
            "device execution reported zero model tokens"
        );
        Ok(())
    }
}

/// Immutable model/optimizer generation restored before a device batch is
/// prepared. Passing it explicitly keeps runtime receipts self-contained and
/// prevents a backend from accidentally authenticating only the examples and
/// objective configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DeviceBatchBaseGeneration<'a> {
    pub model_sha256: &'a str,
    pub optimizer_sha256: Option<&'a str>,
}

impl DeviceBatchBaseGeneration<'_> {
    fn validate(self) -> Result<()> {
        validate_sha256_identity(self.model_sha256, "device-batch base model")?;
        if let Some(optimizer_sha256) = self.optimizer_sha256 {
            validate_sha256_identity(optimizer_sha256, "device-batch base optimizer")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DpoDeviceBatchRequest<'a> {
    pub examples: &'a [TaskExample],
    pub batch_sha256: &'a str,
    pub base: DeviceBatchBaseGeneration<'a>,
    pub max_sequence_tokens: usize,
    pub beta: f64,
    pub label_smoothing: f64,
    pub sequence_reduction: SequenceReduction,
    /// Multiply the mean batch gradient by this value. Returned metrics remain
    /// the unweighted per-example objective sums.
    pub loss_weight: f64,
    /// Exactly two example-major substreams per pair: chosen, then rejected.
    /// Each stochastic forward and its internal backward use the same value.
    pub rng: ModelExecutionRngRange,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DpoDeviceBatchResult {
    pub receipt: DeviceExecutionReceipt,
    pub examples: u64,
    pub loss_sum: f64,
    pub preference_correct: u64,
    pub implicit_reward_margin_sum: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ForwardKlDeviceBatchRequest<'a> {
    pub examples: &'a [TaskExample],
    pub batch_sha256: &'a str,
    pub base: DeviceBatchBaseGeneration<'a>,
    pub max_sequence_tokens: usize,
    pub temperature: f64,
    pub scale_by_temperature_squared: bool,
    pub loss_weight: f64,
    /// Exactly one example-major student substream. Teacher execution is
    /// frozen; the student forward and internal backward reuse this value.
    pub rng: ModelExecutionRngRange,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ForwardKlDeviceBatchResult {
    pub receipt: DeviceExecutionReceipt,
    pub examples: u64,
    pub loss_sum: f64,
    pub forward_kl_sum: f64,
    pub teacher_entropy_sum: f64,
    pub top1_agreement_sum: f64,
}

/// Host-visible portion of a rollout. Token ids and all behavior/current/
/// reference scores remain owned by [`GrpoDeviceBatchRuntime`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GeneratedRolloutText {
    pub text: String,
    pub token_count: usize,
}

/// Opaque device generation plus the text needed by the configured verifier.
/// `generation_sha256` must bind the complete generation request together with
/// runtime-owned token ids and behavior-score state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GrpoGeneratedBatch {
    pub generation_sha256: String,
    pub groups: Vec<Vec<GeneratedRolloutText>>,
}

#[derive(Clone, Copy, Debug)]
pub struct GrpoGenerationBatchRequest<'a> {
    pub examples: &'a [TaskExample],
    pub batch_sha256: &'a str,
    pub base: DeviceBatchBaseGeneration<'a>,
    pub group_size: usize,
    pub max_sequence_tokens: usize,
    pub sampling: &'a RolloutSampling,
    /// One example-major substream per rollout in each prompt group.
    pub rng: ModelExecutionRngRange,
}

#[derive(Clone, Debug)]
pub struct GrpoDeviceBatchRequest<'a> {
    pub examples: &'a [TaskExample],
    pub batch_sha256: &'a str,
    pub base: DeviceBatchBaseGeneration<'a>,
    pub generation: &'a GrpoGeneratedBatch,
    pub rewards: &'a [Vec<f64>],
    pub clip_epsilon: f64,
    pub advantage_epsilon: f64,
    pub kl_coefficient: f64,
    pub loss_weight: f64,
    /// One example-major rescoring substream per generated rollout; internal
    /// policy forward/backward must reuse the same substream.
    pub rng: ModelExecutionRngRange,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GrpoDeviceBatchResult {
    pub receipt: DeviceExecutionReceipt,
    pub examples: u64,
    pub rollouts: u64,
    pub loss_sum: f64,
    pub mean_reward_sum: f64,
    pub reward_stddev_sum: f64,
    pub mean_kl_sum: f64,
    pub clipped_fraction_sum: f64,
}

/// Device-owned DPO execution. A single call must batch all examples, run the
/// trainable and frozen policies, apply backward internally, and return only a
/// compact result. There is deliberately no per-example or host-logit fallback.
/// `prepare_device_batch` stages gradients but must not apply the optimizer;
/// publication invokes `optimizer_step` exactly once through an idempotent
/// publisher after the prepared transaction is durable.
pub trait DpoDeviceBatchRuntime {
    fn trainable_identity(&self) -> &str;
    fn trainable_tokenizer_identity(&self) -> &str;
    fn frozen_identity(&self) -> &str;
    fn frozen_tokenizer_identity(&self) -> &str;
    /// Monotonic host-maintained counter. Reading it must not synchronize a
    /// device or copy a tensor to the host.
    fn model_tokens_processed(&self) -> u64;
    fn prepare_device_batch(
        &mut self,
        request: DpoDeviceBatchRequest<'_>,
    ) -> Result<DpoDeviceBatchResult>;
    fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()>;
}

/// Device-owned forward-KL execution. Teacher and student distributions and
/// student gradients never cross this boundary as host vectors. Preparation
/// stages the complete batch backward but does not apply the optimizer.
pub trait ForwardKlDeviceBatchRuntime {
    fn trainable_identity(&self) -> &str;
    fn trainable_tokenizer_identity(&self) -> &str;
    fn frozen_identity(&self) -> &str;
    fn frozen_tokenizer_identity(&self) -> &str;
    /// Monotonic host-maintained counter. Reading it must not synchronize a
    /// device or copy a tensor to the host.
    fn model_tokens_processed(&self) -> u64;
    fn prepare_device_batch(
        &mut self,
        request: ForwardKlDeviceBatchRequest<'_>,
    ) -> Result<ForwardKlDeviceBatchResult>;
    fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()>;
}

/// Device-owned GRPO numeric execution. Generation exposes only verifier text
/// and token counts; behavior/current/reference scores and backward remain in
/// the runtime. `prepare_device_batch` consumes the whole verified batch once.
/// It stages gradients but leaves optimizer application to the publisher.
pub trait GrpoDeviceBatchRuntime {
    fn trainable_identity(&self) -> &str;
    fn trainable_tokenizer_identity(&self) -> &str;
    fn frozen_identity(&self) -> Option<&str>;
    fn frozen_tokenizer_identity(&self) -> Option<&str>;
    /// Monotonic host-maintained counter. Reading it must not synchronize a
    /// device or copy a tensor to the host.
    fn model_tokens_processed(&self) -> u64;
    fn generate_device_batch(
        &mut self,
        request: GrpoGenerationBatchRequest<'_>,
    ) -> Result<GrpoGeneratedBatch>;
    fn prepare_device_batch(
        &mut self,
        request: GrpoDeviceBatchRequest<'_>,
    ) -> Result<GrpoDeviceBatchResult>;
    fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()>;
}

/// Exactly one device-owned runtime must match the typed phase algorithm.
pub enum PostTrainingPhaseRuntime<'a> {
    Dpo {
        runtime: &'a mut dyn DpoDeviceBatchRuntime,
    },
    ForwardKl {
        runtime: &'a mut dyn ForwardKlDeviceBatchRuntime,
    },
    Grpo {
        runtime: &'a mut dyn GrpoDeviceBatchRuntime,
        verifier: &'a dyn RewardVerifier,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PostTrainingPhaseReport {
    pub phase: String,
    pub algorithm: String,
    pub examples: usize,
    pub optimizer_steps: usize,
    pub trainable_model_identity: String,
    pub frozen_model_identity: Option<String>,
    pub mean_loss: f64,
    pub metrics: BTreeMap<String, f64>,
}

fn validate_frozen_runtime(
    spec: &FrozenModelSpec,
    runtime_identity: &str,
    runtime_tokenizer: &str,
    policy_tokenizer: &str,
) -> Result<()> {
    let expected = spec.immutable_identity()?;
    ensure!(
        runtime_identity == expected,
        "frozen runtime identity mismatch: expected `{expected}`, got `{runtime_identity}`"
    );
    ensure!(
        !runtime_tokenizer.trim().is_empty() && runtime_tokenizer == policy_tokenizer,
        "frozen and trainable runtime tokenizer identities differ"
    );
    Ok(())
}

fn post_training_algorithm_name(config: &PostTrainingConfig) -> &'static str {
    match config {
        PostTrainingConfig::Dpo { .. } => "dpo",
        PostTrainingConfig::ForwardKl { .. } => "forward_kl",
        PostTrainingConfig::Grpo { .. } => "grpo",
    }
}

/// Serialized native post-training cursor contract.
pub const POST_TRAINING_CURSOR_VERSION: u32 = 3;
pub const POST_TRAINING_RESUME_VERSION: u32 = 1;

/// Authenticated cumulative wake clock bound to one immutable model
/// generation. The authority is the registered periodic-sleep controller;
/// its identity is included in the native-host dispatch digest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingClockReceipt {
    pub authority_identity: String,
    pub checkpoint_sha256: String,
    pub optimizer_steps: u64,
    pub model_tokens: u64,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct PostTrainingClockBody<'a> {
    authority_identity: &'a str,
    checkpoint_sha256: &'a str,
    optimizer_steps: u64,
    model_tokens: u64,
}

impl PostTrainingClockReceipt {
    pub fn new(
        authority_identity: impl Into<String>,
        checkpoint_sha256: impl Into<String>,
        optimizer_steps: u64,
        model_tokens: u64,
    ) -> Result<Self> {
        let mut value = Self {
            authority_identity: authority_identity.into(),
            checkpoint_sha256: checkpoint_sha256.into(),
            optimizer_steps,
            model_tokens,
            receipt_sha256: String::new(),
        };
        value.receipt_sha256 = canonical_sha256(&value.body())?;
        value.validate()?;
        Ok(value)
    }

    fn body(&self) -> PostTrainingClockBody<'_> {
        PostTrainingClockBody {
            authority_identity: &self.authority_identity,
            checkpoint_sha256: &self.checkpoint_sha256,
            optimizer_steps: self.optimizer_steps,
            model_tokens: self.model_tokens,
        }
    }

    fn validate(&self) -> Result<()> {
        validate_sha256_identity(&self.authority_identity, "post-training clock authority")?;
        validate_sha256_identity(&self.checkpoint_sha256, "post-training clock checkpoint")?;
        validate_sha256_identity(&self.receipt_sha256, "post-training clock receipt")?;
        ensure!(
            self.receipt_sha256 == canonical_sha256(&self.body())?,
            "post-training clock receipt does not match its content"
        );
        Ok(())
    }

    pub fn selected(&self, clock: UpdateClock) -> u64 {
        match clock {
            UpdateClock::OptimizerSteps => self.optimizer_steps,
            UpdateClock::ModelTokens => self.model_tokens,
        }
    }
}

fn validate_periodic_sleep_clock_window(
    config: &InModelSleepConfig,
    before: &PostTrainingClockReceipt,
    after: PostTrainingClockValues,
) -> Result<()> {
    config.schedule.validate()?;
    let start = before.selected(config.schedule.clock);
    let end = after.selected(config.schedule.clock);
    ensure!(end >= start, "periodic sleep clock cannot move backwards");
    for tier in &config.schedule.tiers {
        let crossed = end / tier.update_period - start / tier.update_period;
        ensure!(
            crossed <= 1,
            "periodic sleep clock advance from {start} to {end} crosses {crossed} `{}` boundaries; one tier gradient accumulator cannot supply multiple updates, so reduce the per-update token budget or increase its update_period",
            tier.id,
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingRecordPosition {
    pub epoch: u64,
    pub record: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingRngCursor {
    pub seed: u64,
    pub counter: u64,
}

/// Immutable identity of the exact bytes streamed by a post-training phase.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedPostTrainingInput {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

impl PinnedPostTrainingInput {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.path.trim().is_empty(),
            "post-training input path is empty"
        );
        validate_sha256_identity(&self.sha256, "post-training input")
    }
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StablePostTrainingFileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    length: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(unix)]
impl StablePostTrainingFileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            length: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

/// One authenticated file description used for both the durable input digest
/// and every task record consumed during this execution attempt. Keeping the
/// opened file prevents a pathname replacement from substituting bytes after
/// the cursor has been content-addressed.
struct AuthenticatedPostTrainingInput {
    description: &'static str,
    path: PathBuf,
    identity: PinnedPostTrainingInput,
    file: File,
    #[cfg(unix)]
    stable_identity: StablePostTrainingFileIdentity,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
}

impl AuthenticatedPostTrainingInput {
    fn open(path: &Path) -> Result<Self> {
        Self::open_labeled(path, "post-training input")
    }

    fn open_labeled(path: &Path, description: &'static str) -> Result<Self> {
        let path_metadata = fs::symlink_metadata(path)
            .with_context(|| format!("failed to inspect {description} {}", path.display()))?;
        ensure!(
            path_metadata.file_type().is_file() && !path_metadata.file_type().is_symlink(),
            "{description} {} must be a non-symlink regular file",
            path.display()
        );
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let mut file = options
            .open(path)
            .with_context(|| format!("failed to open {description} {}", path.display()))?;
        let opened_metadata = file.metadata().with_context(|| {
            format!("failed to inspect opened {description} {}", path.display())
        })?;
        ensure!(
            opened_metadata.is_file(),
            "{description} {} must remain a regular file while opening",
            path.display()
        );
        #[cfg(unix)]
        let stable_identity = StablePostTrainingFileIdentity::from_metadata(&path_metadata);
        #[cfg(unix)]
        ensure!(
            StablePostTrainingFileIdentity::from_metadata(&opened_metadata) == stable_identity,
            "{description} {} changed while it was opened",
            path.display()
        );
        #[cfg(not(unix))]
        let modified = path_metadata.modified().ok();
        #[cfg(not(unix))]
        ensure!(
            opened_metadata.len() == path_metadata.len()
                && opened_metadata.modified().ok() == modified,
            "{description} {} changed while it was opened",
            path.display()
        );

        let expected_bytes = opened_metadata.len();
        let mut bytes = 0_u64;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .with_context(|| format!("failed to hash {description} {}", path.display()))?;
            if read == 0 {
                break;
            }
            bytes = bytes
                .checked_add(read as u64)
                .context("post-training input byte count overflows u64")?;
            ensure!(
                bytes <= expected_bytes,
                "{description} {} grew while it was hashed",
                path.display()
            );
            hasher.update(&buffer[..read]);
        }
        ensure!(
            bytes == expected_bytes,
            "{description} {} changed while it was hashed",
            path.display()
        );
        file.seek(SeekFrom::Start(0))
            .with_context(|| format!("failed to rewind {description} {}", path.display()))?;
        let path_text = path
            .to_str()
            .with_context(|| format!("{description} path is not valid UTF-8"))?
            .to_owned();
        let source = Self {
            description,
            path: path.to_owned(),
            identity: PinnedPostTrainingInput {
                path: path_text,
                sha256: format!("sha256:{:x}", hasher.finalize()),
                bytes,
            },
            file,
            #[cfg(unix)]
            stable_identity,
            #[cfg(not(unix))]
            modified,
        };
        source.ensure_still_published()?;
        Ok(source)
    }

    fn ensure_still_published(&self) -> Result<()> {
        let opened = self.file.metadata().with_context(|| {
            format!(
                "failed to reinspect opened {} {}",
                self.description,
                self.path.display()
            )
        })?;
        let published = fs::symlink_metadata(&self.path).with_context(|| {
            format!(
                "failed to reinspect {} {}",
                self.description,
                self.path.display()
            )
        })?;
        #[cfg(unix)]
        ensure!(
            StablePostTrainingFileIdentity::from_metadata(&opened) == self.stable_identity
                && StablePostTrainingFileIdentity::from_metadata(&published)
                    == self.stable_identity,
            "{} {} changed after it was authenticated",
            self.description,
            self.path.display()
        );
        #[cfg(not(unix))]
        ensure!(
            opened.is_file()
                && published.file_type().is_file()
                && !published.file_type().is_symlink()
                && opened.len() == self.identity.bytes
                && published.len() == self.identity.bytes
                && opened.modified().ok() == self.modified
                && published.modified().ok() == self.modified,
            "{} {} changed after it was authenticated",
            self.description,
            self.path.display()
        );
        Ok(())
    }

    fn reader(&mut self) -> Result<Box<dyn BufRead>> {
        self.ensure_still_published()?;
        self.file.seek(SeekFrom::Start(0)).with_context(|| {
            format!(
                "failed to rewind post-training input {}",
                self.path.display()
            )
        })?;
        // `try_clone` retains the authenticated inode. The retained base
        // handle is never read concurrently and rewinds the shared offset only
        // after the previous reader has been dropped at an epoch boundary.
        let file = self.file.try_clone().with_context(|| {
            format!(
                "failed to clone post-training input {}",
                self.path.display()
            )
        })?;
        if self.path.extension().is_some_and(|ext| ext == "zst") {
            Ok(Box::new(BufReader::new(
                zstd::stream::read::Decoder::new(file).with_context(|| {
                    format!("failed to open zstd task data {}", self.path.display())
                })?,
            )))
        } else {
            Ok(Box::new(BufReader::new(file)))
        }
    }
}

/// All model runtimes are pinned into the cursor. Frozen model identities
/// additionally remain bound to their WorkflowV2 `FrozenModelSpec`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingRuntimeIdentities {
    pub trainable: String,
    pub tokenizer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verifier: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "algorithm", rename_all = "snake_case", deny_unknown_fields)]
pub enum PostTrainingObjectiveSummary {
    Dpo {
        preference_correct: u64,
        implicit_reward_margin_sum: f64,
    },
    ForwardKl {
        forward_kl_sum: f64,
        teacher_entropy_sum: f64,
        top1_agreement_sum: f64,
    },
    Grpo {
        mean_reward_sum: f64,
        reward_stddev_sum: f64,
        mean_kl_sum: f64,
        clipped_fraction_sum: f64,
    },
}

impl PostTrainingObjectiveSummary {
    fn algorithm(&self) -> PostTrainingAlgorithm {
        match self {
            Self::Dpo { .. } => PostTrainingAlgorithm::Dpo,
            Self::ForwardKl { .. } => PostTrainingAlgorithm::ForwardKl,
            Self::Grpo { .. } => PostTrainingAlgorithm::Grpo,
        }
    }

    fn validate(&self, examples: u64) -> Result<()> {
        ensure!(examples > 0, "post-training objective summary is empty");
        match self {
            Self::Dpo {
                preference_correct,
                implicit_reward_margin_sum,
            } => {
                ensure!(
                    *preference_correct <= examples,
                    "DPO correct count exceeds example count"
                );
                ensure!(
                    implicit_reward_margin_sum.is_finite(),
                    "DPO reward-margin sum is not finite"
                );
            }
            Self::ForwardKl {
                forward_kl_sum,
                teacher_entropy_sum,
                top1_agreement_sum,
            } => {
                ensure!(
                    forward_kl_sum.is_finite() && *forward_kl_sum >= 0.0,
                    "forward-KL sum must be finite and non-negative"
                );
                ensure!(
                    teacher_entropy_sum.is_finite() && *teacher_entropy_sum >= 0.0,
                    "teacher-entropy sum must be finite and non-negative"
                );
                ensure!(
                    top1_agreement_sum.is_finite()
                        && *top1_agreement_sum >= 0.0
                        && *top1_agreement_sum <= examples as f64,
                    "top-1 agreement sum is outside the example count"
                );
            }
            Self::Grpo {
                mean_reward_sum,
                reward_stddev_sum,
                mean_kl_sum,
                clipped_fraction_sum,
            } => {
                ensure!(mean_reward_sum.is_finite(), "GRPO reward sum is not finite");
                ensure!(
                    reward_stddev_sum.is_finite() && *reward_stddev_sum >= 0.0,
                    "GRPO reward standard-deviation sum is invalid"
                );
                ensure!(
                    mean_kl_sum.is_finite() && *mean_kl_sum >= 0.0,
                    "GRPO KL sum is invalid"
                );
                ensure!(
                    clipped_fraction_sum.is_finite()
                        && *clipped_fraction_sum >= 0.0
                        && *clipped_fraction_sum <= examples as f64,
                    "GRPO clipped-fraction sum is invalid"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingUpdateSummary {
    pub examples: u64,
    /// Exact trainable-policy tokens observed from the backend's monotonic
    /// counter while recomputing this update.
    pub model_tokens: u64,
    pub loss_sum: f64,
    pub execution_sha256: String,
    pub objective: PostTrainingObjectiveSummary,
}

impl PostTrainingUpdateSummary {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.examples > 0 && self.model_tokens > 0 && self.loss_sum.is_finite(),
            "post-training update summary is empty or non-finite"
        );
        if matches!(
            &self.objective,
            PostTrainingObjectiveSummary::Dpo { .. }
                | PostTrainingObjectiveSummary::ForwardKl { .. }
        ) {
            ensure!(
                self.loss_sum >= 0.0,
                "DPO/forward-KL loss sum must be non-negative"
            );
        }
        validate_sha256_identity(&self.execution_sha256, "post-training execution")?;
        self.objective.validate(self.examples)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedPostTrainingUpdate {
    pub transaction_id: String,
    pub phase_sha256: String,
    pub previous_receipt_chain_sha256: String,
    pub update_index: u64,
    pub start: PostTrainingRecordPosition,
    pub end: PostTrainingRecordPosition,
    pub rng_start: u64,
    pub rng_end: u64,
    pub batch_sha256: String,
    pub base_model: ImmutableModelCheckpoint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_optimizer: Option<ImmutableArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_before: Option<PostTrainingClockReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_after: Option<PostTrainingClockValues>,
    pub summary: PostTrainingUpdateSummary,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingClockValues {
    pub optimizer_steps: u64,
    pub model_tokens: u64,
}

impl PostTrainingClockValues {
    fn selected(self, clock: UpdateClock) -> u64 {
        match clock {
            UpdateClock::OptimizerSteps => self.optimizer_steps,
            UpdateClock::ModelTokens => self.model_tokens,
        }
    }
}

#[derive(Serialize)]
struct PreparedUpdateBody<'a> {
    phase_sha256: &'a str,
    previous_receipt_chain_sha256: &'a str,
    update_index: u64,
    start: PostTrainingRecordPosition,
    end: PostTrainingRecordPosition,
    rng_start: u64,
    rng_end: u64,
    batch_sha256: &'a str,
    base_model: &'a ImmutableModelCheckpoint,
    base_optimizer: &'a Option<ImmutableArtifact>,
    clock_before: &'a Option<PostTrainingClockReceipt>,
    clock_after: &'a Option<PostTrainingClockValues>,
    summary: &'a PostTrainingUpdateSummary,
}

impl PreparedPostTrainingUpdate {
    #[allow(clippy::too_many_arguments)]
    fn new(
        phase_sha256: String,
        previous_receipt_chain_sha256: String,
        update_index: u64,
        start: PostTrainingRecordPosition,
        end: PostTrainingRecordPosition,
        rng_start: u64,
        rng_end: u64,
        batch_sha256: String,
        base_model: ImmutableModelCheckpoint,
        base_optimizer: Option<ImmutableArtifact>,
        clock_before: Option<PostTrainingClockReceipt>,
        summary: PostTrainingUpdateSummary,
    ) -> Result<Self> {
        let clock_after = clock_before
            .as_ref()
            .map(|clock| {
                Ok::<_, anyhow::Error>(PostTrainingClockValues {
                    optimizer_steps: clock
                        .optimizer_steps
                        .checked_add(1)
                        .context("cumulative post-training optimizer clock overflows u64")?,
                    model_tokens: clock
                        .model_tokens
                        .checked_add(summary.model_tokens)
                        .context("cumulative post-training model-token clock overflows u64")?,
                })
            })
            .transpose()?;
        let mut update = Self {
            transaction_id: String::new(),
            phase_sha256,
            previous_receipt_chain_sha256,
            update_index,
            start,
            end,
            rng_start,
            rng_end,
            batch_sha256,
            base_model,
            base_optimizer,
            clock_before,
            clock_after,
            summary,
        };
        update.transaction_id = update.computed_transaction_id()?;
        update.validate()?;
        Ok(update)
    }

    fn body(&self) -> PreparedUpdateBody<'_> {
        PreparedUpdateBody {
            phase_sha256: &self.phase_sha256,
            previous_receipt_chain_sha256: &self.previous_receipt_chain_sha256,
            update_index: self.update_index,
            start: self.start,
            end: self.end,
            rng_start: self.rng_start,
            rng_end: self.rng_end,
            batch_sha256: &self.batch_sha256,
            base_model: &self.base_model,
            base_optimizer: &self.base_optimizer,
            clock_before: &self.clock_before,
            clock_after: &self.clock_after,
            summary: &self.summary,
        }
    }

    fn computed_transaction_id(&self) -> Result<String> {
        canonical_sha256(&self.body())
    }

    fn validate(&self) -> Result<()> {
        validate_sha256_identity(&self.phase_sha256, "post-training phase")?;
        validate_sha256_identity(
            &self.previous_receipt_chain_sha256,
            "post-training receipt chain",
        )?;
        validate_sha256_identity(&self.batch_sha256, "post-training batch")?;
        self.base_model.validate()?;
        if let Some(optimizer) = &self.base_optimizer {
            optimizer.validate()?;
        }
        self.summary.validate()?;
        match (&self.clock_before, &self.clock_after) {
            (Some(before), Some(after)) => {
                before.validate()?;
                ensure!(
                    before.checkpoint_sha256 == self.base_model.sha256(),
                    "post-training clock belongs to another base checkpoint"
                );
                ensure!(
                    after.optimizer_steps
                        == before
                            .optimizer_steps
                            .checked_add(1)
                            .context("post-training optimizer clock overflows u64")?
                        && after.model_tokens
                            == before
                                .model_tokens
                                .checked_add(self.summary.model_tokens)
                                .context("post-training model-token clock overflows u64")?,
                    "post-training clock advance disagrees with the prepared update"
                );
            }
            (None, None) => {}
            _ => bail!("prepared post-training update has a partial clock advance"),
        }
        ensure!(
            self.end.epoch > self.start.epoch
                || (self.end.epoch == self.start.epoch && self.end.record > self.start.record),
            "post-training update does not advance its input cursor"
        );
        let reserved_rngs = self
            .rng_end
            .checked_sub(self.rng_start)
            .context("post-training RNG range is inverted")?;
        match &self.summary.objective {
            PostTrainingObjectiveSummary::Dpo { .. } => {
                let expected = self
                    .summary
                    .examples
                    .checked_mul(2)
                    .context("DPO model RNG range overflows u64")?;
                ensure!(
                    reserved_rngs == expected,
                    "DPO prepared update must reserve exactly two model RNG substreams per example"
                );
            }
            PostTrainingObjectiveSummary::ForwardKl { .. } => ensure!(
                reserved_rngs == self.summary.examples,
                "forward-KL prepared update must reserve exactly one model RNG substream per example"
            ),
            PostTrainingObjectiveSummary::Grpo { .. } => {
                let per_group = self
                    .summary
                    .examples
                    .checked_mul(2)
                    .context("GRPO model RNG range overflows u64")?;
                ensure!(
                    reserved_rngs % per_group == 0,
                    "GRPO prepared update has an incomplete model RNG group"
                );
                let group_size = reserved_rngs / per_group;
                let maximum = u64::try_from(MAX_POST_TRAINING_ROLLOUTS_PER_PROMPT)
                    .expect("GRPO group-size limit fits u64");
                ensure!(
                    (2..=maximum).contains(&group_size),
                    "GRPO prepared update model RNG range encodes an invalid group size"
                );
            }
        }
        ensure!(
            self.transaction_id == self.computed_transaction_id()?,
            "post-training transaction id does not match its content"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingCommittedState {
    pub model: ImmutableModelCheckpoint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<ImmutableArtifact>,
}

impl PostTrainingCommittedState {
    fn validate(&self) -> Result<()> {
        self.model.validate()?;
        if let Some(optimizer) = &self.optimizer {
            optimizer.validate()?;
        }
        Ok(())
    }
}

/// The publisher's proof that the attached trainable runtime was restored to
/// the exact committed model/optimizer generation before recomputation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostTrainingRestoreReceipt {
    pub publisher_identity: String,
    pub model_sha256: String,
    pub optimizer_sha256: Option<String>,
}

impl PostTrainingRestoreReceipt {
    pub fn for_state(
        publisher_identity: impl Into<String>,
        state: &PostTrainingCommittedState,
    ) -> Self {
        Self {
            publisher_identity: publisher_identity.into(),
            model_sha256: state.model.sha256().to_owned(),
            optimizer_sha256: state
                .optimizer
                .as_ref()
                .map(|artifact| artifact.sha256().to_owned()),
        }
    }
}

/// Immutable optimizer publication receipt. A retry for an existing
/// `transaction_id` must return byte-for-byte equal fields and must not invoke
/// the optimizer closure again.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingUpdateReceipt {
    pub transaction_id: String,
    pub previous_receipt_chain_sha256: String,
    pub publisher_identity: String,
    pub checkpoint: ImmutableModelCheckpoint,
    pub optimizer: ImmutableArtifact,
    pub receipt_uri: String,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct UpdateReceiptBody<'a> {
    transaction_id: &'a str,
    previous_receipt_chain_sha256: &'a str,
    publisher_identity: &'a str,
    checkpoint: &'a ImmutableModelCheckpoint,
    optimizer: &'a ImmutableArtifact,
    receipt_uri: &'a str,
}

impl PostTrainingUpdateReceipt {
    pub fn new(
        plan: &PreparedPostTrainingUpdate,
        publisher_identity: impl Into<String>,
        checkpoint: ImmutableModelCheckpoint,
        optimizer: ImmutableArtifact,
        receipt_uri: impl Into<String>,
    ) -> Result<Self> {
        let mut receipt = Self {
            transaction_id: plan.transaction_id.clone(),
            previous_receipt_chain_sha256: plan.previous_receipt_chain_sha256.clone(),
            publisher_identity: publisher_identity.into(),
            checkpoint,
            optimizer,
            receipt_uri: receipt_uri.into(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.computed_sha256()?;
        receipt.validate(plan, &receipt.publisher_identity)?;
        Ok(receipt)
    }

    fn body(&self) -> UpdateReceiptBody<'_> {
        UpdateReceiptBody {
            transaction_id: &self.transaction_id,
            previous_receipt_chain_sha256: &self.previous_receipt_chain_sha256,
            publisher_identity: &self.publisher_identity,
            checkpoint: &self.checkpoint,
            optimizer: &self.optimizer,
            receipt_uri: &self.receipt_uri,
        }
    }

    fn computed_sha256(&self) -> Result<String> {
        canonical_sha256(&self.body())
    }

    fn validate(&self, plan: &PreparedPostTrainingUpdate, publisher: &str) -> Result<()> {
        self.checkpoint.validate()?;
        self.optimizer.validate()?;
        ensure!(
            self.transaction_id == plan.transaction_id,
            "optimizer receipt belongs to another update transaction"
        );
        ensure!(
            self.previous_receipt_chain_sha256 == plan.previous_receipt_chain_sha256,
            "optimizer receipt belongs to another receipt chain"
        );
        ensure!(
            self.publisher_identity == publisher,
            "optimizer receipt belongs to another publisher"
        );
        ensure!(
            !self.receipt_uri.trim().is_empty(),
            "optimizer receipt URI is empty"
        );
        validate_sha256_identity(&self.receipt_sha256, "optimizer receipt")?;
        ensure!(
            self.receipt_sha256 == self.computed_sha256()?,
            "optimizer receipt hash does not match its content"
        );
        ensure!(
            self.checkpoint.uri() != plan.base_model.uri()
                && !self.checkpoint.same_content(&plan.base_model),
            "post-training optimizer update did not publish a new model generation"
        );
        if let Some(base) = &plan.base_optimizer {
            ensure!(
                self.optimizer.uri() != base.uri() && self.optimizer.sha256() != base.sha256(),
                "post-training optimizer update did not publish a new optimizer generation"
            );
        }
        Ok(())
    }
}

/// Publisher which atomically owns model/optimizer restoration and immutable
/// update publication. It may keep a transaction table in local or remote
/// durable storage, but publication must be idempotent by the plan's content
/// hash. On a retry it returns the original receipt without invoking `apply`.
pub trait PostTrainingUpdatePublisher {
    /// Content hash of the exact publisher implementation/configuration.
    fn identity(&self) -> &str;

    /// Restore the policy and optimizer attached to this publisher to `state`,
    /// clearing any uncommitted gradients left by a failed attempt.
    fn restore_committed(
        &mut self,
        state: &PostTrainingCommittedState,
    ) -> Result<PostTrainingRestoreReceipt>;

    /// Apply and durably publish one update. Implementations must key the
    /// operation by `plan.transaction_id` and never overwrite an artifact. A
    /// new transaction invokes `apply` exactly once; an idempotent retry
    /// returns its original receipt without invoking `apply`.
    fn publish_update(
        &mut self,
        plan: &PreparedPostTrainingUpdate,
        apply: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PostTrainingUpdateReceipt>;
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingBoundaryReceipt {
    pub transaction_id: String,
    pub hook_identity: String,
    pub input_model_sha256: String,
    pub state: PostTrainingCommittedState,
    pub clock: PostTrainingClockReceipt,
    pub native_sleep_checkpoint_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct BoundaryReceiptBody<'a> {
    transaction_id: &'a str,
    hook_identity: &'a str,
    input_model_sha256: &'a str,
    state: &'a PostTrainingCommittedState,
    clock: &'a PostTrainingClockReceipt,
    native_sleep_checkpoint_sha256: &'a str,
}

impl PostTrainingBoundaryReceipt {
    pub fn new(
        transaction_id: impl Into<String>,
        hook_identity: impl Into<String>,
        input: &PostTrainingCommittedState,
        state: PostTrainingCommittedState,
        clock: PostTrainingClockReceipt,
        native_sleep_checkpoint_sha256: impl Into<String>,
    ) -> Result<Self> {
        let mut receipt = Self {
            transaction_id: transaction_id.into(),
            hook_identity: hook_identity.into(),
            input_model_sha256: input.model.sha256().to_owned(),
            state,
            clock,
            native_sleep_checkpoint_sha256: native_sleep_checkpoint_sha256.into(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = canonical_sha256(&receipt.body())?;
        Ok(receipt)
    }

    fn body(&self) -> BoundaryReceiptBody<'_> {
        BoundaryReceiptBody {
            transaction_id: &self.transaction_id,
            hook_identity: &self.hook_identity,
            input_model_sha256: &self.input_model_sha256,
            state: &self.state,
            clock: &self.clock,
            native_sleep_checkpoint_sha256: &self.native_sleep_checkpoint_sha256,
        }
    }

    fn validate(
        &self,
        transaction_id: &str,
        hook_identity: &str,
        input: &PostTrainingCommittedState,
        expected_clock: PostTrainingClockValues,
    ) -> Result<()> {
        input.validate()?;
        self.state.validate()?;
        ensure!(
            self.transaction_id == transaction_id
                && self.hook_identity == hook_identity
                && self.input_model_sha256 == input.model.sha256(),
            "periodic-sleep boundary receipt belongs to another boundary"
        );
        validate_sha256_identity(&self.receipt_sha256, "periodic-sleep receipt")?;
        validate_sha256_identity(
            &self.native_sleep_checkpoint_sha256,
            "periodic-sleep native checkpoint",
        )?;
        self.clock.validate()?;
        ensure!(
            self.clock.authority_identity == hook_identity
                && self.clock.checkpoint_sha256 == self.state.model.sha256()
                && self.clock.optimizer_steps == expected_clock.optimizer_steps
                && self.clock.model_tokens == expected_clock.model_tokens,
            "periodic-sleep boundary clock receipt is invalid"
        );
        ensure!(
            self.receipt_sha256 == canonical_sha256(&self.body())?,
            "periodic-sleep receipt hash does not match its content"
        );
        Ok(())
    }
}

pub struct PostTrainingBoundaryRequest<'a> {
    pub transaction_id: &'a str,
    pub workflow_signature: &'a str,
    pub phase_name: &'a str,
    pub clock_before: &'a PostTrainingClockReceipt,
    pub clock_after: PostTrainingClockValues,
    pub config: &'a InModelSleepConfig,
    pub input: &'a PostTrainingCommittedState,
}

/// Injected tensor/model implementation used by the first-party periodic
/// post-training controller. Implementations load the authenticated native
/// sleep cursor from the immutable input generation and own the same tensor,
/// optimizer, judge, and publisher adapters used by ordinary periodic sleep.
pub trait NativePostTrainingSleepRuntime {
    fn identity(&self) -> &str;

    fn restore_boundary_cursor(
        &mut self,
        request: &PostTrainingBoundaryRequest<'_>,
    ) -> Result<NativeSleepCheckpoint>;

    fn advance_and_drain(
        &mut self,
        request: &PostTrainingBoundaryRequest<'_>,
        checkpoint: &mut NativeSleepCheckpoint,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<PostTrainingCommittedState>;
}

/// Production orchestration for periodic DPO/forward-KL/GRPO sleep. Tensor
/// execution remains injected, but clock validation, complete due-queue
/// draining, final cursor persistence, and receipt construction are owned by
/// trainer core.
pub struct NativePostTrainingBoundaryController<R> {
    runtime: R,
}

impl<R: NativePostTrainingSleepRuntime> NativePostTrainingBoundaryController<R> {
    pub fn new(runtime: R) -> Result<Self> {
        validate_sha256_identity(runtime.identity(), "native post-training sleep runtime")?;
        Ok(Self { runtime })
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }
}

/// Explicit bridge to trainer-controlled in-model sleep at post-training
/// optimizer boundaries. It must be transaction-idempotent because a process
/// can stop after the hook publishes but before the phase cursor is durable.
pub trait PostTrainingBoundaryHook {
    fn identity(&self) -> &str;
    fn drive_boundary(
        &mut self,
        request: &PostTrainingBoundaryRequest<'_>,
        resume: Option<NativeSleepCheckpoint>,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<PostTrainingBoundaryReceipt>;
}

impl<R: NativePostTrainingSleepRuntime> PostTrainingBoundaryHook
    for NativePostTrainingBoundaryController<R>
{
    fn identity(&self) -> &str {
        self.runtime.identity()
    }

    fn drive_boundary(
        &mut self,
        request: &PostTrainingBoundaryRequest<'_>,
        resume: Option<NativeSleepCheckpoint>,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<PostTrainingBoundaryReceipt> {
        validate_sha256_identity(request.transaction_id, "post-training sleep transaction")?;
        validate_sha256_identity(request.workflow_signature, "post-training sleep workflow")?;
        request.clock_before.validate()?;
        request.input.validate()?;
        ensure!(
            request.clock_before.authority_identity == self.identity()
                && request.clock_before.checkpoint_sha256 == request.input.model.sha256(),
            "post-training sleep clock/input binding is invalid"
        );
        let expected_optimizer = request
            .clock_before
            .optimizer_steps
            .checked_add(1)
            .context("post-training optimizer clock overflows u64")?;
        ensure!(
            request.clock_after.optimizer_steps == expected_optimizer
                && request.clock_after.model_tokens > request.clock_before.model_tokens,
            "post-training sleep target clock does not advance exactly one optimizer update"
        );
        validate_periodic_sleep_clock_window(
            request.config,
            request.clock_before,
            request.clock_after,
        )?;

        let resumed = resume.is_some();
        let mut checkpoint = match resume {
            Some(checkpoint) => checkpoint,
            None => self.runtime.restore_boundary_cursor(request)?,
        };
        checkpoint.live_checkpoint.validate()?;
        checkpoint.sleep.validate_resume()?;
        ensure!(
            checkpoint.workflow_signature == request.workflow_signature
                && checkpoint.phase_name == request.phase_name,
            "native post-training sleep cursor belongs to another workflow phase"
        );
        ensure!(
            checkpoint.input_checkpoint.uri == request.input.model.uri()
                && checkpoint.input_checkpoint.sha256 == request.input.model.sha256(),
            "native post-training sleep cursor belongs to another boundary input checkpoint"
        );
        if !resumed {
            ensure!(
                checkpoint.live_checkpoint.uri == request.input.model.uri()
                    && checkpoint.live_checkpoint.sha256 == request.input.model.sha256(),
                "post-training sleep runtime restored another input generation"
            );
            ensure!(
                checkpoint.sleep.clock
                    == request.clock_before.selected(request.config.schedule.clock),
                "post-training sleep cursor starts at a different cumulative clock"
            );
        }
        ensure!(
            checkpoint.sleep.clock <= request.clock_after.selected(request.config.schedule.clock),
            "resumed post-training sleep cursor is ahead of its target clock"
        );

        let state = self
            .runtime
            .advance_and_drain(request, &mut checkpoint, progress)?;
        checkpoint.live_checkpoint.validate()?;
        checkpoint.sleep.validate_resume()?;
        state.validate()?;
        ensure!(
            checkpoint.workflow_signature == request.workflow_signature
                && checkpoint.phase_name == request.phase_name
                && checkpoint.input_checkpoint.uri == request.input.model.uri()
                && checkpoint.input_checkpoint.sha256 == request.input.model.sha256(),
            "post-training sleep runtime changed its workflow/boundary identity"
        );
        let target = request.clock_after.selected(request.config.schedule.clock);
        ensure!(
            checkpoint.sleep.clock == target
                && checkpoint.sleep.phase == SleepPhase::Wake
                && checkpoint.sleep.pending.is_none()
                && checkpoint.sleep.due_senders.is_empty()
                && checkpoint.sleep.due_clocks.is_empty(),
            "post-training sleep runtime returned before every due sender was drained"
        );
        ensure!(
            checkpoint.live_checkpoint.uri == state.model.uri()
                && checkpoint.live_checkpoint.sha256 == state.model.sha256(),
            "post-training sleep cursor and returned model generation differ"
        );
        ensure!(
            state.optimizer.is_some(),
            "post-training sleep runtime omitted the optimizer generation"
        );
        progress.persist(&checkpoint)?;
        let clock = PostTrainingClockReceipt::new(
            self.identity().to_owned(),
            state.model.sha256().to_owned(),
            request.clock_after.optimizer_steps,
            request.clock_after.model_tokens,
        )?;
        let native_sleep_checkpoint_sha256 = canonical_sha256(&checkpoint)?;
        PostTrainingBoundaryReceipt::new(
            request.transaction_id,
            self.identity().to_owned(),
            request.input,
            state,
            clock,
            native_sleep_checkpoint_sha256,
        )
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingProgress {
    pub examples: u64,
    pub optimizer_steps: u64,
    pub model_tokens: u64,
    pub loss_sum: f64,
    pub preference_correct: u64,
    pub implicit_reward_margin_sum: f64,
    pub forward_kl_sum: f64,
    pub teacher_entropy_sum: f64,
    pub top1_agreement_sum: f64,
    pub mean_reward_sum: f64,
    pub reward_stddev_sum: f64,
    pub mean_kl_sum: f64,
    pub clipped_fraction_sum: f64,
}

impl PostTrainingProgress {
    fn apply(&mut self, summary: &PostTrainingUpdateSummary) -> Result<()> {
        summary.validate()?;
        self.examples = self
            .examples
            .checked_add(summary.examples)
            .context("post-training example counter overflows u64")?;
        self.optimizer_steps = self
            .optimizer_steps
            .checked_add(1)
            .context("post-training optimizer-step counter overflows u64")?;
        self.model_tokens = self
            .model_tokens
            .checked_add(summary.model_tokens)
            .context("post-training model-token counter overflows u64")?;
        self.loss_sum += summary.loss_sum;
        match &summary.objective {
            PostTrainingObjectiveSummary::Dpo {
                preference_correct,
                implicit_reward_margin_sum,
            } => {
                self.preference_correct = self
                    .preference_correct
                    .checked_add(*preference_correct)
                    .context("DPO correct counter overflows u64")?;
                self.implicit_reward_margin_sum += implicit_reward_margin_sum;
            }
            PostTrainingObjectiveSummary::ForwardKl {
                forward_kl_sum,
                teacher_entropy_sum,
                top1_agreement_sum,
            } => {
                self.forward_kl_sum += forward_kl_sum;
                self.teacher_entropy_sum += teacher_entropy_sum;
                self.top1_agreement_sum += top1_agreement_sum;
            }
            PostTrainingObjectiveSummary::Grpo {
                mean_reward_sum,
                reward_stddev_sum,
                mean_kl_sum,
                clipped_fraction_sum,
            } => {
                self.mean_reward_sum += mean_reward_sum;
                self.reward_stddev_sum += reward_stddev_sum;
                self.mean_kl_sum += mean_kl_sum;
                self.clipped_fraction_sum += clipped_fraction_sum;
            }
        }
        ensure!(
            [
                self.loss_sum,
                self.implicit_reward_margin_sum,
                self.forward_kl_sum,
                self.teacher_entropy_sum,
                self.top1_agreement_sum,
                self.mean_reward_sum,
                self.reward_stddev_sum,
                self.mean_kl_sum,
                self.clipped_fraction_sum,
            ]
            .iter()
            .all(|value| value.is_finite()),
            "post-training progress became non-finite"
        );
        Ok(())
    }

    fn report(
        &self,
        phase: &PhaseV2,
        algorithm: &str,
        identities: &PostTrainingRuntimeIdentities,
    ) -> Result<PostTrainingPhaseReport> {
        ensure!(
            self.examples > 0 && self.optimizer_steps > 0,
            "post-training phase `{}` produced no optimizer steps",
            phase.name
        );
        let count = self.examples as f64;
        let metrics = match algorithm {
            "dpo" => BTreeMap::from([
                (
                    "preference_accuracy".to_owned(),
                    self.preference_correct as f64 / count,
                ),
                (
                    "implicit_reward_margin".to_owned(),
                    self.implicit_reward_margin_sum / count,
                ),
            ]),
            "forward_kl" => BTreeMap::from([
                ("forward_kl".to_owned(), self.forward_kl_sum / count),
                (
                    "teacher_entropy".to_owned(),
                    self.teacher_entropy_sum / count,
                ),
                ("top1_agreement".to_owned(), self.top1_agreement_sum / count),
            ]),
            "grpo" => BTreeMap::from([
                ("mean_reward".to_owned(), self.mean_reward_sum / count),
                ("reward_stddev".to_owned(), self.reward_stddev_sum / count),
                ("mean_kl".to_owned(), self.mean_kl_sum / count),
                (
                    "clipped_fraction".to_owned(),
                    self.clipped_fraction_sum / count,
                ),
            ]),
            _ => bail!("unsupported post-training algorithm `{algorithm}`"),
        };
        Ok(PostTrainingPhaseReport {
            phase: phase.name.clone(),
            algorithm: algorithm.to_owned(),
            examples: usize::try_from(self.examples).context("example count exceeds usize")?,
            optimizer_steps: usize::try_from(self.optimizer_steps)
                .context("optimizer-step count exceeds usize")?,
            trainable_model_identity: identities.trainable.clone(),
            frozen_model_identity: identities.frozen.clone(),
            mean_loss: self.loss_sum / count,
            metrics,
        })
    }
}

/// Complete durable state of a native post-training phase. A prepared update
/// is deliberately retained until its model and optimizer receipts, optional
/// sleep boundary, metric, and this cursor are durably checkpointed together.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingCursor {
    pub version: u32,
    pub workflow_signature: String,
    pub phase_index: usize,
    pub phase_name: String,
    pub phase_kind: PhaseKind,
    pub phase_sha256: String,
    pub input_checkpoint: ImmutableModelCheckpoint,
    pub input_data: PinnedPostTrainingInput,
    pub runtime_identities: PostTrainingRuntimeIdentities,
    pub publisher_identity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_hook_identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_clock: Option<PostTrainingClockReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<PostTrainingClockReceipt>,
    pub position: PostTrainingRecordPosition,
    pub rng: PostTrainingRngCursor,
    pub committed: PostTrainingCommittedState,
    pub receipt_chain_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_update_receipt: Option<PostTrainingUpdateReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_boundary_receipt: Option<PostTrainingBoundaryReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<PreparedPostTrainingUpdate>,
    pub progress: PostTrainingProgress,
}

impl PostTrainingCursor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        workflow_signature: String,
        request: &PhaseExecutionRequest,
        input_data: PinnedPostTrainingInput,
        runtime_identities: PostTrainingRuntimeIdentities,
        publisher_identity: String,
        boundary_hook_identity: Option<String>,
        clock: Option<PostTrainingClockReceipt>,
    ) -> Result<Self> {
        let input_checkpoint = request
            .input_checkpoint
            .clone()
            .context("native post-training requires an immutable input checkpoint")?;
        let phase_sha256 = canonical_sha256(&request.phase)?;
        let rng_seed = deterministic_rng_seed(
            &workflow_signature,
            &phase_sha256,
            input_checkpoint.sha256(),
        );
        let cursor = Self {
            version: POST_TRAINING_CURSOR_VERSION,
            workflow_signature,
            phase_index: request.phase_index,
            phase_name: request.phase.name.clone(),
            phase_kind: request.phase.kind,
            phase_sha256,
            input_checkpoint: input_checkpoint.clone(),
            input_data,
            runtime_identities,
            publisher_identity,
            boundary_hook_identity,
            input_clock: clock.clone(),
            clock,
            position: PostTrainingRecordPosition {
                epoch: 0,
                record: 0,
            },
            rng: PostTrainingRngCursor {
                seed: rng_seed,
                counter: 0,
            },
            committed: PostTrainingCommittedState {
                model: input_checkpoint,
                optimizer: None,
            },
            receipt_chain_sha256: empty_receipt_chain(),
            last_update_receipt: None,
            last_boundary_receipt: None,
            pending: None,
            progress: PostTrainingProgress::default(),
        };
        cursor.validate_internal()?;
        Ok(cursor)
    }

    fn validate_internal(&self) -> Result<()> {
        ensure!(
            self.version == POST_TRAINING_CURSOR_VERSION,
            "unsupported post-training cursor version {}",
            self.version
        );
        validate_sha256_identity(&self.workflow_signature, "workflow signature")?;
        validate_sha256_identity(&self.phase_sha256, "post-training phase")?;
        self.input_checkpoint.validate()?;
        self.input_data.validate()?;
        self.committed.validate()?;
        validate_sha256_identity(&self.publisher_identity, "post-training publisher")?;
        if let Some(identity) = &self.boundary_hook_identity {
            validate_sha256_identity(identity, "post-training boundary hook")?;
        }
        match (&self.boundary_hook_identity, &self.input_clock, &self.clock) {
            (Some(identity), Some(input_clock), Some(clock)) => {
                input_clock.validate()?;
                clock.validate()?;
                ensure!(
                    input_clock.authority_identity == *identity
                        && clock.authority_identity == *identity,
                    "post-training clock authority differs from its boundary hook"
                );
                ensure!(
                    input_clock.checkpoint_sha256 == self.input_checkpoint.sha256(),
                    "post-training input clock belongs to another input checkpoint"
                );
                ensure!(
                    clock.checkpoint_sha256 == self.committed.model.sha256(),
                    "post-training clock belongs to another committed checkpoint"
                );
                ensure!(
                    clock.optimizer_steps
                        == input_clock
                            .optimizer_steps
                            .checked_add(self.progress.optimizer_steps)
                            .context("post-training cumulative optimizer clock overflows u64")?
                        && clock.model_tokens
                            == input_clock
                                .model_tokens
                                .checked_add(self.progress.model_tokens)
                                .context(
                                    "post-training cumulative model-token clock overflows u64"
                                )?,
                    "post-training cumulative clock disagrees with phase progress"
                );
            }
            (None, None, None) => {}
            _ => bail!("post-training cursor has a partial periodic-sleep clock binding"),
        }
        validate_sha256_identity(&self.receipt_chain_sha256, "post-training receipt chain")?;
        ensure!(
            self.progress.examples >= self.progress.optimizer_steps,
            "post-training progress has more updates than examples"
        );
        match (&self.last_update_receipt, self.progress.optimizer_steps) {
            (None, 0) => {
                ensure!(
                    self.committed.model == self.input_checkpoint
                        && self.committed.optimizer.is_none()
                        && self.receipt_chain_sha256 == empty_receipt_chain(),
                    "empty post-training cursor has committed update state"
                );
            }
            (Some(receipt), steps) if steps > 0 => {
                receipt.checkpoint.validate()?;
                receipt.optimizer.validate()?;
                validate_sha256_identity(&receipt.transaction_id, "post-training transaction")?;
                validate_sha256_identity(
                    &receipt.previous_receipt_chain_sha256,
                    "previous post-training receipt chain",
                )?;
                validate_sha256_identity(&receipt.receipt_sha256, "optimizer receipt")?;
                ensure!(
                    receipt.publisher_identity == self.publisher_identity
                        && receipt.receipt_sha256 == receipt.computed_sha256()?,
                    "latest post-training optimizer receipt is invalid"
                );
                ensure!(
                    receipt.checkpoint == self.committed.model
                        || self.last_boundary_receipt.is_some(),
                    "post-training committed model disagrees with its latest receipt"
                );
                ensure!(
                    self.committed.optimizer.is_some(),
                    "post-training committed optimizer is missing after an update"
                );
                if let Some(boundary) = &self.last_boundary_receipt {
                    validate_sha256_identity(
                        &boundary.transaction_id,
                        "periodic-sleep transaction",
                    )?;
                    validate_sha256_identity(&boundary.receipt_sha256, "periodic-sleep receipt")?;
                    ensure!(
                        self.boundary_hook_identity.as_deref()
                            == Some(boundary.hook_identity.as_str())
                            && boundary.input_model_sha256 == receipt.checkpoint.sha256()
                            && boundary.state == self.committed
                            && self.clock.as_ref() == Some(&boundary.clock)
                            && boundary.receipt_sha256 == canonical_sha256(&boundary.body())?,
                        "latest periodic-sleep boundary receipt is invalid"
                    );
                } else {
                    ensure!(
                        self.committed.model == receipt.checkpoint
                            && self.committed.optimizer.as_ref() == Some(&receipt.optimizer),
                        "committed post-training state differs from its optimizer receipt"
                    );
                }
                let expected_chain = canonical_sha256(&(
                    "post_training_receipt_chain_v1",
                    &receipt.previous_receipt_chain_sha256,
                    &receipt.receipt_sha256,
                    self.last_boundary_receipt
                        .as_ref()
                        .map(|receipt| receipt.receipt_sha256.as_str()),
                ))?;
                ensure!(
                    self.receipt_chain_sha256 == expected_chain,
                    "post-training receipt chain does not match the latest publication"
                );
            }
            _ => bail!("post-training update receipt and counters disagree"),
        }
        if let Some(pending) = &self.pending {
            pending.validate()?;
            ensure!(
                pending.phase_sha256 == self.phase_sha256
                    && pending.previous_receipt_chain_sha256 == self.receipt_chain_sha256
                    && pending.update_index == self.progress.optimizer_steps
                    && pending.start == self.position
                    && pending.rng_start == self.rng.counter
                    && pending.base_model == self.committed.model
                    && pending.base_optimizer == self.committed.optimizer
                    && pending.clock_before == self.clock,
                "prepared post-training update does not start at the committed cursor"
            );
        }
        Ok(())
    }
}

/// Durable optimizer publication and optional inner native-sleep cursor. It
/// is checkpointed before the first inner persistence, so a crash can never
/// lose the immutable optimizer receipt or deserialize a bare sleep cursor as
/// post-training state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingBoundaryInFlight {
    pub transaction_id: String,
    pub update_receipt: PostTrainingUpdateReceipt,
    pub published_state: PostTrainingCommittedState,
    pub clock_before: PostTrainingClockReceipt,
    pub clock_after: PostTrainingClockValues,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_sleep: Option<NativeSleepCheckpoint>,
}

impl PostTrainingBoundaryInFlight {
    fn validate(
        &self,
        expected: &PreparedPostTrainingUpdate,
        publisher_identity: &str,
        config: &InModelSleepConfig,
    ) -> Result<()> {
        self.published_state.validate()?;
        let before = expected
            .clock_before
            .as_ref()
            .context("periodic update has no starting clock")?;
        let after = expected
            .clock_after
            .context("periodic update has no target clock")?;
        self.clock_before.validate()?;
        ensure!(
            self.transaction_id
                == boundary_transaction_id(
                    expected,
                    &self.update_receipt,
                    config,
                    &self.clock_before,
                    after,
                )?
                && self.clock_before.authority_identity == before.authority_identity
                && self.clock_before.optimizer_steps == before.optimizer_steps
                && self.clock_before.model_tokens == before.model_tokens
                && self.clock_after == after,
            "in-flight post-training boundary belongs to another clock/update"
        );
        self.update_receipt.validate(expected, publisher_identity)?;
        ensure!(
            self.published_state.model == self.update_receipt.checkpoint
                && self.published_state.optimizer.as_ref() == Some(&self.update_receipt.optimizer)
                && self.clock_before.checkpoint_sha256 == self.published_state.model.sha256(),
            "in-flight post-training boundary has a forged published state"
        );
        if let Some(native) = &self.native_sleep {
            native.live_checkpoint.validate()?;
            native.sleep.validate_resume()?;
            ensure!(
                native.input_checkpoint.uri == self.published_state.model.uri()
                    && native.input_checkpoint.sha256 == self.published_state.model.sha256(),
                "in-flight native sleep cursor belongs to another published update"
            );
            ensure!(
                native.sleep.clock >= before.selected(config.schedule.clock)
                    && native.sleep.clock <= after.selected(config.schedule.clock),
                "in-flight native sleep cursor is outside its clock interval"
            );
        }
        Ok(())
    }
}

/// Versioned tagged resume envelope. There is intentionally no parser for the
/// former bare cursor: accepting it would make a native-sleep checkpoint and
/// an outer post-training cursor ambiguous after interruption.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PostTrainingResumeEnvelope {
    Wake {
        version: u32,
        cursor: PostTrainingCursor,
    },
    Boundary {
        version: u32,
        cursor: PostTrainingCursor,
        boundary: Box<PostTrainingBoundaryInFlight>,
    },
}

impl PostTrainingResumeEnvelope {
    fn wake(cursor: PostTrainingCursor) -> Self {
        Self::Wake {
            version: POST_TRAINING_RESUME_VERSION,
            cursor,
        }
    }

    fn boundary(cursor: PostTrainingCursor, boundary: PostTrainingBoundaryInFlight) -> Self {
        Self::Boundary {
            version: POST_TRAINING_RESUME_VERSION,
            cursor,
            boundary: Box::new(boundary),
        }
    }

    fn into_parts(self) -> Result<(PostTrainingCursor, Option<PostTrainingBoundaryInFlight>)> {
        match self {
            Self::Wake { version, cursor } => {
                ensure!(
                    version == POST_TRAINING_RESUME_VERSION,
                    "unsupported post-training resume-envelope version {version}"
                );
                Ok((cursor, None))
            }
            Self::Boundary {
                version,
                cursor,
                boundary,
            } => {
                ensure!(
                    version == POST_TRAINING_RESUME_VERSION,
                    "unsupported post-training resume-envelope version {version}"
                );
                Ok((cursor, Some(*boundary)))
            }
        }
    }
}

/// One phase's concrete device runtime. It is intentionally borrowed and
/// contains no site-specific loader, storage client, or global singleton. The
/// runtime and publisher must refer to the same model state: restoring a
/// committed publisher generation must also discard any previously staged
/// device batch before deterministic replay.
pub struct PostTrainingExecutionContext<'a> {
    pub runtime: PostTrainingPhaseRuntime<'a>,
    pub publisher: &'a mut dyn PostTrainingUpdatePublisher,
    pub boundary_hook: Option<&'a mut dyn PostTrainingBoundaryHook>,
    /// Authenticated cumulative clocks from the exact input checkpoint.
    /// Required for periodic sleep and rejected for an ordinary phase.
    pub starting_clock: Option<PostTrainingClockReceipt>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResumablePostTrainingOutcome {
    Yielded(PostTrainingCursor),
    Complete {
        cursor: PostTrainingCursor,
        report: PostTrainingPhaseReport,
    },
}

/// Native WorkflowV2 executor for DPO, forward-KL, and GRPO. `update_budget`
/// is a cooperative-yield control only; it does not alter batch geometry or
/// the serialized transaction sequence.
#[derive(Clone, Debug)]
pub struct NativePostTrainingPhaseExecutor {
    workflow_signature: String,
    update_budget: usize,
}

impl NativePostTrainingPhaseExecutor {
    pub fn new(workflow_signature: impl Into<String>) -> Result<Self> {
        Self::with_update_budget(workflow_signature, usize::MAX)
    }

    pub fn with_update_budget(
        workflow_signature: impl Into<String>,
        update_budget: usize,
    ) -> Result<Self> {
        let workflow_signature = workflow_signature.into();
        validate_sha256_identity(&workflow_signature, "workflow signature")?;
        ensure!(
            update_budget > 0,
            "post-training update budget must be positive"
        );
        Ok(Self {
            workflow_signature,
            update_budget,
        })
    }
}

impl<'a> PhaseExecutor<PostTrainingExecutionContext<'a>> for NativePostTrainingPhaseExecutor {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        context: &mut PostTrainingExecutionContext<'a>,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        let resume = request
            .resume_state
            .clone()
            .map(serde_json::from_value::<PostTrainingResumeEnvelope>)
            .transpose()
            .context("invalid native post-training resume envelope")?;
        let outcome = drive_resumable_post_training_phase(
            &self.workflow_signature,
            request,
            &mut context.runtime,
            context.publisher,
            &mut context.boundary_hook,
            context.starting_clock.as_ref(),
            resume,
            self.update_budget,
            progress,
        )?;
        match outcome {
            ResumablePostTrainingOutcome::Yielded(cursor) => Ok(PhaseExecutionResult::Yielded {
                resume_state: serde_json::to_value(PostTrainingResumeEnvelope::wake(cursor))?,
            }),
            ResumablePostTrainingOutcome::Complete { cursor, .. } => Ok(
                PhaseExecutionResult::Complete(PhaseProduct::ModelCandidate {
                    checkpoint: cursor.committed.model,
                }),
            ),
        }
    }
}

/// Drive deterministic updates from a committed cursor. The runner publishes
/// a prepared cursor before calling the optimizer publisher and a committed
/// cursor after restoring the returned immutable generations. Consequently,
/// either interruption window is retried with the same transaction id.
#[allow(clippy::too_many_arguments)]
pub fn drive_resumable_post_training_phase(
    workflow_signature: &str,
    request: &PhaseExecutionRequest,
    runtime: &mut PostTrainingPhaseRuntime<'_>,
    publisher: &mut dyn PostTrainingUpdatePublisher,
    boundary_hook: &mut Option<&mut dyn PostTrainingBoundaryHook>,
    starting_clock: Option<&PostTrainingClockReceipt>,
    resume: Option<PostTrainingResumeEnvelope>,
    update_budget: usize,
    progress: &mut dyn PhaseProgressSink,
) -> Result<ResumablePostTrainingOutcome> {
    validate_sha256_identity(workflow_signature, "workflow signature")?;
    ensure!(
        update_budget > 0,
        "post-training update budget must be positive"
    );
    let phase = &request.phase;
    validate_resumable_phase(phase)?;
    let input_checkpoint = request
        .input_checkpoint
        .as_ref()
        .context("native post-training requires an immutable input checkpoint")?;
    input_checkpoint.validate()?;
    let identities = validate_and_capture_runtime_identities(phase, runtime)?;
    validate_sha256_identity(publisher.identity(), "post-training publisher")?;
    let (hook_identity, initial_clock) = match (
        &phase.periodic_sleep,
        boundary_hook.as_deref(),
        starting_clock,
    ) {
        (Some(_), Some(hook), Some(clock)) => {
            validate_sha256_identity(hook.identity(), "post-training boundary hook")?;
            clock.validate()?;
            ensure!(
                clock.authority_identity == hook.identity()
                    && clock.checkpoint_sha256 == input_checkpoint.sha256(),
                "post-training starting clock is not authenticated for its input/hook"
            );
            (Some(hook.identity().to_owned()), Some(clock.clone()))
        }
        (Some(_), None, _) => bail!(
            "post-training phase `{}` has periodic_sleep but no explicit boundary hook",
            phase.name
        ),
        (Some(_), Some(_), None) => bail!(
            "post-training phase `{}` has periodic_sleep but no authenticated starting clock",
            phase.name
        ),
        (None, Some(_), _) => bail!(
            "post-training phase `{}` supplied a boundary hook without periodic_sleep",
            phase.name
        ),
        (None, None, Some(_)) => bail!(
            "post-training phase `{}` supplied a starting sleep clock without periodic_sleep",
            phase.name
        ),
        (None, None, None) => (None, None),
    };
    // Perform large immutable-file hashes once per execution attempt, after
    // cheap runtime, publisher, and boundary-clock preflight. Rehashing a
    // multi-gigabyte teacher/reference for every update would serialize the
    // hot loop on storage.
    phase
        .post_training
        .as_ref()
        .expect("validated native post-training algorithm")
        .verify_local_artifacts()?;
    let data = phase
        .data
        .as_deref()
        .context("native post-training phase has no data")?;
    let authenticated_input = AuthenticatedPostTrainingInput::open(data)?;
    let actual_input = authenticated_input.identity.clone();
    let phase_sha256 = canonical_sha256(phase)?;
    let (resume_cursor, mut in_flight) = resume
        .map(PostTrainingResumeEnvelope::into_parts)
        .transpose()?
        .map_or((None, None), |(cursor, boundary)| (Some(cursor), boundary));
    let mut cursor = match resume_cursor {
        Some(cursor) => {
            cursor.validate_internal()?;
            ensure!(
                cursor.workflow_signature == workflow_signature
                    && cursor.phase_index == request.phase_index
                    && cursor.phase_name == phase.name
                    && cursor.phase_kind == phase.kind
                    && cursor.phase_sha256 == phase_sha256,
                "native post-training cursor belongs to another workflow phase"
            );
            ensure!(
                &cursor.input_checkpoint == input_checkpoint,
                "native post-training cursor belongs to another input checkpoint"
            );
            ensure!(
                cursor.input_data == actual_input,
                "native post-training cursor belongs to different input bytes"
            );
            ensure!(
                cursor.runtime_identities == identities,
                "native post-training device-runtime identity changed across resume"
            );
            ensure!(
                cursor.publisher_identity == publisher.identity(),
                "native post-training publisher identity changed across resume"
            );
            ensure!(
                cursor.boundary_hook_identity == hook_identity,
                "native post-training boundary-hook identity changed across resume"
            );
            ensure!(
                cursor.input_clock.as_ref() == initial_clock.as_ref(),
                "native post-training starting clock changed across resume"
            );
            cursor
        }
        None => PostTrainingCursor::new(
            workflow_signature.to_owned(),
            request,
            actual_input,
            identities,
            publisher.identity().to_owned(),
            hook_identity,
            initial_clock,
        )?,
    };
    let mut batches = PostTrainingBatchStream::new(
        authenticated_input,
        phase
            .task
            .as_ref()
            .expect("validated post-training task")
            .clone(),
        cursor.position,
        u64::try_from(phase.epochs_or_default()).context("epoch count exceeds u64")?,
    )?;
    ensure!(
        in_flight.is_none()
            || (phase.periodic_sleep.is_some()
                && cursor.pending.is_some()
                && cursor.clock.is_some()),
        "post-training boundary envelope has no matching prepared periodic update"
    );

    let mut updates_this_drive = 0usize;
    loop {
        cursor.validate_internal()?;
        validate_and_capture_runtime_identities(phase, runtime).and_then(|current| {
            ensure!(
                current == cursor.runtime_identities,
                "native post-training device-runtime identity drifted during execution"
            );
            Ok(())
        })?;
        // Runtime identity callbacks are external code and can take long
        // enough for the published input path to change.  Bracket the restore
        // itself so an already-invalid phase never pays for, or observes side
        // effects from, restoring a model generation it can no longer use.
        batches.ensure_still_published()?;
        validate_restore_receipt(
            &publisher.restore_committed(&cursor.committed)?,
            publisher.identity(),
            &cursor.committed,
        )?;
        // Restoration is deployment-controlled and may perform arbitrary I/O.
        // Recheck the authenticated pathname before either returning a final
        // product or consuming another batch from the retained description.
        batches.ensure_still_published()?;

        if phase_is_complete(phase, &cursor)? {
            ensure!(
                cursor.pending.is_none() && in_flight.is_none(),
                "completed phase retains a prepared update or sleep boundary"
            );
            let report = cursor.progress.report(
                phase,
                post_training_algorithm_name(
                    phase
                        .post_training
                        .as_ref()
                        .expect("validated post-training config"),
                ),
                &cursor.runtime_identities,
            )?;
            return Ok(ResumablePostTrainingOutcome::Complete { cursor, report });
        }

        let update_examples = update_examples(phase)?;
        let start = cursor.position;
        let batch = batches.next_batch(update_examples)?;
        ensure!(
            !batch.examples.is_empty(),
            "post-training phase `{}` reached no data before its configured end",
            phase.name
        );
        let batch_sha256 = canonical_sha256(&batch.examples)?;
        let rng_start = cursor.rng.counter;
        let prepared_computation = accumulate_resumable_update(
            phase,
            &batch.examples,
            &batch_sha256,
            &cursor.committed,
            runtime,
            cursor.rng.seed,
            rng_start,
        )?;
        let expected = PreparedPostTrainingUpdate::new(
            cursor.phase_sha256.clone(),
            cursor.receipt_chain_sha256.clone(),
            cursor.progress.optimizer_steps,
            start,
            batch.end,
            rng_start,
            prepared_computation.rng_end,
            batch_sha256,
            cursor.committed.model.clone(),
            cursor.committed.optimizer.clone(),
            cursor.clock.clone(),
            prepared_computation.summary,
        )?;
        if let Some(config) = phase.periodic_sleep.as_ref() {
            validate_periodic_sleep_clock_window(
                config,
                expected
                    .clock_before
                    .as_ref()
                    .context("periodic update has no starting clock")?,
                expected
                    .clock_after
                    .context("periodic update has no target clock")?,
            )?;
        }
        match &cursor.pending {
            Some(saved) => ensure!(
                saved == &expected,
                "recomputed post-training update differs from its prepared cursor"
            ),
            None => {
                cursor.pending = Some(expected.clone());
                progress.checkpoint(serde_json::to_value(PostTrainingResumeEnvelope::wake(
                    cursor.clone(),
                ))?)?;
            }
        }
        // The prepared update is durable, but no model mutation may be
        // published from a batch whose pathname stopped naming the exact
        // authenticated file after preparation.
        batches.ensure_still_published()?;

        let (receipt, published_state) = match &in_flight {
            Some(boundary) => {
                let config = phase
                    .periodic_sleep
                    .as_ref()
                    .context("ordinary post-training phase resumed a sleep boundary")?;
                boundary.validate(&expected, publisher.identity(), config)?;
                (
                    boundary.update_receipt.clone(),
                    boundary.published_state.clone(),
                )
            }
            None => {
                let learning_rate_scale = phase.learning_rate_scale_or_default();
                let mut apply = || optimizer_step(runtime, learning_rate_scale);
                let receipt = publisher.publish_update(&expected, &mut apply)?;
                receipt.validate(&expected, publisher.identity())?;
                let published_state = PostTrainingCommittedState {
                    model: receipt.checkpoint.clone(),
                    optimizer: Some(receipt.optimizer.clone()),
                };
                (receipt, published_state)
            }
        };
        // A publisher can take long enough for the source pathname to change.
        // Reject before paying to restore the new generation or entering an
        // optional sleep boundary; its immutable artifacts remain unreferenced.
        batches.ensure_still_published()?;
        validate_restore_receipt(
            &publisher.restore_committed(&published_state)?,
            publisher.identity(),
            &published_state,
        )?;
        batches.ensure_still_published()?;

        let (committed, boundary_receipt) =
            match (phase.periodic_sleep.as_ref(), boundary_hook.as_deref_mut()) {
                (Some(config), Some(hook)) => {
                    let clock_before = expected
                        .clock_before
                        .as_ref()
                        .context("periodic update has no starting clock")?;
                    let clock_after = expected
                        .clock_after
                        .context("periodic update has no target clock")?;
                    let mut durable = match in_flight.take() {
                        Some(boundary) => boundary,
                        None => {
                            let published_clock = PostTrainingClockReceipt::new(
                                clock_before.authority_identity.clone(),
                                published_state.model.sha256().to_owned(),
                                clock_before.optimizer_steps,
                                clock_before.model_tokens,
                            )?;
                            let transaction_id = boundary_transaction_id(
                                &expected,
                                &receipt,
                                config,
                                &published_clock,
                                clock_after,
                            )?;
                            let boundary = PostTrainingBoundaryInFlight {
                                transaction_id: transaction_id.clone(),
                                update_receipt: receipt.clone(),
                                published_state: published_state.clone(),
                                clock_before: published_clock,
                                clock_after,
                                native_sleep: None,
                            };
                            boundary.validate(&expected, publisher.identity(), config)?;
                            progress.checkpoint(serde_json::to_value(
                                PostTrainingResumeEnvelope::boundary(
                                    cursor.clone(),
                                    boundary.clone(),
                                ),
                            )?)?;
                            boundary
                        }
                    };
                    durable.validate(&expected, publisher.identity(), config)?;
                    let transaction_id = durable.transaction_id.clone();
                    let boundary_clock_before = durable.clock_before.clone();
                    let boundary_request = PostTrainingBoundaryRequest {
                        transaction_id: &transaction_id,
                        workflow_signature,
                        phase_name: &phase.name,
                        clock_before: &boundary_clock_before,
                        clock_after,
                        config,
                        input: &published_state,
                    };
                    let resume_native = durable.native_sleep.clone();
                    let mut bridge = PostTrainingSleepProgressBridge {
                        inner: progress,
                        request,
                        cursor: &cursor,
                        boundary: &mut durable,
                    };
                    let boundary =
                        hook.drive_boundary(&boundary_request, resume_native, &mut bridge)?;
                    let final_native = durable.native_sleep.as_ref().context(
                        "post-training sleep hook returned without persisting its native cursor",
                    )?;
                    ensure!(
                        canonical_sha256(final_native)? == boundary.native_sleep_checkpoint_sha256,
                        "post-training sleep receipt does not bind the persisted native cursor"
                    );
                    boundary.validate(
                        &transaction_id,
                        hook.identity(),
                        &published_state,
                        clock_after,
                    )?;
                    (boundary.state.clone(), Some(boundary))
                }
                (None, None) => {
                    ensure!(
                        in_flight.is_none(),
                        "ordinary post-training phase resumed a periodic boundary"
                    );
                    (published_state, None)
                }
                _ => unreachable!("boundary-hook presence validated before execution"),
            };
        // Publication is immutable, so a late input replacement can leave an
        // unreferenced artifact but must not advance the durable phase cursor
        // or become its final product.
        batches.ensure_still_published()?;

        let next_chain = canonical_sha256(&(
            "post_training_receipt_chain_v1",
            &cursor.receipt_chain_sha256,
            &receipt.receipt_sha256,
            boundary_receipt
                .as_ref()
                .map(|receipt| receipt.receipt_sha256.as_str()),
        ))?;
        let mut next = cursor.clone();
        next.position = batch.end;
        next.rng.counter = expected.rng_end;
        next.committed = committed;
        next.receipt_chain_sha256 = next_chain;
        next.last_update_receipt = Some(receipt.clone());
        next.last_boundary_receipt = boundary_receipt;
        next.clock = next
            .last_boundary_receipt
            .as_ref()
            .map(|receipt| receipt.clock.clone());
        next.pending = None;
        next.progress.apply(&expected.summary)?;
        next.validate_internal()?;

        let metric = update_metric(&expected, &receipt)?;
        metric.validate()?;
        progress.metric(metric_context(request, &next)?, metric)?;
        batches.ensure_still_published()?;
        progress.checkpoint(serde_json::to_value(PostTrainingResumeEnvelope::wake(
            next.clone(),
        ))?)?;
        batches.ensure_still_published()?;
        cursor = next;
        updates_this_drive += 1;

        if phase_is_complete(phase, &cursor)? {
            let report = cursor.progress.report(
                phase,
                post_training_algorithm_name(
                    phase
                        .post_training
                        .as_ref()
                        .expect("validated post-training config"),
                ),
                &cursor.runtime_identities,
            )?;
            return Ok(ResumablePostTrainingOutcome::Complete { cursor, report });
        }
        if updates_this_drive >= update_budget {
            return Ok(ResumablePostTrainingOutcome::Yielded(cursor));
        }
    }
}

fn boundary_transaction_id(
    update: &PreparedPostTrainingUpdate,
    receipt: &PostTrainingUpdateReceipt,
    config: &InModelSleepConfig,
    clock_before: &PostTrainingClockReceipt,
    clock_after: PostTrainingClockValues,
) -> Result<String> {
    canonical_sha256(&(
        "post_training_periodic_sleep_v2",
        &update.transaction_id,
        &receipt.receipt_sha256,
        clock_before,
        clock_after,
        config,
    ))
}

struct PostTrainingSleepProgressBridge<'a> {
    inner: &'a mut dyn PhaseProgressSink,
    request: &'a PhaseExecutionRequest,
    cursor: &'a PostTrainingCursor,
    boundary: &'a mut PostTrainingBoundaryInFlight,
}

impl PostTrainingSleepProgressBridge<'_> {
    fn persist_wrapped(&mut self, checkpoint: &NativeSleepCheckpoint) -> Result<()> {
        self.boundary.native_sleep = Some(checkpoint.clone());
        self.inner
            .checkpoint(serde_json::to_value(PostTrainingResumeEnvelope::boundary(
                self.cursor.clone(),
                self.boundary.clone(),
            ))?)
    }
}

impl NativeSleepProgressSink for PostTrainingSleepProgressBridge<'_> {
    fn persist(&mut self, checkpoint: &NativeSleepCheckpoint) -> Result<()> {
        self.persist_wrapped(checkpoint)
    }

    fn metric(&mut self, checkpoint: &NativeSleepCheckpoint, event: MetricEvent) -> Result<()> {
        event.validate()?;
        self.inner.metric(
            MetricContext {
                global_step: self.boundary.clock_after.optimizer_steps,
                phase: MetricPhase {
                    index: self
                        .request
                        .phase_index
                        .try_into()
                        .context("post-training phase index exceeds metric schema")?,
                    name: self.request.phase.name.clone(),
                    kind: MetricPhaseKind::Sleep,
                },
                checkpoint_hash: Some(checkpoint.live_checkpoint.sha256.clone()),
            },
            event,
        )?;
        // Commit the metric prefix only alongside the composite outer/inner
        // cursor which produced it.
        self.persist_wrapped(checkpoint)
    }
}

/// Validate the exact phase surface consumed by the built-in native
/// post-training executor. Native host dispatch calls this before creating a
/// runtime checkpoint so a syntactically valid WorkflowV2 phase cannot fail
/// only after durable state has already been published.
pub(crate) fn validate_resumable_phase(phase: &PhaseV2) -> Result<()> {
    ensure!(
        !phase.name.trim().is_empty(),
        "native post-training phase name must not be empty"
    );
    ensure!(
        matches!(
            phase.kind,
            PhaseKind::Preference | PhaseKind::Distillation | PhaseKind::Rl
        ),
        "native post-training executor received unsupported `{}` phase",
        phase.kind.name()
    );
    ensure!(
        phase.shuffle_buffer.is_none(),
        "native post-training requires deterministic input order"
    );
    ensure!(
        phase.parameters.is_empty(),
        "native post-training rejects unconsumed raw parameters"
    );
    ensure!(
        phase.memory_update_mode.is_none(),
        "native post-training does not implement memory_update_mode; use periodic_sleep or a tier-aware phase executor"
    );
    let task = phase
        .task
        .as_ref()
        .context("native post-training phase has no task")?;
    task.validate()?;
    let execution = task.contract().execution;
    match phase.kind {
        PhaseKind::Preference => ensure!(
            execution == TaskExecution::PairwisePreference,
            "native DPO requires a pairwise-preference task"
        ),
        PhaseKind::Distillation => ensure!(
            matches!(
                execution,
                TaskExecution::AutoregressiveTokenPrediction | TaskExecution::SupervisedGeneration
            ),
            "native forward-KL requires an autoregressive or supervised-generation task"
        ),
        PhaseKind::Rl => ensure!(
            execution == TaskExecution::VerifiableReward,
            "native GRPO requires a verifiable-reward task"
        ),
        _ => unreachable!("phase kind was restricted above"),
    }
    let data = phase
        .data
        .as_ref()
        .context("native post-training phase has no data")?;
    ensure!(
        !data.as_os_str().is_empty(),
        "native post-training phase has an empty data path"
    );
    let sequence_length = phase
        .sequence_length
        .context("native post-training phase has no sequence_length")?;
    ensure!(
        (1..=MAX_POST_TRAINING_SEQUENCE_TOKENS).contains(&sequence_length),
        "native post-training sequence_length must be between 1 and {MAX_POST_TRAINING_SEQUENCE_TOKENS}"
    );
    ensure!(
        phase.epochs_or_default() > 0,
        "native post-training epochs must be positive"
    );
    ensure!(
        phase.steps.is_none_or(|steps| steps > 0),
        "native post-training steps must be positive when set"
    );
    let loss_weight = phase.loss_weight_or_default();
    ensure!(
        loss_weight.is_finite() && loss_weight > 0.0,
        "native post-training loss_weight must be finite and positive"
    );
    let learning_rate_scale = phase.learning_rate_scale_or_default();
    ensure!(
        learning_rate_scale.is_finite() && learning_rate_scale > 0.0,
        "native post-training learning_rate_scale must be finite and positive"
    );
    let update_examples = update_examples(phase)?;
    let config = phase
        .post_training
        .as_ref()
        .context("native post-training phase has no typed algorithm")?;
    config.validate()?;
    match (phase.kind, config) {
        (PhaseKind::Preference, PostTrainingConfig::Dpo { .. })
        | (PhaseKind::Distillation, PostTrainingConfig::ForwardKl { .. })
        | (PhaseKind::Rl, PostTrainingConfig::Grpo { .. }) => {}
        _ => bail!("native post-training phase kind and algorithm do not match"),
    }
    if let PostTrainingConfig::Grpo { sampling, .. } = config {
        ensure!(
            sampling.max_new_tokens <= phase.sequence_length.expect("checked present"),
            "GRPO max_new_tokens exceeds sequence_length"
        );
    }
    if let PostTrainingConfig::Grpo {
        group_size,
        sampling,
        ..
    } = config
    {
        let update_tokens = group_size
            .checked_mul(sampling.max_new_tokens)
            .and_then(|tokens| tokens.checked_mul(update_examples))
            .context("GRPO rollout-update token geometry overflows usize")?;
        ensure!(
            update_tokens <= MAX_POST_TRAINING_ROLLOUT_TOKENS,
            "GRPO rollout update exceeds the {MAX_POST_TRAINING_ROLLOUT_TOKENS}-token native limit"
        );
    }
    Ok(())
}

fn update_examples(phase: &PhaseV2) -> Result<usize> {
    let examples = phase
        .batch_size
        .context("native post-training phase has no batch_size")?
        .checked_mul(
            phase
                .gradient_accumulation
                .context("native post-training phase has no gradient_accumulation")?,
        )
        .context("native post-training update example count overflows usize")?;
    ensure!(examples > 0, "native post-training update batch is empty");
    ensure!(
        examples <= MAX_POST_TRAINING_UPDATE_EXAMPLES,
        "native post-training update has {examples} examples, exceeding the limit of {MAX_POST_TRAINING_UPDATE_EXAMPLES}"
    );
    Ok(examples)
}

fn phase_is_complete(phase: &PhaseV2, cursor: &PostTrainingCursor) -> Result<bool> {
    let step_limit = phase
        .steps
        .map(u64::try_from)
        .transpose()
        .context("post-training step limit exceeds u64")?;
    if step_limit.is_some_and(|steps| cursor.progress.optimizer_steps >= steps) {
        return Ok(true);
    }
    Ok(cursor.position.epoch
        >= u64::try_from(phase.epochs_or_default()).context("epoch count exceeds u64")?)
}

#[derive(Debug)]
struct CollectedPostTrainingBatch {
    examples: Vec<TaskExample>,
    end: PostTrainingRecordPosition,
}

#[derive(Debug)]
struct RawPostTrainingRecord {
    line: String,
    line_number: usize,
}

fn read_post_training_record_bounded(
    reader: &mut (impl BufRead + ?Sized),
    output: &mut Vec<u8>,
    maximum_bytes: usize,
) -> Result<usize> {
    ensure!(
        maximum_bytes > 0,
        "post-training record byte limit must be positive"
    );
    output.clear();
    let capture_bytes = maximum_bytes
        .checked_add(1)
        .context("post-training record byte limit overflows usize")?;
    let read = reader
        .take(u64::try_from(capture_bytes).context("post-training record byte limit exceeds u64")?)
        .read_until(b'\n', output)
        .context("failed to read post-training record")?;
    let payload_bytes = output
        .len()
        .checked_sub(usize::from(output.last() == Some(&b'\n')))
        .context("post-training record framing underflow")?;
    ensure!(
        payload_bytes <= maximum_bytes,
        "post-training record exceeds the maximum of {maximum_bytes} bytes"
    );
    Ok(read)
}

/// Sequential batch reader over one authenticated file description. Resume
/// performs at most one prefix scan to the durable record cursor; ordinary
/// updates continue from the existing buffered reader instead of reopening
/// and rescanning the file from record zero.
struct PostTrainingBatchStream {
    source: AuthenticatedPostTrainingInput,
    task: TaskConfig,
    reader: Box<dyn BufRead>,
    jsonl: bool,
    line: Vec<u8>,
    line_number: usize,
    epoch: u64,
    record: u64,
    records_in_epoch: u64,
    epochs: u64,
    lookahead: Option<RawPostTrainingRecord>,
}

impl PostTrainingBatchStream {
    fn new(
        mut source: AuthenticatedPostTrainingInput,
        task: TaskConfig,
        start: PostTrainingRecordPosition,
        epochs: u64,
    ) -> Result<Self> {
        ensure!(
            start.epoch < epochs || (start.epoch == epochs && start.record == 0),
            "post-training input cursor exceeds the configured epoch range"
        );
        task.validate()?;
        let jsonl =
            task.contract().data_format == TaskDataFormat::Jsonl || is_jsonl_path(&source.path);
        ensure!(
            task.contract().data_format != TaskDataFormat::Jsonl || jsonl,
            "task `{}` requires JSONL framing",
            task.name()
        );
        let reader = source.reader()?;
        let mut stream = Self {
            source,
            task,
            reader,
            jsonl,
            line: Vec::new(),
            line_number: 0,
            epoch: start.epoch,
            record: 0,
            records_in_epoch: 0,
            epochs,
            lookahead: None,
        };
        while stream.record < start.record {
            ensure!(
                stream.read_record()?.is_some(),
                "post-training cursor record {} exceeds epoch {} length {}",
                start.record,
                start.epoch,
                stream.records_in_epoch
            );
            stream.record = stream
                .record
                .checked_add(1)
                .context("post-training record cursor overflows u64")?;
        }
        stream.source.ensure_still_published()?;
        Ok(stream)
    }

    fn read_raw_record(&mut self) -> Result<Option<RawPostTrainingRecord>> {
        loop {
            let read = read_post_training_record_bounded(
                &mut self.reader,
                &mut self.line,
                MAX_POST_TRAINING_RECORD_BYTES,
            )
            .with_context(|| {
                format!(
                    "failed to read task data {}:{}",
                    self.source.path.display(),
                    self.line_number.saturating_add(1)
                )
            })?;
            if read == 0 {
                return Ok(None);
            }
            self.line_number = self
                .line_number
                .checked_add(1)
                .context("task-data line counter overflows usize")?;
            let line = String::from_utf8(std::mem::take(&mut self.line)).with_context(|| {
                format!(
                    "task data is not UTF-8 at {}:{}",
                    self.source.path.display(),
                    self.line_number
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }
            return Ok(Some(RawPostTrainingRecord {
                line,
                line_number: self.line_number,
            }));
        }
    }

    fn construct_record(&mut self, raw: RawPostTrainingRecord) -> Result<(TaskExample, usize)> {
        let input_bytes = raw.line.len();
        let record = if self.jsonl {
            serde_json::from_str(&raw.line).with_context(|| {
                format!(
                    "invalid JSON task record at {}:{}",
                    self.source.path.display(),
                    raw.line_number
                )
            })?
        } else {
            serde_json::json!({"text": raw.line.trim_end_matches(['\r', '\n'])})
        };
        let example = self.task.construct_example(&record).with_context(|| {
            format!(
                "invalid task record at {}:{}",
                self.source.path.display(),
                raw.line_number
            )
        })?;
        self.records_in_epoch = self
            .records_in_epoch
            .checked_add(1)
            .context("post-training per-epoch record counter overflows u64")?;
        Ok((example, input_bytes))
    }

    fn read_record(&mut self) -> Result<Option<(TaskExample, usize)>> {
        self.read_raw_record()?
            .map(|raw| self.construct_record(raw))
            .transpose()
    }

    fn ensure_still_published(&self) -> Result<()> {
        self.source.ensure_still_published()
    }

    fn advance_epoch(&mut self) -> Result<()> {
        ensure!(
            self.records_in_epoch > 0,
            "task data {} contains no examples",
            self.source.path.display()
        );
        self.source.ensure_still_published()?;
        self.epoch = self
            .epoch
            .checked_add(1)
            .context("post-training epoch cursor overflows u64")?;
        self.record = 0;
        self.records_in_epoch = 0;
        self.line_number = 0;
        if self.epoch < self.epochs {
            // Drop the cloned handle before rewinding the retained base file;
            // both descriptors intentionally refer to the same open file.
            self.reader = Box::new(std::io::empty());
            self.reader = self.source.reader()?;
        }
        Ok(())
    }

    fn next_batch(&mut self, target: usize) -> Result<CollectedPostTrainingBatch> {
        ensure!(target > 0, "post-training target batch is empty");
        ensure!(
            target <= MAX_POST_TRAINING_UPDATE_EXAMPLES,
            "post-training target batch exceeds {MAX_POST_TRAINING_UPDATE_EXAMPLES} examples"
        );
        self.source.ensure_still_published()?;
        let mut examples = Vec::with_capacity(target);
        let mut input_bytes = 0usize;
        while self.epoch < self.epochs && examples.len() < target {
            let raw = match self.lookahead.take() {
                Some(raw) => Some(raw),
                None => self.read_raw_record()?,
            };
            match raw {
                Some(raw) => {
                    let next_input_bytes = input_bytes
                        .checked_add(raw.line.len())
                        .context("post-training update input byte count overflows usize")?;
                    ensure!(
                        next_input_bytes <= MAX_POST_TRAINING_UPDATE_INPUT_BYTES,
                        "post-training update input exceeds the {MAX_POST_TRAINING_UPDATE_INPUT_BYTES}-byte limit"
                    );
                    let (example, record_bytes) = self.construct_record(raw)?;
                    input_bytes = input_bytes
                        .checked_add(record_bytes)
                        .context("post-training update input byte count overflows usize")?;
                    debug_assert_eq!(input_bytes, next_input_bytes);
                    examples.push(example);
                    self.record = self
                        .record
                        .checked_add(1)
                        .context("post-training record cursor overflows u64")?;
                }
                None => self.advance_epoch()?,
            }
        }

        // Resolve the exact-EOF case now so a batch ending on the last record
        // durably records the next epoch boundary instead of requiring an
        // empty follow-up update attempt. Preserve one framed but unparsed
        // record as lookahead when more data remains in this epoch: a phase
        // ending at its step limit must not validate data it never consumes.
        if examples.len() == target && self.epoch < self.epochs {
            match self.read_raw_record()? {
                Some(raw) => self.lookahead = Some(raw),
                None => self.advance_epoch()?,
            }
        }
        self.source.ensure_still_published()?;
        Ok(CollectedPostTrainingBatch {
            examples,
            end: PostTrainingRecordPosition {
                epoch: self.epoch,
                record: self.record,
            },
        })
    }
}

#[derive(Debug)]
struct PreparedComputation {
    summary: PostTrainingUpdateSummary,
    rng_end: u64,
}

fn validate_and_capture_runtime_identities(
    phase: &PhaseV2,
    runtime: &mut PostTrainingPhaseRuntime<'_>,
) -> Result<PostTrainingRuntimeIdentities> {
    let config = phase
        .post_training
        .as_ref()
        .context("post-training phase has no typed algorithm")?;
    match (config, runtime) {
        (
            PostTrainingConfig::Dpo {
                reference: spec, ..
            },
            PostTrainingPhaseRuntime::Dpo { runtime },
        ) => {
            validate_frozen_runtime(
                spec,
                runtime.frozen_identity(),
                runtime.frozen_tokenizer_identity(),
                runtime.trainable_tokenizer_identity(),
            )?;
            validate_runtime_identity(runtime.trainable_identity(), "trainable DPO policy")?;
            validate_runtime_identity(runtime.trainable_tokenizer_identity(), "DPO tokenizer")?;
            Ok(PostTrainingRuntimeIdentities {
                trainable: runtime.trainable_identity().to_owned(),
                tokenizer: runtime.trainable_tokenizer_identity().to_owned(),
                frozen: Some(runtime.frozen_identity().to_owned()),
                verifier: None,
            })
        }
        (
            PostTrainingConfig::ForwardKl { teacher: spec, .. },
            PostTrainingPhaseRuntime::ForwardKl { runtime },
        ) => {
            validate_frozen_runtime(
                spec,
                runtime.frozen_identity(),
                runtime.frozen_tokenizer_identity(),
                runtime.trainable_tokenizer_identity(),
            )?;
            validate_runtime_identity(
                runtime.trainable_identity(),
                "trainable distillation policy",
            )?;
            validate_runtime_identity(
                runtime.trainable_tokenizer_identity(),
                "distillation tokenizer",
            )?;
            Ok(PostTrainingRuntimeIdentities {
                trainable: runtime.trainable_identity().to_owned(),
                tokenizer: runtime.trainable_tokenizer_identity().to_owned(),
                frozen: Some(runtime.frozen_identity().to_owned()),
                verifier: None,
            })
        }
        (
            PostTrainingConfig::Grpo {
                reference: reference_spec,
                kl_coefficient,
                ..
            },
            PostTrainingPhaseRuntime::Grpo { runtime, verifier },
        ) => {
            ensure!(
                runtime.frozen_identity().is_some()
                    == runtime.frozen_tokenizer_identity().is_some(),
                "GRPO device runtime has a partial frozen-reference identity"
            );
            match (reference_spec.as_ref(), runtime.frozen_identity()) {
                (Some(spec), Some(identity)) => validate_frozen_runtime(
                    spec,
                    identity,
                    runtime
                        .frozen_tokenizer_identity()
                        .expect("checked frozen tokenizer present"),
                    runtime.trainable_tokenizer_identity(),
                )?,
                (Some(_), None) => bail!("GRPO phase requires its frozen reference runtime"),
                (None, Some(_)) => bail!("GRPO phase supplied an unconfigured reference runtime"),
                (None, None) => ensure!(
                    *kl_coefficient == 0.0,
                    "GRPO without a reference requires zero KL coefficient"
                ),
            }
            validate_runtime_identity(runtime.trainable_identity(), "trainable GRPO policy")?;
            validate_runtime_identity(runtime.trainable_tokenizer_identity(), "GRPO tokenizer")?;
            validate_sha256_identity(verifier.identity(), "GRPO verifier implementation")?;
            let TaskConfig::VerifiableRl {
                verifier: verifier_spec,
            } = phase.task.as_ref().expect("validated post-training task")
            else {
                bail!("GRPO phase requires a verifiable-RL task");
            };
            ensure!(
                verifier.adapter_name() == verifier_spec.adapter,
                "verifier adapter mismatch: task requires `{}`, executor supplied `{}`",
                verifier_spec.adapter,
                verifier.adapter_name()
            );
            Ok(PostTrainingRuntimeIdentities {
                trainable: runtime.trainable_identity().to_owned(),
                tokenizer: runtime.trainable_tokenizer_identity().to_owned(),
                frozen: runtime.frozen_identity().map(str::to_owned),
                verifier: Some(verifier.identity().to_owned()),
            })
        }
        _ => bail!("post-training algorithm and device runtime do not match"),
    }
}

fn optimizer_step(
    runtime: &mut PostTrainingPhaseRuntime<'_>,
    learning_rate_scale: f64,
) -> Result<()> {
    match runtime {
        PostTrainingPhaseRuntime::Dpo { runtime } => runtime.optimizer_step(learning_rate_scale),
        PostTrainingPhaseRuntime::ForwardKl { runtime } => {
            runtime.optimizer_step(learning_rate_scale)
        }
        PostTrainingPhaseRuntime::Grpo { runtime, .. } => {
            runtime.optimizer_step(learning_rate_scale)
        }
    }
}

fn runtime_model_tokens_processed(runtime: &PostTrainingPhaseRuntime<'_>) -> u64 {
    match runtime {
        PostTrainingPhaseRuntime::Dpo { runtime } => runtime.model_tokens_processed(),
        PostTrainingPhaseRuntime::ForwardKl { runtime } => runtime.model_tokens_processed(),
        PostTrainingPhaseRuntime::Grpo { runtime, .. } => runtime.model_tokens_processed(),
    }
}

fn exact_model_token_delta(start: u64, end: u64) -> Result<u64> {
    let delta = end
        .checked_sub(start)
        .context("trainable-policy model-token counter regressed during an update")?;
    ensure!(
        delta > 0,
        "trainable device runtime reported zero model tokens for a non-empty update"
    );
    Ok(delta)
}

fn accumulate_resumable_update(
    phase: &PhaseV2,
    examples: &[TaskExample],
    batch_sha256: &str,
    base_state: &PostTrainingCommittedState,
    runtime: &mut PostTrainingPhaseRuntime<'_>,
    rng_seed: u64,
    rng_start: u64,
) -> Result<PreparedComputation> {
    ensure!(
        !examples.is_empty() && examples.len() <= MAX_POST_TRAINING_UPDATE_EXAMPLES,
        "post-training prepared update example geometry is invalid"
    );
    let model_tokens_start = runtime_model_tokens_processed(runtime);
    let sequence_length = phase.sequence_length.expect("validated sequence length");
    ensure!(
        sequence_length <= MAX_POST_TRAINING_SEQUENCE_TOKENS,
        "post-training sequence length exceeds the native limit"
    );
    validate_sha256_identity(batch_sha256, "post-training batch")?;
    base_state.validate()?;
    let base = DeviceBatchBaseGeneration {
        model_sha256: base_state.model.sha256(),
        optimizer_sha256: base_state.optimizer.as_ref().map(ImmutableArtifact::sha256),
    };
    base.validate()?;
    let loss_weight = phase.loss_weight_or_default();
    ensure!(
        loss_weight.is_finite() && loss_weight > 0.0,
        "post-training loss weight must be finite and positive"
    );
    let example_count = u64::try_from(examples.len()).context("batch size exceeds u64")?;
    let config = phase
        .post_training
        .as_ref()
        .expect("validated post-training config");
    match (config, runtime) {
        (
            PostTrainingConfig::Dpo {
                beta,
                label_smoothing,
                sequence_reduction,
                ..
            },
            PostTrainingPhaseRuntime::Dpo { runtime },
        ) => {
            for example in examples {
                if !matches!(example, TaskExample::PairwisePreference { .. }) {
                    bail!("DPO executor received a non-pairwise task example");
                }
            }
            let rng = ModelExecutionRngRange::reserve(
                rng_seed,
                rng_start,
                example_count
                    .checked_mul(2)
                    .context("DPO model RNG range overflows u64")?,
            )?;
            let result = runtime.prepare_device_batch(DpoDeviceBatchRequest {
                examples,
                batch_sha256,
                base,
                max_sequence_tokens: sequence_length,
                beta: *beta,
                label_smoothing: *label_smoothing,
                sequence_reduction: *sequence_reduction,
                loss_weight,
                rng,
            })?;
            result.receipt.validate()?;
            ensure!(
                result.examples == example_count,
                "DPO device runtime returned a partial batch summary"
            );
            let model_tokens =
                exact_model_token_delta(model_tokens_start, runtime.model_tokens_processed())?;
            ensure!(
                result.receipt.model_tokens == model_tokens,
                "DPO device receipt disagrees with the runtime model-token counter"
            );
            let execution_sha256 = canonical_sha256(&(
                "dpo_device_batch_v1",
                batch_sha256,
                base,
                sequence_length,
                beta,
                label_smoothing,
                sequence_reduction,
                loss_weight,
                rng,
                &result,
            ))?;
            let summary = PostTrainingUpdateSummary {
                examples: example_count,
                model_tokens,
                loss_sum: result.loss_sum,
                execution_sha256,
                objective: PostTrainingObjectiveSummary::Dpo {
                    preference_correct: result.preference_correct,
                    implicit_reward_margin_sum: result.implicit_reward_margin_sum,
                },
            };
            summary.validate()?;
            Ok(PreparedComputation {
                summary,
                rng_end: rng.end,
            })
        }
        (
            PostTrainingConfig::ForwardKl {
                temperature,
                scale_by_temperature_squared,
                ..
            },
            PostTrainingPhaseRuntime::ForwardKl { runtime },
        ) => {
            for example in examples {
                if !matches!(
                    example,
                    TaskExample::Autoregressive { .. } | TaskExample::SupervisedGeneration { .. }
                ) {
                    bail!("forward-KL executor received an incompatible task example");
                }
            }
            let rng = ModelExecutionRngRange::reserve(rng_seed, rng_start, example_count)?;
            let result = runtime.prepare_device_batch(ForwardKlDeviceBatchRequest {
                examples,
                batch_sha256,
                base,
                max_sequence_tokens: sequence_length,
                temperature: *temperature,
                scale_by_temperature_squared: *scale_by_temperature_squared,
                loss_weight,
                rng,
            })?;
            result.receipt.validate()?;
            ensure!(
                result.examples == example_count,
                "forward-KL device runtime returned a partial batch summary"
            );
            let model_tokens =
                exact_model_token_delta(model_tokens_start, runtime.model_tokens_processed())?;
            ensure!(
                result.receipt.model_tokens == model_tokens,
                "forward-KL device receipt disagrees with the runtime model-token counter"
            );
            let execution_sha256 = canonical_sha256(&(
                "forward_kl_device_batch_v1",
                batch_sha256,
                base,
                sequence_length,
                temperature,
                scale_by_temperature_squared,
                loss_weight,
                rng,
                &result,
            ))?;
            let summary = PostTrainingUpdateSummary {
                examples: example_count,
                model_tokens,
                loss_sum: result.loss_sum,
                execution_sha256,
                objective: PostTrainingObjectiveSummary::ForwardKl {
                    forward_kl_sum: result.forward_kl_sum,
                    teacher_entropy_sum: result.teacher_entropy_sum,
                    top1_agreement_sum: result.top1_agreement_sum,
                },
            };
            summary.validate()?;
            Ok(PreparedComputation {
                summary,
                rng_end: rng.end,
            })
        }
        (
            PostTrainingConfig::Grpo {
                group_size,
                clip_epsilon,
                advantage_epsilon,
                kl_coefficient,
                sampling,
                ..
            },
            PostTrainingPhaseRuntime::Grpo { runtime, verifier },
        ) => {
            let task = phase.task.as_ref().expect("validated task");
            for example in examples {
                if !matches!(example, TaskExample::VerifiableRollout { .. }) {
                    bail!("GRPO executor received a non-verifiable task example");
                }
            }
            let group_count = u64::try_from(*group_size).context("GRPO group exceeds u64")?;
            let rollouts = example_count
                .checked_mul(group_count)
                .context("GRPO rollout count overflows u64")?;
            let generation_rng = ModelExecutionRngRange::reserve(rng_seed, rng_start, rollouts)?;
            let scoring_rng =
                ModelExecutionRngRange::reserve(rng_seed, generation_rng.end, rollouts)?;
            let generated = runtime.generate_device_batch(GrpoGenerationBatchRequest {
                examples,
                batch_sha256,
                base,
                group_size: *group_size,
                max_sequence_tokens: sequence_length,
                sampling,
                rng: generation_rng,
            })?;
            validate_sha256_identity(&generated.generation_sha256, "GRPO device generation")?;
            ensure!(
                generated.groups.len() == examples.len(),
                "GRPO device runtime returned {} prompt groups, expected {}",
                generated.groups.len(),
                examples.len()
            );
            let mut rewards = Vec::with_capacity(examples.len());
            let mut rollout_tokens = 0usize;
            let mut rollout_text_bytes = 0usize;
            for (example_index, (example, group)) in
                examples.iter().zip(&generated.groups).enumerate()
            {
                ensure!(
                    group.len() == *group_size,
                    "GRPO device runtime returned {} rollouts for prompt {example_index}, expected {group_size}",
                    group.len()
                );
                let mut group_rewards = Vec::with_capacity(*group_size);
                for (rollout_index, rollout) in group.iter().enumerate() {
                    ensure!(
                        !rollout.text.trim().is_empty(),
                        "rollout {example_index}:{rollout_index} completion is empty"
                    );
                    ensure!(
                        rollout.token_count > 0
                            && rollout.token_count <= sampling.max_new_tokens
                            && rollout.token_count <= sequence_length,
                        "rollout {example_index}:{rollout_index} exceeds its configured token limit"
                    );
                    rollout_tokens = rollout_tokens
                        .checked_add(rollout.token_count)
                        .context("GRPO rollout token count overflows usize")?;
                    ensure!(
                        rollout_tokens <= MAX_POST_TRAINING_ROLLOUT_TOKENS,
                        "GRPO rollout group exceeds the native token limit"
                    );
                    rollout_text_bytes = rollout_text_bytes
                        .checked_add(rollout.text.len())
                        .context("GRPO rollout text byte count overflows usize")?;
                    ensure!(
                        rollout_text_bytes <= MAX_POST_TRAINING_UPDATE_INPUT_BYTES,
                        "GRPO rollout text exceeds the native byte limit"
                    );
                    group_rewards
                        .push(verify_rollout(task, example, *verifier, &rollout.text)?.reward);
                }
                rewards.push(group_rewards);
            }
            let result = runtime.prepare_device_batch(GrpoDeviceBatchRequest {
                examples,
                batch_sha256,
                base,
                generation: &generated,
                rewards: &rewards,
                clip_epsilon: *clip_epsilon,
                advantage_epsilon: *advantage_epsilon,
                kl_coefficient: *kl_coefficient,
                loss_weight,
                rng: scoring_rng,
            })?;
            result.receipt.validate()?;
            ensure!(
                result.examples == example_count && result.rollouts == rollouts,
                "GRPO device runtime returned a partial batch summary"
            );
            let model_tokens =
                exact_model_token_delta(model_tokens_start, runtime.model_tokens_processed())?;
            ensure!(
                result.receipt.model_tokens == model_tokens,
                "GRPO device receipt disagrees with the runtime model-token counter"
            );
            let verification_sha256 = canonical_sha256(&(&generated, &rewards))?;
            let execution_sha256 = canonical_sha256(&(
                "grpo_device_batch_v1",
                batch_sha256,
                base,
                sequence_length,
                group_size,
                clip_epsilon,
                advantage_epsilon,
                kl_coefficient,
                sampling,
                loss_weight,
                generation_rng,
                scoring_rng,
                verification_sha256,
                &result,
            ))?;
            let summary = PostTrainingUpdateSummary {
                examples: example_count,
                model_tokens,
                loss_sum: result.loss_sum,
                execution_sha256,
                objective: PostTrainingObjectiveSummary::Grpo {
                    mean_reward_sum: result.mean_reward_sum,
                    reward_stddev_sum: result.reward_stddev_sum,
                    mean_kl_sum: result.mean_kl_sum,
                    clipped_fraction_sum: result.clipped_fraction_sum,
                },
            };
            summary.validate()?;
            Ok(PreparedComputation {
                summary,
                rng_end: scoring_rng.end,
            })
        }
        _ => bail!("post-training algorithm and device runtime do not match"),
    }
}

fn validate_restore_receipt(
    receipt: &PostTrainingRestoreReceipt,
    publisher_identity: &str,
    state: &PostTrainingCommittedState,
) -> Result<()> {
    state.validate()?;
    ensure!(
        receipt.publisher_identity == publisher_identity
            && receipt.model_sha256 == state.model.sha256()
            && receipt.optimizer_sha256
                == state
                    .optimizer
                    .as_ref()
                    .map(|artifact| artifact.sha256().to_owned()),
        "post-training publisher restored a different model/optimizer generation"
    );
    Ok(())
}

fn update_metric(
    plan: &PreparedPostTrainingUpdate,
    receipt: &PostTrainingUpdateReceipt,
) -> Result<MetricEvent> {
    let count = plan.summary.examples as f64;
    let mut metric = PostTrainingUpdateMetrics {
        transaction_id: plan.transaction_id.clone(),
        algorithm: plan.summary.objective.algorithm(),
        epoch: plan.start.epoch,
        first_record: plan.start.record,
        records: plan.summary.examples,
        optimizer_step: plan.update_index + 1,
        rng_counter_start: plan.rng_start,
        rng_counter_end: plan.rng_end,
        loss: plan.summary.loss_sum / count,
        checkpoint_sha256: receipt.checkpoint.sha256().to_owned(),
        optimizer_sha256: receipt.optimizer.sha256().to_owned(),
        preference_accuracy: None,
        implicit_reward_margin: None,
        forward_kl: None,
        teacher_entropy: None,
        top1_agreement: None,
        mean_reward: None,
        reward_stddev: None,
        mean_kl: None,
        clipped_fraction: None,
    };
    match &plan.summary.objective {
        PostTrainingObjectiveSummary::Dpo {
            preference_correct,
            implicit_reward_margin_sum,
        } => {
            metric.preference_accuracy = Some(*preference_correct as f64 / count);
            metric.implicit_reward_margin = Some(*implicit_reward_margin_sum / count);
        }
        PostTrainingObjectiveSummary::ForwardKl {
            forward_kl_sum,
            teacher_entropy_sum,
            top1_agreement_sum,
        } => {
            metric.forward_kl = Some(*forward_kl_sum / count);
            metric.teacher_entropy = Some(*teacher_entropy_sum / count);
            metric.top1_agreement = Some(*top1_agreement_sum / count);
        }
        PostTrainingObjectiveSummary::Grpo {
            mean_reward_sum,
            reward_stddev_sum,
            mean_kl_sum,
            clipped_fraction_sum,
        } => {
            metric.mean_reward = Some(*mean_reward_sum / count);
            metric.reward_stddev = Some(*reward_stddev_sum / count);
            metric.mean_kl = Some(*mean_kl_sum / count);
            metric.clipped_fraction = Some(*clipped_fraction_sum / count);
        }
    }
    let event = MetricEvent::PostTrainingUpdate(Box::new(metric));
    event.validate()?;
    Ok(event)
}

fn metric_context(
    request: &PhaseExecutionRequest,
    cursor: &PostTrainingCursor,
) -> Result<MetricContext> {
    ensure!(
        matches!(
            request.phase.kind,
            PhaseKind::Preference | PhaseKind::Distillation | PhaseKind::Rl
        ),
        "unsupported post-training metric phase"
    );
    Ok(MetricContext {
        global_step: cursor
            .clock
            .as_ref()
            .map_or(cursor.progress.optimizer_steps, |clock| {
                clock.optimizer_steps
            }),
        phase: MetricPhase {
            index: u32::try_from(request.phase_index).context("phase index exceeds u32")?,
            name: request.phase.name.clone(),
            kind: request.phase.kind.into(),
        },
        checkpoint_hash: Some(cursor.committed.model.sha256().to_owned()),
    })
}

fn validate_runtime_identity(identity: &str, name: &str) -> Result<()> {
    ensure!(
        !identity.trim().is_empty() && !identity.contains(['\n', '\r']),
        "{name} identity must be non-empty and single-line"
    );
    Ok(())
}

struct JsonDigestWriter {
    digest: Sha256,
    bytes: u64,
}

impl JsonDigestWriter {
    fn new() -> Self {
        Self {
            digest: Sha256::new(),
            bytes: 0,
        }
    }

    fn finish(self) -> (u64, String) {
        (self.bytes, format!("{:x}", self.digest.finalize()))
    }
}

impl Write for JsonDigestWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(u64::try_from(buffer.len()).map_err(std::io::Error::other)?)
            .ok_or_else(|| std::io::Error::other("serialized JSON length overflows u64"))?;
        self.digest.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialized_json_digest<T: Serialize + ?Sized>(
    value: &T,
    context: &'static str,
) -> Result<(u64, String)> {
    let mut writer = JsonDigestWriter::new();
    serde_json::to_writer(&mut writer, value).context(context)?;
    Ok(writer.finish())
}

fn canonical_sha256(value: &impl Serialize) -> Result<String> {
    let (_, digest) = serialized_json_digest(value, "failed to serialize content-addressed value")?;
    sha256_identity_from_hex(&digest, "serialized JSON digest")
}

fn deterministic_rng_seed(workflow: &str, phase: &str, checkpoint: &str) -> u64 {
    let digest = Sha256::digest(format!(
        "post_training_rng_v1\0{workflow}\0{phase}\0{checkpoint}"
    ));
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

fn empty_receipt_chain() -> String {
    sha256_identity(b"post_training_receipt_chain_v1")
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationRequest<'a> {
    pub prompt: &'a str,
    pub completion: &'a str,
    pub verifier_payload: &'a serde_json::Value,
    pub reference_answer: Option<&'a str>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Verification {
    pub reward: f64,
    pub passed: bool,
    pub components: BTreeMap<String, f64>,
}

/// Named, executor-injected reward implementation. The adapter name must
/// exactly match the task's verifier spec before it can score a rollout.
pub trait RewardVerifier {
    /// Content identity of the verifier implementation. Task parameters are
    /// already part of the phase hash; this identity prevents another binary
    /// with the same adapter name from changing rewards after resume.
    fn identity(&self) -> &str;
    fn adapter_name(&self) -> &str;
    fn verify(&self, spec: &VerifierSpec, request: VerificationRequest<'_>)
    -> Result<Verification>;
}

/// Score a rollout only when the task, task example, and injected verifier all
/// agree. This prevents accidental supervised/RL task coercion.
pub fn verify_rollout(
    task: &TaskConfig,
    example: &TaskExample,
    verifier: &dyn RewardVerifier,
    completion: &str,
) -> Result<Verification> {
    let Some(RewardSpec::Verifier { verifier: spec }) = task.reward_spec() else {
        bail!("task `{}` does not define a verifiable reward", task.name());
    };
    ensure!(
        verifier.adapter_name() == spec.adapter,
        "verifier adapter mismatch: task requires `{}`, executor supplied `{}`",
        spec.adapter,
        verifier.adapter_name()
    );
    let TaskExample::VerifiableRollout {
        prompt,
        verifier_payload,
        reference_answer,
    } = example
    else {
        bail!(
            "task `{}` did not construct a verifiable rollout",
            task.name()
        );
    };
    let verification = verifier.verify(
        &spec,
        VerificationRequest {
            prompt,
            completion,
            verifier_payload,
            reference_answer: reference_answer.as_deref(),
        },
    )?;
    ensure!(
        verification.reward.is_finite(),
        "verifier returned a non-finite reward"
    );
    ensure!(
        verification
            .components
            .iter()
            .all(|(name, value)| !name.trim().is_empty() && value.is_finite()),
        "verifier returned an empty component name or non-finite component value"
    );
    Ok(verification)
}

/// Built-in deterministic verifier for exact-answer curricula. Arbitrary test
/// harnesses, theorem provers, and judges remain explicit injected adapters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactAnswerVerifier;

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ExactAnswerParameters {
    case_fold: bool,
    trim: bool,
}

impl Default for ExactAnswerParameters {
    fn default() -> Self {
        Self {
            case_fold: false,
            trim: true,
        }
    }
}

impl RewardVerifier for ExactAnswerVerifier {
    fn identity(&self) -> &str {
        "sha256:ff08a60a76f25d2e611dd1e1755fcfe137e92913fa1ee1b8e1d59552ecce928b"
    }

    fn adapter_name(&self) -> &str {
        "exact_answer"
    }

    fn verify(
        &self,
        spec: &VerifierSpec,
        request: VerificationRequest<'_>,
    ) -> Result<Verification> {
        ensure!(
            spec.adapter == self.adapter_name(),
            "exact-answer verifier received `{}` spec",
            spec.adapter
        );
        let parameters: ExactAnswerParameters = serde_json::from_value(serde_json::Value::Object(
            spec.parameters.clone().into_iter().collect(),
        ))
        .context("invalid exact_answer verifier parameters")?;
        let reference = request
            .reference_answer
            .context("exact_answer verifier requires `reference_answer`")?;
        let normalize = |value: &str| {
            let value = if parameters.trim { value.trim() } else { value };
            if parameters.case_fold {
                value.to_lowercase()
            } else {
                value.to_owned()
            }
        };
        let passed = normalize(request.completion) == normalize(reference);
        let reward = f64::from(passed);
        Ok(Verification {
            reward,
            passed,
            components: BTreeMap::from([("exact_match".to_owned(), reward)]),
        })
    }
}

/// Stream validated task examples from JSONL (optionally zstd-compressed).
/// Causal `text_or_jsonl` data may instead be newline-delimited plain text;
/// the path suffix chooses framing and malformed JSON is never treated as text.
pub fn visit_task_examples(
    path: &Path,
    task: &TaskConfig,
    mut visit: impl FnMut(TaskExample) -> Result<()>,
) -> Result<usize> {
    visit_task_examples_while(path, task, |example| {
        visit(example)?;
        Ok(true)
    })
}

/// Control-flow variant of [`visit_task_examples`]. Returning `false` stops
/// before reading the next record, which lets a step-capped phase avoid
/// scanning the remainder of a multi-billion-token shard.
pub fn visit_task_examples_while(
    path: &Path,
    task: &TaskConfig,
    visit: impl FnMut(TaskExample) -> Result<bool>,
) -> Result<usize> {
    visit_task_examples_while_with_record_limit(path, task, MAX_POST_TRAINING_RECORD_BYTES, visit)
}

fn visit_task_examples_while_with_record_limit(
    path: &Path,
    task: &TaskConfig,
    maximum_record_bytes: usize,
    mut visit: impl FnMut(TaskExample) -> Result<bool>,
) -> Result<usize> {
    task.validate()?;
    ensure!(
        maximum_record_bytes > 0,
        "task record byte limit must be positive"
    );
    let file =
        File::open(path).with_context(|| format!("failed to open task data {}", path.display()))?;
    let mut reader: Box<dyn BufRead> = if path.extension().is_some_and(|ext| ext == "zst") {
        Box::new(BufReader::new(
            zstd::stream::read::Decoder::new(file)
                .with_context(|| format!("failed to open zstd task data {}", path.display()))?,
        ))
    } else {
        Box::new(BufReader::new(file))
    };
    let jsonl = task.contract().data_format == TaskDataFormat::Jsonl || is_jsonl_path(path);
    ensure!(
        task.contract().data_format != TaskDataFormat::Jsonl || jsonl,
        "task `{}` requires JSONL framing",
        task.name()
    );

    let mut count = 0usize;
    let mut line_number = 0usize;
    let mut line_bytes = Vec::new();
    loop {
        let next_line = line_number
            .checked_add(1)
            .context("task-data line counter overflows usize")?;
        let read =
            read_post_training_record_bounded(&mut *reader, &mut line_bytes, maximum_record_bytes)
                .with_context(|| {
                    format!("failed to read task data {}:{next_line}", path.display())
                })?;
        if read == 0 {
            break;
        }
        line_number = next_line;
        let line = String::from_utf8(std::mem::take(&mut line_bytes)).with_context(|| {
            format!("task data is not UTF-8 at {}:{line_number}", path.display())
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record = if jsonl {
            serde_json::from_str(&line).with_context(|| {
                format!(
                    "invalid JSON task record at {}:{line_number}",
                    path.display()
                )
            })?
        } else {
            serde_json::json!({"text": line})
        };
        let example = task
            .construct_example(&record)
            .with_context(|| format!("invalid task record at {}:{line_number}", path.display()))?;
        count = count
            .checked_add(1)
            .context("task example count overflows usize")?;
        if !visit(example)? {
            break;
        }
    }
    ensure!(
        count > 0,
        "task data {} contains no examples",
        path.display()
    );
    Ok(count)
}

fn is_jsonl_path(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    name.ends_with(".jsonl") || name.ends_with(".jsonl.zst")
}

fn ensure_finite(values: &[f64], name: &str) -> Result<()> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{name} must all be finite"
    );
    Ok(())
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn log_softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max) / temperature;
    let normalizer = logits
        .iter()
        .map(|logit| (logit / temperature - maximum).exp())
        .sum::<f64>()
        .ln()
        + maximum;
    logits
        .iter()
        .map(|logit| logit / temperature - normalizer)
        .collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("validated non-empty logits")
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::Write;
    use std::rc::Rc;

    use burn::prelude::Device;
    use serde_json::json;

    use super::*;
    use crate::native_host::{
        NativeHostMetricJournal, NativePostTrainingContextFactory, NativeWorkflowAdapters,
        NativeWorkflowHost,
    };
    use crate::runtime::RuntimeStatus;
    use crate::workflow::ResolvedWorkflow;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected:.16}, got {actual:.16}"
        );
    }

    #[test]
    fn post_training_record_reader_enforces_a_hard_allocation_bound() {
        let mut accepted = std::io::Cursor::new(b"12345\nrest".to_vec());
        let mut output = Vec::new();
        assert_eq!(
            read_post_training_record_bounded(&mut accepted, &mut output, 5).unwrap(),
            6
        );
        assert_eq!(output, b"12345\n");

        let mut oversized = std::io::Cursor::new(b"123456\n".to_vec());
        let error = read_post_training_record_bounded(&mut oversized, &mut output, 5)
            .expect_err("delimiter beyond the byte limit must be rejected");
        assert!(error.to_string().contains("maximum of 5 bytes"));

        let mut empty = std::io::Cursor::new(Vec::<u8>::new());
        assert!(read_post_training_record_bounded(&mut empty, &mut output, 0).is_err());
    }

    #[test]
    fn streaming_json_digest_matches_canonical_json_bytes() {
        let value = json!({
            "algorithm": "forward_kl",
            "rows": [[1.0, 2.0], [3.0, 4.0]],
            "enabled": true
        });
        let encoded = serde_json::to_vec(&value).unwrap();
        let (bytes, digest) = serialized_json_digest(&value, "test serialization").unwrap();
        assert_eq!(bytes, u64::try_from(encoded.len()).unwrap());
        assert_eq!(digest, format!("{:x}", Sha256::digest(&encoded)));
        assert_eq!(
            canonical_sha256(&value).unwrap(),
            format!("sha256:{digest}")
        );
    }

    #[test]
    fn dpo_golden_equal_margin_is_log_two_with_exact_gradient() {
        let result = dpo_loss(
            PairwiseLogProbabilities {
                policy_chosen: -2.0,
                policy_rejected: -3.0,
                reference_chosen: -4.0,
                reference_rejected: -5.0,
            },
            0.2,
            0.0,
        )
        .unwrap();
        close(result.loss, std::f64::consts::LN_2);
        close(result.implicit_reward_margin, 0.0);
        close(result.d_policy_chosen, -0.1);
        close(result.d_policy_rejected, 0.1);
        assert!(!result.preference_correct);
    }

    #[test]
    fn dpo_golden_known_logit_and_label_smoothing() {
        let z = 3.0_f64.ln();
        let result = dpo_loss(
            PairwiseLogProbabilities {
                policy_chosen: z,
                policy_rejected: 0.0,
                reference_chosen: 0.0,
                reference_rejected: 0.0,
            },
            1.0,
            0.0,
        )
        .unwrap();
        close(result.loss, -(0.75_f64).ln());
        close(result.d_policy_chosen, -0.25);
        assert!(result.preference_correct);

        let smoothed = dpo_loss(
            PairwiseLogProbabilities {
                policy_chosen: z,
                policy_rejected: 0.0,
                reference_chosen: 0.0,
                reference_rejected: 0.0,
            },
            1.0,
            0.1,
        )
        .unwrap();
        close(smoothed.loss, -0.9 * 0.75_f64.ln() - 0.1 * 0.25_f64.ln());
        close(smoothed.d_policy_chosen, -0.15);
    }

    #[test]
    fn dpo_tensor_core_matches_scalar_golden() {
        let device = Device::ndarray();
        let z = 3.0_f32.ln();
        let loss = dpo_loss_tensor(
            Tensor::<1>::from_floats([z], &device),
            Tensor::<1>::from_floats([0.0], &device),
            Tensor::<1>::from_floats([0.0], &device),
            Tensor::<1>::from_floats([0.0], &device),
            1.0,
            0.0,
        )
        .unwrap()
        .into_scalar::<f32>();
        assert!((f64::from(loss) + 0.75_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn sequence_reduction_is_explicit() {
        close(
            reduce_sequence_log_probs(&[-1.0, -3.0], SequenceReduction::Sum).unwrap(),
            -4.0,
        );
        close(
            reduce_sequence_log_probs(&[-1.0, -3.0], SequenceReduction::Mean).unwrap(),
            -2.0,
        );
        assert!(reduce_sequence_log_probs(&[], SequenceReduction::Sum).is_err());
    }

    #[test]
    fn forward_kl_golden_binary_distribution_and_gradient() {
        let teacher = [3.0_f64.ln(), 0.0];
        let student = [0.0, 0.0];
        let result = forward_kl_distillation(
            &[DistillationToken {
                teacher_logits: &teacher,
                student_logits: &student,
                weight: 1.0,
            }],
            1.0,
            true,
        )
        .unwrap();
        let expected = 0.75 * 1.5_f64.ln() + 0.25 * 0.5_f64.ln();
        close(result.loss, expected);
        close(result.mean_forward_kl, expected);
        close(result.student_gradients[0][0], -0.25);
        close(result.student_gradients[0][1], 0.25);
        close(result.top1_agreement, 0.0);
    }

    #[test]
    fn distillation_masking_and_temperature_scaling_are_exact() {
        let same = [2.0, -1.0];
        let ignored_teacher = [100.0, -100.0];
        let ignored_student = [-100.0, 100.0];
        let result = forward_kl_distillation(
            &[
                DistillationToken {
                    teacher_logits: &same,
                    student_logits: &same,
                    weight: 2.0,
                },
                DistillationToken {
                    teacher_logits: &ignored_teacher,
                    student_logits: &ignored_student,
                    weight: 0.0,
                },
            ],
            2.0,
            true,
        )
        .unwrap();
        close(result.loss, 0.0);
        close(result.student_gradients[0][0], 0.0);
        close(result.student_gradients[1][0], 0.0);
        close(result.top1_agreement, 1.0);
    }

    #[test]
    fn distillation_tensor_core_matches_scalar_golden() {
        let device = Device::ndarray();
        let teacher = Tensor::<2>::from_floats([[3.0_f32.ln(), 0.0]], &device);
        let student = Tensor::<2>::from_floats([[0.0, 0.0]], &device);
        let loss = forward_kl_distillation_tensor(teacher, student, 1.0, true)
            .unwrap()
            .into_scalar::<f32>();
        let expected = 0.75 * 1.5_f64.ln() + 0.25 * 0.5_f64.ln();
        assert!((f64::from(loss) - expected).abs() < 1e-6);
    }

    #[test]
    fn grpo_golden_group_normalization_and_clipping() {
        let current_a = [1.5_f64.ln()];
        let current_b = [0.5_f64.ln()];
        let behavior = [0.0];
        let result = grpo_loss(
            &[
                GrpoRollout {
                    reward: 1.0,
                    current_token_log_probs: &current_a,
                    behavior_token_log_probs: &behavior,
                    reference_token_log_probs: None,
                },
                GrpoRollout {
                    reward: 0.0,
                    current_token_log_probs: &current_b,
                    behavior_token_log_probs: &behavior,
                    reference_token_log_probs: None,
                },
            ],
            0.2,
            1e-6,
            0.0,
        )
        .unwrap();
        close(result.mean_reward, 0.5);
        close(result.reward_stddev, 0.5);
        assert_eq!(result.advantages, vec![1.0, -1.0]);
        close(result.loss, -0.2);
        close(result.clipped_fraction, 1.0);
        close(result.current_log_prob_gradients[0][0], 0.0);
        close(result.current_log_prob_gradients[1][0], 0.0);
    }

    #[test]
    fn grpo_unclipped_gradient_and_kl_estimator_are_exact() {
        let current = [0.0];
        let behavior = [0.0];
        let reference = [2.0_f64.ln()];
        let result = grpo_loss(
            &[
                GrpoRollout {
                    reward: 1.0,
                    current_token_log_probs: &current,
                    behavior_token_log_probs: &behavior,
                    reference_token_log_probs: Some(&reference),
                },
                GrpoRollout {
                    reward: 0.0,
                    current_token_log_probs: &current,
                    behavior_token_log_probs: &behavior,
                    reference_token_log_probs: Some(&reference),
                },
            ],
            0.2,
            1e-6,
            0.5,
        )
        .unwrap();
        let kl = 2.0 - 2.0_f64.ln() - 1.0;
        close(result.mean_kl, kl);
        close(result.loss, 0.5 * kl);
        // Group scaling is 1/2: policy term -A/2 plus KL derivative -1/4.
        close(result.current_log_prob_gradients[0][0], -0.75);
        close(result.current_log_prob_gradients[1][0], 0.25);
    }

    #[test]
    fn grpo_tensor_core_matches_scalar_golden() {
        let device = Device::ndarray();
        let loss = grpo_loss_tensor(
            Tensor::<1>::from_floats([1.0, 0.0], &device),
            Tensor::<2>::from_floats([[1.5_f32.ln()], [0.5_f32.ln()]], &device),
            Tensor::<2>::from_floats([[0.0], [0.0]], &device),
            None,
            Tensor::<2>::from_floats([[1.0], [1.0]], &device),
            0.2,
            1e-6,
            0.0,
        )
        .unwrap()
        .into_scalar::<f32>();
        assert!((f64::from(loss) + 0.2).abs() < 1e-6);
    }

    #[test]
    fn grpo_tensor_rejects_fractional_and_empty_masks() {
        let device = Device::ndarray();
        for (mask, expected) in [
            ([[1.0, 0.5], [1.0, 0.0]], "binary zero/one"),
            ([[1.0, 0.0], [0.0, 0.0]], "at least one active token"),
        ] {
            let error = grpo_loss_tensor(
                Tensor::<1>::from_floats([1.0, 0.0], &device),
                Tensor::<2>::from_floats([[0.0, 0.0], [0.0, 0.0]], &device),
                Tensor::<2>::from_floats([[0.0, 0.0], [0.0, 0.0]], &device),
                None,
                Tensor::<2>::from_floats(mask, &device),
                0.2,
                1e-6,
                0.0,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn grpo_tensor_rejects_non_finite_and_unsafe_exponent_inputs() {
        let device = Device::ndarray();
        let error = grpo_loss_tensor(
            Tensor::<1>::from_floats([1.0, 0.0], &device),
            Tensor::<2>::from_floats([[f32::NAN], [0.0]], &device),
            Tensor::<2>::from_floats([[0.0], [0.0]], &device),
            None,
            Tensor::<2>::from_floats([[1.0], [1.0]], &device),
            0.2,
            1e-6,
            0.0,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("current log probabilities"), "{error}");

        let error = grpo_loss_tensor(
            Tensor::<1>::from_floats([1.0, 0.0], &device),
            Tensor::<2>::from_floats([[81.0], [0.0]], &device),
            Tensor::<2>::from_floats([[0.0], [0.0]], &device),
            None,
            Tensor::<2>::from_floats([[1.0], [1.0]], &device),
            0.2,
            1e-6,
            0.0,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("importance log-ratio"), "{error}");

        let error = grpo_loss_tensor(
            Tensor::<1>::from_floats([1.0, 0.0], &device),
            Tensor::<2>::from_floats([[0.0], [0.0]], &device),
            Tensor::<2>::from_floats([[0.0], [0.0]], &device),
            Some(Tensor::<2>::from_floats([[81.0], [0.0]], &device)),
            Tensor::<2>::from_floats([[1.0], [1.0]], &device),
            0.2,
            1e-6,
            0.1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("reference log-ratio"), "{error}");
    }

    #[test]
    fn grpo_tensor_rejects_finite_inputs_that_overflow_the_objective() {
        let device = Device::ndarray();
        let error = grpo_loss_tensor(
            Tensor::<1>::from_floats([1.0, 0.0], &device),
            Tensor::<2>::from_floats([[-80.0], [-80.0]], &device),
            Tensor::<2>::from_floats([[-80.0], [-80.0]], &device),
            Some(Tensor::<2>::from_floats([[0.0], [0.0]], &device)),
            Tensor::<2>::from_floats([[1.0], [1.0]], &device),
            0.2,
            1e-6,
            1.0e10,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("tensor objective overflowed"), "{error}");
    }

    #[test]
    fn grpo_scalar_rejects_finite_inputs_that_overflow_derived_values() {
        let zero = [0.0];
        let reward_overflow = [
            GrpoRollout {
                reward: f64::MAX,
                current_token_log_probs: &zero,
                behavior_token_log_probs: &zero,
                reference_token_log_probs: None,
            },
            GrpoRollout {
                reward: -f64::MAX,
                current_token_log_probs: &zero,
                behavior_token_log_probs: &zero,
                reference_token_log_probs: None,
            },
        ];
        let error = grpo_loss(&reward_overflow, 0.2, 1e-6, 0.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("reward normalization overflowed"), "{error}");

        let current = [-f64::MAX];
        let behavior = [f64::MAX];
        let unsafe_ratio = [
            GrpoRollout {
                reward: 1.0,
                current_token_log_probs: &current,
                behavior_token_log_probs: &behavior,
                reference_token_log_probs: None,
            },
            GrpoRollout {
                reward: 0.0,
                current_token_log_probs: &zero,
                behavior_token_log_probs: &zero,
                reference_token_log_probs: None,
            },
        ];
        let error = grpo_loss(&unsafe_ratio, 0.2, 1e-6, 0.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsafe importance ratio"), "{error}");

        let low = [-80.0];
        let reference = [0.0];
        let objective_overflow = [
            GrpoRollout {
                reward: 1.0,
                current_token_log_probs: &low,
                behavior_token_log_probs: &low,
                reference_token_log_probs: Some(&reference),
            },
            GrpoRollout {
                reward: 0.0,
                current_token_log_probs: &low,
                behavior_token_log_probs: &low,
                reference_token_log_probs: Some(&reference),
            },
        ];
        let error = grpo_loss(&objective_overflow, 0.2, 1e-6, f64::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("GRPO objective overflowed"), "{error}");
    }

    #[test]
    fn grpo_rejects_missing_reference_instead_of_dropping_kl() {
        let values = [0.0];
        let error = grpo_loss(
            &[
                GrpoRollout {
                    reward: 1.0,
                    current_token_log_probs: &values,
                    behavior_token_log_probs: &values,
                    reference_token_log_probs: None,
                },
                GrpoRollout {
                    reward: 0.0,
                    current_token_log_probs: &values,
                    behavior_token_log_probs: &values,
                    reference_token_log_probs: None,
                },
            ],
            0.2,
            1e-6,
            0.1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("needs reference"), "{error}");
    }

    #[test]
    fn exact_answer_verifier_is_strict_and_task_bound() {
        let task: TaskConfig = serde_json::from_value(json!({
            "type": "verifiable_rl",
            "verifier": {
                "adapter": "exact_answer",
                "parameters": {"case_fold": true, "trim": true}
            }
        }))
        .unwrap();
        let example = task
            .construct_example(&json!({
                "prompt": "2 + 2?",
                "verifier_payload": {"source": "arithmetic"},
                "reference_answer": "FOUR"
            }))
            .unwrap();
        let result = verify_rollout(&task, &example, &ExactAnswerVerifier, " four ").unwrap();
        assert!(result.passed);
        close(result.reward, 1.0);

        let wrong = verify_rollout(&task, &example, &ExactAnswerVerifier, "five").unwrap();
        assert!(!wrong.passed);
        close(wrong.reward, 0.0);
    }

    #[test]
    fn verifier_adapter_mismatch_is_not_coerced() {
        struct WrongVerifier;
        impl RewardVerifier for WrongVerifier {
            fn identity(&self) -> &str {
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"
            }

            fn adapter_name(&self) -> &str {
                "unit_tests"
            }

            fn verify(
                &self,
                _spec: &VerifierSpec,
                _request: VerificationRequest<'_>,
            ) -> Result<Verification> {
                unreachable!()
            }
        }

        let task: TaskConfig = serde_json::from_value(json!({
            "type": "verifiable_rl",
            "verifier": {"adapter": "exact_answer"}
        }))
        .unwrap();
        let example = task
            .construct_example(&json!({
                "prompt": "p",
                "verifier_payload": {},
                "reference_answer": "a"
            }))
            .unwrap();
        let error = verify_rollout(&task, &example, &WrongVerifier, "a")
            .unwrap_err()
            .to_string();
        assert!(error.contains("adapter mismatch"), "{error}");
    }

    #[test]
    fn grpo_verifier_mismatch_rejects_before_generating_rollouts() {
        struct WrongVerifier;

        impl RewardVerifier for WrongVerifier {
            fn identity(&self) -> &str {
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"
            }

            fn adapter_name(&self) -> &str {
                "wrong_verifier"
            }

            fn verify(
                &self,
                _spec: &VerifierSpec,
                _request: VerificationRequest<'_>,
            ) -> Result<Verification> {
                unreachable!("verifier mismatch must reject during adapter preflight")
            }
        }

        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Grpo);
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestGrpoRuntime::new();
        let mut adapters = PostTrainingPhaseRuntime::Grpo {
            runtime: &mut policy,
            verifier: &WrongVerifier,
        };
        let error = drive_resumable_post_training_phase(
            &test_sha("wrong-verifier-preflight"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut None,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("verifier adapter mismatch"), "{error}");
        assert_eq!(policy.model_tokens, 0);
        assert_eq!(publisher.restore_calls, 0);
    }

    #[test]
    fn grpo_rejects_non_finite_verifier_reward_before_numeric_batch() {
        struct NonFiniteVerifier;

        impl RewardVerifier for NonFiniteVerifier {
            fn identity(&self) -> &str {
                "sha256:3333333333333333333333333333333333333333333333333333333333333333"
            }

            fn adapter_name(&self) -> &str {
                "exact_answer"
            }

            fn verify(
                &self,
                _spec: &VerifierSpec,
                _request: VerificationRequest<'_>,
            ) -> Result<Verification> {
                Ok(Verification {
                    reward: f64::NAN,
                    passed: false,
                    components: BTreeMap::new(),
                })
            }
        }

        let (_dir, mut request) = test_request(TestPostTrainingAlgorithm::Grpo);
        request.phase.steps = Some(1);
        let mut publisher = TestUpdatePublisher::new(false);
        let mut runtime = TestGrpoRuntime::new();
        let mut phase_runtime = PostTrainingPhaseRuntime::Grpo {
            runtime: &mut runtime,
            verifier: &NonFiniteVerifier,
        };
        let error = drive_resumable_post_training_phase(
            &test_sha("non-finite-verifier-reward"),
            &request,
            &mut phase_runtime,
            &mut publisher,
            &mut None,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("non-finite reward"), "{error}");
        assert_eq!(runtime.generation_calls, 1);
        assert_eq!(runtime.prepare_calls, 0);
        assert_eq!(runtime.optimizer_steps, 0);
        assert_eq!(publisher.apply_calls, 0);
    }

    #[test]
    fn deserialized_post_training_state_revalidates_artifact_identities() {
        let empty_uri: PostTrainingCommittedState = serde_json::from_value(json!({
            "model": {"uri": "", "sha256": test_sha("model")}
        }))
        .unwrap();
        let error = empty_uri.validate().unwrap_err().to_string();
        assert!(error.contains("checkpoint URI is empty"), "{error}");

        let malformed_optimizer: PostTrainingCommittedState = serde_json::from_value(json!({
            "model": {"uri": "test://model", "sha256": test_sha("model")},
            "optimizer": {"uri": "test://optimizer", "sha256": "not-a-digest"}
        }))
        .unwrap();
        let error = malformed_optimizer.validate().unwrap_err().to_string();
        assert!(error.contains("artifact digest"), "{error}");
    }

    #[test]
    fn task_example_reader_validates_jsonl_without_coercion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preference.jsonl");
        let mut file = File::create(&path).unwrap();
        writeln!(
            file,
            "{}",
            json!({"prompt": "p", "chosen": "yes", "rejected": "no"})
        )
        .unwrap();
        let task: TaskConfig =
            serde_json::from_value(json!({"type": "pairwise_preference"})).unwrap();
        let mut examples = Vec::new();
        let count = visit_task_examples(&path, &task, |example| {
            examples.push(example);
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 1);
        assert!(matches!(
            &examples[0],
            TaskExample::PairwisePreference { chosen, rejected, .. }
                if chosen == "yes" && rejected == "no"
        ));

        let bad_path = dir.path().join("bad.jsonl");
        std::fs::write(&bad_path, "plain text\n").unwrap();
        let error = visit_task_examples(&bad_path, &task, |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid JSON"), "{error}");
    }

    #[test]
    fn task_example_reader_rejects_oversized_plain_record_before_visiting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized.txt");
        fs::write(&path, format!("{}\n", "x".repeat(65))).unwrap();
        let task: TaskConfig = serde_json::from_value(json!({"type": "causal_lm"})).unwrap();
        let mut visits = 0usize;
        let error = visit_task_examples_while_with_record_limit(&path, &task, 64, |_| {
            visits += 1;
            Ok(true)
        })
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("maximum of 64 bytes"), "{error}");
        assert_eq!(visits, 0);
    }

    #[test]
    fn task_example_reader_rejects_oversized_zstd_record_before_visiting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized.txt.zst");
        let payload = format!("{}\n", "x".repeat(65));
        let compressed = zstd::stream::encode_all(payload.as_bytes(), 0).unwrap();
        fs::write(&path, compressed).unwrap();
        let task: TaskConfig = serde_json::from_value(json!({"type": "causal_lm"})).unwrap();
        let mut visits = 0usize;
        let error = visit_task_examples_while_with_record_limit(&path, &task, 64, |_| {
            visits += 1;
            Ok(true)
        })
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("maximum of 64 bytes"), "{error}");
        assert_eq!(visits, 0);
    }

    #[test]
    fn task_example_reader_early_stop_does_not_read_later_oversized_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("early-stop.txt");
        fs::write(&path, format!("first\n{}\n", "x".repeat(65))).unwrap();
        let task: TaskConfig = serde_json::from_value(json!({"type": "causal_lm"})).unwrap();
        let mut visits = 0usize;
        let count = visit_task_examples_while_with_record_limit(&path, &task, 64, |_| {
            visits += 1;
            Ok(false)
        })
        .unwrap();

        assert_eq!(count, 1);
        assert_eq!(visits, 1);
    }

    #[test]
    fn authenticated_post_training_batches_stream_sequentially_across_epochs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preference.jsonl");
        let rows = ["p0", "p1", "p2"]
            .into_iter()
            .map(|prompt| json!({"prompt": prompt, "chosen": "yes", "rejected": "no"}))
            .map(|row| serde_json::to_string(&row).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(&path, rows).unwrap();
        let task: TaskConfig =
            serde_json::from_value(json!({"type": "pairwise_preference"})).unwrap();
        let source = AuthenticatedPostTrainingInput::open(&path).unwrap();
        let mut stream = PostTrainingBatchStream::new(
            source,
            task,
            PostTrainingRecordPosition {
                epoch: 0,
                record: 0,
            },
            2,
        )
        .unwrap();

        let error = stream
            .next_batch(MAX_POST_TRAINING_UPDATE_EXAMPLES + 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("target batch exceeds"), "{error}");

        let first = stream.next_batch(2).unwrap();
        assert_eq!(first.end.epoch, 0);
        assert_eq!(first.end.record, 2);
        assert!(matches!(
            &first.examples[..],
            [
                TaskExample::PairwisePreference { prompt: first, .. },
                TaskExample::PairwisePreference { prompt: second, .. }
            ] if first == "p0" && second == "p1"
        ));

        let second = stream.next_batch(2).unwrap();
        assert_eq!(second.end.epoch, 1);
        assert_eq!(second.end.record, 1);
        assert!(matches!(
            &second.examples[..],
            [
                TaskExample::PairwisePreference { prompt: first, .. },
                TaskExample::PairwisePreference { prompt: second, .. }
            ] if first == "p2" && second == "p0"
        ));

        let final_batch = stream.next_batch(2).unwrap();
        assert_eq!(final_batch.end.epoch, 2);
        assert_eq!(final_batch.end.record, 0);
        assert!(matches!(
            &final_batch.examples[..],
            [
                TaskExample::PairwisePreference { prompt: first, .. },
                TaskExample::PairwisePreference { prompt: second, .. }
            ] if first == "p1" && second == "p2"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn authenticated_post_training_stream_rejects_path_replacement_before_consumption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preference.jsonl");
        let replacement = dir.path().join("replacement.jsonl");
        fs::write(
            &path,
            serde_json::to_string(
                &json!({"prompt": "original", "chosen": "yes", "rejected": "no"}),
            )
            .unwrap()
                + "\n",
        )
        .unwrap();
        fs::write(
            &replacement,
            serde_json::to_string(
                &json!({"prompt": "replaced", "chosen": "yes", "rejected": "no"}),
            )
            .unwrap()
                + "\n",
        )
        .unwrap();
        let task: TaskConfig =
            serde_json::from_value(json!({"type": "pairwise_preference"})).unwrap();
        let source = AuthenticatedPostTrainingInput::open(&path).unwrap();
        let mut stream = PostTrainingBatchStream::new(
            source,
            task,
            PostTrainingRecordPosition {
                epoch: 0,
                record: 0,
            },
            1,
        )
        .unwrap();

        fs::rename(&replacement, &path).unwrap();
        let error = stream.next_batch(1).unwrap_err().to_string();
        assert!(
            error.contains("changed after it was authenticated"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn authenticated_post_training_stream_rejects_same_inode_mutation() {
        use std::os::unix::fs::MetadataExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preference.jsonl");
        let original = serde_json::to_string(
            &json!({"prompt": "original", "chosen": "yes", "rejected": "no"}),
        )
        .unwrap()
            + "\n";
        let modified = serde_json::to_string(
            &json!({"prompt": "modified", "chosen": "yes", "rejected": "no"}),
        )
        .unwrap()
            + "\n";
        assert_eq!(original.len(), modified.len());
        fs::write(&path, original).unwrap();
        let inode = fs::metadata(&path).unwrap().ino();
        let task: TaskConfig =
            serde_json::from_value(json!({"type": "pairwise_preference"})).unwrap();
        let source = AuthenticatedPostTrainingInput::open(&path).unwrap();
        let mut stream = PostTrainingBatchStream::new(
            source,
            task,
            PostTrainingRecordPosition {
                epoch: 0,
                record: 0,
            },
            1,
        )
        .unwrap();

        fs::write(&path, modified).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().ino(), inode);
        let error = stream.next_batch(1).unwrap_err().to_string();
        assert!(
            error.contains("changed after it was authenticated"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn authenticated_post_training_input_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.jsonl");
        let link = dir.path().join("input.jsonl");
        fs::write(
            &target,
            "{\"prompt\":\"p\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n",
        )
        .unwrap();
        symlink(&target, &link).unwrap();

        let error = match AuthenticatedPostTrainingInput::open(&link) {
            Ok(_) => panic!("symlinked post-training input unexpectedly authenticated"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("non-symlink regular file"), "{error}");
    }

    const ABC_SHA256: &str =
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn test_device_receipt(content: &impl Serialize, model_tokens: u64) -> DeviceExecutionReceipt {
        DeviceExecutionReceipt {
            execution_sha256: canonical_sha256(content).unwrap(),
            model_tokens,
        }
    }

    #[derive(Clone, Copy)]
    enum TestDpoFault {
        MalformedReceipt,
        PartialSummary,
        TokenCountMismatch,
        NonFiniteSummary,
        NegativeLoss,
    }

    struct TestDpoRuntime {
        identity: String,
        frozen_identity: String,
        frozen_tokenizer_identity: String,
        prepare_calls: usize,
        optimizer_steps: usize,
        model_tokens: u64,
        rng_ranges: Vec<ModelExecutionRngRange>,
        fault: Option<TestDpoFault>,
    }

    impl TestDpoRuntime {
        fn new(identity: impl Into<String>) -> Self {
            Self {
                identity: identity.into(),
                frozen_identity: test_sha("abc"),
                frozen_tokenizer_identity: "tokenizer-sha256:test".to_owned(),
                prepare_calls: 0,
                optimizer_steps: 0,
                model_tokens: 0,
                rng_ranges: Vec::new(),
                fault: None,
            }
        }
    }

    impl DpoDeviceBatchRuntime for TestDpoRuntime {
        fn trainable_identity(&self) -> &str {
            &self.identity
        }

        fn trainable_tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn frozen_identity(&self) -> &str {
            &self.frozen_identity
        }

        fn frozen_tokenizer_identity(&self) -> &str {
            &self.frozen_tokenizer_identity
        }

        fn model_tokens_processed(&self) -> u64 {
            self.model_tokens
        }

        fn prepare_device_batch(
            &mut self,
            request: DpoDeviceBatchRequest<'_>,
        ) -> Result<DpoDeviceBatchResult> {
            validate_sha256_identity(request.batch_sha256, "test DPO batch")?;
            request.base.validate()?;
            let examples = u64::try_from(request.examples.len()).unwrap();
            ensure!(
                request.rng.len() == examples * 2,
                "test DPO runtime received the wrong RNG geometry"
            );
            let mut loss_sum = 0.0;
            let mut preference_correct = 0;
            let mut implicit_reward_margin_sum = 0.0;
            for (index, example) in request.examples.iter().enumerate() {
                let TaskExample::PairwisePreference { .. } = example else {
                    bail!("test DPO runtime received another task type");
                };
                let offset = u64::try_from(index).unwrap() * 2;
                let chosen_rng = request.rng.substream(offset)?;
                let rejected_rng = request.rng.substream(offset + 1)?;
                let chosen_jitter =
                    ((chosen_rng.seed ^ chosen_rng.counter) & 0xffff) as f64 * f64::EPSILON;
                let rejected_jitter =
                    ((rejected_rng.seed ^ rejected_rng.counter) & 0xffff) as f64 * f64::EPSILON;
                let objective = dpo_loss(
                    PairwiseLogProbabilities {
                        policy_chosen: -0.2 + chosen_jitter,
                        policy_rejected: -0.8 + rejected_jitter,
                        reference_chosen: -0.5,
                        reference_rejected: -0.5,
                    },
                    request.beta,
                    request.label_smoothing,
                )?;
                loss_sum += objective.loss;
                preference_correct += u64::from(objective.preference_correct);
                implicit_reward_margin_sum += objective.implicit_reward_margin;
            }
            ensure!(request.loss_weight.is_finite(), "invalid test loss weight");
            self.prepare_calls += 1;
            self.rng_ranges.push(request.rng);
            let model_tokens = examples * 2;
            self.model_tokens += model_tokens;
            let mut result = DpoDeviceBatchResult {
                receipt: test_device_receipt(
                    &(
                        "test-dpo-device-v1",
                        request.batch_sha256,
                        request.base,
                        request.max_sequence_tokens,
                        request.beta,
                        request.label_smoothing,
                        request.sequence_reduction,
                        request.loss_weight,
                        request.rng,
                        loss_sum,
                        preference_correct,
                        implicit_reward_margin_sum,
                        "backward-staged",
                    ),
                    model_tokens,
                ),
                examples,
                loss_sum,
                preference_correct,
                implicit_reward_margin_sum,
            };
            match self.fault {
                Some(TestDpoFault::MalformedReceipt) => {
                    result.receipt.execution_sha256 = "not-a-digest".to_owned();
                }
                Some(TestDpoFault::PartialSummary) => result.examples -= 1,
                Some(TestDpoFault::TokenCountMismatch) => {
                    result.receipt.model_tokens += 1;
                }
                Some(TestDpoFault::NonFiniteSummary) => result.loss_sum = f64::NAN,
                Some(TestDpoFault::NegativeLoss) => result.loss_sum = -1.0,
                None => {}
            }
            Ok(result)
        }

        fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()> {
            close(learning_rate_scale, 0.5);
            self.optimizer_steps += 1;
            Ok(())
        }
    }

    #[test]
    fn dpo_device_runtime_rejects_tokenizer_misalignment_before_model_work() {
        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Dpo);
        let mut runtime = TestDpoRuntime::new("candidate:dpo");
        runtime.frozen_tokenizer_identity = "tokenizer-sha256:drifted".to_owned();
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut runtime,
        };
        let error = validate_and_capture_runtime_identities(&request.phase, &mut adapters)
            .unwrap_err()
            .to_string();
        assert!(error.contains("tokenizer identities differ"), "{error}");
        assert_eq!(runtime.prepare_calls, 0);
    }

    struct TestForwardKlRuntime {
        prepare_calls: usize,
        optimizer_steps: usize,
        model_tokens: u64,
        rng_ranges: Vec<ModelExecutionRngRange>,
    }

    impl TestForwardKlRuntime {
        fn new() -> Self {
            Self {
                prepare_calls: 0,
                optimizer_steps: 0,
                model_tokens: 0,
                rng_ranges: Vec::new(),
            }
        }
    }

    impl ForwardKlDeviceBatchRuntime for TestForwardKlRuntime {
        fn trainable_identity(&self) -> &str {
            "candidate:student"
        }

        fn trainable_tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn frozen_identity(&self) -> &str {
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        }

        fn frozen_tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn model_tokens_processed(&self) -> u64 {
            self.model_tokens
        }

        fn prepare_device_batch(
            &mut self,
            request: ForwardKlDeviceBatchRequest<'_>,
        ) -> Result<ForwardKlDeviceBatchResult> {
            validate_sha256_identity(request.batch_sha256, "test forward-KL batch")?;
            request.base.validate()?;
            let examples = u64::try_from(request.examples.len()).unwrap();
            ensure!(
                request.rng.len() == examples,
                "test forward-KL runtime received the wrong RNG geometry"
            );
            let teacher = [3.0_f64.ln(), 0.0];
            let mut loss_sum = 0.0;
            let mut forward_kl_sum = 0.0;
            let mut teacher_entropy_sum = 0.0;
            let mut top1_agreement_sum = 0.0;
            for (index, example) in request.examples.iter().enumerate() {
                ensure!(
                    matches!(
                        example,
                        TaskExample::Autoregressive { .. }
                            | TaskExample::SupervisedGeneration { .. }
                    ),
                    "test forward-KL runtime received another task type"
                );
                let execution_rng = request.rng.substream(u64::try_from(index).unwrap())?;
                let jitter =
                    ((execution_rng.seed ^ execution_rng.counter) & 0xffff) as f64 * f64::EPSILON;
                let student = [jitter, 0.0];
                let objective = forward_kl_distillation(
                    &[DistillationToken {
                        teacher_logits: &teacher,
                        student_logits: &student,
                        weight: 1.0,
                    }],
                    request.temperature,
                    request.scale_by_temperature_squared,
                )?;
                loss_sum += objective.loss;
                forward_kl_sum += objective.mean_forward_kl;
                teacher_entropy_sum += objective.teacher_entropy;
                top1_agreement_sum += objective.top1_agreement;
            }
            ensure!(request.loss_weight.is_finite(), "invalid test loss weight");
            self.prepare_calls += 1;
            self.rng_ranges.push(request.rng);
            self.model_tokens += examples;
            Ok(ForwardKlDeviceBatchResult {
                receipt: test_device_receipt(
                    &(
                        "test-forward-kl-device-v1",
                        request.batch_sha256,
                        request.base,
                        request.max_sequence_tokens,
                        request.temperature,
                        request.scale_by_temperature_squared,
                        request.loss_weight,
                        request.rng,
                        loss_sum,
                        forward_kl_sum,
                        teacher_entropy_sum,
                        top1_agreement_sum,
                        "backward-staged",
                    ),
                    examples,
                ),
                examples,
                loss_sum,
                forward_kl_sum,
                teacher_entropy_sum,
                top1_agreement_sum,
            })
        }

        fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()> {
            close(learning_rate_scale, 1.0);
            self.optimizer_steps += 1;
            Ok(())
        }
    }

    struct TestGrpoRuntime {
        generation_calls: usize,
        prepare_calls: usize,
        optimizer_steps: usize,
        model_tokens: u64,
        rollout_tokens: usize,
        has_reference: bool,
        generation_ranges: Vec<ModelExecutionRngRange>,
        scoring_ranges: Vec<ModelExecutionRngRange>,
    }

    impl TestGrpoRuntime {
        fn new() -> Self {
            Self {
                generation_calls: 0,
                prepare_calls: 0,
                optimizer_steps: 0,
                model_tokens: 0,
                rollout_tokens: 1,
                has_reference: false,
                generation_ranges: Vec::new(),
                scoring_ranges: Vec::new(),
            }
        }

        fn with_reference() -> Self {
            Self {
                has_reference: true,
                ..Self::new()
            }
        }
    }

    impl GrpoDeviceBatchRuntime for TestGrpoRuntime {
        fn trainable_identity(&self) -> &str {
            "candidate:policy"
        }

        fn trainable_tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn frozen_identity(&self) -> Option<&str> {
            self.has_reference.then_some(
                "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
        }

        fn frozen_tokenizer_identity(&self) -> Option<&str> {
            self.has_reference.then_some("tokenizer-sha256:test")
        }

        fn model_tokens_processed(&self) -> u64 {
            self.model_tokens
        }

        fn generate_device_batch(
            &mut self,
            request: GrpoGenerationBatchRequest<'_>,
        ) -> Result<GrpoGeneratedBatch> {
            validate_sha256_identity(request.batch_sha256, "test GRPO batch")?;
            request.base.validate()?;
            let examples = u64::try_from(request.examples.len()).unwrap();
            let group_size = u64::try_from(request.group_size).unwrap();
            ensure!(
                request.rng.len() == examples * group_size,
                "test GRPO generator received the wrong RNG geometry"
            );
            self.generation_calls += 1;
            self.generation_ranges.push(request.rng);
            self.model_tokens += examples * group_size;
            let groups = request
                .examples
                .iter()
                .map(|_| {
                    (0..request.group_size)
                        .map(|index| GeneratedRolloutText {
                            text: if index == 0 { "ok" } else { "wrong" }.to_owned(),
                            token_count: self.rollout_tokens,
                        })
                        .collect()
                })
                .collect();
            Ok(GrpoGeneratedBatch {
                generation_sha256: canonical_sha256(&(
                    "test-grpo-generation-v1",
                    request.batch_sha256,
                    request.base,
                    request.group_size,
                    request.max_sequence_tokens,
                    request.sampling,
                    request.rng,
                    self.rollout_tokens,
                ))?,
                groups,
            })
        }

        fn prepare_device_batch(
            &mut self,
            request: GrpoDeviceBatchRequest<'_>,
        ) -> Result<GrpoDeviceBatchResult> {
            validate_sha256_identity(request.batch_sha256, "test GRPO batch")?;
            request.base.validate()?;
            let examples = u64::try_from(request.examples.len()).unwrap();
            let rollouts = request
                .generation
                .groups
                .iter()
                .try_fold(0_u64, |total, group| {
                    total.checked_add(u64::try_from(group.len()).unwrap())
                })
                .context("test GRPO rollout count overflow")?;
            ensure!(
                request.rng.len() == rollouts,
                "test GRPO scorer received the wrong RNG geometry"
            );
            ensure!(
                request.rewards.len() == request.generation.groups.len(),
                "test GRPO rewards are not grouped"
            );
            let mut loss_sum = 0.0;
            let mut mean_reward_sum = 0.0;
            let mut reward_stddev_sum = 0.0;
            let mut mean_kl_sum = 0.0;
            let mut clipped_fraction_sum = 0.0;
            for (group, rewards) in request.generation.groups.iter().zip(request.rewards) {
                ensure!(
                    group.len() == rewards.len(),
                    "test GRPO reward count differs"
                );
                let current: Vec<_> = group
                    .iter()
                    .map(|rollout| vec![0.0; rollout.token_count])
                    .collect();
                let behavior = current.clone();
                let reference: Option<Vec<_>> = self.has_reference.then(|| {
                    group
                        .iter()
                        .map(|rollout| vec![-0.1; rollout.token_count])
                        .collect()
                });
                let scalar: Vec<_> = rewards
                    .iter()
                    .enumerate()
                    .map(|(index, reward)| GrpoRollout {
                        reward: *reward,
                        current_token_log_probs: &current[index],
                        behavior_token_log_probs: &behavior[index],
                        reference_token_log_probs: reference
                            .as_ref()
                            .map(|rows| rows[index].as_slice()),
                    })
                    .collect();
                let objective = grpo_loss(
                    &scalar,
                    request.clip_epsilon,
                    request.advantage_epsilon,
                    request.kl_coefficient,
                )?;
                loss_sum += objective.loss;
                mean_reward_sum += objective.mean_reward;
                reward_stddev_sum += objective.reward_stddev;
                mean_kl_sum += objective.mean_kl;
                clipped_fraction_sum += objective.clipped_fraction;
            }
            ensure!(request.loss_weight.is_finite(), "invalid test loss weight");
            self.prepare_calls += 1;
            self.scoring_ranges.push(request.rng);
            self.model_tokens += rollouts;
            Ok(GrpoDeviceBatchResult {
                receipt: test_device_receipt(
                    &(
                        "test-grpo-device-v1",
                        request.batch_sha256,
                        request.base,
                        &request.generation.generation_sha256,
                        request.rewards,
                        request.clip_epsilon,
                        request.advantage_epsilon,
                        request.kl_coefficient,
                        request.loss_weight,
                        request.rng,
                        loss_sum,
                        mean_reward_sum,
                        reward_stddev_sum,
                        mean_kl_sum,
                        clipped_fraction_sum,
                        "backward-staged",
                    ),
                    rollouts * 2,
                ),
                examples,
                rollouts,
                loss_sum,
                mean_reward_sum,
                reward_stddev_sum,
                mean_kl_sum,
                clipped_fraction_sum,
            })
        }

        fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()> {
            close(learning_rate_scale, 1.0);
            self.optimizer_steps += 1;
            Ok(())
        }
    }

    #[test]
    fn post_training_configuration_is_strict_and_pinned() {
        let digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
        let config: PostTrainingConfig = serde_json::from_value(json!({
            "algorithm": "dpo",
            "reference": {
                "adapter": "hermes_checkpoint",
                "artifact": "reference",
                "sha256": digest
            }
        }))
        .unwrap();
        config.validate().unwrap();

        let noncanonical: PostTrainingConfig = serde_json::from_value(json!({
            "algorithm": "dpo",
            "reference": {
                "adapter": "hermes_checkpoint",
                "artifact": "reference",
                "sha256": digest.trim_start_matches("sha256:")
            }
        }))
        .unwrap();
        let error = noncanonical.validate().unwrap_err().to_string();
        assert!(error.contains("sha256:<64 lowercase hex>"), "{error}");

        let unpinned: PostTrainingConfig = serde_json::from_value(json!({
            "algorithm": "forward_kl",
            "teacher": {"adapter": "remote_teacher"}
        }))
        .unwrap();
        let error = unpinned.validate().unwrap_err().to_string();
        assert!(error.contains("sha256 or immutable revision"), "{error}");

        let unknown = serde_json::from_value::<PostTrainingConfig>(json!({
            "algorithm": "grpo",
            "group_size": 8,
            "clip": 0.1,
            "sampling": {"max_new_tokens": 32}
        }))
        .unwrap_err()
        .to_string();
        assert!(unknown.contains("clip"), "{unknown}");
    }

    #[test]
    fn revision_identity_binds_the_loader_and_its_parameters() {
        let base = FrozenModelSpec {
            adapter: "remote_model".to_owned(),
            artifact: None,
            sha256: None,
            revision: Some("commit-123".to_owned()),
            parameters: BTreeMap::from([("model".to_owned(), json!("owner/teacher"))]),
        };
        let identity = base.immutable_identity().unwrap();
        validate_sha256_identity(&identity, "revision identity").unwrap();
        assert_eq!(identity, base.clone().immutable_identity().unwrap());

        let mut another_adapter = base.clone();
        another_adapter.adapter = "another_remote_model".to_owned();
        assert_ne!(identity, another_adapter.immutable_identity().unwrap());

        let mut another_model = base.clone();
        another_model
            .parameters
            .insert("model".to_owned(), json!("owner/another-teacher"));
        assert_ne!(identity, another_model.immutable_identity().unwrap());

        let mut ambiguous = base;
        ambiguous.revision = Some(" commit-123".to_owned());
        let error = ambiguous.immutable_identity().unwrap_err().to_string();
        assert!(error.contains("trimmed"), "{error}");
    }

    #[test]
    fn native_post_training_rejects_unbounded_geometry_before_model_use() {
        let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.batch_size = Some(MAX_POST_TRAINING_UPDATE_EXAMPLES + 1);
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut no_hook = None;
        let error = drive_resumable_post_training_phase(
            &test_sha("bounded-geometry"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeding the limit"), "{error}");
        assert_eq!(policy.model_tokens, 0);
        assert_eq!(publisher.restore_calls, 0);

        let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Grpo);
        request.phase.sequence_length = Some(MAX_POST_TRAINING_SEQUENCE_TOKENS);
        let Some(PostTrainingConfig::Grpo {
            group_size,
            sampling,
            ..
        }) = request.phase.post_training.as_mut()
        else {
            panic!("test request is not GRPO");
        };
        *group_size = 9;
        sampling.max_new_tokens = MAX_POST_TRAINING_SEQUENCE_TOKENS;
        let error = validate_resumable_phase(&request.phase)
            .unwrap_err()
            .to_string();
        assert!(error.contains("rollout update exceeds"), "{error}");
    }

    #[test]
    fn native_post_training_revalidates_optimizer_scalars() {
        let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.loss_weight = Some(-1.0);
        let error = validate_resumable_phase(&request.phase)
            .unwrap_err()
            .to_string();
        assert!(error.contains("loss_weight"), "{error}");

        request.phase.loss_weight = Some(1.0);
        request.phase.learning_rate_scale = Some(f64::NAN);
        let error = validate_resumable_phase(&request.phase)
            .unwrap_err()
            .to_string();
        assert!(error.contains("learning_rate_scale"), "{error}");
    }

    #[test]
    fn native_post_training_rejects_unimplemented_memory_update_mode() {
        let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.memory_update_mode = Some(
            serde_json::from_value(json!({
                "type": "wake_only",
                "schedule": {
                    "clock": "optimizer_steps",
                    "terminal_consolidation": "distill_into_base_v1",
                    "tiers": [
                        {"id": "fast", "update_period": 2, "reserve_slots": 1},
                        {"id": "slow", "update_period": 4, "reserve_slots": 1}
                    ]
                }
            }))
            .unwrap(),
        );

        let error = validate_resumable_phase(&request.phase)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("does not implement memory_update_mode"),
            "{error}"
        );
    }

    #[test]
    fn malformed_device_batch_receipts_never_reach_optimizer_publication() {
        for (fault, expected) in [
            (TestDpoFault::MalformedReceipt, "device execution receipt"),
            (TestDpoFault::PartialSummary, "partial batch summary"),
            (TestDpoFault::TokenCountMismatch, "model-token counter"),
            (TestDpoFault::NonFiniteSummary, "empty or non-finite"),
            (TestDpoFault::NegativeLoss, "must be non-negative"),
        ] {
            let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
            request.phase.steps = Some(1);
            let mut publisher = TestUpdatePublisher::new(false);
            let mut runtime = TestDpoRuntime::new("candidate:dpo");
            runtime.fault = Some(fault);
            let mut adapters = PostTrainingPhaseRuntime::Dpo {
                runtime: &mut runtime,
            };
            let error = drive_resumable_post_training_phase(
                &test_sha("adversarial-device-receipt"),
                &request,
                &mut adapters,
                &mut publisher,
                &mut None,
                None,
                None,
                usize::MAX,
                &mut TestPhaseProgress::default(),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "{error}");
            assert_eq!(runtime.prepare_calls, 1);
            assert_eq!(runtime.optimizer_steps, 0);
            assert_eq!(publisher.apply_calls, 0);
        }
    }

    #[test]
    fn each_update_uses_one_device_batch_call_per_algorithm() {
        {
            let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
            request.phase.batch_size = Some(2);
            request.phase.steps = Some(1);
            let mut publisher = TestUpdatePublisher::new(false);
            let mut runtime = TestDpoRuntime::new("candidate:dpo");
            let mut adapters = PostTrainingPhaseRuntime::Dpo {
                runtime: &mut runtime,
            };
            let outcome = drive_resumable_post_training_phase(
                &test_sha("batched-dpo"),
                &request,
                &mut adapters,
                &mut publisher,
                &mut None,
                None,
                None,
                usize::MAX,
                &mut TestPhaseProgress::default(),
            )
            .unwrap();
            let ResumablePostTrainingOutcome::Complete { cursor, .. } = outcome else {
                panic!("batched DPO did not complete");
            };
            assert_eq!(runtime.prepare_calls, 1);
            assert_eq!(runtime.rng_ranges[0].len(), 4);
            assert_eq!(cursor.progress.examples, 2);
        }

        {
            let (directory, mut request) = test_request(TestPostTrainingAlgorithm::ForwardKl);
            let data = directory.path().join("instruction-distillation.jsonl");
            std::fs::write(
                &data,
                concat!(
                    "{\"instruction\":\"first\",\"response\":\"one\"}\n",
                    "{\"instruction\":\"second\",\"response\":\"two\"}\n"
                ),
            )
            .unwrap();
            request.phase.task = Some(TaskConfig::InstructionTuning {
                instruction: "Follow the instruction.".to_owned(),
            });
            request.phase.data = Some(data);
            request.phase.batch_size = Some(2);
            request.phase.steps = Some(1);
            let mut publisher = TestUpdatePublisher::new(false);
            let mut runtime = TestForwardKlRuntime::new();
            let mut adapters = PostTrainingPhaseRuntime::ForwardKl {
                runtime: &mut runtime,
            };
            drive_resumable_post_training_phase(
                &test_sha("batched-forward-kl"),
                &request,
                &mut adapters,
                &mut publisher,
                &mut None,
                None,
                None,
                usize::MAX,
                &mut TestPhaseProgress::default(),
            )
            .unwrap();
            assert_eq!(runtime.prepare_calls, 1);
            assert_eq!(runtime.rng_ranges[0].len(), 2);
        }

        {
            let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Grpo);
            request.phase.batch_size = Some(2);
            request.phase.steps = Some(1);
            let mut publisher = TestUpdatePublisher::new(false);
            let mut runtime = TestGrpoRuntime::new();
            let mut adapters = PostTrainingPhaseRuntime::Grpo {
                runtime: &mut runtime,
                verifier: &ExactAnswerVerifier,
            };
            drive_resumable_post_training_phase(
                &test_sha("batched-grpo"),
                &request,
                &mut adapters,
                &mut publisher,
                &mut None,
                None,
                None,
                usize::MAX,
                &mut TestPhaseProgress::default(),
            )
            .unwrap();
            assert_eq!(runtime.generation_calls, 1);
            assert_eq!(runtime.prepare_calls, 1);
            assert_eq!(runtime.generation_ranges[0].len(), 4);
            assert_eq!(runtime.scoring_ranges[0].len(), 4);
        }
    }

    #[test]
    fn grpo_rejects_oversized_generated_rollouts_before_scoring_or_backward() {
        let (_directory, mut request) = test_request(TestPostTrainingAlgorithm::Grpo);
        request.phase.steps = Some(1);
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestGrpoRuntime::new();
        policy.rollout_tokens = 33;
        let mut adapters = PostTrainingPhaseRuntime::Grpo {
            runtime: &mut policy,
            verifier: &ExactAnswerVerifier,
        };
        let mut no_hook = None;
        let error = drive_resumable_post_training_phase(
            &test_sha("oversized-rollout"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("configured token limit"), "{error}");
        assert_eq!(policy.model_tokens, 2, "only generation may have run");
        assert_eq!(policy.prepare_calls, 0);
        assert_eq!(publisher.apply_calls, 0);
    }

    #[test]
    fn frozen_local_artifacts_are_verified_and_changed_bytes_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teacher.safetensors");
        std::fs::write(&path, b"abc").unwrap();
        let spec = FrozenModelSpec {
            adapter: "hermes_checkpoint".to_owned(),
            artifact: Some(path.clone()),
            sha256: Some(
                "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                    .to_owned(),
            ),
            revision: None,
            parameters: BTreeMap::new(),
        };
        spec.verify_local_artifact().unwrap();
        std::fs::write(&path, b"abcd").unwrap();
        let error = spec.verify_local_artifact().unwrap_err().to_string();
        assert!(error.contains("sha256 mismatch"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn frozen_local_artifacts_reject_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("teacher.safetensors");
        let link = dir.path().join("teacher-link.safetensors");
        std::fs::write(&target, b"abc").unwrap();
        symlink(&target, &link).unwrap();
        let spec = FrozenModelSpec {
            adapter: "hermes_checkpoint".to_owned(),
            artifact: Some(link),
            sha256: Some(
                "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                    .to_owned(),
            ),
            revision: None,
            parameters: BTreeMap::new(),
        };
        let error = spec.verify_local_artifact().unwrap_err().to_string();
        assert!(error.contains("non-symlink regular file"), "{error}");
    }

    #[derive(Default)]
    struct TestPhaseProgress {
        checkpoint_calls: usize,
        fail_checkpoint_call: Option<usize>,
        durable: Option<serde_json::Value>,
        metrics: Vec<(MetricContext, MetricEvent)>,
        replace_input_after_metric: Option<(PathBuf, PathBuf)>,
    }

    impl PhaseProgressSink for TestPhaseProgress {
        fn checkpoint(&mut self, resume_state: serde_json::Value) -> Result<()> {
            self.checkpoint_calls += 1;
            if self.fail_checkpoint_call == Some(self.checkpoint_calls) {
                bail!("injected checkpoint interruption");
            }
            self.durable = Some(resume_state);
            Ok(())
        }

        fn metric(&mut self, context: MetricContext, event: MetricEvent) -> Result<()> {
            event.validate()?;
            self.metrics.push((context, event));
            if let Some((replacement, input)) = self.replace_input_after_metric.take() {
                fs::rename(replacement, input)?;
            }
            Ok(())
        }
    }

    struct TestUpdatePublisher {
        identity: String,
        fail_before_publication_once: bool,
        apply_calls: usize,
        restore_calls: usize,
        plans: BTreeMap<String, PreparedPostTrainingUpdate>,
        receipts: BTreeMap<String, PostTrainingUpdateReceipt>,
        replace_input_after_restore: Option<(PathBuf, PathBuf)>,
        replace_input_after_publication: Option<(PathBuf, PathBuf)>,
    }

    impl TestUpdatePublisher {
        fn new(fail_before_publication_once: bool) -> Self {
            Self {
                identity: test_sha("publisher-v1"),
                fail_before_publication_once,
                apply_calls: 0,
                restore_calls: 0,
                plans: BTreeMap::new(),
                receipts: BTreeMap::new(),
                replace_input_after_restore: None,
                replace_input_after_publication: None,
            }
        }
    }

    impl PostTrainingUpdatePublisher for TestUpdatePublisher {
        fn identity(&self) -> &str {
            &self.identity
        }

        fn restore_committed(
            &mut self,
            state: &PostTrainingCommittedState,
        ) -> Result<PostTrainingRestoreReceipt> {
            self.restore_calls += 1;
            if let Some((replacement, input)) = self.replace_input_after_restore.take() {
                fs::rename(replacement, input)?;
            }
            Ok(PostTrainingRestoreReceipt::for_state(
                self.identity.clone(),
                state,
            ))
        }

        fn publish_update(
            &mut self,
            plan: &PreparedPostTrainingUpdate,
            apply: &mut dyn FnMut() -> Result<()>,
        ) -> Result<PostTrainingUpdateReceipt> {
            if let Some(saved) = self.plans.get(&plan.transaction_id) {
                ensure!(saved == plan, "transaction id was reused for another plan");
                return Ok(self
                    .receipts
                    .get(&plan.transaction_id)
                    .expect("saved plan has receipt")
                    .clone());
            }
            if self.fail_before_publication_once {
                self.fail_before_publication_once = false;
                bail!("injected interruption before optimizer publication");
            }
            apply()?;
            self.apply_calls += 1;
            let checkpoint = ImmutableModelCheckpoint::new(
                format!("test://model/{}", plan.transaction_id),
                test_sha(&format!("model:{}", plan.transaction_id)),
            )?;
            let optimizer = ImmutableArtifact::new(
                format!("test://optimizer/{}", plan.transaction_id),
                test_sha(&format!("optimizer:{}", plan.transaction_id)),
            )?;
            let receipt = PostTrainingUpdateReceipt::new(
                plan,
                self.identity.clone(),
                checkpoint,
                optimizer,
                format!("test://receipt/{}", plan.transaction_id),
            )?;
            self.plans.insert(plan.transaction_id.clone(), plan.clone());
            self.receipts
                .insert(plan.transaction_id.clone(), receipt.clone());
            if let Some((replacement, input)) = self.replace_input_after_publication.take() {
                fs::rename(replacement, input)?;
            }
            Ok(receipt)
        }
    }

    struct TestBoundaryHook {
        identity: String,
        fail_before_commit_once: bool,
        calls: usize,
        receipts: BTreeMap<String, PostTrainingBoundaryReceipt>,
        cursors: BTreeMap<String, NativeSleepCheckpoint>,
    }

    impl TestBoundaryHook {
        fn new(fail_before_commit_once: bool) -> Self {
            Self {
                identity: test_sha("periodic-sleep-hook-v1"),
                fail_before_commit_once,
                calls: 0,
                receipts: BTreeMap::new(),
                cursors: BTreeMap::new(),
            }
        }
    }

    fn test_native_sleep_cursor(
        request: &PostTrainingBoundaryRequest<'_>,
    ) -> Result<NativeSleepCheckpoint> {
        let mut sleep = crate::sleep::SleepState::new(&request.config.schedule, 1)?;
        sleep.clock = request.clock_before.selected(request.config.schedule.clock);
        for (saved, configured) in sleep.tiers.iter_mut().zip(&request.config.schedule.tiers) {
            let last_boundary = sleep.clock / configured.update_period * configured.update_period;
            saved.last_update_clock = last_boundary;
            saved.last_boundary_clock = last_boundary;
        }
        let tiers = request
            .config
            .schedule
            .tiers
            .iter()
            .enumerate()
            .map(|(tier, configured)| crate::sleep::TierOptimizerScope {
                tier,
                tier_id: configured.id.clone(),
                parameter_ids: Vec::new(),
                update_clock: sleep.clock,
                transfer_clock: sleep.clock,
                accumulated_micro_steps: 0,
                generation: 0,
                transfer_generation: 0,
                artifact: None,
            })
            .collect();
        Ok(NativeSleepCheckpoint {
            version: crate::native_sleep::NATIVE_SLEEP_CHECKPOINT_VERSION,
            workflow_signature: request.workflow_signature.to_owned(),
            phase_name: request.phase_name.to_owned(),
            input_checkpoint: crate::native_sleep::NativeCheckpointRef::new(
                request.input.model.uri().to_owned(),
                request.input.model.sha256().to_owned(),
            )?,
            live_checkpoint: crate::native_sleep::NativeCheckpointRef::new(
                request.input.model.uri().to_owned(),
                request.input.model.sha256().to_owned(),
            )?,
            retention_suite: crate::native_sleep::PinnedNativeArtifact {
                path: request
                    .config
                    .retention_suite
                    .to_string_lossy()
                    .into_owned(),
                sha256: request.config.retention_suite_sha256.clone(),
            },
            wake_context_journal: None,
            sleep,
            optimizer_scopes: crate::sleep::MemoryOptimizerScopes {
                wake_parameter_ids: Vec::new(),
                tiers,
            },
        })
    }

    impl PostTrainingBoundaryHook for TestBoundaryHook {
        fn identity(&self) -> &str {
            &self.identity
        }

        fn drive_boundary(
            &mut self,
            request: &PostTrainingBoundaryRequest<'_>,
            resume: Option<NativeSleepCheckpoint>,
            progress: &mut dyn NativeSleepProgressSink,
        ) -> Result<PostTrainingBoundaryReceipt> {
            self.calls += 1;
            if let Some(receipt) = self.receipts.get(request.transaction_id) {
                ensure!(
                    receipt.input_model_sha256 == request.input.model.sha256(),
                    "boundary transaction was retried with another input"
                );
                progress.persist(
                    self.cursors
                        .get(request.transaction_id)
                        .expect("saved boundary receipt has native cursor"),
                )?;
                return Ok(receipt.clone());
            }
            if self.fail_before_commit_once {
                self.fail_before_commit_once = false;
                bail!("injected interruption before periodic-sleep commit");
            }
            let mut cursor = match resume {
                Some(cursor) => cursor,
                None => test_native_sleep_cursor(request)?,
            };
            let target = request.clock_after.selected(request.config.schedule.clock);
            if cursor.sleep.clock < target {
                cursor
                    .sleep
                    .advance_clock(&request.config.schedule, target)?;
            }
            let had_due = !cursor.sleep.due_senders.is_empty();
            while !cursor.sleep.due_senders.is_empty() {
                let sender = cursor.sleep.due_senders.remove(0);
                let clock = cursor.sleep.due_clocks.remove(0);
                cursor.sleep.tiers[sender].last_update_clock = clock;
                cursor.sleep.tiers[sender].last_boundary_clock = clock;
            }
            let state = if had_due {
                PostTrainingCommittedState {
                    model: ImmutableModelCheckpoint::new(
                        format!("test://sleep/{}", request.clock_after.optimizer_steps),
                        test_sha(&format!("sleep:{}", request.transaction_id)),
                    )?,
                    optimizer: request.input.optimizer.clone(),
                }
            } else {
                request.input.clone()
            };
            cursor.live_checkpoint = crate::native_sleep::NativeCheckpointRef::new(
                state.model.uri().to_owned(),
                state.model.sha256().to_owned(),
            )?;
            progress.persist(&cursor)?;
            let native_sha256 = canonical_sha256(&cursor)?;
            let clock = PostTrainingClockReceipt::new(
                self.identity.clone(),
                state.model.sha256().to_owned(),
                request.clock_after.optimizer_steps,
                request.clock_after.model_tokens,
            )?;
            let receipt = PostTrainingBoundaryReceipt::new(
                request.transaction_id,
                self.identity.clone(),
                request.input,
                state,
                clock,
                native_sha256,
            )?;
            self.receipts
                .insert(request.transaction_id.to_owned(), receipt.clone());
            self.cursors
                .insert(request.transaction_id.to_owned(), cursor);
            Ok(receipt)
        }
    }

    struct TestNativePostTrainingSleepRuntime {
        identity: String,
        due: Rc<RefCell<Vec<(u64, usize)>>>,
        clocks: Option<Rc<RefCell<BTreeMap<String, PostTrainingClockValues>>>>,
    }

    impl NativePostTrainingSleepRuntime for TestNativePostTrainingSleepRuntime {
        fn identity(&self) -> &str {
            &self.identity
        }

        fn restore_boundary_cursor(
            &mut self,
            request: &PostTrainingBoundaryRequest<'_>,
        ) -> Result<NativeSleepCheckpoint> {
            test_native_sleep_cursor(request)
        }

        fn advance_and_drain(
            &mut self,
            request: &PostTrainingBoundaryRequest<'_>,
            checkpoint: &mut NativeSleepCheckpoint,
            progress: &mut dyn NativeSleepProgressSink,
        ) -> Result<PostTrainingCommittedState> {
            let target = request.clock_after.selected(request.config.schedule.clock);
            if checkpoint.sleep.clock < target {
                checkpoint
                    .sleep
                    .advance_clock(&request.config.schedule, target)?;
                progress.persist(checkpoint)?;
            }
            let mut changed = checkpoint.live_checkpoint.sha256 != request.input.model.sha256();
            while let (Some(sender), Some(clock)) = (
                checkpoint.sleep.due_senders.first().copied(),
                checkpoint.sleep.due_clocks.first().copied(),
            ) {
                self.due.borrow_mut().push((clock, sender));
                checkpoint.sleep.due_senders.remove(0);
                checkpoint.sleep.due_clocks.remove(0);
                checkpoint.sleep.tiers[sender].last_update_clock = clock;
                checkpoint.sleep.tiers[sender].last_boundary_clock = clock;
                changed = true;
                checkpoint.live_checkpoint = crate::native_sleep::NativeCheckpointRef::new(
                    format!(
                        "test://native-sleep/{}",
                        request.clock_after.optimizer_steps
                    ),
                    test_sha(&format!("native-sleep:{}", request.transaction_id)),
                )?;
                progress.persist(checkpoint)?;
            }
            let model = if changed {
                ImmutableModelCheckpoint::new(
                    checkpoint.live_checkpoint.uri.clone(),
                    checkpoint.live_checkpoint.sha256.clone(),
                )?
            } else {
                request.input.model.clone()
            };
            checkpoint.live_checkpoint = crate::native_sleep::NativeCheckpointRef::new(
                model.uri().to_owned(),
                model.sha256().to_owned(),
            )?;
            if let Some(clocks) = &self.clocks {
                clocks
                    .borrow_mut()
                    .insert(model.sha256().to_owned(), request.clock_after);
            }
            Ok(PostTrainingCommittedState {
                model,
                optimizer: request.input.optimizer.clone(),
            })
        }
    }

    struct TestHostPostTrainingFactory {
        identity: String,
        registered_controller_identity: String,
        lent_controller_identity: String,
        due: Rc<RefCell<Vec<(u64, usize)>>>,
        clocks: Rc<RefCell<BTreeMap<String, PostTrainingClockValues>>>,
    }

    impl NativePostTrainingContextFactory for TestHostPostTrainingFactory {
        fn identity(&self) -> &str {
            &self.identity
        }

        fn periodic_sleep_identity(&self) -> Option<&str> {
            Some(&self.registered_controller_identity)
        }

        fn with_context(
            &mut self,
            request: &PhaseExecutionRequest,
            operation: &mut dyn for<'a> FnMut(&mut PostTrainingExecutionContext<'a>) -> Result<()>,
        ) -> Result<()> {
            let input = request
                .input_checkpoint
                .as_ref()
                .context("test host context has no input checkpoint")?;
            let clock = self
                .clocks
                .borrow()
                .get(input.sha256())
                .copied()
                .context("test host checkpoint has no authenticated cumulative clock")?;
            let starting_clock = PostTrainingClockReceipt::new(
                self.lent_controller_identity.clone(),
                input.sha256().to_owned(),
                clock.optimizer_steps,
                clock.model_tokens,
            )?;
            let runtime = TestNativePostTrainingSleepRuntime {
                identity: self.lent_controller_identity.clone(),
                due: self.due.clone(),
                clocks: Some(self.clocks.clone()),
            };
            let mut controller = NativePostTrainingBoundaryController::new(runtime)?;
            let mut publisher = TestUpdatePublisher::new(false);
            match request
                .phase
                .post_training
                .as_ref()
                .context("test host phase has no post-training config")?
            {
                PostTrainingConfig::Dpo { .. } => {
                    let mut policy = TestDpoRuntime::new("candidate:dpo");
                    let mut context = PostTrainingExecutionContext {
                        runtime: PostTrainingPhaseRuntime::Dpo {
                            runtime: &mut policy,
                        },
                        publisher: &mut publisher,
                        boundary_hook: Some(&mut controller),
                        starting_clock: Some(starting_clock),
                    };
                    operation(&mut context)
                }
                PostTrainingConfig::ForwardKl { .. } => {
                    let mut policy = TestForwardKlRuntime::new();
                    let mut context = PostTrainingExecutionContext {
                        runtime: PostTrainingPhaseRuntime::ForwardKl {
                            runtime: &mut policy,
                        },
                        publisher: &mut publisher,
                        boundary_hook: Some(&mut controller),
                        starting_clock: Some(starting_clock),
                    };
                    operation(&mut context)
                }
                PostTrainingConfig::Grpo { .. } => {
                    let mut policy = TestGrpoRuntime::new();
                    let mut context = PostTrainingExecutionContext {
                        runtime: PostTrainingPhaseRuntime::Grpo {
                            runtime: &mut policy,
                            verifier: &ExactAnswerVerifier,
                        },
                        publisher: &mut publisher,
                        boundary_hook: Some(&mut controller),
                        starting_clock: Some(starting_clock),
                    };
                    operation(&mut context)
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    enum TestPostTrainingAlgorithm {
        Dpo,
        ForwardKl,
        Grpo,
    }

    fn test_sha(label: &str) -> String {
        sha256_identity(label.as_bytes())
    }

    fn test_request(
        algorithm: TestPostTrainingAlgorithm,
    ) -> (tempfile::TempDir, PhaseExecutionRequest) {
        let dir = tempfile::tempdir().unwrap();
        let (data, phase) = match algorithm {
            TestPostTrainingAlgorithm::Dpo => {
                let artifact = dir.path().join("reference.safetensors");
                std::fs::write(&artifact, b"abc").unwrap();
                let data = dir.path().join("preference.jsonl");
                std::fs::write(
                    &data,
                    concat!(
                        "{\"prompt\":\"p1\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n",
                        "{\"prompt\":\"p2\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n"
                    ),
                )
                .unwrap();
                let phase: PhaseV2 = serde_json::from_value(json!({
                    "name": "native-dpo",
                    "type": "preference",
                    "task": {"type": "pairwise_preference"},
                    "data": data,
                    "sequence_length": 128,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 2,
                    "learning_rate_scale": 0.5,
                    "post_training": {
                        "algorithm": "dpo",
                        "reference": {
                            "adapter": "test",
                            "artifact": artifact,
                            "sha256": ABC_SHA256
                        }
                    }
                }))
                .unwrap();
                (data, phase)
            }
            TestPostTrainingAlgorithm::ForwardKl => {
                let artifact = dir.path().join("teacher.safetensors");
                std::fs::write(&artifact, b"abc").unwrap();
                let data = dir.path().join("distill.txt");
                std::fs::write(&data, "first\nsecond\n").unwrap();
                let phase: PhaseV2 = serde_json::from_value(json!({
                    "name": "native-forward-kl",
                    "type": "distillation",
                    "task": {"type": "causal_lm"},
                    "data": data,
                    "sequence_length": 128,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 2,
                    "post_training": {
                        "algorithm": "forward_kl",
                        "teacher": {
                            "adapter": "test",
                            "artifact": artifact,
                            "sha256": ABC_SHA256
                        }
                    }
                }))
                .unwrap();
                (data, phase)
            }
            TestPostTrainingAlgorithm::Grpo => {
                let data = dir.path().join("rl.jsonl");
                std::fs::write(
                    &data,
                    concat!(
                        "{\"prompt\":\"one\",\"verifier_payload\":{},\"reference_answer\":\"ok\"}\n",
                        "{\"prompt\":\"two\",\"verifier_payload\":{},\"reference_answer\":\"ok\"}\n"
                    ),
                )
                .unwrap();
                let phase: PhaseV2 = serde_json::from_value(json!({
                    "name": "native-grpo",
                    "type": "rl",
                    "task": {
                        "type": "verifiable_rl",
                        "verifier": {"adapter": "exact_answer"}
                    },
                    "data": data,
                    "sequence_length": 128,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 2,
                    "post_training": {
                        "algorithm": "grpo",
                        "group_size": 2,
                        "kl_coefficient": 0.0,
                        "sampling": {"max_new_tokens": 32}
                    }
                }))
                .unwrap();
                (data, phase)
            }
        };
        assert_eq!(phase.data.as_deref(), Some(data.as_path()));
        let request = PhaseExecutionRequest {
            phase_index: 3,
            phase,
            input_checkpoint: Some(
                ImmutableModelCheckpoint::new("test://model/input", test_sha("input-model"))
                    .unwrap(),
            ),
            resume_state: None,
        };
        (dir, request)
    }

    fn install_periodic_sleep(request: &mut PhaseExecutionRequest) {
        let workflow: serde_json::Value =
            serde_json::from_str(include_str!("../workflow.education.example.json")).unwrap();
        let sleep = workflow
            .get("phases")
            .and_then(serde_json::Value::as_array)
            .and_then(|phases| phases.first())
            .and_then(|phase| phase.get("periodic_sleep"))
            .cloned()
            .expect("education workflow has periodic sleep");
        request.phase.periodic_sleep = Some(serde_json::from_value(sleep).unwrap());
    }

    fn install_positive_grpo_reference(directory: &Path, request: &mut PhaseExecutionRequest) {
        let artifact = directory.join("grpo-reference.safetensors");
        fs::write(&artifact, b"abc").unwrap();
        let PostTrainingConfig::Grpo {
            kl_coefficient,
            reference,
            ..
        } = request.phase.post_training.as_mut().unwrap()
        else {
            panic!("positive GRPO reference installed on a non-GRPO phase");
        };
        *kl_coefficient = 0.04;
        *reference = Some(FrozenModelSpec {
            adapter: "test".to_owned(),
            artifact: Some(artifact),
            sha256: Some(ABC_SHA256.to_owned()),
            revision: None,
            parameters: BTreeMap::new(),
        });
    }

    fn run_interruption_case(algorithm: TestPostTrainingAlgorithm, after_publication: bool) {
        run_interruption_case_with_grpo_reference(algorithm, after_publication, false);
    }

    fn run_interruption_case_with_grpo_reference(
        algorithm: TestPostTrainingAlgorithm,
        after_publication: bool,
        positive_grpo_kl: bool,
    ) {
        assert!(
            !positive_grpo_kl || matches!(algorithm, TestPostTrainingAlgorithm::Grpo),
            "a positive GRPO reference is only valid for GRPO"
        );
        let (directory, mut request) = test_request(algorithm);
        if positive_grpo_kl {
            install_positive_grpo_reference(directory.path(), &mut request);
        }
        let workflow = test_sha("workflow-v2");
        let mut publisher = TestUpdatePublisher::new(!after_publication);
        let mut first_sink = TestPhaseProgress {
            fail_checkpoint_call: after_publication.then_some(2),
            ..TestPhaseProgress::default()
        };
        let mut no_hook = None;
        let mut dpo_policy = TestDpoRuntime::new("candidate:dpo");
        let mut student = TestForwardKlRuntime::new();
        let mut grpo_policy = if positive_grpo_kl {
            TestGrpoRuntime::with_reference()
        } else {
            TestGrpoRuntime::new()
        };
        let mut adapters = match algorithm {
            TestPostTrainingAlgorithm::Dpo => PostTrainingPhaseRuntime::Dpo {
                runtime: &mut dpo_policy,
            },
            TestPostTrainingAlgorithm::ForwardKl => PostTrainingPhaseRuntime::ForwardKl {
                runtime: &mut student,
            },
            TestPostTrainingAlgorithm::Grpo => PostTrainingPhaseRuntime::Grpo {
                runtime: &mut grpo_policy,
                verifier: &ExactAnswerVerifier,
            },
        };
        let first = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut first_sink,
        );
        assert!(first.is_err());
        let durable: PostTrainingResumeEnvelope = serde_json::from_value(
            first_sink
                .durable
                .clone()
                .expect("prepared cursor must be durable"),
        )
        .unwrap();
        let (durable_cursor, boundary) = durable.clone().into_parts().unwrap();
        assert!(durable_cursor.pending.is_some());
        assert!(boundary.is_none());
        if after_publication {
            assert_eq!(publisher.apply_calls, 1);
        } else {
            assert_eq!(publisher.apply_calls, 0);
        }

        let mut resumed_sink = TestPhaseProgress::default();
        let outcome = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            Some(durable),
            usize::MAX,
            &mut resumed_sink,
        )
        .unwrap();
        let ResumablePostTrainingOutcome::Complete { cursor, report } = outcome else {
            panic!("resumed two-step test phase did not complete");
        };
        assert_eq!(publisher.apply_calls, 2);
        assert_eq!(cursor.progress.optimizer_steps, 2);
        assert_eq!(cursor.progress.examples, 2);
        let expected_rng_counter = match algorithm {
            TestPostTrainingAlgorithm::Dpo => 4,
            TestPostTrainingAlgorithm::ForwardKl => 2,
            TestPostTrainingAlgorithm::Grpo => 8,
        };
        assert_eq!(cursor.rng.counter, expected_rng_counter);
        assert_eq!(report.optimizer_steps, 2);
        assert_eq!(resumed_sink.metrics.len(), 2);
        if positive_grpo_kl {
            assert!(report.metrics["mean_kl"] > 0.0);
            let expected_reference = test_sha("abc");
            assert_eq!(
                report.frozen_model_identity.as_deref(),
                Some(expected_reference.as_str())
            );
        }
        match algorithm {
            TestPostTrainingAlgorithm::Dpo => {
                assert_eq!(dpo_policy.prepare_calls, 3);
                assert_eq!(dpo_policy.rng_ranges[0], dpo_policy.rng_ranges[1]);
                assert_eq!(dpo_policy.rng_ranges[2].start, 2);
            }
            TestPostTrainingAlgorithm::ForwardKl => {
                assert_eq!(student.prepare_calls, 3);
                assert_eq!(student.rng_ranges[0], student.rng_ranges[1]);
                assert_eq!(student.rng_ranges[2].start, 1);
            }
            TestPostTrainingAlgorithm::Grpo => {
                assert_eq!(grpo_policy.generation_calls, 3);
                assert_eq!(grpo_policy.prepare_calls, 3);
                assert_eq!(
                    grpo_policy.generation_ranges[0],
                    grpo_policy.generation_ranges[1]
                );
                assert_eq!(grpo_policy.scoring_ranges[0], grpo_policy.scoring_ranges[1]);
                assert_eq!(grpo_policy.generation_ranges[2].start, 4);
            }
        }
    }

    fn run_periodic_boundary_interruption_case(
        algorithm: TestPostTrainingAlgorithm,
        after_boundary_publication: bool,
    ) {
        let (_dir, mut request) = test_request(algorithm);
        install_periodic_sleep(&mut request);
        let workflow = test_sha("workflow-v2-periodic-sleep");
        let mut publisher = TestUpdatePublisher::new(false);
        let mut hook = TestBoundaryHook::new(!after_boundary_publication);
        let starting_clock = PostTrainingClockReceipt::new(
            hook.identity.clone(),
            request
                .input_checkpoint
                .as_ref()
                .unwrap()
                .sha256()
                .to_owned(),
            58_250,
            12_000_000,
        )
        .unwrap();
        let mut first_sink = TestPhaseProgress {
            fail_checkpoint_call: after_boundary_publication.then_some(4),
            ..TestPhaseProgress::default()
        };
        let mut dpo_policy = TestDpoRuntime::new("candidate:dpo");
        let mut student = TestForwardKlRuntime::new();
        let mut grpo_policy = TestGrpoRuntime::new();
        let mut adapters = match algorithm {
            TestPostTrainingAlgorithm::Dpo => PostTrainingPhaseRuntime::Dpo {
                runtime: &mut dpo_policy,
            },
            TestPostTrainingAlgorithm::ForwardKl => PostTrainingPhaseRuntime::ForwardKl {
                runtime: &mut student,
            },
            TestPostTrainingAlgorithm::Grpo => PostTrainingPhaseRuntime::Grpo {
                runtime: &mut grpo_policy,
                verifier: &ExactAnswerVerifier,
            },
        };
        let mut boundary_hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut hook);
        let first = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut boundary_hook,
            Some(&starting_clock),
            None,
            usize::MAX,
            &mut first_sink,
        );
        assert!(first.is_err());
        let durable: PostTrainingResumeEnvelope = serde_json::from_value(
            first_sink
                .durable
                .clone()
                .expect("prepared pre-boundary cursor must be durable"),
        )
        .unwrap();
        let (durable_cursor, boundary) = durable.clone().into_parts().unwrap();
        assert!(durable_cursor.pending.is_some());
        assert!(boundary.is_some());
        assert_eq!(publisher.apply_calls, 1);
        assert_eq!(hook.receipts.len(), usize::from(after_boundary_publication));

        let mut resumed_sink = TestPhaseProgress::default();
        let mut boundary_hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut hook);
        let outcome = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut boundary_hook,
            Some(&starting_clock),
            Some(durable),
            usize::MAX,
            &mut resumed_sink,
        )
        .unwrap();
        let ResumablePostTrainingOutcome::Complete { cursor, report } = outcome else {
            panic!("resumed periodic post-training phase did not complete");
        };
        assert_eq!(publisher.apply_calls, 2, "optimizer update was duplicated");
        assert_eq!(hook.receipts.len(), 2, "a sleep boundary was skipped");
        assert_eq!(
            hook.calls, 3,
            "the interrupted boundary was not retried transactionally"
        );
        assert_eq!(cursor.progress.optimizer_steps, 2);
        assert_eq!(report.optimizer_steps, 2);
        assert!(
            cursor.committed.model.uri().starts_with("test://model/"),
            "phase completed from a pre-sleep optimizer checkpoint"
        );
        assert!(cursor.last_boundary_receipt.is_some());
    }

    #[cfg(unix)]
    #[test]
    fn late_input_replacement_cannot_commit_a_published_update() {
        let (dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.steps = Some(1);
        let input = request.phase.data.clone().unwrap();
        let replacement = dir.path().join("replacement-preference.jsonl");
        fs::write(
            &replacement,
            concat!(
                "{\"prompt\":\"changed-1\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n",
                "{\"prompt\":\"changed-2\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n"
            ),
        )
        .unwrap();
        let mut publisher = TestUpdatePublisher::new(false);
        publisher.replace_input_after_publication = Some((replacement, input));
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut no_hook = None;
        let mut progress = TestPhaseProgress::default();
        let error = drive_resumable_post_training_phase(
            &test_sha("late-input-replacement"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut progress,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("changed after it was authenticated"),
            "{error}"
        );
        assert_eq!(publisher.apply_calls, 1);
        assert_eq!(
            publisher.restore_calls, 1,
            "changed input should reject before restoring the published update"
        );
        let envelope: PostTrainingResumeEnvelope =
            serde_json::from_value(progress.durable.unwrap()).unwrap();
        let (cursor, boundary) = envelope.into_parts().unwrap();
        assert!(boundary.is_none());
        assert!(cursor.pending.is_some());
        assert_eq!(cursor.progress.optimizer_steps, 0);
        assert_eq!(cursor.committed.model, cursor.input_checkpoint);
    }

    #[cfg(unix)]
    #[test]
    fn metric_callback_input_replacement_cannot_commit_the_phase_cursor() {
        let (dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.steps = Some(1);
        let input = request.phase.data.clone().unwrap();
        let replacement = dir.path().join("replacement-after-metric.jsonl");
        fs::write(
            &replacement,
            concat!(
                "{\"prompt\":\"changed-1\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n",
                "{\"prompt\":\"changed-2\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n"
            ),
        )
        .unwrap();
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut no_hook = None;
        let mut progress = TestPhaseProgress {
            replace_input_after_metric: Some((replacement, input)),
            ..TestPhaseProgress::default()
        };
        let error = drive_resumable_post_training_phase(
            &test_sha("metric-input-replacement"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut progress,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("changed after it was authenticated"),
            "{error}"
        );
        assert_eq!(progress.checkpoint_calls, 1);
        let envelope: PostTrainingResumeEnvelope =
            serde_json::from_value(progress.durable.unwrap()).unwrap();
        let (cursor, boundary) = envelope.into_parts().unwrap();
        assert!(boundary.is_none());
        assert!(cursor.pending.is_some());
        assert_eq!(cursor.progress.optimizer_steps, 0);
    }

    #[cfg(unix)]
    #[test]
    fn completed_resume_rechecks_input_after_restoring_its_final_checkpoint() {
        let (dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.steps = Some(1);
        let workflow = test_sha("completed-resume-input-replacement");
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut no_hook = None;
        let complete = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap();
        let ResumablePostTrainingOutcome::Complete { cursor, .. } = complete else {
            panic!("one-step phase did not complete");
        };

        let input = request.phase.data.clone().unwrap();
        let replacement = dir.path().join("replacement-on-final-restore.jsonl");
        fs::write(
            &replacement,
            concat!(
                "{\"prompt\":\"changed-1\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n",
                "{\"prompt\":\"changed-2\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n"
            ),
        )
        .unwrap();
        publisher.replace_input_after_restore = Some((replacement, input));

        let error = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            Some(PostTrainingResumeEnvelope::wake(cursor)),
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("changed after it was authenticated"),
            "{error}"
        );
    }

    #[test]
    fn completed_exact_eof_cursor_replays_without_another_update() {
        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Dpo);
        let workflow = test_sha("completed-exact-eof-resume");
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut no_hook = None;
        let complete = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap();
        let ResumablePostTrainingOutcome::Complete { cursor, report } = complete else {
            panic!("two-record phase did not complete");
        };
        assert_eq!(
            cursor.position,
            PostTrainingRecordPosition {
                epoch: 1,
                record: 0
            }
        );
        assert_eq!(report.optimizer_steps, 2);

        let replayed = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            Some(PostTrainingResumeEnvelope::wake(cursor.clone())),
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap();
        let ResumablePostTrainingOutcome::Complete {
            cursor: replayed,
            report,
        } = replayed
        else {
            panic!("completed exact-EOF cursor unexpectedly yielded");
        };
        assert_eq!(replayed, cursor);
        assert_eq!(report.optimizer_steps, 2);
        assert_eq!(
            publisher.apply_calls, 2,
            "completed replay applied an update"
        );
    }

    #[test]
    fn step_limited_phase_does_not_parse_unconsumed_lookahead() {
        let (_dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.steps = Some(1);
        fs::write(
            request.phase.data.as_ref().unwrap(),
            concat!(
                "{\"prompt\":\"used\",\"chosen\":\"yes\",\"rejected\":\"no\"}\n",
                "this record is outside the configured step limit\n"
            ),
        )
        .unwrap();
        let workflow = test_sha("step-limited-raw-lookahead");
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut no_hook = None;
        let complete = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap();
        let ResumablePostTrainingOutcome::Complete { cursor, report } = complete else {
            panic!("one-step phase did not complete");
        };
        assert_eq!(cursor.position.epoch, 0);
        assert_eq!(cursor.position.record, 1);
        assert_eq!(report.optimizer_steps, 1);
        assert_eq!(publisher.apply_calls, 1);
    }

    #[test]
    fn dpo_resumes_before_optimizer_publication_without_double_apply() {
        run_interruption_case(TestPostTrainingAlgorithm::Dpo, false);
    }

    #[test]
    fn dpo_resumes_after_optimizer_publication_without_double_apply() {
        run_interruption_case(TestPostTrainingAlgorithm::Dpo, true);
    }

    #[test]
    fn forward_kl_resumes_before_optimizer_publication_without_double_apply() {
        run_interruption_case(TestPostTrainingAlgorithm::ForwardKl, false);
    }

    #[test]
    fn forward_kl_resumes_after_optimizer_publication_without_double_apply() {
        run_interruption_case(TestPostTrainingAlgorithm::ForwardKl, true);
    }

    #[test]
    fn grpo_resumes_before_optimizer_publication_without_double_apply() {
        run_interruption_case(TestPostTrainingAlgorithm::Grpo, false);
    }

    #[test]
    fn grpo_resumes_after_optimizer_publication_without_double_apply() {
        run_interruption_case(TestPostTrainingAlgorithm::Grpo, true);
    }

    #[test]
    fn positive_kl_grpo_resumes_before_optimizer_publication_exactly() {
        run_interruption_case_with_grpo_reference(TestPostTrainingAlgorithm::Grpo, false, true);
    }

    #[test]
    fn positive_kl_grpo_resumes_after_optimizer_publication_exactly() {
        run_interruption_case_with_grpo_reference(TestPostTrainingAlgorithm::Grpo, true, true);
    }

    #[test]
    fn prepared_update_rejects_rehashed_but_forged_rng_geometry() {
        let (_directory, request) = test_request(TestPostTrainingAlgorithm::Dpo);
        let mut publisher = TestUpdatePublisher::new(true);
        let mut runtime = TestDpoRuntime::new("candidate:dpo");
        let mut phase_runtime = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut runtime,
        };
        let mut progress = TestPhaseProgress::default();
        let error = drive_resumable_post_training_phase(
            &test_sha("forged-rng-geometry"),
            &request,
            &mut phase_runtime,
            &mut publisher,
            &mut None,
            None,
            None,
            usize::MAX,
            &mut progress,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("injected interruption"), "{error}");
        let envelope: PostTrainingResumeEnvelope =
            serde_json::from_value(progress.durable.unwrap()).unwrap();
        let (mut cursor, boundary) = envelope.into_parts().unwrap();
        assert!(boundary.is_none());
        let pending = cursor.pending.as_mut().unwrap();
        pending.rng_end += 1;
        pending.transaction_id = pending.computed_transaction_id().unwrap();
        let error = cursor.validate_internal().unwrap_err().to_string();
        assert!(
            error.contains("exactly two model RNG substreams"),
            "{error}"
        );
    }

    #[test]
    fn dpo_periodic_sleep_resumes_before_and_after_boundary_publication() {
        run_periodic_boundary_interruption_case(TestPostTrainingAlgorithm::Dpo, false);
        run_periodic_boundary_interruption_case(TestPostTrainingAlgorithm::Dpo, true);
    }

    #[test]
    fn forward_kl_periodic_sleep_resumes_before_and_after_boundary_publication() {
        run_periodic_boundary_interruption_case(TestPostTrainingAlgorithm::ForwardKl, false);
        run_periodic_boundary_interruption_case(TestPostTrainingAlgorithm::ForwardKl, true);
    }

    #[test]
    fn grpo_periodic_sleep_resumes_before_and_after_boundary_publication() {
        run_periodic_boundary_interruption_case(TestPostTrainingAlgorithm::Grpo, false);
        run_periodic_boundary_interruption_case(TestPostTrainingAlgorithm::Grpo, true);
    }

    #[test]
    fn first_party_controller_uses_exact_model_token_clock_and_drains_coincident_tiers() {
        let (_dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.steps = Some(1);
        install_periodic_sleep(&mut request);
        request
            .phase
            .periodic_sleep
            .as_mut()
            .unwrap()
            .schedule
            .clock = UpdateClock::ModelTokens;
        let identity = test_sha("native-post-training-controller");
        let due = Rc::new(RefCell::new(Vec::new()));
        let runtime = TestNativePostTrainingSleepRuntime {
            identity: identity.clone(),
            due: due.clone(),
            clocks: None,
        };
        let mut controller = NativePostTrainingBoundaryController::new(runtime).unwrap();
        let starting_clock = PostTrainingClockReceipt::new(
            identity,
            request
                .input_checkpoint
                .as_ref()
                .unwrap()
                .sha256()
                .to_owned(),
            57_000,
            398,
        )
        .unwrap();
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut controller);
        let outcome = drive_resumable_post_training_phase(
            &test_sha("model-token-workflow"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut hook,
            Some(&starting_clock),
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap();
        let ResumablePostTrainingOutcome::Complete { cursor, .. } = outcome else {
            panic!("one-step controller test did not complete");
        };
        assert_eq!(cursor.clock.as_ref().unwrap().optimizer_steps, 57_001);
        assert_eq!(cursor.clock.as_ref().unwrap().model_tokens, 400);
        assert_eq!(&*due.borrow(), &[(400, 0), (400, 1)]);
    }

    #[test]
    fn model_token_jump_rejects_before_publishing_one_update_for_multiple_boundaries() {
        let (_dir, mut request) = test_request(TestPostTrainingAlgorithm::Grpo);
        request.phase.steps = Some(1);
        install_periodic_sleep(&mut request);
        let schedule = &mut request.phase.periodic_sleep.as_mut().unwrap().schedule;
        schedule.clock = UpdateClock::ModelTokens;
        schedule.tiers[0].update_period = 1;
        schedule.tiers[1].update_period = 2;
        schedule.tiers[2].update_period = 4;
        let identity = test_sha("coarse-model-token-controller");
        let due = Rc::new(RefCell::new(Vec::new()));
        let runtime = TestNativePostTrainingSleepRuntime {
            identity: identity.clone(),
            due: due.clone(),
            clocks: None,
        };
        let mut controller = NativePostTrainingBoundaryController::new(runtime).unwrap();
        let starting_clock = PostTrainingClockReceipt::new(
            identity,
            request
                .input_checkpoint
                .as_ref()
                .unwrap()
                .sha256()
                .to_owned(),
            8_000,
            0,
        )
        .unwrap();
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestGrpoRuntime::new();
        let mut adapters = PostTrainingPhaseRuntime::Grpo {
            runtime: &mut policy,
            verifier: &ExactAnswerVerifier,
        };
        let mut hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut controller);
        let error = drive_resumable_post_training_phase(
            &test_sha("coarse-model-token-workflow"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut hook,
            Some(&starting_clock),
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("crosses 4 `fast` boundaries"), "{error}");
        assert_eq!(policy.optimizer_steps, 0);
        assert!(due.borrow().is_empty());
    }

    #[test]
    fn first_party_controller_resumes_every_wrapped_native_persistence_window() {
        for failed_checkpoint in 3..=7 {
            let (_dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
            request.phase.steps = Some(1);
            install_periodic_sleep(&mut request);
            let identity = test_sha("native-post-training-crash-controller");
            let runtime = TestNativePostTrainingSleepRuntime {
                identity: identity.clone(),
                due: Rc::new(RefCell::new(Vec::new())),
                clocks: None,
            };
            let mut controller = NativePostTrainingBoundaryController::new(runtime).unwrap();
            let starting_clock = PostTrainingClockReceipt::new(
                identity,
                request
                    .input_checkpoint
                    .as_ref()
                    .unwrap()
                    .sha256()
                    .to_owned(),
                399,
                9_000,
            )
            .unwrap();
            let mut publisher = TestUpdatePublisher::new(false);
            let mut policy = TestDpoRuntime::new("candidate:dpo");
            let mut first_sink = TestPhaseProgress {
                fail_checkpoint_call: Some(failed_checkpoint),
                ..TestPhaseProgress::default()
            };
            {
                let mut adapters = PostTrainingPhaseRuntime::Dpo {
                    runtime: &mut policy,
                };
                let mut hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut controller);
                let first = drive_resumable_post_training_phase(
                    &test_sha("wrapped-native-crash-workflow"),
                    &request,
                    &mut adapters,
                    &mut publisher,
                    &mut hook,
                    Some(&starting_clock),
                    None,
                    usize::MAX,
                    &mut first_sink,
                );
                assert!(
                    first.is_err(),
                    "checkpoint {failed_checkpoint} did not interrupt"
                );
            }
            let resume: PostTrainingResumeEnvelope = serde_json::from_value(
                first_sink
                    .durable
                    .clone()
                    .expect("boundary crash lost the outer resume envelope"),
            )
            .unwrap();
            let (_, boundary) = resume.clone().into_parts().unwrap();
            assert!(boundary.is_some(), "boundary publication was not durable");
            let outcome = {
                let mut adapters = PostTrainingPhaseRuntime::Dpo {
                    runtime: &mut policy,
                };
                let mut hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut controller);
                drive_resumable_post_training_phase(
                    &test_sha("wrapped-native-crash-workflow"),
                    &request,
                    &mut adapters,
                    &mut publisher,
                    &mut hook,
                    Some(&starting_clock),
                    Some(resume),
                    usize::MAX,
                    &mut TestPhaseProgress::default(),
                )
                .unwrap()
            };
            let ResumablePostTrainingOutcome::Complete { cursor, .. } = outcome else {
                panic!("resumed boundary did not complete");
            };
            assert_eq!(publisher.apply_calls, 1, "optimizer update was duplicated");
            assert_eq!(cursor.clock.as_ref().unwrap().optimizer_steps, 400);
            assert!(cursor.pending.is_none());
        }
    }

    #[test]
    fn periodic_post_training_fails_before_model_work_without_authenticated_start_clock() {
        let (_dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        install_periodic_sleep(&mut request);
        let mut publisher = TestUpdatePublisher::new(false);
        let mut hook = TestBoundaryHook::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let mut adapters = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let mut boundary_hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut hook);
        let error = drive_resumable_post_training_phase(
            &test_sha("missing-start-clock"),
            &request,
            &mut adapters,
            &mut publisher,
            &mut boundary_hook,
            None,
            None,
            usize::MAX,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("authenticated starting clock"), "{error}");
        assert_eq!(policy.model_tokens, 0);
        assert_eq!(publisher.apply_calls, 0);
    }

    fn run_native_host_periodic_post_training(algorithm: TestPostTrainingAlgorithm) {
        let (_data_dir, mut request) = test_request(algorithm);
        install_periodic_sleep(&mut request);
        let workflow = ResolvedWorkflow {
            version: 2,
            name: Some("native-post-training-host-test".to_owned()),
            phases: vec![request.phase.clone()],
        };
        workflow.validate().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let state_path = temporary.path().join("runtime.json");
        let metrics_path = temporary.path().join("metrics.jsonl");
        let due = Rc::new(RefCell::new(Vec::new()));
        let clocks = Rc::new(RefCell::new(BTreeMap::from([(
            request
                .input_checkpoint
                .as_ref()
                .unwrap()
                .sha256()
                .to_owned(),
            PostTrainingClockValues {
                optimizer_steps: 399,
                model_tokens: 9_000,
            },
        )])));
        let controller_identity = test_sha("host-periodic-controller");
        let mut adapters = NativeWorkflowAdapters::new();
        adapters
            .register_post_training_factory(TestHostPostTrainingFactory {
                identity: test_sha("host-post-training-factory"),
                registered_controller_identity: controller_identity.clone(),
                lent_controller_identity: controller_identity,
                due: due.clone(),
                clocks,
            })
            .unwrap();
        let mut host = NativeWorkflowHost::start(
            workflow,
            adapters,
            &state_path,
            Some(NativeHostMetricJournal {
                path: metrics_path,
                run_id: "native-post-training-host-test".to_owned(),
            }),
            request.input_checkpoint.clone(),
        )
        .unwrap();
        let status = host.drive_until_yield_or_complete().unwrap();
        assert_eq!(status, RuntimeStatus::AlreadyComplete);
        assert!(host.state().is_complete(host.workflow()));
        assert_eq!(&*due.borrow(), &[(400, 0), (400, 1)]);
    }

    #[test]
    fn native_workflow_host_executes_periodic_dpo_forward_kl_and_grpo() {
        run_native_host_periodic_post_training(TestPostTrainingAlgorithm::Dpo);
        run_native_host_periodic_post_training(TestPostTrainingAlgorithm::ForwardKl);
        run_native_host_periodic_post_training(TestPostTrainingAlgorithm::Grpo);
    }

    #[test]
    fn native_host_carries_authenticated_clocks_across_post_training_phases() {
        let (dpo_dir, mut dpo) = test_request(TestPostTrainingAlgorithm::Dpo);
        let (kl_dir, mut kl) = test_request(TestPostTrainingAlgorithm::ForwardKl);
        let (grpo_dir, mut grpo) = test_request(TestPostTrainingAlgorithm::Grpo);
        install_periodic_sleep(&mut dpo);
        install_periodic_sleep(&mut kl);
        install_periodic_sleep(&mut grpo);
        let _keep_data_alive = (dpo_dir, kl_dir, grpo_dir);
        let initial = dpo.input_checkpoint.clone().unwrap();
        let workflow = ResolvedWorkflow {
            version: 2,
            name: Some("cross-phase-post-training-clock".to_owned()),
            phases: vec![dpo.phase, kl.phase, grpo.phase],
        };
        workflow.validate().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let due = Rc::new(RefCell::new(Vec::new()));
        let clocks = Rc::new(RefCell::new(BTreeMap::from([(
            initial.sha256().to_owned(),
            PostTrainingClockValues {
                optimizer_steps: 399,
                model_tokens: 9_000,
            },
        )])));
        let controller_identity = test_sha("cross-phase-controller");
        let mut adapters = NativeWorkflowAdapters::new();
        adapters
            .register_post_training_factory(TestHostPostTrainingFactory {
                identity: test_sha("cross-phase-factory"),
                registered_controller_identity: controller_identity.clone(),
                lent_controller_identity: controller_identity,
                due: due.clone(),
                clocks: clocks.clone(),
            })
            .unwrap();
        let mut host = NativeWorkflowHost::start(
            workflow,
            adapters,
            temporary.path().join("runtime.json"),
            Some(NativeHostMetricJournal {
                path: temporary.path().join("metrics.jsonl"),
                run_id: "cross-phase-post-training-clock".to_owned(),
            }),
            Some(initial),
        )
        .unwrap();
        assert_eq!(
            host.drive_until_yield_or_complete().unwrap(),
            RuntimeStatus::AlreadyComplete
        );
        let final_checkpoint = host.state().current_checkpoint().unwrap();
        let final_clock = clocks
            .borrow()
            .get(final_checkpoint.sha256())
            .copied()
            .unwrap();
        assert_eq!(final_clock.optimizer_steps, 405);
        assert_eq!(final_clock.model_tokens, 9_014);
        assert_eq!(&*due.borrow(), &[(400, 0), (400, 1)]);
    }

    #[test]
    fn native_host_rejects_malformed_and_lent_controller_identity_drift() {
        let due = Rc::new(RefCell::new(Vec::new()));
        let mut malformed = NativeWorkflowAdapters::new();
        let error = malformed
            .register_post_training_factory(TestHostPostTrainingFactory {
                identity: test_sha("host-post-training-factory"),
                registered_controller_identity: "not-a-sha256".to_owned(),
                lent_controller_identity: test_sha("lent-controller"),
                due: due.clone(),
                clocks: Rc::new(RefCell::new(BTreeMap::new())),
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("periodic-sleep controller"), "{error}");

        let (_data_dir, mut request) = test_request(TestPostTrainingAlgorithm::Dpo);
        request.phase.steps = Some(1);
        install_periodic_sleep(&mut request);
        let workflow = ResolvedWorkflow {
            version: 2,
            name: Some("controller-drift".to_owned()),
            phases: vec![request.phase.clone()],
        };
        let temporary = tempfile::tempdir().unwrap();
        let clocks = Rc::new(RefCell::new(BTreeMap::from([(
            request
                .input_checkpoint
                .as_ref()
                .unwrap()
                .sha256()
                .to_owned(),
            PostTrainingClockValues {
                optimizer_steps: 399,
                model_tokens: 9_000,
            },
        )])));
        let mut adapters = NativeWorkflowAdapters::new();
        adapters
            .register_post_training_factory(TestHostPostTrainingFactory {
                identity: test_sha("host-post-training-factory"),
                registered_controller_identity: test_sha("registered-controller"),
                lent_controller_identity: test_sha("lent-controller"),
                due,
                clocks,
            })
            .unwrap();
        let mut host = NativeWorkflowHost::start(
            workflow,
            adapters,
            temporary.path().join("runtime.json"),
            Some(NativeHostMetricJournal {
                path: temporary.path().join("metrics.jsonl"),
                run_id: "controller-drift".to_owned(),
            }),
            request.input_checkpoint,
        )
        .unwrap();
        let error = host
            .drive_until_yield_or_complete()
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside its registered identity"), "{error}");
    }

    #[test]
    fn resume_rejects_cursor_tamper_and_device_runtime_drift() {
        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Dpo);
        let workflow = test_sha("workflow-v2");
        let mut publisher = TestUpdatePublisher::new(false);
        let mut sink = TestPhaseProgress::default();
        let mut no_hook = None;
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        let cursor = {
            let mut adapters = PostTrainingPhaseRuntime::Dpo {
                runtime: &mut policy,
            };
            let yielded = drive_resumable_post_training_phase(
                &workflow,
                &request,
                &mut adapters,
                &mut publisher,
                &mut no_hook,
                None,
                None,
                1,
                &mut sink,
            )
            .unwrap();
            let ResumablePostTrainingOutcome::Yielded(cursor) = yielded else {
                panic!("one-update budget should yield");
            };

            let mut tampered = cursor.clone();
            tampered
                .last_update_receipt
                .as_mut()
                .unwrap()
                .receipt_sha256 = test_sha("tampered-receipt");
            let error = drive_resumable_post_training_phase(
                &workflow,
                &request,
                &mut adapters,
                &mut publisher,
                &mut no_hook,
                None,
                Some(PostTrainingResumeEnvelope::wake(tampered)),
                1,
                &mut TestPhaseProgress::default(),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("receipt is invalid"), "{error}");
            cursor
        };

        policy.identity = "candidate:drifted".to_owned();
        let mut drifted = PostTrainingPhaseRuntime::Dpo {
            runtime: &mut policy,
        };
        let error = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut drifted,
            &mut publisher,
            &mut no_hook,
            None,
            Some(PostTrainingResumeEnvelope::wake(cursor)),
            1,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("identity changed"), "{error}");
    }

    #[test]
    fn grpo_resume_rejects_verifier_implementation_drift() {
        struct VersionedExactVerifier(&'static str);

        impl RewardVerifier for VersionedExactVerifier {
            fn identity(&self) -> &str {
                self.0
            }

            fn adapter_name(&self) -> &str {
                ExactAnswerVerifier.adapter_name()
            }

            fn verify(
                &self,
                spec: &VerifierSpec,
                request: VerificationRequest<'_>,
            ) -> Result<Verification> {
                ExactAnswerVerifier.verify(spec, request)
            }
        }

        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Grpo);
        let workflow = test_sha("workflow-v2-verifier-identity");
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestGrpoRuntime::new();
        let mut no_hook = None;
        let first = VersionedExactVerifier(
            "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        );
        let cursor = {
            let mut adapters = PostTrainingPhaseRuntime::Grpo {
                runtime: &mut policy,
                verifier: &first,
            };
            let outcome = drive_resumable_post_training_phase(
                &workflow,
                &request,
                &mut adapters,
                &mut publisher,
                &mut no_hook,
                None,
                None,
                1,
                &mut TestPhaseProgress::default(),
            )
            .unwrap();
            let ResumablePostTrainingOutcome::Yielded(cursor) = outcome else {
                panic!("one-update budget should yield");
            };
            cursor
        };

        let second = VersionedExactVerifier(
            "sha256:2222222222222222222222222222222222222222222222222222222222222222",
        );
        let mut adapters = PostTrainingPhaseRuntime::Grpo {
            runtime: &mut policy,
            verifier: &second,
        };
        let error = drive_resumable_post_training_phase(
            &workflow,
            &request,
            &mut adapters,
            &mut publisher,
            &mut no_hook,
            None,
            Some(PostTrainingResumeEnvelope::wake(cursor)),
            1,
            &mut TestPhaseProgress::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("identity changed across resume"), "{error}");
    }

    #[test]
    fn native_executor_integrates_with_phase_executor_contract() {
        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Dpo);
        let workflow = test_sha("workflow-v2");
        let mut executor = NativePostTrainingPhaseExecutor::new(&workflow).unwrap();
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestDpoRuntime::new("candidate:dpo");
        {
            let mut context = PostTrainingExecutionContext {
                runtime: PostTrainingPhaseRuntime::Dpo {
                    runtime: &mut policy,
                },
                publisher: &mut publisher,
                boundary_hook: None,
                starting_clock: None,
            };
            let result = executor
                .execute(&request, &mut context, &mut TestPhaseProgress::default())
                .unwrap();
            assert!(matches!(
                result,
                PhaseExecutionResult::Complete(PhaseProduct::ModelCandidate { .. })
            ));
        }
        assert_eq!(publisher.apply_calls, 2);
    }
}
