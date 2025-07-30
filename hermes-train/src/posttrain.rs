//! Backend-neutral post-training objectives and executor interfaces.
//!
//! This module owns the mathematics and validation for preference,
//! distillation, and verifiable-RL phases. Model execution remains injected:
//! a backend supplies differentiable policy scores/logits and generated
//! rollouts, while these cores return losses, metrics, and derivatives with
//! respect to those supplied scores. No task is converted to another task
//! shape implicitly.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use burn::prelude::Tensor;
use burn::tensor::activation::{log_sigmoid, log_softmax as tensor_log_softmax};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
use crate::task::{RewardSpec, TaskAdapter, TaskConfig, TaskDataFormat, TaskExample, VerifierSpec};
use crate::workflow::{InModelSleepConfig, PhaseKind, PhaseV2};

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
    /// SHA-256 of the exact artifact bytes. Required for every local artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, serde_json::Value>,
}

impl FrozenModelSpec {
    fn validate(&self, field: &str) -> Result<()> {
        ensure!(
            !self.adapter.trim().is_empty(),
            "`{field}.adapter` must not be empty"
        );
        ensure!(
            self.artifact
                .as_ref()
                .is_none_or(|path| !path.as_os_str().is_empty()),
            "`{field}.artifact` must not be empty when set"
        );
        ensure!(
            self.revision
                .as_deref()
                .is_none_or(|revision| !revision.trim().is_empty()),
            "`{field}.revision` must not be empty when set"
        );
        ensure!(
            self.artifact.is_none() || self.sha256.is_some(),
            "`{field}.artifact` requires an exact `sha256`"
        );
        if let Some(sha256) = &self.sha256 {
            validate_sha256(sha256).with_context(|| format!("invalid `{field}.sha256`"))?;
        }
        ensure!(
            self.sha256.is_some() || self.revision.is_some(),
            "`{field}` must pin an exact sha256 or immutable revision"
        );
        Ok(())
    }

    /// Stable identity required from an injected provider.
    pub fn immutable_identity(&self) -> Result<String> {
        self.validate("frozen_model")?;
        Ok(match &self.sha256 {
            Some(sha256) => format!("sha256:{}", normalize_sha256(sha256)),
            None => format!(
                "revision:{}",
                self.revision
                    .as_deref()
                    .context("validated frozen model has no immutable identity")?
            ),
        })
    }

    /// Verify local artifact bytes before an executor is allowed to load them.
    pub fn verify_local_artifact(&self) -> Result<()> {
        self.validate("frozen_model")?;
        let Some(path) = &self.artifact else {
            return Ok(());
        };
        let metadata = std::fs::symlink_metadata(path).with_context(|| {
            format!("failed to inspect frozen model artifact {}", path.display())
        })?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "frozen model artifact {} must be a non-symlink regular file",
            path.display()
        );
        let expected = normalize_sha256(
            self.sha256
                .as_deref()
                .expect("validated local artifact has sha256"),
        );
        let mut file = File::open(path)
            .with_context(|| format!("failed to open frozen model artifact {}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).with_context(|| {
                format!("failed to hash frozen model artifact {}", path.display())
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = format!("{:x}", hasher.finalize());
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
                ensure!(*group_size >= 2, "GRPO group_size must be at least two");
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
        ensure!(self.max_new_tokens > 0, "max_new_tokens must be positive");
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

/// Differentiable sequence scores supplied by policy/reference executors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairwiseLogProbabilities {
    pub policy_chosen: f64,
    pub policy_rejected: f64,
    pub reference_chosen: f64,
    pub reference_rejected: f64,
}

/// DPO scalar result. Derivatives target the aggregate policy sequence log
/// probabilities and can be seeded into any autodiff backend.
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
    Ok(DpoLoss {
        loss,
        policy_margin,
        reference_margin,
        implicit_reward_margin,
        preference_correct: policy_margin > reference_margin,
        d_policy_chosen: beta * d_z,
        d_policy_rejected: -beta * d_z,
    })
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

#[derive(Clone, Debug, PartialEq)]
pub struct DistillationToken<'a> {
    pub teacher_logits: &'a [f64],
    pub student_logits: &'a [f64],
    /// Zero masks a token; fractional values support packed-example weights.
    pub weight: f64,
}

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
    let mut weighted_kl = 0.0;
    let mut weighted_entropy = 0.0;
    let mut weighted_agreement = 0.0;
    let mut student_gradients = Vec::with_capacity(tokens.len());

    for token in tokens {
        let teacher_log_probs = log_softmax(token.teacher_logits, temperature);
        let student_log_probs = log_softmax(token.student_logits, temperature);
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
        student_gradients.push(
            student_probs
                .iter()
                .zip(&teacher_probs)
                .map(|(student, teacher)| row_scale * (student - teacher))
                .collect(),
        );
    }

    let mean_forward_kl = weighted_kl / total_weight;
    Ok(DistillationLoss {
        loss: loss_scale * mean_forward_kl,
        mean_forward_kl,
        teacher_entropy: weighted_entropy / total_weight,
        top1_agreement: weighted_agreement / total_weight,
        student_gradients,
    })
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

/// One on-policy/near-policy rollout in a GRPO prompt group.
#[derive(Clone, Debug, PartialEq)]
pub struct GrpoRollout<'a> {
    pub reward: f64,
    pub current_token_log_probs: &'a [f64],
    pub behavior_token_log_probs: &'a [f64],
    pub reference_token_log_probs: Option<&'a [f64]>,
}

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
    let variance = rewards
        .iter()
        .map(|reward| (reward - mean_reward).powi(2))
        .sum::<f64>()
        / rewards.len() as f64;
    let reward_stddev = variance.sqrt();
    let denominator = reward_stddev.max(advantage_epsilon);
    let advantages: Vec<f64> = rewards
        .iter()
        .map(|reward| (reward - mean_reward) / denominator)
        .collect();

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
                log_ratio <= 80.0,
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
                        log_ref_minus_policy <= 80.0,
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
    let reward_mean = rewards.clone().mean();
    let reward_stddev = (rewards.clone() - reward_mean.clone())
        .powi_scalar(2)
        .mean()
        .sqrt()
        .clamp_min(advantage_epsilon);
    let advantages = ((rewards - reward_mean) / reward_stddev).reshape([group, 1]);
    let ratio = (current_token_log_probs.clone() - behavior_token_log_probs.detach()).exp();
    let clipped_ratio = ratio.clone().clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon);
    let surrogate = (ratio * advantages.clone()).min_pair(clipped_ratio * advantages);
    let mask = active_mask.detach();
    let active_per_rollout = mask.clone().sum_dim(1);
    let token_objective = match reference_token_log_probs {
        Some(reference) => {
            let log_ref_minus_policy = reference.detach() - current_token_log_probs;
            let kl = log_ref_minus_policy.clone().exp() - log_ref_minus_policy - 1;
            surrogate - kl.mul_scalar(kl_coefficient)
        }
        None => surrogate,
    };
    let sequence_objective = (token_objective * mask).sum_dim(1) / active_per_rollout;
    Ok(sequence_objective.mean().neg())
}

/// Request sent to an executor-owned rollout generator.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutRequest<'a> {
    pub prompt: &'a str,
    pub count: usize,
    pub max_sequence_tokens: usize,
    pub sampling: &'a RolloutSampling,
    /// Stable seed owned by the durable phase cursor.
    pub rng_seed: u64,
    /// First counter reserved for this request. Implementations must derive
    /// all sampling randomness from `(rng_seed, rng_counter)` and consume one
    /// deterministic substream per requested rollout.
    pub rng_counter: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedRollout {
    pub text: String,
    pub token_ids: Vec<u32>,
    /// Log probabilities under the behavior policy that generated the sample.
    pub behavior_token_log_probs: Vec<f64>,
}

/// A model runtime capable of generating a pinned group of policy rollouts.
pub trait RolloutGenerator {
    fn identity(&self) -> &str;
    fn generate(&mut self, request: RolloutRequest<'_>) -> Result<Vec<GeneratedRollout>>;
}

/// Frozen policy scoring used by DPO and distillation executors. Implementors
/// must preserve token alignment and return one log-probability per supplied
/// continuation token.
pub trait SequenceLogProbabilityProvider {
    /// Must equal [`FrozenModelSpec::immutable_identity`] for frozen providers.
    fn identity(&self) -> &str;
    fn tokenizer_identity(&self) -> &str;
    fn continuation_log_probs(
        &mut self,
        prompt: &str,
        continuation: &str,
        max_sequence_tokens: usize,
    ) -> Result<Vec<f64>>;
}

/// Frozen teacher distribution used by a distillation executor. Rows must be
/// aligned with the student target-token rows for the exact task example.
pub trait TeacherDistributionProvider {
    fn identity(&self) -> &str;
    fn tokenizer_identity(&self) -> &str;
    fn token_logits(
        &mut self,
        example: &TaskExample,
        max_sequence_tokens: usize,
    ) -> Result<Vec<Vec<f64>>>;
}

/// Trainable DPO policy. Backward receives gradients for exactly the token
/// log-probabilities returned by `continuation_log_probs`.
pub trait PreferencePolicy: SequenceLogProbabilityProvider {
    /// Exact cumulative number of trainable-policy model tokens processed by
    /// this adapter. The counter must be monotonic for the lifetime of the
    /// borrowed adapter and include every policy forward/generation executed
    /// by the methods below. The resumable executor samples it around one
    /// deterministic update; it never estimates tokens from sequence length.
    fn model_tokens_processed(&self) -> u64;

    fn backward_pairwise_log_probs(
        &mut self,
        example: &TaskExample,
        chosen_gradients: &[f64],
        rejected_gradients: &[f64],
    ) -> Result<()>;
    fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()>;
}

/// Trainable student distribution. Backward rows are aligned one-for-one with
/// the rows returned by `student_token_logits`.
pub trait DistillationPolicy {
    fn identity(&self) -> &str;
    fn tokenizer_identity(&self) -> &str;
    /// Exact cumulative trainable-student model-token counter. See
    /// [`PreferencePolicy::model_tokens_processed`].
    fn model_tokens_processed(&self) -> u64;
    fn student_token_logits(
        &mut self,
        example: &TaskExample,
        max_sequence_tokens: usize,
    ) -> Result<Vec<Vec<f64>>>;
    fn backward_student_logits(
        &mut self,
        example: &TaskExample,
        gradients: &[Vec<f64>],
    ) -> Result<()>;
    fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()>;
}

/// Trainable GRPO policy. Rescoring and backward preserve the generated
/// rollout's token alignment exactly.
pub trait GrpoPolicy: RolloutGenerator {
    fn tokenizer_identity(&self) -> &str;
    /// Exact cumulative trainable-policy model-token counter, including both
    /// generation and policy rescoring. See
    /// [`PreferencePolicy::model_tokens_processed`].
    fn model_tokens_processed(&self) -> u64;
    fn current_token_log_probs(
        &mut self,
        prompt: &str,
        rollout: &GeneratedRollout,
        max_sequence_tokens: usize,
    ) -> Result<Vec<f64>>;
    fn backward_rollout_log_probs(
        &mut self,
        prompt: &str,
        rollout: &GeneratedRollout,
        gradients: &[f64],
    ) -> Result<()>;
    fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()>;
}

/// Frozen reference scorer consuming the generator's exact tokenization.
pub trait AlignedReferencePolicy {
    fn identity(&self) -> &str;
    fn tokenizer_identity(&self) -> &str;
    fn rollout_token_log_probs(
        &mut self,
        prompt: &str,
        rollout: &GeneratedRollout,
        max_sequence_tokens: usize,
    ) -> Result<Vec<f64>>;
}

/// Exactly one adapter set must match the typed algorithm on the phase.
pub enum PostTrainingPhaseAdapters<'a> {
    Dpo {
        policy: &'a mut dyn PreferencePolicy,
        reference: &'a mut dyn SequenceLogProbabilityProvider,
    },
    ForwardKl {
        policy: &'a mut dyn DistillationPolicy,
        teacher: &'a mut dyn TeacherDistributionProvider,
    },
    Grpo {
        policy: &'a mut dyn GrpoPolicy,
        reference: Option<&'a mut dyn AlignedReferencePolicy>,
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

#[derive(Default)]
struct ReportAccumulator {
    examples: usize,
    optimizer_steps: usize,
    loss_sum: f64,
    metrics: BTreeMap<String, f64>,
}

impl ReportAccumulator {
    fn add_metric(&mut self, name: &str, value: f64) {
        *self.metrics.entry(name.to_owned()).or_default() += value;
    }

    fn finish(
        mut self,
        phase: &PhaseV2,
        algorithm: &str,
        trainable_model_identity: String,
        frozen_model_identity: Option<String>,
    ) -> Result<PostTrainingPhaseReport> {
        ensure!(
            self.examples > 0 && self.optimizer_steps > 0,
            "post-training phase `{}` produced no optimizer steps",
            phase.name
        );
        for value in self.metrics.values_mut() {
            *value /= self.examples as f64;
        }
        Ok(PostTrainingPhaseReport {
            phase: phase.name.clone(),
            algorithm: algorithm.to_owned(),
            examples: self.examples,
            optimizer_steps: self.optimizer_steps,
            trainable_model_identity,
            frozen_model_identity,
            mean_loss: self.loss_sum / self.examples as f64,
            metrics: self.metrics,
        })
    }
}

/// Execute a complete preference, distillation, or verifiable-RL phase with
/// injected model adapters. Epochs, optimizer geometry, step caps, loss
/// weight, and learning-rate scale are honored. Unsupported hooks and raw
/// parameters are rejected rather than ignored.
pub fn execute_post_training_phase(
    phase: &PhaseV2,
    adapters: PostTrainingPhaseAdapters<'_>,
) -> Result<PostTrainingPhaseReport> {
    let task = phase
        .task
        .as_ref()
        .with_context(|| format!("post-training phase `{}` has no task", phase.name))?;
    let data = phase
        .data
        .as_deref()
        .with_context(|| format!("post-training phase `{}` has no data", phase.name))?;
    let sequence_length = phase.sequence_length.with_context(|| {
        format!(
            "post-training phase `{}` has no sequence_length",
            phase.name
        )
    })?;
    let update_examples = phase
        .batch_size
        .with_context(|| format!("post-training phase `{}` has no batch_size", phase.name))?
        .checked_mul(phase.gradient_accumulation.with_context(|| {
            format!(
                "post-training phase `{}` has no gradient_accumulation",
                phase.name
            )
        })?)
        .context("post-training optimizer example count overflows usize")?;
    ensure!(
        update_examples > 0,
        "post-training optimizer batch is empty"
    );
    ensure!(
        phase.shuffle_buffer.is_none(),
        "post-training phase `{}` requests shuffle_buffer, but this deterministic executor requires an injected ordering adapter",
        phase.name
    );
    ensure!(
        phase.periodic_sleep.is_none(),
        "post-training phase `{}` has periodic_sleep; WorkflowV2 orchestration must own the sleep boundary",
        phase.name
    );
    ensure!(
        phase.parameters.is_empty(),
        "post-training phase `{}` has unconsumed raw parameters; move algorithm settings into `post_training`",
        phase.name
    );
    task.validate()?;
    let config = phase.post_training.as_ref().with_context(|| {
        format!(
            "post-training phase `{}` has no typed post_training settings",
            phase.name
        )
    })?;
    config.validate()?;
    if let PostTrainingConfig::Grpo { sampling, .. } = config {
        ensure!(
            sampling.max_new_tokens <= sequence_length,
            "post-training phase `{}` GRPO max_new_tokens exceeds sequence_length",
            phase.name
        );
    }

    match (phase.kind, config, adapters) {
        (
            PhaseKind::Preference,
            PostTrainingConfig::Dpo {
                reference: reference_spec,
                beta,
                label_smoothing,
                sequence_reduction,
            },
            PostTrainingPhaseAdapters::Dpo { policy, reference },
        ) => execute_dpo_phase(
            phase,
            task,
            data,
            sequence_length,
            update_examples,
            reference_spec,
            *beta,
            *label_smoothing,
            *sequence_reduction,
            policy,
            reference,
        ),
        (
            PhaseKind::Distillation,
            PostTrainingConfig::ForwardKl {
                teacher: teacher_spec,
                temperature,
                scale_by_temperature_squared,
            },
            PostTrainingPhaseAdapters::ForwardKl { policy, teacher },
        ) => execute_distillation_phase(
            phase,
            task,
            data,
            sequence_length,
            update_examples,
            teacher_spec,
            *temperature,
            *scale_by_temperature_squared,
            policy,
            teacher,
        ),
        (
            PhaseKind::Rl,
            PostTrainingConfig::Grpo {
                group_size,
                clip_epsilon,
                advantage_epsilon,
                kl_coefficient,
                reference: reference_spec,
                sampling,
            },
            PostTrainingPhaseAdapters::Grpo {
                policy,
                reference,
                verifier,
            },
        ) => execute_grpo_phase(
            phase,
            task,
            data,
            sequence_length,
            update_examples,
            *group_size,
            *clip_epsilon,
            *advantage_epsilon,
            *kl_coefficient,
            reference_spec.as_ref(),
            sampling,
            policy,
            reference,
            verifier,
        ),
        (kind, config, _) => bail!(
            "post-training phase `{}` type `{}` / algorithm `{}` / adapter set do not match",
            phase.name,
            kind.name(),
            post_training_algorithm_name(config)
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_dpo_phase(
    phase: &PhaseV2,
    task: &TaskConfig,
    data: &Path,
    sequence_length: usize,
    update_examples: usize,
    reference_spec: &FrozenModelSpec,
    beta: f64,
    label_smoothing: f64,
    reduction: SequenceReduction,
    policy: &mut dyn PreferencePolicy,
    reference: &mut dyn SequenceLogProbabilityProvider,
) -> Result<PostTrainingPhaseReport> {
    validate_frozen_provider(
        reference_spec,
        reference.identity(),
        reference.tokenizer_identity(),
        policy.tokenizer_identity(),
    )?;
    let policy_identity = policy.identity().to_owned();
    let frozen_identity = reference.identity().to_owned();
    let mut report = ReportAccumulator::default();
    let mut pending = Vec::with_capacity(update_examples);
    let max_steps = phase.steps.unwrap_or(usize::MAX);
    for _ in 0..phase.epochs_or_default() {
        if report.optimizer_steps >= max_steps {
            break;
        }
        visit_task_examples_while(data, task, |example| {
            pending.push(example);
            if pending.len() == update_examples {
                process_dpo_update(
                    &mut pending,
                    sequence_length,
                    beta,
                    label_smoothing,
                    reduction,
                    phase.loss_weight_or_default(),
                    phase.learning_rate_scale_or_default(),
                    policy,
                    reference,
                    &mut report,
                )?;
            }
            Ok(report.optimizer_steps < max_steps)
        })?;
    }
    if !pending.is_empty() && report.optimizer_steps < max_steps {
        process_dpo_update(
            &mut pending,
            sequence_length,
            beta,
            label_smoothing,
            reduction,
            phase.loss_weight_or_default(),
            phase.learning_rate_scale_or_default(),
            policy,
            reference,
            &mut report,
        )?;
    }
    report.finish(phase, "dpo", policy_identity, Some(frozen_identity))
}

#[allow(clippy::too_many_arguments)]
fn process_dpo_update(
    pending: &mut Vec<TaskExample>,
    sequence_length: usize,
    beta: f64,
    label_smoothing: f64,
    reduction: SequenceReduction,
    loss_weight: f64,
    learning_rate_scale: f64,
    policy: &mut dyn PreferencePolicy,
    reference: &mut dyn SequenceLogProbabilityProvider,
    report: &mut ReportAccumulator,
) -> Result<()> {
    let batch_scale = loss_weight / pending.len() as f64;
    for example in pending.iter() {
        let TaskExample::PairwisePreference {
            prompt,
            chosen,
            rejected,
        } = example
        else {
            bail!("DPO executor received a non-pairwise task example");
        };
        let policy_chosen = policy.continuation_log_probs(prompt, chosen, sequence_length)?;
        let policy_rejected = policy.continuation_log_probs(prompt, rejected, sequence_length)?;
        let reference_chosen = reference.continuation_log_probs(prompt, chosen, sequence_length)?;
        let reference_rejected =
            reference.continuation_log_probs(prompt, rejected, sequence_length)?;
        let objective = dpo_loss(
            PairwiseLogProbabilities {
                policy_chosen: reduce_sequence_log_probs(&policy_chosen, reduction)?,
                policy_rejected: reduce_sequence_log_probs(&policy_rejected, reduction)?,
                reference_chosen: reduce_sequence_log_probs(&reference_chosen, reduction)?,
                reference_rejected: reduce_sequence_log_probs(&reference_rejected, reduction)?,
            },
            beta,
            label_smoothing,
        )?;
        let chosen_gradient = distribute_sequence_gradient(
            objective.d_policy_chosen * batch_scale,
            policy_chosen.len(),
            reduction,
        );
        let rejected_gradient = distribute_sequence_gradient(
            objective.d_policy_rejected * batch_scale,
            policy_rejected.len(),
            reduction,
        );
        policy.backward_pairwise_log_probs(example, &chosen_gradient, &rejected_gradient)?;
        report.loss_sum += objective.loss;
        report.add_metric(
            "preference_accuracy",
            f64::from(objective.preference_correct),
        );
        report.add_metric("implicit_reward_margin", objective.implicit_reward_margin);
        report.examples += 1;
    }
    policy.optimizer_step(learning_rate_scale)?;
    report.optimizer_steps += 1;
    pending.clear();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_distillation_phase(
    phase: &PhaseV2,
    task: &TaskConfig,
    data: &Path,
    sequence_length: usize,
    update_examples: usize,
    teacher_spec: &FrozenModelSpec,
    temperature: f64,
    scale_by_temperature_squared: bool,
    policy: &mut dyn DistillationPolicy,
    teacher: &mut dyn TeacherDistributionProvider,
) -> Result<PostTrainingPhaseReport> {
    validate_frozen_provider(
        teacher_spec,
        teacher.identity(),
        teacher.tokenizer_identity(),
        policy.tokenizer_identity(),
    )?;
    let policy_identity = policy.identity().to_owned();
    let frozen_identity = teacher.identity().to_owned();
    let mut report = ReportAccumulator::default();
    let mut pending = Vec::with_capacity(update_examples);
    let max_steps = phase.steps.unwrap_or(usize::MAX);
    for _ in 0..phase.epochs_or_default() {
        if report.optimizer_steps >= max_steps {
            break;
        }
        visit_task_examples_while(data, task, |example| {
            pending.push(example);
            if pending.len() == update_examples {
                process_distillation_update(
                    &mut pending,
                    sequence_length,
                    temperature,
                    scale_by_temperature_squared,
                    phase.loss_weight_or_default(),
                    phase.learning_rate_scale_or_default(),
                    policy,
                    teacher,
                    &mut report,
                )?;
            }
            Ok(report.optimizer_steps < max_steps)
        })?;
    }
    if !pending.is_empty() && report.optimizer_steps < max_steps {
        process_distillation_update(
            &mut pending,
            sequence_length,
            temperature,
            scale_by_temperature_squared,
            phase.loss_weight_or_default(),
            phase.learning_rate_scale_or_default(),
            policy,
            teacher,
            &mut report,
        )?;
    }
    report.finish(phase, "forward_kl", policy_identity, Some(frozen_identity))
}

#[allow(clippy::too_many_arguments)]
fn process_distillation_update(
    pending: &mut Vec<TaskExample>,
    sequence_length: usize,
    temperature: f64,
    scale_by_temperature_squared: bool,
    loss_weight: f64,
    learning_rate_scale: f64,
    policy: &mut dyn DistillationPolicy,
    teacher: &mut dyn TeacherDistributionProvider,
    report: &mut ReportAccumulator,
) -> Result<()> {
    let batch_scale = loss_weight / pending.len() as f64;
    for example in pending.iter() {
        let teacher_logits = teacher.token_logits(example, sequence_length)?;
        let student_logits = policy.student_token_logits(example, sequence_length)?;
        ensure!(
            teacher_logits.len() == student_logits.len(),
            "teacher/student target-token row counts differ"
        );
        let tokens: Vec<_> = teacher_logits
            .iter()
            .zip(&student_logits)
            .map(|(teacher_logits, student_logits)| DistillationToken {
                teacher_logits,
                student_logits,
                weight: 1.0,
            })
            .collect();
        let objective =
            forward_kl_distillation(&tokens, temperature, scale_by_temperature_squared)?;
        let mut gradients = objective.student_gradients;
        for row in &mut gradients {
            for value in row {
                *value *= batch_scale;
            }
        }
        policy.backward_student_logits(example, &gradients)?;
        report.loss_sum += objective.loss;
        report.add_metric("forward_kl", objective.mean_forward_kl);
        report.add_metric("teacher_entropy", objective.teacher_entropy);
        report.add_metric("top1_agreement", objective.top1_agreement);
        report.examples += 1;
    }
    policy.optimizer_step(learning_rate_scale)?;
    report.optimizer_steps += 1;
    pending.clear();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_grpo_phase(
    phase: &PhaseV2,
    task: &TaskConfig,
    data: &Path,
    sequence_length: usize,
    update_examples: usize,
    group_size: usize,
    clip_epsilon: f64,
    advantage_epsilon: f64,
    kl_coefficient: f64,
    reference_spec: Option<&FrozenModelSpec>,
    sampling: &RolloutSampling,
    policy: &mut dyn GrpoPolicy,
    mut reference: Option<&mut dyn AlignedReferencePolicy>,
    verifier: &dyn RewardVerifier,
) -> Result<PostTrainingPhaseReport> {
    match (reference_spec, reference.as_deref()) {
        (Some(spec), Some(provider)) => validate_frozen_provider(
            spec,
            provider.identity(),
            provider.tokenizer_identity(),
            policy.tokenizer_identity(),
        )?,
        (Some(_), None) => bail!("GRPO phase requires its configured frozen reference adapter"),
        (None, Some(_)) => bail!("GRPO adapter set supplied an unconfigured frozen reference"),
        (None, None) => ensure!(
            kl_coefficient == 0.0,
            "GRPO without a frozen reference requires zero kl_coefficient"
        ),
    }
    let policy_identity = policy.identity().to_owned();
    let frozen_identity = reference
        .as_deref()
        .map(|provider| provider.identity().to_owned());
    let mut report = ReportAccumulator::default();
    let mut pending = Vec::with_capacity(update_examples);
    let max_steps = phase.steps.unwrap_or(usize::MAX);
    for _ in 0..phase.epochs_or_default() {
        if report.optimizer_steps >= max_steps {
            break;
        }
        visit_task_examples_while(data, task, |example| {
            pending.push(example);
            if pending.len() == update_examples {
                process_grpo_update(
                    &mut pending,
                    task,
                    sequence_length,
                    group_size,
                    clip_epsilon,
                    advantage_epsilon,
                    kl_coefficient,
                    sampling,
                    phase.loss_weight_or_default(),
                    phase.learning_rate_scale_or_default(),
                    policy,
                    &mut reference,
                    verifier,
                    &mut report,
                )?;
            }
            Ok(report.optimizer_steps < max_steps)
        })?;
    }
    if !pending.is_empty() && report.optimizer_steps < max_steps {
        process_grpo_update(
            &mut pending,
            task,
            sequence_length,
            group_size,
            clip_epsilon,
            advantage_epsilon,
            kl_coefficient,
            sampling,
            phase.loss_weight_or_default(),
            phase.learning_rate_scale_or_default(),
            policy,
            &mut reference,
            verifier,
            &mut report,
        )?;
    }
    report.finish(phase, "grpo", policy_identity, frozen_identity)
}

#[allow(clippy::too_many_arguments)]
fn process_grpo_update(
    pending: &mut Vec<TaskExample>,
    task: &TaskConfig,
    sequence_length: usize,
    group_size: usize,
    clip_epsilon: f64,
    advantage_epsilon: f64,
    kl_coefficient: f64,
    sampling: &RolloutSampling,
    loss_weight: f64,
    learning_rate_scale: f64,
    policy: &mut dyn GrpoPolicy,
    reference: &mut Option<&mut dyn AlignedReferencePolicy>,
    verifier: &dyn RewardVerifier,
    report: &mut ReportAccumulator,
) -> Result<()> {
    let batch_scale = loss_weight / pending.len() as f64;
    for example in pending.iter() {
        let TaskExample::VerifiableRollout { prompt, .. } = example else {
            bail!("GRPO executor received a non-verifiable task example");
        };
        let generated = policy.generate(RolloutRequest {
            prompt,
            count: group_size,
            max_sequence_tokens: sequence_length,
            sampling,
            rng_seed: 0,
            rng_counter: u64::try_from(report.examples)
                .context("GRPO example count exceeds u64")?
                .checked_mul(u64::try_from(group_size).context("GRPO group size exceeds u64")?)
                .context("GRPO compatibility RNG counter overflows u64")?,
        })?;
        ensure!(
            generated.len() == group_size,
            "rollout generator returned {} candidates, expected {group_size}",
            generated.len()
        );
        let mut rewards = Vec::with_capacity(group_size);
        let mut current_scores = Vec::with_capacity(group_size);
        let mut reference_scores = Vec::with_capacity(group_size);
        for (index, rollout) in generated.iter().enumerate() {
            ensure!(
                !rollout.text.trim().is_empty(),
                "rollout {index} completion is empty"
            );
            ensure!(
                !rollout.token_ids.is_empty()
                    && rollout.token_ids.len() == rollout.behavior_token_log_probs.len(),
                "rollout {index} token ids and behavior log-probs are not aligned"
            );
            ensure_finite(
                &rollout.behavior_token_log_probs,
                "rollout behavior log probabilities",
            )?;
            rewards.push(verify_rollout(task, example, verifier, &rollout.text)?.reward);
            current_scores.push(policy.current_token_log_probs(
                prompt,
                rollout,
                sequence_length,
            )?);
            reference_scores.push(match reference.as_deref_mut() {
                Some(reference) => {
                    Some(reference.rollout_token_log_probs(prompt, rollout, sequence_length)?)
                }
                None => None,
            });
        }
        let scalar_rollouts: Vec<_> = generated
            .iter()
            .enumerate()
            .map(|(index, rollout)| GrpoRollout {
                reward: rewards[index],
                current_token_log_probs: &current_scores[index],
                behavior_token_log_probs: &rollout.behavior_token_log_probs,
                reference_token_log_probs: reference_scores[index].as_deref(),
            })
            .collect();
        let objective = grpo_loss(
            &scalar_rollouts,
            clip_epsilon,
            advantage_epsilon,
            kl_coefficient,
        )?;
        for (index, rollout) in generated.iter().enumerate() {
            let gradients: Vec<f64> = objective.current_log_prob_gradients[index]
                .iter()
                .map(|gradient| gradient * batch_scale)
                .collect();
            policy.backward_rollout_log_probs(prompt, rollout, &gradients)?;
        }
        report.loss_sum += objective.loss;
        report.add_metric("mean_reward", objective.mean_reward);
        report.add_metric("reward_stddev", objective.reward_stddev);
        report.add_metric("mean_kl", objective.mean_kl);
        report.add_metric("clipped_fraction", objective.clipped_fraction);
        report.examples += 1;
    }
    policy.optimizer_step(learning_rate_scale)?;
    report.optimizer_steps += 1;
    pending.clear();
    Ok(())
}

fn validate_frozen_provider(
    spec: &FrozenModelSpec,
    provider_identity: &str,
    provider_tokenizer: &str,
    policy_tokenizer: &str,
) -> Result<()> {
    spec.verify_local_artifact()?;
    let expected = spec.immutable_identity()?;
    ensure!(
        provider_identity == expected,
        "frozen provider identity mismatch: expected `{expected}`, got `{provider_identity}`"
    );
    ensure!(
        !provider_tokenizer.trim().is_empty() && provider_tokenizer == policy_tokenizer,
        "frozen provider and trainable policy tokenizer identities differ"
    );
    Ok(())
}

fn distribute_sequence_gradient(
    aggregate_gradient: f64,
    token_count: usize,
    reduction: SequenceReduction,
) -> Vec<f64> {
    let value = match reduction {
        SequenceReduction::Sum => aggregate_gradient,
        SequenceReduction::Mean => aggregate_gradient / token_count as f64,
    };
    vec![value; token_count]
}

fn post_training_algorithm_name(config: &PostTrainingConfig) -> &'static str {
    match config {
        PostTrainingConfig::Dpo { .. } => "dpo",
        PostTrainingConfig::ForwardKl { .. } => "forward_kl",
        PostTrainingConfig::Grpo { .. } => "grpo",
    }
}

/// Serialized native post-training cursor contract.
pub const POST_TRAINING_CURSOR_VERSION: u32 = 2;
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
        validate_prefixed_sha256(&self.authority_identity, "post-training clock authority")?;
        validate_prefixed_sha256(&self.checkpoint_sha256, "post-training clock checkpoint")?;
        validate_prefixed_sha256(&self.receipt_sha256, "post-training clock receipt")?;
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
    fn from_path(path: &Path) -> Result<Self> {
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("failed to inspect post-training input {}", path.display()))?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "post-training input {} must be a non-symlink regular file",
            path.display()
        );
        let path_text = path
            .to_str()
            .context("post-training input path is not valid UTF-8")?
            .to_owned();
        let mut file = File::open(path)
            .with_context(|| format!("failed to open post-training input {}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).with_context(|| {
                format!("failed to hash post-training input {}", path.display())
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(Self {
            path: path_text,
            sha256: format!("sha256:{:x}", hasher.finalize()),
            bytes: metadata.len(),
        })
    }

    fn verify(&self, path: &Path) -> Result<()> {
        validate_prefixed_sha256(&self.sha256, "post-training input")?;
        let actual = Self::from_path(path)?;
        ensure!(
            &actual == self,
            "post-training input identity changed: expected {} ({} bytes), got {} ({} bytes)",
            self.sha256,
            self.bytes,
            actual.sha256,
            actual.bytes
        );
        Ok(())
    }
}

/// All runtime adapters are pinned into the cursor. Frozen model identities
/// additionally remain bound to their WorkflowV2 `FrozenModelSpec`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingAdapterIdentities {
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
        validate_prefixed_sha256(&self.execution_sha256, "post-training execution")?;
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
        validate_prefixed_sha256(&self.phase_sha256, "post-training phase")?;
        validate_prefixed_sha256(
            &self.previous_receipt_chain_sha256,
            "post-training receipt chain",
        )?;
        validate_prefixed_sha256(&self.batch_sha256, "post-training batch")?;
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
        ensure!(
            self.rng_end >= self.rng_start,
            "post-training RNG range is inverted"
        );
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

/// The publisher's proof that the attached trainable adapter was restored to
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
        validate_prefixed_sha256(&self.receipt_sha256, "optimizer receipt")?;
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

/// Adapter which atomically owns model/optimizer restoration and immutable
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
    /// operation by `plan.transaction_id` and never overwrite an artifact.
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
        ensure!(
            self.transaction_id == transaction_id
                && self.hook_identity == hook_identity
                && self.input_model_sha256 == input.model.sha256(),
            "periodic-sleep boundary receipt belongs to another boundary"
        );
        validate_prefixed_sha256(&self.receipt_sha256, "periodic-sleep receipt")?;
        validate_prefixed_sha256(
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
        validate_prefixed_sha256(runtime.identity(), "native post-training sleep runtime")?;
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
        validate_prefixed_sha256(request.transaction_id, "post-training sleep transaction")?;
        validate_prefixed_sha256(request.workflow_signature, "post-training sleep workflow")?;
        request.clock_before.validate()?;
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
            checkpoint.sleep.clock
                <= match request.config.schedule.clock {
                    UpdateClock::OptimizerSteps => request.clock_after.optimizer_steps,
                    UpdateClock::ModelTokens => request.clock_after.model_tokens,
                },
            "resumed post-training sleep cursor is ahead of its target clock"
        );

        let state = self
            .runtime
            .advance_and_drain(request, &mut checkpoint, progress)?;
        let target = match request.config.schedule.clock {
            UpdateClock::OptimizerSteps => request.clock_after.optimizer_steps,
            UpdateClock::ModelTokens => request.clock_after.model_tokens,
        };
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
        identities: &PostTrainingAdapterIdentities,
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
    pub adapters: PostTrainingAdapterIdentities,
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
        adapters: PostTrainingAdapterIdentities,
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
            adapters,
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
        validate_prefixed_sha256(&self.workflow_signature, "workflow signature")?;
        validate_prefixed_sha256(&self.phase_sha256, "post-training phase")?;
        validate_prefixed_sha256(&self.publisher_identity, "post-training publisher")?;
        if let Some(identity) = &self.boundary_hook_identity {
            validate_prefixed_sha256(identity, "post-training boundary hook")?;
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
        validate_prefixed_sha256(&self.receipt_chain_sha256, "post-training receipt chain")?;
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
                validate_prefixed_sha256(&receipt.transaction_id, "post-training transaction")?;
                validate_prefixed_sha256(
                    &receipt.previous_receipt_chain_sha256,
                    "previous post-training receipt chain",
                )?;
                validate_prefixed_sha256(&receipt.receipt_sha256, "optimizer receipt")?;
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
                    validate_prefixed_sha256(
                        &boundary.transaction_id,
                        "periodic-sleep transaction",
                    )?;
                    validate_prefixed_sha256(&boundary.receipt_sha256, "periodic-sleep receipt")?;
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
                native.sleep.clock >= before.selected(config.schedule.clock)
                    && native.sleep.clock
                        <= match config.schedule.clock {
                            UpdateClock::OptimizerSteps => after.optimizer_steps,
                            UpdateClock::ModelTokens => after.model_tokens,
                        },
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

/// One phase's concrete adapters. It is intentionally borrowed and contains
/// no site-specific loader, storage client, or global singleton.
pub struct PostTrainingExecutionContext<'a> {
    pub adapters: PostTrainingPhaseAdapters<'a>,
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
        validate_prefixed_sha256(&workflow_signature, "workflow signature")?;
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
            &mut context.adapters,
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
    adapters: &mut PostTrainingPhaseAdapters<'_>,
    publisher: &mut dyn PostTrainingUpdatePublisher,
    boundary_hook: &mut Option<&mut dyn PostTrainingBoundaryHook>,
    starting_clock: Option<&PostTrainingClockReceipt>,
    resume: Option<PostTrainingResumeEnvelope>,
    update_budget: usize,
    progress: &mut dyn PhaseProgressSink,
) -> Result<ResumablePostTrainingOutcome> {
    validate_prefixed_sha256(workflow_signature, "workflow signature")?;
    ensure!(
        update_budget > 0,
        "post-training update budget must be positive"
    );
    let phase = &request.phase;
    validate_resumable_phase(phase)?;
    let data = phase
        .data
        .as_deref()
        .context("native post-training phase has no data")?;
    let actual_input = PinnedPostTrainingInput::from_path(data)?;
    let identities = validate_and_capture_adapter_identities(phase, adapters)?;
    validate_prefixed_sha256(publisher.identity(), "post-training publisher")?;
    let (hook_identity, initial_clock) = match (
        &phase.periodic_sleep,
        boundary_hook.as_deref(),
        starting_clock,
    ) {
        (Some(_), Some(hook), Some(clock)) => {
            validate_prefixed_sha256(hook.identity(), "post-training boundary hook")?;
            clock.validate()?;
            let input = request
                .input_checkpoint
                .as_ref()
                .context("native post-training requires an immutable input checkpoint")?;
            ensure!(
                clock.authority_identity == hook.identity()
                    && clock.checkpoint_sha256 == input.sha256(),
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
            let input = request
                .input_checkpoint
                .as_ref()
                .context("native post-training requires an immutable input checkpoint")?;
            ensure!(
                &cursor.input_checkpoint == input,
                "native post-training cursor belongs to another input checkpoint"
            );
            ensure!(
                cursor.input_data == actual_input,
                "native post-training cursor belongs to different input bytes"
            );
            actual_input.verify(data)?;
            ensure!(
                cursor.adapters == identities,
                "native post-training adapter/provider identity changed across resume"
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
        validate_and_capture_adapter_identities(phase, adapters).and_then(|current| {
            ensure!(
                current == cursor.adapters,
                "native post-training adapter/provider identity drifted during execution"
            );
            Ok(())
        })?;
        validate_restore_receipt(
            &publisher.restore_committed(&cursor.committed)?,
            publisher.identity(),
            &cursor.committed,
        )?;

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
                &cursor.adapters,
            )?;
            return Ok(ResumablePostTrainingOutcome::Complete { cursor, report });
        }

        let update_examples = update_examples(phase)?;
        let start = cursor.position;
        let batch = collect_post_training_batch(
            data,
            phase.task.as_ref().expect("validated post-training task"),
            start,
            update_examples,
            u64::try_from(phase.epochs_or_default()).context("epoch count exceeds u64")?,
        )?;
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
            adapters,
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
                let mut apply = || optimizer_step(adapters, learning_rate_scale);
                let receipt = publisher.publish_update(&expected, &mut apply)?;
                receipt.validate(&expected, publisher.identity())?;
                let published_state = PostTrainingCommittedState {
                    model: receipt.checkpoint.clone(),
                    optimizer: Some(receipt.optimizer.clone()),
                };
                (receipt, published_state)
            }
        };
        validate_restore_receipt(
            &publisher.restore_committed(&published_state)?,
            publisher.identity(),
            &published_state,
        )?;

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
        progress.checkpoint(serde_json::to_value(PostTrainingResumeEnvelope::wake(
            next.clone(),
        ))?)?;
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
                &cursor.adapters,
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

fn validate_resumable_phase(phase: &PhaseV2) -> Result<()> {
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
    phase
        .task
        .as_ref()
        .context("native post-training phase has no task")?
        .validate()?;
    phase
        .data
        .as_ref()
        .context("native post-training phase has no data")?;
    phase
        .sequence_length
        .context("native post-training phase has no sequence_length")?;
    update_examples(phase)?;
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

struct CollectedPostTrainingBatch {
    examples: Vec<TaskExample>,
    end: PostTrainingRecordPosition,
}

fn collect_post_training_batch(
    path: &Path,
    task: &TaskConfig,
    start: PostTrainingRecordPosition,
    target: usize,
    epochs: u64,
) -> Result<CollectedPostTrainingBatch> {
    ensure!(
        start.epoch < epochs,
        "post-training input cursor is already at end"
    );
    let mut epoch = start.epoch;
    let mut record = start.record;
    let mut examples = Vec::with_capacity(target);
    while epoch < epochs && examples.len() < target {
        let mut seen = 0_u64;
        let wanted = record;
        let mut has_unconsumed_record = false;
        visit_task_examples_while(path, task, |example| {
            let index = seen;
            seen = seen
                .checked_add(1)
                .context("post-training per-epoch record counter overflows u64")?;
            if index < wanted {
                return Ok(true);
            }
            if examples.len() == target {
                has_unconsumed_record = true;
                return Ok(false);
            }
            examples.push(example);
            record = seen;
            Ok(true)
        })?;
        ensure!(
            wanted <= seen,
            "post-training cursor record {} exceeds epoch {} length {}",
            wanted,
            epoch,
            seen
        );
        if examples.len() == target {
            if !has_unconsumed_record {
                epoch = epoch
                    .checked_add(1)
                    .context("post-training epoch cursor overflows u64")?;
                record = 0;
            }
            break;
        }
        epoch = epoch
            .checked_add(1)
            .context("post-training epoch cursor overflows u64")?;
        record = 0;
    }
    if examples.len() < target {
        epoch = epochs;
        record = 0;
    }
    Ok(CollectedPostTrainingBatch {
        examples,
        end: PostTrainingRecordPosition { epoch, record },
    })
}

struct PreparedComputation {
    summary: PostTrainingUpdateSummary,
    rng_end: u64,
}

fn validate_and_capture_adapter_identities(
    phase: &PhaseV2,
    adapters: &mut PostTrainingPhaseAdapters<'_>,
) -> Result<PostTrainingAdapterIdentities> {
    let config = phase
        .post_training
        .as_ref()
        .context("post-training phase has no typed algorithm")?;
    match (config, adapters) {
        (
            PostTrainingConfig::Dpo {
                reference: spec, ..
            },
            PostTrainingPhaseAdapters::Dpo { policy, reference },
        ) => {
            validate_frozen_provider(
                spec,
                reference.identity(),
                reference.tokenizer_identity(),
                policy.tokenizer_identity(),
            )?;
            validate_adapter_identity(policy.identity(), "trainable DPO policy")?;
            Ok(PostTrainingAdapterIdentities {
                trainable: policy.identity().to_owned(),
                tokenizer: policy.tokenizer_identity().to_owned(),
                frozen: Some(reference.identity().to_owned()),
                verifier: None,
            })
        }
        (
            PostTrainingConfig::ForwardKl { teacher: spec, .. },
            PostTrainingPhaseAdapters::ForwardKl { policy, teacher },
        ) => {
            validate_frozen_provider(
                spec,
                teacher.identity(),
                teacher.tokenizer_identity(),
                policy.tokenizer_identity(),
            )?;
            validate_adapter_identity(policy.identity(), "trainable distillation policy")?;
            Ok(PostTrainingAdapterIdentities {
                trainable: policy.identity().to_owned(),
                tokenizer: policy.tokenizer_identity().to_owned(),
                frozen: Some(teacher.identity().to_owned()),
                verifier: None,
            })
        }
        (
            PostTrainingConfig::Grpo {
                reference: reference_spec,
                kl_coefficient,
                ..
            },
            PostTrainingPhaseAdapters::Grpo {
                policy,
                reference,
                verifier,
            },
        ) => {
            match (reference_spec.as_ref(), reference.as_deref()) {
                (Some(spec), Some(provider)) => validate_frozen_provider(
                    spec,
                    provider.identity(),
                    provider.tokenizer_identity(),
                    policy.tokenizer_identity(),
                )?,
                (Some(_), None) => bail!("GRPO phase requires its frozen reference adapter"),
                (None, Some(_)) => bail!("GRPO phase supplied an unconfigured reference adapter"),
                (None, None) => ensure!(
                    *kl_coefficient == 0.0,
                    "GRPO without a reference requires zero KL coefficient"
                ),
            }
            validate_adapter_identity(policy.identity(), "trainable GRPO policy")?;
            validate_adapter_identity(verifier.adapter_name(), "GRPO verifier")?;
            Ok(PostTrainingAdapterIdentities {
                trainable: policy.identity().to_owned(),
                tokenizer: policy.tokenizer_identity().to_owned(),
                frozen: reference
                    .as_deref()
                    .map(|provider| provider.identity().to_owned()),
                verifier: Some(verifier.adapter_name().to_owned()),
            })
        }
        _ => bail!("post-training algorithm and adapter set do not match"),
    }
}

fn optimizer_step(
    adapters: &mut PostTrainingPhaseAdapters<'_>,
    learning_rate_scale: f64,
) -> Result<()> {
    match adapters {
        PostTrainingPhaseAdapters::Dpo { policy, .. } => policy.optimizer_step(learning_rate_scale),
        PostTrainingPhaseAdapters::ForwardKl { policy, .. } => {
            policy.optimizer_step(learning_rate_scale)
        }
        PostTrainingPhaseAdapters::Grpo { policy, .. } => {
            policy.optimizer_step(learning_rate_scale)
        }
    }
}

fn adapter_model_tokens_processed(adapters: &PostTrainingPhaseAdapters<'_>) -> u64 {
    match adapters {
        PostTrainingPhaseAdapters::Dpo { policy, .. } => policy.model_tokens_processed(),
        PostTrainingPhaseAdapters::ForwardKl { policy, .. } => policy.model_tokens_processed(),
        PostTrainingPhaseAdapters::Grpo { policy, .. } => policy.model_tokens_processed(),
    }
}

fn exact_model_token_delta(start: u64, end: u64) -> Result<u64> {
    let delta = end
        .checked_sub(start)
        .context("trainable-policy model-token counter regressed during an update")?;
    ensure!(
        delta > 0,
        "trainable-policy adapter reported zero model tokens for a non-empty update"
    );
    Ok(delta)
}

fn accumulate_resumable_update(
    phase: &PhaseV2,
    examples: &[TaskExample],
    adapters: &mut PostTrainingPhaseAdapters<'_>,
    rng_seed: u64,
    rng_start: u64,
) -> Result<PreparedComputation> {
    let model_tokens_start = adapter_model_tokens_processed(adapters);
    let sequence_length = phase.sequence_length.expect("validated sequence length");
    let loss_weight = phase.loss_weight_or_default();
    let batch_scale = loss_weight / examples.len() as f64;
    let config = phase
        .post_training
        .as_ref()
        .expect("validated post-training config");
    let mut trace = Sha256::new();
    trace_serialized(&mut trace, &phase.name)?;
    trace_serialized(&mut trace, examples)?;
    match (config, adapters) {
        (
            PostTrainingConfig::Dpo {
                beta,
                label_smoothing,
                sequence_reduction,
                ..
            },
            PostTrainingPhaseAdapters::Dpo { policy, reference },
        ) => {
            let mut loss_sum = 0.0;
            let mut preference_correct = 0_u64;
            let mut implicit_reward_margin_sum = 0.0;
            for example in examples {
                let TaskExample::PairwisePreference {
                    prompt,
                    chosen,
                    rejected,
                } = example
                else {
                    bail!("DPO executor received a non-pairwise task example");
                };
                let policy_chosen =
                    policy.continuation_log_probs(prompt, chosen, sequence_length)?;
                let policy_rejected =
                    policy.continuation_log_probs(prompt, rejected, sequence_length)?;
                let reference_chosen =
                    reference.continuation_log_probs(prompt, chosen, sequence_length)?;
                let reference_rejected =
                    reference.continuation_log_probs(prompt, rejected, sequence_length)?;
                let objective = dpo_loss(
                    PairwiseLogProbabilities {
                        policy_chosen: reduce_sequence_log_probs(
                            &policy_chosen,
                            *sequence_reduction,
                        )?,
                        policy_rejected: reduce_sequence_log_probs(
                            &policy_rejected,
                            *sequence_reduction,
                        )?,
                        reference_chosen: reduce_sequence_log_probs(
                            &reference_chosen,
                            *sequence_reduction,
                        )?,
                        reference_rejected: reduce_sequence_log_probs(
                            &reference_rejected,
                            *sequence_reduction,
                        )?,
                    },
                    *beta,
                    *label_smoothing,
                )?;
                let chosen_gradients = distribute_sequence_gradient(
                    objective.d_policy_chosen * batch_scale,
                    policy_chosen.len(),
                    *sequence_reduction,
                );
                let rejected_gradients = distribute_sequence_gradient(
                    objective.d_policy_rejected * batch_scale,
                    policy_rejected.len(),
                    *sequence_reduction,
                );
                trace_serialized(
                    &mut trace,
                    &(
                        &policy_chosen,
                        &policy_rejected,
                        &reference_chosen,
                        &reference_rejected,
                        objective.loss,
                        &chosen_gradients,
                        &rejected_gradients,
                    ),
                )?;
                policy.backward_pairwise_log_probs(
                    example,
                    &chosen_gradients,
                    &rejected_gradients,
                )?;
                loss_sum += objective.loss;
                preference_correct += u64::from(objective.preference_correct);
                implicit_reward_margin_sum += objective.implicit_reward_margin;
            }
            let model_tokens =
                exact_model_token_delta(model_tokens_start, policy.model_tokens_processed())?;
            let summary = PostTrainingUpdateSummary {
                examples: u64::try_from(examples.len()).context("batch size exceeds u64")?,
                model_tokens,
                loss_sum,
                execution_sha256: format!("sha256:{:x}", trace.finalize()),
                objective: PostTrainingObjectiveSummary::Dpo {
                    preference_correct,
                    implicit_reward_margin_sum,
                },
            };
            summary.validate()?;
            Ok(PreparedComputation {
                summary,
                rng_end: rng_start,
            })
        }
        (
            PostTrainingConfig::ForwardKl {
                temperature,
                scale_by_temperature_squared,
                ..
            },
            PostTrainingPhaseAdapters::ForwardKl { policy, teacher },
        ) => {
            let mut loss_sum = 0.0;
            let mut forward_kl_sum = 0.0;
            let mut teacher_entropy_sum = 0.0;
            let mut top1_agreement_sum = 0.0;
            for example in examples {
                let teacher_logits = teacher.token_logits(example, sequence_length)?;
                let student_logits = policy.student_token_logits(example, sequence_length)?;
                ensure!(
                    teacher_logits.len() == student_logits.len(),
                    "teacher/student target-token row counts differ"
                );
                let tokens: Vec<_> = teacher_logits
                    .iter()
                    .zip(&student_logits)
                    .map(|(teacher_logits, student_logits)| DistillationToken {
                        teacher_logits,
                        student_logits,
                        weight: 1.0,
                    })
                    .collect();
                let objective =
                    forward_kl_distillation(&tokens, *temperature, *scale_by_temperature_squared)?;
                let mut gradients = objective.student_gradients;
                for row in &mut gradients {
                    for value in row {
                        *value *= batch_scale;
                    }
                }
                trace_serialized(
                    &mut trace,
                    &(&teacher_logits, &student_logits, objective.loss, &gradients),
                )?;
                policy.backward_student_logits(example, &gradients)?;
                loss_sum += objective.loss;
                forward_kl_sum += objective.mean_forward_kl;
                teacher_entropy_sum += objective.teacher_entropy;
                top1_agreement_sum += objective.top1_agreement;
            }
            let model_tokens =
                exact_model_token_delta(model_tokens_start, policy.model_tokens_processed())?;
            let summary = PostTrainingUpdateSummary {
                examples: u64::try_from(examples.len()).context("batch size exceeds u64")?,
                model_tokens,
                loss_sum,
                execution_sha256: format!("sha256:{:x}", trace.finalize()),
                objective: PostTrainingObjectiveSummary::ForwardKl {
                    forward_kl_sum,
                    teacher_entropy_sum,
                    top1_agreement_sum,
                },
            };
            summary.validate()?;
            Ok(PreparedComputation {
                summary,
                rng_end: rng_start,
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
            PostTrainingPhaseAdapters::Grpo {
                policy,
                reference,
                verifier,
            },
        ) => {
            let task = phase.task.as_ref().expect("validated task");
            let mut loss_sum = 0.0;
            let mut mean_reward_sum = 0.0;
            let mut reward_stddev_sum = 0.0;
            let mut mean_kl_sum = 0.0;
            let mut clipped_fraction_sum = 0.0;
            let mut rng_counter = rng_start;
            for example in examples {
                let TaskExample::VerifiableRollout { prompt, .. } = example else {
                    bail!("GRPO executor received a non-verifiable task example");
                };
                let generated = policy.generate(RolloutRequest {
                    prompt,
                    count: *group_size,
                    max_sequence_tokens: sequence_length,
                    sampling,
                    rng_seed,
                    rng_counter,
                })?;
                rng_counter = rng_counter
                    .checked_add(u64::try_from(*group_size).context("GRPO group exceeds u64")?)
                    .context("GRPO RNG counter overflows u64")?;
                ensure!(
                    generated.len() == *group_size,
                    "rollout generator returned {} candidates, expected {group_size}",
                    generated.len()
                );
                let mut rewards = Vec::with_capacity(*group_size);
                let mut current_scores = Vec::with_capacity(*group_size);
                let mut reference_scores = Vec::with_capacity(*group_size);
                for (index, rollout) in generated.iter().enumerate() {
                    ensure!(
                        !rollout.text.trim().is_empty(),
                        "rollout {index} completion is empty"
                    );
                    ensure!(
                        !rollout.token_ids.is_empty()
                            && rollout.token_ids.len() == rollout.behavior_token_log_probs.len(),
                        "rollout {index} token ids and behavior log-probs are not aligned"
                    );
                    ensure_finite(
                        &rollout.behavior_token_log_probs,
                        "rollout behavior log probabilities",
                    )?;
                    rewards.push(verify_rollout(task, example, *verifier, &rollout.text)?.reward);
                    current_scores.push(policy.current_token_log_probs(
                        prompt,
                        rollout,
                        sequence_length,
                    )?);
                    reference_scores.push(match reference.as_deref_mut() {
                        Some(reference) => Some(reference.rollout_token_log_probs(
                            prompt,
                            rollout,
                            sequence_length,
                        )?),
                        None => None,
                    });
                }
                let scalar_rollouts: Vec<_> = generated
                    .iter()
                    .enumerate()
                    .map(|(index, rollout)| GrpoRollout {
                        reward: rewards[index],
                        current_token_log_probs: &current_scores[index],
                        behavior_token_log_probs: &rollout.behavior_token_log_probs,
                        reference_token_log_probs: reference_scores[index].as_deref(),
                    })
                    .collect();
                let objective = grpo_loss(
                    &scalar_rollouts,
                    *clip_epsilon,
                    *advantage_epsilon,
                    *kl_coefficient,
                )?;
                let scaled_gradients: Vec<Vec<f64>> = objective
                    .current_log_prob_gradients
                    .iter()
                    .map(|row| row.iter().map(|value| value * batch_scale).collect())
                    .collect();
                trace_serialized(
                    &mut trace,
                    &(
                        &generated,
                        &rewards,
                        &current_scores,
                        &reference_scores,
                        objective.loss,
                        &scaled_gradients,
                    ),
                )?;
                for (index, rollout) in generated.iter().enumerate() {
                    policy.backward_rollout_log_probs(prompt, rollout, &scaled_gradients[index])?;
                }
                loss_sum += objective.loss;
                mean_reward_sum += objective.mean_reward;
                reward_stddev_sum += objective.reward_stddev;
                mean_kl_sum += objective.mean_kl;
                clipped_fraction_sum += objective.clipped_fraction;
            }
            let model_tokens =
                exact_model_token_delta(model_tokens_start, policy.model_tokens_processed())?;
            let summary = PostTrainingUpdateSummary {
                examples: u64::try_from(examples.len()).context("batch size exceeds u64")?,
                model_tokens,
                loss_sum,
                execution_sha256: format!("sha256:{:x}", trace.finalize()),
                objective: PostTrainingObjectiveSummary::Grpo {
                    mean_reward_sum,
                    reward_stddev_sum,
                    mean_kl_sum,
                    clipped_fraction_sum,
                },
            };
            summary.validate()?;
            Ok(PreparedComputation {
                summary,
                rng_end: rng_counter,
            })
        }
        _ => bail!("post-training algorithm and adapter set do not match"),
    }
}

fn validate_restore_receipt(
    receipt: &PostTrainingRestoreReceipt,
    publisher_identity: &str,
    state: &PostTrainingCommittedState,
) -> Result<()> {
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
    let kind = match request.phase.kind {
        PhaseKind::Preference => MetricPhaseKind::Preference,
        PhaseKind::Distillation => MetricPhaseKind::Distillation,
        PhaseKind::Rl => MetricPhaseKind::Rl,
        _ => bail!("unsupported post-training metric phase"),
    };
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
            kind,
        },
        checkpoint_hash: Some(cursor.committed.model.sha256().to_owned()),
    })
}

fn validate_adapter_identity(identity: &str, name: &str) -> Result<()> {
    ensure!(
        !identity.trim().is_empty() && !identity.contains(['\n', '\r']),
        "{name} identity must be non-empty and single-line"
    );
    Ok(())
}

fn validate_prefixed_sha256(value: &str, name: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .with_context(|| format!("{name} must use sha256:<64 lowercase hex>"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} must use sha256:<64 lowercase hex>"
    );
    Ok(())
}

fn canonical_sha256(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value).context("failed to serialize content-addressed value")?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn trace_serialized<T: Serialize + ?Sized>(hasher: &mut Sha256, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value).context("failed to serialize execution trace")?;
    hasher.update(
        u64::try_from(bytes.len())
            .context("execution trace component exceeds u64")?
            .to_le_bytes(),
    );
    hasher.update(bytes);
    Ok(())
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
    format!(
        "sha256:{:x}",
        Sha256::digest(b"post_training_receipt_chain_v1")
    )
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
    verifier.verify(
        &spec,
        VerificationRequest {
            prompt,
            completion,
            verifier_payload,
            reference_answer: reference_answer.as_deref(),
        },
    )
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
    mut visit: impl FnMut(TaskExample) -> Result<bool>,
) -> Result<usize> {
    task.validate()?;
    let file =
        File::open(path).with_context(|| format!("failed to open task data {}", path.display()))?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|ext| ext == "zst") {
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
    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.with_context(|| {
            format!("failed to read task data {}:{line_number}", path.display())
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

fn normalize_sha256(value: &str) -> String {
    value
        .strip_prefix("sha256:")
        .unwrap_or(value)
        .to_ascii_lowercase()
}

fn validate_sha256(value: &str) -> Result<()> {
    let value = normalize_sha256(value);
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "sha256 must contain exactly 64 hexadecimal digits"
    );
    Ok(())
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

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    struct TestSequencePolicy {
        identity: String,
        backward_calls: usize,
        optimizer_steps: usize,
        model_tokens: u64,
    }

    impl SequenceLogProbabilityProvider for TestSequencePolicy {
        fn identity(&self) -> &str {
            &self.identity
        }

        fn tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn continuation_log_probs(
            &mut self,
            _prompt: &str,
            continuation: &str,
            _max_sequence_tokens: usize,
        ) -> Result<Vec<f64>> {
            self.model_tokens += 1;
            Ok(vec![if continuation == "yes" { -0.2 } else { -0.8 }])
        }
    }

    impl PreferencePolicy for TestSequencePolicy {
        fn model_tokens_processed(&self) -> u64 {
            self.model_tokens
        }

        fn backward_pairwise_log_probs(
            &mut self,
            _example: &TaskExample,
            chosen_gradients: &[f64],
            rejected_gradients: &[f64],
        ) -> Result<()> {
            ensure!(
                chosen_gradients.len() == 1 && rejected_gradients.len() == 1,
                "unexpected test gradient shape"
            );
            self.backward_calls += 1;
            Ok(())
        }

        fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()> {
            close(learning_rate_scale, 0.5);
            self.optimizer_steps += 1;
            Ok(())
        }
    }

    struct TestReference;

    impl SequenceLogProbabilityProvider for TestReference {
        fn identity(&self) -> &str {
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        }

        fn tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn continuation_log_probs(
            &mut self,
            _prompt: &str,
            _continuation: &str,
            _max_sequence_tokens: usize,
        ) -> Result<Vec<f64>> {
            Ok(vec![-0.5])
        }
    }

    #[test]
    fn typed_dpo_phase_executor_runs_records_and_optimizer_geometry() {
        let dir = tempfile::tempdir().unwrap();
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
            "name": "preference",
            "type": "preference",
            "task": {"type": "pairwise_preference"},
            "data": data,
            "sequence_length": 128,
            "batch_size": 1,
            "gradient_accumulation": 2,
            "steps": 1,
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
        let mut policy = TestSequencePolicy {
            identity: "candidate:one".to_owned(),
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut reference = TestReference;
        let report = execute_post_training_phase(
            &phase,
            PostTrainingPhaseAdapters::Dpo {
                policy: &mut policy,
                reference: &mut reference,
            },
        )
        .unwrap();
        assert_eq!(report.algorithm, "dpo");
        assert_eq!(report.examples, 2);
        assert_eq!(report.optimizer_steps, 1);
        assert_eq!(policy.backward_calls, 2);
        assert_eq!(policy.optimizer_steps, 1);
        close(report.metrics["preference_accuracy"], 1.0);
    }

    struct TestStudent {
        gradients: Vec<Vec<Vec<f64>>>,
        optimizer_steps: usize,
        model_tokens: u64,
    }

    impl DistillationPolicy for TestStudent {
        fn identity(&self) -> &str {
            "candidate:student"
        }

        fn tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn model_tokens_processed(&self) -> u64 {
            self.model_tokens
        }

        fn student_token_logits(
            &mut self,
            _example: &TaskExample,
            _max_sequence_tokens: usize,
        ) -> Result<Vec<Vec<f64>>> {
            self.model_tokens += 1;
            Ok(vec![vec![0.0, 0.0]])
        }

        fn backward_student_logits(
            &mut self,
            _example: &TaskExample,
            gradients: &[Vec<f64>],
        ) -> Result<()> {
            self.gradients.push(gradients.to_vec());
            Ok(())
        }

        fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()> {
            close(learning_rate_scale, 1.0);
            self.optimizer_steps += 1;
            Ok(())
        }
    }

    struct TestTeacher;

    impl TeacherDistributionProvider for TestTeacher {
        fn identity(&self) -> &str {
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        }

        fn tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn token_logits(
            &mut self,
            _example: &TaskExample,
            _max_sequence_tokens: usize,
        ) -> Result<Vec<Vec<f64>>> {
            Ok(vec![vec![3.0_f64.ln(), 0.0]])
        }
    }

    #[test]
    fn typed_distillation_phase_executor_applies_teacher_kl_gradients() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join("teacher.safetensors");
        std::fs::write(&artifact, b"abc").unwrap();
        let data = dir.path().join("distill.txt");
        std::fs::write(&data, "student input\n").unwrap();
        let phase: PhaseV2 = serde_json::from_value(json!({
            "name": "distill",
            "type": "distillation",
            "task": {"type": "causal_lm"},
            "data": data,
            "sequence_length": 128,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "steps": 1,
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
        let mut policy = TestStudent {
            gradients: Vec::new(),
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut teacher = TestTeacher;
        let report = execute_post_training_phase(
            &phase,
            PostTrainingPhaseAdapters::ForwardKl {
                policy: &mut policy,
                teacher: &mut teacher,
            },
        )
        .unwrap();
        assert_eq!(report.examples, 1);
        assert_eq!(policy.optimizer_steps, 1);
        close(policy.gradients[0][0][0], -0.25);
        close(policy.gradients[0][0][1], 0.25);
        assert!(report.metrics["forward_kl"] > 0.0);
    }

    struct TestGrpoPolicy {
        backward_calls: usize,
        optimizer_steps: usize,
        model_tokens: u64,
    }

    impl RolloutGenerator for TestGrpoPolicy {
        fn identity(&self) -> &str {
            "candidate:policy"
        }

        fn generate(&mut self, request: RolloutRequest<'_>) -> Result<Vec<GeneratedRollout>> {
            ensure!(request.count == 2, "test expected a two-rollout group");
            ensure!(
                request.max_sequence_tokens == 128,
                "test sequence length was not propagated"
            );
            self.model_tokens += u64::try_from(request.count).unwrap();
            Ok(vec![
                GeneratedRollout {
                    text: "ok".to_owned(),
                    token_ids: vec![1],
                    behavior_token_log_probs: vec![0.0],
                },
                GeneratedRollout {
                    text: "wrong".to_owned(),
                    token_ids: vec![2],
                    behavior_token_log_probs: vec![0.0],
                },
            ])
        }
    }

    impl GrpoPolicy for TestGrpoPolicy {
        fn tokenizer_identity(&self) -> &str {
            "tokenizer-sha256:test"
        }

        fn model_tokens_processed(&self) -> u64 {
            self.model_tokens
        }

        fn current_token_log_probs(
            &mut self,
            _prompt: &str,
            _rollout: &GeneratedRollout,
            _max_sequence_tokens: usize,
        ) -> Result<Vec<f64>> {
            self.model_tokens += 1;
            Ok(vec![0.0])
        }

        fn backward_rollout_log_probs(
            &mut self,
            _prompt: &str,
            _rollout: &GeneratedRollout,
            gradients: &[f64],
        ) -> Result<()> {
            ensure!(gradients.len() == 1, "unexpected test gradient shape");
            self.backward_calls += 1;
            Ok(())
        }

        fn optimizer_step(&mut self, learning_rate_scale: f64) -> Result<()> {
            close(learning_rate_scale, 1.0);
            self.optimizer_steps += 1;
            Ok(())
        }
    }

    #[test]
    fn typed_grpo_phase_executor_generates_verifies_and_backpropagates() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("rl.jsonl");
        std::fs::write(
            &data,
            "{\"prompt\":\"answer\",\"verifier_payload\":{},\"reference_answer\":\"ok\"}\n",
        )
        .unwrap();
        let phase: PhaseV2 = serde_json::from_value(json!({
            "name": "rl",
            "type": "rl",
            "task": {
                "type": "verifiable_rl",
                "verifier": {"adapter": "exact_answer"}
            },
            "data": data,
            "sequence_length": 128,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "steps": 1,
            "post_training": {
                "algorithm": "grpo",
                "group_size": 2,
                "kl_coefficient": 0.0,
                "sampling": {"max_new_tokens": 32}
            }
        }))
        .unwrap();
        let mut policy = TestGrpoPolicy {
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let report = execute_post_training_phase(
            &phase,
            PostTrainingPhaseAdapters::Grpo {
                policy: &mut policy,
                reference: None,
                verifier: &ExactAnswerVerifier,
            },
        )
        .unwrap();
        assert_eq!(report.examples, 1);
        assert_eq!(report.optimizer_steps, 1);
        assert_eq!(policy.backward_calls, 2);
        assert_eq!(policy.optimizer_steps, 1);
        close(report.metrics["mean_reward"], 0.5);
        close(report.mean_loss, 0.0);
    }

    #[test]
    fn post_training_configuration_is_strict_and_pinned() {
        let digest = "0000000000000000000000000000000000000000000000000000000000000000";
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
    fn frozen_local_artifacts_are_verified_and_changed_bytes_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teacher.safetensors");
        std::fs::write(&path, b"abc").unwrap();
        let spec = FrozenModelSpec {
            adapter: "hermes_checkpoint".to_owned(),
            artifact: Some(path.clone()),
            sha256: Some(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_owned(),
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
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_owned(),
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
            Ok(())
        }
    }

    struct TestUpdatePublisher {
        identity: String,
        fail_before_publication_once: bool,
        apply_calls: usize,
        plans: BTreeMap<String, PreparedPostTrainingUpdate>,
        receipts: BTreeMap<String, PostTrainingUpdateReceipt>,
    }

    impl TestUpdatePublisher {
        fn new(fail_before_publication_once: bool) -> Self {
            Self {
                identity: test_sha("publisher-v1"),
                fail_before_publication_once,
                apply_calls: 0,
                plans: BTreeMap::new(),
                receipts: BTreeMap::new(),
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
            let target = match request.config.schedule.clock {
                UpdateClock::OptimizerSteps => request.clock_after.optimizer_steps,
                UpdateClock::ModelTokens => request.clock_after.model_tokens,
            };
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
            let target = match request.config.schedule.clock {
                UpdateClock::OptimizerSteps => request.clock_after.optimizer_steps,
                UpdateClock::ModelTokens => request.clock_after.model_tokens,
            };
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
                    let mut policy = TestSequencePolicy {
                        identity: "candidate:dpo".to_owned(),
                        backward_calls: 0,
                        optimizer_steps: 0,
                        model_tokens: 0,
                    };
                    let mut reference = TestReference;
                    let mut context = PostTrainingExecutionContext {
                        adapters: PostTrainingPhaseAdapters::Dpo {
                            policy: &mut policy,
                            reference: &mut reference,
                        },
                        publisher: &mut publisher,
                        boundary_hook: Some(&mut controller),
                        starting_clock: Some(starting_clock),
                    };
                    operation(&mut context)
                }
                PostTrainingConfig::ForwardKl { .. } => {
                    let mut policy = TestStudent {
                        gradients: Vec::new(),
                        optimizer_steps: 0,
                        model_tokens: 0,
                    };
                    let mut teacher = TestTeacher;
                    let mut context = PostTrainingExecutionContext {
                        adapters: PostTrainingPhaseAdapters::ForwardKl {
                            policy: &mut policy,
                            teacher: &mut teacher,
                        },
                        publisher: &mut publisher,
                        boundary_hook: Some(&mut controller),
                        starting_clock: Some(starting_clock),
                    };
                    operation(&mut context)
                }
                PostTrainingConfig::Grpo { .. } => {
                    let mut policy = TestGrpoPolicy {
                        backward_calls: 0,
                        optimizer_steps: 0,
                        model_tokens: 0,
                    };
                    let mut context = PostTrainingExecutionContext {
                        adapters: PostTrainingPhaseAdapters::Grpo {
                            policy: &mut policy,
                            reference: None,
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
        format!("sha256:{:x}", Sha256::digest(label.as_bytes()))
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
        let workflow = crate::workflow::load_workflow(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.education.example.json"),
        )
        .unwrap();
        request.phase.periodic_sleep = Some(
            workflow
                .phases
                .first()
                .and_then(|phase| phase.periodic_sleep.clone())
                .expect("education workflow has periodic sleep"),
        );
    }

    fn run_interruption_case(algorithm: TestPostTrainingAlgorithm, after_publication: bool) {
        let (_dir, request) = test_request(algorithm);
        let workflow = test_sha("workflow-v2");
        let mut publisher = TestUpdatePublisher::new(!after_publication);
        let mut first_sink = TestPhaseProgress {
            fail_checkpoint_call: after_publication.then_some(2),
            ..TestPhaseProgress::default()
        };
        let mut no_hook = None;
        let mut dpo_policy = TestSequencePolicy {
            identity: "candidate:dpo".to_owned(),
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut dpo_reference = TestReference;
        let mut student = TestStudent {
            gradients: Vec::new(),
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut teacher = TestTeacher;
        let mut grpo_policy = TestGrpoPolicy {
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut adapters = match algorithm {
            TestPostTrainingAlgorithm::Dpo => PostTrainingPhaseAdapters::Dpo {
                policy: &mut dpo_policy,
                reference: &mut dpo_reference,
            },
            TestPostTrainingAlgorithm::ForwardKl => PostTrainingPhaseAdapters::ForwardKl {
                policy: &mut student,
                teacher: &mut teacher,
            },
            TestPostTrainingAlgorithm::Grpo => PostTrainingPhaseAdapters::Grpo {
                policy: &mut grpo_policy,
                reference: None,
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
        assert_eq!(
            cursor.rng.counter > 0,
            matches!(algorithm, TestPostTrainingAlgorithm::Grpo)
        );
        assert_eq!(report.optimizer_steps, 2);
        assert_eq!(resumed_sink.metrics.len(), 2);
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
        let mut dpo_policy = TestSequencePolicy {
            identity: "candidate:dpo".to_owned(),
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut dpo_reference = TestReference;
        let mut student = TestStudent {
            gradients: Vec::new(),
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut teacher = TestTeacher;
        let mut grpo_policy = TestGrpoPolicy {
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut adapters = match algorithm {
            TestPostTrainingAlgorithm::Dpo => PostTrainingPhaseAdapters::Dpo {
                policy: &mut dpo_policy,
                reference: &mut dpo_reference,
            },
            TestPostTrainingAlgorithm::ForwardKl => PostTrainingPhaseAdapters::ForwardKl {
                policy: &mut student,
                teacher: &mut teacher,
            },
            TestPostTrainingAlgorithm::Grpo => PostTrainingPhaseAdapters::Grpo {
                policy: &mut grpo_policy,
                reference: None,
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
        let mut policy = TestSequencePolicy {
            identity: "candidate:dpo".to_owned(),
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut reference = TestReference;
        let mut adapters = PostTrainingPhaseAdapters::Dpo {
            policy: &mut policy,
            reference: &mut reference,
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
    fn model_token_jump_drains_every_crossed_boundary_fastest_to_slowest() {
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
        let mut policy = TestGrpoPolicy {
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut adapters = PostTrainingPhaseAdapters::Grpo {
            policy: &mut policy,
            reference: None,
            verifier: &ExactAnswerVerifier,
        };
        let mut hook: Option<&mut dyn PostTrainingBoundaryHook> = Some(&mut controller);
        let outcome = drive_resumable_post_training_phase(
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
        .unwrap();
        let ResumablePostTrainingOutcome::Complete { cursor, .. } = outcome else {
            panic!("coarse model-token phase did not complete");
        };
        assert_eq!(cursor.clock.as_ref().unwrap().model_tokens, 4);
        assert_eq!(
            &*due.borrow(),
            &[(1, 0), (2, 0), (2, 1), (3, 0), (4, 0), (4, 1), (4, 2),]
        );
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
            let mut policy = TestSequencePolicy {
                identity: "candidate:dpo".to_owned(),
                backward_calls: 0,
                optimizer_steps: 0,
                model_tokens: 0,
            };
            let mut reference = TestReference;
            let mut first_sink = TestPhaseProgress {
                fail_checkpoint_call: Some(failed_checkpoint),
                ..TestPhaseProgress::default()
            };
            {
                let mut adapters = PostTrainingPhaseAdapters::Dpo {
                    policy: &mut policy,
                    reference: &mut reference,
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
                let mut adapters = PostTrainingPhaseAdapters::Dpo {
                    policy: &mut policy,
                    reference: &mut reference,
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
        let mut policy = TestSequencePolicy {
            identity: "candidate:dpo".to_owned(),
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut reference = TestReference;
        let mut adapters = PostTrainingPhaseAdapters::Dpo {
            policy: &mut policy,
            reference: &mut reference,
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
    fn resume_rejects_cursor_tamper_and_adapter_drift() {
        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Dpo);
        let workflow = test_sha("workflow-v2");
        let mut publisher = TestUpdatePublisher::new(false);
        let mut sink = TestPhaseProgress::default();
        let mut no_hook = None;
        let mut policy = TestSequencePolicy {
            identity: "candidate:dpo".to_owned(),
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut reference = TestReference;
        let cursor = {
            let mut adapters = PostTrainingPhaseAdapters::Dpo {
                policy: &mut policy,
                reference: &mut reference,
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
        let mut drifted = PostTrainingPhaseAdapters::Dpo {
            policy: &mut policy,
            reference: &mut reference,
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
    fn native_executor_integrates_with_phase_executor_contract() {
        let (_dir, request) = test_request(TestPostTrainingAlgorithm::Dpo);
        let workflow = test_sha("workflow-v2");
        let mut executor = NativePostTrainingPhaseExecutor::new(&workflow).unwrap();
        let mut publisher = TestUpdatePublisher::new(false);
        let mut policy = TestSequencePolicy {
            identity: "candidate:dpo".to_owned(),
            backward_calls: 0,
            optimizer_steps: 0,
            model_tokens: 0,
        };
        let mut reference = TestReference;
        {
            let mut context = PostTrainingExecutionContext {
                adapters: PostTrainingPhaseAdapters::Dpo {
                    policy: &mut policy,
                    reference: &mut reference,
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
