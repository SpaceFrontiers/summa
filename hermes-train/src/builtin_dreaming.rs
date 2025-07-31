//! First-party, deterministic Dreaming for in-model sleep.
//!
//! Dream candidates are generated exclusively from the immutable wake-context
//! journal and use [`MemoryRouting::Dream`], which adds the paper's random
//! extra expert.  Candidates, manifests, isolated LoRA adapters, and ReSTEM
//! policy states are content addressed and published without replacement.
//! No corpus, network handle, or caller-provided reward enters this module.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[cfg(test)]
use std::fs;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::tensor::{Int, Tensor, TensorData};
use burn_nn::loss::CrossEntropyLossConfig;
use burn_optim::GradientsParams;
use hermes_llm::{Device, MemoryRouting, Transformer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::artifact_io::{
    atomic_write_new, ensure_directory, read_regular_bounded, sha256_identity,
    validate_sha256_identity,
};
use crate::builtin_sleep_adapters::{PinnedLocalArtifact, PinnedWakeContextJournal};
use crate::sleep::{ConsolidationTxn, DreamTrial, DreamingConfig, GeneratedDream, RngReservation};
use crate::tensor_sleep::{TokenRolloutBatch, TransformerDreamOps, model_parameter_hash};

pub const DREAM_SEQUENCE_SET_VERSION: u32 = 1;
const DREAM_CANDIDATE_VERSION: u32 = 3;
const DREAM_MANIFEST_VERSION: u32 = 3;
const DREAM_GENERATION_RECEIPT_VERSION: u32 = 3;
const DREAM_ADAPTER_VERSION: u32 = 3;
pub(crate) const RESTEM_POLICY_VERSION: u32 = 3;
const GENERATION_POLICY_ADAPTER_VERSION: u32 = 2;
const DREAM_ROUTING_NAME: &str = "memory_dream_cached_random_extra_expert_v2";
const DREAM_LORA_TARGET_MODULE: &str = "transformer.output_projection";
const GENERATION_POLICY_TARGET_MODULE: &str = "transformer.output_projection.dream_policy";
const GRADIENT_PROJECTION_DOMAIN: u64 = 0x6772_6164_2d70_726f;
const GENERATION_DOMAIN: u64 = 0x6472_6561_6d2d_6765;
const LORA_DOMAIN: u64 = 0x6c6f_7261_2d74_7269;

const MAX_DREAM_JSON_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DREAM_ADAPTER_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DREAM_ADAPTER_PARAMETER_BYTES: u64 = MAX_DREAM_ADAPTER_BYTES - MAX_DREAM_JSON_BYTES;
const MAX_DREAM_LM_HEAD_DATA_BYTES: usize = 512 * 1024 * 1024;
const MAX_DREAM_SEQUENCES: usize = 4_096;
const MAX_DREAM_SEQUENCE_TOKENS: usize = 65_536;
const MAX_DREAM_SEQUENCE_TOTAL_TOKENS: usize = 4 * 1024 * 1024;
const MAX_DREAM_SEQUENCE_ID_BYTES: usize = 1_024;
const MAX_RESTEM_POLICY_GENERATIONS: usize = 4_096;
const MAX_DREAM_CANDIDATE_TOKENS: usize =
    MAX_DREAM_SEQUENCE_TOKENS + BuiltinDreamingRuntimeConfig::MAX_NEW_TOKENS;

fn default_max_new_tokens() -> usize {
    32
}

fn default_gradient_dimensions() -> usize {
    256
}

fn default_lora_steps() -> usize {
    8
}

fn default_lora_learning_rate() -> f32 {
    1e-3
}

fn default_restem_learning_rate() -> f32 {
    0.1
}

fn default_generation_temperature() -> f32 {
    0.8
}

fn default_generation_policy_rank() -> usize {
    64
}

fn default_generation_policy_alpha() -> usize {
    128
}

/// Pinned inputs and bounded compute used by the production Dreaming backend.
/// Relative paths are resolved by [`Self::resolve_paths`] before loading.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BuiltinDreamingRuntimeConfig {
    pub artifact_directory: PathBuf,
    /// Required when Dreaming is loaded directly. Periodic training leaves it
    /// absent in deployment config and injects the freshly sealed boundary
    /// journal before constructing [`BuiltinDreamOps`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_context_journal: Option<PinnedLocalArtifact>,
    pub reference_set: PinnedLocalArtifact,
    pub independent_evaluation_set: PinnedLocalArtifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_policy: Option<PinnedLocalArtifact>,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_gradient_dimensions")]
    pub gradient_dimensions: usize,
    #[serde(default = "default_lora_steps")]
    pub lora_steps: usize,
    #[serde(default = "default_lora_learning_rate")]
    pub lora_learning_rate: f32,
    #[serde(default = "default_restem_learning_rate")]
    pub restem_learning_rate: f32,
    /// Sampling temperature for dream continuations. Sampling remains fully
    /// deterministic because its random stream is transaction-reserved.
    #[serde(default = "default_generation_temperature")]
    pub generation_temperature: f32,
    /// Fixed geometry of the isolated LM-head generation-policy LoRA.
    #[serde(default = "default_generation_policy_rank")]
    pub generation_policy_rank: usize,
    #[serde(default = "default_generation_policy_alpha")]
    pub generation_policy_alpha: usize,
}

impl BuiltinDreamingRuntimeConfig {
    pub const MAX_NEW_TOKENS: usize = 4_096;
    pub const MAX_GRADIENT_DIMENSIONS: usize = 4_096;
    pub const MAX_LORA_STEPS: usize = 1_024;

    pub fn resolve_paths(mut self, base: &Path) -> Result<Self> {
        ensure!(!base.as_os_str().is_empty(), "Dreaming path base is empty");
        if self.artifact_directory.is_relative() {
            self.artifact_directory = base.join(&self.artifact_directory);
        }
        self.wake_context_journal = self
            .wake_context_journal
            .map(|artifact| artifact.resolve(base))
            .transpose()?;
        self.reference_set = self.reference_set.resolve(base)?;
        self.independent_evaluation_set = self.independent_evaluation_set.resolve(base)?;
        self.initial_policy = self
            .initial_policy
            .map(|artifact| artifact.resolve(base))
            .transpose()?;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.artifact_directory.as_os_str().is_empty(),
            "Dreaming artifact directory is empty"
        );
        ensure!(self.max_new_tokens > 0, "Dreaming must generate tokens");
        ensure!(
            self.max_new_tokens <= Self::MAX_NEW_TOKENS,
            "Dreaming max_new_tokens exceeds the {}-token operational limit",
            Self::MAX_NEW_TOKENS
        );
        ensure!(
            self.gradient_dimensions > 0,
            "Dreaming gradient projection is empty"
        );
        ensure!(
            self.gradient_dimensions <= Self::MAX_GRADIENT_DIMENSIONS,
            "Dreaming gradient dimensions exceed the {}-dimension operational limit",
            Self::MAX_GRADIENT_DIMENSIONS
        );
        ensure!(self.lora_steps > 0, "Dreaming LoRA steps must be positive");
        ensure!(
            self.lora_steps <= Self::MAX_LORA_STEPS,
            "Dreaming LoRA steps exceed the {}-step operational limit",
            Self::MAX_LORA_STEPS
        );
        ensure!(
            self.generation_temperature.is_finite() && self.generation_temperature > 0.0,
            "Dreaming generation temperature must be finite and positive"
        );
        ensure!(
            self.generation_policy_rank > 0 && self.generation_policy_alpha > 0,
            "Dreaming generation-policy LoRA geometry must be positive"
        );
        ensure!(
            self.generation_policy_rank <= DreamingConfig::MAX_LORA_RANK,
            "Dreaming generation-policy LoRA rank exceeds the {}-rank operational limit",
            DreamingConfig::MAX_LORA_RANK
        );
        ensure!(
            self.generation_policy_alpha <= DreamingConfig::MAX_LORA_ALPHA,
            "Dreaming generation-policy LoRA alpha exceeds the {} operational limit",
            DreamingConfig::MAX_LORA_ALPHA
        );
        ensure!(
            self.lora_learning_rate.is_finite() && self.lora_learning_rate > 0.0,
            "Dreaming LoRA learning rate must be finite and positive"
        );
        ensure!(
            self.restem_learning_rate.is_finite() && self.restem_learning_rate > 0.0,
            "Dreaming ReSTEM learning rate must be finite and positive"
        );
        if let Some(journal) = &self.wake_context_journal {
            validate_sha256_identity(&journal.sha256, "wake-context journal")?;
        }
        validate_sha256_identity(&self.reference_set.sha256, "Dreaming reference set")?;
        validate_sha256_identity(
            &self.independent_evaluation_set.sha256,
            "Dreaming independent evaluator",
        )?;
        if let Some(policy) = &self.initial_policy {
            validate_sha256_identity(&policy.sha256, "Dreaming initial policy")?;
        }
        Ok(())
    }

    /// Resolve and authenticate a policy emitted by this runtime for a
    /// completed sleep transaction.  The transaction binding matters on
    /// resume: a digest from another run must not silently become the parent of
    /// the next Dreaming cycle merely because it exists in the same artifact
    /// store.
    pub(crate) fn committed_policy_artifact(
        &self,
        transaction_id: u64,
        sha256: &str,
    ) -> Result<PinnedLocalArtifact> {
        let artifact = PinnedLocalArtifact {
            path: addressed_path(&self.artifact_directory, "policies", sha256, "json")?,
            sha256: sha256.to_owned(),
        };
        let state: RestemPolicyState = artifact.verify_json()?;
        state.validate()?;
        ensure!(
            state.transaction_id == transaction_id,
            "committed ReSTEM policy belongs to transaction {}, expected {transaction_id}",
            state.transaction_id
        );
        Ok(artifact)
    }
}

/// A content-pinned collection of token sequences. Reference and independent
/// evaluation sets share this deliberately small, tokenizer-free schema.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DreamSequenceSet {
    pub version: u32,
    pub sequences: Vec<DreamSequence>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DreamSequence {
    pub id: String,
    pub token_ids: Vec<i64>,
}

impl DreamSequenceSet {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == DREAM_SEQUENCE_SET_VERSION,
            "unsupported Dreaming sequence-set version {}",
            self.version
        );
        ensure!(!self.sequences.is_empty(), "Dreaming sequence set is empty");
        ensure!(
            self.sequences.len() <= MAX_DREAM_SEQUENCES,
            "Dreaming sequence set exceeds the {MAX_DREAM_SEQUENCES}-sequence operational limit"
        );
        let mut ids = BTreeSet::new();
        let mut total_tokens = 0usize;
        for sequence in &self.sequences {
            ensure!(
                !sequence.id.trim().is_empty() && sequence.id.len() <= MAX_DREAM_SEQUENCE_ID_BYTES,
                "Dreaming sequence id must contain 1..={MAX_DREAM_SEQUENCE_ID_BYTES} bytes"
            );
            ensure!(
                ids.insert(sequence.id.as_str()),
                "Dreaming sequence id `{}` is repeated",
                sequence.id
            );
            ensure!(
                sequence.token_ids.len() >= 2,
                "Dreaming sequence `{}` needs at least two tokens",
                sequence.id
            );
            ensure!(
                sequence.token_ids.len() <= MAX_DREAM_SEQUENCE_TOKENS,
                "Dreaming sequence `{}` exceeds the {MAX_DREAM_SEQUENCE_TOKENS}-token operational limit",
                sequence.id
            );
            total_tokens = total_tokens
                .checked_add(sequence.token_ids.len())
                .context("Dreaming sequence-set token count overflow")?;
            ensure!(
                total_tokens <= MAX_DREAM_SEQUENCE_TOTAL_TOKENS,
                "Dreaming sequence set exceeds the {MAX_DREAM_SEQUENCE_TOTAL_TOKENS}-token operational limit"
            );
            ensure!(
                sequence
                    .token_ids
                    .iter()
                    .all(|token| u32::try_from(*token).is_ok()),
                "Dreaming sequence `{}` contains a token outside the u32 range",
                sequence.id
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DreamCandidateArtifact {
    version: u32,
    transaction_id: u64,
    ordinal: usize,
    wake_journal_sha256: String,
    source_checkpoint_sha256: String,
    wake_record_id: String,
    wake_token_count: usize,
    generation_seed: u64,
    routing_seed: u64,
    routing: String,
    generation_policy_sha256: Option<String>,
    generation_policy_adapter_sha256: Option<String>,
    generation_temperature: f32,
    token_ids: Vec<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DreamManifest {
    version: u32,
    transaction_id: u64,
    wake_journal_sha256: String,
    source_checkpoint_sha256: String,
    generation_reservation: RngReservation,
    routing: String,
    generation_policy_sha256: Option<String>,
    generation_policy_adapter_sha256: Option<String>,
    generation_temperature: f32,
    dreams: Vec<GeneratedDream>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DreamGenerationReceipt {
    version: u32,
    transaction_id: u64,
    shared_candidate_sha256: String,
    wake_journal_sha256: String,
    generation_reservation: RngReservation,
    routing_seed: u64,
    candidate_count: usize,
    generation_policy_sha256: Option<String>,
    generation_policy_adapter_sha256: Option<String>,
    generation_temperature: f32,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AdapterReceipt {
    version: u32,
    transaction_id: u64,
    base_checkpoint_sha256: String,
    base_model_parameter_sha256: String,
    candidate_id: String,
    candidate_artifact_hash: String,
    evaluator_hash: String,
    target_module: String,
    input_features: usize,
    output_features: usize,
    rank: usize,
    alpha: usize,
    steps: usize,
    learning_rate: f32,
    training_loss_before: f32,
    training_loss_after: f32,
    evaluation_loss_before: f32,
    evaluation_loss_after: f32,
    independent_task_improvement: f32,
    a_shape: [usize; 2],
    b_shape: [usize; 2],
}

impl AdapterReceipt {
    fn validate(&self) -> Result<(usize, usize)> {
        ensure!(
            self.version == DREAM_ADAPTER_VERSION,
            "unsupported dream-adapter version {}",
            self.version
        );
        validate_sha256_identity(&self.base_checkpoint_sha256, "dream LoRA base checkpoint")?;
        validate_sha256_identity(
            &self.base_model_parameter_sha256,
            "dream LoRA base-model parameters",
        )?;
        validate_sha256_identity(
            &self.candidate_artifact_hash,
            "dream LoRA candidate artifact",
        )?;
        validate_sha256_identity(&self.evaluator_hash, "dream LoRA evaluator")?;
        ensure!(
            !self.candidate_id.trim().is_empty()
                && self.candidate_id.len() <= MAX_DREAM_SEQUENCE_ID_BYTES,
            "dream LoRA candidate ID is invalid"
        );
        ensure!(
            self.target_module == DREAM_LORA_TARGET_MODULE,
            "dream LoRA targets unsupported module `{}`",
            self.target_module
        );
        ensure!(
            self.input_features > 0 && self.output_features > 1 && self.rank > 0 && self.alpha > 0,
            "dream LoRA receipt has invalid geometry"
        );
        ensure!(
            self.rank <= DreamingConfig::MAX_LORA_RANK
                && self.alpha <= DreamingConfig::MAX_LORA_ALPHA,
            "dream LoRA receipt geometry exceeds operational limits"
        );
        ensure!(
            self.a_shape == [self.rank, self.input_features]
                && self.b_shape == [self.rank, self.output_features],
            "dream LoRA tensor shapes do not match its target geometry"
        );
        ensure!(
            self.steps > 0
                && self.steps <= BuiltinDreamingRuntimeConfig::MAX_LORA_STEPS
                && self.learning_rate.is_finite()
                && self.learning_rate > 0.0
                && [
                    self.training_loss_before,
                    self.training_loss_after,
                    self.evaluation_loss_before,
                    self.evaluation_loss_after,
                    self.independent_task_improvement,
                ]
                .into_iter()
                .all(f32::is_finite),
            "dream LoRA receipt has invalid training evidence"
        );
        ensure!(
            (self.evaluation_loss_before - self.evaluation_loss_after).to_bits()
                == self.independent_task_improvement.to_bits(),
            "dream LoRA reward disagrees with its evaluation losses"
        );
        let a_values = self
            .rank
            .checked_mul(self.input_features)
            .context("dream LoRA A shape overflows usize")?;
        let b_values = self
            .rank
            .checked_mul(self.output_features)
            .context("dream LoRA B shape overflows usize")?;
        Ok((a_values, b_values))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RestemPolicyState {
    version: u32,
    parent_policy_sha256: Option<String>,
    parent_adapter_sha256: Option<String>,
    transaction_id: u64,
    source_checkpoint_sha256: String,
    source_model_parameter_sha256: String,
    topology_sha256: String,
    adapter_sha256: String,
    target_module: String,
    input_features: usize,
    output_features: usize,
    rank: usize,
    alpha: usize,
    iterations: usize,
    learning_rate: f32,
    accepted_candidates: Vec<String>,
    accepted_adapters: Vec<String>,
}

impl RestemPolicyState {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == RESTEM_POLICY_VERSION,
            "unsupported ReSTEM policy version {}",
            self.version
        );
        if let Some(parent) = &self.parent_policy_sha256 {
            validate_sha256_identity(parent, "ReSTEM parent policy")?;
        }
        if let Some(parent) = &self.parent_adapter_sha256 {
            validate_sha256_identity(parent, "ReSTEM parent adapter")?;
        }
        ensure!(
            self.parent_policy_sha256.is_some() == self.parent_adapter_sha256.is_some(),
            "ReSTEM parent binding is incomplete"
        );
        validate_sha256_identity(&self.source_checkpoint_sha256, "ReSTEM source checkpoint")?;
        validate_sha256_identity(
            &self.source_model_parameter_sha256,
            "ReSTEM source model parameters",
        )?;
        validate_sha256_identity(&self.topology_sha256, "ReSTEM model topology")?;
        validate_sha256_identity(&self.adapter_sha256, "ReSTEM generation adapter")?;
        ensure!(
            self.target_module == GENERATION_POLICY_TARGET_MODULE,
            "ReSTEM policy targets unsupported module `{}`",
            self.target_module
        );
        ensure!(
            self.input_features > 0 && self.output_features > 1 && self.rank > 0 && self.alpha > 0,
            "ReSTEM policy has invalid adapter geometry"
        );
        ensure!(
            self.rank <= DreamingConfig::MAX_LORA_RANK
                && self.alpha <= DreamingConfig::MAX_LORA_ALPHA,
            "ReSTEM policy adapter geometry exceeds operational limits"
        );
        ensure!(
            self.iterations <= DreamingConfig::MAX_RESTEM_ITERATIONS,
            "ReSTEM policy iteration count exceeds the operational limit"
        );
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "ReSTEM policy has an invalid learning rate"
        );
        ensure!(
            self.accepted_candidates.len() == self.accepted_adapters.len(),
            "ReSTEM policy accepted-candidate evidence is incomplete"
        );
        ensure!(
            self.accepted_candidates.len() <= DreamingConfig::MAX_CANDIDATES,
            "ReSTEM policy accepted-candidate evidence exceeds the operational limit"
        );
        ensure!(
            (self.iterations == 0) == self.accepted_candidates.is_empty(),
            "ReSTEM policy update count disagrees with accepted trials"
        );
        let mut candidates = BTreeSet::new();
        for hash in &self.accepted_candidates {
            validate_sha256_identity(hash, "ReSTEM accepted candidate")?;
            ensure!(
                candidates.insert(hash),
                "ReSTEM policy repeats an accepted candidate"
            );
        }
        for hash in &self.accepted_adapters {
            validate_sha256_identity(hash, "ReSTEM accepted adapter")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationPolicyAdapterReceipt {
    version: u32,
    transaction_id: u64,
    source_checkpoint_sha256: String,
    source_model_parameter_sha256: String,
    topology_sha256: String,
    parent_policy_sha256: Option<String>,
    parent_adapter_sha256: Option<String>,
    target_module: String,
    input_features: usize,
    output_features: usize,
    rank: usize,
    alpha: usize,
    iterations: usize,
    learning_rate: f32,
    accepted_candidates: Vec<String>,
    accepted_trial_adapters: Vec<String>,
    accepted_rewards: Vec<f32>,
    a_shape: [usize; 2],
    b_shape: [usize; 2],
}

impl GenerationPolicyAdapterReceipt {
    fn validate(&self) -> Result<(usize, usize)> {
        ensure!(
            self.version == GENERATION_POLICY_ADAPTER_VERSION,
            "unsupported generation-policy adapter version {}",
            self.version
        );
        validate_sha256_identity(
            &self.source_checkpoint_sha256,
            "generation-policy source checkpoint",
        )?;
        validate_sha256_identity(
            &self.source_model_parameter_sha256,
            "generation-policy source model parameters",
        )?;
        validate_sha256_identity(&self.topology_sha256, "generation-policy topology")?;
        if let Some(parent) = &self.parent_policy_sha256 {
            validate_sha256_identity(parent, "generation-policy parent policy")?;
        }
        if let Some(parent) = &self.parent_adapter_sha256 {
            validate_sha256_identity(parent, "generation-policy parent adapter")?;
        }
        ensure!(
            self.parent_policy_sha256.is_some() == self.parent_adapter_sha256.is_some(),
            "generation-policy parent binding is incomplete"
        );
        ensure!(
            self.target_module == GENERATION_POLICY_TARGET_MODULE,
            "generation-policy adapter targets unsupported module `{}`",
            self.target_module
        );
        ensure!(
            self.input_features > 0 && self.output_features > 1 && self.rank > 0 && self.alpha > 0,
            "generation-policy adapter has invalid geometry"
        );
        ensure!(
            self.rank <= DreamingConfig::MAX_LORA_RANK
                && self.alpha <= DreamingConfig::MAX_LORA_ALPHA,
            "generation-policy adapter geometry exceeds operational limits"
        );
        ensure!(
            self.a_shape == [self.rank, self.input_features]
                && self.b_shape == [self.rank, self.output_features],
            "generation-policy adapter shapes do not match its geometry"
        );
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "generation-policy adapter has invalid learning rate"
        );
        ensure!(
            self.iterations <= DreamingConfig::MAX_RESTEM_ITERATIONS,
            "generation-policy adapter iteration count exceeds the operational limit"
        );
        ensure!(
            self.accepted_candidates.len() <= DreamingConfig::MAX_CANDIDATES,
            "generation-policy adapter accepted-trial evidence exceeds the operational limit"
        );
        ensure!(
            self.accepted_candidates.len() == self.accepted_trial_adapters.len()
                && self.accepted_candidates.len() == self.accepted_rewards.len()
                && self
                    .accepted_rewards
                    .iter()
                    .all(|reward| reward.is_finite() && *reward > 0.0),
            "generation-policy adapter has invalid accepted-trial evidence"
        );
        ensure!(
            (self.iterations == 0) == self.accepted_candidates.is_empty(),
            "generation-policy adapter update count disagrees with accepted trials"
        );
        for hash in self
            .accepted_candidates
            .iter()
            .chain(&self.accepted_trial_adapters)
        {
            validate_sha256_identity(hash, "generation-policy accepted artifact")?;
        }
        let a_values = self
            .rank
            .checked_mul(self.input_features)
            .context("generation-policy A shape overflow")?;
        let b_values = self
            .rank
            .checked_mul(self.output_features)
            .context("generation-policy B shape overflow")?;
        Ok((a_values, b_values))
    }
}

/// Deterministic Transformer operations used by [`crate::tensor_sleep::TensorDreamBackend`].
pub struct BuiltinDreamOps {
    config: BuiltinDreamingRuntimeConfig,
    journal: PinnedWakeContextJournal,
    reference_set: DreamSequenceSet,
    evaluation_set: DreamSequenceSet,
    initial_policy: Option<RestemPolicyState>,
    generation_policy: Option<HostLmHeadLora>,
    evaluation_cache: Option<DeviceEvaluationCache>,
    bound_teacher_sha256: String,
    device: Device,
}

impl BuiltinDreamOps {
    pub fn load(config: BuiltinDreamingRuntimeConfig, device: Device) -> Result<Self> {
        config.validate()?;
        ensure_real_directory(&config.artifact_directory)?;
        for child in [
            "candidates",
            "manifests",
            "adapters",
            "policy-adapters",
            "policies",
            "receipts",
        ] {
            ensure_real_directory(&config.artifact_directory.join(child))?;
        }

        let journal_artifact = config
            .wake_context_journal
            .as_ref()
            .context("Dreaming runtime has no bound wake-context journal")?;
        let journal =
            PinnedWakeContextJournal::load(&journal_artifact.path, &journal_artifact.sha256)?;
        let reference_set: DreamSequenceSet = config.reference_set.verify_json()?;
        reference_set.validate()?;
        let evaluation_set: DreamSequenceSet = config.independent_evaluation_set.verify_json()?;
        evaluation_set.validate()?;
        let initial_policy = match &config.initial_policy {
            Some(artifact) => {
                let policy: RestemPolicyState = artifact.verify_json()?;
                policy.validate()?;
                Some(policy)
            }
            None => None,
        };
        let bound_teacher_sha256 = journal.source_checkpoint_sha256().to_owned();
        let mut operations = Self {
            config,
            journal,
            reference_set,
            evaluation_set,
            initial_policy,
            generation_policy: None,
            evaluation_cache: None,
            bound_teacher_sha256,
            device,
        };
        if let (Some(artifact), Some(state)) = (
            &operations.config.initial_policy,
            &operations.initial_policy,
        ) {
            let adapter = operations.validate_policy_chain(&artifact.sha256, state)?;
            operations.generation_policy = Some(adapter);
        }
        Ok(operations)
    }

    pub fn config(&self) -> &BuiltinDreamingRuntimeConfig {
        &self.config
    }

    pub fn wake_context_journal(&self) -> &PinnedWakeContextJournal {
        &self.journal
    }

    pub fn reference_set_hash(&self) -> &str {
        &self.config.reference_set.sha256
    }

    pub fn independent_evaluator_hash(&self) -> &str {
        &self.config.independent_evaluation_set.sha256
    }

    /// Bind the current transaction teacher to the authenticated descendant
    /// chain owned by the workflow runtime. The journal remains tied to the
    /// immutable phase input; coincident/later boundaries may legitimately
    /// use a descendant candidate as their teacher.
    pub fn bind_phase_teacher(
        &mut self,
        phase_input_sha256: &str,
        teacher_sha256: &str,
    ) -> Result<()> {
        validate_sha256_identity(phase_input_sha256, "Dreaming phase input")?;
        validate_sha256_identity(teacher_sha256, "Dreaming transaction teacher")?;
        ensure!(
            self.journal.source_checkpoint_sha256() == phase_input_sha256,
            "Dreaming wake journal belongs to {}, authenticated phase input is {}",
            self.journal.source_checkpoint_sha256(),
            phase_input_sha256
        );
        self.bound_teacher_sha256 = teacher_sha256.to_owned();
        Ok(())
    }

    /// A bounded, model-owned probe for [`crate::tensor_sleep::TensorDreamBackend`].
    /// It is derived from the pinned wake journal, never from an external
    /// caller or newly-read corpus data.
    pub fn probe(&self, model: &Transformer) -> Result<TokenRolloutBatch> {
        let record = self
            .journal
            .records()
            .first()
            .context("Dreaming wake journal is empty")?;
        let keep = record.token_ids.len().min(model.config().max_seq_len);
        ensure!(keep > 0, "Dreaming probe has no tokens");
        let tokens = record.token_ids[record.token_ids.len() - keep..].to_vec();
        ensure!(
            tokens
                .iter()
                .all(|token| *token >= 0 && (*token as usize) < model.config().vocab_size),
            "Dreaming probe contains a token outside vocabulary {}",
            model.config().vocab_size
        );
        TokenRolloutBatch::new(1, tokens.len(), tokens)
    }

    fn validate_transaction(&self, txn: &ConsolidationTxn) -> Result<()> {
        ensure!(txn.committed, "Dreaming requires a committed consolidation");
        ensure!(
            self.bound_teacher_sha256 == txn.teacher_hash,
            "Dreaming is bound to teacher {}, transaction teacher is {}",
            self.bound_teacher_sha256,
            txn.teacher_hash
        );
        Ok(())
    }

    fn candidate_path(&self, hash: &str) -> Result<PathBuf> {
        addressed_path(&self.config.artifact_directory, "candidates", hash, "json")
    }

    fn manifest_path(&self, hash: &str) -> Result<PathBuf> {
        addressed_path(&self.config.artifact_directory, "manifests", hash, "json")
    }

    fn adapter_path(&self, hash: &str) -> Result<PathBuf> {
        addressed_path(&self.config.artifact_directory, "adapters", hash, "bin")
    }

    fn policy_path(&self, hash: &str) -> Result<PathBuf> {
        addressed_path(&self.config.artifact_directory, "policies", hash, "json")
    }

    fn generation_policy_adapter_path(&self, hash: &str) -> Result<PathBuf> {
        addressed_path(
            &self.config.artifact_directory,
            "policy-adapters",
            hash,
            "bin",
        )
    }

    fn generation_receipt_path(&self, txn: &ConsolidationTxn) -> PathBuf {
        let mut key = Sha256::new();
        key.update(b"hermes-dream-generation-receipt-key-v2\0");
        key.update(txn.id.to_le_bytes());
        key.update(self.journal.sha256().as_bytes());
        key.update(
            txn.candidate_hash
                .as_deref()
                .unwrap_or(&txn.student_hash)
                .as_bytes(),
        );
        self.config
            .artifact_directory
            .join("receipts")
            .join(format!("generation-{:x}.json", key.finalize()))
    }

    fn generation_policy_identity(&self) -> (Option<&str>, Option<&str>) {
        (
            self.config
                .initial_policy
                .as_ref()
                .map(|artifact| artifact.sha256.as_str()),
            self.initial_policy
                .as_ref()
                .map(|policy| policy.adapter_sha256.as_str()),
        )
    }

    fn completed_generation(
        &mut self,
        txn: &ConsolidationTxn,
        candidate_count: usize,
        reservation: RngReservation,
        routing_seed: u64,
    ) -> Result<Option<(String, Vec<GeneratedDream>)>> {
        let path = self.generation_receipt_path(txn);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = read_regular_json(&path, "Dreaming generation receipt")?;
        let receipt: DreamGenerationReceipt = serde_json::from_slice(&bytes)?;
        let shared_candidate = txn.candidate_hash.as_deref().unwrap_or(&txn.student_hash);
        let (policy_sha256, adapter_sha256) = self.generation_policy_identity();
        ensure!(
            receipt.version == DREAM_GENERATION_RECEIPT_VERSION
                && receipt.transaction_id == txn.id
                && receipt.shared_candidate_sha256 == shared_candidate
                && receipt.wake_journal_sha256 == self.journal.sha256()
                && receipt.generation_reservation == reservation
                && receipt.routing_seed == routing_seed
                && receipt.candidate_count == candidate_count
                && receipt.generation_policy_sha256.as_deref() == policy_sha256
                && receipt.generation_policy_adapter_sha256.as_deref() == adapter_sha256
                && receipt.generation_temperature.to_bits()
                    == self.config.generation_temperature.to_bits(),
            "completed Dreaming generation receipt disagrees with retry request"
        );
        validate_sha256_identity(&receipt.manifest_sha256, "Dreaming receipt manifest")?;
        let dreams = self.load(txn, &receipt.manifest_sha256)?;
        ensure!(
            dreams.len() == candidate_count,
            "Dreaming receipt manifest changed candidate count"
        );
        Ok(Some((receipt.manifest_sha256, dreams)))
    }

    fn read_candidate(&self, hash: &str) -> Result<DreamCandidateArtifact> {
        let bytes = read_addressed_json(&self.candidate_path(hash)?, hash, "dream candidate")?;
        let artifact: DreamCandidateArtifact = serde_json::from_slice(&bytes)?;
        ensure!(
            artifact.version == DREAM_CANDIDATE_VERSION,
            "unsupported dream-candidate version {}",
            artifact.version
        );
        ensure!(
            artifact.wake_journal_sha256 == self.journal.sha256()
                && artifact.source_checkpoint_sha256 == self.journal.source_checkpoint_sha256(),
            "dream candidate belongs to another wake journal or source checkpoint"
        );
        ensure!(
            artifact.routing == DREAM_ROUTING_NAME,
            "candidate used wake routing"
        );
        let (policy_sha256, adapter_sha256) = self.generation_policy_identity();
        ensure!(
            artifact.generation_policy_sha256.as_deref() == policy_sha256
                && artifact.generation_policy_adapter_sha256.as_deref() == adapter_sha256
                && artifact.generation_temperature.to_bits()
                    == self.config.generation_temperature.to_bits(),
            "dream candidate generation-policy binding mismatch"
        );
        ensure!(
            artifact.token_ids.len() >= 2,
            "dream candidate is too short"
        );
        ensure!(
            artifact.token_ids.len() <= MAX_DREAM_CANDIDATE_TOKENS,
            "dream candidate exceeds the {MAX_DREAM_CANDIDATE_TOKENS}-token operational limit"
        );
        ensure!(
            artifact.wake_record_id.len() <= MAX_DREAM_SEQUENCE_ID_BYTES
                && !artifact.wake_record_id.trim().is_empty(),
            "dream candidate wake-record id is invalid"
        );
        ensure!(
            artifact.ordinal < DreamingConfig::MAX_CANDIDATES,
            "dream candidate ordinal exceeds the operational limit"
        );
        ensure!(
            artifact.wake_token_count > 0 && artifact.wake_token_count < artifact.token_ids.len(),
            "dream candidate has an invalid wake-prefix length"
        );
        ensure!(
            artifact
                .token_ids
                .iter()
                .all(|token| u32::try_from(*token).is_ok()),
            "dream candidate contains a token outside the u32 range"
        );
        ensure!(
            artifact.generation_temperature.is_finite() && artifact.generation_temperature > 0.0,
            "dream candidate has an invalid generation temperature"
        );
        ensure!(
            artifact.generation_policy_sha256.is_some()
                == artifact.generation_policy_adapter_sha256.is_some(),
            "dream candidate has an incomplete generation-policy binding"
        );
        if let Some(policy) = &artifact.generation_policy_sha256 {
            validate_sha256_identity(policy, "dream candidate generation policy")?;
        }
        if let Some(adapter) = &artifact.generation_policy_adapter_sha256 {
            validate_sha256_identity(adapter, "dream candidate generation adapter")?;
        }
        Ok(artifact)
    }

    fn select_context_index(&self, seed: u64) -> usize {
        mix64(seed) as usize % self.journal.records().len()
    }

    fn model_sequence(&self, model: &Transformer, tokens: &[i64], label: &str) -> Result<()> {
        ensure!(tokens.len() >= 2, "{label} needs at least two tokens");
        ensure!(
            tokens.len() - 1 <= model.config().max_seq_len,
            "{label} exceeds model context length {}",
            model.config().max_seq_len
        );
        ensure!(
            tokens
                .iter()
                .all(|token| *token >= 0 && (*token as usize) < model.config().vocab_size),
            "{label} contains a token outside vocabulary {}",
            model.config().vocab_size
        );
        Ok(())
    }

    fn sequence_lm_head_rows(&self, model: &Transformer, tokens: &[i64]) -> Result<LmHeadRows> {
        self.wake_lm_head_rows(model, tokens, 1, "LoRA sequence")
    }

    fn candidate_lm_head_rows(
        &self,
        model: &Transformer,
        artifact: &DreamCandidateArtifact,
    ) -> Result<LmHeadRows> {
        ensure!(
            artifact.wake_token_count > 0 && artifact.wake_token_count < artifact.token_ids.len(),
            "dream LoRA candidate has an invalid wake-prefix length"
        );
        self.wake_lm_head_rows(
            model,
            &artifact.token_ids,
            artifact.wake_token_count,
            "dream LoRA candidate",
        )
    }

    fn evaluation_lm_head_rows(
        &mut self,
        model: &Transformer,
        model_parameter_sha256: &str,
    ) -> Result<Vec<DeviceLmHeadRows>> {
        if let Some(cache) = self
            .evaluation_cache
            .as_ref()
            .filter(|cache| cache.model_parameter_sha256 == model_parameter_sha256)
        {
            return Ok(cache.rows.clone());
        }
        let total_rows =
            self.evaluation_set
                .sequences
                .iter()
                .try_fold(0usize, |total, sequence| {
                    total
                        .checked_add(sequence.token_ids.len() - 1)
                        .context("Dreaming evaluation row count overflow")
                })?;
        checked_lm_head_row_geometry(
            total_rows,
            model.config().hidden_size,
            model.config().vocab_size,
        )?;
        let rows = self
            .evaluation_set
            .sequences
            .iter()
            .map(|sequence| self.sequence_lm_head_rows(model, &sequence.token_ids))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|rows| rows.into_device(&self.device))
            .collect::<Result<Vec<_>>>()?;
        self.evaluation_cache = Some(DeviceEvaluationCache {
            model_parameter_sha256: model_parameter_sha256.to_owned(),
            rows: rows.clone(),
        });
        Ok(rows)
    }

    fn wake_lm_head_rows(
        &self,
        model: &Transformer,
        tokens: &[i64],
        target_start: usize,
        label: &str,
    ) -> Result<LmHeadRows> {
        self.model_sequence(model, tokens, label)?;
        ensure!(
            target_start > 0 && target_start < tokens.len(),
            "{label} has an invalid supervised-token range"
        );
        let input_rows = tokens.len() - 1;
        let rows = tokens.len() - target_start;
        let devices = model.devices();
        ensure!(
            devices.len() == 1,
            "frozen Dreaming model spans {} devices",
            devices.len()
        );
        let device = devices[0].clone();
        let inputs = Tensor::<2, Int>::from_data(
            TensorData::new(tokens[..input_rows].to_vec(), [1, input_rows]),
            &device,
        );
        let positions = Tensor::<1, Int>::from_data(
            TensorData::new(
                ((target_start - 1) as i64..input_rows as i64).collect::<Vec<_>>(),
                [rows],
            ),
            &device,
        );
        let projector = model.prepare_selected_logits(inputs, positions);
        let features = projector
            .hidden(0..rows)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading frozen LM-head input features")?;
        let base_logits = projector
            .logits(0..rows)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading frozen LM-head base logits")?;
        let hidden = model.config().hidden_size;
        let vocab = model.config().vocab_size;
        let (feature_values, logit_values) = checked_lm_head_row_geometry(rows, hidden, vocab)?;
        ensure!(
            features.len() == feature_values && base_logits.len() == logit_values,
            "frozen LM-head feature/logit shape mismatch"
        );
        ensure!(
            features
                .iter()
                .chain(&base_logits)
                .all(|value| value.is_finite()),
            "frozen LM-head features or logits are non-finite"
        );
        Ok(LmHeadRows {
            rows,
            hidden,
            vocab: model.config().vocab_size,
            features,
            base_logits,
            targets: tokens[target_start..]
                .iter()
                .map(|token| *token as usize)
                .collect(),
        })
    }

    fn dream_sequence_lm_head_rows(
        &self,
        model: &Transformer,
        artifact: &DreamCandidateArtifact,
    ) -> Result<LmHeadRows> {
        self.model_sequence(model, &artifact.token_ids, "ReSTEM dream sequence")?;
        ensure!(
            artifact.wake_token_count > 0 && artifact.wake_token_count < artifact.token_ids.len(),
            "ReSTEM dream sequence has an invalid wake-prefix length"
        );
        let rows = artifact.token_ids.len() - artifact.wake_token_count;
        let (feature_values, logit_values) = checked_lm_head_row_geometry(
            rows,
            model.config().hidden_size,
            model.config().vocab_size,
        )?;
        let mut feature_rows = Vec::new();
        feature_rows
            .try_reserve_exact(rows)
            .context("reserving ReSTEM hidden-state tensors")?;
        let mut logit_rows = Vec::new();
        logit_rows
            .try_reserve_exact(rows)
            .context("reserving ReSTEM logit tensors")?;
        let mut targets = Vec::new();
        targets
            .try_reserve_exact(rows)
            .context("reserving ReSTEM target rows")?;
        let devices = model.devices();
        ensure!(
            devices.len() == 1,
            "frozen ReSTEM model spans {} devices",
            devices.len()
        );
        let device = &devices[0];
        let mut state = model.make_state_with_capacity(1, artifact.token_ids.len() - 1, device);
        for generated in 0..rows {
            let prefix = artifact.wake_token_count + generated;
            let step_seed =
                mix64(artifact.generation_seed ^ generated as u64 ^ artifact.routing_seed);
            device.seed(step_seed);
            let input_tokens = if generated == 0 {
                &artifact.token_ids[..prefix]
            } else {
                &artifact.token_ids[prefix - 1..prefix]
            };
            let input = Tensor::<2, Int>::from_data(
                TensorData::new(input_tokens.to_vec(), [1, input_tokens.len()]),
                device,
            );
            let (hidden, logits) = model
                .forward_next_features_and_logits_with_state_and_memory_routing(
                    input,
                    &mut state,
                    MemoryRouting::Dream { seed: step_seed },
                );
            feature_rows.push(hidden);
            logit_rows.push(logits);
            targets.push(artifact.token_ids[prefix] as usize);
        }
        let features = Tensor::cat(feature_rows, 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading ReSTEM dream hidden states")?;
        let base_logits = Tensor::cat(logit_rows, 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading ReSTEM dream base logits")?;
        ensure!(
            features.len() == feature_values
                && base_logits.len() == logit_values
                && targets.len() == rows,
            "ReSTEM LM-head feature/logit shape mismatch"
        );
        ensure!(
            features
                .iter()
                .chain(&base_logits)
                .all(|value| value.is_finite()),
            "ReSTEM dream features or logits are non-finite"
        );
        Ok(LmHeadRows {
            rows,
            hidden: model.config().hidden_size,
            vocab: model.config().vocab_size,
            features,
            base_logits,
            targets,
        })
    }

    fn read_adapter_receipt(&self, hash: &str) -> Result<AdapterReceipt> {
        let bytes = read_addressed(&self.adapter_path(hash)?, hash, "dream LoRA adapter")?;
        ensure!(bytes.len() >= 8, "dream LoRA adapter is truncated");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        ensure!(
            header_len <= bytes.len() - 8,
            "dream LoRA adapter header is truncated"
        );
        let receipt: AdapterReceipt = serde_json::from_slice(&bytes[8..8 + header_len])?;
        let (a_values, b_values) = receipt.validate()?;
        let expected = 8usize
            .checked_add(header_len)
            .and_then(|value| {
                a_values.checked_add(b_values).and_then(|weights| {
                    weights
                        .checked_mul(std::mem::size_of::<f32>())
                        .and_then(|bytes| value.checked_add(bytes))
                })
            })
            .context("dream LoRA adapter size overflow")?;
        ensure!(
            bytes.len() == expected,
            "dream LoRA adapter length is invalid"
        );
        ensure!(
            bytes[8 + header_len..].chunks_exact(4).all(|encoded| {
                f32::from_bits(u32::from_le_bytes(encoded.try_into().unwrap())).is_finite()
            }),
            "dream LoRA adapter contains a non-finite parameter"
        );
        Ok(receipt)
    }

    fn read_generation_policy_adapter(
        &self,
        hash: &str,
    ) -> Result<(GenerationPolicyAdapterReceipt, HostLmHeadLora)> {
        let bytes = read_addressed(
            &self.generation_policy_adapter_path(hash)?,
            hash,
            "generation-policy adapter",
        )?;
        ensure!(bytes.len() >= 8, "generation-policy adapter is truncated");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        ensure!(
            header_len <= bytes.len() - 8,
            "generation-policy adapter header is truncated"
        );
        let receipt: GenerationPolicyAdapterReceipt =
            serde_json::from_slice(&bytes[8..8 + header_len])?;
        let (a_values, b_values) = receipt.validate()?;
        let parameter_bytes = a_values
            .checked_add(b_values)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("generation-policy adapter size overflow")?;
        ensure!(
            bytes.len() == 8 + header_len + parameter_bytes,
            "generation-policy adapter length is invalid"
        );
        let values = bytes[8 + header_len..]
            .chunks_exact(4)
            .map(|encoded| f32::from_bits(u32::from_le_bytes(encoded.try_into().unwrap())))
            .collect::<Vec<_>>();
        let adapter = HostLmHeadLora::from_parts(
            receipt.input_features,
            receipt.output_features,
            receipt.rank,
            receipt.alpha,
            values[..a_values].to_vec(),
            values[a_values..].to_vec(),
        )?;
        Ok((receipt, adapter))
    }

    fn validate_policy_artifacts(
        &self,
        state: &RestemPolicyState,
        receipt: &GenerationPolicyAdapterReceipt,
        adapter: &HostLmHeadLora,
    ) -> Result<()> {
        state.validate()?;
        receipt.validate()?;
        ensure!(
            state.target_module == receipt.target_module
                && state.topology_sha256 == receipt.topology_sha256
                && state.input_features == receipt.input_features
                && state.output_features == receipt.output_features
                && state.rank == receipt.rank
                && state.alpha == receipt.alpha
                && adapter.hidden == state.input_features
                && adapter.vocab == state.output_features
                && adapter.rank == state.rank
                && adapter.alpha == state.alpha,
            "ReSTEM policy and generation adapter geometry/topology disagree"
        );
        ensure!(
            state.rank == self.config.generation_policy_rank
                && state.alpha == self.config.generation_policy_alpha,
            "ReSTEM policy geometry differs from runtime configuration"
        );
        if state.iterations > 0 || state.parent_policy_sha256.is_none() {
            ensure!(
                state.transaction_id == receipt.transaction_id
                    && state.source_checkpoint_sha256 == receipt.source_checkpoint_sha256
                    && state.source_model_parameter_sha256 == receipt.source_model_parameter_sha256
                    && state.parent_policy_sha256 == receipt.parent_policy_sha256
                    && state.parent_adapter_sha256 == receipt.parent_adapter_sha256
                    && state.iterations == receipt.iterations
                    && state.learning_rate.to_bits() == receipt.learning_rate.to_bits()
                    && state.accepted_candidates == receipt.accepted_candidates
                    && state.accepted_adapters == receipt.accepted_trial_adapters,
                "ReSTEM policy receipt does not authenticate its adapter"
            );
        } else {
            ensure!(
                state.adapter_sha256 == state.parent_adapter_sha256.as_deref().unwrap(),
                "all-rejected ReSTEM policy changed its parent adapter"
            );
        }
        Ok(())
    }

    fn validate_policy_chain(
        &self,
        head_sha256: &str,
        head: &RestemPolicyState,
    ) -> Result<HostLmHeadLora> {
        validate_sha256_identity(head_sha256, "ReSTEM head policy")?;
        let mut current_hash = head_sha256.to_owned();
        let mut current = head.clone();
        let mut seen = BTreeSet::new();
        let mut head_adapter = None;
        loop {
            ensure!(
                seen.len() < MAX_RESTEM_POLICY_GENERATIONS,
                "ReSTEM policy chain exceeds the {MAX_RESTEM_POLICY_GENERATIONS}-generation operational limit"
            );
            ensure!(
                seen.insert(current_hash.clone()),
                "ReSTEM policy parent chain contains a cycle"
            );
            let (receipt, adapter) =
                self.read_generation_policy_adapter(&current.adapter_sha256)?;
            self.validate_policy_artifacts(&current, &receipt, &adapter)?;
            if head_adapter.is_none() {
                head_adapter = Some(adapter);
            }
            let Some(parent_hash) = current.parent_policy_sha256.clone() else {
                ensure!(
                    current.parent_adapter_sha256.is_none(),
                    "root ReSTEM policy names a parent adapter"
                );
                break;
            };
            let bytes = read_addressed_json(
                &self.policy_path(&parent_hash)?,
                &parent_hash,
                "ReSTEM parent policy",
            )?;
            let parent: RestemPolicyState = serde_json::from_slice(&bytes)?;
            parent.validate()?;
            ensure!(
                current.parent_adapter_sha256.as_deref() == Some(parent.adapter_sha256.as_str()),
                "ReSTEM policy parent adapter binding is invalid"
            );
            ensure!(
                current.topology_sha256 == parent.topology_sha256
                    && current.input_features == parent.input_features
                    && current.output_features == parent.output_features
                    && current.rank == parent.rank
                    && current.alpha == parent.alpha,
                "ReSTEM policy parent changed fixed topology or geometry"
            );
            current_hash = parent_hash;
            current = parent;
        }
        Ok(head_adapter.expect("policy chain contains its head"))
    }

    fn validate_generation_policy_for_model(&self, model: &Transformer) -> Result<()> {
        let Some(state) = &self.initial_policy else {
            ensure!(
                self.generation_policy.is_none(),
                "Dreaming has adapter state without an authenticated policy"
            );
            return Ok(());
        };
        let topology = model_topology_sha256(model)?;
        ensure!(
            state.topology_sha256 == topology
                && state.input_features == model.config().hidden_size
                && state.output_features == model.config().vocab_size,
            "Dreaming generation policy does not match the current model topology"
        );
        ensure!(
            self.generation_policy.is_some(),
            "Dreaming generation policy has no authenticated adapter"
        );
        Ok(())
    }
}

impl TransformerDreamOps for BuiltinDreamOps {
    fn generate(
        &mut self,
        txn: &ConsolidationTxn,
        model: &Transformer,
        candidate_count: usize,
        routing: MemoryRouting,
    ) -> Result<(String, Vec<GeneratedDream>)> {
        self.validate_transaction(txn)?;
        ensure!(candidate_count > 0, "Dreaming candidate count is zero");
        ensure!(
            candidate_count <= DreamingConfig::MAX_CANDIDATES,
            "Dreaming candidate count exceeds the {}-candidate operational limit",
            DreamingConfig::MAX_CANDIDATES
        );
        let reservation = txn
            .dream_generation_rng
            .context("Dreaming generation has no persisted RNG reservation")?;
        ensure!(
            candidate_count as u64 <= reservation.count,
            "Dreaming candidate count exceeds its RNG reservation"
        );
        let MemoryRouting::Dream { seed: routing_seed } = routing else {
            bail!("Dreaming generation must use random-extra-expert routing");
        };
        ensure!(
            routing_seed == txn.id,
            "Dreaming routing seed must equal the persisted transaction ID"
        );
        if let Some(completed) =
            self.completed_generation(txn, candidate_count, reservation, routing_seed)?
        {
            return Ok(completed);
        }
        self.validate_generation_policy_for_model(model)?;
        let generation_model = model.clone().valid();
        let generation_devices = generation_model.devices();
        ensure!(
            generation_devices.len() == 1,
            "Dreaming generation model spans {} devices",
            generation_devices.len()
        );
        let generation_device = &generation_devices[0];
        let generation_policy = self
            .generation_policy
            .as_ref()
            .map(|policy| DeviceLmHeadPolicy::from_host(policy, generation_device));
        ensure!(
            self.config.max_new_tokens < generation_model.config().max_seq_len,
            "Dreaming continuation must be shorter than model context"
        );

        let mut dreams = Vec::with_capacity(candidate_count);
        for ordinal in 0..candidate_count {
            let generation_seed = derive_seed(txn, reservation, GENERATION_DOMAIN, ordinal as u64);
            let record = &self.journal.records()[self.select_context_index(generation_seed)];
            let available = generation_model.config().max_seq_len - self.config.max_new_tokens;
            let keep = record.token_ids.len().min(available).max(1);
            let mut tokens = record.token_ids[record.token_ids.len() - keep..].to_vec();
            ensure!(
                tokens.iter().all(|token| {
                    *token >= 0 && (*token as usize) < generation_model.config().vocab_size
                }),
                "wake context `{}` contains an out-of-vocabulary token",
                record.id
            );
            let state_capacity = keep
                .checked_add(self.config.max_new_tokens - 1)
                .context("Dreaming inference-state capacity overflow")?;
            let mut state =
                generation_model.make_state_with_capacity(1, state_capacity, generation_device);
            for step in 0..self.config.max_new_tokens {
                let step_seed = mix64(generation_seed ^ step as u64 ^ routing_seed);
                generation_device.seed(step_seed);
                let input_tokens = if step == 0 {
                    &tokens[..]
                } else {
                    &tokens[tokens.len() - 1..]
                };
                let input = Tensor::<2, Int>::from_data(
                    TensorData::new(input_tokens.to_vec(), [1, input_tokens.len()]),
                    generation_device,
                );
                let (features, base_logits) = generation_model
                    .forward_next_features_and_logits_with_state_and_memory_routing(
                        input,
                        &mut state,
                        MemoryRouting::Dream { seed: step_seed },
                    );
                let logits = match &generation_policy {
                    Some(policy) => policy.adapted(features, base_logits)?,
                    None => base_logits,
                }
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .context("reading Dreaming generation logits")?;
                ensure!(
                    logits.len() == generation_model.config().vocab_size,
                    "Dreaming generation logit shape mismatch"
                );
                let next = sample_temperature(
                    &logits,
                    self.config.generation_temperature,
                    mix64(step_seed ^ GENERATION_DOMAIN),
                )?;
                tokens.push(next as i64);
            }
            self.model_sequence(&generation_model, &tokens, "generated dream")?;
            let (policy_sha256, adapter_sha256) = self.generation_policy_identity();
            let artifact = DreamCandidateArtifact {
                version: DREAM_CANDIDATE_VERSION,
                transaction_id: txn.id,
                ordinal,
                wake_journal_sha256: self.journal.sha256().to_owned(),
                source_checkpoint_sha256: self.journal.source_checkpoint_sha256().to_owned(),
                wake_record_id: record.id.clone(),
                wake_token_count: keep,
                generation_seed,
                routing_seed,
                routing: DREAM_ROUTING_NAME.to_owned(),
                generation_policy_sha256: policy_sha256.map(str::to_owned),
                generation_policy_adapter_sha256: adapter_sha256.map(str::to_owned),
                generation_temperature: self.config.generation_temperature,
                token_ids: tokens.clone(),
            };
            let bytes = canonical_json(&artifact)?;
            let artifact_hash = sha256_identity(&bytes);
            publish_immutable(&self.candidate_path(&artifact_hash)?, &bytes)?;
            let gradient = gradient_fingerprint(
                model,
                &self.device,
                &tokens,
                self.config.gradient_dimensions,
                generation_seed,
            )?;
            let id = format!("dream-{}-{ordinal}-{}", txn.id, short_hash(&artifact_hash));
            dreams.push(GeneratedDream {
                id,
                artifact_hash,
                gradient,
                diversity_key: mix64(generation_seed ^ ordinal as u64),
            });
        }

        let (policy_sha256, adapter_sha256) = self.generation_policy_identity();
        let manifest = DreamManifest {
            version: DREAM_MANIFEST_VERSION,
            transaction_id: txn.id,
            wake_journal_sha256: self.journal.sha256().to_owned(),
            source_checkpoint_sha256: self.journal.source_checkpoint_sha256().to_owned(),
            generation_reservation: reservation,
            routing: DREAM_ROUTING_NAME.to_owned(),
            generation_policy_sha256: policy_sha256.map(str::to_owned),
            generation_policy_adapter_sha256: adapter_sha256.map(str::to_owned),
            generation_temperature: self.config.generation_temperature,
            dreams: dreams.clone(),
        };
        let bytes = canonical_json(&manifest)?;
        let hash = sha256_identity(&bytes);
        publish_immutable(&self.manifest_path(&hash)?, &bytes)?;
        let receipt = DreamGenerationReceipt {
            version: DREAM_GENERATION_RECEIPT_VERSION,
            transaction_id: txn.id,
            shared_candidate_sha256: txn
                .candidate_hash
                .as_deref()
                .unwrap_or(&txn.student_hash)
                .to_owned(),
            wake_journal_sha256: self.journal.sha256().to_owned(),
            generation_reservation: reservation,
            routing_seed,
            candidate_count,
            generation_policy_sha256: policy_sha256.map(str::to_owned),
            generation_policy_adapter_sha256: adapter_sha256.map(str::to_owned),
            generation_temperature: self.config.generation_temperature,
            manifest_sha256: hash.clone(),
        };
        publish_immutable(
            &self.generation_receipt_path(txn),
            &canonical_json(&receipt)?,
        )?;
        Ok((hash, dreams))
    }

    fn load(&mut self, txn: &ConsolidationTxn, manifest: &str) -> Result<Vec<GeneratedDream>> {
        self.validate_transaction(txn)?;
        let bytes =
            read_addressed_json(&self.manifest_path(manifest)?, manifest, "dream manifest")?;
        let artifact: DreamManifest = serde_json::from_slice(&bytes)?;
        ensure!(
            artifact.version == DREAM_MANIFEST_VERSION,
            "unsupported dream-manifest version {}",
            artifact.version
        );
        ensure!(
            artifact.transaction_id == txn.id,
            "dream manifest transaction mismatch"
        );
        ensure!(
            Some(artifact.generation_reservation) == txn.dream_generation_rng,
            "dream manifest RNG reservation differs from its transaction"
        );
        ensure!(
            artifact.wake_journal_sha256 == self.journal.sha256()
                && artifact.source_checkpoint_sha256 == self.journal.source_checkpoint_sha256(),
            "dream manifest wake-journal identity mismatch"
        );
        ensure!(
            artifact.routing == DREAM_ROUTING_NAME,
            "dream manifest used wake routing"
        );
        let (policy_sha256, adapter_sha256) = self.generation_policy_identity();
        ensure!(
            artifact.generation_policy_sha256.as_deref() == policy_sha256
                && artifact.generation_policy_adapter_sha256.as_deref() == adapter_sha256
                && artifact.generation_temperature.to_bits()
                    == self.config.generation_temperature.to_bits(),
            "dream manifest generation-policy binding mismatch"
        );
        ensure!(!artifact.dreams.is_empty(), "dream manifest is empty");
        ensure!(
            artifact.dreams.len() <= DreamingConfig::MAX_CANDIDATES,
            "dream manifest exceeds the {}-candidate operational limit",
            DreamingConfig::MAX_CANDIDATES
        );
        let mut ids = BTreeSet::new();
        for (ordinal, dream) in artifact.dreams.iter().enumerate() {
            ensure!(
                !dream.id.trim().is_empty() && dream.id.len() <= MAX_DREAM_SEQUENCE_ID_BYTES,
                "dream manifest contains an invalid id"
            );
            ensure!(
                ids.insert(dream.id.as_str()),
                "dream manifest repeats an id"
            );
            validate_sha256_identity(&dream.artifact_hash, "dream candidate")?;
            ensure!(
                dream.gradient.len() == self.config.gradient_dimensions
                    && dream.gradient.iter().all(|value| value.is_finite()),
                "dream `{}` has an invalid gradient fingerprint",
                dream.id
            );
            let candidate = self.read_candidate(&dream.artifact_hash)?;
            ensure!(
                candidate.transaction_id == txn.id,
                "dream candidate transaction mismatch"
            );
            ensure!(
                candidate.ordinal == ordinal
                    && candidate.generation_seed
                        == derive_seed(
                            txn,
                            artifact.generation_reservation,
                            GENERATION_DOMAIN,
                            ordinal as u64,
                        )
                    && candidate.routing_seed == txn.id,
                "dream candidate ordinal or generation seed is inconsistent with its manifest"
            );
            ensure!(
                candidate.token_ids.len() - candidate.wake_token_count
                    == self.config.max_new_tokens,
                "dream candidate continuation length differs from runtime configuration"
            );
            let record = self
                .journal
                .records()
                .iter()
                .find(|record| record.id == candidate.wake_record_id)
                .context("dream candidate wake record is absent from its journal")?;
            ensure!(
                candidate.wake_token_count <= record.token_ids.len()
                    && candidate.token_ids[..candidate.wake_token_count]
                        == record.token_ids[record.token_ids.len() - candidate.wake_token_count..],
                "dream candidate wake prefix differs from its journal record"
            );
            ensure!(
                dream.id
                    == format!(
                        "dream-{}-{ordinal}-{}",
                        txn.id,
                        short_hash(&dream.artifact_hash)
                    )
                    && dream.diversity_key == mix64(candidate.generation_seed ^ ordinal as u64),
                "dream manifest identity metadata is inconsistent with its candidate"
            );
        }
        Ok(artifact.dreams)
    }

    fn reference_gradient(
        &mut self,
        txn: &ConsolidationTxn,
        model: &Transformer,
        reference_hash: &str,
    ) -> Result<Vec<f32>> {
        self.validate_transaction(txn)?;
        ensure!(
            reference_hash == self.config.reference_set.sha256,
            "Dreaming reference hash is not the pinned reference set"
        );
        // Re-verify at use time so replacing the file after construction is detected.
        self.config.reference_set.verify_bytes()?;
        let mut aggregate = vec![0.0_f32; self.config.gradient_dimensions];
        for (ordinal, sequence) in self.reference_set.sequences.iter().enumerate() {
            self.model_sequence(model, &sequence.token_ids, "reference sequence")?;
            let gradient = gradient_fingerprint(
                model,
                &self.device,
                &sequence.token_ids,
                self.config.gradient_dimensions,
                derive_seed(
                    txn,
                    txn.dream_generation_rng
                        .context("missing Dreaming RNG reservation")?,
                    GRADIENT_PROJECTION_DOMAIN,
                    ordinal as u64,
                ),
            )?;
            for (sum, value) in aggregate.iter_mut().zip(gradient) {
                *sum += value;
            }
        }
        normalize(&mut aggregate)?;
        Ok(aggregate)
    }

    fn isolated_lora_trial(
        &mut self,
        txn: &ConsolidationTxn,
        isolated_model: Transformer,
        candidate: &GeneratedDream,
        rank: usize,
        alpha: usize,
    ) -> Result<DreamTrial> {
        self.validate_transaction(txn)?;
        ensure!(rank > 0 && alpha > 0, "Dreaming LoRA geometry is empty");
        ensure!(
            rank <= DreamingConfig::MAX_LORA_RANK && alpha <= DreamingConfig::MAX_LORA_ALPHA,
            "Dreaming LoRA geometry exceeds operational limits"
        );
        let trial_rng = txn
            .dream_trial_rngs
            .iter()
            .find(|entry| entry.candidate_id == candidate.id)
            .with_context(|| format!("dream `{}` has no trial RNG reservation", candidate.id))?
            .reservation;
        let artifact = self.read_candidate(&candidate.artifact_hash)?;
        ensure!(
            artifact.transaction_id == txn.id,
            "candidate transaction mismatch"
        );
        self.config.independent_evaluation_set.verify_bytes()?;
        let base_checkpoint_sha256 = txn
            .candidate_hash
            .as_deref()
            .context("Dreaming LoRA transaction has no immutable candidate checkpoint hash")?;
        validate_sha256_identity(base_checkpoint_sha256, "Dreaming LoRA base checkpoint")?;
        let base_model_parameter_sha256 = txn
            .dream_shared_checkpoint_hash
            .as_deref()
            .context("Dreaming LoRA transaction has no shared model-parameter hash")?;
        validate_sha256_identity(
            base_model_parameter_sha256,
            "Dreaming LoRA base-model parameters",
        )?;

        // `valid` detaches the forked model and disables training-only dropout.
        // All optimization below happens in two isolated device-owned adapter
        // tensors; the frozen Transformer supplies only final hidden features
        // and base output-projection logits.
        let frozen_model = isolated_model.valid();
        let training = self
            .candidate_lm_head_rows(&frozen_model, &artifact)?
            .into_device(&self.device)?;
        let evaluation =
            self.evaluation_lm_head_rows(&frozen_model, base_model_parameter_sha256)?;
        let seed = derive_seed(txn, trial_rng, LORA_DOMAIN, 0);
        let adapter = HostLmHeadLora::new(training.hidden, training.vocab, rank, alpha, seed)?;
        let mut adapter = DeviceLmHeadLora::from_host(adapter, &self.device);
        let training_loss_before = adapter.loss_value(&training)?;
        let evaluation_loss_before = adapter.mean_loss_value(&evaluation)?;
        for _ in 0..self.config.lora_steps {
            adapter.train_step(&training, self.config.lora_learning_rate)?;
        }
        let training_loss_after = adapter.loss_value(&training)?;
        let evaluation_loss_after = adapter.mean_loss_value(&evaluation)?;
        let improvement = evaluation_loss_before - evaluation_loss_after;
        ensure!(
            improvement.is_finite(),
            "Dreaming trial produced a non-finite reward"
        );
        let adapter = adapter.into_host()?;

        let receipt = AdapterReceipt {
            version: DREAM_ADAPTER_VERSION,
            transaction_id: txn.id,
            base_checkpoint_sha256: base_checkpoint_sha256.to_owned(),
            base_model_parameter_sha256: base_model_parameter_sha256.to_owned(),
            candidate_id: candidate.id.clone(),
            candidate_artifact_hash: candidate.artifact_hash.clone(),
            evaluator_hash: self.config.independent_evaluation_set.sha256.clone(),
            target_module: DREAM_LORA_TARGET_MODULE.to_owned(),
            input_features: adapter.hidden,
            output_features: adapter.vocab,
            rank,
            alpha,
            steps: self.config.lora_steps,
            learning_rate: self.config.lora_learning_rate,
            training_loss_before,
            training_loss_after,
            evaluation_loss_before,
            evaluation_loss_after,
            independent_task_improvement: improvement,
            a_shape: [adapter.rank, adapter.hidden],
            b_shape: [adapter.rank, adapter.vocab],
        };
        receipt.validate()?;
        let header = serde_json::to_vec(&receipt)?;
        let capacity = adapter_payload_capacity(header.len(), adapter.a.len(), adapter.b.len())?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .context("reserving isolated dream LoRA artifact")?;
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for value in adapter.a.iter().chain(&adapter.b) {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let adapter_hash = sha256_identity(&bytes);
        publish_immutable(&self.adapter_path(&adapter_hash)?, &bytes)?;
        Ok(DreamTrial {
            candidate_id: candidate.id.clone(),
            adapter_hash,
            evaluator_hash: self.config.independent_evaluation_set.sha256.clone(),
            independent_task_improvement: improvement,
        })
    }

    fn restem_update_policy(
        &mut self,
        txn: &ConsolidationTxn,
        model: &Transformer,
        accepted: &[DreamTrial],
        iterations: usize,
    ) -> Result<String> {
        self.validate_transaction(txn)?;
        ensure!(iterations > 0, "ReSTEM iterations must be positive");
        ensure!(
            iterations <= DreamingConfig::MAX_RESTEM_ITERATIONS,
            "ReSTEM iteration count exceeds the operational limit"
        );
        ensure!(
            accepted.len() <= DreamingConfig::MAX_CANDIDATES,
            "ReSTEM accepted-trial count exceeds the operational limit"
        );
        let policy_work = accepted
            .len()
            .checked_mul(iterations)
            .context("ReSTEM policy work overflows usize")?;
        ensure!(
            policy_work <= DreamingConfig::MAX_POLICY_WORK,
            "ReSTEM policy work exceeds the operational limit"
        );
        self.validate_generation_policy_for_model(model)?;
        let source_checkpoint_sha256 = txn
            .candidate_hash
            .as_deref()
            .context("ReSTEM transaction has no immutable candidate checkpoint hash")?;
        validate_sha256_identity(source_checkpoint_sha256, "ReSTEM source checkpoint")?;
        let source_model_parameter_sha256 = txn
            .dream_shared_checkpoint_hash
            .as_deref()
            .context("ReSTEM transaction has no shared model-parameter hash")?;
        validate_sha256_identity(
            source_model_parameter_sha256,
            "ReSTEM source model parameters",
        )?;
        ensure!(
            model_parameter_hash(model)? == source_model_parameter_sha256,
            "ReSTEM shared model parameters do not match the transaction binding"
        );
        let topology_sha256 = model_topology_sha256(model)?;
        let manifest_hash = txn
            .generated_manifest
            .as_deref()
            .context("ReSTEM transaction has no generated manifest")?;
        let dreams = self.load(txn, manifest_hash)?;
        let by_id = dreams
            .iter()
            .map(|dream| (dream.id.as_str(), dream))
            .collect::<BTreeMap<_, _>>();
        let mut verified = Vec::with_capacity(accepted.len());
        let mut seen = BTreeSet::new();
        for trial in accepted {
            ensure!(
                seen.insert(trial.candidate_id.as_str()),
                "ReSTEM repeats a trial"
            );
            let dream = by_id.get(trial.candidate_id.as_str()).with_context(|| {
                format!(
                    "accepted dream `{}` is absent from manifest",
                    trial.candidate_id
                )
            })?;
            let receipt = self.read_adapter_receipt(&trial.adapter_hash)?;
            ensure!(
                receipt.transaction_id == txn.id
                    && Some(receipt.base_checkpoint_sha256.as_str())
                        == txn.candidate_hash.as_deref()
                    && Some(receipt.base_model_parameter_sha256.as_str())
                        == txn.dream_shared_checkpoint_hash.as_deref()
                    && receipt.candidate_id == trial.candidate_id
                    && receipt.candidate_artifact_hash == dream.artifact_hash
                    && receipt.evaluator_hash == self.config.independent_evaluation_set.sha256
                    && receipt.independent_task_improvement.to_bits()
                        == trial.independent_task_improvement.to_bits(),
                "accepted trial does not match its isolated adapter receipt"
            );
            ensure!(
                receipt.independent_task_improvement > 0.0,
                "ReSTEM received a rejected trial"
            );
            let candidate = self.read_candidate(&dream.artifact_hash)?;
            verified.push((dream, receipt, candidate, trial.adapter_hash.clone()));
        }
        verified.sort_by(|left, right| left.0.id.cmp(&right.0.id));

        let parent_policy_sha256 = self
            .config
            .initial_policy
            .as_ref()
            .map(|artifact| artifact.sha256.clone());
        let parent_adapter_sha256 = self
            .initial_policy
            .as_ref()
            .map(|policy| policy.adapter_sha256.clone());
        let mut adapter = match &self.generation_policy {
            Some(adapter) => adapter.clone(),
            None => HostLmHeadLora::new(
                model.config().hidden_size,
                model.config().vocab_size,
                self.config.generation_policy_rank,
                self.config.generation_policy_alpha,
                derive_seed(
                    txn,
                    txn.dream_generation_rng
                        .context("ReSTEM transaction has no generation RNG reservation")?,
                    LORA_DOMAIN ^ GENERATION_DOMAIN,
                    0,
                ),
            )?,
        };
        let frozen_model = model.clone().valid();
        let training_sets = verified
            .iter()
            .map(|(_, _, candidate, _)| self.dream_sequence_lm_head_rows(&frozen_model, candidate))
            .collect::<Result<Vec<_>>>()?;
        if !training_sets.is_empty() {
            let training_sets = training_sets
                .into_iter()
                .map(|rows| rows.into_device(&self.device))
                .collect::<Result<Vec<_>>>()?;
            let mut device_adapter = DeviceLmHeadLora::from_host(adapter, &self.device);
            let reward_sum = verified
                .iter()
                .map(|(_, receipt, _, _)| receipt.independent_task_improvement)
                .sum::<f32>();
            ensure!(
                reward_sum.is_finite() && reward_sum > 0.0,
                "ReSTEM accepted rewards do not have positive finite mass"
            );
            for _ in 0..iterations {
                for (data, (_, receipt, _, _)) in training_sets.iter().zip(&verified) {
                    let reward_weight =
                        receipt.independent_task_improvement * verified.len() as f32 / reward_sum;
                    device_adapter
                        .train_step(data, self.config.restem_learning_rate * reward_weight)?;
                }
            }
            adapter = device_adapter.into_host()?;
        }
        let accepted_candidates = verified
            .iter()
            .map(|(dream, _, _, _)| dream.artifact_hash.clone())
            .collect::<Vec<_>>();
        let accepted_adapters = verified
            .iter()
            .map(|(_, _, _, adapter_hash)| adapter_hash.clone())
            .collect::<Vec<_>>();
        let accepted_rewards = verified
            .iter()
            .map(|(_, receipt, _, _)| receipt.independent_task_improvement)
            .collect::<Vec<_>>();
        let applied_iterations = if verified.is_empty() { 0 } else { iterations };
        let adapter_sha256 = if verified.is_empty() && parent_adapter_sha256.is_some() {
            parent_adapter_sha256.clone().unwrap()
        } else {
            let receipt = GenerationPolicyAdapterReceipt {
                version: GENERATION_POLICY_ADAPTER_VERSION,
                transaction_id: txn.id,
                source_checkpoint_sha256: source_checkpoint_sha256.to_owned(),
                source_model_parameter_sha256: source_model_parameter_sha256.to_owned(),
                topology_sha256: topology_sha256.clone(),
                parent_policy_sha256: parent_policy_sha256.clone(),
                parent_adapter_sha256: parent_adapter_sha256.clone(),
                target_module: GENERATION_POLICY_TARGET_MODULE.to_owned(),
                input_features: adapter.hidden,
                output_features: adapter.vocab,
                rank: adapter.rank,
                alpha: adapter.alpha,
                iterations: applied_iterations,
                learning_rate: self.config.restem_learning_rate,
                accepted_candidates: accepted_candidates.clone(),
                accepted_trial_adapters: accepted_adapters.clone(),
                accepted_rewards,
                a_shape: [adapter.rank, adapter.hidden],
                b_shape: [adapter.rank, adapter.vocab],
            };
            receipt.validate()?;
            let header = serde_json::to_vec(&receipt)?;
            let capacity =
                adapter_payload_capacity(header.len(), adapter.a.len(), adapter.b.len())?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(capacity)
                .context("reserving generation-policy LoRA artifact")?;
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&header);
            for value in adapter.a.iter().chain(&adapter.b) {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            let hash = sha256_identity(&bytes);
            publish_immutable(&self.generation_policy_adapter_path(&hash)?, &bytes)?;
            hash
        };
        let state = RestemPolicyState {
            version: RESTEM_POLICY_VERSION,
            parent_policy_sha256,
            parent_adapter_sha256,
            transaction_id: txn.id,
            source_checkpoint_sha256: source_checkpoint_sha256.to_owned(),
            source_model_parameter_sha256: source_model_parameter_sha256.to_owned(),
            topology_sha256,
            adapter_sha256,
            target_module: GENERATION_POLICY_TARGET_MODULE.to_owned(),
            input_features: adapter.hidden,
            output_features: adapter.vocab,
            rank: adapter.rank,
            alpha: adapter.alpha,
            // An all-rejected set publishes a transaction-bound node while
            // carrying the exact authenticated parent adapter forward.
            iterations: applied_iterations,
            learning_rate: self.config.restem_learning_rate,
            accepted_candidates,
            accepted_adapters,
        };
        state.validate()?;
        let bytes = canonical_json(&state)?;
        let hash = sha256_identity(&bytes);
        publish_immutable(&self.policy_path(&hash)?, &bytes)?;
        Ok(hash)
    }
}

#[derive(Clone)]
struct LmHeadRows {
    rows: usize,
    hidden: usize,
    vocab: usize,
    features: Vec<f32>,
    base_logits: Vec<f32>,
    targets: Vec<usize>,
}

fn checked_lm_head_row_geometry(
    rows: usize,
    hidden: usize,
    vocab: usize,
) -> Result<(usize, usize)> {
    ensure!(rows > 0 && hidden > 0 && vocab > 1, "invalid LM-head rows");
    let feature_values = rows
        .checked_mul(hidden)
        .context("LM-head feature geometry overflows usize")?;
    let logit_values = rows
        .checked_mul(vocab)
        .context("LM-head logit geometry overflows usize")?;
    let data_bytes = feature_values
        .checked_add(logit_values)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("LM-head dataset geometry overflows usize")?;
    ensure!(
        data_bytes <= MAX_DREAM_LM_HEAD_DATA_BYTES,
        "LM-head dataset exceeds the {MAX_DREAM_LM_HEAD_DATA_BYTES}-byte operational limit"
    );
    Ok((feature_values, logit_values))
}

impl LmHeadRows {
    fn validate_for(&self, hidden: usize, vocab: usize) -> Result<()> {
        let (feature_values, logit_values) =
            checked_lm_head_row_geometry(self.rows, hidden, vocab)?;
        ensure!(
            self.hidden == hidden
                && self.vocab == vocab
                && self.rows == self.targets.len()
                && self.features.len() == feature_values
                && self.base_logits.len() == logit_values,
            "LM-head LoRA dataset shape mismatch"
        );
        ensure!(
            self.targets.iter().all(|target| *target < vocab),
            "LM-head LoRA target is outside vocabulary"
        );
        Ok(())
    }

    fn into_device(self, device: &Device) -> Result<DeviceLmHeadRows> {
        self.validate_for(self.hidden, self.vocab)?;
        let rows = self.rows;
        let hidden = self.hidden;
        let vocab = self.vocab;
        let targets = self
            .targets
            .into_iter()
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("LM-head LoRA target exceeds i64")?;
        Ok(DeviceLmHeadRows {
            rows,
            hidden,
            vocab,
            features: Tensor::<2>::from_data(
                TensorData::new(self.features, [rows, hidden]),
                device,
            ),
            base_logits: Tensor::<2>::from_data(
                TensorData::new(self.base_logits, [rows, vocab]),
                device,
            ),
            targets: Tensor::<1, Int>::from_data(TensorData::new(targets, [rows]), device),
        })
    }
}

#[derive(Clone)]
struct DeviceLmHeadRows {
    rows: usize,
    hidden: usize,
    vocab: usize,
    features: Tensor<2>,
    base_logits: Tensor<2>,
    targets: Tensor<1, Int>,
}

struct DeviceEvaluationCache {
    model_parameter_sha256: String,
    rows: Vec<DeviceLmHeadRows>,
}

struct DeviceLmHeadLora {
    hidden: usize,
    vocab: usize,
    rank: usize,
    alpha: usize,
    scale: f32,
    a: Tensor<2>,
    b: Tensor<2>,
}

struct DeviceLmHeadPolicy {
    hidden: usize,
    vocab: usize,
    scale: f32,
    a: Tensor<2>,
    b: Tensor<2>,
}

impl DeviceLmHeadPolicy {
    fn from_host(adapter: &HostLmHeadLora, device: &Device) -> Self {
        Self {
            hidden: adapter.hidden,
            vocab: adapter.vocab,
            scale: adapter.scale,
            a: Tensor::<2>::from_data(
                TensorData::new(adapter.a.clone(), [adapter.rank, adapter.hidden]),
                device,
            ),
            b: Tensor::<2>::from_data(
                TensorData::new(adapter.b.clone(), [adapter.rank, adapter.vocab]),
                device,
            ),
        }
    }

    fn adapted(&self, features: Tensor<2>, base_logits: Tensor<2>) -> Result<Tensor<2>> {
        ensure!(
            features.dims() == [1, self.hidden] && base_logits.dims() == [1, self.vocab],
            "device generation-policy feature/logit shape mismatch"
        );
        Ok(base_logits
            + features
                .matmul(self.a.clone().transpose())
                .matmul(self.b.clone())
                .mul_scalar(self.scale))
    }
}

impl DeviceLmHeadLora {
    fn from_host(adapter: HostLmHeadLora, device: &Device) -> Self {
        let a = Tensor::<2>::from_data(
            TensorData::new(adapter.a, [adapter.rank, adapter.hidden]),
            device,
        )
        .require_grad();
        let b = Tensor::<2>::from_data(
            TensorData::new(adapter.b, [adapter.rank, adapter.vocab]),
            device,
        )
        .require_grad();
        Self {
            hidden: adapter.hidden,
            vocab: adapter.vocab,
            rank: adapter.rank,
            alpha: adapter.alpha,
            scale: adapter.scale,
            a,
            b,
        }
    }

    fn loss(&self, data: &DeviceLmHeadRows) -> Result<Tensor<1>> {
        ensure!(
            data.rows > 0 && data.hidden == self.hidden && data.vocab == self.vocab,
            "device LM-head LoRA dataset shape mismatch"
        );
        let logits = data.base_logits.clone()
            + data
                .features
                .clone()
                .matmul(self.a.clone().transpose())
                .matmul(self.b.clone())
                .mul_scalar(self.scale);
        Ok(CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, data.targets.clone()))
    }

    fn loss_value(&self, data: &DeviceLmHeadRows) -> Result<f32> {
        let value = self
            .loss(data)?
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading device LM-head LoRA loss")?[0];
        ensure!(value.is_finite(), "device LM-head LoRA loss is non-finite");
        Ok(value)
    }

    fn mean_loss_value(&self, sets: &[DeviceLmHeadRows]) -> Result<f32> {
        ensure!(!sets.is_empty(), "Dreaming evaluation set is empty");
        let losses = sets
            .iter()
            .map(|set| self.loss(set))
            .collect::<Result<Vec<_>>>()?;
        let value = Tensor::cat(losses, 0)
            .mean()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading mean device LM-head LoRA loss")?[0];
        ensure!(
            value.is_finite(),
            "mean device LM-head LoRA loss is non-finite"
        );
        Ok(value)
    }

    fn train_step(&mut self, data: &DeviceLmHeadRows, learning_rate: f32) -> Result<()> {
        ensure!(
            learning_rate.is_finite() && learning_rate > 0.0,
            "device LM-head LoRA learning rate is invalid"
        );
        let mut gradients = self.loss(data)?.backward();
        let grad_a = self
            .a
            .grad_remove(&mut gradients)
            .context("device LM-head LoRA produced no A gradient")?;
        let grad_b = self
            .b
            .grad_remove(&mut gradients)
            .context("device LM-head LoRA produced no B gradient")?;
        let mean_gradient_norm = (grad_a.clone().square().sum() + grad_b.clone().square().sum())
            .sqrt()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading device LM-head LoRA gradient norm")?[0];
        ensure!(
            mean_gradient_norm.is_finite(),
            "device LM-head LoRA gradient norm is non-finite"
        );
        // The paper-inspired host recipe clipped the summed full-batch
        // gradient. CrossEntropyLoss emits a mean gradient, so restore the sum
        // norm before applying the same update rule.
        let denominator = (mean_gradient_norm * data.rows as f32).max(1.0);
        let rate = learning_rate / denominator;
        self.a =
            Tensor::from_inner(self.a.clone().inner() - grad_a.mul_scalar(rate)).require_grad();
        self.b =
            Tensor::from_inner(self.b.clone().inner() - grad_b.mul_scalar(rate)).require_grad();
        Ok(())
    }

    fn into_host(self) -> Result<HostLmHeadLora> {
        let a = self
            .a
            .detach()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading trained LM-head LoRA A")?;
        let b = self
            .b
            .detach()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading trained LM-head LoRA B")?;
        HostLmHeadLora::from_parts(self.hidden, self.vocab, self.rank, self.alpha, a, b)
    }
}

/// Isolated LoRA on the frozen model's logical output projection.
///
/// `A` is row-major `[rank, hidden]`, `B` is row-major `[rank, vocab]`,
/// and the delta is `(hidden @ A^T) @ B * alpha/rank`.  Base Transformer
/// parameters and base logits are immutable inputs to this host-side trial.
#[derive(Clone)]
struct HostLmHeadLora {
    hidden: usize,
    vocab: usize,
    rank: usize,
    alpha: usize,
    scale: f32,
    a: Vec<f32>,
    b: Vec<f32>,
}

fn checked_lora_parameter_geometry(
    hidden: usize,
    vocab: usize,
    rank: usize,
    alpha: usize,
) -> Result<(usize, usize)> {
    ensure!(
        hidden > 0 && vocab > 1 && rank > 0 && alpha > 0,
        "invalid LM-head LoRA shape"
    );
    ensure!(
        rank <= DreamingConfig::MAX_LORA_RANK && alpha <= DreamingConfig::MAX_LORA_ALPHA,
        "LM-head LoRA geometry exceeds the configured operational limits"
    );
    let a_values = rank
        .checked_mul(hidden)
        .context("LM-head LoRA A shape overflow")?;
    let b_values = rank
        .checked_mul(vocab)
        .context("LM-head LoRA B shape overflow")?;
    let parameter_bytes = a_values
        .checked_add(b_values)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("LM-head LoRA parameter size overflow")?;
    ensure!(
        u64::try_from(parameter_bytes).context("LM-head LoRA parameter size exceeds u64")?
            <= MAX_DREAM_ADAPTER_PARAMETER_BYTES,
        "LM-head LoRA parameters exceed the {MAX_DREAM_ADAPTER_PARAMETER_BYTES}-byte operational limit"
    );
    Ok((a_values, b_values))
}

fn adapter_payload_capacity(
    header_bytes: usize,
    a_values: usize,
    b_values: usize,
) -> Result<usize> {
    let capacity = a_values
        .checked_add(b_values)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|parameter_bytes| parameter_bytes.checked_add(header_bytes))
        .and_then(|payload| payload.checked_add(std::mem::size_of::<u64>()))
        .context("dream LoRA adapter payload size overflow")?;
    ensure!(
        u64::try_from(capacity).context("dream LoRA adapter payload exceeds u64")?
            <= MAX_DREAM_ADAPTER_BYTES,
        "dream LoRA adapter payload exceeds the {MAX_DREAM_ADAPTER_BYTES}-byte limit"
    );
    Ok(capacity)
}

fn zeroed_f32(count: usize, label: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .with_context(|| format!("reserving {label}"))?;
    values.resize(count, 0.0);
    Ok(values)
}

impl HostLmHeadLora {
    fn new(hidden: usize, vocab: usize, rank: usize, alpha: usize, seed: u64) -> Result<Self> {
        let (a_values, b_values) = checked_lora_parameter_geometry(hidden, vocab, rank, alpha)?;
        let mut state = seed;
        let mut a = Vec::new();
        a.try_reserve_exact(a_values)
            .context("reserving LM-head LoRA A parameters")?;
        for _ in 0..a_values {
            state = mix64(state);
            a.push((((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32) * 0.01);
        }
        let b = zeroed_f32(b_values, "LM-head LoRA B parameters")?;
        Ok(Self {
            hidden,
            vocab,
            rank,
            alpha,
            scale: alpha as f32 / rank as f32,
            a,
            b,
        })
    }

    fn from_parts(
        hidden: usize,
        vocab: usize,
        rank: usize,
        alpha: usize,
        a: Vec<f32>,
        b: Vec<f32>,
    ) -> Result<Self> {
        let (a_values, b_values) = checked_lora_parameter_geometry(hidden, vocab, rank, alpha)?;
        ensure!(
            a.len() == a_values && b.len() == b_values,
            "LM-head LoRA parameter lengths do not match geometry"
        );
        ensure!(
            a.iter().chain(&b).all(|value| value.is_finite()),
            "LM-head LoRA contains a non-finite parameter"
        );
        Ok(Self {
            hidden,
            vocab,
            rank,
            alpha,
            scale: alpha as f32 / rank as f32,
            a,
            b,
        })
    }

    #[cfg(test)]
    fn adapted(&self, features: &[f32], base_logits: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            features.len() == self.hidden,
            "LM-head LoRA row has wrong hidden size"
        );
        ensure!(
            base_logits.len() == self.vocab,
            "LM-head LoRA row has wrong vocabulary size"
        );
        ensure!(
            features
                .iter()
                .chain(base_logits)
                .all(|value| value.is_finite()),
            "LM-head LoRA input is non-finite"
        );
        let mut low_rank = zeroed_f32(self.rank, "LM-head LoRA activation")?;
        for (rank, low_rank_value) in low_rank.iter_mut().enumerate() {
            for (feature, value) in features.iter().copied().enumerate() {
                *low_rank_value += value * self.a[rank * self.hidden + feature];
            }
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(base_logits.len())
            .context("reserving adapted LM-head logits")?;
        output.extend_from_slice(base_logits);
        for (token, output_value) in output.iter_mut().enumerate() {
            let mut residual = 0.0;
            for (rank, low_rank_value) in low_rank.iter().copied().enumerate() {
                residual += low_rank_value * self.b[rank * self.vocab + token];
            }
            *output_value += self.scale * residual;
        }
        Ok((low_rank, output))
    }

    #[cfg(test)]
    fn loss(&self, data: &LmHeadRows) -> Result<f32> {
        data.validate_for(self.hidden, self.vocab)?;
        let mut sum = 0.0;
        for row in 0..data.rows {
            let (_, logits) = self.adapted(
                &data.features[row * self.hidden..(row + 1) * self.hidden],
                &data.base_logits[row * self.vocab..(row + 1) * self.vocab],
            )?;
            sum += cross_entropy(&logits, data.targets[row])?;
        }
        Ok(sum / data.rows as f32)
    }

    #[cfg(test)]
    fn train_step(&mut self, data: &LmHeadRows, learning_rate: f32) -> Result<()> {
        data.validate_for(self.hidden, self.vocab)?;
        let mut grad_a = zeroed_f32(self.a.len(), "LM-head LoRA A gradient")?;
        let mut grad_b = zeroed_f32(self.b.len(), "LM-head LoRA B gradient")?;
        for row in 0..data.rows {
            let features = &data.features[row * self.hidden..(row + 1) * self.hidden];
            let base_logits = &data.base_logits[row * self.vocab..(row + 1) * self.vocab];
            let (low_rank, logits) = self.adapted(features, base_logits)?;
            let mut error = softmax_host(&logits)?;
            error[data.targets[row]] -= 1.0;
            let mut grad_low_rank = zeroed_f32(self.rank, "LM-head LoRA activation gradient")?;
            for (rank, grad_low_rank_value) in grad_low_rank.iter_mut().enumerate() {
                for (token, error_value) in error.iter().copied().enumerate() {
                    grad_b[rank * self.vocab + token] += self.scale * low_rank[rank] * error_value;
                    *grad_low_rank_value +=
                        self.scale * error_value * self.b[rank * self.vocab + token];
                }
            }
            for (rank, grad_low_rank_value) in grad_low_rank.iter().copied().enumerate() {
                for (feature, feature_value) in features.iter().copied().enumerate() {
                    grad_a[rank * self.hidden + feature] += feature_value * grad_low_rank_value;
                }
            }
        }
        let rate = learning_rate / data.rows as f32;
        let norm = grad_a
            .iter()
            .chain(&grad_b)
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1.0);
        let rate = rate / norm;
        for (value, gradient) in self.a.iter_mut().zip(grad_a) {
            *value -= rate * gradient;
        }
        for (value, gradient) in self.b.iter_mut().zip(grad_b) {
            *value -= rate * gradient;
        }
        ensure!(
            self.a.iter().chain(&self.b).all(|value| value.is_finite()),
            "LoRA update produced a non-finite parameter"
        );
        Ok(())
    }
}

#[cfg(test)]
fn softmax_host(logits: &[f32]) -> Result<Vec<f32>> {
    ensure!(!logits.is_empty(), "softmax input is empty");
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ensure!(max.is_finite(), "softmax input is non-finite");
    let mut values = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    ensure!(sum.is_finite() && sum > 0.0, "softmax normalization failed");
    for value in &mut values {
        *value /= sum;
    }
    Ok(values)
}

#[cfg(test)]
fn cross_entropy(logits: &[f32], target: usize) -> Result<f32> {
    ensure!(
        target < logits.len(),
        "cross-entropy target is out of range"
    );
    Ok(-softmax_host(logits)?[target].max(1e-30).ln())
}

struct GradientProjection<'a> {
    gradients: &'a GradientsParams,
    values: Vec<f64>,
    scalar_count: usize,
    failure: Option<String>,
}

impl ModuleVisitor for GradientProjection<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if self.failure.is_some() {
            return;
        }
        let Some(gradient) = self.gradients.get::<D>(parameter.id) else {
            return;
        };
        match gradient.into_data().convert::<f32>().to_vec::<f32>() {
            Ok(gradient) => {
                for (index, value) in gradient.into_iter().enumerate() {
                    if !value.is_finite() {
                        self.failure = Some(format!(
                            "parameter {} has a non-finite gradient",
                            parameter.id.val()
                        ));
                        return;
                    }
                    let mixed = mix64(
                        GRADIENT_PROJECTION_DOMAIN
                            ^ parameter.id.val().rotate_left(17)
                            ^ index as u64,
                    );
                    let bucket = mixed as usize % self.values.len();
                    let sign = if mixed & (1 << 63) == 0 { 1.0 } else { -1.0 };
                    self.values[bucket] += sign * f64::from(value);
                    self.scalar_count += 1;
                }
            }
            Err(error) => self.failure = Some(error.to_string()),
        }
    }
}

fn gradient_fingerprint(
    model: &Transformer,
    device: &Device,
    tokens: &[i64],
    dimensions: usize,
    seed: u64,
) -> Result<Vec<f32>> {
    ensure!(
        tokens.len() >= 2 && dimensions > 0,
        "invalid gradient fingerprint input"
    );
    device.seed(seed);
    let inputs = Tensor::<2, Int>::from_data(
        TensorData::new(tokens[..tokens.len() - 1].to_vec(), [1, tokens.len() - 1]),
        device,
    );
    let targets = Tensor::<2, Int>::from_data(
        TensorData::new(tokens[1..].to_vec(), [1, tokens.len() - 1]),
        device,
    );
    let mut raw = model.forward_loss(inputs, targets).backward();
    let gradients = GradientsParams::from_module(&mut raw, model);
    ensure!(
        !gradients.is_empty(),
        "Dreaming model loss produced no gradients"
    );
    let mut projection = GradientProjection {
        gradients: &gradients,
        values: vec![0.0; dimensions],
        scalar_count: 0,
        failure: None,
    };
    model.visit(&mut projection);
    if let Some(error) = projection.failure {
        bail!("projecting Dreaming loss gradients: {error}");
    }
    ensure!(
        projection.scalar_count > 0,
        "Dreaming model exposes no gradient tensors"
    );
    let mut values = projection
        .values
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    normalize(&mut values)?;
    Ok(values)
}

fn normalize(values: &mut [f32]) -> Result<()> {
    let norm = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "gradient fingerprint has zero/non-finite norm"
    );
    for value in values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

fn derive_seed(
    txn: &ConsolidationTxn,
    reservation: RngReservation,
    domain: u64,
    ordinal: u64,
) -> u64 {
    let mut hash = Sha256::new();
    hash.update(b"hermes-builtin-dreaming-seed-v1\0");
    hash.update(txn.id.to_le_bytes());
    hash.update(txn.trigger_clock.to_le_bytes());
    hash.update((reservation.stream as u64).to_le_bytes());
    hash.update(reservation.start.to_le_bytes());
    hash.update(reservation.count.to_le_bytes());
    hash.update(domain.to_le_bytes());
    hash.update(ordinal.to_le_bytes());
    let bytes = hash.finalize();
    u64::from_le_bytes(bytes[..8].try_into().unwrap())
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sample_temperature(logits: &[f32], temperature: f32, seed: u64) -> Result<usize> {
    ensure!(!logits.is_empty(), "sampling logits are empty");
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "sampling temperature must be finite and positive"
    );
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "sampling logits are non-finite"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = logits
        .iter()
        .map(|value| f64::from((*value - max) / temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    ensure!(
        total.is_finite() && total > 0.0,
        "sampling normalization failed"
    );
    let unit = (mix64(seed) >> 11) as f64 / (1_u64 << 53) as f64;
    let mut target = unit * total;
    for (index, weight) in weights.iter().copied().enumerate() {
        if target < weight {
            return Ok(index);
        }
        target -= weight;
    }
    Ok(logits.len() - 1)
}

fn model_topology_sha256(model: &Transformer) -> Result<String> {
    Ok(sha256_identity(&canonical_json(model.config())?))
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn short_hash(hash: &str) -> &str {
    &hash[7..19]
}

fn addressed_path(root: &Path, kind: &str, hash: &str, extension: &str) -> Result<PathBuf> {
    validate_sha256_identity(hash, kind)?;
    Ok(root
        .join(kind)
        .join(format!("{}.{}", &hash[7..], extension)))
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_regular_bounded(path, MAX_DREAM_ADAPTER_BYTES, label)
        .with_context(|| format!("failed to read {label} {}", path.display()))
}

fn read_regular_json(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_regular_bounded(path, MAX_DREAM_JSON_BYTES, label)
        .with_context(|| format!("failed to read {label} {}", path.display()))
}

fn read_addressed(path: &Path, expected: &str, label: &str) -> Result<Vec<u8>> {
    validate_sha256_identity(expected, label)?;
    let bytes = read_regular_file(path, label)?;
    let observed = sha256_identity(&bytes);
    ensure!(
        observed == expected,
        "{label} content hash changed: expected {expected}, observed {observed}"
    );
    Ok(bytes)
}

fn read_addressed_json(path: &Path, expected: &str, label: &str) -> Result<Vec<u8>> {
    validate_sha256_identity(expected, label)?;
    let bytes = read_regular_json(path, label)?;
    let observed = sha256_identity(&bytes);
    ensure!(
        observed == expected,
        "{label} content hash changed: expected {expected}, observed {observed}"
    );
    Ok(bytes)
}

fn ensure_real_directory(path: &Path) -> Result<()> {
    ensure_directory(path, "Dreaming directory")
}

fn publish_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_new(path, bytes)
}

#[cfg(test)]
mod tests {
    use hermes_llm::{ModelDef, parse_mal};
    use tempfile::TempDir;

    use super::*;
    use crate::builtin_sleep_adapters::{WakeContextJournal, WakeContextRecord};
    use crate::sleep::{DreamTrialRng, DreamingBackend};
    use crate::tensor_sleep::TensorDreamBackend;

    fn hash(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn tiny_config() -> ModelDef {
        let mut config = parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12 activation: swiglu dropout: 0.0
                moe { experts: 3 top_k: 1 shared_experts: 0 }
            }
            memory cms {
                tier fast { ffn: routed reserve_experts { capacity: 1 rank: 3 top_k: 1 } }
                tier slow { ffn: routed residual_init: zero reserve_experts { capacity: 1 rank: 3 top_k: 1 } }
            }
            model dream-test {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: {
                    attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
                    memory: cms
                    dropout: 0.0
                }
            }
            "#,
        )
        .unwrap();
        config.embeddings.dropout = 0.0;
        config
    }

    fn write_pinned<T: Serialize>(directory: &Path, name: &str, value: &T) -> PinnedLocalArtifact {
        let path = directory.join(name);
        let bytes = canonical_json(value).unwrap();
        fs::write(&path, &bytes).unwrap();
        PinnedLocalArtifact {
            path,
            sha256: sha256_identity(&bytes),
        }
    }

    fn transaction() -> ConsolidationTxn {
        ConsolidationTxn {
            id: 41,
            trigger_clock: 10,
            sender: 0,
            receiver: 1,
            receiver_slot: 0,
            terminal: false,
            sender_slots_to_reset: vec![0],
            teacher_checkpoint: "teacher.safetensors".into(),
            teacher_hash: hash('a'),
            student_checkpoint: "student.safetensors".into(),
            student_hash: hash('b'),
            prospective_update_hash: hash('c'),
            candidate_checkpoint: Some("candidate.safetensors".into()),
            candidate_hash: Some(hash('d')),
            knowledge_rng: None,
            imitation_rng: None,
            dream_generation_rng: Some(RngReservation {
                stream: 2,
                start: 7,
                count: 4,
            }),
            dream_selection_rng: Some(RngReservation {
                stream: 3,
                start: 11,
                count: 4,
            }),
            dream_trial_rngs: Vec::new(),
            tensor_transaction_generation: Some("txn-41".into()),
            tensor_transaction_manifest_hash: Some(hash('e')),
            generated_manifest: None,
            dream_shared_checkpoint_hash: None,
            dream_selected: Vec::new(),
            dream_trials: Vec::new(),
            dream_policy_receipt: None,
            committed: true,
        }
    }

    struct Seed {
        _directory: TempDir,
        device: Device,
        model: Transformer,
        config: BuiltinDreamingRuntimeConfig,
        txn: ConsolidationTxn,
    }

    fn seed(evaluation: Vec<i64>, lora_steps: usize) -> Seed {
        let directory = tempfile::tempdir().unwrap();
        let mut journal = WakeContextJournal::new(hash('a')).unwrap();
        journal
            .push(WakeContextRecord {
                id: "wake-10-0".into(),
                optimizer_step: 10,
                token_ids: vec![2, 3, 4, 1],
            })
            .unwrap();
        let journal_path = directory.path().join("wake.json");
        let journal = journal.publish(&journal_path).unwrap();
        let reference_set = DreamSequenceSet {
            version: DREAM_SEQUENCE_SET_VERSION,
            sequences: vec![DreamSequence {
                id: "reference".into(),
                token_ids: vec![1, 2, 3, 4],
            }],
        };
        let evaluation_set = DreamSequenceSet {
            version: DREAM_SEQUENCE_SET_VERSION,
            sequences: vec![DreamSequence {
                id: "evaluation".into(),
                token_ids: evaluation,
            }],
        };
        let config = BuiltinDreamingRuntimeConfig {
            artifact_directory: directory.path().join("artifacts"),
            wake_context_journal: Some(PinnedLocalArtifact {
                path: journal.path().to_owned(),
                sha256: journal.sha256().to_owned(),
            }),
            reference_set: write_pinned(directory.path(), "reference.json", &reference_set),
            independent_evaluation_set: write_pinned(
                directory.path(),
                "evaluation.json",
                &evaluation_set,
            ),
            initial_policy: None,
            max_new_tokens: 1,
            gradient_dimensions: 16,
            lora_steps,
            lora_learning_rate: 0.25,
            restem_learning_rate: 0.1,
            generation_temperature: 0.8,
            generation_policy_rank: 4,
            generation_policy_alpha: 8,
        };
        let device = Device::ndarray().autodiff();
        device.seed(17);
        let model = Transformer::new(&tiny_config(), &device).unwrap();
        Seed {
            _directory: directory,
            device,
            model,
            config,
            txn: transaction(),
        }
    }

    fn manual_candidate(
        operations: &BuiltinDreamOps,
        txn: &ConsolidationTxn,
        tokens: Vec<i64>,
    ) -> GeneratedDream {
        let ordinal = 0;
        let generation_seed = derive_seed(
            txn,
            txn.dream_generation_rng.unwrap(),
            GENERATION_DOMAIN,
            ordinal,
        );
        let artifact = DreamCandidateArtifact {
            version: DREAM_CANDIDATE_VERSION,
            transaction_id: txn.id,
            ordinal: ordinal as usize,
            wake_journal_sha256: operations.journal.sha256().to_owned(),
            source_checkpoint_sha256: operations.journal.source_checkpoint_sha256().to_owned(),
            wake_record_id: operations.journal.records()[0].id.clone(),
            wake_token_count: 1,
            generation_seed,
            routing_seed: txn.id,
            routing: DREAM_ROUTING_NAME.into(),
            generation_policy_sha256: operations
                .config
                .initial_policy
                .as_ref()
                .map(|artifact| artifact.sha256.clone()),
            generation_policy_adapter_sha256: operations
                .initial_policy
                .as_ref()
                .map(|policy| policy.adapter_sha256.clone()),
            generation_temperature: operations.config.generation_temperature,
            token_ids: tokens,
        };
        let bytes = canonical_json(&artifact).unwrap();
        let artifact_hash = sha256_identity(&bytes);
        publish_immutable(&operations.candidate_path(&artifact_hash).unwrap(), &bytes).unwrap();
        GeneratedDream {
            id: format!("dream-{}-{ordinal}-{}", txn.id, short_hash(&artifact_hash)),
            artifact_hash,
            gradient: vec![0.25; operations.config.gradient_dimensions],
            diversity_key: mix64(generation_seed ^ ordinal),
        }
    }

    fn publish_manifest(
        operations: &BuiltinDreamOps,
        txn: &ConsolidationTxn,
        dreams: Vec<GeneratedDream>,
    ) -> String {
        let manifest = DreamManifest {
            version: DREAM_MANIFEST_VERSION,
            transaction_id: txn.id,
            wake_journal_sha256: operations.journal.sha256().to_owned(),
            source_checkpoint_sha256: operations.journal.source_checkpoint_sha256().to_owned(),
            generation_reservation: txn.dream_generation_rng.unwrap(),
            routing: DREAM_ROUTING_NAME.into(),
            generation_policy_sha256: operations
                .config
                .initial_policy
                .as_ref()
                .map(|artifact| artifact.sha256.clone()),
            generation_policy_adapter_sha256: operations
                .initial_policy
                .as_ref()
                .map(|policy| policy.adapter_sha256.clone()),
            generation_temperature: operations.config.generation_temperature,
            dreams,
        };
        let bytes = canonical_json(&manifest).unwrap();
        let hash = sha256_identity(&bytes);
        publish_immutable(&operations.manifest_path(&hash).unwrap(), &bytes).unwrap();
        hash
    }

    #[test]
    fn dreaming_json_reads_reject_oversized_input_before_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.json");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_DREAM_JSON_BYTES + 1).unwrap();
        drop(file);

        let error = read_regular_json(&path, "dreaming fixture").unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("byte limit"), "{error}");
    }

    #[test]
    fn dream_sequence_sets_reject_semantically_unbounded_work() {
        let sequence = |id: String, token_ids: Vec<i64>| DreamSequence { id, token_ids };
        let too_many = DreamSequenceSet {
            version: DREAM_SEQUENCE_SET_VERSION,
            sequences: (0..=MAX_DREAM_SEQUENCES)
                .map(|index| sequence(format!("sequence-{index}"), vec![1, 2]))
                .collect(),
        };
        let error = too_many.validate().unwrap_err().to_string();
        assert!(error.contains("sequence operational limit"), "{error}");

        let too_long = DreamSequenceSet {
            version: DREAM_SEQUENCE_SET_VERSION,
            sequences: vec![sequence(
                "too-long".into(),
                vec![1; MAX_DREAM_SEQUENCE_TOKENS + 1],
            )],
        };
        let error = too_long.validate().unwrap_err().to_string();
        assert!(error.contains("token operational limit"), "{error}");

        let invalid_token = DreamSequenceSet {
            version: DREAM_SEQUENCE_SET_VERSION,
            sequences: vec![sequence("invalid-token".into(), vec![1, i64::MAX])],
        };
        let error = invalid_token.validate().unwrap_err().to_string();
        assert!(error.contains("outside the u32 range"), "{error}");
    }

    #[test]
    fn lm_head_lora_starts_as_an_exact_base_noop_and_updates_only_adapter_tensors() {
        let data = LmHeadRows {
            rows: 1,
            hidden: 2,
            vocab: 3,
            features: vec![0.5, -1.0],
            base_logits: vec![0.1, -0.2, 0.3],
            targets: vec![1],
        };
        let mut adapter = HostLmHeadLora::new(2, 3, 2, 4, 17).unwrap();
        let (_, initial_logits) = adapter.adapted(&data.features, &data.base_logits).unwrap();
        assert_eq!(initial_logits, data.base_logits);

        let base_before = data.base_logits.clone();
        let features_before = data.features.clone();
        let a_before = adapter.a.clone();
        adapter.train_step(&data, 0.25).unwrap();

        // Standard zero-B initialization gives A no first-step gradient, while
        // B becomes the only changed state. Frozen features/base logits stay
        // byte-for-byte untouched.
        assert_eq!(adapter.a, a_before);
        assert!(adapter.b.iter().any(|value| *value != 0.0));
        assert_eq!(data.base_logits, base_before);
        assert_eq!(data.features, features_before);
    }

    #[test]
    fn device_lm_head_lora_uses_batched_tensor_updates() {
        let rows = LmHeadRows {
            rows: 2,
            hidden: 2,
            vocab: 3,
            features: vec![0.5, -1.0, -0.25, 0.75],
            base_logits: vec![0.1, -0.2, 0.3, -0.4, 0.2, 0.1],
            targets: vec![1, 2],
        };
        let device = Device::ndarray().autodiff();
        let rows = rows.into_device(&device).unwrap();
        let adapter = HostLmHeadLora::new(2, 3, 2, 4, 17).unwrap();
        let mut adapter = DeviceLmHeadLora::from_host(adapter, &device);
        let before = adapter.loss_value(&rows).unwrap();
        for _ in 0..4 {
            adapter.train_step(&rows, 0.25).unwrap();
        }
        let after = adapter.loss_value(&rows).unwrap();
        assert!(after < before, "before={before}, after={after}");
        let adapter = adapter.into_host().unwrap();
        assert!(adapter.b.iter().any(|value| *value != 0.0));

        let features = vec![0.25, -0.75];
        let base_logits = vec![0.1, -0.2, 0.3];
        let expected = adapter.adapted(&features, &base_logits).unwrap().1;
        let inference_device = Device::ndarray();
        let policy = DeviceLmHeadPolicy::from_host(&adapter, &inference_device);
        let actual = policy
            .adapted(
                Tensor::<2>::from_data(TensorData::new(features, [1, 2]), &inference_device),
                Tensor::<2>::from_data(TensorData::new(base_logits, [1, 3]), &inference_device),
            )
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-5)
        );
    }

    #[test]
    fn isolated_candidate_lora_supervises_only_the_dream_continuation() {
        let seed = seed(vec![1, 2], 1);
        let operations = BuiltinDreamOps::load(seed.config.clone(), seed.device.clone()).unwrap();
        let mut artifact = operations
            .read_candidate(
                &manual_candidate(&operations, &seed.txn, vec![1, 2, 3, 4]).artifact_hash,
            )
            .unwrap();
        artifact.wake_token_count = 3;

        let rows = operations
            .candidate_lm_head_rows(&seed.model, &artifact)
            .unwrap();
        assert_eq!(rows.rows, 1);
        assert_eq!(rows.targets, vec![4]);
        assert_eq!(rows.features.len(), tiny_config().hidden_size);
        assert_eq!(rows.base_logits.len(), tiny_config().vocab_size);
    }

    #[test]
    fn lora_geometry_overflow_fails_before_parameter_allocation() {
        let oversized_rows = MAX_DREAM_LM_HEAD_DATA_BYTES / std::mem::size_of::<f32>() / 5 + 1;
        let error = checked_lm_head_row_geometry(oversized_rows, 2, 3).unwrap_err();
        assert!(
            format!("{error:#}").contains("dataset exceeds"),
            "{error:#}"
        );

        let error = HostLmHeadLora::new(usize::MAX, 3, 2, 4, 17)
            .err()
            .expect("overflowing LoRA geometry must be rejected");
        assert!(format!("{error:#}").contains("shape overflow"), "{error:#}");

        let adapter = HostLmHeadLora::new(2, 3, 2, 4, 17).unwrap();
        let malformed_rows = LmHeadRows {
            rows: usize::MAX,
            hidden: 2,
            vocab: 3,
            features: Vec::new(),
            base_logits: Vec::new(),
            targets: Vec::new(),
        };
        let error = adapter.loss(&malformed_rows).unwrap_err();
        assert!(
            format!("{error:#}").contains("geometry overflows"),
            "{error:#}"
        );
    }

    #[test]
    fn dream_generation_uses_random_extra_routing_and_resumes_deterministically() {
        let seed = seed(vec![1, 2], 1);
        let mut operations =
            BuiltinDreamOps::load(seed.config.clone(), seed.device.clone()).unwrap();
        let (manifest, dreams) = operations
            .generate(
                &seed.txn,
                &seed.model,
                2,
                MemoryRouting::Dream { seed: seed.txn.id },
            )
            .unwrap();
        assert_eq!(dreams.len(), 2);
        assert!(dreams.iter().all(|dream| {
            dream.gradient.len() == 16
                && (dream
                    .gradient
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    - 1.0)
                    .abs()
                    < 1e-4
        }));
        let bytes = read_addressed(
            &operations.manifest_path(&manifest).unwrap(),
            &manifest,
            "test manifest",
        )
        .unwrap();
        let decoded: DreamManifest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.routing, DREAM_ROUTING_NAME);

        // A fresh process can load the same immutable receipt, and retrying
        // generation produces byte-identical identities.
        let mut resumed = BuiltinDreamOps::load(seed.config, seed.device).unwrap();
        assert_eq!(resumed.load(&seed.txn, &manifest).unwrap(), dreams);
        let retried = resumed
            .generate(
                &seed.txn,
                &seed.model,
                2,
                MemoryRouting::Dream { seed: seed.txn.id },
            )
            .unwrap();
        assert_eq!(retried, (manifest.clone(), dreams.clone()));
        let reference_hash = resumed.reference_set_hash().to_owned();
        let reference = resumed
            .reference_gradient(&seed.txn, &seed.model, &reference_hash)
            .unwrap();
        assert_eq!(reference.len(), 16);
        assert!((reference.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn dream_manifest_reauthenticates_candidate_provenance() {
        let seed = seed(vec![1, 2], 1);
        let mut operations =
            BuiltinDreamOps::load(seed.config.clone(), seed.device.clone()).unwrap();
        let mut dream = manual_candidate(&operations, &seed.txn, vec![1, 2]);
        let mut artifact = operations.read_candidate(&dream.artifact_hash).unwrap();
        artifact.generation_seed ^= 1;
        let bytes = canonical_json(&artifact).unwrap();
        dream.artifact_hash = sha256_identity(&bytes);
        publish_immutable(
            &operations.candidate_path(&dream.artifact_hash).unwrap(),
            &bytes,
        )
        .unwrap();
        dream.id = format!(
            "dream-{}-0-{}",
            seed.txn.id,
            short_hash(&dream.artifact_hash)
        );
        dream.diversity_key = mix64(artifact.generation_seed);
        let manifest = publish_manifest(&operations, &seed.txn, vec![dream]);

        let error = operations.load(&seed.txn, &manifest).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("generation seed is inconsistent"),
            "{error:#}"
        );
    }

    #[test]
    fn isolated_lm_head_lora_is_shape_bound_and_cannot_mutate_shared_model() {
        // Candidate and evaluator have the same input token but competing
        // targets. A real gradient step toward token 2 therefore worsens the
        // held-out token-3 objective and must be rejected by run_dreaming.
        let mut seed = seed(vec![1, 3], 8);
        let operations = BuiltinDreamOps::load(seed.config, seed.device.clone()).unwrap();
        let candidate = manual_candidate(&operations, &seed.txn, vec![1, 2]);
        seed.txn.dream_trial_rngs.push(DreamTrialRng {
            candidate_id: candidate.id.clone(),
            reservation: RngReservation {
                stream: 4,
                start: 13,
                count: 1,
            },
        });
        let probe = operations.probe(&seed.model).unwrap();
        let mut backend =
            TensorDreamBackend::new(seed.model, seed.device, probe, operations).unwrap();
        let before = backend.shared_checkpoint_hash().unwrap();
        seed.txn.dream_shared_checkpoint_hash = Some(before.clone());
        let trial = backend
            .isolated_lora_trial(&seed.txn, &candidate, 4, 8)
            .unwrap();
        assert_eq!(
            backend
                .operations()
                .evaluation_cache
                .as_ref()
                .map(|cache| cache.model_parameter_sha256.as_str()),
            Some(before.as_str())
        );
        assert!(
            trial.independent_task_improvement < 0.0,
            "competing-target trial unexpectedly improved by {}",
            trial.independent_task_improvement
        );
        assert_eq!(backend.shared_checkpoint_hash().unwrap(), before);
        let receipt = backend
            .operations()
            .read_adapter_receipt(&trial.adapter_hash)
            .unwrap();
        assert!(receipt.training_loss_after < receipt.training_loss_before);
        assert_eq!(
            receipt.base_checkpoint_sha256,
            seed.txn.candidate_hash.clone().unwrap()
        );
        assert_eq!(receipt.base_model_parameter_sha256, before);
        assert_eq!(receipt.target_module, DREAM_LORA_TARGET_MODULE);
        assert_eq!(receipt.a_shape, [4, tiny_config().hidden_size]);
        assert_eq!(receipt.b_shape, [4, tiny_config().vocab_size]);
    }

    #[test]
    fn accepted_lora_drives_a_content_addressed_idempotent_restem_receipt() {
        let mut seed = seed(vec![1, 2], 1);
        let operations = BuiltinDreamOps::load(seed.config.clone(), seed.device.clone()).unwrap();
        let candidate = manual_candidate(&operations, &seed.txn, vec![1, 2]);
        let candidate_artifact = operations.read_candidate(&candidate.artifact_hash).unwrap();
        let policy_training_rows = operations
            .dream_sequence_lm_head_rows(&seed.model, &candidate_artifact)
            .unwrap();
        let target = policy_training_rows.targets[0];
        let base_probability =
            softmax_host(&policy_training_rows.base_logits[..policy_training_rows.vocab]).unwrap()
                [target];
        seed.txn.generated_manifest = Some(publish_manifest(
            &operations,
            &seed.txn,
            vec![candidate.clone()],
        ));
        seed.txn.dream_trial_rngs.push(DreamTrialRng {
            candidate_id: candidate.id.clone(),
            reservation: RngReservation {
                stream: 4,
                start: 17,
                count: 1,
            },
        });
        let probe = operations.probe(&seed.model).unwrap();
        let mut backend =
            TensorDreamBackend::new(seed.model.clone(), seed.device.clone(), probe, operations)
                .unwrap();
        let shared_before = backend.shared_checkpoint_hash().unwrap();
        seed.txn.dream_shared_checkpoint_hash = Some(shared_before.clone());
        let trial = backend
            .isolated_lora_trial(&seed.txn, &candidate, 4, 8)
            .unwrap();
        assert!(trial.independent_task_improvement > 0.0);
        assert_eq!(backend.shared_checkpoint_hash().unwrap(), shared_before);
        let mut rebound = seed.txn.clone();
        rebound.dream_shared_checkpoint_hash = Some(hash('f'));
        assert!(
            backend
                .restem_update(&rebound, std::slice::from_ref(&trial), 3)
                .unwrap_err()
                .to_string()
                .contains("shared model parameters")
        );
        let first = backend
            .restem_update(&seed.txn, std::slice::from_ref(&trial), 3)
            .unwrap();
        let second = backend
            .restem_update(&seed.txn, std::slice::from_ref(&trial), 3)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(backend.shared_checkpoint_hash().unwrap(), shared_before);

        let policy_artifact = seed
            .config
            .committed_policy_artifact(seed.txn.id, &first)
            .unwrap();
        let policy: RestemPolicyState = policy_artifact.verify_json().unwrap();
        assert_eq!(policy.rank, seed.config.generation_policy_rank);
        assert_eq!(policy.alpha, seed.config.generation_policy_alpha);
        let (policy_receipt, updated_adapter) = backend
            .operations()
            .read_generation_policy_adapter(&policy.adapter_sha256)
            .unwrap();
        assert_eq!(policy_receipt.transaction_id, seed.txn.id);
        assert_eq!(policy_receipt.source_checkpoint_sha256, hash('d'));
        assert_eq!(policy_receipt.source_model_parameter_sha256, shared_before);
        let updated_logits = updated_adapter
            .adapted(
                &policy_training_rows.features[..policy_training_rows.hidden],
                &policy_training_rows.base_logits[..policy_training_rows.vocab],
            )
            .unwrap()
            .1;
        let updated_probability = softmax_host(&updated_logits).unwrap()[target];
        assert!(
            updated_probability > base_probability,
            "positive ReSTEM trial did not increase next-token likelihood: {base_probability} -> {updated_probability}"
        );

        let mut resumed = BuiltinDreamOps::load(seed.config.clone(), seed.device.clone()).unwrap();
        let resumed_receipt = resumed
            .restem_update_policy(&seed.txn, &seed.model, &[trial], 3)
            .unwrap();
        assert_eq!(resumed_receipt, first);
        assert!(resumed.policy_path(&first).unwrap().is_file());

        let mut next_config = seed.config.clone();
        next_config.initial_policy = Some(policy_artifact);
        let mut next_cycle = BuiltinDreamOps::load(next_config, seed.device.clone()).unwrap();
        let loaded_logits = next_cycle
            .generation_policy
            .as_ref()
            .unwrap()
            .adapted(
                &policy_training_rows.features[..policy_training_rows.hidden],
                &policy_training_rows.base_logits[..policy_training_rows.vocab],
            )
            .unwrap()
            .1;
        assert_eq!(loaded_logits, updated_logits);
        let mut next_txn = seed.txn.clone();
        next_txn.id += 1;
        next_txn.trigger_clock += 1;
        next_txn.generated_manifest = None;
        next_txn.dream_shared_checkpoint_hash = None;
        next_txn.dream_policy_receipt = None;
        next_txn.dream_generation_rng.as_mut().unwrap().start += 100;
        let (_, next_dreams) = next_cycle
            .generate(
                &next_txn,
                &seed.model,
                1,
                MemoryRouting::Dream { seed: next_txn.id },
            )
            .unwrap();
        let next_candidate = next_cycle
            .read_candidate(&next_dreams[0].artifact_hash)
            .unwrap();
        assert_eq!(
            next_candidate.generation_policy_sha256.as_deref(),
            Some(first.as_str())
        );
        assert_eq!(
            next_candidate.generation_policy_adapter_sha256.as_deref(),
            Some(policy.adapter_sha256.as_str())
        );
    }

    #[test]
    fn all_rejected_trials_publish_an_idempotent_noop_policy() {
        let mut seed = seed(vec![1, 3], 1);
        let mut operations =
            BuiltinDreamOps::load(seed.config.clone(), seed.device.clone()).unwrap();
        let candidate = manual_candidate(&operations, &seed.txn, vec![1, 2]);
        seed.txn.generated_manifest =
            Some(publish_manifest(&operations, &seed.txn, vec![candidate]));
        seed.txn.dream_shared_checkpoint_hash = Some(model_parameter_hash(&seed.model).unwrap());

        let first = operations
            .restem_update_policy(&seed.txn, &seed.model, &[], 3)
            .unwrap();
        let second = operations
            .restem_update_policy(&seed.txn, &seed.model, &[], 3)
            .unwrap();
        assert_eq!(first, second);

        let artifact = seed
            .config
            .committed_policy_artifact(seed.txn.id, &first)
            .unwrap();
        let policy: RestemPolicyState = artifact.verify_json().unwrap();
        assert_eq!(policy.transaction_id, seed.txn.id);
        assert_eq!(policy.iterations, 0);
        assert_eq!(policy.rank, seed.config.generation_policy_rank);
        assert_eq!(policy.alpha, seed.config.generation_policy_alpha);
        assert!(policy.accepted_adapters.is_empty());
        assert!(
            operations
                .generation_policy_adapter_path(&policy.adapter_sha256)
                .unwrap()
                .is_file()
        );
        let adapter_bytes = read_addressed(
            &operations
                .generation_policy_adapter_path(&policy.adapter_sha256)
                .unwrap(),
            &policy.adapter_sha256,
            "test generation policy",
        )
        .unwrap();

        let mut next_config = seed.config.clone();
        next_config.initial_policy = Some(artifact.clone());
        let mut next_operations =
            BuiltinDreamOps::load(next_config.clone(), seed.device.clone()).unwrap();
        let mut next_txn = seed.txn.clone();
        next_txn.id += 1;
        next_txn.trigger_clock += 1;
        next_txn.dream_generation_rng.as_mut().unwrap().start += 100;
        next_txn.dream_shared_checkpoint_hash = Some(model_parameter_hash(&seed.model).unwrap());
        let next_candidate = manual_candidate(&next_operations, &next_txn, vec![1, 2]);
        next_txn.generated_manifest = Some(publish_manifest(
            &next_operations,
            &next_txn,
            vec![next_candidate],
        ));
        let next_policy_hash = next_operations
            .restem_update_policy(&next_txn, &seed.model, &[], 3)
            .unwrap();
        let next_policy_artifact = next_config
            .committed_policy_artifact(next_txn.id, &next_policy_hash)
            .unwrap();
        let next_policy: RestemPolicyState = next_policy_artifact.verify_json().unwrap();
        assert_eq!(
            next_policy.parent_policy_sha256.as_deref(),
            Some(first.as_str())
        );
        assert_eq!(next_policy.adapter_sha256, policy.adapter_sha256);
        assert_eq!(next_policy.iterations, 0);
        assert_eq!(
            read_addressed(
                &next_operations
                    .generation_policy_adapter_path(&next_policy.adapter_sha256)
                    .unwrap(),
                &next_policy.adapter_sha256,
                "carried generation policy",
            )
            .unwrap(),
            adapter_bytes
        );
        next_config.initial_policy = Some(next_policy_artifact);
        BuiltinDreamOps::load(next_config, seed.device.clone()).unwrap();

        let mut resumed = BuiltinDreamOps::load(seed.config, seed.device).unwrap();
        assert_eq!(
            resumed
                .restem_update_policy(&seed.txn, &seed.model, &[], 3)
                .unwrap(),
            first
        );
    }

    #[test]
    fn generation_policy_rejects_invalid_geometry_temperature_and_tampered_weights() {
        let mut seed = seed(vec![1, 3], 1);
        let mut invalid = seed.config.clone();
        invalid.generation_temperature = 0.0;
        assert!(
            invalid
                .validate()
                .unwrap_err()
                .to_string()
                .contains("temperature")
        );
        invalid = seed.config.clone();
        invalid.generation_policy_rank = 0;
        assert!(
            invalid
                .validate()
                .unwrap_err()
                .to_string()
                .contains("geometry")
        );
        for (invalid, expected) in [
            (
                BuiltinDreamingRuntimeConfig {
                    max_new_tokens: BuiltinDreamingRuntimeConfig::MAX_NEW_TOKENS + 1,
                    ..seed.config.clone()
                },
                "max_new_tokens",
            ),
            (
                BuiltinDreamingRuntimeConfig {
                    gradient_dimensions: BuiltinDreamingRuntimeConfig::MAX_GRADIENT_DIMENSIONS + 1,
                    ..seed.config.clone()
                },
                "gradient dimensions",
            ),
            (
                BuiltinDreamingRuntimeConfig {
                    lora_steps: BuiltinDreamingRuntimeConfig::MAX_LORA_STEPS + 1,
                    ..seed.config.clone()
                },
                "LoRA steps",
            ),
            (
                BuiltinDreamingRuntimeConfig {
                    generation_policy_rank: DreamingConfig::MAX_LORA_RANK + 1,
                    ..seed.config.clone()
                },
                "LoRA rank",
            ),
            (
                BuiltinDreamingRuntimeConfig {
                    generation_policy_alpha: DreamingConfig::MAX_LORA_ALPHA + 1,
                    ..seed.config.clone()
                },
                "LoRA alpha",
            ),
        ] {
            let error = invalid.validate().unwrap_err();
            assert!(format!("{error:#}").contains(expected), "{error:#}");
        }

        let mut operations =
            BuiltinDreamOps::load(seed.config.clone(), seed.device.clone()).unwrap();
        let candidate = manual_candidate(&operations, &seed.txn, vec![1, 2]);
        seed.txn.generated_manifest =
            Some(publish_manifest(&operations, &seed.txn, vec![candidate]));
        seed.txn.dream_shared_checkpoint_hash = Some(model_parameter_hash(&seed.model).unwrap());
        let policy_hash = operations
            .restem_update_policy(&seed.txn, &seed.model, &[], 1)
            .unwrap();
        let policy_artifact = seed
            .config
            .committed_policy_artifact(seed.txn.id, &policy_hash)
            .unwrap();
        let policy: RestemPolicyState = policy_artifact.verify_json().unwrap();

        let mut wrong_geometry = seed.config.clone();
        wrong_geometry.initial_policy = Some(policy_artifact.clone());
        wrong_geometry.generation_policy_rank += 1;
        let error = BuiltinDreamOps::load(wrong_geometry, seed.device.clone())
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("geometry"), "{error}");

        let adapter_path = operations
            .generation_policy_adapter_path(&policy.adapter_sha256)
            .unwrap();
        let mut bytes = fs::read(&adapter_path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&adapter_path, bytes).unwrap();
        let mut tampered = seed.config;
        tampered.initial_policy = Some(policy_artifact);
        let error = BuiltinDreamOps::load(tampered, seed.device)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("content hash changed"), "{error}");
    }
}
