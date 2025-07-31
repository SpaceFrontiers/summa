//! Transformer-backed in-model consolidation and isolated dreaming.
//!
//! [`crate::sleep`] owns the durable state machine. This module performs its
//! tensor work in-process: content-pinned teacher/student staging, chunked
//! forward KL, receiver-slot-only AdamW updates, GRPO-style imitation,
//! retention gates, atomic publication, exact rollback, and isolated dream
//! trials. Rollout generation and semantic evaluation stay injectable because
//! they are deployment artifacts, but the losses and parameter filtering are
//! implemented here rather than delegated to callbacks.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use std::fs::OpenOptions;
#[cfg(test)]
use std::io::Write;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Bool, Bytes, Device, Int, Tensor, TensorData};
use burn_optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use hermes_llm::{MemoryRouting, ModelDef, Transformer, load_safetensors_bytes, save_safetensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::artifact_io::{
    AuthenticatedDirectorySnapshot, ensure_directory, ensure_real_directory, ensure_regular_file,
    hash_regular_file as hash_file, read_regular_bounded, sha256_identity, sync_directory,
    sync_regular_file, validate_sha256_identity, write_new_synced,
};
use crate::optimizer_artifact::{
    canonical_module_optimizer_bytes, save_canonical_module_optimizer,
};
use crate::sleep::{
    CommittedCandidate, ConsolidationBackend, ConsolidationTxn, DreamTrial, DreamingBackend,
    GeneratedDream, ImitationConfig, KnowledgeSeedingConfig, RngReservation, SleepProgressSink,
    SleepState, run_consolidation_with_progress,
};

const KNOWLEDGE_DEVICE_RNG_DOMAIN: u64 = 0x6b6e_6f77_6c65_6467;
const IMITATION_DEVICE_RNG_DOMAIN: u64 = 0x696d_6974_6174_696f;

// These limits apply at the algorithm-neutral tensor boundary as well as in
// the first-party rollout adapters. An injected adapter is still untrusted
// operational input: without aggregate ceilings it could turn one sleep
// transaction into billions of model tokens or edit-distance cells.
const MAX_TENSOR_ROLLOUT_BATCH_ROWS: usize = 4_096;
const MAX_TENSOR_FORWARD_TOKENS: usize = 65_536;
const MAX_TENSOR_SUBPHASE_TOKENS: usize = 16 * 1024 * 1024;
const MAX_TENSOR_ROLLOUT_BATCHES: usize = 4_096;
pub(crate) const MAX_TENSOR_IMITATION_GROUPS: usize = 256;
const MAX_TENSOR_EDIT_DISTANCE_CELLS: usize = 64 * 1024 * 1024;
const MAX_TENSOR_GRPO_PADDED_TOKENS: usize = 1024 * 1024;
const FINGERPRINT_CHUNK_VALUES: usize = 4_096;

fn tensor_subphase_seed(txn: &ConsolidationTxn, reservation: RngReservation, domain: u64) -> u64 {
    let mut value = domain
        ^ txn.id.rotate_left(7)
        ^ txn.trigger_clock.rotate_left(19)
        ^ (reservation.stream as u64).rotate_left(31)
        ^ reservation.start.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ reservation.count.rotate_left(43);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// A strict content-addressed Transformer checkpoint loaded before sleep.
#[derive(Clone)]
pub struct ImmutableTransformerCheckpoint {
    pub uri: String,
    pub sha256: String,
    pub model: Transformer,
}

impl ImmutableTransformerCheckpoint {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.uri.trim().is_empty(), "checkpoint URI is empty");
        validate_sha256_identity(&self.sha256, "checkpoint identity")
    }
}

/// Sender update staged by the wake optimizer but not applied to the live
/// model. Its checkpoint and update identities are checked against the txn.
#[derive(Clone)]
pub struct ProspectiveTransformerCandidate {
    pub checkpoint: ImmutableTransformerCheckpoint,
    pub update_sha256: String,
}

/// Opaque, byte-exact state of the wake-side prospective-update adapter. It
/// includes sender optimizer moments, pending gradient accumulators, loss
/// scaler state, and any update clock needed to reproduce or roll back a
/// transaction. The native tensor store authenticates these bytes in the same
/// immutable manifest as teacher/student weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProspectiveUpdateSnapshot {
    bytes: Vec<u8>,
}

impl ProspectiveUpdateSnapshot {
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        ensure!(
            !bytes.is_empty(),
            "prospective-update snapshot must not be empty"
        );
        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub trait ProspectiveTransformerUpdate {
    /// Snapshot all state which can be observed or mutated by [`Self::stage`]
    /// or [`Self::clear_reclaimed_optimizer_state`]. Identical state must
    /// produce identical bytes.
    fn snapshot_state(&mut self, txn: &ConsolidationTxn) -> Result<ProspectiveUpdateSnapshot>;

    /// Restore an exact snapshot. This operation must be transaction-
    /// idempotent and failure-atomic.
    fn restore_state(
        &mut self,
        txn: &ConsolidationTxn,
        snapshot: &ProspectiveUpdateSnapshot,
    ) -> Result<()>;

    /// Must be idempotent for `txn.id`. If this method or validation of its
    /// output fails, the backend restores the pre-stage snapshot before
    /// returning.
    fn stage(
        &mut self,
        txn: &ConsolidationTxn,
        teacher: &Transformer,
    ) -> Result<ProspectiveTransformerCandidate>;

    /// Reset reclaimed sender optimizer moments after candidate publication.
    /// This operation must be idempotent by transaction ID and failure-atomic:
    /// returning an error must leave the optimizer exactly unchanged.
    fn clear_reclaimed_optimizer_state(
        &mut self,
        txn: &ConsolidationTxn,
        parameter_ids: &[ParamId],
    ) -> Result<()>;
}

struct ParameterFingerprintVisitor<'a> {
    included: &'a BTreeSet<u64>,
    values: BTreeMap<u64, [u8; 32]>,
    failure: Option<String>,
}

impl ParameterFingerprintVisitor<'_> {
    fn hasher(&self, kind: u8, shape: &[usize]) -> Sha256 {
        let mut hash = Sha256::new();
        hash.update(b"hermes-parameter-fingerprint-v1\0");
        hash.update([kind]);
        hash.update((shape.len() as u64).to_le_bytes());
        for dimension in shape {
            hash.update((*dimension as u64).to_le_bytes());
        }
        hash
    }

    fn record(&mut self, id: ParamId, hash: Sha256) {
        if self.failure.is_some() {
            return;
        }
        let digest: [u8; 32] = hash.finalize().into();
        if self.values.insert(id.val(), digest).is_some() {
            self.failure = Some(format!("model repeats parameter ID {}", id.val()));
        }
    }
}

impl ModuleVisitor for ParameterFingerprintVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if !self.included.contains(&parameter.id.val()) {
            return;
        }
        match parameter.val().into_data().convert::<f32>().to_vec::<f32>() {
            Ok(values) => {
                let mut hash = self.hasher(b'f', &parameter.shape().dims::<D>());
                let mut bytes =
                    Vec::with_capacity(FINGERPRINT_CHUNK_VALUES * std::mem::size_of::<f32>());
                for chunk in values.chunks(FINGERPRINT_CHUNK_VALUES) {
                    bytes.clear();
                    bytes.extend(chunk.iter().flat_map(|value| value.to_le_bytes()));
                    hash.update(&bytes);
                }
                self.record(parameter.id, hash);
            }
            Err(error) => self.failure = Some(error.to_string()),
        }
    }

    fn visit_int<const D: usize>(&mut self, parameter: &Param<Tensor<D, Int>>) {
        if !self.included.contains(&parameter.id.val()) {
            return;
        }
        match parameter.val().into_data().to_vec::<i64>() {
            Ok(values) => {
                let mut hash = self.hasher(b'i', &parameter.shape().dims::<D>());
                let mut bytes =
                    Vec::with_capacity(FINGERPRINT_CHUNK_VALUES * std::mem::size_of::<i64>());
                for chunk in values.chunks(FINGERPRINT_CHUNK_VALUES) {
                    bytes.clear();
                    bytes.extend(chunk.iter().flat_map(|value| value.to_le_bytes()));
                    hash.update(&bytes);
                }
                self.record(parameter.id, hash);
            }
            Err(error) => self.failure = Some(error.to_string()),
        }
    }

    fn visit_bool<const D: usize>(&mut self, parameter: &Param<Tensor<D, Bool>>) {
        if !self.included.contains(&parameter.id.val()) {
            return;
        }
        match parameter.val().into_data().to_vec::<bool>() {
            Ok(values) => {
                let mut hash = self.hasher(b'b', &parameter.shape().dims::<D>());
                let mut bytes = Vec::with_capacity(FINGERPRINT_CHUNK_VALUES);
                for chunk in values.chunks(FINGERPRINT_CHUNK_VALUES) {
                    bytes.clear();
                    bytes.extend(chunk.iter().copied().map(u8::from));
                    hash.update(&bytes);
                }
                self.record(parameter.id, hash);
            }
            Err(error) => self.failure = Some(error.to_string()),
        }
    }
}

fn parameter_fingerprints(
    model: &Transformer,
    included: &BTreeSet<u64>,
) -> Result<BTreeMap<u64, [u8; 32]>> {
    let mut visitor = ParameterFingerprintVisitor {
        included,
        values: BTreeMap::new(),
        failure: None,
    };
    model.visit(&mut visitor);
    if let Some(error) = visitor.failure {
        bail!("fingerprinting prospective update parameters: {error}");
    }
    ensure!(
        visitor.values.keys().copied().eq(included.iter().copied()),
        "prospective update fingerprint scope contains a parameter absent from the model"
    );
    Ok(visitor.values)
}

fn prospective_update_scope(
    teacher: &Transformer,
    student: &Transformer,
    sender: usize,
) -> Result<(BTreeSet<u64>, BTreeSet<u64>)> {
    ensure!(
        serde_json::to_vec(teacher.config())? == serde_json::to_vec(student.config())?,
        "prospective student changed model topology"
    );
    let teacher_statuses = teacher.memory_slot_statuses();
    let student_statuses = student.memory_slot_statuses();
    ensure!(
        teacher_statuses.len() == student_statuses.len()
            && teacher_statuses
                .iter()
                .zip(&student_statuses)
                .all(|(left, right)| {
                    left.layer == right.layer
                        && left.tier == right.tier
                        && left.tier_name == right.tier_name
                        && left.slot == right.slot
                        && left.active == right.active
                        && left.generation == right.generation
                        && left.parameter_ids == right.parameter_ids
                }),
        "prospective student changed memory masks, generations, or topology"
    );
    let teacher_parameter_ids = burn::module::list_param_ids(teacher);
    let teacher_parameters = teacher_parameter_ids
        .iter()
        .map(|id| id.val())
        .collect::<BTreeSet<_>>();
    let student_parameter_ids = burn::module::list_param_ids(student);
    let student_parameters = student_parameter_ids
        .iter()
        .map(|id| id.val())
        .collect::<BTreeSet<_>>();
    ensure!(
        !teacher_parameters.is_empty()
            && teacher_parameters.len() == teacher_parameter_ids.len()
            && student_parameters.len() == student_parameter_ids.len()
            && teacher_parameters == student_parameters,
        "prospective student parameter IDs differ from teacher"
    );
    let mut eligible = teacher
        .memory_tier_base_parameter_ids_all_layers(sender)?
        .into_iter()
        .map(|id| id.val())
        .collect::<BTreeSet<_>>();
    for status in teacher_statuses
        .iter()
        .filter(|status| status.tier == sender && status.active)
    {
        eligible.extend(status.parameter_ids.iter().map(|id| id.val()));
    }
    ensure!(
        !eligible.is_empty(),
        "sender update has no eligible parameters"
    );
    ensure!(
        eligible.is_subset(&teacher_parameters),
        "sender update scope contains a parameter absent from the model"
    );
    Ok((eligible, teacher_parameters))
}

fn prospective_update_hash_inner(
    teacher: &Transformer,
    student: &Transformer,
    sender: usize,
    validate_outside_scope: bool,
) -> Result<String> {
    let (eligible, all_parameters) = prospective_update_scope(teacher, student, sender)?;
    if validate_outside_scope {
        let outside = all_parameters
            .difference(&eligible)
            .copied()
            .collect::<BTreeSet<_>>();
        let teacher_outside = parameter_fingerprints(teacher, &outside)?;
        let student_outside = parameter_fingerprints(student, &outside)?;
        let escaped = teacher_outside
            .iter()
            .filter_map(|(id, before)| (student_outside.get(id) != Some(before)).then_some(*id))
            .collect::<Vec<_>>();
        ensure!(
            escaped.is_empty(),
            "prospective update escaped sender tier {sender}: parameter IDs {escaped:?}"
        );
    }

    // The hot-path identity contains only tensors which the independently
    // scoped sender optimizer is allowed to mutate. In particular, this avoids
    // copying embeddings, mixers, heads, and every other memory tier back to
    // the host merely to derive one due-tier receipt.
    let teacher_eligible = parameter_fingerprints(teacher, &eligible)?;
    let student_eligible = parameter_fingerprints(student, &eligible)?;
    ensure!(
        teacher_eligible
            .iter()
            .any(|(id, before)| student_eligible.get(id) != Some(before)),
        "prospective sender update is empty"
    );
    let mut hash = Sha256::new();
    hash.update(b"hermes-prospective-update-v2\0");
    hash.update((sender as u64).to_le_bytes());
    hash.update((eligible.len() as u64).to_le_bytes());
    for (id, before) in teacher_eligible {
        let after = student_eligible
            .get(&id)
            .expect("eligible parameter sets checked");
        hash.update(id.to_le_bytes());
        hash.update(before);
        hash.update(after);
    }
    Ok(format!("sha256:{:x}", hash.finalize()))
}

/// Bind the exact sender tier's base plus currently active reserve slots.
/// The wake hot path constructs its student with an independently scoped
/// optimizer, so hashing unrelated model tensors here would add a full-model
/// device-to-host synchronization to every due update.
pub fn prospective_update_hash(
    teacher: &Transformer,
    student: &Transformer,
    sender: usize,
) -> Result<String> {
    prospective_update_hash_inner(teacher, student, sender, false)
}

/// Validate full-model parity outside the sender scope and return the same
/// sender-only identity. This expensive check belongs at a consolidation
/// transaction boundary, where updates may come from an injected adapter, not
/// on ordinary wake-only optimizer updates.
fn prospective_update_hash_at_boundary(
    teacher: &Transformer,
    student: &Transformer,
    sender: usize,
) -> Result<String> {
    prospective_update_hash_inner(teacher, student, sender, true)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenRolloutBatch {
    pub batch: usize,
    pub sequence: usize,
    pub token_ids: Vec<i64>,
}

impl TokenRolloutBatch {
    pub fn new(batch: usize, sequence: usize, token_ids: Vec<i64>) -> Result<Self> {
        ensure!(
            batch > 0 && sequence > 0,
            "rollout dimensions must be positive"
        );
        ensure!(
            batch.checked_mul(sequence) == Some(token_ids.len()),
            "rollout shape [{batch}, {sequence}] disagrees with {} tokens",
            token_ids.len()
        );
        ensure!(
            batch <= MAX_TENSOR_ROLLOUT_BATCH_ROWS,
            "rollout batch exceeds the {MAX_TENSOR_ROLLOUT_BATCH_ROWS}-row operational limit"
        );
        ensure!(
            token_ids.len() <= MAX_TENSOR_FORWARD_TOKENS,
            "rollout batch exceeds the {MAX_TENSOR_FORWARD_TOKENS}-token operational limit"
        );
        Ok(Self {
            batch,
            sequence,
            token_ids,
        })
    }

    fn validate_for(&self, model: &Transformer) -> Result<()> {
        ensure!(
            self.batch > 0
                && self.sequence > 0
                && self.batch.checked_mul(self.sequence) == Some(self.token_ids.len()),
            "rollout dimensions no longer match its token storage"
        );
        ensure!(
            self.batch <= MAX_TENSOR_ROLLOUT_BATCH_ROWS,
            "rollout batch exceeds the {MAX_TENSOR_ROLLOUT_BATCH_ROWS}-row operational limit"
        );
        ensure!(
            self.token_ids.len() <= MAX_TENSOR_FORWARD_TOKENS,
            "rollout batch exceeds the {MAX_TENSOR_FORWARD_TOKENS}-token operational limit"
        );
        ensure!(
            self.sequence <= model.config().max_seq_len,
            "rollout exceeds model sequence limit"
        );
        ensure!(
            self.token_ids
                .iter()
                .all(|id| usize::try_from(*id).is_ok_and(|id| id < model.config().vocab_size)),
            "rollout contains an out-of-vocabulary token"
        );
        Ok(())
    }

    fn tensor(&self, device: &Device) -> Tensor<2, Int> {
        Tensor::from_data(
            TensorData::new(self.token_ids.clone(), [self.batch, self.sequence]),
            device,
        )
    }
}

fn validate_rollout_batches(
    batches: &[TokenRolloutBatch],
    model: &Transformer,
    label: &str,
) -> Result<u64> {
    ensure!(
        batches.len() <= MAX_TENSOR_ROLLOUT_BATCHES,
        "{label} returned more than {MAX_TENSOR_ROLLOUT_BATCHES} rollout batches"
    );
    let mut total_tokens = 0_usize;
    for batch in batches {
        batch.validate_for(model)?;
        total_tokens = total_tokens
            .checked_add(batch.token_ids.len())
            .with_context(|| format!("{label} token count overflow"))?;
        ensure!(
            total_tokens <= MAX_TENSOR_SUBPHASE_TOKENS,
            "{label} exceeds the {MAX_TENSOR_SUBPHASE_TOKENS}-token subphase limit"
        );
    }
    total_tokens
        .try_into()
        .with_context(|| format!("{label} token count exceeds u64"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RolloutOwner {
    Teacher,
    DetachedStudent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImitationGroup {
    pub prefix: Vec<i64>,
    pub teacher_continuation: Vec<i64>,
    pub candidates: Vec<Vec<i64>>,
}

/// Generates model-owned token rollouts. Host token IDs are detached by
/// construction and this interface deliberately has no corpus/search input.
/// Implementations must derive randomness solely from the reservation stored
/// in `txn`, and return identical artifacts for the same transaction/subphase
/// after interruption.
pub trait ConsolidationRollouts {
    fn knowledge_rollouts(
        &mut self,
        txn: &ConsolidationTxn,
        owner: RolloutOwner,
        model: &Transformer,
        count: usize,
    ) -> Result<Vec<TokenRolloutBatch>>;

    fn imitation_groups(
        &mut self,
        txn: &ConsolidationTxn,
        teacher: &Transformer,
        student: &Transformer,
        group_size: usize,
    ) -> Result<Vec<ImitationGroup>>;
}

/// Frozen semantic evaluator. `artifact_hash` pins both implementation and
/// weights; scoring must be deterministic for identical token sequences.
pub trait SemanticJudge {
    fn artifact_hash(&self) -> &str;
    fn score(&mut self, prefix: &[i64], teacher: &[i64], candidate: &[i64]) -> Result<f32>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetentionScores {
    pub stable_anchor: f32,
    pub incorporation: f32,
}

/// Frozen evaluator bound to the exact immutable retention-suite bytes. Calls
/// may be retried and must return identical rollouts/scores for one transaction.
pub trait RetentionEvaluator {
    fn artifact_hash(&self) -> &str;
    fn suite_hash(&self) -> &str;
    fn anchor_rollouts(&mut self, txn: &ConsolidationTxn) -> Result<Vec<TokenRolloutBatch>>;
    fn score(&mut self, txn: &ConsolidationTxn, model: &Transformer) -> Result<RetentionScores>;
}

/// Durable publication is called before the backend swaps its live pointer.
/// Both methods must be idempotent by transaction ID and failure-atomic. A
/// failed call must not expose a partial mutable candidate.
pub trait AtomicCandidatePublisher {
    fn publish_candidate(
        &mut self,
        txn: &ConsolidationTxn,
        candidate: &Transformer,
    ) -> Result<ImmutableTransformerCheckpoint>;
    fn restore_teacher(
        &mut self,
        txn: &ConsolidationTxn,
        teacher: &ImmutableTransformerCheckpoint,
    ) -> Result<()>;
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionGateConfig {
    pub evaluator_hash: String,
    pub suite_hash: String,
    pub max_anchor_forward_kl: f32,
    pub max_anchor_regression: f32,
    pub min_incorporation_gain: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorConsolidationConfig {
    pub knowledge: KnowledgeSeedingConfig,
    pub imitation: ImitationConfig,
    pub retention: RetentionGateConfig,
    pub receiver_learning_rate: f64,
    pub receiver_weight_decay: f32,
    pub grpo_clip_epsilon: f64,
    pub grpo_advantage_epsilon: f64,
    pub grpo_kl_coefficient: f64,
}

impl TensorConsolidationConfig {
    pub fn validate(&self) -> Result<()> {
        self.knowledge.validate()?;
        self.imitation.validate()?;
        validate_sha256_identity(
            &self.imitation.semantic_judge_hash,
            "semantic judge identity",
        )?;
        validate_sha256_identity(
            &self.retention.evaluator_hash,
            "retention evaluator identity",
        )?;
        validate_sha256_identity(&self.retention.suite_hash, "retention suite identity")?;
        ensure!(
            self.retention.max_anchor_forward_kl.is_finite()
                && self.retention.max_anchor_forward_kl >= 0.0,
            "retention KL gate must be finite and non-negative"
        );
        ensure!(
            self.retention.max_anchor_regression.is_finite()
                && self.retention.max_anchor_regression >= 0.0,
            "retention regression gate must be finite and non-negative"
        );
        ensure!(
            self.retention.min_incorporation_gain.is_finite(),
            "incorporation gate is non-finite"
        );
        ensure!(
            self.receiver_learning_rate.is_finite() && self.receiver_learning_rate > 0.0,
            "receiver learning rate must be finite and positive"
        );
        ensure!(
            self.receiver_weight_decay.is_finite() && self.receiver_weight_decay >= 0.0,
            "receiver weight decay must be finite and non-negative"
        );
        ensure!(
            self.grpo_clip_epsilon.is_finite()
                && self.grpo_clip_epsilon > 0.0
                && self.grpo_clip_epsilon < 1.0,
            "GRPO clip epsilon must be in (0, 1)"
        );
        ensure!(
            self.grpo_advantage_epsilon.is_finite() && self.grpo_advantage_epsilon > 0.0,
            "GRPO advantage epsilon must be positive"
        );
        ensure!(
            self.grpo_kl_coefficient.is_finite() && self.grpo_kl_coefficient >= 0.0,
            "GRPO KL coefficient must be non-negative"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSleepDiagnostics {
    pub knowledge_updates: usize,
    pub imitation_updates: usize,
    pub selected_gradient_tensors: usize,
    pub knowledge_tokens: u64,
    pub knowledge_kl_sum: f64,
    pub teacher_entropy_sum: f64,
    pub student_entropy_sum: f64,
    pub imitation_samples: u64,
    pub imitation_semantic_sum: f64,
    pub imitation_edit_sum: f64,
    pub imitation_threshold_sum: f64,
    pub imitation_reward_sum: f64,
    pub imitation_reward_square_sum: f64,
    pub imitation_grpo_kl_sum: f64,
    pub retention_anchor_kl: f32,
    pub teacher_stable_anchor: f32,
    pub student_stable_anchor: f32,
    pub teacher_incorporation: f32,
    pub student_incorporation: f32,
    pub anchor_delta: f32,
    pub incorporation_gain: f32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorStage {
    Prospective,
    Student,
    Knowledge,
    Imitation,
    Retention,
    Committed,
    RolledBack,
}

/// Exact in-flight tensors and receiver optimizer restored from checkpoint v2.
#[derive(Clone)]
pub struct RecoveredTensorTransaction {
    pub txn_id: u64,
    pub teacher: ImmutableTransformerCheckpoint,
    pub student: ProspectiveTransformerCandidate,
    pub pre_update_state: ProspectiveUpdateSnapshot,
    pub staged_update_state: ProspectiveUpdateSnapshot,
    pub receiver_optimizer: ModuleOptimizer,
    pub completed: BTreeSet<TensorStage>,
    pub retention_result: Option<bool>,
    pub diagnostics: TensorSleepDiagnostics,
}

const TENSOR_TXN_STORE_VERSION: u32 = 2;
const TENSOR_TXN_MANIFEST_VERSION: u32 = 1;
const TENSOR_TXN_POINTER: &str = "current.json";
const TENSOR_TXN_MANIFEST: &str = "manifest.json";
const TENSOR_TXN_METADATA: &str = "transaction.json";
const TENSOR_TXN_TEACHER: &str = "teacher.safetensors";
const TENSOR_TXN_STUDENT: &str = "student.safetensors";
const TENSOR_TXN_OPTIMIZER: &str = "receiver-optimizer.bpk";
const TENSOR_TXN_UPDATE_PRE: &str = "sender-update-pre.bin";
const TENSOR_TXN_UPDATE_STAGED: &str = "sender-update-staged.bin";
const MAX_TENSOR_TXN_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TENSOR_TXN_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSOR_TXN_MEMBER_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const TENSOR_TXN_SCHEMA: [&str; 7] = [
    TENSOR_TXN_MANIFEST,
    TENSOR_TXN_METADATA,
    TENSOR_TXN_OPTIMIZER,
    TENSOR_TXN_STUDENT,
    TENSOR_TXN_TEACHER,
    TENSOR_TXN_UPDATE_PRE,
    TENSOR_TXN_UPDATE_STAGED,
];
static TENSOR_TXN_STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorLoadStage {
    AfterCapture,
    AfterStagedLoad,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorTransactionPointer {
    pub version: u32,
    pub txn_id: u64,
    pub generation: String,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TensorTransactionFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TensorTransactionManifest {
    version: u32,
    txn_id: u64,
    files: Vec<TensorTransactionFile>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TensorTransactionMetadata {
    version: u32,
    txn_id: u64,
    teacher_uri: String,
    teacher_sha256: String,
    student_uri: String,
    student_sha256: String,
    prospective_update_sha256: String,
    teacher_parameter_ids: Vec<u64>,
    student_parameter_ids: Vec<u64>,
    completed: BTreeSet<TensorStage>,
    retention_result: Option<bool>,
    diagnostics: TensorSleepDiagnostics,
}

struct TensorStagingGuard(PathBuf);

impl Drop for TensorStagingGuard {
    fn drop(&mut self) {
        if self.0.exists() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

/// Crash-safe, content-addressed store for every in-flight tensor sleep
/// boundary. It seals teacher/student weights, receiver optimizer moments,
/// parameter IDs, and subphase metadata into an immutable generation, then
/// atomically publishes `current.json` last.
#[derive(Clone, Debug)]
pub struct TensorTransactionStore {
    root: PathBuf,
}

impl TensorTransactionStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn publish(
        &self,
        txn: &ConsolidationTxn,
        recovered: &RecoveredTensorTransaction,
    ) -> Result<TensorTransactionPointer> {
        ensure!(
            recovered.txn_id == txn.id,
            "tensor snapshot transaction ID differs"
        );
        recovered.teacher.validate()?;
        recovered.student.checkpoint.validate()?;
        ensure!(
            !recovered.pre_update_state.as_bytes().is_empty()
                && !recovered.staged_update_state.as_bytes().is_empty(),
            "tensor snapshot is missing prospective-update state"
        );
        validate_sha256_identity(&recovered.student.update_sha256, "student update identity")?;
        ensure!(
            recovered.teacher.uri == txn.teacher_checkpoint
                && recovered.teacher.sha256 == txn.teacher_hash,
            "tensor snapshot teacher identity differs from transaction"
        );
        ensure!(
            recovered.student.checkpoint.uri == txn.student_checkpoint
                && recovered.student.checkpoint.sha256 == txn.student_hash
                && recovered.student.update_sha256 == txn.prospective_update_hash,
            "tensor snapshot student identity differs from transaction"
        );
        ensure!(
            recovered.completed.contains(&TensorStage::Retention)
                == recovered.retention_result.is_some(),
            "tensor snapshot retention decision disagrees with completed stages"
        );
        validate_tensor_progress(
            &recovered.completed,
            recovered.retention_result,
            &recovered.diagnostics,
        )?;
        validate_parameter_id_snapshot(
            &parameter_ids(&recovered.teacher.model),
            "teacher parameter IDs",
        )?;
        validate_parameter_id_snapshot(
            &parameter_ids(&recovered.student.checkpoint.model),
            "student parameter IDs",
        )?;

        ensure_directory(&self.root, "tensor transaction root")?;
        let generations = self.root.join("generations");
        ensure_directory(&generations, "tensor transaction generations")?;
        let staging = generations.join(format!(
            ".staging-{}-{}-{}",
            txn.id,
            std::process::id(),
            TENSOR_TXN_STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&staging).with_context(|| {
            format!("creating tensor transaction staging {}", staging.display())
        })?;
        let _guard = TensorStagingGuard(staging.clone());

        let teacher_path = staging.join(TENSOR_TXN_TEACHER);
        let student_path = staging.join(TENSOR_TXN_STUDENT);
        let optimizer_path = staging.join(TENSOR_TXN_OPTIMIZER);
        let update_pre_path = staging.join(TENSOR_TXN_UPDATE_PRE);
        let update_staged_path = staging.join(TENSOR_TXN_UPDATE_STAGED);
        save_safetensors(&recovered.teacher.model.clone().valid(), &teacher_path)?;
        sync_regular_file(&teacher_path, "tensor transaction teacher")?;
        save_safetensors(
            &recovered.student.checkpoint.model.clone().valid(),
            &student_path,
        )?;
        sync_regular_file(&student_path, "tensor transaction student")?;
        save_canonical_module_optimizer(&recovered.receiver_optimizer, &optimizer_path)
            .context("saving receiver optimizer state")?;
        sync_regular_file(&optimizer_path, "tensor transaction optimizer")?;
        write_new_synced(&update_pre_path, recovered.pre_update_state.as_bytes())?;
        write_new_synced(
            &update_staged_path,
            recovered.staged_update_state.as_bytes(),
        )?;

        let metadata = TensorTransactionMetadata {
            version: TENSOR_TXN_STORE_VERSION,
            txn_id: txn.id,
            teacher_uri: recovered.teacher.uri.clone(),
            teacher_sha256: recovered.teacher.sha256.clone(),
            student_uri: recovered.student.checkpoint.uri.clone(),
            student_sha256: recovered.student.checkpoint.sha256.clone(),
            prospective_update_sha256: recovered.student.update_sha256.clone(),
            teacher_parameter_ids: parameter_ids(&recovered.teacher.model),
            student_parameter_ids: parameter_ids(&recovered.student.checkpoint.model),
            completed: recovered.completed.clone(),
            retention_result: recovered.retention_result,
            diagnostics: recovered.diagnostics,
        };
        write_new_synced(
            &staging.join(TENSOR_TXN_METADATA),
            &serde_json::to_vec_pretty(&metadata)?,
        )?;

        let mut files = [
            TENSOR_TXN_METADATA,
            TENSOR_TXN_OPTIMIZER,
            TENSOR_TXN_STUDENT,
            TENSOR_TXN_TEACHER,
            TENSOR_TXN_UPDATE_PRE,
            TENSOR_TXN_UPDATE_STAGED,
        ]
        .into_iter()
        .map(|name| transaction_file(&staging, name))
        .collect::<Result<Vec<_>>>()?;
        files.sort_by(|left, right| left.path.cmp(&right.path));
        let manifest = TensorTransactionManifest {
            version: TENSOR_TXN_MANIFEST_VERSION,
            txn_id: txn.id,
            files,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        let manifest_sha256 = sha256_identity(&manifest_bytes);
        write_new_synced(&staging.join(TENSOR_TXN_MANIFEST), &manifest_bytes)?;
        sync_directory(&staging)?;

        let generation = format!(
            "sha256-{}",
            manifest_sha256
                .strip_prefix("sha256:")
                .expect("hash helper returns prefixed digest")
        );
        let sealed = generations.join(&generation);
        if sealed.exists() {
            verify_generation(&sealed, txn.id, &manifest_sha256)?;
            fs::remove_dir_all(&staging)
                .with_context(|| format!("removing duplicate staging {}", staging.display()))?;
        } else {
            fs::rename(&staging, &sealed).with_context(|| {
                format!("sealing tensor transaction generation {}", sealed.display())
            })?;
        }
        sync_directory(&generations)?;

        let pointer = TensorTransactionPointer {
            version: TENSOR_TXN_STORE_VERSION,
            txn_id: txn.id,
            generation,
            manifest_sha256,
        };
        publish_pointer(&self.root, &pointer)?;
        Ok(pointer)
    }

    /// Load and authenticate the exact tensor/optimizer boundary. `optimizer`
    /// supplies the configured optimizer groups; persisted moments replace its
    /// empty state after parameter IDs are restored.
    pub fn load(
        &self,
        txn: &ConsolidationTxn,
        config: &ModelDef,
        device: &Device,
        optimizer: ModuleOptimizer,
    ) -> Result<RecoveredTensorTransaction> {
        ensure_real_directory(&self.root, "tensor transaction root")?;
        ensure_real_directory(
            &self.root.join("generations"),
            "tensor transaction generations",
        )?;
        let pointer_path = self.root.join(TENSOR_TXN_POINTER);
        let pointer: TensorTransactionPointer = serde_json::from_slice(&read_regular_bounded(
            &pointer_path,
            MAX_TENSOR_TXN_MANIFEST_BYTES,
            "tensor transaction pointer",
        )?)?;
        self.load_pointer(txn, config, device, optimizer, &pointer)
    }

    /// Load the immutable generation named by checkpoint-v2 SleepState. This
    /// remains exact even if an interrupted later subphase advanced the store's
    /// convenience `current.json` pointer.
    pub fn load_recorded(
        &self,
        txn: &ConsolidationTxn,
        config: &ModelDef,
        device: &Device,
        optimizer: ModuleOptimizer,
    ) -> Result<RecoveredTensorTransaction> {
        let pointer = TensorTransactionPointer {
            version: TENSOR_TXN_STORE_VERSION,
            txn_id: txn.id,
            generation: txn
                .tensor_transaction_generation
                .clone()
                .context("SleepState has no tensor transaction generation")?,
            manifest_sha256: txn
                .tensor_transaction_manifest_hash
                .clone()
                .context("SleepState has no tensor transaction manifest hash")?,
        };
        self.load_pointer(txn, config, device, optimizer, &pointer)
    }

    fn load_pointer(
        &self,
        txn: &ConsolidationTxn,
        config: &ModelDef,
        device: &Device,
        optimizer: ModuleOptimizer,
        pointer: &TensorTransactionPointer,
    ) -> Result<RecoveredTensorTransaction> {
        self.load_pointer_inner(txn, config, device, optimizer, pointer, |_, _| Ok(()))
    }

    fn load_pointer_inner(
        &self,
        txn: &ConsolidationTxn,
        config: &ModelDef,
        device: &Device,
        optimizer: ModuleOptimizer,
        pointer: &TensorTransactionPointer,
        mut stage_hook: impl FnMut(TensorLoadStage, &Path) -> Result<()>,
    ) -> Result<RecoveredTensorTransaction> {
        ensure_real_directory(&self.root, "tensor transaction root")?;
        ensure_real_directory(
            &self.root.join("generations"),
            "tensor transaction generations",
        )?;
        ensure!(
            pointer.version == TENSOR_TXN_STORE_VERSION,
            "unsupported tensor transaction pointer version"
        );
        ensure!(
            pointer.txn_id == txn.id,
            "tensor transaction pointer belongs to another txn"
        );
        validate_sha256_identity(&pointer.manifest_sha256, "tensor manifest identity")?;
        validate_generation_identity(&pointer.generation, &pointer.manifest_sha256)?;
        let generation_path = self.root.join("generations").join(&pointer.generation);
        let (metadata, mut generation) =
            capture_verified_generation(&generation_path, txn.id, &pointer.manifest_sha256)?;
        stage_hook(TensorLoadStage::AfterCapture, &generation_path)?;
        ensure!(
            metadata.teacher_uri == txn.teacher_checkpoint
                && metadata.teacher_sha256 == txn.teacher_hash,
            "stored teacher identity differs from transaction"
        );
        ensure!(
            metadata.student_uri == txn.student_checkpoint
                && metadata.student_sha256 == txn.student_hash
                && metadata.prospective_update_sha256 == txn.prospective_update_hash,
            "stored student identity differs from transaction"
        );

        let mut teacher = Transformer::new(config, device)?;
        restore_parameter_ids(&mut teacher, &metadata.teacher_parameter_ids)?;
        load_safetensors_bytes(
            &mut teacher,
            generation.take(TENSOR_TXN_TEACHER, MAX_TENSOR_TXN_MEMBER_BYTES)?,
            "authenticated tensor-sleep teacher",
        )?;
        let mut student = Transformer::new(config, device)?;
        restore_parameter_ids(&mut student, &metadata.student_parameter_ids)?;
        load_safetensors_bytes(
            &mut student,
            generation.take(TENSOR_TXN_STUDENT, MAX_TENSOR_TXN_MEMBER_BYTES)?,
            "authenticated tensor-sleep student",
        )?;
        let optimizer_bytes = generation.take(TENSOR_TXN_OPTIMIZER, MAX_TENSOR_TXN_MEMBER_BYTES)?;
        let optimizer_bytes_len = optimizer_bytes.len();
        let optimizer_sha256 = sha256_identity(&optimizer_bytes);
        let optimizer = optimizer
            .from_bytes(Bytes::from_bytes_vec(optimizer_bytes))
            .context("loading authenticated receiver optimizer state")?;
        let canonical_optimizer = canonical_module_optimizer_bytes(&optimizer)?.to_vec();
        ensure!(
            canonical_optimizer.len() == optimizer_bytes_len
                && sha256_identity(&canonical_optimizer) == optimizer_sha256,
            "receiver optimizer bytes are non-canonical or failed exact restore"
        );
        let pre_update_state = ProspectiveUpdateSnapshot::new(
            generation.take(TENSOR_TXN_UPDATE_PRE, MAX_TENSOR_TXN_MEMBER_BYTES)?,
        )?;
        let staged_update_state = ProspectiveUpdateSnapshot::new(
            generation.take(TENSOR_TXN_UPDATE_STAGED, MAX_TENSOR_TXN_MEMBER_BYTES)?,
        )?;
        stage_hook(TensorLoadStage::AfterStagedLoad, &generation_path)?;
        generation.ensure_still_published()?;
        Ok(RecoveredTensorTransaction {
            txn_id: txn.id,
            teacher: ImmutableTransformerCheckpoint {
                uri: metadata.teacher_uri,
                sha256: metadata.teacher_sha256,
                model: teacher,
            },
            student: ProspectiveTransformerCandidate {
                checkpoint: ImmutableTransformerCheckpoint {
                    uri: metadata.student_uri,
                    sha256: metadata.student_sha256,
                    model: student,
                },
                update_sha256: metadata.prospective_update_sha256,
            },
            pre_update_state,
            staged_update_state,
            receiver_optimizer: optimizer,
            completed: metadata.completed,
            retention_result: metadata.retention_result,
            diagnostics: metadata.diagnostics,
        })
    }
}

fn transaction_file(root: &Path, name: &str) -> Result<TensorTransactionFile> {
    ensure!(
        matches!(
            name,
            TENSOR_TXN_METADATA
                | TENSOR_TXN_OPTIMIZER
                | TENSOR_TXN_STUDENT
                | TENSOR_TXN_TEACHER
                | TENSOR_TXN_UPDATE_PRE
                | TENSOR_TXN_UPDATE_STAGED
        ),
        "unsupported tensor transaction artifact name"
    );
    let (bytes, sha256) = hash_file(&root.join(name))?;
    Ok(TensorTransactionFile {
        path: name.to_owned(),
        bytes,
        sha256,
    })
}

fn validate_generation_name(name: &str) -> Result<()> {
    let digest = name
        .strip_prefix("sha256-")
        .context("tensor generation must use sha256-<64 lowercase hex>")?;
    validate_sha256_identity(&format!("sha256:{digest}"), "tensor generation identity")
}

fn validate_generation_identity(name: &str, manifest_sha256: &str) -> Result<()> {
    validate_generation_name(name)?;
    validate_sha256_identity(manifest_sha256, "tensor manifest identity")?;
    let generation_digest = name
        .strip_prefix("sha256-")
        .expect("validated tensor generation has a prefix");
    let manifest_digest = manifest_sha256
        .strip_prefix("sha256:")
        .expect("validated tensor manifest hash has a prefix");
    ensure!(
        generation_digest == manifest_digest,
        "tensor generation name differs from its manifest hash"
    );
    Ok(())
}

fn capture_verified_generation(
    generation: &Path,
    txn_id: u64,
    expected_manifest_hash: &str,
) -> Result<(TensorTransactionMetadata, AuthenticatedDirectorySnapshot)> {
    let mut captured = AuthenticatedDirectorySnapshot::capture(
        generation,
        &TENSOR_TXN_SCHEMA,
        "tensor transaction generation",
    )?;
    let manifest_bytes =
        captured.read_bounded(TENSOR_TXN_MANIFEST, MAX_TENSOR_TXN_MANIFEST_BYTES)?;
    ensure!(
        sha256_identity(&manifest_bytes) == expected_manifest_hash,
        "tensor transaction manifest hash mismatch"
    );
    let manifest: TensorTransactionManifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest.version == TENSOR_TXN_MANIFEST_VERSION && manifest.txn_id == txn_id,
        "tensor transaction manifest identity/version mismatch"
    );
    ensure!(
        manifest.files.len() == 6
            && manifest
                .files
                .windows(2)
                .all(|pair| pair[0].path < pair[1].path),
        "tensor transaction manifest file list is incomplete or unordered"
    );
    for file in &manifest.files {
        ensure!(
            matches!(
                file.path.as_str(),
                TENSOR_TXN_METADATA
                    | TENSOR_TXN_OPTIMIZER
                    | TENSOR_TXN_STUDENT
                    | TENSOR_TXN_TEACHER
                    | TENSOR_TXN_UPDATE_PRE
                    | TENSOR_TXN_UPDATE_STAGED
            ),
            "tensor transaction manifest contains an unsafe path"
        );
        validate_sha256_identity(&file.sha256, "tensor manifest member identity")?;
        ensure!(
            file.bytes <= MAX_TENSOR_TXN_MEMBER_BYTES,
            "tensor transaction artifact `{}` exceeds the per-member size limit",
            file.path
        );
        captured.verify(&file.path, file.bytes, &file.sha256)?;
    }
    let metadata_bytes =
        captured.read_bounded(TENSOR_TXN_METADATA, MAX_TENSOR_TXN_METADATA_BYTES)?;
    let transaction: TensorTransactionMetadata = serde_json::from_slice(&metadata_bytes)?;
    ensure!(
        transaction.version == TENSOR_TXN_STORE_VERSION && transaction.txn_id == txn_id,
        "tensor transaction metadata identity/version mismatch"
    );
    validate_sha256_identity(&transaction.teacher_sha256, "teacher identity")?;
    validate_sha256_identity(&transaction.student_sha256, "student identity")?;
    validate_sha256_identity(
        &transaction.prospective_update_sha256,
        "prospective update identity",
    )?;
    ensure!(
        !transaction.teacher_uri.trim().is_empty() && !transaction.student_uri.trim().is_empty(),
        "stored tensor transaction has an empty checkpoint URI"
    );
    validate_parameter_id_snapshot(&transaction.teacher_parameter_ids, "teacher parameter IDs")?;
    validate_parameter_id_snapshot(&transaction.student_parameter_ids, "student parameter IDs")?;
    ensure!(
        transaction.teacher_parameter_ids.len() == transaction.student_parameter_ids.len(),
        "stored teacher/student parameter topologies differ"
    );
    validate_tensor_progress(
        &transaction.completed,
        transaction.retention_result,
        &transaction.diagnostics,
    )?;
    Ok((transaction, captured))
}

fn verify_generation(
    generation: &Path,
    txn_id: u64,
    expected_manifest_hash: &str,
) -> Result<TensorTransactionMetadata> {
    let (metadata, captured) =
        capture_verified_generation(generation, txn_id, expected_manifest_hash)?;
    captured.ensure_still_published()?;
    Ok(metadata)
}

fn publish_pointer(root: &Path, pointer: &TensorTransactionPointer) -> Result<()> {
    validate_generation_identity(&pointer.generation, &pointer.manifest_sha256)?;
    let destination = root.join(TENSOR_TXN_POINTER);
    if destination.exists() {
        ensure_regular_file(&destination, "existing tensor transaction pointer")?;
    }
    let temporary = root.join(format!(
        ".current-{}-{}-{}.tmp",
        pointer.txn_id,
        std::process::id(),
        TENSOR_TXN_STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    write_new_synced(&temporary, &serde_json::to_vec_pretty(pointer)?)?;
    fs::rename(&temporary, &destination).context("atomically publishing tensor transaction")?;
    sync_directory(root)?;
    Ok(())
}

fn validate_parameter_id_snapshot(ids: &[u64], label: &str) -> Result<()> {
    ensure!(!ids.is_empty(), "{label} are empty");
    ensure!(
        ids.iter().collect::<BTreeSet<_>>().len() == ids.len(),
        "{label} contain duplicates"
    );
    Ok(())
}

fn validate_tensor_progress(
    completed: &BTreeSet<TensorStage>,
    retention_result: Option<bool>,
    diagnostics: &TensorSleepDiagnostics,
) -> Result<()> {
    ensure!(
        completed.contains(&TensorStage::Student) && completed.contains(&TensorStage::Prospective),
        "tensor transaction snapshot is missing its prospective/student stages"
    );
    for (later, earlier) in [
        (TensorStage::Knowledge, TensorStage::Student),
        (TensorStage::Imitation, TensorStage::Knowledge),
        (TensorStage::Retention, TensorStage::Imitation),
        (TensorStage::Committed, TensorStage::Retention),
    ] {
        ensure!(
            !completed.contains(&later) || completed.contains(&earlier),
            "tensor transaction stages are not a contiguous prefix"
        );
    }
    ensure!(
        completed.contains(&TensorStage::Retention) == retention_result.is_some(),
        "tensor retention decision disagrees with completed stages"
    );
    ensure!(
        !completed.contains(&TensorStage::Committed) || retention_result == Some(true),
        "committed tensor transaction did not pass retention"
    );
    ensure!(
        !(completed.contains(&TensorStage::Committed)
            && completed.contains(&TensorStage::RolledBack)),
        "tensor transaction is both committed and rolled back"
    );
    for value in [
        diagnostics.knowledge_kl_sum,
        diagnostics.teacher_entropy_sum,
        diagnostics.student_entropy_sum,
        diagnostics.imitation_semantic_sum,
        diagnostics.imitation_edit_sum,
        diagnostics.imitation_threshold_sum,
        diagnostics.imitation_reward_sum,
        diagnostics.imitation_reward_square_sum,
        diagnostics.imitation_grpo_kl_sum,
        f64::from(diagnostics.retention_anchor_kl),
        f64::from(diagnostics.teacher_stable_anchor),
        f64::from(diagnostics.student_stable_anchor),
        f64::from(diagnostics.teacher_incorporation),
        f64::from(diagnostics.student_incorporation),
        f64::from(diagnostics.anchor_delta),
        f64::from(diagnostics.incorporation_gain),
    ] {
        ensure!(
            value.is_finite(),
            "tensor transaction diagnostics are non-finite"
        );
    }
    if completed.contains(&TensorStage::Knowledge) {
        ensure!(
            diagnostics.knowledge_updates > 0
                && diagnostics.knowledge_tokens > 0
                && diagnostics.selected_gradient_tensors > 0
                && diagnostics.knowledge_kl_sum >= 0.0
                && diagnostics.teacher_entropy_sum >= 0.0
                && diagnostics.student_entropy_sum >= 0.0,
            "knowledge stage has no durable update diagnostics"
        );
    }
    if completed.contains(&TensorStage::Imitation) {
        ensure!(
            diagnostics.imitation_updates > 0
                && diagnostics.imitation_samples > 0
                && diagnostics.imitation_semantic_sum >= 0.0
                && diagnostics.imitation_edit_sum >= 0.0
                && diagnostics.imitation_threshold_sum >= 0.0
                && diagnostics.imitation_reward_square_sum >= 0.0
                && diagnostics.imitation_grpo_kl_sum >= 0.0,
            "imitation stage has no durable update diagnostics"
        );
    }
    Ok(())
}

fn parameter_ids(model: &Transformer) -> Vec<u64> {
    burn::module::list_param_ids(model)
        .into_iter()
        .map(|id| id.val())
        .collect()
}

struct RestoreParameterIds<'a> {
    ids: std::slice::Iter<'a, u64>,
}

impl RestoreParameterIds<'_> {
    fn next(&mut self) -> ParamId {
        ParamId::from(
            self.ids
                .next()
                .copied()
                .expect("authenticated tensor transaction has too few parameter IDs"),
        )
    }
}

impl ModuleMapper for RestoreParameterIds<'_> {
    fn map_float<const D: usize>(&mut self, parameter: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (_, tensor, mapper) = parameter.consume();
        Param::from_mapped_value(self.next(), tensor, mapper)
    }

    fn map_int<const D: usize>(
        &mut self,
        parameter: Param<Tensor<D, Int>>,
    ) -> Param<Tensor<D, Int>> {
        let (_, tensor, mapper) = parameter.consume();
        Param::from_mapped_value(self.next(), tensor, mapper)
    }

    fn map_bool<const D: usize>(
        &mut self,
        parameter: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        let (_, tensor, mapper) = parameter.consume();
        Param::from_mapped_value(self.next(), tensor, mapper)
    }
}

/// Restore the stable parameter identity ordering persisted alongside a model
/// checkpoint. SafeTensors carries tensor values but not Burn `ParamId`s;
/// optimizer snapshots therefore pin this companion identity vector.
pub fn restore_parameter_ids(model: &mut Transformer, ids: &[u64]) -> Result<()> {
    ensure!(
        ids.iter().copied().collect::<BTreeSet<_>>().len() == ids.len(),
        "tensor transaction repeats a stored parameter ID"
    );
    ensure!(
        ids.len() == burn::module::list_param_ids(model).len(),
        "tensor transaction parameter-ID topology differs from model"
    );
    let mut mapper = RestoreParameterIds { ids: ids.iter() };
    *model = model.clone().map(&mut mapper);
    ensure!(mapper.ids.next().is_none(), "too many stored parameter IDs");
    ensure!(
        parameter_ids(model) == ids,
        "tensor transaction failed to restore its exact parameter-ID ordering"
    );
    Ok(())
}

pub struct TensorConsolidationBackend<U, R, J, E, P> {
    live: ImmutableTransformerCheckpoint,
    device: Device,
    config: TensorConsolidationConfig,
    updates: U,
    rollouts: R,
    judge: J,
    evaluator: E,
    publisher: P,
    teacher: Option<ImmutableTransformerCheckpoint>,
    student: Option<ProspectiveTransformerCandidate>,
    pre_update_state: Option<ProspectiveUpdateSnapshot>,
    staged_update_state: Option<ProspectiveUpdateSnapshot>,
    receiver_ids: Vec<ParamId>,
    reclaimed_sender_ids: Vec<ParamId>,
    receiver_optimizer: ModuleOptimizer,
    stages: BTreeSet<TensorStage>,
    retention: Option<bool>,
    diagnostics: TensorSleepDiagnostics,
}

impl<U, R, J, E, P> TensorConsolidationBackend<U, R, J, E, P>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        live: ImmutableTransformerCheckpoint,
        device: Device,
        config: TensorConsolidationConfig,
        updates: U,
        rollouts: R,
        judge: J,
        evaluator: E,
        publisher: P,
    ) -> Result<Self> {
        live.validate()?;
        config.validate()?;
        ensure!(
            judge.artifact_hash() == config.imitation.semantic_judge_hash,
            "semantic judge hash is not pinned value"
        );
        ensure!(
            evaluator.artifact_hash() == config.retention.evaluator_hash,
            "retention evaluator hash is not pinned value"
        );
        ensure!(
            evaluator.suite_hash() == config.retention.suite_hash,
            "retention evaluator suite hash is not the pinned input"
        );
        let receiver_optimizer = AdamWConfig::new()
            .with_weight_decay(config.receiver_weight_decay)
            .init();
        Ok(Self {
            live,
            device,
            config,
            updates,
            rollouts,
            judge,
            evaluator,
            publisher,
            teacher: None,
            student: None,
            pre_update_state: None,
            staged_update_state: None,
            receiver_ids: Vec::new(),
            reclaimed_sender_ids: Vec::new(),
            receiver_optimizer,
            stages: BTreeSet::new(),
            retention: None,
            diagnostics: TensorSleepDiagnostics::default(),
        })
    }

    pub fn live_checkpoint(&self) -> &ImmutableTransformerCheckpoint {
        &self.live
    }
    pub fn diagnostics(&self) -> TensorSleepDiagnostics {
        self.diagnostics
    }
    pub fn config(&self) -> &TensorConsolidationConfig {
        &self.config
    }
    pub fn receiver_parameter_ids(&self) -> &[ParamId] {
        &self.receiver_ids
    }
    pub fn components(&self) -> (&U, &R, &J, &E, &P) {
        (
            &self.updates,
            &self.rollouts,
            &self.judge,
            &self.evaluator,
            &self.publisher,
        )
    }

    /// Clone the exact teacher/student/optimizer boundary for atomic
    /// publication with [`TensorTransactionStore`]. Call this from the same
    /// trainer checkpoint callback that persists [`SleepState`].
    pub fn snapshot_inflight(&self, txn: &ConsolidationTxn) -> Result<RecoveredTensorTransaction> {
        Ok(RecoveredTensorTransaction {
            txn_id: txn.id,
            teacher: self.teacher(txn)?.clone(),
            student: self.student(txn)?.clone(),
            pre_update_state: self
                .pre_update_state
                .clone()
                .context("pre-update optimizer snapshot is not staged")?,
            staged_update_state: self
                .staged_update_state
                .clone()
                .context("prospective optimizer snapshot is not staged")?,
            receiver_optimizer: self.receiver_optimizer.clone(),
            completed: self.stages.clone(),
            retention_result: self.retention,
            diagnostics: self.diagnostics,
        })
    }

    pub fn restore_inflight(
        &mut self,
        txn: &ConsolidationTxn,
        recovered: RecoveredTensorTransaction,
    ) -> Result<()> {
        ensure!(
            recovered.txn_id == txn.id,
            "recovered tensor transaction ID differs"
        );
        recovered.teacher.validate()?;
        recovered.student.checkpoint.validate()?;
        ensure!(
            recovered.teacher.uri == txn.teacher_checkpoint
                && recovered.teacher.sha256 == txn.teacher_hash,
            "recovered teacher identity differs from transaction"
        );
        ensure!(
            recovered.student.checkpoint.uri == txn.student_checkpoint
                && recovered.student.checkpoint.sha256 == txn.student_hash
                && recovered.student.update_sha256 == txn.prospective_update_hash,
            "recovered student identity differs from transaction"
        );
        ensure!(
            !recovered.completed.contains(&TensorStage::Committed)
                && !recovered.completed.contains(&TensorStage::RolledBack),
            "cannot restore a terminal tensor transaction"
        );
        ensure!(
            recovered.completed.contains(&TensorStage::Retention)
                == recovered.retention_result.is_some(),
            "recovered retention decision disagrees with completed tensor stages"
        );
        validate_tensor_progress(
            &recovered.completed,
            recovered.retention_result,
            &recovered.diagnostics,
        )?;
        let statuses = recovered.student.checkpoint.model.memory_slot_statuses();
        let receiver_ids = if txn.terminal {
            recovered
                .student
                .checkpoint
                .model
                .memory_tier_base_parameter_ids_all_layers(txn.sender)?
        } else {
            let receiver = statuses
                .iter()
                .filter(|s| s.tier == txn.receiver && s.slot == txn.receiver_slot)
                .collect::<Vec<_>>();
            ensure!(
                !receiver.is_empty() && receiver.iter().all(|s| s.active),
                "recovered receiver is not active in every memory layer"
            );
            receiver
                .into_iter()
                .flat_map(|s| s.parameter_ids.clone())
                .collect()
        };
        let reclaimed_sender_ids: Vec<ParamId> = statuses
            .iter()
            .filter(|status| {
                status.tier == txn.sender
                    && txn.sender_slots_to_reset.contains(&status.slot)
                    && !status.active
            })
            .flat_map(|status| status.parameter_ids.clone())
            .collect();
        ensure!(
            txn.sender_slots_to_reset.is_empty() || !reclaimed_sender_ids.is_empty(),
            "recovered student has not reclaimed its planned sender slots"
        );
        // Restore the wake-side optimizer/accumulator only after all tensor
        // metadata and topology checks pass. The adapter contract makes a
        // failed restore locally atomic.
        self.updates
            .restore_state(txn, &recovered.staged_update_state)
            .context("restoring prospective sender optimizer state")?;
        self.teacher = Some(recovered.teacher);
        self.student = Some(recovered.student);
        self.pre_update_state = Some(recovered.pre_update_state);
        self.staged_update_state = Some(recovered.staged_update_state);
        self.receiver_optimizer = recovered.receiver_optimizer;
        self.receiver_ids = receiver_ids;
        self.reclaimed_sender_ids = reclaimed_sender_ids;
        self.stages = recovered.completed;
        self.retention = recovered.retention_result;
        self.diagnostics = recovered.diagnostics;
        Ok(())
    }

    fn teacher(&self, txn: &ConsolidationTxn) -> Result<&ImmutableTransformerCheckpoint> {
        let teacher = self
            .teacher
            .as_ref()
            .context("immutable teacher is not staged")?;
        ensure!(
            teacher.uri == txn.teacher_checkpoint && teacher.sha256 == txn.teacher_hash,
            "teacher identity drifted during transaction"
        );
        Ok(teacher)
    }

    fn student(&self, txn: &ConsolidationTxn) -> Result<&ProspectiveTransformerCandidate> {
        let student = self.student.as_ref().context("student is not staged")?;
        ensure!(
            student.checkpoint.uri == txn.student_checkpoint
                && student.checkpoint.sha256 == txn.student_hash
                && student.update_sha256 == txn.prospective_update_hash,
            "student identity drifted during transaction"
        );
        Ok(student)
    }

    fn optimize_kl(&mut self, txn: &ConsolidationTxn, batch: &TokenRolloutBatch) -> Result<()> {
        let teacher = self.teacher(txn)?.model.clone();
        let mut student = self.student(txn)?.checkpoint.model.clone();
        let (loss, stats) = forward_kl_tensor(
            &teacher,
            &student,
            batch,
            &self.device,
            self.config.knowledge.chunk_tokens,
            self.config.knowledge.temperature,
        )?;
        let mut gradients = loss
            .mul_scalar(self.config.knowledge.forward_kl_weight)
            .backward();
        let selected = GradientsParams::from_params(&mut gradients, &student, &self.receiver_ids);
        ensure!(
            !selected.is_empty(),
            "KL produced no receiver-slot gradients"
        );
        ensure!(
            selected.len() <= self.receiver_ids.len(),
            "gradient scope escaped receiver slot"
        );
        self.diagnostics.selected_gradient_tensors = selected.len();
        student = self.receiver_optimizer.step(
            self.config.receiver_learning_rate.into(),
            student,
            selected,
        );
        self.student
            .as_mut()
            .expect("student checked")
            .checkpoint
            .model = student;
        self.diagnostics.knowledge_updates += 1;
        self.diagnostics.knowledge_tokens = self
            .diagnostics
            .knowledge_tokens
            .checked_add(stats.tokens)
            .context("knowledge token metric overflow")?;
        self.diagnostics.knowledge_kl_sum += f64::from(stats.forward_kl);
        self.diagnostics.teacher_entropy_sum += f64::from(stats.teacher_entropy);
        self.diagnostics.student_entropy_sum += f64::from(stats.student_entropy);
        Ok(())
    }

    fn optimize_grpo(
        &mut self,
        txn: &ConsolidationTxn,
        group: &ImitationGroup,
        behavior: &Transformer,
    ) -> Result<()> {
        validate_group(
            group,
            &self.student(txn)?.checkpoint.model,
            self.config.imitation.grpo_group_size,
        )?;
        let scored = group
            .candidates
            .iter()
            .map(|candidate| {
                let semantic =
                    self.judge
                        .score(&group.prefix, &group.teacher_continuation, candidate)?;
                ensure!(
                    semantic.is_finite() && (0.0..=1.0).contains(&semantic),
                    "semantic judge score is outside [0,1]"
                );
                let edit = thresholded_edit_reward(
                    &group.teacher_continuation,
                    candidate,
                    self.config.imitation.maximum_edit_distance,
                );
                let threshold = self.config.imitation.maximum_edit_distance as f32
                    / group.teacher_continuation.len().max(candidate.len()).max(1) as f32;
                let reward = self.config.imitation.semantic_weight * semantic
                    + (1.0 - self.config.imitation.semantic_weight) * edit;
                Ok((semantic, edit, threshold.min(1.0), reward))
            })
            .collect::<Result<Vec<_>>>()?;
        let rewards = scored
            .iter()
            .map(|(_, _, _, reward)| *reward)
            .collect::<Vec<_>>();
        if rewards
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() <= 1e-7)
        {
            return Ok(());
        }
        let mut student = self.student(txn)?.checkpoint.model.clone();
        let teacher = self.teacher(txn)?.model.clone();
        let max_tokens = group
            .candidates
            .iter()
            .map(Vec::len)
            .max()
            .context("empty GRPO candidate group")?;
        let current_log_probs = padded_group_log_probabilities(
            &student,
            &group.prefix,
            &group.candidates,
            max_tokens,
            &self.device,
        )?;
        let behavior_log_probs = padded_group_log_probabilities(
            behavior,
            &group.prefix,
            &group.candidates,
            max_tokens,
            &self.device,
        )?
        .detach();
        let reference_log_probs = padded_group_log_probabilities(
            &teacher,
            &group.prefix,
            &group.candidates,
            max_tokens,
            &self.device,
        )?
        .detach();
        let active_mask = Tensor::<2>::from_data(
            TensorData::new(
                group
                    .candidates
                    .iter()
                    .flat_map(|candidate| {
                        (0..max_tokens).map(|index| if index < candidate.len() { 1.0 } else { 0.0 })
                    })
                    .collect(),
                [group.candidates.len(), max_tokens],
            ),
            &self.device,
        );
        let log_ref_minus_policy =
            reference_log_probs.clone().detach() - current_log_probs.clone().detach();
        let kl = log_ref_minus_policy.clone().exp() - log_ref_minus_policy - 1;
        let active_tokens = active_mask.clone().sum();
        let mean_kl = (kl * active_mask.clone()).sum() / active_tokens;
        let mean_kl = mean_kl
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading imitation KL metric")?[0];
        ensure!(
            mean_kl.is_finite() && mean_kl >= -1e-6,
            "invalid imitation KL metric {mean_kl}"
        );
        let rewards = Tensor::<1>::from_data(
            TensorData::new(rewards, [group.candidates.len()]),
            &self.device,
        );
        let objective = crate::posttrain::grpo_loss_tensor(
            rewards,
            current_log_probs,
            behavior_log_probs,
            Some(reference_log_probs),
            active_mask,
            self.config.grpo_clip_epsilon,
            self.config.grpo_advantage_epsilon,
            self.config.grpo_kl_coefficient,
        )?;
        let mut gradients = objective.backward();
        let selected = GradientsParams::from_params(&mut gradients, &student, &self.receiver_ids);
        ensure!(
            !selected.is_empty(),
            "GRPO produced no receiver-slot gradients"
        );
        ensure!(
            selected.len() <= self.receiver_ids.len(),
            "GRPO gradient scope escaped receiver slot"
        );
        self.diagnostics.selected_gradient_tensors = selected.len();
        student = self.receiver_optimizer.step(
            self.config.receiver_learning_rate.into(),
            student,
            selected,
        );
        self.student
            .as_mut()
            .expect("student checked")
            .checkpoint
            .model = student;
        self.diagnostics.imitation_updates += 1;
        self.diagnostics.imitation_samples = self
            .diagnostics
            .imitation_samples
            .checked_add(scored.len() as u64)
            .context("imitation sample metric overflow")?;
        for (semantic, edit, threshold, reward) in scored {
            self.diagnostics.imitation_semantic_sum += f64::from(semantic);
            self.diagnostics.imitation_edit_sum += f64::from(edit);
            self.diagnostics.imitation_threshold_sum += f64::from(threshold);
            self.diagnostics.imitation_reward_sum += f64::from(reward);
            self.diagnostics.imitation_reward_square_sum += f64::from(reward * reward);
        }
        self.diagnostics.imitation_grpo_kl_sum += f64::from(mean_kl.max(0.0));
        Ok(())
    }
}

impl<U, R, J, E, P> ConsolidationBackend for TensorConsolidationBackend<U, R, J, E, P>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
{
    fn knowledge_rng_count(&self) -> Result<u64> {
        self.config.knowledge.rollout_count_u64()
    }

    fn imitation_rng_count(&self) -> Result<u64> {
        self.config.imitation.group_size_u64()
    }

    fn compute_prospective_update(&mut self, txn: &ConsolidationTxn) -> Result<()> {
        if self.stages.contains(&TensorStage::Prospective) {
            return Ok(());
        }
        ensure!(
            self.teacher.is_none()
                && self.student.is_none()
                && self.pre_update_state.is_none()
                && self.staged_update_state.is_none(),
            "another tensor transaction is active"
        );
        self.live.validate()?;
        ensure!(
            self.live.uri == txn.teacher_checkpoint && self.live.sha256 == txn.teacher_hash,
            "live checkpoint is not the transaction's content-addressed teacher"
        );
        let teacher = self.live.clone();
        let pre_update_state = self
            .updates
            .snapshot_state(txn)
            .context("snapshotting sender optimizer before prospective update")?;
        let staged = (|| {
            let student = self.updates.stage(txn, &teacher.model)?;
            student.checkpoint.validate()?;
            validate_sha256_identity(&student.update_sha256, "student update identity")?;
            let derived_update = prospective_update_hash_at_boundary(
                &teacher.model,
                &student.checkpoint.model,
                txn.sender,
            )
            .context("validating prospective sender update scope")?;
            ensure!(
                student.checkpoint.uri == txn.student_checkpoint
                    && student.checkpoint.sha256 == txn.student_hash,
                "prospective student checkpoint identity differs from transaction"
            );
            ensure!(
                student.update_sha256 == derived_update
                    && txn.prospective_update_hash == derived_update,
                "prospective update hash differs from canonical sender-tier delta"
            );
            ensure!(
                student.checkpoint.uri != teacher.uri
                    && student.checkpoint.sha256 != teacher.sha256,
                "student aliases immutable teacher"
            );
            let staged_update_state = self
                .updates
                .snapshot_state(txn)
                .context("snapshotting staged sender optimizer update")?;
            Ok((student, staged_update_state))
        })();
        let (student, staged_update_state) = match staged {
            Ok(staged) => staged,
            Err(error) => {
                self.updates
                    .restore_state(txn, &pre_update_state)
                    .context("restoring sender optimizer after failed prospective update")?;
                return Err(error);
            }
        };
        self.teacher = Some(teacher);
        self.student = Some(student);
        self.pre_update_state = Some(pre_update_state);
        self.staged_update_state = Some(staged_update_state);
        self.stages.insert(TensorStage::Prospective);
        Ok(())
    }

    fn stage_student(&mut self, txn: &ConsolidationTxn) -> Result<()> {
        if self.stages.contains(&TensorStage::Student) {
            return Ok(());
        }
        ensure!(
            self.stages.contains(&TensorStage::Prospective),
            "student staged before prospective update"
        );
        let mut student = self.student.take().context("prospective student missing")?;
        self.receiver_ids = if txn.terminal {
            student
                .checkpoint
                .model
                .memory_tier_base_parameter_ids_all_layers(txn.sender)
                .context("selecting terminal tier base parameters")?
        } else {
            student
                .checkpoint
                .model
                .activate_memory_slot_all_layers(txn.receiver, txn.receiver_slot)
                .context("activating receiver reserve slot")?
        };
        ensure!(
            !self.receiver_ids.is_empty(),
            "receiver slot exposes no trainable parameters"
        );
        // Reclamation is private to the student until atomic commit. This is
        // essential for transfer: KL/GRPO must teach the slower receiver to
        // compensate for the sender scratch capacity that will be released.
        self.reclaimed_sender_ids.clear();
        for &slot in &txn.sender_slots_to_reset {
            self.reclaimed_sender_ids.extend(
                student.checkpoint.model.reset_memory_slot_all_layers(
                    txn.sender,
                    slot,
                    txn.id ^ (slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                )?,
            );
        }
        self.student = Some(student);
        self.stages.insert(TensorStage::Student);
        Ok(())
    }

    fn knowledge_seed(&mut self, txn: &ConsolidationTxn) -> Result<()> {
        if self.stages.contains(&TensorStage::Knowledge) {
            return Ok(());
        }
        ensure!(
            self.stages.contains(&TensorStage::Student),
            "knowledge seeding before student stage"
        );
        let old_student = self.student.clone();
        let old_optimizer = self.receiver_optimizer.clone();
        let old_diagnostics = self.diagnostics;
        let reservation = txn
            .knowledge_rng
            .context("knowledge seeding has no persisted RNG reservation")?;
        self.device.seed(tensor_subphase_seed(
            txn,
            reservation,
            KNOWLEDGE_DEVICE_RNG_DOMAIN,
        ));
        let result = (|| {
            let mut rollout_tokens = 0_u64;
            for (owner, count) in [
                (
                    RolloutOwner::Teacher,
                    self.config.knowledge.teacher_rollouts,
                ),
                (
                    RolloutOwner::DetachedStudent,
                    self.config.knowledge.detached_student_rollouts,
                ),
            ] {
                let model = match owner {
                    RolloutOwner::Teacher => self.teacher(txn)?.model.clone(),
                    RolloutOwner::DetachedStudent => self.student(txn)?.checkpoint.model.clone(),
                };
                let batches = self
                    .rollouts
                    .knowledge_rollouts(txn, owner, &model, count)?;
                ensure!(
                    batches.len() == count,
                    "rollout source returned {} {owner:?} batches, expected {count}",
                    batches.len()
                );
                rollout_tokens = rollout_tokens
                    .checked_add(validate_rollout_batches(
                        &batches,
                        &model,
                        "knowledge rollout source",
                    )?)
                    .context("knowledge rollout token count overflow")?;
                ensure!(
                    rollout_tokens <= MAX_TENSOR_SUBPHASE_TOKENS as u64,
                    "knowledge rollouts exceed the {MAX_TENSOR_SUBPHASE_TOKENS}-token subphase limit"
                );
                for batch in &batches {
                    self.optimize_kl(txn, batch)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.student = old_student;
            self.receiver_optimizer = old_optimizer;
            self.diagnostics = old_diagnostics;
            return Err(error);
        }
        self.stages.insert(TensorStage::Knowledge);
        Ok(())
    }

    fn learn_to_imitate(&mut self, txn: &ConsolidationTxn) -> Result<()> {
        if self.stages.contains(&TensorStage::Imitation) {
            return Ok(());
        }
        ensure!(
            self.stages.contains(&TensorStage::Knowledge),
            "imitation before knowledge seeding"
        );
        let teacher = self.teacher(txn)?.model.clone();
        let student_snapshot = self.student(txn)?.checkpoint.model.clone();
        let reservation = txn
            .imitation_rng
            .context("imitation has no persisted RNG reservation")?;
        self.device.seed(tensor_subphase_seed(
            txn,
            reservation,
            IMITATION_DEVICE_RNG_DOMAIN,
        ));
        let groups = self.rollouts.imitation_groups(
            txn,
            &teacher,
            &student_snapshot,
            self.config.imitation.grpo_group_size,
        )?;
        ensure!(!groups.is_empty(), "imitation source returned no groups");
        validate_imitation_groups(
            &groups,
            &student_snapshot,
            self.config.imitation.grpo_group_size,
            self.config.imitation.maximum_edit_distance,
        )?;
        let old_student = self.student.clone();
        let old_optimizer = self.receiver_optimizer.clone();
        let old_diagnostics = self.diagnostics;
        for group in &groups {
            if let Err(error) = self.optimize_grpo(txn, group, &student_snapshot) {
                self.student = old_student;
                self.receiver_optimizer = old_optimizer;
                self.diagnostics = old_diagnostics;
                return Err(error);
            }
        }
        if self.diagnostics.imitation_updates == old_diagnostics.imitation_updates {
            self.student = old_student;
            self.receiver_optimizer = old_optimizer;
            self.diagnostics = old_diagnostics;
            bail!("imitation rewards produced no receiver-slot update");
        }
        self.stages.insert(TensorStage::Imitation);
        Ok(())
    }

    fn retention_passes(&mut self, txn: &ConsolidationTxn) -> Result<bool> {
        if self.stages.contains(&TensorStage::Retention) {
            return self.retention.context("retention stage has no decision");
        }
        let teacher = self.teacher(txn)?.model.clone();
        let student = self.student(txn)?.checkpoint.model.clone();
        let anchors = self.evaluator.anchor_rollouts(txn)?;
        ensure!(
            !anchors.is_empty(),
            "retention evaluator returned no anchors"
        );
        validate_rollout_batches(&anchors, &teacher, "retention anchor evaluator")?;
        let mut anchor_kl = 0.0;
        for batch in &anchors {
            anchor_kl += forward_kl_value(
                &teacher,
                &student,
                batch,
                &self.device,
                self.config.knowledge.chunk_tokens,
                self.config.knowledge.temperature,
            )?;
        }
        anchor_kl /= anchors.len() as f32;
        let teacher_scores = self.evaluator.score(txn, &teacher)?;
        let student_scores = self.evaluator.score(txn, &student)?;
        for value in [
            teacher_scores.stable_anchor,
            teacher_scores.incorporation,
            student_scores.stable_anchor,
            student_scores.incorporation,
        ] {
            ensure!(
                value.is_finite(),
                "retention evaluator returned non-finite score"
            );
        }
        let anchor_delta = student_scores.stable_anchor - teacher_scores.stable_anchor;
        let incorporation_gain = student_scores.incorporation - teacher_scores.incorporation;
        let passes = anchor_kl <= self.config.retention.max_anchor_forward_kl
            && anchor_delta >= -self.config.retention.max_anchor_regression
            && incorporation_gain >= self.config.retention.min_incorporation_gain;
        self.diagnostics.retention_anchor_kl = anchor_kl;
        self.diagnostics.teacher_stable_anchor = teacher_scores.stable_anchor;
        self.diagnostics.student_stable_anchor = student_scores.stable_anchor;
        self.diagnostics.teacher_incorporation = teacher_scores.incorporation;
        self.diagnostics.student_incorporation = student_scores.incorporation;
        self.diagnostics.anchor_delta = anchor_delta;
        self.diagnostics.incorporation_gain = incorporation_gain;
        self.retention = Some(passes);
        self.stages.insert(TensorStage::Retention);
        Ok(passes)
    }

    fn commit(&mut self, txn: &ConsolidationTxn) -> Result<CommittedCandidate> {
        if self.stages.contains(&TensorStage::Committed) {
            return Ok(CommittedCandidate {
                checkpoint: self.live.uri.clone(),
                sha256: self.live.sha256.clone(),
            });
        }
        ensure!(
            self.retention == Some(true),
            "candidate cannot commit before retention passes"
        );
        let candidate = self.student(txn)?.checkpoint.model.clone();
        let candidate_parameters = model_parameter_hash(&candidate)?;
        let published = self.publisher.publish_candidate(txn, &candidate)?;
        published.validate()?;
        // The publisher must attest to the exact candidate it was given; an
        // independent probe hash prevents it returning the teacher identity.
        ensure!(
            published.uri != self.teacher(txn)?.uri
                && published.sha256 != self.teacher(txn)?.sha256,
            "publisher returned the immutable teacher as candidate"
        );
        ensure!(
            model_parameter_hash(&published.model)? == candidate_parameters,
            "publisher returned weights which differ from the committed candidate"
        );
        // Publication is rollbackable through `restore_teacher`; optimizer
        // reclamation is required to be locally failure-atomic. This ordering
        // ensures a failed publication cannot erase live optimizer moments.
        self.updates
            .clear_reclaimed_optimizer_state(txn, &self.reclaimed_sender_ids)?;
        self.live = published;
        self.stages.insert(TensorStage::Committed);
        Ok(CommittedCandidate {
            checkpoint: self.live.uri.clone(),
            sha256: self.live.sha256.clone(),
        })
    }

    fn restore_teacher(&mut self, txn: &ConsolidationTxn) -> Result<()> {
        if self.stages.contains(&TensorStage::RolledBack) {
            return Ok(());
        }
        ensure!(
            !self.stages.contains(&TensorStage::Committed),
            "cannot roll back committed tensor candidate"
        );
        // A failure before prospective staging has no tensor or optimizer
        // mutation to undo.
        if self.teacher.is_none() {
            ensure!(
                self.student.is_none()
                    && self.pre_update_state.is_none()
                    && self.staged_update_state.is_none(),
                "partial prospective transaction has no teacher snapshot"
            );
            self.stages.insert(TensorStage::RolledBack);
            return Ok(());
        }
        let teacher = self.teacher(txn)?.clone();
        let pre_update_state = self
            .pre_update_state
            .as_ref()
            .context("rollback has no pre-update optimizer snapshot")?
            .clone();
        self.updates
            .restore_state(txn, &pre_update_state)
            .context("restoring pre-update sender optimizer state")?;
        self.publisher.restore_teacher(txn, &teacher)?;
        self.live = teacher;
        self.student = None;
        self.pre_update_state = None;
        self.staged_update_state = None;
        self.receiver_ids.clear();
        self.reclaimed_sender_ids.clear();
        self.receiver_optimizer = AdamWConfig::new()
            .with_weight_decay(self.config.receiver_weight_decay)
            .init();
        self.stages.insert(TensorStage::RolledBack);
        Ok(())
    }
}

/// Operational in-process entry point. The caller's progress sink persists
/// SleepState together with the normal model/optimizer checkpoint after every
/// subphase, so this function can be invoked again on resume.
pub fn execute_tensor_consolidation<U, R, J, E, P, S>(
    state: &mut SleepState,
    backend: &mut TensorConsolidationBackend<U, R, J, E, P>,
    progress: &mut S,
) -> Result<bool>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
    S: SleepProgressSink,
{
    run_consolidation_with_progress(state, backend, progress)
}

/// Fully durable variant of [`execute_tensor_consolidation`]. Before each
/// SleepState transition is published, this seals the exact teacher/student
/// weights and receiver optimizer, records its generation/hash in the pending
/// transaction, and only then invokes the caller's checkpoint sink.
pub fn execute_tensor_consolidation_durable<U, R, J, E, P, S>(
    state: &mut SleepState,
    backend: &mut TensorConsolidationBackend<U, R, J, E, P>,
    store: &TensorTransactionStore,
    progress: &mut S,
) -> Result<bool>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
    S: SleepProgressSink,
{
    progress.persist(state)?;
    let txn = state
        .pending
        .clone()
        .context("no consolidation transaction")?;
    let result = (|| loop {
        match state.phase {
            crate::sleep::SleepPhase::ProspectiveUpdate => {
                backend.compute_prospective_update(&txn)?;
                backend.stage_student(&txn)?;
                state.transition(crate::sleep::SleepPhase::KnowledgeSeeding)?;
                persist_tensor_boundary(state, backend, store, progress)?;
            }
            crate::sleep::SleepPhase::KnowledgeSeeding => {
                state.reserve_knowledge_rng(0, backend.config.knowledge.rollout_count_u64()?)?;
                progress.persist(state)?;
                let current_txn = state.pending.clone().expect("transaction checked above");
                backend.knowledge_seed(&current_txn)?;
                state.transition(crate::sleep::SleepPhase::Imitation)?;
                persist_tensor_boundary(state, backend, store, progress)?;
            }
            crate::sleep::SleepPhase::Imitation => {
                state.reserve_imitation_rng(0, backend.config.imitation.group_size_u64()?)?;
                progress.persist(state)?;
                let current_txn = state.pending.clone().expect("transaction checked above");
                backend.learn_to_imitate(&current_txn)?;
                state.transition(crate::sleep::SleepPhase::RetentionValidation)?;
                persist_tensor_boundary(state, backend, store, progress)?;
            }
            crate::sleep::SleepPhase::RetentionValidation => {
                if !backend.retention_passes(&txn)? {
                    break Ok(false);
                }
                state.transition(crate::sleep::SleepPhase::Commit)?;
                persist_tensor_boundary(state, backend, store, progress)?;
            }
            crate::sleep::SleepPhase::Commit => {
                if !state
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.committed)
                {
                    let candidate = backend.commit(&txn)?;
                    state.record_committed_candidate(candidate.checkpoint, candidate.sha256)?;
                    state.commit_consolidation()?;
                    persist_tensor_boundary(state, backend, store, progress)?;
                }
                break Ok(true);
            }
            crate::sleep::SleepPhase::DreamGeneration
            | crate::sleep::SleepPhase::DreamRanking
            | crate::sleep::SleepPhase::DreamTrials
            | crate::sleep::SleepPhase::DreamPolicyUpdate
            | crate::sleep::SleepPhase::Candidate => break Ok(true),
            crate::sleep::SleepPhase::Wake => bail!("consolidation is in wake phase"),
        }
    })();

    match result {
        Ok(true) => Ok(true),
        Ok(false) => {
            backend.restore_teacher(&txn)?;
            let mut restored = state.clone();
            restored.rollback()?;
            progress.persist(&restored)?;
            *state = restored;
            Ok(false)
        }
        Err(error) => {
            if state
                .pending
                .as_ref()
                .is_some_and(|pending| pending.committed)
            {
                return Err(error.context(
                    "tensor consolidation committed but state publication failed; retry persistence",
                ));
            }
            if let Err(restore_error) = backend.restore_teacher(&txn) {
                bail!(
                    "tensor consolidation failed: {error:#}; teacher restore failed: {restore_error:#}"
                );
            }
            let mut restored = state.clone();
            restored.rollback()?;
            progress
                .persist(&restored)
                .context("persisting tensor consolidation rollback")?;
            *state = restored;
            Err(error)
        }
    }
}

fn persist_tensor_boundary<U, R, J, E, P, S>(
    state: &mut SleepState,
    backend: &TensorConsolidationBackend<U, R, J, E, P>,
    store: &TensorTransactionStore,
    progress: &mut S,
) -> Result<()>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
    S: SleepProgressSink,
{
    let txn = state.pending.as_ref().context("no tensor transaction")?;
    let pointer = store.publish(txn, &backend.snapshot_inflight(txn)?)?;
    state.record_tensor_transaction(pointer.generation, pointer.manifest_sha256)?;
    progress.persist(state)
}

pub fn forward_kl_value(
    teacher: &Transformer,
    student: &Transformer,
    batch: &TokenRolloutBatch,
    device: &Device,
    chunk_tokens: usize,
    temperature: f32,
) -> Result<f32> {
    Ok(
        forward_kl_tensor(teacher, student, batch, device, chunk_tokens, temperature)?
            .1
            .forward_kl,
    )
}

#[derive(Clone, Copy, Debug)]
struct ForwardKlStats {
    forward_kl: f32,
    teacher_entropy: f32,
    student_entropy: f32,
    tokens: u64,
}

fn forward_kl_tensor(
    teacher: &Transformer,
    student: &Transformer,
    batch: &TokenRolloutBatch,
    device: &Device,
    chunk_tokens: usize,
    temperature: f32,
) -> Result<(Tensor<1>, ForwardKlStats)> {
    ensure!(chunk_tokens > 0, "KL chunk size must be positive");
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "KL temperature must be positive"
    );
    batch.validate_for(teacher)?;
    batch.validate_for(student)?;
    ensure!(
        teacher.config().vocab_size == student.config().vocab_size,
        "teacher/student vocabularies differ"
    );
    let count = batch.batch * batch.sequence;
    let positions = Tensor::<1, Int>::from_data(
        TensorData::new((0..count).map(|value| value as i64).collect(), [count]),
        device,
    );
    let input = batch.tensor(device);
    let teacher_projector = teacher.prepare_selected_logits(input.clone(), positions.clone());
    let student_projector = student.prepare_selected_logits(input, positions);
    let mut total: Option<Tensor<1>> = None;
    let mut teacher_entropy: Option<Tensor<1>> = None;
    let mut student_entropy: Option<Tensor<1>> = None;
    for start in (0..count).step_by(chunk_tokens) {
        let end = (start + chunk_tokens).min(count);
        let chunk_weight = (end - start) as f32 / count as f32;
        let teacher_logits = teacher_projector.logits(start..end).div_scalar(temperature);
        let teacher_log = log_softmax(teacher_logits.clone(), 1).detach();
        let teacher_probability = softmax(teacher_logits, 1).detach();
        let student_logits = student_projector.logits(start..end).div_scalar(temperature);
        let student_log = log_softmax(student_logits.clone(), 1);
        let student_probability = softmax(student_logits, 1).detach();
        let teacher_entropy_chunk = (teacher_probability.clone() * teacher_log.clone())
            .sum_dim(1)
            .mean()
            .neg()
            .mul_scalar(chunk_weight);
        let student_entropy_chunk = (student_probability * student_log.clone().detach())
            .sum_dim(1)
            .mean()
            .neg()
            .mul_scalar(chunk_weight);
        teacher_entropy = Some(match teacher_entropy {
            Some(sum) => sum + teacher_entropy_chunk,
            None => teacher_entropy_chunk,
        });
        student_entropy = Some(match student_entropy {
            Some(sum) => sum + student_entropy_chunk,
            None => student_entropy_chunk,
        });
        let loss = (teacher_probability * (teacher_log - student_log))
            .sum_dim(1)
            .mean()
            .mul_scalar(temperature * temperature * chunk_weight);
        total = Some(match total {
            Some(sum) => sum + loss,
            None => loss,
        });
    }
    let total = total.context("empty KL rollout")?;
    // Keep chunk metrics on the accelerator and synchronize exactly once.
    // Reading two entropy scalars inside every chunk otherwise serializes the
    // projection loop and creates visible GPU idle gaps during consolidation.
    let metrics = Tensor::cat(
        vec![
            total.clone().detach(),
            teacher_entropy.context("empty teacher entropy")?,
            student_entropy.context("empty student entropy")?,
        ],
        0,
    )
    .into_data()
    .convert::<f32>()
    .to_vec::<f32>()
    .context("reading KL and entropy metrics")?;
    ensure!(metrics.len() == 3, "KL metric tensor has invalid shape");
    let value = metrics[0];
    let teacher_entropy = metrics[1];
    let student_entropy = metrics[2];
    ensure!(
        value.is_finite() && value >= -1e-5,
        "invalid forward KL {value}"
    );
    ensure!(
        teacher_entropy.is_finite()
            && teacher_entropy >= 0.0
            && student_entropy.is_finite()
            && student_entropy >= 0.0,
        "non-finite or negative distillation entropy metric"
    );
    Ok((
        total,
        ForwardKlStats {
            forward_kl: value.max(0.0),
            teacher_entropy,
            student_entropy,
            tokens: count as u64,
        },
    ))
}

fn equal_length_continuation_log_probabilities(
    model: &Transformer,
    prefix: &[i64],
    candidates: &[Vec<i64>],
    candidate_indices: &[usize],
    device: &Device,
) -> Result<Tensor<2>> {
    ensure!(
        !prefix.is_empty() && !candidate_indices.is_empty(),
        "imitation prefix/candidate bucket is empty"
    );
    let continuation_len = candidates[candidate_indices[0]].len();
    ensure!(continuation_len > 0, "imitation continuation is empty");
    ensure!(
        candidate_indices
            .iter()
            .all(|index| candidates[*index].len() == continuation_len),
        "imitation length bucket contains mixed continuation lengths"
    );
    let input_len = prefix
        .len()
        .checked_add(continuation_len)
        .and_then(|length| length.checked_sub(1))
        .context("imitation sequence length overflow")?;
    ensure!(
        input_len <= model.config().max_seq_len,
        "imitation sequence exceeds model limit"
    );
    ensure!(
        prefix
            .iter()
            .chain(
                candidate_indices
                    .iter()
                    .flat_map(|index| candidates[*index].iter()),
            )
            .all(|id| usize::try_from(*id).is_ok_and(|id| id < model.config().vocab_size)),
        "imitation token is outside vocabulary"
    );

    let input_capacity = candidate_indices
        .len()
        .checked_mul(input_len)
        .context("imitation input size overflow")?;
    let output_capacity = candidate_indices
        .len()
        .checked_mul(continuation_len)
        .context("imitation target size overflow")?;
    let mut input_ids = Vec::with_capacity(input_capacity);
    let mut positions = Vec::with_capacity(output_capacity);
    let mut targets = Vec::with_capacity(output_capacity);
    for (row, candidate_index) in candidate_indices.iter().copied().enumerate() {
        let candidate = &candidates[candidate_index];
        input_ids.extend_from_slice(prefix);
        input_ids.extend_from_slice(&candidate[..continuation_len - 1]);
        let row_start = row
            .checked_mul(input_len)
            .and_then(|offset| offset.checked_add(prefix.len() - 1))
            .context("imitation position overflow")?;
        for position in row_start..row_start + continuation_len {
            positions.push(
                i64::try_from(position).context("imitation position exceeds tensor index range")?,
            );
        }
        targets.extend_from_slice(candidate);
    }
    ensure!(
        input_ids.len() == input_capacity
            && positions.len() == output_capacity
            && targets.len() == output_capacity,
        "imitation batch construction produced an invalid shape"
    );
    let input = Tensor::<2, Int>::from_data(
        TensorData::new(input_ids, [candidate_indices.len(), input_len]),
        device,
    );
    let positions =
        Tensor::<1, Int>::from_data(TensorData::new(positions, [output_capacity]), device);
    let targets = Tensor::<1, Int>::from_data(TensorData::new(targets, [output_capacity]), device);
    Ok(
        log_softmax(model.forward_selected_logits(input, positions), 1)
            .gather(1, targets.unsqueeze_dim(1))
            .reshape([candidate_indices.len(), continuation_len]),
    )
}

fn candidate_length_buckets(
    candidates: &[Vec<i64>],
    max_tokens: usize,
) -> Result<BTreeMap<usize, Vec<usize>>> {
    let mut buckets = BTreeMap::<usize, Vec<usize>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        ensure!(!candidate.is_empty(), "GRPO candidate is empty");
        ensure!(
            candidate.len() <= max_tokens,
            "GRPO candidate exceeds padded token width"
        );
        buckets.entry(candidate.len()).or_default().push(index);
    }
    Ok(buckets)
}

fn grpo_rows_per_forward(prefix_len: usize, continuation_len: usize) -> Result<usize> {
    let input_len = prefix_len
        .checked_add(continuation_len)
        .and_then(|length| length.checked_sub(1))
        .context("imitation sequence length overflow")?;
    ensure!(input_len > 0, "imitation model input is empty");
    Ok((MAX_TENSOR_FORWARD_TOKENS / input_len).max(1))
}

fn padded_group_log_probabilities(
    model: &Transformer,
    prefix: &[i64],
    candidates: &[Vec<i64>],
    max_tokens: usize,
    device: &Device,
) -> Result<Tensor<2>> {
    ensure!(!candidates.is_empty(), "GRPO candidate group is empty");
    ensure!(max_tokens > 0, "GRPO candidates have no tokens");
    let buckets = candidate_length_buckets(candidates, max_tokens)?;
    let mut rows = (0..candidates.len())
        .map(|_| None)
        .collect::<Vec<Option<Tensor<2>>>>();
    for (continuation_len, candidate_indices) in buckets {
        let rows_per_forward = grpo_rows_per_forward(prefix.len(), continuation_len)?;
        for candidate_chunk in candidate_indices.chunks(rows_per_forward) {
            let bucket = equal_length_continuation_log_probabilities(
                model,
                prefix,
                candidates,
                candidate_chunk,
                device,
            )?;
            for (bucket_row, candidate_index) in candidate_chunk.iter().copied().enumerate() {
                let row = bucket
                    .clone()
                    .slice([bucket_row..bucket_row + 1, 0..continuation_len]);
                let padded = if continuation_len < max_tokens {
                    Tensor::cat(
                        vec![
                            row,
                            Tensor::<2>::zeros([1, max_tokens - continuation_len], device),
                        ],
                        1,
                    )
                } else {
                    row
                };
                rows[candidate_index] = Some(padded);
            }
        }
    }
    let rows = rows
        .into_iter()
        .map(|row| row.context("GRPO candidate row was not constructed"))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::cat(rows, 0))
}

fn validate_group(group: &ImitationGroup, model: &Transformer, size: usize) -> Result<()> {
    ensure!(
        !group.prefix.is_empty() && !group.teacher_continuation.is_empty(),
        "imitation reference is empty"
    );
    ensure!(
        group.candidates.len() == size,
        "GRPO group has {} candidates, expected {size}",
        group.candidates.len()
    );
    let valid_token = |token: &i64| {
        *token >= 0 && usize::try_from(*token).is_ok_and(|id| id < model.config().vocab_size)
    };
    ensure!(
        group.prefix.iter().all(valid_token)
            && group.teacher_continuation.iter().all(valid_token)
            && group.candidates.iter().flatten().all(valid_token),
        "imitation group contains an out-of-vocabulary token"
    );
    let teacher_sequence = group
        .prefix
        .len()
        .checked_add(group.teacher_continuation.len())
        .and_then(|length| length.checked_sub(1))
        .context("imitation teacher sequence length overflow")?;
    ensure!(
        teacher_sequence <= model.config().max_seq_len,
        "imitation teacher sequence exceeds model limit"
    );
    for candidate in &group.candidates {
        ensure!(!candidate.is_empty(), "imitation candidate is empty");
        let sequence = group
            .prefix
            .len()
            .checked_add(candidate.len())
            .and_then(|length| length.checked_sub(1))
            .context("imitation candidate sequence length overflow")?;
        ensure!(
            sequence <= model.config().max_seq_len,
            "imitation candidate sequence exceeds model limit"
        );
    }
    Ok(())
}

fn validate_imitation_groups(
    groups: &[ImitationGroup],
    model: &Transformer,
    group_size: usize,
    maximum_edit_distance: usize,
) -> Result<()> {
    ensure!(
        groups.len() <= MAX_TENSOR_IMITATION_GROUPS,
        "imitation source returned more than {MAX_TENSOR_IMITATION_GROUPS} groups"
    );
    let mut model_token_evaluations = 0_usize;
    let mut edit_distance_cells = 0_usize;
    for group in groups {
        validate_group(group, model, group_size)?;
        let max_candidate_tokens = group
            .candidates
            .iter()
            .map(Vec::len)
            .max()
            .context("imitation group has no candidates")?;
        let padded_tokens = max_candidate_tokens
            .checked_mul(group.candidates.len())
            .context("imitation padded tensor size overflow")?;
        ensure!(
            padded_tokens <= MAX_TENSOR_GRPO_PADDED_TOKENS,
            "imitation group exceeds the {MAX_TENSOR_GRPO_PADDED_TOKENS}-element padded tensor limit"
        );
        let mut group_tokens = 0_usize;
        for candidate in &group.candidates {
            let sequence = group
                .prefix
                .len()
                .checked_add(candidate.len())
                .and_then(|length| length.checked_sub(1))
                .context("imitation candidate sequence length overflow")?;
            group_tokens = group_tokens
                .checked_add(sequence)
                .context("imitation group model-token work overflow")?;
            ensure!(
                group_tokens <= MAX_TENSOR_FORWARD_TOKENS,
                "imitation group exceeds the {MAX_TENSOR_FORWARD_TOKENS}-token activation limit"
            );
            // Current policy, frozen behavior policy, and teacher reference
            // each evaluate the same causal rows.
            model_token_evaluations = model_token_evaluations
                .checked_add(
                    sequence
                        .checked_mul(3)
                        .context("imitation model-token work overflow")?,
                )
                .context("imitation model-token work overflow")?;
            ensure!(
                model_token_evaluations <= MAX_TENSOR_SUBPHASE_TOKENS,
                "imitation exceeds the {MAX_TENSOR_SUBPHASE_TOKENS}-token model-work limit"
            );

            if group.teacher_continuation.len().abs_diff(candidate.len()) <= maximum_edit_distance {
                let longest = group.teacher_continuation.len().max(candidate.len());
                let band = maximum_edit_distance.min(longest);
                let row_width = band
                    .checked_mul(2)
                    .and_then(|width| width.checked_add(1))
                    .context("imitation edit-distance work overflow")?;
                edit_distance_cells = edit_distance_cells
                    .checked_add(
                        longest
                            .checked_mul(row_width)
                            .context("imitation edit-distance work overflow")?,
                    )
                    .context("imitation edit-distance work overflow")?;
                ensure!(
                    edit_distance_cells <= MAX_TENSOR_EDIT_DISTANCE_CELLS,
                    "imitation exceeds the {MAX_TENSOR_EDIT_DISTANCE_CELLS}-cell edit-distance limit"
                );
            }
        }
    }
    Ok(())
}

pub fn thresholded_edit_reward(reference: &[i64], candidate: &[i64], maximum: usize) -> f32 {
    let Some(distance) = thresholded_levenshtein(reference, candidate, maximum) else {
        return 0.0;
    };
    1.0 - distance as f32 / reference.len().max(candidate.len()).max(1) as f32
}

/// Return the exact edit distance when it is at most `maximum`.
///
/// Imitation only assigns a reward inside that threshold, so visiting the
/// rest of the dynamic-programming matrix wastes quadratic CPU time between
/// accelerator updates. The diagonal band is exact for every accepted result
/// and reduces the usual `O(left * right)` work to
/// `O(max(left, right) * maximum)` for the normal small-threshold recipe.
fn thresholded_levenshtein(left: &[i64], right: &[i64], maximum: usize) -> Option<usize> {
    if left.len().abs_diff(right.len()) > maximum {
        return None;
    }
    if left.is_empty() {
        return (right.len() <= maximum).then_some(right.len());
    }
    if right.is_empty() {
        return (left.len() <= maximum).then_some(left.len());
    }

    // Distance is symmetric. Keeping the shorter sequence on the horizontal
    // axis bounds scratch memory without changing the diagonal-band proof.
    let (left, right) = if right.len() <= left.len() {
        (left, right)
    } else {
        (right, left)
    };
    let band = maximum.min(left.len().max(right.len()));
    let beyond = band + 1;
    let mut previous = vec![beyond; right.len() + 1];
    for (column, value) in previous.iter_mut().enumerate().take(band + 1) {
        *value = column;
    }
    let mut current = vec![beyond; right.len() + 1];
    for (left_index, left_token) in left.iter().enumerate() {
        let row = left_index + 1;
        let first_column = row.saturating_sub(band).max(1);
        let last_column = row.saturating_add(band).min(right.len());
        if first_column > last_column {
            return None;
        }
        // Only the diagonal band is read on the next row. Reset its two
        // sentinels rather than filling the whole sequence-width buffer; a
        // full fill here silently turns the advertised banded algorithm back
        // into O(left * right) CPU work.
        current[0] = if row <= band { row } else { beyond };
        if first_column > 1 {
            current[first_column - 1] = beyond;
        }
        for column in first_column..=last_column {
            current[column] = (previous[column - 1]
                + usize::from(left_token != &right[column - 1]))
            .min(previous[column].saturating_add(1))
            .min(current[column - 1].saturating_add(1));
        }
        if last_column < right.len() {
            current[last_column + 1] = beyond;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= maximum).then_some(previous[right.len()])
}

pub fn normalized_group_advantages(rewards: &[f32]) -> Vec<f32> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
    let variance = rewards
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / rewards.len() as f32;
    let scale = (variance + 1e-6).sqrt();
    rewards.iter().map(|value| (value - mean) / scale).collect()
}

/// A deterministic parameter/state digest used to verify that a publisher
/// attests to the same model it was asked to persist. Parameter IDs are
/// intentionally excluded because an isolated fork may assign fresh IDs.
pub(crate) fn model_parameter_hash(model: &Transformer) -> Result<String> {
    struct DigestVisitor {
        hash: Sha256,
        failure: Option<String>,
        tensors: usize,
    }
    impl DigestVisitor {
        fn shape<const D: usize>(&mut self, tag: u8, shape: [usize; D]) {
            self.hash.update([tag]);
            self.hash.update((D as u64).to_le_bytes());
            for dim in shape {
                self.hash.update((dim as u64).to_le_bytes());
            }
            self.tensors += 1;
        }
    }
    impl ModuleVisitor for DigestVisitor {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            self.shape(b'f', parameter.shape().dims::<D>());
            match parameter.val().into_data().convert::<f32>().to_vec::<f32>() {
                Ok(values) => {
                    let mut bytes =
                        Vec::with_capacity(FINGERPRINT_CHUNK_VALUES * std::mem::size_of::<f32>());
                    for chunk in values.chunks(FINGERPRINT_CHUNK_VALUES) {
                        bytes.clear();
                        bytes.extend(chunk.iter().flat_map(|value| value.to_bits().to_le_bytes()));
                        self.hash.update(&bytes);
                    }
                }
                Err(error) => self.failure = Some(error.to_string()),
            }
        }

        fn visit_int<const D: usize>(&mut self, parameter: &Param<Tensor<D, Int>>) {
            self.shape(b'i', parameter.shape().dims::<D>());
            match parameter.val().into_data().to_vec::<i64>() {
                Ok(values) => {
                    let mut bytes =
                        Vec::with_capacity(FINGERPRINT_CHUNK_VALUES * std::mem::size_of::<i64>());
                    for chunk in values.chunks(FINGERPRINT_CHUNK_VALUES) {
                        bytes.clear();
                        bytes.extend(chunk.iter().flat_map(|value| value.to_le_bytes()));
                        self.hash.update(&bytes);
                    }
                }
                Err(error) => self.failure = Some(error.to_string()),
            }
        }

        fn visit_bool<const D: usize>(&mut self, parameter: &Param<Tensor<D, Bool>>) {
            self.shape(b'b', parameter.shape().dims::<D>());
            match parameter.val().into_data().to_vec::<bool>() {
                Ok(values) => {
                    let mut bytes = Vec::with_capacity(FINGERPRINT_CHUNK_VALUES);
                    for chunk in values.chunks(FINGERPRINT_CHUNK_VALUES) {
                        bytes.clear();
                        bytes.extend(chunk.iter().copied().map(u8::from));
                        self.hash.update(&bytes);
                    }
                }
                Err(error) => self.failure = Some(error.to_string()),
            }
        }
    }

    let mut visitor = DigestVisitor {
        hash: Sha256::new(),
        failure: None,
        tensors: 0,
    };
    visitor.hash.update(b"hermes-tensor-sleep-parameters-v1\0");
    model.visit(&mut visitor);
    ensure!(
        visitor.tensors > 0,
        "Transformer exposes no checkpoint tensors"
    );
    if let Some(error) = visitor.failure {
        bail!("reading Transformer parameters for candidate verification: {error}");
    }
    Ok(format!("sha256:{:x}", visitor.hash.finalize()))
}

/// Operations allowed to mutate an owned trial model or generation-policy
/// state. They never receive mutable access to the shared candidate.
pub trait TransformerDreamOps {
    /// Generation must be deterministic/idempotent for the transaction and
    /// the persisted RNG reservation in `txn.dream_generation_rng`.
    fn generate(
        &mut self,
        txn: &ConsolidationTxn,
        model: &Transformer,
        candidate_count: usize,
        routing: MemoryRouting,
    ) -> Result<(String, Vec<GeneratedDream>)>;
    fn load(&mut self, txn: &ConsolidationTxn, manifest: &str) -> Result<Vec<GeneratedDream>>;
    fn reference_gradient(
        &mut self,
        txn: &ConsolidationTxn,
        model: &Transformer,
        reference_hash: &str,
    ) -> Result<Vec<f32>>;
    /// Must be deterministic/idempotent for `(txn.id, candidate.id)` and its
    /// persisted trial RNG reservation. The supplied model is an isolated fork.
    fn isolated_lora_trial(
        &mut self,
        txn: &ConsolidationTxn,
        isolated_model: Transformer,
        candidate: &GeneratedDream,
        rank: usize,
        alpha: usize,
    ) -> Result<DreamTrial>;
    /// Must be deterministic/idempotent for the transaction, accepted-trial
    /// receipts, iteration count, and persisted Dreaming RNG reservations.
    fn restem_update_policy(
        &mut self,
        txn: &ConsolidationTxn,
        model: &Transformer,
        accepted: &[DreamTrial],
        iterations: usize,
    ) -> Result<String>;
}

/// Structurally isolated Dreaming backend. Each LoRA trial owns a fork with
/// independent parameter IDs/storage; rejected adapters cannot touch shared.
pub struct TensorDreamBackend<O> {
    shared: Transformer,
    immutable_shared: Transformer,
    immutable_shared_hash: String,
    bound_candidate_checkpoint: Option<(String, String)>,
    device: Device,
    operations: O,
}

impl<O: TransformerDreamOps> TensorDreamBackend<O> {
    pub fn new(
        shared: Transformer,
        device: Device,
        probe: TokenRolloutBatch,
        operations: O,
    ) -> Result<Self> {
        probe.validate_for(&shared)?;
        let immutable_shared_hash = model_parameter_hash(&shared)?;
        Ok(Self {
            immutable_shared: shared.clone(),
            shared,
            immutable_shared_hash,
            bound_candidate_checkpoint: None,
            device,
            operations,
        })
    }
    /// Bind the in-memory immutable model to the artifact identity recorded by
    /// the committed consolidation. This is separate from the parameter hash:
    /// checkpoint formats may authenticate metadata in addition to tensors.
    pub fn bind_candidate_checkpoint(&mut self, uri: &str, sha256: &str) -> Result<()> {
        ensure!(
            !uri.trim().is_empty(),
            "dream candidate checkpoint URI is empty"
        );
        validate_sha256_identity(sha256, "published checkpoint identity")?;
        let identity = (uri.to_owned(), sha256.to_owned());
        if let Some(bound) = &self.bound_candidate_checkpoint {
            ensure!(
                bound == &identity,
                "dream backend cannot be rebound to another candidate checkpoint"
            );
        } else {
            self.bound_candidate_checkpoint = Some(identity);
        }
        Ok(())
    }
    pub fn shared_model(&self) -> &Transformer {
        &self.shared
    }
    pub fn operations(&self) -> &O {
        &self.operations
    }
    fn fingerprint(&self) -> Result<String> {
        // The shared model is never exposed mutably. Generation, reference
        // evaluation, and policy updates receive `&Transformer`; LoRA trials
        // receive an independently forked model. Cache the full parameter
        // identity once instead of copying and hashing hundreds of millions
        // of parameters around every trial.
        Ok(self.immutable_shared_hash.clone())
    }
}

impl<O: TransformerDreamOps> DreamingBackend for TensorDreamBackend<O> {
    fn verify_committed_candidate(&mut self, txn: &ConsolidationTxn) -> Result<()> {
        let checkpoint = txn
            .candidate_checkpoint
            .as_ref()
            .context("committed transaction has no candidate checkpoint")?;
        let sha256 = txn
            .candidate_hash
            .as_ref()
            .context("committed transaction has no candidate hash")?;
        ensure!(
            self.bound_candidate_checkpoint.as_ref() == Some(&(checkpoint.clone(), sha256.clone())),
            "dream backend is not bound to the committed candidate checkpoint"
        );
        Ok(())
    }

    fn shared_checkpoint_hash(&mut self) -> Result<String> {
        self.fingerprint()
    }

    fn generate_from_wake_contexts(
        &mut self,
        txn: &ConsolidationTxn,
        count: usize,
        random_extra_expert: bool,
    ) -> Result<(String, Vec<GeneratedDream>)> {
        ensure!(
            random_extra_expert,
            "dream generation requires random extra expert"
        );
        let (manifest, dreams) = self.operations.generate(
            txn,
            &self.shared,
            count,
            MemoryRouting::Dream { seed: txn.id },
        )?;
        validate_sha256_identity(&manifest, "dream manifest identity")?;
        for dream in &dreams {
            validate_sha256_identity(&dream.artifact_hash, "dream artifact identity")?;
        }
        Ok((manifest, dreams))
    }

    fn load_generated_dreams(
        &mut self,
        txn: &ConsolidationTxn,
        manifest: &str,
    ) -> Result<Vec<GeneratedDream>> {
        validate_sha256_identity(manifest, "dream manifest identity")?;
        let dreams = self.operations.load(txn, manifest)?;
        for dream in &dreams {
            validate_sha256_identity(&dream.artifact_hash, "dream artifact identity")?;
        }
        Ok(dreams)
    }

    fn reference_gradient(
        &mut self,
        txn: &ConsolidationTxn,
        reference_hash: &str,
    ) -> Result<Vec<f32>> {
        validate_sha256_identity(reference_hash, "dream reference identity")?;
        self.operations
            .reference_gradient(txn, &self.shared, reference_hash)
    }

    fn isolated_lora_trial(
        &mut self,
        txn: &ConsolidationTxn,
        candidate: &GeneratedDream,
        rank: usize,
        alpha: usize,
    ) -> Result<DreamTrial> {
        let isolated = self.shared.clone().fork(&self.device);
        validate_sha256_identity(&candidate.artifact_hash, "dream candidate identity")?;
        let trial = self
            .operations
            .isolated_lora_trial(txn, isolated, candidate, rank, alpha)?;
        validate_sha256_identity(&trial.adapter_hash, "dream adapter identity")?;
        validate_sha256_identity(&trial.evaluator_hash, "dream evaluator identity")?;
        Ok(trial)
    }

    fn restem_update(
        &mut self,
        txn: &ConsolidationTxn,
        accepted: &[DreamTrial],
        iterations: usize,
    ) -> Result<String> {
        self.operations
            .restem_update_policy(txn, &self.shared, accepted, iterations)
    }

    fn restore_shared_candidate(&mut self, _: &ConsolidationTxn) -> Result<()> {
        self.shared = self.immutable_shared.clone();
        Ok(())
    }
}

#[cfg(test)]
fn model_probe_hash(
    model: &Transformer,
    probe: &TokenRolloutBatch,
    device: &Device,
) -> Result<String> {
    let values = model
        .forward(probe.tensor(device), 0)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .context("reading probe logits")?;
    let mut hash = Sha256::new();
    hash.update(b"hermes-tensor-sleep-probe-v1\0");
    for token in &probe.token_ids {
        hash.update(token.to_le_bytes());
    }
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("sha256:{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use burn::tensor::Tensor;
    use hermes_llm::{ModelDef, parse_mal};

    use super::*;
    use crate::sleep::{
        DreamingConfig, MemoryTierSchedule, SleepSchedule, TerminalConsolidation, UpdateClock,
        run_dreaming,
    };

    fn hash(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    #[test]
    fn checkpoint_json_preserves_diagnostic_float_bits() {
        // Without serde_json's `float_roundtrip` parser this exact value is
        // serialized from ...0000 and reloaded as adjacent ...0001.
        let diagnostics = TensorSleepDiagnostics {
            knowledge_kl_sum: f64::from_bits(0x3f86_95b6_0000_0000),
            ..TensorSleepDiagnostics::default()
        };
        let bytes = serde_json::to_vec(&diagnostics).unwrap();
        let restored: TensorSleepDiagnostics = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            restored.knowledge_kl_sum.to_bits(),
            diagnostics.knowledge_kl_sum.to_bits()
        );
        assert_eq!(bytes, serde_json::to_vec(&restored).unwrap());
    }

    fn tiny_config() -> ModelDef {
        let mut config = parse_mal(r#"
            ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
            memory cms {
                tier fast { ffn: base reserve_experts { capacity: 1 rank: 3 top_k: 1 } }
                tier slow { ffn: base residual_init: zero reserve_experts { capacity: 2 rank: 3 top_k: 1 } }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 dropout: 0.0 position_encoding: none } memory: cms dropout: 0.0 }
            }
        "#).unwrap();
        config.embeddings.dropout = 0.0;
        config
    }

    fn sender_update(teacher: &Transformer, device: &Device, sender: usize) -> Transformer {
        let model = teacher.clone();
        let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], device);
        let target = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], device);
        let mut gradients = model.forward_loss(input, target).backward();
        let mut eligible = model
            .memory_tier_base_parameter_ids_all_layers(sender)
            .unwrap();
        eligible.extend(
            model
                .memory_slot_statuses()
                .into_iter()
                .filter(|status| status.tier == sender && status.active)
                .flat_map(|status| status.parameter_ids),
        );
        let selected = GradientsParams::from_params(&mut gradients, &model, &eligible);
        AdamWConfig::new()
            .with_weight_decay(0.0)
            .init()
            .step(1e-2.into(), model, selected)
    }

    fn sleep_schedule() -> SleepSchedule {
        SleepSchedule {
            clock: UpdateClock::OptimizerSteps,
            terminal_consolidation: TerminalConsolidation::DistillIntoBaseV1,
            tiers: vec![
                MemoryTierSchedule {
                    id: "fast".into(),
                    update_period: 1,
                    reserve_slots: 1,
                },
                MemoryTierSchedule {
                    id: "slow".into(),
                    update_period: 2,
                    reserve_slots: 2,
                },
            ],
        }
    }

    fn begin(model: &mut Transformer, device: &Device) -> (SleepState, ConsolidationTxn) {
        model.activate_memory_slot_all_layers(0, 0).unwrap();
        let schedule = sleep_schedule();
        let mut state = SleepState::new(&schedule, 2).unwrap();
        state.tiers[0].slots[0].active = true;
        state.tiers[0].slots[0].generation = 1;
        state.advance_clock(&schedule, 1).unwrap();
        let update_hash =
            prospective_update_hash(model, &sender_update(model, device, 0), 0).unwrap();
        let txn = state
            .begin(
                0,
                "teacher.bpk".into(),
                hash('a'),
                "student.bpk".into(),
                hash('b'),
                update_hash,
            )
            .unwrap();
        (state, txn)
    }

    struct Update {
        device: Device,
        cleared: usize,
        fail_restore: bool,
    }
    impl ProspectiveTransformerUpdate for Update {
        fn snapshot_state(&mut self, _: &ConsolidationTxn) -> Result<ProspectiveUpdateSnapshot> {
            ProspectiveUpdateSnapshot::new(self.cleared.to_le_bytes().to_vec())
        }

        fn restore_state(
            &mut self,
            _: &ConsolidationTxn,
            snapshot: &ProspectiveUpdateSnapshot,
        ) -> Result<()> {
            ensure!(!self.fail_restore, "injected update-state restore failure");
            let bytes: [u8; std::mem::size_of::<usize>()] = snapshot
                .as_bytes()
                .try_into()
                .context("invalid test update snapshot")?;
            self.cleared = usize::from_le_bytes(bytes);
            Ok(())
        }

        fn stage(
            &mut self,
            txn: &ConsolidationTxn,
            teacher: &Transformer,
        ) -> Result<ProspectiveTransformerCandidate> {
            self.cleared += 1;
            let model = sender_update(teacher, &self.device, txn.sender);
            let update_sha256 = prospective_update_hash(teacher, &model, txn.sender)?;
            Ok(ProspectiveTransformerCandidate {
                checkpoint: ImmutableTransformerCheckpoint {
                    uri: txn.student_checkpoint.clone(),
                    sha256: txn.student_hash.clone(),
                    model,
                },
                update_sha256,
            })
        }
        fn clear_reclaimed_optimizer_state(
            &mut self,
            _: &ConsolidationTxn,
            ids: &[ParamId],
        ) -> Result<()> {
            self.cleared += ids.len();
            Ok(())
        }
    }

    struct Rollouts(TokenRolloutBatch);
    impl ConsolidationRollouts for Rollouts {
        fn knowledge_rollouts(
            &mut self,
            _: &ConsolidationTxn,
            _: RolloutOwner,
            _: &Transformer,
            count: usize,
        ) -> Result<Vec<TokenRolloutBatch>> {
            Ok(vec![self.0.clone(); count])
        }
        fn imitation_groups(
            &mut self,
            _: &ConsolidationTxn,
            _: &Transformer,
            _: &Transformer,
            size: usize,
        ) -> Result<Vec<ImitationGroup>> {
            let mut candidates = vec![vec![3, 4], vec![8, 9]];
            candidates.truncate(size);
            Ok(vec![ImitationGroup {
                prefix: vec![1, 2],
                teacher_continuation: vec![3, 4],
                candidates,
            }])
        }
    }

    struct FailingKnowledgeRollouts;

    impl ConsolidationRollouts for FailingKnowledgeRollouts {
        fn knowledge_rollouts(
            &mut self,
            _: &ConsolidationTxn,
            _: RolloutOwner,
            _: &Transformer,
            _: usize,
        ) -> Result<Vec<TokenRolloutBatch>> {
            bail!("injected knowledge-rollout failure")
        }

        fn imitation_groups(
            &mut self,
            _: &ConsolidationTxn,
            _: &Transformer,
            _: &Transformer,
            _: usize,
        ) -> Result<Vec<ImitationGroup>> {
            bail!("imitation must not run after the injected knowledge failure")
        }
    }

    struct Judge;
    impl SemanticJudge for Judge {
        fn artifact_hash(&self) -> &str {
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        }
        fn score(&mut self, _: &[i64], reference: &[i64], candidate: &[i64]) -> Result<f32> {
            Ok(if reference == candidate { 1.0 } else { 0.0 })
        }
    }

    struct Evaluator {
        batch: TokenRolloutBatch,
        reject: bool,
        calls: usize,
    }
    impl RetentionEvaluator for Evaluator {
        fn artifact_hash(&self) -> &str {
            "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        }
        fn suite_hash(&self) -> &str {
            "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        }
        fn anchor_rollouts(&mut self, _: &ConsolidationTxn) -> Result<Vec<TokenRolloutBatch>> {
            Ok(vec![self.batch.clone()])
        }
        fn score(&mut self, _: &ConsolidationTxn, _: &Transformer) -> Result<RetentionScores> {
            self.calls += 1;
            Ok(RetentionScores {
                stable_anchor: if self.reject && self.calls == 2 {
                    0.0
                } else {
                    1.0
                },
                incorporation: 1.0,
            })
        }
    }

    #[derive(Default)]
    struct Publisher {
        published: BTreeSet<u64>,
        restored: BTreeSet<u64>,
    }
    impl AtomicCandidatePublisher for Publisher {
        fn publish_candidate(
            &mut self,
            txn: &ConsolidationTxn,
            candidate: &Transformer,
        ) -> Result<ImmutableTransformerCheckpoint> {
            self.published.insert(txn.id);
            Ok(ImmutableTransformerCheckpoint {
                uri: format!("candidate-{}.bpk", txn.id),
                sha256: hash(match txn.id {
                    1 => 'f',
                    2 => '9',
                    _ => '8',
                }),
                model: candidate.clone(),
            })
        }
        fn restore_teacher(
            &mut self,
            txn: &ConsolidationTxn,
            _: &ImmutableTransformerCheckpoint,
        ) -> Result<()> {
            self.restored.insert(txn.id);
            Ok(())
        }
    }

    fn config() -> TensorConsolidationConfig {
        TensorConsolidationConfig {
            knowledge: KnowledgeSeedingConfig {
                chunk_tokens: 2,
                teacher_rollouts: 1,
                detached_student_rollouts: 1,
                temperature: 1.0,
                forward_kl_weight: 1.0,
            },
            imitation: ImitationConfig {
                semantic_judge_hash: hash('d'),
                semantic_weight: 0.5,
                maximum_edit_distance: 2,
                grpo_group_size: 2,
            },
            retention: RetentionGateConfig {
                evaluator_hash: hash('e'),
                suite_hash: hash('e'),
                max_anchor_forward_kl: 100.0,
                max_anchor_regression: 0.1,
                min_incorporation_gain: -1.0,
            },
            receiver_learning_rate: 1e-2,
            receiver_weight_decay: 0.0,
            grpo_clip_epsilon: 0.2,
            grpo_advantage_epsilon: 1e-6,
            grpo_kl_coefficient: 0.01,
        }
    }

    type Backend = TensorConsolidationBackend<Update, Rollouts, Judge, Evaluator, Publisher>;
    fn backend(model: Transformer, device: &Device, reject: bool) -> Backend {
        backend_from_checkpoint(
            ImmutableTransformerCheckpoint {
                uri: "teacher.bpk".into(),
                sha256: hash('a'),
                model,
            },
            device,
            reject,
        )
    }

    fn backend_from_checkpoint(
        live: ImmutableTransformerCheckpoint,
        device: &Device,
        reject: bool,
    ) -> Backend {
        let batch = TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4]).unwrap();
        TensorConsolidationBackend::new(
            live,
            device.clone(),
            config(),
            Update {
                device: device.clone(),
                cleared: 0,
                fail_restore: false,
            },
            Rollouts(batch.clone()),
            Judge,
            Evaluator {
                batch,
                reject,
                calls: 0,
            },
            Publisher::default(),
        )
        .unwrap()
    }

    struct Sink {
        snapshots: usize,
    }
    impl SleepProgressSink for Sink {
        fn persist(&mut self, _: &SleepState) -> Result<()> {
            self.snapshots += 1;
            Ok(())
        }
    }

    #[test]
    fn real_kl_grpo_receiver_update_commits_and_replays_idempotently() {
        let device = Device::ndarray().autodiff();
        device.seed(7);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (mut state, txn) = begin(&mut model, &device);
        let mut backend = backend(model, &device, false);
        let mut sink = Sink { snapshots: 0 };
        let directory = tempfile::tempdir().unwrap();
        let store = TensorTransactionStore::new(directory.path());
        assert!(
            execute_tensor_consolidation_durable(&mut state, &mut backend, &store, &mut sink,)
                .unwrap()
        );
        assert!(
            state
                .pending
                .as_ref()
                .unwrap()
                .tensor_transaction_generation
                .is_some()
        );
        let statuses = backend.live_checkpoint().model.memory_slot_statuses();
        assert!(
            statuses
                .iter()
                .filter(|s| s.tier == 0 && s.slot == 0)
                .all(|s| !s.active)
        );
        assert!(
            statuses
                .iter()
                .filter(|s| s.tier == 1 && s.slot == 0)
                .all(|s| s.active)
        );
        assert_eq!(backend.diagnostics().knowledge_updates, 2);
        assert_eq!(backend.diagnostics().imitation_updates, 1);
        assert!(backend.diagnostics().selected_gradient_tensors > 0);
        assert!(
            backend.diagnostics().selected_gradient_tensors
                <= backend.receiver_parameter_ids().len()
        );
        assert_eq!(backend.components().4.published.len(), 1);
        assert!(backend.components().0.cleared > 0);
        backend.commit(&txn).unwrap();
        assert_eq!(backend.components().4.published.len(), 1);
        assert!(sink.snapshots >= 5);
    }

    #[test]
    fn rejection_restores_source_model_exactly_and_rollback_replays() {
        let device = Device::ndarray().autodiff();
        device.seed(11);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let probe = TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4]).unwrap();
        let (mut state, txn) = begin(&mut model, &device);
        let before = model_probe_hash(&model, &probe, &device).unwrap();
        let mut backend = backend(model, &device, true);
        let mut sink = Sink { snapshots: 0 };
        assert!(!execute_tensor_consolidation(&mut state, &mut backend, &mut sink).unwrap());
        assert_eq!(
            model_probe_hash(&backend.live_checkpoint().model, &probe, &device).unwrap(),
            before
        );
        backend.restore_teacher(&txn).unwrap();
        assert_eq!(backend.components().4.restored.len(), 1);
        assert_eq!(backend.components().0.cleared, 0);
    }

    #[test]
    fn durable_rollback_persist_failure_leaves_the_caller_pending() {
        #[derive(Default)]
        struct RejectRollbackSink(Vec<SleepState>);

        impl SleepProgressSink for RejectRollbackSink {
            fn persist(&mut self, state: &SleepState) -> Result<()> {
                if state.phase == crate::sleep::SleepPhase::Wake {
                    bail!("injected durable rollback persistence failure");
                }
                self.0.push(state.clone());
                Ok(())
            }
        }

        let device = Device::ndarray().autodiff();
        device.seed(12);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (mut state, _) = begin(&mut model, &device);
        let mut backend = backend(model, &device, true);
        let directory = tempfile::tempdir().unwrap();
        let store = TensorTransactionStore::new(directory.path());
        let mut sink = RejectRollbackSink::default();

        let error =
            execute_tensor_consolidation_durable(&mut state, &mut backend, &store, &mut sink)
                .unwrap_err()
                .to_string();

        assert!(error.contains("rollback persistence failure"), "{error}");
        assert_eq!(state.phase, crate::sleep::SleepPhase::RetentionValidation);
        assert!(state.pending.is_some());
        assert!(
            sink.0
                .iter()
                .all(|snapshot| snapshot.phase != crate::sleep::SleepPhase::Wake),
            "a failed tensor rollback publication escaped into durable progress"
        );
    }

    #[test]
    fn durable_restore_failure_never_publishes_rollback_metadata() {
        #[derive(Default)]
        struct RecordingSink(Vec<SleepState>);

        impl SleepProgressSink for RecordingSink {
            fn persist(&mut self, state: &SleepState) -> Result<()> {
                self.0.push(state.clone());
                Ok(())
            }
        }

        let device = Device::ndarray().autodiff();
        device.seed(14);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (mut state, _) = begin(&mut model, &device);
        let batch = TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4]).unwrap();
        let mut backend = TensorConsolidationBackend::new(
            ImmutableTransformerCheckpoint {
                uri: "teacher.bpk".into(),
                sha256: hash('a'),
                model,
            },
            device.clone(),
            config(),
            Update {
                device,
                cleared: 0,
                fail_restore: true,
            },
            FailingKnowledgeRollouts,
            Judge,
            Evaluator {
                batch,
                reject: false,
                calls: 0,
            },
            Publisher::default(),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let store = TensorTransactionStore::new(directory.path());
        let mut sink = RecordingSink::default();

        let error =
            execute_tensor_consolidation_durable(&mut state, &mut backend, &store, &mut sink)
                .unwrap_err()
                .to_string();

        assert!(error.contains("teacher restore failed"), "{error}");
        assert_eq!(state.phase, crate::sleep::SleepPhase::KnowledgeSeeding);
        assert!(state.pending.is_some());
        assert!(
            sink.0
                .iter()
                .all(|snapshot| snapshot.phase != crate::sleep::SleepPhase::Wake),
            "a tensor rollback cursor was published despite failed restoration"
        );
    }

    #[test]
    fn exact_receiver_optimizer_snapshot_resumes_from_imitation_boundary() {
        let device = Device::ndarray().autodiff();
        device.seed(13);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (mut state, txn) = begin(&mut model, &device);
        let mut first = backend(model.clone(), &device, false);
        first.compute_prospective_update(&txn).unwrap();
        first.stage_student(&txn).unwrap();
        state
            .transition(crate::sleep::SleepPhase::KnowledgeSeeding)
            .unwrap();
        state.reserve_knowledge_rng(0, 2).unwrap();
        let txn = state.pending.as_ref().unwrap().clone();
        first.knowledge_seed(&txn).unwrap();
        state
            .transition(crate::sleep::SleepPhase::Imitation)
            .unwrap();
        let expected_teacher =
            model_parameter_hash(&first.teacher.as_ref().unwrap().model).unwrap();
        let expected_student =
            model_parameter_hash(&first.student.as_ref().unwrap().checkpoint.model).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let store = TensorTransactionStore::new(directory.path());
        let pointer = store
            .publish(&txn, &first.snapshot_inflight(&txn).unwrap())
            .unwrap();
        validate_sha256_identity(&pointer.manifest_sha256, "tensor manifest identity").unwrap();
        state
            .record_tensor_transaction(pointer.generation.clone(), pointer.manifest_sha256.clone())
            .unwrap();
        let recorded_txn = state.pending.as_ref().unwrap().clone();
        let recovered = store
            .load_recorded(
                &recorded_txn,
                &tiny_config(),
                &device,
                AdamWConfig::new().with_weight_decay(0.0).init(),
            )
            .unwrap();
        assert_eq!(
            model_parameter_hash(&recovered.teacher.model).unwrap(),
            expected_teacher
        );
        assert_eq!(
            model_parameter_hash(&recovered.student.checkpoint.model).unwrap(),
            expected_student
        );
        let mut resumed = backend(model, &device, false);
        resumed.restore_inflight(&txn, recovered).unwrap();
        assert_eq!(resumed.components().0.cleared, 1);
        let mut sink = Sink { snapshots: 0 };
        assert!(execute_tensor_consolidation(&mut state, &mut resumed, &mut sink).unwrap());
        assert_eq!(resumed.diagnostics().knowledge_updates, 2);
        assert_eq!(resumed.diagnostics().imitation_updates, 1);
    }

    #[test]
    fn failed_inflight_restore_does_not_partially_install_tensor_topology() {
        let device = Device::ndarray().autodiff();
        device.seed(41);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (_, txn) = begin(&mut model, &device);
        let mut staged = backend(model.clone(), &device, false);
        staged.compute_prospective_update(&txn).unwrap();
        staged.stage_student(&txn).unwrap();
        let recovered = staged.snapshot_inflight(&txn).unwrap();

        let mut resumed = backend(model, &device, false);
        resumed.updates.fail_restore = true;
        let live_before = model_parameter_hash(&resumed.live.model).unwrap();
        let error = resumed
            .restore_inflight(&txn, recovered)
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("prospective sender optimizer state"),
            "{error}"
        );
        assert!(resumed.teacher.is_none());
        assert!(resumed.student.is_none());
        assert!(resumed.pre_update_state.is_none());
        assert!(resumed.staged_update_state.is_none());
        assert!(resumed.receiver_ids.is_empty());
        assert!(resumed.reclaimed_sender_ids.is_empty());
        assert!(resumed.stages.is_empty());
        assert_eq!(resumed.retention, None);
        assert_eq!(resumed.diagnostics, TensorSleepDiagnostics::default());
        assert_eq!(resumed.updates.cleared, 0);
        assert_eq!(
            model_parameter_hash(&resumed.live.model).unwrap(),
            live_before
        );
    }

    #[test]
    fn durable_tensor_store_rejects_tampering_and_symlink_pointer() {
        let device = Device::ndarray().autodiff();
        device.seed(29);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (_, txn) = begin(&mut model, &device);
        let mut backend = backend(model, &device, false);
        backend.compute_prospective_update(&txn).unwrap();
        backend.stage_student(&txn).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let store = TensorTransactionStore::new(directory.path());
        let pointer = store
            .publish(&txn, &backend.snapshot_inflight(&txn).unwrap())
            .unwrap();
        let student = directory
            .path()
            .join("generations")
            .join(&pointer.generation)
            .join(TENSOR_TXN_STUDENT);
        OpenOptions::new()
            .append(true)
            .open(&student)
            .unwrap()
            .write_all(b"tamper")
            .unwrap();
        assert!(
            store
                .load(
                    &txn,
                    &tiny_config(),
                    &device,
                    AdamWConfig::new().with_weight_decay(0.0).init(),
                )
                .is_err()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let pointer_path = directory.path().join(TENSOR_TXN_POINTER);
            fs::remove_file(&pointer_path).unwrap();
            symlink("generations", &pointer_path).unwrap();
            assert!(
                store
                    .load(
                        &txn,
                        &tiny_config(),
                        &device,
                        AdamWConfig::new().with_weight_decay(0.0).init(),
                    )
                    .is_err()
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn tensor_generation_load_is_aba_safe_and_rejects_persistent_replacement() {
        let device = Device::ndarray().autodiff();
        device.seed(31);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (_, txn) = begin(&mut model, &device);
        let expected_teacher = model_parameter_hash(&model).unwrap();
        let mut backend = backend(model, &device, false);
        backend.compute_prospective_update(&txn).unwrap();
        backend.stage_student(&txn).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let store = TensorTransactionStore::new(directory.path());
        let pointer = store
            .publish(&txn, &backend.snapshot_inflight(&txn).unwrap())
            .unwrap();
        let generation = directory
            .path()
            .join("generations")
            .join(&pointer.generation);
        let held = directory.path().join("held-tensor-generation");

        let recovered = store
            .load_pointer_inner(
                &txn,
                &tiny_config(),
                &device,
                AdamWConfig::new().with_weight_decay(0.0).init(),
                &pointer,
                |stage, path| {
                    match stage {
                        TensorLoadStage::AfterCapture => {
                            fs::rename(path, &held)?;
                            fs::create_dir(path)?;
                        }
                        TensorLoadStage::AfterStagedLoad => {
                            fs::remove_dir(path)?;
                            fs::rename(&held, path)?;
                        }
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(
            model_parameter_hash(&recovered.teacher.model).unwrap(),
            expected_teacher,
            "an A->B->A pathname swap changed the loaded teacher"
        );

        let held = directory.path().join("held-persistent-tensor-generation");
        let persistent = store.load_pointer_inner(
            &txn,
            &tiny_config(),
            &device,
            AdamWConfig::new().with_weight_decay(0.0).init(),
            &pointer,
            |stage, path| {
                if stage == TensorLoadStage::AfterCapture {
                    fs::rename(path, &held)?;
                    fs::create_dir(path)?;
                }
                Ok(())
            },
        );
        let error = match persistent {
            Ok(_) => panic!("persistent tensor generation replacement was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("changed during authenticated load"),
            "{error}"
        );
        fs::remove_dir(&generation).unwrap();
        fs::rename(held, &generation).unwrap();

        let student = generation.join(TENSOR_TXN_STUDENT);
        let held_student = directory.path().join("held-student.safetensors");
        let replacement = store.load_pointer_inner(
            &txn,
            &tiny_config(),
            &device,
            AdamWConfig::new().with_weight_decay(0.0).init(),
            &pointer,
            |stage, _| {
                if stage == TensorLoadStage::AfterCapture {
                    fs::rename(&student, &held_student)?;
                    fs::write(&student, b"persistent replacement")?;
                }
                Ok(())
            },
        );
        let error = match replacement {
            Ok(_) => panic!("persistent tensor child replacement was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("file `student.safetensors` changed"),
            "{error}"
        );
        fs::remove_file(&student).unwrap();
        fs::rename(held_student, &student).unwrap();

        let mutation = store.load_pointer_inner(
            &txn,
            &tiny_config(),
            &device,
            AdamWConfig::new().with_weight_decay(0.0).init(),
            &pointer,
            |stage, _| {
                if stage == TensorLoadStage::AfterCapture {
                    OpenOptions::new()
                        .append(true)
                        .open(&student)?
                        .write_all(b"persistent mutation")?;
                }
                Ok(())
            },
        );
        let error = match mutation {
            Ok(_) => panic!("persistent in-place tensor mutation was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("file `student.safetensors` changed"),
            "{error}"
        );
    }

    #[test]
    fn tensor_generation_name_is_bound_to_manifest_and_progress_is_contiguous() {
        let manifest = hash('a');
        assert!(
            validate_generation_identity(&format!("sha256-{}", "a".repeat(64)), &manifest).is_ok()
        );
        assert!(
            validate_generation_identity(&format!("sha256-{}", "b".repeat(64)), &manifest).is_err()
        );

        let mut stages = BTreeSet::from([
            TensorStage::Prospective,
            TensorStage::Student,
            TensorStage::Imitation,
        ]);
        assert!(
            validate_tensor_progress(&stages, None, &TensorSleepDiagnostics::default()).is_err()
        );
        stages.insert(TensorStage::Knowledge);
        let diagnostics = TensorSleepDiagnostics {
            knowledge_updates: 1,
            imitation_updates: 1,
            selected_gradient_tensors: 1,
            knowledge_tokens: 1,
            imitation_samples: 1,
            ..TensorSleepDiagnostics::default()
        };
        validate_tensor_progress(&stages, None, &diagnostics).unwrap();
        stages.insert(TensorStage::Committed);
        assert!(validate_tensor_progress(&stages, None, &diagnostics).is_err());
    }

    #[test]
    fn terminal_transaction_distills_into_base_then_reclaims_slow_reserve() {
        let device = Device::ndarray().autodiff();
        device.seed(23);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (mut state, _) = begin(&mut model, &device);
        let mut first = backend(model, &device, false);
        let mut sink = Sink { snapshots: 0 };
        assert!(execute_tensor_consolidation(&mut state, &mut first, &mut sink).unwrap());
        state
            .transition(crate::sleep::SleepPhase::Candidate)
            .unwrap();
        state.finish_candidate().unwrap();
        state.advance_clock(&sleep_schedule(), 2).unwrap();
        assert_eq!(state.next_due_sender(), Some(0));

        // Advancing from one to two crosses both the next fast boundary and
        // the coincident slow boundary. Fast must drain first.
        let live = first.live_checkpoint().clone();
        let fast_update_hash =
            prospective_update_hash(&live.model, &sender_update(&live.model, &device, 0), 0)
                .unwrap();
        state
            .begin(
                0,
                live.uri.clone(),
                live.sha256.clone(),
                "second-fast-student.bpk".into(),
                hash('b'),
                fast_update_hash,
            )
            .unwrap();
        let mut second = backend_from_checkpoint(live, &device, false);
        assert!(execute_tensor_consolidation(&mut state, &mut second, &mut sink).unwrap());
        state
            .transition(crate::sleep::SleepPhase::Candidate)
            .unwrap();
        state.finish_candidate().unwrap();
        assert_eq!(state.next_due_sender(), Some(1));

        let live = second.live_checkpoint().clone();
        let terminal_update_hash =
            prospective_update_hash(&live.model, &sender_update(&live.model, &device, 1), 1)
                .unwrap();
        let terminal = state
            .begin(
                1,
                live.uri.clone(),
                live.sha256.clone(),
                "terminal-student.bpk".into(),
                hash('b'),
                terminal_update_hash,
            )
            .unwrap();
        assert!(terminal.terminal);
        let batch = TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4]).unwrap();
        let mut terminal_backend = TensorConsolidationBackend::new(
            live,
            device.clone(),
            config(),
            Update {
                device,
                cleared: 0,
                fail_restore: false,
            },
            Rollouts(batch.clone()),
            Judge,
            Evaluator {
                batch,
                reject: false,
                calls: 0,
            },
            Publisher::default(),
        )
        .unwrap();
        assert!(
            execute_tensor_consolidation(&mut state, &mut terminal_backend, &mut sink).unwrap()
        );
        assert!(
            terminal_backend
                .live_checkpoint()
                .model
                .memory_slot_statuses()
                .iter()
                .filter(|status| status.tier == 1)
                .all(|status| !status.active)
        );
        assert!(!terminal_backend.receiver_parameter_ids().is_empty());
    }

    #[test]
    fn edit_and_group_rewards_are_well_formed() {
        assert_eq!(thresholded_edit_reward(&[1, 2], &[1, 2], 0), 1.0);
        assert_eq!(thresholded_edit_reward(&[1, 2], &[8, 9], 1), 0.0);
        assert!(
            normalized_group_advantages(&[1.0, 0.0, 0.5])
                .iter()
                .sum::<f32>()
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn threshold_banded_edit_distance_matches_full_dynamic_programming() {
        fn full_distance(left: &[i64], right: &[i64]) -> usize {
            let mut previous = (0..=right.len()).collect::<Vec<_>>();
            let mut current = vec![0; right.len() + 1];
            for (left_index, left_token) in left.iter().enumerate() {
                current[0] = left_index + 1;
                for (right_index, right_token) in right.iter().enumerate() {
                    current[right_index + 1] = (previous[right_index]
                        + usize::from(left_token != right_token))
                    .min(previous[right_index + 1] + 1)
                    .min(current[right_index] + 1);
                }
                std::mem::swap(&mut previous, &mut current);
            }
            previous[right.len()]
        }

        for left_len in 0..=5 {
            for right_len in 0..=5 {
                for left_bits in 0..(1usize << left_len) {
                    let left = (0..left_len)
                        .map(|bit| i64::from(((left_bits >> bit) & 1) != 0))
                        .collect::<Vec<_>>();
                    for right_bits in 0..(1usize << right_len) {
                        let right = (0..right_len)
                            .map(|bit| i64::from(((right_bits >> bit) & 1) != 0))
                            .collect::<Vec<_>>();
                        let exact = full_distance(&left, &right);
                        for maximum in 0..=6 {
                            assert_eq!(
                                thresholded_levenshtein(&left, &right, maximum),
                                (exact <= maximum).then_some(exact),
                                "left={left:?} right={right:?} maximum={maximum}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chunked_forward_kl_matches_an_unchunked_batch() {
        let device = Device::ndarray().autodiff();
        device.seed(37);
        let teacher = Transformer::new(&tiny_config(), &device).unwrap();
        device.seed(41);
        let student = Transformer::new(&tiny_config(), &device).unwrap();
        let batch = TokenRolloutBatch::new(1, 5, vec![1, 2, 3, 4, 5]).unwrap();

        let unchunked = forward_kl_value(&teacher, &student, &batch, &device, 5, 1.5).unwrap();
        let uneven_chunks = forward_kl_value(&teacher, &student, &batch, &device, 2, 1.5).unwrap();
        assert!(
            (unchunked - uneven_chunks).abs() < 1e-5,
            "chunking changed KL from {unchunked} to {uneven_chunks}"
        );
    }

    #[test]
    fn grpo_log_probabilities_batch_equal_length_candidates_without_reordering() {
        let device = Device::ndarray().autodiff();
        device.seed(43);
        let model = Transformer::new(&tiny_config(), &device).unwrap();
        let prefix = vec![1, 2];
        let candidates = vec![vec![3, 4], vec![5], vec![6, 7]];
        let buckets = candidate_length_buckets(&candidates, 2).unwrap();
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets.get(&2), Some(&vec![0, 2]));

        let grouped = padded_group_log_probabilities(&model, &prefix, &candidates, 2, &device)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let mut singleton = Vec::new();
        for candidate in &candidates {
            singleton.extend(
                padded_group_log_probabilities(
                    &model,
                    &prefix,
                    std::slice::from_ref(candidate),
                    2,
                    &device,
                )
                .unwrap()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap(),
            );
        }
        assert_eq!(grouped.len(), singleton.len());
        for (batched, separate) in grouped.iter().zip(&singleton) {
            assert!(
                (batched - separate).abs() < 1e-5,
                "batched log probability {batched} differs from {separate}"
            );
        }
    }

    #[test]
    fn injected_sleep_rollout_work_is_bounded_before_tensor_execution() {
        let device = Device::ndarray().autodiff();
        let model = Transformer::new(&tiny_config(), &device).unwrap();
        let oversized_batch = TokenRolloutBatch {
            batch: MAX_TENSOR_ROLLOUT_BATCH_ROWS + 1,
            sequence: 1,
            token_ids: vec![1; MAX_TENSOR_ROLLOUT_BATCH_ROWS + 1],
        };
        assert!(oversized_batch.validate_for(&model).is_err());

        let group = ImitationGroup {
            prefix: vec![1, 2],
            teacher_continuation: vec![3, 4],
            candidates: vec![vec![3, 4], vec![5, 6]],
        };
        let groups = vec![group; MAX_TENSOR_IMITATION_GROUPS + 1];
        let error = validate_imitation_groups(&groups, &model, 2, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than"), "{error}");

        let batch = TokenRolloutBatch::new(1, 1, vec![1]).unwrap();
        let batches = vec![batch; MAX_TENSOR_ROLLOUT_BATCHES + 1];
        let error = validate_rollout_batches(&batches, &model, "hostile source")
            .unwrap_err()
            .to_string();
        assert!(error.contains("rollout batches"), "{error}");
    }

    #[test]
    fn prospective_update_rejects_parameters_outside_sender_tier() {
        let device = Device::ndarray().autodiff();
        let mut teacher = Transformer::new(&tiny_config(), &device).unwrap();
        teacher.activate_memory_slot_all_layers(0, 0).unwrap();
        let mut student = teacher.clone();
        let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
        let target = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], &device);
        let mut gradients = student.forward_loss(input, target).backward();
        let selected = GradientsParams::from_module(&mut gradients, &student);
        student =
            AdamWConfig::new()
                .with_weight_decay(0.0)
                .init()
                .step(1e-2.into(), student, selected);
        let error = prospective_update_hash_at_boundary(&teacher, &student, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("escaped sender tier"), "{error}");
    }

    #[test]
    fn tensor_backend_rejects_a_forged_prospective_delta_hash() {
        let device = Device::ndarray().autodiff();
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (_, mut txn) = begin(&mut model, &device);
        txn.prospective_update_hash = hash('0');
        let mut backend = backend(model, &device, false);
        let error = backend
            .compute_prospective_update(&txn)
            .unwrap_err()
            .to_string();
        assert!(error.contains("canonical sender-tier delta"), "{error}");
    }

    struct DreamOps {
        manifests: BTreeMap<String, Vec<GeneratedDream>>,
        trials: usize,
    }
    impl TransformerDreamOps for DreamOps {
        fn generate(
            &mut self,
            txn: &ConsolidationTxn,
            _: &Transformer,
            count: usize,
            routing: MemoryRouting,
        ) -> Result<(String, Vec<GeneratedDream>)> {
            ensure!(
                matches!(routing, MemoryRouting::Dream { .. }),
                "wake routing used for dreams"
            );
            let values = (0..count)
                .map(|index| GeneratedDream {
                    id: format!("d{index}"),
                    artifact_hash: hash(if index == 0 { '1' } else { '2' }),
                    gradient: vec![1.0, index as f32 + 1.0],
                    diversity_key: index as u64,
                })
                .collect::<Vec<_>>();
            let manifest = hash('3');
            self.manifests
                .insert(format!("{}:{manifest}", txn.id), values.clone());
            Ok((manifest, values))
        }
        fn load(&mut self, txn: &ConsolidationTxn, manifest: &str) -> Result<Vec<GeneratedDream>> {
            self.manifests
                .get(&format!("{}:{manifest}", txn.id))
                .cloned()
                .context("manifest absent")
        }
        fn reference_gradient(
            &mut self,
            _: &ConsolidationTxn,
            _: &Transformer,
            _: &str,
        ) -> Result<Vec<f32>> {
            Ok(vec![1.0, 1.0])
        }
        fn isolated_lora_trial(
            &mut self,
            _: &ConsolidationTxn,
            mut model: Transformer,
            candidate: &GeneratedDream,
            rank: usize,
            alpha: usize,
        ) -> Result<DreamTrial> {
            ensure!(rank == 64 && alpha == 128, "wrong LoRA geometry");
            model.reset_memory_slot_all_layers(1, 0, 99)?;
            self.trials += 1;
            Ok(DreamTrial {
                candidate_id: candidate.id.clone(),
                adapter_hash: hash('4'),
                evaluator_hash: hash('5'),
                independent_task_improvement: 0.1,
            })
        }
        fn restem_update_policy(
            &mut self,
            _: &ConsolidationTxn,
            _: &Transformer,
            _: &[DreamTrial],
            _: usize,
        ) -> Result<String> {
            Ok(hash('6'))
        }
    }

    #[test]
    fn rejected_or_completed_dream_trials_cannot_mutate_shared_model() {
        let device = Device::ndarray().autodiff();
        device.seed(17);
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let (mut state, _) = begin(&mut model, &device);
        let mut consolidation = backend(model, &device, false);
        let mut sink = Sink { snapshots: 0 };
        assert!(execute_tensor_consolidation(&mut state, &mut consolidation, &mut sink).unwrap());
        let probe = TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4]).unwrap();
        let mut dreams = TensorDreamBackend::new(
            consolidation.live_checkpoint().model.clone(),
            device,
            probe,
            DreamOps {
                manifests: BTreeMap::new(),
                trials: 0,
            },
        )
        .unwrap();
        let committed = state.pending.as_ref().unwrap();
        dreams
            .bind_candidate_checkpoint(
                committed.candidate_checkpoint.as_ref().unwrap(),
                committed.candidate_hash.as_ref().unwrap(),
            )
            .unwrap();
        let before = dreams.shared_checkpoint_hash().unwrap();
        let config = DreamingConfig {
            candidate_count: 2,
            retain_top: 1,
            retain_random: 1,
            lora_rank: 64,
            lora_alpha: 128,
            restem_iterations: 1,
            selector_version: "gradient-cosine-v1".into(),
            reference_set_hash: hash('6'),
            trial_evaluator_hash: hash('5'),
        };
        assert_eq!(
            run_dreaming(&mut state, &config, &mut dreams)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(dreams.shared_checkpoint_hash().unwrap(), before);
        assert_eq!(dreams.operations().trials, 2);
    }

    #[test]
    fn model_parameter_identity_covers_dormant_reserve_state() {
        let device = Device::ndarray().autodiff();
        device.seed(31);
        let model = Transformer::new(&tiny_config(), &device).unwrap();
        let probe = TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4]).unwrap();
        let probe_before = model_probe_hash(&model, &probe, &device).unwrap();
        let mut dreams = TensorDreamBackend::new(
            model,
            device.clone(),
            probe.clone(),
            DreamOps {
                manifests: BTreeMap::new(),
                trials: 0,
            },
        )
        .unwrap();
        let checkpoint_before = model_parameter_hash(dreams.shared_model()).unwrap();

        // Reclaiming a slot leaves it dormant, so ordinary logits return to
        // their original value while its generation/stored tensors differ.
        dreams.shared.activate_memory_slot_all_layers(1, 1).unwrap();
        dreams
            .shared
            .reset_memory_slot_all_layers(1, 1, 99)
            .unwrap();
        assert_eq!(
            model_probe_hash(dreams.shared_model(), &probe, &device).unwrap(),
            probe_before
        );
        assert_ne!(
            model_parameter_hash(dreams.shared_model()).unwrap(),
            checkpoint_before
        );
    }

    #[test]
    fn parameter_id_restore_rejects_aliasing_without_mutating_the_model() {
        let device = Device::ndarray().autodiff();
        let mut model = Transformer::new(&tiny_config(), &device).unwrap();
        let original = parameter_ids(&model);
        assert!(original.len() >= 2);
        let mut aliased = original.clone();
        aliased[1] = aliased[0];

        let error = restore_parameter_ids(&mut model, &aliased)
            .unwrap_err()
            .to_string();

        assert!(error.contains("repeats a stored parameter ID"), "{error}");
        assert_eq!(parameter_ids(&model), original);
    }
}
