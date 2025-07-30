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
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use burn_optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use hermes_llm::{MemoryRouting, ModelDef, Transformer, load_safetensors, save_safetensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::optimizer_artifact::save_canonical_module_optimizer;
use crate::sleep::{
    CommittedCandidate, ConsolidationBackend, ConsolidationTxn, DreamTrial, DreamingBackend,
    GeneratedDream, ImitationConfig, KnowledgeSeedingConfig, RngReservation, SleepProgressSink,
    SleepState, run_consolidation_with_progress,
};

const KNOWLEDGE_DEVICE_RNG_DOMAIN: u64 = 0x6b6e_6f77_6c65_6467;
const IMITATION_DEVICE_RNG_DOMAIN: u64 = 0x696d_6974_6174_696f;

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
        validate_sha256(&self.sha256)
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

#[derive(Default)]
struct ParameterFingerprintVisitor {
    values: BTreeMap<u64, [u8; 32]>,
    failure: Option<String>,
}

impl ParameterFingerprintVisitor {
    fn record(&mut self, id: ParamId, kind: u8, shape: &[usize], bytes: &[u8]) {
        if self.failure.is_some() {
            return;
        }
        let mut hash = Sha256::new();
        hash.update(b"hermes-parameter-fingerprint-v1\0");
        hash.update([kind]);
        hash.update((shape.len() as u64).to_le_bytes());
        for dimension in shape {
            hash.update((*dimension as u64).to_le_bytes());
        }
        hash.update(bytes);
        let digest: [u8; 32] = hash.finalize().into();
        if self.values.insert(id.val(), digest).is_some() {
            self.failure = Some(format!("model repeats parameter ID {}", id.val()));
        }
    }
}

impl ModuleVisitor for ParameterFingerprintVisitor {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        match parameter.val().into_data().convert::<f32>().to_vec::<f32>() {
            Ok(values) => {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                self.record(parameter.id, b'f', &parameter.shape().dims::<D>(), &bytes);
            }
            Err(error) => self.failure = Some(error.to_string()),
        }
    }

    fn visit_int<const D: usize>(&mut self, parameter: &Param<Tensor<D, Int>>) {
        match parameter.val().into_data().to_vec::<i64>() {
            Ok(values) => {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                self.record(parameter.id, b'i', &parameter.shape().dims::<D>(), &bytes);
            }
            Err(error) => self.failure = Some(error.to_string()),
        }
    }

    fn visit_bool<const D: usize>(&mut self, parameter: &Param<Tensor<D, Bool>>) {
        match parameter.val().into_data().to_vec::<bool>() {
            Ok(values) => {
                let values = values.into_iter().map(u8::from).collect::<Vec<_>>();
                self.record(parameter.id, b'b', &parameter.shape().dims::<D>(), &values);
            }
            Err(error) => self.failure = Some(error.to_string()),
        }
    }
}

fn parameter_fingerprints(model: &Transformer) -> Result<BTreeMap<u64, [u8; 32]>> {
    let mut visitor = ParameterFingerprintVisitor::default();
    model.visit(&mut visitor);
    if let Some(error) = visitor.failure {
        bail!("fingerprinting prospective update parameters: {error}");
    }
    ensure!(!visitor.values.is_empty(), "model exposes no parameters");
    Ok(visitor.values)
}

/// Canonical scope and identity check for a prospective sender update. Only
/// the sender tier's base plus currently active reserve slots may change.
/// Dormant slots, slower/faster tiers, embeddings, mixers, and heads are
/// immutable at this boundary.
pub fn prospective_update_hash(
    teacher: &Transformer,
    student: &Transformer,
    sender: usize,
) -> Result<String> {
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

    let teacher = parameter_fingerprints(teacher)?;
    let student = parameter_fingerprints(student)?;
    ensure!(
        teacher.keys().eq(student.keys()),
        "prospective student parameter IDs differ from teacher"
    );
    let changed = teacher
        .iter()
        .filter_map(|(id, before)| {
            let after = student.get(id).expect("parameter key sets checked");
            (before != after).then_some((*id, *before, *after))
        })
        .collect::<Vec<_>>();
    ensure!(!changed.is_empty(), "prospective sender update is empty");
    let escaped = changed
        .iter()
        .filter_map(|(id, _, _)| (!eligible.contains(id)).then_some(*id))
        .collect::<Vec<_>>();
    ensure!(
        escaped.is_empty(),
        "prospective update escaped sender tier {sender}: parameter IDs {escaped:?}"
    );
    let mut hash = Sha256::new();
    hash.update(b"hermes-prospective-update-v1\0");
    hash.update((sender as u64).to_le_bytes());
    for (id, before, after) in changed {
        hash.update(id.to_le_bytes());
        hash.update(before);
        hash.update(after);
    }
    Ok(format!("sha256:{:x}", hash.finalize()))
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
        Ok(Self {
            batch,
            sequence,
            token_ids,
        })
    }

    fn validate_for(&self, model: &Transformer) -> Result<()> {
        ensure!(
            self.sequence <= model.config().max_seq_len,
            "rollout exceeds model sequence limit"
        );
        ensure!(
            self.token_ids
                .iter()
                .all(|id| *id >= 0 && (*id as usize) < model.config().vocab_size),
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
        validate_sha256(&self.imitation.semantic_judge_hash)?;
        validate_sha256(&self.retention.evaluator_hash)?;
        validate_sha256(&self.retention.suite_hash)?;
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
static TENSOR_TXN_STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
        validate_sha256(&recovered.student.update_sha256)?;
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
        sync_regular_file(&teacher_path)?;
        save_safetensors(
            &recovered.student.checkpoint.model.clone().valid(),
            &student_path,
        )?;
        sync_regular_file(&student_path)?;
        save_canonical_module_optimizer(&recovered.receiver_optimizer, &optimizer_path)
            .context("saving receiver optimizer state")?;
        sync_regular_file(&optimizer_path)?;
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
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
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
        ensure_existing_directory(&self.root, "tensor transaction root")?;
        ensure_existing_directory(
            &self.root.join("generations"),
            "tensor transaction generations",
        )?;
        let pointer_path = self.root.join(TENSOR_TXN_POINTER);
        ensure_regular_file(&pointer_path, "tensor transaction pointer")?;
        let pointer: TensorTransactionPointer = serde_json::from_slice(&fs::read(&pointer_path)?)?;
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
        ensure_existing_directory(&self.root, "tensor transaction root")?;
        ensure_existing_directory(
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
        validate_sha256(&pointer.manifest_sha256)?;
        validate_generation_identity(&pointer.generation, &pointer.manifest_sha256)?;
        let generation = self.root.join("generations").join(&pointer.generation);
        let metadata = verify_generation(&generation, txn.id, &pointer.manifest_sha256)?;
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
        load_safetensors(&mut teacher, generation.join(TENSOR_TXN_TEACHER))?;
        let mut student = Transformer::new(config, device)?;
        restore_parameter_ids(&mut student, &metadata.student_parameter_ids)?;
        load_safetensors(&mut student, generation.join(TENSOR_TXN_STUDENT))?;
        let optimizer = optimizer
            .load(generation.join(TENSOR_TXN_OPTIMIZER))
            .context("loading receiver optimizer state")?;
        let pre_update_state =
            ProspectiveUpdateSnapshot::new(fs::read(generation.join(TENSOR_TXN_UPDATE_PRE))?)?;
        let staged_update_state =
            ProspectiveUpdateSnapshot::new(fs::read(generation.join(TENSOR_TXN_UPDATE_STAGED))?)?;
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

fn ensure_directory(path: &Path, label: &str) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating {label} {}", path.display()))?;
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} is not a real directory"
    );
    Ok(())
}

fn ensure_existing_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("{label} {} is missing", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} is not a real directory"
    );
    Ok(())
}

fn ensure_regular_file(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("{label} {} is missing", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{label} is not a regular non-symlink file"
    );
    Ok(())
}

fn sync_regular_file(path: &Path) -> Result<()> {
    ensure_regular_file(path, "tensor transaction file")?;
    File::open(path)?.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating immutable file {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    ensure_regular_file(path, "tensor transaction artifact")?;
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .context("tensor transaction artifact length overflow")?;
        hash.update(&buffer[..read]);
    }
    Ok((bytes, format!("sha256:{:x}", hash.finalize())))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
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
    validate_sha256(&format!("sha256:{digest}"))
}

fn validate_generation_identity(name: &str, manifest_sha256: &str) -> Result<()> {
    validate_generation_name(name)?;
    validate_sha256(manifest_sha256)?;
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

fn verify_generation(
    generation: &Path,
    txn_id: u64,
    expected_manifest_hash: &str,
) -> Result<TensorTransactionMetadata> {
    let metadata = fs::symlink_metadata(generation)
        .with_context(|| format!("tensor generation {} is missing", generation.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "tensor generation is not a real directory"
    );
    let manifest_path = generation.join(TENSOR_TXN_MANIFEST);
    ensure_regular_file(&manifest_path, "tensor transaction manifest")?;
    let manifest_bytes = fs::read(&manifest_path)?;
    ensure!(
        sha256_bytes(&manifest_bytes) == expected_manifest_hash,
        "tensor transaction manifest hash mismatch"
    );
    let manifest: TensorTransactionManifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest.version == TENSOR_TXN_MANIFEST_VERSION && manifest.txn_id == txn_id,
        "tensor transaction manifest identity/version mismatch"
    );
    let expected_names = [
        TENSOR_TXN_MANIFEST,
        TENSOR_TXN_METADATA,
        TENSOR_TXN_OPTIMIZER,
        TENSOR_TXN_STUDENT,
        TENSOR_TXN_TEACHER,
        TENSOR_TXN_UPDATE_PRE,
        TENSOR_TXN_UPDATE_STAGED,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    let mut observed_names = BTreeSet::new();
    for entry in fs::read_dir(generation)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("tensor generation contains non-UTF-8 name"))?;
        let entry_metadata = fs::symlink_metadata(entry.path())?;
        ensure!(
            entry_metadata.is_file() && !entry_metadata.file_type().is_symlink(),
            "tensor generation contains a non-file or symlink"
        );
        observed_names.insert(name);
    }
    ensure!(
        observed_names == expected_names,
        "tensor generation file set differs from its fixed schema"
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
        validate_sha256(&file.sha256)?;
        let (bytes, hash) = hash_file(&generation.join(&file.path))?;
        ensure!(
            bytes == file.bytes && hash == file.sha256,
            "tensor transaction artifact `{}` failed authentication",
            file.path
        );
    }
    let transaction_path = generation.join(TENSOR_TXN_METADATA);
    let transaction: TensorTransactionMetadata =
        serde_json::from_slice(&fs::read(&transaction_path)?)?;
    ensure!(
        transaction.version == TENSOR_TXN_STORE_VERSION && transaction.txn_id == txn_id,
        "tensor transaction metadata identity/version mismatch"
    );
    validate_sha256(&transaction.teacher_sha256)?;
    validate_sha256(&transaction.student_sha256)?;
    validate_sha256(&transaction.prospective_update_sha256)?;
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
    Ok(transaction)
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
        ids.len() == burn::module::list_param_ids(model).len(),
        "tensor transaction parameter-ID topology differs from model"
    );
    let mut mapper = RestoreParameterIds { ids: ids.iter() };
    *model = model.clone().map(&mut mapper);
    ensure!(mapper.ids.next().is_none(), "too many stored parameter IDs");
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
        self.receiver_ids = if txn.terminal {
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
        self.reclaimed_sender_ids = statuses
            .iter()
            .filter(|status| {
                status.tier == txn.sender
                    && txn.sender_slots_to_reset.contains(&status.slot)
                    && !status.active
            })
            .flat_map(|status| status.parameter_ids.clone())
            .collect();
        ensure!(
            txn.sender_slots_to_reset.is_empty() || !self.reclaimed_sender_ids.is_empty(),
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
        )?;
        let reference_log_probs = padded_group_log_probabilities(
            &teacher,
            &group.prefix,
            &group.candidates,
            max_tokens,
            &self.device,
        )?;
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
            validate_sha256(&student.update_sha256)?;
            let derived_update =
                prospective_update_hash(&teacher.model, &student.checkpoint.model, txn.sender)
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
                state.reserve_knowledge_rng(
                    0,
                    (backend.config.knowledge.teacher_rollouts
                        + backend.config.knowledge.detached_student_rollouts)
                        as u64,
                )?;
                progress.persist(state)?;
                let current_txn = state.pending.clone().expect("transaction checked above");
                backend.knowledge_seed(&current_txn)?;
                state.transition(crate::sleep::SleepPhase::Imitation)?;
                persist_tensor_boundary(state, backend, store, progress)?;
            }
            crate::sleep::SleepPhase::Imitation => {
                state.reserve_imitation_rng(0, backend.config.imitation.grpo_group_size as u64)?;
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
            state.rollback()?;
            progress.persist(state)?;
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
            let restore = backend.restore_teacher(&txn);
            state.rollback()?;
            let persist = progress.persist(state);
            if let Err(restore_error) = restore {
                bail!(
                    "tensor consolidation failed: {error:#}; teacher restore failed: {restore_error:#}"
                );
            }
            persist.context("persisting tensor consolidation rollback")?;
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
    let mut teacher_entropy = 0.0_f32;
    let mut student_entropy = 0.0_f32;
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
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading teacher entropy metric")?[0];
        let student_entropy_chunk = (student_probability * student_log.clone().detach())
            .sum_dim(1)
            .mean()
            .neg()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .context("reading student entropy metric")?[0];
        teacher_entropy += teacher_entropy_chunk * chunk_weight;
        student_entropy += student_entropy_chunk * chunk_weight;
        let loss = (teacher_probability * (teacher_log - student_log))
            .sum_dim(1)
            .mean()
            .mul_scalar(temperature * temperature * chunk_weight);
        total = Some(total.map_or(loss.clone(), |sum| sum + loss));
    }
    let total = total.context("empty KL rollout")?;
    let value = total
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .context("reading KL scalar")?[0];
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

fn continuation_token_log_probabilities(
    model: &Transformer,
    prefix: &[i64],
    continuation: &[i64],
    device: &Device,
) -> Result<Tensor<1>> {
    ensure!(
        !prefix.is_empty() && !continuation.is_empty(),
        "imitation prefix/continuation is empty"
    );
    let mut full = prefix.to_vec();
    full.extend_from_slice(continuation);
    ensure!(
        full.len() - 1 <= model.config().max_seq_len,
        "imitation sequence exceeds model limit"
    );
    ensure!(
        full.iter()
            .all(|id| *id >= 0 && (*id as usize) < model.config().vocab_size),
        "imitation token is outside vocabulary"
    );
    let input_len = full.len() - 1;
    let start = prefix.len() - 1;
    let input = Tensor::<2, Int>::from_data(
        TensorData::new(full[..input_len].to_vec(), [1, input_len]),
        device,
    );
    let positions = Tensor::<1, Int>::from_data(
        TensorData::new(
            (start..start + continuation.len())
                .map(|value| value as i64)
                .collect(),
            [continuation.len()],
        ),
        device,
    );
    let targets = Tensor::<1, Int>::from_data(
        TensorData::new(continuation.to_vec(), [continuation.len()]),
        device,
    );
    Ok(
        log_softmax(model.forward_selected_logits(input, positions), 1)
            .gather(1, targets.unsqueeze_dim(1))
            .reshape([continuation.len()]),
    )
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
    let rows = candidates
        .iter()
        .map(|candidate| {
            ensure!(
                candidate.len() <= max_tokens,
                "GRPO candidate exceeds padded token width"
            );
            let log_probabilities =
                continuation_token_log_probabilities(model, prefix, candidate, device)?;
            let padded = if candidate.len() < max_tokens {
                Tensor::cat(
                    vec![
                        log_probabilities,
                        Tensor::<1>::zeros([max_tokens - candidate.len()], device),
                    ],
                    0,
                )
            } else {
                log_probabilities
            };
            Ok(padded.unsqueeze_dim::<2>(0))
        })
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
    ensure!(
        group
            .candidates
            .iter()
            .all(|candidate| !candidate.is_empty()
                && group.prefix.len() + candidate.len() - 1 <= model.config().max_seq_len),
        "invalid imitation candidate"
    );
    Ok(())
}

pub fn thresholded_edit_reward(reference: &[i64], candidate: &[i64], maximum: usize) -> f32 {
    let distance = levenshtein(reference, candidate);
    if distance > maximum {
        return 0.0;
    }
    1.0 - distance as f32 / reference.len().max(candidate.len()).max(1) as f32
}

fn levenshtein(left: &[i64], right: &[i64]) -> usize {
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

fn validate_sha256(value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        bail!("identity must start with sha256:");
    };
    ensure!(
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "identity must be sha256 plus 64 lowercase hex digits"
    );
    Ok(())
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
                    for value in values {
                        self.hash.update(value.to_bits().to_le_bytes());
                    }
                }
                Err(error) => self.failure = Some(error.to_string()),
            }
        }

        fn visit_int<const D: usize>(&mut self, parameter: &Param<Tensor<D, Int>>) {
            self.shape(b'i', parameter.shape().dims::<D>());
            match parameter.val().into_data().to_vec::<i64>() {
                Ok(values) => {
                    for value in values {
                        self.hash.update(value.to_le_bytes());
                    }
                }
                Err(error) => self.failure = Some(error.to_string()),
            }
        }

        fn visit_bool<const D: usize>(&mut self, parameter: &Param<Tensor<D, Bool>>) {
            self.shape(b'b', parameter.shape().dims::<D>());
            match parameter.val().into_data().to_vec::<bool>() {
                Ok(values) => {
                    for value in values {
                        self.hash.update([u8::from(value)]);
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
        Ok(Self {
            immutable_shared: shared.clone(),
            shared,
            device,
            operations,
        })
    }
    pub fn shared_model(&self) -> &Transformer {
        &self.shared
    }
    pub fn operations(&self) -> &O {
        &self.operations
    }
    fn fingerprint(&self) -> Result<String> {
        // A probe-logit digest can miss changes to dormant reserve slots or
        // parameters that do not affect that particular input. Dream trials
        // must preserve the entire immutable candidate, including checkpoint
        // state that is intentionally excluded from active routing.
        model_parameter_hash(&self.shared)
    }
}

impl<O: TransformerDreamOps> DreamingBackend for TensorDreamBackend<O> {
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
        validate_sha256(&manifest)?;
        for dream in &dreams {
            validate_sha256(&dream.artifact_hash)?;
        }
        Ok((manifest, dreams))
    }

    fn load_generated_dreams(
        &mut self,
        txn: &ConsolidationTxn,
        manifest: &str,
    ) -> Result<Vec<GeneratedDream>> {
        validate_sha256(manifest)?;
        let dreams = self.operations.load(txn, manifest)?;
        for dream in &dreams {
            validate_sha256(&dream.artifact_hash)?;
        }
        Ok(dreams)
    }

    fn reference_gradient(
        &mut self,
        txn: &ConsolidationTxn,
        reference_hash: &str,
    ) -> Result<Vec<f32>> {
        validate_sha256(reference_hash)?;
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
        validate_sha256(&candidate.artifact_hash)?;
        let trial = self
            .operations
            .isolated_lora_trial(txn, isolated, candidate, rank, alpha)?;
        validate_sha256(&trial.adapter_hash)?;
        validate_sha256(&trial.evaluator_hash)?;
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
        validate_sha256(&pointer.manifest_sha256).unwrap();
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
            Update { device, cleared: 0 },
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
        let error = prospective_update_hash(&teacher, &student, 0)
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
    fn dream_checkpoint_identity_covers_dormant_reserve_state() {
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
        let checkpoint_before = dreams.shared_checkpoint_hash().unwrap();

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
        assert_ne!(dreams.shared_checkpoint_hash().unwrap(), checkpoint_before);
    }
}
