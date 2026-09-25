//! Crash-safe, resumable training checkpoints.
//!
//! Parameter IDs are persisted alongside model and optimizer state because
//! Burn optimizers key their state by those IDs. Checkpoint files are sealed in
//! immutable, content-addressed generation directories. A generation becomes
//! visible only when the small `current.json` pointer is atomically replaced,
//! so an interrupted save cannot hide the preceding complete checkpoint.

#[cfg(not(unix))]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use burn::module::{AutodiffModule, Module, ModuleMapper, Param, ParamId};
use burn::tensor::{Bool, Bytes, Device, Int, Tensor};
use burn_optim::ModuleOptimizer;
use serde::{Deserialize, Deserializer, Serialize};
use summa_llm::{Transformer, load_safetensors_bytes, save_safetensors};

use crate::muon::BatchedMuon;
#[cfg(unix)]
use summa_train::artifact_io::{PinnedDirectory, StableIdentity};
use summa_train::artifact_io::{
    ensure_real_directory, hash_open_file, hash_regular_file_hex, read_regular_bounded, sha256_hex,
    sync_directory, sync_regular_file, validate_sha256_hex, validate_sha256_identity,
    write_new_synced as write_synced_new,
};
use summa_train::benchmark::{
    TRAINING_ACCOUNTING_FILE, TRAINING_ACCOUNTING_VERSION, TrainingAccounting,
};
use summa_train::builtin_sleep_adapters::{
    MAX_WAKE_CONTEXT_ID_BYTES, MAX_WAKE_CONTEXT_RECORDS, MAX_WAKE_CONTEXT_TOKENS,
    MAX_WAKE_CONTEXT_TOTAL_TOKENS, WakeContextRecord,
};
use summa_train::metrics::MetricWriter;
use summa_train::native_sleep::NativeSleepCheckpoint;
use summa_train::optimizer_artifact::{
    canonical_module_optimizer_bytes, save_canonical_module_optimizer,
};
use summa_train::sleep::{MemoryOptimizerScopes, UpdateClock};
use summa_train::workflow::MemoryUpdateMode;

pub(crate) type AdamWOptimizer = ModuleOptimizer;

pub(crate) const TRAINING_STATE_VERSION: u32 = 2;

const CHECKPOINT_POINTER_VERSION: u32 = 1;
const CHECKPOINT_MANIFEST_VERSION: u32 = 1;
const GENERATIONS_DIRECTORY: &str = "generations";
const CURRENT_POINTER: &str = "current.json";
const GENERATION_MANIFEST: &str = "generation-manifest.json";
const TRAINING_STATE_FILE: &str = "training-state.json";
const WEIGHTS_FILE: &str = "weights.safetensors";
const ADAMW_FILE: &str = "adamw-state.bpk";
const MUON_FILE: &str = "muon-state.bpk";
const MAX_CHECKPOINT_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CHECKPOINT_MEMBER_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_CHECKPOINT_GENERATION_FILES: usize = 1_024;
const MAX_CHECKPOINT_GENERATION_DIRECTORIES: usize = 256;
const MAX_CHECKPOINT_GENERATION_DEPTH: usize = 16;

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CurrentCheckpoint {
    version: u32,
    generation: String,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationManifest {
    version: u32,
    training_state_version: u32,
    global_step: usize,
    phase: usize,
    phase_id: String,
    files: Vec<GenerationFile>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug)]
struct GenerationSnapshot {
    files: Vec<GenerationFile>,
    manifest: Option<Vec<u8>>,
    training_state: Option<Vec<u8>>,
    training_accounting: Option<Vec<u8>>,
    access: Option<GenerationAccess>,
}

#[derive(Default)]
struct GenerationTraversalBudget {
    files: usize,
    directories: usize,
}

impl GenerationTraversalBudget {
    fn enter_directory(&mut self, depth: usize) -> Result<()> {
        ensure!(
            depth <= MAX_CHECKPOINT_GENERATION_DEPTH,
            "checkpoint generation exceeds its maximum directory depth"
        );
        if depth > 0 {
            self.directories = self
                .directories
                .checked_add(1)
                .context("checkpoint directory count overflow")?;
            ensure!(
                self.directories <= MAX_CHECKPOINT_GENERATION_DIRECTORIES,
                "checkpoint generation contains too many directories"
            );
        }
        Ok(())
    }

    fn record_file(&mut self, bytes: u64) -> Result<()> {
        self.files = self
            .files
            .checked_add(1)
            .context("checkpoint file count overflow")?;
        ensure!(
            self.files <= MAX_CHECKPOINT_GENERATION_FILES,
            "checkpoint generation contains too many files"
        );
        ensure!(
            bytes <= MAX_CHECKPOINT_MEMBER_BYTES,
            "checkpoint generation contains an oversized file"
        );
        Ok(())
    }
}

#[cfg(unix)]
#[derive(Debug)]
struct GenerationAccess {
    directory: PinnedDirectory,
}

#[cfg(not(unix))]
#[derive(Debug)]
struct GenerationAccess {
    files: BTreeMap<String, File>,
}

/// Stable, machine-readable result returned by the checkpoint verifier.
///
/// The relaunch supervisor uses this API instead of duplicating the trainer's
/// exact-resume schema in Bash/Python. The workflow signature is retained only
/// for binding metric validation to the checkpoint's run identity; it is not
/// part of the stable JSON/TSV descriptor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct VerifiedCheckpoint {
    pub(crate) version: u32,
    pub(crate) generation: String,
    pub(crate) manifest_sha256: String,
    pub(crate) global_step: usize,
    pub(crate) metric_records: u64,
    #[serde(skip_serializing)]
    pub(crate) workflow_signature: String,
}

#[derive(Debug)]
struct SealedGeneration {
    name: String,
    manifest_sha256: String,
    path: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct CheckpointPublication {
    pub(crate) checkpoint_manifest: PathBuf,
    pub(crate) checkpoint_manifest_sha256: String,
}

struct StagingGuard {
    path: PathBuf,
    armed: bool,
}

impl StagingGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

/// A separately clocked optimizer namespace.  The global wake optimizers use
/// `wake`; memory tiers use their MAL tier id.  Paths are relative to the
/// checkpoint root and are content-addressed by the surrounding checkpoint
/// publication.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OptimizerStateRef {
    pub(crate) scope: String,
    pub(crate) adamw: String,
    pub(crate) muon: String,
    pub(crate) gradient_accumulator: Option<String>,
    pub(crate) update_clock: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactRef {
    pub(crate) kind: String,
    pub(crate) manifest: String,
    pub(crate) hash: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RngStreamState {
    pub(crate) name: String,
    pub(crate) seed: u64,
    pub(crate) counter: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QuantizationTrainingState {
    pub(crate) format: String,
    pub(crate) fake_quant_active: bool,
    pub(crate) calibration_step: u64,
    pub(crate) manifest: Option<String>,
    /// Canonical weights identity sealed by `manifest`. It is absent until a
    /// final candidate exists and is checked against this checkpoint's own
    /// authenticated weights file during both save and resume.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub(crate) candidate_weights_sha256: Option<String>,
    pub(crate) teacher_hash: Option<String>,
}

/// Checkpoint-bound memory execution strategy. The wake-only variant carries
/// the exact typed configuration and content-addressed tier optimizer scopes,
/// including every pending gradient accumulator and independent clock.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TrainingMemoryUpdateState {
    Ordinary,
    PeriodicSleep,
    WakeOnly {
        config: MemoryUpdateMode,
        optimizer_scopes: MemoryOptimizerScopes,
    },
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// Complete version-2 trainer state.  The schema intentionally has no serde
/// defaults: a version-1 curriculum checkpoint cannot be mistaken for an
/// exactly resumable WorkflowV2 checkpoint.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainingState {
    pub(crate) version: u32,
    pub(crate) global_step: usize,
    pub(crate) phase: usize,
    pub(crate) phase_id: String,
    pub(crate) phase_kind: String,
    pub(crate) epoch: usize,
    pub(crate) records_in_phase: usize,
    pub(crate) steps_in_phase: usize,
    pub(crate) tokens_seen: usize,
    pub(crate) metric_records: u64,
    pub(crate) workflow_signature: String,
    pub(crate) data_manifest_hash: Option<String>,
    pub(crate) parameter_ids: Vec<u64>,
    pub(crate) optimizer_states: Vec<OptimizerStateRef>,
    pub(crate) memory_update: TrainingMemoryUpdateState,
    pub(crate) sleep: Option<NativeSleepCheckpoint>,
    pub(crate) artifacts: Vec<ArtifactRef>,
    pub(crate) evaluator_hashes: Vec<String>,
    pub(crate) rng_streams: Vec<RngStreamState>,
    /// Bounded, already-tokenized contexts observed since the previous sleep
    /// boundary. Persisting them is required for an exact crash/relaunch just
    /// before the next journal is sealed.
    pub(crate) wake_context_buffer: Vec<WakeContextRecord>,
    pub(crate) quantization: Option<QuantizationTrainingState>,
}

impl TrainingState {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.version == TRAINING_STATE_VERSION,
            "unsupported training-state version {}; this build supports version {TRAINING_STATE_VERSION}",
            self.version
        );
        ensure!(
            !self.phase_id.trim().is_empty(),
            "checkpoint phase_id is empty"
        );
        ensure!(
            !self.phase_kind.trim().is_empty(),
            "checkpoint phase_kind is empty"
        );
        ensure!(
            !self.workflow_signature.trim().is_empty(),
            "checkpoint workflow signature is empty"
        );
        validate_sha256_identity(&self.workflow_signature, "checkpoint workflow signature")?;
        if let Some(hash) = &self.data_manifest_hash {
            validate_sha256_identity(hash, "checkpoint data manifest hash")?;
        }
        ensure!(
            self.global_step == 0 || self.tokens_seen > 0 || self.phase_kind == "evaluation",
            "non-evaluation checkpoint has optimizer progress but no token count"
        );
        ensure!(
            self.global_step == 0 || self.metric_records > 0,
            "checkpoint has optimizer progress but no committed metrics"
        );
        ensure!(
            self.wake_context_buffer.len() <= MAX_WAKE_CONTEXT_RECORDS,
            "checkpoint wake-context buffer exceeds the {MAX_WAKE_CONTEXT_RECORDS}-record limit"
        );
        let checkpoint_step = u64::try_from(self.global_step)
            .context("checkpoint global step exceeds the wake-context clock")?;
        let mut context_ids = BTreeSet::new();
        let mut previous_context_step = None;
        let wake_context_tokens =
            self.wake_context_buffer
                .iter()
                .try_fold(0_usize, |total, record| -> Result<usize> {
                    ensure!(
                        !record.id.trim().is_empty()
                            && record.id.len() <= MAX_WAKE_CONTEXT_ID_BYTES,
                        "checkpoint wake-context id must contain 1..={MAX_WAKE_CONTEXT_ID_BYTES} bytes"
                    );
                    ensure!(
                        context_ids.insert(record.id.as_str()),
                        "checkpoint wake-context buffer repeats identity `{}`",
                        record.id
                    );
                    ensure!(
                        previous_context_step
                            .is_none_or(|previous| previous <= record.optimizer_step)
                            && record.optimizer_step <= checkpoint_step,
                        "checkpoint wake-context buffer has an invalid optimizer-step order"
                    );
                    previous_context_step = Some(record.optimizer_step);
                    ensure!(
                        !record.token_ids.is_empty(),
                        "checkpoint wake context `{}` has no tokens",
                        record.id
                    );
                    ensure!(
                        record.token_ids.len() <= MAX_WAKE_CONTEXT_TOKENS,
                        "checkpoint wake context `{}` exceeds the {MAX_WAKE_CONTEXT_TOKENS}-token limit",
                        record.id
                    );
                    ensure!(
                        record
                            .token_ids
                            .iter()
                            .all(|token| u32::try_from(*token).is_ok()),
                        "checkpoint wake context `{}` contains a token outside the u32 vocabulary range",
                        record.id
                    );
                    total
                        .checked_add(record.token_ids.len())
                        .context("checkpoint wake-context token count overflows usize")
                })?;
        ensure!(
            wake_context_tokens <= MAX_WAKE_CONTEXT_TOTAL_TOKENS,
            "checkpoint wake-context buffer exceeds the {MAX_WAKE_CONTEXT_TOTAL_TOKENS}-token limit"
        );
        let optimizer_scopes = self
            .optimizer_states
            .iter()
            .map(|state| state.scope.as_str())
            .collect::<BTreeSet<_>>();
        ensure!(
            optimizer_scopes.len() == self.optimizer_states.len(),
            "checkpoint repeats an optimizer scope"
        );
        ensure!(
            optimizer_scopes.contains("wake"),
            "checkpoint has no `wake` optimizer scope"
        );
        for state in &self.optimizer_states {
            ensure!(!state.scope.trim().is_empty(), "optimizer scope is empty");
            ensure!(
                !state.adamw.trim().is_empty() && !state.muon.trim().is_empty(),
                "optimizer scope `{}` has an empty state path",
                state.scope
            );
            validate_checkpoint_relative_path(&state.adamw).with_context(|| {
                format!("optimizer scope `{}` has an unsafe AdamW path", state.scope)
            })?;
            validate_checkpoint_relative_path(&state.muon).with_context(|| {
                format!("optimizer scope `{}` has an unsafe Muon path", state.scope)
            })?;
            if let Some(path) = &state.gradient_accumulator {
                validate_checkpoint_relative_path(path).with_context(|| {
                    format!(
                        "optimizer scope `{}` has an unsafe gradient accumulator path",
                        state.scope
                    )
                })?;
            }
            if state.scope == "wake" {
                ensure!(
                    state.adamw == ADAMW_FILE && state.muon == MUON_FILE,
                    "wake optimizer paths do not match the fixed checkpoint schema"
                );
                ensure!(
                    state.update_clock == self.global_step as u64,
                    "wake optimizer clock {} differs from global step {}",
                    state.update_clock,
                    self.global_step
                );
            }
        }
        match &self.memory_update {
            TrainingMemoryUpdateState::Ordinary => ensure!(
                self.sleep.is_none() && self.wake_context_buffer.is_empty(),
                "ordinary checkpoint contains memory-training state"
            ),
            TrainingMemoryUpdateState::PeriodicSleep => ensure!(
                self.sleep.is_some(),
                "periodic_sleep checkpoint has no native sleep cursor"
            ),
            TrainingMemoryUpdateState::WakeOnly {
                config,
                optimizer_scopes,
            } => {
                config.validate("checkpoint wake_only")?;
                ensure!(
                    config.schedule().clock == UpdateClock::OptimizerSteps,
                    "wake_only checkpoint uses a clock unsupported by the stock trainer"
                );
                ensure!(
                    self.sleep.is_none() && self.wake_context_buffer.is_empty(),
                    "wake_only checkpoint contains periodic-sleep state"
                );
                ensure!(
                    optimizer_scopes.tiers.len() == config.schedule().tiers.len()
                        && !optimizer_scopes.wake_parameter_ids.is_empty(),
                    "wake_only checkpoint optimizer topology differs from its schedule"
                );
                let mut owned = optimizer_scopes
                    .wake_parameter_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                ensure!(
                    owned.len() == optimizer_scopes.wake_parameter_ids.len(),
                    "wake_only checkpoint repeats a wake parameter ID"
                );
                let global_clock = u64::try_from(self.global_step)
                    .context("wake_only checkpoint global step exceeds u64")?;
                for (index, (scope, tier)) in optimizer_scopes
                    .tiers
                    .iter()
                    .zip(&config.schedule().tiers)
                    .enumerate()
                {
                    let completed_boundaries = global_clock / tier.update_period;
                    let expected_update_clock = completed_boundaries
                        .checked_mul(tier.update_period)
                        .context("wake_only tier update clock overflows u64")?;
                    let expected_pending_steps = global_clock
                        .checked_sub(expected_update_clock)
                        .context("wake_only pending-step clock underflows")?;
                    let expected_generation = global_clock
                        .checked_add(completed_boundaries)
                        .context("wake_only tier generation overflows u64")?;
                    ensure!(
                        scope.tier == index
                            && scope.tier_id == tier.id
                            && scope.update_clock == expected_update_clock
                            && scope.accumulated_micro_steps == expected_pending_steps
                            && scope.generation == expected_generation
                            && scope.transfer_clock == 0
                            && scope.transfer_generation == 0,
                        "wake_only checkpoint tier `{}` has invalid identity or clocks",
                        tier.id
                    );
                    ensure!(
                        scope.parameter_ids.iter().all(|id| owned.insert(*id)),
                        "wake_only checkpoint repeats a tier parameter ID"
                    );
                    ensure!(
                        scope.artifact.is_some()
                            || (scope.update_clock == 0
                                && scope.accumulated_micro_steps == 0
                                && scope.generation == 0),
                        "wake_only checkpoint tier `{}` has mutable state without an immutable optimizer artifact",
                        tier.id
                    );
                    if let Some(artifact) = &scope.artifact {
                        artifact
                            .validate_pending_steps(scope.accumulated_micro_steps)
                            .with_context(|| {
                                format!(
                                    "wake_only checkpoint tier `{}` has invalid optimizer state",
                                    tier.id
                                )
                            })?;
                    }
                }
                let checkpoint_parameters =
                    self.parameter_ids.iter().copied().collect::<BTreeSet<_>>();
                ensure!(
                    owned == checkpoint_parameters,
                    "wake_only optimizer scopes do not exactly partition checkpoint parameters"
                );
            }
        }
        let optimizer_paths = self
            .optimizer_states
            .iter()
            .flat_map(|state| {
                [&state.adamw, &state.muon]
                    .into_iter()
                    .chain(state.gradient_accumulator.iter())
            })
            .collect::<BTreeSet<_>>();
        ensure!(
            optimizer_paths.len()
                == self
                    .optimizer_states
                    .iter()
                    .map(|state| 2 + usize::from(state.gradient_accumulator.is_some()))
                    .sum::<usize>(),
            "checkpoint optimizer or gradient state paths are not independent"
        );
        let rng_names = self
            .rng_streams
            .iter()
            .map(|stream| stream.name.as_str())
            .collect::<BTreeSet<_>>();
        ensure!(
            rng_names.len() == self.rng_streams.len(),
            "checkpoint repeats an RNG stream"
        );
        for stream in &self.rng_streams {
            ensure!(!stream.name.trim().is_empty(), "RNG stream name is empty");
        }
        let unique_parameter_ids = self.parameter_ids.iter().collect::<BTreeSet<_>>();
        ensure!(
            unique_parameter_ids.len() == self.parameter_ids.len(),
            "checkpoint repeats a parameter ID"
        );
        let mut artifact_references = BTreeSet::new();
        for artifact in &self.artifacts {
            ensure!(
                !artifact.kind.trim().is_empty()
                    && !artifact.manifest.trim().is_empty()
                    && !artifact.hash.trim().is_empty(),
                "checkpoint has an incomplete artifact reference"
            );
            validate_sha256_identity(&artifact.hash, "checkpoint artifact hash")?;
            ensure!(
                artifact_references.insert((artifact.kind.as_str(), artifact.manifest.as_str())),
                "checkpoint repeats an artifact kind and manifest"
            );
        }
        let mut evaluator_hashes = BTreeSet::new();
        for hash in &self.evaluator_hashes {
            validate_sha256_identity(hash, "checkpoint evaluator hash")?;
            ensure!(
                evaluator_hashes.insert(hash.as_str()),
                "checkpoint repeats an evaluator hash"
            );
        }
        if let Some(quantization) = &self.quantization {
            ensure!(
                !quantization.format.trim().is_empty(),
                "checkpoint quantization format is empty"
            );
            if let Some(hash) = &quantization.teacher_hash {
                validate_sha256_identity(hash, "checkpoint quantization teacher hash")?;
            }
            ensure!(
                quantization.manifest.is_some() == quantization.candidate_weights_sha256.is_some(),
                "checkpoint quantization candidate manifest and weights identity must appear together"
            );
            if let Some(hash) = &quantization.candidate_weights_sha256 {
                validate_sha256_identity(hash, "checkpoint quantization candidate weights hash")?;
            }
        }
        if let Some(sleep) = &self.sleep {
            ensure!(
                sleep.workflow_signature == self.workflow_signature,
                "native sleep checkpoint belongs to another workflow"
            );
            ensure!(
                sleep.phase_name == self.phase_id,
                "native sleep checkpoint belongs to phase `{}`, trainer is in `{}`",
                sleep.phase_name,
                self.phase_id
            );
            sleep.input_checkpoint.validate()?;
            sleep.live_checkpoint.validate()?;
            sleep.retention_suite.verify()?;
            sleep.sleep.validate_resume()?;
        }
        Ok(())
    }
}

pub(crate) fn parameter_ids(model: &Transformer) -> Vec<u64> {
    burn::module::list_param_ids(model)
        .into_iter()
        .map(|id| id.val())
        .collect()
}

struct ParameterIdMapper<'a> {
    ids: std::slice::Iter<'a, u64>,
}

impl ParameterIdMapper<'_> {
    fn next(&mut self) -> ParamId {
        ParamId::from(
            self.ids
                .next()
                .copied()
                .expect("checkpoint contains too few parameter IDs"),
        )
    }
}

impl ModuleMapper for ParameterIdMapper<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (_, tensor, mapper) = param.consume();
        Param::from_mapped_value(self.next(), tensor, mapper)
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        let (_, tensor, mapper) = param.consume();
        Param::from_mapped_value(self.next(), tensor, mapper)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        let (_, tensor, mapper) = param.consume();
        Param::from_mapped_value(self.next(), tensor, mapper)
    }
}

fn restore_parameter_ids(model: &mut Transformer, ids: &[u64]) -> Result<()> {
    ensure!(
        ids.len() == burn::module::list_param_ids(model).len(),
        "checkpoint has {} parameter IDs, model has {}",
        ids.len(),
        burn::module::list_param_ids(model).len()
    );
    let mut mapper = ParameterIdMapper { ids: ids.iter() };
    *model = model.clone().map(&mut mapper);
    ensure!(
        mapper.ids.next().is_none(),
        "checkpoint contains too many parameter IDs"
    );
    Ok(())
}

/// Seal a checkpoint, sealing measured resource accounting from the exact
/// model and committed metric prefix inside it, and only then advance
/// `current.json`.
pub(crate) fn save_training_checkpoint_with_evidence(
    model: &Transformer,
    adamw: &AdamWOptimizer,
    muon: &BatchedMuon,
    state: &TrainingState,
    metrics: &mut MetricWriter,
    output: &Path,
) -> Result<CheckpointPublication> {
    let measured_time =
        metrics.committed_training_time(state.metric_records, state.global_step as u64)?;
    let accounting = model.wake_parameter_accounting()?;
    let accounting_input = TrainingAccountingInput {
        training_gpu_hours: measured_time.single_accelerator_hours(),
        parameters: accounting.stored_parameters,
        routed_active_parameters: accounting.routed_active_parameters,
    };
    let (sealed, _) =
        seal_training_checkpoint(model, adamw, muon, state, accounting_input, output)?;
    verify_generation(&sealed.path, &sealed.name, &sealed.manifest_sha256)?;
    publish_current(output, &sealed)?;
    Ok(CheckpointPublication {
        checkpoint_manifest: sealed.path.join(GENERATION_MANIFEST),
        checkpoint_manifest_sha256: sealed.manifest_sha256,
    })
}

fn seal_training_checkpoint(
    model: &Transformer,
    adamw: &AdamWOptimizer,
    muon: &BatchedMuon,
    state: &TrainingState,
    accounting_input: TrainingAccountingInput,
    output: &Path,
) -> Result<(SealedGeneration, TrainingAccounting)> {
    state.validate()?;
    ensure!(
        state.parameter_ids == parameter_ids(model),
        "checkpoint parameter IDs do not match the model"
    );
    fs::create_dir_all(output)?;
    let staging = create_staging_directory(output)?;
    let mut staging_guard = StagingGuard::new(staging.clone());
    let weights = staging.join(WEIGHTS_FILE);
    let adamw_state = staging.join(ADAMW_FILE);
    let muon_state = staging.join(MUON_FILE);

    save_safetensors(&model.clone().valid(), &weights)?;
    sync_regular_file(&weights, "checkpoint weights")?;
    let (weights_bytes, weights_sha256) = hash_file(&weights)?;
    validate_quantization_weights_binding(state, &weights_sha256)?;
    let accounting = TrainingAccounting {
        version: TRAINING_ACCOUNTING_VERSION,
        training_gpu_hours: accounting_input.training_gpu_hours,
        parameters: accounting_input.parameters,
        routed_active_parameters: accounting_input.routed_active_parameters,
        weights_bytes,
        weights_sha256,
    };
    accounting.validate()?;
    write_synced_new(
        &staging.join(TRAINING_ACCOUNTING_FILE),
        &serde_json::to_vec(&accounting)?,
    )?;
    save_canonical_module_optimizer(adamw, &adamw_state).context("failed to save AdamW state")?;
    sync_regular_file(&adamw_state, "checkpoint AdamW state")?;
    muon.save(&muon_state)?;
    sync_regular_file(&muon_state, "checkpoint Muon state")?;
    write_synced_new(
        &staging.join(TRAINING_STATE_FILE),
        &serde_json::to_vec_pretty(state)?,
    )?;

    let sealed = seal_generation(output, &staging)?;
    staging_guard.disarm();
    Ok((sealed, accounting))
}

#[derive(Clone, Copy, Debug)]
struct TrainingAccountingInput {
    training_gpu_hours: f64,
    parameters: u64,
    routed_active_parameters: u64,
}

fn validate_checkpoint_relative_path(path: &str) -> Result<()> {
    ensure!(!path.trim().is_empty(), "checkpoint path is empty");
    ensure!(
        !path.chars().any(|character| {
            matches!(character, '\\' | ':' | '*' | '?' | '[' | ']')
                || character.is_control()
                || character == '\u{7f}'
        }),
        "checkpoint path `{path}` is not portable"
    );
    let path = Path::new(path);
    ensure!(!path.is_absolute(), "checkpoint path must be relative");
    let mut components = 0;
    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "checkpoint path must not contain prefixes, `.` or `..`"
        );
        components += 1;
    }
    ensure!(components > 0, "checkpoint path has no file name");
    Ok(())
}

fn create_staging_directory(output: &Path) -> Result<PathBuf> {
    let generations = output.join(GENERATIONS_DIRECTORY);
    fs::create_dir_all(&generations).with_context(|| {
        format!(
            "failed to create checkpoint generation root {}",
            generations.display()
        )
    })?;
    validate_generation_root(&generations)?;
    sync_directory(output)?;
    for _ in 0..128 {
        let path = generations.join(format!(".staging-{}", unique_suffix()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to create checkpoint staging directory {}",
                        path.display()
                    )
                });
            }
        }
    }
    anyhow::bail!("failed to allocate a unique checkpoint staging directory")
}

fn validate_generation_root(path: &Path) -> Result<()> {
    ensure_real_directory(path, "checkpoint generation root")
}

fn unique_suffix() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("{}-{timestamp}-{sequence}", std::process::id())
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    hash_regular_file_hex(path)
}

fn read_file_stable(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_regular_bounded(path, MAX_CHECKPOINT_METADATA_BYTES, label)
}

impl GenerationSnapshot {
    fn new() -> Self {
        Self {
            files: Vec::new(),
            manifest: None,
            training_state: None,
            training_accounting: None,
            access: None,
        }
    }

    fn record_file(
        &mut self,
        relative: String,
        bytes: u64,
        sha256: String,
        captured: Option<Vec<u8>>,
    ) -> Result<()> {
        match relative.as_str() {
            GENERATION_MANIFEST => {
                ensure!(
                    self.manifest.is_none(),
                    "checkpoint repeats its generation manifest"
                );
                self.manifest = captured;
                return Ok(());
            }
            TRAINING_STATE_FILE => self.training_state = captured,
            TRAINING_ACCOUNTING_FILE => self.training_accounting = captured,
            _ => {}
        }
        validate_checkpoint_relative_path(&relative)?;
        self.files.push(GenerationFile {
            path: relative,
            bytes,
            sha256,
        });
        Ok(())
    }

    fn finish(mut self) -> Result<Self> {
        self.files.sort_by(|left, right| left.path.cmp(&right.path));
        ensure!(
            self.files
                .windows(2)
                .all(|pair| pair[0].path != pair[1].path),
            "checkpoint generation repeats a file path"
        );
        Ok(self)
    }
}

impl GenerationAccess {
    fn read_authenticated(&mut self, relative: &str, expected: &GenerationFile) -> Result<Vec<u8>> {
        ensure!(
            expected.path == relative,
            "checkpoint artifact identity does not match `{relative}`"
        );
        #[cfg(unix)]
        let file = self
            .directory
            .open_relative_file(Path::new(relative), "checkpoint generation")?;
        #[cfg(not(unix))]
        let file = self
            .files
            .remove(relative)
            .with_context(|| format!("checkpoint snapshot did not retain `{relative}`"))?;
        read_authenticated_file(file, expected)
    }
}

fn read_authenticated_file(mut file: File, expected: &GenerationFile) -> Result<Vec<u8>> {
    let (bytes, sha256, captured) = hash_open_file(
        &mut file,
        Some(MAX_CHECKPOINT_MEMBER_BYTES),
        &format!("authenticated checkpoint artifact `{}`", expected.path),
    )?;
    ensure!(
        bytes == expected.bytes && sha256 == expected.sha256,
        "checkpoint artifact `{}` no longer matches the authenticated generation manifest",
        expected.path
    );
    captured.context("authenticated checkpoint read did not capture bytes")
}

#[cfg(unix)]
fn collect_generation_snapshot(root: &Path, retain_access: bool) -> Result<GenerationSnapshot> {
    fn visit(
        directory: &PinnedDirectory,
        relative_root: &Path,
        snapshot: &mut GenerationSnapshot,
        budget: &mut GenerationTraversalBudget,
        depth: usize,
    ) -> Result<()> {
        budget.enter_directory(depth)?;
        let before = directory.identity()?;
        for name in directory.entries("checkpoint generation")? {
            let relative_path = relative_root.join(&name);
            let relative = relative_path
                .to_str()
                .context("checkpoint file name is not valid UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            validate_checkpoint_relative_path(&relative)?;
            let mut child = directory
                .open_child(&name, "checkpoint generation")
                .with_context(|| {
                    format!("failed to open checkpoint generation entry `{relative}`")
                })?;
            let metadata = child.metadata()?;
            if metadata.is_dir() {
                let files_before = snapshot.files.len();
                let child =
                    PinnedDirectory::from_open_directory(child, "checkpoint generation child")?;
                let child_depth = depth
                    .checked_add(1)
                    .context("checkpoint directory depth overflow")?;
                visit(&child, &relative_path, snapshot, budget, child_depth)?;
                ensure!(
                    snapshot.files.len() > files_before,
                    "checkpoint generation contains empty directory `{relative}`"
                );
                continue;
            }
            ensure!(
                metadata.is_file(),
                "checkpoint generation contains non-file `{relative}`"
            );
            budget.record_file(metadata.len())?;
            let capture = matches!(
                relative.as_str(),
                GENERATION_MANIFEST | TRAINING_STATE_FILE | TRAINING_ACCOUNTING_FILE
            );
            let (bytes, sha256, captured) = hash_open_file(
                &mut child,
                capture.then_some(MAX_CHECKPOINT_METADATA_BYTES),
                &format!("checkpoint file `{relative}`"),
            )?;
            snapshot.record_file(relative, bytes, sha256, captured)?;
        }
        ensure!(
            directory.identity()? == before,
            "checkpoint directory changed while it was verified"
        );
        Ok(())
    }

    let (directory, root_identity) = PinnedDirectory::open(root, "checkpoint generation")?;
    let mut snapshot = GenerationSnapshot::new();
    let mut budget = GenerationTraversalBudget::default();
    visit(&directory, Path::new(""), &mut snapshot, &mut budget, 0)?;
    let current = fs::symlink_metadata(root).with_context(|| {
        format!(
            "failed to reinspect checkpoint generation {}",
            root.display()
        )
    })?;
    ensure!(
        current.is_dir()
            && !current.file_type().is_symlink()
            && StableIdentity::from_metadata(&current) == root_identity,
        "checkpoint generation changed while it was verified"
    );
    if retain_access {
        snapshot.access = Some(GenerationAccess { directory });
    }
    snapshot.finish()
}

#[cfg(not(unix))]
fn collect_generation_snapshot(root: &Path, retain_access: bool) -> Result<GenerationSnapshot> {
    fn visit(
        root: &Path,
        directory: &Path,
        snapshot: &mut GenerationSnapshot,
        retained: &mut Option<BTreeMap<String, File>>,
        budget: &mut GenerationTraversalBudget,
        depth: usize,
    ) -> Result<()> {
        budget.enter_directory(depth)?;
        let before = fs::metadata(directory)?;
        let mut entries = Vec::new();
        for entry in fs::read_dir(directory)? {
            entries.push(entry?);
            ensure!(
                entries.len()
                    <= MAX_CHECKPOINT_GENERATION_FILES + MAX_CHECKPOINT_GENERATION_DIRECTORIES,
                "checkpoint directory contains too many entries"
            );
        }
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "checkpoint generation contains symlink {}",
                path.display()
            );
            if metadata.is_dir() {
                let files_before = snapshot.files.len();
                let child_depth = depth
                    .checked_add(1)
                    .context("checkpoint directory depth overflow")?;
                visit(root, &path, snapshot, retained, budget, child_depth)?;
                ensure!(
                    snapshot.files.len() > files_before,
                    "checkpoint generation contains empty directory {}",
                    path.display()
                );
                continue;
            }
            ensure!(
                metadata.is_file(),
                "checkpoint generation contains non-file {}",
                path.display()
            );
            budget.record_file(metadata.len())?;
            let relative = path
                .strip_prefix(root)?
                .to_str()
                .context("checkpoint file name is not valid UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            validate_checkpoint_relative_path(&relative)?;
            let capture = matches!(
                relative.as_str(),
                GENERATION_MANIFEST | TRAINING_STATE_FILE | TRAINING_ACCOUNTING_FILE
            );
            let mut file = File::open(&path)?;
            let (bytes, sha256, captured) = hash_open_file(
                &mut file,
                capture.then_some(MAX_CHECKPOINT_METADATA_BYTES),
                &format!("checkpoint file `{relative}`"),
            )?;
            if let Some(retained) = retained {
                ensure!(
                    retained.insert(relative.clone(), file).is_none(),
                    "checkpoint repeats file `{relative}`"
                );
            }
            snapshot.record_file(relative, bytes, sha256, captured)?;
        }
        let after = fs::metadata(directory)?;
        ensure!(
            before.len() == after.len() && before.modified().ok() == after.modified().ok(),
            "checkpoint directory changed while it was verified"
        );
        Ok(())
    }

    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "checkpoint generation {} is not a real directory",
        root.display()
    );
    let mut snapshot = GenerationSnapshot::new();
    let mut retained = retain_access.then(BTreeMap::new);
    let mut budget = GenerationTraversalBudget::default();
    visit(root, root, &mut snapshot, &mut retained, &mut budget, 0)?;
    snapshot.access = retained.map(|files| GenerationAccess { files });
    snapshot.finish()
}

fn parse_training_state(bytes: &[u8]) -> Result<TrainingState> {
    let state: TrainingState = serde_json::from_slice(bytes)?;
    state.validate()?;
    Ok(state)
}

fn optimizer_file_paths(state: &TrainingState) -> impl Iterator<Item = &str> {
    state.optimizer_states.iter().flat_map(|optimizer| {
        [optimizer.adamw.as_str(), optimizer.muon.as_str()]
            .into_iter()
            .chain(optimizer.gradient_accumulator.as_deref())
    })
}

fn validate_optimizer_files(state: &TrainingState, files: &[GenerationFile]) -> Result<()> {
    let authenticated = files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<BTreeSet<_>>();
    for relative in optimizer_file_paths(state) {
        validate_checkpoint_relative_path(relative)?;
        ensure!(
            authenticated.contains(relative),
            "optimizer or gradient state `{relative}` is missing from the content-addressed generation"
        );
    }
    Ok(())
}

fn ensure_required_files(files: &[GenerationFile]) -> Result<()> {
    let paths = files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        WEIGHTS_FILE,
        ADAMW_FILE,
        MUON_FILE,
        TRAINING_STATE_FILE,
        TRAINING_ACCOUNTING_FILE,
    ] {
        ensure!(
            paths.contains(required),
            "checkpoint generation is missing required file `{required}`"
        );
    }
    Ok(())
}

fn validate_training_accounting(bytes: &[u8], files: &[GenerationFile]) -> Result<()> {
    let accounting: TrainingAccounting =
        serde_json::from_slice(bytes).context("checkpoint training accounting is invalid")?;
    accounting.validate()?;

    let weights = files
        .iter()
        .find(|file| file.path == WEIGHTS_FILE)
        .context("checkpoint manifest has no model weights")?;
    ensure!(
        accounting.weights_bytes == weights.bytes && accounting.weights_sha256 == weights.sha256,
        "checkpoint training accounting does not match its model weights"
    );
    Ok(())
}

fn validate_quantization_weights_binding(
    state: &TrainingState,
    checkpoint_weights_sha256: &str,
) -> Result<()> {
    validate_sha256_hex(checkpoint_weights_sha256, "checkpoint model-weights digest")?;
    if let Some(expected) = state
        .quantization
        .as_ref()
        .and_then(|quantization| quantization.candidate_weights_sha256.as_deref())
    {
        ensure!(
            expected == format!("sha256:{checkpoint_weights_sha256}"),
            "quantization candidate weights differ from checkpoint model weights"
        );
    }
    Ok(())
}

fn validate_generation_name(name: &str) -> Result<&str> {
    let digest = name
        .strip_prefix("sha256-")
        .context("checkpoint generation is not content-addressed")?;
    validate_sha256_hex(digest, "checkpoint generation digest")?;
    Ok(digest)
}

fn seal_generation(output: &Path, staging: &Path) -> Result<SealedGeneration> {
    let GenerationSnapshot {
        files,
        manifest,
        training_state,
        training_accounting,
        access: _,
    } = collect_generation_snapshot(staging, false)?;
    ensure!(
        manifest.is_none(),
        "checkpoint staging directory already contains a generation manifest"
    );
    let state = parse_training_state(
        training_state
            .as_deref()
            .context("checkpoint generation is missing training-state.json")?,
    )?;
    ensure_required_files(&files)?;
    validate_training_accounting(
        training_accounting
            .as_deref()
            .context("checkpoint has no training accounting")?,
        &files,
    )?;
    validate_optimizer_files(&state, &files)?;
    let manifest = GenerationManifest {
        version: CHECKPOINT_MANIFEST_VERSION,
        training_state_version: state.version,
        global_step: state.global_step,
        phase: state.phase,
        phase_id: state.phase_id.clone(),
        files,
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let name = format!("sha256-{manifest_sha256}");
    write_synced_new(&staging.join(GENERATION_MANIFEST), &manifest_bytes)?;
    sync_directory(staging)?;

    let generations = output.join(GENERATIONS_DIRECTORY);
    validate_generation_root(&generations)?;
    let destination = generations.join(&name);
    match fs::symlink_metadata(&destination) {
        Ok(_) => {
            let (existing_manifest, _) = verify_generation(&destination, &name, &manifest_sha256)?;
            ensure!(
                existing_manifest == manifest,
                "content-addressed generation collision for `{name}`"
            );
            fs::remove_dir_all(staging)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match fs::rename(staging, &destination) {
                Ok(()) => {}
                Err(error) => {
                    if fs::symlink_metadata(&destination).is_ok() {
                        let (existing_manifest, _) =
                            verify_generation(&destination, &name, &manifest_sha256)?;
                        ensure!(
                            existing_manifest == manifest,
                            "content-addressed generation collision for `{name}`"
                        );
                        fs::remove_dir_all(staging)?;
                        sync_directory(&generations)?;
                        return Ok(SealedGeneration {
                            name,
                            manifest_sha256,
                            path: destination,
                        });
                    }
                    return Err(error).with_context(|| {
                        format!(
                            "failed to publish checkpoint generation {}",
                            destination.display()
                        )
                    });
                }
            }
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect checkpoint generation {}",
                    destination.display()
                )
            });
        }
    }
    sync_directory(&generations)?;
    Ok(SealedGeneration {
        name,
        manifest_sha256,
        path: destination,
    })
}

fn publish_current(output: &Path, generation: &SealedGeneration) -> Result<()> {
    let generations = output.join(GENERATIONS_DIRECTORY);
    validate_generation_root(&generations)?;
    ensure!(
        generation.path == generations.join(&generation.name),
        "checkpoint generation is outside its output root"
    );
    let generation_digest = validate_generation_name(&generation.name)?;
    ensure!(
        generation_digest == generation.manifest_sha256,
        "checkpoint generation name and manifest digest differ"
    );
    let pointer = CurrentCheckpoint {
        version: CHECKPOINT_POINTER_VERSION,
        generation: generation.name.clone(),
        manifest_sha256: generation.manifest_sha256.clone(),
    };
    let temporary = output.join(format!(".current-{}.tmp", unique_suffix()));
    let publication = (|| -> Result<()> {
        write_synced_new(&temporary, &serde_json::to_vec(&pointer)?)?;
        fs::rename(&temporary, output.join(CURRENT_POINTER))
            .context("failed to atomically publish the current checkpoint pointer")?;
        sync_directory(output)?;
        Ok(())
    })();
    if publication.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    publication
}

fn verify_generation(
    generation: &Path,
    generation_name: &str,
    expected_manifest_sha256: &str,
) -> Result<(GenerationManifest, TrainingState)> {
    let (manifest, state, _) =
        verify_generation_inner(generation, generation_name, expected_manifest_sha256, false)?;
    Ok((manifest, state))
}

fn verify_generation_inner(
    generation: &Path,
    generation_name: &str,
    expected_manifest_sha256: &str,
    retain_access: bool,
) -> Result<(GenerationManifest, TrainingState, Option<GenerationAccess>)> {
    let generation_digest = validate_generation_name(generation_name)?;
    validate_sha256_hex(expected_manifest_sha256, "checkpoint manifest digest")?;
    ensure!(
        generation_digest == expected_manifest_sha256,
        "checkpoint generation name and manifest digest differ"
    );
    let GenerationSnapshot {
        files,
        manifest,
        training_state,
        training_accounting,
        access,
    } = collect_generation_snapshot(generation, retain_access)
        .with_context(|| format!("failed to verify checkpoint generation `{generation_name}`"))?;
    let manifest_bytes = manifest.context("checkpoint generation has no manifest")?;
    ensure!(
        sha256_hex(&manifest_bytes) == expected_manifest_sha256,
        "checkpoint generation manifest digest mismatch"
    );
    let manifest: GenerationManifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest.version == CHECKPOINT_MANIFEST_VERSION,
        "unsupported checkpoint generation manifest version {}",
        manifest.version
    );
    ensure!(
        manifest.training_state_version == TRAINING_STATE_VERSION,
        "checkpoint generation contains training-state version {}",
        manifest.training_state_version
    );
    ensure!(
        files == manifest.files,
        "checkpoint generation contents do not match its manifest"
    );
    ensure_required_files(&manifest.files)?;
    validate_training_accounting(
        training_accounting
            .as_deref()
            .context("checkpoint has no training accounting")?,
        &manifest.files,
    )?;
    let state = parse_training_state(
        training_state
            .as_deref()
            .context("checkpoint generation is missing training-state.json")?,
    )?;
    let checkpoint_weights = manifest
        .files
        .iter()
        .find(|file| file.path == WEIGHTS_FILE)
        .context("checkpoint generation has no model weights")?;
    validate_quantization_weights_binding(&state, &checkpoint_weights.sha256)?;
    ensure!(
        state.version == manifest.training_state_version
            && state.global_step == manifest.global_step
            && state.phase == manifest.phase
            && state.phase_id == manifest.phase_id,
        "checkpoint training state does not match its generation manifest"
    );
    validate_optimizer_files(&state, &manifest.files)?;
    ensure!(
        access.is_some() == retain_access,
        "checkpoint verifier did not retain the requested generation access"
    );
    Ok((manifest, state, access))
}

pub(crate) fn verify_checkpoint_generation(
    generation: &Path,
    generation_name: &str,
    expected_manifest_sha256: &str,
) -> Result<VerifiedCheckpoint> {
    let (_, state) = verify_generation(generation, generation_name, expected_manifest_sha256)?;
    Ok(VerifiedCheckpoint {
        version: 1,
        generation: generation_name.to_owned(),
        manifest_sha256: expected_manifest_sha256.to_owned(),
        global_step: state.global_step,
        metric_records: state.metric_records,
        workflow_signature: state.workflow_signature,
    })
}

pub(crate) fn verify_checkpoint_root(output: &Path) -> Result<VerifiedCheckpoint> {
    let (verified, _, _) = verify_checkpoint_root_with_state(output)?;
    Ok(verified)
}

fn verify_checkpoint_root_with_state(
    output: &Path,
) -> Result<(VerifiedCheckpoint, PathBuf, TrainingState)> {
    let (verified, generation, _, state, _) = verify_checkpoint_root_inner(output, false)?;
    Ok((verified, generation, state))
}

fn verify_checkpoint_root_inner(
    output: &Path,
    retain_access: bool,
) -> Result<(
    VerifiedCheckpoint,
    PathBuf,
    GenerationManifest,
    TrainingState,
    Option<GenerationAccess>,
)> {
    validate_generation_root(output).context("checkpoint root is not a real directory")?;
    let pointer_path = output.join(CURRENT_POINTER);
    let pointer_bytes = read_file_stable(&pointer_path, "checkpoint current pointer").with_context(|| {
        format!(
            "checkpoint has no atomic `{CURRENT_POINTER}` pointer; a version-2 generation checkpoint is required"
        )
    })?;
    let pointer: CurrentCheckpoint = serde_json::from_slice(&pointer_bytes)?;
    ensure!(
        pointer.version == CHECKPOINT_POINTER_VERSION,
        "unsupported checkpoint pointer version {}",
        pointer.version
    );
    validate_generation_name(&pointer.generation)?;
    validate_sha256_hex(&pointer.manifest_sha256, "checkpoint manifest digest")?;
    let generations = output.join(GENERATIONS_DIRECTORY);
    validate_generation_root(&generations)?;
    let generation = generations.join(&pointer.generation);
    let (manifest, state, access) = verify_generation_inner(
        &generation,
        &pointer.generation,
        &pointer.manifest_sha256,
        retain_access,
    )?;
    let verified = VerifiedCheckpoint {
        version: 1,
        generation: pointer.generation,
        manifest_sha256: pointer.manifest_sha256,
        global_step: state.global_step,
        metric_records: state.metric_records,
        workflow_signature: state.workflow_signature.clone(),
    };
    Ok((verified, generation, manifest, state, access))
}

#[cfg(test)]
fn resolve_current_generation(output: &Path) -> Result<(PathBuf, TrainingState)> {
    let (_, generation, state) = verify_checkpoint_root_with_state(output)?;
    Ok((generation, state))
}

fn read_manifest_artifact(
    access: &mut GenerationAccess,
    manifest: &GenerationManifest,
    relative: &str,
) -> Result<Vec<u8>> {
    let expected = manifest
        .files
        .iter()
        .find(|file| file.path == relative)
        .with_context(|| format!("checkpoint manifest has no artifact `{relative}`"))?;
    access.read_authenticated(relative, expected)
}

/// Load one authenticated generation and return its optimizer, typed state,
/// and canonical `sha256:<hex>` model-weights identity. The identity is read
/// from the generation manifest whose exact weights bytes were verified and
/// loaded; callers can bind external artifacts without reserializing `model`.
pub(crate) fn load_training_state(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
) -> Result<(AdamWOptimizer, TrainingState, String)> {
    load_training_state_inner(model, adamw, muon, output, device, |_, _| Ok(()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResumeLoadStage {
    AfterInitialVerify,
    AfterStagedLoad,
}

fn load_training_state_inner(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
    mut stage_hook: impl FnMut(ResumeLoadStage, &Path) -> Result<()>,
) -> Result<(AdamWOptimizer, TrainingState, String)> {
    let (verified_before, generation, manifest, state, access) =
        verify_checkpoint_root_inner(output, true)?;
    let mut access = access.context("checkpoint verifier did not retain generation access")?;
    stage_hook(ResumeLoadStage::AfterInitialVerify, &generation)?;

    let mut loaded_model = model.clone();
    let mut loaded_muon = muon.clone();
    restore_parameter_ids(&mut loaded_model, &state.parameter_ids)?;
    let weights_identity = manifest
        .files
        .iter()
        .find(|file| file.path == WEIGHTS_FILE)
        .context("checkpoint manifest has no authenticated model weights")?;
    let weights_sha256 = format!("sha256:{}", weights_identity.sha256);
    let weights = read_manifest_artifact(&mut access, &manifest, WEIGHTS_FILE)?;
    load_safetensors_bytes(
        &mut loaded_model,
        weights,
        "authenticated training checkpoint weights",
    )?;
    let wake = state
        .optimizer_states
        .iter()
        .find(|optimizer| optimizer.scope == "wake")
        .context("checkpoint has no `wake` optimizer state")?;
    loaded_muon.set_parameter_ids(loaded_model.muon_parameter_ids());
    let muon_bytes = read_manifest_artifact(&mut access, &manifest, &wake.muon)?;
    loaded_muon.load_bytes(muon_bytes, &device.clone().inner())?;
    ensure!(
        state.global_step == 0 || !loaded_muon.is_empty(),
        "Muon checkpoint has no velocities after optimizer progress"
    );
    let adamw_bytes = read_manifest_artifact(&mut access, &manifest, &wake.adamw)?;
    let loaded_adamw = adamw
        .from_bytes(Bytes::from_bytes_vec(adamw_bytes))
        .context("failed to load AdamW state")?;
    let expected_adamw = manifest
        .files
        .iter()
        .find(|file| file.path == wake.adamw)
        .context("checkpoint manifest has no authenticated AdamW state")?;
    let canonical_adamw = canonical_module_optimizer_bytes(&loaded_adamw)
        .context("failed to validate loaded AdamW state")?;
    ensure!(
        canonical_adamw.len() as u64 == expected_adamw.bytes
            && sha256_hex(&canonical_adamw) == expected_adamw.sha256,
        "loaded AdamW state is not a lossless reconstruction of its authenticated artifact"
    );
    stage_hook(ResumeLoadStage::AfterStagedLoad, &generation)?;

    let (verified_after, _, _) = verify_checkpoint_root_with_state(output)?;
    ensure!(
        verified_after == verified_before,
        "checkpoint generation changed while its state was being loaded"
    );

    *model = loaded_model;
    *muon = loaded_muon;
    Ok((loaded_adamw, state, weights_sha256))
}

#[cfg(test)]
pub(crate) fn load_training_state_with_hook(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
    stage_hook: impl FnMut(ResumeLoadStage, &Path) -> Result<()>,
) -> Result<(AdamWOptimizer, TrainingState, String)> {
    load_training_state_inner(model, adamw, muon, output, device, stage_hook)
}

#[cfg(test)]
pub(crate) fn rewrite_current_generation_for_test(
    output: &Path,
    rewrite: impl FnOnce(&Path) -> Result<()>,
) -> Result<PathBuf> {
    let (_, source, manifest, _, _) = verify_checkpoint_root_inner(output, false)?;
    let staging = create_staging_directory(output)?;
    let mut guard = StagingGuard::new(staging.clone());
    for artifact in &manifest.files {
        let destination = staging.join(&artifact.path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source.join(&artifact.path), &destination)
            .with_context(|| format!("failed to copy test artifact `{}`", artifact.path))?;
    }
    rewrite(&staging)?;
    let sealed = seal_generation(output, &staging)?;
    guard.disarm();
    publish_current(output, &sealed)?;
    Ok(sealed.path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_at(global_step: usize) -> TrainingState {
        TrainingState {
            version: TRAINING_STATE_VERSION,
            global_step,
            phase: 1,
            phase_id: "continued-pretrain".into(),
            phase_kind: "continued_pretrain".into(),
            epoch: 0,
            records_in_phase: 12,
            steps_in_phase: global_step,
            tokens_seen: 1024,
            metric_records: global_step as u64 * 2,
            workflow_signature: format!("sha256:{}", "0".repeat(64)),
            data_manifest_hash: Some(format!("sha256:{}", "1".repeat(64))),
            parameter_ids: vec![1, 2],
            optimizer_states: vec![OptimizerStateRef {
                scope: "wake".into(),
                adamw: ADAMW_FILE.into(),
                muon: MUON_FILE.into(),
                gradient_accumulator: None,
                update_clock: global_step as u64,
            }],
            memory_update: TrainingMemoryUpdateState::Ordinary,
            sleep: None,
            artifacts: vec![],
            evaluator_hashes: vec![format!("sha256:{}", "2".repeat(64))],
            rng_streams: vec![RngStreamState {
                name: "data".into(),
                seed: 42,
                counter: 12,
            }],
            wake_context_buffer: Vec::new(),
            quantization: None,
        }
    }

    fn wake_only_state() -> TrainingState {
        let mut state = state_at(3);
        state.parameter_ids = vec![1, 2, 3];
        let config: MemoryUpdateMode = serde_json::from_value(serde_json::json!({
            "type": "wake_only",
            "schedule": {
                "clock": "optimizer_steps",
                "terminal_consolidation": "distill_into_base_v1",
                "tiers": [
                    {"id": "fast", "update_period": 1, "reserve_slots": 1},
                    {"id": "slow", "update_period": 2, "reserve_slots": 2}
                ]
            }
        }))
        .unwrap();
        let optimizer_scopes: MemoryOptimizerScopes = serde_json::from_value(serde_json::json!({
            "wake_parameter_ids": [1],
            "tiers": [
                {
                    "tier": 0,
                    "tier_id": "fast",
                    "parameter_ids": [2],
                    "update_clock": 3,
                    "transfer_clock": 0,
                    "accumulated_micro_steps": 0,
                    "generation": 6,
                    "transfer_generation": 0,
                    "artifact": {
                        "state_uri": "fast/manifest.json",
                        "manifest_hash": format!("sha256:{}", "a".repeat(64)),
                        "optimizer_parameter_ids": [2],
                        "accumulator_parameter_ids": []
                    }
                },
                {
                    "tier": 1,
                    "tier_id": "slow",
                    "parameter_ids": [3],
                    "update_clock": 2,
                    "transfer_clock": 0,
                    "accumulated_micro_steps": 1,
                    "generation": 4,
                    "transfer_generation": 0,
                    "artifact": {
                        "state_uri": "slow/manifest.json",
                        "manifest_hash": format!("sha256:{}", "b".repeat(64)),
                        "optimizer_parameter_ids": [3],
                        "accumulator_parameter_ids": [3]
                    }
                }
            ]
        }))
        .unwrap();
        state.memory_update = TrainingMemoryUpdateState::WakeOnly {
            config,
            optimizer_scopes,
        };
        state
    }

    #[test]
    fn wake_only_checkpoint_requires_exact_pending_accumulator_receipts() {
        let valid = wake_only_state();
        valid.validate().unwrap();

        let mut missing_artifact = valid.clone();
        let TrainingMemoryUpdateState::WakeOnly {
            optimizer_scopes, ..
        } = &mut missing_artifact.memory_update
        else {
            unreachable!()
        };
        optimizer_scopes.tiers[1].artifact = None;
        assert!(missing_artifact.validate().is_err());

        let mut missing_gradients = valid.clone();
        let TrainingMemoryUpdateState::WakeOnly {
            optimizer_scopes, ..
        } = &mut missing_gradients.memory_update
        else {
            unreachable!()
        };
        optimizer_scopes.tiers[1]
            .artifact
            .as_mut()
            .unwrap()
            .accumulator_parameter_ids
            .clear();
        assert!(missing_gradients.validate().is_err());

        let mut unexpected_gradients = valid;
        let TrainingMemoryUpdateState::WakeOnly {
            optimizer_scopes, ..
        } = &mut unexpected_gradients.memory_update
        else {
            unreachable!()
        };
        optimizer_scopes.tiers[1].accumulated_micro_steps = 0;
        assert!(unexpected_gradients.validate().is_err());

        let mut stale_boundary = wake_only_state();
        let TrainingMemoryUpdateState::WakeOnly {
            optimizer_scopes, ..
        } = &mut stale_boundary.memory_update
        else {
            unreachable!()
        };
        optimizer_scopes.tiers[0].update_clock = 2;
        optimizer_scopes.tiers[0].accumulated_micro_steps = 1;
        assert!(stale_boundary.validate().is_err());

        let mut stale_generation = wake_only_state();
        let TrainingMemoryUpdateState::WakeOnly {
            optimizer_scopes, ..
        } = &mut stale_generation.memory_update
        else {
            unreachable!()
        };
        optimizer_scopes.tiers[1].generation -= 1;
        assert!(stale_generation.validate().is_err());
    }

    fn stage_test_generation(output: &Path, state: &TrainingState, tag: &str) -> PathBuf {
        fs::create_dir_all(output).unwrap();
        let staging = create_staging_directory(output).unwrap();
        write_synced_new(
            &staging.join(WEIGHTS_FILE),
            format!("weights-{tag}").as_bytes(),
        )
        .unwrap();
        let (weights_bytes, weights_sha256) = hash_file(&staging.join(WEIGHTS_FILE)).unwrap();
        let accounting = TrainingAccounting {
            version: TRAINING_ACCOUNTING_VERSION,
            training_gpu_hours: 1.0,
            parameters: 100,
            routed_active_parameters: 80,
            weights_bytes,
            weights_sha256,
        };
        write_synced_new(
            &staging.join(TRAINING_ACCOUNTING_FILE),
            &serde_json::to_vec(&accounting).unwrap(),
        )
        .unwrap();
        write_synced_new(&staging.join(ADAMW_FILE), format!("adamw-{tag}").as_bytes()).unwrap();
        write_synced_new(&staging.join(MUON_FILE), format!("muon-{tag}").as_bytes()).unwrap();
        write_synced_new(
            &staging.join(TRAINING_STATE_FILE),
            &serde_json::to_vec(state).unwrap(),
        )
        .unwrap();
        staging
    }

    #[test]
    fn version_one_training_state_is_rejected_without_a_fallback() {
        let version_one = r#"{
            "version": 1,
            "step": 13500,
            "stage": 0,
            "epoch": 0,
            "samples_in_stage": 1782000,
            "parameter_ids": []
        }"#;

        assert!(serde_json::from_str::<TrainingState>(version_one).is_err());
    }

    #[test]
    fn v2_state_requires_all_exact_resume_fields() {
        let state = state_at(7);
        state.validate().unwrap();
        let json = serde_json::to_vec(&state).unwrap();
        let restored: TrainingState = serde_json::from_slice(&json).unwrap();
        restored.validate().unwrap();

        let mut duplicate = restored;
        duplicate.rng_streams.push(duplicate.rng_streams[0].clone());
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn checkpoint_bounds_wake_context_records_and_tokens_before_resume() {
        let record = |index: usize, tokens: usize| WakeContextRecord {
            id: format!("wake-{index}"),
            optimizer_step: 0,
            token_ids: vec![1; tokens],
        };

        let mut too_many_records = state_at(0);
        too_many_records.wake_context_buffer = (0..=MAX_WAKE_CONTEXT_RECORDS)
            .map(|index| record(index, 1))
            .collect();
        let error = too_many_records.validate().unwrap_err().to_string();
        assert!(error.contains("record limit"), "{error}");

        let mut oversized_record = state_at(0);
        oversized_record.wake_context_buffer = vec![record(0, MAX_WAKE_CONTEXT_TOKENS + 1)];
        let error = oversized_record.validate().unwrap_err().to_string();
        assert!(error.contains("token limit"), "{error}");

        let mut oversized_id = state_at(0);
        let mut bad_id = record(0, 1);
        bad_id.id = "x".repeat(MAX_WAKE_CONTEXT_ID_BYTES + 1);
        oversized_id.wake_context_buffer = vec![bad_id];
        let error = oversized_id.validate().unwrap_err().to_string();
        assert!(error.contains("wake-context id"), "{error}");

        let mut invalid_token = state_at(0);
        let mut bad_token = record(0, 1);
        bad_token.token_ids[0] = i64::from(u32::MAX) + 1;
        invalid_token.wake_context_buffer = vec![bad_token];
        let error = invalid_token.validate().unwrap_err().to_string();
        assert!(error.contains("u32 vocabulary range"), "{error}");

        let mut excessive_total = state_at(0);
        let records = MAX_WAKE_CONTEXT_TOTAL_TOKENS / MAX_WAKE_CONTEXT_TOKENS + 1;
        excessive_total.wake_context_buffer = (0..records)
            .map(|index| record(index, MAX_WAKE_CONTEXT_TOKENS))
            .collect();
        let error = excessive_total.validate().unwrap_err().to_string();
        assert!(error.contains("wake-context buffer exceeds"), "{error}");
    }

    #[test]
    fn optimizer_paths_must_be_safe_and_independent() {
        let mut traversal = state_at(7);
        traversal.optimizer_states[0].adamw = "../adamw-state.bpk".into();
        assert!(traversal.validate().is_err());

        let mut absolute = state_at(7);
        absolute.optimizer_states[0].muon = "/tmp/muon-state.bpk".into();
        assert!(absolute.validate().is_err());

        let mut shared = state_at(7);
        shared.optimizer_states[0].muon = ADAMW_FILE.into();
        assert!(shared.validate().is_err());

        let mut control = state_at(7);
        control.optimizer_states[0].muon = "muon\nstate.bpk".into();
        assert!(control.validate().is_err());

        let mut glob = state_at(7);
        glob.optimizer_states[0].muon = "muon[0].bpk".into();
        assert!(glob.validate().is_err());
    }

    #[test]
    fn exact_resume_identities_require_canonical_hashes_and_unique_parameters() {
        let mut state = state_at(7);
        state.workflow_signature = "sha256:workflow".into();
        assert!(state.validate().is_err());

        let mut state = state_at(7);
        state.data_manifest_hash = Some(format!("sha256:{}", "A".repeat(64)));
        assert!(state.validate().is_err());

        let mut state = state_at(7);
        state.evaluator_hashes = vec!["evaluator".into()];
        assert!(state.validate().is_err());

        let mut state = state_at(7);
        state.parameter_ids.push(state.parameter_ids[0]);
        assert!(state.validate().is_err());

        let mut state = state_at(7);
        state
            .evaluator_hashes
            .push(state.evaluator_hashes[0].clone());
        assert!(state.validate().is_err());

        let mut state = state_at(7);
        state.artifacts = vec![ArtifactRef {
            kind: "fixture".into(),
            manifest: "fixture.json".into(),
            hash: format!("sha256:{}", "3".repeat(64)),
        }];
        state.artifacts.push(state.artifacts[0].clone());
        assert!(state.validate().is_err());
    }

    #[test]
    fn quantization_candidate_manifest_and_source_identity_are_atomic() {
        let candidate = QuantizationTrainingState {
            format: "binary_g128".into(),
            fake_quant_active: false,
            calibration_step: 7,
            manifest: Some("candidate.json".into()),
            candidate_weights_sha256: Some(format!("sha256:{}", "a".repeat(64))),
            teacher_hash: None,
        };
        let mut valid = state_at(7);
        valid.quantization = Some(candidate.clone());
        valid.validate().unwrap();
        let mut legacy_json = serde_json::to_value(&valid).unwrap();
        legacy_json["quantization"]
            .as_object_mut()
            .unwrap()
            .remove("candidate_weights_sha256");
        assert!(serde_json::from_value::<TrainingState>(legacy_json).is_err());

        let mut missing_source = valid.clone();
        missing_source
            .quantization
            .as_mut()
            .unwrap()
            .candidate_weights_sha256 = None;
        assert!(missing_source.validate().is_err());

        let mut missing_manifest = valid.clone();
        missing_manifest.quantization.as_mut().unwrap().manifest = None;
        assert!(missing_manifest.validate().is_err());

        let mut malformed_source = valid;
        malformed_source
            .quantization
            .as_mut()
            .unwrap()
            .candidate_weights_sha256 = Some("sha256:not-a-digest".into());
        assert!(malformed_source.validate().is_err());
    }

    #[test]
    fn wake_optimizer_identity_and_clock_are_exact() {
        let mut absent = state_at(7);
        absent.optimizer_states[0].scope = "fast".into();
        assert!(absent.validate().is_err());

        let mut wrong_clock = state_at(7);
        wrong_clock.optimizer_states[0].update_clock = 6;
        assert!(wrong_clock.validate().is_err());

        let mut wrong_path = state_at(7);
        wrong_path.optimizer_states[0].adamw = "other-adamw.bpk".into();
        assert!(wrong_path.validate().is_err());
    }

    #[test]
    fn generation_is_not_visible_until_pointer_publication() {
        let directory = tempfile::tempdir().unwrap();
        let first_state = state_at(7);
        let first_staging = stage_test_generation(directory.path(), &first_state, "first");
        let first = seal_generation(directory.path(), &first_staging).unwrap();
        publish_current(directory.path(), &first).unwrap();

        let second_state = state_at(8);
        let second_staging = stage_test_generation(directory.path(), &second_state, "second");
        let second = seal_generation(directory.path(), &second_staging).unwrap();

        let (current_path, current_state) = resolve_current_generation(directory.path()).unwrap();
        assert_eq!(current_path, first.path);
        assert_eq!(current_state.global_step, 7);

        write_synced_new(
            &directory.path().join(".current-crashed-writer.tmp"),
            b"incomplete",
        )
        .unwrap();
        let (current_path, current_state) = resolve_current_generation(directory.path()).unwrap();
        assert_eq!(current_path, first.path);
        assert_eq!(current_state.global_step, 7);

        publish_current(directory.path(), &second).unwrap();
        let (current_path, current_state) = resolve_current_generation(directory.path()).unwrap();
        assert_eq!(current_path, second.path);
        assert_eq!(current_state.global_step, 8);
        assert_ne!(first.name, second.name);
    }

    #[test]
    fn public_verifier_reports_the_exact_resume_identity() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "verify");
        let generation = seal_generation(directory.path(), &staging).unwrap();
        publish_current(directory.path(), &generation).unwrap();

        let by_root = verify_checkpoint_root(directory.path()).unwrap();
        let by_generation = verify_checkpoint_generation(
            &generation.path,
            &generation.name,
            &generation.manifest_sha256,
        )
        .unwrap();
        assert_eq!(by_root, by_generation);
        assert_eq!(by_root.global_step, 7);
        assert_eq!(by_root.metric_records, 14);
    }

    #[test]
    fn generation_verification_rejects_invalid_training_accounting() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "bad-accounting");
        fs::write(staging.join(TRAINING_ACCOUNTING_FILE), br#"{"version":1}"#).unwrap();

        let error = seal_generation(directory.path(), &staging)
            .unwrap_err()
            .to_string();
        assert!(error.contains("training accounting"), "{error}");
    }

    #[test]
    fn every_declared_optimizer_file_must_be_in_generation() {
        let directory = tempfile::tempdir().unwrap();
        let mut state = state_at(7);
        state.optimizer_states[0].gradient_accumulator = Some("gradients.bpk".into());
        let staging = stage_test_generation(directory.path(), &state, "missing-gradient");
        let error = seal_generation(directory.path(), &staging)
            .unwrap_err()
            .to_string();
        assert!(error.contains("gradients.bpk"), "{error}");
    }

    #[test]
    fn modified_generation_is_rejected_by_content_hash() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "original");
        let generation = seal_generation(directory.path(), &staging).unwrap();
        publish_current(directory.path(), &generation).unwrap();

        fs::write(generation.path.join(WEIGHTS_FILE), b"modified").unwrap();
        let error = resolve_current_generation(directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("contents do not match"), "{error}");
    }

    #[test]
    fn generation_verifier_rejects_unmanifested_empty_directory() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "empty-directory");
        let generation = seal_generation(directory.path(), &staging).unwrap();
        fs::create_dir(generation.path.join("unmanifested")).unwrap();

        let error = verify_checkpoint_generation(
            &generation.path,
            &generation.name,
            &generation.manifest_sha256,
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("empty directory"), "{error}");
    }

    #[test]
    fn generation_sealer_rejects_excessive_directory_depth() {
        let directory = tempfile::tempdir().unwrap();
        let mut state = state_at(7);
        let nested = (0..=MAX_CHECKPOINT_GENERATION_DEPTH)
            .map(|index| format!("d{index}"))
            .collect::<Vec<_>>()
            .join("/");
        let gradient = format!("{nested}/gradients.bpk");
        state.optimizer_states[0].gradient_accumulator = Some(gradient.clone());
        let staging = stage_test_generation(directory.path(), &state, "deep-generation");
        fs::create_dir_all(staging.join(&nested)).unwrap();
        write_synced_new(&staging.join(gradient), b"gradients").unwrap();

        let error = seal_generation(directory.path(), &staging)
            .unwrap_err()
            .to_string();
        assert!(error.contains("maximum directory depth"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn generation_verifier_handles_long_portable_file_names() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "long-name");
        let long_name = format!("{}.bpk", "a".repeat(240));
        write_synced_new(&staging.join(long_name), b"long-name-state").unwrap();
        let generation = seal_generation(directory.path(), &staging).unwrap();

        verify_checkpoint_generation(
            &generation.path,
            &generation.name,
            &generation.manifest_sha256,
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn generation_verifier_rejects_symlinked_file() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "symlink-file");
        let generation = seal_generation(directory.path(), &staging).unwrap();

        let weights = generation.path.join(WEIGHTS_FILE);
        let external = directory.path().join("external-weights.safetensors");
        fs::rename(&weights, &external).unwrap();
        std::os::unix::fs::symlink(&external, &weights).unwrap();

        let error = verify_checkpoint_generation(
            &generation.path,
            &generation.name,
            &generation.manifest_sha256,
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(
            error.contains("failed to open checkpoint generation entry"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn generation_verifier_rejects_symlinked_directory() {
        let directory = tempfile::tempdir().unwrap();
        let mut state = state_at(7);
        state.optimizer_states[0].gradient_accumulator = Some("nested/gradients.bpk".into());
        let staging = stage_test_generation(directory.path(), &state, "symlink-directory");
        fs::create_dir(staging.join("nested")).unwrap();
        write_synced_new(&staging.join("nested/gradients.bpk"), b"gradients").unwrap();
        let generation = seal_generation(directory.path(), &staging).unwrap();

        let nested = generation.path.join("nested");
        let external = directory.path().join("external-nested");
        fs::rename(&nested, &external).unwrap();
        std::os::unix::fs::symlink(&external, &nested).unwrap();

        let error = verify_checkpoint_generation(
            &generation.path,
            &generation.name,
            &generation.manifest_sha256,
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(
            error.contains("failed to open checkpoint generation entry"),
            "{error}"
        );
    }

    #[test]
    fn current_pointer_cannot_escape_generation_root() {
        let directory = tempfile::tempdir().unwrap();
        let pointer = CurrentCheckpoint {
            version: CHECKPOINT_POINTER_VERSION,
            generation: "../elsewhere".into(),
            manifest_sha256: "0".repeat(64),
        };
        write_synced_new(
            &directory.path().join(CURRENT_POINTER),
            &serde_json::to_vec(&pointer).unwrap(),
        )
        .unwrap();
        assert!(resolve_current_generation(directory.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn current_pointer_cannot_be_a_symlink() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("pointer-target.json");
        write_synced_new(&target, b"{}").unwrap();
        std::os::unix::fs::symlink(&target, directory.path().join(CURRENT_POINTER)).unwrap();

        let error = resolve_current_generation(directory.path()).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("not a regular file"), "{error}");
    }
}
