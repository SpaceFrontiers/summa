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
use std::fmt::Write as FmtWrite;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::ffi::{CStr, CString, OsStr, OsString};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use anyhow::{Context, Result, ensure};
use burn::module::{AutodiffModule, Module, ModuleMapper, Param, ParamId};
use burn::tensor::{Bool, Bytes, Device, Int, Tensor};
use burn_optim::ModuleOptimizer;
use hermes_llm::{Transformer, load_safetensors_bytes, save_safetensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::muon::BatchedMuon;
use hermes_train::builtin_sleep_adapters::WakeContextRecord;
use hermes_train::metrics::MetricWriter;
use hermes_train::native_sleep::NativeSleepCheckpoint;
use hermes_train::optimizer_artifact::{
    canonical_module_optimizer_bytes, save_canonical_module_optimizer,
};
use hermes_train::quantization::QuantizationTransactionState;
use hermes_train::training_accounting::{
    TRAINING_ACCOUNTING_FILE, TRAINING_ACCOUNTING_VERSION, TrainingAccounting, TrainingEvidence,
};

pub(crate) type AdamWOptimizer = ModuleOptimizer;

pub(crate) const TRAINING_STATE_VERSION: u32 = 2;

const CHECKPOINT_POINTER_VERSION: u32 = 1;
const CHECKPOINT_MANIFEST_VERSION: u32 = 1;
const GENERATIONS_DIRECTORY: &str = "generations";
const TRAINING_EVIDENCE_DIRECTORY: &str = "training-evidence";
const CURRENT_POINTER: &str = "current.json";
const GENERATION_MANIFEST: &str = "generation-manifest.json";
const TRAINING_STATE_FILE: &str = "training-state.json";
const WEIGHTS_FILE: &str = "weights.safetensors";
const ADAMW_FILE: &str = "adamw-state.bpk";
const MUON_FILE: &str = "muon-state.bpk";

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

#[cfg(unix)]
#[derive(Debug)]
struct GenerationAccess {
    directory: SecureDirectory,
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
    pub(crate) training_evidence: PathBuf,
    pub(crate) training_evidence_sha256: String,
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
    pub(crate) teacher_hash: Option<String>,
    /// Exact interruption point for an in-flight fake-quant/distillation
    /// optimizer update. The backend validates its plan fingerprint on resume.
    pub(crate) transaction: Option<QuantizationTransactionState>,
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
        validate_content_hash(&self.workflow_signature, "checkpoint workflow signature")?;
        if let Some(hash) = &self.data_manifest_hash {
            validate_content_hash(hash, "checkpoint data manifest hash")?;
        }
        ensure!(
            self.global_step == 0 || self.tokens_seen > 0 || self.phase_kind == "evaluation",
            "non-evaluation checkpoint has optimizer progress but no token count"
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
        ensure!(
            self.wake_context_buffer
                .windows(2)
                .all(|pair| pair[0].optimizer_step <= pair[1].optimizer_step),
            "checkpoint wake-context buffer moves backwards"
        );
        let context_ids = self
            .wake_context_buffer
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        ensure!(
            context_ids.len() == self.wake_context_buffer.len(),
            "checkpoint wake-context buffer repeats an identity"
        );
        for record in &self.wake_context_buffer {
            ensure!(
                !record.id.trim().is_empty()
                    && !record.token_ids.is_empty()
                    && record.token_ids.iter().all(|token| *token >= 0),
                "checkpoint contains an invalid wake context"
            );
        }
        let unique_parameter_ids = self.parameter_ids.iter().collect::<BTreeSet<_>>();
        ensure!(
            unique_parameter_ids.len() == self.parameter_ids.len(),
            "checkpoint repeats a parameter ID"
        );
        for artifact in &self.artifacts {
            ensure!(
                !artifact.kind.trim().is_empty()
                    && !artifact.manifest.trim().is_empty()
                    && !artifact.hash.trim().is_empty(),
                "checkpoint has an incomplete artifact reference"
            );
            validate_content_hash(&artifact.hash, "checkpoint artifact hash")?;
        }
        for hash in &self.evaluator_hashes {
            validate_content_hash(hash, "checkpoint evaluator hash")?;
        }
        if let Some(quantization) = &self.quantization {
            ensure!(
                !quantization.format.trim().is_empty(),
                "checkpoint quantization format is empty"
            );
            if let Some(hash) = &quantization.teacher_hash {
                validate_content_hash(hash, "checkpoint quantization teacher hash")?;
            }
            if let Some(transaction) = &quantization.transaction {
                for (hash, label) in [
                    (
                        &transaction.transaction_id,
                        "checkpoint quantization transaction id",
                    ),
                    (
                        &transaction.plan_fingerprint,
                        "checkpoint quantization plan fingerprint",
                    ),
                    (
                        &transaction.pre_update_master_hash,
                        "checkpoint quantization pre-update hash",
                    ),
                ] {
                    validate_content_hash(hash, label)?;
                }
                if let Some(hash) = &transaction.post_update_master_hash {
                    validate_content_hash(hash, "checkpoint quantization post-update hash")?;
                }
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

/// Seal a checkpoint, derive immutable resource evidence from the exact model
/// and committed metric prefix, and only then advance `current.json`.
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
    let (sealed, sealed_accounting) =
        seal_training_checkpoint(model, adamw, muon, state, accounting_input, output)?;
    let (manifest, _) = verify_generation(&sealed.path, &sealed.name, &sealed.manifest_sha256)?;
    let accounting_file = manifest
        .files
        .iter()
        .find(|file| file.path == TRAINING_ACCOUNTING_FILE)
        .context("sealed checkpoint manifest has no training accounting")?;
    let evidence = TrainingEvidence {
        version: 1,
        checkpoint_manifest_sha256: sealed.manifest_sha256.clone(),
        accounting_sha256: accounting_file.sha256.clone(),
        training_gpu_hours: sealed_accounting.training_gpu_hours,
        parameters: sealed_accounting.parameters,
        routed_active_parameters: sealed_accounting.routed_active_parameters,
        stored_bytes: sealed_accounting.weights_bytes,
        weights_sha256: sealed_accounting.weights_sha256.clone(),
    };
    evidence.validate()?;
    let published_evidence = publish_training_evidence(output, &sealed, &evidence)?;
    publish_current(output, &sealed)?;
    Ok(CheckpointPublication {
        checkpoint_manifest: sealed.path.join(GENERATION_MANIFEST),
        checkpoint_manifest_sha256: sealed.manifest_sha256,
        training_evidence: published_evidence.path,
        training_evidence_sha256: published_evidence.sha256,
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
    sync_regular_file(&weights)?;
    let (weights_bytes, weights_sha256) = hash_file(&weights)?;
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
    sync_regular_file(&adamw_state)?;
    muon.save(&muon_state)?;
    sync_regular_file(&muon_state)?;
    write_synced_new(
        &staging.join(TRAINING_STATE_FILE),
        &serde_json::to_vec_pretty(state)?,
    )?;

    let sealed = seal_generation(output, &staging)?;
    staging_guard.disarm();
    Ok((sealed, accounting))
}

#[derive(Debug)]
struct PublishedEvidence {
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Copy, Debug)]
struct TrainingAccountingInput {
    training_gpu_hours: f64,
    parameters: u64,
    routed_active_parameters: u64,
}

fn publish_training_evidence(
    output: &Path,
    generation: &SealedGeneration,
    evidence: &TrainingEvidence,
) -> Result<PublishedEvidence> {
    evidence.validate()?;
    ensure!(
        evidence.checkpoint_manifest_sha256 == generation.manifest_sha256,
        "training evidence does not bind the sealed checkpoint generation"
    );
    ensure!(
        generation.path == output.join(GENERATIONS_DIRECTORY).join(&generation.name),
        "training evidence checkpoint generation is outside its output root"
    );
    let bytes = serde_json::to_vec(evidence)?;
    let sha256 = sha256_bytes(&bytes);
    let directory = output.join(TRAINING_EVIDENCE_DIRECTORY);
    fs::create_dir_all(&directory).with_context(|| {
        format!(
            "failed to create training-evidence directory {}",
            directory.display()
        )
    })?;
    validate_generation_root(&directory)
        .context("training-evidence root is not a real directory")?;
    sync_directory(output)?;
    let path = directory.join(format!("sha256-{sha256}.json"));
    match fs::symlink_metadata(&path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "training evidence {} is not a regular file",
                path.display()
            );
            let existing = read_file_stable(&path, "published training evidence")?;
            ensure!(
                sha256_bytes(&existing) == sha256 && existing == bytes,
                "content-addressed training-evidence collision or tampering at {}",
                path.display()
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let temporary = directory.join(format!(".evidence-{}.tmp", unique_suffix()));
            let publication = (|| -> Result<()> {
                write_synced_new(&temporary, &bytes)?;
                match fs::hard_link(&temporary, &path) {
                    Ok(()) => {
                        fs::remove_file(&temporary)?;
                        Ok(())
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(&path)?;
                        ensure!(
                            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                            "training evidence {} is not a regular file",
                            path.display()
                        );
                        let existing =
                            read_file_stable(&path, "concurrently published training evidence")?;
                        ensure!(
                            sha256_bytes(&existing) == sha256 && existing == bytes,
                            "content-addressed training-evidence collision at {}",
                            path.display()
                        );
                        fs::remove_file(&temporary)?;
                        Ok(())
                    }
                    Err(error) => Err(error).with_context(|| {
                        format!("failed to publish training evidence {}", path.display())
                    }),
                }
            })();
            if publication.is_err() {
                let _ = fs::remove_file(&temporary);
            }
            publication?;
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!("failed to inspect training evidence {}", path.display())
            });
        }
    }
    sync_directory(&directory)?;
    Ok(PublishedEvidence { path, sha256 })
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
    let metadata = fs::symlink_metadata(path).with_context(|| {
        format!(
            "checkpoint generation root {} does not exist",
            path.display()
        )
    })?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "checkpoint generation root {} is not a real directory",
        path.display()
    );
    Ok(())
}

fn unique_suffix() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("{}-{timestamp}-{sequence}", std::process::id())
}

fn write_synced_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_regular_file(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect checkpoint file {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "checkpoint path {} is not a regular file",
        path.display()
    );
    File::open(path)?.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("failed to open directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("failed to sync directory {}", path.display()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StableFileIdentity {
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
impl StableFileIdentity {
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

#[cfg(all(
    unix,
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos",
        target_os = "freebsd"
    )
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    Some(unsafe { libc::__error() })
}

#[cfg(all(
    unix,
    any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "emscripten",
        target_os = "hurd",
        target_os = "redox"
    )
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    Some(unsafe { libc::__errno_location() })
}

#[cfg(all(
    unix,
    any(target_os = "android", target_os = "netbsd", target_os = "openbsd")
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    Some(unsafe { libc::__errno() })
}

#[cfg(all(
    unix,
    not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "emscripten",
        target_os = "hurd",
        target_os = "redox",
        target_os = "android",
        target_os = "netbsd",
        target_os = "openbsd"
    ))
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    None
}

fn hash_open_file(
    file: &mut File,
    capture_bytes: bool,
    label: &str,
) -> Result<(u64, String, Option<Vec<u8>>)> {
    let before = file
        .metadata()
        .with_context(|| format!("failed to inspect opened {label}"))?;
    ensure!(before.is_file(), "{label} is not a regular file");
    #[cfg(unix)]
    let before_identity = StableFileIdentity::from_metadata(&before);
    #[cfg(not(unix))]
    let before_modified = before.modified().ok();

    let mut hasher = Sha256::new();
    let mut captured = if capture_bytes {
        let capacity = usize::try_from(before.len())
            .context("checkpoint file length does not fit address space")?;
        let mut captured = Vec::new();
        captured
            .try_reserve_exact(capacity)
            .context("failed to reserve authenticated checkpoint buffer")?;
        Some(captured)
    } else {
        None
    };
    let mut buffer = [0_u8; 1024 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {label}"))?;
        if read == 0 {
            break;
        }
        let next_bytes = bytes
            .checked_add(read as u64)
            .context("checkpoint file length overflow")?;
        ensure!(
            next_bytes <= before.len(),
            "{label} grew while it was hashed"
        );
        hasher.update(&buffer[..read]);
        if let Some(captured) = &mut captured {
            captured.extend_from_slice(&buffer[..read]);
        }
        bytes = next_bytes;
    }
    let after = file
        .metadata()
        .with_context(|| format!("failed to reinspect opened {label}"))?;
    #[cfg(unix)]
    ensure!(
        StableFileIdentity::from_metadata(&after) == before_identity,
        "{label} changed while it was hashed"
    );
    #[cfg(not(unix))]
    ensure!(
        after.is_file() && after.len() == before.len() && after.modified().ok() == before_modified,
        "{label} changed while it was hashed"
    );
    ensure!(bytes == after.len(), "{label} changed while it was hashed");
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    Ok((bytes, encoded, captured))
}

fn read_hashed_path(
    path: &Path,
    capture_bytes: bool,
    label: &str,
) -> Result<(u64, String, Option<Vec<u8>>)> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect checkpoint path {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "{label} is not a regular file"
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let mut file = options
        .open(path)
        .with_context(|| format!("failed to open {label}"))?;
    #[cfg(unix)]
    ensure!(
        StableFileIdentity::from_metadata(&file.metadata()?)
            == StableFileIdentity::from_metadata(&metadata),
        "{label} changed while it was opened"
    );
    let (bytes, sha256, captured) = hash_open_file(&mut file, capture_bytes, label)?;
    let current = fs::symlink_metadata(path)
        .with_context(|| format!("failed to reinspect checkpoint path {}", path.display()))?;
    #[cfg(unix)]
    ensure!(
        StableFileIdentity::from_metadata(&current) == StableFileIdentity::from_metadata(&metadata),
        "{label} changed while it was hashed"
    );
    #[cfg(not(unix))]
    ensure!(
        current.file_type().is_file()
            && !current.file_type().is_symlink()
            && current.len() == metadata.len()
            && current.modified().ok() == metadata.modified().ok(),
        "{label} changed while it was hashed"
    );
    Ok((bytes, sha256, captured))
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let (bytes, sha256, _) =
        read_hashed_path(path, false, &format!("checkpoint file {}", path.display()))?;
    Ok((bytes, sha256))
}

fn read_file_stable(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_hashed_path(path, true, label)?
        .2
        .context("stable checkpoint read did not capture bytes")
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

#[cfg(unix)]
#[derive(Debug)]
struct SecureDirectory {
    file: File,
}

#[cfg(unix)]
impl SecureDirectory {
    fn open_root(path: &Path) -> Result<Self> {
        let path_metadata = fs::symlink_metadata(path).with_context(|| {
            format!("failed to inspect checkpoint directory {}", path.display())
        })?;
        ensure!(
            path_metadata.is_dir() && !path_metadata.file_type().is_symlink(),
            "checkpoint generation {} is not a real directory",
            path.display()
        );
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .with_context(|| format!("failed to open checkpoint directory {}", path.display()))?;
        ensure!(
            StableFileIdentity::from_metadata(&file.metadata()?)
                == StableFileIdentity::from_metadata(&path_metadata),
            "checkpoint directory {} changed while it was opened",
            path.display()
        );
        Ok(Self { file })
    }

    fn open_child(&self, name: &OsStr) -> Result<File> {
        let name = CString::new(name.as_bytes()).context("checkpoint name contains a NUL byte")?;
        // O_NONBLOCK prevents an attacker-controlled FIFO from hanging the
        // verifier before its file type can be rejected.
        let descriptor = unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
                0,
            )
        };
        if descriptor < 0 {
            return Err(std::io::Error::last_os_error()).context("failed to open checkpoint entry");
        }
        // SAFETY: openat returned a new owned descriptor on success.
        Ok(unsafe { File::from_raw_fd(descriptor) })
    }

    fn open_relative_file(&self, relative: &str) -> Result<File> {
        validate_checkpoint_relative_path(relative)?;
        let mut directory = Self {
            file: self
                .file
                .try_clone()
                .context("failed to duplicate authenticated checkpoint directory")?,
        };
        let mut components = Path::new(relative).components().peekable();
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                anyhow::bail!("checkpoint path `{relative}` is not a safe relative path");
            };
            let child = directory
                .open_child(name)
                .with_context(|| format!("failed to open authenticated artifact `{relative}`"))?;
            let metadata = child.metadata().with_context(|| {
                format!("failed to inspect authenticated artifact `{relative}`")
            })?;
            if components.peek().is_none() {
                ensure!(
                    metadata.is_file(),
                    "authenticated checkpoint artifact `{relative}` is not a regular file"
                );
                return Ok(child);
            }
            ensure!(
                metadata.is_dir(),
                "checkpoint artifact parent for `{relative}` is not a directory"
            );
            directory = Self { file: child };
        }
        anyhow::bail!("checkpoint path `{relative}` has no file name")
    }

    fn entries(&self) -> Result<Vec<OsString>> {
        let duplicate = unsafe { libc::dup(self.file.as_raw_fd()) };
        if duplicate < 0 {
            return Err(std::io::Error::last_os_error())
                .context("failed to duplicate checkpoint directory descriptor");
        }
        let stream = unsafe { libc::fdopendir(duplicate) };
        if stream.is_null() {
            let error = std::io::Error::last_os_error();
            unsafe { libc::close(duplicate) };
            return Err(error).context("failed to enumerate checkpoint directory");
        }

        struct DirectoryStream(*mut libc::DIR);
        impl Drop for DirectoryStream {
            fn drop(&mut self) {
                unsafe { libc::closedir(self.0) };
            }
        }
        let stream = DirectoryStream(stream);
        let mut names = Vec::new();
        loop {
            let errno = errno_slot();
            if let Some(slot) = errno {
                // SAFETY: errno_slot returns this thread's errno storage.
                unsafe { *slot = 0 };
            }
            // SAFETY: `stream` owns a private DIR used only by this loop. The
            // returned entry is copied before the next call to readdir.
            let entry = unsafe { libc::readdir(stream.0) };
            if entry.is_null() {
                let code = errno.map_or(0, |slot| unsafe { *slot });
                if code != 0 {
                    return Err(std::io::Error::from_raw_os_error(code))
                        .context("failed while enumerating checkpoint directory");
                }
                break;
            }
            let bytes = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
            if bytes == b"." || bytes == b".." {
                continue;
            }
            names.push(OsString::from_vec(bytes.to_vec()));
        }
        names.sort();
        Ok(names)
    }
}

impl GenerationAccess {
    fn read_authenticated(&mut self, relative: &str, expected: &GenerationFile) -> Result<Vec<u8>> {
        ensure!(
            expected.path == relative,
            "checkpoint artifact identity does not match `{relative}`"
        );
        #[cfg(unix)]
        let file = self.directory.open_relative_file(relative)?;
        #[cfg(not(unix))]
        let file = self
            .files
            .remove(relative)
            .with_context(|| format!("checkpoint snapshot did not retain `{relative}`"))?;
        read_authenticated_file(file, expected)
    }
}

fn read_authenticated_file(mut file: File, expected: &GenerationFile) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("failed to rewind checkpoint artifact `{}`", expected.path))?;
    let (bytes, sha256, captured) = hash_open_file(
        &mut file,
        true,
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
        directory: &SecureDirectory,
        relative_root: &Path,
        snapshot: &mut GenerationSnapshot,
    ) -> Result<()> {
        let before = StableFileIdentity::from_metadata(&directory.file.metadata()?);
        for name in directory.entries()? {
            let relative_path = relative_root.join(&name);
            let relative = relative_path
                .to_str()
                .context("checkpoint file name is not valid UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            validate_checkpoint_relative_path(&relative)?;
            let mut child = directory.open_child(&name).with_context(|| {
                format!("failed to open checkpoint generation entry `{relative}`")
            })?;
            let metadata = child.metadata()?;
            if metadata.is_dir() {
                let files_before = snapshot.files.len();
                visit(&SecureDirectory { file: child }, &relative_path, snapshot)?;
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
            let capture = matches!(
                relative.as_str(),
                GENERATION_MANIFEST | TRAINING_STATE_FILE | TRAINING_ACCOUNTING_FILE
            );
            let (bytes, sha256, captured) = hash_open_file(
                &mut child,
                capture,
                &format!("checkpoint file `{relative}`"),
            )?;
            snapshot.record_file(relative, bytes, sha256, captured)?;
        }
        ensure!(
            StableFileIdentity::from_metadata(&directory.file.metadata()?) == before,
            "checkpoint directory changed while it was verified"
        );
        Ok(())
    }

    let directory = SecureDirectory::open_root(root)?;
    let root_identity = StableFileIdentity::from_metadata(&directory.file.metadata()?);
    let mut snapshot = GenerationSnapshot::new();
    visit(&directory, Path::new(""), &mut snapshot)?;
    let current = fs::symlink_metadata(root).with_context(|| {
        format!(
            "failed to reinspect checkpoint generation {}",
            root.display()
        )
    })?;
    ensure!(
        current.is_dir()
            && !current.file_type().is_symlink()
            && StableFileIdentity::from_metadata(&current) == root_identity,
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
    ) -> Result<()> {
        let before = fs::metadata(directory)?;
        let mut entries = fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
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
                visit(root, &path, snapshot, retained)?;
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
            let (bytes, sha256, captured) =
                hash_open_file(&mut file, capture, &format!("checkpoint file `{relative}`"))?;
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
    visit(root, root, &mut snapshot, &mut retained)?;
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

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} is not a lowercase SHA-256 digest"
    );
    Ok(())
}

fn validate_content_hash(value: &str, label: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .with_context(|| format!("{label} must use sha256:<64 lowercase hex>"))?;
    validate_sha256(digest, label)
}

fn validate_generation_name(name: &str) -> Result<&str> {
    let digest = name
        .strip_prefix("sha256-")
        .context("checkpoint generation is not content-addressed")?;
    validate_sha256(digest, "checkpoint generation digest")?;
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
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
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
    validate_sha256(expected_manifest_sha256, "checkpoint manifest digest")?;
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
        sha256_bytes(&manifest_bytes) == expected_manifest_sha256,
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
    validate_sha256(&pointer.manifest_sha256, "checkpoint manifest digest")?;
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

pub(crate) fn load_training_state(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
) -> Result<(AdamWOptimizer, TrainingState)> {
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
) -> Result<(AdamWOptimizer, TrainingState)> {
    let (verified_before, generation, manifest, state, access) =
        verify_checkpoint_root_inner(output, true)?;
    let mut access = access.context("checkpoint verifier did not retain generation access")?;
    stage_hook(ResumeLoadStage::AfterInitialVerify, &generation)?;

    let mut loaded_model = model.clone();
    let mut loaded_muon = muon.clone();
    restore_parameter_ids(&mut loaded_model, &state.parameter_ids)?;
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
            && sha256_bytes(&canonical_adamw) == expected_adamw.sha256,
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
    Ok((loaded_adamw, state))
}

#[cfg(test)]
pub(crate) fn load_training_state_with_hook(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
    stage_hook: impl FnMut(ResumeLoadStage, &Path) -> Result<()>,
) -> Result<(AdamWOptimizer, TrainingState)> {
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
    fn training_evidence_is_bound_idempotent_and_outside_the_generation() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "evidence");
        let generation = seal_generation(directory.path(), &staging).unwrap();
        let manifest_before = fs::read(generation.path.join(GENERATION_MANIFEST)).unwrap();
        let manifest: GenerationManifest = serde_json::from_slice(&manifest_before).unwrap();
        let accounting_sha256 = manifest
            .files
            .iter()
            .find(|file| file.path == TRAINING_ACCOUNTING_FILE)
            .unwrap()
            .sha256
            .clone();
        let (_, weights_sha256) = hash_file(&generation.path.join(WEIGHTS_FILE)).unwrap();
        let evidence = TrainingEvidence {
            version: 1,
            checkpoint_manifest_sha256: generation.manifest_sha256.clone(),
            accounting_sha256,
            training_gpu_hours: 1.25,
            parameters: 100,
            routed_active_parameters: 80,
            stored_bytes: 16,
            weights_sha256,
        };

        let first = publish_training_evidence(directory.path(), &generation, &evidence).unwrap();
        let second = publish_training_evidence(directory.path(), &generation, &evidence).unwrap();
        assert_eq!(first.path, second.path);
        assert_eq!(first.sha256, second.sha256);
        assert!(
            first
                .path
                .starts_with(directory.path().join(TRAINING_EVIDENCE_DIRECTORY))
        );
        assert!(!first.path.starts_with(&generation.path));
        assert_eq!(
            fs::read(generation.path.join(GENERATION_MANIFEST)).unwrap(),
            manifest_before
        );
        verify_generation(
            &generation.path,
            &generation.name,
            &generation.manifest_sha256,
        )
        .unwrap();

        fs::write(&first.path, b"tampered").unwrap();
        let error = publish_training_evidence(directory.path(), &generation, &evidence)
            .unwrap_err()
            .to_string();
        assert!(error.contains("tampering"), "{error}");
    }

    #[test]
    fn training_evidence_cannot_bind_a_different_generation() {
        let directory = tempfile::tempdir().unwrap();
        let state = state_at(7);
        let staging = stage_test_generation(directory.path(), &state, "evidence-binding");
        let generation = seal_generation(directory.path(), &staging).unwrap();
        let manifest: GenerationManifest =
            serde_json::from_slice(&fs::read(generation.path.join(GENERATION_MANIFEST)).unwrap())
                .unwrap();
        let accounting_sha256 = manifest
            .files
            .iter()
            .find(|file| file.path == TRAINING_ACCOUNTING_FILE)
            .unwrap()
            .sha256
            .clone();
        let (_, weights_sha256) = hash_file(&generation.path.join(WEIGHTS_FILE)).unwrap();
        let evidence = TrainingEvidence {
            version: 1,
            checkpoint_manifest_sha256: "0".repeat(64),
            accounting_sha256,
            training_gpu_hours: 1.0,
            parameters: 100,
            routed_active_parameters: 80,
            stored_bytes: 16,
            weights_sha256,
        };
        let error = publish_training_evidence(directory.path(), &generation, &evidence)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not bind"), "{error}");
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
