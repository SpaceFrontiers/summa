//! Crash-safe, resumable training checkpoints.
//!
//! Parameter IDs are persisted alongside model and optimizer state because
//! Burn optimizers key their state by those IDs. Checkpoint files are sealed in
//! immutable, content-addressed generation directories. A generation becomes
//! visible only when the small `current.json` pointer is atomically replaced,
//! so an interrupted save cannot hide the preceding complete checkpoint.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use burn::module::{AutodiffModule, Module, ModuleMapper, Param, ParamId};
use burn::tensor::{Bool, Device, Int, Tensor};
use burn_optim::ModuleOptimizer;
use hermes_llm::{Transformer, load_safetensors, save_safetensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::muon::BatchedMuon;
use hermes_train::builtin_sleep_adapters::WakeContextRecord;
use hermes_train::metrics::MetricWriter;
use hermes_train::native_sleep::NativeSleepCheckpoint;
use hermes_train::optimizer_artifact::save_canonical_module_optimizer;
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
            let existing = fs::read(&path)?;
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
                        let existing = fs::read(&path)?;
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
        !path.contains('\\') && !path.contains(':'),
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

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "checkpoint path {} is not a regular file",
        path.display()
    );
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .context("checkpoint file length overflow")?;
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    Ok((bytes, encoded))
}

fn collect_generation_files(root: &Path) -> Result<Vec<GenerationFile>> {
    fn visit(root: &Path, directory: &Path, files: &mut Vec<GenerationFile>) -> Result<()> {
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
                visit(root, &path, files)?;
                continue;
            }
            ensure!(
                metadata.is_file(),
                "checkpoint generation contains non-file {}",
                path.display()
            );
            let relative = path.strip_prefix(root)?;
            let relative = relative
                .to_str()
                .context("checkpoint file name is not valid UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            if relative == GENERATION_MANIFEST {
                continue;
            }
            validate_checkpoint_relative_path(&relative)?;
            let (bytes, sha256) = hash_file(&path)?;
            files.push(GenerationFile {
                path: relative,
                bytes,
                sha256,
            });
        }
        Ok(())
    }

    let mut files = Vec::new();
    visit(root, root, &mut files)?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    ensure!(
        files.windows(2).all(|pair| pair[0].path != pair[1].path),
        "checkpoint generation repeats a file path"
    );
    Ok(files)
}

fn read_training_state(generation: &Path) -> Result<TrainingState> {
    let path = generation.join(TRAINING_STATE_FILE);
    let metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("checkpoint generation is missing {TRAINING_STATE_FILE}"))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "checkpoint training state is not a regular file"
    );
    let state: TrainingState = serde_json::from_slice(&fs::read(&path)?)?;
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

fn validate_optimizer_files(
    state: &TrainingState,
    generation: &Path,
    files: &[GenerationFile],
) -> Result<()> {
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
        let path = generation.join(relative);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("optimizer or gradient state `{relative}` does not exist"))?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "optimizer or gradient state `{relative}` is not a regular file"
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
    let state = read_training_state(staging)?;
    let files = collect_generation_files(staging)?;
    ensure_required_files(&files)?;
    validate_optimizer_files(&state, staging, &files)?;
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
    let generation_digest = validate_generation_name(generation_name)?;
    validate_sha256(expected_manifest_sha256, "checkpoint manifest digest")?;
    ensure!(
        generation_digest == expected_manifest_sha256,
        "checkpoint generation name and manifest digest differ"
    );
    let metadata = fs::symlink_metadata(generation)
        .with_context(|| format!("checkpoint generation `{generation_name}` does not exist"))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "checkpoint generation `{generation_name}` is not a directory"
    );
    let manifest_path = generation.join(GENERATION_MANIFEST);
    let manifest_metadata =
        fs::symlink_metadata(&manifest_path).context("checkpoint generation has no manifest")?;
    ensure!(
        manifest_metadata.file_type().is_file() && !manifest_metadata.file_type().is_symlink(),
        "checkpoint generation manifest is not a regular file"
    );
    let manifest_bytes = fs::read(&manifest_path)?;
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
    let actual_files = collect_generation_files(generation)?;
    ensure!(
        actual_files == manifest.files,
        "checkpoint generation contents do not match its manifest"
    );
    ensure_required_files(&manifest.files)?;
    let state = read_training_state(generation)?;
    ensure!(
        state.version == manifest.training_state_version
            && state.global_step == manifest.global_step
            && state.phase == manifest.phase
            && state.phase_id == manifest.phase_id,
        "checkpoint training state does not match its generation manifest"
    );
    validate_optimizer_files(&state, generation, &manifest.files)?;
    Ok((manifest, state))
}

fn resolve_current_generation(output: &Path) -> Result<(PathBuf, TrainingState)> {
    let pointer_path = output.join(CURRENT_POINTER);
    let metadata = fs::symlink_metadata(&pointer_path).with_context(|| {
        format!(
            "checkpoint has no atomic `{CURRENT_POINTER}` pointer; a version-2 generation checkpoint is required"
        )
    })?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "checkpoint current pointer is not a regular file"
    );
    let pointer: CurrentCheckpoint = serde_json::from_slice(&fs::read(&pointer_path)?)?;
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
    let (_, state) = verify_generation(&generation, &pointer.generation, &pointer.manifest_sha256)?;
    Ok((generation, state))
}

pub(crate) fn load_training_state(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
) -> Result<(AdamWOptimizer, TrainingState)> {
    let (generation, state) = resolve_current_generation(output)?;
    restore_parameter_ids(model, &state.parameter_ids)?;
    load_safetensors(model, generation.join(WEIGHTS_FILE))?;
    let wake = state
        .optimizer_states
        .iter()
        .find(|optimizer| optimizer.scope == "wake")
        .context("checkpoint has no `wake` optimizer state")?;
    muon.set_parameter_ids(model.muon_parameter_ids());
    muon.load(generation.join(&wake.muon), &device.clone().inner())?;
    let adamw = adamw
        .load(generation.join(&wake.adamw))
        .context("failed to load AdamW state")?;
    Ok((adamw, state))
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
}
