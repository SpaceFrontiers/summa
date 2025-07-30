//! Strict first-party composition for standalone and periodic in-model sleep.
//!
//! This is deliberately a local-artifact runtime: every mutable input is
//! content pinned and every output is published into a dedicated immutable
//! store. Standalone phases load a wake journal and tier-optimizer snapshot
//! supplied by the caller; periodic wake training seals both at each due
//! boundary instead. Remote/object-store deployments can implement the same
//! native factory and boundary-driver contracts without weakening these
//! bindings.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use burn::module::list_param_ids;
use burn_optim::{AdamWConfig, ModuleOptimizer};
use hermes_llm::{Device, ModelDef, Transformer, default_device, load_safetensors, parse_mal};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::builtin_dreaming::{BuiltinDreamOps, BuiltinDreamingRuntimeConfig};
use crate::builtin_sleep_adapters::{
    JournalRolloutConfig, JournalRollouts, PinnedLikelihoodRetentionEvaluator, PinnedLocalArtifact,
    PinnedTokenSemanticJudge, PinnedWakeContextJournal,
};
use crate::native_sleep::{
    NativeCheckpointRef, NativeConsolidationOutcome, NativeSleepCheckpoint,
    NativeSleepPhaseContext, NativeSleepPhaseContextFactory, NativeSleepPhaseOutcome,
    NativeSleepProgressSink, PeriodicSleepBoundaryDriver, PinnedNativeArtifact,
    execute_native_consolidation, execute_native_dreaming,
};
use crate::runtime::PhaseExecutionRequest;
use crate::tensor_sleep::{
    ImmutableTransformerCheckpoint, TensorConsolidationBackend, TensorDreamBackend,
    TensorTransactionStore, restore_parameter_ids,
};
use crate::tier_optimizer::{
    AtomicSafetensorsCandidatePublisher, DurableTierOptimizerPublisher, ProspectiveTierUpdate,
    TierOptimizerBank, TierOptimizerConfig,
};
use crate::workflow::InModelSleepConfig;

pub const BUILTIN_SLEEP_RUNTIME_CONFIG_VERSION: u32 = 1;
pub const MODEL_PARAMETER_IDS_VERSION: u32 = 1;
const REJECTION_REPORT_VERSION: u32 = 1;

/// Bind Dreaming to the newest policy which crossed the durable candidate
/// boundary. Pending transactions are intentionally ignored: after a crash in
/// DreamPolicyUpdate, retry must use the same parent as the first attempt, not
/// its own partially published child.
fn bind_latest_committed_dream_policy(
    config: &mut BuiltinDreamingRuntimeConfig,
    sleep: &crate::sleep::SleepState,
) -> Result<()> {
    let Some((transaction_id, receipt)) =
        sleep.completed_transactions.iter().rev().find_map(|txn| {
            txn.dream_policy_receipt
                .as_deref()
                .map(|receipt| (txn.id, receipt))
        })
    else {
        return Ok(());
    };
    config.initial_policy = Some(config.committed_policy_artifact(transaction_id, receipt)?);
    Ok(())
}

fn default_rng_streams() -> usize {
    1
}

fn default_max_wake_context_records() -> usize {
    64
}

/// Deployment configuration shared by both built-in sleep execution modes.
///
/// Relative paths are interpreted relative to this file, never the process
/// working directory. Standalone mode requires `wake_context_journal`,
/// `initial_tier_optimizer_state`, and `initial_model_parameter_ids`; periodic
/// mode rejects those fields because the integrated wake trainer produces
/// fresh boundary artifacts itself.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BuiltinSleepRuntimeConfig {
    pub version: u32,
    /// Execution identity expected by the selected surface: the full
    /// content-derived training run signature for periodic `train`, or the
    /// resolved WorkflowV2 signature for standalone `run-workflow`.
    pub workflow_signature: String,
    pub model_mal: PinnedLocalArtifact,
    /// Standalone-only wake journal. Periodic training supplies and seals a
    /// fresh journal at each due boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_context_journal: Option<PinnedLocalArtifact>,
    pub semantic_judge: PinnedLocalArtifact,
    pub retention_evaluator: PinnedLocalArtifact,
    pub retention_suite: PinnedLocalArtifact,
    /// Exact optimizer/microbatch state produced at the wake boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_tier_optimizer_state: Option<PinnedLocalArtifact>,
    /// Stable Burn parameter identities paired with the standalone input
    /// checkpoint and optimizer snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_model_parameter_ids: Option<PinnedLocalArtifact>,
    pub rollouts: JournalRolloutConfig,
    pub tier_optimizer: TierOptimizerConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dreaming: Option<BuiltinDreamingRuntimeConfig>,
    pub tensor_transaction_directory: PathBuf,
    pub prospective_directory: PathBuf,
    pub tier_optimizer_directory: PathBuf,
    pub candidate_directory: PathBuf,
    pub rejection_report_directory: PathBuf,
    /// Trainer-side bound for the recent model-owned context ring sealed at a
    /// periodic boundary. The runtime consumes only the sealed journal.
    #[serde(default = "default_max_wake_context_records")]
    pub max_wake_context_records: usize,
    #[serde(default = "default_rng_streams")]
    pub rng_streams: usize,
}

impl BuiltinSleepRuntimeConfig {
    pub fn resolve_paths(mut self, base: &Path) -> Result<Self> {
        self.model_mal = self.model_mal.resolve(base)?;
        self.wake_context_journal = self
            .wake_context_journal
            .map(|artifact| artifact.resolve(base))
            .transpose()?;
        self.semantic_judge = self.semantic_judge.resolve(base)?;
        self.retention_evaluator = self.retention_evaluator.resolve(base)?;
        self.retention_suite = self.retention_suite.resolve(base)?;
        self.initial_tier_optimizer_state = self
            .initial_tier_optimizer_state
            .map(|artifact| artifact.resolve(base))
            .transpose()?;
        self.initial_model_parameter_ids = self
            .initial_model_parameter_ids
            .map(|artifact| artifact.resolve(base))
            .transpose()?;
        self.dreaming = self
            .dreaming
            .map(|value| value.resolve_paths(base))
            .transpose()?;
        for path in [
            &mut self.tensor_transaction_directory,
            &mut self.prospective_directory,
            &mut self.tier_optimizer_directory,
            &mut self.candidate_directory,
            &mut self.rejection_report_directory,
        ] {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == BUILTIN_SLEEP_RUNTIME_CONFIG_VERSION,
            "unsupported built-in sleep runtime version {}",
            self.version
        );
        validate_sha256(&self.workflow_signature, "workflow signature")?;
        self.rollouts.validate()?;
        self.tier_optimizer.validate()?;
        ensure!(self.rng_streams > 0, "sleep runtime has no RNG streams");
        ensure!(
            self.max_wake_context_records > 0,
            "sleep runtime wake-context ring is empty"
        );
        let mut directories = vec![
            &self.tensor_transaction_directory,
            &self.prospective_directory,
            &self.tier_optimizer_directory,
            &self.candidate_directory,
            &self.rejection_report_directory,
        ];
        if let Some(dreaming) = &self.dreaming {
            dreaming.validate()?;
            directories.push(&dreaming.artifact_directory);
            if let (Some(dreaming_journal), Some(journal)) =
                (&dreaming.wake_context_journal, &self.wake_context_journal)
            {
                ensure!(
                    *dreaming_journal == *journal,
                    "Dreaming and consolidation must use the same pinned wake journal"
                );
            }
        }
        ensure!(
            directories.iter().all(|path| !path.as_os_str().is_empty()),
            "sleep runtime contains an empty output directory"
        );
        ensure!(
            directories
                .iter()
                .enumerate()
                .all(|(index, path)| directories[index + 1..].iter().all(|other| path != other)),
            "sleep runtime output directories, including Dreaming artifacts, must be distinct"
        );
        Ok(())
    }

    pub fn validate_standalone(&self) -> Result<()> {
        self.validate()?;
        ensure!(
            self.wake_context_journal.is_some(),
            "standalone sleep runtime requires wake_context_journal"
        );
        ensure!(
            self.initial_tier_optimizer_state.is_some(),
            "standalone sleep runtime requires initial_tier_optimizer_state"
        );
        ensure!(
            self.initial_model_parameter_ids.is_some(),
            "standalone sleep runtime requires initial_model_parameter_ids"
        );
        if let Some(dreaming) = &self.dreaming {
            ensure!(
                dreaming.wake_context_journal.as_ref() == self.wake_context_journal.as_ref(),
                "standalone Dreaming requires the exact consolidation wake_context_journal"
            );
        }
        Ok(())
    }

    pub fn validate_periodic(&self) -> Result<()> {
        self.validate()?;
        ensure!(
            self.wake_context_journal.is_none()
                && self.initial_tier_optimizer_state.is_none()
                && self.initial_model_parameter_ids.is_none(),
            "periodic sleep runtime must not contain standalone wake artifacts"
        );
        ensure!(
            self.dreaming
                .as_ref()
                .is_none_or(|dreaming| dreaming.wake_context_journal.is_none()),
            "periodic Dreaming must not contain a standalone wake_context_journal"
        );
        Ok(())
    }

    pub fn standalone_wake_context_journal(&self) -> Result<&PinnedLocalArtifact> {
        self.wake_context_journal
            .as_ref()
            .context("standalone sleep runtime has no wake_context_journal")
    }

    pub fn standalone_tier_optimizer_state(&self) -> Result<&PinnedLocalArtifact> {
        self.initial_tier_optimizer_state
            .as_ref()
            .context("standalone sleep runtime has no initial_tier_optimizer_state")
    }

    pub fn standalone_model_parameter_ids(&self) -> Result<&PinnedLocalArtifact> {
        self.initial_model_parameter_ids
            .as_ref()
            .context("standalone sleep runtime has no initial_model_parameter_ids")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelParameterIdsArtifact {
    pub version: u32,
    pub checkpoint_sha256: String,
    pub parameter_ids: Vec<u64>,
}

impl ModelParameterIdsArtifact {
    pub fn from_model(checkpoint_sha256: impl Into<String>, model: &Transformer) -> Result<Self> {
        let value = Self {
            version: MODEL_PARAMETER_IDS_VERSION,
            checkpoint_sha256: checkpoint_sha256.into(),
            parameter_ids: list_param_ids(model)
                .into_iter()
                .map(|id| id.val())
                .collect(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == MODEL_PARAMETER_IDS_VERSION,
            "unsupported model parameter-id artifact version {}",
            self.version
        );
        validate_sha256(&self.checkpoint_sha256, "parameter-id checkpoint")?;
        ensure!(
            !self.parameter_ids.is_empty(),
            "model parameter-id artifact is empty"
        );
        let unique = self
            .parameter_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            unique.len() == self.parameter_ids.len(),
            "model parameter-id artifact contains duplicates"
        );
        Ok(())
    }
}

/// Content-pinned factory registered with [`crate::native_sleep::NativeSleepContextRegistry`].
#[derive(Clone)]
pub struct BuiltinSleepPhaseContextFactory {
    identity: String,
    config: BuiltinSleepRuntimeConfig,
    device: Device,
    mode: RuntimeMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeMode {
    Standalone,
    Periodic,
}

impl BuiltinSleepPhaseContextFactory {
    pub fn load(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        Self::load_on_device(path, expected_sha256, default_device())
    }

    pub fn load_on_device(
        path: impl AsRef<Path>,
        expected_sha256: &str,
        device: Device,
    ) -> Result<Self> {
        Self::load_mode(
            path.as_ref(),
            expected_sha256,
            device,
            RuntimeMode::Standalone,
        )
    }

    /// Load only common immutable artifacts. Wake contexts and optimizer state
    /// are supplied by the integrated trainer at each periodic boundary.
    pub fn load_periodic(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        Self::load_periodic_on_device(path, expected_sha256, default_device())
    }

    pub fn load_periodic_on_device(
        path: impl AsRef<Path>,
        expected_sha256: &str,
        device: Device,
    ) -> Result<Self> {
        Self::load_mode(
            path.as_ref(),
            expected_sha256,
            device,
            RuntimeMode::Periodic,
        )
    }

    fn load_mode(
        path: &Path,
        expected_sha256: &str,
        device: Device,
        mode: RuntimeMode,
    ) -> Result<Self> {
        let pinned = PinnedLocalArtifact {
            path: path.to_owned(),
            sha256: expected_sha256.to_owned(),
        };
        let config: BuiltinSleepRuntimeConfig = pinned.verify_json()?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let config = config.resolve_paths(base)?;
        match mode {
            RuntimeMode::Standalone => config.validate_standalone()?,
            RuntimeMode::Periodic => config.validate_periodic()?,
        }
        config.model_mal.verify_bytes()?;
        config.semantic_judge.verify_bytes()?;
        config.retention_evaluator.verify_bytes()?;
        config.retention_suite.verify_bytes()?;
        if mode == RuntimeMode::Standalone {
            config
                .wake_context_journal
                .as_ref()
                .expect("validated standalone journal")
                .verify_bytes()?;
            config
                .initial_tier_optimizer_state
                .as_ref()
                .expect("validated standalone optimizer state")
                .verify_bytes()?;
            let parameter_ids: ModelParameterIdsArtifact = config
                .initial_model_parameter_ids
                .as_ref()
                .expect("validated standalone parameter ids")
                .verify_json()?;
            parameter_ids.validate()?;
        }
        for directory in [
            &config.tensor_transaction_directory,
            &config.prospective_directory,
            &config.tier_optimizer_directory,
            &config.candidate_directory,
            &config.rejection_report_directory,
        ] {
            ensure_real_directory(directory)?;
        }
        Ok(Self {
            identity: expected_sha256.to_owned(),
            config,
            device,
            mode,
        })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn config(&self) -> &BuiltinSleepRuntimeConfig {
        &self.config
    }

    fn validate_workflow_binding(
        &self,
        request: &PhaseExecutionRequest,
        sleep: &InModelSleepConfig,
    ) -> Result<()> {
        let input = request
            .input_checkpoint
            .as_ref()
            .context("built-in sleep requires an input checkpoint")?;
        ensure!(
            sleep.standalone_trigger_clock.is_some(),
            "built-in phase factory only supports standalone sleep"
        );
        ensure!(
            sleep.retention_suite == self.config.retention_suite.path
                && sleep.retention_suite_sha256 == self.config.retention_suite.sha256,
            "WorkflowV2 retention suite differs from built-in runtime"
        );
        ensure!(
            sleep.imitation.semantic_judge_hash == self.config.semantic_judge.sha256,
            "WorkflowV2 semantic judge differs from built-in runtime"
        );
        ensure!(
            sleep.retention.evaluator_hash == self.config.retention_evaluator.sha256,
            "WorkflowV2 retention evaluator differs from built-in runtime"
        );
        ensure_same_directory(
            &sleep.candidate_directory,
            &self.config.candidate_directory,
            "candidate directory",
        )?;
        let journal_artifact = self.config.standalone_wake_context_journal()?;
        let journal =
            PinnedWakeContextJournal::load(&journal_artifact.path, &journal_artifact.sha256)?;
        ensure!(
            journal.source_checkpoint_sha256() == input.sha256(),
            "wake journal belongs to {}, phase input is {}",
            journal.source_checkpoint_sha256(),
            input.sha256()
        );
        let parameter_ids: ModelParameterIdsArtifact = self
            .config
            .standalone_model_parameter_ids()?
            .verify_json()?;
        parameter_ids.validate()?;
        ensure!(
            parameter_ids.checkpoint_sha256 == input.sha256(),
            "parameter-id artifact belongs to another input checkpoint"
        );
        match (&sleep.dreaming, &self.config.dreaming) {
            (None, None) => {}
            (Some(workflow), Some(runtime)) => {
                ensure!(
                    workflow.reference_set_hash == runtime.reference_set.sha256
                        && workflow.trial_evaluator_hash
                            == runtime.independent_evaluation_set.sha256,
                    "WorkflowV2 Dreaming artifacts differ from built-in runtime"
                );
            }
            _ => bail!("WorkflowV2 and built-in runtime disagree about Dreaming"),
        }
        Ok(())
    }
}

impl NativeSleepPhaseContextFactory for BuiltinSleepPhaseContextFactory {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn create(
        &mut self,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
    ) -> Result<Box<dyn NativeSleepPhaseContext>> {
        ensure!(
            self.mode == RuntimeMode::Standalone,
            "periodic runtime factory cannot create a standalone phase context"
        );
        self.validate_workflow_binding(request, config)?;
        Ok(Box::new(BuiltinStandaloneSleepContext {
            runtime: self.config.clone(),
            device: self.device.clone(),
        }))
    }
}

struct BuiltinStandaloneSleepContext {
    runtime: BuiltinSleepRuntimeConfig,
    device: Device,
}

struct ProgressForward<'a>(&'a mut dyn NativeSleepProgressSink);

impl NativeSleepProgressSink for ProgressForward<'_> {
    fn persist(&mut self, checkpoint: &NativeSleepCheckpoint) -> Result<()> {
        self.0.persist(checkpoint)
    }

    fn metric(
        &mut self,
        checkpoint: &NativeSleepCheckpoint,
        event: crate::metrics::MetricEvent,
    ) -> Result<()> {
        self.0.metric(checkpoint, event)
    }
}

impl BuiltinStandaloneSleepContext {
    fn model_def(&self) -> Result<ModelDef> {
        let bytes = self.runtime.model_mal.verify_bytes()?;
        let source = std::str::from_utf8(&bytes).context("model MAL is not UTF-8")?;
        parse_mal(source).context("parsing pinned model MAL")
    }

    fn load_checkpoint(
        &self,
        reference: &NativeCheckpointRef,
        model_def: &ModelDef,
    ) -> Result<ImmutableTransformerCheckpoint> {
        let artifact = PinnedLocalArtifact {
            path: PathBuf::from(&reference.uri),
            sha256: reference.sha256.clone(),
        };
        artifact.verify_bytes()?;
        let mut model = Transformer::new(model_def, &self.device)?;
        load_safetensors(&mut model, &artifact.path)?;
        Ok(ImmutableTransformerCheckpoint {
            uri: reference.uri.clone(),
            sha256: reference.sha256.clone(),
            model,
        })
    }

    fn load_standalone_input(
        &self,
        reference: &NativeCheckpointRef,
        model_def: &ModelDef,
    ) -> Result<ImmutableTransformerCheckpoint> {
        let mut checkpoint = self.load_checkpoint(reference, model_def)?;
        let identities: ModelParameterIdsArtifact = self
            .runtime
            .standalone_model_parameter_ids()?
            .verify_json()?;
        identities.validate()?;
        ensure!(
            identities.checkpoint_sha256 == reference.sha256,
            "standalone parameter identities belong to another checkpoint"
        );
        restore_parameter_ids(&mut checkpoint.model, &identities.parameter_ids)?;
        Ok(checkpoint)
    }

    fn load_journal(&self) -> Result<PinnedWakeContextJournal> {
        let artifact = self.runtime.standalone_wake_context_journal()?;
        PinnedWakeContextJournal::load(&artifact.path, &artifact.sha256)
    }

    fn native_journal_artifact(&self) -> Result<PinnedNativeArtifact> {
        let artifact = self.runtime.standalone_wake_context_journal()?;
        PinnedNativeArtifact::from_path(&artifact.path, artifact.sha256.clone())
    }

    fn restore_recorded_backend(
        &self,
        cursor: &NativeSleepCheckpoint,
        model_def: &ModelDef,
        backend: &mut TensorConsolidationBackend<
            ProspectiveTierUpdate,
            JournalRollouts,
            PinnedTokenSemanticJudge,
            PinnedLikelihoodRetentionEvaluator,
            AtomicSafetensorsCandidatePublisher,
        >,
    ) -> Result<()> {
        let Some(txn) = cursor.sleep.pending.as_ref() else {
            return Ok(());
        };
        if txn.tensor_transaction_generation.is_none() {
            return Ok(());
        }
        let store = TensorTransactionStore::new(&self.runtime.tensor_transaction_directory);
        let optimizer: ModuleOptimizer = AdamWConfig::new()
            .with_weight_decay(backend.config().receiver_weight_decay)
            .init();
        let recovered = store.load_recorded(txn, model_def, &self.device, optimizer)?;
        backend.restore_inflight(txn, recovered)
    }

    fn recover_model_for_ref(
        &self,
        cursor: &NativeSleepCheckpoint,
        reference: &NativeCheckpointRef,
        input_model: &ImmutableTransformerCheckpoint,
        model_def: &ModelDef,
        receiver_weight_decay: f32,
    ) -> Result<ImmutableTransformerCheckpoint> {
        if reference.uri == input_model.uri && reference.sha256 == input_model.sha256 {
            return Ok(input_model.clone());
        }
        let store = TensorTransactionStore::new(&self.runtime.tensor_transaction_directory);
        let optimizer = || {
            AdamWConfig::new()
                .with_weight_decay(receiver_weight_decay)
                .init()
        };
        if let Some(txn) = cursor.sleep.pending.as_ref()
            && txn.tensor_transaction_generation.is_some()
        {
            let recovered = store.load_recorded(txn, model_def, &self.device, optimizer())?;
            if reference.uri == recovered.teacher.uri
                && reference.sha256 == recovered.teacher.sha256
            {
                return Ok(recovered.teacher);
            }
            if txn.candidate_checkpoint.as_deref() == Some(reference.uri.as_str())
                && txn.candidate_hash.as_deref() == Some(reference.sha256.as_str())
            {
                return Ok(ImmutableTransformerCheckpoint {
                    uri: reference.uri.clone(),
                    sha256: reference.sha256.clone(),
                    model: recovered.student.checkpoint.model,
                });
            }
        }
        for txn in cursor.sleep.completed_transactions.iter().rev() {
            if txn.candidate_checkpoint.as_deref() == Some(reference.uri.as_str())
                && txn.candidate_hash.as_deref() == Some(reference.sha256.as_str())
                && txn.tensor_transaction_generation.is_some()
            {
                let recovered = store.load_recorded(txn, model_def, &self.device, optimizer())?;
                return Ok(ImmutableTransformerCheckpoint {
                    uri: reference.uri.clone(),
                    sha256: reference.sha256.clone(),
                    model: recovered.student.checkpoint.model,
                });
            }
        }
        self.load_checkpoint(reference, model_def)
    }

    fn rejection(
        &self,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
        input: &NativeCheckpointRef,
        rejected: &NativeSleepCheckpoint,
    ) -> Result<NativeSleepPhaseOutcome> {
        let input_model = self.load_standalone_input(input, &self.model_def()?)?;
        let mut restored = NativeSleepCheckpoint::new(
            self.runtime.workflow_signature.clone(),
            request.phase.name.clone(),
            input.clone(),
            &input_model.model,
            config,
            self.runtime.rng_streams,
        )?;
        restored.advance_clock(
            &input_model.model,
            config,
            config
                .standalone_trigger_clock
                .context("standalone sleep trigger is absent")?,
        )?;
        restored.bind_wake_context_journal(self.native_journal_artifact()?)?;
        let report = RejectionReport {
            version: REJECTION_REPORT_VERSION,
            workflow_signature: self.runtime.workflow_signature.clone(),
            phase: request.phase.name.clone(),
            input_checkpoint: input.clone(),
            rejected_cycle: rejected.sleep.cycle,
            rejected_chain_sha256: rejected.sleep.completed_chain_hash.clone(),
            reason: "retention_gate_rejected".into(),
        };
        let (report_uri, report_sha256) = publish_json(
            &self.runtime.rejection_report_directory,
            "sleep-rejection",
            &report,
        )?;
        Ok(NativeSleepPhaseOutcome::Rejected {
            checkpoint: restored,
            report_uri,
            report_sha256,
        })
    }
}

impl NativeSleepPhaseContext for BuiltinStandaloneSleepContext {
    fn drive_sleep_phase(
        &mut self,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
        resume: Option<NativeSleepCheckpoint>,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<NativeSleepPhaseOutcome> {
        let input = request
            .input_checkpoint
            .as_ref()
            .context("standalone sleep has no input checkpoint")?;
        let input_ref = NativeCheckpointRef::new(input.uri(), input.sha256())?;
        let model_def = self.model_def()?;
        let input_model = self.load_standalone_input(&input_ref, &model_def)?;
        let bank = TierOptimizerBank::new(
            &input_model.model,
            &config.schedule,
            self.runtime.tier_optimizer.clone(),
        )?;
        bank.restore_bytes(
            &self
                .runtime
                .standalone_tier_optimizer_state()?
                .verify_bytes()?,
        )?;
        let transaction_store =
            TensorTransactionStore::new(&self.runtime.tensor_transaction_directory);
        let mut optimizer_publisher = DurableTierOptimizerPublisher::new(
            bank.clone(),
            &self.runtime.tier_optimizer_directory,
            transaction_store.clone(),
            model_def.clone(),
            self.device.clone(),
        )?;
        let mut cursor = match resume {
            Some(cursor) => {
                ensure!(
                    cursor.wake_context_journal.as_ref() == Some(&self.native_journal_artifact()?),
                    "standalone resume wake journal differs from built-in runtime"
                );
                cursor
            }
            None => {
                let mut cursor = NativeSleepCheckpoint::new(
                    self.runtime.workflow_signature.clone(),
                    request.phase.name.clone(),
                    input_ref.clone(),
                    &input_model.model,
                    config,
                    self.runtime.rng_streams,
                )?;
                cursor.optimizer_scopes = optimizer_publisher.publish_checkpoint_scopes()?;
                cursor.advance_clock(
                    &input_model.model,
                    config,
                    config
                        .standalone_trigger_clock
                        .context("standalone sleep trigger is absent")?,
                )?;
                cursor.bind_wake_context_journal(self.native_journal_artifact()?)?;
                progress.persist(&cursor)?;
                cursor
            }
        };

        // Rehydrate every durable optimizer generation before either replaying
        // an in-flight transaction or preparing the next sender.
        let scope_model_ref = if let Some(txn) = &cursor.sleep.pending {
            NativeCheckpointRef::new(&txn.teacher_checkpoint, &txn.teacher_hash)?
        } else {
            cursor.live_checkpoint.clone()
        };
        let scope_model = self.recover_model_for_ref(
            &cursor,
            &scope_model_ref,
            &input_model,
            &model_def,
            config.receiver_weight_decay,
        )?;
        optimizer_publisher.restore_scopes(&cursor.optimizer_scopes, &scope_model.model)?;

        let mut sink = ProgressForward(progress);
        let mut cached_model = Some(scope_model);
        loop {
            if cursor.sleep.pending.is_none() {
                let Some(sender) = cursor.sleep.next_due_sender() else {
                    return Ok(NativeSleepPhaseOutcome::Accepted(cursor));
                };
                let teacher = match cached_model.take() {
                    Some(model)
                        if model.uri == cursor.live_checkpoint.uri
                            && model.sha256 == cursor.live_checkpoint.sha256 =>
                    {
                        model
                    }
                    _ => self.recover_model_for_ref(
                        &cursor,
                        &cursor.live_checkpoint,
                        &input_model,
                        &model_def,
                        config.receiver_weight_decay,
                    )?,
                };
                let mut updates =
                    ProspectiveTierUpdate::new(bank.clone(), &self.runtime.prospective_directory)?;
                let trigger_clock = cursor.sleep.due_clocks[0];
                let plan = updates.prepare_consolidation(
                    cursor.sleep.cycle + 1,
                    sender,
                    trigger_clock,
                    &teacher,
                )?;
                cursor.begin_next(plan)?;
                sink.persist(&cursor)?;
                cached_model = Some(teacher);
            }

            let txn = cursor
                .sleep
                .pending
                .clone()
                .context("standalone sleep lost its transaction")?;
            if txn.committed {
                let committed = match cached_model.take() {
                    Some(model)
                        if model.uri == cursor.live_checkpoint.uri
                            && model.sha256 == cursor.live_checkpoint.sha256 =>
                    {
                        model
                    }
                    _ => self.recover_model_for_ref(
                        &cursor,
                        &cursor.live_checkpoint,
                        &input_model,
                        &model_def,
                        config.receiver_weight_decay,
                    )?,
                };
                match &self.runtime.dreaming {
                    Some(dreaming_config) => {
                        let mut dreaming_config = dreaming_config.clone();
                        bind_latest_committed_dream_policy(&mut dreaming_config, &cursor.sleep)?;
                        let mut ops = BuiltinDreamOps::load(dreaming_config, self.device.clone())?;
                        ops.bind_phase_teacher(input.sha256(), &txn.teacher_hash)?;
                        let probe = ops.probe(&committed.model)?;
                        let mut dreaming = TensorDreamBackend::new(
                            committed.model.clone(),
                            self.device.clone(),
                            probe,
                            ops,
                        )?;
                        execute_native_dreaming(
                            &mut cursor,
                            &committed.model,
                            config,
                            Some(&mut dreaming),
                            &mut sink,
                        )?;
                    }
                    None => {
                        execute_native_dreaming::<NoDreamBackend, _>(
                            &mut cursor,
                            &committed.model,
                            config,
                            None,
                            &mut sink,
                        )?;
                    }
                }
                cached_model = Some(committed);
                continue;
            }

            let teacher_ref = NativeCheckpointRef::new(&txn.teacher_checkpoint, &txn.teacher_hash)?;
            let teacher = match cached_model.take() {
                Some(model)
                    if model.uri == teacher_ref.uri && model.sha256 == teacher_ref.sha256 =>
                {
                    model
                }
                _ => self.recover_model_for_ref(
                    &cursor,
                    &teacher_ref,
                    &input_model,
                    &model_def,
                    config.receiver_weight_decay,
                )?,
            };
            let updates =
                ProspectiveTierUpdate::new(bank.clone(), &self.runtime.prospective_directory)?;
            let journal = self.load_journal()?;
            let rollouts = JournalRollouts::for_phase_teacher(
                journal,
                input.sha256(),
                &txn.teacher_hash,
                self.device.clone(),
                self.runtime.rollouts.clone(),
            )?;
            let judge = PinnedTokenSemanticJudge::load(
                &self.runtime.semantic_judge.path,
                &self.runtime.semantic_judge.sha256,
            )?;
            let evaluator = PinnedLikelihoodRetentionEvaluator::load(
                &self.runtime.retention_evaluator.path,
                &self.runtime.retention_evaluator.sha256,
                &self.runtime.retention_suite.path,
                &self.runtime.retention_suite.sha256,
                self.device.clone(),
            )?;
            let candidate =
                AtomicSafetensorsCandidatePublisher::new(&self.runtime.candidate_directory)?;
            let mut backend = TensorConsolidationBackend::new(
                teacher,
                self.device.clone(),
                config.tensor_config(),
                updates,
                rollouts,
                judge,
                evaluator,
                candidate,
            )?;
            self.restore_recorded_backend(&cursor, &model_def, &mut backend)?;
            let outcome = execute_native_consolidation(
                &mut cursor,
                config,
                &mut backend,
                &transaction_store,
                &mut optimizer_publisher,
                &mut sink,
            )?;
            cached_model = Some(backend.live_checkpoint().clone());
            match outcome {
                NativeConsolidationOutcome::Accepted(_) => continue,
                NativeConsolidationOutcome::Rejected(_) => {
                    bank.restore_bytes(
                        &self
                            .runtime
                            .standalone_tier_optimizer_state()?
                            .verify_bytes()?,
                    )?;
                    return self.rejection(request, config, &input_ref, &cursor);
                }
            }
        }
    }
}

/// Reusable in-process driver for wake-training boundaries. It does not own
/// wake optimization: the trainer supplies the live model, shared tier bank,
/// and a newly sealed wake-context journal for each boundary.
pub struct BuiltinPeriodicSleepBoundaryDriver {
    identity: String,
    runtime: BuiltinSleepRuntimeConfig,
    device: Device,
    bank: TierOptimizerBank,
    live: ImmutableTransformerCheckpoint,
    boundary_journal: Option<PinnedLocalArtifact>,
    boundary_teacher_sha256: Option<String>,
    restore_from_cursor: bool,
}

impl BuiltinPeriodicSleepBoundaryDriver {
    pub fn new(
        factory: &BuiltinSleepPhaseContextFactory,
        bank: TierOptimizerBank,
        live_checkpoint: NativeCheckpointRef,
        live_model: Transformer,
        boundary_journal: PinnedLocalArtifact,
    ) -> Result<Self> {
        let mut value = Self {
            identity: factory.identity.clone(),
            runtime: factory.config.clone(),
            device: factory.device.clone(),
            bank,
            live: ImmutableTransformerCheckpoint {
                uri: live_checkpoint.uri,
                sha256: live_checkpoint.sha256,
                model: live_model,
            },
            boundary_journal: Some(boundary_journal),
            boundary_teacher_sha256: None,
            restore_from_cursor: false,
        };
        value.validate_live_model()?;
        value.bind_journal_to_live()?;
        Ok(value)
    }

    /// Recreate after interruption using only the journal identity persisted
    /// in [`NativeSleepCheckpoint`]. The first drain authenticates and loads
    /// that cursor-owned artifact.
    pub fn resume(
        factory: &BuiltinSleepPhaseContextFactory,
        bank: TierOptimizerBank,
        live_checkpoint: NativeCheckpointRef,
        live_model: Transformer,
    ) -> Result<Self> {
        let value = Self {
            identity: factory.identity.clone(),
            runtime: factory.config.clone(),
            device: factory.device.clone(),
            bank,
            live: ImmutableTransformerCheckpoint {
                uri: live_checkpoint.uri,
                sha256: live_checkpoint.sha256,
                model: live_model,
            },
            boundary_journal: None,
            boundary_teacher_sha256: None,
            restore_from_cursor: true,
        };
        value.validate_live_model()?;
        Ok(value)
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn config(&self) -> &BuiltinSleepRuntimeConfig {
        &self.runtime
    }

    pub fn live_checkpoint(&self) -> NativeCheckpointRef {
        NativeCheckpointRef {
            uri: self.live.uri.clone(),
            sha256: self.live.sha256.clone(),
        }
    }

    pub fn live_model(&self) -> &Transformer {
        &self.live.model
    }

    /// Rebind after the caller completes more wake optimization and seals its
    /// next model-owned context journal.
    pub fn bind_wake_boundary(
        &mut self,
        live_checkpoint: NativeCheckpointRef,
        live_model: Transformer,
        boundary_journal: PinnedLocalArtifact,
    ) -> Result<()> {
        self.live = ImmutableTransformerCheckpoint {
            uri: live_checkpoint.uri,
            sha256: live_checkpoint.sha256,
            model: live_model,
        };
        self.boundary_journal = Some(boundary_journal);
        self.validate_live_model()?;
        self.bind_journal_to_live()
    }

    /// Seal the bank's current wake accumulators into immutable bundles and
    /// attach their exact scopes to the native cursor. The trainer persists
    /// the cursor through its normal progress sink before calling `drain`.
    pub fn checkpoint_wake_scopes(
        &mut self,
        checkpoint: &mut NativeSleepCheckpoint,
        config: &InModelSleepConfig,
    ) -> Result<()> {
        ensure!(
            checkpoint.sleep.phase == crate::sleep::SleepPhase::Wake
                && checkpoint.sleep.pending.is_none(),
            "cannot checkpoint wake optimizer scopes during a sleep transaction"
        );
        ensure!(
            checkpoint.live_checkpoint.uri == self.live.uri
                && checkpoint.live_checkpoint.sha256 == self.live.sha256,
            "wake optimizer scopes belong to another live checkpoint"
        );
        let publisher = DurableTierOptimizerPublisher::new(
            self.bank.clone(),
            &self.runtime.tier_optimizer_directory,
            TensorTransactionStore::new(&self.runtime.tensor_transaction_directory),
            self.live.model.config().clone(),
            self.device.clone(),
        )?;
        checkpoint.optimizer_scopes = publisher.publish_checkpoint_scopes()?;
        checkpoint.validate(&self.live.model, config)?;
        self.restore_from_cursor = false;
        Ok(())
    }

    /// Restore the independently-clocked tier optimizers before resumed wake
    /// backward work, even when no sleep boundary is currently due.
    pub fn restore_wake_scopes(
        &mut self,
        checkpoint: &NativeSleepCheckpoint,
        config: &InModelSleepConfig,
    ) -> Result<()> {
        self.validate_sleep_binding(config)?;
        ensure!(
            checkpoint.sleep.phase == crate::sleep::SleepPhase::Wake
                && checkpoint.sleep.pending.is_none(),
            "cannot restore wake optimizer scopes during a sleep transaction"
        );
        ensure!(
            checkpoint.live_checkpoint.uri == self.live.uri
                && checkpoint.live_checkpoint.sha256 == self.live.sha256,
            "resume optimizer scopes belong to another live checkpoint"
        );
        let publisher = DurableTierOptimizerPublisher::new(
            self.bank.clone(),
            &self.runtime.tier_optimizer_directory,
            TensorTransactionStore::new(&self.runtime.tensor_transaction_directory),
            self.live.model.config().clone(),
            self.device.clone(),
        )?;
        publisher.restore_scopes(&checkpoint.optimizer_scopes, &self.live.model)?;
        checkpoint.validate(&self.live.model, config)?;
        self.restore_from_cursor = false;
        Ok(())
    }

    fn helper(&self) -> BuiltinStandaloneSleepContext {
        BuiltinStandaloneSleepContext {
            runtime: self.runtime.clone(),
            device: self.device.clone(),
        }
    }

    fn validate_live_model(&self) -> Result<()> {
        let helper = self.helper();
        let model_def = helper.model_def()?;
        ensure!(
            serde_json::to_vec(self.live.model.config())? == serde_json::to_vec(&model_def)?,
            "periodic live model topology differs from pinned MAL"
        );
        PinnedLocalArtifact {
            path: PathBuf::from(&self.live.uri),
            sha256: self.live.sha256.clone(),
        }
        .verify_bytes()?;
        self.bank.tier_clocks()?;
        Ok(())
    }

    fn bind_journal_to_live(&mut self) -> Result<()> {
        let artifact = self
            .boundary_journal
            .as_ref()
            .context("periodic boundary has no wake-context journal")?;
        let journal = PinnedWakeContextJournal::load(&artifact.path, &artifact.sha256)?;
        ensure!(
            journal.source_checkpoint_sha256() == self.live.sha256,
            "periodic wake journal belongs to {}, boundary teacher is {}",
            journal.source_checkpoint_sha256(),
            self.live.sha256
        );
        self.boundary_teacher_sha256 = Some(self.live.sha256.clone());
        Ok(())
    }

    fn bind_or_restore_cursor_journal(
        &mut self,
        checkpoint: &mut NativeSleepCheckpoint,
        sink: &mut ProgressForward<'_>,
    ) -> Result<PinnedLocalArtifact> {
        if let Some(supplied) = self.boundary_journal.clone() {
            let journal = PinnedWakeContextJournal::load(&supplied.path, &supplied.sha256)?;
            ensure!(
                journal.source_checkpoint_sha256()
                    == self
                        .boundary_teacher_sha256
                        .as_deref()
                        .context("periodic boundary teacher is absent")?,
                "periodic supplied journal changed after binding"
            );
            let native = PinnedNativeArtifact::from_path(&supplied.path, supplied.sha256.clone())?;
            if checkpoint.wake_context_journal.as_ref() != Some(&native) {
                checkpoint.bind_wake_context_journal(native)?;
                sink.persist(checkpoint)?;
            }
            return Ok(supplied);
        }
        let native = checkpoint
            .wake_context_journal
            .clone()
            .context("periodic resume cursor has no pinned wake-context journal")?;
        native.verify()?;
        let artifact = PinnedLocalArtifact {
            path: PathBuf::from(&native.path),
            sha256: native.sha256,
        };
        let journal = PinnedWakeContextJournal::load(&artifact.path, &artifact.sha256)?;
        self.boundary_teacher_sha256 = Some(journal.source_checkpoint_sha256().to_owned());
        self.boundary_journal = Some(artifact.clone());
        Ok(artifact)
    }

    fn validate_sleep_binding(&self, config: &InModelSleepConfig) -> Result<()> {
        ensure!(
            config.retention_suite == self.runtime.retention_suite.path
                && config.retention_suite_sha256 == self.runtime.retention_suite.sha256,
            "periodic retention suite differs from built-in runtime"
        );
        ensure!(
            config.imitation.semantic_judge_hash == self.runtime.semantic_judge.sha256
                && config.retention.evaluator_hash == self.runtime.retention_evaluator.sha256,
            "periodic evaluator identities differ from built-in runtime"
        );
        ensure_same_directory(
            &config.candidate_directory,
            &self.runtime.candidate_directory,
            "candidate directory",
        )?;
        match (&config.dreaming, &self.runtime.dreaming) {
            (None, None) => {}
            (Some(workflow), Some(runtime)) => {
                ensure!(
                    workflow.reference_set_hash == runtime.reference_set.sha256
                        && workflow.trial_evaluator_hash
                            == runtime.independent_evaluation_set.sha256,
                    "periodic Dreaming artifacts differ from built-in runtime"
                );
            }
            _ => bail!("periodic WorkflowV2 and runtime disagree about Dreaming"),
        }
        Ok(())
    }
}

impl PeriodicSleepBoundaryDriver for BuiltinPeriodicSleepBoundaryDriver {
    fn drain_due_sender(
        &mut self,
        checkpoint: &mut NativeSleepCheckpoint,
        config: &InModelSleepConfig,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<()> {
        self.validate_sleep_binding(config)?;
        ensure!(
            checkpoint.live_checkpoint.uri == self.live.uri
                && checkpoint.live_checkpoint.sha256 == self.live.sha256,
            "periodic driver live model differs from native checkpoint"
        );
        ensure!(
            checkpoint.sleep.pending.is_some() || checkpoint.sleep.next_due_sender().is_some(),
            "periodic driver has no due sender"
        );
        let model_def = self.live.model.config().clone();
        let transaction_store =
            TensorTransactionStore::new(&self.runtime.tensor_transaction_directory);
        let mut optimizer_publisher = DurableTierOptimizerPublisher::new(
            self.bank.clone(),
            &self.runtime.tier_optimizer_directory,
            transaction_store.clone(),
            model_def.clone(),
            self.device.clone(),
        )?;
        if self.restore_from_cursor {
            optimizer_publisher.restore_scopes(&checkpoint.optimizer_scopes, &self.live.model)?;
            self.restore_from_cursor = false;
        } else {
            ensure!(
                self.bank.scopes()? == checkpoint.optimizer_scopes,
                "wake optimizer bank differs from cursor; call checkpoint_wake_scopes before drain"
            );
        }
        let mut sink = ProgressForward(progress);
        let boundary_journal = self.bind_or_restore_cursor_journal(checkpoint, &mut sink)?;
        let boundary_teacher_sha256 = self
            .boundary_teacher_sha256
            .clone()
            .context("periodic boundary teacher is absent")?;

        if checkpoint.sleep.pending.is_none() {
            let sender = checkpoint
                .sleep
                .next_due_sender()
                .context("periodic sleep has no due sender")?;
            let mut updates =
                ProspectiveTierUpdate::new(self.bank.clone(), &self.runtime.prospective_directory)?;
            let plan = updates.prepare_consolidation(
                checkpoint.sleep.cycle + 1,
                sender,
                checkpoint.sleep.due_clocks[0],
                &self.live,
            )?;
            checkpoint.begin_next(plan)?;
            sink.persist(checkpoint)?;
        }

        let txn = checkpoint
            .sleep
            .pending
            .clone()
            .context("periodic sleep lost its transaction")?;
        if !txn.committed {
            let updates =
                ProspectiveTierUpdate::new(self.bank.clone(), &self.runtime.prospective_directory)?;
            let journal =
                PinnedWakeContextJournal::load(&boundary_journal.path, &boundary_journal.sha256)?;
            let rollouts = JournalRollouts::for_phase_teacher(
                journal,
                &boundary_teacher_sha256,
                &txn.teacher_hash,
                self.device.clone(),
                self.runtime.rollouts.clone(),
            )?;
            let judge = PinnedTokenSemanticJudge::load(
                &self.runtime.semantic_judge.path,
                &self.runtime.semantic_judge.sha256,
            )?;
            let evaluator = PinnedLikelihoodRetentionEvaluator::load(
                &self.runtime.retention_evaluator.path,
                &self.runtime.retention_evaluator.sha256,
                &self.runtime.retention_suite.path,
                &self.runtime.retention_suite.sha256,
                self.device.clone(),
            )?;
            let candidate =
                AtomicSafetensorsCandidatePublisher::new(&self.runtime.candidate_directory)?;
            let mut backend = TensorConsolidationBackend::new(
                self.live.clone(),
                self.device.clone(),
                config.tensor_config(),
                updates,
                rollouts,
                judge,
                evaluator,
                candidate,
            )?;
            self.helper()
                .restore_recorded_backend(checkpoint, &model_def, &mut backend)?;
            let outcome = execute_native_consolidation(
                checkpoint,
                config,
                &mut backend,
                &transaction_store,
                &mut optimizer_publisher,
                &mut sink,
            )?;
            self.live = backend.live_checkpoint().clone();
            if matches!(outcome, NativeConsolidationOutcome::Rejected(_)) {
                ensure!(
                    checkpoint.sleep.pending.is_none(),
                    "rejected periodic transaction remains pending"
                );
                return Ok(());
            }
        } else {
            ensure!(
                checkpoint.live_checkpoint.uri == self.live.uri
                    && checkpoint.live_checkpoint.sha256 == self.live.sha256,
                "resumed Dreaming candidate differs from periodic driver"
            );
        }

        let txn = checkpoint
            .sleep
            .pending
            .clone()
            .context("committed periodic transaction disappeared")?;
        match &self.runtime.dreaming {
            Some(runtime_dreaming) => {
                let mut runtime_dreaming = runtime_dreaming.clone();
                runtime_dreaming.wake_context_journal = Some(boundary_journal);
                bind_latest_committed_dream_policy(&mut runtime_dreaming, &checkpoint.sleep)?;
                let mut ops = BuiltinDreamOps::load(runtime_dreaming, self.device.clone())?;
                ops.bind_phase_teacher(&boundary_teacher_sha256, &txn.teacher_hash)?;
                let probe = ops.probe(&self.live.model)?;
                let mut dreaming = TensorDreamBackend::new(
                    self.live.model.clone(),
                    self.device.clone(),
                    probe,
                    ops,
                )?;
                execute_native_dreaming(
                    checkpoint,
                    &self.live.model,
                    config,
                    Some(&mut dreaming),
                    &mut sink,
                )?;
            }
            None => {
                execute_native_dreaming::<NoDreamBackend, _>(
                    checkpoint,
                    &self.live.model,
                    config,
                    None,
                    &mut sink,
                )?;
            }
        }
        ensure!(
            checkpoint.sleep.phase == crate::sleep::SleepPhase::Wake
                && checkpoint.sleep.pending.is_none(),
            "periodic driver did not finish exactly one sender"
        );
        Ok(())
    }
}

// Used only to satisfy the generic type when WorkflowV2 disables Dreaming;
// it is never constructed or invoked.
struct NoDreamBackend;

impl crate::sleep::DreamingBackend for NoDreamBackend {
    fn shared_checkpoint_hash(&mut self) -> Result<String> {
        unreachable!("Dreaming backend is absent")
    }

    fn generate_from_wake_contexts(
        &mut self,
        _: &crate::sleep::ConsolidationTxn,
        _: usize,
        _: bool,
    ) -> Result<(String, Vec<crate::sleep::GeneratedDream>)> {
        unreachable!("Dreaming backend is absent")
    }

    fn load_generated_dreams(
        &mut self,
        _: &crate::sleep::ConsolidationTxn,
        _: &str,
    ) -> Result<Vec<crate::sleep::GeneratedDream>> {
        unreachable!("Dreaming backend is absent")
    }

    fn reference_gradient(
        &mut self,
        _: &crate::sleep::ConsolidationTxn,
        _: &str,
    ) -> Result<Vec<f32>> {
        unreachable!("Dreaming backend is absent")
    }

    fn isolated_lora_trial(
        &mut self,
        _: &crate::sleep::ConsolidationTxn,
        _: &crate::sleep::GeneratedDream,
        _: usize,
        _: usize,
    ) -> Result<crate::sleep::DreamTrial> {
        unreachable!("Dreaming backend is absent")
    }

    fn restem_update(
        &mut self,
        _: &crate::sleep::ConsolidationTxn,
        _: &[crate::sleep::DreamTrial],
        _: usize,
    ) -> Result<String> {
        unreachable!("Dreaming backend is absent")
    }

    fn restore_shared_candidate(&mut self, _: &crate::sleep::ConsolidationTxn) -> Result<()> {
        unreachable!("Dreaming backend is absent")
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct RejectionReport {
    version: u32,
    workflow_signature: String,
    phase: String,
    input_checkpoint: NativeCheckpointRef,
    rejected_cycle: u64,
    rejected_chain_sha256: String,
    reason: String,
}

fn publish_json<T: Serialize>(root: &Path, stem: &str, value: &T) -> Result<(String, String)> {
    ensure_real_directory(root)?;
    let bytes = serde_json::to_vec_pretty(value)?;
    let sha256 = sha256_bytes(&bytes);
    let digest = sha256
        .strip_prefix("sha256:")
        .expect("hash helper returns prefixed digest");
    let path = root.join(format!("{stem}-sha256-{digest}.json"));
    if path.exists() {
        ensure!(
            read_regular(&path)? == bytes,
            "immutable report hash collision"
        );
    } else {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        File::open(root)?.sync_all()?;
    }
    Ok((
        path.to_str()
            .context("report path is not UTF-8")?
            .to_owned(),
        sha256,
    ))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .with_context(|| format!("{label} must use sha256:<64 lowercase hex>"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} must use sha256:<64 lowercase hex>"
    );
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn read_regular(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading metadata for {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{} is not a regular non-symlink file",
        path.display()
    );
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn ensure_real_directory(path: &Path) -> Result<()> {
    if path.exists() {
        let metadata = fs::symlink_metadata(path)?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "{} is not a real directory",
            path.display()
        );
    } else {
        fs::create_dir_all(path)
            .with_context(|| format!("creating sleep runtime directory {}", path.display()))?;
    }
    Ok(())
}

fn ensure_same_directory(left: &Path, right: &Path, label: &str) -> Result<()> {
    ensure_real_directory(left)?;
    ensure_real_directory(right)?;
    ensure!(
        fs::canonicalize(left)? == fs::canonicalize(right)?,
        "WorkflowV2 {label} differs from built-in runtime"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_sleep_adapters::{WakeContextJournal, WakeContextRecord};
    use crate::runtime::ImmutableModelCheckpoint;
    use crate::sleep::{
        ConsolidationTxn, ImitationConfig, KnowledgeSeedingConfig, MemoryTierSchedule,
        SleepSchedule, SleepState, TerminalConsolidation, UpdateClock,
    };
    use crate::tensor_sleep::RetentionGateConfig;
    use burn::module::AutodiffModule;

    #[test]
    fn checked_in_runtime_examples_match_their_strict_modes() {
        let periodic: BuiltinSleepRuntimeConfig =
            serde_json::from_str(include_str!("../sleep-runtime.periodic.example.json")).unwrap();
        periodic.validate_periodic().unwrap();
        assert!(periodic.wake_context_journal.is_none());
        assert!(periodic.initial_tier_optimizer_state.is_none());
        assert!(periodic.initial_model_parameter_ids.is_none());
        assert!(
            periodic
                .dreaming
                .as_ref()
                .is_some_and(|dreaming| dreaming.wake_context_journal.is_none())
        );
        assert!(periodic.validate_standalone().is_err());

        let standalone: BuiltinSleepRuntimeConfig =
            serde_json::from_str(include_str!("../sleep-runtime.standalone.example.json")).unwrap();
        standalone.validate_standalone().unwrap();
        assert_eq!(
            standalone.wake_context_journal.as_ref(),
            standalone
                .dreaming
                .as_ref()
                .and_then(|dreaming| dreaming.wake_context_journal.as_ref())
        );
        assert!(standalone.initial_tier_optimizer_state.is_some());
        assert!(standalone.initial_model_parameter_ids.is_some());
        assert!(standalone.validate_periodic().is_err());
    }

    fn hash(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn pin(path: &Path, bytes: &[u8]) -> PinnedLocalArtifact {
        fs::write(path, bytes).unwrap();
        PinnedLocalArtifact {
            path: path.to_owned(),
            sha256: sha256_bytes(bytes),
        }
    }

    fn completed_transaction_with_policy(id: u64, receipt: String) -> ConsolidationTxn {
        ConsolidationTxn {
            id,
            trigger_clock: id,
            sender: 0,
            receiver: 1,
            receiver_slot: 0,
            terminal: false,
            sender_slots_to_reset: Vec::new(),
            teacher_checkpoint: "teacher.safetensors".into(),
            teacher_hash: hash('1'),
            student_checkpoint: "student.safetensors".into(),
            student_hash: hash('2'),
            prospective_update_hash: hash('3'),
            candidate_checkpoint: Some("candidate.safetensors".into()),
            candidate_hash: Some(hash('4')),
            knowledge_rng: None,
            imitation_rng: None,
            dream_generation_rng: None,
            dream_selection_rng: None,
            dream_trial_rngs: Vec::new(),
            tensor_transaction_generation: None,
            tensor_transaction_manifest_hash: None,
            generated_manifest: Some(hash('5')),
            dream_shared_checkpoint_hash: Some(hash('6')),
            dream_selected: Vec::new(),
            dream_trials: Vec::new(),
            dream_policy_receipt: Some(receipt),
            committed: true,
        }
    }

    fn publish_test_policy(root: &Path, transaction_id: u64) -> String {
        let mut bytes = serde_json::to_vec(&serde_json::json!({
            "version": 2,
            "parent_policy_sha256": null,
            "parent_adapter_sha256": null,
            "transaction_id": transaction_id,
            "source_checkpoint_sha256": hash('a'),
            "source_model_parameter_sha256": hash('b'),
            "topology_sha256": hash('c'),
            "adapter_sha256": hash('d'),
            "target_module": "transformer.output_projection.dream_policy",
            "input_features": 8,
            "output_features": 16,
            "rank": 64,
            "alpha": 128,
            "iterations": 0,
            "learning_rate": 0.1,
            "accepted_candidates": [],
            "accepted_adapters": []
        }))
        .unwrap();
        bytes.push(b'\n');
        let receipt = sha256_bytes(&bytes);
        let policies = root.join("policies");
        fs::create_dir_all(&policies).unwrap();
        fs::write(
            policies.join(format!("{}.json", receipt.strip_prefix("sha256:").unwrap())),
            bytes,
        )
        .unwrap();
        receipt
    }

    fn rollout_config() -> JournalRolloutConfig {
        JournalRolloutConfig {
            max_context_tokens: 4,
            continuation_tokens: 2,
            temperature: 0.8,
            top_k: 4,
            repetition_penalty: 1.05,
            eos_token: None,
            imitation_groups: 1,
        }
    }

    fn periodic_runtime(root: &Path) -> BuiltinSleepRuntimeConfig {
        let model_mal = pin(&root.join("model.mal"), b"model placeholder");
        let semantic_judge = pin(&root.join("judge.json"), b"{}");
        let retention_evaluator = pin(&root.join("evaluator.json"), b"{}");
        let retention_suite = pin(&root.join("retention.json"), b"{}");
        BuiltinSleepRuntimeConfig {
            version: BUILTIN_SLEEP_RUNTIME_CONFIG_VERSION,
            workflow_signature: hash('a'),
            model_mal,
            wake_context_journal: None,
            semantic_judge,
            retention_evaluator,
            retention_suite,
            initial_tier_optimizer_state: None,
            initial_model_parameter_ids: None,
            rollouts: rollout_config(),
            tier_optimizer: TierOptimizerConfig::default(),
            dreaming: None,
            tensor_transaction_directory: root.join("tensor"),
            prospective_directory: root.join("prospective"),
            tier_optimizer_directory: root.join("optimizer"),
            candidate_directory: root.join("candidate"),
            rejection_report_directory: root.join("rejections"),
            max_wake_context_records: 64,
            rng_streams: 1,
        }
    }

    #[test]
    fn dreaming_artifacts_cannot_alias_another_runtime_store() {
        let mut config: BuiltinSleepRuntimeConfig =
            serde_json::from_str(include_str!("../sleep-runtime.periodic.example.json")).unwrap();
        config.dreaming.as_mut().unwrap().artifact_directory = config.candidate_directory.clone();
        let error = config.validate_periodic().unwrap_err().to_string();
        assert!(error.contains("including Dreaming artifacts"), "{error}");
    }

    #[test]
    fn periodic_loader_does_not_require_standalone_wake_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        let config = periodic_runtime(directory.path());
        let path = directory.path().join("runtime.json");
        let bytes = serde_json::to_vec_pretty(&config).unwrap();
        fs::write(&path, &bytes).unwrap();
        let digest = sha256_bytes(&bytes);

        let factory = BuiltinSleepPhaseContextFactory::load_periodic_on_device(
            &path,
            &digest,
            Device::ndarray().autodiff(),
        )
        .unwrap();
        assert_eq!(factory.identity(), digest);
        assert_eq!(factory.config().max_wake_context_records, 64);
        assert!(
            BuiltinSleepPhaseContextFactory::load_on_device(
                &path,
                &digest,
                Device::ndarray().autodiff(),
            )
            .is_err()
        );

        fs::write(&path, b"{}").unwrap();
        assert!(
            BuiltinSleepPhaseContextFactory::load_periodic_on_device(
                &path,
                &digest,
                Device::ndarray().autodiff(),
            )
            .is_err()
        );
    }

    #[test]
    fn periodic_dreaming_does_not_require_a_dummy_wake_journal() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = periodic_runtime(directory.path());
        config.dreaming = Some(BuiltinDreamingRuntimeConfig {
            artifact_directory: directory.path().join("dreams"),
            wake_context_journal: None,
            reference_set: pin(&directory.path().join("reference.json"), b"{}"),
            independent_evaluation_set: pin(&directory.path().join("independent.json"), b"{}"),
            initial_policy: None,
            max_new_tokens: 8,
            gradient_dimensions: 16,
            lora_steps: 1,
            lora_learning_rate: 1e-3,
            restem_learning_rate: 0.1,
            generation_temperature: 0.8,
            generation_policy_rank: 64,
            generation_policy_alpha: 128,
        });
        config.validate_periodic().unwrap();
        assert!(config.validate_standalone().is_err());

        let journal = pin(&directory.path().join("wake.json"), b"{}");
        config.wake_context_journal = Some(journal.clone());
        config.initial_tier_optimizer_state =
            Some(pin(&directory.path().join("optimizer.bin"), b"optimizer"));
        config.initial_model_parameter_ids =
            Some(pin(&directory.path().join("parameter-ids.json"), b"{}"));
        assert!(config.validate_periodic().is_err());
        assert!(config.validate_standalone().is_err());
        config.dreaming.as_mut().unwrap().wake_context_journal = Some(journal);
        config.validate_standalone().unwrap();
        assert!(config.validate_periodic().is_err());
    }

    #[test]
    fn second_periodic_cycle_consumes_first_cycles_committed_policy_after_resume() {
        let directory = tempfile::tempdir().unwrap();
        let artifact_directory = directory.path().join("dreams");
        let first_policy = publish_test_policy(&artifact_directory, 1);
        let mut dreaming = BuiltinDreamingRuntimeConfig {
            artifact_directory,
            wake_context_journal: None,
            reference_set: pin(&directory.path().join("reference.json"), b"{}"),
            independent_evaluation_set: pin(&directory.path().join("evaluation.json"), b"{}"),
            initial_policy: None,
            max_new_tokens: 1,
            gradient_dimensions: 1,
            lora_steps: 1,
            lora_learning_rate: 1e-3,
            restem_learning_rate: 0.1,
            generation_temperature: 0.8,
            generation_policy_rank: 64,
            generation_policy_alpha: 128,
        };
        let schedule = sleep_config(directory.path(), hash('0')).schedule;
        let mut first_cycle = SleepState::new(&schedule, 1).unwrap();
        first_cycle
            .completed_transactions
            .push(completed_transaction_with_policy(1, first_policy.clone()));

        // Exercise the serialized checkpoint form used by exact resume. The
        // pending transaction (if any) is deliberately not consulted by the
        // binding helper.
        let resumed: SleepState =
            serde_json::from_slice(&serde_json::to_vec(&first_cycle).unwrap()).unwrap();
        bind_latest_committed_dream_policy(&mut dreaming, &resumed).unwrap();
        let parent = dreaming.initial_policy.as_ref().unwrap();
        assert_eq!(parent.sha256, first_policy);
        assert!(parent.path.is_file());
        let state: serde_json::Value = parent.verify_json().unwrap();
        assert_eq!(state["transaction_id"], 1);
        assert_eq!(state["adapter_sha256"], hash('d'));
    }

    fn model_source() -> &'static str {
        r#"
        ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
        memory cms {
            tier fast { ffn: base reserve_experts { capacity: 1 rank: 3 top_k: 1 } }
            tier medium { ffn: base residual_init: zero reserve_experts { capacity: 4 rank: 3 top_k: 1 } }
        }
        model sleeper {
            vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
            block: { attention: { num_heads: 1 dropout: 0.0 position_encoding: none } memory: cms dropout: 0.0 }
        }
        "#
    }

    fn test_model() -> Transformer {
        Transformer::new(
            &parse_mal(model_source()).unwrap(),
            &Device::ndarray().autodiff(),
        )
        .unwrap()
    }

    #[test]
    fn standalone_parameter_identity_companion_rebinds_a_fresh_model() {
        let original = test_model();
        let artifact = ModelParameterIdsArtifact::from_model(hash('b'), &original).unwrap();
        let mut reconstructed = test_model();
        assert_ne!(
            list_param_ids(&original)
                .into_iter()
                .map(|id| id.val())
                .collect::<Vec<_>>(),
            list_param_ids(&reconstructed)
                .into_iter()
                .map(|id| id.val())
                .collect::<Vec<_>>()
        );
        restore_parameter_ids(&mut reconstructed, &artifact.parameter_ids).unwrap();
        assert_eq!(
            list_param_ids(&original)
                .into_iter()
                .map(|id| id.val())
                .collect::<Vec<_>>(),
            list_param_ids(&reconstructed)
                .into_iter()
                .map(|id| id.val())
                .collect::<Vec<_>>()
        );
    }

    fn sleep_config(root: &Path, retention_hash: String) -> InModelSleepConfig {
        InModelSleepConfig {
            schedule: SleepSchedule {
                clock: UpdateClock::OptimizerSteps,
                terminal_consolidation: TerminalConsolidation::DistillIntoBaseV1,
                tiers: vec![
                    MemoryTierSchedule {
                        id: "fast".into(),
                        update_period: 100,
                        reserve_slots: 1,
                    },
                    MemoryTierSchedule {
                        id: "medium".into(),
                        update_period: 400,
                        reserve_slots: 4,
                    },
                ],
            },
            standalone_trigger_clock: Some(100),
            knowledge_seeding: KnowledgeSeedingConfig {
                chunk_tokens: 2,
                teacher_rollouts: 1,
                detached_student_rollouts: 1,
                temperature: 1.0,
                forward_kl_weight: 1.0,
            },
            imitation: ImitationConfig {
                semantic_judge_hash: hash('c'),
                semantic_weight: 0.5,
                maximum_edit_distance: 2,
                grpo_group_size: 2,
            },
            dreaming: None,
            retention_suite: root.join("retention.json"),
            retention_suite_sha256: retention_hash.clone(),
            retention: RetentionGateConfig {
                evaluator_hash: hash('d'),
                suite_hash: retention_hash,
                max_anchor_forward_kl: 1.0,
                max_anchor_regression: 1.0,
                min_incorporation_gain: -1.0,
            },
            receiver_learning_rate: 1e-2,
            receiver_weight_decay: 0.0,
            grpo_clip_epsilon: 0.2,
            grpo_advantage_epsilon: 1e-6,
            grpo_kl_coefficient: 0.01,
            candidate_directory: root.join("candidates"),
        }
    }

    #[test]
    fn interruption_cursor_authenticates_the_exact_wake_journal() {
        let directory = tempfile::tempdir().unwrap();
        let retention = pin(&directory.path().join("retention.json"), b"sealed");
        let config = sleep_config(directory.path(), retention.sha256);
        let model = test_model();
        let input = NativeCheckpointRef::new("teacher.safetensors", hash('e')).unwrap();
        let mut cursor =
            NativeSleepCheckpoint::new(hash('f'), "sleep", input, &model, &config, 1).unwrap();
        cursor.advance_clock(&model, &config, 100).unwrap();

        let journal_path = directory.path().join("wake.json");
        let mut journal = WakeContextJournal::new(hash('e')).unwrap();
        journal
            .push(WakeContextRecord {
                id: "wake:100:0".into(),
                optimizer_step: 100,
                token_ids: vec![1, 2, 3],
            })
            .unwrap();
        let pinned = journal.publish(&journal_path).unwrap();
        cursor
            .bind_wake_context_journal(
                PinnedNativeArtifact::from_path(pinned.path(), pinned.sha256()).unwrap(),
            )
            .unwrap();

        let bytes = serde_json::to_vec(&cursor).unwrap();
        let resumed: NativeSleepCheckpoint = serde_json::from_slice(&bytes).unwrap();
        resumed.validate(&model, &config).unwrap();
        assert_eq!(
            resumed.wake_context_journal.as_ref().unwrap().sha256,
            pinned.sha256()
        );

        fs::write(&journal_path, b"tampered").unwrap();
        assert!(resumed.validate(&model, &config).is_err());
    }

    #[derive(Default)]
    struct TestProgress {
        persisted: usize,
    }

    impl NativeSleepProgressSink for TestProgress {
        fn persist(&mut self, _: &NativeSleepCheckpoint) -> Result<()> {
            self.persisted += 1;
            Ok(())
        }
    }

    #[test]
    fn standalone_factory_executes_and_publishes_a_durable_no_due_cursor() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let device = Device::ndarray().autodiff();
        let model_def = parse_mal(model_source()).unwrap();
        let model = Transformer::new(&model_def, &device).unwrap();
        let checkpoint_path = root.join("teacher.safetensors");
        hermes_llm::save_safetensors(&model.clone().valid(), &checkpoint_path).unwrap();
        let checkpoint_bytes = fs::read(&checkpoint_path).unwrap();
        let checkpoint_sha256 = sha256_bytes(&checkpoint_bytes);

        let bank = TierOptimizerBank::new(
            &model,
            &sleep_config(root, hash('0')).schedule,
            TierOptimizerConfig::default(),
        )
        .unwrap();
        let optimizer_state = pin(
            &root.join("tier-state.bin"),
            &bank.snapshot_bytes().unwrap(),
        );
        let identities = ModelParameterIdsArtifact::from_model(&checkpoint_sha256, &model).unwrap();
        let identity_bytes = serde_json::to_vec_pretty(&identities).unwrap();
        let identity_artifact = pin(&root.join("parameter-ids.json"), &identity_bytes);

        let journal_path = root.join("standalone-wake.json");
        let mut journal = WakeContextJournal::new(&checkpoint_sha256).unwrap();
        journal
            .push(WakeContextRecord {
                id: "wake:0:0".into(),
                optimizer_step: 0,
                token_ids: vec![1, 2, 3],
            })
            .unwrap();
        let journal = journal.publish(&journal_path).unwrap();

        let mut runtime = periodic_runtime(root);
        let judge = pin(
            &root.join("judge.json"),
            &serde_json::to_vec_pretty(
                &crate::builtin_sleep_adapters::TokenSemanticJudgeArtifact {
                    version: 1,
                    algorithm: "token_ngram_f1_v1".into(),
                    unigram_weight: 1.0,
                    bigram_weight: 1.0,
                    equivalence_classes: Vec::new(),
                },
            )
            .unwrap(),
        );
        let evaluator = pin(
            &root.join("evaluator.json"),
            &serde_json::to_vec_pretty(
                &crate::builtin_sleep_adapters::LikelihoodRetentionEvaluatorArtifact {
                    version: 1,
                    algorithm: "causal_likelihood_v1".into(),
                },
            )
            .unwrap(),
        );
        let suite = pin(
            &root.join("retention.json"),
            &serde_json::to_vec_pretty(&crate::builtin_sleep_adapters::RetentionSuiteArtifact {
                version: 1,
                stable_anchors: vec![crate::builtin_sleep_adapters::RetentionSequence {
                    id: "anchor".into(),
                    token_ids: vec![1, 2],
                }],
                incorporation: vec![crate::builtin_sleep_adapters::RetentionSequence {
                    id: "new".into(),
                    token_ids: vec![2, 3],
                }],
            })
            .unwrap(),
        );
        let model_mal = pin(&root.join("model.mal"), model_source().as_bytes());
        runtime.model_mal = model_mal;
        runtime.wake_context_journal = Some(PinnedLocalArtifact {
            path: journal.path().to_owned(),
            sha256: journal.sha256().to_owned(),
        });
        runtime.semantic_judge = judge.clone();
        runtime.retention_evaluator = evaluator.clone();
        runtime.retention_suite = suite.clone();
        runtime.initial_tier_optimizer_state = Some(optimizer_state);
        runtime.initial_model_parameter_ids = Some(identity_artifact);
        let runtime_path = root.join("standalone-runtime.json");
        let runtime_bytes = serde_json::to_vec_pretty(&runtime).unwrap();
        fs::write(&runtime_path, &runtime_bytes).unwrap();
        let runtime_sha256 = sha256_bytes(&runtime_bytes);

        let mut sleep = sleep_config(root, suite.sha256.clone());
        sleep.standalone_trigger_clock = Some(0);
        sleep.imitation.semantic_judge_hash = judge.sha256;
        sleep.retention.evaluator_hash = evaluator.sha256;
        sleep.retention.suite_hash = suite.sha256;
        sleep.candidate_directory = runtime.candidate_directory.clone();
        let phase = serde_json::from_value(serde_json::json!({
            "name": "sleep",
            "type": "sleep",
            "sleep": sleep
        }))
        .unwrap();
        let request = PhaseExecutionRequest {
            phase_index: 0,
            phase,
            input_checkpoint: Some(
                ImmutableModelCheckpoint::new(
                    checkpoint_path.to_string_lossy(),
                    checkpoint_sha256.clone(),
                )
                .unwrap(),
            ),
            resume_state: None,
        };
        let mut factory =
            BuiltinSleepPhaseContextFactory::load_on_device(&runtime_path, &runtime_sha256, device)
                .unwrap();
        let config = request.phase.sleep.as_ref().unwrap().clone();
        let mut context = factory.create(&request, &config).unwrap();
        let mut progress = TestProgress::default();
        let outcome = context
            .drive_sleep_phase(&request, &config, None, &mut progress)
            .unwrap();
        let NativeSleepPhaseOutcome::Accepted(cursor) = outcome else {
            panic!("no-due standalone phase did not accept")
        };
        assert_eq!(cursor.live_checkpoint.sha256, checkpoint_sha256);
        assert!(cursor.wake_context_journal.is_some());
        assert!(
            cursor
                .optimizer_scopes
                .tiers
                .iter()
                .all(|tier| tier.artifact.is_some())
        );
        assert!(progress.persisted >= 1);
    }
}
