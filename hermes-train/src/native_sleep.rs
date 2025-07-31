//! First-party, crash-recoverable orchestration for tensor in-model sleep.
//!
//! Model-specific rollout generation and evaluators remain typed adapters, but
//! this module—not a phase worker—owns the state machine, tensor transaction
//! generations, optimizer-scope receipts, rollback, dreaming boundary, and
//! immutable candidate identity.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use hermes_llm::Transformer;
use serde::{Deserialize, Serialize};

use crate::artifact_io::{hash_regular_file, validate_sha256_identity};
use crate::metrics::{
    ActiveCapacityMetrics, DistillationDivergenceMetrics, ImitationRewardMetrics, MemoryTier,
    MetricContext, MetricDirection, MetricEvent, MetricPhase, MetricPhaseKind,
    RetentionDeltaMetrics, TierUpdateMetrics, TierUpdateOutcome,
};
use crate::runtime::{
    ImmutableArtifact, ImmutableModelCheckpoint, PhaseExecutionRequest, PhaseExecutionResult,
    PhaseExecutor, PhaseProduct, PhaseProgressSink,
};
use crate::sleep::{
    ConsolidationBackend, ConsolidationTxn, DreamingBackend, MemoryOptimizerScopes, SleepPhase,
    SleepProgressSink, SleepSchedule, SleepState, TierOptimizerArtifact,
    run_dreaming_with_progress, validate_model_memory_state,
};
use crate::tensor_sleep::{
    AtomicCandidatePublisher, ConsolidationRollouts, ProspectiveTransformerUpdate,
    RetentionEvaluator, SemanticJudge, TensorConsolidationBackend, TensorTransactionPointer,
    TensorTransactionStore,
};
use crate::workflow::InModelSleepConfig;

pub const NATIVE_SLEEP_CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCheckpointRef {
    pub uri: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedNativeArtifact {
    pub path: String,
    pub sha256: String,
}

impl PinnedNativeArtifact {
    pub fn from_path(path: &Path, sha256: impl Into<String>) -> Result<Self> {
        let artifact = Self {
            path: path
                .to_str()
                .context("pinned sleep artifact path is not UTF-8")?
                .to_owned(),
            sha256: sha256.into(),
        };
        artifact.verify()?;
        Ok(artifact)
    }

    /// Verify regular-file type and content on every call. This deliberately
    /// does not trust a prior validation across an interruption boundary.
    pub fn verify(&self) -> Result<()> {
        ensure!(
            !self.path.trim().is_empty(),
            "pinned sleep artifact path is empty"
        );
        validate_sha256_identity(&self.sha256, "pinned sleep artifact hash")?;
        let path = Path::new(&self.path);
        let observed = hash_regular_file(path)
            .with_context(|| format!("reading pinned sleep artifact {}", path.display()))?
            .1;
        ensure!(
            observed == self.sha256,
            "pinned sleep artifact {} changed: expected {}, observed {observed}",
            path.display(),
            self.sha256
        );
        Ok(())
    }
}

impl NativeCheckpointRef {
    pub fn new(uri: impl Into<String>, sha256: impl Into<String>) -> Result<Self> {
        let value = Self {
            uri: uri.into(),
            sha256: sha256.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.uri.trim().is_empty(),
            "native checkpoint URI is empty"
        );
        validate_sha256_identity(&self.sha256, "native checkpoint hash")
    }
}

/// Entire durable cursor owned by the native sleep executor. It is intended
/// to be embedded in WorkflowV2's opaque phase resume state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSleepCheckpoint {
    pub version: u32,
    pub workflow_signature: String,
    pub phase_name: String,
    pub input_checkpoint: NativeCheckpointRef,
    pub live_checkpoint: NativeCheckpointRef,
    pub retention_suite: PinnedNativeArtifact,
    /// Model-owned wake contexts sealed for the current due-boundary. It is
    /// absent between wake boundaries and content-pinned before any sender
    /// transaction begins.
    pub wake_context_journal: Option<PinnedNativeArtifact>,
    pub sleep: SleepState,
    pub optimizer_scopes: MemoryOptimizerScopes,
}

impl NativeSleepCheckpoint {
    pub fn new(
        workflow_signature: impl Into<String>,
        phase_name: impl Into<String>,
        input_checkpoint: NativeCheckpointRef,
        model: &Transformer,
        config: &InModelSleepConfig,
        rng_streams: usize,
    ) -> Result<Self> {
        let schedule = &config.schedule;
        let mut sleep = SleepState::new(schedule, rng_streams)?;
        // A wake checkpoint may already have active fast scratch slots. Import
        // the actual MAL masks/generations instead of assuming an empty model.
        let mut seen_slots = BTreeSet::new();
        for status in model.memory_slot_statuses() {
            let slot = sleep
                .tiers
                .get_mut(status.tier)
                .and_then(|tier| tier.slots.get_mut(status.slot))
                .with_context(|| {
                    format!(
                        "model memory slot {}/{} is absent from the sleep schedule",
                        status.tier, status.slot
                    )
                })?;
            if seen_slots.insert((status.tier, status.slot)) {
                slot.active = status.active;
                slot.generation = status.generation;
            } else {
                ensure!(
                    slot.active == status.active && slot.generation == status.generation,
                    "logical memory slot differs between model layers"
                );
            }
        }
        let optimizer_scopes = MemoryOptimizerScopes::from_model(model, schedule)?;
        let retention_suite = PinnedNativeArtifact::from_path(
            &config.retention_suite,
            config.retention_suite_sha256.clone(),
        )?;
        sleep.evaluator_hashes.push(retention_suite.sha256.clone());
        sleep
            .evaluator_hashes
            .push(config.imitation.semantic_judge_hash.clone());
        sleep
            .evaluator_hashes
            .push(config.retention.evaluator_hash.clone());
        if let Some(dreaming) = &config.dreaming {
            sleep
                .evaluator_hashes
                .push(dreaming.reference_set_hash.clone());
            sleep
                .evaluator_hashes
                .push(dreaming.trial_evaluator_hash.clone());
        }
        let checkpoint = Self {
            version: NATIVE_SLEEP_CHECKPOINT_VERSION,
            workflow_signature: workflow_signature.into(),
            phase_name: phase_name.into(),
            live_checkpoint: input_checkpoint.clone(),
            input_checkpoint,
            retention_suite,
            wake_context_journal: None,
            sleep,
            optimizer_scopes,
        };
        checkpoint.validate(model, config)?;
        Ok(checkpoint)
    }

    pub fn validate(&self, model: &Transformer, config: &InModelSleepConfig) -> Result<()> {
        self.validate_state(model, config)?;
        self.retention_suite.verify()?;
        if let Some(journal) = &self.wake_context_journal {
            journal
                .verify()
                .context("verifying pinned wake-context journal")?;
        }
        Ok(())
    }

    /// Validate the in-memory cursor/model relationship without touching
    /// external artifacts. Ordinary wake steps use this path; content-pinned
    /// files are re-authenticated before a due boundary consumes them.
    fn validate_state(&self, model: &Transformer, config: &InModelSleepConfig) -> Result<()> {
        let schedule = &config.schedule;
        ensure!(
            self.version == NATIVE_SLEEP_CHECKPOINT_VERSION,
            "unsupported native sleep checkpoint version {}",
            self.version
        );
        validate_sha256_identity(&self.workflow_signature, "sleep workflow signature")?;
        ensure!(
            !self.phase_name.trim().is_empty(),
            "sleep phase name is empty"
        );
        self.input_checkpoint.validate()?;
        self.live_checkpoint.validate()?;
        ensure!(
            config.retention_suite.to_str() == Some(self.retention_suite.path.as_str())
                && config.retention_suite_sha256 == self.retention_suite.sha256,
            "native checkpoint retention suite differs from WorkflowV2"
        );
        let mut expected_evaluators = vec![
            self.retention_suite.sha256.clone(),
            config.imitation.semantic_judge_hash.clone(),
            config.retention.evaluator_hash.clone(),
        ];
        if let Some(dreaming) = &config.dreaming {
            expected_evaluators.push(dreaming.reference_set_hash.clone());
            expected_evaluators.push(dreaming.trial_evaluator_hash.clone());
        }
        ensure!(
            self.sleep.evaluator_hashes == expected_evaluators,
            "native sleep evaluator identities or their configured roles have drifted"
        );
        let memory_statuses = model.memory_slot_statuses();
        validate_model_memory_state(schedule, &self.sleep, &memory_statuses)?;
        if let Some(txn) = &self.sleep.pending {
            let expected_knowledge = config.knowledge_seeding.rollout_count_u64()?;
            if let Some(reservation) = txn.knowledge_rng {
                ensure!(
                    reservation.stream == 0 && reservation.count == expected_knowledge,
                    "native knowledge RNG reservation differs from WorkflowV2"
                );
            }
            if let Some(reservation) = txn.imitation_rng {
                ensure!(
                    reservation.stream == 0
                        && reservation.count == config.imitation.group_size_u64()?,
                    "native imitation RNG reservation differs from WorkflowV2"
                );
            }
            match &config.dreaming {
                Some(dreaming) => {
                    if let Some(reservation) = txn.dream_generation_rng {
                        ensure!(
                            reservation.stream == 0
                                && reservation.count
                                    == u64::try_from(dreaming.candidate_count)
                                        .context("dream candidate count exceeds RNG schema")?,
                            "native dream-generation RNG reservation differs from WorkflowV2"
                        );
                    }
                    if let Some(reservation) = txn.dream_selection_rng {
                        ensure!(
                            reservation.stream == 0 && reservation.count == 1,
                            "native dream-selection RNG reservation differs from WorkflowV2"
                        );
                    }
                    ensure!(
                        txn.dream_trial_rngs.iter().all(|trial| {
                            trial.reservation.stream == 0 && trial.reservation.count == 1
                        }),
                        "native dream-trial RNG reservation differs from WorkflowV2"
                    );
                    ensure!(
                        txn.dream_trials
                            .iter()
                            .all(|trial| trial.evaluator_hash == dreaming.trial_evaluator_hash),
                        "native dream trial used an evaluator outside WorkflowV2"
                    );
                    let retained = dreaming.retained_count()?;
                    ensure!(
                        txn.dream_selected.len() <= retained
                            && txn.dream_trials.len() <= txn.dream_selected.len()
                            && txn.dream_trial_rngs.len() <= txn.dream_selected.len(),
                        "native dream evidence exceeds the configured selection quota"
                    );
                    if matches!(
                        self.sleep.phase,
                        SleepPhase::DreamTrials
                            | SleepPhase::DreamPolicyUpdate
                            | SleepPhase::Candidate
                    ) && txn.generated_manifest.is_some()
                    {
                        ensure!(
                            txn.dream_selected.len() == retained,
                            "native dream selection differs from the configured quota"
                        );
                    }
                }
                None => ensure!(
                    txn.dream_generation_rng.is_none()
                        && txn.dream_selection_rng.is_none()
                        && txn.dream_trial_rngs.is_empty()
                        && txn.generated_manifest.is_none()
                        && txn.dream_selected.is_empty()
                        && txn.dream_trials.is_empty()
                        && txn.dream_policy_receipt.is_none(),
                    "native checkpoint contains Dreaming state but WorkflowV2 disables Dreaming"
                ),
            }
        }
        self.optimizer_scopes
            .validate_active_state_after_model_validation(model, schedule, &memory_statuses)?;
        ensure!(
            self.optimizer_scopes
                .tiers
                .iter()
                .zip(&self.sleep.tiers)
                .all(|(scope, tier)| scope.update_clock == tier.last_update_clock
                    && scope.transfer_clock <= self.sleep.clock),
            "native optimizer update clock disagrees with its sleep tier, or a receiver-transfer clock is ahead of the wake/sleep clock"
        );
        if let Some(txn) = &self.sleep.pending
            && txn.committed
        {
            ensure!(
                txn.candidate_checkpoint.as_deref() == Some(self.live_checkpoint.uri.as_str())
                    && txn.candidate_hash.as_deref() == Some(self.live_checkpoint.sha256.as_str()),
                "native live checkpoint differs from committed sleep candidate"
            );
        }
        Ok(())
    }

    /// Validate only state that can change between already authenticated sleep
    /// boundaries. Model topology, active masks, evaluator bindings, and full
    /// optimizer ownership are revalidated at construction/resume and before a
    /// boundary consumes them; rebuilding their parameter-ID sets on every
    /// ordinary wake step would put a large CPU allocation in the hot path.
    fn validate_wake_tick(&self, config: &InModelSleepConfig) -> Result<()> {
        ensure!(
            self.version == NATIVE_SLEEP_CHECKPOINT_VERSION,
            "unsupported native sleep checkpoint version {}",
            self.version
        );
        ensure!(
            self.sleep.phase == SleepPhase::Wake
                && self.sleep.pending.is_none()
                && self.sleep.due_senders.is_empty()
                && self.sleep.due_clocks.is_empty(),
            "ordinary wake clock advance found an unfinished sleep boundary"
        );
        ensure!(
            self.sleep.tiers.len() == config.schedule.tiers.len()
                && self.optimizer_scopes.tiers.len() == config.schedule.tiers.len(),
            "native sleep tier count differs from WorkflowV2"
        );
        for (tier, ((saved, scope), configured)) in self
            .sleep
            .tiers
            .iter()
            .zip(&self.optimizer_scopes.tiers)
            .zip(&config.schedule.tiers)
            .enumerate()
        {
            ensure!(
                saved.id == configured.id
                    && saved.slots.len() == configured.reserve_slots
                    && scope.tier == tier
                    && scope.tier_id == configured.id
                    && scope.update_clock == saved.last_update_clock
                    && scope.transfer_clock <= self.sleep.clock,
                "native wake tier `{}` differs from its schedule/optimizer clock",
                configured.id
            );
        }
        Ok(())
    }

    /// Bind the immutable model-owned contexts before starting the first
    /// sender at a boundary. An in-flight boundary cannot switch journals.
    pub fn bind_wake_context_journal(&mut self, journal: PinnedNativeArtifact) -> Result<()> {
        journal.verify()?;
        ensure!(
            self.sleep.phase == SleepPhase::Wake && self.sleep.pending.is_none(),
            "cannot replace wake-context journal during a sleep transaction"
        );
        self.wake_context_journal = Some(journal);
        Ok(())
    }

    /// Advance the immutable live pointer after an ordinary wake checkpoint
    /// and invalidate contexts sealed for the previous weights.
    pub fn record_wake_checkpoint(&mut self, checkpoint: NativeCheckpointRef) -> Result<()> {
        checkpoint.validate()?;
        ensure!(
            self.sleep.phase == SleepPhase::Wake && self.sleep.pending.is_none(),
            "cannot publish a wake checkpoint during a sleep transaction"
        );
        self.live_checkpoint = checkpoint;
        self.wake_context_journal = None;
        Ok(())
    }

    pub fn advance_clock(
        &mut self,
        model: &Transformer,
        config: &InModelSleepConfig,
        clock: u64,
    ) -> Result<()> {
        let reaches_boundary = reaches_sleep_boundary(&self.sleep, &config.schedule, clock)?;
        if reaches_boundary {
            self.validate_state(model, config)?;
            self.retention_suite.verify()?;
            if let Some(journal) = &self.wake_context_journal {
                journal
                    .verify()
                    .context("verifying pinned wake-context journal")?;
            }
        } else {
            self.validate_wake_tick(config)?;
        }
        // SleepState computes every fallible boundary decision before it
        // mutates clocks/queues, so no full checkpoint clone is required for
        // failure atomicity on the wake hot path.
        self.sleep.advance_clock(&config.schedule, clock)
    }

    pub fn begin_next(&mut self, plan: PlannedConsolidation) -> Result<ConsolidationTxn> {
        plan.validate()?;
        let sender = self
            .sleep
            .next_due_sender()
            .context("native sleep checkpoint has no due sender")?;
        self.sleep.begin(
            sender,
            self.live_checkpoint.uri.clone(),
            self.live_checkpoint.sha256.clone(),
            plan.student_checkpoint,
            plan.student_sha256,
            plan.prospective_update_sha256,
        )
    }

    /// Model topology paired with the durable tier optimizer scopes. Before
    /// commit those scopes still describe the teacher; after commit they
    /// include the activated receiver and reclaimed sender masks in `live`.
    pub(crate) fn optimizer_scope_checkpoint(&self) -> Result<NativeCheckpointRef> {
        match self.sleep.pending.as_ref() {
            Some(txn) if !txn.committed => {
                NativeCheckpointRef::new(&txn.teacher_checkpoint, &txn.teacher_hash)
            }
            _ => Ok(self.live_checkpoint.clone()),
        }
    }
}

fn reaches_sleep_boundary(
    state: &SleepState,
    schedule: &SleepSchedule,
    clock: u64,
) -> Result<bool> {
    for (saved, configured) in state.tiers.iter().zip(&schedule.tiers) {
        let next = saved
            .last_boundary_clock
            .checked_div(configured.update_period)
            .and_then(|multiple| multiple.checked_add(1))
            .and_then(|multiple| multiple.checked_mul(configured.update_period))
            .context("sleep boundary clock overflow")?;
        if next <= clock {
            return Ok(true);
        }
    }
    Ok(false)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedConsolidation {
    pub student_checkpoint: String,
    pub student_sha256: String,
    pub prospective_update_sha256: String,
}

impl PlannedConsolidation {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.student_checkpoint.trim().is_empty(),
            "prospective student checkpoint is empty"
        );
        validate_sha256_identity(&self.student_sha256, "prospective student hash")?;
        validate_sha256_identity(
            &self.prospective_update_sha256,
            "prospective optimizer-update hash",
        )
    }
}

/// Publishes the independent sender and receiver optimizer/accumulator bundles
/// after tensor publication but before the native checkpoint is advanced.
/// Receipts must cover each touched tier exactly once. Implementations must be
/// transaction-idempotent: retrying the same transaction/tensor generation
/// returns identical immutable receipts and must not apply another optimizer
/// step. An error must leave the previously published generation unchanged.
pub trait TierOptimizerPublisher {
    fn publish(
        &mut self,
        txn: &ConsolidationTxn,
        tensor: &TensorTransactionPointer,
        receiver_parameter_ids: &[u64],
    ) -> Result<Vec<TierOptimizerCommit>>;
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TierOptimizerCommit {
    pub tier: usize,
    pub role: TierOptimizerCommitRole,
    pub artifact: TierOptimizerArtifact,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TierOptimizerCommitRole {
    SenderUpdate,
    ReceiverTransfer,
    /// The terminal tier's prospective update and reserve-to-base
    /// distillation are sealed into one optimizer bundle.
    TerminalCombined,
}

/// Persistence hook. Implementations atomically replace one native checkpoint
/// and may then mirror it into WorkflowV2's `resume_state`.
pub trait NativeSleepProgressSink {
    fn persist(&mut self, checkpoint: &NativeSleepCheckpoint) -> Result<()>;

    fn metric(&mut self, _: &NativeSleepCheckpoint, event: MetricEvent) -> Result<()> {
        event.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeConsolidationOutcome {
    Accepted(NativeCheckpointRef),
    Rejected(NativeCheckpointRef),
}

#[derive(Clone, Debug, PartialEq)]
pub enum NativeSleepPhaseOutcome {
    Yielded(NativeSleepCheckpoint),
    Accepted(NativeSleepCheckpoint),
    Rejected {
        checkpoint: NativeSleepCheckpoint,
        report_uri: String,
        report_sha256: String,
    },
}

/// In-process model/evaluator registry used by [`NativeSleepPhaseExecutor`].
/// Implementations call [`execute_native_consolidation`] and
/// [`execute_native_dreaming`]; unlike the generic JSONL worker, this context
/// holds typed tensor adapters and cannot substitute another phase kind.
pub trait NativeSleepPhaseContext {
    fn drive_sleep_phase(
        &mut self,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
        resume: Option<NativeSleepCheckpoint>,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<NativeSleepPhaseOutcome>;
}

/// Factory for deployment-owned model loaders, rollout generators, judges,
/// evaluators, and immutable publishers. The registry keeps those typed
/// adapters in process while WorkflowV2 remains algorithm-neutral.
pub trait NativeSleepPhaseContextFactory {
    /// Content identity of the deployment-owned loader/adapter configuration.
    /// Embedded workflow hosts bind this value into their atomic resume state.
    fn identity(&self) -> &str;

    fn create(
        &mut self,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
    ) -> Result<Box<dyn NativeSleepPhaseContext>>;
}

/// Runtime-owned dispatch point for native sleep. There is deliberately no
/// external-worker fallback: deployments must explicitly register the typed
/// adapter factory whose frozen artifacts are pinned by WorkflowV2.
#[derive(Default)]
pub struct NativeSleepContextRegistry {
    phase_factory: Option<Box<dyn NativeSleepPhaseContextFactory>>,
    periodic_driver: Option<Box<dyn PeriodicSleepBoundaryDriver>>,
}

impl NativeSleepContextRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_phase_factory<F>(&mut self, factory: F) -> Result<()>
    where
        F: NativeSleepPhaseContextFactory + 'static,
    {
        validate_sha256_identity(factory.identity(), "native sleep factory identity")?;
        ensure!(
            self.phase_factory.is_none(),
            "a native sleep phase factory is already registered"
        );
        self.phase_factory = Some(Box::new(factory));
        Ok(())
    }

    pub fn register_periodic_driver<D>(&mut self, driver: D) -> Result<()>
    where
        D: PeriodicSleepBoundaryDriver + 'static,
    {
        ensure!(
            self.periodic_driver.is_none(),
            "a periodic native sleep driver is already registered"
        );
        self.periodic_driver = Some(Box::new(driver));
        Ok(())
    }

    pub fn has_phase_factory(&self) -> bool {
        self.phase_factory.is_some()
    }

    pub fn phase_factory_identity(&self) -> Option<&str> {
        self.phase_factory
            .as_deref()
            .map(|factory| factory.identity())
    }

    pub fn has_periodic_driver(&self) -> bool {
        self.periodic_driver.is_some()
    }
}

impl NativeSleepPhaseContext for NativeSleepContextRegistry {
    fn drive_sleep_phase(
        &mut self,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
        resume: Option<NativeSleepCheckpoint>,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<NativeSleepPhaseOutcome> {
        let factory = self.phase_factory.as_mut().with_context(|| {
            format!(
                "workflow sleep phase `{}` requires a registered in-process NativeSleepPhaseContextFactory; external sleep workers are disabled",
                request.phase.name
            )
        })?;
        factory
            .create(request, config)?
            .drive_sleep_phase(request, config, resume, progress)
    }
}

#[derive(Clone, Debug)]
pub struct NativeSleepPhaseExecutor {
    workflow_signature: String,
}

impl NativeSleepPhaseExecutor {
    pub fn new(workflow_signature: impl Into<String>) -> Result<Self> {
        let workflow_signature = workflow_signature.into();
        validate_sha256_identity(&workflow_signature, "native executor workflow signature")?;
        Ok(Self { workflow_signature })
    }

    fn validate_cursor_identity(
        &self,
        cursor: &NativeSleepCheckpoint,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
    ) -> Result<()> {
        ensure!(
            cursor.version == NATIVE_SLEEP_CHECKPOINT_VERSION,
            "unsupported native sleep checkpoint version {}",
            cursor.version
        );
        cursor.input_checkpoint.validate()?;
        cursor.live_checkpoint.validate()?;
        cursor
            .sleep
            .validate_resume()
            .context("validating native sleep phase cursor")?;
        if let Some(journal) = &cursor.wake_context_journal {
            journal
                .verify()
                .context("verifying native sleep phase wake-context journal")?;
        }
        let input = request
            .input_checkpoint
            .as_ref()
            .context("native sleep phase requires an input checkpoint")?;
        ensure!(
            cursor.workflow_signature == self.workflow_signature,
            "native sleep cursor belongs to another workflow"
        );
        ensure!(
            cursor.phase_name == request.phase.name,
            "native sleep cursor belongs to another phase"
        );
        ensure!(
            cursor.input_checkpoint.uri == input.uri()
                && cursor.input_checkpoint.sha256 == input.sha256(),
            "native sleep cursor belongs to another input checkpoint"
        );
        let expected_path = config
            .retention_suite
            .to_str()
            .context("configured retention-suite path is not UTF-8")?;
        ensure!(
            cursor.retention_suite.path == expected_path
                && cursor.retention_suite.sha256 == config.retention_suite_sha256,
            "native sleep cursor retention suite differs from WorkflowV2"
        );
        cursor.retention_suite.verify()?;
        let mut expected_evaluators = vec![
            config.retention_suite_sha256.clone(),
            config.imitation.semantic_judge_hash.clone(),
            config.retention.evaluator_hash.clone(),
        ];
        if let Some(dreaming) = &config.dreaming {
            expected_evaluators.push(dreaming.reference_set_hash.clone());
            expected_evaluators.push(dreaming.trial_evaluator_hash.clone());
        }
        ensure!(
            cursor.sleep.evaluator_hashes == expected_evaluators,
            "native sleep cursor evaluator identities or configured roles have drifted"
        );
        if let Some(trigger) = config.standalone_trigger_clock {
            ensure!(
                cursor.sleep.clock == trigger,
                "native sleep cursor clock differs from its standalone trigger"
            );
        }
        Ok(())
    }
}

struct WorkflowProgressBridge<'a> {
    inner: &'a mut dyn PhaseProgressSink,
    phase_index: u32,
    phase_name: String,
}

impl NativeSleepProgressSink for WorkflowProgressBridge<'_> {
    fn persist(&mut self, checkpoint: &NativeSleepCheckpoint) -> Result<()> {
        self.inner.checkpoint(serde_json::to_value(checkpoint)?)
    }

    fn metric(&mut self, checkpoint: &NativeSleepCheckpoint, event: MetricEvent) -> Result<()> {
        self.inner.metric(
            MetricContext {
                global_step: checkpoint.sleep.clock,
                phase: MetricPhase {
                    index: self.phase_index,
                    name: self.phase_name.clone(),
                    kind: MetricPhaseKind::Sleep,
                },
                checkpoint_hash: Some(checkpoint.live_checkpoint.sha256.clone()),
            },
            event,
        )
    }
}

impl<C: NativeSleepPhaseContext> PhaseExecutor<C> for NativeSleepPhaseExecutor {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        context: &mut C,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        ensure!(
            request.phase.kind == crate::workflow::PhaseKind::Sleep,
            "native sleep executor received non-sleep phase `{}`",
            request.phase.name
        );
        let config = request
            .phase
            .sleep
            .as_ref()
            .context("native sleep phase has no typed settings")?;
        let input = request
            .input_checkpoint
            .as_ref()
            .context("native sleep phase requires an immutable input checkpoint")?;
        let resume = request
            .resume_state
            .clone()
            .map(serde_json::from_value::<NativeSleepCheckpoint>)
            .transpose()
            .context("invalid native sleep resume cursor")?;
        if let Some(cursor) = &resume {
            self.validate_cursor_identity(cursor, request, config)?;
        } else {
            ensure!(
                config.standalone_trigger_clock.is_some(),
                "standalone native sleep execution has no trigger clock"
            );
            PinnedNativeArtifact::from_path(
                &config.retention_suite,
                config.retention_suite_sha256.clone(),
            )?;
        }
        let mut bridge = WorkflowProgressBridge {
            inner: progress,
            phase_index: request
                .phase_index
                .try_into()
                .context("sleep phase index exceeds metric schema")?,
            phase_name: request.phase.name.clone(),
        };
        match context.drive_sleep_phase(request, config, resume, &mut bridge)? {
            NativeSleepPhaseOutcome::Yielded(cursor) => {
                self.validate_cursor_identity(&cursor, request, config)?;
                Ok(PhaseExecutionResult::Yielded {
                    resume_state: serde_json::to_value(cursor)?,
                })
            }
            NativeSleepPhaseOutcome::Accepted(cursor) => {
                self.validate_cursor_identity(&cursor, request, config)?;
                ensure!(
                    cursor.sleep.phase == SleepPhase::Wake
                        && cursor.sleep.pending.is_none()
                        && cursor.sleep.due_senders.is_empty()
                        && cursor.sleep.due_clocks.is_empty(),
                    "native sleep phase accepted before every due sender completed"
                );
                Ok(PhaseExecutionResult::Complete(
                    PhaseProduct::ModelCandidate {
                        checkpoint: ImmutableModelCheckpoint::new(
                            cursor.live_checkpoint.uri,
                            cursor.live_checkpoint.sha256,
                        )?,
                    },
                ))
            }
            NativeSleepPhaseOutcome::Rejected {
                checkpoint,
                report_uri,
                report_sha256,
            } => {
                self.validate_cursor_identity(&checkpoint, request, config)?;
                ensure!(
                    checkpoint.live_checkpoint.uri == input.uri()
                        && checkpoint.live_checkpoint.sha256 == input.sha256(),
                    "rejected native sleep phase changed the shared candidate"
                );
                Ok(PhaseExecutionResult::Complete(
                    PhaseProduct::MutationRejected {
                        report: ImmutableArtifact::new(report_uri, report_sha256)?,
                    },
                ))
            }
        }
    }
}

/// Typed hook used by wake training at a periodic boundary. One call must
/// finish exactly the current `next_due_sender`; the controller enforces
/// fastest-to-slowest draining before the next wake optimizer step.
pub trait PeriodicSleepBoundaryDriver {
    fn drain_due_sender(
        &mut self,
        checkpoint: &mut NativeSleepCheckpoint,
        config: &InModelSleepConfig,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<()>;
}

/// Minimal crash-safe tensor publication contract used by native sleep. The
/// filesystem-backed [`TensorTransactionStore`] is the production
/// implementation; keeping the boundary explicit also permits deterministic
/// crash injection in acceptance tests.
pub trait NativeTensorTransactionStore {
    fn publish(
        &self,
        txn: &ConsolidationTxn,
        recovered: &crate::tensor_sleep::RecoveredTensorTransaction,
    ) -> Result<TensorTransactionPointer>;
}

impl NativeTensorTransactionStore for TensorTransactionStore {
    fn publish(
        &self,
        txn: &ConsolidationTxn,
        recovered: &crate::tensor_sleep::RecoveredTensorTransaction,
    ) -> Result<TensorTransactionPointer> {
        TensorTransactionStore::publish(self, txn, recovered)
    }
}

impl PeriodicSleepBoundaryDriver for NativeSleepContextRegistry {
    fn drain_due_sender(
        &mut self,
        checkpoint: &mut NativeSleepCheckpoint,
        config: &InModelSleepConfig,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<()> {
        self.periodic_driver
            .as_mut()
            .context(
                "periodic in-model sleep requires a registered in-process PeriodicSleepBoundaryDriver",
            )?
            .drain_due_sender(checkpoint, config, progress)
    }
}

pub fn drain_periodic_sleep_before_wake_step<D, S>(
    checkpoint: &mut NativeSleepCheckpoint,
    model: &Transformer,
    config: &InModelSleepConfig,
    clock: u64,
    driver: &mut D,
    sink: &mut S,
) -> Result<NativeCheckpointRef>
where
    D: PeriodicSleepBoundaryDriver,
    S: NativeSleepProgressSink,
{
    let mut advanced = checkpoint.clone();
    advanced.advance_clock(model, config, clock)?;
    sink.persist(&advanced)?;
    *checkpoint = advanced;
    while let Some(sender) = checkpoint.sleep.next_due_sender() {
        let trigger_clock = checkpoint.sleep.due_clocks[0];
        driver.drain_due_sender(checkpoint, config, sink)?;
        ensure!(
            checkpoint.sleep.phase == SleepPhase::Wake && checkpoint.sleep.pending.is_none(),
            "periodic sleep driver left sender {sender} in an unfinished subphase"
        );
        ensure!(
            checkpoint
                .sleep
                .due_clocks
                .first()
                .copied()
                .zip(checkpoint.sleep.next_due_sender())
                != Some((trigger_clock, sender)),
            "periodic sleep driver did not consume boundary ({trigger_clock}, {sender})"
        );
        sink.persist(checkpoint)?;
    }
    Ok(checkpoint.live_checkpoint.clone())
}

fn persist_tensor_boundary<U, R, J, E, P, T, S>(
    checkpoint: &mut NativeSleepCheckpoint,
    backend: &TensorConsolidationBackend<U, R, J, E, P>,
    store: &T,
    sink: &mut S,
) -> Result<TensorTransactionPointer>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
    T: NativeTensorTransactionStore,
    S: NativeSleepProgressSink,
{
    let txn = checkpoint
        .sleep
        .pending
        .as_ref()
        .context("native sleep boundary has no transaction")?;
    let pointer = store.publish(txn, &backend.snapshot_inflight(txn)?)?;
    checkpoint
        .sleep
        .record_tensor_transaction(pointer.generation.clone(), pointer.manifest_sha256.clone())?;
    sink.persist(checkpoint)?;
    Ok(pointer)
}

fn metric_tier(schedule: &SleepSchedule, tier: usize) -> Result<MemoryTier> {
    let configured = schedule
        .tiers
        .get(tier)
        .with_context(|| format!("metric tier {tier} is absent from sleep schedule"))?;
    match configured.id.to_ascii_lowercase().as_str() {
        "fast" => Ok(MemoryTier::Fast),
        "medium" => Ok(MemoryTier::Medium),
        "slow" => Ok(MemoryTier::Slow),
        _ if tier == 0 => Ok(MemoryTier::Fast),
        _ if tier + 1 == schedule.tiers.len() => Ok(MemoryTier::Slow),
        _ => Ok(MemoryTier::Medium),
    }
}

fn txn_metric_id(txn: &ConsolidationTxn) -> String {
    format!("sleep-{}", txn.id)
}

fn emit_knowledge_metric<S: NativeSleepProgressSink>(
    checkpoint: &NativeSleepCheckpoint,
    txn: &ConsolidationTxn,
    diagnostics: crate::tensor_sleep::TensorSleepDiagnostics,
    sink: &mut S,
) -> Result<()> {
    let updates = diagnostics.knowledge_updates as f64;
    ensure!(
        updates > 0.0 && diagnostics.knowledge_tokens > 0,
        "knowledge seeding completed without metric evidence"
    );
    sink.metric(
        checkpoint,
        MetricEvent::DistillationDivergence(DistillationDivergenceMetrics {
            transaction_id: txn_metric_id(txn),
            teacher_hash: txn.teacher_hash.clone(),
            student_hash: txn.student_hash.clone(),
            chunk_index: 0,
            chunk_count: 1,
            selected_tokens: diagnostics.knowledge_tokens,
            total_tokens: diagnostics.knowledge_tokens,
            forward_kl: diagnostics.knowledge_kl_sum / updates,
            reverse_kl: None,
            teacher_entropy: diagnostics.teacher_entropy_sum / updates,
            student_entropy: diagnostics.student_entropy_sum / updates,
        }),
    )
}

fn emit_imitation_metric<S: NativeSleepProgressSink>(
    checkpoint: &NativeSleepCheckpoint,
    txn: &ConsolidationTxn,
    config: &InModelSleepConfig,
    diagnostics: crate::tensor_sleep::TensorSleepDiagnostics,
    sink: &mut S,
) -> Result<()> {
    let samples = diagnostics.imitation_samples;
    ensure!(samples > 0, "imitation completed without metric evidence");
    let count = samples as f64;
    let reward_mean = diagnostics.imitation_reward_sum / count;
    let reward_variance =
        (diagnostics.imitation_reward_square_sum / count - reward_mean * reward_mean).max(0.0);
    sink.metric(
        checkpoint,
        MetricEvent::ImitationReward(ImitationRewardMetrics {
            transaction_id: txn_metric_id(txn),
            semantic_judge_hash: config.imitation.semantic_judge_hash.clone(),
            samples,
            semantic_score_mean: (diagnostics.imitation_semantic_sum / count).clamp(0.0, 1.0),
            normalized_levenshtein_mean: (diagnostics.imitation_edit_sum / count).clamp(0.0, 1.0),
            levenshtein_threshold: (diagnostics.imitation_threshold_sum / count).clamp(0.0, 1.0),
            reward_mean,
            reward_stddev: reward_variance.sqrt(),
            grpo_kl: diagnostics.imitation_grpo_kl_sum / diagnostics.imitation_updates as f64,
        }),
    )
}

fn emit_retention_metrics<S: NativeSleepProgressSink>(
    checkpoint: &NativeSleepCheckpoint,
    txn: &ConsolidationTxn,
    config: &InModelSleepConfig,
    diagnostics: crate::tensor_sleep::TensorSleepDiagnostics,
    sink: &mut S,
) -> Result<()> {
    let transaction_id = txn_metric_id(txn);
    for metric in [
        RetentionDeltaMetrics {
            transaction_id: transaction_id.clone(),
            suite: "retention-suite".into(),
            metric: "anchor-forward-kl".into(),
            evaluator_hash: config.retention.evaluator_hash.clone(),
            direction: MetricDirection::LowerIsBetter,
            baseline_score: 0.0,
            candidate_score: f64::from(diagnostics.retention_anchor_kl),
            improvement: -f64::from(diagnostics.retention_anchor_kl),
            maximum_allowed_regression: f64::from(config.retention.max_anchor_forward_kl),
            passed: diagnostics.retention_anchor_kl <= config.retention.max_anchor_forward_kl,
        },
        RetentionDeltaMetrics {
            transaction_id: transaction_id.clone(),
            suite: "retention-suite".into(),
            metric: "stable-anchor".into(),
            evaluator_hash: config.retention.evaluator_hash.clone(),
            direction: MetricDirection::HigherIsBetter,
            baseline_score: f64::from(diagnostics.teacher_stable_anchor),
            candidate_score: f64::from(diagnostics.student_stable_anchor),
            improvement: f64::from(diagnostics.anchor_delta),
            maximum_allowed_regression: f64::from(config.retention.max_anchor_regression),
            passed: diagnostics.anchor_delta >= -config.retention.max_anchor_regression,
        },
        RetentionDeltaMetrics {
            transaction_id,
            suite: "retention-suite".into(),
            metric: "incorporation".into(),
            evaluator_hash: config.retention.evaluator_hash.clone(),
            direction: MetricDirection::HigherIsBetter,
            baseline_score: f64::from(diagnostics.teacher_incorporation),
            candidate_score: f64::from(diagnostics.student_incorporation),
            improvement: f64::from(diagnostics.incorporation_gain),
            maximum_allowed_regression: 0.0,
            passed: diagnostics.incorporation_gain >= config.retention.min_incorporation_gain,
        },
    ] {
        sink.metric(checkpoint, MetricEvent::RetentionDelta(metric))?;
    }
    Ok(())
}

fn emit_tier_update_metric<S: NativeSleepProgressSink>(
    checkpoint: &NativeSleepCheckpoint,
    schedule: &SleepSchedule,
    txn: &ConsolidationTxn,
    accumulated_micro_steps: u64,
    outcome: TierUpdateOutcome,
    sink: &mut S,
) -> Result<()> {
    let sender = schedule
        .tiers
        .get(txn.sender)
        .context("metric sender tier is absent from sleep schedule")?;
    let receiver_slot = (!txn.terminal)
        .then(|| u32::try_from(txn.receiver_slot))
        .transpose()
        .context("receiver slot exceeds metric schema")?;
    let receiver_generation = (!txn.terminal)
        .then(|| checkpoint.sleep.tiers[txn.receiver].slots[txn.receiver_slot].generation);
    sink.metric(
        checkpoint,
        MetricEvent::TierUpdate(TierUpdateMetrics {
            transaction_id: txn_metric_id(txn),
            tier: metric_tier(schedule, txn.sender)?,
            receiver_tier: (!txn.terminal)
                .then(|| metric_tier(schedule, txn.receiver))
                .transpose()?,
            tier_clock: txn.trigger_clock,
            update_period: sender.update_period,
            accumulated_micro_steps,
            outcome,
            update_l2_norm: None,
            reserve_slot: receiver_slot,
            reserve_generation: receiver_generation,
            optimizer_state_reset: outcome == TierUpdateOutcome::Committed
                && !txn.sender_slots_to_reset.is_empty(),
        }),
    )
}

fn checked_add_product(total: &mut u64, factors: &[usize], label: &str) -> Result<()> {
    let value = factors.iter().try_fold(1_u64, |product, factor| {
        product
            .checked_mul(
                (*factor)
                    .try_into()
                    .context("parameter count exceeds u64")?,
            )
            .with_context(|| format!("{label} parameter count overflow"))
    })?;
    *total = total
        .checked_add(value)
        .with_context(|| format!("{label} parameter count overflow"))?;
    Ok(())
}

fn emit_active_capacity_metrics<S: NativeSleepProgressSink>(
    checkpoint: &NativeSleepCheckpoint,
    model: &Transformer,
    schedule: &SleepSchedule,
    sink: &mut S,
) -> Result<()> {
    let definition = model.config();
    for tier in 0..schedule.tiers.len() {
        let active_reserve = checkpoint.sleep.tiers[tier]
            .slots
            .iter()
            .filter(|slot| slot.active)
            .count();
        let dormant_reserve = checkpoint.sleep.tiers[tier]
            .slots
            .len()
            .checked_sub(active_reserve)
            .context("active reserve count exceeds stored capacity")?;
        let mut stored_parameters = 0_u64;
        let mut routed_active_parameters = 0_u64;
        let mut configured_pool = None;
        let mut configured_top_k = None;
        for layer in 0..definition.num_layers {
            let block = definition
                .pattern
                .as_ref()
                .map(|pattern| &pattern[layer % pattern.len()])
                .unwrap_or(&definition.block);
            let memory = block
                .memory
                .as_ref()
                .with_context(|| format!("layer {layer} has no memory chain"))?;
            let tier_def = memory
                .tiers
                .get(tier)
                .with_context(|| format!("layer {layer} has no memory tier {tier}"))?;
            let hidden = definition.hidden_size;
            let intermediate = tier_def.ffn.hidden_dim.unwrap_or(hidden * 4);
            let gate_factor = usize::from(tier_def.ffn.gate) + 1;
            let mut expert_parameters = 0_u64;
            checked_add_product(
                &mut expert_parameters,
                &[hidden, intermediate, gate_factor],
                "FFN input projection",
            )?;
            checked_add_product(
                &mut expert_parameters,
                &[intermediate, hidden],
                "FFN output projection",
            )?;
            if tier_def.ffn.bias {
                checked_add_product(
                    &mut expert_parameters,
                    &[intermediate, gate_factor],
                    "FFN input bias",
                )?;
                checked_add_product(&mut expert_parameters, &[hidden], "FFN output bias")?;
            }
            let (base_pool, base_top_k, base_stored, base_routed) =
                if let Some(moe) = &tier_def.ffn.moe {
                    let pool = moe
                        .experts
                        .checked_add(moe.shared_experts)
                        .context("MoE expert count overflow")?;
                    let mut stored = 0_u64;
                    checked_add_product(&mut stored, &[hidden, moe.experts], "MoE router")?;
                    checked_add_product(
                        &mut stored,
                        &[
                            pool,
                            expert_parameters
                                .try_into()
                                .context("expert size exceeds usize")?,
                        ],
                        "MoE experts",
                    )?;
                    let mut routed = 0_u64;
                    checked_add_product(&mut routed, &[hidden, moe.experts], "MoE router")?;
                    checked_add_product(
                        &mut routed,
                        &[
                            moe.top_k + moe.shared_experts,
                            expert_parameters
                                .try_into()
                                .context("expert size exceeds usize")?,
                        ],
                        "routed MoE experts",
                    )?;
                    (pool, moe.top_k, stored, routed)
                } else {
                    (1, 1, expert_parameters, expert_parameters)
                };
            ensure!(
                configured_pool.is_none_or(|value| value == base_pool)
                    && configured_top_k.is_none_or(|value| value == base_top_k),
                "memory tier {tier} has inconsistent expert geometry between layers"
            );
            configured_pool = Some(base_pool);
            configured_top_k = Some(base_top_k);
            stored_parameters = stored_parameters
                .checked_add(base_stored)
                .context("stored base parameter count overflow")?;
            routed_active_parameters = routed_active_parameters
                .checked_add(base_routed)
                .context("routed base parameter count overflow")?;

            let reserve = &tier_def.reserve_experts;
            ensure!(
                reserve.capacity == schedule.tiers[tier].reserve_slots,
                "MAL reserve capacity differs from sleep schedule"
            );
            let mut reserve_slot_parameters = 0_u64;
            checked_add_product(
                &mut reserve_slot_parameters,
                &[hidden],
                "reserve router row",
            )?;
            checked_add_product(
                &mut reserve_slot_parameters,
                &[2, hidden, reserve.rank],
                "reserve low-rank factors",
            )?;
            checked_add_product(
                &mut stored_parameters,
                &[
                    reserve.capacity,
                    reserve_slot_parameters
                        .try_into()
                        .context("reserve slot size exceeds usize")?,
                ],
                "stored reserve experts",
            )?;
            // Router scoring covers every preallocated row even though only
            // top-k low-rank experts execute. Before any real slot is active
            // the model executes its deterministic zero fallback; activation
            // replaces that lane rather than adding a route.
            checked_add_product(
                &mut routed_active_parameters,
                &[reserve.capacity, hidden],
                "fixed reserve router lanes",
            )?;
            checked_add_product(
                &mut routed_active_parameters,
                &[reserve.top_k, 2, hidden, reserve.rank],
                "fixed routed reserve expert lane",
            )?;
        }
        sink.metric(
            checkpoint,
            MetricEvent::ActiveCapacity(ActiveCapacityMetrics {
                tier: metric_tier(schedule, tier)?,
                active_base_experts: configured_pool
                    .context("memory tier has no layers")?
                    .try_into()
                    .context("base expert count exceeds metric schema")?,
                active_reserve_experts: active_reserve
                    .try_into()
                    .context("active reserve count exceeds metric schema")?,
                dormant_reserve_experts: dormant_reserve
                    .try_into()
                    .context("dormant reserve count exceeds metric schema")?,
                routed_top_k: configured_top_k
                    .context("memory tier has no routing geometry")?
                    .try_into()
                    .context("top-k exceeds metric schema")?,
                reserve_routed_top_k: 1,
                routed_active_parameters,
                stored_parameters,
                dream_generation: false,
                random_extra_expert: false,
            }),
        )?;
    }
    Ok(())
}

/// Drive the non-dreaming half of one due sender transaction entirely in
/// process. Every subphase seals tensor/optimizer state before publishing its
/// semantic cursor. A failed pre-commit operation restores the teacher and
/// publishes a rollback cursor; a post-commit persistence failure is retried
/// and never rolled back.
pub fn execute_native_consolidation<U, R, J, E, P, T, O, S>(
    checkpoint: &mut NativeSleepCheckpoint,
    config: &InModelSleepConfig,
    backend: &mut TensorConsolidationBackend<U, R, J, E, P>,
    store: &T,
    optimizer_publisher: &mut O,
    sink: &mut S,
) -> Result<NativeConsolidationOutcome>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
    T: NativeTensorTransactionStore,
    O: TierOptimizerPublisher,
    S: NativeSleepProgressSink,
{
    let schedule = &config.schedule;
    ensure!(
        backend.config() == &config.tensor_config(),
        "tensor backend configuration differs from signed WorkflowV2 sleep settings"
    );
    let txn = checkpoint
        .sleep
        .pending
        .clone()
        .context("native consolidation has no transaction")?;
    let sender_accumulated_micro_steps = checkpoint
        .optimizer_scopes
        .tiers
        .get(txn.sender)
        .context("native consolidation sender optimizer scope is absent")?
        .accumulated_micro_steps;
    let backend_starts_at_teacher = backend.live_checkpoint().uri == txn.teacher_checkpoint
        && backend.live_checkpoint().sha256 == txn.teacher_hash;
    ensure!(
        backend_starts_at_teacher || checkpoint.sleep.phase == SleepPhase::Commit,
        "tensor backend live checkpoint differs from native teacher before commit"
    );
    if backend_starts_at_teacher {
        checkpoint.validate(&backend.live_checkpoint().model, config)?;
    } else {
        // Same-process retry after immutable candidate publication: the
        // semantic cursor is intentionally still the teacher generation.
        let recovered = backend.snapshot_inflight(&txn)?;
        checkpoint.validate(&recovered.teacher.model, config)?;
    }
    sink.persist(checkpoint)?;

    // Candidate publication is immutable and cannot be rolled back. Keep this
    // flag separate from the semantic cursor: every artifact and optimizer
    // receipt is staged before the single durable cursor publication below.
    // If a later operation fails, a retry completes publication idempotently.
    let mut candidate_was_published = !backend_starts_at_teacher;
    let result = (|| loop {
        match checkpoint.sleep.phase {
            SleepPhase::ProspectiveUpdate => {
                backend.compute_prospective_update(&txn)?;
                backend.stage_student(&txn)?;
                checkpoint.sleep.transition(SleepPhase::KnowledgeSeeding)?;
                persist_tensor_boundary(checkpoint, backend, store, sink)?;
            }
            SleepPhase::KnowledgeSeeding => {
                checkpoint
                    .sleep
                    .reserve_knowledge_rng(0, config.knowledge_seeding.rollout_count_u64()?)?;
                sink.persist(checkpoint)?;
                let current_txn = checkpoint
                    .sleep
                    .pending
                    .clone()
                    .expect("transaction checked above");
                backend.knowledge_seed(&current_txn)?;
                emit_knowledge_metric(checkpoint, &current_txn, backend.diagnostics(), sink)?;
                checkpoint.sleep.transition(SleepPhase::Imitation)?;
                persist_tensor_boundary(checkpoint, backend, store, sink)?;
            }
            SleepPhase::Imitation => {
                checkpoint
                    .sleep
                    .reserve_imitation_rng(0, config.imitation.group_size_u64()?)?;
                sink.persist(checkpoint)?;
                let current_txn = checkpoint
                    .sleep
                    .pending
                    .clone()
                    .expect("transaction checked above");
                backend.learn_to_imitate(&current_txn)?;
                emit_imitation_metric(
                    checkpoint,
                    &current_txn,
                    config,
                    backend.diagnostics(),
                    sink,
                )?;
                checkpoint
                    .sleep
                    .transition(SleepPhase::RetentionValidation)?;
                persist_tensor_boundary(checkpoint, backend, store, sink)?;
            }
            SleepPhase::RetentionValidation => {
                let retention_passes = backend.retention_passes(&txn)?;
                emit_retention_metrics(checkpoint, &txn, config, backend.diagnostics(), sink)?;
                if !retention_passes {
                    break Ok(false);
                }
                checkpoint.sleep.transition(SleepPhase::Commit)?;
                persist_tensor_boundary(checkpoint, backend, store, sink)?;
            }
            SleepPhase::Commit => {
                let candidate = backend.commit(&txn)?;
                candidate_was_published = true;
                let pointer =
                    persist_tensor_boundary_without_cursor(&checkpoint.sleep, backend, store)?;
                let receiver_ids = backend
                    .receiver_parameter_ids()
                    .iter()
                    .map(|id| id.val())
                    .collect::<Vec<_>>();
                let commits = optimizer_publisher.publish(&txn, &pointer, &receiver_ids)?;

                // Do not expose a half-committed in-memory cursor. Build and
                // validate the complete next generation, durably publish it,
                // and only then replace the caller's cursor.
                let mut next = checkpoint.clone();
                next.sleep.record_committed_candidate(
                    candidate.checkpoint.clone(),
                    candidate.sha256.clone(),
                )?;
                next.sleep.commit_consolidation()?;
                apply_optimizer_commits(
                    &mut next,
                    schedule,
                    &txn,
                    &backend.live_checkpoint().model,
                    &receiver_ids,
                    commits,
                )?;
                // Keep the last authenticated pre-commit tensor generation in
                // the semantic transaction. Candidate weights and optimizer
                // receipts are independently content-addressed below; a
                // load→save cycle of a backend optimizer is not guaranteed to
                // have canonical bytes and therefore must not re-key the
                // completed transaction history after interruption.
                next.live_checkpoint =
                    NativeCheckpointRef::new(candidate.checkpoint, candidate.sha256)?;
                next.validate(&backend.live_checkpoint().model, config)?;
                emit_tier_update_metric(
                    &next,
                    schedule,
                    &txn,
                    sender_accumulated_micro_steps,
                    TierUpdateOutcome::Committed,
                    sink,
                )?;
                emit_active_capacity_metrics(
                    &next,
                    &backend.live_checkpoint().model,
                    schedule,
                    sink,
                )?;
                sink.persist(&next)?;
                *checkpoint = next;
                break Ok(true);
            }
            SleepPhase::DreamGeneration
            | SleepPhase::DreamRanking
            | SleepPhase::DreamTrials
            | SleepPhase::DreamPolicyUpdate
            | SleepPhase::Candidate => break Ok(true),
            SleepPhase::Wake => bail!("native consolidation transaction is in wake phase"),
        }
    })();

    match result {
        Ok(true) => Ok(NativeConsolidationOutcome::Accepted(
            checkpoint.live_checkpoint.clone(),
        )),
        Ok(false) => {
            backend.restore_teacher(&txn)?;
            let mut restored = checkpoint.clone();
            restored.sleep.rollback()?;
            restored.live_checkpoint =
                NativeCheckpointRef::new(txn.teacher_checkpoint.clone(), txn.teacher_hash.clone())?;
            restored.validate(&backend.live_checkpoint().model, config)?;
            let metric = emit_tier_update_metric(
                &restored,
                schedule,
                &txn,
                sender_accumulated_micro_steps,
                TierUpdateOutcome::RolledBack,
                sink,
            );
            publish_native_rollback(checkpoint, restored, metric, sink)?;
            Ok(NativeConsolidationOutcome::Rejected(
                checkpoint.live_checkpoint.clone(),
            ))
        }
        Err(error) => {
            if candidate_was_published {
                return Err(error.context(
                    "native candidate was published; retry idempotent artifact and cursor publication without rollback",
                ));
            }
            if let Err(restore_error) = backend.restore_teacher(&txn) {
                bail!(
                    "native consolidation failed: {error:#}; teacher restore also failed: {restore_error:#}"
                );
            }
            let mut restored = checkpoint.clone();
            restored.sleep.rollback()?;
            restored.live_checkpoint =
                NativeCheckpointRef::new(txn.teacher_checkpoint.clone(), txn.teacher_hash.clone())?;
            restored.validate(&backend.live_checkpoint().model, config)?;
            let metric = emit_tier_update_metric(
                &restored,
                schedule,
                &txn,
                sender_accumulated_micro_steps,
                TierUpdateOutcome::RolledBack,
                sink,
            );
            publish_native_rollback(checkpoint, restored, metric, sink)?;
            Err(error)
        }
    }
}

/// Publish restored semantic state even if telemetry emission failed. Once the
/// tensor backend has restored its teacher, leaving the durable or in-memory
/// cursor in an in-flight subphase would make a same-process retry disagree
/// with the backend. A failed checkpoint publication still leaves the caller
/// pending so it can retry from the last durable boundary.
fn publish_native_rollback<S: NativeSleepProgressSink>(
    checkpoint: &mut NativeSleepCheckpoint,
    restored: NativeSleepCheckpoint,
    metric: Result<()>,
    sink: &mut S,
) -> Result<()> {
    let persist = sink.persist(&restored);
    if persist.is_ok() {
        *checkpoint = restored;
    }
    metric.context("emitting native sleep rollback metric")?;
    persist.context("persisting native sleep rollback")
}

fn persist_tensor_boundary_without_cursor<U, R, J, E, P, T>(
    state: &SleepState,
    backend: &TensorConsolidationBackend<U, R, J, E, P>,
    store: &T,
) -> Result<TensorTransactionPointer>
where
    U: ProspectiveTransformerUpdate,
    R: ConsolidationRollouts,
    J: SemanticJudge,
    E: RetentionEvaluator,
    P: AtomicCandidatePublisher,
    T: NativeTensorTransactionStore,
{
    let txn = state
        .pending
        .as_ref()
        .context("committed native boundary has no transaction")?;
    store.publish(txn, &backend.snapshot_inflight(txn)?)
}

fn apply_optimizer_commits(
    checkpoint: &mut NativeSleepCheckpoint,
    schedule: &SleepSchedule,
    txn: &ConsolidationTxn,
    committed_model: &Transformer,
    receiver_parameter_ids: &[u64],
    commits: Vec<TierOptimizerCommit>,
) -> Result<()> {
    let sender_period = schedule
        .tiers
        .get(txn.sender)
        .context("transaction sender is absent from optimizer schedule")?
        .update_period;
    ensure!(
        txn.trigger_clock.is_multiple_of(sender_period),
        "sender optimizer update is off its configured boundary"
    );
    let required = [txn.sender, txn.receiver]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let observed = commits
        .iter()
        .map(|commit| commit.tier)
        .collect::<BTreeSet<_>>();
    ensure!(
        observed.len() == commits.len() && observed == required,
        "optimizer publisher must return exactly one receipt for every touched tier"
    );
    let expected_receiver = if txn.terminal {
        committed_model
            .memory_tier_base_parameter_ids_all_layers(txn.sender)?
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>()
    } else {
        committed_model
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| {
                status.tier == txn.receiver && status.slot == txn.receiver_slot && status.active
            })
            .flat_map(|status| status.parameter_ids.into_iter().map(|id| id.val()))
            .collect::<BTreeSet<_>>()
    };
    let provided_receiver = receiver_parameter_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    ensure!(
        !expected_receiver.is_empty()
            && provided_receiver.len() == receiver_parameter_ids.len()
            && provided_receiver == expected_receiver,
        "receiver optimizer parameter IDs differ from the committed model slot"
    );
    for commit in &commits {
        let expected_role = if txn.terminal {
            TierOptimizerCommitRole::TerminalCombined
        } else if commit.tier == txn.sender {
            TierOptimizerCommitRole::SenderUpdate
        } else {
            TierOptimizerCommitRole::ReceiverTransfer
        };
        ensure!(
            commit.role == expected_role,
            "optimizer tier {} has role {:?}; expected {:?}",
            commit.tier,
            commit.role,
            expected_role
        );
        if commit.role != TierOptimizerCommitRole::ReceiverTransfer {
            ensure!(
                commit.artifact.accumulator_parameter_ids.is_empty(),
                "sender optimizer tier {} retains accumulated gradients",
                commit.tier
            );
        }
        // A committed bundle replaces the durable optimizer state for this
        // tier, so it must preserve base and every already-active slot—not
        // merely contain the newly transferred receiver slot.
        let mut expected = committed_model
            .memory_tier_base_parameter_ids_all_layers(commit.tier)?
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        for status in committed_model
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == commit.tier && status.active)
        {
            expected.extend(status.parameter_ids.into_iter().map(|id| id.val()));
        }
        let observed = commit
            .artifact
            .optimizer_parameter_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        ensure!(
            !observed.is_empty() && observed == expected,
            "optimizer tier {} receipt does not cover its exact active/base parameter scope",
            commit.tier
        );
    }
    let mut next_scopes = checkpoint.optimizer_scopes.clone();
    for commit in commits {
        match commit.role {
            TierOptimizerCommitRole::ReceiverTransfer => next_scopes.commit_receiver_transfer(
                commit.tier,
                txn.trigger_clock,
                commit.artifact,
            )?,
            TierOptimizerCommitRole::SenderUpdate | TierOptimizerCommitRole::TerminalCombined => {
                next_scopes.commit_sender_update(
                    schedule,
                    commit.tier,
                    txn.trigger_clock,
                    commit.artifact,
                )?
            }
        }
    }
    next_scopes.validate_active_state(committed_model, schedule, &checkpoint.sleep)?;
    checkpoint.optimizer_scopes = next_scopes;
    Ok(())
}

struct NativeDreamProgress<'a, S> {
    checkpoint: &'a mut NativeSleepCheckpoint,
    sink: &'a mut S,
}

impl<S: NativeSleepProgressSink> SleepProgressSink for NativeDreamProgress<'_, S> {
    fn persist(&mut self, state: &SleepState) -> Result<()> {
        let mut next = self.checkpoint.clone();
        next.sleep = state.clone();
        self.sink.persist(&next)?;
        *self.checkpoint = next;
        Ok(())
    }

    fn metric(&mut self, event: MetricEvent) -> Result<()> {
        self.sink.metric(self.checkpoint, event)
    }
}

/// Run Dreaming from the already committed immutable candidate, then close the
/// transaction. The dream backend has no corpus/search handle by construction.
pub fn execute_native_dreaming<B, S>(
    checkpoint: &mut NativeSleepCheckpoint,
    model: &Transformer,
    config: &InModelSleepConfig,
    backend: Option<&mut B>,
    sink: &mut S,
) -> Result<NativeCheckpointRef>
where
    B: DreamingBackend,
    S: NativeSleepProgressSink,
{
    checkpoint.validate(model, config)?;
    ensure!(
        checkpoint
            .sleep
            .pending
            .as_ref()
            .is_some_and(|txn| txn.committed),
        "native dreaming requires a committed consolidation"
    );
    match (config.dreaming.as_ref(), backend) {
        (Some(dreaming), Some(backend)) => {
            let mut state = checkpoint.sleep.clone();
            let mut progress = NativeDreamProgress { checkpoint, sink };
            run_dreaming_with_progress(&mut state, dreaming, backend, &mut progress)?;
            checkpoint.sleep = state;
        }
        (None, None) => {
            let mut next = checkpoint.clone();
            next.sleep.transition(SleepPhase::Candidate)?;
            sink.persist(&next)?;
            *checkpoint = next;
        }
        _ => bail!("dreaming configuration and backend must be provided together"),
    }
    let mut finished = checkpoint.clone();
    finished.sleep.finish_candidate()?;
    finished.validate(model, config)?;
    sink.persist(&finished)?;
    *checkpoint = finished;
    Ok(checkpoint.live_checkpoint.clone())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    use burn::module::ParamId;
    use burn::tensor::{Int, Tensor};
    use burn_optim::{AdamWConfig, GradientsParams};
    use hermes_llm::{Device, Transformer, parse_mal};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::sleep::{
        ImitationConfig, KnowledgeSeedingConfig, MemoryTierSchedule, TerminalConsolidation,
        UpdateClock, full_scope_validation_count, reset_full_scope_validation_count,
    };
    use crate::tensor_sleep::RetentionGateConfig;
    use crate::tensor_sleep::{
        ImitationGroup, ImmutableTransformerCheckpoint, ProspectiveTransformerCandidate,
        ProspectiveUpdateSnapshot, RetentionScores, RolloutOwner, TokenRolloutBatch,
        prospective_update_hash,
    };

    fn hash(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn schedule() -> SleepSchedule {
        SleepSchedule {
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
                MemoryTierSchedule {
                    id: "slow".into(),
                    update_period: 3200,
                    reserve_slots: 8,
                },
            ],
        }
    }

    fn model() -> Transformer {
        let config = parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
            memory cms {
                tier fast { ffn: base reserve_experts { capacity: 1 rank: 3 top_k: 1 } }
                tier medium { ffn: base residual_init: zero reserve_experts { capacity: 4 rank: 3 top_k: 1 } }
                tier slow { ffn: base residual_init: zero reserve_experts { capacity: 8 rank: 3 top_k: 1 } }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 dropout: 0.0 position_encoding: none } memory: cms dropout: 0.0 }
            }
            "#,
        )
        .unwrap();
        Transformer::new(&config, &Device::ndarray().autodiff()).unwrap()
    }

    fn retention_file() -> (tempfile::TempDir, String) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("retention.json");
        fs::write(&path, b"sealed-retention-v1").unwrap();
        let digest = format!("sha256:{:x}", Sha256::digest(b"sealed-retention-v1"));
        (directory, digest)
    }

    fn sleep_config(directory: &Path, retention_hash: String) -> InModelSleepConfig {
        InModelSleepConfig {
            schedule: schedule(),
            standalone_trigger_clock: None,
            knowledge_seeding: KnowledgeSeedingConfig {
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
            dreaming: None,
            retention_suite: directory.join("retention.json"),
            retention_suite_sha256: retention_hash.clone(),
            retention: RetentionGateConfig {
                evaluator_hash: hash('e'),
                suite_hash: retention_hash.clone(),
                max_anchor_forward_kl: 1.0,
                max_anchor_regression: 1.0,
                min_incorporation_gain: -1.0,
            },
            receiver_learning_rate: 1e-2,
            receiver_weight_decay: 0.0,
            grpo_clip_epsilon: 0.2,
            grpo_advantage_epsilon: 1e-6,
            grpo_kl_coefficient: 0.01,
            candidate_directory: directory.join("candidates"),
        }
    }

    fn checkpoint(model: &Transformer) -> (tempfile::TempDir, NativeSleepCheckpoint) {
        let (directory, retention_hash) = retention_file();
        let config = sleep_config(directory.path(), retention_hash);
        let checkpoint = NativeSleepCheckpoint::new(
            hash('f'),
            "sleep",
            NativeCheckpointRef::new("teacher.safetensors", hash('a')).unwrap(),
            model,
            &config,
            3,
        )
        .unwrap();
        (directory, checkpoint)
    }

    fn artifact(byte: char) -> TierOptimizerArtifact {
        TierOptimizerArtifact {
            state_uri: format!("optimizer-{byte}.bundle"),
            manifest_hash: hash(byte),
            optimizer_parameter_ids: Vec::new(),
            accumulator_parameter_ids: Vec::new(),
        }
    }

    fn artifact_for(byte: char, ids: &[u64]) -> TierOptimizerArtifact {
        let mut artifact = artifact(byte);
        artifact.optimizer_parameter_ids = ids.to_vec();
        artifact
    }

    fn active_tier_ids(model: &Transformer, tier: usize) -> Vec<u64> {
        let mut ids = model
            .memory_tier_base_parameter_ids_all_layers(tier)
            .unwrap()
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        for status in model
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == tier && status.active)
        {
            ids.extend(status.parameter_ids.into_iter().map(|id| id.val()));
        }
        ids.into_iter().collect()
    }

    fn commit_test_transaction(
        checkpoint: &mut NativeSleepCheckpoint,
        model: &mut Transformer,
        txn: &ConsolidationTxn,
        candidate: char,
    ) -> Vec<u64> {
        checkpoint
            .sleep
            .transition(SleepPhase::KnowledgeSeeding)
            .unwrap();
        checkpoint.sleep.reserve_knowledge_rng(0, 2).unwrap();
        checkpoint.sleep.transition(SleepPhase::Imitation).unwrap();
        checkpoint.sleep.reserve_imitation_rng(0, 2).unwrap();
        checkpoint
            .sleep
            .transition(SleepPhase::RetentionValidation)
            .unwrap();
        checkpoint.sleep.transition(SleepPhase::Commit).unwrap();
        checkpoint
            .sleep
            .record_committed_candidate(
                format!("candidate-{}-{candidate}", txn.id),
                hash(candidate),
            )
            .unwrap();
        let receiver_ids = if txn.terminal {
            model
                .memory_tier_base_parameter_ids_all_layers(txn.sender)
                .unwrap()
        } else {
            model
                .activate_memory_slot_all_layers(txn.receiver, txn.receiver_slot)
                .unwrap()
        };
        for &slot in &txn.sender_slots_to_reset {
            model
                .reset_memory_slot_all_layers(txn.sender, slot, txn.id)
                .unwrap();
        }
        checkpoint.sleep.commit_consolidation().unwrap();
        receiver_ids.into_iter().map(|id| id.val()).collect()
    }

    fn finish_test_transaction(checkpoint: &mut NativeSleepCheckpoint) {
        let pending = checkpoint.sleep.pending.as_ref().unwrap();
        checkpoint.live_checkpoint = NativeCheckpointRef::new(
            pending.candidate_checkpoint.clone().unwrap(),
            pending.candidate_hash.clone().unwrap(),
        )
        .unwrap();
        checkpoint.sleep.transition(SleepPhase::Candidate).unwrap();
        checkpoint.sleep.finish_candidate().unwrap();
    }

    #[derive(Default)]
    struct CountingSink(usize);

    impl NativeSleepProgressSink for CountingSink {
        fn persist(&mut self, _: &NativeSleepCheckpoint) -> Result<()> {
            self.0 += 1;
            Ok(())
        }
    }

    #[derive(Default)]
    struct CapacitySink(Vec<ActiveCapacityMetrics>);

    impl NativeSleepProgressSink for CapacitySink {
        fn persist(&mut self, _: &NativeSleepCheckpoint) -> Result<()> {
            Ok(())
        }

        fn metric(&mut self, _: &NativeSleepCheckpoint, event: MetricEvent) -> Result<()> {
            if let MetricEvent::ActiveCapacity(metric) = event {
                self.0.push(metric);
            }
            Ok(())
        }
    }

    #[test]
    fn active_capacity_metrics_include_every_fixed_reserve_router_lane() {
        let model = model();
        let (_, checkpoint) = checkpoint(&model);
        let schedule = schedule();
        let mut sink = CapacitySink::default();

        emit_active_capacity_metrics(&checkpoint, &model, &schedule, &mut sink).unwrap();

        assert_eq!(sink.0.len(), schedule.tiers.len());
        let memory_stored = sink
            .0
            .iter()
            .map(|metric| metric.stored_parameters)
            .sum::<u64>();
        let memory_routed = sink
            .0
            .iter()
            .map(|metric| metric.routed_active_parameters)
            .sum::<u64>();
        let model_accounting = model.wake_parameter_accounting().unwrap();
        let non_memory = model_accounting
            .stored_parameters
            .checked_sub(memory_stored)
            .unwrap();
        assert_eq!(
            non_memory + memory_routed,
            model_accounting.routed_active_parameters
        );
    }

    #[derive(Default)]
    struct RetentionSink(Vec<RetentionDeltaMetrics>);

    impl NativeSleepProgressSink for RetentionSink {
        fn persist(&mut self, _: &NativeSleepCheckpoint) -> Result<()> {
            Ok(())
        }

        fn metric(&mut self, _: &NativeSleepCheckpoint, event: MetricEvent) -> Result<()> {
            if let MetricEvent::RetentionDelta(metric) = event {
                self.0.push(metric);
            }
            Ok(())
        }
    }

    #[test]
    fn incorporation_metric_uses_the_configured_retention_threshold() {
        let model = model();
        let (directory, mut checkpoint) = checkpoint(&model);
        let mut config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        config.retention.min_incorporation_gain = 0.2;
        checkpoint.advance_clock(&model, &config, 100).unwrap();
        let txn = checkpoint
            .begin_next(PlannedConsolidation {
                student_checkpoint: "student-metric".into(),
                student_sha256: hash('b'),
                prospective_update_sha256: hash('c'),
            })
            .unwrap();
        let diagnostics = crate::tensor_sleep::TensorSleepDiagnostics {
            retention_anchor_kl: 0.0,
            teacher_stable_anchor: 0.8,
            student_stable_anchor: 0.8,
            teacher_incorporation: 0.4,
            student_incorporation: 0.5,
            incorporation_gain: 0.1,
            ..Default::default()
        };
        let mut sink = RetentionSink::default();

        emit_retention_metrics(&checkpoint, &txn, &config, diagnostics, &mut sink).unwrap();

        let incorporation = sink
            .0
            .iter()
            .find(|metric| metric.metric == "incorporation")
            .unwrap();
        assert!(!incorporation.passed);
    }

    #[test]
    fn native_resume_rejects_extra_or_reordered_evaluator_roles() {
        let model = model();
        let (directory, checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());

        let mut extra = checkpoint.clone();
        extra.sleep.evaluator_hashes.push(hash('9'));
        assert!(extra.validate(&model, &config).is_err());

        let mut reordered = checkpoint;
        reordered.sleep.evaluator_hashes.swap(0, 1);
        assert!(reordered.validate(&model, &config).is_err());
    }

    #[derive(Default)]
    struct RecordingRollbackSink {
        persisted: Option<NativeSleepCheckpoint>,
        fail_persist: bool,
    }

    impl NativeSleepProgressSink for RecordingRollbackSink {
        fn persist(&mut self, checkpoint: &NativeSleepCheckpoint) -> Result<()> {
            if self.fail_persist {
                bail!("injected rollback persistence failure");
            }
            self.persisted = Some(checkpoint.clone());
            Ok(())
        }
    }

    #[test]
    fn rollback_cursor_stays_aligned_with_the_durable_backend_when_metrics_fail() {
        let model = model();
        let (_, mut checkpoint) = checkpoint(&model);
        let original = checkpoint.clone();
        let mut restored = checkpoint.clone();
        restored.sleep.clock = 1;
        let expected = restored.clone();
        let mut sink = RecordingRollbackSink::default();

        let error = publish_native_rollback(
            &mut checkpoint,
            restored,
            Err(anyhow::anyhow!("injected metric failure")),
            &mut sink,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("rollback metric"), "{error}");
        assert_ne!(checkpoint, original);
        assert_eq!(checkpoint, expected);
        assert_eq!(sink.persisted, Some(expected));

        let before_failed_persist = checkpoint.clone();
        let mut sink = RecordingRollbackSink {
            fail_persist: true,
            ..Default::default()
        };
        let error = publish_native_rollback(&mut checkpoint, original, Ok(()), &mut sink)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("persisting native sleep rollback"),
            "{error}"
        );
        assert_eq!(checkpoint, before_failed_persist);
    }

    #[derive(Default)]
    struct RollbackBoundaryDriver(Vec<(u64, usize)>);

    impl PeriodicSleepBoundaryDriver for RollbackBoundaryDriver {
        fn drain_due_sender(
            &mut self,
            checkpoint: &mut NativeSleepCheckpoint,
            _: &InModelSleepConfig,
            _: &mut dyn NativeSleepProgressSink,
        ) -> Result<()> {
            let sender = checkpoint.sleep.next_due_sender().unwrap();
            let clock = checkpoint.sleep.due_clocks[0];
            checkpoint.begin_next(PlannedConsolidation {
                student_checkpoint: format!("student-{clock}-{sender}"),
                student_sha256: hash('b'),
                prospective_update_sha256: hash('c'),
            })?;
            self.0.push((clock, sender));
            checkpoint.sleep.rollback()?;
            Ok(())
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum CrashEdge {
        PersistBefore(usize),
        PersistAfter(usize),
        TensorSealAfter(usize),
        CandidatePublishAfter,
        OptimizerPublishAfter,
        KnowledgeWork,
        ImitationWork,
        RetentionWork,
        DreamGeneration,
        DreamReference,
        DreamTrial,
        DreamPolicy,
    }

    #[derive(Default)]
    struct CrashState {
        target: Option<CrashEdge>,
        fired: bool,
        persists: usize,
        tensor_seals: usize,
    }

    #[derive(Clone, Default)]
    struct CrashPlan(Rc<RefCell<CrashState>>);

    impl CrashPlan {
        fn targeting(edge: CrashEdge) -> Self {
            Self(Rc::new(RefCell::new(CrashState {
                target: Some(edge),
                ..CrashState::default()
            })))
        }

        fn hit(&self, edge: CrashEdge) {
            let should_crash = {
                let mut state = self.0.borrow_mut();
                if !state.fired && state.target.as_ref() == Some(&edge) {
                    state.fired = true;
                    true
                } else {
                    false
                }
            };
            if should_crash {
                panic!("injected native sleep crash at {edge:?}");
            }
        }

        fn persist_before(&self) -> usize {
            let ordinal = {
                let mut state = self.0.borrow_mut();
                state.persists += 1;
                state.persists
            };
            self.hit(CrashEdge::PersistBefore(ordinal));
            ordinal
        }

        fn persist_after(&self, ordinal: usize) {
            self.hit(CrashEdge::PersistAfter(ordinal));
        }

        fn tensor_after(&self) {
            let ordinal = {
                let mut state = self.0.borrow_mut();
                state.tensor_seals += 1;
                state.tensor_seals
            };
            self.hit(CrashEdge::TensorSealAfter(ordinal));
        }

        fn counts(&self) -> (usize, usize) {
            let state = self.0.borrow();
            (state.persists, state.tensor_seals)
        }

        fn assert_fired(&self) {
            assert!(
                self.0.borrow().fired,
                "configured crash edge was not reached"
            );
        }
    }

    #[derive(Clone, Default)]
    struct DurableJournal {
        checkpoint: Rc<RefCell<Option<NativeSleepCheckpoint>>>,
        metrics: Rc<RefCell<Vec<MetricEvent>>>,
    }

    struct CrashSink {
        journal: DurableJournal,
        pending_metrics: Vec<MetricEvent>,
        plan: CrashPlan,
    }

    impl CrashSink {
        fn new(journal: DurableJournal, plan: CrashPlan) -> Self {
            Self {
                journal,
                pending_metrics: Vec::new(),
                plan,
            }
        }
    }

    impl NativeSleepProgressSink for CrashSink {
        fn persist(&mut self, checkpoint: &NativeSleepCheckpoint) -> Result<()> {
            let ordinal = self.plan.persist_before();
            *self.journal.checkpoint.borrow_mut() = Some(checkpoint.clone());
            self.journal
                .metrics
                .borrow_mut()
                .append(&mut self.pending_metrics);
            self.plan.persist_after(ordinal);
            Ok(())
        }

        fn metric(&mut self, _: &NativeSleepCheckpoint, event: MetricEvent) -> Result<()> {
            event.validate()?;
            self.pending_metrics.push(event);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct CrashTensorStore {
        inner: TensorTransactionStore,
        plan: CrashPlan,
    }

    impl NativeTensorTransactionStore for CrashTensorStore {
        fn publish(
            &self,
            txn: &ConsolidationTxn,
            recovered: &crate::tensor_sleep::RecoveredTensorTransaction,
        ) -> Result<TensorTransactionPointer> {
            let pointer = self.inner.publish(txn, recovered)?;
            self.plan.tensor_after();
            Ok(pointer)
        }
    }

    fn crash_sender_update(teacher: &Transformer, device: &Device, sender: usize) -> Transformer {
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

    struct CrashUpdate {
        device: Device,
        state: u64,
        fail_restore: bool,
    }

    impl ProspectiveTransformerUpdate for CrashUpdate {
        fn snapshot_state(&mut self, _: &ConsolidationTxn) -> Result<ProspectiveUpdateSnapshot> {
            ProspectiveUpdateSnapshot::new(self.state.to_le_bytes().to_vec())
        }

        fn restore_state(
            &mut self,
            _: &ConsolidationTxn,
            snapshot: &ProspectiveUpdateSnapshot,
        ) -> Result<()> {
            ensure!(
                !self.fail_restore,
                "injected native update-state restore failure"
            );
            self.state = u64::from_le_bytes(
                snapshot
                    .as_bytes()
                    .try_into()
                    .context("invalid crash-test update snapshot")?,
            );
            Ok(())
        }

        fn stage(
            &mut self,
            txn: &ConsolidationTxn,
            teacher: &Transformer,
        ) -> Result<ProspectiveTransformerCandidate> {
            self.state = self.state.saturating_add(1);
            let model = crash_sender_update(teacher, &self.device, txn.sender);
            Ok(ProspectiveTransformerCandidate {
                update_sha256: prospective_update_hash(teacher, &model, txn.sender)?,
                checkpoint: ImmutableTransformerCheckpoint {
                    uri: txn.student_checkpoint.clone(),
                    sha256: txn.student_hash.clone(),
                    model,
                },
            })
        }

        fn clear_reclaimed_optimizer_state(
            &mut self,
            _: &ConsolidationTxn,
            parameter_ids: &[ParamId],
        ) -> Result<()> {
            self.state = self.state.saturating_add(parameter_ids.len() as u64);
            Ok(())
        }
    }

    struct CrashRollouts {
        plan: CrashPlan,
        fail_knowledge: bool,
    }

    impl ConsolidationRollouts for CrashRollouts {
        fn knowledge_rollouts(
            &mut self,
            _: &ConsolidationTxn,
            _: RolloutOwner,
            _: &Transformer,
            count: usize,
        ) -> Result<Vec<TokenRolloutBatch>> {
            ensure!(
                !self.fail_knowledge,
                "injected native knowledge-rollout failure"
            );
            self.plan.hit(CrashEdge::KnowledgeWork);
            Ok((0..count)
                .map(|_| TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4]).unwrap())
                .collect())
        }

        fn imitation_groups(
            &mut self,
            _: &ConsolidationTxn,
            _: &Transformer,
            _: &Transformer,
            group_size: usize,
        ) -> Result<Vec<ImitationGroup>> {
            self.plan.hit(CrashEdge::ImitationWork);
            let mut candidates = vec![vec![3, 4], vec![8, 9], vec![5, 6]];
            candidates.truncate(group_size);
            Ok(vec![ImitationGroup {
                prefix: vec![1, 2],
                teacher_continuation: vec![3, 4],
                candidates,
            }])
        }
    }

    struct CrashJudge;

    impl SemanticJudge for CrashJudge {
        fn artifact_hash(&self) -> &str {
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        }

        fn score(&mut self, _: &[i64], teacher: &[i64], candidate: &[i64]) -> Result<f32> {
            Ok(if teacher == candidate { 1.0 } else { 0.0 })
        }
    }

    struct CrashEvaluator {
        plan: CrashPlan,
        suite_hash: String,
    }

    type CrashBackend = TensorConsolidationBackend<
        CrashUpdate,
        CrashRollouts,
        CrashJudge,
        CrashEvaluator,
        CrashCandidatePublisher,
    >;

    impl RetentionEvaluator for CrashEvaluator {
        fn artifact_hash(&self) -> &str {
            "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        }

        fn suite_hash(&self) -> &str {
            &self.suite_hash
        }

        fn anchor_rollouts(&mut self, _: &ConsolidationTxn) -> Result<Vec<TokenRolloutBatch>> {
            self.plan.hit(CrashEdge::RetentionWork);
            Ok(vec![TokenRolloutBatch::new(1, 4, vec![1, 2, 3, 4])?])
        }

        fn score(&mut self, _: &ConsolidationTxn, _: &Transformer) -> Result<RetentionScores> {
            Ok(RetentionScores {
                stable_anchor: 1.0,
                incorporation: 1.0,
            })
        }
    }

    #[derive(Clone, Default)]
    struct DurableCandidate(Rc<RefCell<Option<ImmutableTransformerCheckpoint>>>);

    struct CrashCandidatePublisher {
        durable: DurableCandidate,
        plan: CrashPlan,
    }

    impl AtomicCandidatePublisher for CrashCandidatePublisher {
        fn publish_candidate(
            &mut self,
            txn: &ConsolidationTxn,
            candidate: &Transformer,
        ) -> Result<ImmutableTransformerCheckpoint> {
            if self.durable.0.borrow().is_none() {
                *self.durable.0.borrow_mut() = Some(ImmutableTransformerCheckpoint {
                    uri: format!("candidate-{}.safetensors", txn.id),
                    sha256: hash('9'),
                    model: candidate.clone(),
                });
            }
            self.plan.hit(CrashEdge::CandidatePublishAfter);
            Ok(self.durable.0.borrow().as_ref().unwrap().clone())
        }

        fn restore_teacher(
            &mut self,
            _: &ConsolidationTxn,
            _: &ImmutableTransformerCheckpoint,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct DurableOptimizer(Rc<RefCell<Option<Vec<TierOptimizerCommit>>>>);

    struct CrashOptimizerPublisher {
        durable: DurableOptimizer,
        candidate: DurableCandidate,
        plan: CrashPlan,
    }

    impl TierOptimizerPublisher for CrashOptimizerPublisher {
        fn publish(
            &mut self,
            txn: &ConsolidationTxn,
            _: &TensorTransactionPointer,
            receiver_parameter_ids: &[u64],
        ) -> Result<Vec<TierOptimizerCommit>> {
            if self.durable.0.borrow().is_none() {
                let candidate = self.candidate.0.borrow();
                let model = &candidate
                    .as_ref()
                    .context("optimizer publish precedes candidate publication")?
                    .model;
                let expected_receiver = model
                    .memory_slot_statuses()
                    .into_iter()
                    .filter(|status| {
                        status.tier == txn.receiver
                            && status.slot == txn.receiver_slot
                            && status.active
                    })
                    .flat_map(|status| status.parameter_ids.into_iter().map(|id| id.val()))
                    .collect::<BTreeSet<_>>();
                ensure!(
                    receiver_parameter_ids
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>()
                        == expected_receiver,
                    "optimizer publisher received wrong receiver scope"
                );
                let commits = [
                    (txn.sender, TierOptimizerCommitRole::SenderUpdate, '4'),
                    (txn.receiver, TierOptimizerCommitRole::ReceiverTransfer, '5'),
                ]
                .into_iter()
                .map(|(tier, role, byte)| TierOptimizerCommit {
                    tier,
                    role,
                    artifact: artifact_for(byte, &active_tier_ids(model, tier)),
                })
                .collect();
                *self.durable.0.borrow_mut() = Some(commits);
            }
            self.plan.hit(CrashEdge::OptimizerPublishAfter);
            Ok(self.durable.0.borrow().as_ref().unwrap().clone())
        }
    }

    struct CrashDreamBackend {
        plan: CrashPlan,
    }

    impl CrashDreamBackend {
        fn dreams() -> Vec<crate::sleep::GeneratedDream> {
            vec![
                crate::sleep::GeneratedDream {
                    id: "a".into(),
                    artifact_hash: hash('1'),
                    gradient: vec![1.0, 0.0],
                    diversity_key: 7,
                },
                crate::sleep::GeneratedDream {
                    id: "b".into(),
                    artifact_hash: hash('2'),
                    gradient: vec![0.5, 0.5],
                    diversity_key: 3,
                },
                crate::sleep::GeneratedDream {
                    id: "c".into(),
                    artifact_hash: hash('3'),
                    gradient: vec![0.0, 1.0],
                    diversity_key: 1,
                },
            ]
        }
    }

    impl DreamingBackend for CrashDreamBackend {
        fn verify_committed_candidate(&mut self, txn: &ConsolidationTxn) -> Result<()> {
            ensure!(
                txn.candidate_hash.as_deref() == Some(hash('9').as_str()),
                "crash backend is bound to another candidate"
            );
            Ok(())
        }

        fn shared_checkpoint_hash(&mut self) -> Result<String> {
            Ok(hash('9'))
        }

        fn generate_from_wake_contexts(
            &mut self,
            _: &ConsolidationTxn,
            candidate_count: usize,
            random_extra_expert: bool,
        ) -> Result<(String, Vec<crate::sleep::GeneratedDream>)> {
            self.plan.hit(CrashEdge::DreamGeneration);
            ensure!(
                candidate_count == 3 && random_extra_expert,
                "wrong dream recipe"
            );
            Ok((hash('7'), Self::dreams()))
        }

        fn load_generated_dreams(
            &mut self,
            _: &ConsolidationTxn,
            manifest: &str,
        ) -> Result<Vec<crate::sleep::GeneratedDream>> {
            ensure!(manifest == hash('7'), "wrong dream manifest");
            Ok(Self::dreams())
        }

        fn reference_gradient(
            &mut self,
            _: &ConsolidationTxn,
            reference_set_hash: &str,
        ) -> Result<Vec<f32>> {
            self.plan.hit(CrashEdge::DreamReference);
            ensure!(reference_set_hash == hash('8'), "wrong reference set");
            Ok(vec![1.0, 0.0])
        }

        fn isolated_lora_trial(
            &mut self,
            _: &ConsolidationTxn,
            candidate: &crate::sleep::GeneratedDream,
            rank: usize,
            alpha: usize,
        ) -> Result<crate::sleep::DreamTrial> {
            self.plan.hit(CrashEdge::DreamTrial);
            ensure!(rank == 64 && alpha == 128, "wrong dream LoRA recipe");
            Ok(crate::sleep::DreamTrial {
                candidate_id: candidate.id.clone(),
                adapter_hash: hash(if candidate.id == "a" { '6' } else { '7' }),
                evaluator_hash: hash('c'),
                independent_task_improvement: if candidate.id == "a" { 0.1 } else { -0.1 },
            })
        }

        fn restem_update(
            &mut self,
            _: &ConsolidationTxn,
            _: &[crate::sleep::DreamTrial],
            iterations: usize,
        ) -> Result<String> {
            self.plan.hit(CrashEdge::DreamPolicy);
            ensure!(iterations == 1, "wrong ReSTEM iteration count");
            Ok(hash('6'))
        }

        fn restore_shared_candidate(&mut self, _: &ConsolidationTxn) -> Result<()> {
            Ok(())
        }
    }

    struct CrashSeed {
        _directory: tempfile::TempDir,
        device: Device,
        model: Transformer,
        config: InModelSleepConfig,
        initial: NativeSleepCheckpoint,
    }

    impl CrashSeed {
        fn new() -> Self {
            let mut model = model();
            model.activate_memory_slot_all_layers(0, 0).unwrap();
            let device = Device::ndarray().autodiff();
            let (directory, retention_hash) = retention_file();
            let mut config = sleep_config(directory.path(), retention_hash);
            config.dreaming = Some(crate::sleep::DreamingConfig {
                candidate_count: 3,
                retain_top: 1,
                retain_random: 1,
                lora_rank: 64,
                lora_alpha: 128,
                restem_iterations: 1,
                selector_version: "gradient-cosine-v1".into(),
                reference_set_hash: hash('8'),
                trial_evaluator_hash: hash('c'),
            });
            config.retention.max_anchor_forward_kl = 100.0;
            let mut initial = NativeSleepCheckpoint::new(
                hash('f'),
                "sleep",
                NativeCheckpointRef::new("teacher.safetensors", hash('a')).unwrap(),
                &model,
                &config,
                3,
            )
            .unwrap();
            initial.advance_clock(&model, &config, 100).unwrap();
            let student = crash_sender_update(&model, &device, 0);
            initial
                .begin_next(PlannedConsolidation {
                    student_checkpoint: "student.safetensors".into(),
                    student_sha256: hash('b'),
                    prospective_update_sha256: prospective_update_hash(&model, &student, 0)
                        .unwrap(),
                })
                .unwrap();
            Self {
                _directory: directory,
                device,
                model,
                config,
                initial,
            }
        }
    }

    struct CrashHarness<'a> {
        seed: &'a CrashSeed,
        _store_directory: tempfile::TempDir,
        store: CrashTensorStore,
        journal: DurableJournal,
        candidate: DurableCandidate,
        optimizer: DurableOptimizer,
        plan: CrashPlan,
        fail_restore: bool,
        fail_knowledge: bool,
    }

    struct CrashOutcome {
        checkpoint: NativeSleepCheckpoint,
        metrics: Vec<MetricEvent>,
        optimizer: Vec<TierOptimizerCommit>,
        candidate_probe_bits: Vec<u32>,
        tensor_manifest: serde_json::Value,
        tensor_metadata: serde_json::Value,
        tensor_metadata_bytes: String,
        counts: (usize, usize),
    }

    impl<'a> CrashHarness<'a> {
        fn new(seed: &'a CrashSeed, plan: CrashPlan) -> Self {
            let store_directory = tempfile::tempdir().unwrap();
            let journal = DurableJournal::default();
            *journal.checkpoint.borrow_mut() = Some(seed.initial.clone());
            Self {
                seed,
                store: CrashTensorStore {
                    inner: TensorTransactionStore::new(store_directory.path()),
                    plan: plan.clone(),
                },
                _store_directory: store_directory,
                journal,
                candidate: DurableCandidate::default(),
                optimizer: DurableOptimizer::default(),
                plan,
                fail_restore: false,
                fail_knowledge: false,
            }
        }

        fn build_backend(&self, cursor: &NativeSleepCheckpoint) -> Result<CrashBackend> {
            let txn = cursor
                .sleep
                .pending
                .as_ref()
                .context("crash harness has no consolidation transaction")?;
            let mut backend = TensorConsolidationBackend::new(
                ImmutableTransformerCheckpoint {
                    uri: txn.teacher_checkpoint.clone(),
                    sha256: txn.teacher_hash.clone(),
                    model: self.seed.model.clone(),
                },
                self.seed.device.clone(),
                self.seed.config.tensor_config(),
                CrashUpdate {
                    device: self.seed.device.clone(),
                    state: 0,
                    fail_restore: self.fail_restore,
                },
                CrashRollouts {
                    plan: self.plan.clone(),
                    fail_knowledge: self.fail_knowledge,
                },
                CrashJudge,
                CrashEvaluator {
                    plan: self.plan.clone(),
                    suite_hash: self.seed.config.retention_suite_sha256.clone(),
                },
                CrashCandidatePublisher {
                    durable: self.candidate.clone(),
                    plan: self.plan.clone(),
                },
            )?;
            if txn.tensor_transaction_generation.is_some() {
                let recovered = self.store.inner.load_recorded(
                    txn,
                    self.seed.model.config(),
                    &self.seed.device,
                    AdamWConfig::new()
                        .with_weight_decay(self.seed.config.receiver_weight_decay)
                        .init(),
                )?;
                backend.restore_inflight(txn, recovered)?;
            }
            Ok(backend)
        }

        fn attempt(&mut self) -> Result<bool> {
            let mut cursor = self
                .journal
                .checkpoint
                .borrow()
                .clone()
                .context("crash journal has no checkpoint")?;
            if cursor.sleep.phase == SleepPhase::Wake && cursor.sleep.pending.is_none() {
                return Ok(true);
            }
            let mut sink = CrashSink::new(self.journal.clone(), self.plan.clone());
            if cursor
                .sleep
                .pending
                .as_ref()
                .is_some_and(|txn| txn.committed)
            {
                let model = self
                    .candidate
                    .0
                    .borrow()
                    .as_ref()
                    .context("committed cursor has no durable candidate")?
                    .model
                    .clone();
                let mut dreaming = CrashDreamBackend {
                    plan: self.plan.clone(),
                };
                execute_native_dreaming(
                    &mut cursor,
                    &model,
                    &self.seed.config,
                    Some(&mut dreaming),
                    &mut sink,
                )?;
            } else {
                let mut backend = self.build_backend(&cursor)?;
                let mut optimizer = CrashOptimizerPublisher {
                    durable: self.optimizer.clone(),
                    candidate: self.candidate.clone(),
                    plan: self.plan.clone(),
                };
                execute_native_consolidation(
                    &mut cursor,
                    &self.seed.config,
                    &mut backend,
                    &self.store,
                    &mut optimizer,
                    &mut sink,
                )?;
            }
            Ok(false)
        }

        fn run(mut self) -> CrashOutcome {
            for _ in 0..8 {
                let attempt = catch_unwind(AssertUnwindSafe(|| self.attempt()));
                match attempt {
                    Ok(Ok(true)) => {
                        let candidate = self.candidate.0.borrow();
                        let model = &candidate.as_ref().unwrap().model;
                        let probe = model
                            .forward(
                                Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &self.seed.device),
                                0,
                            )
                            .into_data()
                            .convert::<f32>()
                            .to_vec::<f32>()
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits)
                            .collect();
                        let checkpoint = self.journal.checkpoint.borrow().clone().unwrap();
                        let generation = &checkpoint.sleep.completed_transactions[0]
                            .tensor_transaction_generation;
                        let generation = self
                            .store
                            .inner
                            .root()
                            .join("generations")
                            .join(generation.as_ref().unwrap());
                        let tensor_manifest = serde_json::from_slice(
                            &fs::read(generation.join("manifest.json")).unwrap(),
                        )
                        .unwrap();
                        let tensor_metadata_bytes =
                            fs::read_to_string(generation.join("transaction.json")).unwrap();
                        let tensor_metadata = serde_json::from_str(&tensor_metadata_bytes).unwrap();
                        return CrashOutcome {
                            checkpoint,
                            metrics: self.journal.metrics.borrow().clone(),
                            optimizer: self.optimizer.0.borrow().clone().unwrap(),
                            candidate_probe_bits: probe,
                            tensor_manifest,
                            tensor_metadata,
                            tensor_metadata_bytes,
                            counts: self.plan.counts(),
                        };
                    }
                    Ok(Ok(false)) | Err(_) => {}
                    Ok(Err(error)) => panic!("crash recovery returned an error: {error:#}"),
                }
            }
            panic!("crash harness did not finish")
        }
    }

    #[test]
    fn failed_native_teacher_restore_never_publishes_rollback_metadata() {
        let seed = CrashSeed::new();
        let mut harness = CrashHarness::new(&seed, CrashPlan::default());
        harness.fail_knowledge = true;
        harness.fail_restore = true;

        let error = harness.attempt().unwrap_err().to_string();
        assert!(error.contains("teacher restore also failed"), "{error}");
        let durable = harness.journal.checkpoint.borrow().clone().unwrap();
        assert_eq!(durable.sleep.phase, SleepPhase::KnowledgeSeeding);
        assert!(durable.sleep.pending.is_some());
        assert!(
            harness
                .journal
                .checkpoint
                .borrow()
                .as_ref()
                .is_some_and(|checkpoint| checkpoint.sleep.phase != SleepPhase::Wake),
            "a native rollback cursor was published despite failed restoration"
        );
    }

    #[test]
    fn retention_bytes_are_reverified_before_resume_or_mutation() {
        let model = model();
        let (directory, checkpoint) = checkpoint(&model);
        fs::write(directory.path().join("retention.json"), b"drifted").unwrap();
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        let error = checkpoint
            .validate(&model, &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("changed"), "{error}");
    }

    #[test]
    fn wake_steps_defer_external_artifact_io_until_a_sleep_boundary() {
        let model = model();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        fs::write(directory.path().join("retention.json"), b"drifted").unwrap();

        checkpoint.advance_clock(&model, &config, 1).unwrap();
        let before_boundary = checkpoint.clone();
        let error = checkpoint
            .advance_clock(&model, &config, 100)
            .unwrap_err()
            .to_string();

        assert!(error.contains("changed"), "{error}");
        assert_eq!(checkpoint, before_boundary);
    }

    #[test]
    fn ordinary_wake_ticks_skip_full_parameter_scope_validation() {
        let model = model();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());

        reset_full_scope_validation_count();
        checkpoint.advance_clock(&model, &config, 1).unwrap();
        assert_eq!(full_scope_validation_count(), 0);

        checkpoint.advance_clock(&model, &config, 100).unwrap();
        assert!(
            full_scope_validation_count() > 0,
            "a due boundary must still revalidate the complete model scope"
        );
    }

    #[test]
    fn optimizer_scope_restore_uses_candidate_topology_after_commit() {
        let mut model = model();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        checkpoint.advance_clock(&model, &config, 100).unwrap();
        let txn = checkpoint
            .begin_next(PlannedConsolidation {
                student_checkpoint: "student.safetensors".into(),
                student_sha256: hash('b'),
                prospective_update_sha256: hash('c'),
            })
            .unwrap();

        assert_eq!(
            checkpoint.optimizer_scope_checkpoint().unwrap(),
            NativeCheckpointRef::new(&txn.teacher_checkpoint, &txn.teacher_hash).unwrap()
        );

        commit_test_transaction(&mut checkpoint, &mut model, &txn, 'd');
        let pending = checkpoint.sleep.pending.as_ref().unwrap();
        checkpoint.live_checkpoint = NativeCheckpointRef::new(
            pending.candidate_checkpoint.as_ref().unwrap(),
            pending.candidate_hash.as_ref().unwrap(),
        )
        .unwrap();

        assert_eq!(
            checkpoint.optimizer_scope_checkpoint().unwrap(),
            checkpoint.live_checkpoint
        );
    }

    #[test]
    fn resume_rejects_optimizer_update_clock_drift_from_sleep_state() {
        let model = model();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        checkpoint.advance_clock(&model, &config, 100).unwrap();
        checkpoint.optimizer_scopes.tiers[0].update_clock = 100;

        let error = checkpoint
            .validate(&model, &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("disagrees with its sleep tier"), "{error}");
    }

    #[test]
    fn periodic_controller_rejects_coarse_clock_and_drains_split_boundaries_in_order() {
        let model = model();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        let mut driver = RollbackBoundaryDriver::default();
        let mut sink = CountingSink::default();

        let before = checkpoint.clone();
        let error = drain_periodic_sleep_before_wake_step(
            &mut checkpoint,
            &model,
            &config,
            400,
            &mut driver,
            &mut sink,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("one tier gradient accumulator"), "{error}");
        assert_eq!(checkpoint, before);
        assert!(driver.0.is_empty());

        for clock in [100, 200, 300, 400] {
            drain_periodic_sleep_before_wake_step(
                &mut checkpoint,
                &model,
                &config,
                clock,
                &mut driver,
                &mut sink,
            )
            .unwrap();
        }

        assert_eq!(
            driver.0,
            vec![(100, 0), (200, 0), (300, 0), (400, 0), (400, 1)]
        );
        assert!(checkpoint.sleep.due_senders.is_empty());
        assert!(checkpoint.sleep.due_clocks.is_empty());
        assert_eq!(checkpoint.sleep.tiers[0].last_boundary_clock, 400);
        assert_eq!(checkpoint.sleep.tiers[1].last_boundary_clock, 400);
        assert!(sink.0 >= 6);
    }

    #[test]
    fn wake_scope_excludes_every_memory_tier_and_dormant_moments_are_rejected() {
        let model = model();
        let schedule = schedule();
        let (_, mut checkpoint) = checkpoint(&model);
        let memory = checkpoint
            .optimizer_scopes
            .tiers
            .iter()
            .flat_map(|scope| scope.parameter_ids.iter())
            .copied()
            .collect::<BTreeSet<_>>();
        assert!(
            checkpoint
                .optimizer_scopes
                .wake_parameter_ids
                .iter()
                .all(|id| !memory.contains(id))
        );

        let dormant_id = model
            .memory_slot_statuses()
            .into_iter()
            .find(|status| status.tier == 1 && status.slot == 0)
            .unwrap()
            .parameter_ids[0]
            .val();
        let mut state = artifact('b');
        state.optimizer_parameter_ids.push(dormant_id);
        checkpoint.optimizer_scopes.tiers[1].generation = 1;
        checkpoint.optimizer_scopes.tiers[1].artifact = Some(state);
        let error = checkpoint
            .optimizer_scopes
            .validate_active_state(&model, &schedule, &checkpoint.sleep)
            .unwrap_err()
            .to_string();
        assert!(error.contains("dormant or foreign"), "{error}");
    }

    #[test]
    fn fast_transfer_and_coincident_sender_use_independent_same_clock_generations() {
        let mut model = model();
        let schedule = schedule();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        for (clock, candidate) in [(100, 'd'), (200, 'e'), (300, 'f'), (400, '7')] {
            checkpoint.advance_clock(&model, &config, clock).unwrap();
            let fast = checkpoint
                .begin_next(PlannedConsolidation {
                    student_checkpoint: format!("fast-student-{clock}"),
                    student_sha256: hash('b'),
                    prospective_update_sha256: hash('c'),
                })
                .unwrap();
            let receiver_ids =
                commit_test_transaction(&mut checkpoint, &mut model, &fast, candidate);
            let fast_ids = active_tier_ids(&model, 0);
            let medium_ids = active_tier_ids(&model, 1);
            apply_optimizer_commits(
                &mut checkpoint,
                &schedule,
                &fast,
                &model,
                &receiver_ids,
                vec![
                    TierOptimizerCommit {
                        tier: 0,
                        role: TierOptimizerCommitRole::SenderUpdate,
                        artifact: artifact_for('d', &fast_ids),
                    },
                    TierOptimizerCommit {
                        tier: 1,
                        role: TierOptimizerCommitRole::ReceiverTransfer,
                        artifact: artifact_for('e', &medium_ids),
                    },
                ],
            )
            .unwrap();
            assert_eq!(checkpoint.optimizer_scopes.tiers[1].update_clock, 0);
            assert_eq!(checkpoint.optimizer_scopes.tiers[1].transfer_clock, clock);
            assert_eq!(
                checkpoint.optimizer_scopes.tiers[1].transfer_generation,
                clock / 100
            );
            finish_test_transaction(&mut checkpoint);
        }

        let medium = checkpoint
            .begin_next(PlannedConsolidation {
                student_checkpoint: "medium-student".into(),
                student_sha256: hash('6'),
                prospective_update_sha256: hash('7'),
            })
            .unwrap();
        let receiver_ids = commit_test_transaction(&mut checkpoint, &mut model, &medium, '9');
        let medium_ids = active_tier_ids(&model, 1);
        let slow_ids = active_tier_ids(&model, 2);
        apply_optimizer_commits(
            &mut checkpoint,
            &schedule,
            &medium,
            &model,
            &receiver_ids,
            vec![
                TierOptimizerCommit {
                    tier: 1,
                    role: TierOptimizerCommitRole::SenderUpdate,
                    artifact: artifact_for('8', &medium_ids),
                },
                TierOptimizerCommit {
                    tier: 2,
                    role: TierOptimizerCommitRole::ReceiverTransfer,
                    artifact: artifact_for('9', &slow_ids),
                },
            ],
        )
        .unwrap();
        assert_eq!(checkpoint.optimizer_scopes.tiers[1].update_clock, 400);
        assert_eq!(checkpoint.optimizer_scopes.tiers[1].transfer_clock, 400);
        assert_eq!(checkpoint.optimizer_scopes.tiers[1].generation, 5);
    }

    #[test]
    fn receiver_transfer_preserves_slower_pending_wake_accumulator() {
        let mut model = model();
        let schedule = schedule();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        let medium_before = active_tier_ids(&model, 1);
        let pending_id = medium_before[0];
        let mut pending = artifact_for('a', &medium_before);
        pending.accumulator_parameter_ids = vec![pending_id];
        checkpoint
            .optimizer_scopes
            .record_accumulation(1, 7, pending)
            .unwrap();

        checkpoint.advance_clock(&model, &config, 100).unwrap();
        let txn = checkpoint
            .begin_next(PlannedConsolidation {
                student_checkpoint: "student".into(),
                student_sha256: hash('b'),
                prospective_update_sha256: hash('c'),
            })
            .unwrap();
        let receiver_ids = commit_test_transaction(&mut checkpoint, &mut model, &txn, 'd');
        let fast_ids = active_tier_ids(&model, 0);
        let medium_after = active_tier_ids(&model, 1);
        let mut receiver = artifact_for('e', &medium_after);
        receiver.accumulator_parameter_ids = vec![pending_id];
        apply_optimizer_commits(
            &mut checkpoint,
            &schedule,
            &txn,
            &model,
            &receiver_ids,
            vec![
                TierOptimizerCommit {
                    tier: 0,
                    role: TierOptimizerCommitRole::SenderUpdate,
                    artifact: artifact_for('d', &fast_ids),
                },
                TierOptimizerCommit {
                    tier: 1,
                    role: TierOptimizerCommitRole::ReceiverTransfer,
                    artifact: receiver,
                },
            ],
        )
        .unwrap();

        let medium = &checkpoint.optimizer_scopes.tiers[1];
        assert_eq!(medium.update_clock, 0);
        assert_eq!(medium.transfer_clock, 100);
        assert_eq!(medium.accumulated_micro_steps, 7);
        assert_eq!(
            medium.artifact.as_ref().unwrap().accumulator_parameter_ids,
            vec![pending_id]
        );
    }

    #[test]
    fn optimizer_receipts_cannot_swap_sender_and_receiver_roles() {
        let mut model = model();
        let schedule = schedule();
        let (directory, mut checkpoint) = checkpoint(&model);
        let config = sleep_config(directory.path(), checkpoint.retention_suite.sha256.clone());
        checkpoint.advance_clock(&model, &config, 100).unwrap();
        let txn = checkpoint
            .begin_next(PlannedConsolidation {
                student_checkpoint: "student".into(),
                student_sha256: hash('b'),
                prospective_update_sha256: hash('c'),
            })
            .unwrap();
        let receiver_ids = commit_test_transaction(&mut checkpoint, &mut model, &txn, 'd');
        let fast_ids = active_tier_ids(&model, 0);
        let medium_ids = active_tier_ids(&model, 1);
        let error = apply_optimizer_commits(
            &mut checkpoint,
            &schedule,
            &txn,
            &model,
            &receiver_ids,
            vec![
                TierOptimizerCommit {
                    tier: 0,
                    role: TierOptimizerCommitRole::ReceiverTransfer,
                    artifact: artifact_for('d', &fast_ids),
                },
                TierOptimizerCommit {
                    tier: 1,
                    role: TierOptimizerCommitRole::SenderUpdate,
                    artifact: artifact_for('e', &medium_ids),
                },
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("expected"), "{error}");
    }

    #[test]
    fn every_native_sleep_crash_edge_recovers_exactly() {
        let seed = CrashSeed::new();
        let baseline = CrashHarness::new(&seed, CrashPlan::default()).run();
        assert_eq!(baseline.checkpoint.sleep.phase, SleepPhase::Wake);
        assert!(baseline.checkpoint.sleep.pending.is_none());
        assert!(!baseline.metrics.is_empty());
        let mut edges = vec![
            CrashEdge::CandidatePublishAfter,
            CrashEdge::OptimizerPublishAfter,
            CrashEdge::KnowledgeWork,
            CrashEdge::ImitationWork,
            CrashEdge::RetentionWork,
            CrashEdge::DreamGeneration,
            CrashEdge::DreamReference,
            CrashEdge::DreamTrial,
            CrashEdge::DreamPolicy,
        ];
        for ordinal in 1..=baseline.counts.0 {
            edges.push(CrashEdge::PersistBefore(ordinal));
            edges.push(CrashEdge::PersistAfter(ordinal));
        }
        for ordinal in 1..=baseline.counts.1 {
            edges.push(CrashEdge::TensorSealAfter(ordinal));
        }

        for edge in edges {
            let plan = CrashPlan::targeting(edge.clone());
            let recovered = CrashHarness::new(&seed, plan.clone()).run();
            plan.assert_fired();
            assert_eq!(
                recovered.tensor_metadata_bytes, baseline.tensor_metadata_bytes,
                "tensor metadata bytes differ after {edge:?}"
            );
            assert_eq!(
                recovered.tensor_metadata, baseline.tensor_metadata,
                "tensor metadata differs after {edge:?}"
            );
            assert_eq!(
                recovered.tensor_manifest, baseline.tensor_manifest,
                "tensor manifest differs after {edge:?}"
            );
            assert_eq!(
                serde_json::to_value(&recovered.checkpoint).unwrap(),
                serde_json::to_value(&baseline.checkpoint).unwrap(),
                "checkpoint differs after {edge:?}"
            );
            assert_eq!(
                recovered.optimizer, baseline.optimizer,
                "optimizer receipts differ after {edge:?}"
            );
            assert_eq!(
                recovered.candidate_probe_bits, baseline.candidate_probe_bits,
                "candidate weights differ after {edge:?}"
            );
            assert_eq!(
                recovered.metrics, baseline.metrics,
                "committed metric stream differs after {edge:?}"
            );
        }
    }
}
