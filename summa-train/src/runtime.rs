//! Transactional, algorithm-neutral execution for [`WorkflowV2`](crate::workflow::WorkflowV2).
//!
//! The runtime owns only ordering and durable phase transitions.  A
//! [`PhaseExecutor`] owns the algorithm-specific work (causal LM, preference
//! optimization, sleep, evaluation, and so on), while a
//! [`RuntimeCheckpoint`] atomically persists the opaque executor cursor and
//! the runtime state.  Executors must build model changes in private storage
//! and return a content-addressed immutable checkpoint; the input checkpoint
//! is never adopted or overwritten in place.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::artifact_io::{sha256_identity, validate_sha256_identity};
use crate::metrics::{MetricContext, MetricEvent, MetricPhaseKind};
use crate::workflow::{PhaseClass, PhaseKind, PhaseV2, ResolvedWorkflow};

/// Version of the serialized workflow-runtime state.
pub const WORKFLOW_RUNTIME_STATE_VERSION: u32 = 2;

/// Every phase kind understood by WorkflowV2.  Registries are intentionally
/// keyed by phase kind rather than task kind so the trainer core does not need
/// task-specific branches.
pub const ALL_PHASE_KINDS: [PhaseKind; 10] = [
    PhaseKind::Pretrain,
    PhaseKind::ContinuedPretrain,
    PhaseKind::Sft,
    PhaseKind::Preference,
    PhaseKind::Rl,
    PhaseKind::Distillation,
    PhaseKind::Sleep,
    PhaseKind::Quantization,
    PhaseKind::Evaluation,
    PhaseKind::Promotion,
];

/// Content-addressed model checkpoint.  `uri` is an opaque location, so local
/// paths and object-store URIs use the same orchestration contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImmutableModelCheckpoint {
    uri: String,
    sha256: String,
}

impl ImmutableModelCheckpoint {
    pub fn new(uri: impl Into<String>, sha256: impl Into<String>) -> Result<Self> {
        let checkpoint = Self {
            uri: uri.into(),
            sha256: sha256.into(),
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn same_content(&self, other: &Self) -> bool {
        self.sha256 == other.sha256
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(!self.uri.trim().is_empty(), "checkpoint URI is empty");
        validate_sha256_identity(&self.sha256, "checkpoint digest")
    }
}

/// Content-addressed non-model output, such as an evaluation report,
/// rejection report, or promotion receipt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImmutableArtifact {
    uri: String,
    sha256: String,
}

impl ImmutableArtifact {
    pub fn new(uri: impl Into<String>, sha256: impl Into<String>) -> Result<Self> {
        let artifact = Self {
            uri: uri.into(),
            sha256: sha256.into(),
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(!self.uri.trim().is_empty(), "artifact URI is empty");
        validate_sha256_identity(&self.sha256, "artifact digest")
    }
}

/// A completed executor result.  The variants encode the legal side effects
/// for each [`PhaseClass`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PhaseProduct {
    /// Optimization and model-mutation phases publish a new checkpoint rather
    /// than changing their input checkpoint in place.
    ModelCandidate {
        checkpoint: ImmutableModelCheckpoint,
    },
    /// A rejected sleep/quantization candidate leaves the shared checkpoint
    /// untouched, with an immutable report explaining the decision.
    MutationRejected { report: ImmutableArtifact },
    /// Evaluation produces a report and cannot change the model candidate.
    Assessment { report: ImmutableArtifact },
    /// Promotion publishes the exact current candidate and records an
    /// immutable release receipt.  It cannot substitute another checkpoint.
    Release {
        candidate: ImmutableModelCheckpoint,
        receipt: ImmutableArtifact,
    },
}

/// Durable receipt for one committed phase.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseReceipt {
    phase_index: usize,
    phase_name: String,
    phase_kind: PhaseKind,
    input_checkpoint: Option<ImmutableModelCheckpoint>,
    product: PhaseProduct,
    resulting_checkpoint: Option<ImmutableModelCheckpoint>,
}

impl PhaseReceipt {
    pub fn phase_index(&self) -> usize {
        self.phase_index
    }

    pub fn phase_name(&self) -> &str {
        &self.phase_name
    }

    pub fn phase_kind(&self) -> PhaseKind {
        self.phase_kind
    }

    pub fn input_checkpoint(&self) -> Option<&ImmutableModelCheckpoint> {
        self.input_checkpoint.as_ref()
    }

    pub fn product(&self) -> &PhaseProduct {
        &self.product
    }

    pub fn resulting_checkpoint(&self) -> Option<&ImmutableModelCheckpoint> {
        self.resulting_checkpoint.as_ref()
    }
}

/// The currently executing phase.  `resume_state` is executor-owned JSON and
/// is persisted without interpretation by the runtime.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivePhaseState {
    phase_index: usize,
    phase_name: String,
    phase_kind: PhaseKind,
    input_checkpoint: Option<ImmutableModelCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resume_state: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepared_product: Option<PhaseProduct>,
}

impl ActivePhaseState {
    pub fn phase_index(&self) -> usize {
        self.phase_index
    }

    pub fn phase_name(&self) -> &str {
        &self.phase_name
    }

    pub fn phase_kind(&self) -> PhaseKind {
        self.phase_kind
    }

    pub fn input_checkpoint(&self) -> Option<&ImmutableModelCheckpoint> {
        self.input_checkpoint.as_ref()
    }

    pub fn resume_state(&self) -> Option<&Value> {
        self.resume_state.as_ref()
    }

    /// A prepared phase has already published and durably recorded its
    /// immutable output.  Resuming commits it without re-running the executor.
    pub fn is_prepared(&self) -> bool {
        self.prepared_product.is_some()
    }
}

/// Strictly serializable orchestration cursor.  Private fields prevent normal
/// callers from skipping phases; deserialized state is still fully validated
/// against the resolved workflow before any executor runs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowRunState {
    version: u32,
    workflow_signature: String,
    initial_checkpoint: Option<ImmutableModelCheckpoint>,
    current_checkpoint: Option<ImmutableModelCheckpoint>,
    next_phase: usize,
    completed: Vec<PhaseReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active: Option<ActivePhaseState>,
}

impl WorkflowRunState {
    pub fn new(
        workflow: &ResolvedWorkflow,
        initial_checkpoint: Option<ImmutableModelCheckpoint>,
    ) -> Result<Self> {
        workflow.validate()?;
        if let Some(checkpoint) = &initial_checkpoint {
            checkpoint.validate()?;
        }
        let state = Self {
            version: WORKFLOW_RUNTIME_STATE_VERSION,
            workflow_signature: workflow_signature_after_validation(workflow)?,
            current_checkpoint: initial_checkpoint.clone(),
            initial_checkpoint,
            next_phase: 0,
            completed: Vec::new(),
            active: None,
        };
        state.validate(workflow)?;
        Ok(state)
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn workflow_signature(&self) -> &str {
        &self.workflow_signature
    }

    pub fn next_phase_index(&self) -> usize {
        self.next_phase
    }

    pub fn initial_checkpoint(&self) -> Option<&ImmutableModelCheckpoint> {
        self.initial_checkpoint.as_ref()
    }

    pub fn current_checkpoint(&self) -> Option<&ImmutableModelCheckpoint> {
        self.current_checkpoint.as_ref()
    }

    pub fn active_phase(&self) -> Option<&ActivePhaseState> {
        self.active.as_ref()
    }

    pub fn completed_phases(&self) -> &[PhaseReceipt] {
        &self.completed
    }

    pub fn is_complete(&self, workflow: &ResolvedWorkflow) -> bool {
        self.next_phase == workflow.phases.len() && self.active.is_none()
    }

    /// Validate the complete transition history, not just the current index.
    pub fn validate(&self, workflow: &ResolvedWorkflow) -> Result<()> {
        workflow.validate()?;
        ensure!(
            self.version == WORKFLOW_RUNTIME_STATE_VERSION,
            "unsupported workflow-runtime state version {}; this build supports version {WORKFLOW_RUNTIME_STATE_VERSION}",
            self.version
        );
        ensure!(
            self.workflow_signature == workflow_signature_after_validation(workflow)?,
            "workflow-runtime checkpoint does not belong to this resolved workflow"
        );
        ensure!(
            self.next_phase <= workflow.phases.len(),
            "workflow-runtime next phase {} exceeds workflow length {}",
            self.next_phase,
            workflow.phases.len()
        );
        ensure!(
            self.completed.len() == self.next_phase,
            "workflow-runtime has {} receipts but next phase is {}",
            self.completed.len(),
            self.next_phase
        );
        if let Some(checkpoint) = &self.initial_checkpoint {
            checkpoint.validate()?;
        }

        let mut expected_checkpoint = self.initial_checkpoint.clone();
        for (index, receipt) in self.completed.iter().enumerate() {
            let phase = &workflow.phases[index];
            ensure!(
                receipt.phase_index == index
                    && receipt.phase_name == phase.name
                    && receipt.phase_kind == phase.kind,
                "workflow-runtime receipt {index} does not match ordered phase `{}` ({})",
                phase.name,
                phase.kind.name()
            );
            ensure!(
                receipt.input_checkpoint == expected_checkpoint,
                "workflow-runtime receipt {index} has the wrong input checkpoint"
            );
            let resulting = validate_product(phase, expected_checkpoint.as_ref(), &receipt.product)
                .with_context(|| format!("invalid product for committed phase `{}`", phase.name))?;
            ensure!(
                receipt.resulting_checkpoint == resulting,
                "workflow-runtime receipt {index} records the wrong resulting checkpoint"
            );
            expected_checkpoint = resulting;
        }

        ensure!(
            self.current_checkpoint == expected_checkpoint,
            "workflow-runtime current checkpoint disagrees with its phase receipts"
        );
        if self.next_phase < workflow.phases.len() {
            validate_phase_input(
                &workflow.phases[self.next_phase],
                expected_checkpoint.as_ref(),
            )?;
        }

        match &self.active {
            Some(active) => {
                ensure!(
                    self.next_phase < workflow.phases.len(),
                    "completed workflow cannot have an active phase"
                );
                let phase = &workflow.phases[self.next_phase];
                ensure!(
                    active.phase_index == self.next_phase
                        && active.phase_name == phase.name
                        && active.phase_kind == phase.kind,
                    "workflow-runtime active phase does not match ordered phase `{}` ({})",
                    phase.name,
                    phase.kind.name()
                );
                ensure!(
                    active.input_checkpoint == expected_checkpoint,
                    "workflow-runtime active phase has the wrong input checkpoint"
                );
                if let Some(product) = &active.prepared_product {
                    validate_product(phase, expected_checkpoint.as_ref(), product).with_context(
                        || format!("invalid prepared product for `{}`", phase.name),
                    )?;
                }
            }
            None => ensure!(
                self.next_phase == self.completed.len(),
                "workflow-runtime has an inconsistent idle phase cursor"
            ),
        }
        Ok(())
    }
}

/// Stable signature used to reject resume attempts with another resolved
/// workflow, including changed phase order or parameters.
pub fn workflow_signature(workflow: &ResolvedWorkflow) -> Result<String> {
    workflow.validate()?;
    workflow_signature_after_validation(workflow)
}

fn workflow_signature_after_validation(workflow: &ResolvedWorkflow) -> Result<String> {
    let encoded = serde_json::to_vec(workflow)?;
    Ok(sha256_identity(&encoded))
}

fn validate_product(
    phase: &PhaseV2,
    input: Option<&ImmutableModelCheckpoint>,
    product: &PhaseProduct,
) -> Result<Option<ImmutableModelCheckpoint>> {
    if let Some(input) = input {
        input.validate()?;
    }
    match (phase.kind.class(), product) {
        (PhaseClass::Optimization, PhaseProduct::ModelCandidate { checkpoint }) => {
            checkpoint.validate()?;
            if let Some(input) = input {
                ensure!(
                    checkpoint.uri != input.uri,
                    "phase `{}` candidate aliases its immutable input URI",
                    phase.name
                );
                ensure!(
                    !checkpoint.same_content(input),
                    "phase `{}` did not produce a new model candidate",
                    phase.name
                );
            }
            Ok(Some(checkpoint.clone()))
        }
        (PhaseClass::ModelMutation, PhaseProduct::ModelCandidate { checkpoint }) => {
            let input = input.with_context(|| {
                format!(
                    "model-mutation phase `{}` requires an input checkpoint",
                    phase.name
                )
            })?;
            checkpoint.validate()?;
            ensure!(
                checkpoint.uri != input.uri,
                "phase `{}` candidate aliases its immutable input URI",
                phase.name
            );
            ensure!(
                !checkpoint.same_content(input),
                "phase `{}` did not produce a new model candidate",
                phase.name
            );
            Ok(Some(checkpoint.clone()))
        }
        (PhaseClass::ModelMutation, PhaseProduct::MutationRejected { report }) => {
            ensure!(
                input.is_some(),
                "model-mutation phase `{}` requires an input checkpoint",
                phase.name
            );
            report.validate()?;
            Ok(input.cloned())
        }
        (PhaseClass::Assessment, PhaseProduct::Assessment { report }) => {
            ensure!(
                input.is_some(),
                "assessment phase `{}` requires an input checkpoint",
                phase.name
            );
            report.validate()?;
            Ok(input.cloned())
        }
        (PhaseClass::Release, PhaseProduct::Release { candidate, receipt }) => {
            let input = input.with_context(|| {
                format!(
                    "release phase `{}` requires an input checkpoint",
                    phase.name
                )
            })?;
            candidate.validate()?;
            receipt.validate()?;
            ensure!(
                candidate == input,
                "release phase `{}` must publish the exact current immutable candidate",
                phase.name
            );
            Ok(Some(input.clone()))
        }
        (class, _) => bail!(
            "phase `{}` ({}) returned a product incompatible with its {:?} executor class",
            phase.name,
            phase.kind.name(),
            class
        ),
    }
}

fn validate_phase_input(phase: &PhaseV2, input: Option<&ImmutableModelCheckpoint>) -> Result<()> {
    if phase.kind != PhaseKind::Pretrain {
        ensure!(
            input.is_some(),
            "workflow phase `{}` ({}) requires an input model checkpoint",
            phase.name,
            phase.kind.name()
        );
    }
    Ok(())
}

/// Owned request passed to an executor.  Cloning a phase boundary avoids
/// aliasing the mutable runtime state while the executor emits checkpoints.
#[derive(Clone, Debug)]
pub struct PhaseExecutionRequest {
    pub phase_index: usize,
    pub phase: PhaseV2,
    pub input_checkpoint: Option<ImmutableModelCheckpoint>,
    pub resume_state: Option<Value>,
}

/// Executor result.  Yielding is a normal interruption point and persists the
/// supplied cursor before control returns to the caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhaseExecutionResult {
    Yielded { resume_state: Value },
    Complete(PhaseProduct),
}

/// Allows a long-running executor to durably publish intermediate opaque
/// state.  The callback is transactional: runtime memory advances only after
/// the external checkpoint callback succeeds.
pub trait PhaseProgressSink {
    fn checkpoint(&mut self, resume_state: Value) -> Result<()>;

    /// Emit one typed metric associated with the phase currently being
    /// executed. Persistence implementations decide whether metrics are
    /// enabled and couple their committed prefix to runtime checkpoints.
    fn metric(&mut self, context: MetricContext, event: MetricEvent) -> Result<()>;
}

/// Algorithm/backend adapter for one or more phase kinds.
pub trait PhaseExecutor<C> {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        context: &mut C,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult>;
}

impl<C, F> PhaseExecutor<C> for F
where
    F: FnMut(
        &PhaseExecutionRequest,
        &mut C,
        &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult>,
{
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        context: &mut C,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        self(request, context, progress)
    }
}

/// Registry-backed dispatch keeps task and algorithm choices out of the
/// orchestration core.  Registering a kind twice is rejected explicitly.
pub struct ExecutorRegistry<C> {
    executors: Vec<(PhaseKind, Box<dyn PhaseExecutor<C>>)>,
}

impl<C> Default for ExecutorRegistry<C> {
    fn default() -> Self {
        Self {
            executors: Vec::new(),
        }
    }
}

impl<C> ExecutorRegistry<C> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<E>(&mut self, kind: PhaseKind, executor: E) -> Result<()>
    where
        E: PhaseExecutor<C> + 'static,
    {
        ensure!(
            !self.contains(kind),
            "an executor is already registered for phase kind `{}`",
            kind.name()
        );
        self.executors.push((kind, Box::new(executor)));
        Ok(())
    }

    pub fn contains(&self, kind: PhaseKind) -> bool {
        self.executors
            .iter()
            .any(|(registered, _)| *registered == kind)
    }

    fn get_mut(&mut self, kind: PhaseKind) -> Option<&mut (dyn PhaseExecutor<C> + '_)> {
        for (registered, executor) in &mut self.executors {
            if *registered == kind {
                return Some(executor.as_mut());
            }
        }
        None
    }
}

/// Durable transition at which the caller must atomically checkpoint the
/// supplied [`WorkflowRunState`] and any backend state referenced by it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeBoundary {
    PhaseStarted {
        phase_index: usize,
        phase_name: String,
        phase_kind: PhaseKind,
    },
    PhaseProgress {
        phase_index: usize,
        phase_name: String,
    },
    PhaseYielded {
        phase_index: usize,
        phase_name: String,
    },
    PhasePrepared {
        phase_index: usize,
        phase_name: String,
    },
    PhaseCommitted {
        phase_index: usize,
        phase_name: String,
        workflow_complete: bool,
    },
}

/// Persistence hook invoked before each in-memory state transition is
/// accepted.  Implementations should use atomic replace/publication semantics.
pub trait RuntimeCheckpoint {
    fn persist(&mut self, boundary: &RuntimeBoundary, state: &WorkflowRunState) -> Result<()>;

    /// Append an uncommitted metric. The next successful [`Self::persist`]
    /// call makes the current metric prefix part of the same durable runtime
    /// boundary.
    fn append_metric(&mut self, _context: MetricContext, _event: MetricEvent) -> Result<()> {
        bail!("this runtime checkpoint has no metric journal configured")
    }
}

impl<F> RuntimeCheckpoint for F
where
    F: FnMut(&RuntimeBoundary, &WorkflowRunState) -> Result<()>,
{
    fn persist(&mut self, boundary: &RuntimeBoundary, state: &WorkflowRunState) -> Result<()> {
        self(boundary, state)
    }
}

/// Result of one runtime drive operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeStatus {
    AlreadyComplete,
    Yielded {
        phase_index: usize,
        phase_name: String,
    },
    PhaseCommitted {
        phase_index: usize,
        phase_name: String,
        workflow_complete: bool,
    },
}

fn persist_transition<S: RuntimeCheckpoint + ?Sized>(
    workflow: &ResolvedWorkflow,
    state: &mut WorkflowRunState,
    next: WorkflowRunState,
    boundary: RuntimeBoundary,
    checkpoint: &mut S,
) -> Result<()> {
    next.validate(workflow)?;
    checkpoint.persist(&boundary, &next)?;
    *state = next;
    Ok(())
}

struct ProgressCheckpoint<'a, S: RuntimeCheckpoint + ?Sized> {
    workflow: &'a ResolvedWorkflow,
    state: &'a mut WorkflowRunState,
    checkpoint: &'a mut S,
}

impl<S: RuntimeCheckpoint + ?Sized> PhaseProgressSink for ProgressCheckpoint<'_, S> {
    fn checkpoint(&mut self, resume_state: Value) -> Result<()> {
        let mut next = self.state.clone();
        let active = next
            .active
            .as_mut()
            .context("cannot checkpoint executor progress without an active phase")?;
        ensure!(
            active.prepared_product.is_none(),
            "cannot checkpoint executor progress after a product is prepared"
        );
        active.resume_state = Some(resume_state);
        let boundary = RuntimeBoundary::PhaseProgress {
            phase_index: active.phase_index,
            phase_name: active.phase_name.clone(),
        };
        persist_transition(self.workflow, self.state, next, boundary, self.checkpoint)
    }

    fn metric(&mut self, context: MetricContext, event: MetricEvent) -> Result<()> {
        let active = self
            .state
            .active
            .as_ref()
            .context("cannot emit a metric without an active phase")?;
        let expected_index = u32::try_from(active.phase_index)
            .context("active workflow phase index exceeds the metric schema")?;
        ensure!(
            context.phase.index == expected_index && context.phase.name == active.phase_name,
            "executor metric phase identity does not match active workflow phase `{}`",
            active.phase_name
        );
        let expected_kind = MetricPhaseKind::from(active.phase_kind);
        let periodic_sleep_subphase = context.phase.kind == MetricPhaseKind::Sleep
            && self.workflow.phases[active.phase_index]
                .periodic_sleep
                .is_some();
        ensure!(
            context.phase.kind == expected_kind || periodic_sleep_subphase,
            "executor metric kind does not match active workflow phase `{}` ({})",
            active.phase_name,
            active.phase_kind.name()
        );
        if let Some(checkpoint_hash) = &context.checkpoint_hash {
            validate_sha256_identity(checkpoint_hash, "metric checkpoint digest")?;
        }
        event.validate()?;
        self.checkpoint.append_metric(context, event)
    }
}

fn commit_prepared<S: RuntimeCheckpoint + ?Sized>(
    workflow: &ResolvedWorkflow,
    state: &mut WorkflowRunState,
    checkpoint: &mut S,
) -> Result<RuntimeStatus> {
    let active = state
        .active
        .clone()
        .context("cannot commit without an active phase")?;
    let product = active
        .prepared_product
        .clone()
        .context("cannot commit an unprepared phase")?;
    let phase = &workflow.phases[active.phase_index];
    let resulting_checkpoint = validate_product(phase, active.input_checkpoint.as_ref(), &product)?;

    let mut next = state.clone();
    next.completed.push(PhaseReceipt {
        phase_index: active.phase_index,
        phase_name: active.phase_name.clone(),
        phase_kind: active.phase_kind,
        input_checkpoint: active.input_checkpoint,
        product,
        resulting_checkpoint: resulting_checkpoint.clone(),
    });
    next.current_checkpoint = resulting_checkpoint;
    next.next_phase = active.phase_index + 1;
    next.active = None;
    let workflow_complete = next.next_phase == workflow.phases.len();
    let boundary = RuntimeBoundary::PhaseCommitted {
        phase_index: active.phase_index,
        phase_name: active.phase_name.clone(),
        workflow_complete,
    };
    persist_transition(workflow, state, next, boundary, checkpoint)?;
    Ok(RuntimeStatus::PhaseCommitted {
        phase_index: active.phase_index,
        phase_name: active.phase_name,
        workflow_complete,
    })
}

/// Drive exactly one ordered phase until it either yields or commits.
///
/// If the previous process stopped after `PhasePrepared`, this function commits
/// the already-persisted product without invoking an executor again.  This is
/// the critical interruption window for immutable sleep/quantization
/// candidates and release publication.
///
/// After any error, discard the driver and reload it from its
/// [`RuntimeCheckpoint`]. An executor may have successfully persisted progress
/// before a later operation failed, while metrics emitted after that boundary
/// must be truncated by the persistence implementation during resume.
pub fn run_next_phase<C, S: RuntimeCheckpoint + ?Sized>(
    workflow: &ResolvedWorkflow,
    state: &mut WorkflowRunState,
    registry: &mut ExecutorRegistry<C>,
    context: &mut C,
    checkpoint: &mut S,
) -> Result<RuntimeStatus> {
    state.validate(workflow)?;
    if state.is_complete(workflow) {
        return Ok(RuntimeStatus::AlreadyComplete);
    }
    if state
        .active
        .as_ref()
        .is_some_and(ActivePhaseState::is_prepared)
    {
        return commit_prepared(workflow, state, checkpoint);
    }

    let phase_index = state.next_phase;
    let phase = &workflow.phases[phase_index];
    ensure!(
        registry.contains(phase.kind),
        "unsupported executor for workflow phase `{}` ({}); register a PhaseExecutor for `{}`",
        phase.name,
        phase.kind.name(),
        phase.kind.name()
    );

    if state.active.is_none() {
        let mut next = state.clone();
        next.active = Some(ActivePhaseState {
            phase_index,
            phase_name: phase.name.clone(),
            phase_kind: phase.kind,
            input_checkpoint: state.current_checkpoint.clone(),
            resume_state: None,
            prepared_product: None,
        });
        let boundary = RuntimeBoundary::PhaseStarted {
            phase_index,
            phase_name: phase.name.clone(),
            phase_kind: phase.kind,
        };
        persist_transition(workflow, state, next, boundary, checkpoint)?;
    }

    let active = state
        .active
        .as_ref()
        .context("phase start did not create active runtime state")?;
    let request = PhaseExecutionRequest {
        phase_index,
        phase: phase.clone(),
        input_checkpoint: active.input_checkpoint.clone(),
        resume_state: active.resume_state.clone(),
    };
    let executor = registry
        .get_mut(phase.kind)
        .context("executor disappeared from registry during dispatch")?;
    let result = {
        let mut progress = ProgressCheckpoint {
            workflow,
            state,
            checkpoint,
        };
        executor.execute(&request, context, &mut progress)?
    };

    match result {
        PhaseExecutionResult::Yielded { resume_state } => {
            let mut next = state.clone();
            let active = next
                .active
                .as_mut()
                .context("executor yielded without an active phase")?;
            active.resume_state = Some(resume_state);
            let phase_name = active.phase_name.clone();
            let boundary = RuntimeBoundary::PhaseYielded {
                phase_index,
                phase_name: phase_name.clone(),
            };
            persist_transition(workflow, state, next, boundary, checkpoint)?;
            Ok(RuntimeStatus::Yielded {
                phase_index,
                phase_name,
            })
        }
        PhaseExecutionResult::Complete(product) => {
            validate_product(phase, state.current_checkpoint.as_ref(), &product).with_context(
                || format!("executor returned invalid product for `{}`", phase.name),
            )?;
            let mut next = state.clone();
            let active = next
                .active
                .as_mut()
                .context("executor completed without an active phase")?;
            active.prepared_product = Some(product);
            let boundary = RuntimeBoundary::PhasePrepared {
                phase_index,
                phase_name: phase.name.clone(),
            };
            persist_transition(workflow, state, next, boundary, checkpoint)?;
            commit_prepared(workflow, state, checkpoint)
        }
    }
}

/// Drive ordered phases until an executor explicitly yields or the workflow is
/// complete.
pub fn run_until_yield_or_complete<C, S: RuntimeCheckpoint + ?Sized>(
    workflow: &ResolvedWorkflow,
    state: &mut WorkflowRunState,
    registry: &mut ExecutorRegistry<C>,
    context: &mut C,
    checkpoint: &mut S,
) -> Result<RuntimeStatus> {
    loop {
        match run_next_phase(workflow, state, registry, context, checkpoint)? {
            RuntimeStatus::PhaseCommitted {
                workflow_complete: false,
                ..
            } => {}
            RuntimeStatus::PhaseCommitted {
                workflow_complete: true,
                ..
            }
            | RuntimeStatus::AlreadyComplete => return Ok(RuntimeStatus::AlreadyComplete),
            yielded @ RuntimeStatus::Yielded { .. } => return Ok(yielded),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::workflow::WorkflowV2;

    fn digest(label: &str) -> String {
        sha256_identity(label.as_bytes())
    }

    fn checkpoint(label: &str) -> ImmutableModelCheckpoint {
        ImmutableModelCheckpoint::new(format!("checkpoint://{label}"), digest(label)).unwrap()
    }

    fn artifact(label: &str) -> ImmutableArtifact {
        ImmutableArtifact::new(format!("artifact://{label}"), digest(label)).unwrap()
    }

    fn training_phase(name: &str, kind: &str, task: Value) -> Value {
        serde_json::json!({
            "name": name,
            "type": kind,
            "task": task,
            "data": format!("data/{name}.jsonl"),
            "sequence_length": 64,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "steps": 1
        })
    }

    fn full_workflow() -> ResolvedWorkflow {
        let phases = vec![
            training_phase(
                "pretrain",
                "pretrain",
                serde_json::json!({"type": "causal_lm"}),
            ),
            training_phase(
                "continued",
                "continued_pretrain",
                serde_json::json!({"type": "retrieval_representation"}),
            ),
            training_phase(
                "sft",
                "sft",
                serde_json::json!({"type": "instruction_tuning"}),
            ),
            {
                let mut phase = training_phase(
                    "preference",
                    "preference",
                    serde_json::json!({"type": "pairwise_preference"}),
                );
                phase["post_training"] = serde_json::json!({
                    "algorithm": "dpo",
                    "reference": {"adapter": "test", "revision": "reference-v1"}
                });
                phase
            },
            {
                let mut phase = training_phase(
                    "rl",
                    "rl",
                    serde_json::json!({
                        "type": "verifiable_rl",
                        "verifier": {"adapter": "exact_answer"}
                    }),
                );
                phase["post_training"] = serde_json::json!({
                    "algorithm": "grpo",
                    "reference": {"adapter": "test", "revision": "reference-v1"},
                    "sampling": {"max_new_tokens": 32}
                });
                phase
            },
            {
                let mut phase = training_phase(
                    "distill",
                    "distillation",
                    serde_json::json!({"type": "causal_lm"}),
                );
                phase["post_training"] = serde_json::json!({
                    "algorithm": "forward_kl",
                    "teacher": {"adapter": "test", "revision": "teacher-v1"}
                });
                phase
            },
            serde_json::json!({
                "name": "sleep",
                "type": "sleep",
                "sleep": {
                    "standalone_trigger_clock": 100,
                    "schedule": {
                        "clock": "optimizer_steps",
                        "terminal_consolidation": "distill_into_base_v1",
                        "tiers": [
                            {"id": "fast", "update_period": 10, "reserve_slots": 1},
                            {"id": "slow", "update_period": 20, "reserve_slots": 2}
                        ]
                    },
                    "knowledge_seeding": {
                        "chunk_tokens": 64,
                        "teacher_rollouts": 1,
                        "detached_student_rollouts": 1,
                        "temperature": 1.0,
                        "forward_kl_weight": 1.0
                    },
                    "imitation": {
                        "semantic_judge_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                        "semantic_weight": 0.5,
                        "maximum_edit_distance": 8,
                        "grpo_group_size": 2
                    },
                    "retention_suite": "retention.json",
                    "retention_suite_sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                    "retention": {
                        "evaluator_hash": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                        "suite_hash": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                        "max_anchor_forward_kl": 0.05,
                        "max_anchor_regression": 0.01,
                        "min_incorporation_gain": 0.0
                    },
                    "receiver_learning_rate": 0.0001,
                    "receiver_weight_decay": 0.01,
                    "grpo_clip_epsilon": 0.2,
                    "grpo_advantage_epsilon": 0.000001,
                    "grpo_kl_coefficient": 0.04,
                    "candidate_directory": "candidates"
                }
            }),
            {
                let mut phase = training_phase(
                    "quantize",
                    "quantization",
                    serde_json::json!({"type": "causal_lm"}),
                );
                phase["quantization"] = serde_json::json!({
                    "format": "binary_g128",
                    "training": {"type": "qat"}
                });
                phase
            },
            serde_json::json!({
                "name": "evaluate",
                "type": "evaluation",
                "task": {"type": "qa_reasoning"},
                "data": "data/eval.jsonl",
                "sequence_length": 64,
                "batch_size": 2
            }),
            serde_json::json!({
                "name": "promote",
                "type": "promotion",
                "promotion": crate::workflow::test_promotion_config()
            }),
        ];
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "name": "runtime-test",
            "phases": phases
        }))
        .unwrap();
        workflow
            .resolve(std::path::Path::new("/tmp/runtime-workflow.json"))
            .unwrap()
    }

    #[derive(Default)]
    struct RecordingCheckpoint {
        boundaries: Vec<RuntimeBoundary>,
        metrics: Vec<(MetricContext, MetricEvent)>,
        fail_commit_once: bool,
    }

    impl RuntimeCheckpoint for RecordingCheckpoint {
        fn persist(&mut self, boundary: &RuntimeBoundary, _state: &WorkflowRunState) -> Result<()> {
            if self.fail_commit_once && matches!(boundary, RuntimeBoundary::PhaseCommitted { .. }) {
                self.fail_commit_once = false;
                bail!("simulated interruption while committing")
            }
            self.boundaries.push(boundary.clone());
            Ok(())
        }

        fn append_metric(&mut self, context: MetricContext, event: MetricEvent) -> Result<()> {
            self.metrics.push((context, event));
            Ok(())
        }
    }

    fn timing_event() -> MetricEvent {
        MetricEvent::PhaseTiming(crate::metrics::PhaseTimingMetrics {
            boundary: crate::metrics::PhaseBoundary::Progress,
            elapsed_seconds: 0.0,
            input_wait_seconds: 0.0,
            forward_seconds: 0.0,
            backward_seconds: 0.0,
            optimizer_seconds: 0.0,
            checkpoint_seconds: 0.0,
        })
    }

    struct CompleteExecutor;

    impl PhaseExecutor<Vec<String>> for CompleteExecutor {
        fn execute(
            &mut self,
            request: &PhaseExecutionRequest,
            context: &mut Vec<String>,
            _progress: &mut dyn PhaseProgressSink,
        ) -> Result<PhaseExecutionResult> {
            context.push(request.phase.name.clone());
            let product = match request.phase.kind.class() {
                PhaseClass::Optimization | PhaseClass::ModelMutation => {
                    PhaseProduct::ModelCandidate {
                        checkpoint: checkpoint(&request.phase.name),
                    }
                }
                PhaseClass::Assessment => PhaseProduct::Assessment {
                    report: artifact(&request.phase.name),
                },
                PhaseClass::Release => PhaseProduct::Release {
                    candidate: request
                        .input_checkpoint
                        .clone()
                        .context("test promotion has no candidate")?,
                    receipt: artifact(&request.phase.name),
                },
            };
            Ok(PhaseExecutionResult::Complete(product))
        }
    }

    #[test]
    fn every_phase_kind_dispatches_in_strict_workflow_order() {
        let workflow = full_workflow();
        let mut registry = ExecutorRegistry::new();
        for kind in ALL_PHASE_KINDS {
            registry.register(kind, CompleteExecutor).unwrap();
        }
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        let mut execution_order = Vec::new();
        let mut persistence = RecordingCheckpoint::default();

        assert_eq!(
            run_until_yield_or_complete(
                &workflow,
                &mut state,
                &mut registry,
                &mut execution_order,
                &mut persistence,
            )
            .unwrap(),
            RuntimeStatus::AlreadyComplete
        );
        assert!(state.is_complete(&workflow));
        assert_eq!(
            execution_order,
            workflow
                .phases
                .iter()
                .map(|phase| phase.name.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(state.completed_phases().len(), ALL_PHASE_KINDS.len());
        assert_eq!(
            state.current_checkpoint(),
            Some(&checkpoint("quantize")),
            "evaluation and promotion must not replace the candidate"
        );
        let committed = persistence
            .boundaries
            .iter()
            .filter_map(|boundary| match boundary {
                RuntimeBoundary::PhaseCommitted { phase_name, .. } => Some(phase_name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(committed, execution_order);
        // PhaseKind deliberately need not implement Ord/Hash for registry use;
        // names provide a stable test set.
        let dispatched_names = state
            .completed_phases()
            .iter()
            .map(|receipt| receipt.phase_kind().name())
            .collect::<BTreeSet<_>>();
        assert_eq!(dispatched_names.len(), ALL_PHASE_KINDS.len());
    }

    #[test]
    fn missing_executor_is_explicit_and_does_not_advance_state() {
        let workflow = full_workflow();
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        let pristine = state.clone();
        let mut registry = ExecutorRegistry::<()>::new();
        let mut persistence = RecordingCheckpoint::default();

        let error = run_next_phase(
            &workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unsupported executor"), "{error}");
        assert!(error.contains("pretrain"), "{error}");
        assert_eq!(state, pristine);
        assert!(persistence.boundaries.is_empty());
    }

    #[test]
    fn executor_metrics_are_bound_to_the_active_workflow_phase() {
        let full = full_workflow();
        let workflow = ResolvedWorkflow {
            version: full.version,
            name: Some("metric-binding".to_owned()),
            phases: vec![full.phases[0].clone()],
        };
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(
                PhaseKind::Pretrain,
                |request: &PhaseExecutionRequest,
                 _: &mut (),
                 progress: &mut dyn PhaseProgressSink| {
                    progress.metric(
                        MetricContext {
                            global_step: 1,
                            phase: crate::metrics::MetricPhase {
                                index: u32::try_from(request.phase_index).unwrap(),
                                name: request.phase.name.clone(),
                                kind: request.phase.kind.into(),
                            },
                            checkpoint_hash: None,
                        },
                        timing_event(),
                    )?;
                    Ok(PhaseExecutionResult::Complete(
                        PhaseProduct::ModelCandidate {
                            checkpoint: checkpoint("metric-output"),
                        },
                    ))
                },
            )
            .unwrap();
        let mut persistence = RecordingCheckpoint::default();
        run_next_phase(
            &workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap();
        assert_eq!(persistence.metrics.len(), 1);

        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(
                PhaseKind::Pretrain,
                |request: &PhaseExecutionRequest,
                 _: &mut (),
                 progress: &mut dyn PhaseProgressSink| {
                    progress.metric(
                        MetricContext {
                            global_step: 1,
                            phase: crate::metrics::MetricPhase {
                                index: u32::try_from(request.phase_index).unwrap(),
                                name: request.phase.name.clone(),
                                kind: MetricPhaseKind::Evaluation,
                            },
                            checkpoint_hash: None,
                        },
                        timing_event(),
                    )?;
                    unreachable!("invalid metric must stop the executor")
                },
            )
            .unwrap();
        let mut persistence = RecordingCheckpoint::default();
        let error = run_next_phase(
            &workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("metric kind"), "{error}");
        assert!(persistence.metrics.is_empty());
    }

    #[test]
    fn periodic_sleep_metrics_keep_the_outer_phase_identity() {
        let full = full_workflow();
        let mut phase = full.phases[0].clone();
        let mut periodic_sleep = full.phases[6]
            .sleep
            .clone()
            .expect("full workflow has a standalone sleep fixture");
        periodic_sleep.standalone_trigger_clock = None;
        phase.periodic_sleep = Some(periodic_sleep);
        let workflow = ResolvedWorkflow {
            version: full.version,
            name: Some("periodic-sleep-metric-binding".to_owned()),
            phases: vec![phase],
        };
        workflow.validate().unwrap();

        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(
                PhaseKind::Pretrain,
                |request: &PhaseExecutionRequest,
                 _: &mut (),
                 progress: &mut dyn PhaseProgressSink| {
                    progress.metric(
                        MetricContext {
                            global_step: 1,
                            phase: crate::metrics::MetricPhase {
                                index: u32::try_from(request.phase_index).unwrap(),
                                name: request.phase.name.clone(),
                                kind: MetricPhaseKind::Sleep,
                            },
                            checkpoint_hash: None,
                        },
                        timing_event(),
                    )?;
                    Ok(PhaseExecutionResult::Complete(
                        PhaseProduct::ModelCandidate {
                            checkpoint: checkpoint("periodic-sleep-metric-output"),
                        },
                    ))
                },
            )
            .unwrap();
        let mut persistence = RecordingCheckpoint::default();
        run_next_phase(
            &workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap();
        assert_eq!(persistence.metrics.len(), 1);
        assert_eq!(persistence.metrics[0].0.phase.name, workflow.phases[0].name);
        assert_eq!(persistence.metrics[0].0.phase.kind, MetricPhaseKind::Sleep);
    }

    #[derive(Default)]
    struct ResumeContext {
        calls: usize,
        observed_resume: Option<Value>,
    }

    struct YieldThenComplete;

    impl PhaseExecutor<ResumeContext> for YieldThenComplete {
        fn execute(
            &mut self,
            request: &PhaseExecutionRequest,
            context: &mut ResumeContext,
            progress: &mut dyn PhaseProgressSink,
        ) -> Result<PhaseExecutionResult> {
            context.calls += 1;
            context.observed_resume = request.resume_state.clone();
            if request.resume_state.is_none() {
                progress.checkpoint(serde_json::json!({"substep": 1}))?;
                return Ok(PhaseExecutionResult::Yielded {
                    resume_state: serde_json::json!({"substep": 2}),
                });
            }
            Ok(PhaseExecutionResult::Complete(
                PhaseProduct::ModelCandidate {
                    checkpoint: checkpoint("resumed-output"),
                },
            ))
        }
    }

    #[test]
    fn interruption_resumes_opaque_state_and_prepared_commit_is_idempotent() {
        let full = full_workflow();
        let workflow = ResolvedWorkflow {
            version: full.version,
            name: Some("resume-test".to_owned()),
            phases: vec![full.phases[0].clone()],
        };
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(PhaseKind::Pretrain, YieldThenComplete)
            .unwrap();
        let mut context = ResumeContext::default();
        let mut persistence = RecordingCheckpoint::default();

        assert!(matches!(
            run_next_phase(
                &workflow,
                &mut state,
                &mut registry,
                &mut context,
                &mut persistence,
            )
            .unwrap(),
            RuntimeStatus::Yielded { .. }
        ));
        assert_eq!(
            state
                .active_phase()
                .and_then(ActivePhaseState::resume_state),
            Some(&serde_json::json!({"substep": 2}))
        );

        // Simulate process restart from the durable JSON state.
        state = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        state.validate(&workflow).unwrap();
        let mut interrupted_commit = RecordingCheckpoint {
            fail_commit_once: true,
            ..RecordingCheckpoint::default()
        };
        let error = run_next_phase(
            &workflow,
            &mut state,
            &mut registry,
            &mut context,
            &mut interrupted_commit,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("simulated interruption"), "{error}");
        assert_eq!(context.calls, 2);
        assert_eq!(
            context.observed_resume,
            Some(serde_json::json!({"substep": 2}))
        );
        assert!(state.active_phase().unwrap().is_prepared());
        assert!(state.current_checkpoint().is_none());

        // A prepared result commits after restart without an executor, so an
        // external side effect cannot run twice.
        state = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        let mut empty_registry = ExecutorRegistry::<ResumeContext>::new();
        let mut resumed_persistence = RecordingCheckpoint::default();
        assert!(matches!(
            run_next_phase(
                &workflow,
                &mut state,
                &mut empty_registry,
                &mut context,
                &mut resumed_persistence,
            )
            .unwrap(),
            RuntimeStatus::PhaseCommitted {
                workflow_complete: true,
                ..
            }
        ));
        assert_eq!(context.calls, 2);
        assert_eq!(
            state.current_checkpoint(),
            Some(&checkpoint("resumed-output"))
        );
        assert!(state.is_complete(&workflow));
    }

    #[test]
    fn only_pretrain_can_bootstrap_without_an_input_model() {
        let complete = full_workflow();
        let pretrain_only = ResolvedWorkflow {
            version: complete.version,
            name: Some("pretrain-only".to_owned()),
            phases: vec![complete.phases[0].clone()],
        };
        WorkflowRunState::new(&pretrain_only, None).unwrap();

        for phase in complete.phases.iter().skip(1) {
            let workflow = ResolvedWorkflow {
                version: complete.version,
                name: Some(format!("{}-only", phase.kind.name())),
                phases: vec![phase.clone()],
            };
            let error = WorkflowRunState::new(&workflow, None)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("requires an input model checkpoint"),
                "{} unexpectedly accepted no input: {error}",
                phase.kind.name()
            );
            WorkflowRunState::new(&workflow, Some(checkpoint("input"))).unwrap();
        }
    }

    #[test]
    fn tampered_order_and_in_place_mutation_are_rejected() {
        let workflow = full_workflow();
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        state.next_phase = 1;
        let error = state.validate(&workflow).unwrap_err().to_string();
        assert!(error.contains("receipts"), "{error}");

        let sleep_only = ResolvedWorkflow {
            version: workflow.version,
            name: Some("sleep-only".to_owned()),
            phases: vec![workflow.phases[6].clone()],
        };
        let source = checkpoint("source");
        let mut state = WorkflowRunState::new(&sleep_only, Some(source.clone())).unwrap();
        let mut registry = ExecutorRegistry::new();
        let alias = source.clone();
        registry
            .register(
                PhaseKind::Sleep,
                move |_: &PhaseExecutionRequest, _: &mut (), _: &mut dyn PhaseProgressSink| {
                    Ok(PhaseExecutionResult::Complete(
                        PhaseProduct::ModelCandidate {
                            checkpoint: alias.clone(),
                        },
                    ))
                },
            )
            .unwrap();
        let mut persistence = RecordingCheckpoint::default();
        let error = run_next_phase(
            &sleep_only,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("aliases its immutable input URI"), "{error}");
        assert_eq!(state.current_checkpoint(), Some(&source));
        assert!(!state.active_phase().unwrap().is_prepared());
    }
}
