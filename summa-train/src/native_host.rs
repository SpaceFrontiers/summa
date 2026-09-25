//! Embedded, fail-closed WorkflowV2 host for native sleep and post-training.
//!
//! The stock CLI provides a content-pinned local-artifact runtime for
//! standalone and integrated wake-loop sleep. This host is the extension
//! surface for deployments that need other model stores, judges, evaluators,
//! post-training providers, or periodic wake executors while retaining the
//! same atomic runtime checkpoint and metric journal used by `run-workflow`.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::native_sleep::{
    NativeSleepCheckpoint, NativeSleepContextRegistry, NativeSleepPhaseContext,
    NativeSleepPhaseContextFactory, NativeSleepPhaseExecutor, NativeSleepPhaseOutcome,
    NativeSleepProgressSink,
};
use crate::posttrain::{
    NativePostTrainingPhaseExecutor, PostTrainingExecutionContext, validate_resumable_phase,
};
use crate::promotion::NativePromotionExecutor;
use crate::runtime::{
    ALL_PHASE_KINDS, ExecutorRegistry, ImmutableModelCheckpoint, PhaseExecutionRequest,
    PhaseExecutionResult, PhaseExecutor, PhaseProgressSink, RuntimeStatus, WorkflowRunState,
    run_until_yield_or_complete, workflow_signature,
};
use crate::worker::{AtomicRuntimeCheckpoint, ExternalPhaseExecutor};
use crate::workflow::{InModelSleepConfig, PhaseKind, ResolvedWorkflow};

const NATIVE_HOST_DISPATCH_VERSION: u32 = 2;

/// Optional append-only metric journal owned atomically with runtime state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeHostMetricJournal {
    pub path: PathBuf,
    pub run_id: String,
}

/// Factory which lends one concrete device-owned batch runtime and its
/// idempotent publisher to the built-in DPO, forward-KL, or GRPO executor.
///
/// The callback must be invoked exactly once. Implementations normally load
/// the model/optimizer generation named by `request`, construct
/// [`PostTrainingExecutionContext`], and lend it to `operation`. The runtime
/// may jointly own trainable and frozen models so it can fuse/batch their
/// execution; no per-example host-score adapter is inferred by trainer core.
pub trait NativePostTrainingContextFactory {
    fn identity(&self) -> &str;

    /// Registered identity of the first-party periodic-sleep controller lent
    /// through [`PostTrainingExecutionContext`]. A typed post-training phase
    /// with `periodic_sleep` is rejected during host preflight when absent.
    fn periodic_sleep_identity(&self) -> Option<&str>;

    fn with_context(
        &mut self,
        request: &PhaseExecutionRequest,
        operation: &mut dyn for<'a> FnMut(&mut PostTrainingExecutionContext<'a>) -> Result<()>,
    ) -> Result<()>;
}

/// Deployment-owned in-process executor for wake optimization phases that
/// carry `periodic_sleep` and are not handled by the built-in typed
/// post-training executor. It owns the complete phase, must drive the
/// configured sleep boundary before advancing past a due optimizer step, and
/// publishes only immutable, transaction-idempotent checkpoints. Typed DPO,
/// forward-KL, and GRPO phases instead use
/// [`PostTrainingBoundaryHook`](crate::posttrain::PostTrainingBoundaryHook) through
/// [`PostTrainingExecutionContext`], so their prepared update and sleep receipt
/// share one durable cursor.
pub trait NativePeriodicWakeExecutor {
    fn identity(&self) -> &str;

    fn execute_periodic_wake(
        &mut self,
        request: &PhaseExecutionRequest,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult>;
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeHostDispatch {
    ExternalWorker,
    NativePostTraining,
    NativePeriodicWake,
    NativeSleep,
    NativePromotion,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeHostPhaseDispatch {
    pub phase_index: usize,
    pub phase_name: String,
    pub phase_kind: PhaseKind,
    pub dispatch: NativeHostDispatch,
}

/// Explicit adapter registrations consumed by [`NativeWorkflowHost`].
/// Registering an adapter twice is rejected; omitted routes fail before an
/// atomic runtime checkpoint is created or loaded.
#[derive(Default)]
pub struct NativeWorkflowAdapters {
    external: Option<ExternalPhaseExecutor>,
    sleep: NativeSleepContextRegistry,
    post_training: Option<Box<dyn NativePostTrainingContextFactory>>,
    periodic_wake: Option<Box<dyn NativePeriodicWakeExecutor>>,
}

impl NativeWorkflowAdapters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_external(&mut self, executor: ExternalPhaseExecutor) -> Result<()> {
        ensure!(
            self.external.is_none(),
            "an external phase worker is already registered"
        );
        self.external = Some(executor);
        Ok(())
    }

    pub fn register_sleep_factory<F>(&mut self, factory: F) -> Result<()>
    where
        F: NativeSleepPhaseContextFactory + 'static,
    {
        self.sleep.register_phase_factory(factory)
    }

    pub fn register_post_training_factory<F>(&mut self, factory: F) -> Result<()>
    where
        F: NativePostTrainingContextFactory + 'static,
    {
        validate_identity(factory.identity(), "native post-training factory")?;
        if let Some(identity) = factory.periodic_sleep_identity() {
            validate_identity(identity, "native post-training periodic-sleep controller")?;
        }
        ensure!(
            self.post_training.is_none(),
            "a native post-training context factory is already registered"
        );
        self.post_training = Some(Box::new(factory));
        Ok(())
    }

    pub fn register_periodic_wake_executor<E>(&mut self, executor: E) -> Result<()>
    where
        E: NativePeriodicWakeExecutor + 'static,
    {
        validate_identity(executor.identity(), "native periodic-wake executor")?;
        ensure!(
            self.periodic_wake.is_none(),
            "a native periodic-wake executor is already registered"
        );
        self.periodic_wake = Some(Box::new(executor));
        Ok(())
    }

    /// Resolve every phase to its actual execution surface. This performs the
    /// same fail-closed validation used by `start` and `resume`.
    pub fn dispatch_plan(
        &self,
        workflow: &ResolvedWorkflow,
    ) -> Result<Vec<NativeHostPhaseDispatch>> {
        workflow.validate()?;
        workflow
            .phases
            .iter()
            .enumerate()
            .map(|(phase_index, phase)| {
                let dispatch = match phase.kind {
                    PhaseKind::Sleep => {
                        ensure!(
                            self.sleep.has_phase_factory(),
                            "sleep phase `{}` requires a registered native sleep factory",
                            phase.name
                        );
                        NativeHostDispatch::NativeSleep
                    }
                    PhaseKind::Promotion => NativeHostDispatch::NativePromotion,
                    _ if phase.post_training.is_some() => {
                        validate_resumable_phase(phase).with_context(|| {
                            format!(
                                "phase `{}` is incompatible with the native post-training executor",
                                phase.name
                            )
                        })?;
                        let factory = self.post_training.as_deref().with_context(|| {
                            format!(
                                "phase `{}` has typed post_training and requires a registered native post-training context factory",
                                phase.name
                            )
                        })?;
                        ensure!(
                            phase.periodic_sleep.is_none()
                                || factory.periodic_sleep_identity().is_some(),
                            "phase `{}` has periodic post_training and requires a registered native post-training periodic-sleep controller",
                            phase.name,
                        );
                        NativeHostDispatch::NativePostTraining
                    }
                    _ if phase.periodic_sleep.is_some() => {
                        ensure!(
                            self.periodic_wake.is_some(),
                            "phase `{}` has periodic_sleep and requires a registered in-process native periodic-wake executor",
                            phase.name
                        );
                        NativeHostDispatch::NativePeriodicWake
                    }
                    _ => {
                        ensure!(
                            self.external.is_some(),
                            "phase `{}` ({}) has no registered native route or pinned external worker",
                            phase.name,
                            phase.kind.name()
                        );
                        NativeHostDispatch::ExternalWorker
                    }
                };
                Ok(NativeHostPhaseDispatch {
                    phase_index,
                    phase_name: phase.name.clone(),
                    phase_kind: phase.kind,
                    dispatch,
                })
            })
            .collect()
    }

    fn dispatch_identity(&self, workflow: &ResolvedWorkflow) -> Result<String> {
        let plan = self.dispatch_plan(workflow)?;
        let signature = workflow_signature(workflow)?;
        let external_identity = self
            .external
            .as_ref()
            .map(ExternalPhaseExecutor::execution_identity);
        let mut hasher = Sha256::new();
        hash_part(&mut hasher, b"summa-native-workflow-host");
        hash_part(&mut hasher, &NATIVE_HOST_DISPATCH_VERSION.to_le_bytes());
        hash_part(&mut hasher, signature.as_bytes());
        for route in &plan {
            hash_part(&mut hasher, &serde_json::to_vec(route)?);
        }
        for identity in [
            external_identity.as_deref(),
            self.sleep.phase_factory_identity(),
            self.post_training
                .as_deref()
                .map(NativePostTrainingContextFactory::identity),
            self.post_training
                .as_deref()
                .and_then(NativePostTrainingContextFactory::periodic_sleep_identity),
            self.periodic_wake
                .as_deref()
                .map(NativePeriodicWakeExecutor::identity),
        ] {
            hash_part(&mut hasher, identity.unwrap_or("none").as_bytes());
        }
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }

    fn into_runtime(
        self,
        workflow_signature: &str,
    ) -> Result<(
        ExecutorRegistry<NativeWorkflowContext>,
        NativeWorkflowContext,
    )> {
        let Self {
            external,
            sleep,
            post_training,
            periodic_wake,
        } = self;
        let mut registry = ExecutorRegistry::new();
        let routed = RoutedNativePhaseExecutor {
            external,
            post_training: NativePostTrainingPhaseExecutor::new(workflow_signature.to_owned())?,
        };
        for kind in ALL_PHASE_KINDS {
            match kind {
                PhaseKind::Sleep => registry.register(
                    kind,
                    NativeSleepPhaseExecutor::new(workflow_signature.to_owned())?,
                )?,
                PhaseKind::Promotion => registry.register(kind, NativePromotionExecutor)?,
                _ => registry.register(kind, routed.clone())?,
            }
        }
        Ok((
            registry,
            NativeWorkflowContext {
                sleep,
                post_training,
                periodic_wake,
            },
        ))
    }
}

pub struct NativeWorkflowHost {
    workflow: ResolvedWorkflow,
    state: WorkflowRunState,
    registry: ExecutorRegistry<NativeWorkflowContext>,
    context: NativeWorkflowContext,
    checkpoint: AtomicRuntimeCheckpoint,
    dispatch_sha256: String,
    // A failed executor may have appended metrics after the last durable
    // runtime boundary. Continuing with the same in-memory journal could make
    // that failed-attempt tail part of a later commit. Reopening through
    // `resume` validates and truncates the journal to the exact committed
    // prefix before execution continues.
    recovery_required: bool,
}

impl NativeWorkflowHost {
    /// Start a new embedded workflow after validating every dispatch route.
    /// No state or metric file is created if adapter routing is incomplete.
    pub fn start(
        workflow: ResolvedWorkflow,
        adapters: NativeWorkflowAdapters,
        state_path: impl Into<PathBuf>,
        metrics: Option<NativeHostMetricJournal>,
        initial_checkpoint: Option<ImmutableModelCheckpoint>,
    ) -> Result<Self> {
        Self::open(
            workflow,
            adapters,
            state_path.into(),
            metrics,
            OpenMode::Start(initial_checkpoint),
        )
    }

    /// Resume an embedded workflow with the same workflow and registered
    /// adapter identities. Atomic loading rejects a different dispatch hash or
    /// metric prefix before any executor runs.
    pub fn resume(
        workflow: ResolvedWorkflow,
        adapters: NativeWorkflowAdapters,
        state_path: impl Into<PathBuf>,
        metrics: Option<NativeHostMetricJournal>,
    ) -> Result<Self> {
        Self::open(
            workflow,
            adapters,
            state_path.into(),
            metrics,
            OpenMode::Resume,
        )
    }

    fn open(
        workflow: ResolvedWorkflow,
        adapters: NativeWorkflowAdapters,
        state_path: PathBuf,
        metrics: Option<NativeHostMetricJournal>,
        mode: OpenMode,
    ) -> Result<Self> {
        let dispatch_sha256 = adapters.dispatch_identity(&workflow)?;
        let signature = workflow_signature(&workflow)?;
        let (registry, context) = adapters.into_runtime(&signature)?;
        let mut checkpoint = AtomicRuntimeCheckpoint::new(state_path, &dispatch_sha256)?;
        if let Some(metrics) = metrics {
            checkpoint.configure_metrics(metrics.path, metrics.run_id, mode.is_resume())?;
        }
        let state = match mode {
            OpenMode::Start(initial_checkpoint) => {
                let state = WorkflowRunState::new(&workflow, initial_checkpoint)?;
                checkpoint.initialize(&state)?;
                state
            }
            OpenMode::Resume => checkpoint.load(&workflow)?,
        };
        Ok(Self {
            workflow,
            state,
            registry,
            context,
            checkpoint,
            dispatch_sha256,
            recovery_required: false,
        })
    }

    pub fn dispatch_sha256(&self) -> &str {
        &self.dispatch_sha256
    }

    pub fn state(&self) -> &WorkflowRunState {
        &self.state
    }

    pub fn workflow(&self) -> &ResolvedWorkflow {
        &self.workflow
    }

    pub fn drive_until_yield_or_complete(&mut self) -> Result<RuntimeStatus> {
        ensure!(
            !self.recovery_required,
            "native workflow host must be reopened with resume after an execution or persistence error"
        );
        let result = run_until_yield_or_complete(
            &self.workflow,
            &mut self.state,
            &mut self.registry,
            &mut self.context,
            &mut self.checkpoint,
        );
        if result.is_err() {
            self.recovery_required = true;
        }
        result
    }
}

enum OpenMode {
    Start(Option<ImmutableModelCheckpoint>),
    Resume,
}

impl OpenMode {
    fn is_resume(&self) -> bool {
        matches!(self, Self::Resume)
    }
}

pub struct NativeWorkflowContext {
    sleep: NativeSleepContextRegistry,
    post_training: Option<Box<dyn NativePostTrainingContextFactory>>,
    periodic_wake: Option<Box<dyn NativePeriodicWakeExecutor>>,
}

impl NativeSleepPhaseContext for NativeWorkflowContext {
    fn drive_sleep_phase(
        &mut self,
        request: &PhaseExecutionRequest,
        config: &InModelSleepConfig,
        resume: Option<NativeSleepCheckpoint>,
        progress: &mut dyn NativeSleepProgressSink,
    ) -> Result<NativeSleepPhaseOutcome> {
        self.sleep
            .drive_sleep_phase(request, config, resume, progress)
    }
}

#[derive(Clone)]
struct RoutedNativePhaseExecutor {
    external: Option<ExternalPhaseExecutor>,
    post_training: NativePostTrainingPhaseExecutor,
}

impl PhaseExecutor<NativeWorkflowContext> for RoutedNativePhaseExecutor {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        context: &mut NativeWorkflowContext,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        ensure!(
            !matches!(request.phase.kind, PhaseKind::Sleep | PhaseKind::Promotion),
            "ordinary native host router received reserved phase `{}`",
            request.phase.name
        );
        if let (Some(_), Some(factory)) = (
            request.phase.post_training.as_ref(),
            context.post_training.as_mut(),
        ) {
            let registered_periodic_identity = factory.periodic_sleep_identity().map(str::to_owned);
            let mut result = None;
            let mut operation = |phase_context: &mut PostTrainingExecutionContext<'_>| {
                ensure!(
                    result.is_none(),
                    "post-training factory invoked its callback twice"
                );
                if request.phase.periodic_sleep.is_some() {
                    let actual = phase_context
                        .boundary_hook
                        .as_deref()
                        .context("periodic post-training context omitted its registered boundary controller")?
                        .identity();
                    ensure!(
                        registered_periodic_identity.as_deref() == Some(actual),
                        "periodic post-training context supplied a controller outside its registered identity"
                    );
                }
                result = Some(
                    self.post_training
                        .execute(request, phase_context, progress)?,
                );
                Ok(())
            };
            factory.with_context(request, &mut operation)?;
            return result.context("post-training factory did not provide an adapter context");
        }
        if request.phase.periodic_sleep.is_some() {
            return context
                .periodic_wake
                .as_mut()
                .with_context(|| {
                    format!(
                        "phase `{}` has periodic_sleep but no native periodic-wake executor",
                        request.phase.name
                    )
                })?
                .execute_periodic_wake(request, progress);
        }
        self.external
            .as_mut()
            .with_context(|| {
                format!(
                    "phase `{}` has no native route or external worker",
                    request.phase.name
                )
            })?
            .execute(request, context, progress)
    }
}

fn validate_identity(value: &str, label: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .with_context(|| format!("{label} identity must use sha256:<64 lowercase hex>"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} identity must use sha256:<64 lowercase hex>"
    );
    Ok(())
}

fn hash_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use anyhow::bail;

    use super::*;
    use crate::workflow::load_workflow;

    fn identity(label: &str) -> String {
        format!("sha256:{:x}", Sha256::digest(label.as_bytes()))
    }

    struct UnusedPostTrainingFactory {
        identity: String,
    }

    impl NativePostTrainingContextFactory for UnusedPostTrainingFactory {
        fn identity(&self) -> &str {
            &self.identity
        }

        fn periodic_sleep_identity(&self) -> Option<&str> {
            Some(&self.identity)
        }

        fn with_context(
            &mut self,
            _: &PhaseExecutionRequest,
            _: &mut dyn for<'a> FnMut(&mut PostTrainingExecutionContext<'a>) -> Result<()>,
        ) -> Result<()> {
            bail!("dispatch-plan fixture must not execute post-training")
        }
    }

    struct UnusedPeriodicWake {
        identity: String,
    }

    impl NativePeriodicWakeExecutor for UnusedPeriodicWake {
        fn identity(&self) -> &str {
            &self.identity
        }

        fn execute_periodic_wake(
            &mut self,
            _: &PhaseExecutionRequest,
            _: &mut dyn PhaseProgressSink,
        ) -> Result<PhaseExecutionResult> {
            bail!("dispatch-plan fixture must not execute periodic wake")
        }
    }

    fn external(path: &Path) -> ExternalPhaseExecutor {
        let bytes = b"native-host-test-worker-v1";
        fs::write(path, bytes).unwrap();
        #[cfg(unix)]
        {
            let mut permissions = fs::metadata(path).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(path, permissions).unwrap();
        }
        ExternalPhaseExecutor::new(path, Vec::new()).unwrap()
    }

    fn education_workflow() -> ResolvedWorkflow {
        load_workflow(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.education.example.json"),
        )
        .unwrap()
    }

    fn education_adapters(worker: &Path, periodic: bool) -> NativeWorkflowAdapters {
        let mut adapters = NativeWorkflowAdapters::new();
        adapters.register_external(external(worker)).unwrap();
        adapters
            .register_post_training_factory(UnusedPostTrainingFactory {
                identity: identity("post-training-factory-v1"),
            })
            .unwrap();
        if periodic {
            adapters
                .register_periodic_wake_executor(UnusedPeriodicWake {
                    identity: identity("periodic-wake-v1"),
                })
                .unwrap();
        }
        adapters
    }

    #[test]
    fn education_dispatch_never_sends_periodic_sleep_to_external_worker() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("worker");
        let workflow = education_workflow();

        let incomplete = education_adapters(&worker, false);
        let error = incomplete.dispatch_plan(&workflow).unwrap_err().to_string();
        assert!(
            error.contains("language-foundations")
                && error.contains("periodic_sleep")
                && error.contains("in-process"),
            "{error}"
        );

        let adapters = education_adapters(&worker, true);
        let plan = adapters.dispatch_plan(&workflow).unwrap();
        for route in &plan {
            let expected = match route.phase_kind {
                PhaseKind::Preference | PhaseKind::Distillation | PhaseKind::Rl => {
                    NativeHostDispatch::NativePostTraining
                }
                kind if kind.uses_optimizer() => NativeHostDispatch::NativePeriodicWake,
                PhaseKind::Promotion => NativeHostDispatch::NativePromotion,
                PhaseKind::Evaluation => NativeHostDispatch::ExternalWorker,
                other => panic!("unexpected education phase kind {other:?}"),
            };
            assert_eq!(route.dispatch, expected, "{}", route.phase_name);
        }
    }

    #[test]
    fn typed_post_training_owns_its_periodic_sleep_boundaries() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("worker");
        let workflow = education_workflow();

        let plan = education_adapters(&worker, true)
            .dispatch_plan(&workflow)
            .unwrap();
        let preference = plan
            .iter()
            .find(|route| route.phase_name == "preference-dpo")
            .unwrap();
        assert_eq!(preference.dispatch, NativeHostDispatch::NativePostTraining);
    }

    #[test]
    fn periodic_post_training_does_not_require_a_whole_phase_wake_executor() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("worker");
        let mut workflow = education_workflow();
        workflow
            .phases
            .retain(|phase| phase.name == "preference-dpo");
        let adapters = education_adapters(&worker, false);

        let plan = adapters.dispatch_plan(&workflow).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].dispatch, NativeHostDispatch::NativePostTraining);
    }

    #[test]
    fn invalid_native_post_training_fails_before_runtime_state_creation() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("worker");
        let state = temporary.path().join("runtime.json");
        let mut workflow = education_workflow();
        workflow
            .phases
            .retain(|phase| phase.name == "preference-dpo");
        assert_eq!(workflow.phases.len(), 1);
        workflow.phases[0].shuffle_buffer = Some(32);

        let error = match NativeWorkflowHost::start(
            workflow,
            education_adapters(&worker, false),
            &state,
            None,
            None,
        ) {
            Ok(_) => panic!("incompatible native post-training phase unexpectedly started"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains("deterministic input order"), "{error}");
        assert!(!state.exists(), "invalid phase created runtime state");
    }

    #[test]
    fn execution_error_requires_reopening_at_the_committed_prefix() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("worker");
        let state = temporary.path().join("runtime.json");
        let mut host = NativeWorkflowHost::start(
            education_workflow(),
            education_adapters(&worker, true),
            &state,
            None,
            None,
        )
        .unwrap();

        let first = host
            .drive_until_yield_or_complete()
            .unwrap_err()
            .to_string();
        assert!(first.contains("must not execute periodic wake"), "{first}");
        let second = host
            .drive_until_yield_or_complete()
            .unwrap_err()
            .to_string();
        assert!(second.contains("reopened with resume"), "{second}");
    }

    #[test]
    fn host_validates_routes_before_atomic_state_and_resumes_metric_prefix() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("worker");
        let state = temporary.path().join("runtime.json");
        let metrics = temporary.path().join("metrics.jsonl");
        let workflow = education_workflow();

        let error = match NativeWorkflowHost::start(
            workflow.clone(),
            education_adapters(&worker, false),
            &state,
            None,
            None,
        ) {
            Ok(_) => panic!("incomplete dispatch unexpectedly started"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("periodic_sleep"), "{error}");
        assert!(!state.exists(), "invalid dispatch created runtime state");

        let host = NativeWorkflowHost::start(
            workflow.clone(),
            education_adapters(&worker, true),
            &state,
            Some(NativeHostMetricJournal {
                path: metrics.clone(),
                run_id: "education-host-test".into(),
            }),
            None,
        )
        .unwrap();
        let dispatch = host.dispatch_sha256().to_owned();
        assert!(state.is_file());
        assert!(metrics.is_file());
        drop(host);

        let resumed = NativeWorkflowHost::resume(
            workflow,
            education_adapters(&worker, true),
            &state,
            Some(NativeHostMetricJournal {
                path: metrics,
                run_id: "education-host-test".into(),
            }),
        )
        .unwrap();
        assert_eq!(resumed.dispatch_sha256(), dispatch);
        assert_eq!(resumed.state().next_phase_index(), 0);
    }
}
