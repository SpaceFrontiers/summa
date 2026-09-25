//! Versioned subprocess bridge for WorkflowV2 phase executors.
//!
//! The core runtime stays algorithm-neutral while a local worker executable
//! can implement any phase/task combination. Requests and responses are JSONL;
//! progress records are durably checkpointed as they arrive, and terminal
//! products still pass the runtime's immutable-candidate rules.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
#[cfg(test)]
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[cfg(not(unix))]
use anyhow::bail;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::artifact_io::{read_regular_bounded, validate_sha256_identity};
#[cfg(test)]
use crate::metrics::MetricPhaseKind;
use crate::metrics::{MetricContext, MetricEvent, MetricLogState, MetricPhase, MetricWriter};
#[cfg(unix)]
use crate::protocol_process::{ProtocolRead, SupervisedProcess};
use crate::runtime::{
    ImmutableModelCheckpoint, PhaseExecutionRequest, PhaseExecutionResult, PhaseExecutor,
    PhaseProduct, PhaseProgressSink, RuntimeBoundary, RuntimeCheckpoint, WorkflowRunState,
};
use crate::workflow::ResolvedWorkflow;

pub const PHASE_WORKER_PROTOCOL_VERSION: u32 = 2;
pub const RUNTIME_CHECKPOINT_FILE_VERSION: u32 = 2;
const PHASE_WORKER_EXECUTION_CONTRACT_VERSION: u32 = 1;
const MAX_WORKER_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_RUNTIME_CHECKPOINT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PHASE_WORKER_ARGUMENTS: usize = 128;
const MAX_PHASE_WORKER_ARGUMENT_BYTES: usize = 4 * 1024;
const MAX_PHASE_WORKER_TOTAL_ARGUMENT_BYTES: usize = 64 * 1024;
const DEFAULT_PHASE_WORKER_TIMEOUT: Duration = Duration::from_secs(3_600);

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseWorkerRequest {
    pub version: u32,
    pub phase_index: usize,
    pub phase: crate::workflow::PhaseV2,
    pub input_checkpoint: Option<ImmutableModelCheckpoint>,
    pub resume_state: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PhaseWorkerMessage {
    Metric {
        global_step: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        checkpoint_hash: Option<String>,
        event: MetricEvent,
    },
    Progress {
        resume_state: Value,
    },
    Yielded {
        resume_state: Value,
    },
    Complete {
        product: PhaseProduct,
    },
}

/// A local phase worker. No shell is involved: the executable and each
/// argument are passed directly to `Command`.
#[derive(Clone, Debug)]
pub struct ExternalPhaseExecutor {
    executable: PathBuf,
    arguments: Vec<OsString>,
    timeout: Duration,
}

impl ExternalPhaseExecutor {
    pub fn new(executable: impl Into<PathBuf>, arguments: Vec<OsString>) -> Result<Self> {
        let executor = Self {
            executable: executable.into(),
            arguments,
            timeout: DEFAULT_PHASE_WORKER_TIMEOUT,
        };
        validate_worker_arguments(&executor.arguments)?;
        Ok(executor)
    }

    /// Set the hard wall-clock bound for one phase-worker invocation. Timeout
    /// always terminates and reaps the worker's complete process group.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        ensure!(!timeout.is_zero(), "phase worker timeout must be positive");
        checked_worker_deadline(timeout)?;
        self.timeout = timeout;
        Ok(self)
    }

    /// Stable identity of all executable semantics owned by this adapter.
    /// Runtime checkpoints bind this value so that a resumed run cannot
    /// silently change the program, arguments, or bound it dispatches.
    pub fn execution_identity(&self) -> String {
        let mut hasher = Sha256::new();
        hash_execution_part(&mut hasher, b"summa-external-phase-executor");
        hash_execution_part(&mut hasher, &PHASE_WORKER_PROTOCOL_VERSION.to_le_bytes());
        hash_execution_part(
            &mut hasher,
            &PHASE_WORKER_EXECUTION_CONTRACT_VERSION.to_le_bytes(),
        );
        hash_execution_part(&mut hasher, b"empty-environment");
        hash_execution_part(&mut hasher, b"working-directory:/");
        #[cfg(unix)]
        hash_execution_part(&mut hasher, self.executable.as_os_str().as_bytes());
        #[cfg(not(unix))]
        hash_execution_part(&mut hasher, self.executable.to_string_lossy().as_bytes());
        hash_execution_part(&mut hasher, &self.timeout.as_nanos().to_le_bytes());
        hash_execution_part(&mut hasher, &(self.arguments.len() as u64).to_le_bytes());
        for argument in &self.arguments {
            #[cfg(unix)]
            hash_execution_part(&mut hasher, argument.as_os_str().as_bytes());
            #[cfg(not(unix))]
            {
                let text = argument.to_string_lossy();
                hash_execution_part(&mut hasher, text.as_bytes());
            }
        }
        format!("sha256:{:x}", hasher.finalize())
    }

    #[cfg(unix)]
    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command
            .args(&self.arguments)
            .env_clear()
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .process_group(0);
        command
    }
}

fn validate_worker_arguments(arguments: &[OsString]) -> Result<()> {
    ensure!(
        arguments.len() <= MAX_PHASE_WORKER_ARGUMENTS,
        "phase worker has more than {MAX_PHASE_WORKER_ARGUMENTS} arguments"
    );
    let mut total = 0usize;
    for (index, argument) in arguments.iter().enumerate() {
        #[cfg(unix)]
        let bytes = argument.as_os_str().as_bytes();
        #[cfg(not(unix))]
        let encoded = argument.to_string_lossy();
        #[cfg(not(unix))]
        let bytes = encoded.as_bytes();
        ensure!(
            bytes.len() <= MAX_PHASE_WORKER_ARGUMENT_BYTES,
            "phase worker argument {index} exceeds {MAX_PHASE_WORKER_ARGUMENT_BYTES} bytes"
        );
        ensure!(
            !bytes.contains(&0),
            "phase worker argument {index} contains a NUL byte"
        );
        total = total
            .checked_add(bytes.len())
            .context("phase worker argument bytes overflow usize")?;
    }
    ensure!(
        total <= MAX_PHASE_WORKER_TOTAL_ARGUMENT_BYTES,
        "phase worker arguments exceed {MAX_PHASE_WORKER_TOTAL_ARGUMENT_BYTES} total bytes"
    );
    Ok(())
}

fn hash_execution_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

impl<C> PhaseExecutor<C> for ExternalPhaseExecutor {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        _: &mut C,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        #[cfg(not(unix))]
        {
            let _ = (request, progress);
            bail!("external phase workers require a Unix process host");
        }
        #[cfg(unix)]
        {
            let wire_request = PhaseWorkerRequest {
                version: PHASE_WORKER_PROTOCOL_VERSION,
                phase_index: request.phase_index,
                phase: request.phase.clone(),
                input_checkpoint: request.input_checkpoint.clone(),
                resume_state: request.resume_state.clone(),
            };
            let mut encoded_request = serde_json::to_vec(&wire_request)?;
            ensure!(
                encoded_request.len() < MAX_WORKER_MESSAGE_BYTES,
                "phase worker request is larger than {MAX_WORKER_MESSAGE_BYTES} bytes"
            );
            encoded_request.push(b'\n');
            let deadline = checked_worker_deadline(self.timeout)?;
            let child = self.command().spawn().with_context(|| {
                format!("failed to start phase worker {}", self.executable.display())
            })?;
            let mut child =
                SupervisedProcess::new(child, "phase worker", MAX_WORKER_MESSAGE_BYTES)?;
            let mut written = 0_usize;
            let mut terminal = None;
            loop {
                child.write_available(&encoded_request, &mut written)?;
                if written == encoded_request.len() {
                    child.close_input();
                }
                drain_phase_output(
                    &mut child,
                    request,
                    progress,
                    &mut terminal,
                    deadline,
                    self.timeout,
                )?;
                if let Some(status) = child.try_wait()? {
                    // Do not wait for pipe EOF after leader exit. An unauthorized
                    // setsid() descendant can retain stdout outside the process
                    // group; every byte the leader committed is already readable.
                    child.terminate_process_group();
                    drain_phase_output(
                        &mut child,
                        request,
                        progress,
                        &mut terminal,
                        deadline,
                        self.timeout,
                    )?;
                    child.finish_output_at_leader_exit()?;
                    ensure!(
                        written == encoded_request.len(),
                        "phase worker exited before consuming its complete request"
                    );
                    ensure!(status.success(), "phase worker exited with status {status}");
                    return terminal.context("phase worker exited without a terminal message");
                }
                ensure_worker_before_deadline(deadline, self.timeout)?;
                child.wait_for_activity(written < encoded_request.len(), deadline)?;
            }
        }
    }
}

#[cfg(unix)]
fn drain_phase_output(
    child: &mut SupervisedProcess,
    request: &PhaseExecutionRequest,
    progress: &mut dyn PhaseProgressSink,
    terminal: &mut Option<PhaseExecutionResult>,
    deadline: Instant,
    timeout: Duration,
) -> Result<()> {
    loop {
        ensure_worker_before_deadline(deadline, timeout)?;
        let ProtocolRead::Line(line) = child.read_line()? else {
            return Ok(());
        };
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        ensure!(
            terminal.is_none(),
            "phase worker emitted data after a terminal message"
        );
        let message: PhaseWorkerMessage =
            serde_json::from_slice(&line).context("phase worker emitted invalid protocol JSON")?;
        match message {
            PhaseWorkerMessage::Metric {
                global_step,
                checkpoint_hash,
                event,
            } => {
                if let Some(hash) = &checkpoint_hash {
                    validate_sha256_identity(hash, "phase worker metric checkpoint identity")
                        .context(
                            "phase worker metric checkpoint_hash is not a canonical SHA-256",
                        )?;
                }
                event
                    .validate()
                    .context("phase worker emitted an invalid typed metric")?;
                progress.metric(
                    MetricContext {
                        global_step,
                        phase: metric_phase(&request.phase, request.phase_index)?,
                        checkpoint_hash,
                    },
                    event,
                )?;
            }
            PhaseWorkerMessage::Progress { resume_state } => {
                progress.checkpoint(resume_state)?;
            }
            PhaseWorkerMessage::Yielded { resume_state } => {
                *terminal = Some(PhaseExecutionResult::Yielded { resume_state });
            }
            PhaseWorkerMessage::Complete { product } => {
                *terminal = Some(PhaseExecutionResult::Complete(product));
            }
        }
    }
}

fn checked_worker_deadline(timeout: Duration) -> Result<Instant> {
    Instant::now()
        .checked_add(timeout)
        .context("phase worker timeout is too large for the monotonic clock")
}

fn ensure_worker_before_deadline(deadline: Instant, timeout: Duration) -> Result<()> {
    ensure!(
        Instant::now() < deadline,
        "phase worker exceeded its {timeout:?} wall timeout"
    );
    Ok(())
}

fn metric_phase(phase: &crate::workflow::PhaseV2, phase_index: usize) -> Result<MetricPhase> {
    let index = u32::try_from(phase_index).context("workflow phase index exceeds u32")?;
    Ok(MetricPhase {
        index,
        name: phase.name.clone(),
        kind: phase.kind.into(),
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCheckpointFile {
    pub version: u32,
    pub executor_sha256: String,
    pub metrics: Option<CommittedMetricLog>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary: Option<RuntimeBoundary>,
    pub state: WorkflowRunState,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommittedMetricLog {
    pub run_id: String,
    pub records: u64,
    pub last_global_step: Option<u64>,
}

enum RuntimeMetrics {
    Disabled,
    Active(MetricWriter),
    ResumePending { path: PathBuf, run_id: String },
}

/// Atomic, synced persistence for the generic workflow runtime.
pub struct AtomicRuntimeCheckpoint {
    path: PathBuf,
    executor_sha256: String,
    metrics: RuntimeMetrics,
}

impl AtomicRuntimeCheckpoint {
    pub fn new(path: impl Into<PathBuf>, executor_sha256: impl Into<String>) -> Result<Self> {
        let checkpoint = Self {
            path: path.into(),
            executor_sha256: executor_sha256.into(),
            metrics: RuntimeMetrics::Disabled,
        };
        ensure!(
            !checkpoint.path.as_os_str().is_empty(),
            "workflow-runtime checkpoint path is empty"
        );
        validate_sha256_identity(&checkpoint.executor_sha256, "runtime executor identity")?;
        Ok(checkpoint)
    }

    /// Bind an optional metric journal to this runtime checkpoint. A new run
    /// refuses to replace an existing journal. Resume defers opening and tail
    /// truncation until the runtime checkpoint has supplied the committed
    /// record count.
    pub fn configure_metrics(
        &mut self,
        path: impl Into<PathBuf>,
        run_id: impl Into<String>,
        resume: bool,
    ) -> Result<()> {
        ensure!(
            matches!(self.metrics, RuntimeMetrics::Disabled),
            "workflow-runtime metrics are already configured"
        );
        let path = path.into();
        let run_id = run_id.into();
        ensure!(
            !path.as_os_str().is_empty(),
            "workflow metric journal path is empty"
        );
        ensure!(
            path != self.path,
            "workflow metric journal and runtime checkpoint must use different paths"
        );
        if resume {
            ensure_regular_non_symlink(&path, "workflow metric journal")?;
            self.metrics = RuntimeMetrics::ResumePending { path, run_id };
        } else {
            ensure_path_absent(&path, "workflow metric journal")?;
            self.metrics = RuntimeMetrics::Active(MetricWriter::create(path, run_id)?);
        }
        Ok(())
    }

    pub fn initialize(&mut self, state: &WorkflowRunState) -> Result<()> {
        ensure_path_absent(&self.path, "workflow-runtime checkpoint")
            .with_context(|| "use resume for an existing workflow-runtime checkpoint")?;
        self.publish(None, state)
    }

    fn read_saved(&self) -> Result<RuntimeCheckpointFile> {
        ensure_regular_non_symlink(&self.path, "workflow-runtime checkpoint")?;
        let bytes = read_regular_bounded(
            &self.path,
            MAX_RUNTIME_CHECKPOINT_BYTES,
            "workflow-runtime checkpoint",
        )
        .with_context(|| {
            format!(
                "failed to read workflow-runtime checkpoint {}",
                self.path.display()
            )
        })?;
        let saved: RuntimeCheckpointFile = serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "invalid workflow-runtime checkpoint {}",
                self.path.display()
            )
        })?;
        Ok(saved)
    }

    fn validate_execution_identity(&self, saved: &RuntimeCheckpointFile) -> Result<()> {
        ensure!(
            saved.version == RUNTIME_CHECKPOINT_FILE_VERSION,
            "unsupported workflow-runtime checkpoint-file version {}",
            saved.version
        );
        ensure!(
            saved.executor_sha256 == self.executor_sha256,
            "workflow-runtime checkpoint was produced by a different execution dispatch"
        );
        Ok(())
    }

    /// Read-only preflight for adapter-backed resume. Callers can reject a
    /// replacement native runtime before constructing it or truncating the
    /// metric journal to the checkpoint's committed prefix.
    pub fn verify_execution_identity(&self) -> Result<()> {
        let saved = self.read_saved()?;
        self.validate_execution_identity(&saved)
    }

    pub fn load(&mut self, workflow: &ResolvedWorkflow) -> Result<WorkflowRunState> {
        let saved = self.read_saved()?;
        self.validate_execution_identity(&saved)?;
        saved.state.validate(workflow)?;
        self.resume_metrics(saved.metrics.as_ref())?;
        Ok(saved.state)
    }

    fn resume_metrics(&mut self, committed: Option<&CommittedMetricLog>) -> Result<()> {
        let current = std::mem::replace(&mut self.metrics, RuntimeMetrics::Disabled);
        self.metrics = match (current, committed) {
            (RuntimeMetrics::Disabled, None) => RuntimeMetrics::Disabled,
            (RuntimeMetrics::ResumePending { path, run_id }, Some(committed)) => {
                ensure!(
                    run_id == committed.run_id,
                    "workflow-runtime metric run_id `{}` differs from requested `{run_id}`",
                    committed.run_id
                );
                let writer = MetricWriter::resume_exact_prefix(
                    &path,
                    &run_id,
                    committed.records,
                    committed.last_global_step,
                )?;
                validate_committed_metric_state(writer.state(), committed)?;
                RuntimeMetrics::Active(writer)
            }
            (RuntimeMetrics::Disabled, Some(_)) => {
                anyhow::bail!("workflow-runtime checkpoint requires --metrics and --run-id")
            }
            (RuntimeMetrics::ResumePending { .. }, None) => {
                anyhow::bail!("workflow-runtime checkpoint was created without a metric journal")
            }
            (RuntimeMetrics::Active(_), _) => {
                anyhow::bail!("cannot load a runtime checkpoint with new-run metric state")
            }
        };
        Ok(())
    }

    fn committed_metrics(&mut self) -> Result<Option<CommittedMetricLog>> {
        match &mut self.metrics {
            RuntimeMetrics::Disabled => Ok(None),
            RuntimeMetrics::Active(writer) => {
                // A runtime checkpoint must never refer to data that was only
                // buffered in userspace or the page cache.
                writer.sync_all()?;
                let state = writer.state();
                Ok(Some(CommittedMetricLog {
                    run_id: writer.run_id().to_owned(),
                    records: state.records,
                    last_global_step: state.last_global_step,
                }))
            }
            RuntimeMetrics::ResumePending { .. } => {
                anyhow::bail!("metric journal resume has not loaded its committed position")
            }
        }
    }

    fn publish(
        &mut self,
        boundary: Option<RuntimeBoundary>,
        state: &WorkflowRunState,
    ) -> Result<()> {
        match fs::symlink_metadata(&self.path) {
            Ok(_) => ensure_regular_non_symlink(&self.path, "workflow-runtime checkpoint")?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "inspect workflow-runtime checkpoint {}",
                        self.path.display()
                    )
                });
            }
        }
        let metrics = self.committed_metrics()?;
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .context("workflow-runtime checkpoint file name is not UTF-8")?;
        let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
        let saved = RuntimeCheckpointFile {
            version: RUNTIME_CHECKPOINT_FILE_VERSION,
            executor_sha256: self.executor_sha256.clone(),
            metrics,
            boundary,
            state: state.clone(),
        };
        let publication = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .with_context(|| format!("failed to create {}", temporary.display()))?;
            serde_json::to_writer_pretty(&mut file, &saved)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            fs::rename(&temporary, &self.path).with_context(|| {
                format!(
                    "failed to publish workflow runtime state {}",
                    self.path.display()
                )
            })?;
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if publication.is_err() && temporary.exists() {
            let _ = fs::remove_file(&temporary);
        }
        publication
    }
}

impl RuntimeCheckpoint for AtomicRuntimeCheckpoint {
    fn persist(&mut self, boundary: &RuntimeBoundary, state: &WorkflowRunState) -> Result<()> {
        self.publish(Some(boundary.clone()), state)
    }

    fn append_metric(&mut self, context: MetricContext, event: MetricEvent) -> Result<()> {
        match &mut self.metrics {
            RuntimeMetrics::Active(writer) => {
                writer.append(context, event)?;
                Ok(())
            }
            RuntimeMetrics::Disabled => {
                anyhow::bail!("phase worker emitted a metric but --metrics was not configured")
            }
            RuntimeMetrics::ResumePending { .. } => {
                anyhow::bail!("metric journal resume is not initialized")
            }
        }
    }
}

fn validate_committed_metric_state(
    state: &MetricLogState,
    committed: &CommittedMetricLog,
) -> Result<()> {
    ensure!(
        state.records == committed.records,
        "resumed metric record count differs from runtime checkpoint"
    );
    ensure!(
        state.last_global_step == committed.last_global_step,
        "resumed metric global step differs from runtime checkpoint"
    );
    if committed.records == 0 {
        ensure!(
            state.run_id.is_none(),
            "empty committed metric prefix unexpectedly has a run id"
        );
    } else {
        ensure!(
            state.run_id.as_deref() == Some(committed.run_id.as_str()),
            "resumed metric run id differs from runtime checkpoint"
        );
    }
    Ok(())
}

fn ensure_regular_non_symlink(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("{label} {} is unavailable", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "{label} {} must be a regular non-symlink file",
        path.display()
    );
    Ok(())
}

fn ensure_path_absent(path: &Path, label: &str) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => anyhow::bail!("{label} {} already exists", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {label} {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MetricEvent, PhaseBoundary, PhaseTimingMetrics, validate_metric_log};
    use crate::runtime::{ExecutorRegistry, RuntimeStatus, run_until_yield_or_complete};
    use crate::workflow::{ResolvedWorkflow, WorkflowV2, load_workflow};

    struct NoopProgress;

    impl PhaseProgressSink for NoopProgress {
        fn checkpoint(&mut self, _: Value) -> Result<()> {
            Ok(())
        }

        fn metric(&mut self, _: MetricContext, _: MetricEvent) -> Result<()> {
            Ok(())
        }
    }

    /// Stand-in dispatch identity for runtime checkpoints under test.
    fn dispatch_identity() -> String {
        format!("sha256:{:064x}", 1)
    }

    fn one_phase_workflow(directory: &Path) -> ResolvedWorkflow {
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "pretrain",
                "type": "pretrain",
                "task": {"type": "causal_lm"},
                "data": "data.jsonl",
                "sequence_length": 8,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "steps": 1
            }]
        }))
        .unwrap();
        workflow.resolve(&directory.join("workflow.json")).unwrap()
    }

    fn phase_request(workflow: &ResolvedWorkflow) -> PhaseExecutionRequest {
        PhaseExecutionRequest {
            phase_index: 0,
            phase: workflow.phases[0].clone(),
            input_checkpoint: None,
            resume_state: None,
        }
    }

    #[cfg(unix)]
    fn write_executable(path: &Path, contents: &str) {
        use std::os::unix::fs::PermissionsExt;

        fs::write(path, contents).unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions).unwrap();
    }

    #[cfg(unix)]
    fn python_with_setsid() -> PathBuf {
        [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
        ]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
        .expect("setsid worker tests require an absolute Python 3 interpreter")
    }

    #[cfg(unix)]
    fn kill_detached_pid(path: &Path) {
        for _ in 0..100 {
            if let Ok(value) = fs::read_to_string(path)
                && let Ok(pid) = value.trim().parse::<i32>()
            {
                // SAFETY: the test helper wrote its own positive process ID.
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("detached worker did not publish its pid");
    }

    fn timing_event() -> MetricEvent {
        MetricEvent::PhaseTiming(PhaseTimingMetrics {
            boundary: PhaseBoundary::Progress,
            elapsed_seconds: 1.0,
            input_wait_seconds: 0.1,
            forward_seconds: 0.4,
            backward_seconds: 0.3,
            optimizer_seconds: 0.1,
            checkpoint_seconds: 0.1,
        })
    }

    fn metric_message(step: u64) -> String {
        serde_json::to_string(&PhaseWorkerMessage::Metric {
            global_step: step,
            checkpoint_hash: Some(format!("sha256:{:064x}", step + 100)),
            event: timing_event(),
        })
        .unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn jsonl_worker_emits_metrics_across_progress_and_complete() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let workflow_path = directory.path().join("workflow.json");
        fs::write(
            &workflow_path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "pretrain",
                    "type": "pretrain",
                    "task": {"type": "causal_lm"},
                    "data": "data.jsonl",
                    "sequence_length": 8,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 1
                }]
            }"#,
        )
        .unwrap();
        let workflow = load_workflow(&workflow_path).unwrap();
        let worker_path = directory.path().join("worker.sh");
        let script = [
            "#!/bin/sh\nIFS= read -r request\ntest -n \"$request\"\n".to_owned(),
            format!("echo '{}'\n", metric_message(1)),
            "echo '{\"type\":\"progress\",\"resume_state\":{\"step\":1}}'\n".to_owned(),
            format!("echo '{}'\n", metric_message(2)),
            "echo '{\"type\":\"complete\",\"product\":{\"type\":\"model_candidate\",\"checkpoint\":{\"uri\":\"checkpoint://candidate\",\"sha256\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}}}'\n".to_owned(),
        ]
        .concat();
        fs::write(&worker_path, script).unwrap();
        let mut permissions = fs::metadata(&worker_path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&worker_path, permissions).unwrap();
        let worker = ExternalPhaseExecutor::new(&worker_path, Vec::new()).unwrap();
        let state_path = directory.path().join("runtime.json");
        let metrics_path = directory.path().join("metrics.jsonl");
        let mut sink = AtomicRuntimeCheckpoint::new(&state_path, dispatch_identity()).unwrap();
        sink.configure_metrics(&metrics_path, "worker-run", false)
            .unwrap();
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        sink.initialize(&state).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(crate::workflow::PhaseKind::Pretrain, worker)
            .unwrap();
        let status =
            run_until_yield_or_complete(&workflow, &mut state, &mut registry, &mut (), &mut sink)
                .unwrap();
        assert_eq!(status, RuntimeStatus::AlreadyComplete);
        drop(sink);
        let mut reader = AtomicRuntimeCheckpoint::new(&state_path, dispatch_identity()).unwrap();
        reader
            .configure_metrics(&metrics_path, "worker-run", true)
            .unwrap();
        let resumed = reader.load(&workflow).unwrap();
        assert!(resumed.is_complete(&workflow));
        assert_eq!(resumed.completed_phases().len(), 1);
        drop(reader);
        let metric_state = validate_metric_log(&metrics_path, Some("worker-run")).unwrap();
        assert_eq!(metric_state.records, 2);
        assert_eq!(metric_state.last_global_step, Some(2));
        let records = fs::read_to_string(&metrics_path).unwrap();
        for line in records.lines() {
            let record: crate::metrics::MetricRecord = serde_json::from_str(line).unwrap();
            assert_eq!(record.phase.index, 0);
            assert_eq!(record.phase.name, "pretrain");
            assert_eq!(record.phase.kind, MetricPhaseKind::Pretrain);
        }
    }

    #[cfg(unix)]
    #[test]
    fn yielded_metric_tail_is_truncated_to_the_runtime_commit_on_resume() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let workflow_path = directory.path().join("workflow.json");
        fs::write(
            &workflow_path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "pretrain",
                    "type": "pretrain",
                    "task": {"type": "causal_lm"},
                    "data": "data.jsonl",
                    "sequence_length": 8,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 1
                }]
            }"#,
        )
        .unwrap();
        let workflow = load_workflow(&workflow_path).unwrap();
        let worker_path = directory.path().join("worker.sh");
        let script = [
            "#!/bin/sh\nIFS= read -r request\ntest -n \"$request\"\n".to_owned(),
            format!("echo '{}'\n", metric_message(1)),
            "echo '{\"type\":\"yielded\",\"resume_state\":{\"step\":1}}'\n".to_owned(),
        ]
        .concat();
        fs::write(&worker_path, script).unwrap();
        let mut permissions = fs::metadata(&worker_path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&worker_path, permissions).unwrap();
        let state_path = directory.path().join("runtime.json");
        let metrics_path = directory.path().join("metrics.jsonl");
        {
            let mut sink = AtomicRuntimeCheckpoint::new(&state_path, dispatch_identity()).unwrap();
            sink.configure_metrics(&metrics_path, "yield-run", false)
                .unwrap();
            let mut state = WorkflowRunState::new(&workflow, None).unwrap();
            sink.initialize(&state).unwrap();
            let mut registry = ExecutorRegistry::new();
            registry
                .register(
                    crate::workflow::PhaseKind::Pretrain,
                    ExternalPhaseExecutor::new(&worker_path, Vec::new()).unwrap(),
                )
                .unwrap();
            let status = run_until_yield_or_complete(
                &workflow,
                &mut state,
                &mut registry,
                &mut (),
                &mut sink,
            )
            .unwrap();
            assert_eq!(
                status,
                RuntimeStatus::Yielded {
                    phase_index: 0,
                    phase_name: "pretrain".to_owned(),
                }
            );
        }

        // Simulate a metric that reached the journal after the last durable
        // runtime boundary. It must disappear when that boundary is resumed.
        {
            let mut writer = MetricWriter::resume(&metrics_path, "yield-run").unwrap();
            writer
                .append_at(
                    MetricContext {
                        global_step: 2,
                        phase: MetricPhase {
                            index: 0,
                            name: "pretrain".to_owned(),
                            kind: MetricPhaseKind::Pretrain,
                        },
                        checkpoint_hash: None,
                    },
                    timing_event(),
                    u64::MAX - 1,
                )
                .unwrap();
            writer.sync_all().unwrap();
        }
        assert_eq!(
            validate_metric_log(&metrics_path, Some("yield-run"))
                .unwrap()
                .records,
            2
        );

        let mut resumed = AtomicRuntimeCheckpoint::new(&state_path, dispatch_identity()).unwrap();
        resumed
            .configure_metrics(&metrics_path, "yield-run", true)
            .unwrap();
        let state = resumed.load(&workflow).unwrap();
        assert_eq!(
            state.active_phase().unwrap().resume_state(),
            Some(&serde_json::json!({"step": 1}))
        );
        assert_eq!(
            validate_metric_log(&metrics_path, Some("yield-run"))
                .unwrap()
                .records,
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn invalid_worker_metric_is_rejected_before_it_reaches_the_journal() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let workflow_path = directory.path().join("workflow.json");
        fs::write(
            &workflow_path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "pretrain",
                    "type": "pretrain",
                    "task": {"type": "causal_lm"},
                    "data": "data.jsonl",
                    "sequence_length": 8,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 1
                }]
            }"#,
        )
        .unwrap();
        let workflow = load_workflow(&workflow_path).unwrap();
        let worker_path = directory.path().join("worker.sh");
        fs::write(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "echo '{\"type\":\"metric\",\"global_step\":1,\"event\":{\"type\":\"phase_timing\",\"values\":{\"boundary\":\"progress\",\"elapsed_seconds\":-1.0,\"input_wait_seconds\":0.0,\"forward_seconds\":0.0,\"backward_seconds\":0.0,\"optimizer_seconds\":0.0,\"checkpoint_seconds\":0.0}}}'\n",
                "echo '{\"type\":\"yielded\",\"resume_state\":{\"step\":1}}'\n"
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&worker_path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&worker_path, permissions).unwrap();
        let state_path = directory.path().join("runtime.json");
        let metrics_path = directory.path().join("metrics.jsonl");
        let mut sink = AtomicRuntimeCheckpoint::new(&state_path, dispatch_identity()).unwrap();
        sink.configure_metrics(&metrics_path, "invalid-run", false)
            .unwrap();
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        sink.initialize(&state).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(
                crate::workflow::PhaseKind::Pretrain,
                ExternalPhaseExecutor::new(&worker_path, Vec::new()).unwrap(),
            )
            .unwrap();
        let error =
            run_until_yield_or_complete(&workflow, &mut state, &mut registry, &mut (), &mut sink)
                .unwrap_err()
                .to_string();
        assert!(error.contains("invalid typed metric"), "{error}");
        drop(sink);
        assert_eq!(
            validate_metric_log(&metrics_path, Some("invalid-run"))
                .unwrap()
                .records,
            0
        );
    }

    #[cfg(unix)]
    #[test]
    fn protocol_failure_terminates_the_worker_process_group() {
        use std::os::unix::fs::PermissionsExt;
        use std::time::Duration;

        let directory = tempfile::tempdir().unwrap();
        let workflow_path = directory.path().join("workflow.json");
        fs::write(
            &workflow_path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "pretrain",
                    "type": "pretrain",
                    "task": {"type": "causal_lm"},
                    "data": "data.jsonl",
                    "sequence_length": 8,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 1
                }]
            }"#,
        )
        .unwrap();
        let workflow = load_workflow(&workflow_path).unwrap();
        let worker_path = directory.path().join("worker.sh");
        fs::write(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "echo '{not-json'\n",
                "sleep 1\n",
                "echo leaked > \"$1\"\n"
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&worker_path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&worker_path, permissions).unwrap();
        let leaked_side_effect = directory.path().join("leaked-side-effect");
        let mut registry = ExecutorRegistry::new();
        registry
            .register(
                crate::workflow::PhaseKind::Pretrain,
                ExternalPhaseExecutor::new(
                    &worker_path,
                    vec![leaked_side_effect.clone().into_os_string()],
                )
                .unwrap(),
            )
            .unwrap();
        let state_path = directory.path().join("runtime.json");
        let mut sink = AtomicRuntimeCheckpoint::new(&state_path, dispatch_identity()).unwrap();
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        sink.initialize(&state).unwrap();

        let error =
            run_until_yield_or_complete(&workflow, &mut state, &mut registry, &mut (), &mut sink)
                .unwrap_err()
                .to_string();
        assert!(error.contains("invalid protocol JSON"), "{error}");
        std::thread::sleep(Duration::from_millis(1_200));
        assert!(
            !leaked_side_effect.exists(),
            "failed worker continued mutating state after its protocol error"
        );
    }

    #[cfg(unix)]
    #[test]
    fn wall_timeout_terminates_worker_and_descendants() {
        let directory = tempfile::tempdir().unwrap();
        let workflow = one_phase_workflow(directory.path());
        let worker_path = directory.path().join("worker.sh");
        let leaked_side_effect = directory.path().join("leaked-side-effect");
        write_executable(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "(sleep 1; echo leaked > \"$1\") &\n",
                "sleep 30\n"
            ),
        );
        let mut executor = ExternalPhaseExecutor::new(
            &worker_path,
            vec![leaked_side_effect.clone().into_os_string()],
        )
        .unwrap()
        .with_timeout(Duration::from_millis(150))
        .unwrap();
        let start = Instant::now();
        let error = executor
            .execute(&phase_request(&workflow), &mut (), &mut NoopProgress)
            .unwrap_err()
            .to_string();
        assert!(error.contains("wall timeout"), "{error}");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "phase timeout did not return promptly"
        );
        thread::sleep(Duration::from_millis(1_100));
        assert!(
            !leaked_side_effect.exists(),
            "timed-out worker descendant remained able to mutate state"
        );
    }

    #[cfg(unix)]
    #[test]
    fn wall_timeout_is_bounded_when_setsid_descendant_retains_protocol_pipes() {
        let directory = tempfile::tempdir().unwrap();
        let workflow = one_phase_workflow(directory.path());
        let worker_path = directory.path().join("setsid-timeout.sh");
        let detached_pid = directory.path().join("detached.pid");
        write_executable(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                "/bin/sleep 30\n"
            ),
        );
        let mut executor = ExternalPhaseExecutor::new(
            &worker_path,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
        )
        .unwrap()
        .with_timeout(Duration::from_secs(3))
        .unwrap();

        let started = Instant::now();
        let result = executor.execute(&phase_request(&workflow), &mut (), &mut NoopProgress);
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("wall timeout"), "{error}");
        assert!(
            elapsed < Duration::from_secs(5),
            "setsid descendant delayed phase timeout for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn protocol_error_is_bounded_when_setsid_descendant_retains_stdout() {
        let directory = tempfile::tempdir().unwrap();
        let workflow = one_phase_workflow(directory.path());
        let worker_path = directory.path().join("setsid-error.sh");
        let detached_pid = directory.path().join("detached.pid");
        write_executable(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                "printf '{not-json}\\n'\n",
                "/bin/sleep 30\n"
            ),
        );
        let mut executor = ExternalPhaseExecutor::new(
            &worker_path,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
        )
        .unwrap()
        .with_timeout(Duration::from_secs(3))
        .unwrap();

        let started = Instant::now();
        let result = executor.execute(&phase_request(&workflow), &mut (), &mut NoopProgress);
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("invalid protocol JSON"), "{error}");
        assert!(
            elapsed < Duration::from_secs(2),
            "setsid descendant delayed phase protocol error for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn exited_worker_does_not_wait_for_descendant_holding_stdout() {
        let directory = tempfile::tempdir().unwrap();
        let workflow = one_phase_workflow(directory.path());
        let worker_path = directory.path().join("worker.sh");
        write_executable(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "sleep 30 &\n",
                "echo '{\"type\":\"complete\",\"product\":{\"type\":\"model_candidate\",\"checkpoint\":{\"uri\":\"checkpoint://candidate\",\"sha256\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}}}'\n"
            ),
        );
        let mut executor = ExternalPhaseExecutor::new(&worker_path, Vec::new())
            .unwrap()
            .with_timeout(Duration::from_secs(5))
            .unwrap();
        let start = Instant::now();
        let result = executor
            .execute(&phase_request(&workflow), &mut (), &mut NoopProgress)
            .unwrap();
        assert!(
            matches!(result, PhaseExecutionResult::Complete(_)),
            "worker did not return its terminal product"
        );
        assert!(
            start.elapsed() < Duration::from_secs(4),
            "descendant-held stdout delayed worker completion"
        );
    }

    #[cfg(unix)]
    #[test]
    fn leader_exit_is_bounded_when_setsid_descendant_retains_stdout() {
        let directory = tempfile::tempdir().unwrap();
        let workflow = one_phase_workflow(directory.path());
        let worker_path = directory.path().join("setsid-exit.sh");
        let detached_pid = directory.path().join("detached.pid");
        write_executable(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                "echo '{\"type\":\"complete\",\"product\":{\"type\":\"model_candidate\",\"checkpoint\":{\"uri\":\"checkpoint://candidate\",\"sha256\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}}}'\n"
            ),
        );
        let mut executor = ExternalPhaseExecutor::new(
            &worker_path,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
        )
        .unwrap()
        .with_timeout(Duration::from_secs(2))
        .unwrap();

        let started = Instant::now();
        let result = executor.execute(&phase_request(&workflow), &mut (), &mut NoopProgress);
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        assert!(matches!(result.unwrap(), PhaseExecutionResult::Complete(_)));
        assert!(
            elapsed < Duration::from_secs(3),
            "setsid descendant delayed phase leader exit for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn execution_identity_binds_arguments_and_timeout() {
        let directory = tempfile::tempdir().unwrap();
        let worker_path = directory.path().join("worker.sh");
        write_executable(&worker_path, "#!/bin/sh\nexit 0\n");
        let baseline =
            ExternalPhaseExecutor::new(&worker_path, vec![OsString::from("argument")]).unwrap();
        let same =
            ExternalPhaseExecutor::new(&worker_path, vec![OsString::from("argument")]).unwrap();
        let changed_argument =
            ExternalPhaseExecutor::new(&worker_path, vec![OsString::from("different")]).unwrap();
        let changed_timeout = same.clone().with_timeout(Duration::from_secs(60)).unwrap();

        assert_eq!(baseline.execution_identity(), same.execution_identity());
        assert_ne!(
            baseline.execution_identity(),
            changed_argument.execution_identity()
        );
        assert_ne!(
            baseline.execution_identity(),
            changed_timeout.execution_identity()
        );
    }

    #[cfg(unix)]
    #[test]
    fn worker_runs_with_empty_environment_and_root_working_directory() {
        let directory = tempfile::tempdir().unwrap();
        let workflow = one_phase_workflow(directory.path());
        let worker_path = directory.path().join("worker.sh");
        write_executable(
            &worker_path,
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "test \"$PWD\" = /\n",
                "test -z \"${HOME+x}\"\n",
                "echo '{\"type\":\"complete\",\"product\":{\"type\":\"model_candidate\",\"checkpoint\":{\"uri\":\"checkpoint://candidate\",\"sha256\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}}}'\n"
            ),
        );
        let mut executor = ExternalPhaseExecutor::new(&worker_path, Vec::new())
            .unwrap()
            .with_timeout(Duration::from_secs(2))
            .unwrap();
        assert!(
            executor
                .execute(&phase_request(&workflow), &mut (), &mut NoopProgress)
                .is_ok()
        );
    }

    #[cfg(unix)]
    #[test]
    fn phase_worker_arguments_are_canonical_and_bounded() {
        use std::os::unix::ffi::OsStringExt;

        let directory = tempfile::tempdir().unwrap();
        let worker_path = directory.path().join("worker.sh");
        write_executable(&worker_path, "#!/bin/sh\nexit 0\n");

        let too_many = vec![OsString::from("x"); MAX_PHASE_WORKER_ARGUMENTS + 1];
        let error = ExternalPhaseExecutor::new(&worker_path, too_many)
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than"), "{error}");

        let oversized = OsString::from("x".repeat(MAX_PHASE_WORKER_ARGUMENT_BYTES + 1));
        let error = ExternalPhaseExecutor::new(&worker_path, vec![oversized])
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds"), "{error}");

        let nul = OsString::from_vec(b"a\0b".to_vec());
        let error = ExternalPhaseExecutor::new(&worker_path, vec![nul])
            .unwrap_err()
            .to_string();
        assert!(error.contains("NUL"), "{error}");
    }

    #[test]
    fn checkpoint_rejects_a_different_worker_identity() {
        let directory = tempfile::tempdir().unwrap();
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "promotion",
                "type": "promotion",
                "promotion": crate::workflow::test_promotion_config()
            }]
        }))
        .unwrap();
        let workflow = workflow
            .resolve(&directory.path().join("workflow.json"))
            .unwrap();
        let first = format!("sha256:{:064x}", 1);
        let second = format!("sha256:{:064x}", 2);
        let path = directory.path().join("runtime.json");
        let mut writer = AtomicRuntimeCheckpoint::new(&path, &first).unwrap();
        let initial = ImmutableModelCheckpoint::new(
            "checkpoint://promotion-input",
            format!("sha256:{:064x}", 3),
        )
        .unwrap();
        writer
            .initialize(&WorkflowRunState::new(&workflow, Some(initial)).unwrap())
            .unwrap();
        let mut reader = AtomicRuntimeCheckpoint::new(&path, &second).unwrap();
        assert!(reader.load(&workflow).is_err());
    }

    #[test]
    fn runtime_checkpoint_read_is_bounded_before_json_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("runtime.json");
        File::create(&path)
            .unwrap()
            .set_len(MAX_RUNTIME_CHECKPOINT_BYTES + 1)
            .unwrap();
        let checkpoint = AtomicRuntimeCheckpoint::new(&path, format!("sha256:{:064x}", 1)).unwrap();
        let error = format!("{:#}", checkpoint.verify_execution_identity().unwrap_err());
        assert!(error.contains("byte limit"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn metric_journal_refuses_symlink_targets() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.jsonl");
        fs::write(&target, b"").unwrap();
        let link = directory.path().join("metrics.jsonl");
        symlink(&target, &link).unwrap();
        let hash = format!("sha256:{:064x}", 1);
        let mut checkpoint =
            AtomicRuntimeCheckpoint::new(directory.path().join("runtime.json"), hash).unwrap();
        let error = checkpoint
            .configure_metrics(&link, "run", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("regular non-symlink file"), "{error}");
    }
}
