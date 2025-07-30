//! Versioned subprocess bridge for WorkflowV2 phase executors.
//!
//! The core runtime stays algorithm-neutral while a pinned worker executable
//! can implement any phase/task combination. Requests and responses are JSONL;
//! progress records are durably checkpointed as they arrive, and terminal
//! products still pass the runtime's immutable-candidate rules.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::metrics::{
    MetricContext, MetricEvent, MetricLogState, MetricPhase, MetricPhaseKind, MetricWriter,
};
use crate::runtime::{
    ImmutableModelCheckpoint, PhaseExecutionRequest, PhaseExecutionResult, PhaseExecutor,
    PhaseProduct, PhaseProgressSink, RuntimeBoundary, RuntimeCheckpoint, WorkflowRunState,
};
use crate::workflow::ResolvedWorkflow;

pub const PHASE_WORKER_PROTOCOL_VERSION: u32 = 2;
pub const RUNTIME_CHECKPOINT_FILE_VERSION: u32 = 2;
const MAX_WORKER_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseWorkerRequest {
    pub version: u32,
    pub executor_sha256: String,
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

/// A content-pinned local phase worker. No shell is involved: the executable
/// and each argument are passed directly to `Command`.
#[derive(Clone, Debug)]
pub struct ExternalPhaseExecutor {
    executable: PathBuf,
    arguments: Vec<OsString>,
    expected_sha256: String,
}

impl ExternalPhaseExecutor {
    pub fn new(
        executable: impl Into<PathBuf>,
        arguments: Vec<OsString>,
        expected_sha256: impl Into<String>,
    ) -> Result<Self> {
        let executor = Self {
            executable: executable.into(),
            arguments,
            expected_sha256: expected_sha256.into(),
        };
        executor.verify_identity()?;
        Ok(executor)
    }

    pub fn expected_sha256(&self) -> &str {
        &self.expected_sha256
    }

    pub fn verify_identity(&self) -> Result<()> {
        validate_sha256(&self.expected_sha256)?;
        let metadata = fs::symlink_metadata(&self.executable).with_context(|| {
            format!("phase worker {} is unavailable", self.executable.display())
        })?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "phase worker {} must be a regular non-symlink file",
            self.executable.display()
        );
        ensure!(
            file_sha256(&self.executable)? == self.expected_sha256,
            "phase worker {} does not match its pinned SHA-256",
            self.executable.display()
        );
        Ok(())
    }
}

impl<C> PhaseExecutor<C> for ExternalPhaseExecutor {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        _: &mut C,
        progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        // Re-hash immediately before every spawn so a worker cannot change
        // between workflow initialization and a later resumed phase.
        self.verify_identity()?;
        let mut child = Command::new(&self.executable)
            .args(&self.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!("failed to start phase worker {}", self.executable.display())
            })?;
        let wire_request = PhaseWorkerRequest {
            version: PHASE_WORKER_PROTOCOL_VERSION,
            executor_sha256: self.expected_sha256.clone(),
            phase_index: request.phase_index,
            phase: request.phase.clone(),
            input_checkpoint: request.input_checkpoint.clone(),
            resume_state: request.resume_state.clone(),
        };
        {
            let mut stdin = child
                .stdin
                .take()
                .context("phase worker stdin is unavailable")?;
            serde_json::to_writer(&mut stdin, &wire_request)?;
            stdin.write_all(b"\n")?;
            stdin.flush()?;
        }

        let stdout = child
            .stdout
            .take()
            .context("phase worker stdout is unavailable")?;
        let mut reader = BufReader::new(stdout);
        let mut terminal = None;
        while let Some(line) = read_worker_message(&mut reader)? {
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            ensure!(
                terminal.is_none(),
                "phase worker emitted data after a terminal message"
            );
            let message: PhaseWorkerMessage = serde_json::from_slice(&line)
                .context("phase worker emitted invalid protocol JSON")?;
            match message {
                PhaseWorkerMessage::Metric {
                    global_step,
                    checkpoint_hash,
                    event,
                } => {
                    if let Some(hash) = &checkpoint_hash {
                        validate_sha256(hash).context(
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
                    terminal = Some(PhaseExecutionResult::Yielded { resume_state });
                }
                PhaseWorkerMessage::Complete { product } => {
                    terminal = Some(PhaseExecutionResult::Complete(product));
                }
            }
        }
        let status = child.wait()?;
        ensure!(status.success(), "phase worker exited with status {status}");
        terminal.context("phase worker exited without a terminal message")
    }
}

fn metric_phase(phase: &crate::workflow::PhaseV2, phase_index: usize) -> Result<MetricPhase> {
    let index = u32::try_from(phase_index).context("workflow phase index exceeds u32")?;
    let kind = match phase.kind {
        crate::workflow::PhaseKind::Pretrain => MetricPhaseKind::Pretrain,
        crate::workflow::PhaseKind::ContinuedPretrain => MetricPhaseKind::ContinuedPretrain,
        crate::workflow::PhaseKind::Sft => MetricPhaseKind::Sft,
        crate::workflow::PhaseKind::Preference => MetricPhaseKind::Preference,
        crate::workflow::PhaseKind::Rl => MetricPhaseKind::Rl,
        crate::workflow::PhaseKind::Distillation => MetricPhaseKind::Distillation,
        crate::workflow::PhaseKind::Sleep => MetricPhaseKind::Sleep,
        crate::workflow::PhaseKind::Quantization => MetricPhaseKind::Quantization,
        crate::workflow::PhaseKind::Evaluation => MetricPhaseKind::Evaluation,
        crate::workflow::PhaseKind::Promotion => MetricPhaseKind::Promotion,
    };
    Ok(MetricPhase {
        index,
        name: phase.name.clone(),
        kind,
    })
}

fn read_worker_message(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            ensure!(
                line.is_empty(),
                "phase worker emitted an unterminated JSONL message"
            );
            return Ok(None);
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        ensure!(
            line.len() + take <= MAX_WORKER_MESSAGE_BYTES,
            "phase worker emitted a JSONL message larger than {MAX_WORKER_MESSAGE_BYTES} bytes"
        );
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.ends_with(b"\n") {
            return Ok(Some(line));
        }
    }
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
        validate_sha256(&checkpoint.executor_sha256)?;
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

    pub fn load(&mut self, workflow: &ResolvedWorkflow) -> Result<WorkflowRunState> {
        ensure_regular_non_symlink(&self.path, "workflow-runtime checkpoint")?;
        let bytes = fs::read(&self.path).with_context(|| {
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
        ensure!(
            saved.version == RUNTIME_CHECKPOINT_FILE_VERSION,
            "unsupported workflow-runtime checkpoint-file version {}",
            saved.version
        );
        ensure!(
            saved.executor_sha256 == self.executor_sha256,
            "workflow-runtime checkpoint was produced by a different phase worker"
        );
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

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut input = BufReader::new(
        File::open(path).with_context(|| format!("failed to hash {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        use std::io::Read as _;
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn validate_sha256(value: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .context("SHA-256 identity must start with `sha256:`")?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "SHA-256 identity must contain 64 lowercase hexadecimal digits"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::metrics::{MetricEvent, PhaseBoundary, PhaseTimingMetrics, validate_metric_log};
    use crate::runtime::{ExecutorRegistry, RuntimeStatus, run_until_yield_or_complete};
    use crate::workflow::{WorkflowV2, load_workflow};

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
    fn pinned_jsonl_worker_emits_metrics_across_progress_and_complete() {
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
        let worker_hash = file_sha256(&worker_path).unwrap();
        let worker = ExternalPhaseExecutor::new(&worker_path, Vec::new(), &worker_hash).unwrap();
        let state_path = directory.path().join("runtime.json");
        let metrics_path = directory.path().join("metrics.jsonl");
        let mut sink = AtomicRuntimeCheckpoint::new(&state_path, &worker_hash).unwrap();
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
        let mut reader = AtomicRuntimeCheckpoint::new(&state_path, &worker_hash).unwrap();
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
        let worker_hash = file_sha256(&worker_path).unwrap();
        let state_path = directory.path().join("runtime.json");
        let metrics_path = directory.path().join("metrics.jsonl");
        {
            let mut sink = AtomicRuntimeCheckpoint::new(&state_path, &worker_hash).unwrap();
            sink.configure_metrics(&metrics_path, "yield-run", false)
                .unwrap();
            let mut state = WorkflowRunState::new(&workflow, None).unwrap();
            sink.initialize(&state).unwrap();
            let mut registry = ExecutorRegistry::new();
            registry
                .register(
                    crate::workflow::PhaseKind::Pretrain,
                    ExternalPhaseExecutor::new(&worker_path, Vec::new(), &worker_hash).unwrap(),
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

        let mut resumed = AtomicRuntimeCheckpoint::new(&state_path, &worker_hash).unwrap();
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
        let worker_hash = file_sha256(&worker_path).unwrap();
        let state_path = directory.path().join("runtime.json");
        let metrics_path = directory.path().join("metrics.jsonl");
        let mut sink = AtomicRuntimeCheckpoint::new(&state_path, &worker_hash).unwrap();
        sink.configure_metrics(&metrics_path, "invalid-run", false)
            .unwrap();
        let mut state = WorkflowRunState::new(&workflow, None).unwrap();
        sink.initialize(&state).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(
                crate::workflow::PhaseKind::Pretrain,
                ExternalPhaseExecutor::new(&worker_path, Vec::new(), &worker_hash).unwrap(),
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

    #[test]
    fn checkpoint_rejects_a_different_worker_identity() {
        let directory = tempfile::tempdir().unwrap();
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "phases": [{
                "name": "promotion",
                "type": "promotion"
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
        writer
            .initialize(&WorkflowRunState::new(&workflow, None).unwrap())
            .unwrap();
        let mut reader = AtomicRuntimeCheckpoint::new(&path, &second).unwrap();
        assert!(reader.load(&workflow).is_err());
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

    #[test]
    fn worker_protocol_rejects_unterminated_and_oversized_messages_without_unbounded_reads() {
        let mut unterminated = Cursor::new(b"{}".to_vec());
        assert!(read_worker_message(&mut unterminated).is_err());

        let mut oversized = Cursor::new(vec![b'x'; MAX_WORKER_MESSAGE_BYTES + 1]);
        assert!(read_worker_message(&mut oversized).is_err());

        let mut valid = Cursor::new(b"{}\n".to_vec());
        assert_eq!(
            read_worker_message(&mut valid).unwrap(),
            Some(b"{}\n".to_vec())
        );
        assert_eq!(read_worker_message(&mut valid).unwrap(), None);
    }
}
