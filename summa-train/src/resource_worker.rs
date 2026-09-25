//! Strict execution bridge for promotion resource evidence.
//!
//! The worker receives model transports and writes raw observations back over
//! a versioned JSONL protocol. The host owns process lifetime, the execution
//! timeout, and publication of the resulting resource comparison.

use std::ffi::OsString;
use std::fs;
#[cfg(test)]
use std::io::Read;
#[cfg(test)]
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::artifact_io::atomic_write_new;

use crate::acceptance::{
    AcceptancePolicy, CapacityObservation, ExactResumeEvidence, KernelParityEvidence,
    KernelParitySample, PairedWakeTrial, RESOURCE_COMPARISON_VERSION,
    RESOURCE_EXECUTION_PROTOCOL_VERSION, ResourceComparison,
};
use crate::benchmark::{
    BenchmarkTarget, LoadedBenchmarkRun, verified_resource_benchmark_context, verify_exact_resume,
};
#[cfg(unix)]
use crate::protocol_process::{ProtocolRead, SupervisedProcess};
const MAX_RESOURCE_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_RESOURCE_EVALUATOR_TIMEOUT: Duration = Duration::from_secs(3_600);
const RESOURCE_ARTIFACT_DIRECTORY: &str = "artifacts";
const RESOURCE_COMPARISON_FILE: &str = "resource-comparison.json";

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceWorkerRequest {
    pub version: u32,
    /// Capacity- and compute-matched baseline the candidate is measured
    /// against, with paths resolved against its benchmark run.
    pub baseline: BenchmarkTarget,
    pub candidate: BenchmarkTarget,
    /// Strongest baseline derived from the fixed ablation matrix.
    pub strongest_baseline_id: String,
    pub evaluator_arguments: Vec<String>,
    /// Directory the worker writes its exact-resume artifacts into.
    pub artifact_directory: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceWorkerResponse {
    pub version: u32,
    pub wake_trials: Vec<PairedWakeTrial>,
    pub candidate_capacity: Vec<CapacityObservation>,
    pub grouped_mm_samples: Vec<KernelParitySample>,
    pub pytorch_samples: Vec<KernelParitySample>,
    pub exact_resume: ExactResumeEvidence,
}

#[derive(Clone, Debug)]
pub struct ResourceEvidencePublication {
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct ExternalResourceEvaluator {
    executable: PathBuf,
    arguments: Vec<String>,
    timeout: Duration,
}

impl ExternalResourceEvaluator {
    pub fn new(executable: impl Into<PathBuf>, arguments: Vec<OsString>) -> Result<Self> {
        let executable = executable.into();
        let arguments = arguments
            .into_iter()
            .map(|argument| {
                argument
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("resource evaluator argument is not valid UTF-8"))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            arguments.len() <= 64
                && arguments
                    .iter()
                    .all(|argument| argument.len() <= 4096 && !argument.contains('\0')),
            "resource evaluator arguments exceed protocol limits"
        );
        Ok(Self {
            executable: validate_real_path(&executable, "resource evaluator")?,
            arguments,
            timeout: DEFAULT_RESOURCE_EVALUATOR_TIMEOUT,
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        ensure!(
            !timeout.is_zero(),
            "resource evaluator timeout must be positive"
        );
        ensure!(
            Instant::now().checked_add(timeout).is_some(),
            "resource evaluator timeout is too large for the monotonic clock"
        );
        self.timeout = timeout;
        Ok(self)
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }

    /// Re-resolve the configured executable path and require that no ancestor
    /// became a symlink or moved since construction.
    fn validate_path_identity(&self) -> Result<()> {
        let resolved = validate_real_path(&self.executable, "resource evaluator")?;
        ensure!(
            resolved == self.executable,
            "resource evaluator path identity changed"
        );
        Ok(())
    }

    #[cfg(unix)]
    fn execute(&self, request: &ResourceWorkerRequest) -> Result<ResourceWorkerResponse> {
        let mut bytes = serde_json::to_vec(request)?;
        ensure!(
            bytes.len() < MAX_RESOURCE_MESSAGE_BYTES,
            "resource evaluator request exceeds {MAX_RESOURCE_MESSAGE_BYTES} bytes"
        );
        bytes.push(b'\n');
        // Walk every path ancestor immediately before spawning so a symlinked
        // or relocated evaluator path fails loudly instead of being executed.
        self.validate_path_identity()?;
        let mut command = Command::new(&self.executable);
        command
            .args(&self.arguments)
            .env_clear()
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Worker diagnostics are not evidence and must not become an
            // unbounded side channel or captured output.
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = command.spawn().with_context(|| {
            format!(
                "failed to start resource evaluator {}",
                self.executable.display()
            )
        })?;
        let mut child =
            SupervisedProcess::new(child, "resource evaluator", MAX_RESOURCE_MESSAGE_BYTES)?;
        let deadline = Instant::now()
            .checked_add(self.timeout)
            .context("resource evaluator timeout exceeds the monotonic clock")?;
        let mut written = 0_usize;
        let mut response = None;
        loop {
            child.write_available(&bytes, &mut written)?;
            if written == bytes.len() {
                child.close_input();
            }
            drain_resource_output(&mut child, &mut response)?;
            if let Some(status) = child.try_wait()? {
                // Leader exit is a protocol boundary even if a descendant has
                // escaped the worker group with setsid() and retained stdout.
                // Drain every byte already committed by the leader, then close
                // our nonblocking pipes instead of joining an EOF waiter.
                child.terminate_process_group();
                drain_resource_output(&mut child, &mut response)?;
                child.finish_output_at_leader_exit()?;
                ensure!(
                    written == bytes.len(),
                    "resource evaluator exited before consuming its complete request"
                );
                ensure!(
                    status.success(),
                    "resource evaluator exited with status {status}"
                );
                return response.context("resource evaluator exited before responding");
            }
            if Instant::now() >= deadline {
                bail!(
                    "resource evaluator exceeded its {:?} execution timeout",
                    self.timeout
                );
            }
            child.wait_for_activity(written < bytes.len(), deadline)?;
        }
    }

    #[cfg(not(unix))]
    fn execute(&self, _request: &ResourceWorkerRequest) -> Result<ResourceWorkerResponse> {
        bail!("external resource evaluators require a Unix process host")
    }
}

#[cfg(unix)]
fn drain_resource_output(
    child: &mut SupervisedProcess,
    response: &mut Option<ResourceWorkerResponse>,
) -> Result<()> {
    loop {
        match child.read_line()? {
            ProtocolRead::Line(line) => {
                ensure!(
                    response.is_none(),
                    "resource evaluator emitted output after its response"
                );
                *response = Some(parse_resource_response(&line)?);
            }
            ProtocolRead::Pending | ProtocolRead::Eof => return Ok(()),
        }
    }
}

/// Resolve a path lexically and reject every symlink in the supplied path,
/// rather than accepting a canonical path that silently traversed one.
fn validate_real_path(path: &Path, label: &str) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "{label} path is empty");
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .with_context(|| format!("resolving current directory for {label}"))?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => bail!("{label} path must not contain `..`"),
        }
    }
    ensure!(
        normalized.is_absolute(),
        "{label} path did not resolve absolutely"
    );

    let mut cursor = PathBuf::new();
    let component_count = normalized.components().count();
    for (index, component) in normalized.components().enumerate() {
        cursor.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&cursor)
            .with_context(|| format!("inspecting {label} ancestor {}", cursor.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "{label} path traverses symlink {}",
            cursor.display()
        );
        if index + 1 == component_count {
            ensure!(
                metadata.file_type().is_file(),
                "{label} {} must be a regular file",
                cursor.display()
            );
        } else {
            ensure!(
                metadata.file_type().is_dir(),
                "{label} ancestor {} must be a directory",
                cursor.display()
            );
        }
    }
    ensure!(
        normalized.canonicalize()? == normalized,
        "{label} path does not have a stable canonical identity"
    );
    Ok(normalized)
}

pub fn run_resource_benchmark(
    selected_run: &LoadedBenchmarkRun,
    comparison_runs: &[LoadedBenchmarkRun],
    policy: &AcceptancePolicy,
    evaluator: &ExternalResourceEvaluator,
    output_directory: &Path,
) -> Result<ResourceEvidencePublication> {
    policy.validate()?;
    let strongest_baseline_id = verified_resource_benchmark_context(selected_run, comparison_runs)?;
    fs::create_dir_all(output_directory).with_context(|| {
        format!(
            "creating resource evidence directory {}",
            output_directory.display()
        )
    })?;
    let artifact_directory = output_directory.join(RESOURCE_ARTIFACT_DIRECTORY);
    fs::create_dir_all(&artifact_directory).with_context(|| {
        format!(
            "creating resource artifact directory {}",
            artifact_directory.display()
        )
    })?;
    let mut baseline = selected_run.run().metadata.baseline.clone();
    let mut candidate = selected_run.run().metadata.candidate.clone();
    baseline.resolve_paths(selected_run.path());
    candidate.resolve_paths(selected_run.path());
    let request = ResourceWorkerRequest {
        version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
        baseline: baseline.clone(),
        candidate: candidate.clone(),
        strongest_baseline_id: strongest_baseline_id.clone(),
        evaluator_arguments: evaluator.arguments().to_vec(),
        artifact_directory,
    };
    let response = evaluator.execute(&request)?;

    let comparison = ResourceComparison {
        version: RESOURCE_COMPARISON_VERSION,
        baseline_id: baseline.id.clone(),
        candidate_id: candidate.id.clone(),
        strongest_baseline_id,
        measurement_evaluator_id: policy.resource_evaluator_id.clone(),
        wake_trials: response.wake_trials,
        candidate_capacity: response.candidate_capacity,
        grouped_mm_parity: KernelParityEvidence {
            samples: response.grouped_mm_samples,
        },
        pytorch_parity: KernelParityEvidence {
            samples: response.pytorch_samples,
        },
        exact_resume: response.exact_resume,
    };
    comparison.validate()?;

    let target = output_directory.join(RESOURCE_COMPARISON_FILE);
    verify_exact_resume(&comparison.exact_resume, &target)?;
    atomic_write_new(&target, &pretty_json_bytes(&comparison)?)?;
    Ok(ResourceEvidencePublication { path: target })
}

#[cfg(test)]
fn read_resource_response(stdout: impl Read) -> Result<ResourceWorkerResponse> {
    let mut reader = BufReader::new(stdout);
    let line =
        read_bounded_line(&mut reader)?.context("resource evaluator exited before responding")?;
    ensure!(
        read_bounded_line(&mut reader)?.is_none(),
        "resource evaluator emitted output after its response"
    );
    parse_resource_response(&line)
}

fn parse_resource_response(line: &[u8]) -> Result<ResourceWorkerResponse> {
    let response: ResourceWorkerResponse =
        serde_json::from_slice(line).context("resource evaluator emitted invalid protocol JSON")?;
    ensure!(
        response.version == RESOURCE_EXECUTION_PROTOCOL_VERSION,
        "unsupported resource evaluator response version {}",
        response.version
    );
    Ok(response)
}

#[cfg(test)]
fn read_bounded_line(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            ensure!(
                line.is_empty(),
                "resource evaluator response is unterminated"
            );
            return Ok(None);
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        ensure!(
            line.len() + take <= MAX_RESOURCE_MESSAGE_BYTES,
            "resource evaluator response exceeds {MAX_RESOURCE_MESSAGE_BYTES} bytes"
        );
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.ends_with(b"\n") {
            return Ok(Some(line));
        }
    }
}

fn pretty_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::acceptance::{RESOURCE_EXECUTION_PROTOCOL_VERSION, WakeMeasurement};

    fn target(id: &str) -> BenchmarkTarget {
        BenchmarkTarget {
            id: id.into(),
            checkpoint_manifest: format!("/{id}/generation-manifest.json").into(),
            checkpoint_manifest_sha256: "1".repeat(64),
            training_gpu_hours: 1.0,
            parameters: 100,
            routed_active_parameters: 80,
            stored_bytes: 400,
            representation: crate::benchmark::ModelRepresentationTarget::FullPrecision,
        }
    }

    fn exact_resume(path: impl Into<PathBuf>) -> ExactResumeEvidence {
        let path = path.into();
        let artifact = || crate::acceptance::ExactResumeArtifact { path: path.clone() };
        ExactResumeEvidence {
            interrupted_checkpoint: artifact(),
            uninterrupted_final_state: artifact(),
            resumed_final_state: artifact(),
            uninterrupted_metrics: artifact(),
            resumed_metrics: artifact(),
            interruption_step: 1,
            resumed_from_step: 1,
        }
    }

    fn response() -> ResourceWorkerResponse {
        ResourceWorkerResponse {
            version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
            wake_trials: vec![PairedWakeTrial {
                trial: 0,
                baseline: WakeMeasurement {
                    tokens: 1,
                    elapsed_seconds: 1.0,
                    request_latency_ms: vec![1.0],
                },
                candidate: WakeMeasurement {
                    tokens: 1,
                    elapsed_seconds: 1.0,
                    request_latency_ms: vec![1.0],
                },
            }],
            candidate_capacity: vec![CapacityObservation {
                completed_sleep_cycles: 0,
                routed_active_parameters: 80,
                stored_parameters: 100,
                stored_bytes: 400,
            }],
            grouped_mm_samples: vec![KernelParitySample {
                reference: 1.0,
                candidate: 1.0,
            }],
            pytorch_samples: vec![KernelParitySample {
                reference: 1.0,
                candidate: 1.0,
            }],
            exact_resume: exact_resume("artifacts/exact.json"),
        }
    }

    fn request(arguments: Vec<String>) -> ResourceWorkerRequest {
        ResourceWorkerRequest {
            version: RESOURCE_EXECUTION_PROTOCOL_VERSION,
            baseline: target("baseline"),
            candidate: target("candidate"),
            strongest_baseline_id: "baseline".into(),
            evaluator_arguments: arguments,
            artifact_directory: "/vault/artifacts".into(),
        }
    }

    #[cfg(unix)]
    fn executable(root: &Path, name: &str, source: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = root.join(name);
        fs::write(&path, source).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&path, permissions).unwrap();
        path
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
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("detached worker did not publish its pid");
    }

    #[test]
    fn response_reader_rejects_unterminated_and_oversized_messages() {
        let error = read_resource_response(Cursor::new(b"{}".to_vec()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unterminated"), "{error}");

        let oversized = vec![b'x'; MAX_RESOURCE_MESSAGE_BYTES + 1];
        let error = read_bounded_line(&mut Cursor::new(oversized))
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeds"), "{error}");
    }

    #[test]
    fn evaluator_rejects_a_timeout_outside_the_monotonic_clock_range() {
        let evaluator = ExternalResourceEvaluator {
            executable: PathBuf::from("unused-test-worker"),
            arguments: Vec::new(),
            timeout: DEFAULT_RESOURCE_EVALUATOR_TIMEOUT,
        };

        let error = evaluator
            .with_timeout(Duration::MAX)
            .unwrap_err()
            .to_string();
        assert!(error.contains("monotonic clock"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn hanging_evaluator_is_killed_at_the_bound() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let worker = executable(
            &root,
            "hang.sh",
            "#!/bin/sh\nIFS= read -r request\n/bin/sleep 30\n",
        );
        let evaluator = ExternalResourceEvaluator::new(&worker, Vec::new())
            .unwrap()
            .with_timeout(Duration::from_millis(100))
            .unwrap();
        let started = Instant::now();
        let error = evaluator
            .execute(&request(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("timeout"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[cfg(unix)]
    #[test]
    fn timeout_is_bounded_when_setsid_descendant_retains_protocol_pipes() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let detached_pid = root.join("detached.pid");
        let worker = executable(
            &root,
            "setsid-timeout.sh",
            concat!(
                "#!/bin/sh\n",
                "IFS= read -r request\n",
                "test -n \"$request\"\n",
                "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                "/bin/sleep 30\n"
            ),
        );
        let evaluator = ExternalResourceEvaluator::new(
            &worker,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
        )
        .unwrap()
        .with_timeout(Duration::from_secs(3))
        .unwrap();

        let started = Instant::now();
        let result = evaluator.execute(&request(Vec::new()));
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("timeout"), "{error}");
        assert!(
            elapsed < Duration::from_secs(5),
            "setsid descendant delayed resource timeout for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn protocol_error_is_bounded_when_setsid_descendant_retains_stdout() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let detached_pid = root.join("detached.pid");
        let worker = executable(
            &root,
            "setsid-error.sh",
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
        let evaluator = ExternalResourceEvaluator::new(
            &worker,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
        )
        .unwrap()
        .with_timeout(Duration::from_secs(3))
        .unwrap();

        let started = Instant::now();
        let result = evaluator.execute(&request(Vec::new()));
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("invalid protocol JSON"), "{error}");
        assert!(
            elapsed < Duration::from_secs(2),
            "setsid descendant delayed resource protocol error for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn evaluator_rejects_output_after_the_one_response() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let response_path = root.join("response.jsonl");
        let mut response_bytes = serde_json::to_vec(&response()).unwrap();
        response_bytes.push(b'\n');
        fs::write(&response_path, response_bytes).unwrap();
        let worker = executable(
            &root,
            "extra.sh",
            &format!(
                "#!/bin/sh\nIFS= read -r request\n/bin/cat '{}'\nprintf '{{}}\\n'\n",
                response_path.display()
            ),
        );
        let evaluator = ExternalResourceEvaluator::new(&worker, Vec::new()).unwrap();
        let error = evaluator
            .execute(&request(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("after its response"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn exited_evaluator_cannot_leave_a_descendant_holding_protocol_pipes() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let response_path = root.join("response.jsonl");
        let mut response_bytes = serde_json::to_vec(&response()).unwrap();
        response_bytes.push(b'\n');
        fs::write(&response_path, response_bytes).unwrap();
        let worker = executable(
            &root,
            "orphan.sh",
            &format!(
                "#!/bin/sh\nIFS= read -r request\n/bin/cat '{}'\n/bin/sleep 30 &\nexit 0\n",
                response_path.display()
            ),
        );
        let evaluator = ExternalResourceEvaluator::new(&worker, Vec::new())
            .unwrap()
            // Leave enough scheduling margin when all process-lifecycle tests
            // execute concurrently; the elapsed-time assertion still catches
            // a descendant retaining the pipe for its 30-second lifetime.
            .with_timeout(Duration::from_secs(3))
            .unwrap();

        let started = Instant::now();
        evaluator.execute(&request(Vec::new())).unwrap();
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[cfg(unix)]
    #[test]
    fn leader_exit_is_bounded_when_setsid_descendant_retains_stdout() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let response_path = root.join("response.jsonl");
        let detached_pid = root.join("detached.pid");
        let mut response_bytes = serde_json::to_vec(&response()).unwrap();
        response_bytes.push(b'\n');
        fs::write(&response_path, response_bytes).unwrap();
        let worker = executable(
            &root,
            "setsid-exit.sh",
            &format!(
                concat!(
                    "#!/bin/sh\n",
                    "IFS= read -r request\n",
                    "test -n \"$request\"\n",
                    "\"$1\" -c 'import os,sys,time; os.setsid(); f=open(sys.argv[1],\"w\"); f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno()); time.sleep(30)' \"$2\" &\n",
                    "while [ ! -s \"$2\" ]; do /bin/sleep 0.01; done\n",
                    "/bin/cat '{}'\n"
                ),
                response_path.display()
            ),
        );
        let evaluator = ExternalResourceEvaluator::new(
            &worker,
            vec![
                python_with_setsid().into_os_string(),
                detached_pid.clone().into_os_string(),
            ],
        )
        .unwrap()
        .with_timeout(Duration::from_secs(2))
        .unwrap();

        let started = Instant::now();
        let result = evaluator.execute(&request(Vec::new()));
        let elapsed = started.elapsed();
        kill_detached_pid(&detached_pid);
        result.unwrap();
        assert!(
            elapsed < Duration::from_secs(3),
            "setsid descendant delayed resource leader exit for {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn evaluator_rejects_direct_and_ancestor_symlinks() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let real = root.join("real");
        fs::create_dir(&real).unwrap();
        let worker = executable(&real, "worker.sh", "#!/bin/sh\nexit 0\n");
        let direct = root.join("worker-link");
        symlink(&worker, &direct).unwrap();
        assert!(ExternalResourceEvaluator::new(&direct, Vec::new()).is_err());

        let linked_parent = root.join("linked-parent");
        symlink(&real, &linked_parent).unwrap();
        let error = ExternalResourceEvaluator::new(linked_parent.join("worker.sh"), Vec::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("traverses symlink"), "{error}");
    }
}
