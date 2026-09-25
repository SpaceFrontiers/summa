//! JSONL bridge for benchmark evaluators.
//!
//! Benchmark orchestration verifies suites and model manifests in-process. A
//! long-lived local worker performs model-specific evaluation without making
//! the benchmark framework depend on a particular generation or retrieval
//! implementation. Requests are strictly versioned and every response is
//! validated by `BenchmarkRunner`.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::acceptance::SuiteVisibility;
use crate::benchmark::{
    BenchmarkEvaluator, BenchmarkSpec, BenchmarkTarget, EvaluationMeasurement, EvaluationRequest,
    ResolvedModelRepresentation, SuiteArtifact, TargetRole, validate_benchmark_evaluator_arguments,
};
#[cfg(unix)]
use crate::protocol_process::{ProtocolRead, SupervisedProcess};

pub const BENCHMARK_WORKER_PROTOCOL_VERSION: u32 = 2;
const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_BENCHMARK_EVALUATOR_TIMEOUT: Duration = Duration::from_secs(3_600);

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkWorkerRequest<'a> {
    pub version: u32,
    pub suite_id: &'a str,
    pub visibility: SuiteVisibility,
    pub case: &'a BenchmarkSpec,
    pub artifact: &'a SuiteArtifact,
    pub target: &'a BenchmarkTarget,
    pub model: &'a ResolvedModelRepresentation,
    pub role: TargetRole,
    pub model_seed: u64,
    pub example_order_seed: u64,
    pub pair_ordinal: usize,
    pub max_gpu_hours: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkWorkerResponse {
    version: u32,
    measurement: EvaluationMeasurement,
}

#[cfg(unix)]
struct RunningWorker {
    process: SupervisedProcess,
}

#[cfg(unix)]
impl RunningWorker {
    fn from_child(child: Child) -> Result<Self> {
        Ok(Self {
            process: SupervisedProcess::new(child, "benchmark evaluator", MAX_MESSAGE_BYTES)?,
        })
    }

    fn exchange(&mut self, request: &[u8], timeout: Duration) -> Result<Vec<u8>> {
        match self.process.read_line()? {
            ProtocolRead::Line(_) => {
                bail!("benchmark evaluator emitted an unsolicited response")
            }
            ProtocolRead::Eof => {
                bail!("benchmark evaluator closed stdout before receiving a request")
            }
            ProtocolRead::Pending => ensure!(
                !self.process.has_buffered_output(),
                "benchmark evaluator emitted an unsolicited response"
            ),
        }
        if let Some(status) = self.process.try_wait()? {
            bail!("benchmark evaluator exited with status {status} before receiving a request");
        }

        let deadline = checked_deadline(timeout)?;
        let mut written = 0_usize;
        loop {
            self.process.write_available(request, &mut written)?;
            match self.process.read_line()? {
                ProtocolRead::Line(line) => {
                    ensure!(
                        written == request.len(),
                        "benchmark evaluator responded before consuming the complete request"
                    );
                    // Drain bytes that were queued with the response. A second
                    // frame or partial frame is never assigned to the next
                    // request on this persistent connection.
                    match self.process.read_line()? {
                        ProtocolRead::Line(_) => {
                            bail!("benchmark evaluator emitted data after its response")
                        }
                        ProtocolRead::Pending => ensure!(
                            !self.process.has_buffered_output(),
                            "benchmark evaluator emitted data after its response"
                        ),
                        ProtocolRead::Eof => {}
                    }
                    return Ok(line);
                }
                ProtocolRead::Eof => match self.process.try_wait()? {
                    Some(status) => {
                        bail!("benchmark evaluator exited with status {status} before responding")
                    }
                    None => bail!("benchmark evaluator closed stdout before responding"),
                },
                ProtocolRead::Pending => {}
            }
            if let Some(status) = self.process.try_wait()? {
                bail!("benchmark evaluator exited with status {status} before responding");
            }
            ensure_before_deadline(deadline, timeout)?;
            self.process
                .wait_for_activity(written < request.len(), deadline)?;
        }
    }

    fn finish_protocol(&mut self, timeout: Duration) -> Result<()> {
        self.process.close_input();
        let deadline = checked_deadline(timeout)?;
        loop {
            match self.process.read_line()? {
                ProtocolRead::Line(_) => {
                    bail!("benchmark evaluator emitted an unsolicited response")
                }
                ProtocolRead::Pending => ensure!(
                    !self.process.has_buffered_output(),
                    "benchmark evaluator emitted an unsolicited response"
                ),
                ProtocolRead::Eof => {}
            }
            if let Some(status) = self.process.try_wait()? {
                // A successful leader may leave a descendant holding stdout.
                // The protocol grants no background-process lifetime, so reap
                // the complete group before accepting the exit.
                self.process.terminate_process_group();
                ensure!(
                    status.success(),
                    "benchmark evaluator exited with status {status}"
                );
                return Ok(());
            }
            ensure_before_deadline(deadline, timeout)?;
            self.process.wait_for_activity(false, deadline)?;
        }
    }

    fn terminate(&mut self) {
        self.process.terminate();
    }
}

#[cfg(not(unix))]
struct RunningWorker;

#[cfg(not(unix))]
impl RunningWorker {
    fn exchange(&mut self, _: &[u8], _: Duration) -> Result<Vec<u8>> {
        bail!("external benchmark evaluators require a Unix process host")
    }

    fn finish_protocol(&mut self, _: Duration) -> Result<()> {
        bail!("external benchmark evaluators require a Unix process host")
    }

    fn terminate(&mut self) {}
}

/// Persistent evaluator process used for a complete paired benchmark run.
/// No shell is involved and the worker receives only already-verified local
/// artifact paths and immutable target metadata.
pub struct ExternalBenchmarkEvaluator {
    executable: PathBuf,
    arguments: Vec<String>,
    timeout: Duration,
    running: Option<RunningWorker>,
}

impl ExternalBenchmarkEvaluator {
    pub fn new(executable: impl Into<PathBuf>, arguments: Vec<OsString>) -> Result<Self> {
        let arguments = arguments
            .into_iter()
            .map(|argument| {
                argument
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("benchmark evaluator argument is not valid UTF-8"))
            })
            .collect::<Result<Vec<_>>>()?;
        validate_benchmark_evaluator_arguments(&arguments)?;
        Ok(Self {
            executable: executable.into(),
            arguments,
            timeout: DEFAULT_BENCHMARK_EVALUATOR_TIMEOUT,
            running: None,
        })
    }

    /// Set the hard wall-clock bound for each request and for graceful worker
    /// shutdown. A timeout always kills and reaps the entire worker process
    /// group before returning.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        ensure!(
            !timeout.is_zero(),
            "benchmark evaluator timeout must be positive"
        );
        checked_deadline(timeout)?;
        self.timeout = timeout;
        Ok(self)
    }

    /// Canonical argument vector that is part of benchmark run identity.
    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }

    #[cfg(unix)]
    fn start(&mut self) -> Result<()> {
        if self.running.is_some() {
            return Ok(());
        }
        let mut command = Command::new(&self.executable);
        command
            .args(&self.arguments)
            .env_clear()
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .process_group(0);
        let child = command.spawn().with_context(|| {
            format!(
                "failed to start benchmark evaluator {}",
                self.executable.display()
            )
        })?;
        self.running = Some(RunningWorker::from_child(child)?);
        Ok(())
    }

    #[cfg(not(unix))]
    fn start(&mut self) -> Result<()> {
        bail!("external benchmark evaluators require a Unix process host")
    }

    /// Close the protocol cleanly and require a successful worker exit.
    pub fn finish(mut self) -> Result<()> {
        let Some(mut running) = self.running.take() else {
            return Ok(());
        };
        running.finish_protocol(self.timeout)
    }

    fn evaluate_wire(
        &mut self,
        wire: &BenchmarkWorkerRequest<'_>,
    ) -> Result<EvaluationMeasurement> {
        let mut request = serde_json::to_vec(wire)?;
        ensure!(
            request.len() < MAX_MESSAGE_BYTES,
            "benchmark evaluator request exceeds {MAX_MESSAGE_BYTES} bytes"
        );
        request.push(b'\n');
        self.start()?;
        let result = self
            .running
            .as_mut()
            .expect("benchmark worker was started above")
            .exchange(&request, self.timeout)
            .and_then(|line| {
                let response: BenchmarkWorkerResponse = serde_json::from_slice(&line)
                    .context("benchmark evaluator emitted invalid protocol JSON")?;
                ensure!(
                    response.version == BENCHMARK_WORKER_PROTOCOL_VERSION,
                    "unsupported benchmark evaluator response version {}",
                    response.version
                );
                Ok(response.measurement)
            });
        if result.is_err() {
            // A protocol failure poisons stream framing. Never reuse that
            // process for another evaluation.
            if let Some(mut running) = self.running.take() {
                running.terminate();
            }
        }
        result
    }
}

impl BenchmarkEvaluator for ExternalBenchmarkEvaluator {
    fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
        let wire = BenchmarkWorkerRequest {
            version: BENCHMARK_WORKER_PROTOCOL_VERSION,
            suite_id: request.suite_id,
            visibility: request.visibility,
            case: request.case,
            artifact: request.artifact,
            target: request.target,
            model: request.model,
            role: request.role,
            model_seed: request.model_seed,
            example_order_seed: request.example_order_seed,
            pair_ordinal: request.pair_ordinal,
            max_gpu_hours: request.max_gpu_hours,
        };
        self.evaluate_wire(&wire)
    }
}

fn checked_deadline(timeout: Duration) -> Result<Instant> {
    Instant::now()
        .checked_add(timeout)
        .context("benchmark evaluator timeout is too large for the monotonic clock")
}

fn ensure_before_deadline(deadline: Instant, timeout: Duration) -> Result<()> {
    ensure!(
        Instant::now() < deadline,
        "benchmark evaluator exceeded its {timeout:?} execution timeout"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::time::Instant;

    use super::*;
    use crate::acceptance::{BenchmarkFamily, SuiteVisibility};
    use crate::benchmark::{BenchmarkArtifact, BenchmarkSpec, MetricDirection, TargetRole};

    #[cfg(unix)]
    fn worker_script(directory: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let worker = directory.join("evaluator.sh");
        std::fs::write(&worker, format!("#!/bin/sh\n{body}")).unwrap();
        let mut permissions = std::fs::metadata(&worker).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&worker, permissions).unwrap();
        worker
    }

    fn fixture(
        directory: &Path,
    ) -> (
        SuiteArtifact,
        BenchmarkSpec,
        BenchmarkTarget,
        ResolvedModelRepresentation,
    ) {
        let artifact = SuiteArtifact {
            path: directory.join("fixture.jsonl"),
            bytes: 1,
        };
        let case = BenchmarkSpec {
            id: "reasoning".into(),
            catalog_id: None,
            family: BenchmarkFamily::Reasoning,
            metric: "accuracy".into(),
            direction: MetricDirection::Maximize,
            stable_anchor: false,
            artifact: BenchmarkArtifact {
                path: artifact.path.clone(),
            },
            evaluator: BTreeMap::new(),
        };
        let target = BenchmarkTarget {
            id: "candidate".into(),
            checkpoint_manifest: directory.join("generation-manifest.json"),
            checkpoint_manifest_sha256: "2".repeat(64),
            training_gpu_hours: 1.0,
            parameters: 10,
            routed_active_parameters: 8,
            stored_bytes: 100,
            representation: crate::benchmark::ModelRepresentationTarget::FullPrecision,
        };
        let model = ResolvedModelRepresentation::FullPrecision {
            weights: directory.join("weights.safetensors"),
            stored_bytes: 100,
        };
        (artifact, case, target, model)
    }

    fn evaluate_once(
        evaluator: &mut ExternalBenchmarkEvaluator,
        directory: &Path,
    ) -> Result<EvaluationMeasurement> {
        let (artifact, case, target, model) = fixture(directory);
        evaluator.evaluate(&EvaluationRequest {
            suite_id: "public",
            visibility: SuiteVisibility::Public,
            case: &case,
            artifact: &artifact,
            target: &target,
            model: &model,
            role: TargetRole::Candidate,
            model_seed: 7,
            example_order_seed: 9,
            pair_ordinal: 0,
            max_gpu_hours: 0.1,
        })
    }

    #[cfg(unix)]
    #[test]
    fn persistent_worker_receives_strict_requests_and_finishes() {
        let directory = tempfile::tempdir().unwrap();
        let worker = worker_script(
            directory.path(),
            concat!(
                "while IFS= read -r request; do\n",
                "  test -n \"$request\" || exit 7\n",
                "  case \"$request\" in *'\"model\":{\"type\":\"full_precision\",'*) ;; *) exit 8 ;; esac\n",
                "  test \"$(pwd)\" = / || exit 9\n",
                "  test -z \"${HOME+x}\" || exit 10\n",
                "  test \"$1\" = '--profile=a b' || exit 11\n",
                "  printf '%s\\n' '{\"version\":2,\"measurement\":{\"score\":0.75,\"gpu_hours\":0.01,\"examples\":4}}'\n",
                "done\n"
            ),
        );
        let mut evaluator =
            ExternalBenchmarkEvaluator::new(&worker, vec![OsString::from("--profile=a b")])
                .unwrap();
        assert_eq!(evaluator.arguments(), ["--profile=a b"]);
        let first = evaluate_once(&mut evaluator, directory.path()).unwrap();
        let second = evaluate_once(&mut evaluator, directory.path()).unwrap();
        assert_eq!(first.score, 0.75);
        assert_eq!(second.examples, 4);
        evaluator.finish().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn response_timeout_terminates_the_worker() {
        let directory = tempfile::tempdir().unwrap();
        let worker = worker_script(directory.path(), "read -r request\n/bin/sleep 30\n");
        let mut evaluator = ExternalBenchmarkEvaluator::new(&worker, Vec::new())
            .unwrap()
            .with_timeout(Duration::from_millis(100))
            .unwrap();
        let started = Instant::now();
        let error = evaluate_once(&mut evaluator, directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("execution timeout"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(evaluator.running.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn finish_kills_descendants_that_retain_protocol_pipes() {
        let directory = tempfile::tempdir().unwrap();
        let worker = worker_script(
            directory.path(),
            concat!(
                "while IFS= read -r request; do\n",
                "  printf '%s\\n' '{\"version\":2,\"measurement\":{\"score\":0.5,\"gpu_hours\":0.01,\"examples\":1}}'\n",
                "done\n",
                "/bin/sleep 30 &\n",
                "exit 0\n"
            ),
        );
        let mut evaluator = ExternalBenchmarkEvaluator::new(&worker, Vec::new())
            .unwrap()
            // Process startup can contend with the other lifecycle tests;
            // this remains far below the descendant's 30-second lifetime.
            .with_timeout(Duration::from_secs(2))
            .unwrap();
        evaluate_once(&mut evaluator, directory.path()).unwrap();
        let started = Instant::now();
        evaluator.finish().unwrap();
        assert!(started.elapsed() < Duration::from_secs(4));
    }

    #[cfg(unix)]
    #[test]
    fn finish_timeout_terminates_a_worker_that_does_not_exit() {
        let directory = tempfile::tempdir().unwrap();
        let worker = worker_script(
            directory.path(),
            concat!(
                "while IFS= read -r request; do\n",
                "  printf '%s\\n' '{\"version\":2,\"measurement\":{\"score\":0.5,\"gpu_hours\":0.01,\"examples\":1}}'\n",
                "done\n",
                "/bin/sleep 30\n"
            ),
        );
        let mut evaluator = ExternalBenchmarkEvaluator::new(&worker, Vec::new())
            .unwrap()
            .with_timeout(Duration::from_secs(2))
            .unwrap();
        evaluate_once(&mut evaluator, directory.path()).unwrap();
        let started = Instant::now();
        let error = evaluator.finish().unwrap_err().to_string();
        assert!(error.contains("execution timeout"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(4));
    }

    #[cfg(unix)]
    #[test]
    fn malformed_response_poisoning_cleans_up_the_worker() {
        let directory = tempfile::tempdir().unwrap();
        let worker = worker_script(
            directory.path(),
            "read -r request\nprintf '%s\\n' '{not-json}'\n/bin/sleep 30\n",
        );
        let mut evaluator = ExternalBenchmarkEvaluator::new(&worker, Vec::new())
            .unwrap()
            .with_timeout(Duration::from_secs(3))
            .unwrap();
        let error = evaluate_once(&mut evaluator, directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid protocol JSON"), "{error}");
        assert!(evaluator.running.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn extra_response_is_rejected_and_worker_is_cleaned_up() {
        let directory = tempfile::tempdir().unwrap();
        let response =
            "{\"version\":2,\"measurement\":{\"score\":0.5,\"gpu_hours\":0.01,\"examples\":1}}";
        let worker = worker_script(
            directory.path(),
            &format!(
                "read -r request\nprintf '%s\\n%s\\n' '{response}' '{response}'\n/bin/sleep 30\n"
            ),
        );
        let mut evaluator = ExternalBenchmarkEvaluator::new(&worker, Vec::new())
            .unwrap()
            .with_timeout(Duration::from_secs(3))
            .unwrap();
        let error = evaluate_once(&mut evaluator, directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("data after its response"), "{error}");
        assert!(evaluator.running.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn evaluator_arguments_are_canonical_and_bounded() {
        use std::os::unix::ffi::OsStringExt;

        let directory = tempfile::tempdir().unwrap();
        let worker = worker_script(directory.path(), "exit 0\n");

        let error = ExternalBenchmarkEvaluator::new(&worker, vec![OsString::from_vec(vec![0xff])])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("not valid UTF-8"), "{error}");

        let error = ExternalBenchmarkEvaluator::new(&worker, vec![OsString::from("argument"); 65])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("exceed protocol limits"), "{error}");

        let error = ExternalBenchmarkEvaluator::new(&worker, vec![OsString::from("contains\0nul")])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("exceed protocol limits"), "{error}");
    }

    #[test]
    fn timeout_must_be_positive() {
        let directory = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            let worker = worker_script(directory.path(), "exit 0\n");
            let error = ExternalBenchmarkEvaluator::new(&worker, Vec::new())
                .unwrap()
                .with_timeout(Duration::ZERO)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("timeout must be positive"));
        }
    }
}
