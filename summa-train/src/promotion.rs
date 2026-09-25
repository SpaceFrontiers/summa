//! Trainer-owned WorkflowV2 promotion gate.
//!
//! Promotion is deliberately not a generic worker phase. The executor below
//! independently reloads the complete benchmark/ablation evidence, binds it to
//! the exact runtime checkpoint, evaluates the acceptance policy, and writes a
//! deterministic immutable decision. A rejected decision is retained for
//! audit but can never produce a [`PhaseProduct::Release`].

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::acceptance::{AcceptancePolicy, PromotionReport, ResourceComparison};
#[cfg(test)]
use crate::artifact_io::sha256_hex;
use crate::artifact_io::{atomic_write_new, read_regular_bounded, sha256_identity};
use crate::benchmark::{AblationId, LoadedBenchmarkRun, evaluate_verified_promotion};
use crate::runtime::{
    ImmutableArtifact, ImmutableModelCheckpoint, PhaseExecutionRequest, PhaseExecutionResult,
    PhaseExecutor, PhaseProduct, PhaseProgressSink,
};
use crate::workflow::{PhaseKind, PromotionConfig};

pub const PROMOTION_DECISION_VERSION: u32 = 2;
const MAX_PROMOTION_JSON_BYTES: u64 = 64 * 1024 * 1024;

/// Complete auditable authorization record. The acceptance report is nested
/// beside the exact checkpoint it authorizes.
#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedPromotionDecision {
    pub version: u32,
    pub phase_index: usize,
    pub phase_name: String,
    pub candidate: ImmutableModelCheckpoint,
    pub benchmark_candidate_id: String,
    pub report: PromotionReport,
}

/// Built-in promotion executor registered by the stock CLI and embedded host
/// in place of the external worker for [`PhaseKind::Promotion`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NativePromotionExecutor;

impl<C> PhaseExecutor<C> for NativePromotionExecutor {
    fn execute(
        &mut self,
        request: &PhaseExecutionRequest,
        _context: &mut C,
        _progress: &mut dyn PhaseProgressSink,
    ) -> Result<PhaseExecutionResult> {
        ensure!(
            request.phase.kind == PhaseKind::Promotion,
            "native promotion executor received a `{}` phase",
            request.phase.kind.name()
        );
        ensure!(
            request.resume_state.is_none(),
            "native promotion rejects opaque resume state from an external phase worker"
        );
        let candidate = request
            .input_checkpoint
            .as_ref()
            .context("native promotion requires the current immutable checkpoint")?;
        let config = request
            .phase
            .promotion
            .as_ref()
            .context("native promotion requires typed promotion settings")?;
        config.validate(&request.phase.name)?;

        let (decision, artifact) = evaluate_and_publish(request, candidate, config)?;
        ensure!(
            decision.report.accepted,
            "candidate failed native promotion gates; immutable rejection decision published at {}",
            artifact.uri()
        );
        Ok(PhaseExecutionResult::Complete(PhaseProduct::Release {
            candidate: candidate.clone(),
            receipt: artifact,
        }))
    }
}

fn evaluate_and_publish(
    request: &PhaseExecutionRequest,
    candidate: &ImmutableModelCheckpoint,
    config: &PromotionConfig,
) -> Result<(VerifiedPromotionDecision, ImmutableArtifact)> {
    let selected = LoadedBenchmarkRun::load(&config.selected_run.path)?;
    let mut comparisons = Vec::with_capacity(AblationId::ALL.len());
    comparisons.push(selected.clone());
    for run in &config.comparison_runs {
        comparisons.push(LoadedBenchmarkRun::load(&run.path)?);
    }

    let benchmark_candidate = &selected.run().metadata.candidate;
    let runtime_digest = candidate
        .sha256()
        .strip_prefix("sha256:")
        .expect("immutable checkpoint validation enforces canonical digest");
    ensure!(
        benchmark_candidate
            .checkpoint_manifest_sha256
            .eq_ignore_ascii_case(runtime_digest),
        "selected benchmark candidate checkpoint {} does not match current workflow checkpoint {}",
        benchmark_candidate.checkpoint_manifest_sha256,
        candidate.sha256()
    );

    let resources: ResourceComparison = read_json(&config.resources.path, "resource evidence")?;
    let policy: AcceptancePolicy = read_json(&config.policy.path, "acceptance policy")?;
    let report = evaluate_verified_promotion(
        &selected,
        &comparisons,
        &resources,
        &config.resources.path,
        &policy,
    )?;
    let decision = VerifiedPromotionDecision {
        version: PROMOTION_DECISION_VERSION,
        phase_index: request.phase_index,
        phase_name: request.phase.name.clone(),
        candidate: candidate.clone(),
        benchmark_candidate_id: benchmark_candidate.id.clone(),
        report,
    };
    let bytes = decision_bytes(&decision)?;
    let decision_sha256 = sha256_identity(&bytes);
    let directory = prepare_artifact_directory(&config.artifact_directory)?;
    let path = directory.join(format!("promotion-{:04}.json", request.phase_index));
    atomic_write_new(&path, &bytes)?;
    let uri = path
        .to_str()
        .context("promotion decision path is not valid UTF-8")?
        .to_owned();
    let artifact = ImmutableArtifact::new(uri, decision_sha256)?;
    Ok((decision, artifact))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, label: &str) -> Result<T> {
    let bytes = read_regular_bounded(path, MAX_PROMOTION_JSON_BYTES, label)
        .with_context(|| format!("failed to read {label} {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid {label} JSON in {}", path.display()))
}

fn decision_bytes(decision: &VerifiedPromotionDecision) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(decision)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn prepare_artifact_directory(path: &Path) -> Result<PathBuf> {
    ensure!(
        !path.as_os_str().is_empty(),
        "promotion artifact directory is empty"
    );
    fs::create_dir_all(path)
        .with_context(|| format!("create promotion artifact directory {}", path.display()))?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::acceptance::{
        CapacityObservation, ExactResumeArtifact, ExactResumeEvidence, KernelParityEvidence,
        KernelParitySample, PairedWakeTrial, RESOURCE_COMPARISON_VERSION, ResourceComparison,
        SuiteVisibility, WakeMeasurement,
    };
    use crate::benchmark::{
        BENCHMARK_MANIFEST_VERSION, BenchmarkArtifact, BenchmarkEvaluator, BenchmarkRun,
        BenchmarkRunConfig, BenchmarkRunner, BenchmarkSpec, BenchmarkSuiteManifest,
        BenchmarkTarget, EvaluationMeasurement, EvaluationRequest, LoadedBenchmarkSuite,
        MetricDirection, TargetRole, required_catalog,
    };
    use crate::runtime::{
        ExecutorRegistry, RuntimeBoundary, RuntimeCheckpoint, RuntimeStatus, WorkflowRunState,
        run_next_phase,
    };
    use crate::workflow::PromotionEvidenceRef;
    use crate::workflow::{ResolvedWorkflow, WorkflowV2};
    use anyhow::bail;

    fn raw_sha256(bytes: &[u8]) -> String {
        sha256_hex(bytes)
    }

    fn write_json<T: Serialize>(path: &Path, value: &T) {
        fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    }

    fn target(root: &Path, id: &str) -> BenchmarkTarget {
        use safetensors::{Dtype, tensor::TensorView};

        let output = root.join(id);
        let staging = output.join("generations").join("staging");
        fs::create_dir_all(&staging).unwrap();
        let salt = id.bytes().enumerate().fold(0_u64, |sum, (index, byte)| {
            sum.wrapping_add((index as u64 + 1).wrapping_mul(u64::from(byte)))
        }) as f32;
        let raw_weights = (0..100)
            .map(|index| ((index as f32 + salt) * 0.03125).sin())
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![100], &raw_weights).unwrap();
        let weights = safetensors::tensor::serialize([("weight", view)], None).unwrap();
        let weights_sha256 = raw_sha256(&weights);
        fs::write(staging.join("weights.safetensors"), &weights).unwrap();
        let accounting = crate::benchmark::TrainingAccounting {
            version: crate::benchmark::TRAINING_ACCOUNTING_VERSION,
            training_gpu_hours: 8.0,
            parameters: 100,
            routed_active_parameters: 80,
            weights_bytes: weights.len() as u64,
            weights_sha256: weights_sha256.clone(),
        };
        let accounting_bytes = serde_json::to_vec(&accounting).unwrap();
        let accounting_sha256 = raw_sha256(&accounting_bytes);
        fs::write(
            staging.join(crate::benchmark::TRAINING_ACCOUNTING_FILE),
            &accounting_bytes,
        )
        .unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "training_state_version": 2,
            "global_step": 20,
            "phase": 0,
            "phase_id": "test",
            "files": [
                {"path": crate::benchmark::TRAINING_ACCOUNTING_FILE, "bytes": accounting_bytes.len(), "sha256": accounting_sha256},
                {"path": "weights.safetensors", "bytes": weights.len(), "sha256": weights_sha256}
            ]
        });
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let checkpoint_manifest_sha256 = raw_sha256(&manifest_bytes);
        let generation = output
            .join("generations")
            .join(format!("sha256-{checkpoint_manifest_sha256}"));
        fs::rename(&staging, &generation).unwrap();
        let checkpoint_manifest = generation.join("generation-manifest.json");
        fs::write(&checkpoint_manifest, &manifest_bytes).unwrap();
        BenchmarkTarget {
            id: id.into(),
            checkpoint_manifest,
            checkpoint_manifest_sha256,
            training_gpu_hours: 8.0,
            parameters: 100,
            routed_active_parameters: 80,
            stored_bytes: weights.len() as u64,
            representation: crate::benchmark::ModelRepresentationTarget::FullPrecision,
        }
    }

    fn catalog_suite(
        root: &Path,
        suite_id: &str,
        visibility: SuiteVisibility,
    ) -> LoadedBenchmarkSuite {
        let prefix = match visibility {
            SuiteVisibility::Public => "public",
            SuiteVisibility::Sealed => "sealed",
        };
        let stable_anchors = AcceptancePolicy::default().stable_anchor_catalog_ids;
        let cases = required_catalog()
            .into_iter()
            .filter(|entry| entry.stage.required_now())
            .map(|entry| {
                let case_id = format!("{prefix}-{}", entry.id);
                let artifact = root.join(format!("{case_id}.jsonl"));
                fs::write(&artifact, format!("{{\"id\":\"{case_id}\"}}\n")).unwrap();
                BenchmarkSpec {
                    id: case_id,
                    catalog_id: Some(entry.id.into()),
                    family: entry.family,
                    metric: "score".into(),
                    direction: MetricDirection::Maximize,
                    stable_anchor: stable_anchors.contains(entry.id),
                    artifact: BenchmarkArtifact {
                        path: artifact.file_name().unwrap().into(),
                    },
                    evaluator: BTreeMap::new(),
                }
            })
            .collect();
        let manifest = BenchmarkSuiteManifest {
            version: BENCHMARK_MANIFEST_VERSION,
            suite_id: suite_id.into(),
            visibility,
            cases,
        };
        let path = root.join(format!("{suite_id}.json"));
        write_json(&path, &manifest);
        LoadedBenchmarkSuite::load(path).unwrap()
    }

    struct PassingEvaluator;

    impl BenchmarkEvaluator for PassingEvaluator {
        fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
            let score = match request.role {
                TargetRole::Candidate => 0.9,
                TargetRole::Baseline => {
                    let rank = request
                        .target
                        .id
                        .strip_prefix("baseline-")
                        .context("unexpected baseline fixture id")?
                        .parse::<u64>()?;
                    0.4 + rank as f64 / 100.0
                }
            };
            Ok(EvaluationMeasurement {
                score,
                gpu_hours: request.max_gpu_hours,
                examples: 32,
                metrics: BTreeMap::new(),
            })
        }
    }

    struct Fixture {
        workflow: ResolvedWorkflow,
        checkpoint: ImmutableModelCheckpoint,
        artifact_directory: PathBuf,
        candidate_weights: PathBuf,
        resources_path: PathBuf,
    }

    fn resource_comparison(selected: &BenchmarkRun) -> ResourceComparison {
        use crate::metrics::{
            MetricContext, MetricEvent, MetricPhase, MetricPhaseKind, MetricWriter,
            ThroughputMetrics,
        };

        let baseline = &selected.metadata.baseline;
        let candidate = &selected.metadata.candidate;
        let final_manifest = &candidate.checkpoint_manifest;
        let final_generation = final_manifest.parent().unwrap();
        let evidence_vault = final_generation
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let exact_root = final_generation
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("exact-resume");
        let resumed_generation = exact_root
            .join("resumed")
            .join("generations")
            .join(final_generation.file_name().unwrap());
        fs::create_dir_all(&resumed_generation).unwrap();
        for entry in fs::read_dir(final_generation).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), resumed_generation.join(entry.file_name())).unwrap();
        }
        let resumed_manifest = resumed_generation.join("generation-manifest.json");
        let mut interrupted: serde_json::Value =
            serde_json::from_slice(&fs::read(final_manifest).unwrap()).unwrap();
        interrupted["global_step"] = serde_json::json!(10);
        let interrupted_bytes = serde_json::to_vec(&interrupted).unwrap();
        let interrupted_generation = exact_root
            .join("interrupted")
            .join("generations")
            .join(format!("sha256-{}", raw_sha256(&interrupted_bytes)));
        fs::create_dir_all(&interrupted_generation).unwrap();
        for entry in fs::read_dir(final_generation).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name() != "generation-manifest.json" {
                fs::copy(entry.path(), interrupted_generation.join(entry.file_name())).unwrap();
            }
        }
        let interrupted_manifest = interrupted_generation.join("generation-manifest.json");
        fs::write(&interrupted_manifest, interrupted_bytes).unwrap();
        let write_metrics = |directory: &Path, run_id: &str, elapsed: f64, timestamp: u64| {
            fs::create_dir_all(directory).unwrap();
            let path = directory.join("metrics.jsonl");
            let mut writer = MetricWriter::create(&path, run_id).unwrap();
            writer
                .append_at(
                    MetricContext {
                        global_step: 20,
                        phase: MetricPhase {
                            index: 0,
                            name: "test".into(),
                            kind: MetricPhaseKind::Pretrain,
                        },
                        checkpoint_hash: None,
                    },
                    MetricEvent::Throughput(ThroughputMetrics {
                        optimizer_steps: 20,
                        compute_tokens: 200,
                        supervised_tokens: 200,
                        examples: 20,
                        elapsed_seconds: elapsed,
                        tokens_per_second: 200.0 / elapsed,
                        examples_per_second: 20.0 / elapsed,
                        input_wait_seconds: 0.0,
                        host_to_device_seconds: 0.0,
                        gpu_busy_seconds: elapsed,
                    }),
                    timestamp,
                )
                .unwrap();
            writer.sync_all().unwrap();
            drop(writer);
            path
        };
        let uninterrupted_metrics =
            write_metrics(&exact_root.join("uninterrupted"), "uninterrupted", 2.0, 100);
        let resumed_metrics = write_metrics(&exact_root.join("resumed"), "resumed", 3.0, 200);
        let relative = |path: &Path| path.strip_prefix(evidence_vault).unwrap().to_path_buf();
        ResourceComparison {
            version: RESOURCE_COMPARISON_VERSION,
            baseline_id: baseline.id.clone(),
            candidate_id: candidate.id.clone(),
            strongest_baseline_id: baseline.id.clone(),
            measurement_evaluator_id: "summa-resource-evaluator".into(),
            wake_trials: (0..3)
                .map(|trial| PairedWakeTrial {
                    trial,
                    baseline: WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 10.0,
                        request_latency_ms: vec![10.0],
                    },
                    candidate: WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 10.0,
                        request_latency_ms: vec![10.0],
                    },
                })
                .collect(),
            candidate_capacity: (0..=3)
                .map(|completed_sleep_cycles| CapacityObservation {
                    completed_sleep_cycles,
                    routed_active_parameters: candidate.routed_active_parameters,
                    stored_parameters: candidate.parameters,
                    stored_bytes: candidate.stored_bytes,
                })
                .collect(),
            grouped_mm_parity: KernelParityEvidence {
                samples: (0..1024)
                    .map(|_| KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0,
                    })
                    .collect(),
            },
            pytorch_parity: KernelParityEvidence {
                samples: (0..1024)
                    .map(|_| KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0,
                    })
                    .collect(),
            },
            exact_resume: ExactResumeEvidence {
                interrupted_checkpoint: ExactResumeArtifact {
                    path: relative(&interrupted_manifest),
                },
                uninterrupted_final_state: ExactResumeArtifact {
                    path: relative(final_manifest),
                },
                resumed_final_state: ExactResumeArtifact {
                    path: relative(&resumed_manifest),
                },
                uninterrupted_metrics: ExactResumeArtifact {
                    path: relative(&uninterrupted_metrics),
                },
                resumed_metrics: ExactResumeArtifact {
                    path: relative(&resumed_metrics),
                },
                interruption_step: 10,
                resumed_from_step: 10,
            },
        }
    }

    fn fixture() -> Fixture {
        let temporary = tempfile::tempdir().unwrap().keep();
        let public = catalog_suite(&temporary, "public", SuiteVisibility::Public);
        let sealed = catalog_suite(&temporary, "sealed", SuiteVisibility::Sealed);
        let candidate = target(&temporary, "candidate");
        let candidate_weights = candidate
            .checkpoint_manifest
            .parent()
            .unwrap()
            .join("weights.safetensors");
        let mut run_references = Vec::new();
        let mut selected_run = None;
        for (index, ablation) in AblationId::ALL.into_iter().enumerate() {
            let baseline = target(&temporary, &format!("baseline-{index}"));
            let run = BenchmarkRunner {
                config: BenchmarkRunConfig {
                    evaluator_id: "promotion-test".into(),
                    evaluator_version:
                        "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                            .into(),
                    evaluator_arguments: Vec::new(),
                    paired_seeds: vec![11, 22, 33],
                    order_seed: 7,
                    gpu_hours_per_evaluation: 0.01,
                    ablation: Some(ablation),
                },
            }
            .run(
                &[public.clone(), sealed.clone()],
                &baseline,
                &candidate,
                &mut PassingEvaluator,
            )
            .unwrap();
            let path = temporary.join(format!("run-{index}.json"));
            write_json(&path, &run);
            if index + 1 == AblationId::ALL.len() {
                selected_run = Some(run);
            }
            run_references.push(PromotionEvidenceRef { path });
        }
        let selected_run = selected_run.unwrap();
        let selected_reference = run_references.pop().unwrap();
        let policy_path = temporary.join("policy.json");
        write_json(&policy_path, &AcceptancePolicy::default());
        let resources_path = temporary.join("resources.json");
        write_json(&resources_path, &resource_comparison(&selected_run));
        let artifact_directory = temporary.join("decisions");
        let config = PromotionConfig {
            selected_run: selected_reference,
            comparison_runs: run_references,
            resources: PromotionEvidenceRef {
                path: resources_path.clone(),
            },
            policy: PromotionEvidenceRef { path: policy_path },
            artifact_directory: artifact_directory.clone(),
        };
        let workflow: WorkflowV2 = serde_json::from_value(serde_json::json!({
            "version": 2,
            "name": "native-promotion-test",
            "phases": [{
                "name": "promotion",
                "type": "promotion",
                "promotion": config
            }]
        }))
        .unwrap();
        let workflow = workflow.resolve(&temporary.join("workflow.json")).unwrap();
        let checkpoint = ImmutableModelCheckpoint::new(
            "checkpoint://candidate",
            format!(
                "sha256:{}",
                selected_run.metadata.candidate.checkpoint_manifest_sha256
            ),
        )
        .unwrap();
        Fixture {
            workflow,
            checkpoint,
            artifact_directory,
            candidate_weights,
            resources_path,
        }
    }

    #[derive(Default)]
    struct FailPreparedOnce {
        fail: bool,
    }

    impl RuntimeCheckpoint for FailPreparedOnce {
        fn persist(&mut self, boundary: &RuntimeBoundary, _state: &WorkflowRunState) -> Result<()> {
            if self.fail && matches!(boundary, RuntimeBoundary::PhasePrepared { .. }) {
                self.fail = false;
                bail!("simulated failure after immutable decision publication")
            }
            Ok(())
        }
    }

    #[test]
    fn runtime_retries_the_same_verified_receipt_without_external_promotion() {
        let fixture = fixture();
        let mut state =
            WorkflowRunState::new(&fixture.workflow, Some(fixture.checkpoint.clone())).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(PhaseKind::Promotion, NativePromotionExecutor)
            .unwrap();
        let mut persistence = FailPreparedOnce { fail: true };
        let error = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("simulated failure"), "{error}");
        assert_eq!(
            fs::read_dir(&fixture.artifact_directory).unwrap().count(),
            1
        );

        let status = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut persistence,
        )
        .unwrap();
        assert!(matches!(
            status,
            RuntimeStatus::PhaseCommitted {
                workflow_complete: true,
                ..
            }
        ));
        assert_eq!(
            fs::read_dir(&fixture.artifact_directory).unwrap().count(),
            1
        );
        let receipt = match state.completed_phases()[0].product() {
            PhaseProduct::Release { candidate, receipt } => {
                assert_eq!(candidate, &fixture.checkpoint);
                receipt
            }
            other => panic!("unexpected product: {other:?}"),
        };
        let bytes = fs::read(receipt.uri()).unwrap();
        assert_eq!(receipt.sha256(), format!("sha256:{}", raw_sha256(&bytes)));
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["version"], PROMOTION_DECISION_VERSION);
        assert_eq!(json["report"]["accepted"], true);
        assert_eq!(json["candidate"]["sha256"], fixture.checkpoint.sha256());
        assert_eq!(json["benchmark_candidate_id"], "candidate");

        let mut changed = fixture.workflow.clone();
        changed.phases[0]
            .promotion
            .as_mut()
            .unwrap()
            .artifact_directory
            .push("changed");
        let error = state.validate(&changed).unwrap_err().to_string();
        assert!(error.contains("does not belong"), "{error}");
    }

    #[test]
    fn promotion_rejects_a_candidate_whose_weights_are_missing() {
        let fixture = fixture();
        fs::remove_file(&fixture.candidate_weights).unwrap();
        let mut state = WorkflowRunState::new(&fixture.workflow, Some(fixture.checkpoint)).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(PhaseKind::Promotion, NativePromotionExecutor)
            .unwrap();
        let error = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut FailPreparedOnce::default(),
        )
        .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("model weights"), "{error}");
        assert!(!fixture.artifact_directory.exists());
    }

    #[test]
    fn rejected_report_is_immutable_but_never_becomes_a_release() {
        let fixture = fixture();
        let mut resources: ResourceComparison =
            serde_json::from_slice(&fs::read(&fixture.resources_path).unwrap()).unwrap();
        for trial in &mut resources.wake_trials {
            trial.candidate.request_latency_ms = vec![100.0];
        }
        write_json(&fixture.resources_path, &resources);
        let mut state = WorkflowRunState::new(&fixture.workflow, Some(fixture.checkpoint)).unwrap();
        let mut registry = ExecutorRegistry::new();
        registry
            .register(PhaseKind::Promotion, NativePromotionExecutor)
            .unwrap();
        let error = run_next_phase(
            &fixture.workflow,
            &mut state,
            &mut registry,
            &mut (),
            &mut FailPreparedOnce::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("failed native promotion gates"), "{error}");
        assert!(state.completed_phases().is_empty());
        let path = fs::read_dir(&fixture.artifact_directory)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let json: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(json["report"]["accepted"], false);
    }
}
