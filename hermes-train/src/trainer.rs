//! Objective-aware streaming optimization loop.

use super::*;
use std::collections::VecDeque;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

const TRAINER_LOCK_FILE: &str = ".trainer.lock";
const MAX_PREFETCHED_SAMPLES: usize = 4_096;

fn training_prefetch_capacity(batch_size: usize) -> Result<usize> {
    ensure!(batch_size > 0, "training batch size must be positive");
    Ok(batch_size.saturating_mul(2).min(MAX_PREFETCHED_SAMPLES))
}

fn push_bounded_wake_context(
    contexts: &mut VecDeque<Vec<i64>>,
    context: Vec<i64>,
    maximum_records: usize,
) {
    debug_assert!(maximum_records > 0);
    if contexts.len() == maximum_records {
        contexts.pop_front();
    }
    contexts.push_back(context);
}

/// Process-lifetime ownership of one mutable training output root.
///
/// The relaunch supervisor has its own orchestration lock, but direct trainer
/// invocations must obey the same single-writer invariant. Without this lock,
/// two metric writers can truncate/append the same journal and produce
/// duplicated sequences or sparse NUL-filled holes.
struct TrainingOutputLock {
    file: fs::File,
}

impl TrainingOutputLock {
    fn acquire(output: &Path) -> Result<Self> {
        fs::create_dir_all(output)
            .with_context(|| format!("creating training output {}", output.display()))?;
        let output_metadata = fs::symlink_metadata(output)
            .with_context(|| format!("inspecting training output {}", output.display()))?;
        ensure!(
            output_metadata.is_dir() && !output_metadata.file_type().is_symlink(),
            "training output {} must be a real directory",
            output.display()
        );
        let path = output.join(TRAINER_LOCK_FILE);
        let mut options = fs::OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        options
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let mut file = options
            .open(&path)
            .with_context(|| format!("opening trainer output lock {}", path.display()))?;
        ensure!(
            file.metadata()?.is_file(),
            "trainer output lock {} is not a regular file",
            path.display()
        );

        #[cfg(unix)]
        {
            // SAFETY: `file` owns a valid descriptor for the lifetime of this
            // guard. LOCK_NB makes contention fail instead of hanging a boot
            // supervisor indefinitely.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                let error = std::io::Error::last_os_error();
                if error
                    .raw_os_error()
                    .is_some_and(|code| code == libc::EWOULDBLOCK || code == libc::EAGAIN)
                {
                    bail!(
                        "another hermes-train process already owns output {}",
                        output.display()
                    );
                }
                return Err(error)
                    .with_context(|| format!("locking trainer output root {}", output.display()));
            }
        }
        #[cfg(not(unix))]
        bail!("exclusive trainer output locks are not implemented on this platform");

        file.set_len(0)?;
        writeln!(file, "{}", std::process::id())?;
        file.sync_data()?;
        Ok(Self { file })
    }
}

impl Drop for TrainingOutputLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the descriptor stays valid until after Drop returns.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn publish_sleep_model(
    model: &Transformer,
    root: &Path,
) -> Result<hermes_train::native_sleep::NativeCheckpointRef> {
    fs::create_dir_all(root)
        .with_context(|| format!("creating sleep checkpoint store {}", root.display()))?;
    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "sleep checkpoint store {} is not a real directory",
        root.display()
    );
    let staging_directory = (0_u32..128)
        .find_map(|attempt| {
            let path = root.join(format!(".staging-{}-{attempt}", std::process::id()));
            fs::create_dir(&path).ok().map(|()| path)
        })
        .context("could not allocate a sleep checkpoint staging directory")?;
    let temporary = staging_directory.join("weights.safetensors");
    let publication = (|| -> Result<hermes_train::native_sleep::NativeCheckpointRef> {
        save_safetensors(&model.clone().valid(), &temporary)?;
        fs::File::open(&temporary)?.sync_all()?;
        let sha256 = file_sha256(&temporary)?;
        let digest = sha256
            .strip_prefix("sha256:")
            .expect("file_sha256 returns a prefixed digest");
        let published = root.join(format!("sha256-{digest}.safetensors"));
        match fs::hard_link(&temporary, &published) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                ensure!(
                    file_sha256(&published)? == sha256,
                    "content-addressed sleep checkpoint collision at {}",
                    published.display()
                );
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("publishing sleep checkpoint {}", published.display())
                });
            }
        }
        fs::remove_file(&temporary)?;
        fs::remove_dir(&staging_directory)?;
        fs::File::open(root)?.sync_all()?;
        hermes_train::native_sleep::NativeCheckpointRef::new(
            published
                .to_str()
                .context("sleep checkpoint path is not UTF-8")?,
            sha256,
        )
    })();
    if publication.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = fs::remove_dir(&staging_directory);
    }
    publication
}

fn publish_wake_journal(
    state: &TrainingState,
    source: &hermes_train::native_sleep::NativeCheckpointRef,
    root: &Path,
) -> Result<hermes_train::builtin_sleep_adapters::PinnedLocalArtifact> {
    ensure!(
        !state.wake_context_buffer.is_empty(),
        "periodic sleep boundary has no model-owned wake contexts"
    );
    fs::create_dir_all(root)?;
    let mut journal =
        hermes_train::builtin_sleep_adapters::WakeContextJournal::new(&source.sha256)?;
    for record in &state.wake_context_buffer {
        journal.push(record.clone())?;
    }
    let path = root.join(format!("boundary-step-{}.json", state.global_step));
    let pinned = journal.publish(&path)?;
    Ok(hermes_train::builtin_sleep_adapters::PinnedLocalArtifact {
        path: pinned.path().to_owned(),
        sha256: pinned.sha256().to_owned(),
    })
}

struct TrainerSleepProgress<'a> {
    state: &'a mut TrainingState,
    model_template: &'a Transformer,
    adamw: &'a AdamWOptimizer,
    muon: &'a BatchedMuon,
    metrics: &'a mut MetricWriter,
    output: &'a Path,
    metric_context: MetricContext,
}

impl hermes_train::native_sleep::NativeSleepProgressSink for TrainerSleepProgress<'_> {
    fn persist(
        &mut self,
        checkpoint: &hermes_train::native_sleep::NativeSleepCheckpoint,
    ) -> Result<()> {
        let mut checkpoint_model = self.model_template.clone();
        let checkpoint_path = Path::new(&checkpoint.live_checkpoint.uri);
        let checkpoint_bytes =
            read_pinned_checkpoint_bytes(checkpoint_path, &checkpoint.live_checkpoint.sha256)?;
        hermes_llm::load_safetensors_bytes(
            &mut checkpoint_model,
            checkpoint_bytes,
            &format!("sleep checkpoint {}", checkpoint_path.display()),
        )?;
        let mut staged_state = self.state.clone();
        staged_state.sleep = Some(checkpoint.clone());
        staged_state.metric_records = self.metrics.state().records;
        let _ = save_training_checkpoint_with_evidence(
            &checkpoint_model,
            self.adamw,
            self.muon,
            &staged_state,
            self.metrics,
            self.output,
        )?;
        *self.state = staged_state;
        Ok(())
    }

    fn metric(
        &mut self,
        _: &hermes_train::native_sleep::NativeSleepCheckpoint,
        event: MetricEvent,
    ) -> Result<()> {
        self.metrics.append(self.metric_context.clone(), event)?;
        self.state.metric_records = self.metrics.state().records;
        self.metrics.flush()
    }
}

struct PeriodicTrainingRuntime {
    factory: BuiltinSleepPhaseContextFactory,
    bank: TierOptimizerBank,
    model_store: PathBuf,
    journal_store: PathBuf,
    config_path: PathBuf,
    config_sha256: String,
}

/// Schedule-matched no-sleep ablation runtime. Tier optimizers and pending
/// accumulators are identical in shape and cadence to periodic sleep, while
/// due base updates commit directly and reserve state is never mutated.
struct WakeOnlyMemoryRuntime {
    config: MemoryUpdateMode,
    bank: TierOptimizerBank,
    publisher: DurableTierOptimizerPublisher,
}

impl WakeOnlyMemoryRuntime {
    fn load(
        workflow: &ResolvedWakePlan,
        model: &Transformer,
        output: &Path,
        device: &hermes_llm::Device,
    ) -> Result<Option<Self>> {
        let Some(config) = workflow
            .phases
            .first()
            .and_then(|phase| phase.memory_update_mode.clone())
        else {
            return Ok(None);
        };
        let bank =
            TierOptimizerBank::new(model, config.schedule(), config.tier_optimizer().clone())?;
        let output = fs::canonicalize(output)
            .with_context(|| format!("canonicalizing training output {}", output.display()))?;
        let publisher = DurableTierOptimizerPublisher::new(
            bank.clone(),
            output.join("wake-only-tier-optimizers"),
            TensorTransactionStore::new(output.join("wake-only-tensor-transactions")),
            model.config().clone(),
            device.clone(),
        )?;
        Ok(Some(Self {
            config,
            bank,
            publisher,
        }))
    }

    fn restore(&self, state: &TrainingState, model: &Transformer) -> Result<()> {
        let TrainingMemoryUpdateState::WakeOnly {
            config,
            optimizer_scopes,
        } = &state.memory_update
        else {
            bail!("wake_only runtime cannot restore a checkpoint from another memory mode")
        };
        ensure!(
            config == &self.config,
            "wake_only checkpoint configuration differs from the workflow"
        );
        self.publisher.restore_scopes(optimizer_scopes, model)
    }

    fn checkpoint(&self, state: &mut TrainingState) -> Result<()> {
        state.memory_update = TrainingMemoryUpdateState::WakeOnly {
            config: self.config.clone(),
            optimizer_scopes: self.publisher.publish_checkpoint_scopes()?,
        };
        Ok(())
    }
}

/// Fake-quantized parameter leaves shared by the microbatches that contribute
/// to one master-weight update. A window must never cross an optimizer step.
struct StagedQuantizationWindow {
    format: UltraQuantFormat,
    model: Transformer,
    tensor_count: u64,
}

fn quantization_state_format(format: Option<UltraQuantFormat>) -> String {
    match format {
        None => "full_precision",
        Some(UltraQuantFormat::BinaryG128) => "binary_g128",
        Some(UltraQuantFormat::TernaryG128) => "ternary_g128",
        Some(UltraQuantFormat::TernaryEntropyG128) => "ternary_entropy_g128",
    }
    .to_owned()
}

enum PrefetchedSample {
    CursorReady,
    Sample(data::TrainingSample),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResumeCursorAction {
    Skip,
    CursorReady,
    Emit,
}

fn resume_cursor_action(visited: usize, records_to_skip: usize) -> ResumeCursorAction {
    match visited.cmp(&records_to_skip) {
        std::cmp::Ordering::Less => ResumeCursorAction::Skip,
        std::cmp::Ordering::Equal => ResumeCursorAction::CursorReady,
        std::cmp::Ordering::Greater => ResumeCursorAction::Emit,
    }
}

impl PeriodicTrainingRuntime {
    fn load(
        args: &TrainArgs,
        workflow: &ResolvedWakePlan,
        signature: &str,
        model: &Transformer,
        device: &hermes_llm::Device,
    ) -> Result<Option<Self>> {
        let sleep = workflow
            .phases
            .first()
            .and_then(|phase| phase.periodic_sleep.as_ref());
        let runtime = args
            .sleep_runtime
            .as_ref()
            .zip(args.sleep_runtime_sha256.as_deref());
        match (sleep, runtime) {
            (None, None) => return Ok(None),
            (None, Some(_)) => {
                bail!("--sleep-runtime is only valid for a workflow with periodic_sleep")
            }
            (Some(_), None) => {
                bail!(
                    "a workflow with periodic_sleep requires --sleep-runtime and --sleep-runtime-sha256"
                )
            }
            (Some(_), Some(_)) => {}
        }
        let (path, sha256) = runtime.expect("matched present runtime");
        let factory =
            BuiltinSleepPhaseContextFactory::load_periodic_on_device(path, sha256, device.clone())?;
        ensure!(
            factory.config().workflow_signature == signature,
            "periodic sleep runtime belongs to workflow {}, current run is {signature}",
            factory.config().workflow_signature
        );
        let sleep = sleep.expect("matched present sleep config");
        validate_periodic_runtime_binding(&factory, sleep)?;
        let bank = TierOptimizerBank::new(
            model,
            &sleep.schedule,
            factory.config().tier_optimizer.clone(),
        )?;
        Ok(Some(Self {
            factory,
            bank,
            model_store: args.output.join("sleep-models"),
            journal_store: args.output.join("sleep-wake-contexts"),
            config_path: fs::canonicalize(path)
                .with_context(|| format!("canonicalizing periodic runtime {}", path.display()))?,
            config_sha256: sha256.to_owned(),
        }))
    }

    fn bind_new_checkpoint_artifact(&self, state: &mut TrainingState) -> Result<()> {
        bind_sleep_runtime_artifact(&mut state.artifacts, &self.config_path, &self.config_sha256)
    }

    fn new_cursor(
        &self,
        workflow_signature: &str,
        phase_name: &str,
        model: &Transformer,
        config: &hermes_train::workflow::InModelSleepConfig,
    ) -> Result<NativeSleepCheckpoint> {
        let checkpoint = publish_sleep_model(model, &self.model_store)?;
        NativeSleepCheckpoint::new(
            workflow_signature,
            phase_name,
            checkpoint,
            model,
            config,
            self.factory.config().rng_streams,
        )
    }

    fn restore_bank(
        &self,
        checkpoint: &NativeSleepCheckpoint,
        model: &Transformer,
        config: &hermes_train::workflow::InModelSleepConfig,
    ) -> Result<()> {
        let mut driver = BuiltinPeriodicSleepBoundaryDriver::resume(
            &self.factory,
            self.bank.clone(),
            checkpoint.live_checkpoint.clone(),
            model.clone(),
        )?;
        driver.restore_wake_scopes(checkpoint, config)
    }

    fn checkpoint_wake(
        &self,
        checkpoint: &mut NativeSleepCheckpoint,
        model: &Transformer,
        config: &hermes_train::workflow::InModelSleepConfig,
    ) -> Result<BuiltinPeriodicSleepBoundaryDriver> {
        let reference = publish_sleep_model(model, &self.model_store)?;
        checkpoint.record_wake_checkpoint(reference.clone())?;
        let mut driver = BuiltinPeriodicSleepBoundaryDriver::resume(
            &self.factory,
            self.bank.clone(),
            reference,
            model.clone(),
        )?;
        driver.checkpoint_wake_scopes(checkpoint, config)?;
        Ok(driver)
    }

    fn bind_boundary_journal(
        &self,
        driver: &mut BuiltinPeriodicSleepBoundaryDriver,
        checkpoint: &NativeSleepCheckpoint,
        state: &TrainingState,
        model: &Transformer,
    ) -> Result<()> {
        let journal =
            publish_wake_journal(state, &checkpoint.live_checkpoint, &self.journal_store)?;
        driver.bind_wake_boundary(checkpoint.live_checkpoint.clone(), model.clone(), journal)
    }
}

fn bind_sleep_runtime_artifact(
    artifacts: &mut Vec<ArtifactRef>,
    config_path: &Path,
    config_sha256: &str,
) -> Result<()> {
    ensure!(
        !artifacts
            .iter()
            .any(|artifact| artifact.kind == "sleep_runtime"),
        "new memory-training state already contains a sleep_runtime artifact"
    );
    artifacts.push(ArtifactRef {
        kind: "sleep_runtime".into(),
        manifest: config_path
            .to_str()
            .context("periodic runtime path is not UTF-8")?
            .to_owned(),
        hash: config_sha256.to_owned(),
    });
    Ok(())
}

fn validate_sleep_runtime_artifact(
    artifacts: &[ArtifactRef],
    config_path: &Path,
    config_sha256: &str,
) -> Result<()> {
    let matches = artifacts
        .iter()
        .filter(|artifact| artifact.kind == "sleep_runtime")
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "memory-training checkpoint must contain exactly one sleep_runtime artifact"
    );
    let saved = matches[0];
    ensure!(
        Path::new(&saved.manifest) == config_path && saved.hash == config_sha256,
        "supplied periodic runtime differs from the exact path/digest recorded by the checkpoint"
    );
    Ok(())
}

fn preflight_resumed_sleep_runtime(
    args: &TrainArgs,
    workflow: &ResolvedWakePlan,
    state: &TrainingState,
) -> Result<()> {
    if workflow
        .phases
        .first()
        .and_then(|phase| phase.periodic_sleep.as_ref())
        .is_none()
    {
        return Ok(());
    }
    let path = args
        .sleep_runtime
        .as_ref()
        .context("memory-training resume has no --sleep-runtime")?;
    let sha256 = args
        .sleep_runtime_sha256
        .as_deref()
        .context("memory-training resume has no --sleep-runtime-sha256")?;
    let canonical = fs::canonicalize(path)
        .with_context(|| format!("canonicalizing periodic runtime {}", path.display()))?;
    validate_sleep_runtime_artifact(&state.artifacts, &canonical, sha256)
}

pub(super) fn preflight_resumed_memory_mode(
    workflow: &ResolvedWakePlan,
    state: &TrainingState,
) -> Result<()> {
    let periodic = workflow
        .phases
        .first()
        .and_then(|phase| phase.periodic_sleep.as_ref());
    let wake_only = workflow
        .phases
        .first()
        .and_then(|phase| phase.memory_update_mode.as_ref());
    match (&state.memory_update, periodic, wake_only) {
        (TrainingMemoryUpdateState::Ordinary, None, None)
        | (TrainingMemoryUpdateState::PeriodicSleep, Some(_), None) => Ok(()),
        (TrainingMemoryUpdateState::WakeOnly { config, .. }, None, Some(expected)) => {
            ensure!(
                config == expected,
                "wake_only checkpoint configuration differs from the exact workflow"
            );
            Ok(())
        }
        _ => bail!("checkpoint memory update mode differs from the exact workflow"),
    }
}

fn validate_periodic_runtime_binding(
    factory: &BuiltinSleepPhaseContextFactory,
    sleep: &hermes_train::workflow::InModelSleepConfig,
) -> Result<()> {
    let runtime = factory.config();
    ensure!(
        sleep.retention_suite == runtime.retention_suite.path
            && sleep.retention_suite_sha256 == runtime.retention_suite.sha256
            && sleep.imitation.semantic_judge_hash == runtime.semantic_judge.sha256
            && sleep.retention.evaluator_hash == runtime.retention_evaluator.sha256,
        "periodic sleep evaluators differ from the pinned runtime"
    );
    ensure!(
        fs::canonicalize(&sleep.candidate_directory).with_context(|| format!(
            "canonicalizing periodic candidate directory {}",
            sleep.candidate_directory.display()
        ))? == fs::canonicalize(&runtime.candidate_directory).with_context(|| format!(
            "canonicalizing runtime candidate directory {}",
            runtime.candidate_directory.display()
        ))?,
        "periodic sleep candidate directory differs from the pinned runtime"
    );
    match (&sleep.dreaming, &runtime.dreaming) {
        (None, None) => {}
        (Some(workflow), Some(runtime)) => ensure!(
            workflow.reference_set_hash == runtime.reference_set.sha256
                && workflow.trial_evaluator_hash == runtime.independent_evaluation_set.sha256,
            "periodic Dreaming evaluators differ from the pinned runtime"
        ),
        _ => bail!("periodic workflow and pinned runtime disagree about Dreaming"),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn advance_and_drain_periodic_sleep(
    runtime: &PeriodicTrainingRuntime,
    config: &hermes_train::workflow::InModelSleepConfig,
    phase_index: usize,
    phase: &super::wake::ResolvedWakePhase,
    state: &mut TrainingState,
    model: &mut Transformer,
    adamw: &AdamWOptimizer,
    muon: &BatchedMuon,
    metrics: &mut MetricWriter,
    output: &Path,
) -> Result<bool> {
    let clock = match config.schedule.clock {
        hermes_train::sleep::UpdateClock::OptimizerSteps => state.global_step as u64,
        hermes_train::sleep::UpdateClock::ModelTokens => state.tokens_seen as u64,
    };
    // Keep the last durable cursor installed until a newly persisted cursor
    // replaces it. Driver construction and boundary binding are fallible; a
    // failed attempt must not erase resumable state from the live trainer.
    let mut cursor = state
        .sleep
        .clone()
        .context("periodic training has no native sleep cursor")?;
    let continuing = cursor.sleep.pending.is_some() || cursor.sleep.next_due_sender().is_some();
    if !continuing {
        let mut advanced = cursor.clone();
        advanced.advance_clock(model, config, clock)?;
        if advanced.sleep.next_due_sender().is_none() {
            state.sleep = Some(advanced);
            return Ok(false);
        }
    }

    let sleep_started = Instant::now();
    let mut driver = if continuing {
        BuiltinPeriodicSleepBoundaryDriver::resume(
            &runtime.factory,
            runtime.bank.clone(),
            cursor.live_checkpoint.clone(),
            model.clone(),
        )?
    } else {
        let mut driver = runtime.checkpoint_wake(&mut cursor, model, config)?;
        runtime.bind_boundary_journal(&mut driver, &cursor, state, model)?;
        driver
    };
    let metric_context = MetricContext {
        global_step: state.global_step as u64,
        phase: MetricPhase {
            index: phase_index as u32,
            name: phase.name.clone(),
            kind: MetricPhaseKind::Sleep,
        },
        checkpoint_hash: Some(cursor.live_checkpoint.sha256.clone()),
    };
    let mut sink = TrainerSleepProgress {
        state,
        model_template: model,
        adamw,
        muon,
        metrics,
        output,
        metric_context,
    };
    let committed = drain_periodic_sleep_before_wake_step(
        &mut cursor,
        model,
        config,
        clock,
        &mut driver,
        &mut sink,
    )?;
    drop(sink);
    ensure!(
        committed.uri == driver.live_checkpoint().uri
            && committed.sha256 == driver.live_checkpoint().sha256,
        "periodic controller and tensor driver disagree on the committed checkpoint"
    );
    let sleep_elapsed = sleep_started.elapsed().as_secs_f64();
    *model = driver.live_model().clone();
    state.sleep = Some(cursor);
    state.wake_context_buffer.clear();
    metrics.append(
        MetricContext {
            global_step: state.global_step as u64,
            phase: MetricPhase {
                index: phase_index as u32,
                name: phase.name.clone(),
                kind: MetricPhaseKind::Sleep,
            },
            checkpoint_hash: Some(committed.sha256),
        },
        MetricEvent::PhaseTiming(PhaseTimingMetrics {
            boundary: PhaseBoundary::Completed,
            elapsed_seconds: sleep_elapsed,
            input_wait_seconds: 0.0,
            forward_seconds: 0.0,
            backward_seconds: 0.0,
            optimizer_seconds: 0.0,
            checkpoint_seconds: 0.0,
        }),
    )?;
    state.metric_records = metrics.state().records;
    metrics.flush()?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn publish_quantization_phase_candidate(
    model: &Transformer,
    plan: &WorkflowQuantizationPlan,
    phase_index: usize,
    phase: &super::wake::ResolvedWakePhase,
    state: &mut TrainingState,
    metrics: &mut MetricWriter,
    output: &Path,
    expected_source_checkpoint_sha256: Option<&str>,
) -> Result<()> {
    ensure!(
        state.sleep.as_ref().is_none_or(|checkpoint| {
            checkpoint.sleep.pending.is_none() && checkpoint.sleep.next_due_sender().is_none()
        }),
        "QAT candidate publication requires every due periodic-sleep boundary to be committed first"
    );
    let export_started = Instant::now();
    let key = format!(
        "phase-{phase_index:03}-step-{:012}-{}",
        state.global_step,
        stable_cache_id(&phase.name)
    );
    let candidate_store = output.join("quantized-candidates");
    let publication = match expected_source_checkpoint_sha256 {
        Some(source_sha256) => publish_qat_candidate_from_authenticated_source(
            model,
            &candidate_store,
            &key,
            &plan.recipe,
            source_sha256,
        )?,
        None => publish_qat_candidate(model, &candidate_store, &key, &plan.recipe)?,
    };
    if let Some(expected) = expected_source_checkpoint_sha256 {
        ensure!(
            publication.weights_sha256 == expected,
            "QAT candidate weights {} differ from the bound post-sleep checkpoint {expected}",
            publication.weights_sha256
        );
    }
    let manifest = publication
        .candidate_manifest_path
        .to_str()
        .context("QAT candidate manifest path is not UTF-8")?
        .to_owned();
    let quantization = state
        .quantization
        .as_mut()
        .context("quantization phase has no checkpoint quantization state")?;
    quantization.format = quantization_state_format(Some(plan.recipe.format));
    quantization.fake_quant_active = false;
    quantization.calibration_step = state.global_step as u64;
    quantization.manifest = Some(manifest.clone());
    quantization.candidate_weights_sha256 = Some(publication.weights_sha256.clone());
    if !state.artifacts.iter().any(|artifact| {
        artifact.kind == "hquant_candidate"
            && artifact.hash == publication.candidate_manifest_sha256
    }) {
        state.artifacts.push(ArtifactRef {
            kind: "hquant_candidate".into(),
            manifest,
            hash: publication.candidate_manifest_sha256.clone(),
        });
    }
    metrics.append(
        MetricContext {
            global_step: state.global_step as u64,
            phase: MetricPhase {
                index: phase_index as u32,
                name: phase.name.clone(),
                kind: MetricPhaseKind::Quantization,
            },
            checkpoint_hash: Some(publication.weights_sha256.clone()),
        },
        MetricEvent::Quantization(QuantizationMetrics {
            stage: QuantizationStage::Export,
            format: metric_quantization_format(plan.recipe.format),
            group_size: plan.recipe.group_size as u32,
            progress_fraction: 1.0,
            tensors_quantized: publication.metrics.quantized_tensors,
            weights_quantized: publication.metrics.quantized_elements,
            average_bits_per_weight: Some(publication.metrics.average_bits_per_weight),
            packed_bytes: Some(publication.metrics.packed_bytes),
            mean_squared_error: Some(publication.metrics.weighted_mean_squared_error),
            max_absolute_error: Some(publication.metrics.maximum_absolute_error),
            distillation_forward_kl: None,
            acceptance_delta: None,
            embeddings_quantized: plan.recipe.quantize_embeddings,
            lm_head_quantized: plan.recipe.quantize_lm_head,
        }),
    )?;
    metrics.append(
        MetricContext {
            global_step: state.global_step as u64,
            phase: MetricPhase {
                index: phase_index as u32,
                name: phase.name.clone(),
                kind: MetricPhaseKind::Quantization,
            },
            checkpoint_hash: Some(publication.weights_sha256),
        },
        MetricEvent::PhaseTiming(PhaseTimingMetrics {
            boundary: PhaseBoundary::Completed,
            elapsed_seconds: export_started.elapsed().as_secs_f64(),
            input_wait_seconds: 0.0,
            forward_seconds: 0.0,
            backward_seconds: 0.0,
            optimizer_seconds: 0.0,
            checkpoint_seconds: 0.0,
        }),
    )?;
    state.metric_records = metrics.state().records;
    metrics.flush()
}

fn bind_post_sleep_quantization_source(
    runtime: Option<&PeriodicTrainingRuntime>,
    phase: &super::wake::ResolvedWakePhase,
    state: &mut TrainingState,
    model: &Transformer,
) -> Result<Option<String>> {
    let Some(runtime) = runtime else {
        ensure!(
            state.sleep.is_none(),
            "ordinary QAT unexpectedly contains periodic-sleep state"
        );
        return Ok(None);
    };
    let sleep = phase
        .periodic_sleep
        .as_ref()
        .context("memory-model QAT phase has no periodic_sleep")?;
    let mut cursor = state
        .sleep
        .clone()
        .context("memory-model QAT phase has no native sleep cursor")?;
    ensure!(
        cursor.sleep.pending.is_none() && cursor.sleep.next_due_sender().is_none(),
        "cannot bind a QAT source checkpoint before every due periodic-sleep boundary commits"
    );
    let _ = runtime.checkpoint_wake(&mut cursor, model, sleep)?;
    let source = cursor.live_checkpoint.sha256.clone();
    state.sleep = Some(cursor);
    Ok(Some(source))
}

#[allow(clippy::too_many_arguments)]
fn publish_completed_quantization_candidate_if_pending(
    model: &Transformer,
    phase_index: usize,
    phase: &super::wake::ResolvedWakePhase,
    planned_steps: usize,
    state: &mut TrainingState,
    metrics: &mut MetricWriter,
    output: &Path,
    periodic_runtime: Option<&PeriodicTrainingRuntime>,
    authenticated_model_sha256: Option<&str>,
) -> Result<bool> {
    let Some(plan) = &phase.quantization else {
        return Ok(false);
    };
    if state.steps_in_phase < planned_steps
        || state
            .quantization
            .as_ref()
            .and_then(|quantization| quantization.manifest.as_ref())
            .is_some()
    {
        return Ok(false);
    }
    let sleep_source = bind_post_sleep_quantization_source(periodic_runtime, phase, state, model)?;
    if let (Some(sleep_source), Some(checkpoint_source)) =
        (sleep_source.as_deref(), authenticated_model_sha256)
    {
        ensure!(
            sleep_source == checkpoint_source,
            "post-sleep QAT source differs from the authenticated trainer checkpoint"
        );
    }
    let source = sleep_source.as_deref().or(authenticated_model_sha256);
    publish_quantization_phase_candidate(
        model,
        plan,
        phase_index,
        phase,
        state,
        metrics,
        output,
        source,
    )?;
    Ok(true)
}

fn verify_resumed_quantization_candidate(
    state: &TrainingState,
    phase: &super::wake::ResolvedWakePhase,
    planned_steps: usize,
    resumed_weights_sha256: &str,
) -> Result<()> {
    let Some(plan) = &phase.quantization else {
        ensure!(
            state.quantization.is_none(),
            "non-quantization resume contains quantization state"
        );
        return Ok(());
    };
    let quantization = state
        .quantization
        .as_ref()
        .context("quantization resume has no typed state")?;
    let state_step = u64::try_from(state.global_step)
        .context("quantization checkpoint global step exceeds u64")?;
    let forward_step = if state.steps_in_phase == 0 {
        state_step
    } else {
        state_step
            .checked_sub(1)
            .context("quantization checkpoint has phase steps but no global step")?
    };
    let candidate_published = quantization.manifest.is_some();
    ensure!(
        candidate_published == quantization.candidate_weights_sha256.is_some(),
        "quantization candidate manifest and source identity must appear together"
    );
    let expected_format = if candidate_published {
        Some(plan.recipe.format)
    } else {
        plan.format_at(forward_step)
    };
    ensure!(
        quantization.format == quantization_state_format(expected_format)
            && quantization.fake_quant_active
                == (!candidate_published && expected_format.is_some()),
        "quantization resume format differs from the configured optimizer-step schedule"
    );
    ensure!(
        quantization.calibration_step == state_step,
        "quantization resume calibration clock differs from the trainer global step"
    );
    let expected_teacher = match &plan.training {
        WorkflowQuantizationTraining::Qat { .. } => None,
        WorkflowQuantizationTraining::Distillation { teacher_sha256, .. } => {
            Some(teacher_sha256.as_str())
        }
    };
    ensure!(
        quantization.teacher_hash.as_deref() == expected_teacher,
        "quantization resume teacher identity differs from the workflow"
    );
    if state.steps_in_phase < planned_steps {
        ensure!(
            quantization.manifest.is_none(),
            "incomplete quantization phase already names a final candidate"
        );
        return Ok(());
    }
    if quantization.manifest.is_none() {
        // The final optimizer update and a coincident sleep boundary can be
        // durably committed before the write-once QAT export. Resume finishes
        // that publication from the authenticated post-sleep model.
        return Ok(());
    }
    ensure!(
        state.sleep.as_ref().is_none_or(|checkpoint| {
            checkpoint.sleep.pending.is_none() && checkpoint.sleep.next_due_sender().is_none()
        }),
        "completed QAT checkpoint published its candidate before a due periodic-sleep boundary"
    );
    let manifest = Path::new(
        quantization
            .manifest
            .as_deref()
            .context("completed quantization phase has no HQUANT candidate")?,
    );
    ensure!(
        manifest
            .file_name()
            .is_some_and(|name| name == "candidate.json"),
        "quantization candidate does not name candidate.json"
    );
    let candidate = manifest
        .parent()
        .context("quantization candidate manifest has no parent")?;
    let mut artifacts = state.artifacts.iter().filter(|artifact| {
        artifact.kind == "hquant_candidate" && artifact.manifest == manifest.to_string_lossy()
    });
    let artifact = artifacts
        .next()
        .context("completed quantization phase has no candidate artifact receipt")?;
    ensure!(
        artifacts.next().is_none(),
        "completed quantization phase repeats its candidate artifact receipt"
    );
    let publication = open_qat_candidate_addressed(candidate, &artifact.hash)?;
    ensure!(
        publication.candidate_manifest_path == manifest,
        "quantization candidate resolved to another path"
    );
    ensure!(
        publication.recipe == plan.recipe,
        "quantization candidate recipe differs from the workflow"
    );
    ensure!(
        publication.weights_sha256 == resumed_weights_sha256,
        "quantization candidate weights differ from the authenticated resume checkpoint"
    );
    ensure!(
        quantization.candidate_weights_sha256.as_deref() == Some(resumed_weights_sha256),
        "quantization candidate source identity differs from the authenticated resume checkpoint"
    );
    ensure!(
        artifact.manifest == manifest.to_string_lossy()
            && artifact.hash == publication.candidate_manifest_sha256,
        "quantization candidate receipt differs from the validated archive"
    );
    Ok(())
}

pub(super) fn train(args: TrainArgs) -> Result<()> {
    validate_train_args(&args)?;
    let workflow = resolve_wake_plan(&args)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let tokenizer_hash = file_sha256(&args.tokenizer)?;
    let mut config = load_config(&args.config)?;
    config.vocab_size = tokenizer.vocab_size();
    validate_model_wake_plan(&config, &workflow)?;

    // A signature-only invocation remains read-only and hardware-independent.
    // Real training, however, claims its output and initializes the selected
    // accelerator before hashing a potentially multi-billion-token corpus.
    // This both avoids duplicate startup work and reports a missing CUDA
    // driver immediately instead of after a long CPU-only verification pass.
    let training_runtime = if args.print_run_signature {
        None
    } else {
        let output_lock = TrainingOutputLock::acquire(&args.output)?;
        let device = hermes_llm::default_device().autodiff();
        device.seed(args.seed);
        Some((output_lock, device))
    };
    let mut data_binding_cache = HashMap::new();
    let data_bindings = workflow
        .phases
        .iter()
        .map(|phase| {
            bind_phase_data(
                &phase.data,
                &tokenizer,
                &tokenizer_hash,
                &mut data_binding_cache,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let data_manifests = data_bindings
        .iter()
        .map(|binding| {
            binding.ensure_still_published()?;
            Ok(binding.signature_identity().to_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    let initial_checkpoint_sha256 = args.checkpoint.as_deref().map(file_sha256).transpose()?;
    let signature = run_signature(
        &args,
        &workflow,
        &config,
        &data_manifests,
        initial_checkpoint_sha256.clone(),
    )?;
    for binding in &data_bindings {
        binding.ensure_still_published()?;
    }
    if args.print_run_signature {
        println!("{signature}");
        return Ok(());
    }
    let (_output_lock, device) = training_runtime
        .expect("non-signature training initialized its output lock and accelerator");
    let token_cache_root = args.output.join(".token-cache");
    fs::create_dir_all(&token_cache_root)?;
    let token_cache_paths = data_manifests
        .iter()
        .zip(&workflow.phases)
        .map(|(data, phase)| {
            token_cache_path(&token_cache_root, data, &phase.data, &tokenizer_hash)
        })
        .collect::<Result<Vec<_>>>()?;
    let (phase_plan, total_steps) =
        plan_training(&workflow, &tokenizer, &token_cache_paths, &data_bindings)?;
    ensure!(
        total_steps > 0,
        "training has zero complete optimizer steps"
    );
    let run_id = stable_cache_id(&signature);
    let metrics_path = args.output.join("metrics.jsonl");
    let mut initial_model = Transformer::new(&config, &device)?;
    if let Some(path) = &args.checkpoint {
        let expected = initial_checkpoint_sha256
            .as_deref()
            .expect("checkpoint hash was computed for a configured checkpoint");
        let bytes = read_pinned_checkpoint_bytes(path, expected)
            .context("initial checkpoint changed after its run signature was computed")?;
        hermes_llm::load_safetensors_bytes(
            &mut initial_model,
            bytes,
            &format!("initial checkpoint {}", path.display()),
        )?;
    }
    let mut muon_parameter_ids = initial_model.muon_parameter_ids();
    ensure!(
        !muon_parameter_ids.is_empty(),
        "model has no hidden matrix parameters for Muon"
    );
    let mut muon_optimizer = BatchedMuon::new(muon_parameter_ids.clone());
    let mut adamw_optimizer: AdamWOptimizer = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .with_weight_decay(args.weight_decay)
        .init();
    let (resume_state, resume_weights_sha256) = if args.resume {
        let (optimizer, state, weights_sha256) = load_training_state(
            &mut initial_model,
            adamw_optimizer,
            &mut muon_optimizer,
            &args.output,
            &device,
        )?;
        adamw_optimizer = optimizer;
        ensure!(
            state.global_step <= total_steps,
            "checkpoint step {} exceeds requested total {total_steps}",
            state.global_step
        );
        ensure!(
            state.phase < workflow.phases.len(),
            "checkpoint phase {} is outside the requested {}-phase workflow",
            state.phase,
            workflow.phases.len()
        );
        let resume_phase = &workflow.phases[state.phase];
        ensure!(
            state.phase_id == resume_phase.name
                && state.phase_kind == resume_phase.phase_kind.name(),
            "checkpoint phase identity differs from the requested workflow"
        );
        ensure!(
            state.epoch < resume_phase.epochs,
            "checkpoint epoch {} is outside workflow phase `{}` with {} epochs",
            state.epoch,
            resume_phase.name,
            resume_phase.epochs
        );
        ensure!(
            state.steps_in_phase <= phase_plan[state.phase].steps,
            "checkpoint has {} steps in phase `{}`, whose planned total is {}",
            state.steps_in_phase,
            resume_phase.name,
            phase_plan[state.phase].steps
        );
        ensure!(
            state.workflow_signature == signature,
            "checkpoint workflow or training configuration differs from this invocation"
        );
        ensure!(
            state.data_manifest_hash.as_ref() == Some(&data_manifests[state.phase]),
            "checkpoint data manifest differs from workflow phase `{}`",
            resume_phase.name
        );
        ensure!(
            state.global_step == 0 || state.tokens_seen > 0,
            "checkpoint has no cumulative token count and cannot safely resume"
        );
        validate_wake_rng_state(&state, args.seed)?;
        (Some(state), Some(weights_sha256))
    } else {
        (None, None)
    };
    if let Some(state) = &resume_state {
        preflight_resumed_memory_mode(&workflow, state)?;
        // Authenticate the exact runtime identity before constructing its
        // factory. Factory construction validates/creates configured stores,
        // so a mismatched resume must fail before that first side effect.
        preflight_resumed_sleep_runtime(&args, &workflow, state)?;
        verify_resumed_quantization_candidate(
            state,
            &workflow.phases[state.phase],
            phase_plan[state.phase].steps,
            resume_weights_sha256
                .as_deref()
                .expect("resume checkpoint has an authenticated weights identity"),
        )?;
    }
    let periodic_runtime =
        PeriodicTrainingRuntime::load(&args, &workflow, &signature, &initial_model, &device)?;
    let wake_only_runtime =
        WakeOnlyMemoryRuntime::load(&workflow, &initial_model, &args.output, &device)?;
    ensure!(
        periodic_runtime.is_none() || wake_only_runtime.is_none(),
        "one training run cannot combine periodic_sleep and wake_only runtimes"
    );
    let tier_bank = periodic_runtime
        .as_ref()
        .map(|runtime| &runtime.bank)
        .or_else(|| wake_only_runtime.as_ref().map(|runtime| &runtime.bank));
    if let Some(bank) = tier_bank {
        let wake_ids = bank
            .scopes()?
            .wake_parameter_ids
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        muon_parameter_ids = initial_model
            .muon_parameter_ids()
            .into_iter()
            .filter(|id| wake_ids.contains(&id.val()))
            .collect();
        ensure!(
            !muon_parameter_ids.is_empty(),
            "ordinary wake scope has no hidden matrix parameters for Muon"
        );
        muon_optimizer.set_parameter_ids(muon_parameter_ids.clone());
        if let (Some(runtime), Some(state)) = (&periodic_runtime, &resume_state) {
            let checkpoint = state
                .sleep
                .as_ref()
                .context("memory-training checkpoint has no native sleep cursor")?;
            let sleep = workflow.phases[state.phase]
                .periodic_sleep
                .as_ref()
                .expect("validated memory workflow has periodic sleep");
            if checkpoint.sleep.pending.is_none() && checkpoint.sleep.next_due_sender().is_none() {
                runtime.restore_bank(checkpoint, &initial_model, sleep)?;
            }
        }
        if let (Some(runtime), Some(state)) = (&wake_only_runtime, &resume_state) {
            runtime.restore(state, &initial_model)?;
        }
    } else {
        ensure!(
            resume_state.as_ref().is_none_or(|state| {
                state.sleep.is_none()
                    && matches!(state.memory_update, TrainingMemoryUpdateState::Ordinary)
            }),
            "ordinary-model checkpoint unexpectedly contains memory-training state"
        );
    }
    if let Some(state) = &resume_state {
        // Periodic-memory workflows narrow the global Muon selection to the
        // exact wake scope above. Validate archive geometry only after that
        // scope is known; validating against every model matrix would reject
        // correct checkpoints whose tier matrices use independent optimizers.
        muon_optimizer.validate_for_model(&initial_model, state.global_step == 0)?;
    }
    let mut metrics = if let Some(state) = &resume_state {
        MetricWriter::resume_from_checkpoint(
            &metrics_path,
            &run_id,
            state.metric_records,
            state.global_step as u64,
        )?
    } else {
        MetricWriter::create(&metrics_path, &run_id)?
    };
    let mut device_sampler = start_device_sampler(&args);

    fs::write(
        args.output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    fs::write(
        args.output.join("resolved-workflow.json"),
        serde_json::to_vec_pretty(&workflow)?,
    )?;

    let mut muon_accumulator = GradientsAccumulator::new();
    let mut adamw_accumulator = GradientsAccumulator::new();
    let mut tier_accumulators = tier_bank
        .map(|bank| {
            bank.scopes().map(|scopes| {
                (0..scopes.tiers.len())
                    .map(|_| GradientsAccumulator::new())
                    .collect::<Vec<_>>()
            })
        })
        .transpose()?
        .unwrap_or_default();
    let initial_parameter_ids = parameter_ids(&initial_model);
    // Optimizer::step consumes the module; Option lets the streaming callback
    // move it out and replace it without cloning model parameters.
    let mut model = Some(initial_model);
    let mut step = resume_state.as_ref().map_or(0, |state| state.global_step);
    let mut tokens_seen = resume_state.as_ref().map_or(0, |state| state.tokens_seen);
    let mut micro_step = 0;
    let mut loss_sum: Option<Tensor<1>> = None;
    let mut router_loss_sum: Option<Tensor<1>> = None;
    let mut distillation_loss_sum: Option<Tensor<1>> = None;
    let mut retrieval_correct_sum: Option<Tensor<1>> = None;
    let wake_context_limits = periodic_runtime.as_ref().map(|runtime| {
        (
            runtime.factory.config().max_wake_context_records,
            runtime.factory.config().rollouts.max_context_tokens,
        )
    });
    let mut step_wake_contexts = VecDeque::<Vec<i64>>::new();
    let mut step_stats = BatchStats::default();
    let mut optimizer_step_started = Instant::now();
    let mut step_input_wait_seconds = 0.0f64;
    let mut step_host_to_device_seconds = 0.0f64;
    let mut step_accelerator_seconds = 0.0f64;
    let mut training_state = resume_state.clone().unwrap_or(TrainingState {
        version: TRAINING_STATE_VERSION,
        global_step: 0,
        phase: 0,
        phase_id: workflow.phases[0].name.clone(),
        phase_kind: workflow.phases[0].phase_kind.name().into(),
        epoch: 0,
        records_in_phase: 0,
        steps_in_phase: 0,
        tokens_seen: 0,
        metric_records: 0,
        workflow_signature: signature.clone(),
        data_manifest_hash: Some(data_manifests[0].clone()),
        parameter_ids: initial_parameter_ids.clone(),
        optimizer_states: vec![OptimizerStateRef {
            scope: "wake".into(),
            adamw: "adamw-state.bpk".into(),
            muon: "muon-state.bpk".into(),
            gradient_accumulator: None,
            update_clock: 0,
        }],
        memory_update: if periodic_runtime.is_some() {
            TrainingMemoryUpdateState::PeriodicSleep
        } else if let Some(runtime) = &wake_only_runtime {
            TrainingMemoryUpdateState::WakeOnly {
                config: runtime.config.clone(),
                optimizer_scopes: runtime.bank.scopes()?,
            }
        } else {
            TrainingMemoryUpdateState::Ordinary
        },
        sleep: None,
        artifacts: Vec::new(),
        evaluator_hashes: Vec::new(),
        rng_streams: vec![
            RngStreamState {
                name: DATA_RNG_STREAM.into(),
                seed: args.seed,
                counter: 0,
            },
            RngStreamState {
                name: MODEL_RNG_STREAM.into(),
                seed: args.seed,
                counter: 0,
            },
        ],
        wake_context_buffer: Vec::new(),
        quantization: None,
    });
    if let Some(runtime) = &periodic_runtime {
        let sleep = workflow.phases[training_state.phase]
            .periodic_sleep
            .as_ref()
            .expect("validated memory workflow has periodic sleep");
        if training_state.sleep.is_none() {
            training_state.sleep = Some(runtime.new_cursor(
                &signature,
                &training_state.phase_id,
                model.as_ref().unwrap(),
                sleep,
            )?);
        }
        let max_records = runtime.factory.config().max_wake_context_records;
        ensure!(
            training_state.wake_context_buffer.len() <= max_records,
            "checkpoint wake-context ring exceeds configured bound {max_records}"
        );
        if resume_state.is_none() {
            runtime.bind_new_checkpoint_artifact(&mut training_state)?;
        }
        for hash in [
            &runtime.factory.config().semantic_judge.sha256,
            &runtime.factory.config().retention_evaluator.sha256,
            &runtime.factory.config().retention_suite.sha256,
        ] {
            if !training_state.evaluator_hashes.contains(hash) {
                training_state.evaluator_hashes.push(hash.clone());
            }
        }
        if let Some(dreaming) = &runtime.factory.config().dreaming {
            for hash in [
                &dreaming.reference_set.sha256,
                &dreaming.independent_evaluation_set.sha256,
            ] {
                if !training_state.evaluator_hashes.contains(hash) {
                    training_state.evaluator_hashes.push(hash.clone());
                }
            }
        }
        if training_state.sleep.as_ref().is_some_and(|checkpoint| {
            checkpoint.sleep.pending.is_some() || checkpoint.sleep.next_due_sender().is_some()
        }) {
            let phase_index = training_state.phase;
            let phase = &workflow.phases[phase_index];
            let sleep = phase
                .periodic_sleep
                .as_ref()
                .expect("validated memory phase has periodic sleep");
            let mut current = model.take().expect("training model is present");
            advance_and_drain_periodic_sleep(
                runtime,
                sleep,
                phase_index,
                phase,
                &mut training_state,
                &mut current,
                &adamw_optimizer,
                &muon_optimizer,
                &mut metrics,
                &args.output,
            )?;
            model = Some(current);
            let context = MetricContext {
                global_step: training_state.global_step as u64,
                phase: MetricPhase {
                    index: phase_index as u32,
                    name: phase.name.clone(),
                    kind: phase.phase_kind.into(),
                },
                checkpoint_hash: None,
            };
            drain_device_sampler(&mut device_sampler, &mut metrics, &context)?;
            training_state.metric_records = metrics.state().records;
            metrics.flush()?;
        }
    }

    // A crash can occur after the final optimizer update and its coincident
    // sleep transaction are durable but before the write-once QAT archive is
    // published. Finish that exact post-sleep publication before the phase is
    // considered skippable, then checkpoint its receipt immediately.
    if resume_state.is_some() {
        let phase_index = training_state.phase;
        let phase = &workflow.phases[phase_index];
        if publish_completed_quantization_candidate_if_pending(
            model.as_ref().unwrap(),
            phase_index,
            phase,
            phase_plan[phase_index].steps,
            &mut training_state,
            &mut metrics,
            &args.output,
            periodic_runtime.as_ref(),
            resume_weights_sha256.as_deref(),
        )? {
            training_state.metric_records = metrics.state().records;
            if let Some(runtime) = &wake_only_runtime {
                runtime.checkpoint(&mut training_state)?;
            }
            let publication = save_training_checkpoint_with_evidence(
                model.as_ref().unwrap(),
                &adamw_optimizer,
                &muon_optimizer,
                &training_state,
                &mut metrics,
                &args.output,
            )?;
            print_checkpoint_publication("checkpointed resumed QAT publication", &publication);
        }
    }

    let phase_summary = workflow
        .phases
        .iter()
        .zip(&phase_plan)
        .map(|(phase, plan)| {
            format!(
                "{}:{}:samples={}:steps={}",
                phase.name,
                phase.objective.name(),
                plan.samples
                    .map_or_else(|| "streaming".to_owned(), |n| n.to_string()),
                plan.steps,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "model={} params={} muon_matrices={} device={device:?} phases=[{phase_summary}] steps={total_steps}",
        config.name,
        model.as_ref().unwrap().num_parameters(),
        muon_parameter_ids.len(),
    );

    'phases: for (phase_index, phase) in workflow.phases.iter().enumerate() {
        if resume_state
            .as_ref()
            .is_some_and(|state| phase_index < state.phase)
        {
            continue;
        }
        let mut steps_in_phase = resume_state
            .as_ref()
            .filter(|state| state.phase == phase_index)
            .map_or(0, |state| state.steps_in_phase);
        if steps_in_phase >= phase_plan[phase_index].steps {
            continue;
        }
        let mut phase_limit_reached = false;
        let quantization_teacher =
            load_quantization_teacher(phase.quantization.as_ref(), &config, &device)?;

        for epoch in 0..phase.epochs {
            if resume_state
                .as_ref()
                .is_some_and(|state| phase_index == state.phase && epoch < state.epoch)
            {
                continue;
            }
            let records_to_skip = resume_state
                .as_ref()
                .filter(|state| state.phase == phase_index && state.epoch == epoch)
                .map_or(0, |state| state.records_in_phase);
            let mut records_in_phase = records_to_skip;
            let model_rng = rng_stream(&training_state, MODEL_RNG_STREAM)?.clone();
            let mut native_sleep = training_state.sleep.clone();
            if let Some(cursor) = &mut native_sleep {
                ensure!(
                    cursor.sleep.pending.is_none(),
                    "cannot enter workflow phase `{}` with an unfinished sleep transaction",
                    phase.name
                );
                cursor.phase_name = phase.name.clone();
            }
            let artifacts = training_state.artifacts.clone();
            let evaluator_hashes = training_state.evaluator_hashes.clone();
            let memory_update = training_state.memory_update.clone();
            training_state = TrainingState {
                version: TRAINING_STATE_VERSION,
                global_step: step,
                phase: phase_index,
                phase_id: phase.name.clone(),
                phase_kind: phase.phase_kind.name().into(),
                epoch,
                records_in_phase,
                steps_in_phase,
                tokens_seen,
                metric_records: metrics.state().records,
                workflow_signature: signature.clone(),
                data_manifest_hash: Some(data_manifests[phase_index].clone()),
                parameter_ids: initial_parameter_ids.clone(),
                optimizer_states: vec![OptimizerStateRef {
                    scope: "wake".into(),
                    adamw: "adamw-state.bpk".into(),
                    muon: "muon-state.bpk".into(),
                    gradient_accumulator: None,
                    update_clock: step as u64,
                }],
                memory_update,
                sleep: native_sleep,
                artifacts,
                evaluator_hashes,
                rng_streams: vec![
                    RngStreamState {
                        name: DATA_RNG_STREAM.into(),
                        seed: shuffle_seed(args.seed, phase_index, epoch),
                        counter: u64::try_from(records_to_skip)
                            .context("resume sample cursor exceeds u64")?,
                    },
                    model_rng,
                ],
                wake_context_buffer: training_state.wake_context_buffer.clone(),
                quantization: phase.quantization.as_ref().map(|plan| {
                    let format = plan.format_at(step as u64);
                    QuantizationTrainingState {
                        format: quantization_state_format(format),
                        fake_quant_active: format.is_some(),
                        calibration_step: step as u64,
                        manifest: None,
                        candidate_weights_sha256: None,
                        teacher_hash: quantization_teacher
                            .as_ref()
                            .map(|teacher| teacher.sha256.clone()),
                    }
                }),
            };
            let mut batch = Vec::with_capacity(phase.batch_size);
            // Master weights are immutable throughout one gradient-accumulation
            // window. Keep one fake-quantized leaf model for that whole window
            // instead of re-quantizing every matrix for every microbatch.
            // Parameter IDs are retained by `fake_quantized_transformer`, so
            // each backward pass still accumulates directly into the master
            // optimizer slots. The staged leaves are discarded immediately
            // after the master update.
            let mut quantized_window: Option<StagedQuantizationWindow> = None;
            let shuffle_seed = shuffle_seed(args.seed, phase_index, epoch);
            let tokenizer_ref = &tokenizer;
            let objective = phase.objective.clone();
            let token_cache_path = token_cache_paths[phase_index].clone();
            let data_binding = &data_bindings[phase_index];
            std::thread::scope(|threads| -> Result<()> {
                let prefetch_capacity = training_prefetch_capacity(phase.batch_size)?;
                let (sender, receiver) = std::sync::mpsc::sync_channel(prefetch_capacity);
                let reader = threads.spawn(move || -> Result<()> {
                    let mut visited = 0usize;
                    let mut cursor_ready = false;
                    if records_to_skip == 0 {
                        if sender.send(PrefetchedSample::CursorReady).is_err() {
                            return Ok(());
                        }
                        cursor_ready = true;
                    }
                    visit_samples(
                        &phase.data,
                        &objective,
                        tokenizer_ref,
                        SampleStreamConfig {
                            seq_len: phase.sequence_length,
                            shuffle_buffer: phase.shuffle_buffer,
                            seed: shuffle_seed,
                            token_cache: Some(&token_cache_path),
                            data_binding,
                        },
                        |sample| {
                            visited = visited
                                .checked_add(1)
                                .context("sample-reader cursor overflows usize")?;
                            match resume_cursor_action(visited, records_to_skip) {
                                ResumeCursorAction::Skip => return Ok(true),
                                ResumeCursorAction::CursorReady => {
                                    cursor_ready = sender
                                        .send(PrefetchedSample::CursorReady)
                                        .is_ok();
                                    return Ok(cursor_ready);
                                }
                                ResumeCursorAction::Emit => {}
                            }
                            debug_assert!(cursor_ready);
                            Ok(sender.send(PrefetchedSample::Sample(sample)).is_ok())
                        },
                    )?;
                    ensure!(
                        visited >= records_to_skip,
                        "workflow phase `{}` epoch {} has only {visited} samples, fewer than resume cursor {records_to_skip}",
                        phase.name,
                        epoch + 1
                    );
                    Ok(())
                });
                let mut cursor_ready = false;
                loop {
                    let input_wait_started = Instant::now();
                    let sample = match receiver.recv() {
                        Ok(PrefetchedSample::CursorReady) => {
                            ensure!(
                                !cursor_ready,
                                "sample reader emitted duplicate cursor readiness"
                            );
                            cursor_ready = true;
                            optimizer_step_started = Instant::now();
                            step_input_wait_seconds = 0.0;
                            continue;
                        }
                        Ok(PrefetchedSample::Sample(sample)) => {
                            ensure!(
                                cursor_ready,
                                "sample reader emitted data before resume catch-up"
                            );
                            sample
                        }
                        Err(_) => break,
                    };
                    step_input_wait_seconds += input_wait_started.elapsed().as_secs_f64();
                    records_in_phase = records_in_phase
                        .checked_add(1)
                        .context("phase sample count overflows usize")?;
                    training_state.records_in_phase = records_in_phase;
                    rng_stream_mut(&mut training_state, DATA_RNG_STREAM)?.counter =
                        records_in_phase as u64;
                    batch.push(sample);
                    if batch.len() < phase.batch_size {
                        continue;
                    }

                    let transfer_started = Instant::now();
                    if let Some((max_records, max_context_tokens)) = wake_context_limits {
                        for sample in &batch {
                            let tokens = sample.wake_context_tokens();
                            let keep = tokens.len().min(config.max_seq_len).min(max_context_tokens);
                            push_bounded_wake_context(
                                &mut step_wake_contexts,
                                tokens[tokens.len() - keep..].to_vec(),
                                max_records,
                            );
                        }
                    }
                    let training_batch = make_batch(&batch, phase.sequence_length, &device)?;
                    step_host_to_device_seconds += transfer_started.elapsed().as_secs_f64();
                    batch.clear();
                    let model_rng = rng_stream_mut(&mut training_state, MODEL_RNG_STREAM)?;
                    device.seed(model_microbatch_seed(model_rng.seed, model_rng.counter));
                    model_rng.counter = model_rng
                        .counter
                        .checked_add(1)
                        .context("model RNG counter overflows u64")?;
                    let current = model.as_ref().unwrap();
                    // QAT reconstruction is accelerator work too. Start the
                    // busy interval before building the once-per-window
                    // fake-quantized leaves so utilization metrics do not
                    // misclassify that cost as host/input idle time.
                    let accelerator_started = Instant::now();
                    let quantization_format = phase
                        .quantization
                        .as_ref()
                        .and_then(|plan| plan.format_at(step as u64));
                    if let Some(format) = quantization_format
                        && quantized_window.is_none()
                    {
                        let recipe = &phase
                            .quantization
                            .as_ref()
                            .expect("format came from plan")
                            .recipe;
                        let (staged, tensors) = fake_quantized_transformer(
                            current,
                            format,
                            recipe.quantize_embeddings,
                            recipe.quantize_lm_head,
                        )?;
                        quantized_window = Some(StagedQuantizationWindow {
                            format,
                            model: staged,
                            tensor_count: tensors as u64,
                        });
                    }
                    ensure!(
                        quantized_window
                            .as_ref()
                            .is_none_or(|window| Some(window.format) == quantization_format),
                        "quantization format changed inside a gradient-accumulation window"
                    );
                    let forward_model = quantized_window
                        .as_ref()
                        .map_or(current, |window| &window.model);
                    let teacher_logits = quantization_teacher
                        .as_ref()
                        .map(|teacher| {
                            quantization_teacher_logits(
                                &teacher.model,
                                &training_batch,
                                &phase.objective,
                            )
                        })
                        .transpose()?;
                    let ObjectiveForward {
                        loss: task_loss,
                        router_loss,
                        stats: batch_stats,
                        retrieval_correct,
                        captured_logits,
                    } = objective_loss(
                        forward_model,
                        training_batch,
                        &phase.objective,
                        quantization_teacher.is_some(),
                    )?;
                    let distillation_loss = match (
                        teacher_logits,
                        captured_logits,
                        quantization_teacher.as_ref(),
                    ) {
                        (Some(teacher_logits), Some(student_logits), Some(teacher)) => {
                            Some(forward_kl_distillation_tensor(
                                teacher_logits,
                                student_logits,
                                teacher.temperature,
                                true,
                            )?)
                        }
                        (None, None, None) => None,
                        _ => unreachable!(
                            "teacher and student distillation outputs must be produced together"
                        ),
                    };
                    let detached_loss = task_loss.clone().detach();
                    accumulate_tensor(&mut loss_sum, detached_loss);
                    if let Some(router_loss) = &router_loss {
                        accumulate_tensor(&mut router_loss_sum, router_loss.clone().detach());
                    }
                    if let Some(distillation_loss) = &distillation_loss {
                        accumulate_tensor(
                            &mut distillation_loss_sum,
                            distillation_loss.clone().detach(),
                        );
                    }
                    add_batch_stats(&mut step_stats, batch_stats)?;
                    if let Some(correct) = retrieval_correct {
                        accumulate_tensor(&mut retrieval_correct_sum, correct);
                    }
                    let optimized_loss = match router_loss {
                        Some(router_loss) => task_loss + router_loss,
                        None => task_loss,
                    };
                    let optimized_loss = match (distillation_loss, &quantization_teacher) {
                        (Some(distillation), Some(teacher)) => {
                            optimized_loss + distillation.mul_scalar(teacher.loss_weight)
                        }
                        (None, _) => optimized_loss,
                        (Some(_), None) => unreachable!("distillation loss requires teacher"),
                    };
                    let gradient_accumulation_scale = phase.gradient_accumulation as f64;
                    let backward_loss = if tier_bank.is_some() {
                        // Tier optimizers average their independently retained
                        // raw microbatch gradients at the sender boundary.
                        optimized_loss.mul_scalar(phase.loss_weight)
                    } else {
                        optimized_loss
                            .mul_scalar(phase.loss_weight)
                            .div_scalar(gradient_accumulation_scale)
                    };
                    let mut grads = backward_loss.backward();
                    let mut muon_grads = GradientsParams::from_params(
                        &mut grads,
                        forward_model,
                        &muon_parameter_ids,
                    );
                    let mut adamw_grads = match tier_bank {
                        Some(bank) => {
                            let partitioned = bank.partition_gradients(current, &mut grads)?;
                            ensure!(
                                partitioned.tiers.len() == tier_accumulators.len(),
                                "memory-tier gradient partition changed during wake training"
                            );
                            for (accumulator, tier_gradients) in
                                tier_accumulators.iter_mut().zip(partitioned.tiers)
                            {
                                accumulator.accumulate(current, tier_gradients);
                            }
                            partitioned.wake
                        }
                        None => GradientsParams::from_module(&mut grads, forward_model),
                    };
                    if tier_bank.is_some() {
                        let scale = 1.0 / phase.gradient_accumulation as f32;
                        scale_gradients(current, &mut muon_grads, scale);
                        scale_gradients(current, &mut adamw_grads, scale);
                    }
                    muon_accumulator.accumulate(current, muon_grads);
                    adamw_accumulator.accumulate(current, adamw_grads);
                    micro_step += 1;

                    if micro_step == phase.gradient_accumulation {
                        let lr =
                            learning_rate(&args, step + 1, total_steps) * phase.learning_rate_scale;
                        let muon_lr = lr * MUON_LR_SCALE;
                        let loss = loss_sum
                            .take()
                            .expect("an optimizer step must contain a loss")
                            .div_scalar(phase.gradient_accumulation as f32);
                        let loss = scalar_value(loss)?;
                        let router_loss = router_loss_sum
                            .take()
                            .map(|sum| {
                                scalar_value(sum.div_scalar(phase.gradient_accumulation as f32))
                            })
                            .transpose()?;
                        let distillation_loss = distillation_loss_sum
                            .take()
                            .map(|sum| {
                                scalar_value(sum.div_scalar(phase.gradient_accumulation as f32))
                            })
                            .transpose()?;
                        let optimized_loss = loss
                            + router_loss.unwrap_or(0.0)
                            + distillation_loss.unwrap_or(0.0)
                                * quantization_teacher
                                    .as_ref()
                                    .map_or(0.0, |teacher| teacher.loss_weight as f32);
                        let weighted_loss = optimized_loss * phase.loss_weight as f32;
                        let retrieval_accuracy = retrieval_correct_sum
                            .take()
                            .map(scalar_value)
                            .transpose()?
                            .map(|correct| correct / step_stats.examples as f32);
                        if !loss.is_finite()
                            || router_loss.is_some_and(|loss| !loss.is_finite())
                            || distillation_loss.is_some_and(|loss| !loss.is_finite())
                            || !weighted_loss.is_finite()
                        {
                            bail!(
                                "non-finite loss at optimizer step {}: task={loss}, router={router_loss:?}, distillation={distillation_loss:?}, weighted={weighted_loss}",
                                step + 1
                            );
                        }
                        let mut muon_grads = muon_accumulator.grads();
                        let mut adamw_grads = adamw_accumulator.grads();
                        let mut tier_grads = tier_accumulators
                            .iter_mut()
                            .map(GradientsAccumulator::grads)
                            .collect::<Vec<_>>();
                        if tier_bank.is_some() {
                            let scale = 1.0 / phase.gradient_accumulation as f32;
                            for gradients in &mut tier_grads {
                                scale_gradients(current, gradients, scale);
                            }
                        }
                        let layer_grad_norms = (args.layer_metrics_every > 0
                            && (step + 1) % args.layer_metrics_every == 0)
                            .then(|| {
                                layer_gradient_norms(
                                    current,
                                    &muon_grads,
                                    &adamw_grads,
                                    &tier_grads,
                                )
                            })
                            .transpose()?;
                        let grad_norm = gradient_norm_and_clip(
                            current,
                            &mut muon_grads,
                            &mut adamw_grads,
                            &mut tier_grads,
                            args.grad_clip,
                        )?;
                        if !grad_norm.is_finite() {
                            bail!(
                                "non-finite gradient norm at optimizer step {}: {grad_norm}",
                                step + 1
                            );
                        }
                        if let Some(bank) = tier_bank {
                            bank.commit_tier_gradients(current, tier_grads, 1)?;
                        } else {
                            ensure!(
                                tier_grads.is_empty(),
                                "ordinary training produced memory-tier gradients"
                            );
                        }
                        let current = model.take().unwrap();
                        let current = muon_optimizer.step(muon_lr, current, muon_grads)?;
                        model = Some(adamw_optimizer.step(lr.into(), current, adamw_grads));
                        let next_step = step
                            .checked_add(1)
                            .context("global optimizer-step count overflows usize")?;
                        let wake_only_updates = if let Some(runtime) = &wake_only_runtime {
                            let current = model.take().expect("training model is present");
                            let (updated, report) = runtime
                                .bank
                                .apply_wake_only_due_updates(&current, next_step as u64)?;
                            debug_assert!(
                                report
                                    .updates
                                    .windows(2)
                                    .all(|pair| pair[0].tier < pair[1].tier),
                                "wake_only updates must run fastest-to-slowest"
                            );
                            model = Some(updated);
                            report.updates
                        } else {
                            Vec::new()
                        };
                        let step_quantized_tensors = quantized_window
                            .as_ref()
                            .map_or(0, |window| window.tensor_count);
                        // The authoritative model has changed. Never reuse a
                        // staged QAT leaf across optimizer-step boundaries.
                        quantized_window = None;
                        step_accelerator_seconds += accelerator_started.elapsed().as_secs_f64();
                        step = next_step;
                        steps_in_phase = steps_in_phase
                            .checked_add(1)
                            .context("phase optimizer-step count overflows usize")?;
                        tokens_seen = tokens_seen
                            .checked_add(step_stats.compute_tokens)
                            .context("cumulative model-token count overflows usize")?;
                        training_state.global_step = step;
                        training_state.steps_in_phase = steps_in_phase;
                        training_state.tokens_seen = tokens_seen;
                        training_state.optimizer_states[0].update_clock = step as u64;
                        let max_wake_context_records =
                            wake_context_limits.map_or(0, |(records, _)| records);
                        let retained_existing =
                            max_wake_context_records.saturating_sub(step_wake_contexts.len());
                        if training_state.wake_context_buffer.len() > retained_existing {
                            let discard =
                                training_state.wake_context_buffer.len() - retained_existing;
                            training_state.wake_context_buffer.drain(..discard);
                        }
                        for (ordinal, token_ids) in step_wake_contexts.drain(..).enumerate() {
                            training_state.wake_context_buffer.push(
                                hermes_train::builtin_sleep_adapters::WakeContextRecord {
                                    id: format!("{}:{step}:{ordinal}", phase.name),
                                    optimizer_step: step as u64,
                                    token_ids,
                                },
                            );
                        }
                        debug_assert!(
                            training_state.wake_context_buffer.len() <= max_wake_context_records
                        );
                        if let Some(quantization) = &mut training_state.quantization {
                            let format = phase
                                .quantization
                                .as_ref()
                                .and_then(|plan| plan.format_at((step - 1) as u64));
                            quantization.format = quantization_state_format(format);
                            quantization.fake_quant_active = format.is_some();
                            quantization.calibration_step = step as u64;
                        }
                        let step_seconds = optimizer_step_started.elapsed().as_secs_f64();
                        let tokens_per_second = step_stats.compute_tokens as f64 / step_seconds;
                        println!(
                            "phase={}/{} name={} objective={} epoch={} step={step}/{total_steps} phase_step={} loss={loss:.6} lr={lr:.3e} grad_norm={grad_norm:.3} tokens_per_second={tokens_per_second:.0}",
                            phase_index + 1,
                            workflow.phases.len(),
                            phase.name,
                            phase.objective.name(),
                            epoch + 1,
                            steps_in_phase,
                        );
                        let context = MetricContext {
                            global_step: step as u64,
                            phase: MetricPhase {
                                index: phase_index as u32,
                                name: phase.name.clone(),
                                kind: phase.phase_kind.into(),
                            },
                            checkpoint_hash: None,
                        };
                        drain_device_sampler(&mut device_sampler, &mut metrics, &context)?;
                        metrics.append(
                            context.clone(),
                            MetricEvent::Optimization(OptimizationMetrics {
                                objective: phase.objective.name().into(),
                                loss: f64::from(loss),
                                optimized_loss: f64::from(optimized_loss),
                                weighted_loss: f64::from(weighted_loss),
                                router_aux_loss: router_loss.map(f64::from),
                                retrieval_accuracy: retrieval_accuracy.map(f64::from),
                                learning_rate: lr,
                                muon_learning_rate: muon_lr,
                                gradient_norm: f64::from(grad_norm),
                                layer_gradient_norms: layer_grad_norms
                                    .map(|norms| norms.into_iter().map(f64::from).collect()),
                                sequence_length: phase.sequence_length as u32,
                                batch_size: phase.batch_size as u32,
                                gradient_accumulation: phase.gradient_accumulation as u32,
                                compute_tokens: step_stats.compute_tokens as u64,
                                supervised_tokens: step_stats.supervised_tokens as u64,
                                examples: step_stats.examples as u64,
                                truncated_tokens: step_stats.truncated_tokens as u64,
                                retrieval_candidates: step_stats.retrieval_candidates as u64,
                            }),
                        )?;
                        if let Some(runtime) = &wake_only_runtime {
                            append_wake_only_tier_metrics(
                                &mut metrics,
                                &context,
                                runtime.config.schedule(),
                                &wake_only_updates,
                            )?;
                        } else {
                            ensure!(
                                wake_only_updates.is_empty(),
                                "ordinary training produced wake_only update evidence"
                            );
                        }
                        if let Some(plan) = &phase.quantization {
                            let active_format = plan.format_at((step - 1) as u64);
                            metrics.append(
                                context.clone(),
                                MetricEvent::Quantization(QuantizationMetrics {
                                    stage: if active_format.is_some() {
                                        QuantizationStage::FakeQuantization
                                    } else {
                                        QuantizationStage::Calibration
                                    },
                                    format: metric_quantization_format(
                                        active_format.unwrap_or(plan.recipe.format),
                                    ),
                                    group_size: plan.recipe.group_size as u32,
                                    progress_fraction: steps_in_phase as f64
                                        / phase_plan[phase_index].steps as f64,
                                    tensors_quantized: step_quantized_tensors,
                                    // Exact element and codec-error accounting is
                                    // emitted by the archive export pass.
                                    weights_quantized: 0,
                                    average_bits_per_weight: None,
                                    packed_bytes: None,
                                    mean_squared_error: None,
                                    max_absolute_error: None,
                                    distillation_forward_kl: distillation_loss.map(f64::from),
                                    acceptance_delta: None,
                                    embeddings_quantized: plan.recipe.quantize_embeddings,
                                    lm_head_quantized: plan.recipe.quantize_lm_head,
                                }),
                            )?;
                        }
                        metrics.append(
                            context.clone(),
                            MetricEvent::Throughput(ThroughputMetrics {
                                optimizer_steps: 1,
                                compute_tokens: step_stats.compute_tokens as u64,
                                supervised_tokens: step_stats.supervised_tokens as u64,
                                examples: step_stats.examples as u64,
                                elapsed_seconds: step_seconds,
                                tokens_per_second,
                                examples_per_second: step_stats.examples as f64 / step_seconds,
                                input_wait_seconds: step_input_wait_seconds.min(step_seconds),
                                host_to_device_seconds: step_host_to_device_seconds
                                    .min(step_seconds),
                                gpu_busy_seconds: step_accelerator_seconds.min(step_seconds),
                            }),
                        )?;
                        training_state.metric_records = metrics.state().records;
                        metrics.flush()?;
                        if let Some(runtime) = &periodic_runtime {
                            let sleep = phase
                                .periodic_sleep
                                .as_ref()
                                .expect("validated memory phase has periodic sleep");
                            let mut current = model.take().expect("training model is present");
                            advance_and_drain_periodic_sleep(
                                runtime,
                                sleep,
                                phase_index,
                                phase,
                                &mut training_state,
                                &mut current,
                                &adamw_optimizer,
                                &muon_optimizer,
                                &mut metrics,
                                &args.output,
                            )?;
                            model = Some(current);
                            drain_device_sampler(&mut device_sampler, &mut metrics, &context)?;
                            training_state.metric_records = metrics.state().records;
                            metrics.flush()?;
                        }
                        if steps_in_phase == phase_plan[phase_index].steps {
                            publish_completed_quantization_candidate_if_pending(
                                model.as_ref().unwrap(),
                                phase_index,
                                phase,
                                phase_plan[phase_index].steps,
                                &mut training_state,
                                &mut metrics,
                                &args.output,
                                periodic_runtime.as_ref(),
                                None,
                            )?;
                        }
                        if args.checkpoint_every > 0 && step % args.checkpoint_every == 0 {
                            if let (Some(runtime), Some(sleep)) =
                                (&periodic_runtime, phase.periodic_sleep.as_ref())
                            {
                                let mut cursor = training_state
                                    .sleep
                                    .clone()
                                    .context("periodic checkpoint has no sleep cursor")?;
                                let _ = runtime.checkpoint_wake(
                                    &mut cursor,
                                    model.as_ref().unwrap(),
                                    sleep,
                                )?;
                                training_state.sleep = Some(cursor);
                            }
                            if let Some(runtime) = &wake_only_runtime {
                                runtime.checkpoint(&mut training_state)?;
                            }
                            let publication = save_training_checkpoint_with_evidence(
                                model.as_ref().unwrap(),
                                &adamw_optimizer,
                                &muon_optimizer,
                                &training_state,
                                &mut metrics,
                                &args.output,
                            )?;
                            print_checkpoint_publication("checkpointed", &publication);
                        }
                        micro_step = 0;
                        step_stats = BatchStats::default();
                        optimizer_step_started = Instant::now();
                        step_input_wait_seconds = 0.0;
                        step_host_to_device_seconds = 0.0;
                        step_accelerator_seconds = 0.0;
                        if step >= total_steps
                            || phase.steps.is_some_and(|limit| steps_in_phase >= limit)
                        {
                            phase_limit_reached = true;
                            break;
                        }
                    } else {
                        step_accelerator_seconds += accelerator_started.elapsed().as_secs_f64();
                    }
                }
                if !batch.is_empty() {
                    println!(
                        "phase={} epoch={} dropped_incomplete_batch_examples={}",
                        phase.name,
                        epoch + 1,
                        batch.len()
                    );
                }
                drop(receiver);
                reader
                    .join()
                    .expect("sample reader thread panicked")
                    .map(|_| ())
            })?;

            if micro_step != 0 {
                println!(
                    "phase={} epoch={} dropped_incomplete_optimizer_microbatches={micro_step}",
                    phase.name,
                    epoch + 1
                );
                muon_accumulator = GradientsAccumulator::new();
                adamw_accumulator = GradientsAccumulator::new();
                for accumulator in &mut tier_accumulators {
                    *accumulator = GradientsAccumulator::new();
                }
                micro_step = 0;
                loss_sum = None;
                router_loss_sum = None;
                distillation_loss_sum = None;
                retrieval_correct_sum = None;
                step_wake_contexts.clear();
                step_stats = BatchStats::default();
            }
            if step >= total_steps {
                break 'phases;
            }
            if phase_limit_reached {
                break;
            }
        }

        ensure!(
            steps_in_phase == phase_plan[phase_index].steps,
            "workflow phase `{}` requested {} optimizer steps, but its data and epochs produced {steps_in_phase}",
            phase.name,
            phase_plan[phase_index].steps
        );
        // Phase-boundary publication prevents a relaunch from replaying a
        // completed short fine-tuning phase before the next periodic save.
        if let (Some(runtime), Some(sleep)) = (&periodic_runtime, phase.periodic_sleep.as_ref()) {
            let mut cursor = training_state
                .sleep
                .clone()
                .context("periodic phase boundary has no sleep cursor")?;
            let _ = runtime.checkpoint_wake(&mut cursor, model.as_ref().unwrap(), sleep)?;
            training_state.sleep = Some(cursor);
        }
        let context = MetricContext {
            global_step: training_state.global_step as u64,
            phase: MetricPhase {
                index: phase_index as u32,
                name: phase.name.clone(),
                kind: phase.phase_kind.into(),
            },
            checkpoint_hash: None,
        };
        drain_device_sampler(&mut device_sampler, &mut metrics, &context)?;
        training_state.metric_records = metrics.state().records;
        if let Some(runtime) = &wake_only_runtime {
            runtime.checkpoint(&mut training_state)?;
        }
        let publication = save_training_checkpoint_with_evidence(
            model.as_ref().unwrap(),
            &adamw_optimizer,
            &muon_optimizer,
            &training_state,
            &mut metrics,
            &args.output,
        )?;
        print_checkpoint_publication(
            &format!("checkpointed completed phase `{}`", phase.name),
            &publication,
        );
    }
    ensure!(
        step == total_steps,
        "requested {total_steps} optimizer steps, but the data produced only {step} complete steps"
    );

    if let Some(runtime) = &periodic_runtime {
        let sleep = workflow
            .phases
            .last()
            .and_then(|phase| phase.periodic_sleep.as_ref())
            .expect("validated memory workflow has periodic sleep");
        let mut cursor = training_state
            .sleep
            .clone()
            .context("periodic final checkpoint has no sleep cursor")?;
        let _ = runtime.checkpoint_wake(&mut cursor, model.as_ref().unwrap(), sleep)?;
        training_state.sleep = Some(cursor);
    }
    let final_phase_index = workflow.phases.len() - 1;
    let final_phase = &workflow.phases[final_phase_index];
    let final_context = MetricContext {
        global_step: training_state.global_step as u64,
        phase: MetricPhase {
            index: final_phase_index as u32,
            name: final_phase.name.clone(),
            kind: final_phase.phase_kind.into(),
        },
        checkpoint_hash: None,
    };
    shutdown_device_sampler(&mut device_sampler, &mut metrics, &final_context)?;
    training_state.metric_records = metrics.state().records;
    if let Some(runtime) = &wake_only_runtime {
        runtime.checkpoint(&mut training_state)?;
    }
    let publication = save_training_checkpoint_with_evidence(
        model.as_ref().unwrap(),
        &adamw_optimizer,
        &muon_optimizer,
        &training_state,
        &mut metrics,
        &args.output,
    )?;
    print_checkpoint_publication("saved", &publication);
    Ok(())
}

fn print_checkpoint_publication(label: &str, publication: &CheckpointPublication) {
    println!(
        "{label} checkpoint_manifest={} checkpoint_manifest_sha256={}",
        publication.checkpoint_manifest.display(),
        publication.checkpoint_manifest_sha256,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Int;

    #[test]
    fn training_prefetch_is_two_batches_but_never_scales_with_accumulation() {
        assert_eq!(training_prefetch_capacity(1).unwrap(), 2);
        assert_eq!(training_prefetch_capacity(64).unwrap(), 128);
        assert_eq!(training_prefetch_capacity(usize::MAX).unwrap(), 4_096);
        assert!(training_prefetch_capacity(0).is_err());
    }

    #[test]
    fn wake_context_collection_is_bounded_while_a_step_is_accumulating() {
        let mut contexts = VecDeque::new();
        for token in 0..10 {
            push_bounded_wake_context(&mut contexts, vec![token], 3);
        }
        assert_eq!(
            contexts.into_iter().collect::<Vec<_>>(),
            vec![vec![7], vec![8], vec![9]]
        );
    }

    #[test]
    fn quantization_checkpoint_formats_use_workflow_names() {
        assert_eq!(quantization_state_format(None), "full_precision");
        assert_eq!(
            quantization_state_format(Some(UltraQuantFormat::BinaryG128)),
            "binary_g128"
        );
        assert_eq!(
            quantization_state_format(Some(UltraQuantFormat::TernaryG128)),
            "ternary_g128"
        );
        assert_eq!(
            quantization_state_format(Some(UltraQuantFormat::TernaryEntropyG128)),
            "ternary_entropy_g128"
        );
    }

    #[test]
    fn pinned_checkpoint_reader_authenticates_exact_regular_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.safetensors");
        let bytes = b"authenticated checkpoint fixture";
        fs::write(&path, bytes).unwrap();
        let expected = format!("sha256:{:x}", Sha256::digest(bytes));
        assert_eq!(
            read_pinned_checkpoint_bytes(&path, &expected).unwrap(),
            bytes
        );
        let error = read_pinned_checkpoint_bytes(&path, &format!("sha256:{}", "0".repeat(64)))
            .unwrap_err()
            .to_string();
        assert!(error.contains("hash mismatch"), "{error}");

        let mut mutated = bytes.to_vec();
        mutated[0] ^= 1;
        let error = read_pinned_checkpoint_bytes_after_open(&path, &expected, || {
            fs::write(&path, &mutated).context("mutating opened checkpoint fixture")
        })
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("changed while it was read") || error.contains("hash mismatch"),
            "{error}"
        );
        fs::write(&path, bytes).unwrap();

        let mut grown = bytes.to_vec();
        grown.push(b'!');
        let error = read_pinned_checkpoint_bytes_after_open(&path, &expected, || {
            fs::write(&path, &grown).context("growing opened checkpoint fixture")
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("changed while it was read"), "{error}");
        fs::write(&path, bytes).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let link = directory.path().join("linked.safetensors");
            symlink(&path, &link).unwrap();
            let error = read_pinned_checkpoint_bytes(&link, &expected)
                .unwrap_err()
                .to_string();
            assert!(error.contains("non-symlink"), "{error}");

            let replacement = directory.path().join("replacement.safetensors");
            let parked = directory.path().join("parked.safetensors");
            fs::write(&replacement, bytes).unwrap();
            let error = read_pinned_checkpoint_bytes_after_open(&path, &expected, || {
                fs::rename(&path, &parked)?;
                fs::rename(&replacement, &path)?;
                Ok(())
            })
            .unwrap_err()
            .to_string();
            assert!(error.contains("publication changed"), "{error}");
        }
    }

    fn qat_replay_model() -> ModelDef {
        hermes_llm::parse_mal(
            r#"
            ffn base {
                hidden_dim: 16
                activation: swiglu
                dropout: 0.0
                bias: false
            }
            model qat-replay {
                vocab_size: 32 max_seq_len: 8 hidden_size: 8 num_layers: 2
                block: {
                    attention: {
                        num_heads: 2 num_kv_heads: 1 head_dim: 4
                        position_encoding: none dropout: 0.0
                    }
                    ffn: base
                    dropout: 0.0
                }
                embeddings { tie_weights: false dropout: 0.0 }
            }
            "#,
        )
        .unwrap()
    }

    fn apply_qat_window(
        mut model: Transformer,
        muon: &mut BatchedMuon,
        adamw: &mut AdamWOptimizer,
        device: &Device,
        rng_counters: &[u64],
    ) -> Transformer {
        assert!(!rng_counters.is_empty());
        let muon_ids = model.muon_parameter_ids();
        let (staged, _) =
            fake_quantized_transformer(&model, UltraQuantFormat::BinaryG128, true, true).unwrap();
        let mut muon_accumulator = GradientsAccumulator::new();
        let mut adamw_accumulator = GradientsAccumulator::new();
        for &counter in rng_counters {
            let first = 1 + i64::try_from(counter % 8).unwrap();
            let input =
                Tensor::<2, Int>::from_data([[first, first + 1, first + 2, first + 3]], device);
            let target =
                Tensor::<2, Int>::from_data([[first + 1, first + 2, first + 3, first + 4]], device);
            let mut gradients = staged
                .forward_loss(input, target)
                .div_scalar(rng_counters.len() as f64)
                .backward();
            let muon_gradients = GradientsParams::from_params(&mut gradients, &staged, &muon_ids);
            let adamw_gradients = GradientsParams::from_module(&mut gradients, &staged);
            muon_accumulator.accumulate(&model, muon_gradients);
            adamw_accumulator.accumulate(&model, adamw_gradients);
        }
        let mut muon_gradients = muon_accumulator.grads();
        let mut adamw_gradients = adamw_accumulator.grads();
        gradient_norm_and_clip(
            &model,
            &mut muon_gradients,
            &mut adamw_gradients,
            &mut [],
            1.0,
        )
        .unwrap();
        model = muon.step(2e-2, model, muon_gradients).unwrap();
        adamw.step(1e-3.into(), model, adamw_gradients)
    }

    fn restore_qat_boundary(
        config: &ModelDef,
        device: &Device,
        output: &Path,
    ) -> (Transformer, BatchedMuon, AdamWOptimizer, TrainingState) {
        let mut model = Transformer::new(config, device).unwrap();
        let mut muon = BatchedMuon::new(model.muon_parameter_ids());
        let adamw = AdamWConfig::new()
            .with_beta_2(0.95)
            .with_epsilon(1e-8)
            .with_weight_decay(0.0)
            .init();
        let (adamw, state, _) =
            load_training_state(&mut model, adamw, &mut muon, output, device).unwrap();
        (model, muon, adamw, state)
    }

    #[test]
    fn interrupted_qat_window_replays_exactly_from_atomic_trainer_checkpoint() {
        let config = qat_replay_model();
        let device = Device::ndarray().autodiff();
        let mut model = Transformer::new(&config, &device).unwrap();
        let mut muon = BatchedMuon::new(model.muon_parameter_ids());
        let mut adamw = AdamWConfig::new()
            .with_beta_2(0.95)
            .with_epsilon(1e-8)
            .with_weight_decay(0.0)
            .init();
        model = apply_qat_window(model, &mut muon, &mut adamw, &device, &[0, 1]);

        let directory = tempfile::tempdir().unwrap();
        let mut metrics =
            MetricWriter::create(directory.path().join("metrics.jsonl"), "qat-replay").unwrap();
        metrics
            .append_at(
                MetricContext {
                    global_step: 1,
                    phase: MetricPhase {
                        index: 0,
                        name: "qat".into(),
                        kind: MetricPhaseKind::Quantization,
                    },
                    checkpoint_hash: None,
                },
                MetricEvent::Throughput(ThroughputMetrics {
                    optimizer_steps: 1,
                    compute_tokens: 8,
                    supervised_tokens: 8,
                    examples: 2,
                    elapsed_seconds: 1.0,
                    tokens_per_second: 8.0,
                    examples_per_second: 2.0,
                    input_wait_seconds: 0.0,
                    host_to_device_seconds: 0.0,
                    gpu_busy_seconds: 1.0,
                }),
                1,
            )
            .unwrap();
        let state = TrainingState {
            version: TRAINING_STATE_VERSION,
            global_step: 1,
            phase: 0,
            phase_id: "qat".into(),
            phase_kind: "quantization".into(),
            epoch: 0,
            records_in_phase: 2,
            steps_in_phase: 1,
            tokens_seen: 8,
            metric_records: 1,
            workflow_signature: format!("sha256:{}", "a".repeat(64)),
            data_manifest_hash: Some(format!("sha256:{}", "b".repeat(64))),
            parameter_ids: parameter_ids(&model),
            optimizer_states: vec![OptimizerStateRef {
                scope: "wake".into(),
                adamw: "adamw-state.bpk".into(),
                muon: "muon-state.bpk".into(),
                gradient_accumulator: None,
                update_clock: 1,
            }],
            memory_update: TrainingMemoryUpdateState::Ordinary,
            sleep: None,
            artifacts: Vec::new(),
            evaluator_hashes: Vec::new(),
            rng_streams: vec![
                RngStreamState {
                    name: DATA_RNG_STREAM.into(),
                    seed: 19,
                    counter: 2,
                },
                RngStreamState {
                    name: MODEL_RNG_STREAM.into(),
                    seed: 97,
                    counter: 2,
                },
            ],
            wake_context_buffer: Vec::new(),
            quantization: Some(QuantizationTrainingState {
                format: "binary_g128".into(),
                fake_quant_active: true,
                calibration_step: 1,
                manifest: None,
                candidate_weights_sha256: None,
                teacher_hash: None,
            }),
        };
        save_training_checkpoint_with_evidence(
            &model,
            &adamw,
            &muon,
            &state,
            &mut metrics,
            directory.path(),
        )
        .unwrap();
        let durable_pointer = fs::read(directory.path().join("current.json")).unwrap();

        let (reference_model, mut reference_muon, mut reference_adamw, reference_state) =
            restore_qat_boundary(&config, &device, directory.path());
        let reference = apply_qat_window(
            reference_model,
            &mut reference_muon,
            &mut reference_adamw,
            &device,
            &[2, 3],
        );

        // This update is intentionally never published. Dropping the process-
        // local model and optimizers represents an interruption anywhere in the
        // accumulation/update window before the next immutable checkpoint.
        let (interrupted_model, mut interrupted_muon, mut interrupted_adamw, interrupted_state) =
            restore_qat_boundary(&config, &device, directory.path());
        let interrupted = apply_qat_window(
            interrupted_model,
            &mut interrupted_muon,
            &mut interrupted_adamw,
            &device,
            &[2],
        );
        drop((interrupted, interrupted_muon, interrupted_adamw));
        assert_eq!(
            fs::read(directory.path().join("current.json")).unwrap(),
            durable_pointer,
            "an in-memory QAT window must not advance the durable generation"
        );
        assert_eq!(interrupted_state.rng_streams, reference_state.rng_streams);

        let (replayed_model, mut replayed_muon, mut replayed_adamw, replayed_state) =
            restore_qat_boundary(&config, &device, directory.path());
        assert_eq!(replayed_state.rng_streams, reference_state.rng_streams);
        let replayed = apply_qat_window(
            replayed_model,
            &mut replayed_muon,
            &mut replayed_adamw,
            &device,
            &[2, 3],
        );

        let artifacts = tempfile::tempdir().unwrap();
        let reference_weights = artifacts.path().join("reference.safetensors");
        let replayed_weights = artifacts.path().join("replayed.safetensors");
        let reference_muon_path = artifacts.path().join("reference-muon.bpk");
        let replayed_muon_path = artifacts.path().join("replayed-muon.bpk");
        save_safetensors(&reference.valid(), &reference_weights).unwrap();
        save_safetensors(&replayed.valid(), &replayed_weights).unwrap();
        reference_muon.save(&reference_muon_path).unwrap();
        replayed_muon.save(&replayed_muon_path).unwrap();
        assert_eq!(
            fs::read(reference_weights).unwrap(),
            fs::read(replayed_weights).unwrap(),
            "replayed QAT master weights differ"
        );
        assert_eq!(
            fs::read(reference_muon_path).unwrap(),
            fs::read(replayed_muon_path).unwrap(),
            "replayed QAT Muon state differs"
        );
        assert_eq!(
            &*hermes_train::optimizer_artifact::canonical_module_optimizer_bytes(&reference_adamw,)
                .unwrap(),
            &*hermes_train::optimizer_artifact::canonical_module_optimizer_bytes(&replayed_adamw)
                .unwrap(),
            "replayed QAT AdamW state differs"
        );
    }

    #[test]
    fn resume_cursor_filter_preserves_exact_shuffled_suffix() {
        let shuffled = [7, 2, 9, 1, 5, 8, 0, 6, 3, 4];
        let cursor = 4;
        let actions = (1..=shuffled.len())
            .map(|visited| resume_cursor_action(visited, cursor))
            .collect::<Vec<_>>();
        assert_eq!(
            actions[..cursor],
            [
                ResumeCursorAction::Skip,
                ResumeCursorAction::Skip,
                ResumeCursorAction::Skip,
                ResumeCursorAction::CursorReady,
            ]
        );
        let resumed = shuffled
            .into_iter()
            .zip(actions)
            .filter_map(|(sample, action)| (action == ResumeCursorAction::Emit).then_some(sample))
            .collect::<Vec<_>>();
        assert_eq!(resumed, shuffled[cursor..]);
        assert!(
            (1..=shuffled.len()).all(|visited| resume_cursor_action(visited, shuffled.len() + 1)
                == ResumeCursorAction::Skip)
        );
    }

    #[cfg(unix)]
    #[test]
    fn training_output_lock_rejects_a_second_writer_and_releases_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let first = TrainingOutputLock::acquire(directory.path()).unwrap();
        let error = TrainingOutputLock::acquire(directory.path())
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("already owns output"), "{error}");

        drop(first);
        TrainingOutputLock::acquire(directory.path()).unwrap();

        let parent = tempfile::tempdir().unwrap();
        let link = parent.path().join("linked-output");
        std::os::unix::fs::symlink(directory.path(), &link).unwrap();
        let error = TrainingOutputLock::acquire(&link)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("real directory"), "{error}");
    }

    #[test]
    fn periodic_runtime_artifact_is_write_once_and_exact_on_resume() {
        let path = Path::new("/tmp/pinned-sleep-runtime.json");
        let hash = format!("sha256:{}", "a".repeat(64));
        let mut artifacts = Vec::new();
        bind_sleep_runtime_artifact(&mut artifacts, path, &hash).unwrap();
        assert_eq!(artifacts.len(), 1);
        validate_sleep_runtime_artifact(&artifacts, path, &hash).unwrap();
        assert!(bind_sleep_runtime_artifact(&mut artifacts, path, &hash).is_err());

        let other_hash = format!("sha256:{}", "b".repeat(64));
        assert!(validate_sleep_runtime_artifact(&artifacts, path, &other_hash).is_err());
        assert!(
            validate_sleep_runtime_artifact(
                &artifacts,
                Path::new("/tmp/another-runtime.json"),
                &hash,
            )
            .is_err()
        );
        artifacts.push(artifacts[0].clone());
        assert!(validate_sleep_runtime_artifact(&artifacts, path, &hash).is_err());
    }

    #[test]
    fn final_qat_boundary_publishes_only_post_sleep_digest_and_reopens_idempotently() {
        let directory = tempfile::tempdir().unwrap();
        let retention = directory.path().join("retention.json");
        fs::write(&retention, b"{\"examples\":[]}").unwrap();
        let retention_hash = file_sha256(&retention).unwrap();
        let candidate_directory = directory.path().join("sleep-candidates");
        let sleep = serde_json::json!({
            "schedule": {
                "clock": "optimizer_steps",
                "terminal_consolidation": "distill_into_base_v1",
                "tiers": [
                    {"id": "fast", "update_period": 1, "reserve_slots": 1},
                    {"id": "slow", "update_period": 2, "reserve_slots": 2}
                ]
            },
            "knowledge_seeding": {
                "chunk_tokens": 2,
                "teacher_rollouts": 1,
                "detached_student_rollouts": 1,
                "temperature": 1.0,
                "forward_kl_weight": 1.0
            },
            "imitation": {
                "semantic_judge_hash": format!("sha256:{}", "1".repeat(64)),
                "semantic_weight": 0.5,
                "maximum_edit_distance": 4,
                "grpo_group_size": 2
            },
            "retention_suite": retention,
            "retention_suite_sha256": retention_hash,
            "retention": {
                "evaluator_hash": format!("sha256:{}", "2".repeat(64)),
                "suite_hash": retention_hash,
                "max_anchor_forward_kl": 1.0,
                "max_anchor_regression": 1.0,
                "min_incorporation_gain": -1.0
            },
            "receiver_learning_rate": 0.0001,
            "receiver_weight_decay": 0.0,
            "grpo_clip_epsilon": 0.2,
            "grpo_advantage_epsilon": 0.000001,
            "grpo_kl_coefficient": 0.0,
            "candidate_directory": candidate_directory
        });
        let workflow_path = directory.path().join("workflow.json");
        fs::write(
            &workflow_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "version": 2,
                "phases": [{
                    "name": "binary-qat",
                    "type": "quantization",
                    "task": {"type": "causal_lm"},
                    "data": directory.path().join("data.jsonl"),
                    "sequence_length": 4,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 1,
                    "periodic_sleep": sleep,
                    "quantization": {
                        "format": "binary_g128",
                        "group_size": 128,
                        "start_step": 0,
                        "embeddings": true,
                        "lm_head": true,
                        "training": {
                            "type": "qat",
                            "warmup_steps": 0,
                            "straight_through": true
                        }
                    }
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        let workflow = load_wake_plan(&workflow_path).unwrap();
        let phase = &workflow.phases[0];
        let sleep = phase.periodic_sleep.as_ref().unwrap();
        let plan = phase.quantization.as_ref().unwrap();
        assert_eq!(sleep.schedule.due_senders(1), vec![0]);

        let model_def = hermes_llm::parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 1 rank: 3 top_k: 1 }
                }
                tier slow {
                    ffn: base residual_init: zero
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 32 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: {
                    attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
                    memory: cms
                    dropout: 0.0
                }
            }
            "#,
        )
        .unwrap();
        let device = hermes_llm::Device::ndarray().autodiff();
        let model = Transformer::new(&model_def, &device).unwrap();
        let pre_sleep =
            publish_sleep_model(&model, &directory.path().join("sleep-models")).unwrap();
        let workflow_signature = format!("sha256:{}", "a".repeat(64));
        let mut cursor = NativeSleepCheckpoint::new(
            &workflow_signature,
            &phase.name,
            pre_sleep.clone(),
            &model,
            sleep,
            1,
        )
        .unwrap();
        cursor.advance_clock(&model, sleep, 1).unwrap();
        assert_eq!(cursor.sleep.next_due_sender(), Some(0));

        let mut state = TrainingState {
            version: TRAINING_STATE_VERSION,
            global_step: 1,
            phase: 0,
            phase_id: phase.name.clone(),
            phase_kind: phase.phase_kind.name().into(),
            epoch: 0,
            records_in_phase: 1,
            steps_in_phase: 1,
            tokens_seen: 4,
            metric_records: 0,
            workflow_signature,
            data_manifest_hash: Some(format!("sha256:{}", "b".repeat(64))),
            parameter_ids: parameter_ids(&model),
            optimizer_states: vec![OptimizerStateRef {
                scope: "wake".into(),
                adamw: "adamw-state.bpk".into(),
                muon: "muon-state.bpk".into(),
                gradient_accumulator: None,
                update_clock: 1,
            }],
            memory_update: TrainingMemoryUpdateState::PeriodicSleep,
            sleep: Some(cursor),
            artifacts: Vec::new(),
            evaluator_hashes: Vec::new(),
            rng_streams: vec![
                RngStreamState {
                    name: DATA_RNG_STREAM.into(),
                    seed: 0,
                    counter: 1,
                },
                RngStreamState {
                    name: MODEL_RNG_STREAM.into(),
                    seed: 0,
                    counter: 1,
                },
            ],
            wake_context_buffer: Vec::new(),
            quantization: Some(QuantizationTrainingState {
                format: "binary_g128".into(),
                fake_quant_active: true,
                calibration_step: 1,
                manifest: None,
                candidate_weights_sha256: None,
                teacher_hash: None,
            }),
        };
        let metrics_path = directory.path().join("metrics.jsonl");
        let mut metrics = MetricWriter::create(&metrics_path, "qat-sleep-boundary").unwrap();
        let error = publish_quantization_phase_candidate(
            &model,
            plan,
            0,
            phase,
            &mut state,
            &mut metrics,
            directory.path(),
            Some(&pre_sleep.sha256),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("sleep boundary"), "{error}");
        assert!(state.quantization.as_ref().unwrap().manifest.is_none());

        // Model activation stands in for the accepted consolidation result;
        // the lifecycle invariant under test is that this post-boundary state,
        // never the pre-boundary QAT weights, becomes the sealed candidate.
        let mut post_sleep_model = model.clone();
        post_sleep_model
            .activate_memory_slot_all_layers(1, 0)
            .unwrap();
        let post_sleep =
            publish_sleep_model(&post_sleep_model, &directory.path().join("sleep-models")).unwrap();
        assert_ne!(post_sleep.sha256, pre_sleep.sha256);
        let cursor = state.sleep.as_mut().unwrap();
        cursor.sleep.due_senders.clear();
        cursor.sleep.due_clocks.clear();
        cursor.sleep.tiers[1].slots[0].active = true;
        cursor.sleep.tiers[1].slots[0].generation = 1;
        cursor.record_wake_checkpoint(post_sleep.clone()).unwrap();

        publish_quantization_phase_candidate(
            &post_sleep_model,
            plan,
            0,
            phase,
            &mut state,
            &mut metrics,
            directory.path(),
            Some(&post_sleep.sha256),
        )
        .unwrap();
        let manifest = PathBuf::from(
            state
                .quantization
                .as_ref()
                .unwrap()
                .manifest
                .as_ref()
                .unwrap(),
        );
        let publication = open_qat_candidate(manifest.parent().unwrap()).unwrap();
        assert_eq!(publication.weights_sha256, post_sleep.sha256);
        assert_ne!(publication.weights_sha256, pre_sleep.sha256);
        let sealed = fs::read(&manifest).unwrap();

        // Historical candidates are not execution inputs for this resume. A
        // missing old archive must not trigger multi-gigabyte validation or
        // prevent the current, independently authenticated candidate from
        // resuming.
        state.artifacts.push(ArtifactRef {
            kind: "hquant_candidate".into(),
            manifest: directory
                .path()
                .join("quantized-candidates/historical/candidate.json")
                .to_string_lossy()
                .into_owned(),
            hash: format!("sha256:{}", "4".repeat(64)),
        });

        // Resume authentication reopens the same stable key using the sealed
        // checkpoint weights identity; it neither serializes the mutable model
        // nor republishes or changes any candidate bytes.
        verify_resumed_quantization_candidate(&state, phase, 1, &post_sleep.sha256).unwrap();
        let corruptions: [fn(&mut QuantizationTrainingState); 5] = [
            |quantization: &mut QuantizationTrainingState| {
                quantization.format = "full_precision".into();
            },
            |quantization: &mut QuantizationTrainingState| {
                quantization.fake_quant_active = true;
            },
            |quantization: &mut QuantizationTrainingState| {
                quantization.calibration_step = 0;
            },
            |quantization: &mut QuantizationTrainingState| {
                quantization.teacher_hash = Some(format!("sha256:{}", "3".repeat(64)));
            },
            |quantization: &mut QuantizationTrainingState| {
                quantization.candidate_weights_sha256 = Some(format!("sha256:{}", "5".repeat(64)));
            },
        ];
        for mutate in corruptions {
            let mut corrupted = state.clone();
            mutate(corrupted.quantization.as_mut().unwrap());
            assert!(
                verify_resumed_quantization_candidate(&corrupted, phase, 1, &post_sleep.sha256,)
                    .is_err(),
                "resume accepted quantization state that differs from its workflow clock"
            );
        }
        assert!(
            verify_resumed_quantization_candidate(&state, phase, 1, &pre_sleep.sha256).is_err(),
            "resume accepted a candidate from different authenticated checkpoint weights"
        );
        assert_eq!(fs::read(&manifest).unwrap(), sealed);
        assert_eq!(
            open_qat_candidate(manifest.parent().unwrap())
                .unwrap()
                .weights_sha256,
            post_sleep.sha256
        );
    }
}
