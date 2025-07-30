//! Objective-aware streaming optimization loop.

use super::*;

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
        load_safetensors(
            &mut checkpoint_model,
            Path::new(&checkpoint.live_checkpoint.uri),
        )?;
        self.state.sleep = Some(checkpoint.clone());
        self.state.metric_records = self.metrics.state().records;
        let _ = save_training_checkpoint_with_evidence(
            &checkpoint_model,
            self.adamw,
            self.muon,
            self.state,
            self.metrics,
            self.output,
        )?;
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
    let mut cursor = state
        .sleep
        .take()
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
    let publication = publish_qat_candidate(
        model,
        &output.join("quantized-candidates"),
        &key,
        &plan.recipe,
    )?;
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
    quantization.format = format!("{:?}", plan.recipe.format).to_ascii_lowercase();
    quantization.fake_quant_active = false;
    quantization.calibration_step = state.global_step as u64;
    quantization.manifest = Some(manifest.clone());
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
        .take()
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
    let source = bind_post_sleep_quantization_source(periodic_runtime, phase, state, model)?;
    publish_quantization_phase_candidate(
        model,
        plan,
        phase_index,
        phase,
        state,
        metrics,
        output,
        source.as_deref(),
    )?;
    Ok(true)
}

fn verify_resumed_quantization_candidate(
    state: &TrainingState,
    phase: &super::wake::ResolvedWakePhase,
    planned_steps: usize,
    model: &Transformer,
) -> Result<()> {
    for artifact in state
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == "hquant_candidate")
    {
        let manifest = Path::new(&artifact.manifest);
        ensure!(
            manifest
                .file_name()
                .is_some_and(|name| name == "candidate.json"),
            "HQUANT artifact does not name candidate.json"
        );
        let candidate_root = manifest
            .parent()
            .context("HQUANT artifact manifest has no candidate root")?;
        let publication = open_qat_candidate(candidate_root)?;
        ensure!(
            publication.candidate_manifest_path == manifest
                && publication.candidate_manifest_sha256 == artifact.hash,
            "HQUANT artifact receipt differs from its validated candidate"
        );
    }
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
    let root = candidate
        .parent()
        .context("quantization candidate has no store root")?;
    let key = candidate
        .file_name()
        .and_then(|name| name.to_str())
        .context("quantization candidate key is not UTF-8")?;
    let publication = publish_qat_candidate(model, root, key, &plan.recipe)?;
    ensure!(
        publication.candidate_manifest_path == manifest,
        "quantization candidate retry resolved to another path"
    );
    let artifact = state
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.kind == "hquant_candidate" && artifact.manifest == manifest.to_string_lossy()
        })
        .context("completed quantization phase has no candidate artifact receipt")?;
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
    let data_manifests = workflow
        .phases
        .iter()
        .map(|phase| phase_data_identity(&phase.data, &tokenizer, &tokenizer_hash))
        .collect::<Result<Vec<_>>>()?;
    let initial_checkpoint_sha256 = args.checkpoint.as_deref().map(file_sha256).transpose()?;
    let signature = run_signature(
        &args,
        &workflow,
        &config,
        &data_manifests,
        initial_checkpoint_sha256.clone(),
    )?;
    if args.print_run_signature {
        println!("{signature}");
        return Ok(());
    }
    fs::create_dir_all(&args.output)?;
    let token_cache_root = args
        .output
        .join(".token-cache")
        .join(stable_cache_id(&signature));
    fs::create_dir_all(&token_cache_root)?;
    let (phase_plan, total_steps) = plan_training(&workflow, &tokenizer, &token_cache_root)?;
    ensure!(
        total_steps > 0,
        "training has zero complete optimizer steps"
    );
    let run_id = stable_cache_id(&signature);
    let metrics_path = args.output.join("metrics.jsonl");

    let device = hermes_llm::default_device().autodiff();
    device.seed(args.seed);
    let mut initial_model = Transformer::new(&config, &device)?;
    if let Some(path) = &args.checkpoint {
        load_safetensors(&mut initial_model, path)?;
        ensure!(
            Some(file_sha256(path)?) == initial_checkpoint_sha256,
            "initial checkpoint changed after its run signature was computed"
        );
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
    let resume_state = if args.resume {
        let (optimizer, state) = load_training_state(
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
        Some(state)
    } else {
        None
    };
    if let Some(state) = &resume_state {
        // Authenticate the exact runtime identity before constructing its
        // factory. Factory construction validates/creates configured stores,
        // so a mismatched resume must fail before that first side effect.
        preflight_resumed_sleep_runtime(&args, &workflow, state)?;
        verify_resumed_quantization_candidate(
            state,
            &workflow.phases[state.phase],
            phase_plan[state.phase].steps,
            &initial_model,
        )?;
    }
    let periodic_runtime =
        PeriodicTrainingRuntime::load(&args, &workflow, &signature, &initial_model, &device)?;
    if let Some(runtime) = &periodic_runtime {
        let wake_ids = runtime
            .bank
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
        if let Some(state) = &resume_state {
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
    } else {
        ensure!(
            resume_state
                .as_ref()
                .is_none_or(|state| state.sleep.is_none()),
            "ordinary-model checkpoint unexpectedly contains native sleep state"
        );
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
    let mut tier_accumulators = periodic_runtime
        .as_ref()
        .map(|runtime| {
            runtime.bank.scopes().map(|scopes| {
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
    let mut step_wake_contexts = Vec::<Vec<i64>>::new();
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
                    kind: metric_phase_kind(phase.phase_kind),
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
        )? {
            training_state.metric_records = metrics.state().records;
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
            let mut records_in_phase = 0;
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
                sleep: native_sleep,
                artifacts,
                evaluator_hashes,
                rng_streams: vec![
                    RngStreamState {
                        name: DATA_RNG_STREAM.into(),
                        seed: shuffle_seed(args.seed, phase_index, epoch),
                        counter: 0,
                    },
                    model_rng,
                ],
                wake_context_buffer: training_state.wake_context_buffer.clone(),
                quantization: phase.quantization.as_ref().map(|plan| {
                    let format = plan.format_at(step as u64);
                    QuantizationTrainingState {
                        format: format.map_or_else(
                            || "full_precision".to_owned(),
                            |format| format!("{format:?}").to_ascii_lowercase(),
                        ),
                        fake_quant_active: format.is_some(),
                        calibration_step: step as u64,
                        manifest: None,
                        teacher_hash: quantization_teacher
                            .as_ref()
                            .map(|teacher| teacher.sha256.clone()),
                        transaction: None,
                    }
                }),
            };
            let mut batch = Vec::with_capacity(phase.batch_size);
            let shuffle_seed = shuffle_seed(args.seed, phase_index, epoch);
            let tokenizer_ref = &tokenizer;
            let objective = phase.objective.clone();
            let token_cache_path = token_cache_root.join(format!("phase-{phase_index:03}.tokens"));
            std::thread::scope(|threads| -> Result<()> {
                let prefetch_capacity = phase
                    .batch_size
                    .checked_mul(phase.gradient_accumulation)
                    .and_then(|capacity| capacity.checked_mul(2))
                    .context("training prefetch capacity overflows usize")?;
                let (sender, receiver) = std::sync::mpsc::sync_channel(prefetch_capacity);
                let reader = threads.spawn(move || {
                    visit_samples(
                        &phase.data,
                        &objective,
                        tokenizer_ref,
                        SampleStreamConfig {
                            seq_len: phase.sequence_length,
                            shuffle_buffer: phase.shuffle_buffer,
                            seed: shuffle_seed,
                            token_cache: Some(&token_cache_path),
                        },
                        |sample| Ok(sender.send(sample).is_ok()),
                    )
                });
                loop {
                    let input_wait_started = Instant::now();
                    let sample = match receiver.recv() {
                        Ok(sample) => sample,
                        Err(_) => break,
                    };
                    step_input_wait_seconds += input_wait_started.elapsed().as_secs_f64();
                    records_in_phase = records_in_phase
                        .checked_add(1)
                        .context("phase sample count overflows usize")?;
                    training_state.records_in_phase = records_in_phase;
                    rng_stream_mut(&mut training_state, DATA_RNG_STREAM)?.counter =
                        records_in_phase as u64;
                    if records_in_phase <= records_to_skip {
                        optimizer_step_started = Instant::now();
                        step_input_wait_seconds = 0.0;
                        continue;
                    }
                    batch.push(sample);
                    if batch.len() < phase.batch_size {
                        continue;
                    }

                    let transfer_started = Instant::now();
                    if phase.periodic_sleep.is_some() {
                        step_wake_contexts.extend(batch.iter().map(|sample| {
                            let tokens = sample.wake_context_tokens();
                            let keep = tokens.len().min(config.max_seq_len);
                            tokens[tokens.len() - keep..].to_vec()
                        }));
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
                    let quantized = phase
                        .quantization
                        .as_ref()
                        .and_then(|plan| plan.format_at(step as u64))
                        .map(|format| {
                            fake_quantized_transformer(
                                current,
                                format,
                                phase
                                    .quantization
                                    .as_ref()
                                    .expect("format came from plan")
                                    .recipe
                                    .quantize_embeddings,
                                phase
                                    .quantization
                                    .as_ref()
                                    .expect("format came from plan")
                                    .recipe
                                    .quantize_lm_head,
                            )
                        })
                        .transpose()?;
                    let forward_model = quantized.as_ref().map_or(current, |(staged, _)| staged);
                    let accelerator_started = Instant::now();
                    let distillation_loss = quantization_teacher
                        .as_ref()
                        .map(|teacher| {
                            quantization_forward_kl(
                                forward_model,
                                &teacher.model,
                                &training_batch,
                                &phase.objective,
                                teacher.temperature,
                            )
                        })
                        .transpose()?;
                    let (task_loss, router_loss, batch_stats, retrieval_correct) =
                        objective_loss(forward_model, training_batch, &phase.objective)?;
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
                    let backward_loss = if periodic_runtime.is_some() {
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
                    let mut adamw_grads = match &periodic_runtime {
                        Some(runtime) => {
                            let partitioned =
                                runtime.bank.partition_gradients(current, &mut grads)?;
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
                    if periodic_runtime.is_some() {
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
                        if periodic_runtime.is_some() {
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
                        if let Some(runtime) = &periodic_runtime {
                            runtime.bank.commit_tier_gradients(current, tier_grads, 1)?;
                        } else {
                            ensure!(
                                tier_grads.is_empty(),
                                "ordinary training produced memory-tier gradients"
                            );
                        }
                        let current = model.take().unwrap();
                        let current = muon_optimizer.step(muon_lr, current, muon_grads)?;
                        model = Some(adamw_optimizer.step(lr.into(), current, adamw_grads));
                        step_accelerator_seconds += accelerator_started.elapsed().as_secs_f64();
                        step += 1;
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
                        for (ordinal, token_ids) in step_wake_contexts.drain(..).enumerate() {
                            training_state.wake_context_buffer.push(
                                hermes_train::builtin_sleep_adapters::WakeContextRecord {
                                    id: format!("{}:{step}:{ordinal}", phase.name),
                                    optimizer_step: step as u64,
                                    token_ids,
                                },
                            );
                        }
                        let max_wake_context_records =
                            periodic_runtime.as_ref().map_or(0, |runtime| {
                                runtime.factory.config().max_wake_context_records
                            });
                        if training_state.wake_context_buffer.len() > max_wake_context_records {
                            let discard =
                                training_state.wake_context_buffer.len() - max_wake_context_records;
                            training_state.wake_context_buffer.drain(..discard);
                        }
                        if let Some(quantization) = &mut training_state.quantization {
                            let format = phase
                                .quantization
                                .as_ref()
                                .and_then(|plan| plan.format_at((step - 1) as u64));
                            quantization.format = format.map_or_else(
                                || "full_precision".to_owned(),
                                |format| format!("{format:?}").to_ascii_lowercase(),
                            );
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
                                kind: metric_phase_kind(phase.phase_kind),
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
                        if let Some(plan) = &phase.quantization {
                            let active_format = plan.format_at((step - 1) as u64);
                            let quantized_tensors =
                                quantized.as_ref().map_or(0, |(_, tensors)| *tensors as u64);
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
                                    tensors_quantized: quantized_tensors,
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
                            )?;
                        }
                        if args.checkpoint_every > 0 && step % args.checkpoint_every == 0 {
                            if let (Some(runtime), Some(sleep)) =
                                (&periodic_runtime, phase.periodic_sleep.as_ref())
                            {
                                let mut cursor = training_state
                                    .sleep
                                    .take()
                                    .context("periodic checkpoint has no sleep cursor")?;
                                let _ = runtime.checkpoint_wake(
                                    &mut cursor,
                                    model.as_ref().unwrap(),
                                    sleep,
                                )?;
                                training_state.sleep = Some(cursor);
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
                .take()
                .context("periodic phase boundary has no sleep cursor")?;
            let _ = runtime.checkpoint_wake(&mut cursor, model.as_ref().unwrap(), sleep)?;
            training_state.sleep = Some(cursor);
        }
        let context = MetricContext {
            global_step: training_state.global_step as u64,
            phase: MetricPhase {
                index: phase_index as u32,
                name: phase.name.clone(),
                kind: metric_phase_kind(phase.phase_kind),
            },
            checkpoint_hash: None,
        };
        drain_device_sampler(&mut device_sampler, &mut metrics, &context)?;
        training_state.metric_records = metrics.state().records;
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
            .take()
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
            kind: metric_phase_kind(final_phase.phase_kind),
        },
        checkpoint_hash: None,
    };
    shutdown_device_sampler(&mut device_sampler, &mut metrics, &final_context)?;
    training_state.metric_records = metrics.state().records;
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
        "{label} checkpoint_manifest={} checkpoint_manifest_sha256={} training_evidence={} training_evidence_sha256={}",
        publication.checkpoint_manifest.display(),
        publication.checkpoint_manifest_sha256,
        publication.training_evidence.display(),
        publication.training_evidence_sha256,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
                fake_quant_active: false,
                calibration_step: 1,
                manifest: None,
                teacher_hash: None,
                transaction: None,
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

        // Resume authentication reopens the same stable key with the same
        // post-sleep model; it neither republishes nor changes any bytes.
        verify_resumed_quantization_candidate(&state, phase, 1, &post_sleep_model).unwrap();
        assert_eq!(fs::read(&manifest).unwrap(), sealed);
        assert_eq!(
            open_qat_candidate(manifest.parent().unwrap())
                .unwrap()
                .weights_sha256,
            post_sleep.sha256
        );
    }
}
