//! Objective-aware streaming optimization loop.

use super::*;

pub(super) fn train(args: TrainArgs) -> Result<()> {
    validate_train_args(&args)?;
    let curriculum = resolve_curriculum(&args)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let mut config = load_config(&args.config)?;
    config.vocab_size = tokenizer.vocab_size();
    validate_model_curriculum(&config, &curriculum)?;
    let signature = run_signature(&args, &curriculum, &config)?;
    fs::create_dir_all(&args.output)?;
    let token_cache_root = args
        .output
        .join(".token-cache")
        .join(stable_cache_id(&signature));
    fs::create_dir_all(&token_cache_root)?;
    let (stage_plan, total_steps) = plan_training(&curriculum, &tokenizer, &token_cache_root)?;
    ensure!(
        total_steps > 0,
        "training has zero complete optimizer steps"
    );
    let metrics_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(args.resume)
        .truncate(!args.resume)
        .open(args.output.join("metrics.jsonl"))?;
    let mut metrics = BufWriter::new(metrics_file);

    let device = hermes_llm::default_device().autodiff();
    device.seed(args.seed);
    let mut initial_model = Transformer::new(&config, &device)?;
    if let Some(path) = &args.checkpoint {
        load_safetensors(&mut initial_model, path)?;
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
        muon_parameter_ids = initial_model.muon_parameter_ids();
        ensure!(
            state.step < total_steps,
            "checkpoint step {} has already reached requested total {total_steps}",
            state.step
        );
        ensure!(
            state.stage < curriculum.stages.len(),
            "checkpoint stage {} is outside the requested {}-stage curriculum",
            state.stage,
            curriculum.stages.len()
        );
        let resume_stage = &curriculum.stages[state.stage];
        ensure!(
            state.epoch < resume_stage.epochs,
            "checkpoint epoch {} is outside curriculum stage `{}` with {} epochs",
            state.epoch,
            resume_stage.name,
            resume_stage.epochs
        );
        ensure!(
            state.steps_in_stage <= stage_plan[state.stage].steps,
            "checkpoint has {} steps in stage `{}`, whose planned total is {}",
            state.steps_in_stage,
            resume_stage.name,
            stage_plan[state.stage].steps
        );
        ensure!(
            state.curriculum_signature == signature,
            "checkpoint curriculum or training configuration differs from this invocation"
        );
        ensure!(
            state.step == 0 || state.tokens_seen > 0,
            "checkpoint has no cumulative token count and cannot safely resume"
        );
        Some(state)
    } else {
        None
    };

    fs::write(
        args.output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    fs::write(
        args.output.join("resolved-curriculum.json"),
        serde_json::to_vec_pretty(&curriculum)?,
    )?;

    let mut muon_accumulator = GradientsAccumulator::new();
    let mut adamw_accumulator = GradientsAccumulator::new();
    let initial_parameter_ids = parameter_ids(&initial_model);
    // Optimizer::step consumes the module; Option lets the streaming callback
    // move it out and replace it without cloning model parameters.
    let mut model = Some(initial_model);
    let mut step = resume_state.as_ref().map_or(0, |state| state.step);
    let mut tokens_seen = resume_state.as_ref().map_or(0, |state| state.tokens_seen);
    let mut micro_step = 0;
    let mut loss_sum: Option<Tensor<1>> = None;
    let mut router_loss_sum: Option<Tensor<1>> = None;
    let mut retrieval_correct_sum: Option<Tensor<1>> = None;
    let mut step_stats = BatchStats::default();
    let mut optimizer_step_started = Instant::now();
    let mut training_state = resume_state.clone().unwrap_or(TrainingState {
        version: TRAINING_STATE_VERSION,
        step: 0,
        stage: 0,
        epoch: 0,
        samples_in_stage: 0,
        steps_in_stage: 0,
        tokens_seen: 0,
        curriculum_signature: signature.clone(),
        parameter_ids: initial_parameter_ids.clone(),
    });

    let stage_summary = curriculum
        .stages
        .iter()
        .zip(&stage_plan)
        .map(|(stage, plan)| {
            format!(
                "{}:{}:samples={}:steps={}",
                stage.name,
                stage.objective.name(),
                plan.samples
                    .map_or_else(|| "streaming".to_owned(), |n| n.to_string()),
                plan.steps,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "model={} params={} muon_matrices={} device={device:?} stages=[{stage_summary}] steps={total_steps}",
        config.name,
        model.as_ref().unwrap().num_parameters(),
        muon_parameter_ids.len(),
    );

    'stages: for (stage_index, stage) in curriculum.stages.iter().enumerate() {
        if resume_state
            .as_ref()
            .is_some_and(|state| stage_index < state.stage)
        {
            continue;
        }
        let mut steps_in_stage = resume_state
            .as_ref()
            .filter(|state| state.stage == stage_index)
            .map_or(0, |state| state.steps_in_stage);
        if steps_in_stage >= stage_plan[stage_index].steps {
            continue;
        }
        let mut stage_limit_reached = false;

        for epoch in 0..stage.epochs {
            if resume_state
                .as_ref()
                .is_some_and(|state| stage_index == state.stage && epoch < state.epoch)
            {
                continue;
            }
            let samples_to_skip = resume_state
                .as_ref()
                .filter(|state| state.stage == stage_index && state.epoch == epoch)
                .map_or(0, |state| state.samples_in_stage);
            let mut samples_in_stage = 0;
            training_state = TrainingState {
                version: TRAINING_STATE_VERSION,
                step,
                stage: stage_index,
                epoch,
                samples_in_stage,
                steps_in_stage,
                tokens_seen,
                curriculum_signature: signature.clone(),
                parameter_ids: initial_parameter_ids.clone(),
            };
            let mut batch = Vec::with_capacity(stage.batch_size);
            let shuffle_seed = args
                .seed
                .wrapping_add((stage_index as u64) << 32)
                .wrapping_add(epoch as u64);
            let tokenizer_ref = &tokenizer;
            let objective = stage.objective.clone();
            let token_cache_path = token_cache_root.join(format!("stage-{stage_index:03}.tokens"));
            std::thread::scope(|threads| -> Result<()> {
                let prefetch_capacity = stage
                    .batch_size
                    .checked_mul(2)
                    .context("training prefetch capacity overflows usize")?;
                let (sender, receiver) = std::sync::mpsc::sync_channel(prefetch_capacity);
                let reader = threads.spawn(move || {
                    visit_samples(
                        &stage.data,
                        &objective,
                        tokenizer_ref,
                        SampleStreamConfig {
                            seq_len: stage.sequence_length,
                            shuffle_buffer: stage.shuffle_buffer,
                            seed: shuffle_seed,
                            token_cache: Some(&token_cache_path),
                        },
                        |sample| Ok(sender.send(sample).is_ok()),
                    )
                });
                for sample in &receiver {
                    samples_in_stage = samples_in_stage
                        .checked_add(1)
                        .context("stage sample count overflows usize")?;
                    training_state.samples_in_stage = samples_in_stage;
                    if samples_in_stage <= samples_to_skip {
                        continue;
                    }
                    batch.push(sample);
                    if batch.len() < stage.batch_size {
                        continue;
                    }

                    let training_batch = make_batch(&batch, stage.sequence_length, &device)?;
                    batch.clear();
                    let current = model.as_ref().unwrap();
                    if micro_step == 0 {
                        optimizer_step_started = Instant::now();
                    }
                    let (task_loss, router_loss, batch_stats, retrieval_correct) =
                        objective_loss(current, training_batch, &stage.objective)?;
                    let detached_loss = task_loss.clone().detach();
                    loss_sum = Some(match loss_sum.take() {
                        Some(sum) => sum + detached_loss,
                        None => detached_loss,
                    });
                    if let Some(router_loss) = &router_loss {
                        let detached = router_loss.clone().detach();
                        router_loss_sum = Some(match router_loss_sum.take() {
                            Some(sum) => sum + detached,
                            None => detached,
                        });
                    }
                    add_batch_stats(&mut step_stats, batch_stats)?;
                    if let Some(correct) = retrieval_correct {
                        retrieval_correct_sum = Some(match retrieval_correct_sum.take() {
                            Some(sum) => sum + correct,
                            None => correct,
                        });
                    }
                    let optimized_loss = match router_loss {
                        Some(router_loss) => task_loss + router_loss,
                        None => task_loss,
                    };
                    let mut grads = optimized_loss
                        .mul_scalar(stage.loss_weight)
                        .div_scalar(stage.gradient_accumulation as f64)
                        .backward();
                    let muon_grads =
                        GradientsParams::from_params(&mut grads, current, &muon_parameter_ids);
                    let adamw_grads = GradientsParams::from_module(&mut grads, current);
                    muon_accumulator.accumulate(current, muon_grads);
                    adamw_accumulator.accumulate(current, adamw_grads);
                    micro_step += 1;

                    if micro_step == stage.gradient_accumulation {
                        let lr =
                            learning_rate(&args, step + 1, total_steps) * stage.learning_rate_scale;
                        let muon_lr = lr * MUON_LR_SCALE;
                        let loss = loss_sum
                            .take()
                            .expect("an optimizer step must contain a loss")
                            .div_scalar(stage.gradient_accumulation as f32);
                        let loss = scalar_value(loss)?;
                        let router_loss = router_loss_sum
                            .take()
                            .map(|sum| {
                                scalar_value(sum.div_scalar(stage.gradient_accumulation as f32))
                            })
                            .transpose()?;
                        let optimized_loss = loss + router_loss.unwrap_or(0.0);
                        let weighted_loss = optimized_loss * stage.loss_weight as f32;
                        let retrieval_accuracy = retrieval_correct_sum
                            .take()
                            .map(scalar_value)
                            .transpose()?
                            .map(|correct| correct / step_stats.examples as f32);
                        if !loss.is_finite()
                            || router_loss.is_some_and(|loss| !loss.is_finite())
                            || !weighted_loss.is_finite()
                        {
                            bail!(
                                "non-finite loss at optimizer step {}: task={loss}, router={router_loss:?}, weighted={weighted_loss}",
                                step + 1
                            );
                        }
                        let mut muon_grads = muon_accumulator.grads();
                        let mut adamw_grads = adamw_accumulator.grads();
                        let layer_grad_norms = (args.layer_metrics_every > 0
                            && (step + 1) % args.layer_metrics_every == 0)
                            .then(|| layer_gradient_norms(current, &muon_grads, &adamw_grads))
                            .transpose()?;
                        let grad_norm = gradient_norm_and_clip(
                            current,
                            &mut muon_grads,
                            &mut adamw_grads,
                            args.grad_clip,
                        )?;
                        if !grad_norm.is_finite() {
                            bail!(
                                "non-finite gradient norm at optimizer step {}: {grad_norm}",
                                step + 1
                            );
                        }
                        let current = model.take().unwrap();
                        let current = muon_optimizer.step(muon_lr, current, muon_grads)?;
                        model = Some(adamw_optimizer.step(lr.into(), current, adamw_grads));
                        step += 1;
                        steps_in_stage = steps_in_stage
                            .checked_add(1)
                            .context("stage optimizer-step count overflows usize")?;
                        tokens_seen = tokens_seen
                            .checked_add(step_stats.compute_tokens)
                            .context("cumulative model-token count overflows usize")?;
                        training_state.step = step;
                        training_state.steps_in_stage = steps_in_stage;
                        training_state.tokens_seen = tokens_seen;
                        let step_seconds = optimizer_step_started.elapsed().as_secs_f64();
                        let tokens_per_second = step_stats.compute_tokens as f64 / step_seconds;
                        println!(
                            "stage={}/{} name={} objective={} epoch={} step={step}/{total_steps} stage_step={} loss={loss:.6} lr={lr:.3e} grad_norm={grad_norm:.3} tokens_per_second={tokens_per_second:.0}",
                            stage_index + 1,
                            curriculum.stages.len(),
                            stage.name,
                            stage.objective.name(),
                            epoch + 1,
                            steps_in_stage,
                        );
                        let mut metric = serde_json::json!({
                            "step": step,
                            "stage": stage_index + 1,
                            "stage_name": stage.name,
                            "stage_step": steps_in_stage,
                            "objective": stage.objective.name(),
                            "epoch": epoch + 1,
                            "loss": loss,
                            "weighted_loss": weighted_loss,
                            "optimized_loss": optimized_loss,
                            "loss_weight": stage.loss_weight,
                            "lr": lr,
                            "muon_lr": muon_lr,
                            "grad_norm": grad_norm,
                            "step_seconds": step_seconds,
                            "tokens_per_second": tokens_per_second,
                            "tokens": tokens_seen,
                            "compute_tokens": step_stats.compute_tokens,
                            "supervised_tokens": step_stats.supervised_tokens,
                            "examples": step_stats.examples,
                            "truncated_tokens": step_stats.truncated_tokens,
                            "retrieval_candidates": step_stats.retrieval_candidates,
                            "sequence_length": stage.sequence_length,
                            "batch_size": stage.batch_size,
                            "gradient_accumulation": stage.gradient_accumulation,
                        });
                        metric.as_object_mut().unwrap().insert(
                            format!("{}_loss", stage.objective.name()),
                            serde_json::json!(loss),
                        );
                        if let Some(router_loss) = router_loss {
                            metric.as_object_mut().unwrap().insert(
                                "router_aux_loss".to_owned(),
                                serde_json::json!(router_loss),
                            );
                        }
                        if let Some(accuracy) = retrieval_accuracy {
                            metric.as_object_mut().unwrap().insert(
                                "retrieval_accuracy".to_owned(),
                                serde_json::json!(accuracy),
                            );
                        }
                        if let Some(layer_grad_norms) = layer_grad_norms {
                            metric.as_object_mut().unwrap().insert(
                                "layer_grad_norms".to_owned(),
                                serde_json::json!(layer_grad_norms),
                            );
                        }
                        serde_json::to_writer(&mut metrics, &metric)?;
                        metrics.write_all(b"\n")?;
                        metrics.flush()?;
                        if args.checkpoint_every > 0 && step % args.checkpoint_every == 0 {
                            save_training_checkpoint(
                                model.as_ref().unwrap(),
                                &adamw_optimizer,
                                &muon_optimizer,
                                &training_state,
                                &args.output,
                            )?;
                            println!("checkpointed {}", args.output.display());
                        }
                        micro_step = 0;
                        step_stats = BatchStats::default();
                        if step >= total_steps
                            || stage.steps.is_some_and(|limit| steps_in_stage >= limit)
                        {
                            stage_limit_reached = true;
                            break;
                        }
                    }
                }
                if !batch.is_empty() {
                    println!(
                        "stage={} epoch={} dropped_incomplete_batch_examples={}",
                        stage.name,
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
                    "stage={} epoch={} dropped_incomplete_optimizer_microbatches={micro_step}",
                    stage.name,
                    epoch + 1
                );
                muon_accumulator = GradientsAccumulator::new();
                adamw_accumulator = GradientsAccumulator::new();
                micro_step = 0;
                loss_sum = None;
                router_loss_sum = None;
                retrieval_correct_sum = None;
                step_stats = BatchStats::default();
            }
            if step >= total_steps {
                break 'stages;
            }
            if stage_limit_reached {
                break;
            }
        }

        ensure!(
            steps_in_stage == stage_plan[stage_index].steps,
            "curriculum stage `{}` requested {} optimizer steps, but its data and epochs produced {steps_in_stage}",
            stage.name,
            stage_plan[stage_index].steps
        );
        // Stage-boundary publication prevents a relaunch from replaying a
        // completed short fine-tuning stage before the next periodic save.
        save_training_checkpoint(
            model.as_ref().unwrap(),
            &adamw_optimizer,
            &muon_optimizer,
            &training_state,
            &args.output,
        )?;
        println!("checkpointed completed stage `{}`", stage.name);
    }
    ensure!(
        step == total_steps,
        "requested {total_steps} optimizer steps, but the data produced only {step} complete steps"
    );

    save_training_checkpoint(
        model.as_ref().unwrap(),
        &adamw_optimizer,
        &muon_optimizer,
        &training_state,
        &args.output,
    )?;
    println!("saved {}", args.output.display());
    Ok(())
}
