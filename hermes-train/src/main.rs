use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail, ensure};
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::Tensor;
use burn_optim::{AdamWConfig, GradientsAccumulator, GradientsParams};
use clap::{Parser, Subcommand, ValueEnum};
use hermes_llm::{ModelDef, Tokenizer, Transformer, load_safetensors};

mod checkpoint;
mod data;
mod muon;

use checkpoint::{
    AdamWOptimizer, TrainingState, load_training_state, parameter_ids, save_training_checkpoint,
};
use data::{count_samples, make_batch, visit_samples};
use muon::BatchedMuon;

const MUON_LR_SCALE: f64 = 20.0;

#[derive(Parser)]
#[command(name = "hermes-train", about = "Hermes model training")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Pretrain or fine-tune a MAL-defined language model.
    Train(TrainArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Schedule {
    Wsd,
    Cosine,
}

#[derive(clap::Args)]
struct TrainArgs {
    /// MAL source or exported JSON model configuration.
    #[arg(long)]
    config: PathBuf,
    #[arg(short = 't', long)]
    tokenizer: PathBuf,
    /// Text or JSONL, optionally Zstandard-compressed. Repeat for more files.
    #[arg(short = 'd', long, required = true)]
    data: Vec<PathBuf>,
    /// Number of token sequences held for deterministic streaming shuffle; 0 disables shuffling.
    #[arg(long, default_value_t = 8192)]
    shuffle_buffer: usize,
    #[arg(short = 'o', long, default_value = "checkpoint")]
    output: PathBuf,
    #[arg(short = 'b', long, default_value_t = 8)]
    batch_size: usize,
    /// Number of microbatches per optimizer step.
    #[arg(long, default_value_t = 1)]
    grad_accum: usize,
    #[arg(short = 'e', long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 256)]
    seq_len: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0.1)]
    weight_decay: f32,
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,
    #[arg(long, default_value_t = 1000)]
    warmup_steps: usize,
    #[arg(long, value_enum, default_value_t = Schedule::Wsd)]
    schedule: Schedule,
    #[arg(long)]
    max_steps: Option<usize>,
    /// Save a resumable checkpoint every N optimizer steps; 0 disables it.
    #[arg(long, default_value_t = 100)]
    checkpoint_every: usize,
    /// Safetensors checkpoint to fine-tune from.
    #[arg(long, conflicts_with = "resume")]
    checkpoint: Option<PathBuf>,
    /// Resume weights, optimizer state, schedule, and corpus position from --output.
    #[arg(long)]
    resume: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn load_config(path: &Path) -> Result<ModelDef> {
    if path.extension().is_some_and(|ext| ext == "mal") {
        return hermes_llm::parse_mal_file(path);
    }
    ModelDef::from_json(path)
}

struct SquaredGradientNorm<'a> {
    grads: &'a GradientsParams,
    sum: Option<Tensor<1>>,
}

impl ModuleVisitor for SquaredGradientNorm<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(grad) = self.grads.get::<D>(param.id) else {
            return;
        };
        let squared = grad.square().sum();
        self.sum = Some(match self.sum.take() {
            Some(sum) => sum + squared,
            None => squared,
        });
    }
}

fn squared_gradient_norm(model: &Transformer, grads: &GradientsParams) -> Option<Tensor<1>> {
    let mut visitor = SquaredGradientNorm { grads, sum: None };
    model.visit(&mut visitor);
    visitor.sum
}

struct GradientScaler<'a> {
    grads: &'a mut GradientsParams,
    scale: f32,
}

impl ModuleVisitor for GradientScaler<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(grad) = self.grads.remove::<D>(param.id) else {
            return;
        };
        self.grads
            .register::<D>(param.id, grad.mul_scalar(self.scale));
    }
}

fn scale_gradients(model: &Transformer, grads: &mut GradientsParams, scale: f32) {
    model.visit(&mut GradientScaler { grads, scale });
}

fn gradient_norm_and_clip(
    model: &Transformer,
    muon_grads: &mut GradientsParams,
    adamw_grads: &mut GradientsParams,
    max_norm: f32,
) -> Result<f32> {
    let sum = match (
        squared_gradient_norm(model, muon_grads),
        squared_gradient_norm(model, adamw_grads),
    ) {
        (Some(muon), Some(adamw)) => muon + adamw,
        (Some(sum), None) | (None, Some(sum)) => sum,
        (None, None) => return Ok(0.0),
    };
    let norm = scalar_value(sum.sqrt())?;
    if max_norm > 0.0 && norm > max_norm {
        let scale = max_norm / norm;
        scale_gradients(model, muon_grads, scale);
        scale_gradients(model, adamw_grads, scale);
    }
    Ok(norm)
}

fn scalar_value(tensor: Tensor<1>) -> Result<f32> {
    Ok(tensor.into_data().convert::<f32>().to_vec::<f32>()?[0])
}

fn learning_rate(args: &TrainArgs, step: usize, total_steps: usize) -> f64 {
    if step < args.warmup_steps {
        return args.lr * step as f64 / args.warmup_steps.max(1) as f64;
    }
    let min_lr = args.lr * 0.1;
    let decay_start = match args.schedule {
        Schedule::Wsd => (total_steps as f64 * 0.9) as usize,
        Schedule::Cosine => args.warmup_steps,
    };
    if step < decay_start {
        return args.lr;
    }
    let progress = (step - decay_start) as f64 / (total_steps - decay_start).max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos());
    min_lr + cosine * (args.lr - min_lr)
}

fn validate_train_args(args: &TrainArgs) -> Result<()> {
    ensure!(
        !args.data.is_empty(),
        "at least one data source is required"
    );
    ensure!(args.batch_size > 0, "batch_size must be positive");
    ensure!(args.grad_accum > 0, "grad_accum must be positive");
    ensure!(args.epochs > 0, "epochs must be positive");
    ensure!(args.seq_len > 0, "seq_len must be positive");
    ensure!(
        args.lr.is_finite() && args.lr > 0.0,
        "lr must be finite and positive"
    );
    ensure!(
        args.weight_decay.is_finite() && args.weight_decay >= 0.0,
        "weight_decay must be finite and non-negative"
    );
    ensure!(
        args.grad_clip.is_finite() && args.grad_clip >= 0.0,
        "grad_clip must be finite and non-negative"
    );
    ensure!(
        args.max_steps.is_none_or(|steps| steps > 0),
        "max_steps must be positive when set"
    );
    Ok(())
}

fn train(args: TrainArgs) -> Result<()> {
    validate_train_args(&args)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let mut config = load_config(&args.config)?;
    config.vocab_size = tokenizer.vocab_size();
    ensure!(
        args.seq_len <= config.max_seq_len,
        "seq_len {} exceeds model max_seq_len {}",
        args.seq_len,
        config.max_seq_len
    );
    let sample_counts = match args.max_steps {
        Some(_) => None,
        None => Some(
            args.data
                .iter()
                .map(|path| count_samples(path, &tokenizer, args.seq_len))
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    let total_steps = match (args.max_steps, &sample_counts) {
        (Some(steps), _) => steps,
        (None, Some(counts)) => counts
            .iter()
            .map(|samples| {
                let microbatches = samples / args.batch_size;
                (microbatches / args.grad_accum).saturating_mul(args.epochs)
            })
            .sum(),
        (None, None) => unreachable!(),
    };
    ensure!(
        total_steps > 0,
        "training has zero complete optimizer steps"
    );

    fs::create_dir_all(&args.output)?;
    fs::write(
        args.output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
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
            state.stage < args.data.len() && state.epoch < args.epochs,
            "checkpoint corpus position is outside the requested curriculum"
        );
        Some(state)
    } else {
        None
    };
    let mut muon_accumulator = GradientsAccumulator::new();
    let mut adamw_accumulator = GradientsAccumulator::new();
    let initial_parameter_ids = parameter_ids(&initial_model);
    // Optimizer::step consumes the module; Option lets the streaming callback
    // move it out and replace it without cloning model parameters.
    let mut model = Some(initial_model);
    let mut step = resume_state.as_ref().map_or(0, |state| state.step);
    let mut micro_step = 0;
    let mut loss_sum: Option<Tensor<1>> = None;
    let mut optimizer_step_started = Instant::now();
    let mut training_state = resume_state.clone().unwrap_or(TrainingState {
        step: 0,
        stage: 0,
        epoch: 0,
        samples_in_stage: 0,
        parameter_ids: initial_parameter_ids.clone(),
    });

    let sample_summary = sample_counts.as_ref().map_or_else(
        || "streaming".to_owned(),
        |counts| {
            counts
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        },
    );
    println!(
        "model={} params={} muon_matrices={} device={device:?} stage_samples={sample_summary} steps={total_steps} shuffle_buffer={}",
        config.name,
        model.as_ref().unwrap().num_parameters(),
        muon_parameter_ids.len(),
        args.shuffle_buffer,
    );

    'stages: for (stage, path) in args.data.iter().enumerate() {
        if resume_state
            .as_ref()
            .is_some_and(|state| stage < state.stage)
        {
            continue;
        }
        for epoch in 0..args.epochs {
            if resume_state
                .as_ref()
                .is_some_and(|state| stage == state.stage && epoch < state.epoch)
            {
                continue;
            }
            let samples_to_skip = resume_state
                .as_ref()
                .filter(|state| state.stage == stage && state.epoch == epoch)
                .map_or(0, |state| state.samples_in_stage);
            let mut samples_in_stage = 0;
            training_state = TrainingState {
                step,
                stage,
                epoch,
                samples_in_stage,
                parameter_ids: initial_parameter_ids.clone(),
            };
            let mut batch = Vec::with_capacity(args.batch_size);
            let shuffle_seed = args.seed.wrapping_add((stage * args.epochs + epoch) as u64);
            // Reading, decompression, and tokenization run on a background
            // thread; the bounded channel keeps two batches of samples ready
            // so the accelerator never idles on batch preparation. Dropping
            // the receiver (early stop or an error) hangs up the sender and
            // the reader unwinds cleanly.
            let seq_len = args.seq_len;
            let shuffle_buffer = args.shuffle_buffer;
            let tokenizer_ref = &tokenizer;
            std::thread::scope(|threads| -> Result<()> {
                let (sender, receiver) = std::sync::mpsc::sync_channel(args.batch_size * 2);
                let reader = threads.spawn(move || {
                    visit_samples(
                        path,
                        tokenizer_ref,
                        seq_len,
                        shuffle_buffer,
                        shuffle_seed,
                        |sample| Ok(sender.send(sample).is_ok()),
                    )
                });
                for sample in &receiver {
                    samples_in_stage += 1;
                    training_state.samples_in_stage = samples_in_stage;
                    if samples_in_stage <= samples_to_skip {
                        continue;
                    }
                    batch.push(sample);
                    if batch.len() < args.batch_size {
                        continue;
                    }

                    let (inputs, targets) = make_batch(&batch, args.seq_len, &device);
                    batch.clear();
                    let current = model.as_ref().unwrap();
                    if micro_step == 0 {
                        optimizer_step_started = Instant::now();
                    }
                    let loss = current.forward_loss(inputs, targets);
                    let detached_loss = loss.clone().detach();
                    loss_sum = Some(match loss_sum.take() {
                        Some(sum) => sum + detached_loss,
                        None => detached_loss,
                    });
                    let mut grads = loss.div_scalar(args.grad_accum as f64).backward();
                    let muon_grads =
                        GradientsParams::from_params(&mut grads, current, &muon_parameter_ids);
                    let adamw_grads = GradientsParams::from_module(&mut grads, current);
                    muon_accumulator.accumulate(current, muon_grads);
                    adamw_accumulator.accumulate(current, adamw_grads);
                    micro_step += 1;

                    if micro_step == args.grad_accum {
                        let lr = learning_rate(&args, step + 1, total_steps);
                        let muon_lr = lr * MUON_LR_SCALE;
                        let loss = loss_sum
                            .take()
                            .expect("an optimizer step must contain a loss")
                            .div_scalar(args.grad_accum as f32);
                        let loss = scalar_value(loss)?;
                        if !loss.is_finite() {
                            bail!("non-finite loss at optimizer step {}: {loss}", step + 1);
                        }
                        let mut muon_grads = muon_accumulator.grads();
                        let mut adamw_grads = adamw_accumulator.grads();
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
                        training_state.step = step;
                        let step_seconds = optimizer_step_started.elapsed().as_secs_f64();
                        let step_tokens = args.batch_size * args.grad_accum * args.seq_len;
                        let tokens_per_second = step_tokens as f64 / step_seconds;
                        println!(
                            "stage={}/{} epoch={} step={step}/{total_steps} loss={:.6} lr={lr:.3e} grad_norm={grad_norm:.3} tokens_per_second={tokens_per_second:.0}",
                            stage + 1,
                            args.data.len(),
                            epoch + 1,
                            loss
                        );
                        serde_json::to_writer(
                            &mut metrics,
                            &serde_json::json!({
                                "step": step,
                                "stage": stage + 1,
                                "epoch": epoch + 1,
                                "loss": loss,
                                "lr": lr,
                                "muon_lr": muon_lr,
                                "grad_norm": grad_norm,
                                "step_seconds": step_seconds,
                                "tokens_per_second": tokens_per_second,
                                "tokens": step * args.batch_size * args.grad_accum * args.seq_len,
                            }),
                        )?;
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
                        if step >= total_steps {
                            break;
                        }
                    }
                }
                drop(receiver);
                reader
                    .join()
                    .expect("sample reader thread panicked")
                    .map(|_| ())
            })?;
            if step >= total_steps {
                break 'stages;
            }
            // Optimizer steps never span epoch or curriculum-stage boundaries.
            if micro_step != 0 {
                muon_accumulator = GradientsAccumulator::new();
                adamw_accumulator = GradientsAccumulator::new();
                micro_step = 0;
                loss_sum = None;
            }
        }
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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    match Cli::parse().command {
        Command::Train(args) => train(args),
    }
}

#[cfg(test)]
mod tests {
    use burn::module::{AutodiffModule, ParamId};
    use burn::tensor::{Int, TensorData};
    use hermes_llm::get_builtin_model;

    use super::*;

    fn small_hybrid() -> ModelDef {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 3;
        config.max_seq_len = 16;
        for block in config.pattern.as_mut().unwrap() {
            block.dropout = 0.0;
            block.attention.dropout = 0.0;
            block.attention.num_heads = Some(2);
            block.attention.num_kv_heads = Some(1);
            block.attention.head_dim = Some(4);
            block.ffn.dropout = 0.0;
            block.ffn.hidden_dim = Some(16);
        }
        config
    }

    fn valid_train_args() -> TrainArgs {
        TrainArgs {
            config: "config.mal".into(),
            tokenizer: "tokenizer.json".into(),
            data: vec!["corpus.jsonl".into()],
            shuffle_buffer: 8,
            output: "checkpoint".into(),
            batch_size: 2,
            grad_accum: 1,
            epochs: 1,
            seq_len: 8,
            lr: 3e-4,
            weight_decay: 0.1,
            grad_clip: 1.0,
            warmup_steps: 10,
            schedule: Schedule::Wsd,
            max_steps: Some(1),
            checkpoint_every: 0,
            checkpoint: None,
            resume: false,
            seed: 0,
        }
    }

    #[test]
    fn invalid_numeric_training_arguments_fail_before_loading_files() {
        type Invalidate = fn(&mut TrainArgs);
        let cases: [(&str, Invalidate); 8] = [
            ("batch_size", |args| args.batch_size = 0),
            ("grad_accum", |args| args.grad_accum = 0),
            ("epochs", |args| args.epochs = 0),
            ("seq_len", |args| args.seq_len = 0),
            ("lr", |args| args.lr = f64::NAN),
            ("weight_decay", |args| args.weight_decay = -0.1),
            ("grad_clip", |args| args.grad_clip = f32::INFINITY),
            ("max_steps", |args| args.max_steps = Some(0)),
        ];
        for (field, invalidate) in cases {
            let mut args = valid_train_args();
            invalidate(&mut args);
            let err = validate_train_args(&args).unwrap_err().to_string();
            assert!(err.contains(field), "{field}: {err}");
        }
    }

    #[test]
    fn training_decreases_loss_and_checkpoint_roundtrips() {
        let config = small_hybrid();
        let device = hermes_llm::default_device().autodiff();
        device.seed(41);
        let mut model = Transformer::new(&config, &device).unwrap();
        let muon_parameter_ids = model.muon_parameter_ids();
        assert!(!muon_parameter_ids.is_empty());
        assert!(muon_parameter_ids.len() < burn::module::list_param_ids(&model).len());
        let mut muon_optimizer = BatchedMuon::new(muon_parameter_ids.clone());
        let mut adamw_optimizer = AdamWConfig::new()
            .with_beta_2(0.95)
            .with_epsilon(1e-8)
            .with_weight_decay(0.0)
            .init();
        let inputs = vec![1_i64, 7, 3, 9, 2, 5, 4, 6, 8, 3];
        let targets = vec![7_i64, 3, 9, 2, 5, 4, 6, 8, 3, 1];
        let batch = || {
            (
                Tensor::<2, Int>::from_data(TensorData::new(inputs.clone(), [2, 5]), &device),
                Tensor::<2, Int>::from_data(TensorData::new(targets.clone(), [2, 5]), &device),
            )
        };

        let mut losses = Vec::new();
        for _ in 0..20 {
            let (input, target) = batch();
            let loss = model.forward_loss(input, target);
            losses.push(scalar_value(loss.clone()).unwrap());
            let mut grads = loss.backward();
            let mut muon_grads =
                GradientsParams::from_params(&mut grads, &model, &muon_parameter_ids);
            let mut adamw_grads = GradientsParams::from_module(&mut grads, &model);
            let norm =
                gradient_norm_and_clip(&model, &mut muon_grads, &mut adamw_grads, 1.0).unwrap();
            assert!(norm.is_finite());
            model = muon_optimizer.step(2e-2, model, muon_grads).unwrap();
            model = adamw_optimizer.step(1e-3.into(), model, adamw_grads);
        }
        assert!(
            losses.last().unwrap() < &losses[0],
            "loss did not decrease: {losses:?}"
        );

        let dir = tempfile::tempdir().unwrap();
        let state = TrainingState {
            step: 20,
            stage: 1,
            epoch: 2,
            samples_in_stage: 640,
            parameter_ids: parameter_ids(&model),
        };
        save_training_checkpoint(
            &model,
            &adamw_optimizer,
            &muon_optimizer,
            &state,
            dir.path(),
        )
        .unwrap();

        let mut resumed = Transformer::new(&config, &device).unwrap();
        let mut resumed_ids = resumed.muon_parameter_ids();
        let mut resumed_muon = BatchedMuon::new(resumed_ids.clone());
        let resumed_adamw = AdamWConfig::new()
            .with_beta_2(0.95)
            .with_epsilon(1e-8)
            .with_weight_decay(0.0)
            .init();
        let (mut resumed_adamw, resumed_state) = load_training_state(
            &mut resumed,
            resumed_adamw,
            &mut resumed_muon,
            dir.path(),
            &device,
        )
        .unwrap();
        assert_eq!(resumed_state.step, state.step);
        assert_eq!(resumed_state.stage, state.stage);
        assert_eq!(resumed_state.epoch, state.epoch);
        assert_eq!(resumed_state.samples_in_stage, state.samples_in_stage);
        let restored_ids = resumed.muon_parameter_ids();
        assert_ne!(resumed_ids, restored_ids);
        assert_eq!(muon_parameter_ids, restored_ids);
        resumed_ids = restored_ids;

        let advance = |mut model: Transformer,
                       muon: &mut BatchedMuon,
                       adamw: &mut AdamWOptimizer,
                       muon_ids: &[ParamId]| {
            let (input, target) = batch();
            let mut grads = model.forward_loss(input, target).backward();
            let muon_grads = GradientsParams::from_params(&mut grads, &model, muon_ids);
            let adamw_grads = GradientsParams::from_module(&mut grads, &model);
            model = muon.step(2e-2, model, muon_grads).unwrap();
            adamw.step(1e-3.into(), model, adamw_grads)
        };
        model = advance(
            model,
            &mut muon_optimizer,
            &mut adamw_optimizer,
            &muon_parameter_ids,
        );
        resumed = advance(resumed, &mut resumed_muon, &mut resumed_adamw, &resumed_ids);

        let valid = model.valid();
        let loaded = resumed.valid();
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(inputs[..5].to_vec(), [1, 5]),
            &device.clone().inner(),
        );
        let expected = valid.forward(input.clone(), 0).into_data();
        let actual = loaded.forward(input, 0).into_data();
        let expected = expected.convert::<f32>().to_vec::<f32>().unwrap();
        let actual = actual.convert::<f32>().to_vec::<f32>().unwrap();
        let max_diff = expected
            .into_iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 1e-6, "checkpoint max diff: {max_diff}");
    }
}
