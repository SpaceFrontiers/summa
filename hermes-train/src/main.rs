use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::{Device, Int, Tensor, TensorData};
use burn_optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use clap::{Parser, Subcommand, ValueEnum};
use hermes_llm::{ModelDef, Tokenizer, Transformer, load_safetensors, save_safetensors};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

mod muon;

use muon::BatchedMuon;

type AdamWOptimizer = ModuleOptimizer;

const MUON_LR_SCALE: f64 = 20.0;
const TOKENIZE_BATCH: usize = 1_000;

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

#[derive(Clone, Deserialize, Serialize)]
struct TrainingState {
    step: usize,
    stage: usize,
    epoch: usize,
    samples_in_stage: usize,
    parameter_ids: Vec<u64>,
}

fn load_config(path: &Path) -> Result<ModelDef> {
    if path.extension().is_some_and(|ext| ext == "mal") {
        return hermes_llm::parse_mal_file(path);
    }
    ModelDef::from_json(path)
}

fn open_data(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open training data {}", path.display()))?;
    if path.extension().is_some_and(|ext| ext == "zst") {
        let decoder = zstd::stream::read::Decoder::new(file)
            .with_context(|| format!("failed to open zstd stream {}", path.display()))?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

struct ShuffleBuffer {
    samples: Vec<Vec<i64>>,
    rng: StdRng,
    capacity: usize,
}

impl ShuffleBuffer {
    fn new(capacity: usize, seed: u64) -> Self {
        assert!(capacity > 0);
        Self {
            samples: Vec::with_capacity(capacity),
            rng: StdRng::seed_from_u64(seed),
            capacity,
        }
    }

    fn push(&mut self, sample: Vec<i64>) -> Option<Vec<i64>> {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            return None;
        }
        let index = self.rng.random_range(0..self.samples.len());
        Some(std::mem::replace(&mut self.samples[index], sample))
    }

    fn finish(mut self) -> Vec<Vec<i64>> {
        self.samples.shuffle(&mut self.rng);
        self.samples
    }
}

struct SamplePacker {
    pending: Vec<i64>,
    consumed: usize,
    seq_len: usize,
}

impl SamplePacker {
    fn new(seq_len: usize) -> Self {
        Self {
            pending: Vec::new(),
            consumed: 0,
            seq_len,
        }
    }

    fn push(
        &mut self,
        tokens: impl IntoIterator<Item = i64>,
        count: &mut usize,
        visit: &mut impl FnMut(Vec<i64>) -> Result<bool>,
    ) -> Result<bool> {
        if self.consumed > 0 {
            self.pending.drain(..self.consumed);
            self.consumed = 0;
        }
        self.pending.extend(tokens);
        while self.pending.len() - self.consumed > self.seq_len {
            let end = self.consumed + self.seq_len + 1;
            let sample = self.pending[self.consumed..end].to_vec();
            self.consumed += self.seq_len;
            *count += 1;
            if !visit(sample)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn push_documents(
    documents: &mut Vec<String>,
    tokenizer: &Tokenizer,
    packer: &mut SamplePacker,
    count: &mut usize,
    visit: &mut impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<bool> {
    if documents.is_empty() {
        return Ok(true);
    }
    let encodings = tokenizer.encode_batch(std::mem::take(documents), false)?;
    for tokens in encodings {
        let tokens = tokens
            .into_iter()
            .map(i64::from)
            .chain(std::iter::once(i64::from(tokenizer.eos_token_id())));
        if !packer.push(tokens, count, visit)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Pack the EOS-joined token stream into fixed-length next-token samples.
fn visit_samples_in_order(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    mut visit: impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<usize> {
    ensure!(seq_len > 0, "seq_len must be positive");
    let mut count = 0;
    let mut packer = SamplePacker::new(seq_len);
    let is_jsonl = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"));
    let mut reader = open_data(path)?;
    if is_jsonl {
        let mut documents = Vec::with_capacity(TOKENIZE_BATCH);
        let mut line = String::new();
        let mut line_number = 0;
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            line_number += 1;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)
                .with_context(|| format!("invalid JSONL at {}:{line_number}", path.display()))?;
            let document = value
                .get("text")
                .and_then(|value| value.as_str())
                .with_context(|| {
                    format!(
                        "JSONL row at {}:{line_number} must contain a string `text` field",
                        path.display()
                    )
                })?;
            documents.push(document.to_owned());
            if documents.len() == TOKENIZE_BATCH
                && !push_documents(
                    &mut documents,
                    tokenizer,
                    &mut packer,
                    &mut count,
                    &mut visit,
                )?
            {
                return Ok(count);
            }
        }
        if !push_documents(
            &mut documents,
            tokenizer,
            &mut packer,
            &mut count,
            &mut visit,
        )? {
            return Ok(count);
        }
    } else {
        let mut document = String::new();
        reader.read_to_string(&mut document)?;
        if !push_documents(
            &mut vec![document],
            tokenizer,
            &mut packer,
            &mut count,
            &mut visit,
        )? {
            return Ok(count);
        }
    }
    Ok(count)
}

fn visit_samples(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    shuffle_buffer: usize,
    seed: u64,
    mut visit: impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<usize> {
    if shuffle_buffer == 0 {
        return visit_samples_in_order(path, tokenizer, seq_len, visit);
    }

    let mut shuffler = ShuffleBuffer::new(shuffle_buffer, seed);
    let mut keep_going = true;
    let count = visit_samples_in_order(path, tokenizer, seq_len, |sample| {
        if let Some(sample) = shuffler.push(sample) {
            keep_going = visit(sample)?;
        }
        Ok(keep_going)
    })?;

    if keep_going {
        for sample in shuffler.finish() {
            if !visit(sample)? {
                break;
            }
        }
    }
    Ok(count)
}

fn count_samples(path: &Path, tokenizer: &Tokenizer, seq_len: usize) -> Result<usize> {
    visit_samples_in_order(path, tokenizer, seq_len, |_| Ok(true))
}

fn make_batch(
    samples: &[Vec<i64>],
    seq_len: usize,
    device: &Device,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let mut inputs = Vec::with_capacity(samples.len() * seq_len);
    let mut targets = Vec::with_capacity(samples.len() * seq_len);
    for sample in samples {
        inputs.extend_from_slice(&sample[..seq_len]);
        targets.extend_from_slice(&sample[1..]);
    }
    (
        Tensor::from_data(TensorData::new(inputs, [samples.len(), seq_len]), device),
        Tensor::from_data(TensorData::new(targets, [samples.len(), seq_len]), device),
    )
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

fn parameter_ids(model: &Transformer) -> Vec<u64> {
    burn::module::list_param_ids(model)
        .into_iter()
        .map(|id| id.val())
        .collect()
}

struct ParameterIdMapper<'a> {
    ids: std::slice::Iter<'a, u64>,
}

impl ModuleMapper for ParameterIdMapper<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (_, tensor, mapper) = param.consume();
        let id = self
            .ids
            .next()
            .copied()
            .expect("checkpoint contains too few parameter IDs");
        Param::from_mapped_value(ParamId::from(id), tensor, mapper)
    }
}

fn restore_parameter_ids(model: &mut Transformer, ids: &[u64]) -> Result<()> {
    ensure!(
        ids.len() == burn::module::list_param_ids(model).len(),
        "checkpoint has {} parameter IDs, model has {}",
        ids.len(),
        burn::module::list_param_ids(model).len()
    );
    let mut mapper = ParameterIdMapper { ids: ids.iter() };
    *model = model.clone().map(&mut mapper);
    ensure!(
        mapper.ids.next().is_none(),
        "checkpoint contains too many parameter IDs"
    );
    Ok(())
}

fn save_training_checkpoint(
    model: &Transformer,
    adamw: &AdamWOptimizer,
    muon: &BatchedMuon,
    state: &TrainingState,
    output: &Path,
) -> Result<()> {
    let marker = output.join(".checkpoint-in-progress");
    let weights_temporary = output.join("weights.safetensors.tmp");
    let adamw_temporary = output.join("adamw-state.bpk.tmp");
    let muon_temporary = output.join("muon-state.bpk.tmp");
    let state_temporary = output.join("training-state.json.tmp");

    fs::write(&marker, state.step.to_string())?;
    save_safetensors(&model.clone().valid(), &weights_temporary)?;
    adamw
        .save(&adamw_temporary)
        .context("failed to save AdamW state")?;
    muon.save(&muon_temporary)?;
    fs::write(&state_temporary, serde_json::to_vec_pretty(&state)?)?;
    fs::rename(weights_temporary, output.join("weights.safetensors"))?;
    fs::rename(adamw_temporary, output.join("adamw-state.bpk"))?;
    fs::rename(muon_temporary, output.join("muon-state.bpk"))?;
    fs::rename(state_temporary, output.join("training-state.json"))?;
    fs::remove_file(marker)?;
    Ok(())
}

fn load_training_state(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
) -> Result<(AdamWOptimizer, TrainingState)> {
    ensure!(
        !output.join(".checkpoint-in-progress").exists(),
        "checkpoint was interrupted while being saved"
    );
    let state: TrainingState =
        serde_json::from_slice(&fs::read(output.join("training-state.json"))?)?;
    restore_parameter_ids(model, &state.parameter_ids)?;
    load_safetensors(model, output.join("weights.safetensors"))?;
    muon.set_parameter_ids(model.muon_parameter_ids());
    muon.load(output.join("muon-state.bpk"), &device.clone().inner())?;
    let adamw = adamw
        .load(output.join("adamw-state.bpk"))
        .context("failed to load AdamW state")?;
    Ok((adamw, state))
}

fn train(args: TrainArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch_size must be positive");
    ensure!(args.grad_accum > 0, "grad_accum must be positive");
    ensure!(args.epochs > 0, "epochs must be positive");

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
            visit_samples(
                path,
                &tokenizer,
                args.seq_len,
                args.shuffle_buffer,
                shuffle_seed,
                |sample| {
                    samples_in_stage += 1;
                    training_state.samples_in_stage = samples_in_stage;
                    if samples_in_stage <= samples_to_skip {
                        return Ok(true);
                    }
                    batch.push(sample);
                    if batch.len() < args.batch_size {
                        return Ok(true);
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
                        let mut muon_grads = muon_accumulator.grads();
                        let mut adamw_grads = adamw_accumulator.grads();
                        let grad_norm = gradient_norm_and_clip(
                            current,
                            &mut muon_grads,
                            &mut adamw_grads,
                            args.grad_clip,
                        )?;
                        let current = model.take().unwrap();
                        let current = muon_optimizer.step(muon_lr, current, muon_grads)?;
                        model = Some(adamw_optimizer.step(lr.into(), current, adamw_grads));
                        step += 1;
                        training_state.step = step;
                        let loss = loss_sum
                            .take()
                            .expect("an optimizer step must contain a loss")
                            .div_scalar(args.grad_accum as f32);
                        let loss = scalar_value(loss)?;
                        if !loss.is_finite() {
                            bail!("non-finite loss at optimizer step {step}: {loss}");
                        }
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
                    }
                    Ok(step < total_steps)
                },
            )?;
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
    use hermes_llm::get_builtin_model;
    use std::io::Cursor;

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

    #[test]
    fn zstd_data_reader_streams_decompressed_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl.zst");
        let source = b"{\"text\":\"one\"}\n{\"text\":\"two\"}\n";
        let compressed = zstd::stream::encode_all(Cursor::new(source), 1).unwrap();
        fs::write(&path, compressed).unwrap();

        let mut reader = open_data(&path).unwrap();
        let mut decoded = String::new();
        reader.read_to_string(&mut decoded).unwrap();
        assert_eq!(decoded.as_bytes(), source);
    }

    #[test]
    fn streaming_shuffle_is_bounded_and_deterministic() {
        let shuffle = |seed| {
            let mut buffer = ShuffleBuffer::new(4, seed);
            let mut output = Vec::new();
            for value in 0..32_i64 {
                if let Some(sample) = buffer.push(vec![value]) {
                    output.push(sample[0]);
                }
                assert!(buffer.samples.len() <= 4);
            }
            output.extend(buffer.finish().into_iter().map(|sample| sample[0]));
            output
        };

        let first = shuffle(7);
        assert_eq!(first, shuffle(7));
        assert_ne!(first, (0..32_i64).collect::<Vec<_>>());
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32_i64).collect::<Vec<_>>());
    }

    #[test]
    fn sample_packer_joins_documents_without_dropping_tokens() {
        let mut packer = SamplePacker::new(3);
        let mut samples = Vec::new();
        let mut count = 0;
        let mut collect = |sample| {
            samples.push(sample);
            Ok(true)
        };

        for document in [vec![1, 2, 0], vec![3, 4, 0], vec![5, 6, 0]] {
            assert!(packer.push(document, &mut count, &mut collect).unwrap());
        }

        assert_eq!(count, 2);
        assert_eq!(samples, [vec![1, 2, 0, 3], vec![3, 4, 0, 5]]);
    }
}
