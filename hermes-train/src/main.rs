use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::tensor::backend::Backend as _;
use burn::tensor::{Device, Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use clap::{Parser, Subcommand, ValueEnum};
use hermes_llm::{Backend, ModelDef, Tokenizer, Transformer, load_safetensors, save_safetensors};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

mod muon;

use muon::BatchedMuon;

type TrainBackend = Autodiff<Backend>;

const MUON_LR_SCALE: f64 = 20.0;

#[derive(Parser)]
#[command(name = "hermes-train", about = "Burn-native Hermes model training")]
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
    /// Atomically replace weights.safetensors every N optimizer steps; 0 disables it.
    #[arg(long, default_value_t = 100)]
    checkpoint_every: usize,
    /// Burn-native checkpoint to fine-tune from.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
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

fn visit_document(
    document: &str,
    tokenizer: &Tokenizer,
    seq_len: usize,
    count: &mut usize,
    visit: &mut impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<bool> {
    let mut tokens = tokenizer.encode(document, false)?;
    tokens.push(tokenizer.eos_token_id());
    for chunk in tokens.windows(seq_len + 1).step_by(seq_len) {
        *count += 1;
        if !visit(chunk.iter().map(|&token| i64::from(token)).collect())? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Visit fixed-length next-token samples without retaining the corpus in RAM.
fn visit_samples_in_order(
    paths: &[PathBuf],
    tokenizer: &Tokenizer,
    seq_len: usize,
    mut visit: impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<usize> {
    ensure!(seq_len > 0, "seq_len must be positive");
    let mut count = 0;
    for path in paths {
        let is_jsonl = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"));
        let mut reader = open_data(path)?;
        if is_jsonl {
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
                let value: serde_json::Value = serde_json::from_str(&line).with_context(|| {
                    format!("invalid JSONL at {}:{line_number}", path.display())
                })?;
                let document = value
                    .get("text")
                    .and_then(|value| value.as_str())
                    .with_context(|| {
                        format!(
                            "JSONL row at {}:{line_number} must contain a string `text` field",
                            path.display()
                        )
                    })?;
                if !visit_document(document, tokenizer, seq_len, &mut count, &mut visit)? {
                    return Ok(count);
                }
            }
        } else {
            let mut document = String::new();
            reader.read_to_string(&mut document)?;
            if !visit_document(&document, tokenizer, seq_len, &mut count, &mut visit)? {
                return Ok(count);
            }
        }
    }
    Ok(count)
}

fn visit_samples(
    paths: &[PathBuf],
    tokenizer: &Tokenizer,
    seq_len: usize,
    shuffle_buffer: usize,
    seed: u64,
    mut visit: impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<usize> {
    if shuffle_buffer == 0 {
        return visit_samples_in_order(paths, tokenizer, seq_len, visit);
    }

    let mut shuffler = ShuffleBuffer::new(shuffle_buffer, seed);
    let mut keep_going = true;
    let count = visit_samples_in_order(paths, tokenizer, seq_len, |sample| {
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

fn count_samples(paths: &[PathBuf], tokenizer: &Tokenizer, seq_len: usize) -> Result<usize> {
    visit_samples_in_order(paths, tokenizer, seq_len, |_| Ok(true))
}

fn make_batch(
    samples: &[Vec<i64>],
    seq_len: usize,
    device: &Device<TrainBackend>,
) -> (Tensor<TrainBackend, 2, Int>, Tensor<TrainBackend, 2, Int>) {
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
    sum: Option<Tensor<Backend, 1>>,
}

impl ModuleVisitor<TrainBackend> for SquaredGradientNorm<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TrainBackend, D>>) {
        let Some(grad) = self.grads.get::<Backend, D>(param.id) else {
            return;
        };
        let squared = grad.square().sum();
        self.sum = Some(match self.sum.take() {
            Some(sum) => sum + squared,
            None => squared,
        });
    }
}

fn squared_gradient_norm(
    model: &Transformer<TrainBackend>,
    grads: &GradientsParams,
) -> Option<Tensor<Backend, 1>> {
    let mut visitor = SquaredGradientNorm { grads, sum: None };
    model.visit(&mut visitor);
    visitor.sum
}

struct GradientScaler<'a> {
    grads: &'a mut GradientsParams,
    scale: f32,
}

impl ModuleVisitor<TrainBackend> for GradientScaler<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TrainBackend, D>>) {
        let Some(grad) = self.grads.remove::<Backend, D>(param.id) else {
            return;
        };
        self.grads
            .register::<Backend, D>(param.id, grad.mul_scalar(self.scale));
    }
}

fn scale_gradients(model: &Transformer<TrainBackend>, grads: &mut GradientsParams, scale: f32) {
    model.visit(&mut GradientScaler { grads, scale });
}

fn gradient_norm_and_clip(
    model: &Transformer<TrainBackend>,
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
    let norm = sum.sqrt().into_data().to_vec::<f32>()?[0];
    if max_norm > 0.0 && norm > max_norm {
        let scale = max_norm / norm;
        scale_gradients(model, muon_grads, scale);
        scale_gradients(model, adamw_grads, scale);
    }
    Ok(norm)
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

fn save_checkpoint<B: hermes_llm::MambaBackend>(
    model: &Transformer<B>,
    output: &Path,
) -> Result<()> {
    let temporary = output.join("weights.safetensors.tmp");
    save_safetensors(model, &temporary)?;
    fs::rename(temporary, output.join("weights.safetensors"))?;
    Ok(())
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
    let sample_count = match args.max_steps {
        Some(_) => None,
        None => Some(count_samples(&args.data, &tokenizer, args.seq_len)?),
    };
    let total_steps = match (args.max_steps, sample_count) {
        (Some(steps), _) => steps,
        (None, Some(samples)) => {
            let microbatches = samples / args.batch_size;
            (microbatches / args.grad_accum).saturating_mul(args.epochs)
        }
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
    let mut metrics = BufWriter::new(File::create(args.output.join("metrics.jsonl"))?);

    let device = hermes_llm::default_device();
    Backend::seed(&device, args.seed);
    let mut initial_model = Transformer::<TrainBackend>::new(&config, &device)?;
    if let Some(path) = &args.checkpoint {
        load_safetensors(&mut initial_model, path)?;
    }
    let muon_parameter_ids = initial_model.muon_parameter_ids();
    ensure!(
        !muon_parameter_ids.is_empty(),
        "model has no hidden matrix parameters for Muon"
    );
    let mut muon_optimizer = BatchedMuon::new(muon_parameter_ids.clone());
    let mut adamw_optimizer = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .with_weight_decay(args.weight_decay)
        .init();
    let mut muon_accumulator = GradientsAccumulator::new();
    let mut adamw_accumulator = GradientsAccumulator::new();
    // Optimizer::step consumes the module; Option lets the streaming callback
    // move it out and replace it without cloning model parameters.
    let mut model = Some(initial_model);
    let mut step = 0;
    let mut micro_step = 0;
    let mut loss_sum = 0.0f32;

    println!(
        "model={} params={} muon_matrices={} device={device:?} samples={} steps={total_steps} shuffle_buffer={}",
        config.name,
        model.as_ref().unwrap().num_parameters(),
        muon_parameter_ids.len(),
        sample_count.map_or_else(|| "streaming".to_owned(), |count| count.to_string()),
        args.shuffle_buffer,
    );

    'epochs: for epoch in 0..args.epochs {
        let mut batch = Vec::with_capacity(args.batch_size);
        visit_samples(
            &args.data,
            &tokenizer,
            args.seq_len,
            args.shuffle_buffer,
            args.seed.wrapping_add(epoch as u64),
            |sample| {
                batch.push(sample);
                if batch.len() < args.batch_size {
                    return Ok(true);
                }

                let (inputs, targets) = make_batch(&batch, args.seq_len, &device);
                batch.clear();
                let current = model.as_ref().unwrap();
                let loss = current.forward_loss(inputs, targets);
                let loss_value = loss.clone().into_data().to_vec::<f32>()?[0];
                if !loss_value.is_finite() {
                    bail!(
                        "non-finite loss before optimizer step {}: {loss_value}",
                        step + 1
                    );
                }
                let mut grads = loss.div_scalar(args.grad_accum as f64).backward();
                let muon_grads =
                    GradientsParams::from_params(&mut grads, current, &muon_parameter_ids);
                let adamw_grads = GradientsParams::from_module(&mut grads, current);
                muon_accumulator.accumulate(current, muon_grads);
                adamw_accumulator.accumulate(current, adamw_grads);
                micro_step += 1;
                loss_sum += loss_value;

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
                    model = Some(adamw_optimizer.step(lr, current, adamw_grads));
                    step += 1;
                    let loss = loss_sum / args.grad_accum as f32;
                    println!(
                        "epoch={} step={step}/{total_steps} loss={:.6} lr={lr:.3e} grad_norm={grad_norm:.3}",
                        epoch + 1,
                        loss
                    );
                    serde_json::to_writer(
                        &mut metrics,
                        &serde_json::json!({
                            "step": step,
                            "epoch": epoch + 1,
                            "loss": loss,
                            "lr": lr,
                            "muon_lr": muon_lr,
                            "grad_norm": grad_norm,
                            "tokens": step * args.batch_size * args.grad_accum * args.seq_len,
                        }),
                    )?;
                    metrics.write_all(b"\n")?;
                    metrics.flush()?;
                    if args.checkpoint_every > 0 && step % args.checkpoint_every == 0 {
                        save_checkpoint(&model.as_ref().unwrap().clone().valid(), &args.output)?;
                        println!("checkpointed {}", args.output.display());
                    }
                    micro_step = 0;
                    loss_sum = 0.0;
                }
                Ok(step < total_steps)
            },
        )?;
        if step >= total_steps {
            break 'epochs;
        }
        // An optimizer step always consists of exactly grad_accum microbatches.
        if micro_step != 0 {
            muon_accumulator = GradientsAccumulator::new();
            adamw_accumulator = GradientsAccumulator::new();
            micro_step = 0;
            loss_sum = 0.0;
        }
    }
    ensure!(
        step == total_steps,
        "requested {total_steps} optimizer steps, but the data produced only {step} complete steps"
    );

    let inference_model = model.unwrap().valid();
    save_checkpoint(&inference_model, &args.output)?;
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
    fn burn_training_decreases_loss_and_checkpoint_roundtrips() {
        let config = small_hybrid();
        let device = hermes_llm::default_device();
        Backend::seed(&device, 41);
        let mut model = Transformer::<TrainBackend>::new(&config, &device).unwrap();
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
                Tensor::<TrainBackend, 2, Int>::from_data(
                    TensorData::new(inputs.clone(), [2, 5]),
                    &device,
                ),
                Tensor::<TrainBackend, 2, Int>::from_data(
                    TensorData::new(targets.clone(), [2, 5]),
                    &device,
                ),
            )
        };

        let mut losses = Vec::new();
        for _ in 0..20 {
            let (input, target) = batch();
            let loss = model.forward_loss(input, target);
            losses.push(loss.clone().into_data().to_vec::<f32>().unwrap()[0]);
            let mut grads = loss.backward();
            let mut muon_grads =
                GradientsParams::from_params(&mut grads, &model, &muon_parameter_ids);
            let mut adamw_grads = GradientsParams::from_module(&mut grads, &model);
            let norm =
                gradient_norm_and_clip(&model, &mut muon_grads, &mut adamw_grads, 1.0).unwrap();
            assert!(norm.is_finite());
            model = muon_optimizer.step(2e-2, model, muon_grads).unwrap();
            model = adamw_optimizer.step(1e-3, model, adamw_grads);
        }
        assert!(
            losses.last().unwrap() < &losses[0],
            "loss did not decrease: {losses:?}"
        );

        let valid = model.valid();
        let input = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(inputs[..5].to_vec(), [1, 5]),
            &device,
        );
        let expected = valid.forward(input.clone(), 0).into_data();
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("weights.safetensors");
        save_safetensors(&valid, &checkpoint).unwrap();

        let mut loaded = Transformer::<Backend>::new(&config, &device).unwrap();
        load_safetensors(&mut loaded, &checkpoint).unwrap();
        let actual = loaded.forward(input, 0).into_data();
        let expected = expected.to_vec::<f32>().unwrap();
        let actual = actual.to_vec::<f32>().unwrap();
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
}
