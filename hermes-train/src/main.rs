use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::Tensor;
use burn_nn::loss::CrossEntropyLossConfig;
use burn_optim::{AdamWConfig, GradientsAccumulator, GradientsParams};
use clap::{Parser, Subcommand, ValueEnum};
use hermes_llm::{ModelDef, Tokenizer, Transformer, load_safetensors};
use serde::Serialize;

mod checkpoint;
mod curriculum;
mod data;
mod muon;
mod trainer;

use checkpoint::{
    AdamWOptimizer, TRAINING_STATE_VERSION, TrainingState, load_training_state, parameter_ids,
    save_training_checkpoint,
};
use curriculum::{ObjectiveConfig, ResolvedCurriculum, load_curriculum};
use data::{
    BatchStats, SampleStreamConfig, TrainingBatch, count_samples, make_batch, visit_samples,
};
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

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
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
    /// Versioned JSON curriculum with explicit objectives and stage geometry.
    #[arg(long)]
    curriculum: PathBuf,
    #[arg(short = 'o', long, default_value = "checkpoint")]
    output: PathBuf,
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
    /// Save a resumable checkpoint every N optimizer steps; 0 disables it.
    #[arg(long, default_value_t = 100)]
    checkpoint_every: usize,
    /// Record pre-clip gradient L2 norm for every layer every N optimizer
    /// steps; 0 disables this opt-in visualization/debug metric.
    #[arg(long, default_value_t = 0)]
    layer_metrics_every: usize,
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

fn squared_layer_gradient_norm(
    model: &Transformer,
    layer: usize,
    grads: &GradientsParams,
) -> Result<Option<Tensor<1>>> {
    let mut visitor = SquaredGradientNorm { grads, sum: None };
    model.visit_layer(layer, &mut visitor)?;
    Ok(visitor.sum)
}

fn layer_gradient_norms(
    model: &Transformer,
    muon_grads: &GradientsParams,
    adamw_grads: &GradientsParams,
) -> Result<Vec<f32>> {
    let mut norms = Vec::with_capacity(model.config().num_layers);
    for layer in 0..model.config().num_layers {
        let sum = match (
            squared_layer_gradient_norm(model, layer, muon_grads)?,
            squared_layer_gradient_norm(model, layer, adamw_grads)?,
        ) {
            (Some(muon), Some(adamw)) => muon + adamw,
            (Some(sum), None) | (None, Some(sum)) => sum,
            (None, None) => bail!("layer {} has no gradients", layer + 1),
        };
        norms.push(sum.sqrt());
    }
    let values = Tensor::cat(norms, 0)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "per-layer gradient norms contain a non-finite value"
    );
    Ok(values)
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
    Ok(())
}

fn resolve_curriculum(args: &TrainArgs) -> Result<ResolvedCurriculum> {
    load_curriculum(&args.curriculum)
}

fn validate_model_curriculum(config: &ModelDef, curriculum: &ResolvedCurriculum) -> Result<()> {
    for stage in &curriculum.stages {
        ensure!(
            stage.sequence_length <= config.max_seq_len,
            "curriculum stage `{}` sequence_length {} exceeds model max_seq_len {}",
            stage.name,
            stage.sequence_length,
            config.max_seq_len
        );
        if matches!(
            stage.objective,
            ObjectiveConfig::ContrastiveRetrieval { .. }
        ) {
            let layer = stage
                .objective
                .retrieval_layer()
                .unwrap_or(config.num_layers);
            ensure!(
                layer <= config.num_layers,
                "curriculum stage `{}` requests retrieval layer {layer}, model has {} layers",
                stage.name,
                config.num_layers
            );
            ensure!(
                (0..layer).any(|index| {
                    let block = config.block_for_layer(index);
                    !block.is_ssm() && block.attention.window_size.is_none()
                }),
                "curriculum stage `{}` retrieval layer {layer} has no full-attention layer at or before it",
                stage.name
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct StagePlan {
    samples: Option<usize>,
    steps: usize,
}

fn plan_training(
    curriculum: &ResolvedCurriculum,
    tokenizer: &Tokenizer,
    token_cache_root: &Path,
) -> Result<(Vec<StagePlan>, usize)> {
    let mut total_steps = 0usize;
    let mut plan = Vec::with_capacity(curriculum.stages.len());
    for (stage_index, stage) in curriculum.stages.iter().enumerate() {
        let (samples, steps) = match stage.steps {
            Some(steps) => (None, steps),
            None => {
                let samples = count_samples(
                    &stage.data,
                    &stage.objective,
                    tokenizer,
                    stage.sequence_length,
                    Some(&token_cache_root.join(format!("stage-{stage_index:03}.tokens"))),
                )?;
                let steps_per_epoch =
                    (samples / stage.batch_size).div_euclid(stage.gradient_accumulation);
                let optimizer_steps =
                    steps_per_epoch.checked_mul(stage.epochs).with_context(|| {
                        format!(
                            "curriculum stage `{}` optimizer-step count overflows usize",
                            stage.name
                        )
                    })?;
                (Some(samples), optimizer_steps)
            }
        };
        ensure!(
            steps > 0,
            "curriculum stage `{}` produces zero complete optimizer steps",
            stage.name
        );
        total_steps = total_steps
            .checked_add(steps)
            .ok_or_else(|| anyhow::anyhow!("curriculum optimizer-step count overflows usize"))?;
        plan.push(StagePlan { samples, steps });
    }
    Ok((plan, total_steps))
}

fn file_fingerprint(path: &Path) -> Result<String> {
    // Stable FNV-1a is sufficient for accidental tokenizer/config drift and
    // avoids making resume depend on a platform-specific standard hasher.
    let bytes = fs::read(path)
        .map_err(anyhow::Error::from)
        .map_err(|error| error.context(format!("failed to fingerprint {}", path.display())))?;
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in &bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("fnv1a64:{hash:016x}:{}", bytes.len()))
}

fn stable_cache_id(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[derive(Serialize)]
struct RunSignature<'a> {
    version: u32,
    curriculum: &'a ResolvedCurriculum,
    model: &'a ModelDef,
    tokenizer: String,
    seed: u64,
    learning_rate: f64,
    weight_decay: f32,
    gradient_clip: f32,
    warmup_steps: usize,
    schedule: Schedule,
    muon_learning_rate_scale: f64,
}

fn run_signature(
    args: &TrainArgs,
    curriculum: &ResolvedCurriculum,
    config: &ModelDef,
) -> Result<String> {
    Ok(serde_json::to_string(&RunSignature {
        version: TRAINING_STATE_VERSION,
        curriculum,
        model: config,
        tokenizer: file_fingerprint(&args.tokenizer)?,
        seed: args.seed,
        learning_rate: args.lr,
        weight_decay: args.weight_decay,
        gradient_clip: args.grad_clip,
        warmup_steps: args.warmup_steps,
        schedule: args.schedule,
        muon_learning_rate_scale: MUON_LR_SCALE,
    })?)
}

fn add_batch_stats(total: &mut BatchStats, batch: BatchStats) -> Result<()> {
    total.examples = total
        .examples
        .checked_add(batch.examples)
        .context("optimizer-step example count overflows usize")?;
    total.compute_tokens = total
        .compute_tokens
        .checked_add(batch.compute_tokens)
        .context("optimizer-step compute-token count overflows usize")?;
    total.supervised_tokens = total
        .supervised_tokens
        .checked_add(batch.supervised_tokens)
        .context("optimizer-step supervised-token count overflows usize")?;
    total.truncated_tokens = total
        .truncated_tokens
        .checked_add(batch.truncated_tokens)
        .context("optimizer-step truncated-token count overflows usize")?;
    total.retrieval_candidates = total
        .retrieval_candidates
        .checked_add(batch.retrieval_candidates)
        .context("optimizer-step retrieval-candidate count overflows usize")?;
    Ok(())
}

fn objective_loss(
    model: &Transformer,
    batch: TrainingBatch,
    objective: &ObjectiveConfig,
) -> Result<(Tensor<1>, Option<Tensor<1>>, BatchStats, Option<Tensor<1>>)> {
    let stats = batch.stats();
    let mut retrieval_correct = None;
    let (loss, router_loss) = match batch {
        TrainingBatch::Language(batch) => {
            let data::LanguageBatch {
                input_ids,
                targets,
                loss_positions,
                ..
            } = *batch;
            match objective {
                ObjectiveConfig::CausalLm => {
                    ensure!(
                        loss_positions.is_none(),
                        "causal_lm batch unexpectedly contains a target mask"
                    );
                    model.forward_loss_with_router(input_ids, targets)
                }
                ObjectiveConfig::Summarization { .. }
                | ObjectiveConfig::RetrievalPlanning { .. } => {
                    let positions = loss_positions
                        .ok_or_else(|| anyhow::anyhow!("structured batch has no target mask"))?;
                    model.forward_masked_loss_with_router(input_ids, targets, positions)
                }
                ObjectiveConfig::ContrastiveRetrieval { .. } => {
                    bail!("contrastive_retrieval stage produced a language batch")
                }
            }
        }
        TrainingBatch::Retrieval(batch) => {
            ensure!(
                matches!(objective, ObjectiveConfig::ContrastiveRetrieval { .. }),
                "non-retrieval stage produced a retrieval batch"
            );
            let data::RetrievalBatch {
                query_ids,
                query_end_positions,
                document_ids,
                document_end_positions,
                labels,
                ..
            } = *batch;
            let layer = objective.retrieval_layer();
            let temperature = objective
                .temperature()
                .expect("retrieval objective has a temperature");
            let (queries, query_router_loss) =
                model.forward_embeddings_with_router(query_ids, query_end_positions, layer);
            let (documents, document_router_loss) =
                model.forward_embeddings_with_router(document_ids, document_end_positions, layer);
            let logits = queries
                .matmul(documents.transpose())
                .div_scalar(temperature);
            retrieval_correct = Some(
                logits
                    .clone()
                    .argmax(1)
                    .squeeze_dim::<1>(1)
                    .equal(labels.clone())
                    .float()
                    .sum()
                    .detach(),
            );
            let loss = CrossEntropyLossConfig::new()
                .init(&labels.device())
                .forward(logits, labels);
            let router_loss = match (query_router_loss, document_router_loss) {
                (Some(query), Some(document)) => Some(query + document),
                (Some(loss), None) | (None, Some(loss)) => Some(loss),
                (None, None) => None,
            };
            (loss, router_loss)
        }
    };
    Ok((loss, router_loss, stats, retrieval_correct))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    match Cli::parse().command {
        Command::Train(args) => trainer::train(args),
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
            curriculum: "curriculum.json".into(),
            output: "checkpoint".into(),
            lr: 3e-4,
            weight_decay: 0.1,
            grad_clip: 1.0,
            warmup_steps: 10,
            schedule: Schedule::Wsd,
            checkpoint_every: 0,
            layer_metrics_every: 0,
            checkpoint: None,
            resume: false,
            seed: 0,
        }
    }

    #[test]
    fn invalid_numeric_training_arguments_fail_before_loading_files() {
        type Invalidate = fn(&mut TrainArgs);
        let cases: [(&str, Invalidate); 3] = [
            ("lr", |args| args.lr = f64::NAN),
            ("weight_decay", |args| args.weight_decay = -0.1),
            ("grad_clip", |args| args.grad_clip = f32::INFINITY),
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
            if losses.len() == 1 {
                let layer_norms = layer_gradient_norms(&model, &muon_grads, &adamw_grads).unwrap();
                assert_eq!(layer_norms.len(), config.num_layers);
                assert!(layer_norms.into_iter().all(f32::is_finite));
            }
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
            version: TRAINING_STATE_VERSION,
            step: 20,
            stage: 1,
            epoch: 2,
            samples_in_stage: 640,
            steps_in_stage: 10,
            tokens_seen: 12_800,
            curriculum_signature: Some("test-curriculum".to_owned()),
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
        assert_eq!(resumed_state.steps_in_stage, state.steps_in_stage);
        assert_eq!(resumed_state.tokens_seen, state.tokens_seen);
        assert_eq!(
            resumed_state.curriculum_signature,
            state.curriculum_signature
        );
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
