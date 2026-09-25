// clippy 1.98 added `chunks_exact_to_as_chunks`, which fires on ~40 call sites
// here. Migrating them is a real (if mechanical) improvement — a const-generic
// chunk width lets LLVM drop the per-chunk length check — but most of the sites
// are inside BMP / ScaNN / fast-field wire-format parsers, and rewriting those
// belongs in its own reviewed change rather than riding along with a toolchain
// bump. Tracked as follow-up; see docs/algebraic-float-reductions.md.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::tensor::{Device, Tensor};
use burn_nn::loss::CrossEntropyLossConfig;
use burn_optim::{AdamWConfig, GradientsAccumulator, GradientsParams};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use sha2::{Digest, Sha256};
use summa_llm::{
    BlockDef, ModelDef, Tokenizer, Transformer, save_safetensors, upgrade_safetensors_to_memory,
};
use summa_train::artifact_io::hash_regular_file;
use summa_train::builtin_sleep_runtime::{
    BuiltinPeriodicSleepBoundaryDriver, BuiltinSleepPhaseContextFactory,
};
use summa_train::corpus::{
    CurriculumCompositionConfig, SearchApiPostgresCorpusRecipe, SummaCorpusTokenizer,
};
use summa_train::device_sampler::{
    DeviceSamplerDrain, NvidiaSmiSampler, NvidiaSmiSamplerConfig, validate_physical_device_selector,
};
use summa_train::metrics::{
    MemoryTier as MetricMemoryTier, MetricContext, MetricEvent, MetricPhase, MetricPhaseKind,
    MetricWriter, OptimizationMetrics, PhaseBoundary, PhaseTimingMetrics,
    QuantizationFormat as MetricQuantizationFormat, QuantizationMetrics, QuantizationStage,
    ThroughputMetrics, TierUpdateMetrics, TierUpdateOutcome, validate_metric_prefix_for_run,
    validate_metric_snapshot_for_run,
};
use summa_train::native_sleep::{
    NativeSleepCheckpoint, NativeSleepContextRegistry, NativeSleepPhaseExecutor,
    drain_periodic_sleep_before_wake_step,
};
use summa_train::posttrain::forward_kl_distillation_tensor;
use summa_train::promotion::NativePromotionExecutor;
#[cfg(test)]
use summa_train::qat_candidate::open_qat_candidate;
use summa_train::qat_candidate::{
    open_qat_candidate_addressed, publish_qat_candidate,
    publish_qat_candidate_from_authenticated_source,
};
use summa_train::quantization::{
    BONSAI_GROUP_SIZE, QuantizationRecipe, UltraQuantFormat, WorkflowQuantizationPlan,
    WorkflowQuantizationTraining, export_safetensors_archive, fake_quantized_transformer,
};
use summa_train::runtime::{
    ALL_PHASE_KINDS, ExecutorRegistry, ImmutableArtifact, ImmutableModelCheckpoint, RuntimeStatus,
    WorkflowRunState, run_until_yield_or_complete,
    workflow_signature as runtime_workflow_signature,
};
use summa_train::task::{TaskAdapter, TaskConfig};
use summa_train::tensor_sleep::TensorTransactionStore;
use summa_train::tier_optimizer::{
    DurableTierOptimizerPublisher, TierOptimizerBank, WakeOnlyTierUpdate,
};
use summa_train::worker::{
    AtomicRuntimeCheckpoint, ExternalPhaseExecutor, PHASE_WORKER_PROTOCOL_VERSION,
};
use summa_train::workflow::{
    MemoryUpdateMode, PhaseKind, ResolvedWorkflow, load_workflow as load_workflow_v2,
    validate_sleep_schedule_for_model, validate_workflow_for_model,
};

mod checkpoint;
mod data;
mod eval;
mod generate_eval;
mod muon;
mod retrieval_pool;
mod trainer;
mod wake;

use checkpoint::{
    AdamWOptimizer, ArtifactRef, CheckpointPublication, OptimizerStateRef,
    QuantizationTrainingState, RngStreamState, TRAINING_STATE_VERSION, TrainingMemoryUpdateState,
    TrainingState, load_training_state, parameter_ids, save_training_checkpoint_with_evidence,
    verify_checkpoint_generation, verify_checkpoint_root,
};
#[cfg(test)]
use data::TrainingSample;
use data::{
    BatchStats, OversizedRecordPolicy, PhaseDataBinding, SampleStreamConfig, TrainingBatch,
    count_samples, indexed_causal_sample_count, make_batch, visit_samples,
};
use muon::BatchedMuon;
use wake::{ResolvedWakePlan, load_wake_plan};

const MUON_LR_SCALE: f64 = 20.0;
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const RUN_WORKFLOW_DISPATCH_IDENTITY_VERSION: u32 = 1;
const FNV1A64_PRIME: u64 = 0x100000001b3;
const DATA_RNG_STREAM: &str = "data";
const MODEL_RNG_STREAM: &str = "model_dropout";
const MODEL_RNG_DOMAIN: u64 = 0xd2b7_4407_b1ce_6e93;

#[derive(Parser)]
#[command(name = "summa-train", about = "Summa model training")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Pretrain or fine-tune a MAL-defined language model.
    Train(TrainArgs),
    /// Score a checkpoint on held-out data with a forward-only pass.
    Eval(EvalArgs),
    /// Rank retrieval queries against a full document pool, not their own batch.
    RetrievalPoolEval(RetrievalPoolArgs),
    /// Decode freely from a checkpoint and score the decodes, not the loss.
    GenerateEval(GenerateEvalArgs),
    /// Validate and resolve a strict WorkflowV2 file.
    ValidateWorkflow(ValidateWorkflowArgs),
    /// Verify an exact-resume checkpoint with the trainer's strict schema.
    VerifyCheckpoint(VerifyCheckpointArgs),
    /// Build an immutable corpus using configured search and record adapters.
    PrepareCorpus(PrepareCorpusArgs),
    /// Compose classified corpus rows into fixed, stratified curriculum stages.
    ComposeCurriculum(ComposeCurriculumArgs),
    /// Export a complete binary/ternary ultra-low-bit checkpoint archive.
    Quantize(QuantizeArgs),
    /// Upgrade an ordinary checkpoint into an explicit sleep-memory topology.
    UpgradeMemoryCheckpoint(UpgradeMemoryCheckpointArgs),
    /// Run WorkflowV2 with pinned workers and first-party native sleep.
    RunWorkflow(RunWorkflowArgs),
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Schedule {
    Wsd,
    Cosine,
}

#[cfg(feature = "cuda")]
const fn default_gpu_metrics_interval_ms() -> u64 {
    1_000
}

#[cfg(not(feature = "cuda"))]
const fn default_gpu_metrics_interval_ms() -> u64 {
    0
}

#[derive(clap::Args)]
struct TrainArgs {
    /// MAL source or exported JSON model configuration.
    #[arg(long)]
    config: PathBuf,
    #[arg(short = 't', long)]
    tokenizer: PathBuf,
    /// Versioned JSON workflow with explicit objectives and phase geometry.
    #[arg(long)]
    workflow: PathBuf,
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
    #[arg(long, default_value_t = 500)]
    checkpoint_every: usize,
    /// Record pre-clip gradient L2 norm for every layer every N optimizer
    /// steps; 0 disables this opt-in visualization/debug metric.
    #[arg(long, default_value_t = 0)]
    layer_metrics_every: usize,
    /// Persistent nvidia-smi sampling interval in milliseconds. Zero disables
    /// sampling. CUDA builds default to 1000; other builds default to disabled.
    #[arg(long, default_value_t = default_gpu_metrics_interval_ms())]
    gpu_metrics_interval_ms: u64,
    /// Physical NVIDIA index, GPU UUID, or PCI bus ID passed to nvidia-smi.
    #[arg(long, default_value = "0")]
    gpu_physical_device: String,
    /// Safetensors checkpoint to fine-tune from. Its exact identity is part of
    /// the run signature, so `--resume` must repeat the same value; the resumed
    /// weights themselves always come from --output, never from this file.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Resume weights, optimizer state, schedule, and corpus position from --output.
    #[arg(long)]
    resume: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Content-pinned first-party adapters and durable stores used by every
    /// periodic in-model-sleep hook in the workflow.
    #[arg(long, requires = "sleep_runtime_sha256")]
    sleep_runtime: Option<PathBuf>,
    /// Exact `sha256:<64 lowercase hex>` identity of --sleep-runtime.
    #[arg(long, requires = "sleep_runtime")]
    sleep_runtime_sha256: Option<String>,
    /// Print the exact content-derived training run signature and exit without
    /// creating output, checkpoint, metric, token-cache, or sleep-runtime state.
    #[arg(long)]
    print_run_signature: bool,
}

/// Held-out objectives supported by the forward-only `eval` command. These are
/// the objectives the wake trainer can optimize end to end, so their reported
/// numbers are directly comparable with training-time metrics.
#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum EvalObjective {
    #[value(name = "causal_lm")]
    CausalLm,
    #[value(name = "contrastive_retrieval")]
    ContrastiveRetrieval,
    #[value(name = "summarization")]
    Summarization,
    #[value(name = "instruction_tuning")]
    InstructionTuning,
    #[value(name = "qa_reasoning")]
    QaReasoning,
    #[value(name = "retrieval_planning")]
    RetrievalPlanning,
}

impl EvalObjective {
    /// Supervised-generation objectives score only their target tokens, so they
    /// share one report block, one prompt-framing contract, and the
    /// oversized-record policy that a prompt-plus-target geometry needs.
    fn is_supervised_generation(self) -> bool {
        matches!(
            self,
            Self::Summarization
                | Self::InstructionTuning
                | Self::QaReasoning
                | Self::RetrievalPlanning
        )
    }
}

#[derive(clap::Args)]
struct EvalArgs {
    /// MAL source or exported JSON model configuration.
    #[arg(long)]
    config: PathBuf,
    #[arg(short = 't', long)]
    tokenizer: PathBuf,
    /// Safetensors weights to score. Loaded strictly against --config.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Held-out shard, repeatable. Structured objectives require .jsonl/.jsonl.zst.
    #[arg(long = "data", required = true)]
    data: Vec<PathBuf>,
    #[arg(long, value_enum)]
    objective: EvalObjective,
    #[arg(long)]
    sequence_length: usize,
    #[arg(long)]
    batch_size: usize,
    /// Stop after this many complete batches. Unset evaluates every shard.
    #[arg(long)]
    max_batches: Option<usize>,
    /// Reservoir-shuffle capacity. Zero reads shards in source order, which is
    /// the reproducible default; --seed only matters above zero.
    #[arg(long, default_value_t = 0)]
    shuffle_buffer: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Rank cut-off for the reported retrieval recall.
    #[arg(long, default_value_t = 10)]
    recall_k: usize,
    /// One-based retrieval read-out layer; omitted reads the final layer. Must
    /// match the layer the retrieval training phase used.
    #[arg(long)]
    retrieval_layer: Option<usize>,
    /// Softmax temperature for the contrastive loss. Omitted uses the task
    /// default, exactly as an unset workflow objective would.
    #[arg(long)]
    temperature: Option<f64>,
    /// Instruction text for a supervised-generation objective. It is part of
    /// every prompt, so it must match the training phase's `task.instruction`
    /// for the reported loss to be comparable. Omitted uses the same built-in
    /// default an unset workflow objective would.
    #[arg(long)]
    instruction: Option<String>,
    /// `qa_reasoning` only: require a `reasoning` field and supervise the
    /// `Reasoning:`/`Answer:` target framing, exactly as the training phase's
    /// `require_reasoning` does.
    #[arg(long)]
    require_reasoning: bool,
    /// JSON report path. The human-readable summary is always printed.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

/// Supervised-generation tasks, the only ones that can be decoded from. Value
/// names match `eval`'s so the two commands take the same `--objective` spelling.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum GenerationObjective {
    #[value(name = "summarization")]
    Summarization,
    #[value(name = "instruction_tuning")]
    InstructionTuning,
    #[value(name = "qa_reasoning")]
    QaReasoning,
    #[value(name = "retrieval_planning")]
    RetrievalPlanning,
}

impl GenerationObjective {
    fn task_name(self) -> &'static str {
        match self {
            Self::Summarization => "summarization",
            Self::InstructionTuning => "instruction_tuning",
            Self::QaReasoning => "qa_reasoning",
            Self::RetrievalPlanning => "retrieval_planning",
        }
    }
}

/// Decode freely from a checkpoint using the trainer's own prompt framing and
/// score the decodes. See `docs/generation-eval.md`.
#[derive(clap::Args)]
struct GenerateEvalArgs {
    /// MAL source or exported JSON model configuration.
    #[arg(long)]
    config: PathBuf,
    #[arg(short = 't', long)]
    tokenizer: PathBuf,
    /// Safetensors weights to decode with. Loaded strictly against --config.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Held-out shard, repeatable. Requires .jsonl/.jsonl.zst.
    #[arg(long = "data", required = true)]
    data: Vec<PathBuf>,
    #[arg(long, value_enum)]
    objective: GenerationObjective,
    /// Prompt budget. The gold target's length is reserved inside it, exactly as
    /// training reserves it, so the decode prompt matches the trained prompt.
    #[arg(long)]
    sequence_length: usize,
    #[arg(long, default_value_t = 60)]
    max_new_tokens: usize,
    /// `<= 0` selects greedy decoding, which is the reproducible default and the
    /// setting under which degenerate copying is visible.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Stop after scoring this many records. Unset decodes every record, which is
    /// slow: decoding is autoregressive.
    #[arg(long)]
    max_records: Option<usize>,
    /// Instruction text. It is part of every prompt, so it must match the
    /// training phase's `task.instruction` for decodes to be in distribution.
    #[arg(long)]
    instruction: Option<String>,
    /// `qa_reasoning` only: supervise and expect the `Reasoning:`/`Answer:`
    /// framing, exactly as the training phase's `require_reasoning` does.
    #[arg(long)]
    require_reasoning: bool,
    /// Write every prompt, decode, and gold target here for review. Metrics
    /// summarize; only reading decodes catches a new failure mode.
    #[arg(long)]
    samples: Option<PathBuf>,
    /// JSON report path. The human-readable summary is always printed.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

/// Rank each query's positive against a pool built from every distinct document
/// the shards mention, instead of against its own batch. See
/// `docs/retrieval-pool-eval.md`.
#[derive(clap::Args)]
struct RetrievalPoolArgs {
    /// MAL source or exported JSON model configuration.
    #[arg(long)]
    config: PathBuf,
    #[arg(short = 't', long)]
    tokenizer: PathBuf,
    /// Safetensors weights to score. Loaded strictly against --config.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Retrieval shard, repeatable. Contributes both queries and pool documents.
    #[arg(long = "data", required = true)]
    data: Vec<PathBuf>,
    /// Extra shard whose documents enlarge the pool without contributing
    /// queries. Use it to make ranking harder than the query set alone allows.
    #[arg(long = "distractors")]
    distractors: Vec<PathBuf>,
    #[arg(long)]
    sequence_length: usize,
    /// Sequences embedded per forward pass. Unlike `eval`, this does not change
    /// the candidate set: the pool is always every pooled document.
    #[arg(long)]
    batch_size: usize,
    /// Rank cut-off for the reported recall.
    #[arg(long, default_value_t = 10)]
    recall_k: usize,
    /// One-based retrieval read-out layer; omitted reads the final layer. Must
    /// match the layer the retrieval training phase used.
    #[arg(long)]
    retrieval_layer: Option<usize>,
    /// JSON report path. The human-readable summary is always printed.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

#[derive(clap::Args)]
struct ValidateWorkflowArgs {
    #[arg(long)]
    workflow: PathBuf,
    /// Exact MAL or exported JSON model topology to validate against every
    /// model-mutating workflow phase, including periodic sleep and QAT.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Print only the exact signature consumed by the standalone WorkflowV2
    /// runtime, without creating or advancing runtime state.
    #[arg(long)]
    signature_only: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CheckpointVerificationFormat {
    Json,
    Tsv,
}

#[derive(clap::Args)]
struct VerifyCheckpointArgs {
    /// Checkpoint output root containing current.json and generations/.
    #[arg(long, conflicts_with = "generation")]
    root: Option<PathBuf>,
    /// A materialized immutable generation directory.
    #[arg(long, conflicts_with = "root")]
    generation: Option<PathBuf>,
    /// Content-addressed sha256-... generation name.
    #[arg(long, requires = "generation")]
    generation_name: Option<String>,
    /// Lowercase SHA-256 of generation-manifest.json.
    #[arg(long, requires = "generation")]
    manifest_sha256: Option<String>,
    /// Metric journal whose committed checkpoint prefix must be valid.
    #[arg(long)]
    metrics: Option<PathBuf>,
    /// Require the metric file to end exactly at the committed prefix.
    #[arg(long, requires = "metrics")]
    exact_metrics: bool,
    /// Stable output for automation. TSV fields are step, generation,
    /// manifest digest, and committed metric records.
    #[arg(long, value_enum, default_value_t = CheckpointVerificationFormat::Json)]
    format: CheckpointVerificationFormat,
}

#[derive(clap::Args)]
struct PrepareCorpusArgs {
    /// Search/materialization/pipeline recipe. All field mappings live here.
    #[arg(long)]
    recipe: PathBuf,
    #[arg(short = 't', long)]
    tokenizer: PathBuf,
    #[arg(short = 'o', long)]
    output: PathBuf,
    /// Restart/audit state; never copied into the immutable corpus.
    #[arg(long)]
    work_directory: PathBuf,
}

#[derive(clap::Args)]
struct ComposeCurriculumArgs {
    /// Versioned composition config with a content-pinned source manifest.
    #[arg(long)]
    config: PathBuf,
    /// Parent directory for the immutable curriculum build.
    #[arg(short = 'o', long)]
    output: PathBuf,
    /// Transient bounded-memory stratum spools; removed after publication.
    #[arg(long)]
    work_directory: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum UltraQuantFormatArg {
    BinaryG128,
    TernaryG128,
    TernaryEntropyG128,
}

impl From<UltraQuantFormatArg> for UltraQuantFormat {
    fn from(value: UltraQuantFormatArg) -> Self {
        match value {
            UltraQuantFormatArg::BinaryG128 => Self::BinaryG128,
            UltraQuantFormatArg::TernaryG128 => Self::TernaryG128,
            UltraQuantFormatArg::TernaryEntropyG128 => Self::TernaryEntropyG128,
        }
    }
}

#[derive(clap::Args)]
struct QuantizeArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(short = 'o', long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    format: UltraQuantFormatArg,
    #[arg(long, default_value_t = 0)]
    ternary_warmup_steps: u64,
    #[arg(long, default_value_t = 0.0)]
    distillation_weight: f64,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    quantize_embeddings: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    quantize_lm_head: bool,
}

#[derive(clap::Args)]
struct UpgradeMemoryCheckpointArgs {
    #[arg(long)]
    source_config: PathBuf,
    #[arg(long)]
    target_config: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(short = 'o', long)]
    output: PathBuf,
}

#[derive(clap::Args)]
struct RunWorkflowArgs {
    #[arg(long)]
    workflow: PathBuf,
    /// Local executable implementing the versioned phase-worker protocol.
    #[arg(long)]
    executor: PathBuf,
    #[arg(long = "executor-arg", allow_hyphen_values = true)]
    executor_arguments: Vec<OsString>,
    /// Hard wall-clock bound for each phase-worker process.
    #[arg(long, default_value_t = 3_600)]
    executor_timeout_seconds: u64,
    #[arg(long, default_value = "workflow-runtime.json")]
    state: PathBuf,
    /// Append-only typed JSONL metric journal emitted by the phase worker.
    #[arg(long, requires = "run_id")]
    metrics: Option<PathBuf>,
    /// Stable metric run identity. Required whenever --metrics is set.
    #[arg(long, requires = "metrics")]
    run_id: Option<String>,
    #[arg(long, requires = "initial_checkpoint_sha256")]
    initial_checkpoint_uri: Option<String>,
    #[arg(long, requires = "initial_checkpoint_uri")]
    initial_checkpoint_sha256: Option<String>,
    #[arg(long)]
    resume: bool,
    /// Content-pinned first-party adapter configuration for standalone sleep
    /// phases. Periodic sleep is executed by the integrated `train` command.
    #[arg(long, requires = "sleep_runtime_sha256")]
    sleep_runtime: Option<PathBuf>,
    #[arg(long, requires = "sleep_runtime")]
    sleep_runtime_sha256: Option<String>,
}

fn load_config(path: &Path) -> Result<ModelDef> {
    if path.extension().is_some_and(|ext| ext == "mal") {
        return summa_llm::parse_mal_file(path);
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
    tier_grads: &[GradientsParams],
) -> Result<Vec<f32>> {
    let mut norms = Vec::with_capacity(model.config().num_layers);
    for layer in 0..model.config().num_layers {
        let mut sum = sum_optional_tensors(
            squared_layer_gradient_norm(model, layer, muon_grads)?,
            squared_layer_gradient_norm(model, layer, adamw_grads)?,
        );
        for gradients in tier_grads {
            sum = sum_optional_tensors(sum, squared_layer_gradient_norm(model, layer, gradients)?);
        }
        let sum = sum.ok_or_else(|| anyhow::anyhow!("layer {} has no gradients", layer + 1))?;
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
    tier_grads: &mut [GradientsParams],
    max_norm: f32,
) -> Result<f32> {
    let mut sum = sum_optional_tensors(
        squared_gradient_norm(model, muon_grads),
        squared_gradient_norm(model, adamw_grads),
    );
    for gradients in tier_grads.iter() {
        sum = sum_optional_tensors(sum, squared_gradient_norm(model, gradients));
    }
    let Some(sum) = sum else {
        return Ok(0.0);
    };
    let norm = scalar_value(sum.sqrt())?;
    if max_norm > 0.0 && norm > max_norm {
        let scale = max_norm / norm;
        scale_gradients(model, muon_grads, scale);
        scale_gradients(model, adamw_grads, scale);
        for gradients in tier_grads {
            scale_gradients(model, gradients, scale);
        }
    }
    Ok(norm)
}

fn scalar_value(tensor: Tensor<1>) -> Result<f32> {
    Ok(tensor.into_data().convert::<f32>().to_vec::<f32>()?[0])
}

fn sum_optional_tensors(left: Option<Tensor<1>>, right: Option<Tensor<1>>) -> Option<Tensor<1>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(tensor), None) | (None, Some(tensor)) => Some(tensor),
        (None, None) => None,
    }
}

fn accumulate_tensor(total: &mut Option<Tensor<1>>, value: Tensor<1>) {
    *total = Some(match total.take() {
        Some(current) => current + value,
        None => value,
    });
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

fn validate_wake_only_metric_schedule(schedule: &summa_train::sleep::SleepSchedule) -> Result<()> {
    let ids = schedule
        .tiers
        .iter()
        .map(|tier| tier.id.as_str())
        .collect::<Vec<_>>();
    ensure!(
        matches!(
            ids.as_slice(),
            ["fast", "slow"] | ["fast", "medium", "slow"]
        ),
        "the stock wake_only trainer emits typed tier metrics and therefore requires tier ids ordered as [fast, slow] or [fast, medium, slow]"
    );
    Ok(())
}

fn metric_memory_tier(id: &str) -> Result<MetricMemoryTier> {
    match id {
        "fast" => Ok(MetricMemoryTier::Fast),
        "medium" => Ok(MetricMemoryTier::Medium),
        "slow" => Ok(MetricMemoryTier::Slow),
        _ => bail!("wake_only tier `{id}` has no typed metric representation"),
    }
}

fn append_wake_only_tier_metrics(
    metrics: &mut MetricWriter,
    context: &MetricContext,
    schedule: &summa_train::sleep::SleepSchedule,
    updates: &[WakeOnlyTierUpdate],
) -> Result<()> {
    validate_wake_only_metric_schedule(schedule)?;
    ensure!(
        updates.windows(2).all(|pair| pair[0].tier < pair[1].tier),
        "wake_only tier update report is not ordered fastest-to-slowest"
    );
    for update in updates {
        let configured = schedule
            .tiers
            .get(update.tier)
            .context("wake_only tier update is outside the configured schedule")?;
        ensure!(
            configured.id == update.tier_id,
            "wake_only tier update identity differs from the configured schedule"
        );
        metrics.append(
            context.clone(),
            MetricEvent::TierUpdate(TierUpdateMetrics {
                transaction_id: update.prospective_update_sha256.clone(),
                tier: metric_memory_tier(&update.tier_id)?,
                receiver_tier: None,
                tier_clock: update.trigger_clock,
                update_period: configured.update_period,
                accumulated_micro_steps: update.accumulated_optimizer_steps,
                outcome: TierUpdateOutcome::Committed,
                update_l2_norm: None,
                reserve_slot: None,
                reserve_generation: None,
                optimizer_state_reset: false,
            }),
        )?;
    }
    Ok(())
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
    ensure!(
        args.sleep_runtime.is_some() == args.sleep_runtime_sha256.is_some(),
        "--sleep-runtime and --sleep-runtime-sha256 must be provided together"
    );
    validate_physical_device_selector(&args.gpu_physical_device)?;
    if args.gpu_metrics_interval_ms > 0 {
        ensure!(
            args.gpu_metrics_interval_ms >= 100,
            "gpu_metrics_interval_ms must be zero or at least 100"
        );
    }
    Ok(())
}

fn resolve_wake_plan(args: &TrainArgs) -> Result<ResolvedWakePlan> {
    load_wake_plan(&args.workflow)
}

fn validate_model_wake_plan(config: &ModelDef, workflow: &ResolvedWakePlan) -> Result<()> {
    let memory_layers = (0..config.num_layers)
        .filter(|&layer| config.block_for_layer(layer).memory.is_some())
        .count();
    ensure!(
        memory_layers == 0 || memory_layers == config.num_layers,
        "model mixes {memory_layers} memory layers with {} ordinary-FFN layers; the stock trainer requires one explicit topology across every layer",
        config.num_layers - memory_layers
    );
    let periodic = workflow
        .phases
        .iter()
        .filter_map(|phase| phase.periodic_sleep.as_ref())
        .collect::<Vec<_>>();
    let wake_only = workflow
        .phases
        .iter()
        .filter_map(|phase| phase.memory_update_mode.as_ref())
        .collect::<Vec<_>>();
    if memory_layers > 0 {
        ensure!(
            periodic.len() == workflow.phases.len() || wake_only.len() == workflow.phases.len(),
            "memory MAL models require one update policy on every wake phase: periodic_sleep or memory_update_mode wake_only"
        );
        if periodic.len() == workflow.phases.len() {
            ensure!(
                wake_only.is_empty(),
                "memory phases cannot combine periodic_sleep with memory_update_mode"
            );
            ensure!(
                periodic.iter().all(|config| *config == periodic[0]),
                "all phases in one memory training run must use an identical periodic_sleep configuration"
            );
            ensure!(
                periodic[0].schedule.clock == summa_train::sleep::UpdateClock::OptimizerSteps,
                "the stock train command supports periodic sleep only at optimizer_steps boundaries; model_tokens can cross multiple memory boundaries inside one indivisible optimizer update"
            );
            validate_sleep_schedule_for_model(
                config,
                &periodic[0].schedule,
                &workflow.phases[0].name,
            )?;
        } else {
            ensure!(
                periodic.is_empty(),
                "wake_only memory phases cannot install periodic_sleep"
            );
            ensure!(
                wake_only.iter().all(|mode| *mode == wake_only[0]),
                "all phases in one memory training run must use an identical memory_update_mode configuration"
            );
            ensure!(
                wake_only[0].schedule().clock == summa_train::sleep::UpdateClock::OptimizerSteps,
                "the stock train command supports wake_only tier updates only at optimizer_steps boundaries"
            );
            validate_sleep_schedule_for_model(
                config,
                wake_only[0].schedule(),
                &workflow.phases[0].name,
            )?;
            validate_wake_only_metric_schedule(wake_only[0].schedule())?;
        }
    } else {
        ensure!(
            periodic.is_empty() && wake_only.is_empty(),
            "periodic_sleep and memory_update_mode require a MAL model with an explicit memory hierarchy"
        );
    }
    for phase in &workflow.phases {
        for (field, value) in [
            ("sequence_length", phase.sequence_length),
            ("batch_size", phase.batch_size),
            ("gradient_accumulation", phase.gradient_accumulation),
        ] {
            ensure!(
                u32::try_from(value).is_ok(),
                "workflow phase `{}` {field} exceeds the checkpoint-metric range",
                phase.name
            );
        }
        ensure!(
            phase.sequence_length <= config.max_seq_len,
            "workflow phase `{}` sequence_length {} exceeds model max_seq_len {}",
            phase.name,
            phase.sequence_length,
            config.max_seq_len
        );
        if let Some(plan) = &phase.quantization {
            ensure!(
                !config.embeddings.tie_weights
                    || plan.recipe.quantize_embeddings == plan.recipe.quantize_lm_head,
                "workflow phase `{}` assigns different embedding and lm_head quantization policies to tied model weights",
                phase.name
            );
        }
        if matches!(phase.objective, TaskConfig::RetrievalRepresentation { .. }) {
            let layer = phase
                .objective
                .retrieval_layer()
                .unwrap_or(config.num_layers);
            ensure!(
                layer <= config.num_layers,
                "workflow phase `{}` requests retrieval layer {layer}, model has {} layers",
                phase.name,
                config.num_layers
            );
            ensure!(
                (0..layer).any(|index| {
                    let block = config.block_for_layer(index);
                    !block.is_ssm() && block.attention.window_size.is_none()
                }),
                "workflow phase `{}` retrieval layer {layer} has no full-attention layer at or before it",
                phase.name
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PhasePlan {
    samples: Option<usize>,
    natural_steps: Option<usize>,
    steps: usize,
}

fn select_phase_steps(
    phase_name: &str,
    samples: usize,
    batch_size: usize,
    gradient_accumulation: usize,
    epochs: usize,
    max_steps: Option<usize>,
) -> Result<(usize, usize)> {
    let steps_per_epoch = (samples / batch_size).div_euclid(gradient_accumulation);
    let natural_steps = steps_per_epoch.checked_mul(epochs).with_context(|| {
        format!("workflow phase `{phase_name}` optimizer-step count overflows usize")
    })?;
    let selected_steps = max_steps.map_or(natural_steps, |cap| natural_steps.min(cap));
    Ok((natural_steps, selected_steps))
}

fn planned_sample_count(
    data: &Path,
    objective: &TaskConfig,
    tokenizer: &Tokenizer,
    sequence_length: usize,
    token_cache: &Path,
    data_binding: &PhaseDataBinding,
) -> Result<usize> {
    data_binding.ensure_still_published()?;
    if matches!(objective, TaskConfig::CausalLm {})
        && let Some(samples) = indexed_causal_sample_count(token_cache, sequence_length)?
    {
        data_binding.ensure_still_published()?;
        println!(
            "token_cache={} indexed_samples={samples}",
            token_cache.display()
        );
        return Ok(samples);
    }
    count_samples(
        data,
        objective,
        tokenizer,
        sequence_length,
        Some(token_cache),
        data_binding,
    )
}

fn plan_training(
    workflow: &ResolvedWakePlan,
    tokenizer: &Tokenizer,
    token_cache_paths: &[PathBuf],
    data_bindings: &[PhaseDataBinding],
) -> Result<(Vec<PhasePlan>, usize)> {
    ensure!(
        token_cache_paths.len() == workflow.phases.len()
            && data_bindings.len() == workflow.phases.len(),
        "every workflow phase must have one token-cache path and authenticated data binding"
    );
    let mut total_steps = 0usize;
    let mut plan = Vec::with_capacity(workflow.phases.len());
    for (phase_index, phase) in workflow.phases.iter().enumerate() {
        let (samples, natural_steps, steps) = match phase.steps {
            Some(steps) => (None, None, steps),
            None => {
                let samples = planned_sample_count(
                    &phase.data,
                    &phase.objective,
                    tokenizer,
                    phase.sequence_length,
                    &token_cache_paths[phase_index],
                    &data_bindings[phase_index],
                )?;
                let (natural_steps, selected_steps) = select_phase_steps(
                    &phase.name,
                    samples,
                    phase.batch_size,
                    phase.gradient_accumulation,
                    phase.epochs,
                    phase.max_steps,
                )?;
                (Some(samples), Some(natural_steps), selected_steps)
            }
        };
        ensure!(
            steps > 0,
            "workflow phase `{}` produces zero complete optimizer steps",
            phase.name
        );
        if let Some(quantization) = &phase.quantization {
            let phase_start = u64::try_from(total_steps)
                .context("workflow phase start exceeds the quantization clock")?;
            let phase_steps = u64::try_from(steps)
                .context("workflow phase length exceeds the quantization clock")?;
            quantization
                .validate_phase_window(phase_start, phase_steps)
                .with_context(|| {
                    format!(
                        "workflow quantization phase `{}` never trains its target format",
                        phase.name
                    )
                })?;
        }
        total_steps = total_steps
            .checked_add(steps)
            .ok_or_else(|| anyhow::anyhow!("workflow optimizer-step count overflows usize"))?;
        plan.push(PhasePlan {
            samples,
            natural_steps,
            steps,
        });
    }
    Ok((plan, total_steps))
}

fn file_sha256(path: &Path) -> Result<String> {
    hash_regular_file(path)
        .map(|(_, sha256)| sha256)
        .with_context(|| format!("failed to hash {}", path.display()))
}

/// Read and authenticate the exact regular-file bytes that a model loader will
/// consume. Hashing and then reopening a pathname is insufficient because a
/// replacement between those operations could load a different checkpoint.
fn read_pinned_checkpoint_bytes(path: &Path, expected_sha256: &str) -> Result<Vec<u8>> {
    read_pinned_checkpoint_bytes_after_open(path, expected_sha256, || Ok(()))
}

fn read_pinned_checkpoint_bytes_after_open(
    path: &Path,
    expected_sha256: &str,
    after_open: impl FnOnce() -> Result<()>,
) -> Result<Vec<u8>> {
    let inspected = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting pinned checkpoint {}", path.display()))?;
    ensure!(
        inspected.is_file() && !inspected.file_type().is_symlink(),
        "pinned checkpoint {} must be a regular non-symlink file",
        path.display()
    );
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("opening pinned checkpoint {}", path.display()))?;
    let opened = file.metadata()?;
    ensure!(opened.is_file(), "opened pinned checkpoint is not a file");
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            inspected.dev() == opened.dev() && inspected.ino() == opened.ino(),
            "pinned checkpoint changed while it was opened"
        );
    }
    after_open()?;
    let expected_bytes = usize::try_from(opened.len())
        .context("pinned checkpoint is too large for this process address space")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected_bytes)
        .context("cannot reserve memory for pinned checkpoint bytes")?;
    let mut buffer = [0_u8; 1024 * 1024];
    while bytes.len() < expected_bytes {
        let remaining = expected_bytes - bytes.len();
        let chunk_bytes = remaining.min(buffer.len());
        let read = file
            .read(&mut buffer[..chunk_bytes])
            .with_context(|| format!("reading pinned checkpoint {}", path.display()))?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    let mut extra = [0_u8; 1];
    let has_extra = file
        .read(&mut extra)
        .with_context(|| format!("checking pinned checkpoint length {}", path.display()))?
        != 0;
    let after = file.metadata()?;
    ensure!(
        !has_extra
            && bytes.len() == expected_bytes
            && after.len() == bytes.len() as u64
            && after.len() == opened.len()
            && after.modified().ok() == opened.modified().ok(),
        "pinned checkpoint changed while it was read"
    );
    let published = fs::symlink_metadata(path)
        .with_context(|| format!("reinspecting pinned checkpoint {}", path.display()))?;
    ensure!(
        published.is_file() && !published.file_type().is_symlink(),
        "pinned checkpoint publication changed while it was read"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            published.dev() == opened.dev() && published.ino() == opened.ino(),
            "pinned checkpoint publication changed while it was read"
        );
    }
    let observed = format!("sha256:{:x}", Sha256::digest(&bytes));
    ensure!(
        observed == expected_sha256,
        "pinned checkpoint hash mismatch: expected {expected_sha256}, observed {observed}"
    );
    Ok(bytes)
}

fn stable_cache_id(value: &str) -> String {
    format!("{:016x}", fnv1a64(value.as_bytes()))
}

#[derive(Serialize)]
struct TokenCacheIdentity<'a> {
    /// Bump whenever the document-token cache encoding or replay semantics
    /// change. This prevents a new trainer from interpreting an old cache.
    version: u32,
    data: &'a str,
    source: &'a Path,
    tokenizer: &'a str,
}

fn token_cache_path(root: &Path, data: &str, source: &Path, tokenizer: &str) -> Result<PathBuf> {
    let identity = serde_json::to_vec(&TokenCacheIdentity {
        version: 1,
        data,
        source,
        tokenizer,
    })?;
    Ok(root.join(format!("{:x}.tokens", Sha256::digest(identity))))
}

fn shuffle_seed(seed: u64, phase: usize, epoch: usize) -> u64 {
    seed.wrapping_add((phase as u64) << 32)
        .wrapping_add(epoch as u64)
}

/// SplitMix64 gives every model microbatch a reproducible device seed without
/// depending on opaque backend RNG state.  The counter is checkpointed, so a
/// resumed process sees exactly the same dropout stream as an uninterrupted
/// process even though model construction itself consumes random numbers.
fn model_microbatch_seed(seed: u64, counter: u64) -> u64 {
    let mut value = seed
        .wrapping_add(MODEL_RNG_DOMAIN)
        .wrapping_add(counter.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn rng_stream<'a>(state: &'a TrainingState, name: &str) -> Result<&'a RngStreamState> {
    state
        .rng_streams
        .iter()
        .find(|stream| stream.name == name)
        .with_context(|| format!("checkpoint is missing required `{name}` RNG stream"))
}

fn rng_stream_mut<'a>(state: &'a mut TrainingState, name: &str) -> Result<&'a mut RngStreamState> {
    state
        .rng_streams
        .iter_mut()
        .find(|stream| stream.name == name)
        .with_context(|| format!("checkpoint is missing required `{name}` RNG stream"))
}

fn validate_wake_rng_state(state: &TrainingState, seed: u64) -> Result<()> {
    let data = rng_stream(state, DATA_RNG_STREAM)?;
    ensure!(
        data.seed == shuffle_seed(seed, state.phase, state.epoch)
            && data.counter == state.records_in_phase as u64,
        "checkpoint data RNG stream disagrees with its phase cursor"
    );
    let model = rng_stream(state, MODEL_RNG_STREAM)?;
    ensure!(
        model.seed == seed,
        "checkpoint model RNG seed differs from this invocation"
    );
    Ok(())
}

fn start_device_sampler(args: &TrainArgs) -> Option<NvidiaSmiSampler> {
    if args.gpu_metrics_interval_ms == 0 {
        return None;
    }
    let config = NvidiaSmiSamplerConfig::new(
        Duration::from_millis(args.gpu_metrics_interval_ms),
        args.gpu_physical_device.clone(),
    )
    .expect("validated device sampler configuration");
    let (sampler, diagnostic) = NvidiaSmiSampler::start_fail_soft(config);
    if let Some(diagnostic) = diagnostic {
        eprintln!(
            "warning: GPU utilization sampling is unavailable and training will continue: {diagnostic}"
        );
    }
    sampler
}

fn emit_device_sampler_drain(
    mut drain: DeviceSamplerDrain,
    metrics: &mut MetricWriter,
    context: &MetricContext,
) -> Result<()> {
    for diagnostic in drain.diagnostics {
        eprintln!("warning: GPU utilization sampler: {diagnostic}");
    }
    drain
        .samples
        .sort_by_key(|sample| sample.collected_at_unix_ms);
    for sample in drain.samples {
        ensure!(
            sample.metrics.sampled_at_unix_ms == sample.collected_at_unix_ms,
            "device sampler timestamp receipt is inconsistent"
        );
        // A sample can race with the preceding trainer event or survive a
        // wall-clock adjustment/resume. Preserve its exact collection time in
        // the typed payload while clamping only the append-only record's time
        // to the journal watermark.
        let emitted_at = metrics
            .state()
            .last_emitted_at_unix_ms
            .map_or(sample.collected_at_unix_ms, |last| {
                last.max(sample.collected_at_unix_ms)
            });
        metrics.append_at(
            context.clone(),
            MetricEvent::DeviceUtilization(sample.metrics),
            emitted_at,
        )?;
    }
    Ok(())
}

fn drain_device_sampler(
    sampler: &mut Option<NvidiaSmiSampler>,
    metrics: &mut MetricWriter,
    context: &MetricContext,
) -> Result<()> {
    if let Some(sampler) = sampler {
        emit_device_sampler_drain(sampler.drain(), metrics, context)?;
    }
    Ok(())
}

fn shutdown_device_sampler(
    sampler: &mut Option<NvidiaSmiSampler>,
    metrics: &mut MetricWriter,
    context: &MetricContext,
) -> Result<()> {
    if let Some(sampler) = sampler {
        emit_device_sampler_drain(sampler.shutdown_and_drain(), metrics, context)?;
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A64_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

#[derive(Serialize)]
struct RunSignature<'a> {
    version: u32,
    workflow: &'a ResolvedWakePlan,
    model: &'a ModelDef,
    initial_checkpoint: Option<String>,
    tokenizer: String,
    data_manifests: &'a [String],
    seed: u64,
    learning_rate: f64,
    weight_decay: f32,
    gradient_clip: f32,
    warmup_steps: usize,
    schedule: Schedule,
    muon_learning_rate_scale: f64,
    gpu_metrics_interval_ms: u64,
    gpu_physical_device: &'a str,
}

fn run_signature(
    args: &TrainArgs,
    workflow: &ResolvedWakePlan,
    config: &ModelDef,
    data_manifests: &[String],
    initial_checkpoint: Option<String>,
) -> Result<String> {
    let encoded = serde_json::to_vec(&RunSignature {
        version: TRAINING_STATE_VERSION,
        workflow,
        model: config,
        initial_checkpoint,
        tokenizer: file_sha256(&args.tokenizer)?,
        data_manifests,
        seed: args.seed,
        learning_rate: args.lr,
        weight_decay: args.weight_decay,
        gradient_clip: args.grad_clip,
        warmup_steps: args.warmup_steps,
        schedule: args.schedule,
        muon_learning_rate_scale: MUON_LR_SCALE,
        gpu_metrics_interval_ms: args.gpu_metrics_interval_ms,
        gpu_physical_device: &args.gpu_physical_device,
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(encoded)))
}

fn bind_phase_data(
    path: &Path,
    tokenizer: &Tokenizer,
    tokenizer_hash: &str,
    cache: &mut HashMap<PathBuf, PhaseDataBinding>,
) -> Result<PhaseDataBinding> {
    let binding = match cache.get(path) {
        Some(binding) => binding.clone(),
        None => {
            let binding = PhaseDataBinding::open(path)?;
            cache.insert(path.to_owned(), binding.clone());
            binding
        }
    };
    if let Some(corpus) = binding.authenticated_corpus() {
        let manifest = corpus.manifest();
        ensure!(
            manifest.build.tokenizer.vocabulary_size == tokenizer.vocab_size(),
            "corpus tokenizer vocabulary {} differs from training tokenizer vocabulary {}",
            manifest.build.tokenizer.vocabulary_size,
            tokenizer.vocab_size()
        );
        ensure!(
            manifest.build.tokenizer.revision == tokenizer_hash,
            "corpus tokenizer revision {} differs from training tokenizer {}",
            manifest.build.tokenizer.revision,
            tokenizer_hash
        );
    }
    Ok(binding)
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

struct ObjectiveForward {
    loss: Tensor<1>,
    router_loss: Option<Tensor<1>>,
    stats: BatchStats,
    retrieval_correct: Option<Tensor<1>>,
    /// Model outputs retained from the same forward pass when the caller asks
    /// for them. Language objectives store selected vocabulary logits;
    /// retrieval stores the unscaled similarity matrix. QAT distillation
    /// consumes them as student outputs and forward-only evaluation consumes
    /// the retrieval matrix for rank metrics. Absent otherwise.
    captured_logits: Option<Tensor<2>>,
}

fn objective_loss(
    model: &Transformer,
    batch: TrainingBatch,
    objective: &TaskConfig,
    capture_logits: bool,
) -> Result<ObjectiveForward> {
    let stats = batch.stats();
    let mut retrieval_correct = None;
    let (loss, router_loss, captured_logits) = match batch {
        TrainingBatch::Language(batch) => {
            let data::LanguageBatch {
                input_ids,
                targets,
                loss_positions,
                ..
            } = *batch;
            match objective {
                TaskConfig::CausalLm {} => {
                    ensure!(
                        loss_positions.is_none(),
                        "causal_lm batch unexpectedly contains a target mask"
                    );
                    if capture_logits {
                        let (loss, router_loss, logits) =
                            model.forward_loss_and_logits_with_router(input_ids, targets);
                        (loss, router_loss, Some(logits))
                    } else {
                        let (loss, router_loss) =
                            model.forward_loss_with_router(input_ids, targets);
                        (loss, router_loss, None)
                    }
                }
                TaskConfig::Summarization { .. }
                | TaskConfig::RetrievalPlanning { .. }
                | TaskConfig::InstructionTuning { .. }
                | TaskConfig::QaReasoning { .. } => {
                    let positions = loss_positions
                        .ok_or_else(|| anyhow::anyhow!("structured batch has no target mask"))?;
                    if capture_logits {
                        let (loss, router_loss, logits) = model
                            .forward_masked_loss_and_logits_with_router(
                                input_ids, targets, positions,
                            );
                        (loss, router_loss, Some(logits))
                    } else {
                        let (loss, router_loss) =
                            model.forward_masked_loss_with_router(input_ids, targets, positions);
                        (loss, router_loss, None)
                    }
                }
                TaskConfig::RetrievalRepresentation { .. } => {
                    bail!("contrastive_retrieval phase produced a language batch")
                }
                TaskConfig::RetrievalRanking { .. }
                | TaskConfig::PairwisePreference {}
                | TaskConfig::VerifiableRl { .. } => {
                    unreachable!("wake projection rejects unsupported task contracts")
                }
            }
        }
        TrainingBatch::Retrieval(batch) => {
            ensure!(
                matches!(objective, TaskConfig::RetrievalRepresentation { .. }),
                "non-retrieval phase produced a retrieval batch"
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
            let similarities = queries.matmul(documents.transpose());
            let captured_logits = capture_logits.then(|| similarities.clone());
            let logits = similarities.div_scalar(temperature);
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
            let router_loss = sum_optional_tensors(query_router_loss, document_router_loss);
            (loss, router_loss, captured_logits)
        }
    };
    Ok(ObjectiveForward {
        loss,
        router_loss,
        stats,
        retrieval_correct,
        captured_logits,
    })
}

fn quantization_teacher_logits(
    teacher: &Transformer,
    batch: &TrainingBatch,
    objective: &TaskConfig,
) -> Result<Tensor<2>> {
    let teacher_logits = match batch {
        TrainingBatch::Language(batch) => {
            let [rows, sequence] = batch.input_ids.dims();
            let vocabulary = teacher.config().vocab_size;
            ensure!(
                vocabulary > 1,
                "quantization teacher vocabulary must contain at least two tokens"
            );
            match objective {
                TaskConfig::CausalLm {} => {
                    ensure!(
                        batch.loss_positions.is_none(),
                        "causal_lm distillation batch unexpectedly contains a target mask"
                    );
                    teacher
                        .forward(batch.input_ids.clone().inner(), 0)
                        .reshape([rows * sequence, vocabulary])
                }
                TaskConfig::Summarization { .. }
                | TaskConfig::RetrievalPlanning { .. }
                | TaskConfig::InstructionTuning { .. }
                | TaskConfig::QaReasoning { .. } => {
                    let positions = batch.loss_positions.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("structured distillation batch has no target mask")
                    })?;
                    teacher.forward_selected_logits(
                        batch.input_ids.clone().inner(),
                        positions.clone().inner(),
                    )
                }
                TaskConfig::RetrievalRepresentation { .. } => {
                    bail!("retrieval objective produced a language distillation batch")
                }
                TaskConfig::RetrievalRanking { .. }
                | TaskConfig::PairwisePreference {}
                | TaskConfig::VerifiableRl { .. } => {
                    unreachable!("wake projection rejects unsupported task contracts")
                }
            }
        }
        TrainingBatch::Retrieval(batch) => {
            ensure!(
                matches!(objective, TaskConfig::RetrievalRepresentation { .. }),
                "non-retrieval phase produced a retrieval distillation batch"
            );
            let layer = objective.retrieval_layer();
            let teacher_queries = teacher.forward_embeddings(
                batch.query_ids.clone().inner(),
                batch.query_end_positions.clone().inner(),
                layer,
            );
            let teacher_documents = teacher.forward_embeddings(
                batch.document_ids.clone().inner(),
                batch.document_end_positions.clone().inner(),
                layer,
            );
            teacher_queries.matmul(teacher_documents.transpose())
        }
    };
    debug_assert!(
        !teacher_logits.device().is_autodiff(),
        "frozen quantization teacher unexpectedly built an autodiff graph"
    );
    // The KL loss is part of the student's autodiff graph. Lift the immutable
    // teacher values back onto that backend without replaying teacher ops.
    Ok(Tensor::from_inner(teacher_logits))
}

fn metric_quantization_format(format: UltraQuantFormat) -> MetricQuantizationFormat {
    match format {
        UltraQuantFormat::BinaryG128 => MetricQuantizationFormat::BinaryG128,
        UltraQuantFormat::TernaryG128 => MetricQuantizationFormat::TernaryG128,
        UltraQuantFormat::TernaryEntropyG128 => MetricQuantizationFormat::TernaryEntropyG128,
    }
}

struct LoadedQuantizationTeacher {
    model: Transformer,
    sha256: String,
    temperature: f64,
    loss_weight: f64,
}

fn disable_quantization_teacher_dropout(config: &mut ModelDef) {
    fn disable_block(block: &mut BlockDef) {
        block.dropout = 0.0;
        block.attention.dropout = 0.0;
        block.ffn.dropout = 0.0;
        if let Some(memory) = &mut block.memory {
            for tier in &mut memory.tiers {
                // A tier's FfnDef owns dropout for both dense and MoE expert
                // execution; MoeDef has no independent dropout setting.
                tier.ffn.dropout = 0.0;
            }
        }
    }

    config.embeddings.dropout = 0.0;
    disable_block(&mut config.block);
    if let Some(pattern) = &mut config.pattern {
        for block in pattern {
            disable_block(block);
        }
    }
}

fn load_quantization_teacher(
    plan: Option<&WorkflowQuantizationPlan>,
    config: &ModelDef,
    device: &Device,
) -> Result<Option<LoadedQuantizationTeacher>> {
    let Some(WorkflowQuantizationTraining::Distillation {
        teacher_checkpoint,
        teacher_sha256,
        temperature,
        loss_weight,
    }) = plan.map(|plan| &plan.training)
    else {
        return Ok(None);
    };
    let teacher_bytes = read_pinned_checkpoint_bytes(teacher_checkpoint, teacher_sha256)
        .with_context(|| {
            format!(
                "cannot authenticate frozen quantization teacher {}",
                teacher_checkpoint.display()
            )
        })?;
    let mut teacher_config = config.clone();
    disable_quantization_teacher_dropout(&mut teacher_config);
    let mut teacher = Transformer::new(&teacher_config, device)?;
    summa_llm::load_safetensors_bytes(
        &mut teacher,
        teacher_bytes,
        &format!("quantization teacher {}", teacher_checkpoint.display()),
    )?;
    Ok(Some(LoadedQuantizationTeacher {
        // Distillation never optimizes the teacher. Keep it on Burn's inner
        // backend so its forward pass does not allocate an autodiff tape; the
        // resulting logits are lifted into the student graph as constants.
        model: teacher.valid(),
        sha256: teacher_sha256.clone(),
        temperature: *temperature,
        loss_weight: *loss_weight,
    }))
}

fn validate_workflow_command(args: ValidateWorkflowArgs) -> Result<()> {
    let workflow = load_workflow_v2(&args.workflow)?;
    if let Some(config) = &args.config {
        let model = load_config(config)?;
        validate_workflow_for_model(&workflow, &model)?;
    }
    if args.signature_only {
        println!("{}", runtime_workflow_signature(&workflow)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&workflow)?);
    }
    Ok(())
}

fn verify_checkpoint_command(args: VerifyCheckpointArgs) -> Result<()> {
    let verified = match (
        args.root,
        args.generation,
        args.generation_name,
        args.manifest_sha256,
    ) {
        (Some(root), None, None, None) => verify_checkpoint_root(&root)?,
        (None, Some(generation), Some(name), Some(manifest_sha256)) => {
            verify_checkpoint_generation(&generation, &name, &manifest_sha256)?
        }
        (None, Some(_), _, _) => {
            bail!("--generation requires both --generation-name and --manifest-sha256")
        }
        _ => bail!(
            "provide either --root, or --generation with --generation-name and --manifest-sha256"
        ),
    };
    if let Some(metrics) = args.metrics {
        validate_checkpoint_metrics(&verified, metrics, args.exact_metrics)?;
    }
    match args.format {
        CheckpointVerificationFormat::Json => {
            println!("{}", serde_json::to_string(&verified)?);
        }
        CheckpointVerificationFormat::Tsv => println!(
            "{}\t{}\t{}\t{}",
            verified.global_step,
            verified.generation,
            verified.manifest_sha256,
            verified.metric_records
        ),
    }
    Ok(())
}

fn validate_checkpoint_metrics(
    verified: &checkpoint::VerifiedCheckpoint,
    metrics: impl AsRef<Path>,
    exact: bool,
) -> Result<()> {
    let expected_run_id = stable_cache_id(&verified.workflow_signature);
    if exact {
        validate_metric_snapshot_for_run(
            metrics,
            &expected_run_id,
            verified.metric_records,
            verified.global_step as u64,
        )?;
    } else {
        validate_metric_prefix_for_run(
            metrics,
            &expected_run_id,
            verified.metric_records,
            verified.global_step as u64,
        )?;
    }
    Ok(())
}

fn prepare_corpus_command(args: PrepareCorpusArgs) -> Result<()> {
    let recipe = SearchApiPostgresCorpusRecipe::load(&args.recipe)?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let tokenizer_revision = file_sha256(&args.tokenizer)?;
    let tokenizer = SummaCorpusTokenizer::new(&tokenizer, tokenizer_revision)?;
    let (path, manifest) = recipe.run(&tokenizer, &args.output, &args.work_directory)?;
    println!(
        "published={} manifest={} unique_tokens={} exposure_tokens={}",
        path.display(),
        manifest.manifest_sha256,
        manifest.build.stats.unique_tokens,
        manifest.build.stats.exposure_tokens,
    );
    Ok(())
}

fn compose_curriculum_command(args: ComposeCurriculumArgs) -> Result<()> {
    let config = CurriculumCompositionConfig::load(&args.config)?;
    let (path, manifest) = config.run(&args.config, &args.output, &args.work_directory)?;
    manifest.verify(&path)?;
    println!(
        "published={} manifest={} stages={} tokens={}",
        path.display(),
        manifest.manifest_sha256,
        manifest.build.stages.len(),
        manifest
            .build
            .stages
            .iter()
            .try_fold(0_u64, |total, stage| total.checked_add(stage.tokens))
            .context("curriculum token total overflows u64")?,
    );
    Ok(())
}

fn quantize_command(args: QuantizeArgs) -> Result<()> {
    let recipe = QuantizationRecipe {
        format: args.format.into(),
        group_size: BONSAI_GROUP_SIZE,
        fake_quant_start_step: 0,
        ternary_warmup_steps: args.ternary_warmup_steps,
        distillation_weight: args.distillation_weight,
        quantize_embeddings: args.quantize_embeddings,
        quantize_lm_head: args.quantize_lm_head,
    };
    let manifest = export_safetensors_archive(&args.checkpoint, &args.output, &recipe)?;
    println!(
        "published={} matrices={} floating_tensors={} archive_bits_per_weight={:.4}",
        args.output.display(),
        manifest.matrices.len(),
        manifest.floating_tensors.len(),
        manifest.true_average_bits_per_weight()?,
    );
    Ok(())
}

fn upgrade_memory_checkpoint_command(args: UpgradeMemoryCheckpointArgs) -> Result<()> {
    ensure!(
        !args.output.exists(),
        "memory-upgrade output {} already exists",
        args.output.display()
    );
    let source = load_config(&args.source_config)?;
    let target = load_config(&args.target_config)?;
    let device = summa_llm::default_device();
    let mut model = Transformer::new(&target, &device)?;
    upgrade_safetensors_to_memory(&mut model, &source, &args.checkpoint, &args.output)?;
    println!("published={}", args.output.display());
    Ok(())
}

fn validate_cli_native_sleep_selection(
    workflow: &ResolvedWorkflow,
    sleep_runtime_configured: bool,
) -> Result<()> {
    let standalone = workflow
        .phases
        .iter()
        .find(|phase| phase.kind == PhaseKind::Sleep);
    ensure!(
        standalone.is_some() == sleep_runtime_configured,
        if let Some(phase) = standalone {
            format!(
                "workflow sleep phase `{}` requires --sleep-runtime and --sleep-runtime-sha256",
                phase.name
            )
        } else {
            "--sleep-runtime is only valid for a workflow with a standalone sleep phase".to_owned()
        }
    );
    if let Some(phase) = workflow
        .phases
        .iter()
        .find(|phase| phase.periodic_sleep.is_some())
    {
        bail!(
            "workflow phase `{}` enables periodic in-model sleep but the stock run-workflow CLI has no native wake-boundary executor configured. Embed summa-train, register a NativePeriodicWakeExecutor, and run NativeWorkflowHost",
            phase.name
        );
    }
    Ok(())
}

/// Bind every implementation selected by the stock WorkflowV2 CLI into the
/// atomic resume identity. The external worker still receives its own exact
/// digest on the wire; this composite is only the durable dispatch identity.
fn run_workflow_dispatch_identity(
    external_execution_sha256: &str,
    sleep_runtime_sha256: Option<&str>,
) -> Result<String> {
    let external = ImmutableArtifact::new(
        "dispatch://external-phase-worker",
        external_execution_sha256,
    )?;
    let sleep = sleep_runtime_sha256
        .map(|sha256| ImmutableArtifact::new("dispatch://native-sleep-runtime", sha256))
        .transpose()?;
    let mut hasher = Sha256::new();
    for part in [
        b"summa-run-workflow-dispatch".as_slice(),
        &RUN_WORKFLOW_DISPATCH_IDENTITY_VERSION.to_le_bytes(),
        &PHASE_WORKER_PROTOCOL_VERSION.to_le_bytes(),
        external.sha256().as_bytes(),
        sleep
            .as_ref()
            .map_or(b"none".as_slice(), |artifact| artifact.sha256().as_bytes()),
    ] {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn run_workflow_command(args: RunWorkflowArgs) -> Result<()> {
    let workflow = load_workflow_v2(&args.workflow)?;
    ensure!(
        args.sleep_runtime.is_some() == args.sleep_runtime_sha256.is_some(),
        "--sleep-runtime and --sleep-runtime-sha256 must be provided together"
    );
    validate_cli_native_sleep_selection(&workflow, args.sleep_runtime.is_some())?;
    let executor = ExternalPhaseExecutor::new(&args.executor, args.executor_arguments)?
        .with_timeout(Duration::from_secs(args.executor_timeout_seconds))?;
    let dispatch_sha256 = run_workflow_dispatch_identity(
        &executor.execution_identity(),
        args.sleep_runtime_sha256.as_deref(),
    )?;
    let mut checkpoint = AtomicRuntimeCheckpoint::new(&args.state, dispatch_sha256)?;

    // Authenticate the complete dispatch identity before loading a runtime
    // factory, whose validated configuration creates output stores. This
    // read-only preflight also precedes metric-prefix truncation in `load`.
    if args.resume {
        ensure!(
            args.initial_checkpoint_uri.is_none(),
            "--resume cannot replace the initial immutable checkpoint"
        );
        checkpoint.verify_execution_identity()?;
    }

    let mut native_context = NativeSleepContextRegistry::new();
    if let (Some(path), Some(sha256)) = (&args.sleep_runtime, &args.sleep_runtime_sha256) {
        native_context
            .register_phase_factory(BuiltinSleepPhaseContextFactory::load(path, sha256)?)?;
    }
    let mut state = if args.resume {
        match (&args.metrics, &args.run_id) {
            (Some(metrics), Some(run_id)) => {
                checkpoint.configure_metrics(metrics, run_id, true)?;
            }
            (None, None) => {}
            _ => bail!("--metrics and --run-id must be provided together"),
        }
        checkpoint.load(&workflow)?
    } else {
        match (&args.metrics, &args.run_id) {
            (Some(metrics), Some(run_id)) => {
                checkpoint.configure_metrics(metrics, run_id, false)?;
            }
            (None, None) => {}
            _ => bail!("--metrics and --run-id must be provided together"),
        }
        let initial_checkpoint = args
            .initial_checkpoint_uri
            .zip(args.initial_checkpoint_sha256)
            .map(|(uri, sha256)| ImmutableModelCheckpoint::new(uri, sha256))
            .transpose()?;
        let state = WorkflowRunState::new(&workflow, initial_checkpoint)?;
        checkpoint.initialize(&state)?;
        state
    };
    let mut registry = ExecutorRegistry::new();
    for kind in ALL_PHASE_KINDS {
        if !matches!(kind, PhaseKind::Sleep | PhaseKind::Promotion) {
            registry.register(kind, executor.clone())?;
        }
    }
    registry.register(
        PhaseKind::Sleep,
        NativeSleepPhaseExecutor::new(state.workflow_signature().to_owned())?,
    )?;
    registry.register(PhaseKind::Promotion, NativePromotionExecutor)?;
    let status = run_until_yield_or_complete(
        &workflow,
        &mut state,
        &mut registry,
        &mut native_context,
        &mut checkpoint,
    )?;
    match status {
        RuntimeStatus::AlreadyComplete => println!(
            "workflow=complete phases={} checkpoint={}",
            state.completed_phases().len(),
            state
                .current_checkpoint()
                .map_or("none", ImmutableModelCheckpoint::uri)
        ),
        RuntimeStatus::Yielded {
            phase_index,
            phase_name,
        } => println!(
            "workflow=yielded phase={} name={} state={}",
            phase_index,
            phase_name,
            args.state.display()
        ),
        RuntimeStatus::PhaseCommitted { .. } => {
            unreachable!("run-until returns only yielded or complete")
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    match Cli::parse().command {
        Command::Train(args) => trainer::train(args),
        Command::Eval(args) => eval::evaluate(args),
        Command::RetrievalPoolEval(args) => retrieval_pool::evaluate(args),
        Command::GenerateEval(args) => generate_eval::evaluate(args),
        Command::ValidateWorkflow(args) => validate_workflow_command(args),
        Command::VerifyCheckpoint(args) => verify_checkpoint_command(args),
        Command::PrepareCorpus(args) => prepare_corpus_command(args),
        Command::ComposeCurriculum(args) => compose_curriculum_command(args),
        Command::Quantize(args) => quantize_command(args),
        Command::UpgradeMemoryCheckpoint(args) => upgrade_memory_checkpoint_command(args),
        Command::RunWorkflow(args) => run_workflow_command(args),
    }
}

#[cfg(test)]
mod tests {
    use burn::module::{AutodiffModule, ParamId};
    use burn::tensor::{Int, TensorData};
    use summa_llm::get_builtin_model;

    use super::*;

    #[test]
    fn natural_epoch_step_cap_never_overasks_or_runs_unbounded() {
        // 1,600 samples / 8 per batch / 4-way accumulation = 50 steps.
        assert_eq!(
            select_phase_steps("retrieval", 1_600, 8, 4, 1, None).unwrap(),
            (50, 50)
        );
        assert_eq!(
            select_phase_steps("retrieval", 1_600, 8, 4, 1, Some(20)).unwrap(),
            (50, 20)
        );
        // A cap larger than the data is not an exact request: the natural
        // epoch wins, avoiding the deterministic short-shard crash that an
        // exact `steps` value would cause.
        assert_eq!(
            select_phase_steps("retrieval", 1_600, 8, 4, 1, Some(200)).unwrap(),
            (50, 50)
        );
    }

    #[test]
    fn coincident_wake_only_updates_emit_ordered_committed_metrics() {
        let schedule: summa_train::sleep::SleepSchedule =
            serde_json::from_value(serde_json::json!({
                "clock": "optimizer_steps",
                "terminal_consolidation": "distill_into_base_v1",
                "tiers": [
                    {"id": "fast", "update_period": 1, "reserve_slots": 1},
                    {"id": "slow", "update_period": 2, "reserve_slots": 2}
                ]
            }))
            .unwrap();
        let updates = vec![
            WakeOnlyTierUpdate {
                tier: 0,
                tier_id: "fast".into(),
                trigger_clock: 2,
                accumulated_optimizer_steps: 1,
                prospective_update_sha256: format!("sha256:{}", "a".repeat(64)),
            },
            WakeOnlyTierUpdate {
                tier: 1,
                tier_id: "slow".into(),
                trigger_clock: 2,
                accumulated_optimizer_steps: 2,
                prospective_update_sha256: format!("sha256:{}", "b".repeat(64)),
            },
        ];
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let mut metrics = MetricWriter::create(&path, "wake-only-metrics").unwrap();
        let context = MetricContext {
            global_step: 2,
            phase: MetricPhase {
                index: 0,
                name: "wake-only".into(),
                kind: MetricPhaseKind::Pretrain,
            },
            checkpoint_hash: None,
        };
        append_wake_only_tier_metrics(&mut metrics, &context, &schedule, &updates).unwrap();
        metrics.sync_all().unwrap();
        drop(metrics);

        let records = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<summa_train::metrics::MetricRecord>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        let projected = records
            .iter()
            .map(|record| match &record.event {
                MetricEvent::TierUpdate(update) => (
                    update.tier,
                    update.update_period,
                    update.accumulated_micro_steps,
                    update.outcome,
                    update.receiver_tier,
                    update.reserve_slot,
                    update.optimizer_state_reset,
                ),
                other => panic!("unexpected metric {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            projected,
            vec![
                (
                    MetricMemoryTier::Fast,
                    1,
                    1,
                    TierUpdateOutcome::Committed,
                    None,
                    None,
                    false,
                ),
                (
                    MetricMemoryTier::Slow,
                    2,
                    2,
                    TierUpdateOutcome::Committed,
                    None,
                    None,
                    false,
                ),
            ]
        );
    }

    #[cfg(unix)]
    struct GenerationSwapGuard {
        original: PathBuf,
        replacement: PathBuf,
        parked_original: PathBuf,
        active: bool,
    }

    #[cfg(unix)]
    impl GenerationSwapGuard {
        fn new(original: PathBuf, replacement: PathBuf) -> Self {
            let parked_original =
                original.with_extension(format!("resume-aba-{}", std::process::id()));
            Self {
                original,
                replacement,
                parked_original,
                active: false,
            }
        }

        fn swap(&mut self) -> Result<()> {
            fs::rename(&self.original, &self.parked_original)?;
            if let Err(error) = fs::rename(&self.replacement, &self.original) {
                let _ = fs::rename(&self.parked_original, &self.original);
                return Err(error.into());
            }
            self.active = true;
            Ok(())
        }

        fn restore(&mut self) -> Result<()> {
            if !self.active {
                return Ok(());
            }
            fs::rename(&self.original, &self.replacement)?;
            fs::rename(&self.parked_original, &self.original)?;
            self.active = false;
            Ok(())
        }
    }

    #[cfg(unix)]
    impl Drop for GenerationSwapGuard {
        fn drop(&mut self) {
            let _ = self.restore();
        }
    }

    #[test]
    fn quantization_teacher_disables_every_dropout_without_mutating_student() {
        let student = summa_llm::parse_mal(
            r#"
            ffn dense { hidden_dim: 16 activation: swiglu dropout: 0.11 }
            ffn routed {
                hidden_dim: 12 activation: swiglu dropout: 0.22
                moe { experts: 3 top_k: 1 }
            }
            memory cms {
                tier fast {
                    ffn: routed
                    reserve_experts { capacity: 1 rank: 2 top_k: 1 }
                }
                tier slow {
                    ffn: dense residual_init: zero
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                }
            }
            block ordinary {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 dropout: 0.33 position_encoding: none }
                ffn: dense dropout: 0.44
            }
            block sleeping {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 dropout: 0.55 position_encoding: none }
                memory: cms dropout: 0.66
            }
            model teacher-dropout {
                vocab_size: 32 max_seq_len: 8 hidden_size: 8 num_layers: 2
                block: sleeping pattern: [ordinary, sleeping]
                embeddings { dropout: 0.77 tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let mut teacher = student.clone();
        disable_quantization_teacher_dropout(&mut teacher);

        assert!(student.embeddings.dropout > 0.0);
        assert!(student.block.dropout > 0.0);
        assert!(
            student
                .block
                .memory
                .as_ref()
                .unwrap()
                .tiers
                .iter()
                .all(|tier| tier.ffn.dropout > 0.0)
        );
        assert_eq!(teacher.embeddings.dropout, 0.0);
        for block in std::iter::once(&teacher.block)
            .chain(teacher.pattern.as_deref().unwrap_or_default().iter())
        {
            assert_eq!(block.dropout, 0.0);
            assert_eq!(block.attention.dropout, 0.0);
            assert_eq!(block.ffn.dropout, 0.0);
            if let Some(memory) = &block.memory {
                assert!(memory.tiers.iter().all(|tier| tier.ffn.dropout == 0.0));
            }
        }
    }

    #[test]
    fn tied_model_rejects_split_qat_embedding_and_output_policies_before_training() {
        let mut model = small_hybrid();
        model.embeddings.tie_weights = true;
        let workflow = ResolvedWakePlan {
            version: 2,
            phases: vec![wake::ResolvedWakePhase {
                name: "invalid-tied-qat".into(),
                phase_kind: PhaseKind::Quantization,
                data: "unused.jsonl".into(),
                objective: TaskConfig::CausalLm {},
                sequence_length: 8,
                batch_size: 1,
                gradient_accumulation: 1,
                epochs: 1,
                shuffle_buffer: 1,
                steps: Some(1),
                max_steps: None,
                loss_weight: 1.0,
                learning_rate_scale: 1.0,
                quantization: Some(WorkflowQuantizationPlan {
                    recipe: QuantizationRecipe {
                        format: UltraQuantFormat::BinaryG128,
                        group_size: BONSAI_GROUP_SIZE,
                        fake_quant_start_step: 0,
                        ternary_warmup_steps: 0,
                        distillation_weight: 0.0,
                        quantize_embeddings: true,
                        quantize_lm_head: false,
                    },
                    start_step: 0,
                    end_step: None,
                    warmup_format: None,
                    warmup_steps: 0,
                    training: WorkflowQuantizationTraining::Qat {
                        straight_through: true,
                    },
                }),
                periodic_sleep: None,
                memory_update_mode: None,
            }],
        };

        let error = validate_model_wake_plan(&model, &workflow)
            .unwrap_err()
            .to_string();
        assert!(error.contains("tied model weights"), "{error}");
    }

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
            workflow: "workflow.json".into(),
            output: "checkpoint".into(),
            lr: 3e-4,
            weight_decay: 0.1,
            grad_clip: 1.0,
            warmup_steps: 10,
            schedule: Schedule::Wsd,
            checkpoint_every: 0,
            layer_metrics_every: 0,
            gpu_metrics_interval_ms: 0,
            gpu_physical_device: "0".into(),
            checkpoint: None,
            resume: false,
            seed: 0,
            sleep_runtime: None,
            sleep_runtime_sha256: None,
            print_run_signature: false,
        }
    }

    /// `eval` and `generate-eval` are run back to back on the same objective, so
    /// a reader must not have to remember that one spells it `qa_reasoning` and
    /// the other `qa-reasoning`. Derived value names would kebab-case these.
    #[test]
    fn generate_eval_objectives_are_spelled_exactly_as_eval_spells_them() {
        for objective in [
            "summarization",
            "instruction_tuning",
            "qa_reasoning",
            "retrieval_planning",
        ] {
            for command in ["eval", "generate-eval"] {
                let mut arguments = vec![
                    "summa-train",
                    command,
                    "--config",
                    "model.mal",
                    "--tokenizer",
                    "tokenizer.json",
                    "--checkpoint",
                    "weights.safetensors",
                    "--data",
                    "holdout.jsonl",
                    "--objective",
                    objective,
                    "--sequence-length",
                    "128",
                ];
                if command == "eval" {
                    arguments.extend(["--batch-size", "2"]);
                }
                assert!(
                    Cli::try_parse_from(arguments).is_ok(),
                    "`{command} --objective {objective}` must parse"
                );
            }
        }
    }

    #[test]
    fn curriculum_composition_cli_requires_config_output_and_work_directory() {
        let cli = Cli::try_parse_from([
            "summa-train",
            "compose-curriculum",
            "--config",
            "mixture.json",
            "--output",
            "corpus",
            "--work-directory",
            "work",
        ])
        .unwrap();
        let Command::ComposeCurriculum(args) = cli.command else {
            panic!("wrong parsed command");
        };
        assert_eq!(args.config, Path::new("mixture.json"));
        assert_eq!(args.output, Path::new("corpus"));
        assert_eq!(args.work_directory, Path::new("work"));

        assert!(
            Cli::try_parse_from([
                "summa-train",
                "compose-curriculum",
                "--config",
                "mixture.json",
                "--output",
                "corpus",
            ])
            .is_err()
        );
    }

    #[test]
    fn workflow_metrics_require_a_run_id_and_parse_as_a_pair() {
        let cli = Cli::try_parse_from([
            "summa-train",
            "run-workflow",
            "--workflow",
            "workflow.json",
            "--executor",
            "./worker",
            "--metrics",
            "metrics.jsonl",
            "--run-id",
            "run-42",
        ])
        .unwrap();
        let Command::RunWorkflow(args) = cli.command else {
            panic!("wrong parsed command");
        };
        assert_eq!(args.metrics.as_deref(), Some(Path::new("metrics.jsonl")));
        assert_eq!(args.run_id.as_deref(), Some("run-42"));

        assert!(
            Cli::try_parse_from([
                "summa-train",
                "run-workflow",
                "--workflow",
                "workflow.json",
                "--executor",
                "./worker",
                "--metrics",
                "metrics.jsonl",
            ])
            .is_err()
        );
    }

    #[test]
    fn workflow_resume_identity_binds_the_exact_native_sleep_runtime() {
        let directory = tempfile::tempdir().unwrap();
        let workflow: summa_train::workflow::WorkflowV2 =
            serde_json::from_value(serde_json::json!({
                "version": 2,
                "phases": [{
                    "name": "pretrain",
                    "type": "pretrain",
                    "task": {"type": "causal_lm"},
                    "data": "data.jsonl",
                    "sequence_length": 8,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "steps": 1
                }]
            }))
            .unwrap();
        let workflow = workflow
            .resolve(&directory.path().join("workflow.json"))
            .unwrap();
        let external = format!("sha256:{}", "a".repeat(64));
        let first_sleep = format!("sha256:{}", "b".repeat(64));
        let replacement_sleep = format!("sha256:{}", "c".repeat(64));
        let first_dispatch = run_workflow_dispatch_identity(&external, Some(&first_sleep)).unwrap();
        let replacement_dispatch =
            run_workflow_dispatch_identity(&external, Some(&replacement_sleep)).unwrap();
        assert_ne!(first_dispatch, replacement_dispatch);
        assert_ne!(
            first_dispatch,
            run_workflow_dispatch_identity(&external, None).unwrap()
        );

        let state_path = directory.path().join("runtime.json");
        let metrics_path = directory.path().join("metrics.jsonl");
        let state = WorkflowRunState::new(&workflow, None).unwrap();
        let mut writer = AtomicRuntimeCheckpoint::new(&state_path, first_dispatch).unwrap();
        writer
            .configure_metrics(&metrics_path, "dispatch-binding", false)
            .unwrap();
        writer.initialize(&state).unwrap();
        drop(writer);
        fs::write(&metrics_path, b"uncommitted metric tail\n").unwrap();
        let before = fs::read(&state_path).unwrap();
        let before_metrics = fs::read(&metrics_path).unwrap();

        let mut replacement =
            AtomicRuntimeCheckpoint::new(&state_path, replacement_dispatch).unwrap();
        replacement
            .configure_metrics(&metrics_path, "dispatch-binding", true)
            .unwrap();
        let error = replacement
            .verify_execution_identity()
            .unwrap_err()
            .to_string();
        assert!(error.contains("different execution dispatch"), "{error}");
        assert_eq!(
            fs::read(&state_path).unwrap(),
            before,
            "a mismatched native runtime identity mutated workflow state"
        );
        assert_eq!(
            fs::read(&metrics_path).unwrap(),
            before_metrics,
            "a mismatched native runtime identity truncated the metric journal"
        );
    }

    #[test]
    fn checked_in_workflow_examples_remain_strictly_valid() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for name in [
            "workflow.example.json",
            "workflow.sleep.example.json",
            "workflow.education.example.json",
        ] {
            load_workflow_v2(&root.join(name))
                .unwrap_or_else(|error| panic!("{name} is invalid: {error:#}"));
        }
    }

    #[test]
    fn education_workflow_matches_declared_sleep_mal_and_reserve_topology() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workflow_path = root.join("workflow.education.example.json");
        let model_path = root.join("../summa-mal/well-known/retriever_300m_moe_sleep.mal");
        let workflow = load_workflow_v2(&workflow_path).unwrap();
        let model = load_config(&model_path).unwrap();

        validate_workflow_for_model(&workflow, &model).unwrap();
        let optimizer_phases = workflow
            .phases
            .iter()
            .filter(|phase| phase.kind.uses_optimizer())
            .collect::<Vec<_>>();
        assert!(!optimizer_phases.is_empty());
        assert!(
            optimizer_phases
                .iter()
                .all(|phase| phase.periodic_sleep.is_some())
        );
        assert!(
            optimizer_phases
                .windows(2)
                .all(|pair| { pair[0].periodic_sleep.as_ref() == pair[1].periodic_sleep.as_ref() })
        );
        assert!(
            workflow
                .phases
                .iter()
                .filter(|phase| !phase.kind.uses_optimizer())
                .all(|phase| phase.periodic_sleep.is_none())
        );

        let mut wrong_capacity = workflow.clone();
        for phase in wrong_capacity
            .phases
            .iter_mut()
            .filter(|phase| phase.kind.uses_optimizer())
        {
            phase.periodic_sleep.as_mut().unwrap().schedule.tiers[1].reserve_slots += 1;
        }
        let error = validate_workflow_for_model(&wrong_capacity, &model)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("preallocates") && error.contains("medium"),
            "{error}"
        );
    }

    #[test]
    fn education_qat_warmup_starts_at_its_global_phase_boundary() {
        let workflow = load_workflow_v2(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.education.example.json"),
        )
        .unwrap();
        let qat_index = workflow
            .phases
            .iter()
            .position(|phase| phase.name == "binary-qat")
            .unwrap();
        let preceding_steps = workflow.phases[..qat_index]
            .iter()
            .filter(|phase| phase.kind.updates_model())
            .filter_map(|phase| phase.steps)
            .sum::<usize>();
        let quantization = workflow.phases[qat_index].quantization.as_ref().unwrap();
        assert_eq!(quantization.start_step, preceding_steps);
        let plan = WorkflowQuantizationPlan::from_workflow(quantization).unwrap();
        assert_eq!(
            plan.format_at(preceding_steps as u64),
            Some(UltraQuantFormat::TernaryG128)
        );
        assert_eq!(
            plan.format_at(preceding_steps as u64 + 400),
            Some(UltraQuantFormat::BinaryG128)
        );
    }

    #[test]
    fn stock_workflow_cli_rejects_native_sleep_before_runtime_state_creation() {
        let workflow = load_workflow_v2(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.sleep.example.json"),
        )
        .unwrap();
        let error = validate_cli_native_sleep_selection(&workflow, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("NativePeriodicWakeExecutor"), "{error}");
        assert!(error.contains("stock run-workflow CLI"), "{error}");
    }

    fn sleep_memory_test_model() -> ModelDef {
        summa_llm::parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
                tier medium {
                    ffn: base residual_init: zero
                    reserve_experts { capacity: 4 rank: 3 top_k: 1 }
                }
                tier slow {
                    ffn: base residual_init: zero
                    reserve_experts { capacity: 8 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 32 max_seq_len: 4096 hidden_size: 8 num_layers: 1
                block: {
                    attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
                    memory: cms
                    dropout: 0.0
                }
            }
            "#,
        )
        .unwrap()
    }

    fn write_wake_only_sleep_workflow(
        mutate: impl FnOnce(&mut serde_json::Value),
    ) -> (tempfile::TempDir, ResolvedWakePlan) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.sleep.example.json");
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(source).unwrap()).unwrap();
        json["phases"].as_array_mut().unwrap().truncate(1);
        mutate(&mut json);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("workflow.json");
        fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
        let workflow = load_wake_plan(&path).unwrap();
        (directory, workflow)
    }

    fn write_wake_only_memory_workflow() -> (tempfile::TempDir, ResolvedWakePlan) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("workflow.sleep.example.json");
        let mut json: serde_json::Value =
            serde_json::from_slice(&fs::read(source).unwrap()).unwrap();
        json["phases"].as_array_mut().unwrap().truncate(1);
        let schedule = json["phases"][0]["periodic_sleep"]["schedule"].take();
        json["phases"][0]
            .as_object_mut()
            .unwrap()
            .remove("periodic_sleep");
        json["phases"][0]["memory_update_mode"] = serde_json::json!({
            "type": "wake_only",
            "schedule": schedule
        });
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("workflow.json");
        fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
        let workflow = load_wake_plan(&path).unwrap();
        (directory, workflow)
    }

    #[test]
    fn wake_only_configuration_and_optimizer_scopes_are_checkpoint_bound() {
        let (_directory, workflow) = write_wake_only_memory_workflow();
        let config = sleep_memory_test_model();
        validate_model_wake_plan(&config, &workflow).unwrap();
        let device = summa_llm::Device::ndarray().autodiff();
        let model = Transformer::new(&config, &device).unwrap();
        let mode = workflow.phases[0].memory_update_mode.clone().unwrap();
        let bank =
            TierOptimizerBank::new(&model, mode.schedule(), mode.tier_optimizer().clone()).unwrap();
        let state = TrainingState {
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
            workflow_signature: format!("sha256:{}", "a".repeat(64)),
            data_manifest_hash: Some(format!("sha256:{}", "b".repeat(64))),
            parameter_ids: parameter_ids(&model),
            optimizer_states: vec![OptimizerStateRef {
                scope: "wake".into(),
                adamw: "adamw-state.bpk".into(),
                muon: "muon-state.bpk".into(),
                gradient_accumulator: None,
                update_clock: 0,
            }],
            memory_update: TrainingMemoryUpdateState::WakeOnly {
                config: mode,
                optimizer_scopes: bank.scopes().unwrap(),
            },
            sleep: None,
            artifacts: Vec::new(),
            evaluator_hashes: Vec::new(),
            rng_streams: vec![
                RngStreamState {
                    name: DATA_RNG_STREAM.into(),
                    seed: 0,
                    counter: 0,
                },
                RngStreamState {
                    name: MODEL_RNG_STREAM.into(),
                    seed: 0,
                    counter: 0,
                },
            ],
            wake_context_buffer: Vec::new(),
            quantization: None,
        };
        state.validate().unwrap();
        trainer::preflight_resumed_memory_mode(&workflow, &state).unwrap();

        let mut replacement = workflow.clone();
        let MemoryUpdateMode::WakeOnly { tier_optimizer, .. } =
            replacement.phases[0].memory_update_mode.as_mut().unwrap();
        tier_optimizer.learning_rate *= 2.0;
        let error = trainer::preflight_resumed_memory_mode(&replacement, &state)
            .unwrap_err()
            .to_string();
        assert!(error.contains("differs from the exact workflow"), "{error}");
    }

    #[test]
    fn wake_geometry_must_fit_checkpoint_metrics() {
        if usize::BITS <= 32 {
            return;
        }
        let (_directory, mut workflow) = write_wake_only_memory_workflow();
        workflow.phases[0].batch_size = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
        let error = validate_model_wake_plan(&sleep_memory_test_model(), &workflow)
            .unwrap_err()
            .to_string();
        assert!(error.contains("batch_size"), "{error}");
        assert!(error.contains("checkpoint-metric range"), "{error}");
    }

    #[test]
    fn stock_memory_trainer_rejects_model_token_sleep_boundaries() {
        let (_directory, workflow) = write_wake_only_sleep_workflow(|json| {
            json["phases"][0]["periodic_sleep"]["schedule"]["clock"] =
                serde_json::json!("model_tokens");
        });
        let error = validate_model_wake_plan(&sleep_memory_test_model(), &workflow)
            .unwrap_err()
            .to_string();
        assert!(error.contains("optimizer_steps"), "{error}");
        assert!(error.contains("multiple memory boundaries"), "{error}");
    }

    #[test]
    fn memory_training_requires_identical_periodic_configs_in_every_phase() {
        let (_directory, workflow) = write_wake_only_sleep_workflow(|json| {
            let mut second = json["phases"][0].clone();
            second["name"] = serde_json::json!("wake-two");
            second["data"] = serde_json::json!("corpus/wake-two.jsonl.zst");
            second["periodic_sleep"]["receiver_learning_rate"] = serde_json::json!(0.0002);
            json["phases"].as_array_mut().unwrap().push(second);
        });
        let error = validate_model_wake_plan(&sleep_memory_test_model(), &workflow)
            .unwrap_err()
            .to_string();
        assert!(error.contains("identical periodic_sleep"), "{error}");
    }

    #[test]
    fn invalid_numeric_training_arguments_fail_before_loading_files() {
        type Invalidate = fn(&mut TrainArgs);
        let cases: [(&str, Invalidate); 5] = [
            ("lr", |args| args.lr = f64::NAN),
            ("weight_decay", |args| args.weight_decay = -0.1),
            ("grad_clip", |args| args.grad_clip = f32::INFINITY),
            ("gpu_metrics_interval_ms", |args| {
                args.gpu_metrics_interval_ms = 99
            }),
            ("physical GPU selector", |args| {
                args.gpu_physical_device = "0,1".into()
            }),
        ];
        for (field, invalidate) in cases {
            let mut args = valid_train_args();
            invalidate(&mut args);
            let err = validate_train_args(&args).unwrap_err().to_string();
            assert!(err.contains(field), "{field}: {err}");
        }
    }

    #[test]
    fn training_cli_parses_explicit_physical_gpu_telemetry() {
        let cli = Cli::try_parse_from([
            "summa-train",
            "train",
            "--config",
            "model.mal",
            "--tokenizer",
            "tokenizer.json",
            "--workflow",
            "workflow.json",
            "--gpu-metrics-interval-ms",
            "2500",
            "--gpu-physical-device",
            "GPU-1234-abcd",
        ])
        .unwrap();
        let Command::Train(args) = cli.command else {
            panic!("wrong parsed command");
        };
        assert_eq!(args.gpu_metrics_interval_ms, 2500);
        assert_eq!(args.gpu_physical_device, "GPU-1234-abcd");
        validate_train_args(&args).unwrap();
    }

    #[test]
    fn gpu_telemetry_configuration_is_part_of_resume_identity() {
        let workflow = ResolvedWakePlan {
            version: 2,
            phases: Vec::new(),
        };
        let model = small_hybrid();
        let signature = |args: &TrainArgs| {
            let encoded = serde_json::to_vec(&RunSignature {
                version: TRAINING_STATE_VERSION,
                workflow: &workflow,
                model: &model,
                initial_checkpoint: None,
                tokenizer: format!("sha256:{}", "1".repeat(64)),
                data_manifests: &[],
                seed: args.seed,
                learning_rate: args.lr,
                weight_decay: args.weight_decay,
                gradient_clip: args.grad_clip,
                warmup_steps: args.warmup_steps,
                schedule: args.schedule,
                muon_learning_rate_scale: MUON_LR_SCALE,
                gpu_metrics_interval_ms: args.gpu_metrics_interval_ms,
                gpu_physical_device: &args.gpu_physical_device,
            })
            .unwrap();
            format!("sha256:{:x}", Sha256::digest(encoded))
        };
        let baseline = valid_train_args();
        let mut interval = valid_train_args();
        interval.gpu_metrics_interval_ms = 1000;
        let mut device = valid_train_args();
        device.gpu_physical_device = "GPU-1234".into();
        assert_ne!(signature(&baseline), signature(&interval));
        assert_ne!(signature(&baseline), signature(&device));
    }

    #[test]
    fn queued_and_resumed_device_samples_keep_exact_and_monotonic_timestamps() {
        fn sample(timestamp: u64) -> summa_train::device_sampler::DeviceSample {
            summa_train::device_sampler::DeviceSample {
                collected_at_unix_ms: timestamp,
                metrics: summa_train::metrics::DeviceUtilizationMetrics {
                    sampled_at_unix_ms: timestamp,
                    device_index: 0,
                    sample_window_seconds: 1.0,
                    gpu_utilization_percent: 75.0,
                    sm_active_percent: None,
                    tensor_core_active_percent: None,
                    memory_bandwidth_percent: None,
                    memory_used_bytes: 1,
                    memory_total_bytes: 2,
                    power_watts: Some(100.0),
                    temperature_celsius: Some(50.0),
                },
            }
        }

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let context = MetricContext {
            global_step: 4,
            phase: MetricPhase {
                index: 0,
                name: "pretrain".into(),
                kind: MetricPhaseKind::Pretrain,
            },
            checkpoint_hash: None,
        };
        let mut writer = MetricWriter::create(&path, "telemetry-run").unwrap();
        emit_device_sampler_drain(
            DeviceSamplerDrain {
                samples: vec![sample(1000)],
                diagnostics: Vec::new(),
            },
            &mut writer,
            &context,
        )
        .unwrap();
        emit_device_sampler_drain(
            DeviceSamplerDrain {
                samples: vec![sample(1200), sample(1100)],
                diagnostics: Vec::new(),
            },
            &mut writer,
            &context,
        )
        .unwrap();
        writer.sync_all().unwrap();
        drop(writer);

        let mut writer =
            MetricWriter::resume_from_checkpoint(&path, "telemetry-run", 3, 4).unwrap();
        // Simulate a wall-clock correction after resume. The payload keeps
        // the exact collection time while the record stays monotonic.
        emit_device_sampler_drain(
            DeviceSamplerDrain {
                samples: vec![sample(950)],
                diagnostics: Vec::new(),
            },
            &mut writer,
            &context,
        )
        .unwrap();
        writer.sync_all().unwrap();

        let records = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        let emitted = records
            .iter()
            .map(|record| record["emitted_at_unix_ms"].as_u64().unwrap())
            .collect::<Vec<_>>();
        let sampled = records
            .iter()
            .map(|record| {
                record["event"]["values"]["sampled_at_unix_ms"]
                    .as_u64()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(emitted, [1000, 1100, 1200, 1200]);
        assert_eq!(sampled, [1000, 1100, 1200, 950]);
    }

    #[test]
    fn stable_hash_helpers_share_the_resume_signature_contract() {
        assert_eq!(fnv1a64(b"hello"), 0xa430_d846_80aa_bd0b);
        assert_eq!(stable_cache_id("hello"), "a430d84680aabd0b");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact");
        fs::write(&path, b"hello").unwrap();
        assert_eq!(
            file_sha256(&path).unwrap(),
            "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );

        let cache_root = dir.path().join("cache");
        let source = Path::new("corpus.jsonl.zst");
        let first =
            token_cache_path(&cache_root, "sha256:data", source, "sha256:tokenizer").unwrap();
        assert_eq!(
            first,
            token_cache_path(&cache_root, "sha256:data", source, "sha256:tokenizer").unwrap()
        );
        assert_ne!(
            first,
            token_cache_path(&cache_root, "sha256:other-data", source, "sha256:tokenizer").unwrap()
        );
        assert_ne!(
            first,
            token_cache_path(&cache_root, "sha256:data", source, "sha256:other-tokenizer").unwrap()
        );
        assert_ne!(
            first,
            token_cache_path(
                &cache_root,
                "sha256:data",
                Path::new("corpus.txt.zst"),
                "sha256:tokenizer"
            )
            .unwrap()
        );
        assert_eq!(
            first.extension().and_then(|value| value.to_str()),
            Some("tokens")
        );
    }

    #[test]
    fn checkpoint_metric_verification_rejects_a_swapped_run() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metrics.jsonl");
        let mut writer = MetricWriter::create(&path, "different-run").unwrap();
        writer
            .append_at(
                MetricContext {
                    global_step: 7,
                    phase: MetricPhase {
                        index: 0,
                        name: "pretrain".into(),
                        kind: MetricPhaseKind::Pretrain,
                    },
                    checkpoint_hash: None,
                },
                MetricEvent::Throughput(ThroughputMetrics {
                    optimizer_steps: 1,
                    compute_tokens: 8,
                    supervised_tokens: 8,
                    examples: 1,
                    elapsed_seconds: 1.0,
                    tokens_per_second: 8.0,
                    examples_per_second: 1.0,
                    input_wait_seconds: 0.0,
                    host_to_device_seconds: 0.0,
                    gpu_busy_seconds: 1.0,
                }),
                1,
            )
            .unwrap();
        writer.sync_all().unwrap();
        drop(writer);

        let workflow_signature = format!("sha256:{}", "0".repeat(64));
        let verified = checkpoint::VerifiedCheckpoint {
            version: 1,
            generation: format!("sha256-{}", "1".repeat(64)),
            manifest_sha256: "1".repeat(64),
            global_step: 7,
            metric_records: 1,
            workflow_signature,
        };
        let error = validate_checkpoint_metrics(&verified, &path, false).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("run_id"), "{error}");
        let error = validate_checkpoint_metrics(&verified, &path, true).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("run_id"), "{error}");
    }

    #[test]
    fn quantization_distillation_reuses_student_forward_and_keeps_teacher_off_tape() {
        let config = small_hybrid();
        let device = summa_llm::default_device().autodiff();
        device.seed(11);
        let teacher = Transformer::new(&config, &device).unwrap().valid();
        device.seed(29);
        let student = Transformer::new(&config, &device).unwrap();
        let input_ids = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
        let targets = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], &device);
        let positions = Tensor::<1, Int>::from_data([2], &device);
        let batch = make_batch(
            &[TrainingSample::Supervised {
                tokens: vec![1, 2, 3, 4, 5],
                loss_positions: vec![2],
                truncated_tokens: 0,
            }],
            4,
            &device,
        )
        .unwrap();
        let objective = TaskConfig::Summarization {
            instruction: "summarize".into(),
        };
        let frozen_probe =
            teacher.forward_selected_logits(input_ids.clone().inner(), positions.clone().inner());
        assert!(!frozen_probe.device().is_autodiff());
        assert!(!frozen_probe.is_require_grad());
        let teacher_logits = quantization_teacher_logits(&teacher, &batch, &objective).unwrap();
        assert!(teacher_logits.device().is_autodiff());
        assert!(!teacher_logits.is_require_grad());

        let ObjectiveForward {
            loss: task_loss,
            router_loss,
            captured_logits,
            ..
        } = objective_loss(&student, batch, &objective, true).unwrap();
        assert!(router_loss.is_none());
        let student_logits = captured_logits.unwrap();
        let expected_student_logits =
            student.forward_selected_logits(input_ids.clone(), positions.clone());
        let actual_logits = student_logits
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let expected_logits = expected_student_logits
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual_logits, expected_logits);

        let expected_task_loss =
            scalar_value(student.forward_masked_loss(input_ids, targets, positions)).unwrap();
        let actual_task_loss = scalar_value(task_loss.clone()).unwrap();
        assert!((actual_task_loss - expected_task_loss).abs() < 1e-6);

        let loss =
            forward_kl_distillation_tensor(teacher_logits.clone(), student_logits, 1.0, true)
                .unwrap();
        let expected =
            forward_kl_distillation_tensor(teacher_logits, expected_student_logits, 1.0, true)
                .unwrap();
        let actual_value = scalar_value(loss.clone()).unwrap();
        let expected_value = scalar_value(expected).unwrap();
        assert!((actual_value - expected_value).abs() < 1e-6);

        let mut gradients = (task_loss + loss).backward();
        let student_gradients = GradientsParams::from_module(&mut gradients, &student);
        assert!(!student_gradients.is_empty());
    }

    #[test]
    fn retrieval_qat_reuses_student_embeddings_for_task_and_distillation() {
        let config = small_hybrid();
        let device = summa_llm::default_device().autodiff();
        device.seed(47);
        let teacher = Transformer::new(&config, &device).unwrap().valid();
        device.seed(53);
        let student = Transformer::new(&config, &device).unwrap();
        let encoded = |tokens| data::EncodedText {
            tokens,
            end_position: 1,
        };
        let batch = make_batch(
            &[TrainingSample::Retrieval {
                query: encoded(vec![1, 2, 0, 0]),
                documents: vec![encoded(vec![3, 4, 0, 0]), encoded(vec![5, 6, 0, 0])],
                truncated_tokens: 0,
            }],
            4,
            &device,
        )
        .unwrap();
        let objective = TaskConfig::RetrievalRepresentation {
            temperature: 0.5,
            layer: None,
            query_prefix: "query".into(),
            document_prefix: "document".into(),
        };
        let teacher_logits = quantization_teacher_logits(&teacher, &batch, &objective).unwrap();
        assert!(teacher_logits.device().is_autodiff());
        assert!(!teacher_logits.is_require_grad());
        let ObjectiveForward {
            loss: task_loss,
            router_loss,
            captured_logits,
            ..
        } = objective_loss(&student, batch, &objective, true).unwrap();
        assert!(router_loss.is_none());
        let student_logits = captured_logits.unwrap();

        let query_ids = Tensor::<2, Int>::from_data([[1, 2, 0, 0]], &device);
        let query_positions = Tensor::<1, Int>::from_data([1], &device);
        let document_ids = Tensor::<2, Int>::from_data([[3, 4, 0, 0], [5, 6, 0, 0]], &device);
        let document_positions = Tensor::<1, Int>::from_data([1, 5], &device);
        let expected_logits = student
            .forward_embeddings(query_ids, query_positions, None)
            .matmul(
                student
                    .forward_embeddings(document_ids, document_positions, None)
                    .transpose(),
            );
        let maximum: f32 = (student_logits.clone() - expected_logits)
            .abs()
            .max()
            .into_scalar();
        assert!(maximum < 1e-6, "retrieval QAT logits differ by {maximum}");

        let distillation_loss =
            forward_kl_distillation_tensor(teacher_logits, student_logits, 1.0, true).unwrap();
        let mut gradients = (task_loss + distillation_loss).backward();
        let student_gradients = GradientsParams::from_module(&mut gradients, &student);
        assert!(!student_gradients.is_empty());
    }

    #[test]
    fn clipping_uses_one_norm_and_scale_for_wake_and_tier_gradients() {
        let config = small_hybrid();
        let device = summa_llm::default_device().autodiff();
        device.seed(37);
        let model = Transformer::new(&config, &device).unwrap();
        let muon_ids = model.muon_parameter_ids();
        let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
        let target = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], &device);
        let mut gradients = model.forward_loss(input, target).backward();
        let mut muon = GradientsParams::from_params(&mut gradients, &model, &muon_ids);
        let tier = GradientsParams::from_module(&mut gradients, &model);
        let mut wake = GradientsParams::new();
        let mut tiers = vec![tier];
        let expected = scalar_value(
            sum_optional_tensors(
                squared_gradient_norm(&model, &muon),
                squared_gradient_norm(&model, &tiers[0]),
            )
            .unwrap()
            .sqrt(),
        )
        .unwrap();
        let max_norm = expected * 0.25;
        let observed =
            gradient_norm_and_clip(&model, &mut muon, &mut wake, &mut tiers, max_norm).unwrap();
        assert!((observed - expected).abs() <= expected * 1e-5);
        let clipped = scalar_value(
            sum_optional_tensors(
                squared_gradient_norm(&model, &muon),
                squared_gradient_norm(&model, &tiers[0]),
            )
            .unwrap()
            .sqrt(),
        )
        .unwrap();
        assert!((clipped - max_norm).abs() <= max_norm * 1e-4);
    }

    #[test]
    fn training_decreases_loss_and_checkpoint_roundtrips() {
        let mut config = small_hybrid();
        for block in config.pattern.as_mut().unwrap() {
            // Deliberately zero, and do not "restore" this to a live rate. The
            // dropout masks are drawn from the backend's *global* device RNG,
            // which every concurrently running test that builds a model also
            // draws from. With dropout enabled the in-process model and the
            // checkpoint-restored model see different masks on their shared
            // training step, and the round-trip comparison below fails
            // nondeterministically. Nothing here asserts dropout behaviour, and
            // the final comparison runs under `.valid()` with dropout disabled.
            block.dropout = 0.0;
            block.attention.dropout = 0.0;
            block.ffn.dropout = 0.0;
        }
        let device = summa_llm::default_device().autodiff();
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
        for microbatch in 0..20 {
            device.seed(model_microbatch_seed(41, microbatch));
            let (input, target) = batch();
            let loss = model.forward_loss(input, target);
            losses.push(scalar_value(loss.clone()).unwrap());
            let mut grads = loss.backward();
            let mut muon_grads =
                GradientsParams::from_params(&mut grads, &model, &muon_parameter_ids);
            let mut adamw_grads = GradientsParams::from_module(&mut grads, &model);
            if losses.len() == 1 {
                let layer_norms =
                    layer_gradient_norms(&model, &muon_grads, &adamw_grads, &[]).unwrap();
                assert_eq!(layer_norms.len(), config.num_layers);
                assert!(layer_norms.into_iter().all(f32::is_finite));
            }
            let norm =
                gradient_norm_and_clip(&model, &mut muon_grads, &mut adamw_grads, &mut [], 1.0)
                    .unwrap();
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
            global_step: 20,
            phase: 1,
            phase_id: "sft".into(),
            phase_kind: "sft".into(),
            epoch: 2,
            records_in_phase: 640,
            steps_in_phase: 10,
            tokens_seen: 12_800,
            metric_records: 1,
            workflow_signature: format!("sha256:{}", "0".repeat(64)),
            data_manifest_hash: Some(format!("sha256:{}", "1".repeat(64))),
            parameter_ids: parameter_ids(&model),
            optimizer_states: vec![OptimizerStateRef {
                scope: "wake".into(),
                adamw: "adamw-state.bpk".into(),
                muon: "muon-state.bpk".into(),
                gradient_accumulator: None,
                update_clock: 20,
            }],
            memory_update: TrainingMemoryUpdateState::Ordinary,
            sleep: None,
            artifacts: Vec::new(),
            evaluator_hashes: Vec::new(),
            rng_streams: vec![
                RngStreamState {
                    name: DATA_RNG_STREAM.into(),
                    seed: 0,
                    counter: 640,
                },
                RngStreamState {
                    name: MODEL_RNG_STREAM.into(),
                    seed: 41,
                    counter: 20,
                },
            ],
            wake_context_buffer: Vec::new(),
            quantization: None,
        };
        let mut metrics =
            MetricWriter::create(dir.path().join("metrics.jsonl"), "test-run").unwrap();
        metrics
            .append_at(
                MetricContext {
                    global_step: 20,
                    phase: MetricPhase {
                        index: 1,
                        name: "sft".into(),
                        kind: MetricPhaseKind::Sft,
                    },
                    checkpoint_hash: None,
                },
                MetricEvent::Throughput(ThroughputMetrics {
                    optimizer_steps: 20,
                    compute_tokens: 12_800,
                    supervised_tokens: 12_800,
                    examples: 640,
                    elapsed_seconds: 2.0,
                    tokens_per_second: 6_400.0,
                    examples_per_second: 320.0,
                    input_wait_seconds: 0.1,
                    host_to_device_seconds: 0.05,
                    gpu_busy_seconds: 1.8,
                }),
                1,
            )
            .unwrap();
        let publication = save_training_checkpoint_with_evidence(
            &model,
            &adamw_optimizer,
            &muon_optimizer,
            &state,
            &mut metrics,
            dir.path(),
        )
        .unwrap();
        assert!(publication.checkpoint_manifest.is_file());
        let generation = publication.checkpoint_manifest.parent().unwrap();
        let accounting: summa_train::benchmark::TrainingAccounting = serde_json::from_slice(
            &fs::read(generation.join(summa_train::benchmark::TRAINING_ACCOUNTING_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(accounting.parameters, model.num_parameters() as u64);
        assert_eq!(accounting.routed_active_parameters, accounting.parameters);
        assert_eq!(
            accounting.weights_bytes,
            fs::metadata(generation.join("weights.safetensors"))
                .unwrap()
                .len()
        );
        assert!((accounting.training_gpu_hours - 2.0 / 3600.0).abs() < 1e-15);

        let mut mismatched_qat_state = state.clone();
        mismatched_qat_state.quantization = Some(QuantizationTrainingState {
            format: "binary_g128".into(),
            fake_quant_active: false,
            calibration_step: state.global_step as u64,
            manifest: Some("candidate.json".into()),
            candidate_weights_sha256: Some(format!("sha256:{}", "0".repeat(64))),
            teacher_hash: None,
        });
        let error = save_training_checkpoint_with_evidence(
            &model,
            &adamw_optimizer,
            &muon_optimizer,
            &mismatched_qat_state,
            &mut metrics,
            dir.path(),
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("candidate weights differ"),
            "{error:#}"
        );

        let mut resumed = Transformer::new(&config, &device).unwrap();
        let mut resumed_ids = resumed.muon_parameter_ids();
        let mut resumed_muon = BatchedMuon::new(resumed_ids.clone());
        let resumed_adamw = AdamWConfig::new()
            .with_beta_2(0.95)
            .with_epsilon(1e-8)
            .with_weight_decay(0.0)
            .init();
        let (mut resumed_adamw, resumed_state, resumed_weights_sha256) = load_training_state(
            &mut resumed,
            resumed_adamw,
            &mut resumed_muon,
            dir.path(),
            &device,
        )
        .unwrap();
        assert_eq!(
            resumed_weights_sha256,
            format!("sha256:{}", accounting.weights_sha256)
        );
        assert_eq!(resumed_state.global_step, state.global_step);
        assert_eq!(resumed_state.phase, state.phase);
        assert_eq!(resumed_state.epoch, state.epoch);
        assert_eq!(resumed_state.records_in_phase, state.records_in_phase);
        assert_eq!(resumed_state.steps_in_phase, state.steps_in_phase);
        assert_eq!(resumed_state.tokens_seen, state.tokens_seen);
        assert_eq!(resumed_state.workflow_signature, state.workflow_signature);
        let restored_ids = resumed.muon_parameter_ids();
        assert_ne!(resumed_ids, restored_ids);
        assert_eq!(muon_parameter_ids, restored_ids);
        resumed_ids = restored_ids;

        let advance = |mut model: Transformer,
                       muon: &mut BatchedMuon,
                       adamw: &mut AdamWOptimizer,
                       muon_ids: &[ParamId]| {
            device.seed(model_microbatch_seed(41, 20));
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

        #[cfg(unix)]
        {
            let current_pointer = dir.path().join("current.json");
            let pointer_a = fs::read(&current_pointer).unwrap();
            let generation_a = publication
                .checkpoint_manifest
                .parent()
                .unwrap()
                .to_path_buf();
            let weights_a = fs::read(generation_a.join("weights.safetensors")).unwrap();
            let muon_a = fs::read(generation_a.join("muon-state.bpk")).unwrap();
            let adamw_a = fs::read(generation_a.join("adamw-state.bpk")).unwrap();

            metrics
                .append_at(
                    MetricContext {
                        global_step: 21,
                        phase: MetricPhase {
                            index: 1,
                            name: "sft".into(),
                            kind: MetricPhaseKind::Sft,
                        },
                        checkpoint_hash: None,
                    },
                    MetricEvent::Throughput(ThroughputMetrics {
                        optimizer_steps: 1,
                        compute_tokens: 640,
                        supervised_tokens: 640,
                        examples: 32,
                        elapsed_seconds: 0.1,
                        tokens_per_second: 6_400.0,
                        examples_per_second: 320.0,
                        input_wait_seconds: 0.005,
                        host_to_device_seconds: 0.002,
                        gpu_busy_seconds: 0.09,
                    }),
                    2,
                )
                .unwrap();
            let mut state_b = state.clone();
            state_b.global_step = 21;
            state_b.records_in_phase = 672;
            state_b.steps_in_phase = 11;
            state_b.tokens_seen = 13_440;
            state_b.metric_records = 2;
            state_b.optimizer_states[0].update_clock = 21;
            for stream in &mut state_b.rng_streams {
                match stream.name.as_str() {
                    DATA_RNG_STREAM => stream.counter = 672,
                    MODEL_RNG_STREAM => stream.counter = 21,
                    _ => {}
                }
            }
            let publication_b = save_training_checkpoint_with_evidence(
                &model,
                &adamw_optimizer,
                &muon_optimizer,
                &state_b,
                &mut metrics,
                dir.path(),
            )
            .unwrap();
            let generation_b = publication_b
                .checkpoint_manifest
                .parent()
                .unwrap()
                .to_path_buf();
            let weights_b = fs::read(generation_b.join("weights.safetensors")).unwrap();
            let muon_b = fs::read(generation_b.join("muon-state.bpk")).unwrap();
            let adamw_b = fs::read(generation_b.join("adamw-state.bpk")).unwrap();
            assert_ne!(
                weights_a, weights_b,
                "ABA generations need distinct weights"
            );
            assert_ne!(muon_a, muon_b, "ABA generations need distinct Muon state");
            assert_ne!(
                adamw_a, adamw_b,
                "ABA generations need distinct AdamW state"
            );

            // Point at A, authenticate it, replace its pathname with the valid
            // same-topology generation B, then restore A before the final
            // pointer check. Path-based loaders accept this ABA sequence; the
            // retained generation handle must still load only A's bytes.
            fs::write(&current_pointer, &pointer_a).unwrap();
            let mut aba_model = Transformer::new(&config, &device).unwrap();
            let mut aba_muon = BatchedMuon::new(aba_model.muon_parameter_ids());
            let aba_adamw = AdamWConfig::new()
                .with_beta_2(0.95)
                .with_epsilon(1e-8)
                .with_weight_decay(0.0)
                .init();
            let mut swap = GenerationSwapGuard::new(generation_a.clone(), generation_b);
            let (aba_adamw, aba_state, _) = checkpoint::load_training_state_with_hook(
                &mut aba_model,
                aba_adamw,
                &mut aba_muon,
                dir.path(),
                &device,
                |stage, _| match stage {
                    checkpoint::ResumeLoadStage::AfterInitialVerify => swap.swap(),
                    checkpoint::ResumeLoadStage::AfterStagedLoad => swap.restore(),
                },
            )
            .unwrap();
            assert_eq!(aba_state.global_step, state.global_step);

            let aba_artifacts = tempfile::tempdir().unwrap();
            let loaded_weights = aba_artifacts.path().join("weights.safetensors");
            let loaded_muon = aba_artifacts.path().join("muon-state.bpk");
            save_safetensors(&aba_model, &loaded_weights).unwrap();
            aba_muon.save(&loaded_muon).unwrap();
            let loaded_adamw =
                summa_train::optimizer_artifact::canonical_module_optimizer_bytes(&aba_adamw)
                    .unwrap();
            assert_eq!(fs::read(loaded_weights).unwrap(), weights_a);
            assert_eq!(fs::read(loaded_muon).unwrap(), muon_a);
            assert_eq!(&*loaded_adamw, adamw_a.as_slice());

            let malformed_generation =
                checkpoint::rewrite_current_generation_for_test(dir.path(), |staging| {
                    let path = staging.join("adamw-state.bpk");
                    let reader = burn_pack::Reader::from_file(&path)?;
                    let metadata = reader.metadata().clone();
                    let scalars = reader.scalars().clone();
                    let mut tensors = reader.into_tensors()?;
                    ensure!(
                        tensors.len() > 1,
                        "one-step AdamW fixture has too little state"
                    );
                    tensors.remove(0);
                    let mut writer = burn_pack::Writer::new(tensors);
                    for (key, value) in scalars {
                        writer = writer.with_scalar(&key, value);
                    }
                    for (key, value) in metadata {
                        writer = writer.with_metadata(&key, &value);
                    }
                    let replacement = staging.join("adamw-state-malformed.tmp");
                    writer.write_to_file(&replacement)?;
                    fs::rename(replacement, path)?;
                    Ok(())
                })
                .unwrap();
            assert_ne!(malformed_generation, generation_a);
            verify_checkpoint_root(dir.path()).unwrap();

            let mut rejected_model = aba_model.clone();
            let mut rejected_muon = aba_muon.clone();
            let rejected_artifacts = tempfile::tempdir().unwrap();
            let rejected_model_before = rejected_artifacts.path().join("model-before.safetensors");
            let rejected_model_after = rejected_artifacts.path().join("model-after.safetensors");
            let rejected_muon_before = rejected_artifacts.path().join("muon-before.bpk");
            let rejected_muon_after = rejected_artifacts.path().join("muon-after.bpk");
            save_safetensors(&rejected_model, &rejected_model_before).unwrap();
            rejected_muon.save(&rejected_muon_before).unwrap();
            let rejected_adamw = AdamWConfig::new()
                .with_beta_2(0.95)
                .with_epsilon(1e-8)
                .with_weight_decay(0.0)
                .init();
            let error = load_training_state(
                &mut rejected_model,
                rejected_adamw,
                &mut rejected_muon,
                dir.path(),
                &device,
            )
            .err()
            .expect("AdamW state with a missing moment tensor must be rejected");
            assert!(
                format!("{error:#}").contains("lossless reconstruction"),
                "{error:#}"
            );
            save_safetensors(&rejected_model, &rejected_model_after).unwrap();
            rejected_muon.save(&rejected_muon_after).unwrap();
            assert_eq!(
                fs::read(rejected_model_before).unwrap(),
                fs::read(rejected_model_after).unwrap()
            );
            assert_eq!(
                fs::read(rejected_muon_before).unwrap(),
                fs::read(rejected_muon_after).unwrap()
            );
            fs::write(&current_pointer, &pointer_a).unwrap();
        }

        let snapshots = tempfile::tempdir().unwrap();
        let mut raced_model = resumed.clone();
        let raced_model_ids = parameter_ids(&raced_model);
        let mut raced_muon = resumed_muon.clone();
        let model_before = snapshots.path().join("model-before.safetensors");
        let model_after = snapshots.path().join("model-after.safetensors");
        let muon_before = snapshots.path().join("muon-before.bpk");
        let muon_after = snapshots.path().join("muon-after.bpk");
        save_safetensors(&raced_model, &model_before).unwrap();
        raced_muon.save(&muon_before).unwrap();

        let raced_adamw = AdamWConfig::new()
            .with_beta_2(0.95)
            .with_epsilon(1e-8)
            .with_weight_decay(0.0)
            .init();
        let error = checkpoint::load_training_state_with_hook(
            &mut raced_model,
            raced_adamw,
            &mut raced_muon,
            dir.path(),
            &device,
            |stage, generation| {
                if stage == checkpoint::ResumeLoadStage::AfterStagedLoad {
                    fs::write(
                        generation.join("weights.safetensors"),
                        b"changed after load",
                    )?;
                }
                Ok(())
            },
        )
        .err()
        .expect("checkpoint mutation should fail staged resume loading");
        assert!(
            format!("{error:#}").contains("contents do not match"),
            "{error:#}"
        );

        save_safetensors(&raced_model, &model_after).unwrap();
        raced_muon.save(&muon_after).unwrap();
        assert_eq!(parameter_ids(&raced_model), raced_model_ids);
        assert_eq!(
            fs::read(model_after).unwrap(),
            fs::read(model_before).unwrap()
        );
        assert_eq!(
            fs::read(muon_after).unwrap(),
            fs::read(muon_before).unwrap()
        );
    }
}
