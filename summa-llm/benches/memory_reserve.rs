//! Paired acceptance benchmark for sleep-memory wake overhead.
//!
//! This benchmark answers two different performance questions without
//! conflating them:
//!
//! 1. `static_backbone_vs_dormant_memory` compares the unchanged static model
//!    with its MAL-validated sleep-capable counterpart. This measures the
//!    total wake cost of opting into the memory hierarchy.
//! 2. `one_active_vs_n_active` compares two instances of the *same* memory
//!    topology. Both have identical stored and routed-active parameter counts;
//!    only their checkpointed active-slot masks differ. This is the direct
//!    acceptance check that active compute does not grow across sleep cycles.
//!
//! Each result contains raw, interleaved A/B timings for at least three seeds,
//! forward and forward/backward modes, per-seed summaries, and paired ratios.
//! The default 5% gate is reported but only enforced with `--enforce`.
//!
//! The embedded compact pair is useful for local smoke tests. Official CUDA
//! evidence must use the production pair and explicitly require CUDA:
//!
//! ```text
//! cargo bench -p summa-llm --bench memory_reserve --features training-fusion -- \
//!   --model "$PWD/summa-mal/well-known/retriever_300m_moe_sleep.mal" \
//!   --baseline-model "$PWD/summa-mal/well-known/retriever_300m_moe.mal" \
//!   --tokens 8192 --tier 0 --max-active 2 --require-cuda --enforce \
//!   --output "$PWD/memory-reserve-cuda.json"
//! ```
//!
//! A non-CUDA run is deliberately labelled `local_non_cuda_smoke_only` and
//! must not be presented as CUDA acceptance evidence.

#[path = "memory_reserve/cli.rs"]
mod cli;
#[path = "memory_reserve/fairness.rs"]
mod fairness;
#[path = "memory_reserve/stats.rs"]
mod stats;

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use serde::Serialize;
use summa_llm::{
    ModelDef, Transformer, WakeParameterAccounting, default_device, parse_mal, parse_mal_file,
};

use cli::{Args, backend_name, parse_args};
use stats::{ExecutionOrder, PairedSample, PairedSummary, PairedTrial};

const MODEL_INITIALIZATION_SEED: u64 = 0x4845_524d_4553;

const COMPACT_STATIC_MODEL: &str = r#"
ffn base { hidden_dim: 16 activation: swiglu bias: false }
model static-memory-reserve-bench {
    vocab_size: 128 max_seq_len: 1 hidden_size: 128 num_layers: 1
    block: { attention: { num_heads: 4 } ffn: base }
    embeddings { tie_weights: true }
}
"#;

const COMPACT_MEMORY_MODEL: &str = r#"
ffn base { hidden_dim: 16 activation: swiglu bias: false }
memory reserve {
    tier fast {
        ffn: base
        reserve_experts { capacity: 4 rank: 32 top_k: 1 }
    }
}
model sleep-memory-reserve-bench {
    vocab_size: 128 max_seq_len: 1 hidden_size: 128 num_layers: 1
    block: { attention: { num_heads: 4 } memory: reserve }
    embeddings { tie_weights: true }
}
"#;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
struct ParameterAccounting {
    stored: u64,
    routed_active: u64,
}

impl From<WakeParameterAccounting> for ParameterAccounting {
    fn from(value: WakeParameterAccounting) -> Self {
        Self {
            stored: value.stored_parameters,
            routed_active: value.routed_active_parameters,
        }
    }
}

#[derive(Debug, Serialize)]
struct FairnessEvidence {
    static_backbone_matched: bool,
    layers_checked: usize,
    selected_tier: usize,
    selected_tier_capacity: usize,
    memory_layers: usize,
    later_memory_tiers_are_residual_noops: bool,
    static_accounting: ParameterAccounting,
    memory_accounting: ParameterAccounting,
    added_stored_parameters: u64,
    added_routed_active_parameters: u64,
}

#[derive(Clone, Debug, Serialize)]
struct StateDescription {
    model: String,
    kind: &'static str,
    active_slots_per_memory_layer: usize,
    accounting: ParameterAccounting,
}

#[derive(Debug, Serialize)]
struct Acceptance {
    statistic: &'static str,
    max_overhead_percent: f64,
    observed_overhead_percent: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ModeReport {
    mode: &'static str,
    trials: Vec<PairedTrial>,
    summary: PairedSummary,
    reference_tokens_per_second: f64,
    candidate_tokens_per_second: f64,
    acceptance: Acceptance,
}

#[derive(Debug, Serialize)]
struct ComparisonReport {
    comparison: &'static str,
    contract: &'static str,
    model_initialization_seed: u64,
    reference: StateDescription,
    candidate: StateDescription,
    routed_active_parameters_equal: bool,
    modes: Vec<ModeReport>,
}

#[derive(Debug, Serialize)]
struct Workload {
    logical_tokens: usize,
    batch_size: usize,
    sequence_length: usize,
    backward_objective: &'static str,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct BuildDescription {
    backend: &'static str,
    device: String,
    training_fusion: bool,
    measurement_scope: &'static str,
    cuda_acceptance_eligible: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    implementation: &'static str,
    static_model: String,
    memory_model: String,
    build: BuildDescription,
    workload: Workload,
    warmup_iterations_per_operation: usize,
    measured_iterations_per_seed: usize,
    paired_seeds: Vec<u64>,
    fairness: FairnessEvidence,
    comparisons: Vec<ComparisonReport>,
    gate_enforced: bool,
    all_gates_passed: bool,
}

#[derive(Clone, Copy)]
enum ModelKind {
    Static,
    Memory { active: usize },
}

struct ComparisonSpec<'a> {
    name: &'static str,
    contract: &'static str,
    reference_config: &'a ModelDef,
    reference_kind: ModelKind,
    candidate_config: &'a ModelDef,
    candidate_kind: ModelKind,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let (static_config, memory_config) = load_configs(&args)?;
    fairness::validate_matched_backbone(&static_config, &memory_config)?;

    let backend = backend_name();
    let cuda_acceptance_eligible = backend == "cuda" && cfg!(feature = "training-fusion");
    if args.require_cuda && !cuda_acceptance_eligible {
        bail!(
            "--require-cuda requested, but this binary uses backend={backend}, training_fusion={}; rebuild on Linux with --features training-fusion",
            cfg!(feature = "training-fusion")
        );
    }

    let device = default_device();
    let training_device = device.clone().autodiff();
    let (memory_layers, tier_capacity) = selected_tier_shape(&memory_config, args.tier)?;
    ensure!(
        args.max_active <= tier_capacity,
        "--max-active {} exceeds tier {} capacity {}",
        args.max_active,
        args.tier,
        tier_capacity
    );

    let mut specs = vec![ComparisonSpec {
        name: "static_backbone_vs_dormant_memory",
        contract: "total wake overhead of the MAL-validated sleep-capable topology",
        reference_config: &static_config,
        reference_kind: ModelKind::Static,
        candidate_config: &memory_config,
        candidate_kind: ModelKind::Memory { active: 0 },
    }];
    if args.max_active > 1 {
        specs.push(ComparisonSpec {
            name: "one_active_vs_n_active",
            contract: "wake compute must not grow as additional stored reserve slots activate",
            reference_config: &memory_config,
            reference_kind: ModelKind::Memory { active: 1 },
            candidate_config: &memory_config,
            candidate_kind: ModelKind::Memory {
                active: args.max_active,
            },
        });
    }

    let mut comparisons = Vec::with_capacity(specs.len());
    for spec in specs {
        comparisons.push(run_comparison(spec, &args, &device, &training_device)?);
    }
    let static_accounting = comparisons[0].reference.accounting;
    let memory_accounting = comparisons[0].candidate.accounting;
    let fairness = FairnessEvidence {
        static_backbone_matched: true,
        layers_checked: memory_config.num_layers,
        selected_tier: args.tier,
        selected_tier_capacity: tier_capacity,
        memory_layers,
        later_memory_tiers_are_residual_noops: true,
        static_accounting,
        memory_accounting,
        added_stored_parameters: memory_accounting
            .stored
            .checked_sub(static_accounting.stored)
            .context("memory model stores fewer parameters than its matched static backbone")?,
        added_routed_active_parameters: memory_accounting
            .routed_active
            .checked_sub(static_accounting.routed_active)
            .context("memory model routes fewer parameters than its matched static backbone")?,
    };
    let all_gates_passed = comparisons
        .iter()
        .flat_map(|comparison| &comparison.modes)
        .all(|mode| mode.acceptance.passed);

    let report = Report {
        schema_version: 2,
        implementation: "fixed_width_cached_reserve_router_paired_acceptance",
        static_model: static_config.name,
        memory_model: memory_config.name,
        build: BuildDescription {
            backend,
            device: format!("{device:?}"),
            training_fusion: cfg!(feature = "training-fusion"),
            measurement_scope: if cuda_acceptance_eligible {
                "cuda_single_device_acceptance"
            } else if backend == "cuda" {
                "cuda_without_training_fusion_smoke_only"
            } else {
                "local_non_cuda_smoke_only"
            },
            cuda_acceptance_eligible,
        },
        workload: Workload {
            logical_tokens: args.tokens,
            batch_size: args.tokens,
            sequence_length: 1,
            backward_objective: "deterministically weighted embedding mean plus configured router losses",
            note: "independent one-token rows isolate flattened FFN/reserve routing without quadratic attention",
        },
        warmup_iterations_per_operation: args.warmup,
        measured_iterations_per_seed: args.iterations,
        paired_seeds: args.seeds.clone(),
        fairness,
        comparisons,
        gate_enforced: args.enforce,
        all_gates_passed,
    };

    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");
    if let Some(path) = &args.output {
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    if args.enforce && !all_gates_passed {
        bail!(
            "memory-reserve overhead exceeded {:.3}% (see JSON report)",
            args.max_overhead_percent
        );
    }
    Ok(())
}

fn run_comparison(
    spec: ComparisonSpec<'_>,
    args: &Args,
    inference_device: &Device,
    training_device: &Device,
) -> Result<ComparisonReport> {
    let mut forward_trials = Vec::with_capacity(args.seeds.len());
    let mut backward_trials = Vec::with_capacity(args.seeds.len());
    let reference = instantiate(
        spec.reference_config,
        spec.reference_kind,
        args.tier,
        training_device,
        MODEL_INITIALIZATION_SEED,
    )?;
    let candidate = instantiate(
        spec.candidate_config,
        spec.candidate_kind,
        args.tier,
        training_device,
        MODEL_INITIALIZATION_SEED,
    )?;
    let reference_description =
        describe_state(spec.reference_config, spec.reference_kind, &reference)?;
    let candidate_description =
        describe_state(spec.candidate_config, spec.candidate_kind, &candidate)?;
    let reference_valid = reference.clone().valid();
    let candidate_valid = candidate.clone().valid();

    for (trial_index, &seed) in args.seeds.iter().enumerate() {
        let token_count = i64::try_from(args.tokens)
            .context("--tokens exceeds the backend's signed index range")?;
        let objective_elements = args
            .tokens
            .checked_mul(spec.memory_hidden_size())
            .context("benchmark objective tensor size overflows usize")?;
        let ids = deterministic_token_ids(args.tokens, spec.memory_vocab_size(), seed);
        let inference_input = Tensor::<2, Int>::from_data(
            TensorData::new(ids.clone(), [args.tokens, 1]),
            inference_device,
        );
        let inference_ends = Tensor::<1, Int>::arange(0..token_count, inference_device);
        let training_input =
            Tensor::<2, Int>::from_data(TensorData::new(ids, [args.tokens, 1]), training_device);
        let training_ends = Tensor::<1, Int>::arange(0..token_count, training_device);
        let training_weights = Tensor::<2>::from_data(
            TensorData::new(
                deterministic_objective_weights(objective_elements, seed ^ 0x6a09_e667_f3bc_c909),
                [args.tokens, spec.memory_hidden_size()],
            ),
            training_device,
        );

        let forward = measure_paired(
            args,
            inference_device,
            trial_index,
            || {
                let output = reference_valid.forward_embeddings(
                    inference_input.clone(),
                    inference_ends.clone(),
                    None,
                );
                std::hint::black_box(output);
            },
            || {
                let output = candidate_valid.forward_embeddings(
                    inference_input.clone(),
                    inference_ends.clone(),
                    None,
                );
                std::hint::black_box(output);
            },
        )?;
        forward_trials.push(PairedTrial::new(seed, forward)?);

        let backward = measure_paired(
            args,
            training_device,
            trial_index,
            || {
                backward_embeddings(
                    &reference,
                    training_input.clone(),
                    training_ends.clone(),
                    training_weights.clone(),
                )
            },
            || {
                backward_embeddings(
                    &candidate,
                    training_input.clone(),
                    training_ends.clone(),
                    training_weights.clone(),
                )
            },
        )?;
        backward_trials.push(PairedTrial::new(seed, backward)?);
        training_device.sync()?;
    }

    let routed_active_parameters_equal = reference_description.accounting.routed_active
        == candidate_description.accounting.routed_active;
    if spec.name == "one_active_vs_n_active" {
        ensure!(
            reference_description.accounting == candidate_description.accounting,
            "constant-active comparison changed parameter accounting: {:?} vs {:?}",
            reference_description.accounting,
            candidate_description.accounting
        );
    }

    Ok(ComparisonReport {
        comparison: spec.name,
        contract: spec.contract,
        model_initialization_seed: MODEL_INITIALIZATION_SEED,
        reference: reference_description,
        candidate: candidate_description,
        routed_active_parameters_equal,
        modes: vec![
            mode_report(
                "forward",
                forward_trials,
                args.tokens,
                args.max_overhead_percent,
            )?,
            mode_report(
                "forward_backward",
                backward_trials,
                args.tokens,
                args.max_overhead_percent,
            )?,
        ],
    })
}

impl ComparisonSpec<'_> {
    fn memory_vocab_size(&self) -> usize {
        match (self.reference_kind, self.candidate_kind) {
            (ModelKind::Memory { .. }, _) => self.reference_config.vocab_size,
            (_, ModelKind::Memory { .. }) => self.candidate_config.vocab_size,
            _ => unreachable!("every comparison contains the memory model"),
        }
    }

    fn memory_hidden_size(&self) -> usize {
        match (self.reference_kind, self.candidate_kind) {
            (ModelKind::Memory { .. }, _) => self.reference_config.hidden_size,
            (_, ModelKind::Memory { .. }) => self.candidate_config.hidden_size,
            _ => unreachable!("every comparison contains the memory model"),
        }
    }
}

fn mode_report(
    mode: &'static str,
    trials: Vec<PairedTrial>,
    tokens: usize,
    max_overhead_percent: f64,
) -> Result<ModeReport> {
    let summary = stats::summarize(&trials)?;
    let observed_overhead_percent = (summary.median_of_seed_median_ratios - 1.0) * 100.0;
    let acceptance = Acceptance {
        statistic: "median of per-seed median paired candidate/reference ratios",
        max_overhead_percent,
        observed_overhead_percent,
        passed: observed_overhead_percent <= max_overhead_percent,
    };
    Ok(ModeReport {
        mode,
        trials,
        reference_tokens_per_second: tokens as f64 / (summary.reference_median_ms / 1_000.0),
        candidate_tokens_per_second: tokens as f64 / (summary.candidate_median_ms / 1_000.0),
        summary,
        acceptance,
    })
}

fn backward_embeddings(
    model: &Transformer,
    input: Tensor<2, Int>,
    ends: Tensor<1, Int>,
    weights: Tensor<2>,
) {
    let (output, auxiliary) = model.forward_embeddings_with_router(input, ends, None);
    let mut loss = (output.cast(DType::F32) * weights).mean();
    if let Some(auxiliary) = auxiliary {
        loss = loss + auxiliary;
    }
    let gradients = loss.backward();
    std::hint::black_box(gradients);
}

fn measure_paired(
    args: &Args,
    device: &Device,
    trial_index: usize,
    mut reference: impl FnMut(),
    mut candidate: impl FnMut(),
) -> Result<Vec<PairedSample>> {
    for warmup in 0..args.warmup {
        if (trial_index + warmup).is_multiple_of(2) {
            run_and_sync(device, &mut reference)?;
            run_and_sync(device, &mut candidate)?;
        } else {
            run_and_sync(device, &mut candidate)?;
            run_and_sync(device, &mut reference)?;
        }
    }

    let mut samples = Vec::with_capacity(args.iterations);
    for iteration in 0..args.iterations {
        let reference_first = (trial_index + iteration).is_multiple_of(2);
        let (reference_ms, candidate_ms, order) = if reference_first {
            (
                time_synchronized(device, &mut reference)?,
                time_synchronized(device, &mut candidate)?,
                ExecutionOrder::ReferenceThenCandidate,
            )
        } else {
            let candidate_ms = time_synchronized(device, &mut candidate)?;
            let reference_ms = time_synchronized(device, &mut reference)?;
            (
                reference_ms,
                candidate_ms,
                ExecutionOrder::CandidateThenReference,
            )
        };
        samples.push(PairedSample::new(
            iteration,
            order,
            reference_ms,
            candidate_ms,
        )?);
    }
    Ok(samples)
}

fn run_and_sync(device: &Device, operation: &mut impl FnMut()) -> Result<()> {
    operation();
    device.sync()?;
    Ok(())
}

fn time_synchronized(device: &Device, operation: &mut impl FnMut()) -> Result<f64> {
    device.sync()?;
    let started = Instant::now();
    operation();
    device.sync()?;
    Ok(started.elapsed().as_secs_f64() * 1_000.0)
}

fn instantiate(
    config: &ModelDef,
    kind: ModelKind,
    tier: usize,
    device: &Device,
    seed: u64,
) -> Result<Transformer> {
    device.seed(seed);
    let mut model = Transformer::new(config, device)
        .with_context(|| format!("failed to instantiate model '{}'", config.name))?;
    match kind {
        ModelKind::Static => ensure!(
            model.memory_slot_statuses().is_empty(),
            "static benchmark model '{}' unexpectedly has memory slots",
            config.name
        ),
        ModelKind::Memory { active } => {
            for slot in 0..active {
                model
                    .activate_memory_slot_all_layers(tier, slot)
                    .with_context(|| {
                        format!(
                            "activating tier {tier} slot {slot} in model '{}'",
                            config.name
                        )
                    })?;
            }
        }
    }
    Ok(model)
}

fn describe_state(
    config: &ModelDef,
    kind: ModelKind,
    model: &Transformer,
) -> Result<StateDescription> {
    let (kind, active) = match kind {
        ModelKind::Static => ("static_backbone", 0),
        ModelKind::Memory { active } => ("fixed_capacity_memory", active),
    };
    Ok(StateDescription {
        model: config.name.clone(),
        kind,
        active_slots_per_memory_layer: active,
        accounting: model.wake_parameter_accounting()?.into(),
    })
}

fn selected_tier_shape(config: &ModelDef, tier: usize) -> Result<(usize, usize)> {
    let capacities = (0..config.num_layers)
        .filter_map(|layer| {
            config
                .block_for_layer(layer)
                .memory
                .as_ref()?
                .tiers
                .get(tier)
                .map(|tier| tier.reserve_experts.capacity)
        })
        .collect::<Vec<_>>();
    ensure!(!capacities.is_empty(), "memory tier {tier} does not exist");
    let capacity = capacities[0];
    ensure!(
        capacities.iter().all(|candidate| *candidate == capacity),
        "memory tier {tier} does not have identical reserve slots in every memory layer"
    );
    Ok((capacities.len(), capacity))
}

fn load_configs(args: &Args) -> Result<(ModelDef, ModelDef)> {
    match (&args.baseline_model, &args.model) {
        (Some(static_path), Some(memory_path)) => {
            Ok((parse_file(static_path)?, parse_file(memory_path)?))
        }
        (None, None) => Ok((
            parse_mal(COMPACT_STATIC_MODEL).context("compact static benchmark MAL is invalid")?,
            parse_mal(COMPACT_MEMORY_MODEL).context("compact memory benchmark MAL is invalid")?,
        )),
        _ => bail!("--model and --baseline-model must be supplied together"),
    }
}

fn parse_file(path: &Path) -> Result<ModelDef> {
    parse_mal_file(path).with_context(|| format!("failed to parse {}", path.display()))
}

fn deterministic_token_ids(tokens: usize, vocab_size: usize, seed: u64) -> Vec<i64> {
    let mut state = seed.max(1);
    (0..tokens)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % vocab_size as u64) as i64
        })
        .collect()
}

fn deterministic_objective_weights(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1_u32 << 24) as f32;
            unit * 2.0 - 1.0
        })
        .collect()
}
