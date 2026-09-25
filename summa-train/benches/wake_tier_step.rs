//! Paired profile of the complete memory-tier wake hot path.
//!
//! Unlike the model-only reserve benchmark, every timed operation here runs a
//! real Transformer language-model forward/backward, partitions the resulting
//! graph, commits tier gradients into their independently clocked accumulators,
//! and calls `TierOptimizerBank::apply_wake_only_due_updates`. The paired cases
//! differ only in whether the supplied clock is a tier boundary.
//!
//! The embedded model is intentionally compact for local correctness smoke
//! runs. A local CPU or Metal result is always labelled smoke-only. CUDA
//! profiling must opt in explicitly and build summa-train's `cuda` feature,
//! which enables Summa LLM's CUDA plus `training-fusion` stack:
//!
//! ```text
//! cargo bench -p summa-train --bench wake_tier_step --features cuda -- \
//!   --model "$PWD/summa-mal/well-known/retriever_300m_moe_sleep.mal" \
//!   --batch-size 4 --sequence-length 1024 --periods 100,400,3200 \
//!   --non-due-clock 99 --due-clock 100 --require-cuda \
//!   --output "$PWD/wake-tier-step-cuda.json"
//! ```
//!
//! Model initialization and input construction are outside the timed region.

#[path = "wake_tier_step/cli.rs"]
mod cli;
#[path = "wake_tier_step/stats.rs"]
mod stats;

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use burn::prelude::*;
use burn::tensor::TensorData;
use serde::Serialize;
use summa_llm::{Device, ModelDef, Transformer, default_device, parse_mal};
use summa_train::artifact_io::{read_regular_bounded, sha256_identity};
use summa_train::sleep::{MemoryTierSchedule, SleepSchedule, TerminalConsolidation, UpdateClock};
use summa_train::tier_optimizer::{TierOptimizerBank, TierOptimizerConfig};
use summa_train::workflow::validate_sleep_schedule_for_model;

use cli::{Args, backend_name, parse_args};
use stats::{ExecutionOrder, PairedSample, PairedSummary, PairedTrial};

const MODEL_INITIALIZATION_SEED: u64 = 0x5741_4b45_5449_4552;
const MODEL_EXECUTION_SEED_DOMAIN: u64 = 0x5041_4952_4544_524e;
const MAX_MODEL_MAL_BYTES: u64 = 4 * 1024 * 1024;

const COMPACT_MEMORY_MODEL: &str = r#"
ffn base { hidden_dim: 64 activation: swiglu dropout: 0.0 bias: false }
memory wake_bench {
    tier fast {
        ffn: base
        reserve_experts { capacity: 1 rank: 8 top_k: 1 }
    }
    tier slow {
        ffn: base residual_init: zero
        reserve_experts { capacity: 2 rank: 8 top_k: 1 }
    }
}
model wake-tier-step-bench {
    vocab_size: 128 max_seq_len: 32 hidden_size: 32 num_layers: 1
    block: {
        attention: { num_heads: 4 dropout: 0.0 position_encoding: none }
        memory: wake_bench
        dropout: 0.0
    }
    embeddings { tie_weights: true }
}
"#;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StepObservation {
    wake_gradient_tensors: usize,
    tier_gradient_tensors: Vec<usize>,
    due_tiers: Vec<usize>,
    accumulated_steps_after: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct BuildDescription {
    backend: &'static str,
    device: String,
    summa_llm_training_fusion: bool,
    measurement_scope: &'static str,
    cuda_acceptance_eligible: bool,
}

#[derive(Debug, Serialize)]
struct WorkloadDescription {
    batch_size: usize,
    sequence_length: usize,
    logical_tokens_per_step: usize,
    optimizer_start_state: &'static str,
    paired_model_rng: &'static str,
    timed_operations: [&'static str; 4],
    excluded_from_timing: [&'static str; 3],
}

#[derive(Debug, Serialize)]
struct ClockDescription {
    tier_update_periods: Vec<u64>,
    non_due_clock: u64,
    due_clock: u64,
    due_tiers: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct AppliedUpdateIdentity {
    tier: usize,
    tier_id: String,
    trigger_clock: u64,
    accumulated_optimizer_steps: u64,
    prospective_update_sha256: String,
}

#[derive(Debug, Serialize)]
struct PairedUpdateIdentity {
    seed: u64,
    iteration: usize,
    input_seed: u64,
    non_due_updates: Vec<AppliedUpdateIdentity>,
    due_updates: Vec<AppliedUpdateIdentity>,
}

#[derive(Debug, Serialize)]
struct UpdateDescription {
    optimizer: TierOptimizerConfig,
    prospective_update_identity: &'static str,
    samples: Vec<PairedUpdateIdentity>,
}

#[derive(Debug, Serialize)]
struct ThroughputDescription {
    non_due_tokens_per_second: f64,
    due_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    implementation: &'static str,
    model: String,
    model_mal_sha256: String,
    model_initialization_seed: u64,
    build: BuildDescription,
    workload: WorkloadDescription,
    clocks: ClockDescription,
    warmup_pairs: usize,
    measured_iterations_per_seed: usize,
    paired_seeds: Vec<u64>,
    expected_non_due_observation: StepObservation,
    expected_due_observation: StepObservation,
    updates: UpdateDescription,
    trials: Vec<PairedTrial>,
    summary: PairedSummary,
    throughput: ThroughputDescription,
}

struct PreparedStep {
    model: Transformer,
    bank: TierOptimizerBank,
    input: Tensor<2, Int>,
    targets: Tensor<2, Int>,
    rng_seed: u64,
    trigger_clock: u64,
    expected_due_tiers: Vec<usize>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let (config, model_mal_sha256) = load_config(args.model.as_deref())?;
    ensure!(
        args.sequence_length <= config.max_seq_len,
        "--sequence-length {} exceeds model max_seq_len {}",
        args.sequence_length,
        config.max_seq_len
    );
    ensure!(
        config.vocab_size > 1,
        "benchmark model vocabulary is too small"
    );
    let schedule = schedule_for_model(&config, &args.update_periods)?;
    let optimizer = TierOptimizerConfig::default();

    let backend = backend_name();
    // summa-train's `cuda` feature maps directly to summa-llm's
    // `training-fusion`, which itself includes the CUDA backend.
    let training_fusion = cfg!(feature = "cuda");
    let cuda_acceptance_eligible = backend == "cuda" && training_fusion;
    if args.require_cuda && !cuda_acceptance_eligible {
        bail!(
            "--require-cuda requested, but this benchmark uses backend={backend}, summa_llm_training_fusion={training_fusion}; rebuild on Linux with --features cuda"
        );
    }

    let device = default_device().autodiff();
    device.seed(MODEL_INITIALIZATION_SEED);
    let mut base_model = Transformer::new(&config, &device)
        .with_context(|| format!("failed to instantiate model '{}'", config.name))?;
    base_model
        .activate_memory_slot_all_layers(0, 0)
        .context("activating the fast-tier benchmark reserve")?;

    let due_tiers = schedule.due_senders(args.due_clock);
    ensure!(!due_tiers.is_empty(), "due benchmark clock has no sender");
    ensure!(
        schedule.due_senders(args.non_due_clock).is_empty(),
        "non-due benchmark clock unexpectedly has a sender"
    );

    for warmup in 0..args.warmup {
        let seed = args.seeds[warmup % args.seeds.len()]
            ^ (warmup as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let pair = prepare_pair(&base_model, &schedule, &optimizer, &args, &device, seed)?;
        if warmup.is_multiple_of(2) {
            run_and_sync(pair.0, &device)?;
            run_and_sync(pair.1, &device)?;
        } else {
            run_and_sync(pair.1, &device)?;
            run_and_sync(pair.0, &device)?;
        }
    }

    let mut trials = Vec::with_capacity(args.seeds.len());
    let mut expected_non_due = None;
    let mut expected_due = None;
    let mut update_identities = Vec::with_capacity(
        args.seeds
            .len()
            .checked_mul(args.iterations)
            .context("paired sample count overflowed after validation")?,
    );
    for (trial_index, &seed) in args.seeds.iter().enumerate() {
        let mut samples = Vec::with_capacity(args.iterations);
        for iteration in 0..args.iterations {
            let sample_seed = seed
                ^ (iteration as u64)
                    .wrapping_add(1)
                    .wrapping_mul(0xd1b5_4a32_d192_ed03);
            let (non_due, due) = prepare_pair(
                &base_model,
                &schedule,
                &optimizer,
                &args,
                &device,
                sample_seed,
            )?;
            let non_due_first = (trial_index + iteration).is_multiple_of(2);
            let (non_due_timing, due_timing, order) = if non_due_first {
                (
                    time_synchronized(non_due, &device)?,
                    time_synchronized(due, &device)?,
                    ExecutionOrder::NonDueThenDue,
                )
            } else {
                let due = time_synchronized(due, &device)?;
                let non_due = time_synchronized(non_due, &device)?;
                (non_due, due, ExecutionOrder::DueThenNonDue)
            };
            record_expected(&mut expected_non_due, &non_due_timing.observation)?;
            record_expected(&mut expected_due, &due_timing.observation)?;
            update_identities.push(PairedUpdateIdentity {
                seed,
                iteration,
                input_seed: sample_seed,
                non_due_updates: non_due_timing.updates,
                due_updates: due_timing.updates,
            });
            samples.push(PairedSample::new(
                iteration,
                order,
                non_due_timing.elapsed_ms,
                due_timing.elapsed_ms,
            )?);
        }
        trials.push(PairedTrial::new(seed, samples)?);
    }
    let summary = stats::summarize(&trials)?;
    let logical_tokens = args
        .batch_size
        .checked_mul(args.sequence_length)
        .context("benchmark token count overflowed after validation")?;
    let throughput = ThroughputDescription {
        non_due_tokens_per_second: logical_tokens as f64 / (summary.non_due_median_ms / 1_000.0),
        due_tokens_per_second: logical_tokens as f64 / (summary.due_median_ms / 1_000.0),
    };
    let report = Report {
        schema_version: 1,
        implementation: "paired_complete_memory_tier_wake_step",
        model: config.name,
        model_mal_sha256,
        model_initialization_seed: MODEL_INITIALIZATION_SEED,
        build: BuildDescription {
            backend,
            device: format!("{device:?}"),
            summa_llm_training_fusion: training_fusion,
            measurement_scope: if cuda_acceptance_eligible {
                "cuda_training_fusion_profile"
            } else if backend == "cuda" {
                "cuda_without_training_fusion_smoke_only"
            } else if backend == "ndarray" {
                "local_cpu_smoke_only"
            } else {
                "local_non_cuda_smoke_only"
            },
            cuda_acceptance_eligible,
        },
        workload: WorkloadDescription {
            batch_size: args.batch_size,
            sequence_length: args.sequence_length,
            logical_tokens_per_step: logical_tokens,
            optimizer_start_state: "fresh_bank_without_adam_moments_or_pending_tier_gradients",
            paired_model_rng: "reseeded_identically_before_each_due/non_due_forward",
            timed_operations: [
                "transformer_forward_backward",
                "gradient_partition",
                "tier_gradient_accumulation_commit",
                "wake_only_due_update_check_or_apply",
            ],
            excluded_from_timing: [
                "model_and_optimizer_construction",
                "input_construction",
                "paired_rng_reseed",
            ],
        },
        clocks: ClockDescription {
            tier_update_periods: args.update_periods.clone(),
            non_due_clock: args.non_due_clock,
            due_clock: args.due_clock,
            due_tiers,
        },
        warmup_pairs: args.warmup,
        measured_iterations_per_seed: args.iterations,
        paired_seeds: args.seeds.clone(),
        expected_non_due_observation: expected_non_due
            .context("benchmark produced no non-due observations")?,
        expected_due_observation: expected_due.context("benchmark produced no due observations")?,
        updates: UpdateDescription {
            optimizer,
            prospective_update_identity: "sha256 over canonical before/after tensors in the eligible sender scope (summa-prospective-update-v2)",
            samples: update_identities,
        },
        trials,
        summary,
        throughput,
    };
    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");
    if let Some(path) = &args.output {
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

fn load_config(path: Option<&Path>) -> Result<(ModelDef, String)> {
    let bytes = match path {
        Some(path) => read_regular_bounded(path, MAX_MODEL_MAL_BYTES, "wake-tier benchmark MAL")
            .with_context(|| format!("failed to load benchmark MAL {}", path.display()))?,
        None => COMPACT_MEMORY_MODEL.as_bytes().to_vec(),
    };
    let source = std::str::from_utf8(&bytes).with_context(|| match path {
        Some(path) => format!("benchmark MAL {} is not UTF-8", path.display()),
        None => "embedded benchmark MAL is not UTF-8".to_owned(),
    })?;
    let config = parse_mal(source).with_context(|| match path {
        Some(path) => format!("failed to parse benchmark MAL {}", path.display()),
        None => "invalid embedded benchmark MAL".to_owned(),
    })?;
    Ok((config, sha256_identity(&bytes)))
}

fn schedule_for_model(config: &ModelDef, periods: &[u64]) -> Result<SleepSchedule> {
    let memory = config
        .block_for_layer(0)
        .memory
        .as_ref()
        .context("wake-tier benchmark model has no memory hierarchy")?;
    ensure!(
        memory.tiers.len() == periods.len(),
        "--periods contains {} values, but model defines {} memory tiers",
        periods.len(),
        memory.tiers.len()
    );
    let schedule = SleepSchedule {
        clock: UpdateClock::OptimizerSteps,
        terminal_consolidation: TerminalConsolidation::DistillIntoBaseV1,
        tiers: memory
            .tiers
            .iter()
            .zip(periods)
            .map(|(tier, &update_period)| MemoryTierSchedule {
                id: tier.name.clone(),
                update_period,
                reserve_slots: tier.reserve_experts.capacity,
            })
            .collect(),
    };
    schedule.validate()?;
    validate_sleep_schedule_for_model(config, &schedule, "wake_tier_step benchmark")?;
    Ok(schedule)
}

fn prepare_pair(
    base_model: &Transformer,
    schedule: &SleepSchedule,
    optimizer: &TierOptimizerConfig,
    args: &Args,
    device: &Device,
    seed: u64,
) -> Result<(PreparedStep, PreparedStep)> {
    Ok((
        prepare_step(
            base_model,
            schedule,
            optimizer,
            args,
            device,
            seed,
            args.non_due_clock,
        )?,
        prepare_step(
            base_model,
            schedule,
            optimizer,
            args,
            device,
            seed,
            args.due_clock,
        )?,
    ))
}

fn prepare_step(
    base_model: &Transformer,
    schedule: &SleepSchedule,
    optimizer: &TierOptimizerConfig,
    args: &Args,
    device: &Device,
    seed: u64,
    trigger_clock: u64,
) -> Result<PreparedStep> {
    let logical_tokens = args
        .batch_size
        .checked_mul(args.sequence_length)
        .context("benchmark token count overflowed after validation")?;
    let (input, targets) =
        deterministic_tokens(logical_tokens, base_model.config().vocab_size, seed);
    let model = base_model.clone();
    let bank = TierOptimizerBank::new(&model, schedule, optimizer.clone())?;
    Ok(PreparedStep {
        model,
        bank,
        input: Tensor::<2, Int>::from_data(
            TensorData::new(input, [args.batch_size, args.sequence_length]),
            device,
        ),
        targets: Tensor::<2, Int>::from_data(
            TensorData::new(targets, [args.batch_size, args.sequence_length]),
            device,
        ),
        rng_seed: seed ^ MODEL_EXECUTION_SEED_DOMAIN,
        trigger_clock,
        expected_due_tiers: schedule.due_senders(trigger_clock),
    })
}

fn deterministic_tokens(count: usize, vocab_size: usize, seed: u64) -> (Vec<i64>, Vec<i64>) {
    let mut state = seed ^ 0xa076_1d64_78bd_642f;
    let mut input = Vec::with_capacity(count);
    let mut targets = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let token = state.wrapping_mul(0x2545_f491_4f6c_dd1d) % vocab_size as u64;
        input.push(token as i64);
        targets.push(token.wrapping_add(1) as i64 % vocab_size as i64);
    }
    (input, targets)
}

struct StepResult {
    observation: StepObservation,
    updates: Vec<AppliedUpdateIdentity>,
}

fn execute_step(step: PreparedStep) -> Result<StepResult> {
    let (task_loss, router_loss) = step
        .model
        .forward_loss_with_router(step.input, step.targets);
    let loss = match router_loss {
        Some(router_loss) => task_loss + router_loss,
        None => task_loss,
    };
    let mut gradients = loss.backward();
    let partitioned = step.bank.partition_gradients(&step.model, &mut gradients)?;
    let wake_gradient_tensors = partitioned.report.wake_gradient_tensors;
    let tier_gradient_tensors = partitioned.report.tier_gradient_tensors;
    std::hint::black_box(partitioned.wake);
    let committed = step
        .bank
        .commit_tier_gradients(&step.model, partitioned.tiers, 1)?;
    ensure!(
        committed
            .accumulated_micro_steps
            .iter()
            .all(|steps| *steps == 1),
        "fresh benchmark bank did not commit exactly one optimizer step"
    );
    let (updated_model, update) = step
        .bank
        .apply_wake_only_due_updates(&step.model, step.trigger_clock)?;
    let due_tiers = update
        .updates
        .iter()
        .map(|update| update.tier)
        .collect::<Vec<_>>();
    ensure!(
        due_tiers == step.expected_due_tiers,
        "wake-only update applied tiers {due_tiers:?}, expected {:?}",
        step.expected_due_tiers
    );
    let clocks = step.bank.tier_clocks()?;
    let accumulated_steps_after = clocks
        .iter()
        .map(|(_, _, accumulated)| *accumulated)
        .collect::<Vec<_>>();
    std::hint::black_box(updated_model);
    let updates = update
        .updates
        .into_iter()
        .map(|update| AppliedUpdateIdentity {
            tier: update.tier,
            tier_id: update.tier_id,
            trigger_clock: update.trigger_clock,
            accumulated_optimizer_steps: update.accumulated_optimizer_steps,
            prospective_update_sha256: update.prospective_update_sha256,
        })
        .collect();
    Ok(StepResult {
        observation: StepObservation {
            wake_gradient_tensors,
            tier_gradient_tensors,
            due_tiers,
            accumulated_steps_after,
        },
        updates,
    })
}

struct TimedStep {
    elapsed_ms: f64,
    observation: StepObservation,
    updates: Vec<AppliedUpdateIdentity>,
}

fn run_and_sync(step: PreparedStep, device: &Device) -> Result<StepObservation> {
    device.seed(step.rng_seed);
    device.sync()?;
    let result = execute_step(step)?;
    device.sync()?;
    Ok(result.observation)
}

fn time_synchronized(step: PreparedStep, device: &Device) -> Result<TimedStep> {
    // Model initialization, inputs, and RNG setup are deliberately outside
    // the timed region. Reseeding makes stochastic layers (for example
    // dropout in a supplied MAL) consume identical randomness in each pair.
    device.seed(step.rng_seed);
    device.sync()?;
    let started = Instant::now();
    let result = execute_step(step)?;
    device.sync()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    ensure!(
        elapsed_ms.is_finite() && elapsed_ms > 0.0,
        "benchmark clock returned an invalid duration"
    );
    Ok(TimedStep {
        elapsed_ms,
        observation: result.observation,
        updates: result.updates,
    })
}

fn record_expected(slot: &mut Option<StepObservation>, observed: &StepObservation) -> Result<()> {
    match slot {
        Some(expected) => ensure!(
            expected == observed,
            "benchmark operation shape changed across paired samples"
        ),
        None => *slot = Some(observed.clone()),
    }
    Ok(())
}
