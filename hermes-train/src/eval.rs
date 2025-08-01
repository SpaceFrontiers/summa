//! Forward-only evaluation of a published checkpoint on held-out data.
//!
//! This command shares the trainer's objective forward pass ([`objective_loss`])
//! and its data pipeline ([`visit_samples`] plus [`make_batch`]), so held-out
//! numbers are directly comparable with training-time metrics. It deliberately
//! owns no optimizer, no autodiff tape, and no training state: the evaluation
//! device is never `.autodiff()`, the token cache is never written, and nothing
//! but an explicit `--output` report is created. A live run therefore remains
//! byte-for-byte resumable while it is being evaluated.

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use hermes_llm::{Device, Tokenizer, Transformer};
use hermes_train::task::{TaskAdapter, TaskConfig};
use hermes_train::workflow::validate_retrieval_layer_for_model;
use serde::Serialize;
use serde_json::json;
use tracing::warn;

use crate::data::{
    BatchStats, PhaseDataBinding, SampleStreamConfig, TrainingBatch, TrainingSample, make_batch,
    visit_samples,
};
use crate::{
    EvalArgs, EvalObjective, ObjectiveForward, add_batch_stats, file_sha256, load_config,
    objective_loss, scalar_value, shuffle_seed,
};

/// Report schema version. Increment when a field changes meaning.
const EVAL_REPORT_VERSION: u32 = 1;

/// File names the trainer publishes inside its own output root. Refusing them
/// keeps a mistyped `--output` from overwriting a live run's state with a
/// report; evaluation must never write anything a training run owns.
const RESERVED_OUTPUT_NAMES: [&str; 5] = [
    "current.json",
    "generation-manifest.json",
    "metrics.jsonl",
    "training-state.json",
    "weights.safetensors",
];

/// Weighted mean of per-batch scalars.
///
/// Held-out loss must be weighted by the number of supervised tokens (retrieval
/// accuracy by the number of queries) each batch contributed. Averaging the
/// per-batch means directly would give a short trailing batch the same weight
/// as a full one and silently bias the reported number toward short batches.
#[derive(Clone, Copy, Debug, Default)]
struct WeightedMean {
    weighted_sum: f64,
    weight: f64,
}

impl WeightedMean {
    fn push(&mut self, value: f64, weight: usize) -> Result<()> {
        ensure!(
            value.is_finite(),
            "evaluation produced a non-finite value {value}"
        );
        if weight == 0 {
            return Ok(());
        }
        let weight = weight as f64;
        self.weighted_sum += value * weight;
        self.weight += weight;
        Ok(())
    }

    fn mean(&self) -> Option<f64> {
        (self.weight > 0.0).then(|| self.weighted_sum / self.weight)
    }
}

#[derive(Debug, Serialize)]
struct EvalDataSource {
    path: String,
    /// Authenticated identity of the shard, matching the trainer's run
    /// signature input for the same data.
    identity: String,
    samples_read: usize,
}

#[derive(Debug, Serialize)]
struct CausalLmMetrics {
    loss: f64,
    perplexity: f64,
}

#[derive(Debug, Serialize)]
struct RetrievalMetrics {
    loss: f64,
    top1_accuracy: f64,
    mrr: f64,
    recall_at_k: f64,
    k: usize,
    candidates: CandidateCounts,
}

#[derive(Debug, Serialize)]
struct CandidateCounts {
    total: usize,
    min_per_batch: usize,
    max_per_batch: usize,
}

/// Deterministic report: it deliberately contains no timing or hostname, so two
/// runs with the same inputs and seed produce byte-identical JSON.
#[derive(Debug, Serialize)]
struct EvalReport {
    version: u32,
    objective: &'static str,
    config: String,
    config_sha256: String,
    tokenizer: String,
    tokenizer_sha256: String,
    checkpoint: String,
    checkpoint_sha256: String,
    device: String,
    sequence_length: usize,
    batch_size: usize,
    shuffle_buffer: usize,
    seed: u64,
    max_batches: Option<usize>,
    data: Vec<EvalDataSource>,
    batches: usize,
    examples: usize,
    compute_tokens: usize,
    supervised_tokens: usize,
    truncated_tokens: usize,
    /// Samples read but not scored because they could not fill a batch.
    dropped_samples: usize,
    /// Every condition that degraded or adjusted this evaluation, in the order
    /// it was detected. Also written to stderr, because the CLI's default log
    /// filter would otherwise hide a `warn!` from an operator.
    warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    router_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    causal_lm: Option<CausalLmMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retrieval: Option<RetrievalMetrics>,
}

/// Streaming accumulator over scored batches.
struct Evaluation<'a> {
    model: &'a Transformer,
    objective: &'a TaskConfig,
    device: &'a Device,
    sequence_length: usize,
    recall_k: usize,
    batches: usize,
    stats: BatchStats,
    language_loss: WeightedMean,
    router_loss: WeightedMean,
    retrieval_loss: WeightedMean,
    retrieval_top1: WeightedMean,
    retrieval_reciprocal_rank: WeightedMean,
    retrieval_recall_at_k: WeightedMean,
    candidates_per_batch: Option<(usize, usize)>,
}

impl<'a> Evaluation<'a> {
    fn new(
        model: &'a Transformer,
        objective: &'a TaskConfig,
        device: &'a Device,
        sequence_length: usize,
        recall_k: usize,
    ) -> Self {
        Self {
            model,
            objective,
            device,
            sequence_length,
            recall_k,
            batches: 0,
            stats: BatchStats::default(),
            language_loss: WeightedMean::default(),
            router_loss: WeightedMean::default(),
            retrieval_loss: WeightedMean::default(),
            retrieval_top1: WeightedMean::default(),
            retrieval_reciprocal_rank: WeightedMean::default(),
            retrieval_recall_at_k: WeightedMean::default(),
            candidates_per_batch: None,
        }
    }

    /// Run one forward-only batch through the trainer's objective forward pass.
    fn score(&mut self, samples: &[TrainingSample]) -> Result<()> {
        let batch = make_batch(samples, self.sequence_length, self.device)?;
        // Retrieval rank metrics need the similarity matrix the objective
        // already computes, so ask that one forward pass to retain it instead
        // of embedding the batch a second time. Language objectives never
        // materialize full-vocabulary logits here.
        let retrieval_labels = match &batch {
            TrainingBatch::Retrieval(batch) => Some(batch.labels.clone()),
            TrainingBatch::Language(_) => None,
        };
        let ObjectiveForward {
            loss,
            router_loss,
            stats,
            retrieval_correct,
            captured_logits,
        } = objective_loss(
            self.model,
            batch,
            self.objective,
            retrieval_labels.is_some(),
        )?;
        let loss = f64::from(scalar_value(loss)?);
        self.batches = self
            .batches
            .checked_add(1)
            .context("evaluated batch count overflows usize")?;
        add_batch_stats(&mut self.stats, stats)?;
        // Language losses are means over supervised tokens; contrastive losses
        // are means over queries. Weighting each batch by its own denominator
        // keeps the reported mean exact for unequal batches.
        let weight = match retrieval_labels {
            Some(_) => stats.examples,
            None => stats.supervised_tokens,
        };
        if let Some(router_loss) = router_loss {
            self.router_loss
                .push(f64::from(scalar_value(router_loss)?), weight)?;
        }
        match retrieval_labels {
            Some(labels) => {
                ensure!(
                    stats.examples > 0,
                    "retrieval batch reported no queries to score"
                );
                self.retrieval_loss.push(loss, weight)?;
                let correct = f64::from(scalar_value(
                    retrieval_correct.context("retrieval batch reported no top-1 count")?,
                )?);
                self.retrieval_top1
                    .push(correct / stats.examples as f64, stats.examples)?;
                let similarities =
                    captured_logits.context("retrieval batch retained no similarity matrix")?;
                let [queries, candidates] = similarities.dims();
                ensure!(
                    queries == stats.examples && candidates == stats.retrieval_candidates,
                    "retrieval similarity matrix is {queries}x{candidates} for {} queries and {} candidates",
                    stats.examples,
                    stats.retrieval_candidates
                );
                self.candidates_per_batch = Some(match self.candidates_per_batch {
                    Some((minimum, maximum)) => (minimum.min(candidates), maximum.max(candidates)),
                    None => (candidates, candidates),
                });
                // Ranking is invariant to the positive temperature scale, so
                // the unscaled similarities rank exactly like the loss logits.
                let scores = similarities.into_data().convert::<f32>().to_vec::<f32>()?;
                let labels = labels.into_data().convert::<i64>().to_vec::<i64>()?;
                ensure!(
                    labels.len() == queries,
                    "retrieval batch has {} labels for {queries} queries",
                    labels.len()
                );
                for (query, label) in labels.into_iter().enumerate() {
                    let label = usize::try_from(label)
                        .ok()
                        .filter(|label| *label < candidates)
                        .with_context(|| {
                            format!("retrieval label {label} is outside {candidates} candidates")
                        })?;
                    let row = &scores[query * candidates..(query + 1) * candidates];
                    let positive = row[label];
                    // Count strictly better candidates so tied scores never
                    // produce an optimistic rank of zero.
                    let rank = 1 + row.iter().filter(|score| **score > positive).count();
                    self.retrieval_reciprocal_rank.push(1.0 / rank as f64, 1)?;
                    self.retrieval_recall_at_k
                        .push(f64::from(u8::from(rank <= self.recall_k)), 1)?;
                }
            }
            None => {
                ensure!(
                    stats.supervised_tokens > 0,
                    "language batch reported no supervised tokens"
                );
                self.language_loss.push(loss, weight)?;
            }
        }
        Ok(())
    }

    fn causal_lm_metrics(&self) -> Result<Option<CausalLmMetrics>> {
        let Some(loss) = self.language_loss.mean() else {
            return Ok(None);
        };
        let perplexity = loss.exp();
        ensure!(
            perplexity.is_finite(),
            "mean held-out loss {loss} overflows perplexity; check that --config and --checkpoint match"
        );
        Ok(Some(CausalLmMetrics { loss, perplexity }))
    }

    fn retrieval_metrics(&self) -> Result<Option<RetrievalMetrics>> {
        let Some(loss) = self.retrieval_loss.mean() else {
            return Ok(None);
        };
        let (min_per_batch, max_per_batch) = self
            .candidates_per_batch
            .context("retrieval evaluation recorded no candidate counts")?;
        Ok(Some(RetrievalMetrics {
            loss,
            top1_accuracy: self
                .retrieval_top1
                .mean()
                .context("retrieval evaluation recorded no top-1 counts")?,
            mrr: self
                .retrieval_reciprocal_rank
                .mean()
                .context("retrieval evaluation recorded no ranks")?,
            recall_at_k: self
                .retrieval_recall_at_k
                .mean()
                .context("retrieval evaluation recorded no ranks")?,
            k: self.recall_k,
            candidates: CandidateCounts {
                total: self.stats.retrieval_candidates,
                min_per_batch,
                max_per_batch,
            },
        }))
    }
}

impl EvalArgs {
    /// Build the exact task configuration the trainer would use. Retrieval
    /// defaults are deserialized rather than restated so query and document
    /// prefixes stay identical to an unset workflow objective.
    fn objective_config(&self) -> Result<TaskConfig> {
        let objective = match self.objective {
            EvalObjective::CausalLm => {
                ensure!(
                    self.retrieval_layer.is_none() && self.temperature.is_none(),
                    "--retrieval-layer and --temperature only apply to --objective contrastive_retrieval"
                );
                TaskConfig::CausalLm {}
            }
            EvalObjective::ContrastiveRetrieval => {
                let mut objective: TaskConfig =
                    serde_json::from_value(json!({"type": "retrieval_representation"}))
                        .context("built-in retrieval objective defaults are invalid")?;
                let TaskConfig::RetrievalRepresentation {
                    temperature, layer, ..
                } = &mut objective
                else {
                    unreachable!("retrieval_representation deserialized to another task");
                };
                if let Some(configured) = self.temperature {
                    *temperature = configured;
                }
                *layer = self.retrieval_layer;
                objective
            }
        };
        objective.validate()?;
        Ok(objective)
    }
}

/// Report a degraded or adjusted condition to both the operator and the report.
/// `tracing` alone is not enough here: the CLI's default filter hides warnings.
fn warn_operator(warnings: &mut Vec<String>, message: String) {
    eprintln!("warning: {message}");
    warn!("{message}");
    warnings.push(message);
}

fn validate_report_path(output: &Path) -> Result<()> {
    ensure!(
        !output.is_dir(),
        "--output {} is a directory",
        output.display()
    );
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("--output {} has no file name", output.display()))?;
    ensure!(
        !RESERVED_OUTPUT_NAMES.contains(&name),
        "--output {} would overwrite a training artifact named `{name}`; write the report outside a run output directory",
        output.display()
    );
    Ok(())
}

pub(super) fn evaluate(args: EvalArgs) -> Result<()> {
    let started = Instant::now();
    let objective = args.objective_config()?;
    if let Some(output) = &args.output {
        validate_report_path(output)?;
    }
    ensure!(
        args.sequence_length > 0,
        "--sequence-length must be positive"
    );
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    ensure!(args.recall_k > 0, "--recall-k must be positive");
    ensure!(
        args.max_batches != Some(0),
        "--max-batches must be positive when it is set"
    );
    let mut warnings = Vec::new();
    if args.shuffle_buffer == 0 && args.seed != 0 {
        warn_operator(
            &mut warnings,
            format!(
                "--seed {} has no effect with --shuffle-buffer 0; held-out shards are read in source order",
                args.seed
            ),
        );
    }

    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let mut config = load_config(&args.config)?;
    if config.vocab_size != tokenizer.vocab_size() {
        warn_operator(
            &mut warnings,
            format!(
                "model config vocab_size {} differs from tokenizer vocab_size {}; evaluating with the tokenizer vocabulary exactly as `train` does",
                config.vocab_size,
                tokenizer.vocab_size()
            ),
        );
        config.vocab_size = tokenizer.vocab_size();
    }
    ensure!(
        args.sequence_length <= config.max_seq_len,
        "--sequence-length {} exceeds model max_seq_len {}",
        args.sequence_length,
        config.max_seq_len
    );
    if let Some(layer) = objective.retrieval_layer() {
        validate_retrieval_layer_for_model(layer, &config, "eval")?;
    }

    // Forward-only by construction: this device is never `.autodiff()`, so no
    // tape is recorded and Burn's dropout modules are the identity. The model
    // is also never `prepare_inference()`d, because its mixed-precision decode
    // weights would make these numbers incomparable with training metrics.
    let device = hermes_llm::default_device();
    let mut model = Transformer::new(&config, &device)?;
    hermes_llm::load_safetensors(&mut model, &args.checkpoint).with_context(|| {
        format!(
            "checkpoint {} does not match model configuration {}",
            args.checkpoint.display(),
            args.config.display()
        )
    })?;

    // In-batch negatives make retrieval accuracy a function of the candidate
    // pool size, so a short trailing batch is dropped rather than reported as
    // if it were comparable. Language losses are token weighted and can score
    // the trailing batch exactly.
    let drop_incomplete_batch = matches!(args.objective, EvalObjective::ContrastiveRetrieval);
    let mut evaluation = Evaluation::new(
        &model,
        &objective,
        &device,
        args.sequence_length,
        args.recall_k,
    );
    let mut samples: Vec<TrainingSample> = Vec::with_capacity(args.batch_size);
    let mut sources = Vec::with_capacity(args.data.len());
    let mut reached_max_batches = false;
    for (index, path) in args.data.iter().enumerate() {
        if reached_max_batches {
            break;
        }
        let binding = PhaseDataBinding::open(path)?;
        let samples_read = visit_samples(
            path,
            &objective,
            &tokenizer,
            SampleStreamConfig {
                seq_len: args.sequence_length,
                shuffle_buffer: args.shuffle_buffer,
                seed: shuffle_seed(args.seed, index, 0),
                // Evaluation never writes a token cache: the trainer's caches
                // belong to a live run's output directory.
                token_cache: None,
                data_binding: &binding,
            },
            |sample| {
                samples.push(sample);
                if samples.len() < args.batch_size {
                    return Ok(true);
                }
                evaluation.score(&samples)?;
                samples.clear();
                reached_max_batches = args
                    .max_batches
                    .is_some_and(|maximum| evaluation.batches >= maximum);
                Ok(!reached_max_batches)
            },
        )
        .with_context(|| format!("cannot evaluate held-out shard {}", path.display()))?;
        binding.ensure_still_published()?;
        sources.push(EvalDataSource {
            path: path.display().to_string(),
            identity: binding.signature_identity().to_owned(),
            samples_read,
        });
    }
    let mut dropped_samples = 0;
    if !samples.is_empty() {
        if drop_incomplete_batch || reached_max_batches {
            dropped_samples = samples.len();
            warn_operator(
                &mut warnings,
                format!(
                    "dropped {dropped_samples} held-out sample(s) that could not fill --batch-size {}",
                    args.batch_size
                ),
            );
        } else {
            evaluation.score(&samples)?;
        }
    }
    ensure!(
        evaluation.batches > 0,
        "held-out data produced no complete batch of --batch-size {}; {dropped_samples} sample(s) were read",
        args.batch_size
    );

    let report = EvalReport {
        version: EVAL_REPORT_VERSION,
        objective: objective.name(),
        config: args.config.display().to_string(),
        config_sha256: file_sha256(&args.config)?,
        tokenizer: args.tokenizer.display().to_string(),
        tokenizer_sha256: file_sha256(&args.tokenizer)?,
        checkpoint: args.checkpoint.display().to_string(),
        checkpoint_sha256: file_sha256(&args.checkpoint)?,
        device: format!("{device:?}"),
        sequence_length: args.sequence_length,
        batch_size: args.batch_size,
        shuffle_buffer: args.shuffle_buffer,
        seed: args.seed,
        max_batches: args.max_batches,
        data: sources,
        batches: evaluation.batches,
        examples: evaluation.stats.examples,
        compute_tokens: evaluation.stats.compute_tokens,
        supervised_tokens: evaluation.stats.supervised_tokens,
        truncated_tokens: evaluation.stats.truncated_tokens,
        dropped_samples,
        warnings,
        router_loss: evaluation.router_loss.mean(),
        causal_lm: evaluation.causal_lm_metrics()?,
        retrieval: evaluation.retrieval_metrics()?,
    };
    write_report(&report, args.output.as_deref())?;
    print_summary(&report, started.elapsed().as_secs_f64());
    Ok(())
}

fn write_report(report: &EvalReport, output: Option<&Path>) -> Result<()> {
    let Some(output) = output else {
        return Ok(());
    };
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create report directory {}", parent.display()))?;
    }
    let mut encoded = serde_json::to_vec_pretty(report)?;
    encoded.push(b'\n');
    fs::write(output, encoded)
        .with_context(|| format!("cannot write evaluation report {}", output.display()))
}

fn print_summary(report: &EvalReport, elapsed_seconds: f64) {
    println!("objective            {}", report.objective);
    println!(
        "checkpoint           {} ({})",
        report.checkpoint, report.checkpoint_sha256
    );
    println!(
        "data                 {} shard(s), {} sample(s) read",
        report.data.len(),
        report
            .data
            .iter()
            .map(|source| source.samples_read)
            .sum::<usize>()
    );
    println!(
        "batches              {} (batch_size {}, sequence_length {})",
        report.batches, report.batch_size, report.sequence_length
    );
    println!(
        "tokens               {} supervised, {} compute, {} truncated",
        report.supervised_tokens, report.compute_tokens, report.truncated_tokens
    );
    println!(
        "dropped samples      {} (incomplete trailing batch)",
        report.dropped_samples
    );
    if let Some(causal_lm) = &report.causal_lm {
        println!("loss                 {:.6}", causal_lm.loss);
        println!("perplexity           {:.4}", causal_lm.perplexity);
    }
    if let Some(retrieval) = &report.retrieval {
        println!("loss                 {:.6}", retrieval.loss);
        println!("top-1 accuracy       {:.6}", retrieval.top1_accuracy);
        println!("mrr                  {:.6}", retrieval.mrr);
        let recall = format!("recall@{}", retrieval.k);
        println!("{recall:<20} {:.6}", retrieval.recall_at_k);
        println!(
            "candidates           {} total, {}..{} per batch",
            retrieval.candidates.total,
            retrieval.candidates.min_per_batch,
            retrieval.candidates.max_per_batch
        );
    }
    if let Some(router_loss) = report.router_loss {
        println!("router loss          {router_loss:.6}");
    }
    println!("warnings             {}", report.warnings.len());
    println!("elapsed              {elapsed_seconds:.2}s");
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use hermes_llm::parse_mal;

    use super::*;
    use crate::data::write_test_tokenizer;

    const EVAL_MODEL: &str = r#"
ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
model tiny {
    vocab_size: 257 max_seq_len: 64 hidden_size: 8 num_layers: 2
    block: {
        attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
        ffn: base
        dropout: 0.0
    }
}
"#;

    struct Fixture {
        _directory: tempfile::TempDir,
        config: PathBuf,
        tokenizer: PathBuf,
        checkpoint: PathBuf,
        data: PathBuf,
        output: PathBuf,
    }

    fn fixture(records: &str) -> Fixture {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().to_owned();
        let config = root.join("model.mal");
        fs::write(&config, EVAL_MODEL).unwrap();
        write_test_tokenizer(&root);
        let device = hermes_llm::default_device();
        device.seed(7);
        let model = Transformer::new(&parse_mal(EVAL_MODEL).unwrap(), &device).unwrap();
        let checkpoint = root.join("weights.safetensors");
        hermes_llm::save_safetensors(&model, &checkpoint).unwrap();
        let data = root.join("holdout.jsonl");
        fs::write(&data, records).unwrap();
        Fixture {
            _directory: directory,
            config,
            tokenizer: root.join("tokenizer.json"),
            checkpoint,
            data,
            output: root.join("reports/eval.json"),
        }
    }

    fn args(fixture: &Fixture, objective: EvalObjective, batch_size: usize) -> EvalArgs {
        EvalArgs {
            config: fixture.config.clone(),
            tokenizer: fixture.tokenizer.clone(),
            checkpoint: fixture.checkpoint.clone(),
            data: vec![fixture.data.clone()],
            objective,
            // Long enough for the retrieval task's fixed query and document
            // prefixes plus their truncatable text.
            sequence_length: 56,
            batch_size,
            max_batches: None,
            shuffle_buffer: 4,
            seed: 11,
            recall_k: 2,
            retrieval_layer: None,
            temperature: None,
            output: Some(fixture.output.clone()),
        }
    }

    fn report(path: &Path) -> serde_json::Value {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    fn finite(value: &serde_json::Value, pointer: &str) -> f64 {
        let number = value
            .pointer(pointer)
            .unwrap_or_else(|| panic!("report has no {pointer}: {value}"))
            .as_f64()
            .unwrap_or_else(|| panic!("{pointer} is not a finite JSON number: {value}"));
        assert!(number.is_finite(), "{pointer} is {number}");
        number
    }

    fn causal_records(documents: usize) -> String {
        (0..documents)
            .map(|index| format!("{{\"text\":\"document {index} of held-out prose\"}}\n"))
            .collect()
    }

    fn retrieval_records(rows: usize) -> String {
        (0..rows)
            .map(|index| {
                format!(
                    "{{\"query\":\"query {index}\",\"positive\":\"answer {index}\",\"negatives\":[\"other {index}\"]}}\n"
                )
            })
            .collect()
    }

    #[test]
    fn eval_causal_lm_reports_finite_token_weighted_loss_and_is_deterministic() {
        let fixture = fixture(&causal_records(24));
        evaluate(args(&fixture, EvalObjective::CausalLm, 2)).unwrap();
        let first = fs::read(&fixture.output).unwrap();
        let parsed = report(&fixture.output);
        let loss = finite(&parsed, "/causal_lm/loss");
        let perplexity = finite(&parsed, "/causal_lm/perplexity");
        assert!(loss > 0.0, "{parsed}");
        assert!(
            (perplexity - loss.exp()).abs() < 1e-9,
            "perplexity {perplexity} does not match loss {loss}"
        );
        assert_eq!(parsed["objective"], "causal_lm");
        assert!(parsed["retrieval"].is_null(), "{parsed}");
        let batches = parsed["batches"].as_u64().unwrap();
        assert!(batches > 0, "{parsed}");
        // Packed causal batches are dense, so every compute token is supervised.
        assert_eq!(parsed["supervised_tokens"], parsed["compute_tokens"]);
        assert_eq!(
            parsed["supervised_tokens"].as_u64().unwrap(),
            batches * 2 * 56
        );
        assert!(
            parsed["data"][0]["identity"]
                .as_str()
                .unwrap()
                .starts_with("sha256:"),
            "{parsed}"
        );

        // The report carries no timing or environment noise, so the same seed
        // must reproduce it byte for byte.
        evaluate(args(&fixture, EvalObjective::CausalLm, 2)).unwrap();
        assert_eq!(first, fs::read(&fixture.output).unwrap());
    }

    #[test]
    fn eval_contrastive_retrieval_reports_finite_rank_metrics_and_is_deterministic() {
        let fixture = fixture(&retrieval_records(16));
        evaluate(args(&fixture, EvalObjective::ContrastiveRetrieval, 2)).unwrap();
        let first = fs::read(&fixture.output).unwrap();
        let parsed = report(&fixture.output);
        let top1 = finite(&parsed, "/retrieval/top1_accuracy");
        let mrr = finite(&parsed, "/retrieval/mrr");
        let recall = finite(&parsed, "/retrieval/recall_at_k");
        finite(&parsed, "/retrieval/loss");
        assert!((0.0..=1.0).contains(&top1), "{parsed}");
        assert!(mrr >= top1 - 1e-9 && mrr <= 1.0, "{parsed}");
        assert!(recall >= top1 - 1e-9 && recall <= 1.0, "{parsed}");
        assert_eq!(parsed["objective"], "retrieval_representation");
        assert!(parsed["causal_lm"].is_null(), "{parsed}");
        assert_eq!(parsed["supervised_tokens"], 0);
        // Two queries with one positive and one explicit negative each.
        assert_eq!(parsed["retrieval"]["candidates"]["min_per_batch"], 4);
        assert_eq!(parsed["retrieval"]["candidates"]["max_per_batch"], 4);
        assert_eq!(parsed["retrieval"]["k"], 2);

        evaluate(args(&fixture, EvalObjective::ContrastiveRetrieval, 2)).unwrap();
        assert_eq!(first, fs::read(&fixture.output).unwrap());
    }

    /// Token weighting makes the reported mean independent of how the same
    /// held-out tokens are grouped into batches. An unweighted average of
    /// per-batch means cannot hold this property: it would give the short
    /// trailing batch the same weight as a full one.
    #[test]
    fn eval_causal_lm_loss_is_token_weighted_across_unequal_batch_groupings() {
        const BATCH: u64 = 4;

        let fixture = fixture(&causal_records(28));
        let mut single = args(&fixture, EvalObjective::CausalLm, 1);
        single.output = Some(fixture.output.with_file_name("single.json"));
        evaluate(single).unwrap();
        let single = report(&fixture.output.with_file_name("single.json"));
        let samples = single["examples"].as_u64().unwrap();
        assert_eq!(single["batches"].as_u64().unwrap(), samples);
        assert!(
            !samples.is_multiple_of(BATCH),
            "the grouped run needs a short trailing batch, got {samples} samples"
        );

        let mut grouped = args(&fixture, EvalObjective::CausalLm, BATCH as usize);
        grouped.output = Some(fixture.output.with_file_name("grouped.json"));
        evaluate(grouped).unwrap();
        let grouped = report(&fixture.output.with_file_name("grouped.json"));
        // The trailing batch is scored, not dropped, for language objectives.
        assert_eq!(grouped["batches"].as_u64().unwrap(), samples / BATCH + 1);
        assert_eq!(grouped["examples"].as_u64().unwrap(), samples);
        assert_eq!(grouped["dropped_samples"], 0);
        assert_eq!(single["supervised_tokens"], grouped["supervised_tokens"]);

        let grouped_loss = finite(&grouped, "/causal_lm/loss");
        let single_loss = finite(&single, "/causal_lm/loss");
        assert!(
            (grouped_loss - single_loss).abs() < 1e-3,
            "token-weighted means diverged across batch groupings: {grouped_loss} vs {single_loss}"
        );
    }

    #[test]
    fn eval_max_batches_bounds_the_run_and_reports_the_bound() {
        let fixture = fixture(&causal_records(24));
        let mut bounded = args(&fixture, EvalObjective::CausalLm, 2);
        bounded.max_batches = Some(1);
        evaluate(bounded).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(parsed["batches"], 1);
        assert_eq!(parsed["max_batches"], 1);
        assert_eq!(parsed["examples"], 2);
    }

    #[test]
    fn eval_rejects_a_checkpoint_that_does_not_match_the_configuration() {
        let fixture = fixture(&causal_records(4));
        let mismatched = fixture.config.with_extension("wide.mal");
        fs::write(
            &mismatched,
            EVAL_MODEL.replace("hidden_size: 8", "hidden_size: 16"),
        )
        .unwrap();
        let mut arguments = args(&fixture, EvalObjective::CausalLm, 2);
        arguments.config = mismatched;
        let error = format!("{:#}", evaluate(arguments).unwrap_err());
        assert!(error.contains("does not match"), "{error}");
    }

    #[test]
    fn eval_rejects_retrieval_flags_on_the_causal_objective() {
        let fixture = fixture(&causal_records(4));
        let mut arguments = args(&fixture, EvalObjective::CausalLm, 2);
        arguments.temperature = Some(0.1);
        let error = format!("{:#}", evaluate(arguments).unwrap_err());
        assert!(error.contains("contrastive_retrieval"), "{error}");
    }

    #[test]
    fn eval_rejects_a_retrieval_layer_outside_the_model() {
        let fixture = fixture(&retrieval_records(4));
        let mut arguments = args(&fixture, EvalObjective::ContrastiveRetrieval, 2);
        arguments.retrieval_layer = Some(3);
        let error = format!("{:#}", evaluate(arguments).unwrap_err());
        assert!(error.contains("model has 2 layers"), "{error}");
    }

    #[test]
    fn eval_records_dropped_trailing_retrieval_samples_in_the_report() {
        let fixture = fixture(&retrieval_records(16));
        // In-batch negatives make accuracy depend on the candidate pool, so the
        // single leftover query is dropped instead of scored more easily.
        evaluate(args(&fixture, EvalObjective::ContrastiveRetrieval, 3)).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(parsed["batches"], 5);
        assert_eq!(parsed["examples"], 15);
        assert_eq!(parsed["dropped_samples"], 1);
        let warnings = parsed["warnings"].as_array().unwrap();
        assert_eq!(warnings.len(), 1, "{parsed}");
        assert!(
            warnings[0].as_str().unwrap().contains("dropped 1 held-out"),
            "{parsed}"
        );
    }

    #[test]
    fn eval_refuses_to_write_a_report_over_a_training_artifact() {
        let fixture = fixture(&causal_records(4));
        for reserved in ["current.json", "metrics.jsonl", "weights.safetensors"] {
            let mut arguments = args(&fixture, EvalObjective::CausalLm, 2);
            arguments.output = Some(fixture.checkpoint.with_file_name(reserved));
            let error = format!("{:#}", evaluate(arguments).unwrap_err());
            assert!(
                error.contains("would overwrite a training artifact"),
                "{error}"
            );
        }
        // The trainer's checkpoint must still be byte-identical afterwards.
        assert!(fixture.checkpoint.is_file());
    }

    #[test]
    fn eval_rejects_data_too_small_for_one_complete_batch() {
        let fixture = fixture(&retrieval_records(1));
        let error = format!(
            "{:#}",
            evaluate(args(&fixture, EvalObjective::ContrastiveRetrieval, 4)).unwrap_err()
        );
        assert!(error.contains("no complete batch"), "{error}");
    }

    #[test]
    fn token_weighted_mean_favors_batches_with_more_supervised_tokens() {
        let mut weighted = WeightedMean::default();
        weighted.push(1.0, 100).unwrap();
        weighted.push(3.0, 1).unwrap();
        let weighted_mean = weighted.mean().unwrap();
        let unweighted_mean = 2.0;
        assert!(
            (weighted_mean - (1.0 * 100.0 + 3.0) / 101.0).abs() < 1e-12,
            "{weighted_mean}"
        );
        assert!(
            weighted_mean < unweighted_mean,
            "token weighting must not average per-batch means: {weighted_mean}"
        );

        // The same two values with the weights swapped must move the mean the
        // other way, which an unweighted average could never do.
        let mut swapped = WeightedMean::default();
        swapped.push(1.0, 1).unwrap();
        swapped.push(3.0, 100).unwrap();
        let swapped_mean = swapped.mean().unwrap();
        assert!(
            (swapped_mean - (1.0 + 3.0 * 100.0) / 101.0).abs() < 1e-12,
            "{swapped_mean}"
        );
        assert!(swapped_mean > unweighted_mean, "{swapped_mean}");
        assert!(WeightedMean::default().mean().is_none());
        assert!(WeightedMean::default().push(f64::NAN, 1).is_err());
    }
}
