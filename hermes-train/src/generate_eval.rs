//! Free-running generation evaluation of a published checkpoint.
//!
//! `eval` measures teacher-forced loss: given a correct target prefix, how well
//! the model predicts the next token. That number can improve while autoregressive
//! decoding collapses into copying, and on this project it did — the 300M run's
//! grounded-QA perplexity fell 17% across a corrective SFT pass whose decodes
//! were still degenerate. Perplexity and generation measure different things and
//! both are required.
//!
//! Prompts come from [`TaskConfig::construct_supervised_prompt`] and
//! [`frame_supervised`], the same pair training uses, so the decode prompt is the
//! training prompt by construction. Hand-assembling one puts it out of
//! distribution and reports a failure that is not there; that has already
//! happened once, on `qa_reasoning`, and this command exists partly to make it
//! impossible.
//!
//! Design: `docs/generation-eval.md`.

use std::fs;
use std::io::{BufRead, Read};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use hermes_llm::generate::SamplingConfig;
use hermes_llm::{TextGenerator, Tokenizer, Transformer};
use hermes_train::task::{TaskAdapter, TaskConfig};
use serde::Serialize;

use crate::data::{OversizedSupervisedRecord, PhaseDataBinding, frame_supervised};
use crate::eval::{task_defaults, validate_report_path, warn_operator};
use crate::{GenerateEvalArgs, file_sha256, load_config};

/// Report schema version. Increment when a field changes meaning.
const GENERATE_EVAL_REPORT_VERSION: u32 = 1;

/// A generation whose most frequent word trigram occupies more than this share
/// of all its trigrams is counted degenerate. Chosen because healthy short
/// answers on this data sit near zero while the observed collapse
/// ("Berlin, Berlin, Berlin, …") sits far above it.
const DEGENERATE_TRIGRAM_RATE: f64 = 0.10;

/// One malformed record must not make evaluation allocate an unbounded line.
const MAX_RECORD_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Serialize)]
struct GenerateDataSource {
    path: String,
    identity: String,
    records_read: usize,
    /// Records skipped because prompt, target, and EOS exceed
    /// `--sequence-length`. Excluded from every reported number, exactly as
    /// `eval` excludes them.
    oversized_records: usize,
}

/// Decode-quality statistics. Every field is a mean over scored records, so a
/// report is comparable across checkpoints only at the same `--max-new-tokens`
/// and decoding settings, which the report pins.
#[derive(Debug, Serialize)]
struct GenerationMetrics {
    /// Share of a generation's word trigrams taken by its single most frequent
    /// one, averaged. Rises with repetition; zero when every trigram is unique.
    repeated_trigram_rate: f64,
    /// Share of generations above [`DEGENERATE_TRIGRAM_RATE`].
    degenerate_fraction: f64,
    /// Share whose lowercase alphanumeric word sequence contains the complete
    /// gold target word sequence. A blunt correctness floor: it counts exact
    /// inclusion only, so a right answer phrased differently scores zero.
    target_containment: f64,
    /// Share of a generation's word 4-grams that also occur in the prompt's
    /// source text. Interpretation is task-dependent — extraction is the goal for
    /// grounded QA and the failure mode for summarization — so this is reported
    /// without a verdict.
    source_overlap: f64,
    empty_fraction: f64,
    /// Share that ended by emitting EOS instead of hitting `--max-new-tokens`.
    /// A low value alongside a high repeated-trigram rate is the signature of a
    /// model that cannot stop.
    stopped_at_eos: f64,
    mean_generated_tokens: f64,
    mean_generated_words: f64,
}

#[derive(Debug, Serialize)]
struct GenerationSample {
    prompt: String,
    generated: String,
    target: String,
    repeated_trigram_rate: f64,
    contains_target: bool,
}

#[derive(Debug, Serialize)]
struct GenerateEvalReport {
    version: u32,
    objective: &'static str,
    task: TaskConfig,
    config: String,
    config_sha256: String,
    tokenizer: String,
    tokenizer_sha256: String,
    checkpoint: String,
    checkpoint_sha256: String,
    device: String,
    sequence_length: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    repetition_penalty: f64,
    seed: u64,
    data: Vec<GenerateDataSource>,
    scored_records: usize,
    oversized_records: usize,
    truncated_tokens: usize,
    warnings: Vec<String>,
    generation: GenerationMetrics,
    /// Verbatim decodes, written only when `--samples` asks for them. Reviewing
    /// these is what caught the degenerate decoding that perplexity hid.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    samples: Vec<GenerationSample>,
}

/// Lowercase alphanumeric words, the comparison basis for every text metric here.
fn words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Share of `text`'s word trigrams taken by its most frequent trigram, excluding
/// that trigram's first occurrence so unique text scores zero.
fn repeated_trigram_rate(words: &[String]) -> f64 {
    if words.len() < 5 {
        return 0.0;
    }
    let mut counts = std::collections::HashMap::new();
    let total = words.len() - 2;
    for window in words.windows(3) {
        *counts.entry(window).or_insert(0usize) += 1;
    }
    let most = counts.values().copied().max().unwrap_or(1);
    (most - 1) as f64 / total as f64
}

/// Share of `generated`'s word 4-grams that also occur in `source`.
fn source_overlap(generated: &[String], source: &[String]) -> f64 {
    if generated.len() < 4 {
        return 0.0;
    }
    let reference: std::collections::HashSet<&[String]> = source.windows(4).collect();
    let windows = generated.windows(4);
    let total = windows.len();
    let shared = generated
        .windows(4)
        .filter(|window| reference.contains(*window))
        .count();
    shared as f64 / total as f64
}

fn contains_word_sequence(generated: &[String], target: &[String]) -> bool {
    !target.is_empty()
        && target.len() <= generated.len()
        && generated
            .windows(target.len())
            .any(|window| window == target)
}

/// Running mean that reports zero records as absent rather than as zero.
#[derive(Default)]
struct Mean {
    total: f64,
    count: usize,
}

impl Mean {
    fn push(&mut self, value: f64) {
        self.total += value;
        self.count += 1;
    }

    fn value(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.total / self.count as f64
    }
}

pub(super) fn evaluate(args: GenerateEvalArgs) -> Result<()> {
    let started = Instant::now();
    if let Some(output) = &args.output {
        validate_report_path(output)?;
    }
    if let Some(samples) = &args.samples {
        validate_report_path(samples)?;
    }
    if let (Some(output), Some(samples)) = (&args.output, &args.samples) {
        ensure!(
            output != samples,
            "--output and --samples must name different files"
        );
    }
    ensure!(
        args.sequence_length > 0,
        "--sequence-length must be positive"
    );
    ensure!(args.max_new_tokens > 0, "--max-new-tokens must be positive");

    let mut objective = task_defaults(args.objective.task_name())?;
    if let Some(configured) = &args.instruction {
        let instruction = match &mut objective {
            TaskConfig::Summarization { instruction }
            | TaskConfig::InstructionTuning { instruction }
            | TaskConfig::RetrievalPlanning { instruction }
            | TaskConfig::QaReasoning { instruction, .. } => instruction,
            other => unreachable!("`{}` is not a supervised task", other.name()),
        };
        *instruction = configured.clone();
    }
    if let TaskConfig::QaReasoning {
        require_reasoning, ..
    } = &mut objective
    {
        *require_reasoning = args.require_reasoning;
    } else {
        ensure!(
            !args.require_reasoning,
            "--require-reasoning only applies to --objective qa_reasoning"
        );
    }
    objective.validate()?;

    let mut warnings = Vec::new();
    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let mut config = load_config(&args.config)?;
    if config.vocab_size != tokenizer.vocab_size() {
        warn_operator(
            &mut warnings,
            format!(
                "model config vocab_size {} differs from tokenizer vocab_size {}; decoding with the tokenizer vocabulary exactly as `train` does",
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

    let device = hermes_llm::default_device();
    let mut model = Transformer::new(&config, &device)?;
    hermes_llm::load_safetensors(&mut model, &args.checkpoint).with_context(|| {
        format!(
            "checkpoint {} does not match model configuration {}",
            args.checkpoint.display(),
            args.config.display()
        )
    })?;
    let generator = TextGenerator::new(&model, &device);
    let sampling = SamplingConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        eos_token: Some(tokenizer.eos_token_id()),
        // Always explicit: an unseeded decode would make the report
        // irreproducible, and greedy decoding ignores it anyway.
        seed: Some(args.seed),
    };

    let mut repeated = Mean::default();
    let mut degenerate = Mean::default();
    let mut containment = Mean::default();
    let mut overlap = Mean::default();
    let mut empty = Mean::default();
    let mut stopped = Mean::default();
    let mut generated_tokens = Mean::default();
    let mut generated_words = Mean::default();
    let mut truncated_tokens = 0usize;
    let mut scored_records = 0usize;
    let mut samples = Vec::new();
    let mut sources = Vec::with_capacity(args.data.len());

    for path in &args.data {
        let mut records_read = 0usize;
        let mut oversized = 0usize;
        let binding = PhaseDataBinding::open(path)?;
        binding.with_readers(path, |source_path, reader| {
            let mut line = String::new();
            loop {
                if args
                    .max_records
                    .is_some_and(|maximum| scored_records >= maximum)
                {
                    return Ok(false);
                }
                line.clear();
                let read = Read::take(&mut *reader, MAX_RECORD_BYTES)
                    .read_line(&mut line)
                    .with_context(|| format!("cannot read {}", source_path.display()))?;
                if read == 0 {
                    break;
                }
                ensure!(
                    line.ends_with('\n') || read < MAX_RECORD_BYTES as usize,
                    "record {} in {} exceeds the {MAX_RECORD_BYTES}-byte limit",
                    records_read + 1,
                    source_path.display()
                );
                if line.trim().is_empty() {
                    continue;
                }
                records_read += 1;
                let value: serde_json::Value = serde_json::from_str(&line).with_context(|| {
                    format!(
                        "cannot parse {}:{records_read} as JSON",
                        source_path.display()
                    )
                })?;
                let segments =
                    objective
                        .construct_supervised_prompt(&value)
                        .with_context(|| {
                            format!("invalid record at {}:{records_read}", source_path.display())
                        })?;
                let framing = match frame_supervised(
                    &tokenizer,
                    &segments.prefix,
                    &segments.source,
                    &segments.suffix,
                    &segments.target,
                    args.sequence_length,
                    segments.source_required,
                ) {
                    Ok(framing) => framing,
                    // One over-long held-out record must not void the whole
                    // evaluation; skip and count it, as `eval` does. The cause
                    // chain is walked rather than the top-level error so added
                    // context cannot hide the typed rejection.
                    Err(error)
                        if error
                            .chain()
                            .any(|cause| cause.is::<OversizedSupervisedRecord>()) =>
                    {
                        oversized += 1;
                        continue;
                    }
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("cannot frame {}:{records_read}", source_path.display())
                        });
                    }
                };
                truncated_tokens = truncated_tokens
                    .checked_add(framing.truncated_tokens)
                    .context("truncated-token count overflows usize")?;

                let prompt_tokens: Vec<u32> = framing
                    .prompt_tokens
                    .iter()
                    .map(|token| u32::try_from(*token).context("prompt token exceeds u32"))
                    .collect::<Result<_>>()?;
                let visible_source =
                    tokenizer.decode(&prompt_tokens[framing.source_range.clone()], true)?;
                let decoded = generator.generate(&prompt_tokens, &sampling)?;
                let new_tokens = &decoded[prompt_tokens.len()..];
                let ended_at_eos = new_tokens
                    .last()
                    .is_some_and(|token| *token == tokenizer.eos_token_id());
                let new_tokens = if ended_at_eos {
                    &new_tokens[..new_tokens.len() - 1]
                } else {
                    new_tokens
                };
                let text = tokenizer.decode(new_tokens, true)?;
                let text = text.trim();

                let generated = words(text);
                let source = words(&visible_source);
                let rate = repeated_trigram_rate(&generated);
                let normalized_target = words(&segments.target);
                let contains_target = contains_word_sequence(&generated, &normalized_target);

                repeated.push(rate);
                degenerate.push(f64::from(u8::from(rate > DEGENERATE_TRIGRAM_RATE)));
                containment.push(f64::from(u8::from(contains_target)));
                overlap.push(source_overlap(&generated, &source));
                empty.push(f64::from(u8::from(text.is_empty())));
                stopped.push(f64::from(u8::from(ended_at_eos)));
                generated_tokens.push(new_tokens.len() as f64);
                generated_words.push(generated.len() as f64);
                scored_records += 1;

                if args.samples.is_some() {
                    samples.push(GenerationSample {
                        prompt: tokenizer.decode(&prompt_tokens, true)?,
                        generated: text.to_owned(),
                        target: segments.target.to_string(),
                        repeated_trigram_rate: rate,
                        contains_target,
                    });
                }
            }
            Ok(true)
        })?;
        if oversized > 0 {
            warn_operator(
                &mut warnings,
                format!(
                    "skipped {oversized} record(s) in {} whose prompt, complete target, and EOS exceed --sequence-length {}",
                    path.display(),
                    args.sequence_length
                ),
            );
        }
        sources.push(GenerateDataSource {
            path: path.display().to_string(),
            identity: binding.signature_identity().to_owned(),
            records_read,
            oversized_records: oversized,
        });
    }
    ensure!(
        scored_records > 0,
        "held-out data produced no scorable record"
    );

    let oversized_records = sources.iter().map(|source| source.oversized_records).sum();
    let report = GenerateEvalReport {
        version: GENERATE_EVAL_REPORT_VERSION,
        objective: objective.name(),
        task: objective.clone(),
        config: args.config.display().to_string(),
        config_sha256: file_sha256(&args.config)?,
        tokenizer: args.tokenizer.display().to_string(),
        tokenizer_sha256: file_sha256(&args.tokenizer)?,
        checkpoint: args.checkpoint.display().to_string(),
        checkpoint_sha256: file_sha256(&args.checkpoint)?,
        device: format!("{device:?}"),
        sequence_length: args.sequence_length,
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        data: sources,
        scored_records,
        oversized_records,
        truncated_tokens,
        warnings,
        generation: GenerationMetrics {
            repeated_trigram_rate: repeated.value(),
            degenerate_fraction: degenerate.value(),
            target_containment: containment.value(),
            source_overlap: overlap.value(),
            empty_fraction: empty.value(),
            stopped_at_eos: stopped.value(),
            mean_generated_tokens: generated_tokens.value(),
            mean_generated_words: generated_words.value(),
        },
        samples: if args.samples.is_some() {
            std::mem::take(&mut samples)
        } else {
            Vec::new()
        },
    };
    if let Some(path) = &args.samples {
        write_json(&report.samples, path)?;
    }
    let mut published = report;
    if args.samples.is_some() {
        // The samples file owns the verbatim decodes; keeping them in the metrics
        // report too would silently double a large artifact.
        published.samples = Vec::new();
    }
    if let Some(path) = &args.output {
        write_json(&published, path)?;
    }
    print_summary(&published, started.elapsed().as_secs_f64());
    Ok(())
}

fn write_json<T: Serialize>(value: &T, output: &Path) -> Result<()> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create directory {}", parent.display()))?;
    }
    let mut encoded = serde_json::to_vec_pretty(value)?;
    encoded.push(b'\n');
    fs::write(output, encoded).with_context(|| format!("cannot write {}", output.display()))
}

fn print_summary(report: &GenerateEvalReport, elapsed_seconds: f64) {
    println!("objective            {}", report.objective);
    println!(
        "checkpoint           {} ({})",
        report.checkpoint, report.checkpoint_sha256
    );
    println!(
        "decoding             {} new token(s), temperature {}, repetition_penalty {}, seed {}",
        report.max_new_tokens, report.temperature, report.repetition_penalty, report.seed
    );
    println!(
        "records              {} scored, {} oversized, {} truncated token(s)",
        report.scored_records, report.oversized_records, report.truncated_tokens
    );
    let generation = &report.generation;
    println!(
        "repeated 3-grams     {:.4} mean, {:.4} degenerate (>{DEGENERATE_TRIGRAM_RATE})",
        generation.repeated_trigram_rate, generation.degenerate_fraction
    );
    println!("target containment   {:.4}", generation.target_containment);
    println!("source overlap       {:.4}", generation.source_overlap);
    println!(
        "stopped at eos       {:.4} ({:.1} token(s), {:.1} word(s) mean)",
        generation.stopped_at_eos,
        generation.mean_generated_tokens,
        generation.mean_generated_words
    );
    if generation.empty_fraction > 0.0 {
        println!("empty generations    {:.4}", generation.empty_fraction);
    }
    for warning in &report.warnings {
        println!("warning              {warning}");
    }
    println!("elapsed              {elapsed_seconds:.2}s");
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use hermes_llm::parse_mal;

    use super::*;
    use crate::GenerationObjective;
    use crate::data::write_test_tokenizer;

    const GENERATE_MODEL: &str = r#"
ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
model tiny {
    vocab_size: 257 max_seq_len: 256 hidden_size: 8 num_layers: 2
    block: {
        attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
        ffn: base
        dropout: 0.0
    }
}
"#;

    /// Wide enough for the supervised instruction, markers, and record text.
    const GENERATE_SEQUENCE_LENGTH: usize = 192;

    struct Fixture {
        _directory: tempfile::TempDir,
        root: PathBuf,
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
        fs::write(&config, GENERATE_MODEL).unwrap();
        write_test_tokenizer(&root);
        let device = hermes_llm::default_device();
        device.seed(7);
        let model = Transformer::new(&parse_mal(GENERATE_MODEL).unwrap(), &device).unwrap();
        let checkpoint = root.join("weights.safetensors");
        hermes_llm::save_safetensors(&model, &checkpoint).unwrap();
        let data = root.join("holdout.jsonl");
        fs::write(&data, records).unwrap();
        Fixture {
            _directory: directory,
            tokenizer: root.join("tokenizer.json"),
            output: root.join("reports/generate.json"),
            root,
            config,
            checkpoint,
            data,
        }
    }

    fn args(fixture: &Fixture, objective: GenerationObjective) -> GenerateEvalArgs {
        GenerateEvalArgs {
            config: fixture.config.clone(),
            tokenizer: fixture.tokenizer.clone(),
            checkpoint: fixture.checkpoint.clone(),
            data: vec![fixture.data.clone()],
            objective,
            sequence_length: GENERATE_SEQUENCE_LENGTH,
            max_new_tokens: 8,
            temperature: 0.0,
            top_k: None,
            repetition_penalty: 1.0,
            seed: 0,
            max_records: None,
            instruction: None,
            require_reasoning: false,
            samples: None,
            output: Some(fixture.output.clone()),
        }
    }

    fn qa_records(rows: usize) -> String {
        (0..rows)
            .map(|index| {
                format!(
                    "{{\"question\":\"question {index} about a topic\",\"answer\":\"answer {index}\"}}\n"
                )
            })
            .collect()
    }

    fn report(path: &Path) -> serde_json::Value {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    fn word_list(text: &str) -> Vec<String> {
        words(text)
    }

    #[test]
    fn repeated_trigram_rate_separates_unique_text_from_a_collapsed_loop() {
        assert_eq!(
            repeated_trigram_rate(&word_list("alpha beta gamma delta epsilon zeta")),
            0.0,
            "text with no repeated trigram must score zero"
        );
        // "berlin berlin berlin" repeats across a 7-word run: 5 trigrams, the
        // most frequent occurring 3 times.
        let looped = word_list("paris berlin berlin berlin berlin berlin berlin");
        let rate = repeated_trigram_rate(&looped);
        assert!(rate > 0.5, "collapsed loop scored {rate}");
        // Too short to have a meaningful trigram profile.
        assert_eq!(repeated_trigram_rate(&word_list("one two")), 0.0);
    }

    #[test]
    fn source_overlap_counts_only_shared_four_grams() {
        let source = word_list("the capital of france is paris on the river seine");
        assert_eq!(
            source_overlap(&word_list("the capital of france"), &source),
            1.0,
            "a verbatim quote must score one"
        );
        assert_eq!(
            source_overlap(&word_list("entirely different words here now"), &source),
            0.0
        );
        // Shorter than one 4-gram: nothing to compare.
        assert_eq!(source_overlap(&word_list("paris"), &source), 0.0);
    }

    #[test]
    fn target_containment_matches_words_not_substrings() {
        let generated = word_list("Russia and Belarus are countries");
        assert!(contains_word_sequence(&generated, &word_list("Belarus")));
        assert!(!contains_word_sequence(&generated, &word_list("US")));
        assert!(contains_word_sequence(
            &generated,
            &word_list("Russia and Belarus")
        ));
        assert!(!contains_word_sequence(&generated, &[]));
    }

    #[test]
    fn generation_metrics_are_finite_in_range_and_deterministic() {
        let fixture = fixture(&qa_records(4));
        evaluate(args(&fixture, GenerationObjective::QaReasoning)).unwrap();
        let first = report(&fixture.output);
        evaluate(args(&fixture, GenerationObjective::QaReasoning)).unwrap();
        assert_eq!(
            first,
            report(&fixture.output),
            "greedy decoding is not reproducible"
        );
        assert_eq!(
            first.pointer("/scored_records").unwrap().as_u64(),
            Some(4),
            "{first}"
        );
        for field in [
            "repeated_trigram_rate",
            "degenerate_fraction",
            "target_containment",
            "source_overlap",
            "empty_fraction",
            "stopped_at_eos",
        ] {
            let value = first
                .pointer(&format!("/generation/{field}"))
                .unwrap_or_else(|| panic!("no {field}: {first}"))
                .as_f64()
                .unwrap();
            assert!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "{field} is {value}"
            );
        }
    }

    /// `--max-records` bounds an autoregressive pass without reading the rest of
    /// the shard once the requested number of decodes exists.
    #[test]
    fn max_records_bounds_the_pass_and_reports_the_bound() {
        let fixture = fixture(&qa_records(6));
        let mut arguments = args(&fixture, GenerationObjective::QaReasoning);
        arguments.max_records = Some(2);
        evaluate(arguments).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(
            parsed.pointer("/scored_records").unwrap().as_u64(),
            Some(2),
            "{parsed}"
        );
    }

    /// Decodes belong in the samples file only. Serializing them into the metrics
    /// report as well would silently double a large artifact.
    #[test]
    fn samples_are_written_to_their_own_file_and_omitted_from_the_report() {
        let fixture = fixture(&qa_records(3));
        let samples = fixture.root.join("samples.json");
        let mut arguments = args(&fixture, GenerationObjective::QaReasoning);
        arguments.samples = Some(samples.clone());
        evaluate(arguments).unwrap();

        let written = report(&samples);
        let written = written.as_array().expect("samples file holds an array");
        assert_eq!(written.len(), 3, "{written:?}");
        assert!(
            written[0]
                .get("prompt")
                .and_then(|p| p.as_str())
                .is_some_and(|prompt| prompt.contains("Question:") && prompt.contains("Response:")),
            "samples must record the adapter-framed prompt: {written:?}"
        );
        assert!(
            report(&fixture.output).get("samples").is_none(),
            "metrics report must not duplicate the decodes"
        );
    }

    /// A record too long for the geometry must be skipped, counted, and warned
    /// about rather than voiding the evaluation or vanishing.
    #[test]
    fn oversized_records_are_skipped_counted_and_warned() {
        let long_question = "x".repeat(GENERATE_SEQUENCE_LENGTH * 4);
        let records = format!(
            "{{\"question\":\"short question\",\"answer\":\"answer\"}}\n\
             {{\"question\":\"q\",\"answer\":\"{long_question}\"}}\n"
        );
        let fixture = fixture(&records);
        evaluate(args(&fixture, GenerationObjective::QaReasoning)).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(
            parsed.pointer("/oversized_records").unwrap().as_u64(),
            Some(1),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/scored_records").unwrap().as_u64(),
            Some(1),
            "{parsed}"
        );
        let warnings = parsed.pointer("/warnings").unwrap().as_array().unwrap();
        assert!(
            warnings.iter().any(|warning| {
                let warning = warning.as_str().unwrap_or_default();
                warning.contains("skipped 1 record(s)") && warning.contains("--sequence-length")
            }),
            "the drop must name what was skipped and why: {parsed}"
        );
    }

    #[test]
    fn the_report_pins_the_prompt_framing_and_decoding_settings() {
        let fixture = fixture(&qa_records(2));
        let mut arguments = args(&fixture, GenerationObjective::QaReasoning);
        arguments.instruction = Some("Answer from the passage only.".to_owned());
        evaluate(arguments).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(
            parsed.pointer("/task/instruction").unwrap().as_str(),
            Some("Answer from the passage only."),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/temperature").unwrap().as_f64(),
            Some(0.0),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/max_new_tokens").unwrap().as_u64(),
            Some(8),
            "{parsed}"
        );
    }

    #[test]
    fn require_reasoning_is_rejected_outside_qa_reasoning() {
        let fixture = fixture(
            "{\"document\":\"some held-out prose to summarize\",\"summary\":\"a summary\"}\n",
        );
        let mut arguments = args(&fixture, GenerationObjective::Summarization);
        arguments.require_reasoning = true;
        let error = evaluate(arguments).unwrap_err().to_string();
        assert!(error.contains("qa_reasoning"), "{error}");
    }

    #[test]
    fn a_report_path_naming_a_training_artifact_is_rejected() {
        let fixture = fixture(&qa_records(2));
        let mut arguments = args(&fixture, GenerationObjective::QaReasoning);
        arguments.output = Some(fixture.root.join("current.json"));
        let error = evaluate(arguments).unwrap_err().to_string();
        assert!(error.contains("current.json"), "{error}");
    }

    #[test]
    fn samples_cannot_overwrite_the_metrics_report() {
        let fixture = fixture(&qa_records(2));
        let mut arguments = args(&fixture, GenerationObjective::QaReasoning);
        arguments.samples = arguments.output.clone();
        let error = evaluate(arguments).unwrap_err().to_string();
        assert!(error.contains("different files"), "{error}");
    }
}
