//! Full-pool retrieval evaluation of a published checkpoint.
//!
//! `eval --objective contrastive_retrieval` scores a query against the other
//! rows of its own batch, so its candidate set is `batch_size` documents wide —
//! twelve, at the geometry the 300M run used. That metric saturated at 1.000 and
//! can no longer detect a regression. This command answers the question a search
//! engine actually asks: embed every distinct document the shards mention into
//! one pool, then rank each query's positive against the whole pool.
//!
//! Framing is delegated, never reconstructed. Prefixes come from
//! [`TaskConfig::RetrievalRepresentation`] defaults and encoding from
//! [`encode_retrieval_text`], so a pool embedding is computed from exactly the
//! prompt retrieval training used. Like `eval`, this command is forward-only: the
//! device is never `.autodiff()`, no token cache is written, and only an explicit
//! `--output` report is created.
//!
//! Design: `docs/retrieval-pool-eval.md`.

use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use burn::tensor::{Int, Tensor, TensorData};
use hermes_llm::{Tokenizer, Transformer};
use hermes_train::task::{TaskAdapter, TaskConfig, TaskExample};
use hermes_train::workflow::validate_retrieval_layer_for_model;
use serde::Serialize;

use crate::data::{EncodedText, PhaseDataBinding, encode_retrieval_text};
use crate::eval::{rank_with_index_tiebreak, task_defaults, validate_report_path, warn_operator};
use crate::{RetrievalPoolArgs, file_sha256, load_config};

/// Report schema version. Increment when a field changes meaning.
///
/// v2 replaced the flat `mean_rank`/`worst_rank` pair with a `ranks` block
/// carrying median, p90, p99, and a count beyond rank 100, because a maximum
/// alone cannot distinguish one pathological query from a systematically heavy
/// tail. v3 uses the same deterministic first-index tie-break as top-1 argmax.
const RETRIEVAL_POOL_REPORT_VERSION: u32 = 3;

/// One malformed record must not make evaluation allocate an unbounded line.
/// Matches the training pipeline's own per-record ceiling.
const MAX_RECORD_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Serialize)]
struct PoolDataSource {
    path: String,
    identity: String,
    records_read: usize,
    /// Queries this shard contributed. Zero for `--distractors` shards, which
    /// enlarge the pool without being asked about.
    queries: usize,
    /// Documents this shard introduced that no earlier shard had already
    /// contributed under the same `document_id`.
    documents_added: usize,
}

#[derive(Debug, Serialize)]
struct PoolCounts {
    /// Distinct `document_id`s embedded and ranked against.
    documents: usize,
    queries: usize,
    /// Documents reached only because a `--distractors` shard supplied them.
    from_distractors: usize,
}

#[derive(Debug, Serialize)]
struct PoolRetrievalMetrics {
    top1_accuracy: f64,
    mrr: f64,
    recall_at_k: f64,
    k: usize,
    /// Discounted gain at rank 10. With exactly one relevant document per query
    /// this is `1/log2(rank+1)` inside the cut-off and zero outside it.
    ndcg_at_10: f64,
    ranks: RankDistribution,
}

/// Where the positive actually landed, across queries.
///
/// The ratio metrics above absorb a handful of catastrophic ranks without
/// moving: on the 2652-document advanced pool two checkpoints agreed to three
/// decimals on top-1, MRR, recall@10, and nDCG while their worst ranks were 74
/// and 710. A maximum alone is one query wide, so percentiles and an
/// outside-the-first-page count are reported beside it to show whether a heavy
/// tail is systematic or a single outlier.
#[derive(Debug, Serialize)]
struct RankDistribution {
    mean: f64,
    median: usize,
    p90: usize,
    p99: usize,
    worst: usize,
    /// Queries whose positive fell outside the first 100 candidates — beyond any
    /// plausible re-ranking window, so effectively unretrieved.
    beyond_100: usize,
}

impl RankDistribution {
    /// `ranks` is consumed sorted ascending. Percentiles use the
    /// nearest-rank convention, which needs no interpolation and always names an
    /// observed rank.
    fn from_sorted(ranks: &[usize]) -> Result<Self> {
        ensure!(
            !ranks.is_empty(),
            "rank distribution needs at least one query"
        );
        let quantile = |fraction: f64| {
            let position = (fraction * ranks.len() as f64).ceil() as usize;
            ranks[position.saturating_sub(1).min(ranks.len() - 1)]
        };
        let rank_sum = ranks.iter().try_fold(0usize, |sum, rank| {
            sum.checked_add(*rank)
                .context("rank-distribution sum overflows usize")
        })?;
        Ok(Self {
            mean: rank_sum as f64 / ranks.len() as f64,
            median: quantile(0.5),
            p90: quantile(0.9),
            p99: quantile(0.99),
            worst: *ranks.last().context("sorted ranks are non-empty")?,
            beyond_100: ranks.iter().filter(|rank| **rank > 100).count(),
        })
    }
}

#[derive(Debug, Serialize)]
struct RetrievalPoolReport {
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
    batch_size: usize,
    retrieval_layer: Option<usize>,
    data: Vec<PoolDataSource>,
    pool: PoolCounts,
    /// Sequences embedded (pool documents plus queries) and their token cost.
    embedded_sequences: usize,
    compute_tokens: usize,
    truncated_tokens: usize,
    warnings: Vec<String>,
    retrieval: PoolRetrievalMetrics,
}

struct PooledQuery {
    encoded: EncodedText,
    /// Index into the pool of this query's own positive document.
    positive: usize,
}

/// Accumulates the deduplicated document pool and the query set.
#[derive(Default)]
struct PoolBuilder {
    documents: Vec<EncodedText>,
    /// `document_id` to pool index. Dedup lives here rather than beside the
    /// encodings so the same document never occupies two pool slots, which would
    /// let a query outrank a duplicate of its own positive.
    index: HashMap<String, usize>,
    queries: Vec<PooledQuery>,
    truncated_tokens: usize,
}

impl PoolBuilder {
    /// Insert a document under its id, returning its pool index and whether it
    /// was new. A repeated id must carry the same model-visible text; silently
    /// keeping the first would let a query be scored against the wrong positive.
    fn intern(&mut self, id: &str, encoded: EncodedText) -> Result<(usize, bool)> {
        if let Some(existing) = self.index.get(id) {
            let prior = &self.documents[*existing];
            ensure!(
                prior.end_position == encoded.end_position && prior.tokens == encoded.tokens,
                "document_id `{id}` maps to different model-visible text"
            );
            return Ok((*existing, false));
        }
        let position = self.documents.len();
        self.documents.push(encoded);
        self.index.insert(id.to_owned(), position);
        Ok((position, true))
    }
}

/// Read one retrieval record's ids and adapter-framed texts into the pool.
///
/// `collect_queries` is false for distractor shards: their documents enlarge the
/// pool, but nothing is asked about them.
fn absorb_record(
    builder: &mut PoolBuilder,
    objective: &TaskConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    value: &serde_json::Value,
    collect_queries: bool,
) -> Result<()> {
    let TaskExample::RetrievalRepresentation {
        query,
        documents,
        positive_index,
    } = objective.construct_example(value)?
    else {
        unreachable!("retrieval-representation adapter returned another example type")
    };
    ensure!(
        positive_index == 0,
        "retrieval adapter placed the positive at index {positive_index}; this command assumes index 0"
    );
    let positive_id = value
        .get("document_id")
        .and_then(serde_json::Value::as_str)
        .context("retrieval record has no string `document_id`; the pool is keyed by it")?;
    let negative_ids = match value.get("negative_document_ids") {
        Some(ids) => ids
            .as_array()
            .context("`negative_document_ids` is not an array")?
            .iter()
            .map(|id| {
                id.as_str()
                    .context("`negative_document_ids` contains a non-string id")
            })
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    let negatives = documents.len() - 1;
    ensure!(
        negative_ids.len() == negatives,
        "record has {negatives} negative document(s) but {} negative id(s); the pool cannot be keyed reliably",
        negative_ids.len()
    );

    let mut positive_position = None;
    for (offset, document) in documents.iter().enumerate() {
        let id = if offset == 0 {
            positive_id
        } else {
            negative_ids[offset - 1]
        };
        let (encoded, truncated) = encode_retrieval_text(tokenizer, document, seq_len)?;
        let (position, inserted) = builder.intern(id, encoded)?;
        if inserted {
            builder.truncated_tokens = builder
                .truncated_tokens
                .checked_add(truncated)
                .context("truncated-token count overflows usize")?;
        }
        if offset == 0 {
            positive_position = Some(position);
        }
    }
    let positive = match positive_position {
        Some(position) => position,
        // The positive was already pooled by an earlier record.
        None => *builder
            .index
            .get(positive_id)
            .context("interned positive document is missing from the pool index")?,
    };

    if collect_queries {
        let (encoded, truncated) = encode_retrieval_text(tokenizer, &query, seq_len)?;
        builder.truncated_tokens = builder
            .truncated_tokens
            .checked_add(truncated)
            .context("truncated-token count overflows usize")?;
        builder.queries.push(PooledQuery { encoded, positive });
    }
    Ok(())
}

/// Embed `texts` in `batch_size` chunks, returning a row-major `[texts, hidden]`
/// matrix on the device.
fn embed(
    model: &Transformer,
    texts: &[&EncodedText],
    seq_len: usize,
    batch_size: usize,
    layer: Option<usize>,
    device: &hermes_llm::Device,
) -> Result<Tensor<2>> {
    ensure!(!texts.is_empty(), "nothing to embed");
    let mut chunks = Vec::with_capacity(texts.len().div_ceil(batch_size));
    for chunk in texts.chunks(batch_size) {
        let mut tokens = Vec::with_capacity(chunk.len() * seq_len);
        let mut end_positions = Vec::with_capacity(chunk.len());
        for (row, text) in chunk.iter().enumerate() {
            ensure!(
                text.tokens.len() == seq_len && text.end_position < seq_len,
                "encoded retrieval text does not match sequence_length {seq_len}"
            );
            tokens.extend_from_slice(&text.tokens);
            // `forward_embeddings` selects from the flattened [batch, sequence]
            // hidden states, exactly as `make_batch` does for training.
            end_positions.push(
                i64::try_from(row * seq_len + text.end_position)
                    .context("retrieval end position exceeds i64")?,
            );
        }
        let input_ids: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(tokens, [chunk.len(), seq_len]), device);
        let ends: Tensor<1, Int> =
            Tensor::from_data(TensorData::new(end_positions, [chunk.len()]), device);
        chunks.push(model.forward_embeddings(input_ids, ends, layer));
    }
    Ok(Tensor::cat(chunks, 0))
}

pub(super) fn evaluate(args: RetrievalPoolArgs) -> Result<()> {
    let started = Instant::now();
    if let Some(output) = &args.output {
        validate_report_path(output)?;
    }
    ensure!(
        args.sequence_length > 0,
        "--sequence-length must be positive"
    );
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    ensure!(args.recall_k > 0, "--recall-k must be positive");

    let mut objective = task_defaults("retrieval_representation")?;
    let TaskConfig::RetrievalRepresentation { layer, .. } = &mut objective else {
        unreachable!("retrieval_representation deserialized to another task");
    };
    *layer = args.retrieval_layer;

    let mut warnings = Vec::new();
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
    validate_retrieval_layer_for_model(
        args.retrieval_layer.unwrap_or(config.num_layers),
        &config,
        "retrieval-pool-eval",
    )?;

    // Forward-only by construction, as in `eval`: never `.autodiff()`, never
    // `prepare_inference()`d, so these numbers stay comparable with the
    // in-batch retrieval metric and a live run remains resumable.
    let device = hermes_llm::default_device();
    let mut model = Transformer::new(&config, &device)?;
    hermes_llm::load_safetensors(&mut model, &args.checkpoint).with_context(|| {
        format!(
            "checkpoint {} does not match model configuration {}",
            args.checkpoint.display(),
            args.config.display()
        )
    })?;

    let mut builder = PoolBuilder::default();
    let mut sources = Vec::with_capacity(args.data.len() + args.distractors.len());
    // Query shards are read first so every document a query can be asked about
    // is pooled before distractors enlarge the pool; `documents_from_queries`
    // then separates the two contributions in the report.
    let shards = args
        .data
        .iter()
        .map(|path| (path, true))
        .chain(args.distractors.iter().map(|path| (path, false)));
    let mut documents_from_queries = 0;
    for (path, collect_queries) in shards {
        let documents_before = builder.documents.len();
        let queries_before = builder.queries.len();
        let mut records_read = 0usize;
        let binding = PhaseDataBinding::open(path)?;
        binding.with_readers(path, |source_path, reader| {
            let mut line = String::new();
            loop {
                line.clear();
                // Re-borrow per line so the byte ceiling applies to each record
                // rather than to the whole shard. Spelled through `Read::take`
                // because the method needs a sized `Self`, which the reborrowed
                // reference is and the bare trait object is not.
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
                absorb_record(
                    &mut builder,
                    &objective,
                    &tokenizer,
                    args.sequence_length,
                    &value,
                    collect_queries,
                )
                .with_context(|| format!("cannot pool {}:{records_read}", source_path.display()))?;
            }
            Ok(true)
        })?;
        if collect_queries {
            documents_from_queries = builder.documents.len();
        }
        sources.push(PoolDataSource {
            path: path.display().to_string(),
            identity: binding.signature_identity().to_owned(),
            records_read,
            queries: builder.queries.len() - queries_before,
            documents_added: builder.documents.len() - documents_before,
        });
    }
    ensure!(
        !builder.queries.is_empty(),
        "--data produced no retrieval queries"
    );
    ensure!(
        builder.documents.len() > 1,
        "the candidate pool holds {} document(s); ranking needs at least two",
        builder.documents.len()
    );
    if builder.documents.len() <= args.batch_size {
        warn_operator(
            &mut warnings,
            format!(
                "pool holds only {} document(s), no more than --batch-size {}; this is no harder than the in-batch `eval` metric",
                builder.documents.len(),
                args.batch_size
            ),
        );
    }

    let document_texts: Vec<&EncodedText> = builder.documents.iter().collect();
    let pool = embed(
        &model,
        &document_texts,
        args.sequence_length,
        args.batch_size,
        args.retrieval_layer,
        &device,
    )?;
    // Embeddings are L2-normalized by `forward_embeddings`, so this dot product
    // is cosine similarity and ranking is invariant to the loss temperature.
    let pool = pool.transpose();

    let mut reciprocal_rank = 0.0;
    let mut top1 = 0usize;
    let mut within_k = 0usize;
    let mut ndcg = 0.0;
    // Every rank is retained so the report can describe the tail, not just its
    // maximum. One usize per query is negligible beside the embedding matrix.
    let mut ranks = Vec::with_capacity(builder.queries.len());
    let pool_size = builder.documents.len();
    for chunk in builder.queries.chunks(args.batch_size) {
        let texts: Vec<&EncodedText> = chunk.iter().map(|query| &query.encoded).collect();
        let queries = embed(
            &model,
            &texts,
            args.sequence_length,
            args.batch_size,
            args.retrieval_layer,
            &device,
        )?;
        let scores = queries.matmul(pool.clone());
        let [rows, candidates] = scores.dims();
        ensure!(
            rows == chunk.len() && candidates == pool_size,
            "similarity matrix is {rows}x{candidates} for {} queries and {pool_size} documents",
            chunk.len()
        );
        let scores = scores.into_data().convert::<f32>().to_vec::<f32>()?;
        for (row, query) in chunk.iter().enumerate() {
            let row = &scores[row * candidates..(row + 1) * candidates];
            let rank = rank_with_index_tiebreak(row, query.positive)?;
            reciprocal_rank += 1.0 / rank as f64;
            top1 += usize::from(rank == 1);
            within_k += usize::from(rank <= args.recall_k);
            if rank <= 10 {
                ndcg += 1.0 / ((rank + 1) as f64).log2();
            }
            ranks.push(rank);
        }
    }
    ranks.sort_unstable();

    let queries = builder.queries.len();
    let sequences = pool_size
        .checked_add(queries)
        .context("embedded sequence count overflows usize")?;
    let report = RetrievalPoolReport {
        version: RETRIEVAL_POOL_REPORT_VERSION,
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
        batch_size: args.batch_size,
        retrieval_layer: args.retrieval_layer,
        data: sources,
        pool: PoolCounts {
            documents: pool_size,
            queries,
            from_distractors: pool_size - documents_from_queries,
        },
        embedded_sequences: sequences,
        compute_tokens: sequences
            .checked_mul(args.sequence_length)
            .context("compute-token count overflows usize")?,
        truncated_tokens: builder.truncated_tokens,
        warnings,
        retrieval: PoolRetrievalMetrics {
            top1_accuracy: top1 as f64 / queries as f64,
            mrr: reciprocal_rank / queries as f64,
            recall_at_k: within_k as f64 / queries as f64,
            k: args.recall_k,
            ndcg_at_10: ndcg / queries as f64,
            ranks: RankDistribution::from_sorted(&ranks)?,
        },
    };
    write_report(&report, args.output.as_deref())?;
    print_summary(&report, started.elapsed().as_secs_f64());
    Ok(())
}

fn write_report(report: &RetrievalPoolReport, output: Option<&Path>) -> Result<()> {
    let Some(output) = output else {
        return Ok(());
    };
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create report directory {}", parent.display()))?;
    }
    let mut encoded = serde_json::to_vec_pretty(report)?;
    encoded.push(b'\n');
    std::fs::write(output, encoded)
        .with_context(|| format!("cannot write evaluation report {}", output.display()))
}

fn print_summary(report: &RetrievalPoolReport, elapsed_seconds: f64) {
    println!("objective            {}", report.objective);
    println!(
        "checkpoint           {} ({})",
        report.checkpoint, report.checkpoint_sha256
    );
    println!(
        "pool                 {} document(s), {} quer(ies){}",
        report.pool.documents,
        report.pool.queries,
        if report.pool.from_distractors > 0 {
            format!(", {} from --distractors", report.pool.from_distractors)
        } else {
            String::new()
        }
    );
    println!(
        "embedded             {} sequence(s), {} compute token(s), {} truncated",
        report.embedded_sequences, report.compute_tokens, report.truncated_tokens
    );
    let retrieval = &report.retrieval;
    println!("top-1 accuracy       {:.6}", retrieval.top1_accuracy);
    println!("mrr                  {:.6}", retrieval.mrr);
    println!(
        "{:<20} {:.6}",
        format!("recall@{}", retrieval.k),
        retrieval.recall_at_k
    );
    println!("ndcg@10              {:.6}", retrieval.ndcg_at_10);
    let ranks = &retrieval.ranks;
    println!(
        "rank                 mean {:.2}, median {}, p90 {}, p99 {}, worst {} of {}",
        ranks.mean, ranks.median, ranks.p90, ranks.p99, ranks.worst, report.pool.documents
    );
    println!(
        "beyond rank 100      {} of {} quer(ies)",
        ranks.beyond_100, report.pool.queries
    );
    for warning in &report.warnings {
        println!("warning              {warning}");
    }
    println!("elapsed              {elapsed_seconds:.2}s");
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use hermes_llm::parse_mal;

    use super::*;
    use crate::data::write_test_tokenizer;

    const POOL_MODEL: &str = r#"
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

    /// Wide enough for the retrieval task's fixed document prefix (38 bytes under
    /// the merge-free byte-level test tokenizer), the record text, and EOS.
    const POOL_SEQUENCE_LENGTH: usize = 56;

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
        fs::write(&config, POOL_MODEL).unwrap();
        write_test_tokenizer(&root);
        let device = hermes_llm::default_device();
        device.seed(7);
        let model = Transformer::new(&parse_mal(POOL_MODEL).unwrap(), &device).unwrap();
        let checkpoint = root.join("weights.safetensors");
        hermes_llm::save_safetensors(&model, &checkpoint).unwrap();
        let data = root.join("holdout.jsonl");
        fs::write(&data, records).unwrap();
        Fixture {
            _directory: directory,
            tokenizer: root.join("tokenizer.json"),
            output: root.join("reports/pool.json"),
            root,
            config,
            checkpoint,
            data,
        }
    }

    fn args(fixture: &Fixture, batch_size: usize) -> RetrievalPoolArgs {
        RetrievalPoolArgs {
            config: fixture.config.clone(),
            tokenizer: fixture.tokenizer.clone(),
            checkpoint: fixture.checkpoint.clone(),
            data: vec![fixture.data.clone()],
            distractors: Vec::new(),
            sequence_length: POOL_SEQUENCE_LENGTH,
            batch_size,
            recall_k: 2,
            retrieval_layer: None,
            output: Some(fixture.output.clone()),
        }
    }

    /// Each record owns a distinct positive and a distinct negative, so `rows`
    /// records contribute exactly `2 * rows` pool documents.
    fn records(rows: usize) -> String {
        (0..rows)
            .map(|index| {
                format!(
                    "{{\"query\":\"query {index}\",\"positive\":\"answer {index}\",\"document_id\":\"pos-{index}\",\"negatives\":[\"other {index}\"],\"negative_document_ids\":[\"neg-{index}\"]}}\n"
                )
            })
            .collect()
    }

    fn report(path: &Path) -> serde_json::Value {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }

    fn number(value: &serde_json::Value, pointer: &str) -> f64 {
        let number = value
            .pointer(pointer)
            .unwrap_or_else(|| panic!("report has no {pointer}: {value}"))
            .as_f64()
            .unwrap_or_else(|| panic!("{pointer} is not a JSON number: {value}"));
        assert!(number.is_finite(), "{pointer} is {number}");
        number
    }

    /// The property that separates this command from `eval`: the candidate set is
    /// the whole pool, so `--batch-size` changes only how many sequences are
    /// embedded per forward pass and must not move any reported metric. A
    /// per-batch implementation would score against `batch_size` candidates and
    /// disagree here — and could never report a rank above `batch_size`.
    #[test]
    fn pool_metrics_are_invariant_to_batch_size() {
        let fixture = fixture(&records(6));
        let mut measured = Vec::new();
        for batch_size in [2, 5, 12] {
            evaluate(args(&fixture, batch_size)).unwrap();
            let parsed = report(&fixture.output);
            assert_eq!(
                parsed.pointer("/pool/documents").unwrap().as_u64(),
                Some(12),
                "{parsed}"
            );
            assert_eq!(
                parsed.pointer("/pool/queries").unwrap().as_u64(),
                Some(6),
                "{parsed}"
            );
            measured.push((
                number(&parsed, "/retrieval/top1_accuracy"),
                number(&parsed, "/retrieval/mrr"),
                number(&parsed, "/retrieval/recall_at_k"),
                number(&parsed, "/retrieval/ndcg_at_10"),
                parsed
                    .pointer("/retrieval/ranks/worst")
                    .unwrap()
                    .as_u64()
                    .unwrap(),
            ));
        }
        for other in &measured[1..] {
            assert_eq!(measured[0].4, other.4, "worst rank moved with batch size");
            for (first, second) in [
                (measured[0].0, other.0),
                (measured[0].1, other.1),
                (measured[0].2, other.2),
                (measured[0].3, other.3),
            ] {
                assert!(
                    (first - second).abs() < 1e-9,
                    "metric moved with batch size: {first} vs {second}"
                );
            }
        }
        // A rank above the smallest batch size proves candidates outside the
        // query's own forward pass were ranked against it.
        assert!(
            measured[0].4 > 2,
            "worst rank {} never left the smallest batch",
            measured[0].4
        );
    }

    #[test]
    fn pool_metrics_are_finite_ordered_and_deterministic() {
        let fixture = fixture(&records(6));
        evaluate(args(&fixture, 4)).unwrap();
        let first = report(&fixture.output);
        evaluate(args(&fixture, 4)).unwrap();
        assert_eq!(
            first,
            report(&fixture.output),
            "report is not deterministic"
        );

        let top1 = number(&first, "/retrieval/top1_accuracy");
        let mrr = number(&first, "/retrieval/mrr");
        let recall = number(&first, "/retrieval/recall_at_k");
        let ndcg = number(&first, "/retrieval/ndcg_at_10");
        let mean_rank = number(&first, "/retrieval/ranks/mean");
        for (name, value) in [
            ("top1", top1),
            ("mrr", mrr),
            ("recall", recall),
            ("ndcg", ndcg),
        ] {
            assert!((0.0..=1.0).contains(&value), "{name} is {value}");
        }
        assert!(mrr >= top1 - 1e-9, "mrr {mrr} below top-1 {top1}");
        assert!(recall >= top1 - 1e-9, "recall {recall} below top-1 {top1}");
        assert!(mean_rank >= 1.0, "mean rank {mean_rank} below one");
        let worst = first
            .pointer("/retrieval/ranks/worst")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!((1..=12).contains(&worst), "worst rank {worst} outside pool");
    }

    /// The pool is keyed by `document_id`, so a document two records share must
    /// occupy one slot. Two slots would let a query rank a duplicate of its own
    /// positive above it and silently inflate every metric.
    /// A maximum cannot tell one pathological query from a heavy tail, which is
    /// why v2 reports percentiles beside it. Percentiles use nearest-rank, so
    /// every reported value is a rank some query actually achieved.
    #[test]
    fn rank_distribution_describes_the_tail_not_just_its_maximum() {
        // 100 queries: 99 at rank 1, one catastrophic. Ratio metrics would barely
        // move; the distribution must show the max without implying a bad median.
        let mut one_outlier = vec![1usize; 99];
        one_outlier.push(710);
        one_outlier.sort_unstable();
        let outlier = RankDistribution::from_sorted(&one_outlier).unwrap();
        assert_eq!(outlier.median, 1);
        assert_eq!(outlier.p90, 1);
        assert_eq!(outlier.worst, 710);
        assert_eq!(outlier.beyond_100, 1);

        // Same maximum, but the tail is systematic rather than a single query.
        let mut heavy: Vec<usize> = (0..80).map(|_| 1).collect();
        heavy.extend((0..20).map(|index| 300 + index));
        heavy.sort_unstable();
        let heavy = RankDistribution::from_sorted(&heavy).unwrap();
        assert_eq!(heavy.median, 1);
        assert!(
            heavy.p90 >= 300,
            "p90 must expose a 20% heavy tail: {heavy:?}"
        );
        assert_eq!(heavy.beyond_100, 20);
        assert!(
            heavy.mean > outlier.mean,
            "a systematic tail must outweigh one outlier: {heavy:?} vs {outlier:?}"
        );

        // Degenerate input is rejected rather than reported as zero.
        assert!(RankDistribution::from_sorted(&[]).is_err());

        // A single query is representable: every percentile is that rank.
        let single = RankDistribution::from_sorted(&[4]).unwrap();
        assert_eq!(
            (single.median, single.p90, single.p99, single.worst),
            (4, 4, 4, 4)
        );
    }

    #[test]
    fn repeated_document_ids_collapse_to_one_pool_entry() {
        let shared = (0..4)
            .map(|index| {
                format!(
                    "{{\"query\":\"query {index}\",\"positive\":\"answer {index}\",\"document_id\":\"pos-{index}\",\"negatives\":[\"shared distractor\"],\"negative_document_ids\":[\"neg-shared\"]}}\n"
                )
            })
            .collect::<String>();
        let fixture = fixture(&shared);
        evaluate(args(&fixture, 3)).unwrap();
        let parsed = report(&fixture.output);
        // Four positives plus one shared negative, not four negatives.
        assert_eq!(
            parsed.pointer("/pool/documents").unwrap().as_u64(),
            Some(5),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/data/0/documents_added").unwrap().as_u64(),
            Some(5),
            "{parsed}"
        );
    }

    #[test]
    fn repeated_document_id_with_different_text_is_rejected() {
        let conflicting = concat!(
            "{\"query\":\"query a\",\"positive\":\"first visible text\",\"document_id\":\"shared\",\"negatives\":[\"other a\"],\"negative_document_ids\":[\"neg-a\"]}\n",
            "{\"query\":\"query b\",\"positive\":\"different visible text\",\"document_id\":\"shared\",\"negatives\":[\"other b\"],\"negative_document_ids\":[\"neg-b\"]}\n",
        );
        let fixture = fixture(conflicting);
        let error = format!("{:#}", evaluate(args(&fixture, 2)).unwrap_err());
        assert!(error.contains("different model-visible text"), "{error}");
    }

    /// A query whose positive was already pooled as another record's negative
    /// must still resolve to its own document rather than failing to find it.
    #[test]
    fn a_positive_already_pooled_as_a_negative_still_resolves() {
        let overlapping = concat!(
            "{\"query\":\"query a\",\"positive\":\"answer a\",\"document_id\":\"doc-a\",\"negatives\":[\"answer b\"],\"negative_document_ids\":[\"doc-b\"]}\n",
            "{\"query\":\"query b\",\"positive\":\"answer b\",\"document_id\":\"doc-b\",\"negatives\":[\"answer a\"],\"negative_document_ids\":[\"doc-a\"]}\n",
        );
        let fixture = fixture(overlapping);
        evaluate(args(&fixture, 2)).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(
            parsed.pointer("/pool/documents").unwrap().as_u64(),
            Some(2),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/pool/queries").unwrap().as_u64(),
            Some(2),
            "{parsed}"
        );
        // With two documents and two queries every rank is 1 or 2.
        let worst = parsed
            .pointer("/retrieval/ranks/worst")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(worst <= 2, "{parsed}");
    }

    /// Distractor shards must enlarge the candidate pool without being asked
    /// about, and the report must attribute them separately.
    #[test]
    fn distractor_shards_add_documents_without_adding_queries() {
        let fixture = fixture(&records(3));
        let distractors = fixture.root.join("distractors.jsonl");
        let extra = (10..14)
            .map(|index| {
                format!(
                    "{{\"query\":\"query {index}\",\"positive\":\"answer {index}\",\"document_id\":\"pos-{index}\",\"negatives\":[\"other {index}\"],\"negative_document_ids\":[\"neg-{index}\"]}}\n"
                )
            })
            .collect::<String>();
        fs::write(&distractors, extra).unwrap();
        let mut arguments = args(&fixture, 4);
        arguments.distractors = vec![distractors];
        evaluate(arguments).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(
            parsed.pointer("/pool/queries").unwrap().as_u64(),
            Some(3),
            "{parsed}"
        );
        // Six from the query shard, eight from the distractor shard.
        assert_eq!(
            parsed.pointer("/pool/documents").unwrap().as_u64(),
            Some(14),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/pool/from_distractors").unwrap().as_u64(),
            Some(8),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/data/1/queries").unwrap().as_u64(),
            Some(0),
            "{parsed}"
        );
    }

    /// Prefixes must come from the task adapter, not be rebuilt here: a prompt
    /// assembled locally would be out of distribution and report a fake number.
    #[test]
    fn report_pins_the_adapter_prefixes_and_read_out_layer() {
        let fixture = fixture(&records(4));
        let mut arguments = args(&fixture, 4);
        arguments.retrieval_layer = Some(2);
        evaluate(arguments).unwrap();
        let parsed = report(&fixture.output);
        assert_eq!(
            parsed.pointer("/task/query_prefix").unwrap().as_str(),
            Some("Represent this query for retrieval:\n"),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/task/document_prefix").unwrap().as_str(),
            Some("Represent this document for retrieval:\n"),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/task/layer").unwrap().as_u64(),
            Some(2),
            "{parsed}"
        );
        assert_eq!(
            parsed.pointer("/retrieval_layer").unwrap().as_u64(),
            Some(2),
            "{parsed}"
        );
    }

    #[test]
    fn a_read_out_layer_outside_the_model_is_rejected() {
        let fixture = fixture(&records(4));
        let mut arguments = args(&fixture, 4);
        arguments.retrieval_layer = Some(3);
        let error = evaluate(arguments).unwrap_err().to_string();
        assert!(error.contains("retrieval"), "{error}");
    }

    /// A record whose negatives and negative ids disagree cannot be keyed into
    /// the pool, and must fail loudly rather than be silently dropped.
    #[test]
    fn mismatched_negative_ids_are_rejected() {
        let fixture = fixture(
            "{\"query\":\"q\",\"positive\":\"p\",\"document_id\":\"doc\",\"negatives\":[\"a\",\"b\"],\"negative_document_ids\":[\"only-one\"]}\n",
        );
        let error = format!("{:#}", evaluate(args(&fixture, 1)).unwrap_err());
        assert!(error.contains("negative id"), "{error}");
    }

    /// A record with no `document_id` cannot be pooled: without it two distinct
    /// documents could collapse or one could be duplicated.
    #[test]
    fn a_record_without_a_document_id_is_rejected() {
        let fixture = fixture(
            "{\"query\":\"q\",\"positive\":\"p\",\"negatives\":[\"a\"],\"negative_document_ids\":[\"n\"]}\n",
        );
        let error = format!("{:#}", evaluate(args(&fixture, 1)).unwrap_err());
        assert!(error.contains("document_id"), "{error}");
    }

    /// A pool no larger than the batch is no harder than the in-batch metric this
    /// command exists to replace, so it must say so rather than look like a
    /// stronger result than it is.
    #[test]
    fn a_pool_no_larger_than_the_batch_warns() {
        let fixture = fixture(&records(2));
        evaluate(args(&fixture, 8)).unwrap();
        let parsed = report(&fixture.output);
        let warnings = parsed.pointer("/warnings").unwrap().as_array().unwrap();
        assert!(
            warnings
                .iter()
                .any(|warning| warning.as_str().unwrap_or_default().contains("in-batch")),
            "{parsed}"
        );
    }

    #[test]
    fn a_report_path_naming_a_training_artifact_is_rejected() {
        let fixture = fixture(&records(4));
        let mut arguments = args(&fixture, 4);
        arguments.output = Some(fixture.root.join("weights.safetensors"));
        let error = evaluate(arguments).unwrap_err().to_string();
        assert!(error.contains("weights.safetensors"), "{error}");
    }
}
