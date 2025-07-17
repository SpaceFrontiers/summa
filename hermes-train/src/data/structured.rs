//! JSONL schemas and tokenization for task-aligned objectives.

use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;

use super::{EncodedText, TrainingSample, is_jsonl, open_data, required_string};
use crate::curriculum::ObjectiveConfig;

fn optional_string<'a>(
    value: &'a serde_json::Value,
    field: &str,
    path: &Path,
    line_number: usize,
) -> Result<Option<&'a str>> {
    match value.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => {
            let text = value.as_str().with_context(|| {
                format!(
                    "optional `{field}` at {}:{line_number} must be a string",
                    path.display()
                )
            })?;
            ensure!(
                !text.trim().is_empty(),
                "optional `{field}` at {}:{line_number} must not be empty",
                path.display()
            );
            Ok(Some(text))
        }
    }
}

fn encode_tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>> {
    Ok(tokenizer
        .encode(text, false)?
        .into_iter()
        .map(i64::from)
        .collect())
}

fn make_supervised_sample(
    tokenizer: &Tokenizer,
    prefix: &str,
    source: &str,
    suffix: &str,
    target: &str,
    seq_len: usize,
    source_required: bool,
) -> Result<TrainingSample> {
    let prefix = encode_tokens(tokenizer, prefix)?;
    let source = encode_tokens(tokenizer, source)?;
    let suffix = encode_tokens(tokenizer, suffix)?;
    let target = encode_tokens(tokenizer, target)?;
    ensure!(!target.is_empty(), "target tokenized to an empty sequence");
    if source_required {
        ensure!(!source.is_empty(), "source tokenized to an empty sequence");
    }

    let capacity = seq_len
        .checked_add(1)
        .context("sequence_length is too large to reserve the shifted target token")?;
    let fixed_tokens = prefix
        .len()
        .checked_add(suffix.len())
        .and_then(|tokens| tokens.checked_add(target.len()))
        .and_then(|tokens| tokens.checked_add(1))
        .context("supervised example token count overflows usize")?;
    ensure!(
        fixed_tokens <= capacity,
        "instruction, target marker, complete target, and EOS require {fixed_tokens} tokens but sequence_length {seq_len} provides {capacity}; targets are never truncated"
    );
    let kept_source = source.len().min(capacity - fixed_tokens);
    if source_required {
        ensure!(
            kept_source > 0,
            "sequence_length {seq_len} leaves no token for the source after reserving the target"
        );
    }
    let truncated_tokens = source.len() - kept_source;
    let prompt_len = prefix.len() + kept_source + suffix.len();
    ensure!(prompt_len > 0, "supervised prompt tokenized to empty");

    let mut tokens = Vec::with_capacity(capacity);
    tokens.extend(prefix);
    tokens.extend_from_slice(&source[..kept_source]);
    tokens.extend(suffix);
    tokens.extend(target.iter().copied());
    tokens.push(i64::from(tokenizer.eos_token_id()));
    tokens.resize(capacity, i64::from(tokenizer.eos_token_id()));

    // Combined token `i` is target position `i - 1` in the shifted batch.
    let loss_positions = (prompt_len - 1..prompt_len + target.len()).collect();
    Ok(TrainingSample::Supervised {
        tokens,
        loss_positions,
        truncated_tokens,
    })
}

fn encode_retrieval_text(
    tokenizer: &Tokenizer,
    prefix: &str,
    text: &str,
    seq_len: usize,
) -> Result<(EncodedText, usize)> {
    let prefix = encode_tokens(tokenizer, prefix)?;
    let content = encode_tokens(tokenizer, text)?;
    ensure!(!content.is_empty(), "retrieval text tokenized to empty");
    let fixed_tokens = prefix
        .len()
        .checked_add(1)
        .context("retrieval prefix token count overflows usize")?;
    ensure!(
        fixed_tokens < seq_len,
        "retrieval prefix and EOS leave no content room at sequence_length {seq_len}"
    );
    let kept_content = content.len().min(seq_len - fixed_tokens);
    let truncated_tokens = content.len() - kept_content;
    let mut tokens = Vec::with_capacity(seq_len);
    tokens.extend(prefix);
    tokens.extend_from_slice(&content[..kept_content]);
    tokens.push(i64::from(tokenizer.eos_token_id()));
    let end_position = tokens.len() - 1;
    tokens.resize(seq_len, i64::from(tokenizer.eos_token_id()));
    Ok((
        EncodedText {
            tokens,
            end_position,
        },
        truncated_tokens,
    ))
}

fn retrieval_negatives<'a>(
    value: &'a serde_json::Value,
    path: &Path,
    line_number: usize,
) -> Result<Vec<&'a str>> {
    let Some(value) = value.get("negatives") else {
        return Ok(Vec::new());
    };
    let values = value.as_array().with_context(|| {
        format!(
            "optional `negatives` at {}:{line_number} must be an array of strings",
            path.display()
        )
    })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let text = value.as_str().with_context(|| {
                format!(
                    "`negatives[{index}]` at {}:{line_number} must be a string",
                    path.display()
                )
            })?;
            ensure!(
                !text.trim().is_empty(),
                "`negatives[{index}]` at {}:{line_number} must not be empty",
                path.display()
            );
            Ok(text)
        })
        .collect()
}

fn structured_sample(
    value: &serde_json::Value,
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    path: &Path,
    line_number: usize,
) -> Result<TrainingSample> {
    match objective {
        ObjectiveConfig::CausalLm => unreachable!("causal data has a separate streaming path"),
        ObjectiveConfig::Summarization { instruction } => {
            let document = required_string(value, "document", path, line_number)?;
            let summary = required_string(value, "summary", path, line_number)?;
            make_supervised_sample(
                tokenizer,
                &format!("{instruction}\n\nDocument:\n"),
                document,
                "\n\nSummary:\n",
                summary,
                seq_len,
                true,
            )
            .with_context(|| format!("cannot encode {}:{line_number}", path.display()))
        }
        ObjectiveConfig::RetrievalPlanning { instruction } => {
            let request = required_string(value, "request", path, line_number)?;
            let plan = required_string(value, "plan", path, line_number)?;
            let context = optional_string(value, "context", path, line_number)?;
            let (prefix, source) = match context {
                Some(context) => (
                    format!("{instruction}\n\nRequest:\n{request}\n\nContext:\n"),
                    context,
                ),
                None => (format!("{instruction}\n\nRequest:\n{request}\n"), ""),
            };
            make_supervised_sample(
                tokenizer,
                &prefix,
                source,
                "\nPlan:\n",
                plan,
                seq_len,
                context.is_some(),
            )
            .with_context(|| format!("cannot encode {}:{line_number}", path.display()))
        }
        ObjectiveConfig::ContrastiveRetrieval {
            query_prefix,
            document_prefix,
            ..
        } => {
            let query = required_string(value, "query", path, line_number)?;
            let positive = required_string(value, "positive", path, line_number)?;
            let negatives = retrieval_negatives(value, path, line_number)?;
            let (query, mut truncated_tokens) =
                encode_retrieval_text(tokenizer, query_prefix, query, seq_len).with_context(
                    || format!("cannot encode query at {}:{line_number}", path.display()),
                )?;
            let (positive, truncated) =
                encode_retrieval_text(tokenizer, document_prefix, positive, seq_len).with_context(
                    || format!("cannot encode positive at {}:{line_number}", path.display()),
                )?;
            truncated_tokens = truncated_tokens
                .checked_add(truncated)
                .context("retrieval truncated-token count overflows usize")?;
            let document_count = negatives
                .len()
                .checked_add(1)
                .context("retrieval document count overflows usize")?;
            let mut documents = Vec::with_capacity(document_count);
            documents.push(positive);
            for (index, negative) in negatives.into_iter().enumerate() {
                let (negative, truncated) =
                    encode_retrieval_text(tokenizer, document_prefix, negative, seq_len)
                        .with_context(|| {
                            format!(
                                "cannot encode negative {index} at {}:{line_number}",
                                path.display()
                            )
                        })?;
                truncated_tokens = truncated_tokens
                    .checked_add(truncated)
                    .context("retrieval truncated-token count overflows usize")?;
                documents.push(negative);
            }
            Ok(TrainingSample::Retrieval {
                query,
                documents,
                truncated_tokens,
            })
        }
    }
}

pub(super) fn visit_structured_samples(
    path: &Path,
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    ensure!(
        is_jsonl(path),
        "objective `{}` requires .jsonl or .jsonl.zst data, got {}",
        objective.name(),
        path.display()
    );
    let mut reader = open_data(path)?;
    let mut line = String::new();
    let mut line_number = 0usize;
    let mut count = 0usize;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        line_number = line_number
            .checked_add(1)
            .context("JSONL line count overflows usize")?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSONL at {}:{line_number}", path.display()))?;
        let sample = structured_sample(&value, objective, tokenizer, seq_len, path, line_number)?;
        count = count
            .checked_add(1)
            .context("structured sample count overflows usize")?;
        if !visit(sample)? {
            break;
        }
    }
    Ok(count)
}
