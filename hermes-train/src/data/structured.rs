//! JSONL schemas and tokenization for task-aligned objectives.

use std::cell::Cell;
use std::fmt;
use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;
use hermes_train::task::{
    SegmentedText, SupervisedPromptSegments, TaskAdapter, TaskConfig, TaskExample,
};
use tracing::warn;

use super::{EncodedText, TrainingSample, is_jsonl, read_training_jsonl_record};

/// A supervised-generation record whose prompt framing, complete target, and
/// EOS cannot fit `sequence_length`. Targets are never truncated, so such a
/// record has no valid encoding at this geometry.
///
/// It is a distinct error type rather than a plain message so that a caller can
/// tell "this record does not fit" apart from "this record is malformed"
/// without matching on strings. Training treats both as fatal; forward-only
/// evaluation skips and counts only this one (see [`OversizedRecordPolicy`]).
#[derive(Debug)]
pub(crate) struct OversizedSupervisedRecord {
    /// Tokens the un-truncatable framing needs: prefix, suffix, target, EOS,
    /// and — when the task requires a source — one source token.
    required_tokens: usize,
    /// Tokens the geometry provides, which is `sequence_length + 1` because the
    /// batch is shifted by one position.
    capacity: usize,
    sequence_length: usize,
}

impl fmt::Display for OversizedSupervisedRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "instruction, target marker, complete target, EOS, and any required source token need {} tokens but sequence_length {} provides {}; targets are never truncated",
            self.required_tokens, self.sequence_length, self.capacity
        )
    }
}

impl std::error::Error for OversizedSupervisedRecord {}

/// What to do with a record that [`OversizedSupervisedRecord`] rejects.
///
/// Training must abort: a phase that silently dropped part of its own dataset
/// would report a loss over an unknown subset of it, and the geometry is the
/// operator's to fix. Forward-only evaluation must not, because one over-long
/// held-out record would otherwise make an entire evaluation fail; it skips the
/// record and counts it so the drop is reported rather than silent.
#[derive(Clone, Copy)]
pub(crate) enum OversizedRecordPolicy<'a> {
    Abort,
    Skip(&'a Cell<usize>),
}

impl OversizedRecordPolicy<'_> {
    /// Decide whether streaming may continue past `error`. Returns the error
    /// unchanged unless it is an oversized record this policy absorbs.
    fn absorb(&self, error: anyhow::Error) -> Result<()> {
        let Self::Skip(skipped) = self else {
            return Err(error);
        };
        if !error
            .chain()
            .any(|cause| cause.is::<OversizedSupervisedRecord>())
        {
            return Err(error);
        }
        warn!("skipping oversized record: {error:#}");
        skipped.set(
            skipped
                .get()
                .checked_add(1)
                .context("oversized record count overflows usize")?,
        );
        Ok(())
    }
}

fn supervised_prompt_segments<'a>(
    value: &'a serde_json::Value,
    task: &TaskConfig,
    path: &Path,
    line_number: usize,
) -> Result<SupervisedPromptSegments<'a>> {
    task.construct_supervised_prompt(value).with_context(|| {
        format!(
            "invalid `{}` record at {}:{line_number}",
            task.name(),
            path.display()
        )
    })
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
    // A record needs the whole framing plus, when the task requires a source,
    // at least one source token. Both shortfalls mean the same thing — this
    // record has no valid encoding at this geometry — so both raise the typed
    // error an evaluation can skip and count.
    let required_tokens = fixed_tokens
        .checked_add(usize::from(source_required))
        .context("supervised example token count overflows usize")?;
    if required_tokens > capacity {
        return Err(anyhow::Error::new(OversizedSupervisedRecord {
            required_tokens,
            capacity,
            sequence_length: seq_len,
        }));
    }
    let kept_source = source.len().min(capacity - fixed_tokens);
    debug_assert!(!source_required || kept_source > 0);
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
    text: &SegmentedText,
    seq_len: usize,
) -> Result<(EncodedText, usize)> {
    text.validate()?;
    let encoded = text
        .segments
        .iter()
        .map(|segment| encode_tokens(tokenizer, segment))
        .collect::<Result<Vec<_>>>()?;
    let fixed_tokens = encoded
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != text.truncatable_segment)
        .try_fold(1usize, |total, (_, segment)| {
            total
                .checked_add(segment.len())
                .context("retrieval fixed-token count overflows usize")
        })?;
    ensure!(
        fixed_tokens <= seq_len,
        "retrieval fixed segments and EOS require {fixed_tokens} tokens but sequence_length is {seq_len}"
    );
    let (kept_truncatable, truncated_tokens) = match text.truncatable_segment {
        Some(index) => {
            let available = seq_len - fixed_tokens;
            let kept = encoded[index].len().min(available);
            if text.truncatable_segment_required {
                ensure!(
                    kept > 0,
                    "retrieval fixed segments and EOS leave no token for required segment {index} at sequence_length {seq_len}"
                );
            }
            (kept, encoded[index].len() - kept)
        }
        None => (0, 0),
    };
    let mut tokens = Vec::with_capacity(seq_len);
    for (index, segment) in encoded.into_iter().enumerate() {
        if Some(index) == text.truncatable_segment {
            tokens.extend_from_slice(&segment[..kept_truncatable]);
        } else {
            tokens.extend(segment);
        }
    }
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

fn structured_sample(
    value: &serde_json::Value,
    objective: &TaskConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    path: &Path,
    line_number: usize,
) -> Result<TrainingSample> {
    match objective {
        TaskConfig::CausalLm {} => unreachable!("causal data has a separate streaming path"),
        TaskConfig::Summarization { .. }
        | TaskConfig::RetrievalPlanning { .. }
        | TaskConfig::InstructionTuning { .. }
        | TaskConfig::QaReasoning { .. } => {
            let segments = supervised_prompt_segments(value, objective, path, line_number)?;
            make_supervised_sample(
                tokenizer,
                &segments.prefix,
                &segments.source,
                &segments.suffix,
                &segments.target,
                seq_len,
                segments.source_required,
            )
            .with_context(|| format!("cannot encode {}:{line_number}", path.display()))
        }
        TaskConfig::RetrievalRepresentation { .. } => {
            let TaskExample::RetrievalRepresentation {
                query,
                documents: task_documents,
                positive_index,
            } = objective.construct_example(value).with_context(|| {
                format!(
                    "invalid `{}` record at {}:{line_number}",
                    objective.name(),
                    path.display()
                )
            })?
            else {
                unreachable!("retrieval-representation adapter returned another example type")
            };
            ensure!(
                positive_index < task_documents.len(),
                "retrieval task adapter returned an out-of-range positive index"
            );
            let (query, mut truncated_tokens) = encode_retrieval_text(tokenizer, &query, seq_len)
                .with_context(|| {
                format!("cannot encode query at {}:{line_number}", path.display())
            })?;
            let mut documents = Vec::with_capacity(task_documents.len());
            for (index, document) in task_documents.iter().enumerate() {
                let (document, truncated) = encode_retrieval_text(tokenizer, document, seq_len)
                    .with_context(|| {
                        format!(
                            "cannot encode retrieval document {index} at {}:{line_number}",
                            path.display()
                        )
                    })?;
                truncated_tokens = truncated_tokens
                    .checked_add(truncated)
                    .context("retrieval truncated-token count overflows usize")?;
                documents.push(document);
            }
            documents.swap(0, positive_index);
            Ok(TrainingSample::Retrieval {
                query,
                documents,
                truncated_tokens,
            })
        }
        TaskConfig::RetrievalRanking { .. }
        | TaskConfig::PairwisePreference {}
        | TaskConfig::VerifiableRl { .. } => {
            unreachable!("wake projection rejects tasks unsupported by structured training")
        }
    }
}

pub(super) fn visit_structured_samples(
    path: &Path,
    reader: &mut dyn BufRead,
    objective: &TaskConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    oversized: OversizedRecordPolicy<'_>,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    ensure!(
        is_jsonl(path),
        "objective `{}` requires .jsonl or .jsonl.zst data, got {}",
        objective.name(),
        path.display()
    );
    let mut line = Vec::new();
    let mut line_number = 0usize;
    let mut count = 0usize;
    loop {
        let next_line_number = line_number
            .checked_add(1)
            .context("JSONL line count overflows usize")?;
        let Some(line) = read_training_jsonl_record(reader, &mut line, path, next_line_number)?
        else {
            break;
        };
        line_number = next_line_number;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("invalid JSONL at {}:{line_number}", path.display()))?;
        let sample =
            match structured_sample(&value, objective, tokenizer, seq_len, path, line_number) {
                Ok(sample) => sample,
                Err(error) => {
                    oversized.absorb(error)?;
                    continue;
                }
            };
        count = count
            .checked_add(1)
            .context("structured sample count overflows usize")?;
        if !visit(sample)? {
            break;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hermes_train::task::TaskExample;
    use serde_json::json;

    use super::*;

    fn assert_supervised_contract(
        task: TaskConfig,
        record: serde_json::Value,
        expected: SupervisedPromptSegments<'_>,
    ) {
        let segments =
            supervised_prompt_segments(&record, &task, Path::new("contract.jsonl"), 7).unwrap();
        assert_eq!(segments, expected);
        let expected_prompt = expected.prompt();
        let expected_segments = vec![
            expected.prefix.into_owned(),
            expected.source.into_owned(),
            expected.suffix.into_owned(),
        ];
        let expected_target = expected.target.into_owned();
        let TaskExample::SupervisedGeneration { prompt, target } =
            task.construct_example(&record).unwrap()
        else {
            panic!(
                "wake task `{}` returned the wrong example type",
                task.name()
            );
        };
        assert_eq!(prompt.segments, expected_segments);
        assert_eq!(prompt.render(), expected_prompt);
        assert_eq!(prompt.truncatable_segment, Some(1));
        assert_eq!(
            prompt.truncatable_segment_required,
            expected.source_required
        );
        assert_eq!(target, expected_target);
    }

    #[test]
    fn wake_uses_task_adapter_supervised_prompt_goldens() {
        assert_supervised_contract(
            TaskConfig::Summarization {
                instruction: "Summarize faithfully.".into(),
            },
            json!({"document": "Long source.", "summary": "Short target."}),
            SupervisedPromptSegments {
                prefix: "Summarize faithfully.\n\nDocument:\n".into(),
                source: "Long source.".into(),
                suffix: "\n\nSummary:\n".into(),
                target: "Short target.".into(),
                source_required: true,
            },
        );

        for (context, expected_source, expected_suffix, source_required) in [
            (
                Some("Official documentation."),
                "Official documentation.",
                "\nPlan:\n",
                true,
            ),
            (None, "", "Plan:\n", false),
        ] {
            let mut record = json!({"request": "Find the API.", "plan": "Search, then read."});
            if let Some(context) = context {
                record["context"] = json!(context);
            }
            assert_supervised_contract(
                TaskConfig::RetrievalPlanning {
                    instruction: "Plan retrieval.".into(),
                },
                record,
                SupervisedPromptSegments {
                    prefix: if source_required {
                        "Plan retrieval.\n\nRequest:\nFind the API.\n\nContext:\n".into()
                    } else {
                        "Plan retrieval.\n\nRequest:\nFind the API.\n".into()
                    },
                    source: expected_source.into(),
                    suffix: expected_suffix.into(),
                    target: "Search, then read.".into(),
                    source_required,
                },
            );
        }

        for (system, input, expected_prefix, expected_source, source_required) in [
            (
                Some("Be concise."),
                Some("Bonjour"),
                "System:\nBe concise.\n\nFollow the request.\n\nInstruction:\nTranslate.\n\nInput:\n",
                "Bonjour",
                true,
            ),
            (
                Some("Be concise."),
                None,
                "System:\nBe concise.\n\nFollow the request.\n\nInstruction:\nTranslate.",
                "",
                false,
            ),
            (
                None,
                Some("Bonjour"),
                "Follow the request.\n\nInstruction:\nTranslate.\n\nInput:\n",
                "Bonjour",
                true,
            ),
            (
                None,
                None,
                "Follow the request.\n\nInstruction:\nTranslate.",
                "",
                false,
            ),
        ] {
            let mut record = json!({"instruction": "Translate.", "response": "Hello"});
            if let Some(system) = system {
                record["system"] = json!(system);
            }
            if let Some(input) = input {
                record["input"] = json!(input);
            }
            assert_supervised_contract(
                TaskConfig::InstructionTuning {
                    instruction: "Follow the request.".into(),
                },
                record,
                SupervisedPromptSegments {
                    prefix: expected_prefix.into(),
                    source: expected_source.into(),
                    suffix: "\n\nResponse:\n".into(),
                    target: "Hello".into(),
                    source_required,
                },
            );
        }

        for (reasoning, require_reasoning, expected_target) in [
            (
                Some("Add the operands."),
                true,
                "Reasoning:\nAdd the operands.\n\nAnswer:\n4",
            ),
            (None, false, "4"),
        ] {
            let mut record = json!({"question": "2 + 2?", "answer": "4"});
            if let Some(reasoning) = reasoning {
                record["reasoning"] = json!(reasoning);
            }
            assert_supervised_contract(
                TaskConfig::QaReasoning {
                    instruction: "Reason carefully.".into(),
                    require_reasoning,
                },
                record,
                SupervisedPromptSegments {
                    prefix: "Reason carefully.\n\nQuestion:\n".into(),
                    source: "2 + 2?".into(),
                    suffix: "\n\nResponse:\n".into(),
                    target: expected_target.into(),
                    source_required: true,
                },
            );
        }
    }
}
