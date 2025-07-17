//! Streaming objective data, deterministic shuffling, and tensor batches.
//!
//! Causal-LM documents are EOS-joined and packed without padding. Structured
//! objectives use explicit JSONL contracts and fixed shapes: target-only loss
//! positions prevent EOS padding or prompts from contributing to supervised
//! losses, while retrieval batches retain positive and hard-negative grouping.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::curriculum::ObjectiveConfig;

mod batch;
mod structured;

pub(crate) use batch::{BatchStats, LanguageBatch, RetrievalBatch, TrainingBatch, make_batch};
use batch::{EncodedText, TrainingSample};
use structured::visit_structured_samples;

const TOKENIZE_BATCH: usize = 1_000;

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
    samples: Vec<TrainingSample>,
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

    fn push(&mut self, sample: TrainingSample) -> Option<TrainingSample> {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            return None;
        }
        let index = self.rng.random_range(0..self.samples.len());
        Some(std::mem::replace(&mut self.samples[index], sample))
    }

    fn finish(mut self) -> Vec<TrainingSample> {
        self.samples.shuffle(&mut self.rng);
        self.samples
    }
}

struct SamplePacker {
    pending: Vec<i64>,
    consumed: usize,
    seq_len: usize,
}

impl SamplePacker {
    fn new(seq_len: usize) -> Self {
        Self {
            pending: Vec::new(),
            consumed: 0,
            seq_len,
        }
    }

    fn push(
        &mut self,
        tokens: impl IntoIterator<Item = i64>,
        count: &mut usize,
        visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
    ) -> Result<bool> {
        if self.consumed > 0 {
            self.pending.drain(..self.consumed);
            self.consumed = 0;
        }
        self.pending.extend(tokens);
        while self.pending.len() - self.consumed > self.seq_len {
            let end = self
                .consumed
                .checked_add(self.seq_len)
                .and_then(|end| end.checked_add(1))
                .context("packed sample boundary overflows usize")?;
            let tokens = self.pending[self.consumed..end].to_vec();
            self.consumed = self
                .consumed
                .checked_add(self.seq_len)
                .context("packed-token cursor overflows usize")?;
            *count = count
                .checked_add(1)
                .context("training sample count overflows usize")?;
            if !visit(TrainingSample::Causal { tokens })? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn push_documents(
    documents: &mut Vec<String>,
    tokenizer: &Tokenizer,
    packer: &mut SamplePacker,
    count: &mut usize,
    visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<bool> {
    if documents.is_empty() {
        return Ok(true);
    }
    let encodings = tokenizer.encode_batch(std::mem::take(documents), false)?;
    for tokens in encodings {
        let tokens = tokens
            .into_iter()
            .map(i64::from)
            .chain(std::iter::once(i64::from(tokenizer.eos_token_id())));
        if !packer.push(tokens, count, visit)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn is_jsonl(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"))
}

fn visit_causal_samples(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    let mut count = 0;
    let mut packer = SamplePacker::new(seq_len);
    let mut reader = open_data(path)?;
    if is_jsonl(path) {
        let mut documents = Vec::with_capacity(TOKENIZE_BATCH);
        let mut line = String::new();
        let mut line_number = 0usize;
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
            let document = required_string(&value, "text", path, line_number)?;
            documents.push(document.to_owned());
            if documents.len() == TOKENIZE_BATCH
                && !push_documents(
                    &mut documents,
                    tokenizer,
                    &mut packer,
                    &mut count,
                    &mut visit,
                )?
            {
                return Ok(count);
            }
        }
        if !push_documents(
            &mut documents,
            tokenizer,
            &mut packer,
            &mut count,
            &mut visit,
        )? {
            return Ok(count);
        }
    } else {
        let mut document = String::new();
        reader.read_to_string(&mut document)?;
        if !push_documents(
            &mut vec![document],
            tokenizer,
            &mut packer,
            &mut count,
            &mut visit,
        )? {
            return Ok(count);
        }
    }
    Ok(count)
}

fn required_string<'a>(
    value: &'a serde_json::Value,
    field: &str,
    path: &Path,
    line_number: usize,
) -> Result<&'a str> {
    let text = value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .with_context(|| {
            format!(
                "JSONL row at {}:{line_number} must contain a string `{field}` field",
                path.display()
            )
        })?;
    ensure!(
        !text.trim().is_empty(),
        "JSONL row at {}:{line_number} has an empty `{field}` field",
        path.display()
    );
    Ok(text)
}

/// Visit fixed-shape objective samples in source order.
fn visit_samples_in_order(
    path: &Path,
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    ensure!(seq_len > 0, "sequence_length must be positive");
    match objective {
        ObjectiveConfig::CausalLm => visit_causal_samples(path, tokenizer, seq_len, visit),
        _ => visit_structured_samples(path, objective, tokenizer, seq_len, visit),
    }
}

pub(crate) fn visit_samples(
    path: &Path,
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    shuffle_buffer: usize,
    seed: u64,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    if shuffle_buffer == 0 {
        return visit_samples_in_order(path, objective, tokenizer, seq_len, visit);
    }

    let mut shuffler = ShuffleBuffer::new(shuffle_buffer, seed);
    let mut keep_going = true;
    let count = visit_samples_in_order(path, objective, tokenizer, seq_len, |sample| {
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

pub(crate) fn count_samples(
    path: &Path,
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
) -> Result<usize> {
    visit_samples_in_order(path, objective, tokenizer, seq_len, |_| Ok(true))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Cursor, Read};

    use burn::tensor::Device;

    use super::*;

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
                if let Some(TrainingSample::Causal { tokens }) =
                    buffer.push(TrainingSample::Causal {
                        tokens: vec![value],
                    })
                {
                    output.push(tokens[0]);
                }
                assert!(buffer.samples.len() <= 4);
            }
            output.extend(buffer.finish().into_iter().map(|sample| match sample {
                TrainingSample::Causal { tokens } => tokens[0],
                _ => unreachable!(),
            }));
            output
        };

        let first = shuffle(7);
        assert_eq!(first, shuffle(7));
        assert_ne!(first, (0..32_i64).collect::<Vec<_>>());
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32_i64).collect::<Vec<_>>());
    }

    #[test]
    fn sample_packer_joins_documents_without_dropping_tokens() {
        let mut packer = SamplePacker::new(3);
        let mut samples = Vec::new();
        let mut count = 0;
        let mut collect = |sample| {
            let TrainingSample::Causal { tokens } = sample else {
                unreachable!()
            };
            samples.push(tokens);
            Ok(true)
        };

        for document in [vec![1, 2, 0], vec![3, 4, 0], vec![5, 6, 0]] {
            assert!(packer.push(document, &mut count, &mut collect).unwrap());
        }

        assert_eq!(count, 2);
        assert_eq!(samples, [vec![1, 2, 0, 3], vec![3, 4, 0, 5]]);
    }

    #[test]
    fn supervised_batch_masks_only_target_positions() {
        let device = Device::ndarray();
        let samples = vec![
            TrainingSample::Supervised {
                tokens: vec![10, 11, 20, 21, 0],
                loss_positions: vec![1, 2, 3],
                truncated_tokens: 4,
            },
            TrainingSample::Supervised {
                tokens: vec![12, 13, 22, 23, 0],
                loss_positions: vec![2, 3],
                truncated_tokens: 0,
            },
        ];
        let TrainingBatch::Language(batch) = make_batch(&samples, 4, &device).unwrap() else {
            panic!("expected masked language batch")
        };
        let LanguageBatch {
            loss_positions: Some(positions),
            stats,
            ..
        } = *batch
        else {
            panic!("expected target positions")
        };
        assert_eq!(
            positions.into_data().to_vec::<i64>().unwrap(),
            vec![1, 2, 3, 6, 7]
        );
        assert_eq!(stats.supervised_tokens, 5);
        assert_eq!(stats.truncated_tokens, 4);
    }

    #[test]
    fn retrieval_batch_labels_positives_among_all_candidates() {
        let device = Device::ndarray();
        let encoded = |start| EncodedText {
            tokens: vec![start, start + 1, 0],
            end_position: 1,
        };
        let samples = vec![
            TrainingSample::Retrieval {
                query: encoded(1),
                documents: vec![encoded(10), encoded(20)],
                truncated_tokens: 0,
            },
            TrainingSample::Retrieval {
                query: encoded(2),
                documents: vec![encoded(30)],
                truncated_tokens: 0,
            },
        ];
        let TrainingBatch::Retrieval(batch) = make_batch(&samples, 3, &device).unwrap() else {
            panic!("expected retrieval batch")
        };
        let RetrievalBatch {
            labels,
            query_end_positions,
            document_end_positions,
            stats,
            ..
        } = *batch;
        assert_eq!(labels.into_data().to_vec::<i64>().unwrap(), vec![0, 2]);
        assert_eq!(
            query_end_positions.into_data().to_vec::<i64>().unwrap(),
            vec![1, 4]
        );
        assert_eq!(
            document_end_positions.into_data().to_vec::<i64>().unwrap(),
            vec![1, 4, 7]
        );
        assert_eq!(stats.retrieval_candidates, 3);
        assert_eq!(stats.compute_tokens, 15);
    }
}
