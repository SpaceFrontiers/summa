//! Streaming corpus ingestion, tokenization, shuffling, and batch packing.
//!
//! Documents are joined with EOS and packed without padding. JSONL and plain
//! text inputs may be Zstandard-compressed; only a bounded shuffle buffer and
//! one tokenizer batch are retained in memory.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Int, Tensor, TensorData};
use hermes_llm::Tokenizer;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

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
    samples: Vec<Vec<i64>>,
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

    fn push(&mut self, sample: Vec<i64>) -> Option<Vec<i64>> {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            return None;
        }
        let index = self.rng.random_range(0..self.samples.len());
        Some(std::mem::replace(&mut self.samples[index], sample))
    }

    fn finish(mut self) -> Vec<Vec<i64>> {
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
        visit: &mut impl FnMut(Vec<i64>) -> Result<bool>,
    ) -> Result<bool> {
        if self.consumed > 0 {
            self.pending.drain(..self.consumed);
            self.consumed = 0;
        }
        self.pending.extend(tokens);
        while self.pending.len() - self.consumed > self.seq_len {
            let end = self.consumed + self.seq_len + 1;
            let sample = self.pending[self.consumed..end].to_vec();
            self.consumed += self.seq_len;
            *count += 1;
            if !visit(sample)? {
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
    visit: &mut impl FnMut(Vec<i64>) -> Result<bool>,
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

/// Visit fixed-length next-token samples in their source order.
fn visit_samples_in_order(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    mut visit: impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<usize> {
    ensure!(seq_len > 0, "seq_len must be positive");
    let mut count = 0;
    let mut packer = SamplePacker::new(seq_len);
    let is_jsonl = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"));
    let mut reader = open_data(path)?;
    if is_jsonl {
        let mut documents = Vec::with_capacity(TOKENIZE_BATCH);
        let mut line = String::new();
        let mut line_number = 0;
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            line_number += 1;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)
                .with_context(|| format!("invalid JSONL at {}:{line_number}", path.display()))?;
            let document = value
                .get("text")
                .and_then(|value| value.as_str())
                .with_context(|| {
                    format!(
                        "JSONL row at {}:{line_number} must contain a string `text` field",
                        path.display()
                    )
                })?;
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

pub(crate) fn visit_samples(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    shuffle_buffer: usize,
    seed: u64,
    mut visit: impl FnMut(Vec<i64>) -> Result<bool>,
) -> Result<usize> {
    if shuffle_buffer == 0 {
        return visit_samples_in_order(path, tokenizer, seq_len, visit);
    }

    let mut shuffler = ShuffleBuffer::new(shuffle_buffer, seed);
    let mut keep_going = true;
    let count = visit_samples_in_order(path, tokenizer, seq_len, |sample| {
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

pub(crate) fn count_samples(path: &Path, tokenizer: &Tokenizer, seq_len: usize) -> Result<usize> {
    visit_samples_in_order(path, tokenizer, seq_len, |_| Ok(true))
}

pub(crate) fn make_batch(
    samples: &[Vec<i64>],
    seq_len: usize,
    device: &Device,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let mut inputs = Vec::with_capacity(samples.len() * seq_len);
    let mut targets = Vec::with_capacity(samples.len() * seq_len);
    for sample in samples {
        inputs.extend_from_slice(&sample[..seq_len]);
        targets.extend_from_slice(&sample[1..]);
    }
    (
        Tensor::from_data(TensorData::new(inputs, [samples.len(), seq_len]), device),
        Tensor::from_data(TensorData::new(targets, [samples.len(), seq_len]), device),
    )
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Cursor, Read};

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
                if let Some(sample) = buffer.push(vec![value]) {
                    output.push(sample[0]);
                }
                assert!(buffer.samples.len() <= 4);
            }
            output.extend(buffer.finish().into_iter().map(|sample| sample[0]));
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
            samples.push(sample);
            Ok(true)
        };

        for document in [vec![1, 2, 0], vec![3, 4, 0], vec![5, 6, 0]] {
            assert!(packer.push(document, &mut count, &mut collect).unwrap());
        }

        assert_eq!(count, 2);
        assert_eq!(samples, [vec![1, 2, 0, 3], vec![3, 4, 0, 5]]);
    }
}
