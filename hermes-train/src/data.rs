//! Streaming objective data, deterministic shuffling, and tensor batches.
//!
//! Causal-LM documents are EOS-joined and packed without padding. Structured
//! objectives use explicit JSONL contracts and fixed shapes: target-only loss
//! positions prevent EOS padding or prompts from contributing to supervised
//! losses, while retrieval batches retain positive and hard-negative grouping.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Read, Write};
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
const TOKEN_CACHE_MAGIC: &[u8; 8] = b"HERTOK01";
const MAX_CACHED_DOCUMENT_TOKENS: usize = 100_000_000;

struct TokenCacheWriter {
    writer: BufWriter<File>,
}

impl TokenCacheWriter {
    fn append(&mut self, tokens: &[u32]) -> Result<()> {
        let len = u32::try_from(tokens.len()).context("document is too large for token cache")?;
        self.writer.write_all(&len.to_le_bytes())?;
        for token in tokens {
            self.writer.write_all(&token.to_le_bytes())?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("failed to flush token cache")
    }
}

/// Replay complete cached documents and open an append-only writer at the
/// first missing document. A torn final record is safely discarded because
/// this cache is derived from the authoritative corpus.
fn replay_token_cache(
    path: &Path,
    eos_token: u32,
    packer: &mut SamplePacker,
    count: &mut usize,
    visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<(usize, bool, Option<TokenCacheWriter>)> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut cache = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("failed to open token cache {}", path.display()))?;
    if cache.metadata()?.len() < TOKEN_CACHE_MAGIC.len() as u64 {
        cache.set_len(0)?;
        cache.write_all(TOKEN_CACHE_MAGIC)?;
        cache.flush()?;
    }

    let mut reader = BufReader::new(cache);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    ensure!(
        &magic == TOKEN_CACHE_MAGIC,
        "token cache {} has an unsupported format",
        path.display()
    );
    let mut valid_bytes = TOKEN_CACHE_MAGIC.len() as u64;
    let mut documents = 0usize;
    let mut torn_tail = false;
    loop {
        let mut len_bytes = [0_u8; 4];
        let read = reader.read(&mut len_bytes)?;
        if read == 0 {
            break;
        }
        if read < len_bytes.len()
            && let Err(error) = reader.read_exact(&mut len_bytes[read..])
        {
            if error.kind() == ErrorKind::UnexpectedEof {
                torn_tail = true;
                break;
            }
            return Err(error.into());
        }
        let len = u32::from_le_bytes(len_bytes) as usize;
        ensure!(
            len <= MAX_CACHED_DOCUMENT_TOKENS,
            "token cache {} contains an implausible {len}-token document",
            path.display()
        );
        let byte_len = len
            .checked_mul(std::mem::size_of::<u32>())
            .context("cached document byte length overflows usize")?;
        let mut bytes = vec![0_u8; byte_len];
        if let Err(error) = reader.read_exact(&mut bytes) {
            if error.kind() == ErrorKind::UnexpectedEof {
                torn_tail = true;
                break;
            }
            return Err(error.into());
        }
        valid_bytes = valid_bytes
            .checked_add(4 + byte_len as u64)
            .context("token cache offset overflows u64")?;
        documents = documents
            .checked_add(1)
            .context("cached document count overflows usize")?;
        let tokens = bytes
            .chunks_exact(4)
            .map(|bytes| i64::from(u32::from_le_bytes(bytes.try_into().unwrap())))
            .chain(std::iter::once(i64::from(eos_token)));
        if !packer.push(tokens, count, visit)? {
            return Ok((documents, false, None));
        }
    }
    drop(reader);
    if torn_tail {
        OpenOptions::new()
            .write(true)
            .open(path)?
            .set_len(valid_bytes)?;
    }
    let writer = OpenOptions::new().append(true).open(path)?;
    Ok((
        documents,
        true,
        Some(TokenCacheWriter {
            writer: BufWriter::new(writer),
        }),
    ))
}

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
    cache: &mut Option<TokenCacheWriter>,
) -> Result<bool> {
    if documents.is_empty() {
        return Ok(true);
    }
    let encodings = tokenizer.encode_batch(std::mem::take(documents), false)?;
    for tokens in encodings {
        if let Some(cache) = cache {
            cache.append(&tokens)?;
        }
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
    token_cache: Option<&Path>,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    let mut count = 0;
    let mut packer = SamplePacker::new(seq_len);
    let (cached_documents, keep_going, mut cache) = match token_cache {
        Some(path) => replay_token_cache(
            path,
            tokenizer.eos_token_id(),
            &mut packer,
            &mut count,
            &mut visit,
        )?,
        None => (0, true, None),
    };
    if cached_documents > 0
        && let Some(path) = token_cache
    {
        println!(
            "token_cache={} replayed_documents={cached_documents}",
            path.display()
        );
    }
    if !keep_going {
        return Ok(count);
    }
    let mut reader = open_data(path)?;
    if is_jsonl(path) {
        let mut documents = Vec::with_capacity(TOKENIZE_BATCH);
        let mut line = String::new();
        let mut line_number = 0usize;
        let mut document_number = 0usize;
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
            document_number = document_number
                .checked_add(1)
                .context("JSONL document count overflows usize")?;
            if document_number <= cached_documents {
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
                    &mut cache,
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
            &mut cache,
        )? {
            return Ok(count);
        }
    } else {
        if cached_documents == 0 {
            let mut document = String::new();
            reader.read_to_string(&mut document)?;
            if !push_documents(
                &mut vec![document],
                tokenizer,
                &mut packer,
                &mut count,
                &mut visit,
                &mut cache,
            )? {
                return Ok(count);
            }
        }
    }
    if let Some(cache) = &mut cache {
        cache.flush()?;
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
    token_cache: Option<&Path>,
    visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    ensure!(seq_len > 0, "sequence_length must be positive");
    match objective {
        ObjectiveConfig::CausalLm => {
            visit_causal_samples(path, tokenizer, seq_len, token_cache, visit)
        }
        _ => visit_structured_samples(path, objective, tokenizer, seq_len, visit),
    }
}

pub(crate) struct SampleStreamConfig<'a> {
    pub(crate) seq_len: usize,
    pub(crate) shuffle_buffer: usize,
    pub(crate) seed: u64,
    pub(crate) token_cache: Option<&'a Path>,
}

pub(crate) fn visit_samples(
    path: &Path,
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    config: SampleStreamConfig<'_>,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    if config.shuffle_buffer == 0 {
        return visit_samples_in_order(
            path,
            objective,
            tokenizer,
            config.seq_len,
            config.token_cache,
            visit,
        );
    }

    let mut shuffler = ShuffleBuffer::new(config.shuffle_buffer, config.seed);
    let mut keep_going = true;
    let count = visit_samples_in_order(
        path,
        objective,
        tokenizer,
        config.seq_len,
        config.token_cache,
        |sample| {
            if let Some(sample) = shuffler.push(sample) {
                keep_going = visit(sample)?;
            }
            Ok(keep_going)
        },
    )?;

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
    token_cache: Option<&Path>,
) -> Result<usize> {
    visit_samples_in_order(path, objective, tokenizer, seq_len, token_cache, |_| {
        Ok(true)
    })
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
    fn token_cache_replays_and_repairs_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        let mut bytes = TOKEN_CACHE_MAGIC.to_vec();
        for document in [&[1_u32, 2][..], &[3, 4][..]] {
            bytes.extend_from_slice(&(document.len() as u32).to_le_bytes());
            for token in document {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        let valid_len = bytes.len();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&99_u32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let mut packer = SamplePacker::new(3);
        let mut samples = Vec::new();
        let mut count = 0;
        let (documents, keep_going, mut writer) =
            replay_token_cache(&path, 0, &mut packer, &mut count, &mut |sample| {
                samples.push(sample);
                Ok(true)
            })
            .unwrap();
        assert_eq!(documents, 2);
        assert!(keep_going);
        assert_eq!(fs::metadata(&path).unwrap().len(), valid_len as u64);
        assert_eq!(count, 1);
        let TrainingSample::Causal { tokens } = &samples[0] else {
            unreachable!()
        };
        assert_eq!(tokens, &[1, 2, 0, 3]);

        writer.as_mut().unwrap().append(&[5, 6]).unwrap();
        writer.as_mut().unwrap().flush().unwrap();
        drop(writer);
        let mut replay = SamplePacker::new(3);
        let mut replay_count = 0;
        let (documents, _, _) =
            replay_token_cache(&path, 0, &mut replay, &mut replay_count, &mut |_| Ok(true))
                .unwrap();
        assert_eq!(documents, 3);
        assert_eq!(replay_count, 2);
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
