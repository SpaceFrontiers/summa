use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::Rng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use crate::io as file_io;
use crate::tokenizer::Tokenizer;

#[derive(Deserialize)]
struct JsonlRecord {
    text: String,
}

pub struct Dataset {
    tokens: Vec<u32>,
    seq_len: usize,
}

impl Dataset {
    pub fn new(tokens: Vec<u32>, seq_len: usize) -> Self {
        Self { tokens, seq_len }
    }

    fn from_reader<R: Read>(reader: R, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        let reader = BufReader::new(reader);
        let mut all_tokens = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let record: JsonlRecord = serde_json::from_str(&line)?;
            if !record.text.is_empty() {
                let tokens = tokenizer.encode(&record.text, false)?;
                all_tokens.extend(tokens);
                all_tokens.push(tokenizer.eos_token_id());
            }
        }

        Ok(all_tokens)
    }

    /// Load dataset from a JSONL file where each line has a "text" field.
    /// Supports .gz and .zst/.zstd compressed files.
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        tokenizer: &Tokenizer,
        seq_len: usize,
    ) -> Result<Self> {
        let reader = file_io::open_file(path)?;
        let tokens = Self::from_reader(reader, tokenizer)?;
        Ok(Self::new(tokens, seq_len))
    }

    /// Load dataset from stdin (JSONL format).
    pub fn from_stdin(tokenizer: &Tokenizer, seq_len: usize) -> Result<Self> {
        let stdin = io::stdin().lock();
        let tokens = Self::from_reader(stdin, tokenizer)?;
        Ok(Self::new(tokens, seq_len))
    }

    /// Load dataset from multiple JSONL files.
    /// Supports .gz and .zst/.zstd compressed files.
    pub fn from_files<P: AsRef<Path>>(
        paths: &[P],
        tokenizer: &Tokenizer,
        seq_len: usize,
    ) -> Result<Self> {
        let mut all_tokens = Vec::new();

        for path in paths {
            let reader = file_io::open_file(path)?;

            for line in reader.lines() {
                let line = line?;
                if line.is_empty() {
                    continue;
                }
                let record: JsonlRecord = serde_json::from_str(&line)?;
                if !record.text.is_empty() {
                    let tokens = tokenizer.encode(&record.text, false)?;
                    all_tokens.extend(tokens);
                    all_tokens.push(tokenizer.eos_token_id());
                }
            }
        }

        Ok(Self::new(all_tokens, seq_len))
    }

    pub fn len(&self) -> usize {
        if self.tokens.len() <= self.seq_len {
            0
        } else {
            self.tokens.len() - self.seq_len
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_batch(&self, indices: &[usize], device: &Device) -> Result<(Tensor, Tensor)> {
        let batch_size = indices.len();
        let mut input_data = Vec::with_capacity(batch_size * self.seq_len);
        let mut target_data = Vec::with_capacity(batch_size * self.seq_len);

        for &idx in indices {
            let start = idx;
            let end = start + self.seq_len;

            for i in start..end {
                input_data.push(self.tokens[i]);
                target_data.push(self.tokens[i + 1]);
            }
        }

        let input = Tensor::new(input_data, device)?
            .reshape((batch_size, self.seq_len))?
            .to_dtype(candle_core::DType::U32)?;
        let target = Tensor::new(target_data, device)?
            .reshape((batch_size, self.seq_len))?
            .to_dtype(candle_core::DType::U32)?;

        Ok((input, target))
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_pos: usize,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current_pos = 0;
        if self.shuffle {
            let mut rng = rand::rng();
            self.indices.shuffle(&mut rng);
        }
    }

    pub fn num_batches(&self) -> usize {
        self.dataset.len() / self.batch_size
    }

    pub fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        if self.current_pos + self.batch_size > self.indices.len() {
            return Ok(None);
        }

        let batch_indices: Vec<usize> =
            self.indices[self.current_pos..self.current_pos + self.batch_size].to_vec();
        self.current_pos += self.batch_size;

        let (input, target) = self.dataset.get_batch(&batch_indices, device)?;
        Ok(Some((input, target)))
    }

    pub fn iter<'a>(&'a mut self, device: &'a Device) -> DataLoaderIterator<'a> {
        self.reset();
        DataLoaderIterator {
            loader: self,
            device,
        }
    }
}

pub struct DataLoaderIterator<'a> {
    loader: &'a mut DataLoader,
    device: &'a Device,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.loader.next_batch(self.device) {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

pub fn generate_random_batch(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut rng = rand::rng();
    let input_data: Vec<u32> = (0..batch_size * seq_len)
        .map(|_| rng.random_range(0..vocab_size as u32))
        .collect();
    let target_data: Vec<u32> = (0..batch_size * seq_len)
        .map(|_| rng.random_range(0..vocab_size as u32))
        .collect();

    let input = Tensor::new(input_data, device)?
        .reshape((batch_size, seq_len))?
        .to_dtype(candle_core::DType::U32)?;
    let target = Tensor::new(target_data, device)?
        .reshape((batch_size, seq_len))?
        .to_dtype(candle_core::DType::U32)?;

    Ok((input, target))
}
