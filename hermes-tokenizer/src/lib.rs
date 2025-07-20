//! Stable-Rust, byte-level BPE tokenization for Hermes.
//!
//! The optimized merge engine and pretokenizers are derived from GigaToken
//! 0.9.0. The public wrapper is intentionally narrower: local or in-memory
//! Hugging Face `tokenizer.json` artifacts, persistent per-thread caches,
//! deterministic batch order, and the vocabulary operations required by
//! Hermes training, inference, and visualization.

mod bpe;
mod hf;
mod pretokenize;
mod token;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::bpe::Tokenizer as BpeEngine;
use crate::token::TokenId;

/// GigaToken revision from which the optimized byte-level core was extracted.
pub const UPSTREAM_GIGATOKEN_REVISION: &str = "542367a3efed134883fb4f1140b49c04e6fad3a3";

/// Thread-safe tokenizer with persistent single-document and Rayon-worker
/// caches. Token IDs remain those of the source `tokenizer.json`.
pub struct Tokenizer {
    primary: Mutex<BpeEngine>,
    workers: OnceLock<Vec<Mutex<BpeEngine>>>,
    pieces: Arc<[Option<String>]>,
    token_to_id: Arc<HashMap<String, u32>>,
    special_ids: Arc<HashSet<u32>>,
    prefix_ids: Arc<[u32]>,
    suffix_ids: Arc<[u32]>,
}

impl Tokenizer {
    /// Load a byte-level BPE tokenizer from a Hugging Face `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_loaded(hf::load_file(path)?)
    }

    /// Load a byte-level BPE tokenizer from in-memory `tokenizer.json` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_loaded(hf::load_slice(bytes)?)
    }

    fn from_loaded(loaded: hf::LoadedTokenizer) -> Result<Self> {
        ensure!(!loaded.pieces.is_empty(), "tokenizer vocabulary is empty");
        Ok(Self {
            primary: Mutex::new(loaded.engine),
            workers: OnceLock::new(),
            pieces: loaded.pieces.into(),
            token_to_id: Arc::new(loaded.token_to_id),
            special_ids: Arc::new(loaded.special_ids),
            prefix_ids: loaded.prefix_ids.into(),
            suffix_ids: loaded.suffix_ids.into(),
        })
    }

    /// Encode one UTF-8 string, optionally applying the tokenizer's supported
    /// single-sequence post-processor.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut engine = self.primary.lock();
        Ok(encode_with(
            &mut engine,
            text,
            add_special_tokens,
            &self.prefix_ids,
            &self.suffix_ids,
        ))
    }

    /// Encode documents in input order. Small batches stay on the caller;
    /// larger batches use persistent per-Rayon-thread tokenizer forks.
    pub fn encode_batch(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let total_bytes = texts.iter().map(String::len).sum::<usize>();
        if rayon::current_num_threads() == 1 || total_bytes < 1 << 20 {
            let mut engine = self.primary.lock();
            return Ok(texts
                .iter()
                .map(|text| {
                    encode_with(
                        &mut engine,
                        text,
                        add_special_tokens,
                        &self.prefix_ids,
                        &self.suffix_ids,
                    )
                })
                .collect());
        }

        let workers = self.worker_engines();
        let task_count = workers.len() * 4;
        let chunk_size = texts.len().div_ceil(task_count).max(1);
        let chunks: Vec<Vec<Vec<u32>>> = texts
            .par_chunks(chunk_size)
            .map(|chunk| {
                let worker = rayon::current_thread_index().unwrap_or(0) % workers.len();
                let mut engine = workers[worker].lock();
                chunk
                    .iter()
                    .map(|text| {
                        encode_with(
                            &mut engine,
                            text,
                            add_special_tokens,
                            &self.prefix_ids,
                            &self.suffix_ids,
                        )
                    })
                    .collect()
            })
            .collect();
        Ok(chunks.into_iter().flatten().collect())
    }

    fn worker_engines(&self) -> &[Mutex<BpeEngine>] {
        self.workers.get_or_init(|| {
            let engine = self.primary.lock();
            (0..rayon::current_num_threads().max(1))
                .map(|_| Mutex::new(engine.fork()))
                .collect()
        })
    }

    /// Decode IDs through the byte-level vocabulary. Invalid UTF-8 fragments
    /// use the replacement character, matching the display behavior of the
    /// previous tokenizer backend.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            if skip_special_tokens && self.special_ids.contains(&id) {
                continue;
            }
            ensure!(
                self.pieces.get(id as usize).is_some_and(Option::is_some),
                "tokenizer has no vocabulary entry for token ID {id}"
            );
            tokens.push(TokenId::from(id));
        }
        let engine = self.primary.lock();
        let bytes: Vec<u8> = engine.decode(&tokens).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Return the exact vocabulary spelling stored in `tokenizer.json`.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.pieces.get(id as usize)?.clone()
    }

    /// Resolve an exact model or added-token spelling to its ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Vocabulary extent including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

fn encode_with(
    engine: &mut BpeEngine,
    text: &str,
    add_special_tokens: bool,
    prefix_ids: &[u32],
    suffix_ids: &[u32],
) -> Vec<u32> {
    let mut ids = Vec::new();
    engine.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
    if !add_special_tokens || (prefix_ids.is_empty() && suffix_ids.is_empty()) {
        return ids;
    }
    let mut wrapped = Vec::with_capacity(prefix_ids.len() + ids.len() + suffix_ids.len());
    wrapped.extend_from_slice(prefix_ids);
    wrapped.extend(ids);
    wrapped.extend_from_slice(suffix_ids);
    wrapped
}
