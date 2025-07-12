//! Text generation utilities for LLM inference.

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::model::{InferenceState, Transformer};

/// Sampling configuration for [`TextGenerator::generate`].
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Softmax temperature. `<= 0.0` selects deterministic greedy decoding.
    pub temperature: f64,
    /// Optional top-k filtering (`Some(0)` is rejected).
    pub top_k: Option<usize>,
    /// Stop as soon as this token is generated, if set.
    pub eos_token: Option<u32>,
    /// RNG seed for reproducible sampling; random when `None`.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.8,
            top_k: None,
            eos_token: None,
            seed: None,
        }
    }
}

/// Text generator for autoregressive text generation.
pub struct TextGenerator<'a> {
    model: &'a Transformer,
    device: &'a Device,
}

impl<'a> TextGenerator<'a> {
    pub fn new(model: &'a Transformer, device: &'a Device) -> Self {
        Self { model, device }
    }

    /// Prefill `context` into a fresh inference state and return the
    /// last-position logits. `context` must be non-empty.
    fn prefill(&self, context: &[u32]) -> Result<(InferenceState, Tensor)> {
        debug_assert!(!context.is_empty());
        let mut state = self.model.make_state(1, self.device)?;
        let input = Tensor::new(context, self.device)?
            .unsqueeze(0)?
            .to_dtype(DType::U32)?;
        let logits = self.model.forward_with_state(&input, &mut state)?;
        let last = logits
            .narrow(1, context.len() - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?;
        Ok((state, last))
    }

    /// Generate tokens autoregressively from a prompt.
    ///
    /// Incremental inference: the prompt is prefilled once, then each new
    /// token is a single forward step — Mamba layers advance an O(1)
    /// recurrent state, attention layers extend a KV cache. When the context
    /// window fills up, generation re-prefills from the last half window.
    ///
    /// Returns the full token sequence (prompt + generated). Stops early on
    /// `config.eos_token`. Errors on an empty prompt or `top_k == Some(0)`.
    pub fn generate(&self, prompt_tokens: &[u32], config: &SamplingConfig) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            bail!("cannot generate from an empty prompt");
        }
        if config.top_k == Some(0) {
            bail!("top_k must be >= 1 (got 0)");
        }
        let greedy = config.temperature <= 0.0;

        let max_seq_len = self.model.config().max_seq_len;
        let mut tokens = prompt_tokens.to_vec();
        let seed = config.seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);

        let context_len = tokens.len().min(max_seq_len);
        let (mut state, mut last_logits) = self.prefill(&tokens[tokens.len() - context_len..])?;

        for _ in 0..config.max_new_tokens {
            let next_token = if greedy {
                argmax(&last_logits)?
            } else {
                let logits = if config.temperature != 1.0 {
                    last_logits.affine(1.0 / config.temperature, 0.0)?
                } else {
                    last_logits.clone()
                };
                let logits = match config.top_k {
                    Some(k) => top_k_filter(&logits, k, self.device)?,
                    None => logits,
                };
                sample_from_logits(&logits, &mut rng)?
            };
            tokens.push(next_token);

            if config.eos_token == Some(next_token) {
                break;
            }

            if state.pos() >= max_seq_len {
                // Context full: rebuild state from the last half window.
                let keep = (max_seq_len / 2).max(1);
                let (new_state, logits) = self.prefill(&tokens[tokens.len() - keep..])?;
                state = new_state;
                last_logits = logits;
            } else {
                let input = Tensor::new(&[next_token], self.device)?
                    .unsqueeze(0)?
                    .to_dtype(DType::U32)?;
                let logits = self.model.forward_with_state(&input, &mut state)?;
                last_logits = logits.squeeze(1)?.squeeze(0)?;
            }
        }

        Ok(tokens)
    }
}

/// Index of the maximum logit (greedy decoding).
fn argmax(logits: &Tensor) -> Result<u32> {
    let v: Vec<f32> = logits.to_vec1()?;
    let idx = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(idx as u32)
}

/// Apply top-k filtering to logits (keeps the k highest, masks the rest to
/// `-inf`). `k` is clamped to the vocabulary size; callers reject `k == 0`.
fn top_k_filter(logits: &Tensor, k: usize, device: &Device) -> Result<Tensor> {
    let logits_vec: Vec<f32> = logits.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut masked = vec![f32::NEG_INFINITY; logits_vec.len()];
    for &(idx, val) in indexed.iter().take(k.min(indexed.len())) {
        masked[idx] = val;
    }

    Ok(Tensor::new(masked, device)?)
}

/// Sample a token from logits using the softmax probability distribution.
fn sample_from_logits(logits: &Tensor, rng: &mut impl rand::Rng) -> Result<u32> {
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    let cumsum: Vec<f32> = probs_vec
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();

    let r: f32 = rng.random();
    // Fall back to the last index (not 0) if rounding leaves r above the final
    // cumulative sum, so a degenerate draw picks a high-prob token, not token 0.
    let next_token = cumsum
        .iter()
        .position(|&p| p > r)
        .unwrap_or(cumsum.len().saturating_sub(1)) as u32;

    Ok(next_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(v: &[f32]) -> Tensor {
        Tensor::new(v, &Device::Cpu).unwrap()
    }

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(argmax(&logits(&[0.1, 5.0, 0.2, -1.0])).unwrap(), 1);
    }

    #[test]
    fn top_k_keeps_k_highest() {
        let out: Vec<f32> = top_k_filter(&logits(&[1.0, 3.0, 2.0, 0.5]), 2, &Device::Cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(out[1], 3.0);
        assert_eq!(out[2], 2.0);
        assert!(out[0].is_infinite() && out[3].is_infinite());
    }

    #[test]
    fn sampling_is_deterministic_under_seed() {
        let l = logits(&[1.0, 2.0, 3.0, 0.5, 1.5]);
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            assert_eq!(
                sample_from_logits(&l, &mut a).unwrap(),
                sample_from_logits(&l, &mut b).unwrap()
            );
        }
    }
}
