//! Text generation utilities for LLM inference.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::Rng;

use crate::model::{InferenceState, Transformer};

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
    /// last-position logits.
    fn prefill(&self, context: &[u32]) -> Result<(InferenceState, Tensor)> {
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
    /// window fills up, generation re-prefills from the last half window
    /// (amortized, replacing the old full-recompute-per-token behavior).
    ///
    /// # Arguments
    /// * `prompt_tokens` - Initial token sequence
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no scaling)
    /// * `top_k` - Optional top-k filtering
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
    ) -> Result<Vec<u32>> {
        let max_seq_len = self.model.config().max_seq_len;
        let mut tokens = prompt_tokens.to_vec();
        let mut rng = rand::rng();

        let context_len = tokens.len().min(max_seq_len);
        let (mut state, mut last_logits) = self.prefill(&tokens[tokens.len() - context_len..])?;

        for _ in 0..max_new_tokens {
            let logits = if temperature != 1.0 {
                last_logits.affine(1.0 / temperature, 0.0)?
            } else {
                last_logits.clone()
            };
            let logits = if let Some(k) = top_k {
                top_k_filter(&logits, k, self.device)?
            } else {
                logits
            };
            let next_token = sample_from_logits(&logits, &mut rng)?;
            tokens.push(next_token);

            if state.pos() >= max_seq_len {
                // Context full: rebuild state from the last half window
                let keep = max_seq_len / 2;
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

/// Apply top-k filtering to logits.
fn top_k_filter(logits: &Tensor, k: usize, device: &Device) -> Result<Tensor> {
    let logits_vec: Vec<f32> = logits.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut masked = vec![f32::NEG_INFINITY; logits_vec.len()];
    for i in 0..k.min(indexed.len()) {
        masked[indexed[i].0] = indexed[i].1;
    }

    Ok(Tensor::new(masked, device)?)
}

/// Sample a token from logits using the probability distribution.
fn sample_from_logits(logits: &Tensor, rng: &mut impl Rng) -> Result<u32> {
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
    let next_token = cumsum.iter().position(|&p| p > r).unwrap_or(0) as u32;

    Ok(next_token)
}
