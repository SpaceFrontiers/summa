//! Text generation utilities for LLM inference.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::Rng;

use crate::model::Transformer;

/// Text generator for autoregressive text generation.
pub struct TextGenerator<'a> {
    model: &'a Transformer,
    device: &'a Device,
}

impl<'a> TextGenerator<'a> {
    pub fn new(model: &'a Transformer, device: &'a Device) -> Self {
        Self { model, device }
    }

    /// Generate tokens autoregressively from a prompt.
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
        let mut tokens = prompt_tokens.to_vec();
        let mut rng = rand::rng();

        for _ in 0..max_new_tokens {
            let context_len = tokens.len().min(self.model.config().max_seq_len);
            let context: Vec<u32> = tokens[tokens.len() - context_len..].to_vec();

            let input = Tensor::new(context.as_slice(), self.device)?
                .unsqueeze(0)?
                .to_dtype(DType::U32)?;

            let logits = self.model.forward(&input, 0, false)?;
            // Shape: [1, seq_len, vocab] -> [1, 1, vocab] -> [vocab]
            let logits = logits
                .narrow(1, context_len - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?;

            let logits = if temperature != 1.0 {
                logits.affine(1.0 / temperature, 0.0)?
            } else {
                logits
            };

            let logits = if let Some(k) = top_k {
                top_k_filter(&logits, k, self.device)?
            } else {
                logits
            };

            let next_token = sample_from_logits(&logits, &mut rng)?;
            tokens.push(next_token);
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

/// Cosine learning rate schedule with warmup.
pub fn get_lr_with_warmup(
    step: usize,
    warmup_steps: usize,
    max_lr: f64,
    min_lr: f64,
    total_steps: usize,
) -> f64 {
    if step < warmup_steps {
        max_lr * (step as f64 / warmup_steps as f64)
    } else {
        let decay_ratio = (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
        let coeff = 0.5 * (1.0 + (std::f64::consts::PI * decay_ratio).cos());
        min_lr + coeff * (max_lr - min_lr)
    }
}
