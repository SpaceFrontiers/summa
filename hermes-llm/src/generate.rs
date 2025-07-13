//! Autoregressive text generation.

use anyhow::{Result, bail};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{Backend, Device, InferenceState, Transformer};

/// Sampling configuration for [`TextGenerator::generate`].
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub max_new_tokens: usize,
    /// `<= 0.0` selects greedy decoding.
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub eos_token: Option<u32>,
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

pub struct TextGenerator<'a> {
    model: &'a Transformer<Backend>,
    device: &'a Device,
}

impl<'a> TextGenerator<'a> {
    pub fn new(model: &'a Transformer<Backend>, device: &'a Device) -> Self {
        Self { model, device }
    }

    fn input(&self, tokens: &[u32]) -> Tensor<Backend, 2, Int> {
        let values = tokens.iter().map(|&token| i64::from(token)).collect();
        Tensor::from_data(TensorData::new(values, [1, tokens.len()]), self.device)
    }

    fn prefill(&self, context: &[u32]) -> (InferenceState<Backend>, Tensor<Backend, 1>) {
        debug_assert!(!context.is_empty());
        let mut state = self.model.make_state(1, self.device);
        let logits = self
            .model
            .forward_with_state(self.input(context), &mut state);
        let vocab = self.model.config().vocab_size;
        let last = logits
            .slice([0..1, context.len() - 1..context.len(), 0..vocab])
            .reshape([vocab]);
        (state, last)
    }

    /// Generate tokens with one prefill followed by cached single-token steps.
    pub fn generate(&self, prompt_tokens: &[u32], config: &SamplingConfig) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            bail!("cannot generate from an empty prompt");
        }
        if config.top_k == Some(0) {
            bail!("top_k must be >= 1 (got 0)");
        }
        if config.max_new_tokens == 0 {
            return Ok(prompt_tokens.to_vec());
        }

        let max_seq_len = self.model.config().max_seq_len;
        let vocab = self.model.config().vocab_size;
        let mut tokens = prompt_tokens.to_vec();
        let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
        let context_len = tokens.len().min(max_seq_len);
        let (mut state, mut last_logits) = self.prefill(&tokens[tokens.len() - context_len..]);

        for _ in 0..config.max_new_tokens {
            let values = last_logits
                .clone()
                .into_data()
                .to_vec::<<Backend as burn::tensor::backend::BackendTypes>::FloatElem>()?
                .into_iter()
                .map(|value| value.elem::<f32>())
                .collect::<Vec<f32>>();
            let next_token = if config.temperature <= 0.0 {
                argmax(&values)
            } else {
                sample_from_logits(&values, config.temperature as f32, config.top_k, &mut rng)?
            };
            tokens.push(next_token);

            if config.eos_token == Some(next_token) {
                break;
            }

            if state.pos() >= max_seq_len {
                let keep = (max_seq_len / 2).max(1);
                (state, last_logits) = self.prefill(&tokens[tokens.len() - keep..]);
            } else {
                let logits = self
                    .model
                    .forward_with_state(self.input(&[next_token]), &mut state);
                last_logits = logits.slice([0..1, 0..1, 0..vocab]).reshape([vocab]);
            }
        }

        Ok(tokens)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index as u32)
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    indices.truncate(k.min(indices.len()));
    indices
}

fn sample_from_logits(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    rng: &mut impl rand::Rng,
) -> Result<u32> {
    if logits.is_empty() {
        bail!("cannot sample empty logits");
    }
    let candidates = match top_k {
        Some(k) => top_k_indices(logits, k),
        None => (0..logits.len()).collect(),
    };
    let max = candidates
        .iter()
        .map(|&index| logits[index] / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<_> = candidates
        .iter()
        .map(|&index| (logits[index] / temperature - max).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        bail!("sampling produced an invalid probability distribution");
    }

    let mut draw = rng.random::<f32>() * total;
    for (&index, weight) in candidates.iter().zip(weights) {
        draw -= weight;
        if draw <= 0.0 {
            return Ok(index as u32);
        }
    }
    Ok(*candidates.last().unwrap() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(argmax(&[0.1, 5.0, 0.2, -1.0]), 1);
    }

    #[test]
    fn top_k_keeps_highest_indices() {
        assert_eq!(top_k_indices(&[1.0, 3.0, 2.0, 0.5], 2), vec![1, 2]);
    }

    #[test]
    fn sampling_is_deterministic_under_seed() {
        let logits = [1.0, 2.0, 3.0, 0.5, 1.5];
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            assert_eq!(
                sample_from_logits(&logits, 0.8, Some(3), &mut a).unwrap(),
                sample_from_logits(&logits, 0.8, Some(3), &mut b).unwrap()
            );
        }
    }
}
