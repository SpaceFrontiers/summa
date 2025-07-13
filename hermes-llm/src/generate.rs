//! Autoregressive text generation.

use anyhow::{Result, bail};
use burn::tensor::activation::softmax;
use burn::tensor::{Int, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{Device, InferenceState, Transformer};

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
    model: &'a Transformer,
    device: &'a Device,
}

impl<'a> TextGenerator<'a> {
    pub fn new(model: &'a Transformer, device: &'a Device) -> Self {
        Self { model, device }
    }

    fn input(&self, tokens: &[u32]) -> Tensor<2, Int> {
        let values = tokens.iter().map(|&token| i64::from(token)).collect();
        Tensor::from_data(TensorData::new(values, [1, tokens.len()]), self.device)
    }

    fn prefill(&self, context: &[u32]) -> (InferenceState, Tensor<1>) {
        debug_assert!(!context.is_empty());
        let mut state = self.model.make_state(1, self.device);
        let logits = self
            .model
            .forward_next_logits_with_state(self.input(context), &mut state);
        let vocab = self.model.config().vocab_size;
        let last = logits.reshape([vocab]);
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
        if config.temperature.is_nan() {
            bail!("temperature must not be NaN");
        }
        if config.max_new_tokens == 0 {
            return Ok(prompt_tokens.to_vec());
        }

        let max_seq_len = self.model.config().max_seq_len;
        let vocab = self.model.config().vocab_size;
        let mut tokens = prompt_tokens.to_vec();
        let seed = config.seed.unwrap_or_else(rand::random);
        self.device.seed(seed);
        let mut rng = StdRng::seed_from_u64(seed);
        let context_len = tokens.len().min(max_seq_len);
        let (mut state, mut last_logits) = self.prefill(&tokens[tokens.len() - context_len..]);

        for _ in 0..config.max_new_tokens {
            let next_token = if config.temperature <= 0.0 {
                last_logits.clone().argmax(0).try_into_scalar::<i64>()? as u32
            } else {
                sample_from_logits(
                    last_logits.clone(),
                    config.temperature as f32,
                    config.top_k,
                    &mut rng,
                )?
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
                    .forward_next_logits_with_state(self.input(&[next_token]), &mut state);
                last_logits = logits.reshape([vocab]);
            }
        }

        Ok(tokens)
    }
}

fn sample_from_logits(
    logits: Tensor<1>,
    temperature: f32,
    top_k: Option<usize>,
    rng: &mut impl Rng,
) -> Result<u32> {
    if !temperature.is_finite() || temperature <= 0.0 {
        bail!("sampling temperature must be finite and positive");
    }
    let vocab = logits.dims()[0];
    if let Some(k) = top_k {
        let values = logits.into_data().convert::<f32>().to_vec::<f32>()?;
        let mut indices = (0..vocab).collect::<Vec<_>>();
        if k < vocab {
            indices.select_nth_unstable_by(k, |left, right| {
                values[*right]
                    .total_cmp(&values[*left])
                    .then_with(|| left.cmp(right))
            });
            indices.truncate(k);
        }

        let inverse_temperature = 1.0 / f64::from(temperature);
        let maximum = indices
            .iter()
            .map(|&index| values[index])
            .fold(f32::NEG_INFINITY, f32::max);
        let weight =
            |index: usize| ((f64::from(values[index] - maximum)) * inverse_temperature).exp();
        let total = indices.iter().map(|&index| weight(index)).sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            bail!("cannot sample from non-finite logits");
        }
        let mut draw = rng.random::<f64>() * total;
        for &index in &indices {
            draw -= weight(index);
            if draw <= 0.0 {
                return Ok(index as u32);
            }
        }
        return Ok(*indices.last().expect("top_k is validated as positive") as u32);
    }
    Ok(softmax(logits.div_scalar(temperature), 0)
        .categorical(1)
        .try_into_scalar::<i64>()? as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_highest() {
        let device = Device::ndarray();
        let logits = Tensor::<1>::from_floats([0.1, 5.0, 0.2, -1.0], &device);
        assert_eq!(logits.argmax(0).into_scalar::<i64>(), 1);
    }

    #[test]
    fn sampling_is_deterministic_under_seed() {
        let sample = || {
            let device = Device::ndarray();
            let mut rng = StdRng::seed_from_u64(42);
            let logits = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 0.5, 1.5], &device);
            sample_from_logits(logits, 0.8, Some(3), &mut rng).unwrap()
        };
        assert_eq!(sample(), sample());
    }
}
