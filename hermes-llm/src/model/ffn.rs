//! Position-wise feed-forward layer.
//!
//! Gated or plain MLP using the activation selected by MAL.

use burn::prelude::*;
use burn::tensor::activation::{gelu, gelu_approximate, relu, silu};
use burn_nn::{Dropout, DropoutConfig, Linear, LinearConfig};

use crate::mal::{Activation, BlockDef, ModelDef};

use super::matmul::linear;

#[derive(Module, Debug)]
pub struct FeedForward {
    in_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
    #[module(skip)]
    intermediate: usize,
    #[module(skip)]
    gated: bool,
}

impl FeedForward {
    pub fn new(config: &ModelDef, block: &BlockDef, device: &Device) -> Self {
        let use_gate = block.ffn.gate;
        let use_bias = block.ffn.bias;
        let hidden = config.hidden_size;
        let intermediate = block.intermediate_size(hidden);

        let lin = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out)
                .with_bias(use_bias)
                .init(device)
        };

        let in_proj = lin(hidden, intermediate * if use_gate { 2 } else { 1 });
        let down_proj = lin(intermediate, hidden);

        Self {
            in_proj,
            down_proj,
            dropout: DropoutConfig::new(block.ffn.dropout).init(),
            activation: block.ffn.activation,
            intermediate,
            gated: use_gate,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let act = |t: Tensor<D>| match self.activation {
            Activation::SwiGLU | Activation::SiLU => silu(t),
            Activation::GELU => gelu(t),
            Activation::ReLU => relu(t),
            Activation::GELUNew | Activation::GELUTanh => gelu_approximate(t),
        };
        let projected = linear(&self.in_proj, x);
        let hidden = if self.gated {
            let mut ranges = projected.dims().map(|size| 0..size);
            ranges[D - 1] = 0..self.intermediate;
            let gate = act(projected.clone().slice(ranges.clone()));
            ranges[D - 1] = self.intermediate..2 * self.intermediate;
            gate * projected.slice(ranges)
        } else {
            act(projected)
        };
        linear(&self.down_proj, self.dropout.forward(hidden))
    }
}
