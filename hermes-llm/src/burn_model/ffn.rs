//! Position-wise feed-forward network.
//!
//! Gated or plain MLP using the activation selected by MAL.

use burn::prelude::*;
use burn::tensor::activation::{gelu, gelu_approximate, relu, silu};
use burn_nn::{Dropout, DropoutConfig, Linear, LinearConfig};

use crate::mal::{Activation, BlockDef, ModelDef};

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    gate_proj: Option<Linear<B>>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(config: &ModelDef, block: &BlockDef, device: &burn::tensor::Device<B>) -> Self {
        let use_gate = block.ffn.gate;
        let use_bias = block.ffn.bias;
        let hidden = config.hidden_size;
        let intermediate = block.intermediate_size(hidden);

        let lin = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out)
                .with_bias(use_bias)
                .init(device)
        };

        let gate_proj = use_gate.then(|| lin(hidden, intermediate));
        let up_proj = lin(hidden, intermediate);
        let down_proj = lin(intermediate, hidden);

        Self {
            gate_proj,
            up_proj,
            down_proj,
            dropout: DropoutConfig::new(block.ffn.dropout).init(),
            activation: block.ffn.activation,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let act = |t: Tensor<B, D>| match self.activation {
            Activation::SwiGLU | Activation::SiLU => silu(t),
            Activation::GELU => gelu(t),
            Activation::ReLU => relu(t),
            Activation::GELUNew | Activation::GELUTanh => gelu_approximate(t),
        };
        let hidden = match &self.gate_proj {
            Some(gate_proj) => {
                let gate = act(gate_proj.forward(x.clone()));
                let up = self.up_proj.forward(x);
                gate * up
            }
            None => act(self.up_proj.forward(x)),
        };
        self.down_proj.forward(self.dropout.forward(hidden))
    }
}
