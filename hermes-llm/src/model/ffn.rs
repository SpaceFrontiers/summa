use candle_core::{Module, Result, Tensor};
use candle_nn::{Dropout, Linear, VarBuilder, linear, linear_no_bias};

use crate::mal::{BlockDef, ModelDef};

pub struct FeedForward {
    gate_proj: Option<Linear>,
    up_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
    use_swiglu: bool,
}

impl FeedForward {
    pub fn new(config: &ModelDef, block: &BlockDef, vb: VarBuilder) -> Result<Self> {
        let use_swiglu = block.use_swiglu();
        let use_gate = block.ffn.gate;
        let intermediate_size = block.intermediate_size(config.hidden_size);

        let gate_proj = if use_gate {
            Some(if block.use_bias() {
                linear(config.hidden_size, intermediate_size, vb.pp("gate_proj"))?
            } else {
                linear_no_bias(config.hidden_size, intermediate_size, vb.pp("gate_proj"))?
            })
        } else {
            None
        };

        let (up_proj, down_proj) = if block.use_bias() {
            (
                linear(config.hidden_size, intermediate_size, vb.pp("up_proj"))?,
                linear(intermediate_size, config.hidden_size, vb.pp("down_proj"))?,
            )
        } else {
            (
                linear_no_bias(config.hidden_size, intermediate_size, vb.pp("up_proj"))?,
                linear_no_bias(intermediate_size, config.hidden_size, vb.pp("down_proj"))?,
            )
        };
        let dropout = Dropout::new(block.dropout as f32);
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            dropout,
            use_swiglu,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let hidden = if let Some(gate_proj) = &self.gate_proj {
            let gate = gate_proj.forward(x)?;
            let up = self.up_proj.forward(x)?;
            if self.use_swiglu {
                let gate = candle_nn::ops::silu(&gate)?;
                (gate * up)?
            } else {
                let gate = gate.gelu_erf()?;
                (gate * up)?
            }
        } else {
            // No gating - simple MLP
            let h = self.up_proj.forward(x)?;
            if self.use_swiglu {
                candle_nn::ops::silu(&h)?
            } else {
                h.gelu_erf()?
            }
        };

        let hidden = self.dropout.forward(&hidden, train)?;
        self.down_proj.forward(&hidden)
    }
}
