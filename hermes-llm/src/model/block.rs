//! Hybrid Transformer and Mamba block assembly.

use burn::prelude::*;
use burn_nn::{Dropout, DropoutConfig, RotaryEncoding};

use crate::mal::{BlockDef, ModelDef, NormPosition};

use super::{AttnCache, FeedForward, MambaMixer, MambaState, MultiHeadAttention, Norm};

#[derive(Module, Debug)]
pub struct TransformerBlock {
    attention: Option<MultiHeadAttention>,
    ssm: Option<MambaMixer>,
    feed_forward: FeedForward,
    attn_norm: Norm,
    ffn_norm: Norm,
    residual_dropout: Dropout,
    #[module(skip)]
    norm_position: NormPosition,
    #[module(skip)]
    use_residual: bool,
}

impl TransformerBlock {
    pub fn new(config: &ModelDef, block: &BlockDef, device: &Device) -> Self {
        let (attention, ssm) = match &block.ssm {
            Some(ssm) => (None, Some(MambaMixer::new(config, ssm, device))),
            None => (Some(MultiHeadAttention::new(config, block, device)), None),
        };
        let make_norm = || {
            Norm::new(
                block.norm.norm_type,
                config.hidden_size,
                block.norm_eps(),
                device,
            )
        };
        Self {
            attention,
            ssm,
            feed_forward: FeedForward::new(config, block, device),
            attn_norm: make_norm(),
            ffn_norm: make_norm(),
            residual_dropout: DropoutConfig::new(block.dropout).init(),
            norm_position: block.norm_position,
            use_residual: block.residual,
        }
    }

    fn mix(&self, x: Tensor<3>, rope: &RotaryEncoding, start_pos: usize) -> Tensor<3> {
        match (&self.attention, &self.ssm) {
            (Some(attention), None) => attention.forward(x, rope, start_pos),
            (None, Some(ssm)) => ssm.forward(x),
            _ => unreachable!("a block has exactly one mixer"),
        }
    }

    fn residual(&self, x: Tensor<3>, branch: Tensor<3>) -> Tensor<3> {
        let branch = self.residual_dropout.forward(branch);
        if self.use_residual {
            x + branch
        } else {
            branch
        }
    }

    fn forward_with_mixer(
        &self,
        x: Tensor<3>,
        mut mix: impl FnMut(Tensor<3>) -> Tensor<3>,
    ) -> Tensor<3> {
        let x = match self.norm_position {
            NormPosition::Pre => {
                let branch = mix(self.attn_norm.forward(x.clone()));
                self.residual(x, branch)
            }
            NormPosition::Post => {
                let branch = mix(x.clone());
                self.attn_norm.forward(self.residual(x, branch))
            }
        };

        match self.norm_position {
            NormPosition::Pre => {
                let branch = self.feed_forward.forward(self.ffn_norm.forward(x.clone()));
                self.residual(x, branch)
            }
            NormPosition::Post => {
                let branch = self.feed_forward.forward(x.clone());
                self.ffn_norm.forward(self.residual(x, branch))
            }
        }
    }

    pub fn forward(&self, x: Tensor<3>, rope: &RotaryEncoding, start_pos: usize) -> Tensor<3> {
        self.forward_with_mixer(x, |x| self.mix(x, rope, start_pos))
    }

    pub fn make_state(&self, batch: usize, device: &Device) -> LayerState {
        match (&self.attention, &self.ssm) {
            (Some(attention), None) => LayerState::Attn(attention.make_cache(batch, device)),
            (None, Some(ssm)) => LayerState::Mamba(ssm.make_state(batch, device)),
            _ => unreachable!("a block has exactly one mixer"),
        }
    }

    pub(crate) fn prepare_inference(&mut self) {
        if let Some(attention) = &mut self.attention {
            attention.prepare_inference();
        }
        if let Some(ssm) = &mut self.ssm {
            ssm.prepare_inference();
        }
        self.feed_forward.prepare_inference();
    }

    pub fn forward_with_state(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
        state: &mut LayerState,
    ) -> Tensor<3> {
        self.forward_with_mixer(x, |x| match (&self.attention, &self.ssm, &mut *state) {
            (Some(attention), None, LayerState::Attn(cache)) => {
                attention.forward_cached(x, rope, start_pos, cache)
            }
            (None, Some(ssm), LayerState::Mamba(mamba)) => ssm.forward_with_state(x, Some(mamba)),
            _ => panic!("layer state type does not match the configured mixer type"),
        })
    }
}

#[derive(Debug, Clone)]
pub enum LayerState {
    Attn(AttnCache),
    Mamba(MambaState),
}

#[derive(Debug, Clone)]
pub struct InferenceState {
    pub(crate) layers: Vec<LayerState>,
    pub(crate) pos: usize,
}

impl InferenceState {
    pub fn pos(&self) -> usize {
        self.pos
    }
}
