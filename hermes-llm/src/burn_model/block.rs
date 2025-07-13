//! Hybrid Transformer/Mamba block assembly.

use burn::prelude::*;
use burn_nn::{Dropout, DropoutConfig, RotaryEncoding};

use crate::mal::{BlockDef, ModelDef, NormPosition};

use super::{
    AttnCache, FeedForward, MambaBackend, MambaMixer, MambaState, MultiHeadAttention, Norm,
};

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: Option<MultiHeadAttention<B>>,
    ssm: Option<MambaMixer<B>>,
    feed_forward: FeedForward<B>,
    attn_norm: Norm<B>,
    ffn_norm: Norm<B>,
    residual_dropout: Dropout,
    #[module(skip)]
    norm_position: NormPosition,
    #[module(skip)]
    use_residual: bool,
}

impl<B: MambaBackend> TransformerBlock<B> {
    pub fn new(config: &ModelDef, block: &BlockDef, device: &Device<B>) -> Self {
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

    fn mix(&self, x: Tensor<B, 3>, rope: &RotaryEncoding<B>, start_pos: usize) -> Tensor<B, 3> {
        match (&self.attention, &self.ssm) {
            (Some(attention), None) => attention.forward(x, rope, start_pos),
            (None, Some(ssm)) => ssm.forward(x),
            _ => unreachable!("a block has exactly one mixer"),
        }
    }

    fn residual(&self, x: Tensor<B, 3>, branch: Tensor<B, 3>) -> Tensor<B, 3> {
        let branch = self.residual_dropout.forward(branch);
        if self.use_residual {
            x + branch
        } else {
            branch
        }
    }

    fn forward_with_mixer(
        &self,
        x: Tensor<B, 3>,
        mut mix: impl FnMut(Tensor<B, 3>) -> Tensor<B, 3>,
    ) -> Tensor<B, 3> {
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

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        self.forward_with_mixer(x, |x| self.mix(x, rope, start_pos))
    }

    pub fn make_state(&self, batch: usize, device: &Device<B>) -> LayerState<B> {
        match (&self.attention, &self.ssm) {
            (Some(_), None) => LayerState::Attn(AttnCache::default()),
            (None, Some(ssm)) => LayerState::Mamba(ssm.make_state(batch, device)),
            _ => unreachable!("a block has exactly one mixer"),
        }
    }

    pub fn forward_with_state(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        start_pos: usize,
        state: &mut LayerState<B>,
    ) -> Tensor<B, 3> {
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
pub enum LayerState<B: Backend> {
    Attn(AttnCache<B>),
    Mamba(MambaState<B>),
}

#[derive(Debug, Clone)]
pub struct InferenceState<B: Backend> {
    pub(crate) layers: Vec<LayerState<B>>,
    pub(crate) pos: usize,
}

impl<B: Backend> InferenceState<B> {
    pub fn pos(&self) -> usize {
        self.pos
    }
}
