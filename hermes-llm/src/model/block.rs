use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::mal::{BlockDef, ModelDef, NormPosition};

use super::attention::{AttnCache, MultiHeadAttention};
use super::ffn::FeedForward;
use super::mamba::{MambaMixer, MambaState};
use super::norm::Norm;
use super::rope::RotaryEmbedding;

/// Block mixer: attention or Mamba SSM
pub enum Mixer {
    Attention(MultiHeadAttention),
    Mamba(MambaMixer),
}

pub struct TransformerBlock {
    mixer: Mixer,
    feed_forward: FeedForward,
    attn_norm: Norm,
    ffn_norm: Norm,
    norm_position: NormPosition,
    use_residual: bool,
}

impl TransformerBlock {
    pub fn new(config: &ModelDef, block: &BlockDef, vb: VarBuilder) -> Result<Self> {
        let mixer = match &block.ssm {
            Some(ssm) => Mixer::Mamba(MambaMixer::new(config, ssm, vb.pp("ssm"))?),
            None => Mixer::Attention(MultiHeadAttention::new(config, block, vb.pp("attention"))?),
        };
        let feed_forward = FeedForward::new(config, block, vb.pp("feed_forward"))?;
        let norm_type = block.norm.norm_type;
        // Norm names are positional and shared by both mixer types
        let attn_norm = Norm::new(
            norm_type,
            config.hidden_size,
            block.norm_eps(),
            block.use_bias(),
            vb.pp("attn_norm"),
        )?;
        let ffn_norm = Norm::new(
            norm_type,
            config.hidden_size,
            block.norm_eps(),
            block.use_bias(),
            vb.pp("ffn_norm"),
        )?;
        Ok(Self {
            mixer,
            feed_forward,
            attn_norm,
            ffn_norm,
            norm_position: block.norm_position,
            use_residual: block.residual,
        })
    }

    fn mix(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &RotaryEmbedding,
        start_pos: usize,
        train: bool,
    ) -> Result<Tensor> {
        match &self.mixer {
            Mixer::Attention(attn) => attn.forward(x, mask, rope, start_pos, train),
            Mixer::Mamba(ssm) => ssm.forward(x),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &RotaryEmbedding,
        start_pos: usize,
        train: bool,
    ) -> Result<Tensor> {
        // Mixer (attention or SSM)
        let x = match self.norm_position {
            NormPosition::Pre => {
                let h = self.attn_norm.forward(x)?;
                let h = self.mix(&h, mask, rope, start_pos, train)?;
                if self.use_residual { (x + h)? } else { h }
            }
            NormPosition::Post => {
                let h = self.mix(x, mask, rope, start_pos, train)?;
                let h = if self.use_residual { (x + h)? } else { h };
                self.attn_norm.forward(&h)?
            }
        };

        // FFN
        match self.norm_position {
            NormPosition::Pre => {
                let h = self.ffn_norm.forward(&x)?;
                let h = self.feed_forward.forward(&h, train)?;
                if self.use_residual { &x + h } else { Ok(h) }
            }
            NormPosition::Post => {
                let h = self.feed_forward.forward(&x, train)?;
                let h = if self.use_residual { (&x + h)? } else { h };
                self.ffn_norm.forward(&h)
            }
        }
    }

    /// Create the empty per-layer inference state for this block.
    pub fn make_state(&self, batch: usize, device: &Device) -> Result<LayerState> {
        match &self.mixer {
            Mixer::Attention(_) => Ok(LayerState::Attn(AttnCache::default())),
            Mixer::Mamba(ssm) => Ok(LayerState::Mamba(MambaState {
                conv: Tensor::zeros(
                    (batch, ssm.d_inner, ssm.conv_kernel - 1),
                    DType::F32,
                    device,
                )?,
                h: Tensor::zeros((batch, ssm.d_inner, ssm.state_dim), DType::F32, device)?,
            })),
        }
    }

    /// Stateful forward: attention layers read/extend their KV cache, Mamba
    /// layers advance their recurrent state.
    pub fn forward_with_state(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        start_pos: usize,
        state: &mut LayerState,
    ) -> Result<Tensor> {
        let mix = |h: &Tensor, state: &mut LayerState| -> Result<Tensor> {
            match (&self.mixer, state) {
                (Mixer::Attention(attn), LayerState::Attn(cache)) => {
                    attn.forward_cached(h, rope, start_pos, cache)
                }
                (Mixer::Mamba(ssm), LayerState::Mamba(ms)) => ssm.forward_with_state(h, Some(ms)),
                _ => candle_core::bail!("layer state type does not match mixer type"),
            }
        };

        // Mixer (attention or SSM)
        let x = match self.norm_position {
            NormPosition::Pre => {
                let h = self.attn_norm.forward(x)?;
                let h = mix(&h, state)?;
                if self.use_residual { (x + h)? } else { h }
            }
            NormPosition::Post => {
                let h = mix(x, state)?;
                let h = if self.use_residual { (x + h)? } else { h };
                self.attn_norm.forward(&h)?
            }
        };

        // FFN (inference: train=false)
        match self.norm_position {
            NormPosition::Pre => {
                let h = self.ffn_norm.forward(&x)?;
                let h = self.feed_forward.forward(&h, false)?;
                if self.use_residual { &x + h } else { Ok(h) }
            }
            NormPosition::Post => {
                let h = self.feed_forward.forward(&x, false)?;
                let h = if self.use_residual { (&x + h)? } else { h };
                self.ffn_norm.forward(&h)
            }
        }
    }
}

/// Per-layer inference state: KV cache for attention, recurrent state for Mamba
pub enum LayerState {
    Attn(AttnCache),
    Mamba(MambaState),
}

/// Whole-model incremental inference state.
///
/// Attention layers keep a KV cache (grows with generated length up to
/// max_seq_len); Mamba layers keep a fixed-size recurrent state — the
/// compact persistent memory.
pub struct InferenceState {
    pub(crate) layers: Vec<LayerState>,
    pub(crate) pos: usize,
}

impl InferenceState {
    pub fn pos(&self) -> usize {
        self.pos
    }
}
