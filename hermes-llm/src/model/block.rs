//! Hybrid Transformer and Mamba block assembly.

use burn::module::ParamId;
use burn::prelude::*;
use burn_nn::{Dropout, DropoutConfig, RotaryEncoding};

use crate::mal::{BlockDef, ModelDef, NormPosition};

use super::{
    AttnCache, FeedForward, MambaMixer, MambaState, MemoryChain, MemoryRouting, MemorySlotStatus,
    MultiHeadAttention, Norm,
};

pub(crate) struct BlockDiagnostic {
    pub attention_weights: Option<Tensor<4>>,
    pub total_attention_heads: Option<usize>,
    pub mamba_state: Option<Tensor<3>>,
}

/// One MAL-resolved residual block with an attention or Mamba sequence mixer.
///
/// The mixer is followed by either one ordinary [`FeedForward`] branch or an
/// opt-in fast-to-slow [`MemoryChain`], never both.
#[derive(Module, Debug)]
pub struct TransformerBlock {
    attention: Option<MultiHeadAttention>,
    ssm: Option<MambaMixer>,
    feed_forward: Option<FeedForward>,
    memory: Option<MemoryChain>,
    attn_norm: Norm,
    ffn_norm: Norm,
    residual_dropout: Dropout,
    #[module(skip)]
    norm_position: NormPosition,
    #[module(skip)]
    use_residual: bool,
}

impl TransformerBlock {
    pub fn new(config: &ModelDef, block: &BlockDef, layer: usize, device: &Device) -> Self {
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
            feed_forward: block
                .memory
                .is_none()
                .then(|| FeedForward::new(config, block, device)),
            memory: block
                .memory
                .as_ref()
                .map(|memory| MemoryChain::new(config, block, memory, layer, device)),
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
        let mut branch = self.residual_dropout.forward(branch);
        if self.use_residual {
            // Incremental decode keeps an FP32 stream while the training
            // projections emit the BF16 compute dtype; align the branch to
            // the stream. During training both sides already match (BF16).
            if branch.dtype() != x.dtype() {
                branch = branch.cast(x.dtype());
            }
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
        self.forward_with_mixer_and_routing(x, &mut mix, MemoryRouting::Wake)
    }

    fn forward_with_mixer_and_routing(
        &self,
        x: Tensor<3>,
        mut mix: impl FnMut(Tensor<3>) -> Tensor<3>,
        routing: MemoryRouting,
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
        if let Some(memory) = &self.memory {
            let (output, _) = match self.norm_position {
                NormPosition::Pre => memory.forward_with(
                    x,
                    false,
                    routing,
                    |state| self.ffn_norm.forward(state),
                    |state, branch| self.residual(state, branch),
                ),
                NormPosition::Post => memory.forward_with(
                    x,
                    false,
                    routing,
                    |state| state,
                    |state, branch| self.ffn_norm.forward(self.residual(state, branch)),
                ),
            };
            return output;
        }
        let feed_forward = self
            .feed_forward
            .as_ref()
            .expect("a block has either an FFN or a memory chain");
        match self.norm_position {
            NormPosition::Pre => {
                let branch = feed_forward
                    .forward_internal_with_routing(self.ffn_norm.forward(x.clone()), false, routing)
                    .0;
                self.residual(x, branch)
            }
            NormPosition::Post => {
                let branch = feed_forward
                    .forward_internal_with_routing(x.clone(), false, routing)
                    .0;
                self.ffn_norm.forward(self.residual(x, branch))
            }
        }
    }

    fn forward_with_mixer_and_aux(
        &self,
        x: Tensor<3>,
        mut mix: impl FnMut(Tensor<3>) -> Tensor<3>,
    ) -> (Tensor<3>, Option<Tensor<1>>) {
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

        if let Some(memory) = &self.memory {
            return match self.norm_position {
                NormPosition::Pre => memory.forward_with(
                    x,
                    true,
                    MemoryRouting::Wake,
                    |state| self.ffn_norm.forward(state),
                    |state, branch| self.residual(state, branch),
                ),
                NormPosition::Post => memory.forward_with(
                    x,
                    true,
                    MemoryRouting::Wake,
                    |state| state,
                    |state, branch| self.ffn_norm.forward(self.residual(state, branch)),
                ),
            };
        }
        let feed_forward = self
            .feed_forward
            .as_ref()
            .expect("a block has either an FFN or a memory chain");
        let (output, auxiliary) = match self.norm_position {
            NormPosition::Pre => {
                let (branch, auxiliary) =
                    feed_forward.forward_with_aux(self.ffn_norm.forward(x.clone()));
                (self.residual(x, branch), auxiliary)
            }
            NormPosition::Post => {
                let (branch, auxiliary) = feed_forward.forward_with_aux(x.clone());
                (self.ffn_norm.forward(self.residual(x, branch)), auxiliary)
            }
        };
        (output, auxiliary)
    }

    pub fn forward(&self, x: Tensor<3>, rope: &RotaryEncoding, start_pos: usize) -> Tensor<3> {
        self.forward_with_mixer(x, |x| self.mix(x, rope, start_pos))
    }

    pub(crate) fn forward_with_routing(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
        routing: MemoryRouting,
    ) -> Tensor<3> {
        self.forward_with_mixer_and_routing(x, |x| self.mix(x, rope, start_pos), routing)
    }

    pub(crate) fn forward_with_aux(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
    ) -> (Tensor<3>, Option<Tensor<1>>) {
        self.forward_with_mixer_and_aux(x, |x| self.mix(x, rope, start_pos))
    }

    /// Run the exact block path while retaining bounded values needed by the
    /// opt-in visualization command.
    pub(crate) fn forward_diagnostic(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
        max_attention_heads: usize,
    ) -> (Tensor<3>, BlockDiagnostic) {
        let mut diagnostic = BlockDiagnostic {
            attention_weights: None,
            total_attention_heads: None,
            mamba_state: None,
        };
        let output = self.forward_with_mixer(x, |mixer_input| match (&self.attention, &self.ssm) {
            (Some(attention), None) => {
                diagnostic.attention_weights = Some(attention.diagnostic_weights(
                    mixer_input.clone(),
                    rope,
                    start_pos,
                    max_attention_heads,
                ));
                diagnostic.total_attention_heads = Some(attention.num_heads());
                attention.forward(mixer_input, rope, start_pos)
            }
            (None, Some(ssm)) => {
                let mut state = ssm.make_state(mixer_input.dims()[0], &mixer_input.device());
                let output = ssm.forward_with_state(mixer_input, Some(&mut state));
                diagnostic.mamba_state = Some(state.h);
                output
            }
            _ => unreachable!("a block has exactly one mixer"),
        });
        (output, diagnostic)
    }

    pub fn make_state(&self, batch: usize, device: &Device) -> LayerState {
        self.make_state_with_capacity(batch, usize::MAX, device)
    }

    pub fn make_state_with_capacity(
        &self,
        batch: usize,
        cache_capacity: usize,
        device: &Device,
    ) -> LayerState {
        match (&self.attention, &self.ssm) {
            (Some(attention), None) => {
                let cache = if cache_capacity == usize::MAX {
                    attention.make_cache(batch, device)
                } else {
                    attention.make_cache_with_capacity(batch, cache_capacity, device)
                };
                LayerState::Attn(cache)
            }
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
        if let Some(feed_forward) = &mut self.feed_forward {
            feed_forward.prepare_inference();
        }
        if let Some(memory) = &mut self.memory {
            memory.prepare_inference();
        }
    }

    pub(crate) fn non_muon_parameter_ids(&self) -> Vec<ParamId> {
        match (&self.feed_forward, &self.memory) {
            (Some(feed_forward), None) => feed_forward.router_parameter_id().into_iter().collect(),
            (None, Some(memory)) => memory.non_muon_parameter_ids(),
            _ => unreachable!("a block has either an FFN or a memory chain"),
        }
    }

    pub(crate) fn wake_parameter_counts(&self) -> anyhow::Result<(usize, usize)> {
        let stored = self.num_params();
        let (ffn_stored, ffn_routed) = match (&self.feed_forward, &self.memory) {
            (Some(feed_forward), None) => feed_forward.wake_parameter_counts(),
            (None, Some(memory)) => memory.wake_parameter_counts()?,
            _ => unreachable!("a block has either an FFN or a memory chain"),
        };
        let routed = stored
            .checked_sub(ffn_stored)
            .and_then(|count| count.checked_add(ffn_routed))
            .ok_or_else(|| anyhow::anyhow!("block parameter accounting overflow"))?;
        Ok((stored, routed))
    }

    pub fn forward_with_state(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
        state: &mut LayerState,
    ) -> Tensor<3> {
        self.forward_with_state_and_routing(x, rope, start_pos, state, MemoryRouting::Wake)
    }

    pub(crate) fn forward_with_state_and_routing(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
        state: &mut LayerState,
        routing: MemoryRouting,
    ) -> Tensor<3> {
        self.forward_with_mixer_and_routing(
            x,
            |x| match (&self.attention, &self.ssm, &mut *state) {
                (Some(attention), None, LayerState::Attn(cache)) => {
                    attention.forward_cached(x, rope, start_pos, cache)
                }
                (None, Some(ssm), LayerState::Mamba(mamba)) => {
                    ssm.forward_with_state(x, Some(mamba))
                }
                _ => panic!("layer state type does not match the configured mixer type"),
            },
            routing,
        )
    }

    pub(crate) fn sync_memory_state(&mut self) {
        if let Some(memory) = &mut self.memory {
            memory.sync_state();
        }
    }

    pub(crate) fn validate_memory_checkpoint_state(&self) -> anyhow::Result<()> {
        if let Some(memory) = &self.memory {
            memory.validate_checkpoint_state()?;
        }
        Ok(())
    }

    pub(crate) fn prepare_memory_upgrade_state(&mut self) {
        if let Some(memory) = &mut self.memory {
            memory.prepare_upgrade_state();
        }
    }

    pub(crate) fn memory_statuses(&self, layer: usize) -> Vec<MemorySlotStatus> {
        self.memory
            .as_ref()
            .map_or_else(Vec::new, |memory| memory.statuses(layer))
    }

    pub(crate) fn activate_memory_slot(&mut self, tier: usize, slot: usize) -> anyhow::Result<()> {
        self.memory
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer has no memory chain"))?
            .activate_slot(tier, slot)
    }

    pub(crate) fn deactivate_memory_slot(
        &mut self,
        tier: usize,
        slot: usize,
    ) -> anyhow::Result<()> {
        self.memory
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer has no memory chain"))?
            .deactivate_slot(tier, slot)
    }

    pub(crate) fn reset_memory_slot(
        &mut self,
        tier: usize,
        slot: usize,
        seed: u64,
    ) -> anyhow::Result<()> {
        self.memory
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer has no memory chain"))?
            .reset_slot(tier, slot, seed)
    }

    pub(crate) fn memory_tier_parameter_ids(&self, tier: usize) -> anyhow::Result<Vec<ParamId>> {
        self.memory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer has no memory chain"))?
            .tier_parameter_ids(tier)
    }

    pub(crate) fn has_memory_tier(&self, tier: usize) -> bool {
        self.memory
            .as_ref()
            .is_some_and(|memory| memory.has_tier(tier))
    }

    pub(crate) fn memory_tier_active_parameter_ids(
        &self,
        tier: usize,
    ) -> anyhow::Result<Vec<ParamId>> {
        self.memory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer has no memory chain"))?
            .tier_active_parameter_ids(tier)
    }

    pub(crate) fn dormant_memory_parameter_ids(&self) -> Vec<ParamId> {
        self.memory
            .as_ref()
            .map_or_else(Vec::new, MemoryChain::dormant_parameter_ids)
    }

    pub(crate) fn memory_tier_base_parameter_ids(
        &self,
        tier: usize,
    ) -> anyhow::Result<Vec<ParamId>> {
        self.memory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer has no memory chain"))?
            .tier_base_parameter_ids(tier)
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
    pub(crate) capacity: usize,
}

impl InferenceState {
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Maximum number of positions accepted before this state must be rebuilt.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
