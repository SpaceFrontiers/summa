use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding, linear_no_bias};

use crate::mal::{BlockDef, ModelDef};

use super::block::{InferenceState, TransformerBlock};
use super::norm::Norm;
use super::rope::RotaryEmbedding;

pub struct Transformer {
    embedding: Embedding,
    layers: Vec<TransformerBlock>,
    final_norm: Norm,
    /// None when embeddings.tie_weights: logits are projected through the
    /// embedding matrix and no lm_head.weight tensor exists in checkpoints.
    lm_head: Option<Linear>,
    rope: RotaryEmbedding,
    config: ModelDef,
}

impl Transformer {
    pub fn new(config: &ModelDef, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.hidden_size, vb.pp("embedding"))?;

        // The RoPE table is shared: all attention blocks must agree on
        // head_dim and theta.
        let attn_blocks: Vec<&BlockDef> = (0..config.num_layers)
            .map(|i| config.block_for_layer(i))
            .filter(|b| !b.is_ssm())
            .collect();
        // Only RoPE (or None) is implemented; fail loud rather than silently
        // applying RoPE to a block that configured alibi/learned encodings.
        for b in &attn_blocks {
            match &b.attention.position_encoding {
                crate::mal::PositionEncoding::Rope { .. } | crate::mal::PositionEncoding::None => {}
                other => candle_core::bail!(
                    "position_encoding {other:?} is not implemented for inference; \
                     only rope and none are supported"
                ),
            }
        }
        let (rope_head_dim, rope_theta, rope_scaling) = match attn_blocks.first() {
            Some(first) => {
                let hd = first.head_dim(config.hidden_size);
                let theta = first.rope_theta();
                let scaling = first.rope_scaling();
                for b in &attn_blocks {
                    if b.head_dim(config.hidden_size) != hd
                        || b.rope_theta() != theta
                        || b.rope_scaling() != scaling
                    {
                        candle_core::bail!(
                            "all attention blocks must share head_dim, rope theta, and \
                             rope scaling (shared RoPE table)"
                        );
                    }
                }
                (hd, theta, scaling)
            }
            // Pure-SSM model: RoPE is unused, build a minimal table
            None => (2, 10000.0, None),
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerBlock::new(
                config,
                config.block_for_layer(i),
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        // Final norm follows the first layer's norm style
        let norm_block = config.block_for_layer(0);
        let final_norm = Norm::new(
            norm_block.norm.norm_type,
            config.hidden_size,
            norm_block.norm_eps(),
            norm_block.use_bias(),
            vb.pp("final_norm"),
        )?;
        let lm_head = if config.embeddings.tie_weights {
            None
        } else {
            Some(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };
        let rope = RotaryEmbedding::new(
            rope_head_dim,
            config.max_seq_len,
            rope_theta,
            rope_scaling,
            vb.device(),
        )?;
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            rope,
            config: config.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, start_pos: usize, train: bool) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        let x = self.embedding.forward(input_ids)?;

        let mask = if seq_len > 1 {
            let mut mask_data = vec![0u8; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = 1;
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), input_ids.device())?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            Some(mask)
        } else {
            None
        };

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, mask.as_ref(), &self.rope, start_pos, train)?;
        }

        let x = self.final_norm.forward(&x)?;
        self.project_logits(&x)
    }

    /// Project hidden states to vocab logits — through lm_head, or through
    /// the (live, load-aware) embedding matrix when weights are tied.
    fn project_logits(&self, x: &Tensor) -> Result<Tensor> {
        match &self.lm_head {
            Some(head) => head.forward(x),
            None => {
                let (b, l, h) = x.dims3()?;
                let w = self.embedding.embeddings(); // [V, h]
                let logits = x.reshape((b * l, h))?.matmul(&w.t()?)?;
                logits.reshape((b, l, w.dim(0)?))
            }
        }
    }

    pub fn config(&self) -> &ModelDef {
        &self.config
    }

    pub fn num_parameters(&self) -> usize {
        self.config.estimated_params()
    }

    /// Create an empty incremental inference state (batch size fixed).
    pub fn make_state(&self, batch: usize, device: &Device) -> Result<InferenceState> {
        let layers = self
            .layers
            .iter()
            .map(|l| l.make_state(batch, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(InferenceState { layers, pos: 0 })
    }

    /// Incremental forward: processes `input_ids` (prompt prefill or a single
    /// decode token) continuing from `state`, and advances it. Attention
    /// layers pay O(context) per token via their KV cache; Mamba layers pay
    /// O(1) via their recurrent state. Positions beyond max_seq_len are
    /// rejected (RoPE table bound) — callers re-prefill on overflow.
    pub fn forward_with_state(
        &self,
        input_ids: &Tensor,
        state: &mut InferenceState,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        if state.pos + seq_len > self.config.max_seq_len {
            candle_core::bail!(
                "inference state at position {} + {} tokens exceeds max_seq_len {}",
                state.pos,
                seq_len,
                self.config.max_seq_len
            );
        }

        let mut x = self.embedding.forward(input_ids)?;
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            x = layer.forward_with_state(&x, &self.rope, state.pos, layer_state)?;
        }
        state.pos += seq_len;

        let x = self.final_norm.forward(&x)?;
        self.project_logits(&x)
    }
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    let logits = logits.reshape((batch_size * seq_len, vocab_size))?;
    let targets = targets.reshape((batch_size * seq_len,))?;
    candle_nn::loss::cross_entropy(&logits, &targets)
}
