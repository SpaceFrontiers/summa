use candle_core::{Module, Result, Tensor};
use candle_nn::{Dropout, Linear, VarBuilder, linear, linear_no_bias};

use crate::mal::{BlockDef, ModelDef};

use super::norm::RMSNorm;
use super::rope::RotaryEmbedding;

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = on_false.shape();
    let mask = mask.broadcast_as(shape.dims())?;
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Per-head RMSNorm over head_dim, applied to Q/K before RoPE (qk_norm)
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dropout: Dropout,
    window_size: Option<usize>,
    causal: bool,
}

impl MultiHeadAttention {
    pub fn new(config: &ModelDef, block: &BlockDef, vb: VarBuilder) -> Result<Self> {
        let num_heads = block.num_heads();
        let num_kv_heads = block.num_kv_heads();
        let head_dim = block.head_dim(config.hidden_size);
        let hidden_size = config.hidden_size;
        let kv_dim = num_kv_heads * head_dim;

        let (q_proj, k_proj, v_proj, o_proj) = if block.use_bias() {
            (
                linear(hidden_size, hidden_size, vb.pp("q_proj"))?,
                linear(hidden_size, kv_dim, vb.pp("k_proj"))?,
                linear(hidden_size, kv_dim, vb.pp("v_proj"))?,
                linear(hidden_size, hidden_size, vb.pp("o_proj"))?,
            )
        } else {
            (
                linear_no_bias(hidden_size, hidden_size, vb.pp("q_proj"))?,
                linear_no_bias(hidden_size, kv_dim, vb.pp("k_proj"))?,
                linear_no_bias(hidden_size, kv_dim, vb.pp("v_proj"))?,
                linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?,
            )
        };
        let (q_norm, k_norm) = if block.attention.qk_norm {
            (
                Some(RMSNorm::new(head_dim, block.norm_eps(), vb.pp("q_norm"))?),
                Some(RMSNorm::new(head_dim, block.norm_eps(), vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };
        let dropout = Dropout::new(block.dropout as f32);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            dropout,
            window_size: block.attention.window_size,
            causal: block.attention.causal,
        })
    }

    /// Apply QK-Norm (when configured) to per-head Q/K, pre-RoPE.
    fn apply_qk_norm(&self, q: Tensor, k: Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.q_norm, &self.k_norm) {
            (Some(qn), Some(kn)) => Ok((qn.forward(&q)?, kn.forward(&k)?)),
            _ => Ok((q, k)),
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
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = self.apply_qk_norm(q, k)?;
        let (q, k) = rope.apply(&q, &k, start_pos)?;

        // Repeat KV heads for GQA (Grouped Query Attention)
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            let k = k
                .unsqueeze(2)?
                .expand((batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim))?
                .reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
            let v = v
                .unsqueeze(2)?
                .expand((batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim))?
                .reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
            (k, v)
        } else {
            (k, v)
        };

        // Use Flash Attention if available (CUDA only), otherwise standard attention
        #[cfg(feature = "flash-attn")]
        let attn_output = {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1.0 / (self.head_dim as f32).sqrt();
            let attn = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, self.causal)?;
            attn.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?
        };

        #[cfg(not(feature = "flash-attn"))]
        let attn_output = {
            let scale = (self.head_dim as f64).sqrt();
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn_weights = q.matmul(&k_t)?.affine(1.0 / scale, 0.0)?;

            // Apply causal mask if needed
            let attn_weights = if self.causal {
                match mask {
                    Some(m) => masked_fill(&attn_weights, m, f32::NEG_INFINITY)?,
                    None => attn_weights,
                }
            } else {
                attn_weights
            };

            // Apply sliding window mask if configured
            let attn_weights = if let Some(window) = self.window_size {
                let device = attn_weights.device();
                let mut window_mask = vec![0u8; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if (i as isize - j as isize).unsigned_abs() > window {
                            window_mask[i * seq_len + j] = 1;
                        }
                    }
                }
                let window_mask = Tensor::from_vec(window_mask, (seq_len, seq_len), device)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?;
                masked_fill(&attn_weights, &window_mask, f32::NEG_INFINITY)?
            } else {
                attn_weights
            };

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_weights = if train {
                self.dropout.forward(&attn_weights, train)?
            } else {
                attn_weights
            };

            let output = attn_weights.matmul(&v)?;
            let output = output.transpose(1, 2)?.contiguous()?;
            output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?
        };

        self.o_proj.forward(&attn_output)
    }

    /// Incremental attention over a KV cache. `x` holds S new tokens at
    /// global positions `start_pos..start_pos+S`; K/V for them are appended
    /// to the cache and attention runs over the full cached sequence.
    pub fn forward_cached(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        start_pos: usize,
        cache: &mut AttnCache,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.apply_qk_norm(q, k)?;
        let (q, k) = rope.apply(&q, &k, start_pos)?;

        // Append to cache (pre-GQA-repeat, [B, n_kv, T, hd])
        let (k_all, v_all) = match (&cache.k, &cache.v) {
            (Some(ck), Some(cv)) => (
                Tensor::cat(&[ck, &k], 2)?.contiguous()?,
                Tensor::cat(&[cv, &v], 2)?.contiguous()?,
            ),
            _ => (k, v),
        };
        cache.k = Some(k_all.clone());
        cache.v = Some(v_all.clone());
        let total_len = k_all.dim(2)?;

        // GQA repeat over the full cache
        let (k_all, v_all) = if self.num_kv_heads != self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            let k = k_all
                .unsqueeze(2)?
                .expand((
                    batch_size,
                    self.num_kv_heads,
                    n_rep,
                    total_len,
                    self.head_dim,
                ))?
                .reshape((batch_size, self.num_heads, total_len, self.head_dim))?
                .contiguous()?;
            let v = v_all
                .unsqueeze(2)?
                .expand((
                    batch_size,
                    self.num_kv_heads,
                    n_rep,
                    total_len,
                    self.head_dim,
                ))?
                .reshape((batch_size, self.num_heads, total_len, self.head_dim))?
                .contiguous()?;
            (k, v)
        } else {
            (k_all, v_all)
        };

        let scale = (self.head_dim as f64).sqrt();
        let k_t = k_all.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?.affine(1.0 / scale, 0.0)?; // [B, H, S, T]

        // Rectangular causal (+ window) mask over global positions.
        // For single-token decode without a window nothing is masked.
        let needs_mask =
            (self.causal && seq_len > 1) || (self.window_size.is_some() && total_len > 1);
        let attn_weights = if needs_mask {
            let mut mask = vec![0u8; seq_len * total_len];
            for i in 0..seq_len {
                let gi = start_pos + i;
                for j in 0..total_len {
                    let causal_block = self.causal && j > gi;
                    let window_block = self
                        .window_size
                        .map(|w| gi.saturating_sub(j) > w || j.saturating_sub(gi) > w)
                        .unwrap_or(false);
                    if causal_block || window_block {
                        mask[i * total_len + j] = 1;
                    }
                }
            }
            let mask = Tensor::from_vec(mask, (seq_len, total_len), x.device())?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            masked_fill(&attn_weights, &mask, f32::NEG_INFINITY)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let output = attn_weights.matmul(&v_all)?;
        let output = output.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&output)
    }
}

/// KV cache for one attention layer ([B, n_kv, T, head_dim], pre-GQA-repeat)
#[derive(Default)]
pub struct AttnCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}
