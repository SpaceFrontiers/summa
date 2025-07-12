//! Grouped-query attention with optional QK-Norm, causal + sliding-window
//! masking, and a KV cache for O(context) incremental decode.

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, TensorData};
use burn_nn::{
    Dropout, DropoutConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
};

use crate::mal::{BlockDef, ModelDef};

/// KV cache for one attention layer: [B, n_kv, T, head_dim], pre-GQA-repeat.
#[derive(Debug, Clone, Default)]
pub struct AttnCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    /// Per-head RMSNorm over head_dim, applied to Q/K before RoPE (qk_norm).
    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    window_size: Option<usize>,
    #[module(skip)]
    causal: bool,
    #[module(skip)]
    use_rope: bool,
    attention_dropout: Dropout,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(config: &ModelDef, block: &BlockDef, device: &Device<B>) -> Self {
        let num_heads = block.num_heads();
        let num_kv_heads = block.num_kv_heads();
        let head_dim = block.head_dim(config.hidden_size);
        let hidden = config.hidden_size;
        let kv_dim = num_kv_heads * head_dim;
        let use_bias = block.attention.bias;

        let lin = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out)
                .with_bias(use_bias)
                .init(device)
        };
        let (q_norm, k_norm) = if block.attention.qk_norm {
            let n = || {
                RmsNormConfig::new(head_dim)
                    .with_epsilon(block.norm_eps())
                    .init(device)
            };
            (Some(n()), Some(n()))
        } else {
            (None, None)
        };

        Self {
            q_proj: lin(hidden, hidden),
            k_proj: lin(hidden, kv_dim),
            v_proj: lin(hidden, kv_dim),
            o_proj: lin(hidden, hidden),
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            window_size: block.attention.window_size,
            causal: block.attention.causal,
            use_rope: matches!(
                &block.attention.position_encoding,
                crate::mal::PositionEncoding::Rope { .. }
            ),
            attention_dropout: DropoutConfig::new(block.attention.dropout).init(),
        }
    }

    fn apply_qk_norm(&self, q: Tensor<B, 4>, k: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match (&self.q_norm, &self.k_norm) {
            (Some(qn), Some(kn)) => (qn.forward(q), kn.forward(k)),
            _ => (q, k),
        }
    }

    /// Project x -> per-head Q,K,V, each [B, H|n_kv, S, head_dim].
    fn qkv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, s, _] = x.dims();
        let split =
            |t: Tensor<B, 3>, heads: usize| t.reshape([b, s, heads, self.head_dim]).swap_dims(1, 2);
        let q = split(self.q_proj.forward(x.clone()), self.num_heads);
        let k = split(self.k_proj.forward(x.clone()), self.num_kv_heads);
        let v = split(self.v_proj.forward(x), self.num_kv_heads);
        (q, k, v)
    }

    /// Repeat KV heads to match query heads (GQA).
    fn repeat_kv(&self, t: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.num_kv_heads == self.num_heads {
            return t;
        }
        let n_rep = self.num_heads / self.num_kv_heads;
        let [b, n_kv, s, hd] = t.dims();
        t.unsqueeze_dim::<5>(2)
            .repeat_dim(2, n_rep)
            .reshape([b, n_kv * n_rep, s, hd])
    }

    /// Scaled-dot-product attention over K/V ([B, H, T, hd]) for queries at
    /// global positions `q_start..q_start+S`. Applies causal + window masking.
    fn sdpa(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        q_start: usize,
    ) -> Tensor<B, 4> {
        let device = q.device();
        let [_, _, seq_q, _] = q.dims();
        let total = k.dims()[2];
        if B::ad_enabled(&device) && self.attention_dropout.prob > 0.0 {
            let mut scores = q
                .matmul(k.transpose())
                .div_scalar((self.head_dim as f32).sqrt());
            if let Some(mask) = self.build_mask(seq_q, total, q_start, &device) {
                scores = scores.mask_fill(mask, f32::NEG_INFINITY);
            }
            let weights = self.attention_dropout.forward(softmax(scores, 3));
            return weights.matmul(v);
        }

        // Burn's CubeCL backend dispatches its fused attention kernel when it
        // can express causality directly. Cached/offset and sliding-window
        // attention use our explicit global-position mask instead.
        let fused_causal =
            self.causal && self.window_size.is_none() && q_start == 0 && seq_q == total;
        let mask = if fused_causal {
            None
        } else {
            self.build_mask(seq_q, total, q_start, &device)
        };
        attention(
            q,
            k,
            v,
            mask,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: fused_causal,
            },
        )
    }

    /// Bool mask [1, 1, S, T] (true = blocked), or None when nothing is masked.
    /// Position `q_start + i` (query row i) attends to key j in `0..T`.
    fn build_mask(
        &self,
        seq_q: usize,
        total: usize,
        q_start: usize,
        device: &Device<B>,
    ) -> Option<Tensor<B, 4, Bool>> {
        let needs = (self.causal && (seq_q > 1 || q_start > 0 || total > seq_q))
            || (self.window_size.is_some() && total > 1);
        if !needs {
            return None;
        }
        let mut mask = vec![false; seq_q * total];
        for i in 0..seq_q {
            let gi = q_start + i;
            for j in 0..total {
                let causal_block = self.causal && j > gi;
                let window_block = self
                    .window_size
                    .map(|w| gi.saturating_sub(j) > w || j.saturating_sub(gi) > w)
                    .unwrap_or(false);
                mask[i * total + j] = causal_block || window_block;
            }
        }
        let m = Tensor::<B, 2, Bool>::from_data(TensorData::new(mask, [seq_q, total]), device);
        Some(m.reshape([1, 1, seq_q, total]))
    }

    fn merge_heads(&self, out: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _, s, _] = out.dims();
        out.swap_dims(1, 2)
            .reshape([b, s, self.num_heads * self.head_dim])
    }

    /// Full (stateless) attention over the given sequence.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let (q, k, v) = self.qkv(x);
        let (q, k) = self.apply_qk_norm(q, k);
        let (q, k) = if self.use_rope {
            (rope.apply(q, start_pos), rope.apply(k, start_pos))
        } else {
            (q, k)
        };
        let out = self.sdpa(q, self.repeat_kv(k), self.repeat_kv(v), start_pos);
        self.o_proj.forward(self.merge_heads(out))
    }

    /// Incremental attention over a KV cache. `x` holds S new tokens at global
    /// positions `start_pos..start_pos+S`.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        start_pos: usize,
        cache: &mut AttnCache<B>,
    ) -> Tensor<B, 3> {
        let (q, k, v) = self.qkv(x);
        let (q, k) = self.apply_qk_norm(q, k);
        let (q, k) = if self.use_rope {
            (rope.apply(q, start_pos), rope.apply(k, start_pos))
        } else {
            (q, k)
        };

        // Append to cache (pre-GQA-repeat).
        let k = match cache.k.take() {
            Some(ck) => Tensor::cat(vec![ck, k], 2),
            None => k,
        };
        let v = match cache.v.take() {
            Some(cv) => Tensor::cat(vec![cv, v], 2),
            None => v,
        };
        cache.k = Some(k.clone());
        cache.v = Some(v.clone());

        let out = self.sdpa(q, self.repeat_kv(k), self.repeat_kv(v), start_pos);
        self.o_proj.forward(self.merge_heads(out))
    }
}
