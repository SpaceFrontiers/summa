//! Grouped-query attention with optional QK-Norm, causal and sliding-window
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

use super::fused_attention::{fused_attention, repeat_kv};
use super::matmul::{linear, linear_low_precision, prepare_linear_for_inference};

/// Preallocated KV cache for one attention layer, before GQA head expansion.
#[derive(Debug, Clone)]
pub struct AttnCache {
    k: Option<Tensor<4>>,
    v: Option<Tensor<4>>,
    len: usize,
}

impl AttnCache {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    /// Per-head RMSNorm over head_dim, applied to Q/K before RoPE (qk_norm).
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
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
    #[module(skip)]
    max_seq_len: usize,
    attention_dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(config: &ModelDef, block: &BlockDef, device: &Device) -> Self {
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
            qkv_proj: lin(hidden, hidden + 2 * kv_dim),
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
            max_seq_len: config.max_seq_len,
            attention_dropout: DropoutConfig::new(block.attention.dropout).init(),
        }
    }

    pub fn make_cache(&self, batch: usize, device: &Device) -> AttnCache {
        let shape = [batch, self.num_kv_heads, self.max_seq_len, self.head_dim];
        AttnCache {
            k: Some(Tensor::zeros(shape, device)),
            v: Some(Tensor::zeros(shape, device)),
            len: 0,
        }
    }

    pub(crate) fn prepare_inference(&mut self) {
        prepare_linear_for_inference(&mut self.qkv_proj);
        prepare_linear_for_inference(&mut self.o_proj);
    }

    /// Project and position per-head Q, K, and V tensors.
    fn project_qkv(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [b, s, _] = x.dims();
        let split =
            |t: Tensor<3>, heads: usize| t.reshape([b, s, heads, self.head_dim]).swap_dims(1, 2);
        let qkv = linear(&self.qkv_proj, x);
        let query_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q = split(
            qkv.clone().slice([0..b, 0..s, 0..query_dim]),
            self.num_heads,
        );
        let k = split(
            qkv.clone()
                .slice([0..b, 0..s, query_dim..query_dim + kv_dim]),
            self.num_kv_heads,
        );
        let v = split(
            qkv.slice([0..b, 0..s, query_dim + kv_dim..query_dim + 2 * kv_dim]),
            self.num_kv_heads,
        );
        let (q, k) = match (&self.q_norm, &self.k_norm) {
            (Some(q_norm), Some(k_norm)) => (q_norm.forward(q), k_norm.forward(k)),
            _ => (q, k),
        };
        let (q, k) = if self.use_rope {
            (rope.apply(q, start_pos), rope.apply(k, start_pos))
        } else {
            (q, k)
        };
        (q, k, v)
    }

    /// Scaled-dot-product attention over K/V ([B, H, T, hd]) for queries and
    /// keys at their respective global offsets. Applies causal + window masking.
    fn sdpa(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        q_start: usize,
        key_start: usize,
    ) -> Tensor<4> {
        let device = q.device();
        let [_, _, seq_q, _] = q.dims();
        let total = k.dims()[2];
        if device.is_autodiff() && self.attention_dropout.prob > 0.0 {
            let k = repeat_kv(k, self.num_heads);
            let v = repeat_kv(v, self.num_heads);
            let mut scores = q
                .matmul(k.transpose())
                .div_scalar((self.head_dim as f32).sqrt());
            if let Some(mask) = self.build_mask(seq_q, total, q_start, key_start, &device) {
                scores = scores.mask_fill(mask, f32::NEG_INFINITY);
            }
            let weights = self.attention_dropout.forward(softmax(scores, 3));
            return weights.matmul(v);
        }

        // Full-sequence attention has a custom fused backward on CUDA. Cached,
        // offset, and sliding-window attention retain Burn's mask-capable path.
        let aligned_full_sequence = q_start == key_start && seq_q == total;
        let fused_causal = self.causal && self.window_size.is_none() && aligned_full_sequence;
        let mask = if fused_causal {
            None
        } else {
            self.build_mask(seq_q, total, q_start, key_start, &device)
        };
        if mask.is_none() && aligned_full_sequence {
            fused_attention(q, k, v, fused_causal)
        } else {
            attention(
                q,
                repeat_kv(k, self.num_heads),
                repeat_kv(v, self.num_heads),
                mask,
                None,
                AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                },
            )
        }
    }

    /// Bool mask [1, 1, S, T] (true = blocked), or None when nothing is masked.
    /// Position `q_start + i` (query row i) attends to position
    /// `key_start + j` (key column j).
    fn build_mask(
        &self,
        seq_q: usize,
        total: usize,
        q_start: usize,
        key_start: usize,
        device: &Device,
    ) -> Option<Tensor<4, Bool>> {
        let last_key = key_start + total.saturating_sub(1);
        let needs = (self.causal && last_key > q_start) || self.window_size.is_some();
        if !needs {
            return None;
        }
        let mut mask = vec![false; seq_q * total];
        for i in 0..seq_q {
            let gi = q_start + i;
            for j in 0..total {
                let gj = key_start + j;
                let causal_block = self.causal && gj > gi;
                let window_block = self
                    .window_size
                    .map(|window| gi.abs_diff(gj) > window)
                    .unwrap_or(false);
                mask[i * total + j] = causal_block || window_block;
            }
        }
        let m = Tensor::<2, Bool>::from_data(TensorData::new(mask, [seq_q, total]), device);
        Some(m.reshape([1, 1, seq_q, total]))
    }

    fn merge_heads(&self, out: Tensor<4>) -> Tensor<3> {
        let [b, _, s, _] = out.dims();
        out.swap_dims(1, 2)
            .reshape([b, s, self.num_heads * self.head_dim])
    }

    /// Full (stateless) attention over the given sequence.
    pub fn forward(&self, x: Tensor<3>, rope: &RotaryEncoding, start_pos: usize) -> Tensor<3> {
        let (q, k, v) = self.project_qkv(x, rope, start_pos);
        let out = self.sdpa(q, k, v, start_pos, start_pos);
        // The output projection joins the training residual stream directly
        // in the matmul compute dtype; decode (`forward_cached`) keeps the
        // FP32 promotion for its FP32 stream.
        linear_low_precision(&self.o_proj, self.merge_heads(out))
    }

    /// Incremental attention over a KV cache. `x` holds S new tokens at global
    /// positions `start_pos..start_pos+S`.
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        rope: &RotaryEncoding,
        start_pos: usize,
        cache: &mut AttnCache,
    ) -> Tensor<3> {
        let (q, k, v) = self.project_qkv(x, rope, start_pos);

        assert_eq!(
            start_pos, cache.len,
            "attention cache position does not match inference state"
        );
        let end = start_pos + k.dims()[2];
        assert!(end <= self.max_seq_len, "attention cache capacity exceeded");
        let ranges = [
            0..k.dims()[0],
            0..self.num_kv_heads,
            start_pos..end,
            0..self.head_dim,
        ];
        let k_cache = cache
            .k
            .take()
            .expect("attention cache is initialized")
            .slice_assign(ranges.clone(), k);
        let v_cache = cache
            .v
            .take()
            .expect("attention cache is initialized")
            .slice_assign(ranges, v);
        let k = k_cache.clone().slice([
            0..k_cache.dims()[0],
            0..self.num_kv_heads,
            0..end,
            0..self.head_dim,
        ]);
        let v = v_cache.clone().slice([
            0..v_cache.dims()[0],
            0..self.num_kv_heads,
            0..end,
            0..self.head_dim,
        ]);
        cache.k = Some(k_cache);
        cache.v = Some(v_cache);
        cache.len = end;

        let out = self.sdpa(q, k, v, start_pos, 0);
        linear(&self.o_proj, self.merge_heads(out))
    }
}
