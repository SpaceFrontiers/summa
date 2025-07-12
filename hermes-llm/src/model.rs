use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Dropout, Embedding, Linear, VarBuilder, embedding, linear, linear_no_bias};

use crate::mal::{BlockDef, ModelDef, NormPosition, NormType, SsmDef};

/// Numerically stable softplus: max(x, 0) + ln(1 + exp(-|x|))
fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let neg_abs = x.abs()?.neg()?;
    relu + (neg_abs.exp()? + 1.0)?.log()?
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = on_false.shape();
    let mask = mask.broadcast_as(shape.dims())?;
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x = x.to_dtype(dtype)?;
        x.broadcast_mul(&self.weight)
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    pub fn new(size: usize, eps: f64, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        let bias = if use_bias {
            Some(vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.0))?)
        } else {
            None
        };
        Ok(Self { weight, bias, eps })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_sub(&mean)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x = x.to_dtype(dtype)?;
        let x = x.broadcast_mul(&self.weight)?;
        match &self.bias {
            Some(bias) => x.broadcast_add(bias),
            None => Ok(x),
        }
    }
}

/// Unified normalization layer that can be either RMSNorm or LayerNorm
#[derive(Debug, Clone)]
pub enum Norm {
    RmsNorm(RMSNorm),
    LayerNorm(LayerNorm),
}

impl Norm {
    pub fn new(
        norm_type: NormType,
        size: usize,
        eps: f64,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        match norm_type {
            NormType::RmsNorm | NormType::None => Ok(Self::RmsNorm(RMSNorm::new(size, eps, vb)?)),
            NormType::LayerNorm => Ok(Self::LayerNorm(LayerNorm::new(size, eps, use_bias, vb)?)),
        }
    }
}

impl Module for Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::RmsNorm(n) => n.forward(x),
            Self::LayerNorm(n) => n.forward(x),
        }
    }
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        let q_rot = self.rotate_half(q, &cos, &sin)?;
        let k_rot = self.rotate_half(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, h, seq, d) = x.dims4()?;
        let x1 = x.narrow(3, 0, d / 2)?;
        let x2 = x.narrow(3, d / 2, d / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

        let cos = cos
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, h, seq, d / 2))?;
        let sin = sin
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, h, seq, d / 2))?;
        let cos = Tensor::cat(&[&cos, &cos], 3)?;
        let sin = Tensor::cat(&[&sin, &sin], 3)?;

        let x_cos = x.broadcast_mul(&cos)?;
        let rot_sin = rotated.broadcast_mul(&sin)?;
        let result = x_cos.add(&rot_sin)?;
        Ok(result)
    }
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

/// Recurrent state for one Mamba layer: the conv tail (last k-1 inputs) and
/// the SSM hidden state — the compact persistent memory, O(1) in sequence
/// length.
pub struct MambaState {
    /// [B, d_inner, conv_kernel - 1]
    pub conv: Tensor,
    /// [B, d_inner, state_dim]
    pub h: Tensor,
}

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

/// Selective state-space (Mamba-1) mixer.
///
/// Sequential scan implementation — correct and cache-free, sized for
/// inference; training uses the fused kernels on the hermes-train side.
/// Tensor names follow the mamba-ssm convention under `layers.{i}.ssm.*`.
pub struct MambaMixer {
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    d_inner: usize,
    state_dim: usize,
    dt_rank: usize,
    conv_kernel: usize,
}

impl MambaMixer {
    pub fn new(config: &ModelDef, ssm: &SsmDef, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let d_inner = ssm.expand * hidden;
        let state_dim = ssm.state_dim;
        let dt_rank = config.dt_rank(ssm);
        let conv_kernel = ssm.conv_kernel;

        let in_proj = linear_no_bias(hidden, 2 * d_inner, vb.pp("in_proj"))?;
        let conv_vb = vb.pp("conv1d");
        let conv1d_weight = conv_vb.get_with_hints(
            (d_inner, 1, conv_kernel),
            "weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: (1.0 / conv_kernel as f64).sqrt(),
            },
        )?;
        let conv1d_bias = conv_vb.get_with_hints(d_inner, "bias", candle_nn::Init::Const(0.0))?;
        let x_proj = linear_no_bias(d_inner, dt_rank + 2 * state_dim, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        // A_log/D are always overwritten by checkpoint load (inference-only
        // path); the real S4D init lives on the hermes-train side.
        let a_log =
            vb.get_with_hints((d_inner, state_dim), "A_log", candle_nn::Init::Const(0.0))?;
        let d = vb.get_with_hints(d_inner, "D", candle_nn::Init::Const(1.0))?;
        let out_proj = linear_no_bias(d_inner, hidden, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            d_inner,
            state_dim,
            dt_rank,
            conv_kernel,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_state(x, None)
    }

    /// Forward with optional recurrent state. With state, the conv window and
    /// SSM hidden state continue from previous calls and are updated in place
    /// — O(1) per token, independent of history length.
    pub fn forward_with_state(
        &self,
        x: &Tensor,
        mut state: Option<&mut MambaState>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Input projection → x, z gates
        let xz = self.in_proj.forward(x)?; // [B, L, 2*di]
        let xs = xz.narrow(2, 0, self.d_inner)?;
        let z = xz.narrow(2, self.d_inner, self.d_inner)?;

        // Depthwise causal conv over time: [B, di, L]. Left context is the
        // stored conv tail when stateful, zeros otherwise.
        let xs_t = xs.transpose(1, 2)?.contiguous()?; // [B, di, L]
        let padded = match &state {
            Some(s) => Tensor::cat(&[&s.conv, &xs_t], 2)?.contiguous()?,
            None => xs_t.pad_with_zeros(2, self.conv_kernel - 1, 0)?,
        };
        if let Some(s) = state.as_deref_mut() {
            // New tail: last conv_kernel-1 raw inputs
            let total = padded.dim(2)?;
            s.conv = padded
                .narrow(2, total - (self.conv_kernel - 1), self.conv_kernel - 1)?
                .contiguous()?;
        }
        let conv = padded.conv1d(&self.conv1d_weight, 0, 1, 1, self.d_inner)?;
        let conv = conv.narrow(2, 0, seq_len)?;
        let conv = conv.broadcast_add(&self.conv1d_bias.reshape((1, self.d_inner, 1))?)?;
        let xs = candle_nn::ops::silu(&conv.transpose(1, 2)?.contiguous()?)?; // [B, L, di]

        // Input-dependent Δ, B, C. The narrows are made contiguous: Metal's
        // matmul rejects strided views (dt_proj on the Δ slice).
        let x_dbl = self.x_proj.forward(&xs)?; // [B, L, dt_rank + 2N]
        let delta = x_dbl.narrow(2, 0, self.dt_rank)?.contiguous()?;
        let b_mat = x_dbl
            .narrow(2, self.dt_rank, self.state_dim)?
            .contiguous()?; // [B, L, N]
        let c_mat = x_dbl
            .narrow(2, self.dt_rank + self.state_dim, self.state_dim)?
            .contiguous()?;
        let delta = softplus(&self.dt_proj.forward(&delta)?)?; // [B, L, di]

        let a = self.a_log.exp()?.neg()?; // [di, N]

        // Sequential selective scan over time, resuming from stored h
        let mut h = match &state {
            Some(s) => s.h.clone(),
            None => Tensor::zeros((batch, self.d_inner, self.state_dim), x.dtype(), x.device())?,
        };
        let mut ys = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let dt = delta.narrow(1, t, 1)?.squeeze(1)?; // [B, di]
            let xt = xs.narrow(1, t, 1)?.squeeze(1)?; // [B, di]
            let bt = b_mat.narrow(1, t, 1)?.squeeze(1)?; // [B, N]
            let ct = c_mat.narrow(1, t, 1)?.squeeze(1)?; // [B, N]

            let dt_e = dt.unsqueeze(2)?; // [B, di, 1]
            let da = dt_e.broadcast_mul(&a.unsqueeze(0)?)?.exp()?; // [B, di, N]
            let dbx = dt_e
                .broadcast_mul(&bt.unsqueeze(1)?)? // [B, di, N]
                .broadcast_mul(&xt.unsqueeze(2)?)?;
            h = (h.mul(&da)? + dbx)?;

            let yt = h.broadcast_mul(&ct.unsqueeze(1)?)?.sum(2)?; // [B, di]
            let yt = (yt + xt.broadcast_mul(&self.d.unsqueeze(0)?)?)?;
            ys.push(yt.unsqueeze(1)?);
        }
        let y = Tensor::cat(&ys, 1)?; // [B, L, di]

        // Persist the final SSM state
        if let Some(s) = state {
            s.h = h;
        }

        // Output gate + projection
        let y = y.mul(&candle_nn::ops::silu(&z)?)?;
        self.out_proj.forward(&y)
    }
}

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
    layers: Vec<LayerState>,
    pos: usize,
}

impl InferenceState {
    pub fn pos(&self) -> usize {
        self.pos
    }
}

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
        let (rope_head_dim, rope_theta) = match attn_blocks.first() {
            Some(first) => {
                let hd = first.head_dim(config.hidden_size);
                let theta = first.rope_theta();
                for b in &attn_blocks {
                    if b.head_dim(config.hidden_size) != hd || b.rope_theta() != theta {
                        candle_core::bail!(
                            "all attention blocks must share head_dim and rope theta \
                             (shared RoPE table)"
                        );
                    }
                }
                (hd, theta)
            }
            // Pure-SSM model: RoPE is unused, build a minimal table
            None => (2, 10000.0),
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
        let rope =
            RotaryEmbedding::new(rope_head_dim, config.max_seq_len, rope_theta, vb.device())?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mal::get_builtin_model;
    use candle_nn::{VarBuilder, VarMap};

    fn forward_random(config: &ModelDef) -> Result<Tensor> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(config, vb)?;
        let ids = Tensor::zeros((2, 12), DType::U32, &device)?;
        model.forward(&ids, 0, false)
    }

    #[test]
    fn test_hybrid_forward() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        let logits = forward_random(&config).unwrap();
        assert_eq!(logits.dims3().unwrap(), (2, 12, 128));
        let sum = logits
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(sum.is_finite());
    }

    #[test]
    fn test_hybrid_tensor_names() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let _model = Transformer::new(&config, vb).unwrap();

        let names: std::collections::HashSet<String> =
            var_map.data().lock().unwrap().keys().cloned().collect();

        // Pattern [ssm, ssm, attn] over 6 layers: layers 0,1,3,4 are SSM
        for i in [0usize, 1, 3, 4] {
            for t in [
                "in_proj.weight",
                "conv1d.weight",
                "conv1d.bias",
                "x_proj.weight",
                "dt_proj.weight",
                "dt_proj.bias",
                "A_log",
                "D",
                "out_proj.weight",
            ] {
                assert!(
                    names.contains(&format!("layers.{i}.ssm.{t}")),
                    "missing layers.{i}.ssm.{t}"
                );
            }
            assert!(!names.contains(&format!("layers.{i}.attention.q_proj.weight")));
        }
        // Layers 2, 5 are attention
        for i in [2usize, 5] {
            assert!(names.contains(&format!("layers.{i}.attention.q_proj.weight")));
            assert!(!names.contains(&format!("layers.{i}.ssm.in_proj.weight")));
        }
    }

    #[test]
    fn test_qk_norm_tensors_and_stateful_parity() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        config.block.attention.qk_norm = true;
        if let Some(pattern) = config.pattern.as_mut() {
            for b in pattern.iter_mut() {
                b.attention.qk_norm = true;
            }
        }

        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(&config, vb).unwrap();

        let names: std::collections::HashSet<String> =
            var_map.data().lock().unwrap().keys().cloned().collect();
        // Attention layers (2, 5 in the [ssm,ssm,attn] pattern) get q/k norms
        assert!(names.contains("layers.2.attention.q_norm.weight"));
        assert!(names.contains("layers.5.attention.k_norm.weight"));
        assert!(!names.contains("layers.0.attention.q_norm.weight")); // ssm layer

        // Stateful decode still matches stateless with qk-norm active
        let ids: Vec<u32> = (0..10).map(|i| (i * 5 + 1) % 128).collect();
        let full = Tensor::new(ids.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full_logits = model.forward(&full, 0, false).unwrap();

        let mut state = model.make_state(1, &device).unwrap();
        let prefill = Tensor::new(&ids[..9], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        model.forward_with_state(&prefill, &mut state).unwrap();
        let step = Tensor::new(&ids[9..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let step_logits = model.forward_with_state(&step, &mut state).unwrap();

        let a: Vec<f32> = step_logits.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = full_logits
            .narrow(1, 9, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "qk-norm stateful diverges: {max_diff}");
    }

    #[test]
    fn test_tied_embeddings() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        config.embeddings.tie_weights = true;

        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(&config, vb).unwrap();

        // No lm_head tensor when tied
        let names: Vec<String> = var_map.data().lock().unwrap().keys().cloned().collect();
        assert!(!names.iter().any(|n| n.starts_with("lm_head")));

        let ids = Tensor::zeros((1, 6), DType::U32, &device).unwrap();
        let logits = model.forward(&ids, 0, false).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 6, 128));
    }

    #[test]
    fn test_stateful_matches_stateless() {
        // Prefill + single-token decode must reproduce full-recompute logits
        // for both mixer types (hybrid covers attention KV cache and Mamba
        // recurrent state).
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(&config, vb).unwrap();

        let ids: Vec<u32> = (0..12).map(|i| (i * 7 + 3) % 128).collect();
        let full_input = Tensor::new(ids.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full_logits = model.forward(&full_input, 0, false).unwrap(); // [1, 12, V]

        // Prefill 8 tokens, then decode 4 one at a time
        let mut state = model.make_state(1, &device).unwrap();
        let prefill = Tensor::new(&ids[..8], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mut logits = model.forward_with_state(&prefill, &mut state).unwrap();
        // Last prefill row == stateless row 7
        let check = |step_logits: &Tensor, pos: usize| {
            let a: Vec<f32> = step_logits
                .narrow(1, step_logits.dim(1).unwrap() - 1, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b: Vec<f32> = full_logits
                .narrow(1, pos, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let max_diff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "logits diverge at pos {pos}: {max_diff}");
        };
        check(&logits, 7);

        for (step, &tok) in ids[8..].iter().enumerate() {
            let input = Tensor::new(&[tok], &device).unwrap().unsqueeze(0).unwrap();
            logits = model.forward_with_state(&input, &mut state).unwrap();
            check(&logits, 8 + step);
        }
        assert_eq!(state.pos(), 12);
    }

    #[test]
    fn test_legacy_transformer_forward_unchanged() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.vocab_size = 64;
        let logits = forward_random(&config).unwrap();
        assert_eq!(logits.dims3().unwrap(), (2, 12, 64));
    }
}
