use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Dropout, Embedding, Linear, VarBuilder, embedding, linear, linear_no_bias};

use crate::config::Config;

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
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let (q_proj, k_proj, v_proj, o_proj) = if config.use_bias {
            (
                linear(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?,
                linear(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?,
                linear(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?,
                linear(config.hidden_size, config.hidden_size, vb.pp("o_proj"))?,
            )
        } else {
            (
                linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?,
                linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?,
                linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?,
                linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("o_proj"))?,
            )
        };
        let dropout = Dropout::new(config.dropout as f32);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_heads,
            head_dim,
            dropout,
        })
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
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = rope.apply(&q, &k, start_pos)?;

        // Use Flash Attention if available (CUDA only), otherwise standard attention
        #[cfg(feature = "flash-attn")]
        let attn_output = {
            // Flash attention expects (batch, seq, heads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1.0 / (self.head_dim as f32).sqrt();
            let attn = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, true)?;
            attn.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?
        };

        #[cfg(not(feature = "flash-attn"))]
        let attn_output = {
            let scale = (self.head_dim as f64).sqrt();
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn_weights = q.matmul(&k_t)?.affine(1.0 / scale, 0.0)?;

            let attn_weights = match mask {
                Some(m) => masked_fill(&attn_weights, m, f32::NEG_INFINITY)?,
                None => attn_weights,
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
}

pub struct FeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
    use_swiglu: bool,
}

impl FeedForward {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let use_swiglu = !config.use_bias;
        let (gate_proj, up_proj, down_proj) = if config.use_bias {
            (
                linear(
                    config.hidden_size,
                    config.intermediate_size,
                    vb.pp("gate_proj"),
                )?,
                linear(
                    config.hidden_size,
                    config.intermediate_size,
                    vb.pp("up_proj"),
                )?,
                linear(
                    config.intermediate_size,
                    config.hidden_size,
                    vb.pp("down_proj"),
                )?,
            )
        } else {
            (
                linear_no_bias(
                    config.hidden_size,
                    config.intermediate_size,
                    vb.pp("gate_proj"),
                )?,
                linear_no_bias(
                    config.hidden_size,
                    config.intermediate_size,
                    vb.pp("up_proj"),
                )?,
                linear_no_bias(
                    config.intermediate_size,
                    config.hidden_size,
                    vb.pp("down_proj"),
                )?,
            )
        };
        let dropout = Dropout::new(config.dropout as f32);
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            dropout,
            use_swiglu,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        let hidden = if self.use_swiglu {
            let gate = candle_nn::ops::silu(&gate)?;
            (gate * up)?
        } else {
            (gate + up)?.gelu_erf()?
        };

        let hidden = self.dropout.forward(&hidden, train)?;
        self.down_proj.forward(&hidden)
    }
}

pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(config, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(config, vb.pp("feed_forward"))?;
        let input_layernorm = RMSNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = RMSNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            attention,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &RotaryEmbedding,
        start_pos: usize,
        train: bool,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x, mask, rope, start_pos, train)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.feed_forward.forward(&x, train)?;
        residual + x
    }
}

pub struct GPT {
    embedding: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    config: Config,
}

impl GPT {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.hidden_size, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerBlock::new(
                config,
                vb.pp(format!("layers.{}", i)),
            )?);
        }
        let norm = RMSNorm::new(config.hidden_size, config.layer_norm_eps, vb.pp("norm"))?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            vb.device(),
        )?;
        Ok(Self {
            embedding,
            layers,
            norm,
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

        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn num_parameters(&self) -> usize {
        let c = &self.config;
        let embed_params = c.vocab_size * c.hidden_size;
        let attn_params = 4 * c.hidden_size * c.hidden_size;
        let ff_params = 3 * c.hidden_size * c.intermediate_size;
        let layer_params = attn_params + ff_params + 2 * c.hidden_size;
        let head_params = c.hidden_size * c.vocab_size;
        let norm_params = c.hidden_size;
        embed_params + c.num_layers * layer_params + norm_params + head_params
    }
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    let logits = logits.reshape((batch_size * seq_len, vocab_size))?;
    let targets = targets.reshape((batch_size * seq_len,))?;
    candle_nn::loss::cross_entropy(&logits, &targets)
}
