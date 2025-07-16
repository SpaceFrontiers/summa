//! Full language model assembled from the MAL definition.

use anyhow::{Result, bail};
use burn::module::{Initializer, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn::tensor::Int;
use burn_nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn_nn::{RotaryEncoding, RotaryEncodingConfig};

use crate::mal::{BlockDef, ModelDef, NormConfig, PositionEncoding};

use super::linear_cross_entropy::linear_cross_entropy;
use super::matmul::{matmul_2, matmul_input, prepare_linear_for_inference, stream_cast};
use super::{InferenceState, Norm, TransformerBlock};

const EMBEDDING_STD: f64 = 0.02;
const LOSS_CHUNKS: usize = 4;

fn validate_norm(name: &str, norm: &NormConfig) -> Result<()> {
    if norm.eps != 0.0 && (!norm.eps.is_finite() || norm.eps <= 0.0) {
        bail!(
            "{name} epsilon must be finite and positive, got {}",
            norm.eps
        );
    }
    Ok(())
}

fn validate_config(config: &ModelDef) -> Result<()> {
    if config.num_layers == 0 {
        bail!("model must contain at least one layer");
    }
    if config.hidden_size == 0 || config.vocab_size == 0 || config.max_seq_len == 0 {
        bail!("vocab_size, hidden_size, and max_seq_len must all be positive");
    }
    if !(0.0..1.0).contains(&config.embeddings.dropout) {
        bail!(
            "embedding dropout must be in [0, 1), got {}",
            config.embeddings.dropout
        );
    }
    if config
        .embeddings
        .scale
        .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
    {
        bail!("embedding scale must be finite and positive");
    }
    if let Some(norm) = &config.output.norm {
        validate_norm("output norm", norm)?;
    }

    for i in 0..config.num_layers {
        let block = config.block_for_layer(i);
        for (name, dropout) in [
            ("block", block.dropout),
            ("attention", block.attention.dropout),
            ("ffn", block.ffn.dropout),
        ] {
            if !(0.0..1.0).contains(&dropout) {
                bail!("layer {i} {name} dropout must be in [0, 1), got {dropout}");
            }
        }
        validate_norm(&format!("layer {i} norm"), &block.norm)?;
        let intermediate = match block.ffn.hidden_dim {
            Some(size) => size,
            None => config
                .hidden_size
                .checked_mul(4)
                .ok_or_else(|| anyhow::anyhow!("layer {i} default FFN size overflows usize"))?,
        };
        if intermediate == 0 {
            bail!("layer {i} FFN hidden_dim must be positive");
        }

        if let Some(ssm) = &block.ssm {
            for (name, size) in [
                ("expand", ssm.expand),
                ("state_dim", ssm.state_dim),
                ("conv_kernel", ssm.conv_kernel),
                ("dt_rank", config.dt_rank(ssm)),
            ] {
                if size == 0 {
                    bail!("layer {i} Mamba {name} must be positive");
                }
            }
            if ssm.expand.checked_mul(config.hidden_size).is_none() {
                bail!("layer {i} Mamba expand * hidden_size overflows usize");
            }
            continue;
        }

        let heads = block.attention.num_heads.unwrap_or(12);
        if heads == 0 {
            bail!("layer {i} attention num_heads must be positive");
        }
        let kv_heads = block.attention.num_kv_heads.unwrap_or(heads);
        let head_dim = block
            .attention
            .head_dim
            .unwrap_or(config.hidden_size / heads);
        if kv_heads == 0 || head_dim == 0 {
            bail!("layer {i} attention num_kv_heads and head_dim must be positive");
        }
        if !heads.is_multiple_of(kv_heads) {
            bail!("layer {i} num_heads ({heads}) must be divisible by num_kv_heads ({kv_heads})");
        }
        if heads.checked_mul(head_dim) != Some(config.hidden_size) {
            bail!(
                "layer {i} num_heads ({heads}) * head_dim ({head_dim}) must equal hidden_size ({})",
                config.hidden_size
            );
        }
        if block.attention.window_size == Some(0) {
            bail!("layer {i} attention window_size must be positive");
        }
        match &block.attention.position_encoding {
            PositionEncoding::Rope { theta, scaling } => {
                if !head_dim.is_multiple_of(2) {
                    bail!("layer {i} RoPE head_dim must be even, got {head_dim}");
                }
                if !theta.is_finite()
                    || *theta <= 0.0
                    || scaling.is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
                {
                    bail!("layer {i} RoPE theta and scaling must be finite and positive");
                }
            }
            PositionEncoding::None => {}
            other => {
                bail!("position_encoding {other:?} is not implemented; use rope or none")
            }
        }
    }
    Ok(())
}

fn pad_embedding(mut embedding: Embedding, stored_vocab_size: usize) -> Embedding {
    let [vocab_size, hidden_size] = embedding.weight.shape().dims();
    if vocab_size < stored_vocab_size {
        embedding.weight = embedding.weight.map(|weight| {
            let device = weight.device();
            Tensor::cat(
                vec![
                    weight,
                    Tensor::zeros([stored_vocab_size - vocab_size, hidden_size], &device),
                ],
                0,
            )
        });
    }
    embedding
}

fn pad_output_linear(mut output: Linear, stored_vocab_size: usize) -> Linear {
    let [hidden_size, vocab_size] = output.weight.shape().dims();
    if vocab_size < stored_vocab_size {
        output.weight = output.weight.map(|weight| {
            let device = weight.device();
            Tensor::cat(
                vec![
                    weight,
                    Tensor::zeros([hidden_size, stored_vocab_size - vocab_size], &device),
                ],
                1,
            )
        });
        output.bias = output.bias.map(|bias| {
            bias.map(|bias| {
                let device = bias.device();
                Tensor::cat(
                    vec![
                        bias,
                        Tensor::zeros([stored_vocab_size - vocab_size], &device),
                    ],
                    0,
                )
            })
        });
    }
    output
}

#[derive(Module, Debug)]
pub struct Transformer {
    embedding: Embedding,
    embedding_dropout: Dropout,
    layers: Vec<TransformerBlock>,
    final_norm: Norm,
    /// Absent when embedding weights are tied.
    lm_head: Option<Linear>,
    /// Output bias when the embedding matrix is reused as the output matrix.
    tied_output_bias: Option<Param<Tensor<1>>>,
    /// Mixed-precision view of tied output weights, prepared once for decoding.
    #[module(skip)]
    inference_output_weight: Option<Tensor<2>>,
    rope: RotaryEncoding,
    #[module(skip)]
    embedding_scale: Option<f64>,
    #[module(skip)]
    config: ModelDef,
}

impl Transformer {
    pub fn new(config: &ModelDef, device: &Device) -> Result<Self> {
        validate_config(config)?;

        let attn_blocks: Vec<&BlockDef> = (0..config.num_layers)
            .map(|i| config.block_for_layer(i))
            .filter(|block| !block.is_ssm())
            .collect();

        let rope_blocks: Vec<_> = attn_blocks
            .iter()
            .copied()
            .filter(|block| {
                matches!(
                    &block.attention.position_encoding,
                    PositionEncoding::Rope { .. }
                )
            })
            .collect();
        let (rope_head_dim, rope_theta, rope_scaling) = match rope_blocks.first() {
            Some(first) => {
                let head_dim = first.head_dim(config.hidden_size);
                let theta = first.rope_theta();
                let scaling = first.rope_scaling();
                for block in &rope_blocks {
                    if block.head_dim(config.hidden_size) != head_dim
                        || block.rope_theta() != theta
                        || block.rope_scaling() != scaling
                    {
                        bail!(
                            "all attention blocks must share head_dim, RoPE theta, and RoPE scaling"
                        );
                    }
                }
                (head_dim, theta, scaling)
            }
            None => (2, 10_000.0, None),
        };

        // Burn's Embedding default is N(0, 1), which makes tied-output logits
        // unusably large for language models. Use the standard small LLM scale.
        let stored_vocab_size = config.padded_vocab_size();
        let embedding = pad_embedding(
            EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: EMBEDDING_STD,
                })
                .init(device),
            stored_vocab_size,
        );
        let embedding_dropout = DropoutConfig::new(config.embeddings.dropout).init();
        let layers = (0..config.num_layers)
            .map(|i| TransformerBlock::new(config, config.block_for_layer(i), device))
            .collect();
        let norm_block = config.block_for_layer(0);
        let norm_config = config.output.norm.as_ref().unwrap_or(&norm_block.norm);
        let norm_eps = if norm_config.eps > 0.0 {
            norm_config.eps
        } else {
            1e-5
        };
        let final_norm = Norm::new(norm_config.norm_type, config.hidden_size, norm_eps, device);
        let lm_head = (!config.embeddings.tie_weights).then(|| {
            pad_output_linear(
                LinearConfig::new(config.hidden_size, config.vocab_size)
                    .with_bias(config.output.bias)
                    .init(device),
                stored_vocab_size,
            )
        });
        let tied_output_bias = (config.embeddings.tie_weights && config.output.bias)
            .then(|| Initializer::Zeros.init([stored_vocab_size], device));
        let rope_config = RotaryEncodingConfig::new(config.max_seq_len, rope_head_dim)
            .with_theta(rope_theta as f32);
        let rope = match rope_scaling {
            Some(scale) => rope_config
                .init_with_frequency_scaling(|frequencies| frequencies.div_scalar(scale), device),
            _ => rope_config.init(device),
        };

        Ok(Self {
            embedding,
            embedding_dropout,
            layers,
            final_norm,
            lm_head,
            tied_output_bias,
            inference_output_weight: None,
            rope,
            embedding_scale: config.embeddings.scale,
            config: config.clone(),
        })
    }

    fn embed(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        let x = self.embedding.forward(input_ids);
        let x = match self.embedding_scale {
            Some(scale) => x.mul_scalar(scale),
            None => x,
        };
        self.embedding_dropout.forward(x)
    }

    pub fn forward(&self, input_ids: Tensor<2, Int>, start_pos: usize) -> Tensor<3> {
        self.project_logits(self.forward_hidden(input_ids, start_pos))
    }

    fn forward_hidden(&self, input_ids: Tensor<2, Int>, start_pos: usize) -> Tensor<3> {
        let [_, seq_len] = input_ids.dims();
        assert!(
            start_pos + seq_len <= self.config.max_seq_len,
            "input positions {start_pos}..{} exceed max_seq_len {}",
            start_pos + seq_len,
            self.config.max_seq_len
        );
        // The full-sequence path runs the residual stream in the training
        // compute dtype (BF16 under CUDA training-fusion). Incremental decode
        // (`forward_hidden_with_state`) keeps FP32: its scan/conv step kernels
        // are FP32-only.
        let mut x = stream_cast(self.embed(input_ids));
        for layer in &self.layers {
            x = layer.forward(x, &self.rope, start_pos);
        }
        self.final_norm.forward(x)
    }

    /// Next-token cross-entropy for training. The output projection is chunked
    /// so full-vocabulary logits are never retained for every input token.
    pub fn forward_loss(&self, input_ids: Tensor<2, Int>, targets: Tensor<2, Int>) -> Tensor<1> {
        let [batch, seq_len] = targets.dims();
        let tokens = batch * seq_len;
        let hidden = self
            .forward_hidden(input_ids, 0)
            .reshape([tokens, self.config.hidden_size]);
        let (weight, bias) = self.output_parameters();
        linear_cross_entropy(
            hidden,
            weight,
            bias,
            targets.reshape([tokens]),
            self.config.vocab_size,
            tokens.div_ceil(LOSS_CHUNKS),
        )
    }

    fn project_logits(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, hidden] = x.dims();
        let stored_vocab_size = self.config.padded_vocab_size();
        let (weight, bias) = self.output_parameters();
        let logits = matmul_2(x.reshape([batch * seq_len, hidden]), weight.transpose()).reshape([
            batch,
            seq_len,
            stored_vocab_size,
        ]);
        let logits = match bias {
            Some(bias) => logits + bias.reshape([1, 1, stored_vocab_size]),
            None => logits,
        };
        if stored_vocab_size == self.config.vocab_size {
            logits
        } else {
            logits.slice([0..batch, 0..seq_len, 0..self.config.vocab_size])
        }
    }

    fn project_last_logits(&self, x: Tensor<3>) -> Tensor<2> {
        let [batch, seq_len, hidden] = x.dims();
        let x = x
            .slice([0..batch, seq_len - 1..seq_len, 0..hidden])
            .reshape([batch, hidden]);
        let stored_vocab_size = self.config.padded_vocab_size();
        let (weight, bias) = self.output_parameters();
        let logits = matmul_2(x, weight.transpose());
        let logits = match bias {
            Some(bias) => logits + bias.reshape([1, stored_vocab_size]),
            None => logits,
        };
        if stored_vocab_size == self.config.vocab_size {
            logits
        } else {
            logits.slice([0..batch, 0..self.config.vocab_size])
        }
    }

    fn output_parameters(&self) -> (Tensor<2>, Option<Tensor<1>>) {
        match &self.lm_head {
            Some(head) => (
                head.weight.val().transpose(),
                head.bias.as_ref().map(Param::val),
            ),
            None => (
                self.inference_output_weight
                    .clone()
                    .unwrap_or_else(|| self.embedding.weight.val()),
                self.tied_output_bias.as_ref().map(Param::val),
            ),
        }
    }

    /// Prepare immutable mixed-precision weights once for low-latency decode.
    ///
    /// Call this after loading a checkpoint and before creating inference
    /// state. Training never calls it, so optimizer parameters remain F32.
    pub fn prepare_inference(&mut self) {
        assert!(
            !self.embedding.weight.val().device().is_autodiff(),
            "inference preparation requires a non-autodiff model"
        );
        for layer in &mut self.layers {
            layer.prepare_inference();
        }
        if let Some(head) = &mut self.lm_head {
            prepare_linear_for_inference(head);
        } else {
            self.inference_output_weight = Some(matmul_input(self.embedding.weight.val()));
        }
    }

    pub fn config(&self) -> &ModelDef {
        &self.config
    }

    pub fn num_parameters(&self) -> usize {
        self.num_params()
    }

    /// Parameter IDs optimized by Muon during training.
    ///
    /// Every 2D parameter inside a transformer block uses Muon. Embeddings, the
    /// output head, norms, biases, and convolution kernels remain on AdamW.
    pub fn muon_parameter_ids(&self) -> Vec<ParamId> {
        let mut visitor = MatrixParameterVisitor::default();
        for layer in &self.layers {
            layer.visit(&mut visitor);
        }
        visitor.ids
    }

    pub fn make_state(&self, batch: usize, device: &Device) -> InferenceState {
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.make_state(batch, device))
            .collect();
        InferenceState { layers, pos: 0 }
    }

    pub fn forward_with_state(
        &self,
        input_ids: Tensor<2, Int>,
        state: &mut InferenceState,
    ) -> Tensor<3> {
        self.project_logits(self.forward_hidden_with_state(input_ids, state))
    }

    /// Run cached inference and project only the final position to vocabulary logits.
    pub fn forward_next_logits_with_state(
        &self,
        input_ids: Tensor<2, Int>,
        state: &mut InferenceState,
    ) -> Tensor<2> {
        self.project_last_logits(self.forward_hidden_with_state(input_ids, state))
    }

    fn forward_hidden_with_state(
        &self,
        input_ids: Tensor<2, Int>,
        state: &mut InferenceState,
    ) -> Tensor<3> {
        let [batch, seq_len] = input_ids.dims();
        assert!(
            batch > 0 && seq_len > 0,
            "incremental inference requires non-empty input"
        );
        assert!(
            state.pos + seq_len <= self.config.max_seq_len,
            "inference state at position {} + {} tokens exceeds max_seq_len {}",
            state.pos,
            seq_len,
            self.config.max_seq_len
        );
        assert_eq!(
            state.layers.len(),
            self.layers.len(),
            "inference state belongs to a model with a different layer count"
        );

        let mut x = self.embed(input_ids);
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            x = layer.forward_with_state(x, &self.rope, state.pos, layer_state);
        }
        state.pos += seq_len;
        self.final_norm.forward(x)
    }
}

#[derive(Default)]
struct MatrixParameterVisitor {
    ids: Vec<ParamId>,
}

impl ModuleVisitor for MatrixParameterVisitor {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if D == 2 {
            self.ids.push(param.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mal::get_builtin_model;
    use burn::tensor::TensorData;

    /// End-to-end BF16-residual-stream gate: the model must run forward_loss
    /// + backward under lazy fusion, where dtype mismatches between custom-op
    /// gradients and the BF16 stream only surface at runtime (the plain-CUDA
    /// suite never exercises them). Probes attention-only, mamba-only, and
    /// hybrid variants so a failure localizes to a block type.
    #[cfg(all(feature = "training-fusion", target_os = "linux"))]
    #[test]
    fn training_fusion_bf16_stream_loss_and_gradients_are_finite() {
        use burn::tensor::DType;

        struct GradProbe<'a> {
            grads: &'a burn::tensor::Gradients,
            checked: usize,
            bad: Vec<String>,
        }
        impl ModuleVisitor for GradProbe<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.checked += 1;
                let Some(grad) = param.grad(self.grads) else {
                    self.bad.push(format!("{:?} MISSING", param.shape()));
                    return;
                };
                let scalars =
                    |t: Tensor<1>| t.into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
                let sum = scalars(grad.clone().sum());
                let amax = scalars(grad.abs().max());
                if !sum.is_finite() || amax == 0.0 {
                    self.bad
                        .push(format!("{:?} sum={sum} amax={amax}", param.shape()));
                }
            }
        }

        let run = |label: &str, config: &crate::mal::ModelDef, device: &Device| {
            device.seed(17);
            let model = Transformer::new(config, device).unwrap();
            let (batch, seq_len) = (2, 48);
            let ids: Vec<i64> = (0..batch * (seq_len + 1))
                .map(|i| (i * 7 % config.vocab_size) as i64)
                .collect();
            let tokens =
                Tensor::<2, Int>::from_data(TensorData::new(ids, [batch, seq_len + 1]), device);
            let inputs = tokens.clone().slice([0..batch, 0..seq_len]);
            let targets = tokens.slice([0..batch, 1..seq_len + 1]);

            let loss = model.forward_loss(inputs, targets);
            assert_eq!(loss.dtype(), DType::F32, "{label}: loss must stay FP32");
            let value = loss
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap()[0];
            let grads = loss.backward();
            let mut probe = GradProbe {
                grads: &grads,
                checked: 0,
                bad: Vec::new(),
            };
            model.visit(&mut probe);
            println!(
                "{label}: loss={value} params={} bad={}",
                probe.checked,
                probe.bad.len()
            );
            for line in &probe.bad {
                println!("{label}: BAD {line}");
            }
            assert!(
                value.is_finite(),
                "{label}: loss must be finite, got {value}"
            );
            assert!(
                probe.bad.is_empty(),
                "{label}: every parameter gradient must be finite and non-zero"
            );
        };

        let hybrid = get_builtin_model("hybrid_tiny").unwrap();
        let device = Device::cuda(0).autodiff();

        let mut attention_only = hybrid.clone();
        attention_only.pattern = None;
        run("attention-only", &attention_only, &device);

        let mut mamba_only = hybrid.clone();
        mamba_only.pattern = hybrid
            .pattern
            .as_ref()
            .map(|pattern| vec![pattern[0].clone()]);
        run("mamba-only", &mamba_only, &device);

        run("hybrid", &hybrid, &device);
    }

    #[test]
    fn unaligned_vocabulary_uses_zero_padded_parameters() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.vocab_size = 65;
        config.hidden_size = 8;
        config.num_layers = 1;
        config.block.attention.num_heads = Some(2);
        config.block.attention.num_kv_heads = Some(1);
        config.block.attention.head_dim = Some(4);
        config.block.ffn.hidden_dim = Some(16);
        config.output.bias = true;
        let device = Device::ndarray();

        for tied in [false, true] {
            config.embeddings.tie_weights = tied;
            device.seed(31);
            let model = Transformer::new(&config, &device).unwrap();
            assert_eq!(model.config.vocab_size, 65);
            assert_eq!(model.embedding.weight.shape().dims(), [128, 8]);
            let input =
                Tensor::<2, Int>::from_data(TensorData::new(vec![1_i64, 2], [1, 2]), &device);
            assert_eq!(model.forward(input, 0).dims(), [1, 2, 65]);
            assert!(
                model
                    .embedding
                    .weight
                    .val()
                    .slice([65..128, 0..8])
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .all(|value| value == 0.0)
            );

            match (&model.lm_head, &model.tied_output_bias) {
                (Some(head), None) if !tied => {
                    assert_eq!(head.weight.shape().dims(), [8, 128]);
                    assert_eq!(head.bias.as_ref().unwrap().shape().dims(), [128]);
                }
                (None, Some(bias)) if tied => assert_eq!(bias.shape().dims(), [128]),
                _ => panic!("output parameters do not match weight tying"),
            }
        }
    }
}
