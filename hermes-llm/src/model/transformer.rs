//! Full language model assembled from the MAL definition.

use anyhow::{Result, bail};
use burn::module::{Initializer, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn::tensor::Int;
use burn_nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn_nn::{RotaryEncoding, RotaryEncodingConfig};

use crate::mal::{BlockDef, ModelDef, PositionEncoding};

use super::linear_cross_entropy::linear_cross_entropy;
use super::matmul::{matmul_2, matmul_input, prepare_linear_for_inference};
use super::{InferenceState, Norm, TransformerBlock};

const EMBEDDING_STD: f64 = 0.02;
const LOSS_CHUNKS: usize = 4;

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
        if config.num_layers == 0 {
            bail!("model must contain at least one layer");
        }
        if config.hidden_size == 0 || config.vocab_size == 0 || config.max_seq_len == 0 {
            bail!("vocab_size, hidden_size, and max_seq_len must all be positive");
        }

        let attn_blocks: Vec<&BlockDef> = (0..config.num_layers)
            .map(|i| config.block_for_layer(i))
            .filter(|block| !block.is_ssm())
            .collect();
        for block in &attn_blocks {
            let heads = block.num_heads();
            let kv_heads = block.num_kv_heads();
            let head_dim = block.head_dim(config.hidden_size);
            if heads == 0 || kv_heads == 0 || head_dim == 0 {
                bail!("attention head counts and head_dim must be positive");
            }
            if heads % kv_heads != 0 {
                bail!("num_heads ({heads}) must be divisible by num_kv_heads ({kv_heads})");
            }
            if heads * head_dim != config.hidden_size {
                bail!(
                    "num_heads ({heads}) * head_dim ({head_dim}) must equal hidden_size ({})",
                    config.hidden_size
                );
            }
            match &block.attention.position_encoding {
                PositionEncoding::Rope { theta, scaling } => {
                    if head_dim % 2 != 0 {
                        bail!("RoPE head_dim must be even, got {head_dim}");
                    }
                    if *theta <= 0.0 || scaling.is_some_and(|scale| scale <= 0.0) {
                        bail!("RoPE theta and scaling must be positive");
                    }
                }
                PositionEncoding::None => {}
                other => bail!("position_encoding {other:?} is not implemented; use rope or none"),
            }
        }
        for i in 0..config.num_layers {
            let block = config.block_for_layer(i);
            for (name, dropout) in [
                ("block", block.dropout),
                ("attention", block.attention.dropout),
                ("ffn", block.ffn.dropout),
            ] {
                if !(0.0..1.0).contains(&dropout) {
                    bail!("{name} dropout must be in [0, 1), got {dropout}");
                }
            }
            if block.intermediate_size(config.hidden_size) == 0 {
                bail!("FFN hidden_dim must be positive");
            }
        }
        if !(0.0..1.0).contains(&config.embeddings.dropout) {
            bail!(
                "embedding dropout must be in [0, 1), got {}",
                config.embeddings.dropout
            );
        }

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
        let embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: EMBEDDING_STD,
            })
            .init(device);
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
            LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_bias(config.output.bias)
                .init(device)
        });
        let tied_output_bias = (config.embeddings.tie_weights && config.output.bias)
            .then(|| Initializer::Zeros.init([config.vocab_size], device));
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
        let mut x = self.embed(input_ids);
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
            tokens.div_ceil(LOSS_CHUNKS),
        )
    }

    fn project_logits(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, hidden] = x.dims();
        let (weight, bias) = self.output_parameters();
        let logits = matmul_2(x.reshape([batch * seq_len, hidden]), weight.transpose()).reshape([
            batch,
            seq_len,
            self.config.vocab_size,
        ]);
        match bias {
            Some(bias) => logits + bias.reshape([1, 1, self.config.vocab_size]),
            None => logits,
        }
    }

    fn project_last_logits(&self, x: Tensor<3>) -> Tensor<2> {
        let [batch, seq_len, hidden] = x.dims();
        let x = x
            .slice([0..batch, seq_len - 1..seq_len, 0..hidden])
            .reshape([batch, hidden]);
        let (weight, bias) = self.output_parameters();
        let logits = matmul_2(x, weight.transpose());
        match bias {
            Some(bias) => logits + bias.reshape([1, self.config.vocab_size]),
            None => logits,
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
