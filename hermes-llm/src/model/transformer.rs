//! Full language model assembled from the MAL definition.

use std::ops::Range;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{Initializer, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn::tensor::{DType, Int};
use burn_nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn_nn::{RotaryEncoding, RotaryEncodingConfig};

use crate::mal::{BlockDef, ModelDef, NormConfig, PositionEncoding};

use super::linear_cross_entropy::linear_cross_entropy;
use super::matmul::{matmul_2, matmul_input, prepare_linear_for_inference, stream_cast};
use super::{InferenceState, MemoryRouting, MemorySlotStatus, Norm, TransformerBlock};

pub(crate) struct RawLayerDiagnostic {
    pub activation: Tensor<3>,
    pub attention_weights: Option<Tensor<4>>,
    pub total_attention_heads: Option<usize>,
    pub mamba_state: Option<Tensor<3>>,
}

pub(crate) struct RawModelDiagnostic {
    pub embedding: Tensor<3>,
    pub layers: Vec<RawLayerDiagnostic>,
    pub final_norm: Tensor<3>,
}

/// Exact stored and ordinary wake-routed model capacity.
///
/// `routed_active_parameters` is the conventional per-token parameter
/// equivalent: it includes every non-expert parameter, complete MoE routers,
/// shared experts, and only each sparse layer's ordinary `top_k` experts.
/// Dream-only exploration is excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WakeParameterAccounting {
    pub stored_parameters: u64,
    pub routed_active_parameters: u64,
}

/// Reusable hidden states for chunked teacher/student distribution losses.
///
/// Constructing this value runs the backbone exactly once. Callers can then
/// project small position ranges to vocabulary logits without retaining a
/// full `[batch, sequence, vocabulary]` tensor or rerunning the backbone for
/// every chunk.
pub struct SelectedLogitProjector<'a> {
    model: &'a Transformer,
    hidden: Tensor<2>,
    positions: Tensor<1, Int>,
    selected: usize,
}

impl SelectedLogitProjector<'_> {
    pub fn len(&self) -> usize {
        self.selected
    }

    pub fn is_empty(&self) -> bool {
        self.selected == 0
    }

    /// Final-normalized hidden features consumed by the logical LM head for
    /// the selected positions.
    ///
    /// The returned rows come from the same retained backbone pass as
    /// [`Self::logits`].  This is primarily useful for training an isolated
    /// output-projection adapter while keeping the base Transformer frozen.
    pub fn hidden(&self, range: Range<usize>) -> Tensor<2> {
        assert!(
            range.start < range.end && range.end <= self.selected,
            "selected-hidden range {}..{} is outside 0..{}",
            range.start,
            range.end,
            self.selected
        );
        let positions = self.positions.clone().slice([range]);
        self.hidden.clone().select(0, positions)
    }

    pub fn logits(&self, range: Range<usize>) -> Tensor<2> {
        assert!(
            range.start < range.end && range.end <= self.selected,
            "selected-logit range {}..{} is outside 0..{}",
            range.start,
            range.end,
            self.selected
        );
        let positions = self.positions.clone().slice([range.clone()]);
        self.model
            .project_flat_hidden(self.hidden.clone().select(0, positions), range.len())
    }
}

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
        let sleep_memory = block.memory.is_some();
        for (name, dropout) in [
            ("block", block.dropout),
            ("attention", block.attention.dropout),
        ] {
            if !(0.0..1.0).contains(&dropout) {
                bail!("layer {i} {name} dropout must be in [0, 1), got {dropout}");
            }
        }
        validate_norm(&format!("layer {i} norm"), &block.norm)?;
        let ffns = match &block.memory {
            Some(memory) => {
                if block.ffn != crate::mal::FfnDef::default() {
                    bail!("layer {i} cannot configure both ffn and memory");
                }
                if memory.tiers.is_empty() {
                    bail!("layer {i} memory must contain at least one tier");
                }
                let mut names = std::collections::HashSet::new();
                for tier in &memory.tiers {
                    if !names.insert(&tier.name) {
                        bail!("layer {i} memory tier names must be unique");
                    }
                    let reserve = &tier.reserve_experts;
                    if reserve.capacity == 0 || reserve.rank == 0 {
                        bail!(
                            "layer {i} memory tier '{}' reserve capacity and rank must be positive",
                            tier.name
                        );
                    }
                    if reserve.top_k != 1 {
                        bail!(
                            "layer {i} memory tier '{}' reserve top_k must be 1 so wake active compute stays constant as stored capacity activates",
                            tier.name
                        );
                    }
                }
                memory
                    .tiers
                    .iter()
                    .map(|tier| (format!("memory tier '{}'", tier.name), &tier.ffn))
                    .collect::<Vec<_>>()
            }
            None => vec![("ffn".to_string(), &block.ffn)],
        };
        for (name, ffn) in ffns {
            if !(0.0..1.0).contains(&ffn.dropout) {
                bail!(
                    "layer {i} {name} dropout must be in [0, 1), got {}",
                    ffn.dropout
                );
            }
            let intermediate = match ffn.hidden_dim {
                Some(size) => size,
                None => config.hidden_size.checked_mul(4).ok_or_else(|| {
                    anyhow::anyhow!("layer {i} default {name} size overflows usize")
                })?,
            };
            if intermediate == 0 {
                bail!("layer {i} {name} hidden_dim must be positive");
            }
            if let Some(moe) = &ffn.moe {
                if moe.experts < 2 {
                    bail!("layer {i} {name} MoE experts must be at least 2");
                }
                if moe.top_k == 0 || moe.top_k > moe.experts {
                    bail!(
                        "layer {i} {name} MoE top_k must be in 1..={}, got {}",
                        moe.experts,
                        moe.top_k
                    );
                }
                if sleep_memory && moe.top_k == moe.experts {
                    bail!(
                        "layer {i} {name} persistent MoE experts must exceed top_k so Dreaming can add one distinct exploration expert"
                    );
                }
                for (loss_name, weight) in [
                    ("load_balance_loss_weight", moe.load_balance_loss_weight),
                    ("router_z_loss_weight", moe.router_z_loss_weight),
                ] {
                    if !weight.is_finite() || weight < 0.0 {
                        bail!("layer {i} {name} MoE {loss_name} must be finite and non-negative");
                    }
                }
            }
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

/// Shared MAL-assembled language model used for training, retrieval, and
/// cached autoregressive inference.
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
    /// Whether at least one persistent FFN MoE can supply a route outside its
    /// ordinary top-k. Reserve-memory routers are intentionally excluded.
    #[module(skip)]
    dream_routing_supported: bool,
}

impl Transformer {
    pub fn new(config: &ModelDef, device: &Device) -> Result<Self> {
        validate_config(config)?;
        let dream_routing_supported = (0..config.num_layers).any(|layer| {
            let block = config.block_for_layer(layer);
            match &block.memory {
                Some(memory) => memory.tiers.iter().any(|tier| {
                    tier.ffn
                        .moe
                        .as_ref()
                        .is_some_and(|moe| moe.experts > moe.top_k)
                }),
                None => block
                    .ffn
                    .moe
                    .as_ref()
                    .is_some_and(|moe| moe.experts > moe.top_k),
            }
        });

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
            dream_routing_supported,
        })
    }

    fn assert_dream_routing_supported(&self, routing: MemoryRouting) {
        if matches!(routing, MemoryRouting::Dream { .. }) {
            assert!(
                self.dream_routing_supported,
                "dream generation requires a persistent FFN MoE with at least one expert outside ordinary top-k"
            );
        }
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

    /// Full-sequence logits with dream-only random expert routing enabled.
    /// Wake execution must continue to use [`Self::forward`].
    pub fn forward_with_memory_routing(
        &self,
        input_ids: Tensor<2, Int>,
        start_pos: usize,
        routing: MemoryRouting,
    ) -> Tensor<3> {
        self.project_logits(self.forward_hidden_with_memory_routing(input_ids, start_pos, routing))
    }

    fn forward_hidden_with_memory_routing(
        &self,
        input_ids: Tensor<2, Int>,
        start_pos: usize,
        routing: MemoryRouting,
    ) -> Tensor<3> {
        self.assert_dream_routing_supported(routing);
        let [_, seq_len] = input_ids.dims();
        assert!(start_pos + seq_len <= self.config.max_seq_len);
        let mut hidden = stream_cast(self.embed(input_ids));
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_with_routing(
                hidden,
                &self.rope,
                start_pos,
                routing.salted(index as u64),
            );
        }
        self.final_norm.forward(hidden)
    }

    fn forward_hidden(&self, input_ids: Tensor<2, Int>, start_pos: usize) -> Tensor<3> {
        self.forward_hidden_through(input_ids, start_pos, self.layers.len())
    }

    fn forward_hidden_through(
        &self,
        input_ids: Tensor<2, Int>,
        start_pos: usize,
        layer_count: usize,
    ) -> Tensor<3> {
        self.forward_hidden_through_with_aux(input_ids, start_pos, layer_count)
            .0
    }

    fn forward_hidden_through_with_aux(
        &self,
        input_ids: Tensor<2, Int>,
        start_pos: usize,
        layer_count: usize,
    ) -> (Tensor<3>, Option<Tensor<1>>) {
        let [_, seq_len] = input_ids.dims();
        assert!(
            start_pos + seq_len <= self.config.max_seq_len,
            "input positions {start_pos}..{} exceed max_seq_len {}",
            start_pos + seq_len,
            self.config.max_seq_len
        );
        assert!(
            (1..=self.layers.len()).contains(&layer_count),
            "requested {layer_count} Transformer layers, model has {}",
            self.layers.len()
        );
        // The full-sequence path runs the residual stream in the training
        // compute dtype (BF16 under CUDA training-fusion). Incremental decode
        // (`forward_hidden_with_state`) keeps FP32: its scan/conv step kernels
        // are FP32-only.
        let mut x = stream_cast(self.embed(input_ids));
        let mut auxiliary = None;
        for layer in self.layers.iter().take(layer_count) {
            let (output, layer_auxiliary) = layer.forward_with_aux(x, &self.rope, start_pos);
            x = output;
            if let Some(layer_loss) = layer_auxiliary {
                auxiliary = Some(match auxiliary {
                    Some(loss) => loss + layer_loss,
                    None => layer_loss,
                });
            }
        }
        (self.final_norm.forward(x), auxiliary)
    }

    /// Next-token cross-entropy for training. The output projection is chunked
    /// so full-vocabulary logits are never retained for every input token.
    pub fn forward_loss(&self, input_ids: Tensor<2, Int>, targets: Tensor<2, Int>) -> Tensor<1> {
        self.forward_loss_with_router(input_ids, targets).0
    }

    /// Next-token loss plus the configured MoE router regularization.
    pub fn forward_loss_with_router(
        &self,
        input_ids: Tensor<2, Int>,
        targets: Tensor<2, Int>,
    ) -> (Tensor<1>, Option<Tensor<1>>) {
        let [batch, seq_len] = targets.dims();
        let tokens = batch * seq_len;
        let (hidden, router_loss) =
            self.forward_hidden_through_with_aux(input_ids, 0, self.layers.len());
        let hidden = hidden.reshape([tokens, self.config.hidden_size]);
        let (weight, bias) = self.output_parameters();
        let loss = linear_cross_entropy(
            hidden,
            weight,
            bias,
            targets.reshape([tokens]),
            self.config.vocab_size,
            tokens.div_ceil(LOSS_CHUNKS),
        );
        (loss, router_loss)
    }

    /// Causal cross-entropy over selected flattened token positions.
    ///
    /// `positions` indexes the row-major `[batch, sequence]` target tensor.
    /// This keeps structured fine-tuning target-only without assigning a
    /// sentinel token ID that could collide with a real vocabulary item.
    pub fn forward_masked_loss(
        &self,
        input_ids: Tensor<2, Int>,
        targets: Tensor<2, Int>,
        positions: Tensor<1, Int>,
    ) -> Tensor<1> {
        self.forward_masked_loss_with_router(input_ids, targets, positions)
            .0
    }

    /// Selected-position language loss plus configured MoE regularization.
    pub fn forward_masked_loss_with_router(
        &self,
        input_ids: Tensor<2, Int>,
        targets: Tensor<2, Int>,
        positions: Tensor<1, Int>,
    ) -> (Tensor<1>, Option<Tensor<1>>) {
        let [batch, seq_len] = targets.dims();
        assert_eq!(input_ids.dims(), [batch, seq_len]);
        let [selected_tokens] = positions.dims();
        assert!(selected_tokens > 0, "masked loss requires target tokens");
        let tokens = batch * seq_len;
        let (hidden, router_loss) =
            self.forward_hidden_through_with_aux(input_ids, 0, self.layers.len());
        let hidden = hidden
            .reshape([tokens, self.config.hidden_size])
            .select(0, positions.clone());
        let targets = targets.reshape([tokens]).select(0, positions);
        let (weight, bias) = self.output_parameters();
        let loss = linear_cross_entropy(
            hidden,
            weight,
            bias,
            targets,
            self.config.vocab_size,
            selected_tokens.div_ceil(LOSS_CHUNKS),
        );
        (loss, router_loss)
    }

    /// Vocabulary logits only for selected row-major token positions.
    ///
    /// Knowledge-seeding callers can chunk `positions` to avoid materializing
    /// a `[batch, sequence, vocabulary]` tensor for teacher/student divergence.
    pub fn forward_selected_logits(
        &self,
        input_ids: Tensor<2, Int>,
        positions: Tensor<1, Int>,
    ) -> Tensor<2> {
        let projector = self.prepare_selected_logits(input_ids, positions);
        projector.logits(0..projector.len())
    }

    /// Run the backbone once and retain its flattened hidden states for
    /// chunked selected-position projection.
    pub fn prepare_selected_logits(
        &self,
        input_ids: Tensor<2, Int>,
        positions: Tensor<1, Int>,
    ) -> SelectedLogitProjector<'_> {
        let [batch, sequence] = input_ids.dims();
        let [selected] = positions.dims();
        assert!(
            selected > 0,
            "selected logits require at least one position"
        );
        let hidden = self
            .forward_hidden_through(input_ids, 0, self.layers.len())
            .reshape([batch * sequence, self.config.hidden_size]);
        SelectedLogitProjector {
            model: self,
            hidden,
            positions,
            selected,
        }
    }

    /// Run the backbone once with an explicit memory routing mode and retain
    /// only the requested rows for hidden-state inspection and vocabulary
    /// projection. Dreaming uses this to apply its isolated generation-policy
    /// adapter without materializing logits for every prefix position.
    pub fn prepare_selected_logits_with_memory_routing(
        &self,
        input_ids: Tensor<2, Int>,
        positions: Tensor<1, Int>,
        routing: MemoryRouting,
    ) -> SelectedLogitProjector<'_> {
        let [batch, sequence] = input_ids.dims();
        let [selected] = positions.dims();
        assert!(
            selected > 0,
            "selected logits require at least one position"
        );
        let hidden = self
            .forward_hidden_with_memory_routing(input_ids, 0, routing)
            .reshape([batch * sequence, self.config.hidden_size]);
        SelectedLogitProjector {
            model: self,
            hidden,
            positions,
            selected,
        }
    }

    fn project_flat_hidden(&self, hidden: Tensor<2>, selected: usize) -> Tensor<2> {
        let stored_vocab = self.config.padded_vocab_size();
        let (weight, bias) = self.output_parameters();
        let logits = matmul_2(hidden, weight.transpose());
        let logits = match bias {
            Some(bias) => logits + bias.reshape([1, stored_vocab]),
            None => logits,
        };
        if stored_vocab == self.config.vocab_size {
            logits
        } else {
            logits.slice([0..selected, 0..self.config.vocab_size])
        }
    }

    /// L2-normalized last-meaningful-token embeddings for retrieval training.
    ///
    /// `end_positions` contains one row-major index into the flattened
    /// `[batch, sequence]` hidden states per input row. `layer` is one-based;
    /// `None` reads the final layer. Applying the shared final norm keeps this
    /// path checkpoint-compatible while allowing a hybrid model to read after
    /// a selected global-attention layer.
    pub fn forward_embeddings(
        &self,
        input_ids: Tensor<2, Int>,
        end_positions: Tensor<1, Int>,
        layer: Option<usize>,
    ) -> Tensor<2> {
        self.forward_embeddings_with_router(input_ids, end_positions, layer)
            .0
    }

    /// Retrieval embeddings plus configured router regularization.
    pub fn forward_embeddings_with_router(
        &self,
        input_ids: Tensor<2, Int>,
        end_positions: Tensor<1, Int>,
        layer: Option<usize>,
    ) -> (Tensor<2>, Option<Tensor<1>>) {
        let [batch, seq_len] = input_ids.dims();
        assert_eq!(end_positions.dims(), [batch]);
        let (hidden, router_loss) =
            self.forward_hidden_through_with_aux(input_ids, 0, layer.unwrap_or(self.layers.len()));
        let hidden = hidden
            .reshape([batch * seq_len, self.config.hidden_size])
            .select(0, end_positions)
            .cast(DType::F32);
        let norm = hidden.clone().square().sum_dim(1).sqrt().clamp_min(1e-12);
        (hidden / norm, router_loss)
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

    pub(crate) fn device(&self) -> Device {
        self.embedding.weight.val().device()
    }

    pub fn num_parameters(&self) -> usize {
        self.num_params()
    }

    /// Measure ordinary wake capacity from the instantiated module tree.
    ///
    /// Memory hierarchies use their synchronized checkpoint active-slot masks.
    /// A dormant reserve executes the fixed, non-parameter zero fallback lane,
    /// so its routed parameter-equivalent remains constant as slots activate.
    pub fn wake_parameter_accounting(&self) -> Result<WakeParameterAccounting> {
        let stored = self.num_parameters();
        let mut dormant = 0_usize;
        for (layer, block) in self.layers.iter().enumerate() {
            let (block_stored, block_routed) = block
                .wake_parameter_counts()
                .with_context(|| format!("cannot account wake parameters for layer {layer}"))?;
            dormant = dormant
                .checked_add(block_stored.checked_sub(block_routed).with_context(|| {
                    format!("layer {layer} routed parameter count exceeds stored count")
                })?)
                .context("model dormant parameter count overflows usize")?;
        }
        let routed = stored
            .checked_sub(dormant)
            .context("model routed parameter count exceeds stored count")?;
        Ok(WakeParameterAccounting {
            stored_parameters: stored
                .try_into()
                .context("stored parameter count exceeds u64")?,
            routed_active_parameters: routed
                .try_into()
                .context("routed parameter count exceeds u64")?,
        })
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
        let non_muon_ids = self
            .layers
            .iter()
            .flat_map(TransformerBlock::non_muon_parameter_ids)
            .collect::<Vec<_>>();
        visitor.ids.retain(|id| !non_muon_ids.contains(id));
        visitor.ids
    }

    /// Parameter IDs whose stored tensors are encoded by the ultra-low-bit
    /// archive and replaced by fake-quantized values during QAT.  Keeping the
    /// selection in the model makes the training forward and archive export
    /// agree without relying on fragile serialized parameter names.
    pub fn ultra_quant_parameter_ids(
        &self,
        quantize_embeddings: bool,
        quantize_lm_head: bool,
    ) -> Result<Vec<ParamId>> {
        ensure!(
            !self.config.embeddings.tie_weights || quantize_embeddings == quantize_lm_head,
            "tied embedding/output weights cannot use different quantization policies"
        );
        let mut visitor = StoredMatrixParameterVisitor::default();
        self.visit(&mut visitor);
        if !quantize_embeddings {
            visitor.ids.retain(|id| *id != self.embedding.weight.id);
        }
        if !quantize_lm_head && let Some(head) = &self.lm_head {
            visitor.ids.retain(|id| *id != head.weight.id);
        }
        ensure!(
            !visitor.ids.is_empty(),
            "quantization policy selected no rank-two-or-higher parameters"
        );
        Ok(visitor.ids)
    }

    /// Visit one transformer block without exposing its module fields. This is
    /// used by opt-in trainer diagnostics such as per-layer gradient norms.
    pub fn visit_layer<V: ModuleVisitor>(&self, index: usize, visitor: &mut V) -> Result<()> {
        let layer = self.layers.get(index).ok_or_else(|| {
            anyhow::anyhow!(
                "layer index {index} is outside model with {} layers",
                self.layers.len()
            )
        })?;
        layer.visit(visitor);
        Ok(())
    }

    /// Full-sequence diagnostic pass used only by `hermes-llm trace`.
    pub(crate) fn forward_diagnostic(
        &self,
        input_ids: Tensor<2, Int>,
        max_attention_heads: usize,
    ) -> RawModelDiagnostic {
        let [batch, seq_len] = input_ids.dims();
        assert_eq!(batch, 1, "visualization tracing supports one sequence");
        assert!(seq_len > 0, "visualization tracing requires tokens");
        assert!(
            seq_len <= self.config.max_seq_len,
            "visualization trace length {seq_len} exceeds max_seq_len {}",
            self.config.max_seq_len
        );
        assert!(
            max_attention_heads > 0,
            "visualization tracing requires at least one attention head"
        );

        let mut x = stream_cast(self.embed(input_ids));
        let embedding = x.clone();
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (output, diagnostic) =
                layer.forward_diagnostic(x, &self.rope, 0, max_attention_heads);
            x = output;
            layers.push(RawLayerDiagnostic {
                activation: x.clone(),
                attention_weights: diagnostic.attention_weights,
                total_attention_heads: diagnostic.total_attention_heads,
                mamba_state: diagnostic.mamba_state,
            });
        }
        let final_norm = self.final_norm.forward(x);
        RawModelDiagnostic {
            embedding,
            layers,
            final_norm,
        }
    }

    pub fn make_state(&self, batch: usize, device: &Device) -> InferenceState {
        self.make_state_with_capacity(batch, self.config.max_seq_len, device)
    }

    /// Build recurrent state with a right-sized attention KV allocation.
    /// Mamba state is constant-sized and is unaffected by `capacity`.
    pub fn make_state_with_capacity(
        &self,
        batch: usize,
        capacity: usize,
        device: &Device,
    ) -> InferenceState {
        assert!(capacity > 0, "inference state capacity must be positive");
        assert!(
            capacity <= self.config.max_seq_len,
            "inference state capacity {capacity} exceeds max_seq_len {}",
            self.config.max_seq_len
        );
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.make_state_with_capacity(batch, capacity, device))
            .collect();
        InferenceState {
            layers,
            pos: 0,
            capacity,
        }
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

    /// Cached generation with optional dream-only random expert routing.
    pub fn forward_next_logits_with_state_and_memory_routing(
        &self,
        input_ids: Tensor<2, Int>,
        state: &mut InferenceState,
        routing: MemoryRouting,
    ) -> Tensor<2> {
        self.project_last_logits(
            self.forward_hidden_with_state_and_routing(input_ids, state, routing),
        )
    }

    fn forward_hidden_with_state(
        &self,
        input_ids: Tensor<2, Int>,
        state: &mut InferenceState,
    ) -> Tensor<3> {
        self.forward_hidden_with_state_and_routing(input_ids, state, MemoryRouting::Wake)
    }

    fn forward_hidden_with_state_and_routing(
        &self,
        input_ids: Tensor<2, Int>,
        state: &mut InferenceState,
        routing: MemoryRouting,
    ) -> Tensor<3> {
        self.assert_dream_routing_supported(routing);
        let [batch, seq_len] = input_ids.dims();
        assert!(
            batch > 0 && seq_len > 0,
            "incremental inference requires non-empty input"
        );
        assert!(
            state.pos + seq_len <= state.capacity,
            "inference state at position {} + {} tokens exceeds cache capacity {}",
            state.pos,
            seq_len,
            state.capacity
        );
        assert_eq!(
            state.layers.len(),
            self.layers.len(),
            "inference state belongs to a model with a different layer count"
        );

        // Autoregressive dreaming normally calls this path once per token.
        // Include the absolute position in the exploration seed so its random
        // extra expert does not stay fixed for an entire generated sequence.
        let routing = routing.salted(state.pos as u64);
        let mut x = self.embed(input_ids);
        for (index, (layer, layer_state)) in
            self.layers.iter().zip(state.layers.iter_mut()).enumerate()
        {
            x = layer.forward_with_state_and_routing(
                x,
                &self.rope,
                state.pos,
                layer_state,
                routing.salted(index as u64),
            );
        }
        state.pos += seq_len;
        self.final_norm.forward(x)
    }

    /// Refresh runtime mirrors from checkpointed memory masks. Strict Hermes
    /// loaders call this automatically; call it after applying snapshots with
    /// Burn APIs directly.
    pub fn sync_memory_state(&mut self) {
        for layer in &mut self.layers {
            layer.sync_memory_state();
        }
    }

    pub(crate) fn prepare_memory_upgrade_state(&mut self) {
        for layer in &mut self.layers {
            layer.prepare_memory_upgrade_state();
        }
    }

    pub fn memory_slot_statuses(&self) -> Vec<MemorySlotStatus> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(index, layer)| layer.memory_statuses(index))
            .collect()
    }

    /// Activate one preallocated low-rank reserve slot.
    pub fn activate_memory_slot(&mut self, layer: usize, tier: usize, slot: usize) -> Result<()> {
        self.layers
            .get_mut(layer)
            .ok_or_else(|| anyhow::anyhow!("layer {layer} does not exist"))?
            .activate_memory_slot(tier, slot)
    }

    /// Activate the same logical reserve slot in every block that owns the
    /// requested memory tier. Consolidation treats these per-block tensors as
    /// one model-wide slot, so preflight validation happens before mutation.
    pub fn activate_memory_slot_all_layers(
        &mut self,
        tier: usize,
        slot: usize,
    ) -> Result<Vec<ParamId>> {
        let targets = self
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == tier && status.slot == slot)
            .collect::<Vec<_>>();
        ensure!(
            !targets.is_empty(),
            "memory tier {tier} slot {slot} does not exist"
        );
        ensure!(
            targets.iter().all(|status| !status.active),
            "memory tier {tier} slot {slot} is active in at least one layer"
        );
        ensure!(
            targets
                .windows(2)
                .all(|pair| pair[0].generation == pair[1].generation),
            "memory tier {tier} slot {slot} generations differ across layers"
        );
        let mut parameter_ids = Vec::new();
        for target in targets {
            parameter_ids.extend(target.parameter_ids);
            self.layers[target.layer].activate_memory_slot(tier, slot)?;
        }
        Ok(parameter_ids)
    }

    /// Deactivate a reserve slot without modifying its learned tensors.
    /// Transaction rollback can use this; capacity reclamation should use
    /// [`Self::reset_memory_slot`] instead.
    pub fn deactivate_memory_slot(&mut self, layer: usize, tier: usize, slot: usize) -> Result<()> {
        self.layers
            .get_mut(layer)
            .ok_or_else(|| anyhow::anyhow!("layer {layer} does not exist"))?
            .deactivate_memory_slot(tier, slot)
    }

    /// Reset a reclaimed slot to a dormant no-op and advance its serialized
    /// generation. Optimizer moments for the returned parameter IDs must be
    /// cleared by the trainer in the same transaction.
    pub fn reset_memory_slot(
        &mut self,
        layer_index: usize,
        tier: usize,
        slot: usize,
        seed: u64,
    ) -> Result<Vec<ParamId>> {
        let layer = self
            .layers
            .get_mut(layer_index)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_index} does not exist"))?;
        layer.reset_memory_slot(tier, slot, seed)?;
        Ok(layer
            .memory_statuses(layer_index)
            .into_iter()
            .find(|status| status.tier == tier && status.slot == slot)
            .expect("reset slot remains allocated")
            .parameter_ids)
    }

    /// Reclaim the same logical slot in every memory-bearing layer. The seed
    /// is salted by layer while the serialized generation remains aligned.
    pub fn reset_memory_slot_all_layers(
        &mut self,
        tier: usize,
        slot: usize,
        seed: u64,
    ) -> Result<Vec<ParamId>> {
        let targets = self
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == tier && status.slot == slot)
            .collect::<Vec<_>>();
        ensure!(
            !targets.is_empty(),
            "memory tier {tier} slot {slot} does not exist"
        );
        ensure!(
            targets.iter().all(|status| status.active),
            "memory tier {tier} slot {slot} is dormant in at least one layer"
        );
        ensure!(
            targets
                .windows(2)
                .all(|pair| pair[0].generation == pair[1].generation),
            "memory tier {tier} slot {slot} generations differ across layers"
        );
        let mut parameter_ids = Vec::new();
        for target in targets {
            parameter_ids.extend(self.reset_memory_slot(
                target.layer,
                tier,
                slot,
                seed ^ (target.layer as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            )?);
        }
        Ok(parameter_ids)
    }

    /// Float parameters belonging to a complete memory tier.
    pub fn memory_tier_parameter_ids(&self, layer: usize, tier: usize) -> Result<Vec<ParamId>> {
        self.layers
            .get(layer)
            .ok_or_else(|| anyhow::anyhow!("layer {layer} does not exist"))?
            .memory_tier_parameter_ids(tier)
    }

    /// Float parameters for one logical tier across all memory-bearing layers.
    pub fn memory_tier_parameter_ids_all_layers(&self, tier: usize) -> Result<Vec<ParamId>> {
        let layers = self
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == tier)
            .map(|status| status.layer)
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(!layers.is_empty(), "memory tier {tier} does not exist");
        let mut ids = Vec::new();
        for layer in layers {
            ids.extend(self.memory_tier_parameter_ids(layer, tier)?);
        }
        Ok(ids)
    }

    /// Persistent base FFN/MoE parameters for one memory tier across every
    /// memory-bearing layer. Reserve expert tensors are deliberately excluded.
    pub fn memory_tier_base_parameter_ids_all_layers(&self, tier: usize) -> Result<Vec<ParamId>> {
        let layers = self
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == tier)
            .map(|status| status.layer)
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(!layers.is_empty(), "memory tier {tier} does not exist");
        let mut ids = Vec::new();
        for layer in layers {
            ids.extend(
                self.layers[layer]
                    .memory_tier_base_parameter_ids(tier)
                    .with_context(|| format!("reading layer {layer} memory tier {tier} base"))?,
            );
        }
        ensure!(
            !ids.is_empty(),
            "memory tier {tier} base has no float parameters"
        );
        Ok(ids)
    }
}

#[derive(Default)]
struct MatrixParameterVisitor {
    ids: Vec<ParamId>,
}

#[derive(Default)]
struct StoredMatrixParameterVisitor {
    ids: Vec<ParamId>,
}

impl ModuleVisitor for StoredMatrixParameterVisitor {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if D >= 2 {
            self.ids.push(param.id);
        }
    }
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

    #[test]
    fn wake_parameter_accounting_is_exact_for_dense_and_sparse_ffns() {
        let dense_config = crate::mal::parse_mal(
            r#"
            ffn dense { hidden_dim: 12 activation: swiglu bias: false }
            model dense_test {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } ffn: dense }
            }
            "#,
        )
        .unwrap();
        let dense = Transformer::new(&dense_config, &Device::ndarray()).unwrap();
        let dense_accounting = dense.wake_parameter_accounting().unwrap();
        assert_eq!(
            dense_accounting.stored_parameters,
            dense.num_parameters() as u64
        );
        assert_eq!(
            dense_accounting.routed_active_parameters,
            dense_accounting.stored_parameters
        );

        let sparse_config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12 activation: swiglu bias: false
                moe { experts: 4 top_k: 2 shared_experts: 1 }
            }
            model sparse_test {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } ffn: routed }
            }
            "#,
        )
        .unwrap();
        let sparse = Transformer::new(&sparse_config, &Device::ndarray()).unwrap();
        let sparse_accounting = sparse.wake_parameter_accounting().unwrap();
        // Two of four routed experts are dormant. A gated 8 -> 12 -> 8
        // bias-free expert stores (8 * 24) + (12 * 8) parameters.
        let one_expert = 8_u64 * 24 + 12 * 8;
        assert_eq!(
            sparse_accounting.stored_parameters - sparse_accounting.routed_active_parameters,
            2 * one_expert
        );
    }

    #[test]
    fn memory_accounting_uses_fixed_mask_aware_reserve_lane() {
        let config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } memory: cms }
            }
            "#,
        )
        .unwrap();
        let mut model = Transformer::new(&config, &Device::ndarray()).unwrap();
        let dormant = model.wake_parameter_accounting().unwrap();
        assert_eq!(dormant.stored_parameters, model.num_parameters() as u64);
        model.activate_memory_slot(0, 0, 0).unwrap();
        let active = model.wake_parameter_accounting().unwrap();
        assert_eq!(active.stored_parameters, dormant.stored_parameters);
        assert_eq!(
            active.routed_active_parameters, dormant.routed_active_parameters,
            "activating a preallocated reserve must replace, not add, the fixed route lane"
        );
    }
    use crate::mal::get_builtin_model;
    use burn::tensor::TensorData;

    fn run_configurable_moe_backward(device: Device) {
        let config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 16
                moe {
                    experts: 4
                    top_k: 2
                    shared_experts: 1
                    load_balance_loss_weight: 0.01
                    router_z_loss_weight: 0.001
                }
            }
            model moe_test {
                vocab_size: 32
                max_seq_len: 8
                hidden_size: 8
                num_layers: 1
                block: {
                    attention: { num_heads: 1 position_encoding: none }
                    ffn: routed
                }
            }
            "#,
        )
        .unwrap();
        let device = device.autodiff();
        let model = Transformer::new(&config, &device).unwrap();
        let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
        let targets = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], &device);
        let (task, router) = model.forward_loss_with_router(input, targets);
        let router = router.expect("MoE must emit its configured router objective");
        let total = task + router;
        assert!(
            total
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap()[0]
                .is_finite()
        );
        let _grads = total.backward();
    }

    #[test]
    fn configurable_moe_runs_task_and_router_backward() {
        run_configurable_moe_backward(Device::ndarray());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn configurable_moe_runs_on_metal() {
        run_configurable_moe_backward(Device::metal(burn::tensor::DeviceKind::DefaultDevice));
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn configurable_moe_runs_on_cuda() {
        run_configurable_moe_backward(Device::cuda(0));
    }

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

    #[test]
    fn masked_loss_matches_full_loss_when_every_target_is_selected() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 1;
        config.max_seq_len = 8;
        config.block.attention.num_heads = Some(2);
        config.block.attention.num_kv_heads = Some(1);
        config.block.attention.head_dim = Some(4);
        config.block.ffn.hidden_dim = Some(16);
        config.block.dropout = 0.0;
        config.block.attention.dropout = 0.0;
        config.block.ffn.dropout = 0.0;
        let device = Device::ndarray().autodiff();
        device.seed(37);
        let model = Transformer::new(&config, &device).unwrap();
        let inputs = vec![1_i64, 2, 3, 4, 5, 6, 7, 8];
        let targets = vec![2_i64, 3, 4, 5, 6, 7, 8, 9];
        let batch = || {
            (
                Tensor::<2, Int>::from_data(TensorData::new(inputs.clone(), [2, 4]), &device),
                Tensor::<2, Int>::from_data(TensorData::new(targets.clone(), [2, 4]), &device),
            )
        };
        let (input, target) = batch();
        let full = model.forward_loss(input, target);
        let (input, target) = batch();
        let positions = Tensor::<1, Int>::arange(0..8, &device);
        let masked = model.forward_masked_loss(input, target, positions);
        let value = |loss: Tensor<1>| loss.into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
        assert!((value(full) - value(masked)).abs() < 1e-5);
    }

    #[test]
    fn selected_logits_match_materialized_positions() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 1;
        config.max_seq_len = 8;
        config.block.attention.num_heads = Some(2);
        config.block.attention.num_kv_heads = Some(1);
        config.block.attention.head_dim = Some(4);
        config.block.ffn.hidden_dim = Some(16);
        let device = Device::ndarray();
        device.seed(41);
        let model = Transformer::new(&config, &device).unwrap();
        let ids = vec![1_i64, 2, 3, 4, 5, 6, 7, 8];
        let input = || Tensor::<2, Int>::from_data(TensorData::new(ids.clone(), [2, 4]), &device);
        let positions = Tensor::<1, Int>::from_data([0_i64, 3, 5], &device);
        let expected = model
            .forward(input(), 0)
            .reshape([8, config.vocab_size])
            .select(0, positions.clone());
        let selected = model.forward_selected_logits(input(), positions.clone());
        let maximum: f32 = (expected.clone() - selected).abs().max().into_scalar();
        assert!(maximum < 1e-6, "selected logits differ by {maximum}");

        let projector = model.prepare_selected_logits(input(), positions);
        assert_eq!(projector.len(), 3);
        assert!(!projector.is_empty());
        let hidden = projector.hidden(0..3);
        assert_eq!(hidden.dims(), [3, config.hidden_size]);
        let chunked_hidden = Tensor::cat(vec![projector.hidden(0..1), projector.hidden(1..3)], 0);
        let maximum: f32 = (hidden - chunked_hidden).abs().max().into_scalar();
        assert!(
            maximum < 1e-6,
            "chunked selected hidden features differ by {maximum}"
        );
        let chunked = Tensor::cat(vec![projector.logits(0..1), projector.logits(1..3)], 0);
        let maximum: f32 = (expected - chunked).abs().max().into_scalar();
        assert!(
            maximum < 1e-6,
            "chunked selected logits differ by {maximum}"
        );
    }

    #[test]
    fn retrieval_embeddings_use_requested_layer_and_unit_normalization() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 3;
        config.max_seq_len = 8;
        for block in config.pattern.as_mut().unwrap() {
            block.dropout = 0.0;
            block.attention.dropout = 0.0;
            block.attention.num_heads = Some(2);
            block.attention.num_kv_heads = Some(1);
            block.attention.head_dim = Some(4);
            block.ffn.dropout = 0.0;
            block.ffn.hidden_dim = Some(16);
        }
        let device = Device::ndarray().autodiff();
        device.seed(43);
        let model = Transformer::new(&config, &device).unwrap();
        let input = || {
            Tensor::<2, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 3, 0, 4, 5, 6, 7], [2, 4]),
                &device,
            )
        };
        let ends = || Tensor::<1, Int>::from_data(TensorData::new(vec![2_i64, 7], [2]), &device);
        let early = model.forward_embeddings(input(), ends(), Some(2));
        let final_layer = model.forward_embeddings(input(), ends(), None);
        assert_eq!(early.dims(), [2, 8]);
        let norms = final_layer
            .clone()
            .square()
            .sum_dim(1)
            .sqrt()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert!(norms.into_iter().all(|norm| (norm - 1.0).abs() < 1e-5));
        let max_diff = early
            .sub(final_layer)
            .abs()
            .max()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            max_diff > 1e-6,
            "different layers produced identical embeddings"
        );
    }

    #[test]
    fn diagnostic_pass_matches_forward_and_respects_causal_attention() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 3;
        config.max_seq_len = 8;
        for block in config.pattern.as_mut().unwrap() {
            block.dropout = 0.0;
            block.attention.dropout = 0.0;
            block.attention.num_heads = Some(2);
            block.attention.num_kv_heads = Some(1);
            block.attention.head_dim = Some(4);
            block.ffn.dropout = 0.0;
            block.ffn.hidden_dim = Some(16);
        }
        let device = Device::ndarray();
        device.seed(47);
        let model = Transformer::new(&config, &device).unwrap();
        let input =
            || Tensor::<2, Int>::from_data(TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]), &device);
        let expected = model.forward_hidden(input(), 0);
        let diagnostic = model.forward_diagnostic(input(), 1);
        let max_diff = expected
            .sub(diagnostic.final_norm.clone())
            .abs()
            .max()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            max_diff < 1e-5,
            "diagnostic pass changed output by {max_diff}"
        );
        assert_eq!(diagnostic.embedding.dims(), [1, 4, 8]);
        assert_eq!(diagnostic.layers.len(), 3);

        let mut attention_layers = 0;
        let mut mamba_layers = 0;
        for layer in diagnostic.layers {
            if let Some(weights) = layer.attention_weights {
                attention_layers += 1;
                assert_eq!(weights.dims(), [1, 1, 4, 4]);
                let values = weights
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap();
                for row in 0..4 {
                    for column in row + 1..4 {
                        assert!(
                            values[row * 4 + column].abs() < 1e-7,
                            "future attention at ({row}, {column}) was {}",
                            values[row * 4 + column]
                        );
                    }
                }
            }
            if let Some(state) = layer.mamba_state {
                mamba_layers += 1;
                assert_eq!(state.dims()[0], 1);
            }
        }
        assert!(attention_layers > 0);
        assert!(mamba_layers > 0);
    }

    #[test]
    fn logical_memory_slots_stay_aligned_across_layers() {
        let config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
                tier slow {
                    ffn: base residual_init: zero
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 2
                block: { attention: { num_heads: 1 } memory: cms }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        let mut model = Transformer::new(&config, &device).unwrap();
        let ids = model.activate_memory_slot_all_layers(1, 0).unwrap();
        assert!(!ids.is_empty());
        let base_ids = model.memory_tier_base_parameter_ids_all_layers(1).unwrap();
        let all_tier_ids = model.memory_tier_parameter_ids_all_layers(1).unwrap();
        assert!(!base_ids.is_empty());
        assert!(base_ids.iter().all(|id| all_tier_ids.contains(id)));
        assert!(base_ids.iter().all(|id| !ids.contains(id)));
        let active = model
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == 1 && status.slot == 0)
            .collect::<Vec<_>>();
        assert_eq!(active.len(), 2);
        assert!(
            active
                .iter()
                .all(|status| status.active && status.generation == 1)
        );
        assert!(model.activate_memory_slot_all_layers(1, 0).is_err());

        let reset_ids = model.reset_memory_slot_all_layers(1, 0, 17).unwrap();
        assert_eq!(reset_ids.len(), ids.len());
        let reset = model
            .memory_slot_statuses()
            .into_iter()
            .filter(|status| status.tier == 1 && status.slot == 0)
            .collect::<Vec<_>>();
        assert!(
            reset
                .iter()
                .all(|status| !status.active && status.generation == 2)
        );
        assert!(
            !model
                .memory_tier_parameter_ids_all_layers(1)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn memory_reserve_rejects_growing_top_k_compute() {
        let config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 3 top_k: 2 }
                }
                tier slow {
                    ffn: base residual_init: zero
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } memory: cms }
            }
            "#,
        )
        .unwrap();
        let error = Transformer::new(&config, &Device::ndarray())
            .unwrap_err()
            .to_string();
        assert!(error.contains("reserve top_k must be 1"), "{error}");
    }

    #[test]
    fn first_active_reserve_keeps_dream_exploration_on_persistent_moe() {
        let config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12
                moe { experts: 3 top_k: 1 }
            }
            memory cms {
                tier fast {
                    ffn: routed
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } memory: cms }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(91);
        let mut model = Transformer::new(&config, &device).unwrap();
        model.activate_memory_slot_all_layers(0, 0).unwrap();
        assert_eq!(
            model
                .memory_slot_statuses()
                .iter()
                .filter(|status| status.active)
                .count(),
            1
        );

        let input = Tensor::<2, Int>::from_data([[1_i64, 2, 3]], &device);
        let wake = model.forward(input.clone(), 0);
        let dream = model.forward_with_memory_routing(
            input.clone(),
            0,
            MemoryRouting::Dream { seed: 0x1234 },
        );
        let selected = model
            .prepare_selected_logits_with_memory_routing(
                input,
                Tensor::<1, Int>::from_data([2_i64], &device),
                MemoryRouting::Dream { seed: 0x1234 },
            )
            .logits(0..1);
        let selected_difference: f32 = (selected
            - dream
                .clone()
                .slice([0..1, 2..3, 0..config.vocab_size])
                .reshape([1, config.vocab_size]))
        .abs()
        .max()
        .into_scalar();
        assert!(selected_difference < 1e-6);
        let difference: f32 = (dream - wake).abs().max().into_scalar();
        assert!(
            difference > 0.0,
            "persistent MoE exploration should remain active with one receiver slot"
        );
    }

    #[test]
    fn sleep_memory_rejects_persistent_moe_without_dream_capacity() {
        let config = crate::mal::parse_mal(
            r#"
            ffn routed { hidden_dim: 12 moe { experts: 2 top_k: 2 } }
            memory cms {
                tier fast {
                    ffn: routed
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } memory: cms }
            }
            "#,
        )
        .unwrap();
        let error = Transformer::new(&config, &Device::ndarray())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("persistent MoE experts must exceed top_k"),
            "{error}"
        );
    }

    #[test]
    #[should_panic(expected = "dream generation requires a persistent FFN MoE")]
    fn dense_model_rejects_dream_expert_routing() {
        let config = crate::mal::parse_mal(
            r#"
            ffn dense { hidden_dim: 12 }
            model dense {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } ffn: dense }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        let model = Transformer::new(&config, &device).unwrap();
        let input = Tensor::<2, Int>::from_data([[1_i64, 2]], &device);
        let _ = model.forward_with_memory_routing(input, 0, MemoryRouting::Dream { seed: 1 });
    }
}
