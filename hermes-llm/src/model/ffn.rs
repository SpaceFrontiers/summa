//! Position-wise feed-forward layer.
//!
//! Gated or plain MLP using the activation selected by MAL.

use burn::module::ParamId;
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::activation::{gelu, gelu_approximate, relu, silu, softmax};
use burn_nn::{Dropout, DropoutConfig, Linear, LinearConfig};

use crate::mal::{Activation, BlockDef, ModelDef};

use super::matmul::{linear_low_precision, prepare_linear_for_inference};

#[derive(Module, Debug)]
pub struct FeedForward {
    // Keep the dense fields at their historical module paths. For MoE these
    // are routed expert zero, so dense checkpoints remain load-compatible.
    in_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
    #[module(skip)]
    intermediate: usize,
    #[module(skip)]
    gated: bool,
    moe: Option<SparseMoe>,
}

#[derive(Module, Debug)]
struct DenseFeedForward {
    in_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
    #[module(skip)]
    intermediate: usize,
    #[module(skip)]
    gated: bool,
}

impl DenseFeedForward {
    fn new(config: &ModelDef, block: &BlockDef, device: &Device) -> Self {
        let use_gate = block.ffn.gate;
        let use_bias = block.ffn.bias;
        let hidden = config.hidden_size;
        let intermediate = block.intermediate_size(hidden);

        let lin = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out)
                .with_bias(use_bias)
                .init(device)
        };

        let in_proj = lin(hidden, intermediate * if use_gate { 2 } else { 1 });
        let down_proj = lin(intermediate, hidden);

        Self {
            in_proj,
            down_proj,
            dropout: DropoutConfig::new(block.ffn.dropout).init(),
            activation: block.ffn.activation,
            intermediate,
            gated: use_gate,
        }
    }

    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        dense_forward(
            &self.in_proj,
            &self.down_proj,
            &self.dropout,
            self.activation,
            self.intermediate,
            self.gated,
            x,
        )
    }

    fn prepare_inference(&mut self) {
        prepare_linear_for_inference(&mut self.in_proj);
        prepare_linear_for_inference(&mut self.down_proj);
    }
}

/// Dropless token-choice routing with top-k gate renormalization.
///
/// This backend-portable implementation deliberately uses static expert
/// shapes: every expert receives the token matrix and inactive routes are
/// multiplied by zero. That makes correctness, autodiff, and checkpointing
/// portable today. A grouped sparse matmul kernel can replace the dispatch
/// internally without changing MAL or checkpoint structure.
#[derive(Module, Debug)]
struct SparseMoe {
    router: Linear,
    experts: Vec<DenseFeedForward>,
    shared_experts: Vec<DenseFeedForward>,
    #[module(skip)]
    top_k: usize,
    #[module(skip)]
    load_balance_loss_weight: f64,
    #[module(skip)]
    router_z_loss_weight: f64,
}

impl SparseMoe {
    fn new(config: &ModelDef, block: &BlockDef, device: &Device) -> Self {
        let moe = block.ffn.moe.as_ref().expect("MoE config is present");
        let router = LinearConfig::new(config.hidden_size, moe.experts)
            .with_bias(false)
            .init(device);
        let make_expert = || DenseFeedForward::new(config, block, device);
        Self {
            router,
            // The parent FeedForward's historical dense matrices are expert 0.
            experts: (1..moe.experts).map(|_| make_expert()).collect(),
            shared_experts: (0..moe.shared_experts).map(|_| make_expert()).collect(),
            top_k: moe.top_k,
            load_balance_loss_weight: moe.load_balance_loss_weight,
            router_z_loss_weight: moe.router_z_loss_weight,
        }
    }

    fn forward<const D: usize>(
        &self,
        x: Tensor<D>,
        expert_zero: Tensor<D>,
        collect_auxiliary: bool,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        let shape = x.dims();
        let hidden = shape[D - 1];
        let tokens = shape[..D - 1].iter().product::<usize>();
        let flat = x.reshape([tokens, hidden]);

        // Router math remains FP32 even when the residual/expert stream is BF16.
        let logits = self.router.forward(flat.clone().cast(DType::F32));
        let probabilities = softmax(logits.clone(), 1);
        let (top_weights, top_indices) = probabilities.clone().topk_with_indices(self.top_k, 1);
        let top_weights = top_weights.clone() / top_weights.sum_dim(1).clamp_min(1e-12);

        let mut output = Tensor::zeros_like(&flat);
        let mut balance = None;
        for expert_index in 0..=self.experts.len() {
            let assignments = top_indices.clone().equal_elem(expert_index as i64).float();
            let gate = (assignments.clone() * top_weights.clone())
                .sum_dim(1)
                .cast(flat.dtype());
            let expert_output = if expert_index == 0 {
                expert_zero.clone().reshape([tokens, hidden])
            } else {
                self.experts[expert_index - 1].forward(flat.clone())
            };
            output = output + expert_output * gate;

            if collect_auxiliary && self.load_balance_loss_weight != 0.0 {
                let route_fraction = assignments.mean();
                let probability_fraction = probabilities
                    .clone()
                    .slice([0..tokens, expert_index..expert_index + 1])
                    .mean();
                let term = route_fraction * probability_fraction;
                balance = Some(match balance {
                    Some(sum) => sum + term,
                    None => term,
                });
            }
        }
        for expert in &self.shared_experts {
            output = output + expert.forward(flat.clone());
        }

        let mut auxiliary = balance.map(|loss| {
            loss.mul_scalar(self.load_balance_loss_weight * (self.experts.len() + 1) as f64)
        });
        if collect_auxiliary && self.router_z_loss_weight != 0.0 {
            let maximum = logits.clone().max_dim(1);
            let log_z = (logits - maximum.clone()).exp().sum_dim(1).log() + maximum;
            let z_loss = log_z.square().mean().mul_scalar(self.router_z_loss_weight);
            auxiliary = Some(match auxiliary {
                Some(loss) => loss + z_loss,
                None => z_loss,
            });
        }

        (output.reshape(shape), auxiliary)
    }

    fn prepare_inference(&mut self) {
        prepare_linear_for_inference(&mut self.router);
        for expert in &mut self.experts {
            expert.prepare_inference();
        }
        for expert in &mut self.shared_experts {
            expert.prepare_inference();
        }
    }
}

impl FeedForward {
    pub fn new(config: &ModelDef, block: &BlockDef, device: &Device) -> Self {
        let dense = DenseFeedForward::new(config, block, device);
        Self {
            in_proj: dense.in_proj,
            down_proj: dense.down_proj,
            dropout: dense.dropout,
            activation: dense.activation,
            intermediate: dense.intermediate,
            gated: dense.gated,
            moe: block
                .ffn
                .moe
                .as_ref()
                .map(|_| SparseMoe::new(config, block, device)),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.forward_internal(x, false).0
    }

    pub(crate) fn forward_with_aux<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        self.forward_internal(x, true)
    }

    fn forward_internal<const D: usize>(
        &self,
        x: Tensor<D>,
        collect_auxiliary: bool,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        let expert_zero = dense_forward(
            &self.in_proj,
            &self.down_proj,
            &self.dropout,
            self.activation,
            self.intermediate,
            self.gated,
            x.clone(),
        );
        match &self.moe {
            Some(moe) => moe.forward(x, expert_zero, collect_auxiliary),
            None => (expert_zero, None),
        }
    }

    pub(crate) fn router_parameter_id(&self) -> Option<ParamId> {
        self.moe.as_ref().map(|moe| moe.router.weight.id)
    }

    pub(crate) fn prepare_inference(&mut self) {
        prepare_linear_for_inference(&mut self.in_proj);
        prepare_linear_for_inference(&mut self.down_proj);
        if let Some(moe) = &mut self.moe {
            moe.prepare_inference();
        }
    }
}

fn dense_forward<const D: usize>(
    in_proj: &Linear,
    down_proj: &Linear,
    dropout: &Dropout,
    activation: Activation,
    intermediate: usize,
    gated: bool,
    x: Tensor<D>,
) -> Tensor<D> {
    let act = |t: Tensor<D>| match activation {
        Activation::SwiGLU | Activation::SiLU => silu(t),
        Activation::GELU => gelu(t),
        Activation::ReLU => relu(t),
        Activation::GELUNew | Activation::GELUTanh => gelu_approximate(t),
    };
    // The whole chain stays in the residual-stream dtype (BF16 during CUDA
    // training): the down projection feeds the residual add without an FP32
    // promotion.
    let projected = linear_low_precision(in_proj, x);
    let hidden = if gated {
        let mut ranges = projected.dims().map(|size| 0..size);
        ranges[D - 1] = 0..intermediate;
        let gate = act(projected.clone().slice(ranges.clone()));
        ranges[D - 1] = intermediate..2 * intermediate;
        gate * projected.slice(ranges)
    } else {
        act(projected)
    };
    linear_low_precision(down_proj, dropout.forward(hidden))
}
