//! Position-wise feed-forward layer.
//!
//! Gated or plain MLP using the activation selected by MAL.

use burn::module::{Param, ParamId};
use burn::prelude::*;
use burn::tensor::activation::{gelu, gelu_approximate, relu, silu, softmax};
use burn::tensor::{DType, TensorData};
use burn_nn::{Dropout, DropoutConfig, Linear, LinearConfig};

use crate::mal::{Activation, BlockDef, ModelDef};

#[cfg(feature = "cuda")]
use super::fused_swiglu::fused_swiglu;
#[cfg(feature = "cuda")]
use super::grouped_linear::{grouped_linear, grouped_swiglu_mlp, is_cuda_device};
#[cfg(feature = "cuda")]
use super::matmul::matmul_input;
use super::matmul::{
    batched_linear_low_precision, linear_low_precision, prepare_linear_for_inference,
};
#[cfg(feature = "cuda")]
use super::moe_dispatch::{route_combine, route_gather};
#[cfg(feature = "cuda")]
use super::moe_route::route_plan;
use super::moe_topk::top2_indices;
use super::row_permute::row_permute;

#[derive(Module, Debug)]
pub struct FeedForward {
    // Option doesn't add a module-record path, so dense checkpoints keep their
    // historical `in_proj.*` and `down_proj.*` keys. MoE layers leave these
    // empty and own a separate expert collection.
    in_proj: Option<Linear>,
    down_proj: Option<Linear>,
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

#[derive(Clone, Copy)]
struct DenseFeedForwardView<'a> {
    in_proj: &'a Linear,
    down_proj: &'a Linear,
    dropout: &'a Dropout,
    activation: Activation,
    intermediate: usize,
    gated: bool,
}

impl DenseFeedForwardView<'_> {
    fn forward<const D: usize>(self, x: Tensor<D>) -> Tensor<D> {
        dense_forward(
            self.in_proj,
            self.down_proj,
            self.dropout,
            self.activation,
            self.intermediate,
            self.gated,
            x,
        )
    }
}

#[derive(Module, Debug)]
struct ExpertBank {
    // Experts are stored in their execution layout. Unlike stacking a Vec of
    // Linear modules on every call, these tensors feed batched GEMM directly.
    in_weight: Param<Tensor<3>>,
    in_bias: Option<Param<Tensor<2>>>,
    down_weight: Param<Tensor<3>>,
    down_bias: Option<Param<Tensor<2>>>,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
    #[module(skip)]
    intermediate: usize,
    #[module(skip)]
    gated: bool,
    #[module(skip)]
    expert_count: usize,
}

impl ExpertBank {
    fn new(config: &ModelDef, block: &BlockDef, expert_count: usize, device: &Device) -> Self {
        let experts = (0..expert_count)
            .map(|_| DenseFeedForward::new(config, block, device))
            .collect::<Vec<_>>();
        let stack_bias = |select: fn(&DenseFeedForward) -> &Linear| {
            select(&experts[0]).bias.as_ref().map(|_| {
                Param::from_tensor(
                    Tensor::stack::<2>(
                        experts
                            .iter()
                            .map(|expert| {
                                select(expert)
                                    .bias
                                    .as_ref()
                                    .expect("all MoE experts have the same bias setting")
                                    .val()
                            })
                            .collect(),
                        0,
                    )
                    .detach(),
                )
            })
        };
        Self {
            in_weight: Param::from_tensor(
                Tensor::stack::<3>(
                    experts
                        .iter()
                        .map(|expert| expert.in_proj.weight.val())
                        .collect(),
                    0,
                )
                .detach(),
            ),
            in_bias: stack_bias(|expert| &expert.in_proj),
            down_weight: Param::from_tensor(
                Tensor::stack::<3>(
                    experts
                        .iter()
                        .map(|expert| expert.down_proj.weight.val())
                        .collect(),
                    0,
                )
                .detach(),
            ),
            down_bias: stack_bias(|expert| &expert.down_proj),
            dropout: DropoutConfig::new(block.ffn.dropout).init(),
            activation: block.ffn.activation,
            intermediate: block.intermediate_size(config.hidden_size),
            gated: block.ffn.gate,
            expert_count,
        }
    }

    fn forward_batched(&self, routed_input: Tensor<2>, counts: &[usize]) -> Tensor<2> {
        let [routes, hidden] = routed_input.dims();
        let capacity = counts.iter().copied().max().unwrap_or(0);
        debug_assert!(capacity > 0);
        debug_assert_eq!(counts.iter().sum::<usize>(), routes);

        let mut offset = 0;
        let mut padded = Vec::with_capacity(self.expert_count);
        for &count in counts {
            let input = if count == 0 {
                Tensor::<2>::zeros([capacity, hidden], &routed_input.device())
                    .cast(routed_input.dtype())
            } else {
                let input = routed_input
                    .clone()
                    .slice([offset..offset + count, 0..hidden]);
                if count == capacity {
                    input
                } else {
                    Tensor::cat(
                        vec![
                            input,
                            Tensor::<2>::zeros([capacity - count, hidden], &routed_input.device())
                                .cast(routed_input.dtype()),
                        ],
                        0,
                    )
                }
            };
            padded.push(input);
            offset += count;
        }
        debug_assert_eq!(offset, routes);

        let projected = batched_linear_low_precision(
            Tensor::stack::<3>(padded, 0),
            self.in_weight.val(),
            self.in_bias.as_ref().map(Param::val),
        );
        let hidden_values = if self.gated {
            let gate = activate(
                self.activation,
                projected
                    .clone()
                    .slice([0..self.expert_count, 0..capacity, 0..self.intermediate]),
            );
            let values = projected.slice([
                0..self.expert_count,
                0..capacity,
                self.intermediate..2 * self.intermediate,
            ]);
            gate * values
        } else {
            activate(self.activation, projected)
        };
        let output = batched_linear_low_precision(
            self.dropout.forward(hidden_values),
            self.down_weight.val(),
            self.down_bias.as_ref().map(Param::val),
        );

        let compact = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(expert, &count)| {
                output
                    .clone()
                    .slice([expert..expert + 1, 0..count, 0..hidden])
                    .squeeze_dim::<2>(0)
            })
            .collect();
        Tensor::cat(compact, 0)
    }

    #[cfg(feature = "cuda")]
    fn forward_grouped(&self, routed_input: Tensor<2>, counts: &[usize]) -> Tensor<2> {
        debug_assert!(self.in_bias.is_none() && self.down_bias.is_none());
        let projected = grouped_linear(routed_input, matmul_input(self.in_weight.val()), counts);
        let hidden_values =
            if self.gated && matches!(self.activation, Activation::SwiGLU | Activation::SiLU) {
                fused_swiglu(projected, self.intermediate)
            } else if self.gated {
                let [routes, _] = projected.dims();
                let gate = activate(
                    self.activation,
                    projected.clone().slice([0..routes, 0..self.intermediate]),
                );
                let values = projected.slice([0..routes, self.intermediate..2 * self.intermediate]);
                gate * values
            } else {
                activate(self.activation, projected)
            };
        grouped_linear(
            self.dropout.forward(hidden_values),
            matmul_input(self.down_weight.val()),
            counts,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_grouped_swiglu(&self, routed_input: Tensor<2>, counts: Tensor<1, Int>) -> Tensor<2> {
        debug_assert!(self.in_bias.is_none() && self.down_bias.is_none());
        debug_assert!(self.gated);
        debug_assert!(matches!(
            self.activation,
            Activation::SwiGLU | Activation::SiLU
        ));
        debug_assert_eq!(self.dropout.prob, 0.0);
        grouped_swiglu_mlp(
            routed_input,
            matmul_input(self.in_weight.val()),
            matmul_input(self.down_weight.val()),
            counts,
            self.intermediate,
        )
    }

    fn forward_compact(&self, routed_input: Tensor<2>, counts: &[usize]) -> Tensor<2> {
        let [routes, hidden] = routed_input.dims();
        let mut outputs = Vec::with_capacity(self.expert_count);
        let mut offset = 0;
        for (expert, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let expert_input = routed_input
                .clone()
                .slice([offset..offset + count, 0..hidden]);
            outputs.push(self.forward_expert(expert_input, expert));
            offset += count;
        }
        debug_assert_eq!(offset, routes);
        Tensor::cat(outputs, 0)
    }

    fn forward_routed(
        &self,
        routed_input: Tensor<2>,
        counts: &[usize],
        use_batched: bool,
    ) -> Tensor<2> {
        #[cfg(feature = "cuda")]
        if is_cuda_device(&routed_input.device())
            && self.in_bias.is_none()
            && self.down_bias.is_none()
        {
            return self.forward_grouped(routed_input, counts);
        }

        if use_batched {
            self.forward_batched(routed_input, counts)
        } else {
            self.forward_compact(routed_input, counts)
        }
    }

    fn forward_expert(&self, input: Tensor<2>, expert: usize) -> Tensor<2> {
        let [tokens, _] = input.dims();
        let in_weight = self.in_weight.val();
        let [_, in_rows, in_columns] = in_weight.dims();
        let projected = batched_linear_low_precision(
            input.unsqueeze_dim::<3>(0),
            in_weight.slice([expert..expert + 1, 0..in_rows, 0..in_columns]),
            self.in_bias.as_ref().map(|bias| {
                let values = bias.val();
                let [_, columns] = values.dims();
                values.slice([expert..expert + 1, 0..columns])
            }),
        );
        let hidden_values = if self.gated {
            let gate = activate(
                self.activation,
                projected
                    .clone()
                    .slice([0..1, 0..tokens, 0..self.intermediate]),
            );
            let values =
                projected.slice([0..1, 0..tokens, self.intermediate..2 * self.intermediate]);
            gate * values
        } else {
            activate(self.activation, projected)
        };
        let down_weight = self.down_weight.val();
        let [_, down_rows, down_columns] = down_weight.dims();
        batched_linear_low_precision(
            self.dropout.forward(hidden_values),
            down_weight.slice([expert..expert + 1, 0..down_rows, 0..down_columns]),
            self.down_bias.as_ref().map(|bias| {
                let values = bias.val();
                let [_, columns] = values.dims();
                values.slice([expert..expert + 1, 0..columns])
            }),
        )
        .squeeze_dim::<2>(0)
    }

    fn prepare_inference(&mut self) {
        self.in_weight = self.in_weight.clone().map(super::matmul::matmul_input);
        self.in_bias = self
            .in_bias
            .take()
            .map(|bias| bias.map(super::matmul::matmul_input));
        self.down_weight = self.down_weight.clone().map(super::matmul::matmul_input);
        self.down_bias = self
            .down_bias
            .take()
            .map(|bias| bias.map(super::matmul::matmul_input));
    }
}

/// Dropless token-choice routing with top-k gate renormalization.
///
/// Routed tokens are sorted by expert and each compact group is evaluated by
/// its expert. An inverse permutation restores the original route order before
/// gate weighting and reduction. This is dropless; balanced routing can use a
/// bounded padded batch to reduce expert GEMM launches.
#[derive(Module, Debug)]
struct SparseMoe {
    router: Linear,
    experts: ExpertBank,
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
            experts: ExpertBank::new(config, block, moe.experts, device),
            shared_experts: (0..moe.shared_experts).map(|_| make_expert()).collect(),
            top_k: moe.top_k,
            load_balance_loss_weight: moe.load_balance_loss_weight,
            router_z_loss_weight: moe.router_z_loss_weight,
        }
    }

    fn forward<const D: usize>(
        &self,
        x: Tensor<D>,
        collect_auxiliary: bool,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        let shape = x.dims();
        let hidden = shape[D - 1];
        let tokens = shape[..D - 1].iter().product::<usize>();
        let flat = x.reshape([tokens, hidden]);

        // Router math remains FP32 even when the residual/expert stream is BF16.
        let logits = self.router.forward(flat.clone().cast(DType::F32));
        // Renormalized top-k softmax is exactly softmax over the selected
        // logits. Avoid materializing the full probability matrix unless the
        // load-balancing objective actually consumes it.
        let (top_logits, top_indices) = if self.top_k == 2 {
            let indices = top2_indices(logits.clone());
            (logits.clone().gather(1, indices.clone()), indices)
        } else {
            logits.clone().topk_with_indices(self.top_k, 1)
        };
        let top_weights = softmax(top_logits, 1);
        let expert_count = self.experts.expert_count;
        let routes = tokens
            .checked_mul(self.top_k)
            .expect("MoE route count overflow");
        let device = flat.device();

        // Grouped GEMMs need host-visible row counts, but the full route plan
        // stays on CUDA. Only `expert_count` integers cross to the host.
        #[cfg(feature = "cuda")]
        let (route_order, inverse_order, mut counts, device_counts) = if is_cuda_device(&device) {
            let (order, inverse, counts) = route_plan(top_indices.clone(), expert_count);
            (order, inverse, Vec::new(), Some(counts))
        } else {
            let (order, inverse, counts) =
                host_route_plan(top_indices.clone(), expert_count, routes, &device);
            (order, inverse, counts, None)
        };
        #[cfg(not(feature = "cuda"))]
        let (route_order, inverse_order, counts) =
            host_route_plan(top_indices.clone(), expert_count, routes, &device);
        #[cfg(feature = "cuda")]
        let routed_input = if is_cuda_device(&device) {
            route_gather(
                flat.clone(),
                route_order.clone(),
                inverse_order.clone(),
                self.top_k,
            )
        } else {
            let original_route_input = flat
                .clone()
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, self.top_k)
                .reshape([routes, hidden]);
            row_permute(
                original_route_input,
                route_order.clone(),
                inverse_order.clone(),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let routed_input = {
            let original_route_input = flat
                .clone()
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, self.top_k)
                .reshape([routes, hidden]);
            row_permute(
                original_route_input,
                route_order.clone(),
                inverse_order.clone(),
            )
        };
        #[cfg(feature = "cuda")]
        let use_fused_grouped = is_cuda_device(&device)
            && self.experts.in_bias.is_none()
            && self.experts.down_bias.is_none()
            && self.experts.gated
            && matches!(
                self.experts.activation,
                Activation::SwiGLU | Activation::SiLU
            )
            && self.experts.dropout.prob == 0.0;
        #[cfg(feature = "cuda")]
        if !use_fused_grouped && let Some(device_counts) = device_counts.clone() {
            // Register route gathering before the host read. The fusion
            // runtime can execute useful dispatch work while the tiny count
            // transfer completes, instead of rebuilding the next graph only
            // after the synchronization point.
            counts = device_counts
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .expect("MoE counts must be readable")
                .into_iter()
                .map(|count| count as usize)
                .collect::<Vec<_>>();
        }
        #[cfg(feature = "cuda")]
        debug_assert!(use_fused_grouped || counts.iter().sum::<usize>() == routes);
        #[cfg(not(feature = "cuda"))]
        debug_assert_eq!(counts.iter().sum::<usize>(), routes);
        // Balanced routes use two batched GEMMs instead of launching one pair
        // per expert. If a router becomes badly skewed, compact per-expert
        // execution avoids multiplying a mostly padded batch.
        #[cfg(feature = "cuda")]
        let routed_output = if use_fused_grouped {
            self.experts.forward_grouped_swiglu(
                routed_input,
                device_counts
                    .clone()
                    .expect("CUDA route planning provides device counts"),
            )
        } else {
            let capacity = counts.iter().copied().max().unwrap_or(0);
            let use_batched = collect_auxiliary
                && capacity.saturating_mul(expert_count) <= routes.saturating_mul(3) / 2;
            self.experts
                .forward_routed(routed_input, &counts, use_batched)
        };
        #[cfg(not(feature = "cuda"))]
        let routed_output = {
            let capacity = counts.iter().copied().max().unwrap_or(0);
            let use_batched = collect_auxiliary
                && capacity.saturating_mul(expert_count) <= routes.saturating_mul(3) / 2;
            self.experts
                .forward_routed(routed_input, &counts, use_batched)
        };
        #[cfg(feature = "cuda")]
        let mut output = if is_cuda_device(&device) {
            route_combine(routed_output, top_weights, inverse_order, self.top_k)
        } else {
            (row_permute(routed_output, inverse_order, route_order)
                .reshape([tokens, self.top_k, hidden])
                * top_weights
                    .cast(flat.dtype())
                    .reshape([tokens, self.top_k, 1]))
            .sum_dim(1)
            .reshape([tokens, hidden])
        };
        #[cfg(not(feature = "cuda"))]
        let mut output = (row_permute(routed_output, inverse_order, route_order)
            .reshape([tokens, self.top_k, hidden])
            * top_weights
                .cast(flat.dtype())
                .reshape([tokens, self.top_k, 1]))
        .sum_dim(1)
        .reshape([tokens, hidden]);
        for expert in &self.shared_experts {
            output = output + expert.forward(flat.clone());
        }

        let mut balance = None;
        if collect_auxiliary && self.load_balance_loss_weight != 0.0 {
            let probabilities = softmax(logits.clone(), 1);
            // CUDA consumes `device_counts`; CPU consumes `counts`, so there
            // is no single host iterator that covers both branches.
            #[allow(clippy::needless_range_loop, clippy::single_range_in_vec_init)]
            for expert_index in 0..expert_count {
                let probability_fraction = probabilities
                    .clone()
                    .slice([0..tokens, expert_index..expert_index + 1])
                    .mean();
                #[cfg(feature = "cuda")]
                let term = if let Some(device_counts) = &device_counts {
                    probability_fraction
                        * device_counts
                            .clone()
                            .float()
                            .slice([expert_index..expert_index + 1])
                            .div_scalar(routes as f64)
                } else {
                    probability_fraction.mul_scalar(counts[expert_index] as f64 / routes as f64)
                };
                #[cfg(not(feature = "cuda"))]
                let term =
                    probability_fraction.mul_scalar(counts[expert_index] as f64 / routes as f64);
                balance = Some(match balance {
                    Some(sum) => sum + term,
                    None => term,
                });
            }
        }
        let mut auxiliary = balance
            .map(|loss| loss.mul_scalar(self.load_balance_loss_weight * expert_count as f64));
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
        self.experts.prepare_inference();
        for expert in &mut self.shared_experts {
            expert.prepare_inference();
        }
    }
}

impl FeedForward {
    fn dense_view(&self) -> DenseFeedForwardView<'_> {
        DenseFeedForwardView {
            in_proj: self
                .in_proj
                .as_ref()
                .expect("dense FFN has an input projection"),
            down_proj: self
                .down_proj
                .as_ref()
                .expect("dense FFN has a down projection"),
            dropout: &self.dropout,
            activation: self.activation,
            intermediate: self.intermediate,
            gated: self.gated,
        }
    }

    pub fn new(config: &ModelDef, block: &BlockDef, device: &Device) -> Self {
        let dense = block
            .ffn
            .moe
            .is_none()
            .then(|| DenseFeedForward::new(config, block, device));
        Self {
            in_proj: dense.as_ref().map(|dense| dense.in_proj.clone()),
            down_proj: dense.as_ref().map(|dense| dense.down_proj.clone()),
            dropout: DropoutConfig::new(block.ffn.dropout).init(),
            activation: block.ffn.activation,
            intermediate: block.intermediate_size(config.hidden_size),
            gated: block.ffn.gate,
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

    /// Training forward pass including configured router regularization.
    pub fn forward_with_aux<const D: usize>(&self, x: Tensor<D>) -> (Tensor<D>, Option<Tensor<1>>) {
        self.forward_internal(x, true)
    }

    fn forward_internal<const D: usize>(
        &self,
        x: Tensor<D>,
        collect_auxiliary: bool,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        match &self.moe {
            Some(moe) => moe.forward(x, collect_auxiliary),
            None => (self.dense_view().forward(x), None),
        }
    }

    pub(crate) fn router_parameter_id(&self) -> Option<ParamId> {
        self.moe.as_ref().map(|moe| moe.router.weight.id)
    }

    pub(crate) fn prepare_inference(&mut self) {
        if let Some(in_proj) = &mut self.in_proj {
            prepare_linear_for_inference(in_proj);
        }
        if let Some(down_proj) = &mut self.down_proj {
            prepare_linear_for_inference(down_proj);
        }
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
    // The whole chain stays in the residual-stream dtype (BF16 during CUDA
    // training): the down projection feeds the residual add without an FP32
    // promotion.
    let projected = linear_low_precision(in_proj, x);
    let hidden = if gated {
        let mut ranges = projected.dims().map(|size| 0..size);
        ranges[D - 1] = 0..intermediate;
        let gate = activate(activation, projected.clone().slice(ranges.clone()));
        ranges[D - 1] = intermediate..2 * intermediate;
        gate * projected.slice(ranges)
    } else {
        activate(activation, projected)
    };
    linear_low_precision(down_proj, dropout.forward(hidden))
}

fn activate<const D: usize>(activation: Activation, tensor: Tensor<D>) -> Tensor<D> {
    match activation {
        Activation::SwiGLU | Activation::SiLU => silu(tensor),
        Activation::GELU => gelu(tensor),
        Activation::ReLU => relu(tensor),
        Activation::GELUNew | Activation::GELUTanh => gelu_approximate(tensor),
    }
}

fn host_route_plan(
    top_indices: Tensor<2, Int>,
    expert_count: usize,
    routes: usize,
    device: &Device,
) -> (Tensor<1, Int>, Tensor<1, Int>, Vec<usize>) {
    let route_experts = top_indices
        .reshape([routes])
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .expect("MoE routes must be readable");
    let mut assignments = vec![Vec::<i64>::new(); expert_count];
    for (position, expert) in route_experts.into_iter().enumerate() {
        assignments[expert as usize].push(position as i64);
    }
    let counts = assignments.iter().map(Vec::len).collect::<Vec<_>>();
    let route_order = assignments.iter().flatten().copied().collect::<Vec<_>>();
    let mut inverse_order = vec![0_i64; routes];
    for (sorted, &original) in route_order.iter().enumerate() {
        inverse_order[original as usize] = sorted as i64;
    }
    (
        Tensor::<1, Int>::from_data(TensorData::new(route_order, [routes]), device),
        Tensor::<1, Int>::from_data(TensorData::new(inverse_order, [routes]), device),
        counts,
    )
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, TensorData};

    use super::*;

    fn masked_reference(layer: &FeedForward, x: Tensor<2>) -> Tensor<2> {
        let moe = layer.moe.as_ref().expect("test layer is sparse");
        let expert_count = moe.experts.expert_count;
        let logits = moe.router.forward(x.clone().cast(DType::F32));
        let probabilities = softmax(logits, 1);
        let (top_weights, top_indices) = probabilities.topk_with_indices(moe.top_k, 1);
        let top_weights = top_weights.clone() / top_weights.sum_dim(1).clamp_min(1e-12);
        let mut output = Tensor::zeros_like(&x);
        for expert_index in 0..expert_count {
            let gate = (top_indices.clone().equal_elem(expert_index as i64).float()
                * top_weights.clone())
            .sum_dim(1)
            .cast(x.dtype());
            let expert_output = moe.experts.forward_expert(x.clone(), expert_index);
            output = output + expert_output * gate;
        }
        for expert in &moe.shared_experts {
            output = output + expert.forward(x.clone());
        }
        output
    }

    fn assert_sparse_dispatch_matches_masked_reference(device: Device) {
        let config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12
                moe { experts: 4 top_k: 2 shared_experts: 1 }
            }
            model sparse_test {
                vocab_size: 16
                hidden_size: 8
                num_layers: 1
                block: { attention: { num_heads: 1 } ffn: routed }
            }
            "#,
        )
        .unwrap();
        device.seed(11);
        let layer = FeedForward::new(&config, &config.block, &device);
        let values = (0..48)
            .map(|index| (index as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let input = super::super::matmul::stream_cast(Tensor::<2>::from_data(
            TensorData::new(values, [6, 8]),
            &device,
        ));
        let tolerance = if input.dtype() == DType::BF16 {
            1e-2
        } else {
            1e-5
        };

        let expected = masked_reference(&layer, input.clone())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let compact = layer
            .forward(input.clone())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let batched = layer
            .forward_with_aux(input)
            .0
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        for (name, actual) in [("compact", compact), ("batched", batched)] {
            let max_difference = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_difference < tolerance,
                "{name} max difference {max_difference}"
            );
        }
    }

    #[test]
    fn sparse_dispatch_matches_masked_reference() {
        assert_sparse_dispatch_matches_masked_reference(Device::ndarray());
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    #[test]
    fn gpu_sparse_dispatch_matches_masked_reference() {
        assert_sparse_dispatch_matches_masked_reference(crate::model::default_device());
    }

    #[test]
    fn sparse_expert_bank_receives_gradients() {
        let config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12
                moe {
                    experts: 4
                    top_k: 2
                    load_balance_loss_weight: 0.01
                    router_z_loss_weight: 0.001
                }
            }
            model sparse_test {
                vocab_size: 16
                hidden_size: 8
                num_layers: 1
                block: { attention: { num_heads: 1 } ffn: routed }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray().autodiff();
        device.seed(13);
        let layer = FeedForward::new(&config, &config.block, &device);
        let input = Tensor::<2>::random([6, 8], Distribution::Default, &device);
        let (output, auxiliary) = layer.forward_with_aux(input);
        let gradients = (output.square().mean() + auxiliary.unwrap()).backward();
        let experts = &layer.moe.as_ref().unwrap().experts;

        assert!(experts.in_weight.grad(&gradients).is_some());
        assert!(experts.down_weight.grad(&gradients).is_some());
    }
}
