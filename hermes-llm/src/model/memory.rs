//! Fast-to-slow continuum memory for sleep-capable models.
//!
//! A memory chain is opt-in through MAL and replaces a block's ordinary FFN
//! branch. Each tier remains a residual sublayer, so a zero-output slower tier
//! is an exact no-op. Low-rank reserve experts are allocated up front and are
//! routed only after their checkpointed activation bit is set. Until then
//! parameter-free zero columns occupy their fixed router lanes and an
//! untrainable rank-matched zero route occupies the fixed top-1 expert lane;
//! neither has checkpoint or optimizer state, and activation replaces them.

use anyhow::Result;
use burn::module::{ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, DType, Int, TensorData};
use burn_nn::{Linear, LinearConfig};

use crate::mal::{BlockDef, MemoryDef, MemoryTierInit, ModelDef};

use super::FeedForward;
use super::ffn::host_route_plan;
#[cfg(feature = "cuda")]
use super::grouped_linear::{grouped_linear, is_cuda_device};
use super::matmul::matmul_2;
#[cfg(feature = "cuda")]
use super::matmul::matmul_input;
#[cfg(feature = "cuda")]
use super::moe_dispatch::route_gather;
#[cfg(feature = "cuda")]
use super::moe_route::route_plan;
use super::row_permute::row_permute;

/// Expert-routing behavior for ordinary wake execution and dream generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryRouting {
    #[default]
    Wake,
    /// Add one deterministic random expert to every MoE router. This must only
    /// be selected while generating synthetic dreams.
    Dream { seed: u64 },
}

impl MemoryRouting {
    pub(crate) fn salted(self, salt: u64) -> Self {
        match self {
            Self::Wake => Self::Wake,
            Self::Dream { seed } => Self::Dream {
                seed: mix_seed(seed, salt),
            },
        }
    }

    fn dream_seed(self) -> Option<u64> {
        match self {
            Self::Wake => None,
            Self::Dream { seed } => Some(seed),
        }
    }
}

/// Select exactly one dream-only expert per token from outside that token's
/// ordinary top-k set.  The host read is deliberate: this path is used only
/// while generating dreams, and keeping the exclusion algorithm here avoids
/// adding synchronization or selection kernels to wake execution.
pub(crate) fn dream_extra_indices(
    routing: MemoryRouting,
    top_indices: Tensor<2, Int>,
    candidate_count: usize,
) -> Option<Tensor<2, Int>> {
    let seed = routing.dream_seed()?;
    let [tokens, selected_count] = top_indices.dims();
    assert!(
        candidate_count > selected_count,
        "dream routing requires at least one expert outside ordinary top-k ({candidate_count} experts, top-k {selected_count})"
    );
    if tokens == 0 {
        return None;
    }

    let device = top_indices.device();
    let selected = top_indices
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .expect("dream top-k indices must be readable");
    let mut extras = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let ordinary = &selected[token * selected_count..(token + 1) * selected_count];
        debug_assert!(
            ordinary
                .iter()
                .all(|&expert| expert >= 0 && (expert as usize) < candidate_count)
        );
        debug_assert!((0..ordinary.len()).all(|left| {
            ordinary[left + 1..]
                .iter()
                .all(|right| ordinary[left] != *right)
        }));

        let available = candidate_count - ordinary.len();
        let ordinal = mix_seed(seed, token as u64) as usize % available;
        let expert = (0..candidate_count)
            .filter(|candidate| !ordinary.contains(&(*candidate as i64)))
            .nth(ordinal)
            .expect("top-k complement must contain a dream expert");
        extras.push(expert as i64);
    }

    Some(Tensor::from_data(
        TensorData::new(extras, [tokens, 1]),
        &device,
    ))
}

fn mix_seed(mut value: u64, salt: u64) -> u64 {
    value ^= salt.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Checkpointed state and trainable parameter IDs for one reserve slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySlotStatus {
    pub layer: usize,
    pub tier: usize,
    pub tier_name: String,
    pub slot: usize,
    pub active: bool,
    pub generation: u64,
    pub parameter_ids: Vec<ParamId>,
}

#[derive(Module, Debug)]
struct LowRankExpertSlot {
    router: Linear,
    a: Param<Tensor<2>>,
    b: Param<Tensor<2>>,
    // Bool/int parameters are serialized by Burn but never visited by a float
    // optimizer. The runtime mirror avoids a host synchronization per token.
    active: Param<Tensor<1, Bool>>,
    generation: Param<Tensor<1, Int>>,
    #[module(skip)]
    runtime_active: bool,
}

impl LowRankExpertSlot {
    fn new(hidden: usize, rank: usize, slot: usize, device: &Device) -> Self {
        let a_values = deterministic_values(hidden * rank, mix_seed(slot as u64, 0x41));
        Self {
            router: LinearConfig::new(hidden, 1).with_bias(false).init(device),
            a: Param::from_tensor(Tensor::from_data(
                TensorData::new(a_values, [hidden, rank]),
                device,
            )),
            // A zero B is both a strict residual no-op and the standard LoRA
            // initialization: B receives gradients on the first update.
            b: Param::from_tensor(Tensor::zeros([rank, hidden], device)),
            active: Param::initialized(
                ParamId::new(),
                Tensor::<1, Bool>::from_data([false], device),
            ),
            generation: Param::initialized(
                ParamId::new(),
                Tensor::<1, Int>::from_data([0_i64], device),
            ),
            runtime_active: false,
        }
    }

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let dtype = input.dtype();
        matmul_2(matmul_2(input.cast(DType::F32), self.a.val()), self.b.val()).cast(dtype)
    }

    fn generation(&self) -> u64 {
        self.generation
            .val()
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("memory generation must be readable")[0] as u64
    }

    fn checkpoint_active(&self) -> bool {
        self.active
            .val()
            .into_data()
            .to_vec::<bool>()
            .expect("memory activation mask must be readable")[0]
    }

    fn set_active(&mut self, active: bool) {
        self.active = self.active.clone().map(|value| {
            let device = value.device();
            Tensor::<1, Bool>::from_data([active], &device)
        });
        self.runtime_active = active;
    }

    fn advance_generation(&mut self) {
        let generation = self.generation().saturating_add(1) as i64;
        self.generation = self.generation.clone().map(|value| {
            let device = value.device();
            Tensor::<1, Int>::from_data([generation], &device)
        });
    }

    fn activate(&mut self) {
        self.advance_generation();
        self.set_active(true);
    }

    fn sync_state(&mut self) {
        self.runtime_active = self.checkpoint_active();
    }

    fn reset(&mut self, seed: u64) {
        let [hidden, rank] = self.a.shape().dims();
        let a_values = deterministic_values(hidden * rank, seed);
        self.a = self.a.clone().map(|value| {
            let device = value.device();
            Tensor::from_data(TensorData::new(a_values, [hidden, rank]), &device)
        });
        self.b = self.b.clone().map(|value| Tensor::zeros_like(&value));
        self.router.weight = self
            .router
            .weight
            .clone()
            .map(|value| Tensor::zeros_like(&value));
        self.advance_generation();
        self.set_active(false);
    }

    fn parameter_ids(&self) -> Vec<ParamId> {
        vec![self.router.weight.id, self.a.id, self.b.id]
    }
}

fn deterministic_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1_u32 << 24) as f32;
            (unit * 2.0 - 1.0) * 0.02
        })
        .collect()
}

#[derive(Module, Debug)]
struct LowRankExpertReserve {
    slots: Vec<LowRankExpertSlot>,
    // An untrainable, deterministic zero-output route occupies the reserve
    // route lane until the first real slot activates.  It is deliberately not
    // a `Param`: it cannot acquire gradients, optimizer moments, checkpoint
    // state, or a routing row, and it is not one of the MAL reserve slots.
    // Executing the same rank-shaped pair of matmuls makes wake top-k compute
    // constant from the upgraded checkpoint through every activation cycle.
    fallback_a: Tensor<2>,
    fallback_b: Tensor<2>,
    // One parameter-free zero output column per stored slot keeps the full router
    // projection width fixed. An active slot replaces its corresponding row;
    // a dormant slot's actual router parameter stays entirely off the graph.
    fallback_router: Tensor<2>,
    #[module(skip)]
    top_k: usize,
}

impl LowRankExpertReserve {
    fn new(hidden: usize, capacity: usize, rank: usize, top_k: usize, device: &Device) -> Self {
        Self {
            slots: (0..capacity)
                .map(|slot| LowRankExpertSlot::new(hidden, rank, slot, device))
                .collect(),
            fallback_a: Tensor::from_data(
                TensorData::new(
                    deterministic_values(hidden * rank, mix_seed(0, 0x4641_4c4c_4241_434b)),
                    [hidden, rank],
                ),
                device,
            ),
            fallback_b: Tensor::zeros([rank, hidden], device),
            fallback_router: Tensor::zeros([hidden, capacity], device),
            top_k,
        }
    }

    fn fallback_forward<const D: usize>(&self, input: Tensor<D>, gate: Tensor<2>) -> Tensor<D> {
        let shape = input.dims();
        let hidden = shape[D - 1];
        let tokens = shape[..D - 1].iter().product::<usize>();
        let flat = input.reshape([tokens, hidden]);
        let dtype = flat.dtype();
        (matmul_2(
            matmul_2(flat.cast(DType::F32), self.fallback_a.clone()),
            self.fallback_b.clone(),
        )
        .cast(dtype)
            * gate.cast(dtype))
        .reshape(shape)
    }

    fn forward<const D: usize>(&self, input: Tensor<D>) -> Option<Tensor<D>> {
        let active = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.runtime_active)
            .collect::<Vec<_>>();
        debug_assert_eq!(self.top_k, 1);
        let shape = input.dims();
        let hidden = shape[D - 1];
        let tokens = shape[..D - 1].iter().product::<usize>();
        let flat = input.reshape([tokens, hidden]);
        // Build a fixed-width router from either the active row or the
        // corresponding parameter-free fallback. The matmul and top-k shapes
        // therefore never grow as stored slots activate, while dormant slot
        // parameters remain absent from both the forward and backward graph.
        let router_weight = Tensor::cat(
            self.slots
                .iter()
                .enumerate()
                .map(|(slot, expert)| {
                    if expert.runtime_active {
                        expert.router.weight.val()
                    } else {
                        self.fallback_router
                            .clone()
                            .slice([0..hidden, slot..slot + 1])
                    }
                })
                .collect(),
            1,
        );
        let router_logits = matmul_2(flat.clone().cast(DType::F32), router_weight);
        let mut mask = vec![-1.0e30_f32; self.slots.len()];
        for (global, _) in &active {
            mask[*global] = 0.0;
        }
        let masked = router_logits.clone()
            + Tensor::<2>::from_data(TensorData::new(mask, [1, self.slots.len()]), &flat.device());
        let (_, global_indices) = masked.clone().topk_with_indices(1, 1);
        // Normalize against every active fixed-width row plus a parameter-free
        // zero fallback. Non-selected active rows therefore still receive the
        // same competitive router gradient as a conventional MoE, without
        // introducing any dormant parameter or activation-dependent shape.
        let gate = softmax(
            Tensor::cat(vec![masked, Tensor::zeros([tokens, 1], &flat.device())], 1),
            1,
        )
        .slice([0..tokens, 0..self.slots.len()])
        .gather(1, global_indices.clone());
        if active.is_empty() {
            // Actual reserve slots remain entirely off the graph.  The
            // separate fixed fallback executes one rank-shaped no-op route,
            // matching the top-1 expert cost paid after a slot activates.
            return Some(self.fallback_forward(flat.reshape(shape), gate));
        }

        let device = flat.device();
        // Inactive content buckets are masked before top-1. Convert the
        // resulting global slot number to the compact active-bank index used
        // by grouped dispatch. This selection is content-dependent without
        // scoring a growing bank of learned router rows.
        let mut global_to_local = vec![0_i64; self.slots.len()];
        for (local, (global, _)) in active.iter().enumerate() {
            global_to_local[*global] = local as i64;
        }
        let top_indices = Tensor::<1, Int>::from_data(
            TensorData::new(global_to_local, [self.slots.len()]),
            &device,
        )
        .select(0, global_indices.reshape([tokens]))
        .reshape([tokens, 1]);
        let routes = tokens;
        #[cfg(feature = "cuda")]
        let (route_order, inverse_order, mut counts, device_counts) = if is_cuda_device(&device) {
            let (order, inverse, counts) = route_plan(top_indices.clone(), active.len());
            (order, inverse, Vec::new(), Some(counts))
        } else {
            let (order, inverse, counts) =
                host_route_plan(top_indices.clone(), active.len(), routes, &device);
            (order, inverse, counts, None)
        };
        #[cfg(not(feature = "cuda"))]
        let (route_order, inverse_order, counts) =
            host_route_plan(top_indices, active.len(), routes, &device);

        #[cfg(feature = "cuda")]
        let routed_input = if is_cuda_device(&device) {
            route_gather(flat.clone(), route_order.clone(), inverse_order.clone(), 1)
        } else {
            let repeated = flat
                .clone()
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, 1)
                .reshape([routes, hidden]);
            row_permute(repeated, route_order.clone(), inverse_order.clone())
        };
        #[cfg(not(feature = "cuda"))]
        let routed_input = {
            let repeated = flat
                .clone()
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, 1)
                .reshape([routes, hidden]);
            row_permute(repeated, route_order.clone(), inverse_order.clone())
        };

        #[cfg(feature = "cuda")]
        if let Some(device_counts) = device_counts {
            // Keep the route plan on-device. Only the small per-slot count
            // vector crosses to the host to define grouped-GEMM descriptors.
            counts = device_counts
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .expect("memory route counts must be readable")
                .into_iter()
                .map(|count| count as usize)
                .collect();
        }

        #[cfg(feature = "cuda")]
        let routed_output = if is_cuda_device(&device) {
            let a = Tensor::stack::<3>(active.iter().map(|(_, slot)| slot.a.val()).collect(), 0);
            let b = Tensor::stack::<3>(active.iter().map(|(_, slot)| slot.b.val()).collect(), 0);
            let projected = grouped_linear(routed_input, matmul_input(a), &counts);
            grouped_linear(projected, matmul_input(b), &counts).cast(flat.dtype())
        } else {
            Self::forward_compact(&active, routed_input, &counts, hidden)
        };
        #[cfg(not(feature = "cuda"))]
        let routed_output = Self::forward_compact(&active, routed_input, &counts, hidden);

        let output =
            row_permute(routed_output, inverse_order, route_order) * gate.cast(flat.dtype());
        Some(output.reshape(shape))
    }

    fn forward_compact(
        active: &[(usize, &LowRankExpertSlot)],
        routed_input: Tensor<2>,
        counts: &[usize],
        hidden: usize,
    ) -> Tensor<2> {
        let mut offset = 0;
        let mut outputs = Vec::new();
        for (local, count) in counts.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }
            let input = routed_input
                .clone()
                .slice([offset..offset + count, 0..hidden]);
            outputs.push(active[local].1.forward(input));
            offset += count;
        }
        debug_assert_eq!(offset, routed_input.dims()[0]);
        Tensor::cat(outputs, 0)
    }

    fn sync_state(&mut self) {
        for slot in &mut self.slots {
            slot.sync_state();
        }
    }

    /// Stored reserve parameters and the fixed route paid by every ordinary
    /// wake token. The routed equivalent includes the full fixed-capacity
    /// router projection plus one low-rank expert. Before activation,
    /// parameter-free fallback tensors occupy exactly those lanes; activating
    /// slots replaces rather than adds learned work.
    fn wake_parameter_counts(&self) -> Result<(usize, usize)> {
        let stored = self.num_params();
        let first = self
            .slots
            .first()
            .ok_or_else(|| anyhow::anyhow!("memory reserve has no slots"))?;
        let router_lanes = first
            .router
            .num_params()
            .checked_mul(self.slots.len())
            .ok_or_else(|| anyhow::anyhow!("reserve router-lane count overflows usize"))?;
        let routed_expert = first
            .a
            .num_params()
            .checked_add(first.b.num_params())
            .ok_or_else(|| anyhow::anyhow!("reserve routed parameter count overflows usize"))?;
        let routed = routed_expert
            .checked_mul(self.top_k)
            .and_then(|value| value.checked_add(router_lanes))
            .ok_or_else(|| anyhow::anyhow!("reserve top-k parameter count overflows usize"))?;
        anyhow::ensure!(
            routed <= stored,
            "reserve routed parameter count exceeds stored capacity"
        );
        Ok((stored, routed))
    }
}

#[derive(Module, Debug)]
struct MemoryTier {
    feed_forward: FeedForward,
    reserve: LowRankExpertReserve,
    #[module(skip)]
    name: String,
}

impl MemoryTier {
    fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
        collect_auxiliary: bool,
        routing: MemoryRouting,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        // Paper-style random exploration belongs to the persistent FFN MoE.
        // Reserve memory remains ordinary top-1 during Dreaming: the first
        // consolidation has only one active receiver, while dormant slots
        // must stay entirely outside routing and computation.
        let reserve = self.reserve.forward(input.clone());
        let (base, auxiliary) = self.feed_forward.forward_internal_with_routing(
            input.clone(),
            collect_auxiliary,
            routing,
        );
        (
            match reserve {
                Some(reserve) => base + reserve,
                None => base,
            },
            auxiliary,
        )
    }

    fn wake_parameter_counts(&self) -> Result<(usize, usize)> {
        let (base_stored, base_routed) = self.feed_forward.wake_parameter_counts();
        let (reserve_stored, reserve_routed) = self.reserve.wake_parameter_counts()?;
        Ok((
            base_stored
                .checked_add(reserve_stored)
                .ok_or_else(|| anyhow::anyhow!("memory tier stored parameter count overflows"))?,
            base_routed
                .checked_add(reserve_routed)
                .ok_or_else(|| anyhow::anyhow!("memory tier routed parameter count overflows"))?,
        ))
    }
}

/// Ordered residual chain of fast-to-slow FFN/MoE tiers.
#[derive(Module, Debug)]
pub struct MemoryChain {
    tiers: Vec<MemoryTier>,
}

impl MemoryChain {
    pub(crate) fn new(
        config: &ModelDef,
        block: &BlockDef,
        definition: &MemoryDef,
        device: &Device,
    ) -> Self {
        let tiers = definition
            .tiers
            .iter()
            .map(|definition| {
                let mut tier_block = block.clone();
                tier_block.memory = None;
                tier_block.ffn = definition.ffn.clone();
                let mut feed_forward = FeedForward::new(config, &tier_block, device);
                if matches!(definition.residual_init, MemoryTierInit::ResidualZero) {
                    feed_forward.zero_output();
                }
                MemoryTier {
                    feed_forward,
                    reserve: LowRankExpertReserve::new(
                        config.hidden_size,
                        definition.reserve_experts.capacity,
                        definition.reserve_experts.rank,
                        definition.reserve_experts.top_k,
                        device,
                    ),
                    name: definition.name.clone(),
                }
            })
            .collect();
        Self { tiers }
    }

    pub(crate) fn forward_with<const D: usize>(
        &self,
        mut state: Tensor<D>,
        collect_auxiliary: bool,
        routing: MemoryRouting,
        mut normalize: impl FnMut(Tensor<D>) -> Tensor<D>,
        mut residual: impl FnMut(Tensor<D>, Tensor<D>) -> Tensor<D>,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        let mut auxiliary = None;
        for (index, tier) in self.tiers.iter().enumerate() {
            let (branch, tier_auxiliary) = tier.forward(
                normalize(state.clone()),
                collect_auxiliary,
                routing.salted(index as u64),
            );
            state = residual(state, branch);
            if let Some(loss) = tier_auxiliary {
                auxiliary = Some(match auxiliary {
                    Some(total) => total + loss,
                    None => loss,
                });
            }
        }
        (state, auxiliary)
    }

    pub(crate) fn prepare_inference(&mut self) {
        self.sync_state();
        for tier in &mut self.tiers {
            tier.feed_forward.prepare_inference();
        }
    }

    pub(crate) fn sync_state(&mut self) {
        for tier in &mut self.tiers {
            tier.reserve.sync_state();
        }
    }

    pub(crate) fn wake_parameter_counts(&self) -> Result<(usize, usize)> {
        self.tiers
            .iter()
            .try_fold((0_usize, 0_usize), |(stored, routed), tier| {
                let (tier_stored, tier_routed) = tier.wake_parameter_counts()?;
                Ok((
                    stored.checked_add(tier_stored).ok_or_else(|| {
                        anyhow::anyhow!("memory chain stored parameter count overflows")
                    })?,
                    routed.checked_add(tier_routed).ok_or_else(|| {
                        anyhow::anyhow!("memory chain routed parameter count overflows")
                    })?,
                ))
            })
    }

    pub(crate) fn prepare_upgrade_state(&mut self) {
        for (index, tier) in self.tiers.iter_mut().enumerate() {
            if index > 0 {
                tier.feed_forward.zero_output();
            }
            for slot in &mut tier.reserve.slots {
                slot.set_active(false);
            }
        }
    }

    pub(crate) fn activate_slot(&mut self, tier: usize, slot: usize) -> Result<()> {
        let tier_index = tier;
        let slot_index = slot;
        let tier = self
            .tiers
            .get_mut(tier_index)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier_index} does not exist"))?;
        let slot = tier
            .reserve
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| anyhow::anyhow!("memory slot {slot_index} does not exist"))?;
        if slot.runtime_active {
            anyhow::bail!("memory slot {slot_index} in tier {tier_index} is already active");
        }
        slot.activate();
        Ok(())
    }

    pub(crate) fn deactivate_slot(&mut self, tier: usize, slot: usize) -> Result<()> {
        let tier = self
            .tiers
            .get_mut(tier)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier} does not exist"))?;
        let slot = tier
            .reserve
            .slots
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("memory slot {slot} does not exist"))?;
        slot.set_active(false);
        Ok(())
    }

    pub(crate) fn reset_slot(&mut self, tier: usize, slot: usize, seed: u64) -> Result<()> {
        let tier = self
            .tiers
            .get_mut(tier)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier} does not exist"))?;
        let slot = tier
            .reserve
            .slots
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("memory slot {slot} does not exist"))?;
        slot.reset(seed);
        Ok(())
    }

    pub(crate) fn statuses(&self, layer: usize) -> Vec<MemorySlotStatus> {
        self.tiers
            .iter()
            .enumerate()
            .flat_map(|(tier_index, tier)| {
                tier.reserve
                    .slots
                    .iter()
                    .enumerate()
                    .map(move |(slot_index, slot)| MemorySlotStatus {
                        layer,
                        tier: tier_index,
                        tier_name: tier.name.clone(),
                        slot: slot_index,
                        active: slot.runtime_active,
                        generation: slot.generation(),
                        parameter_ids: slot.parameter_ids(),
                    })
            })
            .collect()
    }

    pub(crate) fn tier_parameter_ids(&self, tier: usize) -> Result<Vec<ParamId>> {
        let tier = self
            .tiers
            .get(tier)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier} does not exist"))?;
        #[derive(Default)]
        struct FloatIds(Vec<ParamId>);
        impl ModuleVisitor for FloatIds {
            fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
                self.0.push(parameter.id);
            }
        }
        let mut ids = FloatIds::default();
        tier.visit(&mut ids);
        Ok(ids.0)
    }

    /// Float parameters of the tier's persistent base FFN/MoE, excluding all
    /// preallocated reserve experts. Terminal consolidation targets exactly
    /// this scope before it can reclaim the bounded slow reserve.
    pub(crate) fn tier_base_parameter_ids(&self, tier: usize) -> Result<Vec<ParamId>> {
        let tier = self
            .tiers
            .get(tier)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier} does not exist"))?;
        #[derive(Default)]
        struct FloatIds(Vec<ParamId>);
        impl ModuleVisitor for FloatIds {
            fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
                self.0.push(parameter.id);
            }
        }
        let mut ids = FloatIds::default();
        tier.feed_forward.visit(&mut ids);
        Ok(ids.0)
    }

    pub(crate) fn non_muon_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids = Vec::new();
        for tier in &self.tiers {
            ids.extend(tier.feed_forward.router_parameter_id());
            // Reserve factors stay out of the always-on wake Muon group. The
            // sleep optimizer selects an active slot by its explicit IDs.
            ids.extend(
                tier.reserve
                    .slots
                    .iter()
                    .flat_map(LowRankExpertSlot::parameter_ids),
            );
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;

    fn memory_config() -> ModelDef {
        crate::mal::parse_mal(
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
        .unwrap()
    }

    #[test]
    fn dream_exploration_selects_one_distinct_expert_per_token() {
        let device = Device::ndarray();
        let ordinary = vec![0_i64, 2, 1, 3, 2, 4, 0, 4];
        let top_indices =
            Tensor::<2, Int>::from_data(TensorData::new(ordinary.clone(), [4, 2]), &device);

        assert!(
            dream_extra_indices(MemoryRouting::Wake, top_indices.clone(), 5).is_none(),
            "wake routing must never add an exploration expert"
        );
        let first = dream_extra_indices(
            MemoryRouting::Dream { seed: 0xfeed_beef },
            top_indices.clone(),
            5,
        )
        .expect("capacity outside top-k permits exploration");
        let second =
            dream_extra_indices(MemoryRouting::Dream { seed: 0xfeed_beef }, top_indices, 5)
                .expect("same seed remains valid");
        assert_eq!(first.dims(), [4, 1]);
        let extras = first.into_data().convert::<i64>().to_vec::<i64>().unwrap();
        assert_eq!(
            extras,
            second.into_data().convert::<i64>().to_vec::<i64>().unwrap(),
            "checkpointed RNG seeds must reproduce dream routes"
        );
        for (token, extra) in extras.into_iter().enumerate() {
            assert!(
                !ordinary[token * 2..token * 2 + 2].contains(&extra),
                "token {token} explored ordinary top-k expert {extra}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "dream routing requires at least one expert outside ordinary top-k")]
    fn dream_exploration_fails_when_top_k_fills_expert_capacity() {
        let device = Device::ndarray();
        let top_indices = Tensor::<2, Int>::from_data([[0_i64, 1]], &device);
        let _ = dream_extra_indices(MemoryRouting::Dream { seed: 7 }, top_indices, 2);
    }

    #[test]
    fn dormant_slots_have_no_forward_or_backward_path() {
        let config = memory_config();
        let device = Device::ndarray().autodiff();
        let block = config.block.clone();
        let mut memory = MemoryChain::new(&config, &block, block.memory.as_ref().unwrap(), &device);
        let run = |memory: &MemoryChain| {
            let input = Tensor::<2>::random([5, 8], Distribution::Default, &device);
            memory
                .forward_with(
                    input,
                    false,
                    MemoryRouting::Wake,
                    |state| state,
                    |state, branch| state + branch,
                )
                .0
                .square()
                .mean()
                .backward()
        };

        let dormant_gradients = run(&memory);
        for slot in &memory.tiers[0].reserve.slots {
            assert!(slot.a.grad(&dormant_gradients).is_none());
            assert!(slot.b.grad(&dormant_gradients).is_none());
            assert!(slot.router.weight.grad(&dormant_gradients).is_none());
        }

        memory.activate_slot(0, 0).unwrap();
        memory.tiers[0].reserve.slots[0].b =
            memory.tiers[0].reserve.slots[0].b.clone().map(|value| {
                let require_grad = value.is_require_grad();
                Tensor::full(value.shape(), 0.1, &value.device()).set_require_grad(require_grad)
            });
        let active_gradients = run(&memory);
        let active = &memory.tiers[0].reserve.slots[0];
        let dormant = &memory.tiers[0].reserve.slots[1];
        assert!(active.a.grad(&active_gradients).is_some());
        assert!(active.b.grad(&active_gradients).is_some());
        let first_router_gradient = active
            .router
            .weight
            .grad(&active_gradients)
            .expect("the fixed fallback logit must train the first active router")
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert!(first_router_gradient.iter().any(|value| value.abs() > 1e-8));
        assert!(dormant.a.grad(&active_gradients).is_none());
        assert!(dormant.b.grad(&active_gradients).is_none());
        assert!(dormant.router.weight.grad(&active_gradients).is_none());

        let error = memory.activate_slot(0, 0).unwrap_err().to_string();
        assert!(error.contains("already active"), "{error}");
        memory.activate_slot(0, 1).unwrap();
        memory.tiers[0].reserve.slots[1].b =
            memory.tiers[0].reserve.slots[1].b.clone().map(|value| {
                let require_grad = value.is_require_grad();
                Tensor::full(value.shape(), -0.1, &value.device()).set_require_grad(require_grad)
            });
        let routed_gradients = run(&memory);
        for slot in &memory.tiers[0].reserve.slots {
            let gradient = slot
                .router
                .weight
                .grad(&routed_gradients)
                .expect("every active router row participates in full-probability gating")
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            assert!(gradient.iter().any(|value| value.abs() > 1e-8));
        }
    }

    #[test]
    fn dormant_reserve_executes_an_exact_untrainable_fallback_route() {
        let config = memory_config();
        let device = Device::ndarray().autodiff();
        let block = config.block.clone();
        let memory = MemoryChain::new(&config, &block, block.memory.as_ref().unwrap(), &device);
        let reserve = &memory.tiers[0].reserve;
        let input = Tensor::<2>::random([5, 8], Distribution::Default, &device);

        let output = reserve
            .forward(input)
            .expect("the fixed fallback always occupies the top-1 route lane");
        let magnitude: f32 = output.abs().sum().into_scalar();
        assert_eq!(magnitude, 0.0, "fallback must be logit-exact no-op");

        let visited = memory.tier_parameter_ids(0).unwrap();
        let slot_ids = memory
            .statuses(0)
            .into_iter()
            .flat_map(|status| status.parameter_ids)
            .collect::<Vec<_>>();
        assert!(slot_ids.iter().all(|id| visited.contains(id)));
        // Plain Burn tensors move with the module but are not Params.  This
        // count proves the fallback did not enter optimizer/checkpoint scope.
        let base_ids = memory.tier_base_parameter_ids(0).unwrap();
        assert_eq!(visited.len(), base_ids.len() + slot_ids.len());
    }

    #[test]
    fn fixed_width_router_accounting_is_constant_through_full_activation() {
        let config = memory_config();
        let device = Device::ndarray();
        let block = config.block.clone();
        let mut memory = MemoryChain::new(&config, &block, block.memory.as_ref().unwrap(), &device);
        let reserve = &memory.tiers[0].reserve;
        let first = &reserve.slots[0];
        let expected = first.router.num_params() * reserve.slots.len()
            + first.a.num_params()
            + first.b.num_params();
        let initial = reserve.wake_parameter_counts().unwrap();
        assert_eq!(
            initial.1, expected,
            "all fixed router lanes must be counted"
        );

        for slot in 0..reserve.slots.len() {
            memory.activate_slot(0, slot).unwrap();
            assert_eq!(
                memory.tiers[0].reserve.wake_parameter_counts().unwrap(),
                initial,
                "activation generation {slot} changed routed compute accounting"
            );
        }
    }

    #[test]
    fn reset_is_dormant_noop_and_advances_generation() {
        let config = memory_config();
        let device = Device::ndarray();
        let block = config.block.clone();
        let mut memory = MemoryChain::new(&config, &block, block.memory.as_ref().unwrap(), &device);
        memory.activate_slot(0, 0).unwrap();
        assert!(memory.statuses(0)[0].active);
        memory.reset_slot(0, 0, 17).unwrap();
        let status = &memory.statuses(0)[0];
        assert!(!status.active);
        assert_eq!(status.generation, 2);
        let b: f32 = memory.tiers[0].reserve.slots[0]
            .b
            .val()
            .abs()
            .sum()
            .into_scalar();
        assert_eq!(b, 0.0);
    }
}
