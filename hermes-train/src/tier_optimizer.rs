//! Durable, independently-clocked optimizer state for MAL memory tiers.
//!
//! This module is the concrete wake-side counterpart to `tensor_sleep`: it
//! partitions one backward graph into the ordinary wake scope and active
//! memory-tier scopes, holds a separate AdamW optimizer and long-lived gradient
//! accumulator for every tier, previews sender-only updates for consolidation,
//! and publishes content-addressed optimizer/candidate generations.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(test)]
use std::fs::OpenOptions;
#[cfg(test)]
use std::io::Write;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param, ParamId, list_param_ids};
use burn::tensor::{Device, Gradients, Tensor, TensorData};
use burn_optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn_pack::{Bytes, Reader, Tensor as PackedTensor, Writer};
use hermes_llm::{ModelDef, Transformer, load_safetensors_bytes, save_safetensors};
use serde::{Deserialize, Serialize};

use crate::artifact_io::{
    AuthenticatedDirectorySnapshot, atomic_write_new, ensure_directory, ensure_real_directory,
    hash_regular_file as hash_file, read_regular_bounded, sha256_identity, sync_directory,
    sync_regular_file, write_new_synced,
};
use crate::native_sleep::{
    PlannedConsolidation, TierOptimizerCommit, TierOptimizerCommitRole, TierOptimizerPublisher,
};
use crate::optimizer_artifact::canonical_module_optimizer_bytes;
use crate::sleep::{ConsolidationTxn, MemoryOptimizerScopes, SleepSchedule, TierOptimizerArtifact};
use crate::tensor_sleep::{
    AtomicCandidatePublisher, ImmutableTransformerCheckpoint, ProspectiveTransformerCandidate,
    ProspectiveTransformerUpdate, ProspectiveUpdateSnapshot, TensorTransactionPointer,
    TensorTransactionStore, prospective_update_hash,
};

const SNAPSHOT_MAGIC: &[u8; 8] = b"HTIER2\0\0";
const SNAPSHOT_VERSION: u32 = 2;
const TIER_BUNDLE_VERSION: u32 = 1;
const CANDIDATE_STORE_VERSION: u32 = 1;
const OPTIMIZER_FILE: &str = "optimizer.bpk";
const GRADIENT_FILE: &str = "gradients.bpk";
const STATE_FILE: &str = "state.json";
const MANIFEST_FILE: &str = "manifest.json";
const WEIGHTS_FILE: &str = "weights.safetensors";
const MAX_TIER_BUNDLE_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TIER_BUNDLE_STATE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TIER_BUNDLE_MEMBER_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const MAX_TIER_SNAPSHOT_HEADER_BYTES: usize = 64 * 1024 * 1024;
const TIER_BUNDLE_SCHEMA: [&str; 4] = [MANIFEST_FILE, OPTIMIZER_FILE, GRADIENT_FILE, STATE_FILE];
static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TierBundleLoadStage {
    AfterCapture,
    AfterStagedLoad,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TierOptimizerConfig {
    pub learning_rate: f64,
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for TierOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            beta_1: 0.9,
            beta_2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.1,
        }
    }
}

impl TierOptimizerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "tier learning rate must be finite and positive"
        );
        ensure!(
            self.beta_1.is_finite() && (0.0..1.0).contains(&self.beta_1),
            "tier AdamW beta_1 must be in [0, 1)"
        );
        ensure!(
            self.beta_2.is_finite() && (0.0..1.0).contains(&self.beta_2),
            "tier AdamW beta_2 must be in [0, 1)"
        );
        ensure!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "tier AdamW epsilon must be finite and positive"
        );
        ensure!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "tier AdamW weight decay must be finite and non-negative"
        );
        Ok(())
    }

    fn optimizer(&self) -> ModuleOptimizer {
        AdamWConfig::new()
            .with_beta_1(self.beta_1)
            .with_beta_2(self.beta_2)
            .with_epsilon(self.epsilon)
            .with_weight_decay(self.weight_decay)
            .init()
    }
}

struct TierRuntimeState {
    tier: usize,
    tier_id: String,
    parameter_ids: Vec<u64>,
    optimizer: ModuleOptimizer,
    accumulator: GradientsParams,
    accumulated_micro_steps: u64,
    update_clock: u64,
    transfer_clock: u64,
    generation: u64,
    transfer_generation: u64,
    artifact: Option<TierOptimizerArtifact>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct StagedUpdateMeta {
    txn_id: u64,
    sender: usize,
    trigger_clock: u64,
    student_uri: String,
    student_sha256: String,
    update_sha256: String,
    cleared_parameter_ids: Vec<u64>,
}

struct BankInner {
    config: TierOptimizerConfig,
    topology_sha256: String,
    update_periods: Vec<u64>,
    scopes: MemoryOptimizerScopes,
    model_layout: Transformer,
    tiers: Vec<TierRuntimeState>,
    staged: Option<StagedUpdateMeta>,
    cached_candidate: Option<Transformer>,
}

/// Shared state used by wake accumulation, the prospective-update adapter, and
/// the optimizer publisher. The mutex is an ownership boundary, not a device
/// synchronization primitive; all tensor work remains on the model device.
#[derive(Clone)]
pub struct TierOptimizerBank {
    inner: Arc<Mutex<BankInner>>,
    #[cfg(test)]
    snapshot_serializations: Arc<AtomicU64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierAccumulationReport {
    pub wake_gradient_tensors: usize,
    pub tier_gradient_tensors: Vec<usize>,
    pub accumulated_micro_steps: Vec<u64>,
}

/// One directly applied due tier update in the `wake_only` sleep ablation.
/// The independently scoped optimizer can touch only this tier; the receipt
/// binds the exact eligible before/after tensors and unchanged reserve state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WakeOnlyTierUpdate {
    pub tier: usize,
    pub tier_id: String,
    pub trigger_clock: u64,
    pub accumulated_optimizer_steps: u64,
    pub prospective_update_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WakeOnlyTierUpdateReport {
    pub trigger_clock: u64,
    pub updates: Vec<WakeOnlyTierUpdate>,
}

/// One backward pass partitioned into the ordinary wake scope and one gradient
/// set per memory tier.  The tier gradients are deliberately not committed to
/// the long-lived bank: the trainer must first finish its whole gradient-
/// accumulation window, average and clip all scopes together, and only then
/// call [`TierOptimizerBank::commit_tier_gradients`].
pub struct PartitionedTierGradients {
    pub wake: GradientsParams,
    pub tiers: Vec<GradientsParams>,
    pub report: TierAccumulationReport,
}

impl TierOptimizerBank {
    pub fn new(
        model: &Transformer,
        schedule: &SleepSchedule,
        config: TierOptimizerConfig,
    ) -> Result<Self> {
        config.validate()?;
        let scopes = MemoryOptimizerScopes::from_model(model, schedule)?;
        let update_periods = schedule
            .tiers
            .iter()
            .map(|tier| tier.update_period)
            .collect::<Vec<_>>();
        let topology_sha256 = optimizer_topology_hash(&scopes, &update_periods)?;
        let tiers = scopes
            .tiers
            .iter()
            .map(|scope| TierRuntimeState {
                tier: scope.tier,
                tier_id: scope.tier_id.clone(),
                parameter_ids: scope.parameter_ids.clone(),
                optimizer: config.optimizer(),
                accumulator: GradientsParams::new(),
                accumulated_micro_steps: 0,
                update_clock: 0,
                transfer_clock: 0,
                generation: 0,
                transfer_generation: 0,
                artifact: None,
            })
            .collect();
        Ok(Self {
            inner: Arc::new(Mutex::new(BankInner {
                config,
                topology_sha256,
                update_periods,
                scopes,
                model_layout: model.clone(),
                tiers,
                staged: None,
                cached_candidate: None,
            })),
            #[cfg(test)]
            snapshot_serializations: Arc::new(AtomicU64::new(0)),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, BankInner>> {
        self.inner
            .lock()
            .map_err(|_| anyhow::anyhow!("tier optimizer bank mutex is poisoned"))
    }

    /// Extract every active memory-tier gradient from one backward graph and
    /// return the exact non-memory wake scope without mutating optimizer state.
    /// Dormant reserve parameters are checked explicitly and must expose no
    /// gradient tensor.
    pub fn partition_gradients(
        &self,
        model: &Transformer,
        gradients: &mut Gradients,
    ) -> Result<PartitionedTierGradients> {
        let mut inner = self.lock()?;
        validate_model_topology(&inner, model)?;
        ensure!(
            inner.staged.is_none(),
            "cannot accumulate wake gradients while a tier update is staged"
        );

        // First partition and validate the complete backward graph. This keeps
        // the live accumulators unchanged when a tier is unexpectedly missing
        // gradients or a dormant reserve leaks into autograd.
        let mut selected_tiers = Vec::with_capacity(inner.tiers.len());
        for tier_index in 0..inner.tiers.len() {
            let active = active_tier_parameter_ids(model, tier_index)?;
            let selected = GradientsParams::from_params(gradients, model, &active);
            ensure!(
                !selected.is_empty(),
                "memory tier {tier_index} produced no wake gradients"
            );
            selected_tiers.push(selected);
        }

        let dormant = dormant_parameter_ids(model);
        let dormant_gradients = GradientsParams::from_params(gradients, model, &dormant);
        ensure!(
            dormant_gradients.is_empty(),
            "dormant memory reserve acquired {} gradient tensors",
            dormant_gradients.len()
        );
        let wake_ids = inner.scopes.wake_parameter_ids();
        let wake = GradientsParams::from_params(gradients, model, &wake_ids);
        ensure!(
            !wake.is_empty(),
            "ordinary wake scope produced no gradients"
        );

        let counts = selected_tiers.iter().map(GradientsParams::len).collect();
        inner.model_layout = model.clone();
        let report = TierAccumulationReport {
            wake_gradient_tensors: wake.len(),
            tier_gradient_tensors: counts,
            accumulated_micro_steps: inner
                .tiers
                .iter()
                .map(|tier| tier.accumulated_micro_steps)
                .collect(),
        };
        Ok(PartitionedTierGradients {
            wake,
            tiers: selected_tiers,
            report,
        })
    }

    /// Commit one already-averaged and already-clipped optimizer-step gradient
    /// set to the independently-clocked tier accumulators. Checkpoint v2 binds
    /// one accumulator generation to every optimizer step, so callers must not
    /// coalesce multiple optimizer steps into one state mutation. The operation
    /// validates every tier before changing any of them.
    pub fn commit_tier_gradients(
        &self,
        model: &Transformer,
        mut tier_gradients: Vec<GradientsParams>,
        optimizer_steps: u64,
    ) -> Result<TierAccumulationReport> {
        ensure!(
            optimizer_steps == 1,
            "tier optimizer checkpoints require exactly one optimizer step per accumulation commit"
        );
        let mut inner = self.lock()?;
        validate_model_topology(&inner, model)?;
        ensure!(
            inner.staged.is_none(),
            "cannot commit wake gradients while a tier update is staged"
        );
        ensure!(
            tier_gradients.len() == inner.tiers.len(),
            "tier gradient count {} differs from optimizer tier count {}",
            tier_gradients.len(),
            inner.tiers.len()
        );

        // Validate all input sets and counter arithmetic before mutating the
        // bank, so an incomplete/foreign window cannot partially enter it.
        let mut next = Vec::with_capacity(inner.tiers.len());
        let mut counts = Vec::with_capacity(inner.tiers.len());
        for (tier_index, selected) in tier_gradients.iter().enumerate() {
            ensure!(
                !selected.is_empty(),
                "memory tier {tier_index} has no completed-step gradients"
            );
            let active = active_tier_parameter_ids(model, tier_index)?
                .into_iter()
                .map(|id| id.val())
                .collect::<BTreeSet<_>>();
            let ids =
                gradient_parameter_ids(model, selected, &inner.tiers[tier_index].parameter_ids)?;
            ensure!(
                ids.iter().all(|id| active.contains(id)),
                "memory tier {tier_index} completed-step gradients include a dormant parameter"
            );
            counts.push(selected.len());
            next.push((
                inner.tiers[tier_index]
                    .accumulated_micro_steps
                    .checked_add(optimizer_steps)
                    .context("tier optimizer-step counter overflow")?,
                inner.tiers[tier_index]
                    .generation
                    .checked_add(1)
                    .context("tier accumulator generation overflow")?,
            ));
        }

        for (tier_index, mut selected) in tier_gradients.drain(..).enumerate() {
            let parameter_ids = inner.tiers[tier_index].parameter_ids.clone();
            let (next_steps, next_generation) = next[tier_index];
            accumulate_gradients(
                model,
                &parameter_ids,
                &mut inner.tiers[tier_index].accumulator,
                &mut selected,
            )?;
            inner.tiers[tier_index].accumulated_micro_steps = next_steps;
            inner.tiers[tier_index].generation = next_generation;
            inner.tiers[tier_index].artifact = None;
        }
        inner.model_layout = model.clone();
        Ok(TierAccumulationReport {
            wake_gradient_tensors: 0,
            tier_gradient_tensors: counts,
            accumulated_micro_steps: inner
                .tiers
                .iter()
                .map(|tier| tier.accumulated_micro_steps)
                .collect(),
        })
    }

    /// Convenience API for callers that already own one complete, normalized
    /// optimizer-step backward graph.
    pub fn partition_and_accumulate(
        &self,
        model: &Transformer,
        gradients: &mut Gradients,
        optimizer_steps: u64,
    ) -> Result<(GradientsParams, TierAccumulationReport)> {
        let partitioned = self.partition_gradients(model, gradients)?;
        let wake_count = partitioned.report.wake_gradient_tensors;
        let wake = partitioned.wake;
        let mut report = self.commit_tier_gradients(model, partitioned.tiers, optimizer_steps)?;
        report.wake_gradient_tensors = wake_count;
        Ok((wake, report))
    }

    pub fn snapshot_bytes(&self) -> Result<Vec<u8>> {
        let inner = self.lock()?;
        self.serialize_snapshot(&inner)
    }

    /// Restore only after fully decoding, authenticating, and reconstructing
    /// every optimizer/gradient tensor. A failure leaves the live bank intact.
    pub fn restore_bytes(&self, bytes: &[u8]) -> Result<()> {
        let mut inner = self.lock()?;
        let restored = restore_inner(&inner, bytes, &inner.model_layout)?;
        *inner = restored;
        Ok(())
    }

    fn serialize_snapshot(&self, inner: &BankInner) -> Result<Vec<u8>> {
        #[cfg(test)]
        self.snapshot_serializations.fetch_add(1, Ordering::Relaxed);
        snapshot_inner(inner)
    }

    #[cfg(test)]
    fn snapshot_serialization_count(&self) -> u64 {
        self.snapshot_serializations.load(Ordering::Relaxed)
    }

    pub fn tier_clocks(&self) -> Result<Vec<(u64, u64, u64)>> {
        Ok(self
            .lock()?
            .tiers
            .iter()
            .map(|tier| {
                (
                    tier.update_clock,
                    tier.transfer_clock,
                    tier.accumulated_micro_steps,
                )
            })
            .collect())
    }

    /// Apply every tier whose configured boundary is due, fastest-to-slowest,
    /// without any consolidation, receiver transfer, reserve activation/reset,
    /// or Dreaming. This is deliberately schedule-bound: it is the exact
    /// `wake_only` ablation, not ordinary every-step AdamW over all tiers.
    ///
    /// The operation is atomic with respect to the bank. Every tier applies
    /// the same averaged prospective base update used by sleep staging, while
    /// the returned model is mutated only through the returned value.
    pub fn apply_wake_only_due_updates(
        &self,
        model: &Transformer,
        trigger_clock: u64,
    ) -> Result<(Transformer, WakeOnlyTierUpdateReport)> {
        ensure!(trigger_clock > 0, "wake_only update clock must be positive");
        // Most wake steps are not tier boundaries. Inspect the immutable
        // schedule before taking the rollback snapshot: snapshotting canonical
        // optimizer/gradient bytes synchronizes every accumulator to the host.
        // There is no state to roll back when no tier is due.
        let mut inner = self.lock()?;
        validate_model_topology(&inner, model)?;
        ensure!(
            inner.staged.is_none(),
            "cannot apply wake_only tier updates while a sleep update is staged"
        );
        if !inner
            .update_periods
            .iter()
            .any(|period| trigger_clock.is_multiple_of(*period))
        {
            inner.model_layout = model.clone();
            inner.cached_candidate = None;
            return Ok((
                model.clone(),
                WakeOnlyTierUpdateReport {
                    trigger_clock,
                    updates: Vec::new(),
                },
            ));
        }
        // Keep the bank lock across snapshot, mutation, and rollback. The bank
        // is cloneable and may be shared by checkpointing/control code; a
        // second caller must never commit between our rollback point and a
        // failed update and then have that successful work silently undone.
        let before_layout = inner.model_layout.clone();
        let before = self.serialize_snapshot(&inner)?;
        let result = (|| -> Result<(Transformer, WakeOnlyTierUpdateReport)> {
            let due = inner
                .update_periods
                .iter()
                .enumerate()
                .filter_map(|(tier, period)| trigger_clock.is_multiple_of(*period).then_some(tier))
                .collect::<Vec<_>>();
            let learning_rate = inner.config.learning_rate;
            let mut updated_model = model.clone();
            let mut updates = Vec::with_capacity(due.len());
            for sender in due {
                let before_tier = updated_model.clone();
                let tier = &mut inner.tiers[sender];
                ensure!(
                    trigger_clock > tier.update_clock,
                    "wake_only tier {sender} update clock did not advance"
                );
                ensure!(
                    tier.accumulated_micro_steps > 0 && !tier.accumulator.is_empty(),
                    "wake_only tier {sender} has no accumulated wake gradients at its due boundary"
                );
                let active_ids = active_tier_parameter_ids(&updated_model, sender)?;
                let active = active_ids
                    .iter()
                    .map(|id| id.val())
                    .collect::<BTreeSet<_>>();
                let accumulator_ids =
                    gradient_parameter_ids(&updated_model, &tier.accumulator, &tier.parameter_ids)?;
                ensure!(
                    accumulator_ids.iter().all(|id| active.contains(id)),
                    "wake_only tier {sender} accumulator contains a dormant parameter"
                );
                let accumulated_optimizer_steps = tier.accumulated_micro_steps;
                let mut gradients = std::mem::take(&mut tier.accumulator);
                average_gradients(
                    &updated_model,
                    &mut gradients,
                    &active_ids,
                    accumulated_optimizer_steps,
                )?;
                let mut optimizer = tier.optimizer.clone();
                let candidate =
                    optimizer.step(learning_rate.into(), updated_model.clone(), gradients);
                let prospective_update_sha256 =
                    prospective_update_hash(&before_tier, &candidate, sender)?;
                tier.optimizer = optimizer;
                tier.accumulated_micro_steps = 0;
                tier.update_clock = trigger_clock;
                tier.generation = tier
                    .generation
                    .checked_add(1)
                    .context("wake_only tier optimizer generation overflow")?;
                tier.artifact = None;
                updates.push(WakeOnlyTierUpdate {
                    tier: sender,
                    tier_id: tier.tier_id.clone(),
                    trigger_clock,
                    accumulated_optimizer_steps,
                    prospective_update_sha256,
                });
                updated_model = candidate;
            }
            inner.model_layout = updated_model.clone();
            inner.cached_candidate = None;
            Ok((
                updated_model,
                WakeOnlyTierUpdateReport {
                    trigger_clock,
                    updates,
                },
            ))
        })();
        match result {
            Ok(updated) => Ok(updated),
            Err(error) => {
                *inner = restore_inner(&inner, &before, &before_layout)
                    .context("restoring tier optimizer bank after failed wake_only update")?;
                Err(error)
            }
        }
    }

    /// Return checkpoint-v2 scopes only when every non-empty tier state has a
    /// matching immutable bundle. Call `publish_checkpoint_scopes` on the
    /// durable publisher after wake accumulation and before persisting a cursor.
    pub fn scopes(&self) -> Result<MemoryOptimizerScopes> {
        let inner = self.lock()?;
        scopes_from_inner(&inner)
    }
}

fn scopes_from_inner(inner: &BankInner) -> Result<MemoryOptimizerScopes> {
    let mut scopes = inner.scopes.clone();
    for (scope, tier) in scopes.tiers.iter_mut().zip(&inner.tiers) {
        ensure!(
            tier.artifact.is_some()
                || (tier.accumulated_micro_steps == 0
                    && tier.update_clock == 0
                    && tier.transfer_clock == 0
                    && tier.generation == 0),
            "tier `{}` has mutable state without an immutable optimizer bundle",
            tier.tier_id
        );
        scope.update_clock = tier.update_clock;
        scope.transfer_clock = tier.transfer_clock;
        scope.accumulated_micro_steps = tier.accumulated_micro_steps;
        scope.generation = tier.generation;
        scope.transfer_generation = tier.transfer_generation;
        scope.artifact = tier.artifact.clone();
    }
    Ok(scopes)
}

fn optimizer_topology_hash(
    scopes: &MemoryOptimizerScopes,
    update_periods: &[u64],
) -> Result<String> {
    #[derive(Serialize)]
    struct Topology<'a> {
        wake: &'a [u64],
        tiers: Vec<(&'a str, &'a [u64], u64)>,
    }
    let bytes = serde_json::to_vec(&Topology {
        wake: &scopes.wake_parameter_ids,
        tiers: scopes
            .tiers
            .iter()
            .zip(update_periods)
            .map(|(tier, period)| {
                (
                    tier.tier_id.as_str(),
                    tier.parameter_ids.as_slice(),
                    *period,
                )
            })
            .collect(),
    })?;
    Ok(sha256_identity(&bytes))
}

fn validate_model_topology(inner: &BankInner, model: &Transformer) -> Result<()> {
    let all = list_param_ids(model)
        .into_iter()
        .map(|id| id.val())
        .collect::<BTreeSet<_>>();
    let expected = inner
        .scopes
        .wake_parameter_ids
        .iter()
        .chain(
            inner
                .scopes
                .tiers
                .iter()
                .flat_map(|tier| tier.parameter_ids.iter()),
        )
        .copied()
        .collect::<BTreeSet<_>>();
    ensure!(
        all == expected,
        "model parameter topology differs from tier optimizer bank"
    );
    Ok(())
}

fn active_tier_parameter_ids(model: &Transformer, tier: usize) -> Result<Vec<ParamId>> {
    let ids = model.memory_tier_active_parameter_ids_all_layers(tier)?;
    let unique = ids.iter().map(|id| id.val()).collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == ids.len(),
        "active tier {tier} repeats a parameter ID"
    );
    Ok(ids)
}

fn dormant_parameter_ids(model: &Transformer) -> Vec<ParamId> {
    model.dormant_memory_parameter_ids()
}

struct GradientAccumulatorVisitor<'a> {
    allowed: &'a BTreeSet<u64>,
    accumulated: &'a mut GradientsParams,
    incoming: &'a mut GradientsParams,
}

struct GradientParameterIdVisitor<'a> {
    allowed: &'a BTreeSet<u64>,
    gradients: &'a GradientsParams,
    ids: Vec<u64>,
}

impl ModuleVisitor for GradientParameterIdVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if self.allowed.contains(&parameter.id.val())
            && self.gradients.get::<D>(parameter.id).is_some()
        {
            self.ids.push(parameter.id.val());
        }
    }
}

fn gradient_parameter_ids(
    model: &Transformer,
    gradients: &GradientsParams,
    allowed: &[u64],
) -> Result<Vec<u64>> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    let mut visitor = GradientParameterIdVisitor {
        allowed: &allowed,
        gradients,
        ids: Vec::new(),
    };
    model.visit(&mut visitor);
    ensure!(
        visitor.ids.len() == gradients.len(),
        "completed tier gradient set contains foreign tensors"
    );
    Ok(visitor.ids)
}

impl ModuleVisitor for GradientAccumulatorVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if !self.allowed.contains(&parameter.id.val()) {
            return;
        }
        let Some(incoming) = self.incoming.remove::<D>(parameter.id) else {
            return;
        };
        let value = self
            .accumulated
            .remove::<D>(parameter.id)
            .map_or(incoming.clone(), |current| current + incoming);
        self.accumulated.register(parameter.id, value);
    }
}

fn accumulate_gradients(
    model: &Transformer,
    allowed: &[u64],
    accumulated: &mut GradientsParams,
    incoming: &mut GradientsParams,
) -> Result<()> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    model.visit(&mut GradientAccumulatorVisitor {
        allowed: &allowed,
        accumulated,
        incoming,
    });
    ensure!(
        incoming.is_empty(),
        "tier accumulator received {} foreign gradient tensors",
        incoming.len()
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SnapshotMeta {
    version: u32,
    config: TierOptimizerConfig,
    topology_sha256: String,
    tiers: Vec<TierSnapshotMeta>,
    staged: Option<StagedUpdateMeta>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TierSnapshotMeta {
    tier: usize,
    tier_id: String,
    parameter_ids: Vec<u64>,
    accumulator_parameter_ids: Vec<u64>,
    accumulated_micro_steps: u64,
    update_clock: u64,
    transfer_clock: u64,
    generation: u64,
    transfer_generation: u64,
    artifact: Option<TierOptimizerArtifact>,
    optimizer_bytes: u64,
    optimizer_sha256: String,
    gradient_bytes: u64,
    gradient_sha256: String,
}

fn snapshot_inner(inner: &BankInner) -> Result<Vec<u8>> {
    let mut chunks = Vec::with_capacity(inner.tiers.len());
    let mut tiers = Vec::with_capacity(inner.tiers.len());
    for tier in &inner.tiers {
        let optimizer = canonical_optimizer(&tier.optimizer)?;
        let (gradients, accumulator_parameter_ids) =
            canonical_gradients(&inner.model_layout, &tier.accumulator, &tier.parameter_ids)?;
        tiers.push(TierSnapshotMeta {
            tier: tier.tier,
            tier_id: tier.tier_id.clone(),
            parameter_ids: tier.parameter_ids.clone(),
            accumulator_parameter_ids,
            accumulated_micro_steps: tier.accumulated_micro_steps,
            update_clock: tier.update_clock,
            transfer_clock: tier.transfer_clock,
            generation: tier.generation,
            transfer_generation: tier.transfer_generation,
            artifact: tier.artifact.clone(),
            optimizer_bytes: optimizer.len() as u64,
            optimizer_sha256: sha256_identity(&optimizer),
            gradient_bytes: gradients.len() as u64,
            gradient_sha256: sha256_identity(&gradients),
        });
        chunks.push((optimizer, gradients));
    }
    let header = serde_json::to_vec(&SnapshotMeta {
        version: SNAPSHOT_VERSION,
        config: inner.config.clone(),
        topology_sha256: inner.topology_sha256.clone(),
        tiers,
        staged: inner.staged.clone(),
    })?;
    let mut output = Vec::with_capacity(
        SNAPSHOT_MAGIC.len()
            + 8
            + header.len()
            + chunks
                .iter()
                .map(|(optimizer, gradients)| optimizer.len() + gradients.len())
                .sum::<usize>(),
    );
    output.extend_from_slice(SNAPSHOT_MAGIC);
    output.extend_from_slice(&(header.len() as u64).to_le_bytes());
    output.extend_from_slice(&header);
    for (optimizer, gradients) in chunks {
        output.extend_from_slice(&optimizer);
        output.extend_from_slice(&gradients);
    }
    Ok(output)
}

fn restore_inner(
    template: &BankInner,
    bytes: &[u8],
    model_layout: &Transformer,
) -> Result<BankInner> {
    validate_model_topology(template, model_layout)?;
    ensure!(
        bytes.len() >= SNAPSHOT_MAGIC.len() + 8 && &bytes[..SNAPSHOT_MAGIC.len()] == SNAPSHOT_MAGIC,
        "invalid tier optimizer snapshot magic"
    );
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[SNAPSHOT_MAGIC.len()..SNAPSHOT_MAGIC.len() + 8]
            .try_into()
            .expect("fixed slice"),
    ))
    .context("tier snapshot header does not fit this platform")?;
    ensure!(
        header_len <= MAX_TIER_SNAPSHOT_HEADER_BYTES,
        "tier optimizer snapshot header exceeds the {MAX_TIER_SNAPSHOT_HEADER_BYTES}-byte limit"
    );
    let header_start = SNAPSHOT_MAGIC.len() + 8;
    let header_end = header_start
        .checked_add(header_len)
        .context("tier snapshot header length overflow")?;
    ensure!(
        header_end <= bytes.len(),
        "truncated tier optimizer snapshot header"
    );
    let meta: SnapshotMeta = serde_json::from_slice(&bytes[header_start..header_end])?;
    ensure!(
        meta.version == SNAPSHOT_VERSION,
        "unsupported tier snapshot version"
    );
    ensure!(
        meta.config == template.config,
        "tier optimizer config changed on restore"
    );
    ensure!(
        meta.topology_sha256 == template.topology_sha256,
        "tier optimizer topology changed on restore"
    );
    ensure!(
        meta.tiers.len() == template.tiers.len(),
        "tier count changed on restore"
    );

    let mut offset = header_end;
    let mut tiers = Vec::with_capacity(meta.tiers.len());
    for (saved, expected) in meta.tiers.iter().zip(&template.tiers) {
        ensure!(
            saved.tier == expected.tier
                && saved.tier_id == expected.tier_id
                && saved.parameter_ids == expected.parameter_ids,
            "tier {} topology changed on restore",
            expected.tier
        );
        let optimizer_len = usize::try_from(saved.optimizer_bytes)
            .context("optimizer snapshot does not fit this platform")?;
        let gradient_len = usize::try_from(saved.gradient_bytes)
            .context("gradient snapshot does not fit this platform")?;
        ensure!(
            saved.optimizer_bytes <= MAX_TIER_BUNDLE_MEMBER_BYTES
                && saved.gradient_bytes <= MAX_TIER_BUNDLE_MEMBER_BYTES,
            "tier {} snapshot payload exceeds the per-member size limit",
            saved.tier
        );
        let optimizer_end = offset
            .checked_add(optimizer_len)
            .context("optimizer snapshot length overflow")?;
        let gradient_end = optimizer_end
            .checked_add(gradient_len)
            .context("gradient snapshot length overflow")?;
        ensure!(
            gradient_end <= bytes.len(),
            "truncated tier snapshot payload"
        );
        let optimizer_bytes = &bytes[offset..optimizer_end];
        let gradient_bytes = &bytes[optimizer_end..gradient_end];
        ensure!(
            sha256_identity(optimizer_bytes) == saved.optimizer_sha256
                && sha256_identity(gradient_bytes) == saved.gradient_sha256,
            "tier {} snapshot payload hash mismatch",
            saved.tier
        );
        let optimizer = optimizer_from_bytes(&template.config, optimizer_bytes)?;
        let (accumulator, ids) =
            restore_gradients(model_layout, gradient_bytes, &saved.parameter_ids)?;
        ensure!(
            ids == saved.accumulator_parameter_ids,
            "gradient ID receipt changed"
        );
        ensure!(
            ids.is_empty() == (saved.accumulated_micro_steps == 0),
            "tier accumulator bytes disagree with micro-step count"
        );
        tiers.push(TierRuntimeState {
            tier: saved.tier,
            tier_id: saved.tier_id.clone(),
            parameter_ids: saved.parameter_ids.clone(),
            optimizer,
            accumulator,
            accumulated_micro_steps: saved.accumulated_micro_steps,
            update_clock: saved.update_clock,
            transfer_clock: saved.transfer_clock,
            generation: saved.generation,
            transfer_generation: saved.transfer_generation,
            artifact: saved.artifact.clone(),
        });
        offset = gradient_end;
    }
    ensure!(
        offset == bytes.len(),
        "tier snapshot contains trailing bytes"
    );
    Ok(BankInner {
        config: meta.config,
        topology_sha256: meta.topology_sha256,
        update_periods: template.update_periods.clone(),
        scopes: template.scopes.clone(),
        model_layout: model_layout.clone(),
        tiers,
        staged: meta.staged,
        cached_candidate: None,
    })
}

fn canonical_optimizer(optimizer: &ModuleOptimizer) -> Result<Vec<u8>> {
    Ok(canonical_module_optimizer_bytes(optimizer)?.to_vec())
}

fn optimizer_from_bytes(config: &TierOptimizerConfig, bytes: &[u8]) -> Result<ModuleOptimizer> {
    let optimizer = config
        .optimizer()
        .from_bytes(Bytes::from_bytes_vec(bytes.to_vec()))
        .context("restoring tier optimizer bytes")?;
    ensure!(
        canonical_optimizer(&optimizer)? == bytes,
        "tier optimizer bytes are not canonical or failed exact restore"
    );
    Ok(optimizer)
}

struct GradientSnapshotVisitor<'a> {
    allowed: &'a BTreeSet<u64>,
    gradients: &'a GradientsParams,
    tensors: Vec<PackedTensor>,
}

impl ModuleVisitor for GradientSnapshotVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if !self.allowed.contains(&parameter.id.val()) {
            return;
        }
        let Some(gradient) = self.gradients.get::<D>(parameter.id) else {
            return;
        };
        let data = gradient.into_data();
        self.tensors.push(PackedTensor::new(
            parameter.id.val().to_string(),
            data.dtype,
            data.shape,
            Some(parameter.id.val()),
            data.bytes,
        ));
    }
}

fn canonical_gradients(
    model: &Transformer,
    gradients: &GradientsParams,
    allowed: &[u64],
) -> Result<(Vec<u8>, Vec<u64>)> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    let mut visitor = GradientSnapshotVisitor {
        allowed: &allowed,
        gradients,
        tensors: Vec::new(),
    };
    model.visit(&mut visitor);
    visitor.tensors.sort_by_key(|tensor| tensor.param_id);
    let ids = visitor
        .tensors
        .iter()
        .map(|tensor| tensor.param_id.expect("gradient carries id"))
        .collect::<Vec<_>>();
    ensure!(
        ids.len() == gradients.len(),
        "gradient accumulator contains foreign tensors"
    );
    let bytes = Writer::new(visitor.tensors)
        .with_metadata("format", "hermes-tier-gradients-v1")
        .into_bytes()?;
    Ok((bytes.to_vec(), ids))
}

struct GradientRestoreVisitor<'a> {
    tensors: &'a mut BTreeMap<u64, PackedTensor>,
    gradients: &'a mut GradientsParams,
    allowed: &'a BTreeSet<u64>,
    failure: Option<String>,
}

impl ModuleVisitor for GradientRestoreVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if self.failure.is_some() || !self.allowed.contains(&parameter.id.val()) {
            return;
        }
        let Some(tensor) = self.tensors.remove(&parameter.id.val()) else {
            return;
        };
        if tensor.shape.rank() != D
            || tensor.shape.dims::<D>() != parameter.val().shape().dims::<D>()
            || tensor.dtype != parameter.val().dtype()
        {
            self.failure = Some(format!(
                "gradient {} shape or dtype differs from model",
                parameter.id
            ));
            return;
        }
        let data = TensorData::from_bytes(tensor.bytes, tensor.shape, tensor.dtype);
        // Gradients extracted by `GradientsParams::from_params` live on the
        // non-autodiff inner backend. Reconstructing them on the parameter's
        // autodiff device would make the next optimizer step combine two
        // different dispatch backends and panic.
        let gradient_device = parameter.val().device().inner();
        self.gradients
            .register(parameter.id, Tensor::<D>::from_data(data, &gradient_device));
    }
}

fn restore_gradients(
    model: &Transformer,
    bytes: &[u8],
    allowed: &[u64],
) -> Result<(GradientsParams, Vec<u64>)> {
    let reader = Reader::from_bytes(Bytes::from_bytes_vec(bytes.to_vec()))?;
    ensure!(
        reader.metadata().get("format").map(String::as_str) == Some("hermes-tier-gradients-v1"),
        "invalid tier gradient format"
    );
    let mut tensors = BTreeMap::new();
    for tensor in reader.into_tensors()? {
        let id = tensor
            .param_id
            .context("gradient tensor has no parameter ID")?;
        ensure!(
            tensors.insert(id, tensor).is_none(),
            "duplicate gradient parameter ID {id}"
        );
    }
    let ids = tensors.keys().copied().collect::<Vec<_>>();
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        ids.iter().all(|id| allowed.contains(id)),
        "gradient artifact contains foreign IDs"
    );
    let mut gradients = GradientsParams::new();
    let mut visitor = GradientRestoreVisitor {
        tensors: &mut tensors,
        gradients: &mut gradients,
        allowed: &allowed,
        failure: None,
    };
    model.visit(&mut visitor);
    if let Some(error) = visitor.failure {
        bail!(error);
    }
    ensure!(
        tensors.is_empty(),
        "gradient artifact IDs are absent from model"
    );
    Ok((gradients, ids))
}

struct GradientScaleVisitor<'a> {
    allowed: &'a BTreeSet<u64>,
    gradients: &'a mut GradientsParams,
    divisor: f64,
}

impl ModuleVisitor for GradientScaleVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if !self.allowed.contains(&parameter.id.val()) {
            return;
        }
        if let Some(gradient) = self.gradients.remove::<D>(parameter.id) {
            self.gradients
                .register(parameter.id, gradient.div_scalar(self.divisor));
        }
    }
}

fn average_gradients(
    model: &Transformer,
    gradients: &mut GradientsParams,
    allowed: &[ParamId],
    micro_steps: u64,
) -> Result<()> {
    ensure!(micro_steps > 0, "cannot average an empty tier accumulator");
    let allowed = allowed.iter().map(|id| id.val()).collect::<BTreeSet<_>>();
    model.visit(&mut GradientScaleVisitor {
        allowed: &allowed,
        gradients,
        divisor: micro_steps as f64,
    });
    Ok(())
}

/// Concrete sender-update adapter. `prepare_consolidation` previews and
/// publishes the deterministic prospective model without consuming live
/// optimizer state; `stage` later applies exactly that update inside the
/// transaction and verifies all three identities from `SleepState`.
#[derive(Clone)]
pub struct ProspectiveTierUpdate {
    bank: TierOptimizerBank,
    root: PathBuf,
}

impl ProspectiveTierUpdate {
    pub fn new(bank: TierOptimizerBank, root: impl Into<PathBuf>) -> Result<Self> {
        let value = Self {
            bank,
            root: root.into(),
        };
        ensure_directory(&value.root, "prospective tier-update root")?;
        Ok(value)
    }

    pub fn bank(&self) -> &TierOptimizerBank {
        &self.bank
    }

    pub fn prepare_consolidation(
        &mut self,
        txn_id: u64,
        sender: usize,
        trigger_clock: u64,
        teacher: &ImmutableTransformerCheckpoint,
    ) -> Result<PlannedConsolidation> {
        teacher.validate()?;
        let teacher_model = load_local_checkpoint(&teacher.model, &teacher.uri, &teacher.sha256)?;
        let mut inner = self.bank.lock()?;
        let before_layout = inner.model_layout.clone();
        let before = self.bank.serialize_snapshot(&inner)?;
        let staged = stage_sender_locked(
            &self.root,
            &mut inner,
            txn_id,
            sender,
            trigger_clock,
            &teacher_model,
        );
        let restored = restore_inner(&inner, &before, &before_layout).map(|restored| {
            *inner = restored;
        });
        match (staged, restored) {
            (Ok(candidate), Ok(())) => Ok(PlannedConsolidation {
                student_checkpoint: candidate.checkpoint.uri,
                student_sha256: candidate.checkpoint.sha256,
                prospective_update_sha256: candidate.update_sha256,
            }),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(restore)) => Err(restore.context(
                "prospective tier update was previewed but optimizer state could not be restored",
            )),
            (Err(error), Err(restore)) => bail!(
                "prospective tier update failed: {error:#}; optimizer restore also failed: {restore:#}"
            ),
        }
    }

    fn stage_sender(
        &mut self,
        txn_id: u64,
        sender: usize,
        trigger_clock: u64,
        teacher: &ImmutableTransformerCheckpoint,
    ) -> Result<ProspectiveTransformerCandidate> {
        teacher.validate()?;
        let teacher_model = load_local_checkpoint(&teacher.model, &teacher.uri, &teacher.sha256)?;
        let mut inner = self.bank.lock()?;
        stage_sender_locked(
            &self.root,
            &mut inner,
            txn_id,
            sender,
            trigger_clock,
            &teacher_model,
        )
    }
}

fn stage_sender_locked(
    root: &Path,
    inner: &mut BankInner,
    txn_id: u64,
    sender: usize,
    trigger_clock: u64,
    teacher_model: &Transformer,
) -> Result<ProspectiveTransformerCandidate> {
    validate_model_topology(inner, teacher_model)?;
    if let Some(staged) = &inner.staged {
        ensure!(
            staged.txn_id == txn_id
                && staged.sender == sender
                && staged.trigger_clock == trigger_clock,
            "another prospective tier update is already staged"
        );
        let model = match &inner.cached_candidate {
            Some(model) => model.clone(),
            None => {
                load_local_checkpoint(teacher_model, &staged.student_uri, &staged.student_sha256)?
            }
        };
        return Ok(ProspectiveTransformerCandidate {
            checkpoint: ImmutableTransformerCheckpoint {
                uri: staged.student_uri.clone(),
                sha256: staged.student_sha256.clone(),
                model,
            },
            update_sha256: staged.update_sha256.clone(),
        });
    }
    let learning_rate = inner.config.learning_rate;
    let update_period = *inner
        .update_periods
        .get(sender)
        .with_context(|| format!("sender optimizer tier {sender} does not exist"))?;
    ensure!(
        trigger_clock.is_multiple_of(update_period),
        "sender tier {sender} update is off its configured boundary"
    );
    let tier = inner
        .tiers
        .get_mut(sender)
        .with_context(|| format!("sender optimizer tier {sender} does not exist"))?;
    ensure!(
        tier.accumulated_micro_steps > 0 && !tier.accumulator.is_empty(),
        "sender tier {sender} has no accumulated wake gradients"
    );
    ensure!(
        trigger_clock > tier.update_clock,
        "sender tier {sender} update clock did not advance"
    );
    let active_ids = active_tier_parameter_ids(teacher_model, sender)?;
    let active_set = active_ids
        .iter()
        .map(|id| id.val())
        .collect::<BTreeSet<_>>();
    // This is an execution-time scope check, not a checkpoint write. Reuse
    // the parameter-only visitor so consolidation does not serialize every
    // gradient (and synchronize it back to the host) merely to inspect IDs.
    let accumulator_ids =
        gradient_parameter_ids(teacher_model, &tier.accumulator, &tier.parameter_ids)?;
    ensure!(
        accumulator_ids.iter().all(|id| active_set.contains(id)),
        "sender tier accumulator contains a dormant parameter"
    );

    let mut gradients = std::mem::take(&mut tier.accumulator);
    average_gradients(
        teacher_model,
        &mut gradients,
        &active_ids,
        tier.accumulated_micro_steps,
    )?;
    let mut optimizer = tier.optimizer.clone();
    let candidate_model = optimizer.step(learning_rate.into(), teacher_model.clone(), gradients);
    let update_sha256 = prospective_update_hash(teacher_model, &candidate_model, sender)?;
    let (student_uri, student_sha256) = publish_immutable_model(root, txn_id, &candidate_model)?;
    tier.optimizer = optimizer;
    tier.accumulated_micro_steps = 0;
    tier.artifact = None;
    let staged = StagedUpdateMeta {
        txn_id,
        sender,
        trigger_clock,
        student_uri: student_uri.clone(),
        student_sha256: student_sha256.clone(),
        update_sha256: update_sha256.clone(),
        cleared_parameter_ids: Vec::new(),
    };
    inner.staged = Some(staged);
    inner.cached_candidate = Some(candidate_model.clone());
    Ok(ProspectiveTransformerCandidate {
        checkpoint: ImmutableTransformerCheckpoint {
            uri: student_uri,
            sha256: student_sha256,
            model: candidate_model,
        },
        update_sha256,
    })
}

impl ProspectiveTransformerUpdate for ProspectiveTierUpdate {
    fn snapshot_state(&mut self, _: &ConsolidationTxn) -> Result<ProspectiveUpdateSnapshot> {
        ProspectiveUpdateSnapshot::new(self.bank.snapshot_bytes()?)
    }

    fn restore_state(
        &mut self,
        _: &ConsolidationTxn,
        snapshot: &ProspectiveUpdateSnapshot,
    ) -> Result<()> {
        self.bank.restore_bytes(snapshot.as_bytes())
    }

    fn stage(
        &mut self,
        txn: &ConsolidationTxn,
        teacher: &Transformer,
    ) -> Result<ProspectiveTransformerCandidate> {
        let teacher = ImmutableTransformerCheckpoint {
            uri: txn.teacher_checkpoint.clone(),
            sha256: txn.teacher_hash.clone(),
            model: teacher.clone(),
        };
        let candidate = self.stage_sender(txn.id, txn.sender, txn.trigger_clock, &teacher)?;
        ensure!(
            candidate.checkpoint.uri == txn.student_checkpoint
                && candidate.checkpoint.sha256 == txn.student_hash
                && candidate.update_sha256 == txn.prospective_update_hash,
            "staged sender update differs from its consolidation plan"
        );
        Ok(candidate)
    }

    fn clear_reclaimed_optimizer_state(
        &mut self,
        txn: &ConsolidationTxn,
        parameter_ids: &[ParamId],
    ) -> Result<()> {
        let mut inner = self.bank.lock()?;
        let staged = inner
            .staged
            .as_ref()
            .context("no prospective tier update is staged")?;
        ensure!(
            staged.txn_id == txn.id && staged.sender == txn.sender,
            "reclaimed optimizer state belongs to another transaction"
        );
        let ids = parameter_ids
            .iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        ensure!(
            ids.len() == parameter_ids.len(),
            "reclaimed optimizer parameter IDs are duplicated"
        );
        let tier = &inner.tiers[txn.sender];
        let tier_ids = tier.parameter_ids.iter().copied().collect::<BTreeSet<_>>();
        let base_ids = inner
            .model_layout
            .memory_tier_base_parameter_ids_all_layers(txn.sender)?
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        ensure!(
            ids.iter()
                .all(|id| tier_ids.contains(id) && !base_ids.contains(id)),
            "optimizer reclamation escaped sender reserve slots"
        );
        let filtered = filter_optimizer(&tier.optimizer, &inner.config, &ids)?;
        inner.tiers[txn.sender].optimizer = filtered;
        let staged = inner.staged.as_mut().expect("checked above");
        staged.cleared_parameter_ids = ids.into_iter().collect();
        Ok(())
    }
}

fn load_local_checkpoint(template: &Transformer, uri: &str, sha256: &str) -> Result<Transformer> {
    load_local_checkpoint_inner(template, uri, sha256, |_| Ok(()))
}

fn load_local_checkpoint_inner(
    template: &Transformer,
    uri: &str,
    sha256: &str,
    stage_hook: impl FnOnce(&Path) -> Result<()>,
) -> Result<Transformer> {
    let path = Path::new(uri);
    let bytes = read_regular_bounded(path, MAX_TIER_BUNDLE_MEMBER_BYTES, "prospective checkpoint")?;
    ensure!(
        sha256_identity(&bytes) == sha256,
        "prospective checkpoint hash changed"
    );
    stage_hook(path)?;
    let mut model = template.clone();
    load_safetensors_bytes(&mut model, bytes, "authenticated prospective checkpoint")?;
    Ok(model)
}

fn filter_optimizer(
    optimizer: &ModuleOptimizer,
    config: &TierOptimizerConfig,
    removed: &BTreeSet<u64>,
) -> Result<ModuleOptimizer> {
    let bytes = canonical_optimizer(optimizer)?;
    let reader = Reader::from_bytes(Bytes::from_bytes_vec(bytes))?;
    let scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let mut tensors = reader.into_tensors()?;
    tensors.retain(|tensor| tensor.param_id.is_none_or(|id| !removed.contains(&id)));
    let mut writer = Writer::new(tensors);
    for (key, value) in scalars {
        if key_parameter_id(&key).is_none_or(|id| !removed.contains(&id)) {
            writer = writer.with_scalar(&key, value);
        }
    }
    for (key, value) in metadata {
        if key
            .parse::<u64>()
            .ok()
            .is_none_or(|id| !removed.contains(&id))
        {
            writer = writer.with_metadata(&key, &value);
        }
    }
    let filtered = writer.into_bytes()?;
    optimizer_from_bytes(config, &filtered)
}

fn key_parameter_id(key: &str) -> Option<u64> {
    key.split('.').next()?.parse().ok()
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateReceipt {
    version: u32,
    txn_id: u64,
    uri: String,
    sha256: String,
}

/// Local immutable candidate store. Candidate generations are addressed by
/// the exact canonical safetensors digest; a transaction receipt is published
/// atomically only after the generation and its parent directory are synced.
#[derive(Clone, Debug)]
pub struct AtomicSafetensorsCandidatePublisher {
    root: PathBuf,
}

impl AtomicSafetensorsCandidatePublisher {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let value = Self { root: root.into() };
        ensure_directory(&value.root, "candidate publisher root")?;
        ensure_directory(
            &value.root.join("transactions"),
            "candidate transaction directory",
        )?;
        Ok(value)
    }
}

impl AtomicCandidatePublisher for AtomicSafetensorsCandidatePublisher {
    fn publish_candidate(
        &mut self,
        txn: &ConsolidationTxn,
        candidate: &Transformer,
    ) -> Result<ImmutableTransformerCheckpoint> {
        let receipt_path = self
            .root
            .join("transactions")
            .join(format!("txn-{}.json", txn.id));
        if receipt_path.exists() {
            let receipt: CandidateReceipt = serde_json::from_slice(&read_regular_bounded(
                &receipt_path,
                MAX_TIER_BUNDLE_MANIFEST_BYTES,
                "candidate receipt",
            )?)?;
            ensure!(
                receipt.version == CANDIDATE_STORE_VERSION && receipt.txn_id == txn.id,
                "candidate receipt belongs to another transaction"
            );
            let (candidate_uri, candidate_sha256) =
                publish_immutable_model(&self.root, txn.id, candidate)?;
            ensure!(
                candidate_uri == receipt.uri && candidate_sha256 == receipt.sha256,
                "candidate retry supplied different model bytes"
            );
            return Ok(ImmutableTransformerCheckpoint {
                uri: receipt.uri,
                sha256: receipt.sha256,
                model: candidate.clone(),
            });
        }
        let (uri, sha256) = publish_immutable_model(&self.root, txn.id, candidate)?;
        let receipt = CandidateReceipt {
            version: CANDIDATE_STORE_VERSION,
            txn_id: txn.id,
            uri: uri.clone(),
            sha256: sha256.clone(),
        };
        atomic_write_new(&receipt_path, &serde_json::to_vec_pretty(&receipt)?)?;
        Ok(ImmutableTransformerCheckpoint {
            uri,
            sha256,
            model: candidate.clone(),
        })
    }

    fn restore_teacher(
        &mut self,
        _: &ConsolidationTxn,
        teacher: &ImmutableTransformerCheckpoint,
    ) -> Result<()> {
        // Publishing a generation never advances a live pointer. Rollback is
        // therefore an identity verification, while an unreferenced immutable
        // candidate may be garbage-collected later.
        teacher.validate()?;
        let _ = load_local_checkpoint(&teacher.model, &teacher.uri, &teacher.sha256)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct OptimizerTxnReceipt {
    version: u32,
    txn_id: u64,
    tensor_generation: String,
    tensor_manifest_sha256: String,
    commits: Vec<TierOptimizerCommit>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TierBundleState {
    version: u32,
    tier: usize,
    tier_id: String,
    topology_sha256: String,
    optimizer_config: TierOptimizerConfig,
    update_clock: u64,
    transfer_clock: u64,
    accumulated_micro_steps: u64,
    generation: u64,
    transfer_generation: u64,
    optimizer_parameter_ids: Vec<u64>,
    accumulator_parameter_ids: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BundleFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TierBundleManifest {
    version: u32,
    tier: usize,
    files: Vec<BundleFile>,
}

/// Transaction-idempotent publisher for the sender and receiver optimizer
/// bundles consumed by `execute_native_consolidation`. It authenticates the
/// exact tensor transaction generation, merges receiver-slot moments without
/// consuming the receiver's pending wake accumulator, publishes all affected
/// tier generations, then atomically publishes one transaction receipt.
#[derive(Clone)]
pub struct DurableTierOptimizerPublisher {
    bank: TierOptimizerBank,
    root: PathBuf,
    tensor_store: TensorTransactionStore,
    model_config: ModelDef,
    device: Device,
    #[cfg(test)]
    fail_after_bundle: Arc<Mutex<Option<usize>>>,
}

impl DurableTierOptimizerPublisher {
    pub fn new(
        bank: TierOptimizerBank,
        root: impl Into<PathBuf>,
        tensor_store: TensorTransactionStore,
        model_config: ModelDef,
        device: Device,
    ) -> Result<Self> {
        let value = Self {
            bank,
            root: root.into(),
            tensor_store,
            model_config,
            device,
            #[cfg(test)]
            fail_after_bundle: Arc::new(Mutex::new(None)),
        };
        ensure_directory(&value.root, "tier optimizer store")?;
        ensure_directory(
            &value.root.join("generations"),
            "tier optimizer generations",
        )?;
        ensure_directory(
            &value.root.join("transactions"),
            "tier optimizer transactions",
        )?;
        Ok(value)
    }

    pub fn bank(&self) -> &TierOptimizerBank {
        &self.bank
    }

    /// Restore every independently-clocked tier from the artifacts referenced
    /// by checkpoint-v2. This is intentionally separate from construction so a
    /// caller can first restore the prospective-update snapshot for an
    /// in-flight transaction, or the committed scope receipts for wake/dreaming.
    pub fn restore_scopes(
        &self,
        scopes: &MemoryOptimizerScopes,
        model: &Transformer,
    ) -> Result<()> {
        let mut inner = self.bank.lock()?;
        let before = self.bank.serialize_snapshot(&inner)?;
        let before_layout = inner.model_layout.clone();
        let result = (|| -> Result<()> {
            ensure!(
                scopes.tiers.len() == inner.tiers.len()
                    && scopes.wake_parameter_ids == inner.scopes.wake_parameter_ids,
                "checkpoint optimizer scopes differ from tier optimizer bank"
            );
            validate_model_topology(&inner, model)?;
            inner.model_layout = model.clone();
            for (expected_tier, saved) in scopes.tiers.iter().enumerate() {
                ensure!(
                    saved.tier == expected_tier,
                    "checkpoint optimizer tiers are not in canonical order"
                );
                let current = inner
                    .tiers
                    .get(saved.tier)
                    .context("checkpoint optimizer tier is absent")?;
                ensure!(
                    current.tier_id == saved.tier_id
                        && current.parameter_ids == saved.parameter_ids,
                    "checkpoint optimizer tier topology differs from bank"
                );
                match &saved.artifact {
                    Some(artifact) => {
                        let restored_tier = restore_tier_bundle(&self.root, &mut inner, artifact)?;
                        ensure!(
                            restored_tier == saved.tier,
                            "checkpoint optimizer artifact belongs to another tier"
                        );
                        let restored = &inner.tiers[saved.tier];
                        ensure!(
                            restored.update_clock == saved.update_clock
                                && restored.transfer_clock == saved.transfer_clock
                                && restored.accumulated_micro_steps
                                    == saved.accumulated_micro_steps
                                && restored.generation == saved.generation
                                && restored.transfer_generation == saved.transfer_generation,
                            "checkpoint optimizer tier clocks differ from immutable bundle"
                        );
                    }
                    None => {
                        ensure!(
                            saved.update_clock == 0
                                && saved.transfer_clock == 0
                                && saved.accumulated_micro_steps == 0
                                && saved.generation == 0
                                && saved.transfer_generation == 0,
                            "checkpoint tier has clocks but no durable optimizer artifact"
                        );
                        let reset_optimizer = inner.config.optimizer();
                        let tier = &mut inner.tiers[saved.tier];
                        tier.optimizer = reset_optimizer;
                        tier.accumulator = GradientsParams::new();
                        tier.accumulated_micro_steps = 0;
                        tier.update_clock = 0;
                        tier.transfer_clock = 0;
                        tier.generation = 0;
                        tier.transfer_generation = 0;
                        tier.artifact = None;
                    }
                }
            }
            inner.staged = None;
            inner.cached_candidate = None;
            Ok(())
        })();
        if let Err(error) = result {
            *inner = restore_inner(&inner, &before, &before_layout)
                .context("restoring tier optimizer bank after failed checkpoint-scope restore")?;
            return Err(error);
        }
        Ok(())
    }

    /// Seal the current optimizer and pending-gradient state of every tier and
    /// return the exact checkpoint-v2 scope view. No clock is advanced by a
    /// checkpoint-only publication.
    pub fn publish_checkpoint_scopes(&self) -> Result<MemoryOptimizerScopes> {
        let mut inner = self.bank.lock()?;
        // Publication itself is read-only. Stage every immutable receipt first
        // and update the in-memory references only after all filesystem work
        // succeeds. This makes failure atomic without serializing and then
        // deserializing the complete optimizer/gradient bank on every wake
        // checkpoint.
        let artifacts = (0..inner.tiers.len())
            .map(|tier| publish_tier_bundle(&self.root, &inner, tier))
            .collect::<Result<Vec<_>>>()?;
        for (tier, artifact) in inner.tiers.iter_mut().zip(artifacts) {
            tier.artifact = Some(artifact);
        }
        scopes_from_inner(&inner)
    }

    fn receipt_path(&self, txn_id: u64) -> PathBuf {
        self.root
            .join("transactions")
            .join(format!("txn-{txn_id}.json"))
    }

    fn apply_receipt(
        &self,
        txn: &ConsolidationTxn,
        pointer: &TensorTransactionPointer,
        receipt: &OptimizerTxnReceipt,
    ) -> Result<Vec<TierOptimizerCommit>> {
        ensure!(
            receipt.version == TIER_BUNDLE_VERSION
                && receipt.txn_id == txn.id
                && receipt.tensor_generation == pointer.generation
                && receipt.tensor_manifest_sha256 == pointer.manifest_sha256,
            "tier optimizer receipt differs from tensor transaction"
        );
        validate_optimizer_receipt_commits(txn, &receipt.commits)?;
        let mut recorded = txn.clone();
        recorded.tensor_transaction_generation = Some(pointer.generation.clone());
        recorded.tensor_transaction_manifest_hash = Some(pointer.manifest_sha256.clone());
        let optimizer = {
            let inner = self.bank.lock()?;
            inner.config.optimizer()
        };
        let committed_model = load_committed_tensor_model(
            &self.tensor_store,
            &recorded,
            &self.model_config,
            &self.device,
            optimizer,
        )?;

        // Keep the ownership lock from the rollback snapshot through the
        // complete receipt application. Releasing it between those points
        // would let another clone commit work which a later failure here could
        // silently erase by restoring a stale snapshot.
        let mut inner = self.bank.lock()?;
        ensure_transaction_tiers_exist(&inner, txn)?;
        let before_layout = inner.model_layout.clone();
        let before = self.bank.serialize_snapshot(&inner)?;
        let result = (|| -> Result<()> {
            inner.model_layout = committed_model;
            for commit in &receipt.commits {
                let restored_tier = restore_tier_bundle(&self.root, &mut inner, &commit.artifact)?;
                ensure!(
                    restored_tier == commit.tier,
                    "optimizer transaction receipt points at another tier's artifact"
                );
                let restored = &inner.tiers[commit.tier];
                match commit.role {
                    TierOptimizerCommitRole::SenderUpdate
                    | TierOptimizerCommitRole::TerminalCombined => ensure!(
                        restored.update_clock == txn.trigger_clock
                            && restored.accumulated_micro_steps == 0
                            && restored.accumulator.is_empty(),
                        "sender optimizer receipt does not represent the committed boundary"
                    ),
                    TierOptimizerCommitRole::ReceiverTransfer => ensure!(
                        restored.transfer_clock == txn.trigger_clock,
                        "receiver optimizer receipt does not represent the committed transfer"
                    ),
                }
            }
            inner.staged = None;
            inner.cached_candidate = None;
            Ok(())
        })();
        if let Err(error) = result {
            *inner = restore_inner(&inner, &before, &before_layout)
                .context("restoring tier optimizer bank after invalid durable receipt")?;
            return Err(error);
        }
        Ok(receipt.commits.clone())
    }

    #[cfg(test)]
    fn inject_failure_after_bundle(&self, bundles: usize) {
        *self.fail_after_bundle.lock().unwrap() = Some(bundles);
    }

    #[cfg(test)]
    fn clear_failure_injection(&self) {
        *self.fail_after_bundle.lock().unwrap() = None;
    }
}

fn validate_optimizer_receipt_commits(
    txn: &ConsolidationTxn,
    commits: &[TierOptimizerCommit],
) -> Result<()> {
    if txn.terminal {
        ensure!(
            commits.len() == 1
                && commits[0].tier == txn.sender
                && commits[0].role == TierOptimizerCommitRole::TerminalCombined,
            "terminal optimizer receipt has the wrong tier or role"
        );
    } else {
        ensure!(
            commits.len() == 2
                && commits[0].tier == txn.sender
                && commits[0].role == TierOptimizerCommitRole::SenderUpdate
                && commits[1].tier == txn.receiver
                && commits[1].role == TierOptimizerCommitRole::ReceiverTransfer,
            "optimizer receipt does not contain canonical sender/receiver commits"
        );
    }
    Ok(())
}

fn ensure_transaction_tiers_exist(inner: &BankInner, txn: &ConsolidationTxn) -> Result<()> {
    ensure!(
        txn.sender < inner.tiers.len(),
        "consolidation sender tier {} is outside optimizer bank with {} tiers",
        txn.sender,
        inner.tiers.len()
    );
    if !txn.terminal {
        ensure!(
            txn.receiver < inner.tiers.len() && txn.receiver != txn.sender,
            "consolidation receiver tier {} is absent or aliases sender tier {}",
            txn.receiver,
            txn.sender
        );
    }
    Ok(())
}

impl TierOptimizerPublisher for DurableTierOptimizerPublisher {
    fn publish(
        &mut self,
        txn: &ConsolidationTxn,
        pointer: &TensorTransactionPointer,
        receiver_parameter_ids: &[u64],
    ) -> Result<Vec<TierOptimizerCommit>> {
        ensure!(
            pointer.txn_id == txn.id,
            "tensor pointer belongs to another transaction"
        );
        let receipt_path = self.receipt_path(txn.id);
        if receipt_path.exists() {
            let receipt: OptimizerTxnReceipt = serde_json::from_slice(&read_regular_bounded(
                &receipt_path,
                MAX_TIER_BUNDLE_MANIFEST_BYTES,
                "optimizer transaction receipt",
            )?)?;
            return self.apply_receipt(txn, pointer, &receipt);
        }

        let mut recorded = txn.clone();
        recorded.tensor_transaction_generation = Some(pointer.generation.clone());
        recorded.tensor_transaction_manifest_hash = Some(pointer.manifest_sha256.clone());
        let receiver_optimizer = {
            let inner = self.bank.lock()?;
            self.tensor_store
                .load_recorded(
                    &recorded,
                    &self.model_config,
                    &self.device,
                    inner.config.optimizer(),
                )?
                .receiver_optimizer
        };

        // The bank is shared by cloned publishers and wake accumulation.
        // Serialize all mutable work from the rollback point onward so a
        // failed transaction can restore only its own changes.
        let mut inner = self.bank.lock()?;
        if receipt_path.exists() {
            // Another clone may have published while this caller loaded the
            // detached tensor transaction. Drop the guard before the normal
            // idempotent receipt path acquires it again.
            drop(inner);
            let receipt: OptimizerTxnReceipt = serde_json::from_slice(&read_regular_bounded(
                &receipt_path,
                MAX_TIER_BUNDLE_MANIFEST_BYTES,
                "optimizer transaction receipt",
            )?)?;
            return self.apply_receipt(txn, pointer, &receipt);
        }
        ensure_transaction_tiers_exist(&inner, txn)?;
        let before_layout = inner.model_layout.clone();
        let before = self.bank.serialize_snapshot(&inner)?;
        let result = (|| {
            let staged = inner
                .staged
                .as_ref()
                .context("tier optimizer publisher has no staged sender update")?;
            ensure!(
                staged.txn_id == txn.id
                    && staged.sender == txn.sender
                    && staged.trigger_clock == txn.trigger_clock
                    && staged.student_uri == txn.student_checkpoint
                    && staged.student_sha256 == txn.student_hash
                    && staged.update_sha256 == txn.prospective_update_hash,
                "staged tier update differs from consolidation transaction"
            );
            ensure!(
                txn.trigger_clock > inner.tiers[txn.sender].update_clock,
                "sender optimizer clock did not advance"
            );
            if !txn.terminal {
                ensure!(
                    txn.trigger_clock > inner.tiers[txn.receiver].transfer_clock,
                    "receiver optimizer transfer clock did not advance"
                );
            }

            let committed_model = load_committed_tensor_model(
                &self.tensor_store,
                &recorded,
                &self.model_config,
                &self.device,
                inner.config.optimizer(),
            )?;
            let expected_reclaimed = committed_model
                .memory_slot_statuses()
                .into_iter()
                .filter(|status| {
                    status.tier == txn.sender
                        && txn.sender_slots_to_reset.contains(&status.slot)
                        && !status.active
                })
                .flat_map(|status| status.parameter_ids.into_iter().map(|id| id.val()))
                .collect::<BTreeSet<_>>();
            let cleared = staged
                .cleared_parameter_ids
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            ensure!(
                (txn.sender_slots_to_reset.is_empty() || !expected_reclaimed.is_empty())
                    && cleared.len() == staged.cleared_parameter_ids.len()
                    && cleared == expected_reclaimed,
                "sender reclaimed-slot optimizer state was not reset exactly"
            );
            let expected_receiver = if txn.terminal {
                committed_model
                    .memory_tier_base_parameter_ids_all_layers(txn.sender)?
                    .into_iter()
                    .map(|id| id.val())
                    .collect::<BTreeSet<_>>()
            } else {
                committed_model
                    .memory_slot_statuses()
                    .into_iter()
                    .filter(|status| {
                        status.tier == txn.receiver
                            && status.slot == txn.receiver_slot
                            && status.active
                    })
                    .flat_map(|status| status.parameter_ids.into_iter().map(|id| id.val()))
                    .collect::<BTreeSet<_>>()
            };
            let provided = receiver_parameter_ids
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            ensure!(
                !expected_receiver.is_empty()
                    && provided.len() == receiver_parameter_ids.len()
                    && provided == expected_receiver,
                "receiver optimizer IDs differ from committed tensor transaction"
            );

            let optimizer_config = inner.config.clone();
            if txn.terminal {
                let sender = &mut inner.tiers[txn.sender];
                sender.optimizer = merge_optimizer(
                    &sender.optimizer,
                    &receiver_optimizer,
                    &optimizer_config,
                    &provided,
                )?;
                sender.update_clock = txn.trigger_clock;
                sender.generation = sender
                    .generation
                    .checked_add(1)
                    .context("terminal tier optimizer generation overflow")?;
            } else {
                let sender = &mut inner.tiers[txn.sender];
                sender.update_clock = txn.trigger_clock;
                sender.generation = sender
                    .generation
                    .checked_add(1)
                    .context("sender tier optimizer generation overflow")?;

                let receiver = &mut inner.tiers[txn.receiver];
                receiver.optimizer = merge_optimizer(
                    &receiver.optimizer,
                    &receiver_optimizer,
                    &optimizer_config,
                    &provided,
                )?;
                receiver.transfer_clock = txn.trigger_clock;
                receiver.transfer_generation = receiver
                    .transfer_generation
                    .checked_add(1)
                    .context("receiver tier transfer generation overflow")?;
                receiver.generation = receiver
                    .generation
                    .checked_add(1)
                    .context("receiver tier optimizer generation overflow")?;
            }
            inner.model_layout = committed_model;

            let affected = if txn.terminal {
                vec![(txn.sender, TierOptimizerCommitRole::TerminalCombined)]
            } else {
                vec![
                    (txn.sender, TierOptimizerCommitRole::SenderUpdate),
                    (txn.receiver, TierOptimizerCommitRole::ReceiverTransfer),
                ]
            };
            let mut commits = Vec::with_capacity(affected.len());
            for (tier, role) in affected {
                let artifact = publish_tier_bundle(&self.root, &inner, tier)?;
                inner.tiers[tier].artifact = Some(artifact.clone());
                commits.push(TierOptimizerCommit {
                    tier,
                    role,
                    artifact,
                });
                #[cfg(test)]
                if self
                    .fail_after_bundle
                    .lock()
                    .unwrap()
                    .is_some_and(|count| commits.len() == count)
                {
                    bail!("injected crash after optimizer bundle publication");
                }
            }
            let receipt = OptimizerTxnReceipt {
                version: TIER_BUNDLE_VERSION,
                txn_id: txn.id,
                tensor_generation: pointer.generation.clone(),
                tensor_manifest_sha256: pointer.manifest_sha256.clone(),
                commits: commits.clone(),
            };
            atomic_write_new(&receipt_path, &serde_json::to_vec_pretty(&receipt)?)?;
            inner.staged = None;
            inner.cached_candidate = None;
            Ok(commits)
        })();
        match result {
            Ok(commits) => Ok(commits),
            Err(error) => {
                *inner = restore_inner(&inner, &before, &before_layout)
                    .context("restoring tier optimizer bank after failed artifact transaction")?;
                Err(error)
            }
        }
    }
}

fn load_committed_tensor_model(
    store: &TensorTransactionStore,
    txn: &ConsolidationTxn,
    config: &ModelDef,
    device: &Device,
    optimizer: ModuleOptimizer,
) -> Result<Transformer> {
    Ok(store
        .load_recorded(txn, config, device, optimizer)?
        .student
        .checkpoint
        .model)
}

fn merge_optimizer(
    base: &ModuleOptimizer,
    overlay: &ModuleOptimizer,
    config: &TierOptimizerConfig,
    overlay_ids: &BTreeSet<u64>,
) -> Result<ModuleOptimizer> {
    let base = optimizer_parts(&canonical_optimizer(base)?)?;
    let overlay = optimizer_parts(&canonical_optimizer(overlay)?)?;
    ensure!(
        overlay
            .tensors
            .iter()
            .filter_map(|tensor| tensor.param_id)
            .all(|id| overlay_ids.contains(&id))
            && overlay
                .scalars
                .keys()
                .filter_map(|key| key_parameter_id(key))
                .all(|id| overlay_ids.contains(&id)),
        "receiver optimizer contains state outside the activated receiver scope"
    );
    let mut tensors = base
        .tensors
        .into_iter()
        .filter(|tensor| tensor.param_id.is_none_or(|id| !overlay_ids.contains(&id)))
        .collect::<Vec<_>>();
    tensors.extend(overlay.tensors);
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let mut scalars = base
        .scalars
        .into_iter()
        .filter(|(key, _)| key_parameter_id(key).is_none_or(|id| !overlay_ids.contains(&id)))
        .collect::<BTreeMap<_, _>>();
    scalars.extend(overlay.scalars);
    let mut metadata = base
        .metadata
        .into_iter()
        .filter(|(key, _)| {
            key.parse::<u64>()
                .ok()
                .is_none_or(|id| !overlay_ids.contains(&id))
        })
        .collect::<BTreeMap<_, _>>();
    metadata.extend(overlay.metadata);
    let mut writer = Writer::new(tensors);
    for (key, value) in scalars {
        writer = writer.with_scalar(&key, value);
    }
    for (key, value) in metadata {
        writer = writer.with_metadata(&key, &value);
    }
    optimizer_from_bytes(config, &writer.into_bytes()?)
}

struct OptimizerParts {
    tensors: Vec<PackedTensor>,
    scalars: BTreeMap<String, burn_pack::Scalar>,
    metadata: BTreeMap<String, String>,
}

fn optimizer_parts(bytes: &[u8]) -> Result<OptimizerParts> {
    let reader = Reader::from_bytes(Bytes::from_bytes_vec(bytes.to_vec()))?;
    let scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let tensors = reader.into_tensors()?;
    Ok(OptimizerParts {
        tensors,
        scalars,
        metadata,
    })
}

fn active_tier_ids_u64(model: &Transformer, tier: usize) -> Result<Vec<u64>> {
    let mut ids = active_tier_parameter_ids(model, tier)?
        .into_iter()
        .map(|id| id.val())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    Ok(ids)
}

fn publish_tier_bundle(
    root: &Path,
    inner: &BankInner,
    tier: usize,
) -> Result<TierOptimizerArtifact> {
    let state = inner
        .tiers
        .get(tier)
        .with_context(|| format!("optimizer tier {tier} does not exist"))?;
    let optimizer = canonical_optimizer(&state.optimizer)?;
    let (gradients, accumulator_parameter_ids) = canonical_gradients(
        &inner.model_layout,
        &state.accumulator,
        &state.parameter_ids,
    )?;
    let optimizer_parameter_ids = active_tier_ids_u64(&inner.model_layout, tier)?;
    let metadata = TierBundleState {
        version: TIER_BUNDLE_VERSION,
        tier,
        tier_id: state.tier_id.clone(),
        topology_sha256: inner.topology_sha256.clone(),
        optimizer_config: inner.config.clone(),
        update_clock: state.update_clock,
        transfer_clock: state.transfer_clock,
        accumulated_micro_steps: state.accumulated_micro_steps,
        generation: state.generation,
        transfer_generation: state.transfer_generation,
        optimizer_parameter_ids: optimizer_parameter_ids.clone(),
        accumulator_parameter_ids: accumulator_parameter_ids.clone(),
    };
    let state_bytes = serde_json::to_vec_pretty(&metadata)?;
    let generations = root.join("generations");
    let staging = generations.join(format!(
        ".staging-tier-{tier}-{}-{}",
        std::process::id(),
        STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&staging)?;
    let _guard = StagingDirectory(staging.clone());
    write_new_synced(&staging.join(OPTIMIZER_FILE), &optimizer)?;
    write_new_synced(&staging.join(GRADIENT_FILE), &gradients)?;
    write_new_synced(&staging.join(STATE_FILE), &state_bytes)?;
    let mut files = [OPTIMIZER_FILE, GRADIENT_FILE, STATE_FILE]
        .into_iter()
        .map(|name| bundle_file(&staging, name))
        .collect::<Result<Vec<_>>>()?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    let manifest = TierBundleManifest {
        version: TIER_BUNDLE_VERSION,
        tier,
        files,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let manifest_hash = sha256_identity(&manifest_bytes);
    write_new_synced(&staging.join(MANIFEST_FILE), &manifest_bytes)?;
    sync_directory(&staging)?;
    let generation = format!(
        "sha256-{}",
        manifest_hash.strip_prefix("sha256:").expect("hash format")
    );
    let sealed = generations.join(&generation);
    if sealed.exists() {
        verify_tier_bundle(&sealed, &manifest_hash)?;
        fs::remove_dir_all(&staging)?;
    } else {
        fs::rename(&staging, &sealed)?;
    }
    sync_directory(&generations)?;
    Ok(TierOptimizerArtifact {
        state_uri: sealed.join(MANIFEST_FILE).to_string_lossy().into_owned(),
        manifest_hash,
        optimizer_parameter_ids,
        accumulator_parameter_ids,
    })
}

fn restore_tier_bundle(
    root: &Path,
    inner: &mut BankInner,
    artifact: &TierOptimizerArtifact,
) -> Result<usize> {
    restore_tier_bundle_inner(root, inner, artifact, |_, _| Ok(()))
}

fn restore_tier_bundle_inner(
    root: &Path,
    inner: &mut BankInner,
    artifact: &TierOptimizerArtifact,
    mut stage_hook: impl FnMut(TierBundleLoadStage, &Path) -> Result<()>,
) -> Result<usize> {
    artifact.validate()?;
    let manifest_path = Path::new(&artifact.state_uri);
    let generations = root.join("generations");
    ensure!(
        manifest_path.file_name().and_then(|name| name.to_str()) == Some(MANIFEST_FILE),
        "tier optimizer artifact escapes configured store"
    );
    let generation = manifest_path
        .parent()
        .context("tier artifact has no generation")?;
    ensure!(
        generation.parent() == Some(generations.as_path()),
        "tier optimizer artifact escapes configured store"
    );
    let expected_generation = format!(
        "sha256-{}",
        artifact
            .manifest_hash
            .strip_prefix("sha256:")
            .context("tier optimizer manifest hash is malformed")?
    );
    ensure!(
        generation.file_name().and_then(|name| name.to_str()) == Some(expected_generation.as_str()),
        "tier optimizer generation name differs from its manifest hash"
    );
    let (manifest, mut captured) =
        capture_verified_tier_bundle(generation, &artifact.manifest_hash)?;
    stage_hook(TierBundleLoadStage::AfterCapture, generation)?;
    let state_bytes = captured.read_bounded(STATE_FILE, MAX_TIER_BUNDLE_STATE_BYTES)?;
    let state: TierBundleState = serde_json::from_slice(&state_bytes)?;
    ensure!(
        state.version == TIER_BUNDLE_VERSION
            && state.tier == manifest.tier
            && state.topology_sha256 == inner.topology_sha256
            && state.optimizer_config == inner.config
            && state.optimizer_parameter_ids == artifact.optimizer_parameter_ids
            && state.accumulator_parameter_ids == artifact.accumulator_parameter_ids,
        "tier optimizer bundle metadata differs from receipt"
    );
    artifact.validate_pending_steps(state.accumulated_micro_steps)?;
    let tier = inner
        .tiers
        .get(state.tier)
        .context("tier optimizer bundle references absent tier")?;
    ensure!(
        tier.tier_id == state.tier_id,
        "tier optimizer bundle id changed"
    );
    let active_ids = active_tier_ids_u64(&inner.model_layout, state.tier)?;
    let active = active_ids.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        state.optimizer_parameter_ids == active_ids
            && state
                .accumulator_parameter_ids
                .iter()
                .all(|id| active.contains(id)),
        "tier optimizer bundle contains dormant or omits active parameter scope"
    );
    let optimizer_bytes = captured.take(OPTIMIZER_FILE, MAX_TIER_BUNDLE_MEMBER_BYTES)?;
    let optimizer = optimizer_from_bytes(&inner.config, &optimizer_bytes)?;
    drop(optimizer_bytes);
    let gradient_bytes = captured.take(GRADIENT_FILE, MAX_TIER_BUNDLE_MEMBER_BYTES)?;
    let (accumulator, ids) =
        restore_gradients(&inner.model_layout, &gradient_bytes, &tier.parameter_ids)?;
    drop(gradient_bytes);
    ensure!(
        ids == state.accumulator_parameter_ids,
        "tier gradient receipt changed"
    );
    ensure!(
        accumulator.is_empty() == (state.accumulated_micro_steps == 0),
        "restored tier gradients disagree with the pending-step counter"
    );
    stage_hook(TierBundleLoadStage::AfterStagedLoad, generation)?;
    captured.ensure_still_published()?;
    let tier = inner
        .tiers
        .get_mut(state.tier)
        .context("tier optimizer bundle references absent tier")?;
    tier.optimizer = optimizer;
    tier.accumulator = accumulator;
    tier.accumulated_micro_steps = state.accumulated_micro_steps;
    tier.update_clock = state.update_clock;
    tier.transfer_clock = state.transfer_clock;
    tier.generation = state.generation;
    tier.transfer_generation = state.transfer_generation;
    tier.artifact = Some(artifact.clone());
    Ok(state.tier)
}

fn verify_tier_bundle(generation: &Path, expected_hash: &str) -> Result<TierBundleManifest> {
    let (manifest, captured) = capture_verified_tier_bundle(generation, expected_hash)?;
    captured.ensure_still_published()?;
    Ok(manifest)
}

fn capture_verified_tier_bundle(
    generation: &Path,
    expected_hash: &str,
) -> Result<(TierBundleManifest, AuthenticatedDirectorySnapshot)> {
    let mut captured = AuthenticatedDirectorySnapshot::capture(
        generation,
        &TIER_BUNDLE_SCHEMA,
        "tier optimizer generation",
    )?;
    let manifest_bytes = captured.read_bounded(MANIFEST_FILE, MAX_TIER_BUNDLE_MANIFEST_BYTES)?;
    ensure!(
        sha256_identity(&manifest_bytes) == expected_hash,
        "tier manifest hash mismatch"
    );
    let manifest: TierBundleManifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest.version == TIER_BUNDLE_VERSION,
        "unsupported tier bundle version"
    );
    let expected_files = [OPTIMIZER_FILE, GRADIENT_FILE, STATE_FILE]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let observed_files = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        manifest.files.len() == expected_files.len() && observed_files == expected_files,
        "tier bundle does not contain the exact required file set"
    );
    for file in &manifest.files {
        ensure!(
            file.bytes <= MAX_TIER_BUNDLE_MEMBER_BYTES,
            "tier bundle file `{}` exceeds the per-member size limit",
            file.path
        );
        captured.verify(&file.path, file.bytes, &file.sha256)?;
    }
    Ok((manifest, captured))
}

fn bundle_file(root: &Path, name: &str) -> Result<BundleFile> {
    let (bytes, sha256) = hash_file(&root.join(name))?;
    Ok(BundleFile {
        path: name.into(),
        bytes,
        sha256,
    })
}

struct StagingDirectory(PathBuf);

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if self.0.exists() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

fn publish_immutable_model(
    root: &Path,
    txn_id: u64,
    model: &Transformer,
) -> Result<(String, String)> {
    ensure_directory(root, "model artifact root")?;
    let generations = root.join("model-generations");
    ensure_directory(&generations, "model generation directory")?;
    let staging = generations.join(format!(
        ".staging-model-{txn_id}-{}-{}",
        std::process::id(),
        STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&staging)?;
    let _guard = StagingDirectory(staging.clone());
    let weights = staging.join(WEIGHTS_FILE);
    save_safetensors(&model.clone().valid(), &weights)?;
    sync_regular_file(&weights, "model generation weights")?;
    let (_, sha256) = hash_file(&weights)?;
    sync_directory(&staging)?;
    let generation = format!(
        "sha256-{}",
        sha256.strip_prefix("sha256:").expect("hash format")
    );
    let sealed = generations.join(generation);
    if sealed.exists() {
        ensure_real_directory(&sealed, "model generation")?;
        ensure!(
            hash_file(&sealed.join(WEIGHTS_FILE))?.1 == sha256,
            "existing model generation differs from its content address"
        );
        fs::remove_dir_all(&staging)?;
    } else {
        fs::rename(&staging, &sealed)?;
    }
    sync_directory(&generations)?;
    Ok((
        sealed.join(WEIGHTS_FILE).to_string_lossy().into_owned(),
        sha256,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use burn::tensor::{Int, Tensor};
    use hermes_llm::{Device, parse_mal};
    use tempfile::TempDir;

    use super::*;
    use crate::sleep::{MemoryTierSchedule, SleepState, TerminalConsolidation, UpdateClock};
    use crate::tensor_sleep::{RecoveredTensorTransaction, TensorSleepDiagnostics, TensorStage};

    fn schedule() -> SleepSchedule {
        SleepSchedule {
            clock: UpdateClock::OptimizerSteps,
            terminal_consolidation: TerminalConsolidation::DistillIntoBaseV1,
            tiers: vec![
                MemoryTierSchedule {
                    id: "fast".into(),
                    update_period: 1,
                    reserve_slots: 1,
                },
                MemoryTierSchedule {
                    id: "slow".into(),
                    update_period: 2,
                    reserve_slots: 2,
                },
            ],
        }
    }

    fn model_with_fast_slot(active: bool) -> (ModelDef, Device, Transformer) {
        let config = parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 1 rank: 3 top_k: 1 }
                }
                tier slow {
                    ffn: base residual_init: zero
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: {
                    attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
                    memory: cms
                    dropout: 0.0
                }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray().autodiff();
        device.seed(7);
        let mut model = Transformer::new(&config, &device).unwrap();
        if active {
            model.activate_memory_slot_all_layers(0, 0).unwrap();
        }
        (config, device, model)
    }

    fn model() -> (ModelDef, Device, Transformer) {
        model_with_fast_slot(true)
    }

    fn accumulate(bank: &TierOptimizerBank, model: &Transformer, device: &Device) {
        let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], device);
        let target = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], device);
        let mut gradients = model.forward_loss(input, target).backward();
        let (_, report) = bank
            .partition_and_accumulate(model, &mut gradients, 1)
            .unwrap();
        assert_eq!(report.tier_gradient_tensors.len(), 2);
        assert!(report.tier_gradient_tensors.iter().all(|count| *count > 0));
    }

    #[test]
    fn discarded_optimizer_window_never_enters_the_tier_bank() {
        let (_, device, model) = model();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        let before = bank.snapshot_bytes().unwrap();

        let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
        let target = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], &device);
        let mut gradients = model.forward_loss(input, target).backward();
        let partitioned = bank.partition_gradients(&model, &mut gradients).unwrap();
        assert!(
            partitioned
                .tiers
                .iter()
                .all(|gradients| !gradients.is_empty())
        );
        drop(partitioned);

        assert_eq!(bank.snapshot_bytes().unwrap(), before);
        assert_eq!(bank.tier_clocks().unwrap(), vec![(0, 0, 0), (0, 0, 0)]);
    }

    #[test]
    fn accumulation_cannot_coalesce_checkpoint_generations() {
        let (_, _, model) = model();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        let before = bank.snapshot_bytes().unwrap();

        let error = bank
            .commit_tier_gradients(&model, Vec::new(), 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exactly one optimizer step"), "{error}");
        assert_eq!(bank.snapshot_bytes().unwrap(), before);
    }

    fn teacher_checkpoint(root: &Path, model: &Transformer) -> ImmutableTransformerCheckpoint {
        let path = root.join("teacher.safetensors");
        save_safetensors(&model.clone().valid(), &path).unwrap();
        ImmutableTransformerCheckpoint {
            uri: path.to_string_lossy().into_owned(),
            sha256: hash_file(&path).unwrap().1,
            model: model.clone(),
        }
    }

    fn transaction_with_sender_state(
        model: &Transformer,
        plan: PlannedConsolidation,
    ) -> (SleepState, ConsolidationTxn) {
        let schedule = schedule();
        let mut state = SleepState::new(&schedule, 3).unwrap();
        let active = model
            .memory_slot_statuses()
            .into_iter()
            .find(|status| status.tier == 0 && status.slot == 0)
            .unwrap();
        state.tiers[0].slots[0].active = active.active;
        state.tiers[0].slots[0].generation = active.generation;
        state.advance_clock(&schedule, 1).unwrap();
        let txn = state
            .begin(
                0,
                "teacher.safetensors".into(),
                format!("sha256:{}", "a".repeat(64)),
                plan.student_checkpoint,
                plan.student_sha256,
                plan.prospective_update_sha256,
            )
            .unwrap();
        (state, txn)
    }

    fn transaction(
        model: &Transformer,
        plan: PlannedConsolidation,
    ) -> (SleepState, ConsolidationTxn) {
        transaction_with_sender_state(model, plan)
    }

    #[test]
    fn snapshots_are_byte_exact_and_dormant_slots_never_enter_accumulators() {
        let (config, device, model) = model();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        accumulate(&bank, &model, &device);
        let first = bank.snapshot_bytes().unwrap();
        let second = bank.snapshot_bytes().unwrap();
        assert_eq!(first, second);

        let restored =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        restored.restore_bytes(&first).unwrap();
        assert_eq!(first, restored.snapshot_bytes().unwrap());

        let directory = TempDir::new().unwrap();
        let publisher = DurableTierOptimizerPublisher::new(
            restored.clone(),
            directory.path().join("optimizers"),
            TensorTransactionStore::new(directory.path().join("tensor")),
            config.clone(),
            device,
        )
        .unwrap();
        let clocks_before_publication = restored.tier_clocks().unwrap();
        let scopes = publisher.publish_checkpoint_scopes().unwrap();
        assert_eq!(restored.tier_clocks().unwrap(), clocks_before_publication);
        assert_eq!(publisher.publish_checkpoint_scopes().unwrap(), scopes);
        let dormant = dormant_parameter_ids(&model)
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        for scope in &scopes.tiers {
            let artifact = scope.artifact.as_ref().unwrap();
            assert!(
                artifact
                    .accumulator_parameter_ids
                    .iter()
                    .all(|id| !dormant.contains(id))
            );
        }

        let before_failed_restore = restored.snapshot_bytes().unwrap();
        let mut corrupted = scopes;
        corrupted.tiers[1].artifact.as_mut().unwrap().manifest_hash =
            format!("sha256:{}", "0".repeat(64));
        assert!(publisher.restore_scopes(&corrupted, &model).is_err());
        assert_eq!(before_failed_restore, restored.snapshot_bytes().unwrap());
    }

    #[test]
    fn snapshot_restore_rejects_an_oversized_header_before_parsing() {
        let (_, _, model) = model();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        let before = bank.snapshot_bytes().unwrap();
        let mut hostile = Vec::from(SNAPSHOT_MAGIC.as_slice());
        hostile.extend_from_slice(
            &u64::try_from(MAX_TIER_SNAPSHOT_HEADER_BYTES + 1)
                .unwrap()
                .to_le_bytes(),
        );

        let error = bank.restore_bytes(&hostile).unwrap_err().to_string();
        assert!(error.contains("snapshot header exceeds"), "{error}");
        assert_eq!(bank.snapshot_bytes().unwrap(), before);
    }

    #[test]
    fn wake_only_non_boundary_does_not_serialize_optimizer_state() {
        let (_, device, model) = model();
        let mut delayed = schedule();
        delayed.tiers[0].update_period = 2;
        delayed.tiers[1].update_period = 4;
        let bank =
            TierOptimizerBank::new(&model, &delayed, TierOptimizerConfig::default()).unwrap();
        accumulate(&bank, &model, &device);
        let serializations_before = bank.snapshot_serialization_count();
        let clocks_before = bank.tier_clocks().unwrap();

        let (unchanged, report) = bank.apply_wake_only_due_updates(&model, 1).unwrap();

        assert!(report.updates.is_empty());
        assert_eq!(bank.snapshot_serialization_count(), serializations_before);
        assert_eq!(bank.tier_clocks().unwrap(), clocks_before);
        assert_eq!(
            crate::tensor_sleep::model_parameter_hash(&unchanged).unwrap(),
            crate::tensor_sleep::model_parameter_hash(&model).unwrap()
        );
    }

    #[test]
    fn wake_only_failure_after_an_earlier_due_tier_rolls_back_the_complete_bank() {
        let (_, device, model) = model();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        accumulate(&bank, &model, &device);
        {
            let mut inner = bank.lock().unwrap();
            inner.tiers[1].generation = u64::MAX;
        }
        let before = bank.snapshot_bytes().unwrap();
        let clocks_before = bank.tier_clocks().unwrap();

        let error = bank.apply_wake_only_due_updates(&model, 2).unwrap_err();

        assert!(
            error.to_string().contains("generation overflow"),
            "{error:#}"
        );
        assert_eq!(bank.snapshot_bytes().unwrap(), before);
        assert_eq!(bank.tier_clocks().unwrap(), clocks_before);
    }

    #[test]
    fn wake_only_updates_due_tiers_and_resumes_pending_accumulation_exactly() {
        let (config, device, model) = model();
        let schedule = schedule();
        let optimizer = TierOptimizerConfig::default();
        let source = TierOptimizerBank::new(&model, &schedule, optimizer.clone()).unwrap();
        let initial_slots = model.memory_slot_statuses();
        let initial_hash = crate::tensor_sleep::model_parameter_hash(&model).unwrap();

        accumulate(&source, &model, &device);
        let (after_one, first) = source.apply_wake_only_due_updates(&model, 1).unwrap();
        assert_eq!(first.updates.len(), 1);
        assert_eq!(first.updates[0].tier, 0);
        assert_eq!(first.updates[0].accumulated_optimizer_steps, 1);
        assert_ne!(
            crate::tensor_sleep::model_parameter_hash(&after_one).unwrap(),
            initial_hash,
            "the due fast tier did not update model parameters"
        );
        assert_eq!(after_one.memory_slot_statuses(), initial_slots);
        assert_eq!(source.tier_clocks().unwrap(), vec![(1, 0, 0), (0, 0, 1)]);

        let directory = TempDir::new().unwrap();
        let root = directory.path().join("wake-only-optimizers");
        let publisher = DurableTierOptimizerPublisher::new(
            source.clone(),
            &root,
            TensorTransactionStore::new(directory.path().join("unused-tensors")),
            config.clone(),
            device.clone(),
        )
        .unwrap();
        let scopes = publisher.publish_checkpoint_scopes().unwrap();
        assert_eq!(scopes.tiers[1].accumulated_micro_steps, 1);
        let checkpoint_snapshot = source.snapshot_bytes().unwrap();

        let resumed = TierOptimizerBank::new(&after_one, &schedule, optimizer).unwrap();
        let resumed_publisher = DurableTierOptimizerPublisher::new(
            resumed.clone(),
            &root,
            TensorTransactionStore::new(directory.path().join("unused-tensors")),
            config,
            device.clone(),
        )
        .unwrap();
        resumed_publisher
            .restore_scopes(&scopes, &after_one)
            .unwrap();
        assert_eq!(
            resumed.snapshot_bytes().unwrap(),
            checkpoint_snapshot,
            "wake_only optimizer moments, clocks, or pending gradients changed on resume"
        );

        accumulate(&source, &after_one, &device);
        accumulate(&resumed, &after_one, &device);
        let (source_after_two, source_report) =
            source.apply_wake_only_due_updates(&after_one, 2).unwrap();
        let (resumed_after_two, resumed_report) =
            resumed.apply_wake_only_due_updates(&after_one, 2).unwrap();
        assert_eq!(source_report, resumed_report);
        assert_eq!(
            source_report
                .updates
                .iter()
                .map(|update| (update.tier, update.accumulated_optimizer_steps))
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 2)],
            "coincident boundaries must update fastest-to-slowest and retain the slow pending window"
        );
        assert_eq!(
            crate::tensor_sleep::model_parameter_hash(&source_after_two).unwrap(),
            crate::tensor_sleep::model_parameter_hash(&resumed_after_two).unwrap(),
            "resumed wake_only update produced different parameters"
        );
        assert_eq!(
            source.snapshot_bytes().unwrap(),
            resumed.snapshot_bytes().unwrap()
        );
        assert_eq!(source.tier_clocks().unwrap(), vec![(2, 0, 0), (2, 0, 0)]);
        assert_eq!(source_after_two.memory_slot_statuses(), initial_slots);
    }

    #[cfg(unix)]
    #[test]
    fn tier_bundle_load_is_aba_safe_and_rejects_persistent_replacement() {
        let (config, device, model) = model();
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("optimizers");
        let source =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        accumulate(&source, &model, &device);
        let publisher = DurableTierOptimizerPublisher::new(
            source.clone(),
            root.clone(),
            TensorTransactionStore::new(directory.path().join("tensor")),
            config,
            device,
        )
        .unwrap();
        let scopes = publisher.publish_checkpoint_scopes().unwrap();
        let artifact = scopes.tiers[0].artifact.as_ref().unwrap().clone();
        let generation = Path::new(&artifact.state_uri).parent().unwrap().to_owned();
        let held = root.join("held-tier-generation");

        let restored =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        {
            let mut inner = restored.lock().unwrap();
            restore_tier_bundle_inner(&root, &mut inner, &artifact, |stage, path| {
                match stage {
                    TierBundleLoadStage::AfterCapture => {
                        fs::rename(path, &held)?;
                        fs::create_dir(path)?;
                    }
                    TierBundleLoadStage::AfterStagedLoad => {
                        fs::remove_dir(path)?;
                        fs::rename(&held, path)?;
                    }
                }
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(
            restored.tier_clocks().unwrap()[0],
            source.tier_clocks().unwrap()[0],
            "an A->B->A pathname swap changed the restored tier state"
        );

        let persistent =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        let before = persistent.snapshot_bytes().unwrap();
        let held = root.join("held-persistent-tier-generation");
        let result = {
            let mut inner = persistent.lock().unwrap();
            restore_tier_bundle_inner(&root, &mut inner, &artifact, |stage, path| {
                if stage == TierBundleLoadStage::AfterCapture {
                    fs::rename(path, &held)?;
                    fs::create_dir(path)?;
                }
                Ok(())
            })
        };
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("changed during authenticated load"),
            "{error}"
        );
        assert_eq!(persistent.snapshot_bytes().unwrap(), before);
        fs::remove_dir(&generation).unwrap();
        fs::rename(held, &generation).unwrap();

        let state_path = generation.join(STATE_FILE);
        let held_state = root.join("held-tier-state.json");
        let replacement =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        let before = replacement.snapshot_bytes().unwrap();
        let result = {
            let mut inner = replacement.lock().unwrap();
            restore_tier_bundle_inner(&root, &mut inner, &artifact, |stage, _| {
                if stage == TierBundleLoadStage::AfterCapture {
                    fs::rename(&state_path, &held_state)?;
                    fs::write(&state_path, b"persistent replacement")?;
                }
                Ok(())
            })
        };
        let error = result.unwrap_err().to_string();
        assert!(error.contains("file `state.json` changed"), "{error}");
        assert_eq!(replacement.snapshot_bytes().unwrap(), before);
        fs::remove_file(&state_path).unwrap();
        fs::rename(held_state, state_path).unwrap();

        let optimizer_path = generation.join(OPTIMIZER_FILE);
        let mutated =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        let before = mutated.snapshot_bytes().unwrap();
        let result = {
            let mut inner = mutated.lock().unwrap();
            restore_tier_bundle_inner(&root, &mut inner, &artifact, |stage, _| {
                if stage == TierBundleLoadStage::AfterCapture {
                    OpenOptions::new()
                        .append(true)
                        .open(&optimizer_path)?
                        .write_all(b"persistent mutation")?;
                }
                Ok(())
            })
        };
        let error = result.unwrap_err().to_string();
        assert!(error.contains("file `optimizer.bpk` changed"), "{error}");
        assert_eq!(mutated.snapshot_bytes().unwrap(), before);
    }

    #[test]
    fn prospective_preview_restores_exact_state_and_candidate_publication_is_idempotent() {
        let (_, device, model) = model();
        let directory = TempDir::new().unwrap();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        accumulate(&bank, &model, &device);
        let teacher = teacher_checkpoint(directory.path(), &model);
        let before = bank.snapshot_bytes().unwrap();
        let mut update =
            ProspectiveTierUpdate::new(bank.clone(), directory.path().join("prospective")).unwrap();
        let plan = update.prepare_consolidation(1, 0, 1, &teacher).unwrap();
        assert_eq!(before, bank.snapshot_bytes().unwrap());

        let (_, mut txn) = transaction(&model, plan);
        txn.teacher_checkpoint = teacher.uri.clone();
        txn.teacher_hash = teacher.sha256.clone();
        let candidate = update.stage(&txn, &model).unwrap();
        let repeated = update.stage(&txn, &model).unwrap();
        assert_eq!(candidate.checkpoint.uri, repeated.checkpoint.uri);
        assert_eq!(candidate.checkpoint.sha256, repeated.checkpoint.sha256);
        assert_eq!(candidate.update_sha256, repeated.update_sha256);

        let mut publisher =
            AtomicSafetensorsCandidatePublisher::new(directory.path().join("candidates")).unwrap();
        // Simulate a crash after the immutable generation is sealed but before
        // its transaction receipt exists.
        publish_immutable_model(
            &directory.path().join("candidates"),
            txn.id,
            &candidate.checkpoint.model,
        )
        .unwrap();
        let first = publisher
            .publish_candidate(&txn, &candidate.checkpoint.model)
            .unwrap();
        let second = publisher
            .publish_candidate(&txn, &candidate.checkpoint.model)
            .unwrap();
        assert_eq!(first.uri, second.uri);
        assert_eq!(first.sha256, second.sha256);
    }

    #[test]
    fn failed_prospective_preview_restores_a_consumed_sender_accumulator() {
        let (_, device, model) = model();
        let directory = TempDir::new().unwrap();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        accumulate(&bank, &model, &device);
        let teacher = teacher_checkpoint(directory.path(), &model);
        let root = directory.path().join("prospective-failure");
        let mut update = ProspectiveTierUpdate::new(bank.clone(), &root).unwrap();
        // Staging consumes the sender accumulator before publishing its
        // immutable candidate. Force that publication to fail after the
        // consume point and require byte-exact in-memory rollback.
        fs::write(root.join("model-generations"), b"not a directory").unwrap();
        let before = bank.snapshot_bytes().unwrap();

        assert!(update.prepare_consolidation(1, 0, 1, &teacher).is_err());
        assert_eq!(bank.snapshot_bytes().unwrap(), before);
        assert_eq!(bank.tier_clocks().unwrap(), vec![(0, 0, 1), (0, 0, 1)]);
    }

    #[test]
    fn prospective_checkpoint_load_uses_the_authenticated_bytes_after_path_replacement() {
        let (config, device, model) = model();
        let directory = TempDir::new().unwrap();
        let checkpoint = directory.path().join("prospective.safetensors");
        let held = directory.path().join("held-prospective.safetensors");
        let replacement = directory.path().join("replacement.safetensors");
        save_safetensors(&model.clone().valid(), &checkpoint).unwrap();
        let expected_sha256 = hash_file(&checkpoint).unwrap().1;
        let expected_parameters = crate::tensor_sleep::model_parameter_hash(&model).unwrap();

        device.seed(97);
        let other = Transformer::new(&config, &device).unwrap();
        assert_ne!(
            crate::tensor_sleep::model_parameter_hash(&other).unwrap(),
            expected_parameters
        );
        save_safetensors(&other.valid(), &replacement).unwrap();

        let loaded = load_local_checkpoint_inner(
            &model,
            checkpoint.to_str().unwrap(),
            &expected_sha256,
            |path| {
                fs::rename(path, &held)?;
                fs::rename(&replacement, path)?;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            crate::tensor_sleep::model_parameter_hash(&loaded).unwrap(),
            expected_parameters,
            "prospective checkpoint loader reopened a replaced pathname"
        );
    }

    fn run_tier_bundle_crash_scenario(sender_active: bool) {
        let (config, device, model) = model_with_fast_slot(sender_active);
        let directory = TempDir::new().unwrap();
        let bank =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        accumulate(&bank, &model, &device);
        let teacher = teacher_checkpoint(directory.path(), &model);
        let mut update =
            ProspectiveTierUpdate::new(bank.clone(), directory.path().join("prospective")).unwrap();
        let plan = update.prepare_consolidation(1, 0, 1, &teacher).unwrap();
        let (_, mut txn) = transaction(&model, plan);
        txn.teacher_checkpoint = teacher.uri.clone();
        txn.teacher_hash = teacher.sha256.clone();
        let candidate = update.stage(&txn, &model).unwrap();
        let pre = ProspectiveUpdateSnapshot::new(bank.snapshot_bytes().unwrap()).unwrap();

        let mut committed = candidate.checkpoint.model.clone();
        let receiver_ids = committed
            .activate_memory_slot_all_layers(txn.receiver, txn.receiver_slot)
            .unwrap();
        let reclaimed_ids = if txn.sender_slots_to_reset.is_empty() {
            Vec::new()
        } else {
            committed
                .reset_memory_slot_all_layers(txn.sender, 0, 91)
                .unwrap()
        };
        update
            .clear_reclaimed_optimizer_state(&txn, &reclaimed_ids)
            .unwrap();
        let staged = ProspectiveUpdateSnapshot::new(bank.snapshot_bytes().unwrap()).unwrap();

        let diagnostics = TensorSleepDiagnostics {
            knowledge_updates: 1,
            knowledge_tokens: 4,
            selected_gradient_tensors: receiver_ids.len(),
            imitation_updates: 1,
            imitation_samples: 2,
            ..TensorSleepDiagnostics::default()
        };
        let tensor_store = TensorTransactionStore::new(directory.path().join("tensor"));
        let recovered = RecoveredTensorTransaction {
            txn_id: txn.id,
            teacher: teacher.clone(),
            student: ProspectiveTransformerCandidate {
                checkpoint: ImmutableTransformerCheckpoint {
                    uri: candidate.checkpoint.uri.clone(),
                    sha256: candidate.checkpoint.sha256.clone(),
                    model: committed.clone(),
                },
                update_sha256: candidate.update_sha256.clone(),
            },
            pre_update_state: pre,
            staged_update_state: staged,
            receiver_optimizer: TierOptimizerConfig::default().optimizer(),
            completed: [
                TensorStage::Prospective,
                TensorStage::Student,
                TensorStage::Knowledge,
                TensorStage::Imitation,
                TensorStage::Retention,
                TensorStage::Committed,
            ]
            .into_iter()
            .collect(),
            retention_result: Some(true),
            diagnostics,
        };
        let pointer = tensor_store.publish(&txn, &recovered).unwrap();
        let receiver_ids = receiver_ids
            .into_iter()
            .map(|id| id.val())
            .collect::<Vec<_>>();
        let before_publish = bank.snapshot_bytes().unwrap();
        let before_publish_layout = bank.lock().unwrap().model_layout.memory_slot_statuses();
        let mut publisher = DurableTierOptimizerPublisher::new(
            bank.clone(),
            directory.path().join("optimizers"),
            tensor_store.clone(),
            config.clone(),
            device.clone(),
        )
        .unwrap();
        publisher.inject_failure_after_bundle(1);
        assert!(publisher.publish(&txn, &pointer, &receiver_ids).is_err());
        assert_eq!(before_publish, bank.snapshot_bytes().unwrap());
        assert_eq!(
            before_publish_layout,
            bank.lock().unwrap().model_layout.memory_slot_statuses()
        );
        publisher.clear_failure_injection();
        let commits = publisher.publish(&txn, &pointer, &receiver_ids).unwrap();
        assert_eq!(commits.len(), 2);
        assert_eq!(bank.tier_clocks().unwrap(), vec![(1, 0, 0), (0, 1, 1)]);

        let dormant = dormant_parameter_ids(&committed)
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        assert!(commits.iter().all(|commit| {
            commit
                .artifact
                .optimizer_parameter_ids
                .iter()
                .all(|id| !dormant.contains(id))
        }));

        // Recreate from the exact pre-publication snapshot. The durable txn
        // receipt must install both committed bundles without reapplying a step.
        let fresh =
            TierOptimizerBank::new(&model, &schedule(), TierOptimizerConfig::default()).unwrap();
        fresh.restore_bytes(&before_publish).unwrap();
        let mut resumed = DurableTierOptimizerPublisher::new(
            fresh.clone(),
            directory.path().join("optimizers"),
            tensor_store,
            config.clone(),
            device,
        )
        .unwrap();
        let repeated = resumed.publish(&txn, &pointer, &receiver_ids).unwrap();
        assert_eq!(repeated, commits);
        assert_eq!(fresh.tier_clocks().unwrap(), bank.tier_clocks().unwrap());

        let scopes = bank.scopes().unwrap();
        let restored =
            TierOptimizerBank::new(&committed, &schedule(), TierOptimizerConfig::default())
                .unwrap();
        let restorer = DurableTierOptimizerPublisher::new(
            restored.clone(),
            directory.path().join("optimizers"),
            TensorTransactionStore::new(directory.path().join("tensor")),
            config,
            Device::ndarray().autodiff(),
        )
        .unwrap();
        restorer.restore_scopes(&scopes, &committed).unwrap();
        assert_eq!(restored.tier_clocks().unwrap(), bank.tier_clocks().unwrap());
    }

    #[test]
    fn tier_bundle_crash_rolls_back_and_receipt_recovers_on_fresh_bank() {
        run_tier_bundle_crash_scenario(true);
    }

    #[test]
    fn first_all_dormant_boundary_commits_without_fake_reclamation() {
        run_tier_bundle_crash_scenario(false);
    }
}
