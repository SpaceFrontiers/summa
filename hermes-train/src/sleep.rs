//! Trainer-controlled, model-internal memory consolidation and dreaming.
//!
//! This module deliberately separates the lifecycle transaction from the
//! tensor implementation.  A backend may use Muon/AdamW, LoRA, GRPO, or a
//! future optimizer, but it must obey the same snapshot, stage, validate,
//! commit, and rollback ordering.  That makes interrupted sleep recoverable
//! and prevents a rejected candidate from mutating the wake checkpoint.

use std::collections::BTreeSet;

#[cfg(test)]
use std::cell::Cell;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{ParamId, list_param_ids};
use hermes_llm::{MemorySlotStatus, Transformer};
use serde::{Deserialize, Serialize};

use crate::artifact_io::validate_sha256_identity;
use crate::metrics::{DreamSelectionMetrics, DreamTrialMetrics, MetricEvent};

pub const MAX_MEMORY_TIERS: usize = 16;
pub const MAX_RESERVE_SLOTS_PER_TIER: usize = 4_096;
pub const MAX_TOTAL_RESERVE_SLOTS: usize = 16_384;
pub const MAX_SLEEP_RNG_STREAMS: usize = 1_024;
const MAX_MEMORY_TIER_ID_BYTES: usize = 128;
const MAX_CHECKPOINT_LOCATION_BYTES: usize = 16 * 1_024;
const MAX_SLEEP_EVALUATOR_HASHES: usize = 32;
const MAX_DREAM_CANDIDATE_ID_BYTES: usize = 1_024;
const MAX_DREAM_GRADIENT_DIMENSIONS: usize = 4_096;

#[cfg(test)]
thread_local! {
    static FULL_SCOPE_VALIDATIONS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_full_scope_validation_count() {
    FULL_SCOPE_VALIDATIONS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn full_scope_validation_count() -> u64 {
    FULL_SCOPE_VALIDATIONS.with(Cell::get)
}

/// Unit used by a memory tier's update period.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateClock {
    OptimizerSteps,
    ModelTokens,
}

/// Bounded final-tier behavior. Versioning the algorithm prevents a future
/// implementation from silently changing what capacity reclamation means.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalConsolidation {
    /// Distill every active terminal reserve into that tier's base FFN/MoE,
    /// apply the slow prospective base update, and reclaim reserves only after
    /// the ordinary imitation and retention gates pass.
    DistillIntoBaseV1,
}

/// One fast-to-slow parameter-memory tier.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryTierSchedule {
    pub id: String,
    pub update_period: u64,
    pub reserve_slots: usize,
}

/// Ordered memory schedule.  Index zero is the fastest tier.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SleepSchedule {
    pub clock: UpdateClock,
    pub terminal_consolidation: TerminalConsolidation,
    pub tiers: Vec<MemoryTierSchedule>,
}

impl SleepSchedule {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=MAX_MEMORY_TIERS).contains(&self.tiers.len()),
            "sleep requires 2..={MAX_MEMORY_TIERS} memory tiers"
        );
        let mut ids = BTreeSet::new();
        let mut total_reserve_slots = 0usize;
        for (index, tier) in self.tiers.iter().enumerate() {
            ensure!(
                !tier.id.trim().is_empty() && tier.id.len() <= MAX_MEMORY_TIER_ID_BYTES,
                "memory tier {index} id must contain 1..={MAX_MEMORY_TIER_ID_BYTES} bytes"
            );
            ensure!(
                ids.insert(tier.id.as_str()),
                "duplicate memory tier id `{}`",
                tier.id
            );
            ensure!(
                tier.update_period > 0,
                "memory tier `{}` update_period must be positive",
                tier.id
            );
            ensure!(
                (1..=MAX_RESERVE_SLOTS_PER_TIER).contains(&tier.reserve_slots),
                "memory tier `{}` reserve_slots must be in 1..={MAX_RESERVE_SLOTS_PER_TIER}",
                tier.id
            );
            total_reserve_slots = total_reserve_slots
                .checked_add(tier.reserve_slots)
                .context("sleep reserve-slot count overflow")?;
            ensure!(
                total_reserve_slots <= MAX_TOTAL_RESERVE_SLOTS,
                "sleep schedule exceeds the {MAX_TOTAL_RESERVE_SLOTS}-slot total reserve limit"
            );
            if let Some(faster) = index.checked_sub(1).map(|i| &self.tiers[i]) {
                ensure!(
                    tier.update_period > faster.update_period,
                    "memory tier `{}` must update less often than `{}`",
                    tier.id,
                    faster.id
                );
                ensure!(
                    tier.update_period % faster.update_period == 0,
                    "memory tier `{}` period {} must be divisible by faster tier `{}` period {}",
                    tier.id,
                    tier.update_period,
                    faster.id,
                    faster.update_period
                );
                let transfers_per_boundary = tier.update_period / faster.update_period;
                ensure!(
                    tier.reserve_slots as u64 >= transfers_per_boundary,
                    "memory tier `{}` needs at least {transfers_per_boundary} reserve slots to receive every `{}` boundary before reclamation",
                    tier.id,
                    faster.id
                );
            }
        }
        Ok(())
    }

    /// Return sender indexes due at this clock, fastest first. The terminal
    /// tier runs last and consolidates its bounded reserve into its base.
    pub fn due_senders(&self, clock: u64) -> Vec<usize> {
        if clock == 0 {
            return Vec::new();
        }
        self.tiers
            .iter()
            .enumerate()
            .filter_map(|(index, tier)| clock.is_multiple_of(tier.update_period).then_some(index))
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReserveSlotState {
    pub active: bool,
    /// Increased on both activation and reclamation so stale optimizer state
    /// cannot be mistaken for the current incarnation of a slot.
    pub generation: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryTierState {
    pub id: String,
    pub last_update_clock: u64,
    pub last_boundary_clock: u64,
    pub slots: Vec<ReserveSlotState>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepPhase {
    Wake,
    ProspectiveUpdate,
    KnowledgeSeeding,
    Imitation,
    RetentionValidation,
    Commit,
    DreamGeneration,
    DreamRanking,
    DreamTrials,
    DreamPolicyUpdate,
    Candidate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RngReservation {
    pub stream: usize,
    pub start: u64,
    pub count: u64,
}

impl RngReservation {
    fn validate(&self, counters: &[u64]) -> Result<()> {
        ensure!(self.count > 0, "sleep RNG reservation is empty");
        let end = self
            .start
            .checked_add(self.count)
            .context("sleep RNG range overflow")?;
        ensure!(
            counters
                .get(self.stream)
                .is_some_and(|counter| end <= *counter),
            "sleep RNG reservation exceeds its persisted stream counter"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DreamTrialRng {
    pub candidate_id: String,
    pub reservation: RngReservation,
}

/// Durable identity and mutation plan for one consolidation boundary.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConsolidationTxn {
    pub id: u64,
    pub trigger_clock: u64,
    pub sender: usize,
    pub receiver: usize,
    pub receiver_slot: usize,
    /// True for the versioned final-tier reserve-to-base transaction. In this
    /// case `receiver == sender` and `receiver_slot` is unused.
    pub terminal: bool,
    pub sender_slots_to_reset: Vec<usize>,
    pub teacher_checkpoint: String,
    pub teacher_hash: String,
    pub student_checkpoint: String,
    pub student_hash: String,
    pub prospective_update_hash: String,
    /// Immutable candidate published by the tensor backend at commit.  This
    /// is recorded before the metadata commit is made durable, so a process
    /// that resumes in Dreaming/Candidate can recover the exact accepted
    /// weights instead of guessing from the prospective (pre-distillation)
    /// student identity.
    pub candidate_checkpoint: Option<String>,
    pub candidate_hash: Option<String>,
    pub knowledge_rng: Option<RngReservation>,
    pub imitation_rng: Option<RngReservation>,
    pub dream_generation_rng: Option<RngReservation>,
    pub dream_selection_rng: Option<RngReservation>,
    pub dream_trial_rngs: Vec<DreamTrialRng>,
    pub tensor_transaction_generation: Option<String>,
    pub tensor_transaction_manifest_hash: Option<String>,
    pub generated_manifest: Option<String>,
    pub dream_shared_checkpoint_hash: Option<String>,
    pub dream_selected: Vec<String>,
    pub dream_trials: Vec<DreamTrial>,
    /// Content-addressed ReSTEM policy state. The policy updater must be
    /// transaction-idempotent and return the same receipt after interruption.
    pub dream_policy_receipt: Option<String>,
    /// Makes replay of the commit subphase detectable. Tensor backends must
    /// likewise key their commit by transaction id and make it idempotent.
    pub committed: bool,
}

/// Checkpointed sleep lifecycle.  Tensor and optimizer state stay in their
/// normal checkpoint files; this records the exact semantic transaction.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SleepState {
    /// Highest consolidation attempt ID allocated so far. Successful and
    /// rolled-back attempts both consume an ID so immutable receipts from a
    /// rejected attempt can never alias a later transaction.
    pub cycle: u64,
    pub clock: u64,
    pub phase: SleepPhase,
    pub tiers: Vec<MemoryTierState>,
    /// Senders due at the current clock, persisted fastest-to-slowest. A
    /// boundary is removed only after commit or fully restored rollback.
    pub due_senders: Vec<usize>,
    /// Trigger clock paired one-to-one with `due_senders`. This preserves all
    /// boundaries crossed by a coarse optimizer-step/model-token advance.
    pub due_clocks: Vec<u64>,
    pub pending: Option<ConsolidationTxn>,
    pub rng_counters: Vec<u64>,
    pub evaluator_hashes: Vec<String>,
    pub artifact_manifests: Vec<String>,
    /// Bounded tail of immutable completed transactions. `completed_chain_hash`
    /// commits to the complete history, including entries compacted from this
    /// tail, so audit evidence is never silently dropped.
    pub completed_transactions: Vec<ConsolidationTxn>,
    pub completed_count: u64,
    pub completed_chain_hash: String,
}

const COMPLETED_TRANSACTION_TAIL: usize = 64;
const ARTIFACT_MANIFEST_TAIL: usize = COMPLETED_TRANSACTION_TAIL + 1;

fn referenced_artifact_manifests(
    completed: &[ConsolidationTxn],
    pending: Option<&ConsolidationTxn>,
) -> Result<Vec<String>> {
    let mut manifests = BTreeSet::new();
    for manifest in completed
        .iter()
        .chain(pending)
        .filter_map(|txn| txn.generated_manifest.as_ref())
    {
        validate_sha256_identity(manifest, "sleep artifact manifest")?;
        manifests.insert(manifest.clone());
    }
    ensure!(
        manifests.len() <= ARTIFACT_MANIFEST_TAIL,
        "sleep artifact-manifest tail exceeds its bounded transaction history"
    );
    Ok(manifests.into_iter().collect())
}

fn validate_completed_transaction(
    txn: &ConsolidationTxn,
    tiers: &[MemoryTierState],
    rng_counters: &[u64],
) -> Result<()> {
    ensure!(
        txn.committed,
        "completed sleep transaction is not committed"
    );
    ensure!(
        txn.sender < tiers.len() && txn.receiver < tiers.len(),
        "completed sleep transaction tier is out of range"
    );
    ensure!(
        txn.trigger_clock > 0 && txn.trigger_clock <= tiers[txn.sender].last_update_clock,
        "completed sleep transaction trigger is ahead of its sender update clock"
    );
    ensure!(
        txn.terminal == (txn.sender + 1 == tiers.len())
            && ((txn.terminal && txn.receiver == txn.sender)
                || (!txn.terminal && txn.sender + 1 == txn.receiver)),
        "completed sleep transaction has invalid tier topology"
    );
    ensure!(
        txn.terminal || txn.receiver_slot < tiers[txn.receiver].slots.len(),
        "completed sleep transaction receiver slot is out of range"
    );
    ensure!(
        txn.sender_slots_to_reset
            .iter()
            .all(|slot| *slot < tiers[txn.sender].slots.len())
            && txn
                .sender_slots_to_reset
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
        "completed sleep transaction has an invalid sender reset plan"
    );
    ensure!(
        !txn.teacher_checkpoint.trim().is_empty()
            && txn.teacher_checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES
            && !txn.student_checkpoint.trim().is_empty()
            && txn.student_checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES,
        "completed sleep transaction has invalid checkpoint locations"
    );
    for (hash, label) in [
        (&txn.teacher_hash, "completed teacher hash"),
        (&txn.student_hash, "completed student hash"),
        (
            &txn.prospective_update_hash,
            "completed prospective-update hash",
        ),
    ] {
        validate_sha256_identity(hash, label)?;
    }
    let candidate_checkpoint = txn
        .candidate_checkpoint
        .as_deref()
        .context("completed sleep transaction has no candidate checkpoint")?;
    ensure!(
        !candidate_checkpoint.trim().is_empty()
            && candidate_checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES,
        "completed sleep transaction has an invalid candidate checkpoint"
    );
    validate_sha256_identity(
        txn.candidate_hash
            .as_deref()
            .context("completed sleep transaction has no candidate hash")?,
        "completed candidate hash",
    )?;

    let mut reservations = [
        txn.knowledge_rng
            .context("completed sleep transaction has no Knowledge Seeding RNG receipt")?,
        txn.imitation_rng
            .context("completed sleep transaction has no imitation RNG receipt")?,
    ]
    .into_iter()
    .chain(txn.dream_generation_rng)
    .chain(txn.dream_selection_rng)
    .chain(txn.dream_trial_rngs.iter().map(|trial| trial.reservation))
    .collect::<Vec<_>>();
    for reservation in &reservations {
        reservation.validate(rng_counters)?;
    }
    reservations.sort_unstable_by_key(|reservation| (reservation.stream, reservation.start));
    ensure!(
        reservations.windows(2).all(|pair| {
            pair[0].stream != pair[1].stream || pair[0].start + pair[0].count <= pair[1].start
        }),
        "completed sleep transaction has overlapping RNG receipts"
    );

    ensure!(
        txn.tensor_transaction_generation.is_some()
            == txn.tensor_transaction_manifest_hash.is_some(),
        "completed sleep tensor transaction identity is incomplete"
    );
    if let (Some(generation), Some(hash)) = (
        &txn.tensor_transaction_generation,
        &txn.tensor_transaction_manifest_hash,
    ) {
        let digest = generation
            .strip_prefix("sha256-")
            .context("completed tensor generation has an unsafe name")?;
        validate_sha256_identity(&format!("sha256:{digest}"), "completed tensor generation")?;
        validate_sha256_identity(hash, "completed tensor manifest hash")?;
        ensure!(
            hash.strip_prefix("sha256:") == Some(digest),
            "completed tensor generation differs from its manifest"
        );
    }

    let selected = txn
        .dream_selected
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let trial_ids = txn
        .dream_trials
        .iter()
        .map(|trial| trial.candidate_id.as_str())
        .collect::<BTreeSet<_>>();
    let trial_rng_ids = txn
        .dream_trial_rngs
        .iter()
        .map(|trial| trial.candidate_id.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        selected.len() == txn.dream_selected.len()
            && selected
                .iter()
                .all(|id| { !id.trim().is_empty() && id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES })
            && trial_ids.len() == txn.dream_trials.len()
            && trial_rng_ids.len() == txn.dream_trial_rngs.len(),
        "completed sleep transaction has invalid dream candidate IDs"
    );
    for trial in &txn.dream_trials {
        ensure!(
            trial.candidate_id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES
                && trial.independent_task_improvement.is_finite(),
            "completed dream trial has invalid evidence"
        );
        validate_sha256_identity(&trial.adapter_hash, "completed dream adapter hash")?;
        validate_sha256_identity(&trial.evaluator_hash, "completed dream evaluator hash")?;
    }
    match &txn.generated_manifest {
        Some(manifest) => {
            validate_sha256_identity(manifest, "completed dream manifest")?;
            validate_sha256_identity(
                txn.dream_shared_checkpoint_hash
                    .as_deref()
                    .context("completed dream has no shared-checkpoint hash")?,
                "completed dream shared-checkpoint hash",
            )?;
            let candidate_count = usize::try_from(
                txn.dream_generation_rng
                    .context("completed dream has no generation RNG receipt")?
                    .count,
            )
            .context("completed dream candidate count exceeds usize")?;
            ensure!(
                txn.dream_selection_rng.is_some()
                    && !selected.is_empty()
                    && selected.len() <= candidate_count
                    && trial_ids == selected
                    && trial_rng_ids == selected,
                "completed dream selection/trial evidence is incomplete"
            );
            validate_sha256_identity(
                txn.dream_policy_receipt
                    .as_deref()
                    .context("completed dream has no ReSTEM policy receipt")?,
                "completed ReSTEM policy receipt",
            )?;
        }
        None => ensure!(
            txn.dream_generation_rng.is_none()
                && txn.dream_selection_rng.is_none()
                && txn.dream_trial_rngs.is_empty()
                && txn.dream_shared_checkpoint_hash.is_none()
                && txn.dream_selected.is_empty()
                && txn.dream_trials.is_empty()
                && txn.dream_policy_receipt.is_none(),
            "completed transaction contains partial Dreaming evidence"
        ),
    }
    Ok(())
}

/// Content-addressed optimizer/accumulator snapshot for one memory tier.
/// The tensors live in the normal candidate checkpoint store; this receipt is
/// what makes independently clocked tier state part of deterministic resume.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TierOptimizerArtifact {
    /// Immutable transaction bundle containing the tier optimizer moments,
    /// its zeroed post-boundary gradient accumulator, and parameter IDs.
    pub state_uri: String,
    pub manifest_hash: String,
    /// Parameter IDs for which optimizer moments exist in the bundle.
    pub optimizer_parameter_ids: Vec<u64>,
    /// Parameter IDs with pending accumulated gradients. This is normally
    /// empty immediately after a committed tier boundary.
    pub accumulator_parameter_ids: Vec<u64>,
}

impl TierOptimizerArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.state_uri.trim().is_empty(),
            "tier optimizer artifact has an empty URI"
        );
        validate_sha256_identity(&self.manifest_hash, "tier optimizer-state manifest hash")
            .context("invalid tier optimizer-state bundle")?;
        for (ids, label) in [
            (&self.optimizer_parameter_ids, "optimizer"),
            (&self.accumulator_parameter_ids, "gradient accumulator"),
        ] {
            let unique = ids.iter().copied().collect::<BTreeSet<_>>();
            ensure!(
                unique.len() == ids.len(),
                "tier {label} state repeats a parameter ID"
            );
        }
        Ok(())
    }

    /// Bind the durable gradient receipt to its independently persisted
    /// accumulation clock. Empty gradient bytes with a positive counter would
    /// silently shorten the next tier update; gradients with a zero counter
    /// would be divided by an invalid wake window.
    pub fn validate_pending_steps(&self, accumulated_micro_steps: u64) -> Result<()> {
        self.validate()?;
        ensure!(
            self.accumulator_parameter_ids.is_empty() == (accumulated_micro_steps == 0),
            "tier optimizer accumulator receipt disagrees with its pending-step counter"
        );
        Ok(())
    }
}

/// Durable, independent clock and optimizer namespace for one MAL memory
/// tier. Parameter IDs are authenticated so a wake optimizer cannot silently
/// absorb fast/medium/slow parameters on resume.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TierOptimizerScope {
    pub tier: usize,
    pub tier_id: String,
    pub parameter_ids: Vec<u64>,
    /// Last boundary at which this tier acted as a sender and consumed its
    /// accumulated wake gradients.
    pub update_clock: u64,
    /// Last boundary at which a faster tier transferred into this tier. A
    /// transfer must not consume this tier's independently accumulated wake
    /// gradients or advance its sender-update clock.
    pub transfer_clock: u64,
    pub accumulated_micro_steps: u64,
    /// Generation of the complete optimizer/accumulator artifact. This
    /// advances for wake accumulation, sender updates, and receiver transfers.
    pub generation: u64,
    pub transfer_generation: u64,
    pub artifact: Option<TierOptimizerArtifact>,
}

/// First-party parameter partition used by wake and sleep execution. The wake
/// scope is the exact complement of all memory-tier parameters and every tier
/// owns a separate optimizer/gradient-accumulator receipt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryOptimizerScopes {
    pub wake_parameter_ids: Vec<u64>,
    pub tiers: Vec<TierOptimizerScope>,
}

impl MemoryOptimizerScopes {
    pub fn from_model(model: &Transformer, schedule: &SleepSchedule) -> Result<Self> {
        schedule.validate()?;
        let all = list_param_ids(model)
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        ensure!(!all.is_empty(), "model exposes no parameter IDs");
        let mut owned = BTreeSet::new();
        let mut tiers = Vec::with_capacity(schedule.tiers.len());
        for (tier, configured) in schedule.tiers.iter().enumerate() {
            let mut ids = model
                .memory_tier_parameter_ids_all_layers(tier)
                .with_context(|| format!("selecting memory tier `{}` parameters", configured.id))?
                .into_iter()
                .map(|id| id.val())
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ensure!(
                ids.windows(2).all(|pair| pair[0] != pair[1]),
                "memory tier `{}` repeats a parameter ID",
                configured.id
            );
            for id in &ids {
                ensure!(
                    all.contains(id),
                    "memory tier `{}` contains a parameter absent from the model",
                    configured.id
                );
                ensure!(
                    owned.insert(*id),
                    "parameter {id} belongs to more than one memory tier"
                );
            }
            tiers.push(TierOptimizerScope {
                tier,
                tier_id: configured.id.clone(),
                parameter_ids: ids,
                update_clock: 0,
                transfer_clock: 0,
                accumulated_micro_steps: 0,
                generation: 0,
                transfer_generation: 0,
                artifact: None,
            });
        }
        let wake_parameter_ids = all.difference(&owned).copied().collect::<Vec<_>>();
        let scopes = Self {
            wake_parameter_ids,
            tiers,
        };
        scopes.validate(model, schedule)?;
        Ok(scopes)
    }

    pub fn validate(&self, model: &Transformer, schedule: &SleepSchedule) -> Result<()> {
        #[cfg(test)]
        FULL_SCOPE_VALIDATIONS.with(|count| count.set(count.get().saturating_add(1)));
        schedule.validate()?;
        ensure!(
            self.tiers.len() == schedule.tiers.len(),
            "optimizer tier count differs from sleep schedule"
        );
        let all = list_param_ids(model)
            .into_iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        let wake = self
            .wake_parameter_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        ensure!(
            wake.len() == self.wake_parameter_ids.len(),
            "wake optimizer repeats a parameter ID"
        );
        ensure!(
            wake.iter().all(|id| all.contains(id)),
            "wake optimizer contains a parameter absent from the model"
        );
        let mut memory = BTreeSet::new();
        for (tier, (scope, configured)) in self.tiers.iter().zip(&schedule.tiers).enumerate() {
            ensure!(
                scope.tier == tier && scope.tier_id == configured.id,
                "optimizer scope {tier} differs from MAL/schedule tier `{}`",
                configured.id
            );
            // A slower tier can receive a consolidation transfer at a fast
            // tier boundary. Its optimizer clock is therefore monotonic but
            // need not be a multiple of its own sender update period.
            let expected = model
                .memory_tier_parameter_ids_all_layers(tier)?
                .into_iter()
                .map(|id| id.val())
                .collect::<BTreeSet<_>>();
            let observed = scope.parameter_ids.iter().copied().collect::<BTreeSet<_>>();
            ensure!(
                observed.len() == scope.parameter_ids.len() && observed == expected,
                "optimizer tier `{}` parameter scope differs from the model",
                scope.tier_id
            );
            ensure!(
                scope.update_clock.is_multiple_of(configured.update_period),
                "optimizer tier `{}` sender-update clock is off schedule",
                scope.tier_id
            );
            if tier == 0 {
                ensure!(
                    scope.transfer_clock == 0 && scope.transfer_generation == 0,
                    "fastest optimizer tier cannot contain a receiver-transfer receipt"
                );
            } else {
                ensure!(
                    scope
                        .transfer_clock
                        .is_multiple_of(schedule.tiers[tier - 1].update_period),
                    "optimizer tier `{}` receiver-transfer clock is off schedule",
                    scope.tier_id
                );
            }
            ensure!(
                scope.transfer_generation <= scope.generation,
                "optimizer tier `{}` transfer generation exceeds its artifact generation",
                scope.tier_id
            );
            ensure!(
                scope.artifact.is_some() == (scope.generation > 0),
                "optimizer tier `{}` artifact presence disagrees with its generation",
                scope.tier_id
            );
            ensure!(
                wake.is_disjoint(&observed),
                "wake optimizer contains parameters from memory tier `{}`",
                scope.tier_id
            );
            ensure!(
                observed.iter().all(|id| memory.insert(*id)),
                "memory optimizer scopes overlap"
            );
            if let Some(artifact) = &scope.artifact {
                artifact.validate()?;
            }
        }
        ensure!(
            wake.union(&memory).copied().collect::<BTreeSet<_>>() == all,
            "wake and memory optimizer scopes do not partition model parameters"
        );
        Ok(())
    }

    /// Reject optimizer moments or accumulated gradients for dormant reserve
    /// experts. Base parameters are always eligible; reserve parameters become
    /// eligible only when every memory-bearing layer agrees the logical slot
    /// is active in [`SleepState`].
    pub fn validate_active_state(
        &self,
        model: &Transformer,
        schedule: &SleepSchedule,
        state: &SleepState,
    ) -> Result<()> {
        let statuses = model.memory_slot_statuses();
        validate_model_memory_state(schedule, state, &statuses)?;
        self.validate_active_state_after_model_validation(model, schedule, &statuses)
    }

    /// Validate optimizer ownership after the caller has already reconciled
    /// the same model and `SleepState`. This avoids repeating the full bounded
    /// transaction-history and slot-topology walk on every wake clock tick.
    pub(crate) fn validate_active_state_after_model_validation(
        &self,
        model: &Transformer,
        schedule: &SleepSchedule,
        statuses: &[MemorySlotStatus],
    ) -> Result<()> {
        self.validate(model, schedule)?;
        for scope in &self.tiers {
            let Some(artifact) = &scope.artifact else {
                continue;
            };
            let mut allowed = model
                .memory_tier_base_parameter_ids_all_layers(scope.tier)?
                .into_iter()
                .map(|id| id.val())
                .collect::<BTreeSet<_>>();
            for status in statuses
                .iter()
                .filter(|status| status.tier == scope.tier && status.active)
            {
                allowed.extend(status.parameter_ids.iter().map(|id| id.val()));
            }
            for (ids, label) in [
                (&artifact.optimizer_parameter_ids, "optimizer moment"),
                (&artifact.accumulator_parameter_ids, "accumulated gradient"),
            ] {
                ensure!(
                    ids.iter().all(|id| allowed.contains(id)),
                    "tier `{}` {label} state contains a dormant or foreign parameter",
                    scope.tier_id
                );
            }
            artifact
                .validate_pending_steps(scope.accumulated_micro_steps)
                .with_context(|| {
                    format!(
                        "tier `{}` accumulator receipt disagrees with its micro-step counter",
                        scope.tier_id
                    )
                })?;
        }
        Ok(())
    }

    pub fn wake_parameter_ids(&self) -> Vec<ParamId> {
        self.wake_parameter_ids
            .iter()
            .copied()
            .map(ParamId::from)
            .collect()
    }

    pub fn tier_parameter_ids(&self, tier: usize) -> Result<Vec<ParamId>> {
        Ok(self
            .tiers
            .get(tier)
            .with_context(|| format!("optimizer tier {tier} does not exist"))?
            .parameter_ids
            .iter()
            .copied()
            .map(ParamId::from)
            .collect())
    }

    /// Publish a durable accumulator generation after wake microbatches. The
    /// artifact must contain the pending gradient tensors; a counter without
    /// the corresponding immutable state is not resumable.
    pub fn record_accumulation(
        &mut self,
        tier: usize,
        micro_steps: u64,
        artifact: TierOptimizerArtifact,
    ) -> Result<()> {
        ensure!(
            micro_steps > 0,
            "tier accumulation increment must be positive"
        );
        artifact.validate()?;
        ensure!(
            !artifact.accumulator_parameter_ids.is_empty(),
            "tier accumulation artifact contains no pending gradients"
        );
        let scope = self
            .tiers
            .get_mut(tier)
            .with_context(|| format!("optimizer tier {tier} does not exist"))?;
        scope.accumulated_micro_steps = scope
            .accumulated_micro_steps
            .checked_add(micro_steps)
            .context("tier gradient-accumulator clock overflow")?;
        scope.generation = scope
            .generation
            .checked_add(1)
            .context("tier generation overflow")?;
        scope.artifact = Some(artifact);
        Ok(())
    }

    /// Commit a prospective base update for a tier acting as sender. This is
    /// the only sleep operation that consumes and clears that tier's wake
    /// accumulator.
    pub fn commit_sender_update(
        &mut self,
        schedule: &SleepSchedule,
        tier: usize,
        clock: u64,
        artifact: TierOptimizerArtifact,
    ) -> Result<()> {
        artifact.validate()?;
        let configured = schedule
            .tiers
            .get(tier)
            .with_context(|| format!("optimizer tier {tier} is absent from schedule"))?;
        ensure!(
            clock > 0 && clock.is_multiple_of(configured.update_period),
            "optimizer tier `{}` update clock is zero or off schedule",
            configured.id
        );
        let scope = self
            .tiers
            .get_mut(tier)
            .with_context(|| format!("optimizer tier {tier} does not exist"))?;
        ensure!(
            clock > scope.update_clock,
            "optimizer tier `{}` sender update was replayed or moved backwards",
            scope.tier_id
        );
        ensure!(
            artifact.accumulator_parameter_ids.is_empty(),
            "sender optimizer commit retains accumulated gradients"
        );
        scope.update_clock = clock;
        scope.accumulated_micro_steps = 0;
        scope.generation = scope
            .generation
            .checked_add(1)
            .context("tier generation overflow")?;
        scope.artifact = Some(artifact);
        Ok(())
    }

    /// Commit a faster-tier transfer into this receiver while preserving the
    /// receiver's independent wake accumulator and sender-update clock.
    pub fn commit_receiver_transfer(
        &mut self,
        tier: usize,
        clock: u64,
        artifact: TierOptimizerArtifact,
    ) -> Result<()> {
        artifact.validate()?;
        ensure!(clock > 0, "receiver-transfer clock is zero");
        let scope = self
            .tiers
            .get_mut(tier)
            .with_context(|| format!("optimizer tier {tier} does not exist"))?;
        ensure!(
            clock > scope.transfer_clock,
            "optimizer tier `{}` receiver transfer was replayed or moved backwards",
            scope.tier_id
        );
        if scope.accumulated_micro_steps == 0 {
            ensure!(
                artifact.accumulator_parameter_ids.is_empty(),
                "receiver transfer introduced gradients without accumulated microsteps"
            );
        } else {
            let previous = scope
                .artifact
                .as_ref()
                .context("receiver has accumulated microsteps without a durable artifact")?;
            ensure!(
                !artifact.accumulator_parameter_ids.is_empty()
                    && artifact.accumulator_parameter_ids == previous.accumulator_parameter_ids,
                "receiver transfer did not preserve the exact pending gradient accumulator"
            );
        }
        scope.transfer_clock = clock;
        scope.transfer_generation = scope
            .transfer_generation
            .checked_add(1)
            .context("tier transfer generation overflow")?;
        scope.generation = scope
            .generation
            .checked_add(1)
            .context("tier generation overflow")?;
        scope.artifact = Some(artifact);
        Ok(())
    }
}

impl SleepState {
    pub fn new(schedule: &SleepSchedule, rng_streams: usize) -> Result<Self> {
        schedule.validate()?;
        ensure!(
            (1..=MAX_SLEEP_RNG_STREAMS).contains(&rng_streams),
            "sleep requires 1..={MAX_SLEEP_RNG_STREAMS} RNG streams"
        );
        Ok(Self {
            cycle: 0,
            clock: 0,
            phase: SleepPhase::Wake,
            tiers: schedule
                .tiers
                .iter()
                .map(|tier| MemoryTierState {
                    id: tier.id.clone(),
                    last_update_clock: 0,
                    last_boundary_clock: 0,
                    slots: (0..tier.reserve_slots)
                        .map(|_| ReserveSlotState {
                            active: false,
                            generation: 0,
                        })
                        .collect(),
                })
                .collect(),
            due_senders: Vec::new(),
            due_clocks: Vec::new(),
            pending: None,
            rng_counters: vec![0; rng_streams],
            evaluator_hashes: Vec::new(),
            artifact_manifests: Vec::new(),
            completed_transactions: Vec::new(),
            completed_count: 0,
            completed_chain_hash: format!("sha256:{}", "0".repeat(64)),
        })
    }

    /// Validate checkpoint-local invariants without needing workflow config.
    /// Topology equality with MAL and schedule periods is validated separately
    /// when the workflow is resolved.
    pub fn validate_resume(&self) -> Result<()> {
        ensure!(
            (2..=MAX_MEMORY_TIERS).contains(&self.tiers.len()),
            "sleep checkpoint must contain 2..={MAX_MEMORY_TIERS} tiers"
        );
        ensure!(
            (1..=MAX_SLEEP_RNG_STREAMS).contains(&self.rng_counters.len()),
            "sleep checkpoint must contain 1..={MAX_SLEEP_RNG_STREAMS} RNG streams"
        );
        let mut tier_ids = BTreeSet::new();
        let mut total_reserve_slots = 0usize;
        for tier in &self.tiers {
            ensure!(
                !tier.id.trim().is_empty() && tier.id.len() <= MAX_MEMORY_TIER_ID_BYTES,
                "sleep checkpoint tier id must contain 1..={MAX_MEMORY_TIER_ID_BYTES} bytes"
            );
            ensure!(
                tier_ids.insert(tier.id.as_str()),
                "sleep checkpoint repeats tier `{}`",
                tier.id
            );
            ensure!(
                (1..=MAX_RESERVE_SLOTS_PER_TIER).contains(&tier.slots.len()),
                "sleep tier `{}` must contain 1..={MAX_RESERVE_SLOTS_PER_TIER} reserve slots",
                tier.id
            );
            total_reserve_slots = total_reserve_slots
                .checked_add(tier.slots.len())
                .context("sleep checkpoint reserve-slot count overflow")?;
            ensure!(
                total_reserve_slots <= MAX_TOTAL_RESERVE_SLOTS,
                "sleep checkpoint exceeds the {MAX_TOTAL_RESERVE_SLOTS}-slot total reserve limit"
            );
            ensure!(
                tier.last_update_clock <= tier.last_boundary_clock
                    && tier.last_boundary_clock <= self.clock,
                "sleep tier `{}` has inconsistent update/boundary clocks",
                tier.id
            );
        }
        let due = self
            .due_clocks
            .iter()
            .copied()
            .zip(self.due_senders.iter().copied())
            .collect::<Vec<_>>();
        // A rejected terminal consolidation is retried before ordinary
        // senders so no further transfer can fill its already-full reserve.
        // That exceptional first entry may therefore precede an older due
        // clock; the remainder always retains normal chronological ordering.
        let terminal_retry_prefix = due.first().is_some_and(|(_, sender)| {
            self.tiers.get(*sender).is_some_and(|tier| {
                *sender + 1 == self.tiers.len() && tier.last_update_clock < tier.last_boundary_clock
            })
        });
        let ordered_start = usize::from(terminal_retry_prefix);
        let unique_due_senders = self.due_senders.iter().copied().collect::<BTreeSet<_>>();
        ensure!(
            self.due_senders.len() == self.due_clocks.len()
                && unique_due_senders.len() == self.due_senders.len()
                && self
                    .due_senders
                    .iter()
                    .all(|sender| *sender < self.tiers.len())
                && self
                    .due_clocks
                    .iter()
                    .all(|clock| *clock > 0 && *clock <= self.clock)
                && due[ordered_start..]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "sleep checkpoint has an invalid due-boundary queue"
        );
        ensure!(
            self.evaluator_hashes.len() <= MAX_SLEEP_EVALUATOR_HASHES,
            "sleep checkpoint exceeds the {MAX_SLEEP_EVALUATOR_HASHES}-evaluator limit"
        );
        for hash in &self.evaluator_hashes {
            validate_sha256_identity(hash, "sleep evaluator hash")?;
        }
        validate_sha256_identity(&self.completed_chain_hash, "sleep completed-history hash")?;
        let expected_completed_tail =
            usize::try_from(self.completed_count.min(COMPLETED_TRANSACTION_TAIL as u64))
                .context("sleep completed-transaction tail length exceeds usize")?;
        ensure!(
            self.completed_count <= self.cycle
                && self.completed_transactions.len() == expected_completed_tail,
            "sleep completed-transaction audit tail is inconsistent"
        );
        ensure!(
            self.completed_transactions
                .windows(2)
                .all(|pair| pair[0].id < pair[1].id)
                && self
                    .completed_transactions
                    .iter()
                    .all(|txn| txn.id <= self.cycle
                        && txn.committed
                        && txn.candidate_hash.is_some()),
            "sleep completed-transaction audit tail is invalid"
        );
        for txn in &self.completed_transactions {
            validate_completed_transaction(txn, &self.tiers, &self.rng_counters)?;
        }
        ensure!(
            self.artifact_manifests
                == referenced_artifact_manifests(
                    &self.completed_transactions,
                    self.pending.as_ref(),
                )?,
            "sleep artifact-manifest index differs from its bounded transaction history"
        );
        match (self.phase, self.pending.as_ref()) {
            (SleepPhase::Wake, None) => {}
            (SleepPhase::Wake, Some(_)) => bail!("wake checkpoint has a pending sleep transaction"),
            (_, None) => bail!("sleep subphase {:?} has no pending transaction", self.phase),
            (_, Some(txn)) => {
                ensure!(
                    txn.sender < self.tiers.len(),
                    "sleep transaction sender tier is out of range"
                );
                ensure!(
                    txn.terminal == (txn.sender + 1 == self.tiers.len()),
                    "sleep transaction terminal marker disagrees with its sender tier"
                );
                ensure!(
                    (txn.terminal && txn.receiver == txn.sender)
                        || (!txn.terminal && txn.sender.checked_add(1) == Some(txn.receiver)),
                    "sleep transaction has an invalid receiver topology"
                );
                ensure!(
                    if txn.committed {
                        !self.due_senders.contains(&txn.sender)
                    } else {
                        self.due_senders.first() == Some(&txn.sender)
                    },
                    "pending sleep transaction disagrees with the due-sender queue"
                );
                ensure!(
                    txn.receiver < self.tiers.len(),
                    "sleep transaction tier is out of range"
                );
                ensure!(
                    self.due_clocks.first().copied() == Some(txn.trigger_clock) || txn.committed,
                    "sleep transaction trigger clock differs from due-boundary queue"
                );
                ensure!(
                    txn.trigger_clock > 0 && txn.trigger_clock <= self.clock,
                    "sleep transaction trigger clock is zero or ahead of the sleep clock"
                );
                if txn.committed {
                    ensure!(
                        self.tiers[txn.sender].last_update_clock == txn.trigger_clock
                            && self.tiers[txn.sender].last_boundary_clock == txn.trigger_clock,
                        "committed sleep transaction disagrees with its sender clocks"
                    );
                } else {
                    ensure!(
                        self.tiers[txn.sender].last_boundary_clock < txn.trigger_clock,
                        "uncommitted sleep transaction does not own a future sender boundary"
                    );
                }
                let expected_id = if txn.committed {
                    self.cycle
                } else {
                    self.cycle
                        .checked_add(1)
                        .context("sleep transaction id overflow")?
                };
                ensure!(
                    txn.id == expected_id,
                    "sleep transaction id disagrees with cycle/commit state"
                );
                ensure!(
                    self.completed_transactions
                        .last()
                        .is_none_or(|completed| completed.id < txn.id),
                    "pending sleep transaction reuses a completed transaction id"
                );
                ensure!(
                    txn.terminal || txn.receiver_slot < self.tiers[txn.receiver].slots.len(),
                    "sleep receiver slot is out of range"
                );
                ensure!(
                    txn.sender_slots_to_reset
                        .iter()
                        .all(|slot| *slot < self.tiers[txn.sender].slots.len()),
                    "sleep sender slot is out of range"
                );
                ensure!(
                    txn.sender_slots_to_reset
                        .windows(2)
                        .all(|pair| pair[0] < pair[1]),
                    "sleep sender reset slots are duplicated or unordered"
                );
                if !txn.committed {
                    let active_sender_slots = self.tiers[txn.sender]
                        .slots
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, state)| state.active.then_some(slot))
                        .collect::<Vec<_>>();
                    ensure!(
                        txn.sender_slots_to_reset == active_sender_slots,
                        "sleep sender reset plan does not contain the exact active-slot set"
                    );
                } else {
                    ensure!(
                        self.tiers[txn.sender].slots.iter().all(|slot| !slot.active),
                        "committed sleep sender retains an active reserve slot"
                    );
                }
                ensure!(
                    txn.sender_slots_to_reset.iter().all(|slot| {
                        self.tiers[txn.sender].slots[*slot].active != txn.committed
                    }),
                    "sleep sender reset masks disagree with commit state"
                );
                if !txn.terminal {
                    ensure!(
                        self.tiers[txn.receiver].slots[txn.receiver_slot].active == txn.committed,
                        "sleep receiver mask disagrees with commit state"
                    );
                }
                ensure!(
                    !txn.teacher_checkpoint.trim().is_empty()
                        && txn.teacher_checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES
                        && !txn.student_checkpoint.trim().is_empty()
                        && txn.student_checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES,
                    "sleep transaction has incomplete checkpoint locations"
                );
                validate_sha256_identity(&txn.teacher_hash, "sleep teacher hash")?;
                validate_sha256_identity(&txn.student_hash, "sleep student hash")?;
                validate_sha256_identity(
                    &txn.prospective_update_hash,
                    "sleep prospective-update hash",
                )?;
                let mut reservations = [
                    txn.knowledge_rng,
                    txn.imitation_rng,
                    txn.dream_generation_rng,
                    txn.dream_selection_rng,
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
                for reservation in &reservations {
                    reservation.validate(&self.rng_counters)?;
                }
                let mut trial_rng_ids = BTreeSet::new();
                for trial_rng in &txn.dream_trial_rngs {
                    ensure!(
                        !trial_rng.candidate_id.trim().is_empty()
                            && trial_rng.candidate_id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES
                            && trial_rng_ids.insert(trial_rng.candidate_id.as_str()),
                        "sleep dream-trial RNG reservations have duplicate or empty IDs"
                    );
                    trial_rng.reservation.validate(&self.rng_counters)?;
                    reservations.push(trial_rng.reservation);
                }
                reservations
                    .sort_unstable_by_key(|reservation| (reservation.stream, reservation.start));
                ensure!(
                    reservations.windows(2).all(|pair| {
                        pair[0].stream != pair[1].stream
                            || pair[0].start + pair[0].count <= pair[1].start
                    }),
                    "sleep RNG reservations overlap"
                );
                if matches!(
                    self.phase,
                    SleepPhase::Imitation
                        | SleepPhase::RetentionValidation
                        | SleepPhase::Commit
                        | SleepPhase::DreamGeneration
                        | SleepPhase::DreamRanking
                        | SleepPhase::DreamTrials
                        | SleepPhase::DreamPolicyUpdate
                        | SleepPhase::Candidate
                ) {
                    ensure!(
                        txn.knowledge_rng.is_some(),
                        "sleep checkpoint passed Knowledge Seeding without an RNG reservation"
                    );
                }
                if matches!(
                    self.phase,
                    SleepPhase::RetentionValidation
                        | SleepPhase::Commit
                        | SleepPhase::DreamGeneration
                        | SleepPhase::DreamRanking
                        | SleepPhase::DreamTrials
                        | SleepPhase::DreamPolicyUpdate
                        | SleepPhase::Candidate
                ) {
                    ensure!(
                        txn.imitation_rng.is_some(),
                        "sleep checkpoint passed imitation without an RNG reservation"
                    );
                }
                ensure!(
                    txn.candidate_checkpoint.is_some() == txn.candidate_hash.is_some(),
                    "sleep transaction candidate identity is incomplete"
                );
                if let (Some(checkpoint), Some(hash)) =
                    (&txn.candidate_checkpoint, &txn.candidate_hash)
                {
                    ensure!(
                        !checkpoint.trim().is_empty()
                            && checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES,
                        "sleep transaction candidate checkpoint is empty"
                    );
                    validate_sha256_identity(hash, "sleep candidate hash")?;
                }
                ensure!(
                    !txn.committed || txn.candidate_checkpoint.is_some(),
                    "committed sleep transaction has no immutable candidate identity"
                );
                ensure!(
                    txn.tensor_transaction_generation.is_some()
                        == txn.tensor_transaction_manifest_hash.is_some(),
                    "sleep tensor transaction generation/hash pair is incomplete"
                );
                if let (Some(generation), Some(hash)) = (
                    &txn.tensor_transaction_generation,
                    &txn.tensor_transaction_manifest_hash,
                ) {
                    let digest = generation
                        .strip_prefix("sha256-")
                        .context("sleep tensor generation has an unsafe name")?;
                    validate_sha256_identity(
                        &format!("sha256:{digest}"),
                        "sleep tensor generation",
                    )?;
                    validate_sha256_identity(hash, "sleep tensor manifest hash")?;
                    ensure!(
                        hash.strip_prefix("sha256:") == Some(digest),
                        "sleep tensor generation does not match its manifest hash"
                    );
                }
                if let Some(manifest) = &txn.generated_manifest {
                    validate_sha256_identity(manifest, "dream artifact manifest")?;
                }
                if let Some(hash) = &txn.dream_shared_checkpoint_hash {
                    validate_sha256_identity(hash, "dream shared-checkpoint hash")?;
                }
                for trial in &txn.dream_trials {
                    ensure!(
                        !trial.candidate_id.trim().is_empty()
                            && trial.candidate_id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES
                            && trial.independent_task_improvement.is_finite(),
                        "dream trial has an invalid candidate identity or score"
                    );
                    validate_sha256_identity(&trial.adapter_hash, "dream adapter hash")?;
                    validate_sha256_identity(&trial.evaluator_hash, "dream evaluator hash")?;
                }
                if let Some(receipt) = &txn.dream_policy_receipt {
                    validate_sha256_identity(receipt, "dream policy receipt")?;
                }
                let selected = txn
                    .dream_selected
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>();
                ensure!(
                    selected.len() == txn.dream_selected.len()
                        && selected.iter().all(|id| {
                            !id.trim().is_empty() && id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES
                        }),
                    "dream selection contains an empty or duplicate candidate id"
                );
                let trial_ids = txn
                    .dream_trials
                    .iter()
                    .map(|trial| trial.candidate_id.as_str())
                    .collect::<BTreeSet<_>>();
                ensure!(
                    trial_ids.len() == txn.dream_trials.len()
                        && trial_ids.iter().all(|id| selected.contains(id)),
                    "dream trials are duplicated or absent from the selected candidates"
                );
                ensure!(
                    trial_rng_ids.iter().all(|id| selected.contains(id)),
                    "dream-trial RNG reservation is absent from the selected candidates"
                );
                match txn.dream_generation_rng {
                    Some(reservation) => {
                        let candidate_count = usize::try_from(reservation.count)
                            .context("dream candidate count exceeds usize")?;
                        ensure!(
                            txn.dream_selected.len() <= candidate_count
                                && txn.dream_trials.len() <= candidate_count
                                && txn.dream_trial_rngs.len() <= candidate_count,
                            "dream checkpoint evidence exceeds its generated-candidate count"
                        );
                    }
                    None => ensure!(
                        txn.generated_manifest.is_none()
                            && txn.dream_selection_rng.is_none()
                            && txn.dream_trial_rngs.is_empty()
                            && txn.dream_selected.is_empty()
                            && txn.dream_trials.is_empty()
                            && txn.dream_policy_receipt.is_none(),
                        "dream checkpoint contains evidence without a generation reservation"
                    ),
                }
                let requires_commit = matches!(
                    self.phase,
                    SleepPhase::DreamGeneration
                        | SleepPhase::DreamRanking
                        | SleepPhase::DreamTrials
                        | SleepPhase::DreamPolicyUpdate
                        | SleepPhase::Candidate
                );
                ensure!(
                    !requires_commit || txn.committed,
                    "sleep transaction commit marker disagrees with subphase {:?}",
                    self.phase
                );
                let is_dreaming = matches!(
                    self.phase,
                    SleepPhase::DreamGeneration
                        | SleepPhase::DreamRanking
                        | SleepPhase::DreamTrials
                        | SleepPhase::DreamPolicyUpdate
                );
                ensure!(
                    !is_dreaming || txn.dream_shared_checkpoint_hash.is_some(),
                    "dream subphase {:?} has no shared checkpoint identity",
                    self.phase
                );
                if matches!(
                    self.phase,
                    SleepPhase::DreamRanking
                        | SleepPhase::DreamTrials
                        | SleepPhase::DreamPolicyUpdate
                ) {
                    ensure!(
                        txn.generated_manifest.is_some(),
                        "dream subphase {:?} has no generated-artifact manifest",
                        self.phase
                    );
                    ensure!(
                        txn.generated_manifest
                            .as_ref()
                            .is_some_and(|manifest| self.artifact_manifests.contains(manifest)),
                        "dream manifest is absent from the checkpoint artifact list"
                    );
                    ensure!(
                        txn.dream_generation_rng.is_some(),
                        "dream generation has no persisted RNG reservation"
                    );
                }
                if matches!(
                    self.phase,
                    SleepPhase::DreamTrials | SleepPhase::DreamPolicyUpdate
                ) {
                    ensure!(
                        !txn.dream_selected.is_empty(),
                        "dream subphase {:?} has no selected candidates",
                        self.phase
                    );
                    ensure!(
                        txn.dream_selection_rng.is_some(),
                        "dream selection has no persisted RNG reservation"
                    );
                }
                if matches!(self.phase, SleepPhase::DreamPolicyUpdate)
                    || (self.phase == SleepPhase::Candidate && txn.generated_manifest.is_some())
                {
                    ensure!(
                        trial_ids == selected,
                        "completed dreaming checkpoint does not contain one trial per selected candidate"
                    );
                    ensure!(
                        trial_rng_ids == selected,
                        "completed dream trials lack exact RNG reservations"
                    );
                }
                ensure!(
                    self.phase != SleepPhase::Candidate
                        || txn.generated_manifest.is_none()
                        || txn.dream_policy_receipt.is_some(),
                    "completed dreaming checkpoint has no ReSTEM policy receipt"
                );
                ensure!(
                    txn.generated_manifest.is_none() || txn.dream_shared_checkpoint_hash.is_some(),
                    "dream artifacts have no immutable shared-checkpoint identity"
                );
                let no_dream_evidence = txn.dream_generation_rng.is_none()
                    && txn.dream_selection_rng.is_none()
                    && txn.dream_trial_rngs.is_empty()
                    && txn.generated_manifest.is_none()
                    && txn.dream_shared_checkpoint_hash.is_none()
                    && txn.dream_selected.is_empty()
                    && txn.dream_trials.is_empty()
                    && txn.dream_policy_receipt.is_none();
                match self.phase {
                    SleepPhase::ProspectiveUpdate => ensure!(
                        txn.knowledge_rng.is_none()
                            && txn.imitation_rng.is_none()
                            && no_dream_evidence,
                        "prospective-update checkpoint contains later-subphase evidence"
                    ),
                    SleepPhase::KnowledgeSeeding => ensure!(
                        txn.imitation_rng.is_none() && no_dream_evidence,
                        "Knowledge Seeding checkpoint contains later-subphase evidence"
                    ),
                    SleepPhase::Imitation
                    | SleepPhase::RetentionValidation
                    | SleepPhase::Commit => ensure!(
                        no_dream_evidence,
                        "pre-Dreaming checkpoint contains Dreaming evidence"
                    ),
                    SleepPhase::DreamGeneration => ensure!(
                        txn.generated_manifest.is_none()
                            && txn.dream_selection_rng.is_none()
                            && txn.dream_trial_rngs.is_empty()
                            && txn.dream_selected.is_empty()
                            && txn.dream_trials.is_empty()
                            && txn.dream_policy_receipt.is_none(),
                        "Dream Generation checkpoint contains later-subphase evidence"
                    ),
                    SleepPhase::DreamRanking => ensure!(
                        txn.dream_trial_rngs.is_empty()
                            && txn.dream_selected.is_empty()
                            && txn.dream_trials.is_empty()
                            && txn.dream_policy_receipt.is_none(),
                        "Dream Ranking checkpoint contains later-subphase evidence"
                    ),
                    SleepPhase::DreamTrials => ensure!(
                        txn.dream_policy_receipt.is_none(),
                        "Dream Trials checkpoint contains a premature ReSTEM receipt"
                    ),
                    SleepPhase::DreamPolicyUpdate => ensure!(
                        txn.dream_policy_receipt.is_none()
                            || (trial_ids == selected && trial_rng_ids == selected),
                        "ReSTEM receipt precedes complete isolated-trial evidence"
                    ),
                    SleepPhase::Candidate => ensure!(
                        no_dream_evidence
                            || (txn.generated_manifest.is_some()
                                && txn.dream_generation_rng.is_some()
                                && txn.dream_selection_rng.is_some()
                                && trial_ids == selected
                                && trial_rng_ids == selected
                                && txn.dream_policy_receipt.is_some()),
                        "sleep candidate contains incomplete Dreaming evidence"
                    ),
                    SleepPhase::Wake => unreachable!("wake checkpoints cannot have a transaction"),
                }
            }
        }
        Ok(())
    }

    pub fn advance_clock(&mut self, schedule: &SleepSchedule, clock: u64) -> Result<()> {
        schedule.validate()?;
        ensure!(
            self.pending.is_none(),
            "cannot advance wake clock during consolidation"
        );
        ensure!(
            self.due_senders.is_empty(),
            "cannot advance sleep clock before finishing every due sender"
        );
        ensure!(
            schedule.tiers.len() == self.tiers.len()
                && schedule
                    .tiers
                    .iter()
                    .zip(&self.tiers)
                    .all(|(configured, saved)| configured.id == saved.id
                        && configured.reserve_slots == saved.slots.len()),
            "sleep schedule topology differs from checkpoint state"
        );
        ensure!(clock >= self.clock, "sleep clock cannot move backwards");
        let mut due = Vec::new();
        for (sender, tier) in schedule.tiers.iter().enumerate() {
            let last = self.tiers[sender].last_boundary_clock;
            let first = last
                .checked_div(tier.update_period)
                .and_then(|multiple| multiple.checked_add(1))
                .and_then(|multiple| multiple.checked_mul(tier.update_period))
                .context("sleep boundary clock overflow")?;
            if first <= clock {
                let second = first
                    .checked_add(tier.update_period)
                    .context("sleep boundary clock overflow")?;
                ensure!(
                    second > clock,
                    "sleep clock advance from {} to {clock} crosses multiple `{}` boundaries ({first} and {second}); one tier gradient accumulator cannot supply multiple updates, so the host must split the advance at a boundary",
                    self.clock,
                    tier.id,
                );
                due.push((first, sender));
            }
        }
        due.sort_unstable();

        // A failed terminal transfer leaves its active slots intact by
        // design. Backpressure ordinary transfers until the next configured
        // terminal boundary, then retry terminal first so capacity is either
        // reclaimed before a new transfer or remains safely unchanged.
        let terminal = self.tiers.len() - 1;
        let terminal_retry_pending =
            self.tiers[terminal].last_update_clock < self.tiers[terminal].last_boundary_clock;
        if terminal_retry_pending {
            if let Some(index) = due.iter().position(|(_, sender)| *sender == terminal) {
                let retry = due.remove(index);
                due.insert(0, retry);
            } else {
                // No terminal attempt is scheduled yet. Consume only the
                // logical boundaries, not their accumulated gradients, so
                // wake learning can continue without filling more reserves.
                for (trigger_clock, sender) in &due {
                    self.tiers[*sender].last_boundary_clock = *trigger_clock;
                }
                due.clear();
            }
        }

        self.clock = clock;
        self.due_clocks = due.iter().map(|(clock, _)| *clock).collect();
        self.due_senders = due.into_iter().map(|(_, sender)| sender).collect();
        Ok(())
    }

    pub fn next_due_boundary(&self) -> Option<(usize, u64)> {
        self.due_senders
            .first()
            .copied()
            .zip(self.due_clocks.first().copied())
    }

    pub fn next_due_sender(&self) -> Option<usize> {
        self.next_due_boundary().map(|(sender, _)| sender)
    }

    fn ensure_next_due_boundary(
        &self,
        expected_sender: usize,
        expected_clock: u64,
        operation: &str,
    ) -> Result<()> {
        ensure!(
            self.next_due_boundary() == Some((expected_sender, expected_clock)),
            "{operation} boundary ({expected_clock}, {expected_sender}) is not first in the due queue"
        );
        Ok(())
    }

    fn consume_next_due_boundary(
        &mut self,
        expected_sender: usize,
        expected_clock: u64,
        operation: &str,
    ) -> Result<()> {
        self.ensure_next_due_boundary(expected_sender, expected_clock, operation)?;
        self.due_senders.remove(0);
        self.due_clocks.remove(0);
        Ok(())
    }

    /// Allocate the identifier that the next consolidation attempt must use.
    /// Successful and rolled-back attempts both advance `cycle`, so callers
    /// preparing immutable transaction-keyed artifacts must use this checked
    /// helper rather than arithmetic on the checkpoint field.
    pub fn next_transaction_id(&self) -> Result<u64> {
        self.cycle
            .checked_add(1)
            .context("sleep consolidation transaction ID overflows u64")
    }

    pub fn begin(
        &mut self,
        sender: usize,
        teacher_checkpoint: String,
        teacher_hash: String,
        student_checkpoint: String,
        student_hash: String,
        prospective_update_hash: String,
    ) -> Result<ConsolidationTxn> {
        ensure!(
            self.pending.is_none() && self.phase == SleepPhase::Wake,
            "a consolidation transaction is already active"
        );
        ensure!(
            sender < self.tiers.len(),
            "sender tier {sender} is out of range"
        );
        let (next_sender, trigger_clock) = self.next_due_boundary().with_context(|| {
            format!(
                "sender tier {sender} has no paired due boundary at sleep clock {}",
                self.clock
            )
        })?;
        ensure!(
            next_sender == sender,
            "sender tier {sender} is not next at sleep clock {}",
            self.clock
        );
        ensure!(
            !teacher_checkpoint.trim().is_empty()
                && teacher_checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES
                && !student_checkpoint.trim().is_empty()
                && student_checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES,
            "consolidation checkpoint locations must not be empty"
        );
        validate_sha256_identity(&teacher_hash, "teacher hash")?;
        validate_sha256_identity(&student_hash, "student hash")?;
        validate_sha256_identity(&prospective_update_hash, "prospective-update hash")?;
        let terminal = sender + 1 == self.tiers.len();
        let receiver = if terminal { sender } else { sender + 1 };
        let receiver_slot = if terminal {
            0
        } else {
            self.tiers[receiver]
                .slots
                .iter()
                .position(|slot| !slot.active)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "memory tier `{}` has exhausted its reserve slots",
                        self.tiers[receiver].id
                    )
                })?
        };
        let sender_slots_to_reset = self.tiers[sender]
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.active.then_some(index))
            .collect();
        let transaction_id = self.next_transaction_id()?;
        let txn = ConsolidationTxn {
            id: transaction_id,
            trigger_clock,
            sender,
            receiver,
            receiver_slot,
            terminal,
            sender_slots_to_reset,
            teacher_checkpoint,
            teacher_hash,
            student_checkpoint,
            student_hash,
            prospective_update_hash,
            candidate_checkpoint: None,
            candidate_hash: None,
            knowledge_rng: None,
            imitation_rng: None,
            dream_generation_rng: None,
            dream_selection_rng: None,
            dream_trial_rngs: Vec::new(),
            tensor_transaction_generation: None,
            tensor_transaction_manifest_hash: None,
            generated_manifest: None,
            dream_shared_checkpoint_hash: None,
            dream_selected: Vec::new(),
            dream_trials: Vec::new(),
            dream_policy_receipt: None,
            committed: false,
        };
        self.phase = SleepPhase::ProspectiveUpdate;
        self.pending = Some(txn.clone());
        Ok(txn)
    }

    pub fn transition(&mut self, next: SleepPhase) -> Result<()> {
        let allowed = matches!(
            (self.phase, next),
            (SleepPhase::ProspectiveUpdate, SleepPhase::KnowledgeSeeding)
                | (SleepPhase::KnowledgeSeeding, SleepPhase::Imitation)
                | (SleepPhase::Imitation, SleepPhase::RetentionValidation)
                | (SleepPhase::RetentionValidation, SleepPhase::Commit)
                | (SleepPhase::Commit, SleepPhase::DreamGeneration)
                | (SleepPhase::Commit, SleepPhase::Candidate)
                | (SleepPhase::DreamGeneration, SleepPhase::DreamRanking)
                | (SleepPhase::DreamRanking, SleepPhase::DreamTrials)
                | (SleepPhase::DreamTrials, SleepPhase::DreamPolicyUpdate)
                | (SleepPhase::DreamPolicyUpdate, SleepPhase::Candidate)
        );
        ensure!(
            allowed,
            "invalid sleep transition {:?} -> {next:?}",
            self.phase
        );
        ensure!(
            self.pending.is_some(),
            "sleep transition has no transaction"
        );
        if matches!(
            next,
            SleepPhase::DreamGeneration
                | SleepPhase::DreamRanking
                | SleepPhase::DreamTrials
                | SleepPhase::DreamPolicyUpdate
                | SleepPhase::Candidate
        ) {
            ensure!(
                self.pending.as_ref().is_some_and(|txn| txn.committed),
                "sleep cannot enter {next:?} before atomic consolidation commit"
            );
        }
        self.phase = next;
        Ok(())
    }

    fn reserve_rng(&mut self, stream: usize, count: u64) -> Result<RngReservation> {
        ensure!(count > 0, "sleep RNG reservation count must be positive");
        let counter = self
            .rng_counters
            .get_mut(stream)
            .with_context(|| format!("sleep RNG stream {stream} does not exist"))?;
        let reservation = RngReservation {
            stream,
            start: *counter,
            count,
        };
        *counter = counter
            .checked_add(count)
            .context("sleep RNG counter overflow")?;
        Ok(reservation)
    }

    pub fn reserve_knowledge_rng(&mut self, stream: usize, count: u64) -> Result<RngReservation> {
        ensure!(
            self.pending.is_some() && self.phase == SleepPhase::KnowledgeSeeding,
            "knowledge RNG reservation requires the Knowledge Seeding phase"
        );
        if let Some(reservation) = self.pending.as_ref().and_then(|txn| txn.knowledge_rng) {
            ensure!(
                reservation.stream == stream && reservation.count == count,
                "knowledge RNG reservation differs on retry"
            );
            return Ok(reservation);
        }
        let reservation = self.reserve_rng(stream, count)?;
        self.pending
            .as_mut()
            .context("knowledge RNG reservation has no transaction")?
            .knowledge_rng = Some(reservation);
        Ok(reservation)
    }

    pub fn reserve_imitation_rng(&mut self, stream: usize, count: u64) -> Result<RngReservation> {
        ensure!(
            self.pending.is_some() && self.phase == SleepPhase::Imitation,
            "imitation RNG reservation requires the imitation phase"
        );
        if let Some(reservation) = self.pending.as_ref().and_then(|txn| txn.imitation_rng) {
            ensure!(
                reservation.stream == stream && reservation.count == count,
                "imitation RNG reservation differs on retry"
            );
            return Ok(reservation);
        }
        let reservation = self.reserve_rng(stream, count)?;
        self.pending
            .as_mut()
            .context("imitation RNG reservation has no transaction")?
            .imitation_rng = Some(reservation);
        Ok(reservation)
    }

    pub fn reserve_dream_generation_rng(
        &mut self,
        stream: usize,
        count: u64,
    ) -> Result<RngReservation> {
        ensure!(
            self.pending.is_some() && self.phase == SleepPhase::DreamGeneration,
            "dream-generation RNG reservation requires the Dream Generation phase"
        );
        if let Some(reservation) = self
            .pending
            .as_ref()
            .and_then(|txn| txn.dream_generation_rng)
        {
            ensure!(
                reservation.stream == stream && reservation.count == count,
                "dream-generation RNG reservation differs on retry"
            );
            return Ok(reservation);
        }
        let reservation = self.reserve_rng(stream, count)?;
        self.pending
            .as_mut()
            .context("dream-generation RNG reservation has no transaction")?
            .dream_generation_rng = Some(reservation);
        Ok(reservation)
    }

    pub fn reserve_dream_selection_rng(&mut self, stream: usize) -> Result<RngReservation> {
        ensure!(
            self.pending.is_some() && self.phase == SleepPhase::DreamRanking,
            "dream-selection RNG reservation requires the Dream Ranking phase"
        );
        if let Some(reservation) = self
            .pending
            .as_ref()
            .and_then(|txn| txn.dream_selection_rng)
        {
            ensure!(
                reservation.stream == stream && reservation.count == 1,
                "dream-selection RNG reservation differs on retry"
            );
            return Ok(reservation);
        }
        let reservation = self.reserve_rng(stream, 1)?;
        self.pending
            .as_mut()
            .context("dream-selection RNG reservation has no transaction")?
            .dream_selection_rng = Some(reservation);
        Ok(reservation)
    }

    pub fn reserve_dream_trial_rng(
        &mut self,
        stream: usize,
        candidate_id: &str,
    ) -> Result<RngReservation> {
        ensure!(
            !candidate_id.trim().is_empty() && candidate_id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES,
            "dream candidate ID is empty or too long"
        );
        ensure!(
            self.pending.is_some() && self.phase == SleepPhase::DreamTrials,
            "dream-trial RNG reservation requires the Dream Trials phase"
        );
        ensure!(
            self.pending
                .as_ref()
                .is_some_and(|txn| txn.dream_selected.iter().any(|id| id == candidate_id)),
            "dream-trial RNG reservation is not for a selected candidate"
        );
        if let Some(reservation) = self.pending.as_ref().and_then(|txn| {
            txn.dream_trial_rngs
                .iter()
                .find(|trial| trial.candidate_id == candidate_id)
                .map(|trial| trial.reservation)
        }) {
            ensure!(
                reservation.stream == stream && reservation.count == 1,
                "dream-trial RNG reservation differs on retry"
            );
            return Ok(reservation);
        }
        let reservation = self.reserve_rng(stream, 1)?;
        self.pending
            .as_mut()
            .context("dream-trial RNG reservation has no transaction")?
            .dream_trial_rngs
            .push(DreamTrialRng {
                candidate_id: candidate_id.to_owned(),
                reservation,
            });
        Ok(reservation)
    }

    /// Commit only after retention validation.  Backends must commit tensor
    /// weights and optimizer state atomically with persisting this state.
    pub fn commit_consolidation(&mut self) -> Result<()> {
        ensure!(
            self.phase == SleepPhase::Commit,
            "consolidation can commit only in commit phase"
        );
        let txn = self
            .pending
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no consolidation transaction"))?
            .clone();
        if txn.committed {
            return Ok(());
        }
        ensure!(
            txn.sender < self.tiers.len() && txn.receiver < self.tiers.len(),
            "consolidation transaction tier is out of range"
        );
        ensure!(
            txn.terminal == (txn.sender.checked_add(1) == Some(self.tiers.len()))
                && ((txn.terminal && txn.receiver == txn.sender)
                    || (!txn.terminal && txn.sender.checked_add(1) == Some(txn.receiver))),
            "consolidation transaction has an invalid sender/receiver topology"
        );
        ensure!(
            txn.candidate_checkpoint
                .as_ref()
                .is_some_and(|uri| !uri.trim().is_empty()),
            "consolidation has no published candidate checkpoint"
        );
        validate_sha256_identity(
            txn.candidate_hash
                .as_deref()
                .context("consolidation has no published candidate hash")?,
            "published candidate hash",
        )?;
        // Validate the whole metadata transaction before changing any slot.
        // Corrupt resume state and generation exhaustion must fail without a
        // partially activated receiver or partially reclaimed sender.
        self.ensure_next_due_boundary(txn.sender, txn.trigger_clock, "committed")?;
        let sender_tier = self
            .tiers
            .get(txn.sender)
            .context("committed sender tier is out of range")?;
        let active_sender_slots = sender_tier
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, state)| state.active.then_some(slot))
            .collect::<Vec<_>>();
        ensure!(
            txn.sender_slots_to_reset == active_sender_slots,
            "consolidation sender reset plan does not contain the exact active-slot set"
        );
        if !txn.terminal {
            let receiver_slot = self
                .tiers
                .get(txn.receiver)
                .context("committed receiver tier is out of range")?
                .slots
                .get(txn.receiver_slot)
                .context("committed receiver slot is out of range")?;
            ensure!(!receiver_slot.active, "receiver slot is already active");
            ensure!(
                receiver_slot.generation < u64::MAX,
                "receiver reserve generation overflows u64"
            );
        }
        for &slot in &txn.sender_slots_to_reset {
            let sender_slot = sender_tier
                .slots
                .get(slot)
                .context("sender reset plan contains an out-of-range slot")?;
            ensure!(
                sender_slot.active,
                "sender reset plan contains dormant slot"
            );
            ensure!(
                sender_slot.generation < u64::MAX,
                "sender reserve generation overflows u64"
            );
        }
        self.consume_next_due_boundary(txn.sender, txn.trigger_clock, "committed")?;

        if !txn.terminal {
            let receiver_slot = &mut self.tiers[txn.receiver].slots[txn.receiver_slot];
            receiver_slot.active = true;
            receiver_slot.generation += 1;
        }
        for &slot in &txn.sender_slots_to_reset {
            let sender_slot = &mut self.tiers[txn.sender].slots[slot];
            sender_slot.active = false;
            sender_slot.generation += 1;
        }
        self.tiers[txn.sender].last_update_clock = txn.trigger_clock;
        self.tiers[txn.sender].last_boundary_clock = txn.trigger_clock;
        self.cycle = txn.id;
        self.pending
            .as_mut()
            .expect("transaction was validated above")
            .committed = true;
        Ok(())
    }

    /// Bind the exact immutable weights published by a successful tensor
    /// commit before changing masks/clocks. Repeating the same receipt is
    /// idempotent; substituting another candidate is rejected.
    pub fn record_committed_candidate(&mut self, checkpoint: String, hash: String) -> Result<()> {
        ensure!(
            self.phase == SleepPhase::Commit,
            "candidate identity can be recorded only in commit phase"
        );
        ensure!(
            !checkpoint.trim().is_empty() && checkpoint.len() <= MAX_CHECKPOINT_LOCATION_BYTES,
            "candidate checkpoint is empty"
        );
        validate_sha256_identity(&hash, "published candidate hash")?;
        let txn = self
            .pending
            .as_mut()
            .context("candidate identity has no consolidation transaction")?;
        match (&txn.candidate_checkpoint, &txn.candidate_hash) {
            (None, None) => {
                txn.candidate_checkpoint = Some(checkpoint);
                txn.candidate_hash = Some(hash);
            }
            (Some(existing_checkpoint), Some(existing_hash)) => ensure!(
                existing_checkpoint == &checkpoint && existing_hash == &hash,
                "committed candidate identity changed during retry"
            ),
            _ => bail!("consolidation has a partial candidate identity"),
        }
        Ok(())
    }

    pub fn record_generated_manifest(&mut self, manifest: String) -> Result<()> {
        validate_sha256_identity(&manifest, "generated manifest")?;
        ensure!(
            self.phase == SleepPhase::DreamGeneration,
            "generated manifest can be recorded only during dream generation"
        );
        let txn = self
            .pending
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no consolidation transaction"))?;
        if let Some(existing) = &txn.generated_manifest {
            ensure!(
                existing == &manifest,
                "generated manifest identity changed during retry"
            );
        }
        let mut next_pending = txn.clone();
        next_pending.generated_manifest = Some(manifest);
        let next_manifests =
            referenced_artifact_manifests(&self.completed_transactions, Some(&next_pending))?;
        self.pending = Some(next_pending);
        self.artifact_manifests = next_manifests;
        Ok(())
    }

    /// Bind the durable SleepState checkpoint to the exact independently
    /// authenticated tensor/optimizer generation for this subphase.
    pub fn record_tensor_transaction(
        &mut self,
        generation: String,
        manifest_hash: String,
    ) -> Result<()> {
        let digest = generation
            .strip_prefix("sha256-")
            .context("tensor transaction generation has an unsafe name")?;
        validate_sha256_identity(&format!("sha256:{digest}"), "tensor transaction generation")?;
        validate_sha256_identity(&manifest_hash, "tensor transaction manifest hash")?;
        ensure!(
            manifest_hash.strip_prefix("sha256:") == Some(digest),
            "tensor transaction generation does not match its manifest hash"
        );
        let txn = self
            .pending
            .as_mut()
            .context("no consolidation transaction for tensor snapshot")?;
        txn.tensor_transaction_generation = Some(generation);
        txn.tensor_transaction_manifest_hash = Some(manifest_hash);
        Ok(())
    }

    pub fn finish_candidate(&mut self) -> Result<ConsolidationTxn> {
        ensure!(
            self.phase == SleepPhase::Candidate,
            "sleep cycle is not a candidate"
        );
        ensure!(
            self.pending.as_ref().is_some_and(|txn| txn.committed),
            "sleep candidate is not atomically committed"
        );
        // Derive every fallible result before mutating the checkpoint. A
        // counter overflow or serialization error must leave the committed
        // Candidate retryable instead of dropping its pending transaction or
        // advancing only part of the audit chain.
        let txn = self
            .pending
            .as_ref()
            .expect("committed transaction checked above")
            .clone();
        let encoded = serde_json::to_vec(&txn)?;
        let completed_count = self
            .completed_count
            .checked_add(1)
            .context("sleep completed-transaction count overflow")?;
        self.validate_resume()
            .context("validating completed sleep candidate")?;
        let encoded_len = u64::try_from(encoded.len())
            .context("sleep completed-transaction encoding exceeds u64")?;
        let mut hash = sha2::Sha256::new();
        use sha2::Digest as _;
        hash.update(b"hermes-sleep-audit-chain-v1\0");
        hash.update(self.completed_chain_hash.as_bytes());
        hash.update(encoded_len.to_le_bytes());
        hash.update(encoded);
        let completed_chain_hash = format!("sha256:{:x}", hash.finalize());
        let mut completed_transactions = self.completed_transactions.clone();
        completed_transactions
            .try_reserve(1)
            .context("reserving sleep completed-transaction audit tail")?;
        completed_transactions.push(txn.clone());
        if completed_transactions.len() > COMPLETED_TRANSACTION_TAIL {
            completed_transactions.remove(0);
        }
        let artifact_manifests = referenced_artifact_manifests(&completed_transactions, None)?;

        self.pending = None;
        self.completed_chain_hash = completed_chain_hash;
        self.completed_count = completed_count;
        self.completed_transactions = completed_transactions;
        self.artifact_manifests = artifact_manifests;
        self.phase = SleepPhase::Wake;
        Ok(txn)
    }

    /// Roll back metadata.  The backend must restore the immutable teacher
    /// checkpoint before this method is persisted.
    pub fn rollback(&mut self) -> Result<ConsolidationTxn> {
        let (sender, trigger_clock) = self
            .pending
            .as_ref()
            .map(|txn| (txn.sender, txn.trigger_clock))
            .ok_or_else(|| anyhow::anyhow!("no consolidation transaction"))?;
        ensure!(
            sender < self.tiers.len(),
            "rolled-back sender tier is out of range"
        );
        let terminal = self
            .pending
            .as_ref()
            .expect("pending transaction was checked above")
            .terminal;
        if terminal {
            ensure!(
                self.due_senders.len() == self.due_clocks.len()
                    && self
                        .due_senders
                        .iter()
                        .all(|queued_sender| *queued_sender < self.tiers.len()),
                "rolled-back terminal boundary has an invalid trailing due queue"
            );
        }
        self.consume_next_due_boundary(sender, trigger_clock, "rolled-back")?;
        let txn = self
            .pending
            .take()
            .expect("transaction was validated before consuming its due boundary");
        self.tiers[sender].last_boundary_clock = trigger_clock;
        if terminal {
            // A terminal retry has priority because its reserve may already
            // be full. If it is rejected, do not attempt any transfers that
            // were queued behind it in the same coarse host advance. Their
            // gradient accumulators remain intact and will be consumed at a
            // later boundary after terminal retention succeeds.
            for (sender, trigger_clock) in self
                .due_senders
                .iter()
                .copied()
                .zip(self.due_clocks.iter().copied())
            {
                self.tiers[sender].last_boundary_clock = trigger_clock;
            }
            self.due_senders.clear();
            self.due_clocks.clear();
        }
        self.cycle = txn.id;
        self.phase = SleepPhase::Wake;
        Ok(txn)
    }
}

/// Reconcile trainer transaction metadata with the masks and generations
/// serialized inside every memory-bearing Transformer layer. Logical reserve
/// slots must move together across layers; this check runs after load and
/// before/after an atomic commit.
pub fn validate_model_memory_state(
    schedule: &SleepSchedule,
    state: &SleepState,
    statuses: &[MemorySlotStatus],
) -> Result<()> {
    schedule.validate()?;
    state.validate_resume()?;
    ensure!(
        schedule.tiers.len() == state.tiers.len(),
        "sleep workflow and checkpoint tier counts differ"
    );
    ensure!(
        !statuses.is_empty(),
        "model has no sleep-memory reserve slots"
    );
    let memory_layers = statuses
        .iter()
        .map(|status| status.layer)
        .collect::<BTreeSet<_>>();
    ensure!(
        !memory_layers.is_empty(),
        "model has no memory-bearing layers"
    );
    for (tier_index, (configured, saved)) in schedule.tiers.iter().zip(&state.tiers).enumerate() {
        ensure!(
            configured.id == saved.id && configured.reserve_slots == saved.slots.len(),
            "sleep tier {tier_index} topology differs between workflow and checkpoint"
        );
        ensure!(
            saved
                .last_update_clock
                .is_multiple_of(configured.update_period)
                && saved
                    .last_boundary_clock
                    .is_multiple_of(configured.update_period),
            "sleep tier `{}` checkpoint clocks are off schedule",
            configured.id
        );
        for (slot_index, slot) in saved.slots.iter().enumerate() {
            let model_slots = statuses
                .iter()
                .filter(|status| status.tier == tier_index && status.slot == slot_index)
                .collect::<Vec<_>>();
            ensure!(
                model_slots.len() == memory_layers.len(),
                "memory tier `{}` slot {slot_index} is not present in every memory layer",
                configured.id
            );
            ensure!(
                model_slots.iter().all(|status| {
                    status.tier_name == configured.id
                        && status.active == slot.active
                        && status.generation == slot.generation
                }),
                "memory tier `{}` slot {slot_index} mask/generation differs between model and trainer state",
                configured.id
            );
        }
    }
    ensure!(
        statuses.iter().all(|status| {
            schedule
                .tiers
                .get(status.tier)
                .is_some_and(|tier| status.tier_name == tier.id && status.slot < tier.reserve_slots)
        }),
        "model contains a memory slot absent from the sleep workflow"
    );
    let mut expected = Vec::new();
    for (sender, tier) in schedule.tiers.iter().enumerate() {
        let boundary = state.tiers[sender]
            .last_boundary_clock
            .checked_div(tier.update_period)
            .and_then(|multiple| multiple.checked_add(1))
            .and_then(|multiple| multiple.checked_mul(tier.update_period))
            .context("sleep resume boundary clock overflow")?;
        if boundary <= state.clock {
            let second = boundary
                .checked_add(tier.update_period)
                .context("sleep resume boundary clock overflow")?;
            ensure!(
                second > state.clock,
                "sleep checkpoint crosses multiple `{}` boundaries ({boundary} and {second}) without consuming the tier accumulator",
                tier.id,
            );
            expected.push((boundary, sender));
        }
    }
    expected.sort_unstable();
    let terminal = schedule.tiers.len() - 1;
    let terminal_retry_pending =
        state.tiers[terminal].last_update_clock < state.tiers[terminal].last_boundary_clock;
    if terminal_retry_pending
        && let Some(index) = expected.iter().position(|(_, sender)| *sender == terminal)
    {
        let retry = expected.remove(index);
        expected.insert(0, retry);
    }
    let expected_clocks = expected.iter().map(|(clock, _)| *clock).collect::<Vec<_>>();
    let expected_due = expected
        .into_iter()
        .map(|(_, sender)| sender)
        .collect::<Vec<_>>();
    ensure!(
        state.due_senders == expected_due && state.due_clocks == expected_clocks,
        "sleep checkpoint due-sender queue differs from its schedule clocks"
    );
    Ok(())
}

/// Backend contract for one atomic compute-consolidate-update transaction.
pub trait ConsolidationBackend {
    /// Number of independently sampled teacher/student rollouts consumed by
    /// Knowledge Seeding. Persisting the exact range makes a retry reproduce
    /// every sample instead of merely reseeding the first rollout.
    fn knowledge_rng_count(&self) -> Result<u64>;
    /// Number of group-relative continuations consumed by imitation/GRPO.
    fn imitation_rng_count(&self) -> Result<u64>;
    fn compute_prospective_update(&mut self, txn: &ConsolidationTxn) -> Result<()>;
    fn stage_student(&mut self, txn: &ConsolidationTxn) -> Result<()>;
    fn knowledge_seed(&mut self, txn: &ConsolidationTxn) -> Result<()>;
    fn learn_to_imitate(&mut self, txn: &ConsolidationTxn) -> Result<()>;
    fn retention_passes(&mut self, txn: &ConsolidationTxn) -> Result<bool>;
    /// Atomically publish the prospective sender update, receiver slot, sender
    /// reclamation, and corresponding optimizer moments. This call must be
    /// idempotent for `txn.id` so a checkpoint restored in `Commit` is safe.
    fn commit(&mut self, txn: &ConsolidationTxn) -> Result<CommittedCandidate>;
    fn restore_teacher(&mut self, txn: &ConsolidationTxn) -> Result<()>;
}

/// Exact immutable output of a successful backend commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedCandidate {
    pub checkpoint: String,
    pub sha256: String,
}

/// Durable checkpoint hook invoked after every sleep state transition and
/// after each isolated trial result. Production sinks publish the candidate
/// checkpoint and [`SleepState`] together before returning.
pub trait SleepProgressSink {
    fn persist(&mut self, state: &SleepState) -> Result<()>;

    /// Emit first-party sleep evidence at the same durable subphase boundary.
    /// Sinks which do not configure a metric journal may ignore it.
    fn metric(&mut self, _: MetricEvent) -> Result<()> {
        Ok(())
    }
}

struct NoopSleepProgress;

impl SleepProgressSink for NoopSleepProgress {
    fn persist(&mut self, _: &SleepState) -> Result<()> {
        Ok(())
    }
}

/// Distribution-distillation controls used during Knowledge Seeding.  The
/// rollout source is explicit because the paper uses both teacher samples and
/// detached student samples.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeSeedingConfig {
    pub chunk_tokens: usize,
    pub teacher_rollouts: usize,
    pub detached_student_rollouts: usize,
    pub temperature: f32,
    pub forward_kl_weight: f32,
}

impl KnowledgeSeedingConfig {
    /// Operational ceilings keep a malformed workflow from turning one sleep
    /// boundary into an unbounded rollout or logits allocation. They are well
    /// above the paper-reproduction recipe and may only be raised alongside
    /// memory-profile evidence.
    pub const MAX_CHUNK_TOKENS: usize = 16_384;
    pub const MAX_ROLLOUTS: usize = 1_024;
    pub const MAX_TOKEN_WORK: usize = Self::MAX_CHUNK_TOKENS * Self::MAX_ROLLOUTS;

    pub fn rollout_count(&self) -> Result<usize> {
        self.teacher_rollouts
            .checked_add(self.detached_student_rollouts)
            .context("knowledge-seeding rollout count overflow")
    }

    pub fn rollout_count_u64(&self) -> Result<u64> {
        self.rollout_count()?
            .try_into()
            .context("knowledge-seeding rollout count exceeds u64")
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.chunk_tokens > 0,
            "knowledge-seeding chunk_tokens must be positive"
        );
        ensure!(
            self.chunk_tokens <= Self::MAX_CHUNK_TOKENS,
            "knowledge-seeding chunk_tokens exceeds the {}-token operational limit",
            Self::MAX_CHUNK_TOKENS
        );
        ensure!(
            self.teacher_rollouts > 0 && self.detached_student_rollouts > 0,
            "knowledge seeding requires teacher and detached-student rollouts"
        );
        let rollout_count = self.rollout_count()?;
        ensure!(
            rollout_count <= Self::MAX_ROLLOUTS,
            "knowledge-seeding rollout count exceeds the {}-rollout operational limit",
            Self::MAX_ROLLOUTS
        );
        let token_work = self
            .chunk_tokens
            .checked_mul(rollout_count)
            .context("knowledge-seeding token work overflow")?;
        ensure!(
            token_work <= Self::MAX_TOKEN_WORK,
            "knowledge-seeding token work exceeds the {}-token operational limit",
            Self::MAX_TOKEN_WORK
        );
        u64::try_from(rollout_count).context("knowledge-seeding rollout count exceeds u64")?;
        ensure!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "knowledge-seeding temperature must be positive"
        );
        ensure!(
            self.forward_kl_weight.is_finite() && self.forward_kl_weight >= 0.0,
            "knowledge-seeding KL weight must be non-negative"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImitationConfig {
    pub semantic_judge_hash: String,
    pub semantic_weight: f32,
    pub maximum_edit_distance: usize,
    pub grpo_group_size: usize,
}

impl ImitationConfig {
    /// The edit reward uses a threshold-banded dynamic program and GRPO holds
    /// one group of rollouts together, so both axes need explicit ceilings.
    pub const MAX_EDIT_DISTANCE: usize = 4_096;
    pub const MAX_GRPO_GROUP_SIZE: usize = 256;
    pub const MAX_DISTANCE_WORK: usize = Self::MAX_EDIT_DISTANCE * Self::MAX_GRPO_GROUP_SIZE;

    pub fn group_size_u64(&self) -> Result<u64> {
        self.grpo_group_size
            .try_into()
            .context("imitation GRPO group size exceeds u64")
    }

    pub fn validate(&self) -> Result<()> {
        validate_sha256_identity(&self.semantic_judge_hash, "imitation semantic-judge hash")?;
        ensure!(
            self.semantic_weight.is_finite() && (0.0..=1.0).contains(&self.semantic_weight),
            "imitation semantic_weight must be in [0, 1]"
        );
        ensure!(
            self.grpo_group_size >= 2,
            "imitation GRPO group must contain at least two samples"
        );
        ensure!(
            self.maximum_edit_distance <= Self::MAX_EDIT_DISTANCE,
            "imitation maximum edit distance exceeds the {}-token operational limit",
            Self::MAX_EDIT_DISTANCE
        );
        ensure!(
            self.grpo_group_size <= Self::MAX_GRPO_GROUP_SIZE,
            "imitation GRPO group size exceeds the {}-sample operational limit",
            Self::MAX_GRPO_GROUP_SIZE
        );
        let distance_work = self
            .maximum_edit_distance
            .checked_mul(self.grpo_group_size)
            .context("imitation edit-distance work overflow")?;
        ensure!(
            distance_work <= Self::MAX_DISTANCE_WORK,
            "imitation edit-distance work exceeds the {}-cell operational limit",
            Self::MAX_DISTANCE_WORK
        );
        u64::try_from(self.grpo_group_size).context("imitation GRPO group size exceeds u64")?;
        Ok(())
    }
}

/// Reproduction defaults for the paper-inspired isolated adapter trial.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DreamingConfig {
    pub candidate_count: usize,
    pub retain_top: usize,
    pub retain_random: usize,
    pub lora_rank: usize,
    pub lora_alpha: usize,
    pub restem_iterations: usize,
    pub selector_version: String,
    pub reference_set_hash: String,
    pub trial_evaluator_hash: String,
}

impl DreamingConfig {
    /// Dream candidates and isolated adapters are materialized by the built-in
    /// backend. These ceilings make their maximum cardinality and rank an
    /// explicit part of the trainer contract instead of inheriting `usize`.
    pub const MAX_CANDIDATES: usize = 1_024;
    pub const MAX_LORA_RANK: usize = 1_024;
    pub const MAX_LORA_ALPHA: usize = 65_536;
    pub const MAX_RESTEM_ITERATIONS: usize = 128;
    pub const MAX_POLICY_WORK: usize = Self::MAX_CANDIDATES * Self::MAX_RESTEM_ITERATIONS;

    pub fn paper_reproduction() -> Self {
        Self {
            candidate_count: 64,
            retain_top: 8,
            retain_random: 2,
            lora_rank: 64,
            lora_alpha: 128,
            restem_iterations: 1,
            selector_version: "gradient-cosine-v1".into(),
            reference_set_hash: format!("sha256:{}", "0".repeat(64)),
            trial_evaluator_hash: format!("sha256:{}", "0".repeat(64)),
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.candidate_count > 0,
            "dream candidate count must be positive"
        );
        ensure!(
            self.candidate_count <= Self::MAX_CANDIDATES,
            "dream candidate count exceeds the {}-candidate operational limit",
            Self::MAX_CANDIDATES
        );
        let retained = self.retained_count()?;
        ensure!(
            retained <= self.candidate_count,
            "dream selection quotas exceed candidate count"
        );
        ensure!(
            retained > 0,
            "dream selection must retain at least one candidate"
        );
        for (value, label) in [
            (self.candidate_count, "dream candidate count"),
            (self.retain_top, "dream alignment quota"),
            (self.retain_random, "dream diversity quota"),
            (self.lora_rank, "dream LoRA rank"),
        ] {
            u32::try_from(value).with_context(|| format!("{label} exceeds metric schema"))?;
        }
        ensure!(
            self.lora_rank > 0 && self.lora_alpha > 0,
            "dream LoRA geometry must be positive"
        );
        ensure!(
            self.lora_rank <= Self::MAX_LORA_RANK,
            "dream LoRA rank exceeds the {}-rank operational limit",
            Self::MAX_LORA_RANK
        );
        ensure!(
            self.lora_alpha <= Self::MAX_LORA_ALPHA,
            "dream LoRA alpha exceeds the {} operational limit",
            Self::MAX_LORA_ALPHA
        );
        ensure!(
            self.restem_iterations > 0,
            "ReSTEM iterations must be positive"
        );
        ensure!(
            self.restem_iterations <= Self::MAX_RESTEM_ITERATIONS,
            "ReSTEM iterations exceed the {}-iteration operational limit",
            Self::MAX_RESTEM_ITERATIONS
        );
        let policy_work = retained
            .checked_mul(self.restem_iterations)
            .context("dream policy work overflow")?;
        ensure!(
            policy_work <= Self::MAX_POLICY_WORK,
            "dream policy work exceeds the {}-trial operational limit",
            Self::MAX_POLICY_WORK
        );
        ensure!(
            self.selector_version == "gradient-cosine-v1",
            "unsupported dream selector `{}`; this build implements only `gradient-cosine-v1`",
            self.selector_version
        );
        validate_sha256_identity(&self.reference_set_hash, "dream reference-set hash")?;
        validate_sha256_identity(&self.trial_evaluator_hash, "dream trial-evaluator hash")?;
        Ok(())
    }

    pub(crate) fn retained_count(&self) -> Result<usize> {
        self.retain_top
            .checked_add(self.retain_random)
            .context("dream selection quota overflow")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedDream {
    pub id: String,
    pub artifact_hash: String,
    pub gradient: Vec<f32>,
    pub diversity_key: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DreamTrial {
    pub candidate_id: String,
    pub adapter_hash: String,
    pub evaluator_hash: String,
    pub independent_task_improvement: f32,
}

/// Tensor-level dreaming implementation. Candidate generation receives no
/// corpus handle: its only source is model-owned wake contexts. Trials must be
/// isolated and the shared hash is checked around every trial.
pub trait DreamingBackend {
    /// Verify that the backend's shared model was loaded from this exact
    /// committed checkpoint. The checkpoint artifact identity and the model
    /// parameter fingerprint are deliberately separate: a serialized
    /// checkpoint hash is not necessarily a hash of only its tensors.
    fn verify_committed_candidate(&mut self, txn: &ConsolidationTxn) -> Result<()>;
    /// Return the sealed full-parameter identity of the shared candidate.
    /// Backends with interior mutability must recompute it; structurally
    /// immutable backends may return an identity cached when they sealed the
    /// candidate, avoiding a device-to-host model copy around every trial.
    fn shared_checkpoint_hash(&mut self) -> Result<String>;
    fn generate_from_wake_contexts(
        &mut self,
        txn: &ConsolidationTxn,
        candidate_count: usize,
        random_extra_expert: bool,
    ) -> Result<(String, Vec<GeneratedDream>)>;
    fn load_generated_dreams(
        &mut self,
        txn: &ConsolidationTxn,
        manifest: &str,
    ) -> Result<Vec<GeneratedDream>>;
    fn reference_gradient(
        &mut self,
        txn: &ConsolidationTxn,
        reference_set_hash: &str,
    ) -> Result<Vec<f32>>;
    fn isolated_lora_trial(
        &mut self,
        txn: &ConsolidationTxn,
        candidate: &GeneratedDream,
        rank: usize,
        alpha: usize,
    ) -> Result<DreamTrial>;
    /// Must be idempotent for `txn.id` and return the same content-addressed
    /// policy receipt after retry.
    fn restem_update(
        &mut self,
        txn: &ConsolidationTxn,
        accepted: &[DreamTrial],
        iterations: usize,
    ) -> Result<String>;
    /// Restore the immutable consolidated candidate if an isolated trial or
    /// policy update violates the shared-checkpoint boundary.
    fn restore_shared_candidate(&mut self, txn: &ConsolidationTxn) -> Result<()>;
}

/// Execute Dreaming without exposing a corpus/search interface or allowing an
/// isolated adapter to mutate the shared candidate checkpoint.
pub fn run_dreaming<B: DreamingBackend>(
    state: &mut SleepState,
    config: &DreamingConfig,
    backend: &mut B,
) -> Result<Vec<DreamTrial>> {
    run_dreaming_with_progress(state, config, backend, &mut NoopSleepProgress)
}

pub fn run_dreaming_with_progress<B: DreamingBackend, P: SleepProgressSink>(
    state: &mut SleepState,
    config: &DreamingConfig,
    backend: &mut B,
    progress: &mut P,
) -> Result<Vec<DreamTrial>> {
    config.validate()?;
    state
        .validate_resume()
        .context("validating Dreaming resume state")?;
    ensure!(
        state.pending.as_ref().is_some_and(|txn| txn.committed),
        "dreaming requires committed transaction metadata"
    );
    ensure_committed_candidate_binding(
        state.pending.as_ref().expect("transaction checked above"),
        backend,
        "binding Dreaming to the committed candidate",
    )?;
    loop {
        let txn = state
            .pending
            .clone()
            .ok_or_else(|| anyhow::anyhow!("dreaming has no transaction"))?;
        match state.phase {
            SleepPhase::Commit => {
                let shared_hash = read_shared_checkpoint_hash(
                    &txn,
                    backend,
                    "fingerprinting the committed candidate for Dreaming",
                )?;
                state
                    .pending
                    .as_mut()
                    .expect("transaction checked above")
                    .dream_shared_checkpoint_hash = Some(shared_hash);
                state.transition(SleepPhase::DreamGeneration)?;
                progress.persist(state)?;
            }
            SleepPhase::DreamGeneration => {
                state.reserve_dream_generation_rng(
                    0,
                    config
                        .candidate_count
                        .try_into()
                        .context("dream candidate count exceeds RNG schema")?,
                )?;
                progress.persist(state)?;
                let txn = state.pending.clone().expect("transaction checked above");
                let (manifest, candidates) = run_shared_checkpoint_operation(
                    &txn,
                    backend,
                    "dream generation",
                    |backend| {
                        backend.generate_from_wake_contexts(&txn, config.candidate_count, true)
                    },
                )?;
                validate_generated_dreams(&candidates, config.candidate_count)?;
                state.record_generated_manifest(manifest)?;
                state.transition(SleepPhase::DreamRanking)?;
                progress.persist(state)?;
            }
            SleepPhase::DreamRanking => {
                let selection_rng = state.reserve_dream_selection_rng(0)?;
                progress.persist(state)?;
                let txn = state.pending.clone().expect("transaction checked above");
                let manifest = txn
                    .generated_manifest
                    .as_deref()
                    .expect("validated dream manifest");
                let candidates = run_shared_checkpoint_operation(
                    &txn,
                    backend,
                    "loading generated dreams",
                    |backend| backend.load_generated_dreams(&txn, manifest),
                )?;
                validate_generated_dreams(&candidates, config.candidate_count)?;
                let reference = run_shared_checkpoint_operation(
                    &txn,
                    backend,
                    "dream reference-gradient evaluation",
                    |backend| backend.reference_gradient(&txn, &config.reference_set_hash),
                )?;
                ensure!(
                    !reference.is_empty()
                        && reference.len() <= MAX_DREAM_GRADIENT_DIMENSIONS
                        && reference.iter().all(|value| value.is_finite()),
                    "dream reference gradient is empty, non-finite, or exceeds the operational dimension limit"
                );
                let scores = candidates
                    .iter()
                    .map(|candidate| {
                        Ok(DreamCandidateScore {
                            id: candidate.id.clone(),
                            importance: gradient_cosine(&candidate.gradient, &reference)?,
                            diversity_key: candidate.diversity_key,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut scores = scores;
                let salt = selection_rng
                    .start
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left(17);
                for score in &mut scores {
                    score.diversity_key ^= salt;
                }
                let selected = select_dreams(&scores, config.retain_top, config.retain_random)?;
                let cosine_mean = scores
                    .iter()
                    .map(|score| f64::from(score.importance))
                    .sum::<f64>()
                    / scores.len() as f64;
                let cosine_max = scores
                    .iter()
                    .map(|score| f64::from(score.importance))
                    .fold(f64::NEG_INFINITY, f64::max);
                progress.metric(MetricEvent::DreamSelection(DreamSelectionMetrics {
                    transaction_id: format!("sleep-{}", txn.id),
                    selector_version: config.selector_version.clone(),
                    reference_set_hash: config.reference_set_hash.clone(),
                    candidates_generated: config
                        .candidate_count
                        .try_into()
                        .context("dream candidate count exceeds metric schema")?,
                    selected_by_alignment: config
                        .retain_top
                        .try_into()
                        .context("dream alignment quota exceeds metric schema")?,
                    selected_random: config
                        .retain_random
                        .try_into()
                        .context("dream diversity quota exceeds metric schema")?,
                    random_quota: config
                        .retain_random
                        .try_into()
                        .context("dream diversity quota exceeds metric schema")?,
                    gradient_cosine_mean: cosine_mean,
                    gradient_cosine_max: cosine_max,
                    selected_manifest_hash: manifest.to_owned(),
                }))?;
                state
                    .pending
                    .as_mut()
                    .expect("transaction checked above")
                    .dream_selected = selected;
                state.transition(SleepPhase::DreamTrials)?;
                progress.persist(state)?;
            }
            SleepPhase::DreamTrials => {
                let manifest = txn
                    .generated_manifest
                    .as_deref()
                    .expect("validated dream manifest");
                let candidates = run_shared_checkpoint_operation(
                    &txn,
                    backend,
                    "loading dreams for isolated trials",
                    |backend| backend.load_generated_dreams(&txn, manifest),
                )?;
                validate_generated_dreams(&candidates, config.candidate_count)?;
                let completed = txn
                    .dream_trials
                    .iter()
                    .map(|trial| trial.candidate_id.as_str())
                    .collect::<BTreeSet<_>>();
                for id in &txn.dream_selected {
                    if completed.contains(id.as_str()) {
                        continue;
                    }
                    let candidate = candidates
                        .iter()
                        .find(|candidate| candidate.id == *id)
                        .ok_or_else(|| anyhow::anyhow!("selected dream `{id}` disappeared"))?;
                    state.reserve_dream_trial_rng(0, id)?;
                    progress.persist(state)?;
                    let current_txn = state.pending.clone().expect("transaction checked above");
                    let trial = run_shared_checkpoint_operation(
                        &current_txn,
                        backend,
                        &format!("isolated dream trial `{id}`"),
                        |backend| {
                            backend.isolated_lora_trial(
                                &current_txn,
                                candidate,
                                config.lora_rank,
                                config.lora_alpha,
                            )
                        },
                    )?;
                    ensure!(
                        trial.candidate_id == *id && trial.independent_task_improvement.is_finite(),
                        "dream trial `{id}` returned incomplete or invalid evidence"
                    );
                    validate_sha256_identity(&trial.adapter_hash, "dream trial adapter hash")?;
                    validate_sha256_identity(&trial.evaluator_hash, "dream trial evaluator hash")?;
                    ensure!(
                        trial.evaluator_hash == config.trial_evaluator_hash,
                        "dream trial `{id}` used an evaluator other than the frozen configured artifact"
                    );
                    progress.metric(MetricEvent::DreamTrial(DreamTrialMetrics {
                        transaction_id: format!("sleep-{}", txn.id),
                        candidate_hash: candidate.artifact_hash.clone(),
                        adapter_hash: trial.adapter_hash.clone(),
                        evaluator_hash: trial.evaluator_hash.clone(),
                        lora_rank: config
                            .lora_rank
                            .try_into()
                            .context("dream LoRA rank exceeds metric schema")?,
                        lora_alpha: config.lora_alpha as f64,
                        independent_task_delta: f64::from(trial.independent_task_improvement),
                        reward: f64::from(trial.independent_task_improvement),
                        elapsed_seconds: 0.0,
                        accepted: trial.independent_task_improvement > 0.0,
                        isolated: true,
                        shared_checkpoint_unchanged: true,
                    }))?;
                    state
                        .pending
                        .as_mut()
                        .expect("transaction checked above")
                        .dream_trials
                        .push(trial);
                    progress.persist(state)?;
                }
                state.transition(SleepPhase::DreamPolicyUpdate)?;
                progress.persist(state)?;
            }
            SleepPhase::DreamPolicyUpdate => {
                let accepted = txn
                    .dream_trials
                    .iter()
                    .filter(|trial| trial.independent_task_improvement > 0.0)
                    .cloned()
                    .collect::<Vec<_>>();
                if txn.dream_policy_receipt.is_none() {
                    let receipt = run_shared_checkpoint_operation(
                        &txn,
                        backend,
                        "ReSTEM policy update",
                        |backend| backend.restem_update(&txn, &accepted, config.restem_iterations),
                    )?;
                    validate_sha256_identity(&receipt, "ReSTEM policy receipt")?;
                    state
                        .pending
                        .as_mut()
                        .expect("transaction checked above")
                        .dream_policy_receipt = Some(receipt);
                    progress.persist(state)?;
                }
                state.transition(SleepPhase::Candidate)?;
                progress.persist(state)?;
            }
            SleepPhase::Candidate => {
                ensure_shared_checkpoint_unchanged(
                    &txn,
                    backend,
                    "finalizing the dream candidate",
                )?;
                return Ok(txn
                    .dream_trials
                    .into_iter()
                    .filter(|trial| trial.independent_task_improvement > 0.0)
                    .collect());
            }
            phase => bail!("dreaming cannot resume from subphase {phase:?}"),
        }
    }
}

fn validate_generated_dreams(candidates: &[GeneratedDream], expected: usize) -> Result<()> {
    ensure!(
        candidates.len() == expected,
        "dream generator returned {} candidates; expected {expected}",
        candidates.len()
    );
    let mut ids = BTreeSet::new();
    let mut dimensions = None;
    let mut gradient_elements = 0usize;
    for candidate in candidates {
        ensure!(
            !candidate.id.trim().is_empty()
                && candidate.id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES
                && !candidate.gradient.is_empty()
                && candidate.gradient.len() <= MAX_DREAM_GRADIENT_DIMENSIONS
                && candidate.gradient.iter().all(|value| value.is_finite()),
            "generated dream has incomplete identity or gradient"
        );
        ensure!(
            dimensions.is_none_or(|expected| expected == candidate.gradient.len()),
            "generated dreams use inconsistent gradient dimensions"
        );
        dimensions = Some(candidate.gradient.len());
        gradient_elements = gradient_elements
            .checked_add(candidate.gradient.len())
            .context("generated-dream gradient element count overflow")?;
        ensure!(
            gradient_elements <= DreamingConfig::MAX_CANDIDATES * MAX_DREAM_GRADIENT_DIMENSIONS,
            "generated-dream gradient work exceeds its operational limit"
        );
        validate_sha256_identity(&candidate.artifact_hash, "generated dream artifact hash")?;
        ensure!(
            ids.insert(candidate.id.as_str()),
            "duplicate dream candidate `{}`",
            candidate.id
        );
    }
    Ok(())
}

fn ensure_shared_checkpoint_unchanged<B: DreamingBackend>(
    txn: &ConsolidationTxn,
    backend: &mut B,
    operation: &str,
) -> Result<()> {
    let expected = txn
        .dream_shared_checkpoint_hash
        .as_deref()
        .context("dream transaction has no shared checkpoint hash")?;
    ensure_shared_checkpoint_hash(txn, backend, expected, operation)
}

fn ensure_committed_candidate_binding<B: DreamingBackend>(
    txn: &ConsolidationTxn,
    backend: &mut B,
    operation: &str,
) -> Result<()> {
    let first_error = match backend.verify_committed_candidate(txn) {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };
    backend.restore_shared_candidate(txn).with_context(|| {
        format!("{operation} failed and restoration of the committed candidate also failed")
    })?;
    backend.verify_committed_candidate(txn).with_context(|| {
        format!(
            "{operation} failed ({first_error:#}); restoration did not recover the committed candidate"
        )
    })?;
    bail!(
        "{operation} did not start from the committed candidate; immutable candidate was restored"
    )
}

fn read_shared_checkpoint_hash<B: DreamingBackend>(
    txn: &ConsolidationTxn,
    backend: &mut B,
    operation: &str,
) -> Result<String> {
    let observation = backend.shared_checkpoint_hash().and_then(|observed| {
        validate_sha256_identity(&observed, "shared-checkpoint hash")?;
        Ok(observed)
    });
    match observation {
        Ok(hash) => Ok(hash),
        Err(error) => {
            backend.restore_shared_candidate(txn).with_context(|| {
                format!("{operation} failed and restoration of the committed candidate also failed")
            })?;
            backend
                .verify_committed_candidate(txn)
                .with_context(|| format!("verifying the restored candidate after {operation}"))?;
            bail!("{operation} failed ({error:#}); immutable candidate was restored")
        }
    }
}

fn ensure_shared_checkpoint_hash<B: DreamingBackend>(
    txn: &ConsolidationTxn,
    backend: &mut B,
    expected: &str,
    operation: &str,
) -> Result<()> {
    validate_sha256_identity(expected, "expected shared-checkpoint hash")?;
    let observation = backend.shared_checkpoint_hash().and_then(|observed| {
        validate_sha256_identity(&observed, "observed shared-checkpoint hash")?;
        Ok(observed)
    });
    if observation
        .as_ref()
        .is_ok_and(|observed| observed == expected)
    {
        return Ok(());
    }

    // If an operation returned an error, malformed evidence, or even made the
    // fingerprint temporarily unreadable, treat the shared candidate as
    // suspect. Restoration is mandatory before the caller can retry.
    backend.restore_shared_candidate(txn).with_context(|| {
        format!(
            "{operation} did not preserve a verifiable shared checkpoint and restoration failed"
        )
    })?;
    let restored = backend
        .shared_checkpoint_hash()
        .with_context(|| format!("reading the shared checkpoint after restoring {operation}"))?;
    validate_sha256_identity(&restored, "restored shared-checkpoint hash")?;
    ensure!(
        restored == expected,
        "{operation} did not preserve the shared checkpoint and restoration recovered a different candidate"
    );
    match observation {
        Ok(observed) => bail!(
            "{operation} mutated the shared checkpoint from {expected} to {observed}; immutable candidate was restored"
        ),
        Err(error) => bail!(
            "{operation} left the shared checkpoint unverifiable ({error:#}); immutable candidate was restored"
        ),
    }
}

/// Run one potentially fallible Dreaming backend operation behind the shared
/// candidate boundary. The postcondition is checked regardless of whether the
/// operation succeeds; a backend error can never bypass restoration.
fn run_shared_checkpoint_operation<B, T>(
    txn: &ConsolidationTxn,
    backend: &mut B,
    operation: &str,
    action: impl FnOnce(&mut B) -> Result<T>,
) -> Result<T>
where
    B: DreamingBackend,
{
    ensure_shared_checkpoint_unchanged(txn, backend, &format!("before {operation}"))?;
    let result = action(backend);
    let integrity = ensure_shared_checkpoint_unchanged(txn, backend, operation);
    match (result, integrity) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error).with_context(|| format!("{operation} failed")),
        (Ok(_), Err(integrity_error)) => Err(integrity_error),
        (Err(error), Err(integrity_error)) => bail!(
            "{operation} failed: {error:#}; shared-candidate recovery also reported: {integrity_error:#}"
        ),
    }
}

/// Execute the non-dreaming half of sleep with rollback on every failure.
pub fn run_consolidation<B: ConsolidationBackend>(
    state: &mut SleepState,
    backend: &mut B,
) -> Result<bool> {
    run_consolidation_with_progress(state, backend, &mut NoopSleepProgress)
}

pub fn run_consolidation_with_progress<B: ConsolidationBackend, P: SleepProgressSink>(
    state: &mut SleepState,
    backend: &mut B,
    progress: &mut P,
) -> Result<bool> {
    state
        .validate_resume()
        .context("validating consolidation resume state")?;
    progress.persist(state)?;
    let txn = state
        .pending
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no consolidation transaction"))?;
    let result = (|| loop {
        match state.phase {
            SleepPhase::ProspectiveUpdate => {
                backend.compute_prospective_update(&txn)?;
                backend.stage_student(&txn)?;
                state.transition(SleepPhase::KnowledgeSeeding)?;
                progress.persist(state)?;
            }
            SleepPhase::KnowledgeSeeding => {
                state.reserve_knowledge_rng(0, backend.knowledge_rng_count()?)?;
                progress.persist(state)?;
                let current_txn = state.pending.clone().expect("transaction checked above");
                backend.knowledge_seed(&current_txn)?;
                state.transition(SleepPhase::Imitation)?;
                progress.persist(state)?;
            }
            SleepPhase::Imitation => {
                state.reserve_imitation_rng(0, backend.imitation_rng_count()?)?;
                progress.persist(state)?;
                let current_txn = state.pending.clone().expect("transaction checked above");
                backend.learn_to_imitate(&current_txn)?;
                state.transition(SleepPhase::RetentionValidation)?;
                progress.persist(state)?;
            }
            SleepPhase::RetentionValidation => {
                if !backend.retention_passes(&txn)? {
                    break Ok(false);
                }
                state.transition(SleepPhase::Commit)?;
                progress.persist(state)?;
            }
            SleepPhase::Commit => {
                if !state
                    .pending
                    .as_ref()
                    .is_some_and(|transaction| transaction.committed)
                {
                    let candidate = backend.commit(&txn)?;
                    state.record_committed_candidate(candidate.checkpoint, candidate.sha256)?;
                    state.commit_consolidation()?;
                    progress.persist(state)?;
                }
                break Ok(true);
            }
            SleepPhase::DreamGeneration
            | SleepPhase::DreamRanking
            | SleepPhase::DreamTrials
            | SleepPhase::DreamPolicyUpdate
            | SleepPhase::Candidate => break Ok(true),
            SleepPhase::Wake => bail!("consolidation transaction is in wake phase"),
        }
    })();

    match result {
        Ok(true) => Ok(true),
        Ok(false) => {
            backend.restore_teacher(&txn)?;
            let mut restored = state.clone();
            restored.rollback()?;
            progress.persist(&restored)?;
            *state = restored;
            Ok(false)
        }
        Err(error) => {
            if state
                .pending
                .as_ref()
                .is_some_and(|transaction| transaction.committed)
            {
                return Err(error.context(
                    "consolidation committed in memory but durable state publication failed; retry persistence without rolling back",
                ));
            }
            if let Err(restore_error) = backend.restore_teacher(&txn) {
                bail!(
                    "consolidation failed: {error:#}; teacher restore also failed: {restore_error:#}"
                );
            }
            let mut restored = state.clone();
            restored.rollback()?;
            progress
                .persist(&restored)
                .context("failed to persist restored consolidation rollback")?;
            *state = restored;
            Err(error)
        }
    }
}

/// Thresholded normalized edit similarity from the paper's imitation reward.
pub fn normalized_levenshtein_reward(
    generated: &[u32],
    teacher: &[u32],
    maximum_distance: usize,
) -> f32 {
    let distance = levenshtein(generated, teacher);
    if distance > maximum_distance {
        return 0.0;
    }
    let length = generated.len().max(teacher.len());
    if length == 0 {
        return 1.0;
    }
    1.0 - distance as f32 / length as f32
}

pub fn imitation_reward(semantic_equivalent: bool, edit_reward: f32, gamma: f32) -> Result<f32> {
    ensure!(
        gamma.is_finite() && (0.0..=1.0).contains(&gamma),
        "semantic reward weight must be in [0, 1]"
    );
    ensure!(
        edit_reward.is_finite() && (0.0..=1.0).contains(&edit_reward),
        "edit reward must be in [0, 1]"
    );
    Ok(gamma * f32::from(semantic_equivalent) + (1.0 - gamma) * edit_reward)
}

fn levenshtein(left: &[u32], right: &[u32]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];
    for (i, &a) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, &b) in right.iter().enumerate() {
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + usize::from(a != b));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// Scalarize the paper's underspecified gradient score as cosine alignment
/// with a training-only reference gradient.
pub fn gradient_cosine(candidate: &[f32], reference: &[f32]) -> Result<f32> {
    ensure!(
        !candidate.is_empty() && candidate.len() == reference.len(),
        "gradient vectors must have the same positive length"
    );
    let mut dot = 0.0f64;
    let mut candidate_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    for (&left, &right) in candidate.iter().zip(reference) {
        ensure!(
            left.is_finite() && right.is_finite(),
            "gradient contains non-finite value"
        );
        dot += f64::from(left) * f64::from(right);
        candidate_norm += f64::from(left) * f64::from(left);
        reference_norm += f64::from(right) * f64::from(right);
    }
    if candidate_norm == 0.0 || reference_norm == 0.0 {
        return Ok(0.0);
    }
    Ok((dot / (candidate_norm.sqrt() * reference_norm.sqrt())) as f32)
}

#[derive(Clone, Debug, PartialEq)]
pub struct DreamCandidateScore {
    pub id: String,
    pub importance: f32,
    /// Deterministic key from the checkpointed dream RNG stream.
    pub diversity_key: u64,
}

/// Keep the highest-scoring candidates plus a disjoint deterministic random
/// quota.  IDs make the output stable across input ordering.
pub fn select_dreams(
    candidates: &[DreamCandidateScore],
    top: usize,
    random: usize,
) -> Result<Vec<String>> {
    let retained = top
        .checked_add(random)
        .context("dream selection quota overflow")?;
    ensure!(
        retained <= candidates.len(),
        "dream selection exceeds candidate count"
    );
    ensure!(
        candidates
            .iter()
            .all(|candidate| candidate.importance.is_finite()),
        "dream importance must be finite"
    );
    let unique_ids = candidates
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        unique_ids.len() == candidates.len()
            && unique_ids
                .iter()
                .all(|id| { !id.trim().is_empty() && id.len() <= MAX_DREAM_CANDIDATE_ID_BYTES }),
        "dream candidates must have unique, non-empty IDs"
    );
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|left, right| {
        right
            .importance
            .total_cmp(&left.importance)
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut selected = ranked
        .iter()
        .take(top)
        .map(|candidate| candidate.id.clone())
        .collect::<BTreeSet<_>>();
    let mut diversity = candidates
        .iter()
        .filter(|candidate| !selected.contains(&candidate.id))
        .cloned()
        .collect::<Vec<_>>();
    diversity.sort_by(|left, right| {
        left.diversity_key
            .cmp(&right.diversity_key)
            .then_with(|| left.id.cmp(&right.id))
    });
    selected.extend(
        diversity
            .into_iter()
            .take(random)
            .map(|candidate| candidate.id),
    );
    Ok(selected.into_iter().collect())
}

/// Group-relative standardized advantages used by the default LTI GRPO step.
pub fn group_relative_advantages(rewards: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    ensure!(!rewards.is_empty(), "GRPO reward group is empty");
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "GRPO epsilon must be positive"
    );
    ensure!(
        rewards.iter().all(|reward| reward.is_finite()),
        "GRPO reward is non-finite"
    );
    let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
    let variance = rewards
        .iter()
        .map(|reward| (reward - mean).powi(2))
        .sum::<f32>()
        / rewards.len() as f32;
    let scale = variance.sqrt().max(epsilon);
    Ok(rewards
        .iter()
        .map(|reward| (reward - mean) / scale)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::ParamId;

    fn test_hash(digit: char) -> String {
        format!("sha256:{}", digit.to_string().repeat(64))
    }

    fn schedule() -> SleepSchedule {
        SleepSchedule {
            clock: UpdateClock::OptimizerSteps,
            terminal_consolidation: TerminalConsolidation::DistillIntoBaseV1,
            tiers: vec![
                MemoryTierSchedule {
                    id: "fast".into(),
                    update_period: 2,
                    reserve_slots: 2,
                },
                MemoryTierSchedule {
                    id: "mid".into(),
                    update_period: 4,
                    reserve_slots: 2,
                },
                MemoryTierSchedule {
                    id: "slow".into(),
                    update_period: 8,
                    reserve_slots: 2,
                },
            ],
        }
    }

    #[test]
    fn schedule_orders_coincident_boundaries_fastest_first() {
        let schedule = schedule();
        schedule.validate().unwrap();
        assert_eq!(schedule.due_senders(2), vec![0]);
        assert_eq!(schedule.due_senders(4), vec![0, 1]);
        assert_eq!(schedule.due_senders(8), vec![0, 1, 2]);
        assert!(schedule.due_senders(0).is_empty());
    }

    #[test]
    fn schedule_rejects_non_divisible_periods() {
        let mut schedule = schedule();
        schedule.tiers[1].update_period = 3;
        assert!(schedule.validate().is_err());
    }

    #[test]
    fn sleep_topology_and_rng_state_are_operationally_bounded() {
        let bounded_tiers = |count: usize, reserve_slots: usize| SleepSchedule {
            clock: UpdateClock::OptimizerSteps,
            terminal_consolidation: TerminalConsolidation::DistillIntoBaseV1,
            tiers: (0..count)
                .map(|index| MemoryTierSchedule {
                    id: format!("tier-{index}"),
                    update_period: 1_u64 << index,
                    reserve_slots,
                })
                .collect(),
        };

        assert!(bounded_tiers(MAX_MEMORY_TIERS + 1, 2).validate().is_err());
        assert!(
            bounded_tiers(2, MAX_RESERVE_SLOTS_PER_TIER + 1)
                .validate()
                .is_err()
        );
        assert!(
            bounded_tiers(5, MAX_RESERVE_SLOTS_PER_TIER)
                .validate()
                .is_err()
        );

        let schedule = schedule();
        assert!(SleepState::new(&schedule, 0).is_err());
        assert!(SleepState::new(&schedule, MAX_SLEEP_RNG_STREAMS + 1).is_err());
        let mut state = SleepState::new(&schedule, 1).unwrap();
        state.rng_counters.resize(MAX_SLEEP_RNG_STREAMS + 1, 0);
        assert!(state.validate_resume().is_err());
    }

    #[test]
    fn rng_reservations_without_a_transaction_are_failure_atomic() {
        let mut state = SleepState::new(&schedule(), 1).unwrap();
        let initial = state.clone();

        assert!(state.reserve_knowledge_rng(0, 1).is_err());
        assert_eq!(state, initial);
        assert!(state.reserve_imitation_rng(0, 1).is_err());
        assert_eq!(state, initial);
        assert!(state.reserve_dream_generation_rng(0, 1).is_err());
        assert_eq!(state, initial);
        assert!(state.reserve_dream_selection_rng(0).is_err());
        assert_eq!(state, initial);
        assert!(state.reserve_dream_trial_rng(0, "candidate").is_err());
        assert_eq!(state, initial);
    }

    #[test]
    fn rng_reservations_in_the_wrong_subphase_are_failure_atomic() {
        let mut state = begun_state();
        let prospective = state.clone();
        assert!(state.reserve_knowledge_rng(0, 1).is_err());
        assert_eq!(state, prospective);

        state.transition(SleepPhase::KnowledgeSeeding).unwrap();
        let knowledge = state.clone();
        assert!(state.reserve_imitation_rng(0, 1).is_err());
        assert_eq!(state, knowledge);
        assert!(state.reserve_dream_generation_rng(0, 1).is_err());
        assert_eq!(state, knowledge);
    }

    #[test]
    fn resume_rejects_multiple_unconsumed_boundaries_for_one_sender() {
        let mut state = SleepState::new(&schedule(), 1).unwrap();
        state.clock = 4;
        state.due_senders = vec![0, 0];
        state.due_clocks = vec![2, 4];

        let error = state.validate_resume().unwrap_err().to_string();
        assert!(error.contains("due-boundary queue"), "{error}");
    }

    #[test]
    fn immutable_identities_require_canonical_sha256() {
        let mut state = SleepState::new(&schedule(), 1).unwrap();
        state.advance_clock(&schedule(), 2).unwrap();
        assert!(
            state
                .begin(
                    0,
                    "teacher".into(),
                    "sha256:not-a-digest".into(),
                    "student".into(),
                    test_hash('b'),
                    test_hash('c'),
                )
                .is_err()
        );
        assert!(
            ImitationConfig {
                semantic_judge_hash: "sha256:ABC".into(),
                semantic_weight: 0.5,
                maximum_edit_distance: 2,
                grpo_group_size: 2,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn tensor_generation_is_bound_to_its_manifest_without_partial_mutation() {
        let mut state = begun_state();
        let before = state.clone();

        let error = state
            .record_tensor_transaction(format!("sha256-{}", "1".repeat(64)), test_hash('2'))
            .unwrap_err()
            .to_string();

        assert!(error.contains("does not match"), "{error}");
        assert_eq!(state, before);
    }

    #[test]
    fn resume_rejects_transaction_clock_mask_and_dream_drift() {
        let mut wrong_clock = begun_state();
        wrong_clock.pending.as_mut().unwrap().trigger_clock += 1;
        assert!(wrong_clock.validate_resume().is_err());

        let mut wrong_receiver_mask = begun_state();
        wrong_receiver_mask.tiers[1].slots[0].active = true;
        assert!(wrong_receiver_mask.validate_resume().is_err());

        let mut wrong_committed_clock = committed_state();
        wrong_committed_clock
            .pending
            .as_mut()
            .unwrap()
            .trigger_clock = 1;
        let error = wrong_committed_clock
            .validate_resume()
            .unwrap_err()
            .to_string();
        assert!(error.contains("sender clocks"), "{error}");

        let mut duplicate_selection = committed_state();
        duplicate_selection
            .pending
            .as_mut()
            .unwrap()
            .dream_shared_checkpoint_hash = Some(test_hash('d'));
        duplicate_selection
            .transition(SleepPhase::DreamGeneration)
            .unwrap();
        duplicate_selection
            .record_generated_manifest(test_hash('7'))
            .unwrap();
        duplicate_selection
            .transition(SleepPhase::DreamRanking)
            .unwrap();
        duplicate_selection.pending.as_mut().unwrap().dream_selected =
            vec!["same".into(), "same".into()];
        duplicate_selection
            .transition(SleepPhase::DreamTrials)
            .unwrap();
        assert!(duplicate_selection.validate_resume().is_err());
    }

    #[test]
    fn resume_rejects_forged_terminal_topology_and_incomplete_reset_plan() {
        let mut forged_terminal = begun_state();
        let pending = forged_terminal.pending.as_mut().unwrap();
        pending.terminal = true;
        pending.receiver = pending.sender;
        let error = forged_terminal.validate_resume().unwrap_err().to_string();
        assert!(error.contains("terminal marker"), "{error}");

        let schedule = schedule();
        let mut omitted_slot = SleepState::new(&schedule, 1).unwrap();
        omitted_slot.tiers[0].slots[1].active = true;
        omitted_slot.advance_clock(&schedule, 2).unwrap();
        omitted_slot
            .begin(
                0,
                "teacher/reset-plan".into(),
                test_hash('a'),
                "student/reset-plan".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        assert_eq!(
            omitted_slot.pending.as_ref().unwrap().sender_slots_to_reset,
            vec![1]
        );
        omitted_slot
            .pending
            .as_mut()
            .unwrap()
            .sender_slots_to_reset
            .clear();
        let error = omitted_slot.validate_resume().unwrap_err().to_string();
        assert!(error.contains("exact active-slot set"), "{error}");

        let mut committed_omission = SleepState::new(&schedule, 1).unwrap();
        committed_omission.tiers[0].slots[1].active = true;
        committed_omission.advance_clock(&schedule, 2).unwrap();
        committed_omission
            .begin(
                0,
                "teacher/committed-reset-plan".into(),
                test_hash('a'),
                "student/committed-reset-plan".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        committed_omission
            .transition(SleepPhase::KnowledgeSeeding)
            .unwrap();
        committed_omission
            .transition(SleepPhase::Imitation)
            .unwrap();
        committed_omission
            .transition(SleepPhase::RetentionValidation)
            .unwrap();
        committed_omission.transition(SleepPhase::Commit).unwrap();
        committed_omission
            .record_committed_candidate("candidate/committed-reset-plan".into(), test_hash('d'))
            .unwrap();
        committed_omission.commit_consolidation().unwrap();
        committed_omission.tiers[0].slots[1].active = true;
        committed_omission
            .pending
            .as_mut()
            .unwrap()
            .sender_slots_to_reset
            .clear();
        let error = committed_omission
            .validate_resume()
            .unwrap_err()
            .to_string();
        assert!(error.contains("retains an active reserve slot"), "{error}");
    }

    #[test]
    fn schedule_rejects_receiver_capacity_that_exhausts_before_reclamation() {
        let mut schedule = schedule();
        schedule.tiers[1].update_period = 8;
        schedule.tiers[2].update_period = 16;
        let error = schedule.validate().unwrap_err().to_string();
        assert!(error.contains("at least 4 reserve slots"), "{error}");
    }

    #[test]
    fn terminal_base_consolidation_bounds_capacity_past_two_slow_periods() {
        let schedule = schedule();
        let mut state = SleepState::new(&schedule, 1).unwrap();
        let mut terminal_cycles = 0;
        for clock in (2..=18).step_by(2) {
            state.advance_clock(&schedule, clock).unwrap();
            while let Some(sender) = state.next_due_sender() {
                let txn = state
                    .begin(
                        sender,
                        format!("teacher/{clock}/{sender}"),
                        format!("sha256:{}", "a".repeat(64)),
                        format!("student/{clock}/{sender}"),
                        format!("sha256:{}", "b".repeat(64)),
                        format!("sha256:{}", "c".repeat(64)),
                    )
                    .unwrap();
                state.transition(SleepPhase::KnowledgeSeeding).unwrap();
                state.reserve_knowledge_rng(0, 1).unwrap();
                state.transition(SleepPhase::Imitation).unwrap();
                state.reserve_imitation_rng(0, 1).unwrap();
                state.transition(SleepPhase::RetentionValidation).unwrap();
                state.transition(SleepPhase::Commit).unwrap();
                state
                    .record_committed_candidate(
                        format!("candidate/{clock}/{sender}"),
                        test_hash('d'),
                    )
                    .unwrap();
                state.commit_consolidation().unwrap();
                if txn.terminal {
                    terminal_cycles += 1;
                    assert!(state.tiers[2].slots.iter().all(|slot| !slot.active));
                }
                state.transition(SleepPhase::Candidate).unwrap();
                state.finish_candidate().unwrap();
            }
        }
        assert_eq!(terminal_cycles, 2);
        assert!(state.clock > 2 * schedule.tiers[2].update_period);
        assert!(state.tiers[2].slots.iter().all(|slot| !slot.active));
    }

    #[test]
    fn model_masks_and_generations_must_match_every_layer() {
        let schedule = schedule();
        let state = SleepState::new(&schedule, 1).unwrap();
        let mut statuses = (0..2)
            .flat_map(|layer| {
                schedule
                    .tiers
                    .iter()
                    .enumerate()
                    .flat_map(move |(tier, definition)| {
                        (0..definition.reserve_slots).map(move |slot| MemorySlotStatus {
                            layer,
                            tier,
                            tier_name: definition.id.clone(),
                            slot,
                            active: false,
                            generation: 0,
                            parameter_ids: vec![ParamId::new()],
                        })
                    })
            })
            .collect::<Vec<_>>();
        validate_model_memory_state(&schedule, &state, &statuses).unwrap();
        statuses
            .iter_mut()
            .find(|status| status.layer == 1 && status.tier == 1 && status.slot == 0)
            .unwrap()
            .generation = 1;
        assert!(validate_model_memory_state(&schedule, &state, &statuses).is_err());
    }

    #[test]
    fn model_resume_rejects_a_due_queue_or_clock_off_schedule() {
        let schedule = schedule();
        let mut state = SleepState::new(&schedule, 1).unwrap();
        state.clock = 4;
        state.due_senders = vec![1];
        let statuses = (0..2)
            .flat_map(|layer| {
                schedule
                    .tiers
                    .iter()
                    .enumerate()
                    .flat_map(move |(tier, definition)| {
                        (0..definition.reserve_slots).map(move |slot| MemorySlotStatus {
                            layer,
                            tier,
                            tier_name: definition.id.clone(),
                            slot,
                            active: false,
                            generation: 0,
                            parameter_ids: vec![ParamId::new()],
                        })
                    })
            })
            .collect::<Vec<_>>();
        assert!(validate_model_memory_state(&schedule, &state, &statuses).is_err());

        state.due_senders = vec![0, 1];
        state.tiers[0].last_boundary_clock = 3;
        assert!(validate_model_memory_state(&schedule, &state, &statuses).is_err());
    }

    #[test]
    fn commit_activates_receiver_and_reclaims_sender_slots() {
        let schedule = schedule();
        let mut state = SleepState::new(&schedule, 3).unwrap();
        state.tiers[0].slots[1].active = true;
        state.advance_clock(&schedule, 2).unwrap();
        state
            .begin(
                0,
                "teacher/2".into(),
                test_hash('a'),
                "student/2".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        state.transition(SleepPhase::KnowledgeSeeding).unwrap();
        state.transition(SleepPhase::Imitation).unwrap();
        state.transition(SleepPhase::RetentionValidation).unwrap();
        state.transition(SleepPhase::Commit).unwrap();
        state
            .record_committed_candidate("candidate/2".into(), test_hash('d'))
            .unwrap();
        state.commit_consolidation().unwrap();
        assert!(!state.tiers[0].slots[1].active);
        assert_eq!(state.tiers[0].slots[1].generation, 1);
        assert!(state.tiers[1].slots[0].active);
        assert_eq!(state.tiers[1].slots[0].generation, 1);
    }

    #[test]
    fn transaction_and_generation_exhaustion_fail_without_partial_mutation() {
        let schedule = schedule();

        let mut exhausted_id = SleepState::new(&schedule, 1).unwrap();
        exhausted_id.cycle = u64::MAX;
        exhausted_id.advance_clock(&schedule, 2).unwrap();
        let before = exhausted_id.clone();
        let error = exhausted_id
            .begin(
                0,
                "teacher/id-overflow".into(),
                test_hash('a'),
                "student/id-overflow".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("transaction ID overflows"), "{error}");
        assert_eq!(exhausted_id, before);

        let mut exhausted_generation = SleepState::new(&schedule, 1).unwrap();
        exhausted_generation.tiers[0].slots[0].active = true;
        exhausted_generation.tiers[0].slots[0].generation = u64::MAX;
        exhausted_generation.advance_clock(&schedule, 2).unwrap();
        exhausted_generation
            .begin(
                0,
                "teacher/generation-overflow".into(),
                test_hash('a'),
                "student/generation-overflow".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        exhausted_generation
            .transition(SleepPhase::KnowledgeSeeding)
            .unwrap();
        exhausted_generation
            .transition(SleepPhase::Imitation)
            .unwrap();
        exhausted_generation
            .transition(SleepPhase::RetentionValidation)
            .unwrap();
        exhausted_generation.transition(SleepPhase::Commit).unwrap();
        exhausted_generation
            .record_committed_candidate("candidate/generation-overflow".into(), test_hash('d'))
            .unwrap();
        let before = exhausted_generation.clone();
        let error = exhausted_generation
            .commit_consolidation()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("sender reserve generation overflows"),
            "{error}"
        );
        assert_eq!(exhausted_generation, before);
    }

    #[derive(Default)]
    struct MockBackend {
        calls: Vec<&'static str>,
        retention: bool,
        fail_seed: bool,
        fail_restore: bool,
        knowledge_samples: u64,
        imitation_samples: u64,
    }

    impl ConsolidationBackend for MockBackend {
        fn knowledge_rng_count(&self) -> Result<u64> {
            Ok(self.knowledge_samples.max(1))
        }

        fn imitation_rng_count(&self) -> Result<u64> {
            Ok(self.imitation_samples.max(1))
        }

        fn compute_prospective_update(&mut self, _: &ConsolidationTxn) -> Result<()> {
            self.calls.push("compute");
            Ok(())
        }
        fn stage_student(&mut self, _: &ConsolidationTxn) -> Result<()> {
            self.calls.push("stage");
            Ok(())
        }
        fn knowledge_seed(&mut self, _: &ConsolidationTxn) -> Result<()> {
            self.calls.push("seed");
            ensure!(!self.fail_seed, "seed failure");
            Ok(())
        }
        fn learn_to_imitate(&mut self, _: &ConsolidationTxn) -> Result<()> {
            self.calls.push("imitate");
            Ok(())
        }
        fn retention_passes(&mut self, _: &ConsolidationTxn) -> Result<bool> {
            self.calls.push("validate");
            Ok(self.retention)
        }
        fn commit(&mut self, _: &ConsolidationTxn) -> Result<CommittedCandidate> {
            self.calls.push("commit");
            Ok(CommittedCandidate {
                checkpoint: "candidate/2".into(),
                sha256: test_hash('d'),
            })
        }
        fn restore_teacher(&mut self, _: &ConsolidationTxn) -> Result<()> {
            self.calls.push("restore");
            ensure!(!self.fail_restore, "restore failure");
            Ok(())
        }
    }

    fn begun_state() -> SleepState {
        let mut state = SleepState::new(&schedule(), 1).unwrap();
        state.advance_clock(&schedule(), 2).unwrap();
        state
            .begin(
                0,
                "teacher/2".into(),
                test_hash('a'),
                "student/2".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        state
    }

    #[test]
    fn rejected_consolidation_restores_teacher_without_slot_mutation() {
        let mut state = begun_state();
        let before = state
            .tiers
            .iter()
            .map(|tier| tier.slots.clone())
            .collect::<Vec<_>>();
        let mut backend = MockBackend::default();
        assert!(!run_consolidation(&mut state, &mut backend).unwrap());
        assert_eq!(backend.calls.last(), Some(&"restore"));
        assert_eq!(state.phase, SleepPhase::Wake);
        assert_eq!(
            state
                .tiers
                .iter()
                .map(|tier| tier.slots.clone())
                .collect::<Vec<_>>(),
            before
        );
        assert!(state.due_senders.is_empty());
    }

    #[test]
    fn generic_consolidation_reserves_every_backend_sample() {
        let mut state = begun_state();
        let mut backend = MockBackend {
            retention: true,
            knowledge_samples: 5,
            imitation_samples: 7,
            ..Default::default()
        };

        assert!(run_consolidation(&mut state, &mut backend).unwrap());
        let txn = state.pending.unwrap();
        assert_eq!(txn.knowledge_rng.unwrap().count, 5);
        assert_eq!(txn.imitation_rng.unwrap().count, 7);
        assert_eq!(state.rng_counters, vec![12]);
    }

    #[test]
    fn rollback_consumes_attempt_id_before_any_later_receipt_can_be_published() {
        let schedule = schedule();
        let mut state = SleepState::new(&schedule, 1).unwrap();
        state.advance_clock(&schedule, 2).unwrap();
        let first = state
            .begin(
                0,
                "teacher/first".into(),
                test_hash('a'),
                "student/first".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        state.rollback().unwrap();
        assert_eq!(state.cycle, first.id);

        state.advance_clock(&schedule, 4).unwrap();
        let second = state
            .begin(
                0,
                "teacher/second".into(),
                test_hash('a'),
                "student/second".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        assert_eq!(second.id, first.id + 1);
    }

    #[test]
    fn rollback_rejects_a_missing_due_clock_without_mutation_or_panic() {
        let mut state = begun_state();
        state.due_clocks.clear();
        let before = state.clone();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.rollback()));
        let error = result
            .expect("rollback panicked on a missing paired due clock")
            .unwrap_err()
            .to_string();

        assert!(error.contains("rolled-back boundary"), "{error}");
        assert_eq!(state, before);
    }

    #[test]
    fn rollback_rejects_a_mismatched_due_clock_without_mutation_or_panic() {
        let mut state = begun_state();
        state.due_clocks[0] += 1;
        let before = state.clone();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.rollback()));
        let error = result
            .expect("rollback panicked on a mismatched paired due clock")
            .unwrap_err()
            .to_string();

        assert!(error.contains("rolled-back boundary"), "{error}");
        assert_eq!(state, before);
    }

    #[test]
    fn terminal_rollback_rejects_an_invalid_trailing_queue_without_mutation() {
        let mut state = begun_state();
        state.pending.as_mut().unwrap().terminal = true;
        state.due_senders.push(usize::MAX);
        state.due_clocks.push(4);
        let before = state.clone();

        let error = state.rollback().unwrap_err().to_string();

        assert!(error.contains("invalid trailing due queue"), "{error}");
        assert_eq!(state, before);
    }

    #[test]
    fn commit_rejects_an_invalid_receiver_slot_without_mutation_or_panic() {
        let mut state = begun_state();
        state.transition(SleepPhase::KnowledgeSeeding).unwrap();
        state.transition(SleepPhase::Imitation).unwrap();
        state.transition(SleepPhase::RetentionValidation).unwrap();
        state.transition(SleepPhase::Commit).unwrap();
        state
            .record_committed_candidate("candidate/invalid-slot".into(), test_hash('d'))
            .unwrap();
        state.pending.as_mut().unwrap().receiver_slot = usize::MAX;
        let before = state.clone();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.commit_consolidation()
        }));
        let error = result
            .expect("commit panicked on an invalid receiver slot")
            .unwrap_err()
            .to_string();

        assert!(error.contains("receiver slot is out of range"), "{error}");
        assert_eq!(state, before);
    }

    #[test]
    fn commit_rejects_an_incomplete_sender_reset_plan_without_mutation() {
        let schedule = schedule();
        let mut state = SleepState::new(&schedule, 1).unwrap();
        state.tiers[0].slots[0].active = true;
        state.advance_clock(&schedule, 2).unwrap();
        state
            .begin(
                0,
                "teacher/incomplete-reset".into(),
                test_hash('a'),
                "student/incomplete-reset".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        state
            .pending
            .as_mut()
            .unwrap()
            .sender_slots_to_reset
            .clear();
        state.transition(SleepPhase::KnowledgeSeeding).unwrap();
        state.transition(SleepPhase::Imitation).unwrap();
        state.transition(SleepPhase::RetentionValidation).unwrap();
        state.transition(SleepPhase::Commit).unwrap();
        state
            .record_committed_candidate("candidate/incomplete-reset".into(), test_hash('d'))
            .unwrap();
        let before = state.clone();

        let error = state.commit_consolidation().unwrap_err().to_string();

        assert!(error.contains("exact active-slot set"), "{error}");
        assert_eq!(state, before);
    }

    #[test]
    fn resume_rejects_completed_receipts_that_can_alias_attempt_ids() {
        let mut finished = committed_state();
        finished.transition(SleepPhase::Candidate).unwrap();
        let completed = finished.finish_candidate().unwrap();
        finished.validate_resume().unwrap();
        assert_eq!(finished.cycle, completed.id);
        assert_eq!(finished.completed_count, 1);

        let mut future_receipt = finished.clone();
        future_receipt.completed_transactions[0].id = future_receipt.cycle + 1;
        let error = future_receipt.validate_resume().unwrap_err().to_string();
        assert!(error.contains("audit tail is invalid"), "{error}");

        let mut dropped_receipt = finished.clone();
        dropped_receipt.completed_transactions.clear();
        let error = dropped_receipt.validate_resume().unwrap_err().to_string();
        assert!(error.contains("audit tail is inconsistent"), "{error}");

        // A committed-but-not-finished transaction owns `cycle` already. Its
        // immutable receipt key must not collide with a completed tail entry.
        let mut duplicate_pending = committed_state();
        duplicate_pending.completed_count = finished.completed_count;
        duplicate_pending.completed_chain_hash = finished.completed_chain_hash.clone();
        duplicate_pending.completed_transactions = finished.completed_transactions.clone();
        let error = duplicate_pending.validate_resume().unwrap_err().to_string();
        assert!(
            error.contains("reuses a completed transaction id"),
            "{error}"
        );
    }

    #[test]
    fn resume_revalidates_the_bounded_completed_transaction_evidence() {
        let mut finished = committed_state();
        finished.transition(SleepPhase::Candidate).unwrap();
        finished.finish_candidate().unwrap();

        let mut forged_hash = finished.clone();
        forged_hash.completed_transactions[0].candidate_hash = Some("sha256:not-a-hash".into());
        let error = forged_hash.validate_resume().unwrap_err().to_string();
        assert!(error.contains("completed candidate hash"), "{error}");

        let mut partial_dream = finished;
        let start = partial_dream.rng_counters[0];
        partial_dream.rng_counters[0] += 1;
        partial_dream.completed_transactions[0].dream_generation_rng = Some(RngReservation {
            stream: 0,
            start,
            count: 1,
        });
        let error = partial_dream.validate_resume().unwrap_err().to_string();
        assert!(error.contains("partial Dreaming evidence"), "{error}");
    }

    #[test]
    fn finish_candidate_overflow_is_failure_atomic() {
        let mut state = committed_state();
        state.transition(SleepPhase::Candidate).unwrap();
        state.completed_count = u64::MAX;
        let before = state.clone();

        let error = state.finish_candidate().unwrap_err().to_string();

        assert!(error.contains("count overflow"), "{error}");
        assert_eq!(state, before);
    }

    #[test]
    fn completed_candidate_prunes_the_manifest_index_with_the_audit_tail() {
        let mut state = committed_state();
        state.transition(SleepPhase::Candidate).unwrap();
        let rng_start = state.rng_counters[0];
        state.rng_counters[0] += 3;
        let pending = state.pending.as_mut().unwrap();
        pending.dream_generation_rng = Some(RngReservation {
            stream: 0,
            start: rng_start,
            count: 1,
        });
        pending.dream_selection_rng = Some(RngReservation {
            stream: 0,
            start: rng_start + 1,
            count: 1,
        });
        pending.dream_trial_rngs = vec![DreamTrialRng {
            candidate_id: "dream".into(),
            reservation: RngReservation {
                stream: 0,
                start: rng_start + 2,
                count: 1,
            },
        }];
        pending.dream_shared_checkpoint_hash = Some(test_hash('8'));
        pending.dream_selected = vec!["dream".into()];
        pending.dream_trials = vec![DreamTrial {
            candidate_id: "dream".into(),
            adapter_hash: test_hash('9'),
            evaluator_hash: test_hash('6'),
            independent_task_improvement: 0.1,
        }];
        pending.dream_policy_receipt = Some(test_hash('5'));
        let template = state.pending.as_ref().unwrap().clone();
        state.completed_transactions = (1..=COMPLETED_TRANSACTION_TAIL)
            .map(|id| {
                let mut txn = template.clone();
                txn.id = id as u64;
                txn.generated_manifest = Some(format!("sha256:{:064x}", id));
                txn
            })
            .collect();
        state.completed_count = COMPLETED_TRANSACTION_TAIL as u64;
        state.cycle = ARTIFACT_MANIFEST_TAIL as u64;
        let pending = state.pending.as_mut().unwrap();
        pending.id = state.cycle;
        pending.generated_manifest = Some(format!("sha256:{:064x}", ARTIFACT_MANIFEST_TAIL));
        state.artifact_manifests =
            referenced_artifact_manifests(&state.completed_transactions, state.pending.as_ref())
                .unwrap();
        assert_eq!(state.artifact_manifests.len(), ARTIFACT_MANIFEST_TAIL);

        state.finish_candidate().unwrap();

        assert_eq!(
            state.completed_transactions.len(),
            COMPLETED_TRANSACTION_TAIL
        );
        assert_eq!(state.artifact_manifests.len(), COMPLETED_TRANSACTION_TAIL);
        assert!(
            !state
                .artifact_manifests
                .contains(&format!("sha256:{:064x}", 1))
        );
        state.validate_resume().unwrap();
    }

    #[test]
    fn failed_consolidation_restores_teacher() {
        let mut state = begun_state();
        let mut backend = MockBackend {
            retention: true,
            fail_seed: true,
            ..Default::default()
        };
        assert!(run_consolidation(&mut state, &mut backend).is_err());
        assert_eq!(backend.calls, vec!["compute", "stage", "seed", "restore"]);
        assert_eq!(state.phase, SleepPhase::Wake);
    }

    #[test]
    fn failed_teacher_restore_never_publishes_rollback_metadata() {
        #[derive(Default)]
        struct RecordingProgress(Vec<SleepState>);

        impl SleepProgressSink for RecordingProgress {
            fn persist(&mut self, state: &SleepState) -> Result<()> {
                self.0.push(state.clone());
                Ok(())
            }
        }

        let mut state = begun_state();
        let mut backend = MockBackend {
            fail_seed: true,
            fail_restore: true,
            ..Default::default()
        };
        let mut progress = RecordingProgress::default();
        let error = run_consolidation_with_progress(&mut state, &mut backend, &mut progress)
            .unwrap_err()
            .to_string();

        assert!(error.contains("teacher restore also failed"), "{error}");
        assert_eq!(state.phase, SleepPhase::KnowledgeSeeding);
        assert!(state.pending.is_some());
        assert!(
            progress
                .0
                .iter()
                .all(|snapshot| snapshot.phase != SleepPhase::Wake && snapshot.pending.is_some()),
            "a rollback cursor was published despite failed teacher restoration"
        );
    }

    #[test]
    fn failed_rollback_persistence_leaves_the_caller_pending() {
        #[derive(Default)]
        struct RejectRollbackProgress(Vec<SleepState>);

        impl SleepProgressSink for RejectRollbackProgress {
            fn persist(&mut self, state: &SleepState) -> Result<()> {
                if state.phase == SleepPhase::Wake {
                    bail!("injected rollback persistence failure");
                }
                self.0.push(state.clone());
                Ok(())
            }
        }

        for mut backend in [
            MockBackend::default(),
            MockBackend {
                fail_seed: true,
                ..Default::default()
            },
        ] {
            let mut state = begun_state();
            let mut progress = RejectRollbackProgress::default();
            let error = format!(
                "{:#}",
                run_consolidation_with_progress(&mut state, &mut backend, &mut progress)
                    .unwrap_err()
            );

            assert!(error.contains("rollback persistence failure"), "{error}");
            assert_ne!(state.phase, SleepPhase::Wake);
            assert!(state.pending.is_some());
            assert!(
                progress
                    .0
                    .iter()
                    .all(|snapshot| snapshot.phase != SleepPhase::Wake),
                "a failed rollback publication escaped into durable progress"
            );
        }
    }

    #[test]
    fn successful_consolidation_preserves_transaction_until_dreaming_finishes() {
        let mut state = begun_state();
        let mut backend = MockBackend {
            retention: true,
            ..Default::default()
        };
        assert!(run_consolidation(&mut state, &mut backend).unwrap());
        assert_eq!(state.phase, SleepPhase::Commit);
        assert!(state.pending.is_some());
        assert!(state.tiers[1].slots[0].active);
    }

    #[test]
    fn consolidation_resumes_from_checkpointed_subphase_and_commit_is_idempotent() {
        let mut state = begun_state();
        state.transition(SleepPhase::KnowledgeSeeding).unwrap();
        let mut backend = MockBackend {
            retention: true,
            ..Default::default()
        };
        assert!(run_consolidation(&mut state, &mut backend).unwrap());
        assert_eq!(backend.calls, vec!["seed", "imitate", "validate", "commit"]);
        assert!(state.pending.as_ref().unwrap().committed);
        let committed_tiers = state.tiers.clone();
        assert!(run_consolidation(&mut state, &mut backend).unwrap());
        assert_eq!(backend.calls, vec!["seed", "imitate", "validate", "commit"]);
        assert_eq!(state.tiers, committed_tiers);
    }

    #[test]
    fn coarse_clock_advance_fails_before_mutating_the_scheduler() {
        let schedule = schedule();
        let mut state = SleepState::new(&schedule, 1).unwrap();
        let before = state.clone();
        let error = state.advance_clock(&schedule, 4).unwrap_err().to_string();
        assert!(
            error.contains("crosses multiple `fast` boundaries"),
            "{error}"
        );
        assert_eq!(state, before);
    }

    #[test]
    fn terminal_rejection_retries_before_the_next_transfer() {
        let schedule = SleepSchedule {
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
        };
        let mut state = SleepState::new(&schedule, 1).unwrap();

        let finish = |state: &mut SleepState, sender: usize, suffix: &str| {
            let txn = state
                .begin(
                    sender,
                    format!("teacher/{suffix}"),
                    test_hash('a'),
                    format!("student/{suffix}"),
                    test_hash('b'),
                    test_hash('c'),
                )
                .unwrap();
            state.transition(SleepPhase::KnowledgeSeeding).unwrap();
            state.reserve_knowledge_rng(0, 1).unwrap();
            state.transition(SleepPhase::Imitation).unwrap();
            state.reserve_imitation_rng(0, 1).unwrap();
            state.transition(SleepPhase::RetentionValidation).unwrap();
            state.transition(SleepPhase::Commit).unwrap();
            state
                .record_committed_candidate(format!("candidate/{suffix}"), test_hash('d'))
                .unwrap();
            state.commit_consolidation().unwrap();
            state.transition(SleepPhase::Candidate).unwrap();
            state.finish_candidate().unwrap();
            txn
        };

        state.advance_clock(&schedule, 1).unwrap();
        finish(&mut state, 0, "fast-1");
        state.advance_clock(&schedule, 2).unwrap();
        finish(&mut state, 0, "fast-2");
        assert!(state.tiers[1].slots.iter().all(|slot| slot.active));

        state
            .begin(
                1,
                "teacher/terminal-2".into(),
                test_hash('a'),
                "student/terminal-2".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        let rejected_id = state.pending.as_ref().unwrap().id;
        state.rollback().unwrap();
        assert_eq!(state.cycle, rejected_id);

        state.advance_clock(&schedule, 3).unwrap();
        assert!(state.due_senders.is_empty());
        assert_eq!(state.tiers[0].last_boundary_clock, 3);
        let statuses = state
            .tiers
            .iter()
            .enumerate()
            .flat_map(|(tier, tier_state)| {
                tier_state
                    .slots
                    .iter()
                    .enumerate()
                    .map(move |(slot, slot_state)| MemorySlotStatus {
                        layer: 0,
                        tier,
                        tier_name: tier_state.id.clone(),
                        slot,
                        active: slot_state.active,
                        generation: slot_state.generation,
                        parameter_ids: vec![ParamId::new()],
                    })
            })
            .collect::<Vec<_>>();
        validate_model_memory_state(&schedule, &state, &statuses).unwrap();

        state.advance_clock(&schedule, 4).unwrap();
        assert_eq!(state.due_senders, vec![1, 0]);
        validate_model_memory_state(&schedule, &state, &statuses).unwrap();
        let retry = state
            .begin(
                1,
                "teacher/terminal-retry-4".into(),
                test_hash('a'),
                "student/terminal-retry-4".into(),
                test_hash('b'),
                test_hash('c'),
            )
            .unwrap();
        assert!(retry.id > rejected_id);
        state.rollback().unwrap();
        assert!(state.due_senders.is_empty());

        // A repeated rejection backpressures the coincident fast boundary.
        state.advance_clock(&schedule, 5).unwrap();
        assert!(state.due_senders.is_empty());
        state.advance_clock(&schedule, 6).unwrap();
        assert_eq!(state.due_senders, vec![1, 0]);
        let accepted_retry = finish(&mut state, 1, "terminal-retry-6");
        assert!(accepted_retry.id > retry.id);
        assert!(state.tiers[1].slots.iter().all(|slot| !slot.active));

        // The coincident fast transfer now has reclaimed receiver capacity.
        let transfer = finish(&mut state, 0, "fast-6");
        assert!(!transfer.terminal);
        assert_eq!(transfer.receiver_slot, 0);
        assert!(state.tiers[1].slots[0].active);
    }

    #[test]
    fn imitation_rewards_have_expected_direction() {
        let exact = normalized_levenshtein_reward(&[1, 2], &[1, 2], 2);
        let near = normalized_levenshtein_reward(&[1, 3], &[1, 2], 2);
        let rejected = normalized_levenshtein_reward(&[3, 4], &[1, 2], 1);
        assert_eq!(exact, 1.0);
        assert!(near < exact && near > rejected);
        assert!(
            imitation_reward(true, near, 0.7).unwrap()
                > imitation_reward(false, near, 0.7).unwrap()
        );
    }

    #[test]
    fn gradient_alignment_and_dream_selection_are_deterministic() {
        assert!((gradient_cosine(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(gradient_cosine(&[0.0, 0.0], &[1.0, 0.0]).unwrap(), 0.0);
        let selected = select_dreams(
            &[
                DreamCandidateScore {
                    id: "a".into(),
                    importance: 3.0,
                    diversity_key: 9,
                },
                DreamCandidateScore {
                    id: "b".into(),
                    importance: 2.0,
                    diversity_key: 1,
                },
                DreamCandidateScore {
                    id: "c".into(),
                    importance: 1.0,
                    diversity_key: 0,
                },
            ],
            1,
            1,
        )
        .unwrap();
        assert_eq!(selected, vec!["a", "c"]);
    }

    #[test]
    fn dreaming_requires_a_nonempty_unambiguous_selection() {
        let mut config = dreaming_config();
        config.retain_top = 0;
        config.retain_random = 0;
        assert!(config.validate().is_err());

        let mut unsupported = dreaming_config();
        unsupported.selector_version = "future-cosine-v2".into();
        let error = unsupported.validate().unwrap_err().to_string();
        assert!(error.contains("gradient-cosine-v1"), "{error}");

        let duplicated = [
            DreamCandidateScore {
                id: "same".into(),
                importance: 1.0,
                diversity_key: 1,
            },
            DreamCandidateScore {
                id: "same".into(),
                importance: 0.5,
                diversity_key: 2,
            },
        ];
        assert!(select_dreams(&duplicated, 1, 1).is_err());
        assert!(select_dreams(&[], usize::MAX, 1).is_err());

        let mut overflow = dreaming_config();
        overflow.retain_top = usize::MAX;
        overflow.retain_random = 1;
        assert!(overflow.validate().is_err());

        let mut rollout_overflow = KnowledgeSeedingConfig {
            chunk_tokens: 1,
            teacher_rollouts: usize::MAX,
            detached_student_rollouts: 1,
            temperature: 1.0,
            forward_kl_weight: 1.0,
        };
        assert!(rollout_overflow.validate().is_err());
        rollout_overflow.detached_student_rollouts = 0;
        assert!(rollout_overflow.validate().is_err());
    }

    #[test]
    fn generated_dream_validation_rejects_unbounded_or_nonfinite_gradients() {
        let candidate = |gradient| GeneratedDream {
            id: "candidate".into(),
            artifact_hash: test_hash('d'),
            gradient,
            diversity_key: 0,
        };

        assert!(validate_generated_dreams(&[candidate(vec![f32::NAN])], 1).is_err());
        assert!(
            validate_generated_dreams(
                &[candidate(vec![0.0; MAX_DREAM_GRADIENT_DIMENSIONS + 1])],
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn sleep_configuration_accepts_boundaries_and_rejects_unbounded_work() {
        let knowledge_boundary = KnowledgeSeedingConfig {
            chunk_tokens: KnowledgeSeedingConfig::MAX_CHUNK_TOKENS,
            teacher_rollouts: KnowledgeSeedingConfig::MAX_ROLLOUTS - 1,
            detached_student_rollouts: 1,
            temperature: 1.0,
            forward_kl_weight: 1.0,
        };
        knowledge_boundary.validate().unwrap();

        let mut invalid_knowledge = knowledge_boundary.clone();
        invalid_knowledge.chunk_tokens = KnowledgeSeedingConfig::MAX_CHUNK_TOKENS + 1;
        let error = invalid_knowledge.validate().unwrap_err();
        assert!(format!("{error:#}").contains("chunk_tokens"), "{error:#}");
        invalid_knowledge = knowledge_boundary.clone();
        invalid_knowledge.teacher_rollouts = KnowledgeSeedingConfig::MAX_ROLLOUTS;
        let error = invalid_knowledge.validate().unwrap_err();
        assert!(format!("{error:#}").contains("rollout count"), "{error:#}");

        let imitation_boundary = ImitationConfig {
            semantic_judge_hash: test_hash('a'),
            semantic_weight: 0.5,
            maximum_edit_distance: ImitationConfig::MAX_EDIT_DISTANCE,
            grpo_group_size: ImitationConfig::MAX_GRPO_GROUP_SIZE,
        };
        imitation_boundary.validate().unwrap();

        let mut invalid_imitation = imitation_boundary.clone();
        invalid_imitation.maximum_edit_distance = ImitationConfig::MAX_EDIT_DISTANCE + 1;
        let error = invalid_imitation.validate().unwrap_err();
        assert!(format!("{error:#}").contains("edit distance"), "{error:#}");
        invalid_imitation = imitation_boundary.clone();
        invalid_imitation.grpo_group_size = ImitationConfig::MAX_GRPO_GROUP_SIZE + 1;
        let error = invalid_imitation.validate().unwrap_err();
        assert!(format!("{error:#}").contains("group size"), "{error:#}");

        let dreaming_boundary = DreamingConfig {
            candidate_count: DreamingConfig::MAX_CANDIDATES,
            retain_top: DreamingConfig::MAX_CANDIDATES,
            retain_random: 0,
            lora_rank: DreamingConfig::MAX_LORA_RANK,
            lora_alpha: DreamingConfig::MAX_LORA_ALPHA,
            restem_iterations: DreamingConfig::MAX_RESTEM_ITERATIONS,
            selector_version: "gradient-cosine-v1".into(),
            reference_set_hash: test_hash('b'),
            trial_evaluator_hash: test_hash('c'),
        };
        dreaming_boundary.validate().unwrap();

        for (mut invalid, expected) in [
            (
                DreamingConfig {
                    candidate_count: DreamingConfig::MAX_CANDIDATES + 1,
                    ..dreaming_boundary.clone()
                },
                "candidate count",
            ),
            (
                DreamingConfig {
                    lora_rank: DreamingConfig::MAX_LORA_RANK + 1,
                    ..dreaming_boundary.clone()
                },
                "LoRA rank",
            ),
            (
                DreamingConfig {
                    lora_alpha: DreamingConfig::MAX_LORA_ALPHA + 1,
                    ..dreaming_boundary.clone()
                },
                "LoRA alpha",
            ),
            (
                DreamingConfig {
                    restem_iterations: DreamingConfig::MAX_RESTEM_ITERATIONS + 1,
                    ..dreaming_boundary.clone()
                },
                "ReSTEM iterations",
            ),
        ] {
            // Keep quotas structurally valid when testing the candidate cap so
            // the operational ceiling is the first rejected condition.
            invalid.retain_top = invalid.retain_top.min(invalid.candidate_count);
            let error = invalid.validate().unwrap_err();
            assert!(format!("{error:#}").contains(expected), "{error:#}");
        }
    }

    #[test]
    fn grpo_advantages_are_centered() {
        let values = group_relative_advantages(&[1.0, 2.0, 3.0], 1e-6).unwrap();
        assert!(values.iter().sum::<f32>().abs() < 1e-6);
        assert!(values[2] > values[1] && values[1] > values[0]);
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum DreamFailure {
        Generation,
        Load,
        Reference,
        Trial,
        Policy,
    }

    #[derive(Default)]
    struct MockDreamBackend {
        committed_hash: String,
        shared_hash: String,
        generation_calls: usize,
        load_calls: usize,
        trial_calls: Vec<String>,
        restem_calls: usize,
        restores: usize,
        all_rejected: bool,
        fail_once: Option<DreamFailure>,
        invalid_trial: bool,
    }

    impl MockDreamBackend {
        fn new() -> Self {
            Self {
                committed_hash: test_hash('d'),
                shared_hash: test_hash('d'),
                ..Self::default()
            }
        }

        fn all_rejected() -> Self {
            Self {
                all_rejected: true,
                ..Self::new()
            }
        }

        fn failing(operation: DreamFailure) -> Self {
            Self {
                fail_once: Some(operation),
                ..Self::new()
            }
        }

        fn fail_if_requested(&mut self, operation: DreamFailure) -> Result<()> {
            if self.fail_once == Some(operation) {
                self.fail_once = None;
                self.shared_hash = test_hash('e');
                bail!("injected {operation:?} failure after shared-candidate mutation");
            }
            Ok(())
        }

        fn dreams() -> Vec<GeneratedDream> {
            vec![
                GeneratedDream {
                    id: "a".into(),
                    artifact_hash: test_hash('1'),
                    gradient: vec![1.0, 0.0],
                    diversity_key: 9,
                },
                GeneratedDream {
                    id: "b".into(),
                    artifact_hash: test_hash('2'),
                    gradient: vec![0.8, 0.2],
                    diversity_key: 1,
                },
                GeneratedDream {
                    id: "c".into(),
                    artifact_hash: test_hash('3'),
                    gradient: vec![0.0, 1.0],
                    diversity_key: 0,
                },
            ]
        }

        fn trial(id: &str) -> DreamTrial {
            DreamTrial {
                candidate_id: id.into(),
                adapter_hash: test_hash(if id == "a" { '4' } else { '5' }),
                evaluator_hash: test_hash('6'),
                independent_task_improvement: if id == "a" { 0.25 } else { -0.1 },
            }
        }
    }

    impl DreamingBackend for MockDreamBackend {
        fn verify_committed_candidate(&mut self, txn: &ConsolidationTxn) -> Result<()> {
            ensure!(
                txn.candidate_hash.as_ref() == Some(&self.committed_hash),
                "dream backend is bound to another committed checkpoint"
            );
            ensure!(
                self.shared_hash == test_hash('d'),
                "dream backend shared parameters differ from its immutable candidate"
            );
            Ok(())
        }

        fn shared_checkpoint_hash(&mut self) -> Result<String> {
            Ok(self.shared_hash.clone())
        }

        fn generate_from_wake_contexts(
            &mut self,
            _: &ConsolidationTxn,
            candidate_count: usize,
            random_extra_expert: bool,
        ) -> Result<(String, Vec<GeneratedDream>)> {
            ensure!(random_extra_expert, "dream exploration was not enabled");
            ensure!(candidate_count == 3, "unexpected candidate count");
            self.fail_if_requested(DreamFailure::Generation)?;
            self.generation_calls += 1;
            Ok((test_hash('7'), Self::dreams()))
        }

        fn load_generated_dreams(
            &mut self,
            _: &ConsolidationTxn,
            manifest: &str,
        ) -> Result<Vec<GeneratedDream>> {
            ensure!(manifest == test_hash('7'), "wrong dream manifest");
            self.fail_if_requested(DreamFailure::Load)?;
            self.load_calls += 1;
            Ok(Self::dreams())
        }

        fn reference_gradient(
            &mut self,
            _: &ConsolidationTxn,
            reference_set_hash: &str,
        ) -> Result<Vec<f32>> {
            ensure!(reference_set_hash == test_hash('8'), "wrong reference set");
            self.fail_if_requested(DreamFailure::Reference)?;
            Ok(vec![1.0, 0.0])
        }

        fn isolated_lora_trial(
            &mut self,
            _: &ConsolidationTxn,
            candidate: &GeneratedDream,
            rank: usize,
            alpha: usize,
        ) -> Result<DreamTrial> {
            ensure!(rank == 64 && alpha == 128, "wrong LoRA recipe");
            self.fail_if_requested(DreamFailure::Trial)?;
            self.trial_calls.push(candidate.id.clone());
            let mut trial = Self::trial(&candidate.id);
            if self.all_rejected {
                trial.independent_task_improvement = -0.1;
            }
            if self.invalid_trial {
                trial.candidate_id = "wrong-candidate".into();
            }
            Ok(trial)
        }

        fn restem_update(
            &mut self,
            _: &ConsolidationTxn,
            accepted: &[DreamTrial],
            iterations: usize,
        ) -> Result<String> {
            ensure!(iterations == 1, "wrong ReSTEM iterations");
            self.fail_if_requested(DreamFailure::Policy)?;
            if self.all_rejected {
                ensure!(accepted.is_empty(), "all-rejected run accepted a dream");
            } else {
                ensure!(
                    accepted.len() == 1 && accepted[0].candidate_id == "a",
                    "wrong accepted dreams"
                );
            }
            self.restem_calls += 1;
            Ok(test_hash('9'))
        }

        fn restore_shared_candidate(&mut self, _: &ConsolidationTxn) -> Result<()> {
            self.shared_hash = test_hash('d');
            self.restores += 1;
            Ok(())
        }
    }

    fn committed_state() -> SleepState {
        let mut state = begun_state();
        let mut consolidation = MockBackend {
            retention: true,
            ..Default::default()
        };
        assert!(run_consolidation(&mut state, &mut consolidation).unwrap());
        state
    }

    fn dreaming_config() -> DreamingConfig {
        DreamingConfig {
            candidate_count: 3,
            retain_top: 1,
            retain_random: 1,
            lora_rank: 64,
            lora_alpha: 128,
            restem_iterations: 1,
            selector_version: "gradient-cosine-v1".into(),
            reference_set_hash: test_hash('8'),
            trial_evaluator_hash: test_hash('6'),
        }
    }

    #[test]
    fn dreaming_is_resumable_from_every_durable_subphase() {
        for resume_phase in [
            SleepPhase::Commit,
            SleepPhase::DreamGeneration,
            SleepPhase::DreamRanking,
            SleepPhase::DreamTrials,
            SleepPhase::DreamPolicyUpdate,
            SleepPhase::Candidate,
        ] {
            let mut state = committed_state();
            if resume_phase != SleepPhase::Commit {
                state.pending.as_mut().unwrap().dream_shared_checkpoint_hash = Some(test_hash('d'));
                state.transition(SleepPhase::DreamGeneration).unwrap();
            }
            if matches!(
                resume_phase,
                SleepPhase::DreamRanking
                    | SleepPhase::DreamTrials
                    | SleepPhase::DreamPolicyUpdate
                    | SleepPhase::Candidate
            ) {
                state.reserve_dream_generation_rng(0, 3).unwrap();
                state.record_generated_manifest(test_hash('7')).unwrap();
                state.transition(SleepPhase::DreamRanking).unwrap();
            }
            if matches!(
                resume_phase,
                SleepPhase::DreamTrials | SleepPhase::DreamPolicyUpdate | SleepPhase::Candidate
            ) {
                state.reserve_dream_selection_rng(0).unwrap();
                state.pending.as_mut().unwrap().dream_selected = vec!["a".into(), "c".into()];
                state.transition(SleepPhase::DreamTrials).unwrap();
            }
            if matches!(
                resume_phase,
                SleepPhase::DreamPolicyUpdate | SleepPhase::Candidate
            ) {
                state.reserve_dream_trial_rng(0, "a").unwrap();
                state.reserve_dream_trial_rng(0, "c").unwrap();
                state.pending.as_mut().unwrap().dream_trials =
                    vec![MockDreamBackend::trial("a"), MockDreamBackend::trial("c")];
                state.transition(SleepPhase::DreamPolicyUpdate).unwrap();
            }
            if resume_phase == SleepPhase::Candidate {
                state.pending.as_mut().unwrap().dream_policy_receipt = Some(test_hash('9'));
                state.transition(SleepPhase::Candidate).unwrap();
            }
            state.validate_resume().unwrap();
            let mut backend = MockDreamBackend::new();
            let accepted = run_dreaming(&mut state, &dreaming_config(), &mut backend).unwrap();
            assert_eq!(state.phase, SleepPhase::Candidate);
            assert_eq!(accepted.len(), 1);
            assert_eq!(accepted[0].candidate_id, "a");
        }
    }

    #[test]
    fn dreaming_repairs_but_never_adopts_a_wrong_shared_candidate() {
        let mut wrongly_bound_state = committed_state();
        let wrongly_bound_before = wrongly_bound_state.clone();
        let mut wrongly_bound = MockDreamBackend::new();
        wrongly_bound.committed_hash = test_hash('e');
        let error = run_dreaming(
            &mut wrongly_bound_state,
            &dreaming_config(),
            &mut wrongly_bound,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("did not recover the committed candidate"),
            "{error}"
        );
        assert_eq!(wrongly_bound_state, wrongly_bound_before);

        let mut state = committed_state();
        let before = state.clone();
        let mut backend = MockDreamBackend::new();
        backend.shared_hash = test_hash('e');

        let error = run_dreaming(&mut state, &dreaming_config(), &mut backend)
            .unwrap_err()
            .to_string();
        assert!(error.contains("committed candidate"), "{error}");
        assert_eq!(state, before);
        assert_eq!(backend.shared_hash, test_hash('d'));
        assert_eq!(backend.restores, 1);

        run_dreaming(&mut state, &dreaming_config(), &mut backend).unwrap();
        assert_eq!(state.phase, SleepPhase::Candidate);

        backend.shared_hash = test_hash('e');
        let before = state.clone();
        let error = run_dreaming(&mut state, &dreaming_config(), &mut backend)
            .unwrap_err()
            .to_string();
        assert!(error.contains("committed candidate"), "{error}");
        assert_eq!(state, before);
        assert_eq!(backend.shared_hash, test_hash('d'));
    }

    #[test]
    fn dreaming_finishes_when_every_isolated_trial_is_rejected() {
        let mut state = committed_state();
        let mut backend = MockDreamBackend::all_rejected();
        let accepted = run_dreaming(&mut state, &dreaming_config(), &mut backend).unwrap();
        assert!(accepted.is_empty());
        assert_eq!(state.phase, SleepPhase::Candidate);
        assert_eq!(backend.restem_calls, 1);
        assert!(
            state
                .pending
                .as_ref()
                .unwrap()
                .dream_policy_receipt
                .is_some()
        );
    }

    #[test]
    fn dreaming_backend_errors_always_restore_the_shared_candidate_before_retry() {
        for operation in [
            DreamFailure::Generation,
            DreamFailure::Load,
            DreamFailure::Reference,
            DreamFailure::Trial,
            DreamFailure::Policy,
        ] {
            let mut state = committed_state();
            let mut backend = MockDreamBackend::failing(operation);
            let error = run_dreaming(&mut state, &dreaming_config(), &mut backend)
                .unwrap_err()
                .to_string();
            assert!(error.contains("shared"), "{operation:?}: {error}");
            assert_eq!(backend.shared_hash, test_hash('d'), "{operation:?}");
            assert_eq!(backend.restores, 1, "{operation:?}");
            assert!(state.pending.is_some(), "{operation:?}");

            let accepted = run_dreaming(&mut state, &dreaming_config(), &mut backend).unwrap();
            assert_eq!(accepted.len(), 1, "{operation:?}");
            assert_eq!(state.phase, SleepPhase::Candidate, "{operation:?}");
        }
    }

    #[test]
    fn invalid_dream_evidence_fails_without_changing_the_shared_candidate() {
        let mut state = committed_state();
        let mut backend = MockDreamBackend {
            invalid_trial: true,
            ..MockDreamBackend::new()
        };
        let error = run_dreaming(&mut state, &dreaming_config(), &mut backend)
            .unwrap_err()
            .to_string();
        assert!(error.contains("incomplete or invalid evidence"), "{error}");
        assert_eq!(backend.shared_hash, test_hash('d'));
        assert_eq!(backend.restores, 0);
        assert_eq!(state.phase, SleepPhase::DreamTrials);

        backend.invalid_trial = false;
        let accepted = run_dreaming(&mut state, &dreaming_config(), &mut backend).unwrap();
        assert_eq!(accepted.len(), 1);
    }

    #[test]
    fn dreaming_emits_valid_selection_and_isolated_trial_metrics() {
        #[derive(Default)]
        struct Sink(Vec<MetricEvent>);
        impl SleepProgressSink for Sink {
            fn persist(&mut self, _: &SleepState) -> Result<()> {
                Ok(())
            }

            fn metric(&mut self, event: MetricEvent) -> Result<()> {
                event.validate()?;
                self.0.push(event);
                Ok(())
            }
        }

        let mut state = committed_state();
        let mut backend = MockDreamBackend::new();
        let mut sink = Sink::default();
        run_dreaming_with_progress(&mut state, &dreaming_config(), &mut backend, &mut sink)
            .unwrap();
        assert_eq!(
            sink.0
                .iter()
                .filter(|event| matches!(event, MetricEvent::DreamSelection(_)))
                .count(),
            1
        );
        assert_eq!(
            sink.0
                .iter()
                .filter(|event| matches!(event, MetricEvent::DreamTrial(_)))
                .count(),
            2
        );
    }

    #[test]
    fn failed_selection_metric_leaves_a_resumable_ranking_cursor() {
        struct RejectSelectionMetric;

        impl SleepProgressSink for RejectSelectionMetric {
            fn persist(&mut self, _: &SleepState) -> Result<()> {
                Ok(())
            }

            fn metric(&mut self, event: MetricEvent) -> Result<()> {
                if matches!(event, MetricEvent::DreamSelection(_)) {
                    bail!("injected dream-selection metric failure");
                }
                Ok(())
            }
        }

        let mut state = committed_state();
        let mut backend = MockDreamBackend::new();
        let error = run_dreaming_with_progress(
            &mut state,
            &dreaming_config(),
            &mut backend,
            &mut RejectSelectionMetric,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("metric failure"), "{error}");
        assert_eq!(state.phase, SleepPhase::DreamRanking);
        assert!(state.pending.as_ref().unwrap().dream_selected.is_empty());
        state.validate_resume().unwrap();

        let accepted = run_dreaming(&mut state, &dreaming_config(), &mut backend).unwrap();
        assert_eq!(accepted.len(), 1);
        assert_eq!(state.phase, SleepPhase::Candidate);
    }
}
