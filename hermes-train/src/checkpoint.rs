//! Atomic resumable training checkpoints.
//!
//! Parameter IDs are persisted alongside model and optimizer state because
//! Burn optimizers key their state by those IDs. A marker prevents resuming a
//! checkpoint whose multi-file publication was interrupted.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn::module::{AutodiffModule, Module, ModuleMapper, Param, ParamId};
use burn::tensor::{Device, Tensor};
use burn_optim::ModuleOptimizer;
use hermes_llm::{Transformer, load_safetensors, save_safetensors};
use serde::{Deserialize, Serialize};

use crate::muon::BatchedMuon;

pub(crate) type AdamWOptimizer = ModuleOptimizer;

pub(crate) const TRAINING_STATE_VERSION: u32 = 1;

fn current_training_state_version() -> u32 {
    TRAINING_STATE_VERSION
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct TrainingState {
    #[serde(default = "current_training_state_version")]
    pub(crate) version: u32,
    pub(crate) step: usize,
    pub(crate) stage: usize,
    pub(crate) epoch: usize,
    pub(crate) samples_in_stage: usize,
    #[serde(default)]
    pub(crate) steps_in_stage: usize,
    #[serde(default)]
    pub(crate) tokens_seen: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) curriculum_signature: Option<String>,
    pub(crate) parameter_ids: Vec<u64>,
}

pub(crate) fn parameter_ids(model: &Transformer) -> Vec<u64> {
    burn::module::list_param_ids(model)
        .into_iter()
        .map(|id| id.val())
        .collect()
}

struct ParameterIdMapper<'a> {
    ids: std::slice::Iter<'a, u64>,
}

impl ModuleMapper for ParameterIdMapper<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (_, tensor, mapper) = param.consume();
        let id = self
            .ids
            .next()
            .copied()
            .expect("checkpoint contains too few parameter IDs");
        Param::from_mapped_value(ParamId::from(id), tensor, mapper)
    }
}

fn restore_parameter_ids(model: &mut Transformer, ids: &[u64]) -> Result<()> {
    ensure!(
        ids.len() == burn::module::list_param_ids(model).len(),
        "checkpoint has {} parameter IDs, model has {}",
        ids.len(),
        burn::module::list_param_ids(model).len()
    );
    let mut mapper = ParameterIdMapper { ids: ids.iter() };
    *model = model.clone().map(&mut mapper);
    ensure!(
        mapper.ids.next().is_none(),
        "checkpoint contains too many parameter IDs"
    );
    Ok(())
}

pub(crate) fn save_training_checkpoint(
    model: &Transformer,
    adamw: &AdamWOptimizer,
    muon: &BatchedMuon,
    state: &TrainingState,
    output: &Path,
) -> Result<()> {
    let marker = output.join(".checkpoint-in-progress");
    let weights_temporary = output.join("weights.safetensors.tmp");
    let adamw_temporary = output.join("adamw-state.bpk.tmp");
    let muon_temporary = output.join("muon-state.bpk.tmp");
    let state_temporary = output.join("training-state.json.tmp");

    fs::write(&marker, state.step.to_string())?;
    save_safetensors(&model.clone().valid(), &weights_temporary)?;
    adamw
        .save(&adamw_temporary)
        .context("failed to save AdamW state")?;
    muon.save(&muon_temporary)?;
    fs::write(&state_temporary, serde_json::to_vec_pretty(&state)?)?;
    fs::rename(weights_temporary, output.join("weights.safetensors"))?;
    fs::rename(adamw_temporary, output.join("adamw-state.bpk"))?;
    fs::rename(muon_temporary, output.join("muon-state.bpk"))?;
    fs::rename(state_temporary, output.join("training-state.json"))?;
    fs::remove_file(marker)?;
    Ok(())
}

pub(crate) fn load_training_state(
    model: &mut Transformer,
    adamw: AdamWOptimizer,
    muon: &mut BatchedMuon,
    output: &Path,
    device: &Device,
) -> Result<(AdamWOptimizer, TrainingState)> {
    ensure!(
        !output.join(".checkpoint-in-progress").exists(),
        "checkpoint was interrupted while being saved"
    );
    let state: TrainingState =
        serde_json::from_slice(&fs::read(output.join("training-state.json"))?)?;
    ensure!(
        state.version == TRAINING_STATE_VERSION,
        "unsupported training-state version {}; this build supports version {TRAINING_STATE_VERSION}",
        state.version
    );
    restore_parameter_ids(model, &state.parameter_ids)?;
    load_safetensors(model, output.join("weights.safetensors"))?;
    muon.set_parameter_ids(model.muon_parameter_ids());
    muon.load(output.join("muon-state.bpk"), &device.clone().inner())?;
    let adamw = adamw
        .load(output.join("adamw-state.bpk"))
        .context("failed to load AdamW state")?;
    Ok((adamw, state))
}
