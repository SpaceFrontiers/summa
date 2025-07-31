//! Strict WorkflowV2 projection for the current wake-training loop.
//!
//! Full lifecycle orchestration consumes [`hermes_train::workflow`] directly.
//! This module keeps the existing trainer executable while making unsupported
//! phase/task execution fail loudly instead of dropping workflow phases.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};
use hermes_train::quantization::WorkflowQuantizationPlan;
use hermes_train::task::{TaskAdapter, TaskConfig};
use hermes_train::workflow::{
    InModelSleepConfig, MemoryUpdateMode, PhaseKind, ResolvedWorkflow as WorkflowV2,
    load_workflow as load_workflow_v2,
};
use serde::Serialize;

fn project_task(phase_name: &str, task: &TaskConfig) -> Result<TaskConfig> {
    match task {
        TaskConfig::CausalLm {}
        | TaskConfig::Summarization { .. }
        | TaskConfig::RetrievalPlanning { .. }
        | TaskConfig::InstructionTuning { .. }
        | TaskConfig::QaReasoning { .. }
        | TaskConfig::RetrievalRepresentation { .. } => Ok(task.clone()),
        task => bail!(
            "workflow phase `{phase_name}` task `{}` is not executable by the wake trainer; dispatch it through its WorkflowV2 task executor",
            task.name()
        ),
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResolvedWakePhase {
    pub(crate) name: String,
    pub(crate) phase_kind: PhaseKind,
    pub(crate) data: PathBuf,
    /// Exact WorkflowV2 task contract consumed by data construction and loss
    /// dispatch. Keeping this type intact avoids a second wake-only schema.
    pub(crate) objective: TaskConfig,
    pub(crate) sequence_length: usize,
    pub(crate) batch_size: usize,
    pub(crate) gradient_accumulation: usize,
    pub(crate) epochs: usize,
    pub(crate) shuffle_buffer: usize,
    pub(crate) steps: Option<usize>,
    pub(crate) loss_weight: f64,
    pub(crate) learning_rate_scale: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) quantization: Option<WorkflowQuantizationPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) periodic_sleep: Option<InModelSleepConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) memory_update_mode: Option<MemoryUpdateMode>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResolvedWakePlan {
    pub(crate) version: u32,
    pub(crate) phases: Vec<ResolvedWakePhase>,
}

fn project_wake_workflow(workflow: WorkflowV2) -> Result<ResolvedWakePlan> {
    let mut phases = Vec::with_capacity(workflow.phases.len());
    for phase in workflow.phases {
        if !matches!(
            phase.kind,
            PhaseKind::Pretrain
                | PhaseKind::ContinuedPretrain
                | PhaseKind::Sft
                | PhaseKind::Quantization
        ) {
            bail!(
                "workflow phase `{}` ({}) requires a WorkflowV2 phase executor and cannot be run by the wake-only trainer",
                phase.name,
                phase.kind.name()
            );
        }
        ensure!(
            phase.parameters.is_empty(),
            "workflow phase `{}` contains raw parameters that the wake trainer does not consume; use typed WorkflowV2 fields or a registered phase executor",
            phase.name
        );
        let task = phase
            .task
            .as_ref()
            .expect("validated task-data phase has a task");
        let objective = project_task(&phase.name, task)?;
        let epochs = phase.epochs_or_default();
        let shuffle_buffer = phase.shuffle_buffer_or_default();
        let loss_weight = phase.loss_weight_or_default();
        let learning_rate_scale = phase.learning_rate_scale_or_default();
        let quantization = phase
            .quantization
            .as_ref()
            .map(WorkflowQuantizationPlan::from_workflow)
            .transpose()?;
        phases.push(ResolvedWakePhase {
            name: phase.name,
            phase_kind: phase.kind,
            data: phase
                .data
                .expect("validated task-data phase has a data path"),
            objective,
            sequence_length: phase
                .sequence_length
                .expect("validated task-data phase has sequence_length"),
            batch_size: phase
                .batch_size
                .expect("validated task-data phase has batch_size"),
            gradient_accumulation: phase
                .gradient_accumulation
                .expect("validated optimization phase has gradient_accumulation"),
            epochs,
            shuffle_buffer,
            steps: phase.steps,
            loss_weight,
            learning_rate_scale,
            quantization,
            periodic_sleep: phase.periodic_sleep,
            memory_update_mode: phase.memory_update_mode,
        });
    }
    Ok(ResolvedWakePlan {
        version: workflow.version,
        phases,
    })
}

pub(crate) fn load_wake_plan(path: &Path) -> Result<ResolvedWakePlan> {
    project_wake_workflow(load_workflow_v2(path)?)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn workflow_v2_resolves_paths_and_projects_supported_wake_phases() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "summaries",
                    "type": "sft",
                    "data": "data/summaries.jsonl",
                    "task": {"type": "summarization"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "steps": 10,
                    "learning_rate_scale": 0.25
                }]
            }"#,
        )
        .unwrap();

        let workflow = load_wake_plan(&path).unwrap();
        let phase = &workflow.phases[0];
        assert_eq!(workflow.version, 2);
        assert_eq!(phase.phase_kind, PhaseKind::Sft);
        assert_eq!(phase.data, dir.path().join("data/summaries.jsonl"));
        assert_eq!(phase.sequence_length, 512);
        assert_eq!(phase.batch_size, 8);
        assert_eq!(phase.gradient_accumulation, 2);
        assert_eq!(phase.epochs, 1);
        assert_eq!(phase.shuffle_buffer, 8192);
        assert_eq!(phase.steps, Some(10));
        assert_eq!(phase.learning_rate_scale, 0.25);
        assert_eq!(phase.objective.name(), "summarization");
    }

    #[test]
    fn workflow_version_one_has_no_compatibility_parser() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(&path, r#"{"version":1,"phases":[]}"#).unwrap();
        let error = load_wake_plan(&path).unwrap_err().to_string();
        assert!(error.contains("unsupported workflow version 1"), "{error}");
    }

    #[test]
    fn unsupported_phase_is_never_silently_projected_away() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "sleep",
                    "type": "sleep"
                }]
            }"#,
        )
        .unwrap();
        let error = load_wake_plan(&path).unwrap_err().to_string();
        assert!(error.contains("sleep"), "{error}");
        assert!(
            error.contains("cannot be run") || error.contains("requires typed sleep settings"),
            "{error}"
        );
    }

    #[test]
    fn unsupported_task_is_never_coerced_to_a_different_loss() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "ranking",
                    "type": "continued_pretrain",
                    "data": "ranking.jsonl",
                    "task": {"type": "retrieval_ranking"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "steps": 10
                }]
            }"#,
        )
        .unwrap();
        let error = load_wake_plan(&path).unwrap_err().to_string();
        assert!(error.contains("retrieval_ranking"), "{error}");
        assert!(error.contains("not executable"), "{error}");
    }

    #[test]
    fn unknown_phase_fields_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "retrieval",
                    "type": "continued_pretrain",
                    "data": "pairs.jsonl",
                    "task": {"type": "retrieval_representation"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "sequnce_length": 512
                }]
            }"#,
        )
        .unwrap();
        let error = format!("{:#}", load_wake_plan(&path).unwrap_err());
        assert!(error.contains("sequnce_length"), "{error}");
    }

    #[test]
    fn wake_projection_rejects_raw_parameters_instead_of_dropping_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.json");
        fs::write(
            &path,
            r#"{
                "version": 2,
                "phases": [{
                    "name": "custom-causal",
                    "type": "pretrain",
                    "data": "tokens.jsonl",
                    "task": {"type": "causal_lm"},
                    "sequence_length": 128,
                    "batch_size": 4,
                    "gradient_accumulation": 1,
                    "steps": 1,
                    "parameters": {"unconsumed_learning_rule": "v1"}
                }]
            }"#,
        )
        .unwrap();

        let error = load_wake_plan(&path).unwrap_err().to_string();
        assert!(error.contains("raw parameters"), "{error}");
        assert!(error.contains("does not consume"), "{error}");
    }
}
