//! Versioned training curriculum configuration.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

pub(crate) const CURRICULUM_VERSION: u32 = 1;

fn default_temperature() -> f64 {
    0.05
}

fn default_summary_instruction() -> String {
    "Summarize the document faithfully and concisely.".to_owned()
}

fn default_planning_instruction() -> String {
    "Create a concise retrieval plan as an ordered action trace.".to_owned()
}

fn default_query_prefix() -> String {
    "Represent this query for retrieval:\n".to_owned()
}

fn default_document_prefix() -> String {
    "Represent this document for retrieval:\n".to_owned()
}

fn default_one_f64() -> f64 {
    1.0
}

fn default_one_usize() -> usize {
    1
}

fn default_shuffle_buffer() -> usize {
    8192
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ObjectiveConfig {
    CausalLm,
    Summarization {
        #[serde(default = "default_summary_instruction")]
        instruction: String,
    },
    RetrievalPlanning {
        #[serde(default = "default_planning_instruction")]
        instruction: String,
    },
    ContrastiveRetrieval {
        #[serde(default = "default_temperature")]
        temperature: f64,
        /// One-based Transformer layer; omitted means the final layer.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        layer: Option<usize>,
        #[serde(default = "default_query_prefix")]
        query_prefix: String,
        #[serde(default = "default_document_prefix")]
        document_prefix: String,
    },
}

impl ObjectiveConfig {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::CausalLm => "causal_lm",
            Self::Summarization { .. } => "summarization",
            Self::RetrievalPlanning { .. } => "retrieval_planning",
            Self::ContrastiveRetrieval { .. } => "contrastive_retrieval",
        }
    }

    pub(crate) fn retrieval_layer(&self) -> Option<usize> {
        match self {
            Self::ContrastiveRetrieval { layer, .. } => *layer,
            _ => None,
        }
    }

    pub(crate) fn temperature(&self) -> Option<f64> {
        match self {
            Self::ContrastiveRetrieval { temperature, .. } => Some(*temperature),
            _ => None,
        }
    }

    fn validate(&self, stage_name: &str) -> Result<()> {
        match self {
            Self::CausalLm => {}
            Self::Summarization { instruction } | Self::RetrievalPlanning { instruction } => {
                ensure!(
                    !instruction.trim().is_empty(),
                    "curriculum stage `{stage_name}` objective instruction must not be empty"
                )
            }
            Self::ContrastiveRetrieval {
                temperature,
                layer,
                query_prefix,
                document_prefix,
            } => {
                ensure!(
                    temperature.is_finite() && *temperature > 0.0,
                    "curriculum stage `{stage_name}` retrieval temperature must be finite and positive"
                );
                ensure!(
                    layer.is_none_or(|layer| layer > 0),
                    "curriculum stage `{stage_name}` retrieval layer is one-based and must be positive"
                );
                ensure!(
                    !query_prefix.is_empty() && !document_prefix.is_empty(),
                    "curriculum stage `{stage_name}` retrieval prefixes must not be empty"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurriculumFile {
    version: u32,
    stages: Vec<StageFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageFile {
    name: String,
    data: PathBuf,
    objective: ObjectiveConfig,
    sequence_length: usize,
    batch_size: usize,
    gradient_accumulation: usize,
    #[serde(default = "default_one_usize")]
    epochs: usize,
    #[serde(default = "default_shuffle_buffer")]
    shuffle_buffer: usize,
    steps: Option<usize>,
    #[serde(default = "default_one_f64")]
    loss_weight: f64,
    #[serde(default = "default_one_f64")]
    learning_rate_scale: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResolvedStage {
    pub(crate) name: String,
    pub(crate) data: PathBuf,
    pub(crate) objective: ObjectiveConfig,
    pub(crate) sequence_length: usize,
    pub(crate) batch_size: usize,
    pub(crate) gradient_accumulation: usize,
    pub(crate) epochs: usize,
    pub(crate) shuffle_buffer: usize,
    pub(crate) steps: Option<usize>,
    pub(crate) loss_weight: f64,
    pub(crate) learning_rate_scale: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ResolvedCurriculum {
    pub(crate) version: u32,
    pub(crate) stages: Vec<ResolvedStage>,
}

fn validate_stage(stage: &ResolvedStage) -> Result<()> {
    let name = &stage.name;
    ensure!(
        !name.trim().is_empty(),
        "curriculum stage name must not be empty"
    );
    ensure!(
        stage.sequence_length > 0,
        "curriculum stage `{name}` sequence_length must be positive"
    );
    ensure!(
        stage.batch_size > 0,
        "curriculum stage `{name}` batch_size must be positive"
    );
    ensure!(
        stage.gradient_accumulation > 0,
        "curriculum stage `{name}` gradient_accumulation must be positive"
    );
    ensure!(
        stage.epochs > 0,
        "curriculum stage `{name}` epochs must be positive"
    );
    ensure!(
        stage.steps.is_none_or(|steps| steps > 0),
        "curriculum stage `{name}` steps must be positive when set"
    );
    ensure!(
        stage.loss_weight.is_finite() && stage.loss_weight > 0.0,
        "curriculum stage `{name}` loss_weight must be finite and positive"
    );
    ensure!(
        stage.learning_rate_scale.is_finite() && stage.learning_rate_scale > 0.0,
        "curriculum stage `{name}` learning_rate_scale must be finite and positive"
    );
    stage.objective.validate(name)
}

pub(crate) fn load_curriculum(path: &Path) -> Result<ResolvedCurriculum> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read curriculum {}", path.display()))?;
    let file: CurriculumFile = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid curriculum JSON in {}", path.display()))?;
    ensure!(
        file.version == CURRICULUM_VERSION,
        "unsupported curriculum version {}; this build supports version {CURRICULUM_VERSION}",
        file.version
    );
    ensure!(!file.stages.is_empty(), "curriculum contains no stages");

    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut names = BTreeSet::new();
    let mut stages = Vec::with_capacity(file.stages.len());
    for stage in file.stages {
        ensure!(
            names.insert(stage.name.clone()),
            "duplicate curriculum stage name `{}`",
            stage.name
        );
        let data = if stage.data.is_absolute() {
            stage.data
        } else {
            base.join(stage.data)
        };
        let stage = ResolvedStage {
            name: stage.name,
            data,
            objective: stage.objective,
            sequence_length: stage.sequence_length,
            batch_size: stage.batch_size,
            gradient_accumulation: stage.gradient_accumulation,
            epochs: stage.epochs,
            shuffle_buffer: stage.shuffle_buffer,
            steps: stage.steps,
            loss_weight: stage.loss_weight,
            learning_rate_scale: stage.learning_rate_scale,
        };
        validate_stage(&stage)?;
        stages.push(stage);
    }

    Ok(ResolvedCurriculum {
        version: CURRICULUM_VERSION,
        stages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curriculum_resolves_paths_and_applies_documented_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("curriculum.json");
        fs::write(
            &path,
            r#"{
                "version": 1,
                "stages": [{
                    "name": "summaries",
                    "data": "data/summaries.jsonl",
                    "objective": {"type": "summarization"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "steps": 10,
                    "learning_rate_scale": 0.25
                }]
            }"#,
        )
        .unwrap();

        let curriculum = load_curriculum(&path).unwrap();
        let stage = &curriculum.stages[0];
        assert_eq!(stage.data, dir.path().join("data/summaries.jsonl"));
        assert_eq!(stage.sequence_length, 512);
        assert_eq!(stage.batch_size, 8);
        assert_eq!(stage.gradient_accumulation, 2);
        assert_eq!(stage.epochs, 1);
        assert_eq!(stage.shuffle_buffer, 8192);
        assert_eq!(stage.steps, Some(10));
        assert_eq!(stage.learning_rate_scale, 0.25);
        assert_eq!(stage.objective.name(), "summarization");
    }

    #[test]
    fn unsupported_curriculum_version_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("curriculum.json");
        fs::write(&path, r#"{"version":2,"stages":[]}"#).unwrap();
        let error = load_curriculum(&path).unwrap_err().to_string();
        assert!(
            error.contains("unsupported curriculum version 2"),
            "{error}"
        );
    }

    #[test]
    fn unknown_stage_fields_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("curriculum.json");
        fs::write(
            &path,
            r#"{
                "version": 1,
                "stages": [{
                    "name": "retrieval",
                    "data": "pairs.jsonl",
                    "objective": {"type": "contrastive_retrieval"},
                    "sequence_length": 512,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "sequnce_length": 512
                }]
            }"#,
        )
        .unwrap();
        let error = format!("{:#}", load_curriculum(&path).unwrap_err());
        assert!(error.contains("sequnce_length"), "{error}");
    }
}
