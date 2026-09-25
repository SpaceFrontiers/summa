//! Algorithm-neutral task contracts and record schemas.
//!
//! A task describes the inputs and optimization signal a training algorithm
//! consumes. It deliberately does not choose an optimizer, preference
//! algorithm, RL algorithm, or model implementation; those choices belong to
//! workflow phases and their executors.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

fn default_temperature() -> f64 {
    0.05
}

fn default_summary_instruction() -> String {
    "Summarize the document faithfully and concisely.".to_owned()
}

fn default_planning_instruction() -> String {
    "Create a concise retrieval plan as an ordered action trace.".to_owned()
}

fn default_instruction_tuning_instruction() -> String {
    "Follow the instruction and provide the requested response.".to_owned()
}

fn default_reasoning_instruction() -> String {
    "Solve the problem carefully and provide the final answer.".to_owned()
}

fn default_query_prefix() -> String {
    "Represent this query for retrieval:\n".to_owned()
}

fn default_document_prefix() -> String {
    "Represent this document for retrieval:\n".to_owned()
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Includes the positive document. Wake retrieval materializes every document
/// as one fixed-length token sequence before shuffling, so this also provides
/// a finite geometry contract to WorkflowV2.
const MAX_RETRIEVAL_REPRESENTATION_DOCUMENTS: usize = 32;
/// Ranking is dispatched to a dedicated executor and can use larger lists,
/// but a single bounded JSONL row must still not create an unbounded example.
const MAX_RETRIEVAL_RANKING_DOCUMENTS: usize = 1_024;

/// The model-facing operation required by a task.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskExecution {
    AutoregressiveTokenPrediction,
    SupervisedGeneration,
    ContrastiveRepresentation,
    ListwiseRanking,
    PairwisePreference,
    VerifiableReward,
}

/// On-disk record framing accepted by a task package.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskDataFormat {
    TextOrJsonl,
    Jsonl,
}

/// Declarative masking policy. Tensor construction remains backend-owned.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LossMaskPolicy {
    AllNextTokens,
    TargetTokensOnly,
    NotApplicable,
}

/// Standard metric outputs requested from an executor by a task package.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskMetric {
    Loss,
    Perplexity,
    ExactMatch,
    RetrievalTop1,
    Ndcg,
    PreferenceAccuracy,
    MeanReward,
    PassRate,
}

/// Stable, inspectable capabilities used by phase executors for dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TaskContract {
    pub execution: TaskExecution,
    pub data_format: TaskDataFormat,
    pub required_fields: &'static [&'static str],
    pub input_fields: &'static [&'static str],
    pub target_fields: &'static [&'static str],
    pub loss_mask: LossMaskPolicy,
    pub metrics: &'static [TaskMetric],
}

/// Text whose tokenization boundaries are part of the task contract.
///
/// Executors must tokenize each segment independently and concatenate the
/// resulting token IDs. Tokenizing [`Self::render`] as one string is not
/// equivalent for boundary-sensitive tokenizers such as BPE. At most one
/// segment may be truncated; fixed prompt text must remain intact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentedText {
    /// Ordered strings whose independently encoded token IDs are concatenated.
    pub segments: Vec<String>,
    /// The only segment an executor may shorten to meet its sequence limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncatable_segment: Option<usize>,
    /// Whether truncation must retain at least one token from that segment.
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncatable_segment_required: bool,
}

impl SegmentedText {
    fn prefixed(prefix: &str, text: String) -> Self {
        Self {
            segments: vec![prefix.to_owned(), text],
            truncatable_segment: Some(1),
            truncatable_segment_required: true,
        }
    }

    /// Validate structural invariants before an externally supplied example is
    /// dispatched to a model executor.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.segments.is_empty(),
            "segmented text must not be empty"
        );
        if let Some(index) = self.truncatable_segment {
            ensure!(
                index < self.segments.len(),
                "truncatable segment {index} is outside {} segments",
                self.segments.len()
            );
            if self.truncatable_segment_required {
                ensure!(
                    !self.segments[index].is_empty(),
                    "required truncatable segment {index} must not be empty"
                );
            }
        } else {
            ensure!(
                !self.truncatable_segment_required,
                "a required truncatable segment needs an index"
            );
        }
        Ok(())
    }

    /// Rendered text for inspection, logging, or APIs that consume text. Model
    /// execution must preserve the segment boundaries documented above.
    pub fn render(&self) -> String {
        let capacity = self
            .segments
            .iter()
            .fold(0usize, |total, segment| total.saturating_add(segment.len()));
        let mut rendered = String::with_capacity(capacity);
        for segment in &self.segments {
            rendered.push_str(segment);
        }
        rendered
    }
}

/// Backend-neutral example produced from one validated task record. Executors
/// tokenize these inputs and apply the accompanying [`LossSpec`] instead of
/// hard-coding storage-field names in trainer core.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskExample {
    Autoregressive {
        text: String,
    },
    SupervisedGeneration {
        prompt: SegmentedText,
        target: String,
    },
    RetrievalRepresentation {
        query: SegmentedText,
        documents: Vec<SegmentedText>,
        positive_index: usize,
    },
    RetrievalRanking {
        query: SegmentedText,
        documents: Vec<SegmentedText>,
        relevance: Vec<f64>,
    },
    PairwisePreference {
        prompt: String,
        chosen: String,
        rejected: String,
    },
    VerifiableRollout {
        prompt: String,
        verifier_payload: serde_json::Value,
        reference_answer: Option<String>,
    },
}

/// Optimization signal requested by a task. Algorithm variants (DPO vs IPO,
/// GRPO vs PPO, forward vs reverse distillation) remain phase parameters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LossSpec {
    TokenCrossEntropy { mask: LossMaskPolicy },
    ContrastiveRepresentation { temperature: f64 },
    ListwiseRanking { temperature: f64 },
    PairwisePreference,
    PolicyGradient,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RewardSpec {
    Verifier { verifier: VerifierSpec },
}

/// Common interface implemented by all built-in and future task packages.
pub trait TaskAdapter {
    fn name(&self) -> &'static str;
    fn contract(&self) -> TaskContract;
    fn validate(&self) -> Result<()>;
    fn validate_record(&self, record: &serde_json::Value) -> Result<()>;
    fn construct_example(&self, record: &serde_json::Value) -> Result<TaskExample>;
    fn loss_spec(&self) -> LossSpec;
    fn reward_spec(&self) -> Option<RewardSpec>;
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskConfig {
    CausalLm {},
    Summarization {
        #[serde(default = "default_summary_instruction")]
        instruction: String,
    },
    RetrievalRepresentation {
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
    RetrievalRanking {
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
    RetrievalPlanning {
        #[serde(default = "default_planning_instruction")]
        instruction: String,
    },
    InstructionTuning {
        #[serde(default = "default_instruction_tuning_instruction")]
        instruction: String,
    },
    QaReasoning {
        #[serde(default = "default_reasoning_instruction")]
        instruction: String,
        #[serde(default)]
        require_reasoning: bool,
    },
    PairwisePreference {},
    VerifiableRl {
        verifier: VerifierSpec,
    },
}

/// Canonical rendering of a supervised-generation record.
///
/// The source is kept separate from the fixed prompt text so an executor can
/// truncate long documents, questions, or optional inputs without truncating
/// the instruction, response marker, or target. All built-in executors must
/// consume these segments instead of rebuilding task prompts independently.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupervisedPromptSegments<'a> {
    pub prefix: Cow<'a, str>,
    pub source: Cow<'a, str>,
    pub suffix: Cow<'a, str>,
    pub target: Cow<'a, str>,
    pub source_required: bool,
}

impl SupervisedPromptSegments<'_> {
    pub fn prompt(&self) -> String {
        let mut prompt = String::with_capacity(
            self.prefix
                .len()
                .saturating_add(self.source.len())
                .saturating_add(self.suffix.len()),
        );
        prompt.push_str(&self.prefix);
        prompt.push_str(&self.source);
        prompt.push_str(&self.suffix);
        prompt
    }

    fn into_example(self) -> TaskExample {
        let prompt = SegmentedText {
            segments: vec![
                self.prefix.into_owned(),
                self.source.into_owned(),
                self.suffix.into_owned(),
            ],
            truncatable_segment: Some(1),
            truncatable_segment_required: self.source_required,
        };
        debug_assert!(prompt.validate().is_ok());
        TaskExample::SupervisedGeneration {
            prompt,
            target: self.target.into_owned(),
        }
    }
}

impl TaskConfig {
    /// Conservative upper bound on fixed-length model sequences retained for
    /// one example by the built-in wake data path. Workflow validation uses it
    /// for batch and shuffle-memory geometry instead of assuming one sequence
    /// for a contrastive retrieval example.
    pub fn maximum_model_sequences_per_example(&self) -> usize {
        match self {
            Self::RetrievalRepresentation { .. } => 1 + MAX_RETRIEVAL_REPRESENTATION_DOCUMENTS,
            _ => 1,
        }
    }

    pub fn retrieval_layer(&self) -> Option<usize> {
        match self {
            Self::RetrievalRepresentation { layer, .. } | Self::RetrievalRanking { layer, .. } => {
                *layer
            }
            _ => None,
        }
    }

    pub fn temperature(&self) -> Option<f64> {
        match self {
            Self::RetrievalRepresentation { temperature, .. }
            | Self::RetrievalRanking { temperature, .. } => Some(*temperature),
            _ => None,
        }
    }

    /// Parse and render one record for a built-in supervised-generation task.
    /// The returned segmentation is the single prompt contract shared by wake
    /// training and post-training executors.
    pub fn construct_supervised_prompt<'a>(
        &self,
        record: &'a serde_json::Value,
    ) -> Result<SupervisedPromptSegments<'a>> {
        self.validate()?;
        self.construct_supervised_prompt_after_validation(record)
    }

    fn construct_supervised_prompt_after_validation<'a>(
        &self,
        record: &'a serde_json::Value,
    ) -> Result<SupervisedPromptSegments<'a>> {
        match self {
            Self::Summarization { instruction } => {
                let record = parse_record::<SummarizationRecord>(record)?;
                record.validate()?;
                Ok(SupervisedPromptSegments {
                    prefix: format!("{instruction}\n\nDocument:\n").into(),
                    source: record.document,
                    suffix: "\n\nSummary:\n".into(),
                    target: record.summary,
                    source_required: true,
                })
            }
            Self::RetrievalPlanning { instruction } => {
                let record = parse_record::<RetrievalPlanningRecord>(record)?;
                record.validate()?;
                let (prefix, source, suffix, source_required) = match record.context {
                    Some(context) => (
                        Cow::Owned(format!(
                            "{instruction}\n\nRequest:\n{}\n\nContext:\n",
                            record.request
                        )),
                        context,
                        Cow::Borrowed("\nPlan:\n"),
                        true,
                    ),
                    None => (
                        Cow::Owned(format!("{instruction}\n\nRequest:\n{}\n", record.request)),
                        Cow::Borrowed(""),
                        Cow::Borrowed("Plan:\n"),
                        false,
                    ),
                };
                Ok(SupervisedPromptSegments {
                    prefix,
                    source,
                    suffix,
                    target: record.plan,
                    source_required,
                })
            }
            Self::InstructionTuning { instruction } => {
                let record = parse_record::<InstructionTuningRecord>(record)?;
                record.validate()?;
                let mut prefix = String::new();
                if let Some(system) = record.system {
                    prefix.push_str("System:\n");
                    prefix.push_str(&system);
                    prefix.push_str("\n\n");
                }
                prefix.push_str(instruction);
                prefix.push_str("\n\nInstruction:\n");
                prefix.push_str(&record.instruction);
                let (source, source_required) = match record.input {
                    Some(input) => {
                        prefix.push_str("\n\nInput:\n");
                        (input, true)
                    }
                    None => (Cow::Borrowed(""), false),
                };
                Ok(SupervisedPromptSegments {
                    prefix: prefix.into(),
                    source,
                    suffix: "\n\nResponse:\n".into(),
                    target: record.response,
                    source_required,
                })
            }
            Self::QaReasoning {
                instruction,
                require_reasoning,
            } => {
                let record = parse_record::<QaReasoningRecord>(record)?;
                record.validate(*require_reasoning)?;
                let target = match record.reasoning {
                    Some(reasoning) => Cow::Owned(format!(
                        "Reasoning:\n{reasoning}\n\nAnswer:\n{}",
                        record.answer
                    )),
                    None => record.answer,
                };
                Ok(SupervisedPromptSegments {
                    prefix: format!("{instruction}\n\nQuestion:\n").into(),
                    source: record.question,
                    suffix: "\n\nResponse:\n".into(),
                    target,
                    source_required: true,
                })
            }
            _ => bail!("task `{}` is not a supervised-generation task", self.name()),
        }
    }
}

impl TaskAdapter for TaskConfig {
    fn name(&self) -> &'static str {
        match self {
            Self::CausalLm {} => "causal_lm",
            Self::Summarization { .. } => "summarization",
            Self::RetrievalRepresentation { .. } => "retrieval_representation",
            Self::RetrievalRanking { .. } => "retrieval_ranking",
            Self::RetrievalPlanning { .. } => "retrieval_planning",
            Self::InstructionTuning { .. } => "instruction_tuning",
            Self::QaReasoning { .. } => "qa_reasoning",
            Self::PairwisePreference {} => "pairwise_preference",
            Self::VerifiableRl { .. } => "verifiable_rl",
        }
    }

    fn contract(&self) -> TaskContract {
        match self {
            Self::CausalLm {} => TaskContract {
                execution: TaskExecution::AutoregressiveTokenPrediction,
                data_format: TaskDataFormat::TextOrJsonl,
                required_fields: &["text"],
                input_fields: &["text"],
                target_fields: &["text"],
                loss_mask: LossMaskPolicy::AllNextTokens,
                metrics: &[TaskMetric::Loss, TaskMetric::Perplexity],
            },
            Self::Summarization { .. } => TaskContract {
                execution: TaskExecution::SupervisedGeneration,
                data_format: TaskDataFormat::Jsonl,
                required_fields: &["document", "summary"],
                input_fields: &["document"],
                target_fields: &["summary"],
                loss_mask: LossMaskPolicy::TargetTokensOnly,
                metrics: &[TaskMetric::Loss],
            },
            Self::RetrievalRepresentation { .. } => TaskContract {
                execution: TaskExecution::ContrastiveRepresentation,
                data_format: TaskDataFormat::Jsonl,
                required_fields: &["query", "positive"],
                input_fields: &["query", "positive", "negatives"],
                target_fields: &["positive"],
                loss_mask: LossMaskPolicy::NotApplicable,
                metrics: &[TaskMetric::Loss, TaskMetric::RetrievalTop1],
            },
            Self::RetrievalRanking { .. } => TaskContract {
                execution: TaskExecution::ListwiseRanking,
                data_format: TaskDataFormat::Jsonl,
                required_fields: &["query", "documents"],
                input_fields: &["query", "documents.document"],
                target_fields: &["documents.relevance"],
                loss_mask: LossMaskPolicy::NotApplicable,
                metrics: &[TaskMetric::Loss, TaskMetric::Ndcg],
            },
            Self::RetrievalPlanning { .. } => TaskContract {
                execution: TaskExecution::SupervisedGeneration,
                data_format: TaskDataFormat::Jsonl,
                required_fields: &["request", "plan"],
                input_fields: &["request", "context"],
                target_fields: &["plan"],
                loss_mask: LossMaskPolicy::TargetTokensOnly,
                metrics: &[TaskMetric::Loss, TaskMetric::ExactMatch],
            },
            Self::InstructionTuning { .. } => TaskContract {
                execution: TaskExecution::SupervisedGeneration,
                data_format: TaskDataFormat::Jsonl,
                required_fields: &["instruction", "response"],
                input_fields: &["system", "instruction", "input"],
                target_fields: &["response"],
                loss_mask: LossMaskPolicy::TargetTokensOnly,
                metrics: &[TaskMetric::Loss],
            },
            Self::QaReasoning {
                require_reasoning, ..
            } => TaskContract {
                execution: TaskExecution::SupervisedGeneration,
                data_format: TaskDataFormat::Jsonl,
                required_fields: if *require_reasoning {
                    &["question", "answer", "reasoning"]
                } else {
                    &["question", "answer"]
                },
                input_fields: &["question"],
                target_fields: &["reasoning", "answer"],
                loss_mask: LossMaskPolicy::TargetTokensOnly,
                metrics: &[TaskMetric::Loss, TaskMetric::ExactMatch],
            },
            Self::PairwisePreference {} => TaskContract {
                execution: TaskExecution::PairwisePreference,
                data_format: TaskDataFormat::Jsonl,
                required_fields: &["prompt", "chosen", "rejected"],
                input_fields: &["prompt", "chosen", "rejected"],
                target_fields: &["chosen", "rejected"],
                loss_mask: LossMaskPolicy::NotApplicable,
                metrics: &[TaskMetric::Loss, TaskMetric::PreferenceAccuracy],
            },
            Self::VerifiableRl { .. } => TaskContract {
                execution: TaskExecution::VerifiableReward,
                data_format: TaskDataFormat::Jsonl,
                required_fields: &["prompt", "verifier_payload"],
                input_fields: &["prompt", "verifier_payload", "reference_answer"],
                target_fields: &[],
                loss_mask: LossMaskPolicy::NotApplicable,
                metrics: &[TaskMetric::MeanReward, TaskMetric::PassRate],
            },
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::CausalLm {} | Self::PairwisePreference {} => {}
            Self::Summarization { instruction }
            | Self::RetrievalPlanning { instruction }
            | Self::InstructionTuning { instruction }
            | Self::QaReasoning { instruction, .. } => {
                ensure!(
                    !instruction.trim().is_empty(),
                    "{} instruction must not be empty",
                    self.name()
                );
            }
            Self::RetrievalRepresentation {
                temperature,
                layer,
                query_prefix,
                document_prefix,
            }
            | Self::RetrievalRanking {
                temperature,
                layer,
                query_prefix,
                document_prefix,
            } => {
                ensure!(
                    temperature.is_finite() && *temperature > 0.0,
                    "{} temperature must be finite and positive",
                    self.name()
                );
                ensure!(
                    layer.is_none_or(|layer| layer > 0),
                    "{} layer is one-based and must be positive",
                    self.name()
                );
                ensure!(
                    !query_prefix.is_empty() && !document_prefix.is_empty(),
                    "{} query and document prefixes must not be empty",
                    self.name()
                );
            }
            Self::VerifiableRl { verifier } => verifier.validate()?,
        }
        Ok(())
    }

    fn validate_record(&self, record: &serde_json::Value) -> Result<()> {
        self.construct_example(record).map(|_| ())
    }

    fn construct_example(&self, record: &serde_json::Value) -> Result<TaskExample> {
        self.validate()?;
        let example = (|| match self {
            Self::CausalLm {} => {
                let record = parse_record::<CausalLmRecord>(record)?;
                record.validate()?;
                Ok(TaskExample::Autoregressive { text: record.text })
            }
            Self::Summarization { .. }
            | Self::RetrievalPlanning { .. }
            | Self::InstructionTuning { .. }
            | Self::QaReasoning { .. } => Ok(self
                .construct_supervised_prompt_after_validation(record)?
                .into_example()),
            Self::RetrievalRepresentation {
                query_prefix,
                document_prefix,
                ..
            } => {
                let record = parse_record::<RetrievalRepresentationRecord>(record)?;
                record.validate()?;
                let mut documents = Vec::with_capacity(record.negatives.len() + 1);
                documents.push(SegmentedText::prefixed(document_prefix, record.positive));
                documents.extend(
                    record
                        .negatives
                        .into_iter()
                        .map(|document| SegmentedText::prefixed(document_prefix, document)),
                );
                Ok(TaskExample::RetrievalRepresentation {
                    query: SegmentedText::prefixed(query_prefix, record.query),
                    documents,
                    positive_index: 0,
                })
            }
            Self::RetrievalRanking {
                query_prefix,
                document_prefix,
                ..
            } => {
                let record = parse_record::<RetrievalRankingRecord>(record)?;
                record.validate()?;
                let mut documents = Vec::with_capacity(record.documents.len());
                let mut relevance = Vec::with_capacity(record.documents.len());
                for document in record.documents {
                    documents.push(SegmentedText::prefixed(document_prefix, document.document));
                    relevance.push(document.relevance);
                }
                Ok(TaskExample::RetrievalRanking {
                    query: SegmentedText::prefixed(query_prefix, record.query),
                    documents,
                    relevance,
                })
            }
            Self::PairwisePreference {} => {
                let record = parse_record::<PairwisePreferenceRecord>(record)?;
                record.validate()?;
                Ok(TaskExample::PairwisePreference {
                    prompt: record.prompt,
                    chosen: record.chosen,
                    rejected: record.rejected,
                })
            }
            Self::VerifiableRl { .. } => {
                let record = parse_record::<VerifiableRlRecord>(record)?;
                record.validate()?;
                Ok(TaskExample::VerifiableRollout {
                    prompt: record.prompt,
                    verifier_payload: record.verifier_payload,
                    reference_answer: record.reference_answer,
                })
            }
        })();
        example.map_err(|error: anyhow::Error| {
            anyhow::anyhow!("invalid {} task record: {error:#}", self.name())
        })
    }

    fn loss_spec(&self) -> LossSpec {
        match self {
            Self::CausalLm {} => LossSpec::TokenCrossEntropy {
                mask: LossMaskPolicy::AllNextTokens,
            },
            Self::Summarization { .. }
            | Self::RetrievalPlanning { .. }
            | Self::InstructionTuning { .. }
            | Self::QaReasoning { .. } => LossSpec::TokenCrossEntropy {
                mask: LossMaskPolicy::TargetTokensOnly,
            },
            Self::RetrievalRepresentation { temperature, .. } => {
                LossSpec::ContrastiveRepresentation {
                    temperature: *temperature,
                }
            }
            Self::RetrievalRanking { temperature, .. } => LossSpec::ListwiseRanking {
                temperature: *temperature,
            },
            Self::PairwisePreference {} => LossSpec::PairwisePreference,
            Self::VerifiableRl { .. } => LossSpec::PolicyGradient,
        }
    }

    fn reward_spec(&self) -> Option<RewardSpec> {
        match self {
            Self::VerifiableRl { verifier } => Some(RewardSpec::Verifier {
                verifier: verifier.clone(),
            }),
            _ => None,
        }
    }
}

fn parse_record<'de, T: Deserialize<'de>>(value: &'de serde_json::Value) -> Result<T> {
    T::deserialize(value).context("record does not match the task schema")
}

fn ensure_nonempty(value: &str, field: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "`{field}` must not be empty");
    Ok(())
}

fn ensure_optional_nonempty(value: Option<&str>, field: &str) -> Result<()> {
    if let Some(value) = value {
        ensure_nonempty(value, field)?;
    }
    Ok(())
}

/// A named verifier resolved by the RL executor, with task-owned parameters.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierSpec {
    pub adapter: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, serde_json::Value>,
}

impl VerifierSpec {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.adapter, "verifier.adapter")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CausalLmRecord {
    pub text: String,
}

impl CausalLmRecord {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.text, "text")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SummarizationRecord<'a> {
    #[serde(borrow)]
    pub document: Cow<'a, str>,
    #[serde(borrow)]
    pub summary: Cow<'a, str>,
}

impl SummarizationRecord<'_> {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.document, "document")?;
        ensure_nonempty(&self.summary, "summary")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RetrievalRepresentationRecord {
    pub query: String,
    pub positive: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub negatives: Vec<String>,
}

impl RetrievalRepresentationRecord {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.query, "query")?;
        ensure_nonempty(&self.positive, "positive")?;
        let mut documents = BTreeSet::from([self.positive.as_str()]);
        for (index, negative) in self.negatives.iter().enumerate() {
            ensure_nonempty(negative, &format!("negatives[{index}]"))?;
            ensure!(
                documents.insert(negative),
                "`negatives[{index}]` duplicates the positive or another negative document"
            );
        }
        ensure!(
            self.negatives.len() < MAX_RETRIEVAL_REPRESENTATION_DOCUMENTS,
            "retrieval representation has {} documents; at most {MAX_RETRIEVAL_REPRESENTATION_DOCUMENTS} including the positive are supported",
            self.negatives.len().saturating_add(1)
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RankedDocument {
    pub document: String,
    pub relevance: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RetrievalRankingRecord {
    pub query: String,
    pub documents: Vec<RankedDocument>,
}

impl RetrievalRankingRecord {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.query, "query")?;
        ensure!(
            self.documents.len() >= 2,
            "`documents` must contain at least two candidates"
        );
        ensure!(
            self.documents.len() <= MAX_RETRIEVAL_RANKING_DOCUMENTS,
            "`documents` exceeds the {MAX_RETRIEVAL_RANKING_DOCUMENTS}-candidate limit"
        );
        let mut first_relevance = None;
        let mut distinct_relevance = false;
        let mut documents = BTreeSet::new();
        for (index, document) in self.documents.iter().enumerate() {
            ensure_nonempty(&document.document, &format!("documents[{index}].document"))?;
            ensure!(
                documents.insert(document.document.as_str()),
                "`documents[{index}].document` duplicates another ranking candidate"
            );
            ensure!(
                document.relevance.is_finite() && document.relevance >= 0.0,
                "`documents[{index}].relevance` must be finite and non-negative"
            );
            match first_relevance {
                Some(first) => distinct_relevance |= document.relevance != first,
                None => first_relevance = Some(document.relevance),
            }
        }
        ensure!(
            distinct_relevance,
            "`documents` must contain at least two distinct relevance grades"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RetrievalPlanningRecord<'a> {
    #[serde(borrow)]
    pub request: Cow<'a, str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(borrow)]
    pub context: Option<Cow<'a, str>>,
    #[serde(borrow)]
    pub plan: Cow<'a, str>,
}

impl RetrievalPlanningRecord<'_> {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.request, "request")?;
        ensure_optional_nonempty(self.context.as_deref(), "context")?;
        ensure_nonempty(&self.plan, "plan")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InstructionTuningRecord<'a> {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(borrow)]
    pub system: Option<Cow<'a, str>>,
    #[serde(borrow)]
    pub instruction: Cow<'a, str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(borrow)]
    pub input: Option<Cow<'a, str>>,
    #[serde(borrow)]
    pub response: Cow<'a, str>,
}

impl InstructionTuningRecord<'_> {
    fn validate(&self) -> Result<()> {
        ensure_optional_nonempty(self.system.as_deref(), "system")?;
        ensure_nonempty(&self.instruction, "instruction")?;
        ensure_optional_nonempty(self.input.as_deref(), "input")?;
        ensure_nonempty(&self.response, "response")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QaReasoningRecord<'a> {
    #[serde(borrow)]
    pub question: Cow<'a, str>,
    #[serde(borrow)]
    pub answer: Cow<'a, str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(borrow)]
    pub reasoning: Option<Cow<'a, str>>,
}

impl QaReasoningRecord<'_> {
    fn validate(&self, require_reasoning: bool) -> Result<()> {
        ensure_nonempty(&self.question, "question")?;
        ensure_nonempty(&self.answer, "answer")?;
        ensure_optional_nonempty(self.reasoning.as_deref(), "reasoning")?;
        ensure!(
            !require_reasoning || self.reasoning.is_some(),
            "`reasoning` is required by this task configuration"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PairwisePreferenceRecord {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
}

impl PairwisePreferenceRecord {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.prompt, "prompt")?;
        ensure_nonempty(&self.chosen, "chosen")?;
        ensure_nonempty(&self.rejected, "rejected")?;
        ensure!(
            self.chosen != self.rejected,
            "`chosen` and `rejected` must differ"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VerifiableRlRecord {
    pub prompt: String,
    pub verifier_payload: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_answer: Option<String>,
}

impl VerifiableRlRecord {
    fn validate(&self) -> Result<()> {
        ensure_nonempty(&self.prompt, "prompt")?;
        ensure!(
            !self.verifier_payload.is_null(),
            "`verifier_payload` must not be null"
        );
        ensure_optional_nonempty(self.reference_answer.as_deref(), "reference_answer")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn config(value: serde_json::Value) -> TaskConfig {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn every_task_exposes_an_algorithm_neutral_contract() {
        let tasks = [
            config(json!({"type": "causal_lm"})),
            config(json!({"type": "summarization"})),
            config(json!({"type": "retrieval_representation"})),
            config(json!({"type": "retrieval_ranking"})),
            config(json!({"type": "retrieval_planning"})),
            config(json!({"type": "instruction_tuning"})),
            config(json!({"type": "qa_reasoning"})),
            config(json!({"type": "pairwise_preference"})),
            config(json!({
                "type": "verifiable_rl",
                "verifier": {"adapter": "exact_answer"}
            })),
        ];

        for task in tasks {
            task.validate().unwrap();
            assert!(!task.name().is_empty());
            assert!(!task.contract().required_fields.is_empty());
            assert!(!task.contract().input_fields.is_empty());
            assert!(!task.contract().metrics.is_empty());
        }
    }

    #[test]
    fn qa_contract_exposes_configured_reasoning_requirement() {
        let optional = config(json!({"type": "qa_reasoning"}));
        assert_eq!(optional.contract().required_fields, ["question", "answer"]);

        let required = config(json!({
            "type": "qa_reasoning",
            "require_reasoning": true
        }));
        assert_eq!(
            required.contract().required_fields,
            ["question", "answer", "reasoning"]
        );
    }

    #[test]
    fn supervised_prompt_segments_borrow_large_record_fields() {
        let task = config(json!({"type": "summarization"}));
        let record = json!({
            "document": "a document large enough that copying it is undesirable",
            "summary": "short summary"
        });
        let document = record["document"].as_str().unwrap();
        let summary = record["summary"].as_str().unwrap();
        let segments = task.construct_supervised_prompt(&record).unwrap();
        assert!(matches!(&segments.source, Cow::Borrowed(_)));
        assert!(matches!(&segments.target, Cow::Borrowed(_)));
        assert_eq!(segments.source.as_ptr(), document.as_ptr());
        assert_eq!(segments.target.as_ptr(), summary.as_ptr());

        let reasoning = config(json!({
            "type": "qa_reasoning",
            "require_reasoning": true
        }));
        let record = json!({"question": "why?", "reasoning": "because", "answer": "therefore"});
        let segments = reasoning.construct_supervised_prompt(&record).unwrap();
        assert!(matches!(&segments.source, Cow::Borrowed(_)));
        assert!(matches!(&segments.target, Cow::Owned(_)));
    }

    #[test]
    fn shipped_task_schemas_accept_valid_records() {
        let cases = [
            (
                config(json!({"type": "causal_lm"})),
                json!({"text": "A document."}),
            ),
            (
                config(json!({"type": "summarization"})),
                json!({"document": "Long text", "summary": "Short text"}),
            ),
            (
                config(json!({"type": "retrieval_representation"})),
                json!({"query": "q", "positive": "p", "negatives": ["n"]}),
            ),
            (
                config(json!({"type": "retrieval_ranking"})),
                json!({
                    "query": "q",
                    "documents": [
                        {"document": "best", "relevance": 3.0},
                        {"document": "also relevant", "relevance": 1.0}
                    ]
                }),
            ),
            (
                config(json!({"type": "retrieval_planning"})),
                json!({"request": "find it", "context": "context", "plan": "search"}),
            ),
            (
                config(json!({"type": "instruction_tuning"})),
                json!({"instruction": "Do it", "input": "now", "response": "Done"}),
            ),
            (
                config(json!({"type": "qa_reasoning", "require_reasoning": true})),
                json!({"question": "1+1?", "reasoning": "Add one.", "answer": "2"}),
            ),
            (
                config(json!({"type": "pairwise_preference"})),
                json!({"prompt": "p", "chosen": "good", "rejected": "bad"}),
            ),
            (
                config(json!({
                    "type": "verifiable_rl",
                    "verifier": {"adapter": "unit_tests"}
                })),
                json!({"prompt": "implement", "verifier_payload": {"tests": ["a"]}}),
            ),
        ];

        for (task, record) in cases {
            task.validate_record(&record).unwrap();
        }
    }

    #[test]
    fn record_validation_rejects_semantically_useless_examples() {
        let representation = config(json!({"type": "retrieval_representation"}));
        let error = representation
            .validate_record(&json!({
                "query": "q",
                "positive": "same document",
                "negatives": ["same document"]
            }))
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicates the positive"), "{error}");

        let ranking = config(json!({"type": "retrieval_ranking"}));
        let error = ranking
            .validate_record(&json!({
                "query": "q",
                "documents": [
                    {"document": "a", "relevance": 1.0},
                    {"document": "b", "relevance": 1.0}
                ]
            }))
            .unwrap_err()
            .to_string();
        assert!(error.contains("distinct relevance grades"), "{error}");

        let error = ranking
            .validate_record(&json!({
                "query": "q",
                "documents": [
                    {"document": "same document", "relevance": 1.0},
                    {"document": "same document", "relevance": 0.0}
                ]
            }))
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicates another ranking"), "{error}");

        let error = ranking
            .validate_record(&json!({
                "query": "q",
                "documents": [
                    {"document": "a", "relevance": 1.0},
                    {"document": "b", "relevance": -1.0}
                ]
            }))
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-negative"), "{error}");

        let preference = config(json!({"type": "pairwise_preference"}));
        let error = preference
            .validate_record(&json!({"prompt": "p", "chosen": "same", "rejected": "same"}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("must differ"), "{error}");

        let reasoning = config(json!({"type": "qa_reasoning", "require_reasoning": true}));
        let error = reasoning
            .validate_record(&json!({"question": "q", "answer": "a"}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("reasoning"), "{error}");
    }

    #[test]
    fn retrieval_records_bound_per_example_candidate_materialization() {
        let representation = config(json!({"type": "retrieval_representation"}));
        let accepted_negatives = (0..MAX_RETRIEVAL_REPRESENTATION_DOCUMENTS - 1)
            .map(|index| format!("negative-{index}"))
            .collect::<Vec<_>>();
        representation
            .validate_record(&json!({
                "query": "q",
                "positive": "p",
                "negatives": accepted_negatives
            }))
            .unwrap();
        let excessive_negatives = (0..MAX_RETRIEVAL_REPRESENTATION_DOCUMENTS)
            .map(|index| format!("negative-{index}"))
            .collect::<Vec<_>>();
        let error = representation
            .validate_record(&json!({
                "query": "q",
                "positive": "p",
                "negatives": excessive_negatives
            }))
            .unwrap_err()
            .to_string();
        assert!(error.contains("including the positive"), "{error}");
        assert_eq!(
            representation.maximum_model_sequences_per_example(),
            MAX_RETRIEVAL_REPRESENTATION_DOCUMENTS + 1
        );

        let ranking = config(json!({"type": "retrieval_ranking"}));
        let documents = (0..=MAX_RETRIEVAL_RANKING_DOCUMENTS)
            .map(|index| json!({"document": format!("document-{index}"), "relevance": index}))
            .collect::<Vec<_>>();
        let error = ranking
            .validate_record(&json!({"query": "q", "documents": documents}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("candidate limit"), "{error}");
    }

    #[test]
    fn task_configuration_is_strict_and_validated() {
        let unknown = serde_json::from_value::<TaskConfig>(json!({
            "type": "causal_lm",
            "temperature": 0.1
        }))
        .unwrap_err()
        .to_string();
        assert!(unknown.contains("temperature"), "{unknown}");

        let task = config(json!({
            "type": "retrieval_representation",
            "temperature": 0.0
        }));
        let error = task.validate().unwrap_err().to_string();
        assert!(error.contains("finite and positive"), "{error}");
    }

    #[test]
    fn adapters_construct_inputs_losses_masks_and_rewards() {
        let reasoning = config(json!({
            "type": "qa_reasoning",
            "require_reasoning": true
        }));
        let example = reasoning
            .construct_example(&json!({
                "question": "Why?",
                "reasoning": "Because.",
                "answer": "Therefore."
            }))
            .unwrap();
        let TaskExample::SupervisedGeneration { prompt, target } = example else {
            panic!("expected supervised-generation example")
        };
        assert!(prompt.render().contains("Question:\nWhy?"));
        assert_eq!(prompt.truncatable_segment, Some(1));
        assert!(prompt.truncatable_segment_required);
        assert_eq!(target, "Reasoning:\nBecause.\n\nAnswer:\nTherefore.");
        assert_eq!(
            reasoning.loss_spec(),
            LossSpec::TokenCrossEntropy {
                mask: LossMaskPolicy::TargetTokensOnly
            }
        );
        assert!(reasoning.reward_spec().is_none());

        let retrieval = config(json!({
            "type": "retrieval_representation",
            "query_prefix": "query:",
            "document_prefix": "document:"
        }));
        let example = retrieval
            .construct_example(&json!({
                "query": "needle",
                "positive": "haystack",
                "negatives": ["other"]
            }))
            .unwrap();
        let TaskExample::RetrievalRepresentation {
            query, documents, ..
        } = example
        else {
            panic!("expected retrieval-representation example")
        };
        assert_eq!(query.segments, ["query:", "needle"]);
        assert_eq!(documents[0].segments, ["document:", "haystack"]);
        assert_eq!(documents[1].segments, ["document:", "other"]);
        assert_eq!(query.render(), "query:needle");
        query.validate().unwrap();
        for document in documents {
            document.validate().unwrap();
        }

        let rl = config(json!({
            "type": "verifiable_rl",
            "verifier": {"adapter": "exact_answer", "parameters": {"case_fold": true}}
        }));
        assert!(matches!(rl.loss_spec(), LossSpec::PolicyGradient));
        assert!(matches!(
            rl.reward_spec(),
            Some(RewardSpec::Verifier { verifier }) if verifier.adapter == "exact_answer"
        ));
    }
}
