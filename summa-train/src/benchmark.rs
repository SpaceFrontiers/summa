//! Deterministic execution of public and sealed model acceptance benchmarks.
//!
//! The runner never downloads data: callers materialize public or sealed
//! suites locally and point the runner at their manifests.  Baseline and
//! candidate evaluations receive the same model seed, example-order seed, and
//! GPU-hour allowance.  Raw measurements can be converted directly into the
//! higher-is-better paired inputs consumed by [`crate::acceptance`]. The run
//! artifact is audit evidence and therefore retains sealed case ids and
//! scores, but never suite examples; downstream promotion reports expose
//! sealed results only as an aggregate gate.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::acceptance::{
    ACCEPTANCE_SUITE_VERSION, AcceptancePolicy, AcceptanceSuite, BenchmarkCase, BenchmarkFamily,
    CaseResult, ExactResumeEvidence, PairedRun, PromotionReport, ResourceComparison,
    SuiteVisibility, VerifiedPromotionContext, evaluate_with_verified_context,
};
use crate::artifact_io::{read_regular_bounded, validate_sha256_hex, validate_sha256_identity};
use crate::metrics::metric_log_digests;
use crate::qat_candidate::open_qat_candidate;

pub const BENCHMARK_MANIFEST_VERSION: u32 = 2;
pub const MINIMUM_PAIRED_SEEDS: usize = 3;
pub const MAXIMUM_PAIRED_SEEDS: usize = 64;
const MAX_BENCHMARK_JSON_BYTES: u64 = 64 * 1024 * 1024;

/// Whether a catalog entry gates current promotion or belongs to a later
/// language/long-context sweep from the sleep paper reproduction plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogStage {
    Promotion,
    Sleep,
    LaterSweep,
}

impl CatalogStage {
    pub fn required_now(self) -> bool {
        !matches!(self, Self::LaterSweep)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogEntry {
    pub id: &'static str,
    pub family: BenchmarkFamily,
    pub stage: CatalogStage,
}

/// The minimum capability and in-model-sleep catalog.  `LaterSweep` entries
/// are intentionally registered now so result identifiers do not drift when
/// the more expensive language and long-context sweeps are enabled.
pub fn required_catalog() -> Vec<CatalogEntry> {
    vec![
        CatalogEntry {
            id: "pretraining_causal",
            family: BenchmarkFamily::Pretraining,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "summarization",
            family: BenchmarkFamily::Summarization,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "retrieval_representation_ranking",
            family: BenchmarkFamily::Retrieval,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "retrieval_planning",
            family: BenchmarkFamily::RetrievalPlanning,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "reasoning_qa",
            family: BenchmarkFamily::Reasoning,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "pairwise_preference",
            family: BenchmarkFamily::Preference,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "verifiable_rl",
            family: BenchmarkFamily::VerifiableRl,
            stage: CatalogStage::Promotion,
        },
        CatalogEntry {
            id: "synthetic_retention_smoke",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "clinc_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "banking77_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "mk_niah",
            family: BenchmarkFamily::LongContext,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "qasper",
            family: BenchmarkFamily::LongContext,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "squad_no_context",
            family: BenchmarkFamily::FactualIncorporation,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "arc_dreaming",
            family: BenchmarkFamily::Dreaming,
            stage: CatalogStage::Sleep,
        },
        CatalogEntry {
            id: "manchu_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::LaterSweep,
        },
        CatalogEntry {
            id: "kalamang_continual_learning",
            family: BenchmarkFamily::ContinualRetention,
            stage: CatalogStage::LaterSweep,
        },
        CatalogEntry {
            id: "babilong",
            family: BenchmarkFamily::LongContext,
            stage: CatalogStage::LaterSweep,
        },
    ]
}

/// Capacity-, compute-, and mechanism-matched variants required by the sleep
/// experiment matrix.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AblationId {
    WakeOnly,
    StaticReplay,
    CmsWithoutKnowledgeSeeding,
    FixedCapacityOnPolicyDistillation,
    ExpansionMatchedDistillation,
    ConsolidationWithoutDreaming,
    NoSemanticReward,
    NoImitationReward,
    NoGradientSelection,
    NoRandomExpert,
    StaticallyAvailableEqualCapacityMoe,
}

impl AblationId {
    pub const ALL: [Self; 11] = [
        Self::WakeOnly,
        Self::StaticReplay,
        Self::CmsWithoutKnowledgeSeeding,
        Self::FixedCapacityOnPolicyDistillation,
        Self::ExpansionMatchedDistillation,
        Self::ConsolidationWithoutDreaming,
        Self::NoSemanticReward,
        Self::NoImitationReward,
        Self::NoGradientSelection,
        Self::NoRandomExpert,
        Self::StaticallyAvailableEqualCapacityMoe,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WakeOnly => "wake_only",
            Self::StaticReplay => "static_replay",
            Self::CmsWithoutKnowledgeSeeding => "cms_without_knowledge_seeding",
            Self::FixedCapacityOnPolicyDistillation => "fixed_capacity_on_policy_distillation",
            Self::ExpansionMatchedDistillation => "expansion_matched_distillation",
            Self::ConsolidationWithoutDreaming => "consolidation_without_dreaming",
            Self::NoSemanticReward => "no_semantic_reward",
            Self::NoImitationReward => "no_imitation_reward",
            Self::NoGradientSelection => "no_gradient_selection",
            Self::NoRandomExpert => "no_random_expert",
            Self::StaticallyAvailableEqualCapacityMoe => "statically_available_equal_capacity_moe",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    #[default]
    Maximize,
    Minimize,
}

impl MetricDirection {
    fn acceptance_score(self, raw: f64) -> f64 {
        match self {
            Self::Maximize => raw,
            Self::Minimize => -raw,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkArtifact {
    /// Local file, resolved relative to the suite manifest.
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSpec {
    /// Globally unique result id.  It may identify a public/sealed variant.
    pub id: String,
    /// Stable catalog id; defaults to `id` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_id: Option<String>,
    pub family: BenchmarkFamily,
    pub metric: String,
    #[serde(default)]
    pub direction: MetricDirection,
    pub stable_anchor: bool,
    pub artifact: BenchmarkArtifact,
    /// Versioned evaluator options.  Trainer core does not interpret them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evaluator: BTreeMap<String, serde_json::Value>,
}

impl BenchmarkSpec {
    pub fn catalog_id(&self) -> &str {
        self.catalog_id.as_deref().unwrap_or(&self.id)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSuiteManifest {
    pub version: u32,
    pub suite_id: String,
    pub visibility: SuiteVisibility,
    pub cases: Vec<BenchmarkSpec>,
}

impl BenchmarkSuiteManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == BENCHMARK_MANIFEST_VERSION,
            "unsupported benchmark manifest version {}",
            self.version
        );
        ensure!(
            !self.suite_id.trim().is_empty(),
            "benchmark suite_id is empty"
        );
        ensure!(
            !self.cases.is_empty(),
            "benchmark suite `{}` has no cases",
            self.suite_id
        );
        let mut case_ids = BTreeSet::new();
        for case in &self.cases {
            ensure!(
                !case.id.trim().is_empty(),
                "benchmark suite `{}` contains an empty case id",
                self.suite_id
            );
            ensure!(
                case_ids.insert(case.id.as_str()),
                "benchmark suite `{}` repeats case `{}`",
                self.suite_id,
                case.id
            );
            ensure!(
                !case.catalog_id().trim().is_empty(),
                "benchmark case `{}` has an empty catalog_id",
                case.id
            );
            ensure!(
                !case.metric.trim().is_empty(),
                "benchmark case `{}` has an empty metric",
                case.id
            );
            ensure!(
                !case.artifact.path.as_os_str().is_empty(),
                "benchmark case `{}` has an empty artifact path",
                case.id
            );
        }
        Ok(())
    }
}

/// One suite data file, resolved against the manifest that declared it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SuiteArtifact {
    pub path: PathBuf,
    pub bytes: u64,
}

/// A suite manifest whose declared data files were all located on disk.
#[derive(Clone, Debug)]
pub struct LoadedBenchmarkSuite {
    pub manifest: BenchmarkSuiteManifest,
    pub manifest_path: PathBuf,
    artifacts: BTreeMap<String, SuiteArtifact>,
}

impl LoadedBenchmarkSuite {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = read_json_file(path, "benchmark manifest")?;
        let manifest: BenchmarkSuiteManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid benchmark manifest {}", path.display()))?;
        manifest.validate()?;

        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let mut artifacts = BTreeMap::new();
        for case in &manifest.cases {
            let artifact_path = if case.artifact.path.is_absolute() {
                case.artifact.path.clone()
            } else {
                base.join(&case.artifact.path)
            };
            let metadata = std::fs::metadata(&artifact_path).with_context(|| {
                format!(
                    "artifact for benchmark `{}` is unavailable at {}",
                    case.id,
                    artifact_path.display()
                )
            })?;
            ensure!(
                metadata.is_file(),
                "artifact for benchmark `{}` at {} is not a regular file",
                case.id,
                artifact_path.display()
            );
            artifacts.insert(
                case.id.clone(),
                SuiteArtifact {
                    path: artifact_path,
                    bytes: metadata.len(),
                },
            );
        }
        Ok(Self {
            manifest,
            manifest_path: path.to_path_buf(),
            artifacts,
        })
    }

    pub fn artifact(&self, case_id: &str) -> Option<&SuiteArtifact> {
        self.artifacts.get(case_id)
    }
}

/// Check that manifests cover the fixed catalog with the expected benchmark
/// family.  Later sweeps can be required explicitly for a full reproduction.
pub fn validate_catalog_coverage(
    suites: &[LoadedBenchmarkSuite],
    include_later_sweeps: bool,
) -> Result<()> {
    let mut cases = BTreeMap::new();
    let mut visibilities = BTreeSet::new();
    for suite in suites {
        visibilities.insert(suite.manifest.visibility);
        for case in &suite.manifest.cases {
            let key = (case.catalog_id(), suite.manifest.visibility);
            ensure!(
                cases.insert(key, case).is_none(),
                "benchmark catalog repeats {:?} entry `{}`",
                suite.manifest.visibility,
                case.catalog_id()
            );
        }
    }
    ensure!(
        visibilities.contains(&SuiteVisibility::Public)
            && visibilities.contains(&SuiteVisibility::Sealed),
        "promotion benchmarks require both public and sealed suites"
    );
    for entry in required_catalog() {
        if !entry.stage.required_now() && !include_later_sweeps {
            continue;
        }
        for visibility in [SuiteVisibility::Public, SuiteVisibility::Sealed] {
            let case = cases.get(&(entry.id, visibility)).with_context(|| {
                format!(
                    "benchmark catalog is missing {} `{}`",
                    match visibility {
                        SuiteVisibility::Public => "public",
                        SuiteVisibility::Sealed => "sealed",
                    },
                    entry.id
                )
            })?;
            ensure!(
                case.family == entry.family,
                "benchmark catalog `{}` has {:?} family {:?}, expected {:?}",
                entry.id,
                visibility,
                case.family,
                entry.family
            );
        }
    }
    Ok(())
}

pub const TRAINING_ACCOUNTING_VERSION: u32 = 1;
pub const TRAINING_ACCOUNTING_FILE: &str = "training-accounting.json";
const CHECKPOINT_WEIGHTS_FILE: &str = "weights.safetensors";
const GENERATION_MANIFEST_FILE: &str = "generation-manifest.json";

/// Resource measurements sealed *inside* a checkpoint generation. This layer
/// intentionally does not contain the generation-manifest hash, avoiding a
/// cycle while letting the manifest authenticate every accounting field.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingAccounting {
    pub version: u32,
    pub training_gpu_hours: f64,
    pub parameters: u64,
    pub routed_active_parameters: u64,
    pub weights_bytes: u64,
    pub weights_sha256: String,
}

impl TrainingAccounting {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == TRAINING_ACCOUNTING_VERSION,
            "unsupported training-accounting version {}",
            self.version
        );
        ensure!(
            self.training_gpu_hours.is_finite() && self.training_gpu_hours > 0.0,
            "training accounting has invalid measured GPU hours"
        );
        ensure!(
            self.parameters > 0 && self.routed_active_parameters > 0,
            "training accounting has an empty parameter count"
        );
        ensure!(
            self.routed_active_parameters <= self.parameters,
            "training accounting routes more parameters than it stores"
        );
        ensure!(self.weights_bytes > 0, "training accounting has no weights");
        validate_sha256_hex(&self.weights_sha256, "training-accounting model weights")
            .context("invalid model-weights hash in training accounting")?;
        Ok(())
    }
}

/// The generation-manifest fields the exact-resume gate needs. Checkpoint
/// sealing and its own manifest verification live in the trainer's checkpoint
/// module; this is a read-only view of the progress a generation records.
#[derive(Debug, Deserialize)]
struct CheckpointProgress {
    global_step: usize,
}

/// Concrete model representation evaluated by an acceptance backend. This is
/// required in the current schema: callers may not rely on a historical
/// implicit full-precision default.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelRepresentationTarget {
    FullPrecision,
    Hquant {
        /// Canonical `candidate.json` produced by the QAT candidate publisher.
        candidate_manifest: PathBuf,
        /// Prefixed content identity returned by that publisher.
        candidate_manifest_sha256: String,
    },
}

/// Model transport supplied to the evaluator. HQUANT remains an injected
/// backend responsibility; this contract exposes the sealed archive directly
/// and never substitutes a dequantized FP evaluation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResolvedModelRepresentation {
    FullPrecision {
        weights: PathBuf,
        stored_bytes: u64,
    },
    Hquant {
        candidate_manifest: PathBuf,
        source_weights: PathBuf,
        archive: PathBuf,
        archive_manifest: PathBuf,
        stored_bytes: u64,
    },
}

/// Model identity passed to an evaluator. The checkpoint describes how the
/// model was trained; `representation` names the exact FP or sealed HQUANT
/// bytes whose quality is measured.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkTarget {
    pub id: String,
    pub checkpoint_manifest: PathBuf,
    pub checkpoint_manifest_sha256: String,
    pub training_gpu_hours: f64,
    pub parameters: u64,
    pub routed_active_parameters: u64,
    pub stored_bytes: u64,
    pub representation: ModelRepresentationTarget,
}

impl BenchmarkTarget {
    /// Resolve artifact paths against the target JSON that declared them.
    /// Callers must do this before verification so execution never depends on
    /// the process working directory.
    pub fn resolve_paths(&mut self, source: &Path) {
        let base = source.parent().unwrap_or_else(|| Path::new("."));
        if self.checkpoint_manifest.is_relative() {
            self.checkpoint_manifest = base.join(&self.checkpoint_manifest);
        }
        if let ModelRepresentationTarget::Hquant {
            candidate_manifest, ..
        } = &mut self.representation
            && candidate_manifest.is_relative()
        {
            *candidate_manifest = base.join(&*candidate_manifest);
        }
    }

    fn validate_identity(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "benchmark target id is empty");
        ensure!(
            self.parameters > 0 && self.routed_active_parameters > 0,
            "benchmark target `{}` has an empty parameter count",
            self.id
        );
        ensure!(
            self.routed_active_parameters <= self.parameters,
            "benchmark target `{}` routes more parameters than it stores",
            self.id
        );
        validate_sha256_hex(
            &self.checkpoint_manifest_sha256,
            "benchmark target checkpoint manifest",
        )
        .with_context(|| format!("invalid checkpoint hash for `{}`", self.id))?;
        ensure!(
            self.training_gpu_hours.is_finite() && self.training_gpu_hours > 0.0,
            "benchmark target `{}` has invalid measured training GPU hours",
            self.id
        );
        ensure!(
            self.stored_bytes > 0,
            "benchmark target `{}` has no stored-byte measurement",
            self.id
        );
        if let ModelRepresentationTarget::Hquant {
            candidate_manifest,
            candidate_manifest_sha256,
        } = &self.representation
        {
            ensure!(
                !candidate_manifest.as_os_str().is_empty(),
                "HQUANT benchmark target `{}` has no candidate manifest",
                self.id
            );
            validate_sha256_identity(candidate_manifest_sha256, "HQUANT candidate manifest")
                .with_context(|| format!("invalid HQUANT candidate hash for `{}`", self.id))?;
        }
        Ok(())
    }

    /// Resolve the exact bytes an evaluator must execute. Paths are used as
    /// declared, so callers must call [`Self::resolve_paths`] first.
    pub fn resolve_model(&self) -> Result<ResolvedModelRepresentation> {
        self.validate_identity()?;
        ensure!(
            self.checkpoint_manifest
                .file_name()
                .and_then(|name| name.to_str())
                == Some(GENERATION_MANIFEST_FILE),
            "checkpoint manifest for `{}` must be named {GENERATION_MANIFEST_FILE}",
            self.id
        );
        let generation = self
            .checkpoint_manifest
            .parent()
            .context("checkpoint manifest has no generation directory")?;
        match &self.representation {
            ModelRepresentationTarget::FullPrecision => {
                let weights = generation.join(CHECKPOINT_WEIGHTS_FILE);
                let metadata = std::fs::metadata(&weights).with_context(|| {
                    format!(
                        "model weights for `{}` are unavailable at {}",
                        self.id,
                        weights.display()
                    )
                })?;
                ensure!(
                    metadata.is_file(),
                    "model weights for `{}` at {} are not a regular file",
                    self.id,
                    weights.display()
                );
                ensure!(
                    metadata.len() == self.stored_bytes,
                    "full-precision benchmark target `{}` stored bytes differ from its weights",
                    self.id
                );
                Ok(ResolvedModelRepresentation::FullPrecision {
                    weights,
                    stored_bytes: self.stored_bytes,
                })
            }
            ModelRepresentationTarget::Hquant {
                candidate_manifest, ..
            } => {
                ensure!(
                    candidate_manifest
                        .file_name()
                        .and_then(|name| name.to_str())
                        == Some("candidate.json"),
                    "HQUANT benchmark target `{}` must address candidate.json",
                    self.id
                );
                let candidate_root = candidate_manifest
                    .parent()
                    .context("HQUANT candidate manifest has no candidate directory")?;
                let publication = open_qat_candidate(candidate_root).with_context(|| {
                    format!("invalid sealed HQUANT candidate for `{}`", self.id)
                })?;
                ensure!(
                    publication.metrics.archive_weight_elements == self.parameters,
                    "HQUANT candidate for `{}` has a different parameter inventory",
                    self.id
                );
                ensure!(
                    publication.metrics.archive_weight_bytes == self.stored_bytes,
                    "HQUANT benchmark target `{}` stored bytes differ from its validated archive members",
                    self.id
                );
                Ok(ResolvedModelRepresentation::Hquant {
                    candidate_manifest: publication.candidate_manifest_path,
                    source_weights: publication.weights_path,
                    archive: publication.archive_path,
                    archive_manifest: publication.archive_manifest_path,
                    stored_bytes: self.stored_bytes,
                })
            }
        }
    }
}

fn read_json_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_regular_bounded(path, MAX_BENCHMARK_JSON_BYTES, label)
        .with_context(|| format!("failed to read {label} at {}", path.display()))
}

const MAX_BENCHMARK_EVALUATOR_ARGUMENTS: usize = 64;
const MAX_BENCHMARK_EVALUATOR_ARGUMENT_BYTES: usize = 4_096;

pub(crate) fn validate_benchmark_evaluator_arguments(arguments: &[String]) -> Result<()> {
    ensure!(
        arguments.len() <= MAX_BENCHMARK_EVALUATOR_ARGUMENTS
            && arguments.iter().all(|argument| {
                argument.len() <= MAX_BENCHMARK_EVALUATOR_ARGUMENT_BYTES && !argument.contains('\0')
            }),
        "benchmark evaluator arguments exceed protocol limits"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRunConfig {
    pub evaluator_id: String,
    pub evaluator_version: String,
    /// Exact UTF-8 argument vector passed to the evaluator. Keeping it in the
    /// run configuration prevents two materially different harness invocations
    /// from sharing one evaluator identity.
    pub evaluator_arguments: Vec<String>,
    /// Strictly increasing order; the order itself is part of run identity.
    pub paired_seeds: Vec<u64>,
    /// Controls example ordering inside every evaluator.  A case-specific seed
    /// is derived from it and shared by baseline and candidate.
    pub order_seed: u64,
    /// Equal hard allowance for each (case, seed, target) evaluation.
    pub gpu_hours_per_evaluation: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ablation: Option<AblationId>,
}

impl BenchmarkRunConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.evaluator_id.trim().is_empty() && !self.evaluator_version.trim().is_empty(),
            "benchmark evaluator id and version are required"
        );
        validate_sha256_identity(&self.evaluator_version, "benchmark evaluator version")
            .context("benchmark evaluator_version is not a pinned executable identity")?;
        validate_benchmark_evaluator_arguments(&self.evaluator_arguments)?;
        ensure!(
            (MINIMUM_PAIRED_SEEDS..=MAXIMUM_PAIRED_SEEDS).contains(&self.paired_seeds.len()),
            "benchmarks require between {MINIMUM_PAIRED_SEEDS} and {MAXIMUM_PAIRED_SEEDS} paired seeds"
        );
        ensure!(
            self.paired_seeds.windows(2).all(|pair| pair[0] < pair[1]),
            "paired benchmark seeds must be unique and strictly increasing"
        );
        ensure!(
            self.gpu_hours_per_evaluation.is_finite() && self.gpu_hours_per_evaluation > 0.0,
            "gpu_hours_per_evaluation must be finite and positive"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetRole {
    Baseline,
    Candidate,
}

/// Fully deterministic evaluator input. The evaluator must execute `model`,
/// consume examples in `example_order_seed` order, and stop at
/// `max_gpu_hours`. `target` supplies training lineage and resource identity;
/// it is not permission to substitute the checkpoint's FP weights for an
/// HQUANT `model`.
#[derive(Clone, Copy)]
pub struct EvaluationRequest<'a> {
    pub suite_id: &'a str,
    pub visibility: SuiteVisibility,
    pub case: &'a BenchmarkSpec,
    pub artifact: &'a SuiteArtifact,
    pub target: &'a BenchmarkTarget,
    /// Bytes the backend must execute for this measurement.
    pub model: &'a ResolvedModelRepresentation,
    pub role: TargetRole,
    pub model_seed: u64,
    pub example_order_seed: u64,
    pub pair_ordinal: usize,
    pub max_gpu_hours: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationMeasurement {
    pub score: f64,
    pub gpu_hours: f64,
    pub examples: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
}

impl EvaluationMeasurement {
    pub fn validate(&self, case_id: &str, limit: f64) -> Result<()> {
        ensure!(
            self.score.is_finite(),
            "benchmark `{case_id}` returned a non-finite score"
        );
        ensure!(
            self.gpu_hours.is_finite() && self.gpu_hours >= 0.0,
            "benchmark `{case_id}` returned invalid GPU hours"
        );
        // Accommodate a few floating point ulps at the accounting boundary.
        ensure!(
            self.gpu_hours <= limit + f64::EPSILON * limit.max(1.0) * 8.0,
            "benchmark `{case_id}` exceeded its fixed GPU-hour allowance: {} > {}",
            self.gpu_hours,
            limit
        );
        ensure!(
            self.examples > 0,
            "benchmark `{case_id}` evaluated no examples"
        );
        ensure!(
            self.metrics.values().all(|value| value.is_finite()),
            "benchmark `{case_id}` returned a non-finite auxiliary metric"
        );
        Ok(())
    }
}

/// Implementations can wrap an in-process harness or invoke a local worker.
/// They must reject a representation they cannot execute. Artifact
/// materialization and network access deliberately remain outside this
/// interface.
pub trait BenchmarkEvaluator {
    fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawPairedResult {
    pub seed: u64,
    pub example_order_seed: u64,
    pub baseline: EvaluationMeasurement,
    pub candidate: EvaluationMeasurement,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawCaseResult {
    pub suite_id: String,
    pub case_id: String,
    pub catalog_id: String,
    pub visibility: SuiteVisibility,
    pub family: BenchmarkFamily,
    pub metric: String,
    pub direction: MetricDirection,
    pub stable_anchor: bool,
    pub pairs: Vec<RawPairedResult>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRunMetadata {
    pub evaluator_id: String,
    pub evaluator_version: String,
    pub evaluator_arguments: Vec<String>,
    pub paired_seeds: Vec<u64>,
    pub order_seed: u64,
    pub fixed_case_order: Vec<String>,
    pub gpu_hours_per_evaluation: f64,
    pub gpu_hour_budget_per_target: f64,
    pub baseline: BenchmarkTarget,
    pub candidate: BenchmarkTarget,
    /// Every suite id whose cases appear in this run, in sorted order.
    pub suite_ids: BTreeSet<String>,
    pub ablation: Option<AblationId>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRun {
    pub metadata: BenchmarkRunMetadata,
    pub cases: Vec<RawCaseResult>,
}

impl BenchmarkRun {
    pub fn capacity_matched(&self) -> bool {
        self.metadata.baseline.parameters == self.metadata.candidate.parameters
    }

    pub fn routed_active_parameters_matched(&self) -> bool {
        self.metadata.baseline.routed_active_parameters
            == self.metadata.candidate.routed_active_parameters
    }

    pub fn training_compute_matched(&self) -> bool {
        same_f64(
            self.metadata.baseline.training_gpu_hours,
            self.metadata.candidate.training_gpu_hours,
        )
    }

    /// Convert raw metrics into the higher-is-better convention required by
    /// the paired promotion gates.  Lower-is-better metrics are negated.
    pub fn acceptance_inputs(&self) -> (AcceptanceSuite, Vec<CaseResult>) {
        let suite = AcceptanceSuite {
            version: ACCEPTANCE_SUITE_VERSION,
            cases: self
                .cases
                .iter()
                .map(|case| BenchmarkCase {
                    id: case.case_id.clone(),
                    catalog_id: case.catalog_id.clone(),
                    family: case.family.clone(),
                    visibility: case.visibility,
                    metric: case.metric.clone(),
                    stable_anchor: case.stable_anchor,
                })
                .collect(),
        };
        let results = self
            .cases
            .iter()
            .map(|case| CaseResult {
                case_id: case.case_id.clone(),
                pairs: case
                    .pairs
                    .iter()
                    .map(|pair| PairedRun {
                        seed: pair.seed,
                        baseline: case.direction.acceptance_score(pair.baseline.score),
                        candidate: case.direction.acceptance_score(pair.candidate.score),
                    })
                    .collect(),
            })
            .collect();
        (suite, results)
    }

    pub fn measured_gpu_hours(&self, role: TargetRole) -> f64 {
        self.cases
            .iter()
            .flat_map(|case| &case.pairs)
            .map(|pair| match role {
                TargetRole::Baseline => pair.baseline.gpu_hours,
                TargetRole::Candidate => pair.candidate.gpu_hours,
            })
            .sum()
    }

    /// Validate every relationship that is encoded inside a run artifact.
    /// Checkpoint and suite files are resolved separately, so this stays
    /// portable after a run is copied to an evaluation directory.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.metadata.evaluator_id.trim().is_empty()
                && !self.metadata.evaluator_version.trim().is_empty(),
            "benchmark run has no evaluator identity"
        );
        validate_sha256_identity(
            &self.metadata.evaluator_version,
            "benchmark run evaluator version",
        )
        .context("benchmark run evaluator_version is not a pinned executable identity")?;
        validate_benchmark_evaluator_arguments(&self.metadata.evaluator_arguments)?;
        ensure!(
            (MINIMUM_PAIRED_SEEDS..=MAXIMUM_PAIRED_SEEDS)
                .contains(&self.metadata.paired_seeds.len()),
            "benchmark run requires between {MINIMUM_PAIRED_SEEDS} and {MAXIMUM_PAIRED_SEEDS} paired seeds"
        );
        ensure!(
            self.metadata
                .paired_seeds
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "benchmark run seeds must be unique and strictly increasing"
        );
        ensure!(
            self.metadata.gpu_hours_per_evaluation.is_finite()
                && self.metadata.gpu_hours_per_evaluation > 0.0,
            "benchmark run has an invalid GPU-hour allowance"
        );
        self.metadata.baseline.validate_identity()?;
        self.metadata.candidate.validate_identity()?;
        ensure!(
            self.metadata.baseline.id != self.metadata.candidate.id,
            "benchmark baseline and candidate identities are equal"
        );
        ensure!(!self.cases.is_empty(), "benchmark run contains no cases");
        ensure!(
            self.metadata.fixed_case_order.len() == self.cases.len(),
            "fixed case order does not cover every result"
        );
        ensure!(
            self.metadata
                .fixed_case_order
                .iter()
                .map(String::as_str)
                .eq(self.cases.iter().map(|case| case.case_id.as_str())),
            "benchmark results do not follow the recorded fixed case order"
        );
        ensure!(
            !self.metadata.suite_ids.is_empty(),
            "benchmark run contains no suite ids"
        );
        ensure!(
            self.metadata
                .suite_ids
                .iter()
                .all(|suite_id| !suite_id.trim().is_empty()),
            "empty suite id in run metadata"
        );

        let mut case_ids = BTreeSet::new();
        for case in &self.cases {
            ensure!(
                case_ids.insert(case.case_id.as_str()),
                "duplicate benchmark result `{}`",
                case.case_id
            );
            ensure!(
                !case.case_id.trim().is_empty()
                    && !case.catalog_id.trim().is_empty()
                    && !case.metric.trim().is_empty(),
                "benchmark result contains an empty identity or metric"
            );
            ensure!(
                self.metadata.suite_ids.contains(&case.suite_id),
                "benchmark result `{}` refers to unknown suite `{}`",
                case.case_id,
                case.suite_id
            );
            ensure!(
                case.pairs.len() == self.metadata.paired_seeds.len(),
                "benchmark `{}` does not contain every paired seed",
                case.case_id
            );
            for (ordinal, (pair, expected_seed)) in case
                .pairs
                .iter()
                .zip(&self.metadata.paired_seeds)
                .enumerate()
            {
                ensure!(
                    pair.seed == *expected_seed,
                    "benchmark `{}` seed order differs from run metadata",
                    case.case_id
                );
                ensure!(
                    pair.example_order_seed
                        == derive_order_seed(
                            self.metadata.order_seed,
                            &case.suite_id,
                            &case.case_id,
                            pair.seed,
                        ),
                    "benchmark `{}` seed {} has the wrong example order",
                    case.case_id,
                    pair.seed
                );
                pair.baseline.validate(
                    &format!("{} baseline pair {ordinal}", case.case_id),
                    self.metadata.gpu_hours_per_evaluation,
                )?;
                pair.candidate.validate(
                    &format!("{} candidate pair {ordinal}", case.case_id),
                    self.metadata.gpu_hours_per_evaluation,
                )?;
                ensure!(
                    pair.baseline.examples == pair.candidate.examples,
                    "benchmark `{}` seed {} evaluated different example counts",
                    case.case_id,
                    pair.seed
                );
            }
        }
        let expected_budget = self.metadata.gpu_hours_per_evaluation
            * self.cases.len() as f64
            * self.metadata.paired_seeds.len() as f64;
        ensure!(
            same_f64(self.metadata.gpu_hour_budget_per_target, expected_budget),
            "benchmark run GPU-hour budget does not match its schedule"
        );
        Ok(())
    }
}

/// A benchmark result whose internal relationships have been validated,
/// together with the artifact path its file references resolve against.
#[derive(Clone, Debug)]
pub struct LoadedBenchmarkRun {
    pub(crate) run: BenchmarkRun,
    pub(crate) path: PathBuf,
}

impl LoadedBenchmarkRun {
    pub fn run(&self) -> &BenchmarkRun {
        &self.run
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = read_json_file(path, "benchmark run")?;
        let run: BenchmarkRun = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid benchmark run {}", path.display()))?;
        run.validate()?;
        Ok(Self {
            run,
            path: path.to_path_buf(),
        })
    }
}

/// Prove that an interrupted run and an uninterrupted reference reached
/// byte-identical final state and semantically identical progress from the
/// same checkpoint.
///
/// This is the one exact-resume property promotion still enforces end to end:
/// the final generation manifests must be byte-equal, and the two metric
/// journals must recompute the same semantic progress even though their
/// wall-clock timings and raw bytes differ. Relative artifact paths resolve
/// against `source`, the artifact that declared them.
pub fn verify_exact_resume(evidence: &ExactResumeEvidence, source: &Path) -> Result<()> {
    let mut resolved = evidence.clone();
    resolve_exact_resume_paths(&mut resolved, source);
    verify_exact_resume_artifacts(&resolved)
}

fn resolve_exact_resume_paths(evidence: &mut ExactResumeEvidence, source: &Path) {
    let base = source.parent().unwrap_or_else(|| Path::new("."));
    for artifact in [
        &mut evidence.interrupted_checkpoint,
        &mut evidence.uninterrupted_final_state,
        &mut evidence.resumed_final_state,
        &mut evidence.uninterrupted_metrics,
        &mut evidence.resumed_metrics,
    ] {
        if artifact.path.is_relative() {
            artifact.path = base.join(&artifact.path);
        }
    }
}

fn read_generation_manifest(path: &Path, label: &str) -> Result<(Vec<u8>, CheckpointProgress)> {
    ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(GENERATION_MANIFEST_FILE),
        "{label} must address {GENERATION_MANIFEST_FILE}"
    );
    let bytes = read_json_file(path, label)?;
    let progress: CheckpointProgress =
        serde_json::from_slice(&bytes).with_context(|| format!("invalid {label}"))?;
    Ok((bytes, progress))
}

fn verify_exact_resume_artifacts(evidence: &ExactResumeEvidence) -> Result<()> {
    ensure!(
        evidence.interruption_step == evidence.resumed_from_step,
        "resumed step does not equal the interrupted checkpoint step"
    );
    ensure!(
        evidence.uninterrupted_final_state.path != evidence.resumed_final_state.path
            && evidence.uninterrupted_metrics.path != evidence.resumed_metrics.path,
        "exact-resume comparison requires distinct uninterrupted and resumed artifacts"
    );
    let (_, interrupted) = read_generation_manifest(
        &evidence.interrupted_checkpoint.path,
        "interrupted checkpoint",
    )?;
    let (uninterrupted_bytes, uninterrupted) = read_generation_manifest(
        &evidence.uninterrupted_final_state.path,
        "uninterrupted final state",
    )?;
    let (resumed_bytes, resumed) =
        read_generation_manifest(&evidence.resumed_final_state.path, "resumed final state")?;
    ensure!(
        interrupted.global_step as u64 == evidence.interruption_step,
        "interrupted checkpoint global step does not match resume evidence"
    );
    ensure!(
        uninterrupted.global_step == resumed.global_step
            && uninterrupted.global_step > interrupted.global_step,
        "exact-resume final checkpoints do not describe the same later step"
    );
    // The sealed generation manifest inventories every checkpoint member and
    // its digest, so byte-equal manifests mean byte-equal final state.
    ensure!(
        uninterrupted_bytes == resumed_bytes,
        "interrupted and uninterrupted runs did not reach byte-identical final state"
    );

    let uninterrupted_digest = metric_log_digests(&evidence.uninterrupted_metrics.path, None)?;
    let resumed_digest = metric_log_digests(&evidence.resumed_metrics.path, None)?;
    ensure!(
        uninterrupted_digest.last_global_step == Some(uninterrupted.global_step as u64)
            && resumed_digest.last_global_step == Some(resumed.global_step as u64),
        "exact-resume metric journal does not end at its final checkpoint"
    );
    ensure!(
        uninterrupted_digest.records > 0 && resumed_digest.records > 0,
        "exact-resume metric journals are empty"
    );
    ensure!(
        uninterrupted_digest.semantic_progress_sha256 == resumed_digest.semantic_progress_sha256,
        "interrupted and uninterrupted metric journals differ semantically"
    );
    Ok(())
}

/// Evaluate a promotion from a complete benchmark and resource evidence set.
/// `comparison_runs` is the fixed sleep ablation matrix; it is used to derive
/// (rather than assert) which capacity- and compute-matched baseline is
/// strongest. `resources_path` is the artifact the comparison was read from;
/// exact-resume artifact paths resolve against it.
pub fn evaluate_verified_promotion(
    selected_run: &LoadedBenchmarkRun,
    comparison_runs: &[LoadedBenchmarkRun],
    comparison: &ResourceComparison,
    resources_path: &Path,
    policy: &AcceptancePolicy,
) -> Result<PromotionReport> {
    comparison.validate()?;
    let strongest_baseline_id = verified_resource_benchmark_context(selected_run, comparison_runs)?;

    ensure!(
        comparison.baseline_id == selected_run.run.metadata.baseline.id
            && comparison.candidate_id == selected_run.run.metadata.candidate.id,
        "resource evidence target identities do not match the selected benchmark run"
    );
    ensure!(
        comparison.strongest_baseline_id == strongest_baseline_id,
        "resource evidence names `{}` as strongest, but the comparison runs derive `{}`",
        comparison.strongest_baseline_id,
        strongest_baseline_id
    );
    verify_exact_resume(&comparison.exact_resume, resources_path)?;
    let (suite, results) = selected_run.run.acceptance_inputs();
    let context = VerifiedPromotionContext {
        strongest_baseline_id: &strongest_baseline_id,
        baseline_id: &selected_run.run.metadata.baseline.id,
        candidate_id: &selected_run.run.metadata.candidate.id,
        capacity_matched: selected_run.run.capacity_matched(),
        active_capacity_matched: selected_run.run.routed_active_parameters_matched(),
        fixed_gpu_hour_measured: selected_run.run.training_compute_matched(),
        baseline_stored_parameters: selected_run.run.metadata.baseline.parameters,
        candidate_stored_parameters: selected_run.run.metadata.candidate.parameters,
        baseline_stored_bytes: selected_run.run.metadata.baseline.stored_bytes,
        candidate_stored_bytes: selected_run.run.metadata.candidate.stored_bytes,
        candidate_routed_active_parameters: selected_run
            .run
            .metadata
            .candidate
            .routed_active_parameters,
        exact_resume_verified: true,
    };
    evaluate_with_verified_context(&suite, &results, comparison, policy, &context)
}

pub(crate) fn verified_resource_benchmark_context(
    selected_run: &LoadedBenchmarkRun,
    comparison_runs: &[LoadedBenchmarkRun],
) -> Result<String> {
    selected_run.run.validate()?;
    validate_run_catalog_coverage(&selected_run.run, false)?;
    let selected_occurrences = comparison_runs
        .iter()
        .filter(|run| run.path == selected_run.path)
        .count();
    ensure!(
        selected_occurrences == 1,
        "the selected benchmark run must occur exactly once in the comparison set"
    );
    let mut resolved_targets = BTreeSet::new();
    resolve_target_once(
        &selected_run.run.metadata.baseline,
        &selected_run.path,
        &mut resolved_targets,
    )?;
    resolve_target_once(
        &selected_run.run.metadata.candidate,
        &selected_run.path,
        &mut resolved_targets,
    )?;
    validate_comparison_set(comparison_runs, &mut resolved_targets)?;
    derive_strongest_baseline(comparison_runs)
}

fn validate_run_catalog_coverage(run: &BenchmarkRun, include_later_sweeps: bool) -> Result<()> {
    let mut covered = BTreeMap::new();
    for case in &run.cases {
        let key = (case.catalog_id.as_str(), case.visibility);
        ensure!(
            covered.insert(key, &case.family).is_none(),
            "benchmark run repeats {:?} catalog entry `{}`",
            case.visibility,
            case.catalog_id
        );
    }
    for entry in required_catalog() {
        if !entry.stage.required_now() && !include_later_sweeps {
            continue;
        }
        for visibility in [SuiteVisibility::Public, SuiteVisibility::Sealed] {
            let family = covered.get(&(entry.id, visibility)).with_context(|| {
                format!(
                    "benchmark run is missing {:?} catalog entry `{}`",
                    visibility, entry.id
                )
            })?;
            ensure!(
                **family == entry.family,
                "benchmark run catalog `{}` has {:?} family {:?}, expected {:?}",
                entry.id,
                visibility,
                family,
                entry.family
            );
        }
    }
    Ok(())
}

fn resolve_target_once(
    target: &BenchmarkTarget,
    source: &Path,
    resolved_targets: &mut BTreeSet<String>,
) -> Result<()> {
    // The fixed ablation matrix deliberately reuses one candidate in every
    // run, and an HQUANT archive may be many gigabytes. Resolve each distinct
    // target id once per comparison instead of reopening it for every
    // ablation.
    if !resolved_targets.insert(target.id.clone()) {
        return Ok(());
    }
    // Resolve on a clone so the run stays identical to its stored artifact
    // while resolution reads the files the run actually declared, never paths
    // relative to the process directory.
    let mut target = target.clone();
    target.resolve_paths(source);
    target.resolve_model()?;
    Ok(())
}

fn validate_comparison_set(
    runs: &[LoadedBenchmarkRun],
    resolved_targets: &mut BTreeSet<String>,
) -> Result<()> {
    ensure!(
        runs.len() == AblationId::ALL.len(),
        "promotion comparison set must contain all {} fixed ablations",
        AblationId::ALL.len()
    );
    let reference = runs.first().context("promotion comparison set is empty")?;
    let mut ablations = BTreeSet::new();
    let mut baseline_ids = BTreeSet::new();
    let mut baseline_checkpoints = BTreeSet::new();
    for run in runs {
        resolve_target_once(&run.run.metadata.baseline, &run.path, resolved_targets)?;
        resolve_target_once(&run.run.metadata.candidate, &run.path, resolved_targets)?;
        run.run.validate()?;
        validate_run_catalog_coverage(&run.run, false)?;
        let ablation = run
            .run
            .metadata
            .ablation
            .context("every promotion comparison run must name its ablation")?;
        ensure!(
            ablations.insert(ablation),
            "comparison set repeats ablation `{}`",
            ablation.as_str()
        );
        ensure!(
            baseline_ids.insert(run.run.metadata.baseline.id.as_str()),
            "comparison set repeats baseline `{}`",
            run.run.metadata.baseline.id
        );
        ensure!(
            baseline_checkpoints.insert(
                run.run
                    .metadata
                    .baseline
                    .checkpoint_manifest_sha256
                    .to_ascii_lowercase()
            ),
            "comparison baselines `{}` and another ablation reuse the same checkpoint",
            run.run.metadata.baseline.id
        );
        ensure!(
            run.run.capacity_matched()
                && run.run.routed_active_parameters_matched()
                && run.run.training_compute_matched(),
            "baseline `{}` is not capacity-, routed-active-, and training-compute-matched",
            run.run.metadata.baseline.id
        );
        validate_comparison_compatibility(&reference.run, &run.run)?;
    }
    ensure!(
        AblationId::ALL.into_iter().collect::<BTreeSet<_>>() == ablations,
        "promotion comparison set does not cover the fixed ablation catalog"
    );
    Ok(())
}

fn validate_comparison_compatibility(reference: &BenchmarkRun, run: &BenchmarkRun) -> Result<()> {
    let left = &reference.metadata;
    let right = &run.metadata;
    ensure!(
        left.evaluator_id == right.evaluator_id
            && left.evaluator_version == right.evaluator_version
            && left.evaluator_arguments == right.evaluator_arguments
            && left.paired_seeds == right.paired_seeds
            && left.order_seed == right.order_seed
            && left.fixed_case_order == right.fixed_case_order
            && same_f64(
                left.gpu_hours_per_evaluation,
                right.gpu_hours_per_evaluation
            )
            && same_f64(
                left.gpu_hour_budget_per_target,
                right.gpu_hour_budget_per_target
            )
            && left.suite_ids == right.suite_ids,
        "comparison baseline `{}` was not evaluated with the fixed evaluator invocation, order, suites, seeds, and compute budget",
        right.baseline.id
    );
    ensure!(
        same_target_identity(&left.candidate, &right.candidate),
        "comparison run for baseline `{}` evaluates a different candidate",
        right.baseline.id
    );
    ensure!(
        reference.cases.len() == run.cases.len(),
        "comparison run case counts differ"
    );
    for (expected, actual) in reference.cases.iter().zip(&run.cases) {
        ensure!(
            expected.suite_id == actual.suite_id
                && expected.case_id == actual.case_id
                && expected.catalog_id == actual.catalog_id
                && expected.visibility == actual.visibility
                && expected.family == actual.family
                && expected.metric == actual.metric
                && expected.direction == actual.direction
                && expected.stable_anchor == actual.stable_anchor,
            "comparison run benchmark definitions differ at `{}`",
            expected.case_id
        );
        for (expected_pair, actual_pair) in expected.pairs.iter().zip(&actual.pairs) {
            ensure!(
                expected_pair.seed == actual_pair.seed
                    && expected_pair.example_order_seed == actual_pair.example_order_seed
                    && same_measurement(&expected_pair.candidate, &actual_pair.candidate),
                "candidate measurement is not deterministic for `{}` seed {}",
                expected.case_id,
                expected_pair.seed
            );
        }
    }
    Ok(())
}

fn same_measurement(left: &EvaluationMeasurement, right: &EvaluationMeasurement) -> bool {
    left.score.to_bits() == right.score.to_bits()
        && left.gpu_hours.to_bits() == right.gpu_hours.to_bits()
        && left.examples == right.examples
        && left.metrics.len() == right.metrics.len()
        && left.metrics.iter().all(|(name, value)| {
            right
                .metrics
                .get(name)
                .is_some_and(|other| value.to_bits() == other.to_bits())
        })
}

fn same_target_identity(left: &BenchmarkTarget, right: &BenchmarkTarget) -> bool {
    left.id == right.id
        && left.checkpoint_manifest_sha256 == right.checkpoint_manifest_sha256
        && same_f64(left.training_gpu_hours, right.training_gpu_hours)
        && left.parameters == right.parameters
        && left.routed_active_parameters == right.routed_active_parameters
        && left.stored_bytes == right.stored_bytes
        && same_representation(&left.representation, &right.representation)
}

/// Compare representations without their declaring paths, which legitimately
/// differ between two runs stored in different directories.
fn same_representation(
    left: &ModelRepresentationTarget,
    right: &ModelRepresentationTarget,
) -> bool {
    match (left, right) {
        (ModelRepresentationTarget::FullPrecision, ModelRepresentationTarget::FullPrecision) => {
            true
        }
        (
            ModelRepresentationTarget::Hquant {
                candidate_manifest_sha256: left,
                ..
            },
            ModelRepresentationTarget::Hquant {
                candidate_manifest_sha256: right,
                ..
            },
        ) => left == right,
        _ => false,
    }
}

/// Borda aggregation over every case/seed observation avoids mixing metric
/// scales.  Direction-adjusted scores determine ranks; baseline id provides a
/// deterministic tie-break for byte-identical measurements.
fn derive_strongest_baseline(runs: &[LoadedBenchmarkRun]) -> Result<String> {
    let mut points = BTreeMap::<&str, u128>::new();
    for run in runs {
        points.insert(run.run.metadata.baseline.id.as_str(), 0);
    }
    let reference = &runs[0].run;
    for case_index in 0..reference.cases.len() {
        for pair_index in 0..reference.cases[case_index].pairs.len() {
            let mut ranked = runs
                .iter()
                .map(|run| {
                    let case = &run.run.cases[case_index];
                    (
                        case.direction
                            .acceptance_score(case.pairs[pair_index].baseline.score),
                        run.run.metadata.baseline.id.as_str(),
                    )
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|left, right| {
                right.0.total_cmp(&left.0).then_with(|| left.1.cmp(right.1))
            });
            let count = ranked.len() as u128;
            for (rank, (_, id)) in ranked.into_iter().enumerate() {
                *points
                    .get_mut(id)
                    .expect("all baseline ids were registered") += count - rank as u128;
            }
        }
    }
    points
        .into_iter()
        .max_by(|(left_id, left_points), (right_id, right_points)| {
            left_points
                .cmp(right_points)
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(id, _)| id.to_owned())
        .context("promotion comparison set is empty")
}

#[derive(Clone, Debug)]
pub struct BenchmarkRunner {
    pub config: BenchmarkRunConfig,
}

impl BenchmarkRunner {
    pub fn run<E: BenchmarkEvaluator>(
        &self,
        suites: &[LoadedBenchmarkSuite],
        baseline: &BenchmarkTarget,
        candidate: &BenchmarkTarget,
        evaluator: &mut E,
    ) -> Result<BenchmarkRun> {
        self.config.validate()?;
        ensure!(!suites.is_empty(), "no benchmark suites were supplied");
        let baseline_model = baseline.resolve_model()?;
        let candidate_model = candidate.resolve_model()?;

        // Suite ids are sorted, while each suite's manifest order is retained.
        // Thus CLI argument order cannot silently alter an experiment.
        let mut suite_order = (0..suites.len()).collect::<Vec<_>>();
        suite_order.sort_by(|left, right| {
            suites[*left]
                .manifest
                .suite_id
                .cmp(&suites[*right].manifest.suite_id)
        });
        ensure!(
            suite_order.windows(2).all(|pair| {
                suites[pair[0]].manifest.suite_id != suites[pair[1]].manifest.suite_id
            }),
            "benchmark suite ids must be globally unique"
        );

        let mut global_ids = BTreeSet::new();
        let mut schedule = Vec::new();
        let mut suite_ids = BTreeSet::new();
        for suite_index in suite_order {
            let suite = &suites[suite_index];
            suite.manifest.validate()?;
            suite_ids.insert(suite.manifest.suite_id.clone());
            for case_index in 0..suite.manifest.cases.len() {
                let case = &suite.manifest.cases[case_index];
                ensure!(
                    global_ids.insert(case.id.as_str()),
                    "benchmark case id `{}` occurs in multiple suites",
                    case.id
                );
                schedule.push((suite, case_index));
            }
        }

        let fixed_case_order = schedule
            .iter()
            .map(|(suite, case_index)| suite.manifest.cases[*case_index].id.clone())
            .collect::<Vec<_>>();

        let mut raw_cases = Vec::with_capacity(schedule.len());
        let mut pair_ordinal = 0usize;
        for (suite, case_index) in schedule {
            let case = &suite.manifest.cases[case_index];
            let artifact = suite.artifact(&case.id).with_context(|| {
                format!("verified artifact for benchmark `{}` disappeared", case.id)
            })?;
            let mut pairs = Vec::with_capacity(self.config.paired_seeds.len());
            for &seed in &self.config.paired_seeds {
                let example_order_seed = derive_order_seed(
                    self.config.order_seed,
                    &suite.manifest.suite_id,
                    &case.id,
                    seed,
                );
                let baseline_request = EvaluationRequest {
                    suite_id: &suite.manifest.suite_id,
                    visibility: suite.manifest.visibility,
                    case,
                    artifact,
                    target: baseline,
                    model: &baseline_model,
                    role: TargetRole::Baseline,
                    model_seed: seed,
                    example_order_seed,
                    pair_ordinal,
                    max_gpu_hours: self.config.gpu_hours_per_evaluation,
                };
                let candidate_request = EvaluationRequest {
                    target: candidate,
                    model: &candidate_model,
                    role: TargetRole::Candidate,
                    ..baseline_request
                };
                // Alternate pair order to counterbalance persistent-worker
                // warm-cache, thermal, and allocator effects. Both requests
                // retain the same pair ordinal, model seed, example order,
                // and hard budget; only which role executes first changes.
                let (baseline_measurement, candidate_measurement) = if pair_ordinal
                    .is_multiple_of(2)
                {
                    let baseline = evaluator.evaluate(&baseline_request).with_context(|| {
                        format!("baseline evaluation failed for `{}` seed {seed}", case.id)
                    })?;
                    baseline.validate(&case.id, self.config.gpu_hours_per_evaluation)?;
                    let candidate = evaluator.evaluate(&candidate_request).with_context(|| {
                        format!("candidate evaluation failed for `{}` seed {seed}", case.id)
                    })?;
                    candidate.validate(&case.id, self.config.gpu_hours_per_evaluation)?;
                    (baseline, candidate)
                } else {
                    let candidate = evaluator.evaluate(&candidate_request).with_context(|| {
                        format!("candidate evaluation failed for `{}` seed {seed}", case.id)
                    })?;
                    candidate.validate(&case.id, self.config.gpu_hours_per_evaluation)?;
                    let baseline = evaluator.evaluate(&baseline_request).with_context(|| {
                        format!("baseline evaluation failed for `{}` seed {seed}", case.id)
                    })?;
                    baseline.validate(&case.id, self.config.gpu_hours_per_evaluation)?;
                    (baseline, candidate)
                };

                pairs.push(RawPairedResult {
                    seed,
                    example_order_seed,
                    baseline: baseline_measurement,
                    candidate: candidate_measurement,
                });
                pair_ordinal += 1;
            }
            raw_cases.push(RawCaseResult {
                suite_id: suite.manifest.suite_id.clone(),
                case_id: case.id.clone(),
                catalog_id: case.catalog_id().to_owned(),
                visibility: suite.manifest.visibility,
                family: case.family.clone(),
                metric: case.metric.clone(),
                direction: case.direction,
                stable_anchor: case.stable_anchor,
                pairs,
            });
        }

        let evaluation_count = raw_cases.len() * self.config.paired_seeds.len();
        let run = BenchmarkRun {
            metadata: BenchmarkRunMetadata {
                evaluator_id: self.config.evaluator_id.clone(),
                evaluator_version: self.config.evaluator_version.clone(),
                evaluator_arguments: self.config.evaluator_arguments.clone(),
                paired_seeds: self.config.paired_seeds.clone(),
                order_seed: self.config.order_seed,
                fixed_case_order,
                gpu_hours_per_evaluation: self.config.gpu_hours_per_evaluation,
                gpu_hour_budget_per_target: self.config.gpu_hours_per_evaluation
                    * evaluation_count as f64,
                baseline: baseline.clone(),
                candidate: candidate.clone(),
                suite_ids,
                ablation: self.config.ablation,
            },
            cases: raw_cases,
        };
        run.validate()?;
        Ok(run)
    }
}

fn derive_order_seed(base: u64, suite_id: &str, case_id: &str, model_seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(base.to_le_bytes());
    hasher.update((suite_id.len() as u64).to_le_bytes());
    hasher.update(suite_id.as_bytes());
    hasher.update((case_id.len() as u64).to_le_bytes());
    hasher.update(case_id.as_bytes());
    hasher.update(model_seed.to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

fn same_f64(left: f64, right: f64) -> bool {
    (left - right).abs() <= f64::EPSILON * left.abs().max(right.abs()).max(1.0) * 8.0
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::artifact_io::sha256_hex;

    fn evaluator_sha256() -> String {
        format!("sha256:{}", "e".repeat(64))
    }

    struct MockEvaluator {
        calls: Vec<(String, u64, u64, TargetRole)>,
    }

    impl BenchmarkEvaluator for MockEvaluator {
        fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
            self.calls.push((
                request.case.id.clone(),
                request.model_seed,
                request.example_order_seed,
                request.role,
            ));
            let candidate = request.role == TargetRole::Candidate;
            let score = match request.case.direction {
                MetricDirection::Maximize => 0.5 + if candidate { 0.1 } else { 0.0 },
                MetricDirection::Minimize => 1.0 - if candidate { 0.1 } else { 0.0 },
            };
            Ok(EvaluationMeasurement {
                score,
                gpu_hours: request.max_gpu_hours / 2.0,
                examples: 8,
                metrics: BTreeMap::new(),
            })
        }
    }

    fn write_suite(
        root: &Path,
        suite_id: &str,
        visibility: SuiteVisibility,
        case_id: &str,
        direction: MetricDirection,
    ) -> LoadedBenchmarkSuite {
        let artifact_path = root.join(format!("{case_id}.jsonl"));
        fs::write(&artifact_path, b"{\"input\":\"fixture\"}\n").unwrap();
        let manifest = BenchmarkSuiteManifest {
            version: BENCHMARK_MANIFEST_VERSION,
            suite_id: suite_id.into(),
            visibility,
            cases: vec![BenchmarkSpec {
                id: case_id.into(),
                catalog_id: None,
                family: BenchmarkFamily::Reasoning,
                metric: "score".into(),
                direction,
                stable_anchor: false,
                artifact: BenchmarkArtifact {
                    path: artifact_path.file_name().unwrap().into(),
                },
                evaluator: BTreeMap::new(),
            }],
        };
        let manifest_path = root.join(format!("{suite_id}.json"));
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        LoadedBenchmarkSuite::load(&manifest_path).unwrap()
    }

    fn target(root: &Path, id: &str, parameters: u64) -> BenchmarkTarget {
        use safetensors::{Dtype, tensor::TensorView};

        let output = root.join(id);
        let staging = output.join("generations").join("staging");
        fs::create_dir_all(&staging).unwrap();
        let salt = id.bytes().enumerate().fold(0_u64, |sum, (index, byte)| {
            sum.wrapping_add((index as u64 + 1).wrapping_mul(u64::from(byte)))
        }) as f32;
        let raw_weights = (0..parameters)
            .map(|index| ((index as f32 + salt) * 0.03125).sin())
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![1, parameters as usize], &raw_weights).unwrap();
        let weights = safetensors::tensor::serialize([("weight", view)], None).unwrap();
        let weights_sha256 = sha256_hex(&weights);
        fs::write(staging.join(CHECKPOINT_WEIGHTS_FILE), &weights).unwrap();
        let accounting = TrainingAccounting {
            version: TRAINING_ACCOUNTING_VERSION,
            training_gpu_hours: 8.0,
            parameters,
            routed_active_parameters: 80,
            weights_bytes: weights.len() as u64,
            weights_sha256: weights_sha256.clone(),
        };
        let accounting_bytes = serde_json::to_vec(&accounting).unwrap();
        let accounting_sha256 = sha256_hex(&accounting_bytes);
        fs::write(staging.join(TRAINING_ACCOUNTING_FILE), &accounting_bytes).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "training_state_version": 2,
            "global_step": 20,
            "phase": 0,
            "phase_id": "test",
            "files": [
                {
                    "path": TRAINING_ACCOUNTING_FILE,
                    "bytes": accounting_bytes.len(),
                    "sha256": accounting_sha256,
                },
                {
                    "path": CHECKPOINT_WEIGHTS_FILE,
                    "bytes": weights.len(),
                    "sha256": weights_sha256,
                },
            ],
        });
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let checkpoint_manifest_sha256 = sha256_hex(&manifest_bytes);
        let generation = output
            .join("generations")
            .join(format!("sha256-{checkpoint_manifest_sha256}"));
        fs::rename(&staging, &generation).unwrap();
        let path = generation.join("generation-manifest.json");
        fs::write(&path, &manifest_bytes).unwrap();
        BenchmarkTarget {
            id: id.into(),
            checkpoint_manifest_sha256,
            training_gpu_hours: 8.0,
            checkpoint_manifest: path,
            parameters,
            routed_active_parameters: 80,
            stored_bytes: weights.len() as u64,
            representation: ModelRepresentationTarget::FullPrecision,
        }
    }

    fn attach_hquant_candidate(root: &Path, mut target: BenchmarkTarget) -> BenchmarkTarget {
        use crate::qat_candidate::{QatArchiveMetrics, QatCandidateManifest, open_qat_candidate};
        use crate::quantization::{
            BONSAI_GROUP_SIZE, QuantizationRecipe, QuantizedArchive, UltraQuantFormat,
            export_safetensors_archive,
        };

        let source = target
            .checkpoint_manifest
            .parent()
            .unwrap()
            .join(CHECKPOINT_WEIGHTS_FILE);
        let candidate_key = format!("{}-hquant", target.id);
        let candidate_root = root.join(&candidate_key);
        fs::create_dir(&candidate_root).unwrap();
        let candidate_weights = candidate_root.join("weights.safetensors");
        fs::copy(&source, &candidate_weights).unwrap();
        let recipe = QuantizationRecipe {
            format: UltraQuantFormat::BinaryG128,
            group_size: BONSAI_GROUP_SIZE,
            fake_quant_start_step: 0,
            ternary_warmup_steps: 0,
            distillation_weight: 0.0,
            quantize_embeddings: true,
            quantize_lm_head: true,
        };
        let archive_path = candidate_root.join("hquant");
        let archive_manifest =
            export_safetensors_archive(&candidate_weights, &archive_path, &recipe).unwrap();
        let quantized_elements = archive_manifest
            .matrices
            .iter()
            .map(|matrix| matrix.elements)
            .sum::<u64>();
        let packed_bytes = archive_manifest
            .matrices
            .iter()
            .map(|matrix| matrix.packed_bytes)
            .sum::<u64>();
        let floating_elements = archive_manifest
            .floating_tensors
            .iter()
            .map(|tensor| tensor.elements)
            .sum::<u64>();
        let floating_bytes = archive_manifest
            .floating_tensors
            .iter()
            .map(|tensor| tensor.bytes)
            .sum::<u64>();
        let weighted_mean_squared_error = archive_manifest
            .matrices
            .iter()
            .map(|matrix| matrix.mean_squared_error * matrix.elements as f64)
            .sum::<f64>()
            / quantized_elements as f64;
        let maximum_absolute_error = archive_manifest
            .matrices
            .iter()
            .map(|matrix| f64::from(matrix.maximum_absolute_error))
            .fold(0.0_f64, f64::max);
        let metrics = QatArchiveMetrics {
            quantized_tensors: archive_manifest.matrices.len() as u64,
            quantized_elements,
            packed_bytes,
            floating_tensors: archive_manifest.floating_tensors.len() as u64,
            floating_elements,
            floating_bytes,
            archive_weight_elements: quantized_elements + floating_elements,
            archive_weight_bytes: packed_bytes + floating_bytes,
            average_bits_per_weight: archive_manifest.true_average_bits_per_weight().unwrap(),
            weighted_mean_squared_error,
            maximum_absolute_error,
        };
        let archive = QuantizedArchive::open(&archive_path).unwrap();
        let archive_manifest_path = archive_path.join("manifest.json");
        let manifest = QatCandidateManifest {
            version: 1,
            candidate_key,
            weights_file: "weights.safetensors".into(),
            weights_bytes: fs::metadata(&candidate_weights).unwrap().len(),
            weights_sha256: format!(
                "sha256:{}",
                sha256_hex(&fs::read(&candidate_weights).unwrap())
            ),
            archive_directory: "hquant".into(),
            archive_manifest: "hquant/manifest.json".into(),
            archive_manifest_sha256: format!(
                "sha256:{}",
                sha256_hex(&fs::read(&archive_manifest_path).unwrap())
            ),
            archive_content_sha256: archive.content_hash().unwrap(),
            recipe,
            metrics: metrics.clone(),
        };
        let candidate_manifest = candidate_root.join("candidate.json");
        let candidate_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(&candidate_manifest, &candidate_bytes).unwrap();
        let publication = open_qat_candidate(&candidate_root).unwrap();
        target.stored_bytes = metrics.archive_weight_bytes;
        target.representation = ModelRepresentationTarget::Hquant {
            candidate_manifest,
            candidate_manifest_sha256: publication.candidate_manifest_sha256,
        };
        target
    }

    fn write_catalog_suite(
        root: &Path,
        suite_id: &str,
        visibility: SuiteVisibility,
    ) -> LoadedBenchmarkSuite {
        let prefix = match visibility {
            SuiteVisibility::Public => "public",
            SuiteVisibility::Sealed => "sealed",
        };
        let stable_anchors = AcceptancePolicy::default().stable_anchor_catalog_ids;
        let cases = required_catalog()
            .into_iter()
            .filter(|entry| entry.stage.required_now())
            .map(|entry| {
                let case_id = format!("{prefix}-{}", entry.id);
                let artifact_path = root.join(format!("{case_id}.jsonl"));
                fs::write(&artifact_path, format!("{{\"id\":\"{case_id}\"}}\n")).unwrap();
                BenchmarkSpec {
                    id: case_id,
                    catalog_id: Some(entry.id.into()),
                    family: entry.family,
                    metric: "score".into(),
                    direction: MetricDirection::Maximize,
                    stable_anchor: stable_anchors.contains(entry.id),
                    artifact: BenchmarkArtifact {
                        path: artifact_path.file_name().unwrap().into(),
                    },
                    evaluator: BTreeMap::new(),
                }
            })
            .collect();
        let manifest = BenchmarkSuiteManifest {
            version: BENCHMARK_MANIFEST_VERSION,
            suite_id: suite_id.into(),
            visibility,
            cases,
        };
        let path = root.join(format!("{suite_id}.json"));
        fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        LoadedBenchmarkSuite::load(path).unwrap()
    }

    struct RankingEvaluator;

    impl BenchmarkEvaluator for RankingEvaluator {
        fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
            let score = match request.role {
                TargetRole::Candidate => 0.9,
                TargetRole::Baseline => {
                    let rank = request
                        .target
                        .id
                        .strip_prefix("baseline-")
                        .context("unexpected baseline fixture id")?
                        .parse::<u64>()?;
                    0.4 + rank as f64 / 100.0
                }
            };
            Ok(EvaluationMeasurement {
                score,
                gpu_hours: request.max_gpu_hours,
                examples: 128,
                metrics: BTreeMap::new(),
            })
        }
    }

    fn write_run(root: &Path, index: usize, run: &BenchmarkRun) -> LoadedBenchmarkRun {
        let path = root.join(format!("run-{index:02}.json"));
        fs::write(&path, serde_json::to_vec_pretty(run).unwrap()).unwrap();
        LoadedBenchmarkRun::load(path).unwrap()
    }

    fn resource_comparison(run: &LoadedBenchmarkRun) -> ResourceComparison {
        use crate::metrics::{
            MetricContext, MetricEvent, MetricPhase, MetricPhaseKind, MetricWriter,
            ThroughputMetrics,
        };

        let final_manifest = &run.run.metadata.candidate.checkpoint_manifest;
        let final_generation = final_manifest.parent().unwrap();
        let evidence_vault = final_generation
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let exact_root = final_generation
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("exact-resume");
        let resumed_generation = exact_root
            .join("resumed")
            .join("generations")
            .join(final_generation.file_name().unwrap());
        fs::create_dir_all(&resumed_generation).unwrap();
        for entry in fs::read_dir(final_generation).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), resumed_generation.join(entry.file_name())).unwrap();
        }
        let resumed_manifest = resumed_generation.join("generation-manifest.json");

        let final_bytes = fs::read(final_manifest).unwrap();
        let mut interrupted: serde_json::Value = serde_json::from_slice(&final_bytes).unwrap();
        interrupted["global_step"] = serde_json::json!(10);
        let interrupted_bytes = serde_json::to_vec(&interrupted).unwrap();
        let interrupted_generation = exact_root
            .join("interrupted")
            .join("generations")
            .join(format!("sha256-{}", sha256_hex(&interrupted_bytes)));
        fs::create_dir_all(&interrupted_generation).unwrap();
        let interrupted_manifest = interrupted_generation.join("generation-manifest.json");
        fs::write(&interrupted_manifest, &interrupted_bytes).unwrap();

        let write_metrics = |directory: &Path, run_id: &str, elapsed: f64, timestamp: u64| {
            fs::create_dir_all(directory).unwrap();
            let path = directory.join("metrics.jsonl");
            let mut writer = MetricWriter::create(&path, run_id).unwrap();
            writer
                .append_at(
                    MetricContext {
                        global_step: 20,
                        phase: MetricPhase {
                            index: 0,
                            name: "test".into(),
                            kind: MetricPhaseKind::Pretrain,
                        },
                        checkpoint_hash: None,
                    },
                    MetricEvent::Throughput(ThroughputMetrics {
                        optimizer_steps: 20,
                        compute_tokens: 200,
                        supervised_tokens: 200,
                        examples: 20,
                        elapsed_seconds: elapsed,
                        tokens_per_second: 200.0 / elapsed,
                        examples_per_second: 20.0 / elapsed,
                        input_wait_seconds: 0.0,
                        host_to_device_seconds: 0.0,
                        gpu_busy_seconds: elapsed,
                    }),
                    timestamp,
                )
                .unwrap();
            writer.sync_all().unwrap();
            drop(writer);
            path
        };
        let uninterrupted_metrics =
            write_metrics(&exact_root.join("uninterrupted"), "uninterrupted", 2.0, 100);
        let resumed_metrics = write_metrics(&exact_root.join("resumed"), "resumed", 3.0, 200);

        let relative = |path: &Path| path.strip_prefix(evidence_vault).unwrap().to_path_buf();
        ResourceComparison {
            version: crate::acceptance::RESOURCE_COMPARISON_VERSION,
            baseline_id: run.run.metadata.baseline.id.clone(),
            candidate_id: run.run.metadata.candidate.id.clone(),
            strongest_baseline_id: run.run.metadata.baseline.id.clone(),
            measurement_evaluator_id: "summa-resource-evaluator".into(),
            wake_trials: (0..3)
                .map(|trial| crate::acceptance::PairedWakeTrial {
                    trial,
                    baseline: crate::acceptance::WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 10.0,
                        request_latency_ms: vec![10.0],
                    },
                    candidate: crate::acceptance::WakeMeasurement {
                        tokens: 1_000,
                        elapsed_seconds: 1_000.0 / 96.0,
                        request_latency_ms: vec![10.4],
                    },
                })
                .collect(),
            candidate_capacity: (0..=4)
                .map(
                    |completed_sleep_cycles| crate::acceptance::CapacityObservation {
                        completed_sleep_cycles,
                        routed_active_parameters: run
                            .run
                            .metadata
                            .candidate
                            .routed_active_parameters,
                        stored_parameters: run.run.metadata.candidate.parameters,
                        stored_bytes: run.run.metadata.candidate.stored_bytes,
                    },
                )
                .collect(),
            grouped_mm_parity: crate::acceptance::KernelParityEvidence {
                samples: (0..1024)
                    .map(|_| crate::acceptance::KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0 + 1e-5,
                    })
                    .collect(),
            },
            pytorch_parity: crate::acceptance::KernelParityEvidence {
                samples: (0..1024)
                    .map(|_| crate::acceptance::KernelParitySample {
                        reference: 1.0,
                        candidate: 1.0 + 1e-5,
                    })
                    .collect(),
            },
            exact_resume: crate::acceptance::ExactResumeEvidence {
                interrupted_checkpoint: crate::acceptance::ExactResumeArtifact {
                    path: relative(&interrupted_manifest),
                },
                uninterrupted_final_state: crate::acceptance::ExactResumeArtifact {
                    path: relative(final_manifest),
                },
                resumed_final_state: crate::acceptance::ExactResumeArtifact {
                    path: relative(&resumed_manifest),
                },
                uninterrupted_metrics: crate::acceptance::ExactResumeArtifact {
                    path: relative(&uninterrupted_metrics),
                },
                resumed_metrics: crate::acceptance::ExactResumeArtifact {
                    path: relative(&resumed_metrics),
                },
                interruption_step: 10,
                resumed_from_step: 10,
            },
        }
    }

    fn write_resources(root: &Path, comparison: &ResourceComparison) -> PathBuf {
        let path = root.join("resources.json");
        fs::write(&path, serde_json::to_vec_pretty(comparison).unwrap()).unwrap();
        path
    }

    #[test]
    fn verified_runner_is_paired_ordered_and_acceptance_ready() {
        let temporary = tempfile::tempdir().unwrap();
        // Supply reverse lexical order to ensure suite order is canonical.
        let sealed = write_suite(
            temporary.path(),
            "z-sealed",
            SuiteVisibility::Sealed,
            "sealed-loss",
            MetricDirection::Minimize,
        );
        let public = write_suite(
            temporary.path(),
            "a-public",
            SuiteVisibility::Public,
            "public-score",
            MetricDirection::Maximize,
        );
        let baseline = target(temporary.path(), "baseline", 100);
        let candidate = target(temporary.path(), "candidate", 100);
        let runner = BenchmarkRunner {
            config: BenchmarkRunConfig {
                evaluator_id: "mock".into(),
                evaluator_version: evaluator_sha256(),
                evaluator_arguments: vec!["--profile=fixed".into()],
                paired_seeds: vec![11, 22, 33],
                order_seed: 7,
                gpu_hours_per_evaluation: 0.25,
                ablation: Some(AblationId::WakeOnly),
            },
        };
        let mut evaluator = MockEvaluator { calls: Vec::new() };
        let run = runner
            .run(&[sealed, public], &baseline, &candidate, &mut evaluator)
            .unwrap();

        assert_eq!(
            run.metadata.fixed_case_order,
            ["public-score", "sealed-loss"]
        );
        assert_eq!(run.metadata.evaluator_arguments, ["--profile=fixed"]);
        assert!(run.capacity_matched());
        assert_eq!(run.metadata.gpu_hour_budget_per_target, 1.5);
        let mut differently_invoked = run.clone();
        differently_invoked
            .metadata
            .evaluator_arguments
            .push("--different".into());
        assert!(validate_comparison_compatibility(&run, &differently_invoked).is_err());
        assert_eq!(evaluator.calls.len(), 12);
        for (pair_ordinal, calls) in evaluator.calls.chunks_exact(2).enumerate() {
            assert_eq!(calls[0].0, calls[1].0);
            assert_eq!(calls[0].1, calls[1].1);
            assert_eq!(calls[0].2, calls[1].2);
            let expected_roles = if pair_ordinal.is_multiple_of(2) {
                [TargetRole::Baseline, TargetRole::Candidate]
            } else {
                [TargetRole::Candidate, TargetRole::Baseline]
            };
            assert_eq!([calls[0].3, calls[1].3], expected_roles);
        }

        let (suite, results) = run.acceptance_inputs();
        suite.validate().unwrap();
        assert_eq!(results.len(), 2);
        for result in results {
            assert_eq!(result.pairs.len(), 3);
            for pair in result.pairs {
                assert!((pair.candidate - pair.baseline - 0.1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn benchmark_json_is_bounded_before_deserialization() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("oversized-run.json");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_BENCHMARK_JSON_BYTES + 1).unwrap();
        drop(file);

        let error = format!("{:#}", LoadedBenchmarkRun::load(&path).unwrap_err());
        assert!(error.contains("byte limit"), "{error}");
    }

    #[test]
    fn hquant_target_resolves_its_candidate_archive_and_storage() {
        let temporary = tempfile::tempdir().unwrap();
        let full_precision = target(temporary.path(), "candidate", 100);
        let full_precision_bytes = full_precision.stored_bytes;
        let candidate = attach_hquant_candidate(temporary.path(), full_precision);
        assert!(candidate.stored_bytes < full_precision_bytes);
        let ResolvedModelRepresentation::Hquant {
            candidate_manifest,
            archive,
            stored_bytes,
            ..
        } = candidate.resolve_model().unwrap()
        else {
            panic!("expected a resolved HQUANT representation")
        };
        assert_eq!(candidate_manifest.file_name().unwrap(), "candidate.json");
        assert_eq!(archive.file_name().unwrap(), "hquant");
        assert_eq!(stored_bytes, candidate.stored_bytes);

        let mut wrong_storage = candidate.clone();
        wrong_storage.stored_bytes += 1;
        let error = wrong_storage.resolve_model().unwrap_err().to_string();
        assert!(error.contains("validated archive members"), "{error}");
    }

    struct RepresentationEvaluator {
        hquant_candidate_calls: usize,
    }

    impl BenchmarkEvaluator for RepresentationEvaluator {
        fn evaluate(&mut self, request: &EvaluationRequest<'_>) -> Result<EvaluationMeasurement> {
            match request.role {
                TargetRole::Baseline => ensure!(
                    matches!(
                        request.model,
                        ResolvedModelRepresentation::FullPrecision { .. }
                    ),
                    "baseline did not receive FP weights"
                ),
                TargetRole::Candidate => {
                    let ResolvedModelRepresentation::Hquant {
                        archive,
                        archive_manifest,
                        ..
                    } = request.model
                    else {
                        anyhow::bail!("candidate did not receive HQUANT archive")
                    };
                    ensure!(archive.is_dir(), "candidate HQUANT archive is unavailable");
                    ensure!(
                        archive_manifest.is_file(),
                        "candidate HQUANT manifest is unavailable"
                    );
                    self.hquant_candidate_calls += 1;
                }
            }
            Ok(EvaluationMeasurement {
                score: 0.5,
                gpu_hours: request.max_gpu_hours,
                examples: 1,
                metrics: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn runner_propagates_verified_hquant_archive_to_evaluator() {
        let temporary = tempfile::tempdir().unwrap();
        let suite = write_suite(
            temporary.path(),
            "public-hquant",
            SuiteVisibility::Public,
            "quality",
            MetricDirection::Maximize,
        );
        let baseline = target(temporary.path(), "baseline-hquant", 100);
        let candidate = attach_hquant_candidate(
            temporary.path(),
            target(temporary.path(), "candidate-hquant", 100),
        );
        let runner = BenchmarkRunner {
            config: BenchmarkRunConfig {
                evaluator_id: "hquant-backend".into(),
                evaluator_version: evaluator_sha256(),
                evaluator_arguments: Vec::new(),
                paired_seeds: vec![1, 2, 3],
                order_seed: 9,
                gpu_hours_per_evaluation: 0.25,
                ablation: None,
            },
        };
        let mut evaluator = RepresentationEvaluator {
            hquant_candidate_calls: 0,
        };
        let run = runner
            .run(&[suite], &baseline, &candidate, &mut evaluator)
            .unwrap();
        assert_eq!(evaluator.hquant_candidate_calls, 3);
        assert!(matches!(
            run.metadata.candidate.representation,
            ModelRepresentationTarget::Hquant { .. }
        ));
        assert!(!same_target_identity(&baseline, &candidate));
    }

    #[test]
    fn exact_resume_gate_recomputes_semantic_progress_from_metric_artifacts() {
        use crate::metrics::{
            MetricContext, MetricEvent, MetricPhase, MetricPhaseKind, MetricWriter,
            ThroughputMetrics,
        };

        let temporary = tempfile::tempdir().unwrap();
        let public = write_catalog_suite(temporary.path(), "public", SuiteVisibility::Public);
        let sealed = write_catalog_suite(temporary.path(), "sealed", SuiteVisibility::Sealed);
        let baseline = target(temporary.path(), "baseline-0", 100);
        let candidate = target(temporary.path(), "candidate", 100);
        let run = BenchmarkRunner {
            config: BenchmarkRunConfig {
                evaluator_id: "fixed".into(),
                evaluator_version: evaluator_sha256(),
                evaluator_arguments: Vec::new(),
                paired_seeds: vec![1, 2, 3],
                order_seed: 7,
                gpu_hours_per_evaluation: 0.25,
                ablation: Some(AblationId::WakeOnly),
            },
        }
        .run(
            &[public, sealed],
            &baseline,
            &candidate,
            &mut RankingEvaluator,
        )
        .unwrap();
        let run = write_run(temporary.path(), 99, &run);
        let comparison = resource_comparison(&run);
        let resource_path = temporary.path().join("resources.json");
        // The honest evidence passes: two independently written journals whose
        // wall-clock timings differ still recompute the same semantic progress.
        verify_exact_resume(&comparison.exact_resume, &resource_path).unwrap();

        let path = temporary
            .path()
            .join(&comparison.exact_resume.resumed_metrics.path);
        let mut writer = MetricWriter::create(&path, "resumed-tampered").unwrap();
        writer
            .append_at(
                MetricContext {
                    global_step: 20,
                    phase: MetricPhase {
                        index: 0,
                        name: "test".into(),
                        kind: MetricPhaseKind::Pretrain,
                    },
                    checkpoint_hash: None,
                },
                MetricEvent::Throughput(ThroughputMetrics {
                    optimizer_steps: 20,
                    compute_tokens: 201,
                    supervised_tokens: 201,
                    examples: 20,
                    elapsed_seconds: 3.0,
                    tokens_per_second: 67.0,
                    examples_per_second: 20.0 / 3.0,
                    input_wait_seconds: 0.0,
                    host_to_device_seconds: 0.0,
                    gpu_busy_seconds: 3.0,
                }),
                300,
            )
            .unwrap();
        writer.sync_all().unwrap();
        drop(writer);
        let error = verify_exact_resume(&comparison.exact_resume, &resource_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("differ semantically"), "{error}");

        // Diverging final state is caught even when both journals agree.
        let honest = resource_comparison(&write_run(temporary.path(), 98, run.run()));
        let resumed = temporary
            .path()
            .join(&honest.exact_resume.resumed_final_state.path);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&resumed).unwrap()).unwrap();
        manifest["files"][0]["sha256"] = serde_json::json!("0".repeat(64));
        fs::write(&resumed, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let error = verify_exact_resume(&honest.exact_resume, &resource_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("byte-identical final state"), "{error}");
    }

    #[test]
    fn target_paths_resolve_against_their_manifest() {
        let mut target = BenchmarkTarget {
            id: "candidate".into(),
            checkpoint_manifest: "artifacts/generation-manifest.json".into(),
            checkpoint_manifest_sha256: "1".repeat(64),
            training_gpu_hours: 1.0,
            parameters: 10,
            routed_active_parameters: 8,
            stored_bytes: 100,
            representation: ModelRepresentationTarget::Hquant {
                candidate_manifest: "qat/candidate.json".into(),
                candidate_manifest_sha256: format!("sha256:{}", "3".repeat(64)),
            },
        };
        target.resolve_paths(Path::new("/runs/target.json"));
        assert_eq!(
            target.checkpoint_manifest,
            Path::new("/runs/artifacts/generation-manifest.json")
        );
        let ModelRepresentationTarget::Hquant {
            candidate_manifest, ..
        } = target.representation
        else {
            panic!("expected HQUANT target")
        };
        assert_eq!(candidate_manifest, Path::new("/runs/qat/candidate.json"));
    }

    #[test]
    fn target_schema_requires_an_explicit_model_representation() {
        let temporary = tempfile::tempdir().unwrap();
        let target = target(temporary.path(), "schema-target", 100);
        let mut value = serde_json::to_value(&target).unwrap();
        value.as_object_mut().unwrap().remove("representation");
        assert!(serde_json::from_value::<BenchmarkTarget>(value).is_err());

        let mut hquant = target.clone();
        hquant.representation = ModelRepresentationTarget::Hquant {
            candidate_manifest: temporary.path().join("candidate.json"),
            candidate_manifest_sha256: format!("sha256:{}", "a".repeat(64)),
        };
        assert!(!same_target_identity(&target, &hquant));
    }

    #[test]
    fn runner_rejects_fewer_than_three_or_reordered_seeds() {
        let config = BenchmarkRunConfig {
            evaluator_id: "mock".into(),
            evaluator_version: evaluator_sha256(),
            evaluator_arguments: Vec::new(),
            paired_seeds: vec![2, 1, 3],
            order_seed: 0,
            gpu_hours_per_evaluation: 1.0,
            ablation: None,
        };
        assert!(config.validate().is_err());

        let config = BenchmarkRunConfig {
            paired_seeds: vec![1, 2],
            ..config
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn benchmark_configuration_bounds_and_binds_evaluator_arguments() {
        let mut config = BenchmarkRunConfig {
            evaluator_id: "mock".into(),
            evaluator_version: evaluator_sha256(),
            evaluator_arguments: vec!["--profile=h100".into()],
            paired_seeds: vec![1, 2, 3],
            order_seed: 0,
            gpu_hours_per_evaluation: 1.0,
            ablation: None,
        };
        config.validate().unwrap();

        config.evaluator_arguments = vec!["x".repeat(4_097)];
        assert!(config.validate().is_err());
        config.evaluator_arguments = vec!["x\0y".into()];
        assert!(config.validate().is_err());
        config.evaluator_arguments.clear();
        config.evaluator_version = "latest".into();
        assert!(config.validate().is_err());

        config.evaluator_version = evaluator_sha256();
        config.paired_seeds = (0..=MAXIMUM_PAIRED_SEEDS as u64).collect();
        assert!(config.validate().is_err());
    }

    #[test]
    fn catalog_and_ablation_ids_are_complete_and_stable() {
        let catalog = required_catalog();
        let ids = catalog
            .iter()
            .map(|entry| entry.id)
            .collect::<BTreeSet<_>>();
        for id in [
            "pretraining_causal",
            "summarization",
            "retrieval_representation_ranking",
            "retrieval_planning",
            "reasoning_qa",
            "pairwise_preference",
            "verifiable_rl",
            "synthetic_retention_smoke",
            "clinc_continual_learning",
            "banking77_continual_learning",
            "mk_niah",
            "qasper",
            "squad_no_context",
            "arc_dreaming",
            "manchu_continual_learning",
            "kalamang_continual_learning",
            "babilong",
        ] {
            assert!(ids.contains(id));
        }
        assert_eq!(AblationId::ALL.len(), 11);
        assert_eq!(
            AblationId::CmsWithoutKnowledgeSeeding.as_str(),
            "cms_without_knowledge_seeding"
        );
    }

    #[test]
    fn promotion_is_derived_from_content_addressed_complete_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let public =
            write_catalog_suite(temporary.path(), "public-catalog", SuiteVisibility::Public);
        let sealed =
            write_catalog_suite(temporary.path(), "sealed-catalog", SuiteVisibility::Sealed);
        validate_catalog_coverage(&[public.clone(), sealed.clone()], false).unwrap();

        let candidate = target(temporary.path(), "candidate", 100);
        let mut runs = Vec::new();
        for (index, ablation) in AblationId::ALL.into_iter().enumerate() {
            let baseline = target(temporary.path(), &format!("baseline-{index:02}"), 100);
            let runner = BenchmarkRunner {
                config: BenchmarkRunConfig {
                    evaluator_id: "fixed-evaluator".into(),
                    evaluator_version: evaluator_sha256(),
                    evaluator_arguments: Vec::new(),
                    paired_seeds: vec![11, 22, 33],
                    order_seed: 7,
                    gpu_hours_per_evaluation: 0.25,
                    ablation: Some(ablation),
                },
            };
            let run = runner
                .run(
                    &[public.clone(), sealed.clone()],
                    &baseline,
                    &candidate,
                    &mut RankingEvaluator,
                )
                .unwrap();
            runs.push(write_run(temporary.path(), index, &run));
        }

        // RankingEvaluator monotonically increases baseline strength.
        let selected = runs.last().unwrap();
        let policy = AcceptancePolicy::default();
        let comparison = resource_comparison(selected);
        let resources_path = write_resources(temporary.path(), &comparison);
        assert!(
            [
                &comparison.exact_resume.interrupted_checkpoint,
                &comparison.exact_resume.uninterrupted_final_state,
                &comparison.exact_resume.resumed_final_state,
                &comparison.exact_resume.uninterrupted_metrics,
                &comparison.exact_resume.resumed_metrics,
            ]
            .iter()
            .all(|artifact| artifact.path.is_relative()),
            "resource evidence must retain portable relative artifact paths"
        );
        let report =
            evaluate_verified_promotion(selected, &runs, &comparison, &resources_path, &policy)
                .unwrap();
        assert!(report.accepted);
        assert!(report.resource_gates["exact_resume"]);
        assert!(report.resource_gates["strongest_matched_baseline"]);
        assert_eq!(report.cases.len(), required_catalog().len() - 3);
        assert_eq!(report.sealed.case_count, required_catalog().len() - 3);
        let run_evidence = serde_json::to_string(&selected.run).unwrap();
        assert!(run_evidence.contains("sealed-arc_dreaming"));
        assert!(run_evidence.contains("sealed-pretraining_causal"));
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains("sealed-arc_dreaming"));
        assert!(!serialized.contains("sealed-pretraining_causal"));

        let mut duplicate_baseline_run = runs[1].run.clone();
        let distinct_label = duplicate_baseline_run.metadata.baseline.id.clone();
        duplicate_baseline_run.metadata.baseline = runs[0].run.metadata.baseline.clone();
        duplicate_baseline_run.metadata.baseline.id = distinct_label;
        duplicate_baseline_run.validate().unwrap();
        let duplicate_baseline = write_run(temporary.path(), 99, &duplicate_baseline_run);
        let mut duplicate_matrix = runs.clone();
        duplicate_matrix[1] = duplicate_baseline;
        let error = verified_resource_benchmark_context(selected, &duplicate_matrix)
            .unwrap_err()
            .to_string();
        assert!(error.contains("reuse the same checkpoint"), "{error}");

        let mut mismatched = comparison.clone();
        mismatched.strongest_baseline_id = runs[0].run.metadata.baseline.id.clone();
        let error =
            evaluate_verified_promotion(selected, &runs, &mismatched, &resources_path, &policy)
                .unwrap_err()
                .to_string();
        assert!(error.contains("as strongest"), "{error}");
    }

    #[test]
    fn comparison_matrix_requires_the_same_candidate_acceptance_measurement() {
        let temporary = tempfile::tempdir().unwrap();
        let public =
            write_catalog_suite(temporary.path(), "public-catalog", SuiteVisibility::Public);
        let sealed =
            write_catalog_suite(temporary.path(), "sealed-catalog", SuiteVisibility::Sealed);
        let candidate = target(temporary.path(), "candidate", 100);
        let make_run =
            |baseline_id: &str, ablation: AblationId, evaluator: &mut RankingEvaluator| {
                BenchmarkRunner {
                    config: BenchmarkRunConfig {
                        evaluator_id: "fixed-evaluator".into(),
                        evaluator_version: evaluator_sha256(),
                        evaluator_arguments: Vec::new(),
                        paired_seeds: vec![11, 22, 33],
                        order_seed: 7,
                        gpu_hours_per_evaluation: 0.25,
                        ablation: Some(ablation),
                    },
                }
                .run(
                    &[public.clone(), sealed.clone()],
                    &target(temporary.path(), baseline_id, 100),
                    &candidate,
                    evaluator,
                )
                .unwrap()
            };
        let reference = make_run("baseline-01", AblationId::WakeOnly, &mut RankingEvaluator);
        let unchanged = make_run(
            "baseline-02",
            AblationId::StaticReplay,
            &mut RankingEvaluator,
        );
        let mut changed = unchanged.clone();
        changed.cases[0].pairs[0].candidate.gpu_hours /= 2.0;
        changed.validate().unwrap();
        let error = validate_comparison_compatibility(&reference, &changed)
            .unwrap_err()
            .to_string();
        assert!(error.contains("candidate measurement"), "{error}");

        changed = unchanged.clone();
        changed.cases[0].pairs[0]
            .candidate
            .metrics
            .insert("timing_jitter".into(), 1.0);
        changed.validate().unwrap();
        let error = validate_comparison_compatibility(&reference, &changed)
            .unwrap_err()
            .to_string();
        assert!(error.contains("candidate measurement"), "{error}");

        changed = unchanged;
        changed.cases[0].pairs[0].candidate.score += 0.01;
        changed.validate().unwrap();
        let error = validate_comparison_compatibility(&reference, &changed)
            .unwrap_err()
            .to_string();
        assert!(error.contains("candidate measurement"), "{error}");
    }

    #[test]
    fn coverage_requires_each_public_and_sealed_catalog_case() {
        let temporary = tempfile::tempdir().unwrap();
        let public =
            write_catalog_suite(temporary.path(), "public-catalog", SuiteVisibility::Public);
        let sealed =
            write_catalog_suite(temporary.path(), "sealed-catalog", SuiteVisibility::Sealed);
        assert!(validate_catalog_coverage(std::slice::from_ref(&public), false).is_err());

        let mut incomplete = sealed;
        incomplete.manifest.cases.pop();
        assert!(validate_catalog_coverage(&[public, incomplete], false).is_err());
    }

    #[test]
    fn coverage_rejects_duplicate_catalog_entries_before_evaluation() {
        let temporary = tempfile::tempdir().unwrap();
        let mut public =
            write_catalog_suite(temporary.path(), "public-catalog", SuiteVisibility::Public);
        let sealed =
            write_catalog_suite(temporary.path(), "sealed-catalog", SuiteVisibility::Sealed);
        let mut duplicate = public.manifest.cases[0].clone();
        duplicate.id.push_str("-duplicate");
        public.manifest.cases.push(duplicate);

        let error = validate_catalog_coverage(&[public, sealed], false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("repeats"), "{error}");
    }

    #[test]
    fn benchmark_suite_manifest_v1_is_rejected() {
        let temporary = tempfile::tempdir().unwrap();
        let mut suite = write_suite(
            temporary.path(),
            "legacy-suite",
            SuiteVisibility::Public,
            "legacy-case",
            MetricDirection::Maximize,
        )
        .manifest;
        suite.version = 1;
        let error = suite.validate().unwrap_err().to_string();
        assert!(error.contains("unsupported benchmark manifest version 1"));
    }
}
