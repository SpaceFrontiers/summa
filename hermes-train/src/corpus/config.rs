use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current on-disk corpus configuration and manifest schema.
pub const CORPUS_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusBuildConfig {
    pub version: u32,
    pub build_id: String,
    pub discovery: DiscoveryConfig,
    #[serde(default)]
    pub normalization: NormalizationConfig,
    #[serde(default)]
    pub deduplication: DeduplicationConfig,
    #[serde(default)]
    pub classification: ClassificationConfig,
    #[serde(default)]
    pub transformations: Vec<TransformationConfig>,
    #[serde(default)]
    pub repetition: RepetitionConfig,
    pub token_target: TokenTarget,
    pub sharding: ShardingConfig,
}

impl CorpusBuildConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == CORPUS_SCHEMA_VERSION,
            "unsupported corpus schema version {}; expected {CORPUS_SCHEMA_VERSION}",
            self.version
        );
        ensure!(
            !self.build_id.trim().is_empty(),
            "corpus build_id must not be empty"
        );
        ensure!(
            !self.build_id.contains(['/', '\\']) && self.build_id != "." && self.build_id != "..",
            "corpus build_id must be one safe path component"
        );
        self.discovery.validate()?;
        self.normalization.validate()?;
        self.deduplication.validate()?;
        self.classification.validate()?;
        ensure!(
            self.transformations.iter().all(|item| item.copies > 0),
            "every transformation must emit at least one copy"
        );
        let mut transform_names = std::collections::BTreeSet::new();
        for transform in &self.transformations {
            ensure!(
                !transform.name.trim().is_empty(),
                "transformation names must not be empty"
            );
            ensure!(
                transform_names.insert(transform.name.as_str()),
                "duplicate transformation name `{}`",
                transform.name
            );
            ensure!(
                transform.template.contains("${text}"),
                "transformation `{}` template must contain `${{text}}`",
                transform.name
            );
        }
        self.repetition.validate()?;
        self.token_target.validate()?;
        self.sharding.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryConfig {
    pub queries: Vec<DiscoveryQuery>,
    #[serde(default = "default_materialization_batch")]
    pub materialization_batch_size: usize,
}

fn default_materialization_batch() -> usize {
    500
}

impl DiscoveryConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.queries.is_empty(),
            "corpus discovery requires at least one query"
        );
        ensure!(
            self.materialization_batch_size > 0,
            "materialization_batch_size must be positive"
        );
        let mut names = std::collections::BTreeSet::new();
        for query in &self.queries {
            ensure!(
                !query.name.trim().is_empty() && !query.text.trim().is_empty(),
                "discovery query name and text must not be empty"
            );
            ensure!(
                names.insert(query.name.as_str()),
                "duplicate discovery query name `{}`",
                query.name
            );
            ensure!(
                query.limit > 0,
                "query `{}` limit must be positive",
                query.name
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryQuery {
    pub name: String,
    pub text: String,
    pub limit: usize,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NormalizationConfig {
    pub trim: bool,
    pub normalize_newlines: bool,
    pub collapse_horizontal_whitespace: bool,
    pub max_consecutive_blank_lines: usize,
    pub reject_replacement_character: bool,
    pub minimum_characters: usize,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            trim: true,
            normalize_newlines: true,
            collapse_horizontal_whitespace: true,
            max_consecutive_blank_lines: 2,
            reject_replacement_character: true,
            minimum_characters: 1,
        }
    }
}

impl NormalizationConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.minimum_characters > 0,
            "normalization minimum_characters must be positive"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeduplicationConfig {
    pub by_record_key: bool,
    pub by_normalized_text: bool,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            by_record_key: true,
            by_normalized_text: true,
        }
    }
}

impl DeduplicationConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.by_record_key || self.by_normalized_text,
            "deduplication must enable record-key or normalized-text matching"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClassificationConfig {
    pub topic_rules: Vec<ClassificationRule>,
    pub difficulty_rules: Vec<ClassificationRule>,
    pub default_topic: Option<String>,
    pub default_difficulty: Option<String>,
}

impl ClassificationConfig {
    fn validate(&self) -> Result<()> {
        validate_rules("topic", &self.topic_rules)?;
        validate_rules("difficulty", &self.difficulty_rules)
    }
}

fn validate_rules(kind: &str, rules: &[ClassificationRule]) -> Result<()> {
    let mut labels = std::collections::BTreeSet::new();
    for rule in rules {
        ensure!(
            !rule.label.trim().is_empty(),
            "{kind} classification rule label must not be empty"
        );
        ensure!(
            labels.insert(rule.label.as_str()),
            "duplicate {kind} classification label `{}`",
            rule.label
        );
        ensure!(
            !rule.any_terms.is_empty()
                || !rule.all_terms.is_empty()
                || !rule.metadata_equals.is_empty(),
            "{kind} rule `{}` has no predicates",
            rule.label
        );
        ensure!(
            rule.any_terms
                .iter()
                .chain(&rule.all_terms)
                .chain(&rule.none_terms)
                .all(|term| !term.trim().is_empty()),
            "{kind} rule `{}` contains an empty term",
            rule.label
        );
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationRule {
    pub label: String,
    /// Explicit precedence for overlapping rules. Equal-priority matches are
    /// resolved by specificity and then label, never declaration order.
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub any_terms: Vec<String>,
    #[serde(default)]
    pub all_terms: Vec<String>,
    #[serde(default)]
    pub none_terms: Vec<String>,
    #[serde(default)]
    pub metadata_equals: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransformationConfig {
    pub name: String,
    /// Text template. `${text}`, `${topic}`, `${difficulty}`, and
    /// `${metadata.<key>}` are expanded.
    pub template: String,
    #[serde(default = "default_one")]
    pub copies: usize,
    #[serde(default)]
    pub when: RecordPredicate,
}

fn default_one() -> usize {
    1
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecordPredicate {
    pub topics: Vec<String>,
    pub difficulties: Vec<String>,
    pub metadata_equals: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RepetitionConfig {
    pub base_copies: usize,
    pub max_copies_per_record: usize,
    pub topic_copies: BTreeMap<String, usize>,
    pub difficulty_copies: BTreeMap<String, usize>,
}

impl Default for RepetitionConfig {
    fn default() -> Self {
        Self {
            base_copies: 1,
            max_copies_per_record: 1,
            topic_copies: BTreeMap::new(),
            difficulty_copies: BTreeMap::new(),
        }
    }
}

impl RepetitionConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.base_copies > 0, "base_copies must be positive");
        ensure!(
            self.max_copies_per_record >= self.base_copies,
            "max_copies_per_record must cover base_copies"
        );
        ensure!(
            self.topic_copies
                .values()
                .chain(self.difficulty_copies.values())
                .all(|copies| *copies > 0 && *copies <= self.max_copies_per_record),
            "configured repetition copies must be positive and within max_copies_per_record"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenTarget {
    pub minimum: u64,
    pub desired: u64,
    pub maximum: u64,
}

impl TokenTarget {
    fn validate(&self) -> Result<()> {
        ensure!(self.minimum > 0, "minimum token target must be positive");
        ensure!(
            self.minimum <= self.desired && self.desired <= self.maximum,
            "token targets must satisfy minimum <= desired <= maximum"
        );
        Ok(())
    }

    pub fn accepts(&self, unique_tokens: u64) -> bool {
        (self.minimum..=self.maximum).contains(&unique_tokens)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShardingConfig {
    pub max_tokens_per_shard: u64,
}

impl ShardingConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.max_tokens_per_shard > 0,
            "max_tokens_per_shard must be positive"
        );
        Ok(())
    }
}
