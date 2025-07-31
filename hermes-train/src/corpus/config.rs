use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current on-disk corpus configuration and manifest schema.
pub const CORPUS_SCHEMA_VERSION: u32 = 2;

// These limits bound work derived from one small configuration before the
// pipeline reaches a byte-bounded remote response or shard writer. They are
// deliberately far above the production recipe while preventing values such
// as `usize::MAX` from turning a typo or hostile recipe into an allocation or
// CPU denial of service.
const MAX_DISCOVERY_QUERIES: usize = 4_096;
const MAX_DISCOVERY_HITS: usize = 1_000_000_000;
pub(super) const MAX_DISCOVERY_BATCH_SIZE: usize = 65_536;
const MAX_CLASSIFICATION_RULES: usize = 4_096;
const MAX_CLASSIFICATION_TERMS: usize = 4_096;
const MAX_TRANSFORMATIONS: usize = 1_024;
const MAX_COPIES_PER_RECORD: usize = 4_096;

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
            self.transformations.len() <= MAX_TRANSFORMATIONS,
            "corpus transformations exceed the {MAX_TRANSFORMATIONS}-item limit"
        );
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
                transform.name != "source",
                "transformation name `source` is reserved for canonical corpus views"
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
            self.queries.len() <= MAX_DISCOVERY_QUERIES,
            "corpus discovery queries exceed the {MAX_DISCOVERY_QUERIES}-item limit"
        );
        ensure!(
            (1..=MAX_DISCOVERY_BATCH_SIZE).contains(&self.materialization_batch_size),
            "materialization_batch_size must be within 1..={MAX_DISCOVERY_BATCH_SIZE}"
        );
        let mut names = std::collections::BTreeSet::new();
        let mut total_limit = 0usize;
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
                (1..=MAX_DISCOVERY_HITS).contains(&query.limit),
                "query `{}` limit must be within 1..={MAX_DISCOVERY_HITS}",
                query.name
            );
            total_limit = total_limit.checked_add(query.limit).ok_or_else(|| {
                anyhow::anyhow!("aggregate discovery query limit overflows usize")
            })?;
            ensure!(
                total_limit <= MAX_DISCOVERY_HITS,
                "aggregate discovery query limit exceeds {MAX_DISCOVERY_HITS} hits"
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
    pub(super) fn validate(&self) -> Result<()> {
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
        ensure!(
            self.default_topic
                .as_deref()
                .is_none_or(|label| !label.trim().is_empty())
                && self
                    .default_difficulty
                    .as_deref()
                    .is_none_or(|label| !label.trim().is_empty()),
            "classification default labels must not be empty"
        );
        validate_rules("topic", &self.topic_rules)?;
        validate_rules("difficulty", &self.difficulty_rules)
    }
}

fn validate_rules(kind: &str, rules: &[ClassificationRule]) -> Result<()> {
    ensure!(
        rules.len() <= MAX_CLASSIFICATION_RULES,
        "{kind} classification rules exceed the {MAX_CLASSIFICATION_RULES}-item limit"
    );
    let mut labels = std::collections::BTreeSet::new();
    let mut total_terms = 0usize;
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
        total_terms = total_terms
            .checked_add(rule.any_terms.len())
            .and_then(|count| count.checked_add(rule.all_terms.len()))
            .and_then(|count| count.checked_add(rule.none_terms.len()))
            .ok_or_else(|| anyhow::anyhow!("{kind} classification term count overflows usize"))?;
        ensure!(
            total_terms <= MAX_CLASSIFICATION_TERMS,
            "{kind} classification terms exceed the {MAX_CLASSIFICATION_TERMS}-item limit"
        );
        ensure!(
            !rule.any_terms.is_empty()
                || !rule.all_terms.is_empty()
                || !rule.none_terms.is_empty()
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
            self.max_copies_per_record <= MAX_COPIES_PER_RECORD,
            "max_copies_per_record exceeds the {MAX_COPIES_PER_RECORD}-copy limit"
        );
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
        ensure!(
            self.topic_copies
                .keys()
                .chain(self.difficulty_copies.keys())
                .all(|label| !label.trim().is_empty()),
            "configured repetition labels must not be empty"
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
    pub(super) fn validate(&self) -> Result<()> {
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
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.max_tokens_per_shard > 0,
            "max_tokens_per_shard must be positive"
        );
        Ok(())
    }
}
