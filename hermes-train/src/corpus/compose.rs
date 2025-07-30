//! Deterministic, backend-neutral composition of classified corpus shards.
//!
//! Composition consumes an immutable [`CorpusManifest`], assigns tokenized
//! records to configured stage/stratum predicates, and emits one ordinary
//! corpus manifest per stage.  No search-engine fields or education labels are
//! built into this module: topic, difficulty, view, and arbitrary metadata
//! predicates are all configuration.  Strata are interleaved by fixed token
//! weights using bounded memory; the trainer's shuffle buffer is responsible
//! for stochastic sample order during training.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    CorpusManifest, CorpusManifestBody, CorpusStageStats, ShardManifest, ShardingConfig,
    TokenTarget,
};

/// Current curriculum-composition schema. Earlier curriculum formats are not
/// accepted or projected into this one.
pub const CURRICULUM_COMPOSITION_VERSION: u32 = 2;

const COMPOSITION_PROGRESS_VERSION: u32 = 1;
const PROGRESS_FILE: &str = "composition-progress.json";
const PROGRESS_TEMP_FILE: &str = ".composition-progress.next";
const WORK_IDENTITY_FILE: &str = "composition-work.json";

#[cfg(test)]
thread_local! {
    static TEST_INTERRUPTION: std::cell::RefCell<Option<(&'static str, usize)>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
fn set_test_interruption(boundary: &'static str, occurrence: usize) {
    TEST_INTERRUPTION.with(|configured| {
        *configured.borrow_mut() = Some((boundary, occurrence));
    });
}

#[cfg(test)]
fn maybe_test_interrupt(boundary: &'static str) -> Result<()> {
    TEST_INTERRUPTION.with(|configured| {
        let mut configured = configured.borrow_mut();
        let Some((expected, remaining)) = configured.as_mut() else {
            return Ok(());
        };
        if *expected != boundary {
            return Ok(());
        }
        if *remaining > 1 {
            *remaining -= 1;
            return Ok(());
        }
        *configured = None;
        anyhow::bail!("injected composition interruption at {boundary}")
    })
}

#[cfg(not(test))]
#[inline]
fn maybe_test_interrupt(_boundary: &'static str) -> Result<()> {
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CompositionProgress {
    progress_sha256: String,
    state: CompositionProgressState,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CompositionProgressState {
    version: u32,
    build_id: String,
    composition_sha256: String,
    source_manifest_sha256: String,
    next_source_shard: usize,
    spooling_complete: bool,
    spools: Vec<Vec<SpoolCheckpoint>>,
    completed_stages: Vec<CurriculumStageSummary>,
    current_stage: Option<StageProgress>,
    root_manifest_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SpoolCheckpoint {
    path: String,
    records: u64,
    tokens: u64,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StageProgress {
    stage_index: usize,
    spool_offsets: Vec<u64>,
    emitted_records: Vec<u64>,
    emitted_tokens: Vec<u64>,
    total_records: u64,
    total_tokens: u64,
    topic_counts: BTreeMap<String, u64>,
    difficulty_counts: BTreeMap<String, u64>,
    shards: Vec<ShardManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CompositionWorkIdentity {
    version: u32,
    build_id: String,
    composition_sha256: String,
    source_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurriculumCompositionConfig {
    pub version: u32,
    /// Safe directory name published below the CLI output root.
    pub build_id: String,
    /// Relative paths resolve beside the composition configuration file.
    pub source_manifest: PathBuf,
    /// Canonical `manifest_sha256` from the source corpus manifest.
    pub source_manifest_sha256: String,
    /// Used only to break otherwise equal weighted-scheduling choices.
    pub seed: u64,
    pub stages: Vec<CurriculumStageConfig>,
}

impl CurriculumCompositionConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| {
            format!(
                "failed to read curriculum composition configuration {}",
                path.display()
            )
        })?;
        let config: Self = serde_json::from_slice(&bytes).with_context(|| {
            format!("invalid curriculum composition JSON in {}", path.display())
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == CURRICULUM_COMPOSITION_VERSION,
            "unsupported curriculum composition version {}; expected {CURRICULUM_COMPOSITION_VERSION}",
            self.version
        );
        validate_component("curriculum build_id", &self.build_id)?;
        ensure!(
            !self.source_manifest.as_os_str().is_empty(),
            "source_manifest must not be empty"
        );
        ensure!(
            is_sha256(&self.source_manifest_sha256),
            "source_manifest_sha256 must contain 64 lowercase hexadecimal characters"
        );
        ensure!(
            !self.stages.is_empty(),
            "curriculum composition requires at least one stage"
        );
        let mut names = BTreeSet::new();
        for stage in &self.stages {
            stage.validate()?;
            ensure!(
                names.insert(stage.name.as_str()),
                "duplicate curriculum stage `{}`",
                stage.name
            );
        }
        Ok(())
    }

    /// Compose every stage and publish the complete curriculum atomically as
    /// `output_root/build_id`.
    ///
    /// Progress is content-pinned and synced after each input shard and output
    /// shard. Re-running the same command resumes the `.building` generation;
    /// only the uncommitted shard is replayed. Completed spools, shards, and
    /// manifests are verified before they are trusted. The work directory
    /// contains only transient bounded-memory stratum spools and is removed
    /// before successful publication.
    pub fn run(
        &self,
        config_path: &Path,
        output_root: &Path,
        work_directory: &Path,
    ) -> Result<(PathBuf, CurriculumManifest)> {
        self.validate()?;
        let source_path = resolve_relative(config_path, &self.source_manifest);
        ensure_regular_file(&source_path, "source corpus manifest")?;
        let source: CorpusManifest = serde_json::from_slice(
            &fs::read(&source_path)
                .with_context(|| format!("failed to read {}", source_path.display()))?,
        )
        .with_context(|| format!("invalid corpus manifest {}", source_path.display()))?;
        ensure!(
            source.manifest_sha256 == self.source_manifest_sha256,
            "source corpus manifest identity changed: expected {}, got {}",
            self.source_manifest_sha256,
            source.manifest_sha256
        );
        let source_root = source_path.parent().unwrap_or_else(|| Path::new("."));
        ensure_real_directory(source_root, "source corpus root")?;
        source.verify(source_root)?;

        create_and_validate_directory(output_root, "curriculum output root")?;
        create_and_validate_directory(work_directory, "curriculum work directory")?;
        let final_path = output_root.join(&self.build_id);
        reject_path_if_present(&final_path, "immutable curriculum")?;
        let staging_path = output_root.join(format!(".{}.building", self.build_id));
        let identity = self.identity_sha256()?;
        let work_path = work_directory.join(format!(
            ".{}.{}.composition",
            self.build_id,
            &identity[..16]
        ));

        let progress_path = staging_path.join(PROGRESS_FILE);
        let mut progress = if path_kind(&staging_path)?.is_none() {
            fs::create_dir(&staging_path).with_context(|| {
                format!(
                    "failed to create curriculum staging path {}",
                    staging_path.display()
                )
            })?;
            sync_directory(output_root)?;
            let mut progress = self.initial_progress(&source, &identity)?;
            persist_progress(&progress_path, &mut progress)?;
            progress
        } else {
            ensure_real_directory(&staging_path, "curriculum staging path")?;
            match load_progress(&progress_path)? {
                Some(progress) => progress,
                None => {
                    ensure!(
                        directory_is_empty(&staging_path)?,
                        "curriculum staging path has no durable progress but is not empty"
                    );
                    let mut progress = self.initial_progress(&source, &identity)?;
                    persist_progress(&progress_path, &mut progress)?;
                    progress
                }
            }
        };
        self.validate_progress(&progress, &source, &identity)?;

        if progress.state.root_manifest_sha256.is_none() {
            let mut spools =
                open_or_create_spools(&work_path, self, &source, &identity, &mut progress)?;
            if !progress.state.spooling_complete {
                for shard_index in progress.state.next_source_shard..source.build.shards.len() {
                    spool_source_shard(
                        &source.build.shards[shard_index],
                        source_root,
                        &self.stages,
                        &mut spools,
                    )?;
                    finish_spools(&mut spools)?;
                    progress.state.next_source_shard = shard_index + 1;
                    capture_spool_checkpoints(&spools, &mut progress.state.spools)?;
                    persist_progress(&progress_path, &mut progress)?;
                    maybe_test_interrupt("source_shard")?;
                }
                source
                    .verify(source_root)
                    .context("source corpus changed while curriculum spools were being prepared")?;
                progress.state.spooling_complete = true;
                persist_progress(&progress_path, &mut progress)?;
            }
            drop(spools);
            validate_spool_tree(&work_path, &progress.state, true)?;

            validate_staging_tree(self, &source, &identity, &staging_path, &progress)?;
            while progress.state.completed_stages.len() < self.stages.len() {
                let stage_index = progress.state.completed_stages.len();
                if progress.state.current_stage.is_none() {
                    progress.state.current_stage = Some(StageProgress::new(
                        stage_index,
                        self.stages[stage_index].strata.len(),
                    ));
                    persist_progress(&progress_path, &mut progress)?;
                }
                let summary = compose_stage(
                    self,
                    &source,
                    stage_index,
                    &identity,
                    &staging_path,
                    &work_path,
                    &progress_path,
                    &mut progress,
                )?;
                progress.state.current_stage = None;
                progress.state.completed_stages.push(summary);
                persist_progress(&progress_path, &mut progress)?;
            }
        } else {
            validate_staging_tree(self, &source, &identity, &staging_path, &progress)?;
        }

        let body = CurriculumManifestBody {
            version: self.version,
            build_id: self.build_id.clone(),
            config_sha256: identity,
            source_manifest_sha256: source.manifest_sha256.clone(),
            seed: self.seed,
            stages: progress.state.completed_stages.clone(),
        };
        let manifest = CurriculumManifest {
            manifest_sha256: canonical_json_sha256(&body)?,
            build: body,
        };
        publish_or_verify_root_manifest(&staging_path, &manifest)?;
        match &progress.state.root_manifest_sha256 {
            Some(expected) => ensure!(
                expected == &manifest.manifest_sha256,
                "completed curriculum root manifest identity changed"
            ),
            None => {
                progress.state.root_manifest_sha256 = Some(manifest.manifest_sha256.clone());
                persist_progress(&progress_path, &mut progress)?;
            }
        }
        validate_staging_tree(
            self,
            &source,
            &manifest.build.config_sha256,
            &staging_path,
            &progress,
        )?;
        maybe_test_interrupt("root_ready")?;

        if path_kind(&work_path)?.is_some() {
            ensure_real_directory(&work_path, "curriculum work path")?;
            ensure_tree_has_no_symlinks(&work_path, "curriculum work path")?;
            fs::remove_dir_all(&work_path).with_context(|| {
                format!(
                    "failed to remove completed curriculum work path {}",
                    work_path.display()
                )
            })?;
            sync_directory(work_directory)?;
        }
        fs::rename(&staging_path, &final_path).with_context(|| {
            format!(
                "failed to publish immutable curriculum {}",
                final_path.display()
            )
        })?;
        File::open(output_root)?.sync_all()?;
        Ok((final_path, manifest))
    }

    fn initial_progress(
        &self,
        source: &CorpusManifest,
        composition_sha256: &str,
    ) -> Result<CompositionProgress> {
        let spools = self
            .stages
            .iter()
            .enumerate()
            .map(|(stage_index, stage)| {
                stage
                    .strata
                    .iter()
                    .enumerate()
                    .map(|(stratum_index, _)| SpoolCheckpoint {
                        path: spool_path(stage_index, stratum_index),
                        records: 0,
                        tokens: 0,
                        bytes: 0,
                        sha256: empty_sha256(),
                    })
                    .collect()
            })
            .collect();
        CompositionProgress::new(CompositionProgressState {
            version: COMPOSITION_PROGRESS_VERSION,
            build_id: self.build_id.clone(),
            composition_sha256: composition_sha256.to_owned(),
            source_manifest_sha256: source.manifest_sha256.clone(),
            next_source_shard: 0,
            spooling_complete: source.build.shards.is_empty(),
            spools,
            completed_stages: Vec::new(),
            current_stage: None,
            root_manifest_sha256: None,
        })
    }

    fn validate_progress(
        &self,
        progress: &CompositionProgress,
        source: &CorpusManifest,
        composition_sha256: &str,
    ) -> Result<()> {
        progress.verify_hash()?;
        let state = &progress.state;
        ensure!(
            state.version == COMPOSITION_PROGRESS_VERSION,
            "unsupported composition progress version {}",
            state.version
        );
        ensure!(
            state.build_id == self.build_id,
            "curriculum build identity changed"
        );
        ensure!(
            state.composition_sha256 == composition_sha256,
            "curriculum composition configuration changed during resume"
        );
        ensure!(
            state.source_manifest_sha256 == source.manifest_sha256,
            "source corpus manifest changed during resume"
        );
        ensure!(
            state.next_source_shard <= source.build.shards.len(),
            "composition progress points past the source shard list"
        );
        ensure!(
            !state.spooling_complete || state.next_source_shard == source.build.shards.len(),
            "composition progress has an inconsistent spooling boundary"
        );
        validate_spool_layout(self, &state.spools)?;
        ensure!(
            state.spooling_complete
                || (state.completed_stages.is_empty()
                    && state.current_stage.is_none()
                    && state.root_manifest_sha256.is_none()),
            "composition progress contains stage output before spooling completed"
        );
        ensure!(
            state.completed_stages.len() <= self.stages.len(),
            "composition progress contains too many completed stages"
        );
        for (index, summary) in state.completed_stages.iter().enumerate() {
            ensure!(
                summary.name == self.stages[index].name,
                "composition progress completed stages are not a configuration prefix"
            );
        }
        if let Some(current) = &state.current_stage {
            ensure!(
                state.spooling_complete,
                "composition progress started a stage before spooling completed"
            );
            ensure!(
                current.stage_index == state.completed_stages.len()
                    && current.stage_index < self.stages.len(),
                "composition progress has an invalid current stage"
            );
            current.validate(
                &self.stages[current.stage_index],
                &state.spools[current.stage_index],
            )?;
        }
        ensure!(
            state.root_manifest_sha256.is_none()
                || (state.spooling_complete
                    && state.current_stage.is_none()
                    && state.completed_stages.len() == self.stages.len()),
            "composition progress marks an incomplete curriculum as complete"
        );
        if let Some(hash) = &state.root_manifest_sha256 {
            ensure!(is_sha256(hash), "invalid root curriculum manifest identity");
        }
        Ok(())
    }

    fn identity_sha256(&self) -> Result<String> {
        // Source paths are deployment details. Pin the source manifest content
        // while keeping identical composition portable between directory
        // layouts and machines.
        canonical_json_sha256(&serde_json::json!({
            "version": self.version,
            "build_id": self.build_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "seed": self.seed,
            "stages": self.stages,
        }))
    }
}

impl CompositionProgress {
    fn new(state: CompositionProgressState) -> Result<Self> {
        let progress_sha256 = canonical_json_sha256(&state)?;
        Ok(Self {
            progress_sha256,
            state,
        })
    }

    fn refresh_hash(&mut self) -> Result<()> {
        self.progress_sha256 = canonical_json_sha256(&self.state)?;
        Ok(())
    }

    fn verify_hash(&self) -> Result<()> {
        ensure!(
            is_sha256(&self.progress_sha256),
            "composition progress contains an invalid content identity"
        );
        ensure!(
            self.progress_sha256 == canonical_json_sha256(&self.state)?,
            "composition progress content hash does not match its state"
        );
        Ok(())
    }
}

impl StageProgress {
    fn new(stage_index: usize, strata: usize) -> Self {
        Self {
            stage_index,
            spool_offsets: vec![0; strata],
            emitted_records: vec![0; strata],
            emitted_tokens: vec![0; strata],
            total_records: 0,
            total_tokens: 0,
            topic_counts: BTreeMap::new(),
            difficulty_counts: BTreeMap::new(),
            shards: Vec::new(),
        }
    }

    fn validate(&self, stage: &CurriculumStageConfig, spools: &[SpoolCheckpoint]) -> Result<()> {
        let strata = stage.strata.len();
        ensure!(
            self.spool_offsets.len() == strata
                && self.emitted_records.len() == strata
                && self.emitted_tokens.len() == strata
                && spools.len() == strata,
            "stage `{}` progress has the wrong stratum cardinality",
            stage.name
        );
        ensure!(
            self.total_records == checked_sum(&self.emitted_records, "stage record")?
                && self.total_tokens == checked_sum(&self.emitted_tokens, "stage token")?,
            "stage `{}` progress totals do not match stratum totals",
            stage.name
        );
        ensure!(
            self.total_tokens <= stage.token_target.maximum,
            "stage `{}` progress exceeds its maximum token target",
            stage.name
        );
        for (index, spool) in spools.iter().enumerate() {
            ensure!(
                self.spool_offsets[index] <= spool.bytes
                    && self.emitted_records[index] <= spool.records
                    && self.emitted_tokens[index] <= spool.tokens,
                "stage `{}` progress exceeds stratum `{}` spool bounds",
                stage.name,
                stage.strata[index].name
            );
        }
        let mut shard_records = 0_u64;
        let mut shard_tokens = 0_u64;
        for (index, shard) in self.shards.iter().enumerate() {
            ensure!(
                shard.path == shard_path(index),
                "stage `{}` progress has a non-contiguous shard path",
                stage.name
            );
            ensure!(
                shard.records > 0 && shard.tokens > 0 && is_sha256(&shard.sha256),
                "stage `{}` progress contains an invalid shard",
                stage.name
            );
            shard_records = checked_add(shard_records, shard.records)?;
            shard_tokens = checked_add(shard_tokens, shard.tokens)?;
        }
        ensure!(
            shard_records == self.total_records && shard_tokens == self.total_tokens,
            "stage `{}` progress totals do not match completed shards",
            stage.name
        );
        let classified_topics = self
            .topic_counts
            .values()
            .try_fold(0_u64, |total, count| total.checked_add(*count))
            .context("stage topic counters overflow u64")?;
        let classified_difficulties = self
            .difficulty_counts
            .values()
            .try_fold(0_u64, |total, count| total.checked_add(*count))
            .context("stage difficulty counters overflow u64")?;
        ensure!(
            classified_topics <= self.total_records
                && classified_difficulties <= self.total_records,
            "stage `{}` classification counters are invalid",
            stage.name
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurriculumStageConfig {
    pub name: String,
    #[serde(default)]
    pub when: CurriculumPredicate,
    /// Eligible rows that match no stratum are either rejected or explicitly
    /// excluded. Overlapping strata are always an error.
    pub unmatched: CurriculumUnmatchedPolicy,
    pub token_target: TokenTarget,
    pub sharding: ShardingConfig,
    /// Maximum absolute difference between configured and emitted token share.
    pub max_fraction_deviation: f64,
    pub strata: Vec<CurriculumStratumConfig>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CurriculumUnmatchedPolicy {
    Error,
    Exclude,
}

impl CurriculumStageConfig {
    fn validate(&self) -> Result<()> {
        validate_component("curriculum stage name", &self.name)?;
        validate_token_target(&self.token_target)?;
        ensure!(
            self.sharding.max_tokens_per_shard > 0,
            "stage `{}` max_tokens_per_shard must be positive",
            self.name
        );
        ensure!(
            self.max_fraction_deviation.is_finite()
                && (0.0..=1.0).contains(&self.max_fraction_deviation),
            "stage `{}` max_fraction_deviation must be finite and within [0, 1]",
            self.name
        );
        ensure!(
            !self.strata.is_empty(),
            "stage `{}` requires at least one stratum",
            self.name
        );
        let mut names = BTreeSet::new();
        let mut total_weight = 0_u64;
        for stratum in &self.strata {
            validate_component("curriculum stratum name", &stratum.name)?;
            ensure!(
                names.insert(stratum.name.as_str()),
                "stage `{}` repeats stratum `{}`",
                self.name,
                stratum.name
            );
            ensure!(
                stratum.weight > 0,
                "stage `{}` stratum `{}` weight must be positive",
                self.name,
                stratum.name
            );
            total_weight = total_weight
                .checked_add(stratum.weight)
                .context("curriculum stratum weights overflow u64")?;
        }
        ensure!(total_weight > 0, "curriculum stage has no positive weight");
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurriculumStratumConfig {
    pub name: String,
    pub weight: u64,
    #[serde(default)]
    pub when: CurriculumPredicate,
}

/// Generic predicates over fields already present in tokenized corpus rows.
/// Empty lists are wildcards; non-empty lists are membership tests. Metadata
/// comparisons use exact JSON equality and therefore support strings, numbers,
/// booleans, arrays, or objects without provider-specific field assumptions.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurriculumPredicate {
    pub topics: Vec<String>,
    pub difficulties: Vec<String>,
    pub views: Vec<String>,
    pub metadata_equals: BTreeMap<String, Value>,
}

impl CurriculumPredicate {
    fn matches(&self, record: &TokenizedCorpusRecord) -> bool {
        matches_optional(&self.topics, record.topic.as_deref())
            && matches_optional(&self.difficulties, record.difficulty.as_deref())
            && (self.views.is_empty() || self.views.iter().any(|view| view == &record.view))
            && self
                .metadata_equals
                .iter()
                .all(|(key, value)| record.metadata.get(key) == Some(value))
    }
}

fn matches_optional(configured: &[String], actual: Option<&str>) -> bool {
    configured.is_empty()
        || actual.is_some_and(|actual| configured.iter().any(|value| value == actual))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CurriculumStageSummary {
    pub name: String,
    pub manifest_path: String,
    pub manifest_sha256: String,
    pub records: u64,
    pub tokens: u64,
    pub stratum_records: BTreeMap<String, u64>,
    pub stratum_tokens: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CurriculumManifestBody {
    pub version: u32,
    pub build_id: String,
    pub config_sha256: String,
    pub source_manifest_sha256: String,
    pub seed: u64,
    pub stages: Vec<CurriculumStageSummary>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CurriculumManifest {
    pub manifest_sha256: String,
    pub build: CurriculumManifestBody,
}

impl CurriculumManifest {
    pub fn verify(&self, root: &Path) -> Result<()> {
        ensure_real_directory(root, "curriculum manifest root")?;
        ensure!(
            self.build.version == CURRICULUM_COMPOSITION_VERSION,
            "unsupported curriculum manifest version {}",
            self.build.version
        );
        ensure!(
            is_sha256(&self.build.config_sha256) && is_sha256(&self.build.source_manifest_sha256),
            "curriculum manifest contains an invalid content identity"
        );
        ensure!(
            self.manifest_sha256 == canonical_json_sha256(&self.build)?,
            "curriculum manifest body hash does not match manifest_sha256"
        );
        ensure!(
            !self.build.stages.is_empty(),
            "curriculum manifest contains no stages"
        );
        let mut names = BTreeSet::new();
        for stage in &self.build.stages {
            validate_component("curriculum stage name", &stage.name)?;
            ensure!(
                is_sha256(&stage.manifest_sha256),
                "stage `{}` has an invalid manifest identity",
                stage.name
            );
            ensure!(
                stage.manifest_path == format!("{}/manifest.json", stage.name),
                "stage `{}` manifest path is not canonical",
                stage.name
            );
            ensure!(
                names.insert(stage.name.as_str()),
                "curriculum manifest repeats stage `{}`",
                stage.name
            );
            let path = safe_join(root, &stage.manifest_path)?;
            ensure_regular_file(&path, "curriculum stage manifest")?;
            let manifest: CorpusManifest = serde_json::from_slice(&fs::read(&path)?)
                .with_context(|| format!("invalid stage manifest {}", path.display()))?;
            ensure!(
                manifest.manifest_sha256 == stage.manifest_sha256,
                "stage `{}` manifest identity changed",
                stage.name
            );
            let stage_root = path.parent().unwrap_or_else(|| Path::new("."));
            ensure_real_directory(stage_root, "curriculum stage root")?;
            manifest.verify(stage_root)?;
            ensure!(
                manifest.build.stats.emitted_views == stage.records
                    && manifest.build.stats.exposure_tokens == stage.tokens,
                "stage `{}` summary does not match its corpus manifest",
                stage.name
            );
            ensure!(
                stage
                    .stratum_records
                    .values()
                    .try_fold(0_u64, |total, value| { total.checked_add(*value) })
                    == Some(stage.records),
                "stage `{}` stratum record counts do not sum to the stage total",
                stage.name
            );
            ensure!(
                stage
                    .stratum_tokens
                    .values()
                    .try_fold(0_u64, |total, value| { total.checked_add(*value) })
                    == Some(stage.tokens),
                "stage `{}` stratum token counts do not sum to the stage total",
                stage.name
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenizedCorpusRecord {
    record_key: String,
    uris: Vec<String>,
    topic: Option<String>,
    difficulty: Option<String>,
    view: String,
    copy: usize,
    metadata: BTreeMap<String, Value>,
    tokens: Vec<u32>,
}

impl TokenizedCorpusRecord {
    fn validate(&self) -> Result<()> {
        ensure!(!self.record_key.is_empty(), "corpus record key is empty");
        ensure!(!self.view.is_empty(), "corpus record view is empty");
        ensure!(!self.tokens.is_empty(), "corpus record has no tokens");
        Ok(())
    }
}

struct SpoolWriter {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    records: u64,
    tokens: u64,
    bytes: u64,
    hasher: Sha256,
}

fn spool_source_shard(
    shard: &ShardManifest,
    source_root: &Path,
    stages: &[CurriculumStageConfig],
    spools: &mut [Vec<SpoolWriter>],
) -> Result<()> {
    let path = safe_join(source_root, &shard.path)?;
    let mut reader = open_source_shard(&path)?;
    let mut line = Vec::new();
    let mut line_number = 0_u64;
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        line_number += 1;
        while line
            .last()
            .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
        {
            line.pop();
        }
        ensure!(
            !line.is_empty(),
            "empty corpus record at {}:{line_number}",
            path.display()
        );
        let record: TokenizedCorpusRecord = serde_json::from_slice(&line).with_context(|| {
            format!(
                "invalid tokenized corpus row at {}:{line_number}",
                path.display()
            )
        })?;
        record.validate()?;
        let canonical = serde_json::to_vec(&record)?;
        for (stage_index, stage) in stages.iter().enumerate() {
            if !stage.when.matches(&record) {
                continue;
            }
            let matching = stage
                .strata
                .iter()
                .enumerate()
                .filter(|(_, stratum)| stratum.when.matches(&record))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            ensure!(
                matching.len() <= 1,
                "record `{}` at {}:{line_number} matches multiple strata in stage `{}`",
                record.record_key,
                path.display(),
                stage.name
            );
            let Some(stratum_index) = matching.first().copied() else {
                ensure!(
                    matches!(stage.unmatched, CurriculumUnmatchedPolicy::Exclude),
                    "record `{}` at {}:{line_number} matches no stratum in stage `{}`",
                    record.record_key,
                    path.display(),
                    stage.name
                );
                continue;
            };
            let spool = &mut spools[stage_index][stratum_index];
            let writer = spool.writer.as_mut().expect("spool is still open");
            writer.write_all(&canonical)?;
            writer.write_all(b"\n")?;
            spool.hasher.update(&canonical);
            spool.hasher.update(b"\n");
            spool.records = checked_add(spool.records, 1)?;
            spool.tokens = checked_add(spool.tokens, record.tokens.len())?;
            spool.bytes = checked_add(spool.bytes, canonical.len() + 1)?;
        }
    }
    Ok(())
}

fn open_source_shard(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open source shard {}", path.display()))?;
    if path.extension().is_some_and(|extension| extension == "zst") {
        let decoder = zstd::stream::read::Decoder::new(file)
            .with_context(|| format!("failed to decode source shard {}", path.display()))?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn finish_spools(spools: &mut [Vec<SpoolWriter>]) -> Result<()> {
    for stage in spools {
        for spool in stage {
            let writer = spool.writer.as_mut().expect("spool is still open");
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
    }
    Ok(())
}

fn capture_spool_checkpoints(
    spools: &[Vec<SpoolWriter>],
    checkpoints: &mut [Vec<SpoolCheckpoint>],
) -> Result<()> {
    ensure!(
        spools.len() == checkpoints.len(),
        "spool checkpoint stage cardinality changed"
    );
    for (stage_spools, stage_checkpoints) in spools.iter().zip(checkpoints) {
        ensure!(
            stage_spools.len() == stage_checkpoints.len(),
            "spool checkpoint stratum cardinality changed"
        );
        for (spool, checkpoint) in stage_spools.iter().zip(stage_checkpoints) {
            ensure!(
                spool.path.file_name().and_then(|name| name.to_str())
                    == Some(checkpoint.path.as_str()),
                "spool checkpoint path changed"
            );
            checkpoint.records = spool.records;
            checkpoint.tokens = spool.tokens;
            checkpoint.bytes = spool.bytes;
            checkpoint.sha256 = hex(&spool.hasher.clone().finalize());
        }
    }
    Ok(())
}

fn open_or_create_spools(
    work_path: &Path,
    composition: &CurriculumCompositionConfig,
    source: &CorpusManifest,
    composition_sha256: &str,
    progress: &mut CompositionProgress,
) -> Result<Vec<Vec<SpoolWriter>>> {
    let initial = progress.state.next_source_shard == 0 && !progress.state.spooling_complete;
    match path_kind(work_path)? {
        None => {
            ensure!(
                initial,
                "curriculum work path {} disappeared after durable progress",
                work_path.display()
            );
            fs::create_dir(work_path).with_context(|| {
                format!(
                    "failed to create curriculum work path {}",
                    work_path.display()
                )
            })?;
            sync_directory(
                work_path
                    .parent()
                    .context("curriculum work path has no parent")?,
            )?;
        }
        Some(PathKind::Directory) => ensure_real_directory(work_path, "curriculum work path")?,
        Some(PathKind::Symlink) => anyhow::bail!(
            "curriculum work path {} must not be a symlink",
            work_path.display()
        ),
        Some(_) => anyhow::bail!(
            "curriculum work path {} is not a directory",
            work_path.display()
        ),
    }

    let expected_identity = CompositionWorkIdentity {
        version: COMPOSITION_PROGRESS_VERSION,
        build_id: composition.build_id.clone(),
        composition_sha256: composition_sha256.to_owned(),
        source_manifest_sha256: source.manifest_sha256.clone(),
    };
    let identity_path = work_path.join(WORK_IDENTITY_FILE);
    match path_kind(&identity_path)? {
        None => {
            ensure!(
                initial && directory_is_empty(work_path)?,
                "curriculum work identity is missing from a non-empty or resumed work path"
            );
            write_new_json(&identity_path, &expected_identity)?;
            sync_directory(work_path)?;
        }
        Some(PathKind::File) => {
            ensure_regular_file(&identity_path, "curriculum work identity")?;
            let actual: CompositionWorkIdentity =
                serde_json::from_slice(&fs::read(&identity_path)?)
                    .context("invalid curriculum work identity")?;
            ensure!(
                actual == expected_identity,
                "curriculum work identity changed during resume"
            );
        }
        Some(PathKind::Symlink) => anyhow::bail!(
            "curriculum work identity {} must not be a symlink",
            identity_path.display()
        ),
        Some(_) => anyhow::bail!("curriculum work identity is not a regular file"),
    }

    validate_work_entries(work_path, &progress.state.spools)?;
    let mut result = Vec::with_capacity(progress.state.spools.len());
    for stage in &progress.state.spools {
        let mut writers = Vec::with_capacity(stage.len());
        for checkpoint in stage {
            let path = safe_join(work_path, &checkpoint.path)?;
            match path_kind(&path)? {
                None => {
                    ensure!(
                        initial && checkpoint.bytes == 0,
                        "curriculum spool {} disappeared after durable progress",
                        path.display()
                    );
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)?
                        .sync_all()?;
                }
                Some(PathKind::File) => ensure_regular_file(&path, "curriculum spool")?,
                Some(PathKind::Symlink) => {
                    anyhow::bail!("curriculum spool {} must not be a symlink", path.display())
                }
                Some(_) => {
                    anyhow::bail!("curriculum spool {} is not a regular file", path.display())
                }
            }
            let (actual_bytes, hasher) = hash_file_prefix(&path, checkpoint.bytes)?;
            ensure!(
                hex(&hasher.clone().finalize()) == checkpoint.sha256,
                "curriculum spool {} content changed before its durable boundary",
                path.display()
            );
            ensure!(
                actual_bytes >= checkpoint.bytes,
                "curriculum spool {} is shorter than its durable boundary",
                path.display()
            );
            if progress.state.spooling_complete {
                ensure!(
                    actual_bytes == checkpoint.bytes,
                    "completed curriculum spool {} has trailing data",
                    path.display()
                );
            } else if actual_bytes > checkpoint.bytes {
                OpenOptions::new()
                    .write(true)
                    .open(&path)?
                    .set_len(checkpoint.bytes)?;
                File::open(&path)?.sync_all()?;
            }
            let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
            file.seek(SeekFrom::Start(checkpoint.bytes))?;
            writers.push(SpoolWriter {
                path,
                writer: Some(BufWriter::new(file)),
                records: checkpoint.records,
                tokens: checkpoint.tokens,
                bytes: checkpoint.bytes,
                hasher,
            });
        }
        result.push(writers);
    }
    sync_directory(work_path)?;
    validate_spool_tree(work_path, &progress.state, true)?;
    Ok(result)
}

fn validate_spool_layout(
    composition: &CurriculumCompositionConfig,
    spools: &[Vec<SpoolCheckpoint>],
) -> Result<()> {
    ensure!(
        spools.len() == composition.stages.len(),
        "composition progress has the wrong stage spool cardinality"
    );
    for (stage_index, (stage, checkpoints)) in composition.stages.iter().zip(spools).enumerate() {
        ensure!(
            checkpoints.len() == stage.strata.len(),
            "composition progress has the wrong spool cardinality for stage `{}`",
            stage.name
        );
        for (stratum_index, checkpoint) in checkpoints.iter().enumerate() {
            ensure!(
                checkpoint.path == spool_path(stage_index, stratum_index),
                "composition progress contains an unexpected spool path"
            );
            ensure!(
                is_sha256(&checkpoint.sha256),
                "composition progress contains an invalid spool hash"
            );
            ensure!(
                (checkpoint.records == 0
                    && checkpoint.tokens == 0
                    && checkpoint.bytes == 0
                    && checkpoint.sha256 == empty_sha256())
                    || (checkpoint.records > 0 && checkpoint.tokens > 0 && checkpoint.bytes > 0),
                "composition progress contains inconsistent spool counters"
            );
        }
    }
    Ok(())
}

fn validate_work_entries(work_path: &Path, spools: &[Vec<SpoolCheckpoint>]) -> Result<()> {
    let mut expected = spools
        .iter()
        .flatten()
        .map(|spool| spool.path.as_str())
        .collect::<BTreeSet<_>>();
    expected.insert(WORK_IDENTITY_FILE);
    for entry in fs::read_dir(work_path)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "curriculum work path contains symlink {}",
            path.display()
        );
        ensure!(
            metadata.is_file(),
            "curriculum work path contains non-file {}",
            path.display()
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("curriculum work path contains a non-UTF-8 name"))?;
        ensure!(
            expected.contains(name.as_str()),
            "curriculum work path contains unexpected file `{name}`"
        );
    }
    Ok(())
}

fn validate_spool_tree(
    work_path: &Path,
    state: &CompositionProgressState,
    require_exact: bool,
) -> Result<()> {
    ensure_real_directory(work_path, "curriculum work path")?;
    validate_work_entries(work_path, &state.spools)?;
    let identity_path = work_path.join(WORK_IDENTITY_FILE);
    ensure_regular_file(&identity_path, "curriculum work identity")?;
    let identity: CompositionWorkIdentity = serde_json::from_slice(&fs::read(&identity_path)?)
        .context("invalid curriculum work identity")?;
    ensure!(
        identity
            == (CompositionWorkIdentity {
                version: COMPOSITION_PROGRESS_VERSION,
                build_id: state.build_id.clone(),
                composition_sha256: state.composition_sha256.clone(),
                source_manifest_sha256: state.source_manifest_sha256.clone(),
            }),
        "curriculum work identity changed"
    );
    for checkpoint in state.spools.iter().flatten() {
        let path = safe_join(work_path, &checkpoint.path)?;
        ensure_regular_file(&path, "curriculum spool")?;
        let (actual_bytes, hasher) = hash_file_prefix(&path, checkpoint.bytes)?;
        ensure!(
            hex(&hasher.finalize()) == checkpoint.sha256,
            "curriculum spool {} hash changed",
            path.display()
        );
        ensure!(
            !require_exact || actual_bytes == checkpoint.bytes,
            "curriculum spool {} length changed",
            path.display()
        );
    }
    Ok(())
}

fn hash_file_prefix(path: &Path, prefix_bytes: u64) -> Result<(u64, Sha256)> {
    ensure_regular_file(path, "curriculum content file")?;
    let actual_bytes = fs::metadata(path)?.len();
    ensure!(
        actual_bytes >= prefix_bytes,
        "curriculum file {} is shorter than its durable boundary",
        path.display()
    );
    let mut reader = BufReader::new(File::open(path)?);
    let mut remaining = prefix_bytes;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    while remaining > 0 {
        let limit = usize::try_from(remaining.min(buffer.len() as u64))?;
        let read = reader.read(&mut buffer[..limit])?;
        ensure!(
            read > 0,
            "curriculum file ended before its durable boundary"
        );
        hasher.update(&buffer[..read]);
        remaining -= u64::try_from(read)?;
    }
    Ok((actual_bytes, hasher))
}

fn prepare_current_stage_directory(
    stage_root: &Path,
    stage: &CurriculumStageConfig,
    progress: &StageProgress,
) -> Result<()> {
    match path_kind(stage_root)? {
        None => {
            ensure!(
                progress.shards.is_empty()
                    && progress.total_records == 0
                    && progress.total_tokens == 0,
                "stage `{}` directory disappeared after durable progress",
                stage.name
            );
            fs::create_dir(stage_root)?;
            sync_directory(
                stage_root
                    .parent()
                    .context("curriculum stage has no staging parent")?,
            )?;
        }
        Some(PathKind::Directory) => {
            ensure_real_directory(stage_root, "curriculum stage directory")?
        }
        Some(PathKind::Symlink) => anyhow::bail!(
            "curriculum stage directory {} must not be a symlink",
            stage_root.display()
        ),
        Some(_) => anyhow::bail!(
            "curriculum stage path {} is not a directory",
            stage_root.display()
        ),
    }

    let committed = progress
        .shards
        .iter()
        .map(|shard| shard.path.as_str())
        .collect::<BTreeSet<_>>();
    let next_shard = shard_path(progress.shards.len());
    let next_partial = partial_shard_path(progress.shards.len());
    for entry in fs::read_dir(stage_root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "curriculum stage contains symlink {}",
            path.display()
        );
        ensure!(
            metadata.is_file(),
            "curriculum stage contains non-file {}",
            path.display()
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("curriculum stage contains a non-UTF-8 name"))?;
        let allowed_manifest =
            name == "manifest.json" && progress.total_tokens >= stage.token_target.desired;
        ensure!(
            committed.contains(name.as_str())
                || name == next_shard
                || name == next_partial
                || allowed_manifest,
            "curriculum stage contains unexpected file `{name}`"
        );
    }
    Ok(())
}

fn verify_stage_replay(
    composition: &CurriculumCompositionConfig,
    work_path: &Path,
    stage: &CurriculumStageConfig,
    spools: &[SpoolCheckpoint],
    stage_root: &Path,
    progress: &StageProgress,
) -> Result<()> {
    progress.validate(stage, spools)?;
    let mut spool_readers = spools
        .iter()
        .map(|spool| {
            let path = safe_join(work_path, &spool.path)?;
            ensure_regular_file(&path, "curriculum spool")?;
            Ok(BufReader::new(File::open(path)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let tie_breakers = stage
        .strata
        .iter()
        .map(|stratum| {
            Sha256::digest(
                format!(
                    "{}\0{}\0{}\0{}",
                    composition.seed, progress.stage_index, stage.name, stratum.name
                )
                .as_bytes(),
            )
        })
        .collect::<Vec<_>>();
    let mut emitted_records = vec![0_u64; stage.strata.len()];
    let mut emitted_tokens = vec![0_u64; stage.strata.len()];
    let mut total_records = 0_u64;
    let mut total_tokens = 0_u64;
    let mut topic_counts = BTreeMap::new();
    let mut difficulty_counts = BTreeMap::new();
    let mut expected = Vec::new();
    let mut actual = Vec::new();

    for shard in &progress.shards {
        let path = safe_join(stage_root, &shard.path)?;
        ensure_regular_file(&path, "completed curriculum shard")?;
        let (bytes, hash) = hash_file(&path)?;
        ensure!(
            hash == shard.sha256,
            "completed curriculum shard {} hash changed",
            path.display()
        );
        ensure!(bytes > 0, "completed curriculum shard is empty");
        let mut reader = BufReader::new(File::open(&path)?);
        let mut shard_records = 0_u64;
        let mut shard_tokens = 0_u64;
        loop {
            actual.clear();
            if reader.read_until(b'\n', &mut actual)? == 0 {
                break;
            }
            trim_line_ending(&mut actual);
            ensure!(
                !actual.is_empty(),
                "completed curriculum shard has an empty row"
            );
            let selected = (0..stage.strata.len())
                .min_by(|left, right| {
                    weighted_progress_order(
                        emitted_tokens[*left],
                        stage.strata[*left].weight,
                        &tie_breakers[*left],
                        emitted_tokens[*right],
                        stage.strata[*right].weight,
                        &tie_breakers[*right],
                    )
                })
                .expect("validated stage has strata");
            expected.clear();
            ensure!(
                spool_readers[selected].read_until(b'\n', &mut expected)? > 0,
                "completed stage `{}` consumes beyond its durable spool",
                stage.name
            );
            trim_line_ending(&mut expected);
            ensure!(
                actual == expected,
                "completed stage `{}` violates deterministic weighted ordering",
                stage.name
            );
            let record: TokenizedCorpusRecord = serde_json::from_slice(&actual)?;
            record.validate()?;
            let matching = stage
                .strata
                .iter()
                .enumerate()
                .filter(|(_, stratum)| stratum.when.matches(&record))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            ensure!(
                stage.when.matches(&record) && matching.as_slice() == [selected],
                "completed stage `{}` contains a row outside its configured stratum",
                stage.name
            );
            let tokens = u64::try_from(record.tokens.len())?;
            emitted_records[selected] = checked_add(emitted_records[selected], 1)?;
            emitted_tokens[selected] = checked_add(emitted_tokens[selected], tokens)?;
            total_records = checked_add(total_records, 1)?;
            total_tokens = checked_add(total_tokens, tokens)?;
            shard_records = checked_add(shard_records, 1)?;
            shard_tokens = checked_add(shard_tokens, tokens)?;
            if let Some(topic) = record.topic {
                increment(&mut topic_counts, &topic)?;
            }
            if let Some(difficulty) = record.difficulty {
                increment(&mut difficulty_counts, &difficulty)?;
            }
        }
        ensure!(
            shard_records == shard.records && shard_tokens == shard.tokens,
            "completed curriculum shard {} counters changed",
            path.display()
        );
    }
    let offsets = spool_readers
        .iter_mut()
        .map(BufReader::stream_position)
        .collect::<std::io::Result<Vec<_>>>()?;
    ensure!(
        emitted_records == progress.emitted_records
            && emitted_tokens == progress.emitted_tokens
            && total_records == progress.total_records
            && total_tokens == progress.total_tokens
            && offsets == progress.spool_offsets
            && topic_counts == progress.topic_counts
            && difficulty_counts == progress.difficulty_counts,
        "stage `{}` durable counters or spool offsets changed",
        stage.name
    );
    Ok(())
}

fn validate_staging_tree(
    composition: &CurriculumCompositionConfig,
    source: &CorpusManifest,
    composition_sha256: &str,
    staging_root: &Path,
    progress: &CompositionProgress,
) -> Result<()> {
    composition.validate_progress(progress, source, composition_sha256)?;
    ensure_real_directory(staging_root, "curriculum staging path")?;
    ensure_regular_file(&staging_root.join(PROGRESS_FILE), "composition progress")?;
    reject_path_if_present_if_symlink(
        &staging_root.join(PROGRESS_TEMP_FILE),
        "composition progress temporary file",
    )?;

    let current_name = progress
        .state
        .current_stage
        .as_ref()
        .map(|current| composition.stages[current.stage_index].name.as_str());
    let completed_names = progress
        .state
        .completed_stages
        .iter()
        .map(|summary| summary.name.as_str())
        .collect::<BTreeSet<_>>();
    let all_stages_complete = progress.state.completed_stages.len() == composition.stages.len()
        && progress.state.current_stage.is_none();
    for entry in fs::read_dir(staging_root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "curriculum staging path contains symlink {}",
            path.display()
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("curriculum staging path has a non-UTF-8 name"))?;
        match name.as_str() {
            PROGRESS_FILE => ensure!(metadata.is_file(), "composition progress is not a file"),
            PROGRESS_TEMP_FILE => {
                anyhow::bail!("composition progress temporary file remains after recovery")
            }
            "curriculum-manifest.json" => ensure!(
                metadata.is_file() && all_stages_complete,
                "root curriculum manifest appeared before all stages completed"
            ),
            _ if completed_names.contains(name.as_str()) || current_name == Some(name.as_str()) => {
                ensure!(
                    metadata.is_dir(),
                    "curriculum stage `{name}` is not a directory"
                );
            }
            _ => anyhow::bail!("curriculum staging path contains unexpected entry `{name}`"),
        }
    }

    for (stage_index, summary) in progress.state.completed_stages.iter().enumerate() {
        verify_completed_stage(
            composition,
            source,
            composition_sha256,
            staging_root,
            stage_index,
            summary,
        )?;
    }
    if let Some(current) = &progress.state.current_stage {
        let stage = &composition.stages[current.stage_index];
        let stage_root = staging_root.join(&stage.name);
        prepare_current_stage_directory(&stage_root, stage, current)?;
        // Work-spool replay is performed by `compose_stage` after the work
        // tree itself has been verified.
    }

    let root_manifest_path = staging_root.join("curriculum-manifest.json");
    if path_kind(&root_manifest_path)?.is_some() {
        ensure_regular_file(&root_manifest_path, "root curriculum manifest")?;
        let actual: CurriculumManifest = serde_json::from_slice(&fs::read(&root_manifest_path)?)
            .context("invalid root curriculum manifest")?;
        let body = CurriculumManifestBody {
            version: composition.version,
            build_id: composition.build_id.clone(),
            config_sha256: composition_sha256.to_owned(),
            source_manifest_sha256: source.manifest_sha256.clone(),
            seed: composition.seed,
            stages: progress.state.completed_stages.clone(),
        };
        let expected = CurriculumManifest {
            manifest_sha256: canonical_json_sha256(&body)?,
            build: body,
        };
        ensure!(actual == expected, "root curriculum manifest changed");
        actual.verify(staging_root)?;
        if let Some(expected_hash) = &progress.state.root_manifest_sha256 {
            ensure!(
                expected_hash == &actual.manifest_sha256,
                "root curriculum manifest identity changed"
            );
        }
    } else {
        ensure!(
            progress.state.root_manifest_sha256.is_none(),
            "durable progress references a missing root curriculum manifest"
        );
    }
    Ok(())
}

fn verify_completed_stage(
    composition: &CurriculumCompositionConfig,
    source: &CorpusManifest,
    composition_sha256: &str,
    staging_root: &Path,
    stage_index: usize,
    summary: &CurriculumStageSummary,
) -> Result<()> {
    let stage = &composition.stages[stage_index];
    ensure!(
        summary.name == stage.name
            && summary.manifest_path == format!("{}/manifest.json", stage.name),
        "completed stage `{}` summary path changed",
        stage.name
    );
    let stage_root = staging_root.join(&stage.name);
    ensure_real_directory(&stage_root, "completed curriculum stage")?;
    let manifest_path = stage_root.join("manifest.json");
    ensure_regular_file(&manifest_path, "completed stage manifest")?;
    let manifest: CorpusManifest = serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("invalid stage manifest {}", manifest_path.display()))?;
    ensure!(
        manifest.manifest_sha256 == summary.manifest_sha256,
        "completed stage `{}` manifest identity changed",
        stage.name
    );
    let stage_identity = canonical_json_sha256(&serde_json::json!({
        "composition_sha256": composition_sha256,
        "source_manifest_sha256": source.manifest_sha256,
        "stage": stage,
    }))?;
    ensure!(
        manifest.build.config_sha256 == stage_identity
            && manifest.build.build_id == stage.name
            && manifest.build.config.build_id == stage.name,
        "completed stage `{}` configuration identity changed",
        stage.name
    );
    manifest.verify(&stage_root)?;
    ensure!(
        manifest.build.stats.emitted_views == summary.records
            && manifest.build.stats.exposure_tokens == summary.tokens,
        "completed stage `{}` summary counters changed",
        stage.name
    );
    ensure!(
        manifest.build.deduplicator.get("composition_sha256")
            == Some(&Value::String(composition_sha256.to_owned()))
            && manifest.build.deduplicator.get("source_manifest_sha256")
                == Some(&Value::String(source.manifest_sha256.clone()))
            && manifest.build.deduplicator.get("seed") == Some(&Value::from(composition.seed))
            && manifest.build.deduplicator.get("stage") == Some(&Value::String(stage.name.clone()))
            && manifest.build.deduplicator.get("stratum_records")
                == Some(&serde_json::to_value(&summary.stratum_records)?)
            && manifest.build.deduplicator.get("stratum_tokens")
                == Some(&serde_json::to_value(&summary.stratum_tokens)?),
        "completed stage `{}` composition metadata changed",
        stage.name
    );
    ensure_exact_stage_entries(&stage_root, &manifest.build.shards, true)?;
    Ok(())
}

fn ensure_exact_stage_entries(
    stage_root: &Path,
    shards: &[ShardManifest],
    include_manifest: bool,
) -> Result<()> {
    let mut expected = shards
        .iter()
        .map(|shard| shard.path.as_str())
        .collect::<BTreeSet<_>>();
    if include_manifest {
        expected.insert("manifest.json");
    }
    for entry in fs::read_dir(stage_root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "completed stage contains non-file or symlink {}",
            path.display()
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("completed stage contains a non-UTF-8 name"))?;
        ensure!(
            expected.remove(name.as_str()),
            "completed stage contains unexpected file `{name}`"
        );
    }
    ensure!(
        expected.is_empty(),
        "completed stage is missing immutable files"
    );
    Ok(())
}

fn publish_or_verify_stage_manifest(stage_root: &Path, expected: &CorpusManifest) -> Result<()> {
    let path = stage_root.join("manifest.json");
    match path_kind(&path)? {
        None => write_new_json(&path, expected)?,
        Some(PathKind::File) => {
            ensure_regular_file(&path, "stage manifest")?;
            let actual: CorpusManifest = serde_json::from_slice(&fs::read(&path)?)
                .context("invalid existing stage manifest")?;
            ensure!(
                actual.manifest_sha256 == expected.manifest_sha256
                    && canonical_json_sha256(&actual.build)?
                        == canonical_json_sha256(&expected.build)?,
                "existing stage manifest differs from deterministic resumed output"
            );
        }
        Some(PathKind::Symlink) => {
            anyhow::bail!("stage manifest {} must not be a symlink", path.display())
        }
        Some(_) => anyhow::bail!("stage manifest {} is not a regular file", path.display()),
    }
    sync_directory(stage_root)
}

fn publish_or_verify_root_manifest(
    staging_root: &Path,
    expected: &CurriculumManifest,
) -> Result<()> {
    let path = staging_root.join("curriculum-manifest.json");
    match path_kind(&path)? {
        None => write_new_json(&path, expected)?,
        Some(PathKind::File) => {
            ensure_regular_file(&path, "root curriculum manifest")?;
            let actual: CurriculumManifest = serde_json::from_slice(&fs::read(&path)?)
                .context("invalid existing root curriculum manifest")?;
            ensure!(
                actual == *expected,
                "existing root curriculum manifest differs from deterministic resumed output"
            );
        }
        Some(PathKind::Symlink) => anyhow::bail!(
            "root curriculum manifest {} must not be a symlink",
            path.display()
        ),
        Some(_) => anyhow::bail!(
            "root curriculum manifest {} is not a regular file",
            path.display()
        ),
    }
    sync_directory(staging_root)
}

#[allow(clippy::too_many_arguments)]
fn compose_stage(
    composition: &CurriculumCompositionConfig,
    source: &CorpusManifest,
    stage_index: usize,
    composition_sha256: &str,
    staging_root: &Path,
    work_path: &Path,
    progress_path: &Path,
    progress: &mut CompositionProgress,
) -> Result<CurriculumStageSummary> {
    let stage = &composition.stages[stage_index];
    let spool_checkpoints = progress.state.spools[stage_index].clone();
    let mut stage_progress = progress
        .state
        .current_stage
        .clone()
        .context("composition has no current stage progress")?;
    ensure!(
        stage_progress.stage_index == stage_index,
        "current composition stage changed"
    );
    stage_progress.validate(stage, &spool_checkpoints)?;
    let stage_root = staging_root.join(&stage.name);
    prepare_current_stage_directory(&stage_root, stage, &stage_progress)?;
    verify_stage_replay(
        composition,
        work_path,
        stage,
        &spool_checkpoints,
        &stage_root,
        &stage_progress,
    )?;

    let mut readers = spool_checkpoints
        .iter()
        .zip(&stage_progress.spool_offsets)
        .map(|(spool, offset)| {
            let path = safe_join(work_path, &spool.path)?;
            ensure_regular_file(&path, "curriculum spool")?;
            let mut reader = BufReader::new(File::open(&path)?);
            reader.seek(SeekFrom::Start(*offset))?;
            Ok(reader)
        })
        .collect::<Result<Vec<_>>>()?;
    let tie_breakers = stage
        .strata
        .iter()
        .map(|stratum| {
            Sha256::digest(
                format!(
                    "{}\0{}\0{}\0{}",
                    composition.seed, stage_index, stage.name, stratum.name
                )
                .as_bytes(),
            )
        })
        .collect::<Vec<_>>();
    let mut writer = RawShardWriter::resume(
        &stage_root,
        stage.sharding.max_tokens_per_shard,
        stage_progress.shards.len(),
    )?;
    let mut buffer = Vec::new();

    while stage_progress.total_tokens < stage.token_target.desired {
        let selected = (0..stage.strata.len())
            .min_by(|left, right| {
                weighted_progress_order(
                    stage_progress.emitted_tokens[*left],
                    stage.strata[*left].weight,
                    &tie_breakers[*left],
                    stage_progress.emitted_tokens[*right],
                    stage.strata[*right].weight,
                    &tie_breakers[*right],
                )
            })
            .expect("validated stage has strata");
        let prior_offset = readers[selected].stream_position()?;
        buffer.clear();
        let read = readers[selected].read_until(b'\n', &mut buffer)?;
        ensure!(
            read > 0,
            "stage `{}` stratum `{}` exhausted after {} tokens; available={} while stage target={} (adjust weights, predicates, or target)",
            stage.name,
            stage.strata[selected].name,
            stage_progress.emitted_tokens[selected],
            spool_checkpoints[selected].tokens,
            stage.token_target.desired
        );
        while buffer
            .last()
            .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
        {
            buffer.pop();
        }
        let record: TokenizedCorpusRecord = serde_json::from_slice(&buffer)?;
        record.validate()?;
        let record_tokens = u64::try_from(record.tokens.len())?;
        if writer.would_rotate(record_tokens) {
            readers[selected].seek(SeekFrom::Start(prior_offset))?;
            let shard = writer
                .close()?
                .context("shard rotation requested without an open shard")?;
            stage_progress.shards.push(shard);
            stage_progress.validate(stage, &spool_checkpoints)?;
            progress.state.current_stage = Some(stage_progress.clone());
            persist_progress(progress_path, progress)?;
            maybe_test_interrupt("stage_shard")?;
            continue;
        }
        let prospective = stage_progress
            .total_tokens
            .checked_add(record_tokens)
            .context("curriculum stage token count overflows u64")?;
        ensure!(
            prospective <= stage.token_target.maximum,
            "stage `{}` cannot reach desired target {} without exceeding maximum {} (next record has {} tokens)",
            stage.name,
            stage.token_target.desired,
            stage.token_target.maximum,
            record_tokens
        );
        writer.push(&buffer, record_tokens)?;
        stage_progress.spool_offsets[selected] = readers[selected].stream_position()?;
        stage_progress.emitted_records[selected] =
            checked_add(stage_progress.emitted_records[selected], 1)?;
        stage_progress.emitted_tokens[selected] =
            checked_add(stage_progress.emitted_tokens[selected], record_tokens)?;
        stage_progress.total_records = checked_add(stage_progress.total_records, 1)?;
        stage_progress.total_tokens = prospective;
        if let Some(topic) = record.topic {
            increment(&mut stage_progress.topic_counts, &topic)?;
        }
        if let Some(difficulty) = record.difficulty {
            increment(&mut stage_progress.difficulty_counts, &difficulty)?;
        }
    }
    if let Some(shard) = writer.close()? {
        stage_progress.shards.push(shard);
        stage_progress.validate(stage, &spool_checkpoints)?;
        progress.state.current_stage = Some(stage_progress.clone());
        persist_progress(progress_path, progress)?;
        maybe_test_interrupt("stage_shard")?;
    }
    ensure!(
        stage_progress.total_tokens >= stage.token_target.minimum,
        "stage `{}` emitted {} tokens below minimum {}",
        stage.name,
        stage_progress.total_tokens,
        stage.token_target.minimum
    );
    validate_fraction_deviation(
        stage,
        &stage_progress.emitted_tokens,
        stage_progress.total_tokens,
    )?;
    ensure!(
        !stage_progress.shards.is_empty(),
        "curriculum stage emitted no shards"
    );

    let mut corpus_config = source.build.config.clone();
    corpus_config.build_id = stage.name.clone();
    corpus_config.token_target = stage.token_target.clone();
    corpus_config.sharding = stage.sharding.clone();
    corpus_config.validate()?;
    let stage_identity = canonical_json_sha256(&serde_json::json!({
        "composition_sha256": composition_sha256,
        "source_manifest_sha256": source.manifest_sha256,
        "stage": stage,
    }))?;
    let stratum_records = stage
        .strata
        .iter()
        .zip(&stage_progress.emitted_records)
        .map(|(stratum, records)| (stratum.name.clone(), *records))
        .collect::<BTreeMap<_, _>>();
    let stratum_tokens = stage
        .strata
        .iter()
        .zip(&stage_progress.emitted_tokens)
        .map(|(stratum, tokens)| (stratum.name.clone(), *tokens))
        .collect::<BTreeMap<_, _>>();
    let available_stratum_records = stage
        .strata
        .iter()
        .zip(&spool_checkpoints)
        .map(|(stratum, spool)| (stratum.name.clone(), spool.records))
        .collect::<BTreeMap<_, _>>();
    let available_stratum_tokens = stage
        .strata
        .iter()
        .zip(&spool_checkpoints)
        .map(|(stratum, spool)| (stratum.name.clone(), spool.tokens))
        .collect::<BTreeMap<_, _>>();
    let stats = CorpusStageStats {
        unique_records: stage_progress.total_records,
        unique_tokens: stage_progress.total_tokens,
        emitted_views: stage_progress.total_records,
        exposure_tokens: stage_progress.total_tokens,
        ..CorpusStageStats::default()
    };
    let body = CorpusManifestBody {
        version: source.build.version,
        build_id: stage.name.clone(),
        config_sha256: stage_identity,
        config: corpus_config,
        discovery: source.build.discovery.clone(),
        materializer: source.build.materializer.clone(),
        deduplicator: serde_json::json!({
            "type": "stratified_curriculum_composition",
            "composition_sha256": composition_sha256,
            "source_manifest_sha256": source.manifest_sha256,
            "seed": composition.seed,
            "stage": stage.name,
            "stratum_records": stratum_records,
            "stratum_tokens": stratum_tokens,
            "available_stratum_records": available_stratum_records,
            "available_stratum_tokens": available_stratum_tokens,
        }),
        tokenizer: source.build.tokenizer.clone(),
        stats,
        desired_token_target_reached: true,
        topic_counts: stage_progress.topic_counts.clone(),
        difficulty_counts: stage_progress.difficulty_counts.clone(),
        shards: stage_progress.shards.clone(),
    };
    let manifest = CorpusManifest {
        manifest_sha256: canonical_json_sha256(&body)?,
        build: body,
    };
    publish_or_verify_stage_manifest(&stage_root, &manifest)?;
    manifest.verify(&stage_root)?;
    ensure_exact_stage_entries(&stage_root, &manifest.build.shards, true)?;
    sync_directory(&stage_root)?;

    Ok(CurriculumStageSummary {
        name: stage.name.clone(),
        manifest_path: format!("{}/manifest.json", stage.name),
        manifest_sha256: manifest.manifest_sha256,
        records: stage_progress.total_records,
        tokens: stage_progress.total_tokens,
        stratum_records,
        stratum_tokens,
    })
}

fn weighted_progress_order(
    left_tokens: u64,
    left_weight: u64,
    left_tie: &[u8],
    right_tokens: u64,
    right_weight: u64,
    right_tie: &[u8],
) -> Ordering {
    (u128::from(left_tokens) * u128::from(right_weight))
        .cmp(&(u128::from(right_tokens) * u128::from(left_weight)))
        .then_with(|| left_tie.cmp(right_tie))
}

fn validate_fraction_deviation(
    stage: &CurriculumStageConfig,
    emitted_tokens: &[u64],
    total_tokens: u64,
) -> Result<()> {
    let total_weight = stage
        .strata
        .iter()
        .try_fold(0_u64, |total, stratum| total.checked_add(stratum.weight))
        .context("curriculum stratum weights overflow u64")?;
    for (stratum, emitted) in stage.strata.iter().zip(emitted_tokens) {
        let expected = stratum.weight as f64 / total_weight as f64;
        let actual = *emitted as f64 / total_tokens as f64;
        ensure!(
            (actual - expected).abs() <= stage.max_fraction_deviation,
            "stage `{}` stratum `{}` token fraction {:.6} differs from configured {:.6} by more than {:.6}",
            stage.name,
            stratum.name,
            actual,
            expected,
            stage.max_fraction_deviation
        );
    }
    Ok(())
}

struct RawShardWriter {
    directory: PathBuf,
    max_tokens: u64,
    next_index: usize,
    open: Option<RawOpenShard>,
}

struct RawOpenShard {
    final_path: String,
    temporary_path: PathBuf,
    writer: BufWriter<File>,
    hasher: Sha256,
    records: u64,
    tokens: u64,
}

impl RawShardWriter {
    fn resume(directory: &Path, max_tokens: u64, next_index: usize) -> Result<Self> {
        ensure_real_directory(directory, "curriculum stage directory")?;
        let temporary_path = directory.join(partial_shard_path(next_index));
        match path_kind(&temporary_path)? {
            Some(PathKind::Symlink) => anyhow::bail!(
                "curriculum partial shard {} must not be a symlink",
                temporary_path.display()
            ),
            Some(PathKind::File) => fs::remove_file(&temporary_path)?,
            Some(_) => anyhow::bail!(
                "curriculum partial shard {} is not a regular file",
                temporary_path.display()
            ),
            None => {}
        }
        let uncommitted = directory.join(shard_path(next_index));
        match path_kind(&uncommitted)? {
            Some(PathKind::Symlink) => anyhow::bail!(
                "uncommitted curriculum shard {} must not be a symlink",
                uncommitted.display()
            ),
            Some(PathKind::File) => fs::remove_file(&uncommitted)?,
            Some(_) => anyhow::bail!(
                "uncommitted curriculum shard {} is not a regular file",
                uncommitted.display()
            ),
            None => {}
        }
        sync_directory(directory)?;
        Ok(Self {
            directory: directory.to_owned(),
            max_tokens,
            next_index,
            open: None,
        })
    }

    fn would_rotate(&self, tokens: u64) -> bool {
        self.open.as_ref().is_some_and(|shard| {
            shard.records > 0 && shard.tokens.saturating_add(tokens) > self.max_tokens
        })
    }

    fn push(&mut self, json: &[u8], tokens: u64) -> Result<()> {
        ensure!(
            !self.would_rotate(tokens),
            "curriculum shard must be checkpointed before rotation"
        );
        if self.open.is_none() {
            self.open()?;
        }
        let shard = self.open.as_mut().expect("shard is open");
        shard.writer.write_all(json)?;
        shard.writer.write_all(b"\n")?;
        shard.hasher.update(json);
        shard.hasher.update(b"\n");
        shard.records = checked_add(shard.records, 1)?;
        shard.tokens = checked_add(shard.tokens, tokens)?;
        Ok(())
    }

    fn open(&mut self) -> Result<()> {
        let final_path = shard_path(self.next_index);
        let temporary_path = self.directory.join(partial_shard_path(self.next_index));
        self.next_index += 1;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)?;
        self.open = Some(RawOpenShard {
            final_path,
            temporary_path,
            writer: BufWriter::new(file),
            hasher: Sha256::new(),
            records: 0,
            tokens: 0,
        });
        Ok(())
    }

    fn close(&mut self) -> Result<Option<ShardManifest>> {
        let Some(mut shard) = self.open.take() else {
            return Ok(None);
        };
        shard.writer.flush()?;
        shard.writer.get_ref().sync_all()?;
        drop(shard.writer);
        let final_path = self.directory.join(&shard.final_path);
        reject_path_if_present(&final_path, "uncommitted curriculum shard destination")?;
        fs::rename(&shard.temporary_path, &final_path)
            .with_context(|| format!("failed to seal curriculum shard {}", final_path.display()))?;
        sync_directory(&self.directory)?;
        Ok(Some(ShardManifest {
            path: shard.final_path,
            records: shard.records,
            tokens: shard.tokens,
            sha256: hex(&shard.hasher.finalize()),
        }))
    }
}

fn validate_token_target(target: &TokenTarget) -> Result<()> {
    ensure!(target.minimum > 0, "minimum token target must be positive");
    ensure!(
        target.minimum <= target.desired && target.desired <= target.maximum,
        "token targets must satisfy minimum <= desired <= maximum"
    );
    Ok(())
}

fn validate_component(label: &str, value: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{label} must not be empty");
    ensure!(
        !value.contains(['/', '\\']) && value != "." && value != "..",
        "{label} must be one safe path component"
    );
    Ok(())
}

fn resolve_relative(config_path: &Path, value: &Path) -> PathBuf {
    if value.is_absolute() {
        value.to_owned()
    } else {
        config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(value)
    }
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    ensure!(
        !relative.as_os_str().is_empty()
            && !relative.is_absolute()
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_) | Component::CurDir)),
        "unsafe relative path `{}`",
        relative.display()
    );
    Ok(root.join(relative))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PathKind {
    File,
    Directory,
    Symlink,
    Other,
}

fn path_kind(path: &Path) -> Result<Option<PathKind>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Ok(Some(PathKind::Symlink)),
        Ok(metadata) if metadata.is_file() => Ok(Some(PathKind::File)),
        Ok(metadata) if metadata.is_dir() => Ok(Some(PathKind::Directory)),
        Ok(_) => Ok(Some(PathKind::Other)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("failed to inspect {}", path.display())),
    }
}

fn create_and_validate_directory(path: &Path, label: &str) -> Result<()> {
    match path_kind(path)? {
        None => fs::create_dir_all(path)
            .with_context(|| format!("failed to create {label} {}", path.display()))?,
        Some(PathKind::Directory) => {}
        Some(PathKind::Symlink) => {
            anyhow::bail!("{label} {} must not be a symlink", path.display())
        }
        Some(_) => anyhow::bail!("{label} {} is not a directory", path.display()),
    }
    ensure_real_directory(path, label)
}

fn ensure_real_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} {} must be a real directory",
        path.display()
    );
    Ok(())
}

fn ensure_regular_file(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{label} {} must be a regular non-symlink file",
        path.display()
    );
    Ok(())
}

fn reject_path_if_present(path: &Path, label: &str) -> Result<()> {
    ensure!(
        path_kind(path)?.is_none(),
        "{label} {} already exists",
        path.display()
    );
    Ok(())
}

fn reject_path_if_present_if_symlink(path: &Path, label: &str) -> Result<()> {
    ensure!(
        path_kind(path)? != Some(PathKind::Symlink),
        "{label} {} must not be a symlink",
        path.display()
    );
    Ok(())
}

fn directory_is_empty(path: &Path) -> Result<bool> {
    Ok(fs::read_dir(path)?.next().transpose()?.is_none())
}

fn ensure_tree_has_no_symlinks(path: &Path, label: &str) -> Result<()> {
    ensure_real_directory(path, label)?;
    let mut pending = vec![path.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let entry_path = entry.path();
            let metadata = fs::symlink_metadata(&entry_path)?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "{label} contains symlink {}",
                entry_path.display()
            );
            if metadata.is_dir() {
                pending.push(entry_path);
            } else {
                ensure!(
                    metadata.is_file(),
                    "{label} contains unsupported entry {}",
                    entry_path.display()
                );
            }
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    ensure_real_directory(path, "directory to sync")?;
    File::open(path)?
        .sync_all()
        .with_context(|| format!("failed to sync directory {}", path.display()))
}

fn load_progress(path: &Path) -> Result<Option<CompositionProgress>> {
    let parent = path
        .parent()
        .context("composition progress path has no parent")?;
    ensure_real_directory(parent, "curriculum staging path")?;
    let temporary = parent.join(PROGRESS_TEMP_FILE);
    reject_path_if_present_if_symlink(&temporary, "composition progress temporary file")?;
    reject_path_if_present_if_symlink(path, "composition progress")?;
    match (path_kind(path)?, path_kind(&temporary)?) {
        (Some(PathKind::File), temporary_kind) => {
            if let Some(kind) = temporary_kind {
                ensure!(
                    kind == PathKind::File,
                    "composition progress temporary path is not a regular file"
                );
                fs::remove_file(&temporary)?;
                sync_directory(parent)?;
            }
            ensure_regular_file(path, "composition progress")?;
            let progress: CompositionProgress = serde_json::from_slice(&fs::read(path)?)
                .context("invalid composition progress JSON")?;
            progress.verify_hash()?;
            Ok(Some(progress))
        }
        (None, Some(PathKind::File)) => {
            ensure_regular_file(&temporary, "composition progress temporary file")?;
            let progress: CompositionProgress = serde_json::from_slice(&fs::read(&temporary)?)
                .context("invalid composition progress recovery JSON")?;
            progress.verify_hash()?;
            fs::rename(&temporary, path)?;
            sync_directory(parent)?;
            Ok(Some(progress))
        }
        (None, None) => Ok(None),
        (Some(_), _) => anyhow::bail!("composition progress is not a regular file"),
        (None, Some(_)) => {
            anyhow::bail!("composition progress temporary path is not a regular file")
        }
    }
}

fn persist_progress(path: &Path, progress: &mut CompositionProgress) -> Result<()> {
    let parent = path
        .parent()
        .context("composition progress path has no parent")?;
    ensure_real_directory(parent, "curriculum staging path")?;
    reject_path_if_present_if_symlink(path, "composition progress")?;
    if path_kind(path)?.is_some() {
        ensure_regular_file(path, "composition progress")?;
    }
    let temporary = parent.join(PROGRESS_TEMP_FILE);
    match path_kind(&temporary)? {
        None => {}
        Some(PathKind::File) => fs::remove_file(&temporary)?,
        Some(PathKind::Symlink) => anyhow::bail!(
            "composition progress temporary file {} must not be a symlink",
            temporary.display()
        ),
        Some(_) => anyhow::bail!(
            "composition progress temporary path {} is not a regular file",
            temporary.display()
        ),
    }
    progress.refresh_hash()?;
    write_new_json(&temporary, progress)?;
    fs::rename(&temporary, path).with_context(|| {
        format!(
            "failed to atomically publish composition progress {}",
            path.display()
        )
    })?;
    sync_directory(parent)
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    ensure_regular_file(path, "curriculum content file")?;
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes = checked_add(bytes, read)?;
    }
    Ok((bytes, hex(&hasher.finalize())))
}

fn trim_line_ending(line: &mut Vec<u8>) {
    while line
        .last()
        .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
    {
        line.pop();
    }
}

fn spool_path(stage_index: usize, stratum_index: usize) -> String {
    format!("stage-{stage_index:03}-stratum-{stratum_index:03}.jsonl")
}

fn shard_path(index: usize) -> String {
    format!("shard-{index:05}.tokens.jsonl")
}

fn partial_shard_path(index: usize) -> String {
    format!(".shard-{index:05}.tokens.jsonl.partial")
}

fn empty_sha256() -> String {
    hex(&Sha256::digest([]))
}

fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("failed to create {}", path.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn canonical_json_sha256(value: &impl Serialize) -> Result<String> {
    let value = serde_json::to_value(value)?;
    Ok(hex(&Sha256::digest(serde_json::to_vec(&canonicalize(
        &value,
    ))?)))
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checked_add(current: u64, amount: impl TryInto<u64>) -> Result<u64> {
    let amount = amount
        .try_into()
        .map_err(|_| anyhow::anyhow!("curriculum counter value does not fit u64"))?;
    current
        .checked_add(amount)
        .context("curriculum counter overflows u64")
}

fn checked_sum(values: &[u64], label: &str) -> Result<u64> {
    values
        .iter()
        .try_fold(0_u64, |total, value| total.checked_add(*value))
        .with_context(|| format!("{label} counters overflow u64"))
}

fn increment(counts: &mut BTreeMap<String, u64>, label: &str) -> Result<()> {
    let value = counts.entry(label.to_owned()).or_default();
    *value = value
        .checked_add(1)
        .context("curriculum classification count overflows u64")?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{
        ClassificationConfig, ComponentManifest, CorpusBuildConfig, DeduplicationConfig,
        DiscoveryConfig, DiscoveryQuery, NormalizationConfig, RepetitionConfig, SourceSnapshot,
        TokenizerSnapshot,
    };
    use serde_json::json;

    fn source_corpus(root: &Path) -> (PathBuf, CorpusManifest) {
        fs::create_dir_all(root).unwrap();
        let shard_path = root.join("shard-00000.tokens.jsonl");
        let mut bytes = Vec::new();
        let mut records = 0_u64;
        let mut tokens = 0_u64;
        for difficulty in ["foundation", "school", "university", "scholarly"] {
            for topic in ["language", "stem"] {
                for copy in 0..3 {
                    let record = TokenizedCorpusRecord {
                        record_key: format!("{difficulty}-{topic}-{copy}"),
                        uris: vec![format!("opaque+archive://{difficulty}/{topic}/{copy}")],
                        topic: Some(topic.to_owned()),
                        difficulty: Some(difficulty.to_owned()),
                        view: "source".to_owned(),
                        copy: 0,
                        metadata: BTreeMap::from([("capability".to_owned(), json!(topic))]),
                        tokens: vec![1, 2, 3, 4, 5],
                    };
                    serde_json::to_writer(&mut bytes, &record).unwrap();
                    bytes.push(b'\n');
                    records += 1;
                    tokens += 5;
                }
            }
        }
        fs::write(&shard_path, &bytes).unwrap();
        let config = CorpusBuildConfig {
            version: 2,
            build_id: "source".to_owned(),
            discovery: DiscoveryConfig {
                queries: vec![DiscoveryQuery {
                    name: "all".to_owned(),
                    text: "all".to_owned(),
                    limit: 1,
                    parameters: BTreeMap::new(),
                }],
                materialization_batch_size: 1,
            },
            normalization: NormalizationConfig::default(),
            deduplication: DeduplicationConfig::default(),
            classification: ClassificationConfig::default(),
            transformations: vec![],
            repetition: RepetitionConfig::default(),
            token_target: TokenTarget {
                minimum: tokens,
                desired: tokens,
                maximum: tokens,
            },
            sharding: ShardingConfig {
                max_tokens_per_shard: tokens,
            },
        };
        let component = ComponentManifest {
            name: "test".to_owned(),
            configuration: json!({}),
            snapshot: SourceSnapshot {
                provider: "test".to_owned(),
                revision: "1".to_owned(),
            },
        };
        let body = CorpusManifestBody {
            version: 2,
            build_id: "source".to_owned(),
            config_sha256: "source-config".to_owned(),
            config,
            discovery: component.clone(),
            materializer: component,
            deduplicator: json!({"type": "test"}),
            tokenizer: TokenizerSnapshot {
                implementation: "test".to_owned(),
                revision: "1".to_owned(),
                vocabulary_size: 10,
            },
            stats: CorpusStageStats {
                unique_records: records,
                unique_tokens: tokens,
                emitted_views: records,
                exposure_tokens: tokens,
                ..CorpusStageStats::default()
            },
            desired_token_target_reached: true,
            topic_counts: BTreeMap::new(),
            difficulty_counts: BTreeMap::new(),
            shards: vec![ShardManifest {
                path: "shard-00000.tokens.jsonl".to_owned(),
                records,
                tokens,
                sha256: hex(&Sha256::digest(&bytes)),
            }],
        };
        let manifest = CorpusManifest {
            manifest_sha256: canonical_json_sha256(&body).unwrap(),
            build: body,
        };
        let path = root.join("manifest.json");
        write_new_json(&path, &manifest).unwrap();
        (path, manifest)
    }

    fn config(path: &Path, source: &CorpusManifest) -> CurriculumCompositionConfig {
        CurriculumCompositionConfig {
            version: CURRICULUM_COMPOSITION_VERSION,
            build_id: "education".to_owned(),
            source_manifest: path.to_owned(),
            source_manifest_sha256: source.manifest_sha256.clone(),
            seed: 17,
            stages: ["foundation", "school", "university", "scholarly"]
                .into_iter()
                .map(|difficulty| CurriculumStageConfig {
                    name: difficulty.to_owned(),
                    when: CurriculumPredicate {
                        difficulties: vec![difficulty.to_owned()],
                        ..CurriculumPredicate::default()
                    },
                    unmatched: CurriculumUnmatchedPolicy::Error,
                    token_target: TokenTarget {
                        minimum: 20,
                        desired: 20,
                        maximum: 24,
                    },
                    sharding: ShardingConfig {
                        max_tokens_per_shard: 10,
                    },
                    max_fraction_deviation: 0.0,
                    strata: ["language", "stem"]
                        .into_iter()
                        .map(|topic| CurriculumStratumConfig {
                            name: topic.to_owned(),
                            weight: 1,
                            when: CurriculumPredicate {
                                topics: vec![topic.to_owned()],
                                ..CurriculumPredicate::default()
                            },
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn composes_four_reproducible_weighted_manifests_and_preserves_opaque_uris() {
        let source_dir = tempfile::tempdir().unwrap();
        let (source_path, source) = source_corpus(source_dir.path());
        let config = config(&source_path, &source);
        let config_path = source_dir.path().join("composition.json");
        write_new_json(&config_path, &config).unwrap();

        let first_output = tempfile::tempdir().unwrap();
        let first_work = tempfile::tempdir().unwrap();
        let (first_path, first) = config
            .run(&config_path, first_output.path(), first_work.path())
            .unwrap();
        first.verify(&first_path).unwrap();
        assert_eq!(first.build.stages.len(), 4);
        for stage in &first.build.stages {
            assert_eq!(stage.tokens, 20);
            assert_eq!(stage.stratum_tokens["language"], 10);
            assert_eq!(stage.stratum_tokens["stem"], 10);
        }
        let first_shard =
            fs::read_to_string(first_path.join("foundation/shard-00000.tokens.jsonl")).unwrap();
        assert!(first_shard.contains("opaque+archive://"));

        let second_output = tempfile::tempdir().unwrap();
        let second_work = tempfile::tempdir().unwrap();
        let (second_path, second) = config
            .run(&config_path, second_output.path(), second_work.path())
            .unwrap();
        second.verify(&second_path).unwrap();
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.build.stages, second.build.stages);
        assert_eq!(
            fs::read(first_path.join("foundation/shard-00000.tokens.jsonl")).unwrap(),
            fs::read(second_path.join("foundation/shard-00000.tokens.jsonl")).unwrap()
        );
    }

    #[test]
    fn overlapping_strata_fail_instead_of_using_configuration_order() {
        let source_dir = tempfile::tempdir().unwrap();
        let (source_path, source) = source_corpus(source_dir.path());
        let mut config = config(&source_path, &source);
        config.stages[0].strata[1].when = config.stages[0].strata[0].when.clone();
        let config_path = source_dir.path().join("ambiguous.json");
        write_new_json(&config_path, &config).unwrap();
        let output = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        let error = config
            .run(&config_path, output.path(), work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple strata"), "{error}");
        assert!(!output.path().join("education").exists());
    }

    #[test]
    fn production_composition_example_is_strict_and_has_four_stages() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("curriculum-composition.example.json");
        let config = CurriculumCompositionConfig::load(&path).unwrap();
        assert_eq!(
            config
                .stages
                .iter()
                .map(|stage| stage.name.as_str())
                .collect::<Vec<_>>(),
            ["foundation", "school", "university", "scholarly"]
        );
        assert_eq!(
            config
                .stages
                .iter()
                .map(|stage| stage.token_target.desired)
                .sum::<u64>(),
            14_000_000_000
        );
    }

    #[test]
    fn resumes_exactly_after_source_and_stage_shard_interruptions() {
        let source_dir = tempfile::tempdir().unwrap();
        let (source_path, source) = source_corpus(source_dir.path());
        let config = config(&source_path, &source);
        let config_path = source_dir.path().join("composition.json");
        write_new_json(&config_path, &config).unwrap();

        let clean_output = tempfile::tempdir().unwrap();
        let clean_work = tempfile::tempdir().unwrap();
        let (clean_path, clean_manifest) = config
            .run(&config_path, clean_output.path(), clean_work.path())
            .unwrap();

        let source_output = tempfile::tempdir().unwrap();
        let source_work = tempfile::tempdir().unwrap();
        set_test_interruption("source_shard", 1);
        let error = config
            .run(&config_path, source_output.path(), source_work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("source_shard"), "{error}");
        let identity = config.identity_sha256().unwrap();
        let work_path = source_work.path().join(format!(
            ".{}.{}.composition",
            config.build_id,
            &identity[..16]
        ));
        let progress_path = source_output
            .path()
            .join(".education.building")
            .join(PROGRESS_FILE);
        let progress = load_progress(&progress_path).unwrap().unwrap();
        assert_eq!(progress.state.next_source_shard, source.build.shards.len());
        assert!(!progress.state.spooling_complete);
        let spool = work_path.join(&progress.state.spools[0][0].path);
        OpenOptions::new()
            .append(true)
            .open(&spool)
            .unwrap()
            .write_all(b"uncommitted-crash-tail\n")
            .unwrap();

        let (resumed_source_path, resumed_source) = config
            .run(&config_path, source_output.path(), source_work.path())
            .unwrap();
        assert_eq!(
            resumed_source.manifest_sha256,
            clean_manifest.manifest_sha256
        );
        assert_eq!(
            fs::read(resumed_source_path.join("foundation/shard-00000.tokens.jsonl")).unwrap(),
            fs::read(clean_path.join("foundation/shard-00000.tokens.jsonl")).unwrap()
        );

        let stage_output = tempfile::tempdir().unwrap();
        let stage_work = tempfile::tempdir().unwrap();
        set_test_interruption("stage_shard", 1);
        let error = config
            .run(&config_path, stage_output.path(), stage_work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("stage_shard"), "{error}");
        let staging = stage_output.path().join(".education.building");
        let progress = load_progress(&staging.join(PROGRESS_FILE))
            .unwrap()
            .unwrap();
        let current = progress.state.current_stage.as_ref().unwrap();
        assert_eq!(current.stage_index, 0);
        assert_eq!(current.shards.len(), 1);
        fs::write(
            staging
                .join("foundation")
                .join(partial_shard_path(current.shards.len())),
            b"uncommitted output",
        )
        .unwrap();

        let (resumed_stage_path, resumed_stage) = config
            .run(&config_path, stage_output.path(), stage_work.path())
            .unwrap();
        assert_eq!(
            resumed_stage.manifest_sha256,
            clean_manifest.manifest_sha256
        );
        assert_eq!(
            fs::read(resumed_stage_path.join("foundation/shard-00001.tokens.jsonl")).unwrap(),
            fs::read(clean_path.join("foundation/shard-00001.tokens.jsonl")).unwrap()
        );
        assert!(!stage_output.path().join(".education.building").exists());

        let publish_output = tempfile::tempdir().unwrap();
        let publish_work = tempfile::tempdir().unwrap();
        set_test_interruption("root_ready", 1);
        let error = config
            .run(&config_path, publish_output.path(), publish_work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("root_ready"), "{error}");
        let work_path = publish_work.path().join(format!(
            ".{}.{}.composition",
            config.build_id,
            &identity[..16]
        ));
        fs::remove_file(work_path.join(spool_path(0, 0))).unwrap();
        let (_, resumed_publish) = config
            .run(&config_path, publish_output.path(), publish_work.path())
            .unwrap();
        assert_eq!(
            resumed_publish.manifest_sha256,
            clean_manifest.manifest_sha256
        );
    }

    #[test]
    fn resume_rejects_progress_config_and_committed_shard_tampering() {
        let source_dir = tempfile::tempdir().unwrap();
        let (source_path, source) = source_corpus(source_dir.path());
        let config = config(&source_path, &source);
        let config_path = source_dir.path().join("composition.json");
        write_new_json(&config_path, &config).unwrap();

        let output = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        set_test_interruption("stage_shard", 1);
        config
            .run(&config_path, output.path(), work.path())
            .unwrap_err();
        let staging = output.path().join(".education.building");
        OpenOptions::new()
            .append(true)
            .open(staging.join("foundation/shard-00000.tokens.jsonl"))
            .unwrap()
            .write_all(b"tamper")
            .unwrap();
        let error = config
            .run(&config_path, output.path(), work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("hash changed"), "{error}");

        let output = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        set_test_interruption("source_shard", 1);
        config
            .run(&config_path, output.path(), work.path())
            .unwrap_err();
        let progress_path = output
            .path()
            .join(".education.building")
            .join(PROGRESS_FILE);
        let mut progress_json: Value =
            serde_json::from_slice(&fs::read(&progress_path).unwrap()).unwrap();
        progress_json["state"]["next_source_shard"] = json!(0);
        fs::write(&progress_path, serde_json::to_vec(&progress_json).unwrap()).unwrap();
        let error = config
            .run(&config_path, output.path(), work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("content hash"), "{error}");

        let output = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        set_test_interruption("source_shard", 1);
        config
            .run(&config_path, output.path(), work.path())
            .unwrap_err();
        let mut changed = config.clone();
        changed.seed += 1;
        let error = changed
            .run(&config_path, output.path(), work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("configuration changed"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn resume_rejects_symlinked_spool() {
        use std::os::unix::fs::symlink;

        let source_dir = tempfile::tempdir().unwrap();
        let (source_path, source) = source_corpus(source_dir.path());
        let config = config(&source_path, &source);
        let config_path = source_dir.path().join("composition.json");
        write_new_json(&config_path, &config).unwrap();
        let output = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        set_test_interruption("source_shard", 1);
        config
            .run(&config_path, output.path(), work.path())
            .unwrap_err();

        let identity = config.identity_sha256().unwrap();
        let work_path = work.path().join(format!(
            ".{}.{}.composition",
            config.build_id,
            &identity[..16]
        ));
        let spool = work_path.join(spool_path(0, 0));
        fs::remove_file(&spool).unwrap();
        symlink(&source_path, &spool).unwrap();
        let error = config
            .run(&config_path, output.path(), work.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("symlink"), "{error}");
    }
}
