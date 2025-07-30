use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Component;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    ClassificationRule, CorpusBuildConfig, DeduplicationConfig, DiscoveryHit, MaterializedRecord,
    NormalizationConfig, RecordMaterializer, RecordPredicate, RepetitionConfig, SearchBackend,
    SourceSnapshot, TransformationConfig,
};

pub trait CorpusTokenizer: Send + Sync {
    fn snapshot(&self) -> TokenizerSnapshot;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSnapshot {
    pub implementation: String,
    pub revision: String,
    pub vocabulary_size: usize,
}

pub struct HermesCorpusTokenizer<'a> {
    tokenizer: &'a Tokenizer,
    revision: String,
}

impl<'a> HermesCorpusTokenizer<'a> {
    pub fn new(tokenizer: &'a Tokenizer, revision: impl Into<String>) -> Result<Self> {
        let revision = revision.into();
        ensure!(
            !revision.trim().is_empty(),
            "tokenizer revision must not be empty"
        );
        Ok(Self {
            tokenizer,
            revision,
        })
    }
}

impl CorpusTokenizer for HermesCorpusTokenizer<'_> {
    fn snapshot(&self) -> TokenizerSnapshot {
        TokenizerSnapshot {
            implementation: "hermes-tokenizer".to_owned(),
            revision: self.revision.clone(),
            vocabulary_size: self.tokenizer.vocab_size(),
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }
}

pub trait Deduplicator {
    /// Begin one atomic discovery-page or materialization-batch checkpoint.
    fn begin_checkpoint(&mut self) -> Result<()>;
    fn abort_checkpoint(&mut self) -> Result<()>;
    /// Load the content-hashed progress envelope bound to this build. An
    /// implementation must reject non-empty dedup state without a checkpoint.
    fn load_progress(&mut self, build_id: &str) -> Result<Option<Vec<u8>>>;
    /// Commit all dedup/discovery mutations together with the supplied progress
    /// envelope. The digest is over the exact envelope bytes.
    fn commit_progress(&mut self, build_id: &str, envelope: &[u8], sha256: &str) -> Result<()>;
    /// Insert or deterministically merge one discovery hit. Returns `true` for
    /// an already-staged key, regardless of whether the canonical merged value
    /// changed.
    fn stage_discovery_hit(&mut self, hit: &DiscoveryHit) -> Result<bool>;
    /// Read a bounded canonical record-key-ordered materialization batch.
    fn discovery_batch_after(
        &self,
        after_record_key: Option<&str>,
        limit: usize,
    ) -> Result<Vec<DiscoveryHit>>;
    /// Returns `true` if the record key or normalized text was already seen.
    /// The lookup and any following insertion occur inside the active pipeline
    /// checkpoint, so a target-boundary decision can leave the rejected record
    /// completely outside committed deduplication state.
    fn seen(&self, record_key: &str, normalized_text: &str) -> Result<bool>;
    /// Insert a record previously established to be unseen. Implementations
    /// must reject a duplicate rather than silently accepting a stale decision.
    fn insert_unseen(&mut self, record_key: &str, normalized_text: &str) -> Result<()>;
    fn seen_or_insert(&mut self, record_key: &str, normalized_text: &str) -> Result<bool> {
        if self.seen(record_key, normalized_text)? {
            return Ok(true);
        }
        self.insert_unseen(record_key, normalized_text)?;
        Ok(false)
    }
    fn configuration(&self) -> Value;
}

#[derive(Clone, Default)]
struct InMemoryDedupState {
    discovery_hits: BTreeMap<String, DiscoveryHit>,
    keys: HashSet<String>,
    texts: HashSet<[u8; 32]>,
    progress: Option<(String, Vec<u8>, String)>,
}

pub struct InMemoryDeduplicator {
    config: DeduplicationConfig,
    state: InMemoryDedupState,
    transaction_backup: Option<InMemoryDedupState>,
}

impl InMemoryDeduplicator {
    pub fn new(config: DeduplicationConfig) -> Result<Self> {
        config.validate_for_pipeline()?;
        Ok(Self {
            config,
            state: InMemoryDedupState::default(),
            transaction_backup: None,
        })
    }
}

impl Deduplicator for InMemoryDeduplicator {
    fn begin_checkpoint(&mut self) -> Result<()> {
        ensure!(
            self.transaction_backup.is_none(),
            "deduplicator checkpoint is already active"
        );
        self.transaction_backup = Some(self.state.clone());
        Ok(())
    }

    fn abort_checkpoint(&mut self) -> Result<()> {
        if let Some(backup) = self.transaction_backup.take() {
            self.state = backup;
        }
        Ok(())
    }

    fn load_progress(&mut self, build_id: &str) -> Result<Option<Vec<u8>>> {
        ensure!(
            self.transaction_backup.is_none(),
            "cannot load progress inside a deduplicator checkpoint"
        );
        match &self.state.progress {
            Some((stored_build_id, bytes, digest)) => {
                ensure!(
                    stored_build_id == build_id,
                    "deduplication catalog belongs to build `{stored_build_id}`, not `{build_id}`"
                );
                ensure!(
                    hex(&sha256(bytes)) == *digest,
                    "deduplication progress envelope hash mismatch"
                );
                Ok(Some(bytes.clone()))
            }
            None => {
                ensure!(
                    self.state.discovery_hits.is_empty()
                        && self.state.keys.is_empty()
                        && self.state.texts.is_empty(),
                    "deduplication catalog contains uncheckpointed state"
                );
                Ok(None)
            }
        }
    }

    fn commit_progress(&mut self, build_id: &str, envelope: &[u8], digest: &str) -> Result<()> {
        ensure!(
            self.transaction_backup.is_some(),
            "deduplicator checkpoint is not active"
        );
        ensure!(
            hex(&sha256(envelope)) == digest,
            "progress envelope digest mismatch"
        );
        self.state.progress = Some((build_id.to_owned(), envelope.to_vec(), digest.to_owned()));
        self.transaction_backup = None;
        Ok(())
    }

    fn stage_discovery_hit(&mut self, hit: &DiscoveryHit) -> Result<bool> {
        ensure!(
            self.transaction_backup.is_some(),
            "discovery mutations require an active checkpoint"
        );
        let duplicate = self.state.discovery_hits.contains_key(&hit.record_key);
        self.state
            .discovery_hits
            .entry(hit.record_key.clone())
            .and_modify(|existing| *existing = merge_discovery_hits(existing, hit))
            .or_insert_with(|| hit.clone());
        Ok(duplicate)
    }

    fn discovery_batch_after(
        &self,
        after_record_key: Option<&str>,
        limit: usize,
    ) -> Result<Vec<DiscoveryHit>> {
        ensure!(limit > 0, "discovery batch limit must be positive");
        Ok(self
            .state
            .discovery_hits
            .iter()
            .filter(|(key, _)| after_record_key.is_none_or(|after| key.as_str() > after))
            .take(limit)
            .map(|(_, hit)| hit.clone())
            .collect())
    }

    fn seen(&self, record_key: &str, normalized_text: &str) -> Result<bool> {
        ensure!(
            self.transaction_backup.is_some(),
            "deduplication lookups require an active checkpoint"
        );
        let text_hash = sha256(normalized_text.as_bytes());
        Ok(
            (self.config.by_record_key && self.state.keys.contains(record_key))
                || (self.config.by_normalized_text && self.state.texts.contains(&text_hash)),
        )
    }

    fn insert_unseen(&mut self, record_key: &str, normalized_text: &str) -> Result<()> {
        ensure!(
            self.transaction_backup.is_some(),
            "deduplication mutations require an active checkpoint"
        );
        ensure!(
            !self.seen(record_key, normalized_text)?,
            "cannot insert duplicate corpus record `{record_key}`"
        );
        let text_hash = sha256(normalized_text.as_bytes());
        if self.config.by_record_key {
            self.state.keys.insert(record_key.to_owned());
        }
        if self.config.by_normalized_text {
            self.state.texts.insert(text_hash);
        }
        Ok(())
    }

    fn configuration(&self) -> Value {
        serde_json::json!({
            "type": "in_memory",
            "by_record_key": self.config.by_record_key,
            "by_normalized_text": self.config.by_normalized_text,
        })
    }
}

/// Disk-backed exact deduplication for production-scale builds. Its database
/// belongs to an in-progress build and is not part of the immutable corpus.
pub struct SqliteDeduplicator {
    config: DeduplicationConfig,
    connection: Connection,
    checkpoint_active: bool,
}

impl SqliteDeduplicator {
    pub fn open(path: &Path, config: DeduplicationConfig) -> Result<Self> {
        config.validate_for_pipeline()?;
        let connection = Connection::open(path)
            .with_context(|| format!("failed to open deduplication catalog {}", path.display()))?;
        let legacy_table: i64 = connection.query_row(
            "SELECT count(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'discovery_keys'",
            [],
            |row| row.get(0),
        )?;
        ensure!(
            legacy_table == 0,
            "legacy deduplication catalog detected; use a clean work directory"
        );
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = FULL;
             CREATE TABLE IF NOT EXISTS corpus_keys (
                 record_key TEXT PRIMARY KEY NOT NULL
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS discovery_hits (
                 record_key TEXT PRIMARY KEY NOT NULL,
                 hit_json BLOB NOT NULL
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS corpus_texts (
                 text_sha256 BLOB PRIMARY KEY NOT NULL
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS build_progress (
                 build_id TEXT PRIMARY KEY NOT NULL,
                 envelope BLOB NOT NULL,
                 envelope_sha256 TEXT NOT NULL
             ) WITHOUT ROWID;",
        )?;
        Ok(Self {
            config,
            connection,
            checkpoint_active: false,
        })
    }
}

impl Deduplicator for SqliteDeduplicator {
    fn begin_checkpoint(&mut self) -> Result<()> {
        ensure!(
            !self.checkpoint_active,
            "deduplicator checkpoint is already active"
        );
        self.connection.execute_batch("BEGIN IMMEDIATE")?;
        self.checkpoint_active = true;
        Ok(())
    }

    fn abort_checkpoint(&mut self) -> Result<()> {
        if self.checkpoint_active {
            self.connection.execute_batch("ROLLBACK")?;
            self.checkpoint_active = false;
        }
        Ok(())
    }

    fn load_progress(&mut self, build_id: &str) -> Result<Option<Vec<u8>>> {
        ensure!(
            !self.checkpoint_active,
            "cannot load progress inside a deduplicator checkpoint"
        );
        let stored = self
            .connection
            .query_row(
                "SELECT build_id, envelope, envelope_sha256 FROM build_progress LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;
        match stored {
            Some((stored_build_id, bytes, digest)) => {
                ensure!(
                    stored_build_id == build_id,
                    "deduplication catalog belongs to build `{stored_build_id}`, not `{build_id}`"
                );
                ensure!(
                    hex(&sha256(&bytes)) == digest,
                    "deduplication progress envelope hash mismatch"
                );
                Ok(Some(bytes))
            }
            None => {
                let dirty: i64 = self.connection.query_row(
                    "SELECT (SELECT count(*) FROM discovery_hits)
                          + (SELECT count(*) FROM corpus_keys)
                          + (SELECT count(*) FROM corpus_texts)",
                    [],
                    |row| row.get(0),
                )?;
                ensure!(
                    dirty == 0,
                    "deduplication catalog contains uncheckpointed state; use a clean work directory"
                );
                Ok(None)
            }
        }
    }

    fn commit_progress(&mut self, build_id: &str, envelope: &[u8], digest: &str) -> Result<()> {
        ensure!(
            self.checkpoint_active,
            "deduplicator checkpoint is not active"
        );
        ensure!(
            hex(&sha256(envelope)) == digest,
            "progress envelope digest mismatch"
        );
        let publication = (|| -> Result<()> {
            self.connection.execute(
                "INSERT INTO build_progress(build_id, envelope, envelope_sha256)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(build_id) DO UPDATE SET
                    envelope = excluded.envelope,
                    envelope_sha256 = excluded.envelope_sha256",
                params![build_id, envelope, digest],
            )?;
            self.connection.execute_batch("COMMIT")?;
            Ok(())
        })();
        if publication.is_err() {
            let _ = self.connection.execute_batch("ROLLBACK");
        }
        self.checkpoint_active = false;
        publication
    }

    fn stage_discovery_hit(&mut self, hit: &DiscoveryHit) -> Result<bool> {
        ensure!(
            self.checkpoint_active,
            "discovery mutations require an active checkpoint"
        );
        let existing = self
            .connection
            .query_row(
                "SELECT hit_json FROM discovery_hits WHERE record_key = ?1",
                [&hit.record_key],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        let duplicate = existing.is_some();
        let merged = match existing {
            Some(bytes) => {
                let existing: DiscoveryHit = serde_json::from_slice(&bytes)
                    .context("deduplication catalog contains an invalid discovery hit")?;
                merge_discovery_hits(&existing, hit)
            }
            None => hit.clone(),
        };
        self.connection.execute(
            "INSERT INTO discovery_hits(record_key, hit_json) VALUES (?1, ?2)
             ON CONFLICT(record_key) DO UPDATE SET hit_json = excluded.hit_json",
            params![merged.record_key, serde_json::to_vec(&merged)?],
        )?;
        Ok(duplicate)
    }

    fn discovery_batch_after(
        &self,
        after_record_key: Option<&str>,
        limit: usize,
    ) -> Result<Vec<DiscoveryHit>> {
        ensure!(limit > 0, "discovery batch limit must be positive");
        let limit = i64::try_from(limit).context("discovery batch limit exceeds i64")?;
        let decode = |bytes: Vec<u8>| {
            serde_json::from_slice(&bytes)
                .context("deduplication catalog contains an invalid discovery hit")
        };
        match after_record_key {
            Some(after) => {
                let mut statement = self.connection.prepare(
                    "SELECT hit_json FROM discovery_hits
                     WHERE record_key > ?1 ORDER BY record_key LIMIT ?2",
                )?;
                statement
                    .query_map(params![after, limit], |row| row.get::<_, Vec<u8>>(0))?
                    .map(|row| decode(row?))
                    .collect()
            }
            None => {
                let mut statement = self
                    .connection
                    .prepare("SELECT hit_json FROM discovery_hits ORDER BY record_key LIMIT ?1")?;
                statement
                    .query_map([limit], |row| row.get::<_, Vec<u8>>(0))?
                    .map(|row| decode(row?))
                    .collect()
            }
        }
    }

    fn seen(&self, record_key: &str, normalized_text: &str) -> Result<bool> {
        ensure!(
            self.checkpoint_active,
            "deduplication lookups require an active checkpoint"
        );
        let text_hash = sha256(normalized_text.as_bytes());
        let duplicate_key = if self.config.by_record_key {
            self.connection
                .query_row(
                    "SELECT 1 FROM corpus_keys WHERE record_key = ?1",
                    [record_key],
                    |_| Ok(()),
                )
                .optional()?
                .is_some()
        } else {
            false
        };
        let duplicate_text = if self.config.by_normalized_text {
            self.connection
                .query_row(
                    "SELECT 1 FROM corpus_texts WHERE text_sha256 = ?1",
                    [text_hash.as_slice()],
                    |_| Ok(()),
                )
                .optional()?
                .is_some()
        } else {
            false
        };
        Ok(duplicate_key || duplicate_text)
    }

    fn insert_unseen(&mut self, record_key: &str, normalized_text: &str) -> Result<()> {
        ensure!(
            self.checkpoint_active,
            "deduplication mutations require an active checkpoint"
        );
        ensure!(
            !self.seen(record_key, normalized_text)?,
            "cannot insert duplicate corpus record `{record_key}`"
        );
        let text_hash = sha256(normalized_text.as_bytes());
        if self.config.by_record_key {
            self.connection.execute(
                "INSERT INTO corpus_keys(record_key) VALUES (?1)",
                [record_key],
            )?;
        }
        if self.config.by_normalized_text {
            self.connection.execute(
                "INSERT INTO corpus_texts(text_sha256) VALUES (?1)",
                params![text_hash.as_slice()],
            )?;
        }
        Ok(())
    }

    fn configuration(&self) -> Value {
        serde_json::json!({
            "type": "sqlite_exact",
            "by_record_key": self.config.by_record_key,
            "by_normalized_text": self.config.by_normalized_text,
        })
    }
}

// Keep validation private in config.rs while allowing construction of
// pipeline-owned implementations.
trait DeduplicationValidation {
    fn validate_for_pipeline(&self) -> Result<()>;
}

impl DeduplicationValidation for DeduplicationConfig {
    fn validate_for_pipeline(&self) -> Result<()> {
        ensure!(
            self.by_record_key || self.by_normalized_text,
            "deduplication must enable record-key or normalized-text matching"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComponentManifest {
    pub name: String,
    pub configuration: Value,
    pub snapshot: SourceSnapshot,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CorpusStageStats {
    pub discovery_pages: u64,
    pub discovered_hits: u64,
    pub duplicate_discovery_keys: u64,
    pub materialized_records: u64,
    pub normalization_rejections: u64,
    pub duplicate_records: u64,
    pub unique_records: u64,
    pub unique_tokens: u64,
    pub emitted_views: u64,
    pub exposure_tokens: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ShardManifest {
    pub path: String,
    pub records: u64,
    pub tokens: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifestBody {
    pub version: u32,
    pub build_id: String,
    pub config_sha256: String,
    pub config: CorpusBuildConfig,
    pub discovery: ComponentManifest,
    pub materializer: ComponentManifest,
    pub deduplicator: Value,
    pub tokenizer: TokenizerSnapshot,
    pub stats: CorpusStageStats,
    pub desired_token_target_reached: bool,
    pub topic_counts: BTreeMap<String, u64>,
    pub difficulty_counts: BTreeMap<String, u64>,
    pub shards: Vec<ShardManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    pub manifest_sha256: String,
    pub build: CorpusManifestBody,
}

const CORPUS_PROGRESS_VERSION: u32 = 2;
const CORPUS_PROGRESS_FILE: &str = "progress.json";
const CORPUS_PROGRESS_TEMP: &str = ".progress.json.tmp";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusBuildIdentity {
    build_id: String,
    config_sha256: String,
    search_name: String,
    discovery_snapshot: SourceSnapshot,
    materializer_name: String,
    materializer_snapshot: SourceSnapshot,
    tokenizer: TokenizerSnapshot,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
enum CorpusProgressPhase {
    Discovery {
        query_index: usize,
        offset: usize,
        /// Once a backend declares an exact total for a query, it must retain
        /// that value on every subsequent page and across resume.
        total_hits: Option<u64>,
    },
    Materialization {
        after_record_key: Option<String>,
    },
    Ready,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusProgress {
    version: u32,
    sequence: u64,
    identity: CorpusBuildIdentity,
    phase: CorpusProgressPhase,
    stats: CorpusStageStats,
    topic_counts: BTreeMap<String, u64>,
    difficulty_counts: BTreeMap<String, u64>,
    shards: Vec<ShardManifest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusProgressEnvelope {
    progress_sha256: String,
    progress: CorpusProgress,
}

impl CorpusManifest {
    /// Verify the content-addressed manifest and every immutable token shard.
    /// Training calls this before accepting a prepared corpus so a copied or
    /// partially replaced build cannot silently change a resumed run.
    pub fn verify(&self, root: &Path) -> Result<()> {
        self.build.config.validate()?;
        ensure!(
            self.manifest_sha256 == canonical_json_sha256(&serde_json::to_value(&self.build)?)?,
            "corpus manifest body hash does not match manifest_sha256"
        );
        ensure!(
            !self.build.shards.is_empty(),
            "corpus manifest has no shards"
        );
        let mut paths = HashSet::new();
        let mut records = 0u64;
        let mut tokens = 0u64;
        for shard in &self.build.shards {
            let relative = Path::new(&shard.path);
            ensure!(
                !relative.as_os_str().is_empty()
                    && !relative.is_absolute()
                    && relative.components().all(|component| matches!(
                        component,
                        Component::Normal(_) | Component::CurDir
                    )),
                "corpus shard path `{}` is not a safe relative path",
                shard.path
            );
            ensure!(
                paths.insert(shard.path.as_str()),
                "corpus manifest repeats shard `{}`",
                shard.path
            );
            ensure!(
                shard.records > 0 && shard.tokens > 0,
                "corpus shard `{}` is empty",
                shard.path
            );
            let path = root.join(relative);
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("corpus shard {} is missing", path.display()))?;
            ensure!(
                metadata.file_type().is_file(),
                "corpus shard {} is not a regular, non-symlink file",
                path.display()
            );
            let mut input = BufReader::new(
                File::open(&path)
                    .with_context(|| format!("failed to open corpus shard {}", path.display()))?,
            );
            let mut hasher = Sha256::new();
            let mut buffer = [0u8; 1024 * 1024];
            loop {
                let read = input.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            ensure!(
                hex(&hasher.finalize()) == shard.sha256,
                "corpus shard {} hash does not match its manifest",
                path.display()
            );
            records = records
                .checked_add(shard.records)
                .context("corpus shard record count overflows u64")?;
            tokens = tokens
                .checked_add(shard.tokens)
                .context("corpus shard token count overflows u64")?;
        }
        ensure!(
            records == self.build.stats.emitted_views,
            "corpus manifest emitted-view total does not match its shards"
        );
        ensure!(
            tokens == self.build.stats.exposure_tokens,
            "corpus manifest exposure-token total does not match its shards"
        );
        ensure!(
            self.build.desired_token_target_reached
                == (self.build.stats.unique_tokens >= self.build.config.token_target.desired),
            "corpus desired-token flag disagrees with token totals"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenizedCorpusRecord<'a> {
    record_key: &'a str,
    uris: &'a [String],
    topic: Option<&'a str>,
    difficulty: Option<&'a str>,
    view: &'a str,
    copy: usize,
    metadata: &'a BTreeMap<String, Value>,
    tokens: &'a [u32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaterializationBatchOutcome {
    Continue,
    TargetBoundaryReached,
}

pub struct CorpusPipeline<'a> {
    config: CorpusBuildConfig,
    search: &'a dyn SearchBackend,
    materializer: &'a dyn RecordMaterializer,
    tokenizer: &'a dyn CorpusTokenizer,
    deduplicator: &'a mut dyn Deduplicator,
}

impl<'a> CorpusPipeline<'a> {
    pub fn new(
        config: CorpusBuildConfig,
        search: &'a dyn SearchBackend,
        materializer: &'a dyn RecordMaterializer,
        tokenizer: &'a dyn CorpusTokenizer,
        deduplicator: &'a mut dyn Deduplicator,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            search.page_size() > 0,
            "search backend page size must be positive"
        );
        Ok(Self {
            config,
            search,
            materializer,
            tokenizer,
            deduplicator,
        })
    }

    /// Build or resume `output_root/<build_id>`. Discovery pages and canonical
    /// record-key-ordered materialization batches are independent transactions.
    /// Each transaction commits dedup state and a content-hashed cursor in the
    /// same durable store after its output shards have been synced.
    pub fn run(&mut self, output_root: &Path) -> Result<(PathBuf, CorpusManifest)> {
        let discovery_snapshot = self.search.snapshot()?;
        let materializer_snapshot = self.materializer.snapshot()?;
        let discovery_configuration = self.search.configuration()?;
        let materializer_configuration = self.materializer.configuration()?;
        let config_value = serde_json::to_value(&self.config)?;
        let config_sha256 = canonical_json_sha256(&serde_json::json!({
            "corpus": config_value,
            "search": discovery_configuration,
            "materializer": materializer_configuration,
            "deduplicator": self.deduplicator.configuration(),
            "tokenizer": self.tokenizer.snapshot(),
        }))?;
        let identity = CorpusBuildIdentity {
            build_id: self.config.build_id.clone(),
            config_sha256: config_sha256.clone(),
            search_name: self.search.name().to_owned(),
            discovery_snapshot: discovery_snapshot.clone(),
            materializer_name: self.materializer.name().to_owned(),
            materializer_snapshot: materializer_snapshot.clone(),
            tokenizer: self.tokenizer.snapshot(),
        };

        fs::create_dir_all(output_root).with_context(|| {
            format!(
                "failed to create corpus output root {}",
                output_root.display()
            )
        })?;
        let final_path = output_root.join(&self.config.build_id);
        let staging_path = output_root.join(format!(".{}.building", self.config.build_id));
        if final_path.exists() {
            ensure!(
                !staging_path.exists(),
                "published corpus and staging directory both exist for build `{}`",
                self.config.build_id
            );
            return load_published_manifest(&final_path, &identity);
        }
        ensure_real_directory_or_create(&staging_path)?;

        let stored = self.deduplicator.load_progress(&self.config.build_id)?;
        let mut progress = match stored {
            Some(bytes) => {
                let envelope = decode_progress_envelope(&bytes)?;
                ensure!(
                    envelope.progress.identity == identity,
                    "corpus resume identity changed; configuration, source snapshots, or tokenizer differ"
                );
                reconcile_progress_mirror(&staging_path, &bytes, &envelope)?;
                envelope.progress
            }
            None => {
                ensure_initial_staging(&staging_path)?;
                let mut progress = CorpusProgress {
                    version: CORPUS_PROGRESS_VERSION,
                    sequence: 0,
                    identity: identity.clone(),
                    phase: CorpusProgressPhase::Discovery {
                        query_index: 0,
                        offset: 0,
                        total_hits: None,
                    },
                    stats: CorpusStageStats::default(),
                    topic_counts: BTreeMap::new(),
                    difficulty_counts: BTreeMap::new(),
                    shards: Vec::new(),
                };
                self.deduplicator.begin_checkpoint()?;
                self.commit_progress(&staging_path, &mut progress)?;
                progress
            }
        };
        validate_progress(&progress, &identity, &self.config.discovery.queries)?;
        reconcile_staging_files(&staging_path, &progress.shards)?;
        let mut writer = ImmutableShardWriter::resume(
            &staging_path,
            self.config.sharding.max_tokens_per_shard,
            progress.shards.clone(),
        )?;

        loop {
            match progress.phase.clone() {
                CorpusProgressPhase::Discovery {
                    query_index,
                    offset,
                    total_hits,
                } => {
                    if query_index == self.config.discovery.queries.len() {
                        self.deduplicator.begin_checkpoint()?;
                        progress.phase = CorpusProgressPhase::Materialization {
                            after_record_key: None,
                        };
                        self.commit_progress(&staging_path, &mut progress)?;
                        continue;
                    }
                    let query = &self.config.discovery.queries[query_index];
                    ensure!(
                        offset < query.limit,
                        "corpus progress offset exceeds query `{}` limit",
                        query.name
                    );
                    let request_limit = self.search.page_size().min(query.limit - offset);
                    let page = self
                        .search
                        .discover(query, offset, request_limit)
                        .with_context(|| format!("discovery query `{}` failed", query.name))?;
                    ensure!(
                        page.snapshot == discovery_snapshot,
                        "search backend returned an unpinned snapshot for query `{}`",
                        query.name
                    );
                    let returned = page.hits.len();
                    ensure!(
                        returned <= request_limit,
                        "search backend returned {returned} hits for query `{}` after a request limit of {request_limit}",
                        query.name
                    );
                    let next_offset = offset
                        .checked_add(returned)
                        .context("search pagination offset overflows usize")?;
                    let next_offset_u64 = u64::try_from(next_offset)
                        .context("search pagination offset exceeds u64")?;
                    let observed_total_hits = match (total_hits, page.total_hits) {
                        (Some(expected), Some(actual)) => {
                            ensure!(
                                actual == expected,
                                "search backend changed total_hits for query `{}` from {expected} to {actual}",
                                query.name
                            );
                            Some(expected)
                        }
                        (Some(expected), None) => {
                            anyhow::bail!(
                                "search backend omitted total_hits for query `{}` after declaring {expected}",
                                query.name
                            )
                        }
                        (None, total) => total,
                    };
                    if let Some(total) = observed_total_hits {
                        ensure!(
                            next_offset_u64 <= total,
                            "search backend returned hits through offset {next_offset} for query `{}`, beyond total_hits {total}",
                            query.name
                        );
                        let configured_limit = u64::try_from(query.limit)
                            .context("discovery query limit exceeds u64")?;
                        let expected_end = total.min(configured_limit);
                        ensure!(
                            returned == request_limit || next_offset_u64 >= expected_end,
                            "search backend returned a short page for query `{}` at offset {offset} while total_hits {total} promises more results",
                            query.name
                        );
                    }
                    let query_complete = returned == 0
                        || returned < request_limit
                        || next_offset >= query.limit
                        || observed_total_hits.is_some_and(|total| next_offset_u64 >= total);

                    self.deduplicator.begin_checkpoint()?;
                    let page_result = (|| -> Result<()> {
                        progress.stats.discovery_pages =
                            checked_add(progress.stats.discovery_pages, 1)?;
                        progress.stats.discovered_hits =
                            checked_add(progress.stats.discovered_hits, returned)?;
                        for hit in &page.hits {
                            if self.deduplicator.stage_discovery_hit(hit)? {
                                progress.stats.duplicate_discovery_keys =
                                    checked_add(progress.stats.duplicate_discovery_keys, 1)?;
                            }
                        }
                        progress.phase = if query_complete {
                            CorpusProgressPhase::Discovery {
                                query_index: query_index + 1,
                                offset: 0,
                                total_hits: None,
                            }
                        } else {
                            CorpusProgressPhase::Discovery {
                                query_index,
                                offset: next_offset,
                                total_hits: observed_total_hits,
                            }
                        };
                        Ok(())
                    })();
                    if let Err(error) = page_result {
                        let _ = self.deduplicator.abort_checkpoint();
                        return Err(error);
                    }
                    self.commit_progress(&staging_path, &mut progress)?;
                }
                CorpusProgressPhase::Materialization { after_record_key } => {
                    // A prior materialization transaction may have reached the
                    // target immediately before an interruption. Do not fetch
                    // another canonical batch merely to discover that the run
                    // was already complete.
                    if progress.stats.unique_tokens >= self.config.token_target.desired {
                        self.deduplicator.begin_checkpoint()?;
                        progress.phase = CorpusProgressPhase::Ready;
                        if let Err(error) = self.commit_progress(&staging_path, &mut progress) {
                            let _ = self.deduplicator.abort_checkpoint();
                            return Err(error);
                        }
                        continue;
                    }
                    ensure!(
                        self.materializer.snapshot()? == materializer_snapshot,
                        "record materializer snapshot changed during corpus build"
                    );
                    let hits = self.deduplicator.discovery_batch_after(
                        after_record_key.as_deref(),
                        self.config.discovery.materialization_batch_size,
                    )?;
                    if hits.is_empty() {
                        self.deduplicator.begin_checkpoint()?;
                        progress.phase = CorpusProgressPhase::Ready;
                        self.commit_progress(&staging_path, &mut progress)?;
                        continue;
                    }
                    let last_record_key = hits
                        .last()
                        .expect("non-empty discovery batch")
                        .record_key
                        .clone();
                    self.deduplicator.begin_checkpoint()?;
                    let batch_result = (|| -> Result<()> {
                        let outcome = self.process_batch(
                            &hits,
                            &mut writer,
                            &mut progress.stats,
                            &mut progress.topic_counts,
                            &mut progress.difficulty_counts,
                        )?;
                        progress.shards = writer.checkpoint()?;
                        sync_directory(&staging_path)?;
                        progress.phase = match outcome {
                            MaterializationBatchOutcome::Continue => {
                                CorpusProgressPhase::Materialization {
                                    after_record_key: Some(last_record_key),
                                }
                            }
                            MaterializationBatchOutcome::TargetBoundaryReached => {
                                CorpusProgressPhase::Ready
                            }
                        };
                        self.commit_progress(&staging_path, &mut progress)
                    })();
                    if let Err(error) = batch_result {
                        let _ = self.deduplicator.abort_checkpoint();
                        return Err(error);
                    }
                }
                CorpusProgressPhase::Ready => break,
            }
        }

        let shards = writer.finish()?;
        ensure!(
            shards == progress.shards,
            "corpus shard cursor changed after readiness"
        );
        ensure!(
            self.config
                .token_target
                .accepts(progress.stats.unique_tokens),
            "corpus unique-token count {} is outside configured [{}, {}] target",
            progress.stats.unique_tokens,
            self.config.token_target.minimum,
            self.config.token_target.maximum
        );
        ensure!(
            self.materializer.snapshot()? == materializer_snapshot,
            "record materializer snapshot changed during corpus build"
        );

        let body = CorpusManifestBody {
            version: self.config.version,
            build_id: self.config.build_id.clone(),
            config_sha256,
            config: self.config.clone(),
            discovery: ComponentManifest {
                name: self.search.name().to_owned(),
                configuration: discovery_configuration,
                snapshot: discovery_snapshot,
            },
            materializer: ComponentManifest {
                name: self.materializer.name().to_owned(),
                configuration: materializer_configuration,
                snapshot: materializer_snapshot,
            },
            deduplicator: self.deduplicator.configuration(),
            tokenizer: self.tokenizer.snapshot(),
            stats: progress.stats.clone(),
            desired_token_target_reached: progress.stats.unique_tokens
                >= self.config.token_target.desired,
            topic_counts: progress.topic_counts.clone(),
            difficulty_counts: progress.difficulty_counts.clone(),
            shards,
        };
        let manifest = CorpusManifest {
            manifest_sha256: canonical_json_sha256(&serde_json::to_value(&body)?)?,
            build: body,
        };
        let manifest_path = staging_path.join("manifest.json");
        remove_regular_if_exists(&staging_path.join(CORPUS_PROGRESS_FILE))?;
        write_new_synced_json(&manifest_path, &manifest)?;
        sync_directory(&staging_path)?;
        fs::rename(&staging_path, &final_path).with_context(|| {
            format!(
                "failed to publish immutable corpus {}",
                final_path.display()
            )
        })?;
        sync_directory(output_root)?;
        Ok((final_path, manifest))
    }

    fn commit_progress(&mut self, staging: &Path, progress: &mut CorpusProgress) -> Result<()> {
        progress.sequence = progress
            .sequence
            .checked_add(1)
            .context("corpus progress sequence overflows u64")?;
        let bytes = encode_progress_envelope(progress)?;
        let digest = hex(&sha256(&bytes));
        self.deduplicator
            .commit_progress(&self.config.build_id, &bytes, &digest)?;
        write_progress_mirror(staging, &bytes)
    }

    fn process_batch(
        &mut self,
        hit_batch: &[DiscoveryHit],
        writer: &mut ImmutableShardWriter,
        stats: &mut CorpusStageStats,
        topic_counts: &mut BTreeMap<String, u64>,
        difficulty_counts: &mut BTreeMap<String, u64>,
    ) -> Result<MaterializationBatchOutcome> {
        let materialized = self.materializer.materialize(hit_batch).with_context(|| {
            format!(
                "{} failed to materialize {} discovery hits",
                self.materializer.name(),
                hit_batch.len()
            )
        })?;
        stats.materialized_records = checked_add(stats.materialized_records, materialized.len())?;
        let ordered = order_materialized(hit_batch, materialized)?;
        for mut record in ordered {
            let Some(normalized) = normalize(&record.text, &self.config.normalization) else {
                stats.normalization_rejections = checked_add(stats.normalization_rejections, 1)?;
                continue;
            };
            if self.deduplicator.seen(&record.record_key, &normalized)? {
                stats.duplicate_records = checked_add(stats.duplicate_records, 1)?;
                continue;
            }
            record.text = normalized;
            let canonical_tokens = self
                .tokenizer
                .encode(&record.text)
                .with_context(|| format!("failed to tokenize `{}`", record.record_key))?;
            ensure!(
                !canonical_tokens.is_empty(),
                "tokenizer emitted no tokens for `{}`",
                record.record_key
            );
            let prospective_unique_tokens =
                checked_add(stats.unique_tokens, canonical_tokens.len())?;
            if prospective_unique_tokens > self.config.token_target.maximum {
                ensure!(
                    stats.unique_tokens >= self.config.token_target.minimum,
                    "corpus has no legal canonical prefix: {} unique tokens are below minimum {}, but adding record `{}` ({} tokens) would exceed maximum {}",
                    stats.unique_tokens,
                    self.config.token_target.minimum,
                    record.record_key,
                    canonical_tokens.len(),
                    self.config.token_target.maximum
                );
                return Ok(MaterializationBatchOutcome::TargetBoundaryReached);
            }

            self.deduplicator
                .insert_unseen(&record.record_key, &record.text)?;
            let topic = classify(
                &record,
                &self.config.classification.topic_rules,
                self.config.classification.default_topic.as_deref(),
            );
            let difficulty = classify(
                &record,
                &self.config.classification.difficulty_rules,
                self.config.classification.default_difficulty.as_deref(),
            );
            if let Some(topic) = &topic {
                increment_label(topic_counts, topic)?;
            }
            if let Some(difficulty) = &difficulty {
                increment_label(difficulty_counts, difficulty)?;
            }
            stats.unique_records = checked_add(stats.unique_records, 1)?;
            stats.unique_tokens = prospective_unique_tokens;

            let views = render_views(
                &record,
                topic.as_deref(),
                difficulty.as_deref(),
                &self.config.transformations,
                &self.config.repetition,
            );
            for view in views {
                let tokens = if view.name == "source" && view.copy == 0 {
                    canonical_tokens.clone()
                } else {
                    self.tokenizer.encode(&view.text).with_context(|| {
                        format!(
                            "failed to tokenize view `{}` for `{}`",
                            view.name, record.record_key
                        )
                    })?
                };
                ensure!(
                    !tokens.is_empty(),
                    "tokenizer emitted no tokens for view `{}` of `{}`",
                    view.name,
                    record.record_key
                );
                writer.push(TokenizedCorpusRecord {
                    record_key: &record.record_key,
                    uris: &record.uris,
                    topic: topic.as_deref(),
                    difficulty: difficulty.as_deref(),
                    view: &view.name,
                    copy: view.copy,
                    metadata: &record.metadata,
                    tokens: &tokens,
                })?;
                stats.emitted_views = checked_add(stats.emitted_views, 1)?;
                stats.exposure_tokens = checked_add(stats.exposure_tokens, tokens.len())?;
            }
            if stats.unique_tokens >= self.config.token_target.desired {
                return Ok(MaterializationBatchOutcome::TargetBoundaryReached);
            }
        }
        Ok(MaterializationBatchOutcome::Continue)
    }
}

fn merge_discovery_hits(left: &DiscoveryHit, right: &DiscoveryHit) -> DiscoveryHit {
    debug_assert_eq!(left.record_key, right.record_key);
    let mut uris = left
        .uris
        .iter()
        .chain(&right.uris)
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    uris.shrink_to_fit();
    let mut metadata = left.metadata.clone();
    for (name, value) in &right.metadata {
        metadata
            .entry(name.clone())
            .and_modify(|current| {
                let current_key = serde_json::to_vec(&canonicalize(current))
                    .expect("JSON value serialization cannot fail");
                let candidate_key = serde_json::to_vec(&canonicalize(value))
                    .expect("JSON value serialization cannot fail");
                if candidate_key < current_key {
                    *current = value.clone();
                }
            })
            .or_insert_with(|| value.clone());
    }
    let inline_text = left
        .inline_text
        .iter()
        .chain(right.inline_text.iter())
        .min()
        .cloned();
    DiscoveryHit {
        record_key: left.record_key.clone(),
        score: left.score.max(right.score),
        uris,
        metadata,
        inline_text,
    }
}

fn encode_progress_envelope(progress: &CorpusProgress) -> Result<Vec<u8>> {
    let progress_sha256 = canonical_json_sha256(&serde_json::to_value(progress)?)?;
    let envelope = CorpusProgressEnvelope {
        progress_sha256,
        progress: progress.clone(),
    };
    let mut bytes = serde_json::to_vec_pretty(&envelope)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn decode_progress_envelope(bytes: &[u8]) -> Result<CorpusProgressEnvelope> {
    let envelope: CorpusProgressEnvelope =
        serde_json::from_slice(bytes).context("invalid corpus progress journal")?;
    ensure!(
        envelope.progress.version == CORPUS_PROGRESS_VERSION,
        "unsupported corpus progress version {}",
        envelope.progress.version
    );
    ensure!(
        envelope.progress_sha256
            == canonical_json_sha256(&serde_json::to_value(&envelope.progress)?)?,
        "corpus progress journal content hash mismatch"
    );
    Ok(envelope)
}

fn validate_progress(
    progress: &CorpusProgress,
    identity: &CorpusBuildIdentity,
    queries: &[super::DiscoveryQuery],
) -> Result<()> {
    ensure!(
        progress.version == CORPUS_PROGRESS_VERSION,
        "unsupported corpus progress version {}",
        progress.version
    );
    ensure!(
        &progress.identity == identity,
        "corpus progress identity does not match this build"
    );
    match &progress.phase {
        CorpusProgressPhase::Discovery {
            query_index,
            offset,
            total_hits,
        } => {
            ensure!(
                *query_index <= queries.len(),
                "corpus progress query index is out of range"
            );
            if let Some(query) = queries.get(*query_index) {
                ensure!(
                    *offset < query.limit,
                    "corpus progress offset is outside query `{}`",
                    query.name
                );
                if let Some(total) = total_hits {
                    ensure!(
                        u64::try_from(*offset).is_ok_and(|offset| offset <= *total),
                        "corpus progress offset exceeds total_hits for query `{}`",
                        query.name
                    );
                }
            } else {
                ensure!(*offset == 0, "completed discovery cursor has an offset");
                ensure!(
                    total_hits.is_none(),
                    "completed discovery cursor retains total_hits"
                );
            }
        }
        CorpusProgressPhase::Materialization { after_record_key } => ensure!(
            after_record_key
                .as_deref()
                .is_none_or(|key| !key.is_empty()),
            "corpus materialization cursor is empty"
        ),
        CorpusProgressPhase::Ready => {}
    }
    let shard_records = progress.shards.iter().try_fold(0u64, |total, shard| {
        total
            .checked_add(shard.records)
            .context("progress shard record count overflows u64")
    })?;
    let shard_tokens = progress.shards.iter().try_fold(0u64, |total, shard| {
        total
            .checked_add(shard.tokens)
            .context("progress shard token count overflows u64")
    })?;
    ensure!(
        shard_records == progress.stats.emitted_views
            && shard_tokens == progress.stats.exposure_tokens,
        "corpus progress shard totals disagree with stage statistics"
    );
    Ok(())
}

fn ensure_real_directory_or_create(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "corpus staging path {} is not a real directory",
                path.display()
            );
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => fs::create_dir(path)
            .with_context(|| format!("failed to create corpus staging path {}", path.display())),
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect corpus staging path {}", path.display())),
    }
}

fn ensure_initial_staging(path: &Path) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == CORPUS_PROGRESS_TEMP {
            remove_regular_if_exists(&entry.path())?;
            continue;
        }
        anyhow::bail!(
            "corpus staging directory {} contains state without a committed journal: {}",
            path.display(),
            entry.path().display()
        );
    }
    Ok(())
}

fn reconcile_progress_mirror(
    staging: &Path,
    authoritative_bytes: &[u8],
    authoritative: &CorpusProgressEnvelope,
) -> Result<()> {
    let path = staging.join(CORPUS_PROGRESS_FILE);
    match fs::symlink_metadata(&path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "corpus progress journal is not a regular file"
            );
            let current_bytes = fs::read(&path)?;
            if current_bytes == authoritative_bytes {
                return Ok(());
            }
            let current = decode_progress_envelope(&current_bytes)?;
            ensure!(
                current.progress.identity == authoritative.progress.identity
                    && current.progress.sequence < authoritative.progress.sequence,
                "corpus progress mirror differs from its authoritative dedup checkpoint"
            );
            write_progress_mirror(staging, authoritative_bytes)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            write_progress_mirror(staging, authoritative_bytes)
        }
        Err(error) => Err(error).context("failed to inspect corpus progress journal"),
    }
}

fn reconcile_staging_files(staging: &Path, committed: &[ShardManifest]) -> Result<()> {
    let expected = committed
        .iter()
        .map(|shard| shard.path.as_str())
        .collect::<BTreeSet<_>>();
    for entry in fs::read_dir(staging)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("corpus staging contains a non-UTF-8 entry"))?;
        if name == CORPUS_PROGRESS_FILE {
            continue;
        }
        if name == CORPUS_PROGRESS_TEMP || name == "manifest.json" {
            remove_regular_if_exists(&entry.path())?;
            continue;
        }
        if expected.contains(name.as_str()) {
            continue;
        }
        ensure!(
            is_shard_name(&name),
            "corpus staging contains unexpected entry `{name}`"
        );
        remove_regular_if_exists(&entry.path())?;
    }
    for (index, shard) in committed.iter().enumerate() {
        ensure!(
            shard.path == format!("shard-{index:05}.tokens.jsonl"),
            "corpus progress has a non-contiguous shard path `{}`",
            shard.path
        );
        let path = staging.join(&shard.path);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("committed corpus shard {} is missing", path.display()))?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "committed corpus shard {} is not a regular file",
            path.display()
        );
        ensure!(
            file_sha256(&path)? == shard.sha256,
            "committed corpus shard {} hash mismatch",
            path.display()
        );
    }
    sync_directory(staging)
}

fn is_shard_name(name: &str) -> bool {
    name.len() == "shard-00000.tokens.jsonl".len()
        && name.starts_with("shard-")
        && name.ends_with(".tokens.jsonl")
        && name.as_bytes()[6..11]
            .iter()
            .all(|byte| byte.is_ascii_digit())
}

fn write_progress_mirror(staging: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = staging.join(CORPUS_PROGRESS_TEMP);
    remove_regular_if_exists(&temporary)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, staging.join(CORPUS_PROGRESS_FILE))?;
    sync_directory(staging)
}

fn remove_regular_if_exists(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "refusing to remove non-regular corpus staging entry {}",
                path.display()
            );
            fs::remove_file(path)?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to inspect {}", path.display())),
    }
}

fn write_new_synced_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn load_published_manifest(
    root: &Path,
    identity: &CorpusBuildIdentity,
) -> Result<(PathBuf, CorpusManifest)> {
    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "published corpus {} is not a real directory",
        root.display()
    );
    let manifest_path = root.join("manifest.json");
    let metadata = fs::symlink_metadata(&manifest_path)?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "published corpus manifest is not a regular file"
    );
    let manifest: CorpusManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    manifest.verify(root)?;
    ensure!(
        manifest.build.build_id == identity.build_id
            && manifest.build.config_sha256 == identity.config_sha256
            && manifest.build.discovery.name == identity.search_name
            && manifest.build.discovery.snapshot == identity.discovery_snapshot
            && manifest.build.materializer.name == identity.materializer_name
            && manifest.build.materializer.snapshot == identity.materializer_snapshot
            && manifest.build.tokenizer == identity.tokenizer,
        "published corpus identity differs from this requested build"
    );
    Ok((root.to_owned(), manifest))
}

fn order_materialized(
    hits: &[DiscoveryHit],
    records: Vec<MaterializedRecord>,
) -> Result<Vec<MaterializedRecord>> {
    let mut by_key: HashMap<_, _> = records
        .into_iter()
        .map(|record| (record.record_key.clone(), record))
        .collect();
    let mut ordered = Vec::with_capacity(by_key.len());
    for hit in hits {
        if let Some(record) = by_key.remove(&hit.record_key) {
            ordered.push(record);
        }
    }
    ensure!(
        by_key.is_empty(),
        "materializer returned records absent from its discovery batch"
    );
    Ok(ordered)
}

fn normalize(text: &str, config: &NormalizationConfig) -> Option<String> {
    if config.reject_replacement_character && text.contains('\u{fffd}') {
        return None;
    }
    let mut text = if config.normalize_newlines {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_owned()
    };
    if config.collapse_horizontal_whitespace {
        let mut normalized = String::with_capacity(text.len());
        let mut horizontal = false;
        for character in text.chars() {
            if character != '\n' && character.is_whitespace() {
                horizontal = true;
            } else {
                if horizontal && !normalized.ends_with([' ', '\n']) {
                    normalized.push(' ');
                }
                horizontal = false;
                normalized.push(character);
            }
        }
        text = normalized;
    }
    if config.normalize_newlines {
        let maximum_newlines = config.max_consecutive_blank_lines.saturating_add(1);
        let mut normalized = String::with_capacity(text.len());
        let mut newlines = 0usize;
        for character in text.chars() {
            if character == '\n' {
                newlines += 1;
                if newlines <= maximum_newlines {
                    normalized.push(character);
                }
            } else {
                newlines = 0;
                normalized.push(character);
            }
        }
        text = normalized;
    }
    if config.trim {
        text = text.trim().to_owned();
    }
    (text.chars().count() >= config.minimum_characters).then_some(text)
}

fn classify(
    record: &MaterializedRecord,
    rules: &[ClassificationRule],
    default: Option<&str>,
) -> Option<String> {
    let lowercase = record.text.to_lowercase();
    rules
        .iter()
        .filter_map(|rule| {
            let matched_any = rule
                .any_terms
                .iter()
                .filter(|term| lowercase.contains(&term.to_lowercase()))
                .count();
            let matched_all = rule
                .all_terms
                .iter()
                .filter(|term| lowercase.contains(&term.to_lowercase()))
                .count();
            let matches = (rule.any_terms.is_empty() || matched_any > 0)
                && matched_all == rule.all_terms.len()
                && rule
                    .none_terms
                    .iter()
                    .all(|term| !lowercase.contains(&term.to_lowercase()))
                && rule
                    .metadata_equals
                    .iter()
                    .all(|(name, expected)| record.metadata.get(name) == Some(expected));
            matches.then_some((rule, rule.metadata_equals.len(), matched_any + matched_all))
        })
        .max_by(
            |(left, left_metadata, left_terms), (right, right_metadata, right_terms)| {
                left.priority
                    .cmp(&right.priority)
                    .then(left_metadata.cmp(right_metadata))
                    .then(left_terms.cmp(right_terms))
                    // Lexicographically smaller labels win a complete tie.
                    .then_with(|| right.label.cmp(&left.label))
            },
        )
        .map(|(rule, _, _)| rule.label.clone())
        .or_else(|| default.map(str::to_owned))
}

struct RenderedView {
    name: String,
    copy: usize,
    text: String,
}

fn render_views(
    record: &MaterializedRecord,
    topic: Option<&str>,
    difficulty: Option<&str>,
    transforms: &[TransformationConfig],
    repetition: &RepetitionConfig,
) -> Vec<RenderedView> {
    let base_copies = repetition
        .topic_copies
        .get(topic.unwrap_or_default())
        .into_iter()
        .chain(
            repetition
                .difficulty_copies
                .get(difficulty.unwrap_or_default()),
        )
        .copied()
        .fold(repetition.base_copies, usize::max);
    let mut views = (0..base_copies)
        .map(|copy| RenderedView {
            name: "source".to_owned(),
            copy,
            text: record.text.clone(),
        })
        .collect::<Vec<_>>();
    for transform in transforms {
        if !predicate_matches(&transform.when, record, topic, difficulty) {
            continue;
        }
        for copy in 0..transform.copies {
            views.push(RenderedView {
                name: transform.name.clone(),
                copy,
                text: render_template(&transform.template, record, topic, difficulty),
            });
        }
    }
    views.truncate(repetition.max_copies_per_record);
    views
}

fn predicate_matches(
    predicate: &RecordPredicate,
    record: &MaterializedRecord,
    topic: Option<&str>,
    difficulty: Option<&str>,
) -> bool {
    (predicate.topics.is_empty()
        || topic.is_some_and(|value| predicate.topics.iter().any(|v| v == value)))
        && (predicate.difficulties.is_empty()
            || difficulty.is_some_and(|value| predicate.difficulties.iter().any(|v| v == value)))
        && predicate
            .metadata_equals
            .iter()
            .all(|(name, expected)| record.metadata.get(name) == Some(expected))
}

fn render_template(
    template: &str,
    record: &MaterializedRecord,
    topic: Option<&str>,
    difficulty: Option<&str>,
) -> String {
    let mut rendered = template
        .replace("${text}", &record.text)
        .replace("${topic}", topic.unwrap_or_default())
        .replace("${difficulty}", difficulty.unwrap_or_default());
    for (name, value) in &record.metadata {
        let replacement = value
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| value.to_string());
        rendered = rendered.replace(&format!("${{metadata.{name}}}"), &replacement);
    }
    rendered
}

struct OpenShard {
    path: String,
    writer: BufWriter<File>,
    hasher: Sha256,
    records: u64,
    tokens: u64,
}

struct ImmutableShardWriter {
    directory: PathBuf,
    max_tokens: u64,
    next_index: usize,
    open: Option<OpenShard>,
    completed: Vec<ShardManifest>,
}

impl ImmutableShardWriter {
    fn resume(directory: &Path, max_tokens: u64, completed: Vec<ShardManifest>) -> Result<Self> {
        for (index, shard) in completed.iter().enumerate() {
            ensure!(
                shard.path == format!("shard-{index:05}.tokens.jsonl"),
                "cannot resume a non-contiguous corpus shard sequence"
            );
        }
        Ok(Self {
            directory: directory.to_owned(),
            max_tokens,
            next_index: completed.len(),
            open: None,
            completed,
        })
    }

    fn push(&mut self, record: TokenizedCorpusRecord<'_>) -> Result<()> {
        let token_count = u64::try_from(record.tokens.len())?;
        if self.open.as_ref().is_some_and(|shard| {
            shard.records > 0 && shard.tokens.saturating_add(token_count) > self.max_tokens
        }) {
            self.close()?;
        }
        if self.open.is_none() {
            self.open()?;
        }
        let bytes = serde_json::to_vec(&record)?;
        let shard = self.open.as_mut().expect("shard was opened");
        shard.writer.write_all(&bytes)?;
        shard.writer.write_all(b"\n")?;
        shard.hasher.update(&bytes);
        shard.hasher.update(b"\n");
        shard.records = checked_add(shard.records, 1)?;
        shard.tokens = checked_add(shard.tokens, record.tokens.len())?;
        Ok(())
    }

    fn open(&mut self) -> Result<()> {
        let path = format!("shard-{:05}.tokens.jsonl", self.next_index);
        self.next_index += 1;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(&path))?;
        self.open = Some(OpenShard {
            path,
            writer: BufWriter::new(file),
            hasher: Sha256::new(),
            records: 0,
            tokens: 0,
        });
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        let Some(mut shard) = self.open.take() else {
            return Ok(());
        };
        shard.writer.flush()?;
        shard.writer.get_ref().sync_all()?;
        self.completed.push(ShardManifest {
            path: shard.path,
            records: shard.records,
            tokens: shard.tokens,
            sha256: hex(&shard.hasher.finalize()),
        });
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<Vec<ShardManifest>> {
        self.close()?;
        Ok(self.completed.clone())
    }

    fn finish(mut self) -> Result<Vec<ShardManifest>> {
        self.close()?;
        ensure!(!self.completed.is_empty(), "corpus emitted no shards");
        Ok(self.completed)
    }
}

fn canonical_json_sha256(value: &Value) -> Result<String> {
    let canonical = canonicalize(value);
    Ok(hex(&Sha256::digest(serde_json::to_vec(&canonical)?)))
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(values) => {
            let sorted: BTreeMap<_, _> = values
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect();
            serde_json::to_value(sorted).expect("BTreeMap JSON serialization cannot fail")
        }
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
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

fn checked_add(current: u64, amount: impl TryInto<u64>) -> Result<u64> {
    let amount = amount
        .try_into()
        .map_err(|_| anyhow::anyhow!("corpus counter value does not fit u64"))?;
    current
        .checked_add(amount)
        .context("corpus counter overflows u64")
}

fn increment_label(counts: &mut BTreeMap<String, u64>, label: &str) -> Result<()> {
    let count = counts.entry(label.to_owned()).or_default();
    *count = count
        .checked_add(1)
        .context("classification count overflows u64")?;
    Ok(())
}
