use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Component;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;
use rusqlite::{Connection, OptionalExtension, params};
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::config::MAX_DISCOVERY_BATCH_SIZE;
use super::{
    ClassificationRule, CorpusBuildConfig, DeduplicationConfig, DiscoveryHit, MaterializedRecord,
    NormalizationConfig, RecordMaterializer, RecordPredicate, RepetitionConfig, SearchBackend,
    SourceSnapshot, TransformationConfig,
};

pub trait CorpusTokenizer: Send + Sync {
    /// Identity of the exact encoding implementation and vocabulary. It must
    /// change whenever [`Self::encode`] can produce different token IDs.
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

impl TokenizerSnapshot {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.implementation.trim().is_empty(),
            "tokenizer implementation must not be empty"
        );
        ensure!(
            !self.revision.trim().is_empty(),
            "tokenizer revision must not be empty"
        );
        ensure!(
            self.vocabulary_size > 0,
            "tokenizer vocabulary_size must be positive"
        );
        ensure!(
            u64::try_from(self.vocabulary_size).is_ok_and(|size| size <= u64::from(u32::MAX) + 1),
            "tokenizer vocabulary_size exceeds the u32 token-id space"
        );
        Ok(())
    }
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
        config.validate()?;
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
        config.validate()?;
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
    /// Discovery hits the materializer did not return. Only reachable with
    /// `require_every_record: false`, which tolerates rows that disappeared
    /// between discovery and materialization.
    #[serde(default)]
    pub unmaterialized_records: u64,
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
const MAX_CORPUS_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const VERIFIED_SHARD_READ_BUFFER_BYTES: usize = 1024 * 1024;

/// Corpus rows are deliberately bounded independently of shard size. A
/// missing JSONL delimiter must not make verification or curriculum
/// composition allocate an entire multi-gigabyte shard. The limit still
/// accommodates millions of token IDs in one record, far above training
/// context sizes.
pub(crate) const MAX_CORPUS_JSONL_RECORD_BYTES: usize = 64 * 1024 * 1024;
const MAX_MATERIALIZED_TEXT_BYTES: usize = MAX_CORPUS_JSONL_RECORD_BYTES;
const MAX_RENDERED_VIEW_TEXT_BYTES: usize = MAX_CORPUS_JSONL_RECORD_BYTES;
/// Bound transformation amplification for one source record. Each individual
/// view is bounded above, but many templates can otherwise retain hundreds of
/// large rendered strings (and their token vectors) at once.
const MAX_RETAINED_RENDERED_TEXT_BYTES: usize = 256 * 1024 * 1024;

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

/// One manifest generation whose exact shard files were authenticated once.
///
/// The retained root descriptor and per-shard identities are the data-plane
/// counterpart of the manifest hash used in a training run signature. Shards
/// are opened relative to that root and matched to their verified identities,
/// while publication checks reject persistent replacement. This avoids both a
/// second hash pass and one retained descriptor per shard.
pub struct AuthenticatedCorpus {
    root: PinnedCorpusRoot,
    manifest_identity: StableCorpusFileIdentity,
    manifest: CorpusManifest,
    shards: Vec<PinnedCorpusShard>,
}

struct PinnedCorpusShard {
    relative_path: String,
    display_path: PathBuf,
    identity: StableCorpusFileIdentity,
}

struct VerifiedCorpusShardReader<'a> {
    file: &'a mut File,
    expected: &'a StableCorpusFileIdentity,
    path: &'a Path,
    buffer: Box<[u8]>,
    buffered_start: usize,
    buffered_end: usize,
}

impl Read for VerifiedCorpusShardReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.buffered_start == self.buffered_end {
            let read = self.file.read(&mut self.buffer)?;
            let observed = StableCorpusFileIdentity::from_metadata(&self.file.metadata()?);
            if observed != *self.expected {
                return Err(std::io::Error::other(format!(
                    "authenticated corpus shard {} changed in place while it was streamed",
                    self.path.display()
                )));
            }
            self.buffered_start = 0;
            self.buffered_end = read;
            if read == 0 {
                return Ok(0);
            }
        }
        // Bytes only enter this buffer immediately before a successful
        // identity check. A later in-place write cannot change the captured
        // bytes, so metadata syscalls scale with MiB chunks instead of the
        // caller's (usually 8 KiB) parser buffer.
        let available = &self.buffer[self.buffered_start..self.buffered_end];
        let copied = available.len().min(buffer.len());
        buffer[..copied].copy_from_slice(&available[..copied]);
        self.buffered_start += copied;
        Ok(copied)
    }
}

struct PinnedCorpusRoot {
    path: PathBuf,
    identity: fs::Metadata,
    #[cfg(unix)]
    directory: File,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StableCorpusFileIdentity {
    length: u64,
    modified: Option<std::time::SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

impl StableCorpusFileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            length: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            mode: metadata.mode(),
            #[cfg(unix)]
            changed_seconds: metadata.ctime(),
            #[cfg(unix)]
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

impl PinnedCorpusRoot {
    fn open(path: &Path) -> Result<Self> {
        let identity = fs::symlink_metadata(path)
            .with_context(|| format!("failed to inspect corpus root {}", path.display()))?;
        ensure!(
            identity.is_dir() && !identity.file_type().is_symlink(),
            "corpus root {} is not a real directory",
            path.display()
        );
        #[cfg(unix)]
        let directory = {
            let directory = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
                .open(path)
                .with_context(|| {
                    format!("failed to securely open corpus root {}", path.display())
                })?;
            ensure!(
                same_file_object(&directory.metadata()?, &identity),
                "corpus root {} changed while it was opened",
                path.display()
            );
            directory
        };
        Ok(Self {
            path: path.to_owned(),
            identity,
            #[cfg(unix)]
            directory,
        })
    }

    fn open_file(&self, relative: &str, label: &str) -> Result<File> {
        let mut components = Path::new(relative).components();
        ensure!(
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
            "{label} path `{relative}` is not one safe relative component"
        );
        #[cfg(unix)]
        {
            let name = CString::new(relative).context("corpus file name contains NUL")?;
            // SAFETY: the root descriptor remains owned by `self`; openat
            // returns a newly owned descriptor and O_NOFOLLOW rejects a
            // symlink at the authenticated leaf.
            let descriptor = unsafe {
                libc::openat(
                    self.directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
                    0,
                )
            };
            if descriptor < 0 {
                return Err(std::io::Error::last_os_error()).with_context(|| {
                    format!(
                        "failed to open {label} {}",
                        self.path.join(relative).display()
                    )
                });
            }
            // SAFETY: openat returned a newly owned descriptor on success.
            let file = unsafe { File::from_raw_fd(descriptor) };
            ensure!(
                file.metadata()?.is_file(),
                "{label} {} is not a regular file",
                self.path.join(relative).display()
            );
            Ok(file)
        }
        #[cfg(not(unix))]
        {
            open_regular_non_symlink(&self.path.join(relative), label)
        }
    }

    fn ensure_same_file(
        &self,
        relative: &str,
        expected: &StableCorpusFileIdentity,
        label: &str,
    ) -> Result<()> {
        let file = self.open_file(relative, label)?;
        let observed = file
            .metadata()
            .with_context(|| format!("failed to reinspect published {label} `{relative}`"))?;
        ensure!(
            StableCorpusFileIdentity::from_metadata(&observed) == *expected,
            "published {label} {} changed after corpus authentication",
            self.path.join(relative).display()
        );
        Ok(())
    }

    fn ensure_still_published(&self) -> Result<()> {
        ensure_path_still_same_object(&self.path, &self.identity, "corpus root")
    }
}

impl AuthenticatedCorpus {
    /// Open a directory or explicit `manifest.json` input as one authenticated
    /// corpus generation. Non-manifest files return `None` so callers can keep
    /// their existing generic input behavior.
    pub fn open_data_path(path: &Path) -> Result<Option<Self>> {
        let manifest_path = if path.is_dir() {
            Some(path.join("manifest.json"))
        } else if path.file_name().is_some_and(|name| name == "manifest.json") {
            Some(path.to_owned())
        } else {
            None
        };
        manifest_path
            .map(|path| Self::open_manifest(&path))
            .transpose()
    }

    fn open_manifest(manifest_path: &Path) -> Result<Self> {
        ensure!(
            manifest_path
                .file_name()
                .is_some_and(|name| name == "manifest.json"),
            "corpus manifest path must end in manifest.json"
        );
        let root_path = manifest_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let root = PinnedCorpusRoot::open(root_path)?;
        let manifest_file = root.open_file("manifest.json", "corpus manifest")?;
        let (manifest_bytes, manifest_identity) =
            read_stable_corpus_file(manifest_file, "corpus manifest")?;
        let manifest: CorpusManifest = serde_json::from_slice(&manifest_bytes)
            .with_context(|| format!("invalid corpus manifest {}", manifest_path.display()))?;
        let shards = manifest.verify_with_root(&root)?;
        root.ensure_same_file("manifest.json", &manifest_identity, "corpus manifest")?;
        root.ensure_still_published()?;
        Ok(Self {
            root,
            manifest_identity,
            manifest,
            shards,
        })
    }

    /// Parsed manifest whose exact bytes and referenced shards were verified.
    pub fn manifest(&self) -> &CorpusManifest {
        &self.manifest
    }

    /// Number of immutable shards in this corpus generation.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_path(&self, index: usize) -> Result<&Path> {
        self.shards
            .get(index)
            .map(|shard| shard.display_path.as_path())
            .with_context(|| format!("authenticated corpus has no shard {index}"))
    }

    fn open_shard(&self, index: usize) -> Result<File> {
        let shard = self
            .shards
            .get(index)
            .with_context(|| format!("authenticated corpus has no shard {index}"))?;
        let file = self.root.open_file(&shard.relative_path, "corpus shard")?;
        let observed = file.metadata().with_context(|| {
            format!(
                "failed to inspect authenticated corpus shard {}",
                shard.display_path.display()
            )
        })?;
        ensure!(
            StableCorpusFileIdentity::from_metadata(&observed) == shard.identity,
            "authenticated corpus shard {} changed after verification",
            shard.display_path.display()
        );
        Ok(file)
    }

    fn ensure_open_shard_still_published(&self, index: usize, file: &File) -> Result<()> {
        let shard = self
            .shards
            .get(index)
            .with_context(|| format!("authenticated corpus has no shard {index}"))?;
        let observed = file.metadata().with_context(|| {
            format!(
                "failed to reinspect authenticated corpus shard {}",
                shard.display_path.display()
            )
        })?;
        ensure!(
            StableCorpusFileIdentity::from_metadata(&observed) == shard.identity,
            "authenticated corpus shard {} changed while it was read",
            shard.display_path.display()
        );
        self.root
            .ensure_same_file(&shard.relative_path, &shard.identity, "corpus shard")?;
        self.root
            .ensure_same_file("manifest.json", &self.manifest_identity, "corpus manifest")?;
        self.root.ensure_still_published()
    }

    /// Stream one shard through an exact verified handle and always perform
    /// the matching handle/path identity check before returning.
    pub fn with_shard<T>(
        &self,
        index: usize,
        read: impl FnOnce(&Path, &mut File) -> Result<T>,
    ) -> Result<T> {
        let path = self.shard_path(index)?.to_owned();
        let mut file = self.open_shard(index)?;
        let result = read(&path, &mut file);
        self.ensure_open_shard_still_published(index, &file)?;
        result
    }

    /// Stream one shard while checking the retained handle after each bounded
    /// read. This prevents an in-place mutation from yielding a mixed record
    /// that reaches a concurrent consumer before the enclosing final check.
    pub fn with_verified_shard<T>(
        &self,
        index: usize,
        read: impl FnOnce(&Path, &mut dyn Read) -> Result<T>,
    ) -> Result<T> {
        let path = self.shard_path(index)?.to_owned();
        let mut file = self.open_shard(index)?;
        let shard = self
            .shards
            .get(index)
            .with_context(|| format!("authenticated corpus has no shard {index}"))?;
        let result = {
            let mut verified = VerifiedCorpusShardReader {
                file: &mut file,
                expected: &shard.identity,
                path: &path,
                buffer: vec![0_u8; VERIFIED_SHARD_READ_BUFFER_BYTES].into_boxed_slice(),
                buffered_start: 0,
                buffered_end: 0,
            };
            read(&path, &mut verified)
        };
        self.ensure_open_shard_still_published(index, &file)?;
        result
    }

    /// Reject persistent publication replacement or in-place mutation while
    /// allowing the retained generation to survive a transient A -> B -> A
    /// pathname swap.
    pub fn ensure_still_published(&self) -> Result<()> {
        self.root.ensure_still_published()?;
        self.root
            .ensure_same_file("manifest.json", &self.manifest_identity, "corpus manifest")?;
        for shard in &self.shards {
            self.root
                .ensure_same_file(&shard.relative_path, &shard.identity, "corpus shard")?;
        }
        self.root.ensure_still_published()
    }
}

impl CorpusManifest {
    /// Verify the content-addressed manifest and every immutable token shard.
    /// Training calls this before accepting a prepared corpus so a copied or
    /// partially replaced build cannot silently change a resumed run.
    pub fn verify(&self, root: &Path) -> Result<()> {
        let root = PinnedCorpusRoot::open(root)?;
        self.verify_with_root(&root)?;
        Ok(())
    }

    fn verify_with_root(&self, root: &PinnedCorpusRoot) -> Result<Vec<PinnedCorpusShard>> {
        self.build.config.validate()?;
        ensure!(
            self.build.version == self.build.config.version,
            "corpus manifest version differs from its embedded configuration"
        );
        ensure!(
            self.build.build_id == self.build.config.build_id,
            "corpus manifest build_id differs from its embedded configuration"
        );
        for (label, component) in [
            ("discovery", &self.build.discovery),
            ("materializer", &self.build.materializer),
        ] {
            ensure!(
                !component.name.trim().is_empty(),
                "corpus {label} component name must not be empty"
            );
        }
        self.build
            .discovery
            .snapshot
            .validate()
            .context("invalid corpus discovery snapshot")?;
        self.build
            .materializer
            .snapshot
            .validate()
            .context("invalid corpus materializer snapshot")?;
        self.build
            .tokenizer
            .validate()
            .context("invalid corpus tokenizer snapshot")?;
        ensure!(
            self.build.config_sha256 == corpus_manifest_configuration_sha256(&self.build)?,
            "corpus manifest configuration hash does not match its embedded components"
        );
        ensure!(
            self.manifest_sha256 == canonical_json_sha256(&serde_json::to_value(&self.build)?)?,
            "corpus manifest body hash does not match manifest_sha256"
        );
        ensure!(
            !self.build.shards.is_empty(),
            "corpus manifest has no shards"
        );
        let mut records = 0u64;
        let mut tokens = 0u64;
        let mut verified_shards = Vec::with_capacity(self.build.shards.len());
        for (index, shard) in self.build.shards.iter().enumerate() {
            let expected_path = format!("shard-{index:05}.tokens.jsonl");
            ensure!(
                shard.path == expected_path,
                "corpus shard sequence is non-canonical: expected `{expected_path}`, got `{}`",
                shard.path
            );
            let relative = Path::new(&shard.path);
            ensure!(
                !relative.as_os_str().is_empty()
                    && !relative.is_absolute()
                    && relative
                        .components()
                        .all(|component| matches!(component, Component::Normal(_))),
                "corpus shard path `{}` is not a safe relative path",
                shard.path
            );
            ensure!(
                shard.records > 0 && shard.tokens > 0,
                "corpus shard `{}` is empty",
                shard.path
            );
            ensure!(
                shard.tokens <= self.build.config.sharding.max_tokens_per_shard,
                "corpus shard `{}` exceeds configured max_tokens_per_shard",
                shard.path
            );
            let path = root.path.join(relative);
            let file = root.open_file(&shard.path, "corpus shard")?;
            let before = StableCorpusFileIdentity::from_metadata(&file.metadata()?);
            let mut input = BufReader::new(file);
            let mut hasher = Sha256::new();
            let mut line = Vec::new();
            let mut shard_records = 0u64;
            let mut shard_tokens = 0u64;
            loop {
                line.clear();
                let read = read_corpus_jsonl_record(
                    &mut input,
                    &mut line,
                    "authenticated corpus shard row",
                )?;
                if read == 0 {
                    break;
                }
                hasher.update(&line);
                let record: ManifestShardRecord =
                    serde_json::from_slice(&line).with_context(|| {
                        format!(
                            "corpus shard {} contains an invalid JSONL row {}",
                            path.display(),
                            shard_records.saturating_add(1)
                        )
                    })?;
                ensure!(
                    record.tokens.0 > 0,
                    "corpus shard {} contains an empty token row {}",
                    path.display(),
                    shard_records.saturating_add(1)
                );
                if let Some(maximum_token) = record.tokens.1 {
                    ensure!(
                        u64::from(maximum_token)
                            < u64::try_from(self.build.tokenizer.vocabulary_size)
                                .context("tokenizer vocabulary_size exceeds u64")?,
                        "corpus shard {} row {} contains token id {maximum_token} outside vocabulary size {}",
                        path.display(),
                        shard_records.saturating_add(1),
                        self.build.tokenizer.vocabulary_size
                    );
                }
                shard_records = shard_records
                    .checked_add(1)
                    .context("corpus shard record count overflows u64")?;
                shard_tokens = shard_tokens
                    .checked_add(record.tokens.0)
                    .context("corpus shard token count overflows u64")?;
            }
            let after = StableCorpusFileIdentity::from_metadata(
                &input
                    .get_ref()
                    .metadata()
                    .with_context(|| format!("failed to pin corpus shard {}", path.display()))?,
            );
            ensure!(
                after == before,
                "corpus shard {} changed while it was verified",
                path.display()
            );
            ensure!(
                hex(&hasher.finalize()) == shard.sha256,
                "corpus shard {} hash does not match its manifest",
                path.display()
            );
            ensure!(
                shard_records == shard.records,
                "corpus shard {} contains {shard_records} records but its manifest declares {}",
                path.display(),
                shard.records
            );
            ensure!(
                shard_tokens == shard.tokens,
                "corpus shard {} contains {shard_tokens} tokens but its manifest declares {}",
                path.display(),
                shard.tokens
            );
            records = records
                .checked_add(shard_records)
                .context("corpus shard record count overflows u64")?;
            tokens = tokens
                .checked_add(shard_tokens)
                .context("corpus shard token count overflows u64")?;
            verified_shards.push(PinnedCorpusShard {
                relative_path: shard.path.clone(),
                display_path: path,
                identity: before,
            });
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
            self.build
                .config
                .token_target
                .accepts(self.build.stats.unique_tokens),
            "corpus manifest unique-token total is outside its configured target range"
        );
        ensure!(
            self.build.desired_token_target_reached
                == (self.build.stats.unique_tokens >= self.build.config.token_target.desired),
            "corpus desired-token flag disagrees with token totals"
        );
        validate_stage_stats(
            &self.build.stats,
            &self.build.topic_counts,
            &self.build.difficulty_counts,
        )?;
        for shard in &verified_shards {
            root.ensure_same_file(&shard.relative_path, &shard.identity, "corpus shard")?;
        }
        root.ensure_still_published()?;
        Ok(verified_shards)
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

#[derive(Deserialize)]
struct ManifestShardRecord {
    tokens: ManifestTokenCount,
}

struct ManifestTokenCount(u64, Option<u32>);

impl<'de> Deserialize<'de> for ManifestTokenCount {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TokenCountVisitor;

        impl<'de> Visitor<'de> for TokenCountVisitor {
            type Value = ManifestTokenCount;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an array of unsigned 32-bit token identifiers")
            }

            fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut count = 0u64;
                let mut maximum = None;
                while let Some(token) = sequence.next_element::<u32>()? {
                    count = count.checked_add(1).ok_or_else(|| {
                        <A::Error as serde::de::Error>::custom("token count overflows u64")
                    })?;
                    maximum = Some(maximum.map_or(token, |current: u32| current.max(token)));
                }
                Ok(ManifestTokenCount(count, maximum))
            }
        }

        deserializer.deserialize_seq(TokenCountVisitor)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaterializationBatchOutcome {
    Continue,
    TargetBoundaryReached,
}

/// Lowercasing immutable classification terms once keeps classification cost
/// proportional to corpus text rather than repeatedly allocating one string
/// per rule and term for every materialized document.
struct CaseFoldedRuleTerms {
    any: Vec<String>,
    all: Vec<String>,
    none: Vec<String>,
}

impl CaseFoldedRuleTerms {
    fn compile(rule: &ClassificationRule) -> Self {
        Self {
            any: rule
                .any_terms
                .iter()
                .map(|term| term.to_lowercase())
                .collect(),
            all: rule
                .all_terms
                .iter()
                .map(|term| term.to_lowercase())
                .collect(),
            none: rule
                .none_terms
                .iter()
                .map(|term| term.to_lowercase())
                .collect(),
        }
    }

    fn uses_text(&self) -> bool {
        !self.any.is_empty() || !self.all.is_empty() || !self.none.is_empty()
    }
}

pub struct CorpusPipeline<'a> {
    config: CorpusBuildConfig,
    search: &'a dyn SearchBackend,
    materializer: &'a dyn RecordMaterializer,
    tokenizer: &'a dyn CorpusTokenizer,
    deduplicator: &'a mut dyn Deduplicator,
    topic_rule_terms: Vec<CaseFoldedRuleTerms>,
    difficulty_rule_terms: Vec<CaseFoldedRuleTerms>,
    classification_uses_text: bool,
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
            (1..=MAX_DISCOVERY_BATCH_SIZE).contains(&search.page_size()),
            "search backend page size must be within 1..={MAX_DISCOVERY_BATCH_SIZE}"
        );
        let topic_rule_terms = config
            .classification
            .topic_rules
            .iter()
            .map(CaseFoldedRuleTerms::compile)
            .collect::<Vec<_>>();
        let difficulty_rule_terms = config
            .classification
            .difficulty_rules
            .iter()
            .map(CaseFoldedRuleTerms::compile)
            .collect::<Vec<_>>();
        let classification_uses_text = topic_rule_terms
            .iter()
            .chain(&difficulty_rule_terms)
            .any(CaseFoldedRuleTerms::uses_text);
        Ok(Self {
            config,
            search,
            materializer,
            tokenizer,
            deduplicator,
            topic_rule_terms,
            difficulty_rule_terms,
            classification_uses_text,
        })
    }

    /// Build or resume `output_root/<build_id>`. Discovery pages and canonical
    /// record-key-ordered materialization batches are independent transactions.
    /// Each transaction commits dedup state and a content-hashed cursor in the
    /// same durable store after its output shards have been synced.
    pub fn run(&mut self, output_root: &Path) -> Result<(PathBuf, CorpusManifest)> {
        let discovery_snapshot = self.search.snapshot()?;
        discovery_snapshot
            .validate()
            .context("search backend returned an invalid source snapshot")?;
        let materializer_snapshot = self.materializer.snapshot()?;
        materializer_snapshot
            .validate()
            .context("record materializer returned an invalid source snapshot")?;
        let discovery_configuration = self.search.configuration()?;
        let materializer_configuration = self.materializer.configuration()?;
        let deduplicator_configuration = self.deduplicator.configuration();
        let tokenizer_snapshot = self.tokenizer.snapshot();
        tokenizer_snapshot
            .validate()
            .context("corpus tokenizer returned an invalid snapshot")?;
        let search_name = self.search.name().to_owned();
        let materializer_name = self.materializer.name().to_owned();
        ensure!(
            !search_name.trim().is_empty(),
            "search backend name is empty"
        );
        ensure!(
            !materializer_name.trim().is_empty(),
            "record materializer name is empty"
        );
        let config_sha256 = corpus_configuration_sha256(
            &self.config,
            &discovery_configuration,
            &materializer_configuration,
            &deduplicator_configuration,
            &tokenizer_snapshot,
        )?;
        let identity = CorpusBuildIdentity {
            build_id: self.config.build_id.clone(),
            config_sha256: config_sha256.clone(),
            search_name: search_name.clone(),
            discovery_snapshot: discovery_snapshot.clone(),
            materializer_name: materializer_name.clone(),
            materializer_snapshot: materializer_snapshot.clone(),
            tokenizer: tokenizer_snapshot.clone(),
        };

        fs::create_dir_all(output_root).with_context(|| {
            format!(
                "failed to create corpus output root {}",
                output_root.display()
            )
        })?;
        let output_metadata = fs::symlink_metadata(output_root).with_context(|| {
            format!(
                "failed to inspect corpus output root {}",
                output_root.display()
            )
        })?;
        ensure!(
            output_metadata.is_dir() && !output_metadata.file_type().is_symlink(),
            "corpus output root {} must be a real directory",
            output_root.display()
        );
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
                    for (index, hit) in page.hits.iter().enumerate() {
                        hit.validate().with_context(|| {
                            format!(
                                "search backend returned an invalid hit {index} for query `{}` at offset {offset}",
                                query.name
                            )
                        })?;
                    }
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
                            tokenizer_snapshot.vocabulary_size,
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
        ensure!(
            self.search.snapshot()? == discovery_snapshot,
            "search backend snapshot changed during corpus build"
        );
        ensure!(
            self.search.configuration()? == discovery_configuration,
            "search backend configuration changed during corpus build"
        );
        ensure!(
            self.materializer.configuration()? == materializer_configuration,
            "record materializer configuration changed during corpus build"
        );
        ensure!(
            self.deduplicator.configuration() == deduplicator_configuration,
            "deduplicator configuration changed during corpus build"
        );
        ensure!(
            self.tokenizer.snapshot() == tokenizer_snapshot,
            "tokenizer snapshot changed during corpus build"
        );

        let body = CorpusManifestBody {
            version: self.config.version,
            build_id: self.config.build_id.clone(),
            config_sha256,
            config: self.config.clone(),
            discovery: ComponentManifest {
                name: search_name,
                configuration: discovery_configuration,
                snapshot: discovery_snapshot,
            },
            materializer: ComponentManifest {
                name: materializer_name,
                configuration: materializer_configuration,
                snapshot: materializer_snapshot,
            },
            deduplicator: deduplicator_configuration,
            tokenizer: tokenizer_snapshot,
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
        vocabulary_size: usize,
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
        let unmaterialized = hit_batch
            .len()
            .checked_sub(ordered.len())
            .context("materializer returned more records than its discovery batch")?;
        if unmaterialized > 0 {
            stats.unmaterialized_records =
                checked_add(stats.unmaterialized_records, unmaterialized)?;
            tracing::warn!(
                "{} returned {unmaterialized} fewer records than the {} discovery hits requested; \
                 those documents are permanently skipped because require_every_record is disabled",
                self.materializer.name(),
                hit_batch.len()
            );
        }
        for mut record in ordered {
            ensure!(
                record.text.len() <= MAX_MATERIALIZED_TEXT_BYTES,
                "materialized record `{}` contains {} text bytes, exceeding the {MAX_MATERIALIZED_TEXT_BYTES}-byte limit",
                record.record_key,
                record.text.len()
            );
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
            validate_token_ids(&canonical_tokens, vocabulary_size, &record.record_key)?;
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
            let case_folded_text = self
                .classification_uses_text
                .then(|| record.text.to_lowercase());
            let topic = classify(
                &record,
                case_folded_text.as_deref(),
                &self.config.classification.topic_rules,
                &self.topic_rule_terms,
                self.config.classification.default_topic.as_deref(),
            );
            let difficulty = classify(
                &record,
                case_folded_text.as_deref(),
                &self.config.classification.difficulty_rules,
                &self.difficulty_rule_terms,
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

            let rendered = render_views(
                &record,
                topic.as_deref(),
                difficulty.as_deref(),
                &self.config.transformations,
                &self.config.repetition,
            )?;
            let mut tokenized_texts = (0..rendered.texts.len()).map(|_| None).collect::<Vec<_>>();
            tokenized_texts[0] = Some(canonical_tokens);
            for view in rendered.views {
                if tokenized_texts[view.text_index].is_none() {
                    let tokens = self
                        .tokenizer
                        .encode(&rendered.texts[view.text_index])
                        .with_context(|| {
                            format!(
                                "failed to tokenize view `{}` for `{}`",
                                view.name, record.record_key
                            )
                        })?;
                    ensure!(
                        !tokens.is_empty(),
                        "tokenizer emitted no tokens for view `{}` of `{}`",
                        view.name,
                        record.record_key
                    );
                    validate_token_ids(&tokens, vocabulary_size, &record.record_key)?;
                    tokenized_texts[view.text_index] = Some(tokens);
                }
                let tokens = tokenized_texts[view.text_index]
                    .as_ref()
                    .expect("view text was tokenized");
                writer.push(TokenizedCorpusRecord {
                    record_key: &record.record_key,
                    uris: &record.uris,
                    topic: topic.as_deref(),
                    difficulty: difficulty.as_deref(),
                    view: &view.name,
                    copy: view.copy,
                    metadata: &record.metadata,
                    tokens,
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
    validate_stage_stats(
        &progress.stats,
        &progress.topic_counts,
        &progress.difficulty_counts,
    )?;
    Ok(())
}

fn validate_stage_stats(
    stats: &CorpusStageStats,
    topic_counts: &BTreeMap<String, u64>,
    difficulty_counts: &BTreeMap<String, u64>,
) -> Result<()> {
    ensure!(
        stats.duplicate_discovery_keys <= stats.discovered_hits,
        "corpus duplicate-discovery count exceeds discovered hits"
    );
    ensure!(
        stats.unique_records <= stats.emitted_views,
        "corpus unique-record count exceeds emitted views"
    );
    ensure!(
        stats.unique_tokens <= stats.exposure_tokens,
        "corpus unique-token count exceeds exposure tokens"
    );
    ensure!(
        (stats.unique_records == 0) == (stats.unique_tokens == 0),
        "corpus unique-record and unique-token emptiness disagree"
    );
    ensure!(
        (stats.emitted_views == 0) == (stats.exposure_tokens == 0),
        "corpus emitted-view and exposure-token emptiness disagree"
    );
    for (label, counts) in [("topic", topic_counts), ("difficulty", difficulty_counts)] {
        let classified = counts.values().try_fold(0u64, |total, count| {
            total
                .checked_add(*count)
                .context("corpus classification count overflows u64")
        })?;
        ensure!(
            classified <= stats.unique_records,
            "corpus {label} counts exceed unique records"
        );
        ensure!(
            counts.keys().all(|value| !value.trim().is_empty()),
            "corpus {label} counts contain an empty label"
        );
    }
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
            let current_bytes = read_corpus_metadata_file(&path, "corpus progress journal")?;
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
    let Some(index) = name
        .strip_prefix("shard-")
        .and_then(|name| name.strip_suffix(".tokens.jsonl"))
    else {
        return false;
    };
    index.len() >= 5
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && (index.len() == 5 || !index.starts_with('0'))
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
    let authenticated = AuthenticatedCorpus::open_data_path(root)?
        .context("published corpus path must contain an authenticated manifest")?;
    let manifest = authenticated.manifest().clone();
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
    authenticated.ensure_still_published()?;
    Ok((root.to_owned(), manifest))
}

fn order_materialized(
    hits: &[DiscoveryHit],
    records: Vec<MaterializedRecord>,
) -> Result<Vec<MaterializedRecord>> {
    let mut by_key = HashMap::with_capacity(records.len());
    for record in records {
        let record_key = record.record_key.clone();
        ensure!(
            by_key.insert(record_key.clone(), record).is_none(),
            "materializer returned duplicate record key `{record_key}`"
        );
    }
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

fn validate_token_ids(tokens: &[u32], vocabulary_size: usize, record_key: &str) -> Result<()> {
    let vocabulary_size =
        u64::try_from(vocabulary_size).context("tokenizer vocabulary_size exceeds u64")?;
    if let Some(maximum_token) = tokens.iter().copied().max() {
        ensure!(
            u64::from(maximum_token) < vocabulary_size,
            "tokenizer emitted token id {maximum_token} outside vocabulary size {vocabulary_size} for `{record_key}`"
        );
    }
    Ok(())
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
    // Capping blank lines is independent of CRLF conversion: a recipe that
    // disables `normalize_newlines` still gets the cap it configured.
    {
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
    case_folded_text: Option<&str>,
    rules: &[ClassificationRule],
    folded_terms: &[CaseFoldedRuleTerms],
    default: Option<&str>,
) -> Option<String> {
    debug_assert_eq!(rules.len(), folded_terms.len());
    rules
        .iter()
        .zip(folded_terms)
        .filter_map(|(rule, terms)| {
            let matched_any = terms
                .any
                .iter()
                .filter(|term| case_folded_text.is_some_and(|text| text.contains(*term)))
                .count();
            let matched_all = terms
                .all
                .iter()
                .filter(|term| case_folded_text.is_some_and(|text| text.contains(*term)))
                .count();
            let matches = (terms.any.is_empty() || matched_any > 0)
                && matched_all == terms.all.len()
                && terms
                    .none
                    .iter()
                    .all(|term| case_folded_text.is_some_and(|text| !text.contains(term)))
                && rule
                    .metadata_equals
                    .iter()
                    .all(|(name, expected)| record.metadata.get(name) == Some(expected));
            matches.then_some((
                rule,
                rule.metadata_equals.len(),
                matched_any + matched_all + terms.none.len(),
            ))
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
    text_index: usize,
}

struct RenderedViews<'a> {
    texts: Vec<Cow<'a, str>>,
    views: Vec<RenderedView>,
}

fn render_views<'a>(
    record: &'a MaterializedRecord,
    topic: Option<&str>,
    difficulty: Option<&str>,
    transforms: &[TransformationConfig],
    repetition: &RepetitionConfig,
) -> Result<RenderedViews<'a>> {
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
            text_index: 0,
        })
        .collect::<Vec<_>>();
    let mut texts = vec![Cow::Borrowed(record.text.as_str())];
    let mut retained_text_bytes = record.text.len();
    for transform in transforms {
        let remaining = repetition.max_copies_per_record.saturating_sub(views.len());
        if remaining == 0 {
            break;
        }
        if !predicate_matches(&transform.when, record, topic, difficulty) {
            continue;
        }
        let rendered = render_template(&transform.template, record, topic, difficulty)?;
        let text_index = match texts
            .iter()
            .position(|existing| existing.as_ref() == rendered)
        {
            Some(index) => index,
            None => {
                retained_text_bytes = checked_retained_rendered_text_bytes(
                    retained_text_bytes,
                    rendered.len(),
                    &record.record_key,
                )?;
                texts.push(Cow::Owned(rendered));
                texts.len() - 1
            }
        };
        for copy in 0..transform.copies.min(remaining) {
            views.push(RenderedView {
                name: transform.name.clone(),
                copy,
                text_index,
            });
        }
    }
    Ok(RenderedViews { texts, views })
}

fn checked_retained_rendered_text_bytes(
    current: usize,
    additional: usize,
    record_key: &str,
) -> Result<usize> {
    let retained = current
        .checked_add(additional)
        .context("retained rendered-view byte count overflows usize")?;
    ensure!(
        retained <= MAX_RETAINED_RENDERED_TEXT_BYTES,
        "rendered variants for record `{record_key}` exceed the {MAX_RETAINED_RENDERED_TEXT_BYTES}-byte aggregate limit"
    );
    Ok(retained)
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
) -> Result<String> {
    fn append(rendered: &mut String, value: &str) -> Result<()> {
        let next = rendered
            .len()
            .checked_add(value.len())
            .context("rendered corpus view byte count overflows usize")?;
        ensure!(
            next <= MAX_RENDERED_VIEW_TEXT_BYTES,
            "rendered corpus view exceeds the {MAX_RENDERED_VIEW_TEXT_BYTES}-byte limit"
        );
        rendered.push_str(value);
        Ok(())
    }

    let mut rendered = String::with_capacity(template.len().min(MAX_RENDERED_VIEW_TEXT_BYTES));
    let mut cursor = 0usize;
    while let Some(relative_open) = template[cursor..].find("${") {
        let open = cursor + relative_open;
        append(&mut rendered, &template[cursor..open])?;
        let value_start = open + 2;
        let Some(relative_close) = template[value_start..].find('}') else {
            append(&mut rendered, &template[open..])?;
            return Ok(rendered);
        };
        let close = value_start + relative_close;
        let name = &template[value_start..close];
        let replacement = match name {
            "text" => Some(Cow::Borrowed(record.text.as_str())),
            "topic" => Some(Cow::Borrowed(topic.unwrap_or_default())),
            "difficulty" => Some(Cow::Borrowed(difficulty.unwrap_or_default())),
            name => name
                .strip_prefix("metadata.")
                .and_then(|name| record.metadata.get(name))
                .map(|value| match value.as_str() {
                    Some(value) => Cow::Borrowed(value),
                    None => Cow::Owned(value.to_string()),
                }),
        };
        match replacement {
            Some(value) => append(&mut rendered, &value)?,
            None => append(&mut rendered, &template[open..=close])?,
        }
        cursor = close + 1;
    }
    append(&mut rendered, &template[cursor..])?;
    Ok(rendered)
}

#[cfg(test)]
mod transformation_tests {
    use super::*;

    #[test]
    fn template_expansion_is_single_pass_and_does_not_reinterpret_record_values() {
        let record = MaterializedRecord {
            record_key: "record".to_owned(),
            text: "${metadata.first}".to_owned(),
            uris: Vec::new(),
            metadata: BTreeMap::from([
                (
                    "first".to_owned(),
                    Value::String("${metadata.second}".to_owned()),
                ),
                ("second".to_owned(), Value::from(7)),
            ]),
        };
        let rendered = render_template(
            "${text}|${metadata.first}|${metadata.second}|${topic}|${unknown}",
            &record,
            Some("${difficulty}"),
            None,
        )
        .unwrap();
        assert_eq!(
            rendered,
            "${metadata.first}|${metadata.second}|7|${difficulty}|${unknown}"
        );
    }

    #[test]
    fn compiled_classification_terms_preserve_case_insensitive_matching() {
        let record = MaterializedRecord {
            record_key: "record".to_owned(),
            text: "A GRAMMAR handbook".to_owned(),
            uris: Vec::new(),
            metadata: BTreeMap::new(),
        };
        let rules = vec![ClassificationRule {
            label: "language".to_owned(),
            priority: 0,
            any_terms: vec!["Grammar".to_owned()],
            all_terms: Vec::new(),
            none_terms: vec!["mathematics".to_owned()],
            metadata_equals: BTreeMap::new(),
        }];
        let folded = rules
            .iter()
            .map(CaseFoldedRuleTerms::compile)
            .collect::<Vec<_>>();
        let text = record.text.to_lowercase();
        assert_eq!(
            classify(&record, Some(&text), &rules, &folded, Some("other")),
            Some("language".to_owned())
        );
    }

    #[test]
    fn retained_rendered_text_accounting_rejects_limits_and_overflow() {
        assert_eq!(
            checked_retained_rendered_text_bytes(MAX_RETAINED_RENDERED_TEXT_BYTES - 1, 1, "record")
                .unwrap(),
            MAX_RETAINED_RENDERED_TEXT_BYTES
        );
        let error =
            checked_retained_rendered_text_bytes(MAX_RETAINED_RENDERED_TEXT_BYTES, 1, "record")
                .unwrap_err()
                .to_string();
        assert!(error.contains("record"), "{error}");
        assert!(error.contains("aggregate limit"), "{error}");
        assert!(checked_retained_rendered_text_bytes(usize::MAX, 1, "record").is_err());
    }
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

    fn would_rotate(&self, token_count: u64) -> bool {
        self.open.as_ref().is_some_and(|shard| {
            shard.records > 0
                && shard
                    .tokens
                    .checked_add(token_count)
                    .is_none_or(|total| total > self.max_tokens)
        })
    }

    fn push(&mut self, record: TokenizedCorpusRecord<'_>) -> Result<()> {
        let token_count = u64::try_from(record.tokens.len())?;
        ensure!(
            token_count <= self.max_tokens,
            "one corpus view contains {token_count} tokens, exceeding max_tokens_per_shard {}",
            self.max_tokens
        );
        if self.would_rotate(token_count) {
            self.close()?;
        }
        if self.open.is_none() {
            self.open()?;
        }
        let bytes = serde_json::to_vec(&record)?;
        ensure!(
            bytes
                .len()
                .checked_add(1)
                .is_some_and(|line_bytes| line_bytes <= MAX_CORPUS_JSONL_RECORD_BYTES),
            "serialized corpus view is {} bytes before its newline, exceeding the JSONL record limit of {MAX_CORPUS_JSONL_RECORD_BYTES}",
            bytes.len()
        );
        let shard = self.open.as_mut().expect("shard was opened");
        // Validate accounting before mutating the shard. This keeps an
        // arithmetic failure from writing a row that cannot be represented by
        // the durable manifest.
        let records = checked_add(shard.records, 1)?;
        let total_tokens = checked_add(shard.tokens, record.tokens.len())?;
        shard.writer.write_all(&bytes)?;
        shard.writer.write_all(b"\n")?;
        shard.hasher.update(&bytes);
        shard.hasher.update(b"\n");
        shard.records = records;
        shard.tokens = total_tokens;
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

pub(crate) fn canonical_json_sha256(value: &Value) -> Result<String> {
    let canonical = canonicalize(value);
    Ok(hex(&Sha256::digest(serde_json::to_vec(&canonical)?)))
}

fn corpus_configuration_sha256(
    config: &CorpusBuildConfig,
    discovery: &Value,
    materializer: &Value,
    deduplicator: &Value,
    tokenizer: &TokenizerSnapshot,
) -> Result<String> {
    canonical_json_sha256(&serde_json::json!({
        "corpus": config,
        "search": discovery,
        "materializer": materializer,
        "deduplicator": deduplicator,
        "tokenizer": tokenizer,
    }))
}

pub(crate) fn corpus_manifest_configuration_sha256(body: &CorpusManifestBody) -> Result<String> {
    corpus_configuration_sha256(
        &body.config,
        &body.discovery.configuration,
        &body.materializer.configuration,
        &body.deduplicator,
        &body.tokenizer,
    )
}

fn read_stable_corpus_file(
    mut file: File,
    label: &str,
) -> Result<(Vec<u8>, StableCorpusFileIdentity)> {
    read_stable_corpus_file_with_limit(&mut file, label, MAX_CORPUS_METADATA_BYTES)
}

/// Capture a small corpus configuration, journal, or manifest through one
/// regular-file handle. Both the allocation and concurrent-growth window are
/// bounded, and persistent pathname replacement is rejected before the bytes
/// are returned to a caller.
pub(crate) fn read_corpus_metadata_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    let published = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        published.is_file() && !published.file_type().is_symlink(),
        "{label} {} is not a regular non-symlink file",
        path.display()
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let file = options
        .open(path)
        .with_context(|| format!("failed to open {label} {}", path.display()))?;
    ensure!(
        StableCorpusFileIdentity::from_metadata(&file.metadata()?)
            == StableCorpusFileIdentity::from_metadata(&published),
        "{label} {} changed while it was opened",
        path.display()
    );
    let (bytes, identity) = read_stable_corpus_file(file, label)?;
    let final_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to reinspect {label} {}", path.display()))?;
    ensure!(
        final_metadata.is_file()
            && !final_metadata.file_type().is_symlink()
            && StableCorpusFileIdentity::from_metadata(&final_metadata) == identity,
        "published {label} {} changed while it was read",
        path.display()
    );
    Ok(bytes)
}

fn read_stable_corpus_file_with_limit(
    file: &mut File,
    label: &str,
    maximum_bytes: u64,
) -> Result<(Vec<u8>, StableCorpusFileIdentity)> {
    let before = file
        .metadata()
        .with_context(|| format!("failed to inspect opened {label}"))?;
    ensure!(before.is_file(), "opened {label} is not a regular file");
    ensure!(
        before.len() <= maximum_bytes,
        "opened {label} is {} bytes, exceeding the limit of {maximum_bytes}",
        before.len()
    );
    let identity = StableCorpusFileIdentity::from_metadata(&before);
    let capacity = usize::try_from(before.len())
        .with_context(|| format!("{label} is too large for this address space"))?;
    let bytes = read_bounded(file, capacity, label)?;
    let after = file
        .metadata()
        .with_context(|| format!("failed to reinspect opened {label}"))?;
    ensure!(
        StableCorpusFileIdentity::from_metadata(&after) == identity
            && bytes.len() as u64 == after.len(),
        "opened {label} changed while it was read"
    );
    Ok((bytes, identity))
}

pub(crate) fn read_corpus_jsonl_record(
    reader: &mut (impl BufRead + ?Sized),
    output: &mut Vec<u8>,
    label: &str,
) -> Result<usize> {
    read_jsonl_record_bounded(reader, output, MAX_CORPUS_JSONL_RECORD_BYTES, label)
}

fn read_jsonl_record_bounded(
    reader: &mut (impl BufRead + ?Sized),
    output: &mut Vec<u8>,
    maximum_bytes: usize,
    label: &str,
) -> Result<usize> {
    ensure!(maximum_bytes > 0, "{label} byte limit must be positive");
    output.clear();
    let capture_bytes = maximum_bytes
        .checked_add(1)
        .context("corpus JSONL record limit overflows usize")?;
    let read = reader
        .take(u64::try_from(capture_bytes).context("corpus JSONL record limit exceeds u64")?)
        .read_until(b'\n', output)
        .with_context(|| format!("failed to read {label}"))?;
    let payload_bytes = output
        .len()
        .checked_sub(usize::from(output.last() == Some(&b'\n')))
        .context("corpus JSONL record byte count underflows usize")?;
    ensure!(
        payload_bytes <= maximum_bytes,
        "{label} exceeds the maximum of {maximum_bytes} bytes"
    );
    Ok(read)
}

fn read_bounded(reader: &mut impl Read, capacity: usize, label: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .with_context(|| format!("failed to reserve buffer for {label}"))?;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to read opened {label}"))?;
        if read == 0 {
            break;
        }
        ensure!(
            bytes
                .len()
                .checked_add(read)
                .is_some_and(|length| length <= capacity),
            "opened {label} grew while it was read"
        );
        bytes.extend_from_slice(&buffer[..read]);
    }
    Ok(bytes)
}

#[cfg(not(unix))]
fn open_regular_non_symlink(path: &Path, label: &str) -> Result<File> {
    let expected = fs::symlink_metadata(path)
        .with_context(|| format!("{label} {} is missing", path.display()))?;
    ensure!(
        expected.file_type().is_file() && !expected.file_type().is_symlink(),
        "{label} {} is not a regular, non-symlink file",
        path.display()
    );

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .with_context(|| format!("failed to securely open {label} {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("failed to inspect opened {label} {}", path.display()))?;
    ensure!(
        opened.file_type().is_file(),
        "opened {label} {} is not a regular file",
        path.display()
    );
    #[cfg(unix)]
    ensure!(
        same_file_object(&opened, &expected),
        "{label} {} changed while it was opened",
        path.display()
    );
    #[cfg(not(unix))]
    ensure!(
        opened.len() == expected.len() && opened.modified().ok() == expected.modified().ok(),
        "{label} {} changed while it was opened",
        path.display()
    );
    Ok(file)
}

fn ensure_path_still_same_object(path: &Path, expected: &fs::Metadata, label: &str) -> Result<()> {
    let published = fs::symlink_metadata(path)
        .with_context(|| format!("failed to reinspect {label} {}", path.display()))?;
    ensure!(
        published.is_dir()
            && !published.file_type().is_symlink()
            && same_file_object(expected, &published),
        "published {label} {} changed during verification",
        path.display()
    );
    Ok(())
}

#[cfg(unix)]
fn same_file_object(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_object(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.file_type() == right.file_type()
        && left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
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

#[cfg(test)]
mod path_identity_tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn bounded_capture_rejects_growth_beyond_the_opened_length() {
        let error = read_bounded(&mut Cursor::new(b"growth"), 5, "test file")
            .unwrap_err()
            .to_string();
        assert!(error.contains("grew while it was read"), "{error}");
    }

    #[test]
    fn bounded_jsonl_reader_stops_after_one_byte_over_the_limit() {
        let mut input = BufReader::new(Cursor::new(b"012345678\n"));
        let mut row = Vec::new();
        let error = read_jsonl_record_bounded(&mut input, &mut row, 8, "test row")
            .unwrap_err()
            .to_string();
        assert!(error.contains("maximum of 8 bytes"), "{error}");
        assert_eq!(row.len(), 9);
    }

    #[test]
    fn bounded_jsonl_reader_accepts_a_row_exactly_at_the_limit() {
        let mut input = BufReader::new(Cursor::new(b"12345678\ntrailing"));
        let mut row = Vec::new();
        assert_eq!(
            read_jsonl_record_bounded(&mut input, &mut row, 8, "test row").unwrap(),
            9
        );
        assert_eq!(row, b"12345678\n");
    }

    #[test]
    fn shard_name_recognizes_indices_beyond_the_minimum_padding_width() {
        assert!(is_shard_name("shard-00000.tokens.jsonl"));
        assert!(is_shard_name("shard-99999.tokens.jsonl"));
        assert!(is_shard_name("shard-100000.tokens.jsonl"));
        assert!(!is_shard_name("shard-000001.tokens.jsonl"));
        assert!(!is_shard_name("shard-1234.tokens.jsonl"));
        assert!(!is_shard_name("shard-1234x.tokens.jsonl"));
    }

    #[test]
    fn immutable_shard_writer_rotates_when_token_accounting_would_overflow() {
        let root = tempfile::tempdir().unwrap();
        let mut writer = ImmutableShardWriter::resume(root.path(), u64::MAX, Vec::new()).unwrap();
        writer.open().unwrap();
        let shard = writer.open.as_mut().unwrap();
        shard.records = 1;
        shard.tokens = u64::MAX;

        assert!(writer.would_rotate(1));
    }

    #[test]
    fn stable_manifest_capture_rejects_an_oversized_opened_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("manifest.json");
        fs::write(&path, b"123456").unwrap();
        let mut file = File::open(path).unwrap();
        let error = read_stable_corpus_file_with_limit(&mut file, "test manifest", 5)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeding the limit of 5"), "{error}");
    }

    #[test]
    fn opened_regular_file_rejects_a_persistent_published_replacement() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("shard.jsonl");
        fs::write(&path, b"generation-a").unwrap();
        let pinned_root = PinnedCorpusRoot::open(root.path()).unwrap();
        let opened = pinned_root.open_file("shard.jsonl", "test shard").unwrap();
        let identity = StableCorpusFileIdentity::from_metadata(&opened.metadata().unwrap());

        fs::rename(&path, root.path().join("generation-a.jsonl")).unwrap();
        fs::write(&path, b"generation-b").unwrap();

        let error = pinned_root
            .ensure_same_file("shard.jsonl", &identity, "test shard")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("changed after corpus authentication"),
            "{error}"
        );
    }
}
