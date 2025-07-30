//! First-party, content-pinned inputs for in-model sleep.
//!
//! Wake execution seals token contexts into an immutable journal. Consolidation
//! then generates teacher/student rollouts only from that journal, while the
//! semantic judge and retention suite are independently content addressed.
//! These adapters deliberately have no search, corpus, tokenizer, or network
//! handles, which keeps retries deterministic and prevents sleep from reading
//! new training data.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use burn::tensor::{Int, Tensor, TensorData};
use hermes_llm::generate::SamplingConfig;
use hermes_llm::{Device, TextGenerator, Transformer};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::sleep::{ConsolidationTxn, RngReservation};
use crate::tensor_sleep::{
    ConsolidationRollouts, ImitationGroup, RetentionEvaluator, RetentionScores, RolloutOwner,
    SemanticJudge, TokenRolloutBatch,
};

pub const WAKE_CONTEXT_JOURNAL_VERSION: u32 = 1;
pub const SEMANTIC_JUDGE_ARTIFACT_VERSION: u32 = 1;
pub const RETENTION_EVALUATOR_ARTIFACT_VERSION: u32 = 1;
pub const RETENTION_SUITE_VERSION: u32 = 1;

const KNOWLEDGE_CONTEXT_DOMAIN: u64 = 0x6b6e_6f77_2d63_7478;
const IMITATION_CONTEXT_DOMAIN: u64 = 0x696d_6974_2d63_7478;
const TEACHER_GENERATION_DOMAIN: u64 = 0x7465_6163_6865_722d;
const STUDENT_GENERATION_DOMAIN: u64 = 0x7374_7564_656e_742d;
const RETENTION_EVALUATION_DOMAIN: u64 = 0x7265_7465_6e74_696f;

static TEMPORARY_ORDINAL: AtomicU64 = AtomicU64::new(0);

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    let digest = value
        .strip_prefix("sha256:")
        .with_context(|| format!("{label} must start with `sha256:`"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} must contain 64 lowercase hexadecimal digits"
    );
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{label} {} must be a regular non-symlink file",
        path.display()
    );
    fs::read(path).with_context(|| format!("failed to read {label} {}", path.display()))
}

fn read_pinned_json<T: DeserializeOwned>(
    path: &Path,
    expected_sha256: &str,
    label: &str,
) -> Result<T> {
    validate_sha256(expected_sha256, &format!("{label} hash"))?;
    let bytes = read_regular_file(path, label)?;
    let observed = sha256_bytes(&bytes);
    ensure!(
        observed == expected_sha256,
        "{label} {} changed: expected {expected_sha256}, observed {observed}",
        path.display()
    );
    serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid {label} JSON in {}", path.display()))
}

/// A strict local content-addressed input. Relative paths are resolved once
/// against the runtime-config directory; every read rechecks file type and
/// digest so a resume never trusts validation from an earlier process.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedLocalArtifact {
    pub path: PathBuf,
    pub sha256: String,
}

impl PinnedLocalArtifact {
    pub fn resolve(mut self, base: &Path) -> Result<Self> {
        ensure!(
            !self.path.as_os_str().is_empty(),
            "pinned artifact path is empty"
        );
        validate_sha256(&self.sha256, "pinned artifact hash")?;
        if self.path.is_relative() {
            self.path = base.join(&self.path);
        }
        Ok(self)
    }

    pub fn verify_bytes(&self) -> Result<Vec<u8>> {
        validate_sha256(&self.sha256, "pinned artifact hash")?;
        let bytes = read_regular_file(&self.path, "pinned artifact")?;
        let observed = sha256_bytes(&bytes);
        ensure!(
            observed == self.sha256,
            "pinned artifact {} changed: expected {}, observed {observed}",
            self.path.display(),
            self.sha256
        );
        Ok(bytes)
    }

    pub fn verify_json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.verify_bytes()?)
            .with_context(|| format!("invalid pinned JSON artifact {}", self.path.display()))
    }
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("failed to open directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("failed to sync directory {}", path.display()))
}

/// Publish bytes without replacing an existing artifact. An existing target is
/// accepted only when it is a regular file containing the exact same bytes.
fn publish_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent_metadata = fs::symlink_metadata(parent)
        .with_context(|| format!("failed to inspect output directory {}", parent.display()))?;
    ensure!(
        parent_metadata.is_dir() && !parent_metadata.file_type().is_symlink(),
        "output parent {} must be a real directory",
        parent.display()
    );
    if path.exists() {
        ensure!(
            read_regular_file(path, "immutable artifact")? == bytes,
            "immutable artifact {} already exists with different contents",
            path.display()
        );
        return Ok(());
    }

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("immutable artifact filename is not UTF-8")?;
    let ordinal = TEMPORARY_ORDINAL.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.tmp.{}.{}",
        std::process::id(),
        ordinal
    ));
    let result = (|| {
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        output.write_all(bytes)?;
        output.sync_all()?;
        drop(output);
        match fs::hard_link(&temporary, path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                ensure!(
                    read_regular_file(path, "immutable artifact")? == bytes,
                    "immutable artifact {} won a publication race with different contents",
                    path.display()
                );
            }
            Err(error) => return Err(error).context("publishing immutable artifact"),
        }
        fs::remove_file(&temporary)?;
        sync_directory(parent)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WakeContextRecord {
    /// Stable wake-owned identity, normally `<phase>:<optimizer-step>:<row>`.
    pub id: String,
    pub optimizer_step: u64,
    pub token_ids: Vec<i64>,
}

impl WakeContextRecord {
    fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "wake context id is empty");
        ensure!(
            !self.token_ids.is_empty(),
            "wake context `{}` has no tokens",
            self.id
        );
        ensure!(
            self.token_ids.iter().all(|token| *token >= 0),
            "wake context `{}` contains a negative token id",
            self.id
        );
        Ok(())
    }
}

/// Exact token contexts retained by the model's wake process. This is not a
/// replay corpus: it is a bounded, immutable input to self-generated sleep
/// rollouts and carries no URI from which more source data could be fetched.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WakeContextJournal {
    pub version: u32,
    pub source_checkpoint_sha256: String,
    pub records: Vec<WakeContextRecord>,
}

impl WakeContextJournal {
    pub fn new(source_checkpoint_sha256: impl Into<String>) -> Result<Self> {
        let journal = Self {
            version: WAKE_CONTEXT_JOURNAL_VERSION,
            source_checkpoint_sha256: source_checkpoint_sha256.into(),
            records: Vec::new(),
        };
        validate_sha256(
            &journal.source_checkpoint_sha256,
            "wake-context source checkpoint hash",
        )?;
        Ok(journal)
    }

    pub fn push(&mut self, record: WakeContextRecord) -> Result<()> {
        record.validate()?;
        ensure!(
            !self.records.iter().any(|existing| existing.id == record.id),
            "wake context journal repeats id `{}`",
            record.id
        );
        ensure!(
            self.records
                .last()
                .is_none_or(|previous| previous.optimizer_step <= record.optimizer_step),
            "wake context journal optimizer steps move backwards"
        );
        self.records.push(record);
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == WAKE_CONTEXT_JOURNAL_VERSION,
            "unsupported wake-context journal version {}",
            self.version
        );
        validate_sha256(
            &self.source_checkpoint_sha256,
            "wake-context source checkpoint hash",
        )?;
        ensure!(!self.records.is_empty(), "wake context journal is empty");
        let mut ids = BTreeSet::new();
        let mut previous_step = 0;
        for (index, record) in self.records.iter().enumerate() {
            record.validate()?;
            ensure!(
                ids.insert(record.id.as_str()),
                "wake context journal repeats id `{}`",
                record.id
            );
            ensure!(
                index == 0 || record.optimizer_step >= previous_step,
                "wake context journal optimizer steps move backwards"
            );
            previous_step = record.optimizer_step;
        }
        Ok(())
    }

    /// Seal this journal without overwriting a different artifact.
    pub fn publish(&self, path: &Path) -> Result<PinnedWakeContextJournal> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        let sha256 = sha256_bytes(&bytes);
        publish_immutable(path, &bytes)?;
        PinnedWakeContextJournal::load(path, &sha256)
    }
}

#[derive(Clone, Debug)]
pub struct PinnedWakeContextJournal {
    path: PathBuf,
    sha256: String,
    journal: WakeContextJournal,
}

impl PinnedWakeContextJournal {
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self> {
        let journal: WakeContextJournal =
            read_pinned_json(path, expected_sha256, "wake-context journal")?;
        journal.validate()?;
        Ok(Self {
            path: path.to_owned(),
            sha256: expected_sha256.to_owned(),
            journal,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn source_checkpoint_sha256(&self) -> &str {
        &self.journal.source_checkpoint_sha256
    }

    pub fn records(&self) -> &[WakeContextRecord] {
        &self.journal.records
    }
}

fn default_rollout_temperature() -> f64 {
    0.8
}

fn default_rollout_top_k() -> usize {
    32
}

fn default_repetition_penalty() -> f64 {
    1.05
}

fn default_imitation_groups() -> usize {
    1
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JournalRolloutConfig {
    pub max_context_tokens: usize,
    pub continuation_tokens: usize,
    #[serde(default = "default_rollout_temperature")]
    pub temperature: f64,
    #[serde(default = "default_rollout_top_k")]
    pub top_k: usize,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token: Option<u32>,
    #[serde(default = "default_imitation_groups")]
    pub imitation_groups: usize,
}

impl JournalRolloutConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_context_tokens > 0 && self.continuation_tokens > 0,
            "journal rollout context and continuation lengths must be positive"
        );
        ensure!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "journal rollout temperature must be finite and positive"
        );
        ensure!(self.top_k > 0, "journal rollout top_k must be positive");
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty >= 1.0,
            "journal rollout repetition penalty must be finite and at least one"
        );
        ensure!(
            self.imitation_groups > 0,
            "journal rollout imitation_groups must be positive"
        );
        Ok(())
    }
}

/// Model-owned rollout source backed only by an already-sealed wake journal.
pub struct JournalRollouts {
    journal: PinnedWakeContextJournal,
    /// The transaction teacher accepted by this adapter instance. For the
    /// first consolidation this is the journal source; later senders in the
    /// same atomic phase are descendants authenticated by the native cursor.
    teacher_sha256: String,
    device: Device,
    config: JournalRolloutConfig,
}

impl JournalRollouts {
    pub fn new(
        journal: PinnedWakeContextJournal,
        device: Device,
        config: JournalRolloutConfig,
    ) -> Result<Self> {
        config.validate()?;
        let teacher_sha256 = journal.source_checkpoint_sha256().to_owned();
        Ok(Self {
            journal,
            teacher_sha256,
            device,
            config,
        })
    }

    /// Bind the immutable wake journal to a later teacher in the same native
    /// sleep phase. The caller must authenticate the descendant through the
    /// completed transaction chain before constructing this adapter.
    pub fn for_phase_teacher(
        journal: PinnedWakeContextJournal,
        phase_input_sha256: &str,
        teacher_sha256: &str,
        device: Device,
        config: JournalRolloutConfig,
    ) -> Result<Self> {
        config.validate()?;
        validate_sha256(phase_input_sha256, "sleep phase input checkpoint")?;
        validate_sha256(teacher_sha256, "sleep transaction teacher")?;
        ensure!(
            journal.source_checkpoint_sha256() == phase_input_sha256,
            "wake-context journal belongs to {}, phase input is {}",
            journal.source_checkpoint_sha256(),
            phase_input_sha256
        );
        Ok(Self {
            journal,
            teacher_sha256: teacher_sha256.to_owned(),
            device,
            config,
        })
    }

    pub fn journal(&self) -> &PinnedWakeContextJournal {
        &self.journal
    }

    fn validate_transaction(&self, txn: &ConsolidationTxn) -> Result<()> {
        ensure!(
            self.teacher_sha256 == txn.teacher_hash,
            "rollout adapter is bound to checkpoint {}, transaction teacher is {}",
            self.teacher_sha256,
            txn.teacher_hash
        );
        Ok(())
    }

    fn selected_context(&self, model: &Transformer, seed: u64) -> Result<Vec<u32>> {
        let records = self.journal.records();
        let record = &records[(seed as usize) % records.len()];
        ensure!(
            record
                .token_ids
                .iter()
                .all(|token| (*token as usize) < model.config().vocab_size),
            "wake context `{}` contains a token outside model vocabulary {}",
            record.id,
            model.config().vocab_size
        );
        let available = model
            .config()
            .max_seq_len
            .checked_sub(self.config.continuation_tokens)
            .context("rollout continuation consumes the complete model context")?;
        let keep = record
            .token_ids
            .len()
            .min(self.config.max_context_tokens)
            .min(available);
        ensure!(keep > 0, "model context is too short for sleep rollouts");
        Ok(record.token_ids[record.token_ids.len() - keep..]
            .iter()
            .map(|token| *token as u32)
            .collect())
    }

    fn generate(
        &self,
        model: &Transformer,
        context_seed: u64,
        generation_seed: u64,
    ) -> Result<(Vec<i64>, Vec<i64>)> {
        let context = self.selected_context(model, context_seed)?;
        let config = SamplingConfig {
            max_new_tokens: self.config.continuation_tokens,
            temperature: self.config.temperature,
            top_k: Some(self.config.top_k.min(model.config().vocab_size)),
            repetition_penalty: self.config.repetition_penalty,
            eos_token: self.config.eos_token,
            seed: Some(generation_seed),
        };
        let generated = TextGenerator::new(model, &self.device).generate(&context, &config)?;
        ensure!(
            generated.len() > context.len(),
            "sleep generation returned no continuation"
        );
        let prefix = context.into_iter().map(i64::from).collect::<Vec<_>>();
        let continuation = generated[prefix.len()..]
            .iter()
            .copied()
            .map(i64::from)
            .collect::<Vec<_>>();
        Ok((prefix, continuation))
    }
}

fn derive_seed(
    txn: &ConsolidationTxn,
    reservation: RngReservation,
    domain: u64,
    ordinal: u64,
) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"hermes-sleep-adapter-seed-v1\0");
    hasher.update(txn.id.to_le_bytes());
    hasher.update(txn.trigger_clock.to_le_bytes());
    hasher.update((reservation.stream as u64).to_le_bytes());
    hasher.update(reservation.start.to_le_bytes());
    hasher.update(reservation.count.to_le_bytes());
    hasher.update(domain.to_le_bytes());
    hasher.update(ordinal.to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

impl ConsolidationRollouts for JournalRollouts {
    fn knowledge_rollouts(
        &mut self,
        txn: &ConsolidationTxn,
        owner: RolloutOwner,
        model: &Transformer,
        count: usize,
    ) -> Result<Vec<TokenRolloutBatch>> {
        self.validate_transaction(txn)?;
        let reservation = txn
            .knowledge_rng
            .context("knowledge rollouts have no persisted RNG reservation")?;
        let owner_domain = match owner {
            RolloutOwner::Teacher => TEACHER_GENERATION_DOMAIN,
            RolloutOwner::DetachedStudent => STUDENT_GENERATION_DOMAIN,
        };
        (0..count)
            .map(|ordinal| {
                let ordinal = ordinal as u64;
                let (mut prefix, continuation) = self.generate(
                    model,
                    derive_seed(
                        txn,
                        reservation,
                        KNOWLEDGE_CONTEXT_DOMAIN ^ owner_domain,
                        ordinal,
                    ),
                    derive_seed(txn, reservation, owner_domain, ordinal),
                )?;
                prefix.extend(continuation);
                TokenRolloutBatch::new(1, prefix.len(), prefix)
            })
            .collect()
    }

    fn imitation_groups(
        &mut self,
        txn: &ConsolidationTxn,
        teacher: &Transformer,
        student: &Transformer,
        group_size: usize,
    ) -> Result<Vec<ImitationGroup>> {
        self.validate_transaction(txn)?;
        ensure!(group_size > 0, "imitation group size is zero");
        let reservation = txn
            .imitation_rng
            .context("imitation generation has no persisted RNG reservation")?;
        (0..self.config.imitation_groups)
            .map(|group| {
                let group = group as u64;
                let context_seed = derive_seed(txn, reservation, IMITATION_CONTEXT_DOMAIN, group);
                let (prefix, teacher_continuation) = self.generate(
                    teacher,
                    context_seed,
                    derive_seed(txn, reservation, TEACHER_GENERATION_DOMAIN, group),
                )?;
                let candidates = (0..group_size)
                    .map(|candidate| {
                        let (student_prefix, continuation) = self.generate(
                            student,
                            context_seed,
                            derive_seed(
                                txn,
                                reservation,
                                STUDENT_GENERATION_DOMAIN ^ group.rotate_left(17),
                                candidate as u64,
                            ),
                        )?;
                        ensure!(
                            student_prefix == prefix,
                            "teacher/student imitation context selection drifted"
                        );
                        Ok(continuation)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ImitationGroup {
                    prefix,
                    teacher_continuation,
                    candidates,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenSemanticJudgeArtifact {
    pub version: u32,
    /// Must be `token_ngram_f1_v1`.
    pub algorithm: String,
    pub unigram_weight: f32,
    pub bigram_weight: f32,
    /// Token ids within one class are treated as frozen semantic aliases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub equivalence_classes: Vec<Vec<i64>>,
}

impl TokenSemanticJudgeArtifact {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == SEMANTIC_JUDGE_ARTIFACT_VERSION,
            "unsupported semantic-judge artifact version {}",
            self.version
        );
        ensure!(
            self.algorithm == "token_ngram_f1_v1",
            "unsupported semantic-judge algorithm `{}`",
            self.algorithm
        );
        ensure!(
            self.unigram_weight.is_finite()
                && self.unigram_weight >= 0.0
                && self.bigram_weight.is_finite()
                && self.bigram_weight >= 0.0
                && self.unigram_weight + self.bigram_weight > 0.0,
            "semantic-judge n-gram weights must be finite, non-negative, and not both zero"
        );
        let mut seen = BTreeSet::new();
        for class in &self.equivalence_classes {
            ensure!(
                class.len() >= 2 && class.iter().all(|token| *token >= 0),
                "semantic equivalence classes require at least two non-negative token ids"
            );
            for token in class {
                ensure!(
                    seen.insert(*token),
                    "semantic token {token} appears in multiple equivalence classes"
                );
            }
        }
        Ok(())
    }
}

/// Frozen, content-addressed token-semantic judge. Deployments can replace it
/// with a neural judge through the same trait; this built-in implementation is
/// deterministic and useful for a fully local first-party workflow.
pub struct PinnedTokenSemanticJudge {
    artifact_sha256: String,
    artifact: TokenSemanticJudgeArtifact,
    aliases: BTreeMap<i64, i64>,
}

impl PinnedTokenSemanticJudge {
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self> {
        let artifact: TokenSemanticJudgeArtifact =
            read_pinned_json(path, expected_sha256, "semantic-judge artifact")?;
        artifact.validate()?;
        let aliases = artifact
            .equivalence_classes
            .iter()
            .flat_map(|class| {
                let representative = *class.iter().min().expect("validated class is non-empty");
                class.iter().map(move |token| (*token, representative))
            })
            .collect();
        Ok(Self {
            artifact_sha256: expected_sha256.to_owned(),
            artifact,
            aliases,
        })
    }

    fn canonical(&self, tokens: &[i64]) -> Vec<i64> {
        tokens
            .iter()
            .map(|token| self.aliases.get(token).copied().unwrap_or(*token))
            .collect()
    }
}

impl SemanticJudge for PinnedTokenSemanticJudge {
    fn artifact_hash(&self) -> &str {
        &self.artifact_sha256
    }

    fn score(&mut self, _: &[i64], teacher: &[i64], candidate: &[i64]) -> Result<f32> {
        ensure!(
            !teacher.is_empty() && !candidate.is_empty(),
            "semantic judge received an empty continuation"
        );
        ensure!(
            teacher.iter().chain(candidate).all(|token| *token >= 0),
            "semantic judge received a negative token id"
        );
        let teacher = self.canonical(teacher);
        let candidate = self.canonical(candidate);
        let unigram = multiset_f1(
            teacher.iter().copied().map(|token| (token, i64::MIN)),
            candidate.iter().copied().map(|token| (token, i64::MIN)),
        );
        let bigram = if teacher.len() < 2 || candidate.len() < 2 {
            unigram
        } else {
            multiset_f1(
                teacher.windows(2).map(|pair| (pair[0], pair[1])),
                candidate.windows(2).map(|pair| (pair[0], pair[1])),
            )
        };
        let weight = self.artifact.unigram_weight + self.artifact.bigram_weight;
        Ok(
            (self.artifact.unigram_weight * unigram + self.artifact.bigram_weight * bigram)
                / weight,
        )
    }
}

fn multiset_f1(
    left: impl Iterator<Item = (i64, i64)>,
    right: impl Iterator<Item = (i64, i64)>,
) -> f32 {
    let mut left_counts = BTreeMap::<(i64, i64), usize>::new();
    let mut right_counts = BTreeMap::<(i64, i64), usize>::new();
    for item in left {
        *left_counts.entry(item).or_default() += 1;
    }
    for item in right {
        *right_counts.entry(item).or_default() += 1;
    }
    let left_total = left_counts.values().sum::<usize>();
    let right_total = right_counts.values().sum::<usize>();
    if left_total == 0 || right_total == 0 {
        return 0.0;
    }
    let overlap = left_counts
        .iter()
        .map(|(item, count)| (*count).min(right_counts.get(item).copied().unwrap_or(0)))
        .sum::<usize>();
    2.0 * overlap as f32 / (left_total + right_total) as f32
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LikelihoodRetentionEvaluatorArtifact {
    pub version: u32,
    /// Must be `causal_likelihood_v1`.
    pub algorithm: String,
}

impl LikelihoodRetentionEvaluatorArtifact {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == RETENTION_EVALUATOR_ARTIFACT_VERSION,
            "unsupported retention-evaluator artifact version {}",
            self.version
        );
        ensure!(
            self.algorithm == "causal_likelihood_v1",
            "unsupported retention-evaluator algorithm `{}`",
            self.algorithm
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionSequence {
    pub id: String,
    pub token_ids: Vec<i64>,
}

impl RetentionSequence {
    fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "retention sequence id is empty");
        ensure!(
            self.token_ids.len() >= 2 && self.token_ids.iter().all(|token| *token >= 0),
            "retention sequence `{}` needs at least two non-negative token ids",
            self.id
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionSuiteArtifact {
    pub version: u32,
    pub stable_anchors: Vec<RetentionSequence>,
    pub incorporation: Vec<RetentionSequence>,
}

impl RetentionSuiteArtifact {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == RETENTION_SUITE_VERSION,
            "unsupported retention-suite version {}",
            self.version
        );
        ensure!(
            !self.stable_anchors.is_empty() && !self.incorporation.is_empty(),
            "retention suite requires stable-anchor and incorporation sequences"
        );
        let mut ids = BTreeSet::new();
        for sequence in self.stable_anchors.iter().chain(&self.incorporation) {
            sequence.validate()?;
            ensure!(
                ids.insert(sequence.id.as_str()),
                "retention suite repeats sequence id `{}`",
                sequence.id
            );
        }
        Ok(())
    }
}

/// Frozen causal-likelihood evaluator over an independently pinned suite.
pub struct PinnedLikelihoodRetentionEvaluator {
    artifact_sha256: String,
    suite_sha256: String,
    suite: RetentionSuiteArtifact,
    device: Device,
}

impl PinnedLikelihoodRetentionEvaluator {
    pub fn load(
        evaluator_path: &Path,
        evaluator_sha256: &str,
        suite_path: &Path,
        suite_sha256: &str,
        device: Device,
    ) -> Result<Self> {
        let evaluator: LikelihoodRetentionEvaluatorArtifact = read_pinned_json(
            evaluator_path,
            evaluator_sha256,
            "retention-evaluator artifact",
        )?;
        evaluator.validate()?;
        let suite: RetentionSuiteArtifact =
            read_pinned_json(suite_path, suite_sha256, "retention suite")?;
        suite.validate()?;
        Ok(Self {
            artifact_sha256: evaluator_sha256.to_owned(),
            suite_sha256: suite_sha256.to_owned(),
            suite,
            device,
        })
    }

    fn score_sequences(&self, model: &Transformer, sequences: &[RetentionSequence]) -> Result<f32> {
        let mut total = 0.0_f64;
        for (index, sequence) in sequences.iter().enumerate() {
            ensure!(
                sequence.token_ids.len() <= model.config().max_seq_len,
                "retention sequence `{}` exceeds model context {}",
                sequence.id,
                model.config().max_seq_len
            );
            ensure!(
                sequence
                    .token_ids
                    .iter()
                    .all(|token| (*token as usize) < model.config().vocab_size),
                "retention sequence `{}` contains an out-of-vocabulary token",
                sequence.id
            );
            let mut seed_hasher = Sha256::new();
            seed_hasher.update(b"hermes-retention-evaluation-v1\0");
            seed_hasher.update(self.artifact_sha256.as_bytes());
            seed_hasher.update((index as u64).to_le_bytes());
            seed_hasher.update(RETENTION_EVALUATION_DOMAIN.to_le_bytes());
            let digest = seed_hasher.finalize();
            self.device.seed(u64::from_le_bytes(
                digest[..8].try_into().expect("SHA-256 has eight bytes"),
            ));
            let sequence_len = sequence.token_ids.len() - 1;
            let input = Tensor::<2, Int>::from_data(
                TensorData::new(
                    sequence.token_ids[..sequence_len].to_vec(),
                    [1, sequence_len],
                ),
                &self.device,
            );
            let targets = Tensor::<2, Int>::from_data(
                TensorData::new(sequence.token_ids[1..].to_vec(), [1, sequence_len]),
                &self.device,
            );
            let loss = model
                .forward_loss(input, targets)
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .context("reading retention likelihood")?[0];
            ensure!(
                loss.is_finite() && loss >= 0.0,
                "retention sequence `{}` produced invalid loss {loss}",
                sequence.id
            );
            total += f64::from((-loss.min(80.0)).exp());
        }
        Ok((total / sequences.len() as f64) as f32)
    }
}

impl RetentionEvaluator for PinnedLikelihoodRetentionEvaluator {
    fn artifact_hash(&self) -> &str {
        &self.artifact_sha256
    }

    fn suite_hash(&self) -> &str {
        &self.suite_sha256
    }

    fn anchor_rollouts(&mut self, _: &ConsolidationTxn) -> Result<Vec<TokenRolloutBatch>> {
        self.suite
            .stable_anchors
            .iter()
            .map(|sequence| {
                TokenRolloutBatch::new(1, sequence.token_ids.len(), sequence.token_ids.clone())
            })
            .collect()
    }

    fn score(&mut self, _: &ConsolidationTxn, model: &Transformer) -> Result<RetentionScores> {
        Ok(RetentionScores {
            stable_anchor: self.score_sequences(model, &self.suite.stable_anchors)?,
            incorporation: self.score_sequences(model, &self.suite.incorporation)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::RngReservation;

    fn hash(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn write_json(path: &Path, value: &impl Serialize) -> String {
        let bytes = serde_json::to_vec_pretty(value).unwrap();
        fs::write(path, &bytes).unwrap();
        sha256_bytes(&bytes)
    }

    fn transaction(teacher_hash: String) -> ConsolidationTxn {
        ConsolidationTxn {
            id: 9,
            trigger_clock: 12,
            sender: 0,
            receiver: 1,
            receiver_slot: 0,
            terminal: false,
            sender_slots_to_reset: Vec::new(),
            teacher_checkpoint: "teacher.safetensors".into(),
            teacher_hash,
            student_checkpoint: "student.safetensors".into(),
            student_hash: hash('2'),
            prospective_update_hash: hash('3'),
            candidate_checkpoint: None,
            candidate_hash: None,
            knowledge_rng: Some(RngReservation {
                stream: 0,
                start: 3,
                count: 2,
            }),
            imitation_rng: Some(RngReservation {
                stream: 0,
                start: 5,
                count: 2,
            }),
            dream_generation_rng: None,
            dream_selection_rng: None,
            dream_trial_rngs: Vec::new(),
            tensor_transaction_generation: None,
            tensor_transaction_manifest_hash: None,
            generated_manifest: None,
            dream_shared_checkpoint_hash: None,
            dream_selected: Vec::new(),
            dream_trials: Vec::new(),
            dream_policy_receipt: None,
            committed: false,
        }
    }

    #[test]
    fn immutable_journal_is_content_pinned_and_cannot_be_replaced() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("wake-contexts.json");
        let mut journal = WakeContextJournal::new(hash('1')).unwrap();
        journal
            .push(WakeContextRecord {
                id: "pretrain:7:0".into(),
                optimizer_step: 7,
                token_ids: vec![1, 2, 3],
            })
            .unwrap();
        let pinned = journal.publish(&path).unwrap();
        assert_eq!(pinned.records().len(), 1);
        assert_eq!(
            PinnedWakeContextJournal::load(&path, pinned.sha256())
                .unwrap()
                .source_checkpoint_sha256(),
            hash('1')
        );

        journal.records[0].token_ids.push(4);
        assert!(journal.publish(&path).is_err());
        fs::write(&path, b"{}").unwrap();
        assert!(PinnedWakeContextJournal::load(&path, pinned.sha256()).is_err());
    }

    #[test]
    fn pinned_semantic_judge_applies_frozen_aliases_and_ngrams() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("judge.json");
        let artifact = TokenSemanticJudgeArtifact {
            version: SEMANTIC_JUDGE_ARTIFACT_VERSION,
            algorithm: "token_ngram_f1_v1".into(),
            unigram_weight: 0.25,
            bigram_weight: 0.75,
            equivalence_classes: vec![vec![2, 7]],
        };
        let digest = write_json(&path, &artifact);
        let mut judge = PinnedTokenSemanticJudge::load(&path, &digest).unwrap();
        assert_eq!(judge.artifact_hash(), digest);
        assert_eq!(judge.score(&[1], &[1, 2, 3], &[1, 7, 3]).unwrap(), 1.0);
        assert!(judge.score(&[1], &[1, 2, 3], &[3, 2, 1]).unwrap() < 0.5);
    }

    #[test]
    fn journal_rollouts_are_model_generated_replayable_and_teacher_bound() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("wake-contexts.json");
        let teacher_hash = hash('1');
        let mut journal = WakeContextJournal::new(teacher_hash.clone()).unwrap();
        journal
            .push(WakeContextRecord {
                id: "wake:1:0".into(),
                optimizer_step: 1,
                token_ids: vec![1, 2, 3, 4],
            })
            .unwrap();
        let pinned = journal.publish(&path).unwrap();
        let model = Transformer::new(
            &hermes_llm::parse_mal(
                r#"
                ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
                model rollout-test {
                    vocab_size: 16 max_seq_len: 12 hidden_size: 8 num_layers: 1
                    block: {
                        attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
                        ffn: base dropout: 0.0
                    }
                    embeddings { dropout: 0.0 }
                }
                "#,
            )
            .unwrap(),
            &Device::ndarray(),
        )
        .unwrap();
        let config = JournalRolloutConfig {
            max_context_tokens: 4,
            continuation_tokens: 2,
            temperature: 0.8,
            top_k: 4,
            repetition_penalty: 1.0,
            eos_token: None,
            imitation_groups: 1,
        };
        let txn = transaction(teacher_hash);
        let run = |pinned: PinnedWakeContextJournal| {
            let mut rollouts =
                JournalRollouts::new(pinned, Device::ndarray(), config.clone()).unwrap();
            rollouts
                .knowledge_rollouts(&txn, RolloutOwner::Teacher, &model, 2)
                .unwrap()
        };
        let first = run(pinned.clone());
        let second = run(pinned.clone());
        assert_eq!(first, second);
        assert!(first.iter().all(|batch| batch.sequence == 6));

        let mut wrong = JournalRollouts::new(pinned, Device::ndarray(), config).unwrap();
        let mut wrong_txn = txn;
        wrong_txn.teacher_hash = hash('9');
        assert!(
            wrong
                .knowledge_rollouts(&wrong_txn, RolloutOwner::Teacher, &model, 1)
                .is_err()
        );
    }

    #[test]
    fn pinned_retention_artifacts_are_independent_and_strict() {
        let directory = tempfile::tempdir().unwrap();
        let evaluator_path = directory.path().join("evaluator.json");
        let suite_path = directory.path().join("suite.json");
        let evaluator = LikelihoodRetentionEvaluatorArtifact {
            version: RETENTION_EVALUATOR_ARTIFACT_VERSION,
            algorithm: "causal_likelihood_v1".into(),
        };
        let suite = RetentionSuiteArtifact {
            version: RETENTION_SUITE_VERSION,
            stable_anchors: vec![RetentionSequence {
                id: "anchor".into(),
                token_ids: vec![1, 2, 3],
            }],
            incorporation: vec![RetentionSequence {
                id: "incorporation".into(),
                token_ids: vec![3, 2, 1],
            }],
        };
        let evaluator_hash = write_json(&evaluator_path, &evaluator);
        let suite_hash = write_json(&suite_path, &suite);
        let evaluator = PinnedLikelihoodRetentionEvaluator::load(
            &evaluator_path,
            &evaluator_hash,
            &suite_path,
            &suite_hash,
            Device::ndarray(),
        )
        .unwrap();
        assert_eq!(evaluator.artifact_hash(), evaluator_hash);
        assert_eq!(evaluator.suite_hash(), suite_hash);
        assert!(
            PinnedLikelihoodRetentionEvaluator::load(
                &evaluator_path,
                &suite_hash,
                &suite_path,
                &suite_hash,
                Device::ndarray(),
            )
            .is_err()
        );
    }
}
