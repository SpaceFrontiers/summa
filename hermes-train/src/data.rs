//! Streaming objective data, deterministic shuffling, and tensor batches.
//!
//! Causal-LM documents are EOS-joined and packed without padding. Structured
//! objectives use explicit JSONL contracts and fixed shapes: target-only loss
//! positions prevent EOS padding or prompts from contributing to supervised
//! losses, while retrieval batches retain positive and hard-negative grouping.

use std::ffi::{CString, OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;
use hermes_train::corpus::AuthenticatedCorpus;
use hermes_train::task::TaskConfig;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

mod batch;
mod structured;

pub(crate) use batch::{
    BatchStats, EncodedText, LanguageBatch, RetrievalBatch, TrainingBatch, TrainingSample,
    make_batch,
};
use structured::visit_structured_samples;
pub(crate) use structured::{
    OversizedRecordPolicy, OversizedSupervisedRecord, encode_retrieval_text, frame_supervised,
};

const TOKENIZE_BATCH: usize = 1_000;
const TOKEN_CACHE_MAGIC: &[u8; 8] = b"HERTOK01";
// Fixed-size sidecar, rather than an in-band footer, keeps the append/repair
// record format unchanged. Its final 32 bytes authenticate all preceding
// metadata; the cache digest is also retained for full-scan identity checks.
const TOKEN_CACHE_INDEX_MAGIC: &[u8; 8] = b"HERTIX01";
const TOKEN_CACHE_INDEX_BYTES: usize = 144;
const MAX_CACHED_DOCUMENT_TOKENS: usize = 100_000_000;
/// One malformed input record must not make training allocate an entire
/// (possibly decompressed) multi-gigabyte source. For JSONL this bound excludes
/// the terminal LF, matching corpus serialization limits. It remains far above
/// supported model context sizes.
const MAX_TRAINING_RECORD_BYTES: usize = 64 * 1024 * 1024;
/// `TOKENIZE_BATCH` bounds dispatch overhead, while this independently bounds
/// the raw text retained before batch tokenization.
const MAX_TOKENIZE_BATCH_BYTES: usize = MAX_TRAINING_RECORD_BYTES;
#[cfg(not(test))]
const AUTHENTICATED_READ_BUFFER_BYTES: usize = 1024 * 1024;
#[cfg(test)]
const AUTHENTICATED_READ_BUFFER_BYTES: usize = 4 * 1024;

/// The exact immutable data generation assigned to one workflow phase.
///
/// Both variants retain authenticated handles for the lifetime of training.
/// Consequently, run-signature construction, planning, token-cache creation,
/// and every epoch all refer to the same generation instead of hashing a path
/// and reopening whatever that path happens to name later.
#[derive(Clone)]
pub(crate) struct PhaseDataBinding {
    configured_path: PathBuf,
    source: BoundPhaseData,
}

#[derive(Clone)]
enum BoundPhaseData {
    Corpus {
        signature_identity: String,
        corpus: Arc<AuthenticatedCorpus>,
    },
    Direct(Arc<AuthenticatedPhaseFile>),
}

impl PhaseDataBinding {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        if let Some(corpus) = AuthenticatedCorpus::open_data_path(path)? {
            let signature_identity = format!("sha256:{}", corpus.manifest().manifest_sha256);
            return Ok(Self {
                configured_path: path.to_owned(),
                source: BoundPhaseData::Corpus {
                    signature_identity,
                    corpus: Arc::new(corpus),
                },
            });
        }
        Ok(Self {
            configured_path: path.to_owned(),
            source: BoundPhaseData::Direct(Arc::new(AuthenticatedPhaseFile::open(path)?)),
        })
    }

    pub(crate) fn signature_identity(&self) -> &str {
        match &self.source {
            BoundPhaseData::Corpus {
                signature_identity, ..
            } => signature_identity,
            BoundPhaseData::Direct(file) => &file.signature_identity,
        }
    }

    pub(crate) fn authenticated_corpus(&self) -> Option<&AuthenticatedCorpus> {
        match &self.source {
            BoundPhaseData::Corpus { corpus, .. } => Some(corpus),
            BoundPhaseData::Direct(_) => None,
        }
    }

    fn configured_path(&self) -> &Path {
        &self.configured_path
    }

    fn ensure_matches_path(&self, path: &Path) -> Result<()> {
        ensure!(
            path == self.configured_path(),
            "phase data binding for {} cannot be used to read {}",
            self.configured_path().display(),
            path.display()
        );
        Ok(())
    }

    /// Reject replacement, symlinking, or in-place mutation of the published
    /// input, including on token-cache paths that do not need source bytes.
    pub(crate) fn ensure_still_published(&self) -> Result<()> {
        match &self.source {
            BoundPhaseData::Corpus { corpus, .. } => corpus.ensure_still_published(),
            BoundPhaseData::Direct(file) => file.ensure_still_published(),
        }
    }

    /// Stream every source through authenticated handles. The final identity
    /// check deliberately runs even when the visitor errors or exits early.
    pub(crate) fn with_readers(
        &self,
        path: &Path,
        mut visit: impl FnMut(&Path, &mut dyn BufRead) -> Result<bool>,
    ) -> Result<bool> {
        self.ensure_matches_path(path)?;
        self.ensure_still_published()?;
        let result = match &self.source {
            BoundPhaseData::Corpus { corpus, .. } => (|| -> Result<bool> {
                let mut keep_going = true;
                for index in 0..corpus.shard_count() {
                    keep_going = corpus.with_verified_shard(index, |source_path, file| {
                        with_decoded_reader(source_path, file, |reader| visit(source_path, reader))
                    })?;
                    if !keep_going {
                        break;
                    }
                }
                Ok(keep_going)
            })(),
            BoundPhaseData::Direct(file) => {
                file.with_reader(|source_path, reader| visit(source_path, reader))
            }
        };
        // Integrity errors take precedence over visitor errors: callers must
        // never mistake an errored or shortened pass over mutated data for a
        // harmless application-level failure.
        self.ensure_still_published()?;
        result
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StablePhaseFileIdentity {
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

impl StablePhaseFileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

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

struct AuthenticatedPhaseFile {
    path: PathBuf,
    identity: StablePhaseFileIdentity,
    signature_identity: String,
    file: Mutex<File>,
}

impl AuthenticatedPhaseFile {
    #[cfg(not(unix))]
    fn open(path: &Path) -> Result<Self> {
        anyhow::bail!(
            "exact direct phase-data binding for {} requires stable Unix file identities; use an immutable prepared-corpus manifest on this platform",
            path.display()
        )
    }

    #[cfg(unix)]
    fn open(path: &Path) -> Result<Self> {
        let mut file = open_published_phase_file(path)?;
        let identity = StablePhaseFileIdentity::from_metadata(&file.metadata()?);
        let signature_identity = hash_stable_phase_file(&mut file, &identity, path)?;
        ensure_published_phase_file(path, &identity)?;
        Ok(Self {
            path: path.to_owned(),
            identity,
            signature_identity,
            file: Mutex::new(file),
        })
    }

    fn with_reader(
        &self,
        read: impl FnOnce(&Path, &mut dyn BufRead) -> Result<bool>,
    ) -> Result<bool> {
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("phase-data handle lock was poisoned"))?;
        file.rewind()
            .with_context(|| format!("failed to rewind phase data {}", self.path.display()))?;
        let verified = VerifiedPhaseReader {
            file: &mut file,
            expected: &self.identity,
            path: &self.path,
        };
        with_decoded_reader(&self.path, verified, |reader| read(&self.path, reader))
    }

    fn ensure_still_published(&self) -> Result<()> {
        let file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("phase-data handle lock was poisoned"))?;
        let observed =
            StablePhaseFileIdentity::from_metadata(&file.metadata().with_context(|| {
                format!("failed to inspect open phase data {}", self.path.display())
            })?);
        ensure!(
            observed == self.identity,
            "phase data {} changed in place after authentication",
            self.path.display()
        );
        ensure_published_phase_file(&self.path, &self.identity)
    }
}

/// Detect in-place mutation before any bytes from that read reach parsing,
/// tokenization, prefetch, or a durable trainer checkpoint. A large outer
/// buffer keeps this to roughly one `fstat` per MiB for uncompressed inputs.
struct VerifiedPhaseReader<'a> {
    file: &'a mut File,
    expected: &'a StablePhaseFileIdentity,
    path: &'a Path,
}

impl Read for VerifiedPhaseReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.file.read(buffer)?;
        let observed = StablePhaseFileIdentity::from_metadata(&self.file.metadata()?);
        if observed != *self.expected {
            return Err(std::io::Error::other(format!(
                "phase data {} changed in place while it was streamed",
                self.path.display()
            )));
        }
        Ok(read)
    }
}

#[cfg(unix)]
fn open_published_phase_file(path: &Path) -> Result<File> {
    let inspected = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect phase data {}", path.display()))?;
    ensure!(
        inspected.file_type().is_file() && !inspected.file_type().is_symlink(),
        "phase data {} must be a regular non-symlink file",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("failed to securely open phase data {}", path.display()))?;
    let opened = file.metadata()?;
    ensure!(
        opened.file_type().is_file(),
        "opened phase data {} is not a regular file",
        path.display()
    );
    ensure!(
        StablePhaseFileIdentity::from_metadata(&opened)
            == StablePhaseFileIdentity::from_metadata(&inspected),
        "phase data {} changed while it was opened",
        path.display()
    );
    Ok(file)
}

#[cfg(unix)]
fn ensure_published_phase_file(path: &Path, expected: &StablePhaseFileIdentity) -> Result<()> {
    let published = open_published_phase_file(path)?;
    ensure!(
        StablePhaseFileIdentity::from_metadata(&published.metadata()?) == *expected,
        "published phase data {} changed after authentication",
        path.display()
    );
    Ok(())
}

#[cfg(not(unix))]
fn ensure_published_phase_file(path: &Path, _expected: &StablePhaseFileIdentity) -> Result<()> {
    anyhow::bail!(
        "exact direct phase-data binding for {} requires stable Unix file identities",
        path.display()
    )
}

#[cfg(unix)]
fn hash_stable_phase_file(
    file: &mut File,
    expected: &StablePhaseFileIdentity,
    path: &Path,
) -> Result<String> {
    file.rewind()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes_read = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash phase data {}", path.display()))?;
        if read == 0 {
            break;
        }
        bytes_read = bytes_read
            .checked_add(u64::try_from(read)?)
            .context("phase-data byte count overflows u64")?;
        hasher.update(&buffer[..read]);
    }
    let observed = StablePhaseFileIdentity::from_metadata(&file.metadata()?);
    ensure!(
        observed == *expected && bytes_read == expected.length,
        "phase data {} changed while its run-signature identity was computed",
        path.display()
    );
    file.rewind()?;
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn with_decoded_reader<T, R: Read>(
    path: &Path,
    reader: R,
    read: impl FnOnce(&mut dyn BufRead) -> Result<T>,
) -> Result<T> {
    if path.extension().is_some_and(|extension| extension == "zst") {
        let compressed = BufReader::with_capacity(AUTHENTICATED_READ_BUFFER_BYTES, reader);
        let decoder = zstd::stream::read::Decoder::with_buffer(compressed)
            .with_context(|| format!("failed to open zstd stream {}", path.display()))?;
        let mut reader = BufReader::new(decoder);
        read(&mut reader)
    } else {
        let mut reader = BufReader::with_capacity(AUTHENTICATED_READ_BUFFER_BYTES, reader);
        read(&mut reader)
    }
}

fn read_training_jsonl_record_bounded<'a>(
    reader: &mut (impl BufRead + ?Sized),
    output: &'a mut Vec<u8>,
    maximum_bytes: usize,
    path: &Path,
    line_number: usize,
) -> Result<Option<&'a str>> {
    ensure!(
        maximum_bytes > 0,
        "training JSONL record byte limit must be positive"
    );
    output.clear();
    let capture_bytes = maximum_bytes
        .checked_add(1)
        .context("training JSONL record byte limit overflows usize")?;
    let read = reader
        .take(u64::try_from(capture_bytes).context("training JSONL record byte limit exceeds u64")?)
        .read_until(b'\n', output)
        .with_context(|| {
            format!(
                "failed to read training JSONL record at {}:{line_number}",
                path.display()
            )
        })?;
    if read == 0 {
        return Ok(None);
    }
    let record_bytes = output
        .len()
        .checked_sub(usize::from(output.last() == Some(&b'\n')))
        .context("training JSONL record byte count underflows usize")?;
    ensure!(
        record_bytes <= maximum_bytes,
        "training JSONL record at {}:{line_number} exceeds the maximum of {maximum_bytes} bytes before its newline",
        path.display()
    );
    let record = std::str::from_utf8(output).with_context(|| {
        format!(
            "training JSONL record at {}:{line_number} is not UTF-8",
            path.display()
        )
    })?;
    Ok(Some(record))
}

pub(super) fn read_training_jsonl_record<'a>(
    reader: &mut (impl BufRead + ?Sized),
    output: &'a mut Vec<u8>,
    path: &Path,
    line_number: usize,
) -> Result<Option<&'a str>> {
    read_training_jsonl_record_bounded(reader, output, MAX_TRAINING_RECORD_BYTES, path, line_number)
}

fn read_training_text_document_bounded(
    reader: &mut (impl BufRead + ?Sized),
    maximum_bytes: usize,
    path: &Path,
) -> Result<String> {
    ensure!(
        maximum_bytes > 0,
        "training text document byte limit must be positive"
    );
    let capture_bytes = maximum_bytes
        .checked_add(1)
        .context("training text document byte limit overflows usize")?;
    let mut document = Vec::new();
    reader
        .take(
            u64::try_from(capture_bytes)
                .context("training text document byte limit exceeds u64")?,
        )
        .read_to_end(&mut document)
        .with_context(|| format!("failed to read training text document {}", path.display()))?;
    ensure!(
        document.len() <= maximum_bytes,
        "training text document {} exceeds the maximum of {maximum_bytes} bytes",
        path.display()
    );
    String::from_utf8(document).with_context(|| {
        format!(
            "training text document {} is not valid UTF-8",
            path.display()
        )
    })
}

struct TokenCacheWriter {
    writer: BufWriter<File>,
    location: TokenCacheLocation,
    digest: Sha256,
    documents: u64,
    stream_tokens: u64,
    index_invalidated: bool,
}

impl TokenCacheWriter {
    fn append(&mut self, tokens: &[u32]) -> Result<()> {
        if !self.index_invalidated {
            invalidate_token_cache_index(&self.location)?;
            self.index_invalidated = true;
        }
        let len = u32::try_from(tokens.len()).context("document is too large for token cache")?;
        let len_bytes = len.to_le_bytes();
        self.writer.write_all(&len_bytes)?;
        self.digest.update(len_bytes);
        for token in tokens {
            let token_bytes = token.to_le_bytes();
            self.writer.write_all(&token_bytes)?;
            self.digest.update(token_bytes);
        }
        self.documents = self
            .documents
            .checked_add(1)
            .context("token-cache document count overflows u64")?;
        self.stream_tokens = self
            .stream_tokens
            .checked_add(u64::from(len) + 1)
            .context("token-cache stream length overflows u64")?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("failed to flush token cache")
    }

    /// Durably mark this cache as a complete scan of its authoritative corpus.
    /// Partial caches deliberately have no valid index and are replayed on the
    /// next startup before an index can be published.
    fn publish_index(&mut self) -> Result<()> {
        self.flush()?;
        self.writer
            .get_ref()
            .sync_all()
            .context("failed to sync token cache")?;
        let metadata = self.writer.get_ref().metadata()?;
        let index = TokenCacheIndex {
            identity: TokenCacheFileIdentity::from_metadata(&metadata),
            documents: self.documents,
            stream_tokens: self.stream_tokens,
            cache_digest: self.digest.clone().finalize().into(),
        };
        index.validate_structure()?;

        // A valid index for this exact cache incarnation is immutable. Avoid
        // replacing it on every startup after the authoritative rescan.
        if read_token_cache_index(&self.location, self.writer.get_ref())?.as_ref() == Some(&index) {
            self.index_invalidated = false;
            return Ok(());
        }
        write_token_cache_index_atomic(&self.location, &index.encode())?;
        self.index_invalidated = false;
        Ok(())
    }

    /// Discard all derived records after an authoritative-input integrity
    /// failure. Keeping even an unindexed prefix is unsafe: a later retry would
    /// replay it and skip the corresponding source documents.
    fn reset_to_empty(&mut self) -> Result<()> {
        self.writer.flush().context("flushing failed token cache")?;
        reset_open_token_cache(&self.location, self.writer.get_mut())?;
        self.digest = Sha256::new();
        self.digest.update(TOKEN_CACHE_MAGIC);
        self.documents = 0;
        self.stream_tokens = 0;
        self.index_invalidated = true;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TokenCacheFileIdentity {
    len: u64,
    device: u64,
    inode: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl TokenCacheFileIdentity {
    #[cfg(unix)]
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;

        Self {
            len: metadata.len(),
            device: metadata.dev(),
            inode: metadata.ino(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    #[cfg(not(unix))]
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        use std::time::UNIX_EPOCH;

        let modified = metadata.modified().ok().and_then(|time| {
            time.duration_since(UNIX_EPOCH).ok().and_then(|duration| {
                i64::try_from(duration.as_secs())
                    .ok()
                    .map(|s| (s, duration))
            })
        });
        Self {
            len: metadata.len(),
            device: 0,
            inode: 0,
            modified_seconds: modified.as_ref().map_or(0, |(seconds, _)| *seconds),
            modified_nanoseconds: modified
                .map_or(0, |(_, duration)| i64::from(duration.subsec_nanos())),
            changed_seconds: 0,
            changed_nanoseconds: 0,
        }
    }

    /// Compare an index identity with an already-open cache handle when the
    /// platform exposes a stable file incarnation. Portable `Metadata` only
    /// guarantees length and timestamps; accepting those as an identity would
    /// let a same-length replacement with a preserved timestamp reuse stale
    /// counts. `None` therefore means the sidecar is only derived metadata and
    /// the cache must be structurally replayed against its authoritative input.
    fn matches_metadata(&self, metadata: &fs::Metadata) -> Option<bool> {
        #[cfg(unix)]
        {
            Some(self == &Self::from_metadata(metadata))
        }
        #[cfg(not(unix))]
        {
            let _ = (self, metadata);
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TokenCacheIndex {
    identity: TokenCacheFileIdentity,
    documents: u64,
    stream_tokens: u64,
    cache_digest: [u8; 32],
}

impl TokenCacheIndex {
    fn validate_structure(&self) -> Result<()> {
        ensure!(
            self.identity.len >= TOKEN_CACHE_MAGIC.len() as u64,
            "indexed token cache is shorter than its header"
        );
        let record_bytes = self.identity.len - TOKEN_CACHE_MAGIC.len() as u64;
        ensure!(
            record_bytes.is_multiple_of(std::mem::size_of::<u32>() as u64),
            "indexed token cache is not u32-aligned"
        );
        let stream_tokens = record_bytes / std::mem::size_of::<u32>() as u64;
        ensure!(
            self.stream_tokens == stream_tokens,
            "token-cache index stream length does not match cache length"
        );
        ensure!(
            self.documents <= self.stream_tokens,
            "token-cache index has more documents than stream tokens"
        );
        Ok(())
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TOKEN_CACHE_INDEX_BYTES);
        bytes.extend_from_slice(TOKEN_CACHE_INDEX_MAGIC);
        bytes.extend_from_slice(&self.identity.len.to_le_bytes());
        bytes.extend_from_slice(&self.identity.device.to_le_bytes());
        bytes.extend_from_slice(&self.identity.inode.to_le_bytes());
        bytes.extend_from_slice(&self.identity.modified_seconds.to_le_bytes());
        bytes.extend_from_slice(&self.identity.modified_nanoseconds.to_le_bytes());
        bytes.extend_from_slice(&self.identity.changed_seconds.to_le_bytes());
        bytes.extend_from_slice(&self.identity.changed_nanoseconds.to_le_bytes());
        bytes.extend_from_slice(&self.documents.to_le_bytes());
        bytes.extend_from_slice(&self.stream_tokens.to_le_bytes());
        bytes.extend_from_slice(&self.cache_digest);
        let metadata_digest = Sha256::digest(&bytes);
        bytes.extend_from_slice(&metadata_digest);
        debug_assert_eq!(bytes.len(), TOKEN_CACHE_INDEX_BYTES);
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != TOKEN_CACHE_INDEX_BYTES
            || &bytes[..TOKEN_CACHE_INDEX_MAGIC.len()] != TOKEN_CACHE_INDEX_MAGIC
        {
            return None;
        }
        let payload_end = TOKEN_CACHE_INDEX_BYTES - 32;
        if Sha256::digest(&bytes[..payload_end]).as_slice() != &bytes[payload_end..] {
            return None;
        }
        let mut offset = TOKEN_CACHE_INDEX_MAGIC.len();
        let mut read_u64 = || {
            let end = offset.checked_add(8)?;
            let value = u64::from_le_bytes(bytes.get(offset..end)?.try_into().ok()?);
            offset = end;
            Some(value)
        };
        let len = read_u64()?;
        let device = read_u64()?;
        let inode = read_u64()?;
        let modified_seconds = i64::from_le_bytes(read_u64()?.to_le_bytes());
        let modified_nanoseconds = i64::from_le_bytes(read_u64()?.to_le_bytes());
        let changed_seconds = i64::from_le_bytes(read_u64()?.to_le_bytes());
        let changed_nanoseconds = i64::from_le_bytes(read_u64()?.to_le_bytes());
        let documents = read_u64()?;
        let stream_tokens = read_u64()?;
        let digest_end = offset.checked_add(32)?;
        let cache_digest = bytes.get(offset..digest_end)?.try_into().ok()?;
        if digest_end != payload_end {
            return None;
        }
        Some(Self {
            identity: TokenCacheFileIdentity {
                len,
                device,
                inode,
                modified_seconds,
                modified_nanoseconds,
                changed_seconds,
                changed_nanoseconds,
            },
            documents,
            stream_tokens,
            cache_digest,
        })
    }
}

#[cfg(test)]
fn token_cache_index_path(cache_path: &Path) -> PathBuf {
    let mut index_path = cache_path.as_os_str().to_os_string();
    index_path.push(".index");
    PathBuf::from(index_path)
}

/// An opened cache directory anchors all cache/index operations to one real
/// directory object. On Unix, child operations use `openat`/`renameat`/
/// `unlinkat`, so replacing `.token-cache` or a child with a symlink cannot
/// redirect an already-running trainer outside its output directory.
struct TokenCacheLocation {
    #[cfg(unix)]
    directory: File,
    directory_path: PathBuf,
    cache_name: OsString,
    index_name: OsString,
}

impl TokenCacheLocation {
    fn open(cache_path: &Path, create_directory: bool) -> Result<Option<Self>> {
        let directory_path = cache_path.parent().unwrap_or_else(|| Path::new("."));
        if create_directory {
            fs::create_dir_all(directory_path).with_context(|| {
                format!(
                    "failed to create token-cache directory {}",
                    directory_path.display()
                )
            })?;
        }
        let metadata = match fs::symlink_metadata(directory_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound && !create_directory => {
                return Ok(None);
            }
            Err(error) => return Err(error).context("failed to inspect token-cache directory"),
        };
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "token-cache parent {} must be a real directory",
            directory_path.display()
        );

        #[cfg(unix)]
        let directory = {
            use std::os::unix::fs::MetadataExt;

            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
                .open(directory_path)
                .with_context(|| {
                    format!(
                        "failed to securely open token-cache directory {}",
                        directory_path.display()
                    )
                })?;
            let opened = file.metadata()?;
            ensure!(
                opened.dev() == metadata.dev() && opened.ino() == metadata.ino(),
                "token-cache directory {} changed while it was opened",
                directory_path.display()
            );
            file
        };

        let cache_name = cache_path
            .file_name()
            .context("token-cache path has no file name")?
            .to_os_string();
        let mut index_name = cache_name.clone();
        index_name.push(".index");
        Ok(Some(Self {
            #[cfg(unix)]
            directory,
            directory_path: directory_path.to_owned(),
            cache_name,
            index_name,
        }))
    }

    fn index_path(&self) -> PathBuf {
        self.directory_path.join(&self.index_name)
    }

    #[cfg(unix)]
    fn open_child(&self, name: &OsStr, flags: libc::c_int) -> Result<Option<File>> {
        let name = CString::new(name.as_bytes()).context("token-cache name contains a NUL byte")?;
        let descriptor = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
                0o600,
            )
        };
        if descriptor < 0 {
            let error = std::io::Error::last_os_error();
            if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::AlreadyExists) {
                return Ok(None);
            }
            return Err(error).with_context(|| {
                format!(
                    "failed to securely open an entry in token-cache directory {}",
                    self.directory_path.display()
                )
            });
        }
        let file = unsafe { File::from_raw_fd(descriptor) };
        ensure!(
            file.metadata()?.file_type().is_file(),
            "token-cache entry is not a regular file"
        );
        Ok(Some(file))
    }

    fn open_cache(&self, create: bool) -> Result<Option<File>> {
        #[cfg(unix)]
        {
            let mut flags = if create { libc::O_RDWR } else { libc::O_RDONLY };
            if create {
                flags |= libc::O_CREAT;
            }
            self.open_child(&self.cache_name, flags)
        }
        #[cfg(not(unix))]
        {
            self.open_child_portable(&self.cache_name, create, create)
        }
    }

    fn open_existing_cache_for_reset(&self) -> Result<Option<File>> {
        #[cfg(unix)]
        {
            self.open_child(&self.cache_name, libc::O_RDWR)
        }
        #[cfg(not(unix))]
        {
            self.open_child_portable(&self.cache_name, false, true)
        }
    }

    fn open_index(&self) -> Result<Option<File>> {
        #[cfg(unix)]
        {
            self.open_child(&self.index_name, libc::O_RDONLY)
        }
        #[cfg(not(unix))]
        {
            self.open_child_portable(&self.index_name, false, false)
        }
    }

    #[cfg(not(unix))]
    fn open_child_portable(&self, name: &OsStr, create: bool, write: bool) -> Result<Option<File>> {
        let path = self.directory_path.join(name);
        match fs::symlink_metadata(&path) {
            Ok(metadata) => ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "token-cache entry {} is not a regular file",
                path.display()
            ),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let mut options = OpenOptions::new();
        options.read(true).write(write).create(create);
        match options.open(&path) {
            Ok(file) => {
                ensure!(file.metadata()?.file_type().is_file());
                Ok(Some(file))
            }
            Err(error) if error.kind() == ErrorKind::NotFound && !create => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn create_temporary(&self, name: &OsStr) -> Result<Option<File>> {
        #[cfg(unix)]
        {
            self.open_child(name, libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL)
        }
        #[cfg(not(unix))]
        {
            let path = self.directory_path.join(name);
            match OpenOptions::new().create_new(true).write(true).open(path) {
                Ok(file) => Ok(Some(file)),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
    }

    fn remove_child(&self, name: &OsStr) -> Result<bool> {
        #[cfg(unix)]
        {
            let name =
                CString::new(name.as_bytes()).context("token-cache name contains a NUL byte")?;
            if unsafe { libc::unlinkat(self.directory.as_raw_fd(), name.as_ptr(), 0) } == 0 {
                return Ok(true);
            }
            let error = std::io::Error::last_os_error();
            if error.kind() == ErrorKind::NotFound {
                return Ok(false);
            }
            Err(error.into())
        }
        #[cfg(not(unix))]
        {
            match fs::remove_file(self.directory_path.join(name)) {
                Ok(()) => Ok(true),
                Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
                Err(error) => Err(error.into()),
            }
        }
    }

    fn rename_child(&self, from: &OsStr, to: &OsStr) -> Result<()> {
        #[cfg(unix)]
        {
            let from =
                CString::new(from.as_bytes()).context("token-cache name contains a NUL byte")?;
            let to = CString::new(to.as_bytes()).context("token-cache name contains a NUL byte")?;
            if unsafe {
                libc::renameat(
                    self.directory.as_raw_fd(),
                    from.as_ptr(),
                    self.directory.as_raw_fd(),
                    to.as_ptr(),
                )
            } == 0
            {
                return Ok(());
            }
            Err(std::io::Error::last_os_error().into())
        }
        #[cfg(not(unix))]
        {
            fs::rename(self.directory_path.join(from), self.directory_path.join(to))?;
            Ok(())
        }
    }

    fn sync_directory(&self) -> Result<()> {
        #[cfg(unix)]
        self.directory.sync_all()?;
        #[cfg(not(unix))]
        File::open(&self.directory_path)?.sync_all()?;
        Ok(())
    }
}

enum TokenCacheIndexFile {
    Missing,
    Invalid,
    Decoded(TokenCacheIndex),
}

fn decode_token_cache_index_file(location: &TokenCacheLocation) -> Result<TokenCacheIndexFile> {
    let Some(file) = location.open_index()? else {
        return Ok(TokenCacheIndexFile::Missing);
    };
    let mut bytes = Vec::with_capacity(TOKEN_CACHE_INDEX_BYTES + 1);
    file.take((TOKEN_CACHE_INDEX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    Ok(TokenCacheIndex::decode(&bytes)
        .filter(|index| index.validate_structure().is_ok())
        .map_or(TokenCacheIndexFile::Invalid, TokenCacheIndexFile::Decoded))
}

fn read_token_cache_index(
    location: &TokenCacheLocation,
    cache: &File,
) -> Result<Option<TokenCacheIndex>> {
    let TokenCacheIndexFile::Decoded(index) = decode_token_cache_index_file(location)? else {
        return Ok(None);
    };
    let metadata = cache.metadata()?;
    if !metadata.file_type().is_file() || index.identity.matches_metadata(&metadata) != Some(true) {
        return Ok(None);
    }
    Ok(Some(index))
}

fn write_token_cache_index_atomic(location: &TokenCacheLocation, bytes: &[u8]) -> Result<()> {
    let mut temporary = None;
    for attempt in 0..1_024_u32 {
        let mut name = location.index_name.clone();
        name.push(format!(".tmp-{}-{attempt}", std::process::id()));
        match location.create_temporary(&name)? {
            Some(mut file) => {
                if let Err(error) = (|| {
                    file.write_all(bytes)?;
                    file.sync_all()
                })() {
                    let _ = location.remove_child(&name);
                    return Err(error).with_context(|| {
                        format!(
                            "failed to write token-cache index {}",
                            location.index_path().display()
                        )
                    });
                }
                temporary = Some(name);
                break;
            }
            None => continue,
        }
    }
    let temporary = temporary.context("could not allocate a token-cache index temporary file")?;
    if let Err(error) = location.rename_child(&temporary, &location.index_name) {
        let _ = location.remove_child(&temporary);
        return Err(error)
            .with_context(|| format!("failed to publish {}", location.index_path().display()));
    }
    location.sync_directory()?;
    Ok(())
}

fn invalidate_token_cache_index(location: &TokenCacheLocation) -> Result<()> {
    if location.remove_child(&location.index_name)? {
        location.sync_directory()?;
    }
    Ok(())
}

fn reset_token_cache_path(path: &Path) -> Result<()> {
    let Some(location) = TokenCacheLocation::open(path, false)? else {
        return Ok(());
    };
    let Some(mut cache) = location.open_existing_cache_for_reset()? else {
        invalidate_token_cache_index(&location)?;
        return Ok(());
    };
    reset_open_token_cache(&location, &mut cache)
}

fn reset_open_token_cache(location: &TokenCacheLocation, cache: &mut File) -> Result<()> {
    invalidate_token_cache_index(location)?;
    cache.set_len(0).context("truncating failed token cache")?;
    cache
        .seek(SeekFrom::Start(0))
        .context("rewinding failed token cache")?;
    cache
        .write_all(TOKEN_CACHE_MAGIC)
        .context("rewriting failed token-cache header")?;
    cache.sync_all().context("syncing reset token cache")
}

fn ensure_token_cache_unchanged(
    reader: &mut BufReader<File>,
    expected: &TokenCacheFileIdentity,
    location: &TokenCacheLocation,
    path: &Path,
) -> Result<()> {
    let observed = TokenCacheFileIdentity::from_metadata(&reader.get_ref().metadata()?);
    if observed != *expected {
        reset_open_token_cache(location, reader.get_mut()).with_context(|| {
            format!(
                "failed to reset {} after detecting an in-place mutation",
                path.display()
            )
        })?;
        anyhow::bail!(
            "token cache {} changed while it was replayed; the derived cache was reset",
            path.display()
        );
    }
    Ok(())
}

/// Return the causal sample count from a durably completed token cache without
/// replaying its records. Missing, torn, stale, or corrupt metadata returns
/// `None`, so callers can safely fall back to [`count_samples`].
pub(crate) fn indexed_causal_sample_count(
    token_cache: &Path,
    seq_len: usize,
) -> Result<Option<usize>> {
    ensure!(seq_len > 0, "sequence_length must be positive");
    let Some(location) = TokenCacheLocation::open(token_cache, false)? else {
        return Ok(None);
    };
    let Some(cache) = location.open_cache(false)? else {
        return Ok(None);
    };
    let Some(index) = read_token_cache_index(&location, &cache)? else {
        return Ok(None);
    };
    let sequence_length = u64::try_from(seq_len).context("sequence_length exceeds u64")?;
    let samples = if index.stream_tokens == 0 {
        0
    } else {
        (index.stream_tokens - 1) / sequence_length
    };
    Ok(usize::try_from(samples).ok())
}

/// Replay complete cached documents and open an append-only writer at the
/// first missing document. A torn final record is safely discarded because
/// this cache is derived from the authoritative corpus.
fn replay_token_cache(
    path: &Path,
    eos_token: u32,
    vocab_size: usize,
    packer: &mut SamplePacker,
    count: &mut usize,
    visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<(usize, bool, Option<TokenCacheWriter>, bool)> {
    ensure!(vocab_size > 0, "tokenizer vocabulary must not be empty");
    let location = TokenCacheLocation::open(path, true)?
        .context("token-cache directory disappeared while it was opened")?;
    let mut cache = location
        .open_cache(true)?
        .context("token-cache file was not created")?;
    let mut expected_index = match decode_token_cache_index_file(&location)? {
        TokenCacheIndexFile::Missing => None,
        TokenCacheIndexFile::Invalid => {
            // The sidecar is derived metadata and is atomically replaceable.
            // Preserve a potentially multi-billion-token cache until its own
            // record structure has been scanned; normal authoritative-corpus
            // completion will then publish a fresh index.
            invalidate_token_cache_index(&location)?;
            None
        }
        TokenCacheIndexFile::Decoded(index) => {
            match index.identity.matches_metadata(&cache.metadata()?) {
                Some(true) => Some(index),
                Some(false) => {
                    // A completed cache changed without first invalidating its
                    // sidecar. It is derived data, so discard it and rebuild from
                    // the authoritative source rather than re-blessing payloads.
                    invalidate_token_cache_index(&location)?;
                    cache.set_len(0)?;
                    None
                }
                None => {
                    // This platform cannot prove that the path still names the
                    // indexed file incarnation. Preserve the derived payload,
                    // but require structural replay and an authoritative source
                    // scan instead of treating the sidecar as completion proof.
                    invalidate_token_cache_index(&location)?;
                    None
                }
            }
        }
    };
    if cache.metadata()?.len() < TOKEN_CACHE_MAGIC.len() as u64 {
        cache.set_len(0)?;
        cache.write_all(TOKEN_CACHE_MAGIC)?;
        cache.sync_all()?;
    }
    let replay_identity = TokenCacheFileIdentity::from_metadata(&cache.metadata()?);
    cache.rewind()?;

    let mut reader = BufReader::new(cache);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != TOKEN_CACHE_MAGIC {
        // Token caches are derived data. An unrecognized header cannot be
        // replayed safely, but it also must not permanently wedge training:
        // reset it and rebuild from the authenticated authoritative corpus in
        // this invocation.
        let mut cache = reader.into_inner();
        reset_open_token_cache(&location, &mut cache)?;
        cache.seek(SeekFrom::End(0))?;
        let mut digest = Sha256::new();
        digest.update(TOKEN_CACHE_MAGIC);
        return Ok((
            0,
            true,
            Some(TokenCacheWriter {
                writer: BufWriter::new(cache),
                location,
                digest,
                documents: 0,
                stream_tokens: 0,
                index_invalidated: true,
            }),
            false,
        ));
    }
    let mut valid_bytes = TOKEN_CACHE_MAGIC.len() as u64;
    let mut documents = 0usize;
    let mut stream_tokens = 0_u64;
    let mut digest = Sha256::new();
    digest.update(TOKEN_CACHE_MAGIC);
    let mut torn_tail = false;
    loop {
        let mut len_bytes = [0_u8; 4];
        let read = reader.read(&mut len_bytes)?;
        if read == 0 {
            break;
        }
        if read < len_bytes.len()
            && let Err(error) = reader.read_exact(&mut len_bytes[read..])
        {
            if error.kind() == ErrorKind::UnexpectedEof {
                torn_tail = true;
                break;
            }
            return Err(error.into());
        }
        let len = u32::from_le_bytes(len_bytes) as usize;
        ensure!(
            len <= MAX_CACHED_DOCUMENT_TOKENS,
            "token cache {} contains an implausible {len}-token document",
            path.display()
        );
        let byte_len = len
            .checked_mul(std::mem::size_of::<u32>())
            .context("cached document byte length overflows usize")?;
        let mut bytes = vec![0_u8; byte_len];
        if let Err(error) = reader.read_exact(&mut bytes) {
            if error.kind() == ErrorKind::UnexpectedEof {
                torn_tail = true;
                break;
            }
            return Err(error.into());
        }
        let invalid_token = bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte token chunk")))
            .find(|token| usize::try_from(*token).map_or(true, |token| token >= vocab_size));
        if let Some(token) = invalid_token {
            let mut cache = reader.into_inner();
            reset_open_token_cache(&location, &mut cache)?;
            anyhow::bail!(
                "token cache {} contains token id {token} outside tokenizer vocabulary size {vocab_size}; the derived cache was reset and will be rebuilt on retry",
                path.display()
            );
        }
        ensure_token_cache_unchanged(&mut reader, &replay_identity, &location, path)?;
        valid_bytes = valid_bytes
            .checked_add(4 + byte_len as u64)
            .context("token cache offset overflows u64")?;
        documents = documents
            .checked_add(1)
            .context("cached document count overflows usize")?;
        stream_tokens = stream_tokens
            .checked_add(u64::from(len as u32) + 1)
            .context("token-cache stream length overflows u64")?;
        digest.update(len_bytes);
        digest.update(&bytes);
        let tokens = bytes
            .chunks_exact(4)
            .map(|bytes| i64::from(u32::from_le_bytes(bytes.try_into().unwrap())))
            .chain(std::iter::once(i64::from(eos_token)));
        let keep_going = packer.push(tokens, count, visit);
        // A visitor can stop or fail before the whole-cache digest check. The
        // opened cache incarnation must therefore still be checked at that
        // boundary so a concurrent in-place mutation cannot affect a model
        // update and then hide behind the early return.
        ensure_token_cache_unchanged(&mut reader, &replay_identity, &location, path)?;
        if !keep_going? {
            return Ok((documents, false, None, false));
        }
    }
    ensure_token_cache_unchanged(&mut reader, &replay_identity, &location, path)?;
    let mut cache = reader.into_inner();
    let verified_complete = if let Some(expected) = expected_index.take() {
        ensure!(
            !torn_tail,
            "completed token cache {} has a torn record",
            path.display()
        );
        ensure!(
            expected.documents == u64::try_from(documents)?,
            "completed token cache {} document count changed",
            path.display()
        );
        ensure!(
            expected.stream_tokens == stream_tokens,
            "completed token cache {} stream length changed",
            path.display()
        );
        ensure!(
            expected.cache_digest.as_slice() == digest.clone().finalize().as_slice(),
            "completed token cache {} payload digest changed",
            path.display()
        );
        ensure!(
            expected.identity.matches_metadata(&cache.metadata()?) == Some(true),
            "completed token cache {} changed while it was replayed",
            path.display()
        );
        true
    } else {
        if torn_tail {
            cache.set_len(valid_bytes)?;
            cache.sync_all()?;
        }
        false
    };
    cache.seek(SeekFrom::End(0))?;
    let writer = BufWriter::new(cache);
    if verified_complete {
        // No source scan or index rewrite is necessary for an immutable cache
        // whose handle, record structure, counts, and digest all agree.
        return Ok((
            documents,
            true,
            Some(TokenCacheWriter {
                writer,
                location,
                digest,
                documents: u64::try_from(documents)
                    .context("token-cache document count exceeds u64")?,
                stream_tokens,
                index_invalidated: false,
            }),
            true,
        ));
    }
    Ok((
        documents,
        true,
        Some(TokenCacheWriter {
            writer,
            location,
            digest,
            documents: u64::try_from(documents)
                .context("token-cache document count exceeds u64")?,
            stream_tokens,
            index_invalidated: false,
        }),
        false,
    ))
}

struct ShuffleBuffer {
    samples: Vec<TrainingSample>,
    rng: StdRng,
    capacity: usize,
}

impl ShuffleBuffer {
    fn new(capacity: usize, seed: u64) -> Self {
        assert!(capacity > 0);
        Self {
            samples: Vec::with_capacity(capacity),
            rng: StdRng::seed_from_u64(seed),
            capacity,
        }
    }

    fn push(&mut self, sample: TrainingSample) -> Option<TrainingSample> {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            return None;
        }
        let index = self.rng.random_range(0..self.samples.len());
        Some(std::mem::replace(&mut self.samples[index], sample))
    }

    fn finish(mut self) -> Vec<TrainingSample> {
        self.samples.shuffle(&mut self.rng);
        self.samples
    }
}

struct SamplePacker {
    pending: Vec<i64>,
    consumed: usize,
    seq_len: usize,
}

impl SamplePacker {
    fn new(seq_len: usize) -> Self {
        Self {
            pending: Vec::new(),
            consumed: 0,
            seq_len,
        }
    }

    fn push(
        &mut self,
        tokens: impl IntoIterator<Item = i64>,
        count: &mut usize,
        visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
    ) -> Result<bool> {
        if self.consumed > 0 {
            self.pending.drain(..self.consumed);
            self.consumed = 0;
        }
        self.pending.extend(tokens);
        while self.pending.len() - self.consumed > self.seq_len {
            let end = self
                .consumed
                .checked_add(self.seq_len)
                .and_then(|end| end.checked_add(1))
                .context("packed sample boundary overflows usize")?;
            let tokens = self.pending[self.consumed..end].to_vec();
            self.consumed = self
                .consumed
                .checked_add(self.seq_len)
                .context("packed-token cursor overflows usize")?;
            *count = count
                .checked_add(1)
                .context("training sample count overflows usize")?;
            if !visit(TrainingSample::Causal { tokens })? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

struct DocumentTokenizationBatch {
    documents: Vec<String>,
    bytes: usize,
}

impl DocumentTokenizationBatch {
    fn new() -> Self {
        Self {
            documents: Vec::with_capacity(TOKENIZE_BATCH),
            bytes: 0,
        }
    }

    fn flush(
        &mut self,
        tokenizer: &Tokenizer,
        packer: &mut SamplePacker,
        count: &mut usize,
        visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
        cache: &mut Option<TokenCacheWriter>,
    ) -> Result<bool> {
        if self.documents.is_empty() {
            debug_assert_eq!(self.bytes, 0);
            return Ok(true);
        }
        let documents = std::mem::take(&mut self.documents);
        self.bytes = 0;
        let encodings = tokenizer.encode_batch(documents, false)?;
        for tokens in encodings {
            if let Some(cache) = cache {
                cache.append(&tokens)?;
            }
            let tokens = tokens
                .into_iter()
                .map(i64::from)
                .chain(std::iter::once(i64::from(tokenizer.eos_token_id())));
            if !packer.push(tokens, count, visit)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn queue(
        &mut self,
        document: String,
        tokenizer: &Tokenizer,
        packer: &mut SamplePacker,
        count: &mut usize,
        visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
        cache: &mut Option<TokenCacheWriter>,
    ) -> Result<bool> {
        ensure!(
            document.len() <= MAX_TOKENIZE_BATCH_BYTES,
            "training document is {} bytes, exceeding the tokenization batch limit of {MAX_TOKENIZE_BATCH_BYTES}",
            document.len()
        );
        let prospective_bytes = self
            .bytes
            .checked_add(document.len())
            .context("tokenization batch byte count overflows usize")?;
        if !self.documents.is_empty()
            && prospective_bytes > MAX_TOKENIZE_BATCH_BYTES
            && !self.flush(tokenizer, packer, count, visit, cache)?
        {
            return Ok(false);
        }
        self.bytes = self
            .bytes
            .checked_add(document.len())
            .context("tokenization batch byte count overflows usize")?;
        self.documents.push(document);
        if (self.documents.len() == TOKENIZE_BATCH || self.bytes >= MAX_TOKENIZE_BATCH_BYTES)
            && !self.flush(tokenizer, packer, count, visit, cache)?
        {
            return Ok(false);
        }
        Ok(true)
    }
}

fn is_jsonl(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"))
}

fn token_array(
    value: &serde_json::Value,
    path: &Path,
    line_number: usize,
) -> Result<Option<Vec<u32>>> {
    let Some(tokens) = value.get("tokens") else {
        return Ok(None);
    };
    let tokens = tokens.as_array().with_context(|| {
        format!(
            "`tokens` at {}:{line_number} must be an array",
            path.display()
        )
    })?;
    ensure!(
        !tokens.is_empty(),
        "`tokens` at {}:{line_number} is empty",
        path.display()
    );
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            let token = token.as_u64().with_context(|| {
                format!(
                    "`tokens[{index}]` at {}:{line_number} is not an unsigned integer",
                    path.display()
                )
            })?;
            u32::try_from(token).with_context(|| {
                format!(
                    "`tokens[{index}]` at {}:{line_number} exceeds u32",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

struct CausalVisitOutcome {
    count: usize,
    cache: Option<TokenCacheWriter>,
    publish_cache: bool,
}

impl CausalVisitOutcome {
    fn reset_cache(&mut self, token_cache: Option<&Path>) -> Result<()> {
        match &mut self.cache {
            Some(cache) => cache.reset_to_empty(),
            None => match token_cache {
                Some(path) => reset_token_cache_path(path),
                None => Ok(()),
            },
        }
    }
}

fn visit_causal_samples(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    token_cache: Option<&Path>,
    data_binding: &PhaseDataBinding,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    // Both checks are intentional. The first rejects a replacement already
    // present when an epoch begins. The second covers token-cache fast paths,
    // early visitor exits, and mutation during streaming. Reads in between use
    // only the descriptors captured during run-signature authentication.
    data_binding.ensure_matches_path(path)?;
    if let Err(identity_error) = data_binding.ensure_still_published() {
        if let Some(path) = token_cache {
            reset_token_cache_path(path)?;
        }
        return Err(identity_error);
    }
    let mut result = visit_causal_samples_inner(
        path,
        tokenizer,
        seq_len,
        token_cache,
        data_binding,
        &mut visit,
    );
    // This check runs before inspecting `result`, so mutation is reported even
    // if tokenization or the caller's visitor failed or stopped early.
    if let Err(identity_error) = data_binding.ensure_still_published() {
        match &mut result {
            Ok(outcome) => outcome.reset_cache(token_cache)?,
            Err(_) => {
                // The inner path resets whenever it owns a writer. This covers
                // failures during cache replay, before such a writer exists.
                if let Some(path) = token_cache {
                    reset_token_cache_path(path)?;
                }
            }
        }
        return Err(identity_error);
    }
    let mut outcome = result?;
    if outcome.publish_cache
        && let Some(cache) = &mut outcome.cache
    {
        cache.publish_index()?;
        // Close the small commit window as well. If publication changes while
        // the sidecar is being committed, revoke the completion proof before
        // returning the integrity error.
        if let Err(identity_error) = data_binding.ensure_still_published() {
            cache.reset_to_empty().with_context(|| {
                format!(
                    "failed to reset token cache after phase-data integrity failure: {identity_error:#}"
                )
            })?;
            return Err(identity_error);
        }
    }
    Ok(outcome.count)
}

fn visit_causal_samples_inner(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    token_cache: Option<&Path>,
    data_binding: &PhaseDataBinding,
    visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<CausalVisitOutcome> {
    let mut count = 0;
    let mut packer = SamplePacker::new(seq_len);
    let (cached_documents, keep_going, mut cache, cache_complete) = match token_cache {
        Some(path) => replay_token_cache(
            path,
            tokenizer.eos_token_id(),
            tokenizer.vocab_size(),
            &mut packer,
            &mut count,
            visit,
        )?,
        None => (0, true, None, false),
    };
    if cached_documents > 0
        && let Some(path) = token_cache
    {
        println!(
            "token_cache={} replayed_documents={cached_documents}",
            path.display()
        );
    }
    if !keep_going {
        return Ok(CausalVisitOutcome {
            count,
            cache: None,
            publish_cache: false,
        });
    }
    if cache_complete {
        return Ok(CausalVisitOutcome {
            count,
            cache: None,
            publish_cache: false,
        });
    }
    let mut document_number = 0usize;
    let mut documents = DocumentTokenizationBatch::new();
    let scan_result = (|| -> Result<bool> {
        let fully_read = data_binding.with_readers(path, |source_path, reader| {
            if is_jsonl(source_path) {
                let mut line = Vec::new();
                let mut line_number = 0usize;
                loop {
                    let next_line_number = line_number
                        .checked_add(1)
                        .context("JSONL line count overflows usize")?;
                    let Some(line) = read_training_jsonl_record(
                        reader,
                        &mut line,
                        source_path,
                        next_line_number,
                    )?
                    else {
                        break;
                    };
                    line_number = next_line_number;
                    if line.trim().is_empty() {
                        continue;
                    }
                    document_number = document_number
                        .checked_add(1)
                        .context("JSONL document count overflows usize")?;
                    if document_number <= cached_documents {
                        continue;
                    }
                    let value: serde_json::Value =
                        serde_json::from_str(line).with_context(|| {
                            format!("invalid JSONL at {}:{line_number}", source_path.display())
                        })?;
                    if let Some(tokens) = token_array(&value, source_path, line_number)? {
                        ensure!(
                            tokens
                                .iter()
                                .all(|token| (*token as usize) < tokenizer.vocab_size()),
                            "tokenized corpus row at {}:{line_number} contains a token outside vocabulary size {}",
                            source_path.display(),
                            tokenizer.vocab_size()
                        );
                        if !documents.flush(
                            tokenizer,
                            &mut packer,
                            &mut count,
                            visit,
                            &mut cache,
                        )? {
                            return Ok(false);
                        }
                        if let Some(cache) = &mut cache {
                            cache.append(&tokens)?;
                        }
                        if !packer.push(
                            tokens
                                .into_iter()
                                .map(i64::from)
                                .chain(std::iter::once(i64::from(tokenizer.eos_token_id()))),
                            &mut count,
                            visit,
                        )? {
                            return Ok(false);
                        }
                    } else {
                        let document = required_string(&value, "text", source_path, line_number)?;
                        if !documents.queue(
                            document.to_owned(),
                            tokenizer,
                            &mut packer,
                            &mut count,
                            visit,
                            &mut cache,
                        )? {
                            return Ok(false);
                        }
                    }
                }
            } else {
                document_number = document_number
                    .checked_add(1)
                    .context("document count overflows usize")?;
                if document_number > cached_documents {
                    let document = read_training_text_document_bounded(
                        reader,
                        MAX_TRAINING_RECORD_BYTES,
                        source_path,
                    )?;
                    if !documents.queue(
                        document,
                        tokenizer,
                        &mut packer,
                        &mut count,
                        visit,
                        &mut cache,
                    )? {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        })?;
        if !fully_read {
            return Ok(false);
        }
        if !documents.flush(tokenizer, &mut packer, &mut count, visit, &mut cache)? {
            return Ok(false);
        }
        ensure!(
            document_number >= cached_documents,
            "authoritative corpus has {document_number} documents but token cache has {cached_documents}; remove the stale cache"
        );
        Ok(true)
    })();
    let publish_cache = match scan_result {
        Ok(fully_read) => fully_read,
        Err(error) => {
            if let Some(cache) = &mut cache {
                cache.reset_to_empty()?;
            } else if let Some(path) = token_cache {
                reset_token_cache_path(path)?;
            }
            return Err(error);
        }
    };
    Ok(CausalVisitOutcome {
        count,
        cache,
        publish_cache,
    })
}

fn required_string<'a>(
    value: &'a serde_json::Value,
    field: &str,
    path: &Path,
    line_number: usize,
) -> Result<&'a str> {
    let text = value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .with_context(|| {
            format!(
                "JSONL row at {}:{line_number} must contain a string `{field}` field",
                path.display()
            )
        })?;
    ensure!(
        !text.trim().is_empty(),
        "JSONL row at {}:{line_number} has an empty `{field}` field",
        path.display()
    );
    Ok(text)
}

/// Visit fixed-shape objective samples in source order.
fn visit_samples_in_order(
    path: &Path,
    objective: &TaskConfig,
    tokenizer: &Tokenizer,
    config: SampleStreamConfig<'_>,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    ensure!(config.seq_len > 0, "sequence_length must be positive");
    match objective {
        TaskConfig::CausalLm {} => visit_causal_samples(
            path,
            tokenizer,
            config.seq_len,
            config.token_cache,
            config.data_binding,
            visit,
        ),
        _ => {
            ensure!(
                config.data_binding.authenticated_corpus().is_none(),
                "prepared corpus manifests are only valid for causal-LM phases"
            );
            let mut count = None;
            config
                .data_binding
                .with_readers(path, |source_path, reader| {
                    count = Some(visit_structured_samples(
                        source_path,
                        reader,
                        objective,
                        tokenizer,
                        config.seq_len,
                        config.oversized,
                        &mut visit,
                    )?);
                    Ok(true)
                })?;
            count.context("direct structured phase data produced no reader")
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SampleStreamConfig<'a> {
    pub(crate) seq_len: usize,
    pub(crate) shuffle_buffer: usize,
    pub(crate) seed: u64,
    pub(crate) token_cache: Option<&'a Path>,
    pub(crate) data_binding: &'a PhaseDataBinding,
    /// Disposition of a supervised record that cannot be framed at `seq_len`.
    /// Training aborts; forward-only evaluation skips and counts.
    pub(crate) oversized: OversizedRecordPolicy<'a>,
}

pub(crate) fn visit_samples(
    path: &Path,
    objective: &TaskConfig,
    tokenizer: &Tokenizer,
    config: SampleStreamConfig<'_>,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    if config.shuffle_buffer == 0 {
        return visit_samples_in_order(path, objective, tokenizer, config, visit);
    }

    let mut shuffler = ShuffleBuffer::new(config.shuffle_buffer, config.seed);
    let mut keep_going = true;
    let count = visit_samples_in_order(path, objective, tokenizer, config, |sample| {
        if let Some(sample) = shuffler.push(sample) {
            keep_going = visit(sample)?;
        }
        Ok(keep_going)
    })?;

    if keep_going {
        for sample in shuffler.finish() {
            if !visit(sample)? {
                break;
            }
        }
    }
    Ok(count)
}

pub(crate) fn count_samples(
    path: &Path,
    objective: &TaskConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    token_cache: Option<&Path>,
    data_binding: &PhaseDataBinding,
) -> Result<usize> {
    visit_samples_in_order(
        path,
        objective,
        tokenizer,
        SampleStreamConfig {
            seq_len,
            shuffle_buffer: 0,
            seed: 0,
            token_cache,
            data_binding,
            // Counting is what a training run will stream, so a record the
            // trainer cannot frame must fail here too.
            oversized: OversizedRecordPolicy::Abort,
        },
        |_| Ok(true),
    )
}

/// Smallest self-contained byte-level BPE accepted by the production
/// tokenizer: IDs 0..=255 are the raw byte alphabet and 256 is EOS. Shared by
/// the data-pipeline and forward-only evaluation tests.
#[cfg(test)]
pub(crate) fn write_test_tokenizer(directory: &Path) -> Tokenizer {
    let allowed: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut byte_to_unicode = ['\0'; 256];
    for &byte in &allowed {
        byte_to_unicode[byte as usize] = byte as char;
    }
    let mut offset = 0_u32;
    for byte in 0..=255_u8 {
        if byte_to_unicode[byte as usize] == '\0' {
            byte_to_unicode[byte as usize] = char::from_u32(256 + offset).unwrap();
            offset += 1;
        }
    }
    let mut vocabulary = serde_json::Map::new();
    for (id, piece) in byte_to_unicode.into_iter().enumerate() {
        vocabulary.insert(piece.to_string(), serde_json::json!(id));
    }
    vocabulary.insert("<eos>".to_owned(), serde_json::json!(256));
    let tokenizer = serde_json::json!({
        "model": {
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [],
        },
        "added_tokens": [{
            "id": 256,
            "content": "<eos>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true,
        }],
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "use_regex": true,
        },
        "decoder": { "type": "ByteLevel" },
    });
    let path = directory.join("tokenizer.json");
    fs::write(&path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    Tokenizer::from_file(path).unwrap()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::io::Cursor;

    use burn::tensor::Device;
    use hermes_train::corpus::{
        CORPUS_SCHEMA_VERSION, ClassificationConfig, CorpusBuildConfig, CorpusPipeline,
        CorpusTokenizer, DeduplicationConfig, DiscoveryConfig, DiscoveryHit, DiscoveryPage,
        DiscoveryQuery, InMemoryDeduplicator, InlineRecordMaterializer, NormalizationConfig,
        RepetitionConfig, SearchBackend, ShardingConfig, SourceSnapshot, TokenTarget,
        TokenizerSnapshot,
    };
    use hermes_train::task::{TaskAdapter, TaskExample};
    use serde_json::Value;

    use super::*;

    #[test]
    fn bounded_training_jsonl_reader_accepts_exact_limit_and_stops_at_delimiter() {
        let mut input = BufReader::new(Cursor::new(b"12345678\n{}\n"));
        let mut record = Vec::new();
        assert_eq!(
            read_training_jsonl_record_bounded(
                &mut input,
                &mut record,
                8,
                Path::new("training.jsonl"),
                1,
            )
            .unwrap(),
            Some("12345678\n")
        );
        assert_eq!(
            read_training_jsonl_record_bounded(
                &mut input,
                &mut record,
                8,
                Path::new("training.jsonl"),
                2,
            )
            .unwrap(),
            Some("{}\n")
        );
    }

    #[test]
    fn bounded_training_jsonl_reader_captures_only_one_byte_past_limit() {
        let mut input = BufReader::new(Cursor::new(b"123456789trailing"));
        let mut record = Vec::new();
        let error = read_training_jsonl_record_bounded(
            &mut input,
            &mut record,
            8,
            Path::new("oversized.jsonl"),
            7,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("oversized.jsonl:7"), "{error}");
        assert!(error.contains("maximum of 8 bytes"), "{error}");
        assert_eq!(record.len(), 9);
    }

    #[test]
    fn bounded_training_jsonl_reader_rejects_malformed_utf8() {
        let mut input = BufReader::new(Cursor::new(b"{\"text\":\"\xff\"}\n"));
        let mut record = Vec::new();
        let error = read_training_jsonl_record_bounded(
            &mut input,
            &mut record,
            32,
            Path::new("invalid.jsonl"),
            3,
        )
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("invalid.jsonl:3"), "{error}");
        assert!(error.contains("not UTF-8"), "{error}");
    }

    #[test]
    fn bounded_training_text_reader_preserves_one_multiline_document_at_the_limit() {
        let mut input = BufReader::new(Cursor::new(b"one\ntwo\n"));
        assert_eq!(
            read_training_text_document_bounded(&mut input, 8, Path::new("document.txt")).unwrap(),
            "one\ntwo\n"
        );
    }

    #[test]
    fn bounded_training_text_reader_stops_one_byte_past_the_limit() {
        let mut input = BufReader::new(Cursor::new(b"123456789trailing"));
        let error = read_training_text_document_bounded(&mut input, 8, Path::new("oversized.txt"))
            .unwrap_err()
            .to_string();

        assert!(error.contains("oversized.txt"), "{error}");
        assert!(error.contains("maximum of 8 bytes"), "{error}");
    }

    #[test]
    fn bounded_training_text_reader_rejects_malformed_utf8() {
        let mut input = BufReader::new(Cursor::new(b"text-\xff"));
        let error = read_training_text_document_bounded(&mut input, 8, Path::new("invalid.txt"))
            .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("invalid.txt"), "{error}");
        assert!(error.contains("not valid UTF-8"), "{error}");
    }

    #[test]
    fn malformed_jsonl_utf8_resets_a_partial_token_cache() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("data.jsonl");
        let cache_path = dir.path().join("tokens.bin");
        let tokenizer = write_test_tokenizer(dir.path());
        fs::write(&data_path, b"{\"tokens\":[1,2]}\n{\"text\":\"\xff\"}\n").unwrap();
        let binding = PhaseDataBinding::open(&data_path).unwrap();

        let error = visit_causal_samples(
            &data_path,
            &tokenizer,
            2,
            Some(&cache_path),
            &binding,
            |_| Ok(true),
        )
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("data.jsonl:2"), "{error}");
        assert!(error.contains("not UTF-8"), "{error}");
        assert_eq!(fs::read(&cache_path).unwrap(), TOKEN_CACHE_MAGIC);
        assert!(!token_cache_index_path(&cache_path).exists());
    }

    #[test]
    fn structured_training_uses_the_bounded_utf8_reader() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = write_test_tokenizer(dir.path());
        let task = TaskConfig::Summarization {
            instruction: "Summarize.".into(),
        };
        let mut reader = BufReader::new(Cursor::new(b"{\"document\":\"\xff\"}\n"));

        let error = visit_structured_samples(
            Path::new("structured.jsonl"),
            &mut reader,
            &task,
            &tokenizer,
            64,
            OversizedRecordPolicy::Abort,
            |_| Ok(true),
        )
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("structured.jsonl:1"), "{error}");
        assert!(error.contains("not UTF-8"), "{error}");
    }

    #[test]
    fn bounded_raw_text_path_remains_one_multiline_causal_document() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = write_test_tokenizer(dir.path());
        let text = "one two three\nfour five six\n";
        let text_path = dir.path().join("document.txt");
        let jsonl_path = dir.path().join("document.jsonl");
        fs::write(&text_path, text).unwrap();
        fs::write(
            &jsonl_path,
            format!("{}\n", serde_json::json!({"text": text})),
        )
        .unwrap();

        let collect = |path: &Path| {
            let binding = PhaseDataBinding::open(path).unwrap();
            let mut samples = Vec::new();
            visit_causal_samples(path, &tokenizer, 4, None, &binding, |sample| {
                let TrainingSample::Causal { tokens } = sample else {
                    panic!("expected causal sample")
                };
                samples.push(tokens);
                Ok(true)
            })
            .unwrap();
            samples
        };

        let raw_samples = collect(&text_path);
        assert!(!raw_samples.is_empty());
        assert_eq!(raw_samples, collect(&jsonl_path));
    }

    struct OneRecordSearch;

    impl SearchBackend for OneRecordSearch {
        fn name(&self) -> &str {
            "one_record"
        }

        fn configuration(&self) -> Result<Value> {
            Ok(serde_json::json!({"type": "one_record"}))
        }

        fn snapshot(&self) -> Result<SourceSnapshot> {
            Ok(SourceSnapshot {
                provider: "test".into(),
                revision: "one".into(),
            })
        }

        fn page_size(&self) -> usize {
            1
        }

        fn discover(
            &self,
            _query: &DiscoveryQuery,
            offset: usize,
            _limit: usize,
        ) -> Result<DiscoveryPage> {
            Ok(DiscoveryPage {
                hits: if offset == 0 {
                    vec![DiscoveryHit {
                        record_key: "record".into(),
                        score: 1.0,
                        uris: Vec::new(),
                        metadata: BTreeMap::new(),
                        inline_text: Some("source text".into()),
                    }]
                } else {
                    Vec::new()
                },
                total_hits: Some(1),
                snapshot: self.snapshot()?,
            })
        }
    }

    struct TwoTokenCorpusTokenizer;

    impl CorpusTokenizer for TwoTokenCorpusTokenizer {
        fn snapshot(&self) -> TokenizerSnapshot {
            TokenizerSnapshot {
                implementation: "test".into(),
                revision: "one".into(),
                vocabulary_size: 16,
            }
        }

        fn encode(&self, _text: &str) -> Result<Vec<u32>> {
            Ok(vec![1, 2])
        }
    }

    fn write_authenticated_test_corpus(root: &Path) -> PathBuf {
        let config = CorpusBuildConfig {
            version: CORPUS_SCHEMA_VERSION,
            build_id: "bound-corpus".into(),
            discovery: DiscoveryConfig {
                queries: vec![DiscoveryQuery {
                    name: "all".into(),
                    text: "all".into(),
                    limit: 1,
                    parameters: BTreeMap::new(),
                }],
                materialization_batch_size: 1,
            },
            normalization: NormalizationConfig::default(),
            deduplication: DeduplicationConfig::default(),
            classification: ClassificationConfig::default(),
            transformations: Vec::new(),
            repetition: RepetitionConfig::default(),
            token_target: TokenTarget {
                minimum: 2,
                desired: 2,
                maximum: 2,
            },
            sharding: ShardingConfig {
                max_tokens_per_shard: 2,
            },
        };
        let mut deduplicator = InMemoryDeduplicator::new(DeduplicationConfig::default()).unwrap();
        CorpusPipeline::new(
            config,
            &OneRecordSearch,
            &InlineRecordMaterializer,
            &TwoTokenCorpusTokenizer,
            &mut deduplicator,
        )
        .unwrap()
        .run(root)
        .unwrap()
        .0
    }

    #[test]
    fn zstd_data_reader_streams_decompressed_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl.zst");
        let source = b"{\"text\":\"one\"}\n{\"text\":\"two\"}\n";
        let compressed = zstd::stream::encode_all(Cursor::new(source), 1).unwrap();
        fs::write(&path, compressed).unwrap();

        let mut decoded = String::new();
        let binding = PhaseDataBinding::open(&path).unwrap();
        binding
            .with_readers(&path, |_path, reader| {
                reader.read_to_string(&mut decoded)?;
                Ok(true)
            })
            .unwrap();
        assert_eq!(decoded.as_bytes(), source);
    }

    #[test]
    fn training_record_limit_applies_after_zstd_decompression() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl.zst");
        fs::write(
            &path,
            zstd::stream::encode_all(Cursor::new(b"123456789decompressed"), 1).unwrap(),
        )
        .unwrap();
        let binding = PhaseDataBinding::open(&path).unwrap();

        let error = binding
            .with_readers(&path, |source_path, reader| {
                let mut record = Vec::new();
                read_training_jsonl_record_bounded(reader, &mut record, 8, source_path, 1)?;
                Ok(true)
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("maximum of 8 bytes"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn direct_binding_reads_the_opened_generation_and_rejects_path_swap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        let parked = dir.path().join("opened.jsonl");
        let generation_a = b"{\"text\":\"generation-a\"}\n";
        fs::write(&path, generation_a).unwrap();
        let binding = PhaseDataBinding::open(&path).unwrap();

        let mut observed = Vec::new();
        let error = binding
            .with_readers(&path, |_source, reader| {
                reader.read_to_end(&mut observed)?;
                fs::rename(&path, &parked)?;
                fs::write(&path, b"{\"text\":\"generation-b\"}\n")?;
                Ok(true)
            })
            .unwrap_err()
            .to_string();

        assert_eq!(observed, generation_a);
        assert!(error.contains("phase data"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn direct_zstd_binding_pins_compressed_generation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl.zst");
        let parked = dir.path().join("opened.jsonl.zst");
        let generation_a = b"{\"text\":\"compressed-a\"}\n";
        fs::write(
            &path,
            zstd::stream::encode_all(Cursor::new(generation_a), 1).unwrap(),
        )
        .unwrap();
        let binding = PhaseDataBinding::open(&path).unwrap();

        let mut observed = Vec::new();
        assert!(
            binding
                .with_readers(&path, |_source, reader| {
                    reader.read_to_end(&mut observed)?;
                    fs::rename(&path, &parked)?;
                    fs::write(
                        &path,
                        zstd::stream::encode_all(Cursor::new(b"{\"text\":\"compressed-b\"}\n"), 1)?,
                    )?;
                    Ok(true)
                })
                .is_err()
        );
        assert_eq!(observed, generation_a);
    }

    #[cfg(unix)]
    #[test]
    fn direct_binding_rejects_symlink_at_bind_and_after_bind() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.jsonl");
        let alias = dir.path().join("alias.jsonl");
        fs::write(&target, b"{\"text\":\"target\"}\n").unwrap();
        std::os::unix::fs::symlink(&target, &alias).unwrap();
        assert!(PhaseDataBinding::open(&alias).is_err());

        let path = dir.path().join("data.jsonl");
        let parked = dir.path().join("parked.jsonl");
        fs::write(&path, b"{\"text\":\"opened\"}\n").unwrap();
        let binding = PhaseDataBinding::open(&path).unwrap();
        fs::rename(&path, &parked).unwrap();
        std::os::unix::fs::symlink(&parked, &path).unwrap();
        assert!(binding.ensure_still_published().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn direct_binding_checks_identity_after_error_and_early_exit() {
        for visitor_errors in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("data.jsonl");
            fs::write(&path, b"{\"text\":\"opened\"}\n").unwrap();
            let binding = PhaseDataBinding::open(&path).unwrap();

            let result = binding.with_readers(&path, |_source, _reader| {
                fs::write(&path, b"{\"text\":\"changed-in-place\"}\n")?;
                if visitor_errors {
                    anyhow::bail!("visitor failed")
                }
                Ok(false)
            });
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("changed in place"), "{error}");
            assert!(!error.contains("visitor failed"), "{error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn corpus_binding_checks_manifest_identity_after_visitor_error() {
        let root = tempfile::tempdir().unwrap();
        let corpus_path = write_authenticated_test_corpus(root.path());
        let manifest_path = corpus_path.join("manifest.json");
        let replacement = corpus_path.join("replacement-manifest.json");
        let binding = PhaseDataBinding::open(&corpus_path).unwrap();

        let error = binding
            .with_readers(&corpus_path, |_source, _reader| {
                fs::write(&replacement, fs::read(&manifest_path)?)?;
                fs::rename(&replacement, &manifest_path)?;
                anyhow::bail!("visitor failed")
            })
            .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("corpus manifest"), "{error}");
        assert!(!error.contains("visitor failed"), "{error}");
    }

    #[test]
    fn streaming_shuffle_is_bounded_and_deterministic() {
        let shuffle = |seed| {
            let mut buffer = ShuffleBuffer::new(4, seed);
            let mut output = Vec::new();
            for value in 0..32_i64 {
                if let Some(TrainingSample::Causal { tokens }) =
                    buffer.push(TrainingSample::Causal {
                        tokens: vec![value],
                    })
                {
                    output.push(tokens[0]);
                }
                assert!(buffer.samples.len() <= 4);
            }
            output.extend(buffer.finish().into_iter().map(|sample| match sample {
                TrainingSample::Causal { tokens } => tokens[0],
                _ => unreachable!(),
            }));
            output
        };

        let first = shuffle(7);
        assert_eq!(first, shuffle(7));
        assert_ne!(first, (0..32_i64).collect::<Vec<_>>());
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32_i64).collect::<Vec<_>>());
    }

    #[test]
    fn sample_packer_joins_documents_without_dropping_tokens() {
        let mut packer = SamplePacker::new(3);
        let mut samples = Vec::new();
        let mut count = 0;
        let mut collect = |sample| {
            let TrainingSample::Causal { tokens } = sample else {
                unreachable!()
            };
            samples.push(tokens);
            Ok(true)
        };

        for document in [vec![1, 2, 0], vec![3, 4, 0], vec![5, 6, 0]] {
            assert!(packer.push(document, &mut count, &mut collect).unwrap());
        }

        assert_eq!(count, 2);
        assert_eq!(samples, [vec![1, 2, 0, 3], vec![3, 4, 0, 5]]);
    }

    #[test]
    fn token_cache_replays_and_repairs_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        let mut bytes = TOKEN_CACHE_MAGIC.to_vec();
        for document in [&[1_u32, 2][..], &[3, 4][..]] {
            bytes.extend_from_slice(&(document.len() as u32).to_le_bytes());
            for token in document {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        let valid_len = bytes.len();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&99_u32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let mut packer = SamplePacker::new(3);
        let mut samples = Vec::new();
        let mut count = 0;
        let (documents, keep_going, mut writer, _) =
            replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut |sample| {
                samples.push(sample);
                Ok(true)
            })
            .unwrap();
        assert_eq!(documents, 2);
        assert!(keep_going);
        assert_eq!(fs::metadata(&path).unwrap().len(), valid_len as u64);
        assert_eq!(count, 1);
        let TrainingSample::Causal { tokens } = &samples[0] else {
            unreachable!()
        };
        assert_eq!(tokens, &[1, 2, 0, 3]);

        writer.as_mut().unwrap().append(&[5, 6]).unwrap();
        writer.as_mut().unwrap().flush().unwrap();
        drop(writer);
        let mut replay = SamplePacker::new(3);
        let mut replay_count = 0;
        let (documents, _, _, _) =
            replay_token_cache(&path, 0, 257, &mut replay, &mut replay_count, &mut |_| {
                Ok(true)
            })
            .unwrap();
        assert_eq!(documents, 3);
        assert_eq!(replay_count, 2);
    }

    #[test]
    fn invalid_cache_header_is_rebuilt_and_pretokenized_rows_are_cached_once() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("data.jsonl");
        let cache_path = dir.path().join("tokens.bin");
        let tokenizer = write_test_tokenizer(dir.path());
        fs::write(&data_path, b"{\"tokens\":[1,2]}\n{\"tokens\":[3,4]}\n").unwrap();
        fs::write(&cache_path, b"BADTOK01untrusted-derived-bytes").unwrap();
        let binding = PhaseDataBinding::open(&data_path).unwrap();

        let collect = || {
            let mut samples = Vec::new();
            let count = visit_causal_samples(
                &data_path,
                &tokenizer,
                2,
                Some(&cache_path),
                &binding,
                |sample| {
                    let TrainingSample::Causal { tokens } = sample else {
                        panic!("expected causal sample")
                    };
                    samples.push(tokens);
                    Ok(true)
                },
            )
            .unwrap();
            (count, samples)
        };

        let first = collect();
        assert_eq!(first.0, 2);
        assert_eq!(
            indexed_causal_sample_count(&cache_path, 2).unwrap(),
            Some(2)
        );
        assert_eq!(collect(), first, "cache replay changed the training stream");
    }

    fn write_test_token_cache(path: &Path, documents: &[&[u32]], complete: bool) {
        let mut packer = SamplePacker::new(4);
        let mut count = 0;
        let (_, keep_going, mut writer, _) =
            replay_token_cache(path, 0, 257, &mut packer, &mut count, &mut |_| Ok(true)).unwrap();
        assert!(keep_going);
        let writer = writer.as_mut().unwrap();
        for document in documents {
            writer.append(document).unwrap();
        }
        if complete {
            writer.publish_index().unwrap();
        } else {
            writer.flush().unwrap();
        }
    }

    fn update_causal_digest(
        digest: &mut Sha256,
        samples: &mut usize,
        sample: TrainingSample,
    ) -> Result<bool> {
        let TrainingSample::Causal { tokens } = sample else {
            anyhow::bail!("expected a causal sample")
        };
        *samples += 1;
        digest.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            digest.update(token.to_le_bytes());
        }
        Ok(true)
    }

    #[cfg(unix)]
    #[test]
    fn identity_failure_resets_partial_cache_before_restored_generation_retry() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("data.jsonl");
        let cache_path = dir.path().join("tokens.bin");
        let tokenizer = write_test_tokenizer(dir.path());
        let line = "{\"tokens\":[1,2,3,4,5,6,7,8]}\n";
        let original = line.repeat(1_000).into_bytes();
        assert!(original.len() > AUTHENTICATED_READ_BUFFER_BYTES * 2);
        fs::write(&data_path, &original).unwrap();
        let binding = PhaseDataBinding::open(&data_path).unwrap();

        let mut mutated = false;
        let result = visit_causal_samples(
            &data_path,
            &tokenizer,
            128,
            Some(&cache_path),
            &binding,
            |_| {
                if !mutated {
                    let mut file = OpenOptions::new().write(true).open(&data_path)?;
                    file.seek(SeekFrom::Start(
                        (AUTHENTICATED_READ_BUFFER_BYTES * 2) as u64,
                    ))?;
                    file.write_all(b"!")?;
                    file.sync_all()?;
                    mutated = true;
                }
                Ok(true)
            },
        );
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("changed in place"), "{error}");
        assert_eq!(fs::read(&cache_path).unwrap(), TOKEN_CACHE_MAGIC);
        assert!(!token_cache_index_path(&cache_path).exists());

        fs::write(&data_path, &original).unwrap();
        let restored = PhaseDataBinding::open(&data_path).unwrap();
        let mut retry_digest = Sha256::new();
        let mut retry_samples = 0;
        let retry_count = visit_causal_samples(
            &data_path,
            &tokenizer,
            128,
            Some(&cache_path),
            &restored,
            |sample| update_causal_digest(&mut retry_digest, &mut retry_samples, sample),
        )
        .unwrap();
        assert!(token_cache_index_path(&cache_path).is_file());

        let mut reference_digest = Sha256::new();
        let mut reference_samples = 0;
        let reference_count =
            visit_causal_samples(&data_path, &tokenizer, 128, None, &restored, |sample| {
                update_causal_digest(&mut reference_digest, &mut reference_samples, sample)
            })
            .unwrap();
        assert_eq!(retry_count, reference_count);
        assert_eq!(retry_samples, reference_samples);
        assert_eq!(retry_digest.finalize(), reference_digest.finalize());
    }

    #[cfg(unix)]
    #[test]
    fn completed_cache_fast_path_still_rejects_direct_data_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("data.jsonl");
        let parked = dir.path().join("opened.jsonl");
        let cache_path = dir.path().join("tokens.bin");
        let tokenizer = write_test_tokenizer(dir.path());
        fs::write(&data_path, b"{\"tokens\":[1,2]}\n").unwrap();
        let binding = PhaseDataBinding::open(&data_path).unwrap();
        write_test_token_cache(&cache_path, &[&[1, 2]], true);

        fs::rename(&data_path, &parked).unwrap();
        fs::write(&data_path, b"{\"tokens\":[3,4]}\n").unwrap();
        assert!(
            visit_causal_samples(
                &data_path,
                &tokenizer,
                1,
                Some(&cache_path),
                &binding,
                |_| Ok(true),
            )
            .is_err()
        );
        assert_eq!(fs::read(&cache_path).unwrap(), TOKEN_CACHE_MAGIC);
        assert!(!token_cache_index_path(&cache_path).exists());
    }

    #[test]
    fn completed_token_cache_index_counts_packed_samples_in_constant_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[1, 2], &[], &[3, 4, 5]], true);

        // Eight joined stream tokens: two/zero/three document tokens plus an
        // EOS token for each document. Packing advances by `seq_len` while
        // retaining one next-token label.
        for (seq_len, expected) in [(1, 7), (2, 3), (3, 2), (4, 1), (7, 1), (8, 0), (64, 0)] {
            assert_eq!(
                indexed_causal_sample_count(&path, seq_len).unwrap(),
                Some(expected),
                "sequence length {seq_len}"
            );
        }

        let mut packer = SamplePacker::new(4);
        let mut replayed = 0;
        let (_, _, _, complete) =
            replay_token_cache(&path, 0, 257, &mut packer, &mut replayed, &mut |_| Ok(true))
                .unwrap();
        assert!(complete, "verified caches must not require a source rescan");
        assert_eq!(replayed, 1);
    }

    #[cfg(not(unix))]
    #[test]
    fn portable_metadata_never_fast_trusts_a_completed_cache_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[1, 2], &[3, 4]], true);
        let cache_bytes = fs::read(&path).unwrap();

        // Portable std::fs::Metadata has no stable file-incarnation identity.
        // The constant-time path must therefore fail closed even for a sidecar
        // written by this process.
        assert_eq!(indexed_causal_sample_count(&path, 3).unwrap(), None);

        let mut packer = SamplePacker::new(3);
        let mut count = 0;
        let (documents, keep_going, _, complete) =
            replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut |_| Ok(true)).unwrap();
        assert_eq!(documents, 2);
        assert_eq!(count, 1);
        assert!(keep_going);
        assert!(!complete, "portable metadata is not completion proof");
        assert_eq!(fs::read(&path).unwrap(), cache_bytes);
        assert!(!token_cache_index_path(&path).exists());
    }

    #[test]
    fn torn_or_corrupt_token_cache_index_falls_back_to_scan() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[1, 2], &[3, 4]], true);
        let index_path = token_cache_index_path(&path);
        let valid_index = fs::read(&index_path).unwrap();

        fs::write(&index_path, &valid_index[..valid_index.len() / 2]).unwrap();
        assert_eq!(indexed_causal_sample_count(&path, 3).unwrap(), None);

        let mut corrupt_index = valid_index.clone();
        corrupt_index[80] ^= 0x80;
        fs::write(&index_path, corrupt_index).unwrap();
        assert_eq!(indexed_causal_sample_count(&path, 3).unwrap(), None);
    }

    #[test]
    fn torn_index_reindexes_intact_cache_without_retokenizing_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[1, 2], &[3, 4]], true);
        let cache_bytes = fs::read(&path).unwrap();
        let index_path = token_cache_index_path(&path);
        let index_bytes = fs::read(&index_path).unwrap();
        fs::write(&index_path, &index_bytes[..17]).unwrap();

        let mut packer = SamplePacker::new(3);
        let mut count = 0;
        let mut samples = Vec::new();
        let (documents, keep_going, mut writer, complete) =
            replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut |sample| {
                samples.push(sample);
                Ok(true)
            })
            .unwrap();
        assert_eq!(documents, 2);
        assert_eq!(count, 1);
        assert_eq!(samples.len(), 1);
        assert!(keep_going);
        assert!(!complete, "invalid metadata is not a completion proof");
        assert_eq!(fs::read(&path).unwrap(), cache_bytes);
        assert!(!index_path.exists());

        // Production publishes here only after the authoritative source scan
        // confirms document coverage. Exercise the same post-scan operation.
        writer.as_mut().unwrap().publish_index().unwrap();
        drop(writer);
        assert_eq!(fs::read(&path).unwrap(), cache_bytes);
        assert_eq!(indexed_causal_sample_count(&path, 3).unwrap(), Some(1));
    }

    #[test]
    fn incomplete_or_changed_token_cache_never_uses_stale_index() {
        let dir = tempfile::tempdir().unwrap();
        let incomplete = dir.path().join("incomplete.bin");
        write_test_token_cache(&incomplete, &[&[1, 2]], false);
        assert!(!token_cache_index_path(&incomplete).exists());
        assert_eq!(indexed_causal_sample_count(&incomplete, 2).unwrap(), None);

        let changed = dir.path().join("changed.bin");
        write_test_token_cache(&changed, &[&[1, 2]], true);
        assert_eq!(indexed_causal_sample_count(&changed, 2).unwrap(), Some(1));
        let mut packer = SamplePacker::new(2);
        let mut count = 0;
        let (_, _, mut writer, _) =
            replay_token_cache(&changed, 0, 257, &mut packer, &mut count, &mut |_| Ok(true))
                .unwrap();
        writer.as_mut().unwrap().append(&[]).unwrap();
        assert!(!token_cache_index_path(&changed).exists());
        assert_eq!(indexed_causal_sample_count(&changed, 2).unwrap(), None);
    }

    #[test]
    fn out_of_vocabulary_cached_tokens_are_never_emitted_and_reset_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[300, 1, 2]], true);
        assert_eq!(indexed_causal_sample_count(&path, 2).unwrap(), Some(1));

        let mut packer = SamplePacker::new(2);
        let mut count = 0;
        let mut visited = 0;
        let error = replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut |_| {
            visited += 1;
            Ok(true)
        })
        .err()
        .expect("out-of-vocabulary cache must fail")
        .to_string();

        assert!(error.contains("outside tokenizer vocabulary"), "{error}");
        assert_eq!(visited, 0, "an invalid cached token reached the visitor");
        assert_eq!(count, 0);
        assert_eq!(fs::read(&path).unwrap(), TOKEN_CACHE_MAGIC);
        assert!(!token_cache_index_path(&path).exists());
    }

    #[cfg(unix)]
    #[test]
    fn completed_cache_identity_is_checked_when_the_visitor_stops_early() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[1, 2, 3], &[4, 5, 6]], true);

        let mut packer = SamplePacker::new(2);
        let mut count = 0;
        let error = replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut |_| {
            let mut cache = OpenOptions::new().write(true).open(&path)?;
            cache.seek(SeekFrom::Start(TOKEN_CACHE_MAGIC.len() as u64 + 4))?;
            cache.write_all(&9_u32.to_le_bytes())?;
            cache.sync_all()?;
            Ok(false)
        })
        .err()
        .expect("mutated cache must fail")
        .to_string();

        assert!(error.contains("changed while it was replayed"), "{error}");
        assert_eq!(fs::read(&path).unwrap(), TOKEN_CACHE_MAGIC);
        assert!(!token_cache_index_path(&path).exists());
    }

    #[test]
    fn changed_completed_cache_is_discarded_instead_of_reblessed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[1, 2], &[3, 4]], true);

        let mut cache = OpenOptions::new().write(true).open(&path).unwrap();
        cache.seek(SeekFrom::Start(12)).unwrap();
        cache.write_all(&99_u32.to_le_bytes()).unwrap();
        cache.sync_all().unwrap();
        assert_eq!(indexed_causal_sample_count(&path, 3).unwrap(), None);

        let mut packer = SamplePacker::new(3);
        let mut count = 0;
        let (documents, _, _, complete) =
            replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut |_| Ok(true)).unwrap();
        assert_eq!(documents, 0);
        assert_eq!(count, 0);
        assert!(!complete);
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            TOKEN_CACHE_MAGIC.len() as u64
        );
        assert!(!token_cache_index_path(&path).exists());
    }

    #[test]
    fn completed_cache_digest_is_checked_during_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        write_test_token_cache(&path, &[&[1, 2], &[3, 4]], true);

        let mut cache = OpenOptions::new().write(true).open(&path).unwrap();
        cache.seek(SeekFrom::Start(12)).unwrap();
        cache.write_all(&99_u32.to_le_bytes()).unwrap();
        cache.sync_all().unwrap();
        let location = TokenCacheLocation::open(&path, false).unwrap().unwrap();
        let opened_cache = location.open_cache(false).unwrap().unwrap();
        let TokenCacheIndexFile::Decoded(mut index) =
            decode_token_cache_index_file(&location).unwrap()
        else {
            panic!("expected decoded index")
        };
        // Simulate structurally valid metadata for the new file identity while
        // retaining the original payload digest.
        index.identity = TokenCacheFileIdentity::from_metadata(&opened_cache.metadata().unwrap());
        write_token_cache_index_atomic(&location, &index.encode()).unwrap();
        drop(opened_cache);
        drop(location);

        let mut packer = SamplePacker::new(3);
        let mut count = 0;
        let error = replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut |_| Ok(true))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("payload digest changed"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn token_cache_rejects_symlink_entries_without_touching_targets() {
        let dir = tempfile::tempdir().unwrap();
        let cache_root = dir.path().join("cache");
        fs::create_dir(&cache_root).unwrap();
        let victim = dir.path().join("victim");
        fs::write(&victim, b"do not touch").unwrap();
        let cache_path = cache_root.join("tokens.bin");
        std::os::unix::fs::symlink(&victim, &cache_path).unwrap();

        let mut packer = SamplePacker::new(3);
        let mut count = 0;
        assert!(
            replay_token_cache(&cache_path, 0, 257, &mut packer, &mut count, &mut |_| Ok(
                true
            ),)
            .is_err()
        );
        assert_eq!(fs::read(&victim).unwrap(), b"do not touch");

        fs::remove_file(&cache_path).unwrap();
        write_test_token_cache(&cache_path, &[&[1, 2]], false);
        let index_victim = dir.path().join("index-victim");
        fs::write(&index_victim, b"also do not touch").unwrap();
        std::os::unix::fs::symlink(&index_victim, token_cache_index_path(&cache_path)).unwrap();
        assert!(indexed_causal_sample_count(&cache_path, 3).is_err());
        assert_eq!(fs::read(&index_victim).unwrap(), b"also do not touch");

        let linked_root = dir.path().join("linked-cache-root");
        std::os::unix::fs::symlink(&cache_root, &linked_root).unwrap();
        let mut linked_packer = SamplePacker::new(3);
        let mut linked_count = 0;
        assert!(
            replay_token_cache(
                &linked_root.join("other.tokens"),
                0,
                257,
                &mut linked_packer,
                &mut linked_count,
                &mut |_| Ok(true),
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn opened_token_cache_directory_cannot_be_redirected_by_symlink_swap() {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("cache");
        let moved = dir.path().join("moved-cache");
        let outside = dir.path().join("outside");
        fs::create_dir(&original).unwrap();
        fs::create_dir(&outside).unwrap();
        let path = original.join("tokens.bin");
        let location = TokenCacheLocation::open(&path, false).unwrap().unwrap();

        fs::rename(&original, &moved).unwrap();
        std::os::unix::fs::symlink(&outside, &original).unwrap();
        let mut cache = location.open_cache(true).unwrap().unwrap();
        cache.write_all(TOKEN_CACHE_MAGIC).unwrap();
        cache.sync_all().unwrap();

        assert_eq!(
            fs::read(moved.join("tokens.bin")).unwrap(),
            TOKEN_CACHE_MAGIC
        );
        assert!(!outside.join("tokens.bin").exists());
    }

    #[test]
    fn new_token_cache_rewinds_after_writing_its_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        let mut packer = SamplePacker::new(3);
        let mut count = 0;
        let mut visit = |_| Ok(true);

        let (documents, keep_going, writer, _) =
            replay_token_cache(&path, 0, 257, &mut packer, &mut count, &mut visit).unwrap();

        assert_eq!(documents, 0);
        assert!(keep_going);
        assert!(writer.is_some());
        assert_eq!(fs::read(path).unwrap(), TOKEN_CACHE_MAGIC);
    }

    #[test]
    fn prepared_corpus_token_rows_are_strictly_decoded() {
        let path = Path::new("shard-00000.tokens.jsonl");
        assert_eq!(
            token_array(&serde_json::json!({"tokens": [1, 2, 3]}), path, 7).unwrap(),
            Some(vec![1, 2, 3])
        );
        assert!(token_array(&serde_json::json!({"tokens": []}), path, 7).is_err());
        assert!(token_array(&serde_json::json!({"tokens": [-1]}), path, 7).is_err());
        assert_eq!(
            token_array(&serde_json::json!({"text": "raw"}), path, 7).unwrap(),
            None
        );
    }

    #[test]
    fn retrieval_wake_encoding_consumes_the_task_adapter_segments() {
        fn expected_text(
            tokenizer: &Tokenizer,
            text: &hermes_train::task::SegmentedText,
            seq_len: usize,
        ) -> EncodedText {
            let mut tokens = text
                .segments
                .iter()
                .flat_map(|segment| tokenizer.encode(segment, false).unwrap())
                .map(i64::from)
                .collect::<Vec<_>>();
            tokens.push(i64::from(tokenizer.eos_token_id()));
            let end_position = tokens.len() - 1;
            tokens.resize(seq_len, i64::from(tokenizer.eos_token_id()));
            EncodedText {
                tokens,
                end_position,
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let tokenizer = write_test_tokenizer(dir.path());
        let path = dir.path().join("retrieval.jsonl");
        let record = serde_json::json!({
            "query": "needle",
            "positive": "positive",
            "negatives": ["negative"]
        });
        fs::write(&path, format!("{record}\n")).unwrap();
        let objective = TaskConfig::RetrievalRepresentation {
            temperature: 0.05,
            layer: None,
            query_prefix: "query-prefix:".into(),
            document_prefix: "document-prefix:".into(),
        };
        let binding = PhaseDataBinding::open(&path).unwrap();
        let mut observed = Vec::new();
        assert_eq!(
            visit_samples_in_order(
                &path,
                &objective,
                &tokenizer,
                SampleStreamConfig {
                    seq_len: 64,
                    shuffle_buffer: 0,
                    seed: 0,
                    token_cache: None,
                    data_binding: &binding,
                    oversized: OversizedRecordPolicy::Abort,
                },
                |sample| {
                    observed.push(sample);
                    Ok(true)
                },
            )
            .unwrap(),
            1
        );

        let TaskExample::RetrievalRepresentation {
            query,
            documents,
            positive_index,
        } = objective.construct_example(&record).unwrap()
        else {
            panic!("expected retrieval task example")
        };
        let TrainingSample::Retrieval {
            query: actual_query,
            documents: actual_documents,
            truncated_tokens,
        } = observed.pop().unwrap()
        else {
            panic!("expected retrieval training sample")
        };
        assert_eq!(positive_index, 0);
        let expected_query = expected_text(&tokenizer, &query, 64);
        assert_eq!(actual_query.tokens, expected_query.tokens);
        assert_eq!(actual_query.end_position, expected_query.end_position);
        let expected_documents = documents
            .iter()
            .map(|document| expected_text(&tokenizer, document, 64))
            .collect::<Vec<_>>();
        assert_eq!(actual_documents.len(), expected_documents.len());
        for (actual, expected) in actual_documents.iter().zip(expected_documents) {
            assert_eq!(actual.tokens, expected.tokens);
            assert_eq!(actual.end_position, expected.end_position);
        }
        assert_eq!(truncated_tokens, 0);
    }

    #[test]
    fn supervised_batch_masks_only_target_positions() {
        let device = Device::ndarray();
        let samples = vec![
            TrainingSample::Supervised {
                tokens: vec![10, 11, 20, 21, 0],
                loss_positions: vec![1, 2, 3],
                truncated_tokens: 4,
            },
            TrainingSample::Supervised {
                tokens: vec![12, 13, 22, 23, 0],
                loss_positions: vec![2, 3],
                truncated_tokens: 0,
            },
        ];
        let TrainingBatch::Language(batch) = make_batch(&samples, 4, &device).unwrap() else {
            panic!("expected masked language batch")
        };
        let LanguageBatch {
            loss_positions: Some(positions),
            stats,
            ..
        } = *batch
        else {
            panic!("expected target positions")
        };
        assert_eq!(
            positions.into_data().to_vec::<i64>().unwrap(),
            vec![1, 2, 3, 6, 7]
        );
        assert_eq!(stats.supervised_tokens, 5);
        assert_eq!(stats.truncated_tokens, 4);
    }

    #[test]
    fn retrieval_batch_labels_positives_among_all_candidates() {
        let device = Device::ndarray();
        let encoded = |start| EncodedText {
            tokens: vec![start, start + 1, 0],
            end_position: 1,
        };
        let samples = vec![
            TrainingSample::Retrieval {
                query: encoded(1),
                documents: vec![encoded(10), encoded(20)],
                truncated_tokens: 0,
            },
            TrainingSample::Retrieval {
                query: encoded(2),
                documents: vec![encoded(30)],
                truncated_tokens: 0,
            },
        ];
        let TrainingBatch::Retrieval(batch) = make_batch(&samples, 3, &device).unwrap() else {
            panic!("expected retrieval batch")
        };
        let RetrievalBatch {
            labels,
            query_end_positions,
            document_end_positions,
            stats,
            ..
        } = *batch;
        assert_eq!(labels.into_data().to_vec::<i64>().unwrap(), vec![0, 2]);
        assert_eq!(
            query_end_positions.into_data().to_vec::<i64>().unwrap(),
            vec![1, 4]
        );
        assert_eq!(
            document_end_positions.into_data().to_vec::<i64>().unwrap(),
            vec![1, 4, 7]
        );
        assert_eq!(stats.retrieval_candidates, 3);
        assert_eq!(stats.compute_tokens, 15);
    }
}
