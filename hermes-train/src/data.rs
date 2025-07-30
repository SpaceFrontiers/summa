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

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result, ensure};
use hermes_llm::Tokenizer;
use hermes_train::corpus::CorpusManifest;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

use crate::wake::ObjectiveConfig;

mod batch;
mod structured;

use batch::EncodedText;
pub(crate) use batch::{
    BatchStats, LanguageBatch, RetrievalBatch, TrainingBatch, TrainingSample, make_batch,
};
use structured::visit_structured_samples;

const TOKENIZE_BATCH: usize = 1_000;
const TOKEN_CACHE_MAGIC: &[u8; 8] = b"HERTOK01";
// Fixed-size sidecar, rather than an in-band footer, keeps the append/repair
// record format unchanged. Its final 32 bytes authenticate all preceding
// metadata; the cache digest is also retained for full-scan identity checks.
const TOKEN_CACHE_INDEX_MAGIC: &[u8; 8] = b"HERTIX01";
const TOKEN_CACHE_INDEX_BYTES: usize = 144;
const MAX_CACHED_DOCUMENT_TOKENS: usize = 100_000_000;

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
    packer: &mut SamplePacker,
    count: &mut usize,
    visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<(usize, bool, Option<TokenCacheWriter>, bool)> {
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
    cache.rewind()?;

    let mut reader = BufReader::new(cache);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    ensure!(
        &magic == TOKEN_CACHE_MAGIC,
        "token cache {} has an unsupported format",
        path.display()
    );
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
        if !packer.push(tokens, count, visit)? {
            return Ok((documents, false, None, false));
        }
    }
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

fn open_data(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open training data {}", path.display()))?;
    if path.extension().is_some_and(|ext| ext == "zst") {
        let decoder = zstd::stream::read::Decoder::new(file)
            .with_context(|| format!("failed to open zstd stream {}", path.display()))?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
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

fn push_documents(
    documents: &mut Vec<String>,
    tokenizer: &Tokenizer,
    packer: &mut SamplePacker,
    count: &mut usize,
    visit: &mut impl FnMut(TrainingSample) -> Result<bool>,
    cache: &mut Option<TokenCacheWriter>,
) -> Result<bool> {
    if documents.is_empty() {
        return Ok(true);
    }
    let encodings = tokenizer.encode_batch(std::mem::take(documents), false)?;
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

fn is_jsonl(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"))
}

fn causal_data_paths(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let manifest_path = if path.is_dir() {
        Some(path.join("manifest.json"))
    } else if path.file_name().is_some_and(|name| name == "manifest.json") {
        Some(path.to_owned())
    } else {
        None
    };
    let Some(manifest_path) = manifest_path else {
        return Ok(vec![path.to_owned()]);
    };
    let manifest: CorpusManifest = serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("invalid corpus manifest {}", manifest_path.display()))?;
    ensure!(
        !manifest.build.shards.is_empty(),
        "corpus manifest has no shards"
    );
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    manifest
        .build
        .shards
        .iter()
        .map(|shard| {
            let shard_path = root.join(&shard.path);
            ensure!(
                shard_path.is_file(),
                "corpus shard {} is missing",
                shard_path.display()
            );
            Ok(shard_path)
        })
        .collect()
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

fn visit_causal_samples(
    path: &Path,
    tokenizer: &Tokenizer,
    seq_len: usize,
    token_cache: Option<&Path>,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    let mut count = 0;
    let mut packer = SamplePacker::new(seq_len);
    let (cached_documents, keep_going, mut cache, cache_complete) = match token_cache {
        Some(path) => replay_token_cache(
            path,
            tokenizer.eos_token_id(),
            &mut packer,
            &mut count,
            &mut visit,
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
        return Ok(count);
    }
    if cache_complete {
        return Ok(count);
    }
    let mut document_number = 0usize;
    let mut documents = Vec::with_capacity(TOKENIZE_BATCH);
    for source_path in causal_data_paths(path)? {
        let mut reader = open_data(&source_path)?;
        if is_jsonl(&source_path) {
            let mut line = String::new();
            let mut line_number = 0usize;
            loop {
                line.clear();
                if reader.read_line(&mut line)? == 0 {
                    break;
                }
                line_number = line_number
                    .checked_add(1)
                    .context("JSONL line count overflows usize")?;
                if line.trim().is_empty() {
                    continue;
                }
                document_number = document_number
                    .checked_add(1)
                    .context("JSONL document count overflows usize")?;
                if document_number <= cached_documents {
                    continue;
                }
                let value: serde_json::Value = serde_json::from_str(&line).with_context(|| {
                    format!("invalid JSONL at {}:{line_number}", source_path.display())
                })?;
                if let Some(tokens) = token_array(&value, &source_path, line_number)? {
                    ensure!(
                        tokens
                            .iter()
                            .all(|token| (*token as usize) < tokenizer.vocab_size()),
                        "tokenized corpus row at {}:{line_number} contains a token outside vocabulary size {}",
                        source_path.display(),
                        tokenizer.vocab_size()
                    );
                    if !push_documents(
                        &mut documents,
                        tokenizer,
                        &mut packer,
                        &mut count,
                        &mut visit,
                        &mut cache,
                    )? {
                        return Ok(count);
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
                        &mut visit,
                    )? {
                        return Ok(count);
                    }
                } else {
                    let document = required_string(&value, "text", &source_path, line_number)?;
                    documents.push(document.to_owned());
                    if documents.len() == TOKENIZE_BATCH
                        && !push_documents(
                            &mut documents,
                            tokenizer,
                            &mut packer,
                            &mut count,
                            &mut visit,
                            &mut cache,
                        )?
                    {
                        return Ok(count);
                    }
                }
            }
        } else {
            document_number = document_number
                .checked_add(1)
                .context("document count overflows usize")?;
            if document_number > cached_documents {
                let mut document = String::new();
                reader.read_to_string(&mut document)?;
                documents.push(document);
            }
        }
    }
    if !push_documents(
        &mut documents,
        tokenizer,
        &mut packer,
        &mut count,
        &mut visit,
        &mut cache,
    )? {
        return Ok(count);
    }
    ensure!(
        document_number >= cached_documents,
        "authoritative corpus has {document_number} documents but token cache has {cached_documents}; remove the stale cache"
    );
    if let Some(cache) = &mut cache {
        cache.publish_index()?;
    }
    Ok(count)
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
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    token_cache: Option<&Path>,
    visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    ensure!(seq_len > 0, "sequence_length must be positive");
    match objective {
        ObjectiveConfig::CausalLm => {
            visit_causal_samples(path, tokenizer, seq_len, token_cache, visit)
        }
        _ => visit_structured_samples(path, objective, tokenizer, seq_len, visit),
    }
}

pub(crate) struct SampleStreamConfig<'a> {
    pub(crate) seq_len: usize,
    pub(crate) shuffle_buffer: usize,
    pub(crate) seed: u64,
    pub(crate) token_cache: Option<&'a Path>,
}

pub(crate) fn visit_samples(
    path: &Path,
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    config: SampleStreamConfig<'_>,
    mut visit: impl FnMut(TrainingSample) -> Result<bool>,
) -> Result<usize> {
    if config.shuffle_buffer == 0 {
        return visit_samples_in_order(
            path,
            objective,
            tokenizer,
            config.seq_len,
            config.token_cache,
            visit,
        );
    }

    let mut shuffler = ShuffleBuffer::new(config.shuffle_buffer, config.seed);
    let mut keep_going = true;
    let count = visit_samples_in_order(
        path,
        objective,
        tokenizer,
        config.seq_len,
        config.token_cache,
        |sample| {
            if let Some(sample) = shuffler.push(sample) {
                keep_going = visit(sample)?;
            }
            Ok(keep_going)
        },
    )?;

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
    objective: &ObjectiveConfig,
    tokenizer: &Tokenizer,
    seq_len: usize,
    token_cache: Option<&Path>,
) -> Result<usize> {
    visit_samples_in_order(path, objective, tokenizer, seq_len, token_cache, |_| {
        Ok(true)
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Cursor, Read};

    use burn::tensor::Device;

    use super::*;

    #[test]
    fn zstd_data_reader_streams_decompressed_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl.zst");
        let source = b"{\"text\":\"one\"}\n{\"text\":\"two\"}\n";
        let compressed = zstd::stream::encode_all(Cursor::new(source), 1).unwrap();
        fs::write(&path, compressed).unwrap();

        let mut reader = open_data(&path).unwrap();
        let mut decoded = String::new();
        reader.read_to_string(&mut decoded).unwrap();
        assert_eq!(decoded.as_bytes(), source);
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
            replay_token_cache(&path, 0, &mut packer, &mut count, &mut |sample| {
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
            replay_token_cache(&path, 0, &mut replay, &mut replay_count, &mut |_| Ok(true))
                .unwrap();
        assert_eq!(documents, 3);
        assert_eq!(replay_count, 2);
    }

    fn write_test_token_cache(path: &Path, documents: &[&[u32]], complete: bool) {
        let mut packer = SamplePacker::new(4);
        let mut count = 0;
        let (_, keep_going, mut writer, _) =
            replay_token_cache(path, 0, &mut packer, &mut count, &mut |_| Ok(true)).unwrap();
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
            replay_token_cache(&path, 0, &mut packer, &mut replayed, &mut |_| Ok(true)).unwrap();
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
            replay_token_cache(&path, 0, &mut packer, &mut count, &mut |_| Ok(true)).unwrap();
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
            replay_token_cache(&path, 0, &mut packer, &mut count, &mut |sample| {
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
            replay_token_cache(&changed, 0, &mut packer, &mut count, &mut |_| Ok(true)).unwrap();
        writer.as_mut().unwrap().append(&[]).unwrap();
        assert!(!token_cache_index_path(&changed).exists());
        assert_eq!(indexed_causal_sample_count(&changed, 2).unwrap(), None);
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
            replay_token_cache(&path, 0, &mut packer, &mut count, &mut |_| Ok(true)).unwrap();
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
        let error = replay_token_cache(&path, 0, &mut packer, &mut count, &mut |_| Ok(true))
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
            replay_token_cache(&cache_path, 0, &mut packer, &mut count, &mut |_| Ok(true)).is_err()
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
            replay_token_cache(&path, 0, &mut packer, &mut count, &mut visit).unwrap();

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
