//! Exact, fixed-schema reads from an immutable artifact directory.
//!
//! Resume code must not authenticate a pathname and later reopen that path:
//! a directory or file can be replaced between those operations. The shared
//! primitives in this module open directories and members once, retain those
//! handles, and stream-verify or bounded-read them only when requested. The
//! final identity check distinguishes a transient A->B->A directory swap (the
//! opened generation remains authoritative) from a replacement which is still
//! published when loading finishes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

#[cfg(unix)]
use std::ffi::{CStr, CString, OsStr, OsString};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};

static IMMUTABLE_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const MAX_PINNED_DIRECTORY_ENTRIES: usize = 4_096;
const SHA256_HEX_LENGTH: usize = 64;
const SHA256_IDENTITY_PREFIX: &str = "sha256:";

/// Return the SHA-256 digest as exactly 64 lowercase hexadecimal characters.
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Return the canonical content identity `sha256:<64 lowercase hex>`.
pub fn sha256_identity(bytes: &[u8]) -> String {
    format!("{SHA256_IDENTITY_PREFIX}{}", sha256_hex(bytes))
}

/// Prefix an already-computed canonical raw digest after validating it.
pub fn sha256_identity_from_hex(digest: &str, label: &str) -> Result<String> {
    validate_sha256_hex(digest, label)?;
    Ok(format!("{SHA256_IDENTITY_PREFIX}{digest}"))
}

/// Serialize a fixed-schema value as compact JSON and hash those exact bytes.
pub fn json_sha256_identity<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    Ok(sha256_identity(&serde_json::to_vec(value)?))
}

/// Validate an unprefixed SHA-256 digest in its canonical lowercase form.
pub fn validate_sha256_hex(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == SHA256_HEX_LENGTH
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} must use exactly 64 lowercase hexadecimal characters"
    );
    Ok(())
}

/// Validate a canonical prefixed SHA-256 content identity.
pub fn validate_sha256_identity(value: &str, label: &str) -> Result<()> {
    let digest = value
        .strip_prefix(SHA256_IDENTITY_PREFIX)
        .with_context(|| format!("{label} must use the `sha256:<64 lowercase hex>` form"))?;
    validate_sha256_hex(digest, label)
        .with_context(|| format!("{label} must use the `sha256:<64 lowercase hex>` form"))
}

pub fn ensure_real_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting {label} {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} {} is not a real directory",
        path.display()
    );
    Ok(())
}

pub fn ensure_directory(path: &Path, label: &str) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating {label} {}", path.display()))?;
    ensure_real_directory(path, label)
}

pub fn sync_directory(path: &Path) -> Result<()> {
    ensure_real_directory(path, "synced directory")?;
    File::open(path)?.sync_all()?;
    Ok(())
}

pub(crate) fn open_regular(path: &Path, label: &str) -> Result<(File, StableIdentity)> {
    let path_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting {label} {}", path.display()))?;
    ensure!(
        path_metadata.is_file() && !path_metadata.file_type().is_symlink(),
        "{label} {} is not a regular file or is a symlink",
        path.display()
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let file = options
        .open(path)
        .with_context(|| format!("opening {label} {}", path.display()))?;
    let identity = StableIdentity::from_metadata(&file.metadata()?);
    ensure!(
        identity == StableIdentity::from_metadata(&path_metadata),
        "{label} {} changed while it was opened",
        path.display()
    );
    Ok((file, identity))
}

pub fn ensure_regular_file(path: &Path, label: &str) -> Result<()> {
    let _ = open_regular(path, label)?;
    Ok(())
}

/// Hash an already-open regular file generation. When `capture_limit` is
/// present, return its bytes only after rejecting the size before allocation.
/// The digest is returned as lowercase hexadecimal without a `sha256:` prefix.
pub fn hash_open_file(
    file: &mut File,
    capture_limit: Option<u64>,
    label: &str,
) -> Result<(u64, String, Option<Vec<u8>>)> {
    let before = file
        .metadata()
        .with_context(|| format!("inspecting opened {label}"))?;
    ensure!(before.is_file(), "opened {label} is not a regular file");
    let identity = StableIdentity::from_metadata(&before);
    let mut captured = match capture_limit {
        Some(limit) => {
            ensure!(
                before.len() <= limit,
                "opened {label} exceeds its {limit}-byte limit"
            );
            let capacity = usize::try_from(before.len())
                .with_context(|| format!("{label} is too large for this address space"))?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(capacity)
                .with_context(|| format!("reserving authenticated buffer for {label}"))?;
            Some(bytes)
        }
        None => None,
    };
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("rewinding opened {label}"))?;
    let mut hasher = Sha256::new();
    let mut observed = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("reading opened {label}"))?;
        if read == 0 {
            break;
        }
        observed = observed
            .checked_add(u64::try_from(read).context("artifact read length exceeds u64")?)
            .context("artifact length overflow")?;
        ensure!(
            observed <= before.len(),
            "opened {label} grew while it was read"
        );
        hasher.update(&buffer[..read]);
        if let Some(captured) = &mut captured {
            captured.extend_from_slice(&buffer[..read]);
        }
    }
    let after = file
        .metadata()
        .with_context(|| format!("reinspecting opened {label}"))?;
    ensure!(
        StableIdentity::from_metadata(&after) == identity && observed == after.len(),
        "opened {label} changed while it was read"
    );
    Ok((observed, format!("{:x}", hasher.finalize()), captured))
}

/// Stream-hash an opened file while requiring the length pinned by an earlier
/// caller snapshot. Authentication and consumption still use the same file
/// descriptor, and changes before or during the read fail closed.
pub fn hash_open_file_exact_length(
    file: &mut File,
    expected_bytes: u64,
    label: &str,
) -> Result<String> {
    let observed_before = file
        .metadata()
        .with_context(|| format!("inspecting opened {label}"))?
        .len();
    ensure!(
        observed_before >= expected_bytes,
        "{label} was truncated before it was hashed"
    );
    ensure!(
        observed_before <= expected_bytes,
        "{label} grew before it was hashed"
    );
    let (observed, digest, _) = hash_open_file(file, None, label)?;
    ensure!(
        observed >= expected_bytes,
        "{label} was truncated while it was hashed"
    );
    ensure!(
        observed <= expected_bytes,
        "{label} grew while it was hashed"
    );
    Ok(digest)
}

fn read_hashed_path(
    path: &Path,
    capture_limit: Option<u64>,
    label: &str,
) -> Result<(u64, String, Option<Vec<u8>>)> {
    let (mut file, identity) = open_regular(path, label)?;
    let result = hash_open_file(&mut file, capture_limit, label)?;
    let current = fs::symlink_metadata(path)
        .with_context(|| format!("reinspecting {label} {}", path.display()))?;
    ensure!(
        StableIdentity::from_metadata(&current) == identity,
        "{label} {} changed while it was consumed",
        path.display()
    );
    Ok(result)
}

#[cfg(test)]
fn read_regular(path: &Path) -> Result<Vec<u8>> {
    read_regular_bounded(path, u64::MAX, &format!("artifact {}", path.display()))
}

pub fn read_regular_bounded(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>> {
    read_hashed_path(path, Some(max_bytes), label)?
        .2
        .context("authenticated read did not capture bytes")
}

/// Stream-hash a stable regular file and return an unprefixed lowercase digest.
pub fn hash_regular_file_hex(path: &Path) -> Result<(u64, String)> {
    let label = format!("artifact {}", path.display());
    let (bytes, digest, _) = read_hashed_path(path, None, &label)?;
    Ok((bytes, digest))
}

/// Stream-hash a stable regular file and return its canonical content identity.
pub fn hash_regular_file(path: &Path) -> Result<(u64, String)> {
    let (bytes, digest) = hash_regular_file_hex(path)?;
    Ok((bytes, sha256_identity_from_hex(&digest, "artifact digest")?))
}

pub fn sync_regular_file(path: &Path, label: &str) -> Result<()> {
    let (file, identity) = open_regular(path, label)?;
    file.sync_all()
        .with_context(|| format!("syncing {label} {}", path.display()))?;
    let current = fs::symlink_metadata(path)
        .with_context(|| format!("reinspecting {label} {}", path.display()))?;
    ensure!(
        StableIdentity::from_metadata(&current) == identity,
        "{label} {} changed while it was synced",
        path.display()
    );
    Ok(())
}

pub fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("creating immutable file {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Idempotently publish immutable bytes without ever replacing an existing
/// pathname. Concurrent same-content publishers converge on the same file.
pub fn atomic_write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("atomic artifact has no parent")?;
    ensure_directory(parent, "atomic artifact parent")?;
    let expected_bytes = u64::try_from(bytes.len()).context("immutable artifact exceeds u64")?;
    if path.exists() {
        ensure!(
            read_regular_bounded(path, expected_bytes, "existing immutable artifact")? == bytes,
            "immutable artifact already exists with different bytes"
        );
        return Ok(());
    }
    let name = path
        .file_name()
        .context("atomic artifact has no file name")?
        .to_string_lossy();
    let temporary = parent.join(format!(
        ".{name}.staging-{}-{}",
        std::process::id(),
        IMMUTABLE_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    if let Err(error) = write_new_synced(&temporary, bytes) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    let publication = (|| -> Result<()> {
        match fs::hard_link(&temporary, path) {
            Ok(()) => {
                // `hard_link` resolves its source pathname after the staged
                // handle has been closed. A process with write access to the
                // parent could replace that name in the interval. Never
                // report a successful immutable publication until the exact
                // destination bytes have independently been authenticated.
                ensure!(
                    read_regular_bounded(path, expected_bytes, "published immutable artifact")?
                        == bytes,
                    "published immutable artifact differs from staged bytes"
                );
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                ensure!(
                    read_regular_bounded(path, expected_bytes, "concurrent immutable artifact")?
                        == bytes,
                    "concurrent immutable publication differs"
                );
                Ok(())
            }
            Err(error) => Err(error).context("atomically linking immutable artifact"),
        }
    })();
    let cleanup = fs::remove_file(&temporary)
        .with_context(|| format!("removing immutable staging file {}", temporary.display()));
    match (publication, cleanup) {
        (Err(primary), _) => Err(primary),
        (Ok(()), Err(cleanup)) => Err(cleanup),
        (Ok(()), Ok(())) => sync_directory(parent),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StableIdentity {
    length: u64,
    modified: Option<SystemTime>,
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

impl StableIdentity {
    pub fn from_metadata(metadata: &fs::Metadata) -> Self {
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

    #[cfg(unix)]
    pub fn same_object(&self, other: &Self) -> bool {
        self.device == other.device && self.inode == other.inode && self.mode == other.mode
    }

    #[cfg(not(unix))]
    pub fn same_object(&self, other: &Self) -> bool {
        self == other
    }
}

/// Pinned members and the identity of the directory handle they came from.
pub struct AuthenticatedDirectorySnapshot {
    path: PathBuf,
    label: String,
    identity: StableIdentity,
    files: BTreeMap<String, CapturedFile>,
    file_identities: BTreeMap<String, StableIdentity>,
    #[cfg(unix)]
    directory: PinnedDirectory,
}

struct CapturedFiles {
    identity: StableIdentity,
    files: BTreeMap<String, CapturedFile>,
    file_identities: BTreeMap<String, StableIdentity>,
}

struct CapturedFile {
    file: File,
    identity: StableIdentity,
    verified: Option<(u64, String)>,
}

#[cfg(unix)]
struct UnixCapture {
    captured: CapturedFiles,
    directory: PinnedDirectory,
}

impl AuthenticatedDirectorySnapshot {
    pub(crate) fn capture(path: &Path, expected: &[&str], label: &str) -> Result<Self> {
        let expected = validate_expected_names(expected)?;
        #[cfg(unix)]
        let UnixCapture {
            captured,
            directory,
        } = capture_unix(path, &expected, label)?;
        #[cfg(not(unix))]
        let captured = capture_portable(path, &expected, label)?;
        Ok(Self {
            path: path.to_owned(),
            label: label.to_owned(),
            identity: captured.identity,
            files: captured.files,
            file_identities: captured.file_identities,
            #[cfg(unix)]
            directory,
        })
    }

    /// Read one pinned member, with a caller-owned bound. Directory capture is
    /// deliberately handle-only: eager reads multiply peak memory by every
    /// model/optimizer member in a generation.
    pub(crate) fn read_bounded(&mut self, name: &str, max_bytes: u64) -> Result<Vec<u8>> {
        let member = self
            .files
            .get_mut(name)
            .with_context(|| format!("authenticated {} has no `{name}`", self.label))?;
        member.read_bounded(max_bytes, &format!("{} file `{name}`", self.label))
    }

    /// Stream-authenticate a pinned member without materializing it.
    pub(crate) fn verify(
        &mut self,
        name: &str,
        expected_bytes: u64,
        expected_sha256: &str,
    ) -> Result<()> {
        let member = self
            .files
            .get_mut(name)
            .with_context(|| format!("authenticated {} has no `{name}`", self.label))?;
        member.verify(
            expected_bytes,
            expected_sha256,
            &format!("{} file `{name}`", self.label),
        )
    }

    /// Read a member that was already stream-authenticated with [`Self::verify`].
    pub(crate) fn take(&mut self, name: &str, max_bytes: u64) -> Result<Vec<u8>> {
        let mut member = self
            .files
            .remove(name)
            .with_context(|| format!("authenticated {} has no `{name}`", self.label))?;
        let (expected_bytes, expected_sha256) = member
            .verified
            .clone()
            .with_context(|| format!("authenticated {} `{name}` was not verified", self.label))?;
        member.read_verified(
            expected_bytes,
            &expected_sha256,
            max_bytes,
            &format!("{} file `{name}`", self.label),
        )
    }

    /// Require the exact directory opened at capture time to still occupy its
    /// published pathname.  All returned data remains sourced from captured
    /// handles; this check only prevents accepting a persistent replacement.
    pub(crate) fn ensure_still_published(&self) -> Result<()> {
        self.ensure_still_published_after_children(|| Ok(()))
    }

    fn ensure_still_published_after_children(
        &self,
        after_children: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        self.ensure_published_directory_identity()?;
        #[cfg(unix)]
        {
            let expected = self
                .file_identities
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            ensure!(
                utf8_names(&self.directory, &self.label, expected.len())? == expected,
                "published {} file set changed during authenticated load",
                self.label
            );
            for (name, expected) in &self.file_identities {
                let file = self.directory.open_child(OsStr::new(name), &self.label)?;
                let observed = file.metadata().with_context(|| {
                    format!("reinspecting published {} file `{name}`", self.label)
                })?;
                ensure!(
                    observed.is_file() && StableIdentity::from_metadata(&observed) == *expected,
                    "published {} file `{name}` changed during authenticated load",
                    self.label
                );
            }
        }
        #[cfg(not(unix))]
        {
            let observed =
                portable_utf8_names(&self.path, &self.label, self.file_identities.len())?;
            ensure!(
                observed
                    == self
                        .file_identities
                        .keys()
                        .cloned()
                        .collect::<BTreeSet<_>>(),
                "published {} file set changed during authenticated load",
                self.label
            );
            for (name, expected) in &self.file_identities {
                let metadata = fs::symlink_metadata(self.path.join(name))?;
                ensure!(
                    metadata.is_file()
                        && !metadata.file_type().is_symlink()
                        && StableIdentity::from_metadata(&metadata) == *expected,
                    "published {} file `{name}` changed during authenticated load",
                    self.label
                );
            }
        }
        // The pathname can be swapped after the first identity check while
        // child handles are inspected. Close that avoidable race before the
        // authenticated bytes are allowed to commit caller state.
        after_children()?;
        self.ensure_published_directory_identity()
    }

    fn ensure_published_directory_identity(&self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.path).with_context(|| {
            format!(
                "reinspecting published {} {}",
                self.label,
                self.path.display()
            )
        })?;
        ensure!(
            metadata.is_dir()
                && !metadata.file_type().is_symlink()
                && StableIdentity::from_metadata(&metadata).same_object(&self.identity),
            "published {} changed during authenticated load",
            self.label
        );
        Ok(())
    }
}

fn validate_expected_names(expected: &[&str]) -> Result<BTreeSet<String>> {
    ensure!(
        expected.len() <= MAX_PINNED_DIRECTORY_ENTRIES,
        "authenticated artifact schema exceeds its {MAX_PINNED_DIRECTORY_ENTRIES}-entry limit"
    );
    let mut names = BTreeSet::new();
    for name in expected {
        let mut components = Path::new(name).components();
        ensure!(
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
            "authenticated artifact name `{name}` is not one safe path component"
        );
        ensure!(
            names.insert((*name).to_owned()),
            "authenticated artifact schema repeats `{name}`"
        );
    }
    ensure!(!names.is_empty(), "authenticated artifact schema is empty");
    Ok(names)
}

impl CapturedFile {
    fn open(file: File, label: &str) -> Result<Self> {
        let metadata = file
            .metadata()
            .with_context(|| format!("inspecting opened {label}"))?;
        ensure!(metadata.is_file(), "opened {label} is not a regular file");
        Ok(Self {
            file,
            identity: StableIdentity::from_metadata(&metadata),
            verified: None,
        })
    }

    fn ensure_unchanged(&self, observed_bytes: u64, label: &str) -> Result<()> {
        let after = self
            .file
            .metadata()
            .with_context(|| format!("reinspecting opened {label}"))?;
        ensure!(
            StableIdentity::from_metadata(&after) == self.identity && observed_bytes == after.len(),
            "opened {label} changed while it was consumed"
        );
        Ok(())
    }

    fn read_bounded(&mut self, max_bytes: u64, label: &str) -> Result<Vec<u8>> {
        ensure!(
            self.identity.length <= max_bytes,
            "opened {label} exceeds its {max_bytes}-byte limit"
        );
        let (observed, _, captured) = hash_open_file(&mut self.file, Some(max_bytes), label)?;
        self.ensure_unchanged(observed, label)?;
        captured.context("authenticated read did not capture bytes")
    }

    fn verify(&mut self, expected_bytes: u64, expected_sha256: &str, label: &str) -> Result<()> {
        ensure!(
            self.identity.length == expected_bytes,
            "opened {label} length differs from its manifest"
        );
        let (observed, digest, _) = hash_open_file(&mut self.file, None, label)?;
        self.ensure_unchanged(observed, label)?;
        ensure!(
            format!("sha256:{digest}") == expected_sha256,
            "opened {label} hash differs from its manifest"
        );
        self.verified = Some((expected_bytes, expected_sha256.to_owned()));
        Ok(())
    }

    fn read_verified(
        &mut self,
        expected_bytes: u64,
        expected_sha256: &str,
        max_bytes: u64,
        label: &str,
    ) -> Result<Vec<u8>> {
        ensure!(
            expected_bytes <= max_bytes,
            "opened {label} exceeds its {max_bytes}-byte limit"
        );
        let bytes = self.read_bounded(max_bytes, label)?;
        ensure!(
            bytes.len() as u64 == expected_bytes && sha256_identity(&bytes) == expected_sha256,
            "opened {label} differs from its authenticated manifest entry"
        );
        self.verified = Some((expected_bytes, expected_sha256.to_owned()));
        Ok(bytes)
    }
}

#[cfg(unix)]
#[derive(Debug)]
pub struct PinnedDirectory {
    file: File,
}

#[cfg(unix)]
impl PinnedDirectory {
    pub fn open(path: &Path, label: &str) -> Result<(Self, StableIdentity)> {
        let path_metadata = fs::symlink_metadata(path)
            .with_context(|| format!("inspecting {label} {}", path.display()))?;
        ensure!(
            path_metadata.is_dir() && !path_metadata.file_type().is_symlink(),
            "{label} {} is not a real directory",
            path.display()
        );
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .with_context(|| format!("opening {label} {}", path.display()))?;
        let identity = StableIdentity::from_metadata(&file.metadata()?);
        ensure!(
            identity == StableIdentity::from_metadata(&path_metadata),
            "{label} changed while it was opened"
        );
        Ok((Self { file }, identity))
    }

    pub fn from_open_directory(file: File, label: &str) -> Result<Self> {
        let metadata = file
            .metadata()
            .with_context(|| format!("inspecting opened {label}"))?;
        ensure!(metadata.is_dir(), "opened {label} is not a directory");
        Ok(Self { file })
    }

    pub fn identity(&self) -> Result<StableIdentity> {
        Ok(StableIdentity::from_metadata(&self.file.metadata()?))
    }

    pub fn open_child(&self, name: &OsStr, label: &str) -> Result<File> {
        let mut components = Path::new(name).components();
        ensure!(
            matches!(components.next(), Some(Component::Normal(component)) if component == name)
                && components.next().is_none(),
            "authenticated {label} child `{}` is not one safe path component",
            name.to_string_lossy()
        );
        let name =
            CString::new(name.as_bytes()).context("authenticated artifact name contains NUL")?;
        // SAFETY: the directory descriptor remains owned by `self`; openat
        // returns a new descriptor and O_NOFOLLOW rejects symlink leaves.
        let descriptor = unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
                0,
            )
        };
        if descriptor < 0 {
            return Err(std::io::Error::last_os_error()).with_context(|| {
                format!(
                    "opening authenticated {label} file `{}`",
                    name.to_string_lossy()
                )
            });
        }
        // SAFETY: openat returned a newly owned descriptor on success.
        Ok(unsafe { File::from_raw_fd(descriptor) })
    }

    pub fn open_relative_file(&self, relative: &Path, label: &str) -> Result<File> {
        ensure!(
            !relative.is_absolute(),
            "authenticated relative path must not be absolute"
        );
        let mut directory = Self {
            file: self
                .file
                .try_clone()
                .with_context(|| format!("duplicating authenticated {label} directory"))?,
        };
        let mut components = relative.components().peekable();
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                anyhow::bail!(
                    "authenticated path `{}` is not normalized",
                    relative.display()
                );
            };
            let child = directory.open_child(name, label)?;
            let metadata = child
                .metadata()
                .with_context(|| format!("inspecting authenticated {label} child"))?;
            if components.peek().is_none() {
                ensure!(
                    metadata.is_file(),
                    "authenticated {label} `{}` is not a regular file",
                    relative.display()
                );
                return Ok(child);
            }
            ensure!(
                metadata.is_dir(),
                "authenticated {label} parent for `{}` is not a directory",
                relative.display()
            );
            directory = Self::from_open_directory(child, label)?;
        }
        anyhow::bail!("authenticated path has no file name")
    }

    /// Enumerate a pinned directory with a conservative default cardinality
    /// bound. Fixed-schema callers should use [`Self::entries_bounded`] with
    /// their exact expected member count.
    pub fn entries(&self, label: &str) -> Result<Vec<OsString>> {
        self.entries_bounded(label, MAX_PINNED_DIRECTORY_ENTRIES)
    }

    pub fn entries_bounded(&self, label: &str, maximum_entries: usize) -> Result<Vec<OsString>> {
        let maximum_entries = maximum_entries.min(MAX_PINNED_DIRECTORY_ENTRIES);
        let current = CString::new(".").expect("static path contains no NUL");
        // `dup` would share one open-file-description directory cursor with
        // every concurrent enumeration. Open `.` relative to the pinned
        // descriptor instead so this DIR stream owns an independent cursor.
        // SAFETY: the pinned directory descriptor and static C string are
        // valid for the duration of this call; openat returns a new owned fd.
        let descriptor = unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                current.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW,
                0,
            )
        };
        if descriptor < 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("opening an independent {label} directory cursor"));
        }
        // SAFETY: `descriptor` is valid and ownership transfers to fdopendir.
        let stream = unsafe { libc::fdopendir(descriptor) };
        if stream.is_null() {
            let error = std::io::Error::last_os_error();
            // SAFETY: fdopendir did not take ownership on failure.
            unsafe { libc::close(descriptor) };
            return Err(error).with_context(|| format!("enumerating {label}"));
        }
        struct DirectoryStream(*mut libc::DIR);
        impl Drop for DirectoryStream {
            fn drop(&mut self) {
                // SAFETY: the wrapper uniquely owns the DIR pointer.
                unsafe { libc::closedir(self.0) };
            }
        }
        let stream = DirectoryStream(stream);
        let mut names = Vec::new();
        loop {
            let errno = errno_slot();
            if let Some(slot) = errno {
                // SAFETY: errno_slot returns this thread's errno storage.
                unsafe { *slot = 0 };
            }
            // SAFETY: stream owns a private DIR and each name is copied before
            // the next call.
            let entry = unsafe { libc::readdir(stream.0) };
            if entry.is_null() {
                let code = errno.map_or(0, |slot| unsafe { *slot });
                if code != 0 {
                    return Err(std::io::Error::from_raw_os_error(code))
                        .with_context(|| format!("enumerating {label}"));
                }
                break;
            }
            let raw = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
            if raw == b"." || raw == b".." {
                continue;
            }
            ensure!(
                names.len() < maximum_entries,
                "{label} exceeds its {maximum_entries}-entry limit"
            );
            names.push(OsString::from_vec(raw.to_vec()));
        }
        names.sort();
        Ok(names)
    }
}

#[cfg(unix)]
fn capture_unix(path: &Path, expected: &BTreeSet<String>, label: &str) -> Result<UnixCapture> {
    let (directory, identity) = PinnedDirectory::open(path, label)?;
    ensure!(
        utf8_names(&directory, label, expected.len())? == *expected,
        "{label} file set differs from its fixed schema"
    );
    let mut files = BTreeMap::new();
    let mut file_identities = BTreeMap::new();
    for name in expected {
        let file = directory.open_child(OsStr::new(name), label)?;
        let member = CapturedFile::open(file, &format!("{label} `{name}`"))?;
        file_identities.insert(name.clone(), member.identity.clone());
        files.insert(name.clone(), member);
    }
    ensure!(
        StableIdentity::from_metadata(&directory.file.metadata()?) == identity,
        "{label} changed while its files were captured"
    );
    Ok(UnixCapture {
        captured: CapturedFiles {
            identity,
            files,
            file_identities,
        },
        directory,
    })
}

#[cfg(unix)]
fn utf8_names(
    directory: &PinnedDirectory,
    label: &str,
    maximum_entries: usize,
) -> Result<BTreeSet<String>> {
    directory
        .entries_bounded(label, maximum_entries)?
        .into_iter()
        .map(|name| {
            name.into_string()
                .map_err(|_| anyhow::anyhow!("{label} contains a non-UTF-8 file name"))
        })
        .collect()
}

#[cfg(not(unix))]
fn capture_portable(
    path: &Path,
    expected: &BTreeSet<String>,
    label: &str,
) -> Result<CapturedFiles> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting {label} {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} {} is not a real directory",
        path.display()
    );
    let identity = StableIdentity::from_metadata(&metadata);
    let observed = portable_utf8_names(path, label, expected.len())?;
    ensure!(
        observed == *expected,
        "{label} file set differs from its fixed schema"
    );
    let mut files = BTreeMap::new();
    let mut file_identities = BTreeMap::new();
    for name in expected {
        let child = path.join(name);
        let child_metadata = fs::symlink_metadata(&child)?;
        ensure!(
            child_metadata.is_file() && !child_metadata.file_type().is_symlink(),
            "{label} `{name}` is not a regular non-symlink file"
        );
        let member = CapturedFile::open(File::open(&child)?, &format!("{label} `{name}`"))?;
        file_identities.insert(name.clone(), member.identity.clone());
        files.insert(name.clone(), member);
    }
    ensure!(
        StableIdentity::from_metadata(&fs::symlink_metadata(path)?) == identity,
        "{label} changed while its files were captured"
    );
    Ok(CapturedFiles {
        identity,
        files,
        file_identities,
    })
}

#[cfg(not(unix))]
fn portable_utf8_names(
    path: &Path,
    label: &str,
    maximum_entries: usize,
) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(path)? {
        ensure!(
            names.len() < maximum_entries,
            "{label} exceeds its {maximum_entries}-entry limit"
        );
        let name = entry?
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("{label} contains a non-UTF-8 file name"))?;
        names.insert(name);
    }
    Ok(names)
}

#[cfg(all(
    unix,
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos",
        target_os = "freebsd"
    )
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    // SAFETY: libc returns this thread's errno storage.
    Some(unsafe { libc::__error() })
}

#[cfg(all(
    unix,
    any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "emscripten",
        target_os = "hurd",
        target_os = "redox"
    )
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    // SAFETY: libc returns this thread's errno storage.
    Some(unsafe { libc::__errno_location() })
}

#[cfg(all(
    unix,
    any(target_os = "android", target_os = "netbsd", target_os = "openbsd")
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    // SAFETY: libc returns this thread's errno storage.
    Some(unsafe { libc::__errno() })
}

#[cfg(all(
    unix,
    not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "emscripten",
        target_os = "hurd",
        target_os = "redox",
        target_os = "android",
        target_os = "netbsd",
        target_os = "openbsd"
    ))
))]
fn errno_slot() -> Option<*mut libc::c_int> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_file_publication_is_idempotent_and_leaves_no_staging_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("receipt.json");

        atomic_write_new(&path, b"sealed").unwrap();
        atomic_write_new(&path, b"sealed").unwrap();
        let error = atomic_write_new(&path, b"different")
            .unwrap_err()
            .to_string();

        assert!(error.contains("different bytes"), "{error}");
        assert_eq!(read_regular(&path).unwrap(), b"sealed");
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".staging-")
        }));
    }

    #[test]
    fn directory_capture_pins_large_members_without_eager_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("weights.safetensors");
        let file = File::create(&path).unwrap();
        file.set_len(2 * 1024 * 1024 * 1024).unwrap();
        drop(file);

        let mut snapshot = AuthenticatedDirectorySnapshot::capture(
            directory.path(),
            &["weights.safetensors"],
            "large fixture",
        )
        .unwrap();
        let error = snapshot
            .read_bounded("weights.safetensors", 1024)
            .unwrap_err()
            .to_string();
        assert!(error.contains("1024-byte limit"), "{error}");
    }

    #[test]
    fn fixed_schema_capture_bounds_hostile_directory_cardinality() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("state.json"), b"state").unwrap();
        for index in 0..128 {
            fs::write(directory.path().join(format!("unexpected-{index:03}")), b"").unwrap();
        }

        let error = AuthenticatedDirectorySnapshot::capture(
            directory.path(),
            &["state.json"],
            "hostile fixture",
        )
        .err()
        .expect("oversized directory must fail before collecting every name");
        assert!(format!("{error:#}").contains("1-entry limit"), "{error:#}");
    }

    #[test]
    fn bounded_path_read_rejects_size_before_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.json");
        let file = File::create(&path).unwrap();
        file.set_len(1024 * 1024).unwrap();
        drop(file);

        let error = read_regular_bounded(&path, 1024, "bounded fixture")
            .unwrap_err()
            .to_string();
        assert!(error.contains("1024-byte limit"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn path_helpers_reject_symlinked_files_and_directories() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.bin");
        let file_link = directory.path().join("file-link.bin");
        let directory_link = directory.path().join("directory-link");
        fs::write(&target, b"target").unwrap();
        symlink(&target, &file_link).unwrap();
        symlink(directory.path(), &directory_link).unwrap();

        assert!(read_regular_bounded(&file_link, 64, "symlink fixture").is_err());
        assert!(hash_regular_file(&file_link).is_err());
        assert!(ensure_real_directory(&directory_link, "symlink directory").is_err());
    }

    #[test]
    fn verified_member_is_read_from_the_same_pinned_handle() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("state.bin"), b"pinned-state").unwrap();
        let mut snapshot = AuthenticatedDirectorySnapshot::capture(
            directory.path(),
            &["state.bin"],
            "state fixture",
        )
        .unwrap();
        snapshot
            .verify("state.bin", 12, &sha256_identity(b"pinned-state"))
            .unwrap();
        assert_eq!(snapshot.take("state.bin", 64).unwrap(), b"pinned-state");
    }

    #[test]
    fn sha256_syntax_is_canonical_and_distinguishes_raw_from_prefixed() {
        const ABC_HEX: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_hex(b"abc"), ABC_HEX);
        assert_eq!(sha256_identity(b"abc"), format!("sha256:{ABC_HEX}"));
        validate_sha256_hex(ABC_HEX, "raw digest").unwrap();
        validate_sha256_identity(&format!("sha256:{ABC_HEX}"), "identity").unwrap();

        for invalid in [
            &ABC_HEX[..63],
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
            "ga7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            &format!("sha256:{ABC_HEX}"),
        ] {
            assert!(validate_sha256_hex(invalid, "raw digest").is_err());
        }
        assert!(validate_sha256_identity(ABC_HEX, "identity").is_err());
        assert!(
            validate_sha256_identity(
                "sha256:BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
                "identity"
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn pinned_directory_enumerations_have_independent_cursors() {
        use std::sync::{Arc, Barrier};

        let directory = tempfile::tempdir().unwrap();
        for name in ["alpha", "beta", "gamma"] {
            fs::write(directory.path().join(name), name).unwrap();
        }
        let (pinned, _) = PinnedDirectory::open(directory.path(), "concurrent fixture").unwrap();
        let pinned = Arc::new(pinned);
        let barrier = Arc::new(Barrier::new(8));
        let expected = ["alpha", "beta", "gamma"]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let pinned = Arc::clone(&pinned);
                let barrier = Arc::clone(&barrier);
                let expected = expected.clone();
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..64 {
                        assert_eq!(pinned.entries("concurrent fixture").unwrap(), expected);
                    }
                });
            }
        });
    }

    #[cfg(unix)]
    #[test]
    fn pinned_directory_rejects_non_component_children_and_caps_enumeration() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("safe"), b"safe").unwrap();
        let (pinned, _) = PinnedDirectory::open(directory.path(), "hostile fixture").unwrap();

        assert!(
            pinned
                .open_child(OsStr::new("safe"), "hostile fixture")
                .is_ok()
        );
        for hostile in ["", ".", "..", "../safe", "subdir/safe", "/tmp/safe"] {
            let error = pinned
                .open_child(OsStr::new(hostile), "hostile fixture")
                .unwrap_err()
                .to_string();
            assert!(error.contains("one safe path component"), "{error}");
        }

        assert_eq!(
            pinned
                .entries_bounded("hostile fixture", usize::MAX)
                .unwrap(),
            vec![OsString::from("safe")]
        );
    }

    #[test]
    fn fixed_schema_cardinality_has_a_global_cap() {
        let names = (0..=MAX_PINNED_DIRECTORY_ENTRIES)
            .map(|index| format!("member-{index}"))
            .collect::<Vec<_>>();
        let references = names.iter().map(String::as_str).collect::<Vec<_>>();
        let error = validate_expected_names(&references)
            .unwrap_err()
            .to_string();
        assert!(error.contains("4096-entry limit"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn final_directory_identity_check_rejects_a_late_persistent_swap() {
        let parent = tempfile::tempdir().unwrap();
        let published = parent.path().join("published");
        let replacement = parent.path().join("replacement");
        let parked = parent.path().join("parked");
        fs::create_dir(&published).unwrap();
        fs::create_dir(&replacement).unwrap();
        fs::write(published.join("state.json"), b"authenticated").unwrap();
        fs::write(replacement.join("state.json"), b"authenticated").unwrap();
        let snapshot =
            AuthenticatedDirectorySnapshot::capture(&published, &["state.json"], "fixture")
                .unwrap();

        let error = snapshot
            .ensure_still_published_after_children(|| {
                fs::rename(&published, &parked)?;
                fs::rename(&replacement, &published)?;
                Ok(())
            })
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("changed during authenticated load"),
            "{error}"
        );
    }
}
