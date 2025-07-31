//! Race-free execution of content-pinned local programs.
//!
//! Callers verify an opened file descriptor and execute a private
//! materialization of those exact bytes. This prevents a pathname replacement
//! between hashing an evaluator and `exec` from selecting a different file.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use std::fs::{DirBuilder, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
#[cfg(unix)]
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};

use crate::artifact_io::{
    hash_open_file_exact_length, sha256_identity_from_hex, validate_sha256_identity,
};

#[cfg(unix)]
static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);
#[cfg(unix)]
const MAX_SHEBANG_BYTES: usize = 4 * 1024;

/// An executable selected by the digest of an opened file generation.
#[cfg(unix)]
pub(crate) struct PinnedExecutable {
    staged: StagedExecutable,
}

#[cfg(unix)]
impl PinnedExecutable {
    /// Open and verify `path`, then materialize the verified generation in a
    /// private staging directory for execution.
    pub(crate) fn open(
        path: &Path,
        expected_sha256: &str,
        label: &str,
        namespace: &str,
    ) -> Result<Self> {
        let (mut file, snapshot) = open_verified_executable(path, expected_sha256, label)?;
        let staged =
            stage_opened_executable(&mut file, &snapshot, expected_sha256, label, namespace)?;
        Ok(Self { staged })
    }

    /// Verify the current path generation without preparing it for execution.
    pub(crate) fn verify(path: &Path, expected_sha256: &str, label: &str) -> Result<()> {
        let _ = open_verified_executable(path, expected_sha256, label)?;
        Ok(())
    }

    pub(crate) fn command(&self) -> Command {
        Command::new(&self.staged.path)
    }

    pub(crate) fn into_staged(self) -> StagedExecutable {
        self.staged
    }

    #[cfg(test)]
    pub(crate) fn staged_path(&self) -> &Path {
        &self.staged.path
    }

    #[cfg(test)]
    pub(crate) fn staging_directory(&self) -> &Path {
        &self.staged.directory
    }
}

/// Keeps the private executable pathname alive for a spawned process.
///
/// This is particularly important for shebang scripts: their interpreter may
/// reopen the pathname while the child is starting.
#[cfg(unix)]
pub(crate) struct StagedExecutable {
    directory: PathBuf,
    path: PathBuf,
}

#[cfg(unix)]
impl Drop for StagedExecutable {
    fn drop(&mut self) {
        if let Ok(metadata) = std::fs::metadata(&self.directory) {
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o700);
            let _ = std::fs::set_permissions(&self.directory, permissions);
        }
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

#[cfg(unix)]
fn open_verified_executable(
    path: &Path,
    expected_sha256: &str,
    label: &str,
) -> Result<(File, OpenedFileSnapshot)> {
    validate_sha256_identity(expected_sha256, &format!("{label} identity"))?;
    let path_metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("{label} {} is unavailable", path.display()))?;
    ensure!(
        path_metadata.file_type().is_file() && !path_metadata.file_type().is_symlink(),
        "{label} {} must be a regular non-symlink file",
        path.display()
    );
    ensure!(
        path_metadata.permissions().mode() & 0o111 != 0,
        "{label} {} is not executable",
        path.display()
    );
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("failed to open {label} {}", path.display()))?;
    let opened_metadata = file.metadata()?;
    ensure!(
        opened_metadata.file_type().is_file(),
        "{label} {} must be a regular file",
        path.display()
    );
    ensure!(
        path_metadata.dev() == opened_metadata.dev()
            && path_metadata.ino() == opened_metadata.ino(),
        "{label} {} changed identity while it was opened",
        path.display()
    );
    let snapshot = OpenedFileSnapshot::from_metadata(&opened_metadata, label)?;
    ensure!(
        hash_opened_file(&mut file, &snapshot, label)? == expected_sha256,
        "{label} {} does not match its pinned SHA-256",
        path.display()
    );
    validate_shebang_interpreter(&mut file, &snapshot, label)?;
    Ok((file, snapshot))
}

/// Validate the execution contract imposed by worker `env_clear()` calls.
/// Binary executables have no shebang and pass through unchanged. Scripts must
/// name one absolute interpreter directly; indirection through an `env`
/// executable is rejected because it requires an ambient PATH to resolve the
/// actual interpreter.
#[cfg(unix)]
fn validate_shebang_interpreter(
    file: &mut File,
    snapshot: &OpenedFileSnapshot,
    label: &str,
) -> Result<()> {
    file.seek(SeekFrom::Start(0))?;
    let prefix_len = usize::try_from(snapshot.len.min(MAX_SHEBANG_BYTES as u64))
        .context("executable shebang length exceeds usize")?;
    let mut prefix = vec![0_u8; prefix_len];
    file.read_exact(&mut prefix)?;
    file.seek(SeekFrom::Start(0))?;
    if !prefix.starts_with(b"#!") {
        return Ok(());
    }
    let newline = prefix.iter().position(|byte| *byte == b'\n');
    ensure!(
        newline.is_some() || snapshot.len <= MAX_SHEBANG_BYTES as u64,
        "{label} shebang exceeds {MAX_SHEBANG_BYTES} bytes"
    );
    let directive = &prefix[2..newline.unwrap_or(prefix.len())];
    let directive = directive
        .strip_suffix(b"\r")
        .unwrap_or(directive)
        .split(|byte| *byte == b' ' || *byte == b'\t')
        .find(|part| !part.is_empty())
        .context("worker script shebang has no interpreter")?;
    ensure!(
        directive.starts_with(b"/") && !directive.contains(&0),
        "{label} script must name an absolute shebang interpreter"
    );
    let basename = directive
        .rsplit(|byte| *byte == b'/')
        .next()
        .unwrap_or_default();
    ensure!(
        basename != b"env",
        "{label} may not use an `env` shebang interpreter; name the absolute interpreter directly"
    );
    snapshot.verify(file, label)?;
    Ok(())
}

#[cfg(unix)]
fn stage_opened_executable(
    file: &mut File,
    source_snapshot: &OpenedFileSnapshot,
    expected_sha256: &str,
    label: &str,
    namespace: &str,
) -> Result<StagedExecutable> {
    ensure!(
        !namespace.is_empty()
            && namespace
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "pinned executable namespace is invalid"
    );
    let temporary_root = std::env::temp_dir();
    let invocation = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut directory = None;
    for _ in 0..128 {
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = temporary_root.join(format!(
            ".hermes-{namespace}-{}-{invocation}-{sequence}",
            std::process::id(),
        ));
        match DirBuilder::new().mode(0o700).create(&candidate) {
            Ok(()) => {
                directory = Some(candidate);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "creating private {label} directory beneath {}",
                        temporary_root.display()
                    )
                });
            }
        }
    }
    let directory = directory.context("could not allocate a private executable path")?;
    let staged = StagedExecutable {
        path: directory.join("executable"),
        directory,
    };
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o500)
        .open(&staged.path)
        .with_context(|| format!("creating private {label} executable"))?;
    copy_opened_file_exact(file, source_snapshot, &mut output, label)?;
    output
        .sync_all()
        .with_context(|| format!("syncing pinned {label}"))?;
    drop(output);
    let mut staged_file = File::open(&staged.path)?;
    let staged_snapshot = OpenedFileSnapshot::capture(&staged_file, label)?;
    ensure!(
        staged_snapshot.len == source_snapshot.len,
        "materialized {label} has the wrong length"
    );
    ensure!(
        hash_opened_file(&mut staged_file, &staged_snapshot, label)? == expected_sha256,
        "{label} changed while it was materialized"
    );

    // Remove directory write permission once the staged file is complete. It
    // is restored by `Drop` solely to remove the private materialization.
    let mut permissions = std::fs::metadata(&staged.directory)?.permissions();
    permissions.set_mode(0o500);
    std::fs::set_permissions(&staged.directory, permissions)
        .with_context(|| format!("sealing private {label} directory"))?;
    Ok(staged)
}

#[derive(Clone, Copy)]
struct OpenedFileSnapshot {
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

impl OpenedFileSnapshot {
    fn capture(file: &File, label: &str) -> Result<Self> {
        Self::from_metadata(&file.metadata()?, label)
    }

    fn from_metadata(metadata: &std::fs::Metadata, label: &str) -> Result<Self> {
        ensure!(
            metadata.file_type().is_file(),
            "{label} is not a regular file"
        );
        Ok(Self {
            len: metadata.len(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            modified_seconds: metadata.mtime(),
            #[cfg(unix)]
            modified_nanoseconds: metadata.mtime_nsec(),
            #[cfg(unix)]
            changed_seconds: metadata.ctime(),
            #[cfg(unix)]
            changed_nanoseconds: metadata.ctime_nsec(),
        })
    }

    fn verify(&self, file: &File, label: &str) -> Result<()> {
        let current = Self::capture(file, label)?;
        ensure!(
            current.len == self.len,
            "{label} changed length while it was open"
        );
        #[cfg(unix)]
        ensure!(
            current.device == self.device
                && current.inode == self.inode
                && current.modified_seconds == self.modified_seconds
                && current.modified_nanoseconds == self.modified_nanoseconds
                && current.changed_seconds == self.changed_seconds
                && current.changed_nanoseconds == self.changed_nanoseconds,
            "{label} changed metadata while it was open"
        );
        Ok(())
    }
}

fn hash_opened_file(file: &mut File, snapshot: &OpenedFileSnapshot, label: &str) -> Result<String> {
    let digest = hash_open_file_exact_length(file, snapshot.len, label)?;
    snapshot.verify(file, label)?;
    file.seek(SeekFrom::Start(0))?;
    sha256_identity_from_hex(&digest, &format!("{label} digest"))
}

fn copy_opened_file_exact(
    file: &mut File,
    snapshot: &OpenedFileSnapshot,
    output: &mut File,
    label: &str,
) -> Result<()> {
    file.seek(SeekFrom::Start(0))?;
    let mut buffer = [0_u8; 1024 * 1024];
    let mut remaining = snapshot.len;
    while remaining > 0 {
        let limit = usize::try_from(remaining.min(buffer.len() as u64))
            .context("opened executable copy size exceeds usize")?;
        let read = file.read(&mut buffer[..limit])?;
        ensure!(read != 0, "{label} was truncated while it was materialized");
        output
            .write_all(&buffer[..read])
            .with_context(|| format!("materializing pinned {label}"))?;
        remaining -= read as u64;
    }
    let mut trailing = [0_u8; 1];
    ensure!(
        file.read(&mut trailing)? == 0,
        "{label} grew while it was materialized"
    );
    snapshot.verify(file, label)?;
    file.seek(SeekFrom::Start(0))?;
    Ok(())
}

pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to hash {}", path.display()))?;
    let label = format!("file {}", path.display());
    let snapshot = OpenedFileSnapshot::capture(&file, &label)?;
    hash_opened_file(&mut file, &snapshot, &label)
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use std::process::Stdio;

    #[cfg(unix)]
    use super::*;

    #[cfg(unix)]
    fn executable(root: &Path, name: &str, source: &str) -> PathBuf {
        let path = root.join(name);
        std::fs::write(&path, source).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn opened_generation_survives_adversarial_path_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let worker = executable(
            directory.path(),
            "worker.sh",
            "#!/bin/sh\nprintf 'pinned\\n'\n",
        );
        let hash = file_sha256(&worker).unwrap();
        let pinned =
            PinnedExecutable::open(&worker, &hash, "test evaluator", "test-evaluator").unwrap();

        let replacement = executable(
            directory.path(),
            "replacement.sh",
            "#!/bin/sh\nprintf 'replaced\\n'\n",
        );
        std::fs::rename(&replacement, &worker).unwrap();

        let mut command = pinned.command();
        command.stdout(Stdio::piped());
        let output = command.output().unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"pinned\n");
    }

    #[cfg(unix)]
    #[test]
    fn shebang_interpreter_runs_and_private_materialization_is_cleaned_up() {
        let directory = tempfile::tempdir().unwrap();
        let worker = executable(
            directory.path(),
            "worker.sh",
            "#!/bin/sh\nprintf 'interpreter-ok\\n'\n",
        );
        let hash = file_sha256(&worker).unwrap();
        let pinned =
            PinnedExecutable::open(&worker, &hash, "test evaluator", "test-evaluator").unwrap();
        let staged_path = pinned.staged_path().to_owned();
        let staging_directory = pinned.staging_directory().to_owned();

        let output = pinned.command().output().unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"interpreter-ok\n");
        assert!(staged_path.is_file());
        drop(pinned);
        assert!(!staged_path.exists());
        assert!(!staging_directory.exists());
    }

    #[cfg(unix)]
    #[test]
    fn shebang_requires_a_direct_absolute_interpreter_without_env() {
        let directory = tempfile::tempdir().unwrap();
        for (name, source, expected) in [
            (
                "env.sh",
                "#!/usr/bin/env sh\nexit 0\n",
                "may not use an `env` shebang interpreter",
            ),
            (
                "relative.sh",
                "#!bin/sh\nexit 0\n",
                "must name an absolute shebang interpreter",
            ),
        ] {
            let worker = executable(directory.path(), name, source);
            let hash = file_sha256(&worker).unwrap();
            let error = PinnedExecutable::verify(&worker, &hash, "test worker")
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn stable_hash_rejects_growth_and_truncation_after_open() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mutable");

        std::fs::write(&path, b"original").unwrap();
        let mut grown = File::open(&path).unwrap();
        let grown_snapshot = OpenedFileSnapshot::capture(&grown, "growth test").unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"-growth")
            .unwrap();
        let error = hash_opened_file(&mut grown, &grown_snapshot, "growth test")
            .unwrap_err()
            .to_string();
        assert!(error.contains("grew"), "{error}");

        std::fs::write(&path, b"original").unwrap();
        let mut truncated = File::open(&path).unwrap();
        let truncated_snapshot =
            OpenedFileSnapshot::capture(&truncated, "truncation test").unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1)
            .unwrap();
        let error = hash_opened_file(&mut truncated, &truncated_snapshot, "truncation test")
            .unwrap_err()
            .to_string();
        assert!(error.contains("truncated"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn materialization_rejects_growth_beyond_the_opened_length() {
        let directory = tempfile::tempdir().unwrap();
        let worker = executable(directory.path(), "worker.sh", "#!/bin/sh\nexit 0\n");
        let expected_sha256 = file_sha256(&worker).unwrap();
        let mut source = File::open(&worker).unwrap();
        let source_snapshot = OpenedFileSnapshot::capture(&source, "growth test").unwrap();
        OpenOptions::new()
            .append(true)
            .open(&worker)
            .unwrap()
            .write_all(b"# mutated\n")
            .unwrap();

        let error = stage_opened_executable(
            &mut source,
            &source_snapshot,
            &expected_sha256,
            "growth test",
            "growth-test",
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("grew"), "{error}");
    }
}
