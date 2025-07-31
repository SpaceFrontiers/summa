#!/usr/bin/env bash
# Supervise a hermes-train run across process failures and machine reboots.
#
# The single argument is a trusted Bash configuration file. See
# relaunch.conf.example for the supported settings.

set -Eeuo pipefail
umask 077

RELAUNCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly RELAUNCH_SCRIPT_DIR
readonly CURRENT_POINTER=current.json
readonly GENERATIONS_DIRECTORY=generations
readonly GENERATION_MANIFEST=generation-manifest.json
readonly ARTIFACTS_DIRECTORY=checkpoint-artifacts
readonly ARTIFACT_MANIFEST=artifact-manifest.json
readonly ARTIFACT_OBJECTS_DIRECTORY=checkpoint-objects/sha256
readonly CHECKPOINT_METRICS_DIRECTORY=checkpoint-metrics
readonly -a OBSOLETE_FLAT_CHECKPOINT_FILES=(
  weights.safetensors
  adamw-state.bpk
  muon-state.bpk
  training-state.json
  training-accounting.json
)

log() {
  printf '%s hermes-train-relaunch: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

die() {
  log "error: $*"
  exit 1
}

usage() {
  cat >&2 <<'EOF'
Usage: relaunch.sh <run.conf>

The configuration file must define:
  HERMES_TRAIN_OUTPUT=/path/to/checkpoint
  HERMES_TRAIN_COMMAND=(/path/to/hermes-train train ...)

The supervisor appends --output and, when a complete checkpoint exists,
--resume. See relaunch.conf.example for cloud sync, W&B, and retry settings.
EOF
}

[[ $# -eq 1 ]] || {
  usage
  exit 2
}

readonly RELAUNCH_CONFIG=$1
[[ -r "$RELAUNCH_CONFIG" ]] || die "configuration is not readable: $RELAUNCH_CONFIG"

# The configuration is trusted shell syntax so the training command can be a
# real Bash array without lossy string splitting or eval.
# shellcheck source=/dev/null
source "$RELAUNCH_CONFIG"

: "${HERMES_TRAIN_OUTPUT:?set HERMES_TRAIN_OUTPUT in the configuration}"
declare -p HERMES_TRAIN_COMMAND >/dev/null 2>&1 \
  || die "set HERMES_TRAIN_COMMAND as a Bash array in the configuration"
[[ $(declare -p HERMES_TRAIN_COMMAND) == "declare -a"* ]] \
  || die "HERMES_TRAIN_COMMAND must be a Bash array"
(( ${#HERMES_TRAIN_COMMAND[@]} > 0 )) || die "HERMES_TRAIN_COMMAND is empty"

readonly OUTPUT=${HERMES_TRAIN_OUTPUT%/}
readonly REMOTE=${HERMES_TRAIN_REMOTE_URL:-}
readonly STATE_DIR=${HERMES_TRAIN_STATE_DIR:-"$OUTPUT/.relaunch"}
readonly TRAIN_LOG=${HERMES_TRAIN_LOG:-"$STATE_DIR/train.log"}
readonly SYNC_LOG=${HERMES_TRAIN_SYNC_LOG:-"$STATE_DIR/sync.log"}
readonly WANDB_LOG=${HERMES_TRAIN_WANDB_LOG:-"$STATE_DIR/wandb.log"}
readonly LOCK_FILE=${HERMES_TRAIN_LOCK_FILE:-"$STATE_DIR/lock"}
readonly PYTHON_BIN=${HERMES_TRAIN_PYTHON:-python3}
readonly GCLOUD_BIN=${HERMES_TRAIN_GCLOUD:-gcloud}
readonly SYNC_INTERVAL=${HERMES_TRAIN_SYNC_INTERVAL:-900}
readonly RESTART_DELAY=${HERMES_TRAIN_RESTART_DELAY:-30}
readonly MAX_RESTARTS=${HERMES_TRAIN_MAX_RESTARTS:-0}
readonly WANDB_ENV=${HERMES_TRAIN_WANDB_ENV:-}
readonly WANDB_PYTHON=${HERMES_TRAIN_WANDB_PYTHON:-python3}
readonly WANDB_SCRIPT=${HERMES_TRAIN_WANDB_SCRIPT:-"$RELAUNCH_SCRIPT_DIR/wandb_tail.py"}
readonly WANDB_RESTART_DELAY=${HERMES_TRAIN_WANDB_RESTART_DELAY:-15}
readonly WANDB_FLUSH_DELAY=${HERMES_TRAIN_WANDB_FLUSH_DELAY:-6}
readonly CHECKPOINT_VERIFIER_BIN=${HERMES_TRAIN_CHECKPOINT_VERIFIER_BIN:-${HERMES_TRAIN_COMMAND[0]}}

is_nonnegative_integer() {
  [[ $1 =~ ^[0-9]+$ ]]
}

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

is_positive_integer "$SYNC_INTERVAL" || die "HERMES_TRAIN_SYNC_INTERVAL must be positive"
is_nonnegative_integer "$RESTART_DELAY" || die "HERMES_TRAIN_RESTART_DELAY must be non-negative"
is_nonnegative_integer "$MAX_RESTARTS" || die "HERMES_TRAIN_MAX_RESTARTS must be non-negative"
is_nonnegative_integer "$WANDB_RESTART_DELAY" \
  || die "HERMES_TRAIN_WANDB_RESTART_DELAY must be non-negative"
is_nonnegative_integer "$WANDB_FLUSH_DELAY" \
  || die "HERMES_TRAIN_WANDB_FLUSH_DELAY must be non-negative"

for argument in "${HERMES_TRAIN_COMMAND[@]}"; do
  case "$argument" in
    --resume | --output | --output=* | -o)
      die "leave $argument out of HERMES_TRAIN_COMMAND; the supervisor owns resume and output"
      ;;
  esac
done

if command -v flock >/dev/null 2>&1; then
  readonly LOCK_TOOL=flock
elif command -v shlock >/dev/null 2>&1; then
  readonly LOCK_TOOL=shlock
else
  die "flock (Linux) or shlock (macOS/BSD) is required"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python is required: $PYTHON_BIN"
command -v "${HERMES_TRAIN_COMMAND[0]}" >/dev/null 2>&1 \
  || die "trainer is unavailable: ${HERMES_TRAIN_COMMAND[0]}"
command -v "$CHECKPOINT_VERIFIER_BIN" >/dev/null 2>&1 \
  || die "checkpoint verifier is unavailable: $CHECKPOINT_VERIFIER_BIN"
if [[ -n "$REMOTE" && $REMOTE != file://* ]]; then
  [[ $REMOTE == gs://* ]] || die "remote URL must use gs:// or file://"
  command -v "$GCLOUD_BIN" >/dev/null 2>&1 || die "gcloud is required for $REMOTE"
fi

mkdir -p -- "$OUTPUT" "$STATE_DIR" "$(dirname -- "$TRAIN_LOG")" \
  "$(dirname -- "$SYNC_LOG")" "$(dirname -- "$WANDB_LOG")"

# Children explicitly close fd 9 on Linux, so only this supervisor owns the
# flock. macOS/BSD shlock records the supervisor PID and rejects a live owner.
if [[ $LOCK_TOOL == flock ]]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "another supervisor already owns $LOCK_FILE; nothing to do"
    exit 0
  fi
elif ! shlock -f "$LOCK_FILE" -p "$$"; then
  log "another supervisor already owns $LOCK_FILE; nothing to do"
  exit 0
fi
printf '%s\n' "$$" >"$STATE_DIR/supervisor.pid"

# This helper owns transport envelopes, safe paths, immutable copying, and
# hashing. The trainer's `verify-checkpoint` command remains the authority for
# exact-resume checkpoint contents. Commands print only path-safe,
# whitespace-free descriptors for Bash to consume.
checkpoint_tool() {
  "$PYTHON_BIN" - "$@" <<'PY'
import hashlib
import errno
import fcntl
import json
import os
import secrets
import stat
import sys

LOCAL_POINTER_KEYS = {"version", "generation", "manifest_sha256"}
REMOTE_POINTER_KEYS = {
    "version",
    "generation",
    "manifest_sha256",
    "global_step",
    "metric_records",
    "metrics_path",
    "metrics_bytes",
    "metrics_sha256",
    "artifact_manifest_bytes",
    "artifact_manifest_sha256",
}
FILE_KEYS = {"path", "bytes", "sha256"}
ARTIFACT_ROOT_SPEC_KEYS = {
    "version",
    "sleep_runtime_sha256",
    "dream_initial_policy",
    "roots",
}
ARTIFACT_ROOT_KEYS = {"id", "path"}
PINNED_ARTIFACT_KEYS = {"path", "sha256"}
ARTIFACT_MANIFEST_KEYS = {
    "version",
    "checkpoint_generation",
    "checkpoint_manifest_sha256",
    "sleep_runtime_sha256",
    "dream_initial_policy_sha256",
    "roots",
}
ARTIFACT_MANIFEST_ROOT_KEYS = {"id", "files"}
FIXED_ARTIFACT_ROOTS = (
    ("output.quantized-candidates", "quantized-candidates"),
    ("output.sleep-models", "sleep-models"),
    ("output.sleep-wake-contexts", "sleep-wake-contexts"),
)
SLEEP_ARTIFACT_ROOT_FIELDS = (
    ("sleep.candidates", "candidate_directory"),
    ("sleep.prospective-updates", "prospective_directory"),
    ("sleep.rejections", "rejection_report_directory"),
    ("sleep.tensor-transactions", "tensor_transaction_directory"),
    ("sleep.tier-optimizers", "tier_optimizer_directory"),
)


def fail(message):
    raise SystemExit(message)


def integer(value, label):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > 2**63 - 1
    ):
        fail(f"{label} is not a non-negative integer")
    return value


def version(value, expected, label):
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        fail(f"{label} is not version {expected}")


def digest(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} is not a lowercase SHA-256 digest")
    return value


def safe_path(value):
    if not isinstance(value, str) or not value.strip():
        fail("checkpoint manifest contains an empty path")
    if (
        "\\" in value
        or ":" in value
        or any(character in value for character in "*?[]")
        or value.startswith("/")
    ):
        fail(f"checkpoint path {value!r} is not a safe relative path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        fail(f"checkpoint path {value!r} contains a control character")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        fail(f"checkpoint path {value!r} is not a safe relative path")
    return value


def regular_file(path, label):
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        fail(f"{label} is unavailable: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} is not a regular file")
    return metadata


def real_directory(path, label):
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        fail(f"{label} is unavailable: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label} is not a real directory")


def load_json_file(path, label):
    _, _, raw = read_stable_regular_file(path, label, capture=True)
    return load_unique_json_bytes(raw, label), raw


def parse_pointer(pointer, remote=False):
    expected_keys = REMOTE_POINTER_KEYS if remote else LOCAL_POINTER_KEYS
    if not isinstance(pointer, dict) or set(pointer) != expected_keys:
        fail("checkpoint current pointer has an invalid schema")
    version(pointer["version"], 2 if remote else 1, "checkpoint current pointer")
    generation = pointer["generation"]
    if not isinstance(generation, str) or not generation.startswith("sha256-"):
        fail("checkpoint generation is not content-addressed")
    generation_digest = digest(
        generation[len("sha256-") :], "checkpoint generation digest"
    )
    manifest_digest = digest(
        pointer["manifest_sha256"], "checkpoint manifest digest"
    )
    if generation_digest != manifest_digest:
        fail("checkpoint generation name and manifest digest differ")
    if not remote:
        return generation, manifest_digest

    global_step = integer(pointer["global_step"], "remote pointer global_step")
    metric_records = integer(pointer["metric_records"], "remote pointer metric_records")
    metrics_path = safe_path(pointer["metrics_path"])
    expected_metrics_path = f"checkpoint-metrics/{generation}/metrics.jsonl"
    if metrics_path != expected_metrics_path:
        fail("remote pointer metrics path does not match its checkpoint generation")
    metrics_bytes = integer(pointer["metrics_bytes"], "remote pointer metrics size")
    metrics_sha256 = digest(pointer["metrics_sha256"], "remote pointer metrics digest")
    artifact_bytes = integer(
        pointer["artifact_manifest_bytes"], "remote pointer artifact manifest size"
    )
    artifact_sha256 = digest(
        pointer["artifact_manifest_sha256"],
        "remote pointer artifact manifest digest",
    )
    return (
        generation,
        manifest_digest,
        global_step,
        metric_records,
        metrics_path,
        metrics_bytes,
        metrics_sha256,
        artifact_bytes,
        artifact_sha256,
    )


def read_pointer(path, remote=False):
    pointer, _ = load_json_file(path, "checkpoint current pointer")
    return parse_pointer(pointer, remote=remote)


def read_manifest_transport(path, generation, expected_digest):
    """Extract only safe download paths from a content-addressed manifest.

    The Rust verifier owns the checkpoint schema and authenticates every
    materialized file. This parser deliberately knows only the transport
    envelope needed to fetch those files without path traversal.
    """
    manifest, raw = load_json_file(path, "checkpoint generation manifest")
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        fail("checkpoint generation manifest digest mismatch")
    if generation != "sha256-" + actual_digest:
        fail("checkpoint generation name and manifest digest differ")
    if not isinstance(manifest, dict):
        fail("checkpoint generation manifest is not an object")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        fail("checkpoint manifest files is not an array")
    paths = []
    for entry in entries:
        if not isinstance(entry, dict):
            fail("checkpoint manifest file is not an object")
        path_value = safe_path(entry.get("path"))
        if path_value == "generation-manifest.json":
            fail("checkpoint manifest cannot list itself")
        paths.append(path_value)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        fail("checkpoint manifest file paths are not unique and sorted")
    return paths


def read_stable_regular_file(path, label, capture=False):
    """Read/hash one opened regular file while rejecting path or byte races."""
    metadata = regular_file(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        fail(f"{label} cannot be opened safely: {error}")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            fail(f"{label} is not a regular file")
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(opened) != identity(metadata):
            fail(f"{label} changed while it was opened")
        hasher = hashlib.sha256()
        length = 0
        chunks = [] if capture else None
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            length += len(block)
            hasher.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
        current = regular_file(path, label)
        if identity(opened) != identity(after) or identity(opened) != identity(current):
            fail(f"{label} changed while it was hashed")
        if length != after.st_size:
            fail(f"{label} changed while it was hashed")
        return length, hasher.hexdigest(), None if chunks is None else b"".join(chunks)
    finally:
        os.close(descriptor)


def stable_file_descriptor(path, label):
    length, sha256, _ = read_stable_regular_file(path, label)
    return length, sha256


def load_unique_json_bytes(raw, label):
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                fail(f"{label} repeats JSON field {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is invalid JSON: {error}")


def file_identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def retained_directory_fd(value, label="file remote root"):
    try:
        inherited = int(value)
    except (TypeError, ValueError):
        fail(f"{label} descriptor is invalid")
    try:
        descriptor = os.dup(inherited)
        metadata = os.fstat(descriptor)
    except OSError as error:
        fail(f"{label} descriptor is unavailable: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        fail(f"{label} descriptor is not a directory")
    return descriptor


def directory_open_flags():
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def regular_open_flags():
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def open_child_directory(parent, name, label):
    try:
        descriptor = os.open(name, directory_open_flags(), dir_fd=parent)
    except OSError as error:
        fail(f"{label} is unavailable or unsafe: {error}")
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        fail(f"{label} is not a real directory")
    return descriptor


def open_anchored_parent(root, relative, create=False):
    relative = safe_path(relative)
    parts = relative.split("/")
    current = os.dup(root)
    try:
        for index, component in enumerate(parts[:-1]):
            label = "/".join(parts[: index + 1])
            try:
                child = os.open(
                    component,
                    directory_open_flags(),
                    dir_fd=current,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
                child = open_child_directory(current, component, label)
            except OSError as error:
                fail(f"file remote directory {label!r} is unsafe: {error}")
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                fail(f"file remote directory {label!r} is not a real directory")
            os.close(current)
            current = child
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def ensure_anchored_directory(root, relative):
    parent, leaf = open_anchored_parent(root, relative, create=True)
    try:
        try:
            descriptor = os.open(leaf, directory_open_flags(), dir_fd=parent)
        except FileNotFoundError:
            try:
                os.mkdir(leaf, 0o700, dir_fd=parent)
                os.fsync(parent)
            except FileExistsError:
                pass
            descriptor = open_child_directory(parent, leaf, relative)
        except OSError as error:
            fail(f"file remote directory {relative!r} is unsafe: {error}")
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            fail(f"file remote directory {relative!r} is not a real directory")
        return descriptor
    finally:
        os.close(parent)


def anchored_entry_metadata(root, relative):
    parent, leaf = open_anchored_parent(root, relative)
    try:
        try:
            return os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(parent)


def anchored_entry_kind(root, relative):
    metadata = anchored_entry_metadata(root, relative)
    if metadata is None:
        return "missing"
    if stat.S_ISLNK(metadata.st_mode):
        return "symlink"
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    return "other"


def open_anchored_regular(root, relative, label):
    parent, leaf = open_anchored_parent(root, relative)
    try:
        before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        os.close(parent)
        fail(f"{label} is unavailable")
    if not stat.S_ISREG(before.st_mode):
        os.close(parent)
        fail(f"{label} is not a regular file")
    try:
        descriptor = os.open(leaf, regular_open_flags(), dir_fd=parent)
    except OSError as error:
        os.close(parent)
        fail(f"{label} cannot be opened safely: {error}")
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or file_identity(opened) != file_identity(before):
        os.close(descriptor)
        os.close(parent)
        fail(f"{label} changed while it was opened")
    return parent, leaf, descriptor, opened


def read_anchored_regular(root, relative, label):
    parent, leaf, descriptor, opened = open_anchored_regular(root, relative, label)
    try:
        chunks = []
        hasher = hashlib.sha256()
        length = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            hasher.update(block)
            length += len(block)
        after = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if file_identity(after) != file_identity(opened) or file_identity(current) != file_identity(opened):
            fail(f"{label} changed while it was read")
        if length != after.st_size:
            fail(f"{label} changed while it was read")
        return b"".join(chunks), length, hasher.hexdigest()
    finally:
        os.close(descriptor)
        os.close(parent)


def write_all(descriptor, block):
    offset = 0
    while offset < len(block):
        offset += os.write(descriptor, block[offset:])


def copy_open_regular(source, destination, label):
    hasher = hashlib.sha256()
    length = 0
    while True:
        block = os.read(source, 1024 * 1024)
        if not block:
            break
        write_all(destination, block)
        hasher.update(block)
        length += len(block)
    return length, hasher.hexdigest()


def local_atomic_destination(destination, writer):
    parent_path = os.path.dirname(destination)
    real_directory(parent_path, "local copy destination directory")
    temporary = os.path.join(
        parent_path,
        f".hermes-fd-copy-{os.getpid()}-{secrets.token_hex(8)}.tmp",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        writer(descriptor)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
        parent = os.open(parent_path, directory_open_flags())
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def copy_anchored_out(root, relative, destination):
    label = f"file remote object {relative!r}"
    parent, leaf, source, opened = open_anchored_regular(root, relative, label)
    try:
        observed = None

        def writer(output):
            nonlocal observed
            observed = copy_open_regular(source, output, label)
            after = os.fstat(source)
            current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if file_identity(after) != file_identity(opened) or file_identity(current) != file_identity(opened):
                fail(f"{label} changed while it was copied")
            if observed[0] != after.st_size:
                fail(f"{label} changed while it was copied")

        local_atomic_destination(destination, writer)
        if observed is None:
            fail(f"{label} was not copied")
    finally:
        os.close(source)
        os.close(parent)


def open_stable_source(path, label):
    before = regular_file(path, label)
    try:
        descriptor = os.open(path, regular_open_flags())
    except OSError as error:
        fail(f"{label} cannot be opened safely: {error}")
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or file_identity(opened) != file_identity(before):
        os.close(descriptor)
        fail(f"{label} changed while it was opened")
    return descriptor, opened


def create_anchored_temporary(parent, prefix):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _attempt in range(128):
        name = f".{prefix}-{os.getpid()}-{secrets.token_hex(8)}.tmp"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=parent)
        except FileExistsError:
            continue
    fail("cannot allocate an anchored temporary file")


def hash_anchored_leaf(parent, leaf, label):
    before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not a regular file")
    descriptor = os.open(leaf, regular_open_flags(), dir_fd=parent)
    try:
        opened = os.fstat(descriptor)
        if file_identity(opened) != file_identity(before):
            fail(f"{label} changed while it was opened")
        hasher = hashlib.sha256()
        length = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            length += len(block)
            hasher.update(block)
        after = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if file_identity(after) != file_identity(opened) or file_identity(current) != file_identity(opened):
            fail(f"{label} changed while it was hashed")
        return length, hasher.hexdigest()
    finally:
        os.close(descriptor)


def copy_into_anchored(
    root,
    source_path,
    relative,
    immutable=False,
    expected_bytes=None,
    expected_sha256=None,
):
    label = f"file remote upload source {source_path!r}"
    source, opened = open_stable_source(source_path, label)
    parent, leaf = open_anchored_parent(root, relative, create=True)
    temporary = None
    temporary_descriptor = None
    try:
        temporary, temporary_descriptor = create_anchored_temporary(parent, "upload")
        observed_bytes, observed_sha256 = copy_open_regular(source, temporary_descriptor, label)
        after = os.fstat(source)
        current = regular_file(source_path, label)
        if file_identity(after) != file_identity(opened) or file_identity(current) != file_identity(opened):
            fail(f"{label} changed while it was copied")
        if observed_bytes != after.st_size:
            fail(f"{label} changed while it was copied")
        if expected_bytes is not None and (
            observed_bytes != expected_bytes or observed_sha256 != expected_sha256
        ):
            fail("file remote upload source differs from its expected size or digest")
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None

        if immutable:
            try:
                existing = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISREG(existing.st_mode):
                    fail(f"immutable file remote object {relative!r} is not a regular file")
                length, digest_value = hash_anchored_leaf(
                    parent, leaf, f"immutable file remote object {relative!r}"
                )
                if length != expected_bytes or digest_value != expected_sha256:
                    fail(f"immutable file remote object {relative!r} contains different bytes")
                return
            try:
                os.link(
                    temporary,
                    leaf,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                length, digest_value = hash_anchored_leaf(
                    parent, leaf, f"immutable file remote object {relative!r}"
                )
                if length != expected_bytes or digest_value != expected_sha256:
                    fail(f"immutable file remote object {relative!r} raced with different bytes")
        else:
            os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            temporary = None
        os.fsync(parent)
    finally:
        os.close(source)
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent)
            except OSError:
                pass
        os.close(parent)


def remove_anchored_entry_at(parent, name):
    try:
        metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        directory = open_child_directory(parent, name, f"removal directory {name!r}")
        try:
            for child in os.listdir(directory):
                remove_anchored_entry_at(directory, child)
        finally:
            os.close(directory)
        os.rmdir(name, dir_fd=parent)
    else:
        os.unlink(name, dir_fd=parent)


def remove_anchored_entry(root, relative):
    parent, leaf = open_anchored_parent(root, relative)
    try:
        remove_anchored_entry_at(parent, leaf)
        os.fsync(parent)
    finally:
        os.close(parent)


def copy_regular_between_directories(source_parent, destination_parent, name, label):
    before = os.stat(name, dir_fd=source_parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not a regular file")
    source = os.open(name, regular_open_flags(), dir_fd=source_parent)
    destination = None
    try:
        opened = os.fstat(source)
        if file_identity(opened) != file_identity(before):
            fail(f"{label} changed while it was opened")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        destination = os.open(name, flags, 0o600, dir_fd=destination_parent)
        observed_bytes, _ = copy_open_regular(source, destination, label)
        os.fsync(destination)
        after = os.fstat(source)
        current = os.stat(name, dir_fd=source_parent, follow_symlinks=False)
        if file_identity(after) != file_identity(opened) or file_identity(current) != file_identity(opened):
            fail(f"{label} changed while it was copied")
        if observed_bytes != after.st_size:
            fail(f"{label} changed while it was copied")
    finally:
        os.close(source)
        if destination is not None:
            os.close(destination)


def copy_directory_tree(source, destination, label):
    opened = os.fstat(source)
    if not stat.S_ISDIR(opened.st_mode):
        fail(f"{label} is not a directory")
    names = sorted(os.listdir(source))
    if not names:
        fail(f"{label} contains an empty directory")
    for name in names:
        safe_path(name)
        metadata = os.stat(name, dir_fd=source, follow_symlinks=False)
        child_label = f"{label}/{name}"
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            os.mkdir(name, 0o700, dir_fd=destination)
            source_child = open_child_directory(source, name, child_label)
            destination_child = open_child_directory(destination, name, child_label)
            try:
                copy_directory_tree(source_child, destination_child, child_label)
                os.fsync(destination_child)
            finally:
                os.close(source_child)
                os.close(destination_child)
        elif stat.S_ISREG(metadata.st_mode):
            copy_regular_between_directories(source, destination, name, child_label)
        else:
            fail(f"{child_label} is not a regular file or real directory")
    after = os.fstat(source)
    if file_identity(after) != file_identity(opened):
        fail(f"{label} changed while it was copied")


def stage_anchored_tree(root, source_path, parent_relative):
    source_metadata = os.stat(source_path, follow_symlinks=False)
    if not stat.S_ISDIR(source_metadata.st_mode):
        fail("file remote tree source is not a real directory")
    source = os.open(source_path, directory_open_flags())
    opened_source = os.fstat(source)
    if file_identity(opened_source) != file_identity(source_metadata):
        os.close(source)
        fail("file remote tree source changed while it was opened")
    parent = ensure_anchored_directory(root, parent_relative)
    staging = None
    staging_descriptor = None
    try:
        for _attempt in range(128):
            candidate = f".upload-{os.getpid()}-{secrets.token_hex(8)}"
            try:
                os.mkdir(candidate, 0o700, dir_fd=parent)
                staging = candidate
                break
            except FileExistsError:
                continue
        if staging is None:
            fail("cannot allocate file remote generation staging directory")
        staging_descriptor = open_child_directory(parent, staging, "generation staging")
        copy_directory_tree(source, staging_descriptor, "checkpoint generation source")
        os.fsync(staging_descriptor)
        current_source = os.stat(source_path, follow_symlinks=False)
        if file_identity(current_source) != file_identity(opened_source):
            fail("file remote tree source changed while it was copied")
        os.fsync(parent)
        return f"{safe_path(parent_relative)}/{staging}"
    except BaseException:
        if staging is not None:
            remove_anchored_entry_at(parent, staging)
            os.fsync(parent)
        raise
    finally:
        os.close(source)
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(parent)


def rename_anchored_tree(root, source_relative, destination_relative, replace=False):
    source_parent, source_leaf = open_anchored_parent(root, source_relative)
    destination_parent, destination_leaf = open_anchored_parent(
        root, destination_relative, create=True
    )
    quarantine = None
    try:
        source_metadata = os.stat(
            source_leaf, dir_fd=source_parent, follow_symlinks=False
        )
        if not stat.S_ISDIR(source_metadata.st_mode):
            fail("file remote staged generation is not a real directory")
        destination_metadata = None
        try:
            destination_metadata = os.stat(
                destination_leaf,
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        if destination_metadata is not None:
            if not replace:
                fail("file remote generation destination already exists")
            quarantine = f".corrupt-{os.getpid()}-{secrets.token_hex(8)}"
            os.rename(
                destination_leaf,
                quarantine,
                src_dir_fd=destination_parent,
                dst_dir_fd=destination_parent,
            )
        try:
            os.rename(
                source_leaf,
                destination_leaf,
                src_dir_fd=source_parent,
                dst_dir_fd=destination_parent,
            )
        except BaseException:
            if quarantine is not None:
                os.rename(
                    quarantine,
                    destination_leaf,
                    src_dir_fd=destination_parent,
                    dst_dir_fd=destination_parent,
                )
                quarantine = None
            raise
        os.fsync(source_parent)
        if destination_parent != source_parent:
            os.fsync(destination_parent)
        if quarantine is not None:
            remove_anchored_entry_at(destination_parent, quarantine)
            quarantine = None
            os.fsync(destination_parent)
    finally:
        if quarantine is not None:
            try:
                remove_anchored_entry_at(destination_parent, quarantine)
            except BaseException:
                pass
        os.close(source_parent)
        os.close(destination_parent)


def verify_retained_storage_root(root, path):
    opened = os.fstat(root)
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or file_identity(opened) != file_identity(current):
        fail("file remote root changed while its retained descriptor was opened")


def read_anchored_json(root, relative, label):
    raw, _, _ = read_anchored_regular(root, relative, label)
    return load_unique_json_bytes(raw, label), raw


def compare_remote_pointer_values(candidate, existing):
    parse_pointer(candidate, remote=True)
    parse_pointer(existing, remote=True)
    candidate_step = integer(candidate["global_step"], "candidate remote step")
    existing_step = integer(existing["global_step"], "existing remote step")
    if existing_step > candidate_step:
        return "newer"
    if existing_step == candidate_step:
        if existing["generation"] != candidate["generation"]:
            fail(
                "equal-step remote checkpoint fork: "
                f"{existing['generation']} versus {candidate['generation']}"
            )
        if existing != candidate:
            fail("same-generation remote release envelope differs from candidate")
        return "same"
    return "advance"


def publish_anchored_remote_pointer(root, candidate_path):
    candidate, _ = load_json_file(candidate_path, "candidate remote pointer")
    parse_pointer(candidate, remote=True)
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock = os.open(".hermes-current.lock", lock_flags, 0o600, dir_fd=root)
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            fail("file remote pointer lock is not a regular file")
        fcntl.flock(lock, fcntl.LOCK_EX)
        kind = anchored_entry_kind(root, "current.json")
        if kind != "missing":
            if kind != "file":
                fail("file remote current pointer is not a regular file")
            existing, _ = read_anchored_json(
                root, "current.json", "existing file remote pointer"
            )
            state = compare_remote_pointer_values(candidate, existing)
            if state in ("newer", "same"):
                return state
        copy_into_anchored(root, candidate_path, "current.json")
        published, _ = read_anchored_json(
            root, "current.json", "published file remote pointer"
        )
        if published != candidate:
            fail("published file remote pointer differs from its candidate")
        return "published"
    finally:
        os.close(lock)


def sha256_reference(value, label):
    if not isinstance(value, str) or not value.startswith("sha256:"):
        fail(f"{label} is not a sha256:<digest> reference")
    return digest(value[len("sha256:") :], label)


def safe_root_id(value):
    if (
        not isinstance(value, str)
        or not value
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in value)
    ):
        fail(f"generated-artifact root id {value!r} is not portable")
    return value


def clean_absolute_path(value, label):
    if not isinstance(value, str) or not value:
        fail(f"{label} is empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        fail(f"{label} contains a control character")
    absolute = os.path.abspath(value)
    if os.path.lexists(absolute):
        metadata = os.lstat(absolute)
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} is a symbolic link")
    # Resolve trusted platform aliases in existing parents (for example
    # /var -> /private/var on macOS), then use only the physical path below.
    return os.path.realpath(absolute)


def command_option(arguments, name):
    values = []
    index = 0
    prefix = name + "="
    while index < len(arguments):
        argument = arguments[index]
        if argument == name:
            if index + 1 >= len(arguments):
                fail(f"{name} has no value")
            values.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
        index += 1
    if len(values) > 1:
        fail(f"{name} is repeated in HERMES_TRAIN_COMMAND")
    return values[0] if values else None


def make_artifact_root_spec(output, trainer_arguments):
    output = clean_absolute_path(output, "trainer output root")
    real_directory(output, "trainer output root")
    roots = [
        {"id": root_id, "path": os.path.join(output, relative)}
        for root_id, relative in FIXED_ARTIFACT_ROOTS
    ]

    runtime_path = command_option(trainer_arguments, "--sleep-runtime")
    runtime_reference = command_option(
        trainer_arguments, "--sleep-runtime-sha256"
    )
    if (runtime_path is None) != (runtime_reference is None):
        fail(
            "--sleep-runtime and --sleep-runtime-sha256 must both be present "
            "in HERMES_TRAIN_COMMAND"
        )
    runtime_digest = None
    dream_initial_policy = None
    if runtime_path is not None:
        runtime_path = clean_absolute_path(runtime_path, "sleep runtime configuration")
        runtime_bytes, observed = stable_file_descriptor(
            runtime_path, "sleep runtime configuration"
        )
        del runtime_bytes
        expected = sha256_reference(
            runtime_reference, "sleep runtime configuration digest"
        )
        if observed != expected:
            fail(
                "sleep runtime configuration digest mismatch: "
                f"expected {expected}, observed {observed}"
            )
        runtime_digest = expected
        with open(runtime_path, "rb") as handle:
            raw_runtime = handle.read()
        if hashlib.sha256(raw_runtime).hexdigest() != expected:
            fail("sleep runtime configuration changed after verification")
        runtime = load_unique_json_bytes(raw_runtime, "sleep runtime configuration")
        if not isinstance(runtime, dict):
            fail("sleep runtime configuration is not an object")
        runtime_base = os.path.dirname(runtime_path)
        for root_id, field in SLEEP_ARTIFACT_ROOT_FIELDS:
            value = runtime.get(field)
            if not isinstance(value, str) or not value:
                fail(f"sleep runtime field {field!r} is not a path")
            path = value if os.path.isabs(value) else os.path.join(runtime_base, value)
            roots.append(
                {
                    "id": root_id,
                    "path": clean_absolute_path(path, f"sleep runtime field {field!r}"),
                }
            )
        dreaming = runtime.get("dreaming")
        if dreaming is not None:
            if not isinstance(dreaming, dict):
                fail("sleep runtime dreaming configuration is not an object")
            value = dreaming.get("artifact_directory")
            if not isinstance(value, str) or not value:
                fail("sleep runtime dreaming artifact_directory is not a path")
            path = value if os.path.isabs(value) else os.path.join(runtime_base, value)
            roots.append(
                {
                    "id": "sleep.dreams",
                    "path": clean_absolute_path(
                        path, "sleep runtime dreaming artifact_directory"
                    ),
                }
            )
            initial_policy = dreaming.get("initial_policy")
            if initial_policy is not None:
                if (
                    not isinstance(initial_policy, dict)
                    or set(initial_policy) != PINNED_ARTIFACT_KEYS
                ):
                    fail("sleep runtime dreaming initial_policy has an invalid schema")
                initial_path = initial_policy["path"]
                if not isinstance(initial_path, str) or not initial_path:
                    fail("sleep runtime dreaming initial_policy path is empty")
                initial_path = (
                    initial_path
                    if os.path.isabs(initial_path)
                    else os.path.join(runtime_base, initial_path)
                )
                initial_path = clean_absolute_path(
                    initial_path, "sleep runtime dreaming initial_policy"
                )
                expected_initial = sha256_reference(
                    initial_policy["sha256"],
                    "sleep runtime dreaming initial_policy digest",
                )
                _, observed_initial = stable_file_descriptor(
                    initial_path, "sleep runtime dreaming initial_policy"
                )
                if observed_initial != expected_initial:
                    fail("sleep runtime dreaming initial_policy digest mismatch")
                dream_initial_policy = {
                    "path": initial_path,
                    "sha256": expected_initial,
                }

    roots.sort(key=lambda item: item["id"])
    ids = [safe_root_id(item["id"]) for item in roots]
    paths = [item["path"] for item in roots]
    if len(ids) != len(set(ids)):
        fail("generated-artifact root ids are not unique")
    if len(paths) != len(set(paths)):
        fail("generated-artifact roots alias one another")
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            try:
                common = os.path.commonpath((path, other))
            except ValueError:
                continue
            if common in (path, other):
                fail(
                    "generated-artifact roots overlap: "
                    f"{path!r} and {other!r}"
                )
    return {
        "version": 1,
        "sleep_runtime_sha256": runtime_digest,
        "dream_initial_policy": dream_initial_policy,
        "roots": roots,
    }


def read_artifact_root_spec(path):
    spec, _ = load_json_file(path, "generated-artifact root specification")
    if not isinstance(spec, dict) or set(spec) != ARTIFACT_ROOT_SPEC_KEYS:
        fail("generated-artifact root specification has an invalid schema")
    version(spec["version"], 1, "generated-artifact root specification")
    runtime_digest = spec["sleep_runtime_sha256"]
    if runtime_digest is not None:
        digest(runtime_digest, "generated-artifact sleep runtime digest")
    initial_policy = spec["dream_initial_policy"]
    if initial_policy is not None:
        if (
            not isinstance(initial_policy, dict)
            or set(initial_policy) != PINNED_ARTIFACT_KEYS
        ):
            fail("generated-artifact initial policy has an invalid schema")
        initial_path = initial_policy["path"]
        if (
            not isinstance(initial_path, str)
            or not os.path.isabs(initial_path)
            or os.path.normpath(initial_path) != initial_path
            or any(ord(character) < 32 or ord(character) == 127 for character in initial_path)
        ):
            fail("generated-artifact initial policy has an invalid path")
        digest(initial_policy["sha256"], "generated-artifact initial policy digest")
    roots = spec["roots"]
    if not isinstance(roots, list):
        fail("generated-artifact roots is not an array")
    ids = []
    paths = []
    for root in roots:
        if not isinstance(root, dict) or set(root) != ARTIFACT_ROOT_KEYS:
            fail("generated-artifact root has an invalid schema")
        root_id = safe_root_id(root["id"])
        path_value = root["path"]
        if (
            not isinstance(path_value, str)
            or not os.path.isabs(path_value)
            or any(ord(character) < 32 or ord(character) == 127 for character in path_value)
        ):
            fail(f"generated-artifact root {root_id!r} has an invalid local path")
        if os.path.normpath(path_value) != path_value:
            fail(f"generated-artifact root {root_id!r} is not normalized")
        ids.append(root_id)
        paths.append(path_value)
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        fail("generated-artifact root ids are not unique and sorted")
    fixed_ids = {root_id for root_id, _ in FIXED_ARTIFACT_ROOTS}
    if not fixed_ids.issubset(ids):
        fail("generated-artifact root specification omits a fixed output store")
    if len(paths) != len(set(paths)):
        fail("generated-artifact root specification aliases local paths")
    for index, value in enumerate(paths):
        for other in paths[index + 1 :]:
            if os.path.commonpath((value, other)) in (value, other):
                fail("generated-artifact root specification contains overlapping paths")
    return spec


def hash_open_file(descriptor, label):
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not a regular file")
    hasher = hashlib.sha256()
    length = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        length += len(block)
        hasher.update(block)
    after = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or length != after.st_size
    ):
        fail(f"{label} changed while it was hashed")
    return length, hasher.hexdigest()


def scan_artifact_directory(path, root_id):
    if not os.path.lexists(path):
        return []
    root_metadata = os.lstat(path)
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        fail(f"generated-artifact root {root_id!r} is not a real directory")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    root_descriptor = os.open(path, directory_flags)
    files = []

    def visit(directory_descriptor, prefix):
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as error:
            fail(f"cannot enumerate generated-artifact root {root_id!r}: {error}")
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            safe_path(relative)
            try:
                metadata = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
            except OSError as error:
                fail(f"generated artifact {root_id}:{relative} disappeared: {error}")
            if stat.S_ISLNK(metadata.st_mode):
                fail(f"generated artifact {root_id}:{relative} is a symbolic link")
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(name, directory_flags, dir_fd=directory_descriptor)
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        fail(
                            f"generated-artifact directory {root_id}:{relative} changed"
                        )
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                child = os.open(name, file_flags, dir_fd=directory_descriptor)
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        fail(f"generated artifact {root_id}:{relative} changed")
                    length, observed = hash_open_file(
                        child, f"generated artifact {root_id}:{relative}"
                    )
                finally:
                    os.close(child)
                current = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    fail(f"generated artifact {root_id}:{relative} was replaced")
                files.append(
                    {"path": relative, "bytes": length, "sha256": observed}
                )
            else:
                fail(
                    f"generated artifact {root_id}:{relative} is not a regular file "
                    "or real directory"
                )

    try:
        opened_root = os.fstat(root_descriptor)
        if (opened_root.st_dev, opened_root.st_ino) != (
            root_metadata.st_dev,
            root_metadata.st_ino,
        ):
            fail(f"generated-artifact root {root_id!r} changed while opening")
        visit(root_descriptor, "")
    finally:
        os.close(root_descriptor)
    files.sort(key=lambda item: item["path"])
    return files


def read_artifact_manifest(path, spec, generation, manifest_digest):
    manifest, _ = load_json_file(path, "generated-artifact closure manifest")
    if not isinstance(manifest, dict) or set(manifest) != ARTIFACT_MANIFEST_KEYS:
        fail("generated-artifact closure manifest has an invalid schema")
    version(manifest["version"], 1, "generated-artifact closure manifest")
    digest(manifest_digest, "checkpoint manifest digest")
    if generation != "sha256-" + manifest_digest:
        fail("checkpoint generation and manifest digest differ")
    if (
        manifest["checkpoint_generation"] != generation
        or manifest["checkpoint_manifest_sha256"] != manifest_digest
    ):
        fail("generated-artifact closure belongs to another checkpoint generation")
    if manifest["sleep_runtime_sha256"] != spec["sleep_runtime_sha256"]:
        fail("generated-artifact closure belongs to another sleep runtime")
    expected_initial = spec["dream_initial_policy"]
    expected_initial_sha256 = (
        None if expected_initial is None else expected_initial["sha256"]
    )
    if manifest["dream_initial_policy_sha256"] != expected_initial_sha256:
        fail("generated-artifact closure belongs to another initial Dreaming policy")
    roots = manifest["roots"]
    if not isinstance(roots, list):
        fail("generated-artifact closure roots is not an array")
    expected_ids = [root["id"] for root in spec["roots"]]
    observed_ids = []
    for root in roots:
        if not isinstance(root, dict) or set(root) != ARTIFACT_MANIFEST_ROOT_KEYS:
            fail("generated-artifact closure root has an invalid schema")
        root_id = safe_root_id(root["id"])
        observed_ids.append(root_id)
        entries = root["files"]
        if not isinstance(entries, list):
            fail(f"generated-artifact closure root {root_id!r} files is not an array")
        paths = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != FILE_KEYS:
                fail(f"generated-artifact closure root {root_id!r} has an invalid file")
            relative = safe_path(entry["path"])
            integer(entry["bytes"], f"generated artifact {root_id}:{relative} size")
            digest(entry["sha256"], f"generated artifact {root_id}:{relative} digest")
            paths.append(relative)
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            fail(
                f"generated-artifact closure root {root_id!r} paths are not unique and sorted"
            )
    if observed_ids != expected_ids:
        fail("generated-artifact closure roots differ from the current configuration")
    return manifest


def root_map(spec):
    return {root["id"]: root["path"] for root in spec["roots"]}


def raw_content_digest(value, label):
    if isinstance(value, str) and value.startswith("sha256:"):
        value = value[len("sha256:") :]
    return digest(value, label)


def read_stable_bytes(path, label):
    expected_bytes, expected_digest = stable_file_descriptor(path, label)
    with open(path, "rb") as handle:
        payload = handle.read()
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != expected_digest
    ):
        fail(f"{label} changed after verification")
    return payload


class ArtifactClosureCollector:
    """Select only immutable artifacts referenced by one sealed trainer state."""

    def __init__(self, spec):
        self.spec = spec
        self.roots = root_map(spec)
        self.selected = {root_id: {} for root_id in self.roots}
        self.dream_initial_policy = spec["dream_initial_policy"]

    def _location(self, path):
        if not isinstance(path, str) or not path:
            fail("checkpoint generated-artifact reference has an empty path")
        if any(ord(character) < 32 or ord(character) == 127 for character in path):
            fail("checkpoint generated-artifact reference contains a control character")
        absolute = os.path.abspath(path)
        if os.path.lexists(absolute) and stat.S_ISLNK(os.lstat(absolute).st_mode):
            fail(f"checkpoint generated-artifact reference {path!r} is a symbolic link")
        physical = os.path.realpath(absolute)
        for root_id, root in self.roots.items():
            try:
                if os.path.commonpath((root, physical)) != root:
                    continue
            except ValueError:
                continue
            relative = os.path.relpath(physical, root).replace(os.sep, "/")
            return root_id, safe_path(relative), physical
        return None

    def add(self, path, expected_sha256=None, label="checkpoint artifact"):
        location = self._location(path)
        if location is None:
            # Pinned inputs outside configured generated stores are deployment
            # inputs, not relaunch-owned output. Their existing validators still
            # verify them before training starts.
            return None
        root_id, relative, physical = location
        length, observed = hash_relative_file(self.roots[root_id], relative, label)
        if expected_sha256 is not None:
            expected = raw_content_digest(expected_sha256, f"{label} digest")
            if observed != expected:
                fail(f"{label} does not match its checkpointed digest")
        entry = {"path": relative, "bytes": length, "sha256": observed}
        previous = self.selected[root_id].get(relative)
        if previous is not None and previous != entry:
            fail(f"checkpoint artifact {root_id}:{relative} changed during closure")
        self.selected[root_id][relative] = entry
        return physical

    def add_regular_manifest(self, path, expected_sha256, label):
        physical = self.add(path, expected_sha256, label)
        if physical is None:
            return None
        value = load_unique_json_bytes(read_stable_bytes(physical, label), label)
        if not isinstance(value, dict):
            fail(f"{label} is not an object")
        return physical, value

    def add_file_manifest(self, path, expected_sha256, label):
        loaded = self.add_regular_manifest(path, expected_sha256, label)
        if loaded is None:
            return
        physical, value = loaded
        entries = value.get("files")
        if not isinstance(entries, list):
            fail(f"{label} has no files array")
        parent = os.path.dirname(physical)
        for entry in entries:
            if not isinstance(entry, dict):
                fail(f"{label} has an invalid file entry")
            relative = safe_path(entry.get("path"))
            expected_bytes = integer(entry.get("bytes"), f"{label} member size")
            member = os.path.join(parent, *relative.split("/"))
            added = self.add(member, entry.get("sha256"), f"{label} member {relative}")
            if added is None:
                fail(f"{label} member {relative!r} escapes generated-artifact roots")
            root_id, selected_relative, _ = self._location(member)
            observed = self.selected[root_id][selected_relative]
            if observed["bytes"] != expected_bytes:
                fail(f"{label} member {relative!r} has the wrong size")
        return physical

    def add_quantized_archive_manifest(self, path, expected_sha256, label):
        loaded = self.add_regular_manifest(path, expected_sha256, label)
        if loaded is None:
            return
        physical, value = loaded
        parent = os.path.dirname(physical)
        matrices = value.get("matrices")
        floating = value.get("floating_tensors")
        if not isinstance(matrices, list) or not isinstance(floating, list):
            fail(f"{label} has an invalid archive inventory")
        members = [(item, "packed_bytes") for item in matrices]
        members.extend((item, "bytes") for item in floating)
        for entry, size_key in members:
            if not isinstance(entry, dict):
                fail(f"{label} has an invalid archive member")
            relative = safe_path(entry.get("file"))
            expected_bytes = integer(entry.get(size_key), f"{label} member size")
            member = os.path.join(parent, *relative.split("/"))
            physical_member = self.add(
                member, entry.get("sha256"), f"{label} member {relative}"
            )
            if physical_member is None:
                fail(f"{label} member {relative!r} escapes generated-artifact roots")
            root_id, selected_relative, _ = self._location(member)
            if self.selected[root_id][selected_relative]["bytes"] != expected_bytes:
                fail(f"{label} member {relative!r} has the wrong size")
        return physical

    def add_qat_candidate(self, path, expected_sha256):
        loaded = self.add_regular_manifest(path, expected_sha256, "QAT candidate manifest")
        if loaded is None:
            fail("QAT candidate manifest is outside its configured store")
        physical, value = loaded
        parent = os.path.dirname(physical)
        weights = safe_path(value.get("weights_file"))
        expected_bytes = integer(value.get("weights_bytes"), "QAT weights size")
        weights_path = os.path.join(parent, *weights.split("/"))
        added = self.add(
            weights_path, value.get("weights_sha256"), "QAT candidate weights"
        )
        location = self._location(weights_path)
        if added is None or location is None:
            fail("QAT candidate weights escape generated-artifact roots")
        root_id, relative, _ = location
        if self.selected[root_id][relative]["bytes"] != expected_bytes:
            fail("QAT candidate weights have the wrong size")
        archive_manifest = safe_path(value.get("archive_manifest"))
        if self.add_quantized_archive_manifest(
            os.path.join(parent, *archive_manifest.split("/")),
            value.get("archive_manifest_sha256"),
            "HQUANT archive manifest",
        ) is None:
            fail("HQUANT archive manifest escapes the QAT candidate store")

    def add_dream_manifest(self, root_id, manifest_hash):
        expected = raw_content_digest(manifest_hash, "dream manifest digest")
        path = os.path.join(
            self.roots[root_id], "manifests", f"{expected}.json"
        )
        loaded = self.add_regular_manifest(path, expected, "Dreaming manifest")
        if loaded is None:
            fail("Dreaming manifest is outside its configured store")
        _, value = loaded
        policy = value.get("generation_policy_sha256")
        policy_adapter = value.get("generation_policy_adapter_sha256")
        if policy is not None or policy_adapter is not None:
            if policy is None or policy_adapter is None:
                fail("Dreaming manifest generation policy binding is incomplete")
            self.add_dream_policy_chain(root_id, policy, policy_adapter)
        dreams = value.get("dreams")
        if not isinstance(dreams, list):
            fail("Dreaming manifest has no dreams array")
        for dream in dreams:
            if not isinstance(dream, dict):
                fail("Dreaming manifest has an invalid candidate")
            candidate = raw_content_digest(
                dream.get("artifact_hash"), "dream candidate digest"
            )
            added = self.add(
                os.path.join(
                    self.roots[root_id], "candidates", f"{candidate}.json"
                ),
                candidate,
                "Dreaming candidate",
            )
            if added is None:
                fail("Dreaming candidate escapes its configured store")

    def add_dream_policy_chain(self, root_id, policy_hash, head_adapter_hash=None):
        expected_adapter = head_adapter_hash
        seen = set()
        while policy_hash is not None:
            policy = raw_content_digest(policy_hash, "Dreaming policy digest")
            if policy in seen:
                fail("Dreaming policy parent chain contains a cycle")
            seen.add(policy)
            initial = self.dream_initial_policy
            if initial is not None and initial["sha256"] == policy:
                initial_path = initial["path"]
                payload = read_stable_bytes(initial_path, "deployment-bound Dreaming policy")
                if hashlib.sha256(payload).hexdigest() != policy:
                    fail("deployment-bound Dreaming policy digest mismatch")
                value = load_unique_json_bytes(payload, "deployment-bound Dreaming policy")
                if not isinstance(value, dict):
                    fail("deployment-bound Dreaming policy is not an object")
                # If deployment chose the canonical generated store path, keep
                # the policy itself in the closure as well. Otherwise the pinned
                # runtime digest/path/hash is the explicit deployment contract.
                location = self._location(initial_path)
                if location is not None:
                    loaded = self.add_regular_manifest(
                        initial_path, policy, "Dreaming policy"
                    )
                    _, value = loaded
            else:
                loaded = self.add_regular_manifest(
                    os.path.join(self.roots[root_id], "policies", f"{policy}.json"),
                    policy,
                    "Dreaming policy",
                )
                if loaded is None:
                    fail("Dreaming policy is outside its configured store")
                _, value = loaded
            adapter = raw_content_digest(
                value.get("adapter_sha256"), "Dreaming policy adapter digest"
            )
            if expected_adapter is not None and adapter != raw_content_digest(
                expected_adapter, "Dreaming expected policy adapter digest"
            ):
                fail("Dreaming policy adapter differs from its recorded binding")
            added = self.add(
                os.path.join(
                    self.roots[root_id], "policy-adapters", f"{adapter}.bin"
                ),
                adapter,
                "Dreaming generation-policy adapter",
            )
            if added is None:
                fail("Dreaming generation-policy adapter escapes its configured store")
            accepted_adapters = value.get("accepted_adapters")
            if not isinstance(accepted_adapters, list):
                fail("Dreaming policy accepted adapters is not an array")
            for accepted_hash in accepted_adapters:
                accepted = raw_content_digest(
                    accepted_hash, "Dreaming accepted adapter digest"
                )
                added = self.add(
                    os.path.join(
                        self.roots[root_id], "adapters", f"{accepted}.bin"
                    ),
                    accepted,
                    "Dreaming accepted adapter",
                )
                if added is None:
                    fail("Dreaming accepted adapter escapes its configured store")
            parent = value.get("parent_policy_sha256")
            parent_adapter = value.get("parent_adapter_sha256")
            if (parent is None) != (parent_adapter is None):
                fail("Dreaming policy parent binding is incomplete")
            policy_hash = parent
            expected_adapter = parent_adapter

    def add_dream_transaction(self, transaction):
        if not isinstance(transaction, dict):
            fail("training-state sleep transaction is not an object")
        manifest_hash = transaction.get("generated_manifest")
        if manifest_hash is not None and "sleep.dreams" in self.roots:
            self.add_dream_manifest("sleep.dreams", manifest_hash)
        trials = transaction.get("dream_trials", [])
        if not isinstance(trials, list):
            fail("training-state dream trials is not an array")
        if "sleep.dreams" in self.roots:
            for trial in trials:
                if not isinstance(trial, dict):
                    fail("training-state dream trial is not an object")
                adapter = raw_content_digest(
                    trial.get("adapter_hash"), "Dreaming adapter digest"
                )
                added = self.add(
                    os.path.join(
                        self.roots["sleep.dreams"],
                        "adapters",
                        f"{adapter}.bin",
                    ),
                    adapter,
                    "Dreaming adapter",
                )
                if added is None:
                    fail("Dreaming adapter escapes its configured store")
            policy = transaction.get("dream_policy_receipt")
            if policy is not None:
                self.add_dream_policy_chain("sleep.dreams", policy)

    def add_tensor_transaction(self, transaction):
        if "sleep.tensor-transactions" not in self.roots:
            return
        generation = transaction.get("tensor_transaction_generation")
        manifest_hash = transaction.get("tensor_transaction_manifest_hash")
        if generation is None and manifest_hash is None:
            return
        if not isinstance(generation, str) or manifest_hash is None:
            fail("training-state tensor transaction receipt is incomplete")
        expected = raw_content_digest(manifest_hash, "tensor transaction manifest")
        if generation != "sha256-" + expected:
            fail("tensor transaction generation differs from its manifest digest")
        self.add_file_manifest(
            os.path.join(
                self.roots["sleep.tensor-transactions"],
                "generations",
                generation,
                "manifest.json",
            ),
            expected,
            "tensor transaction manifest",
        )

    def add_model_references(self, transaction):
        for path_key, hash_key, label in [
            ("teacher_checkpoint", "teacher_hash", "sleep teacher checkpoint"),
            ("student_checkpoint", "student_hash", "sleep student checkpoint"),
            ("candidate_checkpoint", "candidate_hash", "sleep candidate checkpoint"),
        ]:
            path = transaction.get(path_key)
            expected = transaction.get(hash_key)
            if path is not None or expected is not None:
                if not isinstance(path, str) or expected is None:
                    fail(f"{label} reference is incomplete")
                self.add(path, expected, label)

    def add_tier_optimizer_artifact(self, artifact):
        if artifact is None:
            return
        if not isinstance(artifact, dict):
            fail("tier optimizer artifact is not an object")
        if self.add_file_manifest(
            artifact.get("state_uri"),
            artifact.get("manifest_hash"),
            "tier optimizer manifest",
        ) is None:
            fail("tier optimizer manifest is outside its configured store")

    def manifest_roots(self):
        return [
            {"id": root_id, "files": list(sorted(files.values(), key=lambda item: item["path"]))}
            for root_id, files in sorted(self.selected.items())
        ]


def build_artifact_closure(generation_path, spec, generation):
    state_path = os.path.join(generation_path, "training-state.json")
    state = load_unique_json_bytes(
        read_stable_bytes(state_path, "sealed training state"), "sealed training state"
    )
    if not isinstance(state, dict) or state.get("version") != 2:
        fail("sealed training state is not version 2")
    collector = ArtifactClosureCollector(spec)

    artifacts = state.get("artifacts", [])
    if not isinstance(artifacts, list):
        fail("training-state artifacts is not an array")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            fail("training-state artifact receipt is not an object")
        path = artifact.get("manifest")
        if artifact.get("kind") == "hquant_candidate":
            collector.add_qat_candidate(path, artifact.get("hash"))
        else:
            collector.add(path, artifact.get("hash"), "checkpoint artifact manifest")

    quantization = state.get("quantization")
    if quantization is not None:
        if not isinstance(quantization, dict):
            fail("training-state quantization state is not an object")
        path = quantization.get("manifest")
        if path is not None and not any(
            isinstance(artifact, dict)
            and artifact.get("kind") == "hquant_candidate"
            and artifact.get("manifest") == path
            for artifact in artifacts
        ):
            fail("quantization manifest has no authenticated artifact receipt")

    sleep = state.get("sleep")
    if sleep is not None:
        if not isinstance(sleep, dict):
            fail("training-state sleep cursor is not an object")
        for checkpoint_name in ("input_checkpoint", "live_checkpoint"):
            checkpoint = sleep.get(checkpoint_name)
            if not isinstance(checkpoint, dict):
                fail(f"sleep {checkpoint_name} is not an object")
            collector.add(
                checkpoint.get("uri"), checkpoint.get("sha256"), f"sleep {checkpoint_name}"
            )
        journal = sleep.get("wake_context_journal")
        if journal is not None:
            if not isinstance(journal, dict):
                fail("sleep wake-context journal is not an object")
            collector.add(
                journal.get("path"), journal.get("sha256"), "sleep wake-context journal"
            )
        sleep_state = sleep.get("sleep")
        if not isinstance(sleep_state, dict):
            fail("training-state sleep state is not an object")
        transactions = []
        pending = sleep_state.get("pending")
        if pending is not None:
            transactions.append(pending)
        completed = sleep_state.get("completed_transactions", [])
        if not isinstance(completed, list):
            fail("completed sleep transactions is not an array")
        transactions.extend(completed)
        for transaction in transactions:
            collector.add_model_references(transaction)
            collector.add_tensor_transaction(transaction)
            collector.add_dream_transaction(transaction)
        artifact_manifests = sleep_state.get("artifact_manifests", [])
        if not isinstance(artifact_manifests, list):
            fail("training-state sleep artifact_manifests is not an array")
        if artifact_manifests and "sleep.dreams" not in collector.roots:
            fail("training-state names Dreaming manifests but runtime has no Dreaming store")
        for manifest_hash in artifact_manifests:
            collector.add_dream_manifest("sleep.dreams", manifest_hash)
        scopes = sleep.get("optimizer_scopes")
        if not isinstance(scopes, dict):
            fail("training-state optimizer scopes is not an object")
        tiers = scopes.get("tiers")
        if not isinstance(tiers, list):
            fail("training-state optimizer tiers is not an array")
        for tier in tiers:
            if not isinstance(tier, dict):
                fail("training-state optimizer tier is not an object")
            collector.add_tier_optimizer_artifact(tier.get("artifact"))

    return collector.manifest_roots()


def hash_relative_file(root, relative, label):
    components = relative.split("/")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(root, directory_flags)
    except OSError as error:
        fail(f"{label} root is unavailable: {error}")
    try:
        for component in components[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        child = os.open(components[-1], file_flags, dir_fd=descriptor)
        try:
            return hash_open_file(child, label)
        finally:
            os.close(child)
    except OSError as error:
        fail(f"{label} is unavailable: {error}")
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def verify_artifact_sources(spec, manifest):
    paths = root_map(spec)
    for root in manifest["roots"]:
        root_id = root["id"]
        source_root = paths[root_id]
        for entry in root["files"]:
            length, observed = hash_relative_file(
                source_root,
                entry["path"],
                f"generated artifact {root_id}:{entry['path']}",
            )
            if length != entry["bytes"] or observed != entry["sha256"]:
                fail(f"generated artifact {root_id}:{entry['path']} changed")


def verify_artifact_snapshot(snapshot_root, manifest):
    declared_ids = {root["id"] for root in manifest["roots"]}
    if not os.path.lexists(snapshot_root):
        fail("generated-artifact snapshot root is unavailable")
    real_directory(snapshot_root, "generated-artifact snapshot root")
    actual_ids = set()
    for name in os.listdir(snapshot_root):
        safe_root_id(name)
        child = os.path.join(snapshot_root, name)
        metadata = os.lstat(child)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            fail(f"generated-artifact snapshot root {name!r} is not a real directory")
        actual_ids.add(name)
    if actual_ids != declared_ids:
        fail("generated-artifact snapshot roots differ from its manifest")
    manifest_roots = {root["id"]: root for root in manifest["roots"]}
    for root_id in sorted(actual_ids):
        observed = scan_artifact_directory(os.path.join(snapshot_root, root_id), root_id)
        if observed != manifest_roots[root_id]["files"]:
            fail(f"generated-artifact snapshot root {root_id!r} differs from its manifest")


def ensure_real_directory_tree(path):
    if not os.path.isabs(path):
        fail(f"restore directory {path!r} is not absolute")
    current = os.path.sep
    for component in [part for part in path.split(os.path.sep) if part]:
        current = os.path.join(current, component)
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            fail(f"restore directory {current!r} is not a real directory")


def install_immutable(source, destination, expected_bytes, expected_digest):
    expected_bytes = integer(expected_bytes, "immutable artifact size")
    expected_digest = digest(expected_digest, "immutable artifact digest")
    source_bytes, source_digest = stable_file_descriptor(source, "immutable source")
    if source_bytes != expected_bytes or source_digest != expected_digest:
        fail("immutable source differs from its manifest")
    parent = os.path.dirname(destination)
    ensure_real_directory_tree(parent)
    if os.path.lexists(destination):
        length, observed = stable_file_descriptor(destination, "immutable destination")
        if length != expected_bytes or observed != expected_digest:
            fail(f"immutable destination {destination!r} contains different bytes")
        return

    temporary = os.path.join(
        parent,
        f".hermes-restore-{os.getpid()}-{secrets.token_hex(8)}.tmp",
    )
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    destination_descriptor = None
    try:
        destination_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
            0o600,
        )
        hasher = hashlib.sha256()
        length = 0
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            offset = 0
            while offset < len(block):
                offset += os.write(destination_descriptor, block[offset:])
            hasher.update(block)
            length += len(block)
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        if length != expected_bytes or hasher.hexdigest() != expected_digest:
            fail("immutable source changed while it was copied")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            existing_bytes, existing_digest = stable_file_descriptor(
                destination, "immutable destination"
            )
            if existing_bytes != expected_bytes or existing_digest != expected_digest:
                fail(f"immutable destination {destination!r} raced with different bytes")
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def restore_artifact_snapshot(spec, manifest, snapshot_root):
    verify_artifact_snapshot(snapshot_root, manifest)
    destinations = root_map(spec)
    for root in manifest["roots"]:
        root_id = root["id"]
        destination_root = destinations[root_id]
        ensure_real_directory_tree(destination_root)
        for entry in root["files"]:
            relative = entry["path"]
            source = os.path.join(snapshot_root, root_id, *relative.split("/"))
            destination = os.path.join(destination_root, *relative.split("/"))
            install_immutable(source, destination, entry["bytes"], entry["sha256"])


def snapshot_metrics(source, committed_records, destination):
    committed_records = integer(committed_records, "committed metric records")
    source_metadata = regular_file(source, "checkpoint metric journal")
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        source_flags |= os.O_NONBLOCK
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        fail(f"checkpoint metric journal cannot be opened safely: {error}")
    opened_source = os.fstat(source_descriptor)
    if (
        not stat.S_ISREG(opened_source.st_mode)
        or (opened_source.st_dev, opened_source.st_ino)
        != (source_metadata.st_dev, source_metadata.st_ino)
    ):
        os.close(source_descriptor)
        fail("checkpoint metric journal changed while it was opened")
    parent = os.path.dirname(destination)
    real_directory(parent, "metric snapshot destination directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    copied = 0
    copied_bytes = 0
    copied_hasher = hashlib.sha256()
    try:
        with os.fdopen(source_descriptor, "rb") as input_file, os.fdopen(
            descriptor, "wb"
        ) as output:
            source_descriptor = None
            descriptor = None
            while copied < committed_records:
                record = input_file.readline()
                if not record or not record.endswith(b"\n"):
                    fail("metric journal has a missing or torn committed record")
                output.write(record)
                copied_bytes += len(record)
                copied_hasher.update(record)
                copied += 1
            # Make the first copy observable/durable before validating it
            # against a second read of the same opened source descriptor.
            output.flush()
            os.fsync(output.fileno())

            input_file.seek(0)
            verified_bytes = 0
            verified_hasher = hashlib.sha256()
            while verified_bytes < copied_bytes:
                block = input_file.read(min(1024 * 1024, copied_bytes - verified_bytes))
                if not block:
                    fail("checkpoint metric journal was truncated while it was copied")
                verified_bytes += len(block)
                verified_hasher.update(block)
            if (
                verified_bytes != copied_bytes
                or verified_hasher.digest() != copied_hasher.digest()
            ):
                fail("checkpoint committed metric prefix changed while it was copied")

            after_source = os.fstat(input_file.fileno())
            current_source = regular_file(source, "checkpoint metric journal")
            opened_identity = (opened_source.st_dev, opened_source.st_ino)
            if opened_identity != (after_source.st_dev, after_source.st_ino) or opened_identity != (
                current_source.st_dev,
                current_source.st_ino,
            ):
                fail("checkpoint metric journal was replaced while it was copied")
            if after_source.st_size < opened_source.st_size:
                fail("checkpoint metric journal was truncated while it was copied")
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if descriptor is not None:
            os.close(descriptor)


command = sys.argv[1]
if command == "artifact-root-spec":
    specification = make_artifact_root_spec(sys.argv[2], sys.argv[3:])
    print(json.dumps(specification, sort_keys=True, separators=(",", ":")))
elif command == "build-artifact-manifest":
    specification = read_artifact_root_spec(sys.argv[2])
    generation = sys.argv[3]
    manifest_digest = digest(sys.argv[4], "checkpoint manifest digest")
    if generation != "sha256-" + manifest_digest:
        fail("checkpoint generation and manifest digest differ")
    closure = {
        "version": 1,
        "checkpoint_generation": generation,
        "checkpoint_manifest_sha256": manifest_digest,
        "sleep_runtime_sha256": specification["sleep_runtime_sha256"],
        "dream_initial_policy_sha256": (
            None
            if specification["dream_initial_policy"] is None
            else specification["dream_initial_policy"]["sha256"]
        ),
        "roots": build_artifact_closure(sys.argv[5], specification, generation),
    }
    print(json.dumps(closure, sort_keys=True, separators=(",", ":")))
elif command == "artifact-plan":
    specification = read_artifact_root_spec(sys.argv[2])
    closure = read_artifact_manifest(
        sys.argv[3], specification, sys.argv[4], sys.argv[5]
    )
    paths = root_map(specification)
    for root in closure["roots"]:
        for entry in root["files"]:
            print(
                root["id"],
                paths[root["id"]],
                entry["path"],
                entry["bytes"],
                entry["sha256"],
                sep="\t",
            )
elif command == "artifact-root-ids":
    specification = read_artifact_root_spec(sys.argv[2])
    closure = read_artifact_manifest(
        sys.argv[3], specification, sys.argv[4], sys.argv[5]
    )
    for root in closure["roots"]:
        print(root["id"])
elif command == "verify-artifact-sources":
    specification = read_artifact_root_spec(sys.argv[2])
    closure = read_artifact_manifest(
        sys.argv[3], specification, sys.argv[4], sys.argv[5]
    )
    verify_artifact_sources(specification, closure)
elif command == "verify-artifact-snapshot":
    specification = read_artifact_root_spec(sys.argv[2])
    closure = read_artifact_manifest(
        sys.argv[3], specification, sys.argv[4], sys.argv[5]
    )
    verify_artifact_snapshot(sys.argv[6], closure)
elif command == "restore-artifact-snapshot":
    specification = read_artifact_root_spec(sys.argv[2])
    closure = read_artifact_manifest(
        sys.argv[3], specification, sys.argv[4], sys.argv[5]
    )
    restore_artifact_snapshot(specification, closure, sys.argv[6])
elif command == "file-descriptor":
    length, observed = stable_file_descriptor(sys.argv[2], "artifact file")
    print(length, observed, sep="\t")
elif command == "verify-storage-fd":
    root = retained_directory_fd(sys.argv[2])
    try:
        verify_retained_storage_root(root, sys.argv[3])
    finally:
        os.close(root)
elif command == "fd-entry-kind":
    root = retained_directory_fd(sys.argv[2])
    try:
        print(anchored_entry_kind(root, sys.argv[3]))
    finally:
        os.close(root)
elif command == "fd-copy-out":
    root = retained_directory_fd(sys.argv[2])
    try:
        copy_anchored_out(root, sys.argv[3], sys.argv[4])
    finally:
        os.close(root)
elif command == "fd-copy-in":
    root = retained_directory_fd(sys.argv[2])
    try:
        copy_into_anchored(root, sys.argv[3], sys.argv[4])
    finally:
        os.close(root)
elif command == "fd-install-immutable":
    root = retained_directory_fd(sys.argv[2])
    expected_bytes = integer(int(sys.argv[5]), "immutable file remote size")
    expected_sha256 = digest(sys.argv[6], "immutable file remote digest")
    try:
        copy_into_anchored(
            root,
            sys.argv[3],
            sys.argv[4],
            immutable=True,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
    finally:
        os.close(root)
elif command == "fd-stage-tree":
    root = retained_directory_fd(sys.argv[2])
    try:
        print(stage_anchored_tree(root, sys.argv[3], sys.argv[4]))
    finally:
        os.close(root)
elif command == "fd-remove-tree":
    root = retained_directory_fd(sys.argv[2])
    try:
        remove_anchored_entry(root, sys.argv[3])
    finally:
        os.close(root)
elif command == "fd-rename-tree":
    root = retained_directory_fd(sys.argv[2])
    mode = sys.argv[5]
    if mode not in ("replace", "no-replace"):
        fail(f"file remote rename mode {mode!r} is invalid")
    try:
        rename_anchored_tree(
            root,
            sys.argv[3],
            sys.argv[4],
            replace=(mode == "replace"),
        )
    finally:
        os.close(root)
elif command == "fd-publish-pointer":
    root = retained_directory_fd(sys.argv[2])
    try:
        print(publish_anchored_remote_pointer(root, sys.argv[3]))
    finally:
        os.close(root)
elif command == "verify-file":
    expected_bytes = integer(int(sys.argv[3]), "artifact file size")
    expected_digest = digest(sys.argv[4], "artifact file digest")
    length, observed = stable_file_descriptor(sys.argv[2], "artifact file")
    if length != expected_bytes or observed != expected_digest:
        fail("artifact file differs from its expected size or digest")
elif command == "install-immutable":
    install_immutable(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
elif command == "canonical-storage-root":
    print(clean_absolute_path(sys.argv[2], "file remote root"))
elif command == "pointer-local":
    generation, manifest_digest = read_pointer(sys.argv[2], remote=False)
    print(generation, manifest_digest, sep="\t")
elif command == "pointer-remote":
    print(*read_pointer(sys.argv[2], remote=True), sep="\t")
elif command == "make-remote-pointer":
    generation, manifest_digest = read_pointer(sys.argv[2], remote=False)
    global_step = integer(int(sys.argv[3]), "remote pointer global_step")
    metric_records = integer(int(sys.argv[4]), "remote pointer metric_records")
    metrics_path = safe_path(sys.argv[5])
    if metrics_path != f"checkpoint-metrics/{generation}/metrics.jsonl":
        fail("remote pointer metrics path does not match its checkpoint generation")
    metrics_bytes = integer(int(sys.argv[6]), "remote pointer metrics size")
    metrics_sha256 = digest(sys.argv[7], "remote pointer metrics digest")
    artifact_bytes = integer(int(sys.argv[8]), "remote pointer artifact manifest size")
    artifact_sha256 = digest(sys.argv[9], "remote pointer artifact manifest digest")
    pointer = {
        "version": 2,
        "generation": generation,
        "manifest_sha256": manifest_digest,
        "global_step": global_step,
        "metric_records": metric_records,
        "metrics_path": metrics_path,
        "metrics_bytes": metrics_bytes,
        "metrics_sha256": metrics_sha256,
        "artifact_manifest_bytes": artifact_bytes,
        "artifact_manifest_sha256": artifact_sha256,
    }
    print(json.dumps(pointer, sort_keys=True, separators=(",", ":")))
elif command == "make-local-pointer":
    pointer = read_pointer(sys.argv[2], remote=True)
    print(
        json.dumps(
            {
                "version": 1,
                "generation": pointer[0],
                "manifest_sha256": pointer[1],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
elif command == "compare-remote-pointers":
    candidate, _ = load_json_file(sys.argv[2], "candidate remote pointer")
    existing, _ = load_json_file(sys.argv[3], "existing remote pointer")
    print(compare_remote_pointer_values(candidate, existing))
elif command == "manifest-files":
    for path in read_manifest_transport(sys.argv[2], sys.argv[3], sys.argv[4]):
        print(path)
elif command == "snapshot-metrics":
    snapshot_metrics(sys.argv[2], int(sys.argv[3]), sys.argv[4])
elif command == "atomic-copy":
    import shutil
    import tempfile

    source, destination = sys.argv[2], sys.argv[3]
    regular_file(source, "atomic copy source")
    parent = os.path.dirname(destination)
    real_directory(parent, "atomic copy destination directory")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(destination)}.restore-", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as output, open(source, "rb") as input_file:
            shutil.copyfileobj(input_file, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
elif command == "sync-directory":
    real_directory(sys.argv[2], "directory to sync")
    descriptor = os.open(sys.argv[2], os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
elif command == "sync-tree":
    root = sys.argv[2]
    real_directory(root, "directory tree to sync")
    directories = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        directories.append(directory)
        for name in names:
            real_directory(os.path.join(directory, name), "directory tree child")
        for name in files:
            path = os.path.join(directory, name)
            regular_file(path, "directory tree file")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
else:
    fail(f"unknown checkpoint helper command {command!r}")
PY
}

ARTIFACT_ROOT_SPEC="$STATE_DIR/generated-artifact-roots.json"
artifact_root_spec_temporary=$(mktemp "$STATE_DIR/generated-artifact-roots.XXXXXX") \
  || die "cannot allocate generated-artifact root specification"
if ! checkpoint_tool artifact-root-spec "$OUTPUT" \
  "${HERMES_TRAIN_COMMAND[@]}" >"$artifact_root_spec_temporary"; then
  rm -f -- "$artifact_root_spec_temporary"
  die "cannot derive generated-artifact roots from the training command"
fi
checkpoint_tool atomic-copy "$artifact_root_spec_temporary" "$ARTIFACT_ROOT_SPEC" \
  || {
    rm -f -- "$artifact_root_spec_temporary"
    die "cannot publish generated-artifact root specification"
  }
rm -f -- "$artifact_root_spec_temporary"
readonly ARTIFACT_ROOT_SPEC

LOCAL_REMOTE_ROOT=
LOCAL_REMOTE_FD=
if [[ $REMOTE == file://* ]]; then
  raw_local_remote_root=${REMOTE#file://}
  [[ -n "$raw_local_remote_root" ]] || die "file remote root is empty"
  if [[ -e "$raw_local_remote_root" || -L "$raw_local_remote_root" ]]; then
    [[ -d "$raw_local_remote_root" && ! -L "$raw_local_remote_root" ]] \
      || die "file remote root is not a real directory: $raw_local_remote_root"
  else
    mkdir -p -- "$raw_local_remote_root" \
      || die "cannot create file remote root: $raw_local_remote_root"
  fi
  LOCAL_REMOTE_ROOT=$(checkpoint_tool canonical-storage-root "$raw_local_remote_root") \
    || die "cannot validate file remote root: $raw_local_remote_root"
  exec 6<"$LOCAL_REMOTE_ROOT" \
    || die "cannot retain file remote root: $LOCAL_REMOTE_ROOT"
  LOCAL_REMOTE_FD=6
  checkpoint_tool verify-storage-fd "$LOCAL_REMOTE_FD" "$LOCAL_REMOTE_ROOT" \
    || die "file remote root changed while it was opened: $LOCAL_REMOTE_ROOT"
fi
readonly LOCAL_REMOTE_ROOT
readonly LOCAL_REMOTE_FD

trainer_generation_descriptor() {
  local -a command=(
    "$CHECKPOINT_VERIFIER_BIN" verify-checkpoint
    --generation "$1"
    --generation-name "$2"
    --manifest-sha256 "$3"
    --format tsv
  )
  [[ -z ${4:-} ]] || command+=(--metrics "$4")
  [[ ${5:-false} != true ]] || command+=(--exact-metrics)
  "${command[@]}"
}

verify_checkpoint_generation() {
  local trainer step generation manifest_sha256 metric_records extra
  trainer=$(trainer_generation_descriptor "$1" "$2" "$3") || return 1
  [[ $trainer != *$'\n'* ]] || return 1
  IFS=$'\t' read -r step generation manifest_sha256 metric_records extra <<<"$trainer"
  is_nonnegative_integer "$step" \
    && is_nonnegative_integer "$metric_records" \
    && [[ $generation == "$2" && $manifest_sha256 == "$3" && -z $extra ]] \
    || return 1
  printf '%s\t%s\n' "$step" "$metric_records"
}

checkpoint_descriptor() {
  local root=$1 trainer
  if trainer=$("$CHECKPOINT_VERIFIER_BIN" verify-checkpoint \
    --root "$root" --metrics "$root/metrics.jsonl" --format tsv 2>/dev/null); then
    printf '%s\n' "$trainer"
    return 0
  fi

  local remote_pointer generation manifest_sha256 pointer_step pointer_records
  local metrics_path metrics_bytes metrics_sha256 _artifact_bytes _artifact_sha256
  remote_pointer=$(checkpoint_tool pointer-remote "$root/$CURRENT_POINTER") || return 1
  IFS=$'\t' read -r generation manifest_sha256 pointer_step pointer_records \
    metrics_path metrics_bytes metrics_sha256 _artifact_bytes _artifact_sha256 \
    <<<"$remote_pointer"
  checkpoint_tool verify-file "$root/$metrics_path" \
    "$metrics_bytes" "$metrics_sha256" || return 1
  trainer=$(trainer_generation_descriptor \
    "$root/$GENERATIONS_DIRECTORY/$generation" \
    "$generation" "$manifest_sha256" "$root/$metrics_path" true) || return 1
  [[ $trainer == "$pointer_step"$'\t'"$generation"$'\t'"$manifest_sha256"$'\t'"$pointer_records" ]] \
    || return 1
  printf '%s\n' "$trainer"
}

checkpoint_artifacts_exist() {
  local file
  [[ -e "$OUTPUT/$CURRENT_POINTER" || -L "$OUTPUT/$CURRENT_POINTER" ]] && return 0
  [[ -e "$OUTPUT/$GENERATIONS_DIRECTORY" || -L "$OUTPUT/$GENERATIONS_DIRECTORY" ]] && return 0
  for file in "${OBSOLETE_FLAT_CHECKPOINT_FILES[@]}"; do
    [[ -e "$OUTPUT/$file" || -L "$OUTPUT/$file" \
      || -e "$OUTPUT/$file.tmp" || -L "$OUTPUT/$file.tmp" ]] && return 0
  done
  return 1
}

remote_path() {
  printf '%s/%s' "${REMOTE%/}" "${1#/}"
}

remote_download() {
  local relative=$1
  local destination=$2
  if [[ $REMOTE == file://* ]]; then
    checkpoint_tool fd-copy-out "$LOCAL_REMOTE_FD" "$relative" "$destination"
  else
    "$GCLOUD_BIN" storage cp "$(remote_path "$relative")" "$destination"
  fi
}

remote_upload_file() {
  local source=$1
  local relative=$2
  if [[ $REMOTE == file://* ]]; then
    checkpoint_tool fd-copy-in "$LOCAL_REMOTE_FD" "$source" "$relative"
  else
    "$GCLOUD_BIN" storage cp "$source" "$(remote_path "$relative")"
  fi
}

remote_upload_immutable_file() {
  local source=$1
  local relative=$2
  local expected_bytes=$3
  local expected_sha256=$4
  local existing

  checkpoint_tool verify-file "$source" "$expected_bytes" "$expected_sha256" \
    || return 1
  if [[ $REMOTE == file://* ]]; then
    checkpoint_tool fd-install-immutable "$LOCAL_REMOTE_FD" "$source" \
      "$relative" "$expected_bytes" "$expected_sha256"
    return
  fi

  existing=$(mktemp "$STATE_DIR/remote-object.XXXXXX") || return 1
  if remote_download "$relative" "$existing" >/dev/null 2>&1; then
    if checkpoint_tool verify-file "$existing" \
      "$expected_bytes" "$expected_sha256"; then
      rm -f -- "$existing"
      return 0
    fi
    rm -f -- "$existing"
    log "immutable remote object already exists with different bytes: $relative"
    return 1
  fi
  rm -f -- "$existing"
  # Never replace an object written by another supervisor. If a concurrent
  # writer wins this precondition, the next sync verifies and reuses its bytes.
  "$GCLOUD_BIN" storage cp --if-generation-match=0 \
    "$source" "$(remote_path "$relative")" || return 1
  existing=$(mktemp "$STATE_DIR/remote-object.XXXXXX") || return 1
  if ! remote_download "$relative" "$existing" >/dev/null \
    || ! checkpoint_tool verify-file "$existing" \
      "$expected_bytes" "$expected_sha256"; then
    rm -f -- "$existing"
    return 1
  fi
  rm -f -- "$existing"
}

compare_remote_pointer_files() {
  checkpoint_tool compare-remote-pointers "$1" "$2"
}

publish_file_remote_pointer() (
  local candidate=$1
  checkpoint_tool fd-publish-pointer "$LOCAL_REMOTE_FD" "$candidate"
)

gcs_pointer_generation() {
  "$GCLOUD_BIN" storage objects describe "$(remote_path "$CURRENT_POINTER")" \
    --format='value(generation)'
}

publish_gcs_remote_pointer() {
  local candidate=$1
  local before after existing state
  existing=$(mktemp "$STATE_DIR/existing-current.XXXXXX") || return 1
  for _attempt in {1..12}; do
    if before=$(gcs_pointer_generation 2>/dev/null); then
      [[ $before =~ ^[1-9][0-9]*$ ]] || {
        rm -f -- "$existing"
        return 1
      }
      if ! remote_download "$CURRENT_POINTER" "$existing" >/dev/null \
        || ! after=$(gcs_pointer_generation 2>/dev/null); then
        continue
      fi
      [[ $before == "$after" ]] || continue
      state=$(compare_remote_pointer_files "$candidate" "$existing") || {
        rm -f -- "$existing"
        return 1
      }
      case "$state" in
        newer | same)
          rm -f -- "$existing"
          printf '%s\n' "$state"
          return 0
          ;;
        advance) ;;
        *)
          rm -f -- "$existing"
          return 1
          ;;
      esac
    else
      before=0
      if remote_download "$CURRENT_POINTER" "$existing" >/dev/null 2>&1; then
        continue
      fi
    fi

    if "$GCLOUD_BIN" storage cp --if-generation-match="$before" \
      "$candidate" "$(remote_path "$CURRENT_POINTER")" >/dev/null; then
      if remote_download "$CURRENT_POINTER" "$existing" >/dev/null \
        && [[ $(compare_remote_pointer_files "$candidate" "$existing") == same ]]; then
        rm -f -- "$existing"
        printf 'published\n'
        return 0
      fi
      rm -f -- "$existing"
      return 1
    fi
  done
  rm -f -- "$existing"
  return 1
}

publish_remote_pointer() {
  if [[ $REMOTE == file://* ]]; then
    publish_file_remote_pointer "$1"
  else
    publish_gcs_remote_pointer "$1"
  fi
}

UPLOADED_ARTIFACT_MANIFEST_BYTES=
UPLOADED_ARTIFACT_MANIFEST_SHA256=
remote_upload_artifacts() {
  local generation=$1
  local manifest_sha256=$2
  local closure plan descriptor root_id source_root relative bytes sha256
  local upload_failed=false

  UPLOADED_ARTIFACT_MANIFEST_BYTES=
  UPLOADED_ARTIFACT_MANIFEST_SHA256=

  closure=$(mktemp "$STATE_DIR/artifact-closure.XXXXXX") || return 1
  plan=$(mktemp "$STATE_DIR/artifact-plan.XXXXXX") || {
    rm -f -- "$closure"
    return 1
  }
  if ! checkpoint_tool build-artifact-manifest "$ARTIFACT_ROOT_SPEC" \
    "$generation" "$manifest_sha256" \
    "$OUTPUT/$GENERATIONS_DIRECTORY/$generation" >"$closure" \
    || ! checkpoint_tool verify-artifact-sources "$ARTIFACT_ROOT_SPEC" \
      "$closure" "$generation" "$manifest_sha256" \
    || ! checkpoint_tool artifact-plan "$ARTIFACT_ROOT_SPEC" \
      "$closure" "$generation" "$manifest_sha256" >"$plan"; then
    rm -f -- "$closure" "$plan"
    return 1
  fi
  while IFS=$'\t' read -r root_id source_root relative bytes sha256; do
    [[ -n "$root_id" && -n "$source_root" && -n "$relative" \
      && -n "$bytes" && -n "$sha256" ]] || {
      upload_failed=true
      break
    }
    if ! remote_upload_immutable_file "$source_root/$relative" \
      "$ARTIFACT_OBJECTS_DIRECTORY/${sha256:0:2}/$sha256" \
      "$bytes" "$sha256"; then
      upload_failed=true
      break
    fi
  done <"$plan"
  rm -f -- "$plan"
  if [[ $upload_failed == true ]] \
    || ! checkpoint_tool verify-artifact-sources "$ARTIFACT_ROOT_SPEC" \
      "$closure" "$generation" "$manifest_sha256"; then
    rm -f -- "$closure"
    return 1
  fi
  descriptor=$(checkpoint_tool file-descriptor "$closure") || {
    rm -f -- "$closure"
    return 1
  }
  IFS=$'\t' read -r bytes sha256 <<<"$descriptor"
  if [[ -z "$bytes" || -z "$sha256" ]] \
    || ! remote_upload_immutable_file "$closure" \
      "$ARTIFACTS_DIRECTORY/$generation/$ARTIFACT_MANIFEST" \
      "$bytes" "$sha256"; then
    rm -f -- "$closure"
    return 1
  fi
  UPLOADED_ARTIFACT_MANIFEST_BYTES=$bytes
  UPLOADED_ARTIFACT_MANIFEST_SHA256=$sha256
  rm -f -- "$closure"
}

download_remote_artifacts() {
  local destination=$1
  local generation=$2
  local manifest_sha256=$3
  local expected_closure_bytes=$4
  local expected_closure_sha256=$5
  local closure="$destination/$ARTIFACT_MANIFEST"
  local roots="$destination/roots"
  local plan root_plan root_id _source_root relative bytes sha256
  local download_failed=false

  mkdir -p -- "$roots" || return 1
  remote_download "$ARTIFACTS_DIRECTORY/$generation/$ARTIFACT_MANIFEST" \
    "$closure" >/dev/null || {
      log "remote checkpoint $generation has no generated-artifact closure"
      return 1
    }
  checkpoint_tool verify-file "$closure" \
    "$expected_closure_bytes" "$expected_closure_sha256" || {
    log "remote checkpoint $generation has a rewritten generated-artifact closure"
    return 1
  }
  root_plan=$(mktemp "$STATE_DIR/artifact-roots.XXXXXX") || return 1
  plan=$(mktemp "$STATE_DIR/artifact-plan.XXXXXX") || {
    rm -f -- "$root_plan"
    return 1
  }
  if ! checkpoint_tool artifact-root-ids "$ARTIFACT_ROOT_SPEC" \
    "$closure" "$generation" "$manifest_sha256" >"$root_plan" \
    || ! checkpoint_tool artifact-plan "$ARTIFACT_ROOT_SPEC" \
      "$closure" "$generation" "$manifest_sha256" >"$plan"; then
    rm -f -- "$root_plan" "$plan"
    return 1
  fi
  while IFS= read -r root_id; do
    [[ -n "$root_id" ]] || {
      download_failed=true
      break
    }
    mkdir -p -- "$roots/$root_id" || {
      download_failed=true
      break
    }
  done <"$root_plan"
  rm -f -- "$root_plan"
  if [[ $download_failed == false ]]; then
    while IFS=$'\t' read -r root_id _source_root relative bytes sha256; do
      [[ -n "$root_id" && -n "$relative" && -n "$bytes" && -n "$sha256" ]] || {
        download_failed=true
        break
      }
      mkdir -p -- "$(dirname -- "$roots/$root_id/$relative")" || {
        download_failed=true
        break
      }
      if ! remote_download \
        "$ARTIFACT_OBJECTS_DIRECTORY/${sha256:0:2}/$sha256" \
        "$roots/$root_id/$relative" >/dev/null \
        || ! checkpoint_tool verify-file "$roots/$root_id/$relative" \
          "$bytes" "$sha256"; then
        log "remote generated artifact $root_id:$relative is missing or corrupt"
        download_failed=true
        break
      fi
    done <"$plan"
  fi
  rm -f -- "$plan"
  [[ $download_failed == false ]] || return 1
  checkpoint_tool verify-artifact-snapshot "$ARTIFACT_ROOT_SPEC" \
    "$closure" "$generation" "$manifest_sha256" "$roots"
}

verify_remote_artifacts() {
  local generation=$1
  local manifest_sha256=$2
  local closure_bytes=$3
  local closure_sha256=$4
  local verification
  verification=$(mktemp -d "$STATE_DIR/verify-artifacts.XXXXXX") || return 1
  if download_remote_artifacts "$verification" \
    "$generation" "$manifest_sha256" "$closure_bytes" "$closure_sha256"; then
    rm -rf -- "$verification"
    return 0
  fi
  rm -rf -- "$verification"
  return 1
}

remote_upload_generation() {
  local generation=$1
  local manifest_sha256=$2
  local source="$OUTPUT/$GENERATIONS_DIRECTORY/$generation"
  local destination destination_kind staging file plan upload_failed=false

  if [[ $REMOTE == file://* ]]; then
    staging=$(checkpoint_tool fd-stage-tree "$LOCAL_REMOTE_FD" \
      "$source" "$GENERATIONS_DIRECTORY") || return 1
    destination="$GENERATIONS_DIRECTORY/$generation"
    destination_kind=$(checkpoint_tool fd-entry-kind \
      "$LOCAL_REMOTE_FD" "$destination") || {
      checkpoint_tool fd-remove-tree "$LOCAL_REMOTE_FD" "$staging" || true
      return 1
    }
    if [[ $destination_kind != missing ]]; then
      checkpoint_tool fd-rename-tree "$LOCAL_REMOTE_FD" \
        "$staging" "$destination" replace || {
        checkpoint_tool fd-remove-tree "$LOCAL_REMOTE_FD" "$staging" || true
        return 1
      }
    else
      checkpoint_tool fd-rename-tree "$LOCAL_REMOTE_FD" \
        "$staging" "$destination" no-replace || {
        checkpoint_tool fd-remove-tree "$LOCAL_REMOTE_FD" "$staging" || true
        return 1
      }
    fi
    return
  fi

  plan=$(mktemp "$STATE_DIR/manifest-files.XXXXXX") || return 1
  if ! checkpoint_tool manifest-files "$source/$GENERATION_MANIFEST" \
    "$generation" "$manifest_sha256" >"$plan"; then
    rm -f -- "$plan"
    return 1
  fi
  while IFS= read -r file; do
    if ! remote_upload_file "$source/$file" \
      "$GENERATIONS_DIRECTORY/$generation/$file"; then
      upload_failed=true
      break
    fi
  done <"$plan"
  rm -f -- "$plan"
  [[ $upload_failed == false ]] || return 1
  remote_upload_file "$source/$GENERATION_MANIFEST" \
    "$GENERATIONS_DIRECTORY/$generation/$GENERATION_MANIFEST"
}

download_remote_generation() {
  local destination=$1
  local generation=$2
  local manifest_sha256=$3
  local download_failed=false file plan

  mkdir -p -- "$destination" || return 1
  remote_download \
    "$GENERATIONS_DIRECTORY/$generation/$GENERATION_MANIFEST" \
    "$destination/$GENERATION_MANIFEST" >/dev/null || return 1
  plan=$(mktemp "$STATE_DIR/manifest-files.XXXXXX") || return 1
  if ! checkpoint_tool manifest-files "$destination/$GENERATION_MANIFEST" \
    "$generation" "$manifest_sha256" >"$plan"; then
    rm -f -- "$plan"
    return 1
  fi
  while IFS= read -r file; do
    mkdir -p -- "$(dirname -- "$destination/$file")"
    if ! remote_download "$GENERATIONS_DIRECTORY/$generation/$file" \
      "$destination/$file" >/dev/null; then
      log "remote generation $generation is incomplete ($file is unavailable)"
      download_failed=true
      break
    fi
  done <"$plan"
  rm -f -- "$plan"
  [[ $download_failed == false ]] || return 1
  verify_checkpoint_generation "$destination" \
    "$generation" "$manifest_sha256" >/dev/null
}

REMOTE_STEP=
REMOTE_GENERATION=
REMOTE_MANIFEST_SHA256=
REMOTE_METRICS_PATH=
REMOTE_ARTIFACT_MANIFEST_BYTES=
REMOTE_ARTIFACT_MANIFEST_SHA256=
REMOTE_SNAPSHOT=
REMOTE_ARTIFACTS=false

clear_remote_snapshot() {
  if [[ -n "$REMOTE_SNAPSHOT" ]]; then
    rm -rf -- "$REMOTE_SNAPSHOT"
    REMOTE_SNAPSHOT=
  fi
}

download_remote_checkpoint() {
  local destination=$1
  local pointer descriptor generation manifest_sha256 step metric_records
  local pointer_step pointer_records metrics_path metrics_bytes metrics_sha256
  local closure_bytes closure_sha256

  mkdir -p -- "$destination/$GENERATIONS_DIRECTORY" || return 1
  pointer="$destination/$CURRENT_POINTER"
  remote_download "$CURRENT_POINTER" "$pointer" >/dev/null || return 1
  REMOTE_ARTIFACTS=true
  descriptor=$(checkpoint_tool pointer-remote "$pointer") || return 1
  IFS=$'\t' read -r generation manifest_sha256 pointer_step pointer_records \
    metrics_path metrics_bytes metrics_sha256 closure_bytes closure_sha256 <<<"$descriptor"
  [[ -n "$generation" && -n "$manifest_sha256" ]] || return 1
  download_remote_generation \
    "$destination/$GENERATIONS_DIRECTORY/$generation" \
    "$generation" "$manifest_sha256" || return 1
  download_remote_artifacts \
    "$destination/$ARTIFACTS_DIRECTORY/$generation" \
    "$generation" "$manifest_sha256" \
    "$closure_bytes" "$closure_sha256" || return 1
  descriptor=$(verify_checkpoint_generation \
    "$destination/$GENERATIONS_DIRECTORY/$generation" \
    "$generation" "$manifest_sha256") || return 1
  IFS=$'\t' read -r step metric_records <<<"$descriptor"
  [[ -n "$step" && -n "$metric_records" \
    && $step == "$pointer_step" && $metric_records == "$pointer_records" ]] || return 1
  mkdir -p -- "$(dirname -- "$destination/$metrics_path")" || return 1
  remote_download "$metrics_path" "$destination/$metrics_path" >/dev/null || {
    log "remote checkpoint at step $step has no immutable metric snapshot"
    return 1
  }
  checkpoint_tool verify-file "$destination/$metrics_path" \
    "$metrics_bytes" "$metrics_sha256" || return 1
  checkpoint_descriptor "$destination" || return 1
}

refresh_remote_checkpoint() {
  local descriptor descriptor_file snapshot
  clear_remote_snapshot
  REMOTE_STEP=
  REMOTE_GENERATION=
  REMOTE_MANIFEST_SHA256=
  REMOTE_METRICS_PATH=
  REMOTE_ARTIFACT_MANIFEST_BYTES=
  REMOTE_ARTIFACT_MANIFEST_SHA256=
  REMOTE_ARTIFACTS=false
  [[ -n "$REMOTE" ]] || return 1
  snapshot=$(mktemp -d "$STATE_DIR/remote-checkpoint.XXXXXX") || return 1
  if [[ $REMOTE == file://* ]]; then
    local pointer_kind
    pointer_kind=$(checkpoint_tool fd-entry-kind \
      "$LOCAL_REMOTE_FD" "$CURRENT_POINTER") || return 1
    [[ $pointer_kind == missing ]] || REMOTE_ARTIFACTS=true
  fi
  descriptor_file=$(mktemp "$STATE_DIR/remote-descriptor.XXXXXX") || {
    rm -rf -- "$snapshot"
    return 1
  }
  if download_remote_checkpoint "$snapshot" >"$descriptor_file"; then
    descriptor=$(<"$descriptor_file")
    rm -f -- "$descriptor_file"
    IFS=$'\t' read -r REMOTE_STEP REMOTE_GENERATION \
      REMOTE_MANIFEST_SHA256 _ <<<"$descriptor"
    descriptor=$(checkpoint_tool pointer-remote "$snapshot/$CURRENT_POINTER") || {
      rm -rf -- "$snapshot"
      return 1
    }
    IFS=$'\t' read -r _ _ _ _ REMOTE_METRICS_PATH _ _ \
      REMOTE_ARTIFACT_MANIFEST_BYTES REMOTE_ARTIFACT_MANIFEST_SHA256 <<<"$descriptor"
    REMOTE_SNAPSHOT=$snapshot
    return 0
  fi
  rm -f -- "$descriptor_file"
  rm -rf -- "$snapshot"
  return 1
}

restore_remote_checkpoint() {
  local expected_step=$1
  local descriptor generation manifest_sha256 step
  local generation_root source destination staging quarantine local_pointer
  [[ -n "$REMOTE_SNAPSHOT" ]] || return 1
  descriptor=$(checkpoint_descriptor "$REMOTE_SNAPSHOT") || return 1
  IFS=$'\t' read -r step generation manifest_sha256 _ <<<"$descriptor"
  [[ $step == "$expected_step" \
    && $generation == "$REMOTE_GENERATION" \
    && $manifest_sha256 == "$REMOTE_MANIFEST_SHA256" ]] || return 1

  generation_root="$OUTPUT/$GENERATIONS_DIRECTORY"
  if [[ -e "$generation_root" || -L "$generation_root" ]]; then
    [[ -d "$generation_root" && ! -L "$generation_root" ]] \
      || return 1
  else
    mkdir -p -- "$generation_root" || return 1
  fi
  source="$REMOTE_SNAPSHOT/$GENERATIONS_DIRECTORY/$generation"
  destination="$generation_root/$generation"
  staging=$(mktemp -d "$generation_root/.restore.XXXXXX") || return 1
  cp -R -- "$source/." "$staging/" || {
    rm -rf -- "$staging"
    return 1
  }
  verify_checkpoint_generation "$staging" \
    "$generation" "$manifest_sha256" >/dev/null || {
      rm -rf -- "$staging"
      return 1
    }
  checkpoint_tool sync-tree "$staging" || {
    rm -rf -- "$staging"
    return 1
  }

  if [[ -e "$destination" || -L "$destination" ]]; then
    if verify_checkpoint_generation "$destination" \
      "$generation" "$manifest_sha256" >/dev/null 2>&1; then
      rm -rf -- "$staging"
    else
      quarantine=$(mktemp -d "$generation_root/.corrupt.XXXXXX") || {
        rm -rf -- "$staging"
        return 1
      }
      rmdir -- "$quarantine" || {
        rm -rf -- "$staging" "$quarantine"
        return 1
      }
      if ! mv -- "$destination" "$quarantine" \
        || ! mv -- "$staging" "$destination"; then
        [[ -e "$destination" || ! -e "$quarantine" ]] \
          || mv -- "$quarantine" "$destination" 2>/dev/null || true
        rm -rf -- "$staging"
        return 1
      fi
      log "replaced corrupt local generation; preserved it at $quarantine"
    fi
  else
    mv -- "$staging" "$destination" || {
      rm -rf -- "$staging"
      return 1
    }
  fi
  checkpoint_tool sync-directory "$generation_root" || return 1

  # Restore every immutable external artifact authenticated by this generation
  # before making its local current.json visible. Existing identical files are
  # reused; conflicting files are preserved and make the restore fail closed.
  checkpoint_tool verify-file \
    "$REMOTE_SNAPSHOT/$ARTIFACTS_DIRECTORY/$generation/$ARTIFACT_MANIFEST" \
    "$REMOTE_ARTIFACT_MANIFEST_BYTES" \
    "$REMOTE_ARTIFACT_MANIFEST_SHA256" || return 1
  checkpoint_tool restore-artifact-snapshot "$ARTIFACT_ROOT_SPEC" \
    "$REMOTE_SNAPSHOT/$ARTIFACTS_DIRECTORY/$generation/$ARTIFACT_MANIFEST" \
    "$generation" "$manifest_sha256" \
    "$REMOTE_SNAPSHOT/$ARTIFACTS_DIRECTORY/$generation/roots" || return 1

  # Remote releases carry the exact committed prefix under an immutable,
  # generation-specific path. Install it before deriving the trainer's strict
  # local v1 pointer so no visible generation can lack its reporting history.
  checkpoint_tool atomic-copy "$REMOTE_SNAPSHOT/$REMOTE_METRICS_PATH" \
    "$OUTPUT/metrics.jsonl" || return 1
  local_pointer=$(mktemp "$STATE_DIR/local-current.XXXXXX") || return 1
  if ! checkpoint_tool make-local-pointer \
    "$REMOTE_SNAPSHOT/$CURRENT_POINTER" >"$local_pointer" \
    || ! checkpoint_tool atomic-copy "$local_pointer" \
      "$OUTPUT/$CURRENT_POINTER"; then
    rm -f -- "$local_pointer"
    return 1
  fi
  rm -f -- "$local_pointer"
  descriptor=$(checkpoint_descriptor "$OUTPUT") || return 1
  IFS=$'\t' read -r step generation manifest_sha256 _ <<<"$descriptor"
  [[ $step == "$expected_step" ]] || return 1
  log "restored remote checkpoint at step $expected_step"
}

RESUME_STEP=
prepare_checkpoint() {
  local descriptor local_step='' local_generation=''
  local remote_available=false remote_invalid=false
  RESUME_STEP=
  clear_remote_snapshot

  if descriptor=$(checkpoint_descriptor "$OUTPUT" 2>>"$SYNC_LOG"); then
    IFS=$'\t' read -r local_step local_generation _ _ <<<"$descriptor"
    log "found complete local checkpoint at step $local_step"
  fi
  if [[ -n "$REMOTE" ]] \
    && refresh_remote_checkpoint >>"$SYNC_LOG" 2>&1; then
    remote_available=true
    log "found remote checkpoint at step $REMOTE_STEP"
  elif [[ $REMOTE_ARTIFACTS == true ]]; then
    remote_invalid=true
    log "remote current.json does not reference a complete verified checkpoint"
  fi

  if [[ $remote_available == true && -n "$local_step" \
    && $REMOTE_STEP -eq local_step && $REMOTE_GENERATION != "$local_generation" ]]; then
    clear_remote_snapshot
    die "equal-step local/remote checkpoint fork: $local_generation versus $REMOTE_GENERATION"
  fi

  # Rehydrate an equal remote generation as well. The checkpoint generation
  # authenticates trainer files, but its generated sleep/QAT closure and exact
  # committed metric prefix live in the remote release envelope. A VM can keep
  # current.json while losing either of those external artifacts; treating the
  # equal generation as already complete would make the subsequent resume
  # depend on damaged local state.
  if [[ $remote_available == true \
    && ( -z "$local_step" || $REMOTE_STEP -ge $local_step ) ]]; then
    restore_remote_checkpoint "$REMOTE_STEP" \
      || die "cannot restore the newest remote checkpoint"
    local_step=$REMOTE_STEP
  fi

  if [[ -z "$local_step" ]]; then
    clear_remote_snapshot
    if [[ $remote_invalid == true ]]; then
      die "remote checkpoint is incomplete and no usable local checkpoint is available"
    fi
    if checkpoint_artifacts_exist; then
      die "local checkpoint is incomplete and no usable remote checkpoint is available"
    fi
    log "no checkpoint found; starting a new run"
    return 1
  fi
  RESUME_STEP=$local_step
  clear_remote_snapshot
}

verify_remote_generation() {
  local generation=$1
  local manifest_sha256=$2
  local verification
  verification=$(mktemp -d "$STATE_DIR/verify-upload.XXXXXX") || return 1
  if download_remote_generation "$verification" \
    "$generation" "$manifest_sha256"; then
    rm -rf -- "$verification"
    return 0
  fi
  rm -rf -- "$verification"
  return 1
}

sync_checkpoint_once() (
  local descriptor step generation manifest_sha256 metric_records
  local after_step after_generation after_manifest_sha256
  local remote_step=-1 remote_generation='' sync_owner sync_lock_owned=false
  local pointer_snapshot='' metrics_snapshot='' remote_pointer=''
  local metrics_relative metrics_bytes metrics_sha256 pointer_result
  exec 9>&-
  [[ -n "$REMOTE" ]] || return 0

  # shellcheck disable=SC2317,SC2329 # Invoked by the EXIT trap below (code differs by ShellCheck version).
  sync_cleanup() {
    [[ -z "$pointer_snapshot" ]] || rm -f -- "$pointer_snapshot"
    [[ -z "$metrics_snapshot" ]] || rm -f -- "$metrics_snapshot"
    [[ -z "$remote_pointer" ]] || rm -f -- "$remote_pointer"
    clear_remote_snapshot
    [[ $LOCK_TOOL != shlock || $sync_lock_owned != true ]] \
      || rm -f -- "$STATE_DIR/sync.lock"
  }
  trap sync_cleanup EXIT

  if [[ $LOCK_TOOL == flock ]]; then
    exec 8>"$STATE_DIR/sync.lock"
    flock -n 8 || return 0
  else
    # Bash 3.2 has no BASHPID. A short child observes this subshell as PPID.
    sync_owner=$(sh -c 'printf "%s" "$PPID"')
    shlock -f "$STATE_DIR/sync.lock" -p "$sync_owner" || return 0
    sync_lock_owned=true
  fi

  descriptor=$(checkpoint_descriptor "$OUTPUT" 2>/dev/null) || {
    log "checkpoint sync skipped: no complete local checkpoint"
    return 0
  }
  IFS=$'\t' read -r step generation manifest_sha256 metric_records <<<"$descriptor"

  pointer_snapshot=$(mktemp "$STATE_DIR/current.XXXXXX") || return 1
  cp -- "$OUTPUT/$CURRENT_POINTER" "$pointer_snapshot" || return 1
  descriptor=$(checkpoint_tool pointer-local "$pointer_snapshot") || return 1
  IFS=$'\t' read -r after_generation after_manifest_sha256 <<<"$descriptor"
  [[ $after_generation == "$generation" \
    && $after_manifest_sha256 == "$manifest_sha256" ]] || return 1

  if refresh_remote_checkpoint; then
    remote_step=$REMOTE_STEP
    remote_generation=$REMOTE_GENERATION
  fi
  clear_remote_snapshot
  if (( remote_step > step )); then
    log "remote checkpoint advanced to step $remote_step; leaving local step $step unchanged"
    return 0
  fi
  if (( remote_step == step )); then
    [[ $remote_generation == "$generation" ]] || {
      log "equal-step remote checkpoint fork: $remote_generation versus $generation"
      return 1
    }
    # refresh_remote_checkpoint already proved this exact immutable release,
    # including its bound closure and committed metric prefix.
    return 0
  fi

  remote_upload_generation "$generation" "$manifest_sha256" || return 1
  verify_remote_generation "$generation" "$manifest_sha256" || return 1
  remote_upload_artifacts "$generation" "$manifest_sha256" || return 1
  [[ -n "$UPLOADED_ARTIFACT_MANIFEST_BYTES" \
    && -n "$UPLOADED_ARTIFACT_MANIFEST_SHA256" ]] || return 1
  verify_remote_artifacts "$generation" "$manifest_sha256" \
    "$UPLOADED_ARTIFACT_MANIFEST_BYTES" \
    "$UPLOADED_ARTIFACT_MANIFEST_SHA256" || return 1

  metrics_snapshot=$(mktemp "$STATE_DIR/metrics-prefix.XXXXXX") || return 1
  rm -f -- "$metrics_snapshot"
  checkpoint_tool snapshot-metrics "$OUTPUT/metrics.jsonl" \
    "$metric_records" "$metrics_snapshot" || return 1
  descriptor=$(trainer_generation_descriptor \
    "$OUTPUT/$GENERATIONS_DIRECTORY/$generation" \
    "$generation" "$manifest_sha256" "$metrics_snapshot" true) || return 1
  [[ $descriptor == "$step"$'\t'"$generation"$'\t'"$manifest_sha256"$'\t'"$metric_records" ]] \
    || return 1
  descriptor=$(checkpoint_tool file-descriptor "$metrics_snapshot") || return 1
  IFS=$'\t' read -r metrics_bytes metrics_sha256 <<<"$descriptor"
  [[ -n "$metrics_bytes" && -n "$metrics_sha256" ]] || return 1
  metrics_relative="$CHECKPOINT_METRICS_DIRECTORY/$generation/metrics.jsonl"
  remote_upload_immutable_file "$metrics_snapshot" "$metrics_relative" \
    "$metrics_bytes" "$metrics_sha256" || return 1

  remote_pointer=$(mktemp "$STATE_DIR/remote-current.XXXXXX") || return 1
  checkpoint_tool make-remote-pointer "$pointer_snapshot" "$step" \
    "$metric_records" "$metrics_relative" "$metrics_bytes" "$metrics_sha256" \
    "$UPLOADED_ARTIFACT_MANIFEST_BYTES" \
    "$UPLOADED_ARTIFACT_MANIFEST_SHA256" >"$remote_pointer" || return 1

  # The trainer may have started publishing another checkpoint while the
  # upload was in flight. Publish only the pointer snapshot whose exact,
  # immutable generation was uploaded and verified.
  descriptor=$(checkpoint_descriptor "$OUTPUT" 2>/dev/null) || {
    log "checkpoint changed during upload; leaving remote current.json unchanged"
    return 1
  }
  IFS=$'\t' read -r after_step after_generation \
    after_manifest_sha256 _ <<<"$descriptor"
  if [[ $after_generation != "$generation" \
    || $after_manifest_sha256 != "$manifest_sha256" ]]; then
    log "checkpoint advanced from $step to $after_step during upload; retrying later"
    return 1
  fi
  pointer_result=$(publish_remote_pointer "$remote_pointer") || return 1
  case "$pointer_result" in
    published) log "published checkpoint step $step to $REMOTE" ;;
    same) log "checkpoint step $step was already published to $REMOTE" ;;
    newer) log "remote checkpoint advanced while step $step uploaded; leaving it unchanged" ;;
    *) return 1 ;;
  esac
)

validate_wandb() {
  [[ -n "$WANDB_ENV" ]] || return 0
  [[ -r "$WANDB_ENV" ]] || die "W&B environment is not readable: $WANDB_ENV"
  [[ -r "$WANDB_SCRIPT" ]] || die "W&B reporter is not readable: $WANDB_SCRIPT"
  command -v "$WANDB_PYTHON" >/dev/null 2>&1 \
    || die "W&B Python is unavailable: $WANDB_PYTHON"
  if ! (
    set -a
    # shellcheck source=/dev/null
    source "$WANDB_ENV"
    set +a
    [[ -n ${WANDB_API_KEY:-} ]] && "$WANDB_PYTHON" -c 'import wandb'
  ); then
    die "W&B is configured but WANDB_API_KEY or the wandb package is unavailable"
  fi
}

wandb_supervisor() {
  exec 9>&-
  [[ -z "$LOCAL_REMOTE_FD" ]] || exec 6<&-
  local reporter_pid='' reporter_status
  trap '[[ -z $reporter_pid ]] || kill "$reporter_pid" 2>/dev/null; wait "$reporter_pid" 2>/dev/null || true; exit 0' TERM INT
  set -a
  # shellcheck source=/dev/null
  source "$WANDB_ENV"
  set +a
  export PYTHONUNBUFFERED=1
  while true; do
    "$WANDB_PYTHON" "$WANDB_SCRIPT" "$OUTPUT/metrics.jsonl" &
    reporter_pid=$!
    set +e
    wait "$reporter_pid"
    reporter_status=$?
    set -e
    reporter_pid=
    log "W&B reporter exited with status $reporter_status; restarting in ${WANDB_RESTART_DELAY}s"
    sleep "$WANDB_RESTART_DELAY"
  done
}

sync_supervisor() {
  exec 9>&-
  local child_pid='' sync_status
  trap '[[ -z $child_pid ]] || kill "$child_pid" 2>/dev/null; wait "$child_pid" 2>/dev/null || true; exit 0' TERM INT
  while true; do
    sync_checkpoint_once &
    child_pid=$!
    set +e
    wait "$child_pid"
    sync_status=$?
    set -e
    child_pid=
    if (( sync_status != 0 )); then
      log "checkpoint sync failed; retrying in ${SYNC_INTERVAL}s"
    fi
    sleep "$SYNC_INTERVAL" &
    child_pid=$!
    wait "$child_pid" || true
    child_pid=
  done
}

TRAIN_PID=
SYNC_PID=
WANDB_PID=

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  set +e
  if [[ -n "$TRAIN_PID" ]]; then
    kill "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
  fi
  if [[ -n "$WANDB_PID" ]]; then
    kill "$WANDB_PID" 2>/dev/null
    wait "$WANDB_PID" 2>/dev/null
  fi
  if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID" 2>/dev/null
    wait "$SYNC_PID" 2>/dev/null
  fi
  if [[ -n "$REMOTE" ]]; then
    sync_checkpoint_once >>"$SYNC_LOG" 2>&1
  fi
  clear_remote_snapshot
  rm -f -- "$STATE_DIR/supervisor.pid"
  [[ $LOCK_TOOL != shlock ]] || rm -f -- "$LOCK_FILE"
  exit "$status"
}

trap cleanup EXIT
trap 'exit 143' TERM INT

validate_wandb
INITIAL_RESUME=false
INITIAL_RESUME_STEP=
if prepare_checkpoint; then
  INITIAL_RESUME=true
  INITIAL_RESUME_STEP=$RESUME_STEP
fi
if [[ -n "$REMOTE" ]]; then
  sync_supervisor >>"$SYNC_LOG" 2>&1 &
  SYNC_PID=$!
fi
if [[ -n "$WANDB_ENV" ]]; then
  wandb_supervisor >>"$WANDB_LOG" 2>&1 &
  WANDB_PID=$!
  log "W&B reporter is supervised (log: $WANDB_LOG)"
else
  log "W&B reporting is disabled; set HERMES_TRAIN_WANDB_ENV to enable it"
fi

restart_count=0
first_launch=true
while true; do
  if [[ $first_launch == true ]]; then
    first_launch=false
    RESUME_STEP=$INITIAL_RESUME_STEP
    resume_checkpoint=$INITIAL_RESUME
  elif prepare_checkpoint; then
    resume_checkpoint=true
  else
    resume_checkpoint=false
  fi
  if [[ $resume_checkpoint == true ]]; then
    trainer=("${HERMES_TRAIN_COMMAND[@]}" --output "$OUTPUT" --resume)
    log "launching training from checkpoint step $RESUME_STEP"
  else
    trainer=("${HERMES_TRAIN_COMMAND[@]}" --output "$OUTPUT")
    log "launching training from scratch"
  fi

  (
    exec 9>&-
    [[ -z "$LOCAL_REMOTE_FD" ]] || exec 6<&-
    exec "${trainer[@]}"
  ) >>"$TRAIN_LOG" 2>&1 &
  TRAIN_PID=$!
  set +e
  wait "$TRAIN_PID"
  trainer_status=$?
  set -e
  TRAIN_PID=

  if [[ -n "$REMOTE" ]]; then
    sync_checkpoint_once >>"$SYNC_LOG" 2>&1 || true
  fi
  if (( trainer_status == 0 )); then
    log "training completed successfully"
    [[ -z "$WANDB_PID" || $WANDB_FLUSH_DELAY -eq 0 ]] || sleep "$WANDB_FLUSH_DELAY"
    exit 0
  fi

  (( restart_count += 1 ))
  log "trainer exited with status $trainer_status (restart $restart_count)"
  if (( MAX_RESTARTS > 0 && restart_count > MAX_RESTARTS )); then
    die "trainer exceeded HERMES_TRAIN_MAX_RESTARTS=$MAX_RESTARTS"
  fi
  sleep "$RESTART_DELAY"
done
