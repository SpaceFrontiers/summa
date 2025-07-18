#!/usr/bin/env bash
# Supervise a hermes-train run across process failures and machine reboots.
#
# The single argument is a trusted Bash configuration file. See
# relaunch.conf.example for the supported settings.

set -Eeuo pipefail
umask 077

RELAUNCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly RELAUNCH_SCRIPT_DIR
readonly -a CHECKPOINT_FILES=(
  weights.safetensors
  adamw-state.bpk
  muon-state.bpk
  training-state.json
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

read_checkpoint_step() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    state = json.load(handle)
step = state.get("step")
if not isinstance(step, int) or isinstance(step, bool) or step < 0:
    raise SystemExit("training-state.json has an invalid step")
print(step)
PY
}

checkpoint_step() {
  local directory=$1
  local file
  [[ ! -e "$directory/.checkpoint-in-progress" ]] || return 1
  for file in "${CHECKPOINT_FILES[@]}"; do
    [[ -s "$directory/$file" ]] || return 1
  done
  read_checkpoint_step "$directory/training-state.json"
}

checkpoint_artifacts_exist() {
  local file
  [[ -e "$OUTPUT/.checkpoint-in-progress" ]] && return 0
  for file in "${CHECKPOINT_FILES[@]}"; do
    [[ -e "$OUTPUT/$file" || -e "$OUTPUT/$file.tmp" ]] && return 0
  done
  return 1
}

remote_path() {
  printf '%s/%s' "${REMOTE%/}" "${1#/}"
}

local_remote_root() {
  printf '%s' "${REMOTE#file://}"
}

remote_download() {
  local relative=$1
  local destination=$2
  if [[ $REMOTE == file://* ]]; then
    cp -- "$(local_remote_root)/$relative" "$destination"
  else
    "$GCLOUD_BIN" storage cp "$(remote_path "$relative")" "$destination"
  fi
}

remote_upload_file() {
  local source=$1
  local relative=$2
  if [[ $REMOTE == file://* ]]; then
    mkdir -p -- "$(dirname -- "$(local_remote_root)/$relative")"
    cp -- "$source" "$(local_remote_root)/$relative"
  else
    "$GCLOUD_BIN" storage cp "$source" "$(remote_path "$relative")"
  fi
}

remote_upload_directory() {
  local relative=$1
  shift
  if [[ $REMOTE == file://* ]]; then
    local destination
    destination="$(local_remote_root)/$relative"
    mkdir -p -- "$destination"
    cp -- "$@" "$destination/"
  else
    "$GCLOUD_BIN" storage cp "$@" "$(remote_path "$relative")/"
  fi
}

remote_promote_checkpoint() {
  local step=$1
  if [[ $REMOTE == file://* ]]; then
    cp -- "$(local_remote_root)/checkpoints/$step/training-state.json" \
      "$(local_remote_root)/latest.json"
  else
    "$GCLOUD_BIN" storage cp \
      "$(remote_path "checkpoints/$step/training-state.json")" \
      "$(remote_path latest.json)"
  fi
}

REMOTE_STEP=
refresh_remote_checkpoint() {
  local temporary
  REMOTE_STEP=
  [[ -n "$REMOTE" ]] || return 1
  temporary=$(mktemp "$STATE_DIR/remote-state.XXXXXX")
  if remote_download latest.json "$temporary" >/dev/null 2>&1; then
    if REMOTE_STEP=$(read_checkpoint_step "$temporary" 2>/dev/null); then
      rm -f -- "$temporary"
      return 0
    fi
  fi
  rm -f -- "$temporary"
  return 1
}

restore_remote_checkpoint() {
  local expected_step=$1
  local restore_dir file downloaded_step
  restore_dir=$(mktemp -d "$STATE_DIR/restore.XXXXXX")

  for file in "${CHECKPOINT_FILES[@]}"; do
    if ! remote_download "checkpoints/$expected_step/$file" \
      "$restore_dir/$file" >>"$SYNC_LOG" 2>&1; then
      log "remote checkpoint $expected_step is incomplete ($file is unavailable)"
      rm -rf -- "$restore_dir"
      return 1
    fi
  done
  if ! downloaded_step=$(checkpoint_step "$restore_dir") \
    || [[ $downloaded_step != "$expected_step" ]]; then
    log "remote checkpoint manifest says $expected_step but its state is invalid"
    rm -rf -- "$restore_dir"
    return 1
  fi

  # Keep the marker present until all data files and training-state.json have
  # been atomically published. A reboot during restore is therefore detected.
  printf '%s\n' "$expected_step" >"$OUTPUT/.checkpoint-in-progress"
  for file in weights.safetensors adamw-state.bpk muon-state.bpk; do
    mv -- "$restore_dir/$file" "$OUTPUT/$file.restore"
    mv -f -- "$OUTPUT/$file.restore" "$OUTPUT/$file"
  done
  mv -- "$restore_dir/training-state.json" "$OUTPUT/training-state.json.restore"
  mv -f -- "$OUTPUT/training-state.json.restore" "$OUTPUT/training-state.json"
  rm -f -- "$OUTPUT/.checkpoint-in-progress"
  rmdir -- "$restore_dir"
  log "restored remote checkpoint at step $expected_step"
}

RESUME_STEP=
prepare_checkpoint() {
  local local_step=
  local remote_available=false
  RESUME_STEP=

  if local_step=$(checkpoint_step "$OUTPUT" 2>/dev/null); then
    log "found complete local checkpoint at step $local_step"
  fi
  if [[ -n "$REMOTE" ]] && refresh_remote_checkpoint; then
    remote_available=true
    log "found remote checkpoint at step $REMOTE_STEP"
  fi

  if [[ $remote_available == true \
    && ( -z "$local_step" || $REMOTE_STEP -gt $local_step ) ]]; then
    restore_remote_checkpoint "$REMOTE_STEP" \
      || die "cannot restore the newest remote checkpoint"
    local_step=$REMOTE_STEP
  fi

  if [[ -z "$local_step" ]]; then
    if checkpoint_artifacts_exist; then
      die "local checkpoint is incomplete and no usable remote checkpoint is available"
    fi
    log "no checkpoint found; starting a new run"
    return 1
  fi
  RESUME_STEP=$local_step
}

sync_checkpoint_once() (
  local step after_step remote_step=-1
  local -a sources=()
  local file sync_owner
  exec 9>&-
  [[ -n "$REMOTE" ]] || return 0

  if [[ $LOCK_TOOL == flock ]]; then
    exec 8>"$STATE_DIR/sync.lock"
    flock -n 8 || return 0
  else
    # Bash 3.2 has no BASHPID. A short child observes this subshell as PPID.
    sync_owner=$(sh -c 'printf "%s" "$PPID"')
    shlock -f "$STATE_DIR/sync.lock" -p "$sync_owner" || return 0
    trap 'rm -f -- "$STATE_DIR/sync.lock"' EXIT
  fi

  if [[ -s "$OUTPUT/metrics.jsonl" ]]; then
    remote_upload_file "$OUTPUT/metrics.jsonl" metrics.jsonl || return 1
  fi
  step=$(checkpoint_step "$OUTPUT" 2>/dev/null) || {
    log "checkpoint sync skipped: no complete local checkpoint"
    return 0
  }
  if refresh_remote_checkpoint; then
    remote_step=$REMOTE_STEP
  fi
  if (( remote_step >= step )); then
    return 0
  fi

  for file in "${CHECKPOINT_FILES[@]}"; do
    sources+=("$OUTPUT/$file")
  done
  [[ -s "$OUTPUT/config.json" ]] && sources+=("$OUTPUT/config.json")
  remote_upload_directory "checkpoints/$step" "${sources[@]}" || return 1

  # The trainer may have started publishing another checkpoint while the
  # upload was in flight. Only publish `latest.json` for an unchanged,
  # complete local snapshot; incomplete version directories are never read.
  after_step=$(checkpoint_step "$OUTPUT" 2>/dev/null) || {
    log "checkpoint changed during upload; leaving remote latest unchanged"
    return 1
  }
  if [[ $after_step != "$step" ]]; then
    log "checkpoint advanced from $step to $after_step during upload; retrying later"
    return 1
  fi
  # Promote the immutable state object that was just uploaded. Copying the
  # live local file here would race with the next checkpoint publication.
  remote_promote_checkpoint "$step" || return 1
  log "published checkpoint step $step to $REMOTE"
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
  rm -f -- "$STATE_DIR/supervisor.pid"
  [[ $LOCK_TOOL != shlock ]] || rm -f -- "$LOCK_FILE"
  exit "$status"
}

trap cleanup EXIT
trap 'exit 143' TERM INT

validate_wandb
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
while true; do
  if prepare_checkpoint; then
    trainer=("${HERMES_TRAIN_COMMAND[@]}" --output "$OUTPUT" --resume)
    log "launching training from checkpoint step $RESUME_STEP"
  else
    trainer=("${HERMES_TRAIN_COMMAND[@]}" --output "$OUTPUT")
    log "launching training from scratch"
  fi

  (
    exec 9>&-
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
