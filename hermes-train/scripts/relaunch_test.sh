#!/usr/bin/env bash

set -Eeuo pipefail

TEST_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_SCRIPT_DIR
TEST_ROOT=$(mktemp -d)
readonly TEST_ROOT
trap 'rm -rf -- "$TEST_ROOT"' EXIT

fail() {
  printf 'relaunch_test: %s\n' "$*" >&2
  exit 1
}

write_checkpoint() {
  local directory=$1
  local step=$2
  mkdir -p -- "$directory"
  printf 'weights-%s\n' "$step" >"$directory/weights.safetensors"
  printf 'adamw-%s\n' "$step" >"$directory/adamw-state.bpk"
  printf 'muon-%s\n' "$step" >"$directory/muon-state.bpk"
  printf '{"step":%s}\n' "$step" >"$directory/training-state.json"
}

fake_trainer=$TEST_ROOT/fake-trainer
fake_wandb_python=$TEST_ROOT/fake-wandb-python

cat >"$fake_trainer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output=
resume=false
while (( $# > 0 )); do
  case "$1" in
    --output)
      output=$2
      shift 2
      ;;
    --resume)
      resume=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done
printf '%s\n' "$resume" >>"$TEST_CALLS"
if [[ ${TEST_BLOCK:-false} == true ]]; then
  : >"$TEST_READY"
  while [[ ! -e $TEST_RELEASE ]]; do
    sleep 0.05
  done
  exit 0
fi
if [[ ${TEST_FAIL_ONCE:-false} == true && ! -e $TEST_FAILURE_MARKER ]]; then
  mkdir -p -- "$output"
  printf 'weights-3\n' >"$output/weights.safetensors"
  printf 'adamw-3\n' >"$output/adamw-state.bpk"
  printf 'muon-3\n' >"$output/muon-state.bpk"
  printf '{"step":3}\n' >"$output/training-state.json"
  printf '{"step":3,"loss":1.0}\n' >"$output/metrics.jsonl"
  : >"$TEST_FAILURE_MARKER"
  exit 17
fi
if [[ -n ${TEST_EXPECT_STEP:-} ]]; then
  [[ $resume == true ]] || exit 91
  actual=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' \
    "$output/training-state.json")
  [[ $actual == "$TEST_EXPECT_STEP" ]] || exit 92
fi
EOF

cat >"$fake_wandb_python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == -c ]]; then
  exit 0
fi
printf 'started\n' >>"$TEST_WANDB_CALLS"
trap 'exit 0' TERM INT
while true; do
  sleep 1
done
EOF
chmod +x "$fake_trainer" "$fake_wandb_python"

run_restart_and_reporting_test() {
  local case_root=$TEST_ROOT/restart
  local config=$case_root/relaunch.conf
  mkdir -p -- "$case_root/remote"
  printf 'WANDB_API_KEY=test-only\n' >"$case_root/wandb.env"
  chmod 600 "$case_root/wandb.env"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_SYNC_INTERVAL=1
HERMES_TRAIN_RESTART_DELAY=0
HERMES_TRAIN_MAX_RESTARTS=1
HERMES_TRAIN_WANDB_ENV=$case_root/wandb.env
HERMES_TRAIN_WANDB_PYTHON=$fake_wandb_python
HERMES_TRAIN_WANDB_FLUSH_DELAY=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_FAILURE_MARKER=$case_root/failed-once
  export TEST_WANDB_CALLS=$case_root/wandb-calls
  export TEST_FAIL_ONCE=true
  unset TEST_EXPECT_STEP

  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"
  [[ $(sed -n '1p' "$TEST_CALLS") == false ]] || fail "first launch was not fresh"
  [[ $(sed -n '2p' "$TEST_CALLS") == true ]] || fail "failed trainer was not resumed"
  [[ -s $TEST_WANDB_CALLS ]] || fail "W&B reporter was not launched"
  if [[ ! -s $case_root/remote/checkpoints/3/weights.safetensors ]]; then
    sed -n '1,160p' "$case_root/state/sync.log" >&2 || true
    fail "checkpoint payload was not synced"
  fi
  [[ $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' \
    "$case_root/remote/latest.json") == 3 ]] || fail "latest manifest was not published last"
}

run_remote_restore_test() {
  local case_root=$TEST_ROOT/restore
  local config=$case_root/relaunch.conf
  mkdir -p -- "$case_root/remote/checkpoints/7" "$case_root/output"
  write_checkpoint "$case_root/remote/checkpoints/7" 7
  cp -- "$case_root/remote/checkpoints/7/training-state.json" "$case_root/remote/latest.json"
  printf 'interrupted\n' >"$case_root/output/.checkpoint-in-progress"
  printf 'stale\n' >"$case_root/output/weights.safetensors"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_SYNC_INTERVAL=60
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_EXPECT_STEP=7
  export TEST_FAIL_ONCE=false
  unset TEST_WANDB_CALLS TEST_FAILURE_MARKER

  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"
  [[ $(cat "$case_root/output/weights.safetensors") == weights-7 ]] \
    || fail "remote checkpoint did not replace interrupted local state"
  [[ ! -e $case_root/output/.checkpoint-in-progress ]] \
    || fail "restore marker survived a complete restore"
}

run_newer_local_wins_test() {
  local case_root=$TEST_ROOT/local-wins
  local config=$case_root/relaunch.conf
  write_checkpoint "$case_root/output" 9
  write_checkpoint "$case_root/remote/checkpoints/7" 7
  cp -- "$case_root/remote/checkpoints/7/training-state.json" "$case_root/remote/latest.json"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_SYNC_INTERVAL=60
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_EXPECT_STEP=9
  export TEST_FAIL_ONCE=false
  unset TEST_WANDB_CALLS TEST_FAILURE_MARKER

  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"
  [[ $(cat "$case_root/output/weights.safetensors") == weights-9 ]] \
    || fail "older remote checkpoint overwrote newer local state"
}

run_legacy_remote_migration_test() {
  local case_root=$TEST_ROOT/legacy
  local config=$case_root/relaunch.conf
  write_checkpoint "$case_root/output" 5
  write_checkpoint "$case_root/remote" 5
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_SYNC_INTERVAL=60
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_EXPECT_STEP=5
  export TEST_FAIL_ONCE=false
  unset TEST_WANDB_CALLS TEST_FAILURE_MARKER

  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"
  [[ -s $case_root/remote/checkpoints/5/weights.safetensors ]] \
    || fail "legacy flat checkpoint was not migrated"
  [[ $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' \
    "$case_root/remote/latest.json") == 5 ]] || fail "migrated manifest is invalid"
}

run_idempotent_lock_test() {
  local case_root=$TEST_ROOT/lock
  local config=$case_root/relaunch.conf
  local supervisor_pid ready=false
  mkdir -p -- "$case_root"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_READY=$case_root/ready
  export TEST_RELEASE=$case_root/release
  export TEST_BLOCK=true
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER

  "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/first.log" 2>&1 &
  supervisor_pid=$!
  for _attempt in {1..100}; do
    if [[ -e $TEST_READY ]]; then
      ready=true
      break
    fi
    sleep 0.05
  done
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/second.log" 2>&1
  : >"$TEST_RELEASE"
  wait "$supervisor_pid"

  [[ $ready == true ]] || fail "first supervisor did not launch its trainer"
  [[ $(wc -l <"$TEST_CALLS") -eq 1 ]] || fail "duplicate supervisor launched a trainer"
  grep -q 'another supervisor already owns' "$case_root/second.log" \
    || fail "duplicate supervisor did not report the held lock"
}

run_restart_and_reporting_test
run_remote_restore_test
run_newer_local_wins_test
run_legacy_remote_migration_test
run_idempotent_lock_test
printf 'relaunch_test: ok\n'
