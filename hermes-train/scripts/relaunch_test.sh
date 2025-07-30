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

fake_trainer=$TEST_ROOT/fake-trainer
fake_wandb_python=$TEST_ROOT/fake-wandb-python
fake_checkpoint_writer=$TEST_ROOT/write-checkpoint
fake_gcloud=$TEST_ROOT/fake-gcloud
fake_artifact_writer=$TEST_ROOT/write-artifacts

cat >"$fake_checkpoint_writer" <<'PY'
#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile

root = pathlib.Path(sys.argv[1])
step = int(sys.argv[2])
version = int(sys.argv[3]) if len(sys.argv) > 3 else 2
tag = sys.argv[4] if len(sys.argv) > 4 else str(step)
state_step = int(sys.argv[5]) if len(sys.argv) > 5 else step
root.mkdir(parents=True, exist_ok=True)
generations = root / "generations"
generations.mkdir(exist_ok=True)
staging = pathlib.Path(tempfile.mkdtemp(prefix=".fixture-", dir=generations))
(staging / "weights.safetensors").write_text(f"weights-{tag}\n")
(staging / "adamw-state.bpk").write_text(f"adamw-{tag}\n")
(staging / "muon-state.bpk").write_text(f"muon-{tag}\n")
state = {
    "version": version,
    "global_step": state_step,
    "phase": 0,
    "phase_id": "test",
    "metric_records": 1,
    "optimizer_states": [
        {
            "scope": "wake",
            "adamw": "adamw-state.bpk",
            "muon": "muon-state.bpk",
            "gradient_accumulator": None,
        }
    ],
}
if os.environ.get("TEST_CHECKPOINT_STATE_OVERLAY"):
    state.update(json.loads(pathlib.Path(os.environ["TEST_CHECKPOINT_STATE_OVERLAY"]).read_text()))
(staging / "training-state.json").write_text(
    json.dumps(state, separators=(",", ":"))
)
files = []
for path in sorted(staging.rglob("*")):
    if path.is_file():
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(staging).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
if len(sys.argv) > 6:
    files[0]["path"] = sys.argv[6]
manifest = {
    "version": 1,
    "training_state_version": version,
    "global_step": step,
    "phase": 0,
    "phase_id": "test",
    "files": files,
}
manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
generation = f"sha256-{manifest_sha256}"
(staging / "generation-manifest.json").write_bytes(manifest_bytes)
destination = generations / generation
if destination.exists():
    shutil.rmtree(staging)
else:
    os.replace(staging, destination)
evidence_bytes = json.dumps(
    {"version": 1, "checkpoint_manifest_sha256": manifest_sha256},
    sort_keys=True,
    separators=(",", ":"),
).encode()
evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
evidence_root = root / "training-evidence"
evidence_root.mkdir(exist_ok=True)
(evidence_root / f"sha256-{evidence_sha256}.json").write_bytes(evidence_bytes)
pointer = {
    "version": 1,
    "generation": generation,
    "manifest_sha256": manifest_sha256,
}
pointer_temporary = root / ".current.fixture.tmp"
pointer_temporary.write_text(json.dumps(pointer, separators=(",", ":")))
os.replace(pointer_temporary, root / "current.json")
(root / "metrics.jsonl").write_text(
    json.dumps(
        {
            "schema_version": 2,
            "sequence": 0,
            "emitted_at_unix_ms": 1,
            "run_id": "test",
            "global_step": step,
            "phase": {"index": 0, "name": "test", "kind": "pretrain"},
            "event": {
                "type": "throughput",
                "values": {
                    "optimizer_steps": 1,
                    "compute_tokens": 1,
                    "supervised_tokens": 1,
                    "examples": 1,
                    "elapsed_seconds": 1.0,
                    "tokens_per_second": 1.0,
                    "examples_per_second": 1.0,
                    "input_wait_seconds": 0.0,
                    "host_to_device_seconds": 0.0,
                    "gpu_busy_seconds": 1.0,
                },
            },
        },
        separators=(",", ":"),
    )
    + "\n"
)
PY

write_checkpoint() {
  "$fake_checkpoint_writer" "$@"
}

current_generation() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1] + "/current.json"))["generation"])' "$1"
}

checkpoint_file() {
  local directory=$1
  local file=$2
  local generation
  generation=$(current_generation "$directory")
  printf '%s/%s/%s/%s' "$directory" generations "$generation" "$file"
}

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
  "$TEST_CHECKPOINT_WRITER" "$output" 3
  : >"$TEST_FAILURE_MARKER"
  exit 17
fi
if [[ -n ${TEST_EXPECT_STEP:-} ]]; then
  [[ $resume == true ]] || exit 91
  actual=$(python3 -c 'import json,sys,pathlib; root=pathlib.Path(sys.argv[1]); pointer=json.load(open(root / "current.json")); print(json.load(open(root / "generations" / pointer["generation"] / "training-state.json"))["global_step"])' "$output")
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
cat >"$fake_gcloud" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == storage ]] || exit 64
shift
gcs_path() {
  printf '%s/%s' "$TEST_GCS_ROOT" "${1#gs://}"
}
generation_file() {
  local path=$1 key
  key=$(printf '%s' "$path" | cksum | awk '{print $1 "-" $2}')
  printf '%s/.object-generations/%s' "$TEST_GCS_ROOT" "$key"
}
if [[ ${1:-} == objects && ${2:-} == describe ]]; then
  shift 2
  [[ $# -eq 2 && $2 == --format=value\(generation\) ]] || exit 65
  object=$(gcs_path "$1")
  metadata=$(generation_file "$object")
  [[ -f $object && ! -L $object && -s $metadata ]] || exit 1
  cat "$metadata"
  exit 0
fi
[[ ${1:-} == cp ]] || exit 64
shift
expected_generation=
while [[ ${1:-} == --* ]]; do
  case "$1" in
    --if-generation-match=*) expected_generation=${1#*=} ;;
  esac
  shift
done
[[ $# -eq 2 ]] || exit 65
source_path=$1
destination_path=$2
if [[ $source_path == gs://* ]]; then
  source_path=$(gcs_path "$source_path")
  [[ -f $source_path && ! -L $source_path ]] || exit 1
  cp -- "$source_path" "$destination_path"
else
  destination_path=$(gcs_path "$destination_path")
  if [[ -n ${TEST_GCS_DELAY_CURRENT_STEP:-} \
    && ${destination_path##*/} == current.json ]]; then
    IFS=$'\t' read -r candidate_step candidate_generation < <(
      python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); print(value["global_step"], value["generation"], sep="\t")' "$source_path"
    )
    if [[ $candidate_step == "$TEST_GCS_DELAY_CURRENT_STEP" \
      && ( -z ${TEST_GCS_DELAY_CURRENT_GENERATION:-} \
        || $candidate_generation == "$TEST_GCS_DELAY_CURRENT_GENERATION" ) ]]; then
      : >"$TEST_GCS_DELAY_READY"
      while [[ ! -e $TEST_GCS_DELAY_RELEASE ]]; do
        sleep 0.01
      done
    fi
  fi
  metadata=$(generation_file "$destination_path")
  lock=$metadata.lock
  mkdir -p -- "$(dirname -- "$metadata")"
  for _attempt in {1..1000}; do
    mkdir "$lock" 2>/dev/null && break
    sleep 0.01
  done
  [[ -d $lock ]] || exit 1
  trap 'rmdir -- "$lock" 2>/dev/null || true' EXIT
  current=0
  [[ ! -s $metadata ]] || current=$(<"$metadata")
  if [[ -n $expected_generation && $expected_generation != "$current" ]]; then
    exit 1
  fi
  mkdir -p -- "$(dirname -- "$destination_path")"
  cp -- "$source_path" "$destination_path"
  printf '%s\n' "$((current + 1))" >"$metadata"
  printf 'UPLOAD\t%s\n' "${2#gs://}" >>"$TEST_GCS_LOG"
fi
EOF
cat >"$fake_artifact_writer" <<'PY'
#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
runtime = pathlib.Path(sys.argv[2])
tag = sys.argv[3]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


teacher = output / "sleep-models" / f"teacher-{tag}.safetensors"
teacher_hash = write(teacher, f"teacher-{tag}\n".encode())
journal = output / "sleep-wake-contexts" / f"journal-{tag}.json"
journal_hash = write(journal, canonical({"version": 1, "tag": tag}))

prospective = runtime / "stores" / "prospective" / "model-generations" / f"student-{tag}" / "weights.safetensors"
prospective_hash = write(prospective, f"student-{tag}\n".encode())
candidate = runtime / "stores" / "candidates" / "model-generations" / f"candidate-{tag}" / "weights.safetensors"
candidate_hash = write(candidate, f"candidate-{tag}\n".encode())

tensor_root = runtime / "stores" / "tensor"
tensor_payload = tensor_root / "generations" / "placeholder" / "transaction.json"
tensor_payload_hash = write(tensor_payload, canonical({"version": 2, "tag": tag}))
tensor_manifest_value = {
    "version": 1,
    "txn_id": 41,
    "files": [
        {
            "path": "transaction.json",
            "bytes": tensor_payload.stat().st_size,
            "sha256": tensor_payload_hash,
        }
    ],
}
tensor_manifest_bytes = canonical(tensor_manifest_value)
tensor_manifest_raw = hashlib.sha256(tensor_manifest_bytes).hexdigest()
tensor_generation = f"sha256-{tensor_manifest_raw}"
tensor_generation_root = tensor_root / "generations" / tensor_generation
tensor_generation_root.mkdir(parents=True, exist_ok=True)
tensor_payload.replace(tensor_generation_root / "transaction.json")
(tensor_root / "generations" / "placeholder").rmdir()
tensor_manifest = tensor_generation_root / "manifest.json"
write(tensor_manifest, tensor_manifest_bytes)

tier_root = runtime / "stores" / "optimizers"
tier_payload = tier_root / "generations" / "placeholder" / "optimizer.bpk"
tier_payload_hash = write(tier_payload, f"optimizer-{tag}\n".encode())
tier_manifest_value = {
    "version": 1,
    "tier": 0,
    "files": [
        {
            "path": "optimizer.bpk",
            "bytes": tier_payload.stat().st_size,
            "sha256": tier_payload_hash,
        }
    ],
}
tier_manifest_bytes = canonical(tier_manifest_value)
tier_manifest_raw = hashlib.sha256(tier_manifest_bytes).hexdigest()
tier_generation = f"sha256-{tier_manifest_raw}"
tier_generation_root = tier_root / "generations" / tier_generation
tier_generation_root.mkdir(parents=True, exist_ok=True)
tier_payload.replace(tier_generation_root / "optimizer.bpk")
(tier_root / "generations" / "placeholder").rmdir()
tier_manifest = tier_generation_root / "manifest.json"
write(tier_manifest, tier_manifest_bytes)

dream_root = runtime / "stores" / "dreams"
runtime_value = json.loads((runtime / "sleep-runtime.json").read_text())
dream_candidate_value = {"version": 1, "transaction_id": 41, "token_ids": [1, 2]}
dream_candidate_bytes = canonical(dream_candidate_value)
dream_candidate_hash = "sha256:" + hashlib.sha256(dream_candidate_bytes).hexdigest()
write(
    dream_root / "candidates" / f"{dream_candidate_hash[7:]}.json",
    dream_candidate_bytes,
)
initial_policy = runtime_value.get("dreaming", {}).get("initial_policy")
if initial_policy is None:
    parent_policy_adapter_bytes = f"parent-policy-adapter-{tag}\n".encode()
    parent_policy_adapter_hash = (
        "sha256:" + hashlib.sha256(parent_policy_adapter_bytes).hexdigest()
    )
    write(
        dream_root / "policy-adapters" / f"{parent_policy_adapter_hash[7:]}.bin",
        parent_policy_adapter_bytes,
    )
    parent_policy_value = {
        "version": 1,
        "transaction_id": 40,
        "adapter_sha256": parent_policy_adapter_hash,
        "parent_policy_sha256": None,
        "parent_adapter_sha256": None,
        "accepted_adapters": [],
    }
    parent_policy_bytes = canonical(parent_policy_value)
    parent_policy_hash = "sha256:" + hashlib.sha256(parent_policy_bytes).hexdigest()
    parent_policy_path = dream_root / "policies" / f"{parent_policy_hash[7:]}.json"
    write(parent_policy_path, parent_policy_bytes)
else:
    parent_policy_path = runtime / initial_policy["path"]
    parent_policy_bytes = parent_policy_path.read_bytes()
    parent_policy_hash = initial_policy["sha256"]
    if "sha256:" + hashlib.sha256(parent_policy_bytes).hexdigest() != parent_policy_hash:
        raise SystemExit("external initial policy fixture digest mismatch")
    parent_policy_value = json.loads(parent_policy_bytes)
    parent_policy_adapter_hash = parent_policy_value["adapter_sha256"]
dream_manifest_value = {
    "version": 1,
    "transaction_id": 41,
    "generation_policy_sha256": parent_policy_hash,
    "generation_policy_adapter_sha256": parent_policy_adapter_hash,
    "dreams": [{"id": "dream-41", "artifact_hash": dream_candidate_hash}],
}
dream_manifest_bytes = canonical(dream_manifest_value)
dream_manifest_hash = "sha256:" + hashlib.sha256(dream_manifest_bytes).hexdigest()
write(
    dream_root / "manifests" / f"{dream_manifest_hash[7:]}.json",
    dream_manifest_bytes,
)
adapter_bytes = f"adapter-{tag}\n".encode()
adapter_hash = "sha256:" + hashlib.sha256(adapter_bytes).hexdigest()
write(dream_root / "adapters" / f"{adapter_hash[7:]}.bin", adapter_bytes)
policy_adapter_bytes = f"policy-adapter-{tag}\n".encode()
policy_adapter_hash = "sha256:" + hashlib.sha256(policy_adapter_bytes).hexdigest()
write(
    dream_root / "policy-adapters" / f"{policy_adapter_hash[7:]}.bin",
    policy_adapter_bytes,
)
policy_bytes = canonical(
    {
        "version": 1,
        "transaction_id": 41,
        "adapter_sha256": policy_adapter_hash,
        "parent_policy_sha256": parent_policy_hash,
        "parent_adapter_sha256": parent_policy_adapter_hash,
        "accepted_adapters": [adapter_hash],
    }
)
policy_hash = "sha256:" + hashlib.sha256(policy_bytes).hexdigest()
write(dream_root / "policies" / f"{policy_hash[7:]}.json", policy_bytes)

qat_root = output / "quantized-candidates" / f"candidate-{tag}"
qat_weights = qat_root / "weights.safetensors"
qat_weights_hash = write(qat_weights, f"qat-weights-{tag}\n".encode())
packed = qat_root / "hquant" / "quantized" / "matrix.hquant"
packed_hash = write(packed, f"packed-{tag}\n".encode())
archive_value = {
    "version": 1,
    "base_checkpoint_hash": qat_weights_hash,
    "matrices": [
        {
            "file": "quantized/matrix.hquant",
            "packed_bytes": packed.stat().st_size,
            "sha256": packed_hash,
        }
    ],
    "floating_tensors": [],
}
archive_bytes = canonical(archive_value)
archive_manifest = qat_root / "hquant" / "manifest.json"
archive_hash = write(archive_manifest, archive_bytes)
candidate_value = {
    "version": 1,
    "candidate_key": f"candidate-{tag}",
    "weights_file": "weights.safetensors",
    "weights_bytes": qat_weights.stat().st_size,
    "weights_sha256": qat_weights_hash,
    "archive_directory": "hquant",
    "archive_manifest": "hquant/manifest.json",
    "archive_manifest_sha256": archive_hash,
}
candidate_bytes = canonical(candidate_value)
candidate_manifest = qat_root / "candidate.json"
candidate_manifest_hash = write(candidate_manifest, candidate_bytes)

# These are deliberately unreferenced and may be newer than the checkpoint.
# A correct closure never captures them or a mutable convenience pointer.
write(output / "sleep-models" / f"future-{tag}.safetensors", b"future-model\n")
write(runtime / "stores" / "rejections" / f"future-{tag}.json", b"future-report\n")

overlay = {
    "artifacts": [
        {
            "kind": "hquant_candidate",
            "manifest": str(candidate_manifest),
            "hash": candidate_manifest_hash,
        }
    ],
    "quantization": {
        "format": "ternary_g128",
        "fake_quant_active": False,
        "calibration_step": 1,
        "manifest": str(candidate_manifest),
        "teacher_hash": None,
        "transaction": None,
    },
    "sleep": {
        "input_checkpoint": {"uri": str(teacher), "sha256": teacher_hash},
        "live_checkpoint": {"uri": str(candidate), "sha256": candidate_hash},
        "wake_context_journal": {"path": str(journal), "sha256": journal_hash},
        "sleep": {
            "pending": {
                "teacher_checkpoint": str(teacher),
                "teacher_hash": teacher_hash,
                "student_checkpoint": str(prospective),
                "student_hash": prospective_hash,
                "candidate_checkpoint": str(candidate),
                "candidate_hash": candidate_hash,
                "tensor_transaction_generation": tensor_generation,
                "tensor_transaction_manifest_hash": f"sha256:{tensor_manifest_raw}",
                "generated_manifest": dream_manifest_hash,
                "dream_trials": [{"adapter_hash": adapter_hash}],
                "dream_policy_receipt": policy_hash,
            },
            "completed_transactions": [],
        },
        "optimizer_scopes": {
            "tiers": [
                {
                    "artifact": {
                        "state_uri": str(tier_manifest),
                        "manifest_hash": f"sha256:{tier_manifest_raw}",
                    }
                }
            ]
        },
    },
}
history_count = int(os.environ.get("TEST_DREAM_HISTORY_COUNT", "0"))
if history_count:
    history = []
    for ordinal in range(history_count):
        historical_candidate = canonical(
            {"version": 1, "transaction_id": ordinal, "token_ids": [ordinal, 7]}
        )
        historical_candidate_hash = (
            "sha256:" + hashlib.sha256(historical_candidate).hexdigest()
        )
        historical_candidate_path = (
            dream_root / "candidates" / f"{historical_candidate_hash[7:]}.json"
        )
        write(historical_candidate_path, historical_candidate)
        historical_manifest = canonical(
            {
                "version": 1,
                "transaction_id": ordinal,
                "generation_policy_sha256": None,
                "generation_policy_adapter_sha256": None,
                "dreams": [
                    {
                        "id": f"historical-dream-{ordinal}",
                        "artifact_hash": historical_candidate_hash,
                    }
                ],
            }
        )
        historical_manifest_hash = (
            "sha256:" + hashlib.sha256(historical_manifest).hexdigest()
        )
        historical_manifest_path = (
            dream_root / "manifests" / f"{historical_manifest_hash[7:]}.json"
        )
        write(historical_manifest_path, historical_manifest)
        history.append(
            (
                historical_manifest_hash,
                historical_manifest_path,
                historical_candidate_path,
            )
        )
    sleep_state = overlay["sleep"]["sleep"]
    sleep_state["artifact_manifests"] = [item[0] for item in history] + [
        dream_manifest_hash
    ]
    sleep_state["completed_transactions"] = [
        {"generated_manifest": item[0], "dream_trials": []}
        for item in history[-64:]
    ]
overlay_path = runtime / f"state-overlay-{tag}.json"
overlay_path.write_bytes(canonical(overlay))
expected = {
    "qat": str(packed),
    "tensor": str(tensor_generation_root / "transaction.json"),
    "tier": str(tier_generation_root / "optimizer.bpk"),
    "dream": str(dream_root / "policies" / f"{policy_hash[7:]}.json"),
    "dream_parent": str(parent_policy_path),
    "dream_policy_adapter": str(
        dream_root / "policy-adapters" / f"{policy_adapter_hash[7:]}.bin"
    ),
    "model": str(teacher),
    "journal": str(journal),
    "future_model": str(output / "sleep-models" / f"future-{tag}.safetensors"),
}
if history_count:
    expected["oldest_dream_manifest"] = str(history[0][1])
    expected["oldest_dream_candidate"] = str(history[0][2])
(runtime / f"expected-{tag}.json").write_bytes(canonical(expected))
print(overlay_path)
PY
chmod +x "$fake_trainer" "$fake_wandb_python" "$fake_checkpoint_writer" "$fake_gcloud" "$fake_artifact_writer"
export TEST_CHECKPOINT_WRITER=$fake_checkpoint_writer

write_sleep_runtime() {
  local case_root=$1
  local runtime_root=$case_root/runtime
  local runtime=$runtime_root/sleep-runtime.json
  mkdir -p -- "$runtime_root/stores/tensor" "$runtime_root/stores/prospective" \
    "$runtime_root/stores/optimizers" "$runtime_root/stores/candidates" \
    "$runtime_root/stores/rejections" "$runtime_root/stores/dreams"
  python3 - "$runtime_root" "$runtime" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
runtime = pathlib.Path(sys.argv[2])
dreaming = {"artifact_directory": "stores/dreams"}
if os.environ.get("TEST_EXTERNAL_INITIAL_POLICY") == "true":
    adapter = b"deployment-initial-policy-adapter\n"
    adapter_sha256 = "sha256:" + hashlib.sha256(adapter).hexdigest()
    adapter_path = root / "stores" / "dreams" / "policy-adapters" / f"{adapter_sha256[7:]}.bin"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_bytes(adapter)
    policy = json.dumps(
        {
            "version": 1,
            "transaction_id": 0,
            "adapter_sha256": adapter_sha256,
            "parent_policy_sha256": None,
            "parent_adapter_sha256": None,
            "accepted_adapters": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    policy_path = root / "deployment" / "initial-policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes(policy)
    dreaming["initial_policy"] = {
        "path": "deployment/initial-policy.json",
        "sha256": "sha256:" + hashlib.sha256(policy).hexdigest(),
    }
value = {
    "tensor_transaction_directory": "stores/tensor",
    "prospective_directory": "stores/prospective",
    "tier_optimizer_directory": "stores/optimizers",
    "candidate_directory": "stores/candidates",
    "rejection_report_directory": "stores/rejections",
    "dreaming": dreaming,
}
runtime.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
PY
  printf '%s\t%s' "$runtime" \
    "$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$runtime")"
}

publish_remote_checkpoint() {
  local case_root=$1
  local step=$2
  local extra_command=${3:-}
  local source_output=$case_root/seed-output
  local config=$case_root/seed-relaunch.conf
  local expected_generation
  [[ -s $source_output/current.json ]] || write_checkpoint "$source_output" "$step"
  expected_generation=$(current_generation "$source_output")
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$source_output
HERMES_TRAIN_STATE_DIR=$case_root/seed-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $extra_command)
HERMES_TRAIN_SYNC_INTERVAL=60
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/seed-calls
  export TEST_READY=$case_root/seed-ready
  export TEST_RELEASE=$case_root/seed-release
  rm -f -- "$TEST_READY" "$TEST_RELEASE"
  export TEST_BLOCK=true
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/seed.log" 2>&1 &
  local supervisor_pid=$!
  local published=false
  for _attempt in {1..1200}; do
    if [[ -s $case_root/remote/current.json ]] \
      && [[ $(current_generation "$case_root/remote" 2>/dev/null) == "$expected_generation" ]]; then
      published=true
      break
    fi
    sleep 0.05
  done
  : >"$TEST_RELEASE"
  wait "$supervisor_pid"
  unset TEST_BLOCK TEST_READY TEST_RELEASE
  [[ $published == true ]] || {
    sed -n '1,240p' "$case_root/seed-state/sync.log" >&2 || true
    fail "fixture checkpoint was not published with its artifact closure"
  }
}

PREPARED_RUNTIME=
PREPARED_RUNTIME_SHA256=
PREPARED_COMMAND=
PREPARED_EXPECTED=
prepare_artifact_checkpoint() {
  local case_root=$1
  local step=$2
  local tag=$3
  local descriptor overlay generation manifest_sha256 evidence evidence_sha256
  descriptor=$(write_sleep_runtime "$case_root")
  IFS=$'\t' read -r PREPARED_RUNTIME PREPARED_RUNTIME_SHA256 <<<"$descriptor"
  overlay=$(
    "$fake_artifact_writer" "$case_root/seed-output" "$case_root/runtime" "$tag"
  )
  export TEST_CHECKPOINT_STATE_OVERLAY=$overlay
  write_checkpoint "$case_root/seed-output" "$step"
  unset TEST_CHECKPOINT_STATE_OVERLAY
  generation=$(current_generation "$case_root/seed-output")
  manifest_sha256=${generation#sha256-}
  evidence=$(python3 - "$case_root/seed-output" "$manifest_sha256" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "training-evidence"
root.mkdir(parents=True, exist_ok=True)
payload = json.dumps(
    {"version": 1, "checkpoint_manifest_sha256": sys.argv[2]},
    sort_keys=True,
    separators=(",", ":"),
).encode()
digest = hashlib.sha256(payload).hexdigest()
path = root / f"sha256-{digest}.json"
path.write_bytes(payload)
print(path)
PY
  )
  evidence_sha256=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$evidence")
  [[ $(basename -- "$evidence") == sha256-$evidence_sha256.json ]] \
    || fail "training-evidence fixture is not content addressed"

  # Simulate G+1 advancing convenience state after sealed checkpoint G. The G
  # closure must select its recorded immutable generation and never this file.
  printf '{"version":2,"generation":"future"}\n' \
    >"$case_root/runtime/stores/tensor/current.json"
  printf 'future-after-checkpoint\n' \
    >"$case_root/runtime/stores/optimizers/future-after-$tag.bpk"

  PREPARED_COMMAND="--sleep-runtime $PREPARED_RUNTIME --sleep-runtime-sha256 sha256:$PREPARED_RUNTIME_SHA256"
  PREPARED_EXPECTED=$case_root/runtime/expected-$tag.json
}

remote_object_for_root() {
  local remote=$1
  local generation=$2
  local root_id=$3
  python3 - "$remote" "$generation" "$root_id" <<'PY'
import json
import pathlib
import sys

remote = pathlib.Path(sys.argv[1])
generation = sys.argv[2]
root_id = sys.argv[3]
manifest = json.loads(
    (remote / "checkpoint-artifacts" / generation / "artifact-manifest.json").read_text()
)
root = next(item for item in manifest["roots"] if item["id"] == root_id)
entry = root["files"][0]
digest = entry["sha256"]
print(remote / "checkpoint-objects" / "sha256" / digest[:2] / digest)
PY
}

artifact_manifest_for() {
  local remote=$1
  local generation=$2
  printf '%s/checkpoint-artifacts/%s/artifact-manifest.json' "$remote" "$generation"
}

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
  local remote_generation=''
  for _attempt in {1..100}; do
    if [[ -s $case_root/remote/current.json ]]; then
      remote_generation=$(current_generation "$case_root/remote")
      break
    fi
    sleep 0.05
  done
  [[ -n $remote_generation ]] || {
    sed -n '1,160p' "$case_root/state/sync.log" >&2 || true
    fail "checkpoint pointer was not synced"
  }
  if [[ ! -s $case_root/remote/generations/$remote_generation/weights.safetensors ]]; then
    sed -n '1,160p' "$case_root/state/sync.log" >&2 || true
    fail "checkpoint generation was not synced"
  fi
  [[ -s $case_root/remote/generations/$remote_generation/generation-manifest.json ]] \
    || fail "generation manifest was not synced"
  [[ -s $case_root/remote/current.json ]] || fail "current pointer was not published"
  local metrics_path
  metrics_path=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metrics_path"])' \
    "$case_root/remote/current.json")
  [[ -s $case_root/remote/$metrics_path ]] \
    || fail "immutable checkpoint metric prefix was not synced"
  [[ ! -e $case_root/remote/metrics.jsonl ]] \
    || fail "mutable root metric journal was published"
  [[ ! -e $case_root/remote/latest.json && ! -e $case_root/remote/checkpoints ]] \
    || fail "obsolete flat remote checkpoint layout was published"
}

run_remote_restore_test() {
  local case_root=$TEST_ROOT/restore
  local config=$case_root/relaunch.conf
  publish_remote_checkpoint "$case_root" 7
  mkdir -p -- "$case_root/output/generations/.staging-interrupted"
  printf 'incomplete\n' >"$case_root/output/current.json"
  printf 'stale metrics\n' >"$case_root/output/metrics.jsonl"
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
  [[ $(cat "$(checkpoint_file "$case_root/output" weights.safetensors)") == weights-7 ]] \
    || fail "remote checkpoint did not replace interrupted local state"
  [[ $(cat "$case_root/output/metrics.jsonl") == *'"global_step":7'* ]] \
    || fail "remote root metric journal was not restored"
}

run_newer_local_wins_test() {
  local case_root=$TEST_ROOT/local-wins
  local config=$case_root/relaunch.conf
  write_checkpoint "$case_root/output" 9
  publish_remote_checkpoint "$case_root" 7
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
  [[ $(cat "$(checkpoint_file "$case_root/output" weights.safetensors)") == weights-9 ]] \
    || fail "older remote checkpoint overwrote newer local state"
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

run_version_one_checkpoint_rejected_test() {
  local case_root=$TEST_ROOT/version-one
  local config=$case_root/relaunch.conf
  mkdir -p -- "$case_root"
  write_checkpoint "$case_root/output" 11 1
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER TEST_BLOCK

  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1; then
    fail "version-one checkpoint was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] || fail "trainer launched with a version-one checkpoint"
  grep -q 'local checkpoint is incomplete' "$case_root/log" \
    || fail "version-one checkpoint failure was not explained"
}

run_modified_generation_rejected_test() {
  local case_root=$TEST_ROOT/modified-generation
  local config=$case_root/relaunch.conf
  write_checkpoint "$case_root/output" 13
  printf 'tampered\n' >>"$(checkpoint_file "$case_root/output" weights.safetensors)"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER TEST_BLOCK

  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1; then
    fail "modified generation was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] || fail "trainer launched with a modified generation"
  grep -q 'local checkpoint is incomplete' "$case_root/log" \
    || fail "modified generation failure was not explained"
}

run_global_step_mismatch_rejected_test() {
  local case_root=$TEST_ROOT/global-step-mismatch
  local config=$case_root/relaunch.conf
  write_checkpoint "$case_root/output" 19 2 mismatch 18
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER TEST_BLOCK

  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1; then
    fail "manifest/training-state global_step mismatch was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] \
    || fail "trainer launched with mismatched checkpoint global_step values"
}

run_unsafe_pointer_rejected_test() {
  local case_root=$TEST_ROOT/unsafe-pointer
  local config=$case_root/relaunch.conf
  write_checkpoint "$case_root/output" 15
  printf '{"version":1,"generation":"../escape","manifest_sha256":"%064d"}\n' 0 \
    >"$case_root/output/current.json"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER TEST_BLOCK

  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1; then
    fail "unsafe current pointer was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] || fail "trainer launched with an unsafe current pointer"
}

run_unsafe_manifest_path_rejected_test() {
  local case_root=$TEST_ROOT/unsafe-manifest
  local config=$case_root/relaunch.conf
  write_checkpoint "$case_root/output" 21 2 unsafe 21 ../escape.bpk
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER TEST_BLOCK

  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1; then
    fail "unsafe manifest path was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] || fail "trainer launched with an unsafe manifest path"
}

run_corrupt_remote_rejected_test() {
  local case_root=$TEST_ROOT/corrupt-remote
  local config=$case_root/relaunch.conf
  publish_remote_checkpoint "$case_root" 17
  printf 'tampered\n' >>"$(checkpoint_file "$case_root/remote" weights.safetensors)"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER TEST_BLOCK

  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1; then
    fail "corrupt remote checkpoint was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] || fail "trainer launched with a corrupt remote checkpoint"
  grep -q 'remote checkpoint is incomplete' "$case_root/log" \
    || fail "corrupt remote checkpoint failure was not explained"
}

run_training_evidence_rejected_test() {
  local mode=$1
  local case_root=$TEST_ROOT/training-evidence-$mode
  local config=$case_root/relaunch.conf
  local generation supervisor_pid observed=false
  write_checkpoint "$case_root/output" 23 2 "evidence-$mode"
  generation=$(current_generation "$case_root/output")
  case "$mode" in
    missing)
      rm -rf -- "$case_root/output/training-evidence"
      ;;
    conflicting)
      python3 - "$case_root/output" "${generation#sha256-}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "training-evidence"
payload = json.dumps(
    {
        "version": 1,
        "checkpoint_manifest_sha256": sys.argv[2],
        "conflicting_receipt": True,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()
digest = hashlib.sha256(payload).hexdigest()
(root / f"sha256-{digest}.json").write_bytes(payload)
PY
      ;;
    *) fail "unknown training-evidence rejection mode: $mode" ;;
  esac
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_SYNC_INTERVAL=1
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_READY=$case_root/ready
  export TEST_RELEASE=$case_root/release
  export TEST_BLOCK=true
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1 &
  supervisor_pid=$!
  for _attempt in {1..200}; do
    if grep -q 'checkpoint must have exactly one checkpoint-bound training-evidence artifact' \
      "$case_root/state/sync.log" 2>/dev/null; then
      observed=true
      break
    fi
    sleep 0.05
  done
  : >"$TEST_RELEASE"
  wait "$supervisor_pid"
  unset TEST_BLOCK TEST_READY TEST_RELEASE
  [[ $observed == true ]] \
    || fail "$mode checkpoint-bound training-evidence failure was not explained"
  [[ ! -e $case_root/remote/current.json ]] \
    || fail "$mode checkpoint-bound training evidence was published"
}

run_sleep_and_qat_vm_loss_restore_test() {
  local case_root=$TEST_ROOT/artifact-vm-loss
  local config=$case_root/relaunch.conf
  local generation expected_future
  prepare_artifact_checkpoint "$case_root" 31 stable
  publish_remote_checkpoint "$case_root" 31 "$PREPARED_COMMAND"
  generation=$(current_generation "$case_root/remote")
  local closure
  closure=$(artifact_manifest_for "$case_root/remote" "$generation")
  [[ -s $closure ]] || fail "generated-artifact closure was not published"
  if grep -q 'future-after\|future-stable\|"path":"current.json"' "$closure"; then
    fail "checkpoint closure captured future or mutable convenience state"
  fi
  [[ -n $(find "$case_root/remote/checkpoint-objects" -type f -print -quit) ]] \
    || fail "generated artifacts were not uploaded into the global CAS"

  expected_future=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["future_model"])' "$PREPARED_EXPECTED")
  rm -rf -- "$case_root/seed-output" "$case_root/runtime/stores"
  mkdir -p -- "$case_root/runtime/stores/candidates/local-only"
  printf 'preserve-local-data\n' \
    >"$case_root/runtime/stores/candidates/local-only/unrelated.bin"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/restore-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_SYNC_INTERVAL=60
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/restore-calls
  export TEST_EXPECT_STEP=31
  export TEST_FAIL_ONCE=false
  unset TEST_BLOCK TEST_READY TEST_RELEASE TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"

  python3 - "$PREPARED_EXPECTED" <<'PY'
import json
import pathlib
import sys

expected = json.load(open(sys.argv[1]))
for name in (
    "qat",
    "tensor",
    "tier",
    "dream",
    "dream_parent",
    "dream_policy_adapter",
    "model",
    "journal",
):
    path = pathlib.Path(expected[name])
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"required restored artifact {name} is unavailable: {path}")
PY
  [[ ! -e $expected_future && ! -L $expected_future ]] \
    || fail "future sleep-model artifact was restored into checkpoint G"
  [[ ! -e $case_root/runtime/stores/tensor/current.json ]] \
    || fail "future tensor current.json was restored into checkpoint G"
  [[ $(cat "$case_root/runtime/stores/candidates/local-only/unrelated.bin") \
    == preserve-local-data ]] \
    || fail "restore overwrote an unrelated local generated artifact"
  [[ $(find "$case_root/seed-output/training-evidence" -type f | wc -l) -eq 1 ]] \
    || fail "generation-bound training evidence was not restored exactly"
}

run_equal_generation_release_rehydration_test() {
  local case_root=$TEST_ROOT/equal-generation-rehydration
  local config=$case_root/relaunch.conf
  local missing_artifact metrics_path
  prepare_artifact_checkpoint "$case_root" 32 equal
  publish_remote_checkpoint "$case_root" 32 "$PREPARED_COMMAND"
  missing_artifact=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["model"])' \
    "$PREPARED_EXPECTED")
  metrics_path=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["metrics_path"])' \
    "$case_root/remote/current.json")

  rm -f -- "$missing_artifact"
  printf '{"locally_corrupt_but_complete_record":true}\n' \
    >"$case_root/seed-output/metrics.jsonl"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/rehydration-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_SYNC_INTERVAL=60
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/rehydration-calls
  export TEST_EXPECT_STEP=32
  export TEST_FAIL_ONCE=false
  unset TEST_BLOCK TEST_READY TEST_RELEASE TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"

  [[ -f $missing_artifact && ! -L $missing_artifact ]] \
    || fail "equal checkpoint generation did not restore its missing artifact closure"
  cmp -- "$case_root/seed-output/metrics.jsonl" \
    "$case_root/remote/$metrics_path" >/dev/null \
    || fail "equal checkpoint generation did not restore its committed metric prefix"
}

run_rewritten_closure_rejected_test() {
  local case_root=$TEST_ROOT/rewritten-closure
  local config=$case_root/relaunch.conf
  local generation closure
  prepare_artifact_checkpoint "$case_root" 33 rewrite
  publish_remote_checkpoint "$case_root" 33 "$PREPARED_COMMAND"
  generation=$(current_generation "$case_root/remote")
  closure=$(artifact_manifest_for "$case_root/remote" "$generation")
  python3 - "$closure" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
for root in value["roots"]:
    root["files"] = []
path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
PY
  rm -rf -- "$case_root/seed-output" "$case_root/runtime/stores"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/restore-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/restore-calls
  export TEST_FAIL_ONCE=false
  unset TEST_BLOCK TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER
  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1; then
    fail "rewritten generated-artifact closure was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] \
    || fail "trainer launched after generated-artifact closure omission"
  grep -q 'rewritten generated-artifact closure' "$case_root/restore-state/sync.log" \
    || fail "rewritten generated-artifact closure failure was not explained"
}

run_full_dream_manifest_history_test() {
  local case_root=$TEST_ROOT/full-dream-history
  local config=$case_root/relaunch.conf
  local oldest_manifest oldest_candidate
  export TEST_DREAM_HISTORY_COUNT=65
  prepare_artifact_checkpoint "$case_root" 35 history
  unset TEST_DREAM_HISTORY_COUNT
  oldest_manifest=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["oldest_dream_manifest"])' "$PREPARED_EXPECTED")
  oldest_candidate=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["oldest_dream_candidate"])' "$PREPARED_EXPECTED")
  publish_remote_checkpoint "$case_root" 35 "$PREPARED_COMMAND"
  rm -rf -- "$case_root/seed-output" "$case_root/runtime/stores"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/restore-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/restore-calls
  export TEST_EXPECT_STEP=35
  export TEST_FAIL_ONCE=false
  unset TEST_BLOCK TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"
  [[ -f $oldest_manifest && ! -L $oldest_manifest ]] \
    || fail "Dreaming manifest older than the 64-transaction tail was not restored"
  [[ -f $oldest_candidate && ! -L $oldest_candidate ]] \
    || fail "Dreaming candidate older than the 64-transaction tail was not restored"
}

run_external_initial_policy_test() {
  local case_root=$TEST_ROOT/external-initial-policy
  local config=$case_root/relaunch.conf
  local generation closure policy_path policy_sha256
  export TEST_EXTERNAL_INITIAL_POLICY=true
  prepare_artifact_checkpoint "$case_root" 39 external-policy
  unset TEST_EXTERNAL_INITIAL_POLICY
  policy_path=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dream_parent"])' "$PREPARED_EXPECTED")
  policy_sha256=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$policy_path")
  publish_remote_checkpoint "$case_root" 39 "$PREPARED_COMMAND"
  generation=$(current_generation "$case_root/remote")
  closure=$(artifact_manifest_for "$case_root/remote" "$generation")
  python3 - "$closure" "$policy_sha256" <<'PY'
import json
import sys

closure = json.load(open(sys.argv[1]))
if closure["dream_initial_policy_sha256"] != sys.argv[2]:
    raise SystemExit("closure does not bind the external initial policy")
selected_digests = {
    entry["sha256"] for root in closure["roots"] for entry in root["files"]
}
if sys.argv[2] in selected_digests:
    raise SystemExit("deployment-owned initial policy was copied into generated-artifact CAS")
PY
  rm -rf -- "$case_root/seed-output" "$case_root/runtime/stores"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/restore-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/restore-calls
  export TEST_EXPECT_STEP=39
  export TEST_FAIL_ONCE=false
  unset TEST_BLOCK TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config"
  [[ -f $policy_path && ! -L $policy_path ]] \
    || fail "deployment-bound initial policy disappeared during restore"

  rm -rf -- "$case_root/seed-output"
  printf 'tampered\n' >>"$policy_path"
  export TEST_CALLS=$case_root/tampered-calls
  unset TEST_EXPECT_STEP
  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/tampered.log" 2>&1; then
    fail "tampered deployment-bound initial policy was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] \
    || fail "trainer launched with a tampered deployment-bound initial policy"
  grep -q 'initial_policy digest mismatch' "$case_root/tampered.log" \
    || fail "external initial policy binding failure was not explained"
}

run_auxiliary_corruption_rejected_test() {
  local mode=$1
  local case_root=$TEST_ROOT/artifact-$mode
  local config=$case_root/relaunch.conf
  local generation object
  prepare_artifact_checkpoint "$case_root" 37 "$mode"
  publish_remote_checkpoint "$case_root" 37 "$PREPARED_COMMAND"
  generation=$(current_generation "$case_root/remote")
  object=$(remote_object_for_root \
    "$case_root/remote" "$generation" output.quantized-candidates)
  if [[ $mode == missing ]]; then
    rm -f -- "$object"
  else
    printf 'tampered\n' >>"$object"
  fi
  rm -rf -- "$case_root/seed-output" "$case_root/runtime/stores"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/restore-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/restore-calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_BLOCK TEST_READY TEST_RELEASE \
    TEST_WANDB_CALLS TEST_FAILURE_MARKER
  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/restore.log" 2>&1; then
    fail "$mode remote generated artifact was accepted"
  fi
  [[ ! -e $TEST_CALLS ]] || fail "trainer launched with a $mode generated artifact"
  grep -q 'remote checkpoint is incomplete' "$case_root/restore.log" \
    || fail "$mode generated-artifact failure was not explained"
}

run_selected_artifact_symlink_rejected_test() {
  local case_root=$TEST_ROOT/artifact-symlink
  local config=$case_root/relaunch.conf
  local selected target supervisor_pid observed=false
  prepare_artifact_checkpoint "$case_root" 43 symlink
  selected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$PREPARED_EXPECTED")
  target=$case_root/symlink-target
  printf 'teacher-symlink\n' >"$target"
  rm -f -- "$selected"
  ln -s -- "$target" "$selected"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_SYNC_INTERVAL=1
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/calls
  export TEST_READY=$case_root/ready
  export TEST_RELEASE=$case_root/release
  export TEST_BLOCK=true
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1 &
  supervisor_pid=$!
  for _attempt in {1..200}; do
    if grep -q 'symbolic link' "$case_root/state/sync.log" 2>/dev/null; then
      observed=true
      break
    fi
    sleep 0.05
  done
  : >"$TEST_RELEASE"
  wait "$supervisor_pid"
  unset TEST_BLOCK TEST_READY TEST_RELEASE
  [[ $observed == true ]] || fail "selected generated-artifact symlink was not rejected"
  [[ ! -e $case_root/remote/current.json ]] \
    || fail "current.json was published after generated-artifact symlink rejection"
}

run_existing_artifact_conflict_rejected_test() {
  local case_root=$TEST_ROOT/artifact-conflict
  local config=$case_root/relaunch.conf
  local selected
  prepare_artifact_checkpoint "$case_root" 45 conflict
  publish_remote_checkpoint "$case_root" 45 "$PREPARED_COMMAND"
  selected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$PREPARED_EXPECTED")
  rm -rf -- "$case_root/seed-output" "$case_root/runtime/stores"
  mkdir -p -- "$(dirname -- "$selected")"
  printf 'local-conflict\n' >"$selected"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/restore-state
HERMES_TRAIN_REMOTE_URL=file://$case_root/remote
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_CALLS=$case_root/restore-calls
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_BLOCK TEST_READY TEST_RELEASE \
    TEST_WANDB_CALLS TEST_FAILURE_MARKER
  if "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/restore.log" 2>&1; then
    fail "conflicting local generated artifact was overwritten"
  fi
  [[ $(cat "$selected") == local-conflict ]] \
    || fail "conflicting local generated artifact bytes changed"
  [[ ! -e $TEST_CALLS ]] || fail "trainer launched after an artifact restore conflict"
  [[ ! -e $case_root/seed-output/current.json ]] \
    || fail "current.json was published after an artifact restore conflict"
  grep -q 'cannot restore the newest remote checkpoint' "$case_root/restore.log" \
    || fail "artifact restore conflict was not explained"
}

run_remote_publication_order_test() {
  local case_root=$TEST_ROOT/artifact-order
  local config=$case_root/relaunch.conf
  local supervisor_pid published=false generation object_line manifest_line current_line
  prepare_artifact_checkpoint "$case_root" 47 order
  generation=$(current_generation "$case_root/seed-output")
  mkdir -p -- "$case_root/gcs"
  : >"$case_root/gcs.log"
  cat >"$config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/seed-output
HERMES_TRAIN_STATE_DIR=$case_root/state
HERMES_TRAIN_REMOTE_URL=gs://test-bucket/run
HERMES_TRAIN_GCLOUD=$fake_gcloud
HERMES_TRAIN_COMMAND=($fake_trainer train $PREPARED_COMMAND)
HERMES_TRAIN_SYNC_INTERVAL=1
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_GCS_ROOT=$case_root/gcs
  export TEST_GCS_LOG=$case_root/gcs.log
  export TEST_CALLS=$case_root/calls
  export TEST_READY=$case_root/ready
  export TEST_RELEASE=$case_root/release
  export TEST_BLOCK=true
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER
  "$TEST_SCRIPT_DIR/relaunch.sh" "$config" >"$case_root/log" 2>&1 &
  supervisor_pid=$!
  for _attempt in {1..400}; do
    if [[ -s $case_root/gcs/test-bucket/run/current.json ]]; then
      published=true
      break
    fi
    sleep 0.05
  done
  : >"$TEST_RELEASE"
  wait "$supervisor_pid"
  unset TEST_BLOCK TEST_READY TEST_RELEASE TEST_GCS_ROOT TEST_GCS_LOG
  [[ $published == true ]] || {
    sed -n '1,240p' "$case_root/state/sync.log" >&2 || true
    fail "gs checkpoint pointer was not published"
  }
  object_line=$(grep -n $'UPLOAD\ttest-bucket/run/checkpoint-objects/sha256/' \
    "$case_root/gcs.log" | head -1 | cut -d: -f1)
  manifest_line=$(grep -n $'UPLOAD\ttest-bucket/run/checkpoint-artifacts/'"$generation"'/artifact-manifest.json' \
    "$case_root/gcs.log" | head -1 | cut -d: -f1)
  current_line=$(grep -n $'UPLOAD\ttest-bucket/run/current.json' \
    "$case_root/gcs.log" | tail -1 | cut -d: -f1)
  [[ -n "$object_line" && -n "$manifest_line" && -n "$current_line" \
    && $object_line -lt $manifest_line && $manifest_line -lt $current_line ]] \
    || fail "current.json was not published strictly after CAS objects and closure manifest"
}

run_concurrent_pointer_race_test() {
  local mode=$1
  local case_root=$TEST_ROOT/concurrent-pointer-$mode
  local first_config=$case_root/first.conf
  local second_config=$case_root/second.conf
  local first_step second_step first_generation second_generation
  local first_pid second_pid delayed=false published=false observed_generation
  case "$mode" in
    stale)
      first_step=61
      second_step=62
      ;;
    fork)
      first_step=63
      second_step=63
      ;;
    *) fail "unknown pointer race mode: $mode" ;;
  esac
  write_checkpoint "$case_root/first-output" "$first_step" 2 "$mode-first"
  write_checkpoint "$case_root/second-output" "$second_step" 2 "$mode-second"
  first_generation=$(current_generation "$case_root/first-output")
  second_generation=$(current_generation "$case_root/second-output")
  [[ $first_generation != "$second_generation" ]] \
    || fail "$mode pointer race fixture did not produce distinct generations"
  mkdir -p -- "$case_root/gcs"
  : >"$case_root/gcs.log"
  cat >"$first_config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/first-output
HERMES_TRAIN_STATE_DIR=$case_root/first-state
HERMES_TRAIN_REMOTE_URL=gs://test-bucket/$mode
HERMES_TRAIN_GCLOUD=$fake_gcloud
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_SYNC_INTERVAL=1
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  cat >"$second_config" <<EOF
HERMES_TRAIN_OUTPUT=$case_root/second-output
HERMES_TRAIN_STATE_DIR=$case_root/second-state
HERMES_TRAIN_REMOTE_URL=gs://test-bucket/$mode
HERMES_TRAIN_GCLOUD=$fake_gcloud
HERMES_TRAIN_COMMAND=($fake_trainer train)
HERMES_TRAIN_SYNC_INTERVAL=1
HERMES_TRAIN_MAX_RESTARTS=0
EOF
  export TEST_GCS_ROOT=$case_root/gcs
  export TEST_GCS_LOG=$case_root/gcs.log
  export TEST_GCS_DELAY_CURRENT_STEP=$first_step
  export TEST_GCS_DELAY_CURRENT_GENERATION=$first_generation
  export TEST_GCS_DELAY_READY=$case_root/delayed
  export TEST_GCS_DELAY_RELEASE=$case_root/release-pointer
  export TEST_BLOCK=true
  export TEST_FAIL_ONCE=false
  unset TEST_EXPECT_STEP TEST_WANDB_CALLS TEST_FAILURE_MARKER
  TEST_CALLS=$case_root/first-calls \
    TEST_READY=$case_root/first-ready \
    TEST_RELEASE=$case_root/release-first \
    "$TEST_SCRIPT_DIR/relaunch.sh" "$first_config" >"$case_root/first.log" 2>&1 &
  first_pid=$!
  for _attempt in {1..400}; do
    if [[ -e $TEST_GCS_DELAY_READY ]]; then
      delayed=true
      break
    fi
    sleep 0.05
  done
  if [[ $delayed != true ]]; then
    : >"$case_root/release-first"
    wait "$first_pid" || true
    fail "$mode pointer publisher did not reach its delayed compare-and-swap"
  fi
  TEST_CALLS=$case_root/second-calls \
    TEST_READY=$case_root/second-ready \
    TEST_RELEASE=$case_root/release-second \
    "$TEST_SCRIPT_DIR/relaunch.sh" "$second_config" >"$case_root/second.log" 2>&1 &
  second_pid=$!
  for _attempt in {1..400}; do
    observed_generation=$(current_generation \
      "$case_root/gcs/test-bucket/$mode" 2>/dev/null || true)
    if [[ $observed_generation == "$second_generation" ]]; then
      published=true
      break
    fi
    sleep 0.05
  done
  : >"$TEST_GCS_DELAY_RELEASE"
  : >"$case_root/release-first"
  : >"$case_root/release-second"
  wait "$first_pid"
  wait "$second_pid"
  unset TEST_BLOCK TEST_GCS_ROOT TEST_GCS_LOG TEST_GCS_DELAY_CURRENT_STEP \
    TEST_GCS_DELAY_CURRENT_GENERATION TEST_GCS_DELAY_READY \
    TEST_GCS_DELAY_RELEASE
  [[ $published == true ]] || fail "$mode winning checkpoint was not published"
  [[ $(current_generation "$case_root/gcs/test-bucket/$mode") == "$second_generation" ]] \
    || fail "$mode delayed publisher rewound or replaced the winning checkpoint"
  case "$mode" in
    stale)
      grep -Eq 'remote checkpoint advanced|leaving it unchanged' \
        "$case_root/first-state/sync.log" \
        || fail "stale concurrent publisher did not report the newer checkpoint"
      ;;
    fork)
      grep -q 'equal-step remote checkpoint fork' \
        "$case_root/first-state/sync.log" \
        || fail "equal-step concurrent fork was not rejected explicitly"
      ;;
  esac
}

run_cross_generation_cas_dedup_test() {
  local case_root=$TEST_ROOT/artifact-dedup
  local first second
  prepare_artifact_checkpoint "$case_root" 53 shared
  publish_remote_checkpoint "$case_root" 53 "$PREPARED_COMMAND"
  first=$(current_generation "$case_root/remote")
  prepare_artifact_checkpoint "$case_root" 54 shared
  publish_remote_checkpoint "$case_root" 54 "$PREPARED_COMMAND"
  second=$(current_generation "$case_root/remote")
  [[ $first != "$second" ]] || fail "dedup fixture did not advance checkpoint generation"
  python3 - "$case_root/remote" "$first" "$second" <<'PY'
import json
import pathlib
import sys

remote = pathlib.Path(sys.argv[1])
sets = []
counts = []
for generation in sys.argv[2:]:
    value = json.loads(
        (remote / "checkpoint-artifacts" / generation / "artifact-manifest.json").read_text()
    )
    digests = [entry["sha256"] for root in value["roots"] for entry in root["files"]]
    sets.append(set(digests))
    counts.append(len(digests))
if not sets[0].intersection(sets[1]):
    raise SystemExit("successive closures did not share any immutable object")
union = sets[0] | sets[1]
objects = {
    path.name
    for path in (remote / "checkpoint-objects" / "sha256").glob("*/*")
    if path.is_file()
}
if objects != union:
    raise SystemExit("global CAS inventory differs from the union of closure digests")
if len(union) >= sum(counts):
    raise SystemExit("successive closure payloads were duplicated instead of reused")
PY
}

run_restart_and_reporting_test
run_remote_restore_test
run_newer_local_wins_test
run_idempotent_lock_test
run_version_one_checkpoint_rejected_test
run_modified_generation_rejected_test
run_global_step_mismatch_rejected_test
run_unsafe_pointer_rejected_test
run_unsafe_manifest_path_rejected_test
run_corrupt_remote_rejected_test
run_training_evidence_rejected_test missing
run_training_evidence_rejected_test conflicting
run_sleep_and_qat_vm_loss_restore_test
run_equal_generation_release_rehydration_test
run_rewritten_closure_rejected_test
run_full_dream_manifest_history_test
run_external_initial_policy_test
run_auxiliary_corruption_rejected_test missing
run_auxiliary_corruption_rejected_test tampered
run_selected_artifact_symlink_rejected_test
run_existing_artifact_conflict_rejected_test
run_remote_publication_order_test
run_concurrent_pointer_race_test stale
run_concurrent_pointer_race_test fork
run_cross_generation_cas_dedup_test
printf 'relaunch_test: ok\n'
