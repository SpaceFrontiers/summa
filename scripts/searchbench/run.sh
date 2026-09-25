#!/usr/bin/env bash
# Run in the dedicated benchmark directory after source/corpus/driver preparation.
# References use upstream start/feed/merge scripts. All engines run sequentially.
set -euo pipefail
RUN_DIR=$(cd "${1:?usage: run.sh RUN_DIRECTORY SUMMA_BINARY}" && pwd)
RUNNER=$(realpath "$0")
SUMMA_BINARY=$(realpath "${2:?missing Summa binary}")
cd "$RUN_DIR"
export SERVER_CORES=0-2,4-6
export SEARCHBENCH_QUERY_CACHE=off
export SEARCHBENCH_HEAP=8g
export LUXIR_BIN=${LUXIR_BIN:-$RUN_DIR/luxir-release/luxir-0.1.0-linux-x86_64-v4}
[[ -x "$LUXIR_BIN" ]]
export TMPDIR="$RUN_DIR/tmp"
mkdir -p "$TMPDIR" results
source "$RUN_DIR/searchbench/scripts/engine-common.sh"
ACTIVE_ENGINE=
SUMMA_PID=
cleanup() {
  local status=$?
  if [[ -n "$SUMMA_PID" ]] && kill -0 "$SUMMA_PID" 2>/dev/null; then
    kill -INT "$SUMMA_PID"
    wait "$SUMMA_PID" || true
  fi
  if [[ -n "$ACTIVE_ENGINE" ]]; then
    "$ROOT/scripts/stop-$ACTIVE_ENGINE.sh" || true
  fi
  printf '%s\n' "$status" > exit-status.txt
}
trap cleanup EXIT
stage() {
  echo "$(date -u +%FT%TZ) $*" | tee status.txt
}
probe() {
  python3 campaign.py probe --searchbench "$ROOT" --out results \
    --engine "$1" --port "$2"
}
measure() {
  python3 campaign.py measure --searchbench "$ROOT" --out results \
    --engine "$1" --port "$2" --server-pid "$3"
}
start_summa() {
  taskset -c "$SERVER_CORES" "$SUMMA_BINARY" serve "$RUN_DIR/summa-index" 9401 6 \
    >> summa-serve.log 2>&1 &
  SUMMA_PID=$!
  wait_ready http://127.0.0.1:9401/health "$SUMMA_PID"
}
stop_summa() {
  kill -INT "$SUMMA_PID"
  wait "$SUMMA_PID"
  SUMMA_PID=
}

# The upstream transform verifies the pinned source SHA and publishes only a
# completed corpus. Never index its .part file.
stage 'Checking prepared 10M corpus'
CORPUS="$ROOT/corpus/corpus-10m-searchbench.ndjson"
[[ -s "$CORPUS" && -s "$CORPUS.sha256" ]] || {
  echo "Run searchbench/scripts/prepare-corpus.sh standard and wait for completion first" >&2
  exit 1
}
DIGEST=$(corpus_sha256 "$CORPUS")
[[ "$DIGEST" == b15e60ad32a0e9f09f3be5335db86ccd885e1be1418086a906d68529e6ec1c9d ]]
cp "$CORPUS.sha256" results/corpus.sha256
cp "$ROOT/engines/versions.json" results/reference-versions.json
cp luxir-release.json results/luxir-release.json
cp source-identity.json results/summa-source-identity.json
sha256sum "$SUMMA_BINARY" "$LUXIR_BIN" campaign.py report.py "$RUNNER" "$ROOT/driver/build/bench_replay" > results/binaries.sha256
lscpu > results/lscpu.txt
uname -a > results/kernel.txt
for engine in elasticsearch opensearch luxir; do
  stage "Indexing and merging $engine"
  ACTIVE_ENGINE=$engine
  ensure_dataset "$engine" "$CORPUS" merged "$DIGEST" 10000000
  stage "Probing all 826 queries: $engine"
  ACTIVE_ENGINE=$engine
  "$ROOT/scripts/start-$engine.sh"
  probe "$engine" "$(engine_port "$engine")"
  "$ROOT/scripts/stop-$engine.sh"
  ACTIVE_ENGINE=
done
stage 'Indexing and merging Summa'
taskset -c "$SERVER_CORES" "$SUMMA_BINARY" index "$RUN_DIR/summa-index" "$CORPUS" 6 \
  > summa-index.log 2>&1
stage 'Probing all 826 queries: Summa'
start_summa
probe summa 9401
stop_summa
stage 'Selecting identical count-agreeing query subset'
python3 campaign.py gate --searchbench "$ROOT" --out results | tee results/coverage.txt
for engine in summa elasticsearch opensearch luxir; do
  stage "Measuring $engine: 1/8/32 clients, 3 x 10 seconds per cell"
  if [[ "$engine" == summa ]]; then
    start_summa
    measure summa 9401 "$SUMMA_PID"
    stop_summa
  else
    ACTIVE_ENGINE=$engine
    "$ROOT/scripts/start-$engine.sh"
    measure "$engine" "$(engine_port "$engine")" "$(cat "$(engine_pidfile "$engine")")"
    "$ROOT/scripts/stop-$engine.sh"
    ACTIVE_ENGINE=
  fi
done
python3 report.py results
stage 'Complete: all four engines measured'
