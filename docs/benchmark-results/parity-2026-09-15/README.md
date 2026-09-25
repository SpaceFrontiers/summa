# RGB-disabled parity experiments — September 15, 2026

This archive records the selected score-admission change, the integrated RGB
correctness work, and five isolated execution experiments. The parity target is
**RGB disabled**. Reordering is not used to obtain any timing reported here.
See [current results](../../search-benchmark-current.md) and the
[performance review](../../search-performance-review.md) for conclusions.

## Reading the archive

`manifest.json` maps each logical filename to a physical ZIP entry and records
its byte length and SHA-256. Identical content is stored once. Every entry was
read back and verified after packaging. Executables and corpus payloads are
excluded; binary and index hashes are retained. `native-check.zip` has its own
manifest and includes both the RGB tie-fix run and the final integrated run.

- `cloud/final/` contains the definitive integrated-source comparison against
  the previous published filtered-window build and Tantivy 0.26.
- `cloud/guard/` measures the scalar score rejection in isolation against that
  published build. `cloud/{batch,bounds,seek,mask,required}/` each use the **guard
  binary** as their timed control. Effects must not be added or multiplied.
- The original x86 `seek` and `required` attempts reused stale Cargo binaries
  because `copy2` preserved old source mtimes. Their timings do not measure
  those candidates: binary hashes equal `bounds` and `mask`, respectively.
  Corrected evidence is under `cloud/seek-recheck/` and
  `cloud/required-recheck/`. Their recipes restore the full control overlay
  using fresh writes, hash-check all sources and require a core rebuild.
- `batch` combines score admission and bitmap word accumulation; `mask` tests
  only score admission. Bitmap accumulation, prepared phrase bounds, the nearby
  seek probe and all-required window reuse were rejected.
- A candidate's `scoring.rs.before` records the file overwritten on the build
  host, which may be the preceding experiment. It is **not necessarily the
  source of its timed control**. Binary hashes and the recipes below identify
  the correct source.
- `local/final/` contains the ARM timings, four-layout oracles, index hashes,
  final source manifest, resource logs and check summary. `local/<experiment>/`
  contains isolated ARM experiments, recipes and inspected assembly.
- `cloud/profile/` is a separate untimed diagnostic run; its `current` label
  means the previous published filtered-window control, not the final source. Profile sample shares
  must not be presented as latency improvements.

## Reproduce the sources

Start a clean checkout of `ce2c96b945fccc4bac58ccc45bb4bc23b809773e`, then apply
`control/final-source.tar.gz` as a full source overlay. Its manifest identifies
the previous published build. Do not use today's `origin/main` as that source.
For the integrated build, apply `cloud/final/source.tar.gz` instead and verify
all entries of `source-manifest.json`. It includes 332 build/source/test files.
The final source includes the earlier dirty workspace changes; a single patch
against the branch tip is not a substitute for this overlay.

For isolated experiments, start again from the control overlay:

1. Apply `local/scoring-reject.rs` to `summa-core/src/query/scoring.rs` for the
   scalar guard control.
2. For `mask`, `seek` or `required`, replace that file with the corresponding
   `local/<experiment>/scoring.rs`.
3. For `batch`, replace both scoring and
   `summa-core/src/structures/postings/posting.rs` with its two candidate files.
4. For `bounds`, keep the guard scoring file and replace `query/bm25.rs` and
   `query/phrase.rs` with its candidate files.

After extracting any source overlay into an existing build directory, run
`cargo clean -p summa-core` before rebuilding, or rewrite the source files with
fresh mtimes as the corrected recipes do. Copy each control binary out of the
Cargo target directory first. Require an actual core compile and verify the
source manifest before timing; source hashes alone do not prove that Cargo
rebuilt a binary. The recorded final build did compile afresh.

Use Rust 1.98.1, the locked dependencies, `RUSTFLAGS='-C target-cpu=native'`,
`CARGO_PROFILE_RELEASE_LTO=true`, `CARGO_INCREMENTAL=0` and four build jobs.
Copy `oracles/traversal_score_oracle.rs` into `summa-core/examples/` for the
verification build. Build `search_benchmark_game` and `traversal_score_oracle`
as release examples. The `run.py`, `paired.py`, ARM scripts and environment
logs preserve exact commands and fixture paths; adjust machine-local paths.
Some scripts wait for a predecessor's verified completion marker to serialize
experiments. For a standalone reproduction, provide a verified predecessor or
remove only that scheduling wait, retaining all correctness and hash checks.

## Protocol and limits

The frozen x86 corpus has 5,032,104 documents. It uses the existing rounded
posting representation, normal impact bounds, `reorder=false`, and
`reorder_on_merge=false`. The second grouped-impact layout is an exact oracle
check, not the latency default. ARM uses the existing 262,144-document canonical
and merged fixtures, with both impact layouts checked and rounded layouts timed.
No index bytes were rewritten during these experiments.

The x86 driver pins itself and queries to CPU 2. Seven complete workload passes
rotate engine order after at least ten seconds of warmup for every engine and
command. ARM alternates before/after per query over seven samples. Results are
geometric means of per-query medians in microseconds, including the identical
line-protocol round trip. Queries comprise 962 official queries and 714
supplemental terms. Per-query raw durations and per-pass ratios are retained.
This is the paired driver, not a new execution of the original upstream runner.

Ordered document IDs, raw score bits and exact counts must match across Summa
sources, and each pruned top-k must match its exhaustive scorer. Counts also
match Tantivy. Cross-engine score/rank identity is not claimed because scoring
representations differ. `/usr/bin/time` captures process peak RSS, including
mapped pages and heap; it is not a heap-only measurement. The score mask adds
no allocation, cache or corpus-sized data. User applications remained active on
the ARM host, so small unrelated ARM movements are treated as inconclusive.

Final native validation passes 1,798 tests, with 25 ignored in the standard
stage, including the separately run real-server broker suite. Formatting,
Clippy, native without sync, portable core and API docs pass. WASM was not
rebuilt under the user's standing instruction. Cold-cache, concurrent-ingest,
p95/p99, upstream-runner and wider browser/client/GPU suites remain unrun.
RGB merge-time reordering and the remaining compatibility/budget audit are
unfinished; see [its design status](../../maxscore-text-reordering.md).

## Final decisions

The integrated guard plus score mask reduces full-corpus supplemental top-10
latency 20.2% and top-1000 10.4%; official top-10 improves 1.3% and remains
1.24× Tantivy. Official scored count regresses 0.7%, and supplemental scored
count regresses 2.7%. Exact count is essentially flat. These tradeoffs are
retained and reported without compounding the isolated experiments.

Corrected seek regresses official top-10 0.8% and is rejected. Corrected
all-required windows regress official top-10 3.0% and top-1000 11.9%; both
architectures reject that prototype. The invalid first attempts remain labeled
by `provenance.json`, and do not establish gains for either algorithm.

The owned benchmark machine is confirmed `TERMINATED` after all runs and verified
downloads. The stop command's status polling encountered a connection reset;
a subsequent direct status check confirmed shutdown. Changes remain uncommitted.
