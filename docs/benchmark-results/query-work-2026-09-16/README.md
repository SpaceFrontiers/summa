# Query-work diagnosis evidence — September 16, 2026

[Findings](../../search-work-diagnosis.md) ·
[Instrumentation and commands](../../query-work-diagnostics.md)

## Contents

- `x86/work/`, `arm/work/`: two passes over all 1,676 queries and four commands;
  structured counters, original stderr and checked protocol responses.
- `x86/verify/`, `arm/verify/`: exhaustive versus optimized checks for all three
  Summa index layouts at k=10/100/1000. VERIFY counters aggregate its sub-runs
  and must not be interpreted as ordinary query-work measurements.
- `x86/norm-ablation/`: seven rotated latency passes on the same quantized index,
  table-enabled and canonical controls, temporary patch, build logs and exact
  oracle hashes. Neither the temporary scorer choice nor a default change is
  retained. Full oracle rows already reside in the
  [compact-format evidence](../compact-text-2026-09-16/README.md).
- `summaries/`: per-family and per-query work deltas, aggregate totals, invariants
  and separated latency summaries.
- `source/`: final diagnostic overlay and capture-era source snapshots. Apply the
  final overlay over the capture snapshot for the completed tests and API docs.
  The capture snapshot retains its draft test fixture; final test refinements and
  comments do not change the measured scoring or traversal implementation.
- `scripts/`, `checks/`, `reports/`: runners, native harness logs, diagnostic tests,
  checker logs and frozen reports. Tantivy diagnostic patches/source are retained
  under `upstream/`; its unavailable counters are never filled with zero.

The x86 run uses 5,032,104 Wikipedia documents, Cascade Lake CPU 2, Rust 1.98.1,
LLVM 22.1.8, release LTO and `target-cpu=native`. ARM uses the preserved 100,000
rows on Apple M4. All indexes have RGB disabled, Rounded payloads, ratio/L1 bounds,
no impact envelopes, a 262,144-byte proof budget and an 8,192-block/4,194,304-byte
term cache. These are existing immutable fixtures; no index writer ran.

Diagnostic timings include instrumentation overhead. Only the separate norm
experiment uses uninstrumented binaries. It runs the official queries followed
by supplemental queries in one process per engine, unlike the earlier separate
workload timing runs; only same-run comparisons are asserted.

The archive excludes executables, corpus/index files and duplicate full oracle
rows. Binary hashes, source hashes and raw samples remain. Each archive member
has its byte length and SHA-256 in `manifest.json`. The cloud download was checked
against its remote archive hash and all 50 member hashes before stopping the machine.

## Download

Results and sources (local archive `results.zip`) · [Member manifest](manifest.json)

Archive SHA-256: `73d303b65ca613754d611b1a93d56ce4bdaf532e0ff8181eca420d71ff8e9e92`.
Size: 17,977,863 bytes; 116 verified members.
