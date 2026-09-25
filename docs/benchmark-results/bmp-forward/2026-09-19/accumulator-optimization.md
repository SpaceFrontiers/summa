# BMP forward rescoring: compact accumulator

## Change and invariant

`query/bmp.rs::score_forward_units` now carries `Option<u32>` through the packet
fold, then constructs the existing query error once at the row boundary. The
previous implementation carried `Result<u32, Error>` through every coordinate;
the measured x86 code copied that large state through the stack repeatedly.
The new accumulator retains checked addition and sticky overflow. Duplicate
query/document dimensions, the exact `u32::MAX` boundary, and final float
conversion keep the same semantics. The decoder, serialized bytes, admission,
cache policy and settings are unchanged. No per-vector allocation is added.

## Measurement protocol

This follow-up uses the same isolated 8-vCPU Intel Cascade Lake machine, Rust 1.98.1,
release thin LTO, one codegen unit, and `-C target-cpu=native` as the
[format comparison](query-latency.md). Both binaries read **the same BMPB files**:
900,000 real retained vectors, 146,460,740 nonzeros and 42,596 documents. The
queries are the same 64 synthetic 16-term templates; this is not production
traffic. The fixed 1,000-document control expands about 20,841 vectors, while
retrieval at k=1,000 expands about 64,653 vectors on average.

Two trials reverse before/after order. Warm runs use three passes, discarding
the first; capped runs use two passes and report first/repeat sweeps separately.
Only selected fixture files receive `POSIX_FADV_DONTNEED` before pressure runs.
Memory caps use cgroup v2, zero swap, and I/O accounting. Pin mode is copy with a
64 MiB metadata budget per segment. Requests run serially; reported QPS is
serial throughput, not a concurrent server capacity claim. Timing excludes
result serialization, and both binaries use the identical streaming runner.

An initial comparison restored an older runner from the source archive. It
buffered its evidence as a JSON array instead of streaming JSONL. All 192
retrieval results matched after parsing, but those timings were discarded.
The optimized executable was rebuilt with the exact previously measured
runner, dependency lock and flags before running the comparison below.

## Results

The compact accumulator makes warm k=1,000 rescoring **2.54× faster** and
approximately halves total query latency. Times are milliseconds; warm and
capped results below exclude the first sweep.

| Workload                 | Before mean / p95 | After mean / p95 |
| ------------------------ | ----------------: | ---------------: |
| Warm k=1,000 backfill    |   116.88 / 188.78 |    45.97 / 74.64 |
| Warm k=1,000 total       |   143.21 / 214.48 |   72.07 / 106.65 |
| Warm k=100 total         |     33.01 / 64.63 |    20.98 / 43.02 |
| Warm fixed 1,000 docs    |     38.94 / 46.12 |    16.06 / 19.05 |
| Warm retrieval k=10      |      7.59 / 22.62 |     7.64 / 22.77 |
| 1 GiB k=1,000 total      |   144.72 / 215.42 |   73.62 / 109.81 |
| 512 MiB fixed 1,000 docs |     39.39 / 45.33 |    16.32 / 19.51 |

The first cold sweep remains dominated by demand paging:

| Cold workload            | Before mean | After mean |
| ------------------------ | ----------: | ---------: |
| 1 GiB k=1,000 total      |      936.22 |     859.57 |
| 512 MiB fixed 1,000 docs |      990.49 |     944.41 |

Memory stays effectively unchanged:

| Measurement              | Before peak | After peak |
| ------------------------ | ----------: | ---------: |
| Warm pipeline RSS        |  1148.8 MiB | 1148.7 MiB |
| 1 GiB pipeline cgroup    |   892.1 MiB |  892.2 MiB |
| 512 MiB fixed-doc cgroup |   441.7 MiB |  441.8 MiB |

Cgroup peaks include file cache, kernel memory and the worker; RSS includes
mapped pages. Neither is an isolated Rust heap measurement. All eight capped
runs completed with zero OOM events. The forward payload remains 402,399,443
bytes, retaining its 45.05% reduction relative to the earlier raw fixture.
Every index-file hash is identical before and after the matrix.

All **24 runs / 4,096 timed requests** completed. All **12 paired comparisons**
match every result ID and score bit, including every candidate score saved
before top-10 truncation. Detailed distributions, serial throughput, RSS,
cgroup peaks, CPU time and major faults are in [summary.json](accumulator/summary.json).

## CPU profile

Separate resident profiles cover 256 fixed-pool requests per binary at 499 Hz,
after the timing matrix, with zero lost samples. The
[before profile](accumulator/before-profile.txt) records about 10.20 CPU seconds;
the [after profile](accumulator/after-profile.txt) records about 4.22. These
instrumented runs identify hotspots; the latency table uses unprofiled runs.

The [before loop](accumulator/before-hot-loop.txt) repeats scalar/vector copies
of the error-capable accumulator. One instruction among those copies receives
41.62% of the scorer's samples. Those large accumulator copies are absent from
the [optimized hot loop](accumulator/after-hot-loop.txt); its hottest sampled
instruction (9.94% of scorer samples) is in SIMD gap unpacking, followed by
prefix sums. This confirms the intended code-generation change without claiming
that a sampled program counter identifies a particular hardware stall.

## Reproduction and evidence

The [matrix driver](accumulator/matrix.py) expects the isolated fixture directory
(`BMP_QUERY_EVIDENCE`, default `/mnt/summa-copy-merge/bmp-query`) to contain
`gap-index/`, `queries.json`, and `optimization/{before,after}-bin`. Run it only
against a disposable benchmark fixture. The checked-in driver adds a path
override and formatting to the measured script; the executed copy is archived.

```sh
python3 docs/benchmark-results/bmp-forward/2026-09-19/accumulator/matrix.py
python3 docs/benchmark-results/bmp-forward/2026-09-19/accumulator/analyze.py \
  /mnt/summa-copy-merge/bmp-query/optimization/results.json \
  /mnt/summa-copy-merge/bmp-query/optimization/summary.json
```

Binary/scorer/runner hashes are in [build-manifest.json](accumulator/build-manifest.json),
host/source/query metadata in [metadata.json](accumulator/metadata.json), and
immutable index hashes in [fixture-hashes.json](accumulator/fixture-hashes.json).
The complete private archive contains per-query score bits, all timings/counters,
source snapshots, executed scripts and full annotated profiles. It is saved at
`.context/bmp-query/optimization/accumulator-evidence.zip` (15,676,712 bytes),
SHA256 `97c9dc7f79e4f0a7901ec93eb571c068f6a58c8adb4d983a489e56043a25a2e7`.

The archive was downloaded and hash-verified before shutdown. The benchmark machine
is confirmed `TERMINATED`; the stop client lost its connection while polling,
and a separate instance-status request confirmed completion.

## Validation

`RUST_TEST_THREADS=1 python3 scripts/check_search.py check` passed strict Clippy,
2,023 native tests (25 normally ignored), native-without-sync and broker feature
checks. Evidence: `.context/search-harness/20260919T111019.522940Z-check`.
The first parallel run timed out during broker discovery; the full serial
rerun passed without relaxing assertions. The WASM release build and all 38
JavaScript tests passed. New tests cover duplicate matches across packets and
fixed/gap encodings, unmatched/empty queries, exactly `u32::MAX`, and sticky
overflow through subsequent unmatched packets. Existing format goldens and
forward/inverted score comparisons remain in the passing native suite.
