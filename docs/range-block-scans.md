# Range block scans: follow-up experiments

## Implementation and invariants

The first two optimizations use `RangeQuery::as_doc_bitset`. The lazy scorer
follow-up below uses `RangeQuery::scorer` and `scorer_sync`. The column reader
owns copied block boundaries, decoding and ordinal remapping. The codec owns
header interpretation. The query owns comparisons and document membership.
Native sync, native async and WASM share these owners. No wire or persisted
format changes, new metadata cache, writer or schema option are introduced.

1. **Reject disjoint blocks using existing headers.** Constant, bitpacked and
   linear headers can imply conservative raw-value intervals without payload
   reads. Unknown, wrapping or unsupported intervals must scan normally.
   Local text ordinals must never be pruned before global remapping. Signed
   ranges initially use only exact constant bounds; other signed blocks scan.
   Missing sentinels never match. Query ranges still inspect the first value
   for multi-value fields, using their existing fallback. This adapts block
   summary pruning without adding a summary format or derived resident index.
2. **Carry a sequential BlockwiseLinear cursor.** Each 256-value read previously
   restarts the variable-length header walk at byte eight. A full scan therefore
   repeats prefixes of its 512-value records. Carry the current record index
   and byte offset across batches, resetting at each copied column block.
   Random reads must keep their existing behavior through the same decoder.
   Preserve residual interpolation, out-of-range zero fill where supported,
   ordinal remapping and first-error callback termination.

The first experiment adds O(copied blocks) header work and can avoid complete
payloads. The second reduces header traversal from quadratic to linear in the
number of codec records. Both keep constant scratch and the existing output
bitset. The cursor adds two `usize` fields (16 bytes of logical state on this arm64 host),
reused across batches and reset for each copied block. It retains no payload
and allocates no heap memory. Decode arithmetic and encoded bytes remain unchanged. Neither changes
an architecture-sensitive codec, planner or configuration default.

## Measurement protocol

`rust_hot_paths/range_scan_layouts` builds one/sixteen-block clustered and ordered columns,
an all-match control, whole missing blocks, and 1K/64K/1M piecewise-linear
columns. The latter assert actual BlockwiseLinear selection. Exact per-document
membership and encoded/output byte accounting are outside timing. Fixture
build/open/merge is excluded; output allocation and destruction are included.
Use the same host/compiler/flags and sequential before/after runs, with no
concurrent task-owned compilation during sampling. Keep the original range
benchmark as a shuffled-data control. Report limitations and unsuccessful
experiments as well as retained changes.

Required validation: codec bounds against decoded values including overflow;
sequential-vs-random decoding across every codec, variable batch sizes, record
boundaries and tails; existing merge/missing/numeric/multi-value regressions;
fallible scan cancellation and remapped ordinals; the search check harness,
native-without-sync regressions and WASM build/tests.

## Findings

Header pruning is useful for clustered bitpacked values spanning multiple copied
blocks. At 65,536 documents with sixteen blocks and a selective range, pruning
alone reduces materialization from 63.219 to 4.092 microseconds. The one-block
control selects BlockwiseLinear and cannot use these cheap bounds: 192.61 versus
193.83 microseconds, with no significant change. Whole missing blocks improve
from 103.22 to 91.921 microseconds. Ordered fixtures select BlockwiseLinear too;
header pruning does not explain their timing differences. The ordered sixteen-
block baseline has wide confidence intervals and is unsuitable for a precise
speedup claim. No codec selection or block sizing defaults were changed.

The cursor reduces the million-document piecewise case from 5.236 to 2.863 ms
relative to pruning alone (45% less time). At 64K documents it improves 191.723
to 178.728 microseconds; at 1K, 2.905 to 2.841 microseconds. These results support
eliminating repeated header traversal, with the gain increasing with record
count. The shared decoder still uses the original interpolation arithmetic.

| Fixture                 | Documents / copied blocks | Before (µs) | Pruning only (µs) | Pruning + cursor (µs) |
| ----------------------- | ------------------------- | ----------: | ----------------: | --------------------: |
| Clustered               | 65,536 / 16               |      63.219 |             4.092 |                 4.083 |
| Clustered               | 65,536 / 1                |     192.605 |           193.829 |               183.082 |
| Ordered                 | 65,536 / 1                |     195.826 |           192.950 |               180.102 |
| Ordered, noisy baseline | 65,536 / 16               |     217.174 |           184.040 |               180.560 |
| Half missing blocks     | 65,536 / 16               |     103.217 |            91.921 |                89.755 |
| All match               | 65,536 / 16               |     186.193 |           184.515 |               179.773 |
| Piecewise               | 1,024 / 1                 |       2.929 |             2.905 |                 2.841 |
| Piecewise               | 65,536 / 1                |     194.348 |           191.723 |               178.728 |
| Piecewise               | 1,048,576 / 1             |   5,171.634 |         5,235.662 |             2,862.518 |

The existing shuffled-data controls also pass exact membership checks and do
not regress: the four cases measure 21.35–21.67 microseconds after both changes,
versus the earlier word-materialization measurements of 24.29–27.86 microseconds.
These are longer-separated controls; their improvement is not attributed solely
to the cursor, which does not accelerate their bitpacked codec.

Encoded column sizes are identical in all runs: for example 98,464 bytes for the
clustered sixteen-block case and 452,617 bytes for the million-document case.
Output bitsets remain 8,192 and 131,072 bytes respectively. `ColumnBlock` (208 B)
and `FastFieldReader` (136 B) stay unchanged. Pruning allocates nothing; cursor
state is constant per scan. Three alternating process RSS runs of the complete
fixture/test driver measure 40.1–40.9 MiB with pruning and 39.3–40.0 MiB with the
cursor too. Setup and allocator noise dominate these small differences; this is
not evidence of lower retained heap or mmap residency. The original unpruned
process RSS was not measured in this follow-up.

The [evidence](benchmark-results/range-scans-2026-09-19/summary.json) includes
Criterion estimates/confidence intervals, commands, compiler/host/flags and
working-diff hashes. Raw logs and process-memory runs are adjacent. The
`range-pruned` baseline is a copy of Criterion's `new` directories from the
pruning-only runs. The clustered baseline was run separately after an initial
assertion incorrectly expected the one-block control to choose bitpacking; the
corrected assertion applies to the sixteen-block fixture. No timing from the
failed setup is used.

These fixtures measure warm range-bitset construction, including allocation
and destruction, on Apple M4 / Rust 1.98.1 with ordinary release flags. They do
not establish cold-storage, full-search, concurrent throughput or x86 speedups.
No new persistence format, block summaries, codec default or reorder policy is
justified by these measurements.

## Validation

Focused codec tests pass, including conservative header bounds across all 65
bit widths and wrapping/descending linear headers; sequential versus random
reads for every codec at nine batch sizes; record boundaries, tails, empty and
backward reads, and existing byte-compatible serializer checks. The range
regression compares merged global text ordinals with independent scorer output.
The complete `check` harness passes 2,029 tests (25 normally ignored), strict
Clippy, native-without-sync and standalone broker compilation. All three
async-only range regressions pass. The WASM release build and all 38 JavaScript tests pass. The
existing fallible/cancellable scan tests pass in the full suite.
Validation records and logs are alongside the measurement evidence.

Not run: the real-server `full` RPC harness (no RPC or lifecycle changes),
x86 benchmarks, cold-storage/concurrent/full-query measurements, and GPU
checks. No architecture-sensitive defaults changed.

## Incremental interpolation experiment (not integrated)

An exact quotient/remainder progression removes per-value signed 128-bit
multiplication and division from the BlockwiseLinear batch decoder. It improved
compressed scans substantially, but repeatable regressions in ordinary bitpacked
range filters disqualified it. Production retains the existing interpolation.
The experimental patches and measurements are preserved in the
[evidence](benchmark-results/range-interpolation-2026-09-19/summary.json).

For `delta = abs(last - first)` and `d = count - 1`, the prototype computes
`q = delta / d` and `r = delta % d` once. Each prediction advances by `q` plus a
remainder carry; descending records subtract the same magnitude. Initializing at
any batch offset preserves the original signed division's truncation toward zero.
The writer and scalar random reader remain independent references. The prototype
adds 48 bytes of logical batch state and no heap allocations or persisted bytes.

The arithmetic passed extreme endpoints, descending slopes, arbitrary offsets,
record tails, wrapping residual additions, and residual widths through 64 bits.
The two-inline-hint candidate passed the complete native `check` harness (2,031
tests, 25 ignored), strict Clippy and feature checks, three async-only range tests,
and the WASM release build with all 38 tests. On x86, all 31 codec tests and three
native range tests passed; every timed fixture also verifies exact membership.
The final kernel-boundary variant passed benchmark membership checks but did not
receive a separate full native/WASM run because it was also rejected.

Apple M4 alternating runs reduced the million-document piecewise scan from
3.325/3.643 ms to 1.614/1.684 ms. Unaffected controls were noisy and sometimes
slower, so a dedicated Cascade Lake machine repeated the comparison with matching
Summa 1.8.146 binaries, Rust 1.98.1, unchanged release flags and CPU affinity.
Both run orders confirmed the compressed-data improvement and the ordinary-filter
regression. Compiler outlining shifted between the shared codec dispatcher and
bitset packing; explicit inline hints did not remove the tradeoff. A final
variant additionally isolated the interpolation kernel with `inline(never)` and
still regressed the controls. No default or code-generation hint was retained.

Representative x86 results below are the arithmetic mean of the two run-level
Criterion slope estimates, in microseconds. Each candidate is compared only
with its own alternating baseline; raw per-run intervals are in the
[cloud summary](benchmark-results/range-interpolation-2026-09-19/cloud-summary.json).

| Fixture                                |     Before | Two inline hints | Before boundary trial | Kernel boundary |
| -------------------------------------- | ---------: | ---------------: | --------------------: | --------------: |
| Shuffled, 65,536 docs, one block, 1%   |    109.706 |          118.852 |               109.863 |         120.139 |
| Clustered, 65,536 docs, sixteen blocks |     15.223 |           15.752 |                15.184 |          15.753 |
| Piecewise, 1,048,576 docs, one block   | 17,711.073 |        4,815.838 |            17,714.172 |       4,812.806 |

All four shuffled controls regress 8.1–8.3% with the two-inline candidate and
9.4–9.6% with the kernel boundary. CPU 2 recorded zero steal ticks across all
eight measured processes. Peak process RSS for three alternating fixture/test
runs is 36.21–36.38 MiB before and 36.06–36.15 MiB after on x86; the M4 ranges
are 39.42–40.05 and 39.38–40.27 MiB. These small differences do not establish
retained-memory savings. Column encodings and output bitset sizes are unchanged.
The remote archive was downloaded and SHA256-verified before deleting the
isolated machine and its boot disk.

The x86 matrix uses 13 cases, two alternating before/after pairs, two-second
warmups and four-second measurements. The final boundary experiment uses six
cases with one-second warmups and two-second measurements. Both use 20 effective
Criterion samples: the benchmark group's setting overrides the first driver's
requested 30. Raw estimates retain their confidence intervals; repetitions are
reported separately rather than combined into a statistical significance claim.
The first local shuffled comparison accidentally used a saved previous-version
binary; it is excluded and replaced by matching-version binaries.

These are warm bitset-construction measurements, not full-query latency,
cold-storage behavior or concurrent throughput. The real-server RPC and GPU
checks were not run for this codec-only experiment. The source changes were
reverted after measurement; the earlier production validation remains applicable.

Two follow-ups were identified after the interpolation trial:

- Bounded batching in the lazy range scorer, implemented and measured below.
- Accepting fully covered block spans through the owning reader's scan protocol,
  preserving missing-value rejection and global text ordinal semantics. This
  remains unimplemented.

## Lazy range scorer batching

`RangeQuery::scorer` and `scorer_sync` share `RangeScorer`. Consecutive scalar
reads previously revisited BlockwiseLinear record headers for every document.
The scorer now switches from scalar probes to a bounded batch cursor after eight
consecutive probes. It retains one 64-bit membership mask and decodes at most 64
values into temporary scratch. A distant seek outside the cached batch resets
the scalar probe count; short scans and isolated probes avoid decoding ahead.

`SingleValueCursor` belongs to the fast-field reader and borrows one immutable
column. It reuses the existing codec cursor, resets at copied-block boundaries,
and returns at most one block's remaining values. Text ordinals use the same
batch remapping helper as full column scans. Multi-value columns retain scalar
first-value reads. No writer, persisted format, admission rule, query planner or
codec-selection policy changes. The scorer preserves inclusive numeric bounds,
missing values, global text ordinals, score bits, forward-only seeks and repeated
exhaustion. A column ending before the segment is treated as missing thereafter,
matching scalar reads.

On 64-bit builds the existing boxed scorer payload grows from 40 to 96 bytes;
there is no additional allocation. Decoded scratch is 512 bytes in a refill
call, independent of corpus size. `ColumnBlock` and `FastFieldReader` metadata
sizes remain unchanged, and encoded column payloads remain evictable. Three
alternating M4 fixture/test process runs peak at 37.89–40.05 MiB before and
38.50–39.31 MiB after; these noisy process-level measurements do not establish a
retained-memory saving.

Assembly inspection matters here: merely extracting a refill helper let LLVM
inline its scratch and vector setup back into every per-hit scan call. The
initial M4 scan frame was 848 bytes including saved registers. An explicit
`inline(never)` refill boundary isolates that work; an `inline` hint on the
small scan state machine removes an extra call from advance/seek. These hints
are local to the measured scorer, not general reader/codec policy.

The [measurement evidence](benchmark-results/lazy-range-2026-09-19/summary.json)
records matching Summa 1.8.147 / Rust 1.98.1 builds, unchanged release flags,
alternating before/after and after/before order, 20 Criterion samples, one-second
warmups and two-second measurements. M4 has intermittent background-load noise;
the isolated Cascade Lake runs pin execution to CPU 2. Full iteration includes
scorer construction and destruction. First-hit controls construct and immediately
drop the scorer; sparse-seek controls issue 65 targets spaced 1,021 documents
apart. All fixtures contain 65,536 documents. The expanded controls cover missing
values, multi-value first-value semantics and complete misses.

Mean of the two run-level Criterion point estimates, in microseconds. Linear
sampling uses the slope estimate; flat sampling uses the mean. Per-run confidence
intervals and samples remain in the archives; these averages are descriptive,
not pooled significance tests.

| Full iteration, 65,536 documents |  M4 before |   M4 after | x86 before |  x86 after |
| -------------------------------- | ---------: | ---------: | ---------: | ---------: |
| Shuffled                         |    339.045 |    156.154 |    829.109 |    330.960 |
| Piecewise compressed             |  6,322.629 |    393.718 | 16,913.621 |  1,391.267 |
| Constant, all match              |    290.351 |    260.807 |    798.591 |    515.909 |
| Missing values                   |    363.020 |    174.183 |    807.640 |    302.934 |
| Multi-value, first value         | 14,476.597 | 14,532.508 | 34,303.341 | 34,182.056 |
| Constant, no match               |    149.284 |     22.519 |    431.970 |    121.502 |

The tradeoff is explicit: x86 short first-hit controls cost
about 3.6–5.4 ns more, and 65 sparse seeks through cheap single-value columns
cost 0.06–0.15 microseconds more. The compressed seek case improves from
1,727.660 to 193.640 microseconds because gaps benefit from batching. Multi-value
full scans remain effectively unchanged. All five existing x86 bitset controls
avoid regression; their small gains are not attributed to lazy scorer batching.

The final x86 RSS pairs peak at 36.21–36.34 MiB before and 36.39–36.51 MiB after.
CPU 2 records zero steal ticks across all measured processes. The remote archive
was downloaded and SHA256-verified before deleting the machine and boot disk.

The complete `check` harness passes 2,031 tests (25 normally ignored), strict
Clippy, native-without-sync and standalone broker compilation. All four async-only
range tests pass. The WASM release build and all 38 JavaScript tests pass. An
initial benchmark-only Clippy finding was corrected before the complete rerun.

Correctness checks compare exact hit IDs and score bits across merged numeric and
text columns, record/batch/block tails, seeks inside and beyond cached batches,
backward/equal seeks and repeated termination. Reader tests additionally compare
scalar and cursor reads across codecs, backward reads, arbitrary scratch sizes
and untouched output tails. This is a warm in-memory scorer optimization; cold
I/O, concurrent full-query throughput, real-server RPC and GPU measurements are
outside this experiment. Fully covered block spans remain an unimplemented
follow-up. Multi-value offset/value batching also merits a separate reader-owned
experiment; the scalar multi-value benchmark shows why it may be worthwhile.
