# Range bitset word materialization

## Source review and selection

The local reference engine review examined posting format dispatch and its
frame-of-reference, StreamVByte and presence containers; range-attribute
summaries and predicate packing; split blob/hash storage and residency policy;
sparse quantization; and recursive graph bisection ordering.

| Technique                                      | Summa fit and decision                                                                                                                                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Four-lane 128-integer FOR blocks               | Already available as opt-in Simd4x. Existing whole-corpus results show a space/latency tradeoff; retain the measured policy.                                                                                                                     |
| StreamVByte with separate control/data streams | Potentially useful for irregular gaps and short tails. Requires a new posting format signal, bounded decoding, position integration and copied-run merge support. Defer until real gap histograms justify it against Rounded/Packed/Pfor/Simd4x. |
| Sparse/bitmap/all presence containers          | Useful for dense Boolean-only lists, but cannot replace BM25 frequencies or positions. Summa already defers frequency decoding. A separate persisted representation needs storage and phrase-workload evidence.                                  |
| Fixed-width range columns with block min/max   | Can skip disjoint blocks, but replacing compressed fast fields increases payload bytes and zigzag signed values do not preserve ordering. A future compact summary directory must preserve missing/first-value semantics and copied blocks.      |
| Pack range comparisons into bitset words       | Direct fit: batch decoding already exists, but materialization branches and read-modify-writes per matching document. Selected experiment below.                                                                                                 |
| Split compact lookup metadata from payload     | Summa already uses SSTable/FST directories, immutable byte views and budgeted metadata residency. Replacing ordered dictionaries with hashes loses prefix/range traversal; do not import whole-payload locking.                                  |
| Quantized sparse integer accumulation          | Summa already has quantized BMP payloads and exact forward rescoring. Per-term scales need cross-segment score and recall validation; not a drop-in exact optimization.                                                                          |
| Recursive bisection document ordering          | Existing Summa bounded reorder owns this responsibility. Reference presets are workload/machine-specific and do not justify changing defaults.                                                                                                   |

These are source observations, not benchmark claims about the reference engine.
No reference implementation or additional dependency is copied into Summa.

## Implemented experiment

Public entry: `RangeQuery::as_doc_bitset`, used by Boolean/BMP filter planning.
`FastFieldReader` owns decoding and ordinal remapping; `RangeQuery` owns bound
semantics; `DocBitset` owns the document mask. Native synchronous, native async
and WASM all use the same materializer. Sparse random probes and range scorers
retain their existing paths.

A crate-private fallible batch visitor extends the existing single-value
scanner. There is one decoder and one ordinal-remapping implementation; the
existing per-value visitor wraps it and still stops on the first callback error.
The numeric bound compiles once, and its raw/signed variant is selected once per
batch. Concrete predicates compare decoded values without per-hit bitset writes
and insert up to 64 matches per mask. Copied fast-field blocks may start at any
document offset, so a mask can span two output words. Final padding stays zero.

Invariant: identical document membership, including absent columns, the missing
sentinel, reversed/open bounds, zigzag signed values, sortable floats, and the
first value of multi-value fields. No writer, wire, metadata-version, scoring,
or query-planning default changes. Persisted bytes and copy-through merges are
unchanged by construction; existing byte-preservation tests remain applicable.

Cost: O(documents) comparisons and O(documents/64 + copied blocks) mask stores,
the existing ceil(documents/64) output words, 2 KiB decode scratch and at most
64 bytes of comparison scratch. No per-batch allocation or corpus-sized derived
metadata. Decoder costs are unchanged, including the previously documented
repeated prefix walk in BlockwiseLinear; this is not a claim that every codec's
full scan has linear complexity.

## Validation and measurement protocol

Compare the real materializer before/after with the existing `rust_hot_paths`
65,536-document fixtures: one/sixteen copied blocks, 1%/50% selectivity, same
compiler, flags and host. Keep building/opening and exact membership checks
outside timing; allocation/destruction remain inside. Inspect optimized code at
the actual range call site. Extend coverage to all bit offsets and short tails,
numeric types and cancellation/ordinal-remapping boundaries. Run the search
check harness, native-without-sync tests and the WASM build/tests.

Report warm CPU latency separately from whole-request/cold-I/O latency. Retained
bitset/scratch bounds are not process RSS. No codec or architecture-sensitive
default will be selected from this single-host experiment.

## Code-generation evidence

The actual release range-scan instantiation on Apple M4 / Rust 1.98.1 uses
NEON `cmhs.2d` comparisons for raw unsigned values and `cmge.2d` for signed
values. The generic predicates disappear; there are no indirect callback calls
in the scan. The previous implementation used scalar per-value type selection
and conditional bitset read/modify/writes. This result costs code size: the scan
instantiation grows from 1,020 to 5,252 bytes; the outer range materializer stays
668 bytes. No unsafe operations, explicit SIMD intrinsics, target-CPU flags, or
inlining attributes are introduced. These measurements describe this compiler's
output, not a cross-platform vectorization guarantee.

## Measured results — September 19

Apple M4 (10 logical CPUs), macOS 15.6.1, Rust 1.98.1 / LLVM 22.1.8,
release benchmark profile, default sync features, empty custom compiler flags.
Baseline and candidate both use `c11169b4` after updating the workspace; the
earlier pre-update baseline is excluded. The benchmark fixture is unchanged.
Criterion uses 20 samples, three seconds of warmup and five seconds of sampling
per case. Central slope estimates for 65,536 documents:

| Copied blocks | Matches | Before, µs | After, µs | Less time |
| ------------: | ------: | ---------: | --------: | --------: |
|             1 |     ~1% |     46.484 |    26.572 |     42.8% |
|             1 |     50% |     69.720 |    27.274 |     60.9% |
|            16 |     ~1% |     45.905 |    27.864 |     39.3% |
|            16 |     50% |     71.869 |    24.288 |     66.2% |

All cases run an exact per-document membership oracle outside timing. The
unchanged random-probe control varies from 5.2% slower to 5.8% faster; desktop
and VPN activity shared the machine. No task-owned compilation/test workload
ran during sampling. The large improvements support retaining this portable
materialization change; small differences between the candidate cases are not
meaningful. These are warm RAM filter-construction results, not end-to-end
search, p99, cold mmap, concurrent ingestion, or x86 results. Signed/floating
types have correctness coverage but no separate timing claim.

Output storage remains 8,192 bytes for the fixture. Source-level scratch is
2,048 bytes of decoded values plus 64 bytes of comparison flags, with no new
heap allocation. Separate `/usr/bin/time -l <binary> --test range_bitset` runs
measure maximum process RSS of 20,889,600 / 18,776,064 bytes and peak footprint of
12,665,384 / 12,812,792 bytes before/after. These include fixture build/open and
runtime costs, are sensitive to process/page state, and do **not** demonstrate
a memory saving. There is no stored-byte size change.

Retained evidence: [summary and environment](benchmark-results/range-word-2026-09-19/summary.json),
[before output](benchmark-results/range-word-2026-09-19/before.txt),
[after output](benchmark-results/range-word-2026-09-19/after.txt),
[before memory](benchmark-results/range-word-2026-09-19/before-memory.txt), and
[after memory](benchmark-results/range-word-2026-09-19/after-memory.txt).
Full local Criterion distributions, source review hashes and assembly are under
`.context/search-harness/criterion/` and `.context/range-word/`.

```sh
python3 scripts/check_search.py bench --bench rust_hot_paths --filter range_bitset --save-baseline range-main-before
# Apply the materialization change, with the same fixture/compiler/flags.
python3 scripts/check_search.py bench --bench rust_hot_paths --filter range_bitset --baseline range-main-before
```

Remaining work: evaluate word packing on x86 and end-to-end filtered vector
workloads; evaluate range summaries on ordered/missing-heavy real columns before
adding metadata. StreamVByte, presence containers and per-term quantization stay
research candidates, with the semantic and measurement gates in the table above.

## Validation outcome

- `RUST_TEST_THREADS=2 python3 scripts/check_search.py check` passes formatting,
  strict Clippy, 2,025 tests (25 intentionally ignored), native-without-sync
  compilation and standalone broker compilation with core writers disabled.
- `cargo test --locked -p summa-core --no-default-features --features native
--test range_bitset` passes both range integration regressions.
- `cd summa-wasm && bash build.sh && npm ci && npm test -- --run` passes the
  release build and all 38 tests in seven files.
- Documentation links/benchmark inventory and `git diff --check` pass.

The new regressions cover every bit offset, lengths 0–257, neighboring matches,
padding and 64 copied blocks. The existing typed-range regression compares
bitsets to expected numeric/first-value membership and native/async scorer
results, including missing blocks and merge tails. Existing fallible numeric
and remapped-text scans still stop at the requested document. Existing codec
round-trip and merge byte-preservation tests pass in the harness.

The first harness run, with four test threads and overlapping async-only
compilation, hit the existing BMP severe-backlog test's 10-second compaction
timeout; 1,794 other core unit tests passed. The unchanged failing test passed
in isolation in 3.98 seconds, then passed with the entire harness at two test
threads and no overlapping build. No timeout or production merge behavior was
changed. Both attempts are retained in the [validation record](benchmark-results/range-word-2026-09-19/validation.json)
and [isolated retry output](benchmark-results/range-word-2026-09-19/backlog-isolated.txt).

The `full` harness's separately enabled real-server RPC tests were not run:
this change does not touch RPC or lifecycle protocols. Linux residency and x86
performance checks were not run; neither is claimed by these measurements.
