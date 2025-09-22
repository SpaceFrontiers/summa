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

## Validation outcome

- `RUST_TEST_THREADS=2 python3 scripts/check_search.py check` passes formatting,
  strict Clippy, 2,025 tests (25 intentionally ignored), native-without-sync
  compilation and standalone broker compilation with core writers disabled.
- `cargo test --locked -p summa-core --no-default-features --features native
--test range_bitset` passes both range integration regressions.
- `cd summa-wasm && bash build.sh && npm ci && npm test -- --run` passes the
  release build and all 38 tests in seven files.

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

The `full` harness's separately enabled real-server RPC tests were not run:
this change does not touch RPC or lifecycle protocols. Linux residency and x86
performance checks were not run; neither is claimed by these measurements.
