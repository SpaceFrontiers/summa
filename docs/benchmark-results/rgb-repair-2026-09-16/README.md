# RGB execution repair evidence — September 16, 2026

[Results and implementation](../../search-rgb-repair.md).

## Artifacts

results.zip (local archive `results.zip`) contains the initial, intermediate and final experiments, raw per-query
samples, work counters, source overlays, scripts, exact-reference checks, memory
snapshots, index/source audits and native validation logs.
[manifest.json](manifest.json) records every logical path, its deduplicated ZIP
member, byte length and SHA-256. Restore a logical file using its `archive_path`;
identical files are stored only once. Every member is verified after packaging.
Binaries and index payloads are omitted; their hashes identify them.

`initial` is the collection-boundary repair. `intermediate` adds typed
conjunctions and cardinality-based counts. `handoff` removes inverse binary
searches and the redundant heap. `selected-reader` adds guarded OR tails and
mapped batch admission. `compact-layout` adds compact-format preservation and
records its measured source overlay. `pair-sums-probe` adds the selected two-term
contribution elision. `delivered` adds expanded test coverage; its production
scorer code is byte-identical to the measured pair-sums source. Graph and reader probes retain
their own source overlays and remain separate from the selected implementation.

`layout-comparison` uses the same selected reader across original, compact,
frequent-term and finer-partition RGB indexes, plus RGB-off, Tantivy and Lucene
controls. Its memory snapshots follow latency. `lucene-profile` contains sampled
CPU attribution outside timing. Rejected experiments are retained for audit.

`simd-layout` and `simd-comparison` measure the existing codec on the same
permutation and selected reader, with exact norms. The x86 latency result is
flat, while index size and resident payload memory improve. Reader probes use
unchanged original RGB indexes, with their own before/after binaries; their
results must not be pooled with layout timings from another run.

The frozen `packed` baseline source, canonical references and original fixture
construction are documented in the
[RGB-off evidence](../block-execution-2026-09-16/README.md) and
[RGB fixture evidence](../rgb-2026-09-16/README.md).

## Reproduction

1. Restore the selected `delivered/arm/source.tar.gz` overlay on the recorded
   checkout. Touch extracted source files before Cargo to prevent stale-mtime
   artifact reuse. Build with Rust 1.98.1, release LTO and native CPU flags.
2. Copy the included `verify_reference.rs` to
   `summa-core/examples/verify_reference.rs` and build that example with the same
   flags. Run it against the preserved oracle. All 1,676
   queries must preserve ordered IDs, raw score bits and exact counts. Do not
   compare raw scores across engines with different scoring configurations.
3. Rebuild the compact RGB fixture from the immutable identity fixture using the
   included `fixture.py`. Audit unchanged payloads and the identical permutation.
   Graph variants deliberately produce different permutations and have their own
   audited indexes; they are not defaults.
4. Run `layout-comparison/arm/measure.py arm|cloud`. It captures all four commands,
   seven official passes and five supplemental passes with rotating engine order.
   The scripts identify every binary, index, compiler and warmup setting.
5. Run the separate memory collector after timing, then audit index/source hashes.
   Profiles and work counters are attribution measurements, not latency samples.
6. Run the native harness and focused diagnostics. Portable compilation is
   recorded; WASM was skipped under the standing instruction.

Scripts capture workspace and machine paths; adapt them deliberately. Do not overlap
latency with builds, work diagnostics or memory sampling on the same machine.
The rejected stale-build admission run and overlapping ARM warmup are excluded
from all performance claims; their rejection is documented in the review.

## Lucene target

`lucene/cloud` records a separate comparison with the pinned upstream Lucene
10.4.0 BP adapter. Its index uses the same canonical full corpus; adapters retain
their upstream analyzers and BM25 settings. Query counts are checked for all
1,676 queries before timing. Official top-10 gets 60 seconds of whole-workload
warmup per engine and seven rotated passes; supplemental terms get ten seconds
and five passes. Indexing and separate memory sampling do not overlap latency.
Source, dependency, index and corpus hashes identify the Lucene fixture.

## Final selection and confirmation

`pair-memory/cloud` contains the final four-engine ranked confirmation, followed
by separate memory collection and immutable index audits. Engine keys are
`before-rgb`, `selected-rgb`, `selected-simd-rgb` and `lucene-rgb`; it is not a
seven-engine run. The attempted late control addition failed its precondition
before changing any running process or measurement script.

The final selected RGB top-10 result is 416.881 µs versus Lucene's 390.625 µs,
so the target remains unmet by 6.7%. SIMD increases selected-reader top-10 by
2.0%, but reduces RSS from 1019.28 to 856.00 MiB (Lucene: 864.87 MiB).
All final query/pass dimensions and 145 exported files are verified. The selected
x86 verifier separately checks all 1,676 canonical references on the SIMD index
after latency and memory collection.

`inline-sync-probe` is diagnostic only: caller-thread execution would bypass
the shared CPU bound under concurrent sync calls. It is not selected. The
ratio-group, canonical-reduction and combined-window probes are also rejected.
The root release binary is byte-identical to the measured ARM pair-sums binary;
its verification covers RGB-off, original RGB and SIMD RGB. The stale first
release artifact was detected by hash before use and rebuilt from verified
selected sources. `delivered` records the rejection, fresh build and verification.

The machine is independently verified **TERMINATED** after all required downloads
and correctness checks. Changes remain uncommitted. Shutdown polling lost its
connection; the subsequent independent status check confirms completion.
