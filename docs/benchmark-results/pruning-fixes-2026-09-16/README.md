# Pruning and count setup fixes — September 16, 2026

See the [implementation and measured results](../../search-pruning-fixes.md)
and [current benchmark comparison](../../search-benchmark-current.md).
RGB is disabled. No format or configuration defaults change.

## Evidence

results.zip (local archive `results.zip`) contains raw latency samples, process residency
snapshots, ideal-threshold work probes, verification summaries, build/check logs,
source overlays, patches and runners. [manifest.json](manifest.json) maps each
logical file to its ZIP member, byte length and SHA-256. Identical files share a
member. All members and the final ZIP are verified. Binaries and indexes are
excluded; binary hashes, index byte totals and reference hashes are retained.

The final selected run is `arm/latency-selected/` and `cloud/latency-selected/`.
The final source is `source/selected-source.tar.gz`, with the regression tests
and exact changed-file manifest in `source/final-overlay.tar.gz` and
`source/final-source-manifest.json`. The implementation patch is relative to
`source/before-source.tar.gz`, the reader present when this task began.
Existing workspace changes are retained. Source overlays are applied to this
repository; they are not independent Cargo workspaces.

Earlier stages are retained for attribution:

- `norm` adds the subsequently rejected lazy table cache.
- `bounds` also prepares BM25 query constants.
- `refinement` also skips impact refinement when a cheap bound rejects.
- `packed` also removes the redundant strict-gap ordering scan.
- `rejection` adds the rejected inverse-threshold predicate and replaces lazy
  norm setup with the retained collector-intent hint.
- `selected` removes the inverse predicate. It retains eager table access for
  ranked collection and skips setup for membership collection.

Do not report a prototype's timings as the selected source's performance.
`latency-rejection-lazy` on ARM predates collector intent. Isolation runs measure
one stage at a time. Final changes relative to starting Summa come from the
same final run, not multiplied improvements from different phases.

## Protocol

Full corpus: 5,032,104 Wikipedia documents. ARM: the first 100,000 documents.
Both use 962 official queries followed by 714 standalone terms, Rust 1.98.1,
release LTO and `-C target-cpu=native`. Seven rotated complete passes follow at
least five seconds of warmup per engine/command. Values are geometric means of
per-query median microseconds, including benchmark protocol overhead. The x86
processes use CPU 2. Builds, verification and memory audits run outside timings.
Small ARM changes are subject to shared Mac load and protocol overhead.

The new combined index uses compact directories, Simd4x gaps and complete impact
bounds, with exact norms, the same corpus order, one indexing thread, a 2 GB
writer buffer, no background merges and an explicit final merge. Existing ratio,
byte-norm, impact and Simd4x fixtures are preserved controls. Proof-cache budget
is 262,144 bytes; term-cache limits are 8,192 blocks / 4,194,304 bytes.

Each selected reader is checked on all five layouts against preserved exhaustive
references: 1,676 queries, top-10/100/1000 exact ordered document IDs and score
bits, and top-100 with exact counts. Byte norms use their own represented-length
oracle. These checks establish reader equivalence; Summa and Tantivy scores
are not required to have identical floating-point bits. The timed protocol also
checks every returned count. The diagnostic feature is absent from timed builds.

The ideal-threshold probe uses an existing seed API and the actual exhaustive
k-th score. It verifies exact top-10 results; it diagnoses threshold warmup and
is not a production algorithm that knows answers beforehand. Encoded byte work
is not physical disk I/O. The fresh-process memory audit records three complete
official passes per command, with `/proc` mappings, anonymous/file attribution
and rollups. RSS and anonymous residency are separate from encoded index size.

## Exclusions and checks

An initial candidate build reused stale Cargo timestamps after source extraction;
those timings were discarded. The preserved build runner touches extracted Rust
sources, and retained candidate binaries have distinct recorded hashes.
A cloud Simd4x fixture with a different document order was excluded and replaced
with the matching-order fixture. An obsolete ARM fixture with invalid position
tails was likewise excluded. Neither is used for a selected latency or correctness
claim. Failed setup logs remain identifiable as diagnostics.

Native formatting, Clippy, tests and the native build without default sync
features are recorded under `checks/`. Corruption, merge and exact-score
regressions remain enabled. WASM builds are skipped per the user's instruction.
The benchmark machine is stopped after the exported evidence is verified locally.
The rejected inverse candidate was stopped after complete top-10 and top-1000
measurements; its unfinished commands are excluded. All its child processes
were drained before compiling and measuring the selected reader. Its separate
planned residency audit was canceled; `memory-selected` is the final audit.

The benchmark machine was confirmed `TERMINATED` after local export verification.
