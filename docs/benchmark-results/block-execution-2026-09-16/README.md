# Posting block execution — September 16, 2026

See [implementation and results](../../search-block-execution.md) and the
[current comparison](../../search-benchmark-current.md). RGB is disabled.
Index formats and configuration defaults remain unchanged.

## Evidence and reproduction

results.zip (local archive `results.zip`) contains raw per-query timings, correctness summaries,
source overlays, implementation patches, scripts, build/check logs, CPU sample
reports, actual caller assembly and process residency snapshots.
[manifest.json](manifest.json) maps logical paths to deduplicated archive members,
byte lengths and SHA-256. Binary and index contents are excluded; hashes identify
them. Every member and the ZIP are verified after packaging.

The selected source is `arm/packed-source.tar.gz`;
`arm/packed-source-manifest.json` identifies its files. The initial confirmation
uses the preceding `inline` source, preserved separately. Apply the overlay to this
repository, including its existing workspace changes. It is not a standalone
Cargo workspace. `arm/current-vs-before.patch` compares against the frozen
starting source, also included. Source candidates and logs from unsuccessful
experiments remain evidence, not selected performance claims.

The `cloud/final-confirmation/` and `arm/final-confirmation/` directories contain seven
rotated official passes after at least ten seconds of warmup per engine/command.
The `final-supplemental/` directories contain five passes of the 714 standalone queries
after at least three seconds of warmup. The 962 official queries are always
reported separately. Every timed count is checked against a reference. Top-k
protocol acknowledgements are supplemented by separate exact-reference checks
for ordered IDs, raw score bits and exact counts across all 1,676 queries.

The x86 fixture contains 5,032,104 Wikipedia documents; ARM uses the first
100,000. Rust 1.98.1 / LLVM 22.1.8, release LTO and `-C target-cpu=native` are
shared across candidates and controls. Engine processes on x86 are pinned to
CPU 2; the Python protocol driver is not pinned. Builds, profiles, full-file
hash audits and memory measurements do not overlap timed runs on the same host.
Times include protocol overhead. Small ARM changes are sensitive to shared Mac
load. Timed binaries do not enable `query-diagnostics`.

Summa cache budgets are 262,144 bytes of posting validation proofs and
8,192 term blocks / 4,194,304 term-cache bytes. `packed` uses the unchanged compact
exact-norm index; `packed-norm` uses byte norms; `packed-combined` uses existing
compact directories, Simd4x gaps and impact bounds with exact norms. No RGB.
Byte norms have their own represented-length scoring reference. Scores need not
be bit-identical between Summa and Tantivy.

`cloud/final-memory/` records three complete official passes for each of the four
commands in a fresh process per engine, including `/proc` mappings, smaps and
anonymous/locked attribution. The opened snapshot follows one small explicit
query; it is not advertised as zero-query RSS. Commands run sequentially in that
process, so later snapshots include pages touched by earlier commands. These
memory runs make no latency claim.

## Attribution and validation

The cumulative candidates are `phrase`, rejected global `block`, `collect`,
`norm`, `gather`, `seek`, `intersect`, `retain`, `bytes`, `mixed`, `counted`, `heap`,
`reuse`, `multi`, `inline`, and the selected `packed` heap. The separate `memo`
phrase-cache candidate is an experiment and is not in the selected source. The final handoff wrapper fix is in `multi` and
`inline`; earlier counted-stage source must not be selected wholesale. Stage
screens isolate hypotheses, and their gains must not be multiplied. The
same-run final before/after comparison is the retained performance evidence.

`arm/arm-work-summary.json` reports diagnostic work, not production latency.
Original and refreshed profiles use CPU-clock sampling with DWARF stacks;
hardware PMU events were unavailable. Refreshed profiles describe the counted
stage before the final buffer reuse and longer-conjunction changes. Assembly
files show actual release call sites on both architectures.

The final native harness, native-without-sync, portable compilation and exact
query references are recorded alongside source/binary hashes. The portable
build retains the existing unused `DocLengths::set_document_units` warning.
WASM was not rebuilt, following the standing user instruction. No lifecycle/RPC
change required the full real-server suite. Cold-cache search and concurrent
indexing/merge were not measured. The byte-view Miri crate contains the extracted
owner implementation, provenance, three tests and its pinned lock file; its
build directory is excluded.
