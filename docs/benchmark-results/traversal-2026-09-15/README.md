# Traversal, exact counting, memory and formats — September 15, 2026

## Outcome

RGB off. Direct integrated before/after/Tantivy comparison, not multiplied
isolated gains: official top-10 -2.2%, top-1000 -2.3%, top-100 plus exact count
-10.2%, exact count -18.8%, all seven passes faster. Summa/Tantivy ratios are
1.211/1.211/1.345/1.137. Standalone top-100 plus count improves 66.4% to 1.002×
Tantivy; standalone top-10 remains 2.783×. Overall parity is still unmet.

Tradeoffs include phrase top-10 +1.8%, phrase top-100 plus count +1.9%, and
negated top-10 +7.6%, all seven slower. All queries, repetitions and grouped
quantiles are retained. No memory reduction is claimed from traversal changes.
The memory audit attributes 96.2% of the official RSS gap to resident positions
and postings, rather than anonymous allocations. Format accounting is separate
from latency: hypothetical packed sizes are not speedup measurements.

## Reproduction and archive layout

`results.zip` is content-deduplicated. `manifest.json` maps every logical file to
its stored `archive_path`, exact byte length and SHA-256. Extraction scripts must
resolve that mapping rather than assume every logical name is physically stored.
Every logical member and the completed ZIP itself are verified.

- `traversal-final/`: combined source overlay (332 files), selected-input hashes,
  nine-file patch, exact main-source verification, ARM/full-corpus timings and
  all oracles, tests, Clippy, native harness, portable check, summarizers and
  index manifests. The baseline is the preceding round's AVX2 selected binary.
- `block-intersection`, `count-batches`, `scored-batches`, `union-count`,
  `exact-top-count`: retained isolated candidates, dependency bases, decisions,
  complete source, exact oracles and paired timings.
- `impact-first`, `exact-candidates`, `cached-positions`, `scored-mask`,
  `posting-simd`, `phrase-blocks`, `length-gather`: rejected candidates with
  reasons and measured evidence. ARM-rejected candidates did not run on cloud.
- `single-profile/`: fixed-workload CPU profiles, actual binary assembly and
  unsupported hardware-counter attempts. Large raw perf.data and executables
  are excluded; identities and text reports are retained.
- `memory-audit/`: 401 verified snapshots/protocol files, per-mapping residency,
  process status, rollups, binary/index hashes and analysis. Fresh processes;
  three passes per command, separate official/supplemental workloads.
- `format-audit/`: read-only byte counters, quantization statistics, scripts,
  pinned Tantivy source/license and diagnostic build errors with corrections.
- `inputs/`: exact workloads, original oracle source and ARM byte manifests.

Linux: Cascade Lake, Rust 1.98.1 / LLVM 22.1.8, native CPU flags and release LTO,
4 build jobs, 1 query CPU (CPU 2). Same 5,032,104 documents and document order,
962 official queries, 714 supplemental terms, RGB off. Seven rotated full passes
after at least ten seconds of warmup per engine/command. Summaries use geometric
means of per-query medians; p50/p95/p99 summarize those query medians, not a
production concurrent-service latency distribution. Builds, full byte scans and
memory auditing never overlap latency on the same host.

ARM: 262,144 documents, canonical/copy-merged Rounded timing and corresponding
grouped-impact correctness fixtures, same compiler/flags and preserved baseline.
Shared Mac load limits interpretation of small changes and cross-run times.

## Validation and limitations

Native harness: 1,804 pass, 25 ignored; formatting, Clippy and native-without-sync
pass. Portable compilation passes with the preexisting `set_document_units`
dead-code warning. ARM isolated core 1,587 pass, x86 1,588 pass, 16 ignored each.
Four ARM and two x86 layouts each preserve 1,676 ordered raw-score/count oracles
and compare ranked/separate-exact-count top-k at k=10/100/1000 against exhaustive
collection. All immutable index bytes match after timing and audits.

WASM is not rebuilt per user instruction. Cold-cache, concurrent merge/ingest,
lifecycle/RPC `full`, and other CPU architectures are unrun for this traversal
change. Hardware cache/branch counters are unavailable on the machine. Existing RGB
merge/lifecycle work is separate and unfinished. No format/default or norm
precision change is included. Source changes remain uncommitted.

## Preserved diagnostic corrections

An early exact-candidate experiment skipped timed term confirmation and was
withdrawn before production integration; its regression test and rejected source
are retained. The ungated separate-count experiment regressed ARM and was
replaced by the measured gated candidate. A cached-position runner initially
wrote the wrong completion marker; the prior artifact was already verified and
unchanged, the dependent runner was stopped before execution, and marker repair
plus original scripts are retained. Format probes needed two compile-only fixes
for the upstream example's Rust 2018 imports/doc-comment placement; no index or
latency result was changed and failed logs remain in the archive.

## Files and reports

- [Current results](../../search-benchmark-current.md)
- [Format and residency findings](../../text-format-comparison.md)
- [Performance review](../../search-performance-review.md)
- Verified archive (local archive `results.zip`)
- [SHA-256 manifest](manifest.json)

All timing, memory and byte audit archives are verified locally. The owned
validation machine is confirmed **TERMINATED**. The stop operation's polling
connection reset; a separate direct status query confirmed termination, and
both logs are preserved.
