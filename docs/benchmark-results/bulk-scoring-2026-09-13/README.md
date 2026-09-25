# Bulk-scoring evidence (2026-09-13)

See [the measured results and limitations](../../search-benchmark-bulk-results.md)
and [Lucene research](../../lucene-11-performance-research.md).

`raw-results.zip` contains full official/supplementary raw timing samples,
exactness gates, profiles, assembly excerpts, RSS reports, compiler/build logs,
index/corpus manifests and pinned research sources. `manifest.json` records
uncompressed member lengths and SHA-256 hashes. `official-speedups.svg` includes
all 962 queries and the complete 301-query union family, including regressions.

Source reconstruction starts at Summa commit
`ce2c96b945fccc4bac58ccc45bb4bc23b809773e`. Overlay the desired `sources/` snapshot;
these contain every changed Rust/Cargo/grammar input, including the benchmark
adapter. The public benchmark base is
`a7c75473e91746280c5f01e69bf594ece5fca560`. Build commands and toolchain details are
in the cloud scripts and logs. After extracting a historical overlay into a
previously built tree, touch `summa-core/src/lib.rs` and
`summa-core/examples/search_benchmark_game.rs` before building; this avoids
Cargo reusing a binary because archived source mtimes precede prior artifacts.
Verify binary hashes and keep builds outside search timing.

Snapshot meanings:

- `summa-ratios-v3`: prior ratio-bound implementation, this pass's baseline.
- `summa-score-window-v1`: Boolean score windows.
- `summa-score-window-v2`: contiguous term-run BM25 accumulation.
- `summa-phrase-bound-v1`: phrase competitive bound, subsequently rejected.
- `summa-score-window-v3`: phrase prototype plus standalone term windows.
- `summa-posting-validation-v1`: v3 plus structural posting validation.
- `summa-bulk-final-v1`: retained version; structural validation and standalone
  term windows, with the experimental phrase hint removed.

`final-paired` compares v2 (`before`) with combined v3 (`phrase`), retaining the
script's historical column name. The paired ARM v3 comparisons use the phrase
prototype as `before` and v3 as `phrase`. These are separate protocols from the
upstream sequential driver; compare implementations within a phase.

The phrase versions trade wins across workloads. Removing the hint did not
recover the ranked-union regression; the extra hook is left out pending a
controlled explanation of its net benefit. All variant results remain available.
The retained version passes 1,649 native tests, 20 WASM tests and all 1,676
ARM/x86 exact-count and exhaustive-ranking gates.

The final cloud archive contains 228 hash-verified evidence files. Both index
manifests are identical to the initial capture. Its full binary/profile archive
is also retained in `.context/bulk-final-cloud-evidence.tar.gz`:
SHA-256 `18cc7ac181e8fc61bafb1484b94bfc52f5edb12d425b3a28f34bc7d5da09fc39`,
117,701,187 bytes. The ZIP excludes large binaries and raw perf recordings but
includes source overlays, results, gates, RSS, relevant assembly and reports.

Discarded timing runs are preserved and labeled, including the first final-ARM
pass that overlapped a brief ZIP job. The accepted final-ARM repeat uses v2 as
`before` and retained code as `phrase`; that legacy JSON key does not imply
phrase pruning is enabled. Source names and build hashes identify each version.

Lucene source excerpts retain their Apache license headers; the pinned LICENSE
and NOTICE files accompany them under `research/`. These are research inputs,
not a claim that Lucene was benchmarked in this pass.

ZIP integrity: 662 members, 10,862,250 bytes, SHA-256
`f407485762b90515122be879d21eeb7ae53c366c0f1f997dd97ec640ab6ab911`.

The machine and boot disk were deleted after capture; see [the cleanup audit](cleanup.json).
