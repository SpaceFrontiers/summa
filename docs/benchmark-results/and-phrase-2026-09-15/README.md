# AND and phrase follow-up — September 15, 2026

Evidence for the follow-up to [the previous integrated comparison](../parity-2026-09-15/README.md).
RGB is disabled for all performance measurements. Initial screens start from
that exact frozen source and compare against its preserved selected binary;
the integrated and AVX2 stages below also include intermediate controls.

## Reproduction

`results.zip` includes complete 332-file source overlays, actual build/test and
driver scripts, source/binary hashes, query inputs, raw seven-pass timings,
ordered document/score-bit/count oracles, process RSS and index hashes. It
excludes executables and the copyrighted corpus. Rebuild each variant with the
same Rust 1.98.1 compiler, `-C target-cpu=native`, release LTO, and original
fixture. Source overlays are written with fresh mtimes and every build must
report `Compiling summa-core`; reusing Cargo artifacts after preserving source
mtimes is invalid. Native ARM uses the canonical and copy-merged 262,144-document
fixtures. Cloud x86 uses the unchanged 5,032,104-document fixture and CPU 2.

The initial `bound` trial includes all four commands on official and supplemental
workloads. `seed` screens official top-10/top-1000; `gallop`, `pair` and `countseek` screen
official top-10/count. A screened subset cannot establish overall parity.
Separate variant manifests and raw results distinguish these protocols.

Family profiles compare the selected pre-experiment binary and Tantivy on AND
top-10 and phrase top-10/count. Profiles warm for at least five seconds then
sample CPU time for at least ten seconds, with DWARF stacks. Profile timings are
not used as latency evidence. Hardware PMU events were unsupported; unresolved
stack frames limit inclusive attribution. Raw self reports and process memory
are retained; large `perf.data` captures remain on the stopped owned machine.

Every logical file is listed in `manifest.json`, with size, SHA-256 and
`archive_path`. Byte-identical files share a stored ZIP entry. All downloaded
archives and every evidence member are hash-verified, then the final ZIP is
read back and checked against its manifest.

## Selected source and final outcome

`avx2/source.tar.gz` and its 332-file manifest are the selected source, verified
against the main workspace after application. `avx2/selected.patch` isolates the
three retained changes against the previous round's exact source: galloping
within posting blocks, borrowed two-cursor phrase alignment, and an eight-lane
packed-mask x86 seek kernel. SSE2, NEON and portable paths retain their contracts;
there are no format, codec-default, executor, allocation or pruning changes.

`and-phrase-final-evidence` first measures the integrated phrase/galloping build
against the original and Tantivy across the complete matrix. The final
`and-phrase-avx2-evidence` rotates FOUR engines across both workloads and all four
commands: `original` is the starting build, `before` is that exact intermediate
binary, `after` is the selected AVX2 build, and `tantivy` is Tantivy 0.26.
`analysis.json` reports incremental after/before changes; `total-analysis.json`
reports after/original changes and all seven paired pass ratios. Percentages from
separate runs must not be multiplied or substituted for this final comparison.

Official selected/original time changes are -3.2% top-10, -2.6% top-1000, -0.5%
top-100 plus count, and -0.3% count. Ranked AND top-10 improves 5.7% and phrase
2.9%, both in all seven passes. AND top-100 plus count regresses 1.6% and AND
count 2.4%, also in all seven passes. Overall count is essentially flat. Selected
Summa/Tantivy ratios remain 1.22 / 1.23 / 1.49 / 1.39: parity is not achieved.
See [the current comparison](../../search-benchmark-current.md) for full tables,
ARM results, process RSS and limitations. RGB remains disabled throughout.

`bound` and `seed` conjunction pruning and the `countseek` portable reduction
were rejected after measured regressions. `gallop` and `pair` are screens only;
the selected claim comes from the full integrated run. Source and evidence from
every rejected candidate remain archived. The final Cascade Lake native
assembly lowers the AVX2 source expression to AVX-512VL compare/mask instructions;
older AVX2-only CPU performance was not measured.

## Validation and limitations

Every selected ARM and x86 layout preserves all 1,676 ordered ID/raw-score/count
oracles, and pruned top-k agrees with exhaustive scoring. Index bytes remain
unchanged, including all 40 files across four ARM layouts. Core scoring differs
between engines, so cross-engine score/rank identity is not claimed.

The final native check passes 1,794 tests with 25 ignored, formatting, Clippy,
and native without sync. The first selected-source check hit two broker backend
registration timeouts. `avx2/native-check-first/` preserves that failure, and
`avx2/broker-retry.log` records all 13 broker integration tests passing on retry.
The subsequent complete check is preserved under `avx2/native-check/`.
Selected x86 validation passes 1,578 core tests (16 ignored) and all-target
Clippy. Portable compilation passes with an existing dead-code warning for
`ChunkMapBuilder::set_document_units` in the no-native configuration.
WASM is skipped under the standing user instruction. Lifecycle/RPC, cold-cache,
concurrent-ingest, p95/p99 and the upstream runner are not new measurements here.

`cloud-stopped.log` confirms the owned machine is TERMINATED. The stop command lost
its connection while polling; direct status verification resolved its state.
All downloads and evidence members were verified before stopping the machine.
