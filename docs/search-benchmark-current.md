# Current full-text benchmark comparison

## Latest: remaining RGB gap — September 17, 2026

The adopted reader improves official RGB top-10 **2.0%**, from 441.487 to
432.761 µs in the independent paired run. Lucene RGB takes 420.303 µs:
**Summa remains 3.0% slower; the parity target is not met.** These are geometric
means of per-query medians over seven passes and 962 official queries on the
frozen 5,032,104-document x86 fixture.

The changes remove optional candidate copying, avoid redundant bound navigation
and score-plane work, amortize small mapped windows, use list costs when choosing
candidate drivers, and prune losing mapped two-term intersections. Existing
scorers, formats, stable-ID tie handling and exact-count paths are reused.

Final-source validation passes **1,861 native tests and 32 WASM tests**, including
native-without-sync compilation. All 1,676 exact reference queries pass on ARM
and x86 with RGB on and off. The separate smaller ARM confirmation improves
RGB/RGB-off top-10 by 1.4%/1.8%; count-only timings remain effectively flat.

The complete x86 archive, source/binary identity and query/pass matrices have
been verified. RSS is essentially unchanged: 1,011.7 MiB after RGB top-10, with
5.79 MiB anonymous; most residency is mapped index data. Tradeoffs remain: RGB
phrase top-10 regresses 1.1%, and RGB-off supplemental top-1000 regresses 1.5%.
Nothing has been committed. The validation machine is confirmed **TERMINATED**.

[Implementation, rejected experiments, validation and verified evidence](search-performance-review.md#september-17-optional-probing-and-window-work).

## Earlier: RGB versus Lucene — September 16, 2026

The target is RGB top-10 faster than **Lucene BP/RGB**. The selected Summa
reader is **6.7% slower** in the final confirmation: 416.881 µs versus
390.625 µs. It improves 1.3% from the same-run predecessor (422.282 µs).
Top-1000 is 824.472 µs versus Lucene's 848.283 µs. The selected fix removes
duplicate two-term contribution work while preserving exact scores and CPU
admission. A Tantivy win alone does not meet the updated target.

The earlier all-command comparison below predates that two-term optimization.
It uses the same host, corpus and seven-pass warm serial protocol, geometric
means of per-query median microseconds; do not pool timings across runs:

| Operation             | Summa RGB | Lucene BP/RGB | Tantivy |
| --------------------- | --------: | ------------: | ------: |
| Top 10                |   424.943 |       393.534 | 532.048 |
| Top 1000              |   839.102 |       854.396 | 927.447 |
| Top 100 + exact count |   724.421 |      1078.629 | 875.129 |
| Exact count           |   349.461 |       378.291 | 428.260 |

The reader retains physical traversal through same-field composition, maps IDs
at collection, reuses typed conjunctions, switches proven OR tails to that
executor, and screens mapped batches before resolving IDs. Explicit reordering
now preserves compact posting headers and position directories. Graph-policy
experiments do not improve query latency and are not selected. A separate
same-permutation SIMD codec run is 2.0% slower than selected/original RGB for
top-10. It remains opt-in for storage and residency: final top-10 RSS is
856.00 MiB versus original RGB's 1019.28 and Lucene's
864.87 MiB. Compact/SIMD index sizes differ by 27.3%.

[Implementation, measured tradeoffs and evidence](search-rgb-repair.md).

## Earlier: block execution — September 16, 2026

**Summa is faster on all four official benchmark commands with RGB disabled.**
The selected `packed` reader uses unchanged compact postings with exact norms.
Same-run seven-pass comparison: 5,032,104 documents, 962 official queries;
geometric means of per-query median microseconds.

| Operation             | Starting Summa | Selected Summa | Tantivy | Summa / Tantivy |
| --------------------- | -------------: | -------------: | ------: | --------------: |
| Top 10                |        613.554 |        507.672 | 528.746 |          0.960× |
| Top 1000              |       1097.240 |        916.958 | 934.137 |          0.982× |
| Top 100 + exact count |       1166.682 |        800.381 | 875.478 |          0.914× |
| Exact count           |        477.260 |        406.881 | 431.714 |          0.942× |

Official-workload parity is reached in this warm serial benchmark. Supplemental
standalone queries, some individual query families and memory usage still lag;
this is not a universal per-query, cold-cache or concurrent-workload claim.
Byte norms and the combined compressed layout remain opt-in, with their tradeoffs
reported separately. No configuration defaults change.

See [implementation, separate workloads, memory and limitations](search-block-execution.md)
and the [verified evidence](benchmark-results/block-execution-2026-09-16/README.md).
Native validation passes 1,828 tests; exact references pass on ARM and x86 across
five layouts. WASM is not rebuilt, following the standing user instruction.

## Earlier: pruning and count setup fixes — September 16, 2026

Implemented collector-directed normalization setup, prepared BM25 bound constants,
cheap-bound-first refinement and cheaper strict-gap validation. The new combined
index uses existing compact/Simd4x/impact options with exact norms. RGB remains
disabled; no configuration defaults change. Tighter statistics require a new
index; the reader changes also apply to existing indexes.

The retained reader changes official top-10 from 663.043 to 659.971 µs
(-0.5%, effectively flat), still **1.155× Tantivy**. Byte-norm official COUNT
improves from 518.756 to 502.416 µs (-3.1%). The requested overheads are removed,
but the official performance gap is not closed. The combined index below is an
opt-in experiment; its standalone-term gain does not generalize to the official
workload.

On the final full-corpus run, combined-index top-10 changes **+4.0%** for the 962 official queries and **-26.6%** for all 1,676 queries. Official top-10 remains **1.208×** Tantivy. **Parity remains unmet.** COUNT normalization-table constructions fall from 1,539 to zero.

The combined configuration regresses the official workload and is retained as an opt-in experiment. Its aggregate gain comes from supplemental standalone terms; it is not a general replacement for the current codec policy.

Final official-workload microseconds (same-run geometric means):

| Operation             | Starting compact | Reader / same index | Byte norms before | Byte norms after | Combined index |  Tantivy |
| --------------------- | ---------------: | ------------------: | ----------------: | ---------------: | -------------: | -------: |
| Top 10                |          663.043 |             659.971 |           710.098 |          701.142 |        689.889 |  571.203 |
| Top 1000              |         1188.057 |            1185.953 |          1249.621 |         1251.817 |       1209.611 | 1055.114 |
| Top 100 + exact count |         1233.330 |            1232.457 |          1243.551 |         1244.729 |       1249.122 |  961.458 |
| Exact count           |          503.535 |             503.291 |           518.756 |          502.416 |        508.361 |  457.774 |

See the [final results, memory attribution and limitations](search-pruning-fixes.md) and [verified evidence archive](benchmark-results/pruning-fixes-2026-09-16/README.md). The comparisons below are earlier experiments preserved for context.

## Earlier work diagnosis — September 16, 2026

[Per-query work measurements](search-work-diagnosis.md) locate the largest
traversal gap: standalone top-10 decodes **8.29×** Tantivy's document blocks
(696/714 terms decode more). Legacy and compact/exact traversal work is identical.
Quantization slightly reduces scoring work, but the official COUNT pass builds
1,539 unused normalization tables. The diagnostic feature is off by default; the production
latency reference below predates the pruning follow-up above.

## Earlier compact formats and byte norms — September 16, 2026

Implemented opt-in compact posting/position directories and one-byte quantized
norms with lookup-based BM25 normalization. RGB is off. **Tantivy parity remains
unmet.** The compact-plus-byte-norm full-corpus time ratios to Tantivy are:
Top 10 **1.254×**, Top 1000 **1.242×**, Top 100 + exact count **1.323×**, Exact count **1.140×**.

Both `IndexConfig::compact_text` and `IndexConfig::quantized_norms` remain false
by default. The benchmark exposes `--compact-text` and `--quantized-norms`.
New indexes are required to measure these encodings. Old segments retain their
original scoring lengths; enabling an option does not reinterpret existing bytes.
Chunked and explicitly reorderable fields retain exact geometry/normalization.
See [format, merge and compatibility rules](compact-text-format.md).

## Same-run x86 latency

5,032,104 Wikipedia documents, 962 official queries and 714 standalone terms.
Rust 1.98.1 / LLVM 22.1.8, release LTO and native CPU flags, same Cascade Lake machine,
CPU 2. A sixth same-run engine retains the first byte-norm reader to isolate the final
header/payload cleanup. Seven rotated complete query passes follow at least ten seconds of warmup
per engine/command. Values below are geometric means of per-query median
microseconds. Starting Summa is the preserved September 16 admission/position
build. The reader-only control isolates code overhead on the unchanged index.
All Summa indexes use Rounded payloads, ratio/L1 bounds and no impact envelopes.
No build, full-file audit or memory run overlaps latency on the same host.

### Official workload

| Operation             | Starting Summa | New reader / old index | Compact / exact norms | Compact / byte norms | Tantivy |
| --------------------- | -------------: | ---------------------: | --------------------: | -------------------: | ------: |
| Top 10                |        599.138 |                603.573 |               596.942 |              629.555 | 501.876 |
| Top 1000              |       1067.240 |               1068.751 |              1061.968 |             1109.537 | 893.010 |
| Top 100 + exact count |       1102.459 |               1138.192 |              1122.861 |             1127.345 | 852.191 |
| Exact count           |        446.752 |                457.626 |               449.035 |              461.769 | 405.000 |

Against starting Summa, compact/exact changes: Top 10 -0.4%, Top 1000 -0.5%, Top 100 + exact count +1.9%, Exact count +0.5%.
Compact/byte changes: Top 10 +5.1%, Top 1000 +4.0%, Top 100 + exact count +2.3%, Exact count +3.4%.
The final reader also shares the descriptor and L0 borrow and reuses decoded
block header/payload lookups. Against the first byte-norm reader in this same
run, final byte-norm latency changes are: Top 10 -2.5%, Top 1000 -1.6%, Top 100 + exact count -2.6%, Exact count -2.5%.
These are direct comparisons, not multiplied improvements from separate runs.
The old-index reader control changes +0.7/+0.1/+3.2/+2.4% in the same command
order; these remaining reader costs also affect indexes using the old formats.

### Supplemental standalone terms

| Operation             | Starting Summa | New reader / old index | Compact / exact norms | Compact / byte norms | Tantivy |
| --------------------- | -------------: | ---------------------: | --------------------: | -------------------: | ------: |
| Top 10                |        136.871 |                135.887 |               136.139 |              141.341 |  50.331 |
| Top 1000              |        566.836 |                569.913 |               569.903 |              650.732 | 424.354 |
| Top 100 + exact count |        259.474 |                260.671 |               261.881 |              260.305 | 255.833 |
| Exact count           |         13.141 |                 13.005 |                12.941 |               12.871 |   9.109 |

Compact/byte changes: Top 10 +3.3%, Top 1000 +14.8%, Top 100 + exact count +0.3%, Exact count -2.1%.
The final reader cleanup regresses standalone byte-norm top-1000 5.5% versus
the first byte-norm reader; that cost is retained alongside its official gains.
Per-family movements, faster/slower pass counts and p50/p95/p99 of query medians
are retained in the [evidence](benchmark-results/compact-text-2026-09-16/README.md).
These are warm single-CPU measurements, not concurrent-service tail latency.

## ARM cross-check

100,000-document fixture on Apple M4, same compiler/flags and seven rotated
passes. Shared Mac load limits small-difference claims. There is no ARM Tantivy
comparison in this run.

| Operation             | Starting Summa | New reader / old index | Compact / exact norms | Compact / byte norms |
| --------------------- | -------------: | ---------------------: | --------------------: | -------------------: |
| Top 10                |         32.769 |                 32.739 |                33.151 |               32.674 |
| Top 1000              |         48.015 |                 48.339 |                48.268 |               48.997 |
| Top 100 + exact count |         37.498 |                 37.448 |                37.954 |               37.336 |
| Exact count           |         25.587 |                 25.757 |                26.179 |               25.925 |

Compact/byte official changes: Top 10 -0.3%, Top 1000 +2.0%, Top 100 + exact count -0.4%, Exact count +1.3%.
The mixed ARM results do not support a default change.

## File bytes and memory

Every encoded payload, block boundary, codec and document ID is unchanged by
compact directories. Full-corpus payload/geometry SHA-256 is
`bc82eebf366f209f4485dd90932d2c304c1cd4935b13bb027a5c515acd4067d3` across all three layouts.
Compact directories save **281.62 MiB**;
byte norms save another **4.80 MiB** (10,064,208 → 5,032,104 payload bytes).
The norm saving is much smaller than the posting/position directory saving.

| Component          | Legacy MiB | Compact MiB | Saved MiB |
| ------------------ | ---------: | ----------: | --------: |
| Postings total     |   2101.873 |    2005.284 |    96.589 |
| Postings metadata  |    652.434 |     555.846 |    96.589 |
| Positions total    |   2625.295 |    2440.267 |   185.028 |
| Positions metadata |    339.989 |     154.961 |   185.028 |

Separate fresh-process residency runs perform three passes per command. The
following snapshots are after the final COUNT pass; anonymous memory is a subset
of RSS. Full mapping attribution is archived. File size, mmap residency and heap
memory are distinct measurements.

| Workload / engine        | RSS MiB | Anonymous MiB |
| ------------------------ | ------: | ------------: |
| official / before        | 1034.65 |          6.16 |
| official / reader        | 1034.68 |          6.17 |
| official / compact       |  816.10 |          6.18 |
| official / quantized     |  811.29 |          6.23 |
| official / tantivy       |  574.09 |          0.77 |
| supplemental / before    |  362.23 |          5.13 |
| supplemental / reader    |  362.26 |          5.14 |
| supplemental / compact   |  355.97 |          5.16 |
| supplemental / quantized |  351.02 |          5.17 |
| supplemental / tantivy   |  230.16 |          0.56 |

The position mapping falls from 671.06 to 458.94 MiB and postings from 286.94
to 280.38 MiB. Most of the RSS saving comes from position pages; byte norms
account for another 4.80 MiB. Anonymous residency is effectively unchanged.

The remaining payloads still use Rounded widths and existing tails/term footers.
This revision removes metadata overhead; it does not implement Tantivy's complete
posting representation. See the [baseline component comparison](text-format-comparison.md).

## Ranking and correctness

Compact/exact ordered IDs, raw score bits and counts match the preserved legacy
oracle. Every new fixture also passes all 1,676 queries against an independent
exhaustive collector, including optimized top-k/exact-count at k=10/100/1000.
Document order is identical across all layouts on both architectures.

Quantization changes scores and rankings. Full-corpus mean set overlap with
exact norms among nonempty queries is **95.15% at top 10**,
**97.44% at top 100**, and
**98.46% at top 1000**.
ARM top-10 overlap is 97.55%. These are ranking differences, not a relevance
assessment. Counts remain exact. Unit tests independently compare all 256 lookup
entries with canonical BM25 score bits across parameters, boosts and frequencies.

Runtime validation remains necessary to reject invalid widths, offsets and
block structure. Compact formats validate separate metadata without scanning
payload pages; selected decoded blocks retain content checks. Tests are a
separate validation layer covering old/new/mixed streams, corruption, budgets,
cancellation, copy merges and row compaction.

## Checks and reproducibility

- Final `python3 scripts/check_search.py full`: **1,816 passed, 25 ignored**, plus
  **4 real-server tests passed**. Formatting, Clippy, native no-sync, portable
  core and documentation builds pass. Portable compilation retains the existing
  `set_document_units` dead-code warning.
- Final x86 core: **1,599 passed, 16 ignored**; compact/norm integration passes.
- The initial parallel harness hit a broker discovery timeout. Its serial rerun
  and the final full run pass; all logs are retained.
- WASM was not rebuilt, following the user's instruction. Cold/concurrent
  performance and judged relevance are unmeasured.

The final 335-file source overlay changes 22 files from the starting build.
The ARM norm-gather prototype was rejected; it is not in this source.
SHA-256: `fa9e4bf6c205058c84f35717015c4527e1aa3d19196e11b0dd18a79c52ff2260`.
[Source snapshots, raw timings, exact oracles, byte manifests, checks and memory
snapshots](benchmark-results/compact-text-2026-09-16/README.md). Earlier selected
measurements remain in the [admission evidence](benchmark-results/admission-2026-09-16/README.md).
