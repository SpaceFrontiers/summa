# Wikipedia search benchmark results (2026-09-13)

Summa improves by **1.42× for TOP_10**, **1.45× for TOP_1000**, and
**5.12× for exact COUNT** on the official workload. Exact counts agree with both
reference engines, but Summa still trails Tantivy by 2.24–3.59× across the five
commands. Frequent standalone-term ranking remains the largest gap.

This is the full 5,032,104-document Search Benchmark, the Game workload, with
962 official queries, measured on a dedicated GCloud `n2-highmem-8` machine (Intel
Xeon 2.80 GHz, 64 GiB RAM). Search drivers and children were pinned to CPU 2;
indexing, compilation and profiling did not overlap timed search. Both Rust
engines used rustc 1.98.1 / LLVM 22.1.8, native CPU instructions and release LTO.
The main before/after runs reuse **identical persisted Summa index bytes**.
Each official command uses 60 seconds of warmup and ten repetitions. Ratios use
the geometric mean of per-query median latencies, not a selected best query.

Base Summa revision is `ce2c96b945fccc4bac58ccc45bb4bc23b809773e`; the baseline
uses the same strict native parser entry point and adapter as the optimized candidate.
Upstream is pinned to `a7c75473e91746280c5f01e69bf594ece5fca560`. Query SHA-256 is
`a8e8f2c1a4bede53063de22fb13be83b585683da1c28208da4921de73ce254bd`; transformed
corpus SHA-256 is `b11c3c90041e4b88296f19c10b6469590bac200b76a4094a3b455e22e9bc71eb`.
See [the protocol and reproduction commands](search-benchmark-game.md) and
[adapter commands](benchmarks.md#search-benchmark-the-game).

## Exactness and implemented changes

Exact counts are feasible. The ordinary search API reports scored documents;
that is not a hit count. `COUNT` uses exhaustive membership, `TOP_100_COUNT`
combines exact membership with top-k, and `--exhaustive` also disables score
pruning for the three ordinary ranked commands. Dictionary frequency is used
only when it is an exact answer: an indexed, non-chunked term with no deletions
and a collector that accepts an aggregate count. Ordinary queries with one
required term and only optional text terms share that term's membership, so the
same guarded shortcut applies without changing optional score contributions.
This is the same general
optimization used by Tantivy, not a cached query answer.

- The native parser now represents required/prohibited clauses and group scope
  explicitly, with strict error handling. Previously a required-term query fell
  through to plain-text OR matching (four matches instead of three in the
  regression). Keyword boundaries also preserve words such as NOTHING, ANDROID
  and ORCHID; the old grammar parsed NOTHING as NOT HING. See the shared
  [query syntax](query-language.md).
- Saturated 16-bit TF bounds now fall back to the full-width list maximum.
  A TF of 100,000 previously produced an unsafe pruning bound. Encoded bytes
  remain unchanged.
- The block-MaxScore executor's BM25 arithmetic now calls the shared formula; an algebraically rearranged formula
  previously changed a tied winner from document 0 to 1. Union scores also retain
  input summation order: the full-corpus gate caught a top-1000 mismatch for
  `west palm beach florida`, reproduced by an 8,192-document regression. Final
  verification requires equal score bits and ordered document IDs.
- Exhaustive text unions stream into collectors instead of retaining and sorting
  every hit. Score-free collection uses bounded 4096-ID membership windows in the
  existing term/Boolean cursors for pure unions. Conjunctions and exclusions keep
  their selective scalar/two-phase driver after a broader batching experiment
  regressed them. Count collectors
  accept population counts; other score-free collectors retain ascending IDs.
  Scoreful, phrase, chunk and filtering paths preserve their existing contracts.
- Conjunctions use the rarest known required cursor. Phrase matching separates
  cheap posting alignment from exact positional verification, reusing the existing
  verifier. The regression reduces confirmation attempts from 35 to four for the
  same two matches. Exact phrase cursors also remain exhausted after termination;
  an old seek could resurrect a match. Cardinality estimates now saturate and term
  cursors expose their existing frequency for planning.
- Nearby seeks reuse the decoded posting block; distant seeks gallop through the
  existing skip hierarchy. Rounded decoding reuses the existing fused SIMD gap
  kernels with an explicit gap convention. Packed and PFor low-bit decoding now
  share the existing bounded decoder. A guard-page regression reproduced an
  out-of-bounds read in the old public exact-width unpacker; it passes on ARM and
  x86 after the fix. Posting serialization remains byte-identical.
- Ordinary ranked terms retain streaming traversal after the complete 714-term
  supplement showed it outperforming a new block-MaxScore route. The improved
  shared decoder and seek paths remain in use.
- Canonical union score scratch uses presence bits instead of clearing unused
  float slots. It is bounded by 1 MiB + 32 KiB for 64 terms, independent of corpus
  size. Dense and sparse local fixtures improved by 18.4% and 61.4% respectively;
  those are separate M4 microbenchmarks, not full-corpus speed claims.

The adapter exposes indexing threads, indexing memory, existing posting codec,
dictionary cache capacity, and exhaustive ranking through CLI options. No
query-specific tuning, stop-word
removal, query cache, approximate candidate budget, corpus change or production
codec-default change was introduced.

## Default-settings results

All 962 exact counts agree with both reference engines. Pruned Summa top-10,
top-100 and top-1000 have identical scores and ordered document IDs to exhaustive
Summa scoring. Every result below includes parsing and the local pipe round-trip;
document hydration is excluded. Latencies are geometric means of per-query
medians in microseconds. A speedup above 1 is faster; an engine latency ratio
above 1 is slower.

| Command       | Summa before µs | Summa after µs | Tantivy µs | Lucene µs | Before/after speedup |
| ------------- | --------------: | -------------: | ---------: | --------: | -------------------: |
| TOP_10        |           2,110 |          1,485 |        552 |       558 |                1.42× |
| TOP_100       |           2,577 |          1,772 |        720 |       774 |                1.45× |
| TOP_1000      |           3,033 |          2,092 |        933 |     1,151 |                1.45× |
| TOP_100_COUNT |           5,196 |          3,097 |        862 |     1,382 |                1.68× |
| COUNT         |           5,051 |            987 |        426 |       485 |                5.12× |

Summa remains slower overall than both reference engines. The main comparison
keeps the existing 256-block dictionary-cache default. Tantivy's command-level
control latencies moved between -8.1% and -1.6% versus the earlier baseline;
small percentage differences should not be treated as decisive wins.

| Query family       | Queries | TOP_10 speedup | TOP_10 Summa/Tantivy | COUNT speedup | COUNT Summa/Tantivy |
| ------------------ | ------: | -------------: | -------------------: | ------------: | ------------------: |
| term               |       1 |          1.15× |               32.92× |      1808.48× |               2.45× |
| union              |     301 |          1.01× |                2.29× |        23.88× |               1.69× |
| intersection       |     300 |          1.69× |                3.03× |         2.10× |               3.15× |
| phrase             |     300 |          1.66× |                2.75× |         1.78× |               2.62× |
| intersection_union |      40 |          1.50× |                2.71× |       122.55× |               0.85× |
| negated            |      19 |          1.40× |                3.18× |         2.62× |               3.08× |
| two-phase-critic   |       1 |          9.73× |                2.03× |         9.80× |               2.08× |

The upstream `intersection_union` family consists of required-plus-optional
queries such as `+climate policy`. Optional terms affect scores, but not membership,
so the new count-equivalence hook removes unnecessary traversal. The official
`term` family contains only `the`, and `two-phase-critic` only `+"the who" +uk`.
Those two n=1 rows must not be generalized; the complete 714-term supplement is
reported separately.

## Exhaustive ranking and supplemental terms

`--exhaustive` disables score pruning using the same public collector. Counts
are exact in both modes; exhaustive ranking changes traversal cost, not answers.

| Command  | Exhaustive Summa µs | Exhaustive/pruned latency |
| -------- | ------------------: | ------------------------: |
| TOP_10   |               2,904 |                     1.96× |
| TOP_100  |               3,055 |                     1.72× |
| TOP_1000 |               3,172 |                     1.52× |

The supplement uses **all 714 distinct lower-case ASCII terms from the official
query set**, generated before timing, with ten seconds of warmup and five
repetitions. It is separate from the official aggregate and includes every term,
not a selection by performance. All supplemental counts and pruned/exhaustive
scores and document IDs pass the same verification gate.

| Command  | Final Summa µs | Before/after speedup | Summa/Tantivy latency |
| -------- | -------------: | -------------------: | --------------------: |
| TOP_10   |          987.3 |                1.12× |                20.23× |
| TOP_1000 |        1,444.1 |                1.06× |                 3.46× |
| COUNT    |           54.5 |               16.76× |                 8.16× |

The term-plan selection used the complete supplement too. The first new pruning
path was 23–27% slower than baseline. Avoiding dense scratch clearing removed
most of that regression, but streaming terms with the shared decoder/seek fixes
was another 11% faster for TOP_10 and 19% faster for TOP_1000. The final code
retains streaming terms. This rejects the losing execution policy rather than
changing thresholds for selected terms. Intermediate selection samples are
preserved separately from the final measurements above.

## Dictionary cache budget

This is a separate configuration of the same final binary and same persisted
index: 1,024 dictionary blocks instead of the unchanged default of 256. It caches
decompressed term metadata, not query results. Every query is verified again.
The repeated warm workload favors retaining its working set; these results do
not establish a universal cache default.

| Workload  | Command       | Cache-1024 Summa µs | Default/cache-1024 speedup |
| --------- | ------------- | ------------------: | -------------------------: |
| Official  | TOP_10        |             1,334.8 |                      1.11× |
| Official  | TOP_100       |             1,551.0 |                      1.14× |
| Official  | TOP_1000      |             1,911.2 |                      1.09× |
| Official  | TOP_100_COUNT |             2,730.4 |                      1.13× |
| Official  | COUNT         |               828.2 |                      1.19× |
| 714 terms | TOP_10        |               818.9 |                      1.21× |
| 714 terms | TOP_1000      |             1,281.8 |                      1.13× |
| 714 terms | COUNT         |                11.5 |                      4.74× |

## Memory and persisted size

Peak process RSS from `/usr/bin/time -v`, in MiB, for the official runs:

| Command       | Summa before | Summa final | Summa cache-1024 | Tantivy | Lucene |
| ------------- | -----------: | ----------: | ---------------: | ------: | -----: |
| TOP_10        |        960.5 |       962.5 |            968.6 |   703.2 |  972.8 |
| TOP_100       |        960.8 |       962.2 |            968.2 |   703.3 |  989.5 |
| TOP_1000      |        961.1 |       962.1 |            968.8 |   703.2 |  989.5 |
| TOP_100_COUNT |      1,036.5 |       962.1 |            967.9 |   703.3 |  981.5 |
| COUNT         |      1,036.9 |       934.0 |            939.4 |   693.1 |  962.6 |

The bounded allocation regression separately falls from 746,157 to 7,490 total
allocated bytes on the x86 fixture (99.6× less allocation volume). That is not a
measurement of process RSS or full-corpus heap size.

| Index         |         Bytes | Relative to original Summa |
| ------------- | ------------: | -------------------------: |
| Summa rounded | 5,322,696,537 |                     1.000× |
| Tantivy 0.26  | 3,029,870,002 |                     0.569× |
| Lucene 10.4.0 | 2,806,588,208 |                     0.527× |
| Summa packed  | 4,800,242,810 |                     0.902× |
| Summa pfor    | 4,772,404,644 |                     0.897× |

The original Summa index uses 2,871,883,607 bytes for positions (54%) and
2,341,034,236 for postings (44%). Posting codecs therefore cannot remove most of
the gap by themselves. Dictionary and other metadata, the stored ID and the
numeric fast field account for the remainder.

## Existing posting-codec experiment

Both alternatives use the existing production codecs and a separately rebuilt
full index, with four workers and the same builder budget. Builds and timed
searches are serialized. All 962 exact counts and pruned/exhaustive ranked
results pass per-index verification. Rebuilding can change document layout,
so these are whole-index comparisons, not isolated decoder benchmarks.

| Command       | Rounded Summa µs | Packed Summa µs | PFor Summa µs |
| ------------- | ---------------: | --------------: | ------------: |
| TOP_10        |            1,485 |           2,032 |         2,238 |
| TOP_100       |            1,772 |           2,496 |         2,735 |
| TOP_1000      |            2,092 |           2,964 |         3,266 |
| TOP_100_COUNT |            3,097 |           4,087 |         4,506 |
| COUNT         |              987 |           1,625 |         1,876 |

| Separate build | Elapsed | Peak RSS MiB | Official search peak RSS MiB |
| -------------- | ------: | -----------: | ---------------------------: |
| packed         | 4:14.94 |     12,988.0 |                        886.3 |
| pfor           | 4:21.48 |     13,264.0 |                        874.2 |

Packed saves 9.8% of total index bytes but increases ranked latency by 32–42%
and COUNT by 65%. PFor saves 10.3%, with ranked latency 46–56% higher and COUNT
90% higher. Search RSS is lower for both compressed indexes. These whole-index
measurements do not isolate decoder CPU from rebuilt document layout.
The production default remains rounded.

## Remaining costs and format proposals

The final result is not a claim that Summa is the fastest engine. Profiles and
query-family measurements identify useful next work. In the final selection
supplement, frequent terms remain especially expensive: TOP_10 for `in` scans
4,006,219 matches in about 52 ms, versus about 0.31 ms for Tantivy. This example
illustrates the pruning gap; the complete 714-term aggregate is reported above
and is the basis for comparison.

- **Block bounds:** the earlier MaxScore candidate's standalone-term diagnostic
  visits 1,317,696 windows and
  skips only 43,283 (3.28%); the median per-term skip fraction is 1.68%.
  These are counters from the existing diagnostic tool, not its timings; the
  final ordinary-term path is scalar. The counters explain the rejected
  pruning route's weakness.
  Summa combines maximum TF and minimum length even when they belong to
  different documents. That is safe but can produce loose bounds. Lucene's
  [competitive impact accumulator](https://github.com/apache/lucene/blob/releases/lucene/10.4.0/lucene/core/src/java/org/apache/lucene/codecs/CompetitiveImpactAccumulator.java)
  retains competitive frequency/norm pairs. A bounded Pareto frontier is a
  candidate to evaluate, with conservative rounding, global statistics and
  configurable BM25 preserved. It needs a versioned format, bounded resident
  metadata, merge-copy semantics and exact ranking tests. No speedup is claimed
  for this unimplemented proposal.
- **Dictionary residency:** the separate 714-term COUNT CPU profile attributes
  47.93% of samples to Zstandard sequence decompression with 256 cache blocks.
  With 1,024 blocks, decompression disappears from the leading samples. The
  profiles have zero lost samples, but only about 2,800 and 383 samples;
  use the complete latency runs for performance comparisons. These profiles
  precede the final term-plan selection; the cache timings above use the final
  binary. A cache-budget tradeoff is already available through the existing configuration and CLI.
- **Packed decoding:** the existing
  [`BitPacker1x` library decoder](https://docs.rs/bitpacking/0.9.3/bitpacking/struct.BitPacker1x.html)
  is a concrete reuse candidate for complete 32-value horizontal groups. A
  separate little-endian M4 probe passes 8,448 layout-oracle/value/canary cases.
  Production-byte, short-tail, unaligned/guard-page, endian and full-query
  validation remain required. This is an unintegrated candidate, with no
  latency claim or new Summa dependency; see the
  [evaluation design](search-benchmark-game.md#existing-library-decoder-candidate).
- **Positions:** positions occupy 54% of the original Summa index. Reuse the
  existing exact-width bit packer before considering another codec. Any future
  position payload change needs format-version rejection, bounded decoding,
  phrase/ordinal parity and merge-copy tests. The offline size estimate below
  is not a tested new format or a query-speed result.

## Comparison limits

The three engines index the same transformed corpus and pass exact membership
checks for every official query. They retain their upstream analyzers and BM25
settings: Summa/Tantivy use k1=1.2, b=0.75; Lucene's adapter uses k1=0.9, b=0.4.
Tantivy drops tokens longer than 40 characters; Lucene splits overlong tokens at
255; Summa retains them. Norm quantization also differs. These differences are
recorded in the [protocol](search-benchmark-game.md#analyzer-and-scoring-differences),
so cross-engine ranking equivalence is not claimed. Within Summa, the exactness
gate compares ordered document identities and `f32::to_bits()` at all three
ranked depths against exhaustive scoring.

Baseline Summa had unsafe saturated TF bounds and floating-point ranking
inconsistencies. Its timings are useful for measuring the engineering changes,
but it is not a fully rank-correct alternative to the final implementation.
Invalid and interrupted intermediate runs are retained as failure evidence and
excluded from the result tables. No query-specific branches, cached query
answers, corpus omissions or approximate result limits were used.

Search is one CPU per engine, with parsing and pipe transport included. Index
open is outside the warmed timed samples; result hydration is excluded. The
machine has enough RAM for these indexes, so these are warm-cache results,
not cold storage or concurrent-server throughput. RSS includes mapped pages
and is not equivalent to heap allocation or cache capacity. Summa/Tantivy
indexing uses four workers and a 2,000,000,000-byte builder budget;
Lucene retains eight upstream workers and a 1024 MiB writer RAM buffer. Initial
indexing jobs overlapped, so no cross-engine indexing-speed claim is made.
The separate posting-codec builds are serialized and have their own measurements.
The indexing CLI memory setting is a builder flush budget, not an RSS cap: the
original Summa indexing process peaked at 14,099,616 KiB including the final
merge. This remains an important memory cost to investigate separately; the
current work does not change the writer or merge lifecycle.

## Validation and reproducibility

The final scalar-policy source passes `python3 scripts/check_search.py check`:
1,630 tests, 26 ignored, 23 suites, plus formatting, Clippy and native compilation
without sync support. The WASM release build and all 20 WASM tests pass. The
cloud x86 core suite passes 1,442 tests (17 ignored), the three benchmark
integration suites pass, and all five 144-request smoke runs pass (three codecs,
cache disabled, cache 1024). Invalid cache capacity is rejected before serving.
The complete 962-query and 714-term gates require bit-identical ranked scores
and ordered document identities. The final binary is byte-identical to the
validated scalar candidate, SHA-256
`58ca420457bdaeb542d8d6ade4073143ea88c43f7fe72d26e0fdd712dec69d2e`.

Local check logs are `.context/search-check-scalar.log` and
`.context/wasm-scalar-{build,test}.log`; cloud logs and source manifests are in
the evidence archive. Documentation/link checks, Python lint and `git diff --check` also pass in the final workspace.

The adapter, corpus transformation, strict verifier, supplemental-query generator
and raw-sample analyzer are in `scripts/search_benchmark/` and
`summa-core/examples/search_benchmark_game.rs`. See the
[reproduction commands](benchmarks.md#search-benchmark-the-game).
The [versioned results bundle](benchmark-results/search-game-2026-09-13/README.md)
contains raw samples and per-file checksums. The full local evidence archive is
`.context/summa-benchmark-evidence.tar.gz`, verified SHA-256
`f9697d68d15aca5357ab4dd85293b484041d9453d5cb044383de1aa2bf17d001`.
It contains each frozen source overlay and SHA-256 manifest,
base revision, binary hashes, compiler/CPU/OS details, full raw query samples,
strict verification records, profiles, memory logs and pipeline scripts. Preserve
`.context/summa-baseline.bundle` and `.context/native-parser-baseline-v4.tar.gz`
with that evidence to recreate the paired baseline. The final workspace overlay
and its manifest are also saved locally as `.context/final-workspace-overlay.tar.gz`
and `.context/final-workspace-manifest.json`.

The benchmark machine and its auto-delete boot disk were removed after the full
archive checksum and all frozen binary checksums were verified. No benchmark
cloud resources remain.

The portable decoder tests include all bit widths 0–32, lengths 0–128, output
canaries, and an OS guard-page subprocess reproducing the old over-read on ARM
and x86. Search regressions cover repeated window slots, exact score bits,
deletions, collector fallback, sync/async parity, phrase termination and
cancellation. Format-preserving decoder/seek changes compare encoded bytes.
`check_search.py full` was not run because this work does not change lifecycle or
RPC behavior. An earlier parallel check encountered a broker discovery timeout;
the isolated retry and subsequent serial complete checks passed without broker
changes.

## Offline position-width estimate

The diagnostic decodes one existing 128-value block at a time through the
production reader, validates the complete current byte accounting, and retains
the existing block boundaries, headers, index entries and footer sizes. It writes
no new index and measures no query latency.

Across 3,964,752 positioned terms, 26,151,287 blocks and 1,330,791,236 values,
exact-width payloads would occupy an estimated **2,303,239,795 bytes**, versus
2,871,883,607 current position bytes: **19.8% less position storage**, or
**10.7% of the original whole index**. This is only a size estimate; format
compatibility and phrase performance remain unimplemented work.
