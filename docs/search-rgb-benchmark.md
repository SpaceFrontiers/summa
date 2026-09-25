# Standalone text RGB benchmark — September 16, 2026

This measures the existing standalone Recursive Graph Bisection (RGB) feature
with the selected `packed` search reader frozen. It is separate from the
[RGB-disabled Tantivy win](search-block-execution.md). The current RGB execution
path has a significant locality regression; this report does not present it as
a completed performance optimization.

The follow-up [execution repair](search-rgb-repair.md) fixes same-field physical
traversal and measures the result on these frozen index files. This document
preserves the original regression evidence.

## Protocol and correctness

- ARM: Apple M4, first 100,000 canonical Wikipedia documents. x86: Cascade Lake,
  all 5,032,104 documents, engine processes pinned to CPU 2; driver unpinned.
- Rust 1.98.1 / LLVM 22.1.8, release LTO, `-C target-cpu=native`. Same frozen
  reader binary and cache limits across Summa layouts. No concurrent builds,
  diagnostics, index audits or memory sampling during latency runs.
- `compact`: selected RGB-off compact/exact index. `identity`: freshly built
  eligible plain-text field with identity mapping. `rgb`: its separately
  reordered copy. Eligibility changes the persisted schema only for new indexes.
- Rounded posting codec, ratio bounds, 4 KiB dictionary blocks, one indexing
  worker, 2,000,000,000-byte indexing budget, background merges disabled. The
  standalone reorder uses the existing 24 GiB BP budget and full convergence
  policy. No index-format, scorer or lifecycle implementation changes.
- Official: 962 queries, seven rotating passes. Supplemental: 714 standalone
  terms, five rotating passes. x86 reports ranked top-10 and top-1000; ARM
  reports those plus top-100 with exact count and count-only. Official warmup
  is at least ten seconds per engine/command.
  Supplemental warmup is at least three seconds. Times below are geometric means of
  per-query median µs, including the subprocess protocol. Ratios above one are
  slower. This is a warm serial workload, not a cold/concurrent benchmark.
- Both layouts on both hosts passed all 1,676 frozen references: ordered
  top-10/100/1000 IDs and raw score bits, complete top-100 plus exact counts.
  Phrase queries are included. Control file hashes and preserved stored-field,
  fast-field and row-statistic byte hashes passed the post-reorder audit.

## Results

### Full corpus, x86

| Operation | compact | identity |      rgb | tantivy | RGB / compact |
| --------- | ------: | -------: | -------: | ------: | ------------: |
| Top 10    | 499.001 |  675.705 | 2410.780 | 517.486 |        4.831× |
| Top 1000  | 909.733 | 1106.567 | 3916.042 | 925.268 |        4.305× |

### Supplemental standalone terms, x86

| Operation | compact | identity |     rgb | tantivy | RGB / compact |
| --------- | ------: | -------: | ------: | ------: | ------------: |
| Top 10    | 130.736 |  291.585 |  98.892 |  54.634 |        0.756× |
| Top 1000  | 564.187 |  927.458 | 854.243 | 435.009 |        1.514× |

### ARM official workload

| Operation             | compact | identity |     rgb | RGB / compact |
| --------------------- | ------: | -------: | ------: | ------------: |
| Top 10                |  31.476 |   32.987 |  51.010 |        1.621× |
| Top 1000              |  46.536 |   48.125 |  73.635 |        1.582× |
| Top 100 + exact count |  35.675 |   36.412 | 166.251 |        4.660× |
| Exact count           |  25.236 |   25.173 | 142.363 |        5.641× |

### ARM supplemental standalone terms

| Operation             | compact | identity |     rgb | RGB / compact |
| --------------------- | ------: | -------: | ------: | ------------: |
| Top 10                |  18.895 |   20.537 |  17.385 |        0.920× |
| Top 1000              |  37.677 |   39.297 |  36.720 |        0.975× |
| Top 100 + exact count |  27.715 |   27.156 | 125.066 |        4.513× |
| Exact count           |  11.559 |   11.496 |  11.460 |        0.991× |

### Full-corpus query families

Ratios are RGB / compact; below one is faster. Full-corpus counted latency is unmeasured; the incomplete counted run was excluded.

| Family             |  Top 10 | Top 1000 |
| ------------------ | ------: | -------: |
| intersection       | 36.044× |  31.814× |
| phrase             |  2.336× |   1.918× |
| union              |  0.906× |   0.959× |
| negated            | 25.608× |  22.886× |
| intersection_union | 47.369× |  21.966× |
| term               |  0.058× |   0.263× |

The single official term query, `the`, improves from 5.00 ms to 0.290 ms at top-10 (17.2×). The 301 ranked union queries improve by 9.4%, while the 300 intersections become 36× slower. This localizes the regression: RGB helps physical-order execution, but logical-order composition discards that benefit.

### ARM query families

Ratios are RGB / compact; below one is faster.

| Family             | Top 10 | Top 1000 | Top 100 + count |   Count |
| ------------------ | -----: | -------: | --------------: | ------: |
| intersection       | 3.756× |   3.610× |          4.324× |  4.356× |
| phrase             | 1.102× |   1.077× |          1.174× |  1.175× |
| union              | 0.827× |   0.868× |         19.937× | 45.382× |
| negated            | 2.505× |   2.433× |          2.613× |  2.844× |
| intersection_union | 6.914× |   4.300× |          5.517× |  1.116× |
| term               | 0.361× |   1.009× |         60.364× |  0.957× |

The official `term` family is the single stop-word query `the`; it is not the 714-query supplemental workload. Ranked unions improve, while mapped complete collection dominates the regressions.

## Footprint and construction cost

### ARM index bytes

| Index    |   MiB |
| -------- | ----: |
| compact  | 96.27 |
| identity | 97.22 |
| rgb      | 99.29 |

These are logical file sizes; clones may share physical storage.

### Full-corpus search residency

Fresh processes run one complete pass each of top-10 and top-1000, sequentially. These snapshots describe ranked-search residency; counted-search residency is unmeasured. This is an observed residency sample, not a long-run plateau claim. RSS is the maximum observed snapshot, anonymous is the final snapshot. No latency claims use these memory runs.

| Index    | Index MiB | Max RSS MiB | Final anonymous MiB |
| -------- | --------: | ----------: | ------------------: |
| compact  |   4550.15 |      759.81 |                5.76 |
| identity |   4598.14 |     1115.71 |                5.75 |
| rgb      |   4481.59 |     1019.41 |                5.80 |
| tantivy  |   2890.67 |      574.46 |                0.73 |

Final ranked-pass RSS attribution from `/proc/PID/smaps`:

| Resident MiB | Compact | Identity |    RGB |
| ------------ | ------: | -------: | -----: |
| .post        |  270.75 |   393.25 | 268.56 |
| .pos         |  423.44 |   590.44 | 637.94 |
| .terms       |   31.45 |    50.26 |  31.07 |
| .chunks      |    9.60 |    57.59 |  57.03 |

RGB adds 214.50 MiB of resident position pages and 47.43 MiB of resident map/length pages versus compact, accounting for nearly all of its 259.61 MiB RSS increase. Anonymous residency changes by only 0.05 MiB. This attributes the increase to mapped payload residency rather than a large query heap. The distinct formats and traversal paths are not isolated here, so the position-page increase cannot be attributed to permutation alone.

| Component MiB | Compact | Identity |     RGB |
| ------------- | ------: | -------: | ------: |
| .post         | 2005.28 |  2005.28 | 1904.10 |
| .pos          | 2440.27 |  2440.27 | 2425.66 |
| .terms        |   53.00 |    53.00 |   52.26 |
| .chunks       |    9.60 |    57.59 |   57.59 |
| .store        |   14.83 |    14.83 |   14.83 |
| .fast         |   19.20 |    19.20 |   19.20 |
| .rowstats     |    7.97 |     7.97 |    7.97 |

### Full-corpus construction resources

| Stage   | Wall time | Peak RSS MiB |
| ------- | --------- | -----------: |
| index   | 12:31.26  |     15615.42 |
| reorder | 8:39.43   |      8964.38 |

The configured indexing memory budget is not a demonstrated RSS cap: a mid-build sample had 10,999,876 KiB anonymous residency. This includes allocator-retained memory and does not measure live allocations. Writer memory accounting remains a separate review finding. Standalone BP configuration and convergence/budget status are preserved in `reorder.log`; construction and reorder use process-wide background workers and are not single-CPU latency runs.

## Why RGB regresses here

`required_text::mapped_documents` first exhausts the physical scorer into a
bitmap of stable document IDs. `DocumentMappedScorer::position` then traverses
those original IDs, looks each one up in the physical map, and probes with
`PostingIterator::seek_physical`. Backward probes search the block directory
again and reload decoded blocks. The identity map bypasses this wrapper.

That restores logical ordering too early and destroys locality for complete
collection and affected phrase/Boolean paths. RGB still improves ranked pruning;
this benefit is overwhelmed by repeated decoding elsewhere. The following are
warm ARM work counters over the same 962 official queries, collected outside
latency runs by a separate diagnostics build of the selected source.

| Command               | Compact doc blocks | Identity doc blocks | RGB doc blocks | RGB / compact |
| --------------------- | -----------------: | ------------------: | -------------: | ------------: |
| Top 10                |             42,105 |              45,979 |        339,303 |         8.06× |
| Top 1000              |             67,392 |              71,200 |        363,426 |         5.39× |
| Top 100 + exact count |             68,932 |              72,555 |     15,011,241 |       217.77× |
| Exact count           |             68,583 |              70,599 |     14,851,059 |       216.54× |

Exact-count compressed gap bytes processed rise from 9,460,850 to 2,640,608,494;
these include repeated decoding of resident bytes, not physical disk I/O.
Instrumented top-10 scoring units fall from 835,690 to 566,549, while executor windows
skipped rise from 824 to 4,533. The partitioner reports a converged nonidentity
permutation with no graph truncation on ARM. Poor partitioning is not enough to
explain the regression: the read path demonstrably multiplies block work.

The complete unordered-map disjunction also enters `RequiredTextScorer`, whose
`position` probes terms and computes BM25 even when the caller only counts.
`complete_text_scorer` does not pass its `skip_scoring_setup` intent to this
fallback. The generic scoring counters do not cover those direct calls, so a
zero counter must not be interpreted as proof of zero scoring work on this path.
On the ARM COUNT pass, decoded term-frequency values rise from 834,484 to
662,866,934 (6,725 → 5,194,781 frequency blocks). Normalization-table builds stay
at zero. Phrase queries legitimately need some frequencies; the fallback adds
massive repeated frequency work beyond that baseline.

The rewrite also changes the representation. The existing standalone writer retains ratio/impact
bound kinds but emits its older postings/positions representation. Its text
rewrite does not use the compact encoder even when `--compact-text` is passed.
Thus this is an end-to-end measurement of today's RGB feature; RGB versus the
identity control includes that representation change, not permutation alone.
On the full corpus, clustering nevertheless reduces total files from 4,550.15 MiB
(compact) and 4,598.14 MiB (identity) to 4,481.59 MiB (RGB). RGB postings shrink
2005.28 → 1904.10 MiB. The measured slowdown is not explained by larger files.

The required fix is to keep compatible same-field intersections, phrase checks
and exact counting in physical order, using existing query owners and bounded
scratch. Translate filters/results at the boundary; preserve stable-ID heap ties,
raw score bits, token positions and independent field permutations. Cross-field
composition requires deliberate mapping. This is a proposal, not an implemented
change in these measurements. Compact encodings should also survive rewriting.

## Lucene comparison

The public site snapshot fetched during this run contains Lucene 10.3.0,
Lucene 10.3.0-bp and Tantivy 0.25. Over all 962 queries, geometric means of
per-query median top-10 latency give 1.33× speedup for Lucene-bp over Lucene and
1.63× over Tantivy. A 3× figure depends on the selected subset or statistic.
These are ratios calculated from the [published raw data](https://tantivy-search.github.io/bench/results.json),
not a same-machine comparison to Summa. The snapshot and arithmetic are kept
in this benchmark's evidence.

Lucene reassigns internal document IDs and writes the reordered index, allowing
ordinary query execution to retain the new locality. Its
[reorder API](https://lucene.apache.org/core/10_4_0/misc/org/apache/lucene/misc/index/BPIndexReorderer.html)
and the implementer's [benchmark analysis](https://jpountz.github.io/2025/05/12/analysis-of-Search-Benchmark-the-Game.html)
describe the mechanism. Summa keeps stable IDs through field-local maps; that
contract does not require replaying every match in original order internally.

## Validation and limitations

The benchmark adapter exposes `index --reorder-text` and `reorder PATH` through
existing schema/writer APIs. The search core matches the frozen selected source;
only the adapter differs. Native harness, formatting, Clippy and native-without-
sync compilation pass. WASM is not rebuilt, following the standing instruction.
The earlier standalone implementation's merge-time and budget/migration audit
limitations remain documented in [the design](maxscore-text-reordering.md).

Full-corpus counted repeats were first reduced to three before any counted result
completed because each RGB pass took many minutes. The counted run was subsequently
stopped during its second timed RGB pass after the ranked results and ARM counters
established the regression. No completed full-corpus counted command is reported;
its partial samples were discarded. The completed seven-pass ranked samples were
retained. All settings, interrupted-run logs and scope changes are preserved.
The separate memory comparison uses one full ranked pass per command for every
engine. Full-corpus count correctness is covered by all 1,676 exact references,
but full-corpus counted latency and residency remain unmeasured.

An initial construction used a 16 KiB dictionary target at reorder time. Its
measurements were excluded, and RGB was rebuilt from the identity control with
the matched 4 KiB setting before the reported runs. Both attempts' logs and
hashes remain identifiable in the evidence. Redundant verification of the rejected
full-corpus fixture was interrupted after its writer finished; the corrected
fixture passed all 1,676 exact references. A non-LTO adapter build was stopped
before constructing indexes and replaced with the matched LTO build.

## Evidence

[Reproduction instructions and verified raw evidence](benchmark-results/rgb-2026-09-16/README.md) include per-query samples, correctness checks, work counters, source provenance and memory measurements.
