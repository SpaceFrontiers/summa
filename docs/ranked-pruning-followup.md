# Ranked conjunction and phrase pruning follow-up

September 24, 2026. This report covers the retained query changes and their
same-index measurements. [Score-guided traversal](score-guided-traversal.md)
separately evaluates research and proposes further experiments; a new persisted
range hierarchy has not been implemented.

## Problem and retained implementation

The preceding comparison left ordinary conjunctions and medium-frequency
phrases behind Luxir. RGB's low-frequency-phrase top-10 result was particularly
poor. Reordering improved locality but could place the best matches late in the
physical document order, leaving the top-k threshold too low to prune early work.

Public search enters `Searcher`, then the segment collector. Required text uses
the typed conjunction executor; ranked phrases use `PhraseScorer` and the
collector's candidate-bound protocol. The changes stay in these existing owners:

- Loaded full posting blocks use the existing fixed-size lower-bound search.
  Partial blocks retain the existing SIMD implementation.
- Conjunctions seek from the rare term only when both the global frequency ratio
  and the current decoded document-span ratio exceed the posting block size, 128. Dense local RGB runs keep SIMD intersection. Existing TF batches,
  predicates, score reduction and cancellation remain shared.
- Exact phrases can use the smallest aligned term frequency as an upper bound
  when the original first position stream certifies unique starts. That same
  certificate allows a singleton position probe from either term. Repeated
  original-first positions and nonzero slop keep their conservative behavior.
- Native two-term phrase setup loads its two readers serially, avoiding nested
  Rayon jobs. Larger phrases retain parallel setup; async setup is unchanged.
- After 1,024 competitive confirmations, a mapped ranked phrase can make one
  bounded pilot traversal to prove a stronger top-k score floor. The pilot
  confirms real phrase matches, restores the original traversal, and emits no
  results. Normal traversal still collects winners and resolves stable-ID ties.

The pilot supports k=1–128. It examines at most 4,096 future block metadata
entries, using low minimum document length to nominate blocks. For k≤16 it
retains at most 64 blocks, aligns at most 8,192 postings, and confirms at most
512 selected candidates. For k=17–128 it retains eight blocks and confirms at
most 1,024 candidates, making it easier to fill a larger proof heap. It returns
no floor unless its independent heap contains k distinct real matches.

The heuristic only selects work: it cannot authorize skipping. Conservative
bounds and a floor proved by actual matches authorize every skip. Equality with
the pilot floor remains competitive until the real heap resolves logical IDs.
Counts and nested membership remain exhaustive. Eligibility wrappers do not
inherit the pilot, because their filters could invalidate a child's proof.
Native, async-only and WASM share the phrase and collector implementation.

No posting, position, document-map, schema, or RPC format changes. Existing
indexes use the changes immediately; no reindex is required. No indexing,
codec, worker-count or public configuration default changes.

## Why RGB's outlier was slow

For `"10px 0"`, baseline RGB top-10 confirmed 52,460 candidates and scored
51,974 matches; the ordinary index confirmed 11,779 and scored 11,445. RGB
actually decoded fewer document blocks: 1,964 versus 3,049. Its poor result
therefore cannot be explained simply by worse block compression.

A temporary diagnostic trace showed RGB's heap floor stuck at 8.128809 from
1,024 through 32,768 confirmations. The actual tenth-best score was 10.1283188.
The initial pilot of eight highest-bound blocks produced only 7.8994, below the
floor already held. Broader coverage of short-document blocks found better real
matches. This is evidence that discovering a useful threshold early matters in
addition to packing related documents together. The temporary trace is absent
from the retained code.

## Final cloud results

Warm throughput, queries/second; higher is better. Percentages compare each
layout with its own baseline. Luxir is a fresh same-host reference, not a
before/after code comparison.

| Workload                          | Summa before → after | Change | RGB before → after |  Change |  Luxir |
| --------------------------------- | -------------------: | -----: | -----------------: | ------: | -----: |
| Conjunction / top 10              |      46,512 → 47,939 |  +3.1% |    74,980 → 75,533 |   +0.7% | 53,361 |
| Conjunction / top 100             |      41,476 → 42,995 |  +3.7% |    62,783 → 62,932 |   +0.2% | 42,653 |
| Conjunction / count               |      77,668 → 75,576 |  -2.7% |    93,540 → 93,064 |   -0.5% | 66,758 |
| Low-frequency phrase / top 10     |      16,213 → 16,982 |  +4.7% |     8,706 → 22,205 | +155.0% |  7,615 |
| Low-frequency phrase / top 100    |        5,101 → 5,147 |  +0.9% |      5,784 → 6,012 |   +3.9% |  3,869 |
| Low-frequency phrase / count      |        1,972 → 1,985 |  +0.6% |      2,915 → 2,898 |   -0.6% |  2,072 |
| Medium-frequency phrase / top 10  |      15,704 → 16,687 |  +6.3% |    17,244 → 18,880 |   +9.5% | 17,641 |
| Medium-frequency phrase / top 100 |      10,540 → 10,988 |  +4.3% |     8,112 → 11,685 |  +44.0% |  5,977 |
| Medium-frequency phrase / count   |            718 → 702 |  -2.2% |          733 → 715 |   -2.4% |    608 |

CPU microseconds/request, including server overhead; lower is better.

| Workload                          | Summa before → after | RGB before → after |
| --------------------------------- | -------------------: | -----------------: |
| Conjunction / top 10              |        542.7 → 529.6 |      346.0 → 343.9 |
| Conjunction / top 100             |        606.1 → 590.4 |      405.3 → 403.9 |
| Conjunction / count               |        316.9 → 318.1 |      183.9 → 184.7 |
| Low-frequency phrase / top 10     |      1630.7 → 1574.5 |    3164.7 → 1190.7 |
| Low-frequency phrase / top 100    |      5346.2 → 5388.1 |    4731.9 → 4650.3 |
| Low-frequency phrase / count      |    14821.1 → 14703.6 |    9918.4 → 9981.5 |
| Medium-frequency phrase / top 10  |      1638.1 → 1559.6 |    1495.2 → 1363.7 |
| Medium-frequency phrase / top 100 |      2467.0 → 2396.4 |    3222.7 → 2236.6 |
| Medium-frequency phrase / count   |    41201.6 → 42325.1 |  40427.3 → 41367.1 |

Peak resident memory across the sessions below includes warm mmap payloads.
Anonymous RSS is reported separately; neither measure is Rust `Pin`. The
pilot adds bounded query scratch, not a corpus-sized cache.

| Layout / version | Peak RSS MiB | Peak anonymous RSS MiB |
| ---------------- | -----------: | ---------------------: |
| plain / before   |       1150.5 |                  108.9 |
| plain / hybrid   |       1148.9 |                  106.9 |
| rgb / before     |       1375.2 |                   99.6 |
| rgb / hybrid     |       1373.6 |                   97.7 |

The strong retained gain is RGB phrase ranking. Ordinary conjunction top-10
still trails fresh Luxir. Count and small-control regressions remain visible
above; the change does not establish universal parity or a universal speedup.

For the RGB `"10px 0"` outlier, uninstrumented median core search changes from 8.271 to 1.696 ms (4.88×). The separately instrumented work audit shows:

| Work per query              | Ordinary before | Ordinary after | RGB before | RGB after |
| --------------------------- | --------------: | -------------: | ---------: | --------: |
| doc_blocks                  |           3,049 |          3,049 |      1,964 |     1,828 |
| phrase_confirmations        |          11,779 |         11,652 |     52,460 |     5,011 |
| phrase_score_units          |          11,445 |         11,429 |     51,974 |     4,564 |
| position_reads              |          23,558 |         23,304 |    104,920 |    10,022 |
| position_blocks             |           3,831 |          3,730 |      3,520 |     1,483 |
| position_values             |         486,854 |        474,504 |    450,560 |   189,776 |
| posting_seeks               |          14,232 |         14,232 |     55,432 |    16,854 |
| phrase_seed_metadata_blocks |               0 |              0 |          0 |       576 |
| phrase_seed_candidates      |               0 |              0 |          0 |       512 |

All final session response gates and exhaustive audit comparisons pass. Every
timed repetition reports zero request errors; memory samplers report no errors.
CPU, utilization, per-session medians and full repetition ranges are retained
in the machine-readable summary.

## Measurement protocol and limits

The cloud fixture is the existing 10-million-document Wikipedia May 2012 chunk
corpus. Before and after read the same immutable ordinary index, and separately
the same immutable RGB index. Historically those indexes used six and thirty
indexing workers respectively. Their cross-layout difference is **not** a pure
reordering ablation.

The benchmark host is an n2-highmem-32: 32 logical CPUs / 16 physical Cascade Lake
cores. Server affinity is CPUs 0–14,16–30; the replay driver uses 15,31. Summa
uses 30 search workers, four HTTP workers and 32 clients, with query caching off
and warm filesystem pages. Both versions use the same compiler and locked
release build with `RUSTFLAGS="-C target-cpu=native"`.

Two rounds reverse the four server sessions: baseline ordinary, candidate
ordinary, candidate RGB, baseline RGB; then the reverse order. Each cell has
three eight-second repetitions per session after session warmup. Aggregate
values are medians of all six repetitions, with session medians and ranges
retained. A separate fresh Luxir comparison uses three ten-second repetitions.
Diagnostic counters come from separately instrumented binaries and are not used
as throughput measurements.

This is a targeted investigation of 15 accepted queries: seven high/low-frequency
conjunctions, seven low-frequency phrases, and one medium-frequency phrase,
covering top-10, top-100 and exact counts. It is not a rerun of all 826 queries.
Elasticsearch and OpenSearch were not rerun. The HTTP benchmark adapter avoids
production RPC and general hydration costs; its QPS is not a production SLA or
a tail-latency guarantee.

## ARM control fixture

A shared Apple M4 ran a 32,768-document fixture with one indexing worker for
both ordinary and reordered layouts. Two before/after rounds used identical
source fixture, flags and query signatures. Each cell has 30 warmups and 40
samples of ten searches per process; the table averages the two process medians.
Values are microseconds, with top-10 / top-100 shown together.

| Layout / query                |       Before µs |        After µs | Latency change |
| ----------------------------- | --------------: | --------------: | -------------: |
| Ordinary moderate conjunction |   56.53 / 65.20 |   56.58 / 65.98 |  +0.1% / +1.2% |
| Ordinary rare conjunction     |   17.99 / 19.62 |   16.92 / 18.31 |  −6.0% / −6.7% |
| Ordinary dense conjunction    |  99.01 / 101.88 | 101.68 / 105.25 |  +2.7% / +3.3% |
| Ordinary phrase               |   31.27 / 95.73 |   28.31 / 86.46 |  −9.5% / −9.7% |
| Ordinary term control         | 147.83 / 150.38 | 146.98 / 149.03 |  −0.6% / −0.9% |
| RGB moderate conjunction      |   43.92 / 55.86 |   45.56 / 58.52 |  +3.7% / +4.8% |
| RGB rare conjunction          |     5.96 / 9.13 |    5.65 / 10.76 | −5.2% / +17.8% |
| RGB dense conjunction         |   95.36 / 99.66 |  92.64 / 101.04 |  −2.9% / +1.4% |
| RGB phrase                    |   31.00 / 75.14 |   28.34 / 63.09 | −8.6% / −16.0% |
| RGB term control              | 128.22 / 143.10 | 121.97 / 138.88 |  −4.9% / −2.9% |

Exact IDs and score bits agree in every before/after cell. This small shared-host
fixture is noisy: the RGB rare top-100 after medians were 9.39 and 12.12 µs,
versus 8.67 and 9.59 before. Its measured regression remains visible; it cannot
be dismissed as a proven non-regression. Process RSS includes fixture creation,
so it does not isolate query scratch. Raw samples and process resource reports
are retained with the results.

## Rejected experiments and review findings

- Early mapped singleton tie filtering did not help and regressed the medium
  phrase screen by about 6%; removed.
- Enabling ordinary conjunction block pruning without the existing metadata
  admission policy regressed the conjunction screen by about 6%; removed.
- A global frequency ratio of four for seek-driven intersection caused an 84%
  ARM moderate-conjunction regression. The retained policy uses 128 and checks
  local spans, preserving SIMD for dense RGB regions.
- Eight highest-bound pilot blocks did not raise the outlier's floor. Eight
  short-document blocks helped top-100 but missed the top-10 outlier. Sampling
  the first sixteen candidates in 64 blocks helped it, but selecting the best
  eight candidate bounds per block worked better. That wider selection hurt a
  top-100 query by about 9%, motivating the retained small-k/larger-k split.
- An ARM trial reused a stale baseline executable after a source archive overlay;
  it is excluded. Subsequent builds touched the source, checked compilation and
  recorded distinct executable hashes. A first diagnostic build enabled only the
  core feature and returned null counters; only server-feature builds contribute
  final counters. Both failures remain in the investigation evidence.

Remaining work: the complete 826-query suite; cold I/O and multi-segment 10M
measurements; independent ARM repetitions; and a same-input/same-worker-policy
RGB ablation. The serial two-reader setup has only warm-query evidence. A
bounded pilot can add work when selected blocks contain no useful matches;
there is no universal guarantee that reordering or sampling is faster.

The next structural experiments are query-dependent range ordering and longer
score-aware skips over existing metadata, described in the
[research assessment](score-guided-traversal.md). Measure metadata work, position
checks, score evaluations, CPU/request and residency before adding a new index
hierarchy or changing defaults.

## Correctness and validation

Cloud audits compare exhaustive counts plus exact top-100 IDs and score bits.
Each timed session also compares all 45 HTTP responses byte-for-byte with its
layout's baseline. Regression tests cover four posting codecs, swapped term
orders, sparse partial batches, predicates, exact counts, repeated positions,
slop, late equal-score winners, underfilled heaps, unsupported k, exhausted tails,
restored position cursors and expired deadlines.

`python3 scripts/check_search.py check` passes all five stages: formatting,
strict search-stack Clippy, 2,049 tests (25 ignored), native-without-sync and
standalone broker compilation. Async-only phrase tests (27), the conjunction
regression, diagnostic-feature Clippy and three diagnostics tests pass. The
WASM release build and all 40 JavaScript tests pass. Subsequent duplicate/slop
and filtered-pilot test extensions pass with native and async-only features;
these and comments are the only code changes after the measured production
snapshot. No lifecycle/RPC code changed, so the `full` RPC harness was not run.

The [retained evidence](benchmark-results/ranked-pruning-2026-09-24/README.md)
contains reproducible analysis, raw measurements, frozen candidate source,
compiler/executable hashes and validation logs. The full cloud archive was
downloaded and SHA256-verified. Infrastructure identifiers and operational
logs are omitted from the public evidence.
