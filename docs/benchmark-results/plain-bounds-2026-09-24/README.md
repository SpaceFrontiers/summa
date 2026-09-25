# Plain-index block-bound admission

September 24, 2026. Paired measurements against Summa 1.9.1, using unchanged
10-million-document ordinary and RGB indexes. This is a query execution change;
no schema, storage format, indexing setting, or rebuild is required.

## Change and correctness contract

`TermQuery` previously required optional ratio bounds before admitting an
ordinary ranked term to MaxScore. It now also uses the conservative maximum-TF
and minimum-length metadata already present in plain postings. The existing
windowed executor handles lists without ratio bounds. Positioned and complete
membership requests keep their cursor paths.

The typed ranked two-term conjunction likewise accepts conservative plain
bounds, retaining finite-score and supported-parameter checks. Exact counted
traversal remains exhaustive. Every skip requires a bound strictly below the
proven heap floor; equal-score document ties remain eligible. Both changes live
in core query owners and share native/async execution. There is no new scorer,
per-query corpus cache, or persisted metadata. A plain term can now initialize
the executor’s existing roughly 49 KiB single-term window scratch per worker;
that bounded scratch is reused and shared with other windowed text queries.

## Protocol

The workload contains 334 queries: 141 single terms, 147 conjunctions, and 46
medium-frequency phrases as controls. Each layout/version runs COUNT, top-10,
and top-100 with 32 clients. Each cell has three five-second repetitions in
each of two rounds. Version order is baseline/candidate/candidate/baseline;
layout order reverses for the second round. Each session warms for 20 seconds,
with the upstream driver's additional connection warmup. These are warm
throughput measurements, not cold-storage or tail-latency claims.

The same Cascade Lake machine supplies 30 logical CPUs to the server and two
to the upstream Searchbench driver. Summa uses 30 search/blocking workers and
four HTTP workers, with 64-request admission. Rust 1.98.1 release builds use
`-C target-cpu=native`; query caches remain off. Network isolation, affinity,
request adapters, immutable indexes, and timing driver are unchanged. Source
builds run on a separate machine. CPU/request and anonymous/total RSS accompany
QPS; mmap residency is not equivalent to heap allocation.

Cross-engine analysis differences are retained: these are before/after Summa
measurements of the exact same input and results, not a claim of equivalent
Lucene analysis. The earlier [expanded comparison](../skipping-2026-09-24/README.md)
contains the Luxir, Elasticsearch and OpenSearch reference measurements.

A proposed OR control exposed an existing audit utility limitation before
measurement: requesting a scorer with limit zero does not guarantee exhaustive
OR enumeration. That control was removed from this campaign; the affected term
and conjunction families and the phrase control retain exhaustive audits. The
failed pre-timing attempt is preserved in the raw evidence.

## Evidence

The companion summary validates every completed timing cell, count, ranked
response, exhaustive ID/score-bit audit, and memory sample before aggregating.
The ARM supplement uses the same release compiler and a 32,768-document RAM
fixture through the public Searcher. Winners occur early, late, or scattered;
fixture construction and exhaustive correctness checks are outside timing.
Its process peak RSS includes index construction and is not query scratch.

The completed campaign contains **168 timing cells / 504 repetitions**, with
zero request or sampling errors. All 1,336 exhaustive query audits agree on
counts and ranked ID/score bits across versions. All 8,016 HTTP checks agree
with the corresponding oracle and before/after response objects are identical.
The [summary script](summarize.py) enforces these gates before producing
[all results](results.md), [numeric evidence](results.json), and
[audit totals](audits.json).

The downloaded raw archive was SHA-256 verified as
`fc680f2958ea198f670eb4e820a5940169607e2ae5831418d263a0f8d96838fc`.
Raw evidence, command logs and provenance are retained in
`.context/luxir-gap-20260924/evidence`; rerun the report with:

```sh
python3 docs/benchmark-results/plain-bounds-2026-09-24/summarize.py \
  .context/luxir-gap-20260924/evidence
```

The [measured implementation patch](implementation.patch) precedes the final
readability cleanup into `can_rank_term` and `supports_text_block_pruning`.
Those helpers preserve the admission predicates and pass the regression suite,
but their final source spelling was not separately timed on the corpus.

## Full-corpus findings

QPS, higher is better; the same query families and index bytes are used before
and after. These rows summarize the most substantial gains; the complete table
also includes every regression and COUNT/phrase control.

| Layout / family                 |   k | Before |  After | Speedup |
| ------------------------------- | --: | -----: | -----: | ------: |
| Plain / high-frequency term     |  10 |    844 |  5,681 |   6.73× |
| Plain / high-frequency term     | 100 |    842 |  3,183 |   3.78× |
| Plain / medium-frequency term   |  10 |  2,569 | 21,176 |   8.24× |
| Plain / medium-frequency term   | 100 |  2,540 | 10,760 |   4.24× |
| Plain / low-frequency term      |  10 |  8,380 | 35,076 |   4.19× |
| Plain / low-frequency term      | 100 |  8,047 | 21,742 |   2.70× |
| RGB / high + high conjunction   |  10 |  1,230 |  2,391 |   1.94× |
| RGB / high + high conjunction   | 100 |  1,220 |  1,609 |   1.32× |
| RGB / high + medium conjunction |  10 |  3,696 |  4,651 |   1.26× |

Plain ranked-term CPU/request falls 64.3–89.1%. RGB high/high conjunction
CPU/request falls 49.3% at k=10 and 24.5% at k=100. Both rounds show these gains.
Peak anonymous RSS rises from 123.6 to 124.8 MiB on plain and from 130.5 to
130.7 MiB on RGB; total resident peaks are approximately 2,016 and 2,274 MiB.
The small memory differences do not establish a change in structural residency.

Conjunction admission has a measurable cost where bounds save little work:
plain high/low regresses 3.4% at k=10 and 2.1% at k=100; plain high/medium
regresses 5.1% at k=100. RGB high/low regresses 2.4% / 1.9%. These regressions
repeat in both rounds and CPU/request increases consistently. RGB terms, whose
planner admission is unchanged, lose 0.6–1.8% throughput with CPU/request
approximately unchanged. COUNT and medium-phrase controls remain within 1.5%.
The results justify retaining the broad term improvement and expose the need
to avoid unproductive conjunction bound checks; they do not show a universal win.

Luxir was not rerun in this paired campaign. The earlier expanded comparison
still shows large term and high/low conjunction gaps: for context, its plain
high-term/top-10 reference is 65,013 QPS versus the candidate's 5,681, and
high/low conjunction/top-10 is 53,707 versus 3,049. These separate-run numbers
are direction-finding context, not a fresh paired engine comparison. Query
analysis and hit-count differences remain visible in the earlier report.

Next targets are the per-open L1 metadata copies and the cost of seeking a
common posting list from rare candidates. Profile those separately from scoring
and bound checks before adding an admission heuristic. OR/pattern expansion
and the 142 resource-limited expressions are outside this change.

## ARM supplement

Median of the two run medians, microseconds per query (lower is better).
These synthetic cases test useful and unproductive pruning separately.

| Winner placement / query / k | Before |  After | Speedup |
| ---------------------------- | -----: | -----: | ------: |
| early / and / 10             | 259.64 |   9.18 |  28.27× |
| early / and / 100            | 261.19 |  11.74 |  22.26× |
| early / term / 10            | 148.20 |   9.17 |  16.17× |
| early / term / 100           | 149.16 |  12.18 |  12.25× |
| late / and / 10              | 297.40 | 307.22 |   0.97× |
| late / and / 100             | 310.71 | 330.21 |   0.94× |
| late / term / 10             | 197.07 | 191.04 |   1.03× |
| late / term / 100            | 169.54 | 194.73 |   0.87× |
| mixed / and / 10             | 266.92 | 151.46 |   1.76× |
| mixed / and / 100            | 311.84 | 275.20 |   1.13× |
| mixed / term / 10            | 154.72 |  70.99 |   2.18× |
| mixed / term / 100           | 176.40 | 163.82 |   1.08× |

Early and scattered winners benefit. Late-winner top-100 term latency increases
14.9%, and conjunction latency increases 6.3%; metadata checks can cost more
than they save when the heap floor rises only at the end. Late term top-10
changes direction between runs and is inconclusive. This change does not make
every query faster. Peak process RSS, including fixture construction, is
104.6 MiB before and 103.9 MiB after; that difference is not a structural memory
saving. See [all samples and estimates](arm.json).
