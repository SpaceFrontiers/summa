# Dictionary scans, FST merge feasibility and the term cache — September 27

Broad wildcards (`h*band`, `a*nder`, … — 42 patterns, each walking a whole
first-letter subtree of up to 830k terms) were the largest remaining Luxir
gap. This campaign measures where that time goes on the unchanged
10M-document index and closes it: **broad-wildcard TOP_10 goes from 338 to
4,712 QPS, from 0.18× to 2.58× Luxir**, with CPU per request from 86 ms to
6 ms (Luxir: 16 ms). No index format or query semantics changed; every result
is exact-audited. Throughput below is from separate paired runs, each against
its immediate predecessor (the fourth was on a slower host session for every
engine, so its ratios, not absolute numbers, carry over).

| Step                                                                            |        Broad-wildcard TOP_10 QPS (vs Luxir) | Status                                             |
| ------------------------------------------------------------------------------- | ------------------------------------------: | -------------------------------------------------- |
| Baseline (256-block per-segment term cache)                                     |                                 338 (0.18×) | —                                                  |
| Skip value decoding only                                                        |                        ≈ −1% (native CPU 0) | Rejected                                           |
| FST term dictionary                                                             | enumeration 4.4× slower than decoded blocks | Not adopted ([note](../../fst-term-dictionary.md)) |
| [Process-wide 256 MiB term cache](../../term-dictionary-cache.md) (new default) |              703 (0.38×), +130 MiB anon RSS | **Adopted**                                        |
| Tight key-only block scan                                                       |                                 792 (0.43×) | **Adopted**                                        |
| Word-sized literal checks in the single-star matcher                            |                                 857 (0.47×) | **Adopted**                                        |
| [Dictionary-block suffix filters](../../dictionary-suffix-filters.md)           |         **4,712 (2.58×)**, +13 MiB anon RSS | **Adopted**                                        |

## Where the time goes

A separate single-CPU probe on the production dictionary
([FST probe](fst-probe.json), [source](fst-probe-source.rs.txt)) scans all 42
patterns three ways; every mode returns identical `TermInfo` sequences:

| Sum over 42 patterns                                                 |  CPU ms |
| -------------------------------------------------------------------- | ------: |
| SSTable, 1,024-block / 16 MiB cache                                  | 1,319.9 |
| SSTable, all touched blocks retained (12,503 blocks, 207 MB decoded) |   850.1 |
| FST range stream                                                     | 3,753.3 |

Blocks whose subtree fits the cache already run at warm speed; larger subtrees
(c, d, f, i, p) re-decompress and cost 2.3×. The server's default term cache is
256 blocks (~4 MiB), so every broad wildcard decompresses its whole subtree.
The FST probe also answers the merge question: an FST dictionary merges 1.49×
faster with bounded memory (+23 vs +870 MiB), but its subtree enumeration is
4.4× slower than a decoded SSTable block, so it cannot close this gap.

## Rejected: skip value decoding

Decoding only keys and skipping `TermInfo` bytes for rejected entries (plus a
restart seek in the first block) passes the 677-query exact audit but does not
help on the real corpus. CPU-0 A–B–B–A diagnostics (20 warmups, 100 samples):
broad wildcards +0.7–1.0%, prefixes +3.3–6.9%, point-lookup terms +1–4%
slower; A2/A1 drift is mostly within ±0.5%. Inline `TermInfo` values are only
a few bytes; key reconstruction and zstd dominate.
[Diagnostics](value-skip-diagnostics.json), [patch](value-skip.patch).

## The term cache: paired HTTP comparison

One binary with an experiment-only environment override of
`IndexConfig.term_cache_blocks`/`term_cache_budget_bytes`
([patch](term-cache.patch)) runs A1/C1/L1/C2/A2/L2 on the 32-vCPU host:
default cache (A), 16,384 blocks capped at 256 MiB (C), and pinned Luxir 0.1.0
(L). Protocol as in the [September 26 campaign](../dictionary-decoding-2026-09-26/README.md):
30 server CPUs, 2 driver CPUs, 32 clients, 40 s warmup, 3×3 s per cell, result
cache off. Both settings pass the 677-query exact audit; every Summa phase
passes 2,031 HTTP checks, and each phase's process environment and server log
prove which setting ran. **30,222,907 requests, zero errors.**
[Results](results.json).

| Family        | Op      | Default QPS | Cache QPS | Luxir QPS | Cache / default | CPU µs/request default → cache |
| ------------- | ------- | ----------: | --------: | --------: | --------------: | -----------------------------: |
| wildcard_scan | TOP_10  |         339 |       708 |     1,846 |       **2.09×** |                86,290 → 41,504 |
| wildcard_scan | TOP_100 |         339 |       707 |     1,836 |           2.09× |                86,318 → 41,491 |
| wildcard_scan | COUNT   |         327 |       654 |     1,768 |           2.00× |                89,220 → 45,353 |
| regex         | COUNT   |       4,730 |     4,815 |    13,727 |           1.02× |                  5,886 → 5,785 |
| and_high_med  | TOP_10  |       4,374 |     4,409 |     9,053 |           1.01× |                  6,385 → 6,379 |
| or_high_low   | TOP_10  |      20,384 |    20,080 |    39,387 |           0.99× |                  1,262 → 1,267 |
| low_term      | TOP_10  |      82,483 |    82,553 |   123,607 |           1.00× |                      302 → 302 |

Other cells stay within ±1.8%. End-of-phase anonymous RSS rises from
246–249 MiB to 375 MiB (+129 MiB); Luxir's is 57 MiB. The cache is per
segment, so a default of this size would scale with segment count — this is
why no default changed here. A process-wide dictionary budget, or dictionary
blocks that can be scanned in place from the mapping, would deliver the same
effect without per-segment anonymous memory.

## Tight key-only scan

The warm-scan profile on Apple M-series showed ~42% of samples in `memmove`
(3–8-byte key suffixes and inline `TermInfo` data), ~16% in `memcmp` (prefix
checks) and the rest in decoding. The candidate:

- decodes keys only and skips values of rejected entries (`SSTableValue::skip`,
  a slice parse for `TermInfo` that accepts/rejects exactly what `deserialize`
  does — pinned by a test over truncations and corrupt tags);
- tracks prefix membership via each entry's shared-prefix length, so in-range
  entries need no comparison;
- copies short suffixes with one fixed-width move instead of a `memmove` call;
- replaces the duplicated async/sync scan loops with one `PrefixScan`.

Locally, warm scans of a 5.3M-term synthetic dictionary drop from 43.9 to
19.0 ms (A–B–B–A, single-thread CPU time). An inlined literal compare for the
matcher was 5% slower than `memcmp` and was reverted.

The four-variant HTTP run ({base, tight} × {default, large cache}, mirrored
A1 B1 C1 D1 L1 D2 C2 B2 A2 L2) contains **48,416,112 requests, zero errors**;
the candidate passes the exact audit under both cache settings.
[Results](tight-results.json).

| Family        | Op      |   Base |          Tight | Base + cache |    Tight + cache | Luxir |
| ------------- | ------- | -----: | -------------: | -----------: | ---------------: | ----: |
| wildcard_scan | TOP_10  |    340 |    358 (+5.1%) |          706 | **792 (+12.1%)** | 1,846 |
| wildcard_scan | COUNT   |    328 |    346 (+5.6%) |          653 |     738 (+12.9%) | 1,771 |
| and_high_med  | TOP_10  |  4,395 |  4,270 (−2.8%) |        4,368 |    4,261 (−2.4%) | 9,098 |
| and_high_med  | TOP_100 |  3,441 |  3,352 (−2.6%) |        3,445 |    3,344 (−2.9%) | 6,248 |
| or_high_low   | COUNT   | 47,366 | 46,385 (−2.1%) |       47,461 |   45,939 (−3.2%) | 9,384 |

The conjunction regression repeats in all four candidate phases and matches the
rejected September 26 surrounding-entry inlining pattern, although conjunctions
perform only a few dictionary point lookups. It is treated as real.

### Why conjunctions move: code placement, not the scan

Two variants were screened on CPU 0 of the 8-vCPU host with the native
diagnostics (18 queries × 2 limits, 20 warmups, 100 samples, mirrored order;
all pass the exact audit). [Rows](layout-screen.json). Time change against the
baseline binary, class means (+ is slower):

| Build                                          | Broad wildcards | Conjunctions | Disjunctions | Single terms | Prefixes |
| ---------------------------------------------- | --------------: | -----------: | -----------: | -----------: | -------: |
| Tight                                          |           −7.3% |        +2.5% |        +0.9% |        +1.5% |    +1.6% |
| V2: tight scan, original point-lookup decoder  |           −8.0% |        +3.5% |        +2.5% |        +2.5% |    +1.6% |
| V3: `#[inline]` instead of `#[inline(always)]` |           −7.6% |        +2.0% |        +0.9% |        +2.2% |    +1.6% |
| Tight vs baseline, both `codegen-units=1`      |           −7.8% |    **+0.4%** |        −0.9% |        +0.4% |    +1.5% |

V2's point-lookup code is source-identical to the baseline yet regresses more,
so the slowdown is not in dictionary lookups. With a single codegen unit the
conjunction, disjunction and single-term shifts vanish; they come from how the
default 16-unit release build partitions and inlines unrelated code after
`sstable.rs` grows. The small prefix cost (+1.5%, a few µs on short range
scans) is real. `codegen-units=1` is not a free default either: against the
default build it makes prefixes 12.9% and conjunctions 2.7% slower and single
terms 3.9% faster.

## Process-wide term cache: the new default

The per-segment result above cannot be a default: it multiplies by segment
count and reader generations. The document store's process-wide cache was
generalized into `SharedBlockCache<V>`, and search-time dictionaries now use it
under `IndexConfig.term_cache_process_bytes` (256 MiB on 64-bit native builds;
`summa-server --term-cache-budget-mb`). Merges keep private caches.
[Design](../../term-dictionary-cache.md).

The paired run compares the previous binary with old defaults (O), the new
binary pinned to per-segment caches with an experiment-only override (P,
[patch](shared-policy.patch)), and the new binary with its defaults (N), in
mirrored order O1 P1 N1 L1 N2 P2 O2 L2. Both new-binary settings pass the
677-query exact audit, and each phase's environment and server log prove the
policy. **38,605,128 requests, zero errors.** [Results](shared-results.json).

| Family        | Op      |  O QPS |  P QPS | **N QPS** | Luxir QPS |     N / O |
| ------------- | ------- | -----: | -----: | --------: | --------: | --------: |
| wildcard_scan | TOP_10  |    338 |    337 |   **703** |     1,835 | **2.08×** |
| wildcard_scan | TOP_100 |    338 |    339 |       703 |     1,832 |     2.08× |
| wildcard_scan | COUNT   |    326 |    326 |       653 |     1,754 |     2.00× |
| and_high_med  | TOP_10  |  4,342 |  4,221 |     4,244 |     8,996 |    0.977× |
| and_high_med  | TOP_100 |  3,379 |  3,318 |     3,315 |     6,217 |    0.981× |
| or_high_low   | TOP_10  | 19,951 | 19,547 |    19,144 |    35,671 |    0.960× |
| or_high_low   | TOP_100 | 13,402 | 13,115 |    12,887 |    16,143 |    0.962× |
| regex         | COUNT   |  4,896 |  4,684 |     4,763 |    13,526 |    0.973× |
| low_term      | TOP_10  | 81,029 | 81,831 |    81,968 |   122,080 |    1.012× |

Broad-wildcard CPU falls from 86.1 to 41.8 ms per request. Anonymous RSS is
376 MiB with the new default versus 243–254 MiB for O and P; Luxir's is 55–56
MiB. The conjunction/disjunction/regex-count shifts appear already in P, which
runs the old per-segment policy: the policy itself (N vs P) moves those cells by
−2.1% to +1.7%. They follow the new binary's code placement, like the tight-scan
builds above, whose conjunction shift vanished with one codegen unit.

**aarch64.** A single-binary A–B–B–A on an Apple M4 ([data](arm-term-cache.json),
[source](term-cache-probe-source.rs.txt)) rotates 12 of the broad-wildcard
patterns through `Searcher::search` over one 6M-document segment (67 MB
compressed dictionary): **129.0 → 70.3 ms process CPU per cycle (1.83× less)**,
identical hits, phases within 0.6%.

## Tight scan on top of the shared cache

The tight key-only scan (above) was re-measured on the shared-cache build,
mirrored S1 T1 L1 T2 S2 L2 ([results](tight2-results.json), 30,159,304
requests, zero errors, exact audit). Broad wildcards rise **703 → 792 QPS
(+12.7%)** and CPU falls 41.6 → 37.0 ms per request; regex TOP_100 +3.2%; every
other cell stays within −1.9% (driver-limited low-term COUNT) and +0.9%. The
earlier −2.8% conjunction shift does not recur against this baseline, which
already carries the new code placement. **Adopted.**

## Word-sized literal checks

The profile of that build showed the single-star matcher plus libc
`memcmp`/`memmove` at about a quarter of broad-wildcard CPU: a library call per
scanned term for 1–5-byte literals. `Literal` now compares the prefix with one
masked 8-byte load and the suffix by its last four bytes first (a load that
stays inside the freshly written key suffix), falling back to the slice
comparison only on a four-byte match or for long literals. A byte-loop variant
was 5% slower than `memcmp` and a full 8-byte suffix load stalled on
store-to-load forwarding; both were discarded. An exhaustive test compares the
checks with `starts_with`/`ends_with` across the 8-byte boundary; the existing
regex-equivalence test still passes. Paired T1 M1 L1 M2 T2 L2 mirror
([results](matcher-results.json), 30,221,629 requests, zero errors, exact
audit): broad wildcards **792 → 857 QPS (+8.3%)**, CPU 37.1 → 34.4 ms; other
cells −1.5% to +1.4%. **Adopted.**

## Suffix filters for cached dictionary blocks

With blocks decoded and the scan loop tightened, the remaining cost is
visiting every term. A block-level probe of the production dictionary
([data](suffix-blocks.json), [source](suffix-block-probe-source.rs.txt)) shows only **4,528 of 34,010**
blocks in the 42 ranges contain a key ending with the pattern's last four
suffix bytes; a 2 KiB bloom filter would pass 5,027. The implementation
([design](../../dictionary-suffix-filters.md)) attaches a 1 KiB filter and an
entry count to each shared-cache block after its first complete scan and skips
**interior** blocks the filter excludes, charging their entries to the scan
budget so results, expansion limits and budget errors are identical (the
826-expression probe still reproduces all 149 budget errors). A regression test
pins identical results and budget-error boundaries with and without skipping.

Paired mirror M1 F1 L1 F2 M2 L2 ([results](filter-results.json), 25,499,419
requests, zero errors, exact audit):

| Family        | Op      | Before QPS | **Filters QPS** | Luxir QPS |    Change | CPU µs/request before → after |
| ------------- | ------- | ---------: | --------------: | --------: | --------: | ----------------------------: |
| wildcard_scan | TOP_10  |        853 |       **4,712** |     1,823 | **5.52×** |                34,430 → 5,957 |
| wildcard_scan | TOP_100 |        853 |           4,667 |     1,820 |     5.47× |                34,455 → 5,986 |
| wildcard_scan | COUNT   |        790 |           3,432 |     1,743 |     4.35× |                37,115 → 8,357 |
| and_high_med  | TOP_10  |      4,158 |           4,213 |     8,972 |    1.013× |                 6,728 → 6,679 |
| or_high_low   | TOP_10  |     18,181 |          18,859 |    34,615 |    1.037× |                 1,400 → 1,358 |
| regex         | COUNT   |      4,828 |           4,746 |    13,556 |    0.983× |                 5,810 → 5,826 |

Other cells move −1.7% to +3.7%. This run's host session was slower for every
engine (single-term QPS ~65k vs ~82k earlier), so the in-run ratios are the
result. Anonymous RSS is 389 vs 376 MiB (+13 MiB of filters, charged to the
256 MiB cache budget). On the Apple M4 probe the same rotation drops from 20.7
to 6.5 ms per cycle (3.2×) with identical hits, measured on a loaded machine.
The new profile moves to the shared cache's lookup lock (8.4% of samples plus
0.7% contended slow path) and posting-union sorting (9.2%).

## Final 19-family comparison with stock defaults

After enabling everything by default, all 19 Searchbench families ran with the
unmodified benchmark adapter in mirrored order N1 O1 L1 L2 O2 N2: the final
build (N, [source archive](final-results.json) `9c6ae24c…4c820`), the
September 26 baseline binary with its defaults (O) and pinned Luxir 0.1.0 (L).
Both Summa binaries pass the 677-query exact audit and 2,031 HTTP checks per
phase. **66,953,465 requests, zero errors.** [Results](final-results.json).

| Family (TOP_10 unless noted) | Old QPS | **New QPS** | Luxir QPS | New / old | **New / Luxir** |
| ---------------------------- | ------: | ----------: | --------: | --------: | --------------: |
| wildcard_scan                |     326 |   **4,672** |     1,825 |     14.3× |       **2.56×** |
| wildcard_scan COUNT          |     316 |       3,445 |     1,747 |     10.9× |           1.97× |
| wildcard                     |  11,469 |      16,394 |    29,589 |     1.43× |           0.55× |
| high_sloppy_phrase           |   7,049 |       7,064 |       374 |     1.00× |           18.9× |
| high_phrase                  |   2,045 |       2,085 |     1,158 |     1.02× |           1.80× |
| high_term TOP_100            |  25,874 |      25,530 |    15,956 |     0.99× |           1.60× |
| high_term                    |  62,037 |      61,543 |    62,656 |     0.99× |           0.98× |
| and_high_high                |   2,323 |       2,267 |     2,297 |     0.98× |           0.99× |
| or_high_high                 |   2,067 |       2,044 |     2,360 |     0.99× |           0.87× |
| med_term                     |  69,024 |      68,183 |    90,950 |     0.99× |           0.75× |
| or_high_med                  |   5,269 |       5,235 |     7,922 |     0.99× |           0.66× |
| regex                        |  70,018 |      69,636 |   118,105 |     0.99× |           0.59× |
| low_term                     |  67,569 |      66,712 |   115,156 |     0.99× |           0.58× |
| prefix3                      |  67,644 |      68,294 |   118,791 |     1.01× |           0.57× |
| and_high_low                 |  27,581 |      26,353 |    48,976 |     0.96× |           0.54× |
| or_high_low                  |  19,307 |      18,917 |    36,772 |     0.98× |           0.51× |
| and_high_med                 |   4,293 |       4,217 |     8,965 |     0.98× |           0.47× |
| regex COUNT                  |   4,891 |       4,886 |    13,623 |     1.00× |           0.36× |

Across all 57 cells, families that do not scan the dictionary move −4.5%
(and_high_low TOP_10) to +5.2% (low_phrase TOP_100); phrase families gain
2–5%, term and conjunction families lose 1–4.5%, consistent with the
code-placement shifts documented above. End-of-phase anonymous RSS is 394 MiB
versus 277–279 MiB for the old build (+116 MiB: the dictionary cache and
filters); total RSS 3,955 versus 3,837 MiB; Luxir 1,196–1,213 MiB total.

## Single-hop search dispatch

A low-term profile attributed about 30% of CPU to scheduling and rayon
work-stealing, and native diagnostics put the search itself at ~35 µs of the
~370 µs per request. Both request paths crossed threads twice: the benchmark
adapter handed each request to a `spawn_blocking` thread that then blocked on
`Searcher`'s rayon pool, and `summa-server` blocked a Tokio worker
(`block_in_place`) that then blocked on the same pool. Running single-segment
work outside the pool is not an option — phrase loading, Seismic and segment
opening use nested rayon parallelism — so the request now becomes one pool
job:

- `Searcher::run_on_search_pool` spawns the work on the index's pool and awaits
  a oneshot; searches inside run inline on the pool thread, panics come back as
  errors, and a dropped future lets the job finish.
- `Searcher::search_budgeted_on_pool(Arc<Self>, Arc<dyn Query>, …)` wraps the
  existing sync search for request handlers; `summa-server`'s search RPC uses
  it. Empty searchers and explicit lazy sparse reads keep the async path.
  A test pins identical results to the borrowed methods.
- The benchmark adapter's new `pool` dispatch is its default.

One binary served both arms (`serve … blocking` vs `serve … pool`), all 19
families, mirrored B1 P1 L1 P2 B2 with pinned Luxir: **70,657,856 requests,
zero errors**; each phase's log records the dispatch mode.
[Results](dispatch-results.json).

| Family        | Op     | Blocking QPS | **Pool QPS** | Luxir QPS | Pool / blocking | **Pool / Luxir** | CPU µs/request blocking → pool |
| ------------- | ------ | -----------: | -----------: | --------: | --------------: | ---------------: | -----------------------------: |
| low_term      | TOP_10 |       82,986 |  **129,300** |   124,420 |           1.56× |        **1.04×** |                      303 → 206 |
| med_term      | TOP_10 |       84,787 |  **123,501** |    96,265 |           1.46× |        **1.28×** |                      305 → 215 |
| high_term     | TOP_10 |       70,463 |   **84,870** |    67,495 |           1.20× |        **1.26×** |                      367 → 301 |
| low_term      | COUNT  |      112,187 |  **149,617** |   149,058 |           1.33× |        **1.00×** |                       57 → 156 |
| regex         | TOP_10 |       89,693 |  **152,006** |   133,203 |           1.69× |        **1.14×** |                      289 → 180 |
| prefix3       | TOP_10 |       87,960 |  **124,675** |   128,253 |           1.42× |        **0.97×** |                      294 → 219 |
| and_high_low  | TOP_10 |       28,665 |   **30,586** |    54,549 |           1.07× |        **0.56×** |                      881 → 831 |
| and_high_med  | TOP_10 |        4,213 |    **4,382** |     9,085 |           1.04× |        **0.48×** |                  6,556 → 6,571 |
| or_high_low   | TOP_10 |       19,884 |   **20,896** |    40,068 |           1.05× |        **0.52×** |                  1,284 → 1,232 |
| wildcard_scan | TOP_10 |        5,197 |    **5,345** |     1,848 |           1.03× |        **2.89×** |                  5,426 → 5,375 |
| regex         | COUNT  |        4,748 |    **4,988** |    13,586 |           1.05× |        **0.37×** |                  5,861 → 5,703 |
| and_high_low  | COUNT  |       54,379 |   **51,231** |    55,419 |           0.94× |        **0.92×** |                      420 → 481 |

Across all 57 cells pool dispatch changes throughput by −5.8% to +69%. Cheap
ranked queries gain the most and now match or beat Luxir: term, regex and
prefix TOP_10/TOP_100. Posting-heavy families gain 3–8%. Exact counts whose
work is tiny rise in throughput (term COUNT +33%) but cost more CPU per request
(57 → 156 µs for low-term COUNT): the adapter's count path never used rayon,
and pool threads spin on the spare capacity of these driver-limited cells.
and_high_low COUNT is the one cell that loses (−5.8%).

Family CPU profiles of the same binary ([dispatch-results.json](dispatch-results.json)
`profiles`) locate the remaining gaps in posting work: `and_high_med` spends 31%
in the block-intersection kernel plus 10% each in 8-bit delta decoding and block
bounds; `or_high_low` spends its time in window execution, bounds and block
seeks; regex spends 8% validating each scanned term as UTF-8 before the DFA. A
jump-to-first-≥ intersection kernel was screened on the extracted kernel
harness and rejected: 0.47× on dense and 0.12× on sparse-right block pairs, but
3.4–6.0× slower on medium densities ([screen](intersection-screen.json),
[source](intersection-screen-source.rs.txt)).

## What would close more of the gap

With single-hop dispatch, Summa matches or beats Luxir on term, phrase,
sloppy-phrase, regex/prefix ranked and broad-wildcard families. The remaining
gaps are posting work: regex COUNT (0.37×), conjunctions with a low- or
medium-frequency term (0.48–0.56×), or_high_low TOP_10 (0.52×), counts of
high-frequency conjunctions/disjunctions (~0.55×) and the narrow wildcard family
(0.58×); memory remains about 3.3× Luxir's RSS.

1. **Conjunction intersection:** collect per-call block-shape distributions
   from real queries before changing the kernel; the extracted harness's
   fixtures do not reproduce the measured per-request cost.
2. **Union counting** for regex/prefix COUNT: decode-and-set-bits is the cost;
   measure decode throughput against Luxir's counts.
3. **Regex term checks:** match bytes directly instead of validating UTF-8 per
   scanned term, and extend suffix filters to three-byte endings and
   multi-prefix lookups.
4. **Shard the shared cache lock** and make the default build layout-stable.

## Validation and reproduction

- Production changes: the process-wide term cache, the tight key-only scan,
  word-sized literal checks, dictionary-block suffix filters, bounded bloom
  construction for dictionary rewrites, single-hop pool dispatch that keeps
  the server's admission permit until the pool job finishes, and the
  broker's bound-address log.
  `scripts/check_search.py check` passes all five stages with 2,137 native
  tests (25 ignored); the WASM release build and 41 JavaScript tests pass on
  the final tree. The cloud binaries predate only the bloom-construction,
  admission-permit and broker-test fixes, none of which touch the benchmark
  adapter's search path over the prebuilt index.
- The value-skip patch applies to the pre-change tree for reference.
- Exact audits: 677/677 for every Summa binary and cache setting used.
- HTTP: `summarize.py cache <artifacts> <archives>` and `summarize.py tight …`
  validate completion, fixture and replay inventories, binary and archive
  hashes, the per-phase environment/log evidence of the cache setting, every
  Summa response against the audit, and Luxir coverage, then write
  [results.json](results.json), [tight-results.json](tight-results.json),
  [shared-results.json](shared-results.json),
  [tight2-results.json](tight2-results.json),
  [matcher-results.json](matcher-results.json),
  [filter-results.json](filter-results.json),
  [final-results.json](final-results.json) and
  [dispatch-results.json](dispatch-results.json) (`summarize.py <campaign> …`);
  several include flat CPU profiles.
- Archives: cache build `2d59f79d…6056ae`, tight build `14cf40b3…caa003`,
  shared-cache build `f51695f8…790a33`, tight-on-shared `013a902a…cadb7`,
  matcher `0e19ff0b…802b3`, filter `8c1cc547…cad6cb3`, final `9c6ae24c…4c820`,
  dispatch `bbe10976…29f4a`;
  raw benchmark captures `ef1ad61b…d377`, `4f74dc1e…b7ec8`, `c55b29e5…ac96`,
  `37d33114…27d27f`, `0d8d3c49…785f58`, `9ad7396c…fdcaee`,
  `64a0c692…09b1` and `506cfc20…220db5`; first
  source-VM capture `7ed04278…a10c`, all retained privately. Local laptop
  criterion runs were too noisy (heterogeneous cores) and are excluded.
- Both cloud VMs were stopped and confirmed `TERMINATED`.
