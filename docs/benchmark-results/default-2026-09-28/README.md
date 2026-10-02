# Ranked conjunctions, disjunction windows and bitmap blocks by default — September 28

x86 GCP campaign (32-vCPU n2 benchmark host: 30 server CPUs, 2 driver CPUs,
32 clients, network-isolated), all 19 Searchbench families, 113,059,400
requests, zero errors, nine mirrored phases (b1 c1 d1 r1 l1 r2 d2 c2 b2):

- **before:** the September 27 campaign binary on the September 24 index.
- **code:** the final tree on the same index: today's query-execution
  changes with the index unchanged.
- **default / rounded:** the final tree on two fresh 30-worker builds of the
  corpus by one indexing binary, with the new default codec
  (`RoundedBitmap`) and with `Rounded`.
- **Luxir 0.1.0**, pinned.

[Results](campaign-results.json), [validator](summarize.py).

## What changed

- **Pruned conjunction probes** (ranked two-term conjunctions): a rarer-term
  document whose bound by frequency plus the common block's bound cannot
  reach the threshold is skipped, and the rest are probed in the common block
  instead of merging it (`probe_competitive`, and the same filter on bitmap
  blocks).
- **Disjunction windows:** globally non-essential cursors first bound a window
  by their list maxima; windows that lose even so are skipped without
  per-block bounds.
- **`RoundedBitmap` by default** for the `adaptive` and `performance`
  optimization modes ([design](../../bitmap-posting-blocks.md)).

## Results (QPS, means of phase medians)

| Family             | Op      |  Before |    Code | Code / before | Rounded (fresh) | Default (fresh) | Default / rounded | Before / Luxir | **Code / Luxir** | **Default / Luxir** |
| ------------------ | ------- | ------: | ------: | ------------: | --------------: | --------------: | ----------------: | -------------: | ---------------: | ------------------: |
| and_high_high      | TOP_10  |   2,274 |   2,701 |         1.19× |           2,671 |           2,668 |             1.00× |          0.99× |        **1.17×** |           **1.16×** |
| and_high_high      | TOP_100 |   1,413 |   1,592 |         1.13× |           1,575 |           1,567 |             1.00× |          0.81× |        **0.91×** |           **0.90×** |
| and_high_high      | COUNT   |   2,370 |   2,383 |         1.01× |           2,384 |           3,569 |             1.50× |          0.56× |        **0.56×** |           **0.84×** |
| and_high_low       | TOP_10  |  27,867 |  27,925 |         1.00× |          26,992 |          26,501 |             0.98× |          0.59× |        **0.59×** |           **0.56×** |
| and_high_low       | TOP_100 |  25,782 |  25,533 |         0.99× |          24,886 |          24,789 |             1.00× |          0.79× |        **0.78×** |           **0.76×** |
| and_high_low       | COUNT   |  43,984 |  44,770 |         1.02× |          43,275 |          47,162 |             1.09× |          0.86× |        **0.88×** |           **0.93×** |
| and_high_med       | TOP_10  |   4,193 |   5,133 |         1.22× |           5,040 |           4,989 |             0.99× |          0.47× |        **0.57×** |           **0.56×** |
| and_high_med       | TOP_100 |   3,258 |   3,780 |         1.16× |           3,744 |           3,693 |             0.99× |          0.53× |        **0.61×** |           **0.60×** |
| and_high_med       | COUNT   |   4,990 |   5,048 |         1.01× |           5,030 |           6,231 |             1.24× |          0.79× |        **0.80×** |           **0.99×** |
| high_phrase        | TOP_10  |   2,045 |   2,089 |         1.02× |           2,057 |           1,967 |             0.96× |          1.79× |        **1.83×** |           **1.72×** |
| high_phrase        | TOP_100 |     510 |     510 |         1.00× |             502 |             476 |             0.95× |          1.58× |        **1.58×** |           **1.47×** |
| high_phrase        | COUNT   |      62 |      59 |         0.94× |              59 |              57 |             0.98× |          0.99× |        **0.93×** |           **0.91×** |
| high_sloppy_phrase | TOP_10  |   7,136 |   7,275 |         1.02× |           7,205 |           7,083 |             0.98× |         19.14× |       **19.52×** |          **19.00×** |
| high_sloppy_phrase | TOP_100 |   2,739 |   2,785 |         1.02× |           2,652 |           2,621 |             0.99× |          8.31× |        **8.45×** |           **7.95×** |
| high_sloppy_phrase | COUNT   |     190 |     182 |         0.96× |             182 |             180 |             0.99× |          1.26× |        **1.20×** |           **1.19×** |
| high_term          | TOP_10  |  70,872 |  72,860 |         1.03× |          71,673 |          70,972 |             0.99× |          1.18× |        **1.22×** |           **1.19×** |
| high_term          | TOP_100 |  26,418 |  26,728 |         1.01× |          25,824 |          26,265 |             1.02× |          1.67× |        **1.69×** |           **1.66×** |
| high_term          | COUNT   | 107,130 | 109,842 |         1.03× |         109,524 |         108,381 |             0.99× |          0.87× |        **0.89×** |           **0.88×** |
| low_phrase         | TOP_10  |   2,368 |   2,390 |         1.01× |           2,284 |           2,376 |             1.04× |          1.68× |        **1.70×** |           **1.69×** |
| low_phrase         | TOP_100 |     900 |     904 |         1.00× |             887 |             883 |             1.00× |          1.16× |        **1.17×** |           **1.14×** |
| low_phrase         | COUNT   |     371 |     362 |         0.98× |             361 |             357 |             0.99× |          1.08× |        **1.05×** |           **1.04×** |
| low_sloppy_phrase  | TOP_10  |     927 |     943 |         1.02× |             922 |             923 |             1.00× |          2.51× |        **2.55×** |           **2.50×** |
| low_sloppy_phrase  | TOP_100 |     517 |     522 |         1.01× |             518 |             519 |             1.00× |          1.63× |        **1.65×** |           **1.64×** |
| low_sloppy_phrase  | COUNT   |     362 |     368 |         1.02× |             372 |             369 |             0.99× |          1.27× |        **1.29×** |           **1.29×** |
| low_term           | TOP_10  |  96,498 |  98,358 |         1.02× |          99,040 |          99,566 |             1.01× |          0.86× |        **0.88×** |           **0.89×** |
| low_term           | TOP_100 |  42,328 |  43,090 |         1.02× |          42,659 |          42,448 |             1.00× |          1.26× |        **1.28×** |           **1.26×** |
| low_term           | COUNT   | 104,618 | 107,227 |         1.02× |         108,388 |         109,108 |             1.01× |          0.87× |        **0.89×** |           **0.90×** |
| med_phrase         | TOP_10  |   1,736 |   1,740 |         1.00× |           1,770 |           1,663 |             0.94× |          1.71× |        **1.71×** |           **1.64×** |
| med_phrase         | TOP_100 |     573 |     578 |         1.01× |             579 |             575 |             0.99× |          1.12× |        **1.13×** |           **1.13×** |
| med_phrase         | COUNT   |     164 |     158 |         0.96× |             157 |             156 |             0.99× |          1.04× |        **1.00×** |           **0.99×** |
| med_sloppy_phrase  | TOP_10  |   1,283 |   1,298 |         1.01× |           1,277 |           1,278 |             1.00× |          3.59× |        **3.63×** |           **3.58×** |
| med_sloppy_phrase  | TOP_100 |     801 |     810 |         1.01× |             804 |             788 |             0.98× |          2.72× |        **2.75×** |           **2.67×** |
| med_sloppy_phrase  | COUNT   |     282 |     280 |         0.99× |             277 |             275 |             0.99× |          1.26× |        **1.25×** |           **1.23×** |
| med_term           | TOP_10  |  96,534 |  97,280 |         1.01× |         101,156 |          97,922 |             0.97× |          1.09× |        **1.10×** |           **1.11×** |
| med_term           | TOP_100 |  33,209 |  33,993 |         1.02× |          33,959 |          34,511 |             1.02× |          1.52× |        **1.56×** |           **1.58×** |
| med_term           | COUNT   | 106,251 | 108,346 |         1.02× |         109,902 |         107,433 |             0.98× |          0.84× |        **0.86×** |           **0.85×** |
| or_high_high       | TOP_10  |   2,049 |   2,092 |         1.02× |           2,046 |           2,049 |             1.00× |          0.87× |        **0.89×** |           **0.87×** |
| or_high_high       | TOP_100 |   1,481 |   1,515 |         1.02× |           1,481 |           1,480 |             1.00× |          0.83× |        **0.84×** |           **0.82×** |
| or_high_high       | COUNT   |   2,328 |   2,350 |         1.01× |           2,334 |           3,639 |             1.56× |          0.54× |        **0.55×** |           **0.85×** |
| or_high_low        | TOP_10  |  19,289 |  22,214 |         1.15× |          21,522 |          21,699 |             1.01× |          0.56× |        **0.64×** |           **0.63×** |
| or_high_low        | TOP_100 |  13,059 |  13,704 |         1.05× |          13,050 |          13,215 |             1.01× |          0.83× |        **0.87×** |           **0.84×** |
| or_high_low        | COUNT   |  39,088 |  39,893 |         1.02× |          39,327 |          41,285 |             1.05× |          4.21× |        **4.30×** |           **4.45×** |
| or_high_med        | TOP_10  |   5,418 |   5,647 |         1.04× |           5,557 |           5,592 |             1.01× |          0.69× |        **0.72×** |           **0.71×** |
| or_high_med        | TOP_100 |   3,936 |   3,995 |         1.02× |           3,931 |           3,973 |             1.01× |          0.73× |        **0.74×** |           **0.73×** |
| or_high_med        | COUNT   |   4,702 |   4,762 |         1.01× |           4,737 |           5,955 |             1.26× |          0.71× |        **0.72×** |           **0.90×** |
| prefix3            | TOP_10  |  87,005 |  93,069 |         1.07× |          99,526 |          82,659 |             0.83× |          0.74× |        **0.79×** |           **0.70×** |
| prefix3            | TOP_100 |  80,948 |  83,989 |         1.04× |          90,709 |          74,926 |             0.83× |          0.76× |        **0.78×** |           **0.70×** |
| prefix3            | COUNT   |   7,731 |   7,820 |         1.01× |           7,792 |           7,811 |             1.00× |          0.77× |        **0.78×** |           **0.78×** |
| regex              | TOP_10  | 110,209 | 107,973 |         0.98× |         107,965 |         107,078 |             0.99× |          0.96× |        **0.94×** |           **0.93×** |
| regex              | TOP_100 |  99,654 |  99,340 |         1.00× |         100,069 |         100,446 |             1.00× |          0.95× |        **0.95×** |           **0.96×** |
| regex              | COUNT   |   7,275 |   7,361 |         1.01× |           7,240 |           8,451 |             1.17× |          0.53× |        **0.54×** |           **0.62×** |
| wildcard           | TOP_10  |  17,037 |  17,016 |         1.00× |          17,093 |          16,719 |             0.98× |          0.58× |        **0.58×** |           **0.57×** |
| wildcard           | TOP_100 |  16,673 |  16,742 |         1.00× |          17,001 |          16,578 |             0.98× |          0.59× |        **0.59×** |           **0.58×** |
| wildcard           | COUNT   |   4,003 |   4,011 |         1.00× |           3,980 |           4,399 |             1.11× |          0.60× |        **0.60×** |           **0.66×** |
| wildcard_scan      | TOP_10  |   4,836 |   4,895 |         1.01× |           5,174 |           5,148 |             0.99× |          2.64× |        **2.68×** |           **2.81×** |
| wildcard_scan      | TOP_100 |   4,820 |   4,868 |         1.01× |           5,142 |           5,125 |             1.00× |          2.65× |        **2.68×** |           **2.82×** |
| wildcard_scan      | COUNT   |   4,110 |   4,062 |         0.99× |           4,178 |           4,164 |             1.00× |          2.35× |        **2.32×** |           **2.38×** |

Highlights:

- **Query execution (code / before, same index):** and_high_med TOP_10
  +22% and TOP_100 +16%, and_high_high TOP_10 +19% and TOP_100 +13%,
  or_high_low TOP_10 +15%, prefix3 TOP_10 +7%. and_high_high TOP_10 now beats
  Luxir (1.18×); and_high_med TOP_10 rises from 0.47× to 0.57×, or_high_low
  TOP_10 from 0.56× to 0.64×.
- **Bitmap blocks (default / rounded, same builder):** COUNT and_high_high
  +50% (0.84× Luxir), or_high_high +56% (0.85×), or_high_med +26% (0.90×),
  and_high_med +24% (0.99×), regex +17%, wildcard +11%, and_high_low +9%
  (0.93×).
- **Fresh builds:** with 30 indexing workers, as for the September 24 index,
  file sizes match it (`.post` 8.75 GB `Rounded`, 8.63 GB bitmap, 8.70 GB on
  September 24) and and_high_low TOP_10 is within 3% of it. The 27% loss
  seen on earlier fresh builds came from building with 8 workers.

## Caveats on the default

The two fresh builds assign document IDs in different orders (parallel
indexing): their `.store`, `.pos` and `.rowstats` bytes differ. Two cells move
against the bitmap build consistently across the mirrored phases:

- **prefix3 TOP_10/TOP_100 −17%/−18%.** An 8-worker pair of builds measured
  +1.3%, and a 1M-document pair on aarch64 is mixed (±5%); a streaming
  top-10 union opens a handful of blocks, so ~45 µs per request is unlikely
  to be decoding. Treated as build variance, unconfirmed.
- **Phrases −3% to −6%** (high_phrase, med_phrase TOP_10). On the 8-worker
  pair the bitmap build confirmed 16% more phrase candidates for the same
  queries (document order, not decoding), but both pairs lean the same way;
  a controlled phrase comparison (identical document order) remains open.

## Validation and reproduction

- `summarize.py <artifacts> <folder with final.tar.gz>` checks the inventory
  of the September 24 index and the pinned replay (unchanged, and identical
  to September 25), both fresh indexes before and after (unchanged), both
  binaries' hashes, audits on the September 24 index against the September
  25 baseline (identical for `before` and `code`), fresh-build audits (every
  count and plan, every ranked score sequence, the IDs of every score group
  but the last; 673 queries each), every Summa HTTP response against the
  audit of the index it served, and Luxir coverage.
- Build: archive `af44788b…d413`, binary `73ada083…948f`, indexing binary
  `394af4d0…15b4` (the same tree with an environment switch forcing
  `Rounded`); raw captures `ade9933b…3079`, retained privately.
- `scripts/check_search.py check` passes all five stages with 2,145 native
  tests (25 ignored); the WASM release build and 41 JavaScript tests pass.
- The temporary scratch disk was deleted; both VMs were stopped.
