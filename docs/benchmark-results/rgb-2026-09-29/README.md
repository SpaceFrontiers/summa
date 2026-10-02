# RGB-reordered body field against Luxir — September 29

x86 GCP campaign (32-vCPU n2 benchmark host: 30 server CPUs, 2 driver CPUs,
32 clients, network-isolated), all 19 Searchbench families, 73,145,095
requests, zero errors, phases c1 r1 l1 r2 c2. One binary (the final tree,
adding only the harness build mode `index-unicode-impacts-rgb`) serves two
fresh 30-worker builds of the corpus with the default codec and impact
bounds:

- **default:** ordinary document order (12m49s build, 18 GB);
- **RGB:** the body field reordered by RGB after the final merge
  ([design](../../maxscore-text-reordering.md); 25m20s build, 28 GB peak
  RSS, 14 GB);
- **Luxir 0.1.0**, pinned.

[Results](campaign-results.json), [validator](summarize.py). Both audits
match the September 25 baseline on every count, plan, ranked score sequence
and score group except the last; every HTTP response matches its index's
audit.

## Results (QPS, means of phase medians)

| Family             | Op      | Default |     RGB | RGB / default | Default / Luxir | **RGB / Luxir** |
| ------------------ | ------- | ------: | ------: | ------------: | --------------: | --------------: |
| and_high_high      | TOP_10  |   2,844 |   4,647 |         1.63× |           1.23× |       **2.01×** |
| and_high_high      | TOP_100 |   1,614 |   2,531 |         1.57× |           0.92× |       **1.44×** |
| and_high_high      | COUNT   |   3,600 |   4,743 |         1.32× |           0.84× |       **1.11×** |
| and_high_low       | TOP_10  |  35,954 |  61,835 |         1.72× |           0.63× |       **1.08×** |
| and_high_low       | TOP_100 |  28,976 |  51,393 |         1.77× |           0.79× |       **1.40×** |
| and_high_low       | COUNT   |  56,412 | 105,836 |         1.88× |           1.00× |       **1.87×** |
| and_high_med       | TOP_10  |   5,487 |   8,886 |         1.62× |           0.60× |       **0.97×** |
| and_high_med       | TOP_100 |   3,852 |   5,847 |         1.52× |           0.62× |       **0.94×** |
| and_high_med       | COUNT   |   6,333 |   8,329 |         1.32× |           1.00× |       **1.32×** |
| high_phrase        | TOP_10  |   2,166 |   1,036 |         0.48× |           1.86× |       **0.89×** |
| high_phrase        | TOP_100 |     504 |     316 |         0.63× |           1.58× |       **0.99×** |
| high_phrase        | COUNT   |      56 |      61 |         1.09× |           0.94× |       **1.03×** |
| high_sloppy_phrase | TOP_10  |   7,950 |   7,787 |         0.98× |          21.07× |      **20.64×** |
| high_sloppy_phrase | TOP_100 |   2,770 |   2,840 |         1.03× |           8.36× |       **8.57×** |
| high_sloppy_phrase | COUNT   |     181 |     187 |         1.04× |           1.18× |       **1.23×** |
| high_term          | TOP_10  |  80,837 |  74,856 |         0.93× |           1.19× |       **1.10×** |
| high_term          | TOP_100 |  28,030 |  25,337 |         0.90× |           1.73× |       **1.57×** |
| high_term          | COUNT   | 149,870 | 149,763 |         1.00× |           1.02× |       **1.01×** |
| low_phrase         | TOP_10  |   2,480 |   2,510 |         1.01× |           1.72× |       **1.74×** |
| low_phrase         | TOP_100 |     900 |     970 |         1.08× |           1.15× |       **1.24×** |
| low_phrase         | COUNT   |     357 |     426 |         1.19× |           0.95× |       **1.13×** |
| low_sloppy_phrase  | TOP_10  |     946 |     886 |         0.94× |           2.60× |       **2.44×** |
| low_sloppy_phrase  | TOP_100 |     522 |     558 |         1.07× |           1.62× |       **1.73×** |
| low_sloppy_phrase  | COUNT   |     356 |     406 |         1.14× |           1.22× |       **1.40×** |
| low_term           | TOP_10  | 129,213 | 136,256 |         1.05× |           1.05× |       **1.11×** |
| low_term           | TOP_100 |  45,344 |  54,143 |         1.19× |           1.25× |       **1.49×** |
| low_term           | COUNT   | 151,088 | 149,338 |         0.99× |           1.02× |       **1.01×** |
| med_phrase         | TOP_10  |   1,699 |   1,556 |         0.92× |           1.67× |       **1.53×** |
| med_phrase         | TOP_100 |     579 |     577 |         1.00× |           1.12× |       **1.12×** |
| med_phrase         | COUNT   |     157 |     176 |         1.12× |           0.99× |       **1.11×** |
| med_sloppy_phrase  | TOP_10  |   1,304 |   1,354 |         1.04× |           3.63× |       **3.77×** |
| med_sloppy_phrase  | TOP_100 |     806 |     855 |         1.06× |           2.70× |       **2.86×** |
| med_sloppy_phrase  | COUNT   |     271 |     303 |         1.12× |           1.19× |       **1.34×** |
| med_term           | TOP_10  | 122,467 | 119,572 |         0.98× |           1.28× |       **1.25×** |
| med_term           | TOP_100 |  37,109 |  36,217 |         0.98× |           1.65× |       **1.61×** |
| med_term           | COUNT   | 150,350 | 150,108 |         1.00× |           1.01× |       **1.01×** |
| or_high_high       | TOP_10  |   2,039 |   4,395 |         2.16× |           0.86× |       **1.85×** |
| or_high_high       | TOP_100 |   1,480 |   2,488 |         1.68× |           0.82× |       **1.37×** |
| or_high_high       | COUNT   |   3,659 |   4,806 |         1.31× |           0.84× |       **1.11×** |
| or_high_low        | TOP_10  |  24,725 |  45,569 |         1.84× |           0.60× |       **1.11×** |
| or_high_low        | TOP_100 |  14,712 |  20,469 |         1.39× |           0.88× |       **1.23×** |
| or_high_low        | COUNT   |  48,175 |  89,546 |         1.86× |           5.11× |       **9.50×** |
| or_high_med        | TOP_10  |   5,712 |   9,062 |         1.59× |           0.71× |       **1.13×** |
| or_high_med        | TOP_100 |   3,997 |   5,854 |         1.46× |           0.72× |       **1.05×** |
| or_high_med        | COUNT   |   6,083 |   7,797 |         1.28× |           0.91× |       **1.17×** |
| prefix3            | TOP_10  | 123,684 |   3,676 |         0.03× |           0.98× |       **0.03×** |
| prefix3            | TOP_100 | 109,202 |   3,687 |         0.03× |           0.97× |       **0.03×** |
| prefix3            | COUNT   |   7,700 |   3,462 |         0.45× |           0.77× |       **0.34×** |
| regex              | TOP_10  | 149,350 |   5,202 |         0.03× |           1.13× |       **0.04×** |
| regex              | TOP_100 | 136,601 |   5,149 |         0.04× |           1.20× |       **0.05×** |
| regex              | COUNT   |   8,749 |   4,496 |         0.51× |           0.64× |       **0.33×** |
| wildcard           | TOP_10  |  36,201 |   1,928 |         0.05× |           1.21× |       **0.06×** |
| wildcard           | TOP_100 |  34,430 |   1,926 |         0.06× |           1.18× |       **0.07×** |
| wildcard           | COUNT   |   5,055 |   1,847 |         0.37× |           0.75× |       **0.27×** |
| wildcard_scan      | TOP_10  |  10,360 |   6,209 |         0.60× |           5.61× |       **3.36×** |
| wildcard_scan      | TOP_100 |  10,246 |   6,175 |         0.60× |           5.56× |       **3.35×** |
| wildcard_scan      | COUNT   |   6,876 |   5,375 |         0.78× |           3.88× |       **3.03×** |

**Ranked conjunctions and disjunctions overtake Luxir.** RGB clusters
documents that share terms, so block bounds separate far better: TOP_10
throughput rises 1.6–2.2× for every AND/OR family, and exact counts over
frequent terms 1.3–1.9× (denser blocks store more bitmaps). All AND/OR
families except and_high_med (0.97× TOP_10) now beat Luxir.

**Pattern queries collapse.** prefix3, regex and wildcard top-k fall to
0.03–0.07× of the default index and their counts to 0.37–0.51×. Every match
of a pattern scores 1.0 and ties break by stable (logical) document ID, a
[reordering requirement](../../maxscore-text-reordering.md). On the default
index the union streams postings in ID order and stops after `k` matches; on a
reordered field postings follow physical order, so finding the `k` smallest
logical IDs materializes the whole union through the document map (one
random bitset write per posting). Counts need no mapping at all but take the
same mapped path, which also forgoes the dense bitmap-window fill.

**high_phrase top-k halves** (0.48× TOP_10, 0.63× TOP_100) while other phrase
families are unchanged; not yet investigated.

## Follow-up: pattern counts in physical IDs

Prefix, wildcard and regex queries now opt into physical traversal for
complete streams (counts), like term queries: their unions are built without
the document map, so dense postings fill whole bitset windows, and the count
does not depend on the numbering. Ranked streams, and ranked Boolean queries
with such a clause, keep the logical union and its stable-ID ties. Test:
`expanded_term_counts_on_rgb_fields_traverse_physical_ids`.

Single-thread count medians on the 10M RGB index (x86, the campaign binary
against the new tree, both built with the count probe, A B A B, identical
counts): all 17 pattern counts 120.8 → 52.0 ms (2.3×); prefix3 2.0–2.2×,
wildcard 2–5×, regex 1.0–5.5×. That matches the default index's counts. On a
1M-document aarch64 index prefix3 counts measured 16–31% slower instead
(wildcard 29% and regex 13% faster).

**high_phrase** runs the same phrase code on both indexes but visits far more
candidates on the RGB one (`"in a"` TOP_10 13.5 → 101 ms). Its top-10 scores
are nearly all distinct, so tie order is not the cause; phrase bounds derive
from the terms' frequencies, and RGB clusters documents rich in both terms
into the same blocks, where the bounds stay high while the phrase is rarer.
