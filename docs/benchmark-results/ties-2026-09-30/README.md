# Physical tie order against Luxir — September 30

x86 GCP campaign (32-vCPU n2 benchmark host: 30 server CPUs, 2 driver CPUs,
32 clients, network-isolated), all 19 Searchbench families, 125,820,447
requests, zero errors, phases d1 p1 g1 l1 g2 p2 d2. One binary (the
September 30 tree) serves the three builds of the
[pairs campaign](../pairs-2026-09-29/README.md) (default, pairs, RGB +
pairs); Luxir 0.1.0 is pinned. Against the pairs campaign's binary the tree
adds [physical tie order](../../physical-tie-order.md) for constant-score
queries on reordered fields, ranked word pairs on the term's logical plan for
reordered fields (the pairs campaign's follow-up), and the fix for ranked
conjunctions of a filter-like clause and one scored clause.

[Results](campaign-results.json), [validator](summarize.py). All three
audits match the September 25 baseline on every count, plan, ranked score
sequence and score group except the last, as in the pairs campaign (656, 656
and 662 queries differ from it only in tie order); every HTTP response
matches its index's audit. For constant-score queries the audit's rank check
accepts any distinct matching documents with the oracle's scores (the
oracle orders ties by logical ID).

The host ran faster than in the pairs campaign for every engine (Luxir
geometric mean 1.05×, default build 1.06×), so compare builds through their
ratios to Luxir, which are measured within one campaign.

## Results (QPS, means of phase medians)

| Family             | Op      | Default |   Pairs | RGB + pairs |   Luxir | Default / Luxir | Pairs / Luxir | RGB + pairs / Luxir, Sep 29 | **RGB + pairs / Luxir** |
| ------------------ | ------- | ------: | ------: | ----------: | ------: | --------------: | ------------: | --------------------------: | ----------------------: |
| and_high_high      | TOP_10  |   2,764 |   2,723 |       4,536 |   2,306 |           1.20× |         1.18× |                       1.92× |               **1.97×** |
| and_high_high      | TOP_100 |   1,570 |   1,568 |       2,476 |   1,758 |           0.89× |         0.89× |                       1.39× |               **1.41×** |
| and_high_high      | COUNT   |   3,577 |   3,574 |       4,727 |   4,256 |           0.84× |         0.84× |                       1.11× |               **1.11×** |
| and_high_low       | TOP_10  |  35,760 |  36,049 |      59,055 |  52,941 |           0.68× |         0.68× |                       1.27× |               **1.12×** |
| and_high_low       | TOP_100 |  28,933 |  28,761 |      48,276 |  34,413 |           0.84× |         0.84× |                       1.50× |               **1.40×** |
| and_high_low       | COUNT   |  55,512 |  54,279 |     104,245 |  54,323 |           1.02× |         1.00× |                       1.86× |               **1.92×** |
| and_high_med       | TOP_10  |   5,451 |   5,453 |       8,638 |   9,082 |           0.60× |         0.60× |                       0.98× |               **0.95×** |
| and_high_med       | TOP_100 |   3,831 |   3,826 |       5,822 |   6,255 |           0.61× |         0.61× |                       0.94× |               **0.93×** |
| and_high_med       | COUNT   |   6,247 |   6,271 |       8,193 |   6,302 |           0.99× |         1.00× |                       1.31× |               **1.30×** |
| high_phrase        | TOP_10  |   2,152 | 101,667 |      88,692 |   1,167 |           1.84× |        87.14× |                       1.05× |              **76.02×** |
| high_phrase        | TOP_100 |     496 |  31,379 |      28,466 |     329 |           1.51× |        95.44× |                       3.62× |              **86.58×** |
| high_phrase        | COUNT   |      58 | 134,326 |     130,995 |      57 |           1.01× |      2338.70× |                    2032.13× |            **2280.72×** |
| high_sloppy_phrase | TOP_10  |   7,755 |   7,179 |       6,998 |     376 |          20.65× |        19.11× |                      18.45× |              **18.63×** |
| high_sloppy_phrase | TOP_100 |   2,749 |   2,686 |       2,757 |     330 |           8.32× |         8.13× |                       8.26× |               **8.35×** |
| high_sloppy_phrase | COUNT   |     181 |     179 |         184 |     151 |           1.20× |         1.18× |                       1.28× |               **1.22×** |
| high_term          | TOP_10  |  79,787 |  80,665 |      71,380 |  66,497 |           1.20× |         1.21× |                       1.02× |               **1.07×** |
| high_term          | TOP_100 |  27,983 |  28,174 |      24,323 |  16,385 |           1.71× |         1.72× |                       1.40× |               **1.48×** |
| high_term          | COUNT   | 148,751 | 148,953 |     149,562 | 147,798 |           1.01× |         1.01× |                       0.98× |               **1.01×** |
| low_phrase         | TOP_10  |   2,460 |   7,560 |       8,590 |   1,436 |           1.71× |         5.27× |                       4.05× |               **5.98×** |
| low_phrase         | TOP_100 |     901 |   2,885 |       3,468 |     777 |           1.16× |         3.71× |                       3.71× |               **4.46×** |
| low_phrase         | COUNT   |     356 |     887 |       1,225 |     369 |           0.97× |         2.40× |                       3.22× |               **3.32×** |
| low_sloppy_phrase  | TOP_10  |     945 |     915 |         884 |     369 |           2.56× |         2.48× |                       2.40× |               **2.40×** |
| low_sloppy_phrase  | TOP_100 |     518 |     520 |         556 |     316 |           1.64× |         1.65× |                       1.83× |               **1.76×** |
| low_sloppy_phrase  | COUNT   |     362 |     355 |         408 |     301 |           1.20× |         1.18× |                       1.34× |               **1.35×** |
| low_term           | TOP_10  | 129,544 | 130,029 |     136,563 | 122,980 |           1.05× |         1.06× |                       1.06× |               **1.11×** |
| low_term           | TOP_100 |  44,688 |  45,410 |      52,076 |  35,802 |           1.25× |         1.27× |                       1.32× |               **1.45×** |
| low_term           | COUNT   | 148,511 | 149,052 |     148,871 | 146,285 |           1.02× |         1.02× |                       0.96× |               **1.02×** |
| med_phrase         | TOP_10  |   1,680 |  38,005 |      33,508 |   1,020 |           1.65× |        37.25× |                       4.04× |              **32.84×** |
| med_phrase         | TOP_100 |     577 |  14,569 |      16,021 |     513 |           1.13× |        28.42× |                       6.98× |              **31.25×** |
| med_phrase         | COUNT   |     159 |   1,044 |       1,262 |     160 |           0.99× |         6.54× |                       7.54× |               **7.91×** |
| med_sloppy_phrase  | TOP_10  |   1,298 |   1,338 |       1,315 |     362 |           3.58× |         3.69× |                       3.51× |               **3.63×** |
| med_sloppy_phrase  | TOP_100 |     807 |     807 |         857 |     295 |           2.73× |         2.73× |                       2.79× |               **2.90×** |
| med_sloppy_phrase  | COUNT   |     272 |     272 |         302 |     221 |           1.23× |         1.23× |                       1.33× |               **1.37×** |
| med_term           | TOP_10  | 121,718 | 120,105 |     117,263 |  95,617 |           1.27× |         1.26× |                       1.04× |               **1.23×** |
| med_term           | TOP_100 |  35,837 |  36,214 |      34,796 |  22,308 |           1.61× |         1.62× |                       1.41× |               **1.56×** |
| med_term           | COUNT   | 147,412 | 150,054 |     148,728 | 147,611 |           1.00× |         1.02× |                       0.96× |               **1.01×** |
| or_high_high       | TOP_10  |   2,034 |   2,009 |       4,253 |   2,372 |           0.86× |         0.85× |                       1.79× |               **1.79×** |
| or_high_high       | TOP_100 |   1,470 |   1,457 |       2,419 |   1,811 |           0.81× |         0.80× |                       1.33× |               **1.34×** |
| or_high_high       | COUNT   |   3,635 |   3,611 |       4,735 |   4,322 |           0.84× |         0.84× |                       1.11× |               **1.10×** |
| or_high_low        | TOP_10  |  24,533 |  24,715 |      44,572 |  36,842 |           0.67× |         0.67× |                       1.19× |               **1.21×** |
| or_high_low        | TOP_100 |  14,260 |  14,513 |      20,062 |  16,340 |           0.87× |         0.89× |                       1.14× |               **1.23×** |
| or_high_low        | COUNT   |  47,071 |  47,014 |      82,027 |   9,387 |           5.01× |         5.01× |                       7.67× |               **8.74×** |
| or_high_med        | TOP_10  |   5,559 |   5,685 |       8,997 |   8,038 |           0.69× |         0.71× |                       1.10× |               **1.12×** |
| or_high_med        | TOP_100 |   3,943 |   3,985 |       5,704 |   5,551 |           0.71× |         0.72× |                       1.02× |               **1.03×** |
| or_high_med        | COUNT   |   5,990 |   6,015 |       7,777 |   6,650 |           0.90× |         0.90× |                       1.15× |               **1.17×** |
| prefix3            | TOP_10  | 122,401 | 122,286 |     123,492 | 126,301 |           0.97× |         0.97× |                       0.02× |               **0.98×** |
| prefix3            | TOP_100 | 109,857 | 105,546 |     103,278 | 113,483 |           0.97× |         0.93× |                       0.02× |               **0.91×** |
| prefix3            | COUNT   |   7,578 |   7,603 |       8,549 |  10,060 |           0.75× |         0.76× |                       0.84× |               **0.85×** |
| regex              | TOP_10  | 150,517 | 146,236 |     147,408 | 129,018 |           1.17× |         1.13× |                       0.03× |               **1.14×** |
| regex              | TOP_100 | 137,556 | 136,246 |     130,383 | 111,228 |           1.24× |         1.22× |                       0.03× |               **1.17×** |
| regex              | COUNT   |   8,652 |   8,606 |       9,303 |  13,730 |           0.63× |         0.63× |                       0.68× |               **0.68×** |
| wildcard           | TOP_10  |  35,291 |  34,476 |      30,337 |  29,933 |           1.18× |         1.15× |                       0.05× |               **1.01×** |
| wildcard           | TOP_100 |  33,355 |  33,038 |      29,471 |  28,810 |           1.16× |         1.15× |                       0.05× |               **1.02×** |
| wildcard           | COUNT   |   4,950 |   4,961 |       5,247 |   6,684 |           0.74× |         0.74× |                       0.84× |               **0.79×** |
| wildcard_scan      | TOP_10  |   9,777 |   9,971 |       9,243 |   1,836 |           5.32× |         5.43× |                       2.82× |               **5.03×** |
| wildcard_scan      | TOP_100 |   9,675 |   9,857 |       9,144 |   1,840 |           5.26× |         5.36× |                       2.73× |               **4.97×** |
| wildcard_scan      | COUNT   |   6,786 |   6,806 |       6,876 |   1,762 |           3.85× |         3.86× |                       3.80× |               **3.90×** |

## Findings

**Pattern top-k on the RGB build moves from 0.02–0.05× to Luxir's level.**
prefix3 top-10 0.02× → 0.98×, regex 0.03× → 1.14×, wildcard 0.05× → 1.01×,
top-100 alike (0.91×, 1.17×, 1.02×); wildcard_scan top-k 2.8× → 5.0×. These
queries score every match 1.0 and now return ties in physical order, so their
ranked streams stop after `k` matches. Against the default build the RGB
build is level on prefix3 top-10 and 2–14% slower on the other pattern top-k
cells (wildcard top-10 the most). Pattern counts stay at 0.68–0.85× Luxir on
every build.

**Ranked phrases on RGB + pairs follow the pairs build.** high_phrase top-10
1.05× → 76.0× Luxir (pairs build 87.1×), top-100 3.6× → 86.6×, med_phrase
top-10 4.0× → 32.8×.

**RGB + pairs now matches or beats Luxir in 50 of 57 cells.** It trails on
and_high_med top-k (0.95×, 0.93×), prefix3 top-10/100 (0.98×, 0.91×) and
pattern counts (prefix3 0.85×, wildcard 0.79×, regex 0.68×). The other
families keep their ratios from the pairs campaign except and_high_low
top-k (1.27× → 1.12× at top-10, 1.50× → 1.40× at top-100): Summa's QPS rose
11% there, while Luxir's rose 26%, the largest drift of any Luxir cell.
