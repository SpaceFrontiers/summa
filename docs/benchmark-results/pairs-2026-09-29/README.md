# Common word pairs against Luxir — September 29

x86 GCP campaign (32-vCPU n2 benchmark host: 30 server CPUs, 2 driver CPUs,
32 clients, network-isolated), all 19 Searchbench families, 104,035,781
requests, zero errors, phases d1 p1 g1 l1 g2 p2 d2. One binary (the final
tree) serves three fresh 30-worker builds of the corpus with the default
codec and impact bounds:

- **default:** no word pairs (12m49s build, 18 GB);
- **pairs:** the body field pairs its 128 most document-frequent words
  ([design](../../common-word-pairs.md); list from the first 200,000
  documents, `searchbench_http common-words`; 14m19s build, postings
  8.63 → 9.15 GB, 18 GB total);
- **RGB + pairs:** pairs plus the RGB-reordered body (29m40s, 14 GB);
- **Luxir 0.1.0**, pinned.

[Results](campaign-results.json), [validator](summarize.py). All three
audits match the September 25 baseline on every count, plan, ranked score
sequence and score group except the last; every HTTP response matches its
index's audit.

## Results (QPS, means of phase medians)

| Family             | Op      | Default |   Pairs | Pairs / default | RGB + pairs | Default / Luxir | **Pairs / Luxir** | **RGB + pairs / Luxir** |
| ------------------ | ------- | ------: | ------: | --------------: | ----------: | --------------: | ----------------: | ----------------------: |
| and_high_high      | TOP_10  |   2,685 |   2,645 |           0.99× |       4,287 |           1.21× |         **1.19×** |               **1.92×** |
| and_high_high      | TOP_100 |   1,499 |   1,526 |           1.02× |       2,352 |           0.89× |         **0.90×** |               **1.39×** |
| and_high_high      | COUNT   |   3,470 |   3,507 |           1.01× |       4,560 |           0.84× |         **0.85×** |               **1.11×** |
| and_high_low       | TOP_10  |  32,491 |  34,233 |           1.05× |      53,211 |           0.77× |         **0.81×** |               **1.27×** |
| and_high_low       | TOP_100 |  26,511 |  26,833 |           1.01× |      43,517 |           0.92× |         **0.93×** |               **1.50×** |
| and_high_low       | COUNT   |  49,341 |  49,538 |           1.00× |      85,753 |           1.07× |         **1.08×** |               **1.86×** |
| and_high_med       | TOP_10  |   5,262 |   5,177 |           0.98× |       8,175 |           0.63× |         **0.62×** |               **0.98×** |
| and_high_med       | TOP_100 |   3,745 |   3,674 |           0.98× |       5,493 |           0.64× |         **0.63×** |               **0.94×** |
| and_high_med       | COUNT   |   6,165 |   6,109 |           0.99× |       7,825 |           1.03× |         **1.02×** |               **1.31×** |
| high_phrase        | TOP_10  |   2,065 |  91,671 |          44.39× |       1,109 |           1.95× |        **86.67×** |               **1.05×** |
| high_phrase        | TOP_100 |     489 |  28,424 |          58.16× |       1,102 |           1.61× |        **93.35×** |               **3.62×** |
| high_phrase        | COUNT   |      58 | 125,652 |        2168.03× |     118,942 |           0.99× |      **2146.78×** |            **2032.13×** |
| high_sloppy_phrase | TOP_10  |   7,467 |   6,757 |           0.90× |       6,618 |          20.82× |        **18.84×** |              **18.45×** |
| high_sloppy_phrase | TOP_100 |   2,669 |   2,536 |           0.95× |       2,583 |           8.54× |         **8.11×** |               **8.26×** |
| high_sloppy_phrase | COUNT   |     178 |     178 |           1.00× |         185 |           1.23× |         **1.23×** |               **1.28×** |
| high_term          | TOP_10  |  74,342 |  74,077 |           1.00× |      60,149 |           1.26× |         **1.26×** |               **1.02×** |
| high_term          | TOP_100 |  25,927 |  25,814 |           1.00× |      21,253 |           1.71× |         **1.70×** |               **1.40×** |
| high_term          | COUNT   | 143,293 | 142,939 |           1.00× |     138,530 |           1.02× |         **1.02×** |               **0.98×** |
| low_phrase         | TOP_10  |   2,419 |   6,934 |           2.87× |       5,625 |           1.74× |         **4.99×** |               **4.05×** |
| low_phrase         | TOP_100 |     888 |   2,735 |           3.08× |       2,812 |           1.17× |         **3.60×** |               **3.71×** |
| low_phrase         | COUNT   |     357 |     867 |           2.43× |       1,177 |           0.97× |         **2.37×** |               **3.22×** |
| low_sloppy_phrase  | TOP_10  |     924 |     881 |           0.95× |         843 |           2.64× |         **2.51×** |               **2.40×** |
| low_sloppy_phrase  | TOP_100 |     506 |     505 |           1.00× |         533 |           1.74× |         **1.74×** |               **1.83×** |
| low_sloppy_phrase  | COUNT   |     355 |     361 |           1.02× |         398 |           1.20× |         **1.21×** |               **1.34×** |
| low_term           | TOP_10  | 121,427 | 121,458 |           1.00× |     121,712 |           1.06× |         **1.06×** |               **1.06×** |
| low_term           | TOP_100 |  41,795 |  40,786 |           0.98× |      43,792 |           1.26× |         **1.23×** |               **1.32×** |
| low_term           | COUNT   | 140,816 | 142,210 |           1.01× |     136,912 |           0.98× |         **0.99×** |               **0.96×** |
| med_phrase         | TOP_10  |   1,636 |  34,621 |          21.16× |       4,058 |           1.63× |        **34.46×** |               **4.04×** |
| med_phrase         | TOP_100 |     570 |  13,011 |          22.83× |       3,533 |           1.13× |        **25.71×** |               **6.98×** |
| med_phrase         | COUNT   |     157 |   1,016 |           6.47× |       1,214 |           0.97× |         **6.31×** |               **7.54×** |
| med_sloppy_phrase  | TOP_10  |   1,253 |   1,242 |           0.99× |       1,240 |           3.55× |         **3.52×** |               **3.51×** |
| med_sloppy_phrase  | TOP_100 |     768 |     776 |           1.01× |         798 |           2.69× |         **2.72×** |               **2.79×** |
| med_sloppy_phrase  | COUNT   |     263 |     264 |           1.00× |         297 |           1.18× |         **1.19×** |               **1.33×** |
| med_term           | TOP_10  | 110,989 | 108,546 |           0.98× |      94,273 |           1.22× |         **1.19×** |               **1.04×** |
| med_term           | TOP_100 |  32,536 |  30,648 |           0.94× |      30,462 |           1.50× |         **1.42×** |               **1.41×** |
| med_term           | COUNT   | 139,134 | 132,490 |           0.95× |     138,189 |           0.97× |         **0.92×** |               **0.96×** |
| or_high_high       | TOP_10  |   1,917 |   1,843 |           0.96× |       4,107 |           0.84× |         **0.80×** |               **1.79×** |
| or_high_high       | TOP_100 |   1,390 |   1,301 |           0.94× |       2,324 |           0.79× |         **0.74×** |               **1.33×** |
| or_high_high       | COUNT   |   3,503 |   3,499 |           1.00× |       4,606 |           0.84× |         **0.84×** |               **1.11×** |
| or_high_low        | TOP_10  |  22,227 |  22,611 |           1.02× |      40,247 |           0.66× |         **0.67×** |               **1.19×** |
| or_high_low        | TOP_100 |  12,986 |  13,319 |           1.03× |      17,559 |           0.84× |         **0.86×** |               **1.14×** |
| or_high_low        | COUNT   |  42,658 |  43,833 |           1.03× |      71,030 |           4.61× |         **4.73×** |               **7.67×** |
| or_high_med        | TOP_10  |   5,165 |   5,225 |           1.01× |       8,446 |           0.67× |         **0.68×** |               **1.10×** |
| or_high_med        | TOP_100 |   3,670 |   3,723 |           1.01× |       5,333 |           0.70× |         **0.71×** |               **1.02×** |
| or_high_med        | COUNT   |   5,781 |   5,872 |           1.02× |       7,497 |           0.88× |         **0.90×** |               **1.15×** |
| prefix3            | TOP_10  | 114,889 | 113,760 |           0.99× |       2,658 |           0.96× |         **0.95×** |               **0.02×** |
| prefix3            | TOP_100 | 103,969 | 102,151 |           0.98× |       2,615 |           0.96× |         **0.94×** |               **0.02×** |
| prefix3            | COUNT   |   7,089 |   7,232 |           1.02× |       7,935 |           0.75× |         **0.76×** |               **0.84×** |
| regex              | TOP_10  | 132,840 | 137,599 |           1.04× |       3,699 |           1.08× |         **1.12×** |               **0.03×** |
| regex              | TOP_100 | 121,113 | 127,463 |           1.05× |       3,685 |           1.12× |         **1.18×** |               **0.03×** |
| regex              | COUNT   |   7,982 |   8,288 |           1.04× |       8,787 |           0.61× |         **0.64×** |               **0.68×** |
| wildcard           | TOP_10  |  31,384 |  32,442 |           1.03× |       1,437 |           1.09× |         **1.13×** |               **0.05×** |
| wildcard           | TOP_100 |  29,993 |  31,631 |           1.05× |       1,406 |           1.09× |         **1.15×** |               **0.05×** |
| wildcard           | COUNT   |   4,596 |   4,728 |           1.03× |       5,314 |           0.73× |         **0.75×** |               **0.84×** |
| wildcard_scan      | TOP_10  |   9,089 |   9,479 |           1.04× |       5,016 |           5.11× |         **5.33×** |               **2.82×** |
| wildcard_scan      | TOP_100 |   8,974 |   9,474 |           1.06× |       4,914 |           4.99× |         **5.27×** |               **2.73×** |
| wildcard_scan      | COUNT   |   6,242 |   6,451 |           1.03× |       6,622 |           3.59× |         **3.71×** |               **3.80×** |

**Phrases of common words become term lookups.** On the default build,
pairs lift high_phrase top-10 44× (1.95× → 86.7× Luxir), top-100 58×, and
its exact counts 2,168×: a covered phrase's count is the pair's document
frequency. med_phrase top-10 rises 21× (34.5× Luxir) and low_phrase 2.9×
(5.0× Luxir); their counts 6.5× and 2.4×, since fewer of their phrases are
covered. The other families move within the campaign's noise (−6% to +5%).

**RGB + pairs combines both effects except for ranked phrases.** Ranked
AND/OR families match the earlier RGB campaign (and_high_high TOP_10 1.92×
Luxir, or_high_high 1.79×, and_high_low 1.27×, or_high_low 1.19×,
or_high_med 1.10×, and_high_med 0.98×) and phrase counts gain as on the
default build, but high_phrase top-10 stays at 1.05× Luxir. On a reordered
field the phrase query opted into physical traversal for ranked streams too,
so the pair's postings were walked in full under stable-ID ties instead of
through the term's ranked plan; see the follow-up below. Pattern top-k keeps
the RGB limitation of the [RGB campaign](../rgb-2026-09-29/README.md).

## Follow-up: ranked pairs on reordered fields

`PhraseQuery::physical_text_field` now declines ranked streams for a phrase
answered by a pair, so the pair runs the term's logical ranked plan on a
reordered field, as a term query does; complete streams (counts) still go
physical. `common_word_pairs_answer_two_word_phrases_like_positions` pins the
routing on its RGB case (it fails without the change).

Single-thread top-10, summed over the 126 audited phrase queries (x86, the
campaign binary against the fixed tree, same builds):

| Build       | Binary   | high_phrase | med_phrase | low_phrase |
| ----------- | -------- | ----------: | ---------: | ---------: |
| RGB + pairs | campaign |    450.3 ms |   196.8 ms |   129.5 ms |
| RGB + pairs | fixed    |      2.8 ms |    17.8 ms |   106.1 ms |
| pairs       | campaign |      2.4 ms |    14.6 ms |   146.7 ms |
| pairs       | fixed    |      2.3 ms |    15.3 ms |   146.6 ms |

On the RGB + pairs build high_phrase top-10 is now 160× faster and level with
the pairs build; the pairs build is unchanged.
