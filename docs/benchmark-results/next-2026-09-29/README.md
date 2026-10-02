# Dictionary entry offsets, rarer-block skips and per-candidate bounds — September 29

Two x86 GCP campaigns (32-vCPU n2 benchmark host: 30 server CPUs, 2 driver
CPUs, 32 clients, network-isolated), all 19 Searchbench families, zero
errors:

1. **59,218,265 requests**, phases b1 c1 l1 c2 b2:

   - **before:** the September 28 campaign binary;
   - **code:** the final tree;
   - **Luxir 0.1.0**, pinned.

   Both Summa binaries serve one fresh 30-worker build of the corpus with the
   default codec (`RoundedBitmap`), built by the new binary.

2. **111,025,430 requests**, phases b1 c1 j1 l1 j2 c2 b2 on the same index,
   adding **jcc:** the final tree built with
   `-C llvm-args=-x86-branches-within-32B-boundaries` (see
   [below](#frequent-term-counts-and-the-jcc-erratum)).

[Results](campaign-results.json) and [validator](summarize.py);
[JCC results](jcc-campaign-results.json) and [validator](summarize-jcc.py).
Every audit matches the September 25 baseline on counts, plans, ranked score
sequences and score groups (the fresh build assigns different document IDs),
the binaries' audits are identical, and every Summa HTTP response matches its
audit.

## What changed

- **[Entry offsets for cached dictionary blocks](../../dictionary-entry-offsets.md):**
  prefix scans of a shared-cache block's second visit jump over rejected
  terms instead of parsing their `TermInfo`.
- **Ranked conjunctions pass losing rarer blocks:** when even the common
  term's list maximum cannot lift a rarer block to the threshold, it is
  skipped and the common cursor moves without decoding or bounding the blocks
  in between.
- **Disjunction windows bound dense non-essential terms per candidate** when
  the window's drivers are at least 64 times sparser.

## Results (campaign 1; QPS, means of phase medians)

| Family             | Op      |  Before |    Code | Code / before | Before / Luxir | **Code / Luxir** |
| ------------------ | ------- | ------: | ------: | ------------: | -------------: | ---------------: |
| and_high_high      | TOP_10  |   2,558 |   2,676 |         1.05× |          1.16× |        **1.22×** |
| and_high_high      | TOP_100 |   1,508 |   1,526 |         1.01× |          0.90× |        **0.91×** |
| and_high_high      | COUNT   |   3,430 |   3,184 |         0.93× |          0.85× |        **0.79×** |
| and_high_low       | TOP_10  |  25,016 |  31,000 |         1.24× |          0.61× |        **0.75×** |
| and_high_low       | TOP_100 |  23,163 |  24,166 |         1.04× |          0.82× |        **0.86×** |
| and_high_low       | COUNT   |  44,877 |  43,578 |         0.97× |          1.04× |        **1.01×** |
| and_high_med       | TOP_10  |   4,730 |   5,141 |         1.09× |          0.57× |        **0.62×** |
| and_high_med       | TOP_100 |   3,523 |   3,669 |         1.04× |          0.61× |        **0.63×** |
| and_high_med       | COUNT   |   5,840 |   5,791 |         0.99× |          0.98× |        **0.97×** |
| high_phrase        | TOP_10  |   1,920 |   1,912 |         1.00× |          1.74× |        **1.74×** |
| high_phrase        | TOP_100 |     477 |     481 |         1.01× |          1.60× |        **1.61×** |
| high_phrase        | COUNT   |      55 |      55 |         1.00× |          0.92× |        **0.92×** |
| high_sloppy_phrase | TOP_10  |   6,374 |   6,497 |         1.02× |         17.44× |       **17.78×** |
| high_sloppy_phrase | TOP_100 |   2,438 |   2,495 |         1.02× |          7.74× |        **7.92×** |
| high_sloppy_phrase | COUNT   |     174 |     173 |         0.99× |          1.16× |        **1.15×** |
| high_term          | TOP_10  |  68,013 |  67,961 |         1.00× |          1.22× |        **1.22×** |
| high_term          | TOP_100 |  23,965 |  24,086 |         1.01× |          1.57× |        **1.58×** |
| high_term          | COUNT   |  99,781 | 101,211 |         1.01× |          0.88× |        **0.90×** |
| low_phrase         | TOP_10  |   2,174 |   2,204 |         1.01× |          1.57× |        **1.59×** |
| low_phrase         | TOP_100 |     850 |     859 |         1.01× |          1.16× |        **1.18×** |
| low_phrase         | COUNT   |     341 |     336 |         0.99× |          0.99× |        **0.97×** |
| low_sloppy_phrase  | TOP_10  |     852 |     863 |         1.01× |          2.49× |        **2.53×** |
| low_sloppy_phrase  | TOP_100 |     493 |     496 |         1.01× |          1.65× |        **1.65×** |
| low_sloppy_phrase  | COUNT   |     343 |     338 |         0.98× |          1.19× |        **1.17×** |
| low_term           | TOP_10  |  90,721 |  89,618 |         0.99× |          0.88× |        **0.87×** |
| low_term           | TOP_100 |  38,979 |  38,620 |         0.99× |          1.20× |        **1.19×** |
| low_term           | COUNT   |  98,092 |  97,216 |         0.99× |          0.89× |        **0.88×** |
| med_phrase         | TOP_10  |   1,679 |   1,693 |         1.01× |          1.75× |        **1.76×** |
| med_phrase         | TOP_100 |     551 |     560 |         1.02× |          1.16× |        **1.18×** |
| med_phrase         | COUNT   |     150 |     150 |         1.00× |          0.99× |        **1.00×** |
| med_sloppy_phrase  | TOP_10  |   1,224 |   1,232 |         1.01× |          3.54× |        **3.57×** |
| med_sloppy_phrase  | TOP_100 |     750 |     756 |         1.01× |          2.73× |        **2.75×** |
| med_sloppy_phrase  | COUNT   |     260 |     260 |         1.00× |          1.22× |        **1.22×** |
| med_term           | TOP_10  |  91,164 |  90,900 |         1.00× |          1.10× |        **1.10×** |
| med_term           | TOP_100 |  30,469 |  31,468 |         1.03× |          1.48× |        **1.52×** |
| med_term           | COUNT   | 100,623 | 100,406 |         1.00× |          0.93× |        **0.93×** |
| or_high_high       | TOP_10  |   1,922 |   1,902 |         0.99× |          0.85× |        **0.84×** |
| or_high_high       | TOP_100 |   1,390 |   1,367 |         0.98× |          0.81× |        **0.80×** |
| or_high_high       | COUNT   |   3,486 |   3,245 |         0.93× |          0.85× |        **0.79×** |
| or_high_low        | TOP_10  |  20,218 |  21,488 |         1.06× |          0.67× |        **0.71×** |
| or_high_low        | TOP_100 |  12,220 |  12,592 |         1.03× |          0.85× |        **0.88×** |
| or_high_low        | COUNT   |  39,223 |  39,296 |         1.00× |          4.42× |        **4.43×** |
| or_high_med        | TOP_10  |   5,181 |   5,129 |         0.99× |          0.70× |        **0.69×** |
| or_high_med        | TOP_100 |   3,691 |   3,654 |         0.99× |          0.71× |        **0.70×** |
| or_high_med        | COUNT   |   5,738 |   5,557 |         0.97× |          0.90× |        **0.88×** |
| prefix3            | TOP_10  |  81,334 |  85,111 |         1.05× |          0.76× |        **0.80×** |
| prefix3            | TOP_100 |  73,648 |  77,678 |         1.05× |          0.74× |        **0.78×** |
| prefix3            | COUNT   |   7,123 |   7,198 |         1.01× |          0.74× |        **0.75×** |
| regex              | TOP_10  |  95,431 |  95,738 |         1.00× |          0.92× |        **0.93×** |
| regex              | TOP_100 |  88,881 |  90,192 |         1.01× |          0.92× |        **0.93×** |
| regex              | COUNT   |   7,724 |   7,966 |         1.03× |          0.60× |        **0.62×** |
| wildcard           | TOP_10  |  15,698 |  30,893 |         1.97× |          0.56× |        **1.11×** |
| wildcard           | TOP_100 |  15,609 |  29,787 |         1.91× |          0.57× |        **1.10×** |
| wildcard           | COUNT   |   4,040 |   4,437 |         1.10× |          0.63× |        **0.70×** |
| wildcard_scan      | TOP_10  |   4,829 |   7,589 |         1.57× |          2.77× |        **4.35×** |
| wildcard_scan      | TOP_100 |   4,800 |   7,479 |         1.56× |          2.76× |        **4.29×** |
| wildcard_scan      | COUNT   |   3,866 |   5,713 |         1.48× |          2.32× |        **3.42×** |

Wildcard top-k now runs ahead of Luxir (0.56× → 1.11×), wildcard_scan
reaches 4.3×, and_high_low TOP_10 gains 24% (0.61× → 0.75×), prefix3 top-k
5%, or_high_low TOP_10 6%. The second campaign reproduces every code / before
ratio within a few percent (and_high_low TOP_10 1.30×, wildcard TOP_10 1.99×,
wildcard_scan 1.86×).

**Luxir ratios of the cheapest families are uncertain.** The second campaign
ran 10–15% faster across the board than the first (Luxir and both Summa
binaries alike; up to 50% for families above 50,000 QPS), and each campaign
measures Luxir in one phase. Code / Luxir moved between the campaigns by up to
±20% for low_term, med_term, high_term COUNT, regex top-k and prefix3 top-k,
but by at most a few percent for the conjunction, disjunction and phrase
families.

## Frequent-term counts and the JCC erratum

Campaign 1 found frequent-term exact counts 3–7% slower with the new binary
(and_high_high and or_high_high COUNT 0.93×, or_high_med COUNT 0.97×) in both
mirrored phase pairs, although none of their code changed. Count-probe
builds of the two trees count 1.5–2.6% faster with the new code on one
thread, and the campaign binaries differ by 2% on an 8-CPU VM. Profiles under 32 clients put the extra time in
`BlockPostingIterator::fill_doc_window` (62% → 65% of samples): in the new
binary the compare and branch that close its bit-setting loop are fused
across a 32-byte boundary (`cmp` at `…8fd`, `jne` at `…900`), the condition
of the Skylake-family JCC erratum, so on these Cascade Lake hosts the loop is
decoded by the legacy decoders, which hyperthread siblings share. The old
binary holds the same pair inside one 32-byte window.

Building with `-x86-branches-within-32B-boundaries` recovers the counts
(and_high_high COUNT 1.06× and or_high_high COUNT 1.07× over the unpadded
build) but costs ranked families 2–4% (and_high_low and or_high_high TOP_10
0.97×, high_term TOP_100 0.96×): the geometric mean over all 57 cells is
0.996×. It is not adopted. Layout shifts of this size remain a hazard for
single-binary comparisons on these hosts; same-binary control arms stay the
rule.

## Remaining gaps (code / Luxir, TOP_10)

and_high_med 0.61×, or_high_med 0.69×, or_high_low 0.62–0.71×, and_high_low
0.70–0.75×, or_high_high 0.85×; exact counts: regex 0.62×, wildcard 0.70×,
prefix3 0.75×, and_high_high and or_high_high 0.79×.
