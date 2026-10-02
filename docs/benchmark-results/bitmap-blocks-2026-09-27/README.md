# Frequent-term counts, regex term checks and bitmap posting blocks — September 27

x86 GCP campaign (32-vCPU n2 benchmark host: 30 server CPUs, 2 driver CPUs,
32 clients, network-isolated), all 19 Searchbench families, 77,618,532
requests, zero errors. One binary (the final tree) serves the September 24
index (**control**, stock `Rounded` postings) and a `RoundedBitmap` build of
the same corpus (**bitmap**, [design](../../bitmap-posting-blocks.md));
Luxir 0.1.0 is pinned. Mirrored phases c1 m1 l1 m2 c2.
[Results](campaign-results.json), [validator](summarize.py),
[single-thread probes](probes.json).

## What changed

On the default path (`Rounded`, no configuration change):

- Regex matching runs a Unicode-mode byte automaton, so scanned terms are no
  longer UTF-8 validated one by one, after a suffix-literal prefilter.
- Posting unions sort collected IDs only below one posting per 1,024
  documents; wider unions use the segment bitset.
- Lists with one document in eight to one in two set membership bits
  branch-free; denser and sparser lists keep their kernels.

Opt-in (`PostingCodec::RoundedBitmap`, metadata format 10): bitmap posting
blocks, which membership windows copy and probes and ranked conjunctions
bit-test.

## Summa / Luxir

`Before` is the previous campaign's pool dispatch
([dispatch-results.json](../dict-scan-2026-09-27/dispatch-results.json)) on
the same September 24 index. QPS are means of phase medians.

| Family             | Op      | Before / Luxir | Before QPS | Control QPS | **Control / Luxir** | Bitmap QPS | **Bitmap / Luxir** |
| ------------------ | ------- | -------------: | ---------: | ----------: | ------------------: | ---------: | -----------------: |
| and_high_high      | TOP_10  |          1.00× |       2309 |        2286 |               0.99× |       2402 |              1.04× |
| and_high_high      | TOP_100 |          0.81× |       1431 |        1414 |               0.80× |       1482 |              0.84× |
| and_high_high      | COUNT   |          0.56× |       2385 |        2384 |               0.56× |       3627 |              0.85× |
| and_high_low       | TOP_10  |          0.56× |      30586 |       29897 |               0.58× |      25640 |              0.50× |
| and_high_low       | TOP_100 |          0.80× |      28689 |       28132 |               0.82× |      29213 |              0.85× |
| and_high_low       | COUNT   |          0.92× |      51231 |       50344 |               0.95× |      60598 |              1.14× |
| and_high_med       | TOP_10  |          0.48× |       4382 |        4218 |               0.47× |       4403 |              0.49× |
| and_high_med       | TOP_100 |          0.55× |       3434 |        3303 |               0.53× |       3414 |              0.55× |
| and_high_med       | COUNT   |          0.80× |       5028 |        5067 |               0.80× |       6436 |              1.02× |
| high_phrase        | TOP_10  |          1.86× |       2187 |        2168 |               1.86× |       2212 |              1.89× |
| high_phrase        | TOP_100 |          1.59× |        525 |         525 |               1.60× |        506 |              1.54× |
| high_phrase        | COUNT   |          1.01× |         62 |          62 |               0.97× |         63 |              0.99× |
| high_sloppy_phrase | TOP_10  |         20.47× |       7657 |        7541 |              19.95× |       7399 |             19.57× |
| high_sloppy_phrase | TOP_100 |          8.50× |       2844 |        2825 |               8.50× |       2793 |              8.41× |
| high_sloppy_phrase | COUNT   |          1.22× |        189 |         190 |               1.26× |        189 |              1.26× |
| high_term          | TOP_10  |          1.26× |      84870 |       80890 |               1.23× |      80509 |              1.22× |
| high_term          | TOP_100 |          1.78× |      29076 |       28585 |               1.76× |      28713 |              1.76× |
| high_term          | COUNT   |          1.02× |     152151 |      149200 |               1.00× |     149888 |              1.01× |
| low_phrase         | TOP_10  |          1.70× |       2441 |        2438 |               1.70× |       2580 |              1.80× |
| low_phrase         | TOP_100 |          1.17× |        920 |         925 |               1.18× |        932 |              1.19× |
| low_phrase         | COUNT   |          1.00× |        368 |         370 |               0.97× |        371 |              0.98× |
| low_sloppy_phrase  | TOP_10  |          2.55× |        950 |         933 |               2.51× |        944 |              2.54× |
| low_sloppy_phrase  | TOP_100 |          1.65× |        524 |         520 |               1.69× |        527 |              1.71× |
| low_sloppy_phrase  | COUNT   |          1.19× |        362 |         365 |               1.20× |        367 |              1.21× |
| low_term           | TOP_10  |          1.04× |     129300 |      126530 |               1.04× |     128298 |              1.05× |
| low_term           | TOP_100 |          1.24× |      45491 |       44935 |               1.25× |      48688 |              1.35× |
| low_term           | COUNT   |          1.00× |     149617 |      148352 |               0.99× |     148811 |              0.99× |
| med_phrase         | TOP_10  |          1.76× |       1780 |        1794 |               1.76× |       1670 |              1.64× |
| med_phrase         | TOP_100 |          1.12× |        581 |         587 |               1.14× |        583 |              1.13× |
| med_phrase         | COUNT   |          1.03× |        166 |         167 |               1.04× |        166 |              1.03× |
| med_sloppy_phrase  | TOP_10  |          3.69× |       1308 |        1277 |               3.53× |       1279 |              3.53× |
| med_sloppy_phrase  | TOP_100 |          2.77× |        821 |         799 |               2.75× |        816 |              2.81× |
| med_sloppy_phrase  | COUNT   |          1.25× |        283 |         283 |               1.26× |        281 |              1.25× |
| med_term           | TOP_10  |          1.28× |     123501 |      121525 |               1.27× |     127735 |              1.33× |
| med_term           | TOP_100 |          1.63× |      36588 |       35990 |               1.61× |      38944 |              1.75× |
| med_term           | COUNT   |          1.01× |     151075 |      149154 |               0.99× |     149633 |              0.99× |
| or_high_high       | TOP_10  |          0.88× |       2084 |        2026 |               0.86× |       2029 |              0.86× |
| or_high_high       | TOP_100 |          0.83× |       1501 |        1473 |               0.82× |       1473 |              0.82× |
| or_high_high       | COUNT   |          0.54× |       2351 |        2335 |               0.54× |       3682 |              0.85× |
| or_high_low        | TOP_10  |          0.52× |      20896 |       20196 |               0.53× |      21490 |              0.56× |
| or_high_low        | TOP_100 |          0.85× |      14081 |       13765 |               0.84× |      14799 |              0.91× |
| or_high_low        | COUNT   |          4.74× |      44574 |       44049 |               4.71× |      51472 |              5.50× |
| or_high_med        | TOP_10  |          0.69× |       5555 |        5418 |               0.68× |       5609 |              0.70× |
| or_high_med        | TOP_100 |          0.72× |       4028 |        3930 |               0.71× |       4060 |              0.73× |
| or_high_med        | COUNT   |          0.72× |       4794 |        4750 |               0.71× |       6127 |              0.92× |
| prefix3            | TOP_10  |          0.97× |     124675 |      124646 |               0.97× |     121771 |              0.95× |
| prefix3            | TOP_100 |          0.99× |     111790 |      112792 |               1.00× |     108858 |              0.96× |
| prefix3            | COUNT   |          0.76× |       7613 |        7509 |               0.75× |       8111 |              0.81× |
| regex              | TOP_10  |          1.14× |     152006 |      147866 |               1.12× |     146501 |              1.11× |
| regex              | TOP_100 |          1.19× |     135678 |      135154 |               1.19× |     131965 |              1.16× |
| regex              | COUNT   |          0.37× |       4988 |        7210 |               0.55× |       8975 |              0.68× |
| wildcard           | TOP_10  |          0.58× |      17458 |       17342 |               0.58× |      17253 |              0.57× |
| wildcard           | TOP_100 |          0.59× |      17319 |       16974 |               0.59× |      17021 |              0.59× |
| wildcard           | COUNT   |          0.59× |       3992 |        3926 |               0.58× |       4384 |              0.65× |
| wildcard_scan      | TOP_10  |          2.89× |       5345 |        5226 |               2.84× |       5319 |              2.89× |
| wildcard_scan      | TOP_100 |          2.90× |       5336 |        5232 |               2.91× |       5275 |              2.93× |
| wildcard_scan      | COUNT   |          2.02× |       3567 |        4194 |               2.38× |       4315 |              2.45× |

Highlights:

- **Default path:** regex COUNT 4,988 → 7,210 QPS (0.37× → 0.55× Luxir)
  and wildcard_scan COUNT +18%. Four ranked cells read 3–5% lower than in
  the previous campaign (high_term TOP_10, and_high_med TOP_10/TOP_100,
  or_high_low TOP_10); the same-index single-thread comparison below puts
  those families within ±1% of the session start, so this is run-to-run
  variation between the two campaigns. All other cells are within ±3%.
- **Bitmap build:** counts over frequent terms rise to 0.85× Luxir for
  and/or_high_high (from 0.54–0.56×), above Luxir for and_high_low (1.14×)
  and and_high_med (1.02×), 0.92× for or_high_med and 0.68× for regex.
- **Bitmap costs:** med_phrase TOP_10 −7%, prefix3 TOP_10/TOP_100 −2 to −3%,
  and and_high_low TOP_10 −14%. The last is not the codec (below).

## Build effects versus codec effects

Parallel indexing assigns different document IDs in every build, so a
bitmap-versus-control difference mixes codec and build. Single-thread TOP_10
sums on the benchmark host, same probe binary, three indexes:

| Family                    | Sept 24 index | Rounded, fresh build | Bitmap, fresh build |
| ------------------------- | ------------: | -------------------: | ------------------: |
| and_high_low (46 queries) |       17.6 ms |              22.3 ms |             22.5 ms |
| med_phrase (46 queries)   |        429 ms |               420 ms |              455 ms |

The and_high_low TOP_10 loss comes from the fresh build (27% slower with
either codec), not from bitmap blocks. Fresh builds (8 indexing workers,
current tree) also differ in file sizes from the September 24 build (30
workers): `.post` 7.17 vs 8.70 GB, `.pos` 2.88 vs 3.54 GB. That is an open
lead. The med_phrase loss (+8%) is the codec: the positions path expands
25–50%-dense bitmap blocks, which decode slower than 8-bit deltas.

The [controlled harness](probes.json) (`codec_top10_controlled`) removes
build effects for term, conjunction and disjunction families (same postings,
both encodings, bit-identical results): all within −2.9%…+2.8%.

## Default path against the session start

Single-thread medians on the 8-vCPU host, A B A B, same September 24 index,
the session-start tree built in its own target directory:

- COUNT (156 queries): regex 71.9 → 34.4 ms (2.09×), and_high_high −5.5%,
  or_high_high −4.5%, wildcard −3.7%, and_high_low −3.3%; prefix3 unchanged
  (±1% over 200 iterations).
- TOP_10 (677 queries, identical result digests): regex 80.2 → 28.1 ms,
  and_high_high −2.6%, prefix3 −5%, and_high_low +2.0%, wildcard +2.7%,
  wildcard_scan +2.0%, all others within ±1%. Sub-10 µs regex queries swing
  ±3 µs with search-pool scheduling (their order reversed under `perf`).

## Validation and reproduction

- `summarize.py <artifacts> <folder with final.tar.gz>` checks both index
  inventories before and after (unchanged), the binary and archive hashes,
  the control audit against the September 25 baseline (identical), the
  bitmap audit (every count and plan, every ranked score sequence, and the
  IDs of every score group but the last, cut by the top-100 boundary; 673
  queries), every Summa HTTP response against the audit of the index it
  served, and Luxir coverage.
- Build: archive `17adcc77…7011`, binary `5c6db160…1a2a`; raw captures
  `1b5d1921…786a`, retained privately. A first run was stopped after its
  first phase when it exposed a directory-search regression in membership
  probes, since fixed.
- `scripts/check_search.py check` passes all five stages with 2,144 native
  tests (25 ignored); the WASM release build and 41 JavaScript tests pass.
- The temporary scratch disk was deleted; both VMs were stopped.
