# Exact pattern counts — October 2

This follow-up targets the pattern COUNT gaps from the
[September 30 campaign](../ties-2026-09-30/README.md). It preserves the
index, codecs, expansion budgets, ranking and query semantics.

## Implementation and cost

For an expansion dominated by one term, count its document frequency and
add only the deduplicated tail IDs absent from that term. The shared union
owner uses the existing materializer and canonical batched membership probes.
The gate requires the largest posting list to contain at least 64 times the
sum of the other lists' document frequencies. A single term uses its metadata
count; balanced expansions use the exact cardinality already computed by the
existing materializer. See the [design](../../regex-query.md).

This path applies only to count-accepting collectors without scores, positions
or deleted rows. Membership-preserving single-clause Boolean wrappers forward
it. Filtered/excluded compositions use their normal execution; chunked fields
remain rejected. Expansion admission runs first. Physical IDs preserve
cardinality on document-level RGB maps. Scratch remains bounded by the existing
union materializer, one posting cursor and a 128-document probe batch.

## Native measurements

Both arms use Rust 1.98.1, release mode and `-C target-cpu=native` on each host.
A benchmark-only `diagnose` command parses each query once and measures 200
iterations after five warmups, on one OS thread. Each phase restarts the
process. Phases A1 B1 B2 A2; report the ratio of the means of phase sums of
per-query median latency. Thread CPU time includes warmup iterations and is
reported separately. These ratios are not concurrent HTTP throughput.

The 59 admitted pattern expressions comprise two prefixes, four regexes,
eleven wildcards and 42 broad wildcard scans. Every count and order-sensitive
top-100 ID/score digest must match across all four phases. The [native probe](probe.rs.txt)
retains the same parser, collector and posting readers as the adapter.

Apple M4, 1M documents, existing bitmap, RGB and older rounded-codec indexes:

| Index   | Prefix COUNT | Regex COUNT | Wildcard COUNT | Broad wildcard COUNT |
| ------- | -----------: | ----------: | -------------: | -------------------: |
| Bitmap  |       1.074× |      1.011× |         2.049× |               1.067× |
| RGB     |       1.083× |      1.025× |         2.037× |               1.082× |
| Rounded |       1.080× |      1.023× |         2.390× |               1.080× |

Top-100 controls range from 0.985× to 1.017×. All counts and digests match.
ARM COUNT peak process RSS is 212–222 MiB before and 210–218 MiB after
(across the three indexes); this includes mapped pages, not just heap scratch.
The strongest individual RGB wildcard improvement is `tw*o`, 244.2 → 23.1 µs
(10.55×). The broad regex `[jkqxz][a-z]*ess` remains dictionary-bound at
1,514.8 → 1,501.7 µs (1.01×).

[ARM results](native-arm.json) retain per-phase latency, CPU time and whole
process peak RSS, including mapped pages. [Native validator](summarize_screen.py).

On x86 (GCP Cascade Lake, 10M documents, pinned to CPU 2), the same
unchanged default + pairs and RGB + pairs indexes as September 30 give:

| Index           | Prefix COUNT | Regex COUNT | Wildcard COUNT | Broad wildcard COUNT |
| --------------- | -----------: | ----------: | -------------: | -------------------: |
| Default + pairs |       1.411× |      1.124× |         2.233× |               1.347× |
| RGB + pairs     |       1.450× |      1.128× |         2.408× |               1.359× |

Top-100 controls range from 1.001× to 1.045×. All 59 counts and rank digests
match in all phases on both indexes. Before/after index inventories match
byte-for-byte, and match the retained September 30 index inventories.

## Concurrent HTTP comparison

GCP n2, 32 Cascade Lake vCPUs: 30 server CPUs and two driver CPUs, 32 clients,
query cache off, isolated loopback. The same 10M-document RGB + pairs index is
served by both Summa binaries; Luxir 0.1.0 is pinned. Phases A1 B1 L1 L2 B2 A2,
40 seconds of session warmup, then three 3-second repetitions with one second
of warmup per cell. Eight families × TOP_10/TOP_100/COUNT: 24 cells, not a new
57-cell sweep. Measured requests total **56,427,366**, with zero errors.

This is the retained **shared-input** comparison, not a claim of equal
cross-engine counts or ranking. All 229 admitted expressions retain identical
Summa counts and top-10/100 IDs against the September 30 audit of this exact
index, in all four Summa phases. The same 102 explicit budget exclusions from
these families are retained. Native count/rank digests and index byte hashes
also pass; the RGB index is hashed again after the HTTP campaign.

[Results](campaign-results.json), [validator](summarize.py),
[x86 native results](native-x86.json). Throughput is the mean of the two
phase medians; raw phase and repetition ranges remain in the results.

| Family        | Operation | Before QPS | After QPS | Luxir QPS | After / before | After / Luxir |
| ------------- | --------- | ---------: | --------: | --------: | -------------: | ------------: |
| and_high_med  | TOP_10    |      8,572 |     8,540 |     9,121 |         0.996× |        0.936× |
| and_high_med  | TOP_100   |      5,749 |     5,779 |     6,242 |         1.005× |        0.926× |
| and_high_med  | COUNT     |      8,167 |     8,199 |     6,315 |         1.004× |        1.298× |
| high_phrase   | TOP_10    |     88,121 |    88,543 |     1,164 |         1.005× |       76.059× |
| high_phrase   | TOP_100   |     27,731 |    27,982 |       323 |         1.009× |       86.645× |
| high_phrase   | COUNT     |    134,536 |   135,726 |        64 |         1.009× |     2130.287× |
| high_term     | TOP_10    |     68,536 |    69,427 |    64,814 |         1.013× |        1.071× |
| high_term     | TOP_100   |     24,263 |    24,122 |    16,349 |         0.994× |        1.475× |
| high_term     | COUNT     |    147,208 |   148,686 |   148,938 |         1.010× |        0.998× |
| or_high_med   | TOP_10    |      8,965 |     8,895 |     8,036 |         0.992× |        1.107× |
| or_high_med   | TOP_100   |      5,682 |     5,640 |     5,550 |         0.993× |        1.016× |
| or_high_med   | COUNT     |      7,732 |     7,738 |     6,662 |         1.001× |        1.162× |
| prefix3       | TOP_10    |    109,621 |   111,707 |   126,301 |         1.019× |        0.884× |
| prefix3       | TOP_100   |     94,244 |    95,074 |   113,063 |         1.009× |        0.841× |
| prefix3       | COUNT     |      8,527 |    10,959 |    10,042 |         1.285× |        1.091× |
| regex         | TOP_10    |    144,430 |   145,267 |   132,283 |         1.006× |        1.098× |
| regex         | TOP_100   |    126,881 |   127,017 |   114,694 |         1.001× |        1.107× |
| regex         | COUNT     |      9,275 |    13,544 |    13,746 |         1.460× |        0.985× |
| wildcard      | TOP_10    |     34,595 |    33,822 |    30,121 |         0.978× |        1.123× |
| wildcard      | TOP_100   |     32,359 |    32,026 |    29,042 |         0.990× |        1.103× |
| wildcard      | COUNT     |      5,575 |    11,987 |     6,720 |         2.150× |        1.784× |
| wildcard_scan | TOP_10    |      9,763 |     8,965 |     1,846 |         0.918× |        4.856× |
| wildcard_scan | TOP_100   |      9,614 |     8,852 |     1,838 |         0.921× |        4.817× |
| wildcard_scan | COUNT     |      7,019 |     7,971 |     1,770 |         1.136× |        4.504× |

Wildcard COUNT improves **2.150×** and prefix COUNT **1.285×**, moving above
Luxir to **1.784×** and **1.091×**. Regex COUNT improves **1.460×**, reaching
**0.985×** Luxir. Broad wildcard COUNT improves **1.136×** (4.504× Luxir).
Measured server CPU per count request falls from 5,091 → 2,229 µs for wildcard,
3,210 → 2,454 µs for prefix, 2,957 → 1,980 µs for regex, and
3,983 → 3,464 µs for broad wildcard.

**Tradeoff:** broad wildcard TOP_10/TOP_100 regresses to **0.918×/0.921×**
baseline, with server CPU per request rising about 9%. It remains
4.856×/4.817× Luxir. Both candidate passes reproduce this, despite unchanged
ranking code and the single-core controls showing no slowdown. The cause is
unresolved; do not dismiss it as timing noise or infer a code-generation cause
without profiling. Other ranked cells range from 0.978× to 1.019× baseline.
`and_high_med` top-k remains at 0.926–0.936× Luxir; prefix top-k remains at
0.841–0.884×. These remain separate optimization targets.

## Memory

| Engine | Peak process RSS, MiB | Peak anonymous RSS, MiB |
| ------ | --------------------: | ----------------------: |
| before |               3,094.1 |                   386.9 |
| after  |               3,094.9 |                   387.7 |
| luxir  |               1,032.9 |                    78.0 |

Summa's anonymous peak rises by 0.84 MiB (0.22%). All index-file mapping RSS
categories in the end-of-phase `smaps` snapshots are identical before/after;
there is no new resident cache. Process RSS includes file-backed pages and
must not be reported as heap. The detailed mapping groups are retained in the
results. Corpus files are unchanged after the entire campaign.

## Rejected experiments

Directly accumulating encoded posting windows into the union bitmap helped
RGB on x86, but default-index single-core wildcard counts fell to 0.936×
baseline ([rejected results](rejected-direct-union.json)). That
posting-reader change is not retained. A less selective scalar probing gate
also regressed a finite regex on ARM. The retained implementation uses the
canonical batch reader with the conservative 64:1 gate. No codec, cache or
index-building default changes follow from these experiments.

## Validation

`python3 scripts/check_search.py check` passes formatting, strict Clippy,
2,159 core/server/broker/tool tests (25 ignored), native without default
features, and the broker check. The WASM build, `npm ci`, and all 41 WASM tests pass. Documentation
links and `git diff --check` pass. The extended lifecycle/RPC harness is not
run: this change does not touch lifecycle or RPC code.

The new behavior regressions cover five posting codecs, sparse/dense blocks,
vector/bitmap tail unions, overlapping and absent matches, inline postings,
RGB, deleted rows, Boolean filters/exclusions, expansion budget failures, and
collectors that accept a zero-count probe but reject the actual cardinality.
The last case was reproduced failing before fixing the new collector shortcut.
After the full harness passed, the integration fixture was enlarged to 20,000
documents to exercise multiple 128-ID probe batches. Its feeder now observes
bounded writer backpressure; the enlarged regression passes separately. No
production code changed after the measured build.

## Build provenance

Both benchmark arms use the initial workspace snapshot with the same
out-of-tree diagnostic command. The candidate adds only the exact-count
implementation; the final source also contains test-only lint and coverage
improvements.
All eight [production file hashes](production-source-sha256.json) match both
the final workspace and the source overlay used for the x86 build.

| Artifact                                 | SHA-256                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| Baseline source archive, including probe | `2f931ad004ca18c0bf9f0274a6ee1657cc3950aa06c1731af811b7a6edc3931d` |
| Candidate source overlay                 | `4ac6f752733d89dff496a133aeadf8d8046e154a9c1da6bd759cc10d97969230` |
| ARM baseline executable                  | `e08a0c6659e7318202413f88aa2a4f281bf5f6a4f49a0eb3d36dacc3f77f616b` |
| ARM candidate executable                 | `1b68abdd4f9ca32cc7cf3ff26d8cc1b2b5d43ce9c66a32e8304254b1f5bc5966` |
| x86 baseline executable                  | `32a317b050d2cd564a3a22023e92f54ccfc98f65a203e48d5e6166a1a78fce2e` |
| x86 candidate executable                 | `0e7c14c3c6364363f2d7bdb5c7864cacccdbed75366481e1af5d55d70b92be83` |

The sealed raw HTTP/native evidence archive has SHA-256
`59ebc1daf32a2c8b768717b485c60ab155b957a5cb67b81ae18bd579e519c55c`.
Source snapshots, binaries, raw logs, inventories and rejected probes are
retained in the workspace's `.context/pattern-count-20261002/` directory and
on the retained benchmark storage. The archive excludes executable copies;
binary hashes identify the separately retained builds.
