# Dictionary block and allocation experiments

Status: both dictionary builds have complete, verified full-corpus results.
The separate Zstd allocation candidate also has verified full-corpus results.
All settings remain opt-in with unchanged defaults. Summa has not established
leadership. This report separates frozen candidates from the current workspace.

## Changes and controls

`SSTableBlockSize` accepts 512 bytes through 1 MiB; the existing 16 KiB target
remains the default. Native/WASM builders and the existing merge, compaction and
text-reorder owners carry this policy. `IndexConfig.term_cache_budget_bytes`
adds an optional cap on actual retained decompressed bytes alongside the existing
block count. Defaults remain 256 blocks and no additional byte cap. Oversized
blocks remain readable without retention. Hash/deque metadata and in-flight
references are outside the payload cap and require separate RSS accounting.

The tool exposes `--term-dict-block-bytes` on index/merge/compact/reorder and
`--term-cache-blocks`, `--term-cache-bytes` and
`--posting-validation-cache-bytes` on search. The benchmark adapter exposes the
same core settings. No codec, scorer, query-answer cache or STB5 version change
is involved. This follows the small-block principle in Lucene's
[block-tree dictionary](https://github.com/apache/lucene/blob/1160d3c8a256ee967b16df7e9e26ec39da3aea01/lucene/core/src/java/org/apache/lucene/codecs/lucene103/blocktree/Lucene103BlockTreeTermsWriter.java),
while retaining Summa's existing Zstd/restart representation.

The experiment also fixes existing failure/resource behavior. Bulk dictionary
prefetch previously read the entire table and expanded its configured cache cap;
it now admits at most 4 MiB compressed input, preserves both caps and performs no
payload I/O with disabled retention. Merge prefetch uses batches of four and
drains started reads before returning an error. The writer rejects unreadable
oversized entries, bounds entry scratch before growth, and refuses to finish
after serialization errors, partial output writes or caught panics. Output
ownership and metadata publication remain with the existing lifecycle owner.

Fixtures isolate dictionary layout: exclusively created private copies use the
canonical reader/writer to regenerate only `.terms`; root metadata is copied
last. Every ordered key and serialized `TermInfo` agrees, all other file hashes
agree, and the 16 KiB reconstruction is byte-identical. Default writer output
also matches the frozen 32,352-byte golden fixture. The old reader successfully
reads smaller blocks. See the [preparation and invariants](search-benchmark-game.md#isolated-dictionary-fixture-preparation).

## Initial full-corpus block-size result

Frozen dictionary-v1, before the later bounded writer-error follow-up and Zstd
capacity fix. Full 5,032,104-document corpus, 962 official queries, same Cascade
Lake n2-highmem-8 machine and Rust 1.98.1 native LTO. CPU 2, upstream complete-workload
passes, ten-second warmup and seven samples. Values are geometric means of
per-query median microseconds, including parsing and pipe round-trip, excluding
index open and hydration. Both block and byte caps are identical across the
three format variants: 8,192 blocks and 4 MiB retained decompressed bytes, with
256 KiB posting-validation reuse. No build or profiling overlaps timing.

| Official workload    |   16 KiB |    4 KiB |    1 KiB | Tantivy 0.26 |
| -------------------- | -------: | -------: | -------: | -----------: |
| TOP10                | 1329.295 | 1131.589 | 1136.539 |      504.630 |
| TOP1000              | 1853.732 | 1681.457 | 1699.222 |      871.500 |
| TOP100 + exact count | 1894.929 | 1680.981 | 1726.391 |      808.604 |
| Exact count          |  949.916 |  794.002 |  786.609 |      410.345 |

The 4 KiB variant improves the matched 16 KiB configuration by 10–20%, but still
takes about 1.93–2.24× Tantivy's time. The old frozen validation-cache binary and
new default-policy control are retained in the raw data to distinguish code
changes from block-size/cache policy. Exact-count and exhaustive ordered-ID /
score-bit gates pass for all three layouts and the old-reader compatibility
control. The supplemental 714-term timing is complete below.

The full dictionary has 3,964,752 entries. At 4 KiB it grows from 54,363,912 to
56,923,053 bytes, with 4,761 → 19,025 blocks. Posting, position, norm, document-ID
and stored-value files are unchanged. This is a dictionary-only size comparison,
not a claim about whole-index growth or indexing throughput.

| All 714 distinct terms |  16 KiB |   4 KiB |   1 KiB | Tantivy 0.26 |
| ---------------------- | ------: | ------: | ------: | -----------: |
| TOP10                  | 288.645 | 201.590 | 205.136 |       43.860 |
| TOP1000                | 784.543 | 684.186 | 690.796 |      408.579 |
| TOP100 + exact count   | 886.354 | 770.761 | 770.582 |      238.884 |
| Exact count            |  49.911 |  10.161 |  10.170 |        6.623 |

The term-only count improvement is 4.91× at 4 KiB; Summa remains 1.53× slower
than Tantivy for that command. Its TOP10 gap remains 4.60×, motivating tighter
score bounds in addition to dictionary work. Peak official ranked RSS is roughly
973 MiB at 16 KiB versus 971–972 MiB at 4 KiB; supplemental COUNT is 58.22 versus
56.45 MiB (Tantivy 47.23 MiB). Equal caps do not imply equal working sets: smaller
blocks can retain useful keys with fewer decoded bytes. At 1 KiB, the full
dictionary grows to 64,270,340 bytes and 76,033 blocks. All other index files
remain byte-identical. These are warm, single-thread benchmark measurements,
not concurrent ingest or cold-storage throughput claims.

## Final writer-fix confirmation

Dictionary-v2 includes the bounded serialization and failed-writer corrections.
The same machine, fixtures, compiler, flags, caps and complete-workload protocol
were repeated with the frozen v1 4 KiB binary as an additional control. These
results precede the independent decompressor-capacity fix.

| Official workload    | v1 4 KiB control | v2 16 KiB | v2 4 KiB | v2 1 KiB | Tantivy 0.26 |
| -------------------- | ---------------: | --------: | -------: | -------: | -----------: |
| TOP10                |         1169.038 |  1339.302 | 1149.490 | 1150.342 |      514.595 |
| TOP1000              |         1723.538 |  1934.891 | 1760.260 | 1768.605 |      882.911 |
| TOP100 + exact count |         1722.783 |  1961.198 | 1749.443 | 1802.104 |      852.032 |
| Exact count          |          805.351 |   962.584 |  814.553 |  814.867 |      419.351 |

| All 714 distinct terms | v1 4 KiB control | v2 16 KiB | v2 4 KiB | v2 1 KiB | Tantivy 0.26 |
| ---------------------- | ---------------: | --------: | -------: | -------: | -----------: |
| TOP10                  |          219.488 |   304.426 |  220.646 |  220.638 |       46.287 |
| TOP1000                |          708.293 |   820.045 |  705.246 |  713.584 |      406.692 |
| TOP100 + exact count   |          806.800 |   986.271 |  824.278 |  814.094 |      262.200 |
| Exact count            |           10.443 |    51.346 |   10.455 |   10.306 |        6.747 |

The v1/v2 4 KiB control changes are mixed, mostly 1–3%; this does not establish
a search-speed effect from the writer failure fix. Within v2, 4 KiB improves the
matched 16 KiB layout by 10–18% on official commands. Summa still takes
1.94–2.23× Tantivy's time. All three layouts pass the 1,676 exhaustive-ranking /
exact-count gates. Rewriting the dictionaries with v2 produces exactly the same
bytes as v1, on both architectures. Fourteen storage tests and three policy tests
also pass on x86.

Peak official ranked RSS is 972.92–973.51 MiB at 16 KiB and 971.29–971.96 MiB at
4 KiB, versus 710.75–710.95 MiB for Tantivy. Official COUNT uses 944.81 / 942.49 /
700.86 MiB respectively; the supplemental COUNT processes use 57.65 / 55.98 /
47.46 MiB. These process peaks include mappings and metadata; they are not
measurements of retained dictionary cache alone.

The full-corpus vocabulary diagnostic measures uncached decode at 46.030 /
19.924 / 12.349 µs per lookup for 16 / 4 / 1 KiB. Warm cache passes take 47.717 /
20.428 / 2.262 µs; the 1 KiB sample fits while larger blocks churn the same cap.
Bounded prefixes take 227.989 / 225.961 / 330.910 µs. The serialized block index
uses 55,579 / 216,082 / 819,388 bytes, while the Bloom filter stays 4,955,960 bytes.
This metadata growth and the prefix cost accompany the exact-lookup benefit.

The [reproducible evidence bundle](benchmark-results/dictionary-blocks-2026-09-13/README.md)
contains raw samples, resource logs, exact source overlays, golden bytes, tests,
query sets and index manifests. The full executable archive has 136 verified
members, 27,178,425 bytes, SHA-256
`5a2129da5db4769146be22af82325cc5b15c3a8604b239263f5a4ff09ea35550`.
All three fixture manifests and the original Summa/Tantivy index manifests
remain unchanged after timing. The machine remains in use for the separate allocation
and conjunction experiments.

## ARM behavior and limitations

On the 100k-document ARM fixture, whole-query changes are modest and vary across
runs; the final 4 KiB variant regresses some supplemental controls. The paired
method rotates engine order over eleven per-query repeats after fifteen seconds
of joint warmup. This gives more immediate query locality than the cloud's
complete-workload passes. The Mac is shared; these data do not support changing
a default. Raw samples preserve both dictionary-v1 and the writer-fix v2 run.

A separate diagnostic samples 2,048 keys across the entire vocabulary, independent
of the public query set. With retention disabled, median-of-three mean lookup
time is roughly 39.8 / 26.2 / 22.9 µs for 16 / 4 / 1 KiB blocks. With a 4 MiB
cap, the 1 KiB sample fits in 2.15 MiB and subsequent passes take about 0.76 µs;
the other layouts continue evicting. That working-set result is not universal.
The lazy-callback variant measures requested byte ranges using immediate in-memory
callbacks, not network or cold-disk latency.

Smaller blocks also have a cost: 64 bounded prefix scans produce identical
56,962 entries, but take 138 / 174 / 406 µs per prefix at 16 / 4 / 1 KiB. Prefix
output is capped at 1,000 per request with explicit truncation. The serialized
block index grows from 4,669 to 16,443 and 57,430 bytes; total dictionaries are
4.677 / 4.844 / 5.311 MB. These tradeoffs reinforce retaining configurable policy.

## Zstd allocation follow-up

The low-level diagnostic exposed an additional avoidable allocation. With the
enabled zstd 0.13.3 features, its [bulk decompressor](https://github.com/gyscos/zstd-rs/blob/v0.13.3/src/bulk/decompressor.rs)
does not derive a smaller capacity;
the supplied 64 MiB reader safety limit becomes a 64 MiB vector reservation for
each dictionary miss. Regressions reproduce that capacity for sub-1 KiB ordinary
and dictionary-compressed frames. The current candidate uses the library's stable
[frame-content-size API](https://docs.rs/zstd-safe/7.2.4/zstd_safe/fn.get_frame_content_size.html)
as an allocation hint. It preserves bounded streaming fallback for size-less and
concatenated frames and continues rejecting corrupt data. This changes no encoder
or persisted bytes. Its measurements are separate from the table above.

Dictionary-v2 passes the full eight-phase harness: 1,671 native tests, 26 ignored,
portable/native-without-sync compilation, API docs and real-server broker tests,
plus the WASM build and 20 tests. The allocation candidate adds five regression
and frame-boundary tests; its focused harness passes 1,676 native tests and the
portable build, with all 20 WASM tests passing. Both dictionary and allocation evidence archives are verified. The machine remains
active for the execution-profile follow-up.

A five-round paired ARM storage repeat now isolates the allocation change with
identical fixtures and alternating process order. No builds or other agent-owned
CPU work overlaps timing. Each diagnostic uses the same 2,048 vocabulary keys;
values below are medians across rounds of the three-pass mean lookup times.
Retention-disabled results measure decompression misses, not physical cold I/O.

| Block target | Previous miss µs | Sized-allocation miss µs | Speedup | Prefix before/after µs |
| ------------ | ---------------: | -----------------------: | ------: | ---------------------: |
| 16 KiB       |           40.212 |                   19.385 |   2.07× |       137.924 / 98.408 |
| 4 KiB        |           26.627 |                    6.354 |   4.19× |       176.292 / 82.400 |
| 1 KiB        |           23.726 |                    3.196 |   7.42× |       415.133 / 93.306 |

The warm per-query benchmark remains mostly flat, as cache hits bypass the
allocation. The storage gains must not be promoted to whole-query speedups.
The prefix tradeoff also changes after removing allocation overhead: 4 KiB is
faster than 16 KiB in this diagnostic, while 1 KiB no longer incurs the earlier
fourfold penalty. This still does not justify changing defaults from one fixture.
Raw repeats and process resource measurements are retained as
`.context/zstd-capacity-paired-lookup.json` and `capacity-lookup-*.time`.

In the paired storage processes, median peak RSS falls from 14.41 to 13.44 MiB,
and median system CPU time from 0.89 seconds to the timer's displayed 0.00 seconds.
The 64 MiB figure is vector capacity/virtual reservation, not a claim that every
miss made 64 MiB resident.

## Full-corpus allocation result

The independent allocation run uses the same full corpus, machine, native-LTO
compiler, CPU 2, 4 MiB / 8,192-block dictionary caps and 256 KiB posting-validation
budget as the dictionary experiment. The before build uses 16 KiB blocks; only
its matched after-16-KiB column isolates the allocation fix. Smaller-layout
columns combine allocation and block-size policy. Seven complete-workload
samples follow ten seconds of warmup.

| Official workload, µs | Before 16 KiB | After 16 KiB | After 4 KiB | After 1 KiB | Tantivy |
| --------------------- | ------------: | -----------: | ----------: | ----------: | ------: |
| TOP10                 |      1378.567 |     1291.465 |    1234.086 |    1165.159 | 513.630 |
| TOP1000               |      1912.483 |     1829.583 |    1713.863 |    1708.737 | 882.169 |
| TOP100 + exact count  |      1970.416 |     1916.958 |    1748.773 |    1772.790 | 875.430 |
| Exact count           |       974.090 |      904.507 |     808.650 |     804.792 | 417.500 |

| All 714 terms, µs    | Before 16 KiB | After 16 KiB | After 4 KiB | After 1 KiB | Tantivy |
| -------------------- | ------------: | -----------: | ----------: | ----------: | ------: |
| TOP10                |       307.605 |      263.772 |     206.541 |     204.503 |  44.086 |
| TOP1000              |       788.194 |      757.904 |     685.281 |     691.442 | 408.883 |
| TOP100 + exact count |       935.321 |      851.825 |     773.423 |     788.975 | 240.488 |
| Exact count          |        51.393 |       37.940 |      10.763 |      10.663 |   6.629 |

The matched layout improves official commands 3–7%, leaving the systemic gap
open. All three layouts pass the 1,676 correctness gates and all recorded counts
agree. Peak official ranked RSS stays roughly 973 MiB at 16 KiB, versus 971–972
MiB at 4 KiB and 711 MiB for Tantivy. The allocation fix does not materially
reduce whole-query RSS.

Five alternating x86 storage repeats give uncached decode before/after of
44.982 / 32.924 µs (16 KiB), 19.173 / 11.512 (4 KiB), and 11.941 / 6.226 (1 KiB).
Bounded prefix scans improve 226.461 → 195.551, 218.400 → 172.598, and
322.353 → 203.854 µs respectively. Diagnostic median peak RSS is 71.64 → 72.67
MiB, while system CPU falls 0.79 → 0.02 seconds. These are warm in-memory storage
operations, not physical cold-I/O results.

The [allocation evidence bundle](benchmark-results/zstd-capacity-2026-09-13/README.md)
retains sources and raw samples. All 88 members of the 18,519,983-byte executable
archive are verified, SHA-256
`60a9c95e44c5a52f7682ca444a4dacae25da3bcac7edf60c4d3b2c81a439c11a`;
all original and rewritten index manifests remain unchanged.
