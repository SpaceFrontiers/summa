# Same-host Searchbench throughput: phrase scan optimization, optional impacts, and Luxir

Count agreement: **15/826 queries**. This is a restricted workload, not the complete published benchmark.

10M Wikipedia chunks; one merged segment; six server hardware threads and two driver threads on a separate physical core. All variants share an isolated loopback network namespace. Query and request caches disabled. Thirty-second session warmup, full untimed validation, one-second connection warmup, three ten-second repetitions per cell.

All four variants run sequentially. The prior and optimized Summa binaries use the same immutable non-RGB index, without impact metadata. The optional impact variant uses a separate index rebuilt from the same corpus; impacts remain disabled by default. Luxir is rerun on the same host. Indexing and compilation finish before query timing.

Summa uses a benchmark HTTP frontend over core, not its production gRPC service. Text-analysis differences exclude many queries; equal corpus counts do not prove general analyzer or relevance equivalence. Luxir uses the official 0.1.0 x86-64-v4 release, not the article's local build. No p99 claim is made.

| Family             | Included | Published queries |
| ------------------ | -------: | ----------------: |
| and_high_high      |        0 |                47 |
| and_high_low       |        7 |                50 |
| and_high_med       |        0 |                50 |
| high_phrase        |        0 |                30 |
| high_sloppy_phrase |        0 |                 7 |
| high_term          |        0 |                45 |
| low_phrase         |        7 |                50 |
| low_sloppy_phrase  |        0 |                37 |
| low_term           |        0 |                47 |
| med_phrase         |        1 |                46 |
| med_sloppy_phrase  |        0 |                27 |
| med_term           |        0 |                49 |
| or_high_high       |        0 |                42 |
| or_high_low        |        0 |                46 |
| or_high_med        |        0 |                45 |
| prefix3            |        0 |                50 |
| regex              |        0 |                13 |
| wildcard           |        0 |                49 |
| wildcard_lead      |        0 |                47 |
| wildcard_scan      |        0 |                49 |

**Throughput (queries/second; higher is better).** Values are the median of three repetitions. The CSV retains repetition min/max and peak process RSS. Raw replay JSON and memory samples accompany each cell.

| Family       | Operation | Clients | Distinct queries | Summa before (QPS) | Summa optimized (QPS) | Summa optimized + impacts (QPS) | Luxir (QPS) |
| ------------ | --------- | ------: | ---------------: | -----------------: | --------------------: | ------------------------------: | ----------: |
| and_high_low | COUNT     |       8 |                7 |           16,823.4 |              16,277.6 |                        15,553.3 |    13,605.6 |
| and_high_low | TOP_10    |       8 |                7 |            9,135.5 |               9,229.0 |                         9,248.0 |    10,642.1 |
| and_high_low | TOP_100   |       8 |                7 |            7,056.7 |               7,170.1 |                         7,121.8 |     8,256.7 |
| low_phrase   | COUNT     |       8 |                7 |              321.2 |                 383.3 |                           382.9 |       413.1 |
| low_phrase   | TOP_10    |       8 |                7 |            2,537.8 |               3,080.6 |                         2,940.3 |     1,495.3 |
| low_phrase   | TOP_100   |       8 |                7 |              952.6 |                 939.9 |                           803.0 |       758.8 |
| med_phrase   | COUNT     |       8 |                1 |               98.5 |                 138.1 |                           137.9 |       121.1 |
| med_phrase   | TOP_10    |       8 |                1 |            2,121.4 |               3,130.3 |                         6,351.6 |     3,539.8 |
| med_phrase   | TOP_100   |       8 |                1 |            1,316.2 |               2,011.5 |                         2,433.8 |     1,180.4 |

## Retained changes and tradeoffs

This comparison starts from the preceding certified rare-term phrase implementation,
not the initial Summa adapter. New work comprises exact singleton admission with
stable-ID-aware ties, inverse-seeded exact length cutoffs, AVX2/AVX-512 candidate
scans, existing L1 group bounds, cost-aware posting intersection, cached per-block
position offsets and singleton/membership position reads. Native scalar and WASM
retain identical matching and cutoff semantics. The follow-up changes no index
format; the already-rebuilt POS5/POS6 indexes are reused.

Without impacts, medium-phrase top-10 improves 47.6%, top-100 52.8%, and exact
counting 40.2%. Low-phrase top-10 improves 21.4%, and counting 19.3%. These changes
are not a uniform win: low-phrase top-100 falls 1.3% and conjunction counting 3.2%.
Ordinary medium-phrase top-10 remains 11.6% below Luxir; low-phrase counting is
7.2% below. No overall parity claim follows from this small subset.

Optional impact metadata closes the medium-phrase top-10 gap on this workload:
6,352 QPS versus Luxir's 3,540. It also reduces low-phrase top-10 by 4.6% and
top-100 by 14.6% compared with optimized Summa without impacts. Impacts therefore
remain disabled by default. They require a separate writer-built index; no live
index is retrofitted.

## Correctness and memory

All 36 timing cells completed without request errors. Before/after ordinary
indexes return identical counts, document IDs and score bits on all 45 HTTP
responses. Exhaustive top-100 oracles also match on every admitted query. The
impact index and original older build have equal count/score-bit sequences, but
independent parallel indexing can assign different internal IDs and break ties
differently. Current-format RGB passed a separate small exhaustive audit;
full current-format RGB throughput belongs to the 32-vCPU campaign.

Peak query RSS is 1,137 MiB before and 1,140 MiB after, versus 214 MiB for this
Luxir session. These include mapped resident pages, not total machine page cache
or heap-only usage. No low-memory or cold-I/O improvement is claimed here.

## Remaining profile evidence

Separate, untimed, single-client profiles run after all throughput cells.
For `"references reflist"` top-10, candidate admission accounts for 32.1% of
self samples, document-delta decoding 10.2%, and block bounds 5.9%.
For the slowest measured low-family count query, `"births category:living"`,
posting seek and intersection account for 15.5% and 13.6%, phrase-position
checking 10.2%, and cached membership testing 7.6%. These are software
`task-clock:u` samples on two queries, not an attribution of Luxir performance.
The initial profile attempt was denied by the host perf setting; the successful
retry temporarily enabled user sampling and restored the previous host settings.

The next work is reducing candidate scans/decoding and conjunction traversal,
not removing cancellation. Common posting-iterator changes also affect ordinary
term queries; preserve the measured conjunction-count regression in future
comparisons. No further runtime change was inferred solely from these profiles.

## Provenance

The optimized binary SHA-256 is
`5fcfdb94592eb594f8b73f7fc9467bbc46d92c531dd66ca041bee1917397527c`.
Rust 1.98.1, `-C target-cpu=native`, Intel Cascade Lake, n2-highmem-8.
The archive and source manifest are retained in `.context/yonik-benchmark/gap/`;
`gap-resumed-evidence.tar.gz` was downloaded and SHA-256 verified before shutdown.
It contains raw replay JSON, memory samples, correctness audits, experiment logs,
profiles, and the same-session Luxir c32/t4 health control. No transport correction
is applied to search QPS. Its median health throughput is 257,594 QPS.
The stop command lost its polling connection, but an independent cloud describe
confirmed the eight-vCPU machine is `TERMINATED`.

Later wildcard/core-parser work is **not present** in the timed binaries and does
not enlarge these 15/826 admitted queries. Native/WASM validation of the combined
working tree is recorded in the performance review.

[CSV with repetition ranges and RSS](searchbench-2026-09-23-gap.csv).

The transport control uses Searchbench's default four-thread client placement
on CPUs 2–3,6–7, overlapping server CPUs 2,6. Search timing uses the disjoint
server/client split stated above. This health result is a separately configured
transport diagnostic and is not used to normalize search QPS.
