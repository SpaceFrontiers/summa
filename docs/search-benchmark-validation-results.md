# Bounded posting validation reuse

Status: implementation, ARM and full-corpus x86 measurements complete. No
production default changes. The cloud machine is retained for the next dictionary
experiment; that subsequent code is outside this frozen measurement.

The [corrected bulk build](search-benchmark-bulk-results.md) rejects malformed
posting structures before exposing infallible hot reads. Revalidating every block
header on every list load cost roughly 6–9% on the full corpus relative to the
preceding unchecked prototype. This experiment reuses successful validation for
immutable byte ranges. It never disables the corruption checks on new bytes.

## Ownership, resources and compatibility

`PostingListReader` owns one immutable file handle and a fixed, direct-mapped
table of `(start, end, validated footer)` records. Exact range equality resolves
hash collisions. The table retains no posting payloads or query answers. Hits
reuse structural validation while retaining the existing L1 extraction. Misses
validate before publication. Read locks are released before construction; no
lock crosses I/O, validation or cancellation.

Only inline RAM/mmap handles qualify. Lazy callbacks can return different bytes
and are always read and validated; errors, short reads and cancellation remain
observable. Independent file owners never share validation, even at identical
offsets. Existing immutable readers remain valid across replacement. External
in-place mutation of live mmap files remains outside the immutable-file contract.

`IndexConfig.posting_validation_cache_bytes` and the benchmark CLI's
`--posting-validation-cache-bytes` configure a **per-segment** table budget. The
default remains zero; admission rejects more than 64 MiB before allocation or
index I/O. Sub-entry budgets allocate nothing. Lazy handles allocate nothing and
warn when a nonzero budget cannot be used. Segment heap estimates include actual
table storage. Overlapping reader generations must be included in memory sizing.

Term and prefix reads share this owner in native sync and async execution.
Standalone constructors retain their previous disabled policy. Compatible
serialization and merge payloads are unchanged; the public posting deserializer
continues to validate unconditionally. See the [design](posting-codecs.md#bounded-reuse-of-posting-validation).

## ARM comparison

Same existing first-100,000-Wikipedia-document ratio index, compiler and native
LTO flags as the frozen corrected baseline. Each command uses 15 seconds of joint
warmup and 11 interleaved rounds, rotating engine order, across all queries. Timings include
parse and pipe round-trip and exclude index open and hydration. Values below are
geometric means of per-query median microseconds. No builds, packaging or other
agent-owned CPU jobs overlap timing. This is a shared Mac, not the full corpus.

| Official 962 queries | Corrected baseline | New build, disabled | New build, 256 KiB | Baseline/cache speedup |
| -------------------- | -----------------: | ------------------: | -----------------: | ---------------------: |
| TOP10                |             49.399 |              49.444 |             48.599 |                 1.016× |
| TOP1000              |             51.845 |              51.630 |             50.536 |                 1.026× |
| TOP100 + exact count |             36.496 |              37.247 |             36.046 |                 1.012× |
| Exact count          |             27.157 |              27.707 |             26.718 |                 1.016× |

| All 714 distinct terms | Corrected baseline | New build, disabled | New build, 256 KiB | Baseline/cache speedup |
| ---------------------- | -----------------: | ------------------: | -----------------: | ---------------------: |
| TOP10                  |             27.248 |              27.747 |             28.147 |                 0.968× |
| TOP1000                |             44.127 |              44.671 |             44.502 |                 0.992× |
| TOP100 + exact count   |             18.390 |              18.979 |             18.720 |                 0.982× |
| Exact count            |              9.440 |               9.978 |              9.632 |                 0.980× |

The official workload improves modestly, while the supplemental terms slow down.
Single-term COUNT does not read external postings; its 2–6% variation across these
processes limits how confidently small changes can be assigned to this mechanism.
These data do not justify a default change or a universal speedup claim. The
full-corpus run separately measured the reader refactor and enabled cache against
Tantivy on a dedicated Cascade Lake machine, with unchanged index bytes and RSS capture.

## Full-corpus Cascade Lake comparison

All 5,032,104 documents and all 962 official queries, with a separate 714-term
coverage workload. Rust 1.98.1, native LTO, n2-highmem-8, pinned CPU 2, ten seconds
of warmup and seven complete passes per command using the upstream harness.
Both Summa builds and Tantivy 0.26 use the same machine; builds and profiles do not
overlap timing. Summa index files are identical before/after. Values are
geometric means of per-query median microseconds, including query parsing and
pipe round-trip, excluding index open and document hydration.

| Official 962 queries | Corrected baseline | New, disabled | New, 256 KiB | Tantivy | Baseline/cache speedup |
| -------------------- | -----------------: | ------------: | -----------: | ------: | ---------------------: |
| TOP10                |           1395.505 |      1399.240 |     1277.779 | 491.495 |                 1.092× |
| TOP1000              |           1954.704 |      1956.654 |     1838.167 | 870.759 |                 1.063× |
| TOP100 + exact count |           1943.889 |      1955.770 |     1844.662 | 810.085 |                 1.054× |
| Exact count          |           1026.635 |      1014.394 |      934.590 | 407.867 |                 1.098× |

| All 714 distinct terms | Corrected baseline | New, disabled | New, 256 KiB | Tantivy | Baseline/cache speedup |
| ---------------------- | -----------------: | ------------: | -----------: | ------: | ---------------------: |
| TOP10                  |            291.341 |       287.261 |      277.006 |  42.501 |                 1.052× |
| TOP1000                |            779.710 |       772.441 |      766.260 | 407.093 |                 1.018× |
| TOP100 + exact count   |            870.583 |       865.911 |      860.461 | 236.902 |                 1.012× |
| Exact count            |             49.162 |        50.310 |       49.684 |   6.569 |                 0.989× |

The cache improves the official workload by 5–10%, but Summa still takes
2.11–2.60× Tantivy's time. Single-term COUNT remains dominated by dictionary
lookup and does not benefit from posting validation reuse. All 1,676 x86
exact-count/exhaustive ranking gates pass with both budgets (3,352 checks),
including ordered IDs and score bits at depths 10, 100 and 1000. Cross-engine
counts agree; cross-engine ranking identity is not claimed.

Peak RSS is 972–973 MiB for official ranked Summa runs versus 710–711 MiB for
Tantivy. Enabling the cache adds at most 0.50 MiB relative to the disabled build
across all eight measured processes; official COUNT changes from 944.19 to
944.51 MiB. This includes mmap residency and must not be interpreted as heap alone.
A separate 199 Hz task-clock profile over all 301 unions (three passes, roughly
1,500 samples) attributes 2.69% of baseline samples to structural block validation,
1.96% with reuse disabled and 0.33% with the cache enabled. Bulk score accumulation
still accounts for about 53%, collection 22% and the driver 13%. These short
profiles support the mechanism but do not explain every whole-workload change.

## Correctness and checks

The full eight-phase search harness passes: 1,654 native tests, 26 ignored, native
without sync, portable core, API docs and real-server broker tests. The WASM build
and all 20 tests pass. Two additional native integration tests and focused Clippy
pass without changes to the measured production code, bringing the native total
to 1,656. All 1,676 ARM exact-count and exhaustive ordered-ID/score-bit checks pass
with both zero and 256 KiB budgets, at ranked depths 10, 100 and 1000.

Coverage includes byte-identical serialization for all three posting codecs,
collisions, disabled/tiny/oversized budgets, replacement corruption at a cached
offset, lazy changing bytes, I/O errors, short reads and cancellation. Public
sync and async search reject the corrupt replacement while the original RAM
reader continues to work. The mmap reload test verifies unchanged reader reuse,
budget propagation to new segments, old-reader results and prefix serialization.

The frozen source overlay is `summa-validation-v1.tar.gz` over base
`ce2c96b945fccc4bac58ccc45bb4bc23b809773e`. The later two-test file is retained
separately from that measured overlay. The current workspace has subsequent dictionary changes; only the frozen overlay
represents this measured cache pass. The complete local cloud archive is
`validation-cloud-evidence.tar.gz` (27,262,114 bytes), SHA-256
`6593272b203f11654d38ad00200bc7175d45fd4d5207a81e8a3ee1f8e6b13b45`.
All 83 manifest members were verified, and before/after index hashes agree.

Compact, hash-verified build inputs and raw results are in the
[evidence bundle](benchmark-results/validation-cache-2026-09-13/README.md).

The historical posting-only cache measurements above are superseded for the
current reader by the [position admission follow-up](search-performance-review.md#batched-lengths-phrase-driver-and-corrected-position-admission-september-14).
Document and position proofs now share the same configured heap budget; position
structural validation is mandatory, including misses and lazy reads. The new
full-workload report includes its performance, resident-memory cost, correctness
regressions, and an unresolved union scoring regression.
