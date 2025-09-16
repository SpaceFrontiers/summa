# Pruning and scoring setup fixes

RGB remains disabled.

The retained reader changes official top-10 from 663.043 to 659.971 µs
(-0.5%, effectively flat), still **1.155× Tantivy**. Byte-norm official COUNT
improves from 518.756 to 502.416 µs (-3.1%). The requested overheads are removed,
but the official performance gap is not closed. The combined index below is an
opt-in experiment; its standalone-term gain does not generalize to the official
workload.

## What changed

- Collectors pass their scoring requirement through the existing scorer options.
  Membership-only term scorers attach document lengths without constructing a
  normalization table. Ranked and positioned collection retains eager tables and
  the original lookup loop. There is no new per-score lazy-cache check. If an
  internal consumer requests a score despite the hint, canonical scoring over the
  same represented lengths remains valid.
- Text cursors prepare query-constant BM25 bound coefficients once. Bounds retain
  conservative rounding and unsupported-parameter fallbacks. Standalone top-k
  avoids decoding impact records when a cheaper bound already rejects the block
  or group. Strict comparisons preserve seeded thresholds and deterministic ties.
- Packed gap decoding uses a bounded ordering proof: at most 127 strictly positive
  gaps of at most `2^25` sum to less than `2^32`. Matching ordered endpoints prove
  every prefix is ordered. Wider gaps retain the full decoded-ID scan, and full
  blocks still validate their reserved first gap. Corrupt input is still rejected.

These changes use the existing scorer, similarity and posting owners. They add
no persisted format, corpus-sized cache or alternative executor. Existing indexes
benefit from the reader changes; obtaining packed gaps or coupled impact bounds
requires an index that contains those existing optional encodings.

## Why pruning was bad

The selected ratio-only fixture combines maximum TF and minimum length/TF from
potentially different documents. A bound can therefore describe a score no actual
document achieves. Giving the executor the exact final top-10 threshold isolates
this from slow threshold discovery. On ARM's 100,000-document fixture, `is` still
decodes 540 blocks with that ideal threshold, versus 548 normally; an impact-bearing
index needs 10 blocks with the ideal threshold and 30 normally. `to` similarly
changes 439 → 438 with ratio bounds, versus 40 → 8 with impact bounds.

On the five-million-document fixture, even the ideal floor leaves 17,524 blocks
for `is` and 23,789 for `to` with ratio bounds. Impact bounds need 26 and 16,
respectively; without the ideal seed, they need 111 and 85. The corresponding
ratio-only unseeded counts are 20,283 and 23,860.

The ideal-threshold experiment uses the existing seed API and verifies the exact
ordered top-10 score bits. It is a diagnostic upper limit on threshold warmup,
not an implementation that knows answers in advance. It confirms that stronger
persisted bounds are needed for these terms. Query arithmetic alone cannot recover
coupled statistics absent from the index.

## Why gap bytes were larger

Compact directories changed metadata, not the arrays. Rounded packing spends
8/16/32 bits on values that may need only a few bits. Simd4x packs full blocks at
exact widths and stores gaps minus one; consecutive full blocks need no gap
payload. It already existed as an option, but previous complete-workload latency
regressed. The new ordering proof removes one redundant pass in that decoder.
The combined fixture tests compact directories, Simd4x and impact bounds together.
Encoded byte counts do not measure disk reads or cache misses.

## Why count queries built tables

Attaching quantized document lengths eagerly built all 256 normalization entries,
even when a Boolean scorer was used only for membership. On the 962-query full-corpus
COUNT pass, table constructions fall from 1,539 to **zero** (ARM: 1,533 → zero). Ranked collection
still uses the same lookup scores. The regression covers both unions and
intersections, all compact/exact/byte combinations, and exactly one table
for complete ranked collection, plus synchronous term membership.

## Reproducing the combined configuration

Use a new index. Existing blocks are preserved during compatible copy merges;
changing writer options does not transcode old payloads or invent impact metadata.
The measured exact-norm combination is:

```rust
IndexConfig {
    compact_text: true,
    posting_codec: Some(PostingCodec::Simd4x),
    posting_impact_bounds: true,
    ..Default::default()
}
```

## Rejected experiments

A lazy table cache removed unused COUNT setup but introduced an initialization
check into ranked scoring. It was replaced with the collector hint described
above. A threshold-inversion predicate tried to reject impact envelopes using
multiplication instead of division. Its ARM results were small and its x86
combined-index top-10 regressed by 3.0% (263.496 → 271.283 µs), so it is absent
from the selected source. The ordinary guarded impact bound remains the only
production calculation.

## Where the official gap remains

The official set contains 300 intersections, 300 phrases and 301 unions, but only one standalone-term query. The supplemental set adds 714 standalone terms. That workload difference explains why a large aggregate gain can coexist with an official regression. Category top-10 geometric means (microseconds):

| Category           | Queries | Starting compact | Reader / same index | Combined index |  Tantivy |
| ------------------ | ------: | ---------------: | ------------------: | -------------: | -------: |
| intersection       |     300 |          519.608 |             524.410 |        528.855 |  422.833 |
| phrase             |     300 |          652.038 |             654.117 |        660.983 |  517.852 |
| union              |     301 |          742.435 |             724.251 |        821.358 |  699.854 |
| intersection_union |      40 |         1441.052 |            1424.991 |       1478.885 | 2542.206 |
| negated            |      19 |          947.618 |             938.046 |        923.299 |  411.197 |

The combined configuration regresses unions most strongly. The retained reader on the original index is much closer to Tantivy for unions than for intersections or phrases. This identifies the remaining workload to optimize; the timing split alone does not prove a particular CPU instruction or I/O cause. Stronger standalone pruning is insufficient to close that gap.

The 100,000-document Apple M4 control changes top 10 -2.9%, top 1000 -0.1%, top 100 + exact count -2.4%, exact count -2.0%. These small ARM differences include protocol overhead and shared-machine noise; they do not justify a default change.

## Storage and resident memory

Fresh processes execute three official passes per command. The table uses the
third top-10 snapshot, before subsequent commands expand the working set. RSS
includes file-backed mappings; anonymous memory is reported separately. No
pages are locked. Raw mappings and later-command snapshots are in the archive.

| Engine              | Index MiB | RSS MiB | Anonymous MiB |
| ------------------- | --------: | ------: | ------------: |
| Starting compact    |   4550.15 |  736.25 |          5.74 |
| Reader / same index |   4550.15 |  736.26 |          5.75 |
| Byte norms after    |   4545.35 |  731.75 |          5.77 |
| Combined index      |   3639.34 |  604.01 |          5.73 |
| Tantivy             |   2890.67 |  565.39 |          0.71 |

Most of the RSS gap is resident file data, not heap. Removing COUNT's temporary
1 KiB tables removes work; it does not explain hundreds of MiB of residency.
Packed gaps and better pruning reduce the payload footprint. Byte norms remain
separate because they change represented scoring lengths and can change ranks.

## Validation and evidence

The selected source passes `python3 scripts/check_search.py check`: formatting,
Clippy, **1,817 tests passed / 25 ignored**, and the native build without default
sync features. The diagnostic regression and its Clippy check also pass. Both
architectures verify all 1,676 queries on five layouts against preserved exhaustive
references, including exact ordered score bits for top-10/100/1000 and complete
top-100 with exact counts. All 962 official COUNT results match the reference
and construct zero normalization tables. Byte norms use their own oracle.
WASM builds are skipped per the user's instruction.

No writer or persisted format changes in this follow-up. Existing corruption
and copy-merge regressions remain enabled; validation is not disabled.
The inverse-threshold prototype was rejected after complete top-10/top-1000
measurements; its unfinished commands and planned memory audit were canceled.
The selected source has a complete separate run and residency audit.