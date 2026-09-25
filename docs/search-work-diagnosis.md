# Where Summa spends extra query work

September 16, 2026. RGB disabled. This is a diagnosis, not a new speedup claim.
[Instrumentation and reproduction](query-work-diagnostics.md).

## The strongest finding: standalone top-k pruning

On the 5,032,104-document Wikipedia fixture, the 714 supplemental standalone
terms produce these warm-pass totals. Summa uses compact directories and exact
norms; Tantivy is the pinned 0.26 benchmark build.

| Operation | Summa document blocks | Tantivy document blocks | Ratio | Gap-payload byte ratio |
| --------- | --------------------: | ----------------------: | ----: | ---------------------: |
| Top 10    |               314,714 |                  37,987 | 8.29× |                  9.42× |
| Top 1000  |               716,470 |                 393,649 | 1.82× |                  2.35× |

This is not just one expensive term: **696 of 714** top-10 queries decode more
blocks in Summa. The median per-query block ratio is **2.54×**, and its geometric
mean is **3.01×**. The totals weight frequent terms heavily. For example, `is`
decodes 20,283 versus 145 blocks, and `to` decodes 23,860 versus 278.

The source gives a concrete bound-quality hypothesis: Summa's selected fixture
combines independent `max_tf`, `min_len` and minimum length/TF bounds. Tantivy's
[pinned block writer](https://github.com/quickwit-oss/tantivy/blob/9e63fc508153ef770f9ff980c8fa2f11e8e2e6db/src/postings/serializer.rs)
chooses a fieldnorm/frequency pair from the block using its BM25 weight. The
former can combine favorable statistics from different documents and admit
blocks whose actual scores are all below the threshold. This is a source-based
hypothesis for the measured excess, not a bound-tightness measurement. Summa's
support for changing scoring statistics and parameters prevents copying a
single-winner bound without proving compatibility.

Byte norms reduce Summa's top-10 supplemental block count to 302,412, still
7.96× Tantivy. The first investigation should target the existing single-term
block/group score bounds and threshold handling. These measurements establish
extra decoding; they do not yet separate bound looseness, tie handling, and
scoring differences between engines. Preserve query-global scoring parameters,
merged-segment semantics, and deterministic ties when testing tighter bounds.

## Other families have different costs

Ratios below are sums of decoded work, Summa compact/exact divided by Tantivy.
They are not geometric-mean query latencies.

| Family / command      | Document blocks | Document-gap bytes | Position blocks | Position bytes |
| --------------------- | --------------: | -----------------: | --------------: | -------------: |
| Union / top 10        |           0.43× |              0.79× |               — |              — |
| Intersection / top 10 |           1.04× |              1.70× |               — |              — |
| Phrase / top 10       |           0.99× |              1.65× |           0.67× |          0.81× |
| Intersection / count  |           1.32× |              2.11× |               — |              — |
| Phrase / count        |           0.99× |              1.65× |           0.99× |          1.29× |

Union and ranked phrase retrieval already avoid substantial work relative to
Tantivy. Their remaining latency cannot be explained by uniformly worse pruning.
Intersection and complete phrase traversal show the payload-density problem:
roughly the same document blocks consume about 65–70% more document-gap bytes.
Count intersections additionally decode 32% more document blocks.

Encoded bytes are not physical reads or cache misses. These results support
separate experiments on pruning, block representation, and per-block/per-score
CPU cost; they do not justify switching every query to another executor or
changing the posting codec default.

## Compact directories and quantized scoring

Legacy and compact/exact Summa have **identical** decoded blocks, values,
payload bytes, seeks, scoring units, phrase checks and selected executor work
for every captured query on both fixtures. This directly explains why the
previous compact-format change reduced RSS without materially reducing warm
query work. The index bytes and defaults are unchanged by this diagnostic work.

On the full corpus, quantization reduces total term-score evaluations by 3.2%
for top-10 and 0.7% for top-1000 across the combined workload. On official queries
alone it reduces them by 1.8% and 1.4%. Pure conjunctions perform exactly the same
scoring work with both norm representations. Most of the measured norm latency
regression therefore requires a per-unit/setup-cost explanation, rather than an
increase in term-score evaluations. Individual queries can still do more work;
per-query deltas are retained.

The official COUNT pass constructs **1,539 norm tables while performing zero
term or phrase BM25 score evaluations**. These are unnecessary setup work in
complete membership scorers, principally unions and intersections. Standalone
COUNT uses document-frequency metadata and builds no norm tables.

## Orchestration and validation

The diagnostic build times parsing, execution, and the segment worker separately.
For compact/exact supplemental top-10, median parsing is 6.09 µs; median execution
outside the sole segment worker is 15.61 µs. For official top-10 these are 10.66
and 16.86 µs. These instrumented timings show where setup/dispatch matters but
must not be quoted as production latency or added as independent medians.

In the repeated top-10 pass, 99.36% of posting-list admissions reuse structural
validation proofs. Official queries structurally inspect 6,280 posting blocks
while decoding 1,962,880 document blocks. This distinguishes admission scans from
decoding; selected payload-content checks still run with decoding and their
exclusive cost is not isolated by these counters. Dropping corruption checks is
not an evidence-backed shortcut.

## Scope and next experiments

1. Tighten single-term pruning safely; measure rejected/scored blocks and exact
   results together. This is the largest demonstrated traversal gap.
2. Separate quantized lookup/setup cost from score-count changes using identical
   persisted byte norms and a score-bit-equivalent canonical-arithmetic control.
3. Investigate intersection count traversal and payload density independently;
   verify codec experiments on both architectures before changing defaults.
4. Reduce setup/dispatch for cheap queries after measuring its share separately
   from the segment executor.

The full-corpus counters use the same Cascade Lake host, Rust 1.98.1, native CPU
flags, release LTO, caches, indexes and query ordering as the preceding format
comparison. A 100,000-document Apple M4 cross-check also shows extra standalone
pruning work and larger payloads. No diagnostic timings are used as production
latency. Tantivy's existing 12-counter instrumentation supplies decoder and
position counts; its score/pruning/setup counters are unavailable, not zero.

## Control experiment: same byte norms, canonical arithmetic

An isolated build disables the three norm-table construction sites, selecting
existing canonical scoring and length gathering on the **same quantized index**.
It changes neither persisted lengths nor bounds. Both builds reproduce every
score bit, document order and count in the preserved full-corpus quantized
oracle for all 1,676 queries and k=10/100/1000. The temporary change is restored;
it is not part of the implementation.

Seven rotated passes on the same x86 CPU, compiler and flags follow ten seconds
of warmup per engine/command. This run mixes the 962 official queries followed by
714 supplemental terms in one process per engine; compare engines within this
run, not its absolute times against earlier separate-process workload tables.
Geometric means of per-query median protocol latency, in microseconds:

| Command         | Compact / exact | Byte norms / lookup | Byte norms / canonical | Canonical versus lookup |
| --------------- | --------------: | ------------------: | ---------------------: | ----------------------: |
| Top 10          |          343.95 |              360.40 |                 359.16 |                   −0.3% |
| Top 1000        |          858.56 |              923.94 |                 900.64 |                   −2.5% |
| Top 100 + count |          629.43 |              633.05 |                 659.48 |                   +4.2% |
| Count           |          121.95 |              124.45 |                 121.04 |                   −2.7% |

The family split matters: canonical arithmetic improves official top-10 3.8%,
but regresses supplemental top-10 4.5% and supplemental top-100 + count 9.8%.
Official COUNT improves 3.5%; metadata-only supplemental COUNT also moves 1.7%
despite not constructing any tables in either build. Treat small timing changes
as mixed with code-generation/measurement variation, not isolated table cost.

Removing table construction and the lookup kernel together does not remove the
ranked gap to exact norms, and regresses complete ranked collection. The experiment
rules out a blanket “disable lookups” fix. It does not isolate one instruction,
cache effect or vectorization decision. Next, compare normalization access and
batch code generation while holding represented lengths and traversal fixed.
No scorer or codec default changes follow from this single-architecture control.

## Verification and limitations

- Required `python3 scripts/check_search.py check`: 1,816 passed, 25 ignored;
  formatting, Clippy and native-without-sync compilation passed.
- Diagnostic feature: 1,601 core unit tests plus the new query-work integration
  test passed. The integration covers legacy/compact and exact/byte combinations;
  scope tests cover nested captures, panic, suspension, cancellation, migration
  and concurrent segment workers. Diagnostic Clippy and native-without-sync
  compilation also passed.
- Both architectures: all 1,676 queries, all three Summa layouts, exact
  optimized-versus-exhaustive score bits/order/counts at k=10/100/1000 passed.
  Every diagnostic response was checked against the existing count oracle.
- Cold/concurrent latency and a new memory comparison were not run. The previous
  RSS/format audit remains the memory reference. Diagnostic storage is bounded
  per capture and worker, independent of corpus size. WASM rebuild skipped as
  previously requested.

[Raw work records, control timings, source overlays and checks](benchmark-results/query-work-2026-09-16/README.md).
