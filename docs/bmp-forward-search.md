# Forward values in BMP search passes

Status: research and measured experiments, 2026-09-05. Forward completion was
removed from production BMP search after the whole-query measurements below.
There is no search query option or automatic dispatch to forward values.
Ordinary BMP search always uses its inverted block scorer. Stored forward values
are used by L1 candidate backfill and BP whenever available; see
[optional storage](bmp-forward-index.md#configuration-and-scoring).

The prototype preserved the existing H/E/D traversal and collector and tried
forward completion only for small survivor sets after phase one. Its best warm
median improvement was 3.9% on one synthetic layout, disappeared after BP, and
did not improve tails consistently. This does not justify production search
integration. Kernel experiments remain isolated ignored tests; the retired
traversal prototype and patch are archived locally in
`.context/retired-forward-search/` alongside the measurement logs.

## What the literature supports

The original [BMP paper (SIGIR 2024)](https://arxiv.org/abs/2405.01117)
compared inverted, per-document forward, and block-local hybrid organizations.
The hybrid won: each selected block stores local inverted lists, enabling
shared query-term probes and accumulator updates. Per-document forward reads
have additional vector lookups. This is a reason to keep Summa's block path
for broadly populated candidate blocks.

[Seismic (SIGIR 2024)](https://arxiv.org/html/2404.18812v1) demonstrates a
complementary use: an aggressively pruned inverted structure nominates IDs,
then a forward index supplies complete vectors for inner products. That is an
approximate retrieval design, not a proof that smaller nomination pools retain
exact top-k. Summa V20 forward values contain the _retained quantized BMP_
postings, not pre-pruning original vectors; they cannot restore terms removed
at ingestion.

[Guided traversal (SIGIR 2022)](https://arxiv.org/abs/2204.11314) uses a cheaper
lexical model to guide evaluation of learned sparse scores. This supports
cross-vertical nomination plus forward scoring as an approximate recall/latency
tradeoff. It does not provide a safe BMP pruning bound by itself.

[LSP (2026 preprint)](https://arxiv.org/html/2602.02883v1) separates cheap
superblock selection from block evaluation and distinguishes rank-safe pruning
from approximate exclusion. Summa already follows this architecture with
H/E/D grids, integer bounds and a gamma-limited superblock policy. Its Table 9
also directly compares forward and flat-inverted document scoring: on that
SPLADE/MS MARCO setup, forward wins at small blocks (up to 64 slots), with the
advantage disappearing around 96–128. At 32 slots, reported times are 11.4 vs
17.0 ms (99% recall budget) and 22.1 vs 31.3 ms (safe retrieval). Summa defaults
to 32 slots, so this is directly relevant, though our adaptive encodings and
logical-order forward layout differ. Replacing
grid aggregation with document-forward scans would discard its main advantage.

## Where to integrate in Summa

The following are engineering inferences from those designs and the current
`query/bmp.rs` executor, not reported results from those papers.

1. **Direct scoring of small selected blocks.** Keep H/E/D selection and
   stopping rules, but evaluate a small selected block from forward values.
   This is the most direct LSP-paper analogue. Benchmark block sizes 8/32/128,
   long versus short queries, and BP-local versus scattered logical placement.
   The candidate scorer already supplies the shared integer dot-product kernel.
2. **Phase-two completion of a few surviving slots.** Today
   `score_superblock_blocks` scores the three heaviest query dimensions first
   when its two-phase policy is active. It rejects a
   whole block using `max(partial) + remaining_upper_bound`; otherwise it scores
   the remaining dimensions over the whole block. Instead, compute the same
   bound per slot and forward-score only survivors when few remain. Reuse the
   already prepared u16 query weights, u32 accumulator units, threshold and tie
   rules. A phase-one untouched slot still has partial score zero and must be
   considered whenever the remaining bound could reach the threshold. Do not
   turn the phase-one touched bitmap into an unsafe candidate filter.
3. **Highly selective eligibility.** When an already materialized filter gives
   a small, enumerable set of logical IDs, forward-score those values directly.
   This can bypass both grids and unrelated block payloads. Decide using value
   count (including ordinals), query length, and estimated bytes, not document
   count alone. Arbitrary predicate callbacks are not enumerable filters.
4. **Threshold seeding from known candidates.** Exactly score a bounded seed
   set using forward values before BMP traversal. A heap seeded with actual
   eligible hits can raise a safe threshold and skip more blocks. The seed hits
   must remain in the collector, with distinct-document/ordinal semantics and
   the existing multi-value threshold-publication rules. A guessed or merely
   approximate score is not a safe lower bound. L1 scores also cannot seed a
   raw BMP heap when the formulas differ.
5. **Cross-branch scoring reuse.** The L1 path already supplies this: retrieve
   cheaply from lexical/dense branches and forward-score missing sparse cells.
   Optional broader nomination or static inverted pruning is a separate,
   explicitly approximate quality experiment. It must retain the learned
   missing-value/organic-score contract and frozen candidate-pool evaluation.

For phase two, approximate costs are:

```
inverted = remaining-term header probes + postings for those terms in the block
forward  = survivors * (logical lookup + retained-vector scan + query intersection)
```

Logical forward order is intentionally independent of physical BP order so
merge/reorder can copy its payload. After BP, survivors may require scattered
forward reads. The experiment must include that locality cost; adding another
unbudgeted physical-to-forward map would defeat the storage design. Small
batched prefetch or a budgeted existing-map extension can be evaluated only if
lookup/fault measurements justify it.

## Summa kernel measurements

The ignored `query::bmp::forward_experiment::measure_forward_completion_against_block_scoring`
test compares the existing adaptive inverted block kernel with the shared forward
integer scorer. Run in release mode with `--ignored --nocapture --test-threads=1`;
set `SUMMA_FORWARD_EXPERIMENT_FULL=1` for whole-query evaluation instead of the
production phase-two mask. The fixture has 64 blocks, 4,096 dimensions, 64 retained
entries/vector, block sizes 8/32/128, queries of 8/32/64 dimensions, and spread or
partly shared document dimensions. It applies the production per-block term mask,
omits blocks with no remaining query terms, and reuses the block parsed by phase
one. Whole-query mode includes block parsing. Every selected integer accumulator matches
the inverted oracle exactly before timing. Nine samples alternate execution
order, each repeating five passes over the active blocks.

Apple M4, macOS 15.6.1, Rust 1.98.1, default release flags, warm RAM, no concurrent
CPU-heavy workload. Representative medians for 32-slot blocks and 64 query terms:

| Evaluation                     | Selected slots | Inverted block, spread/shared | Forward, spread/shared |
| ------------------------------ | -------------- | ----------------------------- | ---------------------- |
| Phase two (61 remaining terms) | 1              | 365.8 / 392.2 ns              | 142.4 / 151.6 ns       |
| Phase two                      | 2              | 364.3 / 393.5 ns              | 284.7 / 301.3 ns       |
| Phase two                      | 4              | 365.6 / 395.3 ns              | 566.8 / 608.1 ns       |
| Whole query                    | 32             | 373.2 / 400.8 ns              | 4,226.9 / 4,649.1 ns   |

Here selective completion wins at one or two survivors, but a whole-block
forward replacement loses by 11.3–11.6 times. Small blocks alone do not reproduce
the LSP paper's result in Summa. Its term/weight arrays and block-local access
patterns differ from our logical directory, per-vector validation, and adaptive
inverted representation. This experiment includes physical-to-logical mapping,
binary lookup, payload validation and dot product; it excludes survivor discovery,
grid traversal, collectors and cold page faults. The inverted microbenchmark zeros
its full accumulator, whereas production also has touched-slot reuse. These are
kernel results, not end-to-end retrieval speedups or a production crossover.

Peak process RSS / macOS memory footprint were 56,573,952 / 29,032,880 bytes for
phase two and 56,623,104 / 27,836,848 for the whole-query experiment. These include
fixture construction and transient buffers; they do not measure incremental
forward-index residency. Evidence: `.context/l1-forward-completion.log` and
`.context/l1-forward-full-block.log` in the Quebec workspace.

These results supersede the first unmasked-header microbenchmark. That earlier
comparison charged inverted scoring for absent query dimensions which production
has already excluded; it overstated the selective speedup. The corrected result
is about 2.6 times for one survivor on this fixture, not the initial 3.5 times.

## Query tables, fused validation and BP locality

The follow-up ignored test
`measure_forward_query_tables_and_record_bp_locality` uses 8,192 mmap vectors
with deliberately interleaved term clusters, 64 entries/vector, 32-slot blocks,
and 4,096 or 105,879 dimensions. It applies record BP with a 128 MiB budget,
changes 8,190 physical slots, and retains logical forward order. It samples up
to 64 active blocks with nonempty phase-two term masks; each of 11 samples
rotates four kernels and repeats ten passes. All selected integer results match
the inverted oracle before timing. The first three kernels are inverted blocks,
current forward intersection, and validated forward lookup with a dense u32
query table. The fourth combines payload validation and table scoring in one
scan. All forward variants include physical-to-logical mapping and binary lookup.

Representative medians after BP, in ns per selected block:

| Dimensions | Selected slots | Inverted | Current forward | Table, separate validation | Table, fused validation |
| ---------- | -------------- | -------- | --------------- | -------------------------- | ----------------------- |
| 4,096      | 1              | 238.0    | 140.3           | 88.0                       | 78.7                    |
| 4,096      | 2              | 239.8    | 308.3           | 194.9                      | 159.3                   |
| 4,096      | 4              | 261.6    | 620.8           | 405.1                      | 356.9                   |
| 105,879    | 1              | 529.2    | 133.3           | 91.7                       | 77.8                    |
| 105,879    | 4              | 572.2    | 568.1           | 401.4                      | 351.4                   |
| 105,879    | 32             | 576.4    | 4,630.6         | 3,481.9                    | 3,050.0                 |

The fused table kernel reduces forward time by 42–44% for one survivor and
34–48% across these rows.
It also moves the selective crossover, but does not make whole-block forward
scoring competitive. With BP, active blocks fell from 160 to 9 at 4,096 dimensions
and from 32 to 3 at 105,879 dimensions. This explains why global query length and
block size alone cannot decide dispatch; actual remaining-term density and BP
locality matter. The small post-BP active sets are a limitation of this synthetic
cluster fixture and are particularly cache friendly.

Setup is material: a 105,879-entry u32 table uses 423,516 bytes. Fresh allocation,
zeroing and preparation took about 3.1 µs, versus about 30 ns to clear the old
query's touched dimensions and populate a different query on an existing table.
At 4,096 dimensions, storage was 16,384 bytes and fresh setup 0.31–0.35 µs.
Setup is per query, not per vector. A future implementation should admit a capped
reusable table or retain sorted intersection for small batches; allocating the
large table for one or two backfills would erase the kernel gain. These setup
measurements alternate two queries; they do not model a production query stream.

The fused kernel is confined to this ignored experiment. It obtains raw views
through a validated fixture and repeats selected-vector validation inside timing;
production still validates selected values through the owning reader. Peak process
RSS / footprint were 92,258,304 / 26,296,704 bytes, including fixture construction,
BP and both mappings. This is warm mmap, not a cold-cache or per-query residency
measurement. Evidence: `.context/l1-forward-lookup.log`. All three experiments
were rerun after rebasing onto `origin/main` at `1844d9af` (Summa 1.8.124).

## Decision and limits

Use stored forward values for L1 and BP, and keep BMP retrieval inverted.
The dense table and fused-validation kernels remain research only. Any future
search proposal needs representative real-query evidence, including whole-query
setup, concurrent load, p50/p95/p99, memory, exact score/ordinal oracles and x86
measurements. The current ARM fixture does not establish a useful crossover.
Do not persist a second physical forward copy merely to mimic a paper's layout.

## Whole-query traversal and record-BP experiment

The retired `measure_forward_completion_in_whole_bmp_traversal` prototype
exercised the actual segment executor, including quantization, H/E/D traversal,
survivor discovery, logical lookup, selected-vector validation, scoring and top-k
collection. These are historical measurements of that prototype; its traversal
hook is no longer compiled or exposed by Summa.

The mmap fixture has 8,192 vectors with 64 entries each, 32-slot blocks, 4,096 or
105,879 dimensions, and interleaved term clusters. Record BP changes 8,190 slots.
There are 64 distinct 64-term queries, three heavy dimensions per query, depths
10/100, gamma=0, alpha=1, and survivor caps 1/2/4/8/32. Cap 32 is a deliberate
whole-block control. Each variant has 192 timings (three passes over the query
stream); disabled/enabled order alternates per query and pass. Every result's
logical ID, ordinal and score bits match the inverted executor outside timing.
An additional cross-layout check compares original inverted results with record-BP
forward-enabled results for all 64 queries. Prototype native and async
public-planner regressions also covered
multi-segment, filtered and multi-value queries, including an empty-field segment.

Apple M4, macOS 15.6.1, Rust 1.98.1, default release flags, no simultaneous build,
benchmark or ingestion work. Representative **whole segment-query** timings in
microseconds, p50/p95/p99:

| Dimensions / layout | Depth | Survivor cap | Inverted                 | Forward enabled          |
| ------------------- | ----- | ------------ | ------------------------ | ------------------------ |
| 4,096 / logical     | 10    | 2            | 169.46 / 216.12 / 281.46 | 162.92 / 218.38 / 281.04 |
| 4,096 / logical     | 100   | 2            | 217.54 / 243.92 / 268.08 | 222.17 / 257.33 / 296.96 |
| 4,096 / record BP   | 10    | 2            | 25.79 / 29.25 / 33.33    | 26.25 / 29.21 / 32.96    |
| 4,096 / record BP   | 10    | 32           | 25.83 / 28.54 / 30.29    | 41.92 / 49.17 / 52.00    |
| 105,879 / logical   | 10    | 2            | 54.62 / 65.38 / 76.75    | 56.79 / 69.83 / 77.29    |
| 105,879 / logical   | 10    | 32           | 55.71 / 65.75 / 69.67    | 128.04 / 138.83 / 150.96 |
| 105,879 / record BP | 10    | 2            | 6.75 / 8.92 / 9.75       | 6.88 / 8.88 / 9.67       |
| 105,879 / record BP | 10    | 32           | 7.33 / 12.08 / 14.38     | 12.54 / 19.29 / 24.12    |

At depth 10 and 4K dimensions, cap 2 completes 2,431 of 4,029 phase-two blocks,
scans 2,651 vectors / 848,320 payload bytes over 64 queries, and improves warm
median latency by 3.9%; the tail is essentially unchanged. At depth 100 it
completes only 417 of 3,846 blocks and regresses. After BP, cap 2 completes zero
blocks: survivors usually fill the remaining blocks, so discovery adds overhead
without saving scoring work. Cap 32 demonstrably executes forward completion on
BP output (229 blocks at 4K dimensions, 74 at 105K) but loses by 1.6–1.7 times at
depth 10. This is why the earlier one-vector kernel speedup is insufficient to
choose a production default.

The second run mode calls `MADV_DONTNEED` on the mapping before each query,
outside its timing. It does **not** establish a cold-disk measurement: measured
query-local minor/major page-fault counts are almost entirely zero and the OS
page cache remains uncontrolled. For 4K/logical/depth-10/cap-2, p50/p95/p99 become
293.58/366.50/393.46 µs inverted versus 312.08/427.58/470.71 µs forward. Even this
reclamation-hint mode loses the warm median benefit. Do not infer storage read
bandwidth from admitted forward payload bytes or treat the hint as cache eviction.

Peak process RSS was 96,354,304 bytes and macOS memory footprint 30,851,720 bytes,
including fixture construction, BP scratch, both mappings and all timed runs.
These are not incremental query residency measurements. The prototype added
under 3 KiB of fixed completion buffers and admitted at most 64 MiB of forward
payload per segment query, with no corpus-sized query cache or dense table.

Evidence: `.context/l1-forward-traversal.log` and
`.context/l1-forward-traversal-metadata.json` in this workspace. All query and
cross-layout oracles passed. The inconsistent benefit supports removing forward
completion from production search.
Real-corpus/concurrent-ingestion latency, controlled cold storage, and x86 runtime
measurements remain unrun; no automatic crossover or default is selected here.
