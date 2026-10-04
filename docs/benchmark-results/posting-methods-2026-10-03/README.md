# Posting intersection and union methods review

October 3, 2026. Baseline `8f9c3f66`. Research and isolated screens complete;
the [follow-up](../posting-dispatch-2026-10-03/README.md) selects a validated x86 dispatcher.

Eight papers were read through Machine Library MCP, using bounded full text
and targeted passages. A ninth was read at the authors' arXiv page after two
Machine Library searches did not return it. [Source metadata](sources.json)
records the canonical citations. Paper speedups are not Summa speedups.

The strongest broadly applicable mechanisms are already present. Fresh ARM and
x86 screens reject blanket replacements and identify **balanced-block SIMD**
as the most promising execution candidate. The [conditional-dispatch follow-up](../posting-dispatch-2026-10-03/README.md)
validates and selects the x86 hybrid through real conjunction and phrase callers. It would be incorrect to claim that every
worthwhile method is implemented or that every method below has been benchmarked.

## Method coverage and decisions

“Retained” means present in the baseline, not newly added by this review.
“Screened” means the isolated algorithm, not a full production query comparison.

| Method                                                                                          | Current implementation / evidence                                                                                                                                                                                                     | Decision                                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shortest-list-first, skip impossible ranges, multiway elimination                               | Ranked conjunctions order by DF, intersect the two rarest blocks, probe remaining clauses, and use later heads to skip impossible candidates. [Block execution](../../search-block-execution.md), current native regressions.         | Retained. Dynamic remaining-length reordering is not separately measured; it adds control work and must preserve TF row identities.                                                               |
| Branch-reduced SIMD block intersection                                                          | AVX2 + POPCNT 1×16 prefix advancement, NEON 1×8 equality groups, portable tail; ordinal pairs and resumable bounded output.                                                                                                           | ARM/portable behavior retained. The follow-up replaces the x86 prefix loop with a measured balanced/galloping hybrid.                                                                             |
| Balanced 4×4 SIMD intersection                                                                  | New ordinal-preserving SSE2/NEON candidate, checked against an independent binary-search oracle.                                                                                                                                      | Selected only inside the follow-up’s x86 hybrid after generic-release query validation. ARM and unconditional replacement remain unselected.                                                      |
| Exponential/galloping search                                                                    | Existing directory skipping and very-skewed ranked seek path; new scalar exponential-search screen. [Earlier screen](../dict-scan-2026-09-27/intersection-screen.json) also found large middle-density regressions.                   | Follow-up retains grouped SIMD galloping as the x86 hybrid’s skewed/tail policy after real-query validation. This is not the complete published V3 family or proof of an optimal threshold.       |
| ARM prefix-count advancement                                                                    | New unsigned NEON 1×8 prefix-count candidate, analogous to the current x86 approach.                                                                                                                                                  | Rejected as implemented: substantially slower than current NEON groups.                                                                                                                           |
| Array/bitmap hybrids; direct bit probes                                                         | Default `RoundedBitmap` posting blocks, direct word windows, membership tests, and popcount-based TF ordinal recovery without ID expansion. [Bitmap design and paired evidence](../../bitmap-posting-blocks.md).                      | Retained; not missing because Summa does not use Roaring as its posting container.                                                                                                                |
| Bitmap AND/OR and cardinality without ID materialization                                        | Boolean DocSet windows, sparse-candidate probes within those windows, expanded unions retain their bitmap; popcount after accumulation. [Dense count evidence](../closing-gap-2026-09-24/dense-count.json).                           | Retained. Avoid repeated intermediate cardinality work where it is not needed.                                                                                                                    |
| Sparse append/sort/deduplicate versus segment bitmap                                            | Shared expanded-union owner selects at one posting per 1,024 segment documents. Fresh sort/bitmap/heap/window count screen includes memory costs.                                                                                     | Retained. Synthetic results confirm workload dependence; they do not justify retuning this threshold.                                                                                             |
| Lazy k-way union                                                                                | Ranked constant-score expansions use heap-ordered pending metadata and open only lists needed for the next document. Complete counts use other paths.                                                                                 | Retained. Fresh complete-count heap timings do not evaluate or invalidate early top-k stopping.                                                                                                   |
| Dominant-list exact union cardinality                                                           | At 64:1 dominance, dictionary DF plus deduplicated tail minus tail/dominant overlap; canonical membership batches. [Pattern-count evidence](../pattern-count-2026-10-02/README.md).                                                   | Retained. Deletions, filters, chunk maps and collector capabilities remain admission constraints.                                                                                                 |
| Bounded-window complete union counting                                                          | Already a DocSet protocol mechanism; new membership-only screen compares a 4,096-ID scratch window against full materialization.                                                                                                      | Not selected as a replacement for expanded-union materialization: smaller scratch, but substantial medium/dense x86 regressions. Sparse large-universe cases remain a memory-oriented experiment. |
| Direct encoded union accumulation / word batching                                               | [Prior direct-union experiment](../pattern-count-2026-10-02/rejected-direct-union.json) regressed default-index wildcard counts; earlier per-hit word-transition batching also regressed counts.                                      | Rejected candidates remain out. This does not reject the already-retained dense bitmap-word readers.                                                                                              |
| Deferred TF/positions and fused delta decoding                                                  | Canonical posting reader defers payloads until needed; SIMD decode integrates prefix sums. [Posting codecs](../../posting-codecs.md) and decoder tests.                                                                               | Retained. A new codec-specific fused decode/intersection loop is unmeasured and would need to remain in this owner.                                                                               |
| Block-Max MaxScore, required terms, conservative score bounds                                   | Existing ranked-OR window executor, essential/nonessential partition, required-term pruning, canonical query-order score reduction; exhaustive count path remains separate. [Lucene review](../../lucene-11-performance-research.md). | Retained. Ranked pruning is not an exact union-count optimization.                                                                                                                                |
| QFilter / BSR, byte filtering and state masks                                                   | Literature evidence is strongest for low-selectivity graph intersections; density/selectivity change the winner. Follow-up screens exact low-byte-prefiltered 4×4 on both hosts and sampled real blocks.                              | Byte filtering is now measured; full QFilter/BSR remains unmeasured. Its compressed representation and state-mask-to-posting-ordinal costs are separate work.                                     |
| Roaring run containers, recursive universe partitions, trie intersection, near-full complements | Standalone Roaring/EF/PEF codecs exist but are explicitly not wired as posting formats. Posting IDs also carry TF/position ordinals and merge-copy requirements.                                                                      | Not selected; representation-level research, not a drop-in kernel. No claim that full run/trie/complement formats were tested by this review.                                                     |
| Variable-size impact blocks and learned algorithm selection                                     | Papers report benefits dependent on query length, k, collection and layout. Summa has fixed posting blocks, block/group bounds and explicit planner capabilities.                                                                     | Unmeasured next-stage designs. Need separate byte/merge/metadata-budget contracts and a held-out workload, not a new scorer copied from a paper.                                                  |
| AVX-512 / VP2INTERSECT                                                                          | Benchmark host has AVX-512F but no VP2INTERSECT. Earlier AVX-512 bitmap expansion regressed integrated queries; that is not a measurement of AVX-512 intersection.                                                                    | Follow-up measures AVX-512F 16×16 comparisons on synthetic and real blocks. Native/emulated VP2INTERSECT remains distinct and untested; no ISA dispatch is justified from those screens alone.    |
| Intra-query multicore/GPU and approximate candidate filters                                     | Extra scheduling competes with existing request/segment concurrency. Bloom/approximate top-k candidates require exact verification or explicitly approximate semantics.                                                               | Not justified for this CPU exact-search path by the reviewed evidence; not benchmarked here.                                                                                                      |

## What the papers change in the review

[Inoue et al.](https://doi.org/10.14778/2735508.2735518) report different winners
for balanced and skewed arrays. Their thresholds are experimental choices, not
constants to copy into a different bounded posting reader. This motivated the
4×4 screen, including the reversed-skew controls that reveal asymmetric costs.
[Lemire, Boytsov and Kurz](https://arxiv.org/html/1401.6399v13) connect SIMD
decoding, skipping and intersection, and evaluate hybrid strategies on real
text collections. Summa already shares bounded decoding and intersection;
optimizing just one isolated kernel is insufficient.

[Roaring](https://doi.org/10.1002/spe.2402) motivates representation-sensitive
operations and deferring cardinality until multiway union accumulation finishes.
[Recursive universe partitioning](https://doi.org/10.1109/dcc50243.2021.00037)
aligns partitions in document space to enable direct word operations. Summa's
window protocol uses that useful principle without changing the on-disk
posting container. Neither source establishes that replacing all Summa posting
lists with a bitmap library would preserve its cost model.

[QFilter/BSR](https://doi.org/10.1145/3183713.3196924) reports sensitivity to
selectivity, skew and density; low-selectivity graph workloads do not establish
a text-ranking speedup. [Multi-set SIMD research](https://doi.org/10.1109/apsec.2017.65)
supports testing candidate-elimination order as well as individual comparisons.
The current executor already intersects the two rarest terms before probing
the rest, preserving scoring identities through an origin map.

[Algorithm selection](https://doi.org/10.1016/j.ipm.2023.103359),
[variable-block WAND](https://doi.org/10.1145/3077136.3080780), and the recent
[RISE study](https://doi.org/10.48550/arxiv.2606.07187) motivate workload-dependent
ranking strategies. RISE also discusses near-full complements and Lucene-style
windowed BMMS. Summa already has the latter class of execution; variable
partitioning and complement encodings remain distinct format experiments.
RISE's fully resident, single-thread collection results are not directly
comparable to Summa's concurrent service measurements.

## Experimental contract

Public Boolean/phrase/pattern queries reach the existing core query planners.
Ranked text conjunctions intersect bounded decoded blocks through
`query/scoring/conjunction.rs` and `structures/simd.rs`; the output carries both
posting ordinals so canonical TF scoring remains intact. Generic exact counts
use the DocSet batch/window protocols. Expanded constant-score unions belong to
`query/term_union.rs`. Ranked OR belongs to the existing windowed MaxScore owner.
Native and async execution share these owners; WASM uses the portable kernels.

For a kernel screen, inputs are strictly increasing unsigned IDs, at most 128
per block. Output must contain matching ordinal pairs in order, respect output
capacity, and resume without lost or duplicated matches. Zero output capacity
must leave cursors unchanged. Scratch stays fixed size, with no allocation in
the timed kernel. There are no persisted or wire format changes.

Screen branch-reduced 4-by-4 SIMD comparisons for balanced lists, exponential
search for skewed lists, and unsigned NEON prefix-count advancement against the
current production kernel and scalar merge. Use randomized fixtures, unsigned
boundaries, partial outputs, and mirrored skew. Compare the same source,
compiler and flags on each host, alternate method order, and retain raw samples.
An isolated kernel win is only a reason for an integrated experiment; it is not
sufficient to change dispatch or defaults. Format-changing run/trie/partition
representations require a separate storage design and lifecycle evaluation.

## Fresh results

[Complete table](table.md), [machine-readable summary](results.json),
[reproducer](summarize.py), and [provenance](provenance.json).
Ratios below are production-kernel median time divided by candidate time;
above 1 is faster. These are **decoded-block calls**, not whole-query gains.

| 128-pair output case   | ARM 4×4 SIMD | ARM repeat | x86 4×4 SIMD |
| ---------------------- | -----------: | ---------: | -----------: |
| Identical blocks       |       1.932× |     1.933× |       4.370× |
| Balanced, dense        |       1.554× |     1.531× |       3.006× |
| Balanced, sparse       |       1.080× |     1.067× |       2.944× |
| Left about 8× sparser  |       0.509× |     0.513× |       0.952× |
| Left about 64× sparser |       0.187× |     0.182× |       0.393× |

The current production convention puts the rarer term on the left. Reversed
skew is useful as a sensitivity control, but cannot be averaged into a promised
production improvement. Scalar merge wins on identical blocks, galloping wins
some extreme-skew controls, and neither wins generally. The prefix8 ARM
candidate is only 0.121–0.766× production outside the trivial one-result
identical case. Reject it as written, not NEON as a technology.

The next justified experiment is a **cheap balanced-block gate plus 4×4 SIMD**,
measured through the actual conjunction/phrase callers on unchanged indexes.
It must include the skewed controls, partial batches, query-score oracles,
and generic fallback/feature dispatch. A microbenchmark cannot measure caller
code layout, score-bound skipping, decoding, or TF work; none is silently
assumed to benefit. The follow-up implements and measures conditional candidates in isolated
checkouts; see its separate decision and complete table.

The union screen uses 1,048,576 document IDs, 2/8/32 lists, and sparse,
medium, dense and fully overlapping shapes. For example:

| Membership-only union count | ARM bitmap / windows µs | x86 bitmap / windows µs | Bitmap / window scratch bytes |
| --------------------------- | ----------------------: | ----------------------: | ----------------------------: |
| 8 sparse lists              |             3.33 / 1.05 |             6.98 / 2.74 |                 131,072 / 576 |
| 8 medium lists              |           18.12 / 18.03 |           28.95 / 43.73 |                 131,072 / 576 |
| 32 dense lists              |     3,218.08 / 4,936.43 |     4,093.69 / 5,681.03 |                 131,072 / 768 |

Sorting is best in the two-list very-sparse control. Full bitmaps dominate
many denser cases; complete heap merging is generally expensive. Windows can
reduce scratch and improve sparse cases, but scanning every child per occupied
window costs more at higher fanout. The current production materializer also
copies encoded dense bitmap words and can use a dominant-list count shortcut;
this decoded-ID screen excludes those advantages. It is not a direct timing of
`count_expanded`, and does not justify replacing or retuning that owner.

## Reproduction, memory and validation

Both hosts use Rust 1.98.1 / LLVM 22.1.8 with
`rustc --edition 2024 -O -C target-cpu=native`. Each host compares methods in
one binary with identical fixtures, alternating/rotating order. The x86
Cascade Lake VM uses CPU 2 with no competing benchmark/build jobs. ARM is an
OS-scheduled Apple M4 desktop, so timings are screening evidence; a second
independent invocation confirms the main direction. Cross-host absolute
latencies are not used as speedups.

The intersection source is extracted verbatim from the baseline's bounded
kernel and its helper functions, with feature-detection stubs and isolated
candidates appended. [Source](intersection-screen.rs.txt). There are 64
deterministic randomized fixture pairs per shape, capacities 1 and 128, and nine
20 ms samples per method/cell. Verification runs first: 5,529,600 ARM and
4,423,680 x86 kernel/fixture/capacity combinations pass. These include both
ordinal coordinates, nonzero starting suffixes, empty output, unsigned sign
boundaries, near-maximum IDs, lane-boundary tails, progress and output canaries.
The second ARM invocation repeats that verification. Per-call output scratch is
256 bytes plus cursor indices; no candidate allocates in its intersection loop.

[Union source](union-screen.rs.txt) checks all four methods against sorted,
deduplicated membership for 12 workload and five boundary fixtures per host.
Seven samples per method/cell run for at least 15 ms each. Scratch allocation
is timed, fixture generation is not. The table's scratch bounds count payload
storage, including the fixed window and cursor array, not allocator headers,
input lists, resident pages or the entire process. The conservative heap bound
accounts for vector capacity rounding. Whole-process resource logs are retained;
they aggregate all methods and must not be attributed to an individual method.
The union process peaks at 51.4 MiB RSS on ARM and 34.0 MiB on x86, including
fixtures and allocator retention.

To reproduce, copy either `.rs.txt` source to a temporary `.rs` path, compile
with the flags above, run the executable with stdout/stderr redirected, then
run `python3 summarize.py` from this directory for the checked-in samples.
`--verify-only` is available for the intersection screen. Source and raw-sample
hashes are in the provenance file; no dependency or public API was added.

`python3 scripts/check_search.py check` passes: formatting, strict Clippy,
**2,166 tests passed, 25 ignored**, async-only native compilation and standalone
broker compilation. [Check log](check.log). This re-exercises existing
intersection/union codec, ranking, count, filter and ordinal regressions.
The initial isolated screen changed no production Rust, so that stage did not
rerun the extended lifecycle/RPC suite or WASM build/tests. The linked follow-up
records the selected x86 implementation and its newer native/WASM validation. These screens do not replace either validation
requirement for a future production change.

The [October 2 total campaign](../review-2026-10-02/README.md) remains the latest
full 57-cell service comparison; it is historical evidence, not rerun here.
This review adds method-level evidence and an explicit list of unmeasured
opportunities. Format-changing approaches remain
open; “all worthwhile methods tested and adopted” is not an established result.

## Conditional-dispatch follow-up: experimental design

Requested after the initial review. The design below was recorded before
implementation; [the follow-up report](../posting-dispatch-2026-10-03/README.md)
contains the resulting measurements and selection decision.
The first candidate estimates density from each unconsumed block suffix's
document span and length. With at least 16 IDs on both sides and mean gaps
within a factor of two, use the ordinal-preserving 4×4 kernel; otherwise retain
the existing kernel. Use integer cross-products, not floating-point estimates.
The condition affects cost only: every path obeys the same sorted-pair and
resume contract. Feature availability remains checked and portable execution
retains its existing implementation. Fixed scratch, stored bytes, codecs,
scoring and query limits do not change.

Test the gate and shared tails at zero/full output, suffix boundaries and high
unsigned IDs before integration. Measure separate matched baseline/candidate
binaries through the actual searcher and phrase/conjunction callers on the
existing ARM 1M and x86 10M indexes, preserving count, top-k ID and score-bit
oracles and immutable-file inventories. Do not time compilation or profiling
alongside these phases. Include balanced/skewed conjunctions, multi-term AND,
phrases and unrelated union/term/pattern controls, plus process memory and CPU.
Inspect real gate hit rates and optimized code before any retention decision.

The next format-preserving screens cover SIMD galloping, byte-prefiltered
4×4 intersections and an AVX-512 comparison candidate where supported. A
byte-filter prototype is not automatically the complete published QFilter/BSR
method. Format-changing run/complement/variable-impact layouts remain separate
designs; this follow-up will not silently wire in a second posting format.
