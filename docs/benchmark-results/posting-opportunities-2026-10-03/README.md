# Remaining posting optimization experiments

The [October 4 follow-up](../posting-followup-2026-10-04/README.md) reduces the BSR
failure to a candidate-compaction contract violation, fixes it, and measures
batching on dedicated ARM hardware. This report preserves the October 3
experiments and their original source/binary provenance.

## Decision and measured result

Retain **x86 density-gated later-clause batching** in the existing conjunction
scorer. Keep the earlier AVX2 dispatcher as the primitive; corrected BSR still
fails the corpus oracle.
ARM/WASM keep individual later-clause seeks. ARM prototype timings are
inconclusive under local timing drift. No posting format or union policy changes.

Ratios below are baseline/candidate: above 1 means faster. The baseline already
includes the previously retained dispatcher. Totals sum per-query medians and
describe this suite, not a production traffic mix. Multi-clause columns use the
26 three- to six-term queries; four duplicate/missing-term controls are reported
separately in the complete table. The final campaign measures the exact source
with batching restricted to x86.

| Host/index    | Multi-clause top-10 | Multi-clause top-100 | All 199 top-10 | All 199 top-100 | All counts |
| ------------- | ------------------: | -------------------: | -------------: | --------------: | ---------: |
| x86/default   |              1.121× |               1.121× |         1.107× |          1.078× |     1.004× |
| x86/rgb-pairs |              1.158× |               1.168× |         1.155× |          1.146× |     0.990× |

Read the [complete table](table.md) for every primitive screen, integrated
candidate, query family, paired round and process resource result. The
[machine-readable results](results.json), [provenance](provenance.json),
[verified evidence manifest](evidence-manifest.json) and `summarize.py` retain
the raw samples, independent audits and reproduction inputs.

| Candidate                                      | Decision             | Evidence                                                                                                   |
| ---------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------- |
| x86 density-gated later-clause batching        | Retain               | Exact final-source queries improve with bounded scratch and exact results                                  |
| ARM later-clause batching                      | Do not select        | Confirmation drifts substantially even in unchanged controls; scalar membership remains                    |
| Size-only batching                             | Superseded           | Positive totals hide skewed-query regressions; local density gating addresses these                        |
| Adaptive later-clause order                    | Reject               | ARM essentially flat initially; small x86 changes are comparable to unchanged count-control drift          |
| Initial packed/streaming BSR, including POPCNT | Incorrect prototypes | Terminal state groups lose cross-block tails; original primitive timing rows are invalid adoption evidence |
| Corrected POPCNT stream                        | Reject               | Unit checks pass, but the full-corpus ranked oracle fails; no query timing is accepted                     |
| Local input swapping                           | Reject               | Most real traces regress after endpoint arithmetic and ordinal swaps                                       |
| AVX-512 mask emulation                         | Reject               | Trace improvements do not survive the integrated comparison against the gated AVX2 caller                  |

### Final CPU, memory and tradeoffs

Ranked process CPU ratios are 1.078–1.149×; peak RSS changes are
+0.69 to +0.81 MiB. CPU includes startup and warmups; RSS
includes mapped index pages. The conjunction path adds one 256-byte fixed pair
buffer and no query allocation.

The gain is not uniform. These are the six lowest ranked family ratios for the
retained candidate:

| Index   | Family             | Limit |  Ratio | Paired ratios |
| ------- | ------------------ | ----: | -----: | ------------- |
| default | or_high_med        |    10 | 0.947× | 0.931/0.967   |
| default | low_phrase         |   100 | 0.955× | 0.954/0.962   |
| default | or_high_med        |   100 | 0.963× | 0.969/0.962   |
| default | high_phrase        |    10 | 0.968× | 0.961/0.980   |
| default | regex              |   100 | 0.982× | 1.002/0.975   |
| default | high_sloppy_phrase |   100 | 0.983× | 1.004/0.968   |

Count-only traversal does not use the new conjunction path. Control changes
and some unaffected query changes are not claimed as algorithmic count gains.
Individual slowdowns and paired ratios remain in `results.json`.

### BSR correctness outcome

The initial state-group prototypes discard cross-block tails. After fixing that
bug, the POPCNT stream passes the new primitive regression and 17 conjunction
unit tests, but still fails the full-corpus oracle on `+the +of +in +and`.
It replaces one expected top-100 document with another; scores for the 99 shared
documents remain unchanged. This is an unresolved prototype interaction, not a
validated BSR speedup. The runner stops before timings, and the prototype is
rejected. The single-query reproducer and full expected/actual vectors are kept.
No failing BSR implementation enters the retained source.

### ARM prototype confirmation (not selected)

Both runs use the same binaries. Large differences between runs, including
unchanged count controls, prevent a reliable ARM performance claim.

| Run/index      | 26-query top-10 | 26-query top-100 | 199-query top-10 | 199-query top-100 | Counts |
| -------------- | --------------: | ---------------: | ---------------: | ----------------: | -----: |
| final/bitmap   |          1.109× |           1.104× |           1.070× |            1.042× | 1.009× |
| final/rgb      |          1.108× |           1.123× |           1.069× |            1.062× | 1.004× |
| confirm/bitmap |          1.220× |           1.321× |           1.132× |            1.196× | 0.973× |
| confirm/rgb    |          0.924× |           1.088× |           0.922× |            1.043× | 0.814× |

Completed campaigns pass **5,654 exhaustive audits** and **260,508 recorded query
executions** (warmups excluded). The rejected BSR audit is recorded separately. Immutable index inventories match.
Both cloud VMs were independently verified `TERMINATED` after evidence collection.

## Experimental contract (before measurement)

This campaign compares against the retained October 3 dispatcher, including the
uncommitted `simd.rs` with SHA-256
`f4794542315ac3d30875aadc0345322a5113698064fda69388d6d4caf4b2b962`.
It does not compare against the older prefix-count kernel at HEAD.
Candidates stay in isolated checkouts until correctness and performance pass.

1. **Adaptive later-clause order.** Keep the two DF-selected lead terms fixed.
   Order the remaining terms by an exponentially smoothed observed survivor
   fraction, updated once per 128-candidate batch. Cursor indices and original TF
   row identities remain unchanged; canonical query-order score reduction stays
   unchanged. The cost is a bounded per-term integer array, a small stable sort,
   and one integer ratio per visited later clause. Test real three- to six-term
   queries, including duplicates, clause permutations and absent terms, against
   the independent exhaustive count/top-k oracle on ARM and x86.
2. **Base/state conversion.** Pack each decoded suffix into 32-ID base/state
   groups and recover both original ordinals with state-mask ranks. Charge
   packing to the measured call. A streaming variant avoids full temporary
   arrays. This tests the suitability of transient BSR for today's decoded-block
   interface, not an on-disk BSR format or the complete QFilter algorithm.
3. **AVX-512 intersection-mask emulation.** Test the six-permutation algorithm
   from Díez-Cañas, with exact right-ordinal recovery and bounded resumable output.
   Compare full primitive cost against the current AVX2 dispatcher, with real
   posting traces and randomized unsigned boundary cases. The available x86 host
   supports AVX-512F, but lacks native VP2INTERSECT.

The shared primitive is owned by `structures/simd.rs`; ranked conjunctions call
it. Phrase traversal uses the separate `PostingIntersection` implementation.
The adaptive candidate belongs solely to
`query/scoring/conjunction.rs` and uses the existing scorer and scratch. No
persisted bytes, merge rules, public APIs, sync/async semantics, exact membership,
TF identity, position identity, or score bits may change. Scratch stays bounded.
No normal-query corruption scans or new validation are introduced.

Warm query latency and resource use are compared within each host and immutable
index. Build/profiling work is excluded from each timed host. Synthetic and trace
results screen candidates; they do not establish service-level gains. Cold I/O,
concurrent throughput and novel persisted representations are outside this work.

A fourth candidate follows from tracing the actual multi-term caller: **batch
later-clause membership** when at least 16 candidates remain, using the existing
ordinal-pair kernel against the cursor's loaded block. Smaller batches retain
individual seeks. This avoids repeating the general seek protocol for each
candidate; it may lose when eager block work outweighs sparse probes. It reuses
one 128-pair stack array and preserves candidate origins and cursor TF rows.
This is tested separately from adaptive clause ordering so their effects are
not conflated.

The AVX-512 emulation passes the trace screen (about 12–14% faster on the four
retained traces), so it advances to an integrated comparison on top of the
batched-conjunction candidate. Generic builds and the same actual query callers
must establish whether mask savings survive dispatch, instruction-cache and
AVX-512 frequency effects. ARM keeps its current SIMD implementation.

## Prior art and what this tests

Machine Library supplied the relevant text of [Han, Zou and Yu's QFilter/BSR paper](https://doi.org/10.1145/3183713.3196924)
and [Song, Yang and Jiang's multiple-set intersection study](https://doi.org/10.1109/apsec.2017.65).
The former motivates base/state grouping; the latter separates multiway
elimination policy from the search primitive. Our adaptive survivor ordering is
an engineering hypothesis inspired by that distinction, not a reproduction of
all the paper's algorithms. The existing DF-selected lead pair stays fixed.

Machine Library had only metadata and the abstract for
[Díez-Cañas's VP2INTERSECT emulation](https://doi.org/10.48550/arxiv.2112.06342).
The [primary full text](https://arxiv.org/html/2112.06342v2) supplies the
six-permutation method: rotate both inputs, compare all lanes, and undo the
first input's rotations in the result mask. The experiment uses AVX-512F masked
comparisons, then recovers each matching right ordinal with exact equality.
This is an emulation on Cascade Lake, not a measurement of native VP2INTERSECT.

## Validation and costs

The new behavioral regression passes on the starting implementation, the
size-only prototype and the density-gated candidate. It covers four/six clauses, clause permutations, duplicate
clauses, document filters, nonconstant TFs and lengths, five posting codecs,
block gaps and partial batches. Sync, async and counted traversal must produce
the independent oracle's exact IDs, score bits and count.

The final x86-only density-gated source passes the native search harness: **2,169 tests,
25 ignored**, strict Clippy, native without sync and standalone broker checks.
The WASM release build and **41 JavaScript tests** pass. Linux adds 17
conjunction tests, the cross-block regression and strict core Clippy. The integrated wide candidate also
passes those 17 conjunction tests and the dispatcher regression. Extended
lifecycle/RPC checks are not rerun because these boundaries do not change.

Initial single-block validation checks **4,423,680 cases on ARM** and **6,635,520 on x86**;
the count differs because x86 includes two additional AVX-512 variants. A
separate full-unsigned-domain matrix adds 204,800/307,200 cases, including zero,
sign-bit crossings, `u32::MAX`, empty output, partial resumes and canaries.
These are kernel/fixture/capacity combinations, not distinct real queries.
Real-trace replay verifies all 7,118 individual calls before timing. These
initial checks did not cover supplying the next decoded block; the follow-up
below records the BSR failure and the stronger regression.

Batched later-clause checks add one 256-byte stack pair buffer and no query
allocation. BSR packing uses two fixed base/state/ordinal arrays (2,320 bytes on
these 64-bit hosts); streaming BSR avoids those arrays. Conversion is inside
the measured call. No persistent BSR metadata or corpus-sized scratch is added.
The generic x86 unpruned conjunction function grows from 8,041 to 9,514 bytes
with size-only batching; its wide variant is 8,972 bytes. The final density-gated
variants are 9,768 bytes (AVX2) and 9,238 bytes (AVX-512). These are individual
function sizes, not a cache-miss profile or total executable size.

The asymmetric synthetic cases also motivate a small **orientation screen**:
when both remaining suffixes contain at least eight IDs and the right mean gap
is more than twice the left mean gap, swap the primitive's inputs and swap the
returned ordinal coordinates back. This tests local probe orientation, distinct
from changing multiway clause order. The additional endpoint arithmetic and
pair swaps are included; neither changes exact membership or original ordinals.

The ARM per-family results expose a cost hidden by the positive suite total:
size-only batching slows skewed conjunctions. The refined candidate therefore
checks local density after aligning the later cursor to the first candidate.
It batches only when the candidate density predicts at least 32 candidates per
current posting-block span (one quarter of a full 128-ID block), retaining the
16-candidate minimum. Products use `u64`; this is a cost hint, not a membership
condition. It also guards the multi-term branch with `!PRUNE`, since pruned
conjunctions are admitted only for two terms. This keeps the unreachable new
body out of that specialization. Both changes are included in the final matched measurements.

The BSR screen additionally checks explicit x86 `POPCNT` dispatch for both
packing and streaming, plus the density-gated streaming policy. Generic release
flags do not imply this feature; assembly confirms hardware population counts
in the specialized rank recovery. The extra dispatch cost is included. This
prevents confusing a portable instruction-selection penalty with the cost of
conversion itself. Its separate correctness matrix repeats 6,635,520 cases.
The density-gated POPCNT stream improves the four real trace totals by
3–13% in the initial single-block screen, motivating a separate 199-query x86
ABCCBA comparison against the starting dispatcher and density-gated conjunctions.
Those initial BSR kernels subsequently fail the cross-block contract, so their
primitive timings are not valid adoption evidence. The corrected candidate uses
the density-gated conjunction source. The clean prototype adds a private
streaming helper under the existing AVX2 dispatch, requires POPCNT at runtime,
and retains the original fallback. It uses scalar state masks, not packed arrays.
ARM and portable code are unchanged. Packed BSR remains slower.

## Reproduction

The final table and JSON summary are generated from split ZIP evidence by
`summarize.py`. The reader checks the archive hash, every member hash, complete
phases, exact audit/count/rank/plan equality, sample counts and immutable index
inventories. Concatenate `evidence.zip.001`, `.002`, … in numeric order to extract
it manually. Each part is below the repository's 1 MB file limit.

The archive contains the complete standalone Rust screens; compile `screen.rs`
(or `orientation-screen.rs`) with `rustc --edition 2024 -O`, run `--verify-only`,
then run without arguments for synthetic fixtures or with the four `trace-*.txt`
paths for replay. `popcnt-screen.rs` is x86-only. Corpus-independent screens need
no Cargo dependencies. Traces are sampled from the earlier ARM 1M corpus, so
x86 replay does not establish the 10M corpus's call distribution.

For real queries, check out `8f9c3f66` in an isolated `source` directory, copy
`baseline-simd.rs` and `baseline-conjunction.rs` into their owning source paths,
and copy `probe.rs` over the existing `searchbench_http/diagnose.rs` example.
Build `summa-server --example searchbench_http` in release mode with empty
`RUSTFLAGS` and preserve the baseline binary. For the final retained variant,
copy `x86-gated-conjunction.rs` to the conjunction source and `retained-simd.rs`
to the SIMD source; the latter adds the cross-block test to the baseline kernel.
`selected-tests.rs` supplies the TF-row regression. The earlier portable prototype
is `gated-conjunction.rs`; `wide.py source` adds the rejected AVX-512 primitive.
`bsr-popcnt-simd.rs` is the rejected tail-corrected BSR source and must not be
confused with the retained SIMD file. No benchmark scorer or writer is introduced.

`run.py` measures the 30-query adaptive/size-only candidates; `run-broad.py` and
`run-final.py` retain the initial ARM broad comparisons. `run-gated.py` measures
199 queries with baseline/refined/refined+wide binaries (wide only on x86).
`run-confirm.py` repeats the ARM prototype; `run-selected.py` measures the exact
final x86 source with ABBA. `run-bsr.py` aborts on its corpus audit and has no
accepted timing results. Runners record paths, binaries and query hashes. Reuse the immutable corpus
inventories and matching fixture paths; index payloads are not included in the
archive. Cloud-specific build scripts preserve the original cache paths and
should be adjusted for a different build host. All timing runs exclude builds,
profiling and other benchmark jobs on the measurement host.

## Remaining scope

The [initial methods table](../posting-methods-2026-10-03/table.md) covers the
union and earlier intersection screens; the [dispatcher table](../posting-dispatch-2026-10-03/table.md)
covers the previously retained AVX2 change. Ratios from separate campaigns must
not be multiplied: this experiment starts from that retained dispatcher.
Union policies and posting formats are unchanged in this campaign.

Transient BSR with ordinal recovery does not test full QFilter or an encoded BSR
reader. Fused codec/intersection decoding, variable impact partitions, run or
complement encodings, trie representations, native VP2INTERSECT and other CPU
families remain unmeasured. Cold I/O, concurrent service throughput and HTTP
hydration also remain outside this warm query campaign. The evidence narrows the
next choices; it does not establish that every useful published method has been
implemented or exhausted.

## Follow-up correctness finding and architecture scope

The initial streaming BSR integration fails two conjunction regressions: a state
group can extend past the end of one decoded block, and consuming that whole
group from the other input loses matches against the next supplied block. The
single-block membership/ordinal/capacity oracle did not exercise this contract.
The initial BSR primitive rows, including POPCNT variants, are therefore marked
invalid for cross-block use; passing that narrower matrix does not establish
complete correctness. Those prototypes never entered the working-tree scorer.

The corrected stream retains the other input's unmatched tail when a terminal
state group straddles a block boundary. A new regression feeds subsequent blocks
with widths 33/41/64/127/128, three output capacities, different densities and
unsigned boundary bases, checking global ordinal pairs against an independent
whole-list oracle. The retained implementation passes it. The corrected BSR
variant must pass it and the conjunction regressions before new query timings.
The original failure log is retained in the evidence. The standalone
`invalid-bsr-tail-repro.rs` reproduces the same problem in all three initial
generic BSR policies: with left blocks `0..41`, then `41..82`, and a right
block `0..128`, they resume the right side at 64 and return only 18 of the
next block's 41 matches. The retained kernel resumes at 40, safely retains the tail, and returns all 41.

The ARM confirmation uses the same binaries and immutable indexes, but shows
large drift even in unchanged count controls. A post-run process snapshot
showed substantial interactive-application CPU load, but was not a continuous
load/frequency profile and does not establish the cause of each outlier. These
results are inconclusive for selecting ARM batching. The final source limits later-clause batching to x86; ARM/WASM
keep individual membership seeks. The unreachable multi-term body remains
excluded from the two-term pruned specialization on all targets. A fresh x86
comparison uses the exact final-scope source, with and without corrected BSR.

The tail-corrected BSR integration passes its unit checks, then fails the ranked
oracle for `+the +of +in +and` after 185 successful query audits. One expected
top-100 ID is replaced; score bits for the other 99 IDs agree. The runner aborts
before timing, and this prototype is rejected with that interaction unresolved.
The single-query fixture, exact source/binary hashes, full expected/actual vectors
and failed audit are preserved. The final successful comparison measures only
the starting dispatcher and the retained x86 batching source; no BSR query
speedup is claimed. This rejects the tested implementation, not the BSR family.
