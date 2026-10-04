# Conditional posting intersection experiments

October 3, 2026. Follow-up to the [Machine Library methods review](../posting-methods-2026-10-03/README.md), baseline `8f9c3f66`.

## Experimental contract and candidates

The selected implementation changes only the bounded decoded-block intersection
kernel on x86 with AVX2. Earlier candidates remain isolated, reproducible
experiments. ARM, WASM and non-AVX2 kernels retain their existing implementation. No stored bytes, ordinal identities, scoring reductions,
query budgets, public APIs, or union policies change. Both inputs contain at
most 128 strictly increasing unsigned IDs. Output contains both original posting
ordinals; partial output resumes without losing or duplicating a match. Output
scratch remains 256 bytes, with no allocation inside the kernel.

[Candidate B](candidate-b.patch) estimates each remaining suffix's mean document gap using integer
cross-products. It selects SSE2/NEON 4×4 comparisons when both suffixes contain
at least 16 IDs and their estimated gaps are within a factor of two; otherwise
it uses the baseline implementation. Architecture features remain checked.
The condition is a cost hint, never a correctness condition.

[Candidate C](candidate-c.patch) tests the effect of forcing the balanced helper to inline. On x86
it additionally uses grouped SIMD galloping for the nonbalanced suffixes when
AVX2 is available. Its final incomplete groups use the existing shared tail.
On ARM it changes only the inlining request relative to B. Compiler output,
not the source annotation alone, determines where the kernel actually inlines.

Standalone screens also compare scalar merge, scalar galloping, unconditional
4×4, grouped SIMD galloping, low-byte-prefiltered 4×4, ARM prefix counting, and
AVX-512F 16×16 comparisons. Every possible byte-filter match is verified with
full 32-bit equality. These are format-preserving prototypes, **not** complete
QFilter/BSR or VP2INTERSECT implementations.

## Measurement design

The broad suite contains 170 queries: admitted term, AND, OR, exact/sloppy
phrase, prefix, wildcard and regex queries, plus five AND boundary/multi-term
controls. It compares A/B/B/A on top-10, top-100 and exact count, with five warmups
and 15 recorded repetitions per query and phase. Parsing and output conversion
are outside the timed section. Ranked timings call the actual synchronous
searcher; count timings use the canonical count collector. HTTP, hydration,
cold starts and concurrent service throughput are outside this experiment.

A separate 30-query set targets three- to six-term conjunctions, balanced/skewed
terms, clause order, duplicates, and absent terms. It compares A/B/C/C/B/A with
five warmups and 21 recorded repetitions. All variants first run the existing
independent exhaustive count/top-100 audit; retained records are also checked
for exact optimized/exhaustive count equality and equality across binaries. Every measured repetition and phase
also compares complete ranked document IDs and score bits, or the exact count
for count requests, plus parsed plans. Ranked probe rows use zero as a count
placeholder; they do not assert an exact-count contract for ranked search.

ARM uses the existing 1M-document bitmap and RGB indexes; x86 uses the existing
10M-document default and RGB-plus-pairs indexes. Full file sizes and SHA-256
inventories are compared before and after each campaign, excluding only the
reader lock file. Results are compared within a host/index, never across
architectures or between different corpora/layouts.

Both hosts use Rust 1.98.1 / LLVM 22.1.8. ARM release builds use empty
`RUSTFLAGS`; the dedicated Cascade Lake x86 host uses `-C target-cpu=native`
and CPU 2. Each host's baseline and candidates use identical flags and probe.
Linux compilation runs on a separate VM. No profiling, tracing, builds or other
benchmark jobs run alongside timed queries on the same measurement host.

Untimed diagnostic binaries count shared-kernel calls, gate hits and the
bitmap/frequency-pruned bypasses. A separate tracing binary samples every 127th
kernel call, bounded to 20,000 trace rows per process. It records input blocks,
starting ordinals and output capacity. Replay compares every kernel to an
independent membership/ordinal oracle before timing. Traces come from the ARM
1M indexes; x86 replay of those traces tests ISA costs, not the 10M workload's
block distribution.

## Results and decision

The first conditional dispatcher does not establish a useful broad-query gain.
The sum of per-query medians gives these ratios (A/B; above 1 is faster):

| Host / index    | Top-10 | Top-100 |  Count |
| --------------- | -----: | ------: | -----: |
| ARM / bitmap    | 1.003× |  0.991× | 0.980× |
| ARM / RGB       | 0.995× |  0.995× | 0.985× |
| x86 / default   | 0.998× |  1.000× | 0.990× |
| x86 / RGB+pairs | 1.005× |  1.000× | 1.001× |

These totals describe the selected query suite, not a production traffic mix.
Individual outliers are retained rather than reported as wins. ARM's plain
high/low AND top-10 ratio is 1.983×, but its paired rounds are 2.928/1.030.
x86 RGB high-phrase count is 1.182×, but its rounds are 1.465/0.986. A refinement
artifact was staged near the early RGB baseline phases; conservatively treat
that first-round result as potentially disturbed. Neither outlier supports
changing dispatch. Unchanged-path controls also show drift and code-layout
sensitivity.

The 30-query multi-term comparison isolates the meaningful gain: C improves
x86 ranked conjunctions, while ARM does not show enough benefit to justify a
new kernel there. Ratios below use the complete 30-query set.

| Host / index    | B top-10 / top-100 | C top-10 / top-100 |
| --------------- | -----------------: | -----------------: |
| ARM / bitmap    |    1.002× / 1.004× |    1.007× / 1.002× |
| ARM / RGB       |    1.023× / 1.015× |    1.025× / 1.011× |
| x86 / default   |    1.016× / 1.014× |    1.092× / 1.089× |
| x86 / RGB+pairs |    1.044× / 1.035× |    1.111× / 1.106× |

For x86 C, the top-10 forward/reverse ratios are 1.094/1.091 on default and
1.101/1.121 on RGB; top-100 gives 1.092/1.085 and 1.106/1.107. Unchanged AND count
controls are 1.006× and 1.009× overall. The final generic-build comparison below confirms the cleaned x86 variant D.

## Dispatch ownership and cost

The public parser/planner keeps query semantics in `core/query`. The two direct
shared-kernel callers are ranked conjunction batching in
`query/scoring/conjunction.rs` and the cached `PostingIntersection` in
`structures/postings/posting.rs`, used by phrases. Ranked two-term conjunctions
normally bypass this kernel through direct bitmap probes or frequency-bound
competitive probes. Multi-term ranked conjunctions use the unpruned pair path,
then retain TF row identities while probing the remaining terms. Generic AND
counts use a different DocSet path.

Phrase intersection caches ordinal pairs per overlapping block pair, returns
already aligned heads directly, and uses seek-driven alignment for skewed DFs.
Thus a phrase can confirm many positions with very few shared-kernel calls.
The ARM plain top-10 suite has only 15 shared calls across its 12 high-frequency
exact phrases; RGB has 1,202. Their exact counts have 111,723 and 105,530 calls.
The 30-query multi-term top-10 suite has 114,406 shared calls on plain and 109,736
on RGB. Gate hits are 74,993 and 66,436 respectively. Untimed counters, including
bypass counts and other limits, are retained in the evidence archive.

The first gate adds four endpoint loads, two span/length cross-products and
comparisons when both tails are long enough. With 128-ID inputs even maximum
unsigned spans fit comfortably in `u64`; the gate cannot affect membership.
It is not free on rejected skewed blocks. The synthetic screen sees several
percent overhead there. Suffix length alone is a poor skew estimator because
both decoded blocks can contain 128 IDs despite very different document gaps.
Global DF alone also misses the dense local runs created by RGB.

ARM assembly shows B's gate in the multi-term caller and calls to either kernel.
C's forced helper inlining instead leaves the whole dispatcher out of line in
that caller. The caller sizes are 5,392 bytes for A, 5,512 for B, and 5,312 for C;
these are function sizes, not total program code or an instruction-cache profile.
The experiment does not establish that hoisting a DF-only decision or adding a
learned selector would improve this cost model.

On x86, A keeps the AVX2 intersection loop in the multi-term caller and calls
out for its tail. Both B and C call the dispatcher out of line. The caller sizes
are 7,440, 7,248 and 7,248 bytes. This is another reason to measure compiled
callers instead of extrapolating from a standalone function-pointer benchmark.

## Validation and remaining boundaries

The expanded eight-kernel screen passes 8,847,360 randomized
kernel/fixture/capacity cases **on each architecture**. Adding the exact refined
dispatch and its shared tail passes 9,953,280 cases per architecture. The latter
also passes with the test harness's NEON/AVX2 availability stubs forced off.
These are overlapping test matrices, not millions of distinct real queries.
They check empty inputs/output, partial batches, nonzero suffixes, lane-boundary
lengths, unsigned sign/max boundaries, both ordinal coordinates, progress,
resumption and output canaries. AVX-512F is tested only on the supporting x86
host; native VP2INTERSECT is unavailable there.

There are 7,118 sampled real calls across four traces. Replay validates nine
kernels on ARM and ten on x86, with 11 rotating/reversed-order timing passes.
The synthetic screen uses nine rotating/reversed-order passes, 64 randomized
pairs per shape, capacities 1/128 and at least 20 ms per sample. Replay samples
run for at least 40 ms. Scalar/galloping/filter prototypes in these screens use
a scalar tail; integrated B/C use the existing shared tail. Do not silently
equate their compiled costs.

The cleaned D implementation passes `python3 scripts/check_search.py check`:
strict Clippy, **2,167 native tests passed, 25 ignored**, native without sync,
and standalone broker compilation. On Linux, 247 posting-related release tests
and strict core Clippy pass. The WASM release build and all 41 JavaScript tests
pass after installing the locked test dependencies; the first test invocation
had no `vitest` executable. Extended lifecycle/RPC checks are not rerun because
this changes only the native primitive, not a lifecycle or wire boundary.

D additionally passes **11,059,200 kernel/fixture/capacity cases** on ARM and
on x86 with both generic and native flags. Forced NEON/AVX2-disabled dispatch
also passes this expanded matrix. These repeat many earlier cases and are not
additive distinct coverage counts.

Complete QFilter/BSR, run/complement/trie representations, variable impact blocks,
codec-specific fused decode/intersection, and native/emulated VP2INTERSECT remain
separate unmeasured designs. They need codec/ordinal, immutable merge, residency
and hardware contracts, not just a kernel substitution. Intra-query GPU work and
approximate candidate selection also remain outside this exact CPU experiment.
The earlier union screens and existing encoded-word/dominant-count policies
remain the union evidence; this follow-up does not rerun the full service matrix
or claim every published algorithm has been tested.

## Reproduction and evidence

The [complete table](table.md) and [machine-readable summary](results.json) are generated by
`python3 summarize.py` from the retained `evidence.zip.001`, `.002`, … parts.
The script reads the parts directly; concatenate them in numeric order to extract
the ZIP manually. Each part stays below the repository’s 1 MB file limit. The script refuses
incomplete runs, mismatched audit/count/ranking results, missing samples or
changed index inventories. [The manifest](evidence-manifest.json) hashes each archive member;
[provenance](provenance.json) records compiler flags and binary/archive hashes. Query files,
all raw timings, full result vectors, audits, resource logs, immutable file
inventories, compiler output, assembly and diagnostic traces are retained.

To rebuild the experiment, extract the archive into an isolated working folder
and check out baseline `8f9c3f66` into its `source` directory. Copy `probe.rs`
over `source/summa-server/examples/searchbench_http/diagnose.rs` in every timing
variant. Build and retain A before applying `patch.py source` for B. Restore the
baseline `summa-core/src/structures/simd.rs`, then apply `refine.py source` for C.
Use the host flags above and preserve a separate binary per variant. The build
scripts record the actual build-host commands; adjust only local paths when
using the same retained corpus. `run.py` and `run-multi.py` record index paths,
phase order and iteration counts.

Compile `screen.rs`, `screen-refined.rs`, `screen-disabled.rs`, and `replay.rs`
with the standalone flags above. The three screens support `--verify-only`;
replay takes the four `trace-*.txt` files. The disabled screen forces only the
NEON/AVX2 availability stubs off: its explicitly invoked ISA-specific comparison
candidates still require the host that supports them. It is not a proof of
execution on an unsupported CPU.

The untimed instrumentation is isolated from every measured binary. The trace
patch is included for provenance and requires a diagnostic source checkout;
its query generator and diagnostic driver are also retained. The final archive
contains source and evidence, not executable binaries or rebuilt indexes.
`patch-selected.py` prepares D from the baseline, or apply `candidate-d.patch`
in an isolated checkout; `run-selected.py` records its generic-build comparison.
The preparation script was renamed from `select.py` when packaging evidence to
avoid shadowing Python’s standard `select` module on macOS.

## Selected generic-build implementation

**Retain [D](candidate-d.patch)**, the cleaned x86-only balanced 4×4 / grouped SIMD-galloping
implementation. AVX2 is checked at the primitive boundary; the bulk loops live
inside its target-feature function and directly reuse the existing eight-lane
equality helper. The previous x86 prefix-count loop is removed. Helpers remain
private in `structures/simd.rs`; no scorer, planner setting, format or public
API is added. The regression checks original ordinals, empty output, partial
resumption and unsigned/density boundaries against an independent oracle and
the existing portable path.

The final comparison combines the query sets into **199 distinct queries** and
uses A/B/B/A with five warmups and seven recorded repetitions. Both binaries
use empty `RUSTFLAGS`. The candidate's SIMD source SHA-256 matches the validated
working tree. No binary transfer or extraction overlaps these timing phases.
The same exhaustive oracles and immutable-file checks pass.

| Generic x86 corpus | Multi-term AND top-10 | Multi-term AND top-100 | All 199 top-10 | All 199 top-100 | All 199 count |
| ------------------ | --------------------: | ---------------------: | -------------: | --------------: | ------------: |
| Default            |                1.064× |                 1.063× |         1.045× |          1.031× |        1.022× |
| RGB+pairs          |                1.076× |                 1.080× |         1.061× |          1.057× |        1.020× |

These correspond to **5.9–7.4% lower multi-term AND latency** and roughly
**3–6% lower ranked-suite latency**. The complete-suite top-10 forward/reverse
ratios are 1.046/1.044 on default and 1.061/1.061 on RGB; top-100 gives
1.031/1.031 and 1.058/1.056. Ordinary multi-term AND counts, which bypass the
changed kernel, remain essentially flat (1.002× on each layout).

This is not a claim of universal improvement. Default low-frequency exact
phrase top-100 regresses 3.5% (paired speedups 0.976/0.956), and some other exact
phrase groups regress around 0.5–1.3%; RGB low-phrase top-100 is 0.981× with
mixed paired direction. Unchanged OR controls also move by up to a few percent.
Keep these cases as review watchpoints, rather than hiding them in the total.
High/medium exact phrase counts improve 1.7–2.3% on default; sloppy-phrase counts
improve about 1.6–4.7% across the two layouts. The selected change is supported
by repeatable multi-term and suite gains, not every individual family.

Whole-process ranked CPU falls about 2.9–5.4% across layouts/limits. Mean paired
peak RSS differs by less than 0.3 MiB at approximately 2.1–2.3 GiB, including
mapped index pages. The kernel adds no heap allocation and retains the 256-byte
output scratch contract. Counts, full result IDs, score bits and plans match
between binaries; all full-file SHA-256 inventories remain unchanged.

There are **2,516 exhaustive query audits** across all integrated campaigns,
and 201,192 recorded query executions, excluding warmups. The report generator
checks the retained audit records as well as the per-phase result vectors.
The scheduled safety shutdown stopped the VMs after completion; the benchmark
VM was briefly restarted only to retrieve final artifacts, then stopped again.
No benchmark timings come from the retrieval restart.
