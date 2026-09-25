# Reordering performance review — 2026-09-05

Review base: `98c86059`, Rust 1.98.1 / LLVM 22.1.8. Measurements cover Apple
M4/macOS and Intel Xeon 8581C/Linux. This review covers the BP computation,
forward graph construction, BMP record/block rewrites, and chunked-text
reorder. It does not establish production latency or Linux cold-I/O behavior.

## Entry points and invariants

`IndexWriter::reorder` and the server optimizer reach
`SegmentManager::reorder_single_segment`, which owns source/output claims and
calls `segment::reorder::reorder_segment`. Merge-time BMP reorder reaches the
same `reorder_bmp_field` through `segment::merger::sparse`. Explicit record or
block granularity bypasses the coherence estimator; `Auto` chooses per field.
The shared background pool and reorder gate remain the scheduling owners.

BMP record reorder builds a padding-aware, compact-term CSR, refines a
permutation, and transposes postings through bounded sorted windows. Block
reorder bisects header-term sets and copies block payloads, remapping document
IDs and rebuilding grids. Chunked-text reorder builds its own CSR in
`text_reorder`, uses the same BP kernel, then permutes virtual chunk IDs,
postings, positions and chunk maps. Document IDs, stores and unrelated fields
retain their semantics. ANN artifacts are never trained by these paths.

The kernel is shared by native builds with/without sync, with a sequential
non-native implementation. The segment reorder adapters are native-only;
portable compilation still covers the graph module. There is no protocol or
persisted format change in this pass (BMP V19 and text/chunk formats remain
owned by their existing encoders).

For `P` CSR postings, `N` entities, `T` compact terms, and `L` admitted degree
lanes, the graph has `4P + 8(N+1)` CSR bytes, `20N` bytes of entity scratch
(order, gains, output and `usize` ranks on this 64-bit host), and approximately
`8.1875TL` degree/initialization bytes. Repeated refinements and levels revisit
postings; this is not an O(P) algorithm overall. Graph scratch is allocated
once and lanes are reused across level-synchronized partitions. The budgeted
gain cache described below adds at most `8TL` heap bytes.

## Implemented: remove repeated gain arithmetic

Previously each posting evaluated two log lookups (calling `log2` when a
degree exceeded 4095) and a division. During a refinement, these values depend
only on the term degrees and the document's half. The kernel now computes the
two signed contributions once per active term and gathers them in the original
document-term order. Half directions, both subtraction operations, division,
and each floating-point addition retain their order. Median tie breaking,
quickselect's within-half order, objective stopping and convergence rules are
unchanged. Deadline-limited runs can naturally complete more work.

Cache allocation is optional and admitted from spare graph memory after the
existing degree-lane choice; it never drops additional dimensions or reduces
lanes. Admission charges cache payload, degree payload, lane descriptors,
entity/CSR scratch and the log table, using checked arithmetic. Tight budgets
retain direct computation. The chunked-text caller's `from_csr` path retains
direct computation too: its current budget omits caller-owned plans and
construction buffers, so spare memory cannot safely be established there.
The cache does not fix that older policy. Cache state is refreshed only for
active terms and reused across refinements and partitions; no allocation or
cache synchronization occurs per document/posting. Debug BP logs report cache
admission.

The initial prototype kept cached and direct scoring in one large callback.
M4 assembly showed a direct call and register spills per document even in the
cached branch. Dispatch now occurs once per refinement through a generic
`fill_gains` helper, separating the small cached callback from the logarithmic
fallback. The compiler still emits a direct call per document; its cached
callback uses a 16-byte frame versus the prototype's 128-byte frame. The
posting loop has scalar loads/additions and bounds checks, with no division,
logarithm or call. No `dyn Fn`, virtual scoring call, forced inline
attribute, unchecked gather, or architecture-specific default was introduced.

The cached posting reduction deliberately remains strict scalar arithmetic.
Reassociating a document's sum to obtain SIMD would change gain bits, ties,
and output order. A future SIMD experiment can instead batch independent
document accumulators while preserving each document's addition order; it
must account for ragged CSR rows, scattered term accesses and cross-platform
results. Generic closures alone are not evidence of virtual dispatch.

## Measurements and validation

The ignored `graph_bisection::gain_tests::bench_bp_gain_review` runs a fixed
seed, four-thread pool and default release flags. Each fixture has one warmup
and three timed full graph calls, including graph scratch allocation and
destruction. Input construction, permutation validation, byte capture and
destruction of returned orders are outside timing. The coarse fixture stops
at 524,288 entities to isolate large-partition work; the others descend to 32.
The sparse fixture intentionally has low posting reuse and an oversized
vocabulary as a control. No default is tuned from these synthetic fixtures.

Three paired runs, alternating before/after order, produced these ranges of
per-run medians. All compilers and this task's other CPU-heavy checks had
stopped; the shared Mac was not CPU-isolated. The earlier prototype run was
slower in both binaries, so the final paired comparison is used here.

| Fixture                    | Entities / postings    |         Before |          After | Paired speedup |
| -------------------------- | ---------------------- | -------------: | -------------: | -------------: |
| Mixed terms, full depth    | 100,003 / 4,339,473    | 164.8–172.0 ms | 109.2–111.5 ms |     1.48–1.57× |
| Sparse control, full depth | 100,003 / 394,309      |   45.7–47.2 ms |   41.3–42.5 ms |     1.11–1.12× |
| Coarse, popular terms      | 1,048,577 / 11,944,199 | 360.7–374.6 ms | 158.7–161.0 ms |     2.26–2.33× |

Complete permutation bytes match across revisions and every repetition.
Process peak RSS, including all three fixture constructions and validation,
was 144.4–153.4 MB before and 142.8–150.5 MB after. Those overlapping ranges
do not establish a memory reduction. The exact additional cache payload is
512 KiB, 4 MiB and 32 KiB respectively, plus 24 bytes per lane for its vector
descriptor on this host. It fits the admitted allowance and is reused; cache
admission never expands the configured budget.

Raw paired evidence: `.context/bp-{before,after}-{1,2,3}.log`, permutations in
`.context/bp-{before,after}-paired-output/`, environment in
`.context/bp-environment.json`, and assembly in `.context/bp-final-{gains,doc-0}.asm`.
The preserved original and final test binaries are `.context/bp-{before,after}`.
These are local, gitignored artifacts, not portable benchmark distributions.

The separate public-entry fixture
`index::tests::bmp::bench_reorder_review_bytes` indexes 32,771 sparse records,
reorders, reopens and compares full-hit document/score bits before and after.
Across original and cached binaries, its complete 4,108,761-byte `.sparse`
file and 275,160 bytes of captured query results also matched. These bytes
exclude random segment IDs in metadata. The small public-entry fixture is a
correctness check, not a repeated end-to-end throughput benchmark. Logs and
bytes are in `.context/bp-bytes-{before,final}*`.

Reproduction:

```sh
cargo test --locked -p summa-core --release --lib --no-run
# Run the emitted test executable directly, after compilation stops:
SUMMA_BP_EVIDENCE_DIR=.context/reorder-evidence /usr/bin/time -l \
  <test-executable> bench_bp_gain_review --ignored --nocapture --test-threads=1
SUMMA_BP_EVIDENCE_DIR=.context/reorder-evidence \
  <test-executable> bench_reorder_review_bytes --ignored --nocapture --test-threads=1
```

Use the same added fixtures against the review base; neither benchmark needs
a public core API solely for measurement.

### x86_64 Linux validation

The same source and fixtures were tested on an idle GCE `c4-highmem-48`
instance with an Intel Xeon Platinum 8581C, Debian Linux 6.1, and the same
Rust 1.98.1 / LLVM 22.1.8 compiler release. Both revisions were built with
default release flags and separately with
`RUSTFLAGS='-C target-cpu=x86-64-v3'` (AVX2). All four builds completed before
tests and timings began. Each process was pinned with `taskset -c 0,1,2,3`
to four distinct physical cores; BP retained its fixed four-thread pool.
This was a cloud machine, not a dedicated physical host.

As on M4, each process had one warmup and three timed calls per fixture;
three process pairs alternated before/after execution order. Ranges below
are the three per-process medians and paired speedups. These remain synthetic
whole-BP timings, not production reorder throughput or cold-storage results.

| Compiler target  | Fixture                    |         Before |          After | Paired speedup |
| ---------------- | -------------------------- | -------------: | -------------: | -------------: |
| Default x86_64   | Mixed terms, full depth    | 326.3–331.8 ms | 186.4–191.4 ms |     1.70–1.76× |
| Default x86_64   | Sparse control, full depth |   94.8–96.1 ms |   77.2–79.0 ms |     1.20–1.23× |
| Default x86_64   | Coarse, popular terms      | 506.7–516.1 ms | 175.1–180.7 ms |     2.81–2.93× |
| x86-64-v3 / AVX2 | Mixed terms, full depth    | 313.2–317.2 ms | 180.8–181.1 ms |     1.73–1.75× |
| x86-64-v3 / AVX2 | Sparse control, full depth |   92.4–95.9 ms |   74.5–78.3 ms |     1.22–1.24× |
| x86-64-v3 / AVX2 | Coarse, popular terms      | 495.6–506.9 ms | 173.1–175.3 ms |     2.85–2.89× |

GNU `time -v` measured process peak RSS across all fixture constructions,
timed calls and permutation checks. Default-target RSS was 95.39–95.44 MiB
before and 95.24–95.57 MiB after; AVX2 was 95.28–95.59 MiB before and
95.72–95.87 MiB after. This does not establish a memory reduction. The exact
cache payload remains 512 KiB, 4 MiB and 32 KiB for the respective fixtures,
plus descriptors, admitted within the existing graph budget.

Each optimized binary passed 1,299 core library tests with 13 ignored. The
ignored BP and public-entry fixtures were then run explicitly. Complete
permutation bytes matched across both revisions, both x86 compiler targets
and all three process repetitions. The 4,108,761-byte sparse file and
275,160-byte query-score capture also matched across both revisions and
targets, and matched the M4 captures. The mixed and sparse permutations
matched M4 too. The coarse permutation differed between Linux/x86 and
macOS/ARM in both the original and optimized implementation: this change
preserves output within each tested platform, without establishing a new
cross-platform permutation guarantee.

For Linux, replace the macOS timing wrapper above with:

```sh
SUMMA_BP_EVIDENCE_DIR=.context/reorder-evidence /usr/bin/time -v \
  taskset -c 0,1,2,3 <test-executable> \
  bench_bp_gain_review --ignored --nocapture --test-threads=1
```

Raw binaries, logs, outputs, CPU/compiler metadata and byte hashes are in
`.context/bp-x86/evidence/`; the source manifest, runner and calculated summary
are in `.context/bp-x86/`. Evidence was copied back and its archive SHA-256
verified before removing this task's remote scratch directory. These are
local, gitignored evidence files, as with the M4 captures.

### Validation status

The resumed audit rechecked all three saved permutation files, the sparse
blob and query-score capture byte for byte. It also reviewed cache refresh
after degree updates, lane reuse, checked allocation admission and the
native/sequential gain dispatch. The implementation remains the one measured
above; this continuation adds review and validation records.

| Check                                   | Result                                                                                                                                                                                                           |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python3 scripts/check_search.py check` | Passed: contracts, format, Clippy, core/server/broker/tool tests with metrics, and native-without-sync compilation. Evidence: `.context/search-harness/20260905T144827.006019Z-check/`.                          |
| Gain regressions                        | Three passed: exact gain bits including log-table overflow, half direction and empty rows; cache admission limits; complete permutations and reused cache storage. Evidence: `.context/bp-final-gain-tests.log`. |
| Portable Rust library                   | `cargo check --locked -p summa-core --no-default-features --lib` passed. Evidence: `.context/bp-resume-portable-check.log`.                                                                                      |
| WASM                                    | `bash build.sh`, `npm ci` and `npm test -- --run` passed; four tests across two files. Evidence: `.context/bp-resume-wasm-{build,npm,tests}.log`.                                                                |
| Documentation                           | `uv run scripts/check_docs.py` passed: 71 Markdown files, 247 local links and 20 benchmark targets. Evidence: `.context/bp-resume-docs-check.log`.                                                               |

The earlier attempt to run Rust library tests without the native feature
failed to compile because other test modules reference native-only APIs
(`Index`, `MmapDirectory`, Rayon and loader helpers). See
`.context/bp-portable-tests.log`. A portable library/WASM build is not evidence
that this Rust test suite passes. No lifecycle or RPC code changed, so the
additional `full` harness (API docs and real-server broker tests) was not run.
Production-corpus timings, Linux cold I/O and runs on a dedicated physical
host remain unmeasured.

## Remaining findings and experiments

1. **P1 — Chunked-text memory admission is incomplete.**
   `text_reorder::plan_field` budgets only 12 bytes per selected posting.
   Vocabulary arrays, chunk counts, fill cursors, retained plans and graph
   entity scratch are additional allocations. `ForwardIndex::from_csr`
   chooses degree lanes using CSR bytes alone. `reorder_term` also materializes
   a `Vec<u32>` per matching chunk, retains all positions for the term, sorts
   32-byte entries on this host, then copies IDs/frequencies into a
   `PostingList`. `rewrite_text_files` additionally retains new length arrays,
   chunk-map builders for every chunked field and ordinary-text norm columns;
   these costs include fields whose order did not change. High-df positioned
   terms can therefore exceed the advertised pass memory budget even after
   graph terms were filtered. Bring text under
   the owning component's complete budget model; stream postings to CSR in
   bounded passes and store positions in flat, budgeted runs. Test tiny budgets,
   multiple reordered fields, large positioned terms and peak heap. No bounded
   memory claim for the entire text path follows from this kernel improvement.

2. **P1 — Text truncation can disappear as successful convergence.**
   `plan_field` returns `None` for an identity permutation or zero retained
   terms, losing `converged = false` from zero-time/memory-limited BP.
   `reorder_segment` treats an empty text-plan list as converged. A depth cap
   above the text posting-block size is likewise not reflected in the plan's
   convergence flag. This can prevent optimizer deepening even though useful
   work remains. Text also starts the full BP time budget after graph building,
   unlike BMP's subtraction of elapsed build time. Return outcome/convergence
   independently of whether bytes require rewriting, and regress zero-time,
   no-retained-term and depth-capped passes at the index boundary. Planning
   currently ignores out-of-range posting IDs while rewrite indexes the inverse
   map directly; reject corrupt IDs explicitly in the same follow-up.

3. **P2 — Rank scratch still costs eight bytes per entity on 64-bit hosts.**
   The graph allocates `Vec<usize>` of length N before refinement, including
   coarse levels that use radix selection and never consult it. Quickselect
   only uses local indices below 1,048,576, so a `u32` rank representation could
   save `4N` bytes (about 221 MiB at 58M entities). Another option is bounded
   per-lane rank storage sized for the largest quickselect partition. Compare
   complete permutation bytes: quickselect's within-half permutation is part
   of BP's symmetry breaking. Do not substitute a different stable sort merely
   because median membership matches.

4. **P2 — Record transpose and grid sorting remain substantial work.**
   Each record window counts source postings, scans again to fill 12-byte
   routed tuples, comparison-sorts by output block/dimension/slot/impact, and
   serially encodes output blocks. Window allocations are bounded, but repeated
   source-block visits can amplify decode cost under scattering. Measure
   source-block scans, tuple bytes, window count and phase time before trying
   counted partitioning/radix sorting or parallel encoding. Preserve exact
   slot/impact order, budget the overlap of scratch/output buffers, and compare
   complete sparse bytes. Blockwise reorder already copies compatible payloads;
   avoid converting it into a record rebuild. Historical parallel-encoding
   wording in `budgeted-reorder.md` is not a description of today's routed
   serial output loop.

5. **P2 — Smaller serial/heap costs deserve isolated measurements.**
   Candidate discovery maintains a heap even when every eligible term fits.
   Radix selection scans gains four times and creates boxed histograms through
   Rayon fold/reduce; `ranked` initialization and final partition copying add
   memory traffic each refinement. The level scheduler replays O(depth) path
   bits per claimed partition and uses shared progress atomics per iteration.
   These are code-path findings, not demonstrated dominant costs. Benchmark
   them separately after the arithmetic saving, preserving deterministic
   candidate/term order and the existing memory/lifecycle protocol. Writer
   trait-object dispatch occurs at output boundaries; removing it without
   measuring call frequency is lower priority than reducing scans and sorts.

Production follow-up should use the same corpus before/after, track elapsed
phase times, peak heap/RSS and faults, encoded bytes, permutation/query
correctness, and pruning quality at a fixed work/deadline budget. Run both
warm/cold storage and concurrent search workloads on Linux. The x86 default
and AVX2 evidence above supports this arithmetic-preserving cache; future
architecture-sensitive defaults still need their own comparisons against
the portable scalar path.
