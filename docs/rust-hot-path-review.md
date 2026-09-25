# Rust hot-path review — 2026-09-05

Scope: core query execution, fast-field decoding, top-k collection, and the
server's entry into them. This extends the [performance review](search-performance-review.md).

## Invariants and experiment design

Before this pass, range bitset materialization constructed a boxed document
predicate and performed random-access field lookup for every document. The
implemented single-value path uses `FastFieldReader::scan_single_values`, which traverses
blocks once and dispatches the codec once per batch of at most 256 values.
Scratch remains bounded at 2 KiB plus the existing document bitset. Multi-value
range queries currently inspect the **first** value, not any value; retain that
behavior. Missing values never match; i64 requires zigzag decoding before signed
comparison; f64 keeps the existing sortable encoding semantics. No stored bytes,
query planning policy, public API, or scoring arithmetic changes.

`rust_hot_paths` measures the real `RangeQuery::as_doc_bitset` entry point against
the original predicate scan on identical 65,536-document RAM fixtures with one
or sixteen encoded blocks and two selectivities. Fixture building, merging,
opening and exact membership checks are outside timing. Both timed paths include
bitset allocation and destruction. The predicate scan is an algorithmic control;
it shares the compact compiled bound introduced in this pass.

A separate abstraction probe scans already decoded u64 values with a generic
closure or an opaque `dyn Fn`. Its purpose is to expose devirtualization and
vectorization in assembly, not to estimate a whole search request's speedup.
Input and the opaque callback are passed through `black_box`; the static closure
keeps its concrete type as a production generic callback would.

The second experiment targets `bitpacked_read_batch`: its original 16/32/64-bit
branches retained a bounds check inside every scalar iteration on this compiler,
despite the vectorization comment. The change selects the complete byte range
with checked length arithmetic once, then zips fixed-size byte chunks with the
output slice.
The invariant is equal input/output element counts; truncated input must still
fail, not be silently shortened by `zip`. Preserve little-endian decoding,
wrapping addition, unaligned starts and empty-output behavior. No unsafe code or
new decoder is needed. Measure all four byte-aligned widths, with 8-bit as a
control because its existing branch already vectorizes.

## Measured results and limitations

Apple M4, 10 logical CPUs, macOS 15.6.1, Rust 1.98.0 / LLVM 22.1.8;
Cargo's release benchmark profile, default `sync` features, no custom RUSTFLAGS.
Criterion uses 20 samples, 3-second warmup and about 5-second measurement per
case. No compilation or other heavy work from this review ran during sampling.
Other applications and unrelated builds shared the machine; control timings
show substantial noise. These are warm CPU microbenchmarks, not server latency,
RSS, cold mmap, or cross-architecture claims.

For **256 values per decoder call**, central timing estimates are:

| Encoded width | Before, ns | After, ns | Ratio of central estimates |
| ------------- | ---------: | --------: | -------------------------: |
| 8 bits        |      36.39 |     22.54 |                       1.6× |
| 16 bits       |     203.61 |     23.05 |                       8.8× |
| 32 bits       |     115.99 |     30.51 |                       3.8× |
| 64 bits       |     118.69 |     25.21 |                       4.7× |

For 16-bit decoding the timing intervals were 190.44–216.28 ns before and
22.98–23.16 ns after. The separate Criterion comparison estimator reported
88.82–89.63% less time. The 32-bit after interval was noisier (25.74–34.74 ns),
so its precise ratio is less stable. The 8-bit branch already vectorized before
this change; its improvement comes with the revised range proof/loop shape,
not newly eliminating a virtual call.

Assembly confirms NEON widening/add/store loops for 16/32 bits and vector
load/add/store loops for 64 bits. The decode callbacks disappear. This does
increase the complete `bitpacked_read_batch` symbol from **1,084 to 1,780 bytes**
on this build (+696 bytes, including its other codec-width branches). No new
heap allocation or architecture-specific intrinsics were introduced.

For **65,536-document range materialization**, the final binary includes both
the revised production path and the original predicate-scan algorithm. Same
fixture, same executable, with setup and membership assertions outside timing:

| Blocks | Approx. matches |  Predicate scan, µs | Batched production, µs |
| ------ | --------------: | ------------------: | ---------------------: |
| 1      |              1% |              489.02 |                  60.69 |
| 1      |             50% |              539.33 |                  83.50 |
| 16     |              1% | 2,762.70 (unstable) |                 156.85 |
| 16     |             50% |              713.24 |                 101.62 |

Three cases show roughly 6.5–8.1× lower time for the batch path. **Do not use the
16-block/1% ratio as a precise claim**: its reference scan interval was
2,059.61–3,391.26 µs, and the preceding reference run was about 1,076 µs. Even
unchanged controls moved, so small before/after effects are inconclusive. The
improvement is the combination of removing block lookups, per-value codec
selection and opaque callbacks; it cannot be attributed solely to virtual-call
latency. No query-planning selectivity threshold changed.

The isolated decoded-value probe measured 6.88 µs for the generic closure versus
83.13 µs for the opaque callback in the final run (65,536 values). Earlier runs
were 6.31/66.81 and 7.50/71.72 µs. Treat this as evidence of lost vectorization,
not a promised 10× benefit from changing any closure anywhere in Summa.

The range bitset retains 8,192 bytes for this fixture in both paths. The batch
path adds a bounded 2,048-byte stack decode buffer and removes the separately
allocated predicate capture. The random-access predicate remains boxed, with a
more compact shared bound. Decoder changes add no heap scratch. These are
layout/allocation bounds, not measured process RSS.

### Reproduction and retained evidence

```sh
python3 scripts/check_search.py bench --bench rust_hot_paths --save-baseline before
# Apply the matching implementation changes, retaining the same fixture.
python3 scripts/check_search.py bench --bench rust_hot_paths --baseline before
# Run just the decoder experiment:
python3 scripts/check_search.py bench --bench rust_hot_paths --filter bitpacked_batch --baseline before

cargo bench --locked -p summa-core --bench rust_hot_paths --no-run
# Use the executable path reported by Cargo, excluding its .d file:
llvm-nm --demangle -n <benchmark-executable>
llvm-objdump --disassemble --demangle --no-show-raw-insn <benchmark-executable>
```

On this Mac LLVM tools are in `/opt/homebrew/opt/llvm/bin`. Inspect named symbols
`count_static`, `count_dynamic`, `heap_order`, `bitpacked_read_batch`, and the
range instantiation of `scan_single_values`. Restrict objdump with the start/stop
addresses from nm to avoid dumping the entire benchmark/dependency binary.
A disassembly is evidence for its compiler/target/profile, not a portable test.

The initial pre-range-change run is
`.context/search-harness/20260905T035015.312002Z-bench/`. An external cleanup
removed `target/`, including those Criterion distributions, during the next
build. The attempted comparison failed explicitly in
`20260905T035633.309303Z-bench`; no comparison p-value is claimed against that
lost baseline. The initial timing log and `.context/rust-hot-paths-before.asm`
survived.

Fresh evidence uses `CRITERION_HOME=.context/search-harness/criterion`:

- `.context/rust-batched-baseline.log` and `rust-batched-environment.json`: the
  rebuilt executable, range batching present, decoder unchanged; saved baseline
  `rust-batched`. The executable was run directly after Cargo built it, with
  `--bench --save-baseline rust-batched` to avoid another build for test-only edits.
- `.context/search-harness/20260905T040649.053043Z-bench/`: final implementation
  against `rust-batched`; complete commands, source fingerprint and results.
- `.context/bitpacked-before.asm`, `bitpacked-after.asm`, and
  `range-codegen-*.asm`: focused native code-generation evidence.

The decoder table is a source before/after comparison. The range table is a
same-binary algorithm comparison, retaining the reference predicate scan in the
benchmark. It is not a fabricated reconstruction of the lost distributions.

## What the optimized code shows

A closure is an anonymous captured-value type; it does not inherently allocate.
The call boundary determines whether its type remains visible. See the Rust
Reference on [closure types](https://doc.rust-lang.org/reference/types/closure.html)
and [trait objects](https://doc.rust-lang.org/reference/types/trait-object.html).
These statements do not imply that every generic callback will inline or that
an optimizer can never devirtualize a trait object.

The release benchmark binary on Apple M4 / Rust 1.98.0 / LLVM 22.1.8 shows:

- `count_static`: the generic `iter().filter(...).count()` closure disappears.
  Its main loop uses NEON `ldp q`, `cmhs.2d` and vector accumulators. There are
  no callback calls in this function. Rewriting it as an index loop would not
  remove an abstraction cost.
- `count_dynamic`: a scalar loop executes `blr x19` once per value; the function
  pointer is hoisted out of the loop, but the callback cannot inline and the
  reduction does not vectorize. This is a deliberately opaque callback probe.
- `heap_order`: both `then_with` closures disappear. The remaining code includes
  sign transformations for `f32::total_cmp`, integer comparisons and conditional
  tie handling. It is not a single comparison instruction. Keep total ordering,
  including signed zero/NaN and doc/ordinal ties; don't substitute `partial_cmp`
  or an ordinary float comparison to shorten assembly.
- The original range bitset entry point called `as_doc_predicate`, allocated its
  48-byte capture, then called `DocBitset::from_predicate`. The latter retained
  an indirect call inside the document loop. The revised shared bound is a
  private tagged comparison rather than separately retained signed and unsigned
  bounds; scorer, probes and bitset scans use the same comparison semantics.

Assembly snippets (register allocation is compiler-specific):

```asm
; Generic decoded-value predicate: multiple values per iteration
ldp q5, q6, [x8, #-0x20]
cmhs.2d v5, v0, v5
sub.2d v1, v1, v5

; Opaque predicate: one callback per value
ldr x1, [x22], #0x8
mov x0, x20
blr x19
add x23, x23, w0, uxtw
```

`#[inline]` is a hint, including its `always` form. Large specializations can
increase instruction-cache pressure. Keep the existing out-of-line deferred TF
and BM25 score computation until whole-query evidence supports changing it.
See [code-generation attributes](https://doc.rust-lang.org/reference/attributes/codegen.html)
and [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html).
No LTO, target CPU, fast-math, unsafe-indexing, or inline-policy default changed.

## Call-site review and next experiments

| Priority    | Site                                                        | Actual cost / recommended experiment                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fixed       | `query/range.rs::as_doc_bitset`                             | Full scans paid predicate dispatch, merged-block lookup and codec decoding per document. Use the existing generic batch visitor; retain random probes and first-value multi access.                                                                                                                                                                                                                                                                     |
| P2          | `query/bmp.rs::score_superblock_blocks`, `query/planner.rs` | A concrete bitset predicate becomes `&dyn Fn` through several layers and is called after score/doc-map pruning. Specialize the bitset kernel or dispatch a small borrowed filter enum once at entry. Benchmark real BMP with 1%, 50%, 100% selectivity and matched candidate counts; the decoded-array SIMD ratio does not predict this scattered-access path. Limit the number of kernel instantiations and inspect text size.                         |
| P2          | `query/collector.rs::drive_scorer`                          | Generic collectors still drive `dyn Scorer::score` and `advance` per hit; positions add another optional call. The existing `precomputed_top_k` handoff already avoids this for supported ranked scorers. Investigate a batch handoff for cheap, dense filters only after a whole-query profile. Preserve collector ordering, count, positions, seek and native/async semantics; avoid a second scorer implementation.                                  |
| P2          | `query/scoring.rs::execute_windowed`                        | Four fixed-size allocations reserve 49,664 bytes per segment executor: scores 16,384, mask 512, candidate docs and scores 16,384 each, plus term-sized arrays and the heap. Borrow bounded scratch from the execution owner as BMP already does. Validate nested/reentrant queries, current-thread execution, cancellation, peak concurrent scratch and retained thread-local memory. A scratch pool is not automatically a win for every thread count. |
| P2          | `structures/fast_field/mod.rs::ColumnBlock`                 | Native `size_of` is 208 bytes per block; the binary search needs only its 4-byte `cumulative_docs`. A compact parallel doc-base directory or separated hot metadata could improve random probes at high merge fanout. Benchmark 1/16/256/4096 blocks, including extra retained bytes and lookup duplication. Batch traversal already avoids this search for full scans.                                                                                 |
| P2          | `codec.rs::blockwise_linear_read_batch`                     | Every batch restarts the variable-length header walk at offset 8. A full column scan in 256-value batches repeatedly visits prefixes of its 512-value subblocks: the header work is still quadratic in subblock count. A stateful sequential decoder should carry the byte offset across batches. Benchmark irregular piecewise-linear columns at 1K/64K/1M values; do not claim every codec now scans in linear time.                                  |
| P2          | `directories/directory.rs::OwnedBytes`                      | A slice shares bytes but clones an Arc owner; it is not zero-cost. Borrow byte slices inside an already-owned block instead of constructing per-value owned views. Empty views currently allocate Arc headers too; measure segment-open allocation counts before adding an empty backing variant. Preserve ownership across asynchronous reads and generation retirement.                                                                               |
| P2          | `observe.rs::Timer`, BMP doc-map phase                      | With native metrics enabled, per-block doc-map timing reads the clock at start and end even without an installed recorder. Metrics-off builds erase this timer. Benchmark server-equivalent features; consider bounded sampling with explicitly revised metric semantics, not silently disabling observability.                                                                                                                                         |
| Investigate | `query/boolean.rs::as_doc_bitset` / predicate composition   | Nested boxed predicates multiply calls; optional SHOULD bitsets are also constructed when MUST already determines filter membership. Quantify construction versus traversal costs before changing planning. Keep unsupported-query behavior and SHOULD scoring distinct from filter membership.                                                                                                                                                         |

The server remains an adapter into these core paths. Removing an RPC-level trait
call while retaining per-document codec dispatch would attack the wrong scale.
`ScorerFuture` and async directory callbacks box futures at construction/read
boundaries, not once per document in the synchronous inner scoring loop. Measure
allocations per request and per I/O, including short current-thread queries,
before replacing the object-safe API or creating divergent native/WASM versions.

## Layout, synchronization and memory access

Observed native layouts are measurements, not stable ABI promises:
`HeapEntry=12`, `DocBitset=24`, `SharedThreshold=40`, `ColumnBlock=208`,
`FastFieldReader=136` bytes. The 12-byte heap entry already keeps score/doc/ordinal
compact. Padding it to a cache line or boxing each entry would multiply memory
traffic. `TermCursor` deliberately uses an inline enum with text metadata and
Vec headers; the decode payloads are heap arrays, not giant inline arrays. The
large-enum lint alone is insufficient evidence to box its text variant.

`SharedThreshold` already uses relaxed monotonic hints, and window deadlines are
checked once per 64 windows. Preserve their pruning-depth and truncation protocol.
The slice cache already reserves stamps in thread-local blocks of 64 and shards
hit counters; it does **not** increment one global stamp on every hit. Its per-slice
stamp and Arc refcounts can still contend on a popular slice. Measure concurrent
same-slice versus disjoint-slice access before changing accounting or padding.
The cache's explicit 64-byte alignment is not proof against false sharing on every
processor; validate the target's cache-line behavior before changing that default.

Software prefetch is a processor hint, not a promise of free cached accesses.
The misleading zero-cost comment was corrected without changing prefetch policy.
Arm defines the effect of [PRFM as implementation-defined](https://support.arm.com/documentation/ddi0602/2024-09/Base-Instructions/PRFM--register---Prefetch-memory--register--?lang=en).
Use hardware counters on the intended deployment host to assess useful prefetches,
cache/TLB misses, bandwidth and instruction pressure. This Mac pass does not
supply Linux perf counters, cold mmap residency, x86 AVX2 or production tail latency.

Existing unchecked cursor access in `query/scoring.rs` depends on decoded-array
lengths, `pos`, and deferred-load state remaining consistent. A debug assertion
is not release validation of a file. Future changes there should establish
slice lengths at the decode boundary, compare against safe scalar results, and
inspect the emitted loop before adding more unchecked indexing. The byte-aligned
decoder change demonstrates that bounds-proof placement can recover SIMD while
keeping safe indexing. This review does not certify every existing unsafe block.

## Validation

`python3 scripts/check_search.py check` passed all four stages; evidence is in
`.context/search-harness/20260905T041018.863837Z-check/`. This includes strict
Clippy on all core/server/broker/tool targets, 1,410 passing Rust tests with
metrics (18 intentionally ignored), formatting, and the native-without-sync
compile boundary. The range integration test also passed with
`--no-default-features --features native`, executing the async configuration
(`.context/range-bitset-native-async.log`). The WASM release build and optimizer
passed via `cd summa-wasm && bash build.sh`; after `npm ci`, `npm test -- --run`
passed all four tests in two files. Build/install/test logs are
`.context/rust-review-wasm-{build,npm,test}.log`. The five Python harness self-tests, Ruff, contract/link checks
and `git diff --check` also passed.

The new integration regression passed on the original range implementation and
on the rewrite. It compares independently expected membership and exact 1.0
score bits against bitset, async scorer and sync scorer after merging uneven
blocks, including an absent-value block. Cases include missing fields, first
multi-values, signed extremes, f64 signed zero/infinities/NaN, inverted/open
bounds, and batch tails. Decoder tests exercise all byte-aligned widths with
unaligned starts, empty output, wrapping values and truncated payloads.

No format writer changed in this continuation: decoder inputs remain borrowed
immutable bytes. Lifecycle/RPC `full`, Linux residency/perf counters and x86
performance measurements were not rerun for these portable query/decoder edits.
