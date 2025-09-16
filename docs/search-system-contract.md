# Search system engineering contract

Scope: `hermes-core`, `hermes-server`, and their broker/tool/protocol/WASM
adapters. This consolidates existing contracts and defines how to maintain them.
It is not a new storage format or permission to change public behavior.

## Ownership and component boundaries

| Owner                                           | Responsibility                                                                             | Keep out                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| `core/dsl`, `tokenizer`                         | Schema/query language, field semantics, tokenization                                       | RPC and process policy                                    |
| `core/structures`                               | Encodings, validated byte views, scalar/SIMD primitives                                    | Schema dispatch, index publication, server settings       |
| `core/directories`                              | Byte ownership, reads, persistence, cold I/O and cache policy                              | Ranking and document semantics                            |
| `core/segment/builder`                          | Encode newly ingested documents using trained artifacts                                    | Publication and global model retraining                   |
| `core/segment/reader`                           | Parse format envelopes; expose immutable segment views                                     | Per-request rebuilding of metadata                        |
| `core/segment/merger`, `reorder`                | Write replacement segments; copy unchanged representations                                 | Metadata commit, lifecycle ownership, implicit retraining |
| `core/merge/segment_manager`, `segment/tracker` | Claims, publication, scheduling, retirement, cleanup                                       | RPC concerns and scoring kernels                          |
| `core/index`                                    | Writer generations, reader snapshots, search orchestration, shared resources               | Protocol types and duplicate storage implementations      |
| `core/query`                                    | Planning, scoring, top-k, fusion, reranking                                                | Filesystem lifecycle and transport admission              |
| `server/search_service`                         | RPC orchestration; `validation` owns request budgets; `response` owns hydration accounting | Encodings, alternate query executors                      |
| `server/registry`, `index_service`, `optimizer` | Per-index leases, RPC mutations, bounded maintenance scheduling                            | A second commit/cleanup protocol                          |
| `server/converters`                             | Wire/core conversion and semantic validation                                               | Storage and process scheduling                            |
| `broker`                                        | Topology, shard routing, distributed statistics and result combination                     | Reimplementation of segment search                        |
| `tool`, `wasm`, clients, `proto`                | Frontends and transport contracts                                                          | Forks of core algorithms                                  |

Paths above are relative to `hermes-<crate>/src`. This is a responsibility map,
not a strict acyclic module graph: existing sibling integrations are allowed.
The harness checks crate dependency boundaries. Storage must not gain an RPC,
training, or UI dependency to implement an adapter feature.

### Splitting and naming

- Name modules for a domain responsibility (`postings`, `validation`, `response`,
  `chunk_maps`), and functions for the operation and its semantics. Avoid new
  `utils`, `common`, `misc`, `v2`, or parallel “fast” implementations with unclear
  ownership. Version names belong to actual serialized formats or kernels.
- Split when a file owns distinct policies or has independent reasons to
  change, not at an arbitrary line count. Keep orchestration in `mod.rs` (or the
  existing parent `.rs`); keep helpers and tests with the policy they exercise.
- Default to private or `pub(super)`. Preserve existing public re-exports during
- Put reusable test fixtures in the narrowest shared test module. Name tests for
  observable behavior; keep cross-layer regressions at the index/RPC boundary.
- A change to a format, setting, or capability must update its docs, defaults,
  validation, diagnostics, and affected adapters together. New serde fields need
  explicit defaults; incompatible data needs an explicit version rejection.

## Merge is a representation-preserving operation

Default merge cost should be sequential bytes copied plus compact metadata
remapping, with bounded scratch. It must not scale heap use with document or
posting count merely to reconstruct existing values.

| Representation         | Normal merge                                                                     | Necessary exceptions                                                                                                |
| ---------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| BM25 external postings | Stream encoded blocks; patch doc bases and skip metadata                         | Inline/mixed terms need bounded decode/re-encode; positions and chunk maps need offset remapping                    |
| Fast fields            | Stack encoded blocks and local dictionaries                                      | Absent source column needs a compact missing-value block, not a document-sized array                                |
| Stored fields          | Copy compressed blocks                                                           | Segment-specific zstd dictionaries currently require streaming recompression                                        |
| BMP sparse             | Copy compatible encoded blocks and forward values; remap block/document metadata | Different weight scales or vocabulary layouts require explicit format-aware conversion; bounded reorder is separate |
| MaxScore sparse        | Merge through the existing sparse posting codec                                  | Posting list ordering and block boundaries require bounded decode/re-encode                                         |
| Seismic sparse         | Copy encoded runs and remap logical row/document directories                     | Explicit bounded maintenance rebuilds selected nomination terms; exact forward values remain single-copy            |
| Trained ANN            | Copy compatible encoded runs with doc/ordinal remapping                          | Reject incompatible global generations/codebooks; training belongs outside merge                                    |

Use `streaming_writer_cold` for bulk merge/reorder output. Prefer bounded
kernel-assisted ranges where available; handle short writes, interruption, and
unsupported filesystems without repeating partially emitted output. Bound source
prefetch and release one-shot source ranges. Cold I/O reduces cache pollution;
it does not replace durability or guarantee a particular kernel's cache state.

References: [posting codecs](posting-codecs.md), [cold I/O](cold-io.md),
[merge-time reorder](merge-time-reorder.md), [store v3](document-store-v3.md),
[row deletion and compaction](row-deletion.md).

## Memory and execution

- Separate per-query-mandatory metadata from candidate payloads. Favor contiguous
  packed arrays, validated `OwnedBytes` views, compact directories and shared
  immutable generations over per-entry heap objects or repeated decoding.
- Pin eligible metadata in priority order under the configured budget. Budget
  is **per segment**, plus each index-global ANN generation, not process-wide.
  Old reader generations can overlap during reload: include them in sizing.
- `mlock` needs OS headroom and can fail; copy mode keeps bytes off the page
  cache but can still swap on a swap-enabled host. Zero budget disables pinning.
  Report intended, pinned, failed/skipped, heap-copy and file-backed bytes.
  Rust `Pin<T>` does not make pages resident.
  The [Linux mlock contract](https://man7.org/linux/man-pages/man2/mlock.2.html)
  also specifies page rounding and non-stacking locks; account for these when
  changing pin ownership or interpreting byte budgets.
- Seismic summary payloads are trusted immutable writer output. Normal opening
  does not scan them, and summary views do not revalidate their format, lengths
  or decoded IDs. Writer/codec tests own those guarantees. Index-envelope
  compatibility and directory ownership remain separate from payload decoding.
- Keep Seismic nomination payloads, compressed ANN runs, and exact vectors evictable.
  Prefetch only selected bounded ranges; a deliberate flat scan is an explicit
  query plan, not a reason to make every payload resident.
- Reuse bounded query scratch and immutable reader state. Avoid per-hit hash
  maps, clones, and full sorts when dense IDs, borrowed slices, or bounded top-k
  suffice. Normal text queries trust writer-produced posting and position contents. Parse
  format envelopes and establish slice extents, without scanning directories or
  checking decoded document order. Explicit deserialization and merge admission
  retain integrity checks; ordinary search is not a corruption audit. Unsafe
  kernels still require proven input/output extents and supported CPU features.
- Use shared bounded search/background pools and I/O gates. Never create a pool
  per request/index reload or run unbounded CPU work on Tokio workers. Preserve
  the current-thread and WASM paths; `block_in_place` requires a multithread runtime.
- Bound decoded request shape before recursive conversion, index open, and
  admission; apply the policy to statistics RPCs too. Share capacity for work
  using the same resources and release permits on every terminal path.
  Statistics extraction allows the broker's flattened Boolean container under
  the aggregate node/clause limits; per-Boolean scoring fanout applies to search.
- Candidate, offset, fusion, text expansion, and hydration limits are separate
  budgets. Charge retained output before cloning, and encoded bytes before
  returning. Request labels must not create unbounded metric cardinality.

References: [metadata pinning](hot-metadata-pinning.md),
[streaming ANN](scann-streaming-index.md), [budgeted BP](budgeted-reorder.md).

### Rust hot loops

- Distinguish a closure's capture from its calling convention. Generic `impl Fn`
  callbacks can inline; `dyn Fn`, `dyn Scorer`, and opaque function pointers may
  require indirect calls. Keep dynamic composition at query/segment boundaries
  where practical, and amortize unavoidable dispatch over blocks or batches.
- Inspect optimized code for the actual call site before replacing iterators,
  closures, enums, or trait objects. Do not blanket-add `inline(always)`, box large
  enum variants, or introduce unchecked indexing. Weigh code size, pointer chasing,
  allocation, bounds checks, and vectorization together.
- Hoist invariant field/codec decisions out of document loops. Use the owning
  reader's batch decoder for full scans; preserve sparse probing for selective
  candidate checks. Preserve missing, first-value, ordinal, and numeric semantics.
- Keep synchronization at the frequency required by the protocol. A relaxed
  atomic is still shared memory traffic; stale pruning hints and lifecycle
  publication have different correctness requirements. Never weaken ordering

reproduction commands, and the limits of the measurements.

## Durability and failure semantics

Every segment has a metadata owner, active operation, or reader/deletion tracker.
Install the next owner before releasing the previous one. Claim outputs before
their first write and source segments before rewriting. Metadata rename is the
commit point; post-rename directory-fsync failure is degraded durability, not
an abort that permits deleting the published output.

Prepared indexing generations publish all-or-nothing. An owned finalizer must
survive requester cancellation. Drain started blocking work and pending deletes
before removing an index. Keep old readers valid until they release their
snapshot. Orphan cleanup must prove all owners absent; it must never “repair” a
missing metadata-live source. Quarantine deterministic corruption and back off
transient failures with finite retry scheduling. Destructive doctor recovery is
an explicit operator action. See [segment lifecycle](segment-lifecycle.md).
Started blocking tasks cannot be aborted by dropping their async wrapper; this
is also specified in [Tokio's spawn_blocking documentation](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html).

## Repeatable validation

From the repository root (Python 3.11+, pinned Rust toolchain, `protoc`):

```sh
python3 scripts/check_search.py contracts
python3 scripts/check_search.py check
python3 scripts/check_search.py full
python3 scripts/check_search.py check --plan
```

`contracts` checks dependency ownership and documentation links without building.
`check` also runs formatting, focused Clippy, core/server/broker/tool tests with
metrics, and the native-without-sync compile boundary. `full` adds API docs,
portable core compilation, and broker end-to-end tests against a real server.
CI retains the wider workspace, GPU, client, and WASM checks; the focused harness
does not replace those. Every run saves commands, status, logs and environment
metadata under `.context/search-harness/`. Failures stop the run with a nonzero
exit code. `--plan` lists commands without running checks.

| Change              | Additional evidence                                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Merge/encoding      | Byte identity for unchanged payloads, reopen and query equivalence, missing/multi-value/tail-block/overflow cases |
| Publication/cleanup | Failure, cancellation, panic, readers held across replacement, shutdown; real-server broker tests                 |
| Metadata residency  | Mmap fixture, copy-mode search equivalence and budget accounting; Linux mlock exercise on a suitable host         |
| Search/planning     | Exact top-k oracle or known expected hits/scores, ordinal/position semantics, native and async paths              |
| Protocol/conversion | Real RPC behavior, overload/error codes, bounded expansion, generated clients when proto changes                  |
| Portable core       | Native without sync; `cd hermes-wasm && bash build.sh && npm ci && npm test`                                      |

### Performance protocol