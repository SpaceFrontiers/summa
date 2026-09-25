# Summa engineering harness

Read [the search system contract](docs/search-system-contract.md) before changing
core, server, broker, protocol, tool, or WASM search code. It is the shared source
of architectural rules; `CLAUDE.md` adds repository workflow and ML guidance.
Existing design documents describe implemented constraints, not optional advice.

## Work loop

1. Trace the public entry point, data owner, and caller before editing. Identify
   native/sync/WASM variants and the persisted or wire formats involved.
2. State the invariant and cost model. For substantial changes, update the
   relevant design document first. Distinguish today's behavior from proposals.
3. Reproduce a bug with a behavior-named regression test. For format-preserving
   rewrites, compare bytes as well as query results. Test failure and cancellation
   at lifecycle boundaries, not only successful output.
4. Keep implementations in their owning component; adapters validate and
   translate. Extract by responsibility with the narrowest visibility. Do not
   introduce a second writer, scorer, schema parser, or lifecycle protocol.
5. Run `python3 scripts/check_search.py check`. Use `full` for lifecycle/RPC
   changes and the WASM build for portable code. See the contract's test matrix.
   Report unrun checks and environmental failures explicitly.
6. Measure performance changes with the same fixture, compiler, machine, and
   flags before/after. Include memory and correctness, not only elapsed time.
   Do not change defaults from one synthetic benchmark or one architecture.

## Non-negotiable constraints

- Merge compatible encoded blocks/runs by copying and remapping small metadata.
  Rebuild only where format semantics require it or an explicit bounded reorder
  policy requests it. Never retrain index-global ANN artifacts inside a merge.
- Keep frequently read metadata compact and budgeted for residency. Keep
  corpus-sized payloads evictable; use cold writers for bulk lifecycle output.
  Heap residency, mmap residency, and Rust `Pin` are different concepts.
- Bound scratch, cache growth, concurrency, retries, candidate expansion, and
  response hydration. Check request limits before conversion, I/O, or admission.
- Publish immutable generations atomically. Claim outputs before writing;
  preserve ownership through cancellation/panic; drain blocking work on deletion.
- Reject corrupt/incompatible data. Never repair live metadata silently or turn
  an error into empty results. Degraded modes and budget truncation are observable.
- Preserve single/multi-value, missing-value, ordinal, and cross-segment scoring
  semantics. Keep native and async execution equivalent unless documented.

Use [the code map and validation matrix](docs/search-system-contract.md) to
maintain these rules with the implementation. Record remaining review findings
and measured evidence in [the performance review](docs/search-performance-review.md).
