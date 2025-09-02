# Hermes engineering harness

Read [the search system contract](docs/search-system-contract.md) before changing
core, server, broker, protocol, tool, or WASM search code. It is the shared source
of architectural rules; `CLAUDE.md` adds repository workflow and ML guidance.
Existing design documents describe implemented constraints, not optional advice.

## Work loop

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
