# hermes-core

`hermes-core` is the storage, indexing, and query engine behind Hermes. It is
an embeddable Rust library with native and WebAssembly build profiles. For an
end-to-end example and the SDL syntax, see the
[repository README](https://github.com/SpaceFrontiers/hermes#readme) and
[schema guide](https://github.com/SpaceFrontiers/hermes/blob/main/docs/schema.md).

## Module map

| Module        | Responsibility                                                                                |
| ------------- | --------------------------------------------------------------------------------------------- |
| `directories` | Async storage abstraction plus RAM, mmap, filesystem, HTTP, and slice-caching implementations |
| `dsl`         | Schema and query-language parsing                                                             |
| `index`       | Public index, reader, writer, and searcher orchestration                                      |
| `merge`       | Segment lifecycle, metadata publication, merge policy, and cleanup                            |
| `query`       | Query planning, scoring, collection, fusion, and reranking                                    |
| `segment`     | Immutable segment construction, loading, vector data, and format code                         |
| `structures`  | SSTables, posting-list codecs, fast fields, SIMD kernels, and vector indexes                  |
| `tokenizer`   | Built-in tokenizers and optional Hugging Face tokenizer support                               |

The important ownership boundary is that `SegmentManager` is the sole writer
of index metadata. Indexing, merging, reordering, publication, and cleanup
must go through its lifecycle protocol; see
[segment lifecycle and recovery](https://github.com/SpaceFrontiers/hermes/blob/main/docs/segment-lifecycle.md).

## Features

| Feature          | Purpose                                                                              |
| ---------------- | ------------------------------------------------------------------------------------ |
| `sync` (default) | Synchronous search paths; implies `native`                                           |
| `native`         | Filesystem/mmap directories, native writer, parallel builders, and native tokenizers |
| `wasm`           | Browser-compatible writer/reader components and tokenizer backend                    |
| `http`           | HTTP-backed directory access                                                         |
| `cjk-dict`       | Embedded Japanese/Korean morphology dictionaries for the `morph` tokenizer option    |
| `metrics`        | Runtime metrics emission                                                             |
| `diagnostics`    | Additional build diagnostics                                                         |
| `fst-index`      | FST-backed SSTable block indexes; enabled by `native` and `wasm`                     |

Common validation profiles are:

```bash
# Default native + synchronous API
cargo test -p hermes-core

# Native async API without synchronous query paths
cargo check -p hermes-core --no-default-features --features native

# Feature set consumed by hermes-wasm
cargo check -p hermes-core --no-default-features --features wasm,http \
  --target wasm32-unknown-unknown
```

## Row deletion and upserts

The native writer supports `delete_primary_key`, `upsert_document`, `compact`,
and `compact_segment`. Initialize primary-key deduplication, stage mutations,
then commit; reload readers to observe the new generation. Compaction preserves
indexed-only fields and trained ANN codes. Old searchers retain their snapshots.
`force_merge()` retains tombstones; `force_merge_with_compaction(true)` additionally
compacts the final output, including a single-segment index.
See the [deletion design](../docs/row-deletion.md) for budgets, the format-7 rebuild
boundary, and the one-pending-upsert-per-key rule.

## Maintenance rules

- Keep storage access behind `Directory`/`DirectoryWriter`; query and segment
  code must not assume local files.
- Keep on-disk decoding shared between point lookup, scans, and iteration.
  Every decoder change needs malformed-input and round-trip coverage.
- Treat synchronous and asynchronous search implementations as one behavior
  contract. Add parity coverage when changing either path.
- Route all `SegmentManager` construction through the index-level config
  adapter so every create/open path receives the same resource policy.
- Preserve public re-exports in `lib.rs` when moving implementation code
  between modules.

Run `cargo fmt --package hermes-core -- --check`, strict rustdoc, and the
narrowest relevant test target before the full crate suite:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc -p hermes-core --no-deps
```

Format and serialization changes should also run their module tests and any
regression test under `hermes-core/tests`.
