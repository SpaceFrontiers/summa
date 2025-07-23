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
