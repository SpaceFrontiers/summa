# summa-core

`summa-core` is the storage, indexing, and query engine behind Summa. It is
an embeddable Rust library with native and WebAssembly build profiles. For
SDL syntax, see the [schema guide](../docs/schema.md).

## Quick start

With `summa-core`, `tokio` (`macros`, `rt-multi-thread`), and `serde_json` dependencies:

```rust,no_run
use summa_core::{Index, IndexConfig, RamDirectory, index_json_document, parse_single_index};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = parse_single_index(r#"
        index articles { field title: text<en_stem> [indexed, stored] }
    "#)?.to_schema();
    let index = Index::create(RamDirectory::new(), schema, IndexConfig::default()).await?;
    let mut writer = index.writer();
    index_json_document(&writer, &serde_json::json!({"title": "Hybrid search"})).await?;
    writer.commit().await?;
    for hit in index.query("hybrid", 10).await?.hits {
        println!("{} {:?}", hit.score, index.get_document(&hit.address).await?);
    }
    Ok(())
}
```

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
[segment lifecycle and recovery](../docs/segment-lifecycle.md).

## Features

| Feature             | Purpose                                                                              |
| ------------------- | ------------------------------------------------------------------------------------ |
| `sync` (default)    | Synchronous search paths; implies `native`                                           |
| `native`            | Filesystem/mmap directories, native writer, parallel builders, and native tokenizers |
| `wasm`              | Browser-compatible writer/reader components and tokenizer backend                    |
| `http`              | HTTP-backed directory access                                                         |
| `cjk-dict`          | Embedded Japanese/Korean morphology dictionaries for the `morph` tokenizer option    |
| `metrics`           | Runtime metrics emission                                                             |
| `diagnostics`       | Additional build diagnostics                                                         |
| `query-diagnostics` | Opt-in per-query work accounting; implies `native`                                   |
| `fst-index`         | FST-backed SSTable block indexes; enabled by `native` and `wasm`                     |

Common validation profiles are:

```bash
# Default native + synchronous API
cargo test -p summa-core

# Native async API without synchronous query paths
cargo check -p summa-core --no-default-features --features native

# Feature set consumed by summa-wasm
cargo check -p summa-core --no-default-features --features wasm,http \
  --target wasm32-unknown-unknown
```

## Row deletion and upserts

The native writer supports `delete_primary_key`, `upsert_document`, `compact`,
and `compact_segment`. Initialize primary-key deduplication, stage mutations,
then commit; reload readers to observe the new generation. Compaction preserves
indexed-only fields and trained ANN codes. Old searchers retain their snapshots.
`force_merge()` retains tombstones; `force_merge_with_compaction(true)` additionally
compacts the final output, including a single-segment index.
See the [deletion design](../docs/row-deletion.md) for budgets and compatibility. Multiple staged replacements of one key are
supported; commit publishes the latest accepted version. Optional
[content hashes](../docs/content-deduplication.md) skip unchanged upserts.

## Benchmarks

See the [benchmark guide](../docs/benchmarks.md) for the complete target list,
filtered Criterion runs, and retrieval-quality datasets.

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

Run `cargo fmt --package summa-core -- --check`, strict rustdoc, and the
narrowest relevant test target before the full crate suite:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc -p summa-core --no-deps
```

Format and serialization changes should also run their module tests and any
regression test under `summa-core/tests`.
