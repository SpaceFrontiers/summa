# Summa Tool

Command-line utilities for building, inspecting, optimizing, and preprocessing
[Summa](https://github.com/SpaceFrontiers/summa) indexes.

## Build and help

```bash
cargo build --release -p summa-tool
target/release/summa-tool --help
target/release/summa-tool <command> --help
```

Set `RUST_LOG=summa_tool=debug` for additional diagnostics.

## Index lifecycle

Create an index from a schema file:

```bash
summa-tool create --index ./my-index --schema ./schema.sdl
```

Or initialize one from inline SDL:

```bash
summa-tool init \
  --index ./my-index \
  --sdl 'index docs { field title: text<simple> [indexed, stored] }'
```

Index JSON Lines from a file or standard input:

```bash
summa-tool index \
  --index ./my-index \
  --documents ./documents.jsonl

zstdcat documents.jsonl.zst |
  summa-tool index --index ./my-index --stdin
```

`index` commits and waits for background merges before returning.
Inspect or maintain the index:

```bash
summa-tool info --index ./my-index
summa-tool diagnose --index ./my-index
summa-tool search --index ./my-index --query 'title:summa' --limit 10
summa-tool merge --index ./my-index
summa-tool reorder --index ./my-index
summa-tool heatmap --index ./my-index --field sparse_embedding
summa-tool warmup --index ./my-index --cache-size 67108864
```

`heatmap` requires a BMP sparse-vector field; the example name assumes your
schema defines `sparse_embedding`.

`index` accepts memory, indexing-thread, compression-thread, and optimization
controls. Run `summa-tool index --help` before tuning them; the defaults are
chosen for general-purpose ingestion.

## JSONL preprocessing

`simhash`, `sort`, and `term-stats` read JSON objects from standard input and
write their primary output to standard output, so they can be composed:

```bash
zstdcat documents.jsonl.zst |
  summa-tool simhash --field title --output title_simhash |
  summa-tool sort --field title_simhash --numeric \
  > ordered.jsonl

zstdcat documents.jsonl.zst |
  summa-tool term-stats --field title --field body \
  > term-stats.json
```

The external sorter writes bounded chunks to a temporary directory. Use
`--chunk-size` to control memory and `--temp-dir` to choose a volume with
sufficient free space.

## Vector utilities

Train IVF coarse centroids from a numeric array field in JSONL:

```bash
summa-tool train-centroids \
  --input ./vectors.jsonl \
  --field embedding \
  --output ./coarse-centroids.bin \
  --clusters 4096 \
  --max-iters 20 \
  --seed 42
```

All accepted vectors should have the same dimension. `--sample-size` limits
the number read.

`retrain-centroids` is currently a diagnostic placeholder: it opens the index
and prints the manual JSONL workflow, but does not extract vectors or rebuild
the index. Use `train-centroids` until the end-to-end operation is implemented.

## Development

From the repository root:

```bash
cargo fmt --all -- --check
cargo clippy -p summa-tool --all-targets -- -D warnings
cargo test -p summa-tool
```

Keep the clap help in `src/main.rs` and this command overview aligned whenever
commands or defaults change.

## Delete, upsert, and compact rows

`delete` and `upsert` require a primary-key field and commit their changes:

```bash
summa-tool merge -i ./my_index --compact
summa-tool delete -i ./my_index --key obsolete-id --key another-id
summa-tool upsert -i ./my_index --document '{"id":"existing-id","title":"replacement"}'
summa-tool compact -i ./my_index --memory-budget-mb 256
summa-tool compact -i ./my_index --segment SEGMENT_HEX_ID --memory-budget-mb 256
```

`upsert` supplies a complete replacement document and inserts a missing key.
`compact` removes deleted rows from each dirty segment, including a singleton.
`merge` combines segments and retains deletion masks; `merge --compact`
physically removes deleted rows after merging. Indexed-only fields are
preserved. Metadata formats 6–8 upgrade to 9 on open; older segment formats may
require rebuilding. See [row deletion](../docs/row-deletion.md) for compatibility
and [diagnostics](../docs/diagnostics.md) for health checks.
