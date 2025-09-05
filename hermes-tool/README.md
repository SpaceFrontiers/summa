# Hermes Tool

Command-line utilities for building, inspecting, optimizing, and preprocessing
[Hermes](https://github.com/SpaceFrontiers/hermes) indexes.

## Build and help

```bash
cargo build --release -p hermes-tool
target/release/hermes-tool --help
target/release/hermes-tool <command> --help
```

Set `RUST_LOG=hermes_tool=debug` for additional diagnostics.

## Index lifecycle

Create an index from a schema file:

```bash
hermes-tool create --index ./my-index --schema ./schema.sdl
```

Or initialize one from inline SDL:

```bash
hermes-tool init \
  --index ./my-index \
  --sdl 'index docs { field title: text<simple> [indexed, stored] }'
```

Index JSON Lines from a file or standard input, then commit:

```bash
hermes-tool index \
  --index ./my-index \
  --documents ./documents.jsonl

zstdcat documents.jsonl.zst |
  hermes-tool index --index ./my-index --stdin

hermes-tool commit --index ./my-index
```

Use the index inspection and maintenance commands:

```bash
hermes-tool info --index ./my-index
hermes-tool search --index ./my-index --query 'title:hermes' --limit 10
hermes-tool merge --index ./my-index
hermes-tool reorder --index ./my-index
hermes-tool heatmap --index ./my-index --field sparse_embedding
hermes-tool warmup --index ./my-index --cache-size 67108864
```

`heatmap` requires a BMP sparse-vector field; the example name assumes your
schema defines `sparse_embedding`.

`index` accepts memory, indexing-thread, compression-thread, and optimization
controls. Run `hermes-tool index --help` before tuning them; the defaults are
chosen for general-purpose ingestion.

## JSONL preprocessing

`simhash`, `sort`, and `term-stats` read JSON objects from standard input and
write their primary output to standard output, so they can be composed:

```bash
zstdcat documents.jsonl.zst |
  hermes-tool simhash --field title --output title_simhash |
  hermes-tool sort --field title_simhash --numeric \
  > ordered.jsonl

zstdcat documents.jsonl.zst |
  hermes-tool term-stats --field title --field body \
  > term-stats.json
```

The external sorter writes bounded chunks to a temporary directory. Use
`--chunk-size` to control memory and `--temp-dir` to choose a volume with
sufficient free space.

## Vector utilities

Train IVF coarse centroids from a numeric array field in JSONL:

```bash
hermes-tool train-centroids \
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
cargo clippy -p hermes-tool --all-targets -- -D warnings
cargo test -p hermes-tool
```

Keep the clap help in `src/main.rs` and this command overview aligned whenever
commands or defaults change.

## License

MIT

### Delete, upsert, and compact rows

These commands require a primary-key field and commit their changes:

```bash
hermes-tool merge -i ./my_index --compact
hermes-tool delete -i ./my_index --key obsolete-id --key another-id
hermes-tool upsert -i ./my_index --document '{"id":"existing-id","title":"replacement"}'
hermes-tool compact -i ./my_index --memory-budget-mb 256
hermes-tool compact -i ./my_index --segment SEGMENT_HEX_ID --memory-budget-mb 256
```

`upsert` supplies a complete replacement document and inserts a missing key.
`compact` removes deleted rows from each dirty segment, including a singleton.
`merge` combines segments and retains deletion masks; `merge --compact`
physically removes deleted rows after merging. Indexed-only fields are
preserved. Index metadata format 7 requires rebuilding older indexes. See the
[row-deletion design](../docs/row-deletion.md) for snapshot, budget, and key semantics.
