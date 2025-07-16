# Hermes

A hybrid search engine combining BM25 text search, sparse vectors (SPLADE), and dense vectors (RaBitQ/ScaNN) in a single embeddable Rust library. Runs natively, over gRPC, in browsers via WASM, and over IPFS.

## Why Hermes?

| Feature                 | Hermes                 | Tantivy | Qdrant  | Elasticsearch |
| ----------------------- | ---------------------- | ------- | ------- | ------------- |
| BM25 Full-text search   | Yes                    | Yes     | No      | Yes           |
| Dense vectors (ANN)     | Yes (RaBitQ, ScaNN/PQ) | No      | Yes     | Plugin        |
| Sparse vectors (SPLADE) | Yes (native)           | No      | Partial | No            |
| WASM / Browser          | Yes                    | No      | No      | No            |
| IPFS storage            | Yes                    | No      | No      | No            |
| Embeddable library      | Yes                    | Yes     | No      | No            |

## Packages

| Package                | Description                                  | Registry                                              |
| ---------------------- | -------------------------------------------- | ----------------------------------------------------- |
| `hermes-core`          | Core search engine library                   | [crates.io](https://crates.io/crates/hermes-core)     |
| `hermes-tool`          | CLI for index management and data processing | [crates.io](https://crates.io/crates/hermes-tool)     |
| `hermes-server`        | gRPC server for remote search                | [crates.io](https://crates.io/crates/hermes-server)   |
| `hermes-client-python` | Python gRPC client                           | [PyPI](https://pypi.org/project/hermes-client-python) |
| `hermes-wasm`          | WASM bindings for browsers                   | [npm](https://www.npmjs.com/package/hermes-wasm)      |

## Quick Start

### CLI

```bash
cargo install hermes-tool

# Create an index from an SDL schema
hermes-tool create -i ./my_index -s schema.sdl

# Index documents from JSONL (with progress logging every 50k docs)
cat documents.jsonl | hermes-tool index -i ./my_index --stdin -p 50000

# Or from compressed files with optimization mode
zstdcat dump.zst | hermes-tool index -i ./my_index --stdin -O performance

# Commit, merge, and inspect
hermes-tool commit -i ./my_index
hermes-tool merge -i ./my_index
hermes-tool info -i ./my_index
```

### Rust Library

```rust
use hermes_core::{
    Index, IndexConfig, MmapDirectory, Document,
    index_json_document, parse_single_index,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create index from SDL
    let dir = MmapDirectory::new("./my_index");
    let schema = parse_single_index(r#"
        index articles {
            field title: text<en_stem> [indexed, stored]
            field body: text [indexed]
            field views: u64 [indexed, stored]
        }
    "#)?;
    let index = Index::create(dir, schema, IndexConfig::default()).await?;

    // Add documents
    let mut writer = index.writer();
    for json in [
        serde_json::json!({"title": "Hybrid Search", "body": "BM25 meets vectors", "views": 42}),
        serde_json::json!({"title": "WASM Search", "body": "Search in the browser", "views": 100}),
    ] {
        index_json_document(&writer, &json).await?;
    }
    writer.commit().await?;

    // Search
    let results = index.query("hybrid search", 10).await?;
    for hit in &results.hits {
        let doc = index.get_document(&hit.address).await?;
        println!("{:.4} {:?}", hit.score, doc);
    }

    Ok(())
}
```

### gRPC Server

```bash
# Run with Docker
docker build -t hermes-server -f hermes-server/Dockerfile .
docker run -p 50051:50051 -v ./data:/data hermes-server --data-dir /data

# Or install directly
cargo install hermes-server
hermes-server --addr 0.0.0.0:50051 --data-dir ./data
```

For production indexes, BP reorder resources are controlled independently:

```bash
hermes-server --data-dir /data \
  --search-threads 12 \
  --max-concurrent-searches 6 \
  --optimizer-threads 16 \
  --optimizer-concurrent-passes 1 \
  --optimizer-max-unconverged-passes 3 \
  --bp-memory-budget-mb 24576
```

- `--search-threads` bounds the process-wide CPU pool shared by nested search
  work across every index; it defaults to one quarter of detected CPUs.
- `--max-concurrent-searches` bounds simultaneous search pipelines and rejects
  overload promptly instead of queueing an unbounded number of decoded RPCs.
  Result windows, fusion/reranker work, query-tree expansion, vector payloads,
  stored-field hydration, and response bytes also have explicit server-side
  budgets; see [Search resource controls](hermes-server/README.md#search-resource-controls).
- `--optimizer-threads` is the width of one process-wide Rayon pool shared by
  every index and BP path. An active pass intentionally keeps that pool busy;
  lower this value when search or indexing needs more CPU. `0` disables the
  periodic optimizer, while merge-time and manual BP use the bounded fallback
  pool.
- `--optimizer-concurrent-passes` limits whole-segment passes across the
  optimizer, merge-time reorder, and manual reorder. It is not a thread count;
  keep it small because each pass can use the shared pool and its own memory
  budget.
- `--optimizer-max-unconverged-passes` is a hard eligibility bound for
  optimizer follow-up on one budget-truncated replacement lineage (the default
  `3` includes the initial partial pass). This prevents a segment that cannot
  converge within its budget from keeping the optimizer BP pool busy forever.
- `--bp-memory-budget-mb` bounds the main per-pass algorithmic working set:
  document maps, the BP forward graph and degree arrays, and record-rewrite
  grid/encode windows. Over-budget record passes fall back to block order and
  graph dimensions are trimmed. It is not a total-process RSS cap: readers,
  mmap/page-cache residency, output buffering, merge state, and indexing are
  additional.

Segment publication, replacement, reader retirement, orphan cleanup, failure
backoff, and index deletion follow one ownership protocol. Missing files that
are still referenced by metadata are quarantined instead of being retried in a
tight merge loop; start once with `--doctor` only when you intentionally want
to remove those corrupt metadata entries. See
[Segment lifecycle and recovery](docs/segment-lifecycle.md) and the full
[server options](hermes-server/README.md#background-merge-and-reorder).

Commit publication is cancellation-safe: after workers flush, an owned
finalizer carries metadata publication, primary-key refresh, and worker resume
to completion even if the client disconnects. A pre-publication storage error
keeps that generation paused and retryable instead of mixing it with new input.

Python client:

```python
from hermes_client_python import HermesClient

async with HermesClient("localhost:50051") as client:
    await client.create_index("articles", '''
        index articles {
            field title: text<en_stem> [indexed, stored]
            field body: text [indexed, stored]
        }
    ''')

    await client.index_documents("articles", [
        {"title": "Hybrid Search", "body": "Combining BM25 with vectors"},
    ])
    await client.commit("articles")

    results = await client.search("articles",
        query={"match": {"field": "title", "text": "hybrid"}})
    for hit in results.hits:
        print(hit.score, hit.address)
```

### WASM (Browser)

Hermes compiles to WebAssembly and can search indexes hosted over HTTP or IPFS directly in the browser, with IndexedDB-backed slice caching for near-zero cold-start latency on repeat visits.

```javascript
import init, { RemoteIndex, IpfsIndex } from "hermes-wasm";

await init();

// HTTP: load from any static file server
const index = new RemoteIndex("https://example.com/my_index");
await index.load();

// IPFS: load from content-addressed storage via verified-fetch
const ipfsIndex = new IpfsIndex("/ipfs/QmYourCID");
await ipfsIndex.load(fetchFn, sizeFn);

// Search (same API for both)
const results = await index.search("hybrid search", 10);
console.log(results);

// Persist cache to IndexedDB for instant reload
await index.save_cache_to_idb();
```

## Key Features

**Unified hybrid search** -- BM25 text ranking, SPLADE sparse vectors, and IVF-RaBitQ/IVF-PQ dense vectors share the same index, the same segments, and the same query pipeline. No sidecar services required.

**6 posting list formats** -- Adaptive format selection per list: HorizontalBP128, VerticalBP128, Elias-Fano, Partitioned Elias-Fano, Roaring bitmaps, and OptP4D. The engine picks the best format based on list density and length.

**Block-Max MaxScore** -- Top-k retrieval uses MaxScore partitioning (Turtle & Flood 1995) combined with block-max pruning (Ding & Suel 2011) and conjunction optimization. A single unified `MaxScoreExecutor` handles both BM25 text and sparse vector queries.

**Multi-value combiners** -- Documents with multiple vectors per field (e.g., chunked passages) are scored with configurable strategies: Sum, Max, Avg, LogSumExp (smooth approximation), or WeightedTopK with exponential decay.

**Matryoshka reranking** -- L2 reranker supports Matryoshka dimensionality reduction: scores candidates on leading dimensions first, then full-dimension exact scoring on survivors only. Skips 50-70% of cosine computations.

**SOAR multi-probe** -- IVF indexes use Google's SOAR (Spilling with Orthogonality-Amplified Residuals) for 5-15% recall improvement by assigning vectors to multiple clusters with orthogonal residuals.

**SimHash dedup pipeline** -- Stream-oriented CLI tools for near-duplicate detection: pipe through `simhash`, `sort`, then `index` to deduplicate million-document corpora before indexing.

**18 language stemmers** -- Snowball stemmers for Arabic, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, and Turkish. Plus HuggingFace tokenizer integration.

**Storage abstraction** -- Filesystem (mmap), HTTP (range requests), RAM, IPFS (JS fetch callbacks), and slice-caching directories. The same index binary works across all backends.

## Schema

Hermes uses a Schema Definition Language (SDL) to define index structure:

```sdl
index articles {
    field url: text [indexed, stored, primary]
    field title: text<en_stem> [indexed, stored]
    field body: text [indexed]
    field author: text<raw_ci> [indexed, stored]
    field published_at: u64 [indexed, stored]
    field embedding: dense_vector<768> [stored]
    field sparse_embedding: sparse_vector [indexed]
}
```

Field types: `text`, `u64`, `i64`, `f64`, `bytes`, `json`, `dense_vector<dim>`, `sparse_vector`
Attributes: `indexed`, `stored`, `primary`, `fast`
Tokenizers: `default`, `simple`, `raw`, `raw_ci`, `en_stem`, `de_stem`, `fr_stem`, `es_stem`, `it_stem`, `pt_stem`, `ru_stem`, `ar_stem`, and [more](docs/schema.md).

Full SDL reference: [docs/schema.md](docs/schema.md)

## Development

### Prerequisites

- Rust 1.97+ (see `rust-toolchain.toml`)
- Python 3.12+ (for Python client and bindings)
- Node.js 20+ (for WASM and web UI)
- wasm-pack (for WASM builds)
- protoc (for gRPC)

### Building

```bash
# Build all Rust packages
cargo build --release

# Build WASM (requires Homebrew LLVM on macOS for zstd cross-compilation)
cd hermes-wasm && bash build.sh

# Build the Python gRPC client
cd hermes-client-python && uv build

# Build the MAL Python binding
cd hermes-mal-python && maturin build --release
```

Alternatively you may build everything in docker via `docker compose`.

Examples:

- `docker compose run --rm cargo-build`
- `docker compose run --rm build-hermes-wasm`

### Testing

```bash
cargo test --all-features
```

LLM contributors should start with the [inference and training code map](docs/llm-code-map.md).
Temporary GPU dependency forks and their upstream removal criteria are listed
in [the fork register](docs/forked-dependencies.md).

### Linting

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Or run all pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## License

MIT
