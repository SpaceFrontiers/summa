# Hermes

A high-performance, embeddable full-text search engine written in Rust.

## Features

- **Fast indexing** - Parallel segment building and compression
- **BM25 ranking** - Industry-standard relevance scoring
- **WAND optimization** - Efficient top-k query processing
- **Multiple platforms** - Native, WASM, and Python bindings
- **Flexible schema** - SDL-based schema definition language
- **Compression** - Zstd compression with configurable levels
- **Slice caching** - Optimized for cold-start latency

## Packages

| Package              | Description                                  | Registry                                            |
| -------------------- | -------------------------------------------- | --------------------------------------------------- |
| `hermes-core`        | Core search engine library                   | [crates.io](https://crates.io/crates/hermes-core)   |
| `hermes-tool`        | CLI for index management and data processing | [crates.io](https://crates.io/crates/hermes-tool)   |
| `hermes-server`      | gRPC server for remote search                | [crates.io](https://crates.io/crates/hermes-server) |
| `hermes-core-python` | Python bindings                              | [PyPI](https://pypi.org/project/hermes-core-python) |
| `hermes-wasm`        | WASM bindings for browsers                   | [npm](https://www.npmjs.com/package/hermes-wasm)    |

## Quick Start

### Installation

```bash
# Install CLI tool
cargo install hermes-tool

# Or use as a library
cargo add hermes-core
```

### Create an Index

```bash
# Create index from SDL schema
hermes-tool create -i ./my_index -s schema.sdl

# Index documents from JSONL
cat documents.jsonl | hermes-tool index -i ./my_index --stdin

# Or from compressed file
zstdcat dump.zst | hermes-tool index -i ./my_index --stdin -p 50000

# Merge segments
hermes-tool merge -i ./my_index

# Show index info
hermes-tool info -i ./my_index
```

### Schema Definition (SDL)

```sdl
index documents {
    field title: text [indexed, stored]
    field content: text [indexed]
    field author: text [indexed, stored]
    field published_at: u64 [indexed, stored]
}
```

### Data Processing Tools

```bash
# Calculate SimHash for near-duplicate detection
cat docs.jsonl | hermes-tool simhash -f title -o title_hash

# Sort documents by field
cat docs.jsonl | hermes-tool sort -f score -N -r

# Pipeline example
zstdcat dump.zst \
    | hermes-tool simhash -f title -o hash \
    | hermes-tool sort -f hash -N \
    | hermes-tool index -i ./my_index --stdin
```

### Using as a Library

```rust
use hermes_core::{Index, IndexConfig, FsDirectory};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = FsDirectory::new("./my_index");
    let config = IndexConfig::default();
    let index = Index::open(dir, config).await?;

    println!("Documents: {}", index.num_docs());

    Ok(())
}
```

### Python Usage

```python
import hermes_core

# Open index
index = hermes_core.Index("./my_index")

# Search
results = index.search("query text", limit=10)
for doc in results:
    print(doc.score, doc.fields)
```

### WASM Usage

```javascript
import init, { HermesIndex } from "hermes-wasm";

await init();
const index = await HermesIndex.open("/ipfs/Qm.../");
const results = await index.search("query text", 10);
```

## Development

### Prerequisites

- Rust 1.92+
- Python 3.12+ (for Python bindings)
- Node.js 20+ (for WASM)
- wasm-pack (for WASM builds)

### Building

```bash
# Build all packages
cargo build --release

# Build WASM
cd hermes-wasm && wasm-pack build --release --target web

# Build Python wheel
cd hermes-core-python && maturin build --release
```

### Testing

```bash
cargo test --all-features
```

### Linting

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run linters
cargo fmt --all
cargo clippy --all-targets --all-features
```

## License

MIT
