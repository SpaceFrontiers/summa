# Hermes

Embeddable Rust search engine: BM25 and phrase search, sparse vectors (BMP,
MaxScore, Seismic), dense search (flat, TQ, IVF-TQ, binary IVF, ScaNN), and hybrid ranking.
Run locally, behind gRPC, or in a browser over HTTP/IPFS with WASM.

- Chunked and multi-value fields, filters, fusion, formula ranking, and reranking.
- Primary-key deletion, full-document upserts, content-hash deduplication, and compaction.
- Sharded search through a broker with shared BM25 statistics.
- A separate MAL-defined Transformer/Mamba stack for training, inference, and inspection.

[Documentation](docs/README.md) · [Schema](docs/schema.md) ·
[Query syntax](docs/query-language.md) · [Contributing](CONTRIBUTING.md)

## Quick start

Run in a new working directory:

```bash
cargo install hermes-tool
hermes-tool init -i ./articles --sdl 'index articles {
    field id: text<raw> [primary, stored]
    field title: text<en_stem> [indexed, stored]
}'
printf '%s\n' '{"id":"1","title":"Hybrid search with Hermes"}' |
  hermes-tool index -i ./articles --stdin
hermes-tool search -i ./articles --query 'title:hybrid' --limit 10
```

`index` commits before returning. For remote access:

```bash
cargo install hermes-server
hermes-server --data-dir . --addr 127.0.0.1:50051
```

Connect with the [Python](hermes-client-python/README.md) or
[TypeScript](hermes-client-typescript/README.md) client. See
[server operations](hermes-server/README.md) for resource limits and maintenance.

## Packages

| Package                                                 | Purpose                                   |
| ------------------------------------------------------- | ----------------------------------------- |
| [hermes-core](hermes-core/README.md)                    | Rust storage, indexing, and query library |
| [hermes-tool](hermes-tool/README.md)                    | Index CLI and JSONL processing            |
| [hermes-server](hermes-server/README.md)                | gRPC search and indexing                  |
| [hermes-broker](hermes-broker/README.md)                | Shard routing and distributed search      |
| [hermes-proto](hermes-proto/README.md)                  | Shared gRPC contract                      |
| [Python client](hermes-client-python/README.md)         | Async gRPC client                         |
| [TypeScript client](hermes-client-typescript/README.md) | Node.js gRPC client                       |
| [hermes-wasm](hermes-wasm/README.md)                    | Browser search and local indexing         |
| [hermes-web](hermes-web/README.md)                      | Vue/WASM search UI                        |
| [hermes-llm](hermes-llm/README.md)                      | Inference, generation, and model traces   |
| [hermes-train](hermes-train/README.md)                  | Training and evaluation workflows         |
| [hermes-model-lab](hermes-model-lab/README.md)          | Local model observability UI              |
| [hermes-mal](hermes-mal/README.md)                      | Model Architecture Language parser        |
| [hermes-mal-python](hermes-mal-python/README.md)        | Python MAL bindings                       |
| [hermes-tokenizer](hermes-tokenizer/README.md)          | Byte-level BPE tokenizer                  |

## Development

Use the pinned [Rust toolchain](rust-toolchain.toml) and `protoc`:

```bash
cargo build --release
python3 scripts/check_search.py check
uv run scripts/check_docs.py
```

Read [AGENTS.md](AGENTS.md) and the [search contract](docs/search-system-contract.md)
before search changes. See [Contributing](CONTRIBUTING.md) for client, WASM, and

## License

MIT
