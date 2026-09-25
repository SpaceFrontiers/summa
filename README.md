# Summa

[Documentation and blog](https://spacefrontiers.github.io/summa/) · [Summa 2 migration](docs/summa-2-migration.md)

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
cargo install summa-tool
summa-tool init -i ./articles --sdl 'index articles {
    field id: text<raw> [primary, stored]
    field title: text<en_stem> [indexed, stored]
}'
printf '%s\n' '{"id":"1","title":"Hybrid search with Summa"}' |
  summa-tool index -i ./articles --stdin
summa-tool search -i ./articles --query 'title:hybrid' --limit 10
```

`index` commits before returning. For remote access:

```bash
cargo install summa-server
summa-server --data-dir . --addr 127.0.0.1:50051
```

Connect with the [Python](summa-client-python/README.md) or
[TypeScript](summa-client-typescript/README.md) client. See
[server operations](summa-server/README.md) for resource limits and maintenance.

## Packages

| Package                                                | Purpose                                   |
| ------------------------------------------------------ | ----------------------------------------- |
| [summa-core](summa-core/README.md)                     | Rust storage, indexing, and query library |
| [summa-tool](summa-tool/README.md)                     | Index CLI and JSONL processing            |
| [summa-server](summa-server/README.md)                 | gRPC search and indexing                  |
| [summa-broker](summa-broker/README.md)                 | Shard routing and distributed search      |
| [summa-proto](summa-proto/README.md)                   | Shared gRPC contract                      |
| [Python client](summa-client-python/README.md)         | Async gRPC client                         |
| [TypeScript client](summa-client-typescript/README.md) | Node.js gRPC client                       |
| [summa-wasm](summa-wasm/README.md)                     | Browser search and local indexing         |
| [summa-web](summa-web/README.md)                       | Vue/WASM search UI                        |
| [summa-llm](summa-llm/README.md)                       | Inference, generation, and model traces   |
| [summa-train](summa-train/README.md)                   | Training and evaluation workflows         |
| [summa-model-lab](summa-model-lab/README.md)           | Local model observability UI              |
| [summa-mal](summa-mal/README.md)                       | Model Architecture Language parser        |
| [summa-mal-python](summa-mal-python/README.md)         | Python MAL bindings                       |
| [summa-tokenizer](summa-tokenizer/README.md)           | Byte-level BPE tokenizer                  |

## Development

Use the pinned [Rust toolchain](rust-toolchain.toml) and `protoc`:

```bash
cargo build --release
python3 scripts/check_search.py check
uv run scripts/check_docs.py
```

Read [AGENTS.md](AGENTS.md) and the [search contract](docs/search-system-contract.md)
before search changes. See [Contributing](CONTRIBUTING.md) for client, WASM, and
GPU checks; [benchmarks](docs/benchmarks.md) for workloads and dated results.

## License

MIT
