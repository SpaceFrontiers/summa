![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)

[![PyPI Version](https://img.shields.io/pypi/v/aiosumma.svg?label=aiosumma%20(Python))](https://pypi.python.org/pypi/aiosumma)
[![Crates.io](https://img.shields.io/crates/v/summa-proto.svg?label=summa-proto%20(Rust))](https://crates.io/crates/summa-proto)
[![NPM](https://img.shields.io/npm/v/summa-wasm.svg?label=summa-wasm%20(JS))](https://www.npmjs.com/package/summa-wasm)

# Summa

Summa is a full-text, IPFS-friendly, and WASM-compatible search engine.

With Summa, your data can be replicated and published through IPFS, allowing for a truly distributed and uncensorable search experience. And, thanks to its compatibility with WASM, Summa can be launched entirely inside your browser, enabling you to search through IPFS-published indexes without ever having to send queries to centralized servers.

If you're ready to experience the power of Summa, be sure to check out our [Quick Start guide](https://izihawa.github.io/summa/guides/quick-start) or our detailed [Core documentation](https://izihawa.github.io/summa/core). 

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](https://izihawa.github.io/summa/apis/python-api) and [CLI](https://izihawa.github.io/summa/apis/python-api)
- [Embedded IPFS implementation](https://github.com/n0-computer/iroh) allowing to seed and replicate index through IPFS network
- [JS-bindings](https://izihawa.github.io/summa/apis/js-api) to launch Summa in browsers
- Also, you may use Kafka for indexing

## Online-documentation

- [Quick Start guide](https://izihawa.github.io/summa/guides/quick-start)
- [Core documentation](https://izihawa.github.io/summa/core)
- [github.io](https://izihawa.github.io/summa)

## Distribution

### Server

⚠️ *The project is under active development, we do not publish `latest` images yet. The best option now
is `testing`*

- [Docker (testing)](https://hub.docker.com/r/izihawa/summa-server/testing)
- [Docker (v0.11.0)](https://hub.docker.com/r/izihawa/summa-server/0.11.0)

### Clients

- [Python](https://pypi.org/project/aiosumma/)
- [Rust (proto)](https://lib.rs/crates/summa-proto)

## Donate

You may support us at [OpenCollective](https://opencollective.com/izihawa) or by cryptos:
- `monero: 464Wws65yssHdqGKGkFsHmbqNhBJ7zoPrbPTGAJma4VmTngtrJmQEaG9i739CUJJak3esALHpbWGXdVwMghzpFToLD6Q7Ne`
- `btc: 3HooXUqJnZ4Ad8AGeqfSZ5QZQE72ZaZgY6`
- `eth: 0x009AeabF4aeDe417d324077E7858956e6d0962D6`
