![Maintenance](https://img.shields.io/badge/maintenance-activly--developing-brightgreen.svg)

[![PyPI Version](https://img.shields.io/pypi/v/aiosumma.svg?label=aiosumma%20(Python))](https://pypi.python.org/pypi/aiosumma)
[![Crates.io](https://img.shields.io/crates/v/summa-proto.svg?label=summa-proto%20(Rust))](https://crates.io/crates/summa-proto)
[![NPM](https://img.shields.io/npm/v/summa-wasm.svg?label=summa-wasm%20(JS))](https://www.npmjs.com/package/summa-wasm)

# Summa

Summa is a full-text IPFS-friendly search engine that may be launched on both large servers and inside your browser.

<img src="docs/assets/gear-logo-removebg.png" width=256 height=256>

Thanks to embedded IPFS daemon, your data can be replicated and published through P2P, allowing for a truly distributed and
uncensorable search experience. And, thanks to compatibility with WASM, Summa can be launched entirely
inside your browser, enabling you to search in network published indices without ever having to execute search queries
on remote servers.

If you're ready to start, be sure to check out our docs:
- [Quick Start guide](https://izihawa.github.io/summa/quick-start)
- Detailed [Core documentation](https://izihawa.github.io/summa/core)

## Key Features

- Full-text search engine written in Rust with a wide range of supported queries and ranking functions
- Server with GRPC API for using the search engine 
- Python asynchronous client [library and CLI](https://izihawa.github.io/summa/apis/python-api) for the API
- [JS-bindings](https://izihawa.github.io/summa/apis/js-api) to launch subset of Summa in browsers
- Also, you may use Kafka for indexing

## Online-documentation

- [Quick Start guide](https://izihawa.github.io/summa/quick-start)
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
