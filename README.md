![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)
[![Crates.io](https://img.shields.io/crates/v/summa-core.svg?label=summa-core)](https://crates.io/crates/summa-core)
[![Crates.io](https://img.shields.io/crates/v/summa-server.svg?label=summa-server)](https://crates.io/crates/summa-server)

# Summa

Summa is a full-text, IPFS-friendly and WASM-compatible blazing fast search engine written in Rust.

- Yes, your data may be replicated and published through IPFS!
- Yes, you may launch Summa entirely inside your browser and then search in IPFS published index withoud sending queries to centralized servers!

These both properties allow you to create distributed (hence uncensorable) and privacy-first search systems.

Start with our [Quick Start guide](https://izihawa.github.io/summa/guides/quick-start) or [Architecture description](https://izihawa.github.io/summa/core/architecture)

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](https://izihawa.github.io/summa/apis/python-api) and [CLI](https://izihawa.github.io/summa/apis/python-api)
- [Embedded IPFS implementation](https://github.com/n0-computer/iroh) allowing to seed and replicate index through IPFS network
- [WASM-bindings](https://izihawa.github.io/summa/apis/wasm-api) to launch Summa in browsers
- Also, you may use Kafka for indexing

## Online-documentation

- [github.io](https://izihawa.github.io/summa)
- [docs.rs](https://docs.rs/summa-core)

