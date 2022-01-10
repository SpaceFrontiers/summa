---
title: Summa
nav_exclude: true
search_exclude: true
---

Summa is a full-text, IPFS-friendly and WASM-compatible search server written in Rust

- Yes, your data may be replicated and published through IPFS!
- Yes, you may launch Summa entirely inside your browser and then search in IPFS published index!

Start with our [Quick Start guide](/summa/guides/quick-start) or [Architecture](/summa/core/architecture) description.

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](/summa/apis/python-api) and [CLI](/summa/apis/python-api)
- [Embedded IPFS implementation](https://github.com/n0-computer/iroh) allowing to seed and replicate index through IPFS network
- [WASM-bindings](/summa/apis/wasm-api) to launch Summa in browsers
- Also, you may use Kafka for indexing