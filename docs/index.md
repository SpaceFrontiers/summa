---
title: Welcome
---

Full-text WASM-compatible search server written in Rust.
Yes, you can launch it entirely inside your browser!

## Key Features

- Full-text index with wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](aiosumma/README.md) and [CLI](aiosumma/README.md)
- [WASM-bindings](summa-wasm) to launch Summa in browsers (yes, entirely in browsers)
- Open remote indices through network. We have already implemented [IPFS](/summa/ipfs-wasm-guide) support out of the box
- Kafka for indexing

## Quick Start

{% include quick-start-snippet.md %}

## Further Reading

### Mandatory
- [Query DSL](/summa/query-dsl)
- [Index Schema](/summa/schema)
- [Collectors](/summa/collectors)

### APIs
- [Python API](/summa/python-api)
- [Kafka Consuming API](/summa/kafka-consuming-api)
- [Metrics API](/summa/metrics-api)

### Expert
- [IPFS Publish + WASM Browsing](/summa/ipfs-wasm-guide)
- [Architecture](/summa/architecture)
- [Development Guide](/summa/development)
