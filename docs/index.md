---
title: Summa
---

Summa is a full-text WASM-compatible search server written in Rust.
Yes, you can launch it entirely inside your browser!

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](/summa/python-api) and [CLI](/summa/python-api)
- [WASM-bindings](/summa/ipfs-wasm-guide) to launch Summa in browsers
- Open remote indices through network. We have already implemented [IPFS](/summa/ipfs-wasm-guide) support out of the box
- Kafka for indexing

## Online-documentation

- [Quick-start](/summa/quick-start)

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

## Development

- [Development Guide](/summa/development)