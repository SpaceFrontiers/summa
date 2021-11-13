---
title: Summa
---

Summa is a full-text WASM-compatible search server written in Rust.
Yes, you can launch it entirely inside your browser!

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](/summa-proto/python-api) and [CLI](/summa-proto/python-api)
- [WASM-bindings](https://github.com/izihawa/summa/tree/master/summa-wasm) to launch Summa in browsers
- Open remote indices through network. We have already implemented [IPFS](/summa-proto/ipfs-wasm-guide) support out of the box
- Kafka for indexing

## Online-documentation

- [Quick-start](/summa-proto/quick-start)

### Mandatory
- [Query DSL](/summa-proto/query-dsl)
- [Index Schema](/summa-proto/schema)
- [Collectors](/summa-proto/collectors)

### APIs
- [Python API](/summa-proto/python-api)
- [Kafka Consuming API](/summa-proto/kafka-consuming-api)
- [Metrics API](/summa-proto/metrics-api)

### Expert
- [IPFS Publish + WASM Browsing](/summa-proto/ipfs-wasm-guide)
- [Architecture](/summa-proto/architecture)

## Development

- [Development Guide](/summa-proto/development)