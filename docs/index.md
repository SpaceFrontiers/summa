---
title: Welcome
---

Full-text search server written in Rust with Python and GRPC API

## Key Features

- Full-text index
- Store and retrieve documents with textual and numeric fields
- Wide range of supported queries. Retrieve documents, do faceted search and collect statistics
- Rank documents with BM25, custom and/or user-defined scoring functions
- Simple GRPC API for managing multiple indices and for search
- Index documents directly through GRPC API or Kafka
- Fine CLI and asynchronous client library [aiosumma](aiosumma/README.md) written in Python
- Extendable query parsing. Create your own rich query parsed, add synonyms and buid query context with [aiosumma](aiosumma/README.md) library
- Tracing with [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-rust) and exposing metrics in Prometheus format
- `Reflection API` for deep insights about text data

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
- [Reflection API](/summa/metrics-api)

### Expert
- [IPFS Replication](/summa/ipfs-replication)
- [Architecture](/summa/architecture)
- [Development Guide](/summa/development)
