---
title: APIs
nav_order: 4
has_children: true
---

# Summa APIs

The server and broker expose the same `summa.SearchService` and
`summa.IndexService` gRPC services on port **50051** by default.
Use matching 2.x clients and server versions. The broker also exposes its own
`summa.broker.BrokerService` control API.

| Interface                       | Package or contract           | Use                                     |
| ------------------------------- | ----------------------------- | --------------------------------------- |
| [Python](python-api.md)         | `summa-client-python`         | Async server and broker client          |
| [TypeScript](typescript-api.md) | `summa-client-typescript`     | Node.js server and broker client        |
| [gRPC](grpc-api.md)             | `summa-proto/summa.proto`     | Other languages and generated clients   |
| [Rust](rust-api.md)             | `summa-core`                  | Embedded indexing and search            |
| [Browser WASM](js-api.md)       | `summa-wasm`                  | Local indexing and remote index readers |
| [Metrics](metrics-api.md)       | `summa_*` Prometheus families | Server and broker monitoring            |

The browser package accesses index files; it is not the Node.js gRPC client.
Summa 2 has no bundled Kafka consumer API or REST/HTTP gateway. Applications
can ingest data through the maintained clients or gRPC ingestion methods.

Read the complete [search and indexing schema](protocol.md) or
[broker control schema](broker-protocol.md) for exact RPC and message fields.
