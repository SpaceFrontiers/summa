---
title: Ingestion from Kafka
parent: APIs
---

# Ingestion from Kafka

Summa 2 does not ship the original Kafka consumer API. Run the consumer in your
application and use [Python](python-api.md), [TypeScript](typescript-api.md), or
[gRPC](grpc-api.md) document ingestion.

Inspect per-document errors and commit accepted writes before advancing the
corresponding Kafka offsets. Handle retries using primary keys and the
[upsert semantics](../guides/mutations.md). A Summa commit and a Kafka offset
commit are separate operations; this integration does not provide a transaction
across both systems. Bound batch sizes and retry concurrency.
