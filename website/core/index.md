---
title: Core
nav_order: 3
has_children: true
---

# Core concepts

`summa-core` owns storage, indexing, scoring and immutable segment publication.
The server, broker, clients and WASM bindings expose that shared engine.
Summa 2 uses its own Rust search engine; the original Tantivy-based API does
not describe its schema or index format.

An index has a [schema](schema.md), a set of immutable segments and a writer.
Indexing stages documents. A commit publishes a new searchable generation;
merges combine compatible segments while preserving their stored semantics.
Document addresses contain a segment ID and a segment-local document ID.

- [Schema Definition Language](schema.md): field types, tokenizers, multi-value semantics and storage.
- [Query language](query-dsl.md): Boolean clauses, phrases, fields, filters and vectors.
- [Ranking and result collection](collectors.md): candidate generation, formulas, fusion and limits.
- [Chunked text](chunked-text.md): passage indexing and document identity.
- [Dense vectors](dense-vectors.md) and [sparse vectors](sparse-vectors.md).
- [Rust API](../apis/rust-api.md): RAM, native filesystem and portable profiles.

Use the [server](../guides/server.md) for persistent indexes accessed over gRPC,
or the [browser API](../apis/js-api.md) for writable local indexes and read-only
HTTP/IPFS readers. Storage capabilities depend on the entry point; an old
`IndexEngine` configuration cannot be passed to the new server.
