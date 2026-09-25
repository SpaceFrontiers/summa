---
title: HTTP and IPFS browser readers
parent: Guides
---

# HTTP and IPFS browser readers

Use `RemoteIndex` for HTTP-hosted index files or `IpfsIndex` with application
fetch and size callbacks. Both readers are read-only. Create compatible index
files with Summa 2, then publish a complete, immutable snapshot.

The [browser WASM guide](../apis/js-api.md) includes current
imports, method names, callback types and caching methods. HTTP hosting must
support range requests and CORS when origins differ. The reader's optional
IndexedDB cache does not make all network traffic private.

IPFS transport is supplied by your application. Summa 2 does not bundle an IPFS
node or an HTTP gateway. The original guide's `aiosumma`, YAML configuration and
old web bundle commands have been replaced by the current WASM API.
