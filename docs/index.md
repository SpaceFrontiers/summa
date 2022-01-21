---
title: Summa
nav_exclude: true
search_exclude: true
---

Summa is a full-text, IPFS-friendly, and WASM-compatible search engine that is designed to be lightning fast.

With Summa, your data can be replicated and published through IPFS, allowing for a truly distributed and uncensorable search experience. And, thanks to its compatibility with WASM, Summa can be launched entirely inside your browser, enabling you to search through IPFS-published indexes without ever having to send queries to centralized servers.

If you're ready to experience the power of Summa, be sure to check out our [Quick Start guide](https://izihawa.github.io/summa/guides/quick-start) or our detailed [Architecture description](https://izihawa.github.io/summa/core/architecture). 

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](/summa/apis/python-api) and [CLI](/summa/apis/python-api)
- [Embedded IPFS implementation](https://github.com/n0-computer/iroh) allowing to seed and replicate index through IPFS network
- [WASM-bindings](/summa/apis/wasm-api) to launch Summa in browsers
- Also, you may use Kafka for indexing