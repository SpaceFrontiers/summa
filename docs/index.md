---
title: Summa
nav_exclude: true
search_exclude: true
---

Summa is a full-text WASM-compatible search server written in Rust.
Yes, you can launch it entirely inside your browser!

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](/summa/apis/python-api) and [CLI](/summa/apis/python-api)
- [WASM-bindings](/summa/guides/ipfs-wasm-guide) to launch Summa in browsers
- Open remote indices through network. We have already implemented [IPFS](/summa/guides/ipfs-wasm-guide) support out of the box
- Kafka for indexing