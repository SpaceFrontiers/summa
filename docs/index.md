---
title: Summa
nav_exclude: true
search_exclude: true
---

# Summa, The Reliable Searcher

[Summa](https://github.com/izihawa/summa) is a full-text IPFS-friendly search engine that may be launched on both large servers and inside your browser.

<img src="/summa/assets/gear-logo-removebg.png">

Thanks to embedded IPFS daemon, your data can be replicated and published through P2P, allowing for a truly distributed and
uncensorable search experience. And, thanks to compatibility with WASM, Summa can be launched entirely
inside your browser, enabling you to search in network published indices without ever having to execute search queries
on remote servers.

If you're ready to start, be sure to check out our docs:
- [Quick Start guide](https://izihawa.github.io/summa/quick-start)
- Detailed [Core documentation](https://izihawa.github.io/summa/core). 

## Key Features

- Full-text search engine written in Rust with a wide range of supported queries and ranking functions
- Server with GRPC API for using the search engine 
- Python asynchronous client [library and CLI](/summa/apis/python-api) for the API
- [JS-bindings](/summa/apis/js-api) to launch subset of Summa in browsers
- Has ready [embedded IPFS implementation](https://github.com/n0-computer/iroh) allowing to seed and replicate index through IPFS network
- Also, you may use Kafka for indexing

## Distribution

### Sources

- [GitHub](https://github.com/izihawa/summa)

### Server

⚠️ *The project is under active development, we do not publish `latest` images yet. The best option now
is `testing`*

- [Docker (testing)](https://hub.docker.com/r/izihawa/summa-server/testing)
- [Docker (v0.11.0)](https://hub.docker.com/r/izihawa/summa-server/0.11.0)

### Clients

- [Python](https://pypi.org/project/aiosumma/)
- [Rust (proto)](https://lib.rs/crates/summa-proto)