![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)
[![Crates.io](https://img.shields.io/crates/v/summa.svg)](https://crates.io/crates/summa)

# Summa

Summa is a full-text WASM-compatible search server written in Rust.
Yes, you can launch it entirely inside your browser!

## Key Features

- Full-text index with a wide range of supported queries and ranking functions
- GRPC API, Python asynchronous client [library](https://izihawa.github.io/summa/python-api) and [CLI](/summa/python-api)
- [WASM-bindings](https://github.com/izihawa/summa/tree/master/summa-wasm) to launch Summa in browsers
- Open remote indices through network. We have already implemented [IPFS](https://izihawa.github.io/summa/ipfs-wasm-guide) support out of the box
- Kafka for indexing

## Online-documentation

- [github.io](https://izihawa.github.io)
- [docs.rs](https://docs.rs/summa)