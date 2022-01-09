---
title: Architecture
parent: Core
nav_order: 1
---

Summa is composed of multiple various parts, most important ones are 
- [Iroh](https://github.com/n0-computer/iroh) allowing to download and distribute indices through [IPFS](https://ipfs.tech) network
- [Tantivy](https://github.com/quickwit-oss/tantivy) using to do search operations in indices
- [WASM](/summa/core/wasm) for compiling and launching the subset of Summa in browsers 

Summa Server combines all server parts together. It operates indices, put them in Iroh Store and manages Iroh
P2P for making indices available through IPFS network.

The main object in Summa is `Index` that represents a set of data with common [schema](/summa/core/schema) and backed with one of available `IndexEngine`.
`IndexEngine` encapsulates all IO operations. There are ready implementations for File, IPFS and network -backed indices.

### Aliases
Server tracks aliases for indices and allows to atomically switch aliases.

