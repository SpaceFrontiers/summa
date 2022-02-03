---
title: Core
nav_order: 1
has_children: true 
has_toc: true
---

Summa is composed of multiple parts, the most important of which are
- [Tantivy](https://github.com/quickwit-oss/tantivy) for creating indices and searching in them
- [Iroh](https://github.com/n0-computer/iroh) for downloading and distributing indices through [IPFS](https://ipfs.tech) network
- [WASM](https://github.com/izihawa/summa/tree/master/summa-wasm) for compiling and launching the subset of Summa in browsers 

Summa Server combines Tantivy and Iroh together. It operates indices and puts them in the Iroh Store, and manages Iroh
P2P for making indices available through the IPFS network.

![architecture](/summa/assets/arch.drawio.png)

The main object in Summa is `Index` that represents a set of data with common [schema](/summa/core/schema) and backed with one of available `IndexEngine`.
`IndexEngine` encapsulates all I/O operations. There are ready implementations for `Memory`, `File`, `IPFS` and `Remote`-backed indices.

### Index Engines

#### Memory

Ephemeral index, with only the schema persisted, but data is wiped on every server restart

```bash
# Create new index
summa-cli 0.0.0.0:8082 create-index test_index Memory \
'[{"name": "title", "type": "text", "options": {"indexing": {"fieldnorms": True, "record": "position", "tokenizer": "default"}, "stored": True}}]'

# Add 3 documents
summa-cli 0.0.0.0:8082 index-document test_index '{"title": "Star Wars"}'
summa-cli 0.0.0.0:8082 index-document test_index '{"title": "2001: A Space Odyssey"}'
summa-cli 0.0.0.0:8082 index-document test_index '{"title": "War of the Worlds"}'

# Commit index
summa-cli 0.0.0.0:8082 commit-index test_index

# Do search
summa-cli 0.0.0.0:8082 search \
'[{"index_alias": "test_index", "query": {"term": {"field": "title", "value": "war"}}, "collectors": [{"top_docs": {"limit": 10}}, {"count": {}}]}]'
```

#### File

Main engine for creating persistent search index. It is the same as memory but backed with files.

#### IPFS

Engine stores index data in Iroh Store and serves queries directly from the store. 
This engine follows two purposes:
- Eliminate files duplication for indices that are using both for serving queries and replication
- Allow Iroh P2P to replicate index files

Keeping files in Iroh Store adds an intermediate layer for reading, so you should enable cache for alleviation IO penalty.

IPFS indices are mutable. After every commit all data is put to Iroh Store and you will obtain new CID for your index.

#### Remote

`Remote` engine allows you to create search that retrieves index files from any remote HTTP storage (s3 including) on demand.

### Replication

Replication is delegated to Iroh. At startup, Summa becomes a part of IPFS swarm capable to distribute files through IPFS to its peers.
You may create and configure your own private swarm or use public swarms if you need to distribute your data publicly.

#### Kick-start Replication
Your index must have IPFS engine for becoming replicatable:

```bash
# Migrate existing File index to IPFS engine
summa-cli 0.0.0.0:8082 migrate-index test_index test_index_ipfs Ipfs
```

After the last step, you will see the CID of the index that can be used for replication and accessing the index through Summa WASM bindings in browsers.
You can retrieve it via usual IPFS tools or directly attach it at another Summa Server instance:

```bash
summa-cli 0.0.0.0:8082 attach-index test_index '{"ipfs": {"cid": "<cid from the previous step>" }}'
```

### Aliases
Server tracks aliases for indices and allows to atomically switch aliases:

```bash
summa-cli 0.0.0.0:8082 set-index-alias test_index test_index_20220113
```
