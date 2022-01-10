---
title: Architecture
parent: Core
nav_order: 1
---

Summa is composed of multiple various parts, most important ones are 
- [Tantivy](https://github.com/quickwit-oss/tantivy) using to do search operations in indices
- [Iroh](https://github.com/n0-computer/iroh) allowing to download and distribute indices through [IPFS](https://ipfs.tech) network
- [WASM](/summa/core/wasm) for compiling and launching the subset of Summa in browsers 

Summa Server combines all server parts together. It operates indices, put them in Iroh Store and manages Iroh
P2P for making indices available through IPFS network.

The main object in Summa is `Index` that represents a set of data with common [schema](/summa/core/schema) and backed with one of available `IndexEngine`.
`IndexEngine` encapsulates all IO operations. There are ready implementations for `Memory`, `File`, `IPFS` and `Remote`-backed indices.

### Index Engines

#### Memory

Ephemeral index having only schema persisted but data is wiped on every server restart.

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

`File` engine is a main engine for creating persistent search.

#### IPFS

`IPFS` engine stores index data in Iroh Store and serves queries directly from the store. It eliminates duplication
of files for indices that are using both for serving queries and replication.
Keeping files in Iroh Store adds an intermediate layer for reading, so you should enable cache for alleviation IO penalty.

```bash 
# Migrate existing File index to IPFS engine
summa-cli 0.0.0.0:8082 migrate-index test_index test_index_ipfs Ipfs
```

After the last step you will see CID of the index that may be used for replication and accessing index through Summa WASM bindings in browsers.

Now, you may get your index at another Summa instance:
```bash 
summa-cli 0.0.0.0:8082 attach-index test_index --ipfs '{"cid": "<cid from previous step>" }'
```

IPFS indices are mutable. After every commit all data is put to Iroh Store and you will obtain new CID for your index.

#### Remote

`Remote` engine allows you to create search retrieving index files from any remote HTTP storage (s3 including).

### Replication

### Aliases
Server tracks aliases for indices and allows to atomically switch aliases.

