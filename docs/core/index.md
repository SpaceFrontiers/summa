---
title: Core
nav_order: 2
has_children: true 
has_toc: true
---

Summa is composed of multiple parts, the most important of which are
- [Tantivy](https://github.com/quickwit-oss/tantivy) for creating indices and searching in them
- [IPFS](https://github.com/ipfs/kubo) for downloading and distributing indices through [IPFS](https://ipfs.tech) network
- [WASM](https://github.com/izihawa/summa/tree/master/summa-wasm) for compiling and launching the subset of Summa in browsers 

Summa Server operates indices. The main object in Summa is `Index` that represents a set of data with common [schema](/summa/core/schema) and backed with one of available `IndexEngine`.
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

#### Remote

`Remote` engine allows you to create search that retrieves index files from any remote HTTP storage (s3 including) on demand.

### Aliases
Server tracks aliases for indices and allows to atomically switch aliases:

```bash
summa-cli 0.0.0.0:8082 set-index-alias test_index test_index_20220113
```
