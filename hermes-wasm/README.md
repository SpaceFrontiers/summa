# hermes-wasm

WebAssembly bindings for the [Hermes](https://github.com/SpaceFrontiers/hermes) search engine. Run a full-text search engine entirely in the browser — including indexing, BM25 ranking, and document storage.

## Features

- **Local indexing** — create indexes, add documents, commit, and search entirely in WASM
- **Pluggable persistence** — bring your own storage (IDB, encrypted, OPFS) via simple JS interface
- **Remote search** — load pre-built indexes over HTTP with slice caching
- **IPFS support** — load indexes from IPFS via JavaScript fetch callbacks
- **15+ language stemmers** — English, German, French, Spanish, Russian, Arabic, and more
- **Query language** — `field:term`, `AND`, `OR`, `NOT`, grouping with parentheses
- **~3 MB** WASM binary (gzipped ~1.2 MB)

## Quick Start

```bash
npm install hermes-wasm
```

### In-Memory Index

```js
import init, { LocalIndex } from "hermes-wasm";

await init();

// Define schema using SDL
const index = await LocalIndex.create(`
  index articles {
    field title: text<en_stem> [indexed, stored]
    field body:  text<en_stem> [indexed, stored]
    field views: u64 [indexed, stored]
  }
`);

// Add documents
await index.addDocuments([
  {
    title: "Rust Programming",
    body: "Rust is a systems language.",
    views: 1500,
  },
  { title: "Search Engines", body: "BM25 is a ranking function.", views: 800 },
]);

// Commit (builds the segment)
await index.commit();

// Search
const results = await index.search("rust", 10);
// { hits: [{ address: { segment_id, doc_id }, score }], total_hits: 1 }

// Get document
const doc = await index.getDocument(
  results.hits[0].address.segment_id,
  results.hits[0].address.doc_id,
);
// { title: "Rust Programming", body: "Rust is a systems language.", views: 1500 }
```

### Persistent Index (Custom Storage)

Bring your own storage backend — IndexedDB, encrypted storage, OPFS, or anything async:

```ts
// Your storage must implement this interface:
interface IFilesStorage {
  write(name: string, buffer: ArrayBuffer): Promise<void>;
  get(name: string): Promise<ArrayBuffer | null>;
  delete(names: string[]): Promise<void>;
  list(): Promise<string[]>;
}
```

```js
// Create with custom storage — auto-saves changed files on each commit
const index = await LocalIndex.withStorage(myStorage, schema);
await index.addDocuments(docs);
await index.commit(); // only new segment files written to storage

// Later (page reload) — same call reopens if storage has files
const index = await LocalIndex.withStorage(myStorage, schema);
const results = await index.search("rust", 10); // works immediately
```

A simple IndexedDB implementation:

```js
class IdbStorage {
  constructor(name) {
    this.prefix = `idx:${name}:`;
  }
  async write(name, buffer) {
    /* idb put this.prefix + name → buffer */
  }
  async get(name) {
    /* idb get this.prefix + name */
  }
  async delete(names) {
    /* idb delete each this.prefix + name */
  }
  async list() {
    /* idb getAllKeys matching this.prefix, strip prefix */
  }
}

const index = await LocalIndex.withStorage(new IdbStorage("articles"), schema);
```

### Remote Index (HTTP)

```js
import init, { RemoteIndex } from "hermes-wasm";

await init();

const index = new RemoteIndex("https://example.com/my-index/");
await index.load_with_idb_cache(); // loads with IndexedDB cache for fast reload

const results = await index.search("query", 10);

// Fetch a document
const doc = await index.get_document(
  results.hits[0].address.segment_id,
  results.hits[0].address.doc_id,
);
```

## API Reference

### `LocalIndex`

| Method                                     | Description                                        |
| ------------------------------------------ | -------------------------------------------------- |
| `LocalIndex.create(sdl)`                   | Create in-memory index from SDL schema             |
| `LocalIndex.withStorage(storage, sdl)`     | Create or open index with pluggable storage        |
| `index.addDocument(json)`                  | Add a single document                              |
| `index.addDocuments(jsonArray)`            | Add multiple documents, returns count              |
| `index.commit()`                           | Commit pending docs, sync to storage if configured |
| `index.search(query, limit)`               | Search with BM25 ranking                           |
| `index.searchOffset(query, limit, offset)` | Search with pagination                             |
| `index.getDocument(segmentId, docId)`      | Retrieve stored document                           |
| `index.numDocs()`                          | Count of committed documents                       |
| `index.pendingDocs()`                      | Count of uncommitted documents                     |
| `index.fieldNames()`                       | List of field names                                |

### `RemoteIndex`

| Method                                      | Description                         |
| ------------------------------------------- | ----------------------------------- |
| `new RemoteIndex(url)`                      | Create remote index pointing to URL |
| `RemoteIndex.with_cache_size(url, bytes)`   | Create with custom cache size       |
| `index.load()`                              | Load index metadata and segments    |
| `index.load_with_idb_cache()`               | Load with IndexedDB cache pre-fill  |
| `index.search(query, limit)`                | Search                              |
| `index.search_offset(query, limit, offset)` | Search with pagination              |
| `index.get_document(segmentId, docId)`      | Retrieve document                   |
| `index.num_docs()`                          | Document count                      |
| `index.num_segments()`                      | Segment count                       |
| `index.field_names()`                       | Field names                         |
| `index.default_fields()`                    | Default search fields               |
| `index.export_cache()`                      | Export slice cache as `Uint8Array`  |
| `index.import_cache(data)`                  | Import previously exported cache    |
| `index.save_cache_to_idb()`                 | Persist slice cache to IndexedDB    |
| `index.load_cache_from_idb()`               | Restore cache from IndexedDB        |
| `index.clear_idb_cache()`                   | Remove persisted cache              |
| `index.cache_stats()`                       | Cache utilization info              |
| `index.network_stats()`                     | HTTP request statistics             |
| `index.reset_network_stats()`               | Clear network statistics            |

### `IpfsIndex`

Same API as `RemoteIndex` but loaded via JavaScript callbacks instead of HTTP:

| Method                                       | Description                              |
| -------------------------------------------- | ---------------------------------------- |
| `new IpfsIndex(basePath)`                    | Create with IPFS path (e.g. `/ipfs/Qm…`) |
| `IpfsIndex.with_cache_size(path, bytes)`     | Create with custom cache size            |
| `index.load(fetchFn, sizeFn)`                | Load using JS callbacks                  |
| `index.load_with_idb_cache(fetchFn, sizeFn)` | Load with IDB cache + JS callbacks       |

`fetchFn`: `(path: string, rangeStart?: number, rangeEnd?: number) => Promise<Uint8Array>`
`sizeFn`: `(path: string) => Promise<number>`

All other methods (`search`, `get_document`, `cache_stats`, etc.) are identical to `RemoteIndex`.

### `IndexRegistry`

Manages multiple named remote indexes:

| Method                                | Description                    |
| ------------------------------------- | ------------------------------ |
| `new IndexRegistry()`                 | Create empty registry          |
| `registry.add_remote(name, url)`      | Load and register remote index |
| `registry.remove(name)`               | Remove index                   |
| `registry.list()`                     | List index names               |
| `registry.search(name, query, limit)` | Search a specific index        |

### Schema Definition Language (SDL)

```
index <name> {
    field <name>: <type> [attributes]
}
```

**Types:** `text`, `u64`, `i64`, `f64`, `bytes`

**Tokenizers:** `text<en_stem>`, `text<de_stem>`, `text<fr_stem>`, `text<es_stem>`, `text<it_stem>`, `text<pt_stem>`, `text<ru_stem>`, `text<ar_stem>`, `text<simple>`, etc.

**Attributes:** `indexed` (searchable), `stored` (retrievable), `fast` (columnar access)

### Query Language

| Syntax | Example                | Description                 |
| ------ | ---------------------- | --------------------------- |
| Term   | `rust`                 | Match across default fields |
| Field  | `title:rust`           | Match in specific field     |
| AND    | `rust AND web`         | Both terms required         |
| OR     | `rust OR python`       | Either term                 |
| NOT    | `rust NOT unsafe`      | Exclude term                |
| Group  | `(rust OR go) AND web` | Grouping                    |
| Phrase | `"search engine"`      | Exact phrase                |

## Building

```bash
cd hermes-wasm
bash build.sh  # requires Homebrew LLVM on macOS
```

The build script sets `CC` and `AR` to Homebrew LLVM binaries for zstd cross-compilation to `wasm32-unknown-unknown`.

## Example

Open `examples/index.html` in a browser (needs to be served, not opened as file):

```bash
cd hermes-wasm
bash build.sh
cp pkg/hermes_wasm.js pkg/hermes_wasm_bg.wasm examples/
cd examples && python3 -m http.server 8080
# Open http://localhost:8080
```

## Architecture

```
Browser JS
    │
    ├── LocalIndex (create/index/search in WASM)
    │       └── WasmIndexWriter → SegmentBuilder → RamDirectory
    │                                                   │
    │                                         [pluggable IFilesStorage]
    │                                         (IDB, encrypted, OPFS, ...)
    │
    ├── RemoteIndex (HTTP range requests)
    │       └── Searcher → SliceCachingDirectory → HttpDirectory
    │
    ├── IpfsIndex (JS fetch callbacks)
    │       └── Searcher → SliceCachingDirectory → JsFetchDirectory
    │
    └── IndexRegistry (multi-index management)
            └── Map<name, RemoteIndex>
```
