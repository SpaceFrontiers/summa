# hermes-wasm

Browser search and indexing through the shared Rust engine. `LocalIndex` is
writable; `RemoteIndex` (HTTP) and `IpfsIndex` (fetch callbacks) are read-only.
Remote readers cache slices in memory and optionally IndexedDB.

## Quick start

```bash
npm install hermes-wasm
```

```js
import init, { LocalIndex } from "hermes-wasm";

await init();
const index = await LocalIndex.create(`index articles {
  field id: text<raw> [primary, stored]
  field title: text<en_stem> [indexed, stored]
}`);
await index.addDocument({ id: "1", title: "Search with Rust" });
await index.commit();
const results = await index.search("rust", 10);
const { segment_id, doc_id } = results.hits[0].address;
const doc = await index.getDocument(segment_id, doc_id);
```

See [SDL](../docs/schema.md) and [query syntax](../docs/query-language.md).

## Persistent local indexes

`LocalIndex.withStorage(storage, sdl)` creates or reopens an index and saves
changed files on commit. Supply this interface:

```ts
interface IFilesStorage {
  write(name: string, buffer: ArrayBuffer): Promise<void>;
  get(name: string): Promise<ArrayBuffer | null>;
  delete(names: string[]): Promise<void>;
  list(): Promise<string[]>;
}
```

Use one writable index per storage namespace and atomic per-file replacement.
IndexedDB, OPFS, and encryption are storage-adapter choices.

## Remote indexes

```js
import init, { RemoteIndex, IpfsIndex } from "hermes-wasm";

await init();
const index = new RemoteIndex("https://example.com/my-index/");
await index.load_with_idb_cache(); // Use load() to skip persistent cache restore.
const results = await index.search("rust", 10);
await index.save_cache_to_idb();

const ipfs = new IpfsIndex("/ipfs/YOUR_CID");
await ipfs.load(fetchFn, sizeFn);
```

HTTP hosting must support range requests and cross-origin access when needed.
IPFS callbacks:

```ts
type FetchFn = (
  path: string,
  start?: number,
  end?: number,
) => Promise<Uint8Array>;
type SizeFn = (path: string) => Promise<number>;
```

## API reference

All three indexes support `search(query, limit)` and `searchStructured(request)`.
Other method names differ:

| Operation                      | LocalIndex                                        | RemoteIndex / IpfsIndex                                             |
| ------------------------------ | ------------------------------------------------- | ------------------------------------------------------------------- |
| Paginated search               | `searchOffset(query, limit, offset)`              | `search_offset(query, limit, offset)`                               |
| Stored document                | `getDocument(segmentId, docId)`                   | `get_document(segmentId, docId)`                                    |
| Selected fields                | `getDocumentWithFields(segmentId, docId, fields)` | `getDocumentWithFields(segmentId, docId, fields)`                   |
| Counts / fields                | `numDocs()`, `pendingDocs()`, `fieldNames()`      | `num_docs()`, `num_segments()`, `field_names()`, `default_fields()` |
| Ingest                         | `addDocument(doc)`, `addDocuments(docs)`          | —                                                                   |
| Publish / discard pending work | `commit()`, `abort()`                             | —                                                                   |

Remote indexes also expose `with_cache_size`, `cache_stats`, `network_stats`,
`reset_network_stats`, `export_cache`, `import_cache`, `save_cache_to_idb`,
`load_cache_from_idb`, and `clear_idb_cache`. `IndexRegistry` manages named HTTP
indexes with `add_remote`, `remove`, `list`, and `search`.

For exact signatures, use the generated `pkg/hermes_wasm.d.ts` after building.
Set diagnostics with `set_log_level("debug")`; the default is `warn`.

## Structured query API

```js
const results = await index.searchStructured({
  query: { match: { field: "title", text: "rust search" } },
  limit: 10,
  fieldsToLoad: ["title"],
});
// { hits: [{ address, score, doc: { title: "..." } }], total_hits }
```

Requests use camelCase; responses use snake_case. Supported variants are
`term`, `match`, `phrase`, `boolean`, `prefix`, `sparseVector`, `denseVector`,
and top-level `fusion`. Term/prefix payloads use `value`, while match/phrase
use `text`. Fusion supports `rrf` and `normalizedWeightedSum`.
See the [request definitions](src/query.rs); server L1 and reranker options
are not exposed by this adapter.

## Structured search diagnostics

`includeRrfScores: true` adds fusion attribution; `tracing: true` captures bounded
branch nominations and selected addresses without changing retrieval depth or
ranking. Discarded candidates have no stored fields. Traces contain one local
shard. The diagnostic window is capped at 10,000 and JSON responses at 64 MiB;
oversized exports fail. See [candidate diagnostics](../docs/candidate-rescoring.md).

## Delete and upsert local documents

A text primary key enables whole-document deletion and full replacement,
including all chunks and indexed-only values:

```js
await index.upsertDocument({ id: "1", title: "Updated title" });
await index.upsertDocument({ id: "1", title: "Latest title" });
await index.commit();
index.deleteDocument("1");
await index.commit();
```

Staged rows may be replaced/deleted before commit; the latest accepted version
wins. Optional [content hashes](../docs/content-deduplication.md) skip unchanged
upserts. `deleteDocuments(keys)` and `await upsertDocuments(docs)` return
`{ acceptedCount, errors: [{ index, error }] }`; inspect errors and commit accepted
work. `abort()` discards the entire pending transaction.

A failed builder requires abort. A failed storage commit can be retried without
replaying mutations. Batches allow 100,000 deletion keys / 8 MiB key bytes or
1,000 replacements / 32 MiB JSON. Physical compaction uses native core, CLI, or
the server. See [row deletion](../docs/row-deletion.md).

## Building

From the repository root, with `wasm-pack`, LLVM, and Node.js 22.12+:

```bash
cd hermes-wasm
bash build.sh
npm ci
npm test -- --run
```

`build.sh` selects available LLVM tools for zstd cross-compilation and generates
`pkg/`. The root `package.json` is the private test harness; `pkg/` is the
publishable package. For a browser UI, see [hermes-web](../hermes-web/README.md).
