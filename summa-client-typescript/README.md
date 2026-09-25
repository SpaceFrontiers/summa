# Summa TypeScript client

Typed Node.js client for the
[Summa](https://github.com/SpaceFrontiers/summa) gRPC search server.

## Installation

```bash
pnpm add summa-client-typescript
```

## Quick start

```typescript
import { SummaClient } from "summa-client-typescript";

const client = new SummaClient("localhost:50051");
client.connect();

try {
  await client.createIndex(
    "articles",
    `
      index articles {
        field id: text<raw> [primary, stored]
        field title: text<simple> [indexed, stored]
        field body: text<simple> [indexed, stored]
      }
    `,
  );

  const [indexedCount, errorCount, errors] = await client.indexDocuments(
    "articles",
    [
      { id: "1", title: "Hello", body: "First article" },
      { id: "2", title: "Summa", body: "Fast search" },
    ],
  );
  if (errorCount) throw new Error(JSON.stringify(errors));
  console.log(`Indexed ${indexedCount} documents`);

  await client.commit("articles");

  const results = await client.search("articles", {
    query: { match: { field: "title", text: "hello" } },
    fieldsToLoad: ["title", "body"],
  });

  for (const hit of results.hits) {
    console.log(hit.address, hit.score, hit.fields);
  }

  if (results.hits.length > 0) {
    const document = await client.getDocument(
      "articles",
      results.hits[0].address,
    );
    console.log(document?.fields);
  }
} finally {
  client.close();
}
```

Call `connect()` before the first RPC and `close()` when finished.

## Index management

```typescript
await client.listIndexes();
const info = await client.getIndexInfo("articles");
await client.reorder("articles");
await client.retrainVectorIndex("articles");
```

`indexDocuments` returns `[indexedCount, errorCount, errors]`;
`indexDocumentsStream` takes an async iterable. Inspect errors, then `commit()`
to publish accepted work. Arrays hold repeated values; flat numeric arrays are
dense vectors. One sparse vector uses `[[[1, 0.5], [8, 0.25]]]`: the outer array
is required because `[[1, 0.5], [8, 0.25]]` means two dense vectors.

### Delete and upsert documents

With a text primary key, delete by exact key or supply a complete replacement:

```typescript
await client.deleteDocument("articles", "2");
await client.upsertDocument("articles", { id: "1", title: "Updated title" });
await client.commit("articles");
```

`deleteDocuments` / `upsertDocuments` return `DocumentMutationResult` with
`acceptedCount` and `errors: [{index, error}]`. Single-item helpers throw on
rejection. Missing deletes succeed; upserts insert missing keys. Staged rows can
be replaced/deleted again before commit; the latest accepted version wins.
Optional [content hashes](../docs/content-deduplication.md) skip unchanged writes.

Limits: 100,000 deletion keys / 8 MiB key bytes; 1,000 replacements / 32 MiB
encoded protobuf, or one replacement / 200 MiB including the request envelope.
Broker commits are atomic per partition. See [mutation semantics](../docs/row-deletion.md).

### Compact deleted rows

```typescript
await client.forceMerge("articles"); // Retain tombstones.
await client.forceMerge("articles", undefined, true); // Physically remove deleted rows.
```

The second argument remains the timeout in milliseconds. Compaction handles
singletons and may change addresses and BM25 scores. Index info exposes
`numDocs`, `physicalNumDocs`, `numDeletedDocs`, and `deletedRatio`.

## Searching

`search(indexName, request)` accepts a typed `SearchRequest`. Its `query` selects
one variant: `term`, `match`, `phrase`, `boolean`, `sparseVector`, `denseVector`,
`binaryDenseVector`, `boost`, `range`, `prefix`, `all`, or `fusion`.
See [client types](src/types.ts) and the [protocol](../summa-proto/summa.proto).

```typescript
const results = await client.search("articles", {
  query: {
    boolean: {
      must: [{ match: { field: "title", text: "search" } }],
      mustNot: [{ term: { field: "title", term: "draft" } }],
    },
  },
  fieldsToLoad: ["title"],
});
```

`getDocument(indexName, hit.address)` uses the full segment/document address
and returns `null` on `NOT_FOUND`. Use primary keys for durable identity.

## Ranking diagnostics and recall traces

Search options `includeRrfScores: true` and `tracing: true` default to false.
RRF diagnostics describe organic branch nominations; traces retain bounded
candidates and query trees, including hits outside the final page. Neither
changes retrieval depth or ranking. Oversized exports fail explicitly.

For named branches, use `l1: { formula: "0.2 * title + 0.8 * body + 3 * rrf" }`.
The formula is the only L1 scoring interface; old coefficient fields are removed.
See [candidate scoring](../docs/candidate-rescoring.md) for backfill, passage
selection, expression limits, capability versions, and distributed behavior.

## Deadlines

RPCs accept a trailing timeout in milliseconds, overriding the constructor's
`defaultTimeoutMs`. Expired calls reject with gRPC `DEADLINE_EXCEEDED`.
An expired mutation may already be staged, and an accepted commit continues
after disconnection. Resolve the outcome before retrying replacements.

## Development

```bash
pnpm install --frozen-lockfile
pnpm check
```

`check` compiles strict TypeScript and runs converter tests. After
[protocol changes](../summa-proto/README.md#regeneration-and-validation), run
`pnpm generate && pnpm check` and include the generated source.
