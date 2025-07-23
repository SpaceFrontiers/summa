# Hermes TypeScript client

Typed Node.js client for the
[Hermes](https://github.com/SpaceFrontiers/hermes) gRPC search server.

## Installation

```bash
pnpm add hermes-client-typescript
```

## Quick start

```typescript
import { HermesClient } from "hermes-client-typescript";

const client = new HermesClient("localhost:50051");
client.connect();

try {
  await client.createIndex(
    "articles",
    `
      index articles {
        field title: text<simple> [indexed, stored]
        field body: text<simple> [indexed, stored]
      }
    `,
  );

  const [indexedCount, errorCount, errors] = await client.indexDocuments(
    "articles",
    [
      { title: "Hello", body: "First article" },
      { title: "Hermes", body: "Fast search" },
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

Call `connect()` before the first RPC and `close()` when the client is no
longer needed.

## Index management

```typescript
await client.createIndex("articles", schema);
const names = await client.listIndexes();
const info = await client.getIndexInfo("articles");

await client.forceMerge("articles");
await client.reorder("articles");
await client.retrainVectorIndex("articles");
await client.deleteIndex("articles");
```

Newly indexed documents become searchable after `commit()`.

### Batch and streaming indexing

```typescript
const [indexed, errorCount, errors] = await client.indexDocuments("articles", [
  { title: "One", tags: ["search", "typescript"] },
  { title: "Two", tags: ["grpc"] },
]);

async function* documents() {
  for (let number = 0; number < 10_000; number += 1) {
    yield { title: `Document ${number}` };
  }
}

const streamed = await client.indexDocumentsStream("articles", documents());
```

Repeated arrays become repeated field entries. Flat numeric arrays are dense
vectors. Sparse vectors use arrays of `[dimension, weight]` pairs inside an
outer repeated-value array—for example, `[[[1, 0.5], [8, 0.25]]]` for one
sparse vector. The outer array is required because `[[1, 0.5], [8, 0.25]]` is
the legacy shape for two dense vectors.

## Searching

`search()` accepts a `SearchRequest`. Its `query` is a discriminated union, so
exactly one query variant is selected:

```typescript
// Exact term
await client.search("articles", {
  query: { term: { field: "title", term: "hermes" } },
});

// Recursive Boolean query
await client.search("articles", {
  query: {
    boolean: {
      must: [{ match: { field: "body", text: "fast search" } }],
      mustNot: [{ term: { field: "title", term: "draft" } }],
    },
  },
});

// Dense retrieval with reranking
await client.search("articles", {
  query: {
    denseVector: {
      field: "embedding",
      vector: [0.1, 0.2, 0.3],
      nprobe: 16,
    },
  },
  reranker: {
    field: "embedding",
    vector: [0.1, 0.2, 0.3],
  },
  candidateLimit: 20,
  limit: 10,
  fieldsToLoad: ["title"],
});

// Hybrid union fusion
await client.search("articles", {
  query: {
    fusion: {
      method: "rrf",
      rrfK: 60,
      queries: [
        {
          query: {
            sparseVector: {
              field: "sparseEmbedding",
              indices: [1, 5],
              values: [0.8, 0.2],
            },
          },
        },
        {
          query: {
            denseVector: {
              field: "embedding",
              vector: [0.1, 0.2, 0.3],
            },
          },
        },
      ],
    },
  },
});
```

Supported variants are `term`, `match`, `boolean`, `sparseVector`,
`denseVector`, `binaryDenseVector`, `boost`, `range`, `prefix`, `all`, and
`fusion`.

`getDocument()` takes the full address returned by a search hit:

```typescript
const document = await client.getDocument("articles", hit.address);
```

It returns `null` when the server responds with gRPC `NOT_FOUND`.

## Deadlines

Every RPC accepts an optional trailing deadline in milliseconds. A per-call
value overrides the client default:

```typescript
const client = new HermesClient("localhost:50051", {
  defaultTimeoutMs: 5_000,
});

await client.search("articles", { query: { all: {} } }, 500);
await client.forceMerge("articles", 3_600_000);
```

Expired calls reject with a gRPC `DEADLINE_EXCEEDED` error.

## Development

```bash
pnpm install --frozen-lockfile
pnpm check
```

`pnpm check` compiles the strict TypeScript sources and runs the pure converter
unit tests. After changing `hermes-proto/hermes.proto`, regenerate and check in
the generated source:

```bash
pnpm generate
pnpm check
```

## License

MIT
