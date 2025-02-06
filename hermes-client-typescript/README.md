# hermes-client-typescript

TypeScript client for [Hermes](https://github.com/SpaceFrontiers/hermes) search server.

## Installation

```bash
pnpm add hermes-client-typescript
```

## Usage

```typescript
import { HermesClient } from "hermes-client-typescript";

const client = new HermesClient("localhost:50051");
client.connect();

// Create index
await client.createIndex(
  "articles",
  `
  index articles {
    title: text indexed stored
    body: text indexed stored
  }
`,
);

// Index documents
await client.indexDocuments("articles", [
  { title: "Hello", body: "World" },
  { title: "Foo", body: "Bar" },
]);
await client.commit("articles");

// Search
const results = await client.search("articles", {
  term: ["title", "hello"],
});
for (const hit of results.hits) {
  console.log(hit.docId, hit.score, hit.fields);
}

// Clean up
client.close();
```

## API

### Index Management

- `createIndex(indexName, schema)` — Create a new index with SDL or JSON schema
- `deleteIndex(indexName)` — Delete an index
- `listIndexes()` — List all index names
- `getIndexInfo(indexName)` — Get index metadata (doc count, segments, schema)

### Document Indexing

- `indexDocuments(indexName, docs)` — Batch index documents, returns `[indexedCount, errorCount]`
- `indexDocument(indexName, doc)` — Index a single document
- `indexDocumentsStream(indexName, asyncIterable)` — Stream documents for indexing
- `commit(indexName)` — Commit pending changes, returns total doc count
- `forceMerge(indexName)` — Force merge segments

### Search

- `search(indexName, options)` — Search with term, boolean, sparse/dense vector queries
- `getDocument(indexName, docId)` — Get document by ID

### Search Options

```typescript
await client.search("docs", {
  // Term query
  term: ["field", "value"],

  // Boolean query
  boolean: {
    must: [["title", "hello"]],
    should: [["body", "world"]],
  },

  // Sparse vector (pre-tokenized)
  sparseVector: ["embedding", [1, 5, 10], [0.5, 0.3, 0.2]],

  // Sparse text (server-side tokenization)
  sparseText: ["embedding", "what is machine learning?"],

  // Dense vector
  denseVector: ["embedding", [0.1, 0.2, 0.3]],

  // Options
  limit: 10,
  offset: 0,
  fieldsToLoad: ["title", "body"],
  combiner: "sum", // "sum" | "max" | "avg"

  // L2 reranker: [field, queryVector, l1Limit]
  reranker: ["embedding", [0.1, 0.2], 100],
});
```

## Development

```bash
pnpm install
pnpm run generate  # regenerate proto stubs
pnpm run build     # compile TypeScript
```
