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

Supported variants are `term`, `match`, `phrase`, `boolean`, `sparseVector`,
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

client.connect();
try {
  await client.search("articles", { query: { all: {} } }, 500);
  await client.forceMerge("articles", 3_600_000);
} finally {
  client.close();
}
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

## Ranking diagnostics and recall traces

Both options default to false and preserve the requested ranking:

```typescript
const response = await client.search("articles", {
  query: {
    fusion: {
      queries: [
        { name: "title", query: { match: { field: "title", text: "rust" } } },
        { name: "body", query: { match: { field: "body", text: "rust" } } },
      ],
    },
  },
  includeRrfScores: true,
  tracing: true,
});
for (const hit of response.hits) {
  console.log(hit.score, hit.rrfScore, hit.rrfContributions);
}
for (const shard of response.trace?.shards ?? []) {
  for (const branch of shard.queries) {
    console.log(shard.shardId, branch.queryName, branch.candidates);
  }
}
```

RRF diagnostics use organic nomination ranks merged across all shards, separate
from L1/reranker scores. Each vote carries the branch index/name, one-based rank,
weighted contribution and optional ordinal. An absent ordinal denotes document
context; ordinal 0 denotes a real passage. Backfilled and score-only features do
not vote.

The trace retains all bounded branch nominations, query trees, common filters
and shard selections, including candidates discarded before the final page.
Trace candidates contain addresses, raw scores and ordinals, without stored
fields. Tracing does not increase retrieval depth or rerun Boolean clauses.
Oversized diagnostics and unsupported backends fail explicitly. See the
[scoring and tracing contract](../docs/candidate-rescoring.md).

For a named, scoped L1 query, specify the complete scoring formula:

```typescript
l1: {
  formula: "0.2 * title + 0.8 * log1p(body) + 3 * rrf";
}
```

`formula` is the only L1 scoring interface. Coefficients, offsets and RRF
multipliers go in the expression; the former coefficient fields are removed.
Arithmetic, powers, logarithms, `sqrt`, `abs`, `exp`, `min`/`max` and trigonometry
are supported. Use `{body.bm25}` for punctuated branch names. `log` and `ln` are
natural logarithms; `log2` and `log10` select those bases. Missing branch values
use configured missing defaults, otherwise zero. Backfill remains optional.

The formula runs before passage selection and the document combiner. A formula
using `rrf` makes the broker obtain the complete bounded candidate and passage
union before global inference. Exports that exceed budgets fail explicitly.
Expressions are bounded to 4 KiB, 256 tokens and 32 parenthesis levels. Invalid
variables, invalid syntax and non-finite predictions fail explicitly. Servers
and brokers must support `formula_v1` (`candidate_scoring_version=3`).

### Compact deleted rows

```typescript
await client.forceMerge("articles"); // Retain deletion masks.
await client.forceMerge("articles", undefined, true); // Physically compact final outputs.
const info = await client.getIndexInfo("articles");
console.log(info.numDeletedDocs, info.physicalNumDocs, info.deletedRatio);
```

The optional second argument remains the timeout in milliseconds. Compaction
also handles a singleton and can change document addresses and BM25 statistics.

### Delete and upsert documents

With a text field declared `[primary]`, delete by exact key and replace by passing
the complete document. Every chunk belongs to its document and is deleted with it.

```typescript
await client.deleteDocument("articles", "obsolete-key");
await client.upsertDocument("articles", {
  id: "article-42",
  body: ["replacement chunk one", "replacement chunk two"],
});
await client.commit("articles");

const result = await client.deleteDocuments("articles", ["old-a", "old-b"]);
console.log(result.acceptedCount, result.errors); // errors: [{ index, error }]
await client.commit("articles"); // publishes accepted operations
```

`upsertDocuments` accepts a list of full replacements and returns the same
`DocumentMutationResult`. Single-item helpers throw on rejection. Missing deletes
are accepted; upserts insert missing keys. Commit before replacing/deleting a key with
a pending insertion. Limits are 100,000 deletion keys / 8 MiB key bytes and 1,000
replacement documents / 32 MiB encoded bytes. Normal deadlines apply; an expired
RPC can have staged work, so do not blindly retry replacements. Broker commits
are atomic within each partition. Physical cleanup remains
`await client.forceMerge("articles", undefined, true)`.
