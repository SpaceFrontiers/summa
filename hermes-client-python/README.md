# Hermes Python client

Async Python client for the
[Hermes](https://github.com/SpaceFrontiers/hermes) gRPC search server.

## Installation

```bash
pip install hermes-client-python
```

Python 3.10 or newer is required.

## Quick start

```python
import asyncio

from hermes_client_python import HermesClient


async def main():
    async with HermesClient("localhost:50051") as client:
        await client.create_index(
            "articles",
            """
            index articles {
                field title: text<simple> [indexed, stored]
                field body: text<simple> [indexed, stored]
            }
            """,
        )

        indexed, error_count, errors = await client.index_documents(
            "articles",
            [
                {"title": "Hello World", "body": "First article"},
                {"title": "Hermes Search", "body": "Fast retrieval"},
            ],
        )
        if error_count:
            raise RuntimeError(errors)
        print(f"Indexed {indexed} documents")

        await client.commit("articles")

        results = await client.search(
            "articles",
            query={"match": {"field": "title", "text": "hello"}},
            fields_to_load=["title", "body"],
        )
        for hit in results.hits:
            print(hit.address, hit.score, hit.fields)

        if results.hits:
            document = await client.get_document("articles", results.hits[0].address)
            print(document.fields if document else "document not found")

        await client.delete_index("articles")


asyncio.run(main())
```

The context manager calls `connect()` and `close()` automatically. For manual
lifecycle management:

```python
client = HermesClient("localhost:50051")
await client.connect()
try:
    ...
finally:
    await client.close()
```

## Index management

```python
await client.create_index("articles", schema_sdl)
names = await client.list_indexes()
info = await client.get_index_info("articles")
print(info.num_docs, info.num_segments, info.vector_stats)

await client.force_merge("articles")
await client.reorder("articles")
await client.retrain_vector_index("articles")
await client.delete_index("articles")
```

`commit()` is required before newly indexed documents become searchable.

### Batch and streaming indexing

```python
indexed, error_count, errors = await client.index_documents(
    "articles",
    [
        {"title": "One", "tags": ["search", "rust"]},
        {"title": "Two", "tags": ["python"]},
    ],
)


async def documents():
    for number in range(10_000):
        yield {"title": f"Document {number}"}


streamed, stream_errors = await client.index_documents_stream("articles", documents())
```

Repeated list values become repeated field entries. Flat numeric lists are
dense vectors; lists of `(dimension, weight)` pairs are sparse vectors.

## Searching

Every search takes one `query` object whose single key matches a Hermes query
variant:

```python
# Exact term
await client.search(
    "articles",
    query={"term": {"field": "title", "term": "hermes"}},
)

# Tokenized full-text match
await client.search(
    "articles",
    query={"match": {"field": "body", "text": "fast retrieval"}},
)

# Recursive boolean query
await client.search(
    "articles",
    query={
        "boolean": {
            "must": [{"match": {"field": "body", "text": "retrieval"}}],
            "must_not": [{"term": {"field": "title", "term": "draft"}}],
        }
    },
)

# Dense vector query and optional reranking
await client.search(
    "articles",
    query={
        "dense_vector": {
            "field": "embedding",
            "vector": [0.1, 0.2, 0.3],
            "nprobe": 16,
        }
    },
    reranker={"field": "embedding", "vector": [0.1, 0.2, 0.3]},
    candidate_limit=20,
    limit=10,
    fields_to_load=["title"],
)

# Hybrid union fusion
await client.search(
    "articles",
    query={
        "fusion": {
            "method": "rrf",
            "rrf_k": 60,
            "queries": [
                {
                    "query": {
                        "sparse_vector": {
                            "field": "sparse_embedding",
                            "indices": [1, 5],
                            "values": [0.8, 0.2],
                        }
                    },
                    "weight": 1.0,
                },
                {
                    "query": {
                        "dense_vector": {
                            "field": "embedding",
                            "vector": [0.1, 0.2, 0.3],
                        }
                    },
                    "weight": 1.0,
                },
            ],
        }
    },
)
```

Other supported variants are `phrase`, `binary_dense_vector`, `boost`, `range`,
`prefix`, and `all`. Search results expose the full `DocAddress` needed by
`get_document()`:

```python
hit = results.hits[0]
document = await client.get_document("articles", hit.address)
```

## Deadlines and errors

Every RPC accepts an optional `timeout` in seconds. A per-call value overrides
the client default:

```python
async with HermesClient("localhost:50051", default_timeout=5.0) as client:
    results = await client.search(
        "articles",
        query={"all": {}},
        timeout=0.5,
    )
    await client.force_merge("articles", timeout=3600)
```

gRPC failures raise `grpc.RpcError` (normally
`grpc.aio.AioRpcError`). `get_document()` is the exception: it returns `None`
for `NOT_FOUND`.

```python
import grpc

try:
    await client.search("missing", query={"all": {}})
except grpc.RpcError as error:
    if error.code() == grpc.StatusCode.NOT_FOUND:
        print("index not found")
    else:
        raise
```

## Development

From `hermes-client-python`:

```bash
uv sync --group dev --group test
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/test_client_unit.py
```

The remaining tests are integration tests and expect a debug
`target/debug/hermes-server` binary. Regenerate checked-in protobuf stubs after
changing `hermes-proto/hermes.proto`:

```bash
uv run --group dev python generate_proto.py
```

## License

MIT

## Ranking diagnostics and recall traces

Both options default to false and preserve the requested ranking:

```python
response = await client.search(
    "articles",
    query={
        "fusion": {
            "queries": [
                {
                    "name": "title",
                    "query": {"match": {"field": "title", "text": "rust"}},
                },
                {"name": "body", "query": {"match": {"field": "body", "text": "rust"}}},
            ]
        }
    },
    include_rrf_scores=True,
    tracing=True,
)
for hit in response.hits:
    print(hit.score, hit.rrf_score, hit.rrf_contributions)
for shard in response.trace.shards:
    for branch in shard.queries:
        print(shard.shard_id, branch.query_name, branch.candidates)
```

`rrf_score` and per-branch votes use organic nomination ranks merged across all
shards, independently of L1 or reranker scores. Ranks start at 1. `ordinal=None`
is document context; ordinal 0 is a real passage. Backfilled and score-only
features contribute no votes.

The trace retains every shard's bounded branch nominations and selected results,
including candidates absent from the final page, plus query trees and common
filters. Candidates contain addresses, raw scores and ordinals; stored fields
are loaded only for returned hits. Tracing does not expand retrieval depth or
rerun individual Boolean clauses. Oversized diagnostics and unsupported backends
fail explicitly. See the [scoring and tracing contract](../docs/candidate-rescoring.md).

For a named, scoped L1 query, specify the complete scoring formula:

```python
l1 = {"formula": "0.2 * title + 0.8 * log1p(body) + 3 * rrf"}
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

```python
await client.force_merge("articles")  # Copy encoded data and retain tombstones.
await client.force_merge("articles", compact=True)  # Remove deleted rows physically.
info = await client.get_index_info("articles")
print(info.num_deleted_docs, info.physical_num_docs, info.deleted_ratio)
```

Compaction also works when there is only one segment. It preserves surviving
values and may change document addresses and BM25 statistics.

### Delete and upsert documents

Declare one text field `[primary]` in the schema. Deletion removes the document
and all of its chunks; upserts replace the entire document, including indexed-only
fields, and insert when the key is absent.

```python
await client.delete_document("articles", "obsolete-key")
await client.upsert_document(
    "articles",
    {"id": "article-42", "body": ["replacement chunk one", "replacement chunk two"]},
)
await client.commit("articles")

result = await client.delete_documents("articles", ["old-a", "old-b"])
print(result.accepted_count, result.errors)  # errors: [{"index": 0, "error": "..."}]
await client.commit("articles")  # publishes accepted operations
```

`upsert_documents` takes a list of complete replacement documents and returns the
same `DocumentMutationResult`. Single-document helpers raise on rejection. Missing
deletion keys are accepted. Commit before deleting/upserting a key with a pending
insertion or replacement. Limits are 100,000 deletion keys / 8 MiB key bytes and
1,000 replacement documents / 32 MiB encoded bytes. Mutations use the usual timeout
argument; an expired RPC may have staged work, so do not blindly retry replacements.
Broker commits are atomic within each partition. Physical cleanup remains
`await client.force_merge("articles", compact=True)`.
