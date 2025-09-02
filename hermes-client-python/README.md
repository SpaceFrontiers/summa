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
