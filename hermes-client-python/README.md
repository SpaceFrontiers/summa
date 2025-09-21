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
                field id: text<raw> [primary, stored]
                field title: text<simple> [indexed, stored]
                field body: text<simple> [indexed, stored]
            }
            """,
        )

        indexed, error_count, errors = await client.index_documents(
            "articles",
            [
                {"id": "1", "title": "Hello World", "body": "First article"},
                {"id": "2", "title": "Hermes Search", "body": "Fast retrieval"},
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


asyncio.run(main())
```

The context manager connects and closes the channel. Manual callers use
`await client.connect()` / `await client.close()`.

## Index management

```python
await client.list_indexes()
info = await client.get_index_info("articles")
await client.reorder("articles")
await client.retrain_vector_index("articles")
```

`index_documents` returns `(indexed_count, error_count, errors)`;
`index_documents_stream` takes an async iterable and returns counts. Inspect
errors, then `commit()` to publish accepted work. Repeated values are lists;
flat numeric lists are dense vectors, `(dimension, weight)` pairs are sparse
vectors. See [client types](src/hermes_client_python/types.py).

### Delete and upsert documents

With a text primary key, delete by exact key or supply a complete replacement:

```python
await client.delete_document("articles", "2")
await client.upsert_document("articles", {"id": "1", "title": "Updated title"})
await client.commit("articles")
```

`delete_documents` / `upsert_documents` return `DocumentMutationResult` with
`accepted_count` and `errors: [{index, error}]`. Single-item helpers raise on
rejection. Missing deletes succeed; upserts insert missing keys. Staged rows can
be replaced/deleted again before commit; the latest accepted version wins.
Optional [content hashes](../docs/content-deduplication.md) skip unchanged writes.

Limits: 100,000 deletion keys / 8 MiB key bytes; 1,000 replacements / 32 MiB
encoded protobuf, or one replacement / 200 MiB including the request envelope.
Broker commits are atomic per partition. See [mutation semantics](../docs/row-deletion.md).

### Compact deleted rows

```python
await client.force_merge("articles")  # Retain tombstones.
await client.force_merge("articles", compact=True)  # Physically remove deleted rows.
```

Compaction handles singleton segments and may change addresses and BM25 scores.
Index info exposes `num_docs`, `physical_num_docs`, `num_deleted_docs`, and
`deleted_ratio`. Use primary keys for durable identity.

## Searching

`search(index_name, query=..., limit=10, fields_to_load=[...])` accepts one query
variant: `term`, `match`, `phrase`, `boolean`, `sparse_vector`, `dense_vector`,
`binary_dense_vector`, `boost`, `range`, `prefix`, `all`, or `fusion`.
See [query types](src/hermes_client_python/types.py) and the
[wire contract](../hermes-proto/hermes.proto) for options.

```python
results = await client.search(
    "articles",
    query={"boolean": {
        "must": [{"match": {"field": "title", "text": "search"}}],
        "must_not": [{"term": {"field": "title", "term": "draft"}}],
    }},
    fields_to_load=["title"],
)
```

`get_document(index_name, hit.address)` uses the full segment/document address
and returns `None` on `NOT_FOUND`.

## Ranking diagnostics and recall traces

Search options `include_rrf_scores=True` and `tracing=True` default to false.
RRF diagnostics describe organic branch nominations; traces retain bounded
candidates and query trees, including hits outside the final page. Neither
changes retrieval depth or ranking. Oversized exports fail explicitly.

For named branches, use `l1={"formula": "0.2 * title + 0.8 * body + 3 * rrf"}`.
The formula is the only L1 scoring interface; old coefficient fields are removed.
See [candidate scoring](../docs/candidate-rescoring.md) for backfill, passage
selection, expression limits, capability versions, and distributed behavior.

## Deadlines and errors

RPCs accept `timeout` in seconds, overriding the constructor's `default_timeout`.
gRPC failures raise `grpc.aio.AioRpcError`, except document `NOT_FOUND` as above.
An expired mutation may already be staged, and an accepted commit continues
after disconnection. Resolve the outcome before retrying replacements.

## Development

From this directory:

```bash
uv sync --group dev --group test
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/test_client_unit.py
uv run --group dev python generate_proto.py
```

Integration tests require `target/debug/hermes-server`. Regenerate bindings after
[protocol changes](../hermes-proto/README.md#regeneration-and-validation).
