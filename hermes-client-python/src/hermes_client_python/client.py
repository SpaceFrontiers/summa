"""Async Hermes client implementation.

All search types mirror the proto API structure exactly.
See types.py for Query, Reranker definitions.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import grpc
from grpc import aio

from . import hermes_pb2 as pb
from . import hermes_pb2_grpc as pb_grpc
from .types import (
    DocAddress,
    Document,
    IndexInfo,
    OrdinalScore,
    SearchHit,
    SearchResponse,
    SearchTimings,
    VectorFieldStats,
)


class HermesClient:
    """Async client for Hermes search server.

    All search types mirror the proto API structure exactly.

    Example:
        async with HermesClient("localhost:50051") as client:
            # Create index
            await client.create_index("articles", '''
                index articles {
                    title: text indexed stored
                    body: text indexed stored
                }
            ''')

            # Index documents
            await client.index_documents("articles", [
                {"title": "Hello", "body": "World"},
                {"title": "Foo", "body": "Bar"},
            ])
            await client.commit("articles")

            # Search
            results = await client.search("articles",
                query={"match": {"field": "title", "text": "hello"}})
            for hit in results.hits:
                print(hit.address, hit.score)
    """

    def __init__(self, address: str = "localhost:50051"):
        """Initialize client.

        Args:
            address: Server address in format "host:port"
        """
        self.address = address
        self._channel: aio.Channel | None = None
        self._index_stub: pb_grpc.IndexServiceStub | None = None
        self._search_stub: pb_grpc.SearchServiceStub | None = None

    async def connect(self) -> None:
        """Connect to the server."""
        # Increase message size limits for large responses (e.g., loading content fields)
        options = [
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
        ]
        # Enable gzip compression for smaller message sizes over the wire
        self._channel = aio.insecure_channel(
            self.address,
            options=options,
            compression=grpc.Compression.Gzip,
        )
        self._index_stub = pb_grpc.IndexServiceStub(self._channel)
        self._search_stub = pb_grpc.SearchServiceStub(self._channel)

    async def close(self) -> None:
        """Close the connection."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._index_stub = None
            self._search_stub = None

    async def __aenter__(self) -> HermesClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _ensure_connected(self) -> None:
        if self._index_stub is None or self._search_stub is None:
            raise RuntimeError(
                "Client not connected. Use 'async with' or call connect() first."
            )

    # =========================================================================
    # Index Management
    # =========================================================================

    async def create_index(self, index_name: str, schema: str) -> bool:
        """Create a new index.

        Args:
            index_name: Name of the index
            schema: Schema definition in SDL or JSON format

        Returns:
            True if successful

        Example SDL schema:
            index myindex {
                field title: text [indexed, stored]
                field body: text [indexed, stored]
                field score: f64 [stored]
            }

        Example JSON schema:
            {
                "fields": [
                    {"name": "title", "type": "text", "indexed": true, "stored": true}
                ]
            }
        """
        self._ensure_connected()
        request = pb.CreateIndexRequest(index_name=index_name, schema=schema)
        response = await self._index_stub.CreateIndex(request)
        return response.success

    async def delete_index(self, index_name: str) -> bool:
        """Delete an index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if successful
        """
        self._ensure_connected()
        request = pb.DeleteIndexRequest(index_name=index_name)
        response = await self._index_stub.DeleteIndex(request)
        return response.success

    async def list_indexes(self) -> list[str]:
        """List all indexes on the server.

        Returns:
            List of index names
        """
        self._ensure_connected()
        request = pb.ListIndexesRequest()
        response = await self._index_stub.ListIndexes(request)
        return list(response.index_names)

    async def get_index_info(self, index_name: str) -> IndexInfo:
        """Get information about an index.

        Args:
            index_name: Name of the index

        Returns:
            IndexInfo with document count, segments, and schema
        """
        self._ensure_connected()
        request = pb.GetIndexInfoRequest(index_name=index_name)
        response = await self._search_stub.GetIndexInfo(request)
        vector_stats = [
            VectorFieldStats(
                field_name=vs.field_name,
                vector_type=vs.vector_type,
                total_vectors=vs.total_vectors,
                dimension=vs.dimension,
            )
            for vs in response.vector_stats
        ]
        return IndexInfo(
            index_name=response.index_name,
            num_docs=response.num_docs,
            num_segments=response.num_segments,
            schema=response.schema,
            vector_stats=vector_stats,
        )

    # =========================================================================
    # Document Indexing
    # =========================================================================

    async def index_documents(
        self, index_name: str, documents: list[dict[str, Any]]
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Index multiple documents in batch.

        Args:
            index_name: Name of the index
            documents: List of documents (dicts with field names as keys)

        Returns:
            Tuple of (indexed_count, error_count, errors) where errors is a list
            of dicts with 'index' (0-based position) and 'error' (message) keys.
        """
        self._ensure_connected()

        named_docs = []
        for doc in documents:
            fields = _to_field_entries(doc)
            named_docs.append(pb.NamedDocument(fields=fields))

        request = pb.BatchIndexDocumentsRequest(
            index_name=index_name, documents=named_docs
        )
        response = await self._index_stub.BatchIndexDocuments(request)
        errors = [{"index": e.index, "error": e.error} for e in response.errors]
        return response.indexed_count, response.error_count, errors

    async def index_document(self, index_name: str, document: dict[str, Any]) -> None:
        """Index a single document.

        Args:
            index_name: Name of the index
            document: Document as dict with field names as keys
        """
        await self.index_documents(index_name, [document])

    async def index_documents_stream(
        self, index_name: str, documents: AsyncIterator[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]]]:
        """Stream documents for indexing.

        Args:
            index_name: Name of the index
            documents: Async iterator of documents

        Returns:
            Tuple of (indexed_count, errors) where errors is a list of dicts
            with 'index' and 'error' keys.
        """
        self._ensure_connected()

        async def request_iterator():
            async for doc in documents:
                fields = _to_field_entries(doc)
                yield pb.IndexDocumentRequest(index_name=index_name, fields=fields)

        response = await self._index_stub.IndexDocuments(request_iterator())
        errors = [{"index": e.index, "error": e.error} for e in response.errors]
        return response.indexed_count, errors

    async def commit(self, index_name: str) -> int:
        """Commit pending changes.

        Args:
            index_name: Name of the index

        Returns:
            Total number of documents in the index
        """
        self._ensure_connected()
        request = pb.CommitRequest(index_name=index_name)
        response = await self._index_stub.Commit(request)
        return response.num_docs

    async def force_merge(self, index_name: str) -> int:
        """Force merge all segments.

        Args:
            index_name: Name of the index

        Returns:
            Number of segments after merge
        """
        self._ensure_connected()
        request = pb.ForceMergeRequest(index_name=index_name)
        response = await self._index_stub.ForceMerge(request)
        return response.num_segments

    async def retrain_vector_index(self, index_name: str) -> bool:
        """Retrain vector index centroids/codebooks from current data.

        Resets the trained ANN structures and rebuilds them from scratch.
        Use this when significant new data has been added and you want
        better centroids, or when the vector distribution has changed.

        Args:
            index_name: Name of the index

        Returns:
            True if successful
        """
        self._ensure_connected()
        request = pb.RetrainVectorIndexRequest(index_name=index_name)
        response = await self._index_stub.RetrainVectorIndex(request)
        return response.success

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        index_name: str,
        *,
        query: dict[str, Any],
        limit: int = 10,
        offset: int = 0,
        fields_to_load: list[str] | None = None,
        reranker: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Search for documents.

        All parameters mirror the proto SearchRequest structure exactly.
        ``query`` is a dict with exactly one key matching the proto Query oneof.

        Args:
            index_name: Name of the index
            query: Query dict with one key: "term", "match", "boolean",
                "sparse_vector", "dense_vector", "boost", or "all".
            limit: Maximum number of results
            offset: Offset for pagination
            fields_to_load: List of fields to include in results
            reranker: Reranker dict matching proto Reranker message

        Returns:
            SearchResponse with hits

        Examples:
            # Term query (exact single token)
            results = await client.search("articles",
                query={"term": {"field": "title", "term": "hello"}})

            # Match query (full-text, tokenized server-side)
            results = await client.search("articles",
                query={"match": {"field": "title", "text": "what is hemoglobin"}})

            # Boolean query
            results = await client.search("articles",
                query={"boolean": {
                    "must": [{"match": {"field": "title", "text": "hello"}}],
                    "should": [{"match": {"field": "body", "text": "world"}}],
                }})

            # Sparse text query (server-side tokenization) with pruning
            results = await client.search("docs",
                query={"sparse_vector": {
                    "field": "embedding",
                    "text": "machine learning",
                    "pruning": 0.5,
                }},
                fields_to_load=["title", "body"])

            # Sparse vector query (pre-computed)
            results = await client.search("docs",
                query={"sparse_vector": {
                    "field": "embedding",
                    "indices": [1, 5, 10],
                    "values": [0.5, 0.3, 0.2],
                }})

            # Dense vector query with reranker
            results = await client.search("docs",
                query={"dense_vector": {
                    "field": "embedding",
                    "vector": [0.1, 0.2, 0.3],
                    "nprobe": 10,
                }},
                reranker={
                    "field": "embedding",
                    "vector": [0.1, 0.2, 0.3],
                    "limit": 100,
                },
                fields_to_load=["title"])

        """
        self._ensure_connected()

        pb_query = _build_query(query)
        pb_reranker = _build_reranker(reranker) if reranker else None

        request = pb.SearchRequest(
            index_name=index_name,
            query=pb_query,
            limit=limit,
            offset=offset,
            fields_to_load=fields_to_load or [],
            reranker=pb_reranker,
        )

        response = await self._search_stub.Search(request)

        hits = [
            SearchHit(
                address=DocAddress(
                    segment_id=hit.address.segment_id,
                    doc_id=hit.address.doc_id,
                ),
                score=hit.score,
                fields={k: _from_field_value_list(v) for k, v in hit.fields.items()},
                ordinal_scores=[
                    OrdinalScore(ordinal=os.ordinal, score=os.score)
                    for os in hit.ordinal_scores
                ],
            )
            for hit in response.hits
        ]

        timings = None
        if response.HasField("timings"):
            t = response.timings
            timings = SearchTimings(
                search_us=t.search_us,
                rerank_us=t.rerank_us,
                load_us=t.load_us,
                total_us=t.total_us,
            )

        return SearchResponse(
            hits=hits,
            total_hits=response.total_hits,
            took_ms=response.took_ms,
            timings=timings,
        )

    async def get_document(
        self, index_name: str, address: DocAddress
    ) -> Document | None:
        """Get a document by address.

        Args:
            index_name: Name of the index
            address: DocAddress from a SearchHit

        Returns:
            Document or None if not found
        """
        self._ensure_connected()
        request = pb.GetDocumentRequest(
            index_name=index_name,
            address=pb.DocAddress(segment_id=address.segment_id, doc_id=address.doc_id),
        )
        try:
            response = await self._search_stub.GetDocument(request)
            fields = {k: _from_field_value_list(v) for k, v in response.fields.items()}
            return Document(fields=fields)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise


# =============================================================================
# Helper functions
# =============================================================================


def _is_sparse_vector(value: list) -> bool:
    """Check if list is a sparse vector: list of (int, float) pairs."""
    if not value:
        return False
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        idx, val = item
        if not isinstance(idx, int) or not isinstance(val, (int, float)):
            return False
    return True


def _is_multi_sparse_vector(value: list) -> bool:
    """Check if list is a multi-value sparse vector: list of sparse vectors."""
    if not value:
        return False
    # All items must be lists and each must be a valid sparse vector
    if not all(isinstance(item, list) for item in value):
        return False
    return all(_is_sparse_vector(item) for item in value)


def _is_dense_vector(value: list) -> bool:
    """Check if list is a dense vector: flat list of numeric values."""
    if not value:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)


def _is_multi_dense_vector(value: list) -> bool:
    """Check if list is a multi-value dense vector: list of dense vectors."""
    if not value:
        return False
    # All items must be lists and each must be a valid dense vector
    if not all(isinstance(item, list) for item in value):
        return False
    return all(_is_dense_vector(item) for item in value)


def _to_field_entries(doc: dict[str, Any]) -> list[pb.FieldEntry]:
    """Convert document dict to list of FieldEntry for multi-value field support.

    Multi-value fields (list of sparse vectors or list of dense vectors) are
    expanded into multiple FieldEntry with the same name.
    """
    entries = []
    for name, value in doc.items():
        if isinstance(value, list):
            # Check for multi-value sparse vectors: [[(idx, val), ...], ...]
            if _is_multi_sparse_vector(value):
                for sv in value:
                    indices = [int(item[0]) for item in sv]
                    values = [float(item[1]) for item in sv]
                    fv = pb.FieldValue(
                        sparse_vector=pb.SparseVector(indices=indices, values=values)
                    )
                    entries.append(pb.FieldEntry(name=name, value=fv))
                continue
            # Check for multi-value dense vectors: [[f1, f2, ...], ...]
            if _is_multi_dense_vector(value):
                for dv in value:
                    fv = pb.FieldValue(
                        dense_vector=pb.DenseVector(values=[float(v) for v in dv])
                    )
                    entries.append(pb.FieldEntry(name=name, value=fv))
                continue
            # Single sparse vector: [(idx, val), ...]
            if _is_sparse_vector(value):
                indices = [int(item[0]) for item in value]
                values = [float(item[1]) for item in value]
                fv = pb.FieldValue(
                    sparse_vector=pb.SparseVector(indices=indices, values=values)
                )
                entries.append(pb.FieldEntry(name=name, value=fv))
                continue
            # Single dense vector: [f1, f2, ...]
            if _is_dense_vector(value):
                fv = pb.FieldValue(
                    dense_vector=pb.DenseVector(values=[float(v) for v in value])
                )
                entries.append(pb.FieldEntry(name=name, value=fv))
                continue
            # Multi-value plain field: ["val1", "val2", ...] -> separate entries
            for item in value:
                entries.append(pb.FieldEntry(name=name, value=_to_field_value(item)))
            continue
        # Single value - use standard conversion
        entries.append(pb.FieldEntry(name=name, value=_to_field_value(value)))
    return entries


def _to_field_value(value: Any) -> pb.FieldValue:
    """Convert Python value to protobuf FieldValue.

    Special handling for vector types:
    - list[(int, float)] -> SparseVector (list of (index, value) tuples)
    - list[float] -> DenseVector (flat list of numeric values)
    - Other lists/dicts -> JSON
    """
    if isinstance(value, str):
        return pb.FieldValue(text=value)
    elif isinstance(value, bool):
        return pb.FieldValue(u64=1 if value else 0)
    elif isinstance(value, int):
        if value >= 0:
            return pb.FieldValue(u64=value)
        else:
            return pb.FieldValue(i64=value)
    elif isinstance(value, float):
        return pb.FieldValue(f64=value)
    elif isinstance(value, bytes):
        return pb.FieldValue(bytes_value=value)
    elif isinstance(value, dict):
        # Dicts are always JSON
        return pb.FieldValue(json_value=json.dumps(value))
    elif isinstance(value, list):
        # Check if it's a sparse vector: list of (index, value) pairs
        if _is_sparse_vector(value):
            indices = [int(item[0]) for item in value]
            values = [float(item[1]) for item in value]
            return pb.FieldValue(
                sparse_vector=pb.SparseVector(indices=indices, values=values)
            )
        # Check if it's a dense vector: flat list of numeric values
        if _is_dense_vector(value):
            return pb.FieldValue(
                dense_vector=pb.DenseVector(values=[float(v) for v in value])
            )
        # Otherwise treat as JSON
        return pb.FieldValue(json_value=json.dumps(value))
    else:
        return pb.FieldValue(text=str(value))


def _from_field_value(fv: pb.FieldValue) -> Any:
    """Convert protobuf FieldValue to Python value."""
    which = fv.WhichOneof("value")
    if which == "text":
        return fv.text
    elif which == "u64":
        return fv.u64
    elif which == "i64":
        return fv.i64
    elif which == "f64":
        return fv.f64
    elif which == "bytes_value":
        return fv.bytes_value
    elif which == "json_value":
        return json.loads(fv.json_value)
    elif which == "sparse_vector":
        return {
            "indices": list(fv.sparse_vector.indices),
            "values": list(fv.sparse_vector.values),
        }
    elif which == "dense_vector":
        return list(fv.dense_vector.values)
    return None


def _from_field_value_list(fvl: pb.FieldValueList) -> Any:
    """Convert protobuf FieldValueList to Python value.

    Single-value fields are unwrapped to a scalar.
    Multi-value fields are returned as a list.
    """
    values = [_from_field_value(v) for v in fvl.values]
    if len(values) == 1:
        return values[0]
    return values


_COMBINER_MAP: dict[str, int] = {
    "log_sum_exp": 0,
    "max": 1,
    "avg": 2,
    "sum": 3,
    "weighted_top_k": 4,
}


def _combiner_to_proto(combiner: str | None) -> int:
    """Convert combiner string to proto MultiValueCombiner enum value."""
    if combiner is None:
        return 0  # LOG_SUM_EXP default
    return _COMBINER_MAP.get(combiner.lower(), 0)


def _build_query(q: dict[str, Any]) -> pb.Query:
    """Recursively convert a Query dict to protobuf Query.

    The dict must have exactly one key matching the proto Query oneof:
    "term", "match", "boolean", "sparse_vector", "dense_vector", "boost", "all".
    """
    if "term" in q:
        t = q["term"]
        return pb.Query(term=pb.TermQuery(field=t["field"], term=t["term"]))

    if "match" in q:
        m = q["match"]
        return pb.Query(match=pb.MatchQuery(field=m["field"], text=m["text"]))

    if "boolean" in q:
        b = q["boolean"]
        return pb.Query(
            boolean=pb.BooleanQuery(
                must=[_build_query(sq) for sq in b.get("must", [])],
                should=[_build_query(sq) for sq in b.get("should", [])],
                must_not=[_build_query(sq) for sq in b.get("must_not", [])],
            )
        )

    if "sparse_vector" in q:
        sv = q["sparse_vector"]
        return pb.Query(
            sparse_vector=pb.SparseVectorQuery(
                field=sv["field"],
                indices=sv.get("indices", []),
                values=sv.get("values", []),
                text=sv.get("text", ""),
                combiner=_combiner_to_proto(sv.get("combiner")),
                heap_factor=sv.get("heap_factor", 0),
                combiner_temperature=sv.get("combiner_temperature", 0),
                combiner_top_k=sv.get("combiner_top_k", 0),
                combiner_decay=sv.get("combiner_decay", 0),
                weight_threshold=sv.get("weight_threshold", 0),
                max_query_dims=sv.get("max_query_dims", 0),
                pruning=sv.get("pruning", 0),
            )
        )

    if "dense_vector" in q:
        dv = q["dense_vector"]
        return pb.Query(
            dense_vector=pb.DenseVectorQuery(
                field=dv["field"],
                vector=dv["vector"],
                nprobe=dv.get("nprobe", 0),
                rerank_factor=dv.get("rerank_factor", 0),
                combiner=_combiner_to_proto(dv.get("combiner")),
                combiner_temperature=dv.get("combiner_temperature", 0),
                combiner_top_k=dv.get("combiner_top_k", 0),
                combiner_decay=dv.get("combiner_decay", 0),
            )
        )

    if "boost" in q:
        bq = q["boost"]
        return pb.Query(
            boost=pb.BoostQuery(
                query=_build_query(bq["query"]),
                boost=bq["boost"],
            )
        )

    if "range" in q:
        rq = q["range"]
        kwargs: dict[str, Any] = {"field": rq["field"]}
        if "min_u64" in rq:
            kwargs["min_u64"] = int(rq["min_u64"])
        if "max_u64" in rq:
            kwargs["max_u64"] = int(rq["max_u64"])
        if "min_i64" in rq:
            kwargs["min_i64"] = int(rq["min_i64"])
        if "max_i64" in rq:
            kwargs["max_i64"] = int(rq["max_i64"])
        if "min_f64" in rq:
            kwargs["min_f64"] = float(rq["min_f64"])
        if "max_f64" in rq:
            kwargs["max_f64"] = float(rq["max_f64"])
        return pb.Query(range=pb.RangeQuery(**kwargs))

    if "all" in q:
        return pb.Query(all=pb.AllQuery())

    # No recognized query key found
    valid_keys = [
        "term",
        "match",
        "boolean",
        "sparse_vector",
        "dense_vector",
        "boost",
        "range",
        "all",
    ]
    raise ValueError(
        f"Unrecognized query key(s): {set(q.keys()) - set(valid_keys)}. "
        f"Valid keys: {valid_keys}"
    )


def _build_reranker(r: dict[str, Any]) -> pb.Reranker:
    """Convert a Reranker dict to protobuf Reranker."""
    return pb.Reranker(
        field=r["field"],
        vector=r["vector"],
        limit=r.get("limit", 0),
        combiner=_combiner_to_proto(r.get("combiner")),
        combiner_temperature=r.get("combiner_temperature", 0),
        combiner_top_k=r.get("combiner_top_k", 0),
        combiner_decay=r.get("combiner_decay", 0),
        matryoshka_dims=r.get("matryoshka_dims", 0),
    )
