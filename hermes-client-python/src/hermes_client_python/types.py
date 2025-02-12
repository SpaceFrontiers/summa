"""Type definitions for Hermes client.

All search-related types mirror the proto API structure exactly.
Query is a dict with exactly one key matching the proto Query oneof variant.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

# =============================================================================
# Multi-value score combiner (mirrors proto MultiValueCombiner)
# =============================================================================

Combiner = Literal["log_sum_exp", "max", "avg", "sum", "weighted_top_k"]

# =============================================================================
# Query types (mirrors proto Query oneof)
# =============================================================================


class TermQuery(TypedDict):
    field: str
    term: str


class MatchQuery(TypedDict):
    field: str
    text: str


class BooleanQuery(TypedDict, total=False):
    must: list["Query"]
    should: list["Query"]
    must_not: list["Query"]


class BoostQuery(TypedDict):
    query: "Query"
    boost: float


class AllQuery(TypedDict):
    pass


class SparseVectorQuery(TypedDict, total=False):
    field: str  # required but total=False for optional fields
    indices: list[int]
    values: list[float]
    text: str
    combiner: Combiner
    heap_factor: float
    combiner_temperature: float
    combiner_top_k: int
    combiner_decay: float
    weight_threshold: float
    max_query_dims: int
    pruning: float


class DenseVectorQuery(TypedDict, total=False):
    field: str  # required but total=False for optional fields
    vector: list[float]
    nprobe: int
    rerank_factor: float
    combiner: Combiner
    combiner_temperature: float
    combiner_top_k: int
    combiner_decay: float


# Query is a dict with exactly one key: "term", "match", "boolean",
# "sparse_vector", "dense_vector", "boost", or "all".
Query = dict[str, Any]

# =============================================================================
# Reranker (mirrors proto Reranker)
# =============================================================================


class Reranker(TypedDict, total=False):
    field: str
    vector: list[float]
    limit: int
    combiner: Combiner
    combiner_temperature: float
    combiner_top_k: int
    combiner_decay: float
    matryoshka_dims: int


# =============================================================================
# Filter (mirrors proto Filter)
# =============================================================================


class Filter(TypedDict, total=False):
    field: str
    eq_u64: int
    eq_i64: int
    eq_f64: float
    eq_text: str
    range: dict[str, float]  # {"min": ..., "max": ...}
    in_values: dict[
        str, list
    ]  # {"text_values": [...], "u64_values": [...], "i64_values": [...]}


# =============================================================================
# Response types
# =============================================================================


@dataclass
class Document:
    """A document with field values."""

    fields: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.fields[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)


@dataclass
class DocAddress:
    """Unique document address: segment + local doc_id."""

    segment_id: str
    doc_id: int


@dataclass
class SearchHit:
    """A single search result."""

    address: DocAddress
    score: float
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchTimings:
    """Detailed timing breakdown for search phases (all values in microseconds)."""

    search_us: int
    rerank_us: int
    load_us: int
    total_us: int


@dataclass
class SearchResponse:
    """Search response with hits and metadata."""

    hits: list[SearchHit]
    total_hits: int
    took_ms: int
    timings: SearchTimings | None = None


@dataclass
class VectorFieldStats:
    """Per-field vector statistics."""

    field_name: str
    vector_type: str  # "dense" or "sparse"
    total_vectors: int
    dimension: int


@dataclass
class IndexInfo:
    """Information about an index."""

    index_name: str
    num_docs: int
    num_segments: int
    schema: str
    vector_stats: list[VectorFieldStats] = field(default_factory=list)
