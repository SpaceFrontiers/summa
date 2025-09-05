"""Type definitions for Hermes client.

All search-related types mirror the proto API structure exactly.
Query is a dict with exactly one key matching the proto Query oneof variant.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Required, TypedDict

# =============================================================================
# Multi-value score combiner (mirrors proto MultiValueCombiner)
# =============================================================================

Combiner = Literal["log_sum_exp", "max", "avg", "sum", "weighted_top_k"]

# =============================================================================
# Query types (mirrors proto Query oneof)
# =============================================================================


class TermQuery(TypedDict, total=False):
    field: Required[str]
    term: Required[str]
    tokenizer_hint: str


class MatchQuery(TypedDict, total=False):
    field: Required[str]
    text: Required[str]
    # Passed to the field's tokenizer; a dynamic stemmer reads it as a
    # comma-separated language list ("ru,en"). Static tokenizers ignore it.
    tokenizer_hint: str


class PhraseQuery(TypedDict, total=False):
    field: Required[str]
    text: Required[str]
    slop: int
    tokenizer_hint: str


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
    lsp_gamma: int


class DenseVectorQuery(TypedDict, total=False):
    field: str  # required but total=False for optional fields
    vector: list[float]
    nprobe: int
    combiner: Combiner
    combiner_temperature: float
    combiner_top_k: int
    combiner_decay: float


class BinaryDenseVectorQuery(TypedDict, total=False):
    field: str  # required but total=False for optional fields
    vector: bytes  # packed-bit query vector (ceil(dim/8) bytes)
    combiner: Combiner
    combiner_temperature: float
    combiner_top_k: int
    combiner_decay: float


class RangeQuery(TypedDict, total=False):
    field: str  # required but total=False for optional fields
    min_u64: int
    max_u64: int
    min_i64: int
    max_i64: int
    min_f64: float
    max_f64: float


# Query is a dict with exactly one key matching a protobuf Query variant,
# including text, Boolean, vector, range/prefix, all, and fusion queries.
Query = dict[str, Any]

# =============================================================================
# Reranker (mirrors proto Reranker)
# =============================================================================


class Reranker(TypedDict, total=False):
    field: str
    vector: list[float]
    combiner: Combiner
    combiner_temperature: float
    combiner_top_k: int
    combiner_decay: float
    matryoshka_dims: int
    binary_vector: bytes  # packed-bit query vector (for binary dense fields)
    rrf_k: float  # Reciprocal Rank Fusion k (0 = disabled, typical: 60)


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
class OrdinalScore:
    """Score contribution from a specific ordinal in a multi-valued field."""

    ordinal: int
    score: float


@dataclass
class PassageScores:
    ordinal: int
    scores: dict[str, float]
    l1_score: float | None = None


@dataclass
class CandidateScores:
    document: dict[str, float]
    passages: list[PassageScores]
    scored_passages: int


@dataclass
class RrfContribution:
    query_index: int
    query_name: str
    rank: int
    score: float
    ordinal: int | None = None


@dataclass
class SearchHit:
    """A single search result."""

    address: DocAddress
    score: float
    fields: dict[str, Any] = field(default_factory=dict)
    ordinal_scores: list[OrdinalScore] = field(default_factory=list)
    candidate_scores: CandidateScores | None = None
    rrf_score: float | None = None
    rrf_contributions: list[RrfContribution] = field(default_factory=list)


@dataclass
class SearchTimings:
    """Detailed timing breakdown for search phases (all values in microseconds)."""

    search_us: int
    rerank_us: int
    load_us: int
    total_us: int
    candidate_scoring_us: int = 0


@dataclass
class FusionCandidate:
    address: DocAddress
    score: float
    ordinal_scores: list[OrdinalScore] = field(default_factory=list)


@dataclass
class FusionCandidateList:
    query_index: int
    candidates: list[FusionCandidate] = field(default_factory=list)


@dataclass
class QueryTrace:
    query_index: int
    query_name: str
    query: dict[str, Any]
    scope: int
    score_only: bool
    candidate_depth: int
    total_seen: int
    candidates: list[FusionCandidate] = field(default_factory=list)


@dataclass
class ShardSearchTrace:
    shard_id: str
    backend_id: str
    index_name: str
    queries: list[QueryTrace]
    selected: list[FusionCandidate]
    ranking_method: str
    truncated: bool
    filters: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SearchTrace:
    shards: list[ShardSearchTrace] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Search response with hits and metadata."""

    hits: list[SearchHit]
    total_hits: int
    took_ms: int
    timings: SearchTimings | None = None
    ranking_method: str = ""
    truncated: bool = False
    fusion_candidates: list[FusionCandidateList] = field(default_factory=list)
    trace: SearchTrace | None = None


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
    physical_num_docs: int = 0
    num_deleted_docs: int = 0
    deleted_ratio: float = 0.0
    candidate_scoring_version: int = 0
    unprepared_candidate_fields: list[str] = field(default_factory=list)


class DocumentMutationError(TypedDict):
    """Rejected operation at a zero-based position in the input batch."""

    index: int
    error: str


@dataclass
class DocumentMutationResult:
    """Staged operations (not affected rows); commit publishes accepted work."""

    accepted_count: int
    errors: list[DocumentMutationError] = field(default_factory=list)
