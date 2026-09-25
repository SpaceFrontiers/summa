"""Async Python client for Summa search server."""

from importlib.metadata import PackageNotFoundError, version

from .client import SummaClient
from .types import (
    AllQuery,
    BinaryDenseVectorQuery,
    BooleanQuery,
    BoostQuery,
    CandidateScores,
    Combiner,
    DenseVectorQuery,
    DocAddress,
    Document,
    DocumentMutationError,
    DocumentMutationResult,
    FusionCandidate,
    FusionCandidateList,
    IndexInfo,
    MatchQuery,
    OrdinalScore,
    PassageScores,
    QueryTrace,
    RangeQuery,
    Reranker,
    RrfContribution,
    SearchHit,
    SearchResponse,
    SearchTimings,
    SearchTrace,
    ShardSearchTrace,
    SparseVectorQuery,
    TermQuery,
    VectorFieldStats,
)

__all__ = [
    "SummaClient",
    "AllQuery",
    "BinaryDenseVectorQuery",
    "BooleanQuery",
    "BoostQuery",
    "Combiner",
    "CandidateScores",
    "FusionCandidate",
    "FusionCandidateList",
    "PassageScores",
    "RrfContribution",
    "QueryTrace",
    "SearchTrace",
    "ShardSearchTrace",
    "DenseVectorQuery",
    "DocAddress",
    "Document",
    "DocumentMutationResult",
    "DocumentMutationError",
    "IndexInfo",
    "MatchQuery",
    "OrdinalScore",
    "RangeQuery",
    "Reranker",
    "SearchHit",
    "SearchResponse",
    "SearchTimings",
    "SparseVectorQuery",
    "TermQuery",
    "VectorFieldStats",
]

try:
    __version__ = version("summa-client-python")
except PackageNotFoundError:
    # Source-only imports (without an installed wheel/editable distribution).
    __version__ = "0.0.0+unknown"
