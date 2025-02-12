"""Async Python client for Hermes search server."""

from .client import HermesClient
from .types import (
    AllQuery,
    BooleanQuery,
    BoostQuery,
    Combiner,
    DenseVectorQuery,
    DocAddress,
    Document,
    Filter,
    IndexInfo,
    MatchQuery,
    Reranker,
    SearchHit,
    SearchResponse,
    SearchTimings,
    SparseVectorQuery,
    TermQuery,
    VectorFieldStats,
)

__all__ = [
    "HermesClient",
    "AllQuery",
    "BooleanQuery",
    "BoostQuery",
    "Combiner",
    "DenseVectorQuery",
    "DocAddress",
    "Document",
    "Filter",
    "IndexInfo",
    "MatchQuery",
    "Reranker",
    "SearchHit",
    "SearchResponse",
    "SearchTimings",
    "SparseVectorQuery",
    "TermQuery",
    "VectorFieldStats",
]

__version__ = "1.0.2"
