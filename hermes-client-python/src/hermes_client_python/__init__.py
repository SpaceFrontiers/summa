"""Async Python client for Hermes search server."""

from .client import HermesClient
from .types import (
    DocAddress,
    Document,
    IndexInfo,
    SearchHit,
    SearchResponse,
    VectorFieldStats,
)

__all__ = [
    "HermesClient",
    "DocAddress",
    "Document",
    "SearchHit",
    "SearchResponse",
    "IndexInfo",
    "VectorFieldStats",
]

__version__ = "1.0.2"
