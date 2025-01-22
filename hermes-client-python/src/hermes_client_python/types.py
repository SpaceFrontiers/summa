"""Type definitions for Hermes client."""

from dataclasses import dataclass, field
from typing import Any


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
class SearchHit:
    """A single search result."""

    doc_id: int
    score: float
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Search response with hits and metadata."""

    hits: list[SearchHit]
    total_hits: int
    took_ms: int


@dataclass
class IndexInfo:
    """Information about an index."""

    index_name: str
    num_docs: int
    num_segments: int
    schema: str
