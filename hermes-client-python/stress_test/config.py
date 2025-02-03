"""Stress test configuration."""

from dataclasses import dataclass, field
from enum import Enum


class IndexType(Enum):
    """Type of index to test."""

    SPARSE = "sparse"
    DENSE = "dense"
    FULLTEXT = "fulltext"
    MIXED = "mixed"


# Schema templates for different index types
SCHEMAS = {
    IndexType.SPARSE: """
index stress_test {
    field id: text [stored]
    field content: text [stored]
    field embedding: sparse_vector [indexed]
}
""".strip(),
    IndexType.DENSE: """
index stress_test {
    field id: text [stored]
    field content: text [stored]
    field embedding: dense_vector<128> [indexed, stored]
}
""".strip(),
    IndexType.FULLTEXT: """
index stress_test {
    field id: text [stored]
    field title: text<en_stem> [indexed, stored]
    field body: text<en_stem> [indexed, stored]
}
""".strip(),
    IndexType.MIXED: """
index stress_test {
    field id: text [stored]
    field title: text<en_stem> [indexed, stored]
    field body: text<en_stem> [indexed, stored]
    field sparse_emb: sparse_vector [indexed]
    field dense_emb: dense_vector<128> [indexed, stored]
}
""".strip(),
}


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""

    # Server connection
    server_address: str = "localhost:50051"
    index_name: str = "stress_test"

    # Index type
    index_type: IndexType = IndexType.SPARSE

    # Document generation
    doc_count: int = 50_000
    batch_size: int = 500

    # Worker configuration
    index_workers: int = 4
    search_workers: int = 4

    # Sparse vector parameters (SPLADE-like)
    vocab_size: int = 30_000  # Vocabulary size for sparse vectors
    avg_nnz: int = 100  # Average non-zero entries per vector
    nnz_std: int = 30  # Standard deviation of non-zero entries
    sparse_vectors_per_doc: int = 3  # Number of sparse vectors per document

    # Dense vector parameters
    dense_dim: int = 128  # Dimension of dense vectors
    dense_vectors_per_doc: int = 1  # Number of dense vectors per document

    # Fulltext parameters
    avg_title_words: int = 10
    avg_body_words: int = 200

    # Search parameters
    search_qps: float = 50.0  # Target queries per second
    search_top_k: int = 10  # Top-k results per query

    # Test timing
    warmup_docs: int = 10_000  # Index this many docs before starting searches
    duration_seconds: int = 120  # Total test duration

    # Monitoring
    segment_poll_interval_ms: int = 100  # How often to poll segment count
    memory_poll_interval_ms: int = 500  # How often to poll memory

    # Memory limit (for verification)
    memory_limit_mb: int = 2048  # Expected server memory limit

    # Schema definition (auto-generated from index_type if not set)
    schema: str = field(default="")

    def __post_init__(self):
        """Set schema based on index_type if not provided."""
        if not self.schema:
            self.schema = SCHEMAS.get(self.index_type, SCHEMAS[IndexType.SPARSE])

    def docs_before_search(self) -> int:
        """Number of docs to index before starting searches."""
        return min(self.warmup_docs, self.doc_count // 4)

    def batches_total(self) -> int:
        """Total number of batches to index."""
        return (self.doc_count + self.batch_size - 1) // self.batch_size
