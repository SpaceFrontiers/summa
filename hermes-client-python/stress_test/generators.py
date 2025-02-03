"""Document and vector generation for stress testing."""

import math
import random
from typing import Any

from .config import IndexType, StressTestConfig

# Sample words for fulltext generation
SAMPLE_WORDS = [
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "search",
    "index",
    "query",
    "document",
    "vector",
    "embedding",
    "sparse",
    "dense",
    "text",
    "field",
    "database",
    "server",
    "client",
    "request",
    "response",
    "batch",
    "commit",
    "segment",
    "memory",
    "performance",
    "latency",
    "throughput",
    "stress",
]


def generate_sparse_vector(
    vocab_size: int,
    avg_nnz: int,
    nnz_std: int = 30,
) -> list[tuple[int, float]]:
    """Generate a realistic SPLADE-like sparse vector.

    Uses log-normal distribution for weights to simulate real embedding patterns
    where a few dimensions have high weights and many have small weights.

    Args:
        vocab_size: Total vocabulary size (max dimension index)
        avg_nnz: Average number of non-zero entries
        nnz_std: Standard deviation of non-zero count

    Returns:
        List of (index, value) tuples representing the sparse vector
    """
    # Sample number of non-zero entries from normal distribution
    nnz = max(10, int(random.gauss(avg_nnz, nnz_std)))
    nnz = min(nnz, vocab_size // 10)  # Cap at 10% of vocab

    # Sample unique indices
    indices = random.sample(range(vocab_size), nnz)

    # Generate log-normal weights (realistic for embeddings)
    # Log-normal gives heavy tail: few high values, many small values
    weights = [random.lognormvariate(0, 0.8) for _ in range(nnz)]

    # Normalize to have max weight around 1.0
    max_weight = max(weights) if weights else 1.0
    weights = [w / max_weight for w in weights]

    return list(zip(indices, weights, strict=True))


def generate_dense_vector(dim: int) -> list[float]:
    """Generate a normalized dense vector.

    Args:
        dim: Vector dimension

    Returns:
        List of float values (normalized to unit length)
    """
    # Generate random values
    values = [random.gauss(0, 1) for _ in range(dim)]

    # Normalize to unit length
    magnitude = math.sqrt(sum(v * v for v in values))
    if magnitude > 0:
        values = [v / magnitude for v in values]

    return values


def generate_text(avg_words: int, std_words: int = 0) -> str:
    """Generate random text.

    Args:
        avg_words: Average number of words
        std_words: Standard deviation (default: avg_words/4)

    Returns:
        Generated text string
    """
    if std_words == 0:
        std_words = max(1, avg_words // 4)

    num_words = max(3, int(random.gauss(avg_words, std_words)))
    words = [random.choice(SAMPLE_WORDS) for _ in range(num_words)]

    # Capitalize first word and add period
    if words:
        words[0] = words[0].capitalize()

    return " ".join(words) + "."


def generate_document(
    doc_id: int,
    config: StressTestConfig,
) -> dict[str, Any]:
    """Generate a document based on index type.

    Args:
        doc_id: Unique document identifier
        config: Test configuration

    Returns:
        Document dictionary ready for indexing
    """
    doc: dict[str, Any] = {"id": f"doc_{doc_id}"}

    if config.index_type == IndexType.SPARSE:
        # Generate multiple sparse vectors per document
        embeddings = [
            generate_sparse_vector(
                vocab_size=config.vocab_size,
                avg_nnz=config.avg_nnz,
                nnz_std=config.nnz_std,
            )
            for _ in range(config.sparse_vectors_per_doc)
        ]
        doc["content"] = f"Document {doc_id}"
        doc["embedding"] = embeddings

    elif config.index_type == IndexType.DENSE:
        # Generate dense vectors
        embeddings = [
            generate_dense_vector(config.dense_dim)
            for _ in range(config.dense_vectors_per_doc)
        ]
        doc["content"] = f"Document {doc_id}"
        # Single vector or multi-value
        doc["embedding"] = embeddings if len(embeddings) > 1 else embeddings[0]

    elif config.index_type == IndexType.FULLTEXT:
        # Generate text content
        doc["title"] = generate_text(config.avg_title_words)
        doc["body"] = generate_text(config.avg_body_words)

    elif config.index_type == IndexType.MIXED:
        # Generate all types
        doc["title"] = generate_text(config.avg_title_words)
        doc["body"] = generate_text(config.avg_body_words)
        doc["sparse_emb"] = [
            generate_sparse_vector(
                vocab_size=config.vocab_size,
                avg_nnz=config.avg_nnz,
                nnz_std=config.nnz_std,
            )
        ]
        doc["dense_emb"] = generate_dense_vector(config.dense_dim)

    return doc


def generate_query_vector(
    config: StressTestConfig,
) -> tuple[list[int], list[float]] | list[float] | str:
    """Generate a query for search testing based on index type.

    Args:
        config: Test configuration

    Returns:
        Query data appropriate for the index type:
        - SPARSE: Tuple of (indices, values)
        - DENSE: List of floats
        - FULLTEXT: Search query string
        - MIXED: Tuple of (indices, values) for sparse search
    """
    if config.index_type == IndexType.DENSE:
        return generate_dense_vector(config.dense_dim)

    elif config.index_type == IndexType.FULLTEXT:
        # Generate a short query from random words
        num_terms = random.randint(1, 4)
        terms = [random.choice(SAMPLE_WORDS) for _ in range(num_terms)]
        return " ".join(terms)

    else:  # SPARSE or MIXED
        # Query vectors are usually sparser (fewer terms)
        query_nnz = max(5, config.avg_nnz // 3)

        sparse = generate_sparse_vector(
            vocab_size=config.vocab_size,
            avg_nnz=query_nnz,
            nnz_std=query_nnz // 3,
        )

        indices = [idx for idx, _ in sparse]
        values = [val for _, val in sparse]

        return indices, values


def generate_batch(
    start_id: int,
    batch_size: int,
    config: StressTestConfig,
) -> list[dict[str, Any]]:
    """Generate a batch of documents.

    Args:
        start_id: Starting document ID
        batch_size: Number of documents to generate
        config: Test configuration

    Returns:
        List of document dictionaries
    """
    return [generate_document(start_id + i, config) for i in range(batch_size)]
