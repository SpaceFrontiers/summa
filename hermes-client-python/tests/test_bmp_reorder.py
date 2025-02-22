"""Integration test: BMP reorder pipeline.

Tests the full lifecycle: build → search → merge → search → reorder → search.
Verifies that reorder preserves correctness while the index remains queryable
at every stage.
"""

import contextlib
import random

import pytest
import pytest_asyncio

INDEX_NAME = "test_bmp_reorder"

# BMP sparse vector schema with two fields to test multi-field reorder
SCHEMA = """
index test_bmp_reorder {
    field title: text<simple> [indexed, stored]
    field doc_id: u64 [indexed, stored]
    field embedding: sparse_vector<bmp> [indexed, stored]
}
"""

NUM_DOCS = 500
VOCAB_SIZE = 30000


def _make_sparse_vector(doc_idx: int, rng: random.Random) -> list[tuple[int, float]]:
    """Generate a SPLADE-like sparse vector with a unique dim for each doc."""
    num_dims = rng.randint(30, 80)
    entries = {}
    # Unique dim for this doc (for needle-in-haystack verification)
    entries[doc_idx % VOCAB_SIZE] = rng.uniform(0.5, 2.0)
    # Shared dims (topic-like clustering)
    topic = doc_idx % 10
    for _ in range(5):
        dim = 20000 + topic * 100 + rng.randint(0, 99)
        entries[dim] = rng.uniform(0.3, 1.5)
    # Random dims
    for _ in range(num_dims):
        dim = rng.randint(0, VOCAB_SIZE - 1)
        weight = rng.uniform(0.01, 1.0)
        if dim not in entries:
            entries[dim] = weight
    return [(dim, weight) for dim, weight in sorted(entries.items())]


def _generate_documents() -> list[dict]:
    rng = random.Random(42)
    docs = []
    for i in range(NUM_DOCS):
        sv = _make_sparse_vector(i, rng)
        docs.append(
            {
                "title": f"document_{i}",
                "doc_id": i,
                "embedding": sv,
            }
        )
    return docs


DOCUMENTS = _generate_documents()


@pytest_asyncio.fixture(autouse=True)
async def setup_index(client):
    """Create, populate, and clean up the test index."""
    with contextlib.suppress(Exception):
        await client.delete_index(INDEX_NAME)

    await client.create_index(INDEX_NAME, SCHEMA)
    yield
    await client.delete_index(INDEX_NAME)


async def _search_unique_dim(client, doc_idx: int) -> list:
    """Search for a doc by its unique dimension."""
    dim = doc_idx % VOCAB_SIZE
    results = await client.search(
        INDEX_NAME,
        query={
            "sparse_vector": {
                "field": "embedding",
                "indices": [dim],
                "values": [1.0],
            }
        },
        limit=10,
        fields_to_load=["title", "doc_id"],
    )
    return results.hits


async def _search_topic(client, topic: int, limit: int = 50) -> list:
    """Search for docs by topic dims."""
    indices = [20000 + topic * 100 + i for i in range(10)]
    values = [1.0] * len(indices)
    results = await client.search(
        INDEX_NAME,
        query={
            "sparse_vector": {
                "field": "embedding",
                "indices": indices,
                "values": values,
            }
        },
        limit=limit,
        fields_to_load=["doc_id"],
    )
    return results.hits


# =========================================================================
# Phase 1: Build + Search (single segment)
# =========================================================================


@pytest.mark.asyncio
async def test_build_search_merge_reorder(client):
    """Full pipeline: build → search → merge → search → reorder → search."""

    # -- Phase 1: Index first half as segment 1 --
    half = NUM_DOCS // 2
    indexed, errors, errs = await client.index_documents(INDEX_NAME, DOCUMENTS[:half])
    assert indexed == half, f"Expected {half} indexed, got {indexed}, errors: {errs}"
    num_docs = await client.commit(INDEX_NAME)
    assert num_docs == half

    # -- Phase 1b: Index second half as segment 2 --
    indexed2, errors2, errs2 = await client.index_documents(
        INDEX_NAME, DOCUMENTS[half:]
    )
    assert (
        indexed2 == NUM_DOCS - half
    ), f"Expected {NUM_DOCS - half}, got {indexed2}, errors: {errs2}"
    num_docs = await client.commit(INDEX_NAME)
    assert num_docs == NUM_DOCS

    # -- Phase 1c: Search before merge (2 segments) --
    # Needle-in-haystack: find specific docs by unique dim
    for probe_idx in [0, 42, 100, 249, 499]:
        hits = await _search_unique_dim(client, probe_idx)
        doc_ids = [h.fields["doc_id"] for h in hits]
        assert (
            probe_idx in doc_ids
        ), f"Doc {probe_idx} not found before merge. Got doc_ids={doc_ids}"

    # Topic search: should find docs from the same topic
    hits = await _search_topic(client, topic=3)
    topic3_ids = {h.fields["doc_id"] for h in hits}
    expected_topic3 = {i for i in range(NUM_DOCS) if i % 10 == 3}
    overlap = topic3_ids & expected_topic3
    assert len(overlap) > 0, "No topic-3 docs found before merge"

    # -- Phase 2: Force merge → single segment --
    info_before = await client.get_index_info(INDEX_NAME)
    num_segments_before = info_before.num_segments
    assert num_segments_before >= 1

    num_segments = await client.force_merge(INDEX_NAME)
    assert (
        num_segments <= num_segments_before
    ), f"Merge should reduce segments: before={num_segments_before}, after={num_segments}"

    # -- Phase 2b: Search after merge --
    for probe_idx in [0, 42, 100, 249, 499]:
        hits = await _search_unique_dim(client, probe_idx)
        doc_ids = [h.fields["doc_id"] for h in hits]
        assert (
            probe_idx in doc_ids
        ), f"Doc {probe_idx} not found after merge. Got doc_ids={doc_ids}"

    hits = await _search_topic(client, topic=7)
    topic7_ids = {h.fields["doc_id"] for h in hits}
    expected_topic7 = {i for i in range(NUM_DOCS) if i % 10 == 7}
    overlap = topic7_ids & expected_topic7
    assert len(overlap) > 0, "No topic-7 docs found after merge"

    # -- Phase 3: Reorder --
    num_segments = await client.reorder(INDEX_NAME)
    assert num_segments >= 1

    # -- Phase 3b: Search after reorder -- all docs must still be findable --
    for probe_idx in [0, 1, 42, 99, 100, 200, 249, 300, 498, 499]:
        hits = await _search_unique_dim(client, probe_idx)
        doc_ids = [h.fields["doc_id"] for h in hits]
        assert (
            probe_idx in doc_ids
        ), f"Doc {probe_idx} not found after reorder. Got doc_ids={doc_ids}"

    # Topic search after reorder
    for topic in range(10):
        hits = await _search_topic(client, topic=topic, limit=100)
        topic_ids = {h.fields["doc_id"] for h in hits}
        expected = {i for i in range(NUM_DOCS) if i % 10 == topic}
        overlap = topic_ids & expected
        assert len(overlap) > 0, f"No topic-{topic} docs found after reorder"

    # Verify total doc count unchanged
    info_after = await client.get_index_info(INDEX_NAME)
    assert (
        info_after.num_docs == NUM_DOCS
    ), f"Doc count changed after reorder: {info_after.num_docs} != {NUM_DOCS}"


@pytest.mark.asyncio
async def test_reorder_idempotent(client):
    """Reorder twice should not corrupt the index."""
    indexed, _, _ = await client.index_documents(INDEX_NAME, DOCUMENTS)
    assert indexed == NUM_DOCS
    await client.commit(INDEX_NAME)

    # First reorder
    await client.reorder(INDEX_NAME)

    # Second reorder
    await client.reorder(INDEX_NAME)

    # All docs must still be findable
    for probe_idx in [0, 42, 249, 499]:
        hits = await _search_unique_dim(client, probe_idx)
        doc_ids = [h.fields["doc_id"] for h in hits]
        assert (
            probe_idx in doc_ids
        ), f"Doc {probe_idx} not found after double reorder. Got doc_ids={doc_ids}"

    info = await client.get_index_info(INDEX_NAME)
    assert info.num_docs == NUM_DOCS


@pytest.mark.asyncio
async def test_reorder_preserves_stored_fields(client):
    """Reorder must not corrupt stored field values."""
    indexed, _, _ = await client.index_documents(INDEX_NAME, DOCUMENTS)
    assert indexed == NUM_DOCS
    await client.commit(INDEX_NAME)
    await client.reorder(INDEX_NAME)

    # Search for a specific doc and verify stored fields
    hits = await _search_unique_dim(client, 42)
    found = [h for h in hits if h.fields.get("doc_id") == 42]
    assert len(found) == 1, f"Expected exactly 1 hit for doc_id=42, got {len(found)}"
    assert found[0].fields["title"] == "document_42"
