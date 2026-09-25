"""Seismic search and stored values survive copy merge and bounded maintenance.

Sparse maintenance follows nomination debt; no field-level reorder flag is
required. Small term lists retain every posting, so this fixture can compare
against exact expected scores while exercising the default nomination path.
"""

import contextlib
import random

import pytest
import pytest_asyncio

INDEX_NAME = "test_seismic_maintenance"
SCHEMA = """
index test_seismic_maintenance {
    field title: text<raw> [primary, stored]
    field doc_id: u64 [indexed, stored]
    field embedding: sparse_vector<u32> [indexed<format: seismic, quantization: float32, dims: 30000>, stored]
}
"""
NUM_DOCS = 500


def _generate_documents() -> list[dict]:
    rng = random.Random(42)
    documents = []
    for doc_id in range(NUM_DOCS):
        # Reserved needle dimensions never appear in topic/noise coordinates.
        entries = {doc_id: 1.0 + doc_id / 1024}
        topic = doc_id % 10
        for offset in range(8):
            entries[20000 + topic * 100 + offset] = 1.0 + doc_id / 1024
        for dim in rng.sample(range(NUM_DOCS, 20000), 32):
            entries[dim] = rng.randint(1, 32) / 64
        documents.append(
            {
                "title": f"document_{doc_id}",
                "doc_id": doc_id,
                "embedding": sorted(entries.items()),
            }
        )
    return documents


DOCUMENTS = _generate_documents()


@pytest_asyncio.fixture(autouse=True)
async def setup_index(client):
    with contextlib.suppress(Exception):
        await client.delete_index(INDEX_NAME)
    await client.create_index(INDEX_NAME, SCHEMA)
    yield
    await client.delete_index(INDEX_NAME)


async def _commit_documents(client, documents):
    indexed, errors, details = await client.index_documents(INDEX_NAME, documents)
    assert indexed == len(documents), details
    assert errors == 0, details
    await client.commit(INDEX_NAME)


async def _search_dimension(client, dimension, limit=100):
    result = await client.search(
        INDEX_NAME,
        query={
            "sparse_vector": {
                "field": "embedding",
                "indices": [dimension],
                "values": [1.0],
            }
        },
        limit=limit,
        fields_to_load=["title", "doc_id"],
    )
    return result.hits


async def _assert_search_and_stored_values(client, deleted=frozenset()):
    for doc_id in [0, 1, 42, 99, 249, 300, 498, 499]:
        hits = await _search_dimension(client, doc_id)
        if doc_id in deleted:
            assert not hits
            continue
        assert len(hits) == 1
        assert hits[0].fields["doc_id"] == doc_id
        assert hits[0].fields["title"] == f"document_{doc_id}"
        assert hits[0].score == pytest.approx(1.0 + doc_id / 1024)

    for topic in range(10):
        hits = await _search_dimension(client, 20000 + topic * 100)
        expected = [
            i for i in reversed(range(NUM_DOCS)) if i % 10 == topic and i not in deleted
        ]
        assert [hit.fields["doc_id"] for hit in hits] == expected
        assert [hit.score for hit in hits] == pytest.approx(
            [1.0 + i / 1024 for i in expected]
        )

    info = await client.get_index_info(INDEX_NAME)
    assert info.num_docs == NUM_DOCS - len(deleted)


@pytest.mark.asyncio
async def test_copy_merge_and_maintenance_preserve_scores_and_stored_fields(client):
    half = NUM_DOCS // 2
    await _commit_documents(client, DOCUMENTS[:half])
    await _commit_documents(client, DOCUMENTS[half:])
    await _assert_search_and_stored_values(client)

    assert await client.force_merge(INDEX_NAME) == 1
    await _assert_search_and_stored_values(client)

    # The existing reorder RPC runs a bounded maintenance pass. It need not
    # discharge all nomination debt in one call.
    assert await client.reorder(INDEX_NAME) == 1
    await _assert_search_and_stored_values(client)


@pytest.mark.asyncio
async def test_repeated_maintenance_preserves_converged_segment_results(client):
    await _commit_documents(client, DOCUMENTS)
    await _assert_search_and_stored_values(client)
    segments = (await client.get_index_info(INDEX_NAME)).num_segments
    for _ in range(2):
        assert await client.reorder(INDEX_NAME) == segments
        await _assert_search_and_stored_values(client)


@pytest.mark.asyncio
async def test_deleted_documents_stay_hidden_through_merge_maintenance_and_compaction(
    client,
):
    half = NUM_DOCS // 2
    await _commit_documents(client, DOCUMENTS[:half])
    await _commit_documents(client, DOCUMENTS[half:])
    deleted = {42, 249, 499}
    result = await client.delete_documents(
        INDEX_NAME, [f"document_{i}" for i in sorted(deleted)]
    )
    assert result.accepted_count == len(deleted)
    assert not result.errors
    await client.commit(INDEX_NAME)
    await _assert_search_and_stored_values(client, deleted)

    assert await client.force_merge(INDEX_NAME) == 1
    await _assert_search_and_stored_values(client, deleted)
    assert await client.reorder(INDEX_NAME) == 1
    await _assert_search_and_stored_values(client, deleted)
    assert await client.force_merge(INDEX_NAME, compact=True) == 1
    await _assert_search_and_stored_values(client, deleted)
