"""Integration test: multi-value URI text field.

Creates an index with a multi-value text field for URIs, indexes documents
with multiple URIs each, and verifies search + field retrieval work correctly.
"""

import contextlib

import pytest
import pytest_asyncio

INDEX_NAME = "test_multivalue_uri"

SCHEMA = """
index test_multivalue_uri {
    field title: text<simple> [indexed, stored]
    field uris: text<raw> [indexed, stored<multi>]
}
"""

DOCUMENTS = [
    {
        "title": "Rust language",
        "uris": [
            "https://github.com/rust-lang/rust",
            "https://doc.rust-lang.org/book/",
            "https://crates.io/",
        ],
    },
    {
        "title": "Python language",
        "uris": [
            "https://github.com/python/cpython",
            "https://docs.python.org/3/",
            "https://pypi.org/",
        ],
    },
    {
        "title": "Hermes search",
        "uris": [
            "https://github.com/SpaceFrontiers/hermes",
            "https://crates.io/crates/hermes-core",
        ],
    },
    {
        "title": "Tokio runtime",
        "uris": [
            "https://github.com/tokio-rs/tokio",
            "https://docs.rs/tokio/latest/tokio/",
            "https://crates.io/crates/tokio",
        ],
    },
    {
        "title": "Serde serialization",
        "uris": [
            "https://github.com/serde-rs/serde",
            "https://serde.rs/",
            "https://crates.io/crates/serde",
            "https://docs.rs/serde/latest/serde/",
        ],
    },
]


@pytest_asyncio.fixture(autouse=True)
async def setup_index(client):
    """Create, populate, and clean up the test index."""
    # Clean up from previous run if needed
    with contextlib.suppress(Exception):
        await client.delete_index(INDEX_NAME)

    await client.create_index(INDEX_NAME, SCHEMA)
    await client.index_documents(INDEX_NAME, DOCUMENTS)
    await client.commit(INDEX_NAME)
    yield
    await client.delete_index(INDEX_NAME)


@pytest.mark.asyncio
async def test_search_by_exact_uri(client):
    """Searching for an exact URI should return the document containing it."""
    results = await client.search(
        INDEX_NAME,
        query={"term": {"field": "uris", "term": "https://github.com/rust-lang/rust"}},
        fields_to_load=["title", "uris"],
    )
    assert results.total_hits >= 1
    hit = results.hits[0]
    assert hit.fields["title"] == "Rust language"


@pytest.mark.asyncio
async def test_search_returns_all_uris(client):
    """Loading the uris field should return all stored values as a list."""
    results = await client.search(
        INDEX_NAME,
        query={"term": {"field": "uris", "term": "https://serde.rs/"}},
        fields_to_load=["uris"],
    )
    assert results.total_hits >= 1
    uris = results.hits[0].fields["uris"]
    # Multi-value field should come back as a list
    assert isinstance(uris, list), f"Expected list, got {type(uris)}: {uris}"
    assert len(uris) == 4
    assert "https://serde.rs/" in uris
    assert "https://github.com/serde-rs/serde" in uris


@pytest.mark.asyncio
async def test_search_no_false_positives(client):
    """Searching for a URI not in any document should return no results."""
    results = await client.search(
        INDEX_NAME,
        query={"term": {"field": "uris", "term": "https://example.com/nonexistent"}},
    )
    assert results.total_hits == 0
    assert len(results.hits) == 0


@pytest.mark.asyncio
async def test_search_shared_domain(client):
    """Multiple docs share crates.io URIs — searching one should hit only exact match."""
    results = await client.search(
        INDEX_NAME,
        query={
            "term": {"field": "uris", "term": "https://crates.io/crates/hermes-core"}
        },
        fields_to_load=["title"],
    )
    assert results.total_hits == 1
    assert results.hits[0].fields["title"] == "Hermes search"


@pytest.mark.asyncio
async def test_get_document_multi_value(client):
    """get_document should also return all multi-value field entries."""
    # First find a document via search
    results = await client.search(
        INDEX_NAME,
        query={"term": {"field": "uris", "term": "https://github.com/python/cpython"}},
    )
    assert results.total_hits >= 1
    address = results.hits[0].address

    # Now fetch the full document
    doc = await client.get_document(INDEX_NAME, address)
    assert doc is not None
    uris = doc.fields["uris"]
    assert isinstance(uris, list)
    assert len(uris) == 3
    assert "https://github.com/python/cpython" in uris
    assert "https://docs.python.org/3/" in uris
    assert "https://pypi.org/" in uris


@pytest.mark.asyncio
async def test_single_value_field_is_scalar(client):
    """Single-value fields should be returned as a scalar, not a list."""
    results = await client.search(
        INDEX_NAME,
        query={"term": {"field": "uris", "term": "https://serde.rs/"}},
        fields_to_load=["title"],
    )
    assert results.total_hits >= 1
    title = results.hits[0].fields["title"]
    assert isinstance(title, str)
    assert title == "Serde serialization"
