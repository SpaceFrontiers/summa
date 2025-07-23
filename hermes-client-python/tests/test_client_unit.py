"""Fast unit coverage for client-side protobuf conversion and delegation."""

from importlib.metadata import version
from unittest.mock import AsyncMock

import pytest
from hermes_client_python import HermesClient, __version__
from hermes_client_python.client import (
    _build_query,
    _from_field_value_list,
    _to_field_entries,
)
from hermes_client_python.hermes_pb2 import FieldValue, FieldValueList


def test_runtime_version_comes_from_distribution_metadata():
    assert __version__ == version("hermes-client-python")


def test_field_entries_preserve_multi_value_and_vector_shapes():
    entries = _to_field_entries(
        {
            "tags": ["rust", "search"],
            "sparse": [[(1, 0.5)], [(2, 0.25)]],
            "dense": [[1, 2.5], [3.0, 4]],
        }
    )

    by_name = {
        name: [entry.value for entry in entries if entry.name == name]
        for name in ("tags", "sparse", "dense")
    }
    assert [value.text for value in by_name["tags"]] == ["rust", "search"]
    assert [
        (list(value.sparse_vector.indices), list(value.sparse_vector.values))
        for value in by_name["sparse"]
    ] == [([1], [0.5]), ([2], [0.25])]
    assert list(by_name["dense"][0].dense_vector.values) == pytest.approx([1.0, 2.5])
    assert list(by_name["dense"][1].dense_vector.values) == pytest.approx([3.0, 4.0])


def test_field_value_lists_keep_scalar_unwrapping_contract():
    assert _from_field_value_list(FieldValueList(values=[])) == []
    assert (
        _from_field_value_list(FieldValueList(values=[FieldValue(text="one")])) == "one"
    )
    assert _from_field_value_list(
        FieldValueList(values=[FieldValue(text="one"), FieldValue(text="two")])
    ) == ["one", "two"]


def test_query_builder_preserves_recursive_fusion_options():
    query = _build_query(
        {
            "fusion": {
                "method": "normalized_weighted_sum",
                "rrf_k": 42,
                "combiner": "max",
                "queries": [
                    {
                        "query": {
                            "boolean": {
                                "must": [
                                    {
                                        "match": {
                                            "field": "title",
                                            "text": "search engine",
                                        }
                                    }
                                ]
                            }
                        },
                        "weight": 0.75,
                    },
                    {"query": {"all": {}}},
                ],
            }
        }
    )

    assert query.WhichOneof("query") == "fusion"
    assert query.fusion.rrf_k == 42
    assert query.fusion.queries[0].weight == pytest.approx(0.75)
    assert query.fusion.queries[0].query.boolean.must[0].WhichOneof("query") == "match"


def test_sparse_query_preserves_optional_lsp_gamma_presence():
    unset = _build_query({"sparse_vector": {"field": "embedding"}})
    assert not unset.sparse_vector.HasField("lsp_gamma")

    exhaustive = _build_query({"sparse_vector": {"field": "embedding", "lsp_gamma": 0}})
    assert exhaustive.sparse_vector.HasField("lsp_gamma")
    assert exhaustive.sparse_vector.lsp_gamma == 0


@pytest.mark.asyncio
async def test_index_document_delegates_timeout_to_batch_path():
    client = HermesClient(default_timeout=5)
    client.index_documents = AsyncMock(return_value=(1, 0, []))

    await client.index_document("articles", {"title": "Hermes"}, timeout=0.25)

    client.index_documents.assert_awaited_once_with(
        "articles", [{"title": "Hermes"}], timeout=0.25
    )
