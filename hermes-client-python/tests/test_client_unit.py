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


@pytest.mark.asyncio
async def test_named_l1_roundtrip_preserves_zero_negative_missing_and_scope():
    from hermes_client_python import hermes_pb2 as pb

    client = HermesClient("localhost:50051")
    client._ensure_connected = lambda: None
    client._search_stub = AsyncMock()
    client._search_stub.Search.return_value = pb.SearchResponse(
        ranking_method="linear_v1",
        hits=[
            pb.SearchHit(
                score=-0.25,
                candidate_scores=pb.CandidateScores(
                    document={"title": 0.0},
                    passages=[
                        pb.PassageScores(
                            ordinal=7, scores={"dense": -0.5}, l1_score=-0.25
                        )
                    ],
                    scored_passages=1,
                ),
            )
        ],
    )
    result = await client.search(
        "docs",
        query={
            "fusion": {
                "queries": [
                    {
                        "name": "dense",
                        "scope": "chunk",
                        "query": {"dense_vector": {"field": "dense", "vector": [1.0]}},
                    },
                    {
                        "name": "title",
                        "scope": "document",
                        "score_only": True,
                        "query": {"match": {"field": "title", "text": "hemoglobin"}},
                    },
                ],
                "candidate_depth": 2,
                "filters": [{"phrase": {"field": "body", "text": "red blood cells"}}],
            }
        },
        l1={"weights": {"dense": 0.5}},
        score_export={},
    )
    sent = client._search_stub.Search.call_args.args[0]
    assert sent.query.fusion.queries[0].weight == 0.0
    assert sent.query.fusion.queries[0].scope == pb.SCORE_SCOPE_CHUNK
    assert sent.query.fusion.queries[1].score_only
    assert sent.query.fusion.candidate_depth == 2
    assert sent.query.fusion.filters[0].phrase.text == "red blood cells"
    assert sent.l1.weights["dense"] == 0.5
    assert sent.HasField("score_export")
    assert result.ranking_method == "linear_v1"
    raw = result.hits[0].candidate_scores
    assert raw.document == {"title": 0.0}
    assert raw.passages[0].scores == {"dense": -0.5}
    assert raw.passages[0].l1_score == -0.25


@pytest.mark.asyncio
async def test_candidate_export_roundtrip_preserves_branch_identity_and_negative_passage_score():
    from hermes_client_python import hermes_pb2 as pb

    client = HermesClient("localhost:50051")
    client._ensure_connected = lambda: None
    client._search_stub = AsyncMock()
    client._search_stub.Search.return_value = pb.SearchResponse(
        ranking_method="fusion_candidates_v1",
        fusion_candidates=[
            pb.FusionCandidateList(
                query_index=0,
                candidates=[
                    pb.FusionCandidate(
                        address=pb.DocAddress(segment_id="abc", doc_id=2),
                        score=-0.5,
                        ordinal_scores=[pb.OrdinalScore(ordinal=7, score=-0.25)],
                    )
                ],
            )
        ],
    )
    result = await client.search(
        "docs",
        query={
            "fusion": {
                "method": "candidates",
                "candidate_depth": 12,
                "queries": [
                    {"query": {"match": {"field": "body", "text": "hemoglobin"}}}
                ],
            }
        },
    )
    sent = client._search_stub.Search.call_args.args[0]
    assert sent.query.fusion.method == pb.FUSION_CANDIDATES
    branch = result.fusion_candidates[0]
    assert branch.query_index == 0
    assert branch.candidates[0].address.segment_id == "abc"
    assert branch.candidates[0].score == -0.5
    assert branch.candidates[0].ordinal_scores[0].ordinal == 7
    assert branch.candidates[0].ordinal_scores[0].score == -0.25
