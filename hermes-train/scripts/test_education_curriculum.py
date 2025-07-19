import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from build_education_curriculum import SearchApiClient
from education_curriculum import (
    build_record_pools,
    discover_with_search_api,
    mix_replay,
    resolve_from_alloydb,
    select_documents,
    validate_config,
    write_outputs,
)


def minimal_config():
    return {
        "version": 1,
        "search_api": {
            "url": "http://search-api/internal/v2/search/",
            "index": "documents",
            "mode": "hybrid",
            "page_size": 2,
            "limit_per_search": 2,
        },
        "alloydb": {"batch_size": 2},
        "output": {
            "compression": "none",
            "seed": 3,
            "validation_fraction": 0,
            "max_chunk_chars": 500,
            "min_chunk_chars": 10,
            "max_chunks_per_document": 4,
            "hard_negatives": 1,
        },
        "stages": [
            {
                "name": "foundations",
                "languages": ["en"],
                "document_types": ["book"],
                "min_content_chars": 20,
                "max_documents": 10,
                "min_documents": 2,
                "title_allow_patterns": ["Primer"],
                "searches": [
                    {
                        "name": "primers",
                        "language": "en",
                        "query": "beginner primer",
                    }
                ],
                "training": {
                    "causal": {
                        "sequence_length": 128,
                        "batch_size": 2,
                        "gradient_accumulation": 1,
                    },
                    "retrieval": {
                        "sequence_length": 64,
                        "batch_size": 2,
                        "gradient_accumulation": 1,
                    },
                },
            }
        ],
    }


class FakeSearchApi:
    def __init__(self):
        self.calls = []
        self.hits = [
            {
                "id": "doc-a",
                "score": 2.0,
                "uris": ["s3://private-layout/a"],
                "content": "SEARCH COPY MUST NEVER ENTER TRAINING",
            },
            {
                "id": "doc-b",
                "score": 1.0,
                "uris": ["urn:arbitrary:document:b"],
                "content": "SEARCH COPY MUST NEVER ENTER TRAINING",
            },
        ]

    async def get_index_info(self, index_name):
        return {"index_name": index_name, "num_docs": 20, "num_segments": 2}

    async def search(self, index_name, **kwargs):
        self.calls.append((index_name, kwargs))
        offset = kwargs["offset"]
        limit = kwargs["limit"]
        return {
            "hits": self.hits[offset : offset + limit],
            "total_hits": len(self.hits),
        }


class FakeConnection:
    def __init__(self):
        self.queries = []

    async def fetch(self, query, ids):
        self.queries.append((query, list(ids)))
        return [
            {
                "id": document_id,
                "type": "book",
                "uris": [f"custom+archive://opaque/{document_id}"],
                "blob": {
                    "title": "First Primer"
                    if document_id == "doc-a"
                    else "Second Primer",
                    "abstract": "Canonical abstract",
                    "content": (
                        f"CANONICAL FULL COPY {document_id}. "
                        "Letters form words and words form clear sentences."
                    ),
                    "languages": ["en"],
                },
            }
            for document_id in ids
        ]


class Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()

    def acquire(self):
        return Acquire(self.connection)


class EducationCurriculumTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_api_retries_capacity_errors_and_discards_text(self):
        class FlakyClient(SearchApiClient):
            def __init__(self):
                super().__init__(
                    {
                        "url": "http://search-api/internal/v2/search/",
                        "max_retries": 2,
                        "retry_initial_seconds": 0,
                        "retry_max_seconds": 0,
                    }
                )
                self.attempts = 0
                self.payload = None

            def _request_once(self, url, payload=None):
                self.attempts += 1
                self.payload = payload
                if self.attempts < 3:
                    raise HTTPError(url, 500, "capacity", {}, None)
                return {
                    "embed_ms": 1,
                    "total_hits": 1,
                    "hits": [
                        {
                            "id": "doc-a",
                            "score": 2.0,
                            "document": {
                                "uris": ["urn:document:a"],
                                "content": "must be discarded",
                            },
                        }
                    ],
                }

        client = FlakyClient()
        response = await client.search(
            "documents",
            query="primer",
            limit=1,
            offset=0,
            language="en",
        )

        self.assertEqual(client.attempts, 3)
        self.assertEqual(client.payload["mode"], "hybrid")
        self.assertEqual(client.payload["index_names"], ["documents"])
        self.assertNotIn("vector", client.payload)
        self.assertNotIn("embedding", client.payload)
        self.assertEqual(
            response,
            {
                "hits": [{"id": "doc-a", "score": 2.0, "uris": ["urn:document:a"]}],
                "total_hits": 1,
            },
        )

    async def test_search_api_discovers_and_alloydb_supplies_full_copy(self):
        config = minimal_config()
        validate_config(config)
        search_api = FakeSearchApi()
        discovery = await discover_with_search_api(search_api, config)

        self.assertEqual(len(search_api.calls), 1)
        self.assertEqual(search_api.calls[0][1]["query"], "beginner primer")
        self.assertEqual(search_api.calls[0][1]["language"], "en")

        pool = FakePool()
        documents = await resolve_from_alloydb(pool, ["doc-a", "doc-b"], batch_size=2)
        query, ids = pool.connection.queries[0]
        self.assertIn("FROM public.documents_assembled", query)
        self.assertIn("id = ANY($1::text[])", query)
        self.assertEqual(ids, ["doc-a", "doc-b"])

        selected, rejections = select_documents(config, discovery, documents)
        self.assertEqual(len(selected["foundations"]), 2)
        self.assertEqual(rejections["foundations"], {})
        self.assertEqual(
            selected["foundations"][0].document.uris[0],
            "custom+archive://opaque/doc-a",
        )
        train_causal, _, train_retrieval, _ = build_record_pools(config, selected)
        self.assertIn("CANONICAL FULL COPY", train_causal["foundations"][0]["text"])
        self.assertNotIn("SEARCH COPY", train_causal["foundations"][0]["text"])
        self.assertEqual(len(train_retrieval["foundations"]), 2)
        self.assertEqual(len(train_retrieval["foundations"][0]["negatives"]), 1)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = write_outputs(config, output, discovery, selected, rejections)
            record = json.loads(
                (output / "01-foundations-causal.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("CANONICAL FULL COPY", record["text"])
            self.assertEqual(
                manifest["contract"],
                "Search API discovery IDs -> AlloyDB documents_assembled full copies",
            )
            curriculum = json.loads((output / "curriculum.json").read_text())
            self.assertEqual(
                [stage["objective"]["type"] for stage in curriculum["stages"]],
                ["causal_lm", "contrastive_retrieval"],
            )

    def test_replay_is_deterministic_and_does_not_duplicate(self):
        current = [{"document_id": f"new-{index}", "chunk": 0} for index in range(8)]
        old = [{"document_id": f"old-{index}", "chunk": 0} for index in range(8)]
        first = mix_replay(
            "school",
            current,
            {"foundations": old},
            {"foundations": 0.2},
            seed=9,
        )
        second = mix_replay(
            "school",
            current,
            {"foundations": old},
            {"foundations": 0.2},
            seed=9,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len({record["document_id"] for record in first}), 10)

    def test_non_text_query_is_rejected(self):
        config = minimal_config()
        config["stages"][0]["searches"][0]["query"] = {
            "boolean": {"must": [{"all": {}}]}
        }
        with self.assertRaisesRegex(ValueError, "query text"):
            validate_config(config)

    def test_non_fusion_search_mode_is_rejected(self):
        config = minimal_config()
        config["search_api"]["mode"] = "sparse"
        with self.assertRaisesRegex(ValueError, r"sparse\+dense fusion"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
