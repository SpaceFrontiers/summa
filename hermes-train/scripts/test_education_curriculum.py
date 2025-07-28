import asyncio
import json
import tempfile
import unittest
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

from audit_education_curriculum import audit
from build_education_curriculum import SearchApiClient
from curriculum_streaming import _sampled_records, build_live_streaming
from education_curriculum import (
    Candidate,
    FullDocument,
    SearchPageCapacityError,
    _eligible,
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
                        "\x1f\ufffd More words."
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


class FakeTokenCounter:
    @property
    def metadata(self):
        return {
            "engine": "test",
            "version": "1",
            "tokenizer_sha256": "fake-tokenizer",
            "vocab_size": 100,
        }

    def count_batch(self, texts):
        return [len(text.split()) for text in texts]


class EducationCurriculumTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_splits_capacity_limited_pages_without_losing_depth(self):
        class PageLimitedSearch(FakeSearchApi):
            def __init__(self):
                super().__init__()
                self.hits = [
                    {"id": f"doc-{index}", "score": 100 - index, "uris": []}
                    for index in range(100)
                ]

            async def search(self, index_name, **kwargs):
                self.calls.append((index_name, kwargs))
                if kwargs["limit"] > 50:
                    raise SearchPageCapacityError("capacity")
                offset = kwargs["offset"]
                limit = kwargs["limit"]
                return {
                    "hits": self.hits[offset : offset + limit],
                    "total_hits": len(self.hits),
                }

        config = minimal_config()
        config["search_api"]["page_size"] = 100
        config["search_api"]["limit_per_search"] = 100
        search_api = PageLimitedSearch()
        discovery = await discover_with_search_api(search_api, config)
        self.assertEqual([call[1]["limit"] for call in search_api.calls], [100, 50, 50])
        self.assertEqual(len(discovery.candidates["foundations"]), 100)

    async def test_search_contract_errors_do_not_trigger_page_splitting(self):
        class InvalidSearch:
            def __init__(self):
                self.calls = 0

            async def get_index_info(self, index_name):
                return {"num_documents": 1, "num_segments": 1}

            async def search(self, index_name, **kwargs):
                self.calls += 1
                raise RuntimeError("hybrid response omitted embed_ms")

        search_api = InvalidSearch()
        config = minimal_config()
        with self.assertRaisesRegex(RuntimeError, "embed_ms"):
            await discover_with_search_api(search_api, config)
        self.assertEqual(search_api.calls, 1)

    async def test_search_api_does_not_retry_hydration_budget_errors(self):
        class OversizedClient(SearchApiClient):
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

            def _request_once(self, url, payload=None):
                self.attempts += 1
                raise HTTPError(
                    url,
                    500,
                    "resource exhausted",
                    {},
                    BytesIO(b"Search response exceeds the hydration budget"),
                )

        client = OversizedClient()
        with self.assertRaisesRegex(RuntimeError, "page_size"):
            await client.search(
                "documents",
                query="primer",
                limit=250,
                offset=0,
                language="en",
            )
        self.assertEqual(client.attempts, 1)

    async def test_search_api_retries_capacity_errors_and_discards_text(self):
        class FlakyClient(SearchApiClient):
            def __init__(self, cache_dir):
                super().__init__(
                    {
                        "url": "http://search-api/internal/v2/search/",
                        "max_retries": 2,
                        "retry_initial_seconds": 0,
                        "retry_max_seconds": 0,
                        "filter_language": False,
                        "filter_document_types": True,
                        "heap_factor": 0.2,
                        "cross_rerank": False,
                    },
                    cache_dir=cache_dir,
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

        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            client = FlakyClient(cache_dir)
            response = await client.search(
                "documents",
                query="primer",
                limit=1,
                offset=0,
                language="en",
                document_types=("book",),
            )
            cached_response = await client.search(
                "documents",
                query="primer",
                limit=1,
                offset=0,
                language="en",
                document_types=("book",),
            )
            cache_text = next(cache_dir.glob("search-*.json")).read_text()

        self.assertEqual(client.attempts, 3)
        self.assertEqual(client.payload["mode"], "hybrid")
        self.assertEqual(client.payload["index_names"], ["documents"])
        self.assertFalse(client.payload["rerank"])
        self.assertFalse(client.payload["cross_rerank"])
        self.assertFalse(client.payload["return_documents"])
        self.assertNotIn("possible_languages", client.payload)
        self.assertEqual(client.payload["filter_types"], ["book"])
        self.assertNotIn("rerank_factor", client.payload)
        self.assertNotIn("reranker_limit", client.payload)
        self.assertEqual(client.payload["heap_factor"], 0.2)
        self.assertNotIn("vector", client.payload)
        self.assertNotIn("embedding", client.payload)
        self.assertEqual(
            response,
            {
                "hits": [{"id": "doc-a", "score": 2.0, "uris": ["urn:document:a"]}],
                "total_hits": 1,
            },
        )
        self.assertEqual(cached_response, response)
        self.assertNotIn("must be discarded", cache_text)

    async def test_id_only_search_reuses_legacy_sanitized_cache(self):
        class OfflineClient(SearchApiClient):
            def _request_once(self, url, payload=None):
                raise AssertionError("legacy cache should avoid a network request")

        with tempfile.TemporaryDirectory() as temporary:
            client = OfflineClient(
                {
                    "url": "http://search-api/internal/v2/search/",
                    "filter_language": False,
                    "filter_document_types": True,
                    "heap_factor": 0.2,
                    "deduplicate": True,
                },
                cache_dir=Path(temporary),
            )
            legacy_payload = {
                "query": "primer",
                "index_names": ["documents"],
                "mode": "hybrid",
                "limit": 1,
                "offset": 0,
                "rerank": False,
                "cross_rerank": False,
                "deduplicate": True,
                "filter_types": ["book"],
                "heap_factor": 0.2,
            }
            expected = {
                "hits": [{"id": "doc-a", "score": 2.0, "uris": []}],
                "total_hits": 1,
            }
            client._store_cached(legacy_payload, expected)
            response = await client.search(
                "documents",
                query="primer",
                limit=1,
                offset=0,
                language="en",
                document_types=("book",),
            )

        self.assertEqual(response, expected)

    async def test_search_api_discovers_and_alloydb_supplies_full_copy(self):
        config = minimal_config()
        validate_config(config)
        search_api = FakeSearchApi()
        discovery = await discover_with_search_api(search_api, config)

        self.assertEqual(len(search_api.calls), 1)
        self.assertEqual(search_api.calls[0][1]["query"], "beginner primer")
        self.assertEqual(search_api.calls[0][1]["language"], "en")
        self.assertEqual(search_api.calls[0][1]["document_types"], ("book",))

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
            self.assertNotIn("\x1f", record["text"])
            self.assertNotIn("\ufffd", record["text"])
            self.assertEqual(
                manifest["contract"],
                "Search API discovery IDs -> AlloyDB documents_assembled full copies",
            )
            stage_stats = manifest["stages"]["foundations"]
            self.assertEqual(stage_stats["languages"], {"en": 2})
            self.assertEqual(stage_stats["document_types"], {"book": 2})
            self.assertGreater(stage_stats["canonical_content_characters"], 0)
            causal_file = next(
                item
                for item in manifest["files"]
                if item["path"] == "01-foundations-causal.jsonl"
            )
            self.assertEqual(causal_file["uncompressed_bytes"], causal_file["bytes"])
            self.assertGreater(causal_file["payload_characters"], 0)
            curriculum = json.loads((output / "curriculum.json").read_text())
            self.assertEqual(
                [stage["objective"]["type"] for stage in curriculum["stages"]],
                ["causal_lm", "contrastive_retrieval"],
            )

    async def test_partitioned_streaming_build_is_exact_and_bounded(self):
        config = minimal_config()
        config["search_api"]["concurrency"] = 2
        config["search_api"]["time_partition_profiles"] = {
            "test-yearly": [
                {"name": "old", "issued_before": 0},
                {"name": "new", "issued_after": 0},
            ]
        }
        config["stages"][0]["time_partition_profile"] = "test-yearly"
        config["output"].update(
            {
                "streaming": True,
                "minimum_causal_tokens": 1,
                "target_causal_tokens": 100,
                "cleanup_build_artifacts": True,
            }
        )
        validate_config(config)
        search_api = FakeSearchApi()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = await build_live_streaming(
                search_api,
                FakePool(),
                config,
                output,
                FakeTokenCounter(),
            )
            records = [
                json.loads(line)
                for line in (output / "01-foundations-causal.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            audit_result = audit(
                Namespace(
                    output=output,
                    tokenizer=None,
                    token_batch_characters=1_000_000,
                )
            )
            manifest_path = output / "manifest.json"
            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered["config"]["output"]["seed"] += 1
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "config_sha256"):
                audit(
                    Namespace(
                        output=output,
                        tokenizer=None,
                        token_batch_characters=1_000_000,
                    )
                )
            self.assertFalse((output / ".build").exists())

        self.assertEqual(len(search_api.calls), 2)
        calls = [call[1] for call in search_api.calls]
        self.assertEqual(
            {(call["issued_after"], call["issued_before"]) for call in calls},
            {(None, 0), (0, None)},
        )
        self.assertEqual(manifest["tokens"]["minimum_required"], 1)
        self.assertTrue(manifest["tokens"]["minimum_enforced"])
        self.assertEqual(manifest["config"], config)
        self.assertGreater(manifest["tokens"]["unique_train_causal"], 1)
        self.assertEqual(
            manifest["tokens"]["unique_train_causal"],
            sum(record["token_count"] for record in records),
        )
        self.assertEqual(audit_result["train_eval_overlap"], 0)
        self.assertTrue(audit_result["config_verified"])
        self.assertEqual(
            audit_result["unique_train_causal_tokens"],
            manifest["tokens"]["unique_train_causal"],
        )

    async def test_final_streaming_stage_stops_at_exact_token_target(self):
        concurrency = {"active": 0, "maximum": 0}

        class SlowConnection(FakeConnection):
            async def fetch(self, query, ids):
                concurrency["active"] += 1
                concurrency["maximum"] = max(
                    concurrency["maximum"], concurrency["active"]
                )
                try:
                    await asyncio.sleep(0.001)
                    return await super().fetch(query, ids)
                finally:
                    concurrency["active"] -= 1

        class ConcurrentPool:
            def acquire(self):
                return Acquire(SlowConnection())

        search_api = FakeSearchApi()
        search_api.hits = [
            {"id": f"doc-{letter}", "score": 10 - index, "uris": []}
            for index, letter in enumerate("abcdef")
        ]
        config = minimal_config()
        config["search_api"]["page_size"] = 2
        config["search_api"]["limit_per_search"] = 6
        config["alloydb"]["connections"] = 2
        config["alloydb"]["prefetch_batches"] = 2
        config["output"].update(
            {
                "streaming": True,
                "token_batch_characters": 1,
                "minimum_causal_tokens": 1,
                "target_causal_tokens": 1,
                "cleanup_build_artifacts": True,
            }
        )
        validate_config(config)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = await build_live_streaming(
                search_api,
                ConcurrentPool(),
                config,
                Path(temporary),
                FakeTokenCounter(),
            )

        stage = manifest["stages"]["foundations"]
        self.assertEqual(stage["selected_documents"], 2)
        self.assertEqual(stage["rejections"]["token_target"], 4)
        self.assertEqual(concurrency["maximum"], 2)
        self.assertGreaterEqual(manifest["tokens"]["unique_train_causal"], 1)

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

    def test_streaming_replay_selects_exactly_in_constant_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"document_id": f"doc-{index}"}) + "\n"
                    for index in range(100)
                ),
                encoding="utf-8",
            )
            first = list(
                _sampled_records(
                    path, 100, 23, seed=7, namespace="advanced:foundations"
                )
            )
            second = list(
                _sampled_records(
                    path, 100, 23, seed=7, namespace="advanced:foundations"
                )
            )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 23)
        self.assertEqual(len({record["document_id"] for record in first}), 23)

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

    def test_unknown_time_partition_profile_is_rejected(self):
        config = minimal_config()
        config["stages"][0]["time_partition_profile"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown time partition profile"):
            validate_config(config)

    def test_curriculum_discovery_cannot_enable_reranking(self):
        config = minimal_config()
        config["search_api"]["cross_rerank"] = True
        with self.assertRaisesRegex(ValueError, "cross_rerank must be false"):
            validate_config(config)

        config = minimal_config()
        config["search_api"]["rerank_factor"] = 2
        with self.assertRaisesRegex(ValueError, "rerank_factor is not supported"):
            validate_config(config)

    def test_concentrated_extraction_corruption_is_rejected(self):
        stage = minimal_config()["stages"][0]
        candidate = Candidate(document_id="corrupt", stage="foundations")
        base = {
            "title": "Primer",
            "languages": ["en"],
        }
        replacement = FullDocument(
            document_id="corrupt",
            document_type="book",
            uris=(),
            blob={**base, "content": "letters and words " * 20 + "\ufffd" * 20},
        )
        control = FullDocument(
            document_id="corrupt",
            document_type="book",
            uris=(),
            blob={
                **base,
                "content": "letters and words " * 10
                + "\x1f" * 20
                + "more letters and words " * 10,
            },
        )
        quality = {
            "max_control_character_fraction": 0.01,
            "max_replacement_character_fraction": 0.001,
        }
        self.assertEqual(
            _eligible(stage, candidate, replacement, quality),
            "replacement_characters",
        )
        self.assertEqual(
            _eligible(stage, candidate, control, quality),
            "control_characters",
        )


if __name__ == "__main__":
    unittest.main()
