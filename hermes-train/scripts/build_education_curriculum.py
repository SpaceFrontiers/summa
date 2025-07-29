#!/usr/bin/env python3
"""Build a staged educational corpus from Search API IDs and AlloyDB full text."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from curriculum_streaming import GigaTokenCounter, build_live_streaming
from education_curriculum import SearchPageCapacityError, build_live, load_config


class SearchApiClient:
    """Minimal client for Search API's embedding-backed hybrid path."""

    RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        config: dict,
        url_override: str | None = None,
        cache_dir: Path | None = None,
    ):
        url = url_override or config["url"]
        self.base_url = url.rstrip("/") + "/"
        self.timeout = float(config.get("timeout_seconds", 60))
        self.deduplicate = bool(config.get("deduplicate", True))
        self.max_retries = int(config.get("max_retries", 6))
        self.retry_initial_seconds = float(config.get("retry_initial_seconds", 1))
        self.retry_max_seconds = float(config.get("retry_max_seconds", 15))
        self.filter_language = bool(config.get("filter_language", True))
        self.filter_document_types = bool(config.get("filter_document_types", False))
        self.heap_factor = config.get("heap_factor")
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, payload: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        return self.cache_dir / f"search-{digest}.json"

    def _load_cached(self, payload: dict) -> dict | None:
        payloads = [payload]
        if payload.get("return_documents") is False:
            # ID-only caches contain the same sanitized fields as older
            # hydrated requests. Reuse them so deploying the Search API fix
            # does not invalidate a multi-hour discovery pass.
            legacy = dict(payload)
            legacy.pop("return_documents")
            payloads.append(legacy)
        for candidate in payloads:
            path = self._cache_path(candidate)
            if path is None or not path.exists():
                continue
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                logging.warning(
                    "ignoring unreadable discovery cache %s: %s", path, error
                )
                continue
            if not isinstance(cached, dict) or not isinstance(cached.get("hits"), list):
                logging.warning("ignoring invalid discovery cache %s", path)
                continue
            return cached
        return None

    def _store_cached(self, payload: dict, result: dict) -> None:
        path = self._cache_path(payload)
        if path is None:
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _request_once(self, url: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-Source": "training-curriculum",
            },
            method="GET" if payload is None else "POST",
        )
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            parsed = json.load(response)
        if not isinstance(parsed, dict):
            raise RuntimeError("Search API returned a non-object response")
        return parsed

    async def _request(self, url: str, payload: dict | None = None) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(self._request_once, url, payload)
            except HTTPError as error:
                response_body = (
                    error.read(4096).lower() if error.fp is not None else b""
                )
                hydration_budget_exceeded = b"hydration budget" in response_body
                retryable = (
                    error.code in self.RETRYABLE_STATUS
                    and not hydration_budget_exceeded
                )
                reason = (
                    f"HTTP {error.code} (hydration budget)"
                    if hydration_budget_exceeded
                    else f"HTTP {error.code}"
                )
                error.close()
                if hydration_budget_exceeded:
                    raise SearchPageCapacityError(
                        "Search API hydration budget exceeded; reduce "
                        "search_api.page_size or the per-query search limit"
                    ) from error
                if not retryable or attempt == self.max_retries:
                    error_type = SearchPageCapacityError if retryable else RuntimeError
                    raise error_type(f"Search API request failed: {reason}") from error
            except (URLError, TimeoutError) as error:
                reason = type(error).__name__
                if attempt == self.max_retries:
                    raise SearchPageCapacityError(
                        f"Search API request failed: {reason}"
                    ) from error

            delay = min(
                self.retry_initial_seconds * (2**attempt),
                self.retry_max_seconds,
            )
            logging.warning(
                "Search API request failed with %s; retrying in %.1fs (%d/%d)",
                reason,
                delay,
                attempt + 1,
                self.max_retries,
            )
            await asyncio.sleep(delay)

        raise AssertionError("unreachable Search API retry state")

    async def get_index_info(self, index_name: str) -> dict:
        query = urlencode({"index_name": index_name})
        return await self._request(f"{self.base_url}index-info?{query}")

    async def search(
        self,
        index_name: str,
        *,
        query: str,
        limit: int,
        offset: int,
        language: str | None,
        document_types: tuple[str, ...] = (),
        issued_after: int | None = None,
        issued_before: int | None = None,
    ) -> dict:
        payload = {
            "query": query,
            "index_names": [index_name],
            "mode": "hybrid",
            "limit": limit,
            "offset": offset,
            "rerank": False,
            "cross_rerank": False,
            "deduplicate": self.deduplicate,
            "return_documents": False,
        }
        if language and self.filter_language:
            payload["possible_languages"] = [language]
        if document_types and self.filter_document_types:
            payload["filter_types"] = list(document_types)
        if issued_after is not None:
            payload["filter_issued_after"] = issued_after
        if issued_before is not None:
            payload["filter_issued_before"] = issued_before
        if self.heap_factor is not None:
            payload["heap_factor"] = self.heap_factor

        cached = self._load_cached(payload)
        if cached is not None:
            return cached

        response = await self._request(self.base_url, payload)
        if response.get("embed_ms") is None:
            raise RuntimeError(
                "Search API hybrid request did not run its dense query embedder"
            )

        # Deliberately discard every enriched field except ID and URI metadata.
        # Canonical title/abstract/content must come from the later AlloyDB read.
        hits = []
        for raw_hit in response.get("hits", []):
            document = raw_hit.get("document") or {}
            hits.append(
                {
                    "id": raw_hit.get("id", ""),
                    "score": raw_hit.get("score", 0.0),
                    "uris": document.get("uris", []),
                }
            )
        result = {"hits": hits, "total_hits": int(response.get("total_hits", 0))}
        self._store_cached(payload, result)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--search-api-url",
        default=os.environ.get("SEARCH_API_URL"),
        help="override search_api.url (for example, an SSH tunnel)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="ID-only discovery cache (defaults to OUTPUT/.discovery-cache)",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        help="cap each configured Search API query (useful for a smoke build)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Hermes tokenizer.json used for exact multi-billion-token sizing",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    try:
        import asyncpg
    except ImportError as error:
        raise SystemExit(
            "install live builder dependencies: pip install asyncpg zstandard"
        ) from error

    config = load_config(args.config)
    cache_dir = args.cache_dir or args.output / ".discovery-cache"
    search_client = SearchApiClient(
        config["search_api"],
        args.search_api_url,
        cache_dir=cache_dir,
    )
    alloydb = config.get("alloydb", {})
    pool = await asyncpg.create_pool(
        min_size=1,
        max_size=alloydb.get("connections", 4),
        command_timeout=alloydb.get("timeout_seconds", 120),
    )
    try:
        if config.get("output", {}).get("streaming", False):
            if args.tokenizer is None:
                raise SystemExit("streaming curriculum builds require --tokenizer")
            manifest = await build_live_streaming(
                search_client,
                pool,
                config,
                args.output,
                GigaTokenCounter(args.tokenizer),
                search_limit_override=args.search_limit,
            )
        else:
            manifest = await build_live(
                search_client,
                pool,
                config,
                args.output,
                search_limit_override=args.search_limit,
            )
    finally:
        await pool.close()

    selected = sum(stage["selected_documents"] for stage in manifest["stages"].values())
    logging.info(
        "built %d selected documents into %s; curriculum=%s",
        selected,
        args.output,
        args.output / manifest["curriculum"],
    )
    if "tokens" in manifest:
        logging.info(
            "unique causal tokens=%d curriculum causal tokens=%d",
            manifest["tokens"]["unique_train_causal"],
            manifest["tokens"]["curriculum_train_causal"],
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.search_limit is not None and args.search_limit <= 0:
        raise SystemExit("--search-limit must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
