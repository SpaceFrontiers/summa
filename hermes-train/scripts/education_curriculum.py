"""Search-API-discovered, AlloyDB-resolved educational curriculum builder.

Search API is deliberately the only discovery surface. AlloyDB is queried only
by the document IDs returned by Search API and supplies the canonical full copies.
The pure transformation functions in this module are kept separate from the
live clients so selection, replay, and leakage controls are testable.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_SEARCH_WINDOW = 50_000
MAX_SEARCH_API_PAGE = 500


@dataclass(frozen=True)
class SearchMatch:
    stage: str
    search: str
    retrieval_query: str
    language: str | None
    rank: int
    score: float


@dataclass
class Candidate:
    document_id: str
    stage: str
    uris: tuple[str, ...] = ()
    matches: list[SearchMatch] = field(default_factory=list)

    @property
    def best_rank(self) -> int:
        return min(match.rank for match in self.matches)

    @property
    def best_score(self) -> float:
        return max(match.score for match in self.matches)


@dataclass(frozen=True)
class FullDocument:
    document_id: str
    document_type: str
    uris: tuple[str, ...]
    blob: dict[str, Any]

    @property
    def title(self) -> str:
        value = self.blob.get("title", "")
        return value.strip() if isinstance(value, str) else ""

    @property
    def abstract(self) -> str:
        value = self.blob.get("abstract", "")
        return value.strip() if isinstance(value, str) else ""

    @property
    def content(self) -> str:
        value = self.blob.get("content", "")
        return value.strip() if isinstance(value, str) else ""


@dataclass(frozen=True)
class SelectedDocument:
    stage: str
    candidate: Candidate
    document: FullDocument
    language: str
    validation: bool


@dataclass
class Discovery:
    candidates: dict[str, dict[str, Candidate]]
    searches: list[dict[str, Any]]
    index: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0, label
    )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0, label
    )
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0,
        label,
    )
    return float(value)


def validate_config(config: dict[str, Any]) -> None:
    """Validate the versioned build configuration before any network I/O."""
    _require(
        config.get("version") == 1, "only education curriculum version 1 is supported"
    )
    search_api = config.get("search_api")
    _require(isinstance(search_api, dict), "search_api configuration is required")
    _require(bool(search_api.get("index")), "search_api.index is required")
    _require(bool(search_api.get("url")), "search_api.url is required")
    _require(
        search_api.get("mode", "hybrid") == "hybrid",
        "search_api.mode must be hybrid (sparse+dense fusion)",
    )
    page_size = _positive_int(
        search_api.get("page_size", 250), "search_api.page_size must be positive"
    )
    _require(
        page_size <= MAX_SEARCH_API_PAGE,
        "search_api.page_size exceeds the Search API page limit",
    )
    _nonnegative_int(
        search_api.get("max_retries", 6),
        "search_api.max_retries must be non-negative",
    )
    initial_retry = _nonnegative_number(
        search_api.get("retry_initial_seconds", 1),
        "search_api.retry_initial_seconds must be non-negative",
    )
    maximum_retry = _nonnegative_number(
        search_api.get("retry_max_seconds", 15),
        "search_api.retry_max_seconds must be non-negative",
    )
    _require(
        maximum_retry >= initial_retry,
        "search_api.retry_max_seconds must be at least retry_initial_seconds",
    )

    stages = config.get("stages")
    _require(isinstance(stages, list) and stages, "at least one stage is required")
    names: set[str] = set()
    for index, stage in enumerate(stages):
        _require(isinstance(stage, dict), f"stage {index} must be an object")
        name = stage.get("name")
        _require(isinstance(name, str) and name.strip(), f"stage {index} needs a name")
        _require(name not in names, f"duplicate stage name {name!r}")
        searches = stage.get("searches")
        _require(
            isinstance(searches, list) and searches, f"stage {name!r} needs searches"
        )
        search_names: set[str] = set()
        for search in searches:
            _require(
                isinstance(search, dict), f"stage {name!r} search must be an object"
            )
            search_name = search.get("name")
            _require(
                isinstance(search_name, str) and search_name.strip(),
                f"stage {name!r} search needs a name",
            )
            _require(
                search_name not in search_names,
                f"duplicate search {search_name!r} in stage {name!r}",
            )
            query = search.get("query")
            _require(
                isinstance(query, str) and query.strip(),
                f"search {search_name!r} needs non-empty query text",
            )
            limit = _positive_int(
                search.get("limit", search_api.get("limit_per_search", 1_000)),
                f"search {search_name!r} limit must be positive",
            )
            _require(
                limit <= MAX_SEARCH_WINDOW,
                f"search {search_name!r} exceeds 50,000 hits",
            )
            search_names.add(search_name)

        replay = stage.get("replay", {})
        _require(isinstance(replay, dict), f"stage {name!r} replay must be an object")
        total_replay = 0.0
        for source, fraction in replay.items():
            _require(
                source in names, f"stage {name!r} replays non-prior stage {source!r}"
            )
            _require(
                isinstance(fraction, (int, float)) and 0 < fraction < 1,
                f"stage {name!r} replay fraction for {source!r} must be between zero and one",
            )
            total_replay += float(fraction)
        _require(
            total_replay < 1, f"stage {name!r} replay fractions must sum below one"
        )
        _validate_training(name, stage.get("training"))
        names.add(name)

    output = config.get("output", {})
    _require(isinstance(output, dict), "output must be an object")
    _require(
        output.get("compression", "zstd") in {"none", "zstd"}, "invalid compression"
    )
    validation_fraction = output.get("validation_fraction", 0.01)
    _require(
        isinstance(validation_fraction, (int, float))
        and 0 <= validation_fraction < 0.5,
        "output.validation_fraction must be in [0, 0.5)",
    )


def _validate_training(stage_name: str, training: Any) -> None:
    _require(
        isinstance(training, dict), f"stage {stage_name!r} needs training geometry"
    )
    for objective in ("causal", "retrieval"):
        geometry = training.get(objective)
        _require(
            isinstance(geometry, dict),
            f"stage {stage_name!r} needs {objective} geometry",
        )
        _positive_int(
            geometry.get("sequence_length"),
            f"stage {stage_name!r} {objective} sequence_length must be positive",
        )
        batch_size = _positive_int(
            geometry.get("batch_size"),
            f"stage {stage_name!r} {objective} batch_size must be positive",
        )
        if objective == "retrieval":
            _require(
                batch_size >= 2,
                f"stage {stage_name!r} retrieval batch must be at least two",
            )
        _positive_int(
            geometry.get("gradient_accumulation"),
            f"stage {stage_name!r} {objective} gradient_accumulation must be positive",
        )


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "curriculum config must be a JSON object")
    validate_config(config)
    return config


def _as_scalar(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return value.strip() if isinstance(value, str) else ""


def _as_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


async def discover_with_search_api(
    client: Any,
    config: dict[str, Any],
    *,
    search_limit_override: int | None = None,
) -> Discovery:
    """Discover through Search API's sparse+dense fusion path only."""
    search_api = config["search_api"]
    index_name = search_api["index"]
    info = await client.get_index_info(index_name)
    index = {
        "name": info.get("index_name", index_name),
        "documents": info.get("num_docs"),
        "segments": info.get("num_segments"),
    }
    page_size = search_api.get("page_size", 250)
    candidates: dict[str, dict[str, Candidate]] = {
        stage["name"]: {} for stage in config["stages"]
    }
    search_stats: list[dict[str, Any]] = []

    for stage in config["stages"]:
        stage_candidates = candidates[stage["name"]]
        for search in stage["searches"]:
            configured_limit = search.get(
                "limit", search_api.get("limit_per_search", 1_000)
            )
            limit = (
                min(configured_limit, search_limit_override)
                if search_limit_override is not None
                else configured_limit
            )
            offset = 0
            unique_ids: set[str] = set()
            total_hits: int | None = None
            while offset < limit:
                request_limit = min(page_size, limit - offset)
                response = await client.search(
                    index_name,
                    query=search["query"],
                    limit=request_limit,
                    offset=offset,
                    language=search.get("language"),
                )
                total_hits = response["total_hits"]
                if not response["hits"]:
                    break
                for page_rank, hit in enumerate(response["hits"]):
                    document_id = _as_scalar(hit.get("id"))
                    if not document_id or document_id in unique_ids:
                        continue
                    unique_ids.add(document_id)
                    candidate = stage_candidates.setdefault(
                        document_id,
                        Candidate(
                            document_id=document_id,
                            stage=stage["name"],
                            uris=_as_strings(hit.get("uris")),
                        ),
                    )
                    score = float(hit.get("score", 0.0))
                    candidate.matches.append(
                        SearchMatch(
                            stage=stage["name"],
                            search=search["name"],
                            retrieval_query=search["query"],
                            language=search.get("language"),
                            rank=offset + page_rank,
                            score=score if math.isfinite(score) else 0.0,
                        )
                    )
                offset += len(response["hits"])
                if (
                    len(response["hits"]) < request_limit
                    or offset >= response["total_hits"]
                ):
                    break
            search_stats.append(
                {
                    "stage": stage["name"],
                    "search": search["name"],
                    "total_hits": total_hits,
                    "fetched_unique_ids": len(unique_ids),
                    "limit": limit,
                }
            )
    return Discovery(candidates=candidates, searches=search_stats, index=index)


async def resolve_from_alloydb(
    pool: Any,
    document_ids: list[str],
    *,
    batch_size: int = 500,
) -> dict[str, FullDocument]:
    """Resolve only Search-API-returned IDs from AlloyDB's canonical full-text view."""
    documents: dict[str, FullDocument] = {}
    query = """
        SELECT id, type, uris, blob
        FROM public.documents_assembled
        WHERE id = ANY($1::text[])
          AND is_retracted = FALSE
    """
    async with pool.acquire() as connection:
        for offset in range(0, len(document_ids), batch_size):
            ids = document_ids[offset : offset + batch_size]
            rows = await connection.fetch(query, ids)
            for raw_row in rows:
                row = dict(raw_row)
                blob = row["blob"]
                if isinstance(blob, str):
                    blob = json.loads(blob)
                if not isinstance(blob, dict):
                    continue
                document_id = str(row["id"])
                documents[document_id] = FullDocument(
                    document_id=document_id,
                    document_type=str(row.get("type") or ""),
                    uris=_as_strings(row.get("uris")),
                    blob=blob,
                )
    return documents


def _document_languages(document: FullDocument, candidate: Candidate) -> list[str]:
    raw = document.blob.get("languages", [])
    if isinstance(raw, str):
        raw = [raw]
    languages = [value.casefold() for value in raw if isinstance(value, str) and value]
    if languages:
        return languages
    return [
        match.language.casefold()
        for match in candidate.matches
        if isinstance(match.language, str) and match.language
    ]


def _first_language(
    document: FullDocument, candidate: Candidate, stage: dict[str, Any]
) -> str:
    languages = _document_languages(document, candidate)
    accepted = [value.casefold() for value in stage.get("languages", [])]
    if accepted:
        for language in languages:
            if language in accepted:
                return language
    return languages[0] if languages else "und"


def _duplicate_line_fraction(content: str) -> float:
    lines = [
        re.sub(r"\s+", " ", line).strip().casefold() for line in content.splitlines()
    ]
    lines = [line for line in lines if len(line) >= 20]
    if not lines:
        return 0.0
    unique = len(set(lines))
    return 1.0 - unique / len(lines)


def _eligible(
    stage: dict[str, Any],
    candidate: Candidate,
    document: FullDocument,
) -> str | None:
    title = document.title
    content = document.content
    if len(content) < stage.get("min_content_chars", 500):
        return "content_too_short"
    if len(content) > stage.get("max_content_chars", 5_000_000):
        return "content_too_long"
    allowed_types = stage.get("document_types", [])
    if allowed_types and document.document_type not in allowed_types:
        return "document_type"
    allowed_languages = [value.casefold() for value in stage.get("languages", [])]
    languages = _document_languages(document, candidate)
    if (
        allowed_languages
        and not set(languages).intersection(allowed_languages)
        and (languages or not stage.get("allow_unknown_language", False))
    ):
        return "language"
    allow_patterns = stage.get("title_allow_patterns", [])
    if allow_patterns and not any(
        re.search(pattern, title, re.IGNORECASE) for pattern in allow_patterns
    ):
        return "title_not_allowed"
    if any(
        re.search(pattern, title, re.IGNORECASE)
        for pattern in stage.get("title_deny_patterns", [])
    ):
        return "title_denied"
    if _duplicate_line_fraction(content) > stage.get(
        "max_duplicate_line_fraction", 0.35
    ):
        return "repeated_lines"
    return None


def _content_fingerprint(document: FullDocument) -> str:
    normalized = re.sub(r"\s+", " ", document.content).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validation_member(document_id: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:split:{document_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def select_documents(
    config: dict[str, Any],
    discovery: Discovery,
    documents: dict[str, FullDocument],
) -> tuple[dict[str, list[SelectedDocument]], dict[str, dict[str, int]]]:
    """Apply stage quality, deduplication, quota, and split policy."""
    output = config.get("output", {})
    seed = int(output.get("seed", 17))
    validation_fraction = float(output.get("validation_fraction", 0.01))
    selected: dict[str, list[SelectedDocument]] = {}
    rejection_stats: dict[str, dict[str, int]] = {}
    assigned_ids: set[str] = set()
    fingerprints: set[str] = set()

    for stage in config["stages"]:
        name = stage["name"]
        accepted: list[SelectedDocument] = []
        rejected: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        candidates = sorted(
            discovery.candidates[name].values(),
            key=lambda candidate: (
                -candidate.best_score,
                candidate.best_rank,
                candidate.document_id,
            ),
        )
        max_documents = stage.get("max_documents", len(candidates))
        per_language = stage.get("max_documents_per_language")
        for candidate in candidates:
            if len(accepted) >= max_documents:
                rejected["stage_quota"] += 1
                continue
            if candidate.document_id in assigned_ids:
                rejected["assigned_to_earlier_stage"] += 1
                continue
            document = documents.get(candidate.document_id)
            if document is None:
                rejected["missing_in_alloydb"] += 1
                continue
            reason = _eligible(stage, candidate, document)
            if reason is not None:
                rejected[reason] += 1
                continue
            language = _first_language(document, candidate, stage)
            if per_language is not None and language_counts[language] >= per_language:
                rejected["language_quota"] += 1
                continue
            fingerprint = _content_fingerprint(document)
            if fingerprint in fingerprints:
                rejected["duplicate_content"] += 1
                continue
            assigned_ids.add(candidate.document_id)
            fingerprints.add(fingerprint)
            language_counts[language] += 1
            accepted.append(
                SelectedDocument(
                    stage=name,
                    candidate=candidate,
                    document=document,
                    language=language,
                    validation=_validation_member(
                        candidate.document_id,
                        seed,
                        validation_fraction,
                    ),
                )
            )
        minimum = stage.get("min_documents", 2)
        _require(
            len(accepted) >= minimum,
            f"stage {name!r} selected {len(accepted)} documents, below min_documents {minimum}",
        )
        selected[name] = accepted
        rejection_stats[name] = dict(sorted(rejected.items()))
    return selected, rejection_stats


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    pieces: list[str] = []
    rest = paragraph.strip()
    while len(rest) > limit:
        split_at = rest.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        pieces.append(rest[:split_at].strip())
        rest = rest[split_at:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def chunk_document(
    selected: SelectedDocument,
    *,
    max_chars: int,
    min_chars: int,
    max_chunks: int,
) -> list[str]:
    """Split a canonical document on paragraph boundaries without overlap."""
    document = selected.document
    title = document.title
    abstract = document.abstract
    content = document.content.replace("\x00", "").replace("\r\n", "\n")
    prefix_parts = []
    if title:
        prefix_parts.append(f"# {title}")
    if abstract and abstract.casefold() not in content[: len(abstract) * 2].casefold():
        prefix_parts.append(abstract)
    prefix = "\n\n".join(prefix_parts)
    body_limit = max(128, max_chars - len(prefix) - 2)
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", content):
        paragraph = paragraph.strip()
        if paragraph:
            paragraphs.extend(_split_long_paragraph(paragraph, body_limit))

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if current else 0)
        if current and current_chars + addition > body_limit:
            chunks.append("\n\n".join(current))
            current = []
            current_chars = 0
        current.append(paragraph)
        current_chars += len(paragraph) + (2 if len(current) > 1 else 0)
    if current:
        chunks.append("\n\n".join(current))
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    rendered = [f"{prefix}\n\n{chunk}".strip() if prefix else chunk for chunk in chunks]
    return [chunk for chunk in rendered if len(chunk) >= min_chars][:max_chunks]


def _record_key(record: dict[str, Any], seed: int, namespace: str) -> bytes:
    identity = f"{record.get('document_id')}:{record.get('chunk', 0)}"
    return hashlib.sha256(f"{seed}:{namespace}:{identity}".encode()).digest()


def _ordered_sample(
    records: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: _record_key(record, seed, namespace))[
        :count
    ]


def mix_replay(
    stage_name: str,
    current: list[dict[str, Any]],
    prior: dict[str, list[dict[str, Any]]],
    replay: dict[str, float],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if not replay:
        return _ordered_sample(current, len(current), seed=seed, namespace=stage_name)
    current_fraction = 1.0 - sum(float(value) for value in replay.values())
    target_total = math.ceil(len(current) / current_fraction)
    mixed = list(current)
    for source, fraction in replay.items():
        requested = round(target_total * float(fraction))
        mixed.extend(
            _ordered_sample(
                prior[source],
                min(requested, len(prior[source])),
                seed=seed,
                namespace=f"{stage_name}:replay:{source}",
            )
        )
    return _ordered_sample(mixed, len(mixed), seed=seed, namespace=f"{stage_name}:mix")


def _causal_record(
    selected: SelectedDocument, chunk: str, index: int
) -> dict[str, Any]:
    return {
        "text": chunk,
        "document_id": selected.document.document_id,
        "chunk": index,
        "curriculum_stage": selected.stage,
        "language": selected.language,
    }


def _primary_match_rank(selected: SelectedDocument, search: str) -> int:
    ranks = [
        match.rank for match in selected.candidate.matches if match.search == search
    ]
    return min(ranks) if ranks else MAX_SEARCH_WINDOW


def _retrieval_records(
    selected: list[SelectedDocument],
    chunks: dict[str, list[str]],
    *,
    negative_count: int,
) -> list[dict[str, Any]]:
    by_search: dict[str, list[SelectedDocument]] = defaultdict(list)
    for item in selected:
        for search in {match.search for match in item.candidate.matches}:
            by_search[search].append(item)
    for search, items in by_search.items():
        items.sort(
            key=lambda item: (
                _primary_match_rank(item, search),
                item.document.document_id,
            )
        )

    records: list[dict[str, Any]] = []
    for item in selected:
        own_chunks = chunks.get(item.document.document_id, [])
        if not own_chunks:
            continue
        title = item.document.title
        fallback_query = min(
            item.candidate.matches, key=lambda match: match.rank
        ).retrieval_query
        query = title if len(title.split()) >= 2 else fallback_query
        negatives: list[str] = []
        negative_ids: list[str] = []
        for match in sorted(item.candidate.matches, key=lambda value: value.rank):
            for other in by_search[match.search]:
                if other.document.document_id == item.document.document_id:
                    continue
                if other.document.title.casefold() == title.casefold():
                    continue
                other_chunks = chunks.get(other.document.document_id, [])
                if not other_chunks or other.document.document_id in negative_ids:
                    continue
                negatives.append(other_chunks[0])
                negative_ids.append(other.document.document_id)
                if len(negatives) >= negative_count:
                    break
            if len(negatives) >= negative_count:
                break
        record: dict[str, Any] = {
            "query": query,
            "positive": own_chunks[0],
            "document_id": item.document.document_id,
            "curriculum_stage": item.stage,
            "language": item.language,
            "discovery_searches": sorted(
                {match.search for match in item.candidate.matches}
            ),
        }
        if negatives:
            record["negatives"] = negatives
            record["negative_document_ids"] = negative_ids
        records.append(record)
    return records


def build_record_pools(
    config: dict[str, Any],
    selected: dict[str, list[SelectedDocument]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    output = config.get("output", {})
    max_chars = int(output.get("max_chunk_chars", 24_000))
    min_chars = int(output.get("min_chunk_chars", 300))
    max_chunks = int(output.get("max_chunks_per_document", 256))
    negative_count = int(output.get("hard_negatives", 2))
    train_causal: dict[str, list[dict[str, Any]]] = {}
    eval_causal: dict[str, list[dict[str, Any]]] = {}
    train_retrieval: dict[str, list[dict[str, Any]]] = {}
    eval_retrieval: dict[str, list[dict[str, Any]]] = {}

    for stage in config["stages"]:
        name = stage["name"]
        train_items = [item for item in selected[name] if not item.validation]
        eval_items = [item for item in selected[name] if item.validation]
        split_outputs = []
        for items in (train_items, eval_items):
            chunks: dict[str, list[str]] = {}
            causal: list[dict[str, Any]] = []
            kept_items: list[SelectedDocument] = []
            for item in items:
                document_chunks = chunk_document(
                    item,
                    max_chars=max_chars,
                    min_chars=min_chars,
                    max_chunks=max_chunks,
                )
                if not document_chunks:
                    continue
                chunks[item.document.document_id] = document_chunks
                kept_items.append(item)
                causal.extend(
                    _causal_record(item, chunk, index)
                    for index, chunk in enumerate(document_chunks)
                )
            retrieval = _retrieval_records(
                kept_items,
                chunks,
                negative_count=negative_count,
            )
            split_outputs.append((causal, retrieval))
        (
            (train_causal[name], train_retrieval[name]),
            (
                eval_causal[name],
                eval_retrieval[name],
            ),
        ) = split_outputs
    return train_causal, eval_causal, train_retrieval, eval_retrieval


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _open_jsonl(path: Path, compression: str) -> io.TextIOBase:
    if compression == "none":
        return path.open("w", encoding="utf-8", newline="\n")
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("zstd output requires `pip install zstandard`") from error
    raw = path.open("wb")
    compressed = zstandard.ZstdCompressor(level=3).stream_writer(raw, closefd=True)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")


def _write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
    compression: str,
) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp")
    with _open_jsonl(temporary, compression) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    os.replace(temporary, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.name,
        "records": len(records),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _curriculum_stage(
    name: str,
    data: str,
    objective: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "name": name,
        "data": data,
        "objective": objective,
        "sequence_length": geometry["sequence_length"],
        "batch_size": geometry["batch_size"],
        "gradient_accumulation": geometry["gradient_accumulation"],
        "epochs": geometry.get("epochs", 1),
        "shuffle_buffer": geometry.get("shuffle_buffer", 65_536),
        "learning_rate_scale": geometry.get("learning_rate_scale", 1.0),
    }
    if "steps" in geometry:
        result["steps"] = geometry["steps"]
    return result


def write_outputs(
    config: dict[str, Any],
    output_dir: Path,
    discovery: Discovery,
    selected: dict[str, list[SelectedDocument]],
    rejection_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Write cumulative training stages, held-out tiers, curriculum, and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = config.get("output", {})
    compression = output.get("compression", "zstd")
    suffix = ".jsonl.zst" if compression == "zstd" else ".jsonl"
    seed = int(output.get("seed", 17))
    train_causal, eval_causal, train_retrieval, eval_retrieval = build_record_pools(
        config,
        selected,
    )
    files: list[dict[str, Any]] = []
    curriculum_stages: list[dict[str, Any]] = []
    prior_causal: dict[str, list[dict[str, Any]]] = {}
    prior_retrieval: dict[str, list[dict[str, Any]]] = {}

    for index, stage in enumerate(config["stages"], start=1):
        name = stage["name"]
        slug = f"{index:02d}-{_slug(name)}"
        replay = {key: float(value) for key, value in stage.get("replay", {}).items()}
        causal_records = mix_replay(
            name,
            train_causal[name],
            prior_causal,
            replay,
            seed=seed,
        )
        retrieval_records = mix_replay(
            name,
            train_retrieval[name],
            prior_retrieval,
            replay,
            seed=seed,
        )
        _require(causal_records, f"stage {name!r} produced no causal records")
        _require(
            len(retrieval_records) >= 2,
            f"stage {name!r} produced fewer than two retrieval records",
        )
        causal_path = output_dir / f"{slug}-causal{suffix}"
        retrieval_path = output_dir / f"{slug}-retrieval{suffix}"
        eval_causal_path = output_dir / f"{slug}-eval-causal{suffix}"
        eval_retrieval_path = output_dir / f"{slug}-eval-retrieval{suffix}"
        files.extend(
            [
                _write_jsonl(causal_path, causal_records, compression),
                _write_jsonl(retrieval_path, retrieval_records, compression),
                _write_jsonl(eval_causal_path, eval_causal[name], compression),
                _write_jsonl(eval_retrieval_path, eval_retrieval[name], compression),
            ]
        )
        training = stage["training"]
        curriculum_stages.append(
            _curriculum_stage(
                f"{name}-causal",
                causal_path.name,
                {"type": "causal_lm"},
                training["causal"],
            )
        )
        curriculum_stages.append(
            _curriculum_stage(
                f"{name}-retrieval",
                retrieval_path.name,
                {
                    "type": "contrastive_retrieval",
                    "temperature": training["retrieval"].get("temperature", 0.05),
                    "layer": training["retrieval"].get("layer", 24),
                },
                training["retrieval"],
            )
        )
        prior_causal[name] = train_causal[name]
        prior_retrieval[name] = train_retrieval[name]

    curriculum = {"version": 1, "stages": curriculum_stages}
    curriculum_path = output_dir / "curriculum.json"
    temporary = output_dir / ".curriculum.json.tmp"
    temporary.write_text(
        json.dumps(curriculum, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, curriculum_path)

    stage_stats = {}
    for stage in config["stages"]:
        name = stage["name"]
        stage_stats[name] = {
            "selected_documents": len(selected[name]),
            "training_documents": sum(not item.validation for item in selected[name]),
            "validation_documents": sum(item.validation for item in selected[name]),
            "rejections": rejection_stats[name],
        }
    manifest = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "contract": "Search API discovery IDs -> AlloyDB documents_assembled full copies",
        "search_index": discovery.index,
        "searches": discovery.searches,
        "stages": stage_stats,
        "files": files,
        "curriculum": curriculum_path.name,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


async def build_live(
    client: Any,
    pool: Any,
    config: dict[str, Any],
    output_dir: Path,
    *,
    search_limit_override: int | None = None,
) -> dict[str, Any]:
    discovery = await discover_with_search_api(
        client,
        config,
        search_limit_override=search_limit_override,
    )
    ids = sorted(
        {
            document_id
            for stage_candidates in discovery.candidates.values()
            for document_id in stage_candidates
        }
    )
    documents = await resolve_from_alloydb(
        pool,
        ids,
        batch_size=config.get("alloydb", {}).get("batch_size", 500),
    )
    selected, rejection_stats = select_documents(config, discovery, documents)
    return write_outputs(config, output_dir, discovery, selected, rejection_stats)
