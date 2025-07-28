"""Bounded-memory writer for multi-billion-token education curricula.

Discovery metadata stays in memory, but canonical document bodies never do.
AlloyDB rows are validated, tokenized, and written in bounded batches.  A small
SQLite catalog retains only selection metadata and the first chunk needed for
hard-negative construction.  Completed tiers are restartable; an interrupted
tier is discarded and rebuilt atomically from the ID-only discovery cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from education_curriculum import (
    Candidate,
    Discovery,
    SelectedDocument,
    _content_fingerprint,
    _curriculum_stage,
    _eligible,
    _first_language,
    _open_jsonl,
    _sanitize_training_text,
    _slug,
    _validation_member,
    chunk_document,
    discover_with_search_api,
    resolve_from_alloydb,
)


class TokenCounter(Protocol):
    """Exact tokenizer used to size causal data before training."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def count_batch(self, texts: list[str]) -> list[int]: ...


class GigaTokenCounter:
    """Exact, batched token counts from a local Hermes ``tokenizer.json``."""

    def __init__(self, tokenizer_path: Path):
        try:
            import awkward as ak  # type: ignore[import-not-found]
            import gigatoken as gt  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "exact corpus sizing requires `pip install gigatoken==0.10.0`"
            ) from error
        self._ak = ak
        self._tokenizer = gt.Tokenizer(tokenizer_path)
        self._metadata = {
            "engine": "gigatoken",
            "version": importlib.metadata.version("gigatoken"),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": _sha256_file(tokenizer_path),
            "vocab_size": int(self._tokenizer.vocab_size),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def count_batch(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encoded = self._tokenizer.encode_batch(texts)
        lengths = self._ak.to_list(self._ak.num(encoded, axis=1))
        return [int(length) for length in lengths]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_characters(record: dict[str, Any]) -> int:
    total = 0
    for key in (
        "text",
        "document",
        "summary",
        "request",
        "plan",
        "context",
        "query",
        "positive",
    ):
        value = record.get(key)
        if isinstance(value, str):
            total += len(value)
    negatives = record.get("negatives")
    if isinstance(negatives, list):
        total += sum(len(value) for value in negatives if isinstance(value, str))
    return total


class JsonlWriter:
    """Atomic JSONL writer with manifest and exact-token accounting."""

    def __init__(self, path: Path, compression: str):
        self.path = path
        self.compression = compression
        self.temporary = path.with_name(f".{path.name}.tmp")
        self.records = 0
        self.uncompressed_bytes = 0
        self.payload_characters = 0
        self.content_tokens = 0
        self._stream: io.TextIOBase | None = None

    def __enter__(self) -> JsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self._stream = _open_jsonl(self.temporary, self.compression)
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("JSONL writer is not open")
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        line = f"{serialized}\n"
        self._stream.write(line)
        self.records += 1
        self.uncompressed_bytes += len(line.encode("utf-8"))
        self.payload_characters += _payload_characters(record)
        token_count = record.get("token_count", 0)
        if isinstance(token_count, int) and token_count >= 0:
            self.content_tokens += token_count

    def __exit__(self, exc_type, exc, traceback) -> bool:
        assert self._stream is not None
        try:
            self._stream.close()
        finally:
            self._stream = None
        if exc_type is None:
            os.replace(self.temporary, self.path)
        else:
            self.temporary.unlink(missing_ok=True)
        return False

    def stats(self, *, manifest_path: str | None = None) -> dict[str, Any]:
        result = {
            "path": manifest_path or self.path.name,
            "records": self.records,
            "bytes": self.path.stat().st_size,
            "uncompressed_bytes": self.uncompressed_bytes,
            "payload_characters": self.payload_characters,
            "sha256": _sha256_file(self.path),
        }
        if self.content_tokens:
            result["content_tokens"] = self.content_tokens
            result["trainer_tokens_with_eos"] = self.content_tokens + self.records
        return result


def _open_records(path: Path) -> Iterator[dict[str, Any]]:
    raw = path.open("rb")
    try:
        if path.suffix == ".zst":
            try:
                import zstandard  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError(
                    "zstd input requires `pip install zstandard`"
                ) from error
            binary = zstandard.ZstdDecompressor().stream_reader(raw)
        else:
            binary = raw
        with io.TextIOWrapper(binary, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSON at {path}:{line_number}")
                yield value
    finally:
        if not raw.closed:
            raw.close()


def _sampled_records(
    path: Path,
    source_count: int,
    requested: int,
    *,
    seed: int,
    namespace: str,
) -> Iterator[dict[str, Any]]:
    """Select exactly ``requested`` records with a constant-memory permutation."""
    requested = min(requested, source_count)
    if requested <= 0:
        return
    if requested == source_count:
        yield from _open_records(path)
        return
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    multiplier = int.from_bytes(digest[:8], "big") % source_count
    while math.gcd(multiplier, source_count) != 1:
        multiplier = (multiplier + 1) % source_count
    offset = int.from_bytes(digest[8:16], "big") % source_count
    emitted = 0
    seen = 0
    for index, record in enumerate(_open_records(path)):
        seen = index + 1
        if (multiplier * index + offset) % source_count < requested:
            emitted += 1
            yield record
    if emitted != requested:
        raise RuntimeError(
            f"replay source {path} contained {seen} records; expected {source_count}"
        )


def _state_connection(
    path: Path, config_hash: str, tokenizer_hash: str
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS selected (
            document_id TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            validation INTEGER NOT NULL,
            language TEXT NOT NULL,
            document_type TEXT NOT NULL,
            title TEXT NOT NULL,
            first_chunk TEXT NOT NULL,
            query TEXT NOT NULL,
            searches_json TEXT NOT NULL,
            fingerprint TEXT NOT NULL UNIQUE,
            canonical_content_characters INTEGER NOT NULL,
            causal_records INTEGER NOT NULL,
            causal_tokens INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS selected_stage_split
            ON selected(stage, validation, document_id);
        CREATE TABLE IF NOT EXISTS matches (
            stage TEXT NOT NULL,
            search TEXT NOT NULL,
            document_id TEXT NOT NULL REFERENCES selected(document_id) ON DELETE CASCADE,
            rank INTEGER NOT NULL,
            PRIMARY KEY(stage, search, document_id)
        );
        CREATE INDEX IF NOT EXISTS matches_negative_lookup
            ON matches(stage, search, rank, document_id);
        CREATE TABLE IF NOT EXISTS stage_stats (
            stage TEXT PRIMARY KEY,
            stage_index INTEGER NOT NULL,
            stats_json TEXT NOT NULL
        );
        """
    )
    expected = {
        "format": "1",
        "config_sha256": config_hash,
        "tokenizer_sha256": tokenizer_hash,
    }
    existing = {
        row["key"]: row["value"]
        for row in connection.execute("SELECT key, value FROM metadata")
    }
    if existing and any(existing.get(key) != value for key, value in expected.items()):
        connection.close()
        raise RuntimeError(
            "streaming build state belongs to a different config or tokenizer; use a new output directory"
        )
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        expected.items(),
    )
    connection.commit()
    return connection


def _suffix(compression: str) -> str:
    return ".jsonl.zst" if compression == "zstd" else ".jsonl"


def _raw_paths(
    build_dir: Path, stage_index: int, stage_name: str, compression: str
) -> dict[str, Path]:
    slug = f"{stage_index:02d}-{_slug(stage_name)}"
    suffix = _suffix(compression)
    raw = build_dir / "raw"
    return {
        "train_causal": raw / f"{slug}-causal{suffix}",
        "eval_causal": raw / f"{slug}-eval-causal{suffix}",
        "train_retrieval": raw / f"{slug}-retrieval{suffix}",
        "eval_retrieval": raw / f"{slug}-eval-retrieval{suffix}",
    }


def _completed_prefix(
    connection: sqlite3.Connection,
    config: dict[str, Any],
    build_dir: Path,
) -> dict[str, dict[str, Any]]:
    compression = config.get("output", {}).get("compression", "zstd")
    completed: dict[str, dict[str, Any]] = {}
    incomplete = False
    for index, stage in enumerate(config["stages"], start=1):
        name = stage["name"]
        row = connection.execute(
            "SELECT stats_json FROM stage_stats WHERE stage = ?", (name,)
        ).fetchone()
        paths = _raw_paths(build_dir, index, name, compression)
        if (
            not incomplete
            and row is not None
            and all(path.exists() for path in paths.values())
        ):
            completed[name] = json.loads(row["stats_json"])
            continue
        incomplete = True
        connection.execute("DELETE FROM selected WHERE stage = ?", (name,))
        connection.execute("DELETE FROM stage_stats WHERE stage = ?", (name,))
        for path in paths.values():
            path.unlink(missing_ok=True)
            path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
    connection.commit()
    return completed


@dataclass
class PendingDocument:
    selected: SelectedDocument
    chunks: list[str]
    fingerprint: str


def _match_map(candidate: Candidate) -> dict[str, tuple[int, str]]:
    matches: dict[str, tuple[int, str]] = {}
    for match in candidate.matches:
        current = matches.get(match.search)
        if current is None or match.rank < current[0]:
            matches[match.search] = (match.rank, match.retrieval_query)
    return matches


def _write_pending(
    pending: list[PendingDocument],
    counter: TokenCounter,
    train_writer: JsonlWriter,
    eval_writer: JsonlWriter,
    connection: sqlite3.Connection,
) -> None:
    if not pending:
        return
    texts = [chunk for item in pending for chunk in item.chunks]
    token_counts = counter.count_batch(texts)
    if len(token_counts) != len(texts):
        raise RuntimeError("token counter returned the wrong number of rows")
    token_offset = 0
    for pending_document in pending:
        item = pending_document.selected
        chunks = pending_document.chunks
        counts = token_counts[token_offset : token_offset + len(chunks)]
        token_offset += len(chunks)
        writer = eval_writer if item.validation else train_writer
        for chunk_index, (chunk, token_count) in enumerate(
            zip(chunks, counts, strict=True)
        ):
            writer.write(
                {
                    "text": chunk,
                    "document_id": item.document.document_id,
                    "chunk": chunk_index,
                    "curriculum_stage": item.stage,
                    "language": item.language,
                    "token_count": token_count,
                }
            )
        matches = _match_map(item.candidate)
        _primary_search, (_primary_rank, primary_query) = min(
            matches.items(), key=lambda value: (value[1][0], value[0])
        )
        title = _sanitize_training_text(item.document.title)
        query = title if len(title.split()) >= 2 else primary_query
        connection.execute(
            """
            INSERT INTO selected(
                document_id, stage, validation, language, document_type,
                title, first_chunk, query, searches_json, fingerprint,
                canonical_content_characters, causal_records, causal_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.document.document_id,
                item.stage,
                int(item.validation),
                item.language,
                item.document.document_type,
                title,
                chunks[0],
                query,
                json.dumps(sorted(matches), ensure_ascii=False),
                pending_document.fingerprint,
                len(item.document.content),
                len(chunks),
                sum(counts),
            ),
        )
        connection.executemany(
            """
            INSERT INTO matches(stage, search, document_id, rank)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(stage, search, document_id)
            DO UPDATE SET rank = MIN(rank, excluded.rank)
            """,
            (
                (item.stage, search, item.document.document_id, rank)
                for search, (rank, _query) in matches.items()
            ),
        )
    connection.commit()


def _negative_rows(
    connection: sqlite3.Connection,
    *,
    stage: str,
    validation: bool,
    document_id: str,
    searches: list[str],
    count: int,
) -> list[sqlite3.Row]:
    negatives: list[sqlite3.Row] = []
    seen = {document_id}
    for search in searches:
        rows = connection.execute(
            """
            SELECT s.document_id, s.first_chunk
            FROM matches AS m
            JOIN selected AS s ON s.document_id = m.document_id
            WHERE m.stage = ? AND m.search = ? AND s.validation = ?
              AND s.document_id <> ?
            ORDER BY m.rank, s.document_id
            LIMIT ?
            """,
            (stage, search, int(validation), document_id, max(count * 4, 8)),
        )
        for row in rows:
            if row["document_id"] in seen:
                continue
            seen.add(row["document_id"])
            negatives.append(row)
            if len(negatives) >= count:
                return negatives
    if len(negatives) < count:
        rows = connection.execute(
            """
            SELECT document_id, first_chunk
            FROM selected
            WHERE stage = ? AND validation = ? AND document_id <> ?
            ORDER BY document_id
            LIMIT ?
            """,
            (stage, int(validation), document_id, max(count * 4, 8)),
        )
        for row in rows:
            if row["document_id"] in seen:
                continue
            seen.add(row["document_id"])
            negatives.append(row)
            if len(negatives) >= count:
                break
    return negatives


def _write_retrieval_split(
    connection: sqlite3.Connection,
    path: Path,
    compression: str,
    *,
    stage: str,
    validation: bool,
    negative_count: int,
) -> dict[str, Any]:
    with JsonlWriter(path, compression) as writer:
        rows = connection.execute(
            """
            SELECT document_id, language, query, first_chunk, searches_json
            FROM selected
            WHERE stage = ? AND validation = ?
            ORDER BY document_id
            """,
            (stage, int(validation)),
        )
        for row in rows:
            searches = json.loads(row["searches_json"])
            negatives = _negative_rows(
                connection,
                stage=stage,
                validation=validation,
                document_id=row["document_id"],
                searches=searches,
                count=negative_count,
            )
            record: dict[str, Any] = {
                "query": row["query"],
                "positive": row["first_chunk"],
                "document_id": row["document_id"],
                "curriculum_stage": stage,
                "language": row["language"],
                "discovery_searches": searches,
            }
            if negatives:
                record["negatives"] = [row["first_chunk"] for row in negatives]
                record["negative_document_ids"] = [
                    row["document_id"] for row in negatives
                ]
            writer.write(record)
    return writer.stats(manifest_path=path.name)


async def _select_stage(
    pool: Any,
    config: dict[str, Any],
    stage: dict[str, Any],
    stage_index: int,
    discovery: Discovery,
    connection: sqlite3.Connection,
    counter: TokenCounter,
    build_dir: Path,
    assigned_ids: set[str],
    fingerprints: set[str],
    target_train_tokens: int | None = None,
) -> dict[str, Any]:
    output = config.get("output", {})
    compression = output.get("compression", "zstd")
    paths = _raw_paths(build_dir, stage_index, stage["name"], compression)
    rejection_stats: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    document_types: Counter[str] = Counter()
    accepted = 0
    train_documents = 0
    validation_documents = 0
    canonical_content_characters = 0
    candidates = sorted(
        discovery.candidates[stage["name"]].values(),
        key=lambda candidate: (
            -candidate.best_score,
            candidate.best_rank,
            candidate.document_id,
        ),
    )
    maximum = int(stage.get("max_documents", len(candidates)))
    minimum = int(stage.get("min_documents", 2))
    per_language = stage.get("max_documents_per_language")
    alloydb = config.get("alloydb", {})
    batch_size = int(alloydb.get("batch_size", 128))
    prefetch_batches = int(
        alloydb.get("prefetch_batches", alloydb.get("connections", 1))
    )
    token_batch_characters = int(output.get("token_batch_characters", 16_000_000))
    pending: list[PendingDocument] = []
    pending_characters = 0
    available_candidates: list[Candidate] = []
    for candidate in candidates:
        if candidate.document_id in assigned_ids:
            rejection_stats["assigned_to_earlier_stage"] += 1
        else:
            available_candidates.append(candidate)
    processed_candidates = 0

    with (
        JsonlWriter(paths["train_causal"], compression) as train_writer,
        JsonlWriter(paths["eval_causal"], compression) as eval_writer,
    ):
        window_size = batch_size * prefetch_batches
        stop_reason: str | None = None
        for window_offset in range(0, len(available_candidates), window_size):
            window = available_candidates[window_offset : window_offset + window_size]
            batches = [
                window[offset : offset + batch_size]
                for offset in range(0, len(window), batch_size)
            ]
            resolved_batches = await asyncio.gather(
                *(
                    resolve_from_alloydb(
                        pool,
                        [candidate.document_id for candidate in batch],
                        batch_size=batch_size,
                    )
                    for batch in batches
                )
            )
            for batch, documents in zip(batches, resolved_batches, strict=True):
                if accepted >= maximum:
                    stop_reason = "stage_quota"
                    break
                if (
                    target_train_tokens is not None
                    and accepted >= minimum
                    and train_writer.content_tokens >= target_train_tokens
                ):
                    stop_reason = "token_target"
                    break
                for candidate in batch:
                    if accepted >= maximum:
                        rejection_stats["stage_quota"] += 1
                        continue
                    document = documents.get(candidate.document_id)
                    if document is None:
                        rejection_stats["missing_in_alloydb"] += 1
                        continue
                    reason = _eligible(
                        stage, candidate, document, config.get("quality", {})
                    )
                    if reason is not None:
                        rejection_stats[reason] += 1
                        continue
                    language = _first_language(document, candidate, stage)
                    if (
                        per_language is not None
                        and language_counts[language] >= per_language
                    ):
                        rejection_stats["language_quota"] += 1
                        continue
                    fingerprint = _content_fingerprint(document)
                    if fingerprint in fingerprints:
                        rejection_stats["duplicate_content"] += 1
                        continue
                    selected = SelectedDocument(
                        stage=stage["name"],
                        candidate=candidate,
                        document=document,
                        language=language,
                        validation=_validation_member(
                            candidate.document_id,
                            int(output.get("seed", 17)),
                            float(output.get("validation_fraction", 0.01)),
                        ),
                    )
                    chunks = chunk_document(
                        selected,
                        max_chars=int(output.get("max_chunk_chars", 24_000)),
                        min_chars=int(output.get("min_chunk_chars", 300)),
                        max_chunks=int(output.get("max_chunks_per_document", 256)),
                    )
                    if not chunks:
                        rejection_stats["no_training_chunks"] += 1
                        continue
                    assigned_ids.add(candidate.document_id)
                    fingerprints.add(fingerprint)
                    language_counts[language] += 1
                    document_types[document.document_type] += 1
                    accepted += 1
                    if selected.validation:
                        validation_documents += 1
                    else:
                        train_documents += 1
                    canonical_content_characters += len(document.content)
                    pending.append(PendingDocument(selected, chunks, fingerprint))
                    pending_characters += sum(map(len, chunks))
                    if pending_characters >= token_batch_characters:
                        _write_pending(
                            pending, counter, train_writer, eval_writer, connection
                        )
                        pending.clear()
                        pending_characters = 0
                processed_candidates += len(batch)
                logging.info(
                    "resolved stage=%s candidates=%d/%d accepted=%d",
                    stage["name"],
                    len(candidates) - len(available_candidates) + processed_candidates,
                    len(candidates),
                    accepted,
                )
            if stop_reason is not None:
                rejection_stats[stop_reason] += (
                    len(available_candidates) - processed_candidates
                )
                break
        _write_pending(pending, counter, train_writer, eval_writer, connection)

    if accepted < minimum:
        raise ValueError(
            f"stage {stage['name']!r} selected {accepted} documents, below min_documents {minimum}"
        )
    train_causal_stats = train_writer.stats(manifest_path=paths["train_causal"].name)
    eval_causal_stats = eval_writer.stats(manifest_path=paths["eval_causal"].name)
    train_retrieval_stats = _write_retrieval_split(
        connection,
        paths["train_retrieval"],
        compression,
        stage=stage["name"],
        validation=False,
        negative_count=int(output.get("hard_negatives", 2)),
    )
    eval_retrieval_stats = _write_retrieval_split(
        connection,
        paths["eval_retrieval"],
        compression,
        stage=stage["name"],
        validation=True,
        negative_count=int(output.get("hard_negatives", 2)),
    )
    if train_retrieval_stats["records"] < 2:
        raise ValueError(
            f"stage {stage['name']!r} produced fewer than two retrieval records"
        )
    stats = {
        "selected_documents": accepted,
        "training_documents": train_documents,
        "validation_documents": validation_documents,
        "languages": dict(sorted(language_counts.items())),
        "document_types": dict(sorted(document_types.items())),
        "canonical_content_characters": canonical_content_characters,
        "unique_train_causal_tokens": train_causal_stats.get("content_tokens", 0),
        "rejections": dict(sorted(rejection_stats.items())),
        "raw_files": {
            "train_causal": train_causal_stats,
            "eval_causal": eval_causal_stats,
            "train_retrieval": train_retrieval_stats,
            "eval_retrieval": eval_retrieval_stats,
        },
    }
    connection.execute(
        "INSERT OR REPLACE INTO stage_stats(stage, stage_index, stats_json) VALUES (?, ?, ?)",
        (stage["name"], stage_index, json.dumps(stats, ensure_ascii=False)),
    )
    connection.commit()
    logging.info(
        "selected stage=%s accepted=%d candidates=%d causal_tokens=%d rejected=%s",
        stage["name"],
        accepted,
        len(candidates),
        stats["unique_train_causal_tokens"],
        json.dumps(stats["rejections"], sort_keys=True),
    )
    return stats


def _finalize_training_file(
    output_path: Path,
    compression: str,
    current_path: Path,
    current_count: int,
    prior: dict[str, tuple[Path, int]],
    replay: dict[str, float],
    *,
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    with JsonlWriter(output_path, compression) as writer:
        for record in _open_records(current_path):
            writer.write(record)
        if replay:
            current_fraction = 1.0 - sum(replay.values())
            target_total = math.ceil(current_count / current_fraction)
            for source, fraction in replay.items():
                source_path, source_count = prior[source]
                requested = min(round(target_total * fraction), source_count)
                for record in _sampled_records(
                    source_path,
                    source_count,
                    requested,
                    seed=seed,
                    namespace=f"{namespace}:{source}",
                ):
                    writer.write(record)
    return writer.stats(manifest_path=output_path.name)


def _copy_file_with_stats(
    source: Path,
    destination: Path,
    raw_stats: dict[str, Any],
) -> dict[str, Any]:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    stats = dict(raw_stats)
    stats["path"] = destination.name
    stats["bytes"] = destination.stat().st_size
    stats["sha256"] = _sha256_file(destination)
    return stats


def _write_curriculum_and_manifest(
    config: dict[str, Any],
    output_dir: Path,
    build_dir: Path,
    discovery: Discovery,
    stage_stats: dict[str, dict[str, Any]],
    counter: TokenCounter,
    config_hash: str,
    *,
    enforce_minimum: bool,
) -> dict[str, Any]:
    output = config.get("output", {})
    compression = output.get("compression", "zstd")
    suffix = _suffix(compression)
    seed = int(output.get("seed", 17))
    files: list[dict[str, Any]] = []
    curriculum_stages: list[dict[str, Any]] = []
    prior_causal: dict[str, tuple[Path, int]] = {}
    prior_retrieval: dict[str, tuple[Path, int]] = {}

    for index, stage in enumerate(config["stages"], start=1):
        name = stage["name"]
        slug = f"{index:02d}-{_slug(name)}"
        raw_paths = _raw_paths(build_dir, index, name, compression)
        replay = {key: float(value) for key, value in stage.get("replay", {}).items()}
        causal_path = output_dir / f"{slug}-causal{suffix}"
        retrieval_path = output_dir / f"{slug}-retrieval{suffix}"
        eval_causal_path = output_dir / f"{slug}-eval-causal{suffix}"
        eval_retrieval_path = output_dir / f"{slug}-eval-retrieval{suffix}"
        raw = stage_stats[name]["raw_files"]
        causal_stats = _finalize_training_file(
            causal_path,
            compression,
            raw_paths["train_causal"],
            int(raw["train_causal"]["records"]),
            prior_causal,
            replay,
            seed=seed,
            namespace=f"{name}:causal",
        )
        retrieval_stats = _finalize_training_file(
            retrieval_path,
            compression,
            raw_paths["train_retrieval"],
            int(raw["train_retrieval"]["records"]),
            prior_retrieval,
            replay,
            seed=seed,
            namespace=f"{name}:retrieval",
        )
        files.extend(
            [
                causal_stats,
                retrieval_stats,
                _copy_file_with_stats(
                    raw_paths["eval_causal"], eval_causal_path, raw["eval_causal"]
                ),
                _copy_file_with_stats(
                    raw_paths["eval_retrieval"],
                    eval_retrieval_path,
                    raw["eval_retrieval"],
                ),
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
        prior_causal[name] = (
            raw_paths["train_causal"],
            int(raw["train_causal"]["records"]),
        )
        prior_retrieval[name] = (
            raw_paths["train_retrieval"],
            int(raw["train_retrieval"]["records"]),
        )

    curriculum = {"version": 1, "stages": curriculum_stages}
    curriculum_path = output_dir / "curriculum.json"
    temporary = output_dir / ".curriculum.json.tmp"
    temporary.write_text(
        json.dumps(curriculum, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, curriculum_path)
    unique_tokens = sum(
        int(stats["unique_train_causal_tokens"]) for stats in stage_stats.values()
    )
    final_causal_files = [
        file
        for file in files
        if file["path"].endswith(f"-causal{suffix}") and "-eval-" not in file["path"]
    ]
    training_tokens = sum(
        int(file.get("content_tokens", 0)) for file in final_causal_files
    )
    training_tokens_with_eos = sum(
        int(file.get("trainer_tokens_with_eos", 0)) for file in final_causal_files
    )
    manifest = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "contract": "Search API discovery IDs -> AlloyDB documents_assembled full copies",
        "search_index": discovery.index,
        "searches": discovery.searches,
        "stages": stage_stats,
        "files": files,
        "curriculum": curriculum_path.name,
        "tokenizer": counter.metadata,
        "tokens": {
            "unique_train_causal": unique_tokens,
            "curriculum_train_causal": training_tokens,
            "curriculum_train_causal_with_eos": training_tokens_with_eos,
            "minimum_required": output.get("minimum_causal_tokens"),
            "minimum_enforced": enforce_minimum,
            "target": output.get("target_causal_tokens"),
        },
        "config": config,
        "config_sha256": config_hash,
    }
    manifest_path = output_dir / "manifest.json"
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    minimum = int(output.get("minimum_causal_tokens", 0))
    if enforce_minimum and unique_tokens < minimum:
        raise RuntimeError(
            f"corpus contains {unique_tokens:,} unique causal tokens, below the required {minimum:,}"
        )
    return manifest


async def build_live_streaming(
    client: Any,
    pool: Any,
    config: dict[str, Any],
    output_dir: Path,
    counter: TokenCounter,
    *,
    search_limit_override: int | None = None,
) -> dict[str, Any]:
    """Build a restartable corpus without retaining canonical bodies in memory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    build_dir = output_dir / ".build"
    connection = _state_connection(
        build_dir / "state.sqlite3",
        config_hash,
        str(counter.metadata["tokenizer_sha256"]),
    )
    try:
        completed = _completed_prefix(connection, config, build_dir)
        discovery = await discover_with_search_api(
            client,
            config,
            search_limit_override=search_limit_override,
        )
        assigned_ids = {
            row["document_id"]
            for row in connection.execute("SELECT document_id FROM selected")
        }
        fingerprints = {
            row["fingerprint"]
            for row in connection.execute("SELECT fingerprint FROM selected")
        }
        stage_stats = dict(completed)
        for index, stage in enumerate(config["stages"], start=1):
            if stage["name"] in completed:
                logging.info("resuming after completed stage=%s", stage["name"])
                continue
            target_train_tokens = None
            if index == len(config["stages"]):
                target = int(config.get("output", {}).get("target_causal_tokens", 0))
                if target:
                    prior_tokens = sum(
                        int(stats["unique_train_causal_tokens"])
                        for stats in stage_stats.values()
                    )
                    target_train_tokens = max(target - prior_tokens, 0)
            stage_stats[stage["name"]] = await _select_stage(
                pool,
                config,
                stage,
                index,
                discovery,
                connection,
                counter,
                build_dir,
                assigned_ids,
                fingerprints,
                target_train_tokens=target_train_tokens,
            )
        manifest = _write_curriculum_and_manifest(
            config,
            output_dir,
            build_dir,
            discovery,
            stage_stats,
            counter,
            config_hash,
            enforce_minimum=search_limit_override is None,
        )
    finally:
        connection.close()
    if config.get("output", {}).get("cleanup_build_artifacts", True):
        shutil.rmtree(build_dir)
    return manifest
