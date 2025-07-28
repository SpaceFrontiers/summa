#!/usr/bin/env python3
"""Verify checksums, splits, text hygiene, and exact tokens in a built corpus."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from curriculum_streaming import GigaTokenCounter, _payload_characters
from education_curriculum import CONTROL_CHARACTER_RE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="re-encode every causal record and verify its stored exact count",
    )
    parser.add_argument(
        "--token-batch-characters",
        type=int,
        default=16_000_000,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path: Path) -> Iterator[tuple[int, str, dict[str, Any]]]:
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
                yield line_number, line, value
    finally:
        if not raw.closed:
            raw.close()


def text_values(record: dict[str, Any]) -> Iterator[str]:
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
            yield value
    negatives = record.get("negatives")
    if isinstance(negatives, list):
        yield from (value for value in negatives if isinstance(value, str))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_token_batch(
    counter: GigaTokenCounter,
    path: Path,
    texts: list[str],
    expected: list[tuple[int, int]],
) -> None:
    actual = counter.count_batch(texts)
    for measured, (line_number, stored) in zip(actual, expected, strict=True):
        require(
            measured == stored,
            f"token mismatch at {path}:{line_number}: stored {stored}, measured {measured}",
        )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_config = manifest.get("config")
    require(isinstance(embedded_config, dict), "manifest is missing its build config")
    config_hash = hashlib.sha256(
        json.dumps(
            embedded_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    require(
        config_hash == manifest.get("config_sha256"),
        "embedded build config does not match config_sha256",
    )
    counter = GigaTokenCounter(args.tokenizer) if args.tokenizer else None
    if counter is not None:
        expected_tokenizer = manifest.get("tokenizer", {})
        measured_tokenizer = counter.metadata
        for key in ("engine", "version", "tokenizer_sha256", "vocab_size"):
            require(
                measured_tokenizer.get(key) == expected_tokenizer.get(key),
                f"audit tokenizer {key} does not match the build manifest",
            )
    train_documents: set[str] = set()
    eval_documents: set[str] = set()
    unique_causal: dict[tuple[str, int], int] = {}
    total_train_causal_tokens = 0
    total_train_causal_with_eos = 0
    total_records = 0
    file_results: list[dict[str, Any]] = []

    for expected in manifest["files"]:
        path = args.output / expected["path"]
        logging.info("auditing file=%s", path.name)
        require(path.exists(), f"missing manifest file {path}")
        require(
            path.stat().st_size == expected["bytes"], f"byte size mismatch for {path}"
        )
        require(
            sha256_file(path) == expected["sha256"], f"checksum mismatch for {path}"
        )
        is_causal = "-causal.jsonl" in path.name
        is_eval = "-eval-" in path.name
        record_count = 0
        uncompressed_bytes = 0
        payload_characters = 0
        content_tokens = 0
        pending_texts: list[str] = []
        pending_expected: list[tuple[int, int]] = []
        pending_characters = 0

        for line_number, line, record in records(path):
            record_count += 1
            if record_count % 100_000 == 0:
                logging.info(
                    "auditing file=%s records=%d",
                    path.name,
                    record_count,
                )
            uncompressed_bytes += len(line.encode("utf-8"))
            payload_characters += _payload_characters(record)
            document_id = record.get("document_id")
            require(
                isinstance(document_id, str) and document_id,
                f"missing document_id at {path}:{line_number}",
            )
            (eval_documents if is_eval else train_documents).add(document_id)
            for text in text_values(record):
                require(
                    "\ufffd" not in text,
                    f"replacement character at {path}:{line_number}",
                )
                require(
                    CONTROL_CHARACTER_RE.search(text) is None,
                    f"control character at {path}:{line_number}",
                )
            if is_causal:
                text = record.get("text")
                token_count = record.get("token_count")
                chunk = record.get("chunk")
                require(
                    isinstance(text, str) and text,
                    f"missing text at {path}:{line_number}",
                )
                require(
                    isinstance(token_count, int) and token_count > 0,
                    f"missing token_count at {path}:{line_number}",
                )
                require(
                    isinstance(chunk, int) and chunk >= 0,
                    f"missing chunk index at {path}:{line_number}",
                )
                content_tokens += token_count
                if not is_eval:
                    total_train_causal_tokens += token_count
                    total_train_causal_with_eos += token_count + 1
                    unique_causal.setdefault((document_id, chunk), token_count)
                if counter is not None:
                    pending_texts.append(text)
                    pending_expected.append((line_number, token_count))
                    pending_characters += len(text)
                    if pending_characters >= args.token_batch_characters:
                        verify_token_batch(
                            counter, path, pending_texts, pending_expected
                        )
                        pending_texts.clear()
                        pending_expected.clear()
                        pending_characters = 0
        if pending_texts and counter is not None:
            verify_token_batch(counter, path, pending_texts, pending_expected)
        require(
            record_count == expected["records"], f"record count mismatch for {path}"
        )
        require(
            uncompressed_bytes == expected["uncompressed_bytes"],
            f"uncompressed byte mismatch for {path}",
        )
        require(
            payload_characters == expected["payload_characters"],
            f"payload character mismatch for {path}",
        )
        if is_causal:
            require(
                content_tokens == expected.get("content_tokens", 0),
                f"content token mismatch for {path}",
            )
        total_records += record_count
        logging.info(
            "verified file=%s records=%d content_tokens=%d",
            path.name,
            record_count,
            content_tokens,
        )
        file_results.append(
            {
                "path": path.name,
                "records": record_count,
                "content_tokens": content_tokens if is_causal else None,
            }
        )

    overlap = train_documents & eval_documents
    require(not overlap, f"train/eval document overlap: {len(overlap)} IDs")
    unique_tokens = sum(unique_causal.values())
    tokens = manifest["tokens"]
    require(
        unique_tokens == tokens["unique_train_causal"],
        "unique causal token total does not match manifest",
    )
    require(
        total_train_causal_tokens == tokens["curriculum_train_causal"],
        "curriculum causal token total does not match manifest",
    )
    require(
        total_train_causal_with_eos == tokens["curriculum_train_causal_with_eos"],
        "curriculum causal-with-EOS total does not match manifest",
    )
    minimum = tokens.get("minimum_required")
    if tokens.get("minimum_enforced", True) and minimum is not None:
        require(unique_tokens >= minimum, "corpus is below its minimum token contract")
    return {
        "files": file_results,
        "records": total_records,
        "train_documents": len(train_documents),
        "eval_documents": len(eval_documents),
        "train_eval_overlap": 0,
        "unique_train_causal_tokens": unique_tokens,
        "curriculum_train_causal_tokens": total_train_causal_tokens,
        "curriculum_train_causal_tokens_with_eos": total_train_causal_with_eos,
        "tokenizer_recounted": counter is not None,
        "config_verified": True,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.token_batch_characters <= 0:
        raise SystemExit("--token-batch-characters must be positive")
    print(json.dumps(audit(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
