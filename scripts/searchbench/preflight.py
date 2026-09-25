#!/usr/bin/env python3
"""Audit a pinned Searchbench query file against Summa's native parser.

This is capability discovery, not a performance benchmark or count-agreement
check. Bare regex patterns need HTTP adapter translation into RegexQuery,
even when the native grammar accepts them as a different expression. The
existing benchmark binary owns native parsing.
"""

import argparse
import collections
import concurrent.futures
import json
import subprocess
import tempfile
from pathlib import Path

ADAPTER_TRANSLATED_CLASSES = frozenset({"regex"})


def probe(binary, row):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as query_file:
        query_file.write(json.dumps(row) + "\n")
        query_file.flush()
        result = subprocess.run(
            [str(binary), "validate-queries", query_file.name],
            capture_output=True,
            text=True,
            timeout=30,
        )
    parsed = result.returncode == 0
    query_class = row["class"]
    if query_class in ADAPTER_TRANSLATED_CLASSES:
        status = "requires_adapter_translation"
    elif not parsed:
        status = "native_syntax_rejected"
    else:
        status = "requires_count_agreement"
    return {
        "class": query_class,
        "query": row["query"],
        "native_parser_accepted": parsed,
        "status": status,
        "detail": result.stderr.splitlines()[-1] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    rows = [
        json.loads(line)
        for line in args.queries.read_text().splitlines()
        if line.strip()
    ]
    # Four independent short-lived parser probes; no engine benchmarking here.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda row: probe(binary, row), rows))
    classes = {}
    for result in results:
        counts = classes.setdefault(result["class"], collections.Counter())
        counts[result["status"]] += 1
        counts["total"] += 1
    report = {
        "scope": "Native syntax and adapter translation requirements only; no count or ranking equivalence established",
        "queries": str(args.queries),
        "classes": classes,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(classes, indent=2))


if __name__ == "__main__":
    main()
