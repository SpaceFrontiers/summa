"""Export completed benchmark evidence without machine names or private paths."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def metrics(path):
    summary = read(path)
    raw = read(path.with_suffix(".raw.json"))
    assert read(path.parent / "complete.json") == {"complete": True}
    assert not summary["errors"] and not raw["errors"]
    assert summary["memory"]["error"] is None
    reps = raw["repetitions"]
    assert len(reps) == 3 and all(rep["errors"] == 0 for rep in reps)
    requests = sum(rep["requests"] for rep in reps)
    elapsed = sum(rep["elapsed_s"] for rep in reps)
    qps = [rep["requests"] / rep["elapsed_s"] for rep in reps]
    return {
        "family": summary["family"],
        "operation": summary["operation"],
        "query_count": summary["query_count"],
        "median_qps": statistics.median(qps),
        "min_qps": min(qps),
        "max_qps": max(qps),
        "server_cpu_us_per_request": 1e6
        * sum(r["server_cpu_s"] for r in reps)
        / requests,
        "driver_cpu_us_per_request": 1e6
        * sum(r["client_cpu_s"] for r in reps)
        / requests,
        "mean_server_cores": sum(r["server_cpu_s"] for r in reps) / elapsed,
        "mean_driver_cores": sum(r["client_cpu_s"] for r in reps) / elapsed,
        "peak_rss_mib": summary["memory"]["peak_vmrss_kb"] / 1024,
        "peak_anonymous_rss_mib": summary["memory"]["peak_rssanon_kb"] / 1024,
        "requests": requests,
        "repetitions": len(reps),
        "errors": 0,
    }


def matrix(folder):
    runs = []
    for run in sorted(folder.iterdir()):
        # Profile-artifact transfer at 03:56 UTC overlapped this round.
        if folder.name == "close-gap-inline-matrix" and run.name == "inline-unicode":
            continue
        if not run.is_dir() or not (run / "completed.json").exists():
            continue
        assert read(run / "completed.json") == {"complete": True}
        engine = "luxir" if run.name.startswith("luxir") else "summa"
        selected = [row for row in read(run / "agreement.json") if row["include"]]
        query_keys = sorted(row["key"] for row in selected)
        entry = {
            "label": run.name,
            "engine": engine,
            "query_count": len(selected),
            "query_set_sha256": hashlib.sha256(
                "\n".join(query_keys).encode()
            ).hexdigest(),
            "cells": [
                metrics(path) for path in sorted((run / engine).glob("*-c32.json"))
            ],
        }
        if engine == "summa":
            entry["binary_sha256"] = read(run / "provenance.json")["binary_sha256"]
            audits = [
                json.loads(line)
                for line in (run / "audit.jsonl").read_text().splitlines()
            ]
            assert len(audits) == len(selected)
            assert all(row["optimized"] == row["exhaustive"] for row in audits)
            responses = read(run / "responses.json")
            assert set(responses) == set(query_keys)
            for row in audits:
                response = responses[row["class"] + "\t" + row["query"]]
                assert response["0"]["found"] == row["exhaustive"]
                for limit in [10, 100]:
                    assert [d["id"] for d in response[str(limit)]["docs"]] == [
                        d["id"] for d in row["ranked"][:limit]
                    ]
            entry["audited_queries"] = len(audits)
        runs.append(entry)
    return runs


def coverage(artifacts, out):
    previous = read(out / "tokenizer-counts.json")
    probes = {
        row["class"] + "\t" + row["query"]: row
        for row in map(
            json.loads,
            (artifacts / "close-gap-unicode-probe-ranges/counts.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    for row in previous["queries"]:
        probe = probes[row["family"] + "\t" + row["query"]]
        found = probe["response"].get("found") if probe["status"] == 200 else None
        row["unicode_word"] = found
        row["error"] = None if found is not None else probe["response"]
        row["exact"] = found is not None and found == row["reference"]
        row["within_5_percent"] = found is not None and abs(
            found - row["reference"]
        ) <= 0.05 * max(1, row["reference"])
    previous["summary"] = {
        "successful": sum(r["unicode_word"] is not None for r in previous["queries"]),
        "exact": sum(r["exact"] for r in previous["queries"]),
        "within_5_percent": sum(r["within_5_percent"] for r in previous["queries"]),
    }
    write(out / "coverage.json", previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    for name in [
        "matrix",
        "regex-matrix",
        "union-matrix",
        "final-matrix",
        "norm-matrix",
        "norm-matrix-valid",
        "workers-matrix",
        "inline-matrix",
        "deferred-matrix",
        "complete-matrix",
    ]:
        folder = args.artifacts / ("close-gap-" + name)
        if folder.exists():
            write(args.out / (name + ".json"), matrix(folder))
    coverage(args.artifacts, args.out)


if __name__ == "__main__":
    main()
