"""Summarize completed raw runs; fail closed on missing/error evidence."""

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def metrics(paths):
    summaries = [read(path) for path in paths]
    assert all(
        read(path.parent / "complete.json") == {"complete": True} for path in paths
    )
    raw = [read(path.with_suffix(".raw.json")) for path in paths]
    reps = [rep for run in raw for rep in run["repetitions"]]
    assert all(not run["errors"] for run in summaries + raw)
    assert all(rep["errors"] == 0 for rep in reps)
    assert all(run["memory"]["error"] is None for run in summaries)
    qps = [rep["requests"] / rep["elapsed_s"] for rep in reps]
    requests = sum(rep["requests"] for rep in reps)
    return {
        "query_count": summaries[0]["query_count"],
        "repetitions": len(reps),
        "median_qps": statistics.median(qps),
        "min_qps": min(qps),
        "max_qps": max(qps),
        "round_median_qps": [run["median_qps"] for run in summaries],
        "server_cpu_ms_per_request": 1000
        * sum(rep["server_cpu_s"] for rep in reps)
        / requests,
        "driver_cpu_ms_per_request": 1000
        * sum(rep["client_cpu_s"] for rep in reps)
        / requests,
        "mean_driver_cores_used": sum(rep["client_cpu_s"] for rep in reps)
        / sum(rep["elapsed_s"] for rep in reps),
        "mean_server_cores_used": sum(rep["server_cpu_s"] for rep in reps)
        / sum(rep["elapsed_s"] for rep in reps),
        "peak_rss_mib": max(run["memory"]["peak_vmrss_kb"] for run in summaries) / 1024,
        "peak_anon_rss_mib": max(run["memory"]["peak_rssanon_kb"] for run in summaries)
        / 1024,
        "requests": requests,
        "errors": 0,
    }


def measured_paths(folder):
    return sorted(
        path
        for path in folder.glob("*-c32.json")
        if not path.name.endswith(".raw.json")
    )


def skip(results, out):
    variants = [
        "current",
        "no-pilot",
        "short-pilot",
        "bound-priority",
        "single-block",
        "no-rare-seek",
    ]
    rows = []
    audits = []
    for layout in ["plain", "rgb"]:
        baseline = {
            (r["class"], r["query"]): r
            for r in map(
                json.loads,
                (results / f"1-current-{layout}" / "audit.jsonl")
                .read_text()
                .splitlines(),
            )
        }
        assert len(baseline) == 344
        for variant in variants:
            first = results / f"1-{variant}-{layout}"
            second = results / f"2-{variant}-{layout}"
            audit = list(
                map(json.loads, (first / "audit.jsonl").read_text().splitlines())
            )
            expected = (
                344
                if variant == "current"
                else 147
                if variant == "no-rare-seek"
                else 197
            )
            assert len(audit) == expected
            for row in audit:
                assert row["optimized"] == row["exhaustive"]
                assert row == baseline[(row["class"], row["query"])]
            for folder in [first, second]:
                responses = read(folder / "responses.json")
                assert len(responses) == expected
                for row in audit:
                    response = responses[row["class"] + "\t" + row["query"]]
                    assert response["0"]["found"] == row["optimized"]
                    for limit in [10, 100]:
                        assert [doc["id"] for doc in response[str(limit)]["docs"]] == [
                            doc["id"] for doc in row["ranked"][:limit]
                        ]
            audits.append(
                {
                    "variant": variant,
                    "layout": layout,
                    "exhaustive_audits": expected,
                    "http_checks": expected * 3 * 2,
                }
            )
            paths = measured_paths(first / "summa")
            assert len(paths) == (
                27 if variant == "current" else 9 if variant == "no-rare-seek" else 18
            )
            for path in paths:
                summary = read(path)
                candidate = metrics([path, second / "summa" / path.name])
                control = metrics(
                    [
                        results / f"{r}-current-{layout}" / "summa" / path.name
                        for r in [1, 2]
                    ]
                )
                rows.append(
                    {
                        "variant": variant,
                        "layout": layout,
                        "family": summary["family"],
                        "operation": summary["operation"],
                        **candidate,
                        "qps_ratio_to_current": candidate["median_qps"]
                        / control["median_qps"],
                        "cpu_ratio_to_current": candidate["server_cpu_ms_per_request"]
                        / control["server_cpu_ms_per_request"],
                        "round_qps_ratios": [
                            a / b
                            for a, b in zip(
                                candidate["round_median_qps"],
                                control["round_median_qps"],
                                strict=True,
                            )
                        ],
                    }
                )
    write(out / "skipping.json", rows)
    write(out / "audits.json", audits)
    for layout in ["plain", "rgb"]:
        for variant in variants[1:]:
            cells = [
                r
                for r in rows
                if r["layout"] == layout
                and r["variant"] == variant
                and r["operation"] != "COUNT"
            ]
            print(
                layout,
                variant,
                "ranked geometric mean ratio",
                round(
                    math.exp(
                        statistics.mean(
                            math.log(r["qps_ratio_to_current"]) for r in cells
                        )
                    ),
                    4,
                ),
            )


def broad(results, out):
    families, queries = [], []
    for label in ["summa", "summa-rgb", "elasticsearch", "opensearch", "luxir"]:
        engine = "summa" if label == "summa-rgb" else label
        folder = results / "broad" / label
        agreement = read(folder / "agreement.json")
        eligible = {r["key"]: r for r in agreement if r["include"]}
        assert len(eligible) == 684
        paths = measured_paths(folder / engine)
        assert len(paths) == 54
        for path in paths:
            summary = read(path)
            families.append(
                {
                    "engine": label,
                    "family": summary["family"],
                    "operation": summary["operation"],
                    **metrics([path]),
                }
            )
            raw = read(path.with_suffix(".raw.json"))
            buckets = raw["aggregate"]["per_bucket"]
            assert len(buckets) == summary["query_count"]
            assert set(buckets) == {
                key
                for key, row in eligible.items()
                if row["query"]["query_class"] == summary["family"]
            }
            for key, bucket in buckets.items():
                assert bucket["errors"] == 0 and bucket["requests"] > 0
                row = eligible[key]
                own_count = row["observed"][engine]["count"]
                reference = row["observed"]["elasticsearch"]["count"]
                queries.append(
                    {
                        "engine": label,
                        "key": key,
                        "operation": summary["operation"],
                        "count": own_count,
                        "reference_count": reference,
                        "summa_relative_count_difference": abs(
                            row["observed"]["summa"]["count"] - reference
                        )
                        / max(1, reference),
                        "requests": bucket["requests"],
                        "mean_latency_ms": bucket["latency_us"]["mean"] / 1000,
                        "errors": 0,
                    }
                )
    assert len(queries) == 684 * 3 * 5
    write(out / "engines.json", families)
    write(out / "query-latencies.json", queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part", choices=["skipping", "broad", "all"], default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part in ["skipping", "all"]:
        skip(args.results, args.out)
    if args.part in ["broad", "all"]:
        broad(args.results, args.out)
