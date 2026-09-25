#!/usr/bin/env python3
"""Summarize a completed four-engine campaign without hiding coverage or errors."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from campaign import ENGINES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--followup",
        action="store_true",
        help="compare before/summa, optimized/summa and rgb/summa",
    )
    modes.add_argument(
        "--phrase-followup",
        action="store_true",
        help="compare first-term admission, certified rare-term admission, and rerun Luxir",
    )
    modes.add_argument(
        "--gap-followup",
        action="store_true",
        help="compare prior/current Summa, current Summa with impacts, and rerun Luxir",
    )
    modes.add_argument(
        "--scaling-followup",
        action="store_true",
        help="32-vCPU comparison of prior/current/RGB Summa and three reference engines",
    )
    modes.add_argument(
        "--conjunction-followup",
        action="store_true",
        help="compare HTTP response encoding and automatic HTTP workers",
    )
    modes.add_argument(
        "--handoff-followup",
        action="store_true",
        help="compare borrowed-ID responses, baseline and Luxir at 1/32/64 clients",
    )
    modes.add_argument(
        "--worker-followup",
        action="store_true",
        help="compare current/candidate worker configurations and Luxir at 32 clients",
    )
    args = parser.parse_args()
    root = args.results
    agreement = json.loads((root / "agreement.json").read_text())
    coverage = Counter(row["query"]["query_class"] for row in agreement)
    included = Counter(
        row["query"]["query_class"] for row in agreement if row["include"]
    )
    shared_input = any(row.get("comparison") == "shared-input" for row in agreement)
    coverage_text = (
        f"Shared-input coverage: **{sum(included.values())}/{len(agreement)} queries**. "
        f"Exact counts agree for **{sum(row.get('counts_agree', False) for row in agreement)}**. "
        "Count differences remain in agreement.json; this does not establish equal work or ranking."
        if shared_input
        else f"Count agreement: **{sum(included.values())}/{len(agreement)} queries**. "
        "This is a restricted workload, not the complete published benchmark."
    )
    if args.worker_followup:
        variants = [
            ("current", "current/summa", "Summa current workers"),
            ("candidate", "candidate/summa", "Summa candidate workers"),
            ("luxir", "luxir/luxir", "Luxir"),
        ]
        title = "worker configuration and count repeatability"
    elif args.handoff_followup:
        variants = [
            ("before", "before/summa", "Summa before"),
            ("typed", "typed/summa", "Summa borrowed IDs"),
            ("luxir", "luxir/luxir", "Luxir"),
        ]
        title = "borrowed-ID responses and search-pool handoffs"
    elif args.conjunction_followup:
        variants = [
            ("before", "before/summa", "Summa before (HTTP 2)"),
            ("encoded", "encoded/summa", "Summa encoded (HTTP 2)"),
            ("encoded-auto", "encoded-auto/summa", "Summa encoded (HTTP auto: 4)"),
            ("luxir", "luxir/luxir", "Luxir"),
        ]
        title = "conjunction response encoding and CPU-based HTTP workers"
    elif args.scaling_followup:
        variants = [
            ("before", "before/summa", "Summa before"),
            ("optimized", "optimized/summa", "Summa optimized"),
            ("rgb", "rgb/summa", "Summa optimized + RGB"),
            ("elasticsearch", "elasticsearch/elasticsearch", "Elasticsearch"),
            ("opensearch", "opensearch/opensearch", "OpenSearch"),
            ("luxir", "luxir/luxir", "Luxir"),
        ]
        if (root / "impacts").is_dir():
            variants.insert(
                3, ("impacts", "impacts/summa", "Summa optimized + impacts")
            )
        title = "32-vCPU host, 32 clients, Summa/RGB and references"
    elif args.gap_followup:
        variants = [
            ("before", "before/summa", "Summa before"),
            ("optimized", "optimized/summa", "Summa optimized"),
            ("impacts", "impacts/summa", "Summa optimized + impacts"),
            ("luxir", "luxir/luxir", "Luxir"),
        ]
        title = "phrase scan optimization, optional impacts, and Luxir"
    elif args.phrase_followup:
        variants = [
            ("before", "before/summa", "Summa first-term bounds"),
            ("optimized", "optimized/summa", "Summa certified rare-term bounds"),
            ("luxir", "luxir/luxir", "Luxir"),
        ]
        title = "non-RGB phrase optimization and Luxir"
    elif args.followup:
        variants = [
            ("before", "before/summa", "Summa before"),
            ("optimized", "optimized/summa", "Summa optimized"),
            ("rgb", "rgb/summa", "Summa optimized + RGB"),
        ]
        title = "Summa optimization and RGB"
    else:
        variants = [
            (
                engine,
                engine,
                {
                    "summa": "Summa",
                    "elasticsearch": "Elasticsearch",
                    "opensearch": "OpenSearch",
                    "luxir": "Luxir",
                }[engine],
            )
            for engine in ENGINES
        ]
        title = "four engines"
    engines = [engine for engine, _, _ in variants]
    cells = {}
    for engine, relative, _ in variants:
        directory = root / relative
        if not json.loads((directory / "complete.json").read_text())["complete"]:
            raise RuntimeError(f"{engine} is incomplete")
        for path in sorted(directory.glob("*-c*.json")):
            if path.name.endswith(".raw.json"):
                continue
            row = json.loads(path.read_text())
            if row["errors"] or any(rep["errors"] for rep in row["repetitions"]):
                raise RuntimeError(f"{path} contains errors")
            key = (row["family"], row["operation"], row["clients"])
            cells.setdefault(key, {})[engine] = row
    if not cells or any(set(rows) != set(engines) for rows in cells.values()):
        raise RuntimeError("engine cell sets differ")
    columns = ["family", "operation", "clients", "queries"]
    for engine in engines:
        columns += [
            f"{engine}_median_qps",
            f"{engine}_min_qps",
            f"{engine}_max_qps",
            f"{engine}_peak_rss_mib",
        ]
    with (root / "comparison.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        for (family, operation, clients), results in sorted(cells.items()):
            row = [family, operation, clients, included[family]]
            for engine in engines:
                result = results[engine]
                qps = [rep["qps"] for rep in result["repetitions"]]
                row.extend(
                    [
                        result["median_qps"],
                        min(qps),
                        max(qps),
                        result["memory"]["peak_vmrss_kb"] / 1024,
                    ]
                )
            writer.writerow(row)
    lines = [
        "# Same-host Searchbench throughput: " + title,
        "",
        coverage_text,
        "",
        "10M Wikipedia chunks; one merged segment; "
        + (
            "32-vCPU host (16 physical cores with SMT), 30 server hardware threads "
            "and two driver threads on the reserved physical core. "
            if args.scaling_followup
            or args.conjunction_followup
            or args.handoff_followup
            or args.worker_followup
            else "six server hardware threads and two driver threads on a separate physical core. "
        )
        + (
            "Reference JVM heaps: 8 GiB. "
            if not (
                args.followup
                or args.phrase_followup
                or args.gap_followup
                or args.conjunction_followup
                or args.handoff_followup
                or args.worker_followup
            )
            else ""
        )
        + "Query and request caches disabled. Thirty-second session warmup, full untimed "
        "validation, one-second connection warmup, three ten-second repetitions per cell.",
        "",
        "Summa uses a benchmark HTTP frontend over core, not its production gRPC service. "
        + (
            "Text-analysis differences remain visible in the shared inputs; "
            "errors are excluded from successful-query throughput. "
            if shared_input
            else "Text-analysis differences exclude many queries; equal corpus counts do not prove "
            "general analyzer or relevance equivalence. "
        )
        + (
            ""
            if args.followup
            else "Luxir uses the official 0.1.0 x86-64-v4 release, not the article's local build. "
        )
        + "No p99 claim is made.",
        "",
        "| Family | Included | Published queries |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {family} | {included[family]} | {count} |"
        for family, count in sorted(coverage.items())
    )
    lines += [
        "",
        "**Throughput (queries/second; higher is better).** Values are the median "
        "of three repetitions. The CSV retains repetition "
        "min/max and peak process RSS. Raw replay JSON and memory samples accompany each cell.",
        "",
        "| Family | Operation | Clients | Distinct queries | "
        + " | ".join(label + " (QPS)" for _, _, label in variants)
        + " |",
        "| --- | --- | ---: | ---: | " + " | ".join("---:" for _ in variants) + " |",
    ]
    for (family, operation, clients), results in sorted(cells.items()):
        values = " | ".join(
            f"{results[engine]['median_qps']:,.1f}" for engine in engines
        )
        lines.append(
            f"| {family} | {operation} | {clients} | {included[family]} | {values} |"
        )
    if args.worker_followup:
        selection = json.loads((root / "selection.json").read_text())
        lines[6:6] = [
            "All 15 count-compatible queries are measured at 32 clients. "
            "Both Summa configurations use the same frozen borrowed-ID executable and "
            "ordinary index. Current: 30 search/blocking workers, four HTTP workers. "
            f"Candidate: {selection['workers']} search/blocking workers, "
            f"{selection['http']} HTTP workers. Admission remains 64. "
            "The WORKERS argument couples search and blocking pool sizes; this is not "
            "an isolated ablation of either pool. No runtime implementation or default "
            "changes. Server CPUs: 0–14,16–30; driver CPUs: 15,31. "
            "No copying, compilation or indexing overlaps timing.",
            "",
        ]
    elif args.handoff_followup:
        lines[6:6] = [
            "All 15 count-compatible queries are measured at 32 clients; the seven "
            "conjunctions are also measured at one and 64 clients. All three variants run "
            "sequentially in isolated loopback networking. Both Summa variants use the same "
            "ordinary index, automatic HTTP workers (four), 30 blocking/search workers and "
            "64-request admission. RGB and impacts remain separate. Server CPUs: 0–14,16–30; "
            "driver CPUs: 15,31. No copying, compilation or indexing overlaps timing. "
            "Core search algorithms are unchanged.",
            "",
        ]
    elif args.conjunction_followup:
        lines[6:6] = [
            "Only the seven agreeing `and_high_low` queries are timed here, at one and 32 clients. "
            "All four variants run sequentially in isolated loopback networking; all Summa "
            "variants use the same ordinary index, without RGB or impact metadata. "
            "Server CPUs: 0–14,16–30; driver CPUs: 15,31. No copying, compilation or indexing "
            "overlaps timed runs. Core search algorithms are unchanged by this follow-up.",
            "",
        ]
    elif args.scaling_followup:
        lines[6:6] = [
            "All variants run sequentially with isolated loopback networking. "
            "Prior/current Summa use the same immutable ordinary index; RGB is a separate "
            "current-format build from the same corpus. Impacts are disabled except in "
            "the explicitly labeled optional-impact follow-up, when present. "
            "No indexing, copying or compilation overlaps a timed run. "
            "Server CPUs: 0–14,16–30; driver CPUs: 15,31. This measures 32 concurrent "
            "clients on 30 server hardware threads, not 32 physical cores.",
            "",
        ]
    elif args.gap_followup:
        lines[6:6] = [
            "All four variants run sequentially. The prior and optimized Summa binaries "
            "use the same immutable non-RGB index, without impact metadata. The optional "
            "impact variant uses a separate index rebuilt from the same corpus; impacts remain "
            "disabled by default. Luxir is rerun on the same host. "
            "Indexing and compilation finish before query timing.",
            "",
        ]
    elif args.phrase_followup:
        lines[6:6] = [
            "All three variants are measured sequentially in this run. Both Summa binaries "
            "use the same rebuilt non-RGB index, without impact metadata. The comparison "
            "binary disables certified rare-term admission; all other code and writer output "
            "are identical. Luxir is rerun on the same host. "
            "Indexing and compilation finish before query timing.",
            "",
        ]
    elif args.followup:
        lines[6:6] = [
            "The three Summa instances run sequentially: the unchanged binary and the optimized "
            "binary use the same original index; RGB uses a separately built index from the same "
            "corpus with the body field reordered. "
            "Reference-engine timings remain in the original report; no mixed-run speedup is claimed.",
            "",
        ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(root / "comparison.md")


if __name__ == "__main__":
    main()
