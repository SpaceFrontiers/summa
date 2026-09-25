#!/usr/bin/env python3
"""Summarize upstream raw samples without turning one query into a speed claim."""

import argparse
import json
import math
import statistics


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def analyze(data, baseline, candidate, candidate_data=None):
    candidate_data = data if candidate_data is None else candidate_data
    rows = []
    for command, engines in data["results"].items():
        candidates = candidate_data["results"].get(command, {})
        if baseline not in engines or candidate not in candidates:
            continue
        ref = {q["query"]: q for q in engines[baseline]}
        trial = {q["query"]: q for q in candidates[candidate]}
        if set(ref) != set(trial):
            raise ValueError(f"query set mismatch for {command}")
        for tag in (
            "all",
            "term",
            "intersection",
            "union",
            "phrase",
            "intersection_union",
            "negated",
            "two-phase-critic",
        ):
            pairs = [
                (ref[q], trial[q]) for q in ref if tag == "all" or tag in ref[q]["tags"]
            ]
            if not pairs:
                continue
            ratios, base_samples, candidate_samples = [], [], []
            mismatches = []
            minimum_ratios = []
            for a, b in pairs:
                if not a["duration"] or not b["duration"]:
                    raise ValueError(f"missing samples: {command}: {a['query']}")
                if a["count"] != b["count"]:
                    mismatches.append(
                        {
                            "query": a["query"],
                            baseline: a["count"],
                            candidate: b["count"],
                        }
                    )
                ratios.append(
                    statistics.median(b["duration"])
                    / max(1, statistics.median(a["duration"]))
                )
                minimum_ratios.append(min(b["duration"]) / max(1, min(a["duration"])))
                base_samples.extend(a["duration"])
                candidate_samples.extend(b["duration"])
            rows.append(
                {
                    "command": command,
                    "family": tag,
                    "queries": len(pairs),
                    "candidate_over_baseline_median_ratio": statistics.median(ratios),
                    "candidate_over_baseline_geomean_ratio": math.exp(
                        statistics.mean(map(math.log, ratios))
                    ),
                    "candidate_over_baseline_minimum_geomean_ratio": math.exp(
                        statistics.mean(map(math.log, minimum_ratios))
                    ),
                    "baseline_us": {
                        str(p): percentile(base_samples, p) for p in (0.5, 0.95, 0.99)
                    },
                    "candidate_us": {
                        str(p): percentile(candidate_samples, p)
                        for p in (0.5, 0.95, 0.99)
                    },
                    "count_mismatches": mismatches,
                }
            )
    if not rows:
        raise ValueError("no comparable engine/command pairs")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("--candidate-results", help="separate candidate results file")
    parser.add_argument("--baseline", default="tantivy-0.26")
    parser.add_argument("--candidate", default="summa")
    args = parser.parse_args()
    candidate_data = None
    if args.candidate_results:
        with open(args.candidate_results) as f:
            candidate_data = json.load(f)
    with open(args.results) as f:
        rows = analyze(json.load(f), args.baseline, args.candidate, candidate_data)
    print(json.dumps(rows, indent=2))
