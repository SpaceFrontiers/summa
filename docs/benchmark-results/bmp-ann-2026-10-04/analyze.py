#!/usr/bin/env python3
"""Validate paired evidence and regenerate all measured comparison tables."""

import collections
import gzip
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads(gzip.decompress((ROOT / "evidence.json.gz").read_bytes()))
groups = collections.defaultdict(lambda: collections.defaultdict(list))
hashes = collections.defaultdict(set)
replay = collections.defaultdict(set)
audits = collections.defaultdict(set)
for row in data["records"]:
    groups[row["campaign"], row["case"]][row["variant"]].append(row)
    if row["kind"] == "binary":
        assert len(row["samples_ns"]) == 112
        hashes[row["width"], row["probes"], row["k"]].add(
            (row["bytes_hash"], row["result_hash"])
        )
    elif row["kind"] == "replay":
        replay[row["k"], row["gamma"], row["query"]].add(
            (row["hits_hash"], row["hit_count"])
        )
for row in data["audits"]:
    audits[tuple(row[k] for k in ("dims", "block", "distribution", "k", "gamma"))].add(
        row["hash"]
    )
assert len(hashes) == 12 and all(len(x) == 1 for x in hashes.values())
assert len(replay) == 864 and all(len(x) == 1 for x in replay.values())
assert audits and all(len(x) == 1 for x in audits.values())


def samples(rows):
    kind = rows[0]["kind"]
    if kind == "binary":
        return [n for row in rows for n in row["samples_ns"]]
    if kind == "replay":
        return [row["ns"] for row in rows]
    return [row["median_ns"] for row in rows]


def percentile95(values):
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


lines = [
    "# Complete measured comparison table",
    "",
    "Negative latency change is faster. All values are microseconds.",
    "",
    "Binary and replay medians/p95 pool per-query samples from the two runs",
    "of each variant. Criterion values are the median of two run medians;",
    "each run rotates queries across iterations. They are not request-tail",
    "latency distributions, so p95 is omitted. ABBA order throughout.",
    "",
]
previous = None
for (campaign, case), variants in sorted(groups.items()):
    assert set(variants) == {"baseline", "candidate"}, (campaign, case)
    for rows in variants.values():
        assert len({r["run"] for r in rows}) == 2, (campaign, case)
    if campaign != previous:
        lines += [
            "",
            f"## {campaign}",
            "",
            "| Case | Baseline | Candidate | Change | Baseline p95 | Candidate p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        previous = campaign
    a, b = samples(variants["baseline"]), samples(variants["candidate"])
    ma, mb = statistics.median(a), statistics.median(b)
    tails = (
        ("—", "—")
        if variants["baseline"][0]["kind"] == "criterion"
        else (f"{percentile95(a) / 1000:.2f}", f"{percentile95(b) / 1000:.2f}")
    )
    lines.append(
        f"| {case} | {ma / 1000:.2f} | {mb / 1000:.2f} | {100 * (mb / ma - 1):+.1f}% | {tails[0]} | {tails[1]} |"
    )
lines += [
    "",
    "## Process peak RSS",
    "",
    "Ranges across runs; includes fixture generation, oracle, mappings and",
    "runtime overhead. This is not incremental query heap usage.",
    "",
    "| Campaign | Baseline MiB | Candidate MiB |",
    "| --- | ---: | ---: |",
]
memory = collections.defaultdict(lambda: collections.defaultdict(list))
for row in data["resources"]:
    memory[row["campaign"]][row["variant"]].append(row["rss_bytes"] / 2**20)
for campaign, variants in sorted(memory.items()):

    def span(v, variants=variants):
        values = variants[v]
        return f"{min(values):.1f}–{max(values):.1f}"

    lines.append(f"| {campaign} | {span('baseline')} | {span('candidate')} |")
(ROOT / "tables.md").write_text("\n".join(lines) + "\n")
print(
    f"{len(groups)} paired cells; 12 binary byte/result hash keys, "
    f"864 real replay keys, {len(audits)} synthetic audit keys agree."
)
