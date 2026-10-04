"""Validate the evidence archive and regenerate all numerical result tables."""

import csv
import hashlib
import io
import json
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "evidence-manifest.json").read_text())
    archive = b"".join((ROOT / p).read_bytes() for p in manifest["parts"])
    assert hashlib.sha256(archive).hexdigest() == manifest["sha256"]
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        files = {name: z.read(name) for name in z.namelist()}
    assert set(files) == set(manifest["members"])
    for name, data in files.items():
        assert hashlib.sha256(data).hexdigest() == manifest["members"][name], name

    def read(name):
        return files[name].decode()

    def js(name):
        return json.loads(read(name))

    def rows(name):
        return [json.loads(line) for line in read(name).splitlines()]

    provenance = js("provenance.json")
    for name, digest in provenance["sources"].items():
        assert hashlib.sha256(files[name]).hexdigest() == digest, name
    for variant in ["size-only", "gated", "final"]:
        assert js(f"validation/{variant}/run.json")["success"]

    results = {
        "micro": [],
        "queries": [],
        "resources": [],
        "individual": [],
        "validation": {},
    }
    lines = [
        "# Complete opportunity experiment table",
        "",
        "Ratios are baseline/candidate; above 1 is faster. Query totals sum the",
        "median wall time of each query, pooling both timing rounds. They describe",
        "this suite, not a production traffic mix. Paired ratios use each run separately.",
        "`multi_total` includes all 30 targeted queries; `multi_active` excludes the",
        "four duplicate/missing-term controls, leaving 26 three- to six-term queries.",
        "",
        "## Primitive screens",
        "",
        "The current dispatcher is the baseline. BSR includes conversion and both",
        "ordinal ranks; VP2 includes right-ordinal recovery. All calls use bounded",
        "output. Real traces contain 7,118 sampled ARM calls; x86 replay is not a",
        "sample of the 10M index's posting distribution.",
        "",
        "**BSR rows are invalid for cross-block use:** the initial single-block",
        "oracle missed lost tails when a decoded block ends inside a state group.",
        "These timings are archived experiments, not valid adoption evidence.",
        "The corrected BSR prototype also fails the corpus oracle; its campaign has no timing results.",
        "",
        "| Host | Screen | Fixture | Candidate | Baseline ns/call | Candidate ns/call | Ratio |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for host in ["arm", "x86"]:
        for kind in [
            "screen",
            "replay",
            "orientation-screen",
            "orientation-replay",
            "popcnt-screen",
            "popcnt-replay",
        ]:
            if host == "arm" and kind.startswith("popcnt"):
                continue
            samples = defaultdict(dict)
            for row in csv.DictReader(io.StringIO(read(f"{host}-{kind}.csv"))):
                key = row["fixture"], row["kernel"]
                p = int(row["pass"])
                assert p not in samples[key]
                samples[key][p] = int(row["elapsed_ns"]) / int(row["calls"])
            assert all(set(s) == set(range(11)) for s in samples.values())
            expected_kernels = {
                "current",
                "bsr_packed",
                "bsr_stream",
                "bsr_conditional",
            }
            if kind.startswith("orientation"):
                expected_kernels = {"current", "oriented"}
            elif kind.startswith("popcnt"):
                expected_kernels = {
                    "current",
                    "bsr_packed",
                    "bsr_stream",
                    "bsr_packed_popcnt",
                    "bsr_stream_popcnt",
                    "bsr_conditional_popcnt",
                }
            elif host == "x86":
                expected_kernels |= {"vp2", "vp2_conditional"}
            for fixture in {f for f, _ in samples}:
                assert {k for f, k in samples if f == fixture} == expected_kernels
            for (fixture, kernel), sample in samples.items():
                if kernel == "current":
                    continue
                before = statistics.median(samples[fixture, "current"].values())
                after = statistics.median(sample.values())
                cell = {
                    "host": host,
                    "kind": kind,
                    "fixture": fixture,
                    "candidate": kernel,
                    "before_ns": before,
                    "after_ns": after,
                    "ratio": before / after,
                    "cross_block_invalid": kernel.startswith("bsr_"),
                }
                results["micro"].append(cell)
                lines.append(
                    f"| {host} | {kind} | {fixture} | {kernel}{' (invalid resume)' if kernel.startswith('bsr_') else ''} | {before:.2f} | {after:.2f} | {before / after:.3f}× |"
                )

    lines += [
        "",
        "## Real-query comparisons",
        "",
        "Every cell validates the independent exhaustive audit, complete recorded",
        "ID/score-bit/count/plan equality, repetition counts and immutable file inventories.",
        "",
        "| Campaign | Host/index | Comparison | Limit | Queries | Before ms | After ms | Ratio | Paired ratios |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    audit_count = 0
    execution_count = 0
    inventories = {}
    for campaign in [
        "adaptive-integrated",
        "batched-integrated",
        "broad",
        "batched-final",
        "final",
        "confirm",
        "selected-final",
    ]:
        for host in ["arm", "x86"]:
            if campaign in ["broad", "batched-final", "confirm"] and host == "x86":
                continue
            if campaign == "selected-final" and host == "arm":
                continue
            directory = f"{host}-{campaign}"
            done = js(f"{directory}/done.json")
            assert done["status"] == "pass"
            assert js(f"{directory}/inventory-before.json") == js(
                f"{directory}/inventory-after.json"
            )
            inventory = js(f"{directory}/inventory-before.json")
            if host in inventories:
                assert inventory == inventories[host]
            else:
                inventories[host] = inventory
            broad = campaign in [
                "broad",
                "batched-final",
                "final",
                "confirm",
                "selected-final",
            ]
            query_file = "broad-queries.jsonl" if broad else "queries.jsonl"
            queries = rows(query_file)
            count, iterations = (199, 7) if broad else (30, 15)
            assert len(queries) == count
            assert len({q["query"] for q in queries}) == count
            environment = js(f"{directory}/environment.json")
            assert (
                environment["query_sha256"]
                == hashlib.sha256(files[query_file]).hexdigest()
            )
            assert environment["iterations"] == iterations
            assert environment["warmups"] == 5
            assert environment["flags"] == ""
            families = {q["query"]: q.get("family", q["class"]) for q in queries}
            multi_queries = {q["query"] for q in rows("queries.jsonl")}
            active_multi_queries = {
                q["query"]
                for q in rows("queries.jsonl")
                if q["family"] != "multi_boundary"
            }
            variants = list(done["binary_sha256"])
            assert variants[0] == "baseline"
            if campaign in ["final", "confirm"]:
                assert variants == ["baseline", "gated"] + (
                    ["wide-gated"] if host == "x86" else []
                )
                for variant, digest in done["binary_sha256"].items():
                    assert (
                        digest == provenance["final_binaries"][f"{host}-{variant}-bin"]
                    )
            if campaign == "selected-final":
                assert variants == ["baseline", "gated"]
                for variant, digest in done["binary_sha256"].items():
                    assert (
                        digest == provenance["selected_binaries"][f"x86-{variant}-bin"]
                    )
            phase_variant = {"a": variants[0], "b": variants[1]}
            if len(variants) == 3:
                phase_variant["c"] = variants[2]
            comparisons = [("a", "b")]
            if "c" in phase_variant:
                comparisons += [("a", "c"), ("b", "c")]
            for index in inventory:
                audits = []
                for variant in variants:
                    audit = rows(f"{directory}/{index}-{variant}-audit.jsonl")
                    assert len(audit) == count
                    assert all(r["optimized"] == r["exhaustive"] for r in audit)
                    audits.append(audit)
                    audit_count += len(audit)
                assert all(a == audits[0] for a in audits)
                for limit in [10, 100, 0]:
                    actual = {}
                    resource = {}
                    for phase in phase_variant:
                        for repeat in [1, 2]:
                            label = f"{index}-{phase}{repeat}-{limit}"
                            data = rows(f"{directory}/{label}.jsonl")
                            assert len(data) == count
                            assert [r["query"] for r in data] == [
                                q["query"] for q in queries
                            ]
                            assert all(len(r["samples_ns"]) == iterations for r in data)
                            execution_count += count * iterations
                            actual[phase, repeat] = data
                            text = read(f"{directory}/{label}.resources")
                            if host == "arm":
                                times = re.search(
                                    r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys",
                                    text,
                                )
                                cpu = float(times[2]) + float(times[3])
                                rss = (
                                    int(
                                        re.search(
                                            r"(\d+)\s+maximum resident set size", text
                                        )[1]
                                    )
                                    / 1024
                                )
                            else:
                                cpu = sum(
                                    float(
                                        re.search(
                                            rf"{label} time \(seconds\): ([\d.]+)", text
                                        )[1]
                                    )
                                    for label in ["User", "System"]
                                )
                                rss = int(
                                    re.search(
                                        r"Maximum resident set size \(kbytes\): (\d+)",
                                        text,
                                    )[1]
                                )
                            resource[phase, repeat] = {
                                "cpu_seconds": cpu,
                                "rss_kib": rss,
                            }
                    reference = [
                        [
                            r[k]
                            for k in [
                                "query",
                                "class",
                                "limit",
                                "count",
                                "hits",
                                "plan",
                            ]
                        ]
                        for r in actual["a", 1]
                    ]
                    for data in actual.values():
                        assert [
                            [
                                r[k]
                                for k in [
                                    "query",
                                    "class",
                                    "limit",
                                    "count",
                                    "hits",
                                    "plan",
                                ]
                            ]
                            for r in data
                        ] == reference
                    for before_phase, after_phase in comparisons:
                        comparison = f"{phase_variant[before_phase]}/{phase_variant[after_phase]}"
                        for family in ["all", "multi_total", "multi_active"] + sorted(
                            set(families.values())
                        ):
                            chosen = [
                                i
                                for i, q in enumerate(queries)
                                if family == "all"
                                or (
                                    family == "multi_total"
                                    and q["query"] in multi_queries
                                )
                                or (
                                    family == "multi_active"
                                    and q["query"] in active_multi_queries
                                )
                                or families[q["query"]] == family
                            ]

                            def total(phase, repeats, actual=actual, chosen=chosen):
                                return sum(
                                    statistics.median(
                                        s[0]
                                        for rep in repeats
                                        for s in actual[phase, rep][i]["samples_ns"]
                                    )
                                    for i in chosen
                                )

                            before, after = (
                                total(before_phase, [1, 2]),
                                total(after_phase, [1, 2]),
                            )
                            paired = [
                                total(before_phase, [rep]) / total(after_phase, [rep])
                                for rep in [1, 2]
                            ]
                            cell = {
                                "campaign": campaign,
                                "host": host,
                                "index": index,
                                "comparison": comparison,
                                "limit": limit,
                                "family": family,
                                "queries": len(chosen),
                                "before_ns": before,
                                "after_ns": after,
                                "ratio": before / after,
                                "paired_ratios": paired,
                            }
                            results["queries"].append(cell)
                            if family == "all":
                                lines.append(
                                    f"| {campaign} | {host}/{index} | {comparison} | {limit} | {len(chosen)} | {before / 1e6:.3f} | {after / 1e6:.3f} | {before / after:.3f}× | {paired[0]:.3f}/{paired[1]:.3f} |"
                                )
                        for i, query in enumerate(queries):
                            before = statistics.median(
                                s[0]
                                for rep in [1, 2]
                                for s in actual[before_phase, rep][i]["samples_ns"]
                            )
                            after = statistics.median(
                                s[0]
                                for rep in [1, 2]
                                for s in actual[after_phase, rep][i]["samples_ns"]
                            )
                            results["individual"].append(
                                {
                                    "campaign": campaign,
                                    "host": host,
                                    "index": index,
                                    "comparison": comparison,
                                    "limit": limit,
                                    "query": query["query"],
                                    "family": families[query["query"]],
                                    "before_ns": before,
                                    "after_ns": after,
                                    "ratio": before / after,
                                    "paired_ratios": [
                                        statistics.median(
                                            s[0]
                                            for s in actual[before_phase, rep][i][
                                                "samples_ns"
                                            ]
                                        )
                                        / statistics.median(
                                            s[0]
                                            for s in actual[after_phase, rep][i][
                                                "samples_ns"
                                            ]
                                        )
                                        for rep in [1, 2]
                                    ],
                                }
                            )
                        before_cpu = sum(
                            resource[before_phase, rep]["cpu_seconds"] for rep in [1, 2]
                        )
                        after_cpu = sum(
                            resource[after_phase, rep]["cpu_seconds"] for rep in [1, 2]
                        )
                        results["resources"].append(
                            {
                                "campaign": campaign,
                                "host": host,
                                "index": index,
                                "comparison": comparison,
                                "limit": limit,
                                "cpu_ratio": before_cpu / after_cpu,
                                "before_cpu_seconds": before_cpu,
                                "after_cpu_seconds": after_cpu,
                                "before_rss_kib": statistics.mean(
                                    resource[before_phase, rep]["rss_kib"]
                                    for rep in [1, 2]
                                ),
                                "after_rss_kib": statistics.mean(
                                    resource[after_phase, rep]["rss_kib"]
                                    for rep in [1, 2]
                                ),
                            }
                        )
    lines += [
        "",
        "## Final per-family results",
        "",
        "| Campaign | Host/index | Comparison | Family | Limit | Ratio | Paired ratios |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for c in results["queries"]:
        if (
            c["campaign"] in ["final", "confirm", "selected-final"]
            and c["family"] != "all"
        ):
            a, b = c["paired_ratios"]
            lines.append(
                f"| {c['campaign']} | {c['host']}/{c['index']} | {c['comparison']} | {c['family']} | {c['limit']} | {c['ratio']:.3f}× | {a:.3f}/{b:.3f} |"
            )
    lines += [
        "",
        "## Process resources",
        "",
        "CPU is total process user + system time, including startup and warmups.",
        "Peak RSS is the mean of the two process maxima; it includes mapped index pages.",
        "Caller-thread timestamps in raw rows are not total query CPU and are not used here.",
        "",
        "| Campaign | Host/index | Comparison | Limit | CPU ratio | Before RSS MiB | After RSS MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in results["resources"]:
        lines.append(
            f"| {c['campaign']} | {c['host']}/{c['index']} | {c['comparison']} | {c['limit']} | {c['cpu_ratio']:.3f}× | {c['before_rss_kib'] / 1024:.2f} | {c['after_rss_kib'] / 1024:.2f} |"
        )
    rejected = "x86-bsr-rejected"
    assert f"{rejected}/done.json" not in files
    assert js(f"{rejected}/inventory-before.json") == inventories["x86"]
    assert "ranked results differ from exhaustive oracle" in read(
        f"{rejected}/default-bsr-audit.log"
    )
    rejected_audits = {
        variant: rows(f"{rejected}/default-{variant}-audit.jsonl")
        for variant in ["baseline", "gated", "bsr"]
    }
    assert [len(rejected_audits[v]) for v in ["baseline", "gated", "bsr"]] == [
        199,
        199,
        185,
    ]
    assert rejected_audits["baseline"] == rejected_audits["gated"]
    assert rejected_audits["bsr"] == rejected_audits["baseline"][:185]
    failure = js("bsr-failure-diff.json")
    diagnostic = rows("bsr-failure-actual.jsonl")[0]
    reference = next(
        r
        for r in rows("x86-selected-final/default-a1-100.jsonl")
        if r["query"] == failure["query"]
    )
    assert diagnostic["query"] == failure["query"] == "+the +of +in +and"
    assert diagnostic["hits"] == failure["actual"] != failure["expected"]
    assert reference["hits"] == failure["expected"]
    results["rejected_bsr"] = {
        "status": "rejected before timing",
        "successful_baseline_audits": 398,
        "successful_candidate_audits": 185,
        "failing_query": failure["query"],
        "expected_ids_missing": sorted(
            set(dict(failure["expected"])) - set(dict(failure["actual"]))
        ),
        "unexpected_ids": sorted(
            set(dict(failure["actual"])) - set(dict(failure["expected"]))
        ),
    }
    lines += [
        "",
        "## Rejected BSR corpus audit",
        "",
        "The tail-corrected BSR prototype passes 185 query audits, then fails exact",
        "ranked output for `+the +of +in +and`. The starting and gated binaries pass",
        "all 199 audits. No BSR query timing is reported. Its reproducer and full",
        "expected/actual vectors are retained in the evidence.",
        "",
    ]
    results["validation"] = {
        "exhaustive_audits": audit_count,
        "recorded_query_executions": execution_count,
        "archive_sha256": manifest["sha256"],
    }
    lines += [
        "",
        f"Validated {audit_count:,} exhaustive audits and {execution_count:,} recorded query executions (warmups excluded).",
        "",
    ]
    individual = results.pop("individual")
    results["largest_final_slowdowns"] = sorted(
        (
            r
            for r in individual
            if r["campaign"] in ["final", "confirm", "selected-final"]
            and r["before_ns"] >= 100_000
        ),
        key=lambda r: r["ratio"],
    )[:40]
    results["retained_final_slowdowns"] = sorted(
        (
            r
            for r in individual
            if r["campaign"] == "selected-final"
            and r["comparison"] == "baseline/gated"
            and r["before_ns"] >= 100_000
            and r["limit"] != 0
        ),
        key=lambda r: r["ratio"],
    )[:40]
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (ROOT / "table.md").write_text("\n".join(lines))
    print(json.dumps(results["validation"], indent=2))


if __name__ == "__main__":
    main()
