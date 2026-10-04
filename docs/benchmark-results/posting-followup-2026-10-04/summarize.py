"""Verify archived evidence and regenerate the complete follow-up tables."""

import gzip
import hashlib
import io
import json
import re
import statistics
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        files = {
            str(p.relative_to(source)): p.read_bytes()
            for p in source.rglob("*")
            if p.is_file()
        }
    else:
        manifest = json.loads((ROOT / "evidence-manifest.json").read_text())
        archive = b"".join((ROOT / p).read_bytes() for p in manifest["parts"])
        assert hashlib.sha256(archive).hexdigest() == manifest["sha256"]
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            files = {name: z.read(name) for name in z.namelist()}
        assert set(files) == set(manifest["members"])
        for name, data in files.items():
            assert hashlib.sha256(data).hexdigest() == manifest["members"][name], name

    def js(name):
        return json.loads(files[name])

    def rows(name):
        return [json.loads(line) for line in files[name].splitlines()]

    if "provenance.json" in files:
        provenance = js("provenance.json")
        for path, digest in provenance["selected_sources"].items():
            assert hashlib.sha256(files[path]).hexdigest() == digest, path
        for path, digest in provenance["source_variants"].items():
            assert hashlib.sha256(files[path]).hexdigest() == digest, path
        assert js("validation/native/run.json")["success"]
        assert provenance["x86_binaries"] == js("corrected-final/done.json")["binaries"]
        assert (
            provenance["arm_binaries"]
            == js("arm-results/arm-dedicated/done.json")["binary_sha256"]
        )

    queries = rows("broad-queries.jsonl")
    active = {
        q["query"] for q in rows("queries.jsonl") if q["family"] != "multi_boundary"
    }
    assert len(active) == 26
    results = {"groups": [], "individual": [], "resources": [], "validation": {}}
    audits = executions = 0
    inventories = {}
    architecture_audits = {}
    campaigns = [
        ("x86", "corrected-final", ["default", "rgb-pairs"]),
        ("arm", "arm-results/arm-dedicated", ["default", "rgb-pairs"]),
        ("repeat", "repeat", [None]),
        ("preserved", "preserved", [None]),
    ]
    for campaign, folder, indexes in campaigns:
        if f"{folder}/done.json" not in files:
            assert len(sys.argv) > 1, f"Missing completed campaign: {campaign}"
            continue
        done = js(f"{folder}/done.json")
        assert done["status"] == "pass"
        qs = queries if campaign in ["x86", "arm"] else rows(f"{folder}/queries.jsonl")
        limits = done.get("limits", [10, 100, 0])
        if campaign == "arm":
            phases = [
                ("a1", "baseline"),
                ("b1", "gated"),
                ("b2", "gated"),
                ("a2", "baseline"),
            ]
        else:
            phases = [(str(i), v) for i, v in enumerate(done["phases"])]
        variants = list(dict.fromkeys(v for _, v in phases))
        comparisons = [("baseline", v) for v in variants if v != "baseline"]
        if campaign == "x86":
            comparisons = [("head", v) for v in variants if v != "head"] + [
                ("baseline", "gated")
            ]
            if "bsr" in variants:
                comparisons += [("gated", "bsr")]
        for index in indexes:
            prefix = f"{index}-" if index else ""
            audit_ref = None
            for v in variants:
                # Focused runners name audits without an index prefix.
                name = f"{folder}/{prefix}{v}-audit.jsonl"
                audit = rows(name)
                assert len(audit) == len(qs)
                assert all(r["exhaustive"] == r["optimized"] for r in audit)
                if audit_ref is not None:
                    assert audit == audit_ref
                audit_ref = audit
                audits += len(audit)
            if campaign in ["x86", "arm"]:
                if index in architecture_audits:
                    assert audit_ref == architecture_audits[index], (campaign, index)
                architecture_audits[index] = audit_ref
            before_inventory = js(f"{folder}/inventory-before.json")
            assert before_inventory == js(f"{folder}/inventory-after.json")
            fixture = before_inventory[index] if index else before_inventory
            fixture_name = index or "default"
            if fixture_name in inventories:
                assert fixture == inventories[fixture_name], (campaign, fixture_name)
            inventories[fixture_name] = fixture
            for limit in limits:
                data, resources, by_variant = {}, {}, {}
                reference = None
                for phase, variant in phases:
                    label = (
                        f"{prefix}{phase}-{variant}-{limit}"
                        if campaign != "arm"
                        else f"{prefix}{phase}-{limit}"
                    )
                    values = rows(f"{folder}/{label}.jsonl")
                    assert len(values) == len(qs)
                    assert [v["query"] for v in values] == [q["query"] for q in qs]
                    expected_samples = 7 if campaign in ["x86", "arm"] else 21
                    assert all(len(v["samples_ns"]) == expected_samples for v in values)
                    actual = [
                        (
                            v["query"],
                            v["class"],
                            v["limit"],
                            v["count"],
                            v["hits"],
                            v["plan"],
                        )
                        for v in values
                    ]
                    if reference is not None:
                        assert actual == reference
                    reference = actual
                    executions += len(values) * expected_samples
                    data[phase] = values
                    by_variant.setdefault(variant, []).append(phase)
                    text = files[f"{folder}/{label}.resources"].decode()

                    def metric(key, text=text):
                        return float(
                            re.search(re.escape(key) + r":\s*([\d.]+)", text).group(1)
                        )

                    resources[phase] = {
                        "cpu": metric("User time (seconds)")
                        + metric("System time (seconds)"),
                        "rss": metric("Maximum resident set size (kbytes)"),
                    }
                for before, after in comparisons:
                    bp, ap = by_variant[before], by_variant[after]
                    assert len(bp) == len(ap) == 2

                    def total(phases_selected, chosen, data=data):
                        return sum(
                            statistics.median(
                                t[0]
                                for phase in phases_selected
                                for t in data[phase][i]["samples_ns"]
                            )
                            for i in chosen
                        )

                    families = ["all", "multi_active"] + sorted(
                        {q["family"] for q in qs}
                    )
                    for family in families:
                        chosen = [
                            i
                            for i, q in enumerate(qs)
                            if family == "all"
                            or (family == "multi_active" and q["query"] in active)
                            or q["family"] == family
                        ]
                        if not chosen:
                            continue
                        b, a = total(bp, chosen), total(ap, chosen)
                        results["groups"].append(
                            {
                                "campaign": campaign,
                                "index": index or "default",
                                "comparison": f"{before}/{after}",
                                "limit": limit,
                                "family": family,
                                "queries": len(chosen),
                                "before_ms": b / 1e6,
                                "after_ms": a / 1e6,
                                "ratio": b / a,
                                "paired_ratios": [
                                    total([x], chosen) / total([y], chosen)
                                    for x, y in zip(bp, ap, strict=True)
                                ],
                            }
                        )
                    for i, q in enumerate(qs):
                        b, a = total(bp, [i]), total(ap, [i])
                        results["individual"].append(
                            {
                                "campaign": campaign,
                                "index": index or "default",
                                "comparison": f"{before}/{after}",
                                "limit": limit,
                                "query": q["query"],
                                "family": q["family"],
                                "before_ms": b / 1e6,
                                "after_ms": a / 1e6,
                                "ratio": b / a,
                                "paired_ratios": [
                                    total([x], [i]) / total([y], [i])
                                    for x, y in zip(bp, ap, strict=True)
                                ],
                            }
                        )
                    bc, ac = [
                        sum(resources[p]["cpu"] for p in phases_selected)
                        for phases_selected in [bp, ap]
                    ]
                    br, ar = [
                        statistics.mean(resources[p]["rss"] for p in phases_selected)
                        / 1024
                        for phases_selected in [bp, ap]
                    ]
                    results["resources"].append(
                        {
                            "campaign": campaign,
                            "index": index or "default",
                            "comparison": f"{before}/{after}",
                            "limit": limit,
                            "before_cpu_seconds": bc,
                            "after_cpu_seconds": ac,
                            "cpu_ratio": bc / ac,
                            "before_rss_mib": br,
                            "after_rss_mib": ar,
                        }
                    )
    results["validation"] = {
        "exhaustive_audits": audits,
        "recorded_query_executions": executions,
    }
    lines = [
        "# Complete follow-up table",
        "",
        "Ratios are before/after; above 1 is faster. Totals sum per-query median wall",
        "times, pooling both rounds. Paired ratios use each round separately. These are",
        "warm single-worker measurements, not a production traffic mix. `multi_active`",
        "contains the 26 added three- to six-term cases, excluding four boundary controls.",
        "",
        "## Suite totals",
        "",
        "| Campaign | Index | Comparison | Limit | Group | Before ms | After ms | Ratio | Paired |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]

    def group_line(c):
        return f"| {c['campaign']} | {c['index']} | {c['comparison']} | {c['limit']} | {c['family']} | {c['before_ms']:.3f} | {c['after_ms']:.3f} | {c['ratio']:.3f} | {c['paired_ratios'][0]:.3f}/{c['paired_ratios'][1]:.3f} |"

    lines += [
        group_line(c)
        for c in results["groups"]
        if c["family"] in ["all", "multi_active"]
    ]
    lines += [
        "",
        "## Every family",
        "",
        "| Campaign | Index | Comparison | Limit | Family | Before ms | After ms | Ratio | Paired |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    lines += [
        group_line(c)
        for c in results["groups"]
        if c["family"] not in ["all", "multi_active"]
    ]
    lines += [
        "",
        "## Process resources",
        "",
        "CPU is process user + system time, including startup and warmups. RSS is the",
        "mean of two process maxima, including mapped index pages. Caller-thread CPU",
        "timestamps in raw query samples are not used as total query CPU.",
        "",
        "| Campaign | Index | Comparison | Limit | CPU ratio | Before RSS MiB | After RSS MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {c['campaign']} | {c['index']} | {c['comparison']} | {c['limit']} | {c['cpu_ratio']:.3f} | {c['before_rss_mib']:.2f} | {c['after_rss_mib']:.2f} |"
        for c in results["resources"]
    ]
    lines += [
        "",
        f"Verified {audits:,} exhaustive audits and {executions:,} recorded query executions (warmups excluded).",
        "",
    ]
    individual = results.pop("individual")
    (ROOT / "individual.jsonl.gz").write_bytes(
        gzip.compress(
            ("\n".join(json.dumps(c) for c in individual) + "\n").encode(), mtime=0
        )
    )
    results["largest_ranked_slowdowns"] = sorted(
        (c for c in individual if c["limit"] != 0 and c["before_ms"] >= 0.1),
        key=lambda c: c["ratio"],
    )[:60]
    if "provenance.json" in files:
        selected = js("provenance.json")["selection"]["selected_x86_variant"]
        results["selected_ranked_slowdowns"] = sorted(
            (
                c
                for c in individual
                if c["limit"] != 0
                and c["before_ms"] >= 0.1
                and (
                    (c["campaign"] == "x86" and c["comparison"] == f"head/{selected}")
                    or (c["campaign"] == "arm" and c["comparison"] == "baseline/gated")
                )
            ),
            key=lambda c: c["ratio"],
        )[:60]
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (ROOT / "table.md").write_text("\n".join(lines))
    print(json.dumps(results["validation"]))


if __name__ == "__main__":
    main()
