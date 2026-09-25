#!/usr/bin/env python3
"""Summa comparison orchestration using the pinned Searchbench native replay.

No timing loop or REST request translation is reimplemented here. Count probes
retain all exclusions. Exact-count comparisons require equal counts; explicitly
labelled shared-input comparisons retain count differences and validate each
engine against its own probe. Raw wire blobs and replay/memory data are retained.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ENGINES = ("summa", "elasticsearch", "opensearch", "luxir")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def key(item):
    return f"{item.query_class}\t{item.text}"


def install_adapter():
    from adapters import BaseAdapter

    class SummaAdapter(BaseAdapter):
        _fail_rules = (("present", b'"error"'), ("absent", b'"docs":'))

        def build(self, params, item):
            return self.encode(
                "POST",
                "/search",
                {
                    "query": item.text,
                    "class": item.query_class,
                    "limit": params["limit"],
                },
                params["name"],
                item.text,
            )

        def validate(self, params, raw, item=None):
            value = json.loads(raw)
            if "error" in value:
                raise RuntimeError(value["error"])
            docs = value.get("docs")
            if not isinstance(docs, list) or len(docs) > params["limit"]:
                raise RuntimeError("invalid docs response")
            if any(not isinstance(doc.get("id"), str) for doc in docs):
                raise RuntimeError("missing external ID")
            if len({doc["id"] for doc in docs}) != len(docs):
                raise RuntimeError("duplicate external IDs")
            if params["limit"] == 0:
                found = value.get("found")
                if type(found) is not int or found < 0:
                    raise RuntimeError("missing exact count")
                return found
            if "found" in value:
                raise RuntimeError("ranked response unexpectedly counted")
            return None

    return SummaAdapter("searchbench")


def parameters(item, operation):
    from presets import resolve

    return dict(resolve(operation, "exact"), name=key(item))


async def probe(args, adapter, items):
    """Record a result for every selected query, including unsupported errors."""
    connection = adapter.connect(args.host, args.port, args.timeout)
    rows = []
    try:
        for item in items:
            row = {"key": key(item), "query": item.text, "class": item.query_class}
            try:
                params = parameters(item, "COUNT")
                request = adapter.build(params, item)
                raw = await connection.request(request)
                row["count"] = adapter.validate(params, raw, item)
            except Exception as error:
                row["error"] = repr(error)
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"{args.engine}: probed {len(rows)}/{len(items)}", flush=True)
    finally:
        await connection.close()
    write(args.out / f"{args.engine}-counts.json", rows)


def gate(args, items):
    comparison = getattr(args, "comparison", "exact-count")
    counts = {
        engine: {
            row["key"]: row
            for row in json.loads((args.out / f"{engine}-counts.json").read_text())
        }
        for engine in ENGINES
    }
    rows = []
    for item in items:
        observed = {engine: counts[engine][key(item)] for engine in ENGINES}
        values = [row.get("count") for row in observed.values()]
        successful = all(type(value) is int for value in values)
        same = successful and len(set(values)) == 1
        # Actual errors and counts own eligibility; a static operator blacklist
        # would keep newly supported query types out of the comparison forever.
        include = successful if comparison == "shared-input" else same
        reason = "count disagreement/error" if not include else None
        rows.append(
            {
                "key": key(item),
                "query": asdict(item),
                "include": reason is None,
                "comparison": comparison,
                "counts_agree": same,
                "reason": reason,
                "observed": observed,
            }
        )
    write(args.out / "agreement.json", rows)
    print(
        json.dumps(
            {
                "total": len(rows),
                "comparison": comparison,
                "included": sum(row["include"] for row in rows),
                "counts_agree": sum(row["counts_agree"] for row in rows),
                "included_by_class": dict(
                    Counter(
                        row["query"]["query_class"] for row in rows if row["include"]
                    )
                ),
            },
            indent=2,
        )
    )


def topology(args, adapter):
    from adapters import Request
    from topology_probe import assert_quiescent_topology, index_topology

    if args.engine != "summa":
        result = index_topology(adapter, args.host, args.port, args.timeout)
        assert_quiescent_topology(result, 1)
        if result["live_docs"] != args.documents:
            raise RuntimeError(f"wrong reference document count: {result}")
        return result

    async def fetch():
        connection = adapter.connect(args.host, args.port, args.timeout)
        try:
            return json.loads(
                await connection.request(Request("GET", "/stats", b"", "stats", ""))
            )
        finally:
            await connection.close()

    result = asyncio.run(fetch())
    if result["documents"] != args.documents or len(result["segments"]) != 1:
        raise RuntimeError(f"wrong Summa topology: {result}")
    return result


class PairedAdapter:
    """Add corpus-specific hit-count/ID assertions to upstream validation."""

    def __init__(self, adapter, counts):
        self.adapter = adapter
        self.counts = counts

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def validate(self, params, raw, item=None):
        found = self.adapter.validate(params, raw, item)
        expected = self.counts[key(item)]
        if params["limit"] == 0:
            if found != expected:
                raise RuntimeError(f"exact count changed: {found} != {expected}")
        else:
            if hasattr(self.adapter, "query_results"):
                value = {
                    "docs": [
                        doc
                        for result in self.adapter.query_results(raw)
                        for doc in result["docs"]
                    ]
                }
            else:
                value = json.loads(raw)
            if "hits" in value:
                docs = value["hits"]["hits"]
                ids = [doc.get("fields", {}).get("id", []) for doc in docs]
                if any(
                    len(values) != 1 or not isinstance(values[0], str) for values in ids
                ):
                    raise RuntimeError("missing REST external ID")
                ids = [values[0] for values in ids]
            else:
                docs = value["docs"]
                ids = [doc["id"] for doc in docs]
            if len(docs) != min(expected, params["limit"]) or len(set(ids)) != len(ids):
                raise RuntimeError("ranked response has wrong number of unique hits")
        return found


def eligible_counts(rows, engine):
    """Validate a timed engine against its own successful untimed observations."""
    return {
        row["key"]: row["observed"][engine]["count"] for row in rows if row["include"]
    }


def measure(args, adapter, items):
    from driver import (
        host_config,
        materialize_workload,
        replay_metrics,
        run_fixed,
        run_replay,
        server_session,
        source_file_config,
    )
    from sampler import ProcSampler

    agreement = json.loads((args.out / "agreement.json").read_text())
    selected = eligible_counts(agreement, args.engine)
    adapter = PairedAdapter(adapter, selected)
    shared_input = any(row.get("comparison") == "shared-input" for row in agreement)
    items = [item for item in items if key(item) in selected]
    if args.families:
        requested = set(args.families)
        missing = requested - {item.query_class for item in items}
        if missing:
            raise RuntimeError(f"no eligible queries for families: {sorted(missing)}")
        items = [item for item in items if item.query_class in requested]
    if not items:
        raise RuntimeError("no comparable queries")
    directory = args.out / args.engine
    directory.mkdir(exist_ok=False)
    before = topology(args, adapter)
    write(
        directory / "context.json",
        {
            "args": vars(args),
            "host": host_config(args.server_pid),
            "session": server_session(args.server_pid),
            "topology": before,
            "server_binary": source_file_config(
                Path(f"/proc/{args.server_pid}/exe").resolve()
            ),
            "agreement_sha256": hashlib.sha256(
                (args.out / "agreement.json").read_bytes()
            ).hexdigest(),
            "comparison_modes": sorted(
                {row.get("comparison", "unspecified") for row in agreement}
            ),
            "queries": [asdict(item) for item in items],
        },
    )
    common = {
        "host": args.host,
        "port": args.port,
        "threads": 2,
        "client_cores": args.client_cores,
        "order": "shuffle",
        "warmup_seconds": 1,
        "max_requests": 0,
        "warmup_requests": 0,
        "timeout": args.timeout,
        "server_pid": args.server_pid,
        "seed": 42,
        "duration": args.duration,
        "repetitions": args.repetitions,
    }
    # Session warmup uses the same selected queries and all three operations.
    warmwork = [
        (parameters(item, operation), item)
        for item in items
        for operation in ("TOP_10", "TOP_100", "COUNT")
    ]
    blob, _, _ = materialize_workload(
        warmwork, adapter, args.host, args.port, directory / "warmup.sbwl"
    )
    warmargs = SimpleNamespace(
        **(
            common
            | {"concurrency": 8, "duration": args.session_warmup, "repetitions": 1}
        )
    )
    warm = run_replay(
        args.searchbench,
        warmargs,
        adapter,
        "top_docs",
        blob,
        directory / "warmup.raw.json",
    )
    if warm["errors"]:
        raise RuntimeError(f"session warmup failed: {warm['errors']}")
    for clients in args.clients:
        for family in sorted({item.query_class for item in items}):
            family_items = [item for item in items if item.query_class == family]
            for operation in ("TOP_10", "TOP_100", "COUNT"):
                name = f"{family}-{operation}-c{clients}"
                params = dict(parameters(family_items[0], operation), name=name)
                # Preserve a bucket per query in the broader comparison: count
                # outliers must remain visible rather than disappearing into a
                # family aggregate. The request bodies and replay order stay
                # under the same upstream adapter/driver.
                workload = [
                    (dict(params, name=key(item)) if shared_input else params, item)
                    for item in family_items
                ]
                _, errors, _, _, responses = asyncio.run(
                    run_fixed(
                        args.host, args.port, args.timeout, 8, adapter, workload, True
                    )
                )
                if errors:
                    write(directory / f"{name}.validation-errors.json", errors)
                    raise RuntimeError(f"{name}: validation failed")
                blob, labels, requests = materialize_workload(
                    workload, adapter, args.host, args.port, directory / f"{name}.sbwl"
                )
                sampler = ProcSampler(
                    args.server_pid, str(directory / f"{name}.memory.csv")
                )
                sampler.start()
                try:
                    raw = run_replay(
                        args.searchbench,
                        SimpleNamespace(
                            **(
                                common
                                | {"concurrency": clients, "threads": min(2, clients)}
                            )
                        ),
                        adapter,
                        "top_docs",
                        blob,
                        directory / f"{name}.raw.json",
                    )
                finally:
                    memory = sampler.stop()
                reps = [
                    replay_metrics(rep["overall"], rep["elapsed_s"])
                    for rep in raw["repetitions"]
                ]
                result = {
                    "engine": args.engine,
                    "family": family,
                    "operation": operation,
                    "clients": clients,
                    "query_count": len(family_items),
                    "parameters": params,
                    "repetitions": reps,
                    "median_qps": statistics.median(rep["qps"] for rep in reps),
                    "memory": memory,
                    "errors": raw["errors"],
                    "requests": requests,
                    "responses": responses,
                    "replay_executable": raw["executable"],
                }
                write(directory / f"{name}.json", result)
                if (
                    raw["errors"]
                    or any(rep["errors"] for rep in reps)
                    or memory["error"]
                ):
                    raise RuntimeError(f"{name}: timing or sampler errors")
                if raw["workload"]["buckets"] != {
                    str(i): label for i, label in enumerate(labels)
                }:
                    raise RuntimeError(f"{name}: replay bucket mismatch")
                print(
                    f"{args.engine} {name}: {result['median_qps']:.1f} QPS", flush=True
                )
    after = topology(args, adapter)
    write(directory / "topology-after.json", after)
    if before != after:
        raise RuntimeError("topology changed during measurement")
    write(directory / "complete.json", {"complete": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("probe", "gate", "measure"))
    parser.add_argument("--searchbench", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--engine", choices=ENGINES, default="summa")
    parser.add_argument(
        "--comparison",
        choices=("exact-count", "shared-input"),
        default="exact-count",
        help="gate by equal counts, or by successful execution of identical inputs",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9401)
    parser.add_argument("--server-pid", type=int, default=0)
    parser.add_argument("--documents", type=int, default=10_000_000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--session-warmup", type=float, default=30)
    parser.add_argument(
        "--families", nargs="+", help="measure only these eligible families"
    )
    parser.add_argument("--clients", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--client-cores", default="3,7")
    args = parser.parse_args()
    args.searchbench = args.searchbench.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.searchbench / "python"))
    from adapters import make_adapter
    from query_source import read_queries

    items = list(read_queries(args.searchbench / "queries/luceneutil/queries.txt"))
    if args.operation == "gate":
        gate(args, items)
        return
    adapter = (
        install_adapter()
        if args.engine == "summa"
        else make_adapter(args.engine, "searchbench")
    )
    topology(args, adapter)
    if args.operation == "probe":
        asyncio.run(probe(args, adapter, items))
    else:
        os.sched_setaffinity(0, {int(cpu) for cpu in args.client_cores.split(",")})
        measure(args, adapter, items)


if __name__ == "__main__":
    main()
