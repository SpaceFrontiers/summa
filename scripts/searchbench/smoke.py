#!/usr/bin/env python3
"""Exercise the real benchmark frontend: indexing, ID projection and exact counts."""

import argparse
import concurrent.futures
import http.client
import json
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--http-workers", type=int, choices=range(1, 65))
    parser.add_argument("--dispatch", choices=["pool", "blocking", "in-place"])
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="searchbench-smoke-") as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        bodies = ["alpha beta", "alpha gamma beta", "beta", "delta"]
        corpus.write_text(
            "".join(
                json.dumps({"id": f"external-{i}", "body": body}) + "\n"
                for i, body in enumerate(bodies)
            )
        )
        subprocess.run(
            [
                args.binary,
                "index-rgb" if args.rgb else "index",
                root / "index",
                corpus,
                "2",
            ],
            check=True,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [args.binary, "serve", root / "index", str(port), "2"]
        if args.http_workers is not None or args.dispatch is not None:
            command.append(str(args.http_workers or 2))
        if args.dispatch is not None:
            command.append(args.dispatch)
        server = subprocess.Popen(command)

        def request(query="alpha", family="high_term", limit=0):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                connection.request(
                    "POST",
                    "/search",
                    json.dumps({"query": query, "class": family, "limit": limit}),
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    status, count = request()
                    break
                except ConnectionRefusedError:
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("server did not start") from None
                    time.sleep(0.1)
            assert status == 200 and count == {"docs": [], "found": 2}, count
            status, ranked = request(limit=10)
            assert status == 200 and "found" not in ranked, ranked
            assert {doc["id"] for doc in ranked["docs"]} == {"external-0", "external-1"}
            assert request('"alpha beta"', "high_phrase")[1]["found"] == 1
            assert request('"alpha beta"~4', "high_sloppy_phrase")[1]["found"] == 2
            assert request("al*", "prefix3")[1]["found"] == 2
            assert request("a*a", "wildcard")[1]["found"] == 2
            assert request("a*a", "high_term")[1]["found"] == 2
            assert request("*ta", "wildcard_lead")[1]["found"] == 4
            assert request("b?ta", "wildcard_scan")[1]["found"] == 3
            assert request(".*", "regex")[0] == 400
            assert request(limit=1000)[0] == 400
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                assert all(
                    value == (200, count)
                    for value in pool.map(lambda _: request(), range(64))
                )
            if args.diagnostics:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                connection.request("GET", "/diagnostics")
                response = connection.getresponse()
                diagnostic = json.loads(response.read())
                connection.close()
                assert response.status == 200, diagnostic
                assert diagnostic["successful_requests"] >= 64, diagnostic
                assert diagnostic["failed_workers"] >= 1, diagnostic
                assert diagnostic["stages"]["core_pool_queue"]["sum_ns"] > 0, diagnostic
                assert diagnostic["stages"]["blocking_return"]["sum_ns"] > 0, diagnostic
            server.send_signal(signal.SIGINT)
            assert server.wait(timeout=15) == 0
            print(
                "PASS: exact count, external IDs, ranked no-count, phrase/slop/prefix/wildcard, request errors, concurrent search, shutdown"
            )
        finally:
            if server.poll() is None:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
