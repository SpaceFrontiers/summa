"""Probe query acceptance through the real HTTP adapter, without timing claims."""

import argparse
import collections
import concurrent.futures
import hashlib
import http.client
import json
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(binary, output):
    queries = HERE / "queries.jsonl"
    rows = [json.loads(line) for line in queries.read_text().splitlines()]
    assert len(rows) == 826
    before = json.loads((HERE / "before.json").read_text())
    assert hashlib.sha256(queries.read_bytes()).hexdigest() == before["queries_sha256"]
    with tempfile.TemporaryDirectory(prefix="summa-query-support-") as temporary:
        directory = Path(temporary)
        corpus = directory / "corpus.jsonl"
        corpus.write_text(
            "".join(
                json.dumps({"id": str(i), "body": text}) + "\n"
                for i, text in enumerate(
                    ["alpha beta", "alpha gamma beta", "beta", "delta"]
                )
            )
        )
        subprocess.run(
            [str(binary), "index", str(directory / "index"), str(corpus), "2"],
            check=True,
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryFile() as log:
            server = subprocess.Popen(
                [str(binary), "serve", str(directory / "index"), str(port), "2"],
                stdout=log,
                stderr=log,
            )
            try:
                for _ in range(100):
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                            break
                    except OSError:
                        if server.poll() is not None:
                            log.seek(0)
                            raise RuntimeError(
                                log.read().decode(errors="replace")
                            ) from None
                        time.sleep(0.1)
                else:
                    raise RuntimeError("server startup timed out")

                def probe(row):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", port, timeout=20
                    )
                    try:
                        connection.request(
                            "POST",
                            "/search",
                            json.dumps(dict(row, limit=0)),
                            {"Content-Type": "application/json"},
                        )
                        response = connection.getresponse()
                        return {
                            "class": row["class"],
                            "query": row["query"],
                            "status": response.status,
                            "response": json.loads(response.read()),
                        }
                    finally:
                        connection.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(probe, rows))
                counts = collections.Counter(str(row["status"]) for row in results)
                report = {
                    "scope": "Four-document HTTP capability fixture; not 10M corpus counts, throughput or cross-engine parity",
                    "counts": counts,
                    "results": results,
                    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "queries_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
                }
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(dict(counts))
                assert counts == {"200": 826}, "query errors retained in output"
                for old, new in zip(before["results"], results, strict=True):
                    assert (old["class"], old["query"]) == (new["class"], new["query"])
                    if old["status"] == 200:
                        assert old["response"] == new["response"], new["query"]
                print("All 801 previously accepted responses are unchanged")
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=HERE.parents[2] / "target/debug/examples/searchbench_http",
    )
    parser.add_argument("--output", type=Path, default=HERE / "after.json")
    args = parser.parse_args()
    verify(args.binary.resolve(), args.output)
