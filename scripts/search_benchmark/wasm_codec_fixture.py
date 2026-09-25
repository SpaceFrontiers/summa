#!/usr/bin/env python3
"""Generate the WASM decoder's native-written index fixture via the public writer.

Every fixture pairs a control index with a variant built from the same corpus:
the "rounded" control uses the rounded codec with --posting-ratio-bounds, the
variant swaps the codec (simd4x, ratio bounds) or the bound layout (impacts,
--posting-impact-bounds). Counts must agree; the tests compare scores.
"""

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--impact-bounds",
        action="store_true",
        help="compare rounded ratio and impact layouts",
    )
    parser.add_argument(
        "--compact-text",
        action="store_true",
        help="write compact directories and byte norms in both indexes",
    )
    parser.add_argument("--documents", type=int, default=389)
    args = parser.parse_args()
    if not 1 <= args.documents <= 1_000_000:
        parser.error("documents must be between 1 and 1000000")
    binary = args.binary.resolve()
    corpus = "".join(
        json.dumps(
            {
                "id": str(i),
                "text": "alpha beta " * (i % 11 + 1)
                + "padding " * (i % 71)
                + ("rare" if i % 7 == 0 else ""),
                "sort_field": i,
            }
        )
        + "\n"
        for i in range(args.documents)
    )
    queries = [
        "alpha",
        "rare",
        "alpha AND rare",
        "alpha OR rare",
        '"alpha beta"',
        '"beta padding"',
    ]
    fixture = {
        "generator": "scripts/search_benchmark/wasm_codec_fixture.py",
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "documents": args.documents,
        "queries": queries,
        "indexes": {},
    }
    with tempfile.TemporaryDirectory(prefix="summa-wasm-codec-") as tmp:
        for codec in (
            ["rounded", "impacts"] if args.impact_bounds else ["rounded", "simd4x"]
        ):
            index = Path(tmp) / codec
            subprocess.run(
                [
                    str(binary),
                    "index",
                    str(index),
                    "--posting-codec",
                    "rounded" if codec == "impacts" else codec,
                    "--indexing-threads",
                    "1",
                    "--indexing-memory-bytes",
                    "16777216",
                    "--posting-impact-bounds"
                    if codec == "impacts"
                    else "--posting-ratio-bounds",
                    "--term-dict-block-bytes",
                    "512",
                    *(
                        ["--compact-text", "--quantized-norms"]
                        if args.compact_text
                        else []
                    ),
                ],
                input=corpus,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            result = subprocess.run(
                [
                    str(binary),
                    "serve",
                    str(index),
                ],
                input="".join(f"VERIFY\t{q}\n" for q in queries),
                text=True,
                capture_output=True,
                check=True,
            )
            counts = [int(line) for line in result.stdout.splitlines()]
            assert len(counts) == len(queries)
            files = {}
            hashes = {}
            for path in sorted(index.iterdir()):
                if not path.is_file() or path.suffix == ".lock":
                    continue
                data = path.read_bytes()
                files[path.name] = base64.b64encode(data).decode()
                hashes[path.name] = hashlib.sha256(data).hexdigest()
            fixture["indexes"][codec] = {
                "files": files,
                "sha256": hashes,
                "counts": counts,
            }
    assert (
        fixture["indexes"]["rounded"]["counts"]
        == fixture["indexes"]["impacts" if args.impact_bounds else "simd4x"]["counts"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
