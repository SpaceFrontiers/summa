import collections
import gzip
import hashlib
import json
import pathlib
import re
import sys

P = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
records, audits, resources = [], [], []


def campaign(name):
    if re.match(r"arm-bmp-\d", name):
        return "arm/bmp-a"
    if name.startswith("arm-bmp-b-"):
        return "arm/bmp-b"
    if name.startswith("arm-controls-d-"):
        return "arm/bmp-d"
    if name.startswith("arm-controls-e-"):
        return "arm/bmp-e"
    if name.startswith("arm-controls-b"):
        return "arm/bmp-c"
    if name.startswith("arm-scann_vectors-"):
        return "arm/scann"
    if name.startswith("arm-vector_indexing-"):
        return "arm/tq"
    if name.startswith("arm-binary-final-"):
        return "arm/binary-final"
    if name.startswith("arm-binary-round"):
        return "arm/binary-b"
    if name.startswith("x86-binary-final-"):
        return "x86/binary-final"
    if name.startswith("x86-binary-round"):
        return "x86/binary-c"
    if name.startswith("x86-scann-"):
        return "x86/scann"
    if name.startswith("x86-real-bmp-"):
        return "x86/bmp-c-real"
    return None


for folder in [P / "results", P / "x86-final/results"]:
    for path in sorted(folder.iterdir()):
        group = campaign(path.name)
        if not group:
            continue
        variant = "baseline" if "baseline" in path.name else "candidate"
        meta = {"campaign": group, "variant": variant, "run": path.name}
        log = path / "run.log" if path.is_dir() else path
        # Remote Criterion logs live alongside their output directory.
        if not log.exists():
            log = path.with_suffix(".log")
        if path.is_file() and path.with_suffix("").is_dir():
            continue
        text = log.read_text()
        rss = re.search(r"(\d+)\s+maximum resident set size", text)
        linux_rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        if rss or linux_rss:
            resources.append(
                {**meta, "rss_bytes": int(rss[1]) if rss else int(linux_rss[1]) * 1024}
            )
        for line in text.splitlines():
            if "BMP_AUDIT " in line:
                fields = dict(
                    item.split("=") for item in line.split("BMP_AUDIT ")[1].split()
                )
                audits.append({**meta, **fields})
            elif "BINARY_SCAN " in line:
                value = json.loads(line.split("BINARY_SCAN ")[1])
                records.append(
                    {
                        **meta,
                        "kind": "binary",
                        **value,
                        "case": f"{value['width'] * 8} bits / probes {value['probes']} / k{value['k']}",
                    }
                )
            elif line.startswith("{") and '"hits"' in line:
                value = json.loads(line)
                hits = value.pop("hits")
                value["hits_hash"] = hashlib.sha256(
                    json.dumps(hits, separators=(",", ":")).encode()
                ).hexdigest()
                value["hit_count"] = len(hits)
                records.append(
                    {
                        **meta,
                        "kind": "replay",
                        **value,
                        "case": f"k{value['k']} / gamma {value['gamma']}",
                    }
                )
        if path.is_dir():
            for sample in sorted(path.glob("criterion/**/new/sample.json")):
                estimates = json.loads(sample.with_name("estimates.json").read_text())
                case = str(sample.parent.parent.relative_to(path / "criterion"))
                shape = re.search(r"-b(\d+)-q\d+-", path.name)
                if shape:
                    case = f"block{shape[1]} / " + case
                records.append(
                    {
                        **meta,
                        "kind": "criterion",
                        "case": case,
                        "sample": json.loads(sample.read_text()),
                        "median_ns": estimates["median"]["point_estimate"],
                    }
                )

data = {
    "baseline_commit": "9ea50c169ec7c2b9782f99e9533e8ccf8e463c72",
    "records": records,
    "audits": audits,
    "resources": resources,
}
with (OUT / "evidence.json.gz").open("wb") as f:
    f.write(
        gzip.compress(
            json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0
        )
    )
print(collections.Counter((x["campaign"], x["kind"]) for x in records))
