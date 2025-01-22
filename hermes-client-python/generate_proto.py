#!/usr/bin/env python3
"""Generate Python protobuf stubs from hermes.proto."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent
    proto_dir = root.parent / "hermes-proto"
    output_dir = root / "src" / "hermes_client"

    proto_file = proto_dir / "hermes.proto"

    if not proto_file.exists():
        print(f"Error: {proto_file} not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(proto_file),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    # Fix imports in generated files
    grpc_file = output_dir / "hermes_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        content = content.replace(
            "import hermes_pb2 as hermes__pb2",
            "from . import hermes_pb2 as hermes__pb2",
        )
        grpc_file.write_text(content)

    print("Generated:")
    print(f"  - {output_dir / 'hermes_pb2.py'}")
    print(f"  - {output_dir / 'hermes_pb2_grpc.py'}")


if __name__ == "__main__":
    main()
