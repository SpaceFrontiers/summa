#!/usr/bin/env python3
"""
Merge checkpoints from multi-GPU training by averaging weights.
Usage: python merge_checkpoints.py checkpoint1.safetensors checkpoint2.safetensors -o merged.safetensors
"""

import argparse
import sys

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("Please install: pip install safetensors")
    sys.exit(1)


def merge_checkpoints(checkpoint_paths: list[str], output_path: str):
    """Average weights from multiple checkpoints."""

    if len(checkpoint_paths) < 2:
        print("Need at least 2 checkpoints to merge")
        sys.exit(1)

    print(f"Merging {len(checkpoint_paths)} checkpoints...")

    # Load first checkpoint
    merged = {}
    with safe_open(checkpoint_paths[0], framework="pt") as f:
        for key in f:
            merged[key] = f.get_tensor(key).float()

    # Add remaining checkpoints
    for path in checkpoint_paths[1:]:
        print(f"  Adding {path}")
        with safe_open(path, framework="pt") as f:
            for key in f:
                merged[key] += f.get_tensor(key).float()

    # Average
    n = len(checkpoint_paths)
    for key in merged:
        merged[key] = merged[key] / n

    # Save
    save_file(merged, output_path)
    print(f"Saved merged checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge multi-GPU checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    args = parser.parse_args()

    merge_checkpoints(args.checkpoints, args.output)


if __name__ == "__main__":
    main()
