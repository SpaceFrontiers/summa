#!/usr/bin/env python3
"""Weights & Biases sidecar for hermes-train.

Follows a training run's `metrics.jsonl` and mirrors every optimizer step
to W&B — the same contract the pre-Burn trainer had: `WANDB_API_KEY` set
means live curves (project `hermes-retriever` unless `WANDB_PROJECT`
overrides, run name from `WANDB_NAME`); no key means the sidecar exits
quietly and training is untouched. Because it replays the file from the
beginning, attaching it mid-run (or after a preemption resume) backfills
the full history; a stable run id keeps every resume in one W&B run.

Usage: WANDB_API_KEY=... wandb_tail.py <path/to/metrics.jsonl>
"""

import json
import os
import sys
import time


def main() -> int:
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb_tail: WANDB_API_KEY not set; exiting (training unaffected)")
        return 0
    if len(sys.argv) != 2:
        print("usage: wandb_tail.py <metrics.jsonl>", file=sys.stderr)
        return 2
    path = sys.argv[1]

    import wandb  # deferred so a missing package never blocks training setup

    name = os.environ.get("WANDB_NAME", "retriever-100m")
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "hermes-retriever"),
        name=name,
        id=os.environ.get("WANDB_RUN_ID", f"{name}-stage1"),
        resume="allow",
    )

    last_step = run.step or 0
    position = 0
    while True:
        if not os.path.exists(path):
            time.sleep(5)
            continue
        with open(path, encoding="utf-8") as handle:
            handle.seek(position)
            while True:
                line = handle.readline()
                if not line:
                    position = handle.tell()
                    break
                if not line.endswith("\n"):
                    # Partial write: re-read this line on the next pass.
                    break
                position = handle.tell()
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = int(record.get("step", 0))
                if step <= last_step:
                    continue  # already logged before a resume/backfill overlap
                wandb.log(record, step=step)
                last_step = step
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
