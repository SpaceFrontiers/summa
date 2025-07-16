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
import signal
import sys
import threading


def main() -> int:
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb_tail: WANDB_API_KEY not set; exiting (training unaffected)")
        return 0
    if len(sys.argv) != 2:
        print("usage: wandb_tail.py <metrics.jsonl>", file=sys.stderr)
        return 2
    path = sys.argv[1]

    import wandb  # deferred so a missing package never blocks training setup

    project = os.environ.get("WANDB_PROJECT", "hermes-retriever")
    name = os.environ.get("WANDB_NAME", "retriever-100m")
    run_id = os.environ.get("WANDB_RUN_ID", f"{name}-stage1")
    run = wandb.init(
        project=project,
        name=name,
        id=run_id,
        resume="allow",
    )

    # `run.step` starts from zero in some W&B SDK versions even when attaching
    # to an existing run. The public run record is authoritative and prevents
    # a resumed reporter from attempting to emit thousands of duplicate steps.
    remote_run = wandb.Api().run(f"{run.entity}/{project}/{run_id}")
    last_step = max(run.step or 0, remote_run.lastHistoryStep or 0)
    position = 0
    identity = None
    stop = threading.Event()

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop.is_set():
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                stop.wait(5)
                continue
            current_identity = (stat.st_dev, stat.st_ino)
            if current_identity != identity or stat.st_size < position:
                # A restore may atomically replace or truncate metrics.jsonl.
                # Re-read it; last_step filters the overlapping history.
                identity = current_identity
                position = 0
            with open(path, encoding="utf-8") as handle:
                handle.seek(position)
                while True:
                    line_position = handle.tell()
                    line = handle.readline()
                    if not line:
                        position = handle.tell()
                        break
                    if not line.endswith("\n"):
                        # Partial write: re-read this line on the next pass.
                        position = line_position
                        break
                    position = handle.tell()
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw_step = record.get("step")
                    if not isinstance(raw_step, int) or isinstance(raw_step, bool):
                        continue
                    step = raw_step
                    if step <= last_step:
                        continue  # already logged before a resume/backfill overlap
                    wandb.log(record, step=step)
                    last_step = step
            stop.wait(5)
    finally:
        run.finish()


if __name__ == "__main__":
    sys.exit(main())
