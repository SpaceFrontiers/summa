#!/usr/bin/env python3
"""Weights & Biases sidecar for hermes-train.

Follows a training run's schema-v2 `metrics.jsonl` and mirrors every typed event
to W&B: `WANDB_API_KEY` set
means live curves (project `hermes-retriever` unless `WANDB_PROJECT`
overrides, run name from `WANDB_NAME`); no key means the sidecar exits
quietly and training is untouched. Because it replays the file from the
beginning, attaching it mid-run (or after a preemption resume) backfills
the full history; a stable run id keeps every resume in one W&B run.

Usage: WANDB_API_KEY=... wandb_tail.py <path/to/metrics.jsonl>
"""

import json
import math
import os
import signal
import sys
import threading


def wandb_payload(record: dict) -> dict:
    """Flatten one strict schema-v2 event for W&B."""
    if not isinstance(record, dict):
        raise ValueError("metric record must be an object")
    if record.get("schema_version") != 2:
        raise ValueError("unsupported metric schema_version")
    sequence = record.get("sequence")
    global_step = record.get("global_step")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step < 0
    ):
        raise ValueError("global_step must be a non-negative integer")
    event = record.get("event")
    phase = record.get("phase")
    if not isinstance(event, dict) or not isinstance(event.get("values"), dict):
        raise ValueError("event must contain typed values")
    if not isinstance(phase, dict):
        raise ValueError("phase must be an object")
    if not isinstance(event.get("type"), str) or not event["type"]:
        raise ValueError("event type must be a non-empty string")
    if (
        not isinstance(phase.get("index"), int)
        or isinstance(phase["index"], bool)
        or phase["index"] < 0
        or not isinstance(phase.get("name"), str)
        or not phase["name"]
        or not isinstance(phase.get("kind"), str)
        or not phase["kind"]
    ):
        raise ValueError("phase coordinates are invalid")
    payload = event["values"].copy()
    payload.update(
        {
            "global_step": global_step,
            "metric_sequence": sequence,
            "event_type": event["type"],
            "phase/index": phase["index"],
            "phase/name": phase["name"],
            "phase/kind": phase["kind"],
        }
    )
    layer_norms = payload.pop("layer_gradient_norms", None)
    if layer_norms is None:
        return payload
    if not isinstance(layer_norms, list) or not layer_norms:
        raise ValueError("layer_gradient_norms must be a non-empty array")
    for index, value in enumerate(layer_norms, start=1):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"layer_gradient_norms[{index - 1}] is not finite")
        payload[f"layer_grad_norm/layer_{index}"] = value
    return payload


def remote_sequence_floor(last_history_step) -> int:
    """Return the last durable W&B sequence, preserving a fresh sequence 0."""
    if last_history_step is None:
        return -1
    if (
        not isinstance(last_history_step, int)
        or isinstance(last_history_step, bool)
        or last_history_step < -1
    ):
        raise ValueError("W&B lastHistoryStep is invalid")
    return last_history_step


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
    name = os.environ.get("WANDB_NAME", "hermes-train")
    run_id = os.environ.get("WANDB_RUN_ID", f"{name}-workflow-v2")
    run = wandb.init(
        project=project,
        name=name,
        id=run_id,
        resume="allow",
    )

    # `run.step` is the *next* client step in some SDK versions and starts at
    # zero for an empty run. The public history record is authoritative; using
    # `run.step` here would skip sequence zero on a new run and the newest
    # sequence after a resume.
    remote_run = wandb.Api().run(f"{run.entity}/{project}/{run_id}")
    try:
        last_sequence = remote_sequence_floor(remote_run.lastHistoryStep)
    except ValueError as error:
        print(f"wandb_tail: {error}", file=sys.stderr)
        run.finish()
        return 1
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
                    raw_sequence = record.get("sequence")
                    if not isinstance(raw_sequence, int) or isinstance(
                        raw_sequence, bool
                    ):
                        continue
                    sequence = raw_sequence
                    if sequence <= last_sequence:
                        continue  # already logged before a resume/backfill overlap
                    try:
                        payload = wandb_payload(record)
                    except ValueError as error:
                        print(
                            f"wandb_tail: invalid metrics at sequence {sequence}: {error}",
                            file=sys.stderr,
                        )
                        continue
                    wandb.log(payload, step=sequence)
                    last_sequence = sequence
            stop.wait(5)
    finally:
        run.finish()


if __name__ == "__main__":
    sys.exit(main())
