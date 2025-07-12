"""CLI mirroring the old hermes-llm train/train-tokenizer/dpo subcommands.

Model configs are JSON exported from MAL: `hermes-llm export --model <name|.mal>`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_train(args: argparse.Namespace) -> int:
    from hermes_train.config import ModelDef
    from hermes_train.data import DataLoader, Dataset
    from hermes_train.tokenizer import Tokenizer, train_bpe
    from hermes_train.train import Trainer

    config = ModelDef.from_json(args.config)

    data_files = args.data or []

    if Path(args.tokenizer).exists():
        tokenizer = Tokenizer.from_file(args.tokenizer)
    elif data_files:
        print("Tokenizer not found, training a new one...")
        tokenizer = train_bpe([data_files[0]], args.tokenizer)
    else:
        print("error: --data required to train a tokenizer", file=sys.stderr)
        return 1
    config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Each -d file is a curriculum stage, trained sequentially under one
    # LR schedule (WSD keeps this sane: decay only hits the final stage).
    if data_files:
        datasets = [Dataset.from_file(p, tokenizer, args.seq_len) for p in data_files]
    else:
        print("Loading dataset from stdin...")
        datasets = [Dataset.from_stdin(tokenizer, args.seq_len)]
    for i, ds in enumerate(datasets):
        print(f"Stage {i + 1}: {len(ds.tokens)} tokens")

    trainer = Trainer(
        config,
        lr=args.lr,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        schedule=args.schedule,
        doc_masking=not args.no_doc_masking,
    )
    loaders = [
        DataLoader(
            ds,
            args.batch_size,
            shuffle=True,
            rank=trainer.rank,
            world_size=trainer.world_size,
        )
        for ds in datasets
    ]
    total_steps = (
        sum(loader.num_batches() for loader in loaders) // args.grad_accum
    ) * args.epochs

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    resume_state = None
    if args.resume:
        if len(loaders) > 1:
            print(
                "warning: --resume with multiple stages restarts from stage 1 "
                "(cross-stage positions are not checkpointed)",
                file=sys.stderr,
            )
        resume_state = trainer.load_training_state(output)
        if resume_state is None:
            print("No resumable state found, starting fresh")
    elif args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    if args.freeze_layers > 0:
        trainer.freeze_layers(args.freeze_layers)

    if trainer.is_main:
        # Persist the config with vocab_size synced to the tokenizer, like the
        # old Rust trainer did — hermes-llm generate reads this file.
        import json

        raw = json.loads(Path(args.config).read_text())
        raw["vocab_size"] = tokenizer.vocab_size
        (output / "config.json").write_text(json.dumps(raw, indent=2))

    completed = True
    for stage, loader in enumerate(loaders):
        if len(loaders) > 1:
            print(f"=== Stage {stage + 1}/{len(loaders)} ===")
        completed = trainer.train(
            loader,
            args.epochs,
            checkpoint_dir=output,
            resume_state=resume_state if stage == 0 else None,
            total_steps=total_steps,
            max_steps=args.max_steps,
            save_every=args.save_every,
        )
        if not completed:
            break
    if completed:
        if trainer.is_main:
            trainer.save_training_state(output, epoch=args.epochs, batch_position=0)
            print("Training complete!")
    else:
        print(
            f"Training interrupted, resume with: hermes-train train --resume --output {output} ..."
        )
    return 0


def cmd_train_tokenizer(args: argparse.Namespace) -> int:
    from hermes_train.tokenizer import train_bpe

    tokenizer = train_bpe(args.input, args.output, vocab_size=args.vocab_size)
    print(
        f"Tokenizer trained and saved to {args.output} (vocab size: {tokenizer.vocab_size})"
    )
    return 0


def cmd_dpo(args: argparse.Namespace) -> int:
    from hermes_train.config import ModelDef
    from hermes_train.dpo import DpoTrainer, PreferenceDataset
    from hermes_train.tokenizer import Tokenizer
    from hermes_train.train import pick_device

    config = ModelDef.from_json(args.config)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    dataset = PreferenceDataset.from_file(args.data)

    trainer = DpoTrainer(
        config,
        args.checkpoint,
        device=pick_device(),
        lr=args.lr,
        beta=args.beta,
        max_len=args.max_len,
    )
    trainer.train(
        dataset, tokenizer, args.epochs, args.batch_size, output_dir=args.output
    )
    print("DPO training complete!")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-train", description="Train Hermes LLMs (PyTorch)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Pretrain or fine-tune a model")
    p.add_argument(
        "--config", required=True, help="Model config JSON (from `hermes-llm export`)"
    )
    p.add_argument("-t", "--tokenizer", required=True, help="Path to tokenizer.json")
    p.add_argument(
        "-d",
        "--data",
        action="append",
        help="Training JSONL (.gz/.zst ok); repeat for curriculum stages "
        "trained sequentially; stdin if omitted",
    )
    p.add_argument(
        "--schedule",
        choices=["wsd", "cosine"],
        default="wsd",
        help="LR schedule (default: warmup-stable-decay)",
    )
    p.add_argument(
        "--no-doc-masking",
        action="store_true",
        help="Let attention/SSM state cross packed-document boundaries",
    )
    p.add_argument("-o", "--output", default="checkpoints")
    p.add_argument("-b", "--batch-size", type=int, default=32)
    p.add_argument("-e", "--epochs", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--max-steps", type=int, help="Stop after N optimizer steps")
    p.add_argument(
        "--save-every",
        type=int,
        help="Snapshot weights.safetensors every N optimizer steps (atomic; "
        "lets a separate process eval mid-run without interrupting training)",
    )
    p.add_argument("--checkpoint", help="Checkpoint to fine-tune from")
    p.add_argument("--freeze-layers", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Resume interrupted training")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-tokenizer", help="Train a BPE tokenizer")
    p.add_argument("-i", "--input", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-v", "--vocab-size", type=int, default=32000)
    p.set_defaults(func=cmd_train_tokenizer)

    p = sub.add_parser("dpo", help="Direct Preference Optimization")
    p.add_argument("--config", required=True)
    p.add_argument("-t", "--tokenizer", required=True)
    p.add_argument(
        "-d", "--data", required=True, help="JSONL with prompt/chosen/rejected"
    )
    p.add_argument(
        "-c", "--checkpoint", required=True, help="SFT checkpoint to start from"
    )
    p.add_argument("-o", "--output", default="checkpoints-dpo")
    p.add_argument("-b", "--batch-size", type=int, default=4)
    p.add_argument("-e", "--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=512)
    p.set_defaults(func=cmd_dpo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
