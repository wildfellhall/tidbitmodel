#!/usr/bin/env python3
"""Provisional tidbitmodel pipeline (Python 3.10+).

    python -m pip install -r requirements.txt
    python main.py prepare
    python main.py train
    python main.py test --model-dir runs/tidbit/best
    python main.py generate --model-dir runs/tidbit/best --prompt "A telescope is"
    python -m unittest discover -s tests -v

Use each subcommand's --help for settings. Preparation reads corpus.py without
executing it, downloads Gutenberg texts through a bulk mirror, splits whole
books, and trains a fresh tokenizer. The default model has about 10.1M total
parameters at an 8,192-token vocabulary: 6 layers, width 320, 5 heads, and a
256-token context. All weights are initialized from scratch.

Training prints and saves sample responses every 100 optimizer steps. Runs
include validation metrics, resumable checkpoints, the best validation model,
and a final model. Test books are used only by the explicit test command.
The language-model objective learns text continuations; it does not establish
the accuracy or present-day applicability of the historical source material.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parser():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = cli.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Parse corpus.py, download/clean books, split and tokenize.",
        formatter_class=formatter,
    )
    prepare.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus.py",
        help="Python file containing a literal sources list; local .txt paths also work.",
    )
    prepare.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/prepared"),
        help="New directory for the prepared corpus.",
    )
    prepare.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw"),
        help="Reusable plain-text download cache.",
    )
    prepare.add_argument(
        "--mirror",
        default="https://mirror.cs.odu.edu/gutenberg-epub",
        help="Base URL of a Gutenberg generated-text mirror.",
    )
    prepare.add_argument(
        "--download-delay",
        type=float,
        default=1.0,
        help="Seconds between uncached downloads.",
    )
    prepare.add_argument(
        "--max-sources",
        type=int,
        default=0,
        help="Use only the first N sources; zero uses all sources.",
    )
    prepare.add_argument(
        "--skip-errors",
        action="store_true",
        help="Explicitly permit preparing an incomplete corpus when some sources fail.",
    )
    prepare.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Target vocabulary size, including special tokens.",
    )
    prepare.add_argument(
        "--validation-fraction",
        type=float,
        default=0.05,
        help="Fraction of distinct books reserved for validation.",
    )
    prepare.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Fraction of distinct books reserved for final testing.",
    )
    prepare.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducible book splits."
    )

    train = commands.add_parser(
        "train",
        help="Train a randomly initialized GPT-2; print samples every 100 updates.",
        formatter_class=formatter,
    )
    train.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/prepared"),
        help="Prepared corpus directory.",
    )
    train.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/tidbit"),
        help="Run directory for metrics, samples, and checkpoints.",
    )
    train.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Total optimizer updates, including any already completed on resume.",
    )
    train.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sequences per accumulation microbatch.",
    )
    train.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Microbatches per optimizer update.",
    )
    train.add_argument(
        "--context", type=int, default=256, help="Maximum sequence length in tokens."
    )
    train.add_argument(
        "--layers", type=int, default=6, help="Transformer decoder layers."
    )
    train.add_argument(
        "--width",
        type=int,
        default=320,
        help="Hidden width; must be divisible by heads.",
    )
    train.add_argument(
        "--heads", type=int, default=5, help="Attention heads per layer."
    )
    train.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Attention, embedding, and residual dropout.",
    )
    train.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Peak AdamW learning rate."
    )
    train.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW decay for matrix parameters.",
    )
    train.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Warmup before cosine learning-rate decay.",
    )
    train.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Optimizer steps between validation evaluations.",
    )
    train.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Validation batches per evaluation; zero evaluates the full split.",
    )
    train.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Optimizer steps between resumable checkpoints.",
    )
    train.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Optimizer steps between training metrics.",
    )
    train.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint directory from this pipeline. Supply the original training settings.",
    )
    train.add_argument(
        "--sample-prompts",
        type=Path,
        default=ROOT / "sample_prompts.json",
        help="JSON list of continuation prompts.",
    )
    train.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per sample (bounded by context).",
    )
    runtime_arguments(train)

    test = commands.add_parser(
        "test",
        help="Evaluate a saved model on held-out books and print sample responses.",
        formatter_class=formatter,
    )
    test.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/prepared"),
        help="The corpus used to train this model.",
    )
    test.add_argument(
        "--model-dir",
        type=Path,
        default=Path("runs/tidbit/best"),
        help="Local saved checkpoint or final model.",
    )
    test.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Held-out split to evaluate.",
    )
    test.add_argument(
        "--batch-size", type=int, default=8, help="Evaluation batch size."
    )
    test.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit evaluation batches; zero evaluates the entire split.",
    )
    test.add_argument(
        "--report",
        type=Path,
        help="JSON report path; defaults to the model's run directory.",
    )
    test.add_argument(
        "--sample-prompts",
        type=Path,
        default=ROOT / "sample_prompts.json",
        help="JSON list of continuation prompts.",
    )
    test.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per sample.",
    )
    runtime_arguments(test)

    generate = commands.add_parser(
        "generate",
        help="Generate continuations from a local model.",
        formatter_class=formatter,
    )
    generate.add_argument(
        "--model-dir",
        type=Path,
        default=Path("runs/tidbit/best"),
        help="Local saved checkpoint or final model.",
    )
    generate.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Prompt text; repeat this option for multiple prompts.",
    )
    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per prompt.",
    )
    generate.add_argument(
        "--temperature", type=float, default=0.8, help="Positive sampling temperature."
    )
    runtime_arguments(generate)
    return cli


def runtime_arguments(command):
    command.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device; auto prefers CUDA, then Apple MPS, then CPU.",
    )
    command.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads.")
    command.add_argument(
        "--seed", type=int, default=42, help="Training or generation seed."
    )


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "prepare":
            from tidbit.data import prepare

            prepare(args)
        else:
            from tidbit.training import generate, test_model, train

            {"train": train, "test": test_model, "generate": generate}[args.command](
                args
            )
    except (ValueError, OSError) as exc:
        cli.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
