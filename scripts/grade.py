#!/usr/bin/env python
"""
Evaluate predictions against gold answers with a configurable top-k cutoff.

Each prediction line is a string of candidate characters in confidence order.
A prediction is correct if the gold character appears within the first --top-k
characters (caseless). Defaults to top-3 to match the original task spec.

Usage:
    # Single cutoff, one model
    python scripts/grade.py \
        --pred   work/preds_char6_top64.txt \
        --answer data/madlad_multilang_clean_15k_optionB_kenlm/answer_valid.txt \
        --top-k  3

    # Sweep K=1..64, compare models, save TSV
    python scripts/grade.py \
        --pred   work/preds_char5_top64.txt work/preds_char6_top64.txt \
        --names  char5 char6 \
        --answer data/madlad_multilang_clean_15k_optionB_kenlm/answer_valid.txt \
        --sweep  --sweep-output work/sweep.tsv

    # Verbose per-line output at a single cutoff
    python scripts/grade.py \
        --pred   output/gemini_pred.txt \
        --answer example/answer.txt \
        --top-k  3 --verbose
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def grade(
    preds: list[str], golds: list[str], top_k: int, verbose: bool = False
) -> float:
    correct = 0
    for i, (p, g) in enumerate(zip(preds, golds)):
        hit = g.lower() in p[:top_k].lower()
        correct += hit
        if verbose:
            print(f"  {'✓' if hit else '✗'} [{i}] gold='{g}' pred='{p[:top_k]}'")
    return correct / len(golds) if golds else 0.0


def sweep_model(preds: list[str], golds: list[str]) -> pd.DataFrame:
    """Compute accuracy at every K from 1 to max prediction length."""
    max_k = max((len(p) for p in preds), default=1)
    rows = []
    for k in tqdm(range(1, max_k + 1), desc="sweeping k"):
        acc = grade(preds, golds, top_k=k)
        rows.append(
            {
                "k": k,
                "accuracy": acc,
                "correct": int(acc * len(golds)),
                "total": len(golds),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Grade predictions against gold answers"
    )
    parser.add_argument(
        "--pred", required=True, nargs="+", help="Path(s) to predictions file(s)"
    )
    parser.add_argument("--answer", required=True, help="Path to gold answer file")
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Display names for each --pred file (defaults to filename stems)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of leading characters to consider for single-cutoff mode (default: 3)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Print accuracy at every K from 1 to prediction width (oracle curve)",
    )
    parser.add_argument(
        "--sweep-output",
        default=None,
        help="Path to save sweep results as TSV (only used with --sweep)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-line results (single-cutoff only)",
    )
    args = parser.parse_args()

    # Resolve model names
    names = args.names if args.names else [Path(p).stem for p in args.pred]
    if len(names) != len(args.pred):
        print("Error: --names must have the same number of entries as --pred")
        sys.exit(1)

    golds = load_lines(args.answer)

    print(f"\n  Answers  : {args.answer}")
    print(f"  Examples : {len(golds)}")

    if args.sweep:
        all_dfs = []
        for pred_path, name in zip(args.pred, names):
            preds = load_lines(pred_path)
            if len(preds) < len(golds):
                print(
                    f"Warning [{name}]: {len(golds) - len(preds)} missing predictions, treating as wrong"
                )
                preds.extend([""] * (len(golds) - len(preds)))
            df = sweep_model(preds, golds)
            df["model"] = name
            all_dfs.append(df)

        # Print table: K | model1_acc | model2_acc | ...
        combined = pd.concat(all_dfs)
        pivot = combined.pivot(index="k", columns="model", values="accuracy")
        print(f"\n  {'K':>4}  " + "  ".join(f"{n:>10}" for n in names))
        print(f"  {'-' * 4}  " + "  ".join("-" * 10 for _ in names))
        for k, row in pivot.iterrows():
            vals = "  ".join(f"{row[n]:>9.2%}" for n in names)
            print(f"  {int(k):>4}  {vals}")

        if args.sweep_output:
            Path(args.sweep_output).parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(args.sweep_output, sep="\t", index=False)
            print(f"\n  Saved sweep table to {args.sweep_output}")

    else:
        # Single cutoff, one pred file expected (use first if multiple given)
        if len(args.pred) > 1:
            print(
                "Warning: multiple --pred files given without --sweep; grading only the first"
            )
        preds = load_lines(args.pred[0])
        if len(preds) < len(golds):
            print(
                f"Warning: {len(golds) - len(preds)} missing predictions, treating as wrong"
            )
            preds.extend([""] * (len(golds) - len(preds)))
        print(f"  Predictions : {args.pred[0]}")
        acc = grade(preds, golds, top_k=args.top_k, verbose=args.verbose)
        n_correct = int(acc * len(golds))
        print(f"  Top-k       : {args.top_k}")
        print(f"  Accuracy    : {acc:.2%}  ({n_correct}/{len(golds)})")


if __name__ == "__main__":
    main()
