#!/usr/bin/env python
"""
Evaluate predictions against gold answers with a configurable top-k cutoff.

Each prediction line is a string of candidate characters in confidence order.
A prediction is correct if the gold character appears within the first --top-k
characters (caseless). Defaults to top-3 to match the original task spec.

Usage:
    python scripts/grade.py \
        --pred   output/gemini_pred.txt \
        --answer example/answer.txt \
        --top-k  3 \
        --verbose
"""

import argparse


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


def main():
    parser = argparse.ArgumentParser(
        description="Grade predictions against gold answers"
    )
    parser.add_argument("--pred", required=True, help="Path to predictions file")
    parser.add_argument("--answer", required=True, help="Path to gold answer file")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of leading characters to consider (default: 3)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-line results")
    args = parser.parse_args()

    preds = load_lines(args.pred)
    golds = load_lines(args.answer)

    if len(preds) < len(golds):
        print(
            f"Warning: {len(golds) - len(preds)} missing predictions, treating as wrong"
        )
        preds.extend([""] * (len(golds) - len(preds)))

    acc = grade(preds, golds, top_k=args.top_k, verbose=args.verbose)
    n_correct = int(acc * len(golds))
    print(f"\n  Predictions : {args.pred}")
    print(f"  Answers     : {args.answer}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Accuracy    : {acc:.2%}  ({n_correct}/{len(golds)})")


if __name__ == "__main__":
    main()
