#!/usr/bin/env python
"""
Extend each input line with each of the 3 Gemini predictions.

Given input.txt (partial strings) and pred.txt (3-char predictions),
produces three output files where each line is the original input
concatenated with the 1st, 2nd, or 3rd predicted character respectively.

Usage:
    python scripts/extend_inputs.py \
        --input  output/test_input.txt \
        --pred   output/gemini_pred.txt \
        --outdir output/
"""

import argparse
from pathlib import Path


def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser(
        description="Extend inputs with Gemini predictions"
    )
    parser.add_argument("--input", required=True, help="Path to test input.txt")
    parser.add_argument("--pred", required=True, help="Path to gemini predictions file")
    parser.add_argument("--outdir", default="output", help="Directory for output files")
    args = parser.parse_args()

    inputs = load_lines(args.input)
    preds = load_lines(args.pred)

    if len(inputs) != len(preds):
        raise ValueError(
            f"Line count mismatch: {len(inputs)} inputs vs {len(preds)} predictions"
        )

    stem = Path(args.input).stem  # e.g. "test_input"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for rank in range(3):
        lines = []
        for inp, pred in zip(inputs, preds):
            char = pred[rank] if rank < len(pred) else " "
            lines.append(inp + char)

        out_path = outdir / f"{stem}_ext{rank + 1}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} lines → {out_path}")


if __name__ == "__main__":
    main()
