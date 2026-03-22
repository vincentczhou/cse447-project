#!/usr/bin/env python
"""
Create input_valid.txt and answer_valid.txt for the distill dataset.

For each (test_input, gemini_pred) pair, emits 3 validation examples:
  - input_valid:  the test input (repeated 3 times per original input)
  - answer_valid: the 1st, 2nd, and 3rd predicted character respectively

Usage:
    uv run python scripts/prepare_distill_valid.py \
        --input  temp/test_input.txt \
        --preds  temp/gemini_pred_test.txt \
        --outdir data/distill
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Test input file")
    parser.add_argument(
        "--preds", required=True, help="Gemini predictions file (3 chars/line)"
    )
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    inputs = Path(args.input).read_text(encoding="utf-8").splitlines()
    preds = Path(args.preds).read_text(encoding="utf-8").splitlines()

    if len(inputs) != len(preds):
        raise ValueError(
            f"Line count mismatch: {len(inputs)} inputs vs {len(preds)} preds"
        )

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    input_lines = []
    answer_lines = []
    skipped = 0
    for inp, pred in zip(inputs, preds):
        if len(pred) < 3:
            skipped += 1
            continue
        for ch in pred[:3]:
            input_lines.append(inp)
            answer_lines.append(ch)

    (out / "input_valid.txt").write_text(
        "\n".join(input_lines) + "\n", encoding="utf-8"
    )
    (out / "answer_valid.txt").write_text(
        "\n".join(answer_lines) + "\n", encoding="utf-8"
    )

    print(
        f"Written {len(input_lines)} examples to {out}/input_valid.txt + answer_valid.txt"
    )
    if skipped:
        print(f"Skipped {skipped} lines with fewer than 3 predicted characters")


if __name__ == "__main__":
    main()
