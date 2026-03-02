#!/usr/bin/env python
"""
Augment a train.txt file by appending tokenized extended-input lines.

Each line from each --ext file is normalized and char-tokenized (matching
the format preprocess.py uses) before being written --repeat times.

Usage:
    python scripts/augment_train.py \
        --train  data/.../train.txt \
        --ext    output/test_input_ext1.txt \
                 output/test_input_ext2.txt \
                 output/test_input_ext3.txt \
        --output data/.../train_augmented.txt \
        --repeat 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.text_utils import normalize_text, char_tokenize


def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser(
        description="Augment train.txt with tokenized extended inputs"
    )
    parser.add_argument("--train", required=True, help="Path to original train.txt")
    parser.add_argument(
        "--ext", nargs="+", required=True, help="One or more ext input files to append"
    )
    parser.add_argument(
        "--output", required=True, help="Path for augmented output file"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each ext line (default: 1)",
    )
    args = parser.parse_args()

    train_lines = load_lines(args.train)
    print(f"Train lines  : {len(train_lines)}")

    ext_lines: list[str] = []
    for ext_path in args.ext:
        raw_lines = load_lines(ext_path)
        tokenized = [
            char_tokenize(normalize_text(line))
            for line in raw_lines
            if normalize_text(line)
        ]
        ext_lines.extend(tokenized * args.repeat)
        print(
            f"  + {ext_path}: {len(tokenized)} lines × {args.repeat} = {len(tokenized) * args.repeat}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
        for line in ext_lines:
            f.write(line + "\n")

    total = len(train_lines) + len(ext_lines)
    print(f"\nTotal lines  : {total} → {args.output}")


if __name__ == "__main__":
    main()
