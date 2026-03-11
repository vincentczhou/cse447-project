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
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

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

    train_path = Path(args.train)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Copy train file directly instead of reading into memory
    shutil.copy2(train_path, out)
    print(f"Copied train : {train_path}")

    ext_count = 0
    with out.open("a", encoding="utf-8") as f:
        for ext_path in args.ext:
            raw_lines = load_lines(ext_path)
            tokenized = [
                char_tokenize(normalize_text(line))
                for line in tqdm(raw_lines, desc=f"Tokenizing {ext_path}")
                if normalize_text(line)
            ]
            for _ in range(args.repeat):
                for line in tokenized:
                    f.write(line + "\n")
            ext_count += len(tokenized) * args.repeat
            print(
                f"  + {ext_path}: {len(tokenized)} lines × {args.repeat} = {len(tokenized) * args.repeat}"
            )

    print(f"\nExt lines added: {ext_count} → {args.output}")


if __name__ == "__main__":
    main()
