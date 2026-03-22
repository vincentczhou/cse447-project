#!/usr/bin/env python
"""
Tokenize extended-input files into a single sequences file (train.txt format).

Usage:
    uv run python scripts/prepare_distill_sequences.py \
        --ext temp/gemini/test_input_ext1.txt \
             temp/gemini/test_input_ext2.txt \
             temp/gemini/test_input_ext3.txt \
        --output data/.../distill_sequences.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.text_utils import char_tokenize, normalize_text


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize ext files into sequences format"
    )
    parser.add_argument("--ext", nargs="+", required=True, help="Extended input files")
    parser.add_argument("--output", required=True, help="Output sequences file")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as f:
        for ext_path in args.ext:
            count = 0
            with open(ext_path, encoding="utf-8") as ef:
                for line in ef:
                    text = normalize_text(
                        line.rstrip("\n"), preserve_trailing_space=True
                    )
                    if text:
                        f.write(char_tokenize(text) + "\n")
                        count += 1
            print(f"{ext_path}: {count} lines")
            total += count

    print(f"Total: {total} lines → {args.output}")


if __name__ == "__main__":
    main()
