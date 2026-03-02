#!/usr/bin/env python
"""
Convert a W&B predictions table JSON to flat input.txt / pred.txt files.

Each row's "input" field is written as a single line (internal newlines
replaced with a space). The "prediction" field is written the same way.

Usage:
    python scripts/parse_predictions_table.py \
        --table  predictions.table.json \
        --input  output/test_input.txt \
        --pred   output/test_pred.txt   # optional
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="W&B table JSON → input/pred txt")
    parser.add_argument("--table", required=True, help="Path to *.table.json")
    parser.add_argument("--input", required=True, help="Output path for input.txt")
    parser.add_argument(
        "--pred", default=None, help="Output path for pred.txt (optional)"
    )
    args = parser.parse_args()

    with open(args.table, encoding="utf-8") as f:
        table = json.load(f)

    columns = table["columns"]
    col = {name: i for i, name in enumerate(columns)}
    rows = table["data"]

    def clean(value) -> str:
        return (
            str(value).replace("\n", " ").replace("\r", "") if value is not None else ""
        )

    inputs = [clean(row[col["input"]]) for row in rows]

    Path(args.input).parent.mkdir(parents=True, exist_ok=True)
    Path(args.input).write_text("\n".join(inputs) + "\n", encoding="utf-8")
    print(f"Wrote {len(inputs)} inputs → {args.input}")

    if args.pred and "prediction" in col:
        preds = [clean(row[col["prediction"]]) for row in rows]
        Path(args.pred).parent.mkdir(parents=True, exist_ok=True)
        Path(args.pred).write_text("\n".join(preds) + "\n", encoding="utf-8")
        print(f"Wrote {len(preds)} predictions → {args.pred}")


if __name__ == "__main__":
    main()
