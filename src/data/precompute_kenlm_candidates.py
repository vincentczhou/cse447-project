#!/usr/bin/env python
"""
Precompute KenLM top-K candidates for every position in train.txt and valid.txt.

For each line in the input file, and for each position within that line, we:
  1. Build the KenLM context state by feeding all preceding tokens
  2. Score every vocab token with BaseScore
  3. Take the top-K candidates (force-include the gold token if missing)
  4. Write one row per position to a parquet file

Output columns:
    seq_idx       int   - line index in the input file
    pos           int   - position within the line (1-indexed, i.e. target position)
    candidates    str   - \x01-separated top-K token strings (e.g. "e\x01a\x01 \x01t\x01...")
    kenlm_scores  str   - \x01-separated log10 probs, same order as candidates
    gold          str   - the correct next token

Usage:
    uv run python src/data/precompute_kenlm_candidates.py --split train --work_dir work --k 64
    uv run python src/data/precompute_kenlm_candidates.py --split valid --work_dir work --k 64
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import kenlm
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

_CONFIG_PATH = _REPO / "config.yaml"
with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f)

_RCFG = _CONFIG.get("reranker", {})

# ---------------------------------------------------------------------------
# Worker globals (set once per process via initializer)
# ---------------------------------------------------------------------------

_worker_model: kenlm.Model | None = None
_worker_vocab_tokens: list[str] | None = None
_worker_k: int | None = None
_worker_positions_per_line: int | None = None


def _worker_init(
    model_path: str, vocab_tokens: list[str], k: int, positions_per_line: int | None
) -> None:
    global _worker_model, _worker_vocab_tokens, _worker_k, _worker_positions_per_line
    _worker_model = kenlm.Model(model_path)
    _worker_vocab_tokens = vocab_tokens
    _worker_k = k
    _worker_positions_per_line = positions_per_line


# ---------------------------------------------------------------------------
# Per-line worker
# ---------------------------------------------------------------------------


def _process_line(args: tuple[int, list[str]]) -> list[str]:
    """Score every position in one tokenized sequence.

    Args:
        args: (seq_idx, token_list) — char tokens for one line.

    Returns:
        List of TSV row strings, one per position (pos 1..len-1).
    """
    seq_idx, tokens = args
    model = _worker_model
    vocab_tokens = _worker_vocab_tokens
    k = _worker_k
    positions_per_line = _worker_positions_per_line

    # Positions where we emit a row: pos >= 1 (need at least one context token)
    valid_positions = range(1, len(tokens))
    if positions_per_line is not None and positions_per_line < len(valid_positions):
        emit_set = set(random.sample(list(valid_positions), positions_per_line))
    else:
        emit_set = set(valid_positions)

    rows: list[str] = []
    # \x01 (ASCII SOH) separates candidates/scores within a field — safe because
    # it's a control character that never appears as a natural-language token.
    sep = "\x01"

    state = kenlm.State()
    model.BeginSentenceWrite(state)
    out_state = kenlm.State()
    tmp = kenlm.State()

    for pos in range(len(tokens)):
        gold_token = tokens[pos]

        if pos in emit_set:
            # Score all vocab tokens from the current context state
            scored: list[tuple[float, str]] = [
                (model.BaseScore(state, tok, tmp), tok) for tok in vocab_tokens
            ]
            top_k = heapq.nlargest(k, scored, key=lambda x: x[0])
            cand_tokens = [t for _, t in top_k]
            cand_scores = [s for s, _ in top_k]

            # Force-include gold if missing
            if gold_token not in cand_tokens:
                gold_score = model.BaseScore(state, gold_token, tmp)
                cand_tokens[-1] = gold_token
                cand_scores[-1] = gold_score

            rows.append(
                f"{seq_idx}\t{pos}\t"
                f"{sep.join(cand_tokens)}\t"
                f"{sep.join(f'{s:.6f}' for s in cand_scores)}\t"
                f"{gold_token}"
            )

        # Advance KenLM state by feeding current token
        model.BaseScore(state, gold_token, out_state)
        state, out_state = out_state, state

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute KenLM top-K candidates")
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--work_dir", default="work")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Limit number of input lines (for testing)",
    )
    parser.add_argument(
        "--positions_per_line",
        type=int,
        default=None,
        help="Randomly sample this many positions per line (default: all positions)",
    )
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    data_cfg = _RCFG.get("data", {})

    if args.split == "train":
        input_path = _REPO / data_cfg.get(
            "train_path", "data/madlad_multilang_clean_15k_optionB_kenlm/train.txt"
        )
    else:
        input_path = _REPO / data_cfg.get(
            "valid_path", "data/madlad_multilang_clean_15k_optionB_kenlm/valid.txt"
        )
    model_stem = Path(_CONFIG["model"]["binary"]).stem
    output_path = (
        input_path.parent / f"candidates_{args.split}_k{args.k}_{model_stem}.tsv"
    )

    vocab_path = _REPO / data_cfg.get(
        "vocab_path", "data/madlad_multilang_clean_15k_optionB_kenlm/vocab.json"
    )
    model_path = str(work_dir / _CONFIG["model"]["binary"])
    exclude = set(_CONFIG["model"]["exclude_tokens"])

    print(f"KenLM model : {model_path}")
    print(f"Vocab       : {vocab_path}")
    print(f"Input       : {input_path}")
    print(f"Output      : {output_path}")
    print(f"K           : {args.k}")
    print(
        f"Positions/line: {args.positions_per_line if args.positions_per_line else 'all'}"
    )
    print(f"Workers     : {args.num_workers}")

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_counts: dict[str, int] = json.load(f)

    vocab_tokens = [t for t in vocab_counts if t not in exclude]
    print(f"Scoreable vocab: {len(vocab_tokens)} tokens")

    with input_path.open("r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if args.max_lines is not None and i >= args.max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line.split())

    print(f"Sequences   : {len(lines)}")
    print(f"Total positions: {sum(len(s) - 1 for s in lines):,}")

    work_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(enumerate(lines))
    initargs = (model_path, vocab_tokens, args.k, args.positions_per_line)
    chunksize = max(8, len(tasks) // (args.num_workers * 4))

    header = "seq_idx\tpos\tcandidates\tkenlm_scores\tgold\n"
    total_rows = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        out_f.write(header)

        if args.num_workers <= 1:
            _worker_init(*initargs)
            for task in tqdm(tasks, desc="lines", unit="lines"):
                rows = _process_line(task)
                if rows:
                    out_f.write("\n".join(rows) + "\n")
                total_rows += len(rows)
        else:
            with Pool(
                processes=args.num_workers,
                initializer=_worker_init,
                initargs=initargs,
            ) as pool:
                pbar = tqdm(total=len(tasks), desc="lines", unit="lines")
                for rows in pool.imap_unordered(
                    _process_line, tasks, chunksize=chunksize
                ):
                    if rows:
                        out_f.write("\n".join(rows) + "\n")
                    total_rows += len(rows)
                    pbar.update(1)
                    pbar.set_postfix(rows=f"{total_rows:,}")
                pbar.close()

    print(f"Done. {total_rows:,} rows written to {output_path}")


if __name__ == "__main__":
    main()
