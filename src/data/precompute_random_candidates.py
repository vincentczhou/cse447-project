#!/usr/bin/env python
"""
Precompute random candidates for every position in train.txt and valid.txt.

Same TSV format as precompute_kenlm_candidates.py, but candidates are drawn
uniformly at random from the vocab (no KenLM scoring). kenlm_scores are all 0.
Gold is always force-included. Useful as a baseline to isolate the reranker's
contribution vs the KenLM candidate quality.

Output columns:
    seq_idx       int   - line index in the input file
    pos           int   - position within the line
    candidates    str   - \x01-separated random token strings (gold always included)
    kenlm_scores  str   - \x01-separated zeros (same length as candidates)
    gold          str   - the correct next token

Usage:
    uv run python src/data/precompute_random_candidates.py --split train
    uv run python src/data/precompute_random_candidates.py --split valid
    uv run python src/data/precompute_random_candidates.py --input custom.txt --output out.tsv

Arguments:
    --split {train,valid}     Use train/valid path from config.yaml (required unless --input given)
    --input PATH              Custom input file; overrides --split
    --output PATH             Custom output TSV path (default: auto-named next to input)
    --k INT                   Number of random candidates per position (default: 64)
    --max_lines INT           Limit number of input lines, for testing (default: all)
    --stratified_sample INT   Sample N lines total, balanced across language groups
    --num_languages INT       Number of language groups for stratified sampling (default: 52)
    --seed INT                Random seed (default: 42)
    --positions_per_line INT  Randomly sample this many positions per line (default: all)
    --last_position_only      Only emit the last position per line
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

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
# Per-line processing (no KenLM — pure random sampling)
# ---------------------------------------------------------------------------


def _process_line(
    args: tuple[int, list[str]],
    vocab_tokens: list[str],
    k: int,
    positions_per_line: int | None,
    last_position_only: bool,
    rng: random.Random,
    exclude_tokens: set[str] | None = None,
) -> list[str]:
    """Sample k random candidates for each emitted position in one sequence.

    Args:
        exclude_tokens: Tokens to exclude from the random candidate pool.
            Used to prevent sibling gold tokens (from the same input context)
            from appearing as negatives during distillation training.
    """
    seq_idx, tokens = args
    sep = "\x01"
    zero_scores = sep.join(["0.000000"] * k)

    valid_positions = range(1, len(tokens))
    if last_position_only:
        emit_set = {len(tokens) - 1} if len(tokens) > 1 else set()
    elif positions_per_line is not None and positions_per_line < len(valid_positions):
        emit_set = set(rng.sample(list(valid_positions), positions_per_line))
    else:
        emit_set = set(valid_positions)

    # Filter vocab pool if sibling golds should be excluded.
    if exclude_tokens:
        pool = [t for t in vocab_tokens if t not in exclude_tokens]
    else:
        pool = vocab_tokens

    rows: list[str] = []
    for pos in valid_positions:
        if pos not in emit_set:
            continue
        gold_token = tokens[pos]
        cand_tokens = rng.sample(pool, k)
        # Force-include gold
        if gold_token not in cand_tokens:
            cand_tokens[-1] = gold_token
        rows.append(
            f"{seq_idx}\t{pos}\t{sep.join(cand_tokens)}\t{zero_scores}\t{gold_token}"
        )
    return rows


# ---------------------------------------------------------------------------
# Stratified sampling helper
# ---------------------------------------------------------------------------


def _compute_stratified_indices(
    path: Path, n_sample: int, n_langs: int, seed: int
) -> set[int]:
    """Return a set of non-empty-line indices, balanced across language blocks."""
    total = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
    if n_sample >= total:
        return set(range(total))
    chunk_size = total // n_langs
    per_lang = n_sample // n_langs
    remainder = n_sample % n_langs
    rng = random.Random(seed)
    indices: set[int] = set()
    for i in range(n_langs):
        start = i * chunk_size
        end = start + chunk_size if i < n_langs - 1 else total
        n = per_lang + (1 if i < remainder else 0)
        indices.update(rng.sample(range(start, end), min(n, end - start)))
    return indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute random candidates (no KenLM)"
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        default=None,
        help="Use train/valid path from config (required unless --input is given)",
    )
    parser.add_argument(
        "--input", default=None, help="Custom input file (overrides --split)"
    )
    parser.add_argument("--output", default=None, help="Custom output TSV path")
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
    parser.add_argument(
        "--last_position_only",
        action="store_true",
        help="Only emit the last position per line",
    )
    parser.add_argument(
        "--stratified_sample",
        type=int,
        default=None,
        help="Sample N lines total, balanced across language groups (mutually exclusive with --max_lines)",
    )
    parser.add_argument(
        "--num_languages",
        type=int,
        default=52,
        help="Number of language groups for stratified sampling (default: 52)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--exclude_sibling_golds",
        action="store_true",
        help="Exclude sibling gold tokens from candidates (requires --last_position_only). "
        "For distillation: prevents other predictions for the same input from appearing as negatives.",
    )
    args = parser.parse_args()

    if args.input is None and args.split is None:
        parser.error("Either --split or --input is required")
    if args.max_lines is not None and args.stratified_sample is not None:
        parser.error("--max_lines and --stratified_sample are mutually exclusive")
    if args.exclude_sibling_golds and not args.last_position_only:
        parser.error("--exclude_sibling_golds requires --last_position_only")

    data_cfg = _RCFG.get("data", {})

    if args.input is not None:
        input_path = Path(args.input)
    elif args.split == "train":
        input_path = _REPO / data_cfg.get(
            "train_path", "data/madlad_multilang_clean_15k_optionB_kenlm/train.txt"
        )
    else:
        input_path = _REPO / data_cfg.get(
            "valid_path", "data/madlad_multilang_clean_15k_optionB_kenlm/valid.txt"
        )

    if args.output is not None:
        output_path = Path(args.output)
    else:
        split_name = args.split or "custom"
        output_path = (
            input_path.parent / f"candidates_{split_name}_k{args.k}_random.tsv"
        )

    vocab_path = _REPO / data_cfg.get(
        "vocab_path", "data/madlad_multilang_clean_15k_optionB_kenlm/vocab.json"
    )
    exclude = set(_CONFIG["model"]["exclude_tokens"])

    print(f"Vocab       : {vocab_path}")
    print(f"Input       : {input_path}")
    print(f"Output      : {output_path}")
    print(f"K           : {args.k}")
    if args.last_position_only:
        print("Positions/line: last only")
    else:
        print(
            f"Positions/line: {args.positions_per_line if args.positions_per_line else 'all'}"
        )

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_counts: dict[str, int] = json.load(f)

    vocab_tokens = [t for t in vocab_counts if t not in exclude]
    print(f"Vocab size  : {len(vocab_tokens)} tokens")

    # Stratified sampling or simple line counting
    selected_indices: set[int] | None = None
    if args.stratified_sample is not None:
        selected_indices = _compute_stratified_indices(
            input_path,
            args.stratified_sample,
            args.num_languages,
            args.seed,
        )
        num_lines = len(selected_indices)
        per_lang = args.stratified_sample // args.num_languages
        print(
            f"Stratified sample: {num_lines} lines from {args.num_languages} languages (~{per_lang}/lang)"
        )
    else:
        num_lines = 0
        with input_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.max_lines is not None and i >= args.max_lines:
                    break
                if line.strip():
                    num_lines += 1

    print(f"Sequences   : {num_lines}")

    # First pass: build context → set of gold tokens for sibling exclusion.
    # Each entry maps the context (all tokens except the last) to the set of
    # final tokens across all lines sharing that context. During candidate
    # generation, sibling golds are removed from the random pool so the model
    # isn't penalised for predicting a valid alternative.
    sibling_golds: dict[tuple[str, ...], set[str]] | None = None
    if args.exclude_sibling_golds:
        sibling_golds = {}
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    key = tuple(tokens[:-1])
                    sibling_golds.setdefault(key, set()).add(tokens[-1])
        multi = sum(1 for v in sibling_golds.values() if len(v) > 1)
        print(f"Sibling gold groups: {multi} contexts with multiple golds")

    BATCH_SIZE = 1024
    header = "seq_idx\tpos\tcandidates\tkenlm_scores\tgold\n"
    total_rows = 0
    rng = random.Random(args.seed)

    def _iter_batched_tasks():
        with input_path.open("r", encoding="utf-8") as f:
            line_iter = itertools.islice(f, args.max_lines) if args.max_lines else f
            global_line_idx = 0
            batch: list[tuple[int, list[str]]] = []
            for raw_line in line_iter:
                raw = raw_line.strip()
                if raw:
                    if selected_indices is None or global_line_idx in selected_indices:
                        batch.append((global_line_idx, raw.split()))
                        if len(batch) >= BATCH_SIZE:
                            yield batch
                            batch = []
                    global_line_idx += 1
            if batch:
                yield batch

    with output_path.open("w", encoding="utf-8") as out_f:
        out_f.write(header)
        pbar = tqdm(total=num_lines, desc="lines", unit="lines")
        for batch in _iter_batched_tasks():
            for task in batch:
                exclude: set[str] | None = None
                if sibling_golds is not None:
                    tokens = task[1]
                    key = tuple(tokens[:-1])
                    # Exclude sibling golds but NOT the current line's own gold.
                    exclude = sibling_golds.get(key, set()) - {tokens[-1]}
                    if not exclude:
                        exclude = None
                rows = _process_line(
                    task,
                    vocab_tokens,
                    args.k,
                    args.positions_per_line,
                    args.last_position_only,
                    rng,
                    exclude_tokens=exclude,
                )
                if rows:
                    out_f.write("\n".join(rows) + "\n")
                total_rows += len(rows)
                pbar.update(1)
                pbar.set_postfix(rows=f"{total_rows:,}")
        pbar.close()

    print(f"Done. {total_rows:,} rows written to {output_path}")


if __name__ == "__main__":
    main()
