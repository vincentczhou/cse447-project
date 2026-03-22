#!/usr/bin/env python
"""
Precompute KenLM top-K candidates for every position in train.txt and valid.txt.

For each line in the input file, and for each position within that line, we:
  1. Build the KenLM context state by feeding all preceding tokens
  2. Score every vocab token with BaseScore
  3. Take the top-K candidates (optionally force-include the gold token if missing)
  4. Write one row per position to a parquet file

Output columns:
    seq_idx       int   - line index in the input file
    pos           int   - position within the line (1-indexed, i.e. target position)
    candidates    str   - \x01-separated top-K token strings (e.g. "e\x01a\x01 \x01t\x01...")
    kenlm_scores  str   - \x01-separated log10 probs, same order as candidates
    gold          str   - the correct next token

Usage:
    uv run python src/data/precompute_kenlm_candidates.py --split train
    uv run python src/data/precompute_kenlm_candidates.py --split valid
    uv run python src/data/precompute_kenlm_candidates.py --input custom.txt --output out.tsv

Arguments:
    --split {train,valid}     Use train/valid path from config.yaml (required unless --input given)
    --input PATH              Custom input file; overrides --split
    --output PATH             Custom output TSV path (default: auto-named next to input)
    --work_dir DIR            Directory containing the KenLM binary and vocab (default: work)
    --k INT                   Number of top-K candidates per position (default: 64)
    --max_lines INT           Limit number of input lines, for testing (default: all)
    --stratified_sample INT   Sample N lines total, balanced across language groups
    --num_languages INT       Number of language groups for stratified sampling (default: 52)
    --seed INT                Random seed for stratified sampling (default: 42)
    --positions_per_line INT  Randomly sample this many positions per line (default: all)
    --last_position_only      Only emit the last position per line, for distillation
    --num_workers INT         Worker processes for scoring (default: cpu_count)
"""

from __future__ import annotations

import argparse
import heapq
import itertools
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
_worker_last_position_only: bool = False
_worker_force_include_gold: bool = True


def _worker_init(
    model_path: str,
    vocab_tokens: list[str],
    k: int,
    positions_per_line: int | None,
    last_position_only: bool = False,
    force_include_gold: bool = True,
) -> None:
    """Initialize KenLM model and config globals in each worker process."""
    global \
        _worker_model, \
        _worker_vocab_tokens, \
        _worker_k, \
        _worker_positions_per_line, \
        _worker_last_position_only, \
        _worker_force_include_gold
    _worker_model = kenlm.Model(model_path)
    _worker_vocab_tokens = vocab_tokens
    _worker_k = k
    _worker_positions_per_line = positions_per_line
    _worker_last_position_only = last_position_only
    _worker_force_include_gold = force_include_gold


# ---------------------------------------------------------------------------
# Per-line worker
# ---------------------------------------------------------------------------


def _process_line(args: tuple[int, list[str], set[str] | None]) -> list[str]:
    """Score every position in one tokenized sequence.

    Args:
        args: (seq_idx, token_list, exclude_tokens) — char tokens for one line,
            plus an optional set of tokens to exclude from the scored vocab pool
            (used to prevent sibling gold tokens from appearing as negatives).

    Returns:
        List of TSV row strings, one per position (pos 1..len-1).
    """
    seq_idx, tokens, exclude_tokens = args
    model = _worker_model
    vocab_tokens = _worker_vocab_tokens
    if exclude_tokens:
        vocab_tokens = [t for t in vocab_tokens if t not in exclude_tokens]
    k = _worker_k
    positions_per_line = _worker_positions_per_line
    last_position_only = _worker_last_position_only
    force_include_gold = _worker_force_include_gold

    # Positions where we emit a row: pos >= 1 (need at least one context token)
    valid_positions = range(1, len(tokens))
    if last_position_only:
        emit_set = {len(tokens) - 1} if len(tokens) > 1 else set()
    elif positions_per_line is not None and positions_per_line < len(valid_positions):
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

            # Optionally force-include the gold so downstream reranking always
            # has a valid label. Disabling this makes the TSV match live KenLM
            # candidate recall more closely.
            if force_include_gold and gold_token not in cand_tokens:
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
    parser = argparse.ArgumentParser(description="Precompute KenLM top-K candidates")
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
    parser.add_argument(
        "--last_position_only",
        action="store_true",
        help="Only emit the last position per line (for distillation)",
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
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for stratified sampling"
    )
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument(
        "--exclude_sibling_golds",
        action="store_true",
        help="Exclude sibling gold tokens from KenLM scoring pool (requires --last_position_only). "
        "For distillation: prevents other predictions for the same input from appearing as negatives.",
    )
    parser.add_argument(
        "--no_force_include_gold",
        action="store_true",
        help="Do not inject the gold token when KenLM misses it. Useful for validation "
        "sets that should reflect raw KenLM top-K recall.",
    )
    args = parser.parse_args()

    if args.input is None and args.split is None:
        parser.error("Either --split or --input is required")
    if args.max_lines is not None and args.stratified_sample is not None:
        parser.error("--max_lines and --stratified_sample are mutually exclusive")
    if args.exclude_sibling_golds and not args.last_position_only:
        parser.error("--exclude_sibling_golds requires --last_position_only")

    work_dir = Path(args.work_dir)
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
        model_stem = Path(_CONFIG["model"]["binary"]).stem
        split_name = args.split or "custom"
        output_path = (
            input_path.parent / f"candidates_{split_name}_k{args.k}_{model_stem}.tsv"
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
    print("Force gold  : no" if args.no_force_include_gold else "Force gold  : yes")
    if args.last_position_only:
        print("Positions/line: last only")
    else:
        print(
            f"Positions/line: {args.positions_per_line if args.positions_per_line else 'all'}"
        )
    print(f"Workers     : {args.num_workers}")

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_counts: dict[str, int] = json.load(f)

    vocab_tokens = [t for t in vocab_counts if t not in exclude]
    print(f"Scoreable vocab: {len(vocab_tokens)} tokens")

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

    work_dir.mkdir(parents=True, exist_ok=True)

    initargs = (
        model_path,
        vocab_tokens,
        args.k,
        args.positions_per_line,
        args.last_position_only,
        not args.no_force_include_gold,
    )

    BATCH_SIZE = 1024  # lines read into memory at a time

    header = "seq_idx\tpos\tcandidates\tkenlm_scores\tgold\n"
    total_rows = 0

    def _iter_batched_tasks():
        """Yield (seq_idx, tokens) for each non-empty line, in batches of BATCH_SIZE.

        seq_idx is the original line index in the input file (counting only
        non-empty lines).  This must match the index used by load_sequences()
        in src/reranker/dataset.py so that the TSV's seq_idx maps to the correct
        sequence during training.
        """
        with input_path.open("r", encoding="utf-8") as f:
            line_iter = itertools.islice(f, args.max_lines) if args.max_lines else f
            global_line_idx = 0
            batch: list[tuple[int, list[str], set[str] | None]] = []
            for raw_line in line_iter:
                raw = raw_line.strip()
                if raw:
                    if selected_indices is None or global_line_idx in selected_indices:
                        tokens = raw.split()
                        exclude: set[str] | None = None
                        if sibling_golds is not None and len(tokens) >= 2:
                            key = tuple(tokens[:-1])
                            siblings = sibling_golds.get(key, set()) - {tokens[-1]}
                            exclude = siblings if siblings else None
                        batch.append((global_line_idx, tokens, exclude))
                        if len(batch) >= BATCH_SIZE:
                            yield batch
                            batch = []
                    global_line_idx += 1
            if batch:
                yield batch

    with output_path.open("w", encoding="utf-8") as out_f:
        out_f.write(header)
        pbar = tqdm(total=num_lines, desc="lines", unit="lines")

        if args.num_workers <= 1:
            _worker_init(*initargs)
            for batch in _iter_batched_tasks():
                for task in batch:
                    rows = _process_line(task)
                    if rows:
                        out_f.write("\n".join(rows) + "\n")
                    total_rows += len(rows)
                    pbar.update(1)
                    pbar.set_postfix(rows=f"{total_rows:,}")
        else:
            with Pool(
                processes=args.num_workers,
                initializer=_worker_init,
                initargs=initargs,
            ) as pool:
                for batch in _iter_batched_tasks():
                    chunksize = max(8, len(batch) // (args.num_workers * 4))
                    for rows in pool.imap_unordered(
                        _process_line, batch, chunksize=chunksize
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
