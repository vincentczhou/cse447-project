#!/usr/bin/env python
"""Two-stage next-character prediction: KenLM candidate generation + neural reranking.

Stage 1: KenLM scores all vocab tokens, returns top-K candidates per input.
Stage 2: Neural reranker rescores K candidates using Transformer context encoding.
Output: Top-3 characters per input (same format as myprogram.py).

Falls back to KenLM-only if reranker checkpoint is missing.

Usage:
    uv run python src/predict.py --work_dir work --test_data example/input.txt --test_output pred.txt
        [--device cpu|cuda] [--kenlm-only]
"""

from __future__ import annotations

import heapq
import json
import traceback
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path

import kenlm
import torch
import yaml

from reranker import PAD_ID, UNK_ID, load_for_inference
from utils.text_utils import SPACE_TOKEN, input_to_tokens

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

FALLBACK_PRED = CONFIG["prediction"]["fallback"]
TOP_N = CONFIG["prediction"]["top_k"]
MODEL_BINARY = CONFIG["model"]["binary"]
VOCAB_FILE = CONFIG["model"]["vocab"]
EXCLUDE_TOKENS = set(CONFIG["model"]["exclude_tokens"])
MAX_WORKERS = CONFIG["workers"]["max_workers"]
SEQUENTIAL_THRESHOLD = CONFIG["workers"]["sequential_threshold"]
CHUNK_DIVISOR = CONFIG["workers"]["chunk_divisor"]

_RERANKER_CFG = CONFIG.get("reranker", {})
CANDIDATE_K = _RERANKER_CFG.get("training", {}).get("candidate_size", 64)
RERANKER_CHECKPOINT = _RERANKER_CFG.get("output", {}).get(
    "checkpoint_name", "reranker.pt"
)
RERANKER_BATCH_SIZE = _RERANKER_CFG.get("training", {}).get("eval_batch_size", 64)

# ---------------------------------------------------------------------------
# KenLM worker functions (Stage 1)
# ---------------------------------------------------------------------------

_worker_model = None
_worker_vocab = None
_worker_k = None


def _worker_init(model_path: str, vocab: list[str], k: int) -> None:
    """Initialize KenLM model in each worker process."""
    global _worker_model, _worker_vocab, _worker_k
    _worker_model = kenlm.Model(model_path)
    _worker_vocab = vocab
    _worker_k = k


def _score_topk_single(inp: str) -> tuple[list[str], list[str], list[float]]:
    """Score all vocab tokens for one input, return top-K candidates.

    Returns:
        (context_tokens, candidate_tokens, candidate_scores)
        Returns ([], [], []) on failure.
    """
    try:
        tokens = input_to_tokens(inp)

        # Build KenLM context state
        state = kenlm.State()
        _worker_model.BeginSentenceWrite(state)
        out_state = kenlm.State()
        for token in tokens:
            _worker_model.BaseScore(state, token, out_state)
            state, out_state = out_state, state

        # Score all vocab candidates from the same context state
        out_state = kenlm.State()
        scored = []
        for token in _worker_vocab:
            log_prob = _worker_model.BaseScore(state, token, out_state)
            scored.append((log_prob, token))

        topk = heapq.nlargest(_worker_k, scored, key=lambda x: x[0])
        if not topk:
            return (tokens, [], [])
        candidate_scores, candidate_tokens = zip(*topk)

        return (tokens, candidate_tokens, candidate_scores)
    except Exception as e:
        print(f"Warning: KenLM scoring failed for '{inp[:50]}...': {e}")
        return ([], [], [])


# ---------------------------------------------------------------------------
# Batched reranker inference (Stage 2)
# ---------------------------------------------------------------------------


def _rerank_batched(
    reranker: torch.nn.Module,
    stoi: dict[str, int],
    kenlm_results: list[tuple[list[str], list[str], list[float]]],
    max_context_len: int,
    top_n: int,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    """Rerank all KenLM results in batched mode.

    Args:
        reranker: Reranker model in eval mode on device.
        stoi: token string -> reranker ID mapping.
        kenlm_results: output from KenLM stage.
        max_context_len: context truncation window.
        top_n: number of top predictions to return.
        batch_size: mini-batch size for forward passes.
        device: torch device for inference.

    Returns:
        list of top_n-character prediction strings, one per input.
    """

    N = len(kenlm_results)
    preds = [FALLBACK_PRED] * N

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_slice = kenlm_results[batch_start:batch_end]

        valid_offsets = []
        ctx_id_lists = []
        cand_id_lists = []
        score_lists = []
        cand_token_lists = []

        for j, (ctx_toks, cand_toks, cand_scores) in enumerate(batch_slice):
            if cand_toks:
                preds[batch_start + j] = "".join(
                    [" " if t == SPACE_TOKEN else t for t in cand_toks[:top_n]]
                )
            if not ctx_toks or not cand_toks or not cand_scores:
                continue
            valid_offsets.append(j)

            # Context: map to IDs, truncate to most recent max_context_len tokens
            ctx_ids = [stoi.get(t, UNK_ID) for t in ctx_toks[-max_context_len:]]
            ctx_id_lists.append(torch.tensor(ctx_ids, dtype=torch.long))

            # Candidates: map to IDs
            cand_ids = [stoi.get(t, UNK_ID) for t in cand_toks]
            cand_id_lists.append(torch.tensor(cand_ids, dtype=torch.long))

            score_lists.append(torch.tensor(cand_scores, dtype=torch.float))
            cand_token_lists.append(cand_toks)

        if not valid_offsets:
            continue

        # Right-pad contexts (matching training convention)
        context_ids = torch.nn.utils.rnn.pad_sequence(
            ctx_id_lists, batch_first=True, padding_value=PAD_ID
        ).to(device)

        # Candidates all have the same length K, so stack directly
        candidate_ids = torch.stack(cand_id_lists).to(device)
        kenlm_scores = torch.stack(score_lists).to(device)

        with torch.no_grad():
            logits = reranker.score_candidates(context_ids, candidate_ids, kenlm_scores)

        n = min(top_n, logits.size(1))
        topn_indices = logits.topk(n, dim=1).indices.cpu()

        for i, offset in enumerate(valid_offsets):
            cand_toks = cand_token_lists[i]
            indices = topn_indices[i].tolist()
            chars = [
                " " if cand_toks[idx] == SPACE_TOKEN else cand_toks[idx]
                for idx in indices
            ]
            preds[batch_start + offset] = "".join(chars)

    return preds


# ---------------------------------------------------------------------------
# TwoStagePredictor
# ---------------------------------------------------------------------------


class TwoStagePredictor:
    """Two-stage predictor: KenLM candidate generation + neural reranking."""

    def __init__(self) -> None:
        self.kenlm_model = None
        self.kenlm_model_path: str | None = None
        self.vocab: list[str] | None = None
        self.reranker = None
        self.reranker_stoi: dict[str, int] | None = None
        self.device: torch.device | None = None
        self.candidate_k = CANDIDATE_K
        self.max_context_len: int | None = None

    @classmethod
    def load(cls, work_dir: str, device: str | None = None) -> TwoStagePredictor:
        """Load KenLM + reranker. Falls back to KenLM-only if no checkpoint."""
        instance = cls()

        if device is None:
            instance.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            instance.device = torch.device(device)

        # KenLM
        work_path = Path(work_dir)
        model_path = work_path / MODEL_BINARY
        if not model_path.exists():
            raise FileNotFoundError(f"KenLM model not found at {model_path}")
        instance.kenlm_model_path = str(model_path)
        instance.kenlm_model = kenlm.Model(str(model_path))

        vocab_path = work_path / VOCAB_FILE
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        instance.vocab = [t for t in vocab_dict.keys() if t not in EXCLUDE_TOKENS]

        print(f"KenLM: order={instance.kenlm_model.order}, vocab={len(instance.vocab)}")

        # Reranker (optional)
        reranker_path = work_path / RERANKER_CHECKPOINT
        if reranker_path.exists():
            model, _tokens, stoi = load_for_inference(reranker_path, instance.device)
            instance.reranker = model
            instance.reranker_stoi = stoi
            instance.max_context_len = model.cfg.max_context_len
            print(f"Reranker loaded from {reranker_path} on {instance.device}")
        else:
            print(f"No reranker at {reranker_path}, using KenLM-only mode")

        return instance

    def run_pred(self, data: list[str], top_n: int = TOP_N) -> list[str]:
        """Two-stage prediction: KenLM top-K -> reranker top-n."""
        kenlm_results = self._kenlm_stage(data)
        print(f"KenLM stage done: {len(kenlm_results)} results")

        if self.reranker is None:
            preds = [self._kenlm_fallback(r, top_n) for r in kenlm_results]
            print(f"KenLM-only predictions done: {len(preds)} outputs")
            return preds

        preds = _rerank_batched(
            reranker=self.reranker,
            stoi=self.reranker_stoi,
            kenlm_results=kenlm_results,
            max_context_len=self.max_context_len,
            top_n=top_n,
            batch_size=RERANKER_BATCH_SIZE,
            device=self.device,
        )
        print(f"Reranker stage done: {len(preds)} outputs")
        return preds

    def _kenlm_stage(
        self, data: list[str]
    ) -> list[tuple[list[str], list[str], list[float]]]:
        """Run KenLM scoring, return top-K candidates per input."""
        num_workers = min(cpu_count() or 1, MAX_WORKERS)
        # Note: candidate_k is candidate_size from config.yaml
        # If reranker is present, use candidate_k from config; otherwise use top_n for KenLM-only mode
        # If the reranker is trained off of unigram negatives, candidate_size is correct.
        # However, if the reranker is trained off of KenLM top-K negatives, candidate_size should be set to the same K used for prediction (e.g. 64).
        k = self.candidate_k if self.reranker is not None else TOP_N

        if len(data) < SEQUENTIAL_THRESHOLD or num_workers <= 1:
            global _worker_model, _worker_vocab, _worker_k
            _worker_model = self.kenlm_model
            _worker_vocab = self.vocab
            _worker_k = k
            return [_score_topk_single(inp) for inp in data]

        try:
            print(f"KenLM stage: {num_workers} workers, {len(data)} inputs, K={k}")
            chunk_size = max(1, len(data) // (num_workers * CHUNK_DIVISOR))
            with Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(self.kenlm_model_path, self.vocab, k),
            ) as pool:
                return pool.map(_score_topk_single, data, chunksize=chunk_size)
        except Exception as e:
            print(f"Warning: multiprocessing failed ({e}), falling back to sequential")
            _worker_model = self.kenlm_model
            _worker_vocab = self.vocab
            _worker_k = k
            return [_score_topk_single(inp) for inp in data]

    @staticmethod
    def _kenlm_fallback(
        result: tuple[list[str], list[str], list[float]], top_n: int
    ) -> str:
        """Convert KenLM result to prediction string (no reranker)."""
        _ctx_toks, cand_toks, _cand_scores = result
        if not cand_toks:
            return FALLBACK_PRED
        chars = [" " if t == SPACE_TOKEN else t for t in cand_toks[:top_n]]
        return "".join(chars)

    @classmethod
    def load_test_data(cls, fname: str) -> list[str]:
        """Load test data (one context per line)."""
        with open(fname) as f:
            return [line[:-1] for line in f]  # strip trailing newline

    @classmethod
    def write_pred(cls, preds: list[str], fname: str) -> None:
        """Write predictions file (one prediction per line)."""
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        with open(fname, "wt") as f:
            for p in preds:
                f.write(f"{p}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Two-stage next-character prediction (KenLM + reranker)",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--work_dir", default="work", help="directory with model files")
    parser.add_argument("--test_data", default="example/input.txt", help="input file")
    parser.add_argument("--test_output", default="pred.txt", help="output predictions")
    parser.add_argument("--device", default=None, help="torch device (auto if omitted)")
    parser.add_argument("--kenlm-only", action="store_true", help="skip reranker")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override reranker alpha (KenLM blend weight). Overrides the value saved in the checkpoint.",
    )
    args = parser.parse_args()

    preds = []
    test_data = []
    try:
        print("Loading models")
        predictor = TwoStagePredictor.load(args.work_dir, device=args.device)

        if args.kenlm_only:
            predictor.reranker = None
            print("Forced KenLM-only mode")

        if args.alpha is not None and predictor.reranker is not None:
            predictor.reranker.alpha = args.alpha
            print(f"Alpha overridden to {args.alpha}")

        print(f"Loading test data from {args.test_data}")
        test_data = TwoStagePredictor.load_test_data(args.test_data)

        print(f"Predicting {len(test_data)} inputs")
        preds = predictor.run_pred(test_data)

        assert len(preds) == len(test_data), (
            f"Expected {len(test_data)} predictions but got {len(preds)}"
        )

        print(f"Writing predictions to {args.test_output}")
        TwoStagePredictor.write_pred(preds, args.test_output)
        print("Done")
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()
        # Ensure a valid pred.txt is always written
        if not test_data:
            test_data = TwoStagePredictor.load_test_data(args.test_data)
        while len(preds) < len(test_data):
            preds.append(FALLBACK_PRED)
        TwoStagePredictor.write_pred(preds, args.test_output)
        print(f"Wrote {len(preds)} fallback predictions to {args.test_output}")
