#!/usr/bin/env python
"""
Generate KenLM top-K predictions for a set of input prefixes.

Outputs one prediction string per line — the top-K most likely next characters
concatenated in confidence order (e.g. "etaio..." for K=5).
Use scripts/grade.py to evaluate accuracy at any cutoff.

Usage:
    uv run python scripts/eval_kenlm_topk.py test \
        --work_dir work \
        --test_data data/madlad_multilang_clean_35k_optionB_kenlm/input_valid.txt \
        --test_output output/preds_char6_top64.txt \
        --k 64
        # --model 35k_char6_000122.binary  # optional – overrides config.yaml
        # --vocab 35k_vocab_truncated.json # optional – overrides config.yaml

    # Then grade at top-3:
    uv run python scripts/grade.py \
        --pred output/preds_char6_top64.txt \
        --answer data/madlad_multilang_clean_35k_optionB_kenlm/answer_valid.txt \
        --top-k 3
"""

import heapq
import json
import os
import sys
import traceback
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import kenlm
import yaml

# Resolve src/ so utils.text_utils can be imported when running from scripts/
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from utils.text_utils import SPACE_TOKEN, input_to_tokens  # noqa: E402

# Load config
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

FALLBACK_PRED = CONFIG["prediction"]["fallback"]
_DEFAULT_K = CONFIG["prediction"]["top_k"]
MODEL_BINARY = CONFIG["model"]["binary"]
VOCAB_FILE = CONFIG["model"]["vocab"]
EXCLUDE_TOKENS = set(CONFIG["model"]["exclude_tokens"])
MAX_WORKERS = CONFIG["workers"]["max_workers"]
SEQUENTIAL_THRESHOLD = CONFIG["workers"]["sequential_threshold"]
CHUNK_DIVISOR = CONFIG["workers"]["chunk_divisor"]

# Global variables for worker processes (each worker loads its own model)
_worker_model = None
_worker_vocab = None
_worker_k = None


def _worker_init(model_path: str, vocab: list[str], k: int):
    """Initialize KenLM model in each worker process."""
    global _worker_model, _worker_vocab, _worker_k
    _worker_model = kenlm.Model(model_path)
    _worker_vocab = vocab
    _worker_k = k


def _predict_single(inp: str) -> str:
    """Predict top k next characters for a single input. Runs in worker process."""
    try:
        tokens = input_to_tokens(inp)

        # Build context state
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

        # Get top k by log probability
        topk = heapq.nlargest(_worker_k, scored, key=lambda x: x[0])

        # Convert tokens back to characters
        chars = []
        for _, token in topk:
            if token == SPACE_TOKEN:
                chars.append(" ")
            else:
                chars.append(token)

        return "".join(chars)
    except Exception as e:
        print(f"Warning: prediction failed for input '{inp[:50]}...': {e}")
        return FALLBACK_PRED


class MyModel:
    """
    KenLM character-level n-gram model for next-character prediction.
    """

    def __init__(self):
        self.model = None
        self.vocab = None
        self.model_path = None

    @classmethod
    def load_training_data(cls):
        """No-op — KenLM training is done offline with lmplz."""
        return []

    @classmethod
    def load_test_data(cls, fname):
        """Load test data from file, one input per line."""
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions file, one prediction per line."""
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        with open(fname, "wt") as f:
            for p in preds:
                f.write("{}\n".format(p))

    def run_train(self, data, work_dir):
        """No-op — KenLM training is done offline with lmplz."""
        pass

    def run_pred(self, data, k: int = _DEFAULT_K):
        """Predict top k next characters for each input."""
        num_workers = min(cpu_count(), MAX_WORKERS)

        if len(data) < SEQUENTIAL_THRESHOLD or num_workers <= 1:
            # Sequential for small datasets
            global _worker_model, _worker_vocab, _worker_k
            _worker_model = self.model
            _worker_vocab = self.vocab
            _worker_k = k
            return [_predict_single(inp) for inp in data]

        # Parallel for large datasets
        try:
            print(f"Using {num_workers} workers for {len(data)} inputs")
            chunk_size = max(1, len(data) // (num_workers * CHUNK_DIVISOR))
            with Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(self.model_path, self.vocab, k),
            ) as pool:
                preds = pool.map(_predict_single, data, chunksize=chunk_size)
            return preds
        except Exception as e:
            print(f"Warning: multiprocessing failed ({e}), falling back to sequential")
            _worker_model = self.model
            _worker_vocab = self.vocab
            _worker_k = k
            return [_predict_single(inp) for inp in data]

    def save(self, work_dir):
        """Save a dummy checkpoint file (KenLM model is the binary itself)."""
        with open(os.path.join(work_dir, "model.checkpoint"), "wt") as f:
            f.write("dummy save")

    @classmethod
    def load(cls, work_dir, model_binary=MODEL_BINARY, vocab_file=VOCAB_FILE):
        """Load KenLM binary model and vocabulary."""
        instance = cls()

        # Load KenLM model
        model_path = os.path.join(work_dir, model_binary)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"KenLM model not found at {model_path}")
        print(f"Loading KenLM model from {model_path}")
        instance.model_path = model_path
        instance.model = kenlm.Model(model_path)

        # Load vocabulary
        vocab_path = os.path.join(work_dir, vocab_file)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
        print(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        # Exclude KenLM special tokens from candidates
        instance.vocab = [t for t in vocab_dict.keys() if t not in EXCLUDE_TOKENS]

        print(f"Model order: {instance.model.order}, Vocab size: {len(instance.vocab)}")
        return instance


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument(
        "--test_data", help="path to test data", default="example/input.txt"
    )
    parser.add_argument(
        "--test_output", help="path to write test predictions", default="pred.txt"
    )
    parser.add_argument(
        "--k", type=int, default=_DEFAULT_K, help="number of top predictions to output"
    )
    parser.add_argument(
        "--model",
        default=MODEL_BINARY,
        help="KenLM binary filename (relative to work_dir)",
    )
    parser.add_argument(
        "--vocab", default=VOCAB_FILE, help="vocab JSON filename (relative to work_dir)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
        print("Instantiating model")
        model = MyModel()
        print("Loading training data")
        train_data = MyModel.load_training_data()
        print("Training")
        model.run_train(train_data, args.work_dir)
        print("Saving model")
        model.save(args.work_dir)
    elif args.mode == "test":
        try:
            print("Loading model")
            model = MyModel.load(
                args.work_dir, model_binary=args.model, vocab_file=args.vocab
            )
            print("Loading test data from {}".format(args.test_data))
            test_data = MyModel.load_test_data(args.test_data)
            print("Making predictions")
            pred = model.run_pred(test_data, k=args.k)
            print("Writing predictions to {}".format(args.test_output))
            assert len(pred) == len(test_data), (
                "Expected {} predictions but got {}".format(len(test_data), len(pred))
            )
            model.write_pred(pred, args.test_output)
        except Exception as e:
            print(f"Error during test: {e}")
            traceback.print_exc()
            # Ensure a valid pred.txt is always written
            test_data = MyModel.load_test_data(args.test_data)
            # Fill any missing predictions with fallback
            pred = pred if "pred" in locals() else []
            while len(pred) < len(test_data):
                pred.append(FALLBACK_PRED)
            MyModel.write_pred(pred, args.test_output)
            print(f"Wrote {len(pred)} fallback predictions to {args.test_output}")
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
