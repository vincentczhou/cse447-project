#!/usr/bin/env python
import heapq
import json
import os
import re
import unicodedata
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import Pool, cpu_count

import kenlm

SPACE_TOKEN = "<sp>"
_ws_re = re.compile(r"\s+")

# Global variables for worker processes (each worker loads its own model)
_worker_model = None
_worker_vocab = None


def normalize_text(text: str) -> str:
    """Normalize text (same as preprocess.py)"""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _ws_re.sub(" ", text).strip()
    return text


def input_to_tokens(text: str) -> list[str]:
    """Convert raw input to character tokens"""
    normalized = normalize_text(text)
    return [SPACE_TOKEN if ch == " " else ch for ch in normalized]


def _worker_init(model_path: str, vocab: list[str]):
    """Initialize KenLM model in each worker process."""
    global _worker_model, _worker_vocab
    _worker_model = kenlm.Model(model_path)
    _worker_vocab = vocab


def _predict_single(inp: str) -> str:
    """Predict top 3 next characters for a single input. Runs in worker process."""
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

    # Get top 3 by log probability
    top3 = heapq.nlargest(3, scored, key=lambda x: x[0])

    # Convert tokens back to characters
    chars = []
    for _, token in top3:
        if token == SPACE_TOKEN:
            chars.append(" ")
        else:
            chars.append(token)

    return "".join(chars)


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
        # KenLM training is done offline with lmplz
        return []

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, "wt") as f:
            for p in preds:
                f.write("{}\n".format(p))

    def run_train(self, data, work_dir):
        # KenLM training is done offline with lmplz
        pass

    def run_pred(self, data):
        """Predict top 3 next characters for each input."""
        num_workers = min(cpu_count(), 8)

        if len(data) < 100 or num_workers <= 1:
            # Sequential for small datasets
            global _worker_model, _worker_vocab
            _worker_model = self.model
            _worker_vocab = self.vocab
            return [_predict_single(inp) for inp in data]

        # Parallel for large datasets
        print(f"Using {num_workers} workers for {len(data)} inputs")
        chunk_size = max(1, len(data) // (num_workers * 4))
        with Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(self.model_path, self.vocab),
        ) as pool:
            preds = pool.map(_predict_single, data, chunksize=chunk_size)
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, "model.checkpoint"), "wt") as f:
            f.write("dummy save")

    @classmethod
    def load(cls, work_dir):
        """Load KenLM binary model and vocabulary."""
        instance = cls()

        # Load KenLM model
        model_path = os.path.join(work_dir, "char6.binary")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"KenLM model not found at {model_path}")
        print(f"Loading KenLM model from {model_path}")
        instance.model_path = model_path
        instance.model = kenlm.Model(model_path)

        # Load vocabulary
        vocab_path = os.path.join(work_dir, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
        print(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        # Exclude KenLM special tokens from candidates
        instance.vocab = [t for t in vocab_dict.keys() if t not in ("<s>", "</s>")]

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
        print("Loading model")
        model = MyModel.load(args.work_dir)
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred)
        )
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
