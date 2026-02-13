#!/usr/bin/env python
# uv run python oov_test.py --model work/char6.binary --vocab work/vocab.json --token "@"
import argparse
import json
import os

import kenlm


def load_vocab(vocab_path: str) -> set[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    return set(vocab_dict.keys())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="work/char6.binary")
    parser.add_argument("--vocab", default="work/vocab.json")
    parser.add_argument("--token", default="@")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"Vocab not found: {args.vocab}")

    model = kenlm.Model(args.model)
    vocab = load_vocab(args.vocab)

    token = args.token
    print(f"Token: {repr(token)}")
    print(f"In vocab: {token in vocab}")

    state = kenlm.State()
    model.BeginSentenceWrite(state)
    out_state = kenlm.State()
    score = model.BaseScore(state, token, out_state)
    print(f"BaseScore: {score}")


if __name__ == "__main__":
    main()
